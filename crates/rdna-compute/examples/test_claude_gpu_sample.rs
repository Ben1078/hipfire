// claude r2 (CLAUDE_MTP_GPU_SAMPLE): on-device sampling vs the host reference.
//
//  1. Draft sampler: for the same softmax rows (softmax_temp_topp_batched,
//     downloaded for the host) and the same uniform u, the GPU index must be
//     the host `sample_categorical(apply_topp_trunc(p), u)` index, the
//     returned q the host truncated prob, the returned max the host max p.
//     Full vocab (248320) and a compressed 32768 map, several temp/top_p/top_k.
//  2. Accept: random target/draft rows (draft = target + noise), draft picks
//     from the GPU sampler, uniforms U[0..=d]; the device accept must commit
//     exactly what the host `mtp_sampled_accept` logic commits (accept count,
//     tokens incl. the residual / target bonus, EOS stop, uniforms consumed).
//  3. Timing: old per-step host path (3 blocking downloads + host truncation
//     + host sample) vs the device sampler + doorbell, and the accept.
//
// Usage: test_claude_gpu_sample [draws=3000] [trials=1500]
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::claude_doorbell::ClaudeDoorbell;
use rdna_compute::claude_sample::{claude_decode_accept, ClaudeSampleScratch};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

// ---- host reference (verbatim semantics of speculative.rs / mtp_spec.rs) ----
fn apply_topp_trunc(row: &mut [f32], tau_cut: f32, z: f32) {
    if tau_cut <= 0.0 {
        return;
    }
    let inv_z = 1.0f32 / z.max(f32::MIN_POSITIVE);
    for p in row.iter_mut() {
        if *p >= tau_cut {
            *p *= inv_z;
        } else {
            *p = 0.0;
        }
    }
}
fn sample_categorical(probs: &[f32], u: f32) -> u32 {
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
fn argmax_u32(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as u32
}
fn sample_residual(pt: &[f32], pd: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for i in 0..pt.len() {
        let d = pt[i] - pd[i];
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        return argmax_u32(pt);
    }
    let us = u * sum;
    let mut acc = 0.0f32;
    for i in 0..pt.len() {
        let d = pt[i] - pd[i];
        if d > 0.0 {
            acc += d;
            if us < acc {
                return i as u32;
            }
        }
    }
    (pt.len() - 1) as u32
}
/// mtp_sampled_accept's decision logic on host rows (target rows already
/// truncated, draft rows truncated and scattered to full vocab).
fn host_accept(
    trows: &[Vec<f32>],
    drows_full: &[Vec<f32>],
    draft_probs: &[f32],
    cands: &[u32],
    eos: u32,
    u: &[f32],
) -> (Vec<u32>, usize, bool, usize) {
    let mut committed = Vec::new();
    let (mut acc, mut used, mut hit, mut rejected) = (0usize, 0usize, false, false);
    for k in 0..cands.len() {
        let pd = draft_probs[k].max(f32::MIN_POSITIVE);
        let pt = trows[k][cands[k] as usize];
        let uu = u[used];
        used += 1;
        if uu * pd <= pt {
            committed.push(cands[k]);
            acc += 1;
            if cands[k] == eos {
                hit = true;
                break;
            }
        } else {
            let u2 = u[used];
            used += 1;
            let b = sample_residual(&trows[k], &drows_full[k], u2);
            committed.push(b);
            if b == eos {
                hit = true;
            }
            rejected = true;
            break;
        }
    }
    if !rejected && !hit {
        let uu = u[used];
        used += 1;
        let b = sample_categorical(&trows[acc], uu);
        committed.push(b);
        if b == eos {
            hit = true;
        }
    }
    (committed, acc, hit, used)
}

/// LM-like logits: bulk ~N(0, 2.5), a handful of candidates 12-20 above.
fn lm_logits(rng: &mut Lcg, n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|_| ((0..4).map(|_| rng.unit()).sum::<f32>() - 2.0) * 4.3)
        .collect();
    for j in 0..8 {
        let i = (rng.next() as usize) % n;
        v[i] = 20.0 - 1.5 * j as f32 + rng.unit();
    }
    v
}

/// Exact inverse CDF in f64 over the truncated row (u * total).
fn sample_exact(w: &[f32], u: f32) -> u32 {
    let total: f64 = w.iter().map(|&x| x as f64).sum();
    let target = u as f64 * total;
    let mut acc = 0.0f64;
    let mut last = 0usize;
    for (i, &x) in w.iter().enumerate() {
        if x > 0.0 {
            last = i;
        }
        acc += x as f64;
        if x > 0.0 && target < acc {
            return i as u32;
        }
    }
    last as u32
}

fn logits(rng: &mut Lcg, n: usize, spikes: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| (rng.unit() + rng.unit() + rng.unit() - 1.5) * 3.0).collect();
    for _ in 0..spikes {
        let i = (rng.next() as usize) % n;
        v[i] += 8.0 + rng.unit() * 6.0;
    }
    v
}

fn dl(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).unwrap();
    b.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let draws: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let trials: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let mut gpu = Gpu::init().expect("gpu");
    gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
    println!("test_claude_gpu_sample: arch={} draws={draws} trials={trials}", gpu.arch);
    let mut db = ClaudeDoorbell::new(&gpu).unwrap();
    let sc = ClaudeSampleScratch::new(&mut gpu).unwrap();
    let mut rng = Lcg(0x5a3b_1e77);
    let v_full = 248_320usize;
    let v_c = 32_768usize;
    // compressed map: ascending random subset of the full vocab
    let mut map: Vec<u32> = Vec::with_capacity(v_c);
    {
        let mut chosen = vec![false; v_full];
        let mut cnt = 0;
        while cnt < v_c {
            let i = (rng.next() as usize) % v_full;
            if !chosen[i] {
                chosen[i] = true;
                cnt += 1;
            }
        }
        for i in 0..v_full {
            if chosen[i] {
                map.push(i as u32);
            }
        }
    }
    let map_b: Vec<u8> = map.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let map_g = gpu.upload_raw(&map_b, &[map_b.len()]).unwrap();
    let mut inv = vec![-1i32; v_full];
    for (j, &i) in map.iter().enumerate() {
        inv[i as usize] = j as i32;
    }
    let inv_g = up_i32(&mut gpu, &inv);

    let max_d = 5usize;
    let logits_g = gpu.zeros(&[(max_d + 1) * v_full], DType::F32).unwrap();
    let tprobs = gpu.zeros(&[(max_d + 1) * v_full], DType::F32).unwrap();
    let ttau = gpu.zeros(&[max_d + 1], DType::F32).unwrap();
    let tz = gpu.zeros(&[max_d + 1], DType::F32).unwrap();
    let dprobs = gpu.zeros(&[max_d * v_full], DType::F32).unwrap();
    let dtau = gpu.zeros(&[max_d], DType::F32).unwrap();
    let dz = gpu.zeros(&[max_d], DType::F32).unwrap();
    let dout = gpu.zeros(&[max_d * 4], DType::F32).unwrap();
    let aout = gpu.zeros(&[16], DType::F32).unwrap();

    // ---------------- 1. draft sampler ----------------
    let configs: [(usize, f32, f32, usize, f32); 5] = [
        (v_full, 0.7, 0.95, 20, 0.0),
        (v_full, 1.0, 0.95, 0, 0.0),
        (v_full, 0.3, 0.8, 40, 0.05),
        (v_c, 0.7, 0.95, 20, 0.0),
        (v_c, 1.0, 1.0, 0, 0.0),
    ];
    let mut total_bad = 0usize;
    for (ci, &(n, temp, top_p, top_k, min_p)) in configs.iter().enumerate() {
        let compressed = n == v_c;
        let lg = if ci % 2 == 0 { lm_logits(&mut rng, n) } else { logits(&mut rng, n, 12) };
        let row_l = up(&mut gpu, &lg);
        let prow = dprobs.sub_offset(0, n);
        gpu.softmax_temp_topp_batched_into_f32(&row_l, &prow, &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), n, 1, temp, top_p, top_k, min_p).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let p = dl(&gpu, &prow, n);
        let tau = dl(&gpu, &dtau, 1)[0];
        let z = dl(&gpu, &dz, 1)[0];
        let mut q = p.clone();
        apply_topp_trunc(&mut q, tau, z);
        let pmax = p.iter().cloned().fold(0.0f32, f32::max);
        let kept = q.iter().filter(|&&x| x > 0.0).count();
        let (mut idx_bad, mut q_bad, mut top_bad, mut tok_bad) = (0, 0, 0, 0);
        let (mut gpu_vs_exact, mut host_vs_exact) = (0usize, 0usize);
        let mut w = [0u32; 4];
        for _ in 0..draws {
            let u = rng.unit();
            let mp = if compressed { map_g.buf.as_ptr() as *const _ } else { std::ptr::null() };
            let tk = gpu
                .claude_mtp_sample_row(&sc, Some(&mut db), &prow, n, &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), u, &dout.sub_offset(0, 4), mp, None, std::ptr::null_mut())
                .unwrap()
                .unwrap();
            gpu.claude_doorbell_wait(&mut db, &tk, &mut w).unwrap();
            let hi = sample_categorical(&q, u);
            let ex = sample_exact(&q, u);
            gpu_vs_exact += (w[0] != ex) as usize;
            host_vs_exact += (hi != ex) as usize;
            if w[0] != hi {
                idx_bad += 1;
                if idx_bad <= 3 {
                    eprintln!("  idx mismatch n={n} u={u}: gpu {} host {} (q {} / {})", w[0], hi, q[w[0] as usize], q[hi as usize]);
                }
            }
            if f32::from_bits(w[1]).to_bits() != q[w[0] as usize].to_bits() {
                q_bad += 1;
            }
            if f32::from_bits(w[2]).to_bits() != pmax.to_bits() {
                top_bad += 1;
            }
            let want_tok = if compressed { map[w[0] as usize] } else { w[0] };
            if w[3] != want_tok {
                tok_bad += 1;
            }
        }
        // A nucleus of O(1e5) near-equal probs is at the f32 resolution of any
        // cumulative sum (the host's sequential f32 sum drifts much further,
        // see host-f32 vs exact); require exactness for LM-sized nuclei.
        total_bad += q_bad + top_bad + tok_bad + if kept <= 10_000 { gpu_vs_exact } else { 0 };
        println!(
            "draft sampler n={n:6} temp={temp} top_p={top_p} top_k={top_k} min_p={min_p}: nucleus {kept} ids, {draws} draws: gpu != host-f32 {idx_bad}; vs exact f64 inverse CDF: gpu {gpu_vs_exact}, host-f32 {host_vs_exact}; q mismatches {q_bad}, max-p mismatches {top_bad}, token mismatches {tok_bad}"
        );
        let _ = gpu.free_tensor(row_l);
    }

    // ---------------- 2. accept ----------------
    let mut acc_bad = 0usize;
    let (mut bonus_only, mut bonus_gpu_exact, mut bonus_host_exact) = (0usize, 0usize, 0usize);
    let (mut n_rej, mut n_full, mut n_eos, mut n_res_empty) = (0usize, 0usize, 0usize, 0usize);
    for trial in 0..trials {
        let compressed = trial % 3 == 2;
        let vd = if compressed { v_c } else { v_full };
        let d = 1 + (rng.next() as usize) % max_d;
        let (temp, top_p, top_k) = match trial % 4 {
            0 => (0.7f32, 0.95f32, 20usize),
            1 => (1.0, 0.95, 0),
            2 => (0.5, 0.9, 40),
            _ => (1.0, 1.0, 0),
        };
        // target logits for d+1 rows; draft = target (+ noise), compressed rows gathered by map
        let mut t_rows: Vec<Vec<f32>> = Vec::new();
        for _ in 0..=d {
            t_rows.push(logits(&mut rng, v_full, 6));
        }
        let noise = if trial % 7 == 0 { 0.0 } else { 0.8 };
        let mut d_rows: Vec<Vec<f32>> = Vec::new();
        for k in 0..d {
            let full: Vec<f32> = t_rows[k].iter().map(|&x| x + noise * (rng.unit() - 0.5) * 4.0).collect();
            d_rows.push(if compressed { map.iter().map(|&i| full[i as usize]).collect() } else { full });
        }
        // upload + softmax (GPU) both sides
        let tl: Vec<f32> = t_rows.concat();
        let tl_b: Vec<u8> = tl.iter().flat_map(|x| x.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&logits_g.buf, &tl_b).unwrap();
        gpu.softmax_temp_topp_batched_into_f32(&logits_g.sub_offset(0, (d + 1) * v_full), &tprobs.sub_offset(0, (d + 1) * v_full), &ttau, &tz, v_full, d + 1, temp, top_p, top_k, 0.0).unwrap();
        let dl_: Vec<f32> = d_rows.concat();
        let dl_b: Vec<u8> = dl_.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let dlg = gpu.upload_raw(&dl_b, &[dl_b.len()]).unwrap();
        let dlg = GpuTensor { buf: dlg.buf, shape: vec![d * vd], dtype: DType::F32 };
        gpu.softmax_temp_topp_batched_into_f32(&dlg, &dprobs.sub_offset(0, d * vd), &dtau, &dz, vd, d, temp, top_p, top_k, 0.0).unwrap();
        // draft picks with the GPU sampler (as the MTP draft loop does)
        let mut cands = Vec::new();
        let mut dprob_host = Vec::new();
        let mut w = [0u32; 4];
        for k in 0..d {
            let u = rng.unit();
            let mp = if compressed { map_g.buf.as_ptr() as *const _ } else { std::ptr::null() };
            let tk = gpu
                .claude_mtp_sample_row(&sc, Some(&mut db), &dprobs.sub_offset(k * vd, vd), vd, &dtau.sub_offset(k, 1), &dz.sub_offset(k, 1), u, &dout.sub_offset(4 * k, 4), mp, None, std::ptr::null_mut())
                .unwrap()
                .unwrap();
            gpu.claude_doorbell_wait(&mut db, &tk, &mut w).unwrap();
            cands.push(w[3]);
            dprob_host.push(f32::from_bits(w[1]));
        }
        let eos = if trial % 11 == 0 { cands[0] } else if trial % 13 == 0 && d > 1 { cands[1] } else { 248_046 };
        let us: Vec<f32> = (0..=d).map(|_| rng.unit()).collect();
        let tk = gpu
            .claude_mtp_sample_accept(&sc, Some(&mut db), &tprobs, v_full, &ttau, &tz, &dprobs, vd, &dtau, &dz, &dout, if compressed { inv_g.buf.as_ptr() as *const _ } else { std::ptr::null() }, d, eos, &us, &aout)
            .unwrap()
            .unwrap();
        let mut aw = vec![0u32; 4 + d + 1];
        gpu.claude_doorbell_wait(&mut db, &tk, &mut aw).unwrap();
        let gres = claude_decode_accept(&aw);
        // host reference on downloaded rows
        gpu.hip.device_synchronize().unwrap();
        let tp = dl(&gpu, &tprobs, (d + 1) * v_full);
        let tt = dl(&gpu, &ttau, d + 1);
        let tzz = dl(&gpu, &tz, d + 1);
        let dp = dl(&gpu, &dprobs, d * vd);
        let dt = dl(&gpu, &dtau, d);
        let dzz = dl(&gpu, &dz, d);
        let trows: Vec<Vec<f32>> = (0..=d)
            .map(|k| {
                let mut r = tp[k * v_full..(k + 1) * v_full].to_vec();
                apply_topp_trunc(&mut r, tt[k], tzz[k]);
                r
            })
            .collect();
        let drows_full: Vec<Vec<f32>> = (0..d)
            .map(|k| {
                let mut r = dp[k * vd..(k + 1) * vd].to_vec();
                apply_topp_trunc(&mut r, dt[k], dzz[k]);
                if compressed {
                    let mut full = vec![0.0f32; v_full];
                    for (j, &pj) in r.iter().enumerate() {
                        if pj > 0.0 {
                            full[map[j] as usize] = pj;
                        }
                    }
                    full
                } else {
                    r
                }
            })
            .collect();
        let (hc, ha, he, hu) = host_accept(&trows, &drows_full, &dprob_host, &cands, eos, &us);
        if ha < d && !he {
            n_rej += 1;
            let resid: f32 = (0..v_full).map(|i| (trows[ha][i] - drows_full[ha][i]).max(0.0)).sum();
            if resid <= 0.0 {
                n_res_empty += 1;
            }
        } else if he {
            n_eos += 1;
        } else {
            n_full += 1;
        }
        // exact (f64) bonus for the same decision path
        let exact_bonus = if gres.accept_count == ha && gres.n_consumed == hu && !(he && hc.len() == ha) {
            let row = ha;
            let w: Vec<f32> = if ha < d {
                (0..v_full).map(|i| (trows[row][i] - drows_full[row][i]).max(0.0)).collect()
            } else {
                trows[row].clone()
            };
            Some(sample_exact(&w, us[hu - 1]))
        } else {
            None
        };
        let prefix_ok = gres.accept_count == ha
            && gres.n_consumed == hu
            && gres.committed.len() == hc.len()
            && gres.committed[..ha] == hc[..ha];
        if prefix_ok && gres.committed != hc {
            bonus_only += 1;
            if let Some(eb) = exact_bonus {
                bonus_gpu_exact += (gres.committed[ha] == eb) as usize;
                bonus_host_exact += (hc[ha] == eb) as usize;
            }
        }
        if !prefix_ok {
            acc_bad += 1;
            if acc_bad <= 5 {
                eprintln!("  accept mismatch trial {trial} d={d} compressed={compressed}: gpu {:?} host {:?}", gres, (hc, ha, he, hu));
            }
        }
        let _ = gpu.free_tensor(dlg);
    }
    println!(
        "accept: {trials} trials ({n_rej} rejections [{n_res_empty} with empty residual], {n_full} full accepts, {n_eos} EOS stops): decision mismatches (accept count / uniforms / prefix / EOS) {acc_bad}; bonus-token-only differences {bonus_only} (of those, equal to the exact f64 inverse CDF: gpu {bonus_gpu_exact}, host-f32 {bonus_host_exact})"
    );
    println!(
        "{} (draft-sampler mismatches {total_bad}, accept mismatches {acc_bad}; doorbell stats {:?})",
        if total_bad == 0 && acc_bad == 0 { "PASS" } else { "CHECK" },
        db.stats
    );

    // ---------------- 2b. multi-block softmax vs the original ----------------
    let smx = rdna_compute::claude_sample::ClaudeSoftmaxScratch::new(&mut gpu, max_d + 1).unwrap();
    let probs2 = gpu.zeros(&[(max_d + 1) * v_full], DType::F32).unwrap();
    let tau2 = gpu.zeros(&[max_d + 1], DType::F32).unwrap();
    let z2 = gpu.zeros(&[max_d + 1], DType::F32).unwrap();
    let mut smx_rows = 0usize;
    let (mut tau_eq, mut nucleus_diff, mut worst_p, mut worst_z) = (0usize, 0usize, 0.0f64, 0.0f64);
    for trial in 0..40 {
        let rows = 1 + trial % 4;
        let n = if trial % 5 == 4 { v_c } else { v_full };
        let (temp, top_p, top_k, min_p) = match trial % 6 {
            0 => (0.7f32, 0.95f32, 20usize, 0.0f32),
            1 => (1.0, 0.95, 0, 0.0),
            2 => (0.5, 0.9, 40, 0.05),
            3 => (1.0, 1.0, 0, 0.0),
            4 => (0.3, 0.8, 0, 0.0),
            _ => (1.2, 0.99, 100, 0.0),
        };
        let lg: Vec<f32> = (0..rows).flat_map(|r| if (trial + r) % 3 == 0 { logits(&mut rng, n, 12) } else { lm_logits(&mut rng, n) }).collect();
        let lb: Vec<u8> = lg.iter().flat_map(|x| x.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&logits_g.buf, &lb).unwrap();
        let lview = logits_g.sub_offset(0, rows * n);
        gpu.softmax_temp_topp_batched_into_f32(&lview, &tprobs.sub_offset(0, rows * n), &ttau, &tz, n, rows, temp, top_p, top_k, min_p).unwrap();
        gpu.claude_softmax_topp_mb(&smx, &lview, &probs2.sub_offset(0, rows * n), &tau2, &z2, n, rows, temp, top_p, top_k, min_p).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let pa = dl(&gpu, &tprobs, rows * n);
        let pb = dl(&gpu, &probs2, rows * n);
        let ta = dl(&gpu, &ttau, rows);
        let tb = dl(&gpu, &tau2, rows);
        let za = dl(&gpu, &tz, rows);
        let zb = dl(&gpu, &z2, rows);
        for r in 0..rows {
            smx_rows += 1;
            tau_eq += (ta[r].to_bits() == tb[r].to_bits()) as usize;
            worst_z = worst_z.max(((za[r] - zb[r]).abs() / za[r].max(1e-30)) as f64);
            for i in 0..n {
                let a = pa[r * n + i];
                let b = pb[r * n + i];
                if a > 1e-30 {
                    worst_p = worst_p.max(((a - b).abs() / a) as f64);
                }
                if (a >= ta[r]) != (b >= tb[r]) {
                    nucleus_diff += 1;
                }
            }
        }
    }
    println!(
        "softmax_mb vs softmax_temp_topp_batched: {smx_rows} rows: tau bit-equal {tau_eq}/{smx_rows}, nucleus membership differences {nucleus_diff} ids, max rel diff p {worst_p:.2e}, Z {worst_z:.2e}"
    );

    // ---------------- 3. timing ----------------
    let lg = lm_logits(&mut rng, v_full);
    let row_l = up(&mut gpu, &lg);
    {
        // verify rows for the accept timing: LM-like too
        let tl: Vec<f32> = (0..6).flat_map(|_| lm_logits(&mut rng, v_full)).collect();
        let tl_b: Vec<u8> = tl.iter().flat_map(|x| x.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&logits_g.buf, &tl_b).unwrap();
    }
    let (temp, top_p, top_k) = (0.7f32, 0.95f32, 20usize);
    let t_soft1 = time_us(&mut gpu, 20, |g| {
        g.softmax_temp_topp_batched_into_f32(&row_l, &dprobs.sub_offset(0, v_full), &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), v_full, 1, temp, top_p, top_k, 0.0).unwrap();
    });
    let t_soft4 = time_us(&mut gpu, 20, |g| {
        g.softmax_temp_topp_batched_into_f32(&logits_g.sub_offset(0, 4 * v_full), &tprobs.sub_offset(0, 4 * v_full), &ttau, &tz, v_full, 4, temp, top_p, top_k, 0.0).unwrap();
    });
    let t_soft_c = time_us(&mut gpu, 20, |g| {
        g.softmax_temp_topp_batched_into_f32(&row_l, &dprobs.sub_offset(0, v_c), &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), v_c, 1, temp, top_p, top_k, 0.0).unwrap();
    });
    gpu.softmax_temp_topp_batched_into_f32(&row_l, &dprobs.sub_offset(0, v_full), &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), v_full, 1, temp, top_p, top_k, 0.0).unwrap();
    let t_smx1 = time_us(&mut gpu, 20, |g| {
        g.claude_softmax_topp_mb(&smx, &row_l, &probs2.sub_offset(0, v_full), &tau2.sub_offset(0, 1), &z2.sub_offset(0, 1), v_full, 1, temp, top_p, top_k, 0.0).unwrap();
    });
    let t_smx4 = time_us(&mut gpu, 20, |g| {
        g.claude_softmax_topp_mb(&smx, &logits_g.sub_offset(0, 4 * v_full), &probs2.sub_offset(0, 4 * v_full), &tau2, &z2, v_full, 4, temp, top_p, top_k, 0.0).unwrap();
    });
    let t_smx_c = time_us(&mut gpu, 20, |g| {
        g.claude_softmax_topp_mb(&smx, &row_l, &probs2.sub_offset(0, v_c), &tau2.sub_offset(0, 1), &z2.sub_offset(0, 1), v_c, 1, temp, top_p, top_k, 0.0).unwrap();
    });
    println!("timing: softmax_mb n=248320 x1 {t_smx1:.1} us, x4 rows {t_smx4:.1} us, n=32768 x1 {t_smx_c:.1} us");
    let t_samp = time_us(&mut gpu, 50, |g| {
        let _ = g.claude_mtp_sample_row(&sc, None, &dprobs.sub_offset(0, v_full), v_full, &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), 0.37, &dout.sub_offset(0, 4), std::ptr::null(), None, std::ptr::null_mut()).unwrap();
    });
    // old host path per draft step (wall): softmax + 3 blocking downloads + host trunc + sample
    let mut old = Vec::new();
    let mut new = Vec::new();
    for i in 0..60 {
        let t0 = Instant::now();
        gpu.softmax_temp_topp_batched_into_f32(&row_l, &dprobs.sub_offset(0, v_full), &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), v_full, 1, temp, top_p, top_k, 0.0).unwrap();
        let mut p = dl(&gpu, &dprobs.sub_offset(0, v_full), v_full);
        let tau = dl(&gpu, &dtau, 1)[0];
        let z = dl(&gpu, &dz, 1)[0];
        let _top = p.iter().cloned().fold(0.0f32, f32::max);
        apply_topp_trunc(&mut p, tau, z);
        let idx = sample_categorical(&p, 0.37);
        let _keep = p.clone(); // the draft_softmaxes push
        std::hint::black_box(idx);
        old.push(t0.elapsed().as_secs_f64() * 1e6);
        let t1 = Instant::now();
        gpu.softmax_temp_topp_batched_into_f32(&row_l, &dprobs.sub_offset(0, v_full), &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), v_full, 1, temp, top_p, top_k, 0.0).unwrap();
        let tk = gpu.claude_mtp_sample_row(&sc, Some(&mut db), &dprobs.sub_offset(0, v_full), v_full, &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), 0.37, &dout.sub_offset(0, 4), std::ptr::null(), None, std::ptr::null_mut()).unwrap().unwrap();
        let mut w = [0u32; 4];
        gpu.claude_doorbell_wait(&mut db, &tk, &mut w).unwrap();
        new.push(t1.elapsed().as_secs_f64() * 1e6);
        let _ = i;
    }
    let mut new_mb = Vec::new();
    for _ in 0..60 {
        let t1 = Instant::now();
        gpu.claude_softmax_topp_mb(&smx, &row_l, &dprobs.sub_offset(0, v_full), &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), v_full, 1, temp, top_p, top_k, 0.0).unwrap();
        let tk = gpu.claude_mtp_sample_row(&sc, Some(&mut db), &dprobs.sub_offset(0, v_full), v_full, &dtau.sub_offset(0, 1), &dz.sub_offset(0, 1), 0.37, &dout.sub_offset(0, 4), std::ptr::null(), None, std::ptr::null_mut()).unwrap().unwrap();
        let mut w = [0u32; 4];
        gpu.claude_doorbell_wait(&mut db, &tk, &mut w).unwrap();
        new_mb.push(t1.elapsed().as_secs_f64() * 1e6);
    }
    new_mb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("timing: draft pick wall with softmax_mb + device sampler + doorbell {:.1} us", new_mb[new_mb.len() / 2]);
    old.sort_by(|a, b| a.partial_cmp(b).unwrap());
    new.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "timing: softmax_temp_topp n=248320 x1 {t_soft1:.1} us, x4 rows {t_soft4:.1} us, n=32768 x1 {t_soft_c:.1} us | claude_mtp_sample_row {t_samp:.1} us GPU | draft pick wall (softmax + readback + host work): old host path {:.1} us, device sampler + doorbell {:.1} us",
        old[old.len() / 2],
        new[new.len() / 2]
    );
    // accept: old host (4 rows download + host) vs device
    let d = 3usize;
    let us = [0.2f32, 0.6, 0.9, 0.4];
    let mut old_a = Vec::new();
    let mut new_a = Vec::new();
    for _ in 0..30 {
        let t0 = Instant::now();
        gpu.softmax_temp_topp_batched_into_f32(&logits_g.sub_offset(0, 4 * v_full), &tprobs.sub_offset(0, 4 * v_full), &ttau, &tz, v_full, 4, temp, top_p, top_k, 0.0).unwrap();
        let tp = dl(&gpu, &tprobs.sub_offset(0, 4 * v_full), 4 * v_full);
        let tt = dl(&gpu, &ttau, 4);
        let tzz = dl(&gpu, &tz, 4);
        let mut rows = Vec::new();
        for k in 0..4 {
            let mut r = tp[k * v_full..(k + 1) * v_full].to_vec();
            apply_topp_trunc(&mut r, tt[k], tzz[k]);
            rows.push(r);
        }
        std::hint::black_box(sample_categorical(&rows[3], 0.4));
        old_a.push(t0.elapsed().as_secs_f64() * 1e6);
        let t1 = Instant::now();
        gpu.softmax_temp_topp_batched_into_f32(&logits_g.sub_offset(0, 4 * v_full), &tprobs.sub_offset(0, 4 * v_full), &ttau, &tz, v_full, 4, temp, top_p, top_k, 0.0).unwrap();
        let tk = gpu.claude_mtp_sample_accept(&sc, Some(&mut db), &tprobs, v_full, &ttau, &tz, &dprobs, v_full, &dtau, &dz, &dout, std::ptr::null(), d, 248_046, &us, &aout).unwrap().unwrap();
        let mut aw = vec![0u32; 4 + d + 1];
        gpu.claude_doorbell_wait(&mut db, &tk, &mut aw).unwrap();
        new_a.push(t1.elapsed().as_secs_f64() * 1e6);
    }
    let mut new_a_mb = Vec::new();
    for _ in 0..30 {
        let t1 = Instant::now();
        gpu.claude_softmax_topp_mb(&smx, &logits_g.sub_offset(0, 4 * v_full), &tprobs.sub_offset(0, 4 * v_full), &ttau, &tz, v_full, 4, temp, top_p, top_k, 0.0).unwrap();
        let tk = gpu.claude_mtp_sample_accept(&sc, Some(&mut db), &tprobs, v_full, &ttau, &tz, &dprobs, v_full, &dtau, &dz, &dout, std::ptr::null(), d, 248_046, &us, &aout).unwrap().unwrap();
        let mut aw = vec![0u32; 4 + d + 1];
        gpu.claude_doorbell_wait(&mut db, &tk, &mut aw).unwrap();
        new_a_mb.push(t1.elapsed().as_secs_f64() * 1e6);
    }
    new_a_mb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("timing: accept wall with softmax_mb + device accept + doorbell {:.1} us", new_a_mb[new_a_mb.len() / 2]);
    old_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    new_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t_acc = time_us(&mut gpu, 30, |g| {
        let _ = g.claude_mtp_sample_accept(&sc, None, &tprobs, v_full, &ttau, &tz, &dprobs, v_full, &dtau, &dz, &dout, std::ptr::null(), d, 248_046, &us, &aout).unwrap();
    });
    println!(
        "timing: accept (d=3) wall: old host path (softmax 4 rows + 4 MB download + host) {:.1} us, device accept + doorbell {:.1} us; accept kernel {t_acc:.1} us GPU",
        old_a[old_a.len() / 2],
        new_a[new_a.len() / 2]
    );
    sc.free_gpu(&mut gpu);
    db.free(&gpu);
    println!("DONE");
}
