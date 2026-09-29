// claude v4: check/bench of `claude_attn5` (1..=8 verify rows) against
// `claude_attn4` (1..=4 rows), the production rows kernel
// (`attention_flash_q8_0_rows_masked`, >= 2 rows) and a CPU f64 reference.
// Qwen3.8-27B verify shape: 24 q heads / 4 kv heads / head_dim 256, q8_0 KV.
//
//   bench_claude_attn5 --seq 40000 --rows 4 [--check] [--hot] [--iters 20]
//       [--reps 7] [--qamp 24] [--ref-heads 24] [--vram-mb 400]
//
// Cold (default): the KV cache is replicated into enough sets (> 200 MB in
// total, VRAM budget permitting) and the timed calls rotate through them, so
// every call streams its KV from DRAM like a decode step does (the 96 MB
// Infinity Cache is flushed by the weights between layers in production).
// --hot uses one set (the claude_attn4-era bench numbers). GB/s counts the
// q8_0 K+V bytes of `seq` positions (seq * 2176 B).
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn lcg(seed: u32, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5) * amp
        })
        .collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let v = if e == 0 {
        (m as f32) * (2.0f32).powi(-24)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        f32::from_bits(((e + 112) as u32) << 23 | (m << 13))
    };
    if s == 1 { -v } else { v }
}

struct Stats {
    rel_max: f64,
    rel_l2: f64,
    nonfinite: usize,
}

fn compare(out: &[f32], reference: &[f64], heads: &[usize], rows: usize, q_dim: usize) -> Stats {
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    let mut sd = 0.0f64;
    let mut sr = 0.0f64;
    let mut nonfinite = 0usize;
    for r in 0..rows {
        for &h in heads {
            for d in 0..256 {
                let i = r * q_dim + h * 256 + d;
                let o = out[i] as f64;
                if !o.is_finite() {
                    nonfinite += 1;
                }
                let e = (o - reference[i]).abs();
                max_abs = max_abs.max(e);
                max_ref = max_ref.max(reference[i].abs());
                sd += e * e;
                sr += reference[i] * reference[i];
            }
        }
    }
    Stats { rel_max: max_abs / max_ref.max(1e-30), rel_l2: (sd / sr.max(1e-300)).sqrt(), nonfinite }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: f64| -> f64 {
        args.iter().position(|a| a == k).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d)
    };
    let seq_len = argval("--seq", 40000.0) as usize;
    let rows = argval("--rows", 4.0) as usize;
    let iters = argval("--iters", 20.0) as usize;
    let reps = argval("--reps", 7.0) as usize;
    let qamp = argval("--qamp", 24.0) as f32;
    let ref_heads = (argval("--ref-heads", 24.0) as usize).clamp(1, 24);
    let vram_mb = argval("--vram-mb", 400.0);
    let check = args.iter().any(|a| a == "--check");
    let hot = args.iter().any(|a| a == "--hot");
    assert!((1..=8).contains(&rows) && seq_len >= rows);
    let n_heads = 24usize;
    let n_kv_heads = 4usize;
    let head_dim = 256usize;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let per_pos = n_kv_heads * (head_dim / 32) * 34;
    let chunk = 8192usize;
    let max_seq = seq_len.div_ceil(chunk) * chunk;
    let set_bytes = 2 * max_seq * per_pos;

    // partial buffer sizes (floats)
    let rows_kernel_ok = rows >= 2;
    let rows_part = rows * n_heads * max_seq.div_ceil(128) * (2 + head_dim);
    // claude_attn4 span rule (claude_attn3_span): ~320 blocks, clamped 64..=512 keys
    let span_a4 = ((seq_len * n_kv_heads).div_ceil(320).div_ceil(32) * 32).clamp(64, 512);
    let a4_part = rows * n_heads * seq_len.div_ceil(span_a4) * (2 + head_dim);
    let a5_part = rows * n_heads * 64 * rdna_compute::attention::CLAUDE_ATTN5_PSTRIDE;
    let budget = (vram_mb * 1e6) as usize;
    let mut part_floats = a4_part.max(a5_part);
    let mut run_rows = rows_kernel_ok && set_bytes + 4 * rows_part.max(part_floats) <= budget;
    if run_rows {
        part_floats = part_floats.max(rows_part);
    }
    let mut sets = if hot { 1 } else { (200e6 / set_bytes as f64).ceil() as usize }.clamp(1, 8);
    while sets > 1 && sets * set_bytes + 4 * part_floats > budget {
        sets -= 1;
    }
    if set_bytes + 4 * part_floats > budget {
        run_rows = false;
        part_floats = a4_part.max(a5_part);
    }

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "GPU {}  seq={seq_len} rows={rows} sets={sets} ({:.0} MB KV{})",
        gpu.arch,
        (sets * set_bytes) as f64 / 1e6,
        if hot { ", hot" } else { ", cold rotation" }
    );

    // ---- KV set 0 via the production q8_0 writer, replicated into the others ----
    let mut ks: Vec<GpuTensor> = Vec::new();
    let mut vs: Vec<GpuTensor> = Vec::new();
    for _ in 0..sets {
        ks.push(gpu.alloc_tensor(&[max_seq * per_pos], DType::Q8_0).unwrap());
        vs.push(gpu.alloc_tensor(&[max_seq * per_pos], DType::Q8_0).unwrap());
    }
    let d_all_pos = gpu.alloc_tensor(&[max_seq], DType::F32).unwrap();
    let all_pos: Vec<u8> = (0..max_seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_all_pos.buf, &all_pos).unwrap();
    let mut written = 0usize;
    let mut seed = 0xc3c3u32;
    while written < max_seq {
        let d_kf = gpu.upload_f32(&lcg(seed, chunk * kv_dim, 1.0), &[chunk * kv_dim]).unwrap();
        let d_vf = gpu.upload_f32(&lcg(seed ^ 0x9696, chunk * kv_dim, 1.0), &[chunk * kv_dim]).unwrap();
        seed = seed.wrapping_add(7919);
        let pos_view = d_all_pos.sub_offset(written, chunk);
        gpu.kv_cache_write_q8_0_batched(&ks[0], &d_kf, &pos_view, n_kv_heads, head_dim, chunk).unwrap();
        gpu.kv_cache_write_q8_0_batched(&vs[0], &d_vf, &pos_view, n_kv_heads, head_dim, chunk).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(d_kf).unwrap();
        gpu.free_tensor(d_vf).unwrap();
        written += chunk;
    }
    let mut hk = vec![0u8; max_seq * per_pos];
    let mut hv = vec![0u8; max_seq * per_pos];
    gpu.hip.memcpy_dtoh(&mut hk, &ks[0].buf).unwrap();
    gpu.hip.memcpy_dtoh(&mut hv, &vs[0].buf).unwrap();
    for s in 1..sets {
        gpu.hip.memcpy_htod(&ks[s].buf, &hk).unwrap();
        gpu.hip.memcpy_htod(&vs[s].buf, &hv).unwrap();
    }

    let hq = lcg(0xa5a5, rows * q_dim, qamp);
    let d_q = gpu.upload_f32(&hq, &[rows * q_dim]).unwrap();
    let positions: Vec<i32> = (0..rows).map(|i| (seq_len - rows + i) as i32).collect();
    let d_pos = gpu.alloc_tensor(&[rows], DType::F32).unwrap();
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_pos.buf, &pos_bytes).unwrap();
    let d_part = gpu.zeros(&[part_floats], DType::F32).unwrap();
    let d_out_rows = gpu.zeros(&[rows * q_dim], DType::F32).unwrap();
    let d_out_a4 = gpu.zeros(&[rows * q_dim], DType::F32).unwrap();
    let d_out_a5 = gpu.zeros(&[rows * q_dim], DType::F32).unwrap();
    let kv_bytes = (seq_len * per_pos * 2) as f64;

    // ---- f64 reference (heads 0..ref_heads, spread over kv heads) ----
    let heads: Vec<usize> = if ref_heads >= 24 { (0..24).collect() } else { (0..ref_heads).map(|i| (i * 24) / ref_heads).collect() };
    let mut reference = vec![0.0f64; rows * q_dim];
    if check {
        let t0 = Instant::now();
        // dequantize once (code * f16 scale is exact in f32)
        let dequant = |raw: &[u8]| -> Vec<f32> {
            let mut out = vec![0.0f32; seq_len * kv_dim];
            for t in 0..seq_len {
                for kvh in 0..n_kv_heads {
                    for b in 0..head_dim / 32 {
                        let off = t * per_pos + kvh * 272 + b * 34;
                        let sc = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                        let o = (t * n_kv_heads + kvh) * head_dim + b * 32;
                        for i in 0..32 {
                            out[o + i] = (raw[off + 2 + i] as i8) as f32 * sc;
                        }
                    }
                }
            }
            out
        };
        let fk = dequant(&hk);
        let fv = dequant(&hv);
        let jobs: Vec<(usize, usize)> = (0..rows).flat_map(|r| heads.iter().map(move |&h| (r, h))).collect();
        let nth = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
        let results: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..nth)
                .map(|ti| {
                    let jobs = &jobs;
                    let fk = &fk;
                    let fv = &fv;
                    let hq = &hq;
                    let positions = &positions;
                    sc.spawn(move || {
                        let mut outv = Vec::new();
                        for (j, &(r, h)) in jobs.iter().enumerate() {
                            if j % nth != ti {
                                continue;
                            }
                            let kvh = h / 6;
                            let n = positions[r] as usize + 1;
                            let q: Vec<f64> = (0..256).map(|d| hq[r * q_dim + h * 256 + d] as f64).collect();
                            let mut s = vec![0.0f64; n];
                            let mut mx = f64::NEG_INFINITY;
                            for t in 0..n {
                                let kr = &fk[(t * n_kv_heads + kvh) * head_dim..][..head_dim];
                                let mut acc = 0.0f64;
                                for d in 0..256 {
                                    acc += q[d] * kr[d] as f64;
                                }
                                s[t] = acc / 16.0;
                                mx = mx.max(s[t]);
                            }
                            let mut den = 0.0f64;
                            let mut o = vec![0.0f64; 256];
                            for t in 0..n {
                                let p = (s[t] - mx).exp();
                                den += p;
                                let vr = &fv[(t * n_kv_heads + kvh) * head_dim..][..head_dim];
                                for d in 0..256 {
                                    o[d] += p * vr[d] as f64;
                                }
                            }
                            for v in o.iter_mut() {
                                *v /= den;
                            }
                            outv.push((r * q_dim + h * 256, o));
                        }
                        outv
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for part in results {
            for (base, o) in part {
                reference[base..base + 256].copy_from_slice(&o);
            }
        }
        eprintln!("f64 reference: {} (row, head) pairs in {:.1} s", jobs.len(), t0.elapsed().as_secs_f64());
    }

    let mut cur = 0usize;
    let time = |gpu: &mut Gpu, cur: &mut usize, f: &mut dyn FnMut(&mut Gpu, usize) -> bool| -> f64 {
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < 1.0 {
            for _ in 0..20 {
                *cur = (*cur + 1) % sets;
                f(gpu, *cur);
            }
            gpu.hip.device_synchronize().unwrap();
        }
        let mut v = Vec::new();
        for _ in 0..reps {
            gpu.hip.device_synchronize().unwrap();
            let t = Instant::now();
            for _ in 0..iters {
                *cur = (*cur + 1) % sets;
                f(gpu, *cur);
            }
            gpu.hip.device_synchronize().unwrap();
            v.push(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let report = |gpu: &Gpu, name: &str, us: f64, out: &GpuTensor, base: Option<(f64, &str)>| -> bool {
        let mut line = format!("{name:<14} {us:8.1} us  {:5.0} GB/s", kv_bytes / (us * 1e3));
        if let Some((b, bn)) = base {
            line.push_str(&format!("  {:.2}x vs {bn}", b / us));
        }
        let mut ok = true;
        if check {
            let o = gpu.download_f32(out).unwrap();
            let st = compare(&o, &reference, &heads, rows, q_dim);
            line.push_str(&format!("  rel_max {:.2e} rel_l2 {:.2e} nonfinite {}", st.rel_max, st.rel_l2, st.nonfinite));
            ok = st.nonfinite == 0 && st.rel_l2 < 2e-3;
        }
        eprintln!("{line}");
        ok
    };

    let mut ok = true;
    let mut us_rows = None;
    if run_rows {
        let mut f = |gpu: &mut Gpu, s: usize| {
            gpu.attention_flash_q8_0_rows_masked(&d_q, &ks[s], &vs[s], &d_out_rows, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, rows, &d_part).unwrap()
        };
        if f(&mut gpu, 0) {
            gpu.hip.device_synchronize().unwrap();
            let us = time(&mut gpu, &mut cur, &mut f);
            // re-run on set 0 for the parity check (all sets hold identical data)
            f(&mut gpu, 0);
            gpu.hip.device_synchronize().unwrap();
            ok &= report(&gpu, "rows kernel", us, &d_out_rows, None);
            us_rows = Some(us);
        }
    } else {
        eprintln!("rows kernel    n/a ({})", if rows < 2 { "rows < 2" } else { "VRAM budget" });
    }
    let mut us_a4 = None;
    {
        let mut f = |gpu: &mut Gpu, s: usize| {
            gpu.claude_attn4(&d_q, &ks[s], &vs[s], &d_out_a4, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, rows, &d_part).unwrap()
        };
        if f(&mut gpu, 0) {
            let us = time(&mut gpu, &mut cur, &mut f);
            f(&mut gpu, 0);
            gpu.hip.device_synchronize().unwrap();
            ok &= report(&gpu, "claude_attn4", us, &d_out_a4, us_rows.map(|u| (u, "rows")));
            us_a4 = Some(us);
        } else {
            eprintln!("claude_attn4   n/a (rows > 4)");
        }
    }
    {
        let mut f = |gpu: &mut Gpu, s: usize| {
            gpu.claude_attn5(&d_q, &ks[s], &vs[s], &d_out_a5, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, rows, &d_part).unwrap()
        };
        if f(&mut gpu, 0) {
            let us = time(&mut gpu, &mut cur, &mut f);
            f(&mut gpu, 0);
            gpu.hip.device_synchronize().unwrap();
            let base = us_a4.map(|u| (u, "attn4")).or(us_rows.map(|u| (u, "rows")));
            ok &= report(&gpu, "claude_attn5", us, &d_out_a5, base);
            if let (Some(a4), Some(r)) = (us_a4, us_rows) {
                eprintln!("               attn5 {:.2}x vs rows kernel, attn4 {:.2}x vs rows kernel", r / us, r / a4);
            }
        } else {
            eprintln!("claude_attn5 refused this shape (CLAUDE_ATTN5=0?)");
            ok = false;
        }
    }
    if check && !ok {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    eprintln!("OK");
}
