// claude: MTP verify lm_head at 2..8 rows (Qwen3.8-27B: MQ4G256V2 / qt=44,
// M = vocab 248320, K = 5120). Incumbent: gemm_mq4g256v2_batched_lmhead
// (memset + gfx1100 ldsstage residual WMMA, f16 activations). Candidate:
// claude_q8x_project (int8 activations per 32-block, one warp per row, all
// rows read the weight once). Reference numerics: gemv_mq4g256v2 per row
// (f32 activations; the production batch-1 lm_head kernel).
//
//   --ms 62080[,248320]  --bs 2,3,4,8  --iters 20  --trials 64
// A full-vocab M (> 100000) is only allocated when > 3 GiB VRAM is free.
use rdna_compute::{DType, Gpu, GpuTensor};

struct Rng(u64);
impl Rng {
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.u32() as f32 / 2147483648.0
    }
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732
    }
}

fn f16_bits(v: f32) -> u16 {
    half_from_f32(v)
}

// Minimal f32 -> f16 (round-to-nearest-even), normal range only.
fn half_from_f32(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn make_mq4v2(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut v = vec![0u8; m * groups * 136];
    for r in 0..m {
        for g in 0..groups {
            let off = (r * groups + g) * 136;
            for half in 0..2 {
                let sc = 0.002 + 0.01 * rng.unit();
                let zp = -7.5 * sc + (rng.unit() - 0.5) * sc;
                v[off + half * 4..off + half * 4 + 2].copy_from_slice(&f16_bits(sc).to_le_bytes());
                v[off + half * 4 + 2..off + half * 4 + 4]
                    .copy_from_slice(&f16_bits(zp).to_le_bytes());
            }
            for i in 0..128 {
                v[off + 8 + i] = rng.u32() as u8;
            }
        }
    }
    v
}

fn vram_free_mib() -> Option<u64> {
    let rd = |f: &str| -> Option<u64> {
        std::fs::read_to_string(format!("/sys/class/drm/card1/device/{f}"))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    Some((rd("mem_info_vram_total")? - rd("mem_info_vram_used")?) / 1048576)
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let ms: Vec<usize> = arg("--ms")
        .unwrap_or_else(|| "62080".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let bs: Vec<usize> = arg("--bs")
        .unwrap_or_else(|| "2,3,4,8".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let iters: usize = arg("--iters").map(|v| v.parse().unwrap()).unwrap_or(20);
    let trials: usize = arg("--trials").map(|v| v.parse().unwrap()).unwrap_or(64);
    let k = 5120usize;
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  iters={iters} trials={trials}", gpu.arch);
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    let mut rng = Rng(0x1a2b_3c4d_5e6f_7788);
    let bmax = *bs.iter().max().unwrap();

    for &m in &ms {
        let w_bytes = m * (k / 256) * 136;
        if m > 100_000 {
            let free = vram_free_mib().unwrap_or(0);
            let need = (w_bytes as u64 >> 20) + 3072;
            if free < need {
                eprintln!("skip M={m}: {free} MiB free < {need} MiB (weights + 3 GiB headroom)");
                continue;
            }
            eprintln!("M={m}: {free} MiB free");
        }
        let w_host = make_mq4v2(&mut rng, m, k);
        let w = gpu.upload_raw(&w_host, &[w_host.len()]).unwrap();
        drop(w_host);
        let x = gpu.alloc_tensor(&[bmax * k], DType::F32).unwrap();
        let y_ref = gpu.zeros(&[bmax * m], DType::F32).unwrap();
        let y_w = gpu.zeros(&[bmax * m], DType::F32).unwrap();
        let y_q = gpu.zeros(&[bmax * m], DType::F32).unwrap();
        let y_rb = gpu.zeros(&[bmax * m], DType::F32).unwrap();
        let run_ref = |gpu: &mut Gpu, b: usize| {
            for r in 0..b {
                gpu.gemv_mq4g256v2(
                    &w,
                    &x.sub_offset(r * k, k),
                    &y_ref.sub_offset(r * m, m),
                    m,
                    k,
                )
                .unwrap();
            }
        };
        let run_wmma = |gpu: &mut Gpu, b: usize| {
            gpu.gemm_mq4g256v2_batched_lmhead(
                &w,
                &x.sub_offset(0, b * k),
                &y_w.sub_offset(0, b * m),
                m,
                k,
                b,
            )
            .unwrap();
        };
        let run_q8x = |gpu: &mut Gpu, b: usize| {
            let yv = y_q.sub_offset(0, b * m);
            gpu.claude_q8x_project(&[(&w, &yv, m)], &x.sub_offset(0, b * k), k, b, false)
                .unwrap();
        };
        // f32-activation small-batch GEMV (the `CLAUDE_XBATCH_KERNEL=rb` kernel).
        let run_rb = |gpu: &mut Gpu, b: usize| {
            let yv = y_rb.sub_offset(0, b * m);
            gpu.claude_gemv_mq4g256v2_xbatch(&w, &x.sub_offset(0, b * k), &yv, m, k, b, false)
                .unwrap();
        };
        let upload_x = |gpu: &mut Gpu, rng: &mut Rng| {
            let xs: Vec<f32> = (0..bmax * k).map(|_| rng.gauss()).collect();
            let bytes =
                unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) };
            gpu.hip.memcpy_htod(&x.buf, bytes).unwrap();
        };
        // clock warmup ~1 s
        upload_x(&mut gpu, &mut rng);
        let t = std::time::Instant::now();
        while t.elapsed().as_secs_f64() < 1.0 {
            run_wmma(&mut gpu, 4);
            gpu.hip.device_synchronize().unwrap();
        }
        for &b in &bs {
            let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu, usize)| -> f64 {
                for _ in 0..3 {
                    f(gpu, b);
                }
                let mut ts = Vec::with_capacity(iters);
                for _ in 0..iters {
                    gpu.hip.event_record(&ev0, None).unwrap();
                    f(gpu, b);
                    gpu.hip.event_record(&ev1, None).unwrap();
                    gpu.hip.event_synchronize(&ev1).unwrap();
                    ts.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3);
                }
                median(&mut ts)
            };
            let us_ref = time(&mut gpu, &run_ref);
            let us_w = time(&mut gpu, &run_wmma);
            let us_q = time(&mut gpu, &run_q8x);
            let us_rb = time(&mut gpu, &run_rb);
            let gbs = |us: f64| w_bytes as f64 / (us * 1e3);
            println!(
                "M={m} K={k} B={b}: wmma_ldsstage {us_w:8.1} us ({:4.0} GB/s)  q8x {us_q:8.1} us ({:4.0} GB/s)  speedup {:.2}x  | f32 rb-xbatch {us_rb:8.1} us ({:4.0} GB/s) | ref {b}x gemv {us_ref:8.1} us ({:4.0} GB/s per row)",
                gbs(us_w),
                gbs(us_q),
                us_w / us_q,
                gbs(us_rb),
                gbs(us_ref / b as f64)
            );
            // Numerics over `trials` random activation sets.
            let (mut w_max_rel, mut q_max_rel, mut w_l2, mut q_l2) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let (mut w_max_abs, mut q_max_abs) = (0.0f32, 0.0f32);
            let (mut w_flip, mut q_flip, mut qw_flip, mut rows) = (0usize, 0usize, 0usize, 0usize);
            let (mut rb_max_rel, mut rb_flip) = (0.0f64, 0usize);
            let (mut min_gap_rel, mut q_err_at_top) = (f64::MAX, 0.0f64);
            for _ in 0..trials {
                upload_x(&mut gpu, &mut rng);
                run_ref(&mut gpu, b);
                run_wmma(&mut gpu, b);
                run_q8x(&mut gpu, b);
                run_rb(&mut gpu, b);
                gpu.hip.device_synchronize().unwrap();
                let yrb = gpu.download_f32(&y_rb.sub_offset(0, b * m)).unwrap();
                let r = gpu.download_f32(&y_ref.sub_offset(0, b * m)).unwrap();
                let yw = gpu.download_f32(&y_w.sub_offset(0, b * m)).unwrap();
                let yq = gpu.download_f32(&y_q.sub_offset(0, b * m)).unwrap();
                for row in 0..b {
                    let rr = &r[row * m..(row + 1) * m];
                    let rw = &yw[row * m..(row + 1) * m];
                    let rq = &yq[row * m..(row + 1) * m];
                    let scale = rr.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
                    let (mut sw, mut sq, mut sr) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..m {
                        let dw = (rw[i] - rr[i]).abs();
                        let dq = (rq[i] - rr[i]).abs();
                        w_max_abs = w_max_abs.max(dw);
                        q_max_abs = q_max_abs.max(dq);
                        w_max_rel = w_max_rel.max(dw as f64 / scale);
                        q_max_rel = q_max_rel.max(dq as f64 / scale);
                        sw += (dw as f64).powi(2);
                        sq += (dq as f64).powi(2);
                        sr += (rr[i] as f64).powi(2);
                    }
                    w_l2 = w_l2.max((sw / sr).sqrt());
                    q_l2 = q_l2.max((sq / sr).sqrt());
                    let rrb = &yrb[row * m..(row + 1) * m];
                    for i in 0..m {
                        rb_max_rel = rb_max_rel.max((rrb[i] - rr[i]).abs() as f64 / scale);
                    }
                    rb_flip += (argmax(rrb) != argmax(rr)) as usize;
                    let (ar, aw, aq) = (argmax(rr), argmax(rw), argmax(rq));
                    w_flip += (aw != ar) as usize;
                    q_flip += (aq != ar) as usize;
                    qw_flip += (aq != aw) as usize;
                    rows += 1;
                    let mut sorted: Vec<f32> = rr.to_vec();
                    sorted.select_nth_unstable_by(m - 2, |a, b| a.partial_cmp(b).unwrap());
                    let top2 = sorted[m - 2];
                    let gap = (rr[ar] - top2) as f64 / scale;
                    min_gap_rel = min_gap_rel.min(gap);
                    q_err_at_top = q_err_at_top.max((rq[ar] - rr[ar]).abs() as f64 / scale);
                }
            }
            println!(
                "  numerics vs f32 gemv ({rows} rows): wmma max_abs {w_max_abs:.3e} rel_max {w_max_rel:.3e} rel_l2 {w_l2:.3e} argmax_flips {w_flip} | q8x max_abs {q_max_abs:.3e} rel_max {q_max_rel:.3e} rel_l2 {q_l2:.3e} argmax_flips {q_flip} | q8x-vs-wmma flips {qw_flip} | min top1-top2 gap {min_gap_rel:.3e} (rel), q8x err at top1 {q_err_at_top:.3e} | rb rel_max {rb_max_rel:.3e} flips {rb_flip}"
            );
        }
        gpu.free_tensor(y_q).unwrap();
        gpu.free_tensor(y_rb).unwrap();
        gpu.free_tensor(y_w).unwrap();
        gpu.free_tensor(y_ref).unwrap();
        gpu.free_tensor(x).unwrap();
        gpu.free_tensor(w).unwrap();
    }
}
