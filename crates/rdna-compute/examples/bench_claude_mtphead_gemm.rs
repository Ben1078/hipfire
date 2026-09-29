// claude: standalone parity + timing for the MTP head's batched projections.
//
// The Qwen3.8 MTP sidecar ships every projection as MQ4G256 (qt=13: HFQ4-G256
// bytes = f32 scale + f32 zero + 128 B nibbles per 256 K, applied to an
// FWHT-rotated x). `weight_gemm_batched` sent them to the portable scalar
// `gemm_hfq4g256`; this bench compares it with the WMMA / MMQ families the
// trunk uses for the same byte layout, on random weights + activations.
//
//   --shapes eh,wq,wk,wv,wo,gate,down   (default eh,wq,wk,wv)
//   --ns 1,2,3,4,16,128,1023,1024       (batch rows)
//   --iters N  --cands scalar,mmq,wmma,res,k2,ksd,lds,gemv,route
//   --precompile   JIT-compile the kernels into the shared cache WITHOUT
//                  touching the GPU (run this before the timed run).
use rdna_compute::{DType, Gpu, GpuTensor};

/// (module name, source) pairs exactly as the dispatchers load them.
const PRECOMPILE: &[(&str, &str)] = &[
    (
        "gemm_hfq4g256",
        include_str!("../../../kernels/src/gemm_hfq4g256.hip"),
    ),
    (
        "gemm_hfq4g256_wmma",
        include_str!("../../../kernels/src/gemm_hfq4g256_wmma.hip"),
    ),
    (
        "gemm_hfq4g256_residual_mmq",
        include_str!("../../../kernels/src/gemm_hfq4g256_residual_mmq.hip"),
    ),
    (
        "gemm_hfq4g256_residual_wmma_k2",
        include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_k2.hip"),
    ),
    (
        "gemm_hfq4g256_residual_wmma_ksplit_det",
        include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma_ksplit_det.hip"),
    ),
    (
        "gemm_ksplit_det_finalize",
        include_str!("../../../kernels/src/gemm_ksplit_det_finalize.hip"),
    ),
    (
        "gemm_hfq4g256_residual_wmma_ldsstage",
        include_str!("../../../kernels/src/gemm_hfq4g256_residual_wmma.hip"),
    ),
    (
        "convert_f32_to_f16",
        include_str!("../../../kernels/src/gemm_hfq4g256_residual_fp16.hip"),
    ),
];

fn precompile() {
    let arch = std::env::var("CLAUDE_ARCH").unwrap_or_else(|_| "gfx1100".into());
    let flags = rdna_compute::FeatureFlags::from_active_config(&arch);
    let mut c = rdna_compute::KernelCompiler::new(&arch, flags.hipcc_extra_flags.clone())
        .expect("compiler");
    for (name, src) in PRECOMPILE {
        let t = std::time::Instant::now();
        c.compile(name, src).expect("compile");
        eprintln!("precompile {name}: {:.1}s", t.elapsed().as_secs_f64());
    }
}
use std::sync::Arc;

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
    // ~N(0,1) (Irwin-Hall, 4 terms)
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732
    }
}

fn make_hfq4g256(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut v = vec![0u8; m * groups * 136];
    for r in 0..m {
        for g in 0..groups {
            let off = (r * groups + g) * 136;
            let scale = 0.002 + 0.01 * rng.unit();
            let zero = -7.5 * scale + (rng.unit() - 0.5) * scale;
            v[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            v[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for i in 0..128 {
                v[off + 8 + i] = rng.u32() as u8;
            }
        }
    }
    v
}

fn dequant_row(w: &[u8], row: usize, k: usize) -> Vec<f64> {
    let groups = k / 256;
    let mut out = vec![0.0f64; k];
    for g in 0..groups {
        let off = (row * groups + g) * 136;
        let scale = f32::from_le_bytes(w[off..off + 4].try_into().unwrap()) as f64;
        let zero = f32::from_le_bytes(w[off + 4..off + 8].try_into().unwrap()) as f64;
        for i in 0..128 {
            let b = w[off + 8 + i];
            out[g * 256 + 2 * i] = scale * (b & 0xF) as f64 + zero;
            out[g * 256 + 2 * i + 1] = scale * (b >> 4) as f64 + zero;
        }
    }
    out
}

struct Stats {
    max_abs: f32,
    rel_max: f32,
    rel_l2: f64,
    nonfinite: usize,
}

fn compare(reference: &[f32], got: &[f32]) -> Stats {
    let scale = reference
        .iter()
        .fold(0.0f32, |m, x| m.max(x.abs()))
        .max(1e-20);
    let (mut max_abs, mut sq_d, mut sq_r, mut nonfinite) = (0.0f32, 0.0f64, 0.0f64, 0usize);
    for (r, g) in reference.iter().zip(got) {
        if !g.is_finite() {
            nonfinite += 1;
            continue;
        }
        max_abs = max_abs.max((r - g).abs());
        sq_d += ((r - g) as f64).powi(2);
        sq_r += (*r as f64).powi(2);
    }
    Stats {
        max_abs,
        rel_max: max_abs / scale,
        rel_l2: (sq_d / sq_r.max(1e-30)).sqrt(),
        nonfinite,
    }
}

/// Current DPM levels (sclk/mclk/fclk MHz) from sysfs, for spotting clock dips.
fn clocks() -> String {
    let cur = |f: &str| -> String {
        std::fs::read_to_string(format!("/sys/class/drm/card1/device/{f}"))
            .ok()
            .and_then(|t| {
                t.lines()
                    .find(|l| l.ends_with('*'))
                    .map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
            })
            .unwrap_or_else(|| "?".into())
    };
    format!(
        "s{}/m{}/f{}",
        cur("pp_dpm_sclk"),
        cur("pp_dpm_mclk"),
        cur("pp_dpm_fclk")
    )
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--precompile") {
        precompile();
        return;
    }
    if args.iter().any(|a| a == "--fill") {
        fill_bench(&args);
        return;
    }
    if args.iter().any(|a| a == "--concat") {
        concat_bench(&args);
        return;
    }
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let shapes_s = arg("--shapes").unwrap_or_else(|| "eh,wq,wk,wv".into());
    let ns_s = arg("--ns").unwrap_or_else(|| "1,2,3,4,16,128,1023,1024".into());
    let cands_s =
        arg("--cands").unwrap_or_else(|| "scalar,mmq,wmma,res,k2,ksd,lds,gemv,route".into());
    let iters: usize = arg("--iters").map(|v| v.parse().unwrap()).unwrap_or(5);
    let burst_ms: f64 = arg("--burst-ms")
        .map(|v| v.parse().unwrap())
        .unwrap_or(120.0);
    let warm_s: f64 = arg("--warm-s").map(|v| v.parse().unwrap()).unwrap_or(2.0);
    let check_cpu = !args.iter().any(|a| a == "--no-cpu");
    let cands: Vec<&str> = cands_s.split(',').collect();
    let all_shapes: &[(&str, usize, usize)] = &[
        ("eh", 5120, 10240),
        ("wq", 12288, 5120),
        ("wk", 1024, 5120),
        ("wv", 1024, 5120),
        ("wo", 5120, 6144),
        ("gate", 17408, 5120),
        ("down", 5120, 17408),
    ];
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  iters={iters}", gpu.arch);
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    let base_flags: rdna_compute::FeatureFlags = (*gpu.flags).clone();
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let mut warmed = false;
    // --cold: before each timed call, stream a 160 MB junk HFQ4 matrix through
    // a GEMV (READ-only, so the 96 MB Infinity Cache is refilled with clean
    // lines, as when the trunk streams its weights between two MTP calls).
    const FLUSH_ROWS: usize = 61_680; // x 2720 B = 160 MB at K=5120
    let flush = if args.iter().any(|a| a == "--cold") {
        let fw = gpu
            .alloc_tensor(&[FLUSH_ROWS * 2720 / 4], DType::F32)
            .unwrap();
        gpu.hip.memset(&fw.buf, 0x11, FLUSH_ROWS * 2720).unwrap();
        let fx = gpu.zeros(&[5120], DType::F32).unwrap();
        let fy = gpu.zeros(&[FLUSH_ROWS], DType::F32).unwrap();
        Some((fw, fx, fy))
    } else {
        None
    };

    for sname in shapes_s.split(',') {
        let &(_, m, k) = all_shapes
            .iter()
            .find(|s| s.0 == sname)
            .expect("unknown shape");
        let w_host = make_hfq4g256(&mut rng, m, k);
        let w = gpu.upload_raw(&w_host, &[w_host.len()]).unwrap();
        let w_mb = w_host.len() as f64 / 1e6;
        for n in ns_s.split(',').map(|s| s.parse::<usize>().unwrap()) {
            let x_host: Vec<f32> = (0..n * k).map(|_| rng.gauss()).collect();
            let x = gpu.upload_f32(&x_host, &[n * k]).unwrap();
            let xh = gpu.alloc_tensor(&[n * k], DType::F16).unwrap();
            let y_ref = gpu.zeros(&[n * m], DType::F32).unwrap();
            let y = gpu.zeros(&[n * m], DType::F32).unwrap();
            let run = |gpu: &mut Gpu, cand: &str, out: &GpuTensor| -> bool {
                *Arc::make_mut(&mut gpu.flags) = base_flags.clone();
                gpu.scratch.invalidate_x_caches_for(x.buf.as_ptr());
                let zero = |gpu: &mut Gpu| gpu.hip.memset(&out.buf, 0, n * m * 4).unwrap();
                match cand {
                    "scalar" => gpu.gemm_hfq4g256(&w, &x, out, m, k, n).unwrap(),
                    "mmq" => gpu.gemm_hfq4g256_mmq_set(&w, &x, out, m, k, n).unwrap(),
                    "wmma" => {
                        gpu.cast_f32_to_f16(&x, &xh).unwrap();
                        gpu.gemm_hfq4g256_wmma(&w, &xh, out, m, k, n).unwrap()
                    }
                    "res" | "k2" | "ksd" | "lds" => {
                        let f = Arc::make_mut(&mut gpu.flags);
                        match cand {
                            "k2" => f.wo_wmma_variant = Some("k2".into()),
                            "ksd" => f.wo_wmma_variant = Some("ksplit_det".into()),
                            "lds" => {
                                if k % 512 != 0 || n > 96 {
                                    return false;
                                }
                                f.hfq4g256_ldsstage_wmma = true
                            }
                            _ => {}
                        }
                        zero(gpu);
                        gpu.gemm_hfq4g256_residual_wmma(&w, &x, out, m, k, n)
                            .unwrap()
                    }
                    "gemv" => {
                        if n > 8 {
                            return false;
                        }
                        for b in 0..n {
                            let xr = x.sub_offset(b * k, k);
                            let yr = out.sub_offset(b * m, m);
                            gpu.gemv_hfq4g256(&w, &xr, &yr, m, k).unwrap();
                        }
                    }
                    "route" => {
                        if !claude_has_route() {
                            return false;
                        }
                        claude_route(gpu, &w, &x, out, m, k, n)
                    }
                    _ => panic!("unknown candidate {cand}"),
                }
                true
            };
            // Reference: the incumbent scalar kernel.
            run(&mut gpu, "scalar", &y_ref);
            gpu.hip.device_synchronize().unwrap();
            let ref_host = gpu.download_f32(&y_ref).unwrap();
            if check_cpu {
                // f64 CPU spot check of the reference on a few (row, batch) pairs.
                let mut worst = 0.0f64;
                let mut scale = 0.0f64;
                for t in 0..16 {
                    let r = (t * 7919 + 13) % m;
                    let b = (t * 31 + 5) % n;
                    let wr = dequant_row(&w_host, r, k);
                    let dot: f64 = wr
                        .iter()
                        .zip(&x_host[b * k..(b + 1) * k])
                        .map(|(a, c)| a * *c as f64)
                        .sum();
                    worst = worst.max((dot - ref_host[b * m + r] as f64).abs());
                    scale = scale.max(dot.abs());
                }
                eprintln!("[{sname} M={m} K={k} N={n}] scalar vs cpu f64: max_abs={worst:.3e} (max |y| {scale:.2})");
            }
            if !warmed {
                let t = std::time::Instant::now();
                while t.elapsed().as_secs_f64() < warm_s {
                    run(&mut gpu, "scalar", &y);
                    gpu.hip.device_synchronize().unwrap();
                }
                warmed = true;
            }
            // Parity first, then round-robin timing (one launch per candidate
            // per round) so clock drift / foreign load spreads evenly.
            let mut live: Vec<&str> = Vec::new();
            let mut stats = Vec::new();
            for cand in &cands {
                if !run(&mut gpu, cand, &y) {
                    continue;
                }
                gpu.hip.device_synchronize().unwrap();
                let got = gpu.download_f32(&y).unwrap();
                stats.push(compare(&ref_host, &got));
                live.push(cand);
                run(&mut gpu, cand, &y);
            }
            // Burst timing: each candidate runs back-to-back for ~burst_ms
            // between two events (clocks ramp as under a real prefill), in
            // round-robin order over `iters` rounds; per-call = burst / reps.
            let mut reps = vec![1usize; live.len()];
            for (ci, cand) in live.iter().enumerate() {
                gpu.hip.event_record(&ev0, None).unwrap();
                run(&mut gpu, cand, &y);
                gpu.hip.event_record(&ev1, None).unwrap();
                gpu.hip.event_synchronize(&ev1).unwrap();
                let us = gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3;
                reps[ci] = ((burst_ms * 1e3 / us.max(1.0)) as usize).clamp(1, 2000);
            }
            let mut ts: Vec<Vec<f64>> = vec![Vec::with_capacity(iters); live.len()];
            let mut clk: Vec<String> = vec![String::new(); live.len()];
            for _ in 0..iters {
                for (ci, cand) in live.iter().enumerate() {
                    if let Some((fw, fx, fy)) = flush.as_ref() {
                        // Cold weights: evict the Infinity Cache, then time ONE call.
                        gpu.gemv_hfq4g256(fw, fx, fy, FLUSH_ROWS, 5120).unwrap();
                        gpu.hip.event_record(&ev0, None).unwrap();
                        run(&mut gpu, cand, &y);
                        gpu.hip.event_record(&ev1, None).unwrap();
                        gpu.hip.event_synchronize(&ev1).unwrap();
                        ts[ci].push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3);
                        continue;
                    }
                    gpu.hip.event_record(&ev0, None).unwrap();
                    for _ in 0..reps[ci] {
                        run(&mut gpu, cand, &y);
                    }
                    gpu.hip.event_record(&ev1, None).unwrap();
                    clk[ci] = clocks();
                    gpu.hip.event_synchronize(&ev1).unwrap();
                    ts[ci].push(
                        gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3
                            / reps[ci] as f64,
                    );
                }
            }
            for (ci, cand) in live.iter().enumerate() {
                let st = &stats[ci];
                let min = ts[ci].iter().cloned().fold(f64::MAX, f64::min);
                let us = median(&mut ts[ci]);
                let tflops = 2.0 * (m * k * n) as f64 / (us * 1e6);
                let gbs = w_mb * 1e3 / us;
                println!(
                    "{sname:5} M={m:6} K={k:6} N={n:5} {cand:7} {us:10.1} us (min {min:9.1})  {tflops:6.2} TFLOPS  {gbs:5.0} GB/s(W)  max_abs={:.3e} rel_max={:.3e} rel_l2={:.3e} nonfinite={} clk={}",
                    st.max_abs, st.rel_max, st.rel_l2, st.nonfinite, clk[ci]
                );
            }
            gpu.free_tensor(x).unwrap();
            gpu.free_tensor(xh).unwrap();
            gpu.free_tensor(y_ref).unwrap();
            gpu.free_tensor(y).unwrap();
        }
        gpu.free_tensor(w).unwrap();
    }
}

fn claude_has_route() -> bool {
    true
}
/// The routed entry point `weight_gemm_batched` now uses for MQ4G256 / HFQ4G256
/// (`CLAUDE_MTP_HEAD_GEMM` picks the branch; default f16 WMMA).
fn claude_route(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) {
    gpu.claude_gemm_hfq4g256_multi(&[(w, y, m)], x, k, n)
        .unwrap()
}

/// `--fill [--n 1024] [--iters 5]`: the GEMM half of one KV-only batched MTP
/// head forward (`mtp_head_forward_block_batched_xfer(.., kv_only = true)`) on
/// the real Qwen3.8 head shapes, old sequence vs new, with the same Gpu calls
/// the head makes (MQ4 = FWHT rotate into one shared `rot`, then the GEMMs):
///   old: rotate(concat) + scalar eh; 3 x [rotate(tmp) + scalar wq/wk/wv]
///   new: rotate(concat) + routed eh; rotate(tmp) + routed [wk, wv] (skip-Q)
///   new+q: as new with wq in the shared call (CLAUDE_MTP_FILL_SKIP_Q=0)
/// Compares cur / k / v (new vs old) and times each sequence (burst median).
fn fill_bench(args: &[String]) {
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let n: usize = arg("--n").map(|v| v.parse().unwrap()).unwrap_or(1024);
    let iters: usize = arg("--iters").map(|v| v.parse().unwrap()).unwrap_or(5);
    let (dim, qd2, kvd) = (5120usize, 12288usize, 1024usize);
    let mut gpu = Gpu::init().expect("GPU init");
    let mut rng = Rng(0xf111_0000_2222_3333);
    let mut up = |gpu: &mut Gpu, m: usize, k: usize| {
        let h = make_hfq4g256(&mut rng, m, k);
        gpu.upload_raw(&h, &[h.len()]).unwrap()
    };
    let w_eh = up(&mut gpu, dim, 2 * dim);
    let w_q = up(&mut gpu, qd2, dim);
    let w_k = up(&mut gpu, kvd, dim);
    let w_v = up(&mut gpu, kvd, dim);
    let xs: Vec<f32> = (0..n * 2 * dim).map(|_| rng.gauss()).collect();
    let concat = gpu.upload_f32(&xs, &[n * 2 * dim]).unwrap();
    let tmp = concat.sub_offset(0, n * dim); // stands in for the normed tmp
    let rot = gpu.alloc_tensor(&[n * 2 * dim], DType::F32).unwrap();
    let cur = gpu.zeros(&[n * dim], DType::F32).unwrap();
    let qf = gpu.zeros(&[n * qd2], DType::F32).unwrap();
    let kk = gpu.zeros(&[n * kvd], DType::F32).unwrap();
    let vv = gpu.zeros(&[n * kvd], DType::F32).unwrap();
    let old = |gpu: &mut Gpu| {
        gpu.rotate_x_mq_batched(&concat, &rot, 2 * dim, n).unwrap();
        gpu.gemm_hfq4g256(&w_eh, &rot, &cur, dim, 2 * dim, n)
            .unwrap();
        for (w, y, m) in [(&w_q, &qf, qd2), (&w_k, &kk, kvd), (&w_v, &vv, kvd)] {
            gpu.rotate_x_mq_batched(&tmp, &rot, dim, n).unwrap();
            gpu.gemm_hfq4g256(w, &rot, y, m, dim, n).unwrap();
        }
    };
    let new = |gpu: &mut Gpu, with_q: bool| {
        gpu.rotate_x_mq_batched(&concat, &rot, 2 * dim, n).unwrap();
        gpu.claude_gemm_hfq4g256_multi(&[(&w_eh, &cur, dim)], &rot, 2 * dim, n)
            .unwrap();
        gpu.rotate_x_mq_batched(&tmp, &rot, dim, n).unwrap();
        if with_q {
            gpu.claude_gemm_hfq4g256_multi(
                &[(&w_q, &qf, qd2), (&w_k, &kk, kvd), (&w_v, &vv, kvd)],
                &rot,
                dim,
                n,
            )
            .unwrap();
        } else {
            gpu.claude_gemm_hfq4g256_multi(&[(&w_k, &kk, kvd), (&w_v, &vv, kvd)], &rot, dim, n)
                .unwrap();
        }
    };
    let grab = |gpu: &mut Gpu| {
        gpu.hip.device_synchronize().unwrap();
        (
            gpu.download_f32(&cur).unwrap(),
            gpu.download_f32(&kk).unwrap(),
            gpu.download_f32(&vv).unwrap(),
            gpu.download_f32(&qf).unwrap(),
        )
    };
    old(&mut gpu);
    let r_old = grab(&mut gpu);
    new(&mut gpu, false);
    let r_new = grab(&mut gpu);
    new(&mut gpu, true);
    let r_newq = grab(&mut gpu);
    for (name, a, b) in [
        ("cur(eh)", &r_old.0, &r_new.0),
        ("k", &r_old.1, &r_new.1),
        ("v", &r_old.2, &r_new.2),
        ("q_full (new+q)", &r_old.3, &r_newq.3),
        ("k (new+q)", &r_old.1, &r_newq.1),
    ] {
        let st = compare(a, b);
        println!(
            "fill N={n} {name:15} new vs old: max_abs={:.3e} rel_max={:.3e} rel_l2={:.3e} nonfinite={}",
            st.max_abs, st.rel_max, st.rel_l2, st.nonfinite
        );
    }
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() < 2.0 {
        new(&mut gpu, true);
        gpu.hip.device_synchronize().unwrap();
    }
    let mut ts = [Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..iters {
        for (i, ts_i) in ts.iter_mut().enumerate() {
            gpu.hip.event_record(&ev0, None).unwrap();
            match i {
                0 => old(&mut gpu),
                1 => new(&mut gpu, false),
                _ => new(&mut gpu, true),
            }
            gpu.hip.event_record(&ev1, None).unwrap();
            gpu.hip.event_synchronize(&ev1).unwrap();
            ts_i.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3);
        }
    }
    let names = [
        "old (scalar, 4 GEMMs)",
        "new (routed, skip-Q)",
        "new+q (routed, wq kept)",
    ];
    for (i, ts_i) in ts.iter_mut().enumerate() {
        let min = ts_i.iter().cloned().fold(f64::MAX, f64::min);
        println!(
            "fill N={n} {:24} {:9.1} us (min {:9.1})  clk={}",
            names[i],
            median(ts_i),
            min,
            clocks()
        );
    }
}

/// `--concat [--n 1024] [--iters 7]`: the MTP fill's `[e_norm | h_norm]`
/// concat build (mtp_head.rs step 3) at dim 5120:
///   legacy : 2n per-row `hipMemcpy` D2D (blocking; `MtpBatchedXfer::LEGACY`,
///            the prompt-fill transport)
///   ordered: 2n per-row stream-ordered D2D (`memcpy_dtod_at_auto`, the
///            `ordered: true` transport of the decode refresh)
///   strided: 2 `copy_rows_strided_f32` launches (the new code)
/// plus the `inpSA` residual copy (one n x dim D2D, blocking before).
/// Checks the concat bytes of every variant against legacy (random inputs)
/// and times each (host wall clock around the stage incl. a final sync,
/// median; blocking copies make host time the honest measure).
fn concat_bench(args: &[String]) {
    let arg = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].clone())
    };
    let n: usize = arg("--n").map(|v| v.parse().unwrap()).unwrap_or(1024);
    let iters: usize = arg("--iters").map(|v| v.parse().unwrap()).unwrap_or(7);
    let dim = 5120usize;
    let dim_bytes = dim * 4;
    let mut gpu = Gpu::init().expect("GPU init");
    let mut rng = Rng(0xc0ca_7000_1111_2222);
    let xs: Vec<f32> = (0..n * 2 * dim).map(|_| rng.gauss()).collect();
    let src = gpu.upload_f32(&xs, &[n * 2 * dim]).unwrap();
    let e_norm = src.sub_offset(0, n * dim);
    let h_norm = src.sub_offset(n * dim, n * dim);
    let concat = gpu.zeros(&[n * 2 * dim], DType::F32).unwrap();
    let residual = gpu.zeros(&[n * dim], DType::F32).unwrap();
    let legacy = |gpu: &mut Gpu| {
        for i in 0..n {
            gpu.hip
                .memcpy_dtod_at(
                    &concat.buf,
                    i * 2 * dim_bytes,
                    &src.buf,
                    i * dim_bytes,
                    dim_bytes,
                )
                .unwrap();
            gpu.hip
                .memcpy_dtod_at(
                    &concat.buf,
                    i * 2 * dim_bytes + dim_bytes,
                    &src.buf,
                    (n + i) * dim_bytes,
                    dim_bytes,
                )
                .unwrap();
        }
        gpu.hip
            .memcpy_dtod_at(&residual.buf, 0, &src.buf, 0, n * dim_bytes)
            .unwrap();
    };
    let ordered = |gpu: &mut Gpu| {
        for i in 0..n {
            gpu.memcpy_dtod_at_auto(
                &concat.buf,
                i * 2 * dim_bytes,
                &src.buf,
                i * dim_bytes,
                dim_bytes,
            )
            .unwrap();
            gpu.memcpy_dtod_at_auto(
                &concat.buf,
                i * 2 * dim_bytes + dim_bytes,
                &src.buf,
                (n + i) * dim_bytes,
                dim_bytes,
            )
            .unwrap();
        }
        gpu.memcpy_dtod_at_auto(&residual.buf, 0, &src.buf, 0, n * dim_bytes)
            .unwrap();
    };
    let strided = |gpu: &mut Gpu| {
        let cv = concat.sub_offset(0, n * 2 * dim);
        gpu.copy_rows_strided_f32(&e_norm, &cv, n, dim, dim, 2 * dim, 0)
            .unwrap();
        gpu.copy_rows_strided_f32(&h_norm, &cv, n, dim, dim, 2 * dim, dim)
            .unwrap();
        gpu.memcpy_dtod_at_auto(&residual.buf, 0, &src.buf, 0, n * dim_bytes)
            .unwrap();
    };
    let variants: [(&str, &dyn Fn(&mut Gpu)); 3] = [
        ("legacy", &legacy),
        ("ordered", &ordered),
        ("strided", &strided),
    ];
    let mut reference: Option<(Vec<u8>, Vec<u8>)> = None;
    for (name, f) in variants.iter() {
        gpu.hip
            .memset(&concat.buf, 0x7f, n * 2 * dim_bytes)
            .unwrap();
        gpu.hip.memset(&residual.buf, 0x7f, n * dim_bytes).unwrap();
        gpu.hip.device_synchronize().unwrap();
        f(&mut gpu);
        gpu.hip.device_synchronize().unwrap();
        let mut c = vec![0u8; n * 2 * dim_bytes];
        let mut r = vec![0u8; n * dim_bytes];
        gpu.hip.memcpy_dtoh(&mut c, &concat.buf).unwrap();
        gpu.hip.memcpy_dtoh(&mut r, &residual.buf).unwrap();
        match reference.as_ref() {
            None => {
                // Legacy must equal the host-built concat.
                let mut want = vec![0u8; n * 2 * dim_bytes];
                let xb =
                    unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) };
                for i in 0..n {
                    want[i * 2 * dim_bytes..i * 2 * dim_bytes + dim_bytes]
                        .copy_from_slice(&xb[i * dim_bytes..(i + 1) * dim_bytes]);
                    want[i * 2 * dim_bytes + dim_bytes..(i + 1) * 2 * dim_bytes]
                        .copy_from_slice(&xb[(n + i) * dim_bytes..(n + i + 1) * dim_bytes]);
                }
                println!("concat n={n} legacy == host-built concat: {}", c == want);
                reference = Some((c, r));
            }
            Some((rc, rr)) => {
                println!(
                    "concat n={n} {name:8} bytes identical to legacy: concat {} residual {}",
                    &c == rc,
                    &r == rr
                );
            }
        }
    }
    for (name, f) in variants.iter() {
        f(&mut gpu);
        gpu.hip.device_synchronize().unwrap();
        let mut ts = Vec::new();
        for _ in 0..iters {
            gpu.hip.device_synchronize().unwrap();
            let t = std::time::Instant::now();
            f(&mut gpu);
            gpu.hip.device_synchronize().unwrap();
            ts.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let min = ts.iter().cloned().fold(f64::MAX, f64::min);
        println!(
            "concat n={n} {name:8} {:9.1} us (min {:9.1}) wall incl. residual copy",
            median(&mut ts),
            min
        );
    }
}
