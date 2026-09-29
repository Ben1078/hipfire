// claude v3: FA2 (GQA-fused) vs incumbent WMMA prefill attention at long context.
// Times one 512/1024-row chunk at the end of a `--ctx` context (Qwen3.8-27B
// shape: 24 heads, 4 kv heads, hd 256, q8 KV) and checks parity.
//
// Flags:
//   --ctx N --batch N       context length / chunk rows (chunk sits at the end)
//   --iters N --warm N      timed iterations (median reported) / warmup launches
//   --variants 0,3,...      FA2 body variants to time (0 = upstream kernel;
//                           see `claude_fa2_variant` in attention.rs; `auto` =
//                           the production entry, i.e. CLAUDE_FA2_VARIANT or
//                           the arch default)
//   --inc-time              also time the incumbent (always run once for parity)
//   --no-inc                skip the incumbent entirely (profiling runs)
//   --no-check              skip all parity work (profiling runs)
use rdna_compute::{DType, Gpu};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn parity(a: &[f32], b: &[f32]) -> (usize, f32, f32, usize) {
    let mut unequal = 0usize;
    let mut max_abs = 0.0f32;
    let mut nonfinite = 0usize;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            unequal += 1;
        }
        if !y.is_finite() || !x.is_finite() {
            nonfinite += 1;
        }
        max_abs = max_abs.max((x - y).abs());
    }
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
    (unequal, max_abs, max_abs / scale, nonfinite)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: usize| {
        args.iter().position(|a| a == k).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d)
    };
    let flag = |k: &str| args.iter().any(|a| a == k);
    let ctx = argval("--ctx", 100_000);
    let batch = argval("--batch", 1024);
    let iters = argval("--iters", 10);
    let warm = argval("--warm", 3);
    let variants: Vec<u32> = args
        .iter()
        .position(|a| a == "--variants")
        .map(|i| {
            args[i + 1]
                .split(',')
                .map(|v| if v == "auto" { u32::MAX } else { v.parse().unwrap() })
                .collect()
        })
        .unwrap_or_else(|| vec![0]);
    let no_inc = flag("--no-inc");
    let inc_time = flag("--inc-time");
    let no_check = flag("--no-check");
    let (n_heads, n_kv_heads, head_dim) = (24usize, 4usize, 256usize);
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let start = ctx - batch;
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  ctx={ctx} batch={batch} start={start} variants={variants:?}", gpu.arch);

    let kv_bytes_total = ctx * n_kv_heads * (head_dim / 32) * 34;
    let d_k = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let d_v = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let d_all_pos = gpu.alloc_tensor(&[ctx], DType::F32).unwrap();
    let all_pos: Vec<u8> = (0..ctx as i32).flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_all_pos.buf, &all_pos).unwrap();
    let chunk = 8192usize;
    let mut written = 0usize;
    let mut seed = 0xc3c3u32;
    while written < ctx {
        let n = (ctx - written).min(chunk);
        let d_kf = gpu.upload_f32(&lcg(seed, n * kv_dim), &[n * kv_dim]).unwrap();
        let d_vf = gpu.upload_f32(&lcg(seed ^ 0x9696, n * kv_dim), &[n * kv_dim]).unwrap();
        seed = seed.wrapping_add(7919);
        let pos_view = d_all_pos.sub_offset(written, n);
        gpu.kv_cache_write_q8_0_batched(&d_k, &d_kf, &pos_view, n_kv_heads, head_dim, n).unwrap();
        gpu.kv_cache_write_q8_0_batched(&d_v, &d_vf, &pos_view, n_kv_heads, head_dim, n).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(d_kf).unwrap();
        gpu.free_tensor(d_vf).unwrap();
        written += n;
    }
    let d_q = gpu.upload_f32(&lcg(0xa5a5, batch * q_dim), &[batch * q_dim]).unwrap();
    let d_pos = d_all_pos.sub_offset(start, batch);
    let d_out_fa2 = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();
    let d_out_inc = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();

    let pairs = (0..batch).map(|b| (start + b + 1) as f64).sum::<f64>();
    let flop = pairs * (n_heads * head_dim * 4) as f64;

    // Median-of-iters wall time per launch (device-synchronized per launch).
    let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu)| -> (f64, f64) {
        for _ in 0..warm {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let mut v = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            f(gpu);
            gpu.hip.device_synchronize().unwrap();
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0])
    };

    let run_inc = |gpu: &mut Gpu| {
        gpu.attention_q8_0_flash_prefill_wmma_slots(
            &d_q, &d_k, &d_v, &d_out_inc, &d_pos, n_heads, n_kv_heads, head_dim, ctx, batch, None, None, None, None,
        )
        .unwrap();
    };
    let mut inc_out: Option<Vec<f32>> = None;
    if !no_inc {
        run_inc(&mut gpu);
        gpu.hip.device_synchronize().unwrap();
        if !no_check {
            inc_out = Some(gpu.download_f32(&d_out_inc).unwrap());
        }
        if inc_time {
            let (ms, mn) = time(&mut gpu, &run_inc);
            eprintln!("incumbent wmma: {ms:9.3} ms/chunk (min {mn:.3})  {:.1} TFLOPS", flop / (ms * 1e9));
        }
    }

    let mut v0_out: Option<Vec<f32>> = None;
    let mut fail = false;
    for &v in &variants {
        let run_fa2 = |gpu: &mut Gpu| {
            if v == u32::MAX {
                gpu.attention_q8_0_fa2_gqa_gfx11(
                    &d_q, &d_k, &d_v, &d_out_fa2, &d_pos, n_heads, n_kv_heads, head_dim, ctx, batch,
                )
                .unwrap();
            } else {
                gpu.attention_q8_0_fa2_gqa_gfx11_variant(
                    &d_q, &d_k, &d_v, &d_out_fa2, &d_pos, n_heads, n_kv_heads, head_dim, ctx, batch, v,
                )
                .unwrap();
            }
        };
        gpu.hip.memset(&d_out_fa2.buf, 0, batch * q_dim * 4).ok();
        run_fa2(&mut gpu);
        gpu.hip.device_synchronize().unwrap();
        let out = if no_check { None } else { Some(gpu.download_f32(&d_out_fa2).unwrap()) };
        let (ms, mn) = time(&mut gpu, &run_fa2);
        let tag = if v == u32::MAX { "auto".to_string() } else { format!("v{v}") };
        eprintln!(
            "fa2 {tag:<4}: {ms:9.3} ms/chunk (min {mn:.3})  {:.1} TFLOPS",
            flop / (ms * 1e9)
        );
        if let Some(out) = out {
            if let Some(inc) = &inc_out {
                let (uneq, max_abs, rel, nonfinite) = parity(inc, &out);
                let verdict = if nonfinite == 0 && rel < 5e-4 { "OK" } else { "FAIL" };
                if verdict == "FAIL" {
                    fail = true;
                }
                eprintln!(
                    "   parity {tag} vs incumbent: unequal={uneq}/{} max_abs={max_abs:.3e} rel={rel:.3e} nonfinite={nonfinite} {verdict}",
                    out.len()
                );
            }
            if let Some(r0) = &v0_out {
                let (uneq, max_abs, rel, nonfinite) = parity(r0, &out);
                eprintln!(
                    "   parity {tag} vs v0: unequal={uneq}/{} max_abs={max_abs:.3e} rel={rel:.3e} nonfinite={nonfinite}{}",
                    out.len(),
                    if uneq == 0 { " BIT-EXACT" } else { "" }
                );
            } else if v == 0 {
                v0_out = Some(out);
            }
        }
    }
    if fail {
        eprintln!("FAIL");
        std::process::exit(1);
    }
}
