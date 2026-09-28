// claude v3: FA2 (GQA-fused) vs incumbent WMMA prefill attention at long context.
// Checks bit-exactness and times both for one 512/1024-row chunk at the end of a
// `--ctx` context (Qwen3.8-27B shape: 24 heads, 4 kv heads, hd 256, q8 KV).
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: usize| {
        args.iter().position(|a| a == k).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d)
    };
    let ctx = argval("--ctx", 100_000);
    let batch = argval("--batch", 1024);
    let iters = argval("--iters", 5);
    let (n_heads, n_kv_heads, head_dim) = (24usize, 4usize, 256usize);
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let start = ctx - batch;
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  ctx={ctx} batch={batch} start={start}", gpu.arch);

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

    let run_fa2 = |gpu: &mut Gpu| {
        gpu.attention_q8_0_fa2_gqa_gfx11(
            &d_q, &d_k, &d_v, &d_out_fa2, &d_pos, n_heads, n_kv_heads, head_dim, ctx, batch,
        )
        .unwrap();
    };
    let run_inc = |gpu: &mut Gpu| {
        gpu.attention_q8_0_flash_prefill_wmma_slots(
            &d_q, &d_k, &d_v, &d_out_inc, &d_pos, n_heads, n_kv_heads, head_dim, ctx, batch, None, None, None, None,
        )
        .unwrap();
    };
    run_fa2(&mut gpu);
    run_inc(&mut gpu);
    gpu.hip.device_synchronize().unwrap();
    let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu)| -> f64 {
        gpu.hip.device_synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    };
    let ms_fa2 = time(&mut gpu, &run_fa2);
    let ms_inc = time(&mut gpu, &run_inc);
    let pairs = (0..batch).map(|b| (start + b + 1) as f64).sum::<f64>();
    let flop = pairs * (n_heads * head_dim * 4) as f64;
    eprintln!("incumbent wmma: {ms_inc:9.2} ms/chunk  {:.1} TFLOPS", flop / (ms_inc * 1e9));
    eprintln!("fa2 gqa gfx11 : {ms_fa2:9.2} ms/chunk  {:.1} TFLOPS  speedup {:.2}x", flop / (ms_fa2 * 1e9), ms_inc / ms_fa2);
    let a = gpu.download_f32(&d_out_inc).unwrap();
    let b = gpu.download_f32(&d_out_fa2).unwrap();
    let mut unequal = 0usize;
    let mut max_abs = 0.0f32;
    let mut nonfinite = 0usize;
    for (x, y) in a.iter().zip(&b) {
        if x.to_bits() != y.to_bits() {
            unequal += 1;
        }
        if !y.is_finite() || !x.is_finite() {
            nonfinite += 1;
        }
        max_abs = max_abs.max((x - y).abs());
    }
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
    eprintln!(
        "parity fa2 vs incumbent: unequal_words={unequal}/{} max_abs={max_abs:.3e} rel={:.3e} nonfinite={nonfinite}",
        a.len(),
        max_abs / scale
    );
    if unequal == 0 && nonfinite == 0 {
        eprintln!("BIT-EXACT");
    } else if max_abs / scale < 1e-4 && nonfinite == 0 {
        eprintln!("CLOSE (not bit-exact)");
    } else {
        eprintln!("FAIL");
        std::process::exit(1);
    }
}
