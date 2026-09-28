// claude v3: standalone check/bench of `claude_attn3` against the rows kernel.
// Qwen3.8-27B verify shape (24 heads, hd 256, Q8 KV, 4 rows). Allocates only
// `--seq` worth of KV (rounded to 8192) so it can run beside a loaded model.
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
    let seq_len = argval("--seq", 42000);
    let batch = argval("--rows", 4);
    let iters = argval("--iters", 30);
    let warmup = argval("--warmup", 150);
    let n_heads = 24usize;
    let n_kv_heads = 4usize;
    let head_dim = 256usize;
    let check = args.iter().any(|a| a == "--check");

    let max_seq = seq_len.div_ceil(8192) * 8192;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  seq={seq_len} rows={batch} (max_seq {max_seq})", gpu.arch);

    let d_q = gpu.upload_f32(&lcg(0xa5a5, batch * q_dim), &[batch * q_dim]).unwrap();
    let kv_bytes_total = max_seq * n_kv_heads * (head_dim / 32) * 34;
    let d_k = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let d_v = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let chunk = 8192usize;
    let d_all_pos = gpu.alloc_tensor(&[max_seq], DType::F32).unwrap();
    let all_pos: Vec<u8> = (0..max_seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_all_pos.buf, &all_pos).unwrap();
    let mut written = 0usize;
    let mut seed = 0xc3c3u32;
    while written < max_seq {
        let d_kf = gpu.upload_f32(&lcg(seed, chunk * kv_dim), &[chunk * kv_dim]).unwrap();
        let d_vf = gpu.upload_f32(&lcg(seed ^ 0x9696, chunk * kv_dim), &[chunk * kv_dim]).unwrap();
        seed = seed.wrapping_add(7919);
        let pos_view = d_all_pos.sub_offset(written, chunk);
        gpu.kv_cache_write_q8_0_batched(&d_k, &d_kf, &pos_view, n_kv_heads, head_dim, chunk).unwrap();
        gpu.kv_cache_write_q8_0_batched(&d_v, &d_vf, &pos_view, n_kv_heads, head_dim, chunk).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(d_kf).unwrap();
        gpu.free_tensor(d_vf).unwrap();
        written += chunk;
    }

    let d_out_rows = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();
    let d_out_a3 = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();
    let d_out_a4 = gpu.zeros(&[batch * q_dim], DType::F32).unwrap();
    let d_part = gpu.zeros(&[4 * n_heads * (max_seq / 64) * (2 + head_dim)], DType::F32).unwrap();
    let positions: Vec<u8> = (0..batch).flat_map(|i| ((seq_len - batch + i) as i32).to_ne_bytes()).collect();
    let d_pos = gpu.alloc_tensor(&[batch], DType::F32).unwrap();
    gpu.hip.memcpy_htod(&d_pos.buf, &positions).unwrap();
    let kv_bytes = (seq_len * n_kv_heads * (head_dim / 32) * 34 * 2) as f64;

    let run_rows = |gpu: &mut Gpu| {
        gpu.attention_flash_q8_0_rows_masked(
            &d_q, &d_k, &d_v, &d_out_rows, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, batch, &d_part,
        )
        .unwrap()
    };
    let run_a3 = |gpu: &mut Gpu| {
        gpu.claude_attn3(
            &d_q, &d_k, &d_v, &d_out_a3, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, batch, &d_part,
        )
        .unwrap()
    };
    let run_a4 = |gpu: &mut Gpu| {
        gpu.claude_attn4(
            &d_q, &d_k, &d_v, &d_out_a4, &d_pos, n_heads, n_kv_heads, head_dim, seq_len, batch, &d_part,
        )
        .unwrap()
    };
    let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu) -> bool| -> f64 {
        gpu.hip.device_synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    if !run_rows(&mut gpu) {
        eprintln!("rows kernel refused this shape");
        std::process::exit(1);
    }
    if !run_a3(&mut gpu) {
        eprintln!("claude_attn3 refused this shape");
        std::process::exit(1);
    }
    // Clock warmup: keep the GPU busy for ~1 s before timing.
    {
        let t = std::time::Instant::now();
        while t.elapsed().as_secs_f64() < 1.0 {
            for _ in 0..warmup.max(1) {
                run_rows(&mut gpu);
            }
            gpu.hip.device_synchronize().unwrap();
        }
    }
    gpu.hip.device_synchronize().unwrap();
    let us_r = time(&mut gpu, &run_rows);
    let us_a = time(&mut gpu, &run_a3);
    eprintln!("rows kernel : {us_r:8.1} us/call  {:.0} GB/s of KV", kv_bytes / (us_r * 1e3));
    eprintln!("claude_attn3: {us_a:8.1} us/call  {:.0} GB/s of KV  (span {})", kv_bytes / (us_a * 1e3), gpu.claude_attn3_span(seq_len, n_kv_heads));
    eprintln!("speedup: {:.2}x", us_r / us_a);
    let has_a4 = run_a4(&mut gpu);
    if has_a4 {
        let us_4 = time(&mut gpu, &run_a4);
        eprintln!("claude_attn4: {us_4:8.1} us/call  {:.0} GB/s of KV  speedup {:.2}x", kv_bytes / (us_4 * 1e3), us_r / us_4);
        let a = gpu.download_f32(&d_out_rows).unwrap();
        let b = gpu.download_f32(&d_out_a4).unwrap();
        let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
        let mut max_abs = 0.0f32;
        let mut sq_d = 0.0f64;
        let mut sq_r = 0.0f64;
        let mut nonfinite = 0usize;
        for (x, y) in a.iter().zip(&b) {
            if !y.is_finite() {
                nonfinite += 1;
            }
            max_abs = max_abs.max((x - y).abs());
            sq_d += ((x - y) as f64).powi(2);
            sq_r += (*x as f64).powi(2);
        }
        eprintln!(
            "parity attn4 vs rows: max_abs={max_abs:.3e} rel_max={:.3e} rel_l2={:.3e} nonfinite={nonfinite}",
            max_abs / scale,
            (sq_d / sq_r.max(1e-30)).sqrt()
        );
        if check && !(max_abs / scale < 3e-3 && nonfinite == 0) {
            eprintln!("FAIL attn4");
            std::process::exit(1);
        }
    } else {
        eprintln!("claude_attn4 refused this shape");
    }
    let a = gpu.download_f32(&d_out_rows).unwrap();
    let b = gpu.download_f32(&d_out_a3).unwrap();
    let scale = a.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
    let mut max_abs = 0.0f32;
    let mut sq_d = 0.0f64;
    let mut sq_r = 0.0f64;
    let mut nonfinite = 0usize;
    for (x, y) in a.iter().zip(&b) {
        if !y.is_finite() {
            nonfinite += 1;
        }
        max_abs = max_abs.max((x - y).abs());
        sq_d += ((x - y) as f64).powi(2);
        sq_r += (*x as f64).powi(2);
    }
    let rel = max_abs / scale;
    eprintln!(
        "parity attn3 vs rows: max_abs={max_abs:.3e} rel_max={rel:.3e} rel_l2={:.3e} nonfinite={nonfinite}",
        (sq_d / sq_r.max(1e-30)).sqrt()
    );
    if check && !(rel < 2e-3 && nonfinite == 0) {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    eprintln!("OK");
}
