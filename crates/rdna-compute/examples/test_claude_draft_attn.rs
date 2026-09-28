// claude v3: parity + timing of the MTP draft-head decode route
// (`Gpu::claude_draft_attn4` -> WMMA split-K `claude_attn4` + 8-warp reduce,
// rows = 1) against the stock single-row Q8 flash decode that the dispatch
// family runs otherwise (`Gpu::attention_flash_q8_0`: tile + reduce), on the
// Qwen3.8-27B attention shape (24 q heads / 4 kv heads / head_dim 256).
//
// Both paths get the same inputs the draft head passes: a 4-byte i32 `pos_buf`
// holding pos = seq - 1, `seq_len_hint` = pos + 1, `max_seq` = KV capacity,
// and ONE shared partials scratch sized exactly like
// `Qwen35MtpHeadScratch::flash_partials` (n_heads * ceil(max_seq / tile) *
// (2 + head_dim), tile = `q8_flash_tile_size`). KV is allocated for the
// largest `max_seq` only (default 49152 positions, ~107 MB) so it can run
// beside a loaded model.
//
//   cargo build --release -p rdna-compute --example test_claude_draft_attn
//   test_claude_draft_attn [--seqs 9000,42000] [--iters 50] [--extra] [--check]
//
// Parity is checked for q scaled x1 (the bench_claude_attn3 data: near-flat
// softmax) and x8 (sharper scores); rel_max = max|stock - new| / max|stock|.
//
// `--extra` adds edge cases (seq 1 / 100, tile-32 capacity, a partials buffer
// too small for the default span). `--check` exits 1 unless every case is
// routed and rel_max < 3e-3 with no non-finite output.
use rdna_compute::{DType, Gpu, GpuTensor};

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;

struct Stats {
    rel_max: f32,
    rel_l2: f64,
    max_abs: f32,
    nonfinite: usize,
}

fn compare(stock: &[f32], new: &[f32]) -> Stats {
    let scale = stock.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
    let mut max_abs = 0.0f32;
    let mut sq_d = 0.0f64;
    let mut sq_r = 0.0f64;
    let mut nonfinite = 0usize;
    for (x, y) in stock.iter().zip(new) {
        if !y.is_finite() || !x.is_finite() {
            nonfinite += 1;
        }
        max_abs = max_abs.max((x - y).abs());
        sq_d += ((x - y) as f64).powi(2);
        sq_r += (*x as f64).powi(2);
    }
    Stats {
        rel_max: max_abs / scale,
        rel_l2: (sq_d / sq_r.max(1e-30)).sqrt(),
        max_abs,
        nonfinite,
    }
}

/// Span `claude_attn4` will use for this (seq, partials) pair (mirrors its fit loop).
fn a4_span(gpu: &Gpu, seq: usize, partials: &GpuTensor) -> Option<usize> {
    let cap_tiles = partials.numel() / (N_HEADS * (2 + HEAD_DIM));
    let mut span = gpu.claude_attn3_span(seq, N_KV_HEADS);
    while seq.div_ceil(span) > cap_tiles {
        if span >= 4096 {
            return None;
        }
        span += 32;
    }
    Some(span)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let seqs: Vec<usize> = argval("--seqs")
        .unwrap_or_else(|| "9000,42000".into())
        .split(',')
        .map(|s| s.trim().parse().expect("--seqs"))
        .collect();
    let iters: usize = argval("--iters").map(|v| v.parse().unwrap()).unwrap_or(50);
    let extra = args.iter().any(|a| a == "--extra");
    let check = args.iter().any(|a| a == "--check");

    let big = seqs.iter().copied().max().unwrap_or(1).max(8193);
    let kv_cap = big.div_ceil(8192) * 8192;
    // (seq, max_seq used for tile choice + partials sizing)
    let mut cases: Vec<(usize, usize)> = seqs.iter().map(|&s| (s, kv_cap)).collect();
    if extra {
        cases.push((1, kv_cap));
        cases.push((100, kv_cap));
        cases.push((5000, 8192)); // tile-32 stock path, partials 256 tiles/head
        cases.push((9000, 9216)); // tile-128 partials hold 72 tiles/head < default span need
    }

    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}  kv_cap={kv_cap}  iters={iters}", gpu.arch);

    // Synthetic q8_0 KV cache for kv_cap positions (written in 8192-row chunks).
    let q_host = lcg(0xa5a5, q_dim);
    let d_q = gpu.upload_f32(&q_host, &[q_dim]).unwrap();
    let kv_bytes_total = kv_cap * N_KV_HEADS * (HEAD_DIM / 32) * 34;
    let d_k = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let d_v = gpu.alloc_tensor(&[kv_bytes_total], DType::Q8_0).unwrap();
    let chunk = 8192usize;
    let d_all_pos = gpu.alloc_tensor(&[kv_cap], DType::F32).unwrap();
    let all_pos: Vec<u8> = (0..kv_cap as i32).flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&d_all_pos.buf, &all_pos).unwrap();
    let mut written = 0usize;
    let mut seed = 0xc3c3u32;
    while written < kv_cap {
        let d_kf = gpu.upload_f32(&lcg(seed, chunk * kv_dim), &[chunk * kv_dim]).unwrap();
        let d_vf = gpu.upload_f32(&lcg(seed ^ 0x9696, chunk * kv_dim), &[chunk * kv_dim]).unwrap();
        seed = seed.wrapping_add(7919);
        let pos_view = d_all_pos.sub_offset(written, chunk);
        gpu.kv_cache_write_q8_0_batched(&d_k, &d_kf, &pos_view, N_KV_HEADS, HEAD_DIM, chunk).unwrap();
        gpu.kv_cache_write_q8_0_batched(&d_v, &d_vf, &pos_view, N_KV_HEADS, HEAD_DIM, chunk).unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(d_kf).unwrap();
        gpu.free_tensor(d_vf).unwrap();
        written += chunk;
    }
    gpu.free_tensor(d_all_pos).unwrap();

    let d_pos = gpu.alloc_tensor(&[1], DType::F32).unwrap(); // i32 pos_buf
    let d_out_stock = gpu.zeros(&[q_dim], DType::F32).unwrap();
    let d_out_new = gpu.zeros(&[q_dim], DType::F32).unwrap();

    let mut failures = 0usize;
    for &(seq, max_seq) in &cases {
        let tile = rdna_compute::attention::q8_flash_tile_size(&gpu.arch, N_HEADS, N_KV_HEADS, HEAD_DIM, max_seq);
        let max_tiles = max_seq.div_ceil(tile);
        let d_part = gpu.zeros(&[N_HEADS * max_tiles * (2 + HEAD_DIM)], DType::F32).unwrap();
        let pos = (seq - 1) as i32;
        gpu.hip.memcpy_htod(&d_pos.buf, &pos.to_ne_bytes()).unwrap();

        let run_stock = |gpu: &mut Gpu| {
            gpu.attention_flash_q8_0(
                &d_q, &d_k, &d_v, &d_out_stock, &d_pos.buf, seq, N_HEADS, N_KV_HEADS, HEAD_DIM, max_seq, &d_part,
            )
            .unwrap();
            true
        };
        let run_new = |gpu: &mut Gpu| {
            gpu.claude_draft_attn4(
                &d_q, &d_k, &d_v, &d_out_new, &d_pos.buf, seq, N_HEADS, N_KV_HEADS, HEAD_DIM, max_seq, &d_part,
            )
            .unwrap()
        };

        // The route must decline under hipGraph capture / redline recording
        // (checked host-side: nothing is launched when it declines).
        let saved = gpu.graphs.capture_mode;
        gpu.graphs.capture_mode = true;
        let under_capture = run_new(&mut gpu);
        gpu.graphs.capture_mode = saved;
        if under_capture {
            eprintln!("FAIL: route did not decline under capture_mode");
            failures += 1;
        }

        // Parity at q x8 first (sharper softmax), then q x1 (left in place
        // for the timing + stability check below).
        let mut routed = true;
        let mut st8 = None;
        let mut st = None;
        let mut new = Vec::new();
        for qs in [8.0f32, 1.0] {
            let qv: Vec<u8> = q_host.iter().flat_map(|x| (x * qs).to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&d_q.buf, &qv).unwrap();
            run_stock(&mut gpu);
            gpu.hip.device_synchronize().unwrap();
            let stock = gpu.download_f32(&d_out_stock).unwrap();
            // Poison the new output first so a silent no-op cannot pass.
            gpu.hip.memcpy_htod(&d_out_new.buf, &vec![0xffu8; q_dim * 4]).unwrap();
            routed &= run_new(&mut gpu);
            gpu.hip.device_synchronize().unwrap();
            new = gpu.download_f32(&d_out_new).unwrap();
            let c = compare(&stock, &new);
            if qs == 8.0 {
                st8 = Some(c);
            } else {
                st = Some(c);
            }
        }
        if !routed {
            eprintln!("seq={seq:6} max_seq={max_seq:6}: claude_draft_attn4 DECLINED (arch/env/shape)");
            failures += 1;
            gpu.free_tensor(d_part).unwrap();
            continue;
        }
        let (st, st8) = (st.unwrap(), st8.unwrap());

        // Timing: ~0.3 s clock warmup, then `iters` back-to-back calls each.
        let time = |gpu: &mut Gpu, f: &dyn Fn(&mut Gpu) -> bool| -> f64 {
            gpu.hip.device_synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f(gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        {
            let t = std::time::Instant::now();
            while t.elapsed().as_secs_f64() < 0.3 {
                for _ in 0..20 {
                    run_stock(&mut gpu);
                }
                gpu.hip.device_synchronize().unwrap();
            }
        }
        let us_stock = time(&mut gpu, &run_stock);
        let us_new = time(&mut gpu, &run_new);
        let us_stock2 = time(&mut gpu, &run_stock);
        let us_new2 = time(&mut gpu, &run_new);
        let us_s = us_stock.min(us_stock2);
        let us_n = us_new.min(us_new2);
        // Outputs after the timing loops must still match the first run.
        gpu.hip.device_synchronize().unwrap();
        let new_after = gpu.download_f32(&d_out_new).unwrap();
        let stable = new_after.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits());

        let ok = st.rel_max < 3e-3
            && st8.rel_max < 3e-3
            && st.nonfinite == 0
            && st8.nonfinite == 0
            && stable;
        if !ok {
            failures += 1;
        }
        eprintln!(
            "seq={seq:6} max_seq={max_seq:6} stock_tile={tile:3} a4_span={:4} | q x1: rel_max={:.3e} rel_l2={:.3e} max_abs={:.3e} | q x8: rel_max={:.3e} rel_l2={:.3e} max_abs={:.3e} | nonfinite={} stable={} | stock {us_s:7.1} us  claude_attn4 {us_n:7.1} us  speedup {:.2}x  {}",
            a4_span(&gpu, seq, &d_part).map(|s| s as i64).unwrap_or(-1),
            st.rel_max,
            st.rel_l2,
            st.max_abs,
            st8.rel_max,
            st8.rel_l2,
            st8.max_abs,
            st.nonfinite + st8.nonfinite,
            stable,
            us_s / us_n,
            if ok { "OK" } else { "FAIL" }
        );
        gpu.free_tensor(d_part).unwrap();
    }

    gpu.free_tensor(d_q).unwrap();
    gpu.free_tensor(d_k).unwrap();
    gpu.free_tensor(d_v).unwrap();
    gpu.free_tensor(d_pos).unwrap();
    gpu.free_tensor(d_out_stock).unwrap();
    gpu.free_tensor(d_out_new).unwrap();
    if failures == 0 {
        eprintln!("PASS ({} cases)", cases.len());
    } else {
        eprintln!("FAIL ({failures} of {} cases)", cases.len());
        if check {
            std::process::exit(1);
        }
    }
}
