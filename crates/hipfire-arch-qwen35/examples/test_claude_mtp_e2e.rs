//! claude r2: end-to-end MTP decode on a real model file through the runtime's
//! own drafter (`Qwen35MtpDrafter`: prefill + `spec_step_mtp_compressed_serial_with_k`
//! windows), for A/B runs of the r2 switches in separate processes.
//!
//!   test_claude_mtp_e2e <trunk.hfq> <head.mtp> [--temp T] [--top-p P] [--top-k K]
//!       [--seed S] [--cycles N] [--prompt-len L] [--k K] [--ctx C]
//!
//! Prints the generated token stream (fnv + head), per-window draft/accept
//! counts, and fnv checksums of the trunk KV cache, the DeltaNet state and the
//! MTP head KV + prev_hidden. Two runs with different CLAUDE_* switches must
//! print identical lines when the switched code is exact (greedy), and the
//! same token stream when it only changes float order (sampled, typically).
//! Built for the tiny synthetic model from `make_tiny_qwen35.py` (<200 MB).
use hipfire_arch_qwen35::mtp_head;
use hipfire_arch_qwen35::mtp_speculator::Qwen35MtpDrafter;
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use rdna_compute::Gpu;
use std::path::Path;
use std::time::Instant;

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn tbytes(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<u8> {
    let n = t.byte_size().min(t.buf.size());
    let mut out = vec![0u8; n];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh");
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let model = args.get(1).expect("trunk path").clone();
    let head_p = args.get(2).expect("head path").clone();
    let temp: f32 = arg("--temp").map(|v| v.parse().unwrap()).unwrap_or(0.0);
    let top_p: f32 = arg("--top-p").map(|v| v.parse().unwrap()).unwrap_or(1.0);
    let top_k: usize = arg("--top-k").map(|v| v.parse().unwrap()).unwrap_or(0);
    let seed: u64 = arg("--seed").map(|v| v.parse().unwrap()).unwrap_or(0x1357_9BDF);
    let cycles: usize = arg("--cycles").map(|v| v.parse().unwrap()).unwrap_or(60);
    let plen: usize = arg("--prompt-len").map(|v| v.parse().unwrap()).unwrap_or(40);
    let k: usize = arg("--k").map(|v| v.parse().unwrap()).unwrap_or(3);
    let ctx: usize = arg("--ctx").map(|v| v.parse().unwrap()).unwrap_or(512);

    let mut gpu = Gpu::init().expect("gpu");
    let mut cfg = ModelSlotConfig::default();
    cfg.max_seq = ctx;
    let mut slot = ModelSlot::load(&mut gpu, Path::new(&model), "e2e", cfg).expect("load trunk");
    let head = mtp_head::load_mtp_head(Path::new(&head_p), &mut gpu, ctx).expect("load head");
    let cvs = head.weights.compressed_vocab_size;
    let mut drafter = Qwen35MtpDrafter::new(head, k, ctx);
    drafter.configure_request(SpecRequestConfig {
        temp,
        top_p,
        top_k,
        min_p: 0.0,
        cactus_delta: 0.0,
        rng_seed: seed,
        allow_ngram_modifier: false,
    });
    let vocab = slot.config.vocab_size as u64;
    let eos = slot.config.eos_token;
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15 ^ plen as u64;
    let prompt: Vec<u32> = (0..plen)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((x >> 33) % (vocab - 1000)) as u32;
            if t == eos { t + 1 } else { t }
        })
        .collect();
    let no_abort = || false;
    let mut seed_tok = drafter
        .mtp_prefill(&mut gpu, &mut slot, &prompt, &prompt, 0, false, &no_abort)
        .expect("prefill");
    let mut history = prompt.clone();
    history.push(seed_tok);
    let mut pos = prompt.len();
    let (mut drafts, mut accepted, mut windows) = (0usize, 0usize, 0usize);
    let mut shape = Vec::new();
    gpu.hip.device_synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..cycles {
        if pos + k + 2 >= ctx {
            break;
        }
        let w = drafter
            .mtp_step(&mut gpu, &mut slot, pos, seed_tok, &history, k, eos, None)
            .expect("mtp step");
        history.extend_from_slice(&w.committed);
        pos += w.committed.len();
        seed_tok = *history.last().unwrap();
        drafts += w.drafts_generated;
        accepted += w.accepted;
        windows += 1;
        shape.push(format!("{}/{}", w.accepted, w.drafts_generated));
        if w.committed.contains(&eos) {
            break;
        }
    }
    gpu.hip.device_synchronize().unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let gen = &history[prompt.len()..];
    let gen_b: Vec<u8> = gen.iter().flat_map(|t| t.to_le_bytes()).collect();
    println!(
        "e2e: temp={temp} top_p={top_p} top_k={top_k} seed={seed:#x} k={k} head={} draft_vocab={:?}",
        head_p, cvs
    );
    println!(
        "tokens: n={} fnv={:016x} head={:?}",
        gen.len(),
        fnv(0xcbf29ce484222325, &gen_b),
        &gen[..gen.len().min(24)]
    );
    println!("all: {}", gen.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
    println!(
        "windows={windows} drafts={drafts} accepted={accepted} tau={:.3} tok/window={:.3} shapes={}",
        accepted as f64 / windows.max(1) as f64,
        gen.len() as f64 / windows.max(1) as f64,
        shape.join(",")
    );
    // state checksums
    let mut kvh = 0xcbf29ce484222325u64;
    for l in 0..slot.kv_cache.k_gpu.len() {
        kvh = fnv(kvh, &tbytes(&gpu, &slot.kv_cache.k_gpu[l]));
        kvh = fnv(kvh, &tbytes(&gpu, &slot.kv_cache.v_gpu[l]));
    }
    let mut dnh = 0xcbf29ce484222325u64;
    for set in [&slot.dn_state.s_matrices, &slot.dn_state.s_scales, &slot.dn_state.conv_states, &slot.dn_state.s_ef_residual] {
        for t in set.iter() {
            dnh = fnv(dnh, &tbytes(&gpu, t));
        }
    }
    let st = drafter.mtp_live_state().expect("mtp state");
    let mut mh = fnv(0xcbf29ce484222325, &tbytes(&gpu, &st.prev_hidden));
    for b in st.mtp_kv.inner.k_gpu.iter().chain(st.mtp_kv.inner.v_gpu.iter()) {
        mh = fnv(mh, &tbytes(&gpu, b));
    }
    println!("state: trunk_kv={kvh:016x} dn={dnh:016x} mtp={mh:016x}");
    println!(
        "time: {:.3}s for {} tokens in {windows} windows ({:.1} us/window, {:.1} tok/s; tiny model: launch/sync-bound)",
        secs,
        gen.len(),
        secs * 1e6 / windows.max(1) as f64,
        gen.len() as f64 / secs
    );
}
