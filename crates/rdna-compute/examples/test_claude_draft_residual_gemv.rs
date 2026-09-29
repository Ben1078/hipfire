// claude C4a (CLAUDE_DRAFT_CLEANUP): the draft head's `o = W·x; o += r` and
// `h = W·x; h += o` (plain prerotated GEMV + add_inplace_f32) replaced by the
// residual GEMV `y = r; y += W·x` (gemv_hfq4g256_residual). Checks the
// equivalence the draft code relies on (claude_residual_gemv_exact) for
// MQ4G256 = HFQ4-G256 rows on this arch with the env's kernel selection
// (run once more with HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32=0 for the other
// gfx1100 residual source):
//   old_a: t = W·x (gemv_hfq4g256) ; t += r   (the old draft order)
//   old_b: t = W·x ; r' = r ; r' += t          (the cleanup's non-residual fallback)
//   new  : y = r ; y += W·x (gemv_hfq4g256_residual)
// all bit-equal. Shapes: the Qwen3.8-27B draft wo (5120 x 6144), down
// (5120 x 17408), eh_proj-like (5120 x 10240) plus tail-group shapes.
// Timing: old (GEMV + add) vs new (residual GEMV), queued, events.
//
// Usage: test_claude_draft_residual_gemv [reps]   (exit 1 on any FAIL)
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Random HFQ4-G256 / MQ4G256 rows: per 256-group [f32 scale][f32 zero][128 B nibbles].
fn hfq4_weights(gpu: &mut Gpu, m: usize, k: usize, seed: u64) -> GpuTensor {
    let mut r = Lcg(seed);
    let gpr = k / 256;
    let mut bytes = Vec::with_capacity(m * gpr * 136);
    for _ in 0..m * gpr {
        let sc = r.range(0.002, 0.02);
        let zp = -7.5 * sc + r.sym(0.01);
        bytes.extend_from_slice(&sc.to_le_bytes());
        bytes.extend_from_slice(&zp.to_le_bytes());
        for _ in 0..128 {
            bytes.push(r.next() as u8);
        }
    }
    let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload weights");
    t.dtype = DType::MQ4G256;
    t
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "test_claude_draft_residual_gemv: arch={} reps={reps} residual_stage_x32={} gemv_rows={:?}",
        gpu.arch, gpu.flags.rdna3_hfq4_residual_stage_x32, gpu.flags.gemv_rows
    );
    let mut fails = 0usize;
    for &(m, k) in &[
        (5120usize, 6144usize),
        (5120, 17408),
        (5120, 10240),
        (12288, 5120),
        (1024, 2816),
        (777, 768),
        (64, 256),
    ] {
        let mut r = Lcg(0xC4A ^ ((m as u64) << 20) ^ k as u64);
        let w = hfq4_weights(&mut gpu, m, k, 0xBEE ^ (m * 131 + k) as u64);
        let x = up(&mut gpu, &r.acts(k, 0.8, 0));
        let r_h: Vec<f32> = (0..m).map(|_| r.sym(4.0)).collect();
        let rr = up(&mut gpu, &r_h);
        let t = gpu.zeros(&[m], DType::F32).unwrap();
        let rb = up(&mut gpu, &r_h);
        let y = up(&mut gpu, &r_h);

        gpu.gemv_hfq4g256(&w, &x, &t, m, k).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t_bits = bits(&gpu, &t);
        gpu.add_inplace_f32(&rb, &t).unwrap(); // old_b: r' = r + acc
        gpu.add_inplace_f32(&t, &rr).unwrap(); // old_a: t = acc + r
        gpu.gemv_hfq4g256_residual(&w, &x, &y, m, k).unwrap(); // new: y = r + acc
        gpu.hip.device_synchronize().unwrap();
        let a = bits(&gpu, &t);
        let b = bits(&gpu, &rb);
        let n = bits(&gpu, &y);
        let mm_a = mism(&a, &n);
        let mm_b = mism(&b, &n);
        let changed = mism(&t_bits, &a);
        let ok = mm_a == 0 && mm_b == 0 && changed > m / 2;
        if !ok {
            fails += 1;
        }
        let t_old = time_us(&mut gpu, reps, |g| {
            g.gemv_hfq4g256(&w, &x, &t, m, k).unwrap();
            g.add_inplace_f32(&t, &rr).unwrap();
        });
        let t_new = time_us(&mut gpu, reps, |g| {
            g.gemv_hfq4g256_residual(&w, &x, &y, m, k).unwrap();
        });
        println!(
            "{} m={m:<5} k={k:<5} (acc+r vs residual) mm={mm_a}  (r+acc vs residual) mm={mm_b} | GEMV+add {t_old:7.2} us  residual GEMV {t_new:7.2} us",
            if ok { "PASS" } else { "FAIL" }
        );
        for tt in [w, x, rr, t, rb, y] {
            gpu.free_tensor(tt).unwrap();
        }
    }
    if fails == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
}
