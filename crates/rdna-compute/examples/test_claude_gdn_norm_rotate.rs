// claude C2 (CLAUDE_GDN_NORM_ROTATE_FUSE): byte-identity + timing of
// claude_gated_norm_rotate_q8 against the verify GDN sequence it replaces:
//
//   old: gated_norm_f32_batched(x, z, w) -> normed ; claude_rotate_q8(normed, awq) -> x_rot, xq, dx
//   new: claude_gated_norm_rotate_q8(x, z, w, awq) -> normed, x_rot, xq, dx
//
// All four outputs are compared bit for bit (new outputs pre-poisoned). Inputs
// are LCG data with outliers, all-zero 32-blocks in x (amax == 0 path in the
// quant after the FWHT only if a whole group is zero: row 0 group 1 is zeroed),
// large |z| (expf overflow / underflow of the silu), one all-zero head.
// For batch >= 2 the host entry (claude_gated_norm_rotate_awq_prequant) is also
// driven into claude_q8x_project: prequant entry recorded and consumed, GEMV
// bits equal to the standalone-quant path.
// Timing: GPU time per sequence, reps back to back behind a spin kernel,
// event-bracketed (dispatch gaps included).
//
// Usage: test_claude_gdn_norm_rotate [reps]   (exit 1 on any FAIL)
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::{DType, Gpu, GpuTensor};

const EPS: f32 = 1e-6;

/// Random MQ4G256V2 rows (same layout as test_claude_q8_fuse).
fn mq4v2_weights(gpu: &mut Gpu, m: usize, k: usize, seed: u64) -> GpuTensor {
    const SC: [u16; 4] = [0x2148, 0x2A66, 0x1E66, 0x2466];
    const ZP: [u16; 4] = [0xB266, 0x3266, 0xAE66, 0x2E66];
    let mut r = Lcg(seed);
    let gpr = k / 256;
    let mut bytes = Vec::with_capacity(m * gpr * 136);
    for _ in 0..m * gpr {
        for _ in 0..2 {
            bytes.extend_from_slice(&SC[(r.next() & 3) as usize].to_le_bytes());
            bytes.extend_from_slice(&ZP[(r.next() & 3) as usize].to_le_bytes());
        }
        for _ in 0..128 {
            bytes.push(r.next() as u8);
        }
    }
    let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload weights");
    t.dtype = DType::MQ4G256V2;
    t
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut gpu = Gpu::init().expect("gpu init");
    println!("test_claude_gdn_norm_rotate: arch={} reps={reps}", gpu.arch);
    let mut fails = 0usize;
    let hd = 128usize;
    // Qwen3.8-27B GDN: 48 value heads x 128 = 6144. Also 32 x 128 = 4096 and 16 x 128.
    for &nh in &[48usize, 32, 16] {
        let k = nh * hd;
        for &b in &[1usize, 2, 3, 4, 8] {
            let n = b * k;
            let mut r = Lcg(0xC2 ^ ((nh as u64) << 8) ^ ((b as u64) << 20));
            let mut xs = r.acts(n, 2.0, 0);
            for v in &mut xs[256..512] {
                *v = 0.0; // row 0, heads 2-3: an all-zero 256-group (rms = 0 -> x*rsqrt(eps))
            }
            let mut zs: Vec<f32> = (0..n).map(|_| r.sym(6.0)).collect();
            for (i, v) in zs.iter_mut().enumerate() {
                if i % 211 == 7 {
                    *v = 95.0; // expf(-z) underflows
                } else if i % 211 == 9 {
                    *v = -95.0; // expf(-z) overflows to inf -> silu = -0
                }
            }
            let ws: Vec<f32> = (0..hd).map(|_| r.range(0.3, 1.7)).collect();
            let aw: Vec<f32> = (0..k).map(|_| r.range(0.25, 4.0)).collect();
            let x = up(&mut gpu, &xs);
            let z = up(&mut gpu, &zs);
            let w = up(&mut gpu, &ws);
            let awq = up(&mut gpu, &aw);

            let ref_norm = gpu.zeros(&[n], DType::F32).unwrap();
            let ref_rot = gpu.zeros(&[n], DType::F32).unwrap();
            let ref_xq = gpu.zeros(&[n / 4], DType::F32).unwrap();
            let ref_dx = gpu.zeros(&[n / 32], DType::F32).unwrap();
            let new_norm = poison(&mut gpu, n);
            let new_rot = poison(&mut gpu, n);
            let new_xq = poison(&mut gpu, n / 4);
            let new_dx = poison(&mut gpu, n / 32);

            gpu.gated_norm_f32_batched(&x, &z, &w, &ref_norm, nh, hd, EPS, b).unwrap();
            gpu.claude_rotate_q8_batched(&ref_norm, &awq, &ref_rot, ref_xq.buf.as_ptr(), ref_dx.buf.as_ptr(), k, b)
                .unwrap();
            gpu.claude_gated_norm_rotate_q8_batched(
                &x, &z, &w, &awq, &new_norm, &new_rot, new_xq.buf.as_ptr(), new_dx.buf.as_ptr(), nh, hd, EPS, b,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let norm_mm = mism(&bits(&gpu, &ref_norm), &bits(&gpu, &new_norm));
            let rot_mm = mism(&bits(&gpu, &ref_rot), &bits(&gpu, &new_rot));
            let xq_mm = mism(&bits(&gpu, &ref_xq), &bits(&gpu, &new_xq));
            let dx_mm = mism(&bits(&gpu, &ref_dx), &bits(&gpu, &new_dx));
            let nz = bits(&gpu, &ref_xq).iter().filter(|&&v| v != 0).count();
            let mut ok = norm_mm == 0 && rot_mm == 0 && xq_mm == 0 && dx_mm == 0 && nz > n / 8;

            // host entry -> claude_q8x_project (batch >= 2)
            let mut e2e = String::from("e2e=n/a(b<2)");
            if b >= 2 {
                if !gpu.claude_q8_fuse_eligible(k, b) {
                    e2e = "e2e=SKIP(not eligible)".into();
                } else {
                    let m = 256usize;
                    let wts = mq4v2_weights(&mut gpu, m, k, 0xBEEF ^ k as u64);
                    let other = up(&mut gpu, &vec![0.37f32; n]);
                    let d_norm = poison(&mut gpu, n);
                    let d_rot = poison(&mut gpu, n);
                    let y_hit = gpu.zeros(&[b * m], DType::F32).unwrap();
                    let y_ref = gpu.zeros(&[b * m], DType::F32).unwrap();
                    // pollute the shared int8 scratch with another input first
                    gpu.claude_q8x_project(&[(&wts, &y_ref, m)], &other, k, b, false).unwrap();
                    let ran = gpu
                        .claude_gated_norm_rotate_awq_prequant(&x, &z, &w, &awq, &d_norm, &d_rot, nh, hd, EPS, b)
                        .unwrap();
                    let pending = gpu.claude_q8_prequant_pending(&d_rot, k, b);
                    gpu.claude_q8x_project(&[(&wts, &y_hit, m)], &d_rot, k, b, false).unwrap();
                    let consumed = !gpu.claude_q8_prequant_pending(&d_rot, k, b);
                    // reference: old pair + project (standalone quant inside project)
                    gpu.claude_q8x_project(&[(&wts, &y_ref, m)], &ref_rot, k, b, false).unwrap();
                    gpu.hip.device_synchronize().unwrap();
                    let y_mm = mism(&bits(&gpu, &y_hit), &bits(&gpu, &y_ref));
                    let dn_mm = mism(&bits(&gpu, &ref_norm), &bits(&gpu, &d_norm));
                    let dr_mm = mism(&bits(&gpu, &ref_rot), &bits(&gpu, &d_rot));
                    let e_ok = ran && pending && consumed && y_mm == 0 && dn_mm == 0 && dr_mm == 0;
                    ok &= e_ok;
                    e2e = format!(
                        "e2e: ran={ran} pending={pending} consumed={consumed} normed_mm={dn_mm} x_rot_mm={dr_mm} gemv_y_mm={y_mm}"
                    );
                    for t in [wts, other, d_norm, d_rot, y_hit, y_ref] {
                        gpu.free_tensor(t).unwrap();
                    }
                }
            }

            let t_old = time_us(&mut gpu, reps, |g| {
                g.gated_norm_f32_batched(&x, &z, &w, &ref_norm, nh, hd, EPS, b).unwrap();
                g.claude_rotate_q8_batched(&ref_norm, &awq, &ref_rot, ref_xq.buf.as_ptr(), ref_dx.buf.as_ptr(), k, b)
                    .unwrap();
            });
            let t_new = time_us(&mut gpu, reps, |g| {
                g.claude_gated_norm_rotate_q8_batched(
                    &x, &z, &w, &awq, &new_norm, &new_rot, new_xq.buf.as_ptr(), new_dx.buf.as_ptr(), nh, hd, EPS, b,
                )
                .unwrap();
            });
            if !ok {
                fails += 1;
            }
            println!(
                "{} heads={nh} k={k} b={b} normed_mm={norm_mm} x_rot_mm={rot_mm} xq_mm={xq_mm} dx_mm={dx_mm} (nonzero q words {nz}/{}) {e2e} | old 2 launches {t_old:6.2} us  fused {t_new:6.2} us",
                if ok { "PASS" } else { "FAIL" },
                n / 4
            );
            for t in [x, z, w, awq, ref_norm, ref_rot, ref_xq, ref_dx, new_norm, new_rot, new_xq, new_dx] {
                gpu.free_tensor(t).unwrap();
            }
        }
    }
    if fails == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
}
