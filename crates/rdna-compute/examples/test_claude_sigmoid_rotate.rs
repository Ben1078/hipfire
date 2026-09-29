// claude C3 (CLAUDE_FA_GATE_ROTATE_FUSE): byte-identity + timing of
// claude_sigmoid_rotate_{q8,awq,x} against the sequences they replace:
//
//   verify FA wo input : sigmoid_mul_f32(attn, gate) ; claude_rotate_q8(attn, awq) -> x_rot, xq, dx
//                        vs claude_sigmoid_rotate_q8(attn, gate, awq) -> attn, x_rot, xq, dx
//   draft head wo input: sigmoid_mul_f32(attn, gate) ; mq_rotate_x(attn) -> x_rot  (batch 1)
//                        vs claude_sigmoid_rotate_x
//   (AWQ, no int8)     : sigmoid_mul_f32 ; rotate_x_mq_awq (q8 fuse forced off) vs claude_sigmoid_rotate_awq
//
// The old sigmoid runs over the whole scratch (8 rows, as the verify scratch
// is larger than n); rows < batch of attn, all of x_rot and xq/dx are compared
// bit for bit (new outputs pre-poisoned). Gates include +-95 (expf overflow /
// underflow). For batch >= 2 the verify host entry is driven into
// claude_q8x_project (prequant recorded + consumed, GEMV bits equal).
// Timing: GPU time per sequence, reps back to back behind a spin kernel,
// event-bracketed (dispatch gaps included).
//
// Usage: test_claude_sigmoid_rotate [reps]   (exit 1 on any FAIL)
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::{DType, Gpu, GpuTensor};

const ROWS_ALLOC: usize = 8;

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

fn gates(r: &mut Lcg, n: usize) -> Vec<f32> {
    let mut g: Vec<f32> = (0..n).map(|_| r.sym(8.0)).collect();
    for (i, v) in g.iter_mut().enumerate() {
        if i % 193 == 5 {
            *v = 95.0;
        } else if i % 193 == 11 {
            *v = -95.0;
        }
    }
    g
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut gpu = Gpu::init().expect("gpu init");
    println!("test_claude_sigmoid_rotate: arch={} reps={reps}", gpu.arch);
    let mut fails = 0usize;

    // ---------------- verify variant (AWQ + int8) ----------------
    for &k in &[6144usize, 4096, 2048] {
        for &b in &[1usize, 2, 3, 4, 8] {
            let na = ROWS_ALLOC * k;
            let n = b * k;
            let mut r = Lcg(0xC3 ^ ((k as u64) << 8) ^ ((b as u64) << 24));
            let mut attn_h = r.acts(na, 1.5, 0);
            for v in &mut attn_h[256..512] {
                *v = 0.0; // an all-zero 256-group (amax == 0 in the quant)
            }
            let gate_h = gates(&mut r, na);
            let aw: Vec<f32> = (0..k).map(|_| r.range(0.25, 4.0)).collect();
            let gate = up(&mut gpu, &gate_h);
            let awq = up(&mut gpu, &aw);
            let attn_ref = up(&mut gpu, &attn_h);
            let attn_new = up(&mut gpu, &attn_h);
            let ref_rot = gpu.zeros(&[n], DType::F32).unwrap();
            let ref_xq = gpu.zeros(&[n / 4], DType::F32).unwrap();
            let ref_dx = gpu.zeros(&[n / 32], DType::F32).unwrap();
            let new_rot = poison(&mut gpu, n);
            let new_xq = poison(&mut gpu, n / 4);
            let new_dx = poison(&mut gpu, n / 32);

            gpu.sigmoid_mul_f32(&attn_ref, &gate).unwrap();
            gpu.claude_rotate_q8_batched(&attn_ref, &awq, &ref_rot, ref_xq.buf.as_ptr(), ref_dx.buf.as_ptr(), k, b)
                .unwrap();
            gpu.claude_sigmoid_rotate_batched(
                &attn_new,
                &gate,
                Some(&awq),
                &new_rot,
                Some((new_xq.buf.as_ptr(), new_dx.buf.as_ptr())),
                k,
                b,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let a_ref = bits(&gpu, &attn_ref);
            let a_new = bits(&gpu, &attn_new);
            let attn_mm = mism(&a_ref[..n], &a_new[..n]);
            let rot_mm = mism(&bits(&gpu, &ref_rot), &bits(&gpu, &new_rot));
            let xq_mm = mism(&bits(&gpu, &ref_xq), &bits(&gpu, &new_xq));
            let dx_mm = mism(&bits(&gpu, &ref_dx), &bits(&gpu, &new_dx));
            let nz = bits(&gpu, &ref_xq).iter().filter(|&&v| v != 0).count();
            let mut ok = attn_mm == 0 && rot_mm == 0 && xq_mm == 0 && dx_mm == 0 && nz > n / 8;

            let mut e2e = String::from("e2e=n/a(b<2)");
            if b >= 2 {
                if !gpu.claude_q8_fuse_eligible(k, b) {
                    e2e = "e2e=SKIP(not eligible)".into();
                } else {
                    let m = 256usize;
                    let wts = mq4v2_weights(&mut gpu, m, k, 0xFACE ^ k as u64);
                    let other = up(&mut gpu, &vec![0.37f32; n]);
                    let d_attn = up(&mut gpu, &attn_h);
                    let d_rot = poison(&mut gpu, n);
                    let y_hit = gpu.zeros(&[b * m], DType::F32).unwrap();
                    let y_ref = gpu.zeros(&[b * m], DType::F32).unwrap();
                    gpu.claude_q8x_project(&[(&wts, &y_ref, m)], &other, k, b, false).unwrap();
                    let ran = gpu.claude_sigmoid_rotate_awq_prequant(&d_attn, &gate, &awq, &d_rot, k, b).unwrap();
                    let pending = gpu.claude_q8_prequant_pending(&d_rot, k, b);
                    gpu.claude_q8x_project(&[(&wts, &y_hit, m)], &d_rot, k, b, false).unwrap();
                    let consumed = !gpu.claude_q8_prequant_pending(&d_rot, k, b);
                    gpu.claude_q8x_project(&[(&wts, &y_ref, m)], &ref_rot, k, b, false).unwrap();
                    gpu.hip.device_synchronize().unwrap();
                    let y_mm = mism(&bits(&gpu, &y_hit), &bits(&gpu, &y_ref));
                    let da_mm = mism(&a_ref[..n], &bits(&gpu, &d_attn)[..n]);
                    let dr_mm = mism(&bits(&gpu, &ref_rot), &bits(&gpu, &d_rot));
                    let e_ok = ran && pending && consumed && y_mm == 0 && da_mm == 0 && dr_mm == 0;
                    ok &= e_ok;
                    e2e = format!(
                        "e2e: ran={ran} pending={pending} consumed={consumed} attn_mm={da_mm} x_rot_mm={dr_mm} gemv_y_mm={y_mm}"
                    );
                    for t in [wts, other, d_attn, d_rot, y_hit, y_ref] {
                        gpu.free_tensor(t).unwrap();
                    }
                }
            }
            // timing: the production verify scratch holds 4 rows (sigmoid over all 4)
            let attn4 = attn_ref.sub_offset(0, 4 * k);
            let gate4 = gate.sub_offset(0, 4 * k);
            let t_old = time_us(&mut gpu, reps, |g| {
                g.sigmoid_mul_f32(&attn4, &gate4).unwrap();
                g.claude_rotate_q8_batched(&attn_ref, &awq, &ref_rot, ref_xq.buf.as_ptr(), ref_dx.buf.as_ptr(), k, b)
                    .unwrap();
            });
            let t_new = time_us(&mut gpu, reps, |g| {
                g.claude_sigmoid_rotate_batched(
                    &attn_new,
                    &gate,
                    Some(&awq),
                    &new_rot,
                    Some((new_xq.buf.as_ptr(), new_dx.buf.as_ptr())),
                    k,
                    b,
                )
                .unwrap();
            });
            if !ok {
                fails += 1;
            }
            println!(
                "{} verify q8   k={k} b={b} attn_mm={attn_mm} x_rot_mm={rot_mm} xq_mm={xq_mm} dx_mm={dx_mm} (nonzero q words {nz}/{}) {e2e} | old 2 launches {t_old:6.2} us  fused {t_new:6.2} us",
                if ok { "PASS" } else { "FAIL" },
                n / 4
            );
            for t in [gate, awq, attn_ref, attn_new, ref_rot, ref_xq, ref_dx, new_rot, new_xq, new_dx] {
                gpu.free_tensor(t).unwrap();
            }
        }
    }

    // ---------------- draft variant (no AWQ, f32 x_rot, batch 1) ----------------
    for &k in &[6144usize, 5120, 2048, 256] {
        let mut r = Lcg(0xD3 ^ ((k as u64) << 8));
        let attn_h = r.acts(k, 1.5, 0);
        let gate_h = gates(&mut r, k);
        let gate = up(&mut gpu, &gate_h);
        let attn_ref = up(&mut gpu, &attn_h);
        let attn_new = up(&mut gpu, &attn_h);
        let ref_rot = gpu.zeros(&[k], DType::F32).unwrap();
        let new_rot = poison(&mut gpu, k);
        gpu.sigmoid_mul_f32(&attn_ref, &gate).unwrap();
        gpu.rotate_x_mq(&attn_ref, &ref_rot, k).unwrap();
        gpu.claude_sigmoid_rotate_batched(&attn_new, &gate, None, &new_rot, None, k, 1).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let attn_mm = mism(&bits(&gpu, &attn_ref), &bits(&gpu, &attn_new));
        let rot_mm = mism(&bits(&gpu, &ref_rot), &bits(&gpu, &new_rot));
        let ok = attn_mm == 0 && rot_mm == 0;
        let t_old = time_us(&mut gpu, reps, |g| {
            g.sigmoid_mul_f32(&attn_ref, &gate).unwrap();
            g.rotate_x_mq(&attn_ref, &ref_rot, k).unwrap();
        });
        let t_new = time_us(&mut gpu, reps, |g| {
            g.claude_sigmoid_rotate_batched(&attn_new, &gate, None, &new_rot, None, k, 1).unwrap();
        });
        if !ok {
            fails += 1;
        }
        println!(
            "{} draft  x    k={k} b=1 attn_mm={attn_mm} x_rot_mm={rot_mm} | old 2 launches {t_old:6.2} us  fused {t_new:6.2} us",
            if ok { "PASS" } else { "FAIL" }
        );
        for t in [gate, attn_ref, attn_new, ref_rot, new_rot] {
            gpu.free_tensor(t).unwrap();
        }
    }

    // ---------------- AWQ without int8 (rotate_x_mq_awq) ----------------
    gpu.claude_q8_fuse_override(Some(false));
    for &k in &[6144usize, 2048] {
        for &b in &[1usize, 3] {
            let n = b * k;
            let mut r = Lcg(0xA3 ^ ((k as u64) << 8) ^ ((b as u64) << 24));
            let attn_h = r.acts(n, 1.5, 0);
            let gate_h = gates(&mut r, n);
            let aw: Vec<f32> = (0..k).map(|_| r.range(0.25, 4.0)).collect();
            let gate = up(&mut gpu, &gate_h);
            let awq = up(&mut gpu, &aw);
            let attn_ref = up(&mut gpu, &attn_h);
            let attn_new = up(&mut gpu, &attn_h);
            let ref_rot = gpu.zeros(&[n], DType::F32).unwrap();
            let new_rot = poison(&mut gpu, n);
            gpu.sigmoid_mul_f32(&attn_ref, &gate).unwrap();
            gpu.rotate_x_mq_awq_batched(&attn_ref, &awq, &ref_rot, k, b).unwrap();
            gpu.claude_sigmoid_rotate_batched(&attn_new, &gate, Some(&awq), &new_rot, None, k, b).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let attn_mm = mism(&bits(&gpu, &attn_ref), &bits(&gpu, &attn_new));
            let rot_mm = mism(&bits(&gpu, &ref_rot), &bits(&gpu, &new_rot));
            let ok = attn_mm == 0 && rot_mm == 0;
            if !ok {
                fails += 1;
            }
            println!(
                "{} awq    f32  k={k} b={b} attn_mm={attn_mm} x_rot_mm={rot_mm}",
                if ok { "PASS" } else { "FAIL" }
            );
            for t in [gate, awq, attn_ref, attn_new, ref_rot, new_rot] {
                gpu.free_tensor(t).unwrap();
            }
        }
    }
    gpu.claude_q8_fuse_override(None);

    if fails == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
}
