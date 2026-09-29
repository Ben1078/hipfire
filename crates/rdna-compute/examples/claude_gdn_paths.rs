// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! GDN prefill block (everything between the qkvz projection and the output
//! projection) on one 1024-row chunk, Qwen3.8 dims, three routes:
//!
//!  A. MTP-prefill route today: sigmoid/alpha gate, conv1d+SiLU+split,
//!     L2-norm + interleave, two 512-row `gated_delta_net_q8_fast` launches.
//!  B. Same with the quad-row GDN (one launch; conv route per CLAUDE_CONV_TPAR).
//!  C. The WMMA chunk scan the plain prefill route already uses on gfx1100
//!     (`gdn_chunk_prep` + two `gdn_chunk_scan_segment`s), which the MTP
//!     prefill route does not admit.
//!
//! Prints per-stage times and B-vs-A (must be bit-exact) and C-vs-A output
//! differences.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        lo + (hi - lo) * (((self.0 >> 40) as f32) / 16777216.0)
    }
}

fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    let mant = b & 0x7fffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn time_us(gpu: &mut Gpu, reps: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..3 {
        f(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut v = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        f(gpu);
        gpu.hip.device_synchronize().expect("sync");
        v.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    let (n, seg) = (1024usize, 512usize);
    let (hk, hv, hd) = (16usize, 48usize, 128usize);
    let (k_dim, v_dim) = (hk * hd, hv * hd);
    let nc = 2 * k_dim + v_dim;
    let q_scale = 1.0 / (hd as f32).sqrt();
    let eps = 1e-6f32;
    let mut r = Rng(0x1234_5678_9abc_def1);

    let qkv: Vec<f32> = (0..n * nc).map(|_| r.f(-2.0, 2.0)).collect();
    let alpha_raw: Vec<f32> = (0..n * hv).map(|_| r.f(-3.0, 3.0)).collect();
    let beta_raw: Vec<f32> = (0..n * hv).map(|_| r.f(-3.0, 3.0)).collect();
    let dt_bias: Vec<f32> = (0..hv).map(|_| r.f(-0.5, 0.5)).collect();
    let a_log: Vec<f32> = (0..hv).map(|_| r.f(0.0, 2.7)).collect();
    let conv_w: Vec<f32> = (0..nc * 4).map(|_| r.f(-0.6, 0.6)).collect();
    let conv_s: Vec<f32> = (0..nc * 3).map(|_| r.f(-1.0, 1.0)).collect();
    let s8: Vec<u8> = (0..hv * hd * hd).map(|_| (r.f(-60.0, 60.0) as i8) as u8).collect();
    let sc: Vec<f32> = (0..hv * hd).map(|_| r.f(0.001, 0.01)).collect();
    let ef: Vec<u8> = (0..hv * hd * hd).flat_map(|_| f16_bits(r.f(-1e-4, 1e-4)).to_le_bytes()).collect();

    let d_qkv = gpu.upload_f32(&qkv, &[n, nc]).unwrap();
    let d_w = gpu.upload_f32(&conv_w, &[nc, 4]).unwrap();
    let d_dt = gpu.upload_f32(&dt_bias, &[hv]).unwrap();
    let d_al = gpu.upload_f32(&a_log, &[hv]).unwrap();

    struct Arm {
        alpha: GpuTensor,
        beta: GpuTensor,
        conv: GpuTensor,
        s8: GpuTensor,
        sc: GpuTensor,
        ef: GpuTensor,
        out: GpuTensor,
    }
    let mut mk = |gpu: &mut Gpu| Arm {
        alpha: gpu.upload_f32(&alpha_raw, &[n, hv]).unwrap(),
        beta: gpu.upload_f32(&beta_raw, &[n, hv]).unwrap(),
        conv: gpu.upload_f32(&conv_s, &[nc, 3]).unwrap(),
        s8: gpu.upload_raw(&s8, &[s8.len()]).unwrap(),
        sc: gpu.upload_f32(&sc, &[sc.len()]).unwrap(),
        ef: gpu.upload_raw(&ef, &[ef.len()]).unwrap(),
        out: gpu.zeros(&[n, v_dim], DType::F32).unwrap(),
    };
    let arm_a = mk(&mut gpu);
    let arm_b = mk(&mut gpu);
    let arm_c = mk(&mut gpu);

    // Shared intermediates of routes A/B.
    let q_raw = gpu.zeros(&[n, k_dim], DType::F32).unwrap();
    let k_raw = gpu.zeros(&[n, k_dim], DType::F32).unwrap();
    let v_b = gpu.zeros(&[n, v_dim], DType::F32).unwrap();
    let q_x = gpu.zeros(&[n, v_dim], DType::F32).unwrap();
    let k_x = gpu.zeros(&[n, v_dim], DType::F32).unwrap();
    // Route C scratch (f16 q/k compact, v, A).
    let q16 = gpu.zeros(&[n * k_dim], DType::F16).unwrap();
    let k16 = gpu.zeros(&[n * k_dim], DType::F16).unwrap();
    let v16 = gpu.zeros(&[n * v_dim], DType::F16).unwrap();
    let a16 = gpu.zeros(&[(seg + 63) / 64 * 64 * hv * 64], DType::F16).unwrap();

    let pre = |gpu: &mut Gpu, arm: &Arm| {
        gpu.fused_sigmoid_alpha_gate_f32_batched(&arm.beta, &arm.alpha, &d_dt, &d_al, hv, n).unwrap();
        gpu.conv1d_silu_split_f32_n(&q_raw, &k_raw, &v_b, &d_qkv, &d_w, &arm.conv, k_dim, v_dim, n).unwrap();
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(&q_raw, &k_raw, &q_x, &k_x, hk, hv / hk, hd, q_scale, eps, n)
            .unwrap();
    };
    let gdn_fast = |gpu: &mut Gpu, arm: &Arm| {
        for off in (0..n).step_by(seg) {
            let sn = (n - off).min(seg);
            gpu.gated_delta_net_q8_batch_seq(
                &q_x.sub_offset(off * v_dim, sn * v_dim),
                &k_x.sub_offset(off * v_dim, sn * v_dim),
                &v_b.sub_offset(off * v_dim, sn * v_dim),
                &arm.alpha.sub_offset(off * hv, sn * hv),
                &arm.beta.sub_offset(off * hv, sn * hv),
                &arm.s8,
                &arm.sc,
                &arm.out.sub_offset(off * v_dim, sn * v_dim),
                sn,
                hv,
                hd,
                Some(&arm.ef),
            )
            .unwrap();
        }
    };
    let gdn_quad = |gpu: &mut Gpu, arm: &Arm| {
        assert!(gpu
            .gated_delta_net_q8_quadrow_segments(
                &q_x, &k_x, &v_b, &arm.alpha, &arm.beta, &arm.s8, &arm.sc, &arm.out, n, hv, hd, seg,
                Some(&arm.ef),
            )
            .unwrap());
    };
    let chunk = |gpu: &mut Gpu, arm: &Arm| {
        gpu.gdn_chunk_prep(&d_qkv, &d_w, &arm.conv, &arm.alpha, &arm.beta, &d_dt, &d_al, &q16, &k16, &v16, n, q_scale, eps)
            .unwrap();
        for row0 in (0..n).step_by(seg) {
            let rows = (n - row0).min(seg);
            gpu.gdn_chunk_scan_segment(&q16, &k16, &v16, &a16, &arm.alpha, &arm.beta, &arm.s8, &arm.sc, &arm.ef, &arm.out, row0, rows)
                .unwrap();
        }
    };

    // ---- numerics (first run from identical inputs)
    pre(&mut gpu, &arm_a);
    gdn_fast(&mut gpu, &arm_a);
    let out_a = gpu.download_f32(&arm_a.out).unwrap();
    pre(&mut gpu, &arm_b);
    gdn_quad(&mut gpu, &arm_b);
    let out_b = gpu.download_f32(&arm_b.out).unwrap();
    chunk(&mut gpu, &arm_c);
    gpu.hip.device_synchronize().unwrap();
    let out_c = gpu.download_f32(&arm_c.out).unwrap();
    let same_ab = out_a.iter().zip(&out_b).all(|(x, y)| x.to_bits() == y.to_bits());
    let (mut max_abs, mut ref_max, mut sum_sq, mut sum_ref) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in out_a.iter().zip(&out_c) {
        let d = (*x as f64 - *y as f64).abs();
        max_abs = max_abs.max(d);
        ref_max = ref_max.max((*x as f64).abs());
        sum_sq += d * d;
        sum_ref += (*x as f64) * (*x as f64);
    }
    println!("B (quad-row) vs A (fast x2): {}", if same_ab { "BIT-EXACT" } else { "MISMATCH" });
    println!(
        "C (chunk scan) vs A: max|diff| {:.3e} (max|A| {:.3e}), rel RMS {:.3e}",
        max_abs,
        ref_max,
        (sum_sq / sum_ref.max(1e-30)).sqrt()
    );

    // ---- timing (medians of 10; inputs keep being re-transformed in place,
    // which does not matter for timing)
    let reps = 10;
    let t_gate = time_us(&mut gpu, reps, |g| g.fused_sigmoid_alpha_gate_f32_batched(&arm_a.beta, &arm_a.alpha, &d_dt, &d_al, hv, n).unwrap());
    let t_conv = time_us(&mut gpu, reps, |g| {
        g.conv1d_silu_split_f32_n(&q_raw, &k_raw, &v_b, &d_qkv, &d_w, &arm_a.conv, k_dim, v_dim, n).unwrap()
    });
    let t_l2 = time_us(&mut gpu, reps, |g| {
        g.fused_qk_l2_norm_scale_interleave_f32_batched(&q_raw, &k_raw, &q_x, &k_x, hk, hv / hk, hd, q_scale, eps, n).unwrap()
    });
    let t_fast = time_us(&mut gpu, reps, |g| gdn_fast(g, &arm_a));
    let t_quad = time_us(&mut gpu, reps, |g| gdn_quad(g, &arm_b));
    let t_chunk = time_us(&mut gpu, reps, |g| chunk(g, &arm_c));
    println!(
        "per 1024-row chunk (us): gate {t_gate:.1}  conv1d {t_conv:.1} (CLAUDE_CONV_TPAR={:?})  l2norm {t_l2:.1}  gdn fast x2 {t_fast:.1}  gdn quad-row {t_quad:.1}",
        std::env::var("CLAUDE_CONV_TPAR").ok()
    );
    println!(
        "route A {:.1}  route B {:.1}  route C (chunk prep+scan) {:.1}",
        t_gate + t_conv + t_l2 + t_fast,
        t_gate + t_conv + t_l2 + t_quad,
        t_chunk
    );
}
