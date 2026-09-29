// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity probe for the Claude prefill kernels through the real `Gpu`
//! dispatch (gfx1100):
//!
//! * IU4 GEMM (V2C vs V2D): SET and ADD on production-like shapes via
//!   `gemm_mq4g256v2_mmq_{set,add}_prequant_iu4`; the route is chosen by
//!   `CLAUDE_IU4_V2D`, so run the probe twice and `cmp` the dumps.
//! * conv1d + SiLU + split (`conv1d_silu_split_f32_n`, 1024 rows); route by
//!   `CLAUDE_CONV_TPAR`, compared the same way.
//! * GDN: `gated_delta_net_q8_quadrow_segments` vs two 512-row
//!   `gated_delta_net_q8_batch_seq` launches, compared in-process (outputs,
//!   Q8 state, scales, EF residual).
//!
//! Usage: claude_prefill_parity <dump-file>

use rdna_compute::{DType, Gpu, GpuTensor};
use std::io::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() & 0xffffff) as f32 / 16777216.0)
    }
}

fn f16_bits(v: f32) -> u16 {
    // Round-to-nearest-even f32 -> f16 for normal values (test data only).
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

/// MQ4G256V2 rows with the symmetric contract zp = -8*sc.
fn mq4v2_weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut w = vec![0u8; m * groups * 136];
    for g in w.chunks_mut(136) {
        for h in 0..2 {
            let sc = f16_bits(rng.f(0.002, 0.02));
            let scf = f32::from_bits(((sc as u32 & 0x3ff) << 13) | ((((sc >> 10) & 0x1f) as u32 + 112) << 23));
            let zp = f16_bits(-8.0 * scf);
            g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for b in &mut g[8..] {
            *b = rng.next() as u8;
        }
    }
    w
}

/// block_i4_128[h * n + t]: f32 d, i32 s, 64 B signed nibbles.
fn xq_blocks(rng: &mut Rng, k: usize, n: usize) -> Vec<u8> {
    let mut x = vec![0u8; k / 128 * n * 72];
    for b in x.chunks_mut(72) {
        b[0..4].copy_from_slice(&rng.f(0.01, 0.2).to_le_bytes());
        let mut s = 0i32;
        for p in &mut b[8..] {
            let v = rng.next() as u8;
            *p = v;
            for q in [v & 15, v >> 4] {
                s += if q >= 8 { q as i32 - 16 } else { q as i32 };
            }
        }
        b[4..8].copy_from_slice(&s.to_le_bytes());
    }
    x
}

fn download_bytes(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut v = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("dtoh");
    v
}

fn main() {
    let dump_path = std::env::args().nth(1).expect("usage: claude_prefill_parity <dump-file>");
    let mut dump = std::fs::File::create(&dump_path).expect("dump file");
    let mut gpu = Gpu::init().expect("gpu init");
    gpu.mq4v2_symmetric = true;
    eprintln!(
        "claude_prefill_parity: arch={} CLAUDE_IU4_V2D={:?} CLAUDE_CONV_TPAR={:?}",
        gpu.arch,
        std::env::var("CLAUDE_IU4_V2D").ok(),
        std::env::var("CLAUDE_CONV_TPAR").ok()
    );
    let mut rng = Rng(0x9E3779B97F4A7C15);

    // ---- IU4 GEMMs (SET + ADD), N = 1024 prefill chunk.
    for (m, k, add) in [(1024usize, 5120usize, false), (5120, 6144, true)] {
        let n = 1024usize;
        let a = gpu.upload_raw(&mq4v2_weights(&mut rng, m, k), &[m * k / 256 * 136]).expect("A");
        let xq = gpu.upload_raw(&xq_blocks(&mut rng, k, n), &[k / 128 * n * 72]).expect("Xq");
        let y0: Vec<f32> = (0..n * m).map(|_| rng.f(-1.0, 1.0)).collect();
        let y = gpu.upload_f32(&y0, &[n, m]).expect("Y");
        let xp = xq.buf.as_ptr();
        if add {
            gpu.gemm_mq4g256v2_mmq_add_prequant_iu4(&a, xp, &y, m, k, n).expect("add");
        } else {
            gpu.gemm_mq4g256v2_mmq_set_prequant_iu4(&a, xp, &y, m, k, n).expect("set");
        }
        gpu.hip.device_synchronize().expect("sync");
        let out = download_bytes(&gpu, &y);
        dump.write_all(&out).expect("write");
        eprintln!("  gemm {} m={m} k={k} n={n}: {} bytes dumped", if add { "ADD" } else { "SET" }, out.len());
    }

    // ---- attention q/k/v through the prepared IU4 API (grouped launch when
    // CLAUDE_IU4_GROUP is on, three SETs otherwise).
    {
        let (k, n) = (5120usize, 1024usize);
        let ms = [12288usize, 1024, 1024];
        let a: Vec<_> = ms
            .iter()
            .map(|&m| gpu.upload_raw(&mq4v2_weights(&mut rng, m, k), &[m * k / 256 * 136]).expect("A"))
            .collect();
        let y: Vec<_> = ms.iter().map(|&m| gpu.zeros(&[n, m], DType::F32).expect("Y")).collect();
        let xbytes = xq_blocks(&mut rng, k, n);
        let res = gpu.reserve_int4_mmq(k, n).expect("reserve");
        {
            let view = unsafe { hip_bridge::DeviceBuffer::from_raw(res.ptr(), xbytes.len()) };
            gpu.hip.memcpy_htod(&view, &xbytes).expect("htod");
            std::mem::forget(view);
        }
        let prep = rdna_compute::Int4MmqPrepared::from_reservation(res);
        gpu.gemm_qkv_mq4g256v2_wmma_iu4_prepared(
            &a[0], &a[1], &a[2], &prep, &y[0], &y[1], &y[2], ms[0], ms[1], ms[2], k, n,
        )
        .expect("qkv prepared");
        gpu.hip.device_synchronize().expect("sync");
        for t in &y {
            dump.write_all(&download_bytes(&gpu, t)).expect("write");
        }
        eprintln!(
            "  attention q/k/v prepared (CLAUDE_IU4_GROUP={:?}): dumped",
            std::env::var("CLAUDE_IU4_GROUP").ok()
        );
    }

    // ---- conv1d + SiLU + split, 1024 rows, two chunks (state carry).
    {
        let (k_dim, v_dim, n) = (2048usize, 6144usize, 1024usize);
        let nc = 2 * k_dim + v_dim;
        let inp: Vec<f32> = (0..n * nc).map(|i| rng.f(-3.0, 3.0) * if i % 997 == 0 { 40.0 } else { 1.0 }).collect();
        let wts: Vec<f32> = (0..nc * 4).map(|_| rng.f(-1.0, 1.0)).collect();
        let st: Vec<f32> = (0..nc * 3).map(|_| rng.f(-2.0, 2.0)).collect();
        let d_in = gpu.upload_f32(&inp, &[n, nc]).expect("in");
        let d_w = gpu.upload_f32(&wts, &[nc, 4]).expect("w");
        let d_s = gpu.upload_f32(&st, &[nc, 3]).expect("s");
        let d_q = gpu.zeros(&[n, k_dim], DType::F32).expect("q");
        let d_k = gpu.zeros(&[n, k_dim], DType::F32).expect("k");
        let d_v = gpu.zeros(&[n, v_dim], DType::F32).expect("v");
        for _ in 0..2 {
            gpu.conv1d_silu_split_f32_n(&d_q, &d_k, &d_v, &d_in, &d_w, &d_s, k_dim, v_dim, n)
                .expect("conv");
        }
        gpu.hip.device_synchronize().expect("sync");
        for t in [&d_q, &d_k, &d_v, &d_s] {
            dump.write_all(&download_bytes(&gpu, t)).expect("write");
        }
        eprintln!("  conv1d n={n}: q/k/v/state dumped");
    }

    // ---- upstream beb3e864a levers: IU4 RMSNorm `_b8` producers and the
    // gated-norm v2 producer (routes chosen by HIPFIRE_RMSNORM_P1A_BATCHED /
    // HIPFIRE_GFX1100_GATED_NORM_V2, compared across the two runs), and the
    // batched FA prep vs the three-launch chain (in-process).
    {
        let n = 1024usize;
        let sidecar = |gpu: &Gpu, ptr: *mut std::ffi::c_void, k: usize| -> Vec<u8> {
            let bytes = k / 128 * n * 72;
            let view = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
            let mut v = vec![0u8; bytes];
            gpu.hip.memcpy_dtoh(&mut v, &view).expect("dtoh sidecar");
            std::mem::forget(view);
            v
        };
        for awq in [false, true] {
            let k = 5120usize;
            let x: Vec<f32> = (0..n * k).map(|_| rng.f(-2.0, 2.0)).collect();
            let w: Vec<f32> = (0..k).map(|_| rng.f(0.5, 1.5)).collect();
            let a: Vec<f32> = (0..k).map(|_| rng.f(0.5, 2.0)).collect();
            let dx = gpu.upload_f32(&x, &[n, k]).unwrap();
            let dw = gpu.upload_f32(&w, &[k]).unwrap();
            let da = gpu.upload_f32(&a, &[k]).unwrap();
            let dr = gpu.zeros(&[n, k], DType::F32).unwrap();
            let res = gpu.reserve_int4_mmq(k, n).expect("reserve");
            let ptr = res.ptr();
            let _prep = gpu
                .fused_rmsnorm_rotate_mq_i4_batched(&dx, &dw, if awq { Some(&da) } else { None }, Some(&dr), res, k, 1e-6, n)
                .expect("rmsnorm i4");
            gpu.hip.device_synchronize().unwrap();
            dump.write_all(&download_bytes(&gpu, &dr)).unwrap();
            dump.write_all(&sidecar(&gpu, ptr, k)).unwrap();
            eprintln!(
                "  rmsnorm IU4 producer awq={awq} (HIPFIRE_RMSNORM_P1A_BATCHED={:?}): dumped",
                std::env::var("HIPFIRE_RMSNORM_P1A_BATCHED").ok()
            );
        }
        for awq in [false, true] {
            let (heads, hd) = (48usize, 128usize);
            let k = heads * hd;
            let x: Vec<f32> = (0..n * k).map(|_| rng.f(-1.0, 1.0)).collect();
            let z: Vec<f32> = (0..n * k).map(|_| rng.f(-3.0, 3.0)).collect();
            let w: Vec<f32> = (0..hd).map(|_| rng.f(0.5, 1.5)).collect();
            let a: Vec<f32> = (0..k).map(|_| rng.f(0.5, 2.0)).collect();
            let dx = gpu.upload_f32(&x, &[n, k]).unwrap();
            let dz = gpu.upload_f32(&z, &[n, k]).unwrap();
            let dw = gpu.upload_f32(&w, &[hd]).unwrap();
            let da = gpu.upload_f32(&a, &[k]).unwrap();
            let dr = gpu.zeros(&[n, k], DType::F32).unwrap();
            let res = gpu.reserve_int4_mmq(k, n).expect("reserve");
            let ptr = res.ptr();
            let _prep = gpu
                .gated_norm_rotate_mq_i4_gfx11_batched(
                    &dx, &dz, &dw, if awq { Some(&da) } else { None }, Some(&dr), res, heads, hd, 1e-6, k, n,
                )
                .expect("gated norm i4");
            gpu.hip.device_synchronize().unwrap();
            dump.write_all(&download_bytes(&gpu, &dr)).unwrap();
            dump.write_all(&sidecar(&gpu, ptr, k)).unwrap();
            eprintln!(
                "  gated-norm IU4 producer awq={awq} (HIPFIRE_GFX1100_GATED_NORM_V2={:?}): dumped",
                std::env::var("HIPFIRE_GFX1100_GATED_NORM_V2").ok()
            );
        }
        {
            let (hq, hk, hd, n_rot) = (24usize, 4usize, 256usize, 64usize);
            let qf: Vec<f32> = (0..n * hq * hd * 2).map(|_| rng.f(-2.0, 2.0)).collect();
            let kk: Vec<f32> = (0..n * hk * hd).map(|_| rng.f(-2.0, 2.0)).collect();
            let qw: Vec<f32> = (0..hd).map(|_| rng.f(0.5, 1.5)).collect();
            let kw: Vec<f32> = (0..hd).map(|_| rng.f(0.5, 1.5)).collect();
            let pos: Vec<i32> = (0..n as i32).map(|i| 5000 + i).collect();
            let pos_bytes: Vec<u8> = pos.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dqf = gpu.upload_f32(&qf, &[n, hq * hd * 2]).unwrap();
            let dqw = gpu.upload_f32(&qw, &[hd]).unwrap();
            let dkw = gpu.upload_f32(&kw, &[hd]).unwrap();
            let dpos = gpu.upload_raw(&pos_bytes, &[n]).unwrap();
            let mut outs = Vec::new();
            for fused in [false, true] {
                let dq = gpu.zeros(&[n, hq * hd], DType::F32).unwrap();
                let dg = gpu.zeros(&[n, hq * hd], DType::F32).unwrap();
                let dk = gpu.upload_f32(&kk, &[n, hk * hd]).unwrap();
                if fused {
                    gpu.qwen35_fa_prep_batched_gfx1100(&dqf, &dq, &dg, &dk, &dqw, &dkw, &dpos, 1e-6, 1.0e7, 0, hq, hk, n)
                        .expect("fa prep fused");
                } else {
                    gpu.deinterleave_q_rmsnorm_f32_batched(&dqf, &dq, &dg, &dqw, hq, hd, n, 1e-6).expect("deint");
                    gpu.rmsnorm_batched(&dk, &dkw, &dk, n * hk, hd, 1e-6).expect("knorm");
                    gpu.rope_partial_interleaved_f32_batched(&dq, &dk, &dpos, hq, hk, hd, n_rot, 1.0e7, n, 0)
                        .expect("rope");
                }
                gpu.hip.device_synchronize().unwrap();
                outs.push((download_bytes(&gpu, &dq), download_bytes(&gpu, &dg), download_bytes(&gpu, &dk)));
            }
            let same = outs[0] == outs[1];
            println!("FA prep batched vs deinterleave+knorm+rope: {}", if same { "BIT-EXACT" } else { "MISMATCH" });
        }
    }

    // ---- GDN quad-row vs per-segment fast launches (in-process).
    {
        let (h, hd, n, seg) = (48usize, 128usize, 1024usize, 512usize);
        let tok = h * hd;
        let q: Vec<f32> = (0..n * tok).map(|_| rng.f(-0.1, 0.1)).collect();
        let kk: Vec<f32> = (0..n * tok).map(|_| rng.f(-0.2, 0.2)).collect();
        let v: Vec<f32> = (0..n * tok).map(|_| rng.f(-1.0, 1.0)).collect();
        let g: Vec<f32> = (0..n * h).map(|_| -rng.f(0.0, 1.5)).collect();
        let b: Vec<f32> = (0..n * h).map(|_| rng.f(0.05, 0.95)).collect();
        let s8: Vec<u8> = (0..h * hd * hd).map(|_| rng.next() as u8).collect();
        let sc: Vec<f32> = (0..h * hd).map(|_| rng.f(0.002, 0.02)).collect();
        let ef: Vec<u8> = (0..h * hd * hd)
            .flat_map(|_| f16_bits(rng.f(-1e-4, 1e-4)).to_le_bytes())
            .collect();
        let dq = gpu.upload_f32(&q, &[n, tok]).expect("q");
        let dk = gpu.upload_f32(&kk, &[n, tok]).expect("k");
        let dv = gpu.upload_f32(&v, &[n, tok]).expect("v");
        let dg = gpu.upload_f32(&g, &[n, h]).expect("g");
        let db = gpu.upload_f32(&b, &[n, h]).expect("b");
        let mut arms = Vec::new();
        for _ in 0..2 {
            arms.push((
                gpu.upload_raw(&s8, &[s8.len()]).expect("s8"),
                gpu.upload_f32(&sc, &[sc.len()]).expect("sc"),
                gpu.upload_raw(&ef, &[ef.len()]).expect("ef"),
                gpu.zeros(&[n, tok], DType::F32).expect("out"),
            ));
        }
        // Arm 0: production cadence (one fast launch per 512-row segment).
        {
            let (s8t, sct, eft, outt) = &arms[0];
            for off in (0..n).step_by(seg) {
                let sn = (n - off).min(seg);
                let qv = dq.sub_offset(off * tok, sn * tok);
                let kv = dk.sub_offset(off * tok, sn * tok);
                let vv = dv.sub_offset(off * tok, sn * tok);
                let gv = dg.sub_offset(off * h, sn * h);
                let bv = db.sub_offset(off * h, sn * h);
                let ov = outt.sub_offset(off * tok, sn * tok);
                gpu.gated_delta_net_q8_batch_seq(&qv, &kv, &vv, &gv, &bv, s8t, sct, &ov, sn, h, hd, Some(eft))
                    .expect("fast");
            }
        }
        // Arm 1: one quad-row launch.
        {
            let (s8t, sct, eft, outt) = &arms[1];
            let ran = gpu
                .gated_delta_net_q8_quadrow_segments(&dq, &dk, &dv, &dg, &db, s8t, sct, outt, n, h, hd, seg, Some(eft))
                .expect("quadrow");
            assert!(ran, "quadrow path not eligible (CLAUDE_GDN_QUADROW=0?)");
        }
        gpu.hip.device_synchronize().expect("sync");
        let mut all = true;
        for (name, i) in [("q8", 0usize), ("scales", 1), ("ef", 2), ("out", 3)] {
            let pick = |a: &(GpuTensor, GpuTensor, GpuTensor, GpuTensor)| match i {
                0 => download_bytes(&gpu, &a.0),
                1 => download_bytes(&gpu, &a.1),
                2 => download_bytes(&gpu, &a.2),
                _ => download_bytes(&gpu, &a.3),
            };
            let (x, y) = (pick(&arms[0]), pick(&arms[1]));
            let same = x == y;
            all &= same;
            eprintln!("  gdn {name}: {}", if same { "bit-identical" } else { "MISMATCH" });
        }
        println!("GDN {}", if all { "BIT-EXACT" } else { "MISMATCH" });
    }
}
