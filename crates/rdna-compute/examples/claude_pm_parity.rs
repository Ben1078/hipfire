// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! claude: parity + timing probe for the gfx1100 builder V2C route
//! (`CLAUDE_IU4_PM`) through the real `Gpu` dispatch. Runs the production
//! prefill GEMM shapes (N = 1024): SET (attention q/k, GDN qkv, z-fold), ADD
//! (o_proj/out K 6144, FFN down K 17408) via
//! `gemm_mq4g256v2_mmq_{set,add}_prequant_iu4`, and the F1-lite gate/up SiLU
//! via `gemm_gate_up_silu_mq4g256v2_iu4_prepared`, and dumps every output.
//! Run it with `CLAUDE_IU4_PM=0` and with the default and `cmp` the dumps.
//! With `PM_TIME=1` it also reports per-launch device time (events around
//! 8 back-to-back launches, median of 5).
//!
//! Usage: claude_pm_parity <dump-file>

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

/// MQ4G256V2 rows with the symmetric contract zp = -8*sc; bell-shaped codes.
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
            let mut v = 0u8;
            for q in 0..2 {
                let t = (rng.next() % 6 + rng.next() % 6 + rng.next() % 6) as i32 - 7;
                v |= ((t.clamp(-8, 7) + 8) as u8) << (4 * q);
            }
            *b = v;
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

fn time_us(gpu: &mut Gpu, mut launch: impl FnMut(&mut Gpu)) -> f64 {
    let e0 = gpu.hip.event_create().expect("ev");
    let e1 = gpu.hip.event_create().expect("ev");
    let mut v = Vec::new();
    for _ in 0..5 {
        gpu.hip.event_record(&e0, gpu.active_stream.as_ref()).expect("rec");
        for _ in 0..8 {
            launch(gpu);
        }
        gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).expect("rec");
        gpu.hip.event_synchronize(&e1).expect("sync");
        v.push(gpu.hip.event_elapsed_ms(&e0, &e1).expect("el") as f64 * 1000.0 / 8.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

fn main() {
    let dump_path = std::env::args().nth(1).expect("usage: claude_pm_parity <dump-file>");
    let timing = std::env::var("PM_TIME").as_deref() == Ok("1");
    let mut dump = std::fs::File::create(&dump_path).expect("dump file");
    let mut gpu = Gpu::init().expect("gpu init");
    gpu.mq4v2_symmetric = true;
    eprintln!(
        "claude_pm_parity: arch={} CLAUDE_IU4_PM={:?} CLAUDE_IU4_V2D={:?}",
        gpu.arch,
        std::env::var("CLAUDE_IU4_PM").ok(),
        std::env::var("CLAUDE_IU4_V2D").ok()
    );
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let n = 1024usize;
    for (m, k, add) in [
        (12288usize, 5120usize, false),
        (1024, 5120, false),
        (10240, 5120, false),
        (6272, 5120, false),
        (5120, 6144, true),
        (5120, 17408, true),
    ] {
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
        let us = if timing {
            time_us(&mut gpu, |g| {
                if add {
                    g.gemm_mq4g256v2_mmq_add_prequant_iu4(&a, xp, &y, m, k, n).expect("add");
                } else {
                    g.gemm_mq4g256v2_mmq_set_prequant_iu4(&a, xp, &y, m, k, n).expect("set");
                }
            })
        } else {
            0.0
        };
        eprintln!("  {} m={m} k={k} n={n}: {} bytes dumped, {us:.1} us/launch", if add { "ADD" } else { "SET" }, out.len());
        gpu.free_tensor(a).ok();
        gpu.free_tensor(xq).ok();
        gpu.free_tensor(y).ok();
    }
    // F1-lite gate/up SiLU (prepared int4 activation).
    {
        let (m, k) = (17408usize, 5120usize);
        let ag = gpu.upload_raw(&mq4v2_weights(&mut rng, m, k), &[m * k / 256 * 136]).expect("G");
        let au = gpu.upload_raw(&mq4v2_weights(&mut rng, m, k), &[m * k / 256 * 136]).expect("U");
        let h = gpu.zeros(&[n, m], DType::F32).expect("h");
        let xbytes = xq_blocks(&mut rng, k, n);
        let res = gpu.reserve_int4_mmq(k, n).expect("reserve");
        {
            let view = unsafe { hip_bridge::DeviceBuffer::from_raw(res.ptr(), xbytes.len()) };
            gpu.hip.memcpy_htod(&view, &xbytes).expect("htod");
            std::mem::forget(view);
        }
        let prep = rdna_compute::Int4MmqPrepared::from_reservation(res);
        let ran = gpu
            .gemm_gate_up_silu_mq4g256v2_iu4_prepared(&ag, &au, &prep, &h, m, m, k, n)
            .expect("gate_up");
        assert!(ran, "gate/up SiLU route not taken");
        gpu.hip.device_synchronize().expect("sync");
        let out = download_bytes(&gpu, &h);
        dump.write_all(&out).expect("write");
        let us = if timing {
            time_us(&mut gpu, |g| {
                g.gemm_gate_up_silu_mq4g256v2_iu4_prepared(&ag, &au, &prep, &h, m, m, k, n).expect("gate_up");
            })
        } else {
            0.0
        };
        eprintln!("  gate/up SiLU m=2x{m} k={k} n={n}: {} bytes dumped, {us:.1} us/launch", out.len());
    }
    eprintln!("done");
}
