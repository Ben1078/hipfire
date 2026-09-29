// claude fusion tests: shared helpers (included with #[path]; not an example).
#![allow(dead_code)]
use rdna_compute::{DType, Gpu, GpuTensor};

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    /// [0, 1)
    pub fn unit(&mut self) -> f32 {
        self.next() as f32 / 2_147_483_648.0
    }
    pub fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    /// Activation-like values: symmetric noise with sparse outliers and one
    /// all-zero 32-element block every `zero_every` blocks (if > 0).
    pub fn acts(&mut self, n: usize, amp: f32, zero_every: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|i| {
                let x = self.sym(amp);
                if i % 97 == 13 {
                    x * 9.0
                } else {
                    x
                }
            })
            .collect();
        if zero_every > 0 {
            for blk in (0..n / 32).step_by(zero_every) {
                for x in &mut v[blk * 32..blk * 32 + 32] {
                    *x = 0.0;
                }
            }
        }
        v
    }
}

pub fn up(gpu: &mut Gpu, v: &[f32]) -> GpuTensor {
    gpu.upload_f32(v, &[v.len()]).expect("upload_f32")
}

pub fn up_i32(gpu: &mut Gpu, v: &[i32]) -> GpuTensor {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[v.len()]).expect("upload_raw i32");
    t.dtype = DType::F32; // 4-byte elements; only the raw bytes matter
    t
}

pub fn up_raw(gpu: &mut Gpu, b: &[u8]) -> GpuTensor {
    gpu.upload_raw(b, &[b.len()]).expect("upload_raw")
}

pub fn raw(gpu: &Gpu, t: &GpuTensor, nbytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; nbytes];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("dtoh");
    out
}

pub fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t).expect("download").iter().map(|v| v.to_bits()).collect()
}

pub fn mism<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

pub fn poison(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(0x7FC0_1234); n], &[n]).expect("poison")
}

const SPIN_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void claude_bench_spin(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) { __builtin_amdgcn_s_sleep(2); }
}
"#;

/// Occupy the GPU for about `us` microseconds (wall_clock64 runs at 100 MHz on
/// RDNA3) so that everything enqueued behind it is timed back to back, without
/// host launch latency leaking into the GPU timeline.
pub fn spin(gpu: &mut Gpu, us: u64) {
    gpu.ensure_kernel_public("claude_bench_spin", SPIN_SRC, "claude_bench_spin")
        .expect("spin kernel");
    let mut b = hip_bridge::KernargBlob::new();
    b.push_u64(us * 100);
    gpu.launch_kernel_blob("claude_bench_spin", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice())
        .expect("spin launch");
}

/// GPU time per iteration (us) of `f`, `reps` back-to-back iterations queued
/// behind a spin kernel and bracketed by events on the active stream. Launch
/// gaps between the kernels are included; host enqueue time is not.
pub fn time_us(gpu: &mut Gpu, reps: usize, mut f: impl FnMut(&mut Gpu)) -> f64 {
    // warm-up (JIT, caches)
    for _ in 0..3 {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let e0 = gpu.hip.event_create().unwrap();
    let e1 = gpu.hip.event_create().unwrap();
    let mut best = f64::MAX;
    for _ in 0..3 {
        spin(gpu, 4000);
        gpu.hip.event_record(&e0, gpu.active_stream.as_ref()).unwrap();
        for _ in 0..reps {
            f(gpu);
        }
        gpu.hip.event_record(&e1, gpu.active_stream.as_ref()).unwrap();
        gpu.hip.event_synchronize(&e1).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64;
        best = best.min(ms * 1000.0 / reps as f64);
    }
    gpu.hip.event_destroy(e0).unwrap();
    gpu.hip.event_destroy(e1).unwrap();
    best
}
