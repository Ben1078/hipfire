// claude item 6 (CLAUDE_VERIFY_PRESTAGE) support measurements, not a
// byte-identity test of a kernel:
//  1. hipStreamWriteValue32 on the (blocking) active stream: values land in
//     order with kernels on that stream (a kernel reads a slot written just
//     before it and copies it out; checked for 64 slot/value pairs).
//  2. GPU time per write when queued back to back behind a spin kernel (is it
//     a blit dispatch ~ a tiny kernel, or a cheap packet?), vs a tiny kernel.
//  3. Host wall time of the blocking 16-byte memcpy_htod the verify prologue
//     issues twice per cycle, with the GPU idle and with ~300 us of queued
//     work (the snapshot copy in front of it), vs enqueueing 5 writes.
//
// Usage: test_claude_stream_write [reps]
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const COPY_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void claude_copy_slot(const int* __restrict__ src, int* __restrict__ dst, int i) {
    if (threadIdx.x == 0) dst[i] = src[i];
}
"#;

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut gpu = Gpu::init().expect("gpu init");
    gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    println!("test_claude_stream_write: arch={} reps={reps}", gpu.arch);
    gpu.ensure_kernel_public("claude_copy_slot", COPY_SRC, "claude_copy_slot").unwrap();
    let slots = gpu.zeros(&[64], DType::F32).unwrap();
    let out = gpu.zeros(&[64], DType::F32).unwrap();

    // 1. ordering: write slot i then a kernel that copies slot i, 64 times, then
    // overwrite every slot with a different value (must not affect the copies).
    spin(&mut gpu, 2000);
    for i in 0..64usize {
        let v = 1000 + 7 * i as u32;
        {
            let view = slots.sub_offset(i, 1);
            let st = gpu.active_stream.as_ref().unwrap();
            gpu.hip.stream_write_value32(st, &view.buf, v, 0).unwrap();
        }
        let mut b = hip_bridge::KernargBlob::new();
        b.push_ptr(slots.buf.as_ptr());
        b.push_ptr(out.buf.as_ptr());
        b.push_i32(i as i32);
        gpu.launch_kernel_blob("claude_copy_slot", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice()).unwrap();
        let view = slots.sub_offset(i, 1);
        let st = gpu.active_stream.as_ref().unwrap();
        gpu.hip.stream_write_value32(st, &view.buf, 0xDEAD, 0).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let got = bits(&gpu, &out);
    let bad = (0..64).filter(|&i| got[i] != 1000 + 7 * i as u32).count();
    println!(
        "{} ordering: {} of 64 kernel reads saw the value written just before them (stale/late: {bad})",
        if bad == 0 { "PASS" } else { "FAIL" },
        64 - bad
    );

    // 2. GPU cost per op, queued
    let t_write = time_us(&mut gpu, reps, |g| {
        let view = slots.sub_offset(3, 1);
        let st = g.active_stream.as_ref().unwrap();
        g.hip.stream_write_value32(st, &view.buf, 42, 0).unwrap();
    });
    let t_kernel = time_us(&mut gpu, reps, |g| {
        let mut b = hip_bridge::KernargBlob::new();
        b.push_ptr(slots.buf.as_ptr());
        b.push_ptr(out.buf.as_ptr());
        b.push_i32(3);
        g.launch_kernel_blob("claude_copy_slot", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice()).unwrap();
    });
    println!("GPU time per op (queued behind a spin, events): hipStreamWriteValue32 {t_write:6.2} us, tiny kernel {t_kernel:6.2} us");

    // 3. host wall time of the verify-prologue uploads
    let bytes16 = [7u8; 16];
    let mut idle = f64::MAX;
    let mut busy = f64::MAX;
    let mut writes5 = f64::MAX;
    for _ in 0..20 {
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        gpu.hip.memcpy_htod(&slots.buf, &bytes16).unwrap();
        idle = idle.min(t0.elapsed().as_secs_f64() * 1e6);
        spin(&mut gpu, 300);
        let t1 = Instant::now();
        gpu.hip.memcpy_htod(&slots.buf, &bytes16).unwrap();
        busy = busy.min(t1.elapsed().as_secs_f64() * 1e6);
        spin(&mut gpu, 300);
        let t2 = Instant::now();
        for i in 0..5usize {
            let view = slots.sub_offset(i, 1);
            let st = gpu.active_stream.as_ref().unwrap();
            gpu.hip.stream_write_value32(st, &view.buf, i as u32, 0).unwrap();
        }
        writes5 = writes5.min(t2.elapsed().as_secs_f64() * 1e6);
        gpu.hip.device_synchronize().unwrap();
    }
    println!(
        "host wall: blocking 16-B memcpy_htod {idle:6.1} us (GPU idle), {busy:6.1} us (behind 300 us of queued work); \
         enqueue 5 x hipStreamWriteValue32 {writes5:6.1} us (non-blocking)"
    );
    println!("DONE");
}
