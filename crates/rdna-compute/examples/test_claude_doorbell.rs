// claude r2 (CLAUDE_MTP_DOORBELL): doorbell readback ordering/visibility test
// and latency comparison against the blocking hipMemcpy the MTP loop uses.
//
//  1. Ordering: N iterations; a producer kernel writes 8 iteration-dependent
//     words into device memory, `claude_mtp_publish` copies them into a pinned
//     host slot and rings the sequence word, the host waits and checks all 8
//     words (a stale or torn read = FAIL). Host early (a 20-80 us spin kernel
//     in front), host late (synchronize before the wait) and back-to-back
//     iterations are mixed. The device-side extras (next position, token
//     through a vocab map) are checked as well.
//  2. Latency: per-iteration wall time of [spin 100 us, producer, readback of
//     16 B] with the blocking hipMemcpy D2H vs the doorbell.
//  3. GPU time per op (queued): hipStreamWriteValue32 vs claude_mtp_stage.
//
// Usage: test_claude_doorbell [iters=10000]
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::claude_doorbell::ClaudeDoorbell;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const PRODUCER_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void claude_db_producer(unsigned int* __restrict__ dst, unsigned int it) {
    const unsigned int j = threadIdx.x;
    if (j < 8) dst[j] = it * 1000003u + j * 7919u + 17u;
}
"#;

const TS_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void claude_db_spin_ts(unsigned long long ticks, unsigned long long* ts, int it) {
    const unsigned long long t0 = wall_clock64();
    if (threadIdx.x == 0) ts[4 * it] = t0;
    unsigned long long t = t0;
    while ((t = wall_clock64()) - t0 < ticks) { __builtin_amdgcn_s_sleep(2); }
    if (threadIdx.x == 0) ts[4 * it + 1] = t;
}
extern "C" __global__ void claude_db_prod_ts(unsigned int* __restrict__ dst, unsigned int it, unsigned long long* ts, int i) {
    const unsigned int j = threadIdx.x;
    if (j < 8) dst[j] = it * 1000003u + j * 7919u + 17u;
    if (j == 0) ts[4 * i + 2] = wall_clock64();
}
"#;

fn producer(gpu: &mut Gpu, dst: &rdna_compute::GpuTensor, it: u32) {
    let mut b = hip_bridge::KernargBlob::new();
    b.push_ptr(dst.buf.as_ptr());
    b.push_u32(it);
    gpu.launch_kernel_blob("claude_db_producer", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice())
        .unwrap();
}

fn expect(it: u32, j: u32) -> u32 {
    it.wrapping_mul(1000003).wrapping_add(j * 7919).wrapping_add(17)
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let mut gpu = Gpu::init().expect("gpu init");
    gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    println!("test_claude_doorbell: arch={} iters={iters}", gpu.arch);
    gpu.ensure_kernel_public("claude_db_producer", PRODUCER_SRC, "claude_db_producer").unwrap();
    let src = gpu.zeros(&[16], DType::F32).unwrap();
    let pos = gpu.zeros(&[4], DType::F32).unwrap();
    let tok = gpu.zeros(&[4], DType::F32).unwrap();
    // vocab map: map[i] = 3*i + 5 for i < 4096
    let map_host: Vec<u32> = (0..4096u32).map(|i| 3 * i + 5).collect();
    let map_bytes: Vec<u8> = map_host.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let map = gpu.upload_raw(&map_bytes, &[map_bytes.len()]).unwrap();
    // a second source whose word 0 is a small index (for the mapped-token path)
    let isrc = gpu.zeros(&[16], DType::F32).unwrap();
    let mut db = ClaudeDoorbell::new(&gpu).expect("doorbell alloc");

    // 1. ordering / visibility
    let mut bad = 0usize;
    let mut out = [0u32; 8];
    let mut rng = Lcg(0xd00_4be11);
    let (mut early, mut late, mut b2b) = (0, 0, 0);
    let t_all = Instant::now();
    for it in 0..iters as u32 {
        let mode = it % 3;
        if mode == 0 {
            spin(&mut gpu, 20 + (rng.next() % 60) as u64);
            early += 1;
        }
        producer(&mut gpu, &src, it);
        let tk = gpu
            .claude_mtp_publish(&mut db, &src, 8, Some((pos.buf.as_ptr(), it as i32)), None)
            .unwrap();
        if mode == 1 && it % 2 == 0 {
            gpu.hip.stream_synchronize(gpu.active_stream.as_ref().unwrap()).unwrap();
            late += 1;
        } else if mode != 0 {
            b2b += 1;
        }
        gpu.claude_doorbell_wait(&mut db, &tk, &mut out).unwrap();
        for j in 0..8u32 {
            if out[j as usize] != expect(it, j) {
                bad += 1;
                if bad < 10 {
                    eprintln!("stale/torn: it {it} word {j}: got {:#x} want {:#x}", out[j as usize], expect(it, j));
                }
            }
        }
    }
    let secs = t_all.elapsed().as_secs_f64();
    gpu.hip.device_synchronize().unwrap();
    let pos_last = bits(&gpu, &pos)[0];
    // mapped token path: isrc[0] = 1234 -> tok = map[1234] = 3707
    {
        let v: [u32; 1] = [1234];
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&isrc.buf, &b).unwrap();
    }
    let tk = gpu
        .claude_mtp_publish(&mut db, &isrc, 1, None, Some((map.buf.as_ptr(), 0, tok.buf.as_ptr())))
        .unwrap();
    gpu.claude_doorbell_wait(&mut db, &tk, &mut out[..1]).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let tok_v = bits(&gpu, &tok)[0];
    println!(
        "{} ordering: {iters} publishes ({early} host-early, {late} host-late, {b2b} back-to-back) in {secs:.2}s, stale/torn words {bad}; stats fast/slow/fallback {:?}; pos extra {} (want {}), mapped token {} (want 3707, payload {})",
        if bad == 0 && pos_last == iters as u32 - 1 && tok_v == 3707 && db.stats.2 == 0 { "PASS" } else { "FAIL" },
        db.stats,
        pos_last,
        iters - 1,
        tok_v,
        out[0]
    );

    // 2. latency: [spin 100 us, producer, 16-B readback] per iteration
    let reps = 400usize;
    let mut run = |gpu: &mut Gpu, db: &mut ClaudeDoorbell, doorbell: bool| -> f64 {
        let mut ts = Vec::with_capacity(reps);
        let mut o = [0u32; 4];
        for it in 0..reps as u32 {
            let t0 = Instant::now();
            spin(gpu, 100);
            producer(gpu, &src, it);
            if doorbell {
                let tk = gpu.claude_mtp_publish(db, &src, 4, None, None).unwrap();
                gpu.claude_doorbell_wait(db, &tk, &mut o).unwrap();
            } else {
                let mut raw = [0u8; 16];
                gpu.hip.memcpy_dtoh(&mut raw, &src.buf).unwrap();
                o[0] = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
            }
            assert_eq!(o[0], expect(it, 0));
            ts.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2]
    };
    let _ = run(&mut gpu, &mut db, false);
    let blocking = run(&mut gpu, &mut db, false);
    let door = run(&mut gpu, &mut db, true);
    let blocking2 = run(&mut gpu, &mut db, false);
    let door2 = run(&mut gpu, &mut db, true);
    println!(
        "latency (median wall per [spin 100 us + producer + 16-B readback]): blocking hipMemcpy {blocking:.1} / {blocking2:.1} us, doorbell {door:.1} / {door2:.1} us -> saving {:.1} us per readback",
        (blocking + blocking2 - door - door2) / 2.0
    );

    // 2b. GPU-clock gap: producer end -> next iteration's first kernel start,
    // i.e. the whole host round trip (readback + relaunch) as the GPU sees it.
    gpu.ensure_kernel_public("claude_db_ts", TS_SRC, "claude_db_spin_ts").unwrap();
    gpu.ensure_kernel_public("claude_db_ts", TS_SRC, "claude_db_prod_ts").unwrap();
    let ts = gpu.zeros(&[4 * 1024], DType::F32).unwrap(); // u64 [n][2]
    let mut gap_run = |gpu: &mut Gpu, db: &mut ClaudeDoorbell, mode: u32| -> (f64, f64, f64) {
        let n = if std::env::var("CLAUDE_DB_TRACE").is_ok() { 6usize } else { 300usize };
        let mut o = [0u32; 4];
        let mut enq = Vec::new();
        let mut wait = Vec::new();
        for it in 0..n {
            let t0 = Instant::now();
            let mut b = hip_bridge::KernargBlob::new();
            b.push_u64(100 * 100);
            b.push_ptr(ts.buf.as_ptr());
            b.push_i32(it as i32);
            gpu.launch_kernel_blob("claude_db_spin_ts", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice()).unwrap();
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(src.buf.as_ptr());
            b.push_u32(it as u32);
            b.push_ptr(ts.buf.as_ptr());
            b.push_i32(it as i32);
            gpu.launch_kernel_blob("claude_db_prod_ts", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice()).unwrap();
            let tk = if (1..=5).contains(&mode) { Some(gpu.claude_mtp_publish(db, &src, 4, None, None).unwrap()) } else { None };
            let t1 = Instant::now();
            match mode {
                1 => gpu.claude_doorbell_wait(db, tk.as_ref().unwrap(), &mut o).unwrap(),
                2 => {
                    gpu.claude_doorbell_wait(db, tk.as_ref().unwrap(), &mut o).unwrap();
                    let st = gpu.active_stream.as_ref().unwrap();
                    let _ = gpu.hip.stream_query(st).unwrap();
                }
                3 => {
                    gpu.claude_doorbell_wait(db, tk.as_ref().unwrap(), &mut o).unwrap();
                    let st = gpu.active_stream.as_ref().unwrap();
                    gpu.hip.stream_synchronize(st).unwrap();
                }
                4 => {
                    // publish enqueued, but the host stream-syncs first and reads the slot after
                    let st = gpu.active_stream.as_ref().unwrap();
                    gpu.hip.stream_synchronize(st).unwrap();
                    gpu.claude_doorbell_wait(db, tk.as_ref().unwrap(), &mut o).unwrap();
                }
                5 => {
                    // publish enqueued (unused), read with the blocking memcpy
                    let mut raw = [0u8; 16];
                    gpu.hip.memcpy_dtoh(&mut raw, &src.buf).unwrap();
                    o[0] = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                    let _ = tk;
                }
                6 => {
                    // no publish: plain stream sync then the blocking memcpy (reference)
                    let st = gpu.active_stream.as_ref().unwrap();
                    gpu.hip.stream_synchronize(st).unwrap();
                    let mut raw = [0u8; 16];
                    gpu.hip.memcpy_dtoh(&mut raw, &src.buf).unwrap();
                    o[0] = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                }
                7 => {
                    // no publish: stream sync, then read the source through the doorbell's
                    // fallback-free path = a plain host read is impossible; use memcpy after sync
                    let st = gpu.active_stream.as_ref().unwrap();
                    gpu.hip.stream_synchronize(st).unwrap();
                    o[0] = expect(it as u32, 0);
                }
                _ => {
                    let mut raw = [0u8; 16];
                    gpu.hip.memcpy_dtoh(&mut raw, &src.buf).unwrap();
                    o[0] = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                }
            }
            let t2 = Instant::now();
            assert_eq!(o[0], expect(it as u32, 0));
            enq.push((t1 - t0).as_secs_f64() * 1e6);
            wait.push((t2 - t1).as_secs_f64() * 1e6);
        }
        gpu.hip.device_synchronize().unwrap();
        let mut raw = vec![0u8; n * 32];
        gpu.hip.memcpy_dtoh(&mut raw, &ts.buf).unwrap();
        let w = |i: usize| u64::from_ne_bytes(raw[8 * i..8 * i + 8].try_into().unwrap()) as f64 / 100.0;
        // ts layout per it: [spin_start, spin_end, prod_time, -]
        let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let spin_len = med((0..n).map(|i| w(4 * i + 1) - w(4 * i)).collect());
        let spin_to_prod = med((0..n).map(|i| w(4 * i + 2) - w(4 * i + 1)).collect());
        let prod_to_next = med((0..n - 1).map(|i| w(4 * (i + 1)) - w(4 * i + 2)).collect());
        let period = med((0..n - 1).map(|i| w(4 * (i + 1)) - w(4 * i)).collect());
        println!("    GPU clock: spin {spin_len:.1} us, spin end -> producer {spin_to_prod:.1} us, producer -> next spin start {prod_to_next:.1} us, period {period:.1} us");
        enq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        wait.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (prod_to_next, enq[n / 2], wait[n / 2])
    };
    if std::env::var("CLAUDE_DB_TRACE").is_ok() {
        for mode in [0u32, 1] {
            eprintln!("=== TRACE mode {mode}");
            let _ = gap_run(&mut gpu, &mut db, mode);
        }
        return;
    }
    let _ = gap_run(&mut gpu, &mut db, 0);
    for (mode, name) in [(0u32, "blocking hipMemcpy"), (1, "doorbell"), (2, "doorbell+streamQuery"), (3, "doorbell+streamSync"), (4, "publish,streamSync"), (5, "publish,memcpy"), (6, "streamSync,memcpy"), (7, "streamSync only"), (0, "blocking hipMemcpy"), (1, "doorbell")] {
        let (g, e, w) = gap_run(&mut gpu, &mut db, mode);
        println!("GPU gap producer-end -> next kernel start: {name:18} median {g:6.1} us (host enqueue {e:5.1} us, host wait {w:6.1} us)");
    }

    // 2c. Realistic draft-step tail: GEMV over a 32768 x 5120 MQ4G256V2 matrix
    // (89 MB, the compressed draft lm_head), two-stage top-2 + logsumexp into a
    // 4-word pack, then the 16-B readback (blocking hipMemcpy vs doorbell).
    // Wall time per iteration; the GEMV keeps the whole GPU busy as in the
    // real step (unlike the single-wave spin above).
    {
        let (m, k) = (32768usize, 5120usize);
        let bytes = m * (k / 256) * 136;
        let mut lcg = Lcg(0x5eed);
        let w = gpu.upload_raw(&lcg.bytes(bytes), &[bytes]).unwrap();
        let x = up(&mut gpu, &lcg.acts(k, 1.0, 0));
        let y = gpu.zeros(&[m], DType::F32).unwrap();
        let pack = gpu.zeros(&[4], DType::F32).unwrap();
        let step = |gpu: &mut Gpu| {
            gpu.gemv_mq4g256v2(&w, &x, &y, m, k).unwrap();
            gpu.claude_top2lse_any_n(&y, &pack.sub_offset(0, 2), &pack.sub_offset(2, 2), m, 1).unwrap();
        };
        let mut run2 = |gpu: &mut Gpu, db: &mut ClaudeDoorbell, doorbell: bool| -> f64 {
            let mut ts = Vec::new();
            let mut o = [0u32; 4];
            for _ in 0..300 {
                let t0 = Instant::now();
                step(gpu);
                if doorbell {
                    let tk = gpu.claude_mtp_publish(db, &pack, 4, None, None).unwrap();
                    gpu.claude_doorbell_wait(db, &tk, &mut o).unwrap();
                } else {
                    let mut raw = [0u8; 16];
                    gpu.hip.memcpy_dtoh(&mut raw, &pack.buf).unwrap();
                    o[0] = u32::from_ne_bytes([raw[0], raw[1], raw[2], raw[3]]);
                }
                ts.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ts[ts.len() / 2]
        };
        let _ = run2(&mut gpu, &mut db, false);
        let a1 = run2(&mut gpu, &mut db, false);
        let b1 = run2(&mut gpu, &mut db, true);
        let a2 = run2(&mut gpu, &mut db, false);
        let b2 = run2(&mut gpu, &mut db, true);
        // GPU-only time of the step (queued back to back, no readback)
        let t_gpu = time_us(&mut gpu, 50, |g| step(g));
        println!(
            "draft-tail loop [GEMV 89 MB + top-2 + 16-B readback], median wall/iter: blocking {a1:.1} / {a2:.1} us, doorbell {b1:.1} / {b2:.1} us (GPU-only step {t_gpu:.1} us) -> saving {:.1} us per readback",
            (a1 + a2 - b1 - b2) / 2.0
        );
        let _ = gpu.free_tensor(w);
    }

    // 3. GPU cost per op, queued behind a spin
    let reps3 = 200;
    let t_write = time_us(&mut gpu, reps3, |g| {
        let st = g.active_stream.as_ref().unwrap();
        g.hip.stream_write_value32(st, &pos.buf, 42, 0).unwrap();
    });
    let pp = pos.buf.as_ptr();
    let t_stage = time_us(&mut gpu, reps3, |g| {
        g.claude_mtp_stage(None, None, Some((pp, 42))).unwrap();
    });
    let t_stage_full = time_us(&mut gpu, reps3, |g| {
        g.claude_mtp_stage(Some((src.buf.as_ptr(), 1000, 4)), Some((tok.buf.as_ptr(), 7)), Some((pp, 42)))
            .unwrap();
    });
    let t_pub = time_us(&mut gpu, reps3, |g| {
        let _ = g.claude_mtp_publish(&mut db, &src, 4, Some((pp, 1)), None).unwrap();
    });
    gpu.hip.device_synchronize().unwrap();
    // check the stage kernel's writes
    let sv = bits(&gpu, &src);
    let ok_stage = sv[0..4] == [1000, 1001, 1002, 1003] && bits(&gpu, &tok)[0] == 7 && bits(&gpu, &pos)[0] == 1;
    println!(
        "GPU time per op (queued, events): hipStreamWriteValue32 {t_write:.2} us | claude_mtp_stage (1 store) {t_stage:.2} us, (ramp 4 + token + pos) {t_stage_full:.2} us | claude_mtp_publish (4 words + pos) {t_pub:.2} us | stage values {}",
        if ok_stage { "ok" } else { "WRONG" }
    );
    db.free(&gpu);
    println!("DONE");
}
