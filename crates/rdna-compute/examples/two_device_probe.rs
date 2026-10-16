//! Two-device (XTX gfx1100 + Strix Halo iGPU gfx1151) hardware probe (developer
//! example, needs both GPUs visible in ONE process).
//!
//! Validates the substrate of the "iGPU as a non-rank expert engine" design:
//!   emu      HIPFIRE_EMU state (a capacity budget is single-device: refused).
//!   devices  HIP enumeration (index, arch, PCI BDF) and the two selected GPUs.
//!   portable ONE hipHostMalloc(MAPPED|PORTABLE) block; a per-device pointer from
//!            hipHostGetDevicePointer; a bitwise checksum kernel over the whole
//!            block on each device vs the host checksum.
//!   cross_write  each device writes a distinct sub-range through its own pointer;
//!            the host and then BOTH devices must see every byte.
//!   bandwidth  streaming-read GB/s per device over the block, plus the iGPU on a
//!            plain (non-portable) MAPPED block of the same size.
//!   handoff  host-ordered XTX D2H -> event sync -> iGPU zero-copy kernel -> event
//!            sync -> XTX H2D, p50/p95 over `--iters`, per leg, for 10 KB and 20 MB.
//!   concurrency  a ~50 ms XTX load vs an iGPU stream, alone vs overlapped.
//!
//! CLI: [--xtx IDX] [--igpu IDX] [--gib N=4] [--iters N=200] [--reps N=3]
//!      [--release-to-system]   (HIP_EVENT_RELEASE_TO_SYSTEM on the probe events)
//! Defaults pick the first gfx1100 and the first gfx1151 HIP device.
//! Output: one JSON object per check on stdout (`check`, `ok`, ...); exit 1 on
//! any failure.

use hip_bridge::{
    DeviceBuffer, Event, HipRuntime, KernargBlob, HIP_EVENT_DISABLE_TIMING,
    HIP_EVENT_RELEASE_TO_SYSTEM, HIP_HOST_MALLOC_MAPPED, HIP_HOST_MALLOC_PORTABLE,
};
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::ffi::c_void;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;
type Ptr = *mut c_void;
type Launch<'a> = &'a dyn Fn() -> Res<()>;

const HIP_DEVICE_ATTRIBUTE_INTEGRATED: i32 = 16;
const PORTABLE_MAPPED: u32 = HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_PORTABLE;
const MIB: usize = 1 << 20;

const SRC: &str = r#"
#include <hip/hip_runtime.h>
typedef unsigned long long u64;
__device__ __forceinline__ u64 rotl64(u64 x, unsigned r) { return r ? ((x << r) | (x >> (64 - r))) : x; }

// out[0] += sum(w), out[1] ^= xor(rotl(w, i & 63)) over p[0..n): commutative, so bitwise
// deterministic whatever the schedule.
extern "C" __global__ __launch_bounds__(256) void probe_checksum(
    const u64* __restrict__ p, u64 n, u64* __restrict__ out) {
    u64 s = 0, x = 0;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const u64 w = p[i];
        s += w;
        x ^= rotl64(w, (unsigned)(i & 63));
    }
    __shared__ u64 ss[256], sx[256];
    ss[threadIdx.x] = s; sx[threadIdx.x] = x;
    __syncthreads();
    for (unsigned o = 128; o; o >>= 1) {
        if (threadIdx.x < o) { ss[threadIdx.x] += ss[threadIdx.x + o]; sx[threadIdx.x] ^= sx[threadIdx.x + o]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(&out[0], ss[0]); atomicXor(&out[1], sx[0]); }
}

// Streaming 16-byte read, `passes` times over p[0..n16).
extern "C" __global__ __launch_bounds__(256) void probe_stream_read(
    const uint4* __restrict__ p, u64 n16, unsigned passes, u64* __restrict__ out) {
    u64 acc = 0;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (unsigned k = 0; k < passes; ++k) {
        #pragma unroll 4
        for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
            const uint4 v = p[i];
            acc += (u64)v.x ^ (u64)v.y ^ ((u64)v.z << 1) ^ ((u64)v.w << 3);
        }
    }
    if (acc == 0xDEADBEEFCAFEF00Dull) out[0] = acc;
}

extern "C" __global__ __launch_bounds__(256) void probe_write(
    u64* __restrict__ p, u64 start, u64 count, u64 tag) {
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) p[start + i] = tag + i;
}

// The handoff "expert engine": out[i] = in[i] + tag (zero-copy read, zero-copy write).
extern "C" __global__ __launch_bounds__(256) void probe_handoff(
    const u64* __restrict__ in, u64* __restrict__ out, u64 n, u64 tag) {
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) out[i] = in[i] + tag;
}

// Dependent FMA chain: pure ALU, no memory traffic.
extern "C" __global__ __launch_bounds__(256) void probe_busy(float* out, unsigned iters) {
    float a = 1.0f + threadIdx.x * 1e-6f;
    const float b = 0.9999f, c = 1e-7f;
    for (unsigned i = 0; i < iters; ++i) a = fmaf(a, b, c);
    if (a == 12345.678f) out[0] = a;
}
"#;
const FUNCS: [&str; 5] = [
    "probe_checksum",
    "probe_stream_read",
    "probe_write",
    "probe_handoff",
    "probe_busy",
];

// ---------------------------------------------------------------- host helpers

fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Run `f(a, b)` over `[0, n)` split across the host threads.
fn par<R: Send>(n: usize, f: impl Fn(usize, usize) -> R + Sync) -> Vec<R> {
    let t = std::thread::available_parallelism().map_or(4, |v| v.get()).min(32);
    let step = n.div_ceil(t).max(1);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..t)
            .map(|k| {
                let (a, b) = ((k * step).min(n), ((k + 1) * step).min(n));
                let f = &f;
                s.spawn(move || f(a, b))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("host worker")).collect()
    })
}

/// Fill `n` words at `base` with `mix(i ^ salt)`.
fn host_fill(base: Ptr, n: usize, salt: u64) {
    let base = base as usize;
    par(n, |a, b| {
        let p = base as *mut u64;
        for i in a..b {
            // SAFETY: [a, b) lies inside the caller's live n-word allocation and the
            // chunks are disjoint.
            unsafe { p.add(i).write(mix(i as u64 ^ salt)) };
        }
    });
}

/// Host reference of `probe_checksum`.
fn host_checksum(base: Ptr, n: usize) -> (u64, u64) {
    let base = base as usize;
    par(n, |a, b| {
        let p = base as *const u64;
        let (mut s, mut x) = (0u64, 0u64);
        for i in a..b {
            // SAFETY: [a, b) lies inside the caller's live n-word allocation.
            let w = unsafe { p.add(i).read() };
            s = s.wrapping_add(w);
            x ^= w.rotate_left((i & 63) as u32);
        }
        (s, x)
    })
    .into_iter()
    .fold((0, 0), |(s, x), (s2, x2)| (s.wrapping_add(s2), x ^ x2))
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s
}

/// Nearest-rank percentile of a sorted slice.
fn pct(s: &[f64], p: f64) -> f64 {
    s[((p * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1]
}

fn stats(v: &[f64]) -> Value {
    let s = sorted(v);
    json!({ "p50": pct(&s, 0.5), "p95": pct(&s, 0.95), "min": s[0], "max": s[s.len() - 1],
            "mean": v.iter().sum::<f64>() / v.len() as f64 })
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn gbps(bytes: usize, passes: u32, ms: f64) -> f64 {
    bytes as f64 * passes as f64 / 1e9 / (ms / 1e3)
}

struct Out {
    fails: u32,
}

impl Out {
    fn emit(&mut self, check: &str, ok: bool, v: Value) {
        let mut m = v.as_object().cloned().unwrap_or_default();
        m.insert("check".into(), check.into());
        m.insert("ok".into(), ok.into());
        if !ok {
            self.fails += 1;
        }
        println!("{}", Value::Object(m));
    }

    fn fail(&mut self, check: &str, e: impl std::fmt::Display) {
        self.emit(check, false, json!({ "error": e.to_string() }));
    }

    fn finish(&mut self, check: &str, r: Res<()>) {
        if let Err(e) = r {
            self.fail(check, e);
        }
    }
}

// ---------------------------------------------------------------- device

/// One opened device: its `Gpu` (own stream = `active_stream`), an event and a small
/// scratch output buffer. Every method binds the thread to the device first.
struct Dev {
    gpu: Gpu,
    ev: Event,
    out: DeviceBuffer,
}

impl Dev {
    fn open(idx: i32, ev_flags: u32) -> Res<Dev> {
        let mut gpu = Gpu::init_with_device(idx)?;
        if gpu.emu_profile().is_some() {
            return Err("HIPFIRE_EMU is active on this Gpu; unset it".into());
        }
        gpu.ensure_capture_stream()?;
        for f in FUNCS {
            gpu.ensure_kernel_public("two_device_probe", SRC, f)?;
        }
        gpu.bind_thread()?;
        let ev = gpu.hip.event_create_with_flags(ev_flags)?;
        let out = gpu.hip.malloc(64)?;
        Ok(Dev { gpu, ev, out })
    }

    fn bind(&self) -> Res<()> {
        Ok(self.gpu.bind_thread()?)
    }

    fn stream(&self) -> &hip_bridge::Stream {
        self.gpu.active_stream.as_ref().expect("stream installed by Dev::open")
    }

    fn dev_ptr(&self, host: Ptr) -> Res<Ptr> {
        self.bind()?;
        Ok(self.gpu.hip.host_get_device_pointer(host, 0)?)
    }

    fn host_alloc(&self, bytes: usize, flags: u32) -> Res<Ptr> {
        self.bind()?;
        Ok(self.gpu.hip.host_malloc(bytes, flags)?)
    }

    fn launch(&self, f: &str, blocks: u32, mut a: KernargBlob) -> Res<()> {
        a.pad_to(16);
        self.gpu.launch_kernel_blob(f, [blocks, 1, 1], [256, 1, 1], 0, a.as_mut_slice())?;
        Ok(())
    }

    fn record(&self) -> Res<()> {
        self.bind()?;
        Ok(self.gpu.hip.event_record(&self.ev, Some(self.stream()))?)
    }

    fn done(&self) -> Res<bool> {
        self.bind()?;
        Ok(self.gpu.hip.event_query(&self.ev)?)
    }

    /// Record the event and block on it (the host-ordered handoff primitive).
    fn wait(&self) -> Res<()> {
        self.record()?;
        Ok(self.gpu.hip.event_synchronize(&self.ev)?)
    }

    fn zero_out(&self) -> Res<()> {
        self.bind()?;
        Ok(self.gpu.hip.memset(&self.out, 0, 64)?)
    }

    fn checksum(&self, p: Ptr, words: usize) -> Res<(u64, u64)> {
        self.zero_out()?;
        let mut a = KernargBlob::new();
        a.push_ptr(p);
        a.push_u64(words as u64);
        a.push_ptr(self.out.as_ptr());
        self.launch("probe_checksum", 4096, a)?;
        self.wait()?;
        let mut b = [0u8; 16];
        self.gpu.hip.memcpy_dtoh(&mut b, &self.out)?;
        Ok((
            u64::from_ne_bytes(b[..8].try_into().unwrap()),
            u64::from_ne_bytes(b[8..].try_into().unwrap()),
        ))
    }

    fn write_range(&self, p: Ptr, start: usize, count: usize, tag: u64) -> Res<()> {
        let mut a = KernargBlob::new();
        a.push_ptr(p);
        a.push_u64(start as u64);
        a.push_u64(count as u64);
        a.push_u64(tag);
        self.launch("probe_write", 4096, a)?;
        self.wait()
    }

    fn stream_launch(&self, p: Ptr, bytes: usize, passes: u32) -> Res<()> {
        self.zero_out()?;
        let mut a = KernargBlob::new();
        a.push_ptr(p);
        a.push_u64((bytes / 16) as u64);
        a.push_u32(passes);
        a.push_ptr(self.out.as_ptr());
        self.launch("probe_stream_read", 8192, a)
    }

    fn busy_launch(&self, iters: u32) -> Res<()> {
        self.zero_out()?;
        let mut a = KernargBlob::new();
        a.push_ptr(self.out.as_ptr());
        a.push_u32(iters);
        self.launch("probe_busy", 1024, a)
    }

    /// Wall ms of a blocking run of `launch`.
    fn timed(&self, launch: impl FnOnce() -> Res<()>) -> Res<f64> {
        let t = Instant::now();
        launch()?;
        self.wait()?;
        Ok(ms(t))
    }
}

/// Launch the requested sides back to back and poll both events; wall ms from the
/// first launch to the first observation of each completion.
fn time_pair(x: &Dev, g: &Dev, lx: Option<Launch>, lg: Option<Launch>) -> Res<(f64, f64)> {
    let t0 = Instant::now();
    if let Some(l) = lx {
        l()?;
        x.record()?;
    }
    if let Some(l) = lg {
        l()?;
        g.record()?;
    }
    let (mut xd, mut gd) = (lx.is_none(), lg.is_none());
    let (mut xt, mut gt) = (0.0, 0.0);
    while !(xd && gd) {
        if !xd && x.done()? {
            (xd, xt) = (true, ms(t0));
        }
        if !gd && g.done()? {
            (gd, gt) = (true, ms(t0));
        }
    }
    Ok((xt, gt))
}

// ---------------------------------------------------------------- checks

fn check_portable(o: &mut Out, x: &Dev, g: &Dev, host: Ptr, bytes: usize) -> Res<(Ptr, Ptr)> {
    let words = bytes / 8;
    let t = Instant::now();
    host_fill(host, words, 0x5EED);
    let fill_ms = ms(t);
    let (hs, hx) = host_checksum(host, words);
    let (xp, gp) = (x.dev_ptr(host)?, g.dev_ptr(host)?);
    let (xs, xx) = x.checksum(xp, words)?;
    let (gs, gx) = g.checksum(gp, words)?;
    let (xm, gm) = ((xs, xx) == (hs, hx), (gs, gx) == (hs, hx));
    o.emit(
        "portable_alloc",
        xm && gm,
        json!({ "bytes": bytes, "flags": PORTABLE_MAPPED, "host_ptr": format!("{host:p}"),
                "xtx_dev_ptr": format!("{xp:p}"), "igpu_dev_ptr": format!("{gp:p}"),
                "xtx_ptr_eq_host": xp == host, "igpu_ptr_eq_host": gp == host,
                "host_sum": hs, "host_xor": hx, "xtx_sum": xs, "xtx_xor": xx,
                "igpu_sum": gs, "igpu_xor": gx, "xtx_match": xm, "igpu_match": gm,
                "fill_ms": fill_ms }),
    );
    Ok((xp, gp))
}

fn range_ok(host: Ptr, start: usize, count: usize, tag: u64) -> bool {
    let p = host as *const u64;
    // SAFETY: the caller's range lies inside the live allocation; both devices are idle.
    (0..count).all(|i| unsafe { p.add(start + i).read() } == tag + i as u64)
}

fn check_cross_write(o: &mut Out, x: &Dev, g: &Dev, host: Ptr, ptrs: (Ptr, Ptr), bytes: usize) -> Res<()> {
    let words = bytes / 8;
    let w = (64 * MIB / 8).min(words / 4);
    let (xs, gs) = (words / 4, words / 2);
    let (xtag, gtag) = (0xA11C_E000_0000_0000u64, 0xB0B0_0000_0000_0000u64);
    x.write_range(ptrs.0, xs, w, xtag)?;
    g.write_range(ptrs.1, gs, w, gtag)?;
    let (xr, gr) = (range_ok(host, xs, w, xtag), range_ok(host, gs, w, gtag));
    let (hs, hx) = host_checksum(host, words);
    let xv = x.checksum(ptrs.0, words)? == (hs, hx);
    let gv = g.checksum(ptrs.1, words)? == (hs, hx);
    o.emit(
        "cross_write",
        xr && gr && xv && gv,
        json!({ "range_words": w, "xtx_range_start_word": xs, "igpu_range_start_word": gs,
                "host_sees_xtx_range": xr, "host_sees_igpu_range": gr,
                "xtx_sees_all_bitwise": xv, "igpu_sees_all_bitwise": gv, "host_sum": hs, "host_xor": hx }),
    );
    Ok(())
}

/// Returns (xtx best full-pass ms, igpu best full-pass ms) for the concurrency sizing.
fn check_bandwidth(o: &mut Out, x: &Dev, g: &Dev, ptrs: (Ptr, Ptr), bytes: usize, reps: usize) -> Res<(f64, f64)> {
    let pass = |d: &Dev, p: Ptr| -> Res<Vec<f64>> {
        d.timed(|| d.stream_launch(p, bytes.min(64 * MIB), 1))?; // warm
        (0..reps).map(|_| d.timed(|| d.stream_launch(p, bytes, 1))).collect()
    };
    let (xm, gm) = (pass(x, ptrs.0)?, pass(g, ptrs.1)?);
    let best = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
    // Plain (non-portable) MAPPED block of the same size, iGPU only.
    let plain = g.host_alloc(bytes, HIP_HOST_MALLOC_MAPPED)?;
    host_fill(plain, bytes / 8, 0xA5);
    let pm = pass(g, g.dev_ptr(plain)?)?;
    g.gpu.hip.host_free(plain)?;
    o.emit(
        "bandwidth",
        true,
        json!({ "bytes": bytes, "unit": "GB/s (1e9 B)", "reps": reps,
                "xtx_zero_copy_gbps": gbps(bytes, 1, best(&xm)), "xtx_zero_copy_ms": xm,
                "igpu_portable_gbps": gbps(bytes, 1, best(&gm)), "igpu_portable_ms": gm,
                "igpu_plain_mapped_gbps": gbps(bytes, 1, best(&pm)), "igpu_plain_mapped_ms": pm,
                "expect": "igpu ~150-200, xtx ~6 (PCIe Gen4 x4)" }),
    );
    Ok((best(&xm), best(&gm)))
}

fn check_handoff(o: &mut Out, x: &Dev, g: &Dev, size: usize, iters: usize) -> Res<()> {
    let words = size / 8;
    let hin = x.host_alloc(size, PORTABLE_MAPPED)?;
    let hout = x.host_alloc(size, PORTABLE_MAPPED)?;
    let (gin, gout) = (g.dev_ptr(hin)?, g.dev_ptr(hout)?);
    x.bind()?;
    let dx = x.gpu.hip.malloc(size)?;
    let dres = x.gpu.hip.malloc(size)?;
    host_fill(hin, words, 0x77); // staging for the one-time upload of the XTX source
    // SAFETY: hin/hout are live `size`-byte pinned blocks, used only by the sequential
    // calls below (the devices are idle between the host-ordered steps).
    let (hin_s, hout_s) = unsafe {
        (std::slice::from_raw_parts_mut(hin as *mut u8, size), std::slice::from_raw_parts_mut(hout as *mut u8, size))
    };
    x.gpu.hip.memcpy_htod(&dx, hin_s)?;
    let mut one = |tag: u64| -> Res<[f64; 4]> {
        let t0 = Instant::now();
        x.bind()?;
        x.gpu.hip.memcpy_dtoh_async(hin_s, &dx, Some(x.stream()))?; // XTX -> pinned
        x.wait()?;
        let t1 = ms(t0);
        let mut a = KernargBlob::new();
        a.push_ptr(gin);
        a.push_ptr(gout);
        a.push_u64(words as u64);
        a.push_u64(tag);
        g.launch("probe_handoff", (words / 256).clamp(1, 4096) as u32, a)?; // iGPU zero-copy
        g.wait()?;
        let t2 = ms(t0);
        x.bind()?;
        x.gpu.hip.memcpy_htod_async(&dres, hout_s, x.stream())?; // result -> XTX
        x.wait()?;
        let t3 = ms(t0);
        Ok([t3, t1, t2 - t1, t3 - t2])
    };
    let verify = |tag: u64| -> Res<bool> {
        x.bind()?;
        let mut got = vec![0u8; size];
        x.gpu.hip.memcpy_dtoh(&mut got, &dres)?;
        Ok(got.chunks_exact(8).enumerate().all(|(i, c)| {
            u64::from_ne_bytes(c.try_into().unwrap()) == mix(i as u64 ^ 0x77).wrapping_add(tag)
        }))
    };
    for k in 0..5 {
        one(0x100 + k)?;
    }
    let mut legs: [Vec<f64>; 4] = Default::default();
    let mut verified = true;
    for k in 0..iters {
        let tag = k as u64 + 1;
        for (l, v) in legs.iter_mut().zip(one(tag)?) {
            l.push(v);
        }
        if k == 0 || k + 1 == iters {
            verified &= verify(tag)?;
        }
    }
    o.emit(
        "handoff",
        verified,
        json!({ "size_bytes": size, "iters": iters, "verified_first_last": verified,
                "roundtrip_ms": stats(&legs[0]), "xtx_d2h_ms": stats(&legs[1]),
                "igpu_kernel_ms": stats(&legs[2]), "xtx_h2d_ms": stats(&legs[3]) }),
    );
    x.gpu.hip.host_free(hin)?;
    x.gpu.hip.host_free(hout)?;
    Ok(())
}

fn check_concurrency(o: &mut Out, x: &Dev, g: &Dev, ptrs: (Ptr, Ptr), bytes: usize, bw_ms: (f64, f64), reps: usize) -> Res<()> {
    // iGPU: enough passes to cover ~60 ms. XTX: ALU chain calibrated to ~50 ms, or the
    // zero-copy prefix that streams in ~50 ms.
    let passes = (60.0 / bw_ms.1).ceil().max(1.0) as u32;
    let mut iters = 1u32 << 15;
    for _ in 0..3 {
        let t = x.timed(|| x.busy_launch(iters))?;
        iters = (iters as f64 * 50.0 / t).clamp(1.0, 4e9) as u32;
    }
    let xbytes = ((bytes as f64 * 50.0 / bw_ms.0) as usize / 16 * 16).clamp(16, bytes);
    let alu = || x.busy_launch(iters);
    let pcie = || x.stream_launch(ptrs.0, xbytes, 1);
    let lg = || g.stream_launch(ptrs.1, bytes, passes);
    let loads: [(&str, Launch); 2] = [("xtx_alu_busy_50ms", &alu), ("xtx_zero_copy_stream_50ms", &pcie)];
    for (name, lx) in loads {
        let (mut xa, mut ga, mut xo, mut go) = (vec![], vec![], vec![], vec![]);
        for _ in 0..reps {
            xa.push(time_pair(x, g, Some(lx), None)?.0);
            ga.push(time_pair(x, g, None, Some(&lg))?.1);
            let (a, b) = time_pair(x, g, Some(lx), Some(&lg))?;
            xo.push(a);
            go.push(b);
        }
        let med = |v: &[f64]| pct(&sorted(v), 0.5);
        let (xa, ga, xo, go) = (med(&xa), med(&ga), med(&xo), med(&go));
        o.emit(
            "concurrency",
            true,
            json!({ "xtx_load": name, "reps": reps, "igpu_passes": passes, "xtx_stream_bytes": xbytes,
                    "xtx_alone_ms": xa, "igpu_alone_ms": ga, "xtx_overlap_ms": xo, "igpu_overlap_ms": go,
                    "xtx_slowdown": xo / xa, "igpu_slowdown": go / ga,
                    "igpu_alone_gbps": gbps(bytes, passes, ga), "igpu_overlap_gbps": gbps(bytes, passes, go) }),
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- main

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let gib: f64 = arg(&args, "--gib", "4").parse()?;
    let iters: usize = arg(&args, "--iters", "200").parse()?;
    let reps: usize = arg(&args, "--reps", "3").parse()?;
    let bytes = ((gib * (1u64 << 30) as f64) as usize) & !4095;
    let ev_flags = HIP_EVENT_DISABLE_TIMING
        | if args.iter().any(|a| a == "--release-to-system") { HIP_EVENT_RELEASE_TO_SYSTEM } else { 0 };
    if bytes < MIB || iters == 0 || reps == 0 {
        return Err("--gib must give >= 1 MiB; --iters and --reps must be >= 1".into());
    }
    let mut o = Out { fails: 0 };

    // 1a. HIPFIRE_EMU: a capacity budget is one-physical-device-per-process.
    let env = |k: &str| std::env::var(k).ok();
    let emu = hipfire_config::emu::from_process_config(hipfire_config::active_or_local_process_config());
    let emu_active = !matches!(emu, Ok(None));
    let emu_text = match &emu {
        Ok(None) => "off".to_string(),
        Ok(Some(opts)) => format!("profile {}", opts.profile_id),
        Err(e) => format!("invalid: {e}"),
    };
    o.emit(
        "emu",
        !emu_active,
        json!({ "HIPFIRE_EMU": env("HIPFIRE_EMU"), "state": emu_text,
                "message": if emu_active {
                    "HIPFIRE_EMU capacity emulation is single-device (rdna-compute emu.rs: a second device \
                     cannot borrow the budget, second_device_refusal); unset HIPFIRE_EMU and rerun"
                } else { "emulation off; a second Gpu is allowed" } }),
    );
    if emu_active {
        std::process::exit(1);
    }

    // 1b. Enumerate (HIPFIRE_DEVICES visibility must be applied before the runtime loads).
    if let Err(e) = hipfire_config::devices::apply_entry_point_device_visibility() {
        o.fail("devices", format!("HIPFIRE_DEVICES: {e}"));
        std::process::exit(1);
    }
    let rt = HipRuntime::load()?;
    let n = rt.device_count()?;
    let mut devs = vec![];
    for i in 0..n {
        let integrated = rt.get_device_attribute(HIP_DEVICE_ATTRIBUTE_INTEGRATED, i).ok();
        devs.push(json!({ "index": i, "arch": rt.get_arch(i)?, "pci_bdf": rt.device_pci_bus_id(i)?,
                          "integrated": integrated }));
    }
    let find = |p: &str| devs.iter().position(|d| d["arch"].as_str().unwrap_or("").starts_with(p)).map(|i| i as i32);
    let pick = |flag: &str, p: &str| match arg(&args, flag, "").as_str() {
        "" => find(p),
        s => s.parse().ok(),
    };
    let (xi, gi) = (pick("--xtx", "gfx1100"), pick("--igpu", "gfx1151"));
    let sel = match (xi, gi) {
        (Some(a), Some(b)) if a != b && (0..n).contains(&a) && (0..n).contains(&b) => Some((a, b)),
        _ => None,
    };
    o.emit(
        "devices",
        sel.is_some(),
        json!({ "count": n, "devices": devs, "xtx": xi, "igpu": gi,
                "env": { "HIPFIRE_DEVICES": env("HIPFIRE_DEVICES"), "HIP_VISIBLE_DEVICES": env("HIP_VISIBLE_DEVICES"),
                         "ROCR_VISIBLE_DEVICES": env("ROCR_VISIBLE_DEVICES") },
                "error": if sel.is_some() { Value::Null } else {
                    "need two distinct valid devices: pass --xtx/--igpu or make a gfx1100 and a gfx1151 visible".into() } }),
    );
    let Some((xi, gi)) = sel else { std::process::exit(1) };
    let (x, g) = match (Dev::open(xi, ev_flags), Dev::open(gi, ev_flags)) {
        (Ok(x), Ok(g)) => (x, g),
        (Err(e), _) | (_, Err(e)) => {
            o.fail("open_devices", format!("{e} (HIPFIRE_EMU: {emu_text})"));
            std::process::exit(1);
        }
    };
    o.emit(
        "open_devices",
        true,
        json!({ "xtx": { "index": xi, "arch": x.gpu.arch }, "igpu": { "index": gi, "arch": g.gpu.arch } }),
    );

    // 2. Portable mapped block + per-device pointers.
    let host = x.host_alloc(bytes, PORTABLE_MAPPED).and_then(|h| Ok((h, check_portable(&mut o, &x, &g, h, bytes)?)));
    let (host, ptrs) = match host {
        Ok(v) => v,
        Err(e) => {
            o.fail("portable_alloc", e);
            std::process::exit(1);
        }
    };
    let r = check_cross_write(&mut o, &x, &g, host, ptrs, bytes);
    o.finish("cross_write", r);
    // 3. Bandwidth.
    let bw = match check_bandwidth(&mut o, &x, &g, ptrs, bytes, reps) {
        Ok(b) => Some(b),
        Err(e) => {
            o.fail("bandwidth", e);
            None
        }
    };
    // 4. Host-ordered handoff.
    for size in [10 * 1024, 20 * MIB] {
        let r = check_handoff(&mut o, &x, &g, size, iters);
        o.finish("handoff", r);
    }
    // 5. Concurrency.
    match bw {
        Some(bw) => {
            let r = check_concurrency(&mut o, &x, &g, ptrs, bytes, bw, reps);
            o.finish("concurrency", r);
        }
        None => o.fail("concurrency", "skipped: bandwidth check failed"),
    }

    if let Err(e) = x.gpu.hip.host_free(host) {
        o.fail("cleanup", e);
    }
    let fails = o.fails;
    o.emit("summary", fails == 0, json!({ "failed_checks": fails }));
    drop(x);
    drop(g);
    drop(rt);
    if o.fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}
