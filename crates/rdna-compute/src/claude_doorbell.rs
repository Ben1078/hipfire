//! claude r2: doorbell readbacks and argument-staged scalars for the MTP loop
//! (`CLAUDE_MTP_DOORBELL`). See `kernels/src/claude_mtp_doorbell.hip`.
//!
//! A [`ClaudeDoorbell`] owns 4 KiB of pinned, device-mapped, fine-grained
//! host memory split into 16 slots of 64 words. [`Gpu::claude_mtp_publish`]
//! enqueues a one-thread kernel on the active stream that copies up to 63
//! words of device memory into a slot and then stores a fresh sequence number
//! into the slot header; [`Gpu::claude_doorbell_wait`] spins on that header
//! (pause for ~50 us, then `yield_now`) and reads the payload. On a timeout
//! (default 250 ms, `CLAUDE_DOORBELL_TIMEOUT_MS`) it falls back to a real
//! stream synchronize and, if the sequence word still has not arrived, to the
//! blocking `hipMemcpy` of the source words; that last case disables the
//! doorbell for the rest of the process.
use crate::{Gpu, GpuTensor};
use hip_bridge::{HipResult, HostMappedBuffer, KernargBlob};
use std::ffi::c_void;
use std::sync::atomic::{fence, Ordering};

const SRC: &str = include_str!("../../../kernels/src/claude_mtp_doorbell.hip");
const SLOT_WORDS: usize = 64;
const N_SLOTS: usize = 16;

/// Pinned host slots + sequence state for doorbell readbacks.
pub struct ClaudeDoorbell {
    mem: Option<HostMappedBuffer>,
    seq: u32,
    next_slot: usize,
    /// Set after a doorbell failed to arrive even after a stream synchronize.
    pub disabled: bool,
    /// (fast: seen while spinning, slow: needed the synchronize fallback,
    /// fallback: needed the blocking memcpy)
    pub stats: (u64, u64, u64),
}

/// One outstanding publish.
#[derive(Clone, Copy, Debug)]
pub struct DoorbellTicket {
    slot: usize,
    seq: u32,
    n_words: usize,
    src: *mut c_void,
}

impl ClaudeDoorbell {
    pub fn new(gpu: &Gpu) -> HipResult<Self> {
        let mem = gpu.hip.host_malloc_mapped(SLOT_WORDS * N_SLOTS * 4, true)?;
        Ok(Self { mem: Some(mem), seq: 0, next_slot: 0, disabled: false, stats: (0, 0, 0) })
    }

    pub fn free(mut self, gpu: &Gpu) {
        if let Some(m) = self.mem.take() {
            // Callers free only after the stream drained (state teardown).
            let _ = gpu.hip.host_free_mapped(m);
        }
    }

    fn slot_host(&self, slot: usize) -> *const u32 {
        let m = self.mem.as_ref().expect("doorbell freed");
        unsafe { (m.host_ptr() as *const u32).add(slot * SLOT_WORDS) }
    }

    fn slot_dev(&self, slot: usize) -> *mut c_void {
        let m = self.mem.as_ref().expect("doorbell freed");
        unsafe { (m.dev_ptr() as *mut u32).add(slot * SLOT_WORDS) as *mut c_void }
    }
}

/// `CLAUDE_MTP_DOORBELL` (default on): replace the MTP loop's blocking 4-16 B
/// `hipMemcpy` readbacks by doorbell slots.
pub fn claude_mtp_doorbell_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_MTP_DOORBELL").as_deref() != Ok("0"))
}

fn timeout_ms() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CLAUDE_DOORBELL_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &u64| v >= 1)
            .unwrap_or(250)
    })
}

impl Gpu {
    fn claude_doorbell_kernels(&mut self) -> HipResult<()> {
        if !self.functions.contains_key("claude_mtp_publish") {
            self.ensure_kernel("claude_mtp_doorbell", SRC, "claude_mtp_publish")?;
            self.ensure_kernel("claude_mtp_doorbell", SRC, "claude_mtp_stage")?;
        }
        Ok(())
    }

    /// Enqueue a doorbell publish of `n_words` (<= 63) 32-bit words starting
    /// at `src` (device memory, typically written by the previous kernel on
    /// the same stream). Optional device-side extras run in the same launch:
    /// `pos = Some((dst, v))` stores `v` into `dst[0]`; `tok = Some((map,
    /// word, dst))` stores `map[src[word]]` (or `src[word]` when `map` is
    /// null) into `dst[0]`.
    pub fn claude_mtp_publish(
        &mut self,
        db: &mut ClaudeDoorbell,
        src: &GpuTensor,
        n_words: usize,
        pos: Option<(*mut c_void, i32)>,
        tok: Option<(*const c_void, usize, *mut c_void)>,
    ) -> HipResult<DoorbellTicket> {
        assert!(n_words >= 1 && n_words < SLOT_WORDS, "doorbell payload {n_words} words");
        assert!(src.buf.size() >= n_words * 4, "doorbell source too small");
        self.bind_thread()?;
        self.claude_doorbell_kernels()?;
        db.seq = db.seq.wrapping_add(1).max(1);
        let slot = db.next_slot;
        db.next_slot = (db.next_slot + 1) % N_SLOTS;
        // A slot is reused only after N_SLOTS later publishes; the host reads
        // every ticket before issuing the next publish in the MTP loop.
        unsafe {
            let hdr = db.slot_host(slot) as *mut u32;
            std::ptr::write_volatile(hdr, 0);
        }
        let (pos_dst, pos_val) = pos.unwrap_or((std::ptr::null_mut(), 0));
        let (tok_map, tok_word, tok_dst) = tok.unwrap_or((std::ptr::null(), 0, std::ptr::null_mut()));
        let mut b = KernargBlob::new();
        b.push_ptr(src.buf.as_ptr());
        b.push_i32(n_words as i32);
        b.push_u64(db.slot_dev(slot) as u64);
        b.push_u32(db.seq);
        b.push_ptr(pos_dst);
        b.push_i32(pos_val);
        b.push_ptr(tok_map as *mut c_void);
        b.push_i32(tok_word as i32);
        b.push_ptr(tok_dst);
        self.launch_kernel_blob("claude_mtp_publish", [1, 1, 1], [32, 1, 1], 0, b.as_mut_slice())?;
        Ok(DoorbellTicket { slot, seq: db.seq, n_words, src: src.buf.as_ptr() })
    }

    /// Reserve the next slot for a kernel that rings it itself (the sampling
    /// kernels): returns the slot's device address, the sequence number the
    /// kernel must store last, and the ticket to wait on. `src` holds the
    /// same `n_words` in device memory (read by the blocking fallback).
    pub fn claude_doorbell_reserve(
        &mut self,
        db: &mut ClaudeDoorbell,
        src: &GpuTensor,
        n_words: usize,
    ) -> (u64, u32, DoorbellTicket) {
        assert!(n_words >= 1 && n_words < SLOT_WORDS);
        db.seq = db.seq.wrapping_add(1).max(1);
        let slot = db.next_slot;
        db.next_slot = (db.next_slot + 1) % N_SLOTS;
        unsafe {
            let hdr = db.slot_host(slot) as *mut u32;
            std::ptr::write_volatile(hdr, 0);
        }
        (
            db.slot_dev(slot) as u64,
            db.seq,
            DoorbellTicket { slot, seq: db.seq, n_words, src: src.buf.as_ptr() },
        )
    }

    /// Wait for a published slot and copy its payload into `out`
    /// (`out.len() <= n_words`).
    pub fn claude_doorbell_wait(
        &mut self,
        db: &mut ClaudeDoorbell,
        t: &DoorbellTicket,
        out: &mut [u32],
    ) -> HipResult<()> {
        assert!(out.len() <= t.n_words);
        let hdr = db.slot_host(t.slot);
        let t0 = std::time::Instant::now();
        let limit = std::time::Duration::from_millis(timeout_ms());
        let mut spins = 0u32;
        let read = |out: &mut [u32]| {
            fence(Ordering::Acquire);
            for (i, o) in out.iter_mut().enumerate() {
                *o = unsafe { std::ptr::read_volatile(hdr.add(1 + i)) };
            }
        };
        loop {
            if unsafe { std::ptr::read_volatile(hdr) } == t.seq {
                read(out);
                db.stats.0 += 1;
                return Ok(());
            }
            spins += 1;
            if spins < 2048 {
                std::hint::spin_loop();
            } else if t0.elapsed() < limit {
                std::thread::yield_now();
            } else {
                break;
            }
        }
        // Slow path: a real synchronize, then look again.
        match self.active_stream.as_ref() {
            Some(s) => self.hip.stream_synchronize(s)?,
            None => self.hip.device_synchronize()?,
        }
        if unsafe { std::ptr::read_volatile(hdr) } == t.seq {
            read(out);
            db.stats.1 += 1;
            return Ok(());
        }
        // The doorbell never rang although the stream drained: read the source
        // the old way and stop using the doorbell.
        let mut bytes = vec![0u8; out.len() * 4];
        let src = unsafe { hip_bridge::DeviceBuffer::from_raw(t.src, t.n_words * 4) };
        self.hip.memcpy_dtoh(&mut bytes, &src)?;
        std::mem::forget(src);
        for (i, o) in out.iter_mut().enumerate() {
            *o = u32::from_ne_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
        }
        db.stats.2 += 1;
        if !db.disabled {
            eprintln!("[claude] MTP doorbell slot {} did not arrive after a stream sync; using blocking readbacks from now on", t.slot);
        }
        db.disabled = true;
        Ok(())
    }

    /// One launch staging per-cycle scalars from kernel arguments:
    /// `ramp = Some((dst, base, n))` writes `dst[i] = base + i` (n <= 64),
    /// `tok = Some((dst, v))`, `pos = Some((dst, v))` store one i32 each.
    pub fn claude_mtp_stage(
        &mut self,
        ramp: Option<(*mut c_void, i32, usize)>,
        tok: Option<(*mut c_void, i32)>,
        pos: Option<(*mut c_void, i32)>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.claude_doorbell_kernels()?;
        let (rd, rb, rn) = ramp.unwrap_or((std::ptr::null_mut(), 0, 0));
        assert!(rn <= 64);
        let (td, tv) = tok.unwrap_or((std::ptr::null_mut(), 0));
        let (pd, pv) = pos.unwrap_or((std::ptr::null_mut(), 0));
        let mut b = KernargBlob::new();
        b.push_ptr(rd);
        b.push_i32(rb);
        b.push_i32(rn as i32);
        b.push_ptr(td);
        b.push_i32(tv);
        b.push_ptr(pd);
        b.push_i32(pv);
        self.launch_kernel_blob("claude_mtp_stage", [1, 1, 1], [64, 1, 1], 0, b.as_mut_slice())
    }
}
