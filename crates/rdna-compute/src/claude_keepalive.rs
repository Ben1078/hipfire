// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! claude: prefill "keepalive" queue (`CLAUDE_PREFILL_KEEPALIVE`, default on).
//!
//! On the 7900 XTX (gfx1100, ROCm 7.2) a HW queue that is the only busy queue
//! idles ~250 us after a long (>~1 ms), power-limited dispatch before it starts
//! the next packet, although that packet has been in the AQL queue for a long
//! time (HIP trace of the 8K prefill: the kernel after each gap was launched a
//! median 168 ms earlier; 39K prefill: 4871 gaps = 4.8% of the prefill). It is
//! independent of how the host waits (hipDeviceSynchronize, stream/event sync,
//! query spin, pure CPU spin all show it) and of `HSA_ENABLE_INTERRUPT`,
//! `ROC_ACTIVE_WAIT_TIMEOUT`, `HIP_FORCE_DEV_KERNARG`, `AMD_DIRECT_DISPATCH`.
//! It disappears while a second HW queue holds a pending barrier-AND packet
//! (`hipStreamWaitEvent` on another stream): the gap drops to ~6 us
//! (measured with a standalone back-to-back long-kernel reproducer, not included).
//! A `hipStreamWaitValue32` wait on the second queue does not help.
//!
//! [`Gpu::claude_keepalive_arm`] records an event at the current end of the
//! active stream and makes a private non-blocking stream wait for it. The
//! private stream runs no work, so stream order, results and every host sync
//! point are unchanged; the wait retires on its own when the active stream
//! reaches the event (no host involvement, so no deadlock with a later
//! `hipDeviceSynchronize`/`hipFree`). Each arm costs ~16 us of GPU time
//! (the marker), so callers arm a few times per long GPU phase, not per kernel.

use crate::dispatch::Gpu;
use hip_bridge::HipResult;

pub(crate) struct Keepalive {
    stream: hip_bridge::Stream,
    events: [hip_bridge::Event; 2],
    next: usize,
    pub(crate) arms: u64,
}

/// `CLAUDE_PREFILL_KEEPALIVE` (default on; `0` disables).
pub fn claude_keepalive_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_PREFILL_KEEPALIVE").as_deref() != Ok("0"))
}

/// `CLAUDE_PREFILL_KEEPALIVE_MIN_ROWS` (default 256): prefill chunks with
/// fewer rows (verify batches, short tails) are not armed.
pub fn claude_keepalive_min_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CLAUDE_PREFILL_KEEPALIVE_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// Set after the first keepalive failure: the feature switches itself off for
/// the process (a missing keepalive only costs speed, never correctness).
static BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Gpu {
    /// Park a barrier on the private keepalive queue that retires when the
    /// launch stream (the active stream, or the null stream before one is
    /// installed — e.g. the first prefill after a server start) reaches its
    /// current end. Returns whether a barrier was parked; no-op when disabled
    /// or while a graph/replay tape is captured. Never fails: an error is
    /// logged once and disables the keepalive for the process.
    pub fn claude_keepalive_arm(&mut self) -> HipResult<bool> {
        if !claude_keepalive_enabled()
            || self.graphs.capture_mode
            || self.replay.is_recording()
            || BROKEN.load(std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(false);
        }
        match self.claude_keepalive_arm_inner() {
            Ok(()) => Ok(true),
            Err(e) => {
                BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("claude: prefill keepalive disabled after error: {e}");
                Ok(false)
            }
        }
    }

    fn claude_keepalive_arm_inner(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.claude_keepalive.is_none() {
            // Non-blocking: null-stream operations never wait on it.
            let stream = self.hip.stream_create_with_flags(0x01)?;
            let e0 = self.hip.event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?;
            let e1 = self.hip.event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?;
            self.claude_keepalive = Some(Keepalive { stream, events: [e0, e1], next: 0, arms: 0 });
        }
        let ka = self.claude_keepalive.as_mut().expect("keepalive");
        let ev = &ka.events[ka.next];
        self.hip.event_record(ev, self.active_stream.as_ref())?;
        self.hip.stream_wait_event(&ka.stream, ev)?;
        ka.next ^= 1;
        ka.arms += 1;
        Ok(())
    }
}
