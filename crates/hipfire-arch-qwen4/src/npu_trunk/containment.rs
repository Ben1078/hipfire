// SPDX-License-Identifier: Apache-2.0
//! NPU trunk fault containment: the process-wide route latch and the GPU-only rerun of a faulted request.

/// What failed inside a pair: the NPU runtime (chain watchdog, submit, ERT state, relay) or the GPU (a HIP error, whose
/// context is sticky, so a GPU-only rerun could only fail too).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaultClass {
    NpuRuntime,
    GpuHip,
}

impl FaultClass {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NpuRuntime => "NPU runtime",
            Self::GpuHip => "GPU HipError",
        }
    }
}

/// The route's fault latch. The first pair fault of the process stores its reason and turns the route off for good. An
/// NPU runtime fault also marks the request in flight so its entry point reruns it GPU-only; a GPU HipError never does
/// (the request fails with its original error).
#[derive(Default)]
pub(crate) struct FaultLatch {
    fault: Option<String>,
    request: bool,
}

impl FaultLatch {
    /// Records a pair fault. Returns true only for the first trip of the process (the caller logs once).
    pub(crate) fn trip(&mut self, class: FaultClass, reason: &str) -> bool {
        if class == FaultClass::NpuRuntime {
            self.request = true;
        }
        if self.fault.is_some() {
            return false;
        }
        self.fault = Some(reason.to_string());
        true
    }
    pub(crate) fn tripped(&self) -> bool {
        self.fault.is_some()
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn reason(&self) -> Option<&str> {
        self.fault.as_deref()
    }
    /// Whether the request in flight faulted; clears the per-request flag.
    pub(crate) fn take_request_fault(&mut self) -> bool {
        std::mem::take(&mut self.request)
    }
    /// A new request starts: forget the previous request's fault mark (the process latch stays).
    pub(crate) fn begin_request(&mut self) {
        self.request = false;
    }
}

/// Runs `attempt(ctx, start)`; when it fails and the failure tripped the fault latch (`take_fault`), runs it once more
/// from `replan(ctx)`.
///
/// The start is re-planned because the faulted attempt consumed and mutated the live state: the session driver's
/// `begin` took the live continuation, the chunks already forwarded advanced the state, and any uncommitted snapshots
/// of the attempt are dropped by the next turn's start. The only valid reuse points after the fault are the committed
/// snapshots (or a cold start), which is what `replan` returns.
///
/// The rerun is byte-identical to a GPU-only run of the same request from that start: the latched route plans no pair
/// (`faulted()` makes the scheduling exactly the `off` one) and the session machinery restores the chosen snapshot or
/// resets the state before the first row, so nothing the NPU wrote can reach the rerun. A failure that did not trip the
/// latch (abort, validation, GPU error, `on` mode) is returned unchanged and never rerun.
pub(crate) fn rerun_after_fault<C: ?Sized, T, E>(
    ctx: &mut C,
    rows: usize,
    start: usize,
    mut attempt: impl FnMut(&mut C, usize) -> Result<T, E>,
    take_fault: impl Fn(&mut C) -> bool,
    replan: impl Fn(&C) -> usize,
) -> Result<T, E> {
    match attempt(ctx, start) {
        Err(error) => {
            if !take_fault(ctx) {
                return Err(error);
            }
            let restart = replan(ctx);
            eprintln!(
                "npu-trunk: the faulted request restarts GPU-only from row {restart} of {rows}"
            );
            attempt(ctx, restart)
        }
        ok => ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    const NPU_MIX: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const PAIR: usize = 16;
    const CHUNK: usize = 8;
    const FAULT: &str = "NPU chain watchdog timed out";

    /// A toy request owner: a running state, a committed session cache of snapshots at every 8th row, and the route.
    struct Toy {
        state: u64,
        snapshots: BTreeMap<usize, u64>,
        staged: BTreeMap<usize, u64>,
        /// The live continuation of the previous turn: (position, state). `prefill` consumes it.
        live: Option<(usize, u64)>,
        latch: FaultLatch,
        npu_armed: bool,
        fault_at_pair: Option<usize>,
        fault_class: FaultClass,
        npu_pairs_attempted: usize,
        attempts: usize,
    }

    impl Toy {
        fn new(npu_armed: bool, fault_at_pair: Option<usize>) -> Self {
            Self {
                state: 0,
                snapshots: BTreeMap::new(),
                staged: BTreeMap::new(),
                live: None,
                latch: FaultLatch::default(),
                npu_armed,
                fault_at_pair,
                fault_class: FaultClass::NpuRuntime,
                npu_pairs_attempted: 0,
                attempts: 0,
            }
        }
        fn gpu_row(state: u64, tok: usize) -> u64 {
            state.rotate_left(5) ^ (tok as u64).wrapping_mul(GOLDEN)
        }
        fn npu_pair(state: u64, toks: &[usize]) -> u64 {
            let sum = toks.iter().fold(0u64, |a, &t| a.wrapping_add(t as u64));
            state.rotate_left(11) ^ sum.wrapping_mul(NPU_MIX)
        }
        /// The prefill entry: take the live continuation at `start`, else restore `snapshots[start]` (or 0), run NPU pairs while armed and unlatched, then 8-row GPU
        /// chunks; snapshots are staged at each boundary and committed only on success.
        fn prefill(&mut self, prompt: &[usize], start: usize) -> Result<u64, String> {
            self.latch.begin_request();
            self.attempts += 1;
            self.staged.clear();
            let live = self.live.take().filter(|&(position, _)| position == start);
            self.state = match live {
                Some((_, state)) => state,
                None if start == 0 => 0,
                None => *self
                    .snapshots
                    .get(&start)
                    .ok_or("no snapshot at the start")?,
            };
            let mut pos = start;
            while pos < prompt.len() {
                let remaining = prompt.len() - pos;
                if self.npu_armed && !self.latch.tripped() && remaining >= PAIR {
                    let pair = self.npu_pairs_attempted;
                    self.npu_pairs_attempted += 1;
                    if self.fault_at_pair == Some(pair) {
                        self.state = 0xDEAD_BEEF_DEAD_BEEF;
                        self.latch.trip(self.fault_class, FAULT);
                        return Err(FAULT.to_string());
                    }
                    self.state = Self::npu_pair(self.state, &prompt[pos..pos + PAIR]);
                    pos += PAIR;
                } else {
                    let end = (pos + CHUNK).min(prompt.len());
                    for &tok in &prompt[pos..end] {
                        self.state = Self::gpu_row(self.state, tok);
                    }
                    pos = end;
                }
                if pos.is_multiple_of(CHUNK) && pos < prompt.len() {
                    self.staged.insert(pos, self.state);
                }
            }
            self.snapshots.append(&mut self.staged);
            Ok(self.state)
        }
        /// The production shape: the entry wrapped in the rerun helper.
        fn request(&mut self, prompt: &[usize], start: usize) -> Result<u64, String> {
            let rows = prompt.len();
            rerun_after_fault(
                self,
                rows,
                start,
                |toy, start| toy.prefill(prompt, start),
                |toy| toy.latch.take_request_fault(),
                |toy| {
                    toy.snapshots
                        .range(..=start)
                        .next_back()
                        .map_or(0, |(&pos, _)| pos)
                },
            )
        }
    }

    fn prompt(rows: usize) -> Vec<usize> {
        (0..rows).map(|i| i * 7 + 3).collect()
    }

    #[test]
    fn faulted_request_matches_gpu_only() {
        let request = prompt(40);
        let mut reference = Toy::new(false, None);
        let mut faulted = Toy::new(true, Some(0));
        let expected = reference.request(&request, 0).expect("gpu-only request");
        let actual = faulted
            .request(&request, 0)
            .expect("faulted request reruns");
        assert_eq!(actual, expected);
        assert!(faulted.latch.tripped());
        assert_eq!(faulted.latch.reason(), Some(FAULT));
        assert_eq!(faulted.npu_pairs_attempted, 1);
        assert_eq!(
            faulted.snapshots, reference.snapshots,
            "the rerun commits the GPU-only snapshots"
        );
        // The same request through an unfaulted NPU model differs: the toy numerics tell the routes apart.
        let mut healthy = Toy::new(true, None);
        assert_ne!(healthy.request(&request, 0).unwrap(), expected);
    }

    #[test]
    fn latch_persists_across_requests() {
        let mut reference = Toy::new(false, None);
        let mut faulted = Toy::new(true, Some(0));
        let first = prompt(40);
        assert_eq!(
            faulted.request(&first, 0).unwrap(),
            reference.request(&first, 0).unwrap()
        );
        let attempted = faulted.npu_pairs_attempted;
        for rows in [48, 64] {
            let request = prompt(rows);
            assert_eq!(
                faulted.request(&request, 0).unwrap(),
                reference.request(&request, 0).unwrap()
            );
            assert_eq!(
                faulted.npu_pairs_attempted, attempted,
                "a latched route plans no NPU pair"
            );
            assert!(faulted.latch.tripped());
            assert_eq!(faulted.latch.reason(), Some(FAULT));
        }
    }

    #[test]
    fn faulted_live_continuation_replans_to_a_snapshot() {
        let request = prompt(64);
        let mut reference = Toy::new(false, None);
        let live_state = reference.request(&request[..24], 0).unwrap();
        assert_eq!(
            reference.snapshots.keys().copied().collect::<Vec<_>>(),
            [8, 16]
        );
        // The faulted model holds the same committed snapshots and continues live at row 24, which is not a snapshot.
        let mut faulted = Toy::new(true, Some(0));
        faulted.snapshots = reference.snapshots.clone();
        faulted.live = Some((24, live_state));
        let actual = faulted
            .request(&request, 24)
            .expect("faulted continuation reruns");
        assert!(faulted.latch.tripped());
        assert_eq!(faulted.npu_pairs_attempted, 1);
        // The live state was consumed: the rerun started at the largest committed snapshot <= 24 (row 16).
        assert!(faulted.live.is_none());
        let expected = reference.request(&request, 16).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(faulted.snapshots, reference.snapshots);
    }

    #[test]
    fn non_fault_error_is_not_rerun() {
        let mut attempts = 0;
        let mut latch = FaultLatch::default();
        let result: Result<u32, &str> = rerun_after_fault(
            &mut latch,
            10,
            0,
            |_, _| {
                attempts += 1;
                Err("aborted")
            },
            |latch| latch.take_request_fault(),
            |_| panic!("no replan without a fault"),
        );
        assert_eq!(result, Err("aborted"));
        assert_eq!(attempts, 1);
        assert!(!latch.tripped());
    }

    #[test]
    fn gpu_hip_error_fails_the_request_without_a_rerun() {
        let mut faulted = Toy::new(true, Some(0));
        faulted.fault_class = FaultClass::GpuHip;
        let error = faulted.request(&prompt(40), 0).unwrap_err();
        assert_eq!(error, FAULT, "the original error is returned unchanged");
        assert_eq!(faulted.attempts, 1, "a sticky GPU context is never rerun");
        assert!(faulted.latch.tripped(), "the route still latches off");
        assert!(
            faulted.snapshots.is_empty(),
            "nothing of the failed attempt is committed"
        );
        // Later requests run GPU-only (no NPU pair) and match a GPU-only model.
        let mut reference = Toy::new(false, None);
        let request = prompt(48);
        assert_eq!(
            faulted.request(&request, 0).unwrap(),
            reference.request(&request, 0).unwrap()
        );
        assert_eq!(faulted.npu_pairs_attempted, 1);
    }

    #[test]
    fn latch_trips_once_and_request_flag_clears() {
        let mut latch = FaultLatch::default();
        assert!(!latch.tripped() && latch.reason().is_none() && !latch.take_request_fault());
        assert!(latch.trip(FaultClass::NpuRuntime, "first"));
        assert!(!latch.trip(FaultClass::NpuRuntime, "second"));
        assert_eq!(latch.reason(), Some("first"));
        assert!(latch.take_request_fault());
        assert!(!latch.take_request_fault());
        latch.trip(FaultClass::NpuRuntime, "third");
        latch.begin_request();
        assert!(!latch.take_request_fault());
        assert!(latch.tripped());
        // A GPU HipError latches the route but never marks the request for a rerun.
        let mut gpu = FaultLatch::default();
        assert!(gpu.trip(FaultClass::GpuHip, "HIP 700"));
        assert!(gpu.tripped() && !gpu.take_request_fault());
    }
}
