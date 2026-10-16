// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact lane store of the Flash-Next batch route.
//!
//! A lane is one admitted request with its own [`Qwen4State`]; nothing is
//! shared between lanes but the weights. A store staged with a
//! [`LaneSpecConfig`] makes every lane an MTP lane: it also owns the lane's
//! MTP head state, draft request state, pending hidden, GDN capture ring and
//! window policy ([`LaneSpec`]), and the store shares the step buffers MTP
//! windows need (wide-hidden capture, the batched draft's top-1 slots, the
//! prompt-fill append scratch, the step cost model). [`Qwen4LaneStore`] owns
//! the lanes and the step output buffers; [`crate::lane_exec::Qwen4LaneExecutor`]
//! runs plans over it. The host-side lane table ([`LaneTable`]) is generic
//! over its device payload so its admission/release/poison rules are
//! covered by CPU tests.
//!
//! Lane device states are allocated on a slot's first admission and kept
//! across releases: a later admission into that slot resets the state in
//! place instead of allocating again.

use crate::bundle::Qwen4Bundle;
use crate::config::Qwen4Config;
use crate::gpu_forward_lanes::{LaneStagePolicy, LANE_MAX_ROWS};
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::lane_exec::LanePhase;
use crate::mtp_gpu::{MtpAppendScratch, MtpGpuState, MTP_FILL_ROWS};
use crate::mtp_lanes::{LaneCostModel, LaneSpecPolicy, PendingWindow};
use crate::mtp_spec::{
    mtp_fill_scratch_requested, mtp_fill_scratch_wanted, native_mtp_row_capture, Qwen4PromotionParts,
};
use crate::state::{Qwen4State, Qwen4StateFormat};
use hipfire_dispatch::pipeline::DraftHeadRequestState;
use hipfire_runtime::slot_batch::RequestEpoch;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

/// Most lanes one store holds.
const MAX_LANES: usize = 64;

/// What a staged lane store reports (load ack, oracle receipts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneStoreReceipt {
    pub kv_backend: &'static str,
    pub qsa_format: &'static str,
    pub gdn_format: &'static str,
    /// GDN capture rows armed per lane (0: AR lanes arm no ring; MTP lanes
    /// arm `k + 1`).
    pub ring_rows: usize,
    pub max_lanes: usize,
    pub row_budget: usize,
    /// Device bytes the lanes' QSA context arenas commit right now.
    pub mapped_bytes: u64,
    pub stage_policy: String,
    pub stage_evidence: &'static str,
    /// `"mtp"` when every lane is an MTP lane, else `"off"`.
    pub spec: &'static str,
    /// Most drafts one window carries (0 without spec).
    pub spec_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneStatus {
    /// The prompt is being forwarded tile by tile.
    Prefilling,
    /// The prompt is done; one AR row per step.
    Decoding,
    /// A step failed with this lane in it: it runs nothing and commits
    /// nothing until released.
    Poisoned,
}

/// Host bookkeeping of one admitted request.
#[derive(Clone, Debug)]
pub struct LaneHost {
    pub epoch: RequestEpoch,
    /// Stable lane index.
    pub slot: usize,
    pub prompt: Vec<u32>,
    /// Prompt tokens forwarded so far.
    pub prefilled: usize,
    /// Picks committed so far (the first is the prompt-final pick).
    pub generated: Vec<u32>,
    /// The next AR row's token.
    pub pending_seed: Option<u32>,
    pub status: LaneStatus,
    /// Admission order across the store's lifetime.
    pub stamp: u64,
}

/// Staging of the lanes' native MTP windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneSpecConfig {
    /// Most drafts per window, `1..=7` (`k + 1` verify rows fit
    /// [`LANE_MAX_ROWS`] and the lane's `k + 1`-row GDN ring).
    pub k: usize,
    /// The end-of-turn token a window stops at; `None` uses the model's EOS.
    pub end_of_turn: Option<u32>,
}

/// Counters of one MTP lane's windows (read after the request, or at any
/// step boundary).
#[derive(Clone, Copy, Debug, Default)]
pub struct LaneSpecStats {
    /// Native windows: batched windows and `ArSynced` AR rows.
    pub windows: u64,
    /// Head-free AR rows after retirement.
    pub ar_windows: u64,
    /// Drafts verified.
    pub drafts: u64,
    /// Drafts accepted.
    pub accepted: u64,
    /// The lane's floor retired it to head-free AR.
    pub retired: bool,
}

/// Lab route modes of the store's window router.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabRoutes {
    /// The lane floor (production).
    #[default]
    Floor,
    /// `Batched{K}`, `Batched{1}`, `ArSynced`, `Batched{2}` three times,
    /// then `ArRetired`.
    Cycle,
    /// Every window drafts its whole budget.
    Batched,
}

/// One MTP lane's device state next to its trunk state.
pub struct LaneSpec {
    /// The lane's MTP head state (QSA arenas, selection, wide hidden).
    pub head: MtpGpuState,
    /// The draft head's request-local policy (full-vocabulary hold, margin).
    pub draft: DraftHeadRequestState,
    /// `[hc_count * hidden]` F32: the next window's row-0 head hidden.
    pub pending_hidden: GpuTensor,
    pub(crate) policy: LaneSpecPolicy,
    /// The routed window between `draft_step` and `commit_step`.
    pub(crate) window: Option<PendingWindow>,
    pub stats: LaneSpecStats,
}

impl LaneSpec {
    /// A fresh spec state: the trunk `state` gets its `k + 1`-row verify
    /// ring, the head is built on the resident head's backend and context
    /// bound.
    fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        spec: LaneSpecConfig,
        head_max_seq: usize,
        backend: Qwen4KvBackend,
        state: &mut Qwen4State,
    ) -> Result<Self, String> {
        let config = &bundle.config;
        state
            .ensure_row_capture(gpu, spec.k + 1, config.linear_conv_kernel_dim - 1)
            .map_err(|e| format!("lane ring: {e}"))?;
        let head = MtpGpuState::new_with_backend(gpu, config, head_max_seq, backend)
            .map_err(|e| format!("lane head state: {e}"))?;
        let width = config
            .hc_count
            .checked_mul(config.hidden_size)
            .ok_or_else(|| "lane pending hidden width overflow".to_string())?;
        let pending_hidden = match gpu.zeros(&[width], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = head.free_gpu(gpu);
                return Err(format!("lane pending hidden: {error}"));
            }
        };
        Ok(Self {
            head,
            draft: DraftHeadRequestState::default(),
            pending_hidden,
            policy: LaneSpecPolicy::new(spec.k),
            window: None,
            stats: LaneSpecStats::default(),
        })
    }

    /// Back to a request's start: empty head, default draft policy, zero
    /// pending hidden, fresh window policy and counters, no window.
    fn reset(&mut self, gpu: &mut Gpu, k: usize) -> Result<(), String> {
        self.head
            .reset(gpu)
            .map_err(|e| format!("lane head reset: {e}"))?;
        self.draft = DraftHeadRequestState::default();
        gpu.hip
            .memset(
                &self.pending_hidden.buf,
                0,
                self.pending_hidden.buf.size(),
            )
            .map_err(|e| format!("lane pending hidden reset: {e}"))?;
        self.policy = LaneSpecPolicy::new(k);
        self.window = None;
        self.stats = LaneSpecStats::default();
        Ok(())
    }

    /// Start this (kept or fresh) lane spec from a promoted singleton's
    /// `parts`: its pending hidden copied in, the window policy started from
    /// its agreement and retirement, no window, fresh counters. The head and
    /// draft request state are the singleton's own and swap in at commit.
    fn promote(
        &mut self,
        gpu: &mut Gpu,
        k: usize,
        parts: &Qwen4PromotionParts<'_>,
    ) -> Result<(), String> {
        let bytes = self.pending_hidden.byte_size();
        if parts.pending_hidden.byte_size() != bytes {
            return Err(format!(
                "pending hidden is {} bytes, the lane's is {bytes}",
                parts.pending_hidden.byte_size()
            ));
        }
        gpu.copy_d2d(parts.pending_hidden, &self.pending_hidden, bytes)
            .map_err(|e| format!("pending hidden copy: {e}"))?;
        self.draft = DraftHeadRequestState::default();
        self.policy = LaneSpecPolicy::promoted(k, parts.agreement, parts.retired);
        self.window = None;
        self.stats = LaneSpecStats::default();
        Ok(())
    }
}

/// A fresh slot's payload: a lane trunk state and, with `spec_cfg`, its MTP
/// state (a failure frees what was allocated).
fn new_payload(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
    slot: usize,
    spec_cfg: Option<(LaneSpecConfig, usize, Qwen4KvBackend)>,
) -> Result<(Qwen4State, Option<LaneSpec>), String> {
    let mut state = bundle
        .new_lane_state(gpu)
        .map_err(|e| format!("admit: allocate lane {slot}: {e}"))?;
    let spec = match spec_cfg {
        None => None,
        Some((config, head_max_seq, backend)) => {
            match LaneSpec::new(gpu, bundle, config, head_max_seq, backend, &mut state) {
                Ok(spec) => Some(spec),
                Err(error) => {
                    let _ = state.free_gpu(gpu);
                    return Err(format!("admit: allocate lane {slot}: {error}"));
                }
            }
        }
    };
    Ok((state, spec))
}

/// One admitted request's whole device state.
pub struct Qwen4LaneState {
    pub host: LaneHost,
    pub state: Qwen4State,
    /// `Some` on every lane of a store staged with spec, `None` on AR lanes.
    pub spec: Option<LaneSpec>,
}

/// A lane-table cell: host bookkeeping plus a device payload that outlives
/// the request (the table keeps it when the lane is released).
trait LaneCell {
    type Payload;
    fn assemble(host: LaneHost, payload: Self::Payload) -> Self;
    fn host(&self) -> &LaneHost;
    fn host_mut(&mut self) -> &mut LaneHost;
}

impl LaneCell for Qwen4LaneState {
    type Payload = (Qwen4State, Option<LaneSpec>);

    fn assemble(host: LaneHost, (state, spec): Self::Payload) -> Self {
        Self { host, state, spec }
    }

    fn host(&self) -> &LaneHost {
        &self.host
    }

    fn host_mut(&mut self) -> &mut LaneHost {
        &mut self.host
    }
}

/// Slot table of the lanes: which slots hold a live request and the cells
/// (host + payload) allocated so far. Epoch lookups match both the request
/// tag and the owner generation, so a stale generation never matches.
struct LaneTable<C: LaneCell> {
    cells: Vec<Option<C>>,
    live: Vec<bool>,
    next_stamp: u64,
}

impl<C: LaneCell> LaneTable<C> {
    fn new(max_lanes: usize) -> Self {
        Self {
            cells: (0..max_lanes).map(|_| None).collect(),
            live: vec![false; max_lanes],
            next_stamp: 0,
        }
    }

    fn max_lanes(&self) -> usize {
        self.cells.len()
    }

    fn slot_of(&self, epoch: &RequestEpoch) -> Option<usize> {
        self.cells.iter().enumerate().find_map(|(slot, cell)| {
            let cell = cell.as_ref()?;
            (self.live[slot] && cell.host().epoch == *epoch).then_some(slot)
        })
    }

    fn get(&self, epoch: &RequestEpoch) -> Option<&C> {
        self.slot_of(epoch).and_then(|slot| self.cells[slot].as_ref())
    }

    fn get_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut C> {
        let slot = self.slot_of(epoch)?;
        self.cells[slot].as_mut()
    }

    fn active(&self) -> usize {
        self.live.iter().filter(|live| **live).count()
    }

    /// Live epochs in slot order.
    fn epochs(&self) -> Vec<RequestEpoch> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(slot, _)| self.live[*slot])
            .filter_map(|(_, cell)| cell.as_ref().map(|c| c.host().epoch))
            .collect()
    }

    /// Admit `epoch` into the lowest free slot as a request about to prefill.
    /// `prepare(slot, existing)` is called exactly once: with the slot's kept
    /// payload cell (it must reset it in place and return `Ok(None)`), or
    /// with `None` when the slot has never held one (it must return the new
    /// payload). The new host entry is published only after `prepare`
    /// succeeded.
    fn admit(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        self.admit_host(epoch, prompt, None, prepare)
    }

    /// Admit `epoch` as an already-decoding request (a promoted singleton):
    /// the whole `prompt` is prefilled, `generated` picks are committed and
    /// `pending_seed` is the next AR row's token. Same `prepare` contract and
    /// refusals as [`Self::admit`].
    fn admit_decoding(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        generated: Vec<u32>,
        pending_seed: u32,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        self.admit_host(epoch, prompt, Some((generated, pending_seed)), prepare)
    }

    fn admit_host(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        decoding: Option<(Vec<u32>, u32)>,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        if !epoch.is_admitted() {
            return Err(format!("admit: generation 0 is never admitted ({epoch:?})"));
        }
        if self.slot_of(&epoch).is_some() {
            return Err(format!("admit: epoch {epoch:?} is already live"));
        }
        if prompt.is_empty() {
            return Err("admit: empty prompt".to_string());
        }
        let Some(slot) = self.live.iter().position(|live| !*live) else {
            return Err(format!("admit: all {} lanes are busy", self.max_lanes()));
        };
        let (prefilled, generated, pending_seed, status) = match decoding {
            Some((generated, seed)) => (prompt.len(), generated, Some(seed), LaneStatus::Decoding),
            None => (0, Vec::new(), None, LaneStatus::Prefilling),
        };
        let host = LaneHost {
            epoch,
            slot,
            prompt,
            prefilled,
            generated,
            pending_seed,
            status,
            stamp: self.next_stamp,
        };
        match self.cells[slot].as_mut() {
            Some(cell) => {
                if prepare(slot, Some(&mut *cell))?.is_some() {
                    return Err("admit: prepare returned a payload for a kept cell".to_string());
                }
                *cell.host_mut() = host;
            }
            None => {
                let payload = prepare(slot, None)?
                    .ok_or_else(|| "admit: prepare returned no payload for a new cell".to_string())?;
                self.cells[slot] = Some(C::assemble(host, payload));
            }
        }
        self.live[slot] = true;
        self.next_stamp += 1;
        Ok(slot)
    }

    /// Free `epoch`'s slot; its payload stays for the next admission.
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        let slot = self
            .slot_of(epoch)
            .ok_or_else(|| format!("release: unknown or stale epoch {epoch:?}"))?;
        self.live[slot] = false;
        let host = self.cells[slot].as_mut().expect("live slot has a cell").host_mut();
        host.prompt = Vec::new();
        host.generated = Vec::new();
        host.pending_seed = None;
        Ok(())
    }

    /// Mark `epoch` untrusted; unknown epochs are ignored.
    fn poison(&mut self, epoch: &RequestEpoch) {
        if let Some(cell) = self.get_mut(epoch) {
            cell.host_mut().status = LaneStatus::Poisoned;
        }
    }

    /// Replace a Decoding lane's next AR token.
    fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        let host = self
            .get_mut(epoch)
            .ok_or_else(|| format!("force_seed: unknown or stale epoch {epoch:?}"))?
            .host_mut();
        if host.status != LaneStatus::Decoding {
            return Err(format!("force_seed: lane {:?} is {:?}, not Decoding", epoch, host.status));
        }
        host.pending_seed = Some(token);
        Ok(())
    }

    /// Disjoint `&mut` cells of the live lanes `epochs`, in that order.
    fn pick_mut(&mut self, epochs: &[RequestEpoch]) -> Result<Vec<&mut C>, String> {
        let slots = epochs
            .iter()
            .map(|epoch| {
                self.slot_of(epoch)
                    .ok_or_else(|| format!("unknown or stale epoch {epoch:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut refs: Vec<Option<&mut C>> = self.cells.iter_mut().map(|c| c.as_mut()).collect();
        slots
            .iter()
            .map(|&slot| {
                refs[slot]
                    .take()
                    .ok_or_else(|| format!("lane slot {slot} named twice"))
            })
            .collect()
    }

    fn into_cells(self) -> Vec<C> {
        self.cells.into_iter().flatten().collect()
    }
}

/// Buffers and policy shared by the MTP lanes of one store.
pub(crate) struct SpecStore {
    pub config: LaneSpecConfig,
    /// The resident head's context bound: every lane head is built with it.
    pub head_max_seq: usize,
    pub backend: Qwen4KvBackend,
    /// F32, `rows_cap * hc_count * hidden`: the wide hidden of every decode
    /// row of a step (the lane forward's capture).
    pub hidden: GpuTensor,
    /// Raw, `max_lanes * Gpu::TOPK8_OUT_BYTES`: one slot per still-drafting
    /// lane per draft step.
    pub draft_top1: GpuTensor,
    /// Host mirror of `draft_top1`.
    pub draft_top1_host: Vec<u8>,
    /// Shared prompt-fill append scratch (`None` when neither batched fill nor
    /// the DET prompt fill wants it, or the head does not support it).
    pub append_scratch: Option<MtpAppendScratch>,
    pub cost: LaneCostModel,
    pub routes: LabRoutes,
    /// Lab negative control: a rejected verify keeps its full rows.
    pub skip_rollback: bool,
}

impl SpecStore {
    fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        config: LaneSpecConfig,
        max_lanes: usize,
        rows_cap: usize,
    ) -> Result<Self, String> {
        let mtp = bundle
            .mtp
            .as_ref()
            .ok_or_else(|| "lane store: MTP lanes need the attached MTP head".to_string())?;
        let (head_max_seq, backend) = (mtp.max_seq, mtp.state.backend());
        let wide = bundle
            .config
            .hc_count
            .checked_mul(bundle.config.hidden_size)
            .ok_or_else(|| "lane store: hidden width overflow".to_string())?;
        let hidden_elements = rows_cap
            .checked_mul(wide)
            .ok_or_else(|| "lane store: hidden capture extent overflow".to_string())?;
        let hidden = gpu
            .zeros(&[hidden_elements], DType::F32)
            .map_err(|e| format!("lane store: hidden capture: {e}"))?;
        let top1_bytes = max_lanes * Gpu::TOPK8_OUT_BYTES;
        let draft_top1 = match gpu.zeros(&[top1_bytes], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(hidden);
                return Err(format!("lane store: draft top-1: {error}"));
            }
        };
        let append_scratch = if mtp_fill_scratch_wanted(gpu)
            && bundle.mtp_append_rows_supported(gpu)
        {
            let rows = MTP_FILL_ROWS.min(bundle.spec_chunk_rows().unwrap_or(1).max(1));
            // Prompt-id rows = append rows: a tile of at most `rows` rows
            // uploads its ids once; a longer one uploads per sub-chunk
            // (`append_rows`' own upload, the same bytes).
            match MtpAppendScratch::new(gpu, &bundle.config, rows, rows) {
                Ok(scratch) => Some(scratch),
                Err(error) => {
                    let _ = gpu.free_tensor(hidden);
                    let _ = gpu.free_tensor(draft_top1);
                    return Err(format!("lane store: append scratch: {error}"));
                }
            }
        } else {
            None
        };
        Ok(Self {
            config,
            head_max_seq,
            backend,
            hidden,
            draft_top1,
            draft_top1_host: vec![0; top1_bytes],
            append_scratch,
            cost: LaneCostModel::new(),
            routes: LabRoutes::Floor,
            skip_rollback: false,
        })
    }
}

/// Disjoint borrows one step needs out of the store.
pub(crate) struct StepParts<'a> {
    /// The planned lanes, in plan order.
    pub lanes: Vec<&'a mut Qwen4LaneState>,
    /// F32, `(rows_cap + max_lanes) * vocab`.
    pub logits: &'a GpuTensor,
    /// Raw, `(rows_cap + max_lanes) * 4` bytes.
    pub top1: &'a GpuTensor,
    pub policy: LaneStagePolicy,
    /// Host mirror of `top1`.
    pub top1_host: &'a mut Vec<u8>,
    /// Decode rows the step buffers hold: `max_lanes * (k + 1)` with spec,
    /// else `max_lanes`.
    pub rows_cap: usize,
    pub spec: Option<&'a mut SpecStore>,
}

/// The lane owner: lanes, step output buffers, stage policy and the step
/// phase machine. See the module docs of [`crate::lane_exec`].
pub struct Qwen4LaneStore {
    table: LaneTable<Qwen4LaneState>,
    row_budget: usize,
    ring_rows: usize,
    format: Qwen4StateFormat,
    backend: Qwen4KvBackend,
    policy: LaneStagePolicy,
    /// Decode rows the step buffers hold (`max_lanes * (k + 1)` with spec,
    /// else `max_lanes`).
    rows_cap: usize,
    /// Rows `[0, rows_cap)`: the decode rows of a step (the planned decode
    /// requests' rows concatenated in plan order); `[rows_cap, rows_cap +
    /// max_lanes)`: the prompt-completing prefill rows.
    logits: GpuTensor,
    top1: GpuTensor,
    top1_host: Vec<u8>,
    spec: Option<SpecStore>,
    pub(crate) phase: LanePhase,
    pub(crate) next_step_id: u64,
    /// Start of the step's wall clock: set by `draft_step` (or provision when
    /// no draft ran), taken by the forward.
    pub(crate) step_started: Option<Instant>,
}

impl Qwen4LaneStore {
    /// An empty store for `max_lanes` lanes (states are allocated on first
    /// admission into a slot) and its step output buffers. With `spec`
    /// every lane is an MTP lane.
    pub(crate) fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_lanes: usize,
        row_budget: usize,
        spec: Option<LaneSpecConfig>,
    ) -> Result<Self, String> {
        if !bundle.config.profile.is_flash_next() {
            return Err(format!(
                "lane store: lanes run only the Flash-Next profile, not {}",
                bundle.config.profile.id()
            ));
        }
        if max_lanes == 0 || max_lanes > MAX_LANES {
            return Err(format!("lane store: max_lanes {max_lanes} outside 1..={MAX_LANES}"));
        }
        if row_budget == 0 {
            return Err("lane store: row_budget must be at least 1".to_string());
        }
        if let Some(spec) = spec {
            let verify_rows = if bundle.has_cpu_deferred_experts() {
                crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
            } else {
                LANE_MAX_ROWS
            };
            if spec.k == 0 || spec.k >= verify_rows {
                return Err(format!(
                    "lane store: MTP k {} outside 1..={} for this execution mode",
                    spec.k,
                    verify_rows - 1
                ));
            }
            if bundle.mtp.is_none() {
                return Err("lane store: MTP lanes need the attached MTP head".to_string());
            }
            if !native_mtp_row_capture(gpu, &bundle.config) {
                return Err(
                    "lane store: MTP lanes need the GDN row capture this GPU lacks".to_string(),
                );
            }
        }
        let rows_cap = match spec {
            Some(spec) => max_lanes * (spec.k + 1),
            None => max_lanes,
        };
        let rows = rows_cap + max_lanes;
        let vocab = bundle.lane_vocab();
        let logit_elements = rows
            .checked_mul(vocab)
            .ok_or_else(|| "lane store: logits extent overflow".to_string())?;
        let logits = gpu
            .zeros(&[logit_elements], DType::F32)
            .map_err(|e| format!("lane store: logits: {e}"))?;
        let top1 = match gpu.zeros(&[rows * std::mem::size_of::<i32>()], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(logits);
                return Err(format!("lane store: top1: {error}"));
            }
        };
        let spec_store = match spec {
            Some(config) => match SpecStore::new(gpu, bundle, config, max_lanes, rows_cap) {
                Ok(store) => Some(store),
                Err(error) => {
                    let _ = gpu.free_tensor(logits);
                    let _ = gpu.free_tensor(top1);
                    return Err(error);
                }
            },
            None => None,
        };
        Ok(Self {
            table: LaneTable::new(max_lanes),
            row_budget: row_budget.min(max_lanes),
            ring_rows: spec.map_or(0, |spec| spec.k + 1),
            format: bundle.state_format(),
            backend: bundle.state.qsa_backend(),
            // Shared stages only where G0 evidenced them (gfx1151).
            policy: LaneStagePolicy::for_arch(&gpu.arch),
            rows_cap,
            logits,
            top1,
            top1_host: vec![0; rows * std::mem::size_of::<i32>()],
            spec: spec_store,
            phase: LanePhase::Idle,
            next_step_id: 0,
            step_started: None,
        })
    }

    /// Device bytes the store commits at load: `max_lanes` lane states
    /// ([`Qwen4State::device_bytes`]) plus the step logits and top-1
    /// buffers; with `spec_k` each lane also holds its MTP head state, its
    /// `k + 1`-row GDN ring and pending hidden, and the store the wide-hidden
    /// capture, the draft top-1 slots and the prompt-fill append scratch
    /// (counted at its `MTP_FILL_ROWS` bound: the real scratch is
    /// `min(MTP_FILL_ROWS, forward max_chunk)` rows). `_row_budget` is
    /// accepted for the frozen signature: decode rows share the
    /// `max_lanes * (k + 1)` output rows, so the budget adds nothing.
    pub fn device_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        max_lanes: usize,
        _row_budget: usize,
        spec_k: Option<usize>,
    ) -> Option<u64> {
        let f32_bytes = std::mem::size_of::<f32>();
        let mut per_lane = Qwen4State::device_bytes(config, format, context)?;
        let rows_cap = match spec_k {
            Some(k) => max_lanes.checked_mul(k.checked_add(1)?)?,
            None => max_lanes,
        };
        let rows = rows_cap.checked_add(max_lanes)?;
        let mut shared = rows
            .checked_mul(config.vocab_size)?
            .checked_mul(f32_bytes)?
            .checked_add(rows.checked_mul(std::mem::size_of::<i32>())?)?;
        if let Some(k) = spec_k {
            let wide = config.hc_count.checked_mul(config.hidden_size)?;
            let ring = Qwen4State::row_capture_bytes(config, format.gdn, k.checked_add(1)?)?;
            per_lane = per_lane
                .checked_add(MtpGpuState::device_bytes(config, context)?)?
                .checked_add(ring)?
                .checked_add(wide.checked_mul(f32_bytes)?)?;
            let append = if mtp_fill_scratch_requested() {
                MtpAppendScratch::device_bytes(config, MTP_FILL_ROWS, MTP_FILL_ROWS).unwrap_or(0)
            } else {
                0
            };
            shared = shared
                .checked_add(rows_cap.checked_mul(wide)?.checked_mul(f32_bytes)?)?
                .checked_add(max_lanes.checked_mul(Gpu::TOPK8_OUT_BYTES)?)?
                .checked_add(append)?;
        }
        u64::try_from(max_lanes.checked_mul(per_lane)?.checked_add(shared)?).ok()
    }

    /// Admit `epoch` with its canonical `prompt` into the lowest free lane
    /// and return its slot. A slot that held a request before resets its
    /// state in place (an MTP lane also its head, draft policy, pending
    /// hidden, window policy and counters); a fresh slot allocates one.
    pub fn admit(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
    ) -> Result<usize, String> {
        let spec_cfg = self
            .spec
            .as_ref()
            .map(|s| (s.config, s.head_max_seq, s.backend));
        self.table.admit(epoch, prompt, |slot, existing| match existing {
            Some(cell) => {
                cell.state
                    .reset(gpu)
                    .map_err(|e| format!("admit: reset lane {slot}: {e}"))?;
                cell.state.row_capture_armed = false;
                match (spec_cfg, cell.spec.as_mut()) {
                    (Some((config, _, _)), Some(spec)) => spec
                        .reset(gpu, config.k)
                        .map_err(|e| format!("admit: reset lane {slot}: {e}"))?,
                    (None, None) => {}
                    _ => {
                        return Err(format!(
                            "admit: lane {slot} spec state does not match the store"
                        ));
                    }
                }
                Ok(None)
            }
            None => new_payload(gpu, bundle, slot, spec_cfg).map(Some),
        })
    }

    /// Admit `epoch` straight into Decoding (a promoted singleton): `prompt`
    /// is fully prefilled, `generated` picks are committed, `pending_seed`
    /// is the next AR row. The lane's trunk and head states are not touched
    /// here (the bundle swaps the singleton's in at commit); a fresh slot
    /// allocates them like [`Self::admit`]. An MTP store takes the singleton's
    /// pending hidden (one device copy) and the lane policy started from
    /// `parts`; an AR store takes none. Refused while a step is in flight.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit_promoted(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        generated: Vec<u32>,
        pending_seed: u32,
        parts: Option<Qwen4PromotionParts<'_>>,
    ) -> Result<usize, String> {
        if self.phase != LanePhase::Idle {
            return Err("admit_promoted: a step is in flight".to_string());
        }
        let spec_cfg = self
            .spec
            .as_ref()
            .map(|s| (s.config, s.head_max_seq, s.backend));
        match (spec_cfg.is_some(), parts.is_some()) {
            (true, false) => {
                return Err("admit_promoted: an MTP lane store needs the drafter's parts".to_string());
            }
            (false, true) => {
                return Err("admit_promoted: an AR lane store takes no drafter parts".to_string());
            }
            _ => {}
        }
        self.table
            .admit_decoding(epoch, prompt, generated, pending_seed, |slot, existing| match existing {
                Some(cell) => {
                    match (spec_cfg, cell.spec.as_mut(), parts.as_ref()) {
                        (Some((config, _, _)), Some(spec), Some(parts)) => spec
                            .promote(gpu, config.k, parts)
                            .map_err(|e| format!("admit_promoted: lane {slot}: {e}"))?,
                        (None, None, None) => {}
                        _ => {
                            return Err(format!(
                                "admit_promoted: lane {slot} spec state does not match the store"
                            ));
                        }
                    }
                    cell.state.row_capture_armed = false;
                    Ok(None)
                }
                None => {
                    let (state, mut spec) = new_payload(gpu, bundle, slot, spec_cfg)?;
                    let seeded = match (spec_cfg, spec.as_mut(), parts.as_ref()) {
                        (Some((config, _, _)), Some(spec), Some(parts)) => {
                            spec.promote(gpu, config.k, parts)
                        }
                        _ => Ok(()),
                    };
                    if let Err(error) = seeded {
                        let _ = state.free_gpu(gpu);
                        if let Some(spec) = spec {
                            let _ = spec.head.free_gpu(gpu);
                            let _ = gpu.free_tensor(spec.pending_hidden);
                        }
                        return Err(format!("admit_promoted: lane {slot}: {error}"));
                    }
                    Ok(Some((state, spec)))
                }
            })
    }

    /// Free `epoch`'s lane (its device state stays allocated for reuse).
    pub fn release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        self.table.release(epoch)
    }

    pub fn poison(&mut self, epoch: &RequestEpoch) {
        if let Some(lane) = self.table.get_mut(epoch) {
            // A half-run window's tickets die with it: the next admission's
            // reset invalidates them.
            lane.state.row_capture_armed = false;
            if let Some(spec) = lane.spec.as_mut() {
                spec.window = None;
            }
        }
        self.table.poison(epoch);
    }

    /// Replace a Decoding lane's next AR token (the singleton's think-budget
    /// forced close tokens); `generated` is untouched.
    pub fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        self.table.force_seed(epoch, token)
    }

    pub fn lane(&self, epoch: &RequestEpoch) -> Option<&Qwen4LaneState> {
        self.table.get(epoch)
    }

    pub(crate) fn lane_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut Qwen4LaneState> {
        self.table.get_mut(epoch)
    }

    /// Live epochs in slot order.
    pub fn epochs(&self) -> Vec<RequestEpoch> {
        self.table.epochs()
    }

    /// The step logits (F32, `(rows_cap + max_lanes)` rows of `vocab`):
    /// row `r < rows_cap` is the `r`-th decode row of the plan (the planned
    /// decode requests' rows concatenated in plan order: an AR request is
    /// one row, a Verify request `1 + draft_len` rows); the i-th Prefill
    /// request is row `rows_cap + i` (written on a prompt-completing tile of
    /// an AR lane; an MTP lane's prompt-completing pick is read by the
    /// singleton chunk path and leaves no logits row). `rows_cap` is
    /// `max_lanes` without spec, so an AR-only plan is the S3 layout.
    /// Valid until the next forward; read it before `commit_step`.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    /// First prompt-completing logits row: `rows_cap` (`max_lanes * (k + 1)`
    /// with spec, else `max_lanes`).
    pub fn prefill_logits_row(&self) -> usize {
        self.rows_cap
    }

    /// The MTP staging (`None`: AR lanes).
    pub fn spec(&self) -> Option<LaneSpecConfig> {
        self.spec.as_ref().map(|s| s.config)
    }

    /// `epoch`'s window counters (`None`: unknown lane or an AR lane).
    pub fn lane_spec_stats(&self, epoch: &RequestEpoch) -> Option<LaneSpecStats> {
        let spec = self.lane(epoch)?.spec.as_ref()?;
        Some(LaneSpecStats {
            retired: spec.policy.retired(),
            ..spec.stats
        })
    }

    /// Choose the window router (lab): the floor (default), a fixed cycle or
    /// full-budget batched windows. No effect on AR lanes.
    #[cfg(feature = "lab")]
    pub fn lab_set_routes(&mut self, routes: LabRoutes) {
        if let Some(spec) = self.spec.as_mut() {
            spec.routes = routes;
        }
    }

    /// Negative control (lab): a rejected verify keeps its full rows.
    #[cfg(feature = "lab")]
    pub fn lab_set_skip_rollback(&mut self, on: bool) {
        if let Some(spec) = self.spec.as_mut() {
            spec.skip_rollback = on;
        }
    }

    /// Negative control (lab): zero `epoch`'s pending hidden (the next
    /// window's row-0 head hidden), as if a promotion had lost the copy.
    #[cfg(feature = "lab")]
    pub fn lab_zero_pending_hidden(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
    ) -> Result<(), String> {
        let spec = self
            .lane(epoch)
            .ok_or_else(|| format!("lab_zero_pending_hidden: unknown or stale epoch {epoch:?}"))?
            .spec
            .as_ref()
            .ok_or_else(|| "lab_zero_pending_hidden: AR lane".to_string())?;
        gpu.hip
            .memset(&spec.pending_hidden.buf, 0, spec.pending_hidden.buf.size())
            .map_err(|e| format!("lab_zero_pending_hidden: {e}"))
    }

    /// Every byte family of `epoch`'s MTP head state
    /// ([`crate::mtp_gpu::lab_head_items`]), then `draft_request_state`
    /// (`full_steps` u32 LE, `margin` f32 LE) and `pending_hidden` (F32 LE).
    #[cfg(feature = "lab")]
    pub fn lab_lane_items(
        &self,
        gpu: &Gpu,
        epoch: &RequestEpoch,
    ) -> Result<Vec<(String, Vec<u8>)>, String> {
        let spec = self
            .lane(epoch)
            .ok_or_else(|| format!("lab_lane_items: unknown or stale epoch {epoch:?}"))?
            .spec
            .as_ref()
            .ok_or_else(|| "lab_lane_items: AR lane".to_string())?;
        let mut items = crate::mtp_gpu::lab_head_items(gpu, &spec.head)?;
        let mut request = spec.draft.full_steps.to_le_bytes().to_vec();
        request.extend_from_slice(&spec.draft.margin.to_le_bytes());
        items.push(("draft_request_state".to_string(), request));
        let pending = gpu
            .download_f32(&spec.pending_hidden)
            .map_err(|e| format!("lab_lane_items: pending hidden: {e}"))?;
        items.push((
            "pending_hidden".to_string(),
            pending.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        Ok(items)
    }

    /// Disjoint borrows of the `epochs` lanes (plan order) and the step
    /// output buffers.
    pub(crate) fn step_parts(&mut self, epochs: &[RequestEpoch]) -> Result<StepParts<'_>, String> {
        let Self {
            table,
            policy,
            logits,
            top1,
            top1_host,
            rows_cap,
            spec,
            ..
        } = self;
        Ok(StepParts {
            lanes: table.pick_mut(epochs)?,
            logits,
            top1,
            policy: *policy,
            top1_host,
            rows_cap: *rows_cap,
            spec: spec.as_mut(),
        })
    }

    /// Disjoint borrows of the `epochs` lanes (request order) and the MTP
    /// buffers, for the draft phase.
    pub(crate) fn draft_parts(
        &mut self,
        epochs: &[RequestEpoch],
    ) -> Result<(Vec<&mut Qwen4LaneState>, Option<&mut SpecStore>), String> {
        let Self { table, spec, .. } = self;
        Ok((table.pick_mut(epochs)?, spec.as_mut()))
    }

    pub fn receipt(&self, gpu: &Gpu) -> LaneStoreReceipt {
        let mapped: usize = self
            .table
            .cells
            .iter()
            .flatten()
            .map(|lane| {
                lane.state.mapped_context_bytes(gpu).unwrap_or(0)
                    + lane
                        .spec
                        .as_ref()
                        .map_or(0, |spec| spec.head.mapped_context_bytes(gpu).unwrap_or(0))
            })
            .sum();
        LaneStoreReceipt {
            kv_backend: self.backend.name(),
            qsa_format: self.format.qsa.name(),
            gdn_format: self.format.gdn.name(),
            ring_rows: self.ring_rows,
            max_lanes: self.max_lanes(),
            row_budget: self.row_budget,
            mapped_bytes: mapped as u64,
            stage_policy: self.policy.describe(),
            stage_evidence: self.policy.evidence(),
            spec: if self.spec.is_some() { "mtp" } else { "off" },
            spec_k: self.spec.as_ref().map_or(0, |s| s.config.k),
        }
    }

    pub fn stage_policy(&self) -> LaneStagePolicy {
        self.policy
    }

    pub fn set_stage_policy(&mut self, policy: LaneStagePolicy) {
        self.policy = policy;
    }

    pub fn max_lanes(&self) -> usize {
        self.table.max_lanes()
    }

    /// Most AR rows one step may carry.
    pub fn row_budget(&self) -> usize {
        self.row_budget
    }

    /// Live lanes.
    pub fn active(&self) -> usize {
        self.table.active()
    }

    /// Free every lane state and the step buffers; the first error wins.
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        let Self {
            table,
            logits,
            top1,
            spec,
            ..
        } = self;
        let mut first: Option<String> = None;
        for lane in table.into_cells() {
            let Qwen4LaneState { state, spec, .. } = lane;
            if let Err(error) = state.free_gpu(gpu) {
                first.get_or_insert(format!("lane state: {error}"));
            }
            if let Some(LaneSpec {
                head,
                pending_hidden,
                ..
            }) = spec
            {
                if let Some(error) = head.free_gpu(gpu) {
                    first.get_or_insert(format!("lane mtp state: {error}"));
                }
                if let Err(error) = gpu.free_tensor(pending_hidden) {
                    first.get_or_insert(format!("lane pending hidden: {error}"));
                }
            }
        }
        if let Some(SpecStore {
            hidden,
            draft_top1,
            append_scratch,
            ..
        }) = spec
        {
            for (name, tensor) in [("hidden capture", hidden), ("draft top1", draft_top1)] {
                if let Err(error) = gpu.free_tensor(tensor) {
                    first.get_or_insert(format!("lane {name}: {error}"));
                }
            }
            if let Some(scratch) = append_scratch {
                if let Err(error) = scratch.free_gpu(gpu) {
                    first.get_or_insert(format!("lane append scratch: {error}"));
                }
            }
        }
        for (name, tensor) in [("logits", logits), ("top1", top1)] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("lane {name}: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCell {
        host: LaneHost,
        payload: u32,
    }

    impl LaneCell for TestCell {
        type Payload = u32;

        fn assemble(host: LaneHost, payload: u32) -> Self {
            Self { host, payload }
        }

        fn host(&self) -> &LaneHost {
            &self.host
        }

        fn host_mut(&mut self) -> &mut LaneHost {
            &mut self.host
        }
    }

    fn epoch(tag: u64, generation: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: generation,
        }
    }

    /// Admission with a counting payload factory.
    fn admit(
        table: &mut LaneTable<TestCell>,
        epoch: RequestEpoch,
        prompt: &[u32],
        made: &mut usize,
    ) -> Result<usize, String> {
        table.admit(epoch, prompt.to_vec(), |slot, existing| match existing {
            Some(cell) => {
                cell.payload += 100;
                Ok(None)
            }
            None => {
                *made += 1;
                Ok(Some(slot as u32))
            }
        })
    }

    #[test]
    fn admission_takes_the_lowest_free_slot() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        assert_eq!(admit(&mut table, epoch(1, 1), &[7], &mut made), Ok(0));
        assert_eq!(admit(&mut table, epoch(2, 1), &[7], &mut made), Ok(1));
        assert_eq!(admit(&mut table, epoch(3, 1), &[7], &mut made), Ok(2));
        table.release(&epoch(2, 1)).unwrap();
        assert_eq!(admit(&mut table, epoch(4, 1), &[7], &mut made), Ok(1));
        assert_eq!(table.active(), 3);
        let host = &table.get(&epoch(4, 1)).unwrap().host;
        assert_eq!(host.status, LaneStatus::Prefilling);
        assert_eq!((host.prefilled, host.generated.len(), host.pending_seed), (0, 0, None));
        assert_eq!(host.stamp, 3);
        assert_eq!(table.epochs(), vec![epoch(1, 1), epoch(4, 1), epoch(3, 1)]);
    }

    #[test]
    fn admission_refusals() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(admit(&mut table, epoch(1, 0), &[7], &mut made)
            .unwrap_err()
            .contains("generation 0"));
        assert!(admit(&mut table, epoch(1, 1), &[], &mut made)
            .unwrap_err()
            .contains("empty prompt"));
        assert_eq!(made, 0);
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        assert!(admit(&mut table, epoch(1, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        assert!(admit(&mut table, epoch(3, 1), &[7], &mut made)
            .unwrap_err()
            .contains("busy"));
        assert_eq!(made, 2);
        assert_eq!(table.active(), 2);
    }

    /// Promoted admission with a counting payload factory.
    fn admit_decoding(
        table: &mut LaneTable<TestCell>,
        epoch: RequestEpoch,
        prompt: &[u32],
        made: &mut usize,
    ) -> Result<usize, String> {
        table.admit_decoding(epoch, prompt.to_vec(), vec![4, 5], 6, |slot, existing| {
            match existing {
                Some(cell) => {
                    cell.payload += 100;
                    Ok(None)
                }
                None => {
                    *made += 1;
                    Ok(Some(slot as u32))
                }
            }
        })
    }

    #[test]
    fn promoted_admission_publishes_a_decoding_lane() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert_eq!(admit_decoding(&mut table, epoch(1, 1), &[7, 8, 9], &mut made), Ok(0));
        let host = &table.get(&epoch(1, 1)).unwrap().host;
        assert_eq!(host.status, LaneStatus::Decoding);
        assert_eq!(host.prompt, vec![7, 8, 9]);
        assert_eq!(host.prefilled, 3);
        assert_eq!(host.generated, vec![4, 5]);
        assert_eq!(host.pending_seed, Some(6));
        assert_eq!((host.slot, host.stamp), (0, 0));
        assert_eq!(table.active(), 1);
        // A plain admission still starts Prefilling next to it, in order.
        assert_eq!(admit(&mut table, epoch(2, 1), &[7], &mut made), Ok(1));
        assert_eq!(table.get(&epoch(2, 1)).unwrap().host.status, LaneStatus::Prefilling);
        assert_eq!(table.get(&epoch(2, 1)).unwrap().host.stamp, 1);
        // The slot's payload is kept across a release and reused by a
        // promoted admission without remaking it.
        table.release(&epoch(1, 1)).unwrap();
        assert_eq!(admit_decoding(&mut table, epoch(3, 1), &[9, 9], &mut made), Ok(0));
        assert_eq!(made, 2);
        let cell = table.get(&epoch(3, 1)).unwrap();
        assert_eq!(cell.payload, 100);
        assert_eq!(cell.host.status, LaneStatus::Decoding);
        assert_eq!(cell.host.prefilled, 2);
        assert_eq!(cell.host.stamp, 2);
    }

    #[test]
    fn promoted_admission_refusals() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(admit_decoding(&mut table, epoch(1, 0), &[7], &mut made)
            .unwrap_err()
            .contains("generation 0"));
        assert!(admit_decoding(&mut table, epoch(1, 1), &[], &mut made)
            .unwrap_err()
            .contains("empty prompt"));
        assert_eq!((made, table.active(), table.next_stamp), (0, 0, 0));
        admit_decoding(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        assert!(admit_decoding(&mut table, epoch(1, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        // A live plain admission is a duplicate too.
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        assert!(admit_decoding(&mut table, epoch(2, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        assert!(admit_decoding(&mut table, epoch(3, 1), &[7], &mut made)
            .unwrap_err()
            .contains("busy"));
        assert_eq!((made, table.active(), table.next_stamp), (2, 2, 2));
        // A failed prepare leaves no half-published lane.
        table.release(&epoch(2, 1)).unwrap();
        let error = table
            .admit_decoding(epoch(4, 1), vec![7], vec![1], 2, |_, _| Err("no memory".to_string()))
            .unwrap_err();
        assert_eq!(error, "no memory");
        assert!(table.get(&epoch(4, 1)).is_none());
        assert_eq!((table.active(), table.next_stamp), (1, 2));
    }

    #[test]
    fn a_failed_prepare_leaves_the_table_untouched() {
        let mut table = LaneTable::<TestCell>::new(1);
        let error = table
            .admit(epoch(1, 1), vec![7], |_, _| Err("no memory".to_string()))
            .unwrap_err();
        assert_eq!(error, "no memory");
        assert_eq!(table.active(), 0);
        assert_eq!(table.next_stamp, 0);
        assert!(table.get(&epoch(1, 1)).is_none());
    }

    #[test]
    fn release_keeps_the_payload_and_reuse_does_not_remake_it() {
        let mut table = LaneTable::<TestCell>::new(1);
        let mut made = 0;
        admit(&mut table, epoch(1, 1), &[7, 8], &mut made).unwrap();
        assert_eq!(table.get(&epoch(1, 1)).unwrap().payload, 0);
        table.release(&epoch(1, 1)).unwrap();
        assert_eq!(table.active(), 0);
        assert!(table.get(&epoch(1, 1)).is_none());
        admit(&mut table, epoch(2, 1), &[9], &mut made).unwrap();
        assert_eq!(made, 1);
        let cell = table.get(&epoch(2, 1)).unwrap();
        assert_eq!(cell.payload, 100);
        assert_eq!(cell.host.prompt, vec![9]);
        assert_eq!(cell.host.stamp, 1);
    }

    #[test]
    fn release_of_an_unknown_epoch_fails() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(table.release(&epoch(9, 1)).is_err());
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        table.release(&epoch(1, 1)).unwrap();
        assert!(table.release(&epoch(1, 1)).is_err());
    }

    #[test]
    fn a_new_generation_of_the_same_tag_is_admitted_after_release() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        admit(&mut table, epoch(5, 1), &[7], &mut made).unwrap();
        // Same tag, other generation: a different owner, not a duplicate.
        assert!(admit(&mut table, epoch(5, 2), &[7], &mut made).is_ok());
        table.release(&epoch(5, 1)).unwrap();
        // The stale generation matches nothing, the live one still does.
        assert!(table.get(&epoch(5, 1)).is_none());
        assert!(table.get(&epoch(5, 2)).is_some());
        assert!(table.release(&epoch(5, 1)).is_err());
        admit(&mut table, epoch(5, 3), &[7], &mut made).unwrap();
        assert!(table.get(&epoch(5, 3)).is_some());
        assert!(table.get(&epoch(5, 1)).is_none());
    }

    #[test]
    fn poison_marks_only_live_matching_lanes() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        table.poison(&epoch(1, 2));
        table.poison(&epoch(9, 1));
        assert_eq!(table.get(&epoch(1, 1)).unwrap().host.status, LaneStatus::Prefilling);
        table.poison(&epoch(1, 1));
        assert_eq!(table.get(&epoch(1, 1)).unwrap().host.status, LaneStatus::Poisoned);
        assert_eq!(table.get(&epoch(2, 1)).unwrap().host.status, LaneStatus::Prefilling);
    }

    #[test]
    fn force_seed_needs_a_live_decoding_lane() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        for tag in 1..=3 {
            admit(&mut table, epoch(tag, 1), &[7], &mut made).unwrap();
        }
        // Prefilling lane: refused.
        assert!(table.force_seed(&epoch(1, 1), 5).is_err());
        table.get_mut(&epoch(1, 1)).unwrap().host.status = LaneStatus::Decoding;
        table.get_mut(&epoch(1, 1)).unwrap().host.generated.push(4);
        table.force_seed(&epoch(1, 1), 5).unwrap();
        let host = &table.get(&epoch(1, 1)).unwrap().host;
        assert_eq!(host.pending_seed, Some(5));
        assert_eq!(host.generated, vec![4]);
        // Poisoned lane and unknown epoch: refused.
        table.get_mut(&epoch(2, 1)).unwrap().host.status = LaneStatus::Decoding;
        table.poison(&epoch(2, 1));
        assert!(table.force_seed(&epoch(2, 1), 5).is_err());
        assert!(table.force_seed(&epoch(9, 1), 5).is_err());
        assert!(table.force_seed(&epoch(1, 2), 5).is_err());
    }

    #[test]
    fn pick_mut_returns_disjoint_lanes_in_request_order() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        for tag in 1..=3 {
            admit(&mut table, epoch(tag, 1), &[7], &mut made).unwrap();
        }
        let picked = table
            .pick_mut(&[epoch(3, 1), epoch(1, 1)])
            .unwrap();
        assert_eq!(picked.iter().map(|c| c.host.slot).collect::<Vec<_>>(), vec![2, 0]);
        assert!(table.pick_mut(&[epoch(1, 1), epoch(1, 1)]).is_err());
        assert!(table.pick_mut(&[epoch(1, 9)]).is_err());
    }
}
