// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! Eager Flash-Next (qwen4) whole-layer routed-MoE prefill on a second,
//! integrated gfx1151 device (`memory.offload_exec = igpu`).
//!
//! One [`IgpuEngine`] owns the second [`Gpu`], the reusable max-row buffers and
//! the kernel set; each admitted layer holds one [`IgpuLayerEngine`] (an `Arc`
//! onto the shared engine plus that layer's two per-expert pointer tables over
//! borrowed host weights and the weight lease that keeps them alive). The
//! engine is layer-set agnostic: a layer is engined iff the caller binds it,
//! and binding needs only complete portable host backing for all of the
//! layer's experts.
//!
//! ABI (routed-only): input is the NATURAL normalized F32 `x_norm`
//! `[rows, hidden]` plus the original I32 expert ids and F32 router weights
//! `[rows, 10]`, unchanged and in original slot order; output is the canonical
//! BF16-rounded combine as F32 `[rows, hidden]`, computed from +0
//! (`initial_zero` MUST be `true`; there is no target download and no
//! residual). The shared expert and the HC fold stay on the model device.
//!
//! Per call, [`IgpuLayerEngine::execute`] does:
//!
//! 1. model device: D2H of `x_norm`, `ids`, `weights` into the input bank of
//!    one preallocated `hipHostMalloc(MAPPED | PORTABLE)` block, on the model's
//!    active stream, then a stream sync;
//! 2. iGPU: the production symmetric-IU4 pipeline over zero-copy views of that
//!    block (`moe_scatter_stable_top10` -> `qwen4_moe_rotate256_i4` ->
//!    `gemm_qwen4_moe_gate_up_silu_iu4_sym` -> `qwen4_moe_rotate128_i4` ->
//!    `gemm_qwen4_moe_down_iu4_sym` -> `moe_down_combine_grouped_top10_bf16in`
//!    into the output bank), then a device sync;
//! 3. model device: H2D of the output bank into `target` on the same stream,
//!    then a stream sync. On return both devices' readers of both banks have
//!    retired, so the next call (or a drop) never races a copy.
//!
//! Every A4 call is forced regardless of the production >=512-row threshold:
//! the engine never consults `route_rows` / `qwen4_moe_sym_iu4_applies`, so
//! any width the caller dispatches (the arch only dispatches > 4 rows) runs
//! the same recipe, tails included. [`IgpuManifest`] freezes that recipe and
//! its digest is what caches must be bound to.
//!
//! Nothing allocates on the hot path: the host block, the grouped
//! intermediates, the combine order scratch and the int4 sidecars are all
//! sized for `max_rows` at [`IgpuEngine::open`], and every kernel is loaded by
//! the warm-up in the first [`IgpuEngine::bind_layer`]. Each device is only
//! ever selected through `Gpu::bind_thread`; there is no raw `set_device`.
//!
//! Failures (see [`IgpuFailure`]): a call rejected before the first enqueue
//! changes nothing and does not poison. Any error after work may have been
//! submitted drains BOTH devices, then poisons the engine for good (every
//! later `execute` errors; never retry the same logical forward, never fall
//! back to another arithmetic after a fault). If that drain itself fails, or a
//! call never returned, the engine is [`IgpuEngine::unsafe_to_release`]: its
//! `Drop` leaks the second `Gpu`, the host block, the pointer tables and the
//! weight leases rather than free memory a device may still touch.

use std::any::Any;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use hip_bridge::{DeviceBuffer, HIP_HOST_MALLOC_MAPPED, HIP_HOST_MALLOC_PORTABLE};
use rdna_compute::scratch::{qwen4_grouped_rows_bound, qwen4_moe_group_layout_checked};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::types::DispatchError;

/// Only architecture the engine runs on.
pub const IGPU_ARCH: &str = "gfx1151";
/// Routed experts per token (the combine kernel is `top10`).
pub const IGPU_TOP_K: usize = 10;
/// Narrowest forward the arch dispatches to the engine (`> 4` rows). The
/// engine itself accepts any `1..=max_rows`, always on the A4 recipe.
pub const IGPU_MIN_DISPATCH_ROWS: usize = 5;
/// The frozen C2 producer candidate set (`-DIU4_A4_CANDIDATES=2`).
const IU4_A4_CANDIDATES: u32 = 2;
/// Bumped whenever the frozen recipe text in [`IgpuManifest`] changes.
pub const IGPU_MANIFEST_VERSION: u32 = 1;
/// QT44 gate/up bytes per K256 group.
const QT44_GROUP_BYTES: usize = 136;
/// QT53 down bytes per K128 group.
const QT53_GROUP_BYTES: usize = 68;
/// Combine rank-order scratch bytes per routed slot.
const ORDER_BYTES_PER_SLOT: usize = 8;
/// The PM grouped GEMMs index slots with a 22-bit-exact reciprocal.
const PM_MAX_SLOTS: usize = 1 << 22;
/// Builder down entries store Y through a 32-bit byte offset.
const DOWN_EXTENT_LIMIT_BYTES: usize = u32::MAX as usize;
/// Region alignment inside the pinned I/O block.
const IO_ALIGN: usize = 256;
/// Most parity records retained across all layers.
pub const IGPU_PARITY_MAX_RECORDS: usize = 1024;

// ───────────────────────────── arming (pure) ─────────────────────────────

/// Facts [`igpu_refusal`] decides on. `second_arch` is empty when no second
/// device exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct IgpuArmInputs<'a> {
    /// `memory.offload_exec = igpu` was requested. When `false` there is
    /// nothing to arm or refuse ([`IgpuArming::NotRequested`]).
    pub requested: bool,
    pub primary_arch: &'a str,
    pub second_arch: &'a str,
    pub primary_device: i32,
    pub second_device: i32,
    /// An emulated device-memory budget (`emu`) is active.
    pub emu: bool,
    /// Expert parallelism.
    pub ep: bool,
    /// Pipeline parallelism.
    pub pp: bool,
    /// Sealed expert-slab layout.
    pub slab: bool,
    /// Per-expert tiered (mixed VRAM/host) experts.
    pub tier: bool,
    /// A device-resident hot-set expert placement (hot experts kept in VRAM
    /// while the rest are host-backed). The engine needs complete host
    /// backing of every expert of an admitted layer, not a hot/cold split.
    pub hotset: bool,
    /// AWQ-scaled expert weights.
    pub awq: bool,
    /// Experts of one layer do not share one codec/stride (QT44 gate/up + QT53 down).
    pub nonuniform_experts: bool,
    /// Every admitted layer is placed whole on the host (no per-expert tier
    /// inside a layer). Must be `true` to arm.
    pub host_whole_layers_only: bool,
}

/// Outcome of the pure arming decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IgpuArming {
    /// The mode was not requested: a no-op, not a rejection.
    NotRequested,
    /// Requested and every fact holds.
    Armed,
    /// Requested but refused, with the one-line reason.
    Refused(String),
}

/// The pure arming decision. Pure: the caller supplies every fact.
pub fn igpu_arming(i: &IgpuArmInputs<'_>) -> IgpuArming {
    if !i.requested {
        return IgpuArming::NotRequested;
    }
    let no = |why: &str| IgpuArming::Refused(format!("memory.offload_exec=igpu refused: {why}"));
    if i.second_arch.is_empty() {
        return no("no second (integrated gfx1151) device is present");
    }
    if i.second_arch != IGPU_ARCH {
        return no(&format!(
            "second device is {}, not {IGPU_ARCH}",
            i.second_arch
        ));
    }
    if i.primary_device == i.second_device {
        return no("the second device is the primary device");
    }
    if i.emu {
        return no("an emulated device budget (emu) is active");
    }
    if i.ep {
        return no("expert parallelism is configured");
    }
    if i.pp {
        return no("pipeline parallelism is configured");
    }
    if i.slab {
        return no("sealed expert-slab layers are not supported");
    }
    if i.tier {
        return no("per-expert tiered experts are not supported");
    }
    if i.hotset {
        return no("a hot-set expert placement is not supported");
    }
    if i.awq {
        return no("AWQ expert weights are not supported");
    }
    if i.nonuniform_experts {
        return no("experts are not one uniform QT44 gate/up + QT53 down codec");
    }
    if !i.host_whole_layers_only {
        return no("host placement is not whole-layer");
    }
    IgpuArming::Armed
}

/// `Err(one-line reason)` iff the mode was requested and must refuse; `Ok`
/// when armed or when the mode was not requested (see [`igpu_arming`] to tell
/// those apart).
pub fn igpu_refusal(i: &IgpuArmInputs<'_>) -> Result<(), String> {
    match igpu_arming(i) {
        IgpuArming::Refused(why) => Err(why),
        IgpuArming::NotRequested | IgpuArming::Armed => Ok(()),
    }
}

/// Pure parse of `HIPFIRE_IGPU_DEVICE`: a non-negative device index.
pub fn parse_igpu_device(raw: Option<&str>) -> Option<i32> {
    raw.and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|&d| d >= 0)
}

/// `HIPFIRE_IGPU_DEVICE` (developer variable), if set and valid.
pub fn igpu_device_index() -> Option<i32> {
    parse_igpu_device(hipfire_config::developer_var("HIPFIRE_IGPU_DEVICE").ok().as_deref())
}

/// Why a call may not run while the model device records a graph / replay
/// body (mirrors the CPU expert splice: the engine is a host sync point).
pub fn capture_refusal(
    capture_mode: bool,
    replay_recording: bool,
    verify_capturing: bool,
    replay_capturing: bool,
) -> Option<&'static str> {
    if capture_mode || verify_capturing || replay_capturing || replay_recording {
        return Some(
            "igpu whole-layer MoE refuses graph capture / replay recording: it is a host sync point",
        );
    }
    None
}

// ───────────────────────────── numerical manifest ─────────────────────────────

/// The frozen arithmetic of an engined call. Everything that changes the
/// bits of the routed output is in [`IgpuManifest::canonical`]; capacity
/// (`max_rows`) and device index are deliberately not. A cache or snapshot
/// produced under one digest must not be reused under another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IgpuManifest {
    pub version: u32,
    pub arch: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub k_top: usize,
    pub a4_candidates: u32,
    pub gate_up_kernel: String,
    pub down_kernel: String,
    pub down_row_repeat: usize,
}

impl IgpuManifest {
    /// The canonical, line-stable recipe text hashed by [`Self::digest`].
    pub fn canonical(&self) -> String {
        format!(
            "igpu-moe-manifest v{}\n\
             arch={}\n\
             weights=QT44 gate/up K256 {QT44_GROUP_BYTES}B symmetric, QT53 down K128 {QT53_GROUP_BYTES}B symmetric, uniform, no AWQ\n\
             geometry hidden={} intermediate={} experts={} top_k={}\n\
             input=natural F32 x_norm, original i32 ids and F32 router weights, original slot order\n\
             scatter=qwen4_moe_scatter_stable_top10 pad16 stable (expert, flat slot)\n\
             a4_producer=qwen4_moe_rotate256_i4 FWHT256 block_i4_128 C2 IU4_A4_CANDIDATES={}\n\
             gate_up={} swiglu bf16 epilogue\n\
             down_producer=qwen4_moe_rotate128_i4 FWHT128 block_i4_128 C2\n\
             down={} row_repeat={}\n\
             combine=moe_down_combine_grouped_top10_bf16in_zinit canonical ascending expert id bf16 rne every add, weight once\n\
             output=F32 carrying bf16 values, routed-only from +0\n\
             rows=A4 forced at every width 1..=max_rows, no 512-row threshold, tails included\n\
             dispatch_min_rows={IGPU_MIN_DISPATCH_ROWS}\n",
            self.version,
            self.arch,
            self.hidden,
            self.intermediate,
            self.experts,
            self.k_top,
            self.a4_candidates,
            self.gate_up_kernel,
            self.down_kernel,
            self.down_row_repeat,
        )
    }

    /// FNV-1a 64 of [`Self::canonical`]; stable across processes.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.canonical().bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

// ───────────────────────────── sizing (pure) ─────────────────────────────

/// Open-time geometry of the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IgpuEngineSpec {
    /// HIP device index of the iGPU.
    pub device: i32,
    pub max_rows: usize,
    pub hidden: usize,
    pub moe_intermediate: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Candidate layers the caller will try to bind. Informational only: the
    /// engine neither sizes nor reports from it.
    pub layers: usize,
}

/// Every byte extent the engine allocates or validates, with checked
/// arithmetic. The ABI limits of the grouped IU4 kernels are enforced here so
/// no launch can exceed them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IgpuSizing {
    pub max_rows: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub k_top: usize,
    pub max_slots: usize,
    pub max_grouped_rows: usize,
    pub x_off: usize,
    pub x_bytes: usize,
    pub ids_off: usize,
    pub ids_bytes: usize,
    pub weights_off: usize,
    pub weights_bytes: usize,
    /// Output bank (kept apart from the input bank above).
    pub target_off: usize,
    pub target_bytes: usize,
    /// Whole pinned portable I/O block.
    pub io_bytes: usize,
    pub counts_bytes: usize,
    pub offsets_bytes: usize,
    pub sorted_bytes: usize,
    pub tiles_bytes: usize,
    pub inverse_bytes: usize,
    pub y_gate_up_bytes: usize,
    pub y_down_bytes: usize,
    pub order_bytes: usize,
    /// Exact bytes of one expert's gate/up (QT44, `2*mi` rows of `hidden`).
    pub gate_up_expert_bytes: usize,
    /// Exact bytes of one expert's down (QT53, `hidden` rows of `mi`).
    pub down_expert_bytes: usize,
}

fn align_up(v: usize, a: usize) -> Option<usize> {
    v.checked_add(a - 1).map(|x| x / a * a)
}

/// Padded grouped-row bound of `slots` routed slots (same bound the
/// production prefill uses); `None` on overflow.
pub fn igpu_grouped_rows(slots: usize, experts: usize) -> Option<usize> {
    qwen4_grouped_rows_bound(slots, experts)
}

impl IgpuSizing {
    pub fn new(spec: &IgpuEngineSpec) -> Result<Self, String> {
        let e = |m: &str| -> Result<Self, String> { Err(format!("igpu engine sizing: {m}")) };
        let overflow = || "igpu engine sizing: extent overflows".to_string();
        let (rows, hidden, mi, experts, k) = (
            spec.max_rows,
            spec.hidden,
            spec.moe_intermediate,
            spec.num_experts,
            spec.top_k,
        );
        if spec.device < 0 {
            return e("negative device index");
        }
        if rows == 0 || hidden == 0 || mi == 0 || experts == 0 {
            return e("max_rows, hidden, moe_intermediate and num_experts must be nonzero");
        }
        if k != IGPU_TOP_K {
            return e("top_k must be 10 (the combine kernel is top10)");
        }
        if hidden % 256 != 0 {
            return e("hidden must be a multiple of 256 (QT44 K256, down row blocks of 64*4)");
        }
        if mi % 128 != 0 {
            return e("moe_intermediate must be a multiple of 128");
        }
        if experts > 1024 {
            return e("num_experts must be <= 1024 (single-block prefix scan)");
        }
        let slots = rows.checked_mul(k).ok_or_else(overflow)?;
        if slots > PM_MAX_SLOTS {
            return e("max_rows * top_k exceeds the 2^22 slot limit of the PM grouped GEMMs");
        }
        let grouped = igpu_grouped_rows(slots, experts)
            .filter(|g| g % 16 == 0)
            .ok_or_else(overflow)?;
        // Every extent a kernel receives as i32.
        for v in [grouped, slots, hidden, 2 * mi, experts, rows] {
            if i32::try_from(v).is_err() {
                return e("an extent exceeds i32");
            }
        }
        let mul = |a: usize, b: usize| -> Result<usize, String> { a.checked_mul(b).ok_or_else(overflow) };
        let y_down_bytes = mul(mul(grouped, hidden)?, 2)?;
        if y_down_bytes > DOWN_EXTENT_LIMIT_BYTES {
            return e("grouped down output exceeds the 32-bit byte offset of the builder down GEMM");
        }
        let y_gate_up_bytes = mul(mul(grouped, mi)?, 2)?;
        let x_bytes = mul(mul(rows, hidden)?, 4)?;
        let ids_bytes = mul(slots, 4)?;
        let weights_bytes = ids_bytes;
        let target_bytes = x_bytes;
        let ids_off = align_up(x_bytes, IO_ALIGN).ok_or_else(overflow)?;
        let weights_off = ids_off
            .checked_add(ids_bytes)
            .and_then(|v| align_up(v, IO_ALIGN))
            .ok_or_else(overflow)?;
        let target_off = weights_off
            .checked_add(weights_bytes)
            .and_then(|v| align_up(v, IO_ALIGN))
            .ok_or_else(overflow)?;
        let io_bytes = target_off.checked_add(target_bytes).ok_or_else(overflow)?;
        // The grouping scratch layout `reserve_qwen4_moe_sym` allocates.
        qwen4_moe_group_layout_checked(slots, experts).ok_or_else(overflow)?;
        let gate_up_expert_bytes = mul(mul(2 * mi, hidden / 256)?, QT44_GROUP_BYTES)?;
        let down_expert_bytes = mul(mul(hidden, mi / 128)?, QT53_GROUP_BYTES)?;
        Ok(Self {
            max_rows: rows,
            hidden,
            intermediate: mi,
            experts,
            k_top: k,
            max_slots: slots,
            max_grouped_rows: grouped,
            x_off: 0,
            x_bytes,
            ids_off,
            ids_bytes,
            weights_off,
            weights_bytes,
            target_off,
            target_bytes,
            io_bytes,
            // Padded so a kernel that reads a vector past the tail stays inside.
            counts_bytes: mul(experts + 16, 4)?,
            offsets_bytes: mul(experts + 17, 4)?,
            sorted_bytes: mul(grouped + 16, 4)?,
            tiles_bytes: mul(grouped + 16, 4)?,
            inverse_bytes: mul(slots + 16, 4)?,
            y_gate_up_bytes,
            y_down_bytes,
            order_bytes: mul(slots, ORDER_BYTES_PER_SLOT)?,
            gate_up_expert_bytes,
            down_expert_bytes,
        })
    }

    /// Device-side grouped intermediates (excluding the int4 sidecars).
    pub fn grouped_scratch_bytes(&self) -> usize {
        self.counts_bytes
            + self.offsets_bytes
            + self.sorted_bytes
            + self.tiles_bytes
            + self.inverse_bytes
            + self.y_gate_up_bytes
            + self.y_down_bytes
            + self.order_bytes
    }

    /// Checked byte spans of a live `rows`-row call.
    pub fn live(&self, rows: usize) -> Result<IgpuLive, String> {
        if rows == 0 || rows > self.max_rows {
            return Err(format!(
                "igpu engine: rows {rows} outside 1..={}",
                self.max_rows
            ));
        }
        let slots = rows * self.k_top;
        let grouped = igpu_grouped_rows(slots, self.experts)
            .filter(|&g| g <= self.max_grouped_rows && g % 16 == 0)
            .ok_or("igpu engine: grouped rows exceed the reserved maximum")?;
        Ok(IgpuLive {
            rows,
            slots,
            grouped_rows: grouped,
            x_bytes: rows * self.hidden * 4,
            ids_bytes: slots * 4,
            weights_bytes: slots * 4,
            target_bytes: rows * self.hidden * 4,
        })
    }
}

/// Live-call extents inside the reserved maxima.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IgpuLive {
    pub rows: usize,
    pub slots: usize,
    pub grouped_rows: usize,
    pub x_bytes: usize,
    pub ids_bytes: usize,
    pub weights_bytes: usize,
    pub target_bytes: usize,
}

// ───────────────────────────── failure classes ─────────────────────────────

/// What a failed call left behind (the poisoning classes; a call rejected
/// before its first enqueue is neither, and changes nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IgpuFailure {
    /// Work may have been submitted; both devices were drained and the engine
    /// is poisoned. The model state may be partially advanced by the caller's
    /// own earlier layers: restore a checkpoint or invalidate, never retry.
    SubmittedDrained,
    /// A drain failed: memory may still be in flight. The engine, its host
    /// block, its tables and its weight leases are leaked, and the borrowed
    /// host weight owners must be retained.
    DrainFailed,
}

const FAIL_NONE: u8 = 0;
const FAIL_SUBMITTED: u8 = 1;
const FAIL_DRAIN: u8 = 2;

// ───────────────────────────── parity records ─────────────────────────────

/// One engined call, captured in parity mode.
#[derive(Clone, Debug)]
pub struct IgpuParityRecord {
    pub layer: usize,
    pub rows: usize,
    pub hidden: usize,
    pub k_top: usize,
    pub initial_zero: bool,
    /// F32 LE `[rows, hidden]`, natural basis.
    pub x_norm: Vec<u8>,
    /// i32 LE `[rows, k_top]`, exact bytes consumed.
    pub topk_ids: Vec<u8>,
    /// F32 LE `[rows, k_top]`.
    pub topk_weights: Vec<u8>,
    /// Always empty: the routed-only ABI starts the combine from +0 and never
    /// reads the model's target.
    pub target_in: Vec<u8>,
    /// F32 `[rows, hidden]` as returned to the model.
    pub engine_target: Vec<u8>,
    /// Same inputs through the same kernels on separate, non-pinned device buffers.
    pub igpu_standalone_target: Vec<u8>,
}

/// Input of [`IgpuLayerEngine::parity_execute`]: recorded natural activations
/// and the original route, at any row count `1..=max_rows`.
#[derive(Clone, Debug)]
pub struct IgpuParityCase {
    pub rows: usize,
    /// F32 LE `[rows, hidden]`, natural (unrotated) normalized activation.
    pub x_norm: Vec<u8>,
    /// i32 LE `[rows, k_top]`; may hold repeats and ids outside `0..experts`.
    pub topk_ids: Vec<u8>,
    /// F32 LE `[rows, k_top]`.
    pub topk_weights: Vec<u8>,
}

/// Result of [`IgpuLayerEngine::parity_execute`].
#[derive(Clone, Debug)]
pub struct IgpuParityOutcome {
    pub rows: usize,
    /// F32 `[rows, hidden]` routed-only result read back from the model
    /// device after the engine's own D2H -> bank -> iGPU -> H2D round trip.
    pub engine_target: Vec<u8>,
    /// Same inputs through the same kernels on separate non-pinned iGPU
    /// buffers, device-synced.
    pub igpu_standalone_target: Vec<u8>,
}

struct RefBufs {
    x: GpuTensor,
    ids: GpuTensor,
    weights: GpuTensor,
    target: GpuTensor,
}

#[derive(Default)]
struct Parity {
    on: bool,
    layers: Option<Vec<usize>>,
    records: Vec<IgpuParityRecord>,
    refs: Option<RefBufs>,
}

impl Parity {
    fn wants(&self, layer: usize) -> bool {
        self.on
            && self.refs.is_some()
            && self.records.len() < IGPU_PARITY_MAX_RECORDS
            && self.layers.as_ref().is_none_or(|l| l.contains(&layer))
    }
}

// ───────────────────────────── engine ─────────────────────────────

// ───────────────────────────── transfer tickets ─────────────────────────────

/// Where a call's transfer banks are in their hand-off. Transitions are
/// strictly `ExportPending -> HaloPending -> ImportPending -> Retired`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TicketPhase {
    /// The model device is exporting inputs into the input bank.
    ExportPending,
    /// Export complete; the iGPU owns the input bank and writes the output bank.
    HaloPending,
    /// iGPU complete; the output bank is the immutable source of the return H2D.
    ImportPending,
    /// Both devices' readers of both banks have retired.
    Retired,
}

/// Ownership token of one call's banks. `generation` is claimed fresh and
/// monotonically increasing per call; a ticket older than the ledger's last
/// invalidation is stale and can never publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TransferTicket {
    generation: u64,
    layer: usize,
    rows: usize,
    phase: TicketPhase,
}

/// The pure state machine behind the tickets (held under the engine mutex).
/// Plain `Copy` data: nothing here allocates on the success path.
#[derive(Debug, Default)]
struct TicketLedger {
    next: u64,
    /// Tickets with `generation < valid_from` are stale (poison/quiesce).
    valid_from: u64,
    active: Option<TransferTicket>,
}

impl TicketLedger {
    /// Claim the banks for one call. At most one ticket is in flight.
    fn claim(&mut self, layer: usize, rows: usize) -> Result<TransferTicket, String> {
        if let Some(a) = self.active {
            return Err(format!(
                "transfer banks are still owned by generation {} ({:?})",
                a.generation, a.phase
            ));
        }
        if rows == 0 {
            return Err("transfer ticket for zero rows".into());
        }
        let ticket = TransferTicket {
            generation: self.next,
            layer,
            rows,
            phase: TicketPhase::ExportPending,
        };
        self.next = self
            .next
            .checked_add(1)
            .ok_or("transfer generation overflows")?;
        self.active = Some(ticket);
        Ok(ticket)
    }

    /// `ticket` is the live, current-phase, non-stale ticket.
    fn validate(&self, ticket: &TransferTicket, phase: TicketPhase) -> Result<(), String> {
        if ticket.generation < self.valid_from {
            return Err(format!(
                "stale transfer generation {} (valid from {})",
                ticket.generation, self.valid_from
            ));
        }
        match self.active {
            Some(a) if a == *ticket && ticket.phase == phase => Ok(()),
            Some(a) if a.generation == ticket.generation => Err(format!(
                "transfer ticket {} is {:?}, expected {:?}",
                ticket.generation, a.phase, phase
            )),
            _ => Err(format!(
                "transfer ticket {} (layer {}, {} rows) is not the active ticket",
                ticket.generation, ticket.layer, ticket.rows
            )),
        }
    }

    /// Move `ticket` to its single legal next phase and return the new ticket.
    fn advance(
        &mut self,
        ticket: &TransferTicket,
        to: TicketPhase,
    ) -> Result<TransferTicket, String> {
        let expected = match ticket.phase {
            TicketPhase::ExportPending => TicketPhase::HaloPending,
            TicketPhase::HaloPending => TicketPhase::ImportPending,
            TicketPhase::ImportPending => TicketPhase::Retired,
            TicketPhase::Retired => return Err("transfer ticket is already retired".into()),
        };
        self.validate(ticket, ticket.phase)?;
        if to != expected {
            return Err(format!(
                "illegal transfer transition {:?} -> {:?} (next is {:?})",
                ticket.phase, to, expected
            ));
        }
        let next = TransferTicket {
            phase: to,
            ..*ticket
        };
        self.active = (to != TicketPhase::Retired).then_some(next);
        Ok(next)
    }

    /// Retire an `ImportPending` ticket (after the model sync) and free the banks.
    fn retire(&mut self, ticket: &TransferTicket) -> Result<(), String> {
        if ticket.phase != TicketPhase::ImportPending {
            return Err(format!(
                "cannot retire a {:?} transfer ticket",
                ticket.phase
            ));
        }
        self.advance(ticket, TicketPhase::Retired).map(|_| ())
    }

    /// Poison or quiesce: every ticket claimed so far becomes stale.
    fn invalidate(&mut self) {
        self.valid_from = self.next;
    }

    /// After a successful drain of both devices: nothing is in flight.
    fn clear_after_drain(&mut self) {
        self.invalidate();
        self.active = None;
    }
}

struct Io {
    host: *mut c_void,
    x: GpuTensor,
    ids: GpuTensor,
    weights: GpuTensor,
    target: GpuTensor,
}

struct Bufs {
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    y_gate_up: GpuTensor,
    y_down: GpuTensor,
    order: GpuTensor,
}

struct State {
    gpu: Option<Gpu>,
    io: Option<Io>,
    bufs: Option<Bufs>,
    /// Device the model Gpu runs on, recorded at the first call.
    model_device: Option<i32>,
    /// A call is between its first enqueue and its verified retirement. Still
    /// set at drop time (panic, or a failed drain) means unsafe to release.
    inflight: bool,
    /// Generation-tagged ownership of the transfer banks (one call at a time).
    ledger: TicketLedger,
    warmed: bool,
    bound_layers: usize,
    parity: Parity,
}

// SAFETY: `State` holds raw device/host pointers and a `Gpu`; every access is
// serialized by the engine mutex and each HIP call binds its device first.
unsafe impl Send for State {}

/// Shared second-device engine: one `Gpu`, one set of max-row buffers.
pub struct IgpuEngine {
    sizing: IgpuSizing,
    manifest: IgpuManifest,
    device: i32,
    poisoned: AtomicBool,
    unsafe_release: AtomicBool,
    failure: AtomicU8,
    generation: AtomicU64,
    state: Mutex<State>,
}

fn derr(m: impl Into<String>) -> DispatchError {
    DispatchError::Hip(m.into())
}

/// A call refused before its first enqueue (nothing changed).
fn rejected(m: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("igpu rejected before enqueue: {m}"))
}

/// A mutable byte window of the pinned I/O block.
///
/// The caller (holding the engine mutex) guarantees `off + len` lies inside the
/// block `host` addresses and that no other host access overlaps the window
/// while it is used; the HIP copies that touch it are synchronized before any
/// other read.
fn host_span<'a>(host: *mut u8, off: usize, len: usize) -> &'a mut [u8] {
    // SAFETY: see above; extents come from `IgpuSizing::live`, which is checked
    // against the block size.
    unsafe { std::slice::from_raw_parts_mut(host.add(off), len) }
}

fn hip<T>(r: Result<T, hip_bridge::HipError>) -> Result<T, DispatchError> {
    r.map_err(|e| DispatchError::Hip(format!("igpu engine: {e}")))
}

fn hips<T>(r: Result<T, hip_bridge::HipError>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("igpu engine: {what}: {e}"))
}

/// A borrowed (never freed) view of `bytes` at `ptr`.
fn view(ptr: *mut c_void, bytes: usize, shape: &[usize], dtype: DType) -> GpuTensor {
    GpuTensor {
        // SAFETY: `ptr` is a live device alias of at least `bytes` bytes kept
        // alive by the engine for as long as the view exists; `from_raw` is
        // non-owning, so dropping the view frees nothing.
        buf: unsafe { DeviceBuffer::from_raw(ptr, bytes) },
        shape: shape.to_vec(),
        dtype,
    }
}

impl IgpuEngine {
    /// Open the second device and reserve everything for `spec.max_rows`.
    /// Fails closed: no host-pointer fallback, no arithmetic fallback.
    pub fn open(spec: IgpuEngineSpec) -> Result<Arc<Self>, String> {
        let sizing = IgpuSizing::new(&spec)?;
        let mut gpu = hips(Gpu::init_with_device(spec.device), "open device")?;
        if gpu.arch != IGPU_ARCH {
            return Err(format!(
                "igpu engine: device {} is {}, not {IGPU_ARCH}",
                spec.device, gpu.arch
            ));
        }
        if !gpu.qwen4_moe_sym_iu4_requested() {
            return Err(
                "igpu engine: the symmetric IU4 A4 recipe is not admitted on the second device \
                 (HIPFIRE_QWEN4_MOE_SYM_IU4=0 or a non-default IU4_A4_CANDIDATES)"
                    .into(),
            );
        }
        let [gate_up_kernel, down_kernel] = gpu
            .qwen4_moe_sym_gemm_symbols(true)
            .ok_or("igpu engine: no symmetric IU4 GEMM entries on the second device")?;
        let manifest = IgpuManifest {
            version: IGPU_MANIFEST_VERSION,
            arch: gpu.arch.clone(),
            hidden: sizing.hidden,
            intermediate: sizing.intermediate,
            experts: sizing.experts,
            k_top: sizing.k_top,
            a4_candidates: IU4_A4_CANDIDATES,
            gate_up_kernel: gate_up_kernel.to_string(),
            down_kernel: down_kernel.to_string(),
            down_row_repeat: hips(gpu.qwen4_moe_sym_down_rr(), "down row repeat")?,
        };
        hips(gpu.bind_thread(), "bind")?;
        hips(
            gpu.reserve_qwen4_moe_sym(
                sizing.max_rows,
                sizing.k_top,
                sizing.experts,
                sizing.hidden,
                sizing.intermediate,
            ),
            "reserve sym scratch",
        )?;
        // Host I/O block: pinned, mapped and portable, so the model device can
        // DMA into it and the iGPU can dereference it.
        let host = hips(
            gpu.hip.host_malloc(
                sizing.io_bytes,
                HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_PORTABLE,
            ),
            "pinned portable I/O block",
        )?;
        // SAFETY: fresh allocation of `io_bytes` bytes; zeroed so the warm-up
        // and any early parity read see finite data.
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, sizing.io_bytes) };
        let setup = (|| -> Result<(Io, Bufs), String> {
            let dev = hips(
                gpu.hip.host_get_device_pointer(host, 0),
                "iGPU alias of the I/O block",
            )?;
            if dev.is_null() {
                return Err("igpu engine: null iGPU alias of the I/O block".into());
            }
            let at = |off: usize| -> *mut c_void {
                // SAFETY: every offset lies inside the `io_bytes` block.
                unsafe { dev.cast::<u8>().add(off).cast() }
            };
            let io = Io {
                host,
                x: view(
                    at(sizing.x_off),
                    sizing.x_bytes,
                    &[sizing.max_rows, sizing.hidden],
                    DType::F32,
                ),
                ids: view(at(sizing.ids_off), sizing.ids_bytes, &[sizing.max_slots], DType::F32),
                weights: view(
                    at(sizing.weights_off),
                    sizing.weights_bytes,
                    &[sizing.max_slots],
                    DType::F32,
                ),
                target: view(
                    at(sizing.target_off),
                    sizing.target_bytes,
                    &[sizing.max_rows, sizing.hidden],
                    DType::F32,
                ),
            };
            let mut raw =
                |bytes: usize| hips(gpu.alloc_tensor(&[bytes], DType::Raw), "grouped scratch");
            let bufs = Bufs {
                counts: raw(sizing.counts_bytes)?,
                offsets: raw(sizing.offsets_bytes)?,
                sorted: raw(sizing.sorted_bytes)?,
                tiles: raw(sizing.tiles_bytes)?,
                inverse: raw(sizing.inverse_bytes)?,
                y_gate_up: raw(sizing.y_gate_up_bytes)?,
                y_down: raw(sizing.y_down_bytes)?,
                order: raw(sizing.order_bytes)?,
            };
            Ok((io, bufs))
        })();
        let (io, bufs) = match setup {
            Ok(v) => v,
            Err(e) => {
                let _ = gpu.hip.host_free(host);
                return Err(e);
            }
        };
        Ok(Arc::new(Self {
            sizing,
            manifest,
            device: spec.device,
            poisoned: AtomicBool::new(false),
            unsafe_release: AtomicBool::new(false),
            failure: AtomicU8::new(FAIL_NONE),
            generation: AtomicU64::new(0),
            state: Mutex::new(State {
                gpu: Some(gpu),
                io: Some(io),
                bufs: Some(bufs),
                model_device: None,
                inflight: false,
                ledger: TicketLedger::default(),
                warmed: false,
                bound_layers: 0,
                parity: Parity::default(),
            }),
        }))
    }

    pub fn sizing(&self) -> &IgpuSizing {
        &self.sizing
    }

    /// The frozen arithmetic recipe; bind caches/snapshots to
    /// [`IgpuManifest::digest`].
    pub fn manifest(&self) -> &IgpuManifest {
        &self.manifest
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn max_rows(&self) -> usize {
        self.sizing.max_rows
    }

    pub fn poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Why the engine is poisoned, if a call failed after submission.
    pub fn failure(&self) -> Option<IgpuFailure> {
        match self.failure.load(Ordering::Acquire) {
            FAIL_SUBMITTED => Some(IgpuFailure::SubmittedDrained),
            FAIL_DRAIN => Some(IgpuFailure::DrainFailed),
            _ => None,
        }
    }

    /// Bumped by every poisoning and every [`quiesce`](Self::quiesce): a
    /// result tagged with an older generation must never be published.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// `true` once a drain failed or a call never returned: the host weight
    /// owners this engine borrows must not be freed (leak them), because a
    /// device may still read them.
    pub fn unsafe_to_release(&self) -> bool {
        self.unsafe_release.load(Ordering::Acquire)
    }

    /// Layers currently bound to this engine (live layer engines).
    pub fn bound_layers(&self) -> usize {
        self.lock().bound_layers
    }

    /// One load line. Reports the engine-bound layer count, not the caller's
    /// candidate count (some candidates may fail to bind and stay normal).
    pub fn summary_line(&self) -> String {
        let s = &self.sizing;
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        format!(
            "igpu exec: device {} ({IGPU_ARCH}), {} layers bound, max {} rows, \
             pinned I/O {:.1} MiB, grouped scratch {:.1} MiB, manifest {:016x}",
            self.device,
            self.bound_layers(),
            s.max_rows,
            mib(s.io_bytes),
            mib(s.grouped_scratch_bytes()),
            self.manifest.digest(),
        )
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poison| {
            self.poison(IgpuFailure::DrainFailed);
            poison.into_inner()
        })
    }

    fn poison(&self, why: IgpuFailure) {
        let code = match why {
            IgpuFailure::SubmittedDrained => FAIL_SUBMITTED,
            IgpuFailure::DrainFailed => FAIL_DRAIN,
        };
        // A drain failure is never downgraded.
        self.failure.fetch_max(code, Ordering::AcqRel);
        if why == IgpuFailure::DrainFailed {
            self.unsafe_release.store(true, Ordering::Release);
        }
        self.poisoned.store(true, Ordering::Release);
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Drain BOTH devices (+streams): the model device through `model_gpu`,
    /// then the iGPU. `Err` poisons the engine and marks it
    /// [`unsafe_to_release`](Self::unsafe_to_release). On `Ok` nothing is in
    /// flight on either device, and the generation advances.
    pub fn quiesce(&self, model_gpu: &Gpu) -> Result<(), String> {
        let mut st = self.lock();
        let ok = self.drain_both(&mut st, Some(model_gpu));
        self.generation.fetch_add(1, Ordering::AcqRel);
        if ok {
            Ok(())
        } else {
            Err("igpu engine: failed to drain the model and iGPU devices".into())
        }
    }

    /// Drain the model device (when given) and the iGPU, each through its own
    /// `Gpu::bind_thread`. A failed drain poisons and marks unsafe-to-release.
    fn drain_both(&self, st: &mut State, model_gpu: Option<&Gpu>) -> bool {
        let mut ok = true;
        if let Some(m) = model_gpu {
            ok &= m.bind_thread().is_ok() && m.hip.device_synchronize().is_ok();
        }
        match st.gpu.as_ref() {
            Some(g) => ok &= g.bind_thread().is_ok() && g.hip.device_synchronize().is_ok(),
            None => ok = false,
        }
        if let Some(m) = model_gpu {
            ok &= m.bind_thread().is_ok();
        }
        if ok {
            st.inflight = false;
            // Everything is retired: no ticket survives, and every older
            // generation is stale from here on.
            st.ledger.clear_after_drain();
        } else {
            st.ledger.invalidate();
            self.poison(IgpuFailure::DrainFailed);
        }
        ok
    }

    /// Enable/disable parity capture. Allocates the separate reference
    /// buffers here, never in the hot path. `layers` filters by layer index.
    pub fn set_parity(&self, on: bool, layers: Option<&[usize]>) -> Result<(), String> {
        let mut st = self.lock();
        st.parity.records.clear();
        st.parity.layers = layers.map(<[usize]>::to_vec);
        if !on {
            st.parity.on = false;
            if let Some(r) = st.parity.refs.take() {
                if let Some(g) = st.gpu.as_mut() {
                    for t in [r.x, r.ids, r.weights, r.target] {
                        let _ = g.free_tensor(t);
                    }
                }
            }
            return Ok(());
        }
        if st.parity.refs.is_none() {
            let s = self.sizing;
            let g = st.gpu.as_mut().ok_or("igpu engine: device released")?;
            hips(g.bind_thread(), "bind")?;
            let refs = RefBufs {
                x: hips(g.alloc_tensor(&[s.max_rows, s.hidden], DType::F32), "parity x")?,
                ids: hips(g.alloc_tensor(&[s.max_slots], DType::F32), "parity ids")?,
                weights: hips(g.alloc_tensor(&[s.max_slots], DType::F32), "parity weights")?,
                target: hips(
                    g.alloc_tensor(&[s.max_rows, s.hidden], DType::F32),
                    "parity target",
                )?,
            };
            st.parity.refs = Some(refs);
        }
        st.parity.on = true;
        Ok(())
    }

    /// Drain and return all parity records (every layer), oldest first.
    pub fn take_parity_records(&self) -> Vec<IgpuParityRecord> {
        std::mem::take(&mut self.lock().parity.records)
    }

    /// Bind one layer: build its two `[experts]` pointer tables on the iGPU
    /// over the borrowed portable host weights and gate on the symmetric-grid
    /// header check. The first bind also loads every kernel. The device alias
    /// of the host weights is fetched with the iGPU bound through
    /// `Gpu::bind_thread`; a failed or null alias refuses the bind (no
    /// host-pointer fallback).
    pub fn bind_layer(
        self: &Arc<Self>,
        layer_idx: usize,
        w: IgpuLayerWeights,
    ) -> Result<Arc<IgpuLayerEngine>, String> {
        let s = self.sizing;
        let mut st = self.lock();
        if self.poisoned() {
            return Err("igpu engine: poisoned".into());
        }
        let e = s.experts;
        if w.owner_base.is_null() {
            return Err(format!("igpu layer {layer_idx}: null host owner base"));
        }
        let gu_end = checked_expert_extent(
            w.owner_len,
            w.gate_up_offset,
            w.gate_up_stride,
            s.gate_up_expert_bytes,
            e,
        )
        .map_err(|m| format!("igpu layer {layer_idx} gate/up: {m}"))?;
        let dn_end = checked_expert_extent(
            w.owner_len,
            w.down_offset,
            w.down_stride,
            s.down_expert_bytes,
            e,
        )
        .map_err(|m| format!("igpu layer {layer_idx} down: {m}"))?;
        // The two arrays of one layer must not overlap.
        if w.gate_up_offset < dn_end && w.down_offset < gu_end {
            return Err(format!("igpu layer {layer_idx}: gate/up and down arrays overlap"));
        }
        // Nothing below submits work until the first table upload; errors up
        // to the alias lookup are clean refusals.
        let owner = {
            let gpu = st.gpu.as_mut().ok_or("igpu engine: device released")?;
            hips(gpu.bind_thread(), "bind")?;
            // ONE alias of the owner BASE with the iGPU bound; offsets are added.
            let owner = hips(
                gpu.hip.host_get_device_pointer(w.owner_base, 0),
                "iGPU alias of the host owner base (owner must be hipHostMalloc MAPPED|PORTABLE)",
            )?;
            if owner.is_null() {
                return Err(format!("igpu layer {layer_idx}: null iGPU owner alias"));
            }
            owner as usize
        };
        let mut made: Vec<GpuTensor> = Vec::with_capacity(2);
        for (offset, stride) in [
            (w.gate_up_offset, w.gate_up_stride),
            (w.down_offset, w.down_stride),
        ] {
            match Self::upload_table(&mut st, owner, offset, stride, e) {
                Ok(t) => made.push(t),
                Err(m) => return Err(self.bind_runtime_failure(&mut st, made, w.lease, m)),
            }
        }
        match Self::sym_check(&mut st, &made[0], &made[1], &s) {
            Err(m) => return Err(self.bind_runtime_failure(&mut st, made, w.lease, m)),
            Ok((ok_gu, ok_dn)) if !(ok_gu && ok_dn) => {
                // An ordinary non-symmetric answer is a clean layer refusal:
                // the check completed synchronously, so releasing is safe and
                // the engine stays healthy.
                if let Some(g) = st.gpu.as_mut() {
                    for t in made {
                        let _ = g.free_tensor(t);
                    }
                }
                return Err(format!(
                    "igpu layer {layer_idx}: experts are not on the symmetric grid \
                     (gate/up {ok_gu}, down {ok_dn}); the iGPU path would change the model's arithmetic"
                ));
            }
            Ok(_) => {}
        }
        let down = made.pop().expect("two pointer tables");
        let gate_up = made.pop().expect("two pointer tables");
        let tables = Tables { gate_up, down };
        if !st.warmed {
            if let Err(msg) = self.warm(&mut st, &tables) {
                let Tables { gate_up, down } = tables;
                return Err(self.bind_runtime_failure(
                    &mut st,
                    vec![gate_up, down],
                    w.lease,
                    msg,
                ));
            }
            st.warmed = true;
        }
        st.bound_layers += 1;
        drop(st);
        Ok(Arc::new(IgpuLayerEngine {
            engine: Arc::clone(self),
            layer: layer_idx,
            tables: Mutex::new(Some(tables)),
            lease: w.lease,
        }))
    }

    /// Load every kernel the hot path launches by a 1-row pass over the zeroed
    /// I/O block.
    fn warm(&self, st: &mut State, tables: &Tables) -> Result<(), String> {
        let State { gpu, io, bufs, .. } = st;
        let (gpu, io, bufs) = match (gpu.as_mut(), io.as_ref(), bufs.as_ref()) {
            (Some(g), Some(i), Some(b)) => (g, i, b),
            _ => return Err("igpu engine: device released".into()),
        };
        run_stages(
            gpu,
            &self.sizing,
            [&io.x, &io.ids, &io.weights, &io.target],
            bufs,
            tables,
            1,
        )
        .map_err(|e| format!("igpu engine warm-up: {e}"))?;
        hips(gpu.hip.device_synchronize(), "warm-up sync")
    }

    /// Upload one `[experts]` pointer table (expert `x` at `owner + offset +
    /// x * stride`, iGPU addresses) through the iGPU `Gpu`.
    fn upload_table(
        st: &mut State,
        owner: usize,
        offset: usize,
        stride: usize,
        experts: usize,
    ) -> Result<GpuTensor, String> {
        let gpu = st.gpu.as_mut().ok_or("igpu engine: device released")?;
        let mut bytes = Vec::with_capacity(experts * 8);
        for x in 0..experts {
            let p = x
                .checked_mul(stride)
                .and_then(|v| v.checked_add(offset))
                .and_then(|v| v.checked_add(owner))
                .ok_or("igpu engine: weight alias overflows")?;
            bytes.extend_from_slice(&(p as u64).to_le_bytes());
        }
        hips(gpu.bind_thread(), "bind")?;
        hips(gpu.upload_raw(&bytes, &[experts * 8]), "pointer table")
    }

    /// The symmetric-grid header gate over both tables. Synchronous: the
    /// checks have completed (device sync + readback) when this returns
    /// `Ok`, so a `false` leaves nothing in flight.
    fn sym_check(
        st: &mut State,
        gate_up: &GpuTensor,
        down: &GpuTensor,
        s: &IgpuSizing,
    ) -> Result<(bool, bool), String> {
        let gpu = st.gpu.as_mut().ok_or("igpu engine: device released")?;
        let ok_gu = hips(
            gpu.qwen4_moe_sym_check(
                gate_up,
                2 * s.intermediate,
                s.hidden,
                s.experts,
                QT44_GROUP_BYTES,
            ),
            "sym check gate/up",
        )?;
        let ok_dn = hips(
            gpu.qwen4_moe_sym_check(down, s.hidden, s.intermediate, s.experts, QT53_GROUP_BYTES),
            "sym check down",
        )?;
        Ok((ok_gu, ok_dn))
    }

    /// A runtime error after table upload / check / warm-up may have been
    /// submitted: drain the iGPU (the model device takes no part in a bind)
    /// BEFORE releasing anything. On a good drain the tables are freed, the
    /// lease dropped and the engine poisoned
    /// ([`IgpuFailure::SubmittedDrained`]). If the drain fails, the tables
    /// AND the lease are retained (leaked) and the engine is left
    /// [`unsafe_to_release`](Self::unsafe_to_release) even though no layer
    /// bound: the caller must quarantine the model/bundle and keep the host
    /// owner alive. Leaves the thread bound to the iGPU; every `Gpu` method
    /// rebinds its own device.
    fn bind_runtime_failure(
        &self,
        st: &mut State,
        tensors: Vec<GpuTensor>,
        lease: Option<Arc<dyn Any + Send + Sync>>,
        msg: String,
    ) -> String {
        if self.drain_both(st, None) {
            match st.gpu.as_mut() {
                Some(g) => {
                    for t in tensors {
                        let _ = g.free_tensor(t);
                    }
                }
                None => std::mem::forget(tensors),
            }
            self.poison(IgpuFailure::SubmittedDrained);
            drop(lease);
            format!("{msg} (iGPU drained; engine poisoned)")
        } else {
            std::mem::forget(tensors);
            std::mem::forget(lease);
            format!(
                "{msg} (drain failed: engine is unsafe to release; tables, lease and second \
                 device retained, quarantine the model)"
            )
        }
    }
}

/// One layer's expert weights as subviews of ONE immutable host owner.
///
/// `owner_base` is the host pointer exactly as `hipHostMalloc(MAPPED |
/// PORTABLE)` returned it (the BASE of the pooled allocation, never an
/// interior pointer); the layer's QT44 gate/up and QT53 down expert arrays
/// live at byte `*_offset` from it, expert `e` at `offset + e * stride`. The
/// engine maps the base once with the iGPU bound and adds the checked offsets;
/// it never calls `hipHostGetDevicePointer` on an interior pointer.
#[derive(Clone)]
pub struct IgpuLayerWeights {
    pub owner_base: *mut c_void,
    /// Bytes of the whole owner allocation (bounds every extent below).
    pub owner_len: usize,
    pub gate_up_offset: usize,
    pub gate_up_stride: usize,
    pub down_offset: usize,
    pub down_stride: usize,
    /// Keepalive of the immutable host owner. The layer engine holds it for
    /// its whole life, and leaks it (never releases it) when a drain failed.
    pub lease: Option<Arc<dyn Any + Send + Sync>>,
}

/// Byte alignment the grouped GEMMs need of every expert start (b128 loads).
const EXPERT_ALIGN: usize = 16;

/// Checked placement of one expert array inside the owner: returns the end
/// offset. Pure; refuses misaligned starts/strides, strides below the exact
/// expert bytes, and any extent past `owner_len`.
pub fn checked_expert_extent(
    owner_len: usize,
    offset: usize,
    stride: usize,
    expert_bytes: usize,
    experts: usize,
) -> Result<usize, String> {
    if experts == 0 || expert_bytes == 0 {
        return Err("igpu weights: empty expert array".into());
    }
    if offset % EXPERT_ALIGN != 0 || stride % EXPERT_ALIGN != 0 {
        return Err(format!(
            "igpu weights: offset {offset} / stride {stride} must be {EXPERT_ALIGN}-byte aligned"
        ));
    }
    if stride < expert_bytes {
        return Err(format!(
            "igpu weights: stride {stride} < expert bytes {expert_bytes}"
        ));
    }
    let end = (experts - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(expert_bytes))
        .and_then(|v| v.checked_add(offset))
        .ok_or("igpu weights: extent overflows")?;
    if end > owner_len {
        return Err(format!(
            "igpu weights: extent ends at {end}, past the {owner_len}-byte owner"
        ));
    }
    Ok(end)
}

struct Tables {
    gate_up: GpuTensor,
    down: GpuTensor,
}

/// The production symmetric-IU4 MoE pipeline over `[x, ids, weights, target]`,
/// forced A4 at every width, combining from +0.
fn run_stages(
    gpu: &mut Gpu,
    sz: &IgpuSizing,
    [x, ids, weights, target]: [&GpuTensor; 4],
    bufs: &Bufs,
    tables: &Tables,
    rows: usize,
) -> Result<(), DispatchError> {
    let live = sz.live(rows).map_err(derr)?;
    let (slots, grouped) = (live.slots, live.grouped_rows);
    hip(gpu.bind_thread())?;
    hip(gpu.moe_scatter_stable_top10(
        ids,
        &bufs.counts,
        &bufs.offsets,
        &bufs.sorted,
        &bufs.tiles,
        &bufs.inverse,
        slots,
        sz.experts,
        grouped,
    ))?;
    let xq = hip(gpu.qwen4_moe_rotate256_i4(x, sz.hidden, rows))?;
    hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
        &tables.gate_up,
        &bufs.tiles,
        &bufs.sorted,
        &xq,
        &bufs.y_gate_up,
        2 * sz.intermediate,
        sz.hidden,
        sz.k_top,
        grouped,
        rows,
        true,
    ))?;
    let xq = hip(gpu.qwen4_moe_rotate128_i4(
        &bufs.y_gate_up,
        &bufs.sorted,
        sz.intermediate,
        grouped,
        slots,
    ))?;
    hip(gpu.gemm_qwen4_moe_down_iu4_sym(
        &tables.down,
        &bufs.tiles,
        &bufs.sorted,
        &xq,
        &bufs.y_down,
        sz.hidden,
        sz.intermediate,
        1,
        grouped,
        slots,
        true,
    ))?;
    hip(gpu.moe_down_combine_grouped_top10_bf16in(
        &bufs.y_down,
        &bufs.inverse,
        ids,
        weights,
        target,
        &bufs.order,
        sz.hidden,
        grouped,
        rows,
        true,
        sz.experts,
    ))
}

impl Drop for IgpuEngine {
    fn drop(&mut self) {
        let leak_flag = self.unsafe_release.load(Ordering::Acquire);
        let st = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        let drained = !leak_flag
            && !st.inflight
            && match st.gpu.as_ref() {
                Some(g) => g.bind_thread().is_ok() && g.hip.device_synchronize().is_ok(),
                None => false,
            };
        if !drained {
            // A device may still be reading or writing: leak everything.
            self.unsafe_release.store(true, Ordering::Release);
            std::mem::forget(st.gpu.take());
            std::mem::forget(st.io.take());
            std::mem::forget(st.bufs.take());
            std::mem::forget(st.parity.refs.take());
            return;
        }
        if let Some(mut gpu) = st.gpu.take() {
            if let Some(r) = st.parity.refs.take() {
                for t in [r.x, r.ids, r.weights, r.target] {
                    let _ = gpu.free_tensor(t);
                }
            }
            if let Some(b) = st.bufs.take() {
                for t in [
                    b.counts, b.offsets, b.sorted, b.tiles, b.inverse, b.y_gate_up, b.y_down,
                    b.order,
                ] {
                    let _ = gpu.free_tensor(t);
                }
            }
            if let Some(io) = st.io.take() {
                if gpu.bind_thread().is_ok() {
                    let _ = gpu.hip.host_free(io.host);
                }
            }
            drop(gpu);
        }
    }
}

// ───────────────────────────── per-layer engine ─────────────────────────────

/// One admitted layer's handle onto the shared engine.
pub struct IgpuLayerEngine {
    engine: Arc<IgpuEngine>,
    layer: usize,
    tables: Mutex<Option<Tables>>,
    lease: Option<Arc<dyn Any + Send + Sync>>,
}

impl IgpuLayerEngine {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn engine(&self) -> &Arc<IgpuEngine> {
        &self.engine
    }

    pub fn poisoned(&self) -> bool {
        self.engine.poisoned()
    }

    pub fn unsafe_to_release(&self) -> bool {
        self.engine.unsafe_to_release()
    }

    pub fn manifest(&self) -> &IgpuManifest {
        self.engine.manifest()
    }

    pub fn quiesce(&self, model_gpu: &Gpu) -> Result<(), String> {
        self.engine.quiesce(model_gpu)
    }

    pub fn set_parity(&self, on: bool, layers: Option<&[usize]>) -> Result<(), String> {
        self.engine.set_parity(on, layers)
    }

    pub fn take_parity_records(&self) -> Vec<IgpuParityRecord> {
        self.engine.take_parity_records()
    }

    /// Parity-only: run `case` (any `1..=max_rows`, so tails below the
    /// production `>4`-row admission are measurable) through the REAL
    /// transport path and the standalone iGPU reference. The case bytes are
    /// staged into model-device buffers allocated for this call (parity mode
    /// only; freed before return), `target` is pre-filled with a nonzero
    /// poison so the +0 combine is proven, and the engine's own D2H -> pinned
    /// bank -> iGPU kernels -> H2D runs unchanged. Requires
    /// [`set_parity`](Self::set_parity)`(true, ..)`; production admission is
    /// untouched. Errors drain both devices like `execute`.
    pub fn parity_execute(
        &self,
        model: &mut Gpu,
        case: &IgpuParityCase,
    ) -> Result<IgpuParityOutcome, String> {
        let eng = &*self.engine;
        let live = eng.sizing.live(case.rows)?;
        for (name, have, need) in [
            ("x_norm", case.x_norm.len(), live.x_bytes),
            ("topk_ids", case.topk_ids.len(), live.ids_bytes),
            ("topk_weights", case.topk_weights.len(), live.weights_bytes),
        ] {
            if have != need {
                return Err(format!(
                    "igpu parity case: {name} holds {have} bytes, {} rows need {need}",
                    case.rows
                ));
            }
        }
        {
            let st = eng.lock();
            if !(st.parity.on && st.parity.refs.is_some()) {
                return Err("igpu parity case refused: parity mode is off".into());
            }
        }
        let hidden = eng.sizing.hidden;
        hips(model.bind_thread(), "bind model")?;
        let mut owned: Vec<GpuTensor> = Vec::with_capacity(4);
        let staged = (|| -> Result<(), String> {
            for (shape, bytes) in [
                (vec![case.rows, hidden], &case.x_norm),
                (vec![live.slots], &case.topk_ids),
                (vec![live.slots], &case.topk_weights),
            ] {
                let t = hips(model.alloc_tensor(&shape, DType::F32), "stage alloc")?;
                let up = hips(model.hip.memcpy_htod(&t.buf, bytes), "stage upload");
                owned.push(t);
                up?;
            }
            let t = hips(
                model.alloc_tensor(&[case.rows, hidden], DType::F32),
                "stage target",
            )?;
            let up = hips(
                model.hip.memcpy_htod(&t.buf, &vec![0xA5u8; live.target_bytes]),
                "poison target",
            );
            owned.push(t);
            up
        })();
        let run = staged.and_then(|()| {
            self.execute_impl(
                model, &owned[0], &owned[1], &owned[2], &owned[3], case.rows, true, true,
            )
            .map_err(|e| e.to_string())
        });
        let outcome = run.and_then(|record| {
            let record = record.ok_or("igpu parity: no record captured")?;
            let mut engine_target = hips(model.download_raw_bytes(&owned[3]), "download target")?;
            engine_target.truncate(live.target_bytes);
            Ok(IgpuParityOutcome {
                rows: case.rows,
                engine_target,
                igpu_standalone_target: record.igpu_standalone_target,
            })
        });
        if eng.unsafe_to_release() {
            // A device may still touch these buffers: leak them.
            std::mem::forget(owned);
        } else {
            for t in owned {
                let _ = model.free_tensor(t);
            }
        }
        outcome
    }

    /// Run this layer's routed experts on the iGPU and leave the canonical
    /// combine in `target` (`[rows, hidden]` F32). `initial_zero` MUST be
    /// `true`: the result is routed-only from +0 and `target` is never read.
    ///
    /// Synchronous: on `Ok` both devices have retired every access to both
    /// transfer banks. An `Err` whose message starts with
    /// `igpu rejected before enqueue` changed nothing; any other `Err` drained
    /// both devices (see [`IgpuFailure`]) and poisoned the engine for good.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        model_gpu: &mut Gpu,
        x_norm: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        target: &GpuTensor,
        rows: usize,
        initial_zero: bool,
    ) -> Result<(), DispatchError> {
        self.execute_impl(
            model_gpu, x_norm, ids, weights, target, rows, initial_zero, false,
        )
        .map(|_| ())
    }

    /// [`execute`](Self::execute) body. `force_parity` captures this call's
    /// record and returns it instead of appending it to the shared list.
    #[allow(clippy::too_many_arguments)]
    fn execute_impl(
        &self,
        model_gpu: &mut Gpu,
        x_norm: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        target: &GpuTensor,
        rows: usize,
        initial_zero: bool,
        force_parity: bool,
    ) -> Result<Option<IgpuParityRecord>, DispatchError> {
        let eng = &*self.engine;
        if eng.poisoned() {
            return Err(rejected("the engine is poisoned by an earlier failure"));
        }
        if let Some(why) = capture_refusal(
            model_gpu.graphs.capture_mode,
            model_gpu.replay.is_recording(),
            model_gpu.graphs.verify.capturing.is_some(),
            model_gpu.graphs.replay.capturing.is_some(),
        ) {
            return Err(rejected(why));
        }
        if !initial_zero {
            return Err(rejected(
                "the routed-only ABI combines from +0 (initial_zero must be true)",
            ));
        }
        let live = eng.sizing.live(rows).map_err(rejected)?;
        if x_norm.dtype != DType::F32 || target.dtype != DType::F32 {
            return Err(rejected("x_norm and target must be F32"));
        }
        for (name, t, need) in [
            ("x_norm", x_norm, live.x_bytes),
            ("ids", ids, live.ids_bytes),
            ("weights", weights, live.weights_bytes),
            ("target", target, live.target_bytes),
        ] {
            if t.buf.size() < need {
                return Err(rejected(format!(
                    "{name} holds {} bytes, {need} needed",
                    t.buf.size()
                )));
            }
        }
        let tables_guard = self.tables.lock().unwrap_or_else(|p| p.into_inner());
        let Some(tables) = tables_guard.as_ref() else {
            return Err(rejected("layer tables released"));
        };
        let mut st = eng.lock();
        if eng.poisoned() {
            return Err(rejected("the engine is poisoned by an earlier failure"));
        }
        if st.gpu.is_none() || st.io.is_none() || st.bufs.is_none() {
            return Err(rejected("the device was released"));
        }
        match st.model_device {
            Some(d) if d != model_gpu.device_id => {
                return Err(rejected("the engine is bound to a different model device"));
            }
            None => st.model_device = Some(model_gpu.device_id),
            _ => {}
        }
        // Claim a fresh generation before anything is enqueued; a bank that is
        // still owned by an earlier ticket refuses the call.
        let ticket = match st.ledger.claim(self.layer, live.rows) {
            Ok(t) => t,
            Err(e) => return Err(rejected(e)),
        };
        // From here on work may be submitted.
        st.inflight = true;
        let result = self.submit(
            &mut st, tables, model_gpu, x_norm, ids, weights, target, live, force_parity, ticket,
        );
        match result {
            Ok(record) => {
                st.inflight = false;
                Ok(record)
            }
            Err(error) => {
                // Drain BOTH devices before anything is released; poison for good.
                let drained = eng.drain_both(&mut st, Some(model_gpu));
                if drained {
                    eng.poison(IgpuFailure::SubmittedDrained);
                }
                let note = if drained {
                    "drained both devices; engine poisoned"
                } else {
                    "drain failed; engine poisoned and will leak its buffers"
                };
                Err(derr(format!("igpu submit failure ({note}): {error}")))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn submit(
        &self,
        st: &mut State,
        tables: &Tables,
        model_gpu: &mut Gpu,
        x_norm: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        target: &GpuTensor,
        live: IgpuLive,
        force: bool,
        mut ticket: TransferTicket,
    ) -> Result<Option<IgpuParityRecord>, DispatchError> {
        let sz = self.engine.sizing;
        let wants_parity = force || st.parity.wants(self.layer);
        let State {
            gpu,
            io,
            bufs,
            parity,
            ledger,
            ..
        } = st;
        let (igpu, io, bufs) = match (gpu.as_mut(), io.as_ref(), bufs.as_ref()) {
            (Some(g), Some(i), Some(b)) => (g, i, b),
            _ => return Err(derr("igpu engine: device released")),
        };
        let host = io.host.cast::<u8>();
        let span = |off: usize, len: usize| host_span(host, off, len);

        // ── model device: D2H into the input bank ──
        hip(model_gpu.bind_thread())?;
        let stream = model_gpu.active_stream.as_ref();
        hip(model_gpu
            .hip
            .memcpy_dtoh_async(span(sz.x_off, live.x_bytes), &x_norm.buf, stream))?;
        hip(model_gpu
            .hip
            .memcpy_dtoh_async(span(sz.ids_off, live.ids_bytes), &ids.buf, stream))?;
        hip(model_gpu.hip.memcpy_dtoh_async(
            span(sz.weights_off, live.weights_bytes),
            &weights.buf,
            stream,
        ))?;
        hip(match stream {
            Some(s) => model_gpu.hip.stream_synchronize(s),
            None => model_gpu.hip.stream_synchronize_default(),
        })?;
        // Export complete: the input bank is immutable until Halo's last read.
        ticket = ledger
            .advance(&ticket, TicketPhase::HaloPending)
            .map_err(derr)?;

        // ── parity: snapshot the inputs the engine is about to consume ──
        let snapshot = wants_parity.then(|| {
            (
                span(sz.x_off, live.x_bytes).to_vec(),
                span(sz.ids_off, live.ids_bytes).to_vec(),
                span(sz.weights_off, live.weights_bytes).to_vec(),
            )
        });

        // ── iGPU ──
        run_stages(
            igpu,
            &sz,
            [&io.x, &io.ids, &io.weights, &io.target],
            bufs,
            tables,
            live.rows,
        )?;
        hip(igpu.hip.device_synchronize())?;

        let mut forced: Option<IgpuParityRecord> = None;
        if let Some((x, i, w)) = snapshot {
            let engine_target = span(sz.target_off, live.target_bytes).to_vec();
            let standalone = {
                let refs = parity
                    .refs
                    .as_ref()
                    .ok_or_else(|| derr("igpu parity buffers missing"))?;
                hip(igpu.bind_thread())?;
                hip(igpu.hip.memcpy_htod(&refs.x.buf, &x))?;
                hip(igpu.hip.memcpy_htod(&refs.ids.buf, &i))?;
                hip(igpu.hip.memcpy_htod(&refs.weights.buf, &w))?;
                run_stages(
                    igpu,
                    &sz,
                    [&refs.x, &refs.ids, &refs.weights, &refs.target],
                    bufs,
                    tables,
                    live.rows,
                )?;
                hip(igpu.hip.device_synchronize())?;
                let mut bytes = hip(igpu.download_raw_bytes(&refs.target))?;
                bytes.truncate(live.target_bytes);
                bytes
            };
            let record = IgpuParityRecord {
                layer: self.layer,
                rows: live.rows,
                hidden: sz.hidden,
                k_top: sz.k_top,
                initial_zero: true,
                x_norm: x,
                topk_ids: i,
                topk_weights: w,
                target_in: Vec::new(),
                engine_target,
                igpu_standalone_target: standalone,
            };
            if force {
                forced = Some(record);
            } else {
                parity.records.push(record);
            }
        }

        // ── model device: H2D return from the output bank, then retire it ──
        // A result tagged with a stale generation (poison/quiesce since the
        // claim) is never published: validate right before the copy.
        ledger
            .advance(&ticket, TicketPhase::ImportPending)
            .and_then(|t| ledger.validate(&t, TicketPhase::ImportPending).map(|()| t))
            .map(|t| ticket = t)
            .map_err(derr)?;
        hip(model_gpu.bind_thread())?;
        let src: &[u8] = span(sz.target_off, live.target_bytes);
        let stream = model_gpu.active_stream.as_ref();
        hip(match stream {
            Some(s) => model_gpu.hip.memcpy_htod_async(&target.buf, src, s),
            None => model_gpu.hip.memcpy_htod_async_default(&target.buf, src),
        })?;
        hip(match stream {
            Some(s) => model_gpu.hip.stream_synchronize(s),
            None => model_gpu.hip.stream_synchronize_default(),
        })?;
        // Only after the model sync have both readers of both banks retired.
        ledger.retire(&ticket).map_err(derr)?;
        Ok(forced)
    }
}

impl Drop for IgpuLayerEngine {
    fn drop(&mut self) {
        let tables = self
            .tables
            .get_mut()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        let lease = self.lease.take();
        let eng = &*self.engine;
        let mut st = eng.lock();
        // Drain BOTH devices before the tables (which the iGPU may still be
        // reading) or the weight lease are released; a failed or impossible
        // drain leaks them.
        let safe = !eng.unsafe_to_release() && !st.inflight && eng.drain_both(&mut st, None);
        if !safe {
            std::mem::forget(tables);
            std::mem::forget(lease);
            return;
        }
        st.bound_layers = st.bound_layers.saturating_sub(1);
        if let Some(tables) = tables {
            match st.gpu.as_mut() {
                Some(g) => {
                    let _ = g.free_tensor(tables.gate_up);
                    let _ = g.free_tensor(tables.down);
                }
                None => std::mem::forget(tables),
            }
        }
        drop(st);
        drop(lease);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(rows: usize) -> IgpuEngineSpec {
        IgpuEngineSpec {
            device: 1,
            max_rows: rows,
            hidden: 2560,
            moe_intermediate: 640,
            num_experts: 512,
            top_k: 10,
            layers: 40,
        }
    }

    fn arm() -> IgpuArmInputs<'static> {
        IgpuArmInputs {
            requested: true,
            primary_arch: "gfx1100",
            second_arch: "gfx1151",
            primary_device: 0,
            second_device: 1,
            host_whole_layers_only: true,
            ..Default::default()
        }
    }

    fn manifest() -> IgpuManifest {
        IgpuManifest {
            version: IGPU_MANIFEST_VERSION,
            arch: "gfx1151".into(),
            hidden: 2560,
            intermediate: 640,
            experts: 512,
            k_top: 10,
            a4_candidates: IU4_A4_CANDIDATES,
            gate_up_kernel: "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4".into(),
            down_kernel: "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4".into(),
            down_row_repeat: 1,
        }
    }

    #[test]
    fn arms_when_every_fact_holds() {
        assert_eq!(igpu_arming(&arm()), IgpuArming::Armed);
        assert_eq!(igpu_refusal(&arm()), Ok(()));
    }

    #[test]
    fn not_requested_is_a_noop_not_a_rejection() {
        // Even with every other fact wrong, an unrequested mode refuses nothing.
        let off = IgpuArmInputs {
            requested: false,
            second_arch: "",
            ep: true,
            pp: true,
            ..arm()
        };
        assert_eq!(igpu_arming(&off), IgpuArming::NotRequested);
        assert_eq!(igpu_refusal(&off), Ok(()));
        // The same facts, requested, are a refusal.
        assert!(matches!(
            igpu_arming(&IgpuArmInputs { requested: true, ..off }),
            IgpuArming::Refused(_)
        ));
    }

    #[test]
    fn any_primary_arch_arms() {
        for arch in ["gfx1100", "gfx1201", "gfx1030"] {
            assert_eq!(
                igpu_arming(&IgpuArmInputs { primary_arch: arch, ..arm() }),
                IgpuArming::Armed
            );
        }
    }


    #[test]
    fn refuses_each_condition_with_one_line() {
        let cases: Vec<(IgpuArmInputs<'static>, &str)> = vec![
            (IgpuArmInputs { second_arch: "", ..arm() }, "no second"),
            (IgpuArmInputs { second_arch: "gfx1100", ..arm() }, "not gfx1151"),
            (IgpuArmInputs { second_device: 0, ..arm() }, "primary device"),
            (IgpuArmInputs { emu: true, ..arm() }, "emu"),
            (IgpuArmInputs { ep: true, ..arm() }, "expert parallelism"),
            (IgpuArmInputs { pp: true, ..arm() }, "pipeline parallelism"),
            (IgpuArmInputs { slab: true, ..arm() }, "slab"),
            (IgpuArmInputs { tier: true, ..arm() }, "tiered"),
            (IgpuArmInputs { awq: true, ..arm() }, "AWQ"),
            (IgpuArmInputs { hotset: true, ..arm() }, "hot-set"),
            (IgpuArmInputs { nonuniform_experts: true, ..arm() }, "uniform"),
            (IgpuArmInputs { host_whole_layers_only: false, ..arm() }, "whole-layer"),
        ];
        for (inputs, needle) in cases {
            let msg = igpu_refusal(&inputs).expect_err(needle);
            assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
            assert!(!msg.contains('\n'), "refusal must be one line: {msg:?}");
        }
    }

    #[test]
    fn device_index_parse() {
        assert_eq!(parse_igpu_device(None), None);
        assert_eq!(parse_igpu_device(Some(" 1 ")), Some(1));
        assert_eq!(parse_igpu_device(Some("0")), Some(0));
        assert_eq!(parse_igpu_device(Some("-1")), None);
        assert_eq!(parse_igpu_device(Some("x")), None);
        assert_eq!(parse_igpu_device(Some("")), None);
    }

    #[test]
    fn capture_and_replay_are_refused() {
        assert!(capture_refusal(false, false, false, false).is_none());
        assert!(capture_refusal(true, false, false, false).is_some());
        assert!(capture_refusal(false, true, false, false).is_some());
        assert!(capture_refusal(false, false, true, false).is_some());
        assert!(capture_refusal(false, false, false, true).is_some());
    }

    #[test]
    fn flash_next_sizing_is_exact() {
        let s = IgpuSizing::new(&spec(4096)).unwrap();
        assert_eq!(s.max_slots, 40_960);
        // slots + 15 pad rows for each of the 512 active experts, 16-aligned.
        assert_eq!(s.max_grouped_rows, 48_640);
        assert_eq!(s.x_bytes, 4096 * 2560 * 4);
        assert_eq!(s.target_bytes, s.x_bytes);
        assert_eq!(s.ids_bytes, 163_840);
        assert_eq!(s.weights_bytes, 163_840);
        assert_eq!(s.y_gate_up_bytes, 48_640 * 640 * 2);
        assert_eq!(s.y_down_bytes, 48_640 * 2560 * 2);
        assert_eq!(s.order_bytes, 40_960 * 8);
        assert_eq!(s.gate_up_expert_bytes, 1_740_800);
        assert_eq!(s.down_expert_bytes, 870_400);
        for off in [s.ids_off, s.weights_off, s.target_off] {
            assert_eq!(off % IO_ALIGN, 0);
        }
        // Input and output banks are disjoint and ordered inside the block.
        assert!(s.x_off + s.x_bytes <= s.ids_off);
        assert!(s.ids_off + s.ids_bytes <= s.weights_off);
        assert!(s.weights_off + s.weights_bytes <= s.target_off);
        assert_eq!(s.io_bytes, s.target_off + s.target_bytes);
    }

    #[test]
    fn grouped_bound_covers_every_live_row_count() {
        let s = IgpuSizing::new(&spec(4096)).unwrap();
        for rows in [1usize, 5, 512, 513, 2048, 4096] {
            let l = s.live(rows).unwrap();
            assert_eq!(l.slots, rows * 10);
            assert!(l.grouped_rows >= l.slots && l.grouped_rows % 16 == 0);
            assert!(l.grouped_rows <= s.max_grouped_rows);
            assert!(l.x_bytes <= s.x_bytes && l.ids_bytes <= s.ids_bytes);
        }
        assert!(s.live(0).is_err());
        assert!(s.live(4097).is_err());
    }

    #[test]
    fn sizing_refuses_abi_violations() {
        let bad = |f: fn(&mut IgpuEngineSpec)| {
            let mut sp = spec(512);
            f(&mut sp);
            assert!(IgpuSizing::new(&sp).is_err());
        };
        bad(|s| s.max_rows = 0);
        bad(|s| s.top_k = 8);
        bad(|s| s.hidden = 2560 + 128);
        bad(|s| s.moe_intermediate = 640 + 64);
        bad(|s| s.num_experts = 1025);
        bad(|s| s.device = -1);
        bad(|s| s.max_rows = usize::MAX);
        // 100k rows overflow the 32-bit down Y byte offset.
        bad(|s| s.max_rows = 100_000);
        bad(|s| s.max_rows = (PM_MAX_SLOTS / 10) + 1);
        assert!(IgpuSizing::new(&spec(8192)).is_ok());
    }

    #[test]
    fn down_extent_limit_is_the_u32_byte_offset() {
        // Largest legal grouped extent for hidden=2560 is u32::MAX / 5120 rows.
        let max_grouped = DOWN_EXTENT_LIMIT_BYTES / (2560 * 2);
        let mut ok = 0;
        for rows in [8192usize, 16_000, 20_000, 40_000, 80_000, 90_000] {
            if let Ok(s) = IgpuSizing::new(&spec(rows)) {
                assert!(s.max_grouped_rows <= max_grouped);
                ok += 1;
            }
        }
        assert!(ok >= 2);
    }

    fn drive(l: &mut TicketLedger, layer: usize, rows: usize) -> TransferTicket {
        let t = l.claim(layer, rows).unwrap();
        let t = l.advance(&t, TicketPhase::HaloPending).unwrap();
        let t = l.advance(&t, TicketPhase::ImportPending).unwrap();
        l.validate(&t, TicketPhase::ImportPending).unwrap();
        l.retire(&t).unwrap();
        t
    }

    #[test]
    fn ticket_generations_are_fresh_and_monotonic() {
        let mut l = TicketLedger::default();
        let a = drive(&mut l, 3, 512);
        let b = drive(&mut l, 4, 1024);
        let c = drive(&mut l, 3, 513);
        assert!(a.generation < b.generation && b.generation < c.generation);
        assert_eq!((b.layer, b.rows), (4, 1024));
        assert!(l.active.is_none(), "every ticket is retired");
    }

    #[test]
    fn ticket_transitions_are_checked_in_order() {
        let mut l = TicketLedger::default();
        let t = l.claim(0, 16).unwrap();
        assert_eq!(t.phase, TicketPhase::ExportPending);
        // Skipping, repeating or going backwards is refused.
        assert!(l.advance(&t, TicketPhase::ImportPending).is_err());
        assert!(l.advance(&t, TicketPhase::Retired).is_err());
        assert!(l.advance(&t, TicketPhase::ExportPending).is_err());
        assert!(l.retire(&t).is_err(), "only ImportPending retires");
        let h = l.advance(&t, TicketPhase::HaloPending).unwrap();
        // The pre-advance copy is no longer the live ticket.
        assert!(l.advance(&t, TicketPhase::HaloPending).is_err());
        assert!(l.validate(&t, TicketPhase::ExportPending).is_err());
        assert!(l.validate(&h, TicketPhase::ImportPending).is_err(), "wrong phase");
        assert!(l.retire(&h).is_err());
        let i = l.advance(&h, TicketPhase::ImportPending).unwrap();
        assert!(l.advance(&i, TicketPhase::HaloPending).is_err());
        l.retire(&i).unwrap();
        // A retired ticket cannot be reused or advanced.
        assert!(l.advance(&i, TicketPhase::Retired).is_err());
        assert!(l.validate(&i, TicketPhase::ImportPending).is_err());
        assert!(l.retire(&i).is_err());
    }

    #[test]
    fn one_ticket_owns_the_banks_at_a_time() {
        let mut l = TicketLedger::default();
        let t = l.claim(1, 8).unwrap();
        let err = l.claim(2, 8).unwrap_err();
        assert!(err.contains("still owned"), "{err}");
        let h = l.advance(&t, TicketPhase::HaloPending).unwrap();
        assert!(l.claim(2, 8).is_err());
        let i = l.advance(&h, TicketPhase::ImportPending).unwrap();
        l.retire(&i).unwrap();
        assert!(l.claim(2, 8).is_ok());
        assert!(l.claim(0, 0).is_err() || l.active.is_some());
    }

    #[test]
    fn poison_or_quiesce_makes_older_generations_stale() {
        let mut l = TicketLedger::default();
        let t = l.claim(5, 64).unwrap();
        let h = l.advance(&t, TicketPhase::HaloPending).unwrap();
        let i = l.advance(&h, TicketPhase::ImportPending).unwrap();
        // Poison between the iGPU run and the return: the result may not publish.
        l.invalidate();
        let err = l.validate(&i, TicketPhase::ImportPending).unwrap_err();
        assert!(err.contains("stale"), "{err}");
        assert!(l.retire(&i).is_err());
        assert!(l.advance(&h, TicketPhase::ImportPending).is_err());
        // A failed drain leaves the banks owned: no claim until a good drain.
        assert!(l.claim(5, 64).is_err());
        // A good drain frees them; the next claim is fresh, the old one stays stale.
        l.clear_after_drain();
        let n = l.claim(5, 64).unwrap();
        assert!(n.generation > i.generation);
        assert!(l.validate(&i, TicketPhase::ImportPending).is_err());
        assert!(l.validate(&n, TicketPhase::ExportPending).is_ok());
    }

    #[test]
    fn zero_row_ticket_is_refused_and_leaves_the_banks_free() {
        let mut l = TicketLedger::default();
        assert!(l.claim(0, 0).is_err());
        assert!(l.active.is_none());
        assert!(l.claim(0, 1).is_ok());
    }

    #[test]
    fn expert_extent_is_checked_inside_one_owner() {
        let s = IgpuSizing::new(&spec(512)).unwrap();
        let (gu, dn) = (s.gate_up_expert_bytes, s.down_expert_bytes);
        let gu_total = 512 * gu;
        let dn_total = 512 * dn;
        // Exact fit, strides equal to the expert bytes (both are 16-aligned).
        assert_eq!(checked_expert_extent(gu_total, 0, gu, gu, 512), Ok(gu_total));
        // A pooled owner: down array right after gate/up at an aligned offset.
        let owner = gu_total + dn_total;
        assert_eq!(checked_expert_extent(owner, gu_total, dn, dn, 512), Ok(owner));
        // One byte short, a misaligned offset or stride, a short stride and an
        // overflowing extent are all refused.
        assert!(checked_expert_extent(owner - 1, gu_total, dn, dn, 512).is_err());
        assert!(checked_expert_extent(owner + 64, 8, dn, dn, 512).is_err());
        assert!(checked_expert_extent(owner * 2, 0, dn + 8, dn, 512).is_err());
        assert!(checked_expert_extent(owner, 0, dn - 16, dn, 512).is_err());
        assert!(checked_expert_extent(usize::MAX, 16, usize::MAX - 15, dn, 512).is_err());
        assert!(checked_expert_extent(owner, 0, dn, dn, 0).is_err());
    }

    #[test]
    fn grouped_rows_matches_scratch_bound() {
        assert_eq!(igpu_grouped_rows(10, 512), Some(160));
        assert_eq!(igpu_grouped_rows(5120, 512), Some(5120 + 512 * 15));
        assert_eq!(igpu_grouped_rows(usize::MAX, 512), None);
    }

    #[test]
    fn manifest_digest_is_stable_and_recipe_sensitive() {
        let base = manifest();
        assert_eq!(base.digest(), manifest().digest());
        let text = base.canonical();
        // The forced-A4-at-every-width and tail contract is part of the recipe.
        assert!(text.contains("A4 forced at every width"));
        assert!(text.contains("no 512-row threshold"));
        assert!(text.contains(&format!("dispatch_min_rows={IGPU_MIN_DISPATCH_ROWS}")));
        let changed: Vec<IgpuManifest> = vec![
            IgpuManifest { version: 2, ..manifest() },
            IgpuManifest { arch: "gfx1201".into(), ..manifest() },
            IgpuManifest { hidden: 2048, ..manifest() },
            IgpuManifest { intermediate: 512, ..manifest() },
            IgpuManifest { experts: 256, ..manifest() },
            IgpuManifest { a4_candidates: 1, ..manifest() },
            IgpuManifest { gate_up_kernel: "other".into(), ..manifest() },
            IgpuManifest { down_kernel: "other".into(), ..manifest() },
            IgpuManifest { down_row_repeat: 2, ..manifest() },
        ];
        for m in changed {
            assert_ne!(m.digest(), base.digest(), "{}", m.canonical());
        }
    }

    #[test]
    fn manifest_ignores_capacity_and_device() {
        // Same arithmetic at a different capacity/device must keep one digest:
        // the manifest is built from the arithmetic fields only.
        let a = IgpuSizing::new(&spec(512)).unwrap();
        let b = IgpuSizing::new(&IgpuEngineSpec { device: 3, ..spec(4096) }).unwrap();
        let build = |s: &IgpuSizing| IgpuManifest {
            hidden: s.hidden,
            intermediate: s.intermediate,
            experts: s.experts,
            k_top: s.k_top,
            ..manifest()
        };
        assert_eq!(build(&a).digest(), build(&b).digest());
    }
}
