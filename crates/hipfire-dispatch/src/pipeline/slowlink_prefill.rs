// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 slow-link layer-major prefill: the dispatch half of one MoE layer over a whole prompt
//! window.
//!
//! Over a narrow host link the incumbent's chunk-major prefill sweeps every host expert once
//! per chunk.  S1 instead runs, per layer, the mixer of every bounded tile first (the
//! architecture's business), then this module's window-wide routed tail, so each host expert
//! crosses the link once per window:
//!
//! 1. [`prepare_routes`], per tile: the incumbent's own prepare stages (normalize, input
//!    basis, router and shared projections, shared activation, route) through
//!    `execute_sealed_prepare`, then a D2D append of the tile's prepared activation, route
//!    ids/weights and shared activation into the window's [`PromptRouteStorage`].
//! 2. [`finish_route_plan`], once: the window-wide A4 quantization of the saved activations,
//!    the stable `(expert, slot)` pad16 grouping of every K2 slot, and the one count barrier.
//! 3. [`LayerRoutePlan::schedule`] + [`execute_panel`], per bounded panel of grouped rows
//!    (a panel never crosses an expert binding; a single expert is split across panels): the
//!    incumbent's K2 gate/up, down-sidecar and down GEMMs over the panel's tiles, then the
//!    ordered combine of the panel into the window accumulator (`slowlink_combine_panel`).
//!    A short-tail tile (`route_rows <= few_row_limit`) is the incumbent's K1 lane-union
//!    cohort ([`qt44_qt53_prefill::slowlink_k1_panel`](super::qt44_qt53_prefill)), run per
//!    binding against the same weight lease, never merged into the K2 grouping.
//! 4. [`finish_combine`]: the canonical terminal products, once, then the K1 cohorts' own
//!    canonical combine.
//! 5. [`finish_shared`], per tile: the saved shared activation restored and the shared down
//!    folded into the tile's rows of the window accumulator.
//!
//! The arithmetic of every kernel is the incumbent's: this module adds no GEMM algorithm.
//! Nothing here records a stage-free event: retiring a panel's weights is the transport
//! owner's job.  All CPU planning (layout, scheduling, ownership) is pure and tested without
//! a device.

use super::layer_ops::hip;
use super::sealed_moe::{checked_grouped_m_total_bound, execute_sealed_prepare, SealedMoeCall};
use crate::families::moe::{MoePrefillParams, MOE_GROUPED_BLOCK_M};
use crate::types::DispatchError;
use rdna_compute::scratch::Int4MmqPrepared;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::marker::PhantomData;
use std::time::Instant;

/// Route ranks per token of the sealed Qwen4 route.
pub const K_TOP: usize = 10;
const ALIGN: usize = 256;
const F32_BYTES: usize = 4;
const I32_BYTES: usize = 4;
const BF16_BYTES: usize = 2;
/// Bytes of one `block_i4_128` record of the IU4 A4 sidecars.
const A4_RECORD_BYTES: usize = 72;
/// Width of the A4 sidecar block (`k / 128` records per row).
const A4_BLOCK_K: usize = 128;

fn bail(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("slowlink layer-major prefill: {message}"))
}

fn mul(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_mul(b).ok_or_else(|| bail("extent overflows"))
}

fn add(a: usize, b: usize) -> Result<usize, DispatchError> {
    a.checked_add(b).ok_or_else(|| bail("extent overflows"))
}

fn align_up(value: usize) -> Result<usize, DispatchError> {
    Ok(add(value, ALIGN - 1)? / ALIGN * ALIGN)
}

// ───────────────────────────── limits ─────────────────────────────

/// Hard bounds of one panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelLimits {
    /// Experts one transport slot (binding) holds.
    pub experts_per_slot: usize,
    /// Grouped rows one panel may span (a multiple of the 16-row tile).
    pub max_grouped_rows: usize,
}

impl PanelLimits {
    /// The frozen S1 bounds: 32-expert slots, 2048 grouped rows.
    pub const S1: PanelLimits = PanelLimits {
        experts_per_slot: 32,
        max_grouped_rows: 2048,
    };

    pub fn validate(&self) -> Result<(), DispatchError> {
        if self.experts_per_slot == 0
            || self.max_grouped_rows == 0
            || self.max_grouped_rows % MOE_GROUPED_BLOCK_M != 0
        {
            return Err(bail(format!(
                "panel limits need experts_per_slot > 0 and a tile-aligned max_grouped_rows, got {self:?}"
            )));
        }
        Ok(())
    }

    /// Bytes of the two BF16 panel scratches a panel needs: SwiGLU rows `[rows, mi]` and down
    /// rows `[rows, down_m]`.
    pub fn scratch_bytes(&self, mi: usize, down_m: usize) -> Result<usize, DispatchError> {
        self.validate()?;
        mul(self.max_grouped_rows, add(mi, down_m)?.checked_mul(BF16_BYTES).ok_or_else(|| bail("extent overflows"))?)
    }
}

// ───────────────────────────── layout ─────────────────────────────

/// Window geometry the route storage is sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptRouteDims {
    /// Rows of the window (the prompt rows this layer sweep covers).
    pub rows: usize,
    /// Hidden width (`gate_up_k`, the routed down's `m`).
    pub hidden: usize,
    pub n_exp: usize,
    /// Expert intermediate width (`down_k`).
    pub mi: usize,
    /// Intermediate width of the shared expert, 0 when the layer has none.
    pub shared_intermediate: usize,
    /// Upper bound of the window rows lowered to the K1 short-tail cohort, 0 when none.
    pub k1_rows: usize,
}

/// Byte offsets of every field of the route storage (one region, 256-aligned fields).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptRouteLayout {
    dims: PromptRouteDims,
    grouped_rows: usize,
    x_norm: usize,
    route_ids: usize,
    topk_ids: usize,
    topk_weights: usize,
    sorted: usize,
    tiles: usize,
    inverse: usize,
    counts: usize,
    offsets: usize,
    accumulator: usize,
    shared_hidden: usize,
    shared_scalar: usize,
    k1_y0: usize,
    k1_ids: usize,
    total: usize,
}

impl PromptRouteLayout {
    pub fn new(dims: &PromptRouteDims) -> Result<Self, DispatchError> {
        let d = *dims;
        if d.rows == 0 || d.hidden == 0 || d.n_exp == 0 || d.mi == 0 || d.k1_rows > d.rows {
            return Err(bail(format!("degenerate route storage dimensions {d:?}")));
        }
        if d.hidden % 8 != 0 || i32::try_from(d.rows).is_err() || d.n_exp > 1024 {
            return Err(bail(format!("unsupported route storage dimensions {d:?}")));
        }
        let slots = mul(d.rows, K_TOP)?;
        let grouped_rows = checked_grouped_m_total_bound(slots, d.n_exp, MOE_GROUPED_BLOCK_M)?;
        let mut next = 0usize;
        let mut field = |bytes: usize| -> Result<usize, DispatchError> {
            let at = next;
            next = add(at, align_up(bytes)?)?;
            Ok(at)
        };
        let x_norm = field(mul(mul(d.rows, d.hidden)?, F32_BYTES)?)?;
        let route_ids = field(mul(slots, I32_BYTES)?)?;
        let topk_ids = field(mul(slots, I32_BYTES)?)?;
        let topk_weights = field(mul(slots, F32_BYTES)?)?;
        let sorted = field(mul(grouped_rows, I32_BYTES)?)?;
        let tiles = field(mul(grouped_rows / MOE_GROUPED_BLOCK_M, I32_BYTES)?)?;
        let inverse = field(mul(slots, I32_BYTES)?)?;
        let counts = field(mul(d.n_exp, I32_BYTES)?)?;
        let offsets = field(mul(add(d.n_exp, 1)?, I32_BYTES)?)?;
        let accumulator = field(mul(mul(d.rows, d.hidden)?, F32_BYTES)?)?;
        let shared_hidden = field(mul(mul(d.rows, d.shared_intermediate)?, F32_BYTES)?)?;
        let shared_scalar = field(if d.shared_intermediate == 0 {
            0
        } else {
            mul(d.rows, F32_BYTES)?
        })?;
        let k1_y0 = field(mul(mul(mul(d.k1_rows, K_TOP)?, d.hidden)?, F32_BYTES)?)?;
        let k1_ids = field(mul(mul(d.k1_rows, K_TOP)?, I32_BYTES)?)?;
        Ok(Self {
            dims: d,
            grouped_rows,
            x_norm,
            route_ids,
            topk_ids,
            topk_weights,
            sorted,
            tiles,
            inverse,
            counts,
            offsets,
            accumulator,
            shared_hidden,
            shared_scalar,
            k1_y0,
            k1_ids,
            total: next,
        })
    }

    pub fn dims(&self) -> &PromptRouteDims {
        &self.dims
    }

    /// Bytes of the route storage region: [`PromptRouteStorage::alloc`] allocates exactly this.
    /// The A4 sidecars ([`Self::a4_sidecar_bytes`]) are Gpu scratch and NOT part of it.
    pub fn total_bytes(&self) -> usize {
        self.total
    }

    /// Bytes of the two IU4 A4 sidecars at window rows (`int4_mmq_x_scratch` of the gate/up
    /// activation and `qwen4_moe_down_i4_scratch` of the down activation): what the producer
    /// kernels reserve when the window's rows quantize.
    pub fn a4_sidecar_bytes(&self) -> usize {
        let d = &self.dims;
        let gate_up = (d.hidden / A4_BLOCK_K) * d.rows * A4_RECORD_BYTES;
        let down = (d.mi / A4_BLOCK_K) * d.rows * K_TOP * A4_RECORD_BYTES;
        gate_up + down
    }

    /// Saved prepared input, F32 `[rows, hidden]`.
    pub fn x_norm_bytes(&self) -> usize {
        self.dims.rows * self.dims.hidden * F32_BYTES
    }
    /// I32 `[rows, 10]` of the route ids and F32 `[rows, 10]` of the weights.
    pub fn topk_ids_bytes(&self) -> usize {
        self.dims.rows * K_TOP * I32_BYTES
    }
    pub fn topk_weights_bytes(&self) -> usize {
        self.dims.rows * K_TOP * F32_BYTES
    }
    /// The stable-grouping maps: sorted slots, tile ids, inverse, counts, offsets.
    pub fn route_map_bytes(&self) -> usize {
        self.grouped_rows * I32_BYTES
            + (self.grouped_rows / MOE_GROUPED_BLOCK_M) * I32_BYTES
            + self.dims.rows * K_TOP * I32_BYTES
            + self.dims.n_exp * I32_BYTES
            + (self.dims.n_exp + 1) * I32_BYTES
    }
    /// Widened BF16 combine accumulator, F32 `[rows, hidden]`.
    pub fn accumulator_bytes(&self) -> usize {
        self.dims.rows * self.dims.hidden * F32_BYTES
    }
    /// Saved shared activation (`N * I * 4`) and selector scalar (`N * 4`).
    pub fn shared_hidden_bytes(&self) -> usize {
        self.dims.rows * self.dims.shared_intermediate * F32_BYTES
    }
    pub fn shared_scalar_bytes(&self) -> usize {
        if self.dims.shared_intermediate == 0 {
            0
        } else {
            self.dims.rows * F32_BYTES
        }
    }
    /// Unweighted F32 down rows of the K1 cohort plus its masked-id scratch.
    pub fn k1_bytes(&self) -> usize {
        self.dims.k1_rows * K_TOP * (self.dims.hidden * F32_BYTES + I32_BYTES)
    }
    /// Upper bound of the padded grouped rows of the window (`G`).
    pub fn grouped_rows_bound(&self) -> usize {
        self.grouped_rows
    }
}

// ───────────────────────────── prepared tile call ─────────────────────────────

/// One tile of the window: the incumbent's sealed grouped-prefill call for those rows,
/// exactly as the incumbent builds it (rows, `route_rows`, recipe, shared bindings, scratch).
/// S1 runs its prepare half and its shared down, never its routed tail.
pub struct PreparedChunkCall<'a> {
    call: &'a SealedMoeCall<'a>,
    rows: usize,
    route_rows: usize,
    row0: usize,
    position: usize,
    k2: bool,
    use_path2: bool,
    hidden: usize,
    n_exp: usize,
    mi: usize,
    down_m: usize,
    shared_intermediate: usize,
}

impl<'a> PreparedChunkCall<'a> {
    /// Pre-mutation admission.  `Err` means the window must stay on the incumbent route; it
    /// is returned before anything is launched or written.  `gpu.route_rows` must already be
    /// set for this tile's rows.
    pub fn new(
        gpu: &Gpu,
        call: &'a SealedMoeCall<'a>,
        row0: usize,
        position: usize,
    ) -> Result<Self, DispatchError> {
        let params = call
            .prefill_params()
            .ok_or_else(|| bail("the call carries no prefill operands"))?;
        let k2 = super::qt44_qt53_prefill::slowlink_admission(gpu, params)?;
        let selection = match call.kernel_selection() {
            super::moe_program::MoeKernelSelection::Prefill(selection) => selection,
            super::moe_program::MoeKernelSelection::Decode(_) => {
                return Err(bail("the call carries a decode selection"))
            }
        };
        if selection.cpu_split || selection.cpu_deferred_experts {
            return Err(bail("a CPU-split or CPU-deferred call is not a layer-major tile"));
        }
        let use_path2 = selection.uses_grouped_rows();
        let shared_intermediate = params.prelude.shared.as_ref().map_or(0, |s| s.intermediate);
        Ok(Self {
            call,
            rows: params.batch_size,
            route_rows: gpu.route_rows(params.batch_size),
            row0,
            position,
            k2,
            use_path2,
            hidden: params.gate_up_k,
            n_exp: params.n_exp,
            mi: params.mi,
            down_m: params.down_m,
            shared_intermediate,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn route_rows(&self) -> usize {
        self.route_rows
    }
    pub fn row0(&self) -> usize {
        self.row0
    }
    pub fn position(&self) -> usize {
        self.position
    }
    /// Whether the tile runs the K2 grouped kernels (else the K1 short-tail cohort).
    pub fn is_k2(&self) -> bool {
        self.k2
    }

    fn params(&self) -> &MoePrefillParams<'_> {
        self.call
            .prefill_params()
            .expect("admitted at construction")
    }

    /// The numerical/shape signature every tile of one window must share.
    fn signature(&self) -> [usize; 6] {
        [
            self.hidden,
            self.n_exp,
            self.mi,
            self.down_m,
            self.shared_intermediate,
            usize::from(self.use_path2),
        ]
    }
}

// ───────────────────────────── storage ─────────────────────────────

#[derive(Clone, Copy, Debug)]
struct TileRec {
    row0: usize,
    rows: usize,
    k1: bool,
    /// First K1 row of this tile in the K1 y0 / masked-id scratch.
    k1_row0: usize,
}

/// The window's per-layer route state, one region reused by every layer.
pub struct PromptRouteStorage {
    region: GpuTensor,
    owned: bool,
    layout: PromptRouteLayout,
    combined: GpuTensor,
    filled: usize,
    k1_used: usize,
    signature: Option<[usize; 6]>,
    tiles: Vec<TileRec>,
    window_xq: Option<Int4MmqPrepared>,
}

impl PromptRouteStorage {
    /// Allocate one region of exactly `layout.total_bytes()`.
    pub fn alloc(gpu: &mut Gpu, layout: &PromptRouteLayout) -> Result<Self, DispatchError> {
        let region = hip(gpu.alloc_tensor(&[layout.total_bytes()], DType::Raw))?;
        Self::over(region, true, layout)
    }

    /// Carve the storage from a region the caller owns (at least `layout.total_bytes()`
    /// bytes, 256-aligned base); the storage never frees it.
    pub fn from_region(region: &GpuTensor, layout: &PromptRouteLayout) -> Result<Self, DispatchError> {
        if region.buf.size() < layout.total_bytes() || region.buf.as_ptr() as usize % ALIGN != 0 {
            return Err(bail("the route region is too small or unaligned"));
        }
        Self::over(region.shallow_clone(), false, layout)
    }

    fn over(region: GpuTensor, owned: bool, layout: &PromptRouteLayout) -> Result<Self, DispatchError> {
        let combined = GpuTensor {
            buf: region
                .buf
                .byte_view(layout.accumulator, layout.accumulator_bytes()),
            shape: vec![layout.dims.rows * layout.dims.hidden],
            dtype: DType::F32,
        };
        Ok(Self {
            region,
            owned,
            layout: layout.clone(),
            combined,
            filled: 0,
            k1_used: 0,
            signature: None,
            tiles: Vec::new(),
            window_xq: None,
        })
    }

    /// Free the region (when owned).  The caller has drained every stream that reads it.
    pub fn release(self, gpu: &mut Gpu) {
        if self.owned {
            let _ = gpu.free_tensor(self.region);
        }
    }

    /// CPU-only: start a new layer (nothing filled, no plan).
    pub fn reset_for_layer(&mut self) {
        self.filled = 0;
        self.k1_used = 0;
        self.signature = None;
        self.tiles.clear();
        self.window_xq = None;
    }

    pub fn rows(&self) -> usize {
        self.layout.dims.rows
    }
    pub fn filled_rows(&self) -> usize {
        self.filled
    }
    pub fn layout(&self) -> &PromptRouteLayout {
        &self.layout
    }
    /// The layer's routed+shared output, F32 `[rows, hidden]` (the tiles' `moe_output`).
    pub fn combined(&self) -> &GpuTensor {
        &self.combined
    }

    fn view(&self, offset: usize, bytes: usize, dtype: DType) -> GpuTensor {
        let shape = if dtype == DType::Raw {
            vec![bytes]
        } else {
            vec![bytes / dtype.size()]
        };
        GpuTensor {
            buf: self.region.buf.byte_view(offset, bytes),
            shape,
            dtype,
        }
    }

    fn ids_view(&self, offset: usize, row0: usize, rows: usize) -> GpuTensor {
        self.view(offset + row0 * K_TOP * I32_BYTES, rows * K_TOP * I32_BYTES, DType::Raw)
    }
    fn f32_rows(&self, offset: usize, row0: usize, rows: usize, width: usize) -> GpuTensor {
        self.view(offset + row0 * width * F32_BYTES, rows * width * F32_BYTES, DType::F32)
    }
}

// ───────────────────────────── prepare ─────────────────────────────

/// Run the tile's prepare stages and append its saved state to `dst` (tiles ascend and are
/// contiguous; every tile of a window shares one signature).
pub fn prepare_routes(
    gpu: &mut Gpu,
    call: &PreparedChunkCall<'_>,
    dst: &mut PromptRouteStorage,
) -> Result<(), DispatchError> {
    let layout = dst.layout.clone();
    let d = layout.dims;
    if call.row0 != dst.filled {
        return Err(bail(format!(
            "tiles must be contiguous: expected row {}, got {}",
            dst.filled, call.row0
        )));
    }
    if call.rows == 0 || call.row0 + call.rows > d.rows {
        return Err(bail("the tile exceeds the window"));
    }
    if call.hidden != d.hidden
        || call.n_exp != d.n_exp
        || call.mi != d.mi
        || call.down_m != d.hidden
        || call.shared_intermediate != d.shared_intermediate
    {
        return Err(bail("the tile's geometry differs from the window's route storage"));
    }
    match dst.signature {
        Some(sig) if sig != call.signature() => {
            return Err(bail("a tile's numerical arm differs from the window's"))
        }
        _ => {}
    }
    let k1 = !call.k2;
    if k1 && dst.k1_used + call.rows > d.k1_rows {
        return Err(bail("the K1 short-tail cohort exceeds its reserved rows"));
    }
    let params = *call.params();
    let slots = call.rows * K_TOP;
    // The incumbent's own prepare stages, for this tile's own call.
    execute_sealed_prepare(gpu, call.call)?;
    // Append the tile's state to the window.
    let copy = |gpu: &Gpu, dst_off: usize, src: &GpuTensor, bytes: usize| {
        hip(gpu.memcpy_dtod_at_ordered_async(&dst.region.buf, dst_off, &src.buf, 0, bytes))
    };
    copy(
        gpu,
        layout.x_norm + call.row0 * d.hidden * F32_BYTES,
        params.x_norm_batch,
        call.rows * d.hidden * F32_BYTES,
    )?;
    copy(
        gpu,
        layout.topk_ids + call.row0 * K_TOP * I32_BYTES,
        params.topk_indices,
        slots * I32_BYTES,
    )?;
    copy(
        gpu,
        layout.topk_weights + call.row0 * K_TOP * F32_BYTES,
        params.topk_weights,
        slots * F32_BYTES,
    )?;
    let route_ids = dst.ids_view(layout.route_ids, call.row0, call.rows);
    if k1 {
        // The K1 cohort stays out of the K2 grouping: unrouted ids for the scatter/combine.
        hip(gpu.slowlink_fill_bytes(&route_ids, 0xFF, slots * I32_BYTES))?;
    } else {
        copy(
            gpu,
            layout.route_ids + call.row0 * K_TOP * I32_BYTES,
            params.topk_indices,
            slots * I32_BYTES,
        )?;
    }
    if let Some(shared) = params.prelude.shared.as_ref() {
        copy(
            gpu,
            layout.shared_hidden + call.row0 * d.shared_intermediate * F32_BYTES,
            shared.rotated,
            call.rows * d.shared_intermediate * F32_BYTES,
        )?;
        copy(
            gpu,
            layout.shared_scalar + call.row0 * F32_BYTES,
            shared.scalar,
            call.rows * F32_BYTES,
        )?;
    }
    dst.tiles.push(TileRec {
        row0: call.row0,
        rows: call.rows,
        k1,
        k1_row0: dst.k1_used,
    });
    if k1 {
        dst.k1_used += call.rows;
    }
    dst.signature = Some(call.signature());
    dst.filled += call.rows;
    Ok(())
}

// ───────────────────────────── route plan ─────────────────────────────

/// What the one count barrier cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BarrierCost {
    pub sync_ms: f64,
    pub bytes_downloaded: usize,
}

/// One bounded panel of grouped rows of one expert binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelRange {
    /// Position in the schedule (the order panels must execute).
    pub index: usize,
    /// Index into the `binding_ranges` the schedule was built from.
    pub binding: usize,
    pub bind_lo: usize,
    pub bind_hi: usize,
    /// First grouped row and row count (a multiple of 16; 0 for a K1-only/pad panel).
    pub row_base: usize,
    pub rows: usize,
    /// Experts holding rows in this panel: `[expert_lo, expert_hi)`.
    pub expert_lo: usize,
    pub expert_hi: usize,
    /// Experts whose invalid-row slots this panel owns; partitions `[0, n_exp)` in order.
    pub own_lo: usize,
    pub own_hi: usize,
    /// Run the K1 cohort against this panel's binding (first panel of a binding that holds a
    /// K1 expert).
    pub k1: bool,
}

/// The layer's grouping: exact padded counts, row starts and the single barrier's cost.
#[derive(Clone, Debug)]
pub struct LayerRoutePlan {
    pub counts_padded: Vec<u32>,
    /// Grouped-row start of every expert, `[n_exp + 1]`.
    pub offsets: Vec<usize>,
    /// Padded grouped rows of the window.
    pub total_rows: usize,
    /// Experts the K1 cohort routes to (ascending, unique).
    pub k1_experts: Vec<usize>,
    pub barrier: BarrierCost,
}

impl LayerRoutePlan {
    /// The plan of device-produced padded counts (each a multiple of 16).
    pub fn from_counts(
        counts: &[i32],
        k1_experts: Vec<usize>,
        barrier: BarrierCost,
    ) -> Result<Self, DispatchError> {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0usize);
        let mut counts_padded = Vec::with_capacity(counts.len());
        for &count in counts {
            if count < 0 || count as usize % MOE_GROUPED_BLOCK_M != 0 {
                return Err(bail(format!("the stable grouping produced a count of {count}")));
            }
            counts_padded.push(count as u32);
            let last = *offsets.last().unwrap();
            offsets.push(add(last, count as usize)?);
        }
        if let Some(&e) = k1_experts.iter().find(|&&e| e >= counts.len()) {
            return Err(bail(format!("K1 expert {e} is out of range")));
        }
        let total_rows = *offsets.last().unwrap();
        Ok(Self {
            counts_padded,
            offsets,
            total_rows,
            k1_experts,
            barrier,
        })
    }

    pub fn n_exp(&self) -> usize {
        self.counts_padded.len()
    }

    /// Experts with at least one routed K2 slot.
    pub fn experts_with_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.counts_padded
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(e, _)| e)
    }

    fn expert_of_row(&self, row: usize) -> usize {
        (self.offsets.partition_point(|&s| s <= row) - 1).min(self.n_exp() - 1)
    }

    /// Bounded panels over `binding_ranges` (ascending, disjoint `[lo, hi)` runs of global
    /// expert ids, one per transport binding).  Rows never cross a binding; a single expert
    /// is split across panels when it exceeds `max_grouped_rows`; a binding without rows or
    /// K1 experts emits nothing; a binding the K1 cohort needs but no K2 row does emits one
    /// zero-row panel.  Always at least one panel.
    pub fn schedule(
        &self,
        binding_ranges: &[(usize, usize)],
        limits: PanelLimits,
    ) -> Result<Vec<PanelRange>, DispatchError> {
        limits.validate()?;
        let n_exp = self.n_exp();
        if binding_ranges.is_empty() {
            return Err(bail("no expert bindings to schedule"));
        }
        let mut prev_hi = 0usize;
        for (i, &(lo, hi)) in binding_ranges.iter().enumerate() {
            if lo >= hi || hi > n_exp || (i > 0 && lo < prev_hi) {
                return Err(bail(format!(
                    "binding ranges must be ascending, disjoint and inside [0, {n_exp}): {binding_ranges:?}"
                )));
            }
            prev_hi = hi;
        }
        // Every routed expert must lie in some binding, or its rows would be dropped.
        for e in self.experts_with_rows().chain(self.k1_experts.iter().copied()) {
            if !binding_ranges.iter().any(|&(lo, hi)| e >= lo && e < hi) {
                return Err(bail(format!("routed expert {e} is in no binding")));
            }
        }
        let mut panels: Vec<PanelRange> = Vec::new();
        for (binding, &(lo, hi)) in binding_ranges.iter().enumerate() {
            let (start, end) = (self.offsets[lo], self.offsets[hi]);
            let k1_needed = self.k1_experts.iter().any(|&e| e >= lo && e < hi);
            let mut first = true;
            let mut base = start;
            while base < end {
                let rows = (end - base).min(limits.max_grouped_rows);
                panels.push(PanelRange {
                    index: panels.len(),
                    binding,
                    bind_lo: lo,
                    bind_hi: hi,
                    row_base: base,
                    rows,
                    expert_lo: self.expert_of_row(base),
                    expert_hi: self.expert_of_row(base + rows - 1) + 1,
                    own_lo: 0,
                    own_hi: 0,
                    k1: first && k1_needed,
                });
                first = false;
                base += rows;
            }
            if first && k1_needed {
                panels.push(PanelRange {
                    index: panels.len(),
                    binding,
                    bind_lo: lo,
                    bind_hi: hi,
                    row_base: end,
                    rows: 0,
                    expert_lo: lo,
                    expert_hi: lo,
                    own_lo: 0,
                    own_hi: 0,
                    k1: true,
                });
            }
        }
        if panels.is_empty() {
            let (lo, hi) = binding_ranges[0];
            panels.push(PanelRange {
                index: 0,
                binding: 0,
                bind_lo: lo,
                bind_hi: hi,
                row_base: 0,
                rows: 0,
                expert_lo: lo,
                expert_hi: lo,
                own_lo: 0,
                own_hi: 0,
                k1: false,
            });
        }
        // Ownership of invalid-row slots: partition [0, n_exp) in panel order, each panel
        // closing at its last row's expert, the last panel at n_exp.
        let count = panels.len();
        let mut prev = 0usize;
        for (i, panel) in panels.iter_mut().enumerate() {
            let hi = if i + 1 == count {
                n_exp
            } else if panel.rows > 0 {
                (self.expert_of_row(panel.row_base + panel.rows - 1) + 1).max(prev)
            } else {
                prev
            };
            panel.own_lo = prev;
            panel.own_hi = hi;
            prev = hi;
        }
        Ok(panels)
    }

    /// The panel's view of the window's route state.
    pub fn view<'a>(&'a self, routes: &'a PromptRouteStorage, panel: &'a PanelRange) -> RoutedPanelView<'a> {
        RoutedPanelView { routes, panel }
    }
}

/// The window's single stable grouping and count barrier.
pub fn finish_route_plan(
    gpu: &mut Gpu,
    routes: &mut PromptRouteStorage,
    _limits: PanelLimits,
) -> Result<LayerRoutePlan, DispatchError> {
    let layout = routes.layout.clone();
    let d = layout.dims;
    if routes.filled != d.rows {
        return Err(bail(format!(
            "the window has {} of {} rows prepared",
            routes.filled, d.rows
        )));
    }
    let slots = d.rows * K_TOP;
    let x_norm = routes.f32_rows(layout.x_norm, 0, d.rows, d.hidden);
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    let sorted = routes.view(layout.sorted, layout.grouped_rows * I32_BYTES, DType::Raw);
    let tiles = routes.view(
        layout.tiles,
        (layout.grouped_rows / MOE_GROUPED_BLOCK_M) * I32_BYTES,
        DType::Raw,
    );
    let inverse = routes.ids_view(layout.inverse, 0, d.rows);
    let counts = routes.view(layout.counts, d.n_exp * I32_BYTES, DType::Raw);
    let offsets = routes.view(layout.offsets, (d.n_exp + 1) * I32_BYTES, DType::Raw);
    // The window's activations quantize once per layer (row-wise, so independent of the tile
    // that produced a row).
    routes.window_xq = Some(hip(gpu.qwen4_moe_rotate256_i4_slab(&x_norm, d.hidden, d.rows))?);
    hip(gpu.moe_scatter_stable_top10_slab(
        &route_ids,
        &counts,
        &offsets,
        &sorted,
        &tiles,
        &inverse,
        slots,
        d.n_exp,
        layout.grouped_rows,
    ))?;
    // The one count barrier: complete the stream, read the padded counts (and the K1
    // cohort's route ids, a few rows).
    let started = Instant::now();
    let counts_host = hip(gpu.slowlink_sync_download_i32(&counts, d.n_exp))?;
    let mut downloaded = d.n_exp * I32_BYTES;
    let mut used = vec![false; d.n_exp];
    for tile in routes.tiles.iter().filter(|t| t.k1) {
        let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
        let host = hip(gpu.slowlink_sync_download_i32(&ids, tile.rows * K_TOP))?;
        downloaded += tile.rows * K_TOP * I32_BYTES;
        for id in host {
            if id >= 0 && (id as usize) < d.n_exp {
                used[id as usize] = true;
            }
        }
    }
    let barrier = BarrierCost {
        sync_ms: started.elapsed().as_secs_f64() * 1e3,
        bytes_downloaded: downloaded,
    };
    let k1_experts = used
        .iter()
        .enumerate()
        .filter_map(|(e, &u)| u.then_some(e))
        .collect();
    let plan = LayerRoutePlan::from_counts(&counts_host, k1_experts, barrier)?;
    if plan.total_rows > layout.grouped_rows {
        return Err(bail("the stable grouping exceeds its row bound"));
    }
    Ok(plan)
}

// ───────────────────────────── weights and panels ─────────────────────────────

/// Codec identity of a panel's weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelCodec {
    Mq4,
}

/// Where a binding's weights live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingSource {
    Resident,
    Staged { slot: u32 },
}

/// Immutable binding of resident or staged expert weights: `u64` pointer tables indexed by
/// GLOBAL expert id (`n_exp` entries each; only `[expert_lo, expert_hi)` are read).
#[derive(Clone, Copy)]
pub struct ExpertPanelBinding<'a> {
    pub layer: usize,
    pub epoch: u64,
    pub expert_lo: usize,
    pub expert_hi: usize,
    pub gate_up_table: &'a GpuTensor,
    pub down_table: &'a GpuTensor,
    pub codec: PanelCodec,
    pub source: BindingSource,
}

impl<'a> ExpertPanelBinding<'a> {
    /// A binding over the ascending, contiguous run of global expert ids `experts`.
    pub fn from_run(
        layer: usize,
        epoch: u64,
        experts: &[usize],
        gate_up_table: &'a GpuTensor,
        down_table: &'a GpuTensor,
        source: BindingSource,
    ) -> Result<Self, DispatchError> {
        let (&lo, &last) = match (experts.first(), experts.last()) {
            (Some(lo), Some(last)) => (lo, last),
            _ => return Err(bail("an empty expert binding")),
        };
        if experts.windows(2).any(|w| w[1] != w[0] + 1) {
            return Err(bail("a binding must be an ascending contiguous run of expert ids"));
        }
        Ok(Self {
            layer,
            epoch,
            expert_lo: lo,
            expert_hi: last + 1,
            gate_up_table,
            down_table,
            codec: PanelCodec::Mq4,
            source,
        })
    }

    /// Every expert of the layer resident behind one table pair.
    pub fn resident(
        layer: usize,
        n_exp: usize,
        gate_up_table: &'a GpuTensor,
        down_table: &'a GpuTensor,
    ) -> Self {
        Self {
            layer,
            epoch: 0,
            expert_lo: 0,
            expert_hi: n_exp,
            gate_up_table,
            down_table,
            codec: PanelCodec::Mq4,
            source: BindingSource::Resident,
        }
    }
}

/// A panel and the window state it reads.
#[derive(Clone, Copy)]
pub struct RoutedPanelView<'a> {
    routes: &'a PromptRouteStorage,
    panel: &'a PanelRange,
}

impl RoutedPanelView<'_> {
    pub fn panel(&self) -> &PanelRange {
        self.panel
    }
}

/// The BF16 panel scratches and the tile call the K1 cohort borrows its scratch from.
pub struct PanelScratch<'a> {
    params: MoePrefillParams<'a>,
    limits: PanelLimits,
}

impl<'a> PanelScratch<'a> {
    /// Borrow the y scratches of `call` (the model's global grouped scratch: any tile's call
    /// of the layer names the same tensors).
    pub fn new(call: &'a PreparedChunkCall<'a>, limits: PanelLimits) -> Result<Self, DispatchError> {
        let params = *call.params();
        let mi_bytes = mul(mul(limits.max_grouped_rows, params.mi)?, BF16_BYTES)?;
        let down_bytes = mul(mul(limits.max_grouped_rows, params.down_m)?, BF16_BYTES)?;
        if params.y_gate_up_grouped.buf.size() < mi_bytes || params.y_down_grouped.buf.size() < down_bytes {
            return Err(bail(format!(
                "panel scratch holds {} + {} bytes, a {}-row panel needs {mi_bytes} + {down_bytes}",
                params.y_gate_up_grouped.buf.size(),
                params.y_down_grouped.buf.size(),
                limits.max_grouped_rows
            )));
        }
        Ok(Self { params, limits })
    }

    /// Bytes of both panel scratches at the hard bound.
    pub fn bytes(&self) -> usize {
        self.limits
            .scratch_bytes(self.params.mi, self.params.down_m)
            .unwrap_or(usize::MAX)
    }
}

/// The ordered combine's progress over one layer.
pub struct OrderedCombineCursor<'a> {
    rows: usize,
    total_rows: usize,
    expected_panels: usize,
    panels_done: usize,
    next_row_base: usize,
    combined: bool,
    _routes: PhantomData<&'a PromptRouteStorage>,
}

impl<'a> OrderedCombineCursor<'a> {
    /// Zero the window accumulator (+0.0, exactly the incumbent's cleared target) and expect
    /// `panels` in order.
    pub fn start(
        gpu: &mut Gpu,
        routes: &'a PromptRouteStorage,
        panels: &[PanelRange],
    ) -> Result<Self, DispatchError> {
        if panels.is_empty() || panels.iter().enumerate().any(|(i, p)| p.index != i) {
            return Err(bail("the schedule's panels are not a 0-based sequence"));
        }
        let d = routes.layout.dims;
        hip(gpu.slowlink_fill_bytes(&routes.combined, 0, routes.layout.accumulator_bytes()))?;
        let total_rows = panels.iter().map(|p| p.rows).sum();
        Ok(Self {
            rows: d.rows,
            total_rows,
            expected_panels: panels.len(),
            panels_done: 0,
            next_row_base: panels.iter().find(|p| p.rows > 0).map_or(0, |p| p.row_base),
            combined: false,
            _routes: PhantomData,
        })
    }

    pub fn panels_done(&self) -> usize {
        self.panels_done
    }

    pub fn is_complete(&self) -> bool {
        self.combined
    }
}

/// One panel: the K2 GEMMs over the panel's grouped rows and its ordered combine, then the K1
/// cohort against the panel's binding when the schedule says so.
pub fn execute_panel(
    gpu: &mut Gpu,
    weights: ExpertPanelBinding<'_>,
    panel: RoutedPanelView<'_>,
    scratch: &mut PanelScratch<'_>,
    combine: &mut OrderedCombineCursor<'_>,
) -> Result<(), DispatchError> {
    let (routes, range) = (panel.routes, panel.panel);
    let layout = &routes.layout;
    let d = layout.dims;
    if combine.combined || range.index != combine.panels_done || range.index >= combine.expected_panels {
        return Err(bail("panels must execute once each in schedule order"));
    }
    if range.rows > scratch.limits.max_grouped_rows {
        return Err(bail("a panel exceeds the hard row bound"));
    }
    if range.rows > 0 && range.row_base != combine.next_row_base {
        return Err(bail("panel rows are not contiguous"));
    }
    let covers = |lo: usize, hi: usize| weights.expert_lo <= lo && hi <= weights.expert_hi;
    if (range.rows > 0 && !covers(range.expert_lo, range.expert_hi))
        || (range.k1 && !covers(range.bind_lo, range.bind_hi))
    {
        return Err(bail("the weight binding does not hold the panel's experts"));
    }
    let params = scratch.params;
    let slots = d.rows * K_TOP;
    if range.rows > 0 {
        let xq = routes
            .window_xq
            .as_ref()
            .ok_or_else(|| bail("the window's route plan has not run"))?;
        let tile0 = range.row_base / MOE_GROUPED_BLOCK_M;
        let tiles = routes.view(
            layout.tiles + tile0 * I32_BYTES,
            (range.rows / MOE_GROUPED_BLOCK_M) * I32_BYTES,
            DType::Raw,
        );
        let sorted = routes.view(
            layout.sorted + range.row_base * I32_BYTES,
            range.rows * I32_BYTES,
            DType::Raw,
        );
        hip(gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            weights.gate_up_table,
            &tiles,
            &sorted,
            xq,
            params.y_gate_up_grouped,
            2 * d.mi,
            d.hidden,
            K_TOP,
            range.rows,
            d.rows,
            false,
        ))?;
        let xdown = hip(gpu.qwen4_moe_rotate128_i4_slab(
            params.y_gate_up_grouped,
            &sorted,
            d.mi,
            range.rows,
            slots,
        ))?;
        hip(gpu.gemm_qwen4_moe_down_iu4_sym(
            weights.down_table,
            &tiles,
            &sorted,
            &xdown,
            params.y_down_grouped,
            d.hidden,
            d.mi,
            1,
            range.rows,
            slots,
            false,
        ))?;
    }
    // The ordered combine; a zero-row panel still applies its ownership range.
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    let weights_t = routes.f32_rows(layout.topk_weights, 0, d.rows, K_TOP);
    let inverse = routes.ids_view(layout.inverse, 0, d.rows);
    hip(gpu.slowlink_combine_panel(
        params.y_down_grouped,
        &inverse,
        &route_ids,
        &weights_t,
        &routes.combined,
        d.hidden,
        d.rows,
        d.n_exp,
        combine.total_rows,
        range.row_base,
        range.rows,
        range.own_lo,
        range.own_hi,
    ))?;
    if range.k1 {
        run_k1_cohort(gpu, &params, routes, &weights, range)?;
    }
    combine.panels_done += 1;
    combine.next_row_base = range.row_base + range.rows;
    Ok(())
}

/// The K1 cohort of every short-tail tile against one binding: the incumbent's K1 arithmetic
/// over the tile's route ids masked to the binding.
fn run_k1_cohort(
    gpu: &mut Gpu,
    params: &MoePrefillParams<'_>,
    routes: &PromptRouteStorage,
    weights: &ExpertPanelBinding<'_>,
    range: &PanelRange,
) -> Result<(), DispatchError> {
    let layout = &routes.layout;
    let d = layout.dims;
    for tile in routes.tiles.iter().filter(|t| t.k1) {
        let slots = tile.rows * K_TOP;
        let active = slots * d.mi * F32_BYTES;
        if params.gate_batch.buf.size() < active
            || params.up_batch.buf.size() < active
            || params.rot_batch.buf.size() < active
            || params.x_norm_batch.buf.size() < tile.rows * d.hidden * F32_BYTES
        {
            return Err(bail("the model's scratch cannot hold a K1 short-tail tile"));
        }
        // Restore the tile's prepared input into the model's scratch.
        hip(gpu.memcpy_dtod_at_ordered_async(
            &params.x_norm_batch.buf,
            0,
            &routes.region.buf,
            layout.x_norm + tile.row0 * d.hidden * F32_BYTES,
            tile.rows * d.hidden * F32_BYTES,
        ))?;
        let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
        let masked = routes.view(
            layout.k1_ids + tile.k1_row0 * K_TOP * I32_BYTES,
            slots * I32_BYTES,
            DType::Raw,
        );
        hip(gpu.slowlink_mask_ids(&ids, &masked, slots, range.bind_lo, range.bind_hi))?;
        let y0 = routes.f32_rows(layout.k1_y0, tile.k1_row0 * K_TOP, slots, d.hidden);
        let mut view = *params;
        view.batch_size = tile.rows;
        super::qt44_qt53_prefill::slowlink_k1_panel(
            gpu,
            &view,
            &masked,
            weights.gate_up_table,
            weights.down_table,
            &y0,
        )?;
    }
    Ok(())
}

// ───────────────────────────── finish ─────────────────────────────

/// The canonical terminal products (once), then the K1 cohort's own canonical combine.
pub fn finish_combine(
    gpu: &mut Gpu,
    routes: &PromptRouteStorage,
    combine: &mut OrderedCombineCursor<'_>,
) -> Result<(), DispatchError> {
    if combine.combined || combine.panels_done != combine.expected_panels {
        return Err(bail("the ordered combine is incomplete or already finished"));
    }
    let layout = &routes.layout;
    let d = layout.dims;
    if combine.rows != d.rows {
        return Err(bail("the cursor belongs to another window"));
    }
    let route_ids = routes.ids_view(layout.route_ids, 0, d.rows);
    // K1 rows carry no routed id here, so the pad leaves their (+0) accumulator unchanged.
    hip(gpu.slowlink_combine_pad(&route_ids, &routes.combined, d.hidden, d.rows, d.n_exp))?;
    for tile in routes.tiles.iter().filter(|t| t.k1) {
        let slots = tile.rows * K_TOP;
        let y0 = routes.f32_rows(layout.k1_y0, tile.k1_row0 * K_TOP, slots, d.hidden);
        let ids = routes.ids_view(layout.topk_ids, tile.row0, tile.rows);
        let weights = routes.f32_rows(layout.topk_weights, tile.row0, tile.rows, K_TOP);
        let target = routes
            .combined
            .sub_offset(tile.row0 * d.hidden, tile.rows * d.hidden);
        hip(gpu.moe_down_combine_top10_batched(
            &y0, &ids, &weights, &target, d.hidden, tile.rows, d.n_exp,
        ))?;
    }
    combine.combined = true;
    Ok(())
}

/// The tile's shared down: restore its saved shared activation and selector, then fold the
/// shared expert into the tile's rows of the window accumulator.
pub fn finish_shared(
    gpu: &mut Gpu,
    routes: &PromptRouteStorage,
    call: &PreparedChunkCall<'_>,
) -> Result<(), DispatchError> {
    let layout = &routes.layout;
    let d = layout.dims;
    let rec = routes
        .tiles
        .iter()
        .find(|t| t.row0 == call.row0 && t.rows == call.rows)
        .ok_or_else(|| bail("the tile was not prepared in this window"))?;
    let params = call.params();
    let target = routes
        .combined
        .sub_offset(rec.row0 * d.hidden, rec.rows * d.hidden);
    if let Some(shared) = params.prelude.shared.as_ref() {
        hip(gpu.memcpy_dtod_at_ordered_async(
            &shared.rotated.buf,
            0,
            &routes.region.buf,
            layout.shared_hidden + rec.row0 * d.shared_intermediate * F32_BYTES,
            rec.rows * d.shared_intermediate * F32_BYTES,
        ))?;
        hip(gpu.memcpy_dtod_at_ordered_async(
            &shared.scalar.buf,
            0,
            &routes.region.buf,
            layout.shared_scalar + rec.row0 * F32_BYTES,
            rec.rows * F32_BYTES,
        ))?;
    }
    super::qt44_qt53_prefill::slowlink_shared_down(gpu, params, call.use_path2, &target)
}

#[cfg(test)]
mod slowlink_panel_tests {
    use super::*;
    use rdna_compute::slab_moe::slowlink_panel_owns;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    const N_EXP: usize = 512;

    fn dims(rows: usize) -> PromptRouteDims {
        PromptRouteDims {
            rows,
            hidden: 2560,
            n_exp: N_EXP,
            mi: 640,
            shared_intermediate: 640,
            k1_rows: rows.min(16),
        }
    }

    /// CPU twin of `qwen4_moe_group_*`: stable `(expert, slot)` grouping, pad16 per expert.
    struct Grouping {
        counts_padded: Vec<i32>,
        start: Vec<usize>,
        /// `inverse[slot]`, -1 for an unrouted slot.
        inverse: Vec<i32>,
        sorted: Vec<i32>,
    }

    fn group(ids: &[i32]) -> Grouping {
        let mut count = vec![0usize; N_EXP];
        for &id in ids {
            if (0..N_EXP as i32).contains(&id) {
                count[id as usize] += 1;
            }
        }
        let counts_padded: Vec<i32> = count.iter().map(|&c| (c.div_ceil(16) * 16) as i32).collect();
        let mut start = vec![0usize; N_EXP + 1];
        for e in 0..N_EXP {
            start[e + 1] = start[e] + counts_padded[e] as usize;
        }
        let mut fill = vec![0usize; N_EXP];
        let mut inverse = vec![-1i32; ids.len()];
        let mut sorted = vec![-1i32; start[N_EXP]];
        for (slot, &id) in ids.iter().enumerate() {
            if (0..N_EXP as i32).contains(&id) {
                let e = id as usize;
                let row = start[e] + fill[e];
                fill[e] += 1;
                inverse[slot] = row as i32;
                sorted[row] = slot as i32;
            }
        }
        Grouping { counts_padded, start, inverse, sorted }
    }

    fn plan(ids: &[i32], k1: Vec<usize>) -> (Grouping, LayerRoutePlan) {
        let g = group(ids);
        let p = LayerRoutePlan::from_counts(&g.counts_padded, k1, BarrierCost::default()).unwrap();
        (g, p)
    }

    fn runs(width: usize) -> Vec<(usize, usize)> {
        (0..N_EXP.div_ceil(width))
            .map(|i| (i * width, ((i + 1) * width).min(N_EXP)))
            .collect()
    }

    fn random_ids(rng: &mut Rng, tokens: usize, hot: usize) -> Vec<i32> {
        let mut ids = Vec::new();
        for _ in 0..tokens {
            for _ in 0..K_TOP {
                ids.push(match rng.below(20) {
                    0 => -1,
                    1 => N_EXP as i32 + rng.below(9) as i32,
                    2..=6 => (hot + rng.below(3)) as i32,
                    _ => rng.below(N_EXP) as i32,
                });
            }
        }
        ids
    }

    #[test]
    fn slowlink_panel_layout_is_aligned_disjoint_and_census_exact() {
        for rows in [1usize, 3, 16, 1536, 1537, 16384] {
            let layout = PromptRouteLayout::new(&dims(rows)).unwrap();
            assert_eq!(layout.total_bytes() % ALIGN, 0);
            let fields = [
                layout.x_norm,
                layout.route_ids,
                layout.topk_ids,
                layout.topk_weights,
                layout.sorted,
                layout.tiles,
                layout.inverse,
                layout.counts,
                layout.offsets,
                layout.accumulator,
                layout.shared_hidden,
                layout.shared_scalar,
                layout.k1_y0,
                layout.k1_ids,
            ];
            assert!(fields.iter().all(|f| f % ALIGN == 0));
            assert!(fields.windows(2).all(|w| w[0] <= w[1]));
            // The named byte accounts sum (before alignment) to at most the region and the
            // region exceeds them by alignment slack only.
            let named = layout.x_norm_bytes()
                + layout.topk_ids_bytes() * 3
                + layout.route_map_bytes()
                + layout.accumulator_bytes()
                + layout.shared_hidden_bytes()
                + layout.shared_scalar_bytes()
                + layout.k1_bytes();
            assert!(layout.total_bytes() >= named);
            assert!(layout.total_bytes() - named <= 14 * ALIGN, "rows {rows}");
        }
        let l = PromptRouteLayout::new(&dims(16384)).unwrap();
        assert_eq!(l.a4_sidecar_bytes(), 20 * 16384 * 72 + 5 * 16384 * 10 * 72);
        assert_eq!(l.a4_sidecar_bytes(), 82_575_360);
    }

    #[test]
    fn slowlink_panel_layout_refuses_degenerate_and_counts_the_shared_dims() {
        let mut d = dims(1536);
        d.rows = 0;
        assert!(PromptRouteLayout::new(&d).is_err());
        let mut d = dims(1536);
        d.k1_rows = 1537;
        assert!(PromptRouteLayout::new(&d).is_err());
        let with = PromptRouteLayout::new(&dims(1536)).unwrap();
        let mut d = dims(1536);
        d.shared_intermediate = 0;
        d.k1_rows = 0;
        let without = PromptRouteLayout::new(&d).unwrap();
        assert_eq!(without.shared_hidden_bytes(), 0);
        assert_eq!(without.shared_scalar_bytes(), 0);
        assert_eq!(with.shared_hidden_bytes(), 1536 * 640 * 4);
        assert_eq!(with.shared_scalar_bytes(), 1536 * 4);
        assert!(with.total_bytes() > without.total_bytes());
    }

    #[test]
    fn slowlink_panel_counts_are_pad16_and_offsets_monotone() {
        let mut rng = Rng(0x5eed);
        let ids = random_ids(&mut rng, 700, 40);
        let (g, p) = plan(&ids, Vec::new());
        assert!(p.counts_padded.iter().all(|c| c % 16 == 0));
        assert!(p.offsets.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(p.total_rows, g.start[N_EXP]);
        assert_eq!(p.offsets[N_EXP], p.total_rows);
        assert!(LayerRoutePlan::from_counts(&[3], Vec::new(), BarrierCost::default()).is_err());
        assert!(LayerRoutePlan::from_counts(&[-16], Vec::new(), BarrierCost::default()).is_err());
        assert!(LayerRoutePlan::from_counts(&[16], vec![1], BarrierCost::default()).is_err());
    }

    #[test]
    fn slowlink_panel_schedule_is_bounded_contiguous_and_within_bindings() {
        let mut rng = Rng(0xfeed_f00d);
        for trial in 0..24 {
            let tokens = [1usize, 7, 300, 1536][trial % 4];
            let hot = rng.below(N_EXP - 4);
            let ids = random_ids(&mut rng, tokens, hot);
            let (g, p) = plan(&ids, Vec::new());
            for width in [32usize, 64, 512] {
                for max_rows in [16usize, 48, 256, 2048] {
                    let limits = PanelLimits { experts_per_slot: width, max_grouped_rows: max_rows };
                    let bindings = runs(width);
                    let panels = p.schedule(&bindings, limits).unwrap();
                    assert!(!panels.is_empty());
                    let mut covered = vec![false; p.total_rows];
                    for (i, panel) in panels.iter().enumerate() {
                        assert_eq!(panel.index, i);
                        assert!(panel.rows <= limits.max_grouped_rows, "scratch hard bound");
                        assert_eq!(panel.rows % 16, 0);
                        assert_eq!(panel.row_base % 16, 0);
                        let (lo, hi) = bindings[panel.binding];
                        assert_eq!((panel.bind_lo, panel.bind_hi), (lo, hi));
                        if panel.rows > 0 {
                            assert!(panel.row_base >= p.offsets[lo] && panel.row_base + panel.rows <= p.offsets[hi]);
                            assert!(panel.expert_lo >= lo && panel.expert_hi <= hi);
                            assert!(panel.expert_lo < panel.expert_hi);
                            for r in panel.row_base..panel.row_base + panel.rows {
                                assert!(!covered[r], "row {r} scheduled twice");
                                covered[r] = true;
                                let e = p.expert_of_row(r);
                                assert!(e >= panel.expert_lo && e < panel.expert_hi);
                            }
                        }
                    }
                    assert!(covered.iter().all(|&c| c), "a grouped row was dropped");
                    // Panels ascend by row and binding.
                    assert!(panels.windows(2).all(|w| w[0].binding <= w[1].binding
                        && w[0].row_base + w[0].rows <= w[1].row_base));
                    // Ownership partitions [0, n_exp) in panel order.
                    let mut next = 0;
                    for panel in &panels {
                        assert_eq!(panel.own_lo, next);
                        assert!(panel.own_hi >= panel.own_lo);
                        next = panel.own_hi;
                    }
                    assert_eq!(next, N_EXP);
                    let _ = &g;
                }
            }
        }
    }

    #[test]
    fn slowlink_panel_single_expert_splits_across_panels() {
        // 5000 slots on one expert: no unbounded exception, 3 panels of <= 2048 rows.
        let ids = vec![7i32; 5000];
        let (_, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 3);
        assert_eq!(panels.iter().map(|x| x.rows).sum::<usize>(), 5008);
        assert!(panels.iter().all(|x| x.rows <= 2048 && x.expert_lo == 7 && x.expert_hi == 8));
        // The split expert's invalid-row slots are owned by exactly its first panel.
        assert_eq!((panels[0].own_lo, panels[0].own_hi), (0, 8));
        assert_eq!((panels[1].own_lo, panels[1].own_hi), (8, 8));
        assert_eq!((panels[2].own_lo, panels[2].own_hi), (8, N_EXP));
    }

    #[test]
    fn slowlink_panel_sparse_duplicate_invalid_ids_and_empty_windows() {
        // Nothing routed at all: one zero-row combine-only panel owning everything.
        let ids = vec![-1i32; 40];
        let (_, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 1);
        let only = panels[0];
        assert_eq!((only.rows, only.own_lo, only.own_hi, only.k1), (0, 0, N_EXP, false));
        // Sparse: experts 3 and 500 only: two panels in two bindings, the middle ones skipped.
        let mut ids = vec![3i32; 10];
        ids.extend(vec![500i32; 10]);
        ids.extend([3, 3, 500, N_EXP as i32, -5, i32::MAX, i32::MIN, 3, 3, 3]);
        let (_, p) = plan(&ids, Vec::new());
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 2);
        assert_eq!((panels[0].bind_lo, panels[1].bind_lo), (0, 480));
        // An expert outside every binding is refused rather than dropped.
        let err = p.schedule(&[(0, 100)], PanelLimits::S1).unwrap_err();
        assert!(format!("{err:?}").contains("in no binding"));
        // Malformed bindings and limits.
        assert!(p.schedule(&[], PanelLimits::S1).is_err());
        assert!(p.schedule(&[(10, 20), (15, 30)], PanelLimits::S1).is_err());
        assert!(p.schedule(&[(20, 10)], PanelLimits::S1).is_err());
        assert!(p.schedule(&[(0, N_EXP + 1)], PanelLimits::S1).is_err());
        let bad = PanelLimits { experts_per_slot: 32, max_grouped_rows: 100 };
        assert!(p.schedule(&runs(32), bad).is_err());
    }

    #[test]
    fn slowlink_panel_k1_cohort_binding_without_rows_gets_a_zero_row_panel() {
        // K2 rows only in binding 0; the K1 tail routes to experts 40 (binding 1) and 3.
        let ids = vec![3i32; 20];
        let (_, p) = plan(&ids, vec![3, 40]);
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 2);
        assert!(panels[0].k1 && panels[0].rows == 32);
        assert!(panels[1].k1 && panels[1].rows == 0 && panels[1].bind_lo == 32);
        // K1 runs once per binding even when the binding's rows split into several panels.
        let ids = vec![3i32; 5000];
        let (_, p) = plan(&ids, vec![3]);
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.iter().filter(|x| x.k1).count(), 1);
        assert!(panels[0].k1);
        // K1-only window: no K2 rows, the one zero-row panel per K1 binding.
        let (_, p) = plan(&vec![-1i32; 10], vec![5, 500]);
        let panels = p.schedule(&runs(32), PanelLimits::S1).unwrap();
        assert_eq!(panels.len(), 2);
        assert!(panels.iter().all(|x| x.rows == 0 && x.k1));
        assert_eq!(panels.last().unwrap().own_hi, N_EXP);
    }

    #[test]
    fn slowlink_panel_every_slot_is_owned_once_and_inverse_maps_into_its_panel() {
        let mut rng = Rng(0xabcd_ef01);
        for _ in 0..6 {
            let tokens = 100 + rng.below(300);
            let hot = rng.below(N_EXP - 4);
            let ids = random_ids(&mut rng, tokens, hot);
            let (g, p) = plan(&ids, Vec::new());
            for (width, max_rows) in [(32usize, 2048usize), (32, 64), (64, 16), (512, 512)] {
                let panels = p
                    .schedule(&runs(width), PanelLimits { experts_per_slot: width, max_grouped_rows: max_rows })
                    .unwrap();
                for (slot, &id) in ids.iter().enumerate() {
                    let valid = (0..N_EXP as i32).contains(&id);
                    let owners: Vec<usize> = panels
                        .iter()
                        .filter(|q| {
                            slowlink_panel_owns(
                                id,
                                g.inverse[slot],
                                N_EXP,
                                p.total_rows,
                                q.row_base,
                                q.rows,
                                q.own_lo,
                                q.own_hi,
                            )
                        })
                        .map(|q| q.index)
                        .collect();
                    assert_eq!(owners.len(), usize::from(valid), "slot {slot} id {id}");
                    if valid {
                        let q = &panels[owners[0]];
                        let row = g.inverse[slot] as usize;
                        assert!(row >= q.row_base && row < q.row_base + q.rows);
                        // Inverse mapping: the grouped row names its slot and its expert.
                        assert_eq!(g.sorted[row], slot as i32);
                        assert_eq!(p.expert_of_row(row), id as usize);
                        assert!(id as usize >= q.expert_lo && (id as usize) < q.expert_hi);
                    } else {
                        assert_eq!(g.inverse[slot], -1);
                    }
                }
                // Stability: rows of one expert ascend with the flat slot.
                for e in 0..N_EXP {
                    let rows = &g.sorted[g.start[e]..g.start[e + 1]];
                    let live: Vec<i32> = rows.iter().copied().filter(|&s| s >= 0).collect();
                    assert!(live.windows(2).all(|w| w[0] < w[1]));
                }
            }
        }
    }

    #[test]
    fn slowlink_panel_invalid_row_valid_id_slots_are_owned_by_one_panel() {
        // A valid id whose grouped row is invalid (the canonical kernel multiplies +0 by its
        // weight) belongs to exactly the panel whose ownership range holds its id.
        let ids = vec![3i32; 5000];
        let (_, p) = plan(&ids, vec![]);
        let mut ids2 = ids.clone();
        ids2.extend([200, 300, 511]);
        let (_, p2) = plan(&ids2, vec![]);
        for (plan, bad_id) in [(&p, 3i32), (&p2, 200), (&p2, 300), (&p2, 511), (&p2, 0)] {
            let panels = plan.schedule(&runs(32), PanelLimits::S1).unwrap();
            for bad_row in [-1i32, plan.total_rows as i32 + 7, i32::MAX] {
                let owners = panels
                    .iter()
                    .filter(|q| {
                        slowlink_panel_owns(bad_id, bad_row, N_EXP, plan.total_rows, q.row_base, q.rows, q.own_lo, q.own_hi)
                    })
                    .count();
                assert_eq!(owners, 1, "id {bad_id} row {bad_row}");
            }
        }
    }

    #[test]
    fn slowlink_panel_limits_bound_the_scratch() {
        let l = PanelLimits::S1;
        assert_eq!(l.scratch_bytes(640, 2560).unwrap(), 2048 * (640 + 2560) * 2);
        assert!(PanelLimits { experts_per_slot: 0, max_grouped_rows: 16 }.validate().is_err());
        assert!(PanelLimits { experts_per_slot: 32, max_grouped_rows: 0 }.validate().is_err());
        assert!(PanelLimits { experts_per_slot: 32, max_grouped_rows: 17 }.validate().is_err());
        assert!(PanelLimits { experts_per_slot: 32, max_grouped_rows: usize::MAX & !15 }
            .scratch_bytes(640, 2560)
            .is_err());
    }
}
