// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! S1 slow-link layer-major prompt prefill.
//!
//! The incumbent prefills a long prompt chunk-major: every chunk walks all
//! layers, and on a narrow host link every chunk therefore re-streams every
//! host-resident routed expert. This route walks the same prompt layer-major:
//! a window of up to [`S1_WINDOW_ROWS`] rows crosses each layer once, in the
//! original 1,536-row mixer tiles, so each host layer's expert payload crosses
//! the link **once per window** while layer `L`'s tail and layer `L+1`'s mixer
//! compute under the copies of `L+1`'s weights.
//!
//! # Ownership
//!
//! * the typed per-layer operations are built by
//!   [`crate::gpu_forward::LayerPhaseEnv`], the same builder the incumbent
//!   loop uses, so no second arithmetic lowering exists;
//! * the expert page ring and the layer-ahead schedule are
//!   [`crate::gpu_forward::panel_stage`];
//! * route preparation, the single count barrier per layer, panel GEMMs, the
//!   canonical BF16 combine and the shared expert are
//!   `hipfire_dispatch::pipeline::slowlink_prefill`;
//! * admission, the selector truth table and every byte census are
//!   [`crate::slowlink_admission`];
//! * the entry snapshot, per-boundary snapshots, publication and rollback are
//!   the bundle's `prompt_txn_*` transaction.
//!
//! This module owns the schedule: the pure plan (tiles, windows, boundaries),
//! the request selector, the layer-major run and the receipt.
//!
//! # Only host layers are layer-major
//!
//! A device-resident layer reads its experts in place at full speed. Such a
//! layer runs exactly the incumbent per-tile steps (prefix, MoE, HyperWrite) on
//! its tile, so its arithmetic is the incumbent's. A host layer splits at its
//! route barrier: every tile's mixer prefix and route preparation first, then
//! one count barrier, the panels in ascending expert order, and every tile's
//! shared expert and HyperWrite.
//!
//! # Numerical manifest
//!
//! Tile `t` of a window is the incumbent chunk `t` of the same chronology: the
//! tile row counts, the routing-row signature (`Gpu::route_rows`), the HC
//! stream tag (BF16 or F32), the QSA/GDN positions and the PLE history are all
//! per tile and equal the 1,536-row chunk-major run. Only the host layers'
//! expert combine is the canonical panel combine instead of the per-chunk
//! sealed call, so this route is validated by exact greedy ids and paired KLD,
//! never labelled bitwise against the 8,192-row incumbent.

use crate::bundle::{PromptTxn, Qwen4Bundle, SlowlinkFailPoint};
use crate::config::Qwen4Config;
use crate::gpu_forward::panel_stage::{
    ExpertGroupPlan, ExpertPanelRing, OverlapSummary, PanelBackend, PanelBindingParts,
    PanelSchedule, PanelStats, PanelTicket, RingCore,
};
use crate::gpu_forward::{
    f32_view, invalid, layer_desc, layer_scratch, matrix_view, program_dims, router_columns,
    stage_ple_rows, view, LayerPhaseEnv, PleUploadMode, Qwen4GpuForward, Qwen4GpuForwardError,
};
use crate::ple::PleHistory;
use crate::ple_stage::ple_async_upload;
use crate::program::{
    execute_final_hyper, execute_lm_head, validate_final_hyper, validate_lm_head, Qwen4FfnBinding,
    Qwen4ProgramDims,
};
use crate::slowlink_admission::{
    qualify_request, ArenaField, CaptureKind, RequestBypass, RequestShape, SlowlinkMode,
    SlowlinkProfile, EXPERTS_PER_PAGE, S1_K1_TAIL_ROWS, S1_MAX_BOUNDARY_SNAPSHOTS,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::slowlink_prefill as dsp;
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, validate_steps, EmbeddingOp, Step,
};
use hipfire_runtime::external_rows::{RowFetch, RowTicket};
use rdna_compute::tensor_ops::{argmax_f32, ensure_qsa_gathered_wmma_workspace, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::ops::Range;
use std::time::Instant;

type R<T> = Result<T, Qwen4GpuForwardError>;

/// Receipt schema string (`V-RECEIPT-API.md`).
pub(crate) const RECEIPT_SCHEMA: &str = "hipfire.qwen4.slowlink_receipt.v1";

// ── the pure plan ────────────────────────────────────────────────────

/// What one tile's launch signature depends on at runtime: the routing rows
/// every row-count routing decision of the tile sees and the HC stream tag.
/// Both are decided once per tile from [`Gpu::route_rows`] and
/// [`Gpu::qwen4_bf16_streams`] under the tile's scope, exactly as the
/// incumbent decides them per chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileRouting {
    pub(crate) route_rows: usize,
    pub(crate) bf16_streams: bool,
}

/// One mixer tile: the unit the incumbent would have run as one chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TilePlan {
    /// Index in the request's chronological tile list.
    pub(crate) index: usize,
    pub(crate) window: usize,
    /// First request row (0 = first prompt row after the entry position).
    pub(crate) prompt_row: usize,
    /// First row inside its window (the arena row base).
    pub(crate) window_row: usize,
    pub(crate) rows: usize,
    /// Absolute position of the tile's first row.
    pub(crate) position: usize,
    pub(crate) routing: TileRouting,
    /// `Some(i)` when the tile ends at the window's `i`-th strictly-inside
    /// session boundary (a snapshot cut the layer's state is copied at).
    pub(crate) ends_at_cut: Option<usize>,
    pub(crate) last_in_window: bool,
}

impl TilePlan {
    /// Absolute position one past the tile's last row.
    pub(crate) fn end_position(&self) -> usize {
        self.position + self.rows
    }
}

/// One window: the rows that share one expert sweep, one transaction and one
/// route/arena allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WindowPlan {
    pub(crate) index: usize,
    pub(crate) prompt_row: usize,
    pub(crate) rows: usize,
    /// Absolute position of the window's first row.
    pub(crate) position: usize,
    pub(crate) tiles: Range<usize>,
    /// Session boundaries strictly inside the window (absolute, ascending).
    pub(crate) cuts: Vec<usize>,
}

impl WindowPlan {
    pub(crate) fn end_position(&self) -> usize {
        self.position + self.rows
    }
}

/// Everything the profile charges for a request, from the same layouts the
/// loader's census read: the receipt's `reserve` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptByteCensus {
    /// The single window HC-stream arena.
    pub(crate) hc_streams: u64,
    /// The dispatch crate's route storage (saved prepared input, top-k, route
    /// maps, widened accumulator, saved shared activation).
    pub(crate) route_storage: u64,
    pub(crate) a4_sidecars: u64,
    /// Active ring pages / lookahead pages / pointer tables of the page pool.
    pub(crate) ring: u64,
    pub(crate) lookahead: u64,
    pub(crate) tables: u64,
    pub(crate) boundary_state: u64,
    /// Rows of the largest window and of the largest mixer tile.
    pub(crate) window_rows: usize,
    pub(crate) tile_rows: usize,
}

impl PromptByteCensus {
    /// Sum of every device term but the forward's own tile scratch.
    pub(crate) fn total(&self) -> Option<u64> {
        [
            self.hc_streams,
            self.route_storage,
            self.a4_sidecars,
            self.ring,
            self.lookahead,
            self.tables,
            self.boundary_state,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    }
}

/// A request S1 will run, fully decided before any mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen4PromptPlan {
    pub(crate) entry_pos: usize,
    pub(crate) rows: usize,
    pub(crate) tile_rows: usize,
    pub(crate) tiles: Vec<TilePlan>,
    pub(crate) windows: Vec<WindowPlan>,
    /// Session boundaries strictly inside the request (absolute, ascending).
    pub(crate) cuts: Vec<usize>,
    pub(crate) census: PromptByteCensus,
    /// Largest sum of K1 (few-row) tile rows any one window holds.
    pub(crate) k1_rows_max: usize,
}

impl Qwen4PromptPlan {
    /// Chronological exit row boundaries relative to the entry position: the
    /// end row of every tile (what the incumbent's chunk loop would exit at).
    pub(crate) fn launch_cuts(&self) -> Vec<usize> {
        self.tiles
            .iter()
            .map(|tile| tile.prompt_row + tile.rows)
            .collect()
    }

    pub(crate) fn window_tiles(&self, window: &WindowPlan) -> &[TilePlan] {
        &self.tiles[window.tiles.clone()]
    }

    /// Distinct `(rows, route_rows, bf16)` launch signatures in the plan, in
    /// first-use order: what the pre-mutation admission walks.
    pub(crate) fn signatures(&self) -> Vec<(usize, TileRouting)> {
        let mut out: Vec<(usize, TileRouting)> = Vec::new();
        for tile in &self.tiles {
            if !out.iter().any(|(rows, routing)| *rows == tile.rows && *routing == tile.routing) {
                out.push((tile.rows, tile.routing));
            }
        }
        out
    }
}

/// Inputs of [`plan_prompt`]: all pure data. `routing` is the production
/// per-tile routing decision (a closure over the `Gpu` scope at runtime, a
/// model in CPU tests).
pub(crate) struct PromptPlanInputs<'a> {
    pub(crate) profile: &'a SlowlinkProfile,
    pub(crate) entry_pos: usize,
    pub(crate) rows: usize,
    /// Session boundaries strictly inside `(entry_pos, entry_pos + rows)`,
    /// ascending.
    pub(crate) cuts: &'a [usize],
    pub(crate) routing: &'a dyn Fn(usize) -> TileRouting,
    /// Largest routing row count of a few-row (K1, decode-exact) route.
    pub(crate) few_row_limit: usize,
}

/// Why a well-formed request stays on the incumbent route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlanBypass {
    /// A window's K1 tail rows exceed what the route storage was sized for.
    K1TailRows { window: usize, rows: usize, bound: usize },
}

impl std::fmt::Display for PlanBypass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::K1TailRows { window, rows, bound } => write!(
                f,
                "window {window} holds {rows} few-row tail rows, above the {bound} the route storage admits"
            ),
        }
    }
}

/// The plan of one request, or why it is the incumbent's. `Err` is a
/// malformed input (a bug), never a verdict.
pub(crate) fn plan_prompt_verdict(
    inputs: &PromptPlanInputs<'_>,
) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
    let profile = inputs.profile;
    let (tile_rows, window_rows) = (profile.tile_rows, profile.window_rows);
    if inputs.rows == 0 {
        return Err(invalid("slow-link plan has no rows"));
    }
    if tile_rows == 0 || window_rows < tile_rows {
        return Err(invalid(format!(
            "slow-link plan tile {tile_rows} does not fit window {window_rows}"
        )));
    }
    let end_position = inputs
        .entry_pos
        .checked_add(inputs.rows)
        .ok_or_else(|| invalid("slow-link plan end position overflows"))?;
    let mut previous = inputs.entry_pos;
    for &cut in inputs.cuts {
        if cut <= previous || cut >= end_position {
            return Err(invalid(format!(
                "slow-link cut {cut} is not strictly inside ({previous}, {end_position}) in ascending order"
            )));
        }
        previous = cut;
    }

    // Chronology: the incumbent splits at every boundary, then tiles each
    // segment by the tile size. A tile therefore never straddles a cut.
    struct RawTile {
        position: usize,
        rows: usize,
        ends_at_cut: bool,
    }
    let mut raw: Vec<RawTile> = Vec::new();
    let mut position = inputs.entry_pos;
    for (index, &boundary) in inputs
        .cuts
        .iter()
        .chain(std::iter::once(&end_position))
        .enumerate()
    {
        let is_cut = index < inputs.cuts.len();
        while position < boundary {
            let rows = (boundary - position).min(tile_rows);
            raw.push(RawTile {
                position,
                rows,
                ends_at_cut: is_cut && position + rows == boundary,
            });
            position += rows;
        }
    }

    // Windows: contiguous tiles whose rows fit one window and whose interior
    // cuts fit the boundary snapshots. A window ends at a tile end, so every
    // tile keeps the signature the chunk-major run would give it.
    let mut windows: Vec<WindowPlan> = Vec::new();
    let mut tiles: Vec<TilePlan> = Vec::with_capacity(raw.len());
    let mut window_start_tile = 0usize;
    let mut window_rows_used = 0usize;
    let mut interior_cuts = 0usize;
    let mut prompt_row = 0usize;
    let mut window_prompt_row = 0usize;
    let mut window_position = inputs.entry_pos;
    let mut window_cuts: Vec<usize> = Vec::new();
    let mut k1_rows = 0usize;
    let mut k1_rows_max = 0usize;
    let close = |windows: &mut Vec<WindowPlan>,
                 tiles: &mut Vec<TilePlan>,
                 start_tile: usize,
                 rows: usize,
                 prompt_row: usize,
                 position: usize,
                 cuts: Vec<usize>| {
        let index = windows.len();
        for tile in tiles[start_tile..].iter_mut() {
            tile.window = index;
        }
        if let Some(last) = tiles[start_tile..].last_mut() {
            last.last_in_window = true;
            // The window END boundary is the transaction commit's, not a
            // strictly-inside cut.
            last.ends_at_cut = None;
        }
        windows.push(WindowPlan {
            index,
            prompt_row,
            rows,
            position,
            tiles: start_tile..tiles.len(),
            cuts,
        });
    };
    for tile in &raw {
        let previous_ended_at_cut = tiles.last().is_some_and(|last| last.ends_at_cut.is_some());
        let would_rows = window_rows_used + tile.rows;
        let would_cuts = interior_cuts + usize::from(previous_ended_at_cut && window_rows_used > 0);
        if window_rows_used > 0
            && (would_rows > window_rows || would_cuts > S1_MAX_BOUNDARY_SNAPSHOTS)
        {
            let tail = tiles.last().map(|last| last.position + last.rows);
            debug_assert_eq!(tail, Some(window_position + window_rows_used));
            close(
                &mut windows,
                &mut tiles,
                window_start_tile,
                window_rows_used,
                window_prompt_row,
                window_position,
                std::mem::take(&mut window_cuts),
            );
            if k1_rows > S1_K1_TAIL_ROWS {
                return Ok(Err(PlanBypass::K1TailRows {
                    window: windows.len() - 1,
                    rows: k1_rows,
                    bound: S1_K1_TAIL_ROWS,
                }));
            }
            k1_rows_max = k1_rows_max.max(k1_rows);
            k1_rows = 0;
            window_start_tile = tiles.len();
            window_rows_used = 0;
            interior_cuts = 0;
            window_prompt_row = prompt_row;
            window_position = tile.position;
        }
        // The previous tile's cut is interior to this window only when this
        // tile joins it.
        if window_rows_used > 0 {
            if let Some(last) = tiles.last() {
                if last.ends_at_cut.is_some() {
                    window_cuts.push(last.position + last.rows);
                    interior_cuts += 1;
                }
            }
        }
        let routing = (inputs.routing)(tile.rows);
        if routing.route_rows <= inputs.few_row_limit {
            k1_rows += tile.rows;
        }
        let ends_at_cut = tile.ends_at_cut.then_some(window_cuts.len());
        tiles.push(TilePlan {
            index: tiles.len(),
            window: 0,
            prompt_row,
            window_row: window_rows_used,
            rows: tile.rows,
            position: tile.position,
            routing,
            ends_at_cut,
            last_in_window: false,
        });
        window_rows_used += tile.rows;
        prompt_row += tile.rows;
    }
    close(
        &mut windows,
        &mut tiles,
        window_start_tile,
        window_rows_used,
        window_prompt_row,
        window_position,
        std::mem::take(&mut window_cuts),
    );
    if k1_rows > S1_K1_TAIL_ROWS {
        return Ok(Err(PlanBypass::K1TailRows {
            window: windows.len() - 1,
            rows: k1_rows,
            bound: S1_K1_TAIL_ROWS,
        }));
    }
    k1_rows_max = k1_rows_max.max(k1_rows);

    let census = census_for(profile, &windows)?;
    Ok(Ok(Qwen4PromptPlan {
        entry_pos: inputs.entry_pos,
        rows: inputs.rows,
        tile_rows,
        tiles,
        windows,
        cuts: inputs.cuts.to_vec(),
        census,
        k1_rows_max,
    }))
}

/// [`plan_prompt_verdict`] with the verdict folded to `Option`: `Ok(None)` is
/// the incumbent, chosen before any mutation or S1 allocation.
pub(crate) fn plan_prompt(inputs: &PromptPlanInputs<'_>) -> R<Option<Qwen4PromptPlan>> {
    Ok(plan_prompt_verdict(inputs)?.ok())
}

/// The census of a plan: the profile's layouts, checked against the plan's
/// windows. The allocation reads the same layouts, so census == allocation.
fn census_for(profile: &SlowlinkProfile, windows: &[WindowPlan]) -> R<PromptByteCensus> {
    let hc_streams = profile.arena.span(ArenaField::HcStreams);
    let widest = windows.iter().map(|window| window.rows).max().unwrap_or(0);
    let wide = profile.geometry.hc_count * profile.geometry.hidden;
    let need = (widest as u64)
        .checked_mul(wide as u64)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid("slow-link window stream bytes overflow"))?;
    if need > hc_streams.bytes || widest > profile.window_rows {
        return Err(invalid(format!(
            "slow-link window of {widest} rows needs {need} stream bytes, the admitted arena holds {} ({} rows)",
            hc_streams.bytes, profile.window_rows
        )));
    }
    let spec = profile.pool_spec().map_err(invalid)?;
    let reserve = spec
        .reserve()
        .ok_or_else(|| invalid("slow-link page pool reserve overflows"))?;
    let page = reserve.page_bytes as u64;
    Ok(PromptByteCensus {
        hc_streams: profile.arena.total_bytes,
        route_storage: profile.census.route_storage,
        a4_sidecars: profile.census.a4_sidecars,
        ring: page * spec.active_slots as u64,
        lookahead: page * spec.lookahead_pages as u64,
        tables: reserve.tables as u64,
        boundary_state: profile.census.boundary_state,
        window_rows: widest,
        tile_rows: profile.tile_rows,
    })
}

/// The byte census of a plan ([`Qwen4PromptPlan::census`]) re-derived and
/// checked against the plan.
pub(crate) fn prompt_bytes(plan: &Qwen4PromptPlan) -> R<PromptByteCensus> {
    if plan.windows.iter().any(|window| window.rows > plan.census.window_rows) {
        return Err(invalid("slow-link plan window exceeds its census"));
    }
    Ok(plan.census)
}

/// Binding ranges of a host layer's expert groups: ascending, contiguous and
/// covering `[0, num_experts)`, one per group (a pruned expert between two
/// groups belongs to the earlier range; nothing routes to it). Index `i` is
/// the group at position `i` of the layer's schedule.
pub(crate) fn group_binding_ranges(groups: &[ExpertGroupPlan], num_experts: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(groups.len());
    let mut lo = 0usize;
    for (index, group) in groups.iter().enumerate() {
        let hi = if index + 1 == groups.len() {
            num_experts
        } else {
            group.experts.last().map_or(lo, |last| last + 1)
        };
        out.push((lo, hi));
        lo = hi;
    }
    out
}

// ── receipt ──────────────────────────────────────────────────────────

fn mode_label(mode: SlowlinkMode) -> &'static str {
    match mode {
        SlowlinkMode::Off => "env0",
        SlowlinkMode::Force => "env1",
        SlowlinkMode::Auto => "auto",
    }
}

fn zero_reserve() -> Value {
    json!({"tile": 0u64, "arena": 0u64, "ring": 0u64, "lookahead": 0u64, "tables": 0u64, "total": 0u64})
}

/// The incumbent verdict receipt: the selector ran and chose `incumbent`.
pub(crate) fn bypass_receipt(
    mode: SlowlinkMode,
    reason: &str,
    entry_pos: usize,
    rows: usize,
) -> Value {
    json!({
        "schema": RECEIPT_SCHEMA,
        "route": "incumbent",
        "selected_by": mode_label(mode),
        "bypass_reason": reason,
        "entry_pos": entry_pos,
        "prompt_rows": rows,
        "windows": [],
        "launch_cuts": [],
        "tile_rows": 0usize,
        "retained_scratch_rows": 0usize,
        "sweeps": 0usize,
        "resident_layers": 0usize,
        "host_layers": 0usize,
        "host_expert_bytes_total": 0u64,
        "submitted_bytes": 0u64,
        "useful_bytes": 0u64,
        "unused_prefetched_bytes": 0u64,
        "reserve": zero_reserve(),
        "spans_ms": {"wall": 0.0, "copy": 0.0, "compute": 0.0, "route_barrier": 0.0, "nonexpert": 0.0},
        "overlap": {
            "cross_layer_intersections": 0u64,
            "cross_layer_overlap_ms": 0.0,
            "copy_idle_ms": 0.0,
            "page_full_stalls": 0u64,
            "copy_starved_stalls": 0u64,
        },
        "peak_vram_bytes": 0u64,
    })
}

/// Measured quantities of one executed window, summed into the receipt.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WindowReport {
    pub(crate) stats: PanelStats,
    pub(crate) overlap: OverlapSummary,
    pub(crate) useful_bytes: u64,
    pub(crate) sweep_bytes: u64,
    pub(crate) route_barrier_ms: f64,
    pub(crate) wall_ms: f64,
    pub(crate) nonexpert_ms: f64,
}

/// Everything the layer-major receipt reports.
pub(crate) struct ReceiptInputs<'a> {
    pub(crate) mode: SlowlinkMode,
    pub(crate) plan: &'a Qwen4PromptPlan,
    pub(crate) profile: &'a SlowlinkProfile,
    pub(crate) resident_layers: usize,
    pub(crate) host_layers: usize,
    pub(crate) tile_scratch_bytes: u64,
    pub(crate) reports: &'a [WindowReport],
    pub(crate) peak_vram_bytes: u64,
    /// `"ok"` or `"aborted: ..."`.
    pub(crate) status: String,
}

/// The layer-major receipt: measured fields only. A window that never ran
/// contributes zeros and the status says so; nothing is inferred.
pub(crate) fn layer_major_receipt(inputs: &ReceiptInputs<'_>) -> Value {
    let plan = inputs.plan;
    let sum = |f: &dyn Fn(&WindowReport) -> f64| inputs.reports.iter().map(f).sum::<f64>();
    let sum_u = |f: &dyn Fn(&WindowReport) -> u64| inputs.reports.iter().map(f).sum::<u64>();
    let census = &plan.census;
    let reserve_total = census
        .total()
        .and_then(|total| total.checked_add(inputs.tile_scratch_bytes))
        .unwrap_or(u64::MAX);
    let host_expert_bytes_total = inputs
        .reports
        .first()
        .map_or(0, |report| report.sweep_bytes);
    json!({
        "schema": RECEIPT_SCHEMA,
        "route": "layer-major",
        "selected_by": mode_label(inputs.mode),
        "bypass_reason": Value::Null,
        "status": inputs.status,
        "entry_pos": plan.entry_pos,
        "prompt_rows": plan.rows,
        "windows": plan.windows.iter().map(|w| json!({"start": w.prompt_row, "rows": w.rows})).collect::<Vec<_>>(),
        "launch_cuts": plan.launch_cuts(),
        "tile_rows": plan.tile_rows,
        "retained_scratch_rows": plan.tile_rows,
        "sweeps": inputs.reports.len(),
        "resident_layers": inputs.resident_layers,
        "host_layers": inputs.host_layers,
        "host_expert_bytes_total": host_expert_bytes_total,
        "submitted_bytes": sum_u(&|r| r.stats.submitted_payload_bytes),
        "useful_bytes": sum_u(&|r| r.useful_bytes),
        "unused_prefetched_bytes": sum_u(&|r| r.stats.discarded_payload_bytes),
        "reserve": {
            "tile": inputs.tile_scratch_bytes,
            "arena": census.hc_streams + census.route_storage + census.a4_sidecars,
            "ring": census.ring,
            "lookahead": census.lookahead,
            "tables": census.tables,
            "total": reserve_total,
        },
        "spans_ms": {
            "wall": sum(&|r| r.wall_ms),
            "copy": sum(&|r| r.overlap.copy_busy_ms),
            "compute": sum(&|r| r.overlap.compute_busy_ms),
            "route_barrier": sum(&|r| r.route_barrier_ms),
            "nonexpert": sum(&|r| r.nonexpert_ms),
        },
        "overlap": {
            "cross_layer_intersections": sum_u(&|r| r.overlap.cross_layer_intersections),
            "cross_layer_overlap_ms": sum(&|r| r.overlap.cross_layer_overlap_ms),
            "copy_idle_ms": sum(&|r| r.overlap.copy_idle_ms),
            "page_full_stalls": sum_u(&|r| r.stats.page_full_stalls),
            "copy_starved_stalls": sum_u(&|r| r.stats.copy_starved_stalls),
        },
        "timeline_overflow": sum_u(&|r| r.stats.timeline_overflow + r.stats.layer_mark_overflow),
        "peak_vram_bytes": inputs.peak_vram_bytes,
    })
}

// ── request output ───────────────────────────────────────────────────

/// One mixer tile's final-layer hidden rows, handed to a [`PromptOutput::Capture`]
/// sink after the tile's last layer finished.
pub(crate) struct PromptTile<'a> {
    /// Absolute position of the tile's first row.
    pub(crate) start: usize,
    pub(crate) rows: usize,
    /// F32 `rows * hc_count * hidden` HC streams of the tile, in the bundle's
    /// own `spec_hidden` rows `[0, rows)`; valid during the call.
    pub(crate) wide: &'a GpuTensor,
    /// The last tile of the window being run.
    pub(crate) last: bool,
}

/// The sink of a bounded request-owned capture (the MTP target fill).
pub(crate) type CaptureSink<'a> =
    dyn FnMut(&mut Gpu, &mut Qwen4Bundle, PromptTile<'_>) -> Result<(), String> + 'a;

/// What the request returns. An all-row logits capture and any caller-owned
/// external capture stay on the incumbent route.
pub(crate) enum PromptOutput<'a> {
    /// State only.
    Silent,
    /// The final row's logits (and its greedy id when `top1` is given).
    Final {
        logits: &'a GpuTensor,
        top1: Option<&'a GpuTensor>,
    },
    /// Bounded per-tile wide-hidden capture plus the final row's logits/top-1
    /// (the speculative head seed).
    Capture {
        sink: &'a mut CaptureSink<'a>,
        head: Option<(&'a GpuTensor, &'a GpuTensor)>,
    },
}

impl PromptOutput<'_> {
    fn capture_kind(&self) -> CaptureKind {
        match self {
            Self::Silent | Self::Final { .. } => CaptureKind::None,
            Self::Capture { .. } => CaptureKind::Bounded,
        }
    }

    fn head(&self) -> Option<(&GpuTensor, Option<&GpuTensor>)> {
        match self {
            Self::Silent => None,
            Self::Final { logits, top1 } => Some((logits, *top1)),
            Self::Capture { head, .. } => head.as_ref().map(|(logits, top1)| (*logits, Some(*top1))),
        }
    }
}
