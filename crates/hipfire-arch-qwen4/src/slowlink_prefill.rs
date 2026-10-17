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
    ExpertGroupPlan, ExpertPanelRing, OverlapSummary, PanelBindingParts, PanelSchedule, PanelStats,
    PanelTicket,
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
    /// Offset (F32 elements) of the tile's HC-stream region in the window
    /// arena. Each tile's region is sized by its own tag (BF16 rows take half
    /// the bytes of F32 rows), so the arena holds exactly the tagged spans.
    pub(crate) stream_offset: usize,
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
    #[cfg_attr(not(test), allow(dead_code))]
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
    /// Packed bytes of the window's tagged HC-stream regions.
    pub(crate) stream_bytes: u64,
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
    /// The window-wide stable-grouping scatter scratch (`qwen4_moe_group_layout`
    /// at the window's slots), charged beside the A4 sidecars.
    pub(crate) group_scatter: u64,
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
            self.group_scatter,
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
            if !out
                .iter()
                .any(|(rows, routing)| *rows == tile.rows && *routing == tile.routing)
            {
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
    K1TailRows {
        window: usize,
        rows: usize,
        bound: usize,
    },
    /// A window's tagged HC-stream regions (plus the one-tile view slack) do
    /// not fit the admitted arena.
    StreamArena { window: usize, need: u64, have: u64 },
}

impl std::fmt::Display for PlanBypass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::K1TailRows { window, rows, bound } => write!(
                f,
                "window {window} holds {rows} few-row tail rows, above the {bound} the route storage admits"
            ),
            Self::StreamArena { window, need, have } => write!(
                f,
                "window {window} needs {need} bytes of tagged HC streams, the admitted arena holds {have}"
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
            stream_bytes: 0,
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
            stream_offset: 0,
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

    // The window's HC-stream regions: each tile takes the bytes of its own tag
    // (BF16 = 2 bytes per element, F32 = 4), packed in tile order. The F32-typed
    // view every stream op validates reaches past a BF16 region by up to its own
    // size, so the arena also carries one tile of BF16 slack.
    let wide = profile.geometry.hc_count * profile.geometry.hidden;
    if wide % 8 != 0 {
        return Err(invalid(format!(
            "slow-link HC width {wide} is not a multiple of 8 elements"
        )));
    }
    let slack_bytes = (tile_rows * wide * 2) as u64;
    for window in windows.iter_mut() {
        let mut elements = 0usize;
        for tile in tiles[window.tiles.clone()].iter_mut() {
            tile.stream_offset = elements;
            let bytes_per_element = if tile.routing.bf16_streams { 2 } else { 4 };
            elements += tile.rows * wide * bytes_per_element / 4;
        }
        window.stream_bytes = elements as u64 * 4;
        let need = window.stream_bytes + slack_bytes;
        if need > profile.arena.total_bytes {
            return Ok(Err(PlanBypass::StreamArena {
                window: window.index,
                need,
                have: profile.arena.total_bytes,
            }));
        }
    }
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
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn plan_prompt(inputs: &PromptPlanInputs<'_>) -> R<Option<Qwen4PromptPlan>> {
    Ok(plan_prompt_verdict(inputs)?.ok())
}

/// The census of a plan: the profile's layouts, checked against the plan's
/// windows. The allocation reads the same layouts, so census == allocation.
fn census_for(profile: &SlowlinkProfile, windows: &[WindowPlan]) -> R<PromptByteCensus> {
    let widest = windows.iter().map(|window| window.rows).max().unwrap_or(0);
    if widest > profile.window_rows {
        return Err(invalid(format!(
            "slow-link window of {widest} rows exceeds the admitted {} rows",
            profile.window_rows
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
        group_scatter: profile.census.group_scatter,
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
    if plan
        .windows
        .iter()
        .any(|window| window.rows > plan.census.window_rows)
    {
        return Err(invalid("slow-link plan window exceeds its census"));
    }
    Ok(plan.census)
}

/// Binding ranges of a host layer's expert groups: ascending, contiguous and
/// covering `[0, num_experts)`, one per group (a pruned expert between two
/// groups belongs to the earlier range; nothing routes to it). Index `i` is
/// the group at position `i` of the layer's schedule.
pub(crate) fn group_binding_ranges(
    groups: &[ExpertGroupPlan],
    num_experts: usize,
) -> Vec<(usize, usize)> {
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

impl WindowReport {
    /// Submitted payload bytes the window's routed rows never read (the
    /// untouched experts inside acquired groups plus any group never acquired):
    /// `submitted - useful`. `None` when the useful count exceeds the
    /// submission, a measurement that cannot be true.
    pub(crate) fn unused_bytes(&self) -> Option<u64> {
        self.stats
            .submitted_payload_bytes
            .checked_sub(self.useful_bytes)
    }
}

/// Everything the layer-major receipt reports.
pub(crate) struct ReceiptInputs<'a> {
    pub(crate) mode: SlowlinkMode,
    pub(crate) plan: &'a Qwen4PromptPlan,
    pub(crate) resident_layers: usize,
    pub(crate) host_layers: usize,
    pub(crate) tile_scratch_bytes: u64,
    pub(crate) reports: &'a [WindowReport],
    /// Device bytes in use (`total - free`) from ONE `hipMemGetInfo` sample
    /// taken after the request; not a continuous peak. `None` when the query
    /// failed. Reported under the frozen key `peak_vram_bytes`.
    pub(crate) peak_vram_bytes: Option<u64>,
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
        "session_cuts": plan.cuts,
        "tile_rows": plan.tile_rows,
        "retained_scratch_rows": census.tile_rows,
        "sweeps": inputs.reports.len(),
        "resident_layers": inputs.resident_layers,
        "host_layers": inputs.host_layers,
        "host_expert_bytes_total": host_expert_bytes_total,
        "submitted_bytes": sum_u(&|r| r.stats.submitted_payload_bytes),
        "useful_bytes": sum_u(&|r| r.useful_bytes),
        // Submitted payload no routed row read: the untouched experts inside
        // acquired groups and the groups never acquired. `null` (missing) when a
        // window's useful bytes exceed its submission, never an underflow.
        "unused_prefetched_bytes": inputs
            .reports
            .iter()
            .try_fold(0u64, |acc, r| acc.checked_add(r.unused_bytes()?)),
        "discarded_group_payload_bytes": sum_u(&|r| r.stats.discarded_payload_bytes),
        "reserve": {
            "tile": inputs.tile_scratch_bytes,
            "arena": census.hc_streams + census.route_storage + census.a4_sidecars + census.group_scatter,
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
        // Frozen key name. The value is a single post-request device-memory
        // sample (`vram_measurement` names the method); `null` if unavailable.
        "peak_vram_bytes": inputs.peak_vram_bytes,
        "vram_measurement": {
            "method": "hipMemGetInfo_post_request_sample",
            "continuous_peak": false,
        },
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
        /// The drafter's entry `pending` row (the target hidden of the last
        /// already-prefilled position), saved by the selector once the
        /// request is selected.
        pending: Option<&'a GpuTensor>,
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
            Self::Final { logits, top1 } => Some((*logits, *top1)),
            Self::Capture { head, .. } => {
                head.as_ref().map(|(logits, top1)| (*logits, Some(*top1)))
            }
        }
    }
}

// ── errors ───────────────────────────────────────────────────────────

fn bundle_err(what: &'static str) -> impl Fn(crate::bundle::BundleError) -> Qwen4GpuForwardError {
    move |error| invalid(format!("slow-link {what}: {error}"))
}

fn dsp_err(
    what: &'static str,
) -> impl Fn(hipfire_dispatch::types::DispatchError) -> Qwen4GpuForwardError {
    move |error| Qwen4GpuForwardError::Dispatch(format!("slow-link {what}: {error:?}"))
}

// ── runtime ──────────────────────────────────────────────────────────

/// What a forward holds of S1: the admitted profile (set before attach, never
/// mutated by a request) and, once the first qualified request ran, the device
/// resources the profile's census charged.
#[derive(Default)]
pub(crate) struct SlowlinkSlot {
    pub(crate) profile: Option<SlowlinkProfile>,
    runtime: Option<SlowlinkRuntime>,
    /// Sticky: why the resources cannot be built on this load.
    unavailable: Option<String>,
}

impl SlowlinkSlot {
    /// Drain and free the ring, route region and HC-stream arena.
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> R<()> {
        match self.runtime {
            Some(runtime) => runtime.free_gpu(gpu),
            None => Ok(()),
        }
    }
}

/// The S1 device resources of one forward, allocated once from the profile:
/// exactly what [`crate::slowlink_admission::SlowlinkCensus`] charged.
struct SlowlinkRuntime {
    ring: ExpertPanelRing,
    schedule: PanelSchedule,
    /// One region sized for the widest window; every window carves its own
    /// route storage out of it ([`dsp::PromptRouteStorage::from_region`]).
    route_region: GpuTensor,
    /// The window HC-stream arena (F32; a BF16 tile uses its first half).
    streams: GpuTensor,
    /// Trunk layer -> its routed experts live in host memory.
    host_layer: Vec<bool>,
    host_layers: usize,
    resident_layers: usize,
    tile_scratch_bytes: u64,
}

impl SlowlinkRuntime {
    fn free_gpu(self, gpu: &mut Gpu) -> R<()> {
        let Self {
            ring,
            schedule,
            route_region,
            streams,
            ..
        } = self;
        let mut first: Option<Qwen4GpuForwardError> = None;
        // The ring drains both streams before anything it or the route region
        // feeds is released; a failed drain leaks rather than frees.
        let drained = ring.free_gpu(gpu);
        drop(schedule);
        match drained {
            Ok(()) => {
                for tensor in [route_region, streams] {
                    if let Err(error) = gpu.free_tensor(tensor) {
                        first.get_or_insert(error.into());
                    }
                }
            }
            Err(error) => first = Some(error),
        }
        first.map_or(Ok(()), Err)
    }
}

/// Build the runtime of an admitted profile on this forward. Every failure is
/// a bypass reason (nothing partial stays allocated).
fn build_runtime(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    profile: &SlowlinkProfile,
) -> Result<SlowlinkRuntime, String> {
    let config = &bundle.config;
    let mut host_layer = vec![false; config.num_hidden_layers];
    let mut sources = Vec::new();
    let mut resident_layers = 0usize;
    for layer in forward.band_layers() {
        let Some(runtime) = forward.moe.get(layer).and_then(Option::as_ref) else {
            continue;
        };
        if runtime.s1_num_experts() != profile.geometry.num_experts {
            return Err(format!(
                "layer {layer} holds {} experts, the profile admitted {}",
                runtime.s1_num_experts(),
                profile.geometry.num_experts
            ));
        }
        if runtime.s1_host_mapped() {
            host_layer[layer] = true;
            sources.push(
                runtime
                    .s1_host_sources(&mut *gpu, bundle, layer)
                    .map_err(|error| error.to_string())?,
            );
        } else {
            resident_layers += 1;
        }
    }
    let host_layers = sources.len();
    if host_layers == 0 {
        return Err("no host-resident expert layer to stream".to_string());
    }
    let spec = profile.pool_spec()?;
    if config.num_hidden_layers > spec.timeline_layers {
        return Err(format!(
            "{} layers exceed the {} timeline layer marks the pool admitted",
            config.num_hidden_layers, spec.timeline_layers
        ));
    }
    if host_layers * profile.pages.pages_per_layer > spec.timeline_records {
        return Err(format!(
            "{host_layers} host layers exceed the {} timeline records the pool admitted",
            spec.timeline_records
        ));
    }
    let layout =
        dsp::PromptRouteLayout::new(&profile.route_dims()).map_err(|error| format!("{error:?}"))?;
    if layout.total_bytes() as u64 != profile.census.route_storage {
        return Err(format!(
            "route storage layout is {} bytes, the census charged {}",
            layout.total_bytes(),
            profile.census.route_storage
        ));
    }
    let stream_bytes = profile.arena.span(ArenaField::HcStreams).bytes;
    if profile.arena.total_bytes % 4 != 0 || stream_bytes > profile.arena.total_bytes {
        return Err("HC-stream arena bytes are not whole F32 elements".to_string());
    }
    let schedule =
        PanelSchedule::new(sources, EXPERTS_PER_PAGE).map_err(|error| error.to_string())?;
    let ring = ExpertPanelRing::new(gpu, spec).map_err(|error| error.to_string())?;
    let route_region = match gpu.alloc_tensor(&[layout.total_bytes()], DType::Raw) {
        Ok(tensor) => tensor,
        Err(error) => {
            let _ = ring.free_gpu(gpu);
            return Err(format!("route region allocation failed: {error}"));
        }
    };
    let streams = match gpu.alloc_tensor(&[(profile.arena.total_bytes / 4) as usize], DType::F32) {
        Ok(tensor) => tensor,
        Err(error) => {
            let _ = ring.free_gpu(gpu);
            let _ = gpu.free_tensor(route_region);
            return Err(format!("HC-stream arena allocation failed: {error}"));
        }
    };
    Ok(SlowlinkRuntime {
        ring,
        schedule,
        route_region,
        streams,
        host_layer,
        host_layers,
        resident_layers,
        tile_scratch_bytes: forward.slowlink_tile_scratch_bytes(),
    })
}

// ── geometry of one request ──────────────────────────────────────────

/// Per-request constants every tile of the request reads.
struct RunGeo {
    config: Qwen4Config,
    dims: Qwen4ProgramDims,
    wide: usize,
    hidden: usize,
    few_row_limit: usize,
    ple_layer: Option<usize>,
    ple_heads: usize,
    ple_width: usize,
    band: Range<usize>,
    last_layer: usize,
    num_experts: usize,
}

impl RunGeo {
    fn new(
        forward: &Qwen4GpuForward,
        bundle: &Qwen4Bundle,
        few_row_limit: usize,
        num_experts: usize,
    ) -> R<Self> {
        let config = bundle.config.clone();
        let dims = program_dims(&config);
        let band = forward.band_layers();
        let ple_layer = if config.ple.is_some() {
            let index = config
                .ple_layer_ids
                .first()
                .and_then(|id| id.checked_sub(1))
                .ok_or_else(|| invalid("PLE layer id missing"))?;
            if index >= config.num_hidden_layers {
                return Err(invalid("PLE layer id is outside the trunk"));
            }
            Some(index)
        } else {
            None
        };
        let (ple_heads, ple_width) = config
            .ple
            .as_ref()
            .map_or((0, 0), |ple| (ple.head_count, ple.row_width));
        if band.is_empty() {
            return Err(invalid("slow-link request has no trunk layers"));
        }
        Ok(Self {
            wide: dims.wide(),
            hidden: config.hidden_size,
            few_row_limit,
            ple_layer,
            ple_heads,
            ple_width,
            last_layer: band.end - 1,
            band,
            dims,
            config,
            num_experts,
        })
    }
}

/// The routing decision of a `rows`-row tile under the tile's own scope: what
/// the incumbent decides for the chunk of the same rows.
fn tile_routing(forward: &Qwen4GpuForward, gpu: &mut Gpu, rows: usize) -> TileRouting {
    let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
    let routing = TileRouting {
        route_rows: gpu.route_rows(rows),
        bf16_streams: gpu.qwen4_bf16_streams(rows),
    };
    Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
    routing
}

// ── PLE ordered history ──────────────────────────────────────────────

/// The private PLE cursor of a request: the token history each tile of the
/// window starts from (entry history advanced by the earlier tiles' tokens,
/// never the bundle's published history, which only the window commit moves)
/// and the retained lookahead ticket of the next tile's rows.
struct PleCursor {
    retained: Option<RowTicket>,
    histories: Vec<PleHistory>,
}

impl PleCursor {
    /// The histories of `tiles` over `window_tokens`, from `entry`.
    fn histories_for(
        entry: PleHistory,
        tiles: &[TilePlan],
        window_tokens: &[u32],
    ) -> Vec<PleHistory> {
        let mut history = entry;
        let mut out = Vec::with_capacity(tiles.len());
        for tile in tiles {
            out.push(history);
            for &token in &window_tokens[tile.window_row..tile.window_row + tile.rows] {
                history.push(token);
            }
        }
        out
    }

    /// Fetch, stage, upload and widen the PLE rows of tile `ti`, then start
    /// the next tile's fetch so its worker overlaps this tile's compute.
    #[allow(clippy::too_many_arguments)]
    fn stage_tile(
        &mut self,
        owner: &crate::bundle::Qwen4BundlePle,
        host: &mut crate::ple_stage::PleHostStage,
        gpu: &mut Gpu,
        w: &WinCtx<'_>,
        ti: usize,
        staged: &GpuTensor,
        ple_rows: &GpuTensor,
        rows_f16: bool,
        heads: usize,
        width: usize,
    ) -> R<()> {
        let tile = &w.tiles[ti];
        let ids = self.histories[ti].row_ids(&owner.metadata, w.tile_tokens(ti));
        let mut fetch = RowFetch::begin_or_adopt(&owner.rows, self.retained.take(), ids)
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        let mode = if ple_async_upload(host.enabled(), false, true, gpu.graphs.capture_mode) {
            PleUploadMode::Async
        } else {
            PleUploadMode::Blocking
        };
        stage_ple_rows(
            gpu, host, &mut fetch, staged, ple_rows, rows_f16, tile.rows, heads, width, mode,
        )?;
        fetch.complete();
        if ti + 1 < w.tiles.len() {
            let next = w.tile_tokens(ti + 1);
            let fit = owner.rows.max_rows_per_prefetch() / heads.max(1);
            let ahead = &next[..next.len().min(fit)];
            self.retained = owner
                .rows
                .prefetch(
                    owner.rows.current_epoch(),
                    self.histories[ti + 1].row_ids(&owner.metadata, ahead),
                )
                .ok();
        }
        Ok(())
    }
}

// ── the group walk (layer-ahead transport driver) ────────────────────

/// What the layer walk needs of the transport: the production ring (HIP) and
/// the CPU model of the lifecycle tests implement it, so one walker is both.
pub(crate) trait GroupRing {
    type Ctx: ?Sized;
    /// Ticket of current-layer `group` (enqueued if still missing).
    fn require(&mut self, ctx: &Self::Ctx, layer: usize, group: u32) -> R<PanelTicket>;
    /// Acquire `ticket`, run `compute` while the lease is held, release the
    /// lease. The page is freed only by [`Self::retire`].
    fn with_lease(
        &mut self,
        ctx: &mut Self::Ctx,
        ticket: PanelTicket,
        compute: &mut dyn FnMut(&mut Self::Ctx, Option<&PanelBindingParts<'_>>) -> R<()>,
    ) -> R<()>;
    fn retire(&mut self, ctx: &Self::Ctx, ticket: PanelTicket) -> R<()>;
    /// Fill freed pages: the current layer's missing groups first, then the
    /// next layer's.
    fn pump(&mut self, ctx: &Self::Ctx) -> R<usize>;
}

/// The production ring and the window's schedule.
struct HipRing<'a> {
    ring: &'a mut ExpertPanelRing,
    schedule: &'a mut PanelSchedule,
}

impl GroupRing for HipRing<'_> {
    type Ctx = Gpu;

    fn require(&mut self, gpu: &Gpu, layer: usize, group: u32) -> R<PanelTicket> {
        self.ring.require(gpu, self.schedule, layer, group)
    }

    fn with_lease(
        &mut self,
        gpu: &mut Gpu,
        ticket: PanelTicket,
        compute: &mut dyn FnMut(&mut Gpu, Option<&PanelBindingParts<'_>>) -> R<()>,
    ) -> R<()> {
        let lease = self.ring.acquire(&*gpu, ticket)?;
        let parts = lease.parts();
        let result = compute(gpu, Some(&parts));
        result
    }

    fn retire(&mut self, gpu: &Gpu, ticket: PanelTicket) -> R<()> {
        self.ring.retire(gpu, ticket)
    }

    fn pump(&mut self, gpu: &Gpu) -> R<usize> {
        self.ring.pump(gpu, self.schedule)
    }
}

/// The Gpu scope of the tile whose rows a dispatch step handles: the dispatch
/// crate brackets every tile-specific launch (a K1 tail cohort, a tile's shared
/// expert) with `enter(row0)`/`leave(row0)` so a tail's routing rows, canonical
/// scope and HC tag never leak into the K2 panels' (tile 0's) frozen arm, and
/// the K2 arm never leaks into a tail.
struct ForwardScopeHook<'a> {
    forward: &'a Qwen4GpuForward,
    tiles: &'a [TilePlan],
    open: Vec<crate::gpu_forward::TileScopes>,
}

impl dsp::TileScopeHook for ForwardScopeHook<'_> {
    fn enter(
        &mut self,
        gpu: &mut Gpu,
        row0: usize,
    ) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let tile = self
            .tiles
            .iter()
            .find(|tile| tile.window_row == row0)
            .ok_or_else(|| {
                hipfire_dispatch::types::DispatchError::Hip(format!(
                    "slow-link scope hook: no tile starts at window row {row0}"
                ))
            })?;
        let (scopes, _) = self.forward.enter_tile_scopes(gpu, tile.rows);
        self.open.push(scopes);
        Ok(())
    }

    fn leave(
        &mut self,
        gpu: &mut Gpu,
        _row0: usize,
    ) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let scopes = self.open.pop().ok_or_else(|| {
            hipfire_dispatch::types::DispatchError::Hip(
                "slow-link scope hook: leave without a matching enter".to_string(),
            )
        })?;
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        Ok(())
    }
}

/// A group of the layer's schedule: whether any panel reads it. A group no
/// panel reads (no K2 row and no K1 expert) is retired unread, never refetched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupWork {
    pub(crate) group: u32,
    pub(crate) bound: bool,
}

/// What a layer walk did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WalkSummary {
    pub(crate) acquired: usize,
    pub(crate) discarded: usize,
}

/// Walk one host layer's groups in ascending order: `require` each, run
/// `compute(index, binding)` for those panels read, `retire` every one (the
/// last reader of a group is the last panel of that group, all cohorts
/// included), and `pump` after each retire so freed pages immediately refill,
/// the current layer first and then the next layer's weights.
pub(crate) fn walk_layer_groups<G: GroupRing>(
    ring: &mut G,
    ctx: &mut G::Ctx,
    layer: usize,
    work: &[GroupWork],
    compute: &mut dyn FnMut(&mut G::Ctx, usize, Option<&PanelBindingParts<'_>>) -> R<()>,
) -> R<WalkSummary> {
    let mut summary = WalkSummary::default();
    for (index, group) in work.iter().enumerate() {
        let ticket = ring.require(&*ctx, layer, group.group)?;
        if group.bound {
            ring.with_lease(ctx, ticket, &mut |ctx, parts| compute(ctx, index, parts))?;
            summary.acquired += 1;
        } else {
            summary.discarded += 1;
        }
        ring.retire(&*ctx, ticket)?;
        ring.pump(&*ctx)?;
    }
    Ok(summary)
}

/// The panels of each group: ranges into the schedule's panel list. The panels
/// of one binding must be contiguous and bindings ascending (the order the
/// ordered combine requires).
pub(crate) fn panels_by_group(
    bindings: impl Iterator<Item = usize>,
    groups: usize,
) -> R<Vec<Range<usize>>> {
    let mut out = vec![0..0; groups];
    let mut previous: Option<usize> = None;
    for (index, binding) in bindings.enumerate() {
        if binding >= groups {
            return Err(invalid(format!(
                "panel {index} names group {binding} of {groups}"
            )));
        }
        if previous.is_some_and(|prev| binding < prev) {
            return Err(invalid("panels are not in ascending binding order"));
        }
        previous = Some(binding);
        let range = &mut out[binding];
        if range.start >= range.end {
            *range = index..index + 1;
        } else if range.end != index {
            return Err(invalid("the panels of one binding are not contiguous"));
        } else {
            range.end = index + 1;
        }
    }
    Ok(out)
}

// ── abort bookkeeping ────────────────────────────────────────────────

/// How far a window got: what its failure must undo.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    /// The window transaction is open (entry snapshot taken).
    pub(crate) txn_open: bool,
    /// The ring started a window and may have copies or readers in flight.
    pub(crate) ring_primed: bool,
}

/// The cleanup of a failed window, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AbortSteps {
    /// Drain both streams first: an in-flight copy or reader must finish before
    /// anything it reads or writes is restored or released.
    pub(crate) drain_ring: bool,
    /// Then restore the entry state and drop the transaction.
    pub(crate) abort_txn: bool,
}

impl Progress {
    pub(crate) fn abort_steps(self) -> AbortSteps {
        AbortSteps {
            drain_ring: self.ring_primed,
            abort_txn: self.txn_open,
        }
    }
}

// ── selection ────────────────────────────────────────────────────────

struct Selection {
    plan: Qwen4PromptPlan,
    profile: SlowlinkProfile,
    geo: RunGeo,
}

/// The full-hidden and final-row head of the request against the forward's
/// scratch, before any mutation (the incumbent's own preflight).
fn validate_head(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    output: &PromptOutput<'_>,
    geo: &RunGeo,
    last_rows: usize,
) -> Result<(), String> {
    let Some((logits, top1)) = output.head() else {
        return Ok(());
    };
    if logits.dtype != DType::F32 || logits.numel() != geo.config.vocab_size {
        return Err(format!(
            "head logits must be F32 with {} elements",
            geo.config.vocab_size
        ));
    }
    if let Some(top1) = top1 {
        if top1.dtype != DType::Raw || top1.buf.size() < std::mem::size_of::<i32>() {
            return Err("head top1 output must be Raw with one i32".to_string());
        }
    }
    let final_hyper =
        crate::gpu_forward::hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)
            .map_err(|error| error.to_string())?;
    let lm_head = crate::gpu_forward::dense_ref(&bundle.weights, &bundle.weights.root.lm_head)
        .map_err(|error| error.to_string())?;
    let router_logits = matrix_view(
        &forward.scratch.router_logits,
        last_rows,
        router_columns(geo.dims),
    )
    .map_err(|error| error.to_string())?;
    let scratch_desc = layer_scratch(&forward.scratch, &router_logits);
    validate_final_hyper(
        geo.dims,
        &final_hyper,
        &forward.scratch.streams,
        &scratch_desc,
        last_rows,
    )
    .map_err(|error| format!("validate final hyper: {error:?}"))?;
    let hidden = view(&forward.scratch.hc_mixed, 0, last_rows * geo.hidden);
    validate_lm_head(&lm_head, &hidden, logits, last_rows, 1)
        .map_err(|error| format!("validate LM head: {error:?}"))
}

/// The pre-mutation dry run of the dispatch admission: seal the first host
/// layer's MoE call for every distinct tile signature of the plan and ask the
/// dispatch crate to admit it.
fn preflight_tiles(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    geo: &RunGeo,
    plan: &Qwen4PromptPlan,
) -> Result<(), String> {
    let layer = geo
        .band
        .clone()
        .find(|&layer| {
            forward
                .moe
                .get(layer)
                .and_then(Option::as_ref)
                .is_some_and(|runtime| runtime.s1_host_mapped())
        })
        .ok_or_else(|| "no host-resident expert layer".to_string())?;
    let mut arms: Vec<([usize; 6], bool)> = Vec::new();
    let mut signatures = plan.signatures();
    // A window of one-row decode tiles only has no prefill tile to anchor the
    // panel scratch: the sweep uses a stand-in prefill tile of the admitted tile
    // rows, admitted here with the rest.
    if plan.windows.iter().any(|window| {
        plan.window_tiles(window)
            .iter()
            .all(|tile| tile.routing.route_rows == 1)
    }) {
        let rows = forward.scratch.max_chunk;
        let routing = tile_routing(forward, gpu, rows);
        if !signatures.iter().any(|(r, known)| {
            *r == rows
                && known.route_rows == routing.route_rows
                && known.bf16_streams == routing.bf16_streams
        }) {
            signatures.push((rows, routing));
        }
    }
    for (rows, routing) in signatures {
        let stub = TilePlan {
            index: 0,
            window: 0,
            prompt_row: 0,
            window_row: 0,
            stream_offset: 0,
            rows,
            position: plan.entry_pos,
            routing,
            ends_at_cut: None,
            last_in_window: false,
        };
        let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
        let result = with_tile_frame(
            forward,
            bundle,
            gpu,
            &forward.scratch.streams,
            geo,
            &stub,
            layer,
            |gpu, frame| {
                let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                    return Err(invalid("a host layer's FFN is not MoE"));
                };
                let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                dsp::PreparedChunkCall::new(&*gpu, &sealed, 0, stub.position)
                    .map(|call| (call.arm_signature(), call.is_k2()))
                    .map_err(dsp_err("admission"))
            },
        );
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        let (arm, is_k2) = result.map_err(|error| {
            format!(
                "tile of {rows} rows (route rows {}): {error}",
                routing.route_rows
            )
        })?;
        arms.push((arm, is_k2));
    }
    // Every tile shares the layer geometry (the first five signature values);
    // the grouped-path flag is the K2 tiles' one frozen arm. A few-row tail (a
    // prefill K1 call or the one-row decode call) keeps its own original arm as
    // a separate cohort, so it is compared on geometry alone and never refused
    // for its arm. One shared rule with the dispatch crate's prepare.
    if !dsp::window_arms_compatible(&arms) {
        return Err(format!(
            "the plan's tiles seal incompatible dispatch arms (signature, K2): {arms:?}"
        ));
    }
    Ok(())
}

/// Decide the request. `Err` is the bypass reason: the incumbent runs, nothing
/// was mutated and no S1 resource was allocated for this request.
fn select(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    tokens: &[u32],
    output: &PromptOutput<'_>,
    mode: SlowlinkMode,
) -> Result<Selection, String> {
    if mode == SlowlinkMode::Off {
        return Err("HIPFIRE_QWEN4_SLOWLINK_PREFILL=0".to_string());
    }
    let Some(profile) = forward.slowlink_profile().copied() else {
        return Err("no admitted slow-link profile".to_string());
    };
    if let Some(reason) = forward.slowlink.unavailable.clone() {
        return Err(reason);
    }
    bundle.slowlink_prompt_eligible().map_err(str::to_string)?;
    if let Some(why) = forward.slowlink_forward_blocker(&bundle.config) {
        return Err(why.to_string());
    }
    let rows = tokens.len();
    let entry = bundle.state.position;
    let end = entry
        .checked_add(rows)
        .ok_or_else(|| "request position overflows".to_string())?;
    if end > bundle.state.max_seq_len {
        return Err(format!(
            "request end {end} exceeds the context capacity {}",
            bundle.state.max_seq_len
        ));
    }
    if tokens
        .iter()
        .any(|&token| token as usize >= bundle.config.vocab_size)
    {
        return Err("a token id is outside the embedding vocabulary".to_string());
    }
    // The forward's scratch must be EXACTLY the admitted tile: a larger one
    // (an incumbent 4096/8192-row chunk the caller asked for) would be retained
    // scratch the profile never charged, and S1 never reroutes after allocation.
    if forward.scratch.max_chunk != profile.tile_rows {
        return Err(format!(
            "forward scratch holds {} rows, the admitted S1 tile is {}",
            forward.scratch.max_chunk, profile.tile_rows
        ));
    }
    qualify_request(
        mode,
        &profile,
        &RequestShape {
            rows,
            entry_pos: entry,
            session_cuts: 0,
            capture: output.capture_kind(),
        },
    )
    .map_err(|bypass: RequestBypass| format!("{bypass:?}"))?;

    let cuts = bundle.prompt_txn_cuts(entry, end);
    let few_row_limit = {
        let (scopes, _) = forward.enter_tile_scopes(gpu, profile.tile_rows);
        let limit = gpu.few_row_limit();
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        limit
    };
    let verdict = {
        let cell = std::cell::RefCell::new(&mut *gpu);
        let routing_forward: &Qwen4GpuForward = &*forward;
        let routing =
            |tile_rows: usize| tile_routing(routing_forward, &mut **cell.borrow_mut(), tile_rows);
        plan_prompt_verdict(&PromptPlanInputs {
            profile: &profile,
            entry_pos: entry,
            rows,
            cuts: &cuts,
            routing: &routing,
            few_row_limit,
        })
        .map_err(|error| error.to_string())?
    };
    let plan = verdict.map_err(|bypass| bypass.to_string())?;
    prompt_bytes(&plan).map_err(|error| error.to_string())?;
    for window in &plan.windows {
        if bundle.prompt_txn_cuts(window.position, window.end_position()) != window.cuts {
            return Err(format!(
                "window {} cuts differ from the transaction's pending boundaries",
                window.index
            ));
        }
    }
    let num_experts = profile.geometry.num_experts;
    let geo = RunGeo::new(forward, bundle, few_row_limit, num_experts)
        .map_err(|error| error.to_string())?;
    let last_rows = plan.tiles.last().map_or(0, |tile| tile.rows);
    validate_head(forward, bundle, output, &geo, last_rows)?;
    for layer in geo.band.clone() {
        if let Some(runtime) = forward.moe.get(layer).and_then(Option::as_ref) {
            if runtime.s1_host_mapped() {
                runtime
                    .s1_layer_check()
                    .map_err(|why| format!("host layer {layer}: {why}"))?;
            }
        }
    }
    preflight_tiles(forward, bundle, gpu, &geo, &plan)?;
    ensure_runtime(forward, bundle, gpu, &profile)?;
    Ok(Selection { plan, profile, geo })
}

/// Allocate the S1 device resources on first use (from the admitted profile,
/// exactly what its census charged); a failure is sticky for this load.
fn ensure_runtime(
    forward: &mut Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    profile: &SlowlinkProfile,
) -> Result<(), String> {
    if forward.slowlink.runtime.is_some() {
        return Ok(());
    }
    match build_runtime(forward, bundle, gpu, profile) {
        Ok(runtime) => {
            forward.slowlink.runtime = Some(runtime);
            Ok(())
        }
        Err(reason) => {
            forward.slowlink.unavailable = Some(reason.clone());
            Err(reason)
        }
    }
}

// ── tile frame ───────────────────────────────────────────────────────

/// Everything the typed operations of one `(tile, layer)` borrow, built under
/// the tile's scope and handed to a callback ([`with_tile_frame`]).
struct TileFrame<'f> {
    env: &'f LayerPhaseEnv<'f>,
    description: &'f crate::program::Qwen4LayerDescription<'f>,
    ctx: &'f DispatchCtx,
    scratch_desc: &'f crate::program::Qwen4LayerScratch<'f>,
    layer: &'f crate::weights::Qwen4LayerWeights,
}

/// Build the layer description, scratch view and [`LayerPhaseEnv`] of `layer`
/// on `tile` (the tile's rows of `streams` are the layer's HC streams) and run
/// `f` on them. The caller has entered the tile's Gpu scope.
#[allow(clippy::too_many_arguments)]
fn with_tile_frame<T>(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    streams: &GpuTensor,
    geo: &RunGeo,
    tile: &TilePlan,
    layer_index: usize,
    f: impl FnOnce(&mut Gpu, &TileFrame<'_>) -> R<T>,
) -> R<T> {
    with_tile_frames(
        forward,
        bundle,
        gpu,
        streams,
        geo,
        &[tile],
        layer_index,
        |gpu, frames| f(gpu, &frames[0]),
    )
}

/// The frames of several tiles of one layer at once, all alive together (the
/// panel scratch holds the operands of its one-row decode tiles for the whole
/// sweep). `frames[i]` belongs to `tiles[i]`; the caller has entered whichever
/// Gpu scope its own steps need.
#[allow(clippy::too_many_arguments)]
fn with_tile_frames<T>(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    streams: &GpuTensor,
    geo: &RunGeo,
    tiles: &[&TilePlan],
    layer_index: usize,
    f: impl FnOnce(&mut Gpu, &[TileFrame<'_>]) -> R<T>,
) -> R<T> {
    let layer = bundle
        .weights
        .layer(layer_index)
        .ok_or_else(|| invalid(format!("layer {layer_index} is not owned by this band")))?;
    let layer_moe = forward.moe.get(layer_index).and_then(Option::as_ref);
    let tile_streams: Vec<GpuTensor> = tiles
        .iter()
        .map(|tile| f32_view(streams, tile.stream_offset, tile.rows * geo.wide))
        .collect();
    let router_logits = tiles
        .iter()
        .map(|tile| {
            matrix_view(
                &forward.scratch.router_logits,
                tile.rows,
                router_columns(geo.dims),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scratch_descs: Vec<_> = router_logits
        .iter()
        .map(|logits| layer_scratch(&forward.scratch, logits))
        .collect();
    let descriptions = tiles
        .iter()
        .map(|tile| {
            layer_desc(
                &bundle.weights,
                layer,
                layer_moe,
                &geo.config,
                &forward.decode_q8,
                tile.routing.route_rows,
                geo.few_row_limit,
                None,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ctx = DispatchCtx::new(gpu);
    let envs: Vec<LayerPhaseEnv<'_>> = tiles
        .iter()
        .zip(&tile_streams)
        .map(|(tile, tile_streams)| LayerPhaseEnv {
            scratch: &forward.scratch,
            decode_q8: &forward.decode_q8,
            trunk_zba: &forward.trunk_zba,
            trunk_a4: &forward.trunk_a4,
            prefill_scope: forward.prefill_scope,
            bundle,
            dims: geo.dims,
            bf16_state: tile.routing.bf16_streams,
            route_n: tile.routing.route_rows,
            few_row_limit: geo.few_row_limit,
            rows: tile.rows,
            start_position: tile.position,
            streams: tile_streams,
            gates: &forward.scratch.hc_gates,
        })
        .collect();
    let frames: Vec<TileFrame<'_>> = (0..tiles.len())
        .map(|index| TileFrame {
            env: &envs[index],
            description: &descriptions[index],
            ctx: &ctx,
            scratch_desc: &scratch_descs[index],
            layer,
        })
        .collect();
    f(gpu, &frames)
}

// ── the run ──────────────────────────────────────────────────────────

/// The window being run, as the layer loops see it.
#[derive(Clone, Copy)]
struct WinCtx<'w> {
    window: &'w WindowPlan,
    tiles: &'w [TilePlan],
    /// The window's tokens.
    tokens: &'w [u32],
    last_window: bool,
}

impl<'w> WinCtx<'w> {
    fn tile_tokens(&self, ti: usize) -> &'w [u32] {
        let tile = &self.tiles[ti];
        &self.tokens[tile.window_row..tile.window_row + tile.rows]
    }
}

struct Run<'a, 'o> {
    forward: &'a mut Qwen4GpuForward,
    bundle: &'a mut Qwen4Bundle,
    gpu: &'a mut Gpu,
    rt: &'a mut SlowlinkRuntime,
    geo: &'a RunGeo,
    profile: &'a SlowlinkProfile,
    plan: &'a Qwen4PromptPlan,
    tokens: &'a [u32],
    output: &'a mut PromptOutput<'o>,
    ple: PleCursor,
}

impl Run<'_, '_> {
    /// One window: transaction, layer-major walk, commit (or abort).
    fn window(&mut self, window: &WindowPlan, last_window: bool) -> R<WindowReport> {
        let (plan, tokens) = (self.plan, self.tokens);
        let window_tokens = &tokens[window.prompt_row..window.prompt_row + window.rows];
        let w = WinCtx {
            window,
            tiles: plan.window_tiles(window),
            tokens: window_tokens,
            last_window,
        };
        // Each window is its own transaction, restored on its own failure, so
        // the drafter's `pending` row is backed up at THIS window's entry (the
        // previous window's sink left it at the window's first pair), exactly
        // the state the window's abort restores target and head to. A declined
        // request never reaches here, so it never touches `pending`.
        if let PromptOutput::Capture {
            pending: Some(pending),
            ..
        } = &*self.output
        {
            self.bundle
                .slowlink_save_pending(&mut *self.gpu, pending)
                .map_err(bundle_err("save pending hidden"))?;
        }
        let mut txn = self
            .bundle
            .prompt_txn_begin(&mut *self.gpu, window_tokens)
            .map_err(bundle_err("begin"))?;
        let mut progress = Progress {
            txn_open: true,
            ring_primed: false,
        };
        match self.window_body(&w, &mut txn, &mut progress) {
            Ok(report) => self
                .bundle
                .prompt_txn_commit(&mut *self.gpu, txn, window_tokens)
                .map(|()| report)
                .map_err(bundle_err("commit")),
            Err(error) => Err(self.fail_window(txn, progress, error)),
        }
    }

    /// The failed window's cleanup, in order: drain the ring, then restore the
    /// entry state. The original error survives unless a cleanup also failed.
    fn fail_window(
        &mut self,
        txn: PromptTxn,
        progress: Progress,
        error: Qwen4GpuForwardError,
    ) -> Qwen4GpuForwardError {
        let steps = progress.abort_steps();
        let mut extra: Vec<String> = Vec::new();
        if steps.drain_ring {
            if let Err(drain) = self.rt.ring.drain(&*self.gpu) {
                extra.push(format!("ring drain failed: {drain}"));
            }
        }
        if steps.abort_txn {
            if let Err(abort) = self.bundle.prompt_txn_abort(&mut *self.gpu, txn) {
                extra.push(format!("transaction abort failed: {abort}"));
            }
        } else {
            drop(txn);
        }
        if extra.is_empty() {
            error
        } else {
            invalid(format!("{error}; {}", extra.join("; ")))
        }
    }

    fn window_body(
        &mut self,
        w: &WinCtx<'_>,
        txn: &mut PromptTxn,
        progress: &mut Progress,
    ) -> R<WindowReport> {
        let geo = self.geo;
        self.ple.histories =
            PleCursor::histories_for(self.bundle.state.ple_history, w.tiles, w.tokens);
        if w.tiles
            .iter()
            .any(|tile| tile.routing.route_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS)
        {
            if let Some(qsa) = self
                .bundle
                .state
                .qsa
                .first()
                .filter(|qsa| qsa.f16_arena.is_none())
            {
                ensure_qsa_gathered_wmma_workspace(
                    &mut *self.gpu,
                    qsa.format,
                    geo.config.num_key_value_heads,
                    w.window.end_position(),
                )?;
            }
        }
        if self
            .bundle
            .slowlink_take_injection(SlowlinkFailPoint::BeforeDma)
        {
            return Err(invalid(
                "injected slow-link failure before the first expert copy",
            ));
        }
        let wall = Instant::now();
        let mut dims = self.profile.route_dims();
        dims.rows = w.window.rows;
        let layout = dsp::PromptRouteLayout::new(&dims).map_err(dsp_err("route layout"))?;
        let mut routes = dsp::PromptRouteStorage::from_region(&self.rt.route_region, &layout)
            .map_err(dsp_err("route storage"))?;
        let mut report = WindowReport::default();
        progress.ring_primed = true;
        self.rt.ring.prime(&*self.gpu, &mut self.rt.schedule)?;
        for layer in geo.band.clone() {
            self.rt.ring.mark_layer_begin(&*self.gpu, layer)?;
            if self.rt.host_layer[layer] {
                self.host_layer(w, layer, &mut routes, txn, &mut report)?;
            } else {
                self.resident_layer(w, layer, txn)?;
            }
            self.rt.ring.mark_layer_end(&*self.gpu, layer)?;
            if layer == geo.band.start
                && self
                    .bundle
                    .slowlink_take_injection(SlowlinkFailPoint::AfterLayer)
            {
                return Err(invalid("injected slow-link failure after the first layer"));
            }
        }
        self.rt.ring.drain(&*self.gpu)?;
        progress.ring_primed = false;
        let timeline = self.rt.ring.timeline(&*self.gpu)?;
        let stats = timeline.stats;
        if stats.timeline_overflow > 0 || stats.layer_mark_overflow > 0 {
            return Err(invalid(format!(
                "slow-link timeline overflowed ({} copy records, {} layer marks): the receipt would be incomplete",
                stats.timeline_overflow, stats.layer_mark_overflow
            )));
        }
        let sweep = self.rt.schedule.sweep_bytes();
        if stats.submitted_payload_bytes != sweep {
            return Err(invalid(format!(
                "slow-link window submitted {} payload bytes, one sweep is {sweep}",
                stats.submitted_payload_bytes
            )));
        }
        if report.useful_bytes > stats.submitted_payload_bytes {
            return Err(invalid(format!(
                "slow-link window read {} useful bytes of {} submitted",
                report.useful_bytes, stats.submitted_payload_bytes
            )));
        }
        report.stats = stats;
        report.overlap = timeline.overlap;
        report.sweep_bytes = sweep;
        report.wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let panel_busy: f64 = timeline
            .spans
            .iter()
            .filter_map(|span| span.compute.map(|(begin, end)| (end - begin).max(0.0)))
            .sum();
        report.nonexpert_ms = (timeline.overlap.compute_busy_ms - panel_busy).max(0.0);
        Ok(report)
    }

    /// A device-resident layer: the incumbent's per-tile steps.
    fn resident_layer(&mut self, w: &WinCtx<'_>, layer: usize, txn: &mut PromptTxn) -> R<()> {
        let geo = self.geo;
        for (ti, tile) in w.tiles.iter().enumerate() {
            if layer == geo.band.start {
                self.embed_tile(w, ti)?;
            }
            self.tile_layer(w, ti, layer, None)?;
            self.cut_if(txn, tile, layer)?;
            if layer == geo.last_layer {
                self.tile_output(w, ti)?;
            }
            self.rt.ring.pump(&*self.gpu, &mut self.rt.schedule)?;
        }
        Ok(())
    }

    /// A host layer: every tile's prefix and route preparation, the one count
    /// barrier, the panels in ascending expert order, every tile's tail.
    fn host_layer(
        &mut self,
        w: &WinCtx<'_>,
        layer: usize,
        routes: &mut dsp::PromptRouteStorage,
        txn: &mut PromptTxn,
        report: &mut WindowReport,
    ) -> R<()> {
        let geo = self.geo;
        self.rt
            .ring
            .enter_layer(&*self.gpu, &mut self.rt.schedule, layer)?;
        routes.reset_for_layer();
        for (ti, tile) in w.tiles.iter().enumerate() {
            if layer == geo.band.start {
                self.embed_tile(w, ti)?;
            }
            self.tile_layer(w, ti, layer, Some(&mut *routes))?;
            self.cut_if(txn, tile, layer)?;
            self.rt.ring.pump(&*self.gpu, &mut self.rt.schedule)?;
        }
        let route_plan = dsp::finish_route_plan(&mut *self.gpu, routes, dsp::PanelLimits::S1)
            .map_err(dsp_err("route plan"))?;
        report.route_barrier_ms += route_plan.barrier.sync_ms;
        self.sweep_panels(w, layer, routes, &route_plan, report)?;
        for ti in 0..w.tiles.len() {
            self.tile_tail(w, ti, layer, routes)?;
            if layer == geo.last_layer {
                self.tile_output(w, ti)?;
            }
        }
        Ok(())
    }

    /// The layer's panels over the ring's groups.
    fn sweep_panels(
        &mut self,
        w: &WinCtx<'_>,
        layer: usize,
        routes: &dsp::PromptRouteStorage,
        route_plan: &dsp::LayerRoutePlan,
        report: &mut WindowReport,
    ) -> R<()> {
        let geo = self.geo;
        let groups: Vec<ExpertGroupPlan> = self
            .rt
            .schedule
            .layer_groups(layer)
            .ok_or_else(|| invalid(format!("layer {layer} has no scheduled expert groups")))?
            .to_vec();
        let ranges = group_binding_ranges(&groups, geo.num_experts);
        let panels = route_plan
            .schedule(&ranges, dsp::PanelLimits::S1)
            .map_err(dsp_err("panel schedule"))?;
        let by_group = panels_by_group(panels.iter().map(|panel| panel.binding), groups.len())?;
        let work: Vec<GroupWork> = groups
            .iter()
            .enumerate()
            .map(|(index, group)| GroupWork {
                group: group.group,
                bound: by_group[index].start < by_group[index].end,
            })
            .collect();
        // Touched bytes: the experts any K2 row or the K1 cohort read, at the
        // placement's own extents; never the prefetched remainder.
        let mut touched: Vec<usize> = route_plan
            .experts_with_rows()
            .chain(route_plan.k1_experts.iter().copied())
            .collect();
        touched.sort_unstable();
        touched.dedup();
        for expert in touched {
            report.useful_bytes += self
                .rt
                .schedule
                .expert_extent_bytes(layer, expert)
                .unwrap_or(0) as u64;
        }
        let first_bound = work.iter().position(|group| group.bound);
        let inject = self
            .bundle
            .slowlink_take_injection(SlowlinkFailPoint::DuringPanel);
        // The panel scratch is anchored on a prefill tile of the window; every
        // one-row decode tile (the incumbent's decode call, original arithmetic)
        // registers its own operands once. A window of decode tiles only has no
        // prefill tile of its own: its anchor is a stand-in prefill tile of the
        // admitted tile rows, used only to borrow the shared scratch views (it
        // is never prepared, so no row reads or writes through it).
        let decode_tiles: Vec<usize> = (0..w.tiles.len())
            .filter(|&ti| w.tiles[ti].routing.route_rows == 1)
            .collect();
        let anchor_tile = (0..w.tiles.len()).find(|&ti| w.tiles[ti].routing.route_rows != 1);
        let stand_in;
        let anchor: &TilePlan = match anchor_tile {
            Some(ti) => &w.tiles[ti],
            None => {
                let rows = self.profile.tile_rows;
                stand_in = TilePlan {
                    index: 0,
                    window: w.window.index,
                    prompt_row: 0,
                    window_row: 0,
                    stream_offset: 0,
                    rows,
                    position: w.tiles[0].position,
                    routing: tile_routing(&*self.forward, &mut *self.gpu, rows),
                    ends_at_cut: None,
                    last_in_window: false,
                };
                &stand_in
            }
        };
        let mut frame_tiles: Vec<&TilePlan> = vec![anchor];
        frame_tiles.extend(decode_tiles.iter().map(|&ti| &w.tiles[ti]));
        // The K2 panels run under the anchor tile's ambient scope; each decode
        // or K1 step enters its own tile's scope through the hook.
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, anchor.rows);
        let result = with_tile_frames(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            geo,
            &frame_tiles,
            layer,
            |gpu, frames| {
                let mut sealed = Vec::with_capacity(frames.len());
                for frame in frames {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a host layer's FFN is not MoE"));
                    };
                    sealed.push(frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?);
                }
                // Each call is admitted under its own tile's scope (its routing
                // rows), exactly as its route preparation was.
                let mut calls = Vec::with_capacity(sealed.len());
                for (tile, sealed) in frame_tiles.iter().zip(&sealed) {
                    let (scopes, _) = self.forward.enter_tile_scopes(gpu, tile.rows);
                    let call =
                        dsp::PreparedChunkCall::new(&*gpu, sealed, tile.window_row, tile.position);
                    Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                    calls.push(call.map_err(dsp_err("admission"))?);
                }
                let mut scratch = dsp::PanelScratch::new(&calls[0], dsp::PanelLimits::S1)
                    .map_err(dsp_err("panel scratch"))?;
                for call in &calls[1..] {
                    scratch
                        .add_decode_tile(call)
                        .map_err(dsp_err("decode tile"))?;
                }
                let mut cursor = dsp::OrderedCombineCursor::start(gpu, routes, &panels)
                    .map_err(dsp_err("combine start"))?;
                let mut hook = ForwardScopeHook {
                    forward: &*self.forward,
                    tiles: w.tiles,
                    open: Vec::new(),
                };
                let mut ring = HipRing {
                    ring: &mut self.rt.ring,
                    schedule: &mut self.rt.schedule,
                };
                walk_layer_groups(&mut ring, gpu, layer, &work, &mut |gpu, index, parts| {
                    let parts =
                        parts.ok_or_else(|| invalid("slow-link panel lease carries no binding"))?;
                    if inject && Some(index) == first_bound {
                        return Err(invalid("injected slow-link failure during a panel"));
                    }
                    let group = &groups[index];
                    let (Some(&lo), Some(&last)) = (group.experts.first(), group.experts.last())
                    else {
                        return Err(invalid("an empty expert group"));
                    };
                    let binding = dsp::ExpertPanelBinding {
                        layer,
                        epoch: parts.epoch,
                        expert_lo: lo,
                        expert_hi: last + 1,
                        gate_up_table: parts.gate_up_table,
                        down_table: parts.down_table,
                        codec: dsp::PanelCodec::Mq4,
                        source: dsp::BindingSource::Staged {
                            slot: parts.ticket.slot as u32,
                        },
                    };
                    for panel_index in by_group[index].clone() {
                        let panel = &panels[panel_index];
                        dsp::execute_panel(
                            gpu,
                            binding,
                            route_plan.view(routes, panel),
                            &mut scratch,
                            &mut cursor,
                            &mut hook,
                        )
                        .map_err(dsp_err("panel"))?;
                    }
                    Ok(())
                })?;
                dsp::finish_combine(gpu, routes, &mut cursor, &mut hook).map_err(dsp_err("combine"))
            },
        );
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        result
    }

    /// Tile `ti` of `layer`: PLE rows, the mixer prefix, the QSA cursor, and
    /// either the rest of the layer (resident) or the route preparation (host).
    fn tile_layer(
        &mut self,
        w: &WinCtx<'_>,
        ti: usize,
        layer: usize,
        routes: Option<&mut dsp::PromptRouteStorage>,
    ) -> R<()> {
        let tile = &w.tiles[ti];
        let n = tile.rows;
        self.gpu
            .set_router_row_base(if n > 1 { tile.position } else { 0 });
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, n);
        let outcome = self.tile_layer_scoped(w, ti, layer, routes);
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        let qsa = outcome?;
        if let Some((full_len, raw_len, pooled_len, selected_len, position)) = qsa {
            // The layer's private cursor: only this layer's state moves with
            // the tile; the bundle's global position waits for the commit.
            let slot = self
                .bundle
                .state
                .qsa_local(layer)
                .ok_or_else(|| invalid("QSA state slot missing"))?;
            let state = self
                .bundle
                .state
                .qsa_mut(slot)
                .ok_or_else(|| invalid("QSA state slot disappeared"))?;
            state.full_len = full_len;
            state.raw_len = raw_len;
            state.pooled_len = pooled_len;
            state.selected_len = selected_len;
            state.position = position;
        }
        Ok(())
    }

    fn tile_layer_scoped(
        &mut self,
        w: &WinCtx<'_>,
        ti: usize,
        layer: usize,
        routes: Option<&mut dsp::PromptRouteStorage>,
    ) -> R<Option<(usize, usize, usize, usize, usize)>> {
        let geo = self.geo;
        let tile = &w.tiles[ti];
        let n = tile.rows;
        if geo.ple_layer == Some(layer) {
            if let Some(owner) = self.bundle.ple.as_ref() {
                let ple_weights = self
                    .bundle
                    .weights
                    .layer(layer)
                    .and_then(|weights| weights.ple.as_ref())
                    .ok_or_else(|| invalid("PLE weights missing"))?;
                let desc = crate::gpu_forward::ple_desc(&self.bundle.weights, ple_weights)?;
                let rows_f16 = crate::gpu_forward::qwen4_ple_rows_f16(
                    &*self.gpu,
                    tile.routing.bf16_streams,
                    geo.dims.hc_count,
                    geo.hidden,
                    &desc,
                    n,
                );
                let len = n * geo.ple_heads * geo.ple_width;
                let staged = view(&self.forward.scratch.ple_staged, 0, len);
                let rows = view(&self.forward.scratch.ple_rows, 0, len);
                self.ple.stage_tile(
                    owner,
                    &mut self.forward.host_ple,
                    &mut *self.gpu,
                    w,
                    ti,
                    &staged,
                    &rows,
                    rows_f16,
                    geo.ple_heads,
                    geo.ple_width,
                )?;
            }
        }
        with_tile_frame(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            geo,
            tile,
            layer,
            |gpu, frame| {
                let env = frame.env;
                let mut steps: Vec<Step<'_>> = Vec::with_capacity(8);
                if geo.ple_layer == Some(layer) {
                    let ple_weights = frame
                        .layer
                        .ple
                        .as_ref()
                        .ok_or_else(|| invalid("PLE weights missing"))?;
                    let (step, _rows_f16) =
                        env.ple_step(gpu, ple_weights, geo.ple_heads, geo.ple_width)?;
                    steps.push(step);
                }
                env.push_mixer_prefix(&mut steps, layer, frame.layer, frame.description)?;
                if routes.is_none() {
                    match &frame.description.ffn {
                        Qwen4FfnBinding::Moe(moe) => {
                            env.moe_steps(&mut steps, frame.ctx, *moe, frame.scratch_desc)?
                        }
                        Qwen4FfnBinding::Dense(dense) => steps.push(env.dense_step(dense)),
                    }
                    steps.push(env.mlp_write_step(frame.description, &env.scratch.ffn_output));
                }
                validate_steps(&*gpu, &steps).map_err(dsp_err("preflight"))?;
                let mut qsa = None;
                for step in &steps {
                    if let Step::IndexedAttention(op) = step {
                        qsa = Some(op.next_lengths().map_err(dsp_err("derive QSA state"))?);
                    }
                }
                execute_validated_steps(gpu, frame.ctx, &steps)
                    .map_err(dsp_err("layer program"))?;
                if let Some(routes) = routes {
                    let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                        return Err(invalid("a host layer's FFN is not MoE"));
                    };
                    let sealed = env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                    let call =
                        dsp::PreparedChunkCall::new(&*gpu, &sealed, tile.window_row, tile.position)
                            .map_err(dsp_err("admission"))?;
                    dsp::prepare_routes(gpu, &call, routes)
                        .map_err(dsp_err("route preparation"))?;
                }
                Ok(qsa)
            },
        )
    }

    /// Tile `ti`'s shared expert and closing HC write of a host layer.
    fn tile_tail(
        &mut self,
        w: &WinCtx<'_>,
        ti: usize,
        layer: usize,
        routes: &dsp::PromptRouteStorage,
    ) -> R<()> {
        let geo = self.geo;
        let tile = &w.tiles[ti];
        self.gpu
            .set_router_row_base(if tile.rows > 1 { tile.position } else { 0 });
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, tile.rows);
        let result = with_tile_frame(
            &*self.forward,
            &*self.bundle,
            &mut *self.gpu,
            &self.rt.streams,
            geo,
            tile,
            layer,
            |gpu, frame| {
                let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                    return Err(invalid("a host layer's FFN is not MoE"));
                };
                let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
                let call =
                    dsp::PreparedChunkCall::new(&*gpu, &sealed, tile.window_row, tile.position)
                        .map_err(dsp_err("admission"))?;
                let mut hook = ForwardScopeHook {
                    forward: &*self.forward,
                    tiles: w.tiles,
                    open: Vec::new(),
                };
                dsp::finish_shared(gpu, routes, &call, &mut hook)
                    .map_err(dsp_err("shared expert"))?;
                let mixed = f32_view(
                    routes.combined(),
                    tile.window_row * geo.hidden,
                    tile.rows * geo.hidden,
                );
                let step = frame.env.mlp_write_step(frame.description, &mixed);
                let steps = [step];
                validate_steps(&*gpu, &steps).map_err(dsp_err("preflight"))?;
                execute_validated_steps(gpu, frame.ctx, &steps).map_err(dsp_err("hyper write"))
            },
        );
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        result
    }

    /// Tile `ti`'s embedding into its rows of the HC-stream arena.
    fn embed_tile(&mut self, w: &WinCtx<'_>, ti: usize) -> R<()> {
        let geo = self.geo;
        let tile = &w.tiles[ti];
        let n = tile.rows;
        for (index, token) in w.tile_tokens(ti).iter().copied().enumerate() {
            self.forward.host_token_bytes[index * 4..index * 4 + 4]
                .copy_from_slice(&(token as i32).to_ne_bytes());
        }
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, n);
        let result = (|| -> R<()> {
            let scratch = &self.forward.scratch;
            self.gpu.memcpy_htod_auto(
                &scratch.token_ids.buf,
                &self.forward.host_token_bytes[..n * std::mem::size_of::<i32>()],
            )?;
            let ids = view(&scratch.token_ids, 0, n * std::mem::size_of::<i32>());
            let embedding_rot = view(&scratch.embedding_rot, 0, n * geo.hidden);
            let embeddings = view(&scratch.embeddings, 0, n * geo.hidden);
            let table = self
                .bundle
                .weights
                .resident(&self.bundle.weights.root.embedding)?;
            execute_embedding(
                &mut *self.gpu,
                &EmbeddingOp {
                    table,
                    rotated: &embedding_rot,
                    token_ids: &ids,
                    output: &embeddings,
                    rows: n,
                    dim: geo.hidden,
                },
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("Qwen4 embedding: {error:?}"))
            })?;
            let tile_streams = f32_view(&self.rt.streams, tile.stream_offset, n * geo.wide);
            if tile.routing.bf16_streams {
                self.gpu.hc_streams_init_from_embed_batched_bf16(
                    &embeddings,
                    &tile_streams,
                    geo.hidden as i32,
                    geo.config.hc_count as i32,
                    n as i32,
                )?;
            } else {
                self.gpu.hc_streams_init_from_embed_batched(
                    &embeddings,
                    &tile_streams,
                    geo.hidden as i32,
                    geo.config.hc_count as i32,
                    n as i32,
                )?;
            }
            Ok(())
        })();
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        self.forward.slowlink_note_ids_rewritten();
        result
    }

    /// The layer `layer`'s state at the cut tile `tile` ends at.
    fn cut_if(&mut self, txn: &mut PromptTxn, tile: &TilePlan, layer: usize) -> R<()> {
        if let Some(cut) = tile.ends_at_cut {
            self.bundle
                .prompt_txn_layer_cut(&mut *self.gpu, txn, cut, layer)
                .map_err(bundle_err("layer cut"))?;
        }
        Ok(())
    }

    /// What tile `ti` returns once its last layer ran: the wide-hidden capture
    /// sink and, for the last tile of the request, the head.
    fn tile_output(&mut self, w: &WinCtx<'_>, ti: usize) -> R<()> {
        let geo = self.geo;
        let tile = &w.tiles[ti];
        let n = tile.rows;
        let capture = matches!(self.output, PromptOutput::Capture { .. });
        let wants_head = w.last_window && tile.last_in_window && self.output.head().is_some();
        if !capture && !wants_head {
            return Ok(());
        }
        let (scopes, _) = self.forward.enter_tile_scopes(&mut *self.gpu, n);
        let scoped = (|| -> R<Option<GpuTensor>> {
            let tile_streams = f32_view(&self.rt.streams, tile.stream_offset, n * geo.wide);
            let bf16 = tile.routing.bf16_streams;
            let wide = if capture {
                let target = self
                    .bundle
                    .spec_hidden
                    .as_ref()
                    .ok_or_else(|| invalid("Qwen4 spec hidden is not allocated"))?;
                if target.numel() < n * geo.wide {
                    return Err(invalid(
                        "the wide-hidden capture buffer is narrower than the tile",
                    ));
                }
                let destination = f32_view(target, 0, n * geo.wide);
                let source = view(&tile_streams, 0, n * geo.wide);
                if bf16 {
                    rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                        &mut *self.gpu,
                        &source,
                        &destination,
                        n * geo.wide,
                    )?;
                } else {
                    self.gpu
                        .copy_d2d(&source, &destination, source.byte_size())?;
                }
                Some(destination)
            } else {
                None
            };
            if wants_head {
                let (logits, top1) = self
                    .output
                    .head()
                    .ok_or_else(|| invalid("slow-link head disappeared"))?;
                let scratch = &self.forward.scratch;
                let final_hyper = crate::gpu_forward::hyper_read_desc(
                    &self.bundle.weights,
                    &self.bundle.weights.root.final_hyper,
                )?;
                let lm_head = crate::gpu_forward::dense_ref(
                    &self.bundle.weights,
                    &self.bundle.weights.root.lm_head,
                )?;
                let router_logits =
                    matrix_view(&scratch.router_logits, n, router_columns(geo.dims))?;
                let scratch_desc = layer_scratch(scratch, &router_logits);
                execute_final_hyper(
                    &mut *self.gpu,
                    geo.dims,
                    &final_hyper,
                    &tile_streams,
                    &scratch_desc,
                    n,
                    bf16,
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 final hyper: {error:?}"))
                })?;
                let final_hidden = view(&scratch.hc_mixed, 0, n * geo.hidden);
                let head_rotation = view(&scratch.rotation, 0, n * geo.hidden);
                execute_lm_head(
                    &mut *self.gpu,
                    &lm_head,
                    &final_hidden,
                    logits,
                    n,
                    1,
                    Some(&head_rotation),
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 LM head: {error:?}"))
                })?;
                if let Some(top1) = top1 {
                    argmax_f32(
                        &mut *self.gpu,
                        &ArgmaxF32 {
                            logits,
                            indices: top1,
                            rows: 1,
                            vocab: geo.config.vocab_size,
                        },
                    )?;
                }
            }
            Ok(wide)
        })();
        Qwen4GpuForward::leave_tile_scopes(&mut *self.gpu, scopes);
        let wide = scoped?;
        if let (Some(wide), PromptOutput::Capture { sink, .. }) = (wide, &mut *self.output) {
            sink(
                &mut *self.gpu,
                &mut *self.bundle,
                PromptTile {
                    start: tile.position,
                    rows: n,
                    wide: &wide,
                    last: tile.last_in_window,
                },
            )
            .map_err(|error| invalid(format!("slow-link capture sink: {error}")))?;
        }
        Ok(())
    }
}

// ── entry ────────────────────────────────────────────────────────────

/// Run `tokens` (the request's rows after its entry position) layer-major when
/// the request qualifies. `Ok(false)` is the incumbent, decided BEFORE any
/// mutation or S1 allocation for this request; after the first window opened,
/// every outcome is `Ok(true)` or `Err` with the state restored (the incumbent
/// is never rerun on a mixed-position state).
pub(crate) fn try_run_prompt(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    tokens: &[u32],
    mut output: PromptOutput<'_>,
) -> R<bool> {
    let mode = SlowlinkMode::from_env();
    let selection = match select(forward, bundle, gpu, tokens, &output, mode) {
        Ok(selection) => selection,
        Err(reason) => {
            bundle.slowlink_note_bypass(&reason);
            return Ok(false);
        }
    };
    let Selection { plan, profile, geo } = selection;
    // (The head's entry `pending` hidden row is saved per window, right before
    // each window's transaction begins: see `Run::window`.)
    let ple_retained = forward.slowlink_begin(gpu)?;
    let mut runtime = forward
        .slowlink
        .runtime
        .take()
        .ok_or_else(|| invalid("slow-link runtime disappeared after selection"))?;
    let (resident_layers, host_layers, tile_scratch_bytes) = (
        runtime.resident_layers,
        runtime.host_layers,
        runtime.tile_scratch_bytes,
    );
    let mut reports: Vec<WindowReport> = Vec::with_capacity(plan.windows.len());
    let outcome = {
        let mut run = Run {
            forward: &mut *forward,
            bundle: &mut *bundle,
            gpu: &mut *gpu,
            rt: &mut runtime,
            geo: &geo,
            profile: &profile,
            plan: &plan,
            tokens,
            output: &mut output,
            ple: PleCursor {
                retained: ple_retained,
                histories: Vec::new(),
            },
        };
        let mut outcome = Ok(());
        for window in &plan.windows {
            let last = window.index + 1 == plan.windows.len();
            match run.window(window, last) {
                Ok(report) => reports.push(report),
                Err(error) => {
                    outcome = Err(error);
                    break;
                }
            }
        }
        outcome
    };
    forward.slowlink.runtime = Some(runtime);
    // ONE `hipMemGetInfo` (total - free) sample after the request, outside every
    // overlap loop: the device's used bytes at that moment, not a continuous
    // peak. `None` (a null in the receipt) when the query fails.
    let post_request_vram_used = gpu
        .device_mem_info()
        .ok()
        .map(|(free, total)| total.saturating_sub(free) as u64);
    let receipt = |status: String| {
        layer_major_receipt(&ReceiptInputs {
            mode,
            plan: &plan,
            resident_layers,
            host_layers,
            tile_scratch_bytes,
            reports: &reports,
            peak_vram_bytes: post_request_vram_used,
            status,
        })
    };
    match outcome {
        Ok(()) => {
            bundle.slowlink_set_receipt(receipt("ok".to_string()));
            Ok(true)
        }
        Err(error) => {
            bundle.slowlink_set_receipt(receipt(format!("aborted: {error}")));
            Err(error)
        }
    }
}

// ── production-primitive self-test ───────────────────────────────────

/// Compute-stream delay (in repeats of the transport's delay kernel) that
/// keeps each page's reader queued while the page refills.
const SELFTEST_DELAY_REPEATS: usize = 128;
const SELFTEST_SEED: u64 = 0x5eed_51ee_d5b0_0001;

/// `{"name","status":"pass","mismatches":0,"detail"}` of one dispatch
/// primitive that observed every check; a failed check is an `Err` carrying
/// the whole raw report (no mismatch count is invented).
fn primitive_case_json(name: &str, report: &dsp::selftest::PrimitiveReport) -> R<Value> {
    let raw = report.to_json();
    if !report.ok {
        return Err(invalid(format!("slow-link primitive {name} failed: {raw}")));
    }
    let detail: Value = serde_json::from_str(&raw).map_err(|error| {
        invalid(format!(
            "slow-link primitive {name}: unreadable report: {error}"
        ))
    })?;
    Ok(json!({
        "name": name,
        "status": "pass",
        "mismatches": 0u64,
        "detail": detail,
    }))
}

/// The transport primitives, measured by the ring's own diagnostic driven
/// through the same prime / enter / require / acquire / retire / pump path the
/// prompt run uses, on as many real host layers as two full pool sweeps need.
fn transport_case(rt: &mut SlowlinkRuntime, gpu: &mut Gpu, name: &str) -> R<Value> {
    let pages = rt
        .ring
        .spec()
        .reserve()
        .ok_or_else(|| invalid("slow-link page pool reserve overflows"))?
        .pages;
    let (mut layers, mut groups) = (0usize, 0usize);
    for (layer, host) in rt.host_layer.iter().enumerate() {
        if !host {
            continue;
        }
        layers += 1;
        groups += rt.schedule.layer_groups(layer).map_or(0, <[_]>::len);
        if groups >= 2 * pages {
            break;
        }
    }
    if groups < 2 * pages {
        return Err(invalid(format!(
            "the transport check needs {} expert groups to reuse every page twice, the host layers hold {groups}",
            2 * pages
        )));
    }
    let report = crate::gpu_forward::panel_stage::primitive_selftest(
        gpu,
        &mut rt.ring,
        &mut rt.schedule,
        layers,
        SELFTEST_DELAY_REPEATS,
    )?;
    let pass = match name {
        "payload_table_identity" => {
            report.payload_bytes_compared > 0
                && report.table_entries_checked > 0
                && report.pad_bytes_checked > 0
        }
        "repeated_epoch_same_address" => {
            report.repeated_address_epochs > 0 && report.min_epochs_per_address >= 2
        }
        "delayed_reader_reuse" => {
            report.delayed_reader_reuses > 0 && report.stale_ticket_rejections > 0
        }
        other => return Err(invalid(format!("{other:?} is not a transport primitive"))),
    };
    let detail = json!({
        "layers": report.layers,
        "groups_checked": report.groups_checked,
        "payload_bytes_compared": report.payload_bytes_compared,
        "table_entries_checked": report.table_entries_checked,
        "pad_bytes_checked": report.pad_bytes_checked,
        "delayed_reader_reuses": report.delayed_reader_reuses,
        "stale_ticket_rejections": report.stale_ticket_rejections,
        "distinct_addresses": report.distinct_addresses,
        "repeated_address_epochs": report.repeated_address_epochs,
        "min_epochs_per_address": report.min_epochs_per_address,
        "max_epochs_per_address": report.max_epochs_per_address,
        "submitted_payload_bytes": report.submitted_payload_bytes,
    });
    // Only observed counters pass. A case whose required counters were not
    // observed is an error carrying the whole raw report: the check never
    // invents a mismatch count.
    if !pass {
        return Err(invalid(format!(
            "slow-link primitive {name} did not observe its required counters: {detail}"
        )));
    }
    Ok(json!({
        "name": name,
        "status": "pass",
        "mismatches": 0u64,
        "detail": detail,
    }))
}

/// A dispatch primitive on a real sealed tile call and a real staged page
/// binding of the first host layer.
fn dispatch_case(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    rt: &mut SlowlinkRuntime,
    profile: &SlowlinkProfile,
    name: &str,
) -> R<Value> {
    let layer = rt
        .host_layer
        .iter()
        .position(|host| *host)
        .ok_or_else(|| invalid("no host layer to check against"))?;
    let few_row_limit = {
        let (scopes, _) = forward.enter_tile_scopes(gpu, profile.tile_rows);
        let limit = gpu.few_row_limit();
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        limit
    };
    let geo = RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts)?;
    let rows = profile.tile_rows;
    let routing = tile_routing(forward, gpu, rows);
    let stub = TilePlan {
        index: 0,
        window: 0,
        prompt_row: 0,
        window_row: 0,
        stream_offset: 0,
        rows,
        position: bundle.state.position,
        routing,
        ends_at_cut: None,
        last_in_window: true,
    };
    let group = rt
        .schedule
        .layer_groups(layer)
        .and_then(|groups| groups.first())
        .cloned()
        .ok_or_else(|| invalid("the first host layer has no scheduled group"))?;
    rt.ring.prime(&*gpu, &mut rt.schedule)?;
    rt.ring.enter_layer(&*gpu, &mut rt.schedule, layer)?;
    let (scopes, _) = forward.enter_tile_scopes(gpu, rows);
    let result = with_tile_frame(
        forward,
        bundle,
        gpu,
        &rt.streams,
        &geo,
        &stub,
        layer,
        |gpu, frame| {
            let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                return Err(invalid("a host layer's FFN is not MoE"));
            };
            let sealed = frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?;
            let call = dsp::PreparedChunkCall::new(&*gpu, &sealed, 0, stub.position)
                .map_err(dsp_err("admission"))?;
            let mut ring = HipRing {
                ring: &mut rt.ring,
                schedule: &mut rt.schedule,
            };
            let ticket = ring.require(&*gpu, layer, group.group)?;
            let mut out: Option<R<Value>> = None;
            ring.with_lease(gpu, ticket, &mut |gpu, parts| {
                let parts =
                    parts.ok_or_else(|| invalid("slow-link page lease carries no binding"))?;
                let (Some(&lo), Some(&last)) = (group.experts.first(), group.experts.last()) else {
                    return Err(invalid("an empty expert group"));
                };
                let binding = dsp::ExpertPanelBinding {
                    layer,
                    epoch: parts.epoch,
                    expert_lo: lo,
                    expert_hi: last + 1,
                    gate_up_table: parts.gate_up_table,
                    down_table: parts.down_table,
                    codec: dsp::PanelCodec::Mq4,
                    source: dsp::BindingSource::Staged {
                        slot: parts.ticket.slot as u32,
                    },
                };
                let limits = dsp::PanelLimits::S1;
                let report = match name {
                    "within_expert_panel_canary" => dsp::selftest::within_expert_panel_canary(
                        gpu,
                        &call,
                        binding,
                        limits,
                        SELFTEST_SEED,
                    ),
                    "k1_tail_cohort" => {
                        dsp::selftest::k1_tail_cohort(gpu, &call, binding, limits, SELFTEST_SEED)
                    }
                    "zero_row_panel" => {
                        dsp::selftest::zero_row_panel(gpu, &call, binding, limits, SELFTEST_SEED)
                    }
                    other => return Err(invalid(format!("{other:?} is not a dispatch primitive"))),
                }
                .map_err(dsp_err("primitive"))?;
                out = Some(primitive_case_json(name, &report));
                Ok(())
            })?;
            ring.retire(&*gpu, ticket)?;
            out.ok_or_else(|| invalid("slow-link primitive produced no report"))?
        },
    );
    Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
    result
}

/// The one-row decode tail beside a K2 tile, on a device-resident layer: the
/// dispatch crate's `decode_tail_cohort` over the incumbent's real K2 prefill
/// call (64 rows) and real decode call (1 row at window row 64), each sealed
/// the way the model seals it, against the layer's resident all-expert binding
/// (the router may select any expert, a staged 32-expert page cannot hold them).
fn decode_tail_case(
    forward: &Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    gpu: &mut Gpu,
    rt: &SlowlinkRuntime,
    profile: &SlowlinkProfile,
) -> R<Value> {
    let few_row_limit = {
        let (scopes, _) = forward.enter_tile_scopes(gpu, profile.tile_rows);
        let limit = gpu.few_row_limit();
        Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
        limit
    };
    let geo = RunGeo::new(forward, bundle, few_row_limit, profile.geometry.num_experts)?;
    let layer = geo
        .band
        .clone()
        .find(|&layer| {
            forward
                .moe
                .get(layer)
                .and_then(Option::as_ref)
                .is_some_and(|runtime| !runtime.s1_host_mapped() && runtime.s1_layer_check().is_ok())
        })
        .ok_or_else(|| {
            invalid(
                "the decode tail check needs a device-resident qualified expert layer; this placement has none",
            )
        })?;
    let anchor_rows = dsp::selftest::DECODE_TAIL_ANCHOR_ROWS;
    let position = bundle.state.position;
    let anchor_tile = TilePlan {
        index: 0,
        window: 0,
        prompt_row: 0,
        window_row: 0,
        stream_offset: 0,
        rows: anchor_rows,
        position,
        routing: tile_routing(forward, gpu, anchor_rows),
        ends_at_cut: None,
        last_in_window: false,
    };
    let decode_tile = TilePlan {
        index: 1,
        window: 0,
        prompt_row: anchor_rows,
        window_row: anchor_rows,
        stream_offset: anchor_rows * geo.wide,
        rows: 1,
        position: position + anchor_rows,
        routing: tile_routing(forward, gpu, 1),
        ends_at_cut: None,
        last_in_window: true,
    };
    let tiles = [&anchor_tile, &decode_tile];
    with_tile_frames(
        forward,
        bundle,
        gpu,
        &rt.streams,
        &geo,
        &tiles,
        layer,
        |gpu, frames| {
            let mut sealed = Vec::with_capacity(frames.len());
            for frame in frames {
                let Qwen4FfnBinding::Moe(moe) = &frame.description.ffn else {
                    return Err(invalid("a resident layer's FFN is not MoE"));
                };
                sealed.push(frame.env.seal_moe(frame.ctx, *moe, frame.scratch_desc)?);
            }
            let mut calls = Vec::with_capacity(sealed.len());
            for (tile, sealed) in tiles.iter().zip(&sealed) {
                let (scopes, _) = forward.enter_tile_scopes(gpu, tile.rows);
                let call =
                    dsp::PreparedChunkCall::new(&*gpu, sealed, tile.window_row, tile.position);
                Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
                calls.push(call.map_err(dsp_err("admission"))?);
            }
            if !calls[0].is_k2() || calls[0].is_decode() || !calls[1].is_decode() {
                return Err(invalid(format!(
                    "the decode tail check expects a K2 anchor and a decode tile; the seals gave k2={} decode={} and decode={}",
                    calls[0].is_k2(),
                    calls[0].is_decode(),
                    calls[1].is_decode()
                )));
            }
            let Qwen4FfnBinding::Moe(moe) = &frames[0].description.ffn else {
                return Err(invalid("a resident layer's FFN is not MoE"));
            };
            let binding = dsp::ExpertPanelBinding::resident(
                layer,
                geo.num_experts,
                moe.expert_gate_up_ptrs,
                moe.expert_down_ptrs,
            );
            let (scopes, _) = forward.enter_tile_scopes(gpu, anchor_rows);
            let report = dsp::selftest::decode_tail_cohort(
                gpu,
                &calls[0],
                &calls[1],
                binding,
                dsp::PanelLimits::S1,
                SELFTEST_SEED,
            );
            Qwen4GpuForward::leave_tile_scopes(gpu, scopes);
            primitive_case_json(
                "decode_tail_cohort",
                &report.map_err(dsp_err("primitive"))?,
            )
        },
    )
}

/// One named primitive check on the attached forward's real owners; the
/// ring is drained and idle afterwards whatever the outcome.
pub(crate) fn primitive_selftest(
    forward: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    name: &str,
) -> R<Value> {
    let profile = forward
        .slowlink_profile()
        .copied()
        .ok_or_else(|| invalid("slow-link primitive check needs an admitted profile"))?;
    if let Some(why) = forward.slowlink_forward_blocker(&bundle.config) {
        return Err(invalid(format!(
            "slow-link primitive check cannot run: {why}"
        )));
    }
    ensure_runtime(forward, &*bundle, gpu, &profile).map_err(invalid)?;
    let mut rt = forward
        .slowlink
        .runtime
        .take()
        .ok_or_else(|| invalid("slow-link runtime disappeared"))?;
    let prior_scope = std::mem::replace(&mut forward.prefill_scope, true);
    let result = match name {
        "payload_table_identity" | "repeated_epoch_same_address" | "delayed_reader_reuse" => {
            transport_case(&mut rt, gpu, name)
        }
        "canonical_combine_exact" => dsp::selftest::canonical_combine_exact(gpu, SELFTEST_SEED)
            .map_err(dsp_err("canonical combine check"))
            .and_then(|report| primitive_case_json(name, &report)),
        "within_expert_panel_canary" | "zero_row_panel" => {
            dispatch_case(forward, &*bundle, gpu, &mut rt, &profile, name)
        }
        // The few-row cohort is two checks that must BOTH pass: the prefill K1
        // tail on a staged host-layer binding, and the incumbent's REAL one-row
        // decode call beside a K2 tile on a resident all-expert binding.
        "k1_tail_cohort" => dispatch_case(forward, &*bundle, gpu, &mut rt, &profile, name)
            .and_then(|prefill| {
                decode_tail_case(forward, &*bundle, gpu, &rt, &profile).map(|decode| {
                    json!({
                        "name": name,
                        "status": "pass",
                        "mismatches": 0u64,
                        "detail": {
                            "prefill_k1": prefill["detail"],
                            "decode_tail": decode["detail"],
                        },
                    })
                })
            }),
        other => Err(invalid(format!("unknown slow-link primitive {other:?}"))),
    };
    let drained = rt.ring.drain(&*gpu);
    forward.slowlink.runtime = Some(rt);
    forward.prefill_scope = prior_scope;
    let report = result?;
    drained?;
    Ok(report)
}

#[cfg(test)]
mod slowlink_prompt {
    use super::*;
    use crate::slowlink_admission::{
        admit, AdmissionInputs, ModelGeometry, S1_TILE_ROWS, S1_WINDOW_ROWS,
    };
    use rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS;

    const MIB: u64 = 1 << 20;
    const GATE_UP_LAYER: u64 = 512 * 1_740_800;
    const DOWN_LAYER: u64 = 512 * 870_400;

    /// The admitted profile of a Flash-Next MQ4 load with ample memory.
    pub(super) fn profile() -> SlowlinkProfile {
        let geometry = ModelGeometry::from_config(&crate::config::compact_test_config()).unwrap();
        admit(&AdmissionInputs {
            geometry,
            free: 40 << 30,
            non_expert: 5_364_000_000,
            layer_expert_bytes: GATE_UP_LAYER + DOWN_LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: 48,
            base_reserve: 5261 * MIB,
            baseline_layers: 10,
            fixed_state_bytes: 64 * MIB,
            pending_save_bytes: 4 * 2560 * 4,
            checkpoint_stride_rows: 8192,
            window_rows: S1_WINDOW_ROWS,
            link_gbps: 7.09,
        })
        .expect("a 40 GiB load admits S1")
    }

    /// The incumbent's decisions on a gfx1100-class tile: the routing rows are
    /// the launch rows and the HC streams are BF16 from the WMMA minimum up.
    fn gfx1100_routing(rows: usize) -> TileRouting {
        TileRouting {
            route_rows: rows,
            bf16_streams: rows >= QWEN4_F16_WMMA_MIN_TOKENS,
        }
    }

    /// The canonical-routing (batch-invariant) variant: every tile below the
    /// canonical minimum routes as the canonical chunk.
    fn canonical_routing(rows: usize) -> TileRouting {
        let route_rows = rows.max(QWEN4_F16_WMMA_MIN_TOKENS);
        TileRouting {
            route_rows,
            bf16_streams: route_rows >= QWEN4_F16_WMMA_MIN_TOKENS,
        }
    }

    pub(super) fn plan_with(
        entry: usize,
        rows: usize,
        cuts: &[usize],
        routing: &dyn Fn(usize) -> TileRouting,
        few_row_limit: usize,
    ) -> R<Result<Qwen4PromptPlan, PlanBypass>> {
        plan_prompt_verdict(&PromptPlanInputs {
            profile: &profile(),
            entry_pos: entry,
            rows,
            cuts,
            routing,
            few_row_limit,
        })
    }

    /// A resumed request: the session restored 3000 rows, 100 follow.
    fn plan_resumed() -> Qwen4PromptPlan {
        plan(3000, 100, &[])
    }

    pub(super) fn plan(entry: usize, rows: usize, cuts: &[usize]) -> Qwen4PromptPlan {
        plan_with(entry, rows, cuts, &gfx1100_routing, 8)
            .unwrap()
            .expect("a qualified request plans")
    }

    /// Every tile is contiguous with the previous one across windows.
    fn assert_chronology(plan: &Qwen4PromptPlan) {
        let mut position = plan.entry_pos;
        let mut row = 0;
        for tile in &plan.tiles {
            assert_eq!(
                tile.position, position,
                "tile {} starts at the previous end",
                tile.index
            );
            assert_eq!(tile.prompt_row, row);
            assert!(tile.rows > 0 && tile.rows <= S1_TILE_ROWS);
            position += tile.rows;
            row += tile.rows;
        }
        assert_eq!(row, plan.rows);
        let mut window_start = plan.entry_pos;
        for window in &plan.windows {
            assert_eq!(window.position, window_start);
            let tiles = plan.window_tiles(window);
            assert_eq!(tiles.iter().map(|t| t.rows).sum::<usize>(), window.rows);
            assert!(window.rows <= S1_WINDOW_ROWS);
            assert!(tiles.last().unwrap().last_in_window);
            assert!(tiles[..tiles.len() - 1].iter().all(|t| !t.last_in_window));
            let mut window_row = 0;
            for tile in tiles {
                assert_eq!(tile.window_row, window_row);
                assert_eq!(tile.window, window.index);
                window_row += tile.rows;
            }
            window_start += window.rows;
        }
        assert_eq!(window_start, plan.entry_pos + plan.rows);
    }

    #[test]
    fn slowlink_prompt_tiles_are_the_incumbent_chunk_chronology() {
        let plan = plan(0, 16384, &[]);
        assert_chronology(&plan);
        assert_eq!(plan.windows.len(), 1, "16384 rows are one admitted window");
        let rows: Vec<usize> = plan.tiles.iter().map(|t| t.rows).collect();
        assert_eq!(rows, [vec![1536; 10], vec![1024]].concat());
        assert_eq!(plan.launch_cuts().last(), Some(&16384));
        assert!(plan.tiles.iter().all(|t| t.ends_at_cut.is_none()));
        assert!(plan
            .tiles
            .iter()
            .all(|t| t.routing == gfx1100_routing(t.rows)));
        assert_eq!(plan.k1_rows_max, 0);
    }

    #[test]
    fn slowlink_prompt_resumed_entry_keeps_absolute_positions() {
        let plan = plan(3000, 5000, &[]);
        assert_chronology(&plan);
        assert_eq!(plan.tiles[0].position, 3000);
        assert_eq!(plan.tiles.last().unwrap().end_position(), 8000);
        assert_eq!(plan.windows[0].position, 3000);
    }

    #[test]
    fn slowlink_prompt_boundaries_split_tiles_like_the_incumbent() {
        // The boundary cuts the plan's tile list the way the incumbent's
        // `prefill_final_from` split its chunk loop: segments between cuts,
        // each tiled by the tile size, never a tile across a cut.
        for cuts in [
            vec![1],
            vec![8],
            vec![3],
            vec![4],
            vec![1536],
            vec![1537],
            vec![8192],
            vec![3, 4],
            vec![3, 1536, 1537],
        ] {
            let plan = plan(0, 16384, &cuts);
            assert_chronology(&plan);
            for &cut in &cuts {
                assert!(
                    plan.tiles.iter().any(|t| t.end_position() == cut),
                    "cut {cut} is a tile end"
                );
                assert!(
                    plan.tiles
                        .iter()
                        .all(|t| !(t.position < cut && cut < t.end_position())),
                    "no tile straddles cut {cut}"
                );
            }
            assert_eq!(
                plan.windows.len(),
                1,
                "{cuts:?} fit one window's boundary snapshots"
            );
            let window = &plan.windows[0];
            assert_eq!(window.cuts, cuts);
            for tile in plan.window_tiles(window) {
                if let Some(index) = tile.ends_at_cut {
                    assert_eq!(window.cuts[index], tile.end_position());
                }
            }
            let ended: Vec<usize> = plan
                .window_tiles(window)
                .iter()
                .filter_map(|t| t.ends_at_cut.map(|index| window.cuts[index]))
                .collect();
            assert_eq!(
                ended, cuts,
                "every cut is staged by exactly one tile end, in order"
            );
        }
        let at_3 = plan(0, 16384, &[3]);
        assert_eq!(at_3.tiles[0].rows, 3);
        assert_eq!(at_3.tiles[1].rows, 1536);
        assert_eq!(at_3.k1_rows_max, 3, "a 3-row head tile is a K1 tile");
        // The 1-row and 8-row tails the mandatory short-tail contract names: each
        // is its own few-row tile with the original routing rows (1 and 8), which
        // the dispatch admission must take as a K1 cohort, never refuse.
        for rows in [1usize, 8] {
            let tail = plan(0, 16384, &[rows]);
            assert_eq!(tail.tiles[0].rows, rows);
            assert_eq!(tail.tiles[0].routing.route_rows, rows);
            assert!(!tail.tiles[0].routing.bf16_streams);
            assert_eq!(tail.k1_rows_max, rows);
        }
    }

    #[test]
    fn slowlink_prompt_boundary_snapshots_force_budgeted_superwindows() {
        // Five pending boundaries: a window keeps at most four strictly inside
        // it, so the fifth ENDS the first window (the transaction's own rule).
        let cuts = [3, 4, 1536, 1537, 8192];
        let plan = plan(0, 16384, &cuts);
        assert_chronology(&plan);
        assert_eq!(plan.windows.len(), 2);
        assert_eq!(plan.windows[0].end_position(), 8192);
        assert_eq!(plan.windows[0].cuts, vec![3, 4, 1536, 1537]);
        assert_eq!(plan.windows[1].position, 8192);
        assert!(plan.windows[1].cuts.is_empty());
        assert!(
            plan.windows
                .iter()
                .all(|w| w.cuts.len() <= S1_MAX_BOUNDARY_SNAPSHOTS),
            "no window exceeds the admitted boundary snapshots"
        );
        assert!(plan.windows[0].tiles.end == plan.windows[1].tiles.start);
        // The window END boundary is the commit's: its tile carries no cut.
        assert!(plan.tiles[plan.windows[0].tiles.end - 1]
            .ends_at_cut
            .is_none());
    }

    #[test]
    fn slowlink_prompt_row_cap_makes_superwindows_and_sweep_count() {
        let plan = plan(0, 40000, &[]);
        assert_chronology(&plan);
        let windows: Vec<usize> = plan.windows.iter().map(|w| w.rows).collect();
        // A window ends at a tile end, so ten 1536-row tiles (15360 rows) fill it.
        assert_eq!(windows, vec![15360, 15360, 9280]);
        // One sweep per window: a long request can never claim one per prompt.
        let reports = vec![WindowReport::default(); plan.windows.len()];
        let receipt = layer_major_receipt(&ReceiptInputs {
            mode: SlowlinkMode::Force,
            plan: &plan,
            resident_layers: 10,
            host_layers: 38,
            tile_scratch_bytes: 1 << 30,
            reports: &reports,
            peak_vram_bytes: None,
            status: "ok".to_string(),
        });
        assert_eq!(receipt["sweeps"], 3);
        assert_eq!(receipt["windows"].as_array().unwrap().len(), 3);
        assert!(plan.windows.len() >= profile().windows_for(plan.rows));
    }

    #[test]
    fn slowlink_prompt_mixed_signature_tails_keep_their_original_routing() {
        // Both routing models: a tail keeps the routing rows and HC tag of ITS
        // rows, never the prompt total or the 1536-row tile's.
        for routing in [
            &gfx1100_routing as &dyn Fn(usize) -> TileRouting,
            &canonical_routing,
        ] {
            let plan = plan_with(0, 5000, &[3, 1537], routing, 8)
                .unwrap()
                .expect("plans");
            assert_chronology(&plan);
            for tile in &plan.tiles {
                assert_eq!(
                    tile.routing,
                    routing(tile.rows),
                    "tile of {} rows",
                    tile.rows
                );
            }
            let signatures = plan.signatures();
            assert!(signatures.iter().any(|(rows, _)| *rows == 3));
            assert!(signatures.iter().any(|(rows, _)| *rows == S1_TILE_ROWS));
            assert_eq!(
                signatures.len(),
                {
                    let mut distinct: Vec<usize> = plan.tiles.iter().map(|t| t.rows).collect();
                    distinct.sort_unstable();
                    distinct.dedup();
                    distinct.len()
                },
                "one signature per distinct tile shape"
            );
        }
        let tail = plan(0, 1536 + 100, &[]);
        assert_eq!(
            tail.tiles.iter().map(|t| t.rows).collect::<Vec<_>>(),
            [1536, 100]
        );
        assert_ne!(tail.tiles[1].routing, tail.tiles[0].routing);
        assert!(!gfx1100_routing(3).bf16_streams && gfx1100_routing(1536).bf16_streams);
        let canonical = plan_with(0, 5000, &[3], &canonical_routing, 8)
            .unwrap()
            .unwrap();
        assert_eq!(
            canonical.tiles[0].routing.route_rows,
            QWEN4_F16_WMMA_MIN_TOKENS
        );
        assert_eq!(canonical.k1_rows_max, 0, "canonical rows are never K1");
    }

    #[test]
    fn slowlink_prompt_k1_tail_rows_above_the_route_storage_bound_bypass_before_mutation() {
        // Every tile counting as K1 (a limit above the tile) overflows the
        // few-row region the route storage was sized for.
        let verdict = plan_with(0, 16384, &[], &gfx1100_routing, S1_TILE_ROWS).unwrap();
        assert!(matches!(
            verdict,
            Err(PlanBypass::K1TailRows { window: 0, bound, .. }) if bound == S1_K1_TAIL_ROWS
        ));
        // With the incumbent's limit a window's K1 rows are the short tails
        // only and stay within the bound by construction (four interior cuts
        // plus the entry and final tails, eight rows each).
        let worst = plan(0, 30, &[8, 16, 24]);
        assert!(worst.k1_rows_max <= S1_K1_TAIL_ROWS);
    }

    #[test]
    fn slowlink_prompt_malformed_cuts_are_errors_not_verdicts() {
        for cuts in [vec![0], vec![5, 5], vec![9, 4], vec![100]] {
            assert!(
                plan_with(0, 100, &cuts, &gfx1100_routing, 8).is_err(),
                "{cuts:?}"
            );
        }
        assert!(plan_with(0, 0, &[], &gfx1100_routing, 8).is_err());
    }

    #[test]
    fn slowlink_prompt_ordered_ple_histories_follow_the_private_cursor() {
        let plan = plan(0, 1536 * 3 + 7, &[1536]);
        let tokens: Vec<u32> = (0..plan.rows as u32).map(|t| t * 3 + 1).collect();
        let entry = PleHistory::new(151_643);
        let histories = PleCursor::histories_for(entry, &plan.tiles, &tokens);
        assert_eq!(histories.len(), plan.tiles.len());
        assert_eq!(
            histories[0], entry,
            "the first tile starts at the entry history"
        );
        for (index, tile) in plan.tiles.iter().enumerate().skip(1) {
            // The history at a tile is the entry history advanced by every
            // earlier token, exactly what the sequential chunk loop commits.
            let mut expected = entry;
            for &token in &tokens[..tile.prompt_row] {
                expected.push(token);
            }
            assert_eq!(histories[index], expected, "tile {index}");
            assert_eq!(
                histories[index].previous(),
                [tokens[tile.prompt_row - 2], tokens[tile.prompt_row - 1]]
            );
        }
        // Resumed entry: the published history at the window start is the base.
        let resumed = PleHistory::from_previous(151_643, [11, 22]);
        let from_resume = PleCursor::histories_for(resumed, &plan.tiles, &tokens);
        assert_eq!(from_resume[0], resumed);
        assert_ne!(from_resume[0], histories[0], "the entry history is the base");
        // The history is the last TWO tokens: once a tile has consumed two rows
        // the entry no longer matters, so a full-size tile's successors agree.
        assert_eq!(
            from_resume[1], histories[1],
            "past two rows of tokens the entry history is forgotten"
        );
        // A one-row head tile leaves one entry token in its successor's history:
        // that is where the entry must (and does) show, by the private cursor.
        let short = plan(0, 16384, &[1]);
        assert_eq!(short.tiles[0].rows, 1);
        let short_tokens: Vec<u32> = (0..short.rows as u32).map(|t| t * 7 + 2).collect();
        let from_entry = PleCursor::histories_for(entry, &short.tiles, &short_tokens);
        let from_resumed = PleCursor::histories_for(resumed, &short.tiles, &short_tokens);
        assert_ne!(from_entry[1], from_resumed[1]);
        assert_eq!(
            from_resumed[1].previous(),
            [22, short_tokens[0]],
            "one entry token plus the head tile's only token"
        );
        assert_eq!(
            from_entry[1].previous(),
            [entry.previous()[1], short_tokens[0]]
        );
    }

    #[test]
    fn slowlink_prompt_mtp_tiles_pair_the_previous_last_row_in_order() {
        // The head consumes tiles ascending and contiguous; token `p` pairs
        // with the target hidden of `p - 1`, which for a tile's first row is
        // the previous tile's LAST row (`pending`); position 0 uses its own.
        let plan = plan(0, 1536 * 2 + 40, &[1536, 1540]);
        assert_chronology(&plan);
        for pair in plan.tiles.windows(2) {
            assert_eq!(
                pair[1].position,
                pair[0].end_position(),
                "the previous tile's last row is position {} - 1",
                pair[1].position
            );
            assert!(pair[0].rows <= S1_TILE_ROWS, "bounded capture rows");
        }
        assert_eq!(
            plan.tiles[0].position, 0,
            "prompt position 0 pairs with itself"
        );
        // A resumed MTP fill starts mid-prompt: its first tile pairs with the
        // entry `pending` row saved before the window.
        let resumed_plan = plan_resumed();
        assert_eq!(resumed_plan.tiles[0].position, 3000);
        assert_eq!(
            plan.tiles.iter().filter(|t| t.last_in_window).count(),
            plan.windows.len()
        );
    }

    #[test]
    fn slowlink_prompt_hc_streams_are_exact_tagged_spans_per_tile() {
        let profile = profile();
        let wide = profile.geometry.hc_count * profile.geometry.hidden;
        // 1536-row BF16 tiles, then a 100-row F32 tail and a 3-row F32 head.
        let plan = plan(0, 1536 * 2 + 100, &[3]);
        let mut elements = 0usize;
        for tile in &plan.tiles {
            assert_eq!(
                tile.stream_offset, elements,
                "tiles pack back to back in tag size"
            );
            let bytes = if tile.routing.bf16_streams { 2 } else { 4 };
            elements += tile.rows * wide * bytes / 4;
        }
        let window = &plan.windows[0];
        assert_eq!(window.stream_bytes, elements as u64 * 4);
        let all_f32 = (plan.rows * wide * 4) as u64;
        assert!(
            window.stream_bytes < all_f32,
            "BF16 tiles take half the bytes of an all-F32 reservation"
        );
        // BF16 regions reach past their own bytes by one F32 view: the arena
        // keeps one tile of BF16 slack so the last view stays inside it.
        let slack = (profile.tile_rows * wide * 2) as u64;
        assert!(window.stream_bytes + slack <= profile.arena.total_bytes);
    }

    #[test]
    fn slowlink_prompt_arena_that_cannot_hold_the_tagged_spans_bypasses_before_mutation() {
        let mut small = profile();
        small.arena.total_bytes = 1 << 20;
        let verdict = plan_prompt_verdict(&PromptPlanInputs {
            profile: &small,
            entry_pos: 0,
            rows: 16384,
            cuts: &[],
            routing: &gfx1100_routing,
            few_row_limit: 8,
        })
        .unwrap();
        assert!(matches!(
            verdict,
            Err(PlanBypass::StreamArena { window: 0, .. })
        ));
    }

    #[test]
    fn slowlink_prompt_abort_phase_drains_before_restore() {
        // The cleanup of each named failure point.
        let before_dma = Progress {
            txn_open: true,
            ring_primed: false,
        };
        assert_eq!(
            before_dma.abort_steps(),
            AbortSteps {
                drain_ring: false,
                abort_txn: true
            }
        );
        let during_panel = Progress {
            txn_open: true,
            ring_primed: true,
        };
        assert_eq!(
            during_panel.abort_steps(),
            AbortSteps {
                drain_ring: true,
                abort_txn: true
            }
        );
        // After the window drained (commit stage) only the transaction remains
        // to restore; a commit failure already restored it itself.
        let before_commit = Progress {
            txn_open: true,
            ring_primed: false,
        };
        assert_eq!(
            before_commit.abort_steps(),
            AbortSteps {
                drain_ring: false,
                abort_txn: true
            }
        );
        let consumed = Progress::default();
        assert_eq!(
            consumed.abort_steps(),
            AbortSteps {
                drain_ring: false,
                abort_txn: false
            }
        );
    }

    #[test]
    fn slowlink_prompt_census_equals_the_allocation_layouts() {
        let profile = profile();
        let plan = plan(0, 16384, &[]);
        let census = prompt_bytes(&plan).unwrap();
        let layout = dsp::PromptRouteLayout::new(&profile.route_dims()).unwrap();
        assert_eq!(census.route_storage, layout.total_bytes() as u64);
        assert_eq!(census.route_storage, profile.census.route_storage);
        assert_eq!(census.hc_streams, profile.arena.total_bytes);
        let spec = profile.pool_spec().unwrap();
        let reserve = spec.reserve().unwrap();
        assert_eq!(census.ring + census.lookahead, reserve.payload as u64);
        assert_eq!(census.tables, reserve.tables as u64);
        assert_eq!(census.group_scatter, profile.census.group_scatter);
        assert_eq!(
            census.group_scatter,
            crate::slowlink_admission::group_scatter_bytes(&profile.geometry, profile.window_rows)
                .unwrap(),
            "the receipt's scatter charge is the layout's, one source"
        );
        let wide = profile.geometry.hc_count * profile.geometry.hidden;
        let slack = (profile.tile_rows * wide * 2) as u64;
        assert!(
            profile.arena.span(ArenaField::HcStreams).bytes >= plan.windows[0].stream_bytes + slack,
            "the admitted arena holds the window's tagged spans plus one tile of view slack"
        );
        assert!(census.total().is_some());
    }

    #[test]
    fn slowlink_prompt_receipt_carries_every_required_key() {
        let plan = plan(0, 16384, &[1536]);
        let report = WindowReport {
            sweep_bytes: 50_803_507_200,
            stats: PanelStats {
                submitted_payload_bytes: 50_803_507_200,
                ..PanelStats::default()
            },
            ..WindowReport::default()
        };
        let receipt = layer_major_receipt(&ReceiptInputs {
            mode: SlowlinkMode::Force,
            plan: &plan,
            resident_layers: 10,
            host_layers: 38,
            tile_scratch_bytes: 5 << 30,
            reports: &[report],
            peak_vram_bytes: Some(22 << 30),
            status: "ok".to_string(),
        });
        for key in [
            "schema",
            "route",
            "selected_by",
            "bypass_reason",
            "entry_pos",
            "prompt_rows",
            "windows",
            "launch_cuts",
            "tile_rows",
            "retained_scratch_rows",
            "sweeps",
            "resident_layers",
            "host_layers",
            "host_expert_bytes_total",
            "submitted_bytes",
            "useful_bytes",
            "unused_prefetched_bytes",
            "reserve",
            "spans_ms",
            "overlap",
            "peak_vram_bytes",
        ] {
            assert!(receipt.get(key).is_some(), "receipt lacks {key}");
        }
        // The VRAM figure is one post-request sample, and the receipt says so:
        // never a continuous peak.
        assert_eq!(
            receipt["vram_measurement"],
            json!({"method": "hipMemGetInfo_post_request_sample", "continuous_peak": false})
        );
        assert_eq!(receipt["peak_vram_bytes"], 22u64 << 30);
        assert_eq!(receipt["schema"], RECEIPT_SCHEMA);
        assert_eq!(receipt["route"], "layer-major");
        assert_eq!(receipt["selected_by"], "env1");
        assert_eq!(
            receipt["sweeps"],
            receipt["windows"].as_array().unwrap().len()
        );
        assert!(receipt["retained_scratch_rows"].as_u64().unwrap() <= 1536);
        assert!(receipt["tile_rows"].as_u64().unwrap() <= 1536);
        assert_eq!(receipt["submitted_bytes"], 50_803_507_200u64);
        for key in ["tile", "arena", "ring", "lookahead", "tables", "total"] {
            assert!(receipt["reserve"].get(key).is_some(), "reserve lacks {key}");
        }
        for key in ["wall", "copy", "compute", "route_barrier", "nonexpert"] {
            assert!(
                receipt["spans_ms"].get(key).is_some(),
                "spans_ms lacks {key}"
            );
        }
        for key in [
            "cross_layer_intersections",
            "cross_layer_overlap_ms",
            "copy_idle_ms",
            "page_full_stalls",
            "copy_starved_stalls",
        ] {
            assert!(receipt["overlap"].get(key).is_some(), "overlap lacks {key}");
        }
        let total = receipt["reserve"]["total"].as_u64().unwrap();
        let parts: u64 = ["tile", "arena", "ring", "lookahead", "tables"]
            .iter()
            .map(|key| receipt["reserve"][key].as_u64().unwrap())
            .sum();
        assert!(total >= parts, "the charged total covers every named term");
    }

    #[test]
    fn slowlink_prompt_receipt_unused_bytes_include_untouched_experts_of_acquired_groups() {
        let plan = plan(0, 16384, &[1536]);
        let receipt_of = |reports: &[WindowReport]| {
            layer_major_receipt(&ReceiptInputs {
                mode: SlowlinkMode::Force,
                plan: &plan,
                resident_layers: 10,
                host_layers: 38,
                tile_scratch_bytes: 0,
                reports,
                peak_vram_bytes: None,
                status: "ok".to_string(),
            })
        };
        // Every group was acquired (the transport discarded nothing), yet the
        // routed rows read only part of the experts inside them: the rest of
        // the submitted payload was moved for nothing.
        let partial = WindowReport {
            sweep_bytes: 1000,
            useful_bytes: 640,
            stats: PanelStats {
                submitted_payload_bytes: 1000,
                discarded_payload_bytes: 0,
                ..PanelStats::default()
            },
            ..WindowReport::default()
        };
        assert_eq!(partial.unused_bytes(), Some(360));
        let receipt = receipt_of(&[partial, partial]);
        assert_eq!(receipt["submitted_bytes"], 2000u64);
        assert_eq!(receipt["useful_bytes"], 1280u64);
        assert_eq!(
            receipt["unused_prefetched_bytes"], 720u64,
            "submitted minus useful, even with the discard counter at zero"
        );
        assert_eq!(receipt["discarded_group_payload_bytes"], 0u64);
        // Wholly unused groups are part of the same difference, never added twice.
        let with_discard = WindowReport {
            useful_bytes: 640,
            stats: PanelStats {
                submitted_payload_bytes: 1000,
                discarded_payload_bytes: 200,
                ..PanelStats::default()
            },
            ..WindowReport::default()
        };
        let receipt = receipt_of(&[with_discard]);
        assert_eq!(receipt["unused_prefetched_bytes"], 360u64);
        assert_eq!(receipt["discarded_group_payload_bytes"], 200u64);
        // Fully used: nothing unused. Useful above submitted is a broken
        // measurement: the receipt reports it as missing, never underflows.
        let full = WindowReport {
            useful_bytes: 1000,
            ..partial
        };
        assert_eq!(receipt_of(&[full])["unused_prefetched_bytes"], 0u64);
        let broken = WindowReport {
            useful_bytes: 1001,
            ..partial
        };
        assert_eq!(broken.unused_bytes(), None);
        assert!(receipt_of(&[broken])["unused_prefetched_bytes"].is_null());
    }
}

#[cfg(test)]
mod slowlink_overlap {
    use super::*;
    use crate::gpu_forward::panel_stage::{
        ExpertExtent, ExpertSource, LayerExpertSources, PanelBackend, PanelCodec, PanelPoolSpec,
        RingCore, TimelineEvent,
    };
    use std::sync::Arc;

    const EXPERTS: usize = 512;
    const GU: usize = 256;
    const DN: usize = 256;
    const PAGES: usize = 18;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ev {
        Begin { slot: usize, wait_free: bool },
        End { slot: usize },
        Acquire { slot: usize },
        Release { slot: usize },
        Drain,
    }

    /// What the model device saw, in issue order (one log per test thread).
    #[derive(Default)]
    struct Log {
        events: Vec<Ev>,
        payload_bytes: u64,
    }

    thread_local! {
        static LOG: std::cell::RefCell<Log> = std::cell::RefCell::new(Log::default());
    }

    fn record(f: impl FnOnce(&mut Log)) {
        LOG.with(|log| f(&mut log.borrow_mut()));
    }

    /// A CPU model of the device: records stream operations in issue order.
    struct Model {
        tables: Vec<(Vec<u64>, Vec<u64>)>,
    }

    impl Model {
        fn new(pages: usize) -> Self {
            LOG.with(|log| *log.borrow_mut() = Log::default());
            Self {
                tables: (0..pages)
                    .map(|_| (vec![0; EXPERTS], vec![0; EXPERTS]))
                    .collect(),
            }
        }
    }

    impl PanelBackend for Model {
        type Ctx = ();

        fn page_base(&self, slot: usize) -> usize {
            0x4000_0000 + slot * 0x20_0000
        }
        fn tables_mut(&mut self, slot: usize) -> (&mut [u64], &mut [u64]) {
            let (gate_up, down) = &mut self.tables[slot];
            (gate_up.as_mut_slice(), down.as_mut_slice())
        }
        fn await_copy_done(&mut self, _: &(), _: usize) -> R<()> {
            Ok(())
        }
        fn copy_begin(&mut self, _: &(), slot: usize, wait_free: bool, _: Option<usize>) -> R<()> {
            record(|log| log.events.push(Ev::Begin { slot, wait_free }));
            Ok(())
        }
        fn copy_payload(&mut self, _: &(), _: usize, _: usize, _: usize, len: usize) -> R<()> {
            record(|log| log.payload_bytes += len as u64);
            Ok(())
        }
        fn copy_zero(&mut self, _: &(), _: usize, _: usize, _: usize) -> R<()> {
            Ok(())
        }
        fn copy_tables(&mut self, _: &(), _: usize) -> R<()> {
            Ok(())
        }
        fn copy_end(&mut self, _: &(), slot: usize, _: Option<usize>) -> R<()> {
            record(|log| log.events.push(Ev::End { slot }));
            Ok(())
        }
        fn compute_acquire(&mut self, _: &(), slot: usize, _: Option<usize>) -> R<bool> {
            record(|log| log.events.push(Ev::Acquire { slot }));
            Ok(false)
        }
        fn compute_release(&mut self, _: &(), slot: usize, _: Option<usize>) -> R<()> {
            record(|log| log.events.push(Ev::Release { slot }));
            Ok(())
        }
        fn mark_base(&mut self, _: &()) -> R<()> {
            Ok(())
        }
        fn mark_layer(&mut self, _: &(), _: usize, _: bool) -> R<()> {
            Ok(())
        }
        fn drain(&mut self, _: &()) -> R<()> {
            record(|log| log.events.push(Ev::Drain));
            Ok(())
        }
        fn elapsed_ms(&self, _: &(), _: TimelineEvent, _: TimelineEvent) -> R<f64> {
            Ok(0.0)
        }
    }

    /// The model's two transport views of the ring: the walker's `GroupRing`.
    struct CoreRing<'a> {
        core: &'a mut RingCore<Model>,
        schedule: &'a mut PanelSchedule,
    }

    impl GroupRing for CoreRing<'_> {
        type Ctx = ();

        fn require(&mut self, ctx: &(), layer: usize, group: u32) -> R<PanelTicket> {
            self.schedule.require(self.core, ctx, layer, group)
        }
        fn with_lease(
            &mut self,
            ctx: &mut (),
            ticket: PanelTicket,
            compute: &mut dyn FnMut(&mut (), Option<&PanelBindingParts<'_>>) -> R<()>,
        ) -> R<()> {
            self.core.acquire(&*ctx, ticket)?;
            compute(ctx, None)
        }
        fn retire(&mut self, ctx: &(), ticket: PanelTicket) -> R<()> {
            self.core.retire(ctx, ticket)
        }
        fn pump(&mut self, ctx: &()) -> R<usize> {
            self.schedule.pump(self.core, ctx)
        }
    }

    /// One pinned host arena all fixture layers read: expert `e`'s gate/up at
    /// `e * GU`, its down after the gate/up region.
    struct Arena {
        bytes: Vec<u8>,
    }

    impl Arena {
        fn new() -> Self {
            Self {
                bytes: (0..EXPERTS * (GU + DN) + 4096)
                    .map(|i| (i % 251) as u8)
                    .collect(),
            }
        }

        fn sources(&self, layer: usize) -> LayerExpertSources {
            let base = self.bytes.as_ptr() as usize;
            let experts = (0..EXPERTS)
                .map(|id| {
                    Some(ExpertSource {
                        codec: PanelCodec(0),
                        gate_up: ExpertExtent {
                            addr: base + id * GU,
                            len: GU,
                        },
                        down: ExpertExtent {
                            addr: base + EXPERTS * GU + id * DN,
                            len: DN,
                        },
                    })
                })
                .collect();
            // SAFETY: the extents lie inside `self.bytes`, which the test keeps
            // alive (and unmodified) for as long as the sources.
            unsafe { LayerExpertSources::new(layer, base, self.bytes.len(), experts, Arc::new(())) }
                .unwrap()
        }
    }

    fn spec() -> PanelPoolSpec {
        PanelPoolSpec::uniform(2, 16, 32, EXPERTS, GU, DN, 0, 0).unwrap()
    }

    fn fixture(host_layers: &[usize]) -> (Arena, RingCore<Model>, PanelSchedule) {
        let arena = Arena::new();
        let schedule =
            PanelSchedule::new(host_layers.iter().map(|&l| arena.sources(l)).collect(), 32)
                .unwrap();
        let core = RingCore::new(spec(), Model::new(PAGES)).unwrap();
        (arena, core, schedule)
    }

    fn work(
        schedule: &PanelSchedule,
        layer: usize,
        bound: impl Fn(usize) -> bool,
    ) -> Vec<GroupWork> {
        schedule
            .layer_groups(layer)
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, group)| GroupWork {
                group: group.group,
                bound: bound(index),
            })
            .collect()
    }

    /// The window start: lookahead covers the first host layer while the
    /// resident layers compute.
    #[test]
    fn slowlink_overlap_first_host_layer_prefetches_before_any_compute() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        let issued = schedule.prime(&mut core, &()).unwrap();
        assert_eq!(issued, 16, "the whole first host layer fits the page store");
        assert!(core
            .model_events()
            .iter()
            .all(|e| !matches!(e, Ev::Acquire { .. })));
        for group in 0..16u32 {
            assert!(schedule.ticket(10, group).is_some());
        }
        assert!(
            schedule.ticket(11, 0).is_none(),
            "nothing of layer 11 until compute is on layer 10"
        );
    }

    #[test]
    fn slowlink_overlap_next_layer_pages_enqueue_before_the_counts_barrier() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        schedule.prime(&mut core, &()).unwrap();
        // Compute reaches layer 10: its mixer and counts are still ahead, yet
        // the two pages the current layer does not hold already carry layer 11.
        schedule.enter_layer(&mut core, &(), 10).unwrap();
        assert_eq!(
            core.stats().enqueued,
            18,
            "16 current + 2 lookahead pages, no acquire yet"
        );
        assert_eq!(core.stats().acquired, 0);
        assert!(schedule.ticket(11, 0).is_some() && schedule.ticket(11, 1).is_some());
        assert!(schedule.ticket(11, 2).is_none(), "the page store is full");
        // Current-layer groups are enqueued before any next-layer group.
        let order: Vec<usize> = core
            .model_events()
            .iter()
            .filter_map(|e| match e {
                Ev::End { slot } => Some(*slot),
                _ => None,
            })
            .collect();
        assert_eq!(order.len(), 18);
    }

    #[test]
    fn slowlink_overlap_walk_retires_every_group_once_and_refills_in_order() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        schedule.prime(&mut core, &()).unwrap();
        let mut seen: Vec<(usize, usize)> = Vec::new();
        for layer in [10usize, 11, 12] {
            schedule.enter_layer(&mut core, &(), layer).unwrap();
            // Every third group is read by no panel: retired unread, never refetched.
            let work = work(&schedule, layer, |index| index % 3 != 0);
            let summary = walk_layer_groups(
                &mut CoreRing {
                    core: &mut core,
                    schedule: &mut schedule,
                },
                &mut (),
                layer,
                &work,
                &mut |_, index, parts| {
                    assert!(parts.is_none(), "the model has no device binding");
                    seen.push((layer, index));
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(summary.acquired + summary.discarded, 16);
            assert_eq!(summary.discarded, 6, "groups 0,3,6,9,12,15 are unread");
        }
        // Ascending (layer, group), each computed group exactly once.
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        assert_eq!(seen, sorted);
        assert_eq!(seen.len(), 3 * 10);
        let stats = core.stats();
        assert_eq!(
            stats.enqueued, 48,
            "one transfer per group per window, no refetch"
        );
        assert_eq!(stats.retired, 48, "every ticket of every layer retired");
        assert_eq!(stats.discarded, 18);
        assert_eq!(stats.acquired, 30);
        assert_eq!(stats.submitted_payload_bytes, schedule.sweep_bytes());
        assert_eq!(core.model_payload(), schedule.sweep_bytes());
    }

    #[test]
    fn slowlink_overlap_page_reuse_waits_for_the_last_reader() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11, 12]);
        schedule.prime(&mut core, &()).unwrap();
        for layer in [10usize, 11, 12] {
            schedule.enter_layer(&mut core, &(), layer).unwrap();
            let work = work(&schedule, layer, |_| true);
            walk_layer_groups(
                &mut CoreRing {
                    core: &mut core,
                    schedule: &mut schedule,
                },
                &mut (),
                layer,
                &work,
                &mut |_, _, _| Ok(()),
            )
            .unwrap();
        }
        // Per page the event order is Begin, End, Acquire, Release and every
        // refill after a Release has the copy stream wait on its free event.
        let events = core.model_events();
        for slot in 0..PAGES {
            let mut released = false;
            for event in events.iter().filter(|e| match e {
                Ev::Begin { slot: s, .. }
                | Ev::End { slot: s }
                | Ev::Acquire { slot: s }
                | Ev::Release { slot: s } => *s == slot,
                Ev::Drain => false,
            }) {
                match event {
                    Ev::Begin { wait_free, .. } => {
                        assert_eq!(
                            *wait_free, released,
                            "slot {slot}: a reuse waits the free event"
                        );
                    }
                    Ev::Release { .. } => released = true,
                    _ => {}
                }
            }
        }
        assert!(
            events.iter().any(|e| matches!(
                e,
                Ev::Begin {
                    wait_free: true,
                    ..
                }
            )),
            "pages were reused"
        );
    }

    #[test]
    fn slowlink_overlap_abort_drain_leaves_the_ring_reusable_and_tickets_stale() {
        let (_arena, mut core, mut schedule) = fixture(&[10, 11]);
        schedule.prime(&mut core, &()).unwrap();
        schedule.enter_layer(&mut core, &(), 10).unwrap();
        let stale = schedule.ticket(10, 0).unwrap();
        // The window fails after two groups: the abort path drains first.
        let work = work(&schedule, 10, |_| true);
        let mut count = 0;
        let error = walk_layer_groups(
            &mut CoreRing {
                core: &mut core,
                schedule: &mut schedule,
            },
            &mut (),
            10,
            &work,
            &mut |_, _, _| {
                count += 1;
                if count == 3 {
                    Err(invalid("injected failure during a panel"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("injected"));
        core.drain(&()).unwrap();
        assert!(core.model_events().contains(&Ev::Drain));
        assert!(
            core.acquire(&(), stale).is_err(),
            "an old ticket cannot reach a page that may hold other bytes"
        );
        // The next window starts clean: first host layer first, ascending.
        schedule.prime(&mut core, &()).unwrap();
        assert!(schedule.ticket(10, 0).is_some());
        assert_eq!(
            core.stats().enqueued,
            16,
            "statistics restart with the window"
        );
    }

    #[test]
    fn slowlink_overlap_binding_ranges_cover_the_layer_in_group_order() {
        let (_arena, _core, schedule) = fixture(&[10]);
        let groups = schedule.layer_groups(10).unwrap();
        let ranges = group_binding_ranges(groups, EXPERTS);
        assert_eq!(ranges.len(), groups.len());
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, EXPERTS);
        for (index, pair) in ranges.windows(2).enumerate() {
            assert_eq!(
                pair[0].1,
                pair[1].0,
                "ranges {index}/{} are contiguous",
                index + 1
            );
        }
        for (range, group) in ranges.iter().zip(groups) {
            assert!(group.experts.iter().all(|&e| e >= range.0 && e < range.1));
        }
    }

    #[test]
    fn slowlink_overlap_panels_group_by_ascending_binding() {
        let by_group = panels_by_group([0usize, 0, 1, 3, 3, 3].into_iter(), 4).unwrap();
        assert_eq!(by_group, vec![0..2, 2..3, 0..0, 3..6]);
        assert!(
            panels_by_group([1usize, 0].into_iter(), 2).is_err(),
            "bindings must ascend"
        );
        assert!(panels_by_group([0usize, 1, 0].into_iter(), 2).is_err());
        assert!(
            panels_by_group([2usize].into_iter(), 2).is_err(),
            "a panel names a real group"
        );
    }

    #[test]
    fn slowlink_overlap_resident_layers_have_no_scheduled_groups() {
        let (_arena, _core, schedule) = fixture(&[10, 12]);
        assert!(schedule.layer_groups(11).is_none());
        assert!(schedule.layer_groups(3).is_none());
        assert_eq!(schedule.layer_groups(10).unwrap().len(), 16);
    }

    trait ModelView {
        fn model_events(&self) -> Vec<Ev>;
        fn model_payload(&self) -> u64;
    }

    impl ModelView for RingCore<Model> {
        fn model_events(&self) -> Vec<Ev> {
            LOG.with(|log| log.borrow().events.clone())
        }
        fn model_payload(&self) -> u64 {
            LOG.with(|log| log.borrow().payload_bytes)
        }
    }
}
