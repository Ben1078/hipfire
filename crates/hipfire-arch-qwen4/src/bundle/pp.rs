// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 pipeline-parallel (`topology.pp`) ownership and the bundle entry
//! points that route through it.
//!
//! The published bundle is the output band (the root). It owns the peer bands
//! `0..n-1`; the one [`Gpus`] that owns every logical device of the mesh stays
//! with the caller and is passed to every entry point. The root's own
//! `execution` is its forward; each peer's `execution` is its band's forward.
//! There is no family topology, thread pool or generation loop here: a call
//! splits the request into canonical tiles, submits them in order through the
//! generic executor, then drains the whole request once.
//!
//! * Every GPU model operation runs on `gpus.devices` (logical device `i` is
//!   band `i`). The daemon's `Gpu` is ignored for PP forwards.
//! * Generation logits/top-1 tensors are allocated by the daemon on the
//!   output band's physical device and are written directly by the output
//!   band. All tiles are drained (one all-device synchronization) before the
//!   call returns, so the daemon observes the rows complete.
//! * Tiles are the common canonical prefill chunk, in order, never padded;
//!   only the last tile computes the head for `Final` rows. Routing rows are
//!   the actual rows; the explicit [`MeshRoute`] is preserved.
//! * A submit or drain failure quarantines the mesh: it refuses all further
//!   work. A failed drain keeps every resource (including the root's) alive
//!   until process teardown.

use super::*;
use crate::config::Qwen4Config;
use crate::gpu_forward::{
    qwen4_prefill_segment_rows, Qwen4MeshHead, Qwen4MeshLayout, SessionBoundaries,
};
use crate::mesh_bindings::{session_tickets, Qwen4MeshBindings, Qwen4RankBand};
use crate::program::qwen4_step_graph_with_output;
use crate::weights::Qwen4BandScope;
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind, MeshEpoch};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{split_program, SplitPlan};
use hipfire_runtime::multi_gpu::executor::ForwardEpoch;
use hipfire_runtime::multi_gpu::transport::EdgeId;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::session_cache::{MeshSessionCache, MeshSessionRecord};
use hipfire_runtime::weight_manifest::{plan_manifest, StateEntry, WeightEntry};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

/// Band identity of one PP bundle, stamped on every band (root included) at
/// attach. It carries what a session scope and boundary schedule need while the
/// band's forward is taken or the mesh holder is detached: band index/count,
/// the band's layer range, the COMPLETE placement, the physical device of every
/// band, the mesh and load epoch, and the canonical prefill chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen4PpBandMeta {
    index: usize,
    layers: Range<usize>,
    placement: Vec<Range<usize>>,
    devices: Vec<String>,
    mesh_epoch: u64,
    load_id: u64,
    chunk: usize,
}

impl Qwen4PpBandMeta {
    /// `devices[i]` names band `i`'s physical device (`{device_id}@{arch}`).
    pub(crate) fn new(
        index: usize,
        placement: &BandPlacement,
        devices: Vec<String>,
        mesh_epoch: u64,
        load_id: u64,
        chunk: usize,
    ) -> Self {
        let bands = (0..placement.stages())
            .map(|stage| placement.band(stage))
            .collect();
        Self::from_bands(index, bands, devices, mesh_epoch, load_id, chunk)
    }

    fn from_bands(
        index: usize,
        placement: Vec<Range<usize>>,
        devices: Vec<String>,
        mesh_epoch: u64,
        load_id: u64,
        chunk: usize,
    ) -> Self {
        Self {
            index,
            layers: placement[index].clone(),
            placement,
            devices,
            mesh_epoch,
            load_id,
            chunk,
        }
    }

    /// The canonical prefill chunk every band of the mesh shares.
    pub(crate) fn chunk(&self) -> usize {
        self.chunk
    }

    /// The scope suffix that keeps this band's pieces from aliasing any other
    /// band, placement, device set, mesh generation or load.
    pub(crate) fn scope_tag(&self) -> String {
        let join = |parts: Vec<String>| parts.join(",");
        let placement = join(
            self.placement
                .iter()
                .map(|band| format!("{}-{}", band.start, band.end))
                .collect(),
        );
        format!(
            "/pp{}of{}/layers{}-{}/place{}/dev{}/mesh{}/load{}",
            self.index,
            self.placement.len(),
            self.layers.start,
            self.layers.end,
            placement,
            join(self.devices.clone()),
            self.mesh_epoch,
            self.load_id,
        )
    }
}

/// The canonical session schedule of one prefill, computed once by the root and
/// consumed identically by every band.
struct SessionRun<'r> {
    prompt: &'r [u32],
    route: SessionRoute,
    /// First prompt position this prefill computes (the restored prefix).
    start: usize,
    tiles: Vec<Tile>,
    /// Ascending snapshot boundaries; a tile ends on every one inside the
    /// prefill, and every band captures there.
    boundaries: Vec<usize>,
    /// The tail-split row among `boundaries`: captured as a band-local tail,
    /// never an aggregate record (the tail is not a pool piece).
    lineage: Option<usize>,
}

/// The cold schedule of a prefill over `start..end`, all tiles up front. Same
/// scheduler as Single's `prefill_schedule_rows` (`qwen4_prefill_segment_rows`,
/// no NPU pairs) driven by a virtual cursor over the full boundary list, since
/// the driver's own cursor only advances as boundaries are captured: tiles are
/// the global chunks cut at every boundary and the lineage split, never
/// re-chunked relative to the restored prefix.
fn prefill_schedule(
    start: usize,
    end: usize,
    chunk: usize,
    boundaries: &[usize],
    lineage: Option<usize>,
) -> Vec<Tile> {
    let chunk = chunk.max(1);
    let mut tiles = Vec::new();
    let mut position = start;
    while position < end {
        let stop = lineage
            .filter(|&split| split > position)
            .map_or(end, |split| split.min(end));
        let mut upcoming = boundaries.iter().copied().filter(|&at| at > position);
        let next = upcoming.next();
        let after = upcoming.next();
        let rows = qwen4_prefill_segment_rows(
            position,
            stop,
            chunk,
            SessionBoundaries { next, after },
            false,
        );
        tiles.push(Tile {
            offset: position - start,
            rows,
        });
        position += rows;
    }
    tiles
}

/// The pieces of an aggregate candidate captured at `end`: every band named a
/// cold pool piece and `end` is not the lineage (tail) split. `None` means no
/// record can ever cover this boundary, so none is staged.
fn aggregate_candidate<P>(
    lineage: Option<usize>,
    end: usize,
    pieces: Vec<Option<P>>,
) -> Option<Vec<P>> {
    if lineage == Some(end) {
        return None;
    }
    pieces.into_iter().collect()
}

/// The refusal of message-end snapshots under PP, the same sentence admission
/// gives (`qwen4_pp_unsupported`): attach and the setter use it verbatim.
pub(super) fn turn_snapshots_refusal(stages: usize) -> String {
    format!(
        "qwen4: pipeline parallelism (topology.pp = {stages}) does not support session snapshots (HIPFIRE_QWEN4_TURN_SNAPSHOTS); drop the option or set topology.pp = 1"
    )
}

/// Attach-time session verdict from per-band facts (index order): whether each
/// band holds a session cache and whether each has message-end snapshots on.
/// Message-end snapshots refuse; all bands cached = sessions on, none cached =
/// session-free; a mix refuses (a valid cache is never silently left unused).
fn attach_session_policy(
    stages: usize,
    present: &[bool],
    turn_snapshots: &[bool],
) -> Result<bool, String> {
    if turn_snapshots.iter().any(|&on| on) {
        return Err(turn_snapshots_refusal(stages));
    }
    let cached = present.iter().filter(|&&has| has).count();
    match cached {
        0 => Ok(false),
        n if n == present.len() => Ok(true),
        _ => Err(
            "the bands disagree on session-cache presence: every band attaches its own cache or none does"
                .to_string(),
        ),
    }
}

/// Band `index` of the mesh: the peers `0..n-1`, then the root.
fn band_mut<'a>(
    bands: &'a mut [Qwen4Bundle],
    root: &'a mut Qwen4Bundle,
    index: usize,
) -> &'a mut Qwen4Bundle {
    if index < bands.len() {
        &mut bands[index]
    } else {
        root
    }
}

/// Every band's session driver in owner order; `None` unless all have one.
fn session_drivers<'a>(
    bands: &'a mut [Qwen4Bundle],
    root: &'a mut Qwen4Bundle,
) -> Option<Vec<&'a mut SessionDriver>> {
    let mut drivers = Vec::with_capacity(bands.len() + 1);
    for bundle in bands.iter_mut().chain(std::iter::once(root)) {
        drivers.push(bundle.session.as_mut()?);
    }
    Some(drivers)
}

/// Borrowed rank bands over the taken forwards (peers then root).
fn mesh_ranks<'a>(
    bands: &'a mut [Qwen4Bundle],
    root: &'a mut Qwen4Bundle,
    forwards: &'a mut [Qwen4GpuForward],
) -> Vec<Qwen4RankBand<'a>> {
    let (peer_forwards, root_forward) = forwards.split_at_mut(bands.len());
    let mut ranks = Vec::with_capacity(bands.len() + 1);
    for (bundle, forward) in bands.iter_mut().zip(peer_forwards.iter_mut()) {
        ranks.push(Qwen4RankBand { bundle, forward });
    }
    ranks.push(Qwen4RankBand {
        bundle: root,
        forward: &mut root_forward[0],
    });
    ranks
}

/// The epoch a lifecycle call (reset, session restore) runs under: current mesh,
/// load and request. Its op id is not checked by those calls.
fn lifecycle_epoch(mesh: &Qwen4PpMesh, gpus: &Gpus, start_position: usize) -> ForwardEpoch {
    ForwardEpoch {
        mesh_epoch: gpus.mesh.epoch(),
        load_id: mesh.load_id,
        request_id: mesh.request_id,
        op_id: mesh.next_op,
        route: MeshRoute::Prefill,
        start_position,
        actual_rows: 0,
        route_rows: 0,
        policy_id: PP_POLICY_ID,
    }
}

/// Before any band maps context through `end`: while a band's growth does not
/// fit even after its cache released every unpinned snapshot, drop the oldest
/// aggregate record (releasing all its pins) so the band's own
/// `release_session_for_growth` can free those pieces. Once per horizon, never
/// clearing records that are not in the way: the cache stays opportunistic and
/// never holds context hostage.
fn reclaim_records_for_growth(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    end: usize,
) {
    if mesh.sessions.is_empty() {
        return;
    }
    let Qwen4PpMesh {
        bands, sessions, ..
    } = mesh;
    let stages = bands.len() + 1;
    loop {
        let mut short = false;
        for index in 0..stages {
            let bundle = band_mut(bands, root, index);
            let gpu = &mut gpus.devices[index];
            if gpu.bind_thread().is_err() || end <= bundle.state.mapped_context_tokens() {
                continue;
            }
            let (context, gather) = bundle.context_growth_bytes(gpu, end);
            let need = context.saturating_add(gather);
            let Some(session) = bundle.session.as_mut() else {
                continue;
            };
            if need == 0 || !session.needs_release(gpu, need) {
                continue;
            }
            session.release_for(gpu, need);
            short |= session.needs_release(gpu, need);
        }
        if !short {
            break;
        }
        let Some(mut drivers) = session_drivers(bands, root) else {
            break;
        };
        if !sessions.evict_oldest(&mut drivers) {
            break;
        }
    }
}

/// Numerical-policy identity of every PP epoch of this build.
const PP_POLICY_ID: u64 = 1;
const PP_MAX_BANDS: usize = 8;

/// Split-plan slots, one per output mode of a tile.
const MODE_TRUNK: usize = 0;
const MODE_LOGITS: usize = 1;
const MODE_LOGITS_TOP1: usize = 2;
const MODE_COUNT: usize = 3;

static NEXT_LOAD_ID: AtomicU64 = AtomicU64::new(1);

/// Next process-unique load id; fails instead of wrapping.
fn next_load_id() -> Result<u64, String> {
    NEXT_LOAD_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| "load ids are exhausted".to_string())
}

/// Why unload must leak every resource instead of freeing: a failed PLE
/// quiesce (host readers may still run) or a failed device drain (in-flight
/// work may still reference them). `None` = safe to free.
fn teardown_leak_reason(
    quiesce: &Option<BundleError>,
    drained: &Result<(), String>,
) -> Option<String> {
    let mut reasons = Vec::new();
    if let Some(error) = quiesce {
        reasons.push(format!("PLE quiesce failed: {error}"));
    }
    if let Err(error) = drained {
        reasons.push(format!("device drain failed: {error}"));
    }
    (!reasons.is_empty()).then(|| reasons.join("; "))
}

fn pp_error(message: impl AsRef<str>) -> BundleError {
    BundleError::Forward(format!("qwen4 topology.pp: {}", message.as_ref()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Health {
    Ready,
    /// Refuses all work; the reason is the first failure.
    Quarantined(String),
}

/// Attached PP bands, owned by the output-band root (`Qwen4Bundle::pp`). It
/// holds no hardware: every operation takes the caller-owned [`Gpus`]
/// explicitly. Nothing here frees a band's resources by dropping (unload is
/// explicit, see [`Qwen4Bundle::pp_free_gpu`]).
pub(crate) struct Qwen4PpMesh {
    /// Peer bands `0..n-1`; band `i` lives on `gpus.devices[i]`. Each has
    /// `execution: Some` between calls and `pp: None`.
    bands: Vec<Qwen4Bundle>,
    placement: BandPlacement,
    /// Split plans by output mode (`MODE_*`), built once at attach.
    plans: Vec<SplitPlan>,
    /// The common canonical prefill chunk of every band.
    chunk: usize,
    load_id: u64,
    request_id: u64,
    next_op: u64,
    /// Mesh identity of the `Gpus` this holder was validated against.
    mesh_epoch: Option<MeshEpoch>,
    health: Health,
    /// Aggregate session records over the per-band caches: owner-order piece
    /// keys and pins only, no device buffers. Cleared before any band's cache
    /// is cleared or freed.
    sessions: MeshSessionCache,
    /// Every band (root included) holds its own session cache, stamped with
    /// this mesh's load epoch at attach. Without it the mesh runs session-free.
    sessions_enabled: bool,
    /// The request id whose prefill completed with armed sessions and has not
    /// been committed or aborted: the only request whose staged candidates
    /// may publish.
    committable: Option<u64>,
}

impl Qwen4PpMesh {
    /// Healthy, and `gpus` is the mesh this holder was validated against.
    fn require_ready(&self, gpus: &Gpus) -> Result<(), BundleError> {
        if let Health::Quarantined(reason) = &self.health {
            return Err(pp_error(format!("mesh is quarantined: {reason}")));
        }
        if !self.identity_matches(gpus) {
            return Err(pp_error("the supplied devices are not this mesh's"));
        }
        Ok(())
    }

    /// `gpus` is the device set recorded at attach: same mesh epoch and
    /// device count. Health is not consulted (a failed-attach holder must
    /// still be releasable).
    fn identity_matches(&self, gpus: &Gpus) -> bool {
        self.mesh_epoch == Some(gpus.mesh.epoch()) && gpus.devices.len() == self.stages()
    }

    fn quarantine(&mut self, reason: String) {
        if self.health == Health::Ready {
            self.health = Health::Quarantined(reason);
        }
    }

    /// Whether every band has a stamped session cache.
    pub(super) fn sessions_enabled(&self) -> bool {
        self.sessions_enabled
    }

    pub(super) fn stages(&self) -> usize {
        self.bands.len() + 1
    }
}

/// One tile of a request: `rows` tokens from `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    offset: usize,
    rows: usize,
}

/// Canonical in-order tiling: full `chunk` tiles, then the remainder. Never
/// padded or rounded.
fn plan_tiles(len: usize, chunk: usize) -> Vec<Tile> {
    let mut tiles = Vec::with_capacity(len.div_ceil(chunk.max(1)));
    let mut offset = 0;
    while offset < len {
        let rows = (len - offset).min(chunk);
        tiles.push(Tile { offset, rows });
        offset += rows;
    }
    tiles
}

/// Output mode of a tile: `All` rows every tile with a head, `Final` only the
/// last; no logits tensor means no head at all.
fn tile_mode(has_logits: bool, has_top1: bool, rows: Qwen4OutputRows, last_tile: bool) -> usize {
    if !has_logits || (rows == Qwen4OutputRows::Final && !last_tile) {
        MODE_TRUNK
    } else if has_top1 {
        MODE_LOGITS_TOP1
    } else {
        MODE_LOGITS
    }
}

/// Unload ordering: quiesce PLE first, then drain every device. Returns the
/// quiesce error (if any) and the drain verdict. Nothing is freed here.
fn quiesce_then_drain(
    quiesce: impl FnOnce() -> Result<(), BundleError>,
    drain: impl FnOnce() -> Result<(), String>,
) -> (Option<BundleError>, Result<(), String>) {
    let quiesced = quiesce().err();
    (quiesced, drain())
}

fn combined_manifest(
    root: &Qwen4Bundle,
    bands: &[Qwen4Bundle],
) -> (Vec<WeightEntry>, Vec<StateEntry>) {
    let mut weights = Vec::new();
    let mut state = Vec::new();
    for band in bands.iter().chain(std::iter::once(root)) {
        let manifest = band.manifest();
        weights.extend(manifest.weights.iter().cloned());
        state.extend(manifest.state.iter().cloned());
    }
    (weights, state)
}

/// One plan per output mode over the real manifest declarations combined from
/// the rank bands.
fn build_plans(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    state: &[StateEntry],
    mesh: &DeviceMesh,
    placement: &BandPlacement,
) -> Result<Vec<SplitPlan>, String> {
    let manifest = plan_manifest(weights, state, mesh, config.num_hidden_layers, placement)?;
    let mut plans = Vec::with_capacity(MODE_COUNT);
    for (head, top1) in [(false, false), (true, false), (true, true)] {
        let graph = qwen4_step_graph_with_output(config, MeshRoute::Prefill, head, top1)?;
        plans.push(split_program(&graph, &manifest, mesh).map_err(|error| error.to_string())?);
    }
    Ok(plans)
}

/// Move every band's forward out of its bundle (peers in band order, root
/// last). On any absence the forwards already taken are put back.
fn take_forwards(
    bands: &mut [Qwen4Bundle],
    root: &mut Qwen4Bundle,
) -> Result<Vec<Qwen4GpuForward>, BundleError> {
    let peers = bands.len();
    let mut forwards = Vec::with_capacity(peers + 1);
    for index in 0..=peers {
        let bundle = if index < peers {
            &mut bands[index]
        } else {
            &mut *root
        };
        match bundle.execution.take() {
            Some(forward) => forwards.push(forward),
            None => {
                restore_forwards(bands, root, forwards);
                return Err(pp_error("a band's forward resources are not attached"));
            }
        }
    }
    Ok(forwards)
}

/// Inverse of [`take_forwards`]; accepts a contiguous prefix.
fn restore_forwards(
    bands: &mut [Qwen4Bundle],
    root: &mut Qwen4Bundle,
    forwards: Vec<Qwen4GpuForward>,
) {
    let mut forwards = forwards.into_iter();
    for band in bands.iter_mut() {
        if let Some(forward) = forwards.next() {
            band.execution = Some(forward);
        }
    }
    if let Some(forward) = forwards.next() {
        root.execution = Some(forward);
    }
}

/// A failure while tiles were (or may have been) in flight.
struct TileError {
    message: String,
    /// Quarantine the mesh: work was submitted or a drain failed.
    fatal: bool,
}

impl TileError {
    fn refused(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: false,
        }
    }
    fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
        }
    }
}

/// Per-call work shared by every tile.
struct Request<'r> {
    tokens: &'r [u32],
    logits: Option<&'r GpuTensor>,
    top1: Option<&'r GpuTensor>,
    rows: Qwen4OutputRows,
    route: MeshRoute,
    /// The canonical session schedule: boundaries to capture at and the tiles
    /// cut at them. `None` = no capture; tiles are chunk-relative.
    session: Option<&'r SessionRun<'r>>,
}

/// Compute every rank's layout for every tile and prepare all capacity before
/// any submission. Layouts are tile-major (`tile * stages + rank`).
fn prepare_tiles(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    forwards: &mut [Qwen4GpuForward],
    route: MeshRoute,
    tiles: &[Tile],
    start: usize,
) -> Result<Vec<Qwen4MeshLayout>, TileError> {
    let stages = mesh.stages();
    if let Some(last) = tiles.last() {
        reclaim_records_for_growth(root, mesh, gpus, start + last.offset + last.rows);
    }
    let mut layouts = Vec::with_capacity(tiles.len() * stages);
    for tile in tiles {
        let end = start + tile.offset + tile.rows;
        let mut agreed: Option<usize> = None;
        for rank in 0..stages {
            let bundle = if rank + 1 < stages {
                &mut mesh.bands[rank]
            } else {
                &mut *root
            };
            let gpu = &mut gpus.devices[rank];
            gpu.bind_thread()
                .map_err(|error| TileError::refused(error.to_string()))?;
            let forward = &mut forwards[rank];
            if forward.scratch.max_chunk != mesh.chunk {
                return Err(TileError::refused(
                    "a band's prefill chunk differs from the mesh's common chunk",
                ));
            }
            let layout = forward
                .mesh_layout(bundle, gpu, route, tile.rows)
                .map_err(|error| TileError::refused(error.to_string()))?;
            if layout.route_rows < tile.rows || agreed.is_some_and(|rows| rows != layout.route_rows)
            {
                return Err(TileError::refused("bands disagree on the routing rows"));
            }
            agreed = Some(layout.route_rows);
            forward
                .prepare_mesh_capacity(bundle, gpu, end, layout.route_rows)
                .map_err(|error| TileError::refused(error.to_string()))?;
            layouts.push(layout);
        }
    }
    Ok(layouts)
}

/// Submit every canonical tile in order, then acknowledge the request with one
/// all-device drain. Band positions/QSA marks are enqueue frontiers until that
/// drain succeeds; an error quarantines them, never acknowledges a partial
/// request or permits another request to consume the partial state.
/// Failed enqueue marks are not rolled back independently: the quarantined
/// mesh cannot reuse/publish them, and S4 recovery remains unload/reload.
///
/// Backpressure stays on the devices: `EdgeLease`'s two ready/done banks order
/// boundary copies, and each band's stream orders receive -> compute -> send
/// -> next receive, protecting its reusable device scratch.
///
/// Host inputs have separate, local lifetime fences, not mesh barriers:
/// * `forward_chunk_scoped` uploads token ids through `mesh_tokens.stage_buffer`
///   and `upload_async`; `PleHostStage::stage_buffer` drains the previous upload
///   fence before any overwrite (including a shorter tail).
/// * Its PLE upload copies the `RowFetch::wait` lease into `host_ple` through
///   the same fenced staging API before releasing the lease. The next tile may
///   reuse that host allocation only after its copy event completes.
/// * `ple_ahead` is a row-store ticket, not an alias of either upload buffer:
///   `RowFetch::begin_or_adopt` validates its epoch/ids, and the forward releases
///   its lease only after staging/upload. Lookahead never overwrites in-flight
///   host staging bytes. These fences must remain even without per-tile drains.
fn run_tiles(
    root: &mut Qwen4Bundle,
    mesh: &mut Qwen4PpMesh,
    gpus: &mut Gpus,
    forwards: &mut [Qwen4GpuForward],
    request: &Request<'_>,
    tiles: &[Tile],
    layouts: &[Qwen4MeshLayout],
) -> Result<(), TileError> {
    let stages = mesh.stages();
    let vocab = root.config.vocab_size;
    let count = tiles.len();
    let mut tickets = Vec::with_capacity(count);
    let mut submitted = false;
    let submitted_all = (|| {
        for (index, tile) in tiles.iter().enumerate() {
            let last = index + 1 == count;
            let mode = tile_mode(
                request.logits.is_some(),
                request.top1.is_some(),
                request.rows,
                last,
            );
            // Output views; they outlive the adapter that borrows them.
            let (logits_view, top1_view) = match (mode, request.logits) {
                (MODE_TRUNK, _) | (_, None) => (None, None),
                (_, Some(logits)) => {
                    let (logits_offset, top1_offset, rows) = match request.rows {
                        Qwen4OutputRows::All => (tile.offset * vocab, tile.offset * 4, tile.rows),
                        Qwen4OutputRows::Final => (0, 0, 1),
                    };
                    (
                        Some(logits.sub_offset(logits_offset, rows * vocab)),
                        request.top1.map(|top1| {
                            top1.sub_offset(top1_offset, rows * std::mem::size_of::<i32>())
                        }),
                    )
                }
            };
            let tile_layouts = &layouts[index * stages..(index + 1) * stages];
            let op_id = mesh.next_op;
            mesh.next_op = op_id
                .checked_add(1)
                .ok_or_else(|| TileError::refused("mesh operation ids are exhausted"))?;
            let epoch = ForwardEpoch {
                mesh_epoch: gpus.mesh.epoch(),
                load_id: mesh.load_id,
                request_id: mesh.request_id,
                op_id,
                route: request.route,
                start_position: root.state.position,
                actual_rows: tile.rows,
                route_rows: tile_layouts[0].route_rows,
                policy_id: PP_POLICY_ID,
            };
            let mut edge_epochs = Vec::with_capacity(stages - 1);
            for src in 0..stages - 1 {
                let id = EdgeId {
                    src_device: src,
                    dst_device: src + 1,
                };
                match gpus.mesh_edge_epoch(id) {
                    Some(epoch) if epoch != 0 => edge_epochs.push(epoch),
                    _ => return Err(TileError::refused("a band boundary has no transport lease")),
                }
            }
            let (peer_forwards, root_forward) = forwards.split_at_mut(stages - 1);
            let mut ranks = Vec::with_capacity(stages);
            for (bundle, forward) in mesh.bands.iter_mut().zip(peer_forwards.iter_mut()) {
                ranks.push(Qwen4RankBand { bundle, forward });
            }
            ranks.push(Qwen4RankBand {
                bundle: &mut *root,
                forward: &mut root_forward[0],
            });
            let mut bindings = Qwen4MeshBindings::new(ranks).map_err(TileError::refused)?;
            bindings.set_tokens(&request.tokens[tile.offset..tile.offset + tile.rows]);
            bindings.set_edge_epochs(edge_epochs);
            if let Some(run) = request.session {
                bindings.set_session(run.prompt, run.route);
            }
            for (owner, layout) in tile_layouts.iter().enumerate() {
                bindings
                    .set_rank_layout(owner, *layout)
                    .map_err(TileError::refused)?;
            }
            bindings.set_head(logits_view.as_ref().map(|logits| Qwen4MeshHead {
                logits,
                top1: top1_view.as_ref(),
                rows: request.rows,
            }));
            // Treat even a refused submit conservatively once it may have entered
            // a callback: a later tile's failure also invalidates all earlier work.
            submitted = true;
            tickets.push(
                gpus.submit_mesh_epoch(&mesh.plans[mode], &epoch, &mut bindings)
                    .map_err(|error| TileError::fatal(format!("submission failed: {error}")))?,
            );
            if let Some(run) = request.session {
                let end = run.start + tile.offset + tile.rows;
                if run.boundaries.binary_search(&end).is_ok() {
                    // Enqueued on every band's own stream behind its work for
                    // this tile; nothing drains. Pieces stay pending until the
                    // generation layer's commit publishes the aggregate record.
                    gpus.capture_mesh_session(&mut bindings, &epoch)
                        .map_err(|error| {
                            TileError::fatal(format!("session capture failed: {error}"))
                        })?;
                    let pieces = bindings.take_session_captures();
                    if let Some(pieces) = aggregate_candidate(run.lineage, end, pieces) {
                        mesh.sessions.stage(MeshSessionRecord {
                            route: run.route,
                            pieces,
                        });
                    }
                }
            }
        }
        Ok::<(), TileError>(())
    })();
    if let Err(error) = submitted_all {
        if !submitted {
            return Err(error);
        }
        // Also covers local validation/id exhaustion after earlier tiles were
        // enqueued. Stop all edges/devices before releasing borrowed resources;
        // the caller quarantines the request even when this drain succeeds.
        return Err(TileError::fatal(match gpus.drain_mesh_edges() {
            Ok(()) => error.message,
            Err(drain) => format!("{}; {drain}", error.message),
        }));
    }
    gpus.drain_mesh_epochs(tickets)
        .map_err(|error| TileError::fatal(format!("drain failed: {error}")))
}

impl Qwen4Bundle {
    /// Whether this bundle is the output-band root of an attached PP mesh
    /// (including one whose attach failed and is awaiting `free_gpu`).
    pub fn pp_mesh_active(&self) -> bool {
        self.pp.is_some()
    }

    /// Read-only local states in pipeline band order, including the output band.
    pub fn pp_band_states(&self) -> Option<impl Iterator<Item = &Qwen4State>> {
        self.pp.as_ref().map(|mesh| {
            mesh.bands
                .iter()
                .map(|band| &band.state)
                .chain(std::iter::once(&self.state))
        })
    }

    /// Adopt the mesh's hardware and peer bands. `self` is the output band.
    ///
    /// Ownership is installed before any validation: on `Err` the holder
    /// stays attached (quarantined), so the caller releases everything with
    /// [`Self::free_gpu`], which takes the PP teardown path.
    pub fn attach_pp_mesh(
        &mut self,
        gpus: &mut Gpus,
        bands: Vec<Qwen4Bundle>,
        placement: BandPlacement,
    ) -> Result<(), BundleError> {
        if self.pp.is_some() {
            // Ownership cannot be adopted twice; releasing either set would
            // free resources with the wrong devices.
            std::mem::forget(bands);
            return Err(pp_error("a pipeline mesh is already attached"));
        }
        let mut holder = Box::new(Qwen4PpMesh {
            bands,
            placement,
            plans: Vec::new(),
            chunk: 0,
            // Assigned during validation so id exhaustion fails closed.
            load_id: 0,
            request_id: 1,
            next_op: 1,
            // Recorded before validation so a failed attach is still
            // releasable against the device set it was given.
            mesh_epoch: Some(gpus.mesh.epoch()),
            health: Health::Quarantined("attach did not complete".to_string()),
            sessions: MeshSessionCache::new(),
            sessions_enabled: false,
            committable: None,
        });
        let verdict = self.validate_pp_attach(&mut holder, gpus);
        let result = match verdict {
            Ok(()) => {
                holder.health = Health::Ready;
                Ok(())
            }
            Err(message) => {
                holder.health = Health::Quarantined(message.clone());
                Err(pp_error(message))
            }
        };
        self.pp = Some(holder);
        result
    }

    fn validate_pp_attach(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
    ) -> Result<(), String> {
        let stages = mesh.placement.stages();
        let config = self.config.clone();
        if !(2..=PP_MAX_BANDS).contains(&stages)
            || mesh.bands.len() + 1 != stages
            || gpus.devices.len() != stages
            || gpus.mesh.n_devices() != stages
            || gpus.mesh.size_of(DimKind::Pp) != stages
            || gpus.mesh.size_of(DimKind::Tp) != 1
            || gpus.mesh.size_of(DimKind::Ep) != 1
        {
            return Err("the band count, device count and mesh shape disagree".into());
        }
        mesh.placement
            .validate_for(&gpus.mesh, config.num_hidden_layers)?;
        // Tiling and endpoint pins: every band owns exactly its placed band.
        let position = self.state.position;
        let max_seq = self.state.max_seq_len;
        let mut chunk = None;
        for (index, band) in mesh.bands.iter().chain(std::iter::once(&*self)).enumerate() {
            let scope = Qwen4BandScope::from_placement(&config, &mesh.placement, index)?;
            if band.weights.band_scope() != &scope {
                return Err("a band does not own the layers placed on it".into());
            }
            if band.config != config {
                return Err("the bands have different model configurations".into());
            }
            if band.ep.is_some() || band.lanes.is_some() || band.mtp.is_some() {
                return Err("expert parallelism, lanes and native MTP are not admitted".into());
            }
            if index + 1 < stages && band.pp.is_some() {
                return Err("a peer band carries its own pipeline mesh".into());
            }
            if band.state.position != position || band.state.max_seq_len != max_seq {
                return Err("the bands disagree on position or context capacity".into());
            }
            let forward = band
                .execution
                .as_ref()
                .ok_or("a band's forward resources are not attached")?;
            let band_chunk = forward.scratch.max_chunk;
            if band_chunk == 0 || chunk.is_some_and(|common| common != band_chunk) {
                return Err("the bands do not share one canonical prefill chunk".into());
            }
            chunk = Some(band_chunk);
        }
        mesh.load_id = next_load_id()?;
        mesh.mesh_epoch = Some(gpus.mesh.epoch());
        mesh.chunk = chunk.ok_or("no bands")?;
        // Band identity and session stamp: every band (root included) learns its
        // placement, devices and epochs now, so scopes and boundary chunks never
        // depend on a taken forward or the detached holder.
        let load_id = mesh.load_id;
        let mesh_epoch = gpus.mesh.epoch().as_u64();
        let devices: Vec<String> = gpus
            .devices
            .iter()
            .map(|gpu| format!("{}@{}", gpu.device_id, gpu.arch))
            .collect();
        let chunk = mesh.chunk;
        let present: Vec<bool> = mesh
            .bands
            .iter()
            .chain(std::iter::once(&*self))
            .map(|band| band.session.is_some())
            .collect();
        let turn_snapshots: Vec<bool> = mesh
            .bands
            .iter()
            .chain(std::iter::once(&*self))
            .map(|band| band.turn_snapshots)
            .collect();
        let sessions = attach_session_policy(stages, &present, &turn_snapshots)?;
        for (index, band) in mesh
            .bands
            .iter_mut()
            .chain(std::iter::once(&mut *self))
            .enumerate()
        {
            band.pp_band = Some(Qwen4PpBandMeta::new(
                index,
                &mesh.placement,
                devices.clone(),
                mesh_epoch,
                load_id,
                chunk,
            ));
            if let Some(session) = band.session.as_mut().filter(|_| sessions) {
                session
                    .bind_load_epoch(load_id)
                    .map_err(|error| format!("band {index} session cache: {error}"))?;
            }
        }
        mesh.sessions_enabled = sessions;
        // Prepared transport: the output owner and every boundary lease.
        if gpus.mesh_output_owner() != Some(stages - 1) {
            return Err("the mesh output owner is not the last band".into());
        }
        for src in 0..stages - 1 {
            let id = EdgeId {
                src_device: src,
                dst_device: src + 1,
            };
            if !matches!(gpus.mesh_edge_epoch(id), Some(epoch) if epoch != 0) {
                return Err("a band boundary has no transport lease".into());
            }
        }
        let (weights, state) = combined_manifest(self, &mesh.bands);
        mesh.plans = build_plans(&config, &weights, &state, &gpus.mesh, &mesh.placement)?;
        // Dry-run the adapter's ownership validation over the live bands.
        let forwards = take_forwards(&mut mesh.bands, self).map_err(|error| error.to_string())?;
        let mut forwards = forwards;
        let verdict = (|| {
            let (peer_forwards, root_forward) = forwards.split_at_mut(stages - 1);
            let mut ranks = Vec::with_capacity(stages);
            for (bundle, forward) in mesh.bands.iter_mut().zip(peer_forwards.iter_mut()) {
                ranks.push(Qwen4RankBand { bundle, forward });
            }
            ranks.push(Qwen4RankBand {
                bundle: &mut *self,
                forward: &mut root_forward[0],
            });
            Qwen4MeshBindings::new(ranks).map(|_| ())
        })();
        restore_forwards(&mut mesh.bands, self, forwards);
        verdict
    }

    /// Run `tokens` through the pipeline: tiles of the common canonical chunk
    /// in order, the head only where `rows`/`logits` ask for it. `logits` and
    /// `top1` are the daemon's tensors on the output band's physical device;
    /// they are complete when this returns.
    pub fn pp_forward(
        &mut self,
        gpus: &mut Gpus,
        tokens: &[u32],
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        rows: Qwen4OutputRows,
        route: MeshRoute,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_forward_detached(
            &mut mesh,
            gpus,
            &Request {
                tokens,
                logits,
                top1,
                rows,
                route,
                session: None,
            },
        );
        self.pp = Some(mesh);
        result
    }

    fn pp_forward_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        request: &Request<'_>,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let tokens = request.tokens;
        let vocab = self.config.vocab_size;
        if tokens.is_empty() {
            return Err(pp_error("cannot process an empty token slice"));
        }
        match request.route {
            MeshRoute::Verify => return Err(pp_error("verify is not admitted")),
            MeshRoute::Decode if tokens.len() != 1 => {
                return Err(pp_error("a decode step is exactly one token"));
            }
            _ => {}
        }
        if tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(pp_error("token id is outside the embedding vocabulary"));
        }
        let start = self.state.position;
        let end = start
            .checked_add(tokens.len())
            .filter(|&end| end <= self.state.max_seq_len)
            .ok_or_else(|| pp_error("the request exceeds the context capacity"))?;
        if mesh
            .bands
            .iter()
            .any(|band| band.state.position != start || band.state.max_seq_len < end)
        {
            return Err(pp_error(
                "the bands disagree on position or context capacity",
            ));
        }
        let out_rows = request.rows.count(tokens.len());
        match (request.logits, request.top1) {
            (None, Some(_)) => return Err(pp_error("top-1 rows require logits")),
            (Some(logits), top1) => {
                if logits.dtype != rdna_compute::DType::F32 || logits.numel() != out_rows * vocab {
                    return Err(pp_error(
                        "logits must be F32 with one vocabulary row per output row",
                    ));
                }
                if top1.is_some_and(|top1| {
                    top1.dtype != rdna_compute::DType::Raw
                        || top1.buf.size() < out_rows * std::mem::size_of::<i32>()
                }) {
                    return Err(pp_error("top-1 output must hold one i32 per output row"));
                }
            }
            (None, None) => {}
        }
        let tiles = match request.session {
            Some(run) => {
                if run.start != start
                    || run.tiles.iter().map(|tile| tile.rows).sum::<usize>() != tokens.len()
                    || run
                        .tiles
                        .iter()
                        .any(|tile| tile.rows == 0 || tile.rows > mesh.chunk)
                {
                    return Err(pp_error("the session schedule does not cover the request"));
                }
                run.tiles.clone()
            }
            None => plan_tiles(tokens.len(), mesh.chunk),
        };
        let mut forwards = take_forwards(&mut mesh.bands, self)?;
        let outcome = prepare_tiles(
            self,
            mesh,
            gpus,
            &mut forwards,
            request.route,
            &tiles,
            start,
        )
        .and_then(|layouts| run_tiles(self, mesh, gpus, &mut forwards, request, &tiles, &layouts));
        restore_forwards(&mut mesh.bands, self, forwards);
        outcome.map_err(|error| {
            if error.fatal {
                mesh.quarantine(error.message.clone());
            }
            pp_error(error.message)
        })
    }

    /// One token (`Some`) or the device argmax of `logits` as the previous
    /// forward left them (`None`); returns the token. Mirrors
    /// `Qwen4GpuForward::forward_token_or_argmax` on the output device.
    pub fn pp_forward_token_or_argmax(
        &mut self,
        gpus: &mut Gpus,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_token_or_argmax_detached(&mut mesh, gpus, token, logits);
        self.pp = Some(mesh);
        result
    }

    fn pp_token_or_argmax_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        mesh.require_ready(gpus)?;
        let token = match token {
            Some(token) => token,
            None => {
                let vocab = self.config.vocab_size;
                if logits.dtype != rdna_compute::DType::F32 || logits.numel() != vocab {
                    return Err(pp_error("logits must be one F32 vocabulary row"));
                }
                let owner = gpus
                    .mesh_output_owner()
                    .ok_or_else(|| pp_error("the mesh has no output owner"))?;
                let gpu = gpus
                    .devices
                    .get_mut(owner)
                    .ok_or_else(|| pp_error("the output device is unavailable"))?;
                gpu.bind_thread().map_err(BundleError::Hip)?;
                rdna_compute::tensor_ops::argmax_f32_host(gpu, logits).map_err(BundleError::Hip)?
            }
        };
        self.pp_forward_detached(
            mesh,
            gpus,
            &Request {
                tokens: std::slice::from_ref(&token),
                logits: Some(logits),
                top1: None,
                rows: Qwen4OutputRows::Final,
                route: MeshRoute::Decode,
                session: None,
            },
        )?;
        Ok(token)
    }

    /// Reset every band through the generic coordinator: drain every device,
    /// run each band's reset callback (bundle reset, uncommitted session
    /// captures dropped, PLE lookahead released), synchronize every device. The
    /// transport leases are untouched: the next epoch's ids stay monotonic. Any
    /// failure quarantines the mesh. Published aggregate records and their pins
    /// survive. Needs no daemon `Gpu`; errors when no mesh is attached.
    pub fn pp_reset(&mut self, gpus: &mut Gpus) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_reset_detached(&mut mesh, gpus);
        self.pp = Some(mesh);
        result
    }

    /// Prompt prefill of the full canonical `prompt`, keeping only the final
    /// logits row, on the explicit Prefill route. `reused == 0` resets the mesh
    /// (cold); `reused > 0` restores the aggregate record `pp_session_plan`
    /// chose through the mesh's session restore (every band probed before any
    /// is restored) and runs the rest. A missing or evicted piece starts cold;
    /// a failure once a restore began quarantines the mesh. Prompt, `reused`
    /// and `logits` are validated before any side effect.
    pub fn pp_prefill_final(
        &mut self,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_prefill_final_detached(&mut mesh, gpus, prompt, reused, logits);
        self.pp = Some(mesh);
        result
    }

    fn pp_prefill_final_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let vocab = self.config.vocab_size;
        if prompt.is_empty() || reused >= prompt.len() {
            return Err(pp_error(
                "prefill requires a non-empty prompt and a reused prefix shorter than it",
            ));
        }
        if prompt.iter().any(|&token| token as usize >= vocab) {
            return Err(pp_error("token id is outside the embedding vocabulary"));
        }
        if prompt.len() > self.state.max_seq_len {
            return Err(pp_error("the prompt exceeds the context capacity"));
        }
        if logits.dtype != rdna_compute::DType::F32 || logits.numel() != vocab {
            return Err(pp_error("logits must be one F32 vocabulary row"));
        }
        // Message-end snapshots stay refused; they never narrow sessions silently.
        if mesh.sessions_enabled && self.turn_snapshots_active() {
            return Err(BundleError::Forward(turn_snapshots_refusal(mesh.stages())));
        }
        let sessions = mesh.sessions_enabled;
        if reused != 0 && !sessions {
            return Err(pp_error(
                "session reuse needs a session cache on every band",
            ));
        }
        // Nothing of an earlier, uncommitted request may publish.
        mesh.sessions.abort();
        mesh.committable = None;
        let route = SessionRoute::Ar;
        if sessions {
            self.pp_sync_session_policy(mesh);
        }
        let mut start = 0;
        if reused != 0 {
            start = self.pp_restore_planned(mesh, gpus, prompt, reused, route)?;
        }
        if start == 0 {
            self.pp_reset_detached(mesh, gpus)?;
            if sessions {
                self.pp_arm_cold(mesh, gpus, prompt, route)?;
            }
        }
        let run = if sessions {
            Some(self.pp_session_run(mesh, prompt, route, start)?)
        } else {
            None
        };
        let result = self.pp_forward_detached(
            mesh,
            gpus,
            &Request {
                tokens: &prompt[start..],
                logits: Some(logits),
                top1: None,
                rows: Qwen4OutputRows::Final,
                route: MeshRoute::Prefill,
                session: run.as_ref(),
            },
        );
        match &result {
            Ok(()) if sessions => mesh.committable = Some(mesh.request_id),
            Ok(()) => {}
            Err(_) => mesh.sessions.abort(),
        }
        result
    }

    /// The root's tail/message-end settings are authoritative: every band
    /// adopts them before its scope or boundaries are consulted.
    fn pp_sync_session_policy(&self, mesh: &mut Qwen4PpMesh) {
        let policy = self.pp_session_policy();
        for band in &mut mesh.bands {
            band.pp_adopt_session_policy(policy);
        }
    }

    /// Restore the planned aggregate record for `reused` tokens, or decide to
    /// start cold (`Ok(0)`) when it is gone, stale or any band's piece is no
    /// longer resident (every owner is probed before the first restore; nothing
    /// is touched then). `Err` only after a restore began: the mesh is poisoned
    /// and quarantined.
    fn pp_restore_planned(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        route: SessionRoute,
    ) -> Result<usize, BundleError> {
        let stages = mesh.stages();
        // Other records' pins must not make the restore's context growth fail.
        reclaim_records_for_growth(self, mesh, gpus, prompt.len());
        let prefix = &prompt[..reused];
        let planned = match mesh.sessions.planned() {
            Some((len, record))
                if len == reused
                    && record.route == route
                    && record.pieces.len() == stages
                    && record.pieces.iter().enumerate().all(|(index, piece)| {
                        piece.matches_prefix(prefix)
                            && (if index + 1 < stages {
                                &mesh.bands[index]
                            } else {
                                &*self
                            })
                            .pp_session_piece_key(prefix, route)
                            .as_ref()
                                == Some(piece)
                    }) =>
            {
                Some(record.pieces.clone())
            }
            _ => None,
        };
        let Some(pieces) = planned else {
            eprintln!(
                "  qwen4 topology.pp: no valid session record for {reused} tokens; cold start"
            );
            self.pp_release_plan(mesh);
            return Ok(0);
        };
        let mut forwards = take_forwards(&mut mesh.bands, self)?;
        let epoch = lifecycle_epoch(mesh, gpus, reused);
        let (outcome, started) = {
            let ranks = mesh_ranks(&mut mesh.bands, self, &mut forwards);
            match Qwen4MeshBindings::new(ranks) {
                Err(error) => (Err(error), false),
                Ok(mut bindings) => {
                    bindings.set_session(prompt, route);
                    let tickets = session_tickets(&epoch, &pieces);
                    let outcome = bindings
                        .set_session_restore(pieces)
                        .and_then(|()| gpus.restore_mesh_session(&mut bindings, &epoch, &tickets));
                    let started = bindings.session_restore_started();
                    (outcome, started)
                }
            }
        };
        restore_forwards(&mut mesh.bands, self, forwards);
        match outcome {
            Ok(()) => Ok(reused),
            Err(error) if started => {
                mesh.quarantine(error.clone());
                Err(pp_error(error))
            }
            Err(error) => {
                eprintln!("  qwen4 topology.pp: session restore refused ({error}); cold start");
                self.pp_release_plan(mesh);
                Ok(0)
            }
        }
    }

    fn pp_release_plan(&mut self, mesh: &mut Qwen4PpMesh) {
        let Qwen4PpMesh {
            bands, sessions, ..
        } = mesh;
        if let Some(mut drivers) = session_drivers(bands, self) {
            sessions.release_plan(&mut drivers);
        }
    }

    /// Cold start of every band's session turn at 0, after the mesh reset.
    fn pp_arm_cold(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
        prompt: &[u32],
        route: SessionRoute,
    ) -> Result<(), BundleError> {
        let stages = mesh.stages();
        for index in 0..stages {
            let gpu = &mut gpus.devices[index];
            let armed = gpu
                .bind_thread()
                .map_err(|error| error.to_string())
                .and_then(|()| {
                    band_mut(&mut mesh.bands, self, index).pp_session_begin(gpu, prompt, route, 0)
                });
            if let Err(error) = armed {
                let message = format!("band {index} session start failed: {error}");
                mesh.quarantine(message.clone());
                return Err(pp_error(message));
            }
        }
        Ok(())
    }

    /// The canonical schedule: the root's armed boundaries (computed once) are
    /// installed on every band, and the tiles are cut at them.
    fn pp_session_run<'p>(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        prompt: &'p [u32],
        route: SessionRoute,
        start: usize,
    ) -> Result<SessionRun<'p>, BundleError> {
        let (boundaries, lineage) = self
            .pp_session_schedule()
            .ok_or_else(|| pp_error("the output band has no armed session schedule"))?;
        for band in &mut mesh.bands {
            band.pp_session_set_schedule(boundaries.clone(), lineage)
                .map_err(pp_error)?;
        }
        let tiles = prefill_schedule(start, prompt.len(), mesh.chunk, &boundaries, lineage);
        Ok(SessionRun {
            prompt,
            route,
            start,
            tiles,
            boundaries,
            lineage,
        })
    }

    /// The longest published prefix of `prompt` every band holds, strictly
    /// shorter than the prompt (0 on a miss, a refused option or an unhealthy
    /// mesh). Only the aggregate record plans: never the root's own cache or
    /// live state. The chosen record keeps its pins until the prefill restores
    /// it or the plan is abandoned.
    pub fn pp_session_plan(
        &mut self,
        gpus: &mut Gpus,
        prompt: &[u32],
        route: SessionRoute,
    ) -> usize {
        let Some(mut mesh) = self.pp.take() else {
            return 0;
        };
        let planned = self.pp_session_plan_detached(&mut mesh, gpus, prompt, route);
        self.pp = Some(mesh);
        planned
    }

    fn pp_session_plan_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &Gpus,
        prompt: &[u32],
        route: SessionRoute,
    ) -> usize {
        if !mesh.sessions_enabled
            || self.turn_snapshots_active()
            || mesh.require_ready(gpus).is_err()
        {
            return 0;
        }
        self.pp_sync_session_policy(mesh);
        let stages = mesh.stages();
        let planned = {
            let Qwen4PpMesh {
                bands, sessions, ..
            } = &mut *mesh;
            match session_drivers(bands, self) {
                Some(mut drivers) => sessions.plan(&mut drivers, prompt, route),
                None => 0,
            }
        };
        if planned == 0 {
            return 0;
        }
        // A record published under other schedule settings names keys this
        // scope would not find: it is not a hit.
        let prefix = &prompt[..planned];
        let current = mesh.sessions.planned().is_some_and(|(len, record)| {
            len == planned
                && record.pieces.len() == stages
                && record.pieces.iter().enumerate().all(|(index, piece)| {
                    (if index + 1 < stages {
                        &mesh.bands[index]
                    } else {
                        &*self
                    })
                    .pp_session_piece_key(prefix, route)
                    .as_ref()
                        == Some(piece)
                })
        });
        if !current {
            self.pp_release_plan(mesh);
            return 0;
        }
        planned
    }

    /// Publish the request's aggregate record(s) after the generation layer's
    /// commit point, keeping each band's live state when `consumed` extends the
    /// turn. Host-only. Publishes only a completed prefill of the current
    /// request on a healthy mesh; otherwise the staged candidates are dropped
    /// and nothing publishes on any band.
    pub fn pp_session_commit_live(&mut self, gpus: &mut Gpus, consumed: &[u32]) {
        self.pp_session_commit_with(gpus, Some(consumed));
    }

    /// [`Self::pp_session_commit_live`] without keeping any live state.
    pub fn pp_session_commit(&mut self, gpus: &mut Gpus) {
        self.pp_session_commit_with(gpus, None);
    }

    fn pp_session_commit_with(&mut self, gpus: &Gpus, consumed: Option<&[u32]>) {
        let Some(mut mesh) = self.pp.take() else {
            return;
        };
        let valid = mesh.sessions_enabled
            && mesh.require_ready(gpus).is_ok()
            && mesh.committable == Some(mesh.request_id);
        mesh.committable = None;
        if valid {
            for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *self)) {
                band.pp_session_commit_local(consumed);
            }
            let Qwen4PpMesh {
                bands, sessions, ..
            } = &mut *mesh;
            match session_drivers(bands, self) {
                Some(mut drivers) => {
                    sessions.publish(&mut drivers);
                }
                None => sessions.abort(),
            }
        } else {
            mesh.sessions.abort();
        }
        self.pp = Some(mesh);
    }

    /// Drop every aggregate record (releasing all pins) and then every band's
    /// session snapshots; the pins go first so no band's clear can leave a
    /// stale pin. Bands are cleared only on a healthy mesh (a quarantined one
    /// may still have work in flight).
    pub fn pp_session_clear(&mut self, gpus: &mut Gpus) {
        let Some(mut mesh) = self.pp.take() else {
            return;
        };
        mesh.committable = None;
        {
            let Qwen4PpMesh {
                bands, sessions, ..
            } = &mut *mesh;
            match session_drivers(bands, self) {
                Some(mut drivers) => sessions.clear(&mut drivers),
                None => sessions.abort(),
            }
        }
        if mesh.sessions_enabled && mesh.require_ready(gpus).is_ok() {
            for index in 0..mesh.stages() {
                let gpu = &mut gpus.devices[index];
                if gpu.bind_thread().is_ok() {
                    if let Some(session) = band_mut(&mut mesh.bands, self, index).session.as_mut() {
                        session.clear(gpu);
                    }
                }
            }
        }
        self.pp = Some(mesh);
    }

    fn pp_reset_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let mut forwards = take_forwards(&mut mesh.bands, self)?;
        let epoch = lifecycle_epoch(mesh, gpus, self.state.position);
        let outcome = {
            match Qwen4MeshBindings::new(mesh_ranks(&mut mesh.bands, self, &mut forwards)) {
                Ok(mut bindings) => Ok(gpus.reset_mesh(&mut bindings, &epoch)),
                Err(error) => Err(error),
            }
        };
        restore_forwards(&mut mesh.bands, self, forwards);
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => return Err(pp_error(error)),
        };
        let next_request = mesh.request_id.checked_add(1);
        let mut failures: Vec<String> = Vec::new();
        if let Err(error) = outcome {
            failures.push(error);
        }
        if next_request.is_none() {
            failures.push("request ids are exhausted".to_string());
        }
        if !failures.is_empty() {
            // Captured graphs and the replay proof window describe state that
            // no longer matches the bands.
            for gpu in &mut gpus.devices {
                gpu.invalidate_graph_state();
                gpu.replay.invalidate_replay_observation_window();
            }
            let message = failures.join("; ");
            mesh.quarantine(message.clone());
            return Err(pp_error(message));
        }
        mesh.request_id = next_request.unwrap_or(mesh.request_id);
        // Uncommitted candidates belong to the request that just ended.
        mesh.sessions.abort();
        mesh.committable = None;
        Ok(())
    }

    /// `apply_penalty_table` on the output device with the root's staging.
    pub fn pp_apply_penalty_table(
        &mut self,
        gpus: &mut Gpus,
        logits: &GpuTensor,
        table: &PenaltyTable,
    ) -> Result<(), BundleError> {
        if table.is_noop() {
            return Ok(());
        }
        let vocab = self.config.vocab_size;
        let mesh = self
            .pp
            .as_mut()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        mesh.require_ready(gpus)?;
        let owner = gpus
            .mesh_output_owner()
            .ok_or_else(|| pp_error("the mesh has no output owner"))?;
        let gpu = gpus
            .devices
            .get_mut(owner)
            .ok_or_else(|| pp_error("the output device is unavailable"))?;
        let applied = gpu.bind_thread().map_err(BundleError::Hip).and_then(|()| {
            gpu.apply_penalty_table(
                &mut self.penalty_stage,
                logits,
                vocab,
                table.flags(),
                table.row_ends(),
                table.entries(),
            )
            .map_err(BundleError::Hip)
        });
        applied?;
        // The daemon reads on its own stream of this physical device. A failed
        // synchronization leaves the staged penalty work in an unknown state.
        if let Err(error) = gpu.hip.device_synchronize() {
            let message = format!("penalty synchronization failed: {error}");
            mesh.quarantine(message.clone());
            return Err(pp_error(message));
        }
        Ok(())
    }

    /// Mapped QSA context bytes of every band on its own device, MTP = 0.
    pub fn pp_qsa_context_committed_bytes(
        &self,
        gpus: &Gpus,
    ) -> Result<(usize, usize), BundleError> {
        let mesh = self
            .pp
            .as_ref()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        if !mesh.identity_matches(gpus) {
            return Err(pp_error("the supplied devices are not this mesh's"));
        }
        let mut total = 0usize;
        for (index, band) in mesh.bands.iter().chain(std::iter::once(self)).enumerate() {
            let bytes = band
                .state
                .mapped_context_bytes(&gpus.devices[index])
                .map_err(BundleError::State)?;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| pp_error("context byte count overflows"))?;
        }
        Ok((total, 0))
    }

    /// Unload: quiesce PLE, drain every device, then free the peers and the
    /// root on their own devices of the caller's `gpus` (which the caller
    /// still owns and drops last). If the quiesce or the drain fails nothing
    /// is freed: the root and the holder are leaked with `mem::forget`
    /// (in-flight work or host readers may still reference them) and the
    /// caller must forget its `Gpus` too.
    pub fn pp_free_gpu(mut self, gpus: &mut Gpus) -> Result<(), BundleError> {
        let Some(mesh) = self.pp.take() else {
            std::mem::forget(self);
            return Err(pp_error("no pipeline mesh is attached"));
        };
        // The caller's `Gpus` must be the one this holder was attached with;
        // nothing is drained or freed on the wrong hardware.
        if !mesh.identity_matches(gpus) {
            std::mem::forget(self);
            std::mem::forget(mesh);
            return Err(pp_error(
                "unload was given a different device set; the mesh resources stay allocated until process exit",
            ));
        }
        let (quiesce_error, drained) = {
            let bands = &mesh.bands;
            let root = &self;
            let devices = &mut *gpus;
            quiesce_then_drain(
                || {
                    let mut first = None;
                    for band in bands.iter().chain(std::iter::once(root)) {
                        if let Err(error) = band.quiesce_ple() {
                            first.get_or_insert(error);
                        }
                    }
                    first.map_or(Ok(()), Err)
                },
                || devices.drain_mesh_edges(),
            )
        };
        let stages = mesh.stages();
        // A PLE quiesce failure may leave host readers or enqueuers alive, so
        // it forces the same leak as a failed drain, even when the GPU drain
        // succeeded. The caller then forgets `gpus` too.
        if let Some(reason) = teardown_leak_reason(&quiesce_error, &drained) {
            std::mem::forget(self);
            std::mem::forget(mesh);
            return Err(pp_error(format!(
                "unload could not release the mesh; its resources stay allocated until process exit: {reason}"
            )));
        }
        let Qwen4PpMesh {
            mut bands,
            mut sessions,
            ..
        } = *mesh;
        // Aggregate pins go before any band's cache is freed.
        if let Some(mut drivers) = session_drivers(&mut bands, &mut self) {
            sessions.clear(&mut drivers);
        }
        let mut first = None;
        for (index, band) in bands.into_iter().enumerate() {
            if let Err(error) = Qwen4Bundle::free_gpu(band, &mut gpus.devices[index]) {
                first.get_or_insert(error);
            }
        }
        if let Err(error) = Qwen4Bundle::free_gpu(self, &mut gpus.devices[stages - 1]) {
            first.get_or_insert(error);
        }
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn tiles_are_canonical_in_order_and_unpadded() {
        assert_eq!(plan_tiles(1, 4096), [Tile { offset: 0, rows: 1 }]);
        assert_eq!(
            plan_tiles(9, 4),
            [
                Tile { offset: 0, rows: 4 },
                Tile { offset: 4, rows: 4 },
                Tile { offset: 8, rows: 1 },
            ]
        );
        assert_eq!(plan_tiles(8, 4).len(), 2);
        assert!(plan_tiles(0, 4).is_empty());
        let total: usize = plan_tiles(10_001, 1536).iter().map(|tile| tile.rows).sum();
        assert_eq!(total, 10_001);
    }

    #[test]
    fn head_is_computed_only_where_requested() {
        use Qwen4OutputRows::{All, Final};
        // Final rows: only the last tile carries the head.
        assert_eq!(tile_mode(true, false, Final, false), MODE_TRUNK);
        assert_eq!(tile_mode(true, false, Final, true), MODE_LOGITS);
        assert_eq!(tile_mode(true, true, Final, true), MODE_LOGITS_TOP1);
        // All rows: every tile.
        assert_eq!(tile_mode(true, false, All, false), MODE_LOGITS);
        assert_eq!(tile_mode(true, true, All, false), MODE_LOGITS_TOP1);
        // No logits tensor: trunk only, even on the last tile.
        assert_eq!(tile_mode(false, false, Final, true), MODE_TRUNK);
        assert_eq!(tile_mode(false, false, All, true), MODE_TRUNK);
    }

    #[test]
    fn quarantine_is_sticky_and_keeps_the_first_reason() {
        let mut health = Health::Ready;
        assert_eq!(health, Health::Ready);
        let reason = "drain failed".to_string();
        if health == Health::Ready {
            health = Health::Quarantined(reason.clone());
        }
        if health == Health::Ready {
            health = Health::Quarantined("later".to_string());
        }
        assert_eq!(health, Health::Quarantined(reason));
    }

    #[test]
    fn unload_quiesces_before_draining_and_reports_both() {
        let order = RefCell::new(Vec::new());
        let (quiesce, drained) = quiesce_then_drain(
            || {
                order.borrow_mut().push("quiesce");
                Ok(())
            },
            || {
                order.borrow_mut().push("drain");
                Ok(())
            },
        );
        assert!(quiesce.is_none() && drained.is_ok());
        assert_eq!(*order.borrow(), ["quiesce", "drain"]);

        // A failed drain is surfaced (the caller then leaks every resource);
        // a failed quiesce does not skip the drain.
        let order = RefCell::new(Vec::new());
        let (quiesce, drained) = quiesce_then_drain(
            || {
                order.borrow_mut().push("quiesce");
                Err(pp_error("quiesce"))
            },
            || {
                order.borrow_mut().push("drain");
                Err("device lost".to_string())
            },
        );
        assert!(quiesce.is_some());
        assert_eq!(drained, Err("device lost".to_string()));
        assert_eq!(*order.borrow(), ["quiesce", "drain"]);
    }
}

#[cfg(test)]
mod teardown_tests {
    use super::*;

    #[test]
    fn quiesce_failure_retains_resources_even_when_the_drain_succeeds() {
        let (quiesce, drained) = quiesce_then_drain(|| Err(pp_error("timed out")), || Ok(()));
        assert!(drained.is_ok());
        let reason = teardown_leak_reason(&quiesce, &drained).expect("must leak");
        assert!(reason.contains("PLE quiesce failed"));
        assert!(!reason.contains("drain failed"));
    }

    #[test]
    fn drain_failure_retains_resources_and_reports_both_causes() {
        let (quiesce, drained) = quiesce_then_drain(
            || Err(pp_error("timed out")),
            || Err("device lost".to_string()),
        );
        let reason = teardown_leak_reason(&quiesce, &drained).expect("must leak");
        assert!(reason.contains("PLE quiesce failed") && reason.contains("device lost"));
        let (quiesce, drained) = quiesce_then_drain(|| Ok(()), || Err("device lost".to_string()));
        assert!(teardown_leak_reason(&quiesce, &drained).is_some());
    }

    #[test]
    fn clean_quiesce_and_drain_frees() {
        let (quiesce, drained) = quiesce_then_drain(|| Ok(()), || Ok(()));
        assert!(teardown_leak_reason(&quiesce, &drained).is_none());
    }

    #[test]
    fn load_ids_never_wrap() {
        assert!(next_load_id().unwrap() >= 1);
        assert_eq!(u64::MAX.checked_add(1), None);
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use std::collections::HashSet;

    const FORMAT: Qwen4StateFormat = Qwen4StateFormat::F32;

    fn meta(
        index: usize,
        bands: &[Range<usize>],
        devices: &[&str],
        mesh_epoch: u64,
        load_id: u64,
    ) -> Qwen4PpBandMeta {
        Qwen4PpBandMeta::from_bands(
            index,
            bands.to_vec(),
            devices.iter().map(|device| device.to_string()).collect(),
            mesh_epoch,
            load_id,
            8192,
        )
    }

    fn policy() -> PpSessionPolicy {
        PpSessionPolicy {
            turn_snapshots: false,
            tail_snapshot: true,
            tail_snapshot_min_gap: 1024,
            tail_snapshot_min_prompt: 16384,
            turn_end_token: Some(9),
        }
    }

    fn scope(band: &Qwen4PpBandMeta, policy: PpSessionPolicy, chunk: usize) -> String {
        session_scope(SessionRoute::Ar, chunk, FORMAT, false, Some((band, policy)))
    }

    #[test]
    fn band_scopes_differ_by_band_placement_device_epoch_load_and_policy() {
        let even = [0..24, 24..48];
        let skew = [0..20, 20..48];
        let devices = ["0@gfx1100", "1@gfx1100"];
        let base = meta(0, &even, &devices, 5, 1);
        let tail_off = PpSessionPolicy {
            tail_snapshot: false,
            ..policy()
        };
        let other_gap = PpSessionPolicy {
            tail_snapshot_min_gap: 1,
            ..policy()
        };
        let no_token = PpSessionPolicy {
            turn_end_token: None,
            ..policy()
        };
        let scopes = [
            scope(&base, policy(), 8192),
            // another band of the same mesh
            scope(&meta(1, &even, &devices, 5, 1), policy(), 8192),
            // same band index, another placement (even the same layer count of
            // band 0 differs in the complete placement)
            scope(&meta(0, &skew, &devices, 5, 1), policy(), 8192),
            scope(
                &meta(0, &[0..24, 24..47, 47..48], &devices, 5, 1),
                policy(),
                8192,
            ),
            // another physical device set / order / arch
            scope(
                &meta(0, &even, &["1@gfx1100", "0@gfx1100"], 5, 1),
                policy(),
                8192,
            ),
            scope(
                &meta(0, &even, &["0@gfx1100", "1@gfx1201"], 5, 1),
                policy(),
                8192,
            ),
            // another mesh generation and another load
            scope(&meta(0, &even, &devices, 6, 1), policy(), 8192),
            scope(&meta(0, &even, &devices, 5, 2), policy(), 8192),
            // another canonical chunk
            scope(&base, policy(), 4096),
            // the shared schedule settings
            scope(&base, tail_off, 8192),
            scope(&base, other_gap, 8192),
            scope(&base, no_token, 8192),
        ];
        let distinct: HashSet<&String> = scopes.iter().collect();
        assert_eq!(distinct.len(), scopes.len(), "{scopes:#?}");
        let digests: HashSet<_> = scopes
            .iter()
            .map(|scope| hipfire_runtime::serve_contract::sha256_len_prefixed(&[scope.as_bytes()]))
            .collect();
        assert_eq!(digests.len(), scopes.len(), "PP scope digests must differ");
    }

    #[test]
    fn band_scope_is_deterministic_and_carries_the_band_identity() {
        let even = [0..24, 24..48];
        let devices = ["0@gfx1100", "1@gfx1100"];
        let band = meta(1, &even, &devices, 5, 1);
        let first = scope(&band, policy(), 8192);
        assert_eq!(first, scope(&band.clone(), policy(), 8192));
        assert!(first.contains("/pp1of2/layers24-48/place0-24,24-48/"));
        assert!(first.contains("/dev0@gfx1100,1@gfx1100/mesh5/load1"));
        assert_eq!(band.chunk(), 8192);
    }

    #[test]
    fn single_session_scope_is_unchanged() {
        let historical = |turns: &str| {
            format!(
                "qwen4/{:?}/chunk{}/{:?}{turns}",
                SessionRoute::Ar,
                8192,
                FORMAT
            )
        };
        assert_eq!(
            session_scope(SessionRoute::Ar, 8192, FORMAT, false, None),
            historical("")
        );
        assert_eq!(
            session_scope(SessionRoute::Ar, 8192, FORMAT, true, None),
            historical("/turns")
        );
        assert_ne!(
            session_scope(SessionRoute::Mtp, 8192, FORMAT, false, None),
            historical("")
        );
        // A PP scope extends, never rewrites, the Single one.
        let band = meta(0, &[0..24, 24..48], &["0@a", "1@a"], 1, 1);
        assert!(scope(&band, policy(), 8192).starts_with(&historical("")));
    }

    fn ends(tiles: &[Tile], start: usize) -> Vec<usize> {
        tiles
            .iter()
            .map(|tile| start + tile.offset + tile.rows)
            .collect()
    }

    #[test]
    fn schedule_cuts_at_every_boundary_and_the_offgrid_lineage_then_realigns() {
        // Regular boundaries, an off-grid lineage split (20), then the next
        // global chunk boundary (24, 32): no capture skipped, no tile
        // re-chunked relative to the split.
        let boundaries = [8, 16, 20, 24, 32];
        let tiles = prefill_schedule(0, 30, 8, &boundaries, Some(20));
        assert_eq!(ends(&tiles, 0), [8, 16, 20, 24, 30]);
        assert_eq!(
            tiles.iter().map(|tile| tile.offset).collect::<Vec<_>>(),
            [0, 8, 16, 20, 24]
        );
        // A warm start continues the same global schedule.
        let warm = prefill_schedule(16, 30, 8, &boundaries, Some(20));
        assert_eq!(ends(&warm, 16), [20, 24, 30]);
    }

    #[test]
    fn schedule_without_boundaries_is_chunk_relative_and_never_padded() {
        assert_eq!(ends(&prefill_schedule(0, 20, 8, &[], None), 0), [8, 16, 20]);
        assert!(prefill_schedule(5, 5, 8, &[], None).is_empty());
        // An off-grid start still lands on the next global chunk boundary.
        let tiles = prefill_schedule(5, 20, 8, &[8, 16], None);
        assert_eq!(ends(&tiles, 5), [8, 16, 20]);
        assert_eq!(tiles[0], Tile { offset: 0, rows: 3 });
        // A lineage past the prompt end changes nothing.
        assert_eq!(
            prefill_schedule(0, 30, 8, &[8, 16, 24], Some(40)),
            prefill_schedule(0, 30, 8, &[8, 16, 24], None)
        );
    }

    #[test]
    fn every_boundary_inside_the_prefill_is_a_tile_end_exactly_once() {
        let chunk = 8;
        for start in 0..12usize {
            for end in start + 1..41 {
                for lineage in std::iter::once(None).chain((start + 1..end).map(Some)) {
                    let mut boundaries: Vec<usize> = (start / chunk + 1..=end / chunk)
                        .map(|k| k * chunk)
                        .collect();
                    boundaries.extend(lineage);
                    boundaries.sort_unstable();
                    boundaries.dedup();
                    let tiles = prefill_schedule(start, end, chunk, &boundaries, lineage);
                    let ends = ends(&tiles, start);
                    assert_eq!(ends.last(), Some(&end));
                    assert!(ends.windows(2).all(|pair| pair[0] < pair[1]));
                    assert!(tiles.iter().all(|tile| (1..=chunk).contains(&tile.rows)));
                    for boundary in boundaries.iter().filter(|&&at| at <= end) {
                        assert_eq!(
                            ends.iter().filter(|&&at| at == *boundary).count(),
                            1,
                            "start {start} end {end} lineage {lineage:?} boundary {boundary}"
                        );
                    }
                    // Offsets are contiguous from the first computed row.
                    let mut next = 0;
                    for tile in &tiles {
                        assert_eq!(tile.offset, next);
                        next += tile.rows;
                    }
                    assert_eq!(next, end - start);
                }
            }
        }
    }

    #[test]
    fn aggregate_candidates_need_a_cold_piece_on_every_band() {
        assert_eq!(
            aggregate_candidate(None, 8, vec![Some(1), Some(2)]),
            Some(vec![1, 2])
        );
        // One band named no cold piece: no record can cover this boundary.
        assert_eq!(aggregate_candidate(None, 8, vec![Some(1), None]), None);
        assert_eq!(aggregate_candidate::<u8>(None, 8, vec![None, None]), None);
        // The lineage (tail) split is band-local, never an aggregate record.
        assert_eq!(
            aggregate_candidate(Some(8), 8, vec![Some(1), Some(2)]),
            None
        );
        assert_eq!(
            aggregate_candidate(Some(8), 16, vec![Some(1), Some(2)]),
            Some(vec![1, 2])
        );
    }
}

#[cfg(test)]
mod attach_policy_tests {
    use super::*;

    #[test]
    fn all_or_none_session_caches_are_admitted() {
        assert_eq!(
            attach_session_policy(2, &[true, true], &[false, false]),
            Ok(true)
        );
        assert_eq!(
            attach_session_policy(3, &[true, true, true], &[false; 3]),
            Ok(true)
        );
        assert_eq!(
            attach_session_policy(2, &[false, false], &[false, false]),
            Ok(false)
        );
    }

    #[test]
    fn mixed_cache_presence_fails_closed_for_any_band_order() {
        for present in [[true, false], [false, true]] {
            let error = attach_session_policy(2, &present, &[false, false]).unwrap_err();
            assert!(
                error.contains("disagree on session-cache presence"),
                "{error}"
            );
        }
        assert!(attach_session_policy(3, &[true, false, true], &[false; 3]).is_err());
    }

    #[test]
    fn message_end_snapshots_refuse_with_the_admission_sentence() {
        // On any band, with or without caches.
        for turns in [[true, false], [false, true], [true, true]] {
            for present in [[true, true], [false, false], [true, false]] {
                let error = attach_session_policy(2, &present, &turns).unwrap_err();
                assert_eq!(error, turn_snapshots_refusal(2));
            }
        }
        assert_eq!(
            turn_snapshots_refusal(2),
            "qwen4: pipeline parallelism (topology.pp = 2) does not support session snapshots (HIPFIRE_QWEN4_TURN_SNAPSHOTS); drop the option or set topology.pp = 1"
        );
    }
}
