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
use crate::gpu_forward::{Qwen4MeshHead, Qwen4MeshLayout};
use crate::mesh_bindings::{Qwen4MeshBindings, Qwen4RankBand};
use crate::program::qwen4_step_graph_with_output;
use crate::weights::Qwen4BandScope;
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind, MeshEpoch};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{split_program, SplitPlan};
use hipfire_runtime::multi_gpu::executor::ForwardEpoch;
use hipfire_runtime::multi_gpu::transport::EdgeId;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::weight_manifest::{plan_manifest, StateEntry, WeightEntry};
use std::sync::atomic::{AtomicU64, Ordering};

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

    fn stages(&self) -> usize {
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

fn device_sync(gpu: &mut Gpu) -> Result<(), String> {
    gpu.bind_thread().map_err(|error| error.to_string())?;
    gpu.hip
        .device_synchronize()
        .map_err(|error| error.to_string())
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
        let tiles = plan_tiles(tokens.len(), mesh.chunk);
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
            },
        )?;
        Ok(token)
    }

    /// Reset every band locally (the holder is detached, so each band's own
    /// `reset` runs), then synchronize every device. The transport leases are
    /// untouched: the next epoch's ids stay monotonic. Needs no daemon `Gpu`;
    /// errors when no mesh is attached.
    pub fn pp_reset(&mut self, gpus: &mut Gpus) -> Result<(), BundleError> {
        let mut mesh = self
            .pp
            .take()
            .ok_or_else(|| pp_error("no pipeline mesh is attached"))?;
        let result = self.pp_reset_detached(&mut mesh, gpus);
        self.pp = Some(mesh);
        result
    }

    /// Cold prompt prefill: reset once, then forward the whole prompt
    /// (`Final` rows on the explicit Prefill route). Session reuse is not
    /// admitted: `reused` must be 0 and the prompt non-empty.
    pub fn pp_prefill_final(
        &mut self,
        gpus: &mut Gpus,
        prompt: &[u32],
        reused: usize,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        if reused != 0 || prompt.is_empty() {
            return Err(pp_error(
                "prefill requires a non-empty prompt and no session reuse",
            ));
        }
        self.pp_reset(gpus)?;
        self.pp_forward(
            gpus,
            prompt,
            Some(logits),
            None,
            Qwen4OutputRows::Final,
            MeshRoute::Prefill,
        )
    }

    fn pp_reset_detached(
        &mut self,
        mesh: &mut Qwen4PpMesh,
        gpus: &mut Gpus,
    ) -> Result<(), BundleError> {
        mesh.require_ready(gpus)?;
        let stages = mesh.stages();
        // Every band resets and every device synchronizes even after a
        // failure; any failure quarantines the mesh.
        let mut failures: Vec<String> = Vec::new();
        for band in mesh.bands.iter_mut().chain(std::iter::once(&mut *self)) {
            if let Some(forward) = band.execution.as_mut() {
                forward.set_ple_lookahead(&[]);
                drop(forward.ple_ahead.take());
            }
        }
        for (index, band) in mesh.bands.iter_mut().enumerate() {
            if let Err(error) = band.reset(&mut gpus.devices[index]) {
                failures.push(format!("band {index} reset failed: {error}"));
            }
        }
        if let Err(error) = self.reset(&mut gpus.devices[stages - 1]) {
            failures.push(format!("band {} reset failed: {error}", stages - 1));
        }
        for (index, gpu) in gpus.devices.iter_mut().enumerate() {
            if let Err(error) = device_sync(gpu) {
                failures.push(format!("device {index} synchronization failed: {error}"));
            }
        }
        let next_request = mesh.request_id.checked_add(1);
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
        let Qwen4PpMesh { bands, .. } = *mesh;
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
