// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq

//! Default-off, single-threaded execution of an admitted split schedule.
//!
//! Preparation is CPU-only and validates the entire epoch before any submission.
//! Submission never synchronizes HIP. Local marks are enqueued frontiers, not a
//! committed model position: only a successful all-device drain acknowledges an
//! epoch. A failed drain permanently quarantines the mesh's leases.

use super::transport::{self, EdgeDescriptor, EdgeId, EdgeLease, EdgeSpec};
use super::Gpus;
use crate::device_mesh::{DimKind, MeshEpoch};
use crate::mesh_placement::{AdmittedMesh, BandPlacement, EdgeTransport};
use crate::mesh_split::{CollectiveKind, PlannedCollective, RankItem, SplitPlan};
use hipfire_dispatch::pipeline::step_graph::OpId;
use rdna_compute::{Gpu, GpuTensor};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

pub type Result<T, E = String> = std::result::Result<T, E>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshRoute {
    Decode,
    Prefill,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardEpoch {
    pub mesh_epoch: MeshEpoch,
    pub load_id: u64,
    pub request_id: u64,
    pub op_id: u64,
    pub route: MeshRoute,
    pub start_position: usize,
    pub actual_rows: usize,
    pub route_rows: usize,
    pub policy_id: u64,
}

/// Borrowed local activation, never an alias on another rank. `descriptor.epoch`
/// is the provisioned transport epoch returned by `Gpus::mesh_edge_epoch`.
pub struct BoundaryIo<'a> {
    pub view: &'a GpuTensor,
    pub descriptor: EdgeDescriptor,
}

/// Family adapters own state, not GPUs, streams, topology or token decisions.
/// All device callbacks must enqueue on the owner's active stream without host
/// waits. `prepare_epoch` only validates already-prepared capacity and shapes.
pub trait MeshStepBindings {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()>;
    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        ops: &[OpId],
    ) -> Result<()>;
    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>>;
    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()>;
    fn repair_prefix_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        keep: usize,
        tokens: &[u32],
    ) -> Result<()>;
    fn capture_session_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId>;
    fn restore_session_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        checkpoint: MeshCheckpointId,
    ) -> Result<()>;
    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, epoch: &ForwardEpoch) -> Result<()>;
    /// Capture this owner's local state on its borrowed `Gpu`. Must return a
    /// ticket minted with `MeshCheckpointId::issue(*key, owner, ..)`.
    fn checkpoint_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId>;
    /// Roll this owner back to `ticket` on its borrowed `Gpu`.
    fn restore_rank(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;
    /// Host-only fallible pre-commit check; must not mutate state or touch a `Gpu`.
    fn validate_finalize_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;
    /// Host-only commit/retire of this owner's ticket. Infallible: no GPU
    /// submission, no copies, no fallible allocation.
    fn finalize_rank(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId);
}

/// Generation key shared by every ticket of one checkpoint window. A ticket is
/// only valid for the exact `(mesh, load, reset, request, window)` it was
/// issued under; any bump invalidates every outstanding ticket.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MeshCheckpointKey {
    pub mesh_epoch: MeshEpoch,
    pub load_gen: u64,
    pub reset_gen: u64,
    pub request_gen: u64,
    pub window_gen: u64,
}

/// Opaque bounded (`Copy`, fixed-size) ticket for one owner's local checkpoint.
///
/// The generic runtime never interprets `local`; families mint it with
/// [`MeshCheckpointId::issue`] and read it back with [`MeshCheckpointId::local`].
/// The owner index is part of the identity so a ticket issued by one owner is
/// never accepted for another (no rank-zero ticket broadcast to all stages).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct MeshCheckpointId {
    key: MeshCheckpointKey,
    owner: usize,
    local: u64,
}

impl MeshCheckpointId {
    /// Mint a ticket for `owner` under `key`. `local` is family-opaque.
    pub fn issue(key: MeshCheckpointKey, owner: usize, local: u64) -> Self {
        Self { key, owner, local }
    }

    pub fn key(&self) -> MeshCheckpointKey {
        self.key
    }

    pub fn owner(&self) -> usize {
        self.owner
    }

    /// Family-local opaque value; meaningful only to the issuing owner.
    pub fn local(&self) -> u64 {
        self.local
    }
}

/// Aggregate record: one ticket per trunk owner (index == owner) plus the
/// optional output-head ticket, whose owner index is `per_owner.len()`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MeshCheckpoint {
    pub per_owner: Vec<MeshCheckpointId>,
    pub head: Option<MeshCheckpointId>,
}

impl MeshCheckpoint {
    /// Callback owner index used for the output head.
    pub fn head_owner(&self) -> usize {
        self.per_owner.len()
    }

    /// Check every ticket's generation key and owner index against `key`.
    /// Pure; runs before any callback in `restore_all`/`finalize_all`.
    pub fn check_shape(&self, key: &MeshCheckpointKey) -> Result<(), MeshLifecycleError> {
        if self.per_owner.is_empty() {
            return Err(MeshLifecycleError::shape(
                0,
                "checkpoint has no trunk owners",
            ));
        }
        let head_owner = self.head_owner();
        let slots = self
            .per_owner
            .iter()
            .enumerate()
            .chain(self.head.iter().map(|id| (head_owner, id)));
        for (owner, id) in slots {
            if id.owner != owner {
                return Err(MeshLifecycleError::shape(
                    owner,
                    "ticket belongs to a different owner",
                ));
            }
            if id.key != *key {
                return Err(MeshLifecycleError::shape(
                    owner,
                    "ticket generation does not match the live window (stale or foreign)",
                ));
            }
        }
        Ok(())
    }
}

/// Which lifecycle step produced a [`MeshLifecycleError`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshLifecyclePhase {
    /// A ticket set was malformed/stale; no callback ran and no rollback was possible.
    Shape,
    Checkpoint,
    /// `validate_finalize_rank` failed; no `finalize_rank` ran.
    Validate,
    Restore,
}

/// Aggregate lifecycle failure.
///
/// `owner`/`cause` are the first failure that aborted the operation.
/// `restore_failures` lists every owner whose rollback `restore_rank` also
/// failed (empty when rollback was clean). When `poisoned()` is true the
/// request/model state is not known-good and the caller must poison it.
#[derive(Clone, Debug)]
pub struct MeshLifecycleError {
    pub phase: MeshLifecyclePhase,
    pub owner: usize,
    pub cause: String,
    pub restore_failures: Vec<(usize, String)>,
}

impl MeshLifecycleError {
    fn shape(owner: usize, msg: &str) -> Self {
        Self {
            phase: MeshLifecyclePhase::Shape,
            owner,
            cause: msg.to_string(),
            restore_failures: Vec::new(),
        }
    }

    /// True when rollback could not make every owner consistent again: some
    /// restore failed, or the tickets were unusable so none was attempted.
    pub fn poisoned(&self) -> bool {
        self.phase == MeshLifecyclePhase::Shape || !self.restore_failures.is_empty()
    }
}

impl fmt::Display for MeshLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mesh checkpoint {:?} failed at owner {}: {}",
            self.phase, self.owner, self.cause
        )?;
        if !self.restore_failures.is_empty() {
            write!(f, "; rollback failed on owner(s)")?;
            for (owner, err) in &self.restore_failures {
                write!(f, " {owner} ({err})")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for MeshLifecycleError {}

/// Pure per-owner lifecycle callbacks sequenced by [`checkpoint_all`],
/// [`restore_all`] and [`finalize_all`]. Object-safe and CPU-testable: it holds
/// no `Gpu` and is deliberately separate from [`MeshStepBindings`], whose
/// device-touching methods receive the owner's borrowed `&mut Gpu`. A callback
/// provider resolves logical owner indices to its locally borrowed resources;
/// the head's logical owner is distinct even when it shares the output GPU.
///
/// `owner` is `0..trunk_owners` for trunk stages and `trunk_owners` for the
/// output head. `key` is the live window's generation key.
pub trait MeshCheckpointCallbacks {
    /// Capture this owner's local state. Must return a ticket minted with
    /// `MeshCheckpointId::issue(*key, owner, ..)`. On `Err` the callee leaves
    /// its own state unmodified/rolled back; the coordinator will not restore it.
    fn checkpoint_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId>;

    /// Roll this owner back to `ticket`. Called for every owner on any failure,
    /// in reverse owner order, even if earlier restores failed.
    fn restore_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;

    /// Fallible pre-commit check (ticket still valid, captures drained, ...).
    /// Runs for every owner before any `finalize_rank`. Must not mutate state.
    fn validate_finalize_rank(
        &mut self,
        owner: usize,
        key: &MeshCheckpointKey,
        ticket: MeshCheckpointId,
    ) -> Result<()>;

    /// Host-only commit/retire of this owner's ticket. Infallible: no GPU
    /// submission, no copies, no fallible allocation.
    fn finalize_rank(&mut self, owner: usize, key: &MeshCheckpointKey, ticket: MeshCheckpointId);
}

/// Roll back `tickets` (slot index == owner; `head` at `tickets.len()`) in
/// reverse order. Attempts every slot; returns every failure.
fn restore_slots<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    tickets: &[MeshCheckpointId],
    head: Option<MeshCheckpointId>,
) -> Vec<(usize, String)> {
    let mut failures = Vec::new();
    if let Some(id) = head {
        if let Err(e) = cb.restore_rank(tickets.len(), key, id) {
            failures.push((tickets.len(), e));
        }
    }
    for (owner, id) in tickets.iter().enumerate().rev() {
        if let Err(e) = cb.restore_rank(owner, key, *id) {
            failures.push((owner, e));
        }
    }
    failures
}

/// Capture every trunk owner (ascending) and, when `with_head`, the output
/// head as owner `trunk_owners`. If any capture fails (or returns a ticket not
/// issued for that owner/key), every already-captured owner is restored in
/// reverse order and the original failure is returned.
pub fn checkpoint_all<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    trunk_owners: usize,
    with_head: bool,
) -> Result<MeshCheckpoint, MeshLifecycleError> {
    if trunk_owners == 0 {
        return Err(MeshLifecycleError::shape(
            0,
            "checkpoint needs at least one trunk owner",
        ));
    }
    let capture = |cb: &mut C, owner: usize| -> Result<MeshCheckpointId> {
        let id = cb.checkpoint_rank(owner, key)?;
        if id.owner != owner || id.key != *key {
            return Err(
                "checkpoint_rank returned a ticket not issued for this owner/window".to_string(),
            );
        }
        Ok(id)
    };

    let mut per_owner = Vec::with_capacity(trunk_owners);
    for owner in 0..trunk_owners {
        match capture(cb, owner) {
            Ok(id) => per_owner.push(id),
            Err(cause) => {
                let restore_failures = restore_slots(cb, key, &per_owner, None);
                return Err(MeshLifecycleError {
                    phase: MeshLifecyclePhase::Checkpoint,
                    owner,
                    cause,
                    restore_failures,
                });
            }
        }
    }
    let head = if with_head {
        match capture(cb, trunk_owners) {
            Ok(id) => Some(id),
            Err(cause) => {
                let restore_failures = restore_slots(cb, key, &per_owner, None);
                return Err(MeshLifecycleError {
                    phase: MeshLifecyclePhase::Checkpoint,
                    owner: trunk_owners,
                    cause,
                    restore_failures,
                });
            }
        }
    } else {
        None
    };
    Ok(MeshCheckpoint { per_owner, head })
}

/// Restore EVERY owner (head first, then trunk descending), continuing past
/// individual failures. Tickets are validated for shape before any callback.
/// The first failing owner is `owner`/`cause`; all failures are in
/// `restore_failures`.
pub fn restore_all<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    checkpoint: MeshCheckpoint,
) -> Result<(), MeshLifecycleError> {
    checkpoint.check_shape(key)?;
    let failures = restore_slots(cb, key, &checkpoint.per_owner, checkpoint.head);
    if failures.is_empty() {
        return Ok(());
    }
    let (owner, cause) = failures[0].clone();
    // `restore_failures` carries every failure including the primary one.
    Err(MeshLifecycleError {
        phase: MeshLifecyclePhase::Restore,
        owner,
        cause,
        restore_failures: failures,
    })
}

/// Commit the checkpoint. Phase 1 validates EVERY owner (trunk ascending, then
/// head), continuing past failures and reporting the first; phase 2 runs the infallible
/// `finalize_rank` on every owner in the same order. If validation fails, no
/// finalize runs and EVERY owner is restored (reverse order, continuing past
/// restore errors); the validation failure is returned with any rollback
/// failures attached. The success path performs no allocation.
pub fn finalize_all<C: MeshCheckpointCallbacks + ?Sized>(
    cb: &mut C,
    key: &MeshCheckpointKey,
    checkpoint: MeshCheckpoint,
) -> Result<(), MeshLifecycleError> {
    checkpoint.check_shape(key)?;
    let head_owner = checkpoint.head_owner();

    // Validate EVERY owner (trunk ascending, head last) even after a failure;
    // remember the first failure. No finalize runs unless all validate.
    let mut failure: Option<(usize, String)> = None;
    for (owner, id) in checkpoint.per_owner.iter().enumerate() {
        if let Err(e) = cb.validate_finalize_rank(owner, key, *id) {
            failure.get_or_insert((owner, e));
        }
    }
    if let Some(id) = checkpoint.head {
        if let Err(e) = cb.validate_finalize_rank(head_owner, key, id) {
            failure.get_or_insert((head_owner, e));
        }
    }
    if let Some((owner, cause)) = failure {
        let restore_failures = restore_slots(cb, key, &checkpoint.per_owner, checkpoint.head);
        return Err(MeshLifecycleError {
            phase: MeshLifecyclePhase::Validate,
            owner,
            cause,
            restore_failures,
        });
    }

    for (owner, id) in checkpoint.per_owner.iter().enumerate() {
        cb.finalize_rank(owner, key, *id);
    }
    if let Some(id) = checkpoint.head {
        cb.finalize_rank(head_owner, key, id);
    }
    Ok(())
}

/// Acknowledgement capability. Not cloneable and bound to one `Gpus` instance.
#[derive(Debug)]
pub struct MeshTicket {
    issuer: u64,
    epoch: ForwardEpoch,
}

struct OwnedEdge {
    lease: EdgeLease,
    max_bytes: usize,
}

#[derive(Default)]
pub(super) struct MeshExecutionState {
    edges: HashMap<EdgeId, OwnedEdge>,
    mesh_epoch: Option<MeshEpoch>,
    output_owner: Option<usize>,
    issuer: u64,
    last_op: Option<u64>,
    scope: Option<(u64, u64)>,
    /// Successfully issued, not yet drained epochs, oldest first. They all
    /// belong to one request and one mesh epoch with strictly increasing op
    /// ids (`admit`). A failed issue never appears here: it poisons the state.
    outstanding: Vec<ForwardEpoch>,
    poisoned: bool,
    quarantined: bool,
}

impl MeshExecutionState {
    fn poison(&mut self) {
        self.poisoned = true;
        for edge in self.edges.values() {
            edge.lease.poison();
        }
    }

    /// Pure admission check; mutates nothing. A new epoch may join undrained
    /// earlier epochs only for the same request (mesh epoch, load and numerical
    /// policy are pinned by `mesh_epoch`/`scope`) with a strictly larger op id.
    fn admit(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.mesh_epoch != Some(epoch.mesh_epoch) || self.poisoned || self.quarantined {
            return Err("mesh epoch is stale, busy, poisoned, or unprovisioned".into());
        }
        if self
            .outstanding
            .last()
            .is_some_and(|last| last.request_id != epoch.request_id)
        {
            return Err("mesh epoch is busy with another request's undrained epochs".into());
        }
        if self.last_op.is_some_and(|last| epoch.op_id <= last)
            || self
                .scope
                .is_some_and(|scope| scope != (epoch.load_id, epoch.policy_id))
        {
            return Err(
                "mesh operation id is not monotonic or load/numerical policy changed".into(),
            );
        }
        Ok(())
    }

    /// Side-effect-free ticket validation. The tickets must be exactly the
    /// outstanding epochs, oldest first, issued by this state: a device drain
    /// retires every outstanding epoch, so a subset, reordering, stale or
    /// foreign ticket is refused before any device is touched.
    fn validate_tickets(&self, tickets: &[MeshTicket]) -> Result<()> {
        if self.quarantined
            || tickets.is_empty()
            || tickets.len() != self.outstanding.len()
            || tickets
                .iter()
                .zip(&self.outstanding)
                .any(|(ticket, epoch)| ticket.issuer != self.issuer || ticket.epoch != *epoch)
        {
            return Err(
                "mesh tickets do not name exactly the outstanding healthy transport epochs in order"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundaryMeta {
    descriptor: EdgeDescriptor,
    address: usize,
    extent: usize,
    capacity: usize,
    transport_epoch: u64,
}

/// Small internal seam: CPU tests exercise the very same prepare/issue/drain
/// algorithms as HIP, without constructing fake `Gpu` values.
trait DrainBackend {
    fn rank_count(&self) -> usize;
    fn synchronize(&mut self, owner: usize) -> Result<()>;
    fn poison(&mut self);
    fn quarantine(&mut self);
}

trait Backend: DrainBackend {
    fn ready(&self, owner: usize) -> bool;
    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryMeta>;
    fn run_ops(&mut self, owner: usize, epoch: &ForwardEpoch, ops: &[OpId]) -> Result<()>;
    fn transfer(
        &mut self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
        meta: BoundaryMeta,
        publish: bool,
    ) -> Result<()>;
    fn marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()>;
    /// Host-side validation hook run before any enqueue of `epoch`.
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()>;
}

#[derive(Debug)]
enum Action {
    Ops {
        owner: usize,
        start: usize,
        end: usize,
    },
    Transfer {
        owner: usize,
        collective: usize,
        meta: BoundaryMeta,
        publish: bool,
    },
    Marks(usize),
}

#[derive(Debug)]
struct Prepared {
    actions: Vec<Action>,
    ops: Vec<OpId>,
    issued: Vec<u64>,
}

fn endpoints(collective: &PlannedCollective, ranks: usize) -> Result<(usize, usize)> {
    let CollectiveKind::BandXfer { src, dst } = collective.kind else {
        return Err("mesh executor does not support AllReduce collectives in this slice".into());
    };
    // S3 is PP-only. Composed TP/EP groups require a later collective executor.
    if src >= ranks || dst >= ranks || src == dst || collective.group != [src, dst] {
        return Err("BandXfer must name exactly its two logical PP owners".into());
    }
    Ok((src, dst))
}

fn validate_boundary(meta: BoundaryMeta, epoch: &ForwardEpoch) -> Result<()> {
    let d = meta.descriptor;
    if d.epoch != meta.transport_epoch || d.op != epoch.op_id || d.rows != epoch.actual_rows {
        return Err("boundary descriptor epoch/op/rows mismatch".into());
    }
    if !matches!(d.dtype_bytes, 2 | 4)
        || d.bytes == 0
        || d.bytes > meta.capacity
        || d.bytes > meta.extent
        || d.bytes % d.rows != 0
        || (d.bytes / d.rows) % 16 != 0
        || meta.address == 0
        || meta.address % 16 != 0
        || d.bytes / 16 > i32::MAX as usize
    {
        return Err(
            "boundary descriptor exceeds lease or has invalid live extent/alignment".into(),
        );
    }
    Ok(())
}

fn prepare(plan: &SplitPlan, epoch: &ForwardEpoch, backend: &impl Backend) -> Result<Prepared> {
    if plan.mesh_epoch != epoch.mesh_epoch {
        return Err("split plan belongs to a stale mesh epoch".into());
    }
    if epoch.actual_rows == 0
        || epoch.route_rows < epoch.actual_rows
        || epoch
            .start_position
            .checked_add(epoch.actual_rows)
            .is_none()
    {
        return Err("invalid forward epoch row geometry".into());
    }
    if plan.ranks.is_empty() || plan.ranks.len() != backend.rank_count() {
        return Err("split plan rank count does not match devices".into());
    }
    let mut seen_ops = HashSet::new();
    let mut occurrences = vec![[0usize; 2]; plan.collectives.len()];
    let mut metas = Vec::with_capacity(plan.collectives.len());
    for (index, c) in plan.collectives.iter().enumerate() {
        if c.id.0 as usize != index || c.operand.0 as usize >= plan.resources.len() {
            return Err("invalid collective id or resource".into());
        }
        let (src, dst) = endpoints(c, plan.ranks.len())?;
        let source = backend.boundary(src, c, epoch)?;
        let destination = backend.boundary(dst, c, epoch)?;
        validate_boundary(source, epoch)?;
        validate_boundary(destination, epoch)?;
        if source.descriptor != destination.descriptor {
            return Err("BandXfer producer and consumer descriptors differ".into());
        }
        metas.push([source, destination]);
    }
    let mut item_count = 0;
    for (owner, rank) in plan.ranks.iter().enumerate() {
        if rank.device != owner || !backend.ready(owner) {
            return Err("rank program device mismatch or missing non-null launch stream".into());
        }
        for item in &rank.items {
            match *item {
                RankItem::Op(id) => {
                    if !seen_ops.insert((owner, id)) {
                        return Err("duplicate rank op in split plan".into());
                    }
                }
                RankItem::Collective(id) => {
                    let c = plan
                        .collectives
                        .get(id.0 as usize)
                        .ok_or("invalid rank collective id")?;
                    let (src, dst) = endpoints(c, plan.ranks.len())?;
                    let side = if owner == src {
                        0
                    } else if owner == dst {
                        1
                    } else {
                        return Err("collective placed on a nonparticipant rank".into());
                    };
                    occurrences[id.0 as usize][side] += 1;
                }
            }
        }
        item_count += rank.items.len();
    }
    if occurrences.iter().any(|count| *count != [1, 1]) {
        return Err("BandXfer must appear exactly once on each endpoint".into());
    }
    // Build the topological walk before submission. One action per rank per pass
    // allows chunk overlap while receiving each bank before it can be reused.
    let mut prepared = Prepared {
        actions: Vec::with_capacity(item_count + plan.ranks.len()),
        ops: Vec::with_capacity(seen_ops.len()),
        issued: Vec::new(),
    };
    let mut cursors = vec![0; plan.ranks.len()];
    let mut published = vec![false; plan.collectives.len()];
    let mut received = vec![false; plan.collectives.len()];
    let mut busy_edges = HashSet::new();
    let mut remaining = item_count;
    while remaining != 0 {
        let before = remaining;
        for (owner, rank) in plan.ranks.iter().enumerate() {
            let cursor = &mut cursors[owner];
            if *cursor == rank.items.len() {
                continue;
            }
            match rank.items[*cursor] {
                RankItem::Op(_) => {
                    let start = prepared.ops.len();
                    while let Some(RankItem::Op(id)) = rank.items.get(*cursor) {
                        prepared.ops.push(*id);
                        *cursor += 1;
                        remaining -= 1;
                    }
                    prepared.actions.push(Action::Ops {
                        owner,
                        start,
                        end: prepared.ops.len(),
                    });
                }
                RankItem::Collective(id) => {
                    let index = id.0 as usize;
                    let c = &plan.collectives[index];
                    let (src, dst) = endpoints(c, plan.ranks.len())?;
                    let edge = (src, dst);
                    let publish = owner == src;
                    if publish {
                        if busy_edges.contains(&edge) {
                            continue;
                        }
                        published[index] = true;
                        busy_edges.insert(edge);
                    } else {
                        if !published[index] {
                            continue;
                        }
                        received[index] = true;
                        busy_edges.remove(&edge);
                    }
                    prepared.actions.push(Action::Transfer {
                        owner,
                        collective: index,
                        meta: metas[index][usize::from(!publish)],
                        publish,
                    });
                    *cursor += 1;
                    remaining -= 1;
                }
            }
            if *cursor == rank.items.len() {
                prepared.actions.push(Action::Marks(owner));
            }
        }
        if before == remaining {
            return Err("cyclic or inconsistent BandXfer schedule".into());
        }
    }
    // Empty owners still participate in acknowledgements and marks.
    for (owner, rank) in plan.ranks.iter().enumerate() {
        if rank.items.is_empty() {
            prepared.actions.push(Action::Marks(owner));
        }
    }
    debug_assert!(received.iter().all(|issued| *issued));
    prepared
        .issued
        .resize(prepared.actions.len().div_ceil(64), 0);
    Ok(prepared)
}

fn issue(
    prepared: &mut Prepared,
    plan: &SplitPlan,
    epoch: &ForwardEpoch,
    backend: &mut impl Backend,
) -> Result<()> {
    for (index, action) in prepared.actions.iter().enumerate() {
        let mask = 1u64 << (index % 64);
        if prepared.issued[index / 64] & mask != 0 {
            backend.poison();
            return Err("mesh epoch item was already issued".into());
        }
        // Mark before invoking user code: failure cannot retry a partial enqueue.
        prepared.issued[index / 64] |= mask;
        let result = match *action {
            Action::Ops { owner, start, end } => {
                backend.run_ops(owner, epoch, &prepared.ops[start..end])
            }
            Action::Transfer {
                owner,
                collective,
                meta,
                publish,
            } => backend.transfer(owner, &plan.collectives[collective], epoch, meta, publish),
            Action::Marks(owner) => backend.marks(owner, epoch),
        };
        if let Err(error) = result {
            backend.poison();
            return Err(error);
        }
    }
    Ok(())
}

fn drain(backend: &mut impl DrainBackend) -> Result<()> {
    let mut failure = None;
    for owner in 0..backend.rank_count() {
        if let Err(error) = backend.synchronize(owner) {
            failure.get_or_insert(error);
        }
    }
    if let Some(error) = failure {
        backend.poison();
        backend.quarantine();
        return Err(format!("mesh drain failed; resources quarantined: {error}"));
    }
    Ok(())
}

/// Access to the executor state machine from a backend; HIP and the CPU toy
/// tests drive the very same `submit`/`drain_tickets` through it.
trait HasState {
    fn state(&mut self) -> &mut MeshExecutionState;
}

/// Admit, prepare, enqueue. Validation (admission, `prepare_epoch`, `prepare`)
/// happens before any device enqueue and refuses without poisoning: outstanding
/// earlier epochs stay drainable. Only an enqueue failure (`issue`) poisons, and
/// the failed epoch is never recorded outstanding, so no ticket exists for it.
fn submit<B: Backend + HasState>(
    plan: &SplitPlan,
    epoch: &ForwardEpoch,
    backend: &mut B,
) -> Result<MeshTicket> {
    backend.state().admit(epoch)?;
    backend.prepare_epoch(epoch)?;
    let mut prepared = prepare(plan, epoch, &*backend)?;
    let state = backend.state();
    state.last_op = Some(epoch.op_id);
    state.scope = Some((epoch.load_id, epoch.policy_id));
    issue(&mut prepared, plan, epoch, backend)?;
    let state = backend.state();
    state.outstanding.push(*epoch);
    Ok(MeshTicket {
        issuer: state.issuer,
        epoch: *epoch,
    })
}

/// Validate every ticket before touching a device, synchronize every device
/// once, then retire all outstanding epochs. Failed sync poisons and
/// quarantines (S3 `drain`); a poisoned mesh retires its epochs but reports an
/// error for all of them.
fn drain_tickets<B: DrainBackend + HasState>(
    tickets: &[MeshTicket],
    backend: &mut B,
) -> Result<()> {
    backend.state().validate_tickets(tickets)?;
    drain(backend)?;
    let state = backend.state();
    state.outstanding.clear();
    if state.poisoned {
        return Err("mesh submission failed before drain".into());
    }
    Ok(())
}

struct HipBackend<'a> {
    devices: &'a mut [Gpu],
    state: &'a mut MeshExecutionState,
    bindings: &'a mut dyn MeshStepBindings,
}

impl DrainBackend for HipBackend<'_> {
    fn rank_count(&self) -> usize {
        self.devices.len()
    }
    fn synchronize(&mut self, owner: usize) -> Result<()> {
        let gpu = &mut self.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        gpu.hip.device_synchronize().map_err(|e| e.to_string())
    }
    fn poison(&mut self) {
        self.state.poison();
    }
    fn quarantine(&mut self) {
        self.state.quarantined = true;
    }
}

impl Backend for HipBackend<'_> {
    fn ready(&self, owner: usize) -> bool {
        self.devices[owner].active_stream.is_some()
    }
    fn boundary(
        &self,
        owner: usize,
        c: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryMeta> {
        let (src, dst) = endpoints(c, self.devices.len())?;
        let edge = self
            .state
            .edges
            .get(&EdgeId {
                src_device: src,
                dst_device: dst,
            })
            .ok_or("BandXfer has no provisioned lease")?;
        let io = self.bindings.boundary(owner, c, epoch)?;
        let gpu = &self.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        let attributes = gpu
            .hip
            .pointer_get_attributes(&io.view.buf)
            .map_err(|e| e.to_string())?;
        if attributes.device != gpu.device_id
            || attributes.mem_type != 1
            || attributes.is_managed != 0
        {
            return Err("boundary activation is not local device memory of its owner".into());
        }
        Ok(BoundaryMeta {
            descriptor: io.descriptor,
            address: io.view.buf.as_ptr() as usize,
            extent: io.view.buf.size(),
            capacity: edge.max_bytes,
            transport_epoch: edge.lease.epoch(),
        })
    }
    fn run_ops(&mut self, owner: usize, epoch: &ForwardEpoch, ops: &[OpId]) -> Result<()> {
        let gpu = &mut self.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        self.bindings.run_ops(owner, gpu, epoch, ops)
    }
    fn transfer(
        &mut self,
        owner: usize,
        c: &PlannedCollective,
        epoch: &ForwardEpoch,
        meta: BoundaryMeta,
        publish: bool,
    ) -> Result<()> {
        if self.boundary(owner, c, epoch)? != meta {
            return Err("prepared boundary binding changed during submission".into());
        }
        let (src, dst) = endpoints(c, self.devices.len())?;
        let edge = &self.state.edges[&EdgeId {
            src_device: src,
            dst_device: dst,
        }];
        let io = self.bindings.boundary(owner, c, epoch)?;
        let gpu = &mut self.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        if publish {
            edge.lease.publish(gpu, io.view, io.descriptor)
        } else {
            edge.lease.receive(gpu, io.view, io.descriptor)
        }
    }
    fn marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.bindings.apply_rank_marks(owner, epoch)
    }
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.bindings.prepare_epoch(epoch)
    }
}

impl HasState for HipBackend<'_> {
    fn state(&mut self) -> &mut MeshExecutionState {
        self.state
    }
}

/// Drain backend over the admitted device list.
///
/// A failed drain quarantines the whole device `Vec`: `quarantine` leaks it with
/// `std::mem::forget` because `Gpu::drop` frees registered VMM/host-mapped state
/// that in-flight work may still reference. `Gpus` is therefore unusable
/// afterwards (its `devices` is left empty) until process teardown.
struct HipDrain<'a> {
    devices: &'a mut Vec<Gpu>,
    state: &'a mut MeshExecutionState,
}

impl DrainBackend for HipDrain<'_> {
    fn rank_count(&self) -> usize {
        self.devices.len()
    }
    fn synchronize(&mut self, owner: usize) -> Result<()> {
        let gpu = &mut self.devices[owner];
        gpu.bind_thread().map_err(|e| e.to_string())?;
        // Device-wide synchronization includes family copy/PLE side streams.
        gpu.hip.device_synchronize().map_err(|e| e.to_string())
    }
    fn poison(&mut self) {
        self.state.poison();
    }
    fn quarantine(&mut self) {
        self.state.quarantined = true;
        // Gpu::drop frees registered VMM/host-mapped state; a failed drain keeps
        // device ownership until process teardown.
        std::mem::forget(std::mem::take(self.devices));
    }
}

impl HasState for HipDrain<'_> {
    fn state(&mut self) -> &mut MeshExecutionState {
        self.state
    }
}

fn pair(devices: &mut [Gpu], src: usize, dst: usize) -> (&mut Gpu, &mut Gpu) {
    if src < dst {
        let (a, b) = devices.split_at_mut(dst);
        (&mut a[src], &mut b[0])
    } else {
        let (a, b) = devices.split_at_mut(src);
        (&mut b[0], &mut a[dst])
    }
}

static NEXT_ISSUER: AtomicU64 = AtomicU64::new(1);

impl Gpus {
    /// Provision transactionally from admitted logical-to-physical identities.
    /// Existing legacy layout/output accessors are deliberately untouched.
    pub fn provision_mesh_edges(
        &mut self,
        admitted: &AdmittedMesh,
        placement: &BandPlacement,
        max_bytes: usize,
    ) -> Result<()> {
        placement.validate_for(&admitted.mesh, placement.n_trunk_layers)?;
        if self.mesh_execution.mesh_epoch.is_some() || self.mesh_execution.quarantined {
            return Err("mesh edges already provisioned or quarantined".into());
        }
        if admitted.mesh.size_of(DimKind::Tp) != 1 || admitted.mesh.size_of(DimKind::Ep) != 1 {
            return Err("S3 executor admission supports PP with TP=EP=1 only".into());
        }
        if admitted.mesh.n_devices() != self.devices.len()
            || admitted.ranks.len() != self.devices.len()
            || max_bytes == 0
            || max_bytes % 16 != 0
            || max_bytes / 16 > i32::MAX as usize
        {
            return Err("invalid admitted rank count or edge capacity".into());
        }
        for (owner, rank) in admitted.ranks.iter().enumerate() {
            let gpu = &self.devices[owner];
            if rank.logical != owner
                || rank.device_ordinal != gpu.device_id
                || gpu.active_stream.is_none()
                || rank.arch != gpu.arch
            {
                return Err("admitted physical rank or non-null launch stream mismatch".into());
            }
            if let Some(expected_pci) = &rank.pci_bus_id {
                gpu.bind_thread().map_err(|e| e.to_string())?;
                if gpu
                    .hip
                    .device_pci_bus_id(gpu.device_id)
                    .map_err(|e| e.to_string())?
                    != *expected_pci
                {
                    return Err("admitted PCI identity does not match physical device".into());
                }
            }
        }
        let expected: HashSet<_> = placement
            .boundaries()
            .map(|(_, src, dst)| (src, dst))
            .collect();
        let supplied: HashSet<_> = admitted.edges.iter().map(|e| (e.src, e.dst)).collect();
        if expected != supplied || supplied.len() != admitted.edges.len() {
            return Err("admitted edge decisions do not match PP boundaries".into());
        }
        let issuer = NEXT_ISSUER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| "mesh ticket identity exhausted")?;
        let mut edges = HashMap::with_capacity(admitted.edges.len());
        let result = (|| {
            for decision in &admitted.edges {
                let (src, dst) = pair(&mut self.devices, decision.src, decision.dst);
                if decision.transport == EdgeTransport::Peer && !transport::qualify_peer(src, dst)?
                {
                    return Err("admitted Peer edge failed byte qualification; no fallback".into());
                }
                let id = EdgeId {
                    src_device: decision.src,
                    dst_device: decision.dst,
                };
                let lease = transport::provision(
                    src,
                    dst,
                    EdgeSpec {
                        edge: id,
                        transport: decision.transport,
                        max_bytes,
                    },
                )?;
                edges.insert(id, OwnedEdge { lease, max_bytes });
            }
            Ok(())
        })();
        if let Err(error) = result {
            // A failed provision may already have quarantined its private lease.
            // Retain all successful leases until the explicit all-device drain.
            self.mesh_execution.edges = edges;
            self.mesh_execution.mesh_epoch = Some(admitted.mesh.epoch());
            self.mesh_execution.poison();
            return match self.drain_mesh_edges() {
                Ok(()) => Err(error),
                Err(drain) => Err(format!("{error}; {drain}")),
            };
        }
        self.mesh_execution = MeshExecutionState {
            edges,
            mesh_epoch: Some(admitted.mesh.epoch()),
            output_owner: Some(placement.stages() - 1),
            issuer,
            ..Default::default()
        };
        Ok(())
    }

    /// Logical output owner whose admitted physical identity is retained in the
    /// device vector. This is separate from the legacy Qwen35 output accessor.
    pub fn mesh_output_owner(&self) -> Option<usize> {
        self.mesh_execution.output_owner
    }

    pub fn mesh_edge_epoch(&self, edge: EdgeId) -> Option<u64> {
        self.mesh_execution
            .edges
            .get(&edge)
            .map(|e| e.lease.epoch())
    }

    /// Submit one epoch. Epochs of the same request may be outstanding at once
    /// (strictly increasing op ids); every ticket must be drained together by
    /// `drain_mesh_epochs`. A submission failure after any device enqueue
    /// poisons the mesh: all earlier outstanding tickets then drain to an error.
    pub fn submit_mesh_epoch(
        &mut self,
        plan: &SplitPlan,
        epoch: &ForwardEpoch,
        bindings: &mut dyn MeshStepBindings,
    ) -> Result<MeshTicket> {
        submit(
            plan,
            epoch,
            &mut HipBackend {
                devices: &mut self.devices,
                state: &mut self.mesh_execution,
                bindings,
            },
        )
    }

    /// Single-ticket drain: valid only while that ticket names the sole
    /// outstanding epoch.
    pub fn drain_mesh_epoch(&mut self, ticket: MeshTicket) -> Result<()> {
        drain_tickets(
            &[ticket],
            &mut HipDrain {
                devices: &mut self.devices,
                state: &mut self.mesh_execution,
            },
        )
    }

    /// Drain every outstanding epoch with one synchronization of each device.
    /// `tickets` must be exactly the outstanding tickets, oldest first; any
    /// stale, foreign, missing or out-of-order ticket is refused before a
    /// device is touched and leaves the state unchanged. A failed device sync
    /// quarantines exactly as `drain_mesh_edges` does.
    pub fn drain_mesh_epochs(&mut self, tickets: Vec<MeshTicket>) -> Result<()> {
        drain_tickets(
            &tickets,
            &mut HipDrain {
                devices: &mut self.devices,
                state: &mut self.mesh_execution,
            },
        )
    }

    /// Stop and drain every device before releasing any bank/event. A failed
    /// drain never frees any mesh lease, even if other devices are healthy.
    pub fn drain_mesh_edges(&mut self) -> Result<()> {
        if self.mesh_execution.quarantined {
            return Err("mesh leases quarantined until process teardown".into());
        }
        self.mesh_execution.poison();
        drain(&mut HipDrain {
            devices: &mut self.devices,
            state: &mut self.mesh_execution,
        })?;
        for (id, edge) in &mut self.mesh_execution.edges {
            let (src, dst) = pair(&mut self.devices, id.src_device, id.dst_device);
            if let Err(error) = edge.lease.drain(src, dst) {
                self.mesh_execution.quarantined = true;
                return Err(error);
            }
        }
        self.mesh_execution = MeshExecutionState::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_mesh::DeviceMesh;
    use crate::mesh_split::{CollectiveId, RankProgram, ResourceShard, ScheduleRef, ShardSpec};
    use hipfire_dispatch::pipeline::step_graph::{
        OpDecl, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule, StepGraph,
    };

    #[derive(Debug, PartialEq, Eq)]
    enum Call {
        Ops(usize, Vec<OpId>),
        Publish(u32),
        Receive(u32),
        Marks(usize),
        Sync(usize),
    }

    struct Toy {
        ranks: usize,
        calls: Vec<Call>,
        missing: bool,
        oversize: bool,
        wrong_descriptor: bool,
        no_stream: bool,
        fail_ops: Option<usize>,
        fail_drain: Option<usize>,
        fail_prepare: bool,
        prepared: Vec<u64>,
        poisoned: bool,
        quarantined: bool,
        /// The real executor state machine, driven by the production
        /// `submit`/`drain_tickets` through `HasState`.
        state: MeshExecutionState,
    }

    impl Toy {
        fn new(ranks: usize) -> Self {
            Self {
                ranks,
                calls: Vec::new(),
                missing: false,
                oversize: false,
                wrong_descriptor: false,
                no_stream: false,
                fail_ops: None,
                fail_drain: None,
                fail_prepare: false,
                prepared: Vec::new(),
                poisoned: false,
                quarantined: false,
                state: MeshExecutionState::default(),
            }
        }

        /// Provisioned for `epoch`'s mesh with a distinct ticket issuer.
        fn provisioned(ranks: usize, epoch: &ForwardEpoch, issuer: u64) -> Self {
            let mut toy = Self::new(ranks);
            toy.state.mesh_epoch = Some(epoch.mesh_epoch);
            toy.state.issuer = issuer;
            toy
        }

        fn syncs(&self) -> Vec<usize> {
            self.calls
                .iter()
                .filter_map(|call| match call {
                    Call::Sync(owner) => Some(*owner),
                    _ => None,
                })
                .collect()
        }
    }

    impl DrainBackend for Toy {
        fn rank_count(&self) -> usize {
            self.ranks
        }
        fn synchronize(&mut self, owner: usize) -> Result<()> {
            self.calls.push(Call::Sync(owner));
            if self.fail_drain == Some(owner) {
                Err("injected drain failure".into())
            } else {
                Ok(())
            }
        }
        fn poison(&mut self) {
            self.poisoned = true;
            self.state.poison();
        }
        fn quarantine(&mut self) {
            self.quarantined = true;
            self.state.quarantined = true;
        }
    }

    impl Backend for Toy {
        fn ready(&self, _: usize) -> bool {
            !self.no_stream
        }
        fn boundary(
            &self,
            _: usize,
            _: &PlannedCollective,
            epoch: &ForwardEpoch,
        ) -> Result<BoundaryMeta> {
            if self.missing {
                return Err("no lease".into());
            }
            Ok(BoundaryMeta {
                descriptor: EdgeDescriptor {
                    epoch: 17,
                    op: epoch.op_id + u64::from(self.wrong_descriptor),
                    rows: epoch.actual_rows,
                    dtype_bytes: 4,
                    bytes: if self.oversize { 256 } else { 64 },
                },
                address: 0x1000,
                extent: 512,
                capacity: 64,
                transport_epoch: 17,
            })
        }
        fn run_ops(&mut self, owner: usize, _: &ForwardEpoch, ops: &[OpId]) -> Result<()> {
            self.calls.push(Call::Ops(owner, ops.to_vec()));
            if self.fail_ops == Some(owner) {
                Err("injected stage failure".into())
            } else {
                Ok(())
            }
        }
        fn transfer(
            &mut self,
            _: usize,
            c: &PlannedCollective,
            _: &ForwardEpoch,
            _: BoundaryMeta,
            publish: bool,
        ) -> Result<()> {
            self.calls.push(if publish {
                Call::Publish(c.id.0)
            } else {
                Call::Receive(c.id.0)
            });
            Ok(())
        }
        fn marks(&mut self, owner: usize, _: &ForwardEpoch) -> Result<()> {
            self.calls.push(Call::Marks(owner));
            Ok(())
        }
        fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
            self.prepared.push(epoch.op_id);
            if self.fail_prepare {
                Err("injected prepare failure".into())
            } else {
                Ok(())
            }
        }
    }

    impl HasState for Toy {
        fn state(&mut self) -> &mut MeshExecutionState {
            &mut self.state
        }
    }

    fn epoch(mesh: &DeviceMesh) -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: mesh.epoch(),
            load_id: 1,
            request_id: 2,
            op_id: 3,
            route: MeshRoute::Prefill,
            start_position: 0,
            actual_rows: 1,
            route_rows: 1,
            policy_id: 4,
        }
    }

    // A non-model toy family lowers two contiguous fake typed ops per band and
    // chunk into a StepGraph and its rank schedule. No family runtime is used.
    fn toy_plan(bands: usize, chunks: usize) -> (StepGraph, SplitPlan, ForwardEpoch) {
        let mesh = DeviceMesh::rect(&[(DimKind::Pp, bands)]).unwrap();
        let mut graph = StepGraph::new();
        let resource = graph
            .add_resource(ResourceDecl {
                class: ResourceClass::Activation,
                shape: vec![16],
                alias: None,
            })
            .unwrap();
        let mut plan = SplitPlan {
            mesh_epoch: mesh.epoch(),
            ranks: (0..bands)
                .map(|device| RankProgram {
                    device,
                    items: Vec::new(),
                })
                .collect(),
            collectives: Vec::new(),
            resources: vec![ResourceShard {
                entry: ShardSpec::Whole,
                exit: ShardSpec::Whole,
            }],
        };
        for _chunk in 0..chunks {
            for owner in 0..bands {
                let mut producer = OpId(0);
                for _ in 0..2 {
                    producer = graph
                        .add_op(OpDecl {
                            kind: OpKind::Scale,
                            layer: Some(owner),
                            rule: SplitRule::RowComplete,
                            reads: vec![resource],
                            writes: vec![resource],
                        })
                        .unwrap();
                    plan.ranks[owner].items.push(RankItem::Op(producer));
                }
                if owner + 1 < bands {
                    let id = CollectiveId(plan.collectives.len() as u32);
                    plan.collectives.push(PlannedCollective {
                        id,
                        kind: CollectiveKind::BandXfer {
                            src: owner,
                            dst: owner + 1,
                        },
                        operand: resource,
                        producer,
                        group: vec![owner, owner + 1],
                        fulfils: ScheduleRef::Band(owner),
                    });
                    plan.ranks[owner].items.push(RankItem::Collective(id));
                    plan.ranks[owner + 1].items.push(RankItem::Collective(id));
                }
            }
        }
        (graph, plan, epoch(&mesh))
    }

    fn prove_graph(bands: usize) {
        let chunks = 4;
        let (graph, plan, epoch) = toy_plan(bands, chunks);
        let mut toy = Toy::new(bands);
        let mut prepared = prepare(&plan, &epoch, &toy).unwrap();
        assert!(toy.calls.is_empty());
        issue(&mut prepared, &plan, &epoch, &mut toy).unwrap();
        let mut positions = HashMap::new();
        for (position, call) in toy.calls.iter().enumerate() {
            if let Call::Ops(owner, ops) = call {
                assert_eq!(ops.len(), 2, "contiguous op fusion preserved");
                for op in ops {
                    assert!(positions.insert(*op, position).is_none());
                    assert_eq!(graph.ops()[op.0 as usize].layer, Some(*owner));
                }
            }
        }
        assert_eq!(positions.len(), graph.ops().len());
        for c in &plan.collectives {
            let published: Vec<_> = toy
                .calls
                .iter()
                .enumerate()
                .filter(|(_, call)| **call == Call::Publish(c.id.0))
                .map(|(i, _)| i)
                .collect();
            let received: Vec<_> = toy
                .calls
                .iter()
                .enumerate()
                .filter(|(_, call)| **call == Call::Receive(c.id.0))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(published.len(), 1);
            assert_eq!(received.len(), 1);
            assert!(positions[&c.producer] < published[0]);
            assert!(published[0] < received[0]);
            let CollectiveKind::BandXfer { dst, .. } = c.kind else {
                unreachable!()
            };
            let rank = &plan.ranks[dst].items;
            let index = rank
                .iter()
                .position(|item| *item == RankItem::Collective(c.id))
                .unwrap();
            let RankItem::Op(consumer) = rank[index + 1] else {
                panic!("missing consumer")
            };
            assert!(received[0] < positions[&consumer]);
        }
        // Each stage/chunk's ops precede the next stage of that same chunk.
        for chunk in 0..chunks {
            for stage in 0..bands - 1 {
                assert!(
                    positions[&OpId(((chunk * bands + stage) * 2) as u32)]
                        < positions[&OpId(((chunk * bands + stage + 1) * 2) as u32)]
                );
            }
        }
        // Enqueueing is not globally serialized chunk-by-chunk: the first
        // stage can start chunk 1 before the next stage starts chunk 0.
        assert!(positions[&OpId((bands * 2) as u32)] < positions[&OpId(2)]);
        assert!(toy.calls.iter().all(|call| !matches!(call, Call::Sync(_))));
        assert!(issue(&mut prepared, &plan, &epoch, &mut toy).is_err());
        assert!(toy.poisoned);
    }

    #[test]
    fn two_band_toy_exactly_once_and_topological() {
        prove_graph(2);
    }

    #[test]
    fn six_band_toy_exactly_once_and_topological() {
        prove_graph(6);
    }

    #[test]
    fn real_splitter_non_family_graph_executes() {
        for bands in [2, 6] {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, bands)]).unwrap();
            let placement = BandPlacement::uniform(&mesh, bands).unwrap();
            let manifest =
                crate::weight_manifest::plan_manifest(&[], &[], &mesh, bands, &placement).unwrap();
            let mut graph = StepGraph::new();
            let x = graph
                .add_resource(ResourceDecl {
                    class: ResourceClass::Activation,
                    shape: vec![16],
                    alias: None,
                })
                .unwrap();
            for layer in 0..bands {
                graph
                    .add_op(OpDecl {
                        kind: OpKind::Scale,
                        layer: Some(layer),
                        rule: SplitRule::RowComplete,
                        reads: vec![x],
                        writes: vec![x],
                    })
                    .unwrap();
            }
            let plan = crate::mesh_split::split_program(&graph, &manifest, &mesh).unwrap();
            let mut toy = Toy::new(bands);
            let epoch = epoch(&mesh);
            let mut prepared = prepare(&plan, &epoch, &toy).unwrap();
            issue(&mut prepared, &plan, &epoch, &mut toy).unwrap();
            assert_eq!(
                toy.calls
                    .iter()
                    .filter(|call| matches!(call, Call::Ops(..)))
                    .count(),
                bands
            );
            assert_eq!(
                toy.calls
                    .iter()
                    .filter(|call| matches!(call, Call::Publish(_)))
                    .count(),
                bands - 1
            );
        }
    }

    #[test]
    fn stale_epoch_refused_before_calls() {
        let (_, plan, mut epoch) = toy_plan(2, 1);
        epoch.mesh_epoch = DeviceMesh::single().unwrap().epoch();
        let toy = Toy::new(2);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn wrong_device_refused_before_calls() {
        let (_, mut plan, epoch) = toy_plan(2, 1);
        plan.ranks[1].device = 0;
        let toy = Toy::new(2);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn allreduce_kinds_refused_before_calls() {
        for kind in [
            CollectiveKind::AllReduce,
            CollectiveKind::AllReduceAdd {
                into: ResourceId(0),
                replaces: OpId(0),
            },
        ] {
            let (_, mut plan, epoch) = toy_plan(2, 1);
            plan.collectives[0].kind = kind;
            let toy = Toy::new(2);
            assert!(prepare(&plan, &epoch, &toy)
                .unwrap_err()
                .contains("AllReduce"));
            assert!(toy.calls.is_empty());
        }
    }

    #[test]
    fn missing_lease_refused_before_calls() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::new(2);
        toy.missing = true;
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn oversize_descriptor_refused_before_calls() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::new(2);
        toy.oversize = true;
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    #[test]
    fn wrong_descriptor_and_null_stream_refused_before_calls() {
        let (_, plan, epoch) = toy_plan(2, 1);
        for no_stream in [false, true] {
            let mut toy = Toy::new(2);
            toy.no_stream = no_stream;
            toy.wrong_descriptor = !no_stream;
            assert!(prepare(&plan, &epoch, &toy).is_err());
            assert!(toy.calls.is_empty());
        }
    }

    #[test]
    fn stage_error_poisons_and_stops_without_publication() {
        let (_, plan, epoch) = toy_plan(6, 2);
        let mut toy = Toy::new(6);
        toy.fail_ops = Some(0);
        let mut prepared = prepare(&plan, &epoch, &toy).unwrap();
        assert!(issue(&mut prepared, &plan, &epoch, &mut toy).is_err());
        assert!(toy.poisoned);
        assert_eq!(toy.calls.len(), 1);
        assert!(matches!(toy.calls[0], Call::Ops(0, _)));
    }

    #[test]
    fn drain_failure_quarantines_after_attempting_every_owner() {
        let mut toy = Toy::new(6);
        toy.fail_drain = Some(1);
        assert!(drain(&mut toy).is_err());
        assert!(toy.poisoned && toy.quarantined);
        assert_eq!(toy.calls, (0..6).map(Call::Sync).collect::<Vec<_>>());
    }

    #[test]
    fn successful_drain_acknowledges_every_owner() {
        let mut toy = Toy::new(2);
        drain(&mut toy).unwrap();
        assert_eq!(toy.calls, vec![Call::Sync(0), Call::Sync(1)]);
        assert!(!toy.poisoned && !toy.quarantined);
    }

    #[test]
    fn malformed_duplicate_and_cycle_refused_before_calls() {
        let (_, mut plan, epoch) = toy_plan(2, 1);
        plan.ranks[0].items.push(RankItem::Op(OpId(0)));
        let toy = Toy::new(2);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        let (_, mut plan, epoch) = toy_plan(2, 2);
        // Destination waits for chunk 1 before chunk 0, source cannot reuse
        // the ordered edge until chunk 0 has been received.
        let first = plan.ranks[1]
            .items
            .iter()
            .position(|x| *x == RankItem::Collective(CollectiveId(0)))
            .unwrap();
        let second = plan.ranks[1]
            .items
            .iter()
            .position(|x| *x == RankItem::Collective(CollectiveId(1)))
            .unwrap();
        plan.ranks[1].items.swap(first, second);
        assert!(prepare(&plan, &epoch, &toy).is_err());
        assert!(toy.calls.is_empty());
    }

    fn at(epoch: &ForwardEpoch, op_id: u64) -> ForwardEpoch {
        ForwardEpoch { op_id, ..*epoch }
    }

    fn submit_n(
        toy: &mut Toy,
        plan: &SplitPlan,
        epoch: &ForwardEpoch,
        first_op: u64,
        n: u64,
    ) -> Vec<MeshTicket> {
        (first_op..first_op + n)
            .map(|op| submit(plan, &at(epoch, op), toy).unwrap())
            .collect()
    }

    #[test]
    fn n_outstanding_epochs_drain_with_one_sync_per_device() {
        let (_, plan, epoch) = toy_plan(3, 1);
        let mut toy = Toy::provisioned(3, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 4);
        assert_eq!(toy.prepared, vec![10, 11, 12, 13]);
        assert_eq!(toy.state.outstanding.len(), 4);
        assert!(toy.syncs().is_empty(), "submission never waits on devices");
        let publishes = toy
            .calls
            .iter()
            .filter(|call| matches!(call, Call::Publish(_)))
            .count();
        assert_eq!(publishes, 4 * 2);
        drain_tickets(&tickets, &mut toy).unwrap();
        assert_eq!(toy.syncs(), vec![0, 1, 2], "every device synchronized once");
        assert!(toy.state.outstanding.is_empty());
        assert!(!toy.poisoned && !toy.quarantined && !toy.state.poisoned);
        // The healthy mesh admits the next epoch (and even a new request) after
        // the drain, with the op id still strictly increasing.
        let next = ForwardEpoch {
            request_id: 99,
            ..at(&epoch, 14)
        };
        submit(&plan, &next, &mut toy).unwrap();
        assert!(submit(&plan, &at(&epoch, 14), &mut toy).is_err());
    }

    #[test]
    fn single_ticket_drain_keeps_its_meaning() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 3, 1);
        drain_tickets(&tickets, &mut toy).unwrap();
        assert_eq!(toy.syncs(), vec![0, 1]);
        assert!(toy.state.outstanding.is_empty());
        // A second drain of the same (now stale) ticket is refused and does not
        // synchronize again.
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert_eq!(toy.syncs(), vec![0, 1]);
        // One ticket cannot drain two outstanding epochs.
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let mut tickets = submit_n(&mut toy, &plan, &epoch, 3, 2);
        let newest = tickets.pop().unwrap();
        assert!(drain_tickets(&[newest], &mut toy).is_err());
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert!(toy.syncs().is_empty());
        assert_eq!(toy.state.outstanding.len(), 2);
    }

    #[test]
    fn admission_requires_same_request_and_increasing_ops() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        submit_n(&mut toy, &plan, &epoch, 10, 2);
        let prepared = toy.prepared.len();
        let calls = toy.calls.len();
        let other_request = ForwardEpoch {
            request_id: epoch.request_id + 1,
            ..at(&epoch, 12)
        };
        let other_load = ForwardEpoch {
            load_id: epoch.load_id + 1,
            ..at(&epoch, 12)
        };
        let other_policy = ForwardEpoch {
            policy_id: epoch.policy_id + 1,
            ..at(&epoch, 12)
        };
        let stale_mesh = ForwardEpoch {
            mesh_epoch: DeviceMesh::single().unwrap().epoch(),
            ..at(&epoch, 12)
        };
        for refused in [
            at(&epoch, 10),
            at(&epoch, 9),
            at(&epoch, 11),
            other_request,
            other_load,
            other_policy,
            stale_mesh,
        ] {
            assert!(submit(&plan, &refused, &mut toy).is_err());
        }
        // Admission refusals have no side effects at all: no bindings hook, no
        // enqueue, no poisoning, outstanding epochs untouched.
        assert_eq!(toy.prepared.len(), prepared);
        assert_eq!(toy.calls.len(), calls);
        assert!(!toy.poisoned && !toy.state.poisoned);
        assert_eq!(toy.state.outstanding.len(), 2);
        submit(&plan, &at(&epoch, 12), &mut toy).unwrap();
        assert_eq!(toy.state.outstanding.len(), 3);
    }

    #[test]
    fn validation_refusal_keeps_outstanding_epochs_drainable() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 2);
        let calls = toy.calls.len();
        toy.fail_prepare = true;
        assert!(submit(&plan, &at(&epoch, 12), &mut toy).is_err());
        toy.fail_prepare = false;
        toy.oversize = true;
        assert!(submit(&plan, &at(&epoch, 12), &mut toy).is_err());
        toy.oversize = false;
        assert_eq!(toy.calls.len(), calls, "validation never enqueues");
        assert!(!toy.poisoned && !toy.state.poisoned);
        assert_eq!(toy.state.last_op, Some(11), "op id burned only by enqueue");
        drain_tickets(&tickets, &mut toy).unwrap();
        assert!(!toy.poisoned && !toy.quarantined);
    }

    #[test]
    fn out_of_order_foreign_and_stale_tickets_refused_before_any_sync() {
        let (_, plan, epoch) = toy_plan(2, 1);
        let mut toy = Toy::provisioned(2, &epoch, 1);
        let mut tickets = submit_n(&mut toy, &plan, &epoch, 10, 3);
        let mut foreign = Toy::provisioned(2, &epoch, 2);
        let foreign_tickets = submit_n(&mut foreign, &plan, &epoch, 10, 3);

        assert!(drain_tickets(&[], &mut toy).is_err());
        // Missing newest, missing oldest, or not all of them.
        assert!(drain_tickets(&tickets[..2], &mut toy).is_err());
        assert!(drain_tickets(&tickets[1..], &mut toy).is_err());
        // Out of order.
        tickets.swap(0, 2);
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        tickets.swap(0, 2);
        // Foreign issuer, whole set and mixed.
        assert!(drain_tickets(&foreign_tickets, &mut toy).is_err());
        let mixed = vec![
            MeshTicket {
                issuer: 1,
                epoch: tickets[0].epoch,
            },
            MeshTicket {
                issuer: 2,
                epoch: tickets[1].epoch,
            },
            MeshTicket {
                issuer: 1,
                epoch: tickets[2].epoch,
            },
        ];
        assert!(drain_tickets(&mixed, &mut toy).is_err());
        // Right issuer, epoch that was never outstanding.
        let forged: Vec<_> = tickets
            .iter()
            .map(|t| MeshTicket {
                issuer: t.issuer,
                epoch: ForwardEpoch {
                    request_id: t.epoch.request_id + 1,
                    ..t.epoch
                },
            })
            .collect();
        assert!(drain_tickets(&forged, &mut toy).is_err());

        // Every refusal left the state untouched: no sync, no poison, no
        // quarantine, all three epochs still outstanding and drainable.
        assert!(toy.syncs().is_empty());
        assert!(!toy.poisoned && !toy.quarantined && !toy.state.poisoned);
        assert_eq!(toy.state.outstanding.len(), 3);
        drain_tickets(&tickets, &mut toy).unwrap();
        assert_eq!(toy.syncs(), vec![0, 1]);
        // Now stale.
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert_eq!(toy.syncs(), vec![0, 1]);
        assert!(foreign.syncs().is_empty());
    }

    #[test]
    fn mid_sequence_submission_failure_poisons_every_outstanding_ticket() {
        let (_, plan, epoch) = toy_plan(3, 2);
        let mut toy = Toy::provisioned(3, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 2);
        toy.fail_ops = Some(0);
        assert!(submit(&plan, &at(&epoch, 12), &mut toy).is_err());
        assert!(toy.poisoned && toy.state.poisoned);
        assert!(!toy.quarantined);
        // The failed epoch holds no ticket; the earlier ones stay outstanding.
        assert_eq!(toy.state.outstanding.len(), 2);
        assert_eq!(toy.state.last_op, Some(12), "failed op id is burned");
        // Poisoned mesh admits nothing more, even once the cause is gone.
        toy.fail_ops = None;
        assert!(submit(&plan, &at(&epoch, 13), &mut toy).is_err());
        // Drain still quiesces every device once and retires the tickets, but
        // none of them reports success.
        let error = drain_tickets(&tickets, &mut toy).unwrap_err();
        assert!(error.contains("failed before drain"), "{error}");
        assert_eq!(toy.syncs(), vec![0, 1, 2]);
        assert!(toy.state.outstanding.is_empty());
        assert!(toy.state.poisoned && !toy.quarantined);
        assert!(submit(&plan, &at(&epoch, 14), &mut toy).is_err());
    }

    #[test]
    fn drain_failure_with_outstanding_epochs_quarantines_like_s3() {
        let (_, plan, epoch) = toy_plan(3, 1);
        let mut toy = Toy::provisioned(3, &epoch, 1);
        let tickets = submit_n(&mut toy, &plan, &epoch, 10, 3);
        toy.fail_drain = Some(1);
        let error = drain_tickets(&tickets, &mut toy).unwrap_err();
        assert!(error.contains("quarantined"), "{error}");
        assert!(toy.poisoned && toy.quarantined);
        assert!(toy.state.poisoned && toy.state.quarantined);
        assert_eq!(toy.syncs(), vec![0, 1, 2], "every owner attempted once");
        // Quarantine is terminal: nothing is admitted, drained or re-synced.
        assert!(submit(&plan, &at(&epoch, 13), &mut toy).is_err());
        assert!(drain_tickets(&tickets, &mut toy).is_err());
        assert_eq!(toy.syncs(), vec![0, 1, 2]);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::device_mesh::DeviceMesh;
    use std::collections::HashSet;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Ev {
        Checkpoint(usize),
        Restore(usize),
        Validate(usize),
        Finalize(usize),
    }

    #[derive(Default)]
    struct Fake {
        log: Vec<Ev>,
        fail_checkpoint: HashSet<usize>,
        fail_validate: HashSet<usize>,
        fail_restore: HashSet<usize>,
        /// Owner whose checkpoint_rank returns a ticket minted for owner+100.
        wrong_owner_ticket: Option<usize>,
    }

    impl MeshCheckpointCallbacks for Fake {
        fn checkpoint_rank(
            &mut self,
            owner: usize,
            key: &MeshCheckpointKey,
        ) -> Result<MeshCheckpointId> {
            self.log.push(Ev::Checkpoint(owner));
            if self.fail_checkpoint.contains(&owner) {
                return Err("fake checkpoint failure".to_string());
            }
            let minted = if self.wrong_owner_ticket == Some(owner) {
                owner + 100
            } else {
                owner
            };
            Ok(MeshCheckpointId::issue(*key, minted, 7000 + owner as u64))
        }

        fn restore_rank(
            &mut self,
            owner: usize,
            key: &MeshCheckpointKey,
            ticket: MeshCheckpointId,
        ) -> Result<()> {
            assert_eq!(ticket.owner(), owner);
            assert_eq!(ticket.local(), 7000 + owner as u64);
            assert_eq!(ticket.key(), *key);
            self.log.push(Ev::Restore(owner));
            if self.fail_restore.contains(&owner) {
                return Err("fake restore failure".to_string());
            }
            Ok(())
        }

        fn validate_finalize_rank(
            &mut self,
            owner: usize,
            _key: &MeshCheckpointKey,
            _ticket: MeshCheckpointId,
        ) -> Result<()> {
            self.log.push(Ev::Validate(owner));
            if self.fail_validate.contains(&owner) {
                return Err("fake validate failure".to_string());
            }
            Ok(())
        }

        fn finalize_rank(
            &mut self,
            owner: usize,
            _key: &MeshCheckpointKey,
            _ticket: MeshCheckpointId,
        ) {
            self.log.push(Ev::Finalize(owner));
        }
    }

    fn key() -> MeshCheckpointKey {
        MeshCheckpointKey {
            mesh_epoch: DeviceMesh::single().unwrap().epoch(),
            load_gen: 1,
            reset_gen: 2,
            request_gen: 3,
            window_gen: 4,
        }
    }

    fn set(owners: &[usize]) -> HashSet<usize> {
        owners.iter().copied().collect()
    }

    #[test]
    fn checkpoint_all_captures_trunk_then_head_with_keyed_tickets() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        assert_eq!(
            cb.log,
            vec![
                Ev::Checkpoint(0),
                Ev::Checkpoint(1),
                Ev::Checkpoint(2),
                Ev::Checkpoint(3)
            ]
        );
        assert_eq!(cp.per_owner.len(), 3);
        assert_eq!(cp.head_owner(), 3);
        for (owner, id) in cp.per_owner.iter().enumerate() {
            assert_eq!((id.owner(), id.key()), (owner, k));
        }
        let head = cp.head.unwrap();
        assert_eq!((head.owner(), head.key()), (3, k));
        assert!(cp.check_shape(&k).is_ok());
    }

    #[test]
    fn checkpoint_all_without_head_has_no_head_callback() {
        let mut cb = Fake::default();
        let cp = checkpoint_all(&mut cb, &key(), 2, false).unwrap();
        assert!(cp.head.is_none());
        assert_eq!(cb.log, vec![Ev::Checkpoint(0), Ev::Checkpoint(1)]);
    }

    #[test]
    fn checkpoint_failure_restores_captured_owners_in_reverse() {
        let mut cb = Fake {
            fail_checkpoint: set(&[2]),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 4, true).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Checkpoint);
        assert_eq!(err.owner, 2);
        assert!(!err.poisoned());
        assert_eq!(
            cb.log,
            vec![
                Ev::Checkpoint(0),
                Ev::Checkpoint(1),
                Ev::Checkpoint(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn head_checkpoint_failure_restores_every_trunk_owner() {
        let mut cb = Fake {
            fail_checkpoint: set(&[3]),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 3, true).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Checkpoint, 3));
        assert_eq!(
            cb.log,
            vec![
                Ev::Checkpoint(0),
                Ev::Checkpoint(1),
                Ev::Checkpoint(2),
                Ev::Checkpoint(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn checkpoint_failure_keeps_restoring_past_restore_errors_and_poisons() {
        let mut cb = Fake {
            fail_checkpoint: set(&[3]),
            fail_restore: set(&[1]),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 3, true).unwrap_err();
        assert!(err.poisoned());
        assert_eq!(err.restore_failures.len(), 1);
        assert_eq!(err.restore_failures[0].0, 1);
        // owner 0 was still restored after owner 1's restore failed
        assert_eq!(
            &cb.log[4..],
            &[Ev::Restore(2), Ev::Restore(1), Ev::Restore(0)]
        );
    }

    #[test]
    fn checkpoint_rejects_ticket_minted_for_another_owner() {
        let mut cb = Fake {
            wrong_owner_ticket: Some(1),
            ..Fake::default()
        };
        let err = checkpoint_all(&mut cb, &key(), 3, false).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Checkpoint, 1));
        assert_eq!(
            cb.log,
            vec![Ev::Checkpoint(0), Ev::Checkpoint(1), Ev::Restore(0)]
        );
    }

    #[test]
    fn checkpoint_zero_owners_is_refused_without_callbacks() {
        let mut cb = Fake::default();
        let err = checkpoint_all(&mut cb, &key(), 0, true).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Shape);
        assert!(cb.log.is_empty());
    }

    #[test]
    fn finalize_validates_every_owner_before_any_finalize() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        finalize_all(&mut cb, &k, cp).unwrap();
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Finalize(0),
                Ev::Finalize(1),
                Ev::Finalize(2),
                Ev::Finalize(3),
            ]
        );
    }

    #[test]
    fn finalize_without_head_skips_head_owner() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 2, false).unwrap();
        cb.log.clear();
        finalize_all(&mut cb, &k, cp).unwrap();
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Finalize(0),
                Ev::Finalize(1)
            ]
        );
    }

    #[test]
    fn validate_failure_blocks_all_finalize_and_restores_every_owner_with_head() {
        let mut cb = Fake {
            fail_validate: set(&[1]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 1));
        assert!(!err.poisoned());
        assert!(!cb.log.iter().any(|e| matches!(e, Ev::Finalize(_))));
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn head_validate_failure_restores_head_and_all_trunk_owners_without_finalize() {
        let mut cb = Fake {
            fail_validate: set(&[3]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 3));
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn validate_failure_restores_everyone_even_when_some_restores_fail() {
        let mut cb = Fake {
            fail_validate: set(&[0]),
            fail_restore: set(&[3, 1]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert!(err.poisoned());
        let failed: Vec<usize> = err.restore_failures.iter().map(|(o, _)| *o).collect();
        assert_eq!(failed, vec![3, 1]);
        assert_eq!(
            cb.log,
            vec![
                Ev::Validate(0),
                Ev::Validate(1),
                Ev::Validate(2),
                Ev::Validate(3),
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0),
            ]
        );
    }

    #[test]
    fn restore_all_attempts_every_owner_and_reports_all_failures() {
        let mut cb = Fake {
            fail_restore: set(&[2, 0]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = restore_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Restore, 2));
        assert!(err.poisoned());
        let failed: Vec<usize> = err.restore_failures.iter().map(|(o, _)| *o).collect();
        assert_eq!(failed, vec![2, 0]);
        assert_eq!(
            cb.log,
            vec![
                Ev::Restore(3),
                Ev::Restore(2),
                Ev::Restore(1),
                Ev::Restore(0)
            ]
        );
    }

    #[test]
    fn restore_all_clean_path_succeeds() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 2, true).unwrap();
        cb.log.clear();
        restore_all(&mut cb, &k, cp).unwrap();
        assert_eq!(cb.log, vec![Ev::Restore(2), Ev::Restore(1), Ev::Restore(0)]);
    }

    #[test]
    fn stale_window_generation_is_refused_before_any_callback() {
        let mut cb = Fake::default();
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 2, true).unwrap();
        cb.log.clear();
        let newer = MeshCheckpointKey {
            window_gen: k.window_gen + 1,
            ..k
        };
        let err = finalize_all(&mut cb, &newer, cp.clone()).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Shape);
        assert!(err.poisoned());
        let err = restore_all(&mut cb, &newer, cp).unwrap_err();
        assert_eq!(err.phase, MeshLifecyclePhase::Shape);
        assert!(cb.log.is_empty());
    }

    #[test]
    fn foreign_mesh_epoch_and_reset_generation_are_distinct_keys() {
        let k = key();
        let other_mesh = MeshCheckpointKey {
            mesh_epoch: DeviceMesh::single().unwrap().epoch(),
            ..k
        };
        let other_reset = MeshCheckpointKey {
            reset_gen: k.reset_gen + 1,
            ..k
        };
        assert_ne!(k, other_mesh);
        assert_ne!(k, other_reset);
        let id = MeshCheckpointId::issue(k, 0, 1);
        let cp = MeshCheckpoint {
            per_owner: vec![id],
            head: None,
        };
        assert!(cp.check_shape(&other_mesh).is_err());
        assert!(cp.check_shape(&other_reset).is_err());
        assert!(cp.check_shape(&k).is_ok());
    }

    #[test]
    fn ticket_for_wrong_owner_slot_is_rejected_by_shape_check() {
        let k = key();
        let cp = MeshCheckpoint {
            per_owner: vec![
                MeshCheckpointId::issue(k, 0, 1),
                MeshCheckpointId::issue(k, 0, 2),
            ],
            head: Some(MeshCheckpointId::issue(k, 1, 3)),
        };
        let err = cp.check_shape(&k).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Shape, 1));
        // head ticket must carry owner index == trunk owner count
        let cp = MeshCheckpoint {
            per_owner: vec![MeshCheckpointId::issue(k, 0, 1)],
            head: Some(MeshCheckpointId::issue(k, 0, 2)),
        };
        assert_eq!(cp.check_shape(&k).unwrap_err().owner, 1);
    }

    #[test]
    fn coordinator_works_through_dyn_callbacks() {
        let mut cb = Fake::default();
        let k = key();
        let dyn_cb: &mut dyn MeshCheckpointCallbacks = &mut cb;
        let cp = checkpoint_all(dyn_cb, &k, 2, false).unwrap();
        finalize_all(dyn_cb, &k, cp).unwrap();
        assert_eq!(cb.log.len(), 6);
    }

    #[test]
    fn early_validate_failure_still_validates_later_healthy_owners() {
        let mut cb = Fake {
            fail_validate: set(&[0]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, true).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!((err.phase, err.owner), (MeshLifecyclePhase::Validate, 0));
        for owner in 1..=3 {
            assert!(
                cb.log.contains(&Ev::Validate(owner)),
                "owner {owner} was not validated after an early failure"
            );
        }
        assert!(!cb.log.iter().any(|e| matches!(e, Ev::Finalize(_))));
    }

    #[test]
    fn first_of_several_validate_failures_is_reported() {
        let mut cb = Fake {
            fail_validate: set(&[1, 2]),
            ..Fake::default()
        };
        let k = key();
        let cp = checkpoint_all(&mut cb, &k, 3, false).unwrap();
        cb.log.clear();
        let err = finalize_all(&mut cb, &k, cp).unwrap_err();
        assert_eq!(err.owner, 1);
        assert!(cb.log.contains(&Ev::Validate(2)));
    }
}
