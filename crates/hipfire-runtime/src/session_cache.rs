// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Engine-owned cache of prefill snapshots shared by every session on one
//! loaded model.
//!
//! The cache owns keys, planning, eviction, the memory guard, storage
//! placement and every copy. An architecture only describes its live state
//! through [`SessionState`]: where a cold prefill materializes canonical state
//! ([`SessionState::snapshot_boundaries`], a function of the prompt tokens so
//! an architecture can also split at message ends) and which device byte
//! ranges plus host metadata make up that state
//! ([`SessionState::snapshot_parts`]). Drivers call [`SessionCache::begin`]
//! before prefill, split prefill at [`SessionCache::next_boundary`], report
//! each boundary through [`SessionCache::at_boundary`], and
//! [`SessionCache::commit`] once the turn's output reached the client. A turn
//! that continues the live state in place starts with
//! [`SessionCache::begin_live`], which arms boundaries only when asked to
//! capture.
//!
//! [`SessionDriver`] is what an architecture's bundle holds: the cache plus
//! the committed live state of the current conversation. It plans the longer
//! of the live state and a cached snapshot, starts each prefill as a live
//! continuation, a restore or a cold start, and keeps the live state after a
//! committed turn whose consumed history extends the turn's prompt.
//!
//! Snapshots are deltas: one stores the fixed (overwritten-in-place) state
//! whole, but of each append-only [`RowStream`] only the rows above its
//! parent, the deepest snapshot of the same prefix that was present when it
//! was captured. Restoring walks the chain from the root. Pool snapshots
//! never depend on what the live state held before; a snapshot that still
//! has children is pinned, so eviction only ever removes leaves.
//!
//! Besides the pool, the cache keeps fixed-only *resident* snapshots backed by
//! the live row streams: the deepest regular boundary and an optional lineage
//! split before the prompt's final message end. A tail restore preserves the
//! earlier resident's rows, but its descendants never enter the cold-exact
//! pool. Other restores or resets invalidate residents whose rows are rewritten
//! (external owners report resets through [`SessionCache::live_state_reset`]).
//!
//! An optional disk tier ([`SessionCache::with_disk`]) keeps snapshots the
//! device tier evicts (and every published snapshot at [`SessionCache::clear`])
//! under its own byte budget, across restarts of the same build; a restore
//! promotes them back. The parent of a device or pending snapshot is always
//! on the device (I1); the parent of a disk snapshot is on the device or on
//! disk (I2), so a chain's disk links are a suffix.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use hip_bridge::DeviceBuffer;
use rdna_compute::tensor_ops::{copy_regions, CopyRegion};
use rdna_compute::Gpu;

use crate::checkpoint_pool::{prefix_fingerprint, CheckpointBlob, CheckpointPool};
use crate::serve_contract::{CacheDomain, CheckpointId};

mod disk;

/// Byte alignment of each part inside a stored snapshot.
const PART_ALIGN: usize = 256;
/// Free device memory a discrete GPU's snapshot, and the live state's context
/// growth it must leave room for, keep untouched: an overshoot fails one
/// allocation. Unified memory (an APU's carve-out or GTT) sizes its headroom
/// from host RAM instead ([`uma_headroom`]): an overshoot can reach the
/// global OOM killer, so it keeps more.
const VRAM_HEADROOM: u64 = 256 << 20;

/// Which decode route the snapshot serves. Routes own different state (MTP
/// adds the draft head), so their snapshots never alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionRoute {
    Ar,
    Mtp,
}

/// One contiguous device byte range of live state that is overwritten in
/// place; every snapshot copies it whole.
pub struct StatePart<'a> {
    pub buf: &'a DeviceBuffer,
    pub offset: usize,
    pub bytes: usize,
}

/// Append-only rows of live state, from byte 0 of `buf`: `rows` valid rows of
/// `row_bytes` each. A row below a snapshot boundary is never rewritten by a
/// later prefill of the same prefix, so a snapshot shares its parent's rows.
pub struct RowStream<'a> {
    pub buf: &'a DeviceBuffer,
    pub row_bytes: usize,
    pub rows: usize,
}

/// Device ranges that together are the live state, in a fixed order.
pub struct StateLayout<'a> {
    pub fixed: Vec<StatePart<'a>>,
    pub rows: Vec<RowStream<'a>>,
}

/// What an architecture captures at its current position.
pub struct SnapshotParts<'a> {
    pub meta: Vec<u8>,
    pub layout: StateLayout<'a>,
}

/// Implemented by an architecture's model state. It only describes its state;
/// the cache owns keys, planning, placement, eviction and every copy.
pub trait SessionState {
    /// Everything besides the token prefix that decides whether a snapshot is
    /// reusable for `route` (route, prefill chunk, state formats). `None` = not
    /// cacheable now.
    fn snapshot_scope(&self, route: SessionRoute) -> Option<String>;
    /// Ascending, deduplicated positions p of `prompt` with
    /// `after < p <= up_to` (`up_to <= prompt.len()`) where a prefill of this
    /// prompt materializes state: chunk multiples, and whatever token-
    /// dependent split points (message ends) the architecture adds.
    fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize>;
    /// One additional split before `up_to`, reusable only by this live row
    /// lineage, never by the shared snapshot pool.
    fn lineage_boundary(&self, _prompt: &[u32], _after: usize, _up_to: usize) -> Option<usize> {
        None
    }
    /// Layout + host metadata of the live state, which must be exactly at
    /// `position`.
    fn snapshot_parts(
        &mut self,
        route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String>;
    /// Make the live state ready to receive the snapshot described by `meta`
    /// (map capacity, bump epochs) and return its destination layout, in
    /// capture order with the snapshot's row counts.
    fn restore_parts(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<StateLayout<'_>, String>;
    /// Apply host metadata after the device bytes were copied.
    fn finish_restore(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String>;
    /// The position every owner of `route` sits at when the live state can
    /// continue in place; `None` when it cannot (an owner lags, or verify
    /// state is armed).
    fn live_position(&self, route: SessionRoute) -> Option<usize>;
    /// Device bytes the live state may still map or allocate before it
    /// holds `through` tokens, the end of the prompt in flight (the cache
    /// leaves them free: it never takes memory the context needs).
    fn growth_reserve_bytes(&self, gpu: &Gpu, through: usize) -> u64;
    /// Cold-start the live state.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String>;
}

/// Where a snapshot's bytes live: device memory (VRAM, or system RAM on
/// UMA), or one file of the disk tier, `file_bytes` long (header plus
/// payload, what the disk budget counts).
enum SnapshotLocation {
    Device(DeviceBuffer),
    Disk { path: PathBuf, file_bytes: u64 },
}

/// `(scoped domain, boundary, prefix fingerprint)`, the pool's key.
type Key = (CacheDomain, u64, u64);

/// Opaque handle of one published pool snapshot (one band's piece of a mesh
/// session): the cache domain scoped by the architecture, the prefix length
/// and the prefix fingerprint. Built only by [`SessionDriver::piece_key`], so
/// it names exactly the entry [`SessionCache::plan`] and `restore` look up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionPieceKey(Key);

impl SessionPieceKey {
    /// Length of the prefix the piece covers.
    pub fn prefix_len(&self) -> usize {
        self.0 .1 as usize
    }

    /// Whether the piece covers exactly `prefix` (same length and
    /// fingerprint; the domain is the owning driver's).
    pub fn matches_prefix(&self, prefix: &[u32]) -> bool {
        self.prefix_len() == prefix.len() && self.0 .2 == prefix_fingerprint(prefix)
    }

    /// Whether both pieces cover the same prefix (length and fingerprint).
    fn same_prefix(&self, other: &Self) -> bool {
        self.0 .1 == other.0 .1 && self.0 .2 == other.0 .2
    }
}

/// Rows `[from, to)` of one row stream held by a snapshot.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Segment {
    row_bytes: usize,
    from: usize,
    to: usize,
}

impl Segment {
    fn bytes(&self) -> usize {
        (self.to - self.from) * self.row_bytes
    }
}

/// Fixed parts whole, then each stream's segment, packed at [`PART_ALIGN`].
struct StoredSnapshot {
    location: SnapshotLocation,
    /// Holds every stream's rows below `segments[i].from`.
    parent: Option<Key>,
    fixed_bytes: Vec<usize>,
    segments: Vec<Segment>,
    meta: Vec<u8>,
    bytes: u64,
}

impl StoredSnapshot {
    /// Byte offsets of the fixed parts, then of the segments.
    fn offsets(&self) -> Vec<usize> {
        layout(
            self.fixed_bytes
                .iter()
                .copied()
                .chain(self.segments.iter().map(Segment::bytes)),
        )
        .0
    }
}

impl CheckpointBlob for StoredSnapshot {
    fn bytes_len(&self) -> u64 {
        match self.location {
            SnapshotLocation::Device(_) => self.bytes,
            SnapshotLocation::Disk { file_bytes, .. } => file_bytes,
        }
    }
}

struct Turn {
    domain: CacheDomain,
    route: SessionRoute,
    boundaries: VecDeque<usize>,
    /// Extra split captured only in the tail resident.
    lineage: Option<usize>,
    /// False after any lineage split or restore: the pool stays cold-exact.
    pool_capture: bool,
    start: usize,
    /// Length of the prompt in flight.
    end: usize,
}

/// Fixed parts of the live state at boundary `key.1`; the live row streams
/// still hold its `rows`, so a restore copies the fixed parts only.
struct Resident {
    key: Key,
    buf: DeviceBuffer,
    fixed_bytes: Vec<usize>,
    rows: Vec<usize>,
    meta: Vec<u8>,
    cold_exact: bool,
}

pub struct SessionCache {
    pool: CheckpointPool<StoredSnapshot>,
    domain: CacheDomain,
    /// Captured this turn in ascending boundary order, published by
    /// [`Self::commit`].
    pending: Vec<(Key, StoredSnapshot)>,
    /// Displaced by `commit` (which has no GPU); freed at the next `begin`/`clear`.
    release: Vec<StoredSnapshot>,
    /// Snapshots (published or pending) that name each key as parent. A key
    /// listed here is pinned in the pool.
    children: HashMap<Key, usize>,
    turn: Option<Turn>,
    /// The published resident snapshot (module docs).
    resident: Option<Resident>,
    /// This turn's deepest boundary, published as `resident` by `commit`.
    resident_pending: Option<Resident>,
    /// Fixed state before the prompt's last message-end token.
    tail: Option<Resident>,
    tail_pending: Option<Resident>,
    /// Track split lineage even when the memory guard skips its resident.
    live_cold_exact: bool,
    /// Displaced resident buffers; freed at the next `begin`/`clear`.
    release_bufs: Vec<DeviceBuffer>,
    disk: Option<disk::DiskTier>,
}

/// Offsets of `part_bytes` packed at [`PART_ALIGN`], and the total size.
fn layout(part_bytes: impl IntoIterator<Item = usize>) -> (Vec<usize>, usize) {
    let mut offsets = Vec::new();
    let mut end = 0usize;
    for bytes in part_bytes {
        let offset = end.next_multiple_of(PART_ALIGN);
        offsets.push(offset);
        end = offset + bytes;
    }
    (offsets, end)
}

fn free_device(gpu: &mut Gpu, buf: DeviceBuffer) {
    if let Err(error) = gpu.hip.free(buf) {
        eprintln!("  session cache: freeing a snapshot failed: {error}");
    }
}

/// Free a device snapshot's buffer; a disk snapshot is never passed here.
fn free_buffer(gpu: &mut Gpu, snapshot: StoredSnapshot) {
    if let SnapshotLocation::Device(buf) = snapshot.location {
        free_device(gpu, buf);
    }
}

/// Copy `resident`'s fixed parts over the live state, whose row streams must
/// still hold its rows.
fn restore_resident(
    gpu: &mut Gpu,
    state: &mut dyn SessionState,
    route: SessionRoute,
    resident: &Resident,
) -> Result<(), String> {
    let dst = state.restore_parts(gpu, route, &resident.meta)?;
    let fixed = dst.fixed.iter().map(|part| part.bytes);
    let rows = dst.rows.iter().map(|stream| stream.rows);
    if !fixed.eq(resident.fixed_bytes.iter().copied()) || !rows.eq(resident.rows.iter().copied()) {
        return Err("session cache: resident snapshot layout mismatch".to_string());
    }
    let regions: Vec<CopyRegion<'_>> = dst
        .fixed
        .iter()
        .zip(layout(resident.fixed_bytes.iter().copied()).0)
        .map(|(part, src_offset)| CopyRegion {
            dst: part.buf,
            dst_offset: part.offset,
            src: &resident.buf,
            src_offset,
            bytes: part.bytes,
        })
        .collect();
    copy_regions(gpu, &regions).map_err(|e| e.to_string())?;
    state.finish_restore(gpu, route, &resident.meta)
}

/// Lower bound of the unified-memory headroom.
const UMA_HEADROOM_MIN: u64 = 2 << 30;
/// Upper bound of the unified-memory headroom, and the value taken when host
/// RAM cannot be read: the larger guard is the conservative one, and the cache
/// is opportunistic, so an over-cautious guard only skips a snapshot.
const UMA_HEADROOM_MAX: u64 = 8 << 30;
/// Unified-memory headroom is this fraction (`1 / UMA_HEADROOM_DIVISOR`) of
/// host RAM.
const UMA_HEADROOM_DIVISOR: u64 = 16;

/// Headroom of a unified-memory device: `MemTotal / 16` clamped to
/// `[2, 8]` GiB; [`UMA_HEADROOM_MAX`] when `MemTotal` is unreadable.
fn uma_headroom(mem_total: Option<u64>) -> u64 {
    mem_total.map_or(UMA_HEADROOM_MAX, |total| {
        (total / UMA_HEADROOM_DIVISOR).clamp(UMA_HEADROOM_MIN, UMA_HEADROOM_MAX)
    })
}

/// Headroom of the pool the device allocates from: [`uma_headroom`] on
/// unified memory, the fixed [`VRAM_HEADROOM`] on a discrete GPU.
fn headroom_for(uma: bool, mem_total: Option<u64>) -> u64 {
    if uma {
        uma_headroom(mem_total)
    } else {
        VRAM_HEADROOM
    }
}

/// Whether `free` bytes hold `need` plus the live state's `growth` reserve
/// plus `headroom`. A sum that overflows `u64` never fits.
fn fits_with_headroom(free: u64, need: u64, growth: u64, headroom: u64) -> bool {
    need.checked_add(growth)
        .and_then(|bytes| bytes.checked_add(headroom))
        .is_some_and(|required| free >= required)
}

/// Sum of `bytes`, saturating at `u64::MAX` (which never fits).
fn saturating_total(bytes: impl IntoIterator<Item = u64>) -> u64 {
    bytes.into_iter().fold(0, u64::saturating_add)
}

/// Whether `need` more device bytes, with `growth` more for the live state,
/// leave the guard's headroom free in the pool the device allocates from
/// (`Gpu::device_mem_info`: VRAM, a unified-memory APU's carve-out, or its GTT
/// clamped to `MemAvailable` minus `CmaFree` plus TTM's page pool); `None`
/// when the free-memory query itself fails. A caller allocates only right
/// after `Some(true)`.
fn memory_fits(gpu: &Gpu, need: u64, growth: u64) -> Option<bool> {
    let (free, _) = gpu.device_mem_info().ok()?;
    let uma = gpu.is_uma();
    let mem_total = if uma {
        rdna_compute::uma_memory::host_total_bytes()
    } else {
        None
    };
    Some(fits_with_headroom(
        free as u64,
        need,
        growth,
        headroom_for(uma, mem_total),
    ))
}

/// Whether the disk tier's host bounce buffer of `bytes` may be allocated now:
/// host `MemAvailable` (less `CmaFree`, no TTM credit: pageable memory is not
/// GTT) must hold it plus the headroom and, on unified memory where the live
/// state's `growth` comes out of the same RAM, that reserve. An unreadable
/// `MemAvailable` refuses.
fn host_staging_fits(gpu: &Gpu, bytes: u64, growth: u64) -> bool {
    let uma = gpu.is_uma();
    let mem_total = if uma {
        rdna_compute::uma_memory::host_total_bytes()
    } else {
        None
    };
    let growth = if uma { growth } else { 0 };
    rdna_compute::uma_memory::mem_available_bytes().is_some_and(|available| {
        fits_with_headroom(available, bytes, growth, headroom_for(uma, mem_total))
    })
}

impl SessionCache {
    pub fn new(domain: CacheDomain, budget_bytes: u64) -> Self {
        Self {
            pool: CheckpointPool::new(budget_bytes),
            domain,
            pending: Vec::new(),
            release: Vec::new(),
            children: HashMap::new(),
            turn: None,
            resident: None,
            resident_pending: None,
            tail: None,
            tail_pending: None,
            live_cold_exact: true,
            release_bufs: Vec::new(),
            disk: None,
        }
    }

    /// Attach a disk tier in `dir` with `budget_bytes` (0 = none), bound to
    /// this executable's build.
    pub fn with_disk(self, dir: &Path, budget_bytes: u64) -> Self {
        if budget_bytes == 0 {
            return self;
        }
        match disk::build_digest() {
            Ok(build) => self.with_disk_build(dir, budget_bytes, build),
            Err(reason) => {
                eprintln!("  session cache disk tier off: {reason}");
                self
            }
        }
    }

    fn with_disk_build(mut self, dir: &Path, budget: u64, build: [u8; 32]) -> Self {
        match disk::DiskTier::open(dir, budget, build) {
            Ok(tier) => {
                eprintln!(
                    "  session cache disk tier: {} ({} MiB budget, {} snapshots / {} MiB indexed, {} other-build files)",
                    dir.display(),
                    budget >> 20,
                    tier.pool.len(),
                    tier.pool.total_bytes() >> 20,
                    tier.foreign_files()
                );
                self.disk = Some(tier);
            }
            Err(reason) => eprintln!("  session cache disk tier off: {reason}"),
        }
        self
    }

    /// Published on the device or on disk.
    fn present(&self, domain: &CacheDomain, p: u64, fp: u64) -> bool {
        self.pool.contains(domain, p, fp)
            || self
                .disk
                .as_ref()
                .is_some_and(|disk| disk.pool.contains(domain, p, fp))
    }

    /// Longest cached prefix of `prompt` (always shorter than the prompt, so
    /// prefill computes at least the last token); 0 on a miss.
    pub fn plan(&self, state: &dyn SessionState, prompt: &[u32], route: SessionRoute) -> usize {
        let Some(scope) = state.snapshot_scope(route) else {
            return 0;
        };
        if prompt.len() < 2 {
            return 0;
        }
        let domain = self.domain.scoped(&scope);
        let resident = [self.resident.as_ref(), self.tail.as_ref()]
            .into_iter()
            .flatten()
            .map(|resident| &resident.key)
            .filter(|(d, p, fp)| {
                let p = *p as usize;
                *d == domain && p < prompt.len() && *fp == prefix_fingerprint(&prompt[..p])
            })
            .map(|key| key.1 as usize)
            .max()
            .unwrap_or(0);
        // Only a strictly deeper pool snapshot beats the resident one (no row
        // copy, see `begin`); each probe hashes its whole prefix.
        let planned = state
            .snapshot_boundaries(prompt, resident, prompt.len() - 1)
            .into_iter()
            .rev()
            .find(|&p| self.present(&domain, p as u64, prefix_fingerprint(&prompt[..p])))
            .unwrap_or(resident);
        if planned > 0
            && hipfire_config::developer_bool("HIPFIRE_QWEN_CACHE_TRACE", false)
            && self.tail.as_ref().is_some_and(|tail| {
                tail.key.0 == domain
                    && tail.key.1 as usize == planned
                    && tail.key.2 == prefix_fingerprint(&prompt[..planned])
            })
        {
            eprintln!("  session cache: using tail resident at {planned}");
        }
        planned
    }

    /// A published or pending snapshot.
    fn entry(&self, key: &Key) -> Option<&StoredSnapshot> {
        self.pool.peek(&key.0, key.1, key.2).or_else(|| {
            self.pending
                .iter()
                .find(|(pending, _)| pending == key)
                .map(|(_, snapshot)| snapshot)
        })
    }

    /// Record a child of `parent`, pinning it against eviction.
    fn link(&mut self, parent: &Key) {
        *self.children.entry(parent.clone()).or_default() += 1;
        self.pool.pin(&parent.0, parent.1, parent.2);
    }

    /// Drop a child of `parent`; its last child unpins it.
    fn unlink(&mut self, parent: &Key) {
        if let Some(count) = self.children.get_mut(parent) {
            *count -= 1;
            if *count == 0 {
                self.children.remove(parent);
                self.pool.unpin(&parent.0, parent.1, parent.2);
            }
        }
    }

    /// Unlink a snapshot that leaves the cache from its parent and free it.
    fn drop_snapshot(&mut self, gpu: &mut Gpu, snapshot: StoredSnapshot) {
        if let Some(parent) = &snapshot.parent {
            self.unlink(parent);
        }
        free_buffer(gpu, snapshot);
    }

    /// A published device snapshot leaves the device tier: demote it to disk
    /// when there is a disk tier (or drop its disk descendants when that
    /// fails), then free it.
    fn evict_device(&mut self, gpu: &mut Gpu, key: Key, snapshot: StoredSnapshot, growth: u64) {
        if let Some(parent) = &snapshot.parent {
            self.unlink(parent);
        }
        if let Some(disk) = &mut self.disk {
            if let Err(reason) = disk.demote(gpu, &key, &snapshot, growth) {
                eprintln!(
                    "  session cache: dropped {}-token snapshot instead of demoting it ({reason})",
                    key.1
                );
                disk.drop_descendants(&key);
            }
        }
        free_buffer(gpu, snapshot);
    }

    /// Evict device snapshots until `bytes` more fit the budget (counting
    /// pending snapshots) and the memory guard (counting `growth`).
    fn make_room(&mut self, gpu: &mut Gpu, bytes: u64, growth: u64) -> Result<(), &'static str> {
        let pending_bytes: u64 = self.pending.iter().map(|entry| entry.1.bytes).sum();
        loop {
            let over_budget = self
                .pool
                .total_bytes()
                .saturating_add(pending_bytes)
                .saturating_add(bytes)
                > self.pool.max_bytes();
            if !over_budget {
                match memory_fits(gpu, bytes, growth) {
                    Some(true) => return Ok(()),
                    Some(false) => {}
                    // Evicting cannot help a query that fails.
                    None => return Err("memory query failed"),
                }
            }
            match self.pool.pop_lru() {
                Some((key, evicted)) => self.evict_device(gpu, key, evicted, growth),
                None => {
                    return Err(if over_budget {
                        "budget"
                    } else {
                        "memory guard"
                    })
                }
            }
        }
    }

    /// Unlink a snapshot that leaves the cache; free it at the next `begin`.
    fn retire(&mut self, snapshot: StoredSnapshot) {
        if let Some(parent) = &snapshot.parent {
            self.unlink(parent);
        }
        self.release.push(snapshot);
    }

    /// Drop the resident snapshot: the live rows are about to change.
    fn drop_resident(&mut self, gpu: &mut Gpu) {
        if let Some(resident) = self.resident.take() {
            free_device(gpu, resident.buf);
        }
        if let Some(tail) = self.tail.take() {
            free_device(gpu, tail.buf);
        }
    }

    /// The live state was reset outside the cache: its rows no longer back
    /// the resident snapshot.
    pub fn live_state_reset(&mut self) {
        self.live_cold_exact = true;
        for resident in [
            self.resident.take(),
            self.resident_pending.take(),
            self.tail.take(),
            self.tail_pending.take(),
        ]
        .into_iter()
        .flatten()
        {
            self.release_bufs.push(resident.buf);
        }
    }

    /// Bookkeeping head of every turn: free displaced buffers and drop the
    /// previous turn's uncommitted captures.
    fn start_turn(&mut self, gpu: &mut Gpu) {
        for snapshot in std::mem::take(&mut self.release) {
            free_buffer(gpu, snapshot);
        }
        for buf in std::mem::take(&mut self.release_bufs) {
            free_device(gpu, buf);
        }
        for (_, snapshot) in std::mem::take(&mut self.pending) {
            self.drop_snapshot(gpu, snapshot);
        }
        if let Some(resident) = self.resident_pending.take() {
            free_device(gpu, resident.buf);
        }
        if let Some(tail) = self.tail_pending.take() {
            free_device(gpu, tail.buf);
        }
        self.turn = None;
    }

    /// Start a prefill of `prompt`: restore the `reused`-token snapshot
    /// [`Self::plan`] returned (or cold-start the state for 0) and arm the
    /// boundaries this prefill crosses.
    pub fn begin(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
    ) -> Result<(), String> {
        self.start_turn(gpu);
        let Some(scope) = state.snapshot_scope(route) else {
            if reused > 0 {
                return Err(format!(
                    "session cache: route {route:?} has no snapshot scope"
                ));
            }
            self.drop_resident(gpu);
            return state.reset(gpu);
        };
        let domain = self.domain.scoped(&scope);
        let key = (
            domain.clone(),
            reused as u64,
            prefix_fingerprint(&prompt[..reused]),
        );
        let mut pool_capture = true;
        let restored = if reused > 0 && self.tail.as_ref().is_some_and(|r| r.key == key) {
            pool_capture = false;
            // Message-end snapshots may sit above the tail when enabled.
            if self.resident.as_ref().is_some_and(|r| r.key.1 > reused as u64) {
                free_device(gpu, self.resident.take().unwrap().buf);
            }
            restore_resident(gpu, state, route, self.tail.as_ref().unwrap())
        } else if reused > 0 && self.resident.as_ref().is_some_and(|r| r.key == key) {
            if let Some(tail) = self.tail.take() {
                free_device(gpu, tail.buf);
            }
            let resident = self.resident.as_ref().unwrap();
            pool_capture = resident.cold_exact;
            restore_resident(gpu, state, route, resident)
        } else {
            self.drop_resident(gpu);
            if reused == 0 {
                state.reset(gpu)
            } else {
                self.restore(gpu, state, route, &domain, &prompt[..reused], prompt.len())
            }
        };
        if let Err(error) = restored {
            let _ = state.reset(gpu);
            self.drop_resident(gpu);
            return Err(error);
        }
        self.live_cold_exact = pool_capture;
        // A turn restored from the tail does not split at its own tail: that
        // extra small tile would land on every chained append, and a retry of
        // this turn still restores the earlier tail (it stays valid below).
        let tail_restore = reused > 0 && self.tail.as_ref().is_some_and(|r| r.key == key);
        let (boundaries, lineage) = Self::turn_boundaries(state, prompt, reused, !tail_restore);
        self.turn = Some(Turn {
            boundaries,
            lineage,
            pool_capture,
            start: reused,
            domain,
            route,
            end: prompt.len(),
        });
        Ok(())
    }

    /// Start a prefill that continues the live state in place: the caller
    /// keeps every owner as it is (no reset, no restore) and prefills only the
    /// suffix. Does the bookkeeping head of [`Self::begin`] (frees displaced
    /// buffers, drops uncommitted snapshots). The resident snapshot stays: a
    /// continuation only appends rows above it.
    ///
    /// With `capture == false` it arms no boundary, so
    /// [`Self::next_boundary`] is `None` and nothing is captured this turn: a
    /// live-continued turn's state descends from decode, never from a cold
    /// canonical prefill, so it must not enter the cross-session cache.
    ///
    /// With `capture == true` it arms `snapshot_boundaries(prompt, reused,
    /// prompt.len())` under the route's scoped domain exactly like
    /// [`Self::begin`] (still no reset, no restore), and the captures publish
    /// at [`Self::commit`]. The caller passes `true` only for boundaries the
    /// architecture defines as session-lineage reusable (message ends), whose
    /// scope never aliases cold-exact snapshots.
    pub fn begin_live(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
        capture: bool,
    ) {
        self.start_turn(gpu);
        if self.tail.as_ref().is_some_and(|tail| reused < tail.key.1 as usize) {
            free_device(gpu, self.tail.take().unwrap().buf);
        }
        if !capture {
            return;
        }
        let Some(scope) = state.snapshot_scope(route) else {
            return;
        };
        let (boundaries, lineage) = Self::turn_boundaries(state, prompt, reused, true);
        self.turn = Some(Turn {
            boundaries,
            lineage,
            pool_capture: self.live_cold_exact,
            start: reused,
            domain: self.domain.scoped(&scope),
            route,
            end: prompt.len(),
        });
    }

    fn turn_boundaries(
        state: &dyn SessionState,
        prompt: &[u32],
        reused: usize,
        lineage_split: bool,
    ) -> (VecDeque<usize>, Option<usize>) {
        let mut boundaries = state.snapshot_boundaries(prompt, reused, prompt.len());
        let lineage = lineage_split
            .then(|| state.lineage_boundary(prompt, reused, prompt.len()))
            .flatten()
            .filter(|&p| reused < p && p < prompt.len());
        if let Some(p) = lineage {
            boundaries.push(p);
            boundaries.sort_unstable();
            boundaries.dedup();
        }
        (boundaries.into(), lineage)
    }

    /// Copy the snapshot of `prefix` and its ancestors into the live state,
    /// promoting disk links to the device (or staging them in temporary
    /// buffers when the device tier has no room). `end` is the length of the
    /// prompt in flight, the context the state grows through.
    fn restore(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        route: SessionRoute,
        domain: &CacheDomain,
        prefix: &[u32],
        end: usize,
    ) -> Result<(), String> {
        let missing = format!(
            "session cache: planned {}-token snapshot is no longer present",
            prefix.len()
        );
        // Leaf first; each `get` refreshes that link's LRU stamp.
        let mut keys = vec![(
            domain.clone(),
            prefix.len() as u64,
            prefix_fingerprint(prefix),
        )];
        loop {
            let key = keys.last().expect("non-empty chain");
            let parent = match self.pool.get(&key.0, key.1, key.2) {
                Some(snapshot) => snapshot.parent.clone(),
                None => self
                    .disk
                    .as_mut()
                    .and_then(|disk| disk.get(key))
                    .ok_or_else(|| missing.clone())?
                    .parent
                    .clone(),
            };
            match parent {
                Some(parent) => keys.push(parent),
                None => break,
            }
        }
        keys.reverse();
        let growth = state.growth_reserve_bytes(gpu, end);
        if let Some(disk) = &mut self.disk {
            for key in &keys {
                if disk.contains(key) {
                    disk.held.insert(key.clone());
                    disk.pool.pin(&key.0, key.1, key.2);
                }
            }
        }
        let mut temps = Vec::new();
        let result = self.restore_chain(gpu, state, route, &keys, growth, &missing, &mut temps);
        if !temps.is_empty() {
            let _ = gpu.hip.device_synchronize();
            for (_, buf) in temps {
                if let Err(error) = gpu.hip.free(buf) {
                    eprintln!("  session cache: freeing a staging buffer failed: {error}");
                }
            }
        }
        if let Some(disk) = &mut self.disk {
            for key in std::mem::take(&mut disk.held) {
                disk.repin(&key);
            }
        }
        result
    }

    /// Bring the disk links of the root-first chain `keys` to the device
    /// (promoted, or staged into `temps`), then copy the chain into the
    /// live state.
    #[allow(clippy::too_many_arguments)]
    fn restore_chain(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        route: SessionRoute,
        keys: &[Key],
        growth: u64,
        missing: &str,
        temps: &mut Vec<(Key, DeviceBuffer)>,
    ) -> Result<(), String> {
        let first_disk = keys
            .iter()
            .position(|key| !self.pool.contains(&key.0, key.1, key.2))
            .unwrap_or(keys.len());
        let mut promoting = true;
        for (i, key) in keys.iter().enumerate().skip(first_disk) {
            let disk = self
                .disk
                .as_ref()
                .expect("a link off the device is on disk");
            let snapshot = disk.peek(key).ok_or_else(|| missing.to_string())?;
            let (bytes, parent) = (snapshot.bytes, snapshot.parent.clone());
            if promoting {
                if self.promote(gpu, key, bytes, parent.as_ref(), growth)? {
                    continue;
                }
                promoting = false;
            }
            let disk = self
                .disk
                .as_mut()
                .expect("a link off the device is on disk");
            // The host bounce buffer is committed first, so the device
            // free-memory query below already counts it.
            disk.reserve_staging(gpu, growth)
                .map_err(|e| format!("session cache: no memory to stage disk snapshot: {e}"))?;
            // Checked right before every staging allocation: the remaining
            // chain (this link included) plus the context growth must still
            // leave the headroom free, counting the temps already staged.
            let need = saturating_total(
                keys[i..]
                    .iter()
                    .filter_map(|key| disk.peek(key).map(|s| s.bytes)),
            );
            if memory_fits(gpu, need, growth) != Some(true) {
                return Err("session cache: no memory to stage disk snapshot".into());
            }
            let buf = gpu
                .bind_thread()
                .and_then(|()| gpu.hip.malloc(bytes as usize))
                .map_err(|e| format!("session cache: staging a disk snapshot failed: {e}"))?;
            temps.push((key.clone(), buf));
            let buf = &temps.last().expect("just pushed").1;
            if let Err(error) = disk.load(gpu, key, buf) {
                disk.drop_entry(key);
                return Err(format!(
                    "session cache: reading snapshot from disk failed: {error}"
                ));
            }
        }

        let chain: Vec<(&StoredSnapshot, &DeviceBuffer)> = keys
            .iter()
            .map(|key| {
                if let Some((_, buf)) = temps.iter().find(|(temp, _)| temp == key) {
                    let snapshot = self.disk.as_ref().and_then(|disk| disk.peek(key));
                    return snapshot
                        .map(|s| (s, buf))
                        .ok_or_else(|| missing.to_string());
                }
                match self.pool.peek(&key.0, key.1, key.2) {
                    Some(
                        s @ StoredSnapshot {
                            location: SnapshotLocation::Device(buf),
                            ..
                        },
                    ) => Ok((s, buf)),
                    _ => Err(missing.to_string()),
                }
            })
            .collect::<Result<_, _>>()?;
        let (leaf, leaf_src) = *chain.last().expect("non-empty chain");
        let dst = state.restore_parts(gpu, route, &leaf.meta)?;
        let mismatch = || "session cache: snapshot layout mismatch".to_string();
        if dst.fixed.len() != leaf.fixed_bytes.len()
            || dst
                .fixed
                .iter()
                .zip(&leaf.fixed_bytes)
                .any(|(part, &bytes)| part.bytes != bytes)
            || dst.rows.len() != leaf.segments.len()
        {
            return Err(mismatch());
        }
        let mut regions = Vec::new();
        for (part, src_offset) in dst.fixed.iter().zip(leaf.offsets()) {
            regions.push(CopyRegion {
                dst: part.buf,
                dst_offset: part.offset,
                src: leaf_src,
                src_offset,
                bytes: part.bytes,
            });
        }
        let mut next = vec![0usize; dst.rows.len()];
        for &(snapshot, src) in &chain {
            let offsets = snapshot.offsets();
            if snapshot.segments.len() != dst.rows.len() {
                return Err(mismatch());
            }
            for (i, (segment, stream)) in snapshot.segments.iter().zip(&dst.rows).enumerate() {
                if segment.row_bytes != stream.row_bytes || segment.from != next[i] {
                    return Err(mismatch());
                }
                next[i] = segment.to;
                regions.push(CopyRegion {
                    dst: stream.buf,
                    dst_offset: segment.from * segment.row_bytes,
                    src,
                    src_offset: offsets[snapshot.fixed_bytes.len() + i],
                    bytes: segment.bytes(),
                });
            }
        }
        if next
            .iter()
            .zip(&dst.rows)
            .any(|(&rows, stream)| rows != stream.rows)
        {
            return Err(mismatch());
        }
        copy_regions(gpu, &regions).map_err(|e| e.to_string())?;
        state.finish_restore(gpu, route, &leaf.meta)
    }

    /// Move the held disk snapshot at `key` into the device tier. `Ok(false)`
    /// when the device tier has no room for it (nothing changed); `Err` when
    /// reading it failed (it and its disk descendants are dropped).
    fn promote(
        &mut self,
        gpu: &mut Gpu,
        key: &Key,
        bytes: u64,
        parent: Option<&Key>,
        growth: u64,
    ) -> Result<bool, String> {
        if !self.pool.can_afford(key, bytes) {
            return Ok(false);
        }
        // The disk read bounces through host staging memory: refuse, with
        // the snapshot still on disk, when that allocation would not fit.
        let disk = self.disk.as_mut().expect("promoting from disk");
        if disk.reserve_staging(gpu, growth).is_err() {
            return Ok(false);
        }
        // The parent is on the device (I2 for the first disk link, promoted
        // for the rest); pinned before eviction runs.
        if let Some(parent) = parent {
            self.link(parent);
        }
        let buf = self.make_room(gpu, bytes, growth).ok().and_then(|()| {
            gpu.bind_thread()
                .and_then(|()| gpu.hip.malloc(bytes as usize))
                .ok()
        });
        let Some(buf) = buf else {
            if let Some(parent) = parent {
                self.unlink(parent);
            }
            return Ok(false);
        };
        let disk = self.disk.as_mut().expect("promoting from disk");
        if let Err(error) = disk.load(gpu, key, &buf) {
            disk.drop_entry(key);
            if let Some(parent) = parent {
                self.unlink(parent);
            }
            if let Err(error) = gpu.hip.free(buf) {
                eprintln!("  session cache: freeing a snapshot failed: {error}");
            }
            return Err(format!(
                "session cache: reading snapshot from disk failed: {error}"
            ));
        }
        disk.held.remove(key);
        let stored = disk.peek(key).expect("held on disk");
        let snapshot = StoredSnapshot {
            location: SnapshotLocation::Device(buf),
            parent: stored.parent.clone(),
            fixed_bytes: stored.fixed_bytes.clone(),
            segments: stored.segments.clone(),
            meta: stored.meta.clone(),
            bytes: stored.bytes,
        };
        disk.take(key);
        let (domain, p, fp) = key.clone();
        let (_, displaced) = self.pool.insert_unaligned(domain, p, fp, snapshot);
        for snapshot in displaced {
            self.retire(snapshot);
        }
        if self.children.contains_key(key) {
            self.pool.pin(&key.0, key.1, key.2);
        }
        Ok(true)
    }

    /// The next position at which the prefill must stop and call
    /// [`Self::at_boundary`].
    pub fn next_boundary(&self) -> Option<usize> {
        self.turn.as_ref()?.boundaries.front().copied()
    }

    /// First still-pending boundary strictly after `position`. A pipelined
    /// architecture uses this to reject an interior bank-B boundary without
    /// consuming the bank-A boundary or changing the capture cadence.
    pub fn next_boundary_after(&self, position: usize) -> Option<usize> {
        self.turn
            .as_ref()?
            .boundaries
            .iter()
            .copied()
            .find(|&p| p > position)
    }

    /// The turn's lineage split row (the tail resident's boundary), if armed.
    /// A pipelined architecture ends a pair group there instead of carrying it
    /// as a second interior row.
    pub fn turn_lineage(&self) -> Option<usize> {
        self.turn.as_ref()?.lineage
    }

    /// Boundaries still pending in the armed turn, ascending; empty without a
    /// turn.
    fn pending_boundaries(&self) -> Vec<usize> {
        self.turn
            .as_ref()
            .map(|turn| turn.boundaries.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Replace the armed turn's boundary schedule and lineage split with a
    /// schedule computed elsewhere. Nothing changes on error. Without an armed
    /// turn only the empty schedule is accepted.
    fn override_boundaries(
        &mut self,
        boundaries: Vec<usize>,
        lineage: Option<usize>,
    ) -> Result<(), String> {
        let Some(turn) = self.turn.as_mut() else {
            return if boundaries.is_empty() && lineage.is_none() {
                Ok(())
            } else {
                Err("session cache: no prefill in progress to set boundaries on".to_string())
            };
        };
        if boundaries.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!(
                "session cache: boundaries {boundaries:?} are not strictly increasing"
            ));
        }
        if let Some(&first) = boundaries.first() {
            if first <= turn.start {
                return Err(format!(
                    "session cache: boundary {first} is not above the turn start {}",
                    turn.start
                ));
            }
        }
        if let Some(&last) = boundaries.last() {
            if last > turn.end {
                return Err(format!(
                    "session cache: boundary {last} exceeds the turn end {}",
                    turn.end
                ));
            }
        }
        if let Some(split) = lineage {
            if !(turn.start < split && split < turn.end) || !boundaries.contains(&split) {
                return Err(format!(
                    "session cache: lineage split {split} is not an interior boundary of {boundaries:?}"
                ));
            }
        }
        turn.boundaries = boundaries.into();
        turn.lineage = lineage;
        Ok(())
    }

    /// The live state reached `prefix.len()`, the boundary
    /// [`Self::next_boundary`] returned; capture it (pending until commit) as
    /// a delta over the deepest snapshot of a shorter prefix still present,
    /// and as this turn's resident snapshot.
    pub fn at_boundary(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prefix: &[u32],
    ) -> Result<(), String> {
        let turn = self
            .turn
            .as_mut()
            .ok_or("session cache: no prefill in progress")?;
        if turn.boundaries.front() != Some(&prefix.len()) {
            return Err(format!(
                "session cache: driver reported position {} but next boundary is {:?}",
                prefix.len(),
                turn.boundaries.front()
            ));
        }
        turn.boundaries.pop_front();
        let lineage = turn.lineage == Some(prefix.len());
        if lineage {
            turn.pool_capture = false;
            self.live_cold_exact = false;
        }
        let pool_capture = turn.pool_capture;
        let (domain, route, p, end) = (turn.domain.clone(), turn.route, prefix.len(), turn.end);
        let fp = prefix_fingerprint(prefix);
        let in_pool = !pool_capture || self.present(&domain, p as u64, fp);
        let growth = state.growth_reserve_bytes(gpu, end);
        let ancestors: Vec<Key> = if in_pool {
            Vec::new()
        } else {
            state
                .snapshot_boundaries(prefix, 0, p - 1)
                .into_iter()
                .rev()
                .map(|b| (domain.clone(), b as u64, prefix_fingerprint(&prefix[..b])))
                .collect()
        };
        let SnapshotParts { meta, layout: live } = state.snapshot_parts(route, p)?;
        let pending = if lineage {
            &mut self.tail_pending
        } else {
            &mut self.resident_pending
        };
        Self::capture_resident(
            pending, gpu, (domain.clone(), p as u64, fp), &meta, &live, growth, pool_capture,
        )?;
        if lineage
            && self.tail_pending.as_ref().is_some_and(|tail| tail.key.1 == p as u64)
            && hipfire_config::developer_bool("HIPFIRE_QWEN_CACHE_TRACE", false)
        {
            eprintln!("  session cache: tail resident at {p}");
        }
        if in_pool {
            return Ok(());
        }
        // The parent's segment ends are this capture's starts; a parent whose
        // streams do not line up is ignored (rows from 0).
        let parent = ancestors.into_iter().find_map(|key| {
            let segments = &self.entry(&key)?.segments;
            let fits = segments.len() == live.rows.len()
                && segments.iter().zip(&live.rows).all(|(segment, stream)| {
                    segment.row_bytes == stream.row_bytes && segment.to <= stream.rows
                });
            fits.then(|| (key, segments.iter().map(|s| s.to).collect::<Vec<_>>()))
        });
        let segments: Vec<Segment> = live
            .rows
            .iter()
            .enumerate()
            .map(|(i, stream)| Segment {
                row_bytes: stream.row_bytes,
                from: parent.as_ref().map_or(0, |(_, ends)| ends[i]),
                to: stream.rows,
            })
            .collect();
        let parent = parent.map(|(key, _)| key);
        let fixed_bytes: Vec<usize> = live.fixed.iter().map(|part| part.bytes).collect();
        let (offsets, total) = layout(
            fixed_bytes
                .iter()
                .copied()
                .chain(segments.iter().map(Segment::bytes)),
        );
        let bytes = total as u64;
        let skip =
            |reason: &str| eprintln!("  session cache: skipped {p}-token snapshot ({reason})");
        if bytes > self.pool.max_bytes() {
            skip("exceeds budget");
            return Ok(());
        }
        // Pinned before eviction runs, so making room never removes it.
        if let Some(parent) = &parent {
            self.link(parent);
        }
        let fits = self.make_room(gpu, bytes, growth);
        let dst = match fits.and_then(|()| {
            gpu.bind_thread()
                .and_then(|()| gpu.hip.malloc(total))
                .map_err(|_| "allocation")
        }) {
            Ok(dst) => dst,
            Err(reason) => {
                if let Some(parent) = &parent {
                    self.unlink(parent);
                }
                skip(reason);
                return Ok(());
            }
        };
        let fixed = live
            .fixed
            .iter()
            .map(|part| (part.buf, part.offset, part.bytes));
        let rows = live.rows.iter().zip(&segments).map(|(stream, segment)| {
            (
                stream.buf,
                segment.from * segment.row_bytes,
                segment.bytes(),
            )
        });
        let regions: Vec<CopyRegion<'_>> = fixed
            .chain(rows)
            .zip(offsets)
            .map(|((src, src_offset, bytes), dst_offset)| CopyRegion {
                dst: &dst,
                dst_offset,
                src,
                src_offset,
                bytes,
            })
            .collect();
        let copied = copy_regions(gpu, &regions);
        let snapshot = StoredSnapshot {
            location: SnapshotLocation::Device(dst),
            parent,
            fixed_bytes,
            segments,
            meta,
            bytes,
        };
        if let Err(error) = copied {
            self.drop_snapshot(gpu, snapshot);
            return Err(format!("session cache: snapshot copy failed: {error}"));
        }
        self.pending.push(((domain, p as u64, fp), snapshot));
        Ok(())
    }

    /// Copy the fixed parts of `live` (at boundary `key.1`) into this turn's
    /// resident snapshot, reusing its buffer. A capture the memory guard
    /// refuses is skipped and leaves the previous boundary's.
    fn capture_resident(
        pending: &mut Option<Resident>,
        gpu: &mut Gpu,
        key: Key,
        meta: &[u8],
        live: &StateLayout<'_>,
        growth: u64,
        cold_exact: bool,
    ) -> Result<(), String> {
        let fixed_bytes: Vec<usize> = live.fixed.iter().map(|part| part.bytes).collect();
        let (offsets, total) = layout(fixed_bytes.iter().copied());
        let buf = match pending.take() {
            Some(resident) if resident.buf.size() == total => resident.buf,
            previous => {
                if memory_fits(gpu, total as u64, growth) != Some(true) {
                    eprintln!(
                        "  session cache: skipped {}-token resident snapshot (memory guard)",
                        key.1
                    );
                    *pending = previous;
                    return Ok(());
                }
                if let Some(resident) = previous {
                    free_device(gpu, resident.buf);
                }
                gpu.bind_thread()
                    .and_then(|()| gpu.hip.malloc(total))
                    .map_err(|e| format!("session cache: resident snapshot allocation: {e}"))?
            }
        };
        let regions: Vec<CopyRegion<'_>> = live
            .fixed
            .iter()
            .zip(offsets)
            .map(|(part, dst_offset)| CopyRegion {
                dst: &buf,
                dst_offset,
                src: part.buf,
                src_offset: part.offset,
                bytes: part.bytes,
            })
            .collect();
        if let Err(error) = copy_regions(gpu, &regions) {
            free_device(gpu, buf);
            return Err(format!(
                "session cache: resident snapshot copy failed: {error}"
            ));
        }
        *pending = Some(Resident {
            key,
            buf,
            fixed_bytes,
            rows: live.rows.iter().map(|stream| stream.rows).collect(),
            meta: meta.to_vec(),
            cold_exact,
        });
        Ok(())
    }

    /// Publish this turn's snapshots, parents before children. Uncommitted
    /// snapshots are dropped by the next [`Self::begin`], so a failed turn
    /// never publishes state.
    pub fn commit(&mut self) {
        let mut refused = HashSet::new();
        for (key, snapshot) in std::mem::take(&mut self.pending) {
            if snapshot
                .parent
                .as_ref()
                .is_some_and(|parent| refused.contains(parent))
            {
                refused.insert(key);
                self.retire(snapshot);
                continue;
            }
            let (domain, p, fp) = key.clone();
            // `at_boundary` already reserved room for every pending
            // snapshot, so this insert evicts nothing.
            let (id, displaced) = self.pool.insert_unaligned(domain, p, fp, snapshot);
            if id == CheckpointId::NONE {
                refused.insert(key);
            } else if self.children.contains_key(&key) {
                self.pool.pin(&key.0, key.1, key.2);
            }
            for snapshot in displaced {
                self.retire(snapshot);
            }
        }
        if let Some(resident) = self.resident_pending.take() {
            if let Some(displaced) = self.resident.replace(resident) {
                self.release_bufs.push(displaced.buf);
            }
        }
        if let Some(tail) = self.tail_pending.take() {
            if let Some(displaced) = self.tail.replace(tail) {
                self.release_bufs.push(displaced.buf);
            }
        } else if self.tail.as_ref().is_some_and(|tail| {
            self.turn.as_ref().is_some_and(|turn| turn.start < tail.key.1 as usize)
        }) {
            self.release_bufs.push(self.tail.take().unwrap().buf);
        }
        self.turn = None;
    }

    /// Bytes of every published snapshot.
    pub fn stored_bytes(&self) -> u64 {
        self.pool.total_bytes()
    }

    /// Free every snapshot; with a disk tier attached, published snapshots
    /// are demoted to disk instead of dropped.
    pub fn clear(&mut self, gpu: &mut Gpu) {
        if self.disk.is_some() {
            for snapshot in std::mem::take(&mut self.release) {
                free_buffer(gpu, snapshot);
            }
            for (_, snapshot) in std::mem::take(&mut self.pending) {
                self.drop_snapshot(gpu, snapshot);
            }
            // Leaves first: each demotion unpins its parent.
            while let Some((key, snapshot)) = self.pool.pop_lru() {
                self.evict_device(gpu, key, snapshot, 0);
            }
        }
        let pooled = self.pool.drain_blobs();
        let pending = std::mem::take(&mut self.pending)
            .into_iter()
            .map(|entry| entry.1);
        for snapshot in pooled
            .into_iter()
            .chain(pending)
            .chain(std::mem::take(&mut self.release))
        {
            free_buffer(gpu, snapshot);
        }
        for resident in [
            self.resident.take(),
            self.resident_pending.take(),
            self.tail.take(),
            self.tail_pending.take(),
        ]
        .into_iter()
        .flatten()
        {
            free_device(gpu, resident.buf);
        }
        for buf in std::mem::take(&mut self.release_bufs) {
            free_device(gpu, buf);
        }
        self.children.clear();
        self.turn = None;
    }

    /// Make room for `need` more device bytes the live state is about to
    /// map (context growth): free snapshots, displaced ones first, then the
    /// least recently used published leaves, then this turn's pending
    /// captures deepest first, until the memory guard passes or nothing is
    /// left. Snapshots are opportunistic and never hold memory the context
    /// needs. Returns the bytes freed.
    pub fn release_for(&mut self, gpu: &mut Gpu, need: u64) -> u64 {
        let mut freed = 0;
        loop {
            // A failed query cannot be helped by evicting.
            if memory_fits(gpu, need, 0) != Some(false) {
                break;
            }
            if let Some(buf) = self.release_bufs.pop() {
                freed += buf.size() as u64;
                free_device(gpu, buf);
                continue;
            }
            let snapshot = match self.release.pop() {
                Some(snapshot) => {
                    freed += snapshot.bytes;
                    free_buffer(gpu, snapshot);
                    continue;
                }
                None => match self.pool.pop_lru() {
                    Some((key, snapshot)) => {
                        freed += snapshot.bytes;
                        self.evict_device(gpu, key, snapshot, need);
                        continue;
                    }
                    None => self.pending.pop().map(|entry| entry.1),
                },
            };
            let Some(snapshot) = snapshot else {
                break;
            };
            freed += snapshot.bytes;
            self.drop_snapshot(gpu, snapshot);
        }
        if freed > 0 {
            eprintln!(
                "  session cache: released {} MiB for context growth",
                freed >> 20
            );
        }
        freed
    }
}

/// Ascending snapshot boundaries in `(after, up_to]`: multiples of `stride`,
/// plus the position after every `turn_end` token of `prompt` (message ends)
/// when it is `Some`.
pub fn stride_boundaries(
    stride: usize,
    prompt: &[u32],
    turn_end: Option<u32>,
    after: usize,
    up_to: usize,
) -> Vec<usize> {
    let mut boundaries: Vec<usize> = (after / stride + 1..=up_to / stride)
        .map(|k| k * stride)
        .collect();
    if let Some(turn_end) = turn_end {
        let end = up_to.min(prompt.len());
        boundaries.extend(
            (after.min(end)..end)
                .filter(|&i| prompt[i] == turn_end)
                .map(|i| i + 1),
        );
        boundaries.sort_unstable();
        boundaries.dedup();
    }
    boundaries
}

/// Host record of the committed live state: the consumed history it holds and
/// the route that produced it. A live continuation is *session-exact* (its
/// state descends from decode, as in a continuous conversation), not
/// cold-exact, so it is only resumed in place and never captured into the
/// cross-session cache, except by boundaries the architecture scopes apart.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LiveRecord {
    tokens: Vec<u32>,
    route: SessionRoute,
}

/// `Some(L)` when the live state in `record` can serve `prompt` on `route`:
/// it holds exactly `L = record.tokens.len()` tokens (`0 < L < prompt.len()`,
/// so at least the last prompt token is computed), those tokens are
/// `prompt[..L]`, and every owner sits at `L`.
fn live_start(
    record: Option<&LiveRecord>,
    prompt: &[u32],
    route: SessionRoute,
    live_position: Option<usize>,
) -> Option<usize> {
    let record = record?;
    let end = record.tokens.len();
    (record.route == route
        && end > 0
        && end < prompt.len()
        && prompt[..end] == record.tokens[..]
        && live_position == Some(end))
    .then_some(end)
}

/// Whether the state left by the turn that began with `turn_prompt` is the
/// live state after `consumed`, the full host history the client committed:
/// `consumed` extends the turn's prompt and every owner sits at its end. An
/// empty `consumed` (unrepaired terminal) is never live.
fn live_after_commit(turn_prompt: &[u32], consumed: &[u32], live_position: Option<usize>) -> bool {
    let prompt_len = turn_prompt.len();
    prompt_len > 0
        && consumed.len() >= prompt_len
        && consumed[..prompt_len] == *turn_prompt
        && live_position == Some(consumed.len())
}

/// Prompt tokens the next prefill skips: the live state when it reaches at
/// least as far as the longest cached snapshot (a tie prefers live: no copy),
/// else that snapshot.
fn plan_start(snapshot: usize, live: Option<usize>) -> usize {
    match live {
        Some(live) if live >= snapshot => live,
        _ => snapshot,
    }
}

/// The session machinery an architecture's bundle holds: the shared snapshot
/// cache plus the committed live state of the current conversation.
pub struct SessionDriver {
    cache: SessionCache,
    /// Committed live state kept for a continuation of the same conversation;
    /// consumed while a request is in flight.
    live: Option<LiveRecord>,
    /// Prompt and route of the request in flight.
    turn: Option<(Vec<u32>, SessionRoute)>,
}

impl SessionDriver {
    pub fn new(cache: SessionCache) -> Self {
        Self {
            cache,
            live: None,
            turn: None,
        }
    }

    pub fn cache(&self) -> &SessionCache {
        &self.cache
    }

    /// [`SessionCache::release_for`].
    pub fn release_for(&mut self, gpu: &mut Gpu, need: u64) -> u64 {
        self.cache.release_for(gpu, need)
    }

    fn live_hit(
        &self,
        state: &dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
    ) -> Option<usize> {
        live_start(
            self.live.as_ref(),
            prompt,
            route,
            state.live_position(route),
        )
    }

    /// Prompt tokens the next prefill of `prompt` on `route` can skip: the
    /// longer of the live state and a cached snapshot; 0 on a miss.
    pub fn plan(&self, state: &dyn SessionState, prompt: &[u32], route: SessionRoute) -> usize {
        self.plan_capped(state, prompt, route, prompt.len())
    }

    /// [`Self::plan`] restricted to reuse points `<= cap` (the plan over the
    /// first `cap + 1` prompt tokens). With `cap = prompt.len()` it is the
    /// plain plan.
    pub fn plan_capped(
        &self,
        state: &dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        cap: usize,
    ) -> usize {
        let head = &prompt[..prompt.len().min(cap.saturating_add(1))];
        plan_start(
            self.cache.plan(state, head, route),
            self.live_hit(state, prompt, route)
                .filter(|&live| live <= cap),
        )
    }

    /// Start a prefill of `prompt` on `route` that skips the `reused` tokens
    /// [`Self::plan`] returned:
    /// - live: `reused` is the end of the committed live state; every owner
    ///   is kept, nothing is restored, and boundaries above `reused` are
    ///   captured only with `capture_live`;
    /// - snapshot: restore the `reused`-token snapshot;
    /// - `reused == 0`: cold start.
    ///
    /// Returns whether the prefill continues the live state.
    pub fn begin(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
        capture_live: bool,
    ) -> Result<bool, String> {
        let live = reused > 0 && self.live_hit(state, prompt, route) == Some(reused);
        self.live = None;
        self.turn = None;
        if live {
            self.cache
                .begin_live(gpu, state, prompt, route, reused, capture_live);
        } else {
            self.cache.begin(gpu, state, prompt, route, reused)?;
        }
        self.turn = Some((prompt.to_vec(), route));
        Ok(live)
    }

    /// The next position the running prefill must stop at and report.
    pub fn next_boundary(&self) -> Option<usize> {
        self.cache.next_boundary()
    }

    /// The first still-pending boundary strictly after `position`.
    pub fn next_boundary_after(&self, position: usize) -> Option<usize> {
        self.cache.next_boundary_after(position)
    }

    /// The lineage split row of the request in flight, if armed.
    pub fn turn_lineage(&self) -> Option<usize> {
        self.cache.turn_lineage()
    }

    /// Boundaries the running prefill has yet to report, ascending (the
    /// whole schedule right after [`Self::begin`]); empty when none is armed.
    pub fn turn_boundaries(&self) -> Vec<usize> {
        self.cache.pending_boundaries()
    }

    /// Replace the boundary schedule and lineage split [`Self::begin`] armed
    /// with one computed once elsewhere (a mesh root's canonical schedule that
    /// every band consumes). Call it right after `begin`. `boundaries` must be
    /// strictly increasing, above the turn's start and at most its end;
    /// `lineage` must be an interior boundary among them. Nothing changes on
    /// error. A driver whose turn armed no schedule (a live continuation that
    /// does not capture) accepts only the empty schedule.
    pub fn set_boundaries(
        &mut self,
        boundaries: Vec<usize>,
        lineage: Option<usize>,
    ) -> Result<(), String> {
        self.cache.override_boundaries(boundaries, lineage)
    }

    /// Start a prefill that restores the `reused`-token snapshot (or cold
    /// starts for 0) and never continues the live state, whatever
    /// [`Self::plan`] would say: [`Self::begin`] takes the live path when the
    /// live position equals `reused`, but that state descends from decode and
    /// is not the canonical snapshot. Clears the live state and any turn, then
    /// runs [`SessionCache::begin`]; the turn is armed only on success.
    pub fn begin_snapshot(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
    ) -> Result<(), String> {
        self.live = None;
        self.turn = None;
        self.cache.begin(gpu, state, prompt, route, reused)?;
        self.turn = Some((prompt.to_vec(), route));
        Ok(())
    }

    /// Abandon the request in flight after the owners were drained: forget
    /// the live state, retire every resident (their rows no longer back
    /// anything) and free the displaced and uncommitted pending snapshots, so
    /// a reset never leaves a pending capture. Published pool snapshots stay.
    pub fn abort(&mut self, gpu: &mut Gpu) {
        self.live = None;
        self.turn = None;
        self.cache.live_state_reset();
        self.cache.start_turn(gpu);
    }

    /// Whether `need` more device bytes would break the memory guard now
    /// (a failed free-memory query does not count): the caller should
    /// release snapshots, see [`Self::release_for`].
    pub fn needs_release(&self, gpu: &Gpu, need: u64) -> bool {
        memory_fits(gpu, need, 0) == Some(false)
    }

    /// Stamp the cache domain's `model_load_epoch` with `epoch`, so keys of
    /// this driver never alias those of another load. Host-only; call it once
    /// at initial attach on a fresh driver: later calls would orphan existing
    /// snapshots (including disk-tier entries, which are indexed by domain).
    ///
    /// # Errors
    /// When the driver is not fresh: it holds snapshots, pending captures,
    /// residents, a live state, a turn in flight or a disk tier. Nothing
    /// changes then.
    pub fn bind_load_epoch(&mut self, epoch: u64) -> Result<(), String> {
        let cache = &self.cache;
        let fresh = cache.pool.is_empty()
            && cache.pending.is_empty()
            && cache.release.is_empty()
            && cache.release_bufs.is_empty()
            && cache.children.is_empty()
            && cache.resident.is_none()
            && cache.resident_pending.is_none()
            && cache.tail.is_none()
            && cache.tail_pending.is_none()
            && cache.turn.is_none()
            && cache.disk.is_none()
            && self.live.is_none()
            && self.turn.is_none();
        if !fresh {
            return Err(
                "session cache: load epoch can only be bound on a fresh driver".to_string(),
            );
        }
        self.cache.domain.model_load_epoch = epoch;
        Ok(())
    }

    /// The pool key of the snapshot of `prefix` on `route` under `state`'s
    /// scope, in exactly this cache's domain. `None` when `prefix` is empty or
    /// the route has no snapshot scope.
    pub fn piece_key(
        &self,
        state: &dyn SessionState,
        prefix: &[u32],
        route: SessionRoute,
    ) -> Option<SessionPieceKey> {
        if prefix.is_empty() {
            return None;
        }
        let scope = state.snapshot_scope(route)?;
        Some(SessionPieceKey((
            self.cache.domain.scoped(&scope),
            prefix.len() as u64,
            prefix_fingerprint(prefix),
        )))
    }

    /// Whether the piece is a published device-pool snapshot. Pending,
    /// resident-only and disk-only snapshots are not pieces.
    pub fn piece_resident(&self, key: &SessionPieceKey) -> bool {
        self.cache.pool.contains(&key.0 .0, key.0 .1, key.0 .2)
    }

    /// Pin the piece against eviction (one pin per call, counted with the
    /// children refcount). Refuses, changing nothing, unless the piece is
    /// resident in the pool.
    pub fn pin_piece(&mut self, key: &SessionPieceKey) -> bool {
        if !self.piece_resident(key) {
            return false;
        }
        self.cache.link(&key.0);
        true
    }

    /// Release one pin taken by [`Self::pin_piece`]; the last release makes
    /// the piece evictable again.
    pub fn unpin_piece(&mut self, key: &SessionPieceKey) {
        self.cache.unlink(&key.0);
    }

    /// The route of the request in flight.
    pub fn turn_route(&self) -> Option<SessionRoute> {
        self.turn.as_ref().map(|(_, route)| *route)
    }

    /// Report that every owner consumed exactly `prefix` (the boundary).
    pub fn at_boundary(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prefix: &[u32],
    ) -> Result<(), String> {
        self.cache.at_boundary(gpu, state, prefix)
    }

    /// [`Self::at_boundary`] at `position` of the prompt in flight, without
    /// copying its tokens.
    pub fn at_turn_boundary(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        position: usize,
    ) -> Result<(), String> {
        let (prompt, _) = self
            .turn
            .as_ref()
            .ok_or("session boundary has no prompt in flight")?;
        let prefix = prompt
            .get(..position)
            .ok_or("session boundary exceeds the prompt in flight")?;
        self.cache.at_boundary(gpu, state, prefix)
    }

    /// Publish this turn's snapshots and drop the live state.
    pub fn commit(&mut self) {
        self.cache.commit();
        self.live = None;
        self.turn = None;
    }

    /// Publish this turn's snapshots and keep the live state when `consumed`
    /// (the host history the client committed) extends this turn's prompt and
    /// every owner sits exactly at its end. The next turn of the conversation
    /// then prefills only the suffix in place.
    pub fn commit_live(&mut self, state: &dyn SessionState, consumed: &[u32]) {
        self.cache.commit();
        self.live = self.turn.take().and_then(|(prompt, route)| {
            live_after_commit(&prompt, consumed, state.live_position(route)).then(|| LiveRecord {
                tokens: consumed.to_vec(),
                route,
            })
        });
    }

    /// Forget the live state (the owners were reset outside the cache, so
    /// their rows no longer back the resident snapshot either).
    pub fn forget_live(&mut self) {
        self.live = None;
        self.turn = None;
        self.cache.live_state_reset();
    }

    /// Drop every snapshot, pending ones included, and the live state.
    pub fn clear(&mut self, gpu: &mut Gpu) {
        self.live = None;
        self.turn = None;
        self.cache.clear(gpu);
    }
}

/// Most published [`MeshSessionRecord`]s a [`MeshSessionCache`] holds.
pub const MESH_SESSION_MAX_RECORDS: usize = 64;

/// Pin every driver's piece, all or nothing. `pieces[i]` belongs to
/// `drivers[i]`. Residency of all pieces is probed first; if any pin then
/// fails, the pins this call took are released. Returns whether all are
/// pinned.
pub fn probe_and_pin_pieces(
    drivers: &mut [&mut SessionDriver],
    pieces: &[SessionPieceKey],
) -> bool {
    if drivers.is_empty()
        || drivers.len() != pieces.len()
        || !drivers
            .iter()
            .zip(pieces)
            .all(|(driver, piece)| driver.piece_resident(piece))
    {
        return false;
    }
    for i in 0..pieces.len() {
        if !drivers[i].pin_piece(&pieces[i]) {
            for j in 0..i {
                drivers[j].unpin_piece(&pieces[j]);
            }
            return false;
        }
    }
    true
}

/// Release one pin per piece taken by [`probe_and_pin_pieces`].
pub fn unpin_pieces(drivers: &mut [&mut SessionDriver], pieces: &[SessionPieceKey]) {
    for (driver, piece) in drivers.iter_mut().zip(pieces) {
        driver.unpin_piece(piece);
    }
}

/// One mesh session: each band's piece of the one prefix every band
/// captured. The prefix length and fingerprint live in the piece keys (all
/// pieces cover the same prefix), so a record copies no tokens and holds no
/// device buffer.
#[derive(Clone, Debug)]
pub struct MeshSessionRecord {
    pub route: SessionRoute,
    /// In owner order: `pieces[i]` is band `i`'s snapshot of the prefix.
    pub pieces: Vec<SessionPieceKey>,
}

/// Aggregate index over the per-band caches of one mesh. A published record
/// pins every band's piece for its whole lifetime, so a piece never leaves a
/// band's pool while a record references it; a record whose piece is gone
/// anyway (a driver cleared outside the mesh) is dropped at the next
/// [`Self::plan`].
///
/// Only the record index is bounded, to [`MESH_SESSION_MAX_RECORDS`]; the
/// pieces stay opportunistic pool snapshots allocated like the single-device
/// cache's, so pins can make a full band pool refuse new captures
/// ([`Self::evict_oldest`] frees pins under memory pressure). Every method
/// taking `drivers` expects the same bands in the same order for the whole
/// life of the cache.
#[derive(Default)]
pub struct MeshSessionCache {
    staged: Vec<MeshSessionRecord>,
    records: VecDeque<MeshSessionRecord>,
    /// The record the last [`Self::plan`] chose, by route and root piece.
    planned: Option<(SessionRoute, SessionPieceKey)>,
}

impl MeshSessionCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Published records, oldest first.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Candidates awaiting [`Self::publish`].
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Stage a candidate captured at one boundary of the request in flight.
    /// It holds no pin and is invisible to [`Self::plan`] until published.
    pub fn stage(&mut self, record: MeshSessionRecord) {
        self.staged.push(record);
    }

    /// Drop every staged candidate (the request aborted, was cancelled or
    /// failed). Published records are untouched.
    pub fn abort(&mut self) {
        self.staged.clear();
    }

    /// Commit every driver's pending snapshots (without touching its live
    /// state or turn), then publish each staged candidate whose pieces cover
    /// one non-empty prefix and are all resident and all pinned; any other
    /// candidate is dropped and leaves no pin. A published record replaces an
    /// older one of the same route and prefix, and the oldest records are
    /// released beyond [`MESH_SESSION_MAX_RECORDS`]. Returns the number
    /// published.
    pub fn publish(&mut self, drivers: &mut [&mut SessionDriver]) -> usize {
        for driver in drivers.iter_mut() {
            driver.cache.commit();
        }
        let mut published = 0;
        for record in std::mem::take(&mut self.staged) {
            let Some(root) = record.pieces.first() else {
                continue;
            };
            let uniform =
                root.prefix_len() > 0 && record.pieces.iter().all(|piece| piece.same_prefix(root));
            if !uniform || !probe_and_pin_pieces(drivers, &record.pieces) {
                continue;
            }
            let mut kept = VecDeque::with_capacity(self.records.len() + 1);
            for old in std::mem::take(&mut self.records) {
                if old.route == record.route
                    && old
                        .pieces
                        .first()
                        .is_some_and(|piece| piece.same_prefix(root))
                {
                    unpin_pieces(drivers, &old.pieces);
                } else {
                    kept.push_back(old);
                }
            }
            self.records = kept;
            while self.records.len() >= MESH_SESSION_MAX_RECORDS {
                if let Some(old) = self.records.pop_front() {
                    unpin_pieces(drivers, &old.pieces);
                }
            }
            self.records.push_back(record);
            published += 1;
        }
        published
    }

    /// Longest published prefix of `prompt` on `route` that is strictly
    /// shorter than the prompt (so prefill computes at least the last
    /// token); 0 on a miss. Records with a piece no longer resident are
    /// dropped and their pins released. The winning record keeps its pins
    /// (they stay valid for the restore) and is returned by
    /// [`Self::planned`] until it is released or replaced by the next plan.
    pub fn plan(
        &mut self,
        drivers: &mut [&mut SessionDriver],
        prompt: &[u32],
        route: SessionRoute,
    ) -> usize {
        self.planned = None;
        let mut best = 0;
        let mut best_key = None;
        // Fingerprints of `prompt` prefixes already hashed, by length.
        let mut hashed: Vec<(usize, u64)> = Vec::new();
        let mut kept = VecDeque::with_capacity(self.records.len());
        for record in std::mem::take(&mut self.records) {
            let root = record.pieces.first().filter(|_| {
                record.pieces.len() == drivers.len()
                    && drivers
                        .iter()
                        .zip(&record.pieces)
                        .all(|(driver, piece)| driver.piece_resident(piece))
            });
            let Some(root) = root else {
                unpin_pieces(drivers, &record.pieces);
                continue;
            };
            let len = root.prefix_len();
            if record.route == route && len > best && len < prompt.len() {
                let fingerprint = match hashed.iter().find(|(hashed, _)| *hashed == len) {
                    Some(&(_, fingerprint)) => fingerprint,
                    None => {
                        let fingerprint = prefix_fingerprint(&prompt[..len]);
                        hashed.push((len, fingerprint));
                        fingerprint
                    }
                };
                if root.0 .2 == fingerprint {
                    best = len;
                    best_key = Some((record.route, root.clone()));
                }
            }
            kept.push_back(record);
        }
        self.records = kept;
        self.planned = best_key;
        best
    }

    /// The record the last [`Self::plan`] chose, with its prefix length, while
    /// it is still published (its pins are live); `None` after a miss or
    /// once it was released.
    pub fn planned(&self) -> Option<(usize, &MeshSessionRecord)> {
        let (route, key) = self.planned.as_ref()?;
        self.records
            .iter()
            .find(|record| record.route == *route && record.pieces.first() == Some(key))
            .map(|record| (key.prefix_len(), record))
    }

    /// Drop the planned record and release its pins (the restore was
    /// abandoned or failed). Returns whether a record was released.
    pub fn release_plan(&mut self, drivers: &mut [&mut SessionDriver]) -> bool {
        let Some((route, key)) = self.planned.take() else {
            return false;
        };
        let Some(at) = self
            .records
            .iter()
            .position(|record| record.route == route && record.pieces.first() == Some(&key))
        else {
            return false;
        };
        if let Some(record) = self.records.remove(at) {
            unpin_pieces(drivers, &record.pieces);
        }
        true
    }

    /// Release every band's pins of the oldest published record (before any
    /// band frees snapshots under memory pressure). Returns whether a record
    /// was released.
    pub fn evict_oldest(&mut self, drivers: &mut [&mut SessionDriver]) -> bool {
        let Some(old) = self.records.pop_front() else {
            return false;
        };
        unpin_pieces(drivers, &old.pieces);
        true
    }

    /// Release every record's pins and drop the staged candidates. Call it
    /// before clearing the drivers.
    pub fn clear(&mut self, drivers: &mut [&mut SessionDriver]) {
        self.staged.clear();
        self.planned = None;
        for record in std::mem::take(&mut self.records) {
            unpin_pieces(drivers, &record.pieces);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn uma_headroom_scales_with_host_ram_inside_the_clamp() {
        // MemTotal / 16, clamped to [2, 8] GiB.
        assert_eq!(uma_headroom(Some(32 * GIB)), 2 * GIB);
        assert_eq!(uma_headroom(Some(64 * GIB)), 4 * GIB);
        assert_eq!(uma_headroom(Some(128 * GIB)), 8 * GIB);
        // Below 32 GiB the floor holds; above 128 GiB the cap holds.
        assert_eq!(uma_headroom(Some(8 * GIB)), 2 * GIB);
        assert_eq!(uma_headroom(Some(0)), 2 * GIB);
        assert_eq!(uma_headroom(Some(256 * GIB)), 8 * GIB);
        assert_eq!(uma_headroom(Some(u64::MAX)), 8 * GIB);
    }

    #[test]
    fn unreadable_host_total_takes_the_conservative_cap() {
        assert_eq!(uma_headroom(None), UMA_HEADROOM_MAX);
        assert_eq!(UMA_HEADROOM_MAX, 8 * GIB);
    }

    #[test]
    fn discrete_headroom_ignores_host_ram() {
        for total in [None, Some(0), Some(32 * GIB), Some(128 * GIB)] {
            assert_eq!(headroom_for(false, total), VRAM_HEADROOM);
        }
        assert_eq!(VRAM_HEADROOM, 256 << 20);
        assert_eq!(headroom_for(true, Some(64 * GIB)), 4 * GIB);
        assert_eq!(headroom_for(true, None), 8 * GIB);
    }

    #[test]
    fn fits_decides_exactly_at_need_growth_and_headroom() {
        let (need, growth, headroom) = (3 * GIB, 5 * GIB, 2 * GIB);
        let free = need + growth + headroom;
        assert!(fits_with_headroom(free, need, growth, headroom));
        assert!(!fits_with_headroom(free - 1, need, growth, headroom));
        assert!(fits_with_headroom(free + 1, need, growth, headroom));
    }

    #[test]
    fn fits_counts_each_term() {
        let free = 10 * GIB;
        // Each term alone leaves room; together they do not.
        assert!(fits_with_headroom(free, 4 * GIB, 0, 2 * GIB));
        assert!(fits_with_headroom(free, 0, 4 * GIB, 2 * GIB));
        assert!(fits_with_headroom(free, 4 * GIB, 4 * GIB, 2 * GIB));
        assert!(!fits_with_headroom(free, 4 * GIB, 4 * GIB + 1, 2 * GIB));
        // Growth reserve alone can refuse an otherwise small snapshot.
        assert!(!fits_with_headroom(free, 1, 8 * GIB, 2 * GIB));
        // Headroom alone refuses even a zero-byte need.
        assert!(!fits_with_headroom(GIB, 0, 0, 2 * GIB));
        assert!(fits_with_headroom(2 * GIB, 0, 0, 2 * GIB));
    }

    #[test]
    fn fits_refuses_instead_of_overflowing() {
        assert!(!fits_with_headroom(u64::MAX, u64::MAX, 1, 0));
        assert!(!fits_with_headroom(u64::MAX, 1, u64::MAX, 0));
        assert!(!fits_with_headroom(u64::MAX, 0, 1, u64::MAX));
        assert!(!fits_with_headroom(u64::MAX, u64::MAX, u64::MAX, u64::MAX));
        assert!(!fits_with_headroom(0, u64::MAX, u64::MAX, 8 * GIB));
        // The largest representable sum still fits exactly.
        assert!(fits_with_headroom(u64::MAX, u64::MAX - 3, 2, 1));
        assert!(!fits_with_headroom(u64::MAX - 1, u64::MAX - 3, 2, 1));
    }

    #[test]
    fn staged_need_sums_without_overflow() {
        assert_eq!(saturating_total([1, 2, 3]), 6);
        assert_eq!(saturating_total([u64::MAX, 1]), u64::MAX);
        assert_eq!(saturating_total([]), 0);
    }

    #[test]
    fn boundary_lookahead_does_not_consume_the_capture_cursor() {
        let mut cache = SessionCache::new(domain(), 8 << 30);
        cache.turn = Some(Turn {
            domain: domain(),
            route: SessionRoute::Mtp,
            boundaries: VecDeque::from([8192, 10000, 16384]),
            end: 16384,
            lineage: None,
            pool_capture: true,
            start: 0,
        });
        assert_eq!(cache.next_boundary(), Some(8192));
        assert_eq!(cache.next_boundary_after(8192), Some(10000));
        assert_eq!(cache.next_boundary_after(10000), Some(16384));
        assert_eq!(cache.next_boundary_after(16384), None);
        assert_eq!(cache.next_boundary(), Some(8192));
    }

    #[test]
    fn driver_boundary_lookahead_forwards_without_consuming() {
        let mut driver = SessionDriver::new(SessionCache::new(domain(), 8 << 30));
        assert_eq!(driver.next_boundary(), None);
        assert_eq!(driver.next_boundary_after(0), None);
        driver.cache.turn = Some(Turn {
            domain: domain(),
            route: SessionRoute::Mtp,
            boundaries: VecDeque::from([8192, 10000, 16384]),
            end: 16384,
            lineage: None,
            pool_capture: true,
            start: 0,
        });
        for (position, expected) in [
            (0, Some(8192)),
            (8191, Some(8192)),
            (8192, Some(10000)),
            (10000, Some(16384)),
            (16384, None),
            (usize::MAX, None),
        ] {
            assert_eq!(driver.next_boundary_after(position), expected);
            assert_eq!(driver.next_boundary(), Some(8192));
        }
        assert_eq!(
            driver.cache.turn.as_ref().unwrap().boundaries,
            VecDeque::from([8192, 10000, 16384])
        );
        driver.commit();
        assert_eq!(driver.next_boundary_after(0), None);
    }

    #[test]
    fn driver_turn_route_tracks_the_in_flight_request_only() {
        let mut driver = SessionDriver::new(SessionCache::new(domain(), 0));
        assert_eq!(driver.turn_route(), None);
        for route in [SessionRoute::Ar, SessionRoute::Mtp] {
            driver.live = Some(record(&[1, 2], route));
            assert_eq!(driver.turn_route(), None);
            driver.turn = Some((vec![1, 2, 3], route));
            assert_eq!(driver.turn_route(), Some(route));
            driver.commit();
            assert_eq!(driver.turn_route(), None);
            assert!(driver.live.is_none());
            driver.turn = Some((vec![1, 2, 3], route));
            driver.live = Some(record(&[1, 2], route));
            driver.forget_live();
            assert_eq!(driver.turn_route(), None);
            assert!(driver.live.is_none());
        }
    }
    use crate::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };
    use rdna_compute::{DType, GpuTensor};

    const FIXED_BYTES: usize = 256;
    const ROW_BYTES: usize = 16;
    const ROW_CAPACITY: usize = 1024;
    const STRIDE: usize = 128;
    /// A token after which the toy also splits, like a message end.
    const MARKER: u32 = 999_999_999;

    /// A fixed part that depends on the whole prefix and one append-only row
    /// per token that depends only on the tokens up to it.
    struct Toy {
        fixed: GpuTensor,
        rows: GpuTensor,
        position: usize,
        tail_snapshot: bool,
    }

    impl Toy {
        fn layout(&self, rows: usize) -> StateLayout<'_> {
            StateLayout {
                fixed: vec![StatePart {
                    buf: &self.fixed.buf,
                    offset: 0,
                    bytes: FIXED_BYTES,
                }],
                rows: vec![RowStream {
                    buf: &self.rows.buf,
                    row_bytes: ROW_BYTES,
                    rows,
                }],
            }
        }

        /// Stand-in for a prefill chunk.
        fn advance(&mut self, gpu: &mut Gpu, prefix: &[u32]) {
            self.position = prefix.len();
            gpu.hip
                .memcpy_htod(&self.fixed.buf, &fixed_of(prefix))
                .unwrap();
            gpu.hip
                .memcpy_htod(&self.rows.buf, &rows_of(prefix))
                .unwrap();
        }

        /// Fixed bytes and the valid rows.
        fn bytes(&self, gpu: &mut Gpu) -> (Vec<u8>, Vec<u8>) {
            gpu.hip.device_synchronize().unwrap();
            let mut fixed = vec![0; FIXED_BYTES];
            gpu.hip.memcpy_dtoh(&mut fixed, &self.fixed.buf).unwrap();
            let mut rows = vec![0; ROW_CAPACITY * ROW_BYTES];
            gpu.hip.memcpy_dtoh(&mut rows, &self.rows.buf).unwrap();
            rows.truncate(self.position * ROW_BYTES);
            (fixed, rows)
        }
    }

    impl SessionState for Toy {
        fn snapshot_scope(&self, _route: SessionRoute) -> Option<String> {
            Some("toy".to_string())
        }
        fn snapshot_boundaries(&self, prompt: &[u32], after: usize, up_to: usize) -> Vec<usize> {
            let mut boundaries: Vec<usize> = (after / STRIDE + 1..=up_to / STRIDE)
                .map(|k| k * STRIDE)
                .collect();
            if !self.tail_snapshot {
                boundaries.extend(
                    (after..up_to)
                        .filter(|&i| prompt[i] == MARKER)
                        .map(|i| i + 1),
                );
            }
            boundaries.sort_unstable();
            boundaries.dedup();
            boundaries
        }
        fn lineage_boundary(&self, prompt: &[u32], after: usize, up_to: usize) -> Option<usize> {
            if !self.tail_snapshot {
                return None;
            }
            let p = prompt[..up_to].iter().rposition(|&id| id == MARKER)?;
            (p > after && p % STRIDE != 0 && p / STRIDE == up_to / STRIDE).then_some(p)
        }
        fn snapshot_parts(
            &mut self,
            _route: SessionRoute,
            position: usize,
        ) -> Result<SnapshotParts<'_>, String> {
            assert_eq!(self.position, position);
            Ok(SnapshotParts {
                meta: (position as u64).to_le_bytes().to_vec(),
                layout: self.layout(position),
            })
        }
        fn restore_parts(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            meta: &[u8],
        ) -> Result<StateLayout<'_>, String> {
            Ok(self.layout(u64::from_le_bytes(meta.try_into().unwrap()) as usize))
        }
        fn finish_restore(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            meta: &[u8],
        ) -> Result<(), String> {
            self.position = u64::from_le_bytes(meta.try_into().unwrap()) as usize;
            Ok(())
        }
        fn live_position(&self, _route: SessionRoute) -> Option<usize> {
            Some(self.position)
        }
        fn growth_reserve_bytes(&self, _gpu: &Gpu, _through: usize) -> u64 {
            0
        }
        fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
            self.position = 0;
            gpu.hip
                .memset(&self.fixed.buf, 0, FIXED_BYTES)
                .map_err(|e| e.to_string())?;
            gpu.hip
                .memset(&self.rows.buf, 0, ROW_CAPACITY * ROW_BYTES)
                .map_err(|e| e.to_string())
        }
    }

    fn fixed_of(prefix: &[u32]) -> Vec<u8> {
        let seed = prefix_fingerprint(prefix).to_le_bytes();
        (0..FIXED_BYTES).map(|i| seed[i % 8] ^ i as u8).collect()
    }

    fn rows_of(prefix: &[u32]) -> Vec<u8> {
        (1..=prefix.len())
            .flat_map(|end| {
                let seed = prefix_fingerprint(&prefix[..end]).to_le_bytes();
                (0..ROW_BYTES).map(move |i| seed[i % 8] ^ i as u8)
            })
            .collect()
    }

    fn domain() -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![1],
            model_load_epoch: 1,
            sidecar_digests: vec![],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![2],
                config_digest: vec![3],
            },
            template: TemplateIdentity {
                template_digest: vec![4],
                normalization_tag: "jinja".into(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "toy".into(),
                state_abi_tag: String::new(),
                position_attention_tag: String::new(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![],
                v_stride_bytes: vec![],
                layout_tag: String::new(),
            },
            device: DeviceTopology {
                device_id: "gpu-0".into(),
                topology_id: "single".into(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "default".into(),
            },
        }
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 100_000 + i).collect()
    }

    /// One turn: restore what `plan` offers, prefill to the end capturing at
    /// every boundary, optionally commit. Returns the planned reuse.
    fn turn(
        cache: &mut SessionCache,
        gpu: &mut Gpu,
        toy: &mut Toy,
        prompt: &[u32],
        commit: bool,
    ) -> usize {
        let reused = cache.plan(toy, prompt, SessionRoute::Ar);
        cache
            .begin(gpu, toy, prompt, SessionRoute::Ar, reused)
            .unwrap();
        while let Some(b) = cache.next_boundary() {
            toy.advance(gpu, &prompt[..b]);
            cache.at_boundary(gpu, toy, &prompt[..b]).unwrap();
        }
        toy.advance(gpu, prompt);
        if commit {
            cache.commit();
        }
        reused
    }

    /// Restore `prompt`'s planned snapshot and check it equals the state a
    /// cold prefill of that prefix leaves.
    fn assert_restores(
        cache: &mut SessionCache,
        gpu: &mut Gpu,
        toy: &mut Toy,
        prompt: &[u32],
        expected: usize,
    ) {
        let reused = cache.plan(toy, prompt, SessionRoute::Ar);
        assert_eq!(reused, expected);
        cache
            .begin(gpu, toy, prompt, SessionRoute::Ar, reused)
            .unwrap();
        assert_eq!(toy.position, reused);
        assert_eq!(
            toy.bytes(gpu),
            (fixed_of(&prompt[..reused]), rows_of(&prompt[..reused]))
        );
        cache.commit();
    }

    #[test]
    fn tail_reuses_lcp_preserves_chunk_and_never_publishes_lineage_to_pool() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let mut toy = Toy {
            fixed: gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap(),
            rows: gpu
                .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
                .unwrap(),
            position: 0,
            tail_snapshot: true,
        };
        let mut cache = SessionCache::new(domain(), 1 << 20);
        let tail = STRIDE + 30;
        let mut a = prompt(1, tail + 6);
        a[tail] = MARKER;
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, true), 0);
        assert_eq!(cache.resident.as_ref().unwrap().key.1, STRIDE as u64);
        assert!(cache.resident.as_ref().unwrap().cold_exact);
        assert_eq!(cache.tail.as_ref().unwrap().key.1, tail as u64);
        assert!(!cache.tail.as_ref().unwrap().cold_exact);
        let scope = cache.domain.scoped("toy");
        assert!(!cache.present(&scope, tail as u64, prefix_fingerprint(&a[..tail])));
        let pool_bytes = cache.stored_bytes();
        let chunk_key = cache.resident.as_ref().unwrap().key.clone();

        // Extending the final message changes the very token the tail precedes.
        let mut b = a[..tail].to_vec();
        b.extend(prompt(2, 2 * STRIDE));
        b.push(MARKER);
        b.extend([7, 8, 9]);
        assert_eq!(a.iter().zip(&b).take_while(|(a, b)| a == b).count(), tail);
        assert_eq!(cache.plan(&toy, &b, SessionRoute::Ar), tail);
        cache.begin(gpu, &mut toy, &b, SessionRoute::Ar, tail).unwrap();
        assert_eq!(cache.resident.as_ref().unwrap().key, chunk_key);
        assert_eq!(cache.tail.as_ref().unwrap().key.1, tail as u64);
        assert!(!cache.turn.as_ref().unwrap().pool_capture);
        while let Some(p) = cache.next_boundary() {
            toy.advance(gpu, &b[..p]);
            cache.at_boundary(gpu, &mut toy, &b[..p]).unwrap();
            assert!(cache.pending.is_empty());
        }
        toy.advance(gpu, &b);
        cache.commit();
        assert_eq!(cache.stored_bytes(), pool_bytes);
        assert!(!cache.resident.as_ref().unwrap().cold_exact);
        // A tail-restored turn does not split at its own tail: A's tail stays
        // (rows below it are unchanged) and serves a retry of this turn too.
        assert_eq!(cache.tail.as_ref().unwrap().key.1, tail as u64);

        // A lineage-derived chunk restore also cannot publish to the pool.
        let reused = cache.resident.as_ref().unwrap().key.1 as usize;
        cache.begin(gpu, &mut toy, &b, SessionRoute::Ar, reused).unwrap();
        assert!(cache.tail.is_none());
        assert!(!cache.turn.as_ref().unwrap().pool_capture);
        while let Some(p) = cache.next_boundary() {
            toy.advance(gpu, &b[..p]);
            cache.at_boundary(gpu, &mut toy, &b[..p]).unwrap();
        }
        cache.commit();
        assert_eq!(cache.stored_bytes(), pool_bytes);
        assert!(cache.resident.is_some());
        assert!(cache.tail.is_some());
        cache.live_state_reset();
        assert!(cache.resident.is_none());
        assert!(cache.resident_pending.is_none());
        assert!(cache.tail.is_none());
        assert!(cache.tail_pending.is_none());
        assert_eq!(cache.release_bufs.len(), 2);
        cache.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn live_without_capture_arms_no_lineage_boundary() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let mut toy = Toy {
            fixed: gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap(),
            rows: gpu
                .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
                .unwrap(),
            position: STRIDE,
            tail_snapshot: true,
        };
        let mut a = prompt(1, STRIDE + 40);
        a[STRIDE + 30] = MARKER;
        assert_eq!(toy.lineage_boundary(&a, STRIDE, a.len()), Some(STRIDE + 30));
        let mut cache = SessionCache::new(domain(), 1 << 20);
        cache.begin_live(gpu, &mut toy, &a, SessionRoute::Ar, STRIDE, false);
        assert!(cache.turn.is_none());
        assert_eq!(cache.next_boundary(), None);
        assert!(cache.tail_pending.is_none());
        cache.commit();
        assert!(cache.tail.is_none());
        cache.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn delta_snapshots_restore_exactly_share_prefixes_and_evict_leaves() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
            tail_snapshot: false,
        };
        // Every snapshot below holds the fixed part plus one stride of rows.
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 4 * one);
        let a = prompt(1, 3 * STRIDE + 5);

        // Uncommitted captures are invisible and abandoned (with their
        // parent links) by the next begin.
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, false), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), 0);
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, true), 0);
        assert!(cache.pending.is_empty());

        // A three-link chain, each link one stride of rows.
        assert_eq!(cache.pool.total_bytes(), 3 * one);
        assert_restores(&mut cache, gpu, &mut toy, &a, 3 * STRIDE);

        // A branch shares a's first link.
        let mut b = a[..STRIDE].to_vec();
        b.extend(prompt(2, STRIDE + 7));
        assert_eq!(turn(&mut cache, gpu, &mut toy, &b, true), STRIDE);
        assert_eq!(cache.pool.total_bytes(), 4 * one);
        assert_restores(&mut cache, gpu, &mut toy, &b, 2 * STRIDE);

        // A full budget evicts the least recently used leaf (a's deepest),
        // never a parent that still has children.
        let c = prompt(3, STRIDE + 9);
        turn(&mut cache, gpu, &mut toy, &c, true);
        assert_eq!(cache.plan(&toy, &c, SessionRoute::Ar), STRIDE);
        assert_restores(&mut cache, gpu, &mut toy, &a, 2 * STRIDE);
        assert_restores(&mut cache, gpu, &mut toy, &b, 2 * STRIDE);
        assert_eq!(cache.children.values().sum::<usize>(), 2);
        cache.clear(gpu);

        // A snapshot larger than the whole budget is skipped, nothing pooled.
        let mut tiny = SessionCache::new(domain(), one - 1);
        turn(&mut tiny, gpu, &mut toy, &a, true);
        assert_eq!(tiny.stored_bytes(), 0);
        assert!(tiny.children.is_empty());
        tiny.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn resident_snapshot_serves_the_live_rows_until_they_change() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
            tail_snapshot: false,
        };
        // No pool snapshot fits the budget; only the resident one serves.
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), one - 1);
        let a = prompt(1, 3 * STRIDE + 5);
        let mut branch = a[..3 * STRIDE + 2].to_vec();
        branch.extend(prompt(2, 9));

        // Uncommitted it is invisible; committed, the deepest boundary
        // restores exactly over the live rows.
        turn(&mut cache, gpu, &mut toy, &a, false);
        assert_eq!(cache.plan(&toy, &branch, SessionRoute::Ar), 0);
        turn(&mut cache, gpu, &mut toy, &a, true);
        assert_eq!(cache.stored_bytes(), 0);
        assert_restores(&mut cache, gpu, &mut toy, &branch, 3 * STRIDE);
        // A turn that starts from it keeps it.
        assert_eq!(turn(&mut cache, gpu, &mut toy, &branch, true), 3 * STRIDE);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), 3 * STRIDE);

        // A cold turn of another prompt rewrites the live rows: dropped, and
        // that turn's deepest boundary takes its place.
        let c = prompt(3, STRIDE + 3);
        turn(&mut cache, gpu, &mut toy, &c, true);
        assert_eq!(cache.plan(&toy, &branch, SessionRoute::Ar), 0);
        assert_eq!(cache.plan(&toy, &c, SessionRoute::Ar), STRIDE);
        // So does a reset outside the cache.
        cache.live_state_reset();
        assert_eq!(cache.plan(&toy, &c, SessionRoute::Ar), 0);
        cache.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn live_begin_keeps_state_drops_pending_and_captures_nothing() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
            tail_snapshot: false,
        };
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 4 * one);
        let a = prompt(1, 2 * STRIDE + 5);

        // An uncommitted turn leaves pending snapshots and an armed turn.
        turn(&mut cache, gpu, &mut toy, &a, false);
        assert_eq!(cache.pending.len(), 2);

        // A live begin drops them, arms no boundary and leaves the state alone.
        let before = toy.bytes(gpu);
        cache.begin_live(gpu, &mut toy, &a, SessionRoute::Ar, a.len(), false);
        assert!(cache.pending.is_empty());
        assert!(cache.children.is_empty());
        assert_eq!(cache.next_boundary(), None);
        assert_eq!(toy.position, a.len());
        assert_eq!(toy.bytes(gpu), before);

        // Committing the live turn publishes nothing.
        cache.commit();
        assert_eq!(cache.pool.total_bytes(), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), 0);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn prompt_aware_boundaries_plan_arm_and_capture_live() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
            tail_snapshot: false,
        };
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 64 * one);
        let scoped = domain().scoped("toy");
        let held = |cache: &SessionCache, prompt: &[u32], p: usize| {
            cache
                .pool
                .contains(&scoped, p as u64, prefix_fingerprint(&prompt[..p]))
        };

        // Markers at 40 and 70 sit below one stride (and are not page
        // aligned); 200 sits between strides.
        let mut m = prompt(4, 2 * STRIDE + 10);
        for at in [39, 69, 199] {
            m[at] = MARKER;
        }
        assert_eq!(
            toy.snapshot_boundaries(&m, 0, m.len()),
            [40, 70, STRIDE, 200, 2 * STRIDE]
        );
        assert_eq!(toy.snapshot_boundaries(&m, 40, 200), [70, STRIDE, 200]);

        // begin arms the marker boundaries; the cold turn captures each and
        // commit publishes them although they are not page multiples.
        cache.begin(gpu, &mut toy, &m, SessionRoute::Ar, 0).unwrap();
        assert_eq!(cache.next_boundary(), Some(40));
        cache.commit();
        assert_eq!(turn(&mut cache, gpu, &mut toy, &m, true), 0);
        for p in [40, 70, STRIDE, 200, 2 * STRIDE] {
            assert!(held(&cache, &m, p), "boundary {p} missing");
        }
        assert_eq!(cache.pool.len(), 5);

        // plan finds the marker-boundary snapshot below the first stride.
        let mut early = m[..70].to_vec();
        early.extend(prompt(5, 20));
        assert_eq!(cache.plan(&toy, &early, SessionRoute::Ar), 70);
        assert_restores(&mut cache, gpu, &mut toy, &early, 70);

        // A live continuation that captures: state at the end of m, extended
        // by a message with a marker. Only boundaries above `reused` are
        // armed, and commit publishes them.
        assert_eq!(turn(&mut cache, gpu, &mut toy, &m, true), 2 * STRIDE);
        let mut tail = prompt(6, 30);
        tail[9] = MARKER;
        let mut live = m.clone();
        live.extend(tail);
        let mut probe = live.clone();
        probe.push(1);
        assert_eq!(cache.plan(&toy, &probe, SessionRoute::Ar), 2 * STRIDE);
        cache.begin_live(gpu, &mut toy, &live, SessionRoute::Ar, m.len(), true);
        let marker_end = m.len() + 10;
        assert_eq!(cache.next_boundary(), Some(marker_end));
        toy.advance(gpu, &live[..marker_end]);
        cache
            .at_boundary(gpu, &mut toy, &live[..marker_end])
            .unwrap();
        assert_eq!(cache.next_boundary(), None);
        toy.advance(gpu, &live);
        assert_eq!(cache.plan(&toy, &probe, SessionRoute::Ar), 2 * STRIDE);
        cache.commit();
        assert!(held(&cache, &live, marker_end));
        assert_eq!(cache.plan(&toy, &probe, SessionRoute::Ar), marker_end);
        assert_restores(&mut cache, gpu, &mut toy, &probe, marker_end);
        cache.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    #[test]
    fn disk_tier_demotes_promotes_and_survives_reopen() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
            tail_snapshot: false,
        };
        let dir = tempfile::tempdir().unwrap();
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let open = |build: u8| {
            SessionCache::new(domain(), 2 * one).with_disk_build(
                dir.path(),
                64 * (one + 8192),
                [build; 32],
            )
        };
        let snaps = || {
            std::fs::read_dir(dir.path())
                .unwrap()
                .filter(|e| e.as_ref().unwrap().path().extension() == Some("snap".as_ref()))
                .count()
        };
        let c = prompt(1, 2 * STRIDE + 5);
        let d = prompt(2, 2 * STRIDE + 5);

        // d's captures demote c's two snapshots (leaf first).
        let mut cache = open(7);
        assert!(cache.disk.is_some());
        turn(&mut cache, gpu, &mut toy, &c, true);
        turn(&mut cache, gpu, &mut toy, &d, true);
        assert_eq!(snaps(), 2);
        assert_eq!(cache.pool.len(), 2);

        // Restoring c promotes both links, demoting d's.
        assert_restores(&mut cache, gpu, &mut toy, &c, 2 * STRIDE);
        assert_eq!(snaps(), 2);
        let scoped = domain().scoped("toy");
        let c2 = prefix_fingerprint(&c[..2 * STRIDE]);
        assert!(cache.pool.contains(&scoped, 2 * STRIDE as u64, c2));

        // clear demotes everything; a second process is locked out.
        cache.clear(gpu);
        assert_eq!(snaps(), 4);
        assert!(open(7).disk.is_none());

        // A reopened cache restores both chains exactly from disk.
        drop(cache);
        let mut cache = open(7);
        assert_restores(&mut cache, gpu, &mut toy, &d, 2 * STRIDE);
        assert_restores(&mut cache, gpu, &mut toy, &c, 2 * STRIDE);
        cache.clear(gpu);
        drop(cache);

        // Another build never plans this build's files.
        let other = open(8);
        assert_eq!(other.plan(&toy, &c, SessionRoute::Ar), 0);
        drop(other);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    const AR: SessionRoute = SessionRoute::Ar;
    const MTP: SessionRoute = SessionRoute::Mtp;
    const END: u32 = 7;

    fn record(tokens: &[u32], route: SessionRoute) -> LiveRecord {
        LiveRecord {
            tokens: tokens.to_vec(),
            route,
        }
    }

    #[test]
    fn stride_boundaries_union_strides_and_message_ends() {
        let mut prompt = vec![1u32; 20];
        prompt[2] = END; // ends at 3
        prompt[7] = END; // ends at 8: a stride multiple
        prompt[13] = END; // ends at 14
        assert_eq!(stride_boundaries(4, &prompt, None, 0, 20), [4, 8, 12, 16, 20]);
        // Union, sorted and deduplicated (8 is both).
        assert_eq!(
            stride_boundaries(4, &prompt, Some(END), 0, 20),
            [3, 4, 8, 12, 14, 16, 20]
        );
        // `after` is exclusive and `up_to` inclusive for both sources.
        assert_eq!(stride_boundaries(4, &prompt, Some(END), 3, 14), [4, 8, 12, 14]);
        assert_eq!(stride_boundaries(4, &prompt, None, 3, 14), [4, 8, 12]);
        // Message ends below one stride are boundaries on their own.
        assert_eq!(stride_boundaries(64, &prompt, Some(END), 0, 20), [3, 8, 14]);
        assert!(stride_boundaries(64, &prompt, None, 0, 20).is_empty());
        // Consecutive ends each count; `up_to` past the prompt never reads past it.
        assert_eq!(stride_boundaries(64, &[END, END, 1, END], Some(END), 0, 9), [1, 2, 4]);
    }

    #[test]
    fn live_start_requires_a_strict_extension_at_the_owner_position() {
        let rec = record(&[1, 2, 3], AR);
        let at = |prompt: &[u32]| live_start(Some(&rec), prompt, AR, Some(3));
        assert_eq!(at(&[1, 2, 3, 4]), Some(3));
        assert_eq!(at(&[1, 2, 3, 4, 5, 6]), Some(3));
        // Diverges inside the record.
        assert_eq!(at(&[1, 9, 3, 4]), None);
        // L == prompt.len() leaves no suffix; a shorter prompt is no extension.
        assert_eq!(at(&[1, 2, 3]), None);
        assert_eq!(at(&[1, 2]), None);
        // No record, an empty one, another route, or owners off the end.
        assert_eq!(live_start(None, &[1, 2, 3, 4], AR, Some(3)), None);
        assert_eq!(live_start(Some(&record(&[], AR)), &[1, 2], AR, Some(0)), None);
        assert_eq!(live_start(Some(&rec), &[1, 2, 3, 4], MTP, Some(3)), None);
        assert_eq!(live_start(Some(&rec), &[1, 2, 3, 4], AR, Some(2)), None);
        assert_eq!(live_start(Some(&rec), &[1, 2, 3, 4], AR, None), None);
    }

    #[test]
    fn live_after_commit_needs_consumed_to_extend_the_turn_prompt() {
        let prompt = [1, 2, 3];
        let commit = |consumed: &[u32], pos: usize| live_after_commit(&prompt, consumed, Some(pos));
        assert!(commit(&[1, 2, 3, 7, 8], 5));
        // Nothing generated: consumed == prompt.
        assert!(commit(&[1, 2, 3], 3));
        // Unrepaired terminal: the host history is empty.
        assert!(!commit(&[], 0));
        // Shorter than the prompt, or not extending it.
        assert!(!commit(&[1, 2], 2));
        assert!(!commit(&[1, 9, 3, 7], 4));
        // Owners not at the end of the consumed history, or unable to continue.
        assert!(!commit(&[1, 2, 3, 7, 8], 4));
        assert!(!live_after_commit(&prompt, &[1, 2, 3, 7], None));
        // A turn without a prompt is never live.
        assert!(!live_after_commit(&[], &[1], Some(1)));
    }

    #[test]
    fn plan_prefers_live_on_a_tie_and_the_longer_source_otherwise() {
        assert_eq!(plan_start(4096, None), 4096);
        assert_eq!(plan_start(0, None), 0);
        assert_eq!(plan_start(0, Some(300)), 300);
        assert_eq!(plan_start(4096, Some(4096)), 4096);
        assert_eq!(plan_start(4096, Some(9000)), 9000);
        assert_eq!(plan_start(8192, Some(300)), 8192);
    }

    /// GPU: capped snapshot/live selection and canonical boundary forwarding.
    #[test]
    fn driver_continues_live_state_and_restores_snapshots() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let fixed = gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap();
        let rows = gpu
            .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
            .unwrap();
        let mut toy = Toy {
            fixed,
            rows,
            position: 0,
            tail_snapshot: false,
        };
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut driver = SessionDriver::new(SessionCache::new(domain(), 8 * one));
        let prefill = |driver: &mut SessionDriver, toy: &mut Toy, gpu: &mut Gpu, p: &[u32]| {
            let reused = driver.plan(toy, p, AR);
            let live = driver.begin(gpu, toy, p, AR, reused, false).unwrap();
            while let Some(b) = driver.next_boundary() {
                toy.advance(gpu, &p[..b]);
                driver.at_turn_boundary(gpu, toy, b).unwrap();
            }
            toy.advance(gpu, p);
            (reused, live)
        };
        // Turn 1, cold; the decoded tail extends the state, commit keeps it live.
        let turn1 = prompt(1, STRIDE + 5);
        assert_eq!(prefill(&mut driver, &mut toy, gpu, &turn1), (0, false));
        let mut consumed = turn1.clone();
        consumed.extend([11, 12]);
        toy.advance(gpu, &consumed);
        driver.commit_live(&toy, &consumed);
        // Turn 2 extends the consumed history: continue in place.
        let mut turn2 = consumed.clone();
        turn2.extend(prompt(2, 9));
        // The cap applies to each source, not to the final reuse length:
        // rejecting the deeper live record must fall back to a real snapshot.
        for (cap, expected) in [
            (0, 0),
            (STRIDE - 1, 0),
            (STRIDE, STRIDE),
            (consumed.len() - 1, STRIDE),
            (consumed.len(), consumed.len()),
            (consumed.len() + 1, consumed.len()),
            (turn2.len(), consumed.len()),
            (usize::MAX, consumed.len()),
        ] {
            assert_eq!(driver.plan_capped(&toy, &turn2, AR, cap), expected);
        }
        assert_eq!(driver.plan(&toy, &turn2, AR), consumed.len());
        assert_eq!(driver.plan_capped(&toy, &turn2, MTP, usize::MAX), STRIDE);
        let mut divergent = turn2.clone();
        divergent[STRIDE] ^= 1;
        assert_eq!(driver.plan_capped(&toy, &divergent, AR, usize::MAX), STRIDE);
        for cap in [0, STRIDE, usize::MAX] {
            assert_eq!(driver.plan_capped(&toy, &[], AR, cap), 0);
        }
        assert_eq!(
            prefill(&mut driver, &mut toy, gpu, &turn2),
            (consumed.len(), true)
        );
        assert_eq!(toy.bytes(gpu), (fixed_of(&turn2), rows_of(&turn2)));
        // A plain commit drops the live state: another conversation sharing
        // turn 1's first stride restores the snapshot.
        driver.commit();
        let mut other = turn1[..STRIDE].to_vec();
        other.extend(prompt(3, 4));
        assert!(driver.live.is_none());
        for (cap, expected) in [
            (0, 0),
            (STRIDE - 1, 0),
            (STRIDE, STRIDE),
            (STRIDE + 1, STRIDE),
            (usize::MAX, STRIDE),
        ] {
            assert_eq!(driver.plan_capped(&toy, &other, AR, cap), expected);
        }
        assert_eq!(prefill(&mut driver, &mut toy, gpu, &other), (STRIDE, false));
        assert_eq!(toy.bytes(gpu), (fixed_of(&other), rows_of(&other)));
        // A shorter live conversation loses to a deeper snapshot only when
        // that snapshot is within the cap.
        let short = &other[..STRIDE / 2];
        assert!(!driver.begin(gpu, &mut toy, short, AR, 0, false).unwrap());
        toy.advance(gpu, short);
        driver.commit_live(&toy, short);
        for (cap, expected) in [
            (short.len() - 1, 0),
            (short.len(), short.len()),
            (STRIDE - 1, short.len()),
            (STRIDE, STRIDE),
            (usize::MAX, STRIDE),
        ] {
            assert_eq!(driver.plan_capped(&toy, &other, AR, cap), expected);
        }
        driver.clear(gpu);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    /// GPU: real captures distinguish a resident from pooled/live reuse.
    #[test]
    fn driver_validates_canonical_boundaries_and_preserves_resident_until_reset() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let mut toy = Toy {
            fixed: gpu.alloc_tensor(&[FIXED_BYTES], DType::Raw).unwrap(),
            rows: gpu
                .alloc_tensor(&[ROW_CAPACITY * ROW_BYTES], DType::Raw)
                .unwrap(),
            position: 0,
            tail_snapshot: false,
        };
        let one = layout([FIXED_BYTES, STRIDE * ROW_BYTES]).1 as u64;
        let mut driver = SessionDriver::new(SessionCache::new(domain(), one - 1));
        assert_eq!(
            driver.at_turn_boundary(gpu, &mut toy, STRIDE).unwrap_err(),
            "session boundary has no prompt in flight"
        );

        let canonical = prompt(1, 2 * STRIDE + 5);
        let mut request = canonical.clone();
        assert!(!driver.begin(gpu, &mut toy, &request, MTP, 0, false).unwrap());
        assert_eq!(driver.turn_route(), Some(MTP));
        // The caller's buffer is no longer the prompt in flight.
        request[0] ^= 1;
        assert_eq!(
            driver
                .at_turn_boundary(gpu, &mut toy, canonical.len() + 1)
                .unwrap_err(),
            "session boundary exceeds the prompt in flight"
        );
        assert!(driver.at_turn_boundary(gpu, &mut toy, STRIDE - 1).is_err());
        assert_eq!(driver.next_boundary(), Some(STRIDE));
        assert_eq!(toy.position, 0);
        assert!(driver.cache.pending.is_empty());
        assert!(driver.cache.resident_pending.is_none());
        while let Some(b) = driver.next_boundary() {
            toy.advance(gpu, &canonical[..b]);
            driver.at_turn_boundary(gpu, &mut toy, b).unwrap();
        }
        toy.advance(gpu, &canonical);
        assert_eq!(driver.plan(&toy, &canonical, MTP), 0);
        driver.commit();
        assert_eq!(driver.turn_route(), None);
        assert!(driver.live.is_none());
        assert_eq!(driver.cache.stored_bytes(), 0);
        assert_eq!(driver.plan(&toy, &canonical, MTP), 2 * STRIDE);
        assert_eq!(driver.plan(&toy, &request, MTP), 0);
        // Even a second, empty commit must not invalidate the resident.
        driver.commit();
        assert_eq!(driver.plan(&toy, &canonical, MTP), 2 * STRIDE);

        let mut continuation = canonical.clone();
        continuation.extend(prompt(2, STRIDE));
        assert!(!driver
            .begin(gpu, &mut toy, &continuation, MTP, 2 * STRIDE, false)
            .unwrap());
        assert_eq!(
            toy.bytes(gpu),
            (fixed_of(&canonical[..2 * STRIDE]), rows_of(&canonical[..2 * STRIDE]))
        );
        toy.advance(gpu, &continuation[..3 * STRIDE]);
        driver.at_turn_boundary(gpu, &mut toy, 3 * STRIDE).unwrap();
        assert!(driver.cache.resident.is_some());
        assert!(driver.cache.resident_pending.is_some());
        // An external reset invalidates both published and uncommitted
        // residents; deferred buffers are released by clear.
        toy.reset(gpu).unwrap();
        driver.forget_live();
        assert_eq!(driver.turn_route(), None);
        assert!(driver.cache.resident.is_none());
        assert!(driver.cache.resident_pending.is_none());
        assert_eq!(driver.cache.release_bufs.len(), 2);
        assert_eq!(driver.plan(&toy, &continuation, MTP), 0);
        assert_eq!(
            driver.at_turn_boundary(gpu, &mut toy, 3 * STRIDE).unwrap_err(),
            "session boundary has no prompt in flight"
        );
        driver.clear(gpu);
        assert!(driver.cache.release_bufs.is_empty());
        assert_eq!(driver.next_boundary(), None);

        // Clear must also drop a committed live record and resident, not
        // just the deferred buffers from forget_live.
        driver.begin(gpu, &mut toy, &canonical, AR, 0, false).unwrap();
        while let Some(b) = driver.next_boundary() {
            toy.advance(gpu, &canonical[..b]);
            driver.at_turn_boundary(gpu, &mut toy, b).unwrap();
        }
        toy.advance(gpu, &canonical);
        driver.commit_live(&toy, &canonical);
        assert!(driver.live.is_some());
        assert!(driver.cache.resident.is_some());
        driver.clear(gpu);
        assert!(driver.live.is_none());
        assert_eq!(driver.turn_route(), None);
        assert!(driver.cache.resident.is_none());
        assert!(driver.cache.resident_pending.is_none());
        assert_eq!(driver.cache.stored_bytes(), 0);
        assert_eq!(driver.plan(&toy, &continuation, AR), 0);
        gpu.free_tensor(toy.fixed).unwrap();
        gpu.free_tensor(toy.rows).unwrap();
    }

    /// A CPU-only state that only names its snapshot scope.
    struct ScopeOnly(&'static str);

    impl SessionState for ScopeOnly {
        fn snapshot_scope(&self, _route: SessionRoute) -> Option<String> {
            Some(self.0.to_string())
        }
        fn snapshot_boundaries(&self, _: &[u32], _: usize, _: usize) -> Vec<usize> {
            Vec::new()
        }
        fn snapshot_parts(
            &mut self,
            _: SessionRoute,
            _: usize,
        ) -> Result<SnapshotParts<'_>, String> {
            Err("scope only".into())
        }
        fn restore_parts(
            &mut self,
            _: &mut Gpu,
            _: SessionRoute,
            _: &[u8],
        ) -> Result<StateLayout<'_>, String> {
            Err("scope only".into())
        }
        fn finish_restore(&mut self, _: &mut Gpu, _: SessionRoute, _: &[u8]) -> Result<(), String> {
            Err("scope only".into())
        }
        fn live_position(&self, _: SessionRoute) -> Option<usize> {
            None
        }
        fn growth_reserve_bytes(&self, _: &Gpu, _: usize) -> u64 {
            0
        }
        fn reset(&mut self, _: &mut Gpu) -> Result<(), String> {
            Err("scope only".into())
        }
    }

    const BAND: SessionRoute = SessionRoute::Ar;

    fn band_driver() -> SessionDriver {
        SessionDriver::new(SessionCache::new(domain(), 1 << 20))
    }

    fn piece(driver: &SessionDriver, prefix: &[u32]) -> SessionPieceKey {
        driver.piece_key(&ScopeOnly("band"), prefix, BAND).unwrap()
    }

    /// A pool snapshot with a harmless disk location: it owns no device
    /// buffer, so nothing needs freeing.
    fn fixture(parent: Option<Key>) -> StoredSnapshot {
        StoredSnapshot {
            location: SnapshotLocation::Disk {
                path: PathBuf::new(),
                file_bytes: 1,
            },
            parent,
            fixed_bytes: Vec::new(),
            segments: Vec::new(),
            meta: Vec::new(),
            bytes: 1,
        }
    }

    /// Publish `key` into the pool of `driver`, as a child of `parent` when
    /// given (linked like a captured delta).
    fn seed(driver: &mut SessionDriver, key: &SessionPieceKey, parent: Option<&SessionPieceKey>) {
        if let Some(parent) = parent {
            driver.cache.link(&parent.0);
        }
        let (domain, p, fp) = key.0.clone();
        let (id, displaced) = driver.cache.pool.insert_unaligned(
            domain,
            p,
            fp,
            fixture(parent.map(|parent| parent.0.clone())),
        );
        assert!(id != CheckpointId::NONE);
        assert!(displaced.is_empty());
    }

    fn evict(driver: &mut SessionDriver, key: &SessionPieceKey) {
        assert!(driver
            .cache
            .pool
            .evict(&key.0 .0, key.0 .1, key.0 .2)
            .is_some());
    }

    fn pinned(driver: &SessionDriver, key: &SessionPieceKey) -> bool {
        driver.cache.pool.is_pinned(&key.0 .0, key.0 .1, key.0 .2)
    }

    fn mesh_record(pieces: Vec<SessionPieceKey>) -> MeshSessionRecord {
        MeshSessionRecord {
            route: BAND,
            pieces,
        }
    }

    /// Two bands with `prefix` seeded in both pools.
    fn two_bands(prefix: &[u32]) -> (SessionDriver, SessionDriver, Vec<SessionPieceKey>) {
        let (mut a, mut b) = (band_driver(), band_driver());
        let pieces = vec![piece(&a, prefix), piece(&b, prefix)];
        seed(&mut a, &pieces[0], None);
        seed(&mut b, &pieces[1], None);
        (a, b, pieces)
    }

    #[test]
    fn piece_key_names_exactly_the_cache_domain_scope() {
        let driver = band_driver();
        let prefix = [1, 2, 3];
        let key = driver.piece_key(&ScopeOnly("band"), &prefix, BAND).unwrap();
        assert_eq!(key.prefix_len(), 3);
        assert_eq!(
            key.0,
            (
                driver.cache.domain.scoped("band"),
                3,
                prefix_fingerprint(&prefix)
            )
        );
        assert!(key.matches_prefix(&prefix));
        assert!(!key.matches_prefix(&[1, 2, 4]));
        assert!(!key.matches_prefix(&[1, 2]));
        assert_eq!(key.clone(), key);
        assert_ne!(
            driver
                .piece_key(&ScopeOnly("other"), &prefix, BAND)
                .unwrap(),
            key
        );
        assert_ne!(piece(&driver, &[1, 2, 4]), key);
        assert!(driver.piece_key(&ScopeOnly("band"), &[], BAND).is_none());
    }

    #[test]
    fn load_epoch_binds_the_domain_of_a_fresh_driver_only() {
        let mut driver = band_driver();
        let before = piece(&driver, &[1, 2]);
        driver.bind_load_epoch(7).unwrap();
        assert_eq!(driver.cache.domain.model_load_epoch, 7);
        assert_ne!(piece(&driver, &[1, 2]), before);
        // A pooled snapshot makes the driver non-fresh: refused, unchanged.
        let key = piece(&driver, &[1, 2]);
        seed(&mut driver, &key, None);
        assert!(driver.bind_load_epoch(8).is_err());
        assert_eq!(driver.cache.domain.model_load_epoch, 7);
        // So do a pending capture and a live state.
        let mut pending = band_driver();
        let key = piece(&pending, &[1, 2]);
        pending.cache.pending.push((key.0.clone(), fixture(None)));
        assert!(pending.bind_load_epoch(8).is_err());
        assert_eq!(pending.cache.domain.model_load_epoch, 1);
        let mut live = band_driver();
        live.live = Some(LiveRecord {
            tokens: vec![1, 2],
            route: BAND,
        });
        assert!(live.bind_load_epoch(8).is_err());
        assert_eq!(live.cache.domain.model_load_epoch, 1);
    }

    #[test]
    fn set_boundaries_overrides_the_armed_schedule_or_changes_nothing() {
        let mut driver = band_driver();
        // No turn: only the empty schedule is accepted.
        assert!(driver.set_boundaries(Vec::new(), None).is_ok());
        assert!(driver.set_boundaries(vec![4], None).is_err());
        driver.cache.turn = Some(Turn {
            domain: domain(),
            route: BAND,
            boundaries: VecDeque::from([8, 16]),
            lineage: None,
            pool_capture: true,
            start: 2,
            end: 20,
        });
        assert_eq!(driver.turn_boundaries(), vec![8, 16]);
        driver.set_boundaries(vec![4, 12, 20], Some(12)).unwrap();
        assert_eq!(driver.turn_boundaries(), vec![4, 12, 20]);
        assert_eq!(driver.turn_lineage(), Some(12));
        assert_eq!(driver.next_boundary(), Some(4));
        for (boundaries, lineage) in [
            (vec![4, 4], None),
            (vec![8, 4], None),
            (vec![2, 8], None),
            (vec![8, 21], None),
            (vec![4, 12], Some(8)),
            (vec![4, 12], Some(20)),
            (vec![4, 20], Some(20)),
            (vec![4, 12], Some(2)),
        ] {
            assert!(driver.set_boundaries(boundaries, lineage).is_err());
            assert_eq!(driver.turn_boundaries(), vec![4, 12, 20]);
            assert_eq!(driver.turn_lineage(), Some(12));
        }
        driver.set_boundaries(Vec::new(), None).unwrap();
        assert!(driver.turn_boundaries().is_empty());
        assert_eq!(driver.turn_lineage(), None);
    }

    #[test]
    fn pin_piece_needs_a_pool_snapshot_and_counts_pins() {
        let mut driver = band_driver();
        let key = piece(&driver, &[1, 2, 3]);
        assert!(!driver.piece_resident(&key));
        assert!(!driver.pin_piece(&key));
        assert!(driver.cache.children.is_empty());
        seed(&mut driver, &key, None);
        assert!(driver.piece_resident(&key));
        assert!(driver.pin_piece(&key));
        assert!(driver.pin_piece(&key));
        assert!(pinned(&driver, &key));
        driver.unpin_piece(&key);
        assert!(pinned(&driver, &key));
        driver.unpin_piece(&key);
        assert!(!pinned(&driver, &key));
        assert!(driver.cache.children.is_empty());
        assert!(driver.cache.pool.pop_lru().is_some());
    }

    #[test]
    fn an_unpublished_piece_is_never_pinnable() {
        let mut driver = band_driver();
        let key = piece(&driver, &[1, 2, 3]);
        driver.cache.pending.push((key.0.clone(), fixture(None)));
        assert!(!driver.piece_resident(&key));
        assert!(!driver.pin_piece(&key));
        assert!(driver.cache.children.is_empty());
    }

    #[test]
    fn publish_commits_pending_and_plans_the_longest_strict_prefix() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        // A deeper piece is still pending on both bands.
        let deep = [1, 2, 3, 4];
        let deep_pieces = vec![piece(&a, &deep), piece(&b, &deep)];
        a.cache
            .pending
            .push((deep_pieces[0].0.clone(), fixture(None)));
        b.cache
            .pending
            .push((deep_pieces[1].0.clone(), fixture(None)));
        let mut mesh = MeshSessionCache::new();
        mesh.stage(mesh_record(pieces));
        mesh.stage(mesh_record(deep_pieces.clone()));
        assert_eq!(mesh.staged_len(), 2);
        let mut bands = [&mut a, &mut b];
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3, 4, 5], BAND), 0);
        assert_eq!(mesh.publish(&mut bands), 2);
        assert_eq!((mesh.len(), mesh.staged_len()), (2, 0));
        assert!(bands[0].cache.pending.is_empty() && bands[1].cache.pending.is_empty());
        assert!(bands[0].piece_resident(&deep_pieces[0]));
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3, 4, 5], BAND), 4);
        // Strictly shorter than the prompt.
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3, 4], BAND), 2);
        assert_eq!(mesh.plan(&mut bands, &[1, 2], BAND), 0);
        assert_eq!(mesh.plan(&mut bands, &[1, 9, 3, 4, 5], BAND), 0);
        assert_eq!(
            mesh.plan(&mut bands, &[1, 2, 3, 4, 5], SessionRoute::Mtp),
            0
        );
        assert!(pinned(bands[0], &deep_pieces[0]) && pinned(bands[1], &deep_pieces[1]));
    }

    #[test]
    fn abort_drops_staged_candidates_and_keeps_published_records() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        mesh.stage(mesh_record(pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 1);
        mesh.stage(mesh_record(pieces.clone()));
        mesh.abort();
        assert_eq!((mesh.len(), mesh.staged_len()), (1, 0));
        assert_eq!(mesh.publish(&mut bands), 0);
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3], BAND), 2);
        // The aborted candidate took no pin.
        assert_eq!(bands[0].cache.children[&pieces[0].0], 1);
    }

    #[test]
    fn a_missing_or_inconsistent_piece_drops_the_candidate_and_leaves_no_pin() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let gone = piece(&b, &[1, 2, 3]);
        let first = piece(&a, &[1, 2, 3]);
        seed(&mut a, &first, None);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        mesh.stage(mesh_record(vec![first.clone(), gone]));
        mesh.stage(mesh_record(pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 1);
        assert_eq!(mesh.len(), 1);
        assert!(!pinned(bands[0], &first));
        assert_eq!(bands[0].cache.children.len(), 1);
        assert!(bands[1].cache.pool.pop_lru().is_none());
        // Pieces of different prefixes, a different owner count and an empty
        // record are refused as well.
        mesh.stage(mesh_record(vec![pieces[0].clone(), first.clone()]));
        mesh.stage(mesh_record(vec![pieces[0].clone()]));
        mesh.stage(mesh_record(Vec::new()));
        assert_eq!(mesh.publish(&mut bands), 0);
        assert!(!pinned(bands[0], &first));
        assert_eq!(bands[0].cache.children.len(), 1);
    }

    #[test]
    fn probing_pins_all_or_nothing() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let absent = piece(&b, &[1, 2, 3]);
        let first = piece(&a, &[1, 2, 3]);
        seed(&mut a, &first, None);
        let mut bands = [&mut a, &mut b];
        assert!(!probe_and_pin_pieces(&mut bands, &[first.clone(), absent]));
        assert!(!pinned(bands[0], &first));
        assert!(bands[0].cache.children.is_empty() && bands[1].cache.children.is_empty());
        assert!(!probe_and_pin_pieces(&mut bands, &pieces[..1]));
        assert!(probe_and_pin_pieces(&mut bands, &pieces));
        assert!(pinned(bands[0], &pieces[0]) && pinned(bands[1], &pieces[1]));
        unpin_pieces(&mut bands, &pieces);
        assert!(!pinned(bands[0], &pieces[0]) && !pinned(bands[1], &pieces[1]));
        assert!(bands[0].cache.children.is_empty() && bands[1].cache.children.is_empty());
    }

    #[test]
    fn an_evicted_piece_invalidates_its_record_on_every_band() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        mesh.stage(mesh_record(pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 1);
        // A pinned piece is never an LRU victim.
        assert!(bands[0].cache.pool.pop_lru().is_none());
        assert!(bands[1].cache.pool.pop_lru().is_none());
        // One band loses its piece outside the pins.
        evict(bands[1], &pieces[1]);
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3], BAND), 0);
        assert!(mesh.is_empty());
        // The surviving band's pin was released with the record.
        assert!(!pinned(bands[0], &pieces[0]));
        assert!(bands[0].cache.children.is_empty() && bands[1].cache.children.is_empty());
        assert!(bands[0].cache.pool.pop_lru().is_some());
    }

    #[test]
    fn evict_oldest_releases_every_band_pin_of_the_oldest_record() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let newer = [1, 2, 3];
        let newer_pieces = vec![piece(&a, &newer), piece(&b, &newer)];
        seed(&mut a, &newer_pieces[0], None);
        seed(&mut b, &newer_pieces[1], None);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        assert!(!mesh.evict_oldest(&mut bands));
        mesh.stage(mesh_record(pieces.clone()));
        mesh.stage(mesh_record(newer_pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 2);
        assert!(mesh.evict_oldest(&mut bands));
        assert_eq!(mesh.len(), 1);
        assert!(!pinned(bands[0], &pieces[0]) && !pinned(bands[1], &pieces[1]));
        assert!(pinned(bands[0], &newer_pieces[0]) && pinned(bands[1], &newer_pieces[1]));
        // The released pieces are now LRU victims on every band.
        assert!(bands[0].cache.pool.pop_lru().is_some());
        assert!(bands[1].cache.pool.pop_lru().is_some());
        assert!(mesh.evict_oldest(&mut bands));
        assert!(!mesh.evict_oldest(&mut bands));
        assert!(bands[0].cache.children.is_empty() && bands[1].cache.children.is_empty());
    }

    #[test]
    fn records_cap_at_the_bound_and_release_the_oldest() {
        let (mut a, mut b) = (band_driver(), band_driver());
        let mut mesh = MeshSessionCache::new();
        let mut all = Vec::new();
        for len in 1..=MESH_SESSION_MAX_RECORDS + 1 {
            let prefix = vec![5; len];
            let pieces = vec![piece(&a, &prefix), piece(&b, &prefix)];
            seed(&mut a, &pieces[0], None);
            seed(&mut b, &pieces[1], None);
            mesh.stage(mesh_record(pieces.clone()));
            all.push(pieces);
            let mut bands = [&mut a, &mut b];
            assert_eq!(mesh.publish(&mut bands), 1);
            assert!(mesh.len() <= MESH_SESSION_MAX_RECORDS);
        }
        assert_eq!(mesh.len(), MESH_SESSION_MAX_RECORDS);
        let mut bands = [&mut a, &mut b];
        let long = vec![5; MESH_SESSION_MAX_RECORDS + 3];
        assert_eq!(
            mesh.plan(&mut bands, &long, BAND),
            MESH_SESSION_MAX_RECORDS + 1
        );
        let oldest = &all[0];
        assert!(!pinned(bands[0], &oldest[0]) && !pinned(bands[1], &oldest[1]));
        assert!(!bands[0].cache.children.contains_key(&oldest[0].0));
        let newest = all.last().unwrap();
        assert!(pinned(bands[0], &newest[0]) && pinned(bands[1], &newest[1]));
        assert_eq!(bands[0].cache.children.len(), MESH_SESSION_MAX_RECORDS);
    }

    #[test]
    fn republishing_a_prefix_replaces_the_record_without_leaking_pins() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        for _ in 0..3 {
            mesh.stage(mesh_record(pieces.clone()));
            assert_eq!(mesh.publish(&mut bands), 1);
            assert_eq!(mesh.len(), 1);
            assert_eq!(bands[0].cache.children[&pieces[0].0], 1);
        }
    }

    #[test]
    fn shared_children_refcount_keeps_a_parent_pinned_until_both_release() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        // A child snapshot on band a that names the record's piece as parent.
        let child = piece(&a, &[1, 2, 3, 4]);
        seed(&mut a, &child, Some(&pieces[0]));
        assert_eq!(a.cache.children[&pieces[0].0], 1);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        mesh.stage(mesh_record(pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 1);
        assert_eq!(bands[0].cache.children[&pieces[0].0], 2);
        // The record releases its pin; the child still holds the parent.
        mesh.clear(&mut bands);
        assert!(mesh.is_empty());
        assert_eq!(bands[0].cache.children[&pieces[0].0], 1);
        assert!(pinned(bands[0], &pieces[0]));
        assert!(!pinned(bands[1], &pieces[1]));
        // Dropping the child (as `drop_snapshot` does) unpins the parent.
        bands[0].cache.unlink(&pieces[0].0);
        assert!(!pinned(bands[0], &pieces[0]));
        assert!(bands[0].cache.children.is_empty());
    }

    #[test]
    fn clear_releases_published_pins_and_staged_candidates() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        mesh.stage(mesh_record(pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 1);
        mesh.stage(mesh_record(pieces.clone()));
        mesh.clear(&mut bands);
        assert_eq!((mesh.len(), mesh.staged_len()), (0, 0));
        assert!(bands[0].cache.children.is_empty() && bands[1].cache.children.is_empty());
        assert!(!pinned(bands[0], &pieces[0]) && !pinned(bands[1], &pieces[1]));
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3], BAND), 0);
    }

    #[test]
    fn planned_record_is_selectable_and_releasable() {
        let (mut a, mut b, pieces) = two_bands(&[1, 2]);
        let deep = [1, 2, 3];
        let deep_pieces = vec![piece(&a, &deep), piece(&b, &deep)];
        seed(&mut a, &deep_pieces[0], None);
        seed(&mut b, &deep_pieces[1], None);
        let mut mesh = MeshSessionCache::new();
        let mut bands = [&mut a, &mut b];
        assert!(mesh.planned().is_none());
        assert!(!mesh.release_plan(&mut bands));
        mesh.stage(mesh_record(pieces.clone()));
        mesh.stage(mesh_record(deep_pieces.clone()));
        assert_eq!(mesh.publish(&mut bands), 2);
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3, 4], BAND), 3);
        let (len, chosen) = mesh.planned().unwrap();
        assert_eq!(len, 3);
        assert_eq!(chosen.pieces, deep_pieces);
        // A miss clears the plan; the next plan selects again.
        assert_eq!(mesh.plan(&mut bands, &[9, 9, 9], BAND), 0);
        assert!(mesh.planned().is_none());
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3, 4], BAND), 3);
        // Releasing drops only the chosen record and its pins.
        assert!(mesh.release_plan(&mut bands));
        assert!(mesh.planned().is_none());
        assert_eq!(mesh.len(), 1);
        assert!(!pinned(bands[0], &deep_pieces[0]) && !pinned(bands[1], &deep_pieces[1]));
        assert!(pinned(bands[0], &pieces[0]) && pinned(bands[1], &pieces[1]));
        assert!(!mesh.release_plan(&mut bands));
        // A planned record that is evicted afterwards is no longer planned.
        assert_eq!(mesh.plan(&mut bands, &[1, 2, 3], BAND), 2);
        assert!(mesh.planned().is_some());
        assert!(mesh.evict_oldest(&mut bands));
        assert!(mesh.planned().is_none());
    }
}
