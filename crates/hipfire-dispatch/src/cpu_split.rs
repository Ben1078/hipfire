// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Eager CPU-cold/GPU-hot slot transaction. The original route never changes.
//!
//! `prepare` is the sole input-readiness wait, before the GPU routed tail.
//! `join` runs scoped CPU arithmetic while that tail executes, then queues cold
//! slot installation behind its masked zero stores. Neither arm reads residual.
//!
//! Lab diagnostics (`configure`, `SplitParityMode`) are OFF by default. When off,
//! every hot path pays exactly one `Cell<bool>` read and keeps the production
//! byte order, stream order and overlap. Timing/Full modes add the endpoint
//! override, an optional serial join and host-clock/buffer capture; they never
//! run in production.

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::time::Instant;
use hip_bridge::{HipError, PinnedTransferBuffer};
use rdna_compute::{DType, Gpu, GpuTensor};
use hipfire_cpu::quant::CpuQuant;
use crate::cpu_exec::{self, CpuMoeExpert};
use crate::families::moe::RoutedExpertWeights;
use crate::types::DispatchError;

pub const CPU_SPLIT_MAX_ROWS: usize = 4;
const MAX_SLOTS: usize = CPU_SPLIT_MAX_ROWS * 10;

fn err(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("CPU tier split: {message}"))
}

/// CPU-readable bytes of ANY expert of a layer, hot or cold, independent of
/// tier owners. Implemented by the lab over the artifact mmap.
pub trait CpuShadow: Send + Sync {
    fn expert(&self, layer: usize, expert: usize) -> Option<CpuMoeExpert<'_>>;
}

/// Frozen placement for one diagnostic run.
/// `AllHot`: every VALID slot is a GPU slot (host_count = 0, masked ids == ids).
/// `AllCold`: every VALID slot is a host slot (masked ids all -1); EVERY valid
/// slot's CPU bytes come from `shadow`. Invalid ids stay in neither arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitEndpoint { Natural, AllHot, AllCold }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitParityMode { Off, Timing, Full }

#[derive(Clone)]
pub struct SplitParityConfig {
    pub mode: SplitParityMode,
    /// `None` = every armed layer; other layers run production unchanged.
    pub layers: Option<Vec<usize>>,
    pub endpoint: SplitEndpoint,
    /// `join` drains the stream once BEFORE the CPU slot arithmetic.
    pub serial: bool,
    /// REQUIRED when `endpoint == AllCold`.
    pub shadow: Option<Arc<dyn CpuShadow>>,
    /// Record cap; extra calls run (endpoint/serial still apply) unrecorded.
    pub max_records: usize,
}

/// Host `Instant` nanoseconds; only filled for wanted layers in Timing/Full.
/// In Full mode the pre-combine drain makes these serialized, not overlap, times.
#[derive(Clone, Debug, Default)]
pub struct SplitTiming {
    /// prepare() entry -> stream idle AFTER the D2 combine.
    pub wall_ns: u64,
    /// prepare() wall (D2H + completes + payload validate + mask H2D enqueue).
    pub prepare_ns: u64,
    /// End of prepare -> start of join (host time enqueueing the masked GPU tail).
    pub gpu_enqueue_ns: u64,
    /// CPU slot arithmetic inside join (`moe_cpu_host_slot_outputs`).
    pub join_cpu_ns: u64,
    /// join() wall incl. installing H2D enqueues (and the leading sync if serial).
    pub join_total_ns: u64,
    /// End of join -> stream idle after the combine.
    pub tail_ns: u64,
    pub cold_slots: u32,
    pub hot_slots: u32,
    pub invalid_slots: u32,
    pub serial: bool,
}

/// One record per recorded layer call, in call order. Timing mode leaves every
/// tensor Vec empty.
#[derive(Clone, Debug)]
pub struct SplitParityRecord {
    pub layer: usize,
    pub rows: usize,
    pub dim: usize,
    pub mi: usize,
    pub top_k: usize,
    pub endpoint: SplitEndpoint,
    /// `placement()` clone, 0 hot / 1 cold.
    pub placement: Vec<u8>,
    /// [rows*dim] bit-exact executor input.
    pub x_rot: Vec<f32>,
    /// [rows*top_k] ORIGINAL route as consumed.
    pub ids: Vec<i32>,
    /// [rows*top_k]
    pub weights: Vec<f32>,
    /// [rows*dim] combine target BEFORE the D2 combine.
    pub target_in: Vec<f32>,
    /// [rows*top_k*dim] expanded exactly as the combine consumed it.
    pub d2_expanded: Vec<f32>,
    /// [rows*dim] combine target AFTER the D2 combine.
    pub d2_target: Vec<f32>,
    /// [rows*top_k*dim] all-GPU tail on ORIGINAL ids, run after the D2 combine.
    pub gpu_expanded: Vec<f32>,
    pub timing: SplitTiming,
}

/// Hard ceiling on `max_records` (guards unbounded lab memory).
pub const SPLIT_PARITY_MAX_RECORDS: usize = 1 << 20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage { Prepared, Joined, Before, After }

struct Open {
    t0: Instant,
    t_prep_end: Instant,
    t_join_end: Instant,
    stage: Stage,
    full: bool,
    rec: SplitParityRecord,
}

struct ParityState {
    mode: SplitParityMode,
    endpoint: SplitEndpoint,
    serial: bool,
    shadow: Option<Arc<dyn CpuShadow>>,
    max_records: usize,
    records: Vec<SplitParityRecord>,
    open: Option<Open>,
}

fn sync_stream(gpu: &Gpu) -> Result<(), DispatchError> {
    match gpu.active_stream.as_ref() {
        Some(s) => gpu.hip.stream_synchronize(s),
        None => gpu.hip.stream_synchronize_default(),
    }
    .map_err(err)?;
    gpu.take_device_wait_error().map_err(err)
}

fn read_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Result<Vec<f32>, DispatchError> {
    if t.buf.size() < n * 4 { return Err(err("parity capture tensor too small")); }
    let mut v = vec![0.0f32; n];
    // SAFETY: `v` owns exactly n f32s; every byte pattern is a valid f32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).map_err(err)?;
    gpu.take_device_wait_error().map_err(err)?;
    Ok(v)
}

fn ns(from: Instant, to: Instant) -> u64 {
    to.saturating_duration_since(from).as_nanos().min(u64::MAX as u128) as u64
}

struct Transaction {
    input: PinnedTransferBuffer,
    mask: PinnedTransferBuffer,
    output: PinnedTransferBuffer,
    ids: [i32; MAX_SLOTS],
    gpu_ids: [i32; MAX_SLOTS],
    host_slots: [u32; MAX_SLOTS],
    host_count: usize,
    rows: usize,
    ready: bool,
    failed: bool,
    /// Set by `prepare` only for an `AllCold` diagnostic run.
    cold_shadow: Option<Arc<dyn CpuShadow>>,
}

/// Preallocated per-layer transaction and immutable expert ownership plan.
/// A runtime must release this with `free_gpu` while its device is bound, just
/// like its other model-owned tensors. CPU work never outlives `join`.
pub struct CpuSplitExecutor {
    dim: usize,
    mi: usize,
    top_k: usize,
    max_rows: usize,
    device: i32,
    placement: Box<[u8]>,
    masked_indices: GpuTensor,
    state: RefCell<Transaction>,
    /// One-read gate for every diagnostic path; false == production.
    diag_on: Cell<bool>,
    /// A record is open for the in-flight transaction.
    recording: Cell<bool>,
    /// The open record is Full and still awaits `parity_reference`.
    ref_pending: Cell<bool>,
    layer: Cell<usize>,
    parity: RefCell<ParityState>,
}

impl CpuSplitExecutor {
    pub fn new(
        gpu: &Gpu,
        dim: usize,
        mi: usize,
        top_k: usize,
        max_rows: usize,
        placement: &[u8],
    ) -> Result<Self, DispatchError> {
        if dim == 0 || dim % 256 != 0 || mi == 0 || mi % 128 != 0
            || top_k != 10 || max_rows != CPU_SPLIT_MAX_ROWS
            || placement.is_empty() || placement.len() > i32::MAX as usize
            || placement.iter().any(|&v| v > 1)
        {
            return Err(err("unsupported shape, row cap or ownership plan"));
        }
        gpu.bind_thread().map_err(err)?;
        let slots = max_rows.checked_mul(top_k).ok_or_else(|| err("slot overflow"))?;
        let input_values = max_rows.checked_mul(dim)
            .and_then(|v| slots.checked_mul(2).and_then(|t| v.checked_add(t)))
            .ok_or_else(|| err("input overflow"))?;
        let output_values = slots.checked_mul(dim).ok_or_else(|| err("output overflow"))?;
        let input_bytes = input_values.checked_mul(4).ok_or_else(|| err("input byte overflow"))?;
        let mask_bytes = slots.checked_mul(4).ok_or_else(|| err("mask byte overflow"))?;
        let output_bytes = output_values.checked_mul(4).ok_or_else(|| err("output byte overflow"))?;
        let input = gpu.hip.transfer_buffer(input_bytes).map_err(err)?;
        let mask = gpu.hip.transfer_buffer(mask_bytes).map_err(err)?;
        let output = gpu.hip.transfer_buffer(output_bytes).map_err(err)?;
        // Allocate last: every preceding fallible operation has a draining owner.
        let masked_indices = GpuTensor {
            buf: gpu.hip.malloc(mask_bytes).map_err(err)?,
            shape: vec![slots],
            dtype: DType::F32,
        };
        Ok(Self {
            dim, mi, top_k, max_rows, device: gpu.device_id,
            placement: placement.into(), masked_indices,
            state: RefCell::new(Transaction {
                input, mask, output,
                ids: [-1; MAX_SLOTS], gpu_ids: [-1; MAX_SLOTS],
                host_slots: [0; MAX_SLOTS], host_count: 0, rows: 0,
                ready: false, failed: false, cold_shadow: None,
            }),
            diag_on: Cell::new(false),
            recording: Cell::new(false),
            ref_pending: Cell::new(false),
            layer: Cell::new(0),
            parity: RefCell::new(ParityState {
                mode: SplitParityMode::Off,
                endpoint: SplitEndpoint::Natural,
                serial: false,
                shadow: None,
                max_records: 0,
                records: Vec::new(),
                open: None,
            }),
        })
    }

    pub fn masked_indices(&self) -> &GpuTensor { &self.masked_indices }
    pub fn placement(&self) -> &[u8] { &self.placement }
    pub fn max_rows(&self) -> usize { self.max_rows }
    pub fn dim(&self) -> usize { self.dim }
    pub fn mi(&self) -> usize { self.mi }
    pub fn top_k(&self) -> usize { self.top_k }

    /// Freeze one diagnostic run for this layer. Replaces any previous
    /// configuration and drops its records. Takes no GPU: the next `prepare`
    /// retires all pinned owners before touching them, so no drain is needed
    /// here, but an in-flight (prepared, un-joined) or failed transaction is
    /// refused. `Off` restores production exactly and requires a natural,
    /// non-serial config. A layer outside `layers` runs production unchanged.
    pub fn configure(&self, layer: usize, cfg: SplitParityConfig) -> Result<(), DispatchError> {
        let st = self.state.try_borrow().map_err(|_| err("configure during an active transaction"))?;
        if st.failed || st.ready { return Err(err("configure with a failed or unfinished transaction")); }
        drop(st);
        let active = cfg.mode != SplitParityMode::Off;
        if !active {
            if cfg.endpoint != SplitEndpoint::Natural || cfg.serial {
                return Err(err("parity mode Off requires the natural endpoint and serial=false"));
            }
        } else if cfg.max_records == 0 || cfg.max_records > SPLIT_PARITY_MAX_RECORDS {
            return Err(err(format!("max_records must be 1..={SPLIT_PARITY_MAX_RECORDS}")));
        }
        if cfg.endpoint == SplitEndpoint::AllCold && cfg.shadow.is_none() {
            return Err(err("AllCold endpoint requires a CPU shadow"));
        }
        let wanted = active && cfg.layers.as_ref().map_or(true, |l| l.contains(&layer));
        let mut p = self.parity.try_borrow_mut().map_err(|_| err("configure re-entered"))?;
        self.recording.set(false);
        self.ref_pending.set(false);
        self.layer.set(layer);
        p.open = None;
        p.records = Vec::new();
        if wanted {
            p.mode = cfg.mode;
            p.endpoint = cfg.endpoint;
            p.serial = cfg.serial;
            p.shadow = cfg.shadow;
            p.max_records = cfg.max_records;
        } else {
            p.mode = SplitParityMode::Off;
            p.endpoint = SplitEndpoint::Natural;
            p.serial = false;
            p.shadow = None;
            p.max_records = 0;
        }
        self.diag_on.set(wanted);
        Ok(())
    }

    /// Drain recorded parity records in call order.
    pub fn take_parity_records(&self) -> Vec<SplitParityRecord> {
        std::mem::take(&mut self.parity.borrow_mut().records)
    }

    /// True from a successful `prepare` that opened a record until that record
    /// completes (after `parity_after_combine` in Timing, `parity_reference` in
    /// Full) or errors. False in production and for unrecorded calls.
    pub fn wants_parity(&self) -> bool { self.recording.get() }

    /// True while the open record is Full and still awaits the unmasked GPU
    /// reference tail + `parity_reference`.
    pub fn wants_reference(&self) -> bool { self.ref_pending.get() }

    fn close_open(&self) {
        self.recording.set(false);
        self.ref_pending.set(false);
        if let Ok(mut p) = self.parity.try_borrow_mut() { p.open = None; }
    }

    /// Shared hook frame: no-op unless a record is open; validates rows and
    /// eager state; any error discards the open record. `f` returns true when
    /// the record is complete and must be pushed.
    fn hook(
        &self, gpu: &Gpu, rows: usize,
        f: impl FnOnce(&Gpu, &mut Open) -> Result<bool, DispatchError>,
    ) -> Result<(), DispatchError> {
        if !self.recording.get() { return Ok(()); }
        let mut p = self.parity.borrow_mut();
        let result = (|| {
            let open = p.open.as_mut().ok_or_else(|| err("parity record not open"))?;
            if open.rec.rows != rows { return Err(err("parity rows differ from prepare")); }
            self.gate(gpu, rows)?;
            f(gpu, open)
        })();
        match result {
            Ok(false) => Ok(()),
            Ok(true) => {
                let open = p.open.take().ok_or_else(|| err("parity record vanished"))?;
                p.records.push(open.rec);
                self.recording.set(false);
                self.ref_pending.set(false);
                Ok(())
            }
            Err(e) => {
                p.open = None;
                self.recording.set(false);
                self.ref_pending.set(false);
                Err(e)
            }
        }
    }

    /// Call after join, BEFORE the canonical combine. Full: drains the stream and
    /// captures `target` (combine input) and `expanded` as the combine will read
    /// them (lab only, serializes). Timing: no-op. Production/unrecorded: no-op.
    pub fn parity_before_combine(
        &self, gpu: &Gpu, rows: usize, expanded: &GpuTensor, target: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.hook(gpu, rows, |gpu, open| {
            if open.stage != Stage::Joined { return Err(err("parity_before_combine out of order")); }
            if open.full {
                let (nt, ne) = (rows * self.dim, rows * self.top_k * self.dim);
                sync_stream(gpu)?;
                open.rec.target_in = read_f32(gpu, target, nt)?;
                open.rec.d2_expanded = read_f32(gpu, expanded, ne)?;
                open.stage = Stage::Before;
            }
            Ok(false)
        })
    }

    /// Call right after the canonical combine is enqueued. Drains the stream
    /// (the one timing sync), stamps wall/tail times BEFORE any readback, then
    /// Timing completes the record; Full captures the combined `target` and
    /// waits for `parity_reference`.
    pub fn parity_after_combine(
        &self, gpu: &Gpu, rows: usize, target: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.hook(gpu, rows, |gpu, open| {
            let expect = if open.full { Stage::Before } else { Stage::Joined };
            if open.stage != expect { return Err(err("parity_after_combine out of order")); }
            sync_stream(gpu)?;
            let now = Instant::now();
            open.rec.timing.wall_ns = ns(open.t0, now);
            open.rec.timing.tail_ns = ns(open.t_join_end, now);
            if open.full {
                open.rec.d2_target = read_f32(gpu, target, rows * self.dim)?;
                open.stage = Stage::After;
                Ok(false)
            } else {
                Ok(true)
            }
        })
    }

    /// Full only: call after the caller reran the UNMASKED all-GPU tail into the
    /// same `expanded` (after the combine consumed it). Captures `gpu_expanded`
    /// and completes the record. No-op for Timing/unrecorded.
    pub fn parity_reference(
        &self, gpu: &Gpu, rows: usize, expanded: &GpuTensor,
    ) -> Result<(), DispatchError> {
        self.hook(gpu, rows, |gpu, open| {
            if !open.full { return Ok(false); }
            if open.stage != Stage::After { return Err(err("parity_reference out of order")); }
            sync_stream(gpu)?;
            open.rec.gpu_expanded = read_f32(gpu, expanded, rows * self.top_k * self.dim)?;
            Ok(true)
        })
    }

    fn resolve<'a>(
        &self, gpu: &Gpu, experts: &'a dyn RoutedExpertWeights,
        shadow: Option<&'a dyn CpuShadow>, id: usize,
    ) -> Option<CpuMoeExpert<'a>> {
        match shadow {
            Some(s) => s.expert(self.layer.get(), id),
            None => experts.cpu_splice_expert(gpu, id),
        }
    }

    fn gate(&self, gpu: &Gpu, rows: usize) -> Result<(), DispatchError> {
        if rows == 0 || rows > self.max_rows || gpu.device_id != self.device {
            return Err(err("invalid width or device"));
        }
        let capture = gpu.active_stream.as_ref()
            .map(|s| gpu.hip.stream_is_capturing(s)).transpose().map_err(err)?
            .unwrap_or(false);
        if gpu.graphs.capture_mode || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some() || capture
            || gpu.replay.is_recording() || gpu.replay.retained_body_active()
            || gpu.replay.should_route_aql() || gpu.replay.should_route_pm4()
            || gpu.replay.shadow_body_route().is_some()
            || gpu.replay.state() == rdna_compute::replay::ReplayState::ShadowValidated
        {
            return Err(err("requires eager HIP; capture, retained replay, AQL/PM4 and shadow are refused"));
        }
        gpu.bind_thread().map_err(err)
    }

    /// Download ORIGINAL route/rotated activation and retire all old pinned
    /// reads before launching any routed GPU kernels for this generation.
    pub fn prepare(
        &self, gpu: &Gpu, rows: usize, x_rot: &GpuTensor,
        ids: &GpuTensor, weights: &GpuTensor, experts: &dyn RoutedExpertWeights,
    ) -> Result<(), DispatchError> {
        let diag = self.diag_on.get();
        let t0 = diag.then(Instant::now);
        self.gate(gpu, rows)?;
        let slots = rows * self.top_k;
        let nx = rows * self.dim;
        if x_rot.buf.size() < nx * 4 || ids.buf.size() < slots * 4
            || weights.buf.size() < slots * 4 || experts.len() != self.placement.len()
        {
            return Err(err("input extent or expert plan mismatch"));
        }
        // Admission may run on a different thread than orchestration. Reserve
        // its reusable CPU scratch here, before the parallel GPU/CPU phase.
        cpu_exec::reserve_moe_cpu_host_slot_scratch(
            self.dim, self.mi, self.max_rows * self.top_k,
        )?;
        let (endpoint, shadow) = if diag {
            // A record left open by an aborted caller never leaks into this call.
            self.close_open();
            let p = self.parity.borrow();
            let shadow = if p.endpoint == SplitEndpoint::AllCold {
                Some(p.shadow.clone().ok_or_else(|| err("AllCold endpoint lost its CPU shadow"))?)
            } else {
                None
            };
            (p.endpoint, shadow)
        } else {
            (SplitEndpoint::Natural, None)
        };
        let mut st = self.state.borrow_mut();
        if st.failed || st.ready { return Err(err("failed or unfinished transaction")); }
        let result = (|| {
            st.input.download_async(&gpu.hip, &x_rot.buf, 0, nx * 4, gpu.active_stream.as_ref()).map_err(err)?;
            st.input.download_async(&gpu.hip, &ids.buf, nx * 4, slots * 4, gpu.active_stream.as_ref()).map_err(err)?;
            st.input.download_async(&gpu.hip, &weights.buf, (nx + slots) * 4, slots * 4, gpu.active_stream.as_ref()).map_err(err)?;
            st.input.complete(&gpu.hip).map_err(err)?;
            // These may synchronize the stream, intentionally BEFORE its tail.
            st.mask.complete(&gpu.hip).map_err(err)?;
            st.output.complete(&gpu.hip).map_err(err)?;
            for s in 0..slots {
                let at = (nx + s) * 4;
                let id = i32::from_ne_bytes(st.input.as_bytes_mut()[at..at + 4].try_into().unwrap());
                st.ids[s] = id;
                st.gpu_ids[s] = -1;
            }
            st.host_count = 0;
            for s in 0..slots {
                let id = st.ids[s];
                // Invalid original slots belong to neither arm; preserve them
                // for the existing router-fault reporting/canonical reducer.
                if id < 0 || id as usize >= self.placement.len() { continue; }
                let cold = match endpoint {
                    SplitEndpoint::Natural => self.placement[id as usize] == 1,
                    SplitEndpoint::AllHot => false,
                    SplitEndpoint::AllCold => true,
                };
                if cold {
                    let n = st.host_count;
                    st.host_slots[n] = s as u32;
                    st.host_count += 1;
                } else {
                    st.gpu_ids[s] = id;
                }
            }
            self.validate_payloads(gpu, &st, rows, experts, shadow.as_deref())?;
            for s in 0..slots {
                let bytes = st.gpu_ids[s].to_ne_bytes();
                st.mask.as_bytes_mut()[s * 4..s * 4 + 4].copy_from_slice(&bytes);
            }
            st.mask.upload_async(&gpu.hip, &self.masked_indices.buf, 0, slots * 4, gpu.active_stream.as_ref()).map_err(err)?;
            st.rows = rows;
            st.ready = true;
            Ok(())
        })();
        if result.is_err() {
            st.failed = true;
            // Keep this transaction poisoned even after a successful drain:
            // a partially advanced model request must never silently retry.
            let _ = gpu.hip.device_synchronize();
            if diag { self.close_open(); }
        } else {
            st.cold_shadow = shadow;
            if let Some(t0) = t0 {
                self.open_record(&mut st, t0, rows, endpoint);
            }
        }
        result
    }

    /// Lab only: open the parity record for a successfully prepared call.
    fn open_record(&self, st: &mut Transaction, t0: Instant, rows: usize, endpoint: SplitEndpoint) {
        let prepare_ns = ns(t0, Instant::now());
        let slots = rows * self.top_k;
        let nx = rows * self.dim;
        let mut p = self.parity.borrow_mut();
        if p.records.len() >= p.max_records { return; }
        let full = p.mode == SplitParityMode::Full;
        let cold = st.host_count;
        let hot = st.gpu_ids[..slots].iter().filter(|&&v| v >= 0).count();
        let mut rec = SplitParityRecord {
            layer: self.layer.get(), rows, dim: self.dim, mi: self.mi, top_k: self.top_k,
            endpoint,
            placement: Vec::new(), x_rot: Vec::new(), ids: Vec::new(), weights: Vec::new(),
            target_in: Vec::new(), d2_expanded: Vec::new(), d2_target: Vec::new(),
            gpu_expanded: Vec::new(),
            timing: SplitTiming {
                prepare_ns,
                cold_slots: cold as u32,
                hot_slots: hot as u32,
                invalid_slots: (slots - hot - cold) as u32,
                serial: p.serial,
                ..SplitTiming::default()
            },
        };
        if full {
            rec.placement = self.placement.to_vec();
            rec.ids = st.ids[..slots].to_vec();
            let bytes = st.input.as_bytes_mut();
            let f32s = |range: std::ops::Range<usize>| -> Vec<f32> {
                bytes[range.start * 4..range.end * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                    .collect()
            };
            rec.x_rot = f32s(0..nx);
            rec.weights = f32s(nx + slots..nx + 2 * slots);
        }
        let now = Instant::now();
        p.open = Some(Open { t0, t_prep_end: now, t_join_end: now, stage: Stage::Prepared, full, rec });
        self.recording.set(true);
        self.ref_pending.set(full);
    }

    fn validate_payloads(
        &self, gpu: &Gpu, st: &Transaction, rows: usize, experts: &dyn RoutedExpertWeights,
        shadow: Option<&dyn CpuShadow>,
    ) -> Result<(), DispatchError> {
        let dummy = CpuMoeExpert {
            gate_up_quant: CpuQuant::F32, gate_up: &[],
            down_quant: CpuQuant::F32, down: &[],
        };
        let mut resolved = [dummy; MAX_SLOTS];
        for (j, &slot) in st.host_slots[..st.host_count].iter().enumerate() {
            let id = st.ids[slot as usize] as usize;
            resolved[j] = self.resolve(gpu, experts, shadow, id)
                .ok_or_else(|| err(format!("cold expert {id} has no retained host spans")))?;
        }
        cpu_exec::validate_moe_cpu_host_slot_outputs(
            self.dim, self.mi, rows, self.top_k, self.placement.len(),
            &st.ids[..rows * self.top_k], Some(&st.gpu_ids[..rows * self.top_k]),
            &st.host_slots[..st.host_count], &resolved[..st.host_count],
            rows * self.dim, false, rows * self.top_k * self.dim,
        )
    }

    /// GPU down is already queued. NO stream completion belongs in this
    /// method: it would turn the intended CPU/GPU overlap into serialization.
    /// (Lab `serial=true` is the one deliberate exception: a single drain
    /// before the CPU arithmetic.) Cold slots are installed as one H2D per run
    /// of adjacent slots: identical bytes and stream order, fewer HIP calls.
    pub fn join(
        &self, gpu: &Gpu, rows: usize, expanded: &GpuTensor,
        experts: &dyn RoutedExpertWeights,
    ) -> Result<(), DispatchError> {
        self.gate(gpu, rows)?;
        let slots = rows * self.top_k;
        if expanded.buf.size() < slots * self.dim * 4 { return Err(err("short expanded output")); }
        let mut st = self.state.borrow_mut();
        if st.failed || !st.ready || st.rows != rows { return Err(err("join without matching prepare")); }
        let diag = self.diag_on.get();
        let tj0 = diag.then(Instant::now);
        let serial = diag && self.parity.borrow().serial;
        let shadow = st.cold_shadow.clone();
        let result = (|| {
            if serial { sync_stream(gpu)?; }
            let dummy = CpuMoeExpert {
                gate_up_quant: CpuQuant::F32, gate_up: &[],
                down_quant: CpuQuant::F32, down: &[],
            };
            let mut resolved = [dummy; MAX_SLOTS];
            for (j, &slot) in st.host_slots[..st.host_count].iter().enumerate() {
                resolved[j] = self.resolve(gpu, experts, shadow.as_deref(), st.ids[slot as usize] as usize)
                    .ok_or_else(|| err("immutable cold payload disappeared"))?;
            }
            let Transaction { input, output, ids, gpu_ids, host_slots, host_count, .. } = &mut *st;
            // SAFETY: hipHostMalloc storage is aligned, initialized and idle.
            // Input completion and previous output retirement happened BEFORE
            // the GPU tail; neither buffer has been submitted since then.
            let x = unsafe { std::slice::from_raw_parts(input.as_bytes_mut().as_ptr().cast::<f32>(), rows * self.dim) };
            let out = unsafe { std::slice::from_raw_parts_mut(output.as_bytes_mut().as_mut_ptr().cast::<f32>(), slots * self.dim) };
            let t_cpu = diag.then(Instant::now);
            cpu_exec::moe_cpu_host_slot_outputs(
                self.dim, self.mi, rows, self.top_k, self.placement.len(),
                &ids[..slots], Some(&gpu_ids[..slots]), &host_slots[..*host_count],
                &resolved[..*host_count], x, false, out,
            )?;
            let cpu_ns = t_cpu.map_or(0, |t| ns(t, Instant::now()));
            let hs = &host_slots[..*host_count];
            let mut i = 0;
            while i < hs.len() {
                // Row-major ascending slots: adjacent slots are adjacent in both
                // the pinned output and `expanded`, so a run is one contiguous copy.
                let mut j = i + 1;
                while j < hs.len() && hs[j] == hs[j - 1] + 1 { j += 1; }
                let offset = hs[i] as usize * self.dim * 4;
                let len = (j - i) * self.dim * 4;
                // Non-owning DeviceBuffer view: no shape Vec allocation and no
                // GPU arithmetic. Same-stream FIFO orders the down zero stores
                // before these writes and the canonical combine after them.
                let dst = expanded.buf.byte_view(offset, len);
                output.upload_async(&gpu.hip, &dst, offset, len, gpu.active_stream.as_ref()).map_err(err)?;
                i = j;
            }
            Ok(cpu_ns)
        })();
        st.ready = false;
        match result {
            Ok(cpu_ns) => {
                if let Some(tj0) = tj0 {
                    if self.recording.get() {
                        let now = Instant::now();
                        let mut p = self.parity.borrow_mut();
                        if let Some(o) = p.open.as_mut() {
                            o.rec.timing.gpu_enqueue_ns = ns(o.t_prep_end, tj0);
                            o.rec.timing.join_cpu_ns = cpu_ns;
                            o.rec.timing.join_total_ns = ns(tj0, now);
                            o.t_join_end = now;
                            o.stage = Stage::Joined;
                        }
                    }
                }
                Ok(())
            }
            Err(e) => {
                st.failed = true;
                let _ = gpu.hip.device_synchronize();
                if diag { self.close_open(); }
                Err(e)
            }
        }
    }

    /// Per-layer live transfer volume: original x/ids/weights download, masked
    /// ids upload, and just the cold F32 expanded slot returns. No residual.
    pub fn transfer_bytes(&self, rows: usize, cold_slots: usize) -> (usize, usize) {
        (4 * rows * (self.dim + 2 * self.top_k),
         4 * rows * self.top_k + 4 * cold_slots * self.dim)
    }

    pub fn free_gpu(self, gpu: &Gpu) -> Result<(), HipError> {
        gpu.bind_thread()?;
        if let Err(e) = gpu.hip.device_synchronize() {
            // Both pinned copy sources and the mask allocation may still be
            // referenced. Retain the whole owner if retirement is unproved.
            std::mem::forget(self);
            return Err(e);
        }
        let Self { masked_indices, state, .. } = self;
        drop(state);
        gpu.hip.free(masked_indices.buf)
    }
}
