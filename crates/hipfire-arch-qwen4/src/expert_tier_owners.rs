// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Immutable model-level expert payload owners and borrowed layer tables.
//!
//! Materialization streams bounded source spans into at most one device and
//! one pinned, device-mapped host allocation. The lease retains ownership via
//! a mutex, while its immutable CPU binding is read without locking.
//!
//! B5-C dependency: `ExpertTierReplayIdentity` and `route_binding()` must be
//! added only after `rdna_compute::route_binding` lands. This module's generation
//! identifies allocations; it is not a replay authorization token.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use hip_bridge::{DeviceBuffer, HipRuntime, HIP_HOST_MALLOC_MAPPED};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};
use crate::expert_tier_plan::{ExpertTierEntry, ExpertTierPlan, Placement};

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const TOTAL_EXPERTS: usize = TRUNK_LAYERS as usize * EXPERTS;
const COPY_CHUNK_BYTES: u64 = 8 * 1024 * 1024;
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// A materialization failure; no partially built lease is published.
#[derive(Debug)]
pub enum ExpertTierOwnerError {
    Runtime(String),
    Source(String),
    SourceLength {
        expected: u64,
        got: usize,
    },
    MissingBase(Placement),
    Overflow,
    Extent {
        tier: Placement,
        offset: u64,
        len: u64,
        owner_bytes: u64,
    },
    Alignment {
        address: u64,
    },
    TailPad {
        tier: Placement,
    },
    GenerationExhausted,
}

impl fmt::Display for ExpertTierOwnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(message) => write!(f, "expert tier HIP operation: {message}"),
            Self::Source(message) => write!(f, "expert tier source: {message}"),
            Self::SourceLength { expected, got } => write!(
                f,
                "expert tier source returned {got} bytes, expected {expected}"
            ),
            Self::MissingBase(tier) => write!(f, "missing {tier:?} owner base"),
            Self::Overflow => write!(f, "expert tier address or extent overflow"),
            Self::Extent {
                tier,
                offset,
                len,
                owner_bytes,
            } => write!(
                f,
                "{tier:?} extent {offset}+{len} exceeds owner size {owner_bytes}"
            ),
            Self::Alignment { address } => {
                write!(f, "expert tier address {address:#x} is not 256-aligned")
            }
            Self::TailPad { tier } => {
                write!(f, "{tier:?} owner has insufficient readable tail padding")
            }
            Self::GenerationExhausted => write!(f, "expert tier generation exhausted"),
        }
    }
}

impl std::error::Error for ExpertTierOwnerError {}

fn issue_generation(counter: &AtomicU64) -> Result<u64, ExpertTierOwnerError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
            last.checked_add(1)
        })
        .map(|last| last + 1)
        .map_err(|_| ExpertTierOwnerError::GenerationExhausted)
}

#[derive(Debug)]
struct Tables {
    gate_up: Vec<usize>,
    down: Vec<usize>,
    codecs: Vec<u8>,
}

fn tier_index(tier: Placement) -> usize {
    match tier {
        Placement::Device => 0,
        Placement::HostMapped => 1,
    }
}

fn checked_extent(
    tier: Placement,
    offset: u64,
    len: u64,
    owner_bytes: u64,
) -> Result<(), ExpertTierOwnerError> {
    let end = offset
        .checked_add(len)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    if len == 0 || end > owner_bytes {
        return Err(ExpertTierOwnerError::Extent {
            tier,
            offset,
            len,
            owner_bytes,
        });
    }
    Ok(())
}

fn checked_address(base: u64, offset: u64) -> Result<usize, ExpertTierOwnerError> {
    let address = base
        .checked_add(offset)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    if address % ExpertTierPlan::PAYLOAD_ALIGN != 0 {
        return Err(ExpertTierOwnerError::Alignment { address });
    }
    usize::try_from(address).map_err(|_| ExpertTierOwnerError::Overflow)
}

fn tail_pad_bytes() -> u64 {
    let align = ExpertTierPlan::PAYLOAD_ALIGN;
    ((ARENA_TAIL_PAD_BYTES as u64).max(align) + align - 1) & !(align - 1)
}

/// Pure construction of the exact CPU mirrors uploaded to the device tables.
fn build_tables(
    plan: &ExpertTierPlan,
    device_base: Option<u64>,
    host_device_alias: Option<u64>,
) -> Result<Tables, ExpertTierOwnerError> {
    // The HIP pointer-table ABI is 64-bit, not a dtype/quantization tag table.
    if std::mem::size_of::<usize>() != 8 {
        return Err(ExpertTierOwnerError::Overflow);
    }
    let bases = [device_base, host_device_alias];
    for tier in [Placement::Device, Placement::HostMapped] {
        let bytes = plan.owner_bytes(tier);
        if bytes != 0 {
            let base = bases[tier_index(tier)].ok_or(ExpertTierOwnerError::MissingBase(tier))?;
            checked_address(base, 0)?;
            let end = base
                .checked_add(bytes)
                .ok_or(ExpertTierOwnerError::Overflow)?;
            usize::try_from(end).map_err(|_| ExpertTierOwnerError::Overflow)?;
        }
    }
    let mut tables = Tables {
        gate_up: Vec::with_capacity(TOTAL_EXPERTS),
        down: Vec::with_capacity(TOTAL_EXPERTS),
        codecs: Vec::with_capacity(TOTAL_EXPERTS),
    };
    let mut used = [0u64; 2];
    for layer in 0..TRUNK_LAYERS as usize {
        for eid in 0..EXPERTS {
            let entry = plan
                .entry(layer, eid)
                .ok_or(ExpertTierOwnerError::Overflow)?;
            let t = tier_index(entry.tier);
            let bytes = plan.owner_bytes(entry.tier);
            let base = bases[t].ok_or(ExpertTierOwnerError::MissingBase(entry.tier))?;
            for (offset, len) in [
                (entry.gate_up_offset, entry.gate_up_len),
                (entry.down_offset, entry.down_len),
            ] {
                checked_extent(entry.tier, offset, len, bytes)?;
                used[t] = used[t].max(offset + len);
            }
            tables
                .gate_up
                .push(checked_address(base, entry.gate_up_offset)?);
            tables.down.push(checked_address(base, entry.down_offset)?);
            tables.codecs.push(match entry.format {
                ExpertFormat::Mq4Xts => 0,
                ExpertFormat::Mq3I4 => 1,
                ExpertFormat::Mq2I4 => 2,
                ExpertFormat::E8pI4 => 3,
            });
        }
    }
    for window in plan.host_windows() {
        for (span, offset) in &window.spans {
            let tier = Placement::HostMapped;
            checked_extent(tier, *offset, span.len as u64, plan.owner_bytes(tier))?;
            let base = host_device_alias.ok_or(ExpertTierOwnerError::MissingBase(tier))?;
            checked_address(base, *offset)?;
            used[1] = used[1].max(*offset + span.len as u64);
        }
    }
    for tier in [Placement::Device, Placement::HostMapped] {
        let bytes = plan.owner_bytes(tier);
        if bytes != 0 && bytes.checked_sub(used[tier_index(tier)]).unwrap_or(0) < tail_pad_bytes() {
            return Err(ExpertTierOwnerError::TailPad { tier });
        }
    }
    Ok(tables)
}

/// The pinned host address is retained only for CPU copies and `hipHostFree`.
/// Storing opaque addresses as integers follows Gpu's host-mapped registry and
/// leaves ownership Send without adding an unsafe auto-trait implementation.
struct HostPayload {
    cpu_address: usize,
    device_alias: u64,
    bytes: usize,
}

/// Owns all expert allocations. Destruction requires GPU quiescence; retained
/// launches must hold a keepalive until their retirement proves that condition.
pub struct ExpertTierOwners {
    device: Option<DeviceBuffer>,
    host: Option<HostPayload>,
    gate_up_table: Option<GpuTensor>,
    down_table: Option<GpuTensor>,
    codec_table: Option<GpuTensor>,
    binding: Option<Arc<ExpertTierBinding>>,
    plan: Arc<ExpertTierPlan>,
    hip: Arc<HipRuntime>,
    device_id: i32,
    generation: u64,
}

/// Immutable CPU state, including the exact mirrors used for table uploads.
/// No GPU buffer handles are stored here: DeviceBuffer is deliberately !Sync.
pub struct ExpertTierBinding {
    plan: Arc<ExpertTierPlan>,
    tables: Tables,
    gate_up_table_base: usize,
    down_table_base: usize,
    codec_table_base: usize,
    /// `(device-visible base, bytes)` per tier, indexed by `tier_index`; the
    /// host entry is the mapped device alias. `None` for an empty owner.
    owner_ranges: [Option<(usize, usize)>; 2],
    /// `(CPU address, bytes)` of the one pinned host owner, for CPU reads and
    /// staging; `None` when no host payload exists.
    host_cpu_range: Option<(usize, usize)>,
    generation: u64,
    /// Raw SHA-256 (hex) of the hot-set sidecar, identity metadata bound once
    /// by the loader after materialization.
    hotset_sidecar_sha256: std::sync::OnceLock<String>,
    logical_digest: [u8; 32],
}

/// Cheap shared retention of a frozen materialized model binding.
#[derive(Clone)]
pub struct ExpertTierLease {
    binding: Arc<ExpertTierBinding>,
    owners: Arc<Mutex<ExpertTierOwners>>,
}

// A compile-time check, not an unsafe implementation. The mutex supplies Sync
// for the Send-only DeviceBuffer owners; the CPU binding is independently Sync.
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Arc<Mutex<ExpertTierOwners>>>();
    send_sync::<ExpertTierLease>();
};

impl ExpertTierLease {
    pub fn layer(&self, layer: usize) -> Option<LayerExpertTierView<'_>> {
        (layer < TRUNK_LAYERS as usize).then_some(LayerExpertTierView { lease: self, layer })
    }

    pub fn owner_bytes(&self, tier: Placement) -> u64 {
        self.binding.plan.owner_bytes(tier)
    }

    /// Immutable `(device-visible base, bytes)` of the tier's one owner, read
    /// without locking. The host range is the mapped device alias. `None`
    /// when the tier holds no payload.
    pub fn owner_range(&self, tier: Placement) -> Option<(usize, usize)> {
        self.binding.owner_ranges[tier_index(tier)]
    }

    /// Immutable `(CPU address, bytes)` of the pinned host owner, read without
    /// locking; valid while this lease (or its keepalive) is retained. `None`
    /// when the plan places nothing in host memory.
    pub fn host_cpu_range(&self) -> Option<(usize, usize)> {
        self.binding.host_cpu_range
    }

    pub fn keepalive(&self) -> Arc<dyn Any + Send + Sync> {
        self.owners.clone()
    }

    pub fn generation(&self) -> u64 {
        self.binding.generation
    }

    pub fn logical_digest(&self) -> [u8; 32] {
        self.binding.logical_digest
    }

    /// Record the raw SHA-256 (lowercase hex) of the hot-set sidecar this plan
    /// was built from. The loader calls it once after `materialize`; it is
    /// identity metadata only (never authorization) and never alters the
    /// plan. Recording the same value again is accepted; a different value or a
    /// malformed digest is refused.
    pub fn bind_hotset_sidecar_sha256(&self, sha256_hex: &str) -> Result<(), String> {
        if sha256_hex.len() != 64
            || !sha256_hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "hot-set sidecar digest must be 64 lowercase hex digits, got {sha256_hex:?}"
            ));
        }
        let bound = self
            .binding
            .hotset_sidecar_sha256
            .get_or_init(|| sha256_hex.to_string());
        if bound == sha256_hex {
            Ok(())
        } else {
            Err(format!(
                "hot-set sidecar digest already bound to {bound}, refusing {sha256_hex}"
            ))
        }
    }

    /// The sidecar SHA-256 bound by [`Self::bind_hotset_sidecar_sha256`], if any.
    pub fn hotset_sidecar_sha256(&self) -> Option<&str> {
        self.binding.hotset_sidecar_sha256.get().map(String::as_str)
    }
}

/// Borrowed frozen tables for one layer, in original eid order. GPU sub-views
/// are non-owning: never free them, and retain `lease()` across pending GPU work.
pub struct LayerExpertTierView<'a> {
    lease: &'a ExpertTierLease,
    layer: usize,
}

impl<'a> LayerExpertTierView<'a> {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn entry(&self, eid: usize) -> Option<&'a ExpertTierEntry> {
        self.lease.binding.plan.entry(self.layer, eid)
    }

    pub fn placement(&self, eid: usize) -> Option<Placement> {
        self.entry(eid).map(|entry| entry.tier)
    }

    pub fn any_host(&self) -> bool {
        (0..EXPERTS).any(|eid| self.placement(eid) == Some(Placement::HostMapped))
    }

    pub fn gate_up_ptrs(&self) -> &'a [usize] {
        let start = self.layer * EXPERTS;
        &self.lease.binding.tables.gate_up[start..start + EXPERTS]
    }

    pub fn down_ptrs(&self) -> &'a [usize] {
        let start = self.layer * EXPERTS;
        &self.lease.binding.tables.down[start..start + EXPERTS]
    }

    pub fn codec_table(&self) -> &'a [u8] {
        let start = self.layer * EXPERTS;
        &self.lease.binding.tables.codecs[start..start + EXPERTS]
    }

    pub fn gate_up_ptrs_device(&self) -> GpuTensor {
        self.pointer_table_view(self.lease.binding.gate_up_table_base)
    }

    pub fn down_ptrs_device(&self) -> GpuTensor {
        self.pointer_table_view(self.lease.binding.down_table_base)
    }

    pub fn codec_table_device(&self) -> GpuTensor {
        borrowed_table(
            self.lease.binding.codec_table_base,
            TOTAL_EXPERTS,
            DType::Raw,
        )
        .sub_offset(self.layer * EXPERTS, EXPERTS)
    }

    pub fn lease(&self) -> ExpertTierLease {
        self.lease.clone()
    }

    /// Forwards [`ExpertTierLease::owner_range`].
    pub fn owner_range(&self, tier: Placement) -> Option<(usize, usize)> {
        self.lease.owner_range(tier)
    }

    /// Forwards [`ExpertTierLease::host_cpu_range`].
    pub fn host_cpu_range(&self) -> Option<(usize, usize)> {
        self.lease.host_cpu_range()
    }

    fn pointer_table_view(&self, base: usize) -> GpuTensor {
        // Existing MoE pointer tensors use two F32 storage elements per u64
        // address. F32 is metadata only: kernels read absolute 64-bit pointers.
        borrowed_table(base, 2 * TOTAL_EXPERTS, DType::F32)
            .sub_offset(2 * self.layer * EXPERTS, 2 * EXPERTS)
    }
}

fn borrowed_table(base: usize, elements: usize, dtype: DType) -> GpuTensor {
    GpuTensor {
        // SAFETY: only frozen table addresses from a live lease reach here.
        // The returned descriptor is borrowed and must not outlive the lease.
        buf: unsafe {
            DeviceBuffer::from_raw(base as *mut std::ffi::c_void, elements * dtype.size())
        },
        shape: vec![elements],
        dtype,
    }
}

fn runtime_error(error: hip_bridge::HipError) -> ExpertTierOwnerError {
    ExpertTierOwnerError::Runtime(error.to_string())
}

fn pointer_bytes(entries: &[usize]) -> &[u8] {
    // SAFETY: usize has no padding or uninitialized bytes. build_tables rejects
    // non-64-bit platforms; the HIP host/device pointer ABI is native-endian.
    unsafe {
        std::slice::from_raw_parts(
            entries.as_ptr().cast::<u8>(),
            std::mem::size_of_val(entries),
        )
    }
}

fn copy_span<'src>(
    src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
    source_offset: u64,
    len: u64,
    dst_offset: u64,
    mut write_chunk: impl FnMut(u64, &[u8]) -> Result<(), ExpertTierOwnerError>,
) -> Result<(), ExpertTierOwnerError> {
    source_offset
        .checked_add(len)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    dst_offset
        .checked_add(len)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    let mut copied = 0;
    while copied < len {
        let n = (len - copied).min(COPY_CHUNK_BYTES);
        let bytes = src(source_offset + copied, n).map_err(ExpertTierOwnerError::Source)?;
        if bytes.len() as u64 != n {
            return Err(ExpertTierOwnerError::SourceLength {
                expected: n,
                got: bytes.len(),
            });
        }
        write_chunk(dst_offset + copied, bytes)?;
        copied += n;
    }
    Ok(())
}

impl ExpertTierOwners {
    fn allocate_device(&self, bytes: u64) -> Result<Option<DeviceBuffer>, ExpertTierOwnerError> {
        if bytes == 0 {
            return Ok(None);
        }
        let bytes = usize::try_from(bytes).map_err(|_| ExpertTierOwnerError::Overflow)?;
        self.hip.malloc(bytes).map(Some).map_err(runtime_error)
    }

    fn allocate_host(&mut self, bytes: u64) -> Result<(), ExpertTierOwnerError> {
        if bytes == 0 {
            return Ok(());
        }
        let bytes = usize::try_from(bytes).map_err(|_| ExpertTierOwnerError::Overflow)?;
        let cpu = self
            .hip
            .host_malloc(bytes, HIP_HOST_MALLOC_MAPPED)
            .map_err(runtime_error)?;
        // Record the owner immediately, so alias lookup failure also frees it.
        self.host = Some(HostPayload {
            cpu_address: cpu as usize,
            device_alias: 0,
            bytes,
        });
        let alias = self
            .hip
            .host_get_device_pointer(cpu, 0)
            .map_err(runtime_error)?;
        if alias.is_null() {
            return Err(ExpertTierOwnerError::Runtime(
                "hipHostGetDevicePointer returned null".into(),
            ));
        }
        self.host
            .as_mut()
            .expect("host allocated above")
            .device_alias = alias as usize as u64;
        Ok(())
    }

    fn copy_payload<'src>(
        &self,
        tier: Placement,
        source_offset: u64,
        len: u64,
        dst_offset: u64,
        src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
    ) -> Result<(), ExpertTierOwnerError> {
        checked_extent(tier, dst_offset, len, self.plan.owner_bytes(tier))?;
        copy_span(src, source_offset, len, dst_offset, |offset, chunk| {
            let offset = usize::try_from(offset).map_err(|_| ExpertTierOwnerError::Overflow)?;
            match tier {
                Placement::Device => {
                    let owner = self
                        .device
                        .as_ref()
                        .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                    self.hip
                        .memcpy_htod_offset(owner, offset, chunk)
                        .map_err(runtime_error)
                }
                Placement::HostMapped => {
                    let owner = self
                        .host
                        .as_ref()
                        .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                    // SAFETY: the entire span was checked against the owner;
                    // chunk offsets are inside it and the allocation is private
                    // until all payloads and frozen tables have been published.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            chunk.as_ptr(),
                            (owner.cpu_address as *mut u8).add(offset),
                            chunk.len(),
                        );
                    }
                    Ok(())
                }
            }
        })
    }

    fn zero_tail(&self, tier: Placement) -> Result<(), ExpertTierOwnerError> {
        let bytes = self.plan.owner_bytes(tier);
        if bytes == 0 {
            return Ok(());
        }
        let tail = tail_pad_bytes();
        let offset = bytes
            .checked_sub(tail)
            .ok_or(ExpertTierOwnerError::TailPad { tier })?;
        let offset = usize::try_from(offset).map_err(|_| ExpertTierOwnerError::Overflow)?;
        let tail = usize::try_from(tail).map_err(|_| ExpertTierOwnerError::Overflow)?;
        match tier {
            Placement::Device => {
                let owner = self
                    .device
                    .as_ref()
                    .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                self.hip
                    .memset(&owner.byte_view(offset, tail), 0, tail)
                    .map_err(runtime_error)
            }
            Placement::HostMapped => {
                let owner = self
                    .host
                    .as_ref()
                    .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                debug_assert_eq!(offset + tail, owner.bytes);
                // SAFETY: the checked tail range is inside the sole pinned
                // allocation and no consumer has access during materialization.
                unsafe {
                    std::ptr::write_bytes((owner.cpu_address as *mut u8).add(offset), 0, tail)
                };
                Ok(())
            }
        }
    }
}

impl Drop for ExpertTierOwners {
    fn drop(&mut self) {
        // A keepalive may be retired on another thread without a Gpu. HIP frees
        // must target the owning device. A failed bind cannot safely free here.
        let previous_device = match self.hip.current_device() {
            Ok(device) => device,
            Err(_) => return,
        };
        if self.hip.set_device(self.device_id).is_err() {
            return;
        }
        for table in [
            &mut self.gate_up_table,
            &mut self.down_table,
            &mut self.codec_table,
        ] {
            if let Some(table) = table.take() {
                let _ = self.hip.free(table.buf);
            }
        }
        if let Some(owner) = self.device.take() {
            let _ = self.hip.free(owner);
        }
        if let Some(owner) = self.host.take() {
            let _ = self
                .hip
                .host_free(owner.cpu_address as *mut std::ffi::c_void);
        }
        // Do not invalidate rdna-compute's thread-local device-binding cache.
        if previous_device != self.device_id {
            let _ = self.hip.set_device(previous_device);
        }
    }
}

/// Allocate, fill and freeze a validated placement plan. Source requests are
/// at most 8 MiB; no whole-model temporary or per-expert owner is created.
/// The source mapping need only remain valid until this call returns.
pub fn materialize<'src>(
    gpu: &mut Gpu,
    plan: Arc<ExpertTierPlan>,
    src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
) -> Result<ExpertTierLease, ExpertTierOwnerError> {
    gpu.bind_thread().map_err(runtime_error)?;
    // Gpu owns HipRuntime by value and exposes no shared handle. A second
    // dlopen of the same HIP library supplies a standalone lifetime-safe handle
    // without taking Gpu's pool or host registry ownership. No runtime hook is
    // required; a future shared-runtime accessor can avoid this load.
    let hip = Arc::new(HipRuntime::load().map_err(runtime_error)?);
    let mut owners = ExpertTierOwners {
        device: None,
        host: None,
        gate_up_table: None,
        down_table: None,
        codec_table: None,
        binding: None,
        plan,
        hip,
        device_id: gpu.device_id,
        generation: issue_generation(&GENERATION)?,
    };
    owners.device = owners.allocate_device(owners.plan.owner_bytes(Placement::Device))?;
    owners.allocate_host(owners.plan.owner_bytes(Placement::HostMapped))?;
    let tables = build_tables(
        &owners.plan,
        owners
            .device
            .as_ref()
            .map(|owner| owner.as_ptr() as usize as u64),
        owners.host.as_ref().map(|owner| owner.device_alias),
    )?;
    for tier in [Placement::Device, Placement::HostMapped] {
        owners.zero_tail(tier)?;
        for copy in owners.plan.copies(tier) {
            owners.copy_payload(
                tier,
                copy.src.offset as u64,
                copy.src.len as u64,
                copy.dst_offset,
                src,
            )?;
        }
    }
    for window in owners.plan.host_windows() {
        for (span, offset) in &window.spans {
            owners.copy_payload(
                Placement::HostMapped,
                span.offset as u64,
                span.len as u64,
                *offset,
                src,
            )?;
        }
    }
    // Install each allocation in the RAII owner before its upload can fail.
    owners.gate_up_table = Some(GpuTensor {
        buf: owners
            .hip
            .malloc(TOTAL_EXPERTS * 8)
            .map_err(runtime_error)?,
        shape: vec![2 * TOTAL_EXPERTS],
        dtype: DType::F32,
    });
    owners
        .hip
        .memcpy_htod(
            &owners.gate_up_table.as_ref().expect("allocated above").buf,
            pointer_bytes(&tables.gate_up),
        )
        .map_err(runtime_error)?;
    owners.down_table = Some(GpuTensor {
        buf: owners
            .hip
            .malloc(TOTAL_EXPERTS * 8)
            .map_err(runtime_error)?,
        shape: vec![2 * TOTAL_EXPERTS],
        dtype: DType::F32,
    });
    owners
        .hip
        .memcpy_htod(
            &owners.down_table.as_ref().expect("allocated above").buf,
            pointer_bytes(&tables.down),
        )
        .map_err(runtime_error)?;
    owners.codec_table = Some(GpuTensor {
        buf: owners.hip.malloc(TOTAL_EXPERTS).map_err(runtime_error)?,
        shape: vec![TOTAL_EXPERTS],
        dtype: DType::Raw,
    });
    owners
        .hip
        .memcpy_htod(
            &owners.codec_table.as_ref().expect("allocated above").buf,
            &tables.codecs,
        )
        .map_err(runtime_error)?;
    owners.hip.device_synchronize().map_err(runtime_error)?;
    let mut owner_ranges = [None; 2];
    for (tier, base) in [
        (
            Placement::Device,
            owners
                .device
                .as_ref()
                .map(|owner| owner.as_ptr() as usize as u64),
        ),
        (
            Placement::HostMapped,
            owners.host.as_ref().map(|owner| owner.device_alias),
        ),
    ] {
        let bytes = owners.plan.owner_bytes(tier);
        if bytes != 0 {
            let base = base.ok_or(ExpertTierOwnerError::MissingBase(tier))?;
            owner_ranges[tier_index(tier)] = Some((
                usize::try_from(base).map_err(|_| ExpertTierOwnerError::Overflow)?,
                usize::try_from(bytes).map_err(|_| ExpertTierOwnerError::Overflow)?,
            ));
        }
    }
    let binding = Arc::new(ExpertTierBinding {
        hotset_sidecar_sha256: std::sync::OnceLock::new(),
        owner_ranges,
        host_cpu_range: owners
            .host
            .as_ref()
            .map(|host| (host.cpu_address, host.bytes)),
        logical_digest: owners.plan.logical_digest(),
        plan: owners.plan.clone(),
        tables,
        gate_up_table_base: owners
            .gate_up_table
            .as_ref()
            .expect("allocated above")
            .buf
            .as_ptr() as usize,
        down_table_base: owners
            .down_table
            .as_ref()
            .expect("allocated above")
            .buf
            .as_ptr() as usize,
        codec_table_base: owners
            .codec_table
            .as_ref()
            .expect("allocated above")
            .buf
            .as_ptr() as usize,
        generation: owners.generation,
    });
    owners.binding = Some(binding.clone());
    Ok(ExpertTierLease {
        binding,
        owners: Arc::new(Mutex::new(owners)),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
    use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
    use crate::expert_slabs::ExpertFormat;
    use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, ExpertTierPlan, Placement};
    use std::sync::atomic::{AtomicU64, Ordering};

    const TOTAL: usize = 48 * 512;
    const GU: usize = 1000; // pads to 1024
    const DN: usize = 500; // pads to 512
    const PADDED: u64 = 1024 + 512;

    // 256-aligned, nonzero, far apart; host alias is deliberately NOT device + owner.
    const DEV_BASE: u64 = 0x0000_7000_0000_0000;
    const HOST_ALIAS: u64 = 0x0000_5000_1000_0100;

    /// Same arithmetic as the planner's private `tail_pad()`.
    fn tail() -> u64 {
        ((ARENA_TAIL_PAD_BYTES as u64).max(256) + 255) & !255
    }

    fn slabs_with(f: impl Fn(usize) -> (ExpertFormat, usize, usize)) -> Vec<ExpertSlab> {
        let mut cursor = 0usize;
        (0..TOTAL)
            .map(|i| {
                let (format, gu, dn) = f(i);
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: gu,
                };
                cursor += gu;
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: dn,
                };
                cursor += dn;
                ExpertSlab {
                    format,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    fn uniform() -> Vec<ExpertSlab> {
        slabs_with(|_| (ExpertFormat::Mq4Xts, GU, DN))
    }

    fn format_cycle() -> Vec<ExpertSlab> {
        slabs_with(|i| match i % 3 {
            0 => (ExpertFormat::Mq4Xts, GU, DN),
            1 => (ExpertFormat::Mq3I4, GU, DN),
            _ => (ExpertFormat::Mq2I4, GU, DN),
        })
    }

    fn all_host_plan() -> ExpertTierPlan {
        let plan = plan_expert_tiers_from_slabs(&uniform(), &[], 0).unwrap();
        assert_eq!(plan.owner_bytes(Placement::Device), 0);
        assert!(plan.owner_bytes(Placement::HostMapped) > 0);
        plan
    }

    fn all_device_plan() -> ExpertTierPlan {
        let rank: Vec<u32> = (0..TOTAL as u32).collect();
        let plan = plan_expert_tiers_from_slabs(&uniform(), &rank, u64::MAX).unwrap();
        assert_eq!(plan.device_experts(), TOTAL);
        assert!(plan.owner_bytes(Placement::Device) > 0);
        assert_eq!(plan.owner_bytes(Placement::HostMapped), 0);
        plan
    }

    fn mixed_plan() -> ExpertTierPlan {
        let rank = [0u32, 5, 600, 24_575];
        let budget = rank.len() as u64 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&format_cycle(), &rank, budget).unwrap();
        assert_eq!(plan.device_experts(), 4);
        assert!(plan.owner_bytes(Placement::Device) > 0);
        assert!(plan.owner_bytes(Placement::HostMapped) > 0);
        plan
    }

    fn err<T>(r: Result<T, ExpertTierOwnerError>) -> ExpertTierOwnerError {
        match r {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        }
    }

    /// Verify every pointer of every one of the 24,576 experts against the plan.
    fn assert_tables(
        plan: &ExpertTierPlan,
        t: &Tables,
        device_base: Option<u64>,
        host_alias: Option<u64>,
    ) {
        assert_eq!(t.gate_up.len(), TOTAL);
        assert_eq!(t.down.len(), TOTAL);
        assert_eq!(t.codecs.len(), TOTAL);
        for layer in 0..48usize {
            for expert in 0..512usize {
                let i = layer * 512 + expert;
                let e = plan.entry(layer, expert).unwrap();
                let (base, owner) = match e.tier {
                    Placement::Device => {
                        (device_base.unwrap(), plan.owner_bytes(Placement::Device))
                    }
                    Placement::HostMapped => {
                        (host_alias.unwrap(), plan.owner_bytes(Placement::HostMapped))
                    }
                };
                let gu = t.gate_up[i] as u64;
                let dn = t.down[i] as u64;
                assert_eq!(gu, base + e.gate_up_offset, "gate_up {layer}/{expert}");
                assert_eq!(dn, base + e.down_offset, "down {layer}/{expert}");
                assert_eq!(gu % 256, 0, "gate_up align {layer}/{expert}");
                assert_eq!(dn % 256, 0, "down align {layer}/{expert}");
                // Inside the owner, leaving the zeroed tail pad after the payload.
                assert!(gu >= base && gu + e.gate_up_len <= base + owner - tail());
                assert!(dn >= base && dn + e.down_len <= base + owner - tail());
                // Projections of one expert never overlap.
                assert!(gu + e.gate_up_len <= dn || dn + e.down_len <= gu);
                let want = match e.format {
                    ExpertFormat::Mq4Xts => 0u8,
                    ExpertFormat::Mq3I4 => 1u8,
                    ExpertFormat::Mq2I4 => 2u8,
                    ExpertFormat::E8pI4 => 3u8,
                };
                assert_eq!(t.codecs[i], want, "codec {layer}/{expert}");
            }
        }
    }

    // ---- placement mixes -------------------------------------------------

    #[test]
    fn all_host_plan_needs_only_host_alias() {
        let plan = all_host_plan();
        let t = build_tables(&plan, None, Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, None, Some(HOST_ALIAS));
        for i in 0..TOTAL {
            assert!(t.gate_up[i] as u64 >= HOST_ALIAS);
            assert!(t.down[i] as u64 >= HOST_ALIAS);
        }
    }

    #[test]
    fn all_device_plan_needs_only_device_base() {
        let plan = all_device_plan();
        let t = build_tables(&plan, Some(DEV_BASE), None).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), None);
        for i in 0..TOTAL {
            assert!(t.gate_up[i] as u64 >= DEV_BASE);
            assert!(t.down[i] as u64 >= DEV_BASE);
        }
    }

    #[test]
    fn mixed_plan_fills_all_24576_pointers_from_the_right_owner() {
        let plan = mixed_plan();
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), Some(HOST_ALIAS));

        let dev_end = DEV_BASE + plan.owner_bytes(Placement::Device);
        let host_end = HOST_ALIAS + plan.owner_bytes(Placement::HostMapped);
        let mut dev = 0usize;
        let mut host = 0usize;
        for i in 0..TOTAL {
            let e = plan.entry(i / 512, i % 512).unwrap();
            let p = t.gate_up[i] as u64;
            match e.tier {
                Placement::Device => {
                    dev += 1;
                    assert!((DEV_BASE..dev_end).contains(&p));
                }
                Placement::HostMapped => {
                    host += 1;
                    assert!((HOST_ALIAS..host_end).contains(&p));
                }
            }
        }
        assert_eq!(dev, 4);
        assert_eq!(dev + host, TOTAL);
        // The four device experts are exactly the ranked ids.
        for id in [0usize, 5, 600, 24_575] {
            assert_eq!(
                plan.entry(id / 512, id % 512).unwrap().tier,
                Placement::Device
            );
        }
    }

    #[test]
    fn host_pointers_use_the_distinct_alias_not_the_device_base() {
        let plan = mixed_plan();
        let a = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        let other_alias = HOST_ALIAS + 0x10_0000;
        let b = build_tables(&plan, Some(DEV_BASE), Some(other_alias)).unwrap();
        for i in 0..TOTAL {
            let e = plan.entry(i / 512, i % 512).unwrap();
            match e.tier {
                Placement::Device => {
                    // Device pointers are independent of the host alias.
                    assert_eq!(a.gate_up[i], b.gate_up[i]);
                    assert_eq!(a.down[i], b.down[i]);
                }
                Placement::HostMapped => {
                    assert_eq!(b.gate_up[i] - a.gate_up[i], 0x10_0000);
                    assert_eq!(b.down[i] - a.down[i], 0x10_0000);
                    assert_eq!(a.gate_up[i] as u64, HOST_ALIAS + e.gate_up_offset);
                }
            }
        }
        // Changing the device base never moves host pointers.
        let c = build_tables(&plan, Some(DEV_BASE + 0x1000), Some(HOST_ALIAS)).unwrap();
        for i in 0..TOTAL {
            let e = plan.entry(i / 512, i % 512).unwrap();
            if e.tier == Placement::HostMapped {
                assert_eq!(a.gate_up[i], c.gate_up[i]);
                assert_eq!(a.down[i], c.down[i]);
            } else {
                assert_eq!(c.gate_up[i] - a.gate_up[i], 0x1000);
            }
        }
    }

    #[test]
    fn codec_bytes_are_mq4_0_w3_1_w2_2() {
        let plan = mixed_plan();
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_eq!(t.codecs[0], 0); // i % 3 == 0 -> Mq4Xts
        assert_eq!(t.codecs[1], 1); // Mq3I4
        assert_eq!(t.codecs[2], 2); // Mq2I4
        assert_eq!(t.codecs[TOTAL - 1], ((TOTAL - 1) % 3) as u8);
        let mut counts = [0usize; 3];
        for &c in &t.codecs {
            counts[c as usize] += 1;
        }
        assert_eq!(counts, [TOTAL / 3; 3]);
    }

    // ---- base presence ---------------------------------------------------

    #[test]
    fn empty_tier_needs_no_base() {
        // All-host: no device owner, so no device base is required.
        let t = build_tables(&all_host_plan(), None, Some(HOST_ALIAS));
        assert!(t.is_ok());
        // All-device: no host owner, so no host alias is required.
        let t = build_tables(&all_device_plan(), Some(DEV_BASE), None);
        assert!(t.is_ok());
    }

    #[test]
    fn populated_tier_without_base_is_rejected() {
        assert!(matches!(
            err(build_tables(&all_host_plan(), None, None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
        assert!(matches!(
            err(build_tables(&all_host_plan(), Some(DEV_BASE), None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
        assert!(matches!(
            err(build_tables(&all_device_plan(), None, None)),
            ExpertTierOwnerError::MissingBase(Placement::Device)
        ));
        assert!(matches!(
            err(build_tables(&all_device_plan(), None, Some(HOST_ALIAS))),
            ExpertTierOwnerError::MissingBase(Placement::Device)
        ));
        let mixed = mixed_plan();
        assert!(matches!(
            err(build_tables(&mixed, None, Some(HOST_ALIAS))),
            ExpertTierOwnerError::MissingBase(Placement::Device)
        ));
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE), None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
    }

    // ---- alignment / address overflow -------------------------------------

    #[test]
    fn misaligned_base_is_rejected() {
        for bad in [DEV_BASE + 1, DEV_BASE + 128, DEV_BASE + 255] {
            match err(build_tables(&all_device_plan(), Some(bad), None)) {
                ExpertTierOwnerError::Alignment { address } => assert_ne!(address % 256, 0),
                other => panic!("device base {bad:#x}: {other:?}"),
            }
        }
        for bad in [HOST_ALIAS + 1, HOST_ALIAS + 64] {
            match err(build_tables(&all_host_plan(), None, Some(bad))) {
                ExpertTierOwnerError::Alignment { address } => assert_ne!(address % 256, 0),
                other => panic!("host alias {bad:#x}: {other:?}"),
            }
        }
        // Mixed plan: either misaligned base fails even when the other is fine.
        let mixed = mixed_plan();
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE + 4), Some(HOST_ALIAS))),
            ExpertTierOwnerError::Alignment { .. }
        ));
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE), Some(HOST_ALIAS + 4))),
            ExpertTierOwnerError::Alignment { .. }
        ));
    }

    #[test]
    fn aligned_nonzero_base_at_page_and_256_boundaries_is_accepted() {
        for base in [256u64, 0x1000, 0x20_0000, DEV_BASE] {
            let t = build_tables(&all_device_plan(), Some(base), None).unwrap();
            assert_eq!(t.gate_up[0] as u64, base);
        }
    }

    #[test]
    fn address_overflow_is_rejected() {
        // Highest 256-aligned address: base + any nonzero offset or owner size wraps.
        let top = u64::MAX & !255;
        assert!(matches!(
            err(build_tables(&all_device_plan(), Some(top), None)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(build_tables(&all_host_plan(), None, Some(top))),
            ExpertTierOwnerError::Overflow
        ));
        // Base fits but base + owner_bytes does not.
        let plan = all_device_plan();
        let near = (u64::MAX - plan.owner_bytes(Placement::Device) / 2) & !255;
        assert!(matches!(
            err(build_tables(&plan, Some(near), None)),
            ExpertTierOwnerError::Overflow
        ));
        // Mixed: overflow on either side.
        let mixed = mixed_plan();
        assert!(matches!(
            err(build_tables(&mixed, Some(top), Some(HOST_ALIAS))),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE), Some(top))),
            ExpertTierOwnerError::Overflow
        ));
    }

    // ---- checked_extent ----------------------------------------------------

    #[test]
    fn checked_extent_accepts_in_bounds_and_exact_end() {
        assert!(checked_extent(Placement::Device, 0, 1, 4096).is_ok());
        assert!(checked_extent(Placement::Device, 0, 4096, 4096).is_ok());
        assert!(checked_extent(Placement::HostMapped, 4095, 1, 4096).is_ok());
        assert!(checked_extent(Placement::HostMapped, 256, 3840, 4096).is_ok());
        // > 4 GiB offsets stay exact.
        let big = 12u64 << 30;
        assert!(checked_extent(Placement::HostMapped, big, 4096, big + 8192).is_ok());
    }

    #[test]
    fn checked_extent_rejects_out_of_bounds() {
        match err(checked_extent(Placement::Device, 1, 4096, 4096)) {
            ExpertTierOwnerError::Extent {
                tier,
                offset,
                len,
                owner_bytes,
            } => {
                assert_eq!(tier, Placement::Device);
                assert_eq!((offset, len, owner_bytes), (1, 4096, 4096));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            err(checked_extent(Placement::HostMapped, 0, 4097, 4096)),
            ExpertTierOwnerError::Extent {
                tier: Placement::HostMapped,
                ..
            }
        ));
        // Starts at the owner end / beyond it.
        assert!(matches!(
            err(checked_extent(Placement::Device, 4096, 1, 4096)),
            ExpertTierOwnerError::Extent { .. }
        ));
        assert!(matches!(
            err(checked_extent(Placement::Device, 9000, 1, 4096)),
            ExpertTierOwnerError::Extent { .. }
        ));
        // Empty owner holds nothing.
        assert!(matches!(
            err(checked_extent(Placement::Device, 0, 1, 0)),
            ExpertTierOwnerError::Extent { owner_bytes: 0, .. }
        ));
    }

    #[test]
    fn checked_extent_rejects_zero_len() {
        for owner in [0u64, 4096] {
            match err(checked_extent(Placement::Device, 0, 0, owner)) {
                ExpertTierOwnerError::Extent { len, .. } => assert_eq!(len, 0),
                other => panic!("owner {owner}: {other:?}"),
            }
        }
        assert!(matches!(
            err(checked_extent(Placement::HostMapped, 128, 0, 4096)),
            ExpertTierOwnerError::Extent { len: 0, .. }
        ));
    }

    #[test]
    fn checked_extent_rejects_arithmetic_overflow() {
        assert!(matches!(
            err(checked_extent(Placement::Device, u64::MAX, 1, u64::MAX)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(checked_extent(Placement::Device, u64::MAX - 3, 4, u64::MAX)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(checked_extent(Placement::HostMapped, 1, u64::MAX, u64::MAX)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(checked_extent(
                Placement::HostMapped,
                u64::MAX,
                u64::MAX,
                u64::MAX
            )),
            ExpertTierOwnerError::Overflow
        ));
    }

    // ---- host windows ------------------------------------------------------

    fn mtp_spans() -> [ExpertSlabSpan; 2] {
        [
            ExpertSlabSpan {
                offset: 5_000_000_000,
                len: 1000,
            },
            ExpertSlabSpan {
                offset: 6_000_000_000,
                len: 513,
            },
        ]
    }

    #[test]
    fn host_window_on_mixed_plan_is_validated_with_tail_present() {
        let plan = mixed_plan().with_host_window("mtp", &mtp_spans()).unwrap();
        let owner = plan.owner_bytes(Placement::HostMapped);
        // Planner contract the validator relies on: every window extent is
        // inside the host owner and leaves the full tail pad after it.
        let w = &plan.host_windows()[0];
        assert_eq!(w.spans.len(), 2);
        for (span, dst) in &w.spans {
            assert_eq!(dst % 256, 0);
            assert!(checked_extent(Placement::HostMapped, *dst, span.len as u64, owner).is_ok());
            assert!(dst + span.len as u64 + tail() <= owner);
        }
        assert_eq!(owner % 256, 0);
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), Some(HOST_ALIAS));
    }

    #[test]
    fn host_window_on_all_host_plan_does_not_need_device_base() {
        let plan = all_host_plan()
            .with_host_window("mtp", &mtp_spans())
            .unwrap();
        let t = build_tables(&plan, None, Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, None, Some(HOST_ALIAS));
        assert!(matches!(
            err(build_tables(&plan, None, None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
    }

    #[test]
    fn host_window_on_all_device_plan_requires_host_alias() {
        let device = all_device_plan();
        assert_eq!(device.owner_bytes(Placement::HostMapped), 0);
        // Without a window the host alias is not needed.
        assert!(build_tables(&device, Some(DEV_BASE), None).is_ok());

        let plan = device.with_host_window("mtp", &mtp_spans()).unwrap();
        let host = plan.owner_bytes(Placement::HostMapped);
        // The window allocates the previously empty host tier: one tail pad.
        assert_eq!(host, 1024 + 768 + tail());
        assert!(matches!(
            err(build_tables(&plan, Some(DEV_BASE), None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), Some(HOST_ALIAS));
        // Window start must also respect alias alignment and overflow.
        assert!(matches!(
            err(build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS + 8))),
            ExpertTierOwnerError::Alignment { .. }
        ));
        assert!(matches!(
            err(build_tables(&plan, Some(DEV_BASE), Some(u64::MAX & !255))),
            ExpertTierOwnerError::Overflow
        ));
    }

    // ---- generation counter ----------------------------------------------

    #[test]
    fn generation_is_monotonic_from_one() {
        let c = AtomicU64::new(0);
        let mut prev = 0u64;
        for want in 1..=64u64 {
            let g = issue_generation(&c).unwrap();
            assert_eq!(g, want);
            assert!(g > prev);
            prev = g;
        }
        assert_eq!(c.load(Ordering::SeqCst), 64);
    }

    #[test]
    fn generation_accepts_u64_max_then_exhausts_without_modifying() {
        let c = AtomicU64::new(u64::MAX - 2);
        assert_eq!(issue_generation(&c).unwrap(), u64::MAX - 1);
        assert_eq!(issue_generation(&c).unwrap(), u64::MAX);
        assert_eq!(c.load(Ordering::SeqCst), u64::MAX);
        for _ in 0..3 {
            assert!(matches!(
                err(issue_generation(&c)),
                ExpertTierOwnerError::GenerationExhausted
            ));
            assert_eq!(c.load(Ordering::SeqCst), u64::MAX);
        }
    }

    #[test]
    fn generation_exhausted_counter_stays_exhausted_from_max_directly() {
        let c = AtomicU64::new(u64::MAX);
        assert!(matches!(
            err(issue_generation(&c)),
            ExpertTierOwnerError::GenerationExhausted
        ));
        assert_eq!(c.load(Ordering::SeqCst), u64::MAX);
    }

    #[test]
    fn process_generation_static_is_strictly_increasing() {
        // Other tests may issue concurrently, so only strict increase is asserted.
        let a = issue_generation(&GENERATION).unwrap();
        let b = issue_generation(&GENERATION).unwrap();
        assert!(a >= 1);
        assert!(b > a);
    }

    #[test]
    fn source_copies_are_bounded_and_keep_offsets() {
        let chunk = vec![0xa5; COPY_CHUNK_BYTES as usize];
        let requests = std::cell::RefCell::new(Vec::new());
        let source = |offset, len| {
            requests.borrow_mut().push((offset, len));
            Ok(&chunk[..len as usize])
        };
        let mut writes = Vec::new();
        copy_span(
            &source,
            123,
            COPY_CHUNK_BYTES * 2 + 17,
            256,
            |offset, bytes| {
                assert!(bytes.iter().all(|byte| *byte == 0xa5));
                writes.push((offset, bytes.len() as u64));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            *requests.borrow(),
            vec![
                (123, COPY_CHUNK_BYTES),
                (123 + COPY_CHUNK_BYTES, COPY_CHUNK_BYTES),
                (123 + 2 * COPY_CHUNK_BYTES, 17)
            ]
        );
        assert_eq!(
            writes,
            vec![
                (256, COPY_CHUNK_BYTES),
                (256 + COPY_CHUNK_BYTES, COPY_CHUNK_BYTES),
                (256 + 2 * COPY_CHUNK_BYTES, 17)
            ]
        );
    }

    #[test]
    fn pointer_mirrors_are_the_exact_upload_bytes() {
        let tables = build_tables(&mixed_plan(), Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        for entries in [&tables.gate_up, &tables.down] {
            let bytes = pointer_bytes(entries);
            assert_eq!(bytes.len(), TOTAL * 8);
            for (entry, bytes) in entries.iter().zip(bytes.chunks_exact(8)) {
                assert_eq!(bytes, entry.to_ne_bytes());
            }
        }
    }
}
