// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `memory.offload_exec=igpu`: whole host-placed MoE layers of an eager Flash-Next
//! prefill wider than four rows run on a second, integrated gfx1151 device.
//!
//! This module owns the architecture side only: the pure load admission, the
//! second-device choice, the arming of portable host-mapped allocation (which
//! must precede the expert-byte allocation), the per-layer host-owner leases the
//! engine borrows, and the single attach step. The engine itself is
//! [`hipfire_dispatch::igpu_exec`].
//!
//! Every admission failure is one log line and the NORMAL route: nothing here
//! refuses the load and nothing refuses a request. A layer the engine cannot
//! take (not host-placed whole, asymmetric headers, not an owning allocation)
//! simply keeps the GPU grouped route; only layers that bound successfully are
//! reported, engined and part of the cache manifest.

use super::*;
use hipfire_dispatch::igpu_exec::{
    self, IgpuArmInputs, IgpuEngine, IgpuEngineSpec, IgpuLayerEngine, IgpuLayerWeights,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// Architecture-supplied facts of the load that the arming decision needs
/// before any expert byte is allocated. Every field is `true` when the load
/// uses that feature (a refusal), except [`Self::host_whole_layers_only`],
/// which is `true` when every host-placed expert stack is a whole layer (the
/// only placement the engine admits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4IgpuLoadInputs {
    pub ep: bool,
    pub pp: bool,
    pub slab: bool,
    pub tier: bool,
    /// A device-resident hot-set expert placement.
    pub hotset: bool,
    pub awq: bool,
    pub nonuniform_experts: bool,
    pub host_whole_layers_only: bool,
}

impl Qwen4IgpuLoadInputs {
    /// A plain single-device, whole-layer, uniform QT44/QT53 load.
    pub const PLAIN: Self = Self {
        ep: false,
        pp: false,
        slab: false,
        tier: false,
        hotset: false,
        awq: false,
        nonuniform_experts: false,
        host_whole_layers_only: true,
    };
}

/// Whether `memory.offload_exec=igpu` is selected for this process.
pub fn qwen4_igpu_enabled() -> bool {
    hipfire_dispatch::cpu_exec::igpu_exec_enabled()
}

/// Set once [`arm_igpu_for_load`] has run (armed or refused, either way it
/// logged its verdict), so the attach step does not repeat the reason.
static ARM_REPORTED: AtomicBool = AtomicBool::new(false);

/// Normalise a `gcnArchName` (`gfx1151:sramecc+:xnack-`) to its bare arch.
fn bare_arch(raw: &str) -> &str {
    raw.split(':').next().unwrap_or(raw)
}

/// Pure second-device choice over the visible devices' arch names.
///
/// `override_index` (`HIPFIRE_IGPU_DEVICE`) must name a visible device; without
/// an override exactly one non-primary gfx1151 device must exist (none, or
/// several, is refused rather than guessed).
pub fn select_second_device(
    primary: i32,
    override_index: Option<i32>,
    archs: &[String],
) -> Result<i32, String> {
    if let Some(index) = override_index {
        if index < 0 || index as usize >= archs.len() {
            return Err(format!(
                "HIPFIRE_IGPU_DEVICE={index} is not a visible device ({} visible)",
                archs.len()
            ));
        }
        return Ok(index);
    }
    let mut found = archs
        .iter()
        .enumerate()
        .filter(|(index, arch)| *index as i32 != primary && bare_arch(arch) == "gfx1151")
        .map(|(index, _)| index as i32);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        (None, _) => Err(
            "no second gfx1151 device is visible (set HIPFIRE_IGPU_DEVICE to select the iGPU)"
                .to_string(),
        ),
        (Some(_), Some(_)) => Err(
            "several non-primary gfx1151 devices are visible (set HIPFIRE_IGPU_DEVICE to select the iGPU)"
                .to_string(),
        ),
    }
}

/// The visible devices' arch names, index = HIP device ordinal.
fn visible_archs(gpu: &Gpu) -> Result<Vec<String>, String> {
    let count = gpu
        .hip
        .device_count()
        .map_err(|error| format!("device enumeration failed: {error}"))?;
    (0..count)
        .map(|index| {
            gpu.hip
                .get_arch(index)
                .map(|arch| bare_arch(&arch).to_string())
                .map_err(|error| format!("device {index} arch query failed: {error}"))
        })
        .collect()
}

/// The pure refusal: every condition that makes an `igpu` load illegal, as the
/// one line the engine falls back with. `emu` is the primary's `HIPFIRE_EMU`
/// state. `Err` is a refusal; the caller keeps the normal route.
pub fn qwen4_igpu_refusal(
    primary_arch: &str,
    primary_device: i32,
    second_arch: &str,
    second_device: i32,
    emu: bool,
    inputs: &Qwen4IgpuLoadInputs,
) -> Result<(), String> {
    igpu_exec::igpu_refusal(&IgpuArmInputs {
        requested: true,
        primary_arch,
        second_arch,
        primary_device,
        second_device,
        emu,
        ep: inputs.ep,
        pp: inputs.pp,
        slab: inputs.slab,
        tier: inputs.tier,
        hotset: inputs.hotset,
        awq: inputs.awq,
        nonuniform_experts: inputs.nonuniform_experts,
        host_whole_layers_only: inputs.host_whole_layers_only,
    })
}

/// What [`arm_igpu_for_load`] decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4IgpuArm {
    /// HIP ordinal of the integrated device the engine runs on.
    pub device: i32,
    pub arch: String,
}

/// Resolve the second device (override or autodetect) and run the refusal over
/// it. Read-only on `gpu`. `Err` is the one-line reason to fall back.
pub fn resolve_igpu_device(
    gpu: &Gpu,
    inputs: &Qwen4IgpuLoadInputs,
) -> Result<Qwen4IgpuArm, String> {
    let archs = visible_archs(gpu)?;
    let device = select_second_device(gpu.device_id, igpu_exec::igpu_device_index(), &archs)?;
    let arch = archs[device as usize].clone();
    qwen4_igpu_refusal(
        &gpu.arch,
        gpu.device_id,
        &arch,
        device,
        gpu.emu_profile().is_some(),
        inputs,
    )?;
    Ok(Qwen4IgpuArm { device, arch })
}

/// The one-line fallback notice.
fn fallback_line(why: &str) -> String {
    format!("  [igpu-exec] qwen4: off — {why}; using the normal route")
}

/// Load-time arming. Call once, on the model `Gpu`, **before** any expert byte
/// is allocated (`place_routed_experts` / `fulfill_manifest*`): from then on the
/// host-mapped allocations are `hipHostMallocMapped | hipHostMallocPortable` so
/// the second device can map them. Returns `None` and touches nothing unless
/// `memory.offload_exec=igpu`, so the default allocation flags and logs are
/// untouched. A refusal is logged as one line and returns `None`: the load goes
/// on with the normal route and `attach_forward` then attaches no engine.
pub fn arm_igpu_for_load(gpu: &mut Gpu, inputs: &Qwen4IgpuLoadInputs) -> Option<Qwen4IgpuArm> {
    if !qwen4_igpu_enabled() {
        return None;
    }
    ARM_REPORTED.store(true, Ordering::Release);
    match resolve_igpu_device(gpu, inputs) {
        Ok(arm) => {
            gpu.arm_portable_host_mapped();
            Some(arm)
        }
        Err(why) => {
            // An armed Gpu reused by a refused load must not attach later.
            gpu.disarm_portable_host_mapped();
            eprintln!("{}", fallback_line(&why));
            None
        }
    }
}

/// Loader-side refusal (non-Flash-Next profile, dense FFN, no host-placed trunk
/// layer, ...): log the ONE fallback line and mark the verdict reported so the
/// attach step does not repeat it. The load continues on the normal route.
pub fn decline_igpu_for_load(why: &str) {
    ARM_REPORTED.store(true, Ordering::Release);
    eprintln!("{}", fallback_line(why));
}

/// A layer's two host aggregates as the engine borrows them: views into ONE
/// pooled MAPPED|PORTABLE owner (`Gpu::reserve_portable_host_pool`), each with
/// the shared lease on that owner. While any clone of the lease lives — this
/// record's or the engine's — the owning `Gpu` refuses to `hipHostFree` the
/// backing, so the host bytes cannot be freed under an engine that could not be
/// drained. The record never frees and never holds a raw pointer.
pub(super) struct IgpuHostOwners {
    gate_up: rdna_compute::HostMappedView,
    down: rdna_compute::HostMappedView,
    gate_up_stride: usize,
    down_stride: usize,
    keepalive: Arc<(rdna_compute::HostMappedView, rdna_compute::HostMappedView)>,
}

impl IgpuHostOwners {
    /// `None` unless both aggregates are views of ONE portable owner (the pool)
    /// and hold `experts` extents of the given strides.
    pub(super) fn new(
        gpu: &mut Gpu,
        gate_up: &GpuTensor,
        down: &GpuTensor,
        experts: usize,
        gate_up_stride: usize,
        down_stride: usize,
    ) -> Option<Self> {
        if experts == 0 || gate_up_stride == 0 || down_stride == 0 {
            return None;
        }
        if experts.checked_mul(gate_up_stride)? > gpu.host_bytes(gate_up)?.len()
            || experts.checked_mul(down_stride)? > gpu.host_bytes(down)?.len()
        {
            return None;
        }
        let gate_up_view = gpu.host_mapped_view(gate_up).ok()?;
        let down_view = gpu.host_mapped_view(down).ok()?;
        if !gate_up_view.same_owner(&down_view) {
            return None;
        }
        Some(Self {
            keepalive: Arc::new((gate_up_view.clone(), down_view.clone())),
            gate_up: gate_up_view,
            down: down_view,
            gate_up_stride,
            down_stride,
        })
    }

    /// `(owner key, owner bytes)` of the pooled backing both views live in.
    fn owner(&self) -> (usize, usize) {
        (self.gate_up.lease().key(), self.gate_up.owner_len())
    }

    fn weights(&self) -> IgpuLayerWeights {
        IgpuLayerWeights {
            owner_base: self.gate_up.owner_host_base().cast(),
            owner_len: self.gate_up.owner_len(),
            gate_up_offset: self.gate_up.offset(),
            gate_up_stride: self.gate_up_stride,
            down_offset: self.down.offset(),
            down_stride: self.down_stride,
            lease: Some(self.keepalive.clone()),
        }
    }
}

/// Per-layer admission over what the built runtimes say. Pure; `refusal` is the
/// single decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct IgpuLayerAdmission {
    /// The layer's routed aggregates are host-mapped.
    pub host_mapped: bool,
    /// Host owners resolved (both aggregates are portable owners with full extents).
    pub owners: bool,
    pub ep: bool,
    pub slab: bool,
    pub tier: bool,
    pub awq: bool,
    pub fp8: bool,
    /// Every expert is the QT44 gate/up + QT53 down pair.
    pub uniform_qt44_qt53: bool,
}

impl IgpuLayerAdmission {
    /// The layer is host-placed (a candidate for the engine at all). A
    /// VRAM-resident layer is not an engine layer and keeps the GPU route.
    pub(super) fn host_placed(&self) -> bool {
        self.host_mapped
    }

    /// One-line reason a host-placed layer keeps the normal route.
    pub(super) fn refusal(&self) -> Option<&'static str> {
        if self.ep {
            Some("expert parallelism")
        } else if self.slab {
            Some("expert-slab layer")
        } else if self.tier {
            Some("expert-tier or hot-set layer")
        } else if self.fp8 {
            Some("stock FP8 experts")
        } else if self.awq {
            Some("AWQ-scaled experts")
        } else if !self.uniform_qt44_qt53 {
            Some("non-uniform expert formats (QT44 gate/up + QT53 down only)")
        } else if !self.owners {
            Some("host aggregates are not both views of one portable pooled backing")
        } else {
            None
        }
    }
}

fn layer_admission(layer: &Qwen4MoeLayerRuntime, ep: bool) -> IgpuLayerAdmission {
    IgpuLayerAdmission {
        host_mapped: layer.host_mapped,
        owners: layer.igpu_host.is_some(),
        ep,
        slab: layer.slab.is_some(),
        tier: layer.tier.is_some(),
        awq: layer.experts.iter().any(|expert| {
            expert.gate_up.dispatch_ref().awq_scale.is_some()
                || expert.down.dispatch_ref().awq_scale.is_some()
        }),
        fp8: layer.fp8,
        uniform_qt44_qt53: !layer.fp8 && layer.slab.is_none() && layer.qt44_qt53_formats(),
    }
}

fn fallback(why: &str) -> Result<usize, Qwen4GpuForwardError> {
    eprintln!("{}", fallback_line(why));
    Ok(0)
}

/// Attach the engine to the host-placed trunk layers that can take it, at load
/// (inside `Qwen4GpuForward::new_inner`, after the layers are built and before
/// any request). Returns the number of layers that BOUND successfully.
///
/// Anything that cannot be admitted — profile, FP8, capture/recording/shadow
/// replay, no armed load, no second device, a layer whose host owners or
/// symmetric check fail — is one log line and the normal GPU route for that
/// layer (or all layers); it never fails the load. Only successfully bound layers
/// are engined, logged and part of the cache manifest.
pub(super) fn attach_igpu(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
    moe: &mut [Qwen4MoeLayerRuntime],
    ep: bool,
    stock_fp8: bool,
    max_chunk: usize,
) -> Result<usize, Qwen4GpuForwardError> {
    let config = &bundle.config;
    if !config.profile.is_flash_next() {
        return fallback(&format!(
            "profile {} is not Flash-Next (the engine serves the Flash-Next MoE prefill only)",
            config.profile.id()
        ));
    }
    if stock_fp8 {
        return fallback("stock FP8 experts are not served");
    }
    if ep {
        return fallback("expert parallelism");
    }
    if gpu.arch == "gfx1151"
        && (rdna_compute::prefill_invariant::prefill_pm_invariant_enabled()
            || rdna_compute::prefill_invariant::qwen4_append_wmma_enabled())
    {
        return fallback(
            "the batch-invariant / append prefill profiles pin canonical routing rows, which the \
             engine does not accept",
        );
    }
    // Mirror of the CPU splice's attachment gates: the engine is a cross-device
    // host-synchronised step, absent from any retained tape.
    let stream_capture = match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.stream_is_capturing(stream)?,
        None => false,
    };
    if gpu.graphs.capture_mode || gpu.replay.is_recording() || stream_capture {
        return fallback("graph capture/recording is active (set replay.backend=hip)");
    }
    if gpu.replay.request() == rdna_compute::replay::ReplayBackendRequest::Shadow {
        return fallback("replay.backend=shadow cannot retain the cross-device step");
    }
    if !gpu.host_mapped_portable() {
        // `arm_igpu_for_load` already logged why it refused; only a load that
        // never armed needs its own line.
        if ARM_REPORTED.load(Ordering::Acquire) {
            return Ok(0);
        }
        return fallback("the load did not arm portable host-mapped experts (arm_igpu_for_load)");
    }
    let mut candidates = Vec::new();
    let mut skipped: Vec<(usize, String)> = Vec::new();
    for (index, layer) in moe.iter().enumerate() {
        let admission = layer_admission(layer, ep);
        if !admission.host_placed() {
            continue;
        }
        match admission.refusal() {
            Some(why) => skipped.push((layer.label, why.to_string())),
            None => candidates.push(index),
        }
    }
    if candidates.is_empty() {
        return fallback(&match skipped.first() {
            Some((label, why)) => format!("no host-placed trunk layer is admissible (layer {label}: {why})"),
            None => "no host-placed trunk layer to run on the iGPU".to_string(),
        });
    }
    let arm = match resolve_igpu_device(
        gpu,
        &Qwen4IgpuLoadInputs {
            ep,
            ..Qwen4IgpuLoadInputs::PLAIN
        },
    ) {
        Ok(arm) => arm,
        Err(why) => return fallback(&why),
    };
    let engine = match IgpuEngine::open(IgpuEngineSpec {
        device: arm.device,
        max_rows: max_chunk,
        hidden: config.hidden_size,
        moe_intermediate: config.moe_intermediate_size,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        layers: candidates.len(),
    }) {
        Ok(engine) => engine,
        Err(why) => return fallback(&why),
    };
    let mut bound = 0usize;
    for &index in &candidates {
        let layer = &mut moe[index];
        let label = layer.label;
        let attached: Result<Arc<IgpuLayerEngine>, String> = match layer.igpu_host.as_ref() {
            Some(owners) => engine.bind_layer(label, owners.weights()),
            None => Err("host owners vanished".to_string()),
        };
        match attached {
            Ok(layer_engine) => {
                layer.igpu = Some(layer_engine);
                bound += 1;
            }
            // An ordinary failed symmetric check (or any rejected bind) keeps this
            // layer on the normal route. A bind that failed AFTER work may have
            // been in flight and could not be drained is fatal (below): the
            // engine may still read the pool.
            Err(why) => {
                skipped.push((label, why));
                // Unsafe drain: fatal below. Poisoned after a drained runtime
                // failure: the engine is dead, stop binding and treat the whole
                // arm as failed (no mixed engined/normal layers on a dead engine).
                if engine.unsafe_to_release() || engine.poisoned() {
                    break;
                }
            }
        }
    }
    if engine.unsafe_to_release() {
        let detail = skipped
            .last()
            .map_or_else(String::new, |(label, why)| format!(" (layer {label}: {why})"));
        let message = format!(
            "memory.offload_exec=igpu: a failed layer bind left the iGPU engine undrained{detail}; \
             the model Gpu is quarantined and the load fails"
        );
        // No normal fallback while the second device may still read the pool.
        gpu.quarantine(&message);
        return Err(invalid(message));
    }
    if bound == 0 || engine.poisoned() || engine.unsafe_to_release() {
        for layer in moe.iter_mut() {
            layer.igpu = None;
        }
        return fallback(&match skipped.first() {
            Some((label, why)) => format!("no layer bound (layer {label}: {why})"),
            None => "the engine is poisoned".to_string(),
        });
    }
    let engined = moe
        .iter()
        .filter(|layer| layer.igpu.is_some())
        .map(|layer| layer.label)
        .collect::<Vec<_>>();
    let normal = match skipped.first() {
        Some((label, why)) => format!(
            ", {} host layers stay on the normal route (layer {label}: {why})",
            skipped.len()
        ),
        None => String::new(),
    };
    // The pinned host backing is ONE pooled owner shared by every layer: count
    // each distinct owner once, never once per layer.
    let mut owners = std::collections::BTreeMap::new();
    for layer in moe.iter().filter(|layer| layer.igpu.is_some()) {
        if let Some(host) = layer.igpu_host.as_ref() {
            let (key, bytes) = host.owner();
            owners.insert(key, bytes);
        }
    }
    let mapped_bytes: usize = owners.values().sum();
    eprintln!(
        "  [igpu-exec] qwen4: device {} {}, {}/{} trunk layers engined, chunk {} rows, host bytes mapped {} MiB ({} backing), manifest {}, {}{normal}",
        arm.device,
        arm.arch,
        engined.len(),
        moe.len(),
        max_chunk,
        mapped_bytes >> 20,
        owners.len(),
        manifest_digest(&engine.manifest().canonical(), &engined, max_chunk),
        engine.summary_line(),
    );
    // A retained tape cannot own a cross-device host step: retained replay is
    // poisoned before publication, exactly as the CPU splice does.
    gpu.replay
        .poison("qwen4 iGPU prefill experts require eager HIP (memory.offload_exec=igpu)");
    Ok(bound)
}

/// Digest of the frozen numerical manifest: the engine's own canonical recipe
/// text (`IgpuManifest::canonical`: A4 producer and kernel identity, forced rows
/// policy at every width, geometry, canonical combine), the arch-side row gate
/// ([`hipfire_dispatch::families::moe::IGPU_PREFILL_MIN_ROWS`], eager prefill
/// only), the layers that actually BOUND (never the candidates) and the chunk
/// capacity the engine was sized for. Snapshot scopes carry it, so a cached prefix
/// is only reused by a bundle with the same numerical history.
pub fn manifest_digest(engine_canonical: &str, layers: &[usize], max_chunk: usize) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"qwen4-igpu-prefill-v2\n");
    hasher.update((engine_canonical.len() as u64).to_le_bytes());
    hasher.update(engine_canonical.as_bytes());
    hasher.update(
        (hipfire_dispatch::families::moe::IGPU_PREFILL_MIN_ROWS as u64).to_le_bytes(),
    );
    hasher.update((max_chunk as u64).to_le_bytes());
    hasher.update((layers.len() as u64).to_le_bytes());
    for layer in layers {
        hasher.update((*layer as u64).to_le_bytes());
    }
    let digest = hasher.finalize();
    digest[..8].iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The snapshot-scope suffix of a bundle. Empty when `memory.offload_exec` is
/// not `igpu` (the default scope is byte-for-byte unchanged); otherwise the
/// REQUEST mode always tags the scope — `/igpu-<digest>` when layers bound,
/// `/igpu-off` when the request fell back — so igpu and pcie/cpu prefixes never
/// share a cache.
pub fn scope_suffix(requested: bool, manifest: Option<&str>) -> String {
    match (requested, manifest) {
        (false, _) => String::new(),
        (true, Some(digest)) => format!("/igpu-{digest}"),
        (true, None) => "/igpu-off".to_string(),
    }
}

/// Whether one forward exposes the engine: an engined bundle running a REAL
/// prefill tile (`prefill_scope`, set only by the bundle's prefill entries, never
/// decode, MTP verify or the MTP head) that is not a lane tile and is wider than
/// the narrow verify window. Output-independent by construction: hidden-state
/// capture (speculative prompt fill) or an argmax request never changes which
/// arithmetic a prefill tile's experts use, so neither appears here.
pub(super) fn prefill_gate(engined: bool, prefill_scope: bool, lane_scope: bool, rows: usize) -> bool {
    engined
        && prefill_scope
        && !lane_scope
        && rows >= hipfire_dispatch::families::moe::IGPU_PREFILL_MIN_ROWS
}

/// Fill the engine gate of every layer runtime for one forward. Prefill >4
/// rows only: a verify window, decode token or lane step sees `None`.
pub(super) fn set_gate(moe: &[Qwen4MoeLayerRuntime], on: bool) {
    for layer in moe {
        layer.igpu_gate.store(on, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archs(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn prefill_gate_truth_table() {
        let min = hipfire_dispatch::families::moe::IGPU_PREFILL_MIN_ROWS;
        assert_eq!(min, 5);
        // (engined, prefill_scope, lane_scope, rows) -> gate
        let table = [
            // A real prefill tile wider than four rows takes the engine.
            ((true, true, false, 5), true),
            ((true, true, false, 512), true),
            ((true, true, false, 8192), true),
            // Narrow tiles and the decode row never do.
            ((true, true, false, 4), false),
            ((true, true, false, 1), false),
            ((true, true, false, 0), false),
            // Verify / decode / MTP head run with prefill_scope off, at any width.
            ((true, false, false, 5), false),
            ((true, false, false, 16), false),
            ((true, false, false, 4096), false),
            // A lane tile keeps its exact AR route.
            ((true, true, true, 512), false),
            // No engined layer, no engine.
            ((false, true, false, 512), false),
        ];
        for ((engined, prefill, lane, rows), expected) in table {
            assert_eq!(
                prefill_gate(engined, prefill, lane, rows),
                expected,
                "engined={engined} prefill={prefill} lane={lane} rows={rows}"
            );
        }
    }

    #[test]
    fn autodetect_picks_the_single_non_primary_gfx1151() {
        let visible = archs(&["gfx1100", "gfx1151"]);
        assert_eq!(select_second_device(0, None, &visible), Ok(1));
        // The primary is never its own second device, even when it is gfx1151.
        let visible = archs(&["gfx1151", "gfx1100"]);
        assert!(select_second_device(0, None, &visible).is_err());
        let visible = archs(&["gfx1100", "gfx1151:sramecc+:xnack-"]);
        assert_eq!(select_second_device(0, None, &visible), Ok(1));
    }

    #[test]
    fn autodetect_refuses_zero_and_ambiguous() {
        assert!(select_second_device(0, None, &archs(&["gfx1100"])).is_err());
        assert!(select_second_device(0, None, &archs(&["gfx1100", "gfx1201"])).is_err());
        let two = archs(&["gfx1100", "gfx1151", "gfx1151"]);
        let error = select_second_device(0, None, &two).unwrap_err();
        assert!(error.contains("HIPFIRE_IGPU_DEVICE"), "{error}");
        // The override resolves the ambiguity.
        assert_eq!(select_second_device(0, Some(2), &two), Ok(2));
    }

    #[test]
    fn override_must_name_a_visible_device() {
        let visible = archs(&["gfx1100", "gfx1151"]);
        assert_eq!(select_second_device(0, Some(1), &visible), Ok(1));
        assert!(select_second_device(0, Some(2), &visible).is_err());
        assert!(select_second_device(0, Some(-1), &visible).is_err());
    }

    #[test]
    fn plain_load_is_admitted_on_a_distinct_gfx1151() {
        assert_eq!(
            qwen4_igpu_refusal(
                "gfx1100",
                0,
                "gfx1151",
                1,
                false,
                &Qwen4IgpuLoadInputs::PLAIN
            ),
            Ok(())
        );
    }

    #[test]
    fn every_exclusion_refuses_with_a_reason() {
        let base = Qwen4IgpuLoadInputs::PLAIN;
        let cases = [
            ("ep", Qwen4IgpuLoadInputs { ep: true, ..base }),
            ("pp", Qwen4IgpuLoadInputs { pp: true, ..base }),
            ("slab", Qwen4IgpuLoadInputs { slab: true, ..base }),
            ("tier", Qwen4IgpuLoadInputs { tier: true, ..base }),
            ("hotset", Qwen4IgpuLoadInputs { hotset: true, ..base }),
            ("awq", Qwen4IgpuLoadInputs { awq: true, ..base }),
            (
                "nonuniform",
                Qwen4IgpuLoadInputs {
                    nonuniform_experts: true,
                    ..base
                },
            ),
            (
                "partial host layers",
                Qwen4IgpuLoadInputs {
                    host_whole_layers_only: false,
                    ..base
                },
            ),
        ];
        for (name, inputs) in cases {
            let reason = qwen4_igpu_refusal("gfx1100", 0, "gfx1151", 1, false, &inputs)
                .expect_err(name);
            assert!(!reason.is_empty() && !reason.contains('\n'), "{name}: {reason}");
        }
        assert!(qwen4_igpu_refusal("gfx1100", 0, "gfx1151", 1, true, &base).is_err());
        assert!(qwen4_igpu_refusal("gfx1100", 0, "", 1, false, &base).is_err());
        assert!(qwen4_igpu_refusal("gfx1100", 0, "gfx1100", 1, false, &base).is_err());
        assert!(qwen4_igpu_refusal("gfx1100", 1, "gfx1151", 1, false, &base).is_err());
    }

    fn layer(host: bool) -> IgpuLayerAdmission {
        IgpuLayerAdmission {
            host_mapped: host,
            owners: true,
            uniform_qt44_qt53: true,
            ..IgpuLayerAdmission::default()
        }
    }

    #[test]
    fn layer_admission_table() {
        assert_eq!(layer(true).refusal(), None);
        assert!(layer(true).host_placed());
        // A VRAM-resident layer is simply not an engine layer.
        assert!(!layer(false).host_placed());
        let cases: [(&str, IgpuLayerAdmission); 7] = [
            ("ep", IgpuLayerAdmission { ep: true, ..layer(true) }),
            ("slab", IgpuLayerAdmission { slab: true, ..layer(true) }),
            ("tier", IgpuLayerAdmission { tier: true, ..layer(true) }),
            ("fp8", IgpuLayerAdmission { fp8: true, ..layer(true) }),
            ("awq", IgpuLayerAdmission { awq: true, ..layer(true) }),
            (
                "nonuniform",
                IgpuLayerAdmission {
                    uniform_qt44_qt53: false,
                    ..layer(true)
                },
            ),
            ("views", IgpuLayerAdmission { owners: false, ..layer(true) }),
        ];
        for (name, admission) in cases {
            assert!(admission.refusal().is_some(), "{name}");
        }
    }

    #[test]
    fn default_scope_suffix_is_empty_and_request_always_tags() {
        // Default (pcie/cpu): the scope string is byte-for-byte unchanged.
        assert_eq!(scope_suffix(false, None), "");
        assert_eq!(scope_suffix(false, Some("abcd")), "");
        // Requested igpu always separates, armed or fallen back.
        assert_eq!(scope_suffix(true, None), "/igpu-off");
        assert_eq!(scope_suffix(true, Some("0123456789abcdef")), "/igpu-0123456789abcdef");
        assert_ne!(scope_suffix(true, None), scope_suffix(true, Some("00")));
    }

    #[test]
    fn manifest_digest_binds_recipe_layers_and_chunk() {
        let a = manifest_digest("recipe-a", &[0, 1, 2], 4096);
        assert_eq!(a, manifest_digest("recipe-a", &[0, 1, 2], 4096));
        assert_ne!(a, manifest_digest("recipe-b", &[0, 1, 2], 4096));
        // Only the layers that bound count: a layer rejected by the symmetric
        // check changes the manifest.
        assert_ne!(a, manifest_digest("recipe-a", &[0, 2], 4096));
        assert_ne!(a, manifest_digest("recipe-a", &[0, 1, 2], 2048));
        assert_eq!(a.len(), 16);
    }
}
