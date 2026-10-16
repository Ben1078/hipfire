// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Explicit developer/lab two-bundle loader for the eager Flash-Next
//! disaggregation: one full-model gfx1151 (Halo) producer bundle for prefill
//! and one full-model gfx1100 (XTX) consumer bundle for decode, over ONE
//! immutable shared host expert backing. Never reached by ordinary loading and
//! never a daemon route; every other loader is unchanged.
//!
//! # Ownership
//!
//! * The producer's `Gpu` arms portable host-mapped allocation and reserves
//!   ONE `hipHostMalloc(MAPPED | PORTABLE)` pool sized from every routed expert
//!   aggregate (the 48 trunk layers and the MTP layer, whole-layer placement,
//!   no hot-set tier plan). The producer's routed experts are host-mapped
//!   subviews of it and the GPU reads them natively.
//! * After the producer assembles, a `HostMappedView` (a lease on the pool) is
//!   captured per aggregate from the published weights. The pool cannot be freed
//!   while any lease lives.
//! * The consumer's cold layers (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, else the
//!   deterministic conservative fit) borrow the same pinned bytes through
//!   `Gpu::adopt_host_mapped_view` (the XTX's own alias of the owner base), and
//!   its hot layers copy those same host bytes to VRAM with the ordinary
//!   weight upload. No second pinned copy of the experts exists.
//! * Teardown frees the consumer first (its aliases drop their leases), then
//!   the captured leases, then the producer and its pool
//!   ([`DisaggPair::unload`]). Dropping a [`DisaggPair`] without `unload`
//!   frees no device state (a bundle has no destructor), so it never frees the
//!   pool under a live lease.
//!
//! # Execution policy
//!
//! The producer runs `HostExecPolicy::Gpu` (no CPU splice, tier split or iGPU
//! engine, whatever `memory.offload_exec` says); the consumer runs
//! `HostExecPolicy::Cpu` iff the process-wide mode is `cpu`/`passback`, else
//! `Gpu`. [`DisaggLoaded::run`] scopes every execution call by that policy;
//! callers must use it for prefill, decode, MTP fill, export and import.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hipfire_dispatch::cpu_exec::{self, with_host_exec_policy, HostExecPolicy};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::{
    fulfill_manifest_from_payloads, fulfill_manifest_from_payloads_with_residents, WeightHandle,
    WeightOrigin,
};
use rdna_compute::replay::{ReplayBackendRequest, ReplayState};
use rdna_compute::{DType, Gpu, GpuTensor, HostMappedView};
use sha2::{Digest, Sha256};

use crate::artifact::{admit_hfqm_artifact_on, source_forward_scratch_mode, Qwen4HfqmArtifact};
use crate::bundle::Qwen4Bundle;
use crate::disagg::DisaggRole;
use crate::disagg_manifest::{
    flash_next_handoff_payload_bytes, HandoffDomain, NumericalManifest, CONSUMER_ARCH,
    PRODUCER_ARCH,
};
use crate::expert_residency as residency;
use crate::gpu_forward::{qwen4_prefill_chunk_requested, Qwen4ForwardScratchMode};
use crate::weights::Qwen4TierRecipe;
use crate::{resolve_state_format, Qwen4KvBackend, Qwen4StateFormat};

/// Device/host memory a load keeps free beyond the growth it can name. Mirrors
/// `hipfire_loader::admission::QWEN4_MEMORY_HEADROOM_BYTES` (the loader depends
/// on this crate, so the constant cannot be imported).
const HEADROOM_BYTES: u64 = 2 << 30;
/// Draft depth the lab admits for native MTP windows (D2's narrow window is four
/// rows: one seed plus three drafts).
const LAB_MTP_MAX_K: usize = 3;
/// Quant types of the canonical Flash-Next uniform routed-expert aggregates.
const QT_GATE_UP: u8 = 44;
const QT_DOWN: u8 = 53;
const GATE_UP_SUFFIX: &str = ".mlp.experts.gate_up_proj";
const DOWN_SUFFIX: &str = ".mlp.experts.down_proj";
const HOTSET_ENV: &str = "HIPFIRE_QWEN4_EXPERT_HOTSET";
const REQUANT_ENV: &str = "HIPFIRE_QWEN4_REQUANT";
const STATE_QUANT_ENV: &str = "HIPFIRE_STATE_QUANT";
const IDENTITY_VERSION: &str = "qwen4-disagg-v1";

type R<T> = Result<T, String>;

fn is_routed_expert(name: &str) -> bool {
    name.ends_with(GATE_UP_SUFFIX) || name.ends_with(DOWN_SUFFIX)
}

fn is_mtp(name: &str) -> bool {
    name.starts_with("mtp.")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha_hex(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for part in parts {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    hex(&h.finalize())
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn mib(bytes: u64) -> u64 {
    bytes >> 20
}

// ── public API ───────────────────────────────────────────────────────

/// One loaded side: its own `Gpu` and full bundle (forward and native MTP
/// attached), the execution policy every call must run under, and the frozen
/// numerical identity bound into the bundle.
pub struct DisaggLoaded {
    pub gpu: Gpu,
    pub bundle: Qwen4Bundle,
    pub role: DisaggRole,
    /// The host-exec policy this side was assembled under; [`Self::run`] scopes
    /// every execution call by it.
    pub policy: HostExecPolicy,
    pub identity: NumericalManifest,
}

impl DisaggLoaded {
    /// Run `f` on this side with its device bound and its host-exec policy in
    /// scope. Use for every forward, MTP fill, decode, export and import: the
    /// producer must never inherit the process-wide CPU splice mode.
    pub fn run<T>(&mut self, f: impl FnOnce(&mut Gpu, &mut Qwen4Bundle) -> T) -> R<T> {
        self.gpu
            .bind_thread()
            .map_err(|error| format!("bind device {}: {error}", self.gpu.device_id))?;
        let (gpu, bundle) = (&mut self.gpu, &mut self.bundle);
        Ok(with_host_exec_policy(self.policy, || f(gpu, bundle)))
    }
}

/// What the load admitted and charged, in bytes, before the first weight byte
/// was allocated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisaggAdmission {
    pub max_seq: usize,
    /// Routed-expert aggregates in the ONE shared pinned pool (trunk + MTP).
    pub pool_tensors: usize,
    /// Pool capacity: Σ `Gpu::host_pool_footprint` of every aggregate (pad and
    /// page rounding included).
    pub pool_bytes: u64,
    /// `flash_next_handoff_payload_bytes(max_seq)`.
    pub handoff_payload_bytes: u64,
    /// Host bytes reserved for the handoff: one pinned export payload plus one
    /// saved-state byte copy.
    pub handoff_host_bytes: u64,
    pub host_mem_available: Option<u64>,
    pub producer_free: u64,
    pub producer_non_expert: u64,
    pub producer_reserve: u64,
    pub producer_required: u64,
    pub consumer_free: u64,
    pub consumer_non_expert: u64,
    /// Trunk layers whose routed experts are VRAM-resident on the consumer.
    pub consumer_resident_layers: usize,
    /// Whether `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` fixed the placement (else the
    /// deterministic conservative fit chose it).
    pub consumer_layers_explicit: bool,
    pub consumer_hot_bytes: u64,
    pub consumer_reserve: u64,
    pub consumer_margin: u64,
    pub consumer_required: u64,
    pub headroom: u64,
}

impl DisaggAdmission {
    /// The admission as log lines (MiB).
    pub fn lines(&self) -> Vec<String> {
        vec![
            format!(
                "disagg pool: {} routed-expert aggregates, {} MiB in ONE portable pinned backing \
                 (host MemAvailable {} MiB); handoff payload {} MiB, {} MiB host reserved",
                self.pool_tensors,
                mib(self.pool_bytes),
                self.host_mem_available.map_or(0, mib),
                mib(self.handoff_payload_bytes),
                mib(self.handoff_host_bytes)
            ),
            format!(
                "disagg producer: {} MiB free; needs {} MiB = {} non-expert + {} reserve + {} pool + {} handoff + {} headroom",
                mib(self.producer_free),
                mib(self.producer_required),
                mib(self.producer_non_expert),
                mib(self.producer_reserve),
                mib(self.pool_bytes),
                mib(self.handoff_host_bytes),
                mib(self.headroom)
            ),
            format!(
                "disagg consumer: {} MiB free; needs {} MiB = {} non-expert + {} hot experts ({} layers, {}) + {} reserve + {} margin + {} headroom",
                mib(self.consumer_free),
                mib(self.consumer_required),
                mib(self.consumer_non_expert),
                mib(self.consumer_hot_bytes),
                self.consumer_resident_layers,
                if self.consumer_layers_explicit { "explicit" } else { "conservative fit" },
                mib(self.consumer_reserve),
                mib(self.consumer_margin),
                mib(self.headroom)
            ),
        ]
    }
}

/// The explicit two-bundle lab pair over one shared host expert owner.
///
/// Field order is the implicit drop order: the consumer, then the captured
/// leases, then the producer. [`Self::unload`] is the only path that frees
/// device state.
pub struct DisaggPair {
    consumer: Option<DisaggLoaded>,
    /// The consumer's `Gpu` while no consumer bundle is loaded (between a
    /// failed reload and the next one).
    consumer_gpu: Option<Gpu>,
    shared: SharedExperts,
    pub producer: DisaggLoaded,
    /// Immutable ordered (producer, consumer) identity both bundles are bound
    /// to; the cache/handoff domain.
    pub domain: HandoffDomain,
    pub admission: DisaggAdmission,
    pub tokenizer: Tokenizer,
    pub max_seq: usize,
    plan: Plan,
    /// The model source (and, through it, the PLE sidecar rows) stays live
    /// for the pair's whole life: PLE row readers and a consumer reload read it.
    source: HfqModelSource,
    path: PathBuf,
}

impl DisaggPair {
    /// Load both bundles over one shared expert owner. Validates and admits the
    /// whole pair before the first weight allocation; any refusal is one line.
    pub fn load(
        path: &Path,
        prefill_device: i32,
        decode_device: i32,
        max_seq: usize,
    ) -> Result<Self, String> {
        load_pair(path, prefill_device, decode_device, max_seq).map_err(|error| one_line(&error))
    }

    /// The consumer, `Err` after a failed [`Self::reload_consumer`].
    pub fn consumer(&self) -> R<&DisaggLoaded> {
        self.consumer.as_ref().ok_or_else(consumer_missing)
    }

    pub fn consumer_mut(&mut self) -> R<&mut DisaggLoaded> {
        self.consumer.as_mut().ok_or_else(consumer_missing)
    }

    /// Both sides at once (producer, consumer, domain), for export/import.
    pub fn parts_mut(&mut self) -> R<(&mut DisaggLoaded, &mut DisaggLoaded, &HandoffDomain)> {
        let consumer = self.consumer.as_mut().ok_or_else(consumer_missing)?;
        Ok((&mut self.producer, consumer, &self.domain))
    }

    /// Unload the XTX bundle (its aliases release their leases) and load a fresh
    /// one over the same shared host owner. The placement is the one admitted
    /// at load; the new bundle must re-derive the exact bound consumer identity
    /// (same route, placements and canonical chunk), else it is torn down and
    /// the reload refused. On failure the pair keeps no consumer bundle but
    /// keeps its `Gpu`, so a later reload may retry; the producer and the pool
    /// are untouched. No third bundle exists at any time.
    pub fn reload_consumer(&mut self) -> Result<(), String> {
        self.reload_consumer_inner().map_err(|error| one_line(&error))
    }

    fn reload_consumer_inner(&mut self) -> R<()> {
        let mut gpu = match (self.consumer.take(), self.consumer_gpu.take()) {
            (Some(old), _) => {
                let DisaggLoaded { mut gpu, bundle, .. } = old;
                let freed = free_bundle(&mut gpu, bundle);
                if let Err(error) = freed {
                    self.consumer_gpu = Some(gpu);
                    return Err(format!("consumer unload before reload: {error}"));
                }
                if gpu.host_adopted_count() != 0 {
                    let leaked = gpu.host_adopted_count();
                    self.consumer_gpu = Some(gpu);
                    return Err(format!(
                        "consumer unload left {leaked} borrowed expert aliases; reload refused"
                    ));
                }
                gpu
            }
            (None, Some(gpu)) => gpu,
            (None, None) => return Err("consumer Gpu is gone; reload is impossible".to_string()),
        };
        let rebuilt = (|| -> R<(Qwen4Bundle, NumericalManifest)> {
            check_consumer_refit(&gpu, &self.plan, &self.admission)?;
            let bundle = build_consumer(&mut gpu, &self.plan, &self.source, &self.shared, self.max_seq)?;
            let identity = match identity_for(
                &gpu,
                &bundle,
                DisaggRole::Consumer,
                &self.plan.consumer,
                &self.plan,
                self.plan.consumer_policy,
                self.max_seq,
            ) {
                Ok(identity) => identity,
                Err(error) => return Err(teardown_note(&mut gpu, bundle, error)),
            };
            if identity != self.domain.consumer {
                return Err(teardown_note(
                    &mut gpu,
                    bundle,
                    "the reloaded consumer's numerical identity differs from the bound domain \
                     (route, placement or canonical chunk changed)"
                        .to_string(),
                ));
            }
            let mut bundle = bundle;
            if let Err(error) = bundle.bind_disagg(DisaggRole::Consumer, &self.domain) {
                return Err(teardown_note(&mut gpu, bundle, format!("bind consumer domain: {error}")));
            }
            Ok((bundle, identity))
        })();
        match rebuilt {
            Ok((bundle, identity)) => {
                self.consumer = Some(DisaggLoaded {
                    gpu,
                    bundle,
                    role: DisaggRole::Consumer,
                    policy: self.plan.consumer_policy,
                    identity,
                });
                Ok(())
            }
            Err(error) => {
                self.consumer_gpu = Some(gpu);
                Err(error)
            }
        }
    }

    /// Ordered teardown: consumer bundle (its aliases release their leases),
    /// the captured leases, then the producer bundle and the pool. Returns
    /// every failure; a failure never frees anything out of order.
    pub fn unload(self) -> Result<(), String> {
        let DisaggPair { consumer, consumer_gpu, shared, producer, source, .. } = self;
        let mut errors = Vec::new();
        if let Some(DisaggLoaded { mut gpu, bundle, .. }) = consumer {
            if let Err(error) = free_bundle(&mut gpu, bundle) {
                errors.push(format!("consumer: {error}"));
            }
            if gpu.host_adopted_count() != 0 {
                errors.push(format!(
                    "consumer retained {} borrowed expert aliases",
                    gpu.host_adopted_count()
                ));
            }
        }
        drop(consumer_gpu);
        drop(shared);
        let DisaggLoaded { mut gpu, bundle, .. } = producer;
        if let Err(error) = free_bundle(&mut gpu, bundle) {
            errors.push(format!("producer: {error}"));
        }
        if gpu.host_mapped_count() != 0 {
            errors.push(format!(
                "producer retained {} pinned host owners (the shared pool is still leased)",
                gpu.host_mapped_count()
            ));
        }
        drop(source);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

fn consumer_missing() -> String {
    "the consumer bundle is not loaded (a reload failed); call reload_consumer".to_string()
}

// ── plan: everything decided before the first weight allocation ──────

struct SidePlan {
    receipt: Qwen4HfqmArtifact,
    /// The manifest weights with this side's residency applied.
    weights: Vec<WeightEntry>,
    scratch_mode: Qwen4ForwardScratchMode,
    state_format: Qwen4StateFormat,
    chunk_requested: usize,
}

struct Plan {
    producer: SidePlan,
    consumer: SidePlan,
    resident_layers: usize,
    consumer_policy: HostExecPolicy,
    artifact_digest: String,
    pool_sizes: Vec<usize>,
    ple_identity: String,
    kv_request: String,
    state_request: String,
}

/// A captured lease on one pooled aggregate plus what it holds.
struct CapturedExpert {
    view: HostMappedView,
    shape: Vec<usize>,
    dtype: DType,
}

/// Leases on every pooled aggregate, keyed `(tensor name, layer)`. Dropping
/// them (after the consumer's aliases are gone) lets the producer free the pool.
struct SharedExperts {
    views: HashMap<(String, Option<usize>), CapturedExpert>,
}

fn view_bytes(view: &HostMappedView) -> R<&[u8]> {
    let end = view
        .offset()
        .checked_add(view.len())
        .ok_or("shared expert extent overflows")?;
    if end > view.owner_len() {
        return Err(format!("shared expert extent {end} exceeds its {}-byte owner", view.owner_len()));
    }
    // SAFETY: the view's lease keeps the pinned owner allocated and immutable
    // (the producer never writes a weight after upload); the extent was checked
    // against the owner above. The slice borrows the view, which the caller
    // holds for the whole use.
    Ok(unsafe { std::slice::from_raw_parts(view.owner_host_base().add(view.offset()), view.len()) })
}

// ── load ─────────────────────────────────────────────────────────────

fn load_pair(path: &Path, prefill_device: i32, decode_device: i32, max_seq: usize) -> R<DisaggPair> {
    if prefill_device < 0 || decode_device < 0 || prefill_device == decode_device {
        return Err(format!(
            "disagg load needs two distinct non-negative device ids, got {prefill_device} and {decode_device}"
        ));
    }
    if max_seq == 0 {
        return Err("disagg load needs a non-zero max_seq".to_string());
    }
    // Policy refusals that need no device.
    if hipfire_config::developer_var(HOTSET_ENV).is_ok() {
        return Err(format!(
            "disagg lab uses whole-layer expert placement; unset {HOTSET_ENV} (no tier/hot-set composition)"
        ));
    }
    if hipfire_config::developer_var(REQUANT_ENV).is_ok() {
        return Err(format!("disagg lab refuses {REQUANT_ENV}: both sides must run the artifact's own weights"));
    }
    if hipfire_config::topology::qwen4_ep().unwrap_or(1) > 1 {
        return Err("disagg lab refuses expert parallelism (topology.ep > 1)".to_string());
    }
    if cpu_exec::igpu_exec_enabled() {
        return Err("disagg lab refuses memory.offload_exec=igpu (two explicit devices; no iGPU engine)".to_string());
    }
    let consumer_policy = if cpu_exec::cpu_exec_enabled() { HostExecPolicy::Cpu } else { HostExecPolicy::Gpu };

    let mut hfq = HfqFile::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| format!("tokenizer from {}: {error}", path.display()))?;

    // Initialising a Gpu allocates no model memory; the arch gate runs first.
    let mut producer_gpu = Gpu::init_with_device(prefill_device)
        .map_err(|error| format!("init prefill device {prefill_device}: {error}"))?;
    let mut consumer_gpu = Gpu::init_with_device(decode_device)
        .map_err(|error| format!("init decode device {decode_device}: {error}"))?;
    if producer_gpu.arch != PRODUCER_ARCH || consumer_gpu.arch != CONSUMER_ARCH {
        return Err(format!(
            "disagg needs {PRODUCER_ARCH} (device {prefill_device}) -> {CONSUMER_ARCH} (device {decode_device}), \
             got {} -> {}",
            producer_gpu.arch, consumer_gpu.arch
        ));
    }
    if !producer_gpu.is_uma() || consumer_gpu.is_uma() {
        return Err("disagg needs an integrated producer (unified memory) and a discrete consumer".to_string());
    }
    for (side, gpu) in [("producer", &producer_gpu), ("consumer", &consumer_gpu)] {
        if gpu.replay.request() != ReplayBackendRequest::Hip || gpu.replay.state() != ReplayState::Hip {
            return Err(format!(
                "disagg is eager-only: the {side} has replay/graph backend {:?}/{:?} (set replay.backend=hip)",
                gpu.replay.request(),
                gpu.replay.state()
            ));
        }
    }

    // PLE rows resolve ONCE, at the producer's arch, onto the shared file:
    // both bundles read the same rows (the request's `auto` would otherwise
    // pick q4 on gfx1151 and q8 on gfx1100).
    let ple_choice = hipfire_runtime::ple_sidecar::attach_validation_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &producer_gpu.arch,
    )
    .map_err(|error| format!("qwen4 PLE rows: {error}"))?;
    let producer_receipt = admit_hfqm_artifact_on(&hfq, Some(&producer_gpu.arch))
        .map_err(|error| format!("qwen4 admission (producer): {error}"))?;
    let consumer_receipt = admit_hfqm_artifact_on(&hfq, Some(&consumer_gpu.arch))
        .map_err(|error| format!("qwen4 admission (consumer): {error}"))?;
    if producer_receipt.manifest != consumer_receipt.manifest
        || format!("{:?}", producer_receipt.ple_rows) != format!("{:?}", consumer_receipt.ple_rows)
    {
        return Err("the artifact admits to different manifests/PLE rows on the two archs".to_string());
    }
    let config = &producer_receipt.config;
    if !config.profile.is_flash_next() || config.mtp.is_none() {
        return Err("disagg lab supports the Flash-Next profile with its native MTP head only".to_string());
    }
    if !matches!(producer_receipt.manifest.identity.recipe, Qwen4TierRecipe::FlashNextPacked { .. }) {
        return Err("disagg lab supports the packed Flash-Next recipe only (no expert-slab/stock-FP8)".to_string());
    }
    if max_seq > config.max_position_embeddings {
        return Err(format!(
            "max_seq {max_seq} exceeds the model's {} positions",
            config.max_position_embeddings
        ));
    }
    for (side, gpu, receipt) in [
        ("producer", &producer_gpu, &producer_receipt),
        ("consumer", &consumer_gpu, &consumer_receipt),
    ] {
        let overlay = crate::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
            .map_err(|error| format!("qwen4 trunk overlay ({side}): {error}"))?;
        if overlay.is_some() {
            return Err(format!("disagg lab refuses a trunk overlay/NPU composition ({side})"));
        }
    }

    // Index sizes and canonical-expert validation (no allocation).
    let mut sizes = HashMap::new();
    let mut qts = HashMap::new();
    for info in hfq.tensors() {
        sizes.insert(info.name.clone(), info.data_size as u64);
        qts.insert(info.name.clone(), info.quant_type);
    }
    validate_canonical_experts(&producer_receipt.manifest.weights, &qts, config.num_hidden_layers)?;
    let bytes_of = |entry: &WeightEntry| sizes.get(&entry.name).copied();

    // Scratch plans while the mapping is readable.
    let producer_scratch = source_forward_scratch_mode(
        &producer_gpu,
        &hfq,
        &producer_receipt.config,
        &producer_receipt.manifest,
    )
    .map_err(|error| format!("qwen4 forward scratch plan (producer): {error}"))?;
    let consumer_scratch = source_forward_scratch_mode(
        &consumer_gpu,
        &hfq,
        &consumer_receipt.config,
        &consumer_receipt.manifest,
    )
    .map_err(|error| format!("qwen4 forward scratch plan (consumer): {error}"))?;

    // State formats: the pair's storage formats must agree exactly; the
    // device-only F16/pooled mirrors do not travel and may differ.
    let kv_request = hipfire_runtime::config::get().kv_mode.clone();
    let state_request = match std::env::var(STATE_QUANT_ENV) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => "fp32".to_string(),
    };
    let producer_state = resolve_state_format(&kv_request, &state_request, &producer_gpu, config)?;
    let consumer_state = resolve_state_format(&kv_request, &state_request, &consumer_gpu, config)?;
    if producer_state.qsa != consumer_state.qsa || producer_state.gdn != consumer_state.gdn {
        return Err(format!(
            "producer and consumer resolve different state formats (qsa {:?}/{:?}, gdn {:?}/{:?}); \
             pin memory.kv_cache and {STATE_QUANT_ENV}",
            producer_state.qsa, consumer_state.qsa, producer_state.gdn, consumer_state.gdn
        ));
    }

    let artifact_digest = artifact_digest(&hfq, &ple_choice.format.name(), &producer_receipt);

    // Placement: producer = every routed aggregate host-mapped in the ONE pool.
    let mut producer_weights = producer_receipt.manifest.weights.clone();
    if let Some(entry) = producer_weights.iter().find(|e| {
        e.residency != WeightResidency::Resident && !e.residency.is_external()
    }) {
        return Err(format!(
            "'{}' is already host-mapped in the manifest; the lab shares only the routed experts",
            entry.name
        ));
    }
    if producer_weights.iter().any(|e| e.name.ends_with(crate::expert_arena::ARENA_SUFFIX)) {
        return Err("disagg lab refuses expert slab arenas (no honest shared owner)".to_string());
    }
    let mut pool_sizes = Vec::new();
    for entry in producer_weights.iter_mut().filter(|e| is_routed_expert(&e.name)) {
        entry.residency = WeightResidency::HostMapped;
        let bytes = bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        pool_sizes.push(usize::try_from(bytes).map_err(|_| "expert aggregate size exceeds usize".to_string())?);
    }
    if pool_sizes.is_empty() {
        return Err("the artifact has no routed expert aggregates to share".to_string());
    }
    let mut pool_bytes = 0u64;
    for &size in &pool_sizes {
        pool_bytes = pool_bytes
            .checked_add(Gpu::host_pool_footprint(size) as u64)
            .ok_or("shared pool bytes overflow")?;
    }

    // Reserves (the loader's bottom-up census + historical floor) per device.
    let producer_reserve = reserve_for(&producer_gpu, config, &producer_receipt.manifest.weights, &bytes_of, producer_state, producer_scratch, max_seq)?;
    let consumer_reserve = reserve_for(&consumer_gpu, config, &consumer_receipt.manifest.weights, &bytes_of, consumer_state, consumer_scratch, max_seq)?;
    let (non_expert, _layer0) = residency::resident_split(&producer_receipt.manifest.weights, &bytes_of)?;
    let handoff_payload = flash_next_handoff_payload_bytes(max_seq)?;
    let handoff_host = handoff_payload.checked_mul(2).ok_or("handoff host reserve overflows")?;

    // Host-side checks for the one pinned pool (+ the handoff buffers).
    let host_bytes = pool_bytes.checked_add(handoff_host).ok_or("host byte total overflows")?;
    let mem_available = rdna_compute::kv_slots::mem_available_bytes();
    residency::check_host_ram(host_bytes, mem_available, residency::ttm_pool_estimate())
        .map_err(|error| format!("qwen4: {error}"))?;
    residency::check_gtt_cap(pool_bytes, residency::gtt_budget()).map_err(|error| format!("qwen4: {error}"))?;

    let (producer_free, _) = producer_gpu
        .device_mem_info()
        .map_err(|error| format!("producer device memory query: {error}"))?;
    let producer_free = producer_free as u64;
    let producer_required = non_expert
        .checked_add(producer_reserve.bytes())
        .and_then(|v| v.checked_add(pool_bytes))
        .and_then(|v| v.checked_add(handoff_host))
        .and_then(|v| v.checked_add(HEADROOM_BYTES))
        .ok_or("producer requirement overflows")?;
    if producer_free < producer_required {
        return Err(format!(
            "gfx1151 cannot hold the shared pool: needs {} MiB ({} non-expert + {} reserve + {} pool + {} handoff + {} headroom), {} MiB free",
            mib(producer_required),
            mib(non_expert),
            mib(producer_reserve.bytes()),
            mib(pool_bytes),
            mib(handoff_host),
            mib(HEADROOM_BYTES),
            mib(producer_free)
        ));
    }

    // Consumer: whole-layer fixed placement.
    let layer_bytes = trunk_layer_expert_bytes(&producer_receipt.manifest.weights, &bytes_of, config.num_hidden_layers)?;
    let margin = residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES;
    let (consumer_free, _) = consumer_gpu
        .device_mem_info()
        .map_err(|error| format!("consumer device memory query: {error}"))?;
    let consumer_free = consumer_free as u64;
    let charged = consumer_reserve
        .bytes()
        .checked_add(margin)
        .and_then(|v| v.checked_add(HEADROOM_BYTES))
        .ok_or("consumer reserve overflows")?;
    let explicit = match residency::expert_vram_layers_from_env()? {
        None => None,
        Some(residency::ExpertVramLayers::Layers(layers)) => Some(layers.min(config.num_hidden_layers)),
        Some(residency::ExpertVramLayers::Auto) => {
            return Err(format!(
                "disagg lab takes a fixed {} layer count (or unset for the conservative fit), not auto",
                residency::EXPERT_VRAM_LAYERS_ENV
            ))
        }
    };
    let resident_layers = match explicit {
        Some(layers) => {
            let need = consumer_required_bytes(non_expert, &layer_bytes, layers, charged)?;
            if need > consumer_free {
                return Err(format!(
                    "gfx1100 cannot hold {layers} resident expert layers: needs {} MiB, {} MiB free",
                    mib(need),
                    mib(consumer_free)
                ));
            }
            layers
        }
        None => fit_resident_layers(consumer_free, non_expert, &layer_bytes, charged).ok_or_else(|| {
            format!(
                "gfx1100 cannot hold the non-expert weights and reserve ({} MiB needed, {} MiB free) even with every expert layer on the host",
                mib(non_expert.saturating_add(charged)),
                mib(consumer_free)
            )
        })?,
    };
    let hot_bytes: u64 = layer_bytes.iter().take(resident_layers).sum();
    let consumer_required = consumer_required_bytes(non_expert, &layer_bytes, resident_layers, charged)?;

    let mut consumer_weights = consumer_receipt.manifest.weights.clone();
    for entry in consumer_weights.iter_mut().filter(|e| is_routed_expert(&e.name)) {
        let hot = !is_mtp(&entry.name) && entry.layer.is_some_and(|layer| layer < resident_layers);
        entry.residency = if hot { WeightResidency::Resident } else { WeightResidency::HostMapped };
    }

    let admission = DisaggAdmission {
        max_seq,
        pool_tensors: pool_sizes.len(),
        pool_bytes,
        handoff_payload_bytes: handoff_payload,
        handoff_host_bytes: handoff_host,
        host_mem_available: mem_available,
        producer_free,
        producer_non_expert: non_expert,
        producer_reserve: producer_reserve.bytes(),
        producer_required,
        consumer_free,
        consumer_non_expert: non_expert,
        consumer_resident_layers: resident_layers,
        consumer_layers_explicit: explicit.is_some(),
        consumer_hot_bytes: hot_bytes,
        consumer_reserve: consumer_reserve.bytes(),
        consumer_margin: margin,
        consumer_required,
        headroom: HEADROOM_BYTES,
    };
    for line in admission.lines() {
        eprintln!("  {line}");
    }

    let ple_identity = ple_identity(&hfq, &ple_choice.format.name());
    let plan = Plan {
        producer: SidePlan {
            chunk_requested: qwen4_prefill_chunk_requested(&producer_gpu.arch, max_seq),
            receipt: producer_receipt,
            weights: producer_weights,
            scratch_mode: producer_scratch,
            state_format: producer_state,
        },
        consumer: SidePlan {
            chunk_requested: qwen4_prefill_chunk_requested(&consumer_gpu.arch, max_seq),
            receipt: consumer_receipt,
            weights: consumer_weights,
            scratch_mode: consumer_scratch,
            state_format: consumer_state,
        },
        resident_layers,
        consumer_policy,
        artifact_digest,
        pool_sizes,
        ple_identity,
        kv_request,
        state_request,
    };

    // ── from here on the model mutates ──
    if producer_gpu.is_uma() {
        hfq.drop_mmap();
    }
    let source = HfqModelSource::from_hfq(hfq);

    let (producer_bundle, shared) = build_producer(&mut producer_gpu, &plan, &source, max_seq)?;
    let consumer_bundle = match build_consumer(&mut consumer_gpu, &plan, &source, &shared, max_seq) {
        Ok(bundle) => bundle,
        Err(error) => {
            return Err(teardown_producer(&mut producer_gpu, producer_bundle, shared, error));
        }
    };

    let identities = identity_for(
        &producer_gpu,
        &producer_bundle,
        DisaggRole::Producer,
        &plan.producer,
        &plan,
        HostExecPolicy::Gpu,
        max_seq,
    )
    .and_then(|producer| {
        identity_for(
            &consumer_gpu,
            &consumer_bundle,
            DisaggRole::Consumer,
            &plan.consumer,
            &plan,
            consumer_policy,
            max_seq,
        )
        .map(|consumer| (producer, consumer))
    });
    let (producer_identity, consumer_identity) = match identities {
        Ok(pair) => pair,
        Err(error) => return Err(teardown_both(producer_gpu, producer_bundle, consumer_gpu, consumer_bundle, shared, error)),
    };
    let domain = HandoffDomain { producer: producer_identity.clone(), consumer: consumer_identity.clone() };
    let mut producer_bundle = producer_bundle;
    let mut consumer_bundle = consumer_bundle;
    let bound = producer_bundle
        .bind_disagg(DisaggRole::Producer, &domain)
        .map_err(|error| format!("bind producer domain: {error}"))
        .and_then(|()| {
            consumer_bundle
                .bind_disagg(DisaggRole::Consumer, &domain)
                .map_err(|error| format!("bind consumer domain: {error}"))
        });
    if let Err(error) = bound {
        return Err(teardown_both(producer_gpu, producer_bundle, consumer_gpu, consumer_bundle, shared, error));
    }

    Ok(DisaggPair {
        consumer: Some(DisaggLoaded {
            gpu: consumer_gpu,
            bundle: consumer_bundle,
            role: DisaggRole::Consumer,
            policy: consumer_policy,
            identity: consumer_identity,
        }),
        consumer_gpu: None,
        shared,
        producer: DisaggLoaded {
            gpu: producer_gpu,
            bundle: producer_bundle,
            role: DisaggRole::Producer,
            policy: HostExecPolicy::Gpu,
            identity: producer_identity,
        },
        domain,
        admission,
        tokenizer,
        max_seq,
        plan,
        source,
        path: path.to_path_buf(),
    })
}

// ── builders ─────────────────────────────────────────────────────────

fn missing_tensor(name: &str) -> String {
    format!("missing tensor '{name}'")
}

/// The ordinary range payload (the oracle/loader path for every resident,
/// non-host-mapped tensor).
fn range_payload<'a>(source: &'a HfqModelSource, entry: &WeightEntry) -> R<SourcePayload<'a>> {
    source
        .tensor_range(&entry.name)
        .map_err(|error| error.to_string())?
        .map(SourcePayload::Range)
        .ok_or_else(|| missing_tensor(&entry.name))
}

/// Producer payloads: a host-mapped aggregate is read from the file into one
/// owned buffer at a time and copied into its slot of the shared pool (the
/// store rejects a host-mapped entry as a range).
fn producer_payload<'a>(source: &'a HfqModelSource, entry: &WeightEntry) -> R<SourcePayload<'a>> {
    if entry.residency != WeightResidency::HostMapped {
        return range_payload(source, entry);
    }
    let info = source
        .tensor_info(&entry.name)
        .ok_or_else(|| missing_tensor(&entry.name))?
        .clone();
    let range = source
        .tensor_range(&entry.name)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| missing_tensor(&entry.name))?;
    let len = usize::try_from(range.length).map_err(|_| format!("'{}' length exceeds usize", entry.name))?;
    let mut bytes = vec![0u8; len];
    range
        .read_exact(&mut bytes)
        .map_err(|error| format!("read '{}': {error}", entry.name))?;
    Ok(SourcePayload::Owned { info, bytes })
}

/// Consumer payloads: a VRAM-hot routed aggregate uploads from the SHARED host
/// bytes (never from a second pinned copy); everything else is the ordinary
/// range path.
fn consumer_payload<'a>(
    source: &'a HfqModelSource,
    shared: &'a SharedExperts,
    entry: &WeightEntry,
) -> R<SourcePayload<'a>> {
    if is_routed_expert(&entry.name) && entry.residency == WeightResidency::Resident {
        let captured = shared
            .views
            .get(&(entry.name.clone(), entry.layer))
            .ok_or_else(|| format!("no shared host bytes for hot expert '{}'", entry.name))?;
        let info = source
            .tensor_info(&entry.name)
            .ok_or_else(|| missing_tensor(&entry.name))?;
        let bytes = view_bytes(&captured.view)?;
        if bytes.len() as u64 != info.data_size as u64 {
            return Err(format!(
                "shared host bytes of '{}' are {} bytes, the artifact declares {}",
                entry.name,
                bytes.len(),
                info.data_size
            ));
        }
        return Ok(SourcePayload::Borrowed { info, bytes });
    }
    range_payload(source, entry)
}

/// Consumer resident injection: a cold routed aggregate is the XTX's own alias
/// of the producer's pinned owner (no allocation, no copy).
fn consumer_resident(gpu: &mut Gpu, shared: &SharedExperts, entry: &WeightEntry) -> R<Option<GpuTensor>> {
    if entry.residency != WeightResidency::HostMapped || !is_routed_expert(&entry.name) {
        return Ok(None);
    }
    let captured = shared
        .views
        .get(&(entry.name.clone(), entry.layer))
        .ok_or_else(|| format!("no shared owner view for cold expert '{}'", entry.name))?;
    if captured.shape != entry.logical_shape {
        return Err(format!(
            "shared owner shape {:?} of '{}' differs from the manifest {:?}",
            captured.shape, entry.name, entry.logical_shape
        ));
    }
    gpu.adopt_host_mapped_view(&captured.view, &captured.shape, captured.dtype)
        .map(Some)
        .map_err(|error| format!("adopt shared expert '{}': {error}", entry.name))
}

/// Assemble one side's bundle and attach the full forward and the native MTP
/// head. Runs under the side's host-exec policy. On failure everything this
/// call allocated is released.
fn assemble_and_attach(
    gpu: &mut Gpu,
    side: &SidePlan,
    transaction: hipfire_runtime::weight_store::WeightLoadTransaction,
    max_seq: usize,
) -> R<Qwen4Bundle> {
    let backend = Qwen4KvBackend::automatic(gpu);
    let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
        side.receipt.config.clone(),
        &side.receipt.manifest.identity.recipe,
        transaction,
        &side.receipt.placements,
        gpu,
        max_seq,
        side.receipt.ple.clone(),
        side.receipt.ple_rows.clone(),
        side.state_format,
        backend,
        None,
    )
    .map_err(|error| format!("qwen4 bundle assembly: {error}"))?;
    let attached = bundle
        .attach_forward(gpu, max_seq, side.scratch_mode)
        .map_err(|error| format!("qwen4 forward setup: {error}"))
        .and_then(|()| {
            bundle
                .attach_mtp(gpu, max_seq)
                .map_err(|error| format!("qwen4 MTP setup: {error}"))
        });
    match attached {
        Ok(()) => Ok(bundle),
        Err(error) => Err(teardown_note(gpu, bundle, error)),
    }
}

/// Producer: ONE portable pinned pool, every routed aggregate host-mapped in
/// it, then a lease on each aggregate captured from the published weights.
fn build_producer(
    gpu: &mut Gpu,
    plan: &Plan,
    source: &HfqModelSource,
    max_seq: usize,
) -> R<(Qwen4Bundle, SharedExperts)> {
    let side = &plan.producer;
    let mesh = DeviceMesh::single().map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = WeightOrigin::for_single(&mesh, gpu);
    // Portable flags must precede every expert allocation.
    gpu.arm_portable_host_mapped();
    let capacity = match gpu.reserve_portable_host_pool(&plan.pool_sizes) {
        Ok(capacity) => capacity,
        Err(error) => {
            gpu.disarm_portable_host_mapped();
            return Err(format!(
                "portable host pool ({} aggregates): {error}",
                plan.pool_sizes.len()
            ));
        }
    };
    eprintln!(
        "  disagg: {} routed-expert aggregates in one {} MiB portable pinned pool",
        plan.pool_sizes.len(),
        capacity >> 20
    );
    let transaction = match fulfill_manifest_from_payloads(
        &side.weights,
        &mesh,
        side.receipt.config.num_hidden_layers,
        gpu,
        expected,
        |entry| producer_payload(source, entry),
    ) {
        Ok(transaction) => transaction,
        Err(error) => {
            return Err(release_pool_note(
                gpu,
                format!("qwen4 manifest fulfillment (producer): {error}"),
            ));
        }
    };
    let bundle = with_host_exec_policy(HostExecPolicy::Gpu, || {
        assemble_and_attach(gpu, side, transaction, max_seq)
    });
    let bundle = match bundle {
        Ok(bundle) => bundle,
        Err(error) => return Err(release_pool_note(gpu, error)),
    };
    gpu.disarm_portable_host_mapped();
    match capture_shared(gpu, &bundle, &side.weights) {
        Ok(shared) => Ok((bundle, shared)),
        Err(error) => {
            let note = teardown_note(gpu, bundle, error);
            Err(release_pool_note(gpu, note))
        }
    }
}

/// One lease per pooled aggregate, from the producer's published weights.
fn capture_shared(gpu: &mut Gpu, bundle: &Qwen4Bundle, weights: &[WeightEntry]) -> R<SharedExperts> {
    let expected = weights
        .iter()
        .filter(|entry| entry.residency == WeightResidency::HostMapped)
        .count();
    let mut views = HashMap::new();
    for taken in &bundle.weights.taken {
        if !is_routed_expert(&taken.key.name) {
            continue;
        }
        let WeightHandle::Resident(tensor) = &taken.handle else {
            return Err(format!("routed expert '{}' is not a resident handle", taken.key.name));
        };
        if !tensor.buf.is_host_mapped() {
            return Err(format!("routed expert '{}' is not host-mapped on the producer", taken.key.name));
        }
        let view = gpu
            .host_mapped_view(tensor)
            .map_err(|error| format!("lease on '{}': {error}", taken.key.name))?;
        views.insert(
            (taken.key.name.clone(), taken.key.layer),
            CapturedExpert { view, shape: tensor.shape.clone(), dtype: tensor.dtype },
        );
    }
    if views.len() != expected {
        return Err(format!(
            "captured {} shared expert views, the manifest host-maps {expected}",
            views.len()
        ));
    }
    // One owner for all of them: a second owner would be a duplicated backing.
    let mut owners = views.values();
    if let Some(first) = owners.next() {
        if owners.any(|other| !other.view.same_owner(&first.view)) {
            return Err("the shared expert views span more than one pinned owner".to_string());
        }
    }
    Ok(SharedExperts { views })
}

/// Consumer: cold layers alias the producer's pool, hot layers copy the same
/// host bytes to VRAM, under the consumer's host-exec policy.
fn build_consumer(
    gpu: &mut Gpu,
    plan: &Plan,
    source: &HfqModelSource,
    shared: &SharedExperts,
    max_seq: usize,
) -> R<Qwen4Bundle> {
    let side = &plan.consumer;
    let mesh = DeviceMesh::single().map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = WeightOrigin::for_single(&mesh, gpu);
    let transaction = fulfill_manifest_from_payloads_with_residents(
        &side.weights,
        &mesh,
        side.receipt.config.num_hidden_layers,
        gpu,
        expected,
        |entry| consumer_payload(source, shared, entry),
        |gpu, entry| consumer_resident(gpu, shared, entry),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment (consumer): {error}"))?;
    with_host_exec_policy(plan.consumer_policy, || {
        assemble_and_attach(gpu, side, transaction, max_seq)
    })
}

// ── teardown helpers ─────────────────────────────────────────────────

fn free_bundle(gpu: &mut Gpu, bundle: Qwen4Bundle) -> R<()> {
    bundle.free_gpu(gpu).map_err(|error| error.to_string())
}

/// `error` plus the outcome of freeing `bundle`.
fn teardown_note(gpu: &mut Gpu, bundle: Qwen4Bundle, error: String) -> String {
    match free_bundle(gpu, bundle) {
        Ok(()) => error,
        Err(cleanup) => format!("{error}; cleanup: {cleanup}"),
    }
}

/// `error` plus the release of a producer pool nothing (or nothing leased)
/// still lives in; disarms portable allocation.
fn release_pool_note(gpu: &mut Gpu, error: String) -> String {
    gpu.disarm_portable_host_mapped();
    match gpu.release_unused_host_pool() {
        Ok(()) => error,
        Err(cleanup) => format!("{error}; pool release: {cleanup}"),
    }
}

fn teardown_producer(gpu: &mut Gpu, bundle: Qwen4Bundle, shared: SharedExperts, error: String) -> String {
    drop(shared);
    teardown_note(gpu, bundle, error)
}

fn teardown_both(
    mut producer_gpu: Gpu,
    producer_bundle: Qwen4Bundle,
    mut consumer_gpu: Gpu,
    consumer_bundle: Qwen4Bundle,
    shared: SharedExperts,
    error: String,
) -> String {
    // Consumer first (its aliases drop their leases), then the captured
    // leases, then the producer and its pool.
    let error = teardown_note(&mut consumer_gpu, consumer_bundle, error);
    teardown_producer(&mut producer_gpu, producer_bundle, shared, error)
}

fn check_consumer_refit(gpu: &Gpu, _plan: &Plan, admission: &DisaggAdmission) -> R<()> {
    let (free, _) = gpu
        .device_mem_info()
        .map_err(|error| format!("consumer device memory query: {error}"))?;
    if (free as u64) < admission.consumer_required {
        return Err(format!(
            "gfx1100 cannot reload the consumer: needs {} MiB, {} MiB free",
            mib(admission.consumer_required),
            mib(free as u64)
        ));
    }
    Ok(())
}

// ── admission helpers ────────────────────────────────────────────────

/// Every routed aggregate is the canonical uniform Flash-Next pair (gate_up
/// qt 44, down qt 53) on every trunk layer and the MTP layer. Returns the
/// routed tensor count.
fn validate_canonical_experts(
    weights: &[WeightEntry],
    qts: &HashMap<String, u8>,
    layers: usize,
) -> R<usize> {
    let mut gate_up = vec![false; layers];
    let mut down = vec![false; layers];
    let mut mtp = (false, false);
    let mut count = 0usize;
    for entry in weights.iter().filter(|entry| is_routed_expert(&entry.name)) {
        let is_gate_up = entry.name.ends_with(GATE_UP_SUFFIX);
        let want = if is_gate_up { QT_GATE_UP } else { QT_DOWN };
        let qt = *qts.get(&entry.name).ok_or_else(|| missing_tensor(&entry.name))?;
        if qt != want {
            return Err(format!(
                "'{}' is qt={qt}; the disagg lab shares only the canonical uniform QT{QT_GATE_UP}/QT{QT_DOWN} aggregate experts \
                 (no mixed codec, AWQ, slab or other format)",
                entry.name
            ));
        }
        if is_mtp(&entry.name) {
            if is_gate_up {
                mtp.0 = true;
            } else {
                mtp.1 = true;
            }
        } else {
            let layer = entry
                .layer
                .filter(|layer| *layer < layers)
                .ok_or_else(|| format!("routed expert '{}' has no trunk layer", entry.name))?;
            if is_gate_up {
                gate_up[layer] = true;
            } else {
                down[layer] = true;
            }
        }
        count += 1;
    }
    if !gate_up.iter().chain(down.iter()).all(|present| *present) || !(mtp.0 && mtp.1) {
        return Err(format!(
            "the artifact does not declare gate_up and down aggregates on all {layers} trunk layers and the MTP layer"
        ));
    }
    Ok(count)
}

/// Routed-expert bytes of each trunk layer, in layer order.
fn trunk_layer_expert_bytes(
    weights: &[WeightEntry],
    bytes_of: &dyn Fn(&WeightEntry) -> Option<u64>,
    layers: usize,
) -> R<Vec<u64>> {
    let mut per_layer = vec![0u64; layers];
    for entry in weights
        .iter()
        .filter(|entry| is_routed_expert(&entry.name) && !is_mtp(&entry.name))
    {
        let layer = entry
            .layer
            .filter(|layer| *layer < layers)
            .ok_or_else(|| format!("routed expert '{}' has no trunk layer", entry.name))?;
        let bytes = bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        per_layer[layer] = per_layer[layer].checked_add(bytes).ok_or("layer expert bytes overflow")?;
    }
    Ok(per_layer)
}

/// Consumer device bytes with the first `layers` trunk layers resident:
/// non-expert weights, those layers' experts, and the charged reserve.
fn consumer_required_bytes(non_expert: u64, layer_bytes: &[u64], layers: usize, charged: u64) -> R<u64> {
    let hot: u64 = layer_bytes.iter().take(layers).sum();
    non_expert
        .checked_add(hot)
        .and_then(|v| v.checked_add(charged))
        .ok_or_else(|| "consumer requirement overflows".to_string())
}

/// The deterministic conservative default: the most whole trunk layers whose
/// experts fit `free` beside the non-expert weights and the charged reserve.
/// `None` when not even the zero-layer placement fits. No timing input.
fn fit_resident_layers(free: u64, non_expert: u64, layer_bytes: &[u64], charged: u64) -> Option<usize> {
    let mut used = non_expert.checked_add(charged)?;
    if used > free {
        return None;
    }
    let mut layers = 0usize;
    for bytes in layer_bytes {
        match used.checked_add(*bytes) {
            Some(next) if next <= free => {
                used = next;
                layers += 1;
            }
            _ => break,
        }
    }
    Some(layers)
}

/// The loader's placement reserve for one device: the larger of the historical
/// measured formula and the bottom-up census, with native MTP attached.
fn reserve_for(
    gpu: &Gpu,
    config: &crate::Qwen4Config,
    weights: &[WeightEntry],
    bytes_of: &dyn Fn(&WeightEntry) -> Option<u64>,
    state_format: Qwen4StateFormat,
    scratch_mode: Qwen4ForwardScratchMode,
    max_seq: usize,
) -> R<residency::AutoReserve> {
    let _ = bytes_of;
    let backend = Qwen4KvBackend::automatic(gpu);
    let granularity = match backend {
        Qwen4KvBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|error| format!("VMM granularity: {error}"))?,
        _ => 1,
    };
    let qsa_format = state_format.qsa;
    let chunk_rows = qwen4_prefill_chunk_requested(&gpu.arch, max_seq);
    let context = crate::Qwen4ContextCommit::for_expert_reserve(
        backend,
        gpu.is_uma(),
        max_seq,
        chunk_rows,
        granularity,
    );
    let census_context = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
    let head = residency::language_head_dtype(weights).ok_or("manifest has no language head")?;
    let row_capture = crate::mtp_spec::native_mtp_row_capture(gpu, config).then_some(state_format.gdn);
    let mtp_for = |context: &crate::Qwen4ContextCommit| -> R<u64> {
        crate::mtp_spec::native_mtp_device_bytes(
            config,
            context,
            chunk_rows,
            crate::mtp_spec::native_mtp_proposal_drafts(LAB_MTP_MAX_K, &gpu.arch),
            head,
            row_capture,
            crate::mtp_spec::native_mtp_fill_scratch(gpu),
        )
        .ok_or_else(|| "native MTP device bytes overflow".to_string())
    };
    let gather_for = |context: &crate::Qwen4ContextCommit| -> R<Option<u64>> {
        rdna_compute::tensor_ops::qsa_gather_workspace_needed(gpu, qsa_format, state_format.qsa_f16_arena)
            .then(|| {
                rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                    config.num_key_value_heads,
                    context.tokens,
                )
                .map(|bytes| (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64))
                .ok_or_else(|| "QSA gather scratch size overflows".to_string())
            })
            .transpose()
    };
    let mtp_bytes = mtp_for(&context)?;
    let gather_bytes = gather_for(&context)?;
    let historical = residency::auto_vram_reserve(
        config,
        &context,
        chunk_rows,
        scratch_mode,
        state_format.layout(),
        Some(mtp_bytes),
        gather_bytes,
    )?;
    let state = residency::state_census(config, state_format, &census_context, chunk_rows, scratch_mode)?;
    let f16_shadows = if gpu.qwen4_f16_wmma_gfx1201() {
        residency::f16_shadow_bytes(weights, config.num_hidden_layers)?
    } else {
        0
    };
    let dense_gdn = gpu.arch == "gfx1201"
        && state_format.gdn == crate::GdnStateFormat::Q8
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
            == 10240;
    let routed_i4 = config.num_experts > 0
        && config.num_experts_per_tok > 0
        && config.moe_intermediate_size > 0
        && config.hidden_size % 256 == 0
        && config.moe_intermediate_size % 128 == 0;
    let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
        hidden_size: config.hidden_size,
        moe_intermediate_size: config.moe_intermediate_size,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        dense_gdn,
        routed_i4,
        include_short_tails: true,
    };
    let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk_rows, &dims).ok_or(
        "prefill workspace bytes cannot be computed (arithmetic overflow, or a dimension the kernel allocators reject)",
    )?;
    let census = residency::ReserveCensus {
        qsa_context: state.qsa_context,
        state: state.state,
        forward: state.forward,
        f16_shadows,
        workspace,
        qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_scratch_bytes(
            gpu,
            state_format.layout().index_dtype(),
        ) as u64,
        mtp: mtp_for(&census_context)?,
        gather: gather_for(&census_context)?.unwrap_or(0),
    };
    residency::AutoReserve::new(historical, census)
}

// ── identity ─────────────────────────────────────────────────────────

/// The PLE rows both sides read: embedded shards, or a sidecar's content
/// binding (header fingerprints and every segment digest).
fn ple_identity(hfq: &HfqFile, format: &str) -> String {
    match hfq.ple_rows() {
        None => format!("embedded:{format}"),
        Some(rows) => {
            let header = rows.header();
            let mut text = format!(
                "sidecar:{format}:{:?}:{:?}:{:?}:{}:{}:{}:{}",
                header.kind,
                header.order,
                header.row_encoding,
                header.padded_rows,
                header.ple_fingerprint,
                header.rank_payload_sha256,
                header.model_file_bytes
            );
            if let Some(table) = &header.table {
                text.push(':');
                text.push_str(&table.source_rows_sha256);
                for segment in &table.segment_sha256 {
                    text.push(':');
                    text.push_str(segment);
                }
            }
            text
        }
    }
}

/// The artifact both sides serve: the file's content digest (architecture,
/// metadata, tensor manifest, length), the PLE rows source with its sidecar
/// binding, and the recipe the manifest was declared under.
fn artifact_digest(hfq: &HfqFile, ple_format: &str, receipt: &Qwen4HfqmArtifact) -> String {
    let content = hfq.content_digest();
    let ple = ple_identity(hfq, ple_format);
    let recipe = format!("{:?}", receipt.manifest.identity.recipe);
    sha_hex(&[
        IDENTITY_VERSION.as_bytes(),
        b"artifact",
        &content,
        ple.as_bytes(),
        recipe.as_bytes(),
    ])
}

/// Freeze one side's `NumericalManifest` from what ACTUALLY loaded: the realized
/// prefill chunk and tails, the placements, the frozen host arithmetic, the
/// state formats and the decode/verify policy. Never caller-supplied.
fn identity_for(
    gpu: &Gpu,
    bundle: &Qwen4Bundle,
    role: DisaggRole,
    side: &SidePlan,
    plan: &Plan,
    policy: HostExecPolicy,
    max_seq: usize,
) -> R<NumericalManifest> {
    let chunk = bundle
        .spec_chunk_rows()
        .ok_or("the forward is not attached; no route identity")?;
    let exec = bundle.host_exec_identity();
    let role_tag = match role {
        DisaggRole::Producer => "producer",
        DisaggRole::Consumer => "consumer",
    };
    let mut placement = Sha256::new();
    let (mut resident, mut host) = (0usize, 0usize);
    for entry in side.weights.iter().filter(|entry| is_routed_expert(&entry.name)) {
        let tag = match entry.residency {
            WeightResidency::Resident => {
                resident += 1;
                "device"
            }
            _ => {
                host += 1;
                "host"
            }
        };
        placement.update(format!("{}|{:?}|{tag}\n", entry.name, entry.layer).as_bytes());
    }
    let placement = hex(&placement.finalize());
    let backend = Qwen4KvBackend::automatic(gpu);
    let drafts = crate::mtp_spec::native_mtp_proposal_drafts(LAB_MTP_MAX_K, &gpu.arch);
    let ngram = (
        hipfire_config::mtp_ngram_enabled_for_arch(crate::ARCH_ID, &gpu.arch),
        hipfire_config::mtp_ngram_copy_enabled_for_arch(crate::ARCH_ID, &gpu.arch),
    );
    let verify_cap = if policy == HostExecPolicy::Cpu && !exec.is_empty() {
        crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
    } else {
        crate::gpu_forward::QWEN4_SPEC_VERIFY_ROWS
    };
    let exec_tag = if exec.is_empty() { "gpu" } else { exec.as_str() };
    let policy_text = [
        format!("{IDENTITY_VERSION} policy {role_tag}"),
        format!("arch={}", gpu.arch),
        format!("kv_request={} state_request={}", plan.kv_request, plan.state_request),
        format!("state={:?}", side.state_format),
        format!("handoff_formats={:?}", crate::disagg::handoff_formats(&side.state_format)),
        format!("ple={}", plan.ple_identity),
        format!("recipe={:?}", side.receipt.manifest.identity.recipe),
        format!("scratch={:?}", side.scratch_mode),
        format!("chunk_requested={} chunk={chunk} max_seq={max_seq}", side.chunk_requested),
        format!("kv_backend={backend:?}"),
        format!("mtp_drafts={drafts} ngram={ngram:?} verify_rows_cap={verify_cap}"),
        format!("host_exec_policy={policy:?} exec={exec_tag}"),
        format!("placement={placement}"),
    ]
    .join("\n");
    let route_identity = format!(
        "{IDENTITY_VERSION}/{role_tag}/{}/exec={exec_tag}/experts={resident}d{host}h/place={}/chunk={chunk}/scratch={:?}/drafts={drafts}/verify<={verify_cap}",
        gpu.arch,
        &placement[..16],
        side.scratch_mode,
    );
    Ok(NumericalManifest {
        arch: gpu.arch.clone(),
        route_identity,
        artifact_digest: plan.artifact_digest.clone(),
        policy_digest: sha_hex(&[policy_text.as_bytes()]),
    })
}

impl DisaggPair {
    /// The model path both bundles were loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_expert_names_exclude_everything_else() {
        assert!(is_routed_expert("model.language_model.layers.3.mlp.experts.gate_up_proj"));
        assert!(is_routed_expert("mtp.layers.0.mlp.experts.down_proj"));
        assert!(!is_routed_expert("model.language_model.layers.3.mlp.shared_expert.down_proj"));
        assert!(is_mtp("mtp.layers.0.mlp.experts.down_proj"));
    }

    #[test]
    fn fit_takes_the_most_whole_layers_deterministically() {
        let layers = [10u64; 4];
        // 20 non-expert + 5 charged leaves room for exactly two layers in 46.
        assert_eq!(fit_resident_layers(46, 20, &layers, 5), Some(2));
        assert_eq!(fit_resident_layers(25, 20, &layers, 5), Some(0));
        assert_eq!(fit_resident_layers(24, 20, &layers, 5), None);
        assert_eq!(fit_resident_layers(u64::MAX, 20, &layers, 5), Some(4));
        assert_eq!(consumer_required_bytes(20, &layers, 2, 5), Ok(45));
        assert!(consumer_required_bytes(u64::MAX, &layers, 1, 1).is_err());
    }

    #[test]
    fn sha_hex_is_length_framed() {
        assert_ne!(sha_hex(&[b"ab", b"c"]), sha_hex(&[b"a", b"bc"]));
        assert_eq!(sha_hex(&[b"x"]).len(), 64);
    }

    #[test]
    fn one_line_collapses_whitespace() {
        assert_eq!(one_line("a\n  b\tc"), "a b c");
    }
}
