// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single-device, no-MTP Qwen4 load the S1 validation tools share
//! (`qwen4_slowlink_gate`, `qwen4_greedy_ids`, `qwen4_kld`, the parity CLI
//! and the hardware tests).
//!
//! It is the product loader's admission-first order, without the product's
//! MTP, PLE hot store and hot set: the S1 profile is selected FIRST
//! ([`crate::slowlink_admission::select_load_profile`], from the exact
//! placement census [`crate::expert_residency::placement_inputs`]), then the
//! routed-expert placement it implies, then the bundle, with the profile
//! attached before the forward is built. Candidate and baseline arms differ
//! only in `HIPFIRE_QWEN4_SLOWLINK_PREFILL`, so they are separate processes
//! (the config snapshot is read once).

use crate::kv_backend::Qwen4KvBackend;
use rdna_compute::Gpu;
use std::path::Path;

use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};

/// What the shared loader built.
pub struct SlowlinkLoaded {
    pub bundle: crate::bundle::Qwen4Bundle,
    pub vocab: usize,
    pub eos_token: u32,
    pub num_layers: usize,
    pub arch_id: u32,
    /// Routed-expert tensors host-mapped by the placement (0 = all resident).
    pub host_mapped_tensors: usize,
    /// Trunk layers whose routed experts are resident in VRAM.
    pub resident_layers: usize,
    /// How the placement was decided: `slowlink-profile` (the S1 admission's
    /// joint placement), `incumbent-auto` (the product's automatic placement
    /// at the incumbent chunk), `explicit-env` (a fixed
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, which declines S1) or
    /// `unified-memory` (everything resident).
    pub placement: &'static str,
    /// The S1 profile was admitted and attached to the bundle.
    pub slowlink_profile: bool,
    pub ple_format: String,
    pub tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    /// The artifact's chat template (rendering is the caller's).
    pub chat_template: Option<String>,
}

/// The fulfillment payload of one manifest entry: a slab layer arena (one
/// container span), a checked source range (unified memory or an external
/// entry) or the mapped bytes borrowed (a resident or host-mapped discrete
/// entry), as the product loader reads them.
fn payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    recipe: &crate::weights::Qwen4TierRecipe,
    entry: &WeightEntry,
    use_ranges: bool,
) -> Result<hipfire_runtime::model_source::SourcePayload<'a>, String> {
    use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
    if let Some(payload) = crate::expert_arena::recipe_arena_payload(source, recipe, entry) {
        return payload;
    }
    if use_ranges || entry.residency.is_external() {
        return source
            .tensor_range(&entry.name)
            .map_err(|error| error.to_string())?
            .map(SourcePayload::Range)
            .ok_or_else(|| {
                if entry.residency.is_external() {
                    format!("missing external tensor '{}'", entry.name)
                } else {
                    format!("missing resident tensor '{}'", entry.name)
                }
            });
    }
    let (info, bytes) = source
        .tensor_data(&entry.name)
        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
    Ok(SourcePayload::Borrowed { info, bytes })
}

/// Load one real Qwen4 HFQM artifact the way the product loader does for a
/// single-device, no-MTP load: arena payloads, the S1 admission first, then
/// the placement it implies (the profile's resident layers when S1 is
/// admitted, else the incumbent automatic placement at the placement chunk),
/// the borrowed host-mapped payload fallback, no PLE hot store.
/// `session_cache_budget = Some(bytes)` attaches a session cache of that
/// budget. An explicit `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` pins the placement
/// and declines S1, exactly as in the product.
pub fn slowlink_load_bundle(
    gpu: &mut Gpu,
    path: &Path,
    max_seq: usize,
    session_cache_budget: Option<u64>,
) -> Result<SlowlinkLoaded, String> {
    use crate::expert_residency as residency;
    use crate::slowlink_admission as s1;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(path)
        .map_err(|error| format!("open HFQM {}: {error}", path.display()))?;
    let choice = hipfire_runtime::ple_sidecar::attach_validation_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
    )
    .map_err(|error| format!("qwen4 PLE rows: {error}"))?;
    let receipt = crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let trunk_overlay = crate::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
        .map_err(|error| format!("qwen4 trunk overlay: {error}"))?;
    let scratch_mode = crate::artifact::source_forward_scratch_mode(
        gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    let unified = gpu.is_uma();
    if unified {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, gpu);
    let npu_identity = crate::npu_trunk::model_index_identity(&hfq, &gpu.arch);
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let chat_template = hfq.chat_template();
    let arch_id = hfq.arch_id;
    // The session-cache domain binds the artifact, tokenizer and device; it
    // is read from the HFQ file before the source takes ownership of it.
    let cache_domain = session_cache_budget.map(|_| {
        hipfire_runtime::serve_contract::CacheDomain::for_model(
            &hfq,
            &tokenizer,
            None,
            "qwen4",
            gpu.device_id,
        )
    });
    let recipe = receipt.manifest.identity.recipe.clone();
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let state_format = crate::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        gpu,
        &receipt.config,
    )?;
    let backend = Qwen4KvBackend::automatic(gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|error| format!("qwen4: VMM granularity: {error}"))?,
        Qwen4KvBackend::Legacy => 1,
    };
    // Entry sizes as the product loader sizes them: a slab layer arena from
    // the recipe's plan, anything else from the mapped tensor.
    let bytes_of = |entry: &WeightEntry| -> Option<u64> {
        if let crate::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } = &recipe {
            if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
                return entry
                    .layer
                    .and_then(|layer| arenas.layer(layer))
                    .map(|layer| layer.bytes as u64);
            }
        }
        use hipfire_runtime::model_source::ModelSource as _;
        source
            .tensor_data(&entry.name)
            .map(|(_, bytes)| bytes.len() as u64)
    };
    let routed_experts = !receipt.config.ffn.is_dense();
    let mut weights = receipt.manifest.weights.clone();
    let explicit = residency::expert_vram_layers_from_env()?;
    let (resident_layers, placement, profile) = if unified {
        (receipt.config.num_hidden_layers, "unified-memory", None)
    } else if let Some(residency::ExpertVramLayers::Layers(layers)) = explicit {
        residency::place_routed_experts(&mut weights, layers);
        (layers, "explicit-env", None)
    } else {
        // Automatic placement (an explicit `auto` declines S1 through the
        // shape's `explicit_vram_layers`, as in the product).
        let chunk_env = hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS").ok();
        let shape = s1::LoadShape::from_process(
            &gpu.arch,
            false,
            s1::StructuralShape {
                expert_parallel: false,
                pipeline_band: false,
                packed_mq4: matches!(
                    recipe,
                    crate::weights::Qwen4TierRecipe::FlashNextPacked { .. }
                ),
                routed_experts,
                expert_keep: receipt.config.expert_keep.is_some(),
                hotset_sidecar: false,
                whole_first: false,
            },
            max_seq,
            chunk_env.as_deref(),
        )?;
        let load = s1::SingleDeviceLoad {
            gpu: &*gpu,
            config: &receipt.config,
            weights: &weights,
            bytes_of: &bytes_of,
            backend,
            max_seq,
            vmm_granularity,
            native_mtp: false,
            mtp_kept: false,
            max_k: 0,
            state_format,
            scratch_mode,
        };
        let profile = s1::select_load_profile(
            s1::SlowlinkMode::from_env(),
            &shape,
            || crate::host_link::host_link(load.gpu),
            || s1::single_device_facts(&load),
        )?;
        let layers = match &profile {
            Some(p) => p.resident_layers,
            None => {
                // The incumbent's automatic placement at its own chunk.
                let chunk = crate::gpu_forward::qwen4_prefill_chunk_placement_requested(
                    &load.gpu.arch,
                    max_seq,
                );
                let inputs = residency::placement_inputs(&residency::PlacementInputsArgs {
                    gpu: load.gpu,
                    config: load.config,
                    weights: load.weights,
                    bytes_of: load.bytes_of,
                    backend,
                    use_ranges: false,
                    max_seq,
                    vmm_granularity,
                    mtp_kept: false,
                    max_k: 0,
                    routed_experts,
                    state_format,
                    scratch_mode,
                    ranks: 1,
                    chunk_rows: chunk,
                })?;
                let (free, _) = load
                    .gpu
                    .device_mem_info()
                    .map_err(|error| format!("qwen4: device memory query: {error}"))?;
                let charges = residency::PlacementCharges::new(inputs.layer_experts, false)?;
                residency::auto_placement_layers(
                    free as u64,
                    inputs.non_expert,
                    inputs.layer_experts,
                    load.config.num_hidden_layers,
                    inputs.reserve.bytes(),
                    &charges,
                )?
            }
        };
        residency::place_routed_experts(&mut weights, layers);
        let placement = if profile.is_some() {
            "slowlink-profile"
        } else {
            "incumbent-auto"
        };
        (layers, placement, profile)
    };
    let host_mapped_tensors = weights
        .iter()
        .filter(|entry| entry.residency == WeightResidency::HostMapped)
        .count();
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        gpu,
        expected,
        |entry| payload(&source, &recipe, entry, unified),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let vocab = receipt.config.vocab_size;
    let eos_token = receipt.config.eos_token_id;
    let num_layers = receipt.config.num_hidden_layers;
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_ple_rows(
        receipt.config,
        &recipe,
        transaction,
        &receipt.placements,
        gpu,
        max_seq,
        receipt.ple,
        receipt.ple_rows,
        state_format,
        backend,
        None,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    bundle.attach_npu_trunk_identity(npu_identity);
    if let Some(overlay) = trunk_overlay {
        bundle
            .attach_trunk_overlay(overlay)
            .map_err(|error| format!("qwen4 trunk overlay: {error}"))?;
    }
    let slowlink_profile = profile.is_some();
    // The admitted profile is attached before the forward is built: the
    // forward sizes its tile scratch and arenas from it.
    if let Some(profile) = profile {
        bundle.attach_slowlink_profile(profile);
    }
    bundle
        .attach_forward(gpu, max_seq, scratch_mode)
        .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
    if let Some(domain) = cache_domain {
        let cache = hipfire_runtime::session_cache::SessionCache::new(
            domain,
            session_cache_budget.unwrap_or(0),
        );
        bundle.attach_session_cache(cache);
    }
    eprintln!(
        "qwen4 slowlink load: placement={placement} resident layers={resident_layers} \
         host-mapped tensors={host_mapped_tensors} slowlink profile={slowlink_profile}"
    );
    Ok(SlowlinkLoaded {
        bundle,
        vocab,
        eos_token,
        num_layers,
        arch_id,
        host_mapped_tensors,
        resident_layers,
        placement,
        slowlink_profile,
        ple_format: choice.format.name().to_string(),
        tokenizer,
        chat_template,
    })
}
