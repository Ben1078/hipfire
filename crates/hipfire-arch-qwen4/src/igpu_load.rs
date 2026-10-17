// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `memory.offload_exec=igpu` load admission for Qwen4 loads (the product rank
//! loader and the lab evaluators share this one implementation).
//!
//! The mode needs the host-mapped expert aggregates allocated
//! `hipHostMallocPortable`, so the second device can map them. That is decided
//! by [`crate::gpu_forward::igpu_engine::arm_igpu_for_load`], which
//! must run on the model `Gpu` **before the first expert byte is allocated**
//! (the hot-set tier materialisation and the manifest fulfillment). The caller
//! owns every fact that call needs: all of them are read from the parsed
//! artifact (its recipe and manifest) and the resolved placement, never from an
//! environment guess.
//!
//! A request the load cannot serve is not a load failure: the reason is one log
//! line and the load continues on the normal expert path with the `Gpu` left
//! unarmed (so host-mapped allocations keep their default flags and the arch's
//! attach step has nothing to attach).
//!
//! `hipfire-loader` (`load_qwen4_rank`) and the `qwen4_kld` evaluator both call
//! [`arm_qwen4_igpu`], so a lab load arms exactly as the daemon does.

use crate::expert_arena::ARENA_SUFFIX;
use crate::gpu_forward::igpu_engine::{self, Qwen4IgpuArm, Qwen4IgpuLoadInputs};
use crate::weights::{Qwen4TierRecipe, ROUTED_DOWN_DTYPE, ROUTED_GATE_UP_DTYPE};
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};
use std::collections::BTreeMap;

/// Every fact of one rank load the iGPU admission reads. Borrowed from the
/// caller's own locals at the point where placement is resolved.
pub struct Qwen4IgpuFacts<'a> {
    /// This load is a rank of an expert-parallel mesh.
    pub ep: bool,
    /// The admitted pipeline-parallel degree of the load (`ctx.pp`).
    pub pp: usize,
    /// The load holds routed experts as a per-expert tier / hot set (the only
    /// tier source is the hot-set plan).
    pub tiered: bool,
    /// The artifact carries AWQ scale sidecars of ROUTED EXPERT weights
    /// (`HfqFile::has_awq_expert_sidecars`); attention/trunk sidecars do not count.
    pub awq: bool,
    /// The artifact's profile is Flash-Next (the only profile the engine serves).
    pub flash_next: bool,
    /// The profile has a dense FFN (no routed experts at all).
    pub dense_ffn: bool,
    /// Trunk layer count; routed-expert entries at or past it (and every
    /// `mtp.` entry) are the MTP head's, which the engine does not run.
    pub trunk_layers: usize,
    /// The artifact's resolved recipe.
    pub recipe: &'a Qwen4TierRecipe,
    /// The manifest after expert placement (host residency applied).
    pub weights: &'a [WeightEntry],
}

/// One trunk layer's routed-expert entries as the manifest declares them.
#[derive(Default)]
struct RoutedLayer<'a> {
    gate_up: Option<&'a WeightEntry>,
    down: Option<&'a WeightEntry>,
    arena: Option<&'a WeightEntry>,
}

fn is_expert_sharded(entry: &WeightEntry) -> bool {
    matches!(entry.policy, ShardPolicy::ExpertSharded { .. })
}

/// What the placed manifest says about the trunk's host-placed experts.
#[derive(Debug, Default, PartialEq, Eq)]
struct HostExperts {
    /// Trunk layers with at least one host-mapped routed-expert entry.
    layers: usize,
    /// A host-placed layer is a slab arena.
    slab: bool,
    /// A host-placed layer's experts are not the QT44 gate/up + QT53 down pair.
    nonuniform: bool,
    /// Every host-placed layer holds BOTH aggregates, host-mapped, whole.
    whole_layers: bool,
}

fn host_experts(facts: &Qwen4IgpuFacts<'_>) -> HostExperts {
    let mut layers: BTreeMap<usize, RoutedLayer<'_>> = BTreeMap::new();
    for entry in facts.weights {
        let Some(layer) = entry.layer else { continue };
        if layer >= facts.trunk_layers || entry.name.starts_with("mtp.") {
            continue;
        }
        let slot = layers.entry(layer).or_default();
        if entry.name.ends_with(".mlp.experts.gate_up_proj") {
            slot.gate_up = Some(entry);
        } else if entry.name.ends_with(".mlp.experts.down_proj") {
            slot.down = Some(entry);
        } else if entry.name.ends_with(ARENA_SUFFIX) {
            slot.arena = Some(entry);
        }
    }
    let host = |entry: Option<&WeightEntry>| {
        entry.is_some_and(|entry| entry.residency == WeightResidency::HostMapped)
    };
    let mut found = HostExperts {
        whole_layers: true,
        ..HostExperts::default()
    };
    for layer in layers.values() {
        let any_host = host(layer.gate_up) || host(layer.down) || host(layer.arena);
        if !any_host {
            continue;
        }
        found.layers += 1;
        if layer.arena.is_some() {
            // The layer is one arena, never the aggregate pair.
            found.slab = true;
            continue;
        }
        let whole = host(layer.gate_up)
            && host(layer.down)
            && layer.gate_up.is_some_and(|entry| !is_expert_sharded(entry))
            && layer.down.is_some_and(|entry| !is_expert_sharded(entry));
        found.whole_layers &= whole;
        let uniform = layer
            .gate_up
            .zip(layer.down)
            .is_some_and(|(gate_up, down)| {
                gate_up.dtype == ROUTED_GATE_UP_DTYPE && down.dtype == ROUTED_DOWN_DTYPE
            });
        found.nonuniform |= !uniform;
    }
    found
}

/// The pure decision: the arch's [`Qwen4IgpuLoadInputs`] for this load, or the
/// one line saying why the load cannot use the mode at all (a reason the arch
/// inputs cannot express). A load whose inputs carry an exclusion (`ep`, `pp`,
/// `slab`, `tier`, `awq`, non-uniform or partial host layers) is still returned
/// as inputs: the arch owns those refusals and their wording.
pub fn qwen4_igpu_inputs(facts: &Qwen4IgpuFacts<'_>) -> Result<Qwen4IgpuLoadInputs, String> {
    if !facts.flash_next {
        return Err(
            "the profile is not Flash-Next (the engine serves the Flash-Next MoE prefill only)"
                .to_string(),
        );
    }
    if facts.dense_ffn {
        return Err("the profile has a dense FFN (no routed experts to run)".to_string());
    }
    let host = host_experts(facts);
    let inputs = Qwen4IgpuLoadInputs {
        ep: facts.ep,
        pp: facts.pp > 1,
        slab: host.slab || matches!(facts.recipe, Qwen4TierRecipe::FlashNextSlab { .. }),
        tier: facts.tiered,
        hotset: facts.tiered,
        awq: facts.awq,
        nonuniform_experts: host.nonuniform
            || matches!(facts.recipe, Qwen4TierRecipe::StockFp8),
        host_whole_layers_only: host.whole_layers,
    };
    let excluded = inputs.ep
        || inputs.pp
        || inputs.slab
        || inputs.tier
        || inputs.hotset
        || inputs.awq
        || inputs.nonuniform_experts
        || !inputs.host_whole_layers_only;
    if !excluded && host.layers == 0 {
        return Err(
            "no trunk layer's routed experts are host-placed, so nothing would run on the iGPU \
             (place experts in host memory, e.g. HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS)"
                .to_string(),
        );
    }
    Ok(inputs)
}

/// Logical byte length of every host-mapped TRUNK routed-expert aggregate
/// tensor (`gate_up_proj` / `down_proj`) of the placed manifest, in manifest
/// order: the sizes of the ONE portable pinned backing the engine layers'
/// aggregates become aligned subviews of (`Gpu::reserve_portable_host_pool`,
/// which applies its own per-tensor alignment and tail pad). Resident layers
/// and the MTP head's experts are not in it.
///
/// The pool serves an upload only while its exact size is still reserved, in
/// upload order, and the fulfillment uploads in manifest order: a host MTP
/// expert ahead of a trunk one could take a trunk slot, so that order is a
/// precondition (`Err` when violated).
fn host_pool_sizes(
    facts: &Qwen4IgpuFacts<'_>,
    bytes_of: &dyn Fn(&WeightEntry) -> Option<u64>,
) -> Result<Vec<usize>, String> {
    let mut sizes = Vec::new();
    let mut mtp_seen = false;
    for entry in facts.weights {
        if entry.residency != WeightResidency::HostMapped
            || !(entry.name.ends_with(".mlp.experts.gate_up_proj")
                || entry.name.ends_with(".mlp.experts.down_proj"))
        {
            continue;
        }
        let trunk = !entry.name.starts_with("mtp.")
            && entry.layer.is_some_and(|layer| layer < facts.trunk_layers);
        if !trunk {
            mtp_seen = true;
            continue;
        }
        if mtp_seen {
            return Err(format!(
                "host MTP experts precede trunk expert '{}' in the manifest",
                entry.name
            ));
        }
        let bytes = bytes_of(entry)
            .ok_or_else(|| format!("no payload size for host expert tensor '{}'", entry.name))?;
        sizes.push(
            usize::try_from(bytes)
                .map_err(|_| format!("host expert tensor '{}' is too large", entry.name))?,
        );
    }
    Ok(sizes)
}

/// Load-time admission and arming. Call once per rank load, **after** expert
/// placement is resolved and **before** any expert byte is allocated.
///
/// Returns `None` without a log line and without touching `gpu` unless
/// `memory.offload_exec=igpu`. Otherwise either
/// - arms portable host-mapped allocation on `gpu`, reserves the one pinned
///   backing the host-placed trunk expert aggregates then suballocate (from
///   `bytes_of`, the resolved plan's per-tensor payload sizes), logs the arm
///   and returns the second device; or
/// - logs ONE line (the loader's reason, or the arch's refusal) and returns
///   `None` with `gpu` not armed (an armed `Gpu` left by an earlier load is
///   disarmed): the load continues on the normal path and the arch attach step
///   stays silent.
pub fn arm_qwen4_igpu(
    gpu: &mut rdna_compute::Gpu,
    facts: &Qwen4IgpuFacts<'_>,
    bytes_of: &dyn Fn(&WeightEntry) -> Option<u64>,
) -> Option<Qwen4IgpuArm> {
    if !igpu_engine::qwen4_igpu_enabled() {
        return None;
    }
    let inputs = match qwen4_igpu_inputs(facts) {
        Ok(inputs) => inputs,
        Err(reason) => {
            igpu_engine::decline_igpu_for_load(&one_line(&reason));
            gpu.disarm_portable_host_mapped();
            return None;
        }
    };
    // A refusal is logged (and the `Gpu` disarmed) by the arch call itself.
    let arm = igpu_engine::arm_igpu_for_load(gpu, &inputs)?;
    let reserved = host_pool_sizes(facts, bytes_of).and_then(|sizes| {
        gpu.reserve_portable_host_pool(&sizes)
            .map(|capacity| (sizes.len(), capacity))
            .map_err(|error| format!("host expert pool: {error}"))
    });
    match reserved {
        Ok((tensors, capacity)) => {
            eprintln!(
                "  [igpu-exec] qwen4: armed on device {} {}; {tensors} host trunk expert tensors in one {} MiB portable pinned pool",
                arm.device,
                arm.arch,
                capacity >> 20
            );
            Some(arm)
        }
        Err(reason) => {
            igpu_engine::decline_igpu_for_load(&one_line(&reason));
            gpu.disarm_portable_host_mapped();
            None
        }
    }
}

/// Clean up after a rank load that failed: release the portable pinned pool if
/// nothing was carved from it (a pool whose tensors the failed load already
/// freed is released by their last free) and drop the armed state, so a retry on
/// the same `Gpu` starts from the default flags and can reserve a fresh pool.
/// A no-op unless `memory.offload_exec=igpu`.
pub fn release_after_failed_load(gpu: &mut rdna_compute::Gpu) {
    if !igpu_engine::qwen4_igpu_enabled() {
        return;
    }
    // A quarantined `Gpu` retains everything and refuses every release.
    if gpu.quarantined().is_some() {
        gpu.disarm_portable_host_mapped();
        return;
    }
    if let Err(error) = gpu.release_unused_host_pool() {
        eprintln!(
            "  [igpu-exec] qwen4: host expert pool kept after the failed load ({})",
            one_line(&error.to_string())
        );
    }
    gpu.disarm_portable_host_mapped();
}

/// A reason on one line, whatever its own formatting.
fn one_line(reason: &str) -> String {
    reason.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    const TRUNK: usize = 4;

    fn entry(name: &str, layer: usize, dtype: DType, residency: WeightResidency) -> WeightEntry {
        let mut entry = WeightEntry::layer(
            name,
            layer,
            vec![8, 8, 8],
            dtype,
            ShardPolicy::Replicate,
        );
        entry.residency = residency;
        entry
    }

    /// Per-layer aggregate pair, host-mapped from layer `vram_layers` on; the
    /// MTP layer (index `TRUNK`) is always host.
    fn manifest(vram_layers: usize) -> Vec<WeightEntry> {
        let mut weights = Vec::new();
        for layer in 0..=TRUNK {
            let residency = if layer < vram_layers {
                WeightResidency::Resident
            } else {
                WeightResidency::HostMapped
            };
            let prefix = if layer == TRUNK {
                "mtp.layers.0".to_string()
            } else {
                format!("model.language_model.layers.{layer}")
            };
            weights.push(entry(
                &format!("{prefix}.mlp.experts.gate_up_proj"),
                layer,
                ROUTED_GATE_UP_DTYPE,
                residency,
            ));
            weights.push(entry(
                &format!("{prefix}.mlp.experts.down_proj"),
                layer,
                ROUTED_DOWN_DTYPE,
                residency,
            ));
        }
        weights
    }

    fn packed() -> Qwen4TierRecipe {
        Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: DType::Q8_0,
            mtp_source: false,
        }
    }

    fn facts<'a>(recipe: &'a Qwen4TierRecipe, weights: &'a [WeightEntry]) -> Qwen4IgpuFacts<'a> {
        Qwen4IgpuFacts {
            ep: false,
            pp: 1,
            tiered: false,
            awq: false,
            flash_next: true,
            dense_ffn: false,
            trunk_layers: TRUNK,
            recipe,
            weights,
        }
    }

    #[test]
    fn plain_host_placed_load_is_the_plain_inputs() {
        let weights = manifest(2);
        let recipe = packed();
        assert_eq!(
            qwen4_igpu_inputs(&facts(&recipe, &weights)),
            Ok(Qwen4IgpuLoadInputs::PLAIN)
        );
    }

    #[test]
    fn mtp_head_experts_alone_are_not_a_host_trunk() {
        // Every trunk layer resident: only the MTP head's experts are host.
        let weights = manifest(TRUNK);
        let recipe = packed();
        let error = qwen4_igpu_inputs(&facts(&recipe, &weights)).unwrap_err();
        assert!(error.contains("nothing would run on the iGPU"), "{error}");
    }

    #[test]
    fn fully_resident_load_has_no_host_layer() {
        let weights = manifest(TRUNK + 1);
        let recipe = packed();
        assert!(qwen4_igpu_inputs(&facts(&recipe, &weights)).is_err());
    }

    #[test]
    fn every_exclusion_is_derived_from_the_load() {
        let weights = manifest(1);
        let recipe = packed();
        let base = Qwen4IgpuLoadInputs::PLAIN;
        let case = |mutate: &dyn Fn(&mut Qwen4IgpuFacts<'_>)| {
            let mut facts = facts(&recipe, &weights);
            mutate(&mut facts);
            qwen4_igpu_inputs(&facts).unwrap()
        };
        assert_eq!(case(&|f| f.ep = true), Qwen4IgpuLoadInputs { ep: true, ..base });
        assert_eq!(case(&|f| f.pp = 2), Qwen4IgpuLoadInputs { pp: true, ..base });
        assert_eq!(
            case(&|f| f.tiered = true),
            Qwen4IgpuLoadInputs {
                tier: true,
                hotset: true,
                ..base
            }
        );
        assert_eq!(case(&|f| f.awq = true), Qwen4IgpuLoadInputs { awq: true, ..base });
    }

    #[test]
    fn exclusion_wins_over_a_missing_host_layer() {
        // A hot-set load drops the trunk aggregates from the manifest: the arch
        // names the tier refusal, not the loader's "no host layer".
        let weights = manifest(TRUNK + 1);
        let recipe = packed();
        let mut load = facts(&recipe, &weights);
        load.tiered = true;
        assert_eq!(
            qwen4_igpu_inputs(&load),
            Ok(Qwen4IgpuLoadInputs {
                tier: true,
                hotset: true,
                ..Qwen4IgpuLoadInputs::PLAIN
            })
        );
    }

    #[test]
    fn non_flash_next_and_dense_profiles_are_loader_refusals() {
        let weights = manifest(1);
        let recipe = packed();
        let mut load = facts(&recipe, &weights);
        load.flash_next = false;
        assert!(qwen4_igpu_inputs(&load).unwrap_err().contains("Flash-Next"));
        let mut load = facts(&recipe, &weights);
        load.dense_ffn = true;
        assert!(qwen4_igpu_inputs(&load).unwrap_err().contains("dense"));
    }

    #[test]
    fn slab_recipe_and_arena_entries_are_slab() {
        let mut weights = manifest(1);
        let recipe = packed();
        // Replace layer 2's aggregates with a host-mapped arena.
        weights.retain(|entry| entry.layer != Some(2));
        weights.push(entry(
            &format!("model.language_model.layers.2{ARENA_SUFFIX}"),
            2,
            DType::Q8_0,
            WeightResidency::HostMapped,
        ));
        assert!(qwen4_igpu_inputs(&facts(&recipe, &weights)).unwrap().slab);
    }

    #[test]
    fn non_qt44_qt53_host_layer_is_nonuniform() {
        let mut weights = manifest(1);
        let recipe = packed();
        for entry in weights.iter_mut() {
            if entry.layer == Some(2) && entry.name.ends_with("gate_up_proj") {
                entry.dtype = DType::BF16;
            }
        }
        assert!(
            qwen4_igpu_inputs(&facts(&recipe, &weights))
                .unwrap()
                .nonuniform_experts
        );
        // A resident layer's format is not the engine's concern.
        let mut weights = manifest(3);
        for entry in weights.iter_mut() {
            if entry.layer == Some(0) {
                entry.dtype = DType::BF16;
            }
        }
        assert!(
            !qwen4_igpu_inputs(&facts(&recipe, &weights))
                .unwrap()
                .nonuniform_experts
        );
    }

    #[test]
    fn stock_fp8_recipe_is_nonuniform() {
        let weights = manifest(1);
        let recipe = Qwen4TierRecipe::StockFp8;
        assert!(
            qwen4_igpu_inputs(&facts(&recipe, &weights))
                .unwrap()
                .nonuniform_experts
        );
    }

    #[test]
    fn half_host_layer_is_not_whole() {
        let mut weights = manifest(1);
        let recipe = packed();
        for entry in weights.iter_mut() {
            if entry.layer == Some(2) && entry.name.ends_with("down_proj") {
                entry.residency = WeightResidency::Resident;
            }
        }
        assert!(
            !qwen4_igpu_inputs(&facts(&recipe, &weights))
                .unwrap()
                .host_whole_layers_only
        );
        // A missing aggregate of a host layer is not whole either.
        let mut weights = manifest(1);
        weights.retain(|entry| !(entry.layer == Some(2) && entry.name.ends_with("down_proj")));
        assert!(
            !qwen4_igpu_inputs(&facts(&recipe, &weights))
                .unwrap()
                .host_whole_layers_only
        );
    }

    #[test]
    fn expert_sharded_host_layer_is_not_whole() {
        let mut weights = manifest(1);
        let recipe = packed();
        for entry in weights.iter_mut() {
            if entry.layer == Some(2) {
                entry.policy = ShardPolicy::ExpertSharded {
                    n_experts: 512,
                    assign: hipfire_runtime::tp_shard::ExpertAssign::Stride,
                };
            }
        }
        assert!(
            !qwen4_igpu_inputs(&facts(&recipe, &weights))
                .unwrap()
                .host_whole_layers_only
        );
    }

    #[test]
    fn attention_only_awq_does_not_reject_but_expert_awq_does() {
        use hipfire_runtime::hfq::is_awq_expert_sidecar_name;
        let weights = manifest(1);
        let recipe = packed();
        let inputs_for = |tensor_names: &[&str]| {
            let mut load = facts(&recipe, &weights);
            load.awq = tensor_names.iter().any(|name| is_awq_expert_sidecar_name(name));
            qwen4_igpu_inputs(&load).unwrap()
        };
        let attention = "model.language_model.layers.3.self_attn.q_proj.weight.awq_scale.weight";
        let expert = "model.language_model.layers.3.mlp.experts.gate_up_proj.awq_scale.weight";
        assert_eq!(inputs_for(&[attention]), Qwen4IgpuLoadInputs::PLAIN);
        assert!(inputs_for(&[attention, expert]).awq);
    }

    #[test]
    fn one_line_collapses_a_multi_line_reason() {
        let line = one_line("memory.offload_exec=igpu refused:\n  expert parallelism\n");
        assert_eq!(line, "memory.offload_exec=igpu refused: expert parallelism");
    }

    #[test]
    fn pool_sizes_are_the_host_trunk_aggregates_in_manifest_order() {
        // Layer 0 resident; layers 1..=3 host-mapped; the MTP head is host too
        // but is not in the pool.
        let weights = manifest(1);
        let recipe = packed();
        let sizes = host_pool_sizes(&facts(&recipe, &weights), &|entry| {
            Some(if entry.name.ends_with("gate_up_proj") { 300 } else { 100 })
        })
        .unwrap();
        assert_eq!(sizes, [300, 100].repeat(3));
    }

    #[test]
    fn pool_sizes_skip_resident_layers_and_reject_unsized_entries() {
        let recipe = packed();
        let weights = manifest(TRUNK + 1);
        assert_eq!(
            host_pool_sizes(&facts(&recipe, &weights), &|_| Some(1)),
            Ok(Vec::new())
        );
        let weights = manifest(1);
        assert!(host_pool_sizes(&facts(&recipe, &weights), &|_| None).is_err());
    }

    #[test]
    fn pool_sizes_refuse_a_host_mtp_expert_ahead_of_the_trunk() {
        let recipe = packed();
        let mut weights = manifest(1);
        // Move the MTP head's aggregates to the front of the manifest.
        let (mtp, trunk): (Vec<_>, Vec<_>) = weights
            .drain(..)
            .partition(|entry| entry.name.starts_with("mtp."));
        weights = mtp.into_iter().chain(trunk).collect();
        let error = host_pool_sizes(&facts(&recipe, &weights), &|_| Some(1)).unwrap_err();
        assert!(error.contains("MTP"), "{error}");
    }

    /// The `qwen4_kld` lab load: every expert starts resident, then
    /// `place_routed_experts` (`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N`) host-maps
    /// layers `N..` and the MTP head, and the load arms through this module like
    /// the daemon. No ep/pp/tier/awq: plain inputs and the host trunk pool.
    #[test]
    fn lab_load_with_env_placed_experts_arms_plain_with_the_host_trunk_pool() {
        let recipe = packed();
        let mut weights = manifest(TRUNK + 1);
        assert_eq!(
            qwen4_igpu_inputs(&facts(&recipe, &weights)).unwrap_err(),
            "no trunk layer's routed experts are host-placed, so nothing would run on the iGPU \
             (place experts in host memory, e.g. HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS)"
        );
        let moved = crate::expert_residency::place_routed_experts(&mut weights, 2);
        // Trunk layers 2 and 3 plus the MTP head, two aggregates each.
        assert_eq!(moved, 6);
        let load = facts(&recipe, &weights);
        assert_eq!(qwen4_igpu_inputs(&load), Ok(Qwen4IgpuLoadInputs::PLAIN));
        let sizes = host_pool_sizes(&load, &|entry| {
            Some(if entry.name.ends_with("gate_up_proj") { 300 } else { 100 })
        })
        .unwrap();
        assert_eq!(sizes, [300, 100].repeat(2));
    }
}
