// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Routed-expert residency for cards whose VRAM cannot hold every expert.
//!
//! Flash-Next's routed experts are ~65 GB, the rest of the model ~5 GB. On a
//! discrete card the experts of trunk layers at or past the VRAM layer count
//! ([`EXPERT_VRAM_LAYERS_ENV`]) and the MTP layer's are fulfilled into pinned,
//! device-mapped host RAM instead of VRAM. The sealed MoE kernels already read
//! every expert through per-layer pointer tables, so a host-mapped layer is
//! read over PCIe (zero-copy) with no kernel or dispatch change: a decode
//! token reads only its routed experts' bytes from host RAM.

use crate::gpu_forward::Qwen4ForwardScratchMode;
use crate::weights::{Qwen4BandScope, Qwen4EpPlacement};
use crate::Qwen4Config;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};
use rdna_compute::uma_memory::{
    amdgpu_gtt_used_bytes, read_sysfs_u64, TTM_PAGES_LIMIT, TTM_PAGE_BYTES,
};

/// `N` keeps the routed experts of trunk layers `0..N` in VRAM; `auto` picks
/// the largest `N` that fits the card's free VRAM. Unset keeps every expert
/// resident where they fit ([`resolve_expert_vram_layers`]) and is `auto`
/// otherwise. Set, it also keeps host memory out of reclaim from before the
/// HIP runtime loads (hip-bridge owns the name), as a process about to load
/// a Qwen4 model on a discrete GPU does unset
/// ([`keeps_host_memory_out_of_reclaim`]).
pub const EXPERT_VRAM_LAYERS_ENV: &str = hip_bridge::QWEN4_EXPERT_VRAM_LAYERS_ENV;

/// VRAM left free by `auto` beyond the resident non-expert weights: forward
/// scratch, KV and state for [`AUTO_VRAM_RESERVE_MAX_SEQ`] tokens plus
/// headroom, without native MTP. Measured on a gfx1201 R9700 at N=16 with a
/// [`AUTO_VRAM_RESERVE_CHUNK`]-row prefill chunk: every context-sized buffer
/// is allocated at load, and a 32,700-token prefill plus decode at that
/// context peaked at 6,199 MiB beyond the non-expert and expert weights,
/// 457 MiB under this reserve. [`auto_vram_reserve`] adds what a longer
/// context, a different chunk and native MTP take; a shorter context keeps it.
pub const AUTO_VRAM_RESERVE_BYTES: u64 = 6656 << 20;

/// Device memory the HIP runtime holds on an exact gfx1201 beyond every
/// allocation the load and the forward make: code objects, HSA queues and
/// their context-save areas, kernel scratch and the ROCr pools. Measured on an
/// R9700 as the device-wide sysfs `mem_info_vram_used` minus every live
/// allocation `rocprofv3 --memory-allocation-trace` reports, beyond the
/// 57 MiB idle baseline:
/// - pruned W3-slab pack fully resident, compact scratch, 1024-row chunk:
///   929 MiB after load, 1,330 MiB at the peak of a 32K-token request
///   (receipt `slab-scratch-notes/rcpt/h1-proof-run1/p1024`);
/// - shipped `mq3-xts` with 30 routed layers in pinned host RAM, full scratch,
///   1024-row chunk: 974 MiB after load, 1,373 MiB at the peak (receipt
///   `slab-scratch-notes/rcpt/h2a-run2/c1024`). The same load at 4096 and
///   8192 rows (no trace; the traced allocations rebased by the census
///   difference) peaks at about 1,380 and 1,383 MiB, at 256 rows at about
///   956 MiB.
///
/// The growth after load appears only once a request runs at least 512 rows.
/// The constant is the largest peak rounded up to 1,408 MiB.
/// The historical floor's stale F32 context used to cover it; with the floor
/// retired on this route the census carries it as [`ReserveCensus::runtime`].
pub const QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES: u64 = 1408 << 20;

/// The measured HIP runtime overhead the census charges: only where it was
/// measured, an exact gfx1201 discrete (not unified-memory) single-device
/// load. `None` elsewhere, where the historical floor still applies.
pub fn runtime_overhead_bytes(arch: &str, unified_memory: bool, ranks: usize) -> Option<u64> {
    (arch == "gfx1201" && !unified_memory && ranks == 1)
        .then_some(QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES)
}

/// Context [`AUTO_VRAM_RESERVE_BYTES`] was measured at (`hipfire run`'s
/// legacy-KV `max_seq`).
pub const AUTO_VRAM_RESERVE_MAX_SEQ: usize = 32768;

/// Prefill chunk [`AUTO_VRAM_RESERVE_BYTES`] was measured at; its forward
/// scratch then carried speculative logits and argmax for every chunk row.
pub const AUTO_VRAM_RESERVE_CHUNK: usize = 1536;

/// Host RAM that must remain available after the pinned experts are placed.
pub const HOST_RAM_HEADROOM_BYTES: u64 = 4 << 30;

/// Slack `Gpu::upload_raw_host_mapped` adds to every host-mapped tensor.
pub const HOST_MAPPED_PAD_BYTES: u64 = 1 << 20;

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertVramLayers {
    Layers(usize),
    Auto,
}

/// Parse [`EXPERT_VRAM_LAYERS_ENV`]. `Ok(None)` = unset
/// ([`resolve_expert_vram_layers`] picks the placement).
pub fn expert_vram_layers_from_env() -> Result<Option<ExpertVramLayers>, String> {
    match hipfire_config::developer_var(EXPERT_VRAM_LAYERS_ENV) {
        Ok(value) => parse_expert_vram_layers(&value).map(Some),
        Err(_) => Ok(None),
    }
}

/// The placement for a discrete-GPU load: an explicit `policy` always wins.
/// Unset (`None`) keeps every routed expert resident (`Ok(None)`) when
/// `fits_resident` reports that the whole expert footprint, the non-expert
/// weights and the `auto` reserve fit in VRAM, and is `auto` otherwise.
/// `fits_resident` runs only when unset.
pub fn resolve_expert_vram_layers(
    policy: Option<ExpertVramLayers>,
    fits_resident: impl FnOnce() -> Result<bool, String>,
) -> Result<Option<ExpertVramLayers>, String> {
    match policy {
        Some(policy) => Ok(Some(policy)),
        None => Ok((!fits_resident()?).then_some(ExpertVramLayers::Auto)),
    }
}

/// A profile without routed experts (a dense FFN) has nothing to place in
/// host RAM: its gate/up/down projections are ordinary resident weights, so
/// the placement is always fully resident (`resident_split` counts them as
/// non-expert bytes, `routed_expert_bytes` is 0).  An explicit
/// [`EXPERT_VRAM_LAYERS_ENV`] request (a layer count or `auto`) is refused
/// instead of being honored by moving dense weights as fake experts; unset
/// is fine.
pub fn refuse_host_expert_request(
    policy: Option<ExpertVramLayers>,
    profile: &str,
) -> Result<(), String> {
    let Some(policy) = policy else {
        return Ok(());
    };
    let value = match policy {
        ExpertVramLayers::Layers(layers) => layers.to_string(),
        ExpertVramLayers::Auto => "auto".to_string(),
    };
    Err(format!(
        "{EXPERT_VRAM_LAYERS_ENV}={value} requests host-mapped routed experts, but profile \
         {profile} has a dense FFN with no routed experts (its gate/up/down are resident \
         weights); unset {EXPERT_VRAM_LAYERS_ENV}"
    ))
}

/// Whether every routed expert (`expert_bytes`), the non-expert weights and
/// `reserve` ([`auto_vram_reserve`]) fit in `free_vram`.
pub fn fits_fully_resident(
    free_vram: u64,
    non_expert_bytes: u64,
    expert_bytes: u64,
    reserve: u64,
) -> bool {
    non_expert_bytes
        .checked_add(expert_bytes)
        .and_then(|bytes| bytes.checked_add(reserve))
        .is_some_and(|need| need <= free_vram)
}

/// Whether a process about to load the model with HFQ `arch_id` on the GPUs
/// `device_archs` (every card it may use) keeps host memory out of reclaim
/// before the HIP runtime loads (`hip_bridge::keep_host_memory_out_of_reclaim`),
/// as an explicit [`EXPERT_VRAM_LAYERS_ENV`] does: a Qwen4 model on discrete
/// GPUs. It rests on those static facts, not on the placement the load
/// resolves: unset, that depends on the free VRAM and the reserve at load
/// time, after the runtime has read the switches, and the upload of tens of
/// GB out of the mapped file needs them under host-memory pressure whatever
/// the placement. Unified memory never host-maps experts, and an
/// unrecognized arch keeps ROCm's defaults.
pub fn keeps_host_memory_out_of_reclaim(arch_id: u32, device_archs: &[&str]) -> bool {
    arch_id == crate::ARCH_ID
        && !device_archs.is_empty()
        && device_archs
            .iter()
            .all(|arch| hipfire_config::is_discrete_memory_arch(arch))
}

/// Bytes of every routed expert payload (trunk and MTP layers).
pub fn routed_expert_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    routed_expert_bytes_ranks(weights, bytes_of, 1)
}

/// [`routed_expert_bytes`] for one rank of a `ranks`-way expert-parallel load:
/// the largest rank's share (`max_local_count` experts of every routed expert
/// stack), which is what every rank budgets so a layer is resident on all
/// ranks or none.
pub fn routed_expert_bytes_ranks(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    ranks: usize,
) -> Result<u64, String> {
    weights
        .iter()
        .filter(|entry| {
            is_routed_expert(&entry.name)
                && !entry.residency.is_external()
                && !matches!(entry.policy, ShardPolicy::Tied { .. })
        })
        .try_fold(0u64, |total, entry| {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            Ok(total + rank_share(entry, bytes, 0, ranks)?)
        })
}

/// [`routed_expert_bytes`] for one pipeline-parallel band: every expert of
/// the owned trunk layers (all 512, never a modulo-rank share) plus the MTP
/// head's experts only on the output band, from the exact per-entry payload
/// sizes. `Single` equals [`routed_expert_bytes`].
pub fn routed_expert_bytes_scope(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
) -> Result<u64, String> {
    let bytes = band_weight_census(weights, &bytes_of, scope)?;
    bytes
        .trunk
        .checked_add(bytes.mtp)
        .ok_or_else(|| "band routed-expert byte census overflows".to_string())
}

fn parse_expert_vram_layers(value: &str) -> Result<ExpertVramLayers, String> {
    let value = value.trim();
    if value == "auto" {
        return Ok(ExpertVramLayers::Auto);
    }
    value
        .parse::<usize>()
        .map(ExpertVramLayers::Layers)
        .map_err(|_| {
            format!("{EXPERT_VRAM_LAYERS_ENV}={value:?} is neither `auto` nor a layer count")
        })
}

fn is_routed_expert(name: &str) -> bool {
    name.ends_with(".mlp.experts.gate_up_proj")
        || name.ends_with(".mlp.experts.down_proj")
        || name.ends_with(crate::expert_arena::ARENA_SUFFIX)
}

/// Resident bytes outside the routed experts, and the routed-expert bytes of
/// one trunk layer (layer 0), from each entry's payload size. Tied aliases
/// have no payload of their own and external rows stay in the file.
pub fn resident_split(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<(u64, u64), String> {
    resident_split_ranks(weights, bytes_of, 1)
}

/// [`resident_split`] for one rank of a `ranks`-way expert-parallel load: the
/// non-expert bytes are replicated on every rank, the layer-0 routed-expert
/// bytes are the per-expert bytes times `max_local_count` (the largest rank's
/// share; an uneven split gives the low ranks one more expert).
pub fn resident_split_ranks(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    ranks: usize,
) -> Result<(u64, u64), String> {
    let mut non_expert = 0u64;
    let mut layer_experts = 0u64;
    // Slab arenas differ in size from layer to layer (the hot set is not
    // uniform), so the per-layer charge is the largest arena, never layer 0.
    let mut layer_arena_max = 0u64;
    for entry in weights {
        if entry.residency.is_external() || matches!(entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        if !is_routed_expert(&entry.name) {
            non_expert += bytes;
        } else if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
            layer_arena_max = layer_arena_max.max(rank_share(entry, bytes, 0, ranks)?);
        } else if !entry.name.starts_with("mtp.") && entry.layer == Some(0) {
            layer_experts += rank_share(entry, bytes, 0, ranks)?;
        }
    }
    Ok((non_expert, layer_experts + layer_arena_max))
}

/// Gate/up and down payload bytes of the largest trunk layer of a packed
/// (one tensor per projection per layer) manifest, separately: what the S1
/// expert page layout splits over the layer's experts
/// ([`crate::slowlink_admission::PanelPageLayout::new`]). The MTP layer is not
/// a trunk layer and is excluded, like [`resident_split`]. A manifest with no
/// packed projection pair is an error: S1 never sizes an arena layout.
pub fn routed_layer_payload_split(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<(u64, u64), String> {
    let mut per_layer: std::collections::BTreeMap<usize, (u64, u64)> =
        std::collections::BTreeMap::new();
    for entry in weights {
        if entry.residency.is_external()
            || matches!(entry.policy, ShardPolicy::Tied { .. })
            || entry.name.starts_with("mtp.")
        {
            continue;
        }
        let gate_up = entry.name.ends_with(".mlp.experts.gate_up_proj");
        let down = entry.name.ends_with(".mlp.experts.down_proj");
        let Some(layer) = entry.layer.filter(|_| gate_up || down) else {
            continue;
        };
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        let slot = per_layer.entry(layer).or_default();
        let target = if gate_up { &mut slot.0 } else { &mut slot.1 };
        *target = target
            .checked_add(bytes)
            .ok_or_else(|| "routed layer payload byte census overflows".to_string())?;
    }
    let gate_up = per_layer.values().map(|bytes| bytes.0).max().unwrap_or(0);
    let down = per_layer.values().map(|bytes| bytes.1).max().unwrap_or(0);
    if gate_up == 0 || down == 0 {
        return Err("manifest has no packed routed gate/up and down expert payload".to_string());
    }
    Ok((gate_up, down))
}

/// Stored dtype of the language head, which the native MTP draft head ranks
/// with.
pub fn language_head_dtype(weights: &[WeightEntry]) -> Option<rdna_compute::DType> {
    weights
        .iter()
        .find(|entry| entry.name == crate::weights::QWEN4_LM_HEAD)
        .map(|entry| entry.dtype)
}

/// VRAM `auto` leaves free beyond the non-expert weights for a load whose
/// trunk QSA context storage is `context` with a `chunk_rows` prefill chunk:
/// [`AUTO_VRAM_RESERVE_BYTES`], the trunk QSA arenas' committed bytes in
/// `qsa_format` past those of [`AUTO_VRAM_RESERVE_MAX_SEQ`] legacy tokens
/// (the measured layout), the chunk-sized forward resources' change from the
/// measured [`AUTO_VRAM_RESERVE_CHUNK`] layout, `mtp_bytes` when a native MTP
/// speculator attaches (`mtp_spec::native_mtp_device_bytes`), and
/// `gather_bytes` for the gathered QSA prefill attention's workspace
/// (`rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes`). This is
/// the historical measured formula: the floor of [`AutoReserve`], which
/// takes the larger of it and the bottom-up [`ReserveCensus`]. Only
/// committed bytes are charged: VMM context arenas commit the first chunk's
/// pages (inside the measured layout), their virtual extent nothing, so the
/// admitted context does not size a VMM reserve; legacy arenas commit all
/// of `max_seq`. Callers on a discrete card pass legacy storage whatever the
/// backend ([`crate::Qwen4ContextCommit::for_expert_reserve`]).
///
/// `mode` is the scratch plan the forward will actually allocate
/// ([`crate::gpu_forward::qwen4_forward_scratch_mode`]); only the current
/// forward is charged in it. The measured baseline is a historical
/// subtraction of the original layout and stays `Full` with its all-row
/// speculative logits, so a compact plan's saving reaches the placement
/// instead of being rebased away.
pub fn auto_vram_reserve(
    config: &Qwen4Config,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    qsa_layout: impl Into<rdna_compute::tensor_ops::QsaStateLayout>,
    mtp_bytes: Option<u64>,
    gather_bytes: Option<u64>,
) -> Result<u64, String> {
    let qsa_layout = qsa_layout.into();
    let qsa_format = qsa_layout.kv;
    let overflow = |seq| format!("QSA context state for max_seq {seq} overflows");
    let trunk = |bytes: Option<usize>| {
        bytes
            .and_then(|bytes| bytes.checked_mul(config.n_full_layers()))
            .and_then(|bytes| u64::try_from(bytes).ok())
    };
    let committed = trunk(context.committed_layer_bytes(config, qsa_format))
        .ok_or_else(|| overflow(context.max_seq))?;
    let measured_context =
        trunk(config.qsa_context_arena_bytes(AUTO_VRAM_RESERVE_MAX_SEQ, qsa_format))
            .ok_or_else(|| overflow(AUTO_VRAM_RESERVE_MAX_SEQ))?;
    // The derived arenas' (BF-1 F16 K/V, pooled BF16 mirror) extra storage
    // must fit through the admitted context, including on UMA VMM loads that
    // otherwise charge only the first mapped chunk. Do not subtract it with
    // the historical (arena-off) measured layout.
    let arena = if qsa_layout.f16_arena || qsa_layout.pooled_bf16 {
        let capacity = crate::Qwen4ContextCommit {
            tokens: context.max_seq,
            ..*context
        };
        let plain = capacity
            .committed_layer_bytes(config, qsa_format)
            .ok_or_else(|| overflow(context.max_seq))?;
        let with = capacity
            .committed_layer_bytes(config, qsa_layout)
            .ok_or_else(|| overflow(context.max_seq))?;
        trunk(Some(
            with.checked_sub(plain)
                .ok_or_else(|| overflow(context.max_seq))?,
        ))
        .ok_or_else(|| overflow(context.max_seq))?
    } else {
        0
    };
    let context = committed
        .saturating_sub(measured_context)
        .checked_add(arena)
        .ok_or_else(|| overflow(context.max_seq))?;
    let overflow = || format!("forward resources for a {chunk_rows}-row chunk overflow");
    let measured = crate::gpu_forward::Qwen4GpuForwardScratch::device_bytes(
        config,
        AUTO_VRAM_RESERVE_CHUNK,
        Qwen4ForwardScratchMode::Full,
    )
    .ok()
    .and_then(|scratch| {
        let spec = AUTO_VRAM_RESERVE_CHUNK.checked_mul(config.vocab_size.checked_mul(4)? + 4)?;
        scratch.checked_add(spec as u64)
    })
    .ok_or_else(overflow)?;
    let forward = crate::gpu_forward::qwen4_forward_device_bytes(config, chunk_rows, mode)
        .ok_or_else(overflow)?;
    AUTO_VRAM_RESERVE_BYTES
        .checked_add(context)
        .and_then(|bytes| bytes.checked_add(forward))
        .and_then(|bytes| bytes.checked_sub(measured))
        .and_then(|bytes| bytes.checked_add(mtp_bytes.unwrap_or(0)))
        .and_then(|bytes| bytes.checked_add(gather_bytes.unwrap_or(0)))
        .ok_or_else(|| "auto expert VRAM reserve overflows".to_string())
}

/// Number of itemised terms of the kernels' pool workspace
/// (`rdna_compute::scratch::qwen4_prefill_workspace_terms`).
pub const WORKSPACE_TERM_COUNT: usize = 11;

/// A bottom-up census of what one load's first full-context request takes
/// beyond the resident weights, term by term: the state's QSA context at the
/// admitted `max_seq` as the actual backend maps it, the rest of the request
/// state, the chunk-sized forward resources, the lazy F16 weight shadows, the
/// kernels' pool workspace (itemised by the engine), the QSA split
/// selector's scratch, native MTP and the QSA gather workspace. `auto`
/// keeps the larger of this and the historical measured formula
/// ([`auto_vram_reserve`]) free ([`AutoReserve`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReserveCensus {
    pub qsa_context: u64,
    pub state: u64,
    pub forward: u64,
    pub f16_shadows: u64,
    pub workspace: [(&'static str, u64); WORKSPACE_TERM_COUNT],
    pub qsa_selector: u64,
    pub mtp: u64,
    pub gather: u64,
    /// The HIP runtime's own device memory ([`runtime_overhead_bytes`]).
    pub runtime: u64,
}

impl ReserveCensus {
    /// Every term, named, in log order.
    pub fn terms(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        [
            ("QSA context", self.qsa_context),
            ("state", self.state),
            ("forward", self.forward),
            ("F16 weight shadows", self.f16_shadows),
            ("QSA selector", self.qsa_selector),
            ("native MTP", self.mtp),
            ("QSA gather", self.gather),
            ("HIP runtime", self.runtime),
        ]
        .into_iter()
        .chain(self.workspace.iter().copied())
    }

    /// Sum of every term, checked.
    pub fn total(&self) -> Result<u64, String> {
        self.terms().try_fold(0u64, |total, (name, bytes)| {
            total
                .checked_add(bytes)
                .ok_or_else(|| format!("reserve census overflows at '{name}'"))
        })
    }

    /// `name MiB, ...` for the placement log, every term.
    pub fn describe(&self) -> String {
        let mut text = String::new();
        for (name, bytes) in self.terms() {
            if !text.is_empty() {
                text.push_str(", ");
            }
            text.push_str(&format!("{name} {} MiB", bytes >> 20));
        }
        text
    }
}

/// The state-derived census terms ([`ReserveCensus::qsa_context`],
/// [`ReserveCensus::state`], [`ReserveCensus::forward`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCensus {
    pub qsa_context: u64,
    pub state: u64,
    pub forward: u64,
}

/// The census terms the state and forward determine, for a load whose
/// context storage is `context`: pass the ACTUAL backend with the whole
/// admitted context mapped (`Qwen4ContextCommit::new(backend, max_seq,
/// max_seq, granularity)`), never the legacy-forced expert-reserve layout.
///
/// - `qsa_context`: the trunk QSA context arenas as `context` commits them
///   (VMM: each arena's mapped pages rounded to the driver granularity).
/// - `state`: the rest of `Qwen4State::device_bytes` (GDN recurrent and
///   conv with the snapshot arena's mirror, the QSA fixed arenas, PLE conv,
///   hyper feedback), the QSA context counted once in `qsa_context`.
/// - `forward`: [`crate::gpu_forward::qwen4_forward_device_bytes`] at
///   `chunk_rows` in `mode`.
pub fn state_census(
    config: &Qwen4Config,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Result<StateCensus, String> {
    let overflow = |what: &str| format!("reserve census {what} overflows");
    let qsa_context = context
        .committed_layer_bytes(config, format.layout())
        .and_then(|bytes| bytes.checked_mul(config.n_full_layers()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("QSA context"))?;
    let state = crate::state::Qwen4State::device_bytes(config, format, context)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("state"))?;
    let state = state
        .checked_sub(qsa_context)
        .ok_or_else(|| overflow("state without its QSA context"))?;
    let forward = crate::gpu_forward::qwen4_forward_device_bytes(config, chunk_rows, mode)
        .ok_or_else(|| overflow("forward resources"))?;
    Ok(StateCensus {
        qsa_context,
        state,
        forward,
    })
}

/// The forward scratch one pipeline-parallel rank allocates, split by who
/// needs it, so every duplicate across ranks is explicit.
///
/// - `stage_shared`: fields every stage's trunk loop needs (hyper-connection
///   streams and mixers, rotation, projections, attention mixer output, every
///   MoE buffer, the BF16 scratch). Each stage carries its own copy: a declared
///   per-stage duplicate, never shared across devices.
/// - `band_local`: GDN fields only where the band has a GDN layer, QSA fields
///   only where it has a QSA layer.
/// - `input_only`: token ids, embedding rows and the PLE fields, input band
///   only.
/// - `output_only`: the language-model logits and the speculative
///   logit/argmax rows, output band only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardCensus {
    pub stage_shared: u64,
    pub band_local: u64,
    pub input_only: u64,
    pub output_only: u64,
}

impl ForwardCensus {
    /// All four terms, checked.
    pub fn total(&self) -> Result<u64, String> {
        [
            self.stage_shared,
            self.band_local,
            self.input_only,
            self.output_only,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| "forward census overflows".to_string())
    }
}

/// [`crate::gpu_forward::qwen4_forward_device_bytes`] for one pipeline-parallel
/// band: the same plan (`Qwen4GpuForwardScratch::plan`, owning slots only,
/// borrowed views uncharged, plus the speculative logits/argmax), but a field
/// is charged only on a rank that runs its operation. `Single` equals
/// `qwen4_forward_device_bytes` exactly (pinned by a test over both modes).
///
/// Band-aware sizing: the projection scratch spans the QSA q/gate width only
/// with a QSA layer and the GDN qkv width only with a GDN layer; the BF16
/// scratch spans the GDN value width only with a GDN layer. In the compact
/// plan the GDN recurrent output is a view of the QSA q/gate owner, so a
/// compact band with GDN but no QSA layer is refused.
pub fn forward_census_scope(
    config: &Qwen4Config,
    rows: usize,
    mode: Qwen4ForwardScratchMode,
    scope: &Qwen4BandScope,
) -> Result<ForwardCensus, String> {
    use crate::config::LayerType;
    use rdna_compute::DType;
    let overflow = || "forward census size overflows".to_string();
    if rows == 0 {
        return Err("forward census rows is zero".to_string());
    }
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let dims = crate::gpu_forward::program_dims(config);
    let layout = crate::program::Qwen4ScratchLayout::for_rows(dims, rows)
        .map_err(|error| error.to_string())?;
    let has = |kind: LayerType| {
        scope
            .band
            .clone()
            .any(|layer| config.layer_types.get(layer) == Some(&kind))
    };
    let (has_gdn, has_qsa) = (
        has(LayerType::LinearAttention),
        has(LayerType::FullAttention),
    );
    let compact = mode == Qwen4ForwardScratchMode::FnSymGfx1201;
    if compact && has_gdn && !has_qsa {
        return Err(
            "compact forward scratch aliases the GDN output onto QSA scratch, \
            which a band without QSA layers does not own"
                .to_string(),
        );
    }
    let hidden = layout.hidden;
    let wide = layout.wide;
    let q_width = layout.q_width;
    let qsa_qgate = 2 * q_width;
    let gdn_value = layout.gdn_value;
    let max_projection = wide
        .max(hidden)
        .max(if has_qsa { qsa_qgate } else { 0 })
        .max(if has_gdn { layout.gdn_qkv } else { 0 });
    let max_rotation = wide.max(hidden).max(config.ffn.intermediate_size());
    let ple_embed_dim = config
        .ple
        .as_ref()
        .map_or(0, |ple| ple.head_count * ple.row_width);
    let ple_channels = wide;
    let ffn_batch_rows = layout.routed_rows.max(layout.dense_rows);
    let i32_bytes = std::mem::size_of::<i32>();
    let rows_of = |width: usize| mul(rows, width);
    let route_i32_bytes = mul(layout.slots, i32_bytes)?;
    let grouped_bound = layout.grouped_rows;
    let routed_width = mul(config.num_experts_per_tok, config.moe_intermediate_size)?;
    let top_k = config.num_experts_per_tok;
    let moe = config.moe_intermediate_size;

    // MoE buffers whose compact size differs from the full plan.
    let (mut gate_batch, mut up_batch, mut rot_batch) =
        (ffn_batch_rows, ffn_batch_rows, rows_of(routed_width)?);
    let (mut down_expanded, mut y_gate_up, mut y_down) = (
        rows_of(mul(top_k, hidden)?)?,
        mul(grouped_bound, 2 * moe)?,
        mul(grouped_bound, hidden)?,
    );
    if compact {
        if ple_channels != wide {
            return Err(
                "compact scratch requires PLE channels equal to the hyper-connection \
                 width"
                    .to_string(),
            );
        }
        let f32_bytes = std::mem::size_of::<f32>();
        let bf16_bytes = DType::BF16.size();
        let fallback_rows = rows.min(crate::gpu_forward::QWEN4_COMPACT_FALLBACK_ROWS);
        let fallback_grouped =
            crate::program::grouped_m_total_bound(mul(fallback_rows, top_k)?, config.num_experts)
                .map_err(|error| error.to_string())?;
        let facade = |bytes: usize| bytes.div_ceil(f32_bytes);
        let fallback_batch = mul(fallback_rows, routed_width)?;
        (gate_batch, up_batch, rot_batch) = (fallback_batch, fallback_batch, fallback_batch);
        let hidden_f32_bytes = mul(hidden, f32_bytes)?;
        down_expanded = facade(
            mul(mul(fallback_rows, top_k)?, hidden_f32_bytes)?
                .max(mul(rows, hidden_f32_bytes)?)
                .max(mul(mul(rows, top_k)?, 8)?),
        );
        y_gate_up = facade(
            mul(fallback_grouped, mul(2 * moe, f32_bytes)?)?
                .max(mul(grouped_bound, mul(moe, bf16_bytes)?)?),
        );
        y_down = facade(
            mul(fallback_grouped, hidden_f32_bytes)?
                .max(mul(grouped_bound, mul(hidden, bf16_bytes)?)?),
        );
        // `Qwen4GpuForwardScratch::plan`'s alias binds, checked as the plan
        // checks them (every view fits its owner's capacity and row pitch) so
        // a geometry the plan refuses is refused here too, whatever the band.
        // `(owner elements, owner pitch, view pitch, view elements)`.
        let at_least_one = |elements: usize| elements.max(1);
        let wide_rows = at_least_one(rows_of(wide)?);
        let ple_rows = at_least_one(rows_of(ple_channels)?);
        let binds: [(usize, Option<usize>, usize, Vec<usize>); 6] = [
            (
                at_least_one(rows_of(max_projection)?),
                Some(max_projection),
                max_projection,
                vec![wide_rows, ple_rows],
            ),
            (
                at_least_one(rows_of(max_projection)?),
                Some(max_projection),
                max_projection,
                vec![wide_rows, ple_rows],
            ),
            (
                at_least_one(rows_of(max_rotation)?),
                Some(max_rotation),
                max_rotation,
                vec![ple_rows],
            ),
            (y_down, None, ple_channels, vec![ple_rows]),
            (
                at_least_one(rows_of(hidden)?),
                Some(hidden),
                hidden,
                vec![at_least_one(rows_of(hidden)?)],
            ),
            (
                at_least_one(rows_of(qsa_qgate)?),
                Some(qsa_qgate),
                qsa_qgate,
                vec![at_least_one(rows_of(gdn_value)?)],
            ),
        ];
        for (owner_elements, owner_pitch, view_pitch, views) in &binds {
            if let Some(pitch) = owner_pitch {
                if *owner_elements < mul(rows, *pitch)? {
                    return Err(
                        "compact scratch alias owner is smaller than its row pitch".to_string()
                    );
                }
            }
            for view in views {
                if view > owner_elements || *view > mul(rows, *view_pitch)? {
                    return Err("compact scratch alias view does not fit its owner".to_string());
                }
            }
        }
    }

    // Bytes of one owning slot (every slot holds at least one element).
    let slot = |elements: usize, dtype: DType| -> Result<u64, String> {
        u64::try_from(
            elements
                .max(1)
                .checked_mul(dtype.size())
                .ok_or_else(overflow)?,
        )
        .map_err(|_| overflow())
    };
    let sum = |slots: &[Result<u64, String>]| -> Result<u64, String> {
        slots.iter().try_fold(0u64, |total, bytes| {
            total.checked_add(bytes.clone()?).ok_or_else(overflow)
        })
    };
    // Compact plan: eight phase-disjoint fields are uncharged views.
    let owned = |view: bool, bytes: Result<u64, String>| -> Result<u64, String> {
        if compact && view {
            Ok(0)
        } else {
            bytes
        }
    };
    let f32 = DType::F32;
    // Compact plan: the overridden MoE buffers are exact element counts
    // (possibly zero); everywhere else every slot holds at least one element.
    let moe_slot = |elements: usize| -> Result<u64, String> {
        if compact {
            u64::try_from(elements.checked_mul(f32.size()).ok_or_else(overflow)?)
                .map_err(|_| overflow())
        } else {
            slot(elements, f32)
        }
    };
    let stage_shared = sum(&[
        slot(rows_of(wide)?, f32),              // Streams
        owned(true, slot(rows_of(wide)?, f32)), // HcNormalized (view)
        slot(rows_of(config.hc_lowrank)?, f32), // HcLow
        owned(true, slot(rows_of(wide)?, f32)), // HcUp (view)
        slot(rows_of(hidden)?, f32),            // HcMixed
        slot(rows_of(config.hc_count)?, f32),   // HcGates
        slot(rows_of(max_rotation)?, f32),      // Rotation
        slot(rows_of(max_projection)?, f32),    // Projection
        slot(rows_of(max_projection)?, f32),    // Projection2
        // GdnBf16: also the QSA and hyper-connection BF16 scratch.
        slot(
            if has_gdn { gdn_value } else { 0 }
                .max(hidden)
                .max(config.hc_lowrank),
            DType::BF16,
        ),
        slot(rows_of(hidden)?, f32), // AttentionOutput
        slot(rows_of(dims_router_columns(config))?, f32), // RouterLogits
        slot(rows_of(hidden)?, f32), // MoeXRot
        slot(rows_of(2 * moe)?, f32), // MoeGateUp
        slot(rows_of(config.shared_expert_intermediate_size)?, f32), // MoeGate
        slot(rows_of(config.shared_expert_intermediate_size)?, f32), // MoeUp
        slot(rows_of(config.shared_expert_intermediate_size)?, f32), // MoeHidden
        slot(rows_of(hidden)?, f32), // FfnOutput
        slot(rows_of(hidden)?, f32), // MoeSharedOutput
        moe_slot(gate_batch),
        moe_slot(up_batch),
        moe_slot(rot_batch),
        slot(route_i32_bytes, DType::Raw), // MoeTopkIndices
        slot(rows_of(top_k)?, f32),        // MoeTopkWeights
        moe_slot(down_expanded),
        slot(rows_of(config.shared_expert_intermediate_size)?, f32), // MoeScalar
        slot(mul(config.num_experts, i32_bytes)?, DType::Raw),       // MoeExpertTokenCounts
        slot(mul(config.num_experts + 1, i32_bytes)?, DType::Raw),   // MoeExpertOffsets
        slot(mul(grouped_bound, i32_bytes)?, DType::Raw),            // MoeSortedSlotIndex
        slot(mul(grouped_bound / 16, i32_bytes)?, DType::Raw),       // MoeExpertTileIds
        slot(route_i32_bytes, DType::Raw),                           // MoeInversePerm
        moe_slot(y_gate_up),
        moe_slot(y_down),
    ])?;
    let gdn_local = if has_gdn {
        let value_rows = rows_of(gdn_value)?;
        sum(&[
            slot(rows_of(config.linear_num_value_heads)?, f32), // GdnA
            slot(rows_of(config.linear_num_value_heads)?, f32), // GdnB
            slot(rows_of(config.linear_num_value_heads)?, f32), // GdnGate
            slot(rows_of(config.linear_num_value_heads)?, f32), // GdnBeta
            owned(true, slot(value_rows, f32)),                 // GdnRecurrentOutput (view)
            slot(value_rows, f32),                              // GdnZ
            slot(value_rows, f32),                              // GdnOutput
        ])?
    } else {
        0
    };
    let qsa_local = if has_qsa {
        let kv_width = mul(config.num_key_value_heads, config.head_dim)?;
        let selected = mul(mul(rows, config.qsa_selected_capacity())?, i32_bytes)?;
        sum(&[
            slot(rows_of(layout.index_width)?, f32), // QsaIndex
            slot(rows_of(qsa_qgate)?, f32),          // QsaQgate
            slot(rows_of(kv_width)?, f32),           // QsaK
            slot(rows_of(kv_width)?, f32),           // QsaV
            slot(rows_of(q_width)?, f32),            // QsaOutput
            slot(selected, DType::Raw),              // QsaSelected
        ])?
    } else {
        0
    };
    let band_local = gdn_local.checked_add(qsa_local).ok_or_else(overflow)?;
    let input_only = if scope.input {
        sum(&[
            slot(rows_of(i32_bytes)?, DType::Raw),          // TokenIds
            slot(rows_of(hidden)?, f32),                    // EmbeddingRot
            slot(rows_of(hidden)?, f32),                    // Embeddings
            slot(rows_of(ple_embed_dim)?, DType::BF16),     // PleStaged
            slot(rows_of(ple_embed_dim.max(hidden))?, f32), // PleRows
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleKey (view)
            owned(true, slot(rows_of(hidden)?, f32)),       // PleValue (view)
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleQuery (view)
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleGated (view)
            slot(rows_of(ple_channels)?, f32),              // PleNormed (dedicated)
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleOutput (view)
        ])?
    } else {
        0
    };
    let output_only = if scope.output {
        let logit_row_bytes = config
            .vocab_size
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(overflow)?;
        let spec = crate::gpu_forward::qwen4_spec_logit_rows(rows)
            .checked_mul(logit_row_bytes)
            .ok_or_else(overflow)?;
        slot(config.vocab_size, f32)?
            .checked_add(u64::try_from(spec).map_err(|_| overflow())?)
            .ok_or_else(overflow)?
    } else {
        0
    };
    Ok(ForwardCensus {
        stage_shared,
        band_local,
        input_only,
        output_only,
    })
}

/// `router_columns` of the config's program dims (one placeholder column for
/// a dense FFN).
fn dims_router_columns(config: &Qwen4Config) -> usize {
    crate::gpu_forward::router_columns(crate::gpu_forward::program_dims(config))
}

/// [`state_census`] for one pipeline-parallel band: the GDN and QSA state of
/// the band's own layers, the PLE conv history and `hyper_feedback` only on
/// the input band (`Qwen4State::device_bytes_scoped`), the QSA context of
/// the band's own QSA layers and the band's own forward scratch
/// ([`forward_census_scope`]). `Single` equals [`state_census`].
pub fn state_census_scope(
    config: &Qwen4Config,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    scope: &Qwen4BandScope,
) -> Result<StateCensus, String> {
    scope.validate(config)?;
    let overflow = |what: &str| format!("reserve census {what} overflows");
    let qsa_layers = scope
        .band
        .clone()
        .filter(|&layer| {
            config.layer_types.get(layer) == Some(&crate::config::LayerType::FullAttention)
        })
        .count();
    let qsa_context = context
        .committed_layer_bytes(config, format.layout())
        .and_then(|bytes| bytes.checked_mul(qsa_layers))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("QSA context"))?;
    let state = crate::state::Qwen4State::device_bytes_scoped(config, format, context, scope)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("state"))?;
    let state = state
        .checked_sub(qsa_context)
        .ok_or_else(|| overflow("state without its QSA context"))?;
    let forward = forward_census_scope(config, chunk_rows, mode, scope)?.total()?;
    Ok(StateCensus {
        qsa_context,
        state,
        forward,
    })
}

/// Refuse a pipeline band combined with expert parallelism: PP never reuses
/// the modulo-rank expert split, so any `ranks > 1` with a band that does not
/// cover the whole model is an error. `Single` with any rank count passes.
pub fn refuse_band_with_expert_parallel(
    config: &Qwen4Config,
    scope: &Qwen4BandScope,
    ranks: usize,
) -> Result<(), String> {
    let whole = scope.input && scope.output && scope.band == (0..config.num_hidden_layers);
    if ranks > 1 && !whole {
        return Err(format!(
            "pipeline-parallel band {}..{} cannot combine with {ranks}-way expert parallelism",
            scope.band.start, scope.band.end
        ));
    }
    Ok(())
}

/// One pipeline-parallel rank's byte census (Single: the whole model). The
/// weight terms come from the manifest entries the band owns and the same
/// payload sizes as [`resident_split`]; EP is not involved, no routed expert
/// is split by rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandCensus {
    pub band: std::ops::Range<usize>,
    pub input: bool,
    pub output: bool,
    /// MoE layers owned: the band's trunk layers plus the MTP layer on output.
    pub moe_layers: usize,
    /// Owned resident non-routed payloads (tied aliases and external rows have
    /// none).
    pub non_expert_bytes: u64,
    /// Routed experts of the owned trunk layers, every expert, exact codec bytes.
    pub routed_trunk_bytes: u64,
    /// Routed experts of the MTP head's layer (output band only).
    pub routed_mtp_bytes: u64,
    /// Routed bytes of the largest single owned trunk layer.
    pub max_layer_routed_bytes: u64,
    /// The separately owned materialization of a tied source the source's
    /// owner also holds (the MTP embedding of `embed_tokens` on an output
    /// band that is not the input band). Zero for `Single`.
    pub endpoint_duplicate_bytes: u64,
    pub f16_shadow_bytes: u64,
    /// `state.forward` is `forward.total()`.
    pub state: StateCensus,
    /// The band's forward scratch by role: the per-stage shared copy is an
    /// explicit duplicate across ranks, the rest is owned once.
    pub forward: ForwardCensus,
    /// Native MTP reserves of this band (all zero when the load is AR): the
    /// output band's head and request buffers, every band's row capture.
    pub mtp: NativeMtpCensus,
}

impl BandCensus {
    /// Trunk plus MTP routed expert bytes, checked.
    pub fn routed_bytes(&self) -> Result<u64, String> {
        self.routed_trunk_bytes
            .checked_add(self.routed_mtp_bytes)
            .ok_or_else(|| "band routed-expert byte census overflows".to_string())
    }

    /// Every resident weight byte of the band: non-expert, routed and the
    /// endpoint duplicate, checked. Shadows and state are separate terms.
    pub fn weight_bytes(&self) -> Result<u64, String> {
        self.routed_bytes()?
            .checked_add(self.non_expert_bytes)
            .and_then(|bytes| bytes.checked_add(self.endpoint_duplicate_bytes))
            .ok_or_else(|| "band weight byte census overflows".to_string())
    }
}

/// Byte terms of [`band_weight_census`].
struct BandWeightBytes {
    non_expert: u64,
    trunk: u64,
    mtp: u64,
    max_layer: u64,
    endpoint_duplicate: u64,
}

/// Owned routed-expert bytes of trunk layers and of the MTP head (exact sums
/// over the entries' payloads), the owned non-expert bytes, the largest owned
/// trunk layer's routed bytes and the tied endpoint duplicates, for `scope`.
/// Sums are checked; `Single` equals [`routed_expert_bytes`] and
/// [`resident_split`]'s non-expert bytes.
///
/// `weights` may be the full manifest or one already filtered to the band
/// (`Qwen4Manifest::with_band_scope`), which drops a tied source another band
/// owns. Then the source's payload is requested through `bytes_of` from a
/// clone of the tied entry renamed to the source with a replicated policy
/// (same logical shape and dtype): no source is uploaded, the bytes are the
/// output band's own materialization, counted as an endpoint duplicate and
/// never as an alias.
///
/// A non-`Single` scope refuses expert-sharded entries before any `bytes_of`
/// call: pipeline bands never reuse the expert-parallel split.
fn band_weight_census(
    weights: &[WeightEntry],
    bytes_of: &impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
) -> Result<BandWeightBytes, String> {
    if !(scope.input && scope.output)
        && weights.iter().any(|entry| {
            matches!(
                entry.policy,
                ShardPolicy::ExpertSharded { .. } | ShardPolicy::ExpertTensorSharded { .. }
            )
        })
    {
        return Err(
            "pipeline-parallel bands cannot be combined with expert parallelism".to_string(),
        );
    }
    let overflow = |what: &str| format!("band census {what} overflows");
    let payload = |entry: &WeightEntry| {
        bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))
    };
    let (mut non_expert, mut trunk, mut mtp, mut duplicate) = (0u64, 0u64, 0u64, 0u64);
    let mut per_layer = std::collections::BTreeMap::<usize, u64>::new();
    for entry in weights {
        if entry.residency.is_external() {
            continue;
        }
        if let ShardPolicy::Tied { source } = &entry.policy {
            if scope.owns_entry(entry) {
                let origin = weights.iter().find(|candidate| &candidate.name == source);
                let materialized = match origin {
                    Some(origin) if scope.owns_entry(origin) => None,
                    Some(origin) => Some(payload(origin)?),
                    None => {
                        let mut request = entry.clone();
                        request.name = source.clone();
                        request.policy = ShardPolicy::Replicate;
                        Some(payload(&request)?)
                    }
                };
                if let Some(bytes) = materialized {
                    duplicate = duplicate
                        .checked_add(bytes)
                        .ok_or_else(|| overflow("endpoint duplicates"))?;
                }
            }
            continue;
        }
        if !scope.owns_entry(entry) {
            continue;
        }
        let bytes = payload(entry)?;
        if !is_routed_expert(&entry.name) {
            non_expert = non_expert
                .checked_add(bytes)
                .ok_or_else(|| overflow("non-expert bytes"))?;
        } else if entry.name.starts_with("mtp.") {
            mtp = mtp
                .checked_add(bytes)
                .ok_or_else(|| overflow("MTP routed bytes"))?;
        } else {
            let layer = entry
                .layer
                .ok_or_else(|| format!("routed expert '{}' has no trunk layer", entry.name))?;
            trunk = trunk
                .checked_add(bytes)
                .ok_or_else(|| overflow("trunk routed bytes"))?;
            let slot = per_layer.entry(layer).or_default();
            *slot = slot
                .checked_add(bytes)
                .ok_or_else(|| overflow("layer routed bytes"))?;
        }
    }
    let max_layer = per_layer.values().copied().max().unwrap_or(0);
    Ok(BandWeightBytes {
        non_expert,
        trunk,
        mtp,
        max_layer,
        endpoint_duplicate: duplicate,
    })
}

/// The band-aware byte census of one pipeline-parallel rank, from the
/// manifest's own entries and the actual state backend: routed bytes (owned
/// trunk layers; MTP only on output), non-expert bytes, F16 shadows,
/// [`moe_layer_count_scope`] and [`state_census_scope`]. Expert-parallel
/// counts are untouched ([`routed_expert_bytes_ranks`] and friends); a band
/// is validated first and no routed stack is split by rank.
#[allow(clippy::too_many_arguments)]
pub fn band_census(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Result<BandCensus, String> {
    band_census_native_mtp(
        config, weights, bytes_of, scope, format, context, chunk_rows, mode, None,
    )
}

/// [`band_census`] for a load whose native MTP is `mtp` (`None` = AR, every
/// MTP reserve zero; the output band still counts the head's weights, which
/// the band loads whether or not the head attaches). The weight terms are the
/// same entries either way; `mtp` adds [`BandCensus::mtp`] by
/// [`native_mtp_census_scope`].
#[allow(clippy::too_many_arguments)]
pub fn band_census_native_mtp(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    mtp: Option<&NativeMtpCensusInputs>,
) -> Result<BandCensus, String> {
    scope.validate(config)?;
    let weights_census = band_weight_census(weights, &bytes_of, scope)?;
    let mtp = match mtp {
        Some(inputs) => {
            native_mtp_census_scope(config, weights, context, chunk_rows, inputs, scope)?
        }
        None => NativeMtpCensus::default(),
    };
    Ok(BandCensus {
        band: scope.band.clone(),
        input: scope.input,
        output: scope.output,
        moe_layers: moe_layer_count_scope(config, scope),
        non_expert_bytes: weights_census.non_expert,
        routed_trunk_bytes: weights_census.trunk,
        routed_mtp_bytes: weights_census.mtp,
        max_layer_routed_bytes: weights_census.max_layer,
        endpoint_duplicate_bytes: weights_census.endpoint_duplicate,
        f16_shadow_bytes: f16_shadow_bytes_scope(weights, config.num_hidden_layers, scope)?,
        state: state_census_scope(config, format, context, chunk_rows, mode, scope)?,
        forward: forward_census_scope(config, chunk_rows, mode, scope)?,
        mtp,
    })
}

/// The native MTP request of a load, as the census sizes it: the resolved
/// values the Single load passes to `mtp_spec::native_mtp_device_bytes`
/// (the head dtype comes from the manifest's language head).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMtpCensusInputs {
    /// Draft window K the request sizes (`native_mtp_window_rows` rows). The
    /// Single load passes `native_mtp_proposal_drafts` (n-gram takeover
    /// width); a PP load has no n-gram pool or copy lane, so passes its K.
    pub max_k: usize,
    /// The GDN storage format and ring rows of the verify-row capture where
    /// the route captures rows (`mtp_spec::native_mtp_capture`), else `None`.
    pub row_capture: Option<(crate::GdnStateFormat, usize)>,
    /// Whether the request allocates the batched prompt-fill scratch
    /// (`mtp_spec::native_mtp_fill_scratch`).
    pub batched_fill: bool,
}

/// One band's native MTP reserve terms. [`Self::reserve`] of the `Single`
/// scope is exactly `mtp_spec::native_mtp_device_bytes`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeMtpCensus {
    /// What the attached head keeps (`Qwen4MtpGpu`): output band only.
    pub head_resident: u64,
    /// The draft head's build scratch, released at attach: output band only.
    pub head_build_scratch: u64,
    /// What the first speculative request keeps beyond the capture: the
    /// verify (wide-capture) hidden rows, the pending and row hidden carries
    /// and the prompt-fill scratch (`native_mtp_request_bytes` without row
    /// capture): output band only.
    pub request: u64,
    /// This band's own GDN verify-row capture (ring, inputs, output block and
    /// pointer table for its GDN layers): every band, never the whole model's
    /// charged once per rank.
    pub row_capture: u64,
}

impl NativeMtpCensus {
    /// Bytes the band reserves for native MTP: the resident head plus the
    /// larger of its build scratch (released at attach) and what the first
    /// request allocates after it; a band without the head reserves its row
    /// capture alone. Checked.
    pub fn reserve(&self) -> Result<u64, String> {
        let overflow = || "native MTP band reserve overflows".to_string();
        let request = self
            .request
            .checked_add(self.row_capture)
            .ok_or_else(overflow)?;
        self.head_resident
            .checked_add(self.head_build_scratch.max(request))
            .ok_or_else(overflow)
    }
}

/// The native MTP reserve of one pipeline band (`Single` = the whole model).
///
/// Every band charges the row capture of its OWN GDN layers
/// (`Qwen4State::row_capture_bytes_scoped`, `native_mtp_window_rows` rows): each
/// band arms capture before its target snapshot. The output band, which owns
/// the head and the request buffers, additionally charges
/// `Qwen4MtpGpu::device_bytes` and `native_mtp_request_bytes` WITHOUT capture
/// (the capture is band-scoped here, not the whole model's), so the `Single`
/// scope equals `native_mtp_device_bytes` and no rank pays the full-model
/// capture. `context` is the census context, as for the head's QSA arenas.
pub fn native_mtp_census_scope(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mtp: &NativeMtpCensusInputs,
    scope: &Qwen4BandScope,
) -> Result<NativeMtpCensus, String> {
    scope.validate(config)?;
    if config.mtp.is_none() {
        return Err(format!(
            "native MTP census for profile {}, which has no MTP head",
            config.profile.id()
        ));
    }
    let overflow = |what: &str| format!("native MTP census {what} overflows");
    let gdn_layers = scope
        .band
        .clone()
        .filter(|&layer| {
            config.layer_types.get(layer) == Some(&crate::config::LayerType::LinearAttention)
        })
        .count();
    let row_capture = match mtp.row_capture {
        Some((gdn, ring_rows)) => crate::state::Qwen4State::row_capture_bytes_scoped(
            config,
            gdn,
            ring_rows.max(crate::mtp_spec::native_mtp_window_rows(mtp.max_k)),
            gdn_layers,
        )
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("row capture"))?,
        None => 0,
    };
    if !scope.output {
        return Ok(NativeMtpCensus {
            row_capture,
            ..NativeMtpCensus::default()
        });
    }
    let head_dtype = language_head_dtype(weights)
        .ok_or_else(|| "native MTP census: manifest has no language head".to_string())?;
    let (resident, scratch) =
        crate::mtp_gpu::Qwen4MtpGpu::device_bytes(config, context, head_dtype)
            .ok_or_else(|| overflow("head"))?;
    let request = crate::mtp_spec::native_mtp_request_bytes(
        config,
        context.max_seq,
        chunk_rows,
        mtp.max_k,
        None,
        mtp.batched_fill,
    )
    .ok_or_else(|| overflow("request"))?;
    Ok(NativeMtpCensus {
        head_resident: u64::try_from(resident).map_err(|_| overflow("head"))?,
        head_build_scratch: u64::try_from(scratch).map_err(|_| overflow("head scratch"))?,
        request,
        row_capture,
    })
}

/// Name suffixes, below a trunk layer's `model.language_model.layers.{i}.`
/// prefix, of the BF16 matrices the long-prefill F16 WMMA routes read through
/// a model-lifetime F16 shadow (`Gpu::ensure_bf16_f16_shadow`, keyed by the
/// weight pointer): hyper-connection down mixes, the router, the shared
/// expert's gate/up/down and the PLE key/value projections. The direct BF16
/// readers (`input_mix_weight_up`, `shared_expert_gate`) and every other
/// tensor are not shadowed.
const F16_SHADOW_LAYER_SUFFIXES: [&str; 8] = [
    "attn_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "ple.key_proj.weight",
    "ple.value_proj.weight",
];

/// The final hyper-connection's down mix, shadowed like the per-layer ones.
const F16_SHADOW_ROOT: &str =
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight";

/// Device bytes of the lazy F16 weight shadows the long-prefill routes can
/// create for the trunk: for every resident, non-tied BF16 rank-2 entry of
/// [`F16_SHADOW_ROOT`] or a [`F16_SHADOW_LAYER_SUFFIXES`] tensor of a trunk
/// layer `0..num_layers`, the whole F16 matrix (`2 * rows * cols`, the full
/// payload). A conservative cold bound: the shadows exist only when the F16
/// WMMA arms run (exact gfx1201, long prefill, K a multiple of 64), but they
/// are charged whenever the manifest owns the weights, never from the
/// current warm cache. Entries are the manifest's own, so a profile or dtype
/// without them (an MQ-quantized trunk) charges nothing.
pub fn f16_shadow_bytes(weights: &[WeightEntry], num_layers: usize) -> Result<u64, String> {
    f16_shadow_bytes_owned(weights, num_layers, |_| true, true)
}

/// [`f16_shadow_bytes`] for one pipeline-parallel band: only the shadows of
/// the band's own trunk layers (the PLE projections ride with layer 1 on the
/// input band), and the final hyper-connection's down mix only on the output
/// band. `Single` equals [`f16_shadow_bytes`]. The MTP head has no shadows.
pub fn f16_shadow_bytes_scope(
    weights: &[WeightEntry],
    num_layers: usize,
    scope: &Qwen4BandScope,
) -> Result<u64, String> {
    f16_shadow_bytes_owned(
        weights,
        num_layers,
        |layer| scope.owns_layer(layer),
        scope.output,
    )
}

fn f16_shadow_bytes_owned(
    weights: &[WeightEntry],
    num_layers: usize,
    owns_layer: impl Fn(usize) -> bool,
    owns_root: bool,
) -> Result<u64, String> {
    use rdna_compute::DType;
    const LAYER_PREFIX: &str = "model.language_model.layers.";
    let shadowed = |name: &str| {
        if name == F16_SHADOW_ROOT {
            return owns_root;
        }
        name.strip_prefix(LAYER_PREFIX)
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(layer, suffix)| {
                layer
                    .parse::<usize>()
                    .is_ok_and(|layer| layer < num_layers && owns_layer(layer))
                    && F16_SHADOW_LAYER_SUFFIXES.contains(&suffix)
            })
    };
    weights
        .iter()
        .filter(|entry| {
            entry.dtype == DType::BF16
                && entry.logical_shape.len() == 2
                && !entry.residency.is_external()
                && !matches!(entry.policy, ShardPolicy::Tied { .. })
                && shadowed(&entry.name)
        })
        .try_fold(0u64, |total, entry| {
            let bytes = entry.logical_shape[0]
                .checked_mul(entry.logical_shape[1])
                .and_then(|elements| elements.checked_mul(2))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| format!("F16 shadow bytes of '{}' overflow", entry.name))?;
            Ok(bytes)
        })
}

/// The reserve `auto` keeps free: the larger of the historical measured
/// formula (`historical`, [`auto_vram_reserve`], the floor, no longer
/// evidence of what a long context takes) and the bottom-up
/// [`ReserveCensus`]; where the census carries the measured HIP runtime
/// overhead ([`runtime_overhead_bytes`]) the floor is retired
/// (`historical` is `None`) and the census alone is the reserve. The
/// whole-layer stage buffers and safety margin ([`PlacementCharges`]) are
/// added on top of [`Self::bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoReserve {
    pub historical: Option<u64>,
    pub census: ReserveCensus,
    pub census_total: u64,
}

impl AutoReserve {
    pub fn new(historical: Option<u64>, census: ReserveCensus) -> Result<Self, String> {
        let census_total = census.total()?;
        Ok(Self {
            historical,
            census,
            census_total,
        })
    }

    /// The bytes `auto` reserves: `max(historical, census)`, or the census
    /// alone once the floor is retired.
    pub fn bytes(&self) -> u64 {
        match self.historical {
            Some(historical) => historical.max(self.census_total),
            None => self.census_total,
        }
    }

    /// The base reserve a whole-layer placement is sized from. Every load that
    /// is not an admitted S1 load (`slowlink == false`: the incumbent, fast
    /// links, unified memory, every opt-out) reads exactly [`Self::bytes`],
    /// byte for byte. An admitted S1 load reads the bottom-up census alone
    /// ([`Self::census_total`]): the historical floor is a stale measured
    /// formula (5,517,207,808 B at the 1,536-row tile against a 2,237,111,724 B
    /// census) that would put the ten-layer floor out of reach, and the S1
    /// profile adds its own explicit terms and the safety margin on top
    /// ([`PlacementCharges`]); the Hold 1 gate verifies the measured peak
    /// against census + safety + allowance.
    pub fn placement_bytes(&self, slowlink: bool) -> u64 {
        if slowlink {
            self.census_total
        } else {
            self.bytes()
        }
    }

    /// Which formula set [`Self::bytes`].
    pub fn winner(&self) -> &'static str {
        match self.historical {
            None => "census (historical floor retired)",
            Some(historical) if self.census_total > historical => "census",
            Some(_) => "historical floor",
        }
    }

    /// The reserve and its parts for the placement log, every census term.
    pub fn describe(&self) -> String {
        match self.historical {
            Some(historical) => format!(
                "{} MiB = max({} MiB historical floor, {} MiB census [{}]): {} wins",
                self.bytes() >> 20,
                historical >> 20,
                self.census_total >> 20,
                self.census.describe(),
                self.winner()
            ),
            None => format!(
                "{} MiB = census [{}] (historical floor retired: the census carries the \
                 measured HIP runtime overhead)",
                self.bytes() >> 20,
                self.census.describe()
            ),
        }
    }
}

/// The largest trunk-layer count whose routed experts fit in `free_vram`
/// after the non-expert weights and `reserve` ([`auto_vram_reserve`]).
pub fn auto_vram_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
    reserve: u64,
) -> usize {
    let budget = free_vram
        .saturating_sub(non_expert_bytes)
        .saturating_sub(reserve);
    match budget.checked_div(layer_expert_bytes) {
        Some(layers) => (layers as usize).min(num_layers),
        None => num_layers,
    }
}

/// The trunk-layer count a whole-layer `auto` placement keeps resident:
/// [`auto_vram_layers`] against `base_reserve` ([`auto_vram_reserve`]) plus
/// every [`PlacementCharges`] term. Refuses, before anything is allocated,
/// when even zero resident layers cannot hold the non-expert weights and the
/// charged reserve (the load would fail at its first long request instead).
pub fn auto_placement_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
    base_reserve: u64,
    charges: &PlacementCharges,
) -> Result<usize, String> {
    const MIB: u64 = 1 << 20;
    let reserve = charges.reserve(base_reserve)?;
    if non_expert_bytes
        .checked_add(reserve)
        .is_none_or(|need| free_vram < need)
    {
        return Err(format!(
            "{} MiB of free VRAM cannot hold the {} MiB of non-expert weights plus the {} MiB reserve ({} MiB base + {}) even with every routed expert in host RAM; free VRAM on this GPU",
            free_vram / MIB,
            non_expert_bytes / MIB,
            reserve / MIB,
            base_reserve / MIB,
            charges.describe()
        ));
    }
    Ok(auto_vram_layers(
        free_vram,
        non_expert_bytes,
        layer_expert_bytes,
        num_layers,
        reserve,
    ))
}

/// Device bytes an expert-tier lease's frozen tables take beyond the payload:
/// the gate/up and down pointer tables (8 bytes per expert each) and the
/// 1-byte-per-expert codec table, over every trunk expert
/// (`expert_tier_owners::materialize`).
pub const EXPERT_TIER_TABLE_BYTES: u64 = crate::expert_slabs::TRUNK_LAYERS as u64
    * crate::expert_slabs::EXPERTS_PER_LAYER as u64
    * (2 * 8 + 1);

/// Device bytes kept free beyond every other budget term when sizing the
/// `HIPFIRE_QWEN4_EXPERT_HOTSET` payload (driver/allocator headroom).
pub const EXPERT_HOTSET_VRAM_MARGIN_BYTES: u64 = 512 << 20;

/// Device bytes the `HIPFIRE_QWEN4_EXPERT_HOTSET` payload may take:
/// `free_vram` less the non-expert weights, `reserve` ([`auto_vram_reserve`],
/// the caller's own value, not recomputed here), the lease's frozen tables
/// ([`EXPERT_TIER_TABLE_BYTES`]) and [`EXPERT_HOTSET_VRAM_MARGIN_BYTES`].
/// Every subtraction is checked; exactly zero left is `Ok(0)`, any shortfall
/// is `Err`.
pub fn expert_hotset_budget(
    free_vram: u64,
    non_expert_bytes: u64,
    reserve: u64,
) -> Result<u64, String> {
    free_vram
        .checked_sub(non_expert_bytes)
        .and_then(|bytes| bytes.checked_sub(reserve))
        .and_then(|bytes| bytes.checked_sub(EXPERT_TIER_TABLE_BYTES))
        .and_then(|bytes| bytes.checked_sub(EXPERT_HOTSET_VRAM_MARGIN_BYTES))
        .ok_or_else(|| {
            format!(
                "expert hotset has no device budget: free VRAM {free_vram} bytes is below the \
                 non-expert weights {non_expert_bytes} + reserve {reserve} + expert tier tables \
                 {EXPERT_TIER_TABLE_BYTES} + safety margin {EXPERT_HOTSET_VRAM_MARGIN_BYTES}"
            )
        })
}

/// Whether a load on `gpu` may build the long-prefill expert stage (the same
/// switch `Qwen4ExpertStage::new` reads). Admission and the constructor both
/// call this with the same `gpu`, so they share one input:
/// `HIPFIRE_QWEN4_EXPERT_STAGE` (`1`/`0`) when set, else
/// [`crate::host_link::expert_stage_default`] on the device's cached measured
/// host link. An explicit value or gfx1201 never triggers the H2D probe.
///
/// This probes (gfx1100 default), so it must never size an expert placement:
/// placement charges use [`expert_stage_placement_enabled`] and the stage
/// this switch alone turns on borrows donor layers (no dedicated VRAM).
pub fn expert_stage_enabled(gpu: &rdna_compute::Gpu) -> bool {
    expert_stage_switch(
        &gpu.arch,
        hipfire_config::developer_var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV)
            .ok()
            .as_deref(),
        || crate::host_link::host_link(gpu),
    )
}

/// [`expert_stage_enabled`] with the env value and the (lazily measured)
/// link supplied; the strict `1`/`0` spelling of `developer_bool`.
fn expert_stage_switch(
    arch: &str,
    env: Option<&str>,
    link: impl FnOnce() -> crate::host_link::HostLink,
) -> bool {
    match env {
        Some("1") => true,
        Some("0") => false,
        _ => crate::host_link::expert_stage_default(arch, link),
    }
}

/// The stage switch every placement charge reads: the explicit
/// `HIPFIRE_QWEN4_EXPERT_STAGE` value, else gfx1201 only. It is
/// [`expert_stage_enabled`] with the link taken as unmeasured, so it never
/// reads the host link (the gfx1100 narrow-link chunk is the only way the
/// link reaches the placement). A dedicated stage (slab or
/// tier layout) is therefore charged only where it can actually allocate
/// (`Qwen4ExpertStage::new` refuses those layouts otherwise); the borrowed
/// packed stage is charged nothing, it lends resident donor layers.
pub fn expert_stage_placement_enabled(arch: &str) -> bool {
    expert_stage_placement_switch(
        arch,
        hipfire_config::developer_var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV)
            .ok()
            .as_deref(),
    )
}

/// [`expert_stage_placement_enabled`] with the env value supplied.
fn expert_stage_placement_switch(arch: &str, env: Option<&str>) -> bool {
    expert_stage_switch(arch, env, || crate::host_link::HostLink::Unmeasured)
}

/// Device bytes the tier long-prefill staging allocates for `plan`: one VRAM
/// stage buffer per layer parity, each the largest host-tier payload of a
/// layer of that parity (projections 256-byte aligned) plus the zeroed tail
/// pad, as `gpu_forward::slab_stage` sizes them. An upper bound of what it
/// allocates, so the hot set budget can hold it.
pub fn expert_tier_stage_bytes(plan: &crate::expert_tier_plan::ExpertTierPlan) -> u64 {
    use crate::expert_slabs::{EXPERTS_PER_LAYER, TRUNK_LAYERS};
    use crate::expert_tier_plan::Placement;
    let align = |len: u64| len.div_ceil(256) * 256;
    let mut largest = [0u64; 2];
    for layer in 0..TRUNK_LAYERS as usize {
        let host: u64 = (0..EXPERTS_PER_LAYER as usize)
            .filter_map(|expert| plan.entry(layer, expert))
            .filter(|entry| entry.tier == Placement::HostMapped)
            .map(|entry| align(entry.gate_up_len) + align(entry.down_len))
            .sum();
        largest[layer % 2] = largest[layer % 2].max(host);
    }
    largest
        .iter()
        .map(|bytes| bytes + crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64)
        .sum()
}

/// Whether a whole-layer (slab arena) load allocates the two dedicated
/// long-prefill stage buffers at load: the stage switch is on
/// ([`expert_stage_enabled`]), the artifact is a slab-arena recipe (the
/// legacy aggregate layout borrows resident donor layers and allocates
/// nothing; a tier lease charges its own stage in the hot set fixed point)
/// and the load is not expert-parallel (`Qwen4ExpertStage::new_slab` has no
/// EP route). Not conditioned on the prefill length: the buffers exist from
/// load, even if every prompt stays under `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS`.
pub fn dedicated_slab_stage(stage_enabled: bool, slab_recipe: bool, expert_parallel: bool) -> bool {
    stage_enabled && slab_recipe && !expert_parallel
}

/// What a whole-layer placement (`auto`, or the unset fallback that decides
/// whether everything is resident) keeps free beyond [`AutoReserve::bytes`]
/// (the larger of the historical formula and the full-context census, which
/// already hold the chunk-sized forward resources and the pool workspace),
/// itemised for the placement log:
///
/// - `margin`: [`EXPERT_HOTSET_VRAM_MARGIN_BYTES`], the driver/allocator
///   safety margin the hot set budget keeps.
/// - `stage`: the two dedicated long-prefill stage buffers of a slab load,
///   `2 * (largest layer payload + ARENA_TAIL_PAD_BYTES)`
///   ([`dedicated_slab_stage`]); 0 when none are allocated.
/// - `slowlink`: the S1 slow-link prefill profile's complete census
///   ([`crate::slowlink_admission::SlowlinkCensus::total`]: tile arena, expert
///   page store, panel scratch, boundary state); 0 on every load that does
///   not select S1, which keeps that load's reserve and log byte for byte.
///
/// The hot set path does not use this: it subtracts its own margin and
/// tier-plan stage ([`expert_hotset_budget`], [`expert_tier_stage_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementCharges {
    pub margin: u64,
    pub stage: u64,
    pub slowlink: u64,
}

impl PlacementCharges {
    /// `layer_expert_bytes` is the largest layer payload (`resident_split`);
    /// `dedicated_stage` is [`dedicated_slab_stage`].
    pub fn new(layer_expert_bytes: u64, dedicated_stage: bool) -> Result<Self, String> {
        let stage = if dedicated_stage {
            layer_expert_bytes
                .checked_add(crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64)
                .and_then(|buffer| buffer.checked_mul(2))
                .ok_or("long-prefill stage buffer bytes overflow")?
        } else {
            0
        };
        Ok(Self {
            margin: EXPERT_HOTSET_VRAM_MARGIN_BYTES,
            stage,
            slowlink: 0,
        })
    }

    /// These charges plus the S1 profile's census `bytes`.
    pub fn with_slowlink(self, bytes: u64) -> Self {
        Self {
            slowlink: bytes,
            ..self
        }
    }

    /// `base` ([`auto_vram_reserve`]) plus every charge.
    pub fn reserve(&self, base: u64) -> Result<u64, String> {
        base.checked_add(self.margin)
            .and_then(|bytes| bytes.checked_add(self.stage))
            .and_then(|bytes| bytes.checked_add(self.slowlink))
            .ok_or_else(|| "auto expert VRAM reserve with placement charges overflows".to_string())
    }

    /// The charges for the placement log, in MiB.
    pub fn describe(&self) -> String {
        const MIB: u64 = 1 << 20;
        let base = format!(
            "{} MiB safety margin + {} MiB long-prefill stage buffers",
            self.margin / MIB,
            self.stage / MIB
        );
        if self.slowlink == 0 {
            base
        } else {
            format!(
                "{base} + {} MiB slow-link prefill (S1) arena, expert pages and boundary state",
                self.slowlink / MIB
            )
        }
    }
}

/// The long-prefill stage a `whole-first` load runs with
/// (`HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY`), the existing G2 stage unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeFirstStage {
    /// No stage: switch off, no host layer, or fewer than two resident layers
    /// to donate.
    None,
    /// Legacy aggregate (packed) layout: the stage borrows the last two
    /// resident whole layers as its DMA buffers. Nothing is allocated; the
    /// restore copies are pinned host RAM.
    Borrowed {
        donors: [usize; 2],
        /// Resident device bytes the two donors lend (no incremental VRAM).
        capacity: u64,
        /// Pinned host bytes the donors are restored from.
        restore_copies: u64,
    },
    /// Slab layout: two dedicated VRAM stage buffers, charged in
    /// [`PlacementCharges::stage`].
    Dedicated { buffer: u64, allocated: u64 },
}

impl WholeFirstStage {
    /// Device bytes the stage allocates beyond the resident layers.
    pub fn allocated_vram(&self) -> u64 {
        match self {
            Self::Dedicated { allocated, .. } => *allocated,
            Self::None | Self::Borrowed { .. } => 0,
        }
    }

    /// Resident device bytes the stage borrows from donor layers.
    pub fn borrowed_vram(&self) -> u64 {
        match self {
            Self::Borrowed { capacity, .. } => *capacity,
            Self::None | Self::Dedicated { .. } => 0,
        }
    }
}

/// The whole-layer placement `HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY=whole-first`
/// keeps: exactly the `hotset=0` whole-layer plan (no tier plan, owners or
/// sidecar), so the decode route is the whole-layer route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WholeFirstPlan {
    /// Whole resident layers: trunk layers `0..whole_layers`.
    pub whole_layers: usize,
    /// MoE layers of the model (trunk plus MTP).
    pub moe_layers: usize,
    /// Hot experts kept off a host layer: always 0 (no tier lease).
    pub hot_experts: usize,
    /// Largest layer payload the plan was budgeted with.
    pub layer_expert_bytes: u64,
    pub stage: WholeFirstStage,
}

impl WholeFirstPlan {
    /// The stage for `whole_layers` resident layers. `charges` are the
    /// loader's [`PlacementCharges`] for the load (a slab stage is charged in
    /// `charges.stage`); `borrowed_donors` is whether the stage is switched
    /// on and borrows donor layers (legacy aggregate recipe, no expert
    /// parallelism).
    pub fn stage_for(
        whole_layers: usize,
        moe_layers: usize,
        layer_expert_bytes: u64,
        charges: &PlacementCharges,
        borrowed_donors: bool,
    ) -> WholeFirstStage {
        if whole_layers >= moe_layers {
            WholeFirstStage::None
        } else if charges.stage > 0 {
            WholeFirstStage::Dedicated {
                buffer: charges.stage / 2,
                allocated: charges.stage,
            }
        } else if borrowed_donors && whole_layers >= 2 {
            let pair = layer_expert_bytes.saturating_mul(2);
            WholeFirstStage::Borrowed {
                donors: [whole_layers - 2, whole_layers - 1],
                capacity: pair,
                restore_copies: pair,
            }
        } else {
            WholeFirstStage::None
        }
    }

    /// The `auto` placement under `whole-first`: [`auto_placement_layers`]
    /// with the same inputs the `hotset=0` path passes (so never fewer whole
    /// layers), the stage that load runs with, and a check that the
    /// resident layers, the charged reserve (dedicated stage included) and
    /// the non-expert weights fit `free_vram`.
    #[allow(clippy::too_many_arguments)]
    pub fn auto(
        free_vram: u64,
        non_expert_bytes: u64,
        layer_expert_bytes: u64,
        num_layers: usize,
        moe_layers: usize,
        base_reserve: u64,
        charges: &PlacementCharges,
        borrowed_donors: bool,
    ) -> Result<Self, String> {
        let whole_layers = auto_placement_layers(
            free_vram,
            non_expert_bytes,
            layer_expert_bytes,
            num_layers,
            base_reserve,
            charges,
        )?;
        charges
            .reserve(base_reserve)
            .ok()
            .and_then(|reserve| reserve.checked_add(non_expert_bytes))
            .and_then(|bytes| {
                layer_expert_bytes
                    .checked_mul(whole_layers as u64)
                    .and_then(|layers| bytes.checked_add(layers))
            })
            .filter(|need| *need <= free_vram)
            .ok_or_else(|| {
                format!(
                    "whole-first plan of {whole_layers} layers does not fit {} MiB free VRAM",
                    free_vram >> 20
                )
            })?;
        Ok(Self {
            whole_layers,
            moe_layers,
            hot_experts: 0,
            layer_expert_bytes,
            stage: Self::stage_for(
                whole_layers,
                moe_layers,
                layer_expert_bytes,
                charges,
                borrowed_donors,
            ),
        })
    }

    /// The fully resident load (every MoE layer whole, no host layer, no
    /// stage).
    pub fn fully_resident(moe_layers: usize, layer_expert_bytes: u64) -> Self {
        Self {
            whole_layers: moe_layers,
            moe_layers,
            hot_experts: 0,
            layer_expert_bytes,
            stage: WholeFirstStage::None,
        }
    }

    /// The one log line (no newline, no prefix) of an opt-in load.
    /// `resident_bytes` is the measured routed-expert bytes left
    /// device-resident ([`resident_routed_expert_bytes`]). The layer range
    /// is half open, as the placement log prints it.
    pub fn describe(&self, resident_bytes: u64) -> String {
        const MIB: u64 = 1 << 20;
        let stage = match self.stage {
            WholeFirstStage::None => "stage allocated 0 MiB, borrowed 0 MiB".to_string(),
            WholeFirstStage::Borrowed {
                donors,
                capacity,
                restore_copies,
            } => format!(
                "stage allocated 0 MiB, borrowed {} MiB from resident layers {} and {} ({} MiB pinned host restore copies)",
                capacity / MIB,
                donors[0],
                donors[1],
                restore_copies / MIB
            ),
            WholeFirstStage::Dedicated { buffer, allocated } => format!(
                "stage allocated {} MiB (2 dedicated buffers of {} MiB, charged to the placement), borrowed 0 MiB",
                allocated / MIB,
                buffer / MIB
            ),
        };
        format!(
            "policy=whole-first, whole layers 0..{} ({} of {} MoE layers), hot experts {}, expert resident {} MiB, {stage}; hotset=0 whole-layer placement, sidecar not consulted",
            self.whole_layers,
            self.whole_layers,
            self.moe_layers,
            self.hot_experts,
            resident_bytes / MIB
        )
    }
}

/// Routed-expert bytes of the entries left [`WeightResidency::Resident`]
/// (after [`place_routed_experts`], what stays on the device).
pub fn resident_routed_expert_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    weights
        .iter()
        .filter(|entry| {
            entry.residency == WeightResidency::Resident
                && is_routed_expert(&entry.name)
                && !matches!(entry.policy, ShardPolicy::Tied { .. })
        })
        .try_fold(0u64, |total, entry| {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            total
                .checked_add(bytes)
                .ok_or_else(|| "resident routed-expert byte census overflows".to_string())
        })
}

/// Mark the routed-expert entries of trunk layers `>= vram_layers`, and of
/// the MTP layer, [`WeightResidency::HostMapped`]. Returns the number of
/// entries moved to host RAM.
pub fn place_routed_experts(weights: &mut [WeightEntry], vram_layers: usize) -> usize {
    let mut moved = 0;
    for entry in weights.iter_mut() {
        if entry.residency != WeightResidency::Resident || !is_routed_expert(&entry.name) {
            continue;
        }
        let trunk_resident =
            !entry.name.starts_with("mtp.") && entry.layer.is_some_and(|layer| layer < vram_layers);
        if !trunk_resident {
            entry.residency = WeightResidency::HostMapped;
            moved += 1;
        }
    }
    moved
}

/// Pinned host bytes the [`WeightResidency::HostMapped`] entries will take.
pub fn host_mapped_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in weights {
        if entry.residency == WeightResidency::HostMapped {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            total += bytes + HOST_MAPPED_PAD_BYTES;
        }
    }
    Ok(total)
}

/// Pinned host bytes the manifest will hold after
/// `place_routed_experts(weights, vram_layers)` and `with_expert_tiers`,
/// computed without mutating or cloning `weights` so every host RAM/GTT check
/// can run before the manifest changes. `with_expert_tiers` drops every trunk
/// routed-expert tensor (non-`mtp.` with a layer), whatever its placement, so
/// those are never counted. Counted, each with [`HOST_MAPPED_PAD_BYTES`]: every
/// other entry already [`WeightResidency::HostMapped`], and every other
/// [`WeightResidency::Resident`] routed entry [`place_routed_experts`] would
/// move to host (the MTP layer's experts). Sums are checked.
pub fn hotset_host_mapped_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    vram_layers: usize,
) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in weights {
        let routed = is_routed_expert(&entry.name);
        let mtp = entry.name.starts_with("mtp.");
        if routed && !mtp && entry.layer.is_some() {
            continue;
        }
        let host = match entry.residency {
            WeightResidency::HostMapped => true,
            WeightResidency::Resident => {
                routed && !(entry.layer.is_some_and(|layer| layer < vram_layers) && !mtp)
            }
            _ => false,
        };
        if !host {
            continue;
        }
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        total = bytes
            .checked_add(HOST_MAPPED_PAD_BYTES)
            .and_then(|charge| total.checked_add(charge))
            .ok_or_else(|| "hotset host-mapped byte total overflows".to_string())?;
    }
    Ok(total)
}

/// Pinned host bytes every rank of a `ranks`-way expert-parallel load takes for
/// its [`WeightResidency::HostMapped`] entries, indexed by rank (`bytes_of` is
/// the full source payload; a routed expert stack's share on rank `r` is its
/// per-expert bytes times that rank's `local_count`, so an uneven split gives
/// the low ranks more). Every rank pins its own copy in the same host RAM, so
/// [`check_host_ram`] and [`check_gtt_cap`] take the sum of the ranks that
/// have not pinned yet. `ranks == 1` is a single entry equal to
/// [`host_mapped_bytes`].
pub fn host_mapped_bytes_ranks(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    ranks: usize,
) -> Result<Vec<u64>, String> {
    if ranks == 0 {
        return Err("expert-parallel rank count must be at least 1".to_string());
    }
    let mut per_rank = vec![0u64; ranks];
    for entry in weights {
        if entry.residency == WeightResidency::HostMapped {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            for (rank, charge) in per_rank.iter_mut().enumerate() {
                *charge += rank_share(entry, bytes, rank, ranks)? + HOST_MAPPED_PAD_BYTES;
            }
        }
    }
    Ok(per_rank)
}

/// Two donor mirrors per rank share one host RAM/GTT budget. Each donor has
/// separate gate/up and down pinned allocations, including their padding.
pub(crate) fn expert_stage_mirror_bytes(layer_bytes: usize, ranks: usize) -> Result<u64, String> {
    (layer_bytes as u64)
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(4 * HOST_MAPPED_PAD_BYTES))
        .and_then(|bytes| bytes.checked_mul(ranks as u64))
        .filter(|_| ranks > 0)
        .ok_or_else(|| "expert stage mirror host charge overflow or zero ranks".to_string())
}

/// Rank `rank`'s share of an entry's payload: a routed expert stack (logical
/// shape `[n_experts, ..]`) holds its per-expert bytes times the rank's
/// `local_count` (experts `e % ranks == rank`), everything else is replicated.
/// An expert-slab arena (logical shape `[bytes]`, no expert axis) keeps its
/// even byte split; EP refuses slab manifests today
/// (`Qwen4Manifest::with_expert_parallel`).
fn rank_share(entry: &WeightEntry, bytes: u64, rank: usize, ranks: usize) -> Result<u64, String> {
    if ranks == 0 {
        return Err("expert-parallel rank count must be at least 1".to_string());
    }
    if ranks == 1 || !is_routed_expert(&entry.name) {
        return Ok(bytes);
    }
    if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
        let ranks = ranks as u64;
        if bytes % ranks != 0 {
            return Err(format!(
                "'{}' has {bytes} bytes, which do not split across {ranks} ranks",
                entry.name
            ));
        }
        return Ok(bytes / ranks);
    }
    if rank >= ranks {
        return Err(format!(
            "expert-parallel rank {rank} is outside {ranks} ranks"
        ));
    }
    let experts = entry.logical_shape.first().copied().unwrap_or(0);
    if experts < ranks {
        return Err(format!(
            "'{}' has {experts} routed experts, which cannot cover {ranks} ranks",
            entry.name
        ));
    }
    if bytes % experts as u64 != 0 {
        return Err(format!(
            "'{}' has {bytes} bytes, which are not a multiple of its {experts} experts",
            entry.name
        ));
    }
    let local = Qwen4EpPlacement { rank, ranks }.local_count(experts) as u64;
    Ok(bytes / experts as u64 * local)
}

/// MoE layers of the model: the trunk's plus the MTP head's.
pub fn moe_layer_count(config: &Qwen4Config) -> usize {
    config.num_hidden_layers + usize::from(config.mtp.is_some())
}

/// MoE layers one pipeline-parallel band owns: its trunk layers, plus the MTP
/// head's layer only on the output band. `Single` equals [`moe_layer_count`].
/// Expert-parallel counts never use this: they stay [`moe_layer_count`].
pub fn moe_layer_count_scope(config: &Qwen4Config, scope: &Qwen4BandScope) -> usize {
    scope.band.len() + usize::from(scope.output && config.mtp.is_some())
}

/// How many of the model's MoE layers (trunk first, the MTP layer last) one
/// rank with `free_vram` keeps resident: all of them when the trunk, every
/// layer's experts and the reserve fit, otherwise the trunk layers `auto`
/// fits (the MTP layer is then always host-mapped, as in
/// [`place_routed_experts`]). `layer_expert_bytes_per_rank` is one MoE
/// layer's routed experts on the largest rank (`max_local_count` experts), so
/// a layer counted resident fits every rank of an uneven split.
pub fn ep_rank_layers(
    config: &Qwen4Config,
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
) -> usize {
    let moe_layers = moe_layer_count(config);
    let all_experts = layer_expert_bytes_per_rank.saturating_mul(moe_layers as u64);
    if fits_fully_resident(free_vram, non_expert_bytes, all_experts, reserve_bytes) {
        return moe_layers;
    }
    auto_vram_layers(
        free_vram,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        config.num_hidden_layers,
        reserve_bytes,
    )
}

/// The expert VRAM split every rank of an expert-parallel load applies: the
/// minimum over ranks of [`ep_rank_layers`], so each layer is resident on all
/// ranks or on none (the full prefill table relies on equal residency per
/// layer across ranks). A count of [`moe_layer_count`] means fully resident.
/// `free_per_rank` is every rank's free VRAM before the load; the result is
/// the `expert_vram_layers` override passed to every rank's load.
pub fn ep_vram_layers(
    config: &Qwen4Config,
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
) -> usize {
    free_per_rank
        .iter()
        .map(|&free| {
            ep_rank_layers(
                config,
                free,
                non_expert_bytes,
                layer_expert_bytes_per_rank,
                reserve_bytes,
            )
        })
        .min()
        .unwrap_or(0)
}

/// One rank's line of the expert-parallel fit table: free VRAM, the
/// replicated trunk, the rank's routed-expert bytes (`MoE layers` of
/// `layer_expert_bytes_per_rank` each), the split applied (`applied_layers`
/// of the MoE layers resident, the rest pinned host RAM; the MTP layer is the
/// last), what this rank alone could keep, the reserve and the predicted
/// headroom. `layer_expert_bytes_per_rank` is the largest rank's layer bytes
/// (`max_local_count` experts), as every rank budgets it.
pub fn ep_fit_rank_line(
    config: &Qwen4Config,
    rank: usize,
    ranks: usize,
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
    applied_layers: usize,
) -> String {
    let moe_layers = moe_layer_count(config);
    let applied = applied_layers.min(moe_layers);
    let own = ep_rank_layers(
        config,
        free_vram,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        reserve_bytes,
    );
    let resident_bytes = layer_expert_bytes_per_rank.saturating_mul(applied as u64);
    let headroom = i128::from(free_vram)
        - i128::from(non_expert_bytes)
        - i128::from(resident_bytes)
        - i128::from(reserve_bytes);
    let gib = |bytes: u64| bytes as f64 / GIB;
    format!(
        "qwen4 EP rank {rank}/{ranks}: free {:.2} GiB | trunk {:.2} GiB | experts {:.2} GiB/rank \
         ({moe_layers} MoE layers x {:.1} MiB) | resident expert layers {applied} of {moe_layers} \
         ({} host-mapped; this rank alone fits {own}) | reserve {:.2} GiB | headroom {:+.2} GiB",
        gib(free_vram),
        gib(non_expert_bytes),
        gib(layer_expert_bytes_per_rank.saturating_mul(moe_layers as u64)),
        layer_expert_bytes_per_rank as f64 / (1u64 << 20) as f64,
        moe_layers - applied,
        gib(reserve_bytes),
        headroom as f64 / GIB,
    )
}

/// The expert-parallel fit table for `ranks` cards (`free_per_rank[r]` is rank
/// `r`'s free VRAM): a header naming the context and chunk the reserve was
/// sized for and the split every rank applies ([`ep_vram_layers`], the minimum
/// over ranks), then one [`ep_fit_rank_line`] per rank at that split.
/// `non_expert_bytes` is the replicated trunk, `layer_expert_bytes_per_rank`
/// one MoE layer's routed experts on the largest rank (`resident_split_ranks`),
/// `reserve_bytes` ([`auto_vram_reserve`]) what `auto` keeps free.
pub fn ep_fit_report(
    config: &Qwen4Config,
    ranks: usize,
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
    max_seq: usize,
    chunk_rows: usize,
) -> String {
    let split = ep_vram_layers(
        config,
        free_per_rank,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        reserve_bytes,
    );
    let mut report = format!(
        "qwen4 EP fit: {ranks} ranks, max_seq {max_seq}, prefill chunk {chunk_rows} rows, \
         VRAM split {split} of {} MoE layers on every rank (minimum over ranks)",
        moe_layer_count(config)
    );
    for (rank, free) in free_per_rank.iter().enumerate() {
        report.push('\n');
        report.push_str(&ep_fit_rank_line(
            config,
            rank,
            ranks,
            *free,
            non_expert_bytes,
            layer_expert_bytes_per_rank,
            reserve_bytes,
            split,
        ));
    }
    report
}

/// Refuse before any allocation when the pinned experts would not leave
/// [`HOST_RAM_HEADROOM_BYTES`] of the host RAM they can take: `mem_available`
/// (`MemAvailable - CMA`; CMA pageblocks serve only movable allocations)
/// plus `ttm_pool_bytes`, the estimate of freed GTT pages parked in TTM's
/// page pool ([`ttm_pool_estimate`]). Pinned pages cannot be reclaimed, so
/// over-committing them starves the rest of the host.
pub fn check_host_ram(
    host_bytes: u64,
    mem_available: Option<u64>,
    ttm_pool_bytes: u64,
) -> Result<(), String> {
    if host_bytes == 0 {
        return Ok(());
    }
    let Some(available) = mem_available else {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM, but MemAvailable is unreadable",
            host_bytes as f64 / GIB
        ));
    };
    let needed = host_bytes + HOST_RAM_HEADROOM_BYTES;
    if available.saturating_add(ttm_pool_bytes) < needed {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM plus {:.0} GiB headroom, but \
             MemAvailable - CMA is {:.1} GiB (plus {:.1} GiB estimated in TTM's page pool); free host \
             memory or keep more expert layers in VRAM ({EXPERT_VRAM_LAYERS_ENV})",
            host_bytes as f64 / GIB,
            HOST_RAM_HEADROOM_BYTES as f64 / GIB,
            available as f64 / GIB,
            ttm_pool_bytes as f64 / GIB
        ));
    }
    Ok(())
}

fn host_mapped_is_gtt() -> bool {
    std::env::var("HSA_USERPTR_FOR_PAGED_MEM").is_ok_and(|value| value.trim() == "0")
}

/// Host RAM held in TTM's page pool that the host-mapped experts can take
/// ([`rdna_compute::uma_memory::ttm_pool_bytes`]). amdgpu parks the
/// write-combined and uncached pages of freed GTT buffers there, up to
/// `page_pool_size` (half of RAM by default), so a Flash-Next process that
/// exits leaves its host-mapped experts' pages in it. A new GTT allocation
/// takes pages from the pool first, and TTM's shrinker frees it under memory
/// pressure (`echo 2 | sudo tee /proc/sys/vm/drop_caches` empties it at
/// once). 0 unless host-mapped memory is GTT-backed
/// (`HSA_USERPTR_FOR_PAGED_MEM=0`), or when sysfs or `/proc/meminfo` is
/// unreadable. Unified-memory sizing counts the pool through
/// `Gpu::device_mem_info` instead.
pub fn ttm_pool_estimate() -> u64 {
    if !host_mapped_is_gtt() {
        return 0;
    }
    rdna_compute::uma_memory::ttm_pool_bytes()
}

/// GTT room on the host: TTM's cap and what amdgpu devices already hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GttBudget {
    pub limit_bytes: u64,
    pub used_bytes: u64,
}

/// The GTT budget when host-mapped memory is GTT-backed, i.e. when
/// `HSA_USERPTR_FOR_PAGED_MEM` is `0` (hip-bridge's default). It is `None`
/// under userptr, or when sysfs does not expose TTM's limit. `used_bytes`
/// sums `mem_info_gtt_used` over every amdgpu device.
pub fn gtt_budget() -> Option<GttBudget> {
    if !host_mapped_is_gtt() {
        return None;
    }
    let limit_bytes = read_sysfs_u64(TTM_PAGES_LIMIT.as_ref())?.checked_mul(TTM_PAGE_BYTES)?;
    Some(GttBudget {
        limit_bytes,
        used_bytes: amdgpu_gtt_used_bytes()?,
    })
}

/// Refuse before any allocation when GTT-backed host-mapped experts would not
/// fit under TTM's `pages_limit` beside what amdgpu devices already hold.
/// Past it `hipHostMalloc` fails, after the load has uploaded the VRAM
/// weights. `None` (userptr, or no TTM limit in sysfs) skips the check.
pub fn check_gtt_cap(host_bytes: u64, budget: Option<GttBudget>) -> Result<(), String> {
    let Some(GttBudget {
        limit_bytes,
        used_bytes,
    }) = budget
    else {
        return Ok(());
    };
    let free = limit_bytes.saturating_sub(used_bytes);
    if host_bytes > free {
        return Err(format!(
            "routed experts need {:.1} GiB of GTT-backed host RAM, but TTM's GTT cap \
             ({TTM_PAGES_LIMIT} = {} pages, {:.1} GiB) has {:.1} GiB left beside the {:.1} GiB \
             amdgpu devices already hold; keep more expert layers in VRAM \
             ({EXPERT_VRAM_LAYERS_ENV}), raise ttm.pages_limit, or set \
             HSA_USERPTR_FOR_PAGED_MEM=1 for pageable userptr host memory, which host-memory \
             pressure can stall (ROCm/rocm-systems#12528)",
            host_bytes as f64 / GIB,
            limit_bytes / TTM_PAGE_BYTES,
            limit_bytes as f64 / GIB,
            free as f64 / GIB,
            used_bytes as f64 / GIB
        ));
    }
    Ok(())
}

/// Everything [`placement_inputs`] reads: the load's manifest and payload
/// sizes, the state and scratch plan it admitted, and the context geometry.
pub struct PlacementInputsArgs<'a> {
    pub gpu: &'a rdna_compute::Gpu,
    pub config: &'a Qwen4Config,
    pub weights: &'a [WeightEntry],
    pub bytes_of: &'a dyn Fn(&WeightEntry) -> Option<u64>,
    pub backend: crate::Qwen4KvBackend,
    /// Unified memory (the context is charged as committed, not legacy).
    pub use_ranges: bool,
    pub max_seq: usize,
    pub vmm_granularity: usize,
    /// The native MTP head stays attached after the placement
    /// (`native_mtp` and the host-mapped MTP policy kept it).
    pub mtp_kept: bool,
    pub max_k: usize,
    /// The profile has routed experts (not a dense FFN).
    pub routed_experts: bool,
    pub state_format: crate::Qwen4StateFormat,
    pub scratch_mode: Qwen4ForwardScratchMode,
    pub ranks: usize,
    /// The chunk the reserve is sized from; never probes the host link here.
    pub chunk_rows: usize,
}

/// What the expert placement is sized from, for one rank of a `ranks`-way load.
#[derive(Clone, Copy, Debug)]
pub struct PlacementInputs {
    /// Replicated non-expert bytes.
    pub non_expert: u64,
    /// One MoE layer's routed-expert bytes on this rank (the largest layer).
    pub layer_experts: u64,
    pub reserve: AutoReserve,
    pub mtp_bytes: Option<u64>,
    pub gather_bytes: Option<u64>,
}

/// The `auto` reserve ([`AutoReserve`]: the larger of the historical measured
/// formula and a bottom-up census of the first full-context request,
/// itemised; the census alone where it carries the measured HIP runtime
/// overhead, [`runtime_overhead_bytes`]) with its native MTP and QSA gather
/// parts. The one place this is computed: the product loader, the
/// expert-parallel split and every example/gate that asks the S1 admission for
/// a profile read it, so there is one convention.
pub fn placement_inputs(args: &PlacementInputsArgs<'_>) -> Result<PlacementInputs, String> {
    let PlacementInputsArgs {
        gpu,
        config,
        weights,
        bytes_of,
        backend,
        use_ranges,
        max_seq,
        vmm_granularity,
        mtp_kept,
        max_k,
        routed_experts,
        state_format,
        scratch_mode,
        ranks,
        chunk_rows,
    } = *args;
    let qsa_format = state_format.qsa;
    let (non_expert, layer_experts) = resident_split_ranks(weights, bytes_of, ranks)
        .map_err(|error| format!("qwen4: {error}"))?;
    // `chunk_rows` is the caller's: it never probes the host link here.
    let context = crate::Qwen4ContextCommit::for_expert_reserve(
        backend,
        use_ranges,
        max_seq,
        chunk_rows,
        vmm_granularity,
    );
    // The context the census charges: the ACTUAL backend with the whole
    // admitted context mapped (a VMM arena maps ahead to `max_seq`, rounded
    // to the driver granularity), not the legacy-forced layout the historical
    // formula charges on a discrete card.
    let census_context = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, vmm_granularity);
    let mtp_for = |context: &crate::Qwen4ContextCommit| -> Result<Option<u64>, String> {
        if !mtp_kept {
            return Ok(None);
        }
        let head = language_head_dtype(weights).ok_or("qwen4: manifest has no language head")?;
        let drafts = crate::mtp_spec::native_mtp_proposal_drafts(max_k, &gpu.arch);
        let row_capture =
            crate::mtp_spec::native_mtp_capture(gpu, config, state_format.gdn, drafts);
        Ok(Some(
            crate::mtp_spec::native_mtp_device_bytes(
                config,
                context,
                chunk_rows,
                drafts,
                head,
                row_capture,
                crate::mtp_spec::native_mtp_fill_scratch(gpu),
            )
            .ok_or("qwen4: native MTP device bytes overflow")?,
        ))
    };
    // The gathered QSA prefill attention's workspace, committed as the
    // context is (none when the route is off: the reserve is then unchanged).
    // Bytes already committed by an earlier load in this process are out of
    // `free` and only growth past them is charged.
    let gather_for = |context: &crate::Qwen4ContextCommit| -> Result<Option<u64>, String> {
        rdna_compute::tensor_ops::qsa_gather_workspace_needed(
            gpu,
            qsa_format,
            state_format.qsa_f16_arena,
        )
        .then(|| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                context.tokens,
            )
            .map(|bytes| (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64))
            .ok_or("qwen4: QSA gather scratch size overflows".to_string())
        })
        .transpose()
    };
    let mtp_bytes = mtp_for(&context)?;
    let gather_bytes = gather_for(&context)?;
    // The measured HIP runtime overhead (exact gfx1201, discrete,
    // single-device): the census then carries it and the historical floor,
    // whose stale F32 context used to cover it, is retired.
    let runtime = runtime_overhead_bytes(&gpu.arch, use_ranges, ranks);
    let historical = if runtime.is_some() {
        None
    } else {
        Some(
            auto_vram_reserve(
                config,
                &context,
                chunk_rows,
                scratch_mode,
                state_format.layout(),
                mtp_bytes,
                gather_bytes,
            )
            .map_err(|error| format!("qwen4: {error}"))?,
        )
    };
    // The bottom-up census at the full admitted context.
    let state = state_census(
        config,
        state_format,
        &census_context,
        chunk_rows,
        scratch_mode,
    )
    .map_err(|error| format!("qwen4: {error}"))?;
    // Lazy F16 weight shadows of the BF16 owners the long-prefill F16 WMMA
    // arms read (exact gfx1201): the cold bound from the manifest, not the
    // warm cache.
    // The F16 WMMA arms run only at `route_rows >= 512`
    // (`gemm.rs` `qwen4_f16_wmma_applies`); no forward has more rows than the
    // chunk and gfx1201 has no canonical-row routing scope, so a smaller chunk
    // never creates a shadow.
    let f16_shadows = if gpu.qwen4_f16_wmma_gfx1201()
        && chunk_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS
    {
        f16_shadow_bytes(weights, config.num_hidden_layers)
            .map_err(|error| format!("qwen4: {error}"))?
    } else {
        0
    };
    // The kernels' pool workspaces at the largest requested chunk, exactly
    // (every short-tail remainder included). The dense GDN scan's scratch is
    // charged wherever `fn_gdn_dense::applies` can hold at load: exact gfx1201,
    // Q8 GDN state and the pinned 16-key/48-value-head, 128-dim, 10240-wide
    // geometry (its opt-out flag only lowers the charge, so it is ignored).
    let dense_gdn = gpu.arch == "gfx1201"
        && state_format.gdn == crate::GdnStateFormat::Q8
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
            == 10240;
    // The routed-expert I4 MoE terms exist only for a profile whose routed
    // experts run the grouped I4 route; a dense or synthetic profile
    // (no experts, or dimensions that route cannot take) charges none of
    // them and is never refused for them.
    let routed_i4 = routed_experts
        && config.num_experts > 0
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
    // Overflow (or a dimension an owning allocator rejects) refuses the load
    // with a clear error; there is no historical-only fallback.
    let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk_rows, &dims)
        .ok_or("qwen4: prefill workspace bytes cannot be computed (arithmetic overflow, or a hidden/MoE dimension the kernel allocators reject)")?;
    let census = ReserveCensus {
        qsa_context: state.qsa_context,
        state: state.state,
        forward: state.forward,
        f16_shadows,
        workspace,
        // The split QSA selector's score/filter scratch (0 unless its route
        // can run).
        qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_scratch_bytes(
            gpu,
            state_format.layout().index_dtype(),
        ) as u64,
        mtp: mtp_for(&census_context)?.unwrap_or(0),
        gather: gather_for(&census_context)?.unwrap_or(0),
        runtime: runtime.unwrap_or(0),
    };
    let reserve =
        AutoReserve::new(historical, census).map_err(|error| format!("qwen4: {error}"))?;
    Ok(PlacementInputs {
        non_expert,
        layer_experts,
        reserve,
        mtp_bytes,
        gather_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_forward::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use rdna_compute::tensor_ops::QsaKvFormat::{self, F32};
    use rdna_compute::uma_memory::ttm_pool_estimate_from;
    use rdna_compute::DType;

    fn entry(name: &str, layer: usize) -> WeightEntry {
        let mut entry = WeightEntry::model(
            name,
            vec![4, 4, 256],
            DType::MQ4G256V2,
            ShardPolicy::Replicate,
        );
        entry.layer = Some(layer);
        entry
    }

    fn arena_entry(layer: usize, bytes: usize) -> WeightEntry {
        let mut entry = WeightEntry::model(
            format!("model.language_model.layers.{layer}.mlp.experts.slab_arena"),
            vec![bytes],
            DType::Raw,
            ShardPolicy::Replicate,
        );
        entry.layer = Some(layer);
        entry
    }

    #[test]
    fn slab_arenas_are_routed_experts_and_the_layer_charge_is_the_largest_arena() {
        // Non-uniform arenas (the hot set is not spread evenly): layer 0 is the
        // smallest, layer 2 the largest. `element_bytes` doubles the shape product.
        let mut weights = vec![
            arena_entry(0, 100),
            arena_entry(1, 300),
            arena_entry(2, 700),
            arena_entry(3, 200),
            entry("model.language_model.layers.0.self_attn.q_proj.weight", 0),
        ];
        assert!(weights[..4]
            .iter()
            .all(|entry| is_routed_expert(&entry.name)));
        let (non_expert, layer_experts) = resident_split(&weights, |entry| {
            Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
        })
        .expect("split");
        assert_eq!(
            layer_experts,
            700 * 2,
            "the charge is the max arena, never layer 0's"
        );
        assert_eq!(non_expert, 4 * 4 * 256 * 2);
        // Admission totals every arena's physical bytes.
        assert_eq!(
            routed_expert_bytes(&weights, |entry| Some(
                entry.logical_shape.iter().product::<usize>() as u64 * 2
            )),
            Ok((100 + 300 + 700 + 200) * 2)
        );
        // Per-rank shares of a max arena split the same way.
        let (_, per_rank) = resident_split_ranks(
            &weights,
            |entry| Some(entry.logical_shape.iter().product::<usize>() as u64 * 2),
            2,
        )
        .expect("split");
        assert_eq!(per_rank, 700);
        // Layers past the budget host-map their arenas.
        assert_eq!(place_routed_experts(&mut weights, 2), 2);
        assert_eq!(weights[0].residency, WeightResidency::Resident);
        assert_eq!(weights[1].residency, WeightResidency::Resident);
        assert_eq!(weights[2].residency, WeightResidency::HostMapped);
        assert_eq!(weights[3].residency, WeightResidency::HostMapped);
    }

    #[test]
    fn layers_past_the_budget_and_mtp_go_to_host() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
            entry(
                "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
                1,
            ),
            entry("model.language_model.layers.2.mlp.experts.gate_up_proj", 2),
            entry("mtp.layers.0.mlp.experts.down_proj", 0),
        ];
        assert_eq!(place_routed_experts(&mut weights, 2), 2);
        let host: Vec<_> = weights
            .iter()
            .filter(|entry| entry.residency == WeightResidency::HostMapped)
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(
            host,
            [
                "model.language_model.layers.2.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.experts.down_proj"
            ]
        );
    }

    #[test]
    fn hotset_host_mapped_bytes_matches_place_then_drop_trunk_routed() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.2.mlp.experts.down_proj", 2),
            entry(
                "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
                1,
            ),
            entry("mtp.layers.0.mlp.experts.gate_up_proj", 0),
            entry("mtp.layers.0.mlp.experts.down_proj", 0),
            entry("model.language_model.embed_tokens.weight", 0),
        ];
        weights[5].layer = None;
        weights[5].residency = WeightResidency::HostMapped;
        let bytes_of = |entry: &WeightEntry| Some(entry.name.len() as u64 * 1000);
        let before: Vec<_> = weights.iter().map(|entry| entry.residency).collect();
        for vram_layers in [0usize, 1, 3] {
            let got = hotset_host_mapped_bytes(&weights, bytes_of, vram_layers).unwrap();
            let mut placed = weights.clone();
            place_routed_experts(&mut placed, vram_layers);
            placed.retain(|entry| {
                !(is_routed_expert(&entry.name)
                    && !entry.name.starts_with("mtp.")
                    && entry.layer.is_some())
            });
            assert_eq!(
                got,
                host_mapped_bytes(&placed, bytes_of).unwrap(),
                "vram_layers={vram_layers}"
            );
        }
        // MTP experts and the preexisting host entry only; the input never mutates.
        let expected =
            (weights[3].name.len() + weights[4].name.len() + weights[5].name.len()) as u64 * 1000
                + 3 * HOST_MAPPED_PAD_BYTES;
        assert_eq!(
            hotset_host_mapped_bytes(&weights, bytes_of, 3),
            Ok(expected)
        );
        assert_eq!(
            before,
            weights
                .iter()
                .map(|entry| entry.residency)
                .collect::<Vec<_>>()
        );
        assert!(hotset_host_mapped_bytes(&weights, |_| None, 3).is_err());
        assert!(hotset_host_mapped_bytes(&weights, |_| Some(u64::MAX), 3).is_err());
    }

    #[test]
    fn expert_hotset_budget_subtracts_weights_reserve_tables_and_margin_checked() {
        assert_eq!(EXPERT_TIER_TABLE_BYTES, 2 * 24576 * 8 + 24576);
        assert_eq!(EXPERT_HOTSET_VRAM_MARGIN_BYTES, 512 << 20);
        let (free, non_expert, reserve) = (30u64 << 30, 5_364_000_000u64, 6_656u64 << 20);
        assert_eq!(
            expert_hotset_budget(free, non_expert, reserve).unwrap(),
            free - non_expert - reserve - EXPERT_TIER_TABLE_BYTES - EXPERT_HOTSET_VRAM_MARGIN_BYTES
        );
        // Exactly zero left is a (useless but valid) budget; one byte short is not.
        let exact =
            non_expert + reserve + EXPERT_TIER_TABLE_BYTES + EXPERT_HOTSET_VRAM_MARGIN_BYTES;
        assert_eq!(expert_hotset_budget(exact, non_expert, reserve).unwrap(), 0);
        assert_eq!(
            expert_hotset_budget(exact + 1, non_expert, reserve).unwrap(),
            1
        );
        assert!(expert_hotset_budget(exact - 1, non_expert, reserve).is_err());
        assert!(expert_hotset_budget(0, 1, 0).is_err());
        assert!(expert_hotset_budget(u64::MAX, u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn expert_hotset_budget_margin_alone_can_exhaust_the_budget() {
        // Everything but the margin fits; the margin shortfall must not underflow.
        let free = EXPERT_TIER_TABLE_BYTES + EXPERT_HOTSET_VRAM_MARGIN_BYTES - 1;
        let err = expert_hotset_budget(free, 0, 0).unwrap_err();
        assert!(err.contains("safety margin"), "{err}");
        // Tables fit but not the margin, with weights and reserve consuming the rest.
        assert!(expert_hotset_budget(1 << 20, 0, 0).is_err());
        assert_eq!(
            expert_hotset_budget(
                EXPERT_TIER_TABLE_BYTES + EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                0,
                0
            )
            .unwrap(),
            0
        );
        // Full-range inputs do not overflow the chain.
        assert_eq!(
            expert_hotset_budget(u64::MAX, 0, 0).unwrap(),
            u64::MAX - EXPERT_TIER_TABLE_BYTES - EXPERT_HOTSET_VRAM_MARGIN_BYTES
        );
    }

    #[test]
    fn f16_arena_reserve_covers_full_capacity_even_with_vmm() {
        let config = crate::config::compact_test_config();
        let layout = rdna_compute::tensor_ops::QsaStateLayout {
            f16_arena: true,
            ..rdna_compute::tensor_ops::QsaStateLayout::F32
        };
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let context = crate::Qwen4ContextCommit::new(backend, 262145, 8192, 2 << 20);
            let plain = auto_vram_reserve(&config, &context, 8192, Full, F32, None, None).unwrap();
            let with =
                auto_vram_reserve(&config, &context, 8192, Full, layout, None, None).unwrap();
            let extra = if backend == crate::Qwen4KvBackend::Vmm {
                2 * (262148usize * 1024).div_ceil(2 << 20) * (2 << 20) * config.n_full_layers()
            } else {
                262148 * layout.f16_arena_bytes_per_token(2, 256) * config.n_full_layers()
            };
            assert_eq!(with - plain, extra as u64);
        }
    }

    #[test]
    fn pooled_bf16_reserve_covers_full_capacity_even_with_vmm() {
        use hipfire_runtime::kv_backend::{
            KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
        };
        let config = crate::config::compact_test_config();
        let layout = rdna_compute::tensor_ops::QsaStateLayout {
            pooled_bf16: true,
            ..rdna_compute::tensor_ops::QsaStateLayout::F32
        };
        let blocks = 262145usize.div_ceil(4);
        let block_bytes = layout.pooled_bf16_bytes_per_block(128);
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let context = crate::Qwen4ContextCommit::new(backend, 262145, 8192, 2 << 20);
            let plain = auto_vram_reserve(&config, &context, 8192, Full, F32, None, None).unwrap();
            let with =
                auto_vram_reserve(&config, &context, 8192, Full, layout, None, None).unwrap();
            let per_layer = if backend == crate::Qwen4KvBackend::Vmm {
                KvChunkPlan::new(
                    block_bytes,
                    blocks,
                    DEFAULT_KV_CHUNK_TOKENS,
                    2 << 20,
                    DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
                )
                .and_then(|plan| plan.mapped_bytes_for_tokens(blocks))
                .unwrap()
            } else {
                blocks * block_bytes
            };
            assert_eq!(with - plain, (per_layer * config.n_full_layers()) as u64);
        }
    }

    #[test]
    fn auto_fits_the_measured_gfx1201_card() {
        // R9700 before load: 32548 MiB free; Flash-Next non-expert weights
        // 5.364 GB; one trunk layer's routed experts 1.3369 GB.
        let free = 32548u64 << 20;
        let config = crate::config::compact_test_config();
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let reserve =
            |chunk| auto_vram_reserve(&config, &measured, chunk, Full, F32, None, None).unwrap();
        // Verify-sized spec logits free 1472 logit rows of the measured
        // 1536-row layout: one layer past the measured N = 16 fits.
        let measured_chunk = reserve(AUTO_VRAM_RESERVE_CHUNK);
        assert_eq!(
            auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48, measured_chunk),
            17
        );
        // gfx1201's 4096-row chunk spends that and one layer more on
        // forward scratch.
        let reserve = reserve(crate::gpu_forward::qwen4_prefill_chunk_default(
            "gfx1201",
            || crate::host_link::HostLink::Unmeasured,
        ));
        assert!(reserve > measured_chunk);
        assert_eq!(
            auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48, reserve),
            15
        );
        // A card that holds everything keeps every layer resident.
        assert_eq!(
            auto_vram_layers(u64::MAX / 2, 5_364_000_000, 1_336_900_000, 48, reserve),
            48
        );
        // No room past the reserve places every expert in host RAM.
        assert_eq!(
            auto_vram_layers(8 << 30, 5_364_000_000, 1_336_900_000, 48, reserve),
            0
        );
    }

    /// The card-E record (daemon p6, 763 tokens): 32550 MiB free, 5115 MiB
    /// non-expert, 1060 MiB per slab layer, 7927 MiB historical reserve.
    const EVIDENCE: (u64, u64, u64, u64) = (32550 << 20, 5115 << 20, 1060 << 20, 7927 << 20);

    /// A census of the five terms given, a workspace of 153,256,072 B (the
    /// 4096-row target table, in its first term) and zero MTP and gather.
    fn census_with(
        qsa_context: u64,
        state: u64,
        forward: u64,
        f16_shadows: u64,
        qsa_selector: u64,
    ) -> ReserveCensus {
        let mut workspace = [("workspace", 0u64); WORKSPACE_TERM_COUNT];
        workspace[0].1 = 153_256_072;
        ReserveCensus {
            qsa_context,
            state,
            forward,
            f16_shadows,
            workspace,
            qsa_selector,
            mtp: 0,
            gather: 0,
            runtime: 0,
        }
    }

    /// The target profile's actual state formats: fp8 QSA K/V, Q8 GDN.
    fn target_state_format() -> crate::Qwen4StateFormat {
        crate::Qwen4StateFormat {
            qsa: QsaKvFormat::Fp8,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        }
    }

    /// The 48-layer target's F16-shadowed owners (BF16, rank 2): the final
    /// and every layer's two hyper-connection down mixes (97), each router
    /// and shared gate/up/down (48 + 144), and the PLE layer's key/value.
    fn target_f16_manifest() -> Vec<WeightEntry> {
        let layer = |i: usize, suffix: &str| format!("model.language_model.layers.{i}.{suffix}");
        let mut weights = vec![bf16(
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            &[320, 10240],
        )];
        for i in 0..48 {
            weights.push(bf16(
                &layer(i, "attn_hyper_connection.input_mix_weight_down.weight"),
                &[320, 10240],
            ));
            weights.push(bf16(
                &layer(i, "mlp_hyper_connection.input_mix_weight_down.weight"),
                &[320, 10240],
            ));
            weights.push(bf16(&layer(i, "mlp.gate.weight"), &[512, 2560]));
            weights.push(bf16(
                &layer(i, "mlp.shared_expert.gate_proj.weight"),
                &[640, 2560],
            ));
            weights.push(bf16(
                &layer(i, "mlp.shared_expert.up_proj.weight"),
                &[640, 2560],
            ));
            weights.push(bf16(
                &layer(i, "mlp.shared_expert.down_proj.weight"),
                &[2560, 640],
            ));
        }
        weights.push(bf16(&layer(2, "ple.key_proj.weight"), &[6400, 2560]));
        weights.push(bf16(&layer(2, "ple.value_proj.weight"), &[6400, 2560]));
        weights
    }

    /// The full-context census of the target profile on a VMM discrete card
    /// at `max_seq` (native MTP off), every term from its owning query, for a
    /// prefill `chunk` of rows, the forward `mode` it allocates and a
    /// `runtime` overhead term. The F16 shadows are charged only at chunks of
    /// at least [`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`] rows, as
    /// the loader does.
    fn census_at(
        config: &Qwen4Config,
        max_seq: usize,
        chunk: usize,
        mode: Qwen4ForwardScratchMode,
        runtime: u64,
    ) -> ReserveCensus {
        let format = target_state_format();
        let context =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, max_seq, max_seq, 2 << 20);
        let state = state_census(config, format, &context, chunk, mode).unwrap();
        let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
            hidden_size: config.hidden_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hc_count: config.hc_count,
            hc_lowrank: config.hc_lowrank,
            dense_gdn: true,
            routed_i4: true,
            include_short_tails: true,
        };
        let f16_shadows = if chunk >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            f16_shadow_bytes(&target_f16_manifest(), config.num_hidden_layers).unwrap()
        } else {
            0
        };
        ReserveCensus {
            qsa_context: state.qsa_context,
            state: state.state,
            forward: state.forward,
            f16_shadows,
            workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims).unwrap(),
            qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_capacity_bytes() as u64,
            mtp: 0,
            gather: rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                max_seq,
            )
            .unwrap() as u64,
            runtime,
        }
    }

    /// The full-context census at the default 4096-row chunk, `Full` forward,
    /// no runtime term: what the census was before the runtime overhead.
    fn real_census(config: &Qwen4Config, max_seq: usize) -> ReserveCensus {
        census_at(
            config,
            max_seq,
            crate::gpu_forward::qwen4_prefill_chunk_default("gfx1201", || {
                crate::host_link::HostLink::Unmeasured
            }),
            Full,
            0,
        )
    }

    /// Placement of the reproduction card with the real census: the layers
    /// the unfixed planner kept (`historical` alone), what stage and margin
    /// on the historical floor leave, and what the census-backed reserve
    /// leaves.
    fn replay(
        free: u64,
        non_expert: u64,
        layer: u64,
        historical: u64,
        census: ReserveCensus,
    ) -> (usize, usize, usize, AutoReserve, PlacementCharges) {
        let reserve = AutoReserve::new(Some(historical), census).unwrap();
        let charges = PlacementCharges::new(layer, true).unwrap();
        let unfixed = auto_vram_layers(free, non_expert, layer, 49, historical);
        let floor_charged = auto_vram_layers(
            free,
            non_expert,
            layer,
            49,
            charges.reserve(historical).unwrap(),
        );
        let placed =
            auto_placement_layers(free, non_expert, layer, 49, reserve.bytes(), &charges).unwrap();
        (unfixed, floor_charged, placed, reserve, charges)
    }

    #[test]
    fn real_census_terms_come_from_their_owning_queries() {
        let config = crate::config::compact_test_config();
        let census = real_census(&config, 16384);
        // 97 + 144 + 48 + 2 shadowed BF16 owners.
        assert_eq!(census.f16_shadows, 1_298_923_520);
        // The 4096-row pool workspace with every short tail (slab union plan
        // sized for 32 lanes: 7688 B).
        assert_eq!(
            census.workspace.iter().map(|term| term.1).sum::<u64>(),
            153_261_832
        );
        // The split selector's constant score + filter extent.
        assert_eq!(census.qsa_selector, 83_894_272);
        assert!(census.gather > 0 && census.forward > 0 && census.state > 0);
        assert_eq!(census.mtp, 0);
    }

    #[test]
    fn evidence_placement_charges_the_full_context_and_keeps_fewer_layers() {
        let (free, non_expert, layer, historical) = EVIDENCE;
        let config = crate::config::compact_test_config();
        // The p6 run's 16384-token context, VMM fp8 state: 12 trunk QSA
        // layers x 26 MiB, 312 MiB with the 2 MiB granularity.
        let census = real_census(&config, 16384);
        assert_eq!(census.qsa_context, 312 << 20);
        let (unfixed, floor_charged, placed, reserve, charges) =
            replay(free, non_expert, layer, historical, census);
        // The beta that ran out of memory kept the historical reserve alone:
        // (32550 - 5115 - 7927) / 1060 = 18.40 -> 18 resident layers.
        assert_eq!(unfixed, 18);
        // Stage + margin on the floor already cost layers; the census never
        // raises the count.
        assert!(floor_charged < 18);
        assert!(placed <= floor_charged && placed < 18);
        // The whole full-context reserve plus margin and both stage buffers
        // is left free, and one more layer would not leave it.
        assert_eq!(charges.margin, 512 << 20);
        assert_eq!(charges.stage, 2 * ((1060 << 20) + 4096));
        let charged = charges.reserve(reserve.bytes()).unwrap();
        assert!(free - non_expert - placed as u64 * layer >= charged);
        assert!(free - non_expert - (placed as u64 + 1) * layer < charged);
        // Every census term is itemised in the log text.
        let text = reserve.describe();
        for name in [
            "QSA context",
            "state",
            "forward",
            "F16 weight shadows",
            "QSA selector",
            "native MTP",
            "QSA gather",
            "fp16_x",
            "ksplit_det",
            "gdn_pair_counters",
        ] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(charges.describe().contains("512 MiB safety margin"));
        assert!(charges
            .describe()
            .contains("2120 MiB long-prefill stage buffers"));
    }

    #[test]
    fn mq2_16k_failure_card_charges_the_full_qsa_context_and_places_fewer_layers() {
        // r9700.speed.json first_failure: free 32550 MiB, non-expert 5115,
        // reserve 7929, 840 MiB per layer, max_seq 17408 on a VMM fp8 / Q8 GDN
        // state: (32550 - 5115 - 7929) / 840 = 23.2 -> layers 0..23 resident,
        // then the 16343-token prefill ran out of memory mapping the QSA
        // context.
        let (free, non_expert, layer, historical) =
            (32550u64 << 20, 5115u64 << 20, 840u64 << 20, 7929u64 << 20);
        let config = crate::config::compact_test_config();
        let census = real_census(&config, 17_408);
        // 12 trunk QSA layers x 28 MiB with the 2 MiB granularity.
        assert_eq!(census.qsa_context, 336 << 20);
        // The actual full map exceeds what the first prefill chunk maps: the
        // growth the historical floor charges as zero.
        let first =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, 17_408, 1536, 2 << 20);
        let first = state_census(&config, target_state_format(), &first, 4096, Full).unwrap();
        assert!(census.qsa_context > first.qsa_context);
        let (unfixed, floor_charged, placed, reserve, _) =
            replay(free, non_expert, layer, historical, census);
        assert_eq!(unfixed, 23);
        assert!(floor_charged < 23);
        assert!(placed <= floor_charged && placed < 23);
        assert!(reserve.describe().contains("QSA context"));
    }

    #[test]
    fn runtime_overhead_is_charged_only_on_exact_discrete_single_device_gfx1201() {
        assert_eq!(QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES, 1408 << 20);
        assert_eq!(
            runtime_overhead_bytes("gfx1201", false, 1),
            Some(QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES)
        );
        assert_eq!(runtime_overhead_bytes("gfx1201", true, 1), None);
        assert_eq!(runtime_overhead_bytes("gfx1201", false, 2), None);
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942", "gfx12010", ""] {
            assert_eq!(runtime_overhead_bytes(arch, false, 1), None, "{arch}");
        }
    }

    #[test]
    fn a_retired_floor_makes_the_census_the_reserve_and_says_so() {
        let census = ReserveCensus {
            runtime: 1344 << 20,
            ..census_with(1 << 30, 1 << 30, 1 << 30, 0, 0)
        };
        let retired = AutoReserve::new(None, census).unwrap();
        assert_eq!(retired.bytes(), retired.census_total);
        assert_eq!(retired.winner(), "census (historical floor retired)");
        let text = retired.describe();
        assert!(text.contains("HIP runtime 1344 MiB"), "{text}");
        assert!(
            text.contains(
                "(historical floor retired: the census carries the measured HIP runtime overhead)"
            ),
            "{text}"
        );
        // Even a census below what a floor would have been is not raised.
        let huge_floor = AutoReserve::new(Some(retired.census_total * 2), census).unwrap();
        assert_eq!(huge_floor.bytes(), retired.census_total * 2);
        assert_eq!(huge_floor.winner(), "historical floor");
    }

    /// The shipped mq3-xts R9700 card (free 32550 MiB, non-expert 5115 MiB,
    /// 1060 MiB per slab layer, 49 MoE layers, VMM fp8 / Q8 state, 4096-row
    /// chunk, slab stage + margin): the layers `auto` keeps resident today
    /// (historical floor, `Full` forward, no runtime term) and with the
    /// census carrying the HIP runtime overhead on the compact forward.
    fn shipped_r9700_layers(max_seq: usize) -> (usize, usize) {
        let (free, non_expert, layer) = (32550u64 << 20, 5115u64 << 20, 1060u64 << 20);
        let config = crate::config::compact_test_config();
        let format = target_state_format();
        let chunk = 4096;
        let charges = PlacementCharges::new(layer, true).unwrap();
        // The loader's historical floor: legacy-forced context, `Full`.
        let context = crate::Qwen4ContextCommit::for_expert_reserve(
            crate::Qwen4KvBackend::Vmm,
            false,
            max_seq,
            chunk,
            2 << 20,
        );
        let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
            config.num_key_value_heads,
            context.tokens,
        )
        .unwrap() as u64;
        let historical = auto_vram_reserve(
            &config,
            &context,
            chunk,
            Full,
            format.layout(),
            None,
            Some(gather),
        )
        .unwrap();
        let today = AutoReserve::new(
            Some(historical),
            census_at(&config, max_seq, chunk, Full, 0),
        )
        .unwrap();
        let new = AutoReserve::new(
            None,
            census_at(
                &config,
                max_seq,
                chunk,
                FnSymGfx1201,
                QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
            ),
        )
        .unwrap();
        let layers = |reserve: &AutoReserve| {
            auto_placement_layers(free, non_expert, layer, 49, reserve.bytes(), &charges).unwrap()
        };
        (layers(&today), layers(&new))
    }

    #[test]
    fn shipped_r9700_keeps_more_layers_with_the_compact_census_and_retired_floor() {
        let (today_8k, new_8k) = shipped_r9700_layers(8192);
        let (today_32k, new_32k) = shipped_r9700_layers(32768);
        assert_eq!((today_8k, today_32k), (15, 15));
        assert_eq!((new_8k, new_32k), (18, 17));
    }

    #[test]
    fn pruned_w3_slab_pack_fits_fully_resident_at_32k_only_with_a_small_compact_chunk() {
        // 29,556,669,976 device bytes (non-expert + routed experts), free
        // 32550 MiB: margin only, no stage (fully resident has no host layer).
        let (free, non_expert, total) = (32550u64 << 20, 5115u64 << 20, 29_556_669_976u64);
        let config = crate::config::compact_test_config();
        let charges = PlacementCharges::new(1060 << 20, false).unwrap();
        let fits = |chunk: usize| {
            let census = census_at(
                &config,
                32768,
                chunk,
                FnSymGfx1201,
                QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
            );
            let reserve = AutoReserve::new(None, census).unwrap();
            fits_fully_resident(
                free,
                non_expert,
                total - non_expert,
                charges.reserve(reserve.bytes()).unwrap(),
            )
        };
        assert!(fits(256));
        assert!(!fits(1024));
    }

    #[test]
    fn the_compact_1024_census_covers_the_measured_peak_beyond_weights() {
        let config = crate::config::compact_test_config();
        let census = census_at(
            &config,
            32768,
            1024,
            FnSymGfx1201,
            QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
        );
        // 4,141 MiB: the measured peak beyond the weights at this config.
        assert!(census.total().unwrap() >= 4141 << 20);
        assert_eq!(census.total().unwrap() >> 20, 4237);
        // For the record: the `Full` 8192-row chunk at 32K with the runtime.
        let full = census_at(
            &config,
            32768,
            8192,
            Full,
            QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
        );
        assert_eq!(full.total().unwrap() >> 20, 12302);
    }

    /// The shipped Strix Halo (gfx1151, unified memory) admission: VMM QSA
    /// F32 K/V, Q8 GDN, BF16 pooled mirror on, F16 arena off, no MTP,
    /// 8192-row chunk, `max_seq` 16384, free 118,441 MiB (XTS) / 118,415 MiB
    /// (mq3), routed 62,475 / 51,375 MiB, non-expert 5,115 MiB, logged
    /// reserve 12,150 MiB. Unified memory fits fully resident when
    /// `non_expert + routed + reserve <= free`; the reserve there is
    /// `AutoReserve::bytes()`, so this pins both formulas and the decision.
    #[test]
    fn halo_unified_memory_admission_reserve_and_census_decision() {
        let config = crate::config::compact_test_config();
        let format = crate::Qwen4StateFormat {
            qsa: QsaKvFormat::F32,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: true,
        };
        let (max_seq, chunk, granularity) = (16384usize, 8192usize, 2usize << 20);
        // Unified memory: the historical formula charges the first chunk's
        // pages (`for_expert_reserve(use_ranges = true)`), the census the
        // whole admitted context.
        let first = crate::Qwen4ContextCommit::for_expert_reserve(
            crate::Qwen4KvBackend::Vmm,
            true,
            max_seq,
            chunk,
            granularity,
        );
        let full = crate::Qwen4ContextCommit::new(
            crate::Qwen4KvBackend::Vmm,
            max_seq,
            max_seq,
            granularity,
        );
        assert!(
            full.committed_layer_bytes(&config, format.layout())
                > first.committed_layer_bytes(&config, format.layout())
        );
        let gather = |tokens: usize| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                tokens,
            )
            .unwrap() as u64
        };
        let historical = auto_vram_reserve(
            &config,
            &first,
            chunk,
            Full,
            format.layout(),
            None,
            Some(gather(first.tokens)),
        )
        .unwrap();
        // The logged reserve of the shipped runs.
        assert_eq!(historical >> 20, 12150);
        // Census: gfx1151 has no dense GDN scan, F16 shadow or split selector.
        let state = state_census(&config, format, &full, chunk, Full).unwrap();
        let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
            hidden_size: config.hidden_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hc_count: config.hc_count,
            hc_lowrank: config.hc_lowrank,
            dense_gdn: false,
            routed_i4: true,
            include_short_tails: true,
        };
        let census = ReserveCensus {
            qsa_context: state.qsa_context,
            state: state.state,
            forward: state.forward,
            f16_shadows: 0,
            workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims).unwrap(),
            qsa_selector: 0,
            mtp: 0,
            gather: gather(full.tokens),
            runtime: 0,
        };
        let reserve = AutoReserve::new(Some(historical), census).unwrap();
        // Both Halo artifacts fit fully resident under whichever formula wins
        // (exact non-expert bytes 5,364,055,296 = 5,115.6 MiB).
        assert_eq!(reserve.census_total >> 20, 9854, "{}", reserve.describe());
        for (free, routed) in [
            (118_441u64 << 20, 62_475u64 << 20),
            (118_415 << 20, 51_375 << 20),
        ] {
            assert!(
                fits_fully_resident(free, 5_364_055_296, routed, reserve.bytes()),
                "{}",
                reserve.describe()
            );
        }
        // The census decision: the historical floor wins on this admission
        // (no flip), so unified-memory behaviour is unchanged. A failure here
        // is the flip: unified memory must then admit on `historical`.
        assert_eq!(
            reserve.winner(),
            "historical floor",
            "{}",
            reserve.describe()
        );
    }

    #[test]
    fn the_census_winner_sets_the_reserve_and_every_term_is_itemised() {
        let (free, non_expert, layer, historical) = EVIDENCE;
        let census = census_with(
            2000 << 20,
            1000 << 20,
            5000 << 20,
            1_298_923_520,
            83_894_272,
        );
        let total = (2000 + 1000 + 5000) * (1u64 << 20) + 1_298_923_520 + 153_256_072 + 83_894_272;
        let reserve = AutoReserve::new(Some(historical), census).unwrap();
        assert_eq!(reserve.census_total, total);
        assert!(total > historical);
        assert_eq!(reserve.bytes(), total);
        assert_eq!(reserve.winner(), "census");
        let text = reserve.describe();
        for name in [
            "QSA context",
            "state",
            "forward",
            "F16 weight shadows",
            "workspace",
            "QSA selector",
            "native MTP",
            "QSA gather",
        ] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(
            text.contains("census wins") && text.contains("7927 MiB historical floor"),
            "{text}"
        );
        // The census, not the floor, sizes the placement.
        let by_floor = auto_vram_layers(free, non_expert, layer, 49, historical);
        let by_census = auto_vram_layers(free, non_expert, layer, 49, reserve.bytes());
        assert!(by_census < by_floor);
        let overflowing = ReserveCensus {
            gather: 1,
            qsa_context: u64::MAX,
            ..census_with(0, 0, 0, 0, 0)
        };
        assert!(AutoReserve::new(Some(0), overflowing).is_err());
    }

    #[test]
    fn state_census_charges_the_actual_backends_full_context_once() {
        let config = crate::config::compact_test_config();
        let max_seq = 17_408;
        let granularity = 2 << 20;
        // The reference F32 state and the target's actual fp8 K/V + Q8 GDN.
        for format in [crate::Qwen4StateFormat::F32, target_state_format()] {
            for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
                // Whole admitted context mapped, as the actual backend maps it.
                let full = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
                let census = state_census(&config, format, &full, 4096, Full).unwrap();
                let context = full
                    .committed_layer_bytes(&config, format.layout())
                    .unwrap()
                    * config.n_full_layers();
                assert_eq!(census.qsa_context, context as u64, "{backend:?} {format:?}");
                // The rest of the state plus the context is exactly the state.
                let state = crate::state::Qwen4State::device_bytes(&config, format, &full).unwrap();
                assert_eq!(
                    census.qsa_context + census.state,
                    state as u64,
                    "{backend:?} {format:?}"
                );
                assert_eq!(
                    census.forward,
                    crate::gpu_forward::qwen4_forward_device_bytes(&config, 4096, Full).unwrap()
                );
                // Not the first chunk's pages: growth past them is charged.
                let first = crate::Qwen4ContextCommit::new(backend, max_seq, 1536, granularity);
                let first = state_census(&config, format, &first, 4096, Full).unwrap();
                if backend == crate::Qwen4KvBackend::Vmm {
                    assert!(census.qsa_context > first.qsa_context, "{format:?}");
                }
            }
        }
    }

    fn bf16(name: &str, shape: &[usize]) -> WeightEntry {
        WeightEntry::model(name, shape.to_vec(), DType::BF16, ShardPolicy::Replicate)
    }

    #[test]
    fn f16_shadow_bytes_query_the_manifest_for_the_shadowed_owners_only() {
        let layer = |i: usize, suffix: &str| format!("model.language_model.layers.{i}.{suffix}");
        let mut weights = vec![
            // Shadowed: full F16 matrices, 2 bytes per element.
            bf16(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                &[4, 8],
            ),
            bf16(
                &layer(0, "attn_hyper_connection.input_mix_weight_down.weight"),
                &[2, 2],
            ),
            bf16(
                &layer(1, "mlp_hyper_connection.input_mix_weight_down.weight"),
                &[2, 2],
            ),
            bf16(&layer(0, "mlp.gate.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.gate_proj.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.up_proj.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.down_proj.weight"), &[5, 3]),
            bf16(&layer(0, "ple.key_proj.weight"), &[1, 7]),
            bf16(&layer(0, "ple.value_proj.weight"), &[1, 7]),
            // Direct BF16 readers and other tensors: no shadow.
            bf16(
                &layer(0, "attn_hyper_connection.input_mix_weight_up.weight"),
                &[2, 2],
            ),
            bf16(&layer(0, "mlp.shared_expert_gate.weight"), &[1, 9]),
            bf16(&layer(0, "input_layernorm.weight"), &[1, 9]),
            // Past the trunk, wrong rank or wrong dtype: no shadow.
            bf16(&layer(2, "mlp.gate.weight"), &[3, 5]),
            bf16(&layer(0, "ple.key_proj.weight.extra"), &[1, 7]),
            WeightEntry::model(
                layer(1, "mlp.gate.weight"),
                vec![3, 5],
                DType::MQ4G256V2,
                ShardPolicy::Replicate,
            ),
            bf16(&layer(1, "mlp.gate.weight"), &[15]),
        ];
        let expected = 2 * (4 * 8 + 2 * 2 + 2 * 2 + 3 * 5 + 3 * 5 + 3 * 5 + 5 * 3 + 7 + 7);
        assert_eq!(f16_shadow_bytes(&weights, 2), Ok(expected as u64));
        // An alias or external row store has no payload of its own.
        weights.push(WeightEntry::model(
            layer(1, "mlp.gate.weight"),
            vec![3, 5],
            DType::BF16,
            ShardPolicy::Tied { source: "x".into() },
        ));
        assert_eq!(f16_shadow_bytes(&weights, 2), Ok(expected as u64));
        // An MQ-quantized trunk (no BF16 owners) charges nothing.
        assert_eq!(f16_shadow_bytes(&[], 48), Ok(0));
        // The 48-layer total the target profile pins: 97 HC down, 144 shared,
        // 48 routers, 2 PLE.
        let target: u64 = 1_298_923_520;
        assert_eq!(target, 1238 * (1 << 20) + (3 << 18));
    }

    #[test]
    fn stage_switch_is_env_first_then_gfx1201_or_narrow_link() {
        use crate::host_link::HostLink::{self, Discrete, Unified, Unmeasured};
        fn no_probe() -> HostLink {
            unreachable!("the link was probed")
        }
        let narrow = || Discrete { gbps: 12.0 };
        let wide = || Discrete { gbps: 27.3 };
        // Unset: gfx1201 on regardless of link (and never probes); every
        // other arch only on a measured narrow discrete link.
        assert!(expert_stage_switch("gfx1201", None, no_probe));
        assert!(expert_stage_switch("gfx1201", None, || Unmeasured));
        assert!(expert_stage_switch("gfx1201", None, wide));
        assert!(expert_stage_switch("gfx1100", None, narrow));
        assert!(!expert_stage_switch("gfx1100", None, wide));
        assert!(!expert_stage_switch("gfx1100", None, || Unmeasured));
        assert!(!expert_stage_switch("gfx1151", None, || Unified));
        // Anything but the strict `1`/`0` is unset.
        assert!(expert_stage_switch("gfx1100", Some("on"), narrow));
        assert!(!expert_stage_switch("gfx1100", Some(""), wide));
        // Explicit values win over arch and link, and never probe.
        assert!(expert_stage_switch("gfx1100", Some("1"), no_probe));
        assert!(expert_stage_switch("gfx1151", Some("1"), || Unified));
        assert!(!expert_stage_switch("gfx1100", Some("0"), no_probe));
        assert!(!expert_stage_switch("gfx1201", Some("0"), wide));
        // Arithmetic of the probing default only: the load charges placement
        // with the probe-free `expert_stage_placement_switch` instead (see
        // `placement_first_*`), so it never sees this narrow-link result.
        let charged = |arch, link: fn() -> HostLink| {
            dedicated_slab_stage(expert_stage_switch(arch, None, link), true, false)
        };
        assert!(charged("gfx1100", narrow));
        assert!(!charged("gfx1100", wide));
        assert!(!charged("gfx1100", || Unmeasured));
        assert!(charged("gfx1201", wide));
    }

    #[test]
    fn stage_charge_only_where_the_slab_stage_allocates() {
        // Needs the switch, a slab recipe and no expert parallelism.
        assert!(dedicated_slab_stage(true, true, false));
        assert!(!dedicated_slab_stage(false, true, false));
        assert!(!dedicated_slab_stage(true, false, false));
        assert!(!dedicated_slab_stage(true, true, true));
        let (free, non_expert, layer, base) = EVIDENCE;
        let off = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(
            off,
            PlacementCharges {
                margin: 512 << 20,
                stage: 0,
                slowlink: 0,
            }
        );
        assert_eq!(off.reserve(base).unwrap(), base + (512 << 20));
        // No stage: only the margin costs layers
        // ((32550 - 5115 - 7927 - 512) / 1060 = 17.9).
        let unstaged = auto_placement_layers(free, non_expert, layer, 49, base, &off).unwrap();
        assert_eq!(unstaged, 17);
        let staged = PlacementCharges::new(layer, true).unwrap();
        let staged = auto_placement_layers(free, non_expert, layer, 49, base, &staged).unwrap();
        assert!(staged < unstaged);
        // Fully resident: the unset fallback charges the same reserve and
        // margin (never a stage), so a card that only just holds everything
        // without the margin falls to `auto` instead of loading.
        let experts = 20 * layer;
        let need = non_expert + experts + base;
        assert!(fits_fully_resident(need, non_expert, experts, base));
        assert!(!fits_fully_resident(
            need,
            non_expert,
            experts,
            off.reserve(base).unwrap()
        ));
    }

    #[test]
    fn charged_placement_refuses_clearly_when_zero_layers_do_not_fit() {
        let (_, non_expert, layer, base) = EVIDENCE;
        let charges = PlacementCharges::new(layer, true).unwrap();
        let need = non_expert + charges.reserve(base).unwrap();
        // Exactly the weights and the charged reserve: zero resident layers.
        assert_eq!(
            auto_placement_layers(need, non_expert, layer, 49, base, &charges),
            Ok(0)
        );
        let error =
            auto_placement_layers(need - 1, non_expert, layer, 49, base, &charges).unwrap_err();
        assert!(
            error.contains("cannot hold") && error.contains("every routed expert in host RAM"),
            "{error}"
        );
        assert!(error.contains("512 MiB safety margin"), "{error}");
        assert!(
            error.contains("2120 MiB long-prefill stage buffers"),
            "{error}"
        );
        // Without the stage the same card still loads: the 2120 MiB + 8191 B
        // the stage took now holds two whole layers.
        let off = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(
            auto_placement_layers(need - 1, non_expert, layer, 49, base, &off),
            Ok(2)
        );
    }

    #[test]
    fn placement_charges_refuse_overflow() {
        assert!(PlacementCharges::new(u64::MAX, true).is_err());
        assert!(PlacementCharges::new(u64::MAX / 2, true).is_err());
        assert!(PlacementCharges::new(u64::MAX, false).is_ok());
        let charges = PlacementCharges::new(1 << 20, true).unwrap();
        assert!(charges.reserve(u64::MAX).is_err());
        assert!(auto_placement_layers(u64::MAX, 0, 1 << 20, 49, u64::MAX, &charges).is_err());
        // A non-expert size that overflows the sum is a refusal, not a wrap.
        assert!(auto_placement_layers(u64::MAX, u64::MAX, 1 << 20, 49, 1, &charges).is_err());
    }

    const WF_LAYER: u64 = 1275 << 20;
    const WF_NON_EXPERT: u64 = 6000 << 20;
    const WF_BASE: u64 = 8000 << 20;

    /// Free VRAM that places exactly 14 whole 1275 MiB layers after the
    /// non-expert weights, the base reserve and the 512 MiB margin (the
    /// receipt's 14-layer geometry; no stage charge).
    fn wf_free_for_14() -> u64 {
        WF_NON_EXPERT + WF_BASE + (512 << 20) + 14 * WF_LAYER + (100 << 20)
    }

    fn wf_auto(
        free: u64,
        charges: &PlacementCharges,
        borrowed: bool,
    ) -> Result<WholeFirstPlan, String> {
        WholeFirstPlan::auto(
            free,
            WF_NON_EXPERT,
            WF_LAYER,
            48,
            49,
            WF_BASE,
            charges,
            borrowed,
        )
    }

    #[test]
    fn whole_first_never_places_fewer_layers_than_the_hotset0_baseline_and_fits() {
        for dedicated in [false, true] {
            let charges = PlacementCharges::new(WF_LAYER, dedicated).unwrap();
            for borrowed in [false, true] {
                for step in 0..=96u64 {
                    let free = (14_000u64 + step * 250) << 20;
                    let baseline =
                        auto_placement_layers(free, WF_NON_EXPERT, WF_LAYER, 48, WF_BASE, &charges);
                    let plan = wf_auto(free, &charges, borrowed);
                    match (baseline, plan) {
                        (Ok(layers), Ok(plan)) => {
                            assert!(plan.whole_layers >= layers, "free {free}");
                            assert_eq!(plan.hot_experts, 0);
                            assert!(plan.whole_layers <= 48);
                            // Resident layers + charged reserve (dedicated
                            // stage included) + non-expert weights fit.
                            let need = WF_NON_EXPERT
                                + charges.reserve(WF_BASE).unwrap()
                                + plan.whole_layers as u64 * WF_LAYER;
                            assert!(need <= free, "free {free}: need {need}");
                            // One more layer would not have fit.
                            if plan.whole_layers < 48 {
                                assert!(need + WF_LAYER > free);
                            }
                        }
                        (Err(_), Err(_)) => {}
                        (baseline, plan) => panic!("diverged: {baseline:?} vs {plan:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn whole_first_receipt_geometry_is_14_layers_with_2550_mib_borrowed_and_nothing_allocated() {
        let free = wf_free_for_14();
        let off = PlacementCharges::new(WF_LAYER, false).unwrap();
        let plan = wf_auto(free, &off, true).unwrap();
        assert_eq!(plan.whole_layers, 14);
        assert_eq!(
            auto_placement_layers(free, WF_NON_EXPERT, WF_LAYER, 48, WF_BASE, &off),
            Ok(14)
        );
        assert_eq!(
            plan.stage,
            WholeFirstStage::Borrowed {
                donors: [12, 13],
                capacity: 2550 << 20,
                restore_copies: 2550 << 20,
            }
        );
        assert_eq!(plan.stage.allocated_vram(), 0);
        assert_eq!(plan.stage.borrowed_vram(), 2550 << 20);
        let line = plan.describe(14 * WF_LAYER);
        for part in [
            "policy=whole-first",
            "whole layers 0..14 (14 of 49 MoE layers)",
            "hot experts 0",
            "expert resident 17850 MiB",
            "stage allocated 0 MiB, borrowed 2550 MiB from resident layers 12 and 13",
            "2550 MiB pinned host restore copies",
            "sidecar not consulted",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
    }

    #[test]
    fn whole_first_stage_accounting_dedicated_borrowed_and_disabled() {
        let free = wf_free_for_14() + (4000 << 20);
        // Stage switch off or not a donor layout: nothing allocated or borrowed.
        let off = PlacementCharges::new(WF_LAYER, false).unwrap();
        let disabled = wf_auto(free, &off, false).unwrap();
        assert_eq!(disabled.stage, WholeFirstStage::None);
        assert!(disabled
            .describe(0)
            .contains("stage allocated 0 MiB, borrowed 0 MiB"));
        // Borrowed donors cost no VRAM: same layers as the disabled load.
        let borrowed = wf_auto(free, &off, true).unwrap();
        assert_eq!(borrowed.whole_layers, disabled.whole_layers);
        assert_eq!(borrowed.stage.allocated_vram(), 0);
        assert_eq!(borrowed.stage.borrowed_vram(), 2 * WF_LAYER);
        // A slab load's dedicated stage is allocated and charged: fewer
        // layers than the stage-free load, and the buffers are accounted.
        let on = PlacementCharges::new(WF_LAYER, true).unwrap();
        let dedicated = wf_auto(free, &on, false).unwrap();
        assert!(dedicated.whole_layers < disabled.whole_layers);
        let buffer = WF_LAYER + crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64;
        assert_eq!(
            dedicated.stage,
            WholeFirstStage::Dedicated {
                buffer,
                allocated: 2 * buffer
            }
        );
        assert_eq!(dedicated.stage.allocated_vram(), on.stage);
        assert_eq!(dedicated.stage.borrowed_vram(), 0);
        let line = dedicated.describe(0);
        assert!(
            line.contains("borrowed 0 MiB") && line.contains("2 dedicated buffers of"),
            "{line}"
        );
        // The dedicated stage wins over borrowing (slab recipes never borrow).
        assert_eq!(wf_auto(free, &on, true).unwrap().stage, dedicated.stage);
        // Fewer than two resident layers cannot lend donors.
        assert_eq!(
            WholeFirstPlan::stage_for(1, 49, WF_LAYER, &off, true),
            WholeFirstStage::None
        );
        assert!(matches!(
            WholeFirstPlan::stage_for(2, 49, WF_LAYER, &off, true),
            WholeFirstStage::Borrowed { donors: [0, 1], .. }
        ));
        // No host layer, no stage, even with a charge.
        assert_eq!(
            WholeFirstPlan::stage_for(49, 49, WF_LAYER, &on, true),
            WholeFirstStage::None
        );
        let resident = WholeFirstPlan::fully_resident(49, WF_LAYER);
        assert_eq!((resident.whole_layers, resident.hot_experts), (49, 0));
        assert_eq!(resident.stage, WholeFirstStage::None);
        assert!(resident
            .describe(0)
            .contains("whole layers 0..49 (49 of 49"));
    }

    #[test]
    fn whole_first_refuses_exactly_like_the_baseline_when_nothing_fits() {
        let on = PlacementCharges::new(WF_LAYER, true).unwrap();
        let need = WF_NON_EXPERT + on.reserve(WF_BASE).unwrap();
        assert_eq!(wf_auto(need, &on, false).unwrap().whole_layers, 0);
        let plan_error = wf_auto(need - 1, &on, false).unwrap_err();
        let base_error =
            auto_placement_layers(need - 1, WF_NON_EXPERT, WF_LAYER, 48, WF_BASE, &on).unwrap_err();
        assert_eq!(plan_error, base_error);
    }

    #[test]
    fn resident_routed_expert_bytes_counts_only_what_stays_on_device() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.0.mlp.experts.down_proj", 0),
            entry("model.language_model.layers.1.mlp.experts.gate_up_proj", 1),
            entry("model.language_model.layers.0.self_attn.q_proj.weight", 0),
            entry("mtp.layers.0.mlp.experts.gate_up_proj", 0),
        ];
        let bytes_of =
            |entry: &WeightEntry| Some(entry.logical_shape.iter().product::<usize>() as u64);
        let each = 4 * 4 * 256;
        assert_eq!(
            resident_routed_expert_bytes(&weights, bytes_of),
            Ok(4 * each)
        );
        place_routed_experts(&mut weights, 1);
        // Trunk layer 0's two projections stay; layer 1 and the MTP layer move.
        assert_eq!(
            resident_routed_expert_bytes(&weights, bytes_of),
            Ok(2 * each)
        );
        assert!(resident_routed_expert_bytes(&weights, |_| None).is_err());
    }

    #[test]
    fn compact_scratch_lowers_the_reserve_by_exactly_its_saving() {
        let config = crate::config::compact_test_config();
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let reserve = |rows, mode| {
            auto_vram_reserve(&config, &measured, rows, mode, F32, None, None).unwrap()
        };
        let forward = |rows, mode| {
            crate::gpu_forward::qwen4_forward_device_bytes(&config, rows, mode).unwrap()
        };
        // The 8192-row chunk's compact plan is 4240.125 MiB smaller, and the
        // whole of it reaches the reserve: the historical baseline stays on
        // the Full layout, so the saving is not subtracted from both sides.
        let saving = forward(8192, Full) - forward(8192, FnSymGfx1201);
        assert_eq!(saving, 4240 * (1 << 20) + (1 << 20) / 8);
        assert_eq!(reserve(8192, Full) - reserve(8192, FnSymGfx1201), saving);
        // The same holds at every chunk, including the measured layout's own
        // 1536 rows.
        for rows in [AUTO_VRAM_RESERVE_CHUNK, 2048, 4096] {
            let delta = forward(rows, Full) - forward(rows, FnSymGfx1201);
            assert!(delta > 0, "{rows} rows");
            assert_eq!(
                reserve(rows, Full) - reserve(rows, FnSymGfx1201),
                delta,
                "{rows} rows"
            );
        }
        // A compact 8192-row chunk costs no more than today's 4096-row chunk,
        // so `auto` keeps at least the layers it keeps today.
        assert!(reserve(8192, FnSymGfx1201) <= reserve(4096, Full));
        let (free, weights, layer) = (32548u64 << 20, 5_364_000_000u64, 1_336_900_000u64);
        let layers = |reserve| auto_vram_layers(free, weights, layer, 48, reserve);
        assert!(layers(reserve(8192, FnSymGfx1201)) >= layers(reserve(4096, Full)));
        assert!(layers(reserve(8192, FnSymGfx1201)) > layers(reserve(8192, Full)));
    }

    #[test]
    fn native_mtp_and_longer_context_grow_the_reserve() {
        let config = crate::config::compact_test_config();
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let base = auto_vram_reserve(
            &config,
            &measured,
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            F32,
            None,
            None,
        )
        .unwrap();
        let native = |context: &crate::Qwen4ContextCommit, max_k, row_capture| {
            crate::mtp_spec::native_mtp_device_bytes(
                &config,
                context,
                AUTO_VRAM_RESERVE_CHUNK,
                max_k,
                DType::MQ6G256V2,
                row_capture,
                false,
            )
            .unwrap()
        };
        let mtp = native(&measured, 3, Some((crate::GdnStateFormat::Q8, 4)));
        let with_mtp = auto_vram_reserve(
            &config,
            &measured,
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            F32,
            Some(mtp),
            None,
        )
        .unwrap();
        assert_eq!(with_mtp - base, mtp);
        // The gathered QSA attention's workspace is charged on top.
        let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
            config.num_key_value_heads,
            AUTO_VRAM_RESERVE_MAX_SEQ,
        )
        .unwrap() as u64;
        let with_gather = auto_vram_reserve(
            &config,
            &measured,
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            F32,
            None,
            Some(gather),
        )
        .unwrap();
        assert_eq!(with_gather - base, gather);
        // The draft head's F32 requant scratch is row-chunked (at most
        // `REQUANT_G256_CHUNK_ROWS` rows, never vocab x hidden x 4) and goes
        // back to the device at attach, before the first request allocates
        // its verify rows and GDN capture: the reserve holds the larger phase
        // on top of what the head keeps, not both.
        let (resident, scratch) =
            crate::mtp_gpu::Qwen4MtpGpu::device_bytes(&config, &measured, DType::MQ6G256V2)
                .unwrap();
        let (resident, scratch) = (resident as u64, scratch as u64);
        assert_eq!(
            scratch,
            (rdna_compute::Gpu::REQUANT_G256_CHUNK_ROWS * config.hidden_size * 4) as u64
        );
        // The request now outweighs the build scratch at every K.
        let hidden_row = (config.hc_count * config.hidden_size * 4) as u64;
        // Verify rows plus the pending, row and retained-pending hidden carries.
        let verify = (AUTO_VRAM_RESERVE_CHUNK as u64 + 3) * hidden_row;
        assert!(verify > scratch);
        let capture4 =
            crate::state::Qwen4State::row_capture_bytes(&config, crate::GdnStateFormat::Q8, 4)
                .unwrap() as u64;
        assert_eq!(mtp, resident + verify + capture4);
        assert_eq!(native(&measured, 3, None), resident + verify);
        assert_eq!(native(&measured, 10, None), resident + verify);
        // A row-exact verify limit wider than the MTP window (gfx1151: 32)
        // charges its whole ring; a narrower one never shrinks the window's.
        let capture32 =
            crate::state::Qwen4State::row_capture_bytes(&config, crate::GdnStateFormat::Q8, 32)
                .unwrap() as u64;
        assert_eq!(
            native(&measured, 3, Some((crate::GdnStateFormat::Q8, 32))),
            resident + verify + capture32
        );
        assert_eq!(
            native(&measured, 3, Some((crate::GdnStateFormat::Q8, 2))),
            mtp
        );
        // The 11-row GDN capture is charged in the state's real GDN format.
        for gdn in [crate::GdnStateFormat::Q8, crate::GdnStateFormat::F32] {
            let capture =
                crate::state::Qwen4State::row_capture_bytes(&config, gdn, 11).unwrap() as u64;
            let request = (AUTO_VRAM_RESERVE_CHUNK as u64 + 3) * hidden_row + capture;
            assert_eq!(
                native(&measured, 10, Some((gdn, 11))),
                resident + scratch.max(request)
            );
            // The gfx1151 batched prompt fill adds its row scratch (sub-chunks
            // of at most MTP_FILL_ROWS rows) and one chunk of prompt token ids
            // to the request.
            let fill = crate::mtp_gpu::MtpAppendScratch::device_bytes(
                &config,
                crate::mtp_gpu::MTP_FILL_ROWS.min(AUTO_VRAM_RESERVE_CHUNK),
                AUTO_VRAM_RESERVE_CHUNK,
            )
            .unwrap() as u64;
            assert!(fill > 0);
            let batched = crate::mtp_spec::native_mtp_device_bytes(
                &config,
                &measured,
                AUTO_VRAM_RESERVE_CHUNK,
                10,
                DType::MQ6G256V2,
                Some((gdn, 11)),
                true,
            )
            .unwrap();
            assert_eq!(batched, resident + scratch.max(request + fill));
        }
        // On a card whose free VRAM sits exactly on a layer boundary of the
        // base reserve (non-expert weights + base + 16 layers; not a
        // measured card), any positive MTP charge must cost a layer; the
        // MTP bytes then fit beside the chosen layers, and one layer more
        // would not have left them.
        let (weights, layer) = (5_364_000_000u64, 1_336_900_000u64);
        let free = weights + base + 16 * layer;
        assert_eq!(auto_vram_layers(free, weights, layer, 48, base), 16);
        let with = auto_vram_layers(free, weights, layer, 48, with_mtp) as u64;
        assert!(with < auto_vram_layers(free, weights, layer, 48, base) as u64);
        assert!(free - weights - with * layer >= with_mtp);
        assert!(free - weights - (with + 1) * layer < with_mtp);
        // A legacy context past the measured one adds the trunk QSA arenas'
        // growth in the load's QSA format: fp8 K/V grows less than F32.
        let long_seq = 4 * AUTO_VRAM_RESERVE_MAX_SEQ;
        let growth = |format| {
            let long = auto_vram_reserve(
                &config,
                &crate::Qwen4ContextCommit::legacy(long_seq),
                AUTO_VRAM_RESERVE_CHUNK,
                Full,
                format,
                None,
                None,
            )
            .unwrap();
            let arena = (config.qsa_context_arena_bytes(long_seq, format).unwrap()
                - config
                    .qsa_context_arena_bytes(AUTO_VRAM_RESERVE_MAX_SEQ, format)
                    .unwrap())
                * config.n_full_layers();
            assert_eq!(long - base, arena as u64);
            long - base
        };
        assert!(growth(QsaKvFormat::Fp8) < growth(F32));
        // A shorter context keeps the measured reserve.
        let short = auto_vram_reserve(
            &config,
            &crate::Qwen4ContextCommit::legacy(2048),
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            QsaKvFormat::Fp8,
            None,
            None,
        );
        assert_eq!(short.unwrap(), base);
    }

    #[test]
    fn vmm_context_charges_only_committed_pages() {
        let config = crate::config::compact_test_config();
        let chunk = AUTO_VRAM_RESERVE_CHUNK;
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let base = auto_vram_reserve(&config, &measured, chunk, Full, F32, None, None).unwrap();
        let vmm = |max_seq| {
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, max_seq, chunk, 2 << 20)
        };
        // The native logical context, virtual under VMM, does not size the
        // reserve; legacy storage at that context commits every arena.
        let native = crate::QWEN4_MAX_CONTEXT;
        let vmm_native =
            auto_vram_reserve(&config, &vmm(native), chunk, Full, F32, None, None).unwrap();
        assert_eq!(vmm_native, base);
        let legacy_native = auto_vram_reserve(
            &config,
            &crate::Qwen4ContextCommit::legacy(native),
            chunk,
            Full,
            F32,
            None,
            None,
        )
        .unwrap();
        assert!(legacy_native > vmm_native);
        // Native MTP's one F32 QSA layer: the VMM head charges its first
        // chunk's pages instead of the full legacy arenas.
        let mtp = |context: &crate::Qwen4ContextCommit| {
            crate::mtp_gpu::Qwen4MtpGpu::device_bytes(&config, context, DType::MQ6G256V2)
                .unwrap()
                .0
        };
        let legacy_mtp = mtp(&crate::Qwen4ContextCommit::legacy(native));
        let vmm_mtp = mtp(&vmm(native));
        assert_eq!(
            legacy_mtp - vmm_mtp,
            config.qsa_context_arena_bytes(native, F32).unwrap()
                - vmm(native).committed_layer_bytes(&config, F32).unwrap()
        );
    }

    #[test]
    fn discrete_reserve_matches_legacy_and_unified_memory_charges_committed() {
        use crate::{Qwen4ContextCommit as Commit, Qwen4KvBackend as Backend};
        let config = crate::config::compact_test_config();
        let chunk = AUTO_VRAM_RESERVE_CHUNK;
        let granularity = 2 << 20;
        let mtp = |context: &Commit| {
            crate::mtp_gpu::Qwen4MtpGpu::device_bytes(&config, context, DType::MQ6G256V2)
                .unwrap()
                .0
        };
        // The expert layers `auto` places for a load whose reserve charges
        // `context`, with the native MTP and QSA gather parts as the loader
        // computes them from the same context, on the measured R9700 fixture.
        let placement = |context: &Commit, format| {
            let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                context.tokens,
            )
            .unwrap() as u64;
            let reserve = auto_vram_reserve(
                &config,
                context,
                chunk,
                Full,
                format,
                Some(mtp(context) as u64),
                Some(gather),
            )
            .unwrap();
            let free = 32548u64 << 20;
            let (weights, layer) = (5_364_000_000u64, 1_336_900_000u64);
            (reserve, auto_vram_layers(free, weights, layer, 48, reserve))
        };
        for max_seq in [65_536, 262_144] {
            let legacy = Commit::legacy(max_seq);
            for format in [QsaKvFormat::Fp8, F32] {
                // Discrete: a VMM backend's reserve and placement are legacy
                // storage's at this `max_seq`.
                let discrete =
                    Commit::for_expert_reserve(Backend::Vmm, false, max_seq, chunk, granularity);
                assert_eq!(discrete, legacy);
                assert_eq!(placement(&discrete, format), placement(&legacy, format));
                assert_eq!(
                    Commit::for_expert_reserve(Backend::Legacy, false, max_seq, chunk, granularity),
                    legacy
                );
                // Unified memory is unchanged: VMM charges the first chunk's
                // pages, legacy all of `max_seq`.
                let vmm = Commit::new(Backend::Vmm, max_seq, chunk, granularity);
                let uma_vmm =
                    Commit::for_expert_reserve(Backend::Vmm, true, max_seq, chunk, granularity);
                assert_eq!(uma_vmm, vmm);
                assert_eq!(placement(&uma_vmm, format), placement(&vmm, format));
                assert!(placement(&uma_vmm, format).0 < placement(&legacy, format).0);
                assert!(placement(&uma_vmm, format).1 >= placement(&legacy, format).1);
                assert_eq!(
                    Commit::for_expert_reserve(Backend::Legacy, true, max_seq, chunk, granularity),
                    legacy
                );
            }
            assert!(
                mtp(&Commit::for_expert_reserve(
                    Backend::Vmm,
                    true,
                    max_seq,
                    chunk,
                    granularity
                )) < mtp(&legacy)
            );
        }
    }

    #[test]
    fn host_ram_check_refuses_below_headroom() {
        let host = 60u64 << 30;
        assert!(check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES), 0).is_ok());
        let error = check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES - 1), 0).unwrap_err();
        assert!(error.contains("MemAvailable"), "{error}");
        assert!(check_host_ram(host, None, 0).is_err());
        assert!(check_host_ram(host, None, u64::MAX).is_err());
        assert!(check_host_ram(0, None, 0).is_ok());
        // The pool counts toward the room, byte for byte.
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES).is_ok());
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES - 1).is_err());
    }

    #[test]
    fn stage_mirrors_charge_every_rank_on_the_shared_host() {
        let full = 512 * (1_740_800 + 870_400);
        assert_eq!(
            expert_stage_mirror_bytes(full, 1).unwrap(),
            2 * full as u64 + 4 * HOST_MAPPED_PAD_BYTES
        );
        assert_eq!(
            expert_stage_mirror_bytes(full / 2, 2).unwrap(),
            2 * full as u64 + 8 * HOST_MAPPED_PAD_BYTES
        );
        assert!(expert_stage_mirror_bytes(full, 0).is_err());
        assert!(expert_stage_mirror_bytes(usize::MAX, 2).is_err());
    }

    #[test]
    fn ep4_stage_mirrors_charge_all_four_ranks_against_host_and_gtt() {
        let full = 512 * (1_740_800 + 870_400);
        let charge = expert_stage_mirror_bytes(full / 4, 4).unwrap();
        assert_eq!(charge, 2 * full as u64 + 16 * HOST_MAPPED_PAD_BYTES);
        // Same payload as EP2 (the ranks split it); every extra rank adds its own pads.
        let ep2 = expert_stage_mirror_bytes(full / 2, 2).unwrap();
        assert_eq!(charge - ep2, 8 * HOST_MAPPED_PAD_BYTES);
        // Host RAM admission takes the total over all four ranks, byte for byte.
        assert!(check_host_ram(charge, Some(charge + HOST_RAM_HEADROOM_BYTES), 0).is_ok());
        assert!(check_host_ram(charge, Some(charge + HOST_RAM_HEADROOM_BYTES - 1), 0).is_err());
        // So does the GTT cap: one rank's quarter fits where the four-rank total does not.
        let budget = |limit_bytes| {
            Some(GttBudget {
                limit_bytes,
                used_bytes: 0,
            })
        };
        assert!(check_gtt_cap(charge, budget(charge)).is_ok());
        assert!(check_gtt_cap(charge, budget(charge - 1)).is_err());
        assert!(check_gtt_cap(charge / 4, budget(charge - 1)).is_ok());
    }

    /// `/proc/meminfo` text from `(field, MiB)` pairs.
    fn meminfo(fields: &[(&str, u64)]) -> String {
        fields
            .iter()
            .map(|(key, mib)| format!("{key}:{:>16} kB\n", mib << 10))
            .collect()
    }

    /// The measured 5-card gfx1201 host (MemTotal 128865156 kB, pool cap
    /// 16108144 pages), 5 x 16 MiB of idle GTT, and `pool_mib` of pages outside
    /// every counter on top of `baseline_mib`.
    fn gfx1201_host(pool_mib: u64, baseline_mib: u64) -> String {
        let total = 128_865_156 >> 10;
        let (anon, slab, sreclaim) = (14 << 10, 3 << 10, 2 << 10);
        let cached = total - pool_mib - baseline_mib - 80 - anon - slab - 1024;
        meminfo(&[
            ("MemTotal", total),
            ("MemFree", 1024),
            ("MemAvailable", 45_800),
            ("Cached", cached),
            ("SwapCached", 0),
            ("AnonPages", anon),
            ("Shmem", 2048),
            ("KReclaimable", sreclaim),
            ("Slab", slab),
            ("SReclaimable", sreclaim),
        ])
    }

    #[test]
    fn leftover_pool_admits_the_reload_and_a_real_shortage_still_refuses() {
        // N=12 host-maps 46.1 GiB; the reported refusal read MemAvailable 45.8 GiB.
        let host = 47_206u64 << 20;
        let available = Some(46_899u64 << 20);
        let gtt = 5 * (16 << 20);
        assert!(check_host_ram(host, available, 0).is_err());
        let leftover = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, leftover.unwrap()).is_ok());
        // The same MemAvailable with nothing parked in the pool.
        let empty = ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, 16_108_144);
        let error = check_host_ram(host, available, empty.unwrap()).unwrap_err();
        assert!(
            error.contains("MemAvailable - CMA is 45.8 GiB (plus 0.0 GiB"),
            "{error}"
        );
        // A pool smaller than the 4.3 GiB gap still refuses.
        let small = ttm_pool_estimate_from(&gfx1201_host(3 << 10, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, small.unwrap()).is_err());
    }

    #[test]
    fn gtt_cap_counts_what_devices_already_hold() {
        // This host: pages_limit 16108144 (61.4 GiB); N=12 host-maps 46.1 GiB.
        let limit_bytes = 16_108_144 * TTM_PAGE_BYTES;
        let host = 46u64 << 30;
        let fresh = GttBudget {
            limit_bytes,
            used_bytes: 16 << 20,
        };
        assert!(check_gtt_cap(host, Some(fresh)).is_ok());
        assert!(check_gtt_cap(limit_bytes - fresh.used_bytes, Some(fresh)).is_ok());
        let error = check_gtt_cap(limit_bytes - fresh.used_bytes + 1, Some(fresh)).unwrap_err();
        assert!(
            error.contains("pages_limit") && error.contains("16108144 pages"),
            "{error}"
        );
        // A second load beside one already holding 46 GiB of GTT is refused.
        let beside = GttBudget {
            limit_bytes,
            used_bytes: 46 << 30,
        };
        assert!(check_gtt_cap(host, Some(beside)).is_err());
        // Userptr host memory, or no TTM limit in sysfs, is not GTT-capped.
        assert!(check_gtt_cap(u64::MAX, None).is_ok());
    }

    #[test]
    fn parse_accepts_auto_and_counts_only() {
        assert_eq!(
            parse_expert_vram_layers(" auto "),
            Ok(ExpertVramLayers::Auto)
        );
        assert_eq!(
            parse_expert_vram_layers("16"),
            Ok(ExpertVramLayers::Layers(16))
        );
        assert!(parse_expert_vram_layers("16GB").is_err());
    }

    #[test]
    fn unset_is_resident_only_where_everything_fits_and_explicit_wins() {
        const GIB: u64 = 1 << 30;
        // Flash-Next: ~5 GiB non-expert, ~64 GiB experts, 6.5 GiB reserve.
        let r9700 = fits_fully_resident(32 * GIB, 5 * GIB, 64 * GIB, 6 * GIB + GIB / 2);
        let halo = fits_fully_resident(120 * GIB, 5 * GIB, 64 * GIB, 6 * GIB + GIB / 2);
        assert!(!r9700 && halo);
        assert!(fits_fully_resident(10 * GIB, 2 * GIB, 7 * GIB, GIB));
        assert!(!fits_fully_resident(10 * GIB, 2 * GIB, 7 * GIB, GIB + 1));
        assert!(!fits_fully_resident(u64::MAX, 1, u64::MAX, 0));
        assert_eq!(resolve_expert_vram_layers(None, || Ok(true)), Ok(None));
        assert_eq!(
            resolve_expert_vram_layers(None, || Ok(false)),
            Ok(Some(ExpertVramLayers::Auto))
        );
        assert!(resolve_expert_vram_layers(None, || Err("vram".into())).is_err());
        // Explicit values never consult the fit.
        for policy in [
            ExpertVramLayers::Layers(12),
            ExpertVramLayers::Layers(99),
            ExpertVramLayers::Auto,
        ] {
            let resolved =
                resolve_expert_vram_layers(Some(policy), || panic!("explicit consulted the fit"));
            assert_eq!(resolved, Ok(Some(policy)));
        }
    }

    #[test]
    fn only_qwen4_on_discrete_gpus_keeps_host_memory_out_of_reclaim() {
        assert!(keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1201"]
        ));
        assert!(keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1100", "gfx942"]
        ));
        // Another model on the same card (arch 5: the Qwen3.5-family trunk).
        assert!(!keeps_host_memory_out_of_reclaim(5, &["gfx1201"]));
        // Unified memory, alone or beside a discrete card ROCr also exposes.
        assert!(!keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1151"]
        ));
        assert!(!keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1100", "gfx1151"]
        ));
        // No card known, or one whose memory kind is not known.
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &[]));
        assert!(!keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx90a"]
        ));
    }

    /// Total payload bytes of a manifest entry at 2 bytes per element.
    fn element_bytes(entry: &WeightEntry) -> Option<u64> {
        Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
    }

    #[test]
    fn dense_profile_has_no_routed_experts_and_places_nothing() {
        use crate::config::Qwen4FfnConfig;
        let config = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        );
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("dense manifest");
        assert!(config.ffn.is_dense());
        assert_eq!(routed_expert_bytes(&manifest.weights, element_bytes), Ok(0));
        let (non_expert, layer_experts) =
            resident_split(&manifest.weights, element_bytes).expect("split");
        assert_eq!(layer_experts, 0);
        // The dense gate/up/down are resident non-expert weights.
        let dense_ffn: u64 = manifest
            .weights
            .iter()
            .filter(|entry| {
                entry.name.ends_with(".mlp.gate_proj.weight")
                    || entry.name.ends_with(".mlp.up_proj.weight")
                    || entry.name.ends_with(".mlp.down_proj.weight")
            })
            .map(|entry| {
                assert_eq!(entry.residency, WeightResidency::Resident, "{}", entry.name);
                element_bytes(entry).unwrap()
            })
            .sum();
        assert!(dense_ffn > 0 && non_expert >= dense_ffn);
        // Fully resident, no division by the zero per-layer expert bytes,
        // nothing host-mapped under any placement.
        assert!(fits_fully_resident(
            non_expert + (1 << 20),
            non_expert,
            0,
            1 << 20
        ));
        assert!(!fits_fully_resident(
            non_expert + (1 << 20) - 1,
            non_expert,
            0,
            1 << 20
        ));
        assert_eq!(auto_vram_layers(1 << 40, non_expert, 0, 4, 1 << 20), 4);
        assert_eq!(auto_vram_layers(0, non_expert, 0, 4, 1 << 20), 4);
        for vram_layers in [0, 2, 4] {
            assert_eq!(place_routed_experts(&mut manifest.weights, vram_layers), 0);
        }
        assert_eq!(host_mapped_bytes(&manifest.weights, element_bytes), Ok(0));
        assert!(manifest
            .weights
            .iter()
            .all(|entry| entry.residency != WeightResidency::HostMapped));
    }

    #[test]
    fn host_expert_requests_are_refused_for_a_dense_profile_only_when_explicit() {
        assert_eq!(refuse_host_expert_request(None, "dense-dev"), Ok(()));
        for policy in [
            ExpertVramLayers::Auto,
            ExpertVramLayers::Layers(0),
            ExpertVramLayers::Layers(12),
        ] {
            let error = refuse_host_expert_request(Some(policy), "dense-dev").unwrap_err();
            assert!(
                error.contains(EXPERT_VRAM_LAYERS_ENV)
                    && error.contains("dense-dev")
                    && error.contains("no routed experts"),
                "{error}"
            );
        }
    }

    #[test]
    fn flash_next_placement_is_unchanged_by_the_dense_path() {
        // The pinned MoE manifest still counts every routed expert (trunk and
        // MTP) as an expert, none as non-expert, and host-maps exactly them.
        let config = crate::config::compact_test_config();
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        assert!(config.ffn.is_moe());
        let routed = manifest
            .weights
            .iter()
            .filter(|entry| is_routed_expert(&entry.name))
            .count();
        assert_eq!(routed, 2 * (config.num_hidden_layers + 1));
        let (non_expert, layer_experts) =
            resident_split(&manifest.weights, element_bytes).expect("split");
        assert!(layer_experts > 0);
        let experts = routed_expert_bytes(&manifest.weights, element_bytes).unwrap();
        assert!(experts > layer_experts && non_expert > 0);
        assert_eq!(place_routed_experts(&mut manifest.weights, 0), routed);
        assert!(host_mapped_bytes(&manifest.weights, element_bytes).unwrap() > 0);
        assert_eq!(
            auto_vram_layers(
                u64::MAX / 2,
                non_expert,
                layer_experts,
                config.num_hidden_layers,
                0
            ),
            config.num_hidden_layers
        );
    }

    #[test]
    fn ep_ranks_split_only_the_routed_expert_bytes() {
        let config = crate::config::compact_test_config();
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let (non_expert, layer_experts) =
            resident_split(&manifest.weights, element_bytes).expect("split");
        let experts = routed_expert_bytes(&manifest.weights, element_bytes).unwrap();
        // ranks == 1 is the single-device arithmetic.
        assert_eq!(
            resident_split_ranks(&manifest.weights, element_bytes, 1),
            Ok((non_expert, layer_experts))
        );
        assert_eq!(
            routed_expert_bytes_ranks(&manifest.weights, element_bytes, 1),
            Ok(experts)
        );
        for r in [2usize, 3, 4, 5, 6, 8] {
            // The trunk is replicated; every rank budgets the largest share:
            // the per-expert bytes times `max_local_count` experts.
            let max = Qwen4EpPlacement { rank: 0, ranks: r }.max_local_count(512) as u64;
            assert_eq!(
                resident_split_ranks(&manifest.weights, element_bytes, r),
                Ok((non_expert, layer_experts / 512 * max)),
                "ranks={r}"
            );
            assert_eq!(
                routed_expert_bytes_ranks(&manifest.weights, element_bytes, r),
                Ok(experts / 512 * max),
                "ranks={r}"
            );
        }
        // Every rank host-maps its own share; the ranks' charges sum to the
        // exact payload (plus one pad per tensor per rank), and ranks == 1
        // equals host_mapped_bytes.
        let routed = manifest
            .weights
            .iter()
            .filter(|entry| is_routed_expert(&entry.name))
            .count();
        assert_eq!(place_routed_experts(&mut manifest.weights, 0), routed);
        let single = host_mapped_bytes(&manifest.weights, element_bytes).unwrap();
        assert_eq!(
            host_mapped_bytes_ranks(&manifest.weights, element_bytes, 1),
            Ok(vec![single])
        );
        let pad = routed as u64 * HOST_MAPPED_PAD_BYTES;
        for ranks in [2usize, 3, 4, 5, 6, 8] {
            let charges = host_mapped_bytes_ranks(&manifest.weights, element_bytes, ranks).unwrap();
            assert_eq!(charges.len(), ranks);
            for (rank, &charge) in charges.iter().enumerate() {
                let local = Qwen4EpPlacement { rank, ranks }.local_count(512) as u64;
                assert_eq!(
                    charge,
                    (single - pad) / 512 * local + pad,
                    "ranks={ranks} rank={rank}"
                );
            }
            // The host charge equals the full payload: nothing lost or doubled.
            assert_eq!(
                charges.iter().sum::<u64>(),
                (single - pad) + pad * ranks as u64,
                "ranks={ranks}"
            );
        }
        // More ranks than experts, and a zero rank count, are refused.
        assert!(resident_split_ranks(&manifest.weights, element_bytes, 513).is_err());
        assert!(host_mapped_bytes_ranks(&manifest.weights, element_bytes, 513).is_err());
        assert!(host_mapped_bytes_ranks(&manifest.weights, element_bytes, 0).is_err());
        assert!(routed_expert_bytes_ranks(&manifest.weights, element_bytes, 0).is_err());
        // A stack whose bytes are not a multiple of its expert count is refused.
        assert!(resident_split_ranks(&manifest.weights, |_| Some(1), 3).is_err());
    }

    #[test]
    fn uneven_ep_split_uses_the_largest_rank_share() {
        let config = crate::config::compact_test_config();
        assert_eq!(moe_layer_count(&config), 49);
        let (free, non_expert, reserve) = (24_520u64 << 20, 5_115u64 << 20, 6_656u64 << 20);
        let per_expert = 1_336_934_400u64 / 512;
        let split = |ranks: usize| {
            let max = Qwen4EpPlacement { rank: 0, ranks }.max_local_count(512) as u64;
            ep_vram_layers(
                &config,
                &[free; 8][..ranks],
                non_expert,
                per_expert * max,
                reserve,
            )
        };
        assert_eq!(split(2), 19);
        assert_eq!(split(4), 39);
        assert_eq!(split(6), 49);
        // N = 3 keeps 29 layers: an uneven 171/171/170 split budgets 171 experts.
        assert_eq!(split(3), 29);
    }

    #[test]
    fn ep_auto_layers_use_the_per_rank_expert_bytes() {
        let free = 30u64 << 30;
        let non_expert = 5_990_000_000u64;
        let layer = 1_336_900_000u64;
        let reserve = 6_656u64 << 20;
        let single = auto_vram_layers(free, non_expert, layer, 48, reserve);
        let ep4 = auto_vram_layers(free, non_expert, layer / 4, 48, reserve);
        let ep2 = auto_vram_layers(free, non_expert, layer / 2, 48, reserve);
        assert!(single < ep2 && ep2 < ep4);
        // EP4 at 30 GiB keeps every trunk layer resident; EP2 does not.
        assert_eq!(ep4, 48);
        assert!(ep2 < 48);
    }

    #[test]
    fn ep_fit_report_prints_one_line_per_rank() {
        let config = crate::config::compact_test_config();
        let free = [30u64 << 30, 29 << 30, 30 << 30, 28 << 30];
        let report = ep_fit_report(
            &config,
            4,
            &free,
            5_990_000_000,
            334_225_000,
            6_656u64 << 20,
            32768,
            1536,
        );
        let lines: Vec<_> = report.lines().collect();
        assert_eq!(lines.len(), 5, "{report}");
        assert!(
            lines[0].contains("4 ranks") && lines[0].contains("32768") && lines[0].contains("1536")
        );
        for (rank, line) in lines[1..].iter().enumerate() {
            assert!(
                line.starts_with(&format!("qwen4 EP rank {rank}/4:")),
                "{line}"
            );
            for part in [
                "free",
                "trunk",
                "experts",
                "resident expert layers",
                "reserve",
                "headroom",
            ] {
                assert!(line.contains(part), "{part} missing from {line}");
            }
            // 49 MoE layers incl. the MTP head.
            assert!(line.contains("of 49"), "{line}");
        }
        // EP4 at 28..30 GiB free keeps all 49 MoE layers (the MTP layer too) resident.
        assert!(lines[0].contains("VRAM split 49 of 49"), "{}", lines[0]);
        assert!(
            lines[1].contains("resident expert layers 49 of 49 (0 host-mapped"),
            "{}",
            lines[1]
        );
        // Headroom: 30 GiB - 5.99 GB trunk - 49 x 0.334 GB - 6.5 GiB reserve > 0 on rank 0.
        assert!(lines[1].contains("headroom +"), "{}", lines[1]);
        // A rank with too little VRAM for the reserve reports negative headroom.
        let tight = ep_fit_rank_line(
            &config,
            1,
            2,
            8 << 30,
            5_990_000_000,
            668_450_000,
            6_656u64 << 20,
            0,
        );
        assert!(tight.contains("headroom -"), "{tight}");
        assert!(tight.contains("resident expert layers 0 of 49"), "{tight}");
    }

    #[test]
    fn ep_split_is_the_minimum_over_ranks_and_identical_everywhere() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let rank = |free| ep_rank_layers(&config, free, non_expert, layer, reserve);
        // EP2-sized layers: 49 x 0.668 GB does not fit 30 GiB with the reserve.
        let roomy = rank(30 << 30);
        let tight = rank(24 << 30);
        assert!(tight < roomy && roomy < 49 && tight > 0, "{tight} {roomy}");
        assert_eq!(rank(64 << 30), 49, "everything fits");
        assert_eq!(rank(1 << 30), 0, "reserve alone does not fit");
        let free = [30u64 << 30, 24 << 30, 30 << 30, 29 << 30];
        assert_eq!(
            ep_vram_layers(&config, &free, non_expert, layer, reserve),
            tight
        );
        assert_eq!(
            ep_vram_layers(&config, &free[..1], non_expert, layer, reserve),
            roomy
        );
        assert_eq!(ep_vram_layers(&config, &[], non_expert, layer, reserve), 0);
        // The report applies the minimum to every rank.
        let report = ep_fit_report(&config, 4, &free, non_expert, layer, reserve, 32768, 1536);
        let lines: Vec<_> = report.lines().collect();
        assert!(
            lines[0].contains(&format!("VRAM split {tight} of 49")),
            "{}",
            lines[0]
        );
        for line in &lines[1..] {
            assert!(
                line.contains(&format!("resident expert layers {tight} of 49")),
                "{line}"
            );
        }
        assert!(
            lines[2].contains(&format!("this rank alone fits {tight}")),
            "{}",
            lines[2]
        );
        assert!(
            lines[1].contains(&format!("this rank alone fits {roomy}")),
            "{}",
            lines[1]
        );
    }

    fn band_scopes(config: &Qwen4Config, counts: &[usize]) -> Vec<Qwen4BandScope> {
        assert_eq!(counts.iter().sum::<usize>(), config.num_hidden_layers);
        let mut start = 0;
        counts
            .iter()
            .map(|&count| {
                let band = start..start + count;
                start += count;
                let scope = Qwen4BandScope {
                    input: band.start == 0,
                    output: band.end == config.num_hidden_layers,
                    band,
                };
                scope.validate(config).expect("valid band");
                scope
            })
            .collect()
    }

    fn census_of(
        config: &Qwen4Config,
        weights: &[WeightEntry],
        scope: &Qwen4BandScope,
    ) -> BandCensus {
        let context = crate::Qwen4ContextCommit::legacy(1024);
        band_census(
            config,
            weights,
            element_bytes,
            scope,
            target_state_format(),
            &context,
            256,
            Full,
        )
        .expect("band census")
    }

    #[test]
    fn single_band_census_equals_todays_whole_model_queries() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let single = Qwen4BandScope::single(&config);
        let census = census_of(&config, weights, &single);
        let (non_expert, _) = resident_split(weights, element_bytes).unwrap();
        let routed = routed_expert_bytes(weights, element_bytes).unwrap();
        assert!(routed > 0 && non_expert > 0);
        assert_eq!(census.non_expert_bytes, non_expert);
        assert_eq!(census.routed_bytes(), Ok(routed));
        assert_eq!(
            routed_expert_bytes_scope(weights, element_bytes, &single),
            Ok(routed)
        );
        assert_eq!(census.endpoint_duplicate_bytes, 0);
        assert_eq!(census.moe_layers, moe_layer_count(&config));
        assert_eq!(
            moe_layer_count_scope(&config, &single),
            moe_layer_count(&config)
        );
        assert_eq!(
            census.f16_shadow_bytes,
            f16_shadow_bytes(weights, config.num_hidden_layers).unwrap()
        );
        let context = crate::Qwen4ContextCommit::legacy(1024);
        assert_eq!(
            census.state,
            state_census(&config, target_state_format(), &context, 256, Full).unwrap()
        );
        assert_eq!(
            refuse_band_with_expert_parallel(&config, &single, 4),
            Ok(())
        );
    }

    #[test]
    fn pp_band_censuses_sum_to_single_plus_only_the_declared_endpoint_duplicates() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let single = census_of(&config, weights, &Qwen4BandScope::single(&config));
        let embed = weights
            .iter()
            .find(|entry| entry.name == "model.language_model.embed_tokens.weight")
            .map(|entry| element_bytes(entry).unwrap())
            .expect("embedding");
        let layer_bytes = single.routed_trunk_bytes / config.num_hidden_layers as u64;
        assert!(layer_bytes > 0 && single.routed_mtp_bytes > 0);
        assert_eq!(single.max_layer_routed_bytes, layer_bytes);

        for counts in [&[24usize, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let scopes = band_scopes(&config, counts);
            let ranks: Vec<_> = scopes
                .iter()
                .map(|scope| census_of(&config, weights, scope))
                .collect();
            let sum = |field: fn(&BandCensus) -> u64| ranks.iter().map(field).sum::<u64>();
            // Trunk experts: every owned layer exactly once, all experts.
            assert_eq!(
                sum(|c| c.routed_trunk_bytes),
                single.routed_trunk_bytes,
                "{counts:?}"
            );
            for (census, count) in ranks.iter().zip(counts) {
                assert_eq!(census.routed_trunk_bytes, layer_bytes * *count as u64);
                assert_eq!(census.max_layer_routed_bytes, layer_bytes);
                assert_eq!(census.band.len(), *count);
            }
            // The MTP head's experts and layer only on the output rank.
            let last = ranks.len() - 1;
            for (rank, census) in ranks.iter().enumerate() {
                let mtp = rank == last;
                assert_eq!(
                    census.routed_mtp_bytes,
                    if mtp { single.routed_mtp_bytes } else { 0 }
                );
                assert_eq!(census.moe_layers, counts[rank] + usize::from(mtp));
            }
            assert_eq!(
                ranks.iter().map(|c| c.moe_layers).sum::<usize>(),
                single.moe_layers
            );
            assert_eq!(sum(|c| c.routed_mtp_bytes), single.routed_mtp_bytes);
            assert_eq!(sum(|c| c.non_expert_bytes), single.non_expert_bytes);
            assert_eq!(sum(|c| c.f16_shadow_bytes), single.f16_shadow_bytes);
            // The only duplicate is the tied MTP embedding at the output endpoint.
            assert_eq!(sum(|c| c.endpoint_duplicate_bytes), embed);
            assert_eq!(ranks[last].endpoint_duplicate_bytes, embed);
            assert_eq!(
                ranks.iter().map(|c| c.weight_bytes().unwrap()).sum::<u64>(),
                single.weight_bytes().unwrap() + embed
            );
            // State: owned layers only, PLE and feedback only on the input band.
            assert_eq!(sum(|c| c.state.qsa_context), single.state.qsa_context);
            assert_eq!(sum(|c| c.state.state), single.state.state);
            // Forward scratch: input-only and output-only fields exactly once,
            // the per-stage shared copy and the GDN/QSA fields on every band
            // that runs them (every band here has both layer kinds).
            assert_eq!(sum(|c| c.forward.input_only), single.forward.input_only);
            assert_eq!(sum(|c| c.forward.output_only), single.forward.output_only);
            for (rank, census) in ranks.iter().enumerate() {
                assert_eq!(census.forward.stage_shared, single.forward.stage_shared);
                assert_eq!(census.forward.band_local, single.forward.band_local);
                assert_eq!(census.state.forward, census.forward.total().unwrap());
                assert_eq!(census.forward.input_only == 0, rank != 0);
                assert_eq!(census.forward.output_only == 0, rank != last);
            }
            // Input holds the embedding, output the head: heavier than a middle rank.
            if ranks.len() > 2 {
                let middle = &ranks[1];
                assert!(ranks[0].non_expert_bytes > middle.non_expert_bytes);
                assert!(ranks[last].non_expert_bytes > middle.non_expert_bytes);
            }
        }
    }

    #[test]
    fn native_mtp_single_scope_census_is_the_single_load_formula() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let head = language_head_dtype(weights).expect("language head");
        let single = Qwen4BandScope::single(&config);
        let context = crate::Qwen4ContextCommit::legacy(1024);
        for gdn in [
            None,
            Some((crate::GdnStateFormat::Q8, 32)),
            Some((crate::GdnStateFormat::F32, 32)),
        ] {
            for batched_fill in [false, true] {
                for max_k in [1usize, 3, 10, 64] {
                    let inputs = NativeMtpCensusInputs {
                        max_k,
                        row_capture: gdn,
                        batched_fill,
                    };
                    let census =
                        native_mtp_census_scope(&config, weights, &context, 256, &inputs, &single)
                            .expect("single census");
                    assert_eq!(
                        census.reserve().ok(),
                        crate::mtp_spec::native_mtp_device_bytes(
                            &config,
                            &context,
                            256,
                            max_k,
                            head,
                            gdn,
                            batched_fill,
                        ),
                        "gdn {gdn:?} fill {batched_fill} K {max_k}"
                    );
                    assert!(census.head_resident > 0 && census.request > 0);
                    assert_eq!(census.row_capture > 0, gdn.is_some());
                }
            }
        }
    }

    #[test]
    fn native_mtp_pp_bands_charge_their_own_capture_and_only_the_output_the_head() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let gdn = crate::GdnStateFormat::Q8;
        let inputs = NativeMtpCensusInputs {
            max_k: 3,
            row_capture: Some((gdn, 4)),
            batched_fill: true,
        };
        let whole = native_mtp_census_scope(
            &config,
            weights,
            &context,
            256,
            &inputs,
            &Qwen4BandScope::single(&config),
        )
        .unwrap();
        for counts in [&[24usize, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let scopes = band_scopes(&config, counts);
            for scope in &scopes {
                let census =
                    native_mtp_census_scope(&config, weights, &context, 256, &inputs, scope)
                        .unwrap();
                let gdn_layers = scope
                    .band
                    .clone()
                    .filter(|&layer| {
                        config.layer_types[layer] == crate::config::LayerType::LinearAttention
                    })
                    .count();
                // Every band: the capture of its own GDN layers, four rows.
                assert!(gdn_layers > 0);
                assert_eq!(
                    census.row_capture,
                    crate::state::Qwen4State::row_capture_bytes_scoped(&config, gdn, 4, gdn_layers)
                        .unwrap() as u64,
                    "{counts:?} {:?}",
                    scope.band
                );
                // No band is charged the whole model's capture.
                assert!(census.row_capture < whole.row_capture, "{counts:?}");
                if scope.output {
                    // The head and the request buffers once, on the output band.
                    assert_eq!(census.head_resident, whole.head_resident);
                    assert_eq!(census.head_build_scratch, whole.head_build_scratch);
                    assert_eq!(census.request, whole.request);
                    assert_eq!(
                        census.reserve().unwrap(),
                        census.head_resident
                            + census
                                .head_build_scratch
                                .max(census.request + census.row_capture)
                    );
                } else {
                    assert_eq!(
                        (
                            census.head_resident,
                            census.head_build_scratch,
                            census.request
                        ),
                        (0, 0, 0)
                    );
                    assert_eq!(census.reserve().unwrap(), census.row_capture);
                }
            }
            // Row capture off (a GPU without the few-row GDN route): no band
            // charges it, the output band keeps the head and the request.
            let off = NativeMtpCensusInputs {
                row_capture: None,
                ..inputs
            };
            for scope in &scopes {
                let census =
                    native_mtp_census_scope(&config, weights, &context, 256, &off, scope).unwrap();
                assert_eq!(census.row_capture, 0);
                assert_eq!(census.reserve().unwrap() > 0, scope.output);
            }
        }
    }

    #[test]
    fn band_census_adds_only_the_mtp_terms_and_ar_has_none() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let inputs = NativeMtpCensusInputs {
            max_k: 3,
            row_capture: Some((crate::GdnStateFormat::Q8, 4)),
            batched_fill: false,
        };
        for scope in band_scopes(&config, &[24, 24]) {
            let ar = census_of(&config, weights, &scope);
            assert_eq!(ar.mtp, NativeMtpCensus::default());
            // The head's weights are loaded (and counted) on the output band
            // whether or not the head attaches.
            assert_eq!(ar.routed_mtp_bytes > 0, scope.output);
            let with = band_census_native_mtp(
                &config,
                weights,
                element_bytes,
                &scope,
                target_state_format(),
                &context,
                256,
                Full,
                Some(&inputs),
            )
            .unwrap();
            assert_eq!(
                with.mtp,
                native_mtp_census_scope(&config, weights, &context, 256, &inputs, &scope).unwrap()
            );
            assert!(with.mtp.reserve().unwrap() > 0);
            assert_eq!(
                BandCensus {
                    mtp: NativeMtpCensus::default(),
                    ..with
                },
                ar
            );
        }
    }

    #[test]
    fn native_mtp_census_refuses_a_profile_without_a_head() {
        use crate::config::Qwen4FfnConfig;
        let headless = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            false,
        );
        let inputs = NativeMtpCensusInputs {
            max_k: 3,
            row_capture: None,
            batched_fill: false,
        };
        let error = native_mtp_census_scope(
            &headless,
            &[],
            &crate::Qwen4ContextCommit::legacy(64),
            8,
            &inputs,
            &Qwen4BandScope::single(&headless),
        )
        .unwrap_err();
        assert!(error.contains("no MTP head"), "{error}");
    }

    #[test]
    fn single_forward_census_equals_the_whole_model_forward_bytes() {
        use crate::config::Qwen4FfnConfig;
        let dense = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        );
        let mut full_seen = 0;
        for config in [crate::config::compact_test_config(), dense] {
            let single = Qwen4BandScope::single(&config);
            for rows in [1usize, 7, 64, 256, 513, 2048, 8192] {
                for mode in [Full, FnSymGfx1201] {
                    let scoped = forward_census_scope(&config, rows, mode, &single)
                        .and_then(|census| census.total())
                        .ok();
                    let whole = crate::gpu_forward::qwen4_forward_device_bytes(&config, rows, mode);
                    assert_eq!(scoped, whole, "rows {rows} {mode:?}");
                    full_seen += usize::from(scoped.is_some() && mode == FnSymGfx1201);
                }
            }
        }
        assert!(
            full_seen > 0,
            "the compact plan must be compared at least once"
        );
    }

    #[test]
    fn compact_forward_census_refuses_a_gdn_band_without_a_qsa_owner() {
        let config = crate::config::compact_test_config();
        // Layers 0..3 are GDN, the first QSA layer is 3.
        let gdn_only = Qwen4BandScope {
            band: 0..3,
            input: true,
            output: false,
        };
        assert!(forward_census_scope(&config, 256, FnSymGfx1201, &gdn_only).is_err());
        let full = forward_census_scope(&config, 256, Full, &gdn_only).expect("full plan");
        assert!(full.band_local > 0 && full.output_only == 0 && full.input_only > 0);
        let single = forward_census_scope(&config, 256, Full, &Qwen4BandScope::single(&config))
            .expect("single");
        // No QSA scratch and a narrower projection than the whole model.
        assert!(full.band_local < single.band_local && full.stage_shared < single.stage_shared);
    }

    #[test]
    fn band_census_of_a_scope_filtered_manifest_equals_the_full_manifests() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let embed = manifest
            .weights
            .iter()
            .find(|entry| entry.name == "model.language_model.embed_tokens.weight")
            .map(|entry| element_bytes(entry).unwrap())
            .expect("embedding");
        for counts in [&[48usize][..], &[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            for scope in band_scopes(&config, counts) {
                let scoped = manifest
                    .clone()
                    .with_band_scope(&config, &scope)
                    .expect("scope-filtered manifest");
                if !scope.input {
                    // The tied source is gone from the filtered manifest.
                    assert!(scoped
                        .weights
                        .iter()
                        .all(|entry| entry.name != "model.language_model.embed_tokens.weight"));
                }
                let full = census_of(&config, &manifest.weights, &scope);
                let filtered = census_of(&config, &scoped.weights, &scope);
                assert_eq!(full, filtered, "{counts:?} band {:?}", scope.band);
                let duplicate = if scope.output && !scope.input {
                    embed
                } else {
                    0
                };
                assert_eq!(
                    filtered.endpoint_duplicate_bytes, duplicate,
                    "{:?}",
                    scope.band
                );
            }
        }
    }

    #[test]
    fn slab_arena_band_census_charges_each_band_its_own_arenas() {
        use crate::expert_slabs::ExpertFormat;
        // Non-uniform arenas: layer `l` holds `(5 * l) % 64` MQ4 experts and W3 for the rest.
        let tensors = crate::expert_slab_table::full_fn_headers(|layer, expert| {
            if expert < (layer * 5) % 64 {
                ExpertFormat::Mq4Xts
            } else {
                ExpertFormat::Mq3I4
            }
        });
        let table = crate::expert_slab_table::build_expert_slab_table(
            &crate::expert_slabs::ExpertSlabsV1::fn_v1(),
            &tensors,
        )
        .unwrap();
        let plan =
            std::sync::Arc::new(crate::expert_arena::ExpertArenaPlan::from_table(&table).unwrap());
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build_with_recipe(
            &config,
            crate::weights::Qwen4TierRecipe::FlashNextSlab {
                trunk_tier: DType::Q8_0,
                mtp_source: false,
                arenas: plan.clone(),
            },
        )
        .expect("slab manifest");
        // An arena's bytes are its raw length; everything else is 2 B/element.
        let bytes_of = |entry: &WeightEntry| {
            if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
                Some(entry.logical_shape[0] as u64)
            } else {
                element_bytes(entry)
            }
        };
        let arena = |layer: usize| plan.layer(layer).expect("layer").bytes as u64;
        let mtp_bytes: u64 = manifest
            .weights
            .iter()
            .filter(|entry| entry.name.starts_with("mtp.") && entry.name.contains(".mlp.experts."))
            .map(|entry| element_bytes(entry).expect("mtp aggregate"))
            .sum();
        assert!(mtp_bytes > 0, "the manifest declares the MTP aggregates");
        let distinct: std::collections::BTreeSet<_> = (0..48).map(|layer| arena(layer)).collect();
        assert_eq!(distinct.len(), 48, "every layer's arena differs");

        let context = crate::Qwen4ContextCommit::legacy(1024);
        let census = |weights: &[WeightEntry], scope: &Qwen4BandScope| {
            band_census(
                &config,
                weights,
                bytes_of,
                scope,
                target_state_format(),
                &context,
                256,
                Full,
            )
            .expect("band census")
        };
        let single = census(&manifest.weights, &Qwen4BandScope::single(&config));
        assert_eq!(
            single.routed_trunk_bytes,
            (0..48).map(arena).sum::<u64>(),
            "the single scope charges every arena exactly"
        );
        assert_eq!(single.routed_mtp_bytes, mtp_bytes);
        assert_eq!(
            single.max_layer_routed_bytes,
            (0..48).map(arena).max().unwrap()
        );

        for counts in [&[24usize, 24][..], &[16, 16, 16][..]] {
            let mut trunk_sum = 0u64;
            let mut mtp_sum = 0u64;
            for scope in band_scopes(&config, counts) {
                let own: Vec<u64> = scope.band.clone().map(arena).collect();
                let scoped = manifest
                    .clone()
                    .with_band_scope(&config, &scope)
                    .expect("scope-filtered manifest");
                // The full and the scope-filtered manifest give the same census.
                let from_full = census(&manifest.weights, &scope);
                let from_scoped = census(&scoped.weights, &scope);
                assert_eq!(from_full, from_scoped, "{counts:?} band {:?}", scope.band);
                assert_eq!(
                    from_scoped.routed_trunk_bytes,
                    own.iter().sum::<u64>(),
                    "{counts:?} band {:?}",
                    scope.band
                );
                assert_eq!(
                    from_scoped.max_layer_routed_bytes,
                    own.iter().copied().max().unwrap(),
                    "{counts:?} band {:?}",
                    scope.band
                );
                assert_eq!(
                    from_scoped.routed_mtp_bytes,
                    if scope.output { mtp_bytes } else { 0 },
                    "{counts:?} band {:?}",
                    scope.band
                );
                trunk_sum += from_scoped.routed_trunk_bytes;
                mtp_sum += from_scoped.routed_mtp_bytes;
            }
            assert_eq!(trunk_sum, single.routed_trunk_bytes, "{counts:?}");
            assert_eq!(mtp_sum, single.routed_mtp_bytes, "{counts:?}");
        }
    }

    #[test]
    fn band_census_refuses_expert_sharded_entries_for_a_band_but_not_single() {
        use hipfire_runtime::tp_shard::ExpertAssign;
        let config = crate::config::compact_test_config();
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let sharded = manifest
            .weights
            .iter_mut()
            .find(|entry| is_routed_expert(&entry.name) && entry.layer == Some(0))
            .expect("routed expert entry");
        sharded.policy = ShardPolicy::ExpertSharded {
            n_experts: config.num_experts,
            assign: ExpertAssign::Stride,
        };
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let calls = std::cell::Cell::new(0usize);
        let counted = |entry: &WeightEntry| {
            calls.set(calls.get() + 1);
            element_bytes(entry)
        };
        for scope in band_scopes(&config, &[24, 24]) {
            let error = band_census(
                &config,
                &manifest.weights,
                counted,
                &scope,
                target_state_format(),
                &context,
                256,
                Full,
            )
            .unwrap_err();
            assert!(error.contains("expert parallelism"), "{error}");
            assert!(
                routed_expert_bytes_scope(&manifest.weights, counted, &scope)
                    .unwrap_err()
                    .contains("expert parallelism")
            );
        }
        assert_eq!(calls.get(), 0, "refused before any payload callback");
        let single = Qwen4BandScope::single(&config);
        assert!(band_census(
            &config,
            &manifest.weights,
            element_bytes,
            &single,
            target_state_format(),
            &context,
            256,
            Full,
        )
        .is_ok());
    }

    #[test]
    fn band_census_refuses_a_band_without_the_ple_layer_and_pp_with_ep() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let ple_layer = config.ple.as_ref().expect("PLE profile").layer;
        let bad = Qwen4BandScope {
            band: 0..ple_layer,
            input: true,
            output: false,
        };
        let error = band_census(
            &config,
            &manifest.weights,
            element_bytes,
            &bad,
            target_state_format(),
            &context,
            256,
            Full,
        )
        .unwrap_err();
        assert!(error.contains("PLE"), "{error}");
        let scopes = band_scopes(&config, &[24, 24]);
        for scope in &scopes {
            assert_eq!(refuse_band_with_expert_parallel(&config, scope, 1), Ok(()));
            let error = refuse_band_with_expert_parallel(&config, scope, 2).unwrap_err();
            assert!(error.contains("expert parallelism"), "{error}");
        }
    }

    /// One recurrent layer's F32 storage minus its Q8 storage (3145728 - 811008).
    const GDN_SLOT_DELTA: u64 = 2_334_720;

    fn precision_format(qsa: QsaKvFormat, gdn: crate::GdnStateFormat) -> crate::Qwen4StateFormat {
        crate::Qwen4StateFormat {
            qsa,
            gdn,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        }
    }

    #[test]
    fn state_census_charges_f32_gdn_as_live_plus_mirror_over_q8() {
        let config = crate::config::compact_test_config();
        let gdn_layers = config.n_linear_layers() as u64;
        assert_eq!(gdn_layers, 36);
        let max_seq = 17_408;
        let granularity = 2 << 20;
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let full = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
            for qsa in [QsaKvFormat::Fp8, F32, QsaKvFormat::Q8] {
                let q8 = state_census(
                    &config,
                    precision_format(qsa, crate::GdnStateFormat::Q8),
                    &full,
                    4096,
                    Full,
                )
                .unwrap();
                let f32 = state_census(
                    &config,
                    precision_format(qsa, crate::GdnStateFormat::F32),
                    &full,
                    4096,
                    Full,
                )
                .unwrap();
                // Exactly (F32 - Q8) x (live + mirror) x the 36 GDN layers:
                // 2 x 80.15625 MiB. The context, forward scratch and every
                // other term are the same inputs: nothing is lowered.
                assert_eq!(
                    f32.state - q8.state,
                    2 * gdn_layers * GDN_SLOT_DELTA,
                    "{backend:?} {qsa:?}"
                );
                assert_eq!(f32.state - q8.state, 168_099_840);
                assert_eq!(f32.qsa_context, q8.qsa_context, "{backend:?} {qsa:?}");
                assert_eq!(f32.forward, q8.forward, "{backend:?} {qsa:?}");
                // The reserve's total moves by the same bytes and no more.
                let total = |census: StateCensus| {
                    census_with(census.qsa_context, census.state, census.forward, 0, 0)
                        .total()
                        .unwrap()
                };
                assert_eq!(total(f32) - total(q8), 168_099_840);
            }
        }
    }

    #[test]
    fn state_census_scope_charges_local_gdn_layers_and_ep_replicates() {
        let config = crate::config::compact_test_config();
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let q8 = precision_format(QsaKvFormat::Fp8, crate::GdnStateFormat::Q8);
        let f32 = precision_format(QsaKvFormat::Fp8, crate::GdnStateFormat::F32);
        let delta = |scope: &Qwen4BandScope| {
            let q8 = state_census_scope(&config, q8, &context, 256, Full, scope).unwrap();
            let f32 = state_census_scope(&config, f32, &context, 256, Full, scope).unwrap();
            assert_eq!(q8.qsa_context, f32.qsa_context);
            assert_eq!(q8.forward, f32.forward);
            f32.state - q8.state
        };
        let whole = delta(&Qwen4BandScope::single(&config));
        assert_eq!(whole, 2 * 36 * GDN_SLOT_DELTA);
        // PP: each band is charged its own GDN layers (live + mirror); the
        // bands sum to the whole model, nothing is divided or duplicated.
        for counts in [&[24usize, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let mut sum = 0;
            for scope in band_scopes(&config, counts) {
                let (gdn_layers, _) = crate::state::scope_layer_counts(&config, &scope);
                assert!(gdn_layers > 0 && gdn_layers < 36);
                let band = delta(&scope);
                assert_eq!(band, 2 * gdn_layers as u64 * GDN_SLOT_DELTA, "{counts:?}");
                sum += band;
            }
            assert_eq!(sum, whole, "{counts:?}");
        }
        // EP: every rank replicates the recurrent state, so the census of a
        // whole-model rank is the single census at any rank count, and a band
        // never combines with EP.
        let single = Qwen4BandScope::single(&config);
        for ranks in 1..=8 {
            assert_eq!(
                refuse_band_with_expert_parallel(&config, &single, ranks),
                Ok(())
            );
            assert_eq!(
                state_census_scope(&config, f32, &context, 256, Full, &single).unwrap(),
                state_census(&config, f32, &context, 256, Full).unwrap(),
                "{ranks} ranks"
            );
        }
        for scope in band_scopes(&config, &[24, 24]) {
            assert!(refuse_band_with_expert_parallel(&config, &scope, 2).is_err());
        }
    }

    #[test]
    fn state_census_native_mtp_reserve_charges_the_f32_ring_per_band() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let inputs = |gdn, ring| NativeMtpCensusInputs {
            max_k: 3,
            row_capture: Some((gdn, ring)),
            batched_fill: false,
        };
        for ring in [4usize, 8, 32] {
            let census = |gdn, scope: &Qwen4BandScope| {
                native_mtp_census_scope(&config, weights, &context, 256, &inputs(gdn, ring), scope)
                    .unwrap()
            };
            for counts in [&[48usize][..], &[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
                for scope in band_scopes(&config, counts) {
                    let (gdn_layers, _) = crate::state::scope_layer_counts(&config, &scope);
                    let q8 = census(crate::GdnStateFormat::Q8, &scope);
                    let f32 = census(crate::GdnStateFormat::F32, &scope);
                    // The band's own 2R ring slots per local GDN layer.
                    assert_eq!(
                        f32.row_capture - q8.row_capture,
                        gdn_layers as u64 * 2 * ring as u64 * GDN_SLOT_DELTA,
                        "{counts:?} {:?} R={ring}",
                        scope.band
                    );
                    // The head and request terms do not depend on the format.
                    assert_eq!(f32.head_resident, q8.head_resident);
                    assert_eq!(f32.request, q8.request);
                    assert_eq!(
                        f32.reserve().unwrap() - q8.reserve().unwrap(),
                        f32.row_capture - q8.row_capture,
                        "{counts:?} {:?} R={ring}",
                        scope.band
                    );
                }
            }
        }
    }

    // ---- XTX placement: receipt-driven CPU dryrun --------------------------
    //
    // Inputs are the real gfx1100 XTX hold receipts
    // (`g2-xtx-default/hold-20261017T102413Z/pf-{base,cand}/stderr.txt`):
    // Flash-Next xts, `max_seq` 8224, legacy QSA context (q8 K/V), packed
    // (aggregate) experts so no dedicated stage is charged, 1275 MiB per
    // expert layer (512 experts x (1,740,800 + 870,400) B, exact), 5115 MiB
    // of non-expert weights and the logged free VRAM. The logs print free
    // VRAM and non-expert bytes floored to MiB, so those two are exact only
    // to the MiB; the reserve is recomputed by the repo's own
    // `auto_vram_reserve` for each chunk and cross-checked against the
    // receipt's logged MiB below.
    const XTX_MAX_SEQ: usize = 8224;
    const XTX_NON_EXPERT: u64 = 5115 << 20;
    const XTX_GATE_UP: u64 = 512 * 1_740_800;
    const XTX_DOWN: u64 = 512 * 870_400;
    /// Free VRAM the pf-base and pf-cand placement logs print.
    const XTX_FREE_BASE: u64 = 24522 << 20;
    const XTX_FREE_CAND: u64 = 24376 << 20;

    fn xtx_reserve(chunk: usize) -> AutoReserve {
        let config = crate::config::compact_test_config();
        let context = crate::Qwen4ContextCommit::legacy(XTX_MAX_SEQ);
        let historical =
            auto_vram_reserve(&config, &context, chunk, Full, QsaKvFormat::Q8, None, None).unwrap();
        // The census terms the receipts print (QSA context 132 MiB, state
        // 65 MiB, forward at this chunk); it stays below the floor at every
        // rung, as in both receipts, so the historical floor sets the reserve.
        let forward = crate::gpu_forward::qwen4_forward_device_bytes(&config, chunk, Full).unwrap();
        AutoReserve::new(
            Some(historical),
            census_with(132 << 20, 65 << 20, forward, 0, 0),
        )
        .unwrap()
    }

    fn xtx_manifest() -> Vec<WeightEntry> {
        let mut weights = vec![entry("model.language_model.embed_tokens.weight", 0)];
        weights[0].layer = None;
        for layer in 0..48 {
            for suffix in ["gate_up_proj", "down_proj"] {
                weights.push(entry(
                    &format!("model.language_model.layers.{layer}.mlp.experts.{suffix}"),
                    layer,
                ));
            }
        }
        for suffix in ["gate_up_proj", "down_proj"] {
            let mut mtp = entry(&format!("mtp.layers.0.mlp.experts.{suffix}"), 0);
            mtp.layer = Some(0);
            weights.push(mtp);
        }
        weights
    }

    fn xtx_bytes(entry: &WeightEntry) -> Option<u64> {
        Some(if entry.name.ends_with("experts.gate_up_proj") {
            XTX_GATE_UP
        } else if entry.name.ends_with("experts.down_proj") {
            XTX_DOWN
        } else {
            XTX_NON_EXPERT
        })
    }

    /// What one load resolves, in the loader's order: the requested chunk
    /// (the link is read only on gfx1100 without an explicit value), then the
    /// placement sized from it with the placement stage switch.
    struct XtxDryrun {
        placement_rows: usize,
        resident_layers: usize,
        host_tensors: usize,
        host_bytes: u64,
        resident_bytes: u64,
        donors: Option<[usize; 2]>,
        mirror_bytes: u64,
    }

    fn xtx_dryrun(
        arch: &str,
        free: u64,
        chunk_env: Option<&str>,
        stage_env: Option<&str>,
        link: crate::host_link::HostLink,
        slab: bool,
    ) -> XtxDryrun {
        let mut weights = xtx_manifest();
        let (non_expert, layer) = resident_split(&weights, xtx_bytes).unwrap();
        assert_eq!(
            (non_expert, layer),
            (XTX_NON_EXPERT, XTX_GATE_UP + XTX_DOWN)
        );
        // Placement: the requested chunk, the placement stage switch.
        let placement_rows =
            crate::gpu_forward::prefill_chunk_requested_from(arch, chunk_env, XTX_MAX_SEQ, || link);
        let charges = PlacementCharges::new(
            layer,
            dedicated_slab_stage(expert_stage_placement_switch(arch, stage_env), slab, false),
        )
        .unwrap();
        let reserve = xtx_reserve(placement_rows);
        let resident_layers =
            auto_placement_layers(free, non_expert, layer, 48, reserve.bytes(), &charges).unwrap();
        let host_tensors = place_routed_experts(&mut weights, resident_layers);
        let host_bytes = host_mapped_bytes(&weights, xtx_bytes).unwrap();
        let resident_bytes = resident_routed_expert_bytes(&weights, xtx_bytes).unwrap();
        // The borrowed packed stage lends the last two resident layers; its
        // restore copies are separate pinned host RAM.
        let donors = (resident_layers >= 2).then(|| {
            let mut donors = [resident_layers - 2, resident_layers - 1];
            donors.sort_by_key(|layer| layer % 2);
            donors
        });
        let mirror_bytes = expert_stage_mirror_bytes(layer as usize, 1).unwrap();
        XtxDryrun {
            placement_rows,
            resident_layers,
            host_tensors,
            host_bytes,
            resident_bytes,
            donors,
            mirror_bytes,
        }
    }

    impl XtxDryrun {
        fn print(&self, label: &str) {
            const MIB: u64 = 1 << 20;
            let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
            println!(
                "[dryrun {label}] placement chunk {} rows -> {} resident layers ({} MiB experts), \
                 {} host expert tensors ({:.1} GiB pinned, placement only); \
                 G2 borrowed stage: donors {:?}, dedicated VRAM 0 MiB, restore mirrors {} MiB \
                 pinned host RAM charged separately",
                self.placement_rows,
                self.resident_layers,
                self.resident_bytes / MIB,
                self.host_tensors,
                gib(self.host_bytes),
                self.donors,
                (self.mirror_bytes - 4 * HOST_MAPPED_PAD_BYTES) / MIB,
            );
        }
    }

    #[test]
    fn xtx_placement_reserves_match_the_receipts() {
        // The receipts' logged reserves (floored MiB): base 5261 at the
        // 1536-row chunk, 12110 at 8192; the mid rungs grow by the
        // forward-resource deltas of the same formula.
        let at = |rows| xtx_reserve(rows).bytes() >> 20;
        assert_eq!(at(1536), 5261);
        assert_eq!(at(8192), 12110);
        assert!(at(2048) > at(1536) && at(4096) > at(2048) && at(8192) > at(4096));
        assert_eq!(xtx_reserve(1536).winner(), "historical floor");
        assert_eq!(xtx_reserve(8192).winner(), "historical floor");
        // With the margin the receipts print 5773 and 12622 MiB.
        let charges = PlacementCharges::new(XTX_GATE_UP + XTX_DOWN, false).unwrap();
        assert_eq!(
            charges.reserve(xtx_reserve(1536).bytes()).unwrap() >> 20,
            5773
        );
        assert_eq!(
            charges.reserve(xtx_reserve(8192).bytes()).unwrap() >> 20,
            12622
        );
    }

    #[test]
    fn xtx_placement_default_cedes_layers_only_on_a_measured_narrow_link() {
        use crate::host_link::HostLink::{Discrete, Unified, Unmeasured};
        // pf-base: no link policy, beta chunk and stage defaults.
        let beta = xtx_dryrun("gfx1100", XTX_FREE_BASE, None, None, Unmeasured, false);
        // A fast link keeps the beta 1536-row placement on the candidate's
        // free VRAM, as do unified and unmeasured links.
        let fast = xtx_dryrun(
            "gfx1100",
            XTX_FREE_CAND,
            None,
            None,
            Discrete { gbps: 26.0 },
            false,
        );
        let unified = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, None, Unified, false);
        // The failed candidate's environment: a measured 7.1 GB/s narrow link
        // and 146 MiB less free VRAM, nothing set explicitly.
        let narrow = xtx_dryrun(
            "gfx1100",
            XTX_FREE_CAND,
            None,
            None,
            Discrete { gbps: 7.1 },
            false,
        );
        beta.print("beta (pf-base inputs)");
        fast.print("fast link (pf-cand inputs)");
        narrow.print("narrow link default (pf-cand inputs)");
        for run in [&beta, &fast, &unified] {
            assert_eq!(run.placement_rows, 1536);
            assert_eq!(run.resident_layers, 10);
            assert_eq!(run.host_tensors, 78);
            // 38 trunk layers + the MTP layer, each tensor with its pinned pad.
            assert_eq!(
                run.host_bytes,
                39 * (XTX_GATE_UP + XTX_DOWN) + 78 * HOST_MAPPED_PAD_BYTES
            );
            assert_eq!(
                format!("{:.1}", run.host_bytes as f64 / (1u64 << 30) as f64),
                "48.6"
            );
            assert_eq!(run.resident_bytes, 10 * (XTX_GATE_UP + XTX_DOWN));
            // The borrowed donors of ten resident layers, parity ordered.
            assert_eq!(run.donors, Some([8, 9]));
        }
        // The narrow-link default places for the 8192-row reserve: it cedes
        // layers (the receipt's five-layer split) for the wider chunk.
        assert_eq!(narrow.placement_rows, 8192);
        assert_eq!(narrow.resident_layers, 5);
        assert_eq!(narrow.host_tensors, 88);
        assert_eq!(narrow.donors, Some([4, 3]));
        assert!(narrow.resident_bytes < beta.resident_bytes);
        assert!(narrow.host_bytes > beta.host_bytes);
        // The restore mirrors are a separate host charge: two donor layers,
        // 2550 MiB of copies (plus four tensor pads in the host check).
        assert_eq!(narrow.mirror_bytes - 4 * HOST_MAPPED_PAD_BYTES, 2550 << 20);
        assert_eq!(narrow.mirror_bytes, beta.mirror_bytes);
    }

    #[test]
    fn xtx_placement_explicit_chunk_is_authoritative_whatever_the_link() {
        use crate::host_link::HostLink::{Discrete, Unmeasured};
        // HIPFIRE_PREFILL_CHUNK_ROWS=8192 sizes the placement exactly as the
        // narrow-link default does (the failed candidate's five-layer split:
        // the receipt placement, 88 host tensors, donors 4 and 3), and never
        // reads the link: a fast or unmeasured link places the same.
        for link in [Discrete { gbps: 7.1 }, Discrete { gbps: 26.0 }, Unmeasured] {
            let opt_in = xtx_dryrun("gfx1100", XTX_FREE_CAND, Some("8192"), None, link, false);
            opt_in.print("explicit HIPFIRE_PREFILL_CHUNK_ROWS=8192");
            assert_eq!(opt_in.placement_rows, 8192);
            assert_eq!(opt_in.resident_layers, 5);
            assert_eq!(opt_in.host_tensors, 88);
            assert_eq!(opt_in.donors, Some([4, 3]));
        }
        // An explicit smaller value (the 1536 opt-out) is honoured exactly,
        // even on a narrow link.
        let small = xtx_dryrun(
            "gfx1100",
            XTX_FREE_CAND,
            Some("1536"),
            None,
            Discrete { gbps: 7.1 },
            false,
        );
        assert_eq!(small.placement_rows, 1536);
        assert_eq!(small.resident_layers, 10);
    }

    #[test]
    fn xtx_placement_wider_default_chunk_cedes_layers_and_abundant_vram_keeps_all() {
        use crate::host_link::HostLink::Discrete;
        let narrow = Discrete { gbps: 7.1 };
        // The narrow-link placement is sized for the full 8192-row reserve.
        let ceded = xtx_dryrun("gfx1100", XTX_FREE_BASE, None, None, narrow, false);
        let beta = xtx_dryrun(
            "gfx1100",
            XTX_FREE_BASE,
            None,
            None,
            Discrete { gbps: 26.0 },
            false,
        );
        ceded.print("narrow link, base free VRAM");
        assert_eq!(ceded.placement_rows, 8192);
        assert!(ceded.resident_layers < beta.resident_layers);
        assert!(ceded.host_bytes > beta.host_bytes);
        // Abundant VRAM keeps every layer resident at the 8192 reserve.
        let rich = xtx_dryrun("gfx1100", 100_000 << 20, None, None, narrow, false);
        rich.print("abundant VRAM");
        assert_eq!(rich.placement_rows, 8192);
        assert_eq!(rich.resident_layers, 48);
    }

    #[test]
    fn xtx_placement_stage_charge_is_probe_free_and_keeps_explicit_and_gfx1201_semantics() {
        use crate::host_link::HostLink::Discrete;
        // Never reads the link: the placement switch is env-first, then
        // gfx1201, so a narrow-link gfx1100 is not charged a stage.
        assert!(!expert_stage_placement_switch("gfx1100", None));
        assert!(!expert_stage_placement_switch("gfx1151", None));
        assert!(expert_stage_placement_switch("gfx1201", None));
        assert!(expert_stage_placement_switch("gfx1100", Some("1")));
        assert!(!expert_stage_placement_switch("gfx1201", Some("0")));
        assert!(expert_stage_placement_switch("gfx1201", Some("junk")));
        assert!(!expert_stage_placement_switch("gfx1100", Some("on")));
        // The probing default still turns the borrowed stage on afterwards.
        assert!(expert_stage_switch("gfx1100", None, || Discrete {
            gbps: 7.1
        }));
        // A dedicated slab stage is charged only where the placement switch
        // is on: same free VRAM, same layers as the stage-off beta on gfx1100,
        // fewer where the stage is explicit.
        let narrow = Discrete { gbps: 7.1 };
        let default = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, None, narrow, true);
        let off = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, Some("0"), narrow, true);
        let explicit = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, Some("1"), narrow, true);
        assert_eq!(default.resident_layers, off.resident_layers);
        assert!(explicit.resident_layers < default.resident_layers);
        // gfx1151 (unified memory), gfx1201 and gfx942 keep their arch chunk
        // defaults, and the link is never evaluated for them.
        for (arch, rows) in [("gfx1151", 8192), ("gfx1201", 4096), ("gfx942", 1536)] {
            assert_eq!(
                crate::gpu_forward::prefill_chunk_requested_from(arch, None, usize::MAX, || {
                    unreachable!("probed")
                }),
                rows
            );
        }
    }

    #[test]
    fn slowlink_payload_split_is_the_largest_trunk_layer_per_projection() {
        let mut big = entry("model.language_model.layers.1.mlp.experts.gate_up_proj", 1);
        big.logical_shape = vec![8, 4, 256];
        let mut mtp = entry("mtp.layers.0.mlp.experts.gate_up_proj", 0);
        mtp.logical_shape = vec![64, 4, 256];
        let weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.0.mlp.experts.down_proj", 0),
            big,
            entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
            entry("model.language_model.layers.1.self_attn.q_proj.weight", 1),
            mtp,
        ];
        // 2 bytes per element: [4,4,256] = 8192, [8,4,256] = 16384; the MTP
        // layer and non-expert tensors are not a trunk layer's payload.
        assert_eq!(
            routed_layer_payload_split(&weights, element_bytes),
            Ok((16384, 8192))
        );
        assert!(routed_layer_payload_split(&weights[4..5], element_bytes).is_err());
        assert!(routed_layer_payload_split(&[], element_bytes).is_err());
        // A missing payload size is an error, never zero.
        assert!(routed_layer_payload_split(&weights, |_| None).is_err());
    }
}
