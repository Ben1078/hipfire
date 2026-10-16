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
pub fn fits_fully_resident(free_vram: u64, non_expert_bytes: u64, expert_bytes: u64, reserve: u64) -> bool {
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
        && device_archs.iter().all(|arch| hipfire_config::is_discrete_memory_arch(arch))
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
        .map_err(|_| format!("{EXPERT_VRAM_LAYERS_ENV}={value:?} is neither `auto` nor a layer count"))
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
        let bytes = bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
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
    let measured_context = trunk(config.qsa_context_arena_bytes(AUTO_VRAM_RESERVE_MAX_SEQ, qsa_format))
        .ok_or_else(|| overflow(AUTO_VRAM_RESERVE_MAX_SEQ))?;
    // The derived arenas' (BF-1 F16 K/V, pooled BF16 mirror) extra storage
    // must fit through the admitted context, including on UMA VMM loads that
    // otherwise charge only the first mapped chunk. Do not subtract it with
    // the historical (arena-off) measured layout.
    let arena = if qsa_layout.f16_arena || qsa_layout.pooled_bf16 {
        let capacity = crate::Qwen4ContextCommit { tokens: context.max_seq, ..*context };
        let plain = capacity.committed_layer_bytes(config, qsa_format)
            .ok_or_else(|| overflow(context.max_seq))?;
        let with = capacity.committed_layer_bytes(config, qsa_layout)
            .ok_or_else(|| overflow(context.max_seq))?;
        trunk(Some(with.checked_sub(plain).ok_or_else(|| overflow(context.max_seq))?))
            .ok_or_else(|| overflow(context.max_seq))?
    } else {
        0
    };
    let context = committed.saturating_sub(measured_context)
        .checked_add(arena).ok_or_else(|| overflow(context.max_seq))?;
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
    let forward =
        crate::gpu_forward::qwen4_forward_device_bytes(config, chunk_rows, mode).ok_or_else(overflow)?;
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
    Ok(StateCensus { qsa_context, state, forward })
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
    scope.validate(config)?;
    let weights_census = band_weight_census(weights, &bytes_of, scope)?;
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
const F16_SHADOW_ROOT: &str = "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight";

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
/// [`ReserveCensus`]. The whole-layer stage buffers and safety margin
/// ([`PlacementCharges`]) are added on top of [`Self::bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoReserve {
    pub historical: u64,
    pub census: ReserveCensus,
    pub census_total: u64,
}

impl AutoReserve {
    pub fn new(historical: u64, census: ReserveCensus) -> Result<Self, String> {
        let census_total = census.total()?;
        Ok(Self { historical, census, census_total })
    }

    /// The bytes `auto` reserves: `max(historical, census)`.
    pub fn bytes(&self) -> u64 {
        self.historical.max(self.census_total)
    }

    /// Which formula set [`Self::bytes`].
    pub fn winner(&self) -> &'static str {
        if self.census_total > self.historical {
            "census"
        } else {
            "historical floor"
        }
    }

    /// The reserve and its parts for the placement log, every census term.
    pub fn describe(&self) -> String {
        format!(
            "{} MiB = max({} MiB historical floor, {} MiB census [{}]): {} wins",
            self.bytes() >> 20,
            self.historical >> 20,
            self.census_total >> 20,
            self.census.describe(),
            self.winner()
        )
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
    if non_expert_bytes.checked_add(reserve).is_none_or(|need| free_vram < need) {
        return Err(format!(
            "{} MiB of free VRAM cannot hold the {} MiB of non-expert weights plus the {} MiB reserve ({} MiB base + {}) even with every routed expert in host RAM; free VRAM on this GPU",
            free_vram / MIB,
            non_expert_bytes / MIB,
            reserve / MIB,
            base_reserve / MIB,
            charges.describe()
        ));
    }
    Ok(auto_vram_layers(free_vram, non_expert_bytes, layer_expert_bytes, num_layers, reserve))
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

/// Whether a load on `arch` builds the long-prefill expert stage (the same
/// switch `Qwen4ExpertStage::new` reads), so its buffers need a budget.
pub fn expert_stage_enabled(arch: &str) -> bool {
    hipfire_config::developer_bool(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV, arch == "gfx1201")
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
    largest.iter().map(|bytes| bytes + crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64).sum()
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
///
/// The hot set path does not use this: it subtracts its own margin and
/// tier-plan stage ([`expert_hotset_budget`], [`expert_tier_stage_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementCharges {
    pub margin: u64,
    pub stage: u64,
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
        Ok(Self { margin: EXPERT_HOTSET_VRAM_MARGIN_BYTES, stage })
    }

    /// `base` ([`auto_vram_reserve`]) plus both charges.
    pub fn reserve(&self, base: u64) -> Result<u64, String> {
        base.checked_add(self.margin)
            .and_then(|bytes| bytes.checked_add(self.stage))
            .ok_or_else(|| "auto expert VRAM reserve with placement charges overflows".to_string())
    }

    /// The charges for the placement log, in MiB.
    pub fn describe(&self) -> String {
        const MIB: u64 = 1 << 20;
        format!(
            "{} MiB safety margin + {} MiB long-prefill stage buffers",
            self.margin / MIB,
            self.stage / MIB
        )
    }
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
        let trunk_resident = !entry.name.starts_with("mtp.")
            && entry.layer.is_some_and(|layer| layer < vram_layers);
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
        return Err(format!("expert-parallel rank {rank} is outside {ranks} ranks"));
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
    Some(GttBudget { limit_bytes, used_bytes: amdgpu_gtt_used_bytes()? })
}

/// Refuse before any allocation when GTT-backed host-mapped experts would not
/// fit under TTM's `pages_limit` beside what amdgpu devices already hold.
/// Past it `hipHostMalloc` fails, after the load has uploaded the VRAM
/// weights. `None` (userptr, or no TTM limit in sysfs) skips the check.
pub fn check_gtt_cap(host_bytes: u64, budget: Option<GttBudget>) -> Result<(), String> {
    let Some(GttBudget { limit_bytes, used_bytes }) = budget else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_forward::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use rdna_compute::tensor_ops::QsaKvFormat::{self, F32};
    use rdna_compute::DType;
    use rdna_compute::uma_memory::ttm_pool_estimate_from;

    fn entry(name: &str, layer: usize) -> WeightEntry {
        let mut entry =
            WeightEntry::model(name, vec![4, 4, 256], DType::MQ4G256V2, ShardPolicy::Replicate);
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
        assert!(weights[..4].iter().all(|entry| is_routed_expert(&entry.name)));
        let (non_expert, layer_experts) =
            resident_split(&weights, |entry| Some(entry.logical_shape.iter().product::<usize>() as u64 * 2))
                .expect("split");
        assert_eq!(layer_experts, 700 * 2, "the charge is the max arena, never layer 0's");
        assert_eq!(non_expert, 4 * 4 * 256 * 2);
        // Admission totals every arena's physical bytes.
        assert_eq!(
            routed_expert_bytes(&weights, |entry| Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)),
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
            entry("model.language_model.layers.1.mlp.shared_expert.down_proj.weight", 1),
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
            entry("model.language_model.layers.1.mlp.shared_expert.down_proj.weight", 1),
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
                !(is_routed_expert(&entry.name) && !entry.name.starts_with("mtp.") && entry.layer.is_some())
            });
            assert_eq!(got, host_mapped_bytes(&placed, bytes_of).unwrap(), "vram_layers={vram_layers}");
        }
        // MTP experts and the preexisting host entry only; the input never mutates.
        let expected = (weights[3].name.len() + weights[4].name.len() + weights[5].name.len()) as u64
            * 1000
            + 3 * HOST_MAPPED_PAD_BYTES;
        assert_eq!(hotset_host_mapped_bytes(&weights, bytes_of, 3), Ok(expected));
        assert_eq!(before, weights.iter().map(|entry| entry.residency).collect::<Vec<_>>());
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
        assert_eq!(expert_hotset_budget(exact + 1, non_expert, reserve).unwrap(), 1);
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
            f16_arena: true, ..rdna_compute::tensor_ops::QsaStateLayout::F32
        };
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let context = crate::Qwen4ContextCommit::new(backend, 262145, 8192, 2 << 20);
            let plain = auto_vram_reserve(&config, &context, 8192, Full, F32, None, None).unwrap();
            let with = auto_vram_reserve(&config, &context, 8192, Full, layout, None, None).unwrap();
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
            pooled_bf16: true, ..rdna_compute::tensor_ops::QsaStateLayout::F32
        };
        let blocks = 262145usize.div_ceil(4);
        let block_bytes = layout.pooled_bf16_bytes_per_block(128);
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let context = crate::Qwen4ContextCommit::new(backend, 262145, 8192, 2 << 20);
            let plain = auto_vram_reserve(&config, &context, 8192, Full, F32, None, None).unwrap();
            let with = auto_vram_reserve(&config, &context, 8192, Full, layout, None, None).unwrap();
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
        let reserve = reserve(crate::gpu_forward::qwen4_prefill_chunk_default("gfx1201"));
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
            weights.push(bf16(&layer(i, "attn_hyper_connection.input_mix_weight_down.weight"), &[320, 10240]));
            weights.push(bf16(&layer(i, "mlp_hyper_connection.input_mix_weight_down.weight"), &[320, 10240]));
            weights.push(bf16(&layer(i, "mlp.gate.weight"), &[512, 2560]));
            weights.push(bf16(&layer(i, "mlp.shared_expert.gate_proj.weight"), &[640, 2560]));
            weights.push(bf16(&layer(i, "mlp.shared_expert.up_proj.weight"), &[640, 2560]));
            weights.push(bf16(&layer(i, "mlp.shared_expert.down_proj.weight"), &[2560, 640]));
        }
        weights.push(bf16(&layer(2, "ple.key_proj.weight"), &[6400, 2560]));
        weights.push(bf16(&layer(2, "ple.value_proj.weight"), &[6400, 2560]));
        weights
    }

    /// The full-context census of the target profile on a VMM discrete card
    /// at `max_seq` (native MTP off), every term from its owning query.
    fn real_census(config: &Qwen4Config, max_seq: usize) -> ReserveCensus {
        let format = target_state_format();
        let chunk = crate::gpu_forward::qwen4_prefill_chunk_default("gfx1201");
        let context =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, max_seq, max_seq, 2 << 20);
        let state = state_census(config, format, &context, chunk, Full).unwrap();
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
        ReserveCensus {
            qsa_context: state.qsa_context,
            state: state.state,
            forward: state.forward,
            f16_shadows: f16_shadow_bytes(&target_f16_manifest(), config.num_hidden_layers).unwrap(),
            workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims).unwrap(),
            qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_capacity_bytes() as u64,
            mtp: 0,
            gather: rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                max_seq,
            )
            .unwrap() as u64,
        }
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
        let reserve = AutoReserve::new(historical, census).unwrap();
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
        assert_eq!(census.workspace.iter().map(|term| term.1).sum::<u64>(), 153_261_832);
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
        for name in ["QSA context", "state", "forward", "F16 weight shadows", "QSA selector", "native MTP", "QSA gather", "fp16_x", "ksplit_det", "gdn_pair_counters"] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(charges.describe().contains("512 MiB safety margin"));
        assert!(charges.describe().contains("2120 MiB long-prefill stage buffers"));
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
        let first = crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, 17_408, 1536, 2 << 20);
        let first = state_census(&config, target_state_format(), &first, 4096, Full).unwrap();
        assert!(census.qsa_context > first.qsa_context);
        let (unfixed, floor_charged, placed, reserve, _) =
            replay(free, non_expert, layer, historical, census);
        assert_eq!(unfixed, 23);
        assert!(floor_charged < 23);
        assert!(placed <= floor_charged && placed < 23);
        assert!(reserve.describe().contains("QSA context"));
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
            crate::Qwen4KvBackend::Vmm, true, max_seq, chunk, granularity,
        );
        let full =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, max_seq, max_seq, granularity);
        assert!(full.committed_layer_bytes(&config, format.layout())
            > first.committed_layer_bytes(&config, format.layout()));
        let gather = |tokens: usize| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(config.num_key_value_heads, tokens)
                .unwrap() as u64
        };
        let historical = auto_vram_reserve(
            &config, &first, chunk, Full, format.layout(), None, Some(gather(first.tokens)),
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
        };
        let reserve = AutoReserve::new(historical, census).unwrap();
        // Both Halo artifacts fit fully resident under whichever formula wins
        // (exact non-expert bytes 5,364,055,296 = 5,115.6 MiB).
        assert_eq!(reserve.census_total >> 20, 9854, "{}", reserve.describe());
        for (free, routed) in [(118_441u64 << 20, 62_475u64 << 20), (118_415 << 20, 51_375 << 20)] {
            assert!(fits_fully_resident(free, 5_364_055_296, routed, reserve.bytes()), "{}", reserve.describe());
        }
        // The census decision: the historical floor wins on this admission
        // (no flip), so unified-memory behaviour is unchanged. A failure here
        // is the flip: unified memory must then admit on `historical`.
        assert_eq!(reserve.winner(), "historical floor", "{}", reserve.describe());
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
        let reserve = AutoReserve::new(historical, census).unwrap();
        assert_eq!(reserve.census_total, total);
        assert!(total > historical);
        assert_eq!(reserve.bytes(), total);
        assert_eq!(reserve.winner(), "census");
        let text = reserve.describe();
        for name in ["QSA context", "state", "forward", "F16 weight shadows", "workspace", "QSA selector", "native MTP", "QSA gather"] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(text.contains("census wins") && text.contains("7927 MiB historical floor"), "{text}");
        // The census, not the floor, sizes the placement.
        let by_floor = auto_vram_layers(free, non_expert, layer, 49, historical);
        let by_census = auto_vram_layers(free, non_expert, layer, 49, reserve.bytes());
        assert!(by_census < by_floor);
        let overflowing = ReserveCensus {
            gather: 1,
            qsa_context: u64::MAX,
            ..census_with(0, 0, 0, 0, 0)
        };
        assert!(AutoReserve::new(0, overflowing).is_err());
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
                let context = full.committed_layer_bytes(&config, format.layout()).unwrap()
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
            bf16("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", &[4, 8]),
            bf16(&layer(0, "attn_hyper_connection.input_mix_weight_down.weight"), &[2, 2]),
            bf16(&layer(1, "mlp_hyper_connection.input_mix_weight_down.weight"), &[2, 2]),
            bf16(&layer(0, "mlp.gate.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.gate_proj.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.up_proj.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.down_proj.weight"), &[5, 3]),
            bf16(&layer(0, "ple.key_proj.weight"), &[1, 7]),
            bf16(&layer(0, "ple.value_proj.weight"), &[1, 7]),
            // Direct BF16 readers and other tensors: no shadow.
            bf16(&layer(0, "attn_hyper_connection.input_mix_weight_up.weight"), &[2, 2]),
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
    fn stage_charge_only_where_the_slab_stage_allocates() {
        // Needs the switch, a slab recipe and no expert parallelism.
        assert!(dedicated_slab_stage(true, true, false));
        assert!(!dedicated_slab_stage(false, true, false));
        assert!(!dedicated_slab_stage(true, false, false));
        assert!(!dedicated_slab_stage(true, true, true));
        let (free, non_expert, layer, base) = EVIDENCE;
        let off = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(off, PlacementCharges { margin: 512 << 20, stage: 0 });
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
        assert!(!fits_fully_resident(need, non_expert, experts, off.reserve(base).unwrap()));
    }

    #[test]
    fn charged_placement_refuses_clearly_when_zero_layers_do_not_fit() {
        let (_, non_expert, layer, base) = EVIDENCE;
        let charges = PlacementCharges::new(layer, true).unwrap();
        let need = non_expert + charges.reserve(base).unwrap();
        // Exactly the weights and the charged reserve: zero resident layers.
        assert_eq!(auto_placement_layers(need, non_expert, layer, 49, base, &charges), Ok(0));
        let error =
            auto_placement_layers(need - 1, non_expert, layer, 49, base, &charges).unwrap_err();
        assert!(error.contains("cannot hold") && error.contains("every routed expert in host RAM"), "{error}");
        assert!(error.contains("512 MiB safety margin"), "{error}");
        assert!(error.contains("2120 MiB long-prefill stage buffers"), "{error}");
        // Without the stage the same card still loads: the 2120 MiB + 8191 B
        // the stage took now holds two whole layers.
        let off = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(auto_placement_layers(need - 1, non_expert, layer, 49, base, &off), Ok(2));
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
            assert_eq!(reserve(rows, Full) - reserve(rows, FnSymGfx1201), delta, "{rows} rows");
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
        let base =
            auto_vram_reserve(&config, &measured, AUTO_VRAM_RESERVE_CHUNK, Full, F32, None, None)
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
        let mtp = native(&measured, 3, Some(crate::GdnStateFormat::Q8));
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
        let verify = (AUTO_VRAM_RESERVE_CHUNK as u64 + 2) * hidden_row;
        assert!(verify > scratch);
        let capture4 =
            crate::state::Qwen4State::row_capture_bytes(&config, crate::GdnStateFormat::Q8, 4).unwrap()
                as u64;
        assert_eq!(mtp, resident + verify + capture4);
        assert_eq!(native(&measured, 3, None), resident + verify);
        assert_eq!(native(&measured, 10, None), resident + verify);
        // The 11-row GDN capture is charged in the state's real GDN format.
        for gdn in [crate::GdnStateFormat::Q8, crate::GdnStateFormat::F32] {
            let capture =
                crate::state::Qwen4State::row_capture_bytes(&config, gdn, 11).unwrap() as u64;
            let request = (AUTO_VRAM_RESERVE_CHUNK as u64 + 2) * hidden_row + capture;
            assert_eq!(native(&measured, 10, Some(gdn)), resident + scratch.max(request));
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
                Some(gdn),
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
                let discrete = Commit::for_expert_reserve(Backend::Vmm, false, max_seq, chunk, granularity);
                assert_eq!(discrete, legacy);
                assert_eq!(placement(&discrete, format), placement(&legacy, format));
                assert_eq!(
                    Commit::for_expert_reserve(Backend::Legacy, false, max_seq, chunk, granularity),
                    legacy
                );
                // Unified memory is unchanged: VMM charges the first chunk's
                // pages, legacy all of `max_seq`.
                let vmm = Commit::new(Backend::Vmm, max_seq, chunk, granularity);
                let uma_vmm = Commit::for_expert_reserve(Backend::Vmm, true, max_seq, chunk, granularity);
                assert_eq!(uma_vmm, vmm);
                assert_eq!(placement(&uma_vmm, format), placement(&vmm, format));
                assert!(placement(&uma_vmm, format).0 < placement(&legacy, format).0);
                assert!(placement(&uma_vmm, format).1 >= placement(&legacy, format).1);
                assert_eq!(
                    Commit::for_expert_reserve(Backend::Legacy, true, max_seq, chunk, granularity),
                    legacy
                );
            }
            assert!(mtp(&Commit::for_expert_reserve(Backend::Vmm, true, max_seq, chunk, granularity)) < mtp(&legacy));
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
        assert_eq!(expert_stage_mirror_bytes(full, 1).unwrap(), 2 * full as u64 + 4 * HOST_MAPPED_PAD_BYTES);
        assert_eq!(expert_stage_mirror_bytes(full / 2, 2).unwrap(), 2 * full as u64 + 8 * HOST_MAPPED_PAD_BYTES);
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
        let budget = |limit_bytes| Some(GttBudget { limit_bytes, used_bytes: 0 });
        assert!(check_gtt_cap(charge, budget(charge)).is_ok());
        assert!(check_gtt_cap(charge, budget(charge - 1)).is_err());
        assert!(check_gtt_cap(charge / 4, budget(charge - 1)).is_ok());
    }

    /// `/proc/meminfo` text from `(field, MiB)` pairs.
    fn meminfo(fields: &[(&str, u64)]) -> String {
        fields.iter().map(|(key, mib)| format!("{key}:{:>16} kB\n", mib << 10)).collect()
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
        let fresh = GttBudget { limit_bytes, used_bytes: 16 << 20 };
        assert!(check_gtt_cap(host, Some(fresh)).is_ok());
        assert!(check_gtt_cap(limit_bytes - fresh.used_bytes, Some(fresh)).is_ok());
        let error = check_gtt_cap(limit_bytes - fresh.used_bytes + 1, Some(fresh)).unwrap_err();
        assert!(error.contains("pages_limit") && error.contains("16108144 pages"), "{error}");
        // A second load beside one already holding 46 GiB of GTT is refused.
        let beside = GttBudget { limit_bytes, used_bytes: 46 << 30 };
        assert!(check_gtt_cap(host, Some(beside)).is_err());
        // Userptr host memory, or no TTM limit in sysfs, is not GTT-capped.
        assert!(check_gtt_cap(u64::MAX, None).is_ok());
    }

    #[test]
    fn parse_accepts_auto_and_counts_only() {
        assert_eq!(parse_expert_vram_layers(" auto "), Ok(ExpertVramLayers::Auto));
        assert_eq!(parse_expert_vram_layers("16"), Ok(ExpertVramLayers::Layers(16)));
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
        assert_eq!(resolve_expert_vram_layers(None, || Ok(false)), Ok(Some(ExpertVramLayers::Auto)));
        assert!(resolve_expert_vram_layers(None, || Err("vram".into())).is_err());
        // Explicit values never consult the fit.
        for policy in [ExpertVramLayers::Layers(12), ExpertVramLayers::Layers(99), ExpertVramLayers::Auto] {
            let resolved = resolve_expert_vram_layers(Some(policy), || panic!("explicit consulted the fit"));
            assert_eq!(resolved, Ok(Some(policy)));
        }
    }

    #[test]
    fn only_qwen4_on_discrete_gpus_keeps_host_memory_out_of_reclaim() {
        assert!(keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1201"]));
        assert!(keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1100", "gfx942"]));
        // Another model on the same card (arch 5: the Qwen3.5-family trunk).
        assert!(!keeps_host_memory_out_of_reclaim(5, &["gfx1201"]));
        // Unified memory, alone or beside a discrete card ROCr also exposes.
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1151"]));
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx1100", "gfx1151"]));
        // No card known, or one whose memory kind is not known.
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &[]));
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &["gfx90a"]));
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
            let charges =
                host_mapped_bytes_ranks(&manifest.weights, element_bytes, ranks).unwrap();
            assert_eq!(charges.len(), ranks);
            for (rank, &charge) in charges.iter().enumerate() {
                let local = Qwen4EpPlacement { rank, ranks }.local_count(512) as u64;
                assert_eq!(charge, (single - pad) / 512 * local + pad, "ranks={ranks} rank={rank}");
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
            ep_vram_layers(&config, &[free; 8][..ranks], non_expert, per_expert * max, reserve)
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
        assert!(lines[0].contains("4 ranks") && lines[0].contains("32768") && lines[0].contains("1536"));
        for (rank, line) in lines[1..].iter().enumerate() {
            assert!(line.starts_with(&format!("qwen4 EP rank {rank}/4:")), "{line}");
            for part in ["free", "trunk", "experts", "resident expert layers", "reserve", "headroom"] {
                assert!(line.contains(part), "{part} missing from {line}");
            }
            // 49 MoE layers incl. the MTP head.
            assert!(line.contains("of 49"), "{line}");
        }
        // EP4 at 28..30 GiB free keeps all 49 MoE layers (the MTP layer too) resident.
        assert!(lines[0].contains("VRAM split 49 of 49"), "{}", lines[0]);
        assert!(lines[1].contains("resident expert layers 49 of 49 (0 host-mapped"), "{}", lines[1]);
        // Headroom: 30 GiB - 5.99 GB trunk - 49 x 0.334 GB - 6.5 GiB reserve > 0 on rank 0.
        assert!(lines[1].contains("headroom +"), "{}", lines[1]);
        // A rank with too little VRAM for the reserve reports negative headroom.
        let tight =
            ep_fit_rank_line(&config, 1, 2, 8 << 30, 5_990_000_000, 668_450_000, 6_656u64 << 20, 0);
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
        assert_eq!(ep_vram_layers(&config, &free, non_expert, layer, reserve), tight);
        assert_eq!(ep_vram_layers(&config, &free[..1], non_expert, layer, reserve), roomy);
        assert_eq!(ep_vram_layers(&config, &[], non_expert, layer, reserve), 0);
        // The report applies the minimum to every rank.
        let report = ep_fit_report(&config, 4, &free, non_expert, layer, reserve, 32768, 1536);
        let lines: Vec<_> = report.lines().collect();
        assert!(lines[0].contains(&format!("VRAM split {tight} of 49")), "{}", lines[0]);
        for line in &lines[1..] {
            assert!(line.contains(&format!("resident expert layers {tight} of 49")), "{line}");
        }
        assert!(lines[2].contains(&format!("this rank alone fits {tight}")), "{}", lines[2]);
        assert!(lines[1].contains(&format!("this rank alone fits {roomy}")), "{}", lines[1]);
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
}
