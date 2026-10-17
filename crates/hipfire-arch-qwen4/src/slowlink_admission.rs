// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission, placement census and prompt-arena layout of the S1 slow-link
//! layer-major prefill.
//!
//! S1 streams every host-resident routed expert across a narrow PCIe link once
//! per prompt window, layer by layer, instead of once per prefill chunk. This
//! module is the pure policy half: it decides *before any allocation* whether
//! a load may take the S1 profile, sizes everything that profile costs from
//! one layout, and judges request shapes against the frozen profile. It never
//! touches a GPU, so every number here is exercised by CPU tests.
//!
//! # Selector truth table
//!
//! `HIPFIRE_QWEN4_SLOWLINK_PREFILL` (`developer.qwen4_slowlink_prefill`, a
//! typed boolean): `0` incumbent (kill switch), `1` lab candidate, unset
//! automatic. Automatic is gated by [`S1_AUTO_QUALIFIED`], which stays `false`
//! until every promotion gate has passed, so an unset selector is the incumbent
//! today. The candidate (forced or, later, automatic) is selected only when
//! all of these hold, checked in this order, cheapest first:
//!
//! 1. exact `gfx1100`, a discrete device ([`preflight`]);
//! 2. the qualified MQ4 packed recipe on a single-device, whole-model,
//!    auto-placed load: no expert parallelism, bands, hot set, `whole-first`,
//!    explicit expert layer count, `memory.offload_exec` splice, expert-keep
//!    artifact, or `HIPFIRE_QWEN4_EXPERT_STAGE=0` opt-out;
//! 3. a context long enough to hold a multi-chunk window;
//! 4. the measured host link ([`link_verdict`]): discrete and **below**
//!    16 GB/s. A fast link, unified memory and an unmeasured link never select
//!    S1, whatever the selector says;
//! 5. the joint placement ([`admit`]): the 1,536-row tile, the prompt arena,
//!    the expert page store, the panel scratch and the boundary state are all
//!    charged together, and the placement that remains must still keep
//!    [`S1_MIN_RESIDENT_LAYERS`] trunk layers resident. A shortfall is refused
//!    with its exact byte deficit ([`Deficit`]); fewer residents are never
//!    accepted.
//!
//! Steps 1 to 3 are probe-free; only a load that passes them measures the host
//! link, so no other arch and no non-candidate load pays for the probe, and a
//! load that fails any step is the incumbent byte for byte with a zero S1
//! charge ([`crate::expert_residency::PlacementCharges::slowlink`] is `0`).

use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::host_link::{HostLink, NARROW_LINK_GBPS};
use crate::Qwen4Config;
use std::fmt;

/// Developer variable of the selector (`developer.qwen4_slowlink_prefill`).
pub const SLOWLINK_PREFILL_ENV: &str = "HIPFIRE_QWEN4_SLOWLINK_PREFILL";

/// Rows of the S1 mixer tile: the chunk the load sizes its forward scratch
/// from and the largest chunk the forward ever runs. Never the 8192-row
/// incumbent chunk.
pub const S1_TILE_ROWS: usize = 1536;

/// Prompt rows one S1 window admits. A longer prompt runs several windows
/// (each its own expert sweep) and is receipted as such; it never claims one
/// sweep per prompt.
pub const S1_WINDOW_ROWS: usize = 16384;

/// Fewest request rows S1 runs when forced: at least two tiles, so the
/// request is genuinely multi-chunk.
pub const S1_MIN_FORCED_REQUEST_ROWS: usize = 2 * S1_TILE_ROWS;

/// Fewest request rows the automatic selector sends to S1: where the
/// performance qualification starts. The 1,400 tok/s target is not
/// generalised to shorter prompts.
pub const S1_MIN_AUTO_REQUEST_ROWS: usize = 16384;

/// Trunk layers an S1 load must keep resident beside its own charges.
pub const S1_MIN_RESIDENT_LAYERS: usize = 10;

/// Whether an unset selector may choose S1. `false` until every promotion
/// gate has passed; the kill switch and the lab force work either way.
pub const S1_AUTO_QUALIFIED: bool = false;

/// Routed experts one transport page holds.
pub const EXPERTS_PER_PAGE: usize = 32;

/// Dedicated double-buffered active pages (the ring).
pub const ACTIVE_PAGES: usize = 2;

/// Grouped rows one expert panel computes at most (`PanelLimits`).
pub const PANEL_MAX_GROUPED_ROWS: usize = 2048;

/// Boundary snapshots of the fixed per-layer state one window may hold beside
/// the entry snapshot. A request with more session cuts is a bypass.
pub const S1_MAX_BOUNDARY_SNAPSHOTS: usize = 4;

/// Byte alignment of every arena span and page region.
pub const S1_ALIGN: u64 = 256;

fn align(bytes: u64) -> Result<u64, String> {
    bytes
        .checked_add(S1_ALIGN - 1)
        .map(|bytes| bytes / S1_ALIGN * S1_ALIGN)
        .ok_or_else(|| "slow-link prefill byte alignment overflows".to_string())
}

fn checked_mul(a: u64, b: u64, what: &str) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("slow-link prefill {what} overflows"))
}

fn checked_add(a: u64, b: u64, what: &str) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("slow-link prefill {what} overflows"))
}

/// The selector value: `0` is [`Self::Off`], `1` is [`Self::Force`], unset is
/// [`Self::Auto`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlowlinkMode {
    Off,
    Force,
    Auto,
}

impl SlowlinkMode {
    /// From the typed-boolean selector value (`None` when unset).
    pub fn from_typed(value: Option<bool>) -> Self {
        match value {
            Some(false) => Self::Off,
            Some(true) => Self::Force,
            None => Self::Auto,
        }
    }

    /// The process-snapshot selector (`developer.qwen4_slowlink_prefill`).
    pub fn from_env() -> Self {
        Self::from_typed(hipfire_config::developer_typed_bool(SLOWLINK_PREFILL_ENV))
    }
}

/// What the loader knows about a load before it sizes any placement.
#[derive(Clone, Copy, Debug)]
pub struct LoadShape<'a> {
    pub arch: &'a str,
    pub unified_memory: bool,
    /// Expert-parallel mesh, or an agreed expert-parallel layer split.
    pub expert_parallel: bool,
    pub pipeline_band: bool,
    /// The qualified MQ4 packed recipe (`FlashNextPacked`).
    pub packed_mq4: bool,
    pub routed_experts: bool,
    pub expert_keep: bool,
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` set explicitly.
    pub explicit_vram_layers: bool,
    /// A ranked hot-set sidecar applies.
    pub hotset_sidecar: bool,
    pub whole_first: bool,
    /// `HIPFIRE_QWEN4_EXPERT_STAGE=0`: the staged-transport opt-out.
    pub stage_opt_out: bool,
    /// `memory.offload_exec` is not `pcie` (CPU splice or pass-back).
    pub cpu_offload: bool,
    /// Raw `HIPFIRE_PREFILL_CHUNK_ROWS`, when set.
    pub chunk_rows_env: Option<&'a str>,
    pub max_seq: usize,
}

/// Exact byte shortfall of a joint placement that cannot keep
/// [`S1_MIN_RESIDENT_LAYERS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deficit {
    /// Layers the charged placement keeps resident.
    pub resident_layers: usize,
    pub required_layers: usize,
    pub free: u64,
    pub non_expert: u64,
    pub layer_expert_bytes: u64,
    /// The complete charged reserve (base + margin + S1 census).
    pub reserve: u64,
    /// `non_expert + reserve + required_layers * layer` beyond `free`.
    pub deficit_bytes: u64,
}

/// Why a load or request does not take S1. Every variant is an incumbent
/// route chosen before any S1 allocation or state mutation.
#[derive(Clone, Debug, PartialEq)]
pub enum SlowlinkDecline {
    /// `HIPFIRE_QWEN4_SLOWLINK_PREFILL=0`.
    KillSwitch,
    /// Unset selector while [`S1_AUTO_QUALIFIED`] is `false`.
    AutoNotQualified,
    UnqualifiedArch(String),
    UnifiedMemory,
    Unsupported(&'static str),
    /// `HIPFIRE_PREFILL_CHUNK_ROWS` asks for a chunk other than the S1 tile.
    ExplicitChunk { rows: usize },
    ShortContext { max_seq: usize },
    /// Measured discrete link at or above 16 GB/s.
    FastLink { gbps: f64 },
    /// The link probe failed or returned an unusable figure.
    UnmeasuredLink,
    /// Every routed expert already fits: there is no host layer to stream.
    NoHostLayers { resident_layers: usize },
    Deficit(Deficit),
    Geometry(String),
}

impl SlowlinkDecline {
    /// A capacity refusal of an otherwise eligible load (the forced selector
    /// fails the load on it instead of silently running the incumbent).
    pub fn is_capacity(&self) -> bool {
        matches!(self, Self::Deficit(_) | Self::Geometry(_))
    }
}

impl fmt::Display for SlowlinkDecline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const MIB: u64 = 1 << 20;
        match self {
            Self::KillSwitch => write!(f, "{SLOWLINK_PREFILL_ENV}=0"),
            Self::AutoNotQualified => {
                write!(f, "automatic selection is not qualified; unset is the incumbent")
            }
            Self::UnqualifiedArch(arch) => write!(f, "{arch} is not gfx1100"),
            Self::UnifiedMemory => write!(f, "unified memory has no link to stream over"),
            Self::Unsupported(what) => write!(f, "unsupported mode: {what}"),
            Self::ExplicitChunk { rows } => write!(
                f,
                "HIPFIRE_PREFILL_CHUNK_ROWS={rows} conflicts with the fixed {S1_TILE_ROWS}-row tile"
            ),
            Self::ShortContext { max_seq } => write!(
                f,
                "context {max_seq} is shorter than the {S1_MIN_FORCED_REQUEST_ROWS}-row minimum multi-chunk window"
            ),
            Self::FastLink { gbps } => write!(
                f,
                "host link {gbps:.1} GB/s is not narrow (below {NARROW_LINK_GBPS} GB/s)"
            ),
            Self::UnmeasuredLink => write!(f, "host link is unmeasured (never selects S1)"),
            Self::NoHostLayers { resident_layers } => write!(
                f,
                "{resident_layers} resident layers hold every routed expert: nothing streams"
            ),
            Self::Deficit(d) => write!(
                f,
                "joint placement keeps {} of the required {} resident layers: {} MiB free, \
                 {} MiB non-expert, {} MiB reserve, {} MiB per layer; short by {} bytes ({} MiB)",
                d.resident_layers,
                d.required_layers,
                d.free / MIB,
                d.non_expert / MIB,
                d.reserve / MIB,
                d.layer_expert_bytes / MIB,
                d.deficit_bytes,
                d.deficit_bytes / MIB
            ),
            Self::Geometry(error) => write!(f, "{error}"),
        }
    }
}

/// The explicit `HIPFIRE_PREFILL_CHUNK_ROWS` value rounded down to 256-row
/// tiles, `None` when unset, unparsable or below one tile (the same reading
/// the chunk policy uses).
fn explicit_chunk_rows(env: Option<&str>) -> Option<usize> {
    env.and_then(|value| value.trim().parse::<usize>().ok())
        .map(|rows| rows / 256 * 256)
        .filter(|&rows| rows >= 256)
}

/// Probe-free load checks: everything that can refuse S1 without measuring
/// the host link. Order is part of the contract (cheapest and most absolute
/// first), so an automatic or killed selector never reaches the rest.
pub fn preflight(mode: SlowlinkMode, shape: &LoadShape<'_>) -> Result<(), SlowlinkDecline> {
    match mode {
        SlowlinkMode::Off => return Err(SlowlinkDecline::KillSwitch),
        SlowlinkMode::Auto if !S1_AUTO_QUALIFIED => {
            return Err(SlowlinkDecline::AutoNotQualified)
        }
        SlowlinkMode::Force | SlowlinkMode::Auto => {}
    }
    if shape.arch != "gfx1100" {
        return Err(SlowlinkDecline::UnqualifiedArch(shape.arch.to_string()));
    }
    if shape.unified_memory {
        return Err(SlowlinkDecline::UnifiedMemory);
    }
    let unsupported = [
        (shape.expert_parallel, "expert parallelism"),
        (shape.pipeline_band, "pipeline parallelism"),
        (!shape.packed_mq4, "recipe is not the qualified MQ4 packed layout"),
        (!shape.routed_experts, "no routed experts"),
        (shape.expert_keep, "expert-keep artifact"),
        (shape.explicit_vram_layers, "explicit expert VRAM layer count"),
        (shape.hotset_sidecar, "ranked expert hot set"),
        (shape.whole_first, "whole-first placement policy"),
        (shape.stage_opt_out, "HIPFIRE_QWEN4_EXPERT_STAGE=0 opt-out"),
        (shape.cpu_offload, "memory.offload_exec splice"),
    ];
    if let Some((_, what)) = unsupported.into_iter().find(|(hit, _)| *hit) {
        return Err(SlowlinkDecline::Unsupported(what));
    }
    if let Some(rows) = explicit_chunk_rows(shape.chunk_rows_env) {
        if rows != S1_TILE_ROWS {
            return Err(SlowlinkDecline::ExplicitChunk { rows });
        }
    }
    if shape.max_seq < S1_MIN_FORCED_REQUEST_ROWS {
        return Err(SlowlinkDecline::ShortContext {
            max_seq: shape.max_seq,
        });
    }
    Ok(())
}

/// The measured link verdict: the link's GB/s when it is discrete and narrow,
/// else the decline. Fast links, unified memory and an unmeasured link never
/// select S1.
pub fn link_verdict(link: HostLink) -> Result<f64, SlowlinkDecline> {
    match link {
        HostLink::Discrete { gbps } if link.is_narrow_discrete() => Ok(gbps),
        HostLink::Discrete { gbps } => Err(SlowlinkDecline::FastLink { gbps }),
        HostLink::Unified => Err(SlowlinkDecline::UnifiedMemory),
        HostLink::Unmeasured => Err(SlowlinkDecline::UnmeasuredLink),
    }
}

/// The model dimensions the S1 layouts are sized from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelGeometry {
    pub hidden: usize,
    pub hc_count: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate: usize,
}

impl ModelGeometry {
    /// A MoE profile's dimensions; a dense profile has nothing to stream.
    pub fn from_config(config: &Qwen4Config) -> Result<Self, String> {
        match crate::gpu_forward::program_dims(config) {
            dims @ crate::program::Qwen4ProgramDims {
                ffn:
                    crate::program::Qwen4FfnDims::Moe {
                        num_experts,
                        experts_per_token,
                        intermediate,
                        ..
                    },
                ..
            } => Ok(Self {
                hidden: dims.hidden,
                hc_count: dims.hc_count,
                num_experts,
                experts_per_token,
                moe_intermediate: intermediate,
            }),
            _ => Err("slow-link prefill needs a routed-expert (MoE) profile".to_string()),
        }
    }

    fn wide(&self) -> Result<u64, String> {
        checked_mul(self.hc_count as u64, self.hidden as u64, "hyper-connection width")
    }
}

/// One transport page: the payload, tail pads and rebased pointer tables of
/// [`EXPERTS_PER_PAGE`] consecutive experts. Offsets are within the page; the
/// transport allocates exactly [`Self::page_bytes`] per page and the census
/// charges the same number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelPageLayout {
    pub experts_per_page: usize,
    pub num_experts: usize,
    /// Pages of one full layer (the lookahead store).
    pub pages_per_layer: usize,
    pub active_pages: usize,
    pub gate_up_expert_bytes: u64,
    pub down_expert_bytes: u64,
    /// Aligned payload span of a full page, per projection.
    pub gate_up_span: u64,
    pub down_span: u64,
    /// Zeroed tail pad after each projection ([`ARENA_TAIL_PAD_BYTES`]).
    pub tail_pad: u64,
    pub gate_up_offset: u64,
    pub gate_up_pad_offset: u64,
    pub down_offset: u64,
    pub down_pad_offset: u64,
    pub table_offset: u64,
    /// Rebased gate/up pointer table, down pointer table and codec table.
    pub table_bytes: u64,
    pub page_bytes: u64,
}

impl PanelPageLayout {
    /// From one layer's packed gate/up and down payload bytes (the largest
    /// trunk layer's). Every expert's span must be equal and a whole number
    /// of bytes: a non-uniform layer is refused, not rounded.
    pub fn new(
        num_experts: usize,
        gate_up_layer_bytes: u64,
        down_layer_bytes: u64,
    ) -> Result<Self, String> {
        if num_experts == 0 {
            return Err("slow-link prefill page layout has no experts".to_string());
        }
        let experts = num_experts as u64;
        if gate_up_layer_bytes == 0
            || down_layer_bytes == 0
            || gate_up_layer_bytes % experts != 0
            || down_layer_bytes % experts != 0
        {
            return Err(format!(
                "slow-link prefill needs uniform per-expert payloads: {gate_up_layer_bytes} gate/up \
                 and {down_layer_bytes} down bytes do not split over {num_experts} experts"
            ));
        }
        let gate_up_expert_bytes = gate_up_layer_bytes / experts;
        let down_expert_bytes = down_layer_bytes / experts;
        let per_page = EXPERTS_PER_PAGE as u64;
        let gate_up_span = align(checked_mul(per_page, gate_up_expert_bytes, "page gate/up span")?)?;
        let down_span = align(checked_mul(per_page, down_expert_bytes, "page down span")?)?;
        let tail_pad = ARENA_TAIL_PAD_BYTES as u64;
        let pointer_table = align(checked_mul(per_page, 8, "page pointer table")?)?;
        let codec_table = align(per_page)?;
        let table_bytes = checked_add(
            checked_mul(2, pointer_table, "page tables")?,
            codec_table,
            "page tables",
        )?;
        let gate_up_offset = 0;
        let gate_up_pad_offset = gate_up_span;
        let down_offset = checked_add(gate_up_pad_offset, tail_pad, "page layout")?;
        let down_pad_offset = checked_add(down_offset, down_span, "page layout")?;
        let table_offset = checked_add(down_pad_offset, tail_pad, "page layout")?;
        let page_bytes = checked_add(table_offset, table_bytes, "page bytes")?;
        Ok(Self {
            experts_per_page: EXPERTS_PER_PAGE,
            num_experts,
            pages_per_layer: num_experts.div_ceil(EXPERTS_PER_PAGE),
            active_pages: ACTIVE_PAGES,
            gate_up_expert_bytes,
            down_expert_bytes,
            gate_up_span,
            down_span,
            tail_pad,
            gate_up_offset,
            gate_up_pad_offset,
            down_offset,
            down_pad_offset,
            table_offset,
            table_bytes,
            page_bytes,
        })
    }

    /// Active plus lookahead pages: the whole admitted working set.
    pub fn total_pages(&self) -> usize {
        self.active_pages + self.pages_per_layer
    }

    /// Device bytes of the double-buffered active ring.
    pub fn ring_bytes(&self) -> Result<u64, String> {
        checked_mul(self.active_pages as u64, self.page_bytes, "active ring bytes")
    }

    /// Device bytes of the one-layer-ahead lookahead store.
    pub fn lookahead_bytes(&self) -> Result<u64, String> {
        checked_mul(
            self.pages_per_layer as u64,
            self.page_bytes,
            "lookahead store bytes",
        )
    }

    /// Ring plus lookahead.
    pub fn total_bytes(&self) -> Result<u64, String> {
        checked_add(self.ring_bytes()?, self.lookahead_bytes()?, "page store bytes")
    }
}

/// The spans of the prompt arena, in layout order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaField {
    /// Four hyper-connection branch streams of every window row, reserved at
    /// F32 width (the widest tag a tile can exit with; the per-tile BF16/F32
    /// tags are host-side metadata).
    HcStreams,
    /// The saved prepared MoE input of every row, F32.
    PreparedInput,
    /// Per-row saved gates: the hyper-connection gates and the shared-expert
    /// scalar, F32.
    Gates,
    TopkIds,
    TopkWeights,
    SortedSlotIndex,
    InversePerm,
    ExpertCounts,
    ExpertOffsets,
    ExpertTileIds,
    /// The widened (F32) accumulator of the BF16 combine, one hidden row per
    /// window row. No `rows x top-k x hidden` output exists anywhere.
    Accumulator,
}

/// Every [`ArenaField`], in layout order.
pub const ARENA_FIELDS: [ArenaField; 11] = [
    ArenaField::HcStreams,
    ArenaField::PreparedInput,
    ArenaField::Gates,
    ArenaField::TopkIds,
    ArenaField::TopkWeights,
    ArenaField::SortedSlotIndex,
    ArenaField::InversePerm,
    ArenaField::ExpertCounts,
    ArenaField::ExpertOffsets,
    ArenaField::ExpertTileIds,
    ArenaField::Accumulator,
];

impl ArenaField {
    pub fn name(self) -> &'static str {
        match self {
            Self::HcStreams => "HC streams",
            Self::PreparedInput => "prepared input",
            Self::Gates => "gates",
            Self::TopkIds => "top-k ids",
            Self::TopkWeights => "top-k weights",
            Self::SortedSlotIndex => "sorted slot index",
            Self::InversePerm => "inverse perm",
            Self::ExpertCounts => "expert counts",
            Self::ExpertOffsets => "expert offsets",
            Self::ExpertTileIds => "expert tile ids",
            Self::Accumulator => "accumulator",
        }
    }
}

/// One arena span: [`S1_ALIGN`]-aligned offset, exact payload bytes (the
/// next span starts at the aligned end).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSpan {
    pub field: ArenaField,
    pub offset: u64,
    pub bytes: u64,
}

impl ArenaSpan {
    /// One past the last payload byte.
    pub fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}

/// The prompt arena of one window: a single device allocation of
/// [`Self::total_bytes`], sliced by [`Self::span`]. The census charges
/// exactly this number and the allocator allocates exactly this number, from
/// this one function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptArenaLayout {
    pub rows: usize,
    pub tile_rows: usize,
    pub tiles: usize,
    pub spans: [ArenaSpan; ARENA_FIELDS.len()],
    pub total_bytes: u64,
}

impl PromptArenaLayout {
    /// The layout of a window of `rows` prompt rows.
    pub fn new(geometry: &ModelGeometry, rows: usize, tile_rows: usize) -> Result<Self, String> {
        if rows == 0 || tile_rows == 0 {
            return Err("slow-link prefill arena has no rows".to_string());
        }
        let rows64 = rows as u64;
        let f32_bytes = 4u64;
        let top_k = geometry.experts_per_token as u64;
        let experts = geometry.num_experts as u64;
        let slots = checked_mul(rows64, top_k, "arena slots")?;
        let grouped = crate::program::grouped_m_total_bound(
            usize::try_from(slots).map_err(|_| "slow-link prefill arena slots overflow")?,
            geometry.num_experts,
        )
        .map_err(|error| error.to_string())? as u64;
        let row_bytes = |width: u64| -> Result<u64, String> {
            checked_mul(checked_mul(rows64, width, "arena row bytes")?, f32_bytes, "arena bytes")
        };
        let hidden = geometry.hidden as u64;
        let sizes: [u64; ARENA_FIELDS.len()] = [
            row_bytes(geometry.wide()?)?,
            row_bytes(hidden)?,
            row_bytes(geometry.hc_count as u64 + 1)?,
            checked_mul(slots, 4, "top-k id bytes")?,
            checked_mul(slots, f32_bytes, "top-k weight bytes")?,
            checked_mul(grouped, 4, "sorted slot index bytes")?,
            checked_mul(slots, 4, "inverse perm bytes")?,
            checked_mul(experts, 4, "expert count bytes")?,
            checked_mul(experts + 1, 4, "expert offset bytes")?,
            checked_mul(grouped / 16, 4, "expert tile id bytes")?,
            row_bytes(hidden)?,
        ];
        let mut offset = 0u64;
        let mut spans = [ArenaSpan {
            field: ArenaField::HcStreams,
            offset: 0,
            bytes: 0,
        }; ARENA_FIELDS.len()];
        for (index, (field, bytes)) in ARENA_FIELDS.into_iter().zip(sizes).enumerate() {
            spans[index] = ArenaSpan {
                field,
                offset,
                bytes,
            };
            offset = align(checked_add(offset, bytes, "arena bytes")?)?;
        }
        Ok(Self {
            rows,
            tile_rows,
            tiles: rows.div_ceil(tile_rows),
            spans,
            total_bytes: offset,
        })
    }

    /// The span of `field`.
    pub fn span(&self, field: ArenaField) -> ArenaSpan {
        self.spans[ARENA_FIELDS
            .iter()
            .position(|candidate| *candidate == field)
            .expect("every field is in ARENA_FIELDS")]
    }
}

/// Device bytes of the panel scratch: [`PANEL_MAX_GROUPED_ROWS`] grouped rows
/// of the gate/up output, the activation and the down output, all F32.
pub fn panel_scratch_bytes(geometry: &ModelGeometry) -> Result<u64, String> {
    let width = checked_add(
        checked_mul(3, geometry.moe_intermediate as u64, "panel scratch width")?,
        geometry.hidden as u64,
        "panel scratch width",
    )?;
    checked_mul(
        checked_mul(PANEL_MAX_GROUPED_ROWS as u64, width, "panel scratch")?,
        4,
        "panel scratch",
    )
}

/// Everything the S1 profile costs beyond the incumbent's 1,536-row reserve,
/// term by term. [`Self::total`] is the one number placement charges
/// ([`crate::expert_residency::PlacementCharges::slowlink`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlowlinkCensus {
    /// The double-buffered active expert pages.
    pub ring: u64,
    /// The one-layer-ahead lookahead pages.
    pub lookahead: u64,
    pub panel_scratch: u64,
    pub arena: u64,
    /// The entry snapshot plus [`S1_MAX_BOUNDARY_SNAPSHOTS`] boundary copies
    /// of the fixed per-layer state.
    pub boundary_state: u64,
}

impl SlowlinkCensus {
    /// Every term, named, in log order.
    pub fn terms(&self) -> [(&'static str, u64); 5] {
        [
            ("expert ring", self.ring),
            ("expert lookahead", self.lookahead),
            ("panel scratch", self.panel_scratch),
            ("prompt arena", self.arena),
            ("boundary state", self.boundary_state),
        ]
    }

    /// Sum of every term, checked.
    pub fn total(&self) -> Result<u64, String> {
        self.terms().into_iter().try_fold(0u64, |total, (name, bytes)| {
            total
                .checked_add(bytes)
                .ok_or_else(|| format!("slow-link prefill census overflows at '{name}'"))
        })
    }

    /// `name MiB, ...` for the placement log.
    pub fn describe(&self) -> String {
        self.terms()
            .into_iter()
            .map(|(name, bytes)| format!("{name} {} MiB", bytes >> 20))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What [`admit`] needs from the loader's own placement inputs.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionInputs {
    pub geometry: ModelGeometry,
    /// Free device bytes the placement is sized from.
    pub free: u64,
    pub non_expert: u64,
    /// The largest trunk layer's routed-expert bytes (`resident_split`).
    pub layer_expert_bytes: u64,
    /// Gate/up and down payload bytes of that layer, separately.
    pub gate_up_layer_bytes: u64,
    pub down_layer_bytes: u64,
    pub trunk_layers: usize,
    /// [`crate::expert_residency::AutoReserve::bytes`] at [`S1_TILE_ROWS`].
    pub base_reserve: u64,
    /// The incumbent 1,536-row placement's layers, for the log only.
    pub baseline_layers: usize,
    /// Bytes of one copy of the fixed per-layer state
    /// ([`crate::expert_residency::StateCensus::state`]).
    pub fixed_state_bytes: u64,
    pub window_rows: usize,
    pub link_gbps: f64,
}

/// The immutable admitted S1 profile: the frozen placement the load took and
/// the layouts the forward allocates from. Handed to the bundle before the
/// forward attaches; never mutated by a request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlowlinkProfile {
    pub tile_rows: usize,
    pub window_rows: usize,
    pub geometry: ModelGeometry,
    pub pages: PanelPageLayout,
    pub arena: PromptArenaLayout,
    pub census: SlowlinkCensus,
    /// Layers the charged placement keeps resident (at least
    /// [`S1_MIN_RESIDENT_LAYERS`]).
    pub resident_layers: usize,
    /// Layers the incumbent 1,536-row placement kept, for comparison.
    pub baseline_layers: usize,
    pub link_gbps: f64,
}

impl SlowlinkProfile {
    /// The one number placement charges for this profile.
    pub fn reserve_bytes(&self) -> Result<u64, String> {
        self.census.total()
    }

    /// Windows a request of `rows` rows runs ([`S1_WINDOW_ROWS`] each, the
    /// last possibly short).
    pub fn windows_for(&self, rows: usize) -> usize {
        rows.div_ceil(self.window_rows)
    }

    /// The placement log line.
    pub fn describe(&self) -> String {
        format!(
            "slow-link prefill (S1): {} MiB charged [{}], {}-row tile, {}-row window, \
             {} resident layers (1536-row baseline {}), {}+{} expert pages of {} bytes, link {:.1} GB/s",
            self.census.total().unwrap_or(0) >> 20,
            self.census.describe(),
            self.tile_rows,
            self.window_rows,
            self.resident_layers,
            self.baseline_layers,
            self.pages.active_pages,
            self.pages.pages_per_layer,
            self.pages.page_bytes,
            self.link_gbps,
        )
    }
}

/// The joint S1 placement: charge the tile, arena, page store, panel scratch
/// and boundary state together, then count the layers that remain. Refuses,
/// before anything is allocated, with the exact deficit when fewer than
/// [`S1_MIN_RESIDENT_LAYERS`] stay resident.
pub fn admit(inputs: &AdmissionInputs) -> Result<SlowlinkProfile, SlowlinkDecline> {
    let geometry_error = SlowlinkDecline::Geometry;
    let window_rows = inputs.window_rows;
    let pages = PanelPageLayout::new(
        inputs.geometry.num_experts,
        inputs.gate_up_layer_bytes,
        inputs.down_layer_bytes,
    )
    .map_err(geometry_error)?;
    let arena = PromptArenaLayout::new(&inputs.geometry, window_rows, S1_TILE_ROWS)
        .map_err(SlowlinkDecline::Geometry)?;
    let census = (|| -> Result<SlowlinkCensus, String> {
        Ok(SlowlinkCensus {
            ring: pages.ring_bytes()?,
            lookahead: pages.lookahead_bytes()?,
            panel_scratch: panel_scratch_bytes(&inputs.geometry)?,
            arena: arena.total_bytes,
            boundary_state: checked_mul(
                1 + S1_MAX_BOUNDARY_SNAPSHOTS as u64,
                inputs.fixed_state_bytes,
                "boundary state",
            )?,
        })
    })()
    .map_err(SlowlinkDecline::Geometry)?;
    let charges = crate::expert_residency::PlacementCharges::new(inputs.layer_expert_bytes, false)
        .and_then(|charges| census.total().map(|bytes| charges.with_slowlink(bytes)))
        .map_err(SlowlinkDecline::Geometry)?;
    let reserve = charges
        .reserve(inputs.base_reserve)
        .map_err(SlowlinkDecline::Geometry)?;
    let layers = if inputs.non_expert.checked_add(reserve).is_some_and(|need| need <= inputs.free)
    {
        crate::expert_residency::auto_vram_layers(
            inputs.free,
            inputs.non_expert,
            inputs.layer_expert_bytes,
            inputs.trunk_layers,
            reserve,
        )
    } else {
        0
    };
    if layers >= inputs.trunk_layers {
        return Err(SlowlinkDecline::NoHostLayers {
            resident_layers: layers,
        });
    }
    if layers < S1_MIN_RESIDENT_LAYERS {
        let required = (S1_MIN_RESIDENT_LAYERS as u64)
            .checked_mul(inputs.layer_expert_bytes)
            .and_then(|bytes| bytes.checked_add(inputs.non_expert))
            .and_then(|bytes| bytes.checked_add(reserve))
            .ok_or_else(|| SlowlinkDecline::Geometry("slow-link prefill deficit overflows".into()))?;
        return Err(SlowlinkDecline::Deficit(Deficit {
            resident_layers: layers,
            required_layers: S1_MIN_RESIDENT_LAYERS,
            free: inputs.free,
            non_expert: inputs.non_expert,
            layer_expert_bytes: inputs.layer_expert_bytes,
            reserve,
            deficit_bytes: required.saturating_sub(inputs.free),
        }));
    }
    Ok(SlowlinkProfile {
        tile_rows: S1_TILE_ROWS,
        window_rows,
        geometry: inputs.geometry,
        pages,
        arena,
        census,
        resident_layers: layers,
        baseline_layers: inputs.baseline_layers,
        link_gbps: inputs.link_gbps,
    })
}

/// How a request's prefill output is captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureKind {
    /// Final row only (or silent): the product path.
    None,
    /// Bounded row capture that fits the admitted arena (MTP target capture).
    Bounded,
    /// Every row's logits in caller-owned storage: stays incumbent.
    AllRows,
    /// Externally captured or retained prefill: stays incumbent.
    External,
}

/// A request as the architecture sees it before any mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestShape {
    pub rows: usize,
    pub entry_pos: usize,
    /// Session snapshot cuts inside the request.
    pub session_cuts: usize,
    pub capture: CaptureKind,
}

/// A request S1 will run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestVerdict {
    pub rows: usize,
    /// Expert sweeps: 1 means one sweep for the whole prompt.
    pub windows: usize,
}

/// Why a request stays on the incumbent route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestBypass {
    KillSwitch,
    AutoNotQualified,
    TooShort { rows: usize, minimum: usize },
    AllRowCapture,
    ExternalCapture,
    TooManyBoundaries { cuts: usize, maximum: usize },
    /// The profile's tile is not a valid chunk for this request.
    NoRows,
}

/// Judge a request against the frozen profile, before any mutation or S1
/// allocation. `Err` is an incumbent request that uses only resources the
/// profile already admitted.
pub fn qualify_request(
    mode: SlowlinkMode,
    profile: &SlowlinkProfile,
    request: &RequestShape,
) -> Result<RequestVerdict, RequestBypass> {
    let minimum = match mode {
        SlowlinkMode::Off => return Err(RequestBypass::KillSwitch),
        SlowlinkMode::Auto if !S1_AUTO_QUALIFIED => return Err(RequestBypass::AutoNotQualified),
        SlowlinkMode::Auto => S1_MIN_AUTO_REQUEST_ROWS,
        SlowlinkMode::Force => S1_MIN_FORCED_REQUEST_ROWS,
    };
    if request.rows == 0 {
        return Err(RequestBypass::NoRows);
    }
    match request.capture {
        CaptureKind::None | CaptureKind::Bounded => {}
        CaptureKind::AllRows => return Err(RequestBypass::AllRowCapture),
        CaptureKind::External => return Err(RequestBypass::ExternalCapture),
    }
    if request.rows < minimum {
        return Err(RequestBypass::TooShort {
            rows: request.rows,
            minimum,
        });
    }
    if request.session_cuts > S1_MAX_BOUNDARY_SNAPSHOTS {
        return Err(RequestBypass::TooManyBoundaries {
            cuts: request.session_cuts,
            maximum: S1_MAX_BOUNDARY_SNAPSHOTS,
        });
    }
    Ok(RequestVerdict {
        rows: request.rows,
        windows: profile.windows_for(request.rows),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_residency::{auto_placement_layers, PlacementCharges};

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    // Flash-Next MQ4 packed: 512 experts of 1,740,800 gate/up and 870,400 down bytes.
    const GATE_UP_LAYER: u64 = 512 * 1_740_800;
    const DOWN_LAYER: u64 = 512 * 870_400;
    const LAYER: u64 = GATE_UP_LAYER + DOWN_LAYER;

    fn shape() -> LoadShape<'static> {
        LoadShape {
            arch: "gfx1100",
            unified_memory: false,
            expert_parallel: false,
            pipeline_band: false,
            packed_mq4: true,
            routed_experts: true,
            expert_keep: false,
            explicit_vram_layers: false,
            hotset_sidecar: false,
            whole_first: false,
            stage_opt_out: false,
            cpu_offload: false,
            chunk_rows_env: None,
            max_seq: 16640,
        }
    }

    fn geometry() -> ModelGeometry {
        ModelGeometry::from_config(&crate::config::compact_test_config()).unwrap()
    }

    fn inputs(free: u64) -> AdmissionInputs {
        AdmissionInputs {
            geometry: geometry(),
            free,
            non_expert: 5_364_000_000,
            layer_expert_bytes: LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: 48,
            base_reserve: 5261 * MIB,
            baseline_layers: 10,
            fixed_state_bytes: 64 * MIB,
            window_rows: S1_WINDOW_ROWS,
            link_gbps: 7.09,
        }
    }

    fn census_total(inputs: &AdmissionInputs) -> u64 {
        let pages = PanelPageLayout::new(
            inputs.geometry.num_experts,
            inputs.gate_up_layer_bytes,
            inputs.down_layer_bytes,
        )
        .unwrap();
        let arena = PromptArenaLayout::new(&inputs.geometry, inputs.window_rows, S1_TILE_ROWS)
            .unwrap();
        pages.total_bytes().unwrap()
            + panel_scratch_bytes(&inputs.geometry).unwrap()
            + arena.total_bytes
            + (1 + S1_MAX_BOUNDARY_SNAPSHOTS as u64) * inputs.fixed_state_bytes
    }

    /// Free VRAM that keeps exactly `layers` layers beside the S1 charge.
    fn free_for(inputs: &AdmissionInputs, layers: u64) -> u64 {
        inputs.non_expert
            + inputs.base_reserve
            + EXPERT_MARGIN
            + census_total(inputs)
            + layers * LAYER
            + 1
    }
    const EXPERT_MARGIN: u64 = 512 << 20;

    #[test]
    fn slowlink_admission_selector_modes_follow_the_typed_value() {
        assert_eq!(SlowlinkMode::from_typed(Some(false)), SlowlinkMode::Off);
        assert_eq!(SlowlinkMode::from_typed(Some(true)), SlowlinkMode::Force);
        assert_eq!(SlowlinkMode::from_typed(None), SlowlinkMode::Auto);
        assert_eq!(SLOWLINK_PREFILL_ENV, "HIPFIRE_QWEN4_SLOWLINK_PREFILL");
        assert!(hipfire_config::is_typed_bool_developer_key(
            "developer.qwen4_slowlink_prefill"
        ));
    }

    #[test]
    fn slowlink_admission_unset_is_the_incumbent_until_qualified() {
        assert!(!S1_AUTO_QUALIFIED, "default-off until every gate passes");
        assert_eq!(
            preflight(SlowlinkMode::Auto, &shape()),
            Err(SlowlinkDecline::AutoNotQualified)
        );
        assert_eq!(
            preflight(SlowlinkMode::Off, &shape()),
            Err(SlowlinkDecline::KillSwitch)
        );
        assert_eq!(preflight(SlowlinkMode::Force, &shape()), Ok(()));
    }

    #[test]
    fn slowlink_admission_arch_truth_table_is_exact_gfx1100_only() {
        for arch in [
            "gfx906", "gfx908", "gfx90a", "gfx942", "gfx1010", "gfx1030", "gfx1031", "gfx1101",
            "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201", "gfx1100x", "",
        ] {
            let load = LoadShape { arch, ..shape() };
            assert_eq!(
                preflight(SlowlinkMode::Force, &load),
                Err(SlowlinkDecline::UnqualifiedArch(arch.to_string())),
                "{arch}"
            );
        }
        assert_eq!(preflight(SlowlinkMode::Force, &shape()), Ok(()));
        // The arch gate precedes everything else, so no other arch is ever
        // reported as anything but the arch.
        let load = LoadShape {
            arch: "gfx1201",
            unified_memory: true,
            expert_parallel: true,
            ..shape()
        };
        assert_eq!(
            preflight(SlowlinkMode::Force, &load),
            Err(SlowlinkDecline::UnqualifiedArch("gfx1201".into()))
        );
    }

    #[test]
    fn slowlink_admission_link_truth_table() {
        let verdict = |gbps: f64| link_verdict(HostLink::Discrete { gbps });
        assert_eq!(verdict(7.09), Ok(7.09));
        assert_eq!(verdict(12.0), Ok(12.0));
        assert_eq!(verdict(15.99), Ok(15.99));
        assert_eq!(verdict(16.0), Err(SlowlinkDecline::FastLink { gbps: 16.0 }));
        assert_eq!(verdict(26.0), Err(SlowlinkDecline::FastLink { gbps: 26.0 }));
        assert_eq!(verdict(50.0), Err(SlowlinkDecline::FastLink { gbps: 50.0 }));
        assert_eq!(
            link_verdict(HostLink::Unified),
            Err(SlowlinkDecline::UnifiedMemory)
        );
        assert_eq!(
            link_verdict(HostLink::Unmeasured),
            Err(SlowlinkDecline::UnmeasuredLink)
        );
        // Unusable figures classify as unmeasured, never narrow.
        for bad in [Some(f64::NAN), Some(0.0), Some(-3.0), Some(f64::INFINITY), None] {
            assert_eq!(
                link_verdict(HostLink::classify(false, bad)),
                Err(SlowlinkDecline::UnmeasuredLink),
                "{bad:?}"
            );
        }
        assert_eq!(
            link_verdict(HostLink::classify(true, Some(7.0))),
            Err(SlowlinkDecline::UnifiedMemory)
        );
    }

    #[test]
    fn slowlink_admission_unsupported_modes_decline_before_any_probe() {
        let cases: [(&str, LoadShape<'static>); 10] = [
            ("ep", LoadShape { expert_parallel: true, ..shape() }),
            ("pp", LoadShape { pipeline_band: true, ..shape() }),
            ("recipe", LoadShape { packed_mq4: false, ..shape() }),
            ("dense", LoadShape { routed_experts: false, ..shape() }),
            ("keep", LoadShape { expert_keep: true, ..shape() }),
            ("layers", LoadShape { explicit_vram_layers: true, ..shape() }),
            ("hotset", LoadShape { hotset_sidecar: true, ..shape() }),
            ("whole-first", LoadShape { whole_first: true, ..shape() }),
            ("stage off", LoadShape { stage_opt_out: true, ..shape() }),
            ("cpu", LoadShape { cpu_offload: true, ..shape() }),
        ];
        for (name, load) in cases {
            assert!(
                matches!(
                    preflight(SlowlinkMode::Force, &load),
                    Err(SlowlinkDecline::Unsupported(_))
                ),
                "{name}"
            );
        }
        assert_eq!(
            preflight(
                SlowlinkMode::Force,
                &LoadShape { unified_memory: true, ..shape() }
            ),
            Err(SlowlinkDecline::UnifiedMemory)
        );
        for max_seq in [0, 1536, 3071] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &LoadShape { max_seq, ..shape() }),
                Err(SlowlinkDecline::ShortContext { max_seq })
            );
        }
        assert_eq!(
            preflight(SlowlinkMode::Force, &LoadShape { max_seq: 3072, ..shape() }),
            Ok(())
        );
    }

    #[test]
    fn slowlink_admission_explicit_chunk_rows_must_equal_the_tile() {
        let with = |env| LoadShape { chunk_rows_env: Some(env), ..shape() };
        assert_eq!(preflight(SlowlinkMode::Force, &with("1536")), Ok(()));
        assert_eq!(preflight(SlowlinkMode::Force, &with(" 1536 ")), Ok(()));
        // Rounded down to whole 256-row tiles like the chunk policy.
        assert_eq!(preflight(SlowlinkMode::Force, &with("1600")), Ok(()));
        assert_eq!(
            preflight(SlowlinkMode::Force, &with("8192")),
            Err(SlowlinkDecline::ExplicitChunk { rows: 8192 })
        );
        assert_eq!(
            preflight(SlowlinkMode::Force, &with("1024")),
            Err(SlowlinkDecline::ExplicitChunk { rows: 1024 })
        );
        // Unusable values are unset, as in the chunk policy.
        for env in ["", "abc", "100", "-1"] {
            assert_eq!(preflight(SlowlinkMode::Force, &with(env)), Ok(()), "{env:?}");
        }
    }

    #[test]
    fn slowlink_admission_page_layout_matches_the_planned_reserve() {
        let pages = PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER).unwrap();
        assert_eq!(pages.experts_per_page, 32);
        assert_eq!(pages.pages_per_layer, 16);
        assert_eq!(pages.total_pages(), 18);
        assert_eq!(pages.gate_up_expert_bytes, 1_740_800);
        assert_eq!(pages.down_expert_bytes, 870_400);
        assert_eq!(pages.gate_up_span, 32 * 1_740_800);
        assert_eq!(pages.down_span, 32 * 870_400);
        // Regions are ordered, disjoint and end exactly at the page.
        assert_eq!(pages.gate_up_offset, 0);
        assert_eq!(pages.gate_up_pad_offset, pages.gate_up_span);
        assert_eq!(pages.down_offset, pages.gate_up_pad_offset + 4096);
        assert_eq!(pages.down_pad_offset, pages.down_offset + pages.down_span);
        assert_eq!(pages.table_offset, pages.down_pad_offset + 4096);
        assert_eq!(pages.page_bytes, pages.table_offset + pages.table_bytes);
        assert_eq!(pages.table_bytes, 256 + 256 + 256);
        // Plan: ~0.156 GiB ring + ~1.245 GiB lookahead = ~1.401 GiB payload,
        // plus pads and tables.
        let payload = 18 * 32 * 2_611_200u64;
        assert_eq!(payload, pages.total_bytes().unwrap() - 18 * (2 * 4096 + 768));
        assert!(payload > 1_500_000_000 && payload < 1_510_000_000, "{payload}");
        assert_eq!(pages.ring_bytes().unwrap(), 2 * pages.page_bytes);
        assert_eq!(pages.lookahead_bytes().unwrap(), 16 * pages.page_bytes);
    }

    #[test]
    fn slowlink_admission_page_layout_refuses_non_uniform_and_overflow() {
        assert!(PanelPageLayout::new(0, 1, 1).is_err());
        assert!(PanelPageLayout::new(512, 0, DOWN_LAYER).is_err());
        assert!(PanelPageLayout::new(512, GATE_UP_LAYER, 0).is_err());
        assert!(PanelPageLayout::new(512, GATE_UP_LAYER + 1, DOWN_LAYER).is_err());
        assert!(PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER + 7).is_err());
        assert!(PanelPageLayout::new(1, u64::MAX, 1).is_err());
        assert!(PanelPageLayout::new(1, 1, u64::MAX).is_err());
        // A layer that is not a multiple of 32 experts still has a short last page.
        let odd = PanelPageLayout::new(40, 40 * 256, 40 * 256).unwrap();
        assert_eq!(odd.pages_per_layer, 2);
    }

    #[test]
    fn slowlink_admission_arena_spans_are_aligned_disjoint_and_exact() {
        let geometry = geometry();
        let arena = PromptArenaLayout::new(&geometry, S1_WINDOW_ROWS, S1_TILE_ROWS).unwrap();
        assert_eq!(arena.tiles, 11);
        assert_eq!(arena.rows, 16384);
        let wide = (geometry.hc_count * geometry.hidden) as u64;
        let rows = S1_WINDOW_ROWS as u64;
        assert_eq!(arena.span(ArenaField::HcStreams).bytes, rows * wide * 4);
        assert_eq!(
            arena.span(ArenaField::PreparedInput).bytes,
            rows * geometry.hidden as u64 * 4
        );
        assert_eq!(
            arena.span(ArenaField::Accumulator).bytes,
            rows * geometry.hidden as u64 * 4
        );
        assert_eq!(
            arena.span(ArenaField::Gates).bytes,
            rows * (geometry.hc_count as u64 + 1) * 4
        );
        let slots = rows * geometry.experts_per_token as u64;
        assert_eq!(arena.span(ArenaField::TopkIds).bytes, slots * 4);
        assert_eq!(arena.span(ArenaField::TopkWeights).bytes, slots * 4);
        assert_eq!(arena.span(ArenaField::InversePerm).bytes, slots * 4);
        assert_eq!(
            arena.span(ArenaField::ExpertCounts).bytes,
            geometry.num_experts as u64 * 4
        );
        assert_eq!(
            arena.span(ArenaField::ExpertOffsets).bytes,
            (geometry.num_experts as u64 + 1) * 4
        );
        let mut previous_end = 0u64;
        for (span, field) in arena.spans.iter().zip(ARENA_FIELDS) {
            assert_eq!(span.field, field);
            assert_eq!(span.offset % S1_ALIGN, 0, "{}", field.name());
            assert!(span.offset >= previous_end, "{} overlaps", field.name());
            previous_end = span.end();
        }
        assert!(previous_end <= arena.total_bytes);
        assert_eq!(arena.total_bytes % S1_ALIGN, 0);
        assert!(arena.total_bytes - previous_end < S1_ALIGN);
        // One window row more or fewer changes the allocation and the census alike.
        let shorter = PromptArenaLayout::new(&geometry, S1_WINDOW_ROWS - 1, S1_TILE_ROWS).unwrap();
        assert!(shorter.total_bytes < arena.total_bytes);
        assert!(PromptArenaLayout::new(&geometry, 0, S1_TILE_ROWS).is_err());
        assert!(PromptArenaLayout::new(&geometry, 1, 0).is_err());
        assert!(PromptArenaLayout::new(&geometry, usize::MAX / 2, S1_TILE_ROWS).is_err());
    }

    #[test]
    fn slowlink_admission_census_equals_the_allocated_layouts() {
        let inputs = inputs(free_for(&inputs(0), 10));
        let profile = admit(&inputs).unwrap();
        assert_eq!(profile.census.total().unwrap(), census_total(&inputs));
        assert_eq!(profile.census.arena, profile.arena.total_bytes);
        assert_eq!(
            profile.census.ring + profile.census.lookahead,
            profile.pages.total_bytes().unwrap()
        );
        assert_eq!(
            profile.census.panel_scratch,
            panel_scratch_bytes(&profile.geometry).unwrap()
        );
        assert_eq!(profile.reserve_bytes().unwrap(), census_total(&inputs));
        assert_eq!(profile.tile_rows, 1536);
        assert!(profile.tile_rows <= 1536);
        assert_eq!(profile.window_rows, 16384);
        assert_eq!(profile.census.terms().len(), 5);
        assert!(profile.describe().contains("10 resident layers"));
    }

    #[test]
    fn slowlink_admission_keeps_at_least_ten_layers_and_the_loader_agrees() {
        let probe = inputs(0);
        for layers in 10..=14u64 {
            let inputs = inputs(free_for(&probe, layers));
            let profile = admit(&inputs).unwrap();
            assert_eq!(profile.resident_layers as u64, layers);
            // The placement the loader runs is the same arithmetic.
            let charges = PlacementCharges::new(LAYER, false)
                .unwrap()
                .with_slowlink(profile.reserve_bytes().unwrap());
            assert_eq!(
                auto_placement_layers(
                    inputs.free,
                    inputs.non_expert,
                    LAYER,
                    48,
                    inputs.base_reserve,
                    &charges
                )
                .unwrap() as u64,
                layers
            );
        }
    }

    #[test]
    fn slowlink_admission_refuses_a_deficit_with_its_exact_bytes() {
        let probe = inputs(0);
        let nine = inputs(free_for(&probe, 9));
        let Err(SlowlinkDecline::Deficit(deficit)) = admit(&nine) else {
            panic!("nine layers must be refused");
        };
        assert_eq!(deficit.resident_layers, 9);
        assert_eq!(deficit.required_layers, 10);
        assert!(deficit.deficit_bytes > 0 && deficit.deficit_bytes <= LAYER);
        // Adding exactly the deficit admits exactly ten.
        let fixed = inputs(nine.free + deficit.deficit_bytes);
        assert_eq!(admit(&fixed).unwrap().resident_layers, 10);
        // One byte short of the deficit still refuses.
        let short = inputs(nine.free + deficit.deficit_bytes - 1);
        assert!(matches!(admit(&short), Err(SlowlinkDecline::Deficit(_))));
        assert!(SlowlinkDecline::Deficit(deficit).is_capacity());
        assert!(SlowlinkDecline::Deficit(deficit)
            .to_string()
            .contains(&deficit.deficit_bytes.to_string()));
        // Not even the non-expert weights and the charged reserve fit.
        let none = inputs(GIB);
        let Err(SlowlinkDecline::Deficit(none)) = admit(&none) else {
            panic!("a nearly empty card is a deficit");
        };
        assert_eq!(none.resident_layers, 0);
        assert!(none.deficit_bytes > 10 * LAYER);
    }

    #[test]
    fn slowlink_admission_fully_resident_has_nothing_to_stream() {
        let inputs = inputs(200 * GIB);
        assert!(matches!(
            admit(&inputs),
            Err(SlowlinkDecline::NoHostLayers { resident_layers: 48 })
        ));
        assert!(!SlowlinkDecline::NoHostLayers { resident_layers: 48 }.is_capacity());
    }

    #[test]
    fn slowlink_admission_overflow_refuses_before_mutation() {
        let probe = inputs(0);
        let mut huge = inputs(free_for(&probe, 12));
        huge.fixed_state_bytes = u64::MAX;
        assert!(matches!(admit(&huge), Err(SlowlinkDecline::Geometry(_))));
        let mut huge = inputs(free_for(&probe, 12));
        huge.base_reserve = u64::MAX;
        assert!(matches!(admit(&huge), Err(SlowlinkDecline::Geometry(_))));
        let mut huge = inputs(free_for(&probe, 12));
        huge.window_rows = usize::MAX / 4;
        assert!(matches!(admit(&huge), Err(SlowlinkDecline::Geometry(_))));
        let mut bad = inputs(free_for(&probe, 12));
        bad.gate_up_layer_bytes += 1;
        assert!(matches!(admit(&bad), Err(SlowlinkDecline::Geometry(_))));
        assert!(SlowlinkDecline::Geometry(String::new()).is_capacity());
    }

    #[test]
    fn slowlink_admission_fast_route_charges_are_zero() {
        let layer = LAYER;
        let incumbent = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(incumbent.slowlink, 0);
        assert_eq!(incumbent.reserve(123).unwrap(), 123 + incumbent.margin);
        assert!(!incumbent.describe().contains("slow-link"));
        let charged = incumbent.with_slowlink(4 * GIB);
        assert_eq!(charged.reserve(123).unwrap(), 123 + incumbent.margin + 4 * GIB);
        assert!(charged.describe().contains("slow-link"));
        assert!(PlacementCharges::new(layer, true)
            .unwrap()
            .with_slowlink(u64::MAX)
            .reserve(1)
            .is_err());
    }

    fn profile_at_ten() -> SlowlinkProfile {
        admit(&inputs(free_for(&inputs(0), 10))).unwrap()
    }

    #[test]
    fn slowlink_admission_request_truth_table() {
        let profile = profile_at_ten();
        let req = |rows| RequestShape {
            rows,
            entry_pos: 0,
            session_cuts: 0,
            capture: CaptureKind::None,
        };
        // Forced: from two tiles up.
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(3071)),
            Err(RequestBypass::TooShort { rows: 3071, minimum: 3072 })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(3072)),
            Ok(RequestVerdict { rows: 3072, windows: 1 })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(16384)),
            Ok(RequestVerdict { rows: 16384, windows: 1 })
        );
        // A prompt past the window runs superwindows and says so.
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(16385)),
            Ok(RequestVerdict { rows: 16385, windows: 2 })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Off, &profile, &req(16384)),
            Err(RequestBypass::KillSwitch)
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Auto, &profile, &req(16384)),
            Err(RequestBypass::AutoNotQualified)
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(0)),
            Err(RequestBypass::NoRows)
        );
        for (capture, bypass) in [
            (CaptureKind::AllRows, RequestBypass::AllRowCapture),
            (CaptureKind::External, RequestBypass::ExternalCapture),
        ] {
            assert_eq!(
                qualify_request(
                    SlowlinkMode::Force,
                    &profile,
                    &RequestShape { capture, ..req(16384) }
                ),
                Err(bypass)
            );
        }
        for capture in [CaptureKind::None, CaptureKind::Bounded] {
            assert!(qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape { capture, entry_pos: 4096, ..req(16384) }
            )
            .is_ok());
        }
        assert_eq!(
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape { session_cuts: 4, ..req(16384) }
            )
            .unwrap()
            .windows,
            1
        );
        assert_eq!(
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape { session_cuts: 5, ..req(16384) }
            ),
            Err(RequestBypass::TooManyBoundaries { cuts: 5, maximum: 4 })
        );
        assert_eq!(S1_MIN_AUTO_REQUEST_ROWS, 16384);
        assert_eq!(profile.windows_for(1), 1);
        assert_eq!(profile.windows_for(32768), 2);
    }
}
