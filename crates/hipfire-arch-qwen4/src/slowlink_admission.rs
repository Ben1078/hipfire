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
//!    the expert page store, the route storage and A4 sidecars, the HC arena and the boundary state are all
//!    charged together, and the placement that remains must still keep
//!    [`S1_MIN_RESIDENT_LAYERS`] trunk layers resident. A shortfall is refused
//!    with its exact byte deficit ([`Deficit`]); fewer residents are never
//!    accepted.
//!
//! Steps 1 to 3 are probe-free; only a load that passes them measures the host
//! link, so no other arch and no non-candidate load pays for the probe, and a
//! load that fails any step is the incumbent byte for byte with a zero S1
//! charge ([`crate::expert_residency::PlacementCharges::slowlink`] is `0`).

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

/// Fewest request rows S1 runs when forced: one row past a tile, the first
/// genuinely multi-chunk request (a request of at most one tile is a named
/// bypass). The automatic selector starts at [`S1_MIN_AUTO_REQUEST_ROWS`].
pub const S1_MIN_FORCED_REQUEST_ROWS: usize = S1_TILE_ROWS + 1;

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

/// New boundary snapshots of the session fixed state one window may hold (the
/// entry snapshot is the incumbent's own). A request with more session cuts is a bypass.
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

/// Whether the NPU trunk engine is selected for `arch` under `mode`: the same
/// resolution the forward uses ([`crate::npu_trunk::NpuTrunkMode::resolve`]).
/// S1 declines a selected NPU before any reserve, so a pair route never meets
/// the layer-major schedule.
pub fn npu_trunk_selected(arch: &str, mode: crate::npu_trunk::NpuTrunkMode) -> bool {
    !mode.resolve(arch).is_off()
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
    /// The NPU trunk engine is selected (`HIPFIRE_QWEN4_NPU_TRUNK` resolved for
    /// this arch is not `off`: `auto` resolves to off outside gfx1151, an
    /// explicit `on`/`emulate` selects it anywhere). Read from the config
    /// snapshot ([`npu_trunk_selected`]); no device is opened or probed.
    pub npu_trunk: bool,
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
    ExplicitChunk {
        rows: usize,
    },
    ShortContext {
        max_seq: usize,
    },
    /// Measured discrete link at or above 16 GB/s.
    FastLink {
        gbps: f64,
    },
    /// The link probe failed or returned an unusable figure.
    UnmeasuredLink,
    /// Every routed expert already fits: there is no host layer to stream.
    NoHostLayers {
        resident_layers: usize,
    },
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
        SlowlinkMode::Auto if !S1_AUTO_QUALIFIED => return Err(SlowlinkDecline::AutoNotQualified),
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
        (
            !shape.packed_mq4,
            "recipe is not the qualified MQ4 packed layout",
        ),
        (!shape.routed_experts, "no routed experts"),
        (shape.expert_keep, "expert-keep artifact"),
        (
            shape.explicit_vram_layers,
            "explicit expert VRAM layer count",
        ),
        (shape.hotset_sidecar, "ranked expert hot set"),
        (shape.whole_first, "whole-first placement policy"),
        (shape.stage_opt_out, "HIPFIRE_QWEN4_EXPERT_STAGE=0 opt-out"),
        (shape.cpu_offload, "memory.offload_exec splice"),
        (shape.npu_trunk, "NPU trunk selected"),
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
    pub shared_intermediate: usize,
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
                        shared_intermediate: shared,
                    },
                ..
            } => Ok(Self {
                hidden: dims.hidden,
                hc_count: dims.hc_count,
                num_experts,
                experts_per_token,
                moe_intermediate: intermediate,
                shared_intermediate: shared,
            }),
            _ => Err("slow-link prefill needs a routed-expert (MoE) profile".to_string()),
        }
    }

    fn wide(&self) -> Result<u64, String> {
        checked_mul(
            self.hc_count as u64,
            self.hidden as u64,
            "hyper-connection width",
        )
    }
}

/// Copy/compute interval records the receipt timeline keeps per window
/// (`PanelPoolSpec::timeline_records`): 48 trunk layers x 16 pages x 2
/// (current and lookahead tags), with room. Host memory and pooled HIP events
/// only; no device bytes.
pub const S1_TIMELINE_RECORDS: usize = 2048;

/// Whole-layer compute marks the receipt timeline keeps per window.
pub const S1_TIMELINE_LAYERS: usize = 64;

/// The transport page pool of the profile: the geometry that
/// `gpu_forward::panel_stage::PanelPoolSpec` allocates, built here so the
/// census and the pool are ONE reserve (`PanelPoolSpec::reserve` is the single
/// authority for page, table and staging bytes). The pool's page payload is
/// the per-page upper bound [`Self::payload_bytes`]: the pool is constructed from
/// exactly this number, and attach refuses a source set whose real
/// `required_payload_bytes` exceeds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelPageLayout {
    pub experts_per_page: usize,
    pub num_experts: usize,
    /// Pages of one full layer (the lookahead store).
    pub pages_per_layer: usize,
    pub active_pages: usize,
    pub gate_up_expert_bytes: u64,
    pub down_expert_bytes: u64,
    /// Upper bound of one page's packed payload before the tail pad:
    /// `experts_per_page * (gate_up + down)` plus the worst-case run slack
    /// (`2 * experts_per_page` unmerged extents, each at most
    /// `2 * 255` bytes of 256-byte alignment and source-address congruence).
    pub payload_bytes: u64,
    /// Bytes of one page (payload, tail pad, rounded to 256).
    pub page_bytes: u64,
    /// Rebased gate/up and down pointer tables of one page (device).
    pub table_bytes_per_page: u64,
    /// Pinned host staging of every page's tables.
    pub host_staging_bytes: u64,
    pub timeline_records: usize,
    pub timeline_layers: usize,
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
        let to_usize = |bytes: u64| {
            usize::try_from(bytes)
                .map_err(|_| "slow-link prefill expert bytes overflow".to_string())
        };
        let spec = crate::gpu_forward::panel_stage::PanelPoolSpec::uniform(
            ACTIVE_PAGES,
            num_experts.div_ceil(EXPERTS_PER_PAGE),
            EXPERTS_PER_PAGE,
            num_experts,
            to_usize(gate_up_expert_bytes)?,
            to_usize(down_expert_bytes)?,
            S1_TIMELINE_RECORDS,
            S1_TIMELINE_LAYERS,
        )
        .map_err(|error| format!("slow-link prefill page pool: {error:?}"))?;
        let reserve = spec
            .reserve()
            .ok_or_else(|| "slow-link prefill page pool bytes overflow".to_string())?;
        Ok(Self {
            experts_per_page: EXPERTS_PER_PAGE,
            num_experts,
            pages_per_layer: spec.lookahead_pages,
            active_pages: ACTIVE_PAGES,
            gate_up_expert_bytes,
            down_expert_bytes,
            payload_bytes: spec.payload_bytes as u64,
            page_bytes: reserve.page_bytes as u64,
            table_bytes_per_page: (reserve.tables / reserve.pages) as u64,
            host_staging_bytes: reserve.staging as u64,
            timeline_records: S1_TIMELINE_RECORDS,
            timeline_layers: S1_TIMELINE_LAYERS,
        })
    }

    /// The pool the forward constructs (`PanelPool::new(gpu, layout.pool_spec()?)`):
    /// the same `PanelPoolSpec::uniform` the census read.
    pub(crate) fn pool_spec(
        &self,
    ) -> Result<crate::gpu_forward::panel_stage::PanelPoolSpec, String> {
        crate::gpu_forward::panel_stage::PanelPoolSpec::uniform(
            self.active_pages,
            self.pages_per_layer,
            self.experts_per_page,
            self.num_experts,
            usize::try_from(self.gate_up_expert_bytes)
                .map_err(|_| "slow-link prefill expert bytes overflow".to_string())?,
            usize::try_from(self.down_expert_bytes)
                .map_err(|_| "slow-link prefill expert bytes overflow".to_string())?,
            self.timeline_records,
            self.timeline_layers,
        )
        .map_err(|error| format!("slow-link prefill page pool: {error:?}"))
    }

    /// Active plus lookahead pages: the whole admitted working set.
    pub fn total_pages(&self) -> usize {
        self.active_pages + self.pages_per_layer
    }

    /// Device bytes of one page with its pointer tables.
    pub fn device_page_bytes(&self) -> Result<u64, String> {
        checked_add(
            self.page_bytes,
            self.table_bytes_per_page,
            "device page bytes",
        )
    }

    /// Device bytes of the double-buffered active ring.
    pub fn ring_bytes(&self) -> Result<u64, String> {
        checked_mul(
            self.active_pages as u64,
            self.device_page_bytes()?,
            "active ring bytes",
        )
    }

    /// Device bytes of the one-layer-ahead lookahead store.
    pub fn lookahead_bytes(&self) -> Result<u64, String> {
        checked_mul(
            self.pages_per_layer as u64,
            self.device_page_bytes()?,
            "lookahead store bytes",
        )
    }

    /// Ring plus lookahead: `PanelPoolReserve::device_total`.
    pub fn total_bytes(&self) -> Result<u64, String> {
        checked_add(
            self.ring_bytes()?,
            self.lookahead_bytes()?,
            "page store bytes",
        )
    }

    /// Pooled HIP events the timeline keeps (host handles, no device bytes).
    pub fn timeline_events(&self) -> usize {
        4 * self.timeline_records + 2 * self.timeline_layers
    }
}

/// The spans of the prompt arena, in layout order. The arena holds only what
/// the architecture (not the dispatch crate) owns across a layer: the
/// hyper-connection streams. The saved prepared input, top-k ids/weights,
/// route maps and widened accumulator are the dispatch crate's
/// `PromptRouteStorage` and are charged from its own layout
/// ([`route_storage_bytes`]), never duplicated here. Per-tile HC read/write
/// gates, normalised rows and the mixed row are tile-local scratch the 1,536-row
/// forward already owns (`HyperWriteOp` recomputes its gates from the saved
/// streams and the layer's combined output), so no gate is saved per window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaField {
    /// Four hyper-connection branch streams of every window row, each tile
    /// packed at its own tag width (BF16 from `QWEN4_F16_WMMA_MIN_TOKENS` rows
    /// up, F32 below): `(window + S1_F32_STREAM_ROWS + tile) * wide * 2` bytes,
    /// the last term slack for F32-typed validation views.
    HcStreams,
}

/// Every [`ArenaField`], in layout order.
pub const ARENA_FIELDS: [ArenaField; 1] = [ArenaField::HcStreams];

impl ArenaField {
    pub fn name(self) -> &'static str {
        match self {
            Self::HcStreams => "HC streams",
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

/// The architecture-owned prompt arena of one window: a single device
/// allocation of [`Self::total_bytes`], sliced by [`Self::span`]. The census
/// charges exactly this number and the allocator allocates exactly this number,
/// from this one function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PromptArenaLayout {
    pub rows: usize,
    pub tile_rows: usize,
    pub tiles: usize,
    pub spans: [ArenaSpan; ARENA_FIELDS.len()],
    pub total_bytes: u64,
}

/// Largest tile (in rows) whose streams stay F32: one row below the BF16 stream
/// predicate's threshold ([`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`]).
/// A change of that threshold changes the arena, never silently narrows it.
pub const S1_F32_TILE_ROWS_MAX: usize = rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS - 1;

/// Rows of one window whose tiles can be F32 under the BF16-stream predicate:
/// the window's final tile, one tile per [`S1_MAX_BOUNDARY_SNAPSHOTS`] session
/// cut segment boundary and the resumed entry head, each at most
/// [`S1_F32_TILE_ROWS_MAX`] rows: `(4 + 2) * 511 = 3066`. A tile of 512 or more
/// rows is BF16 on the production predicate; a plan that needs more F32 rows
/// bypasses before any mutation.
pub const S1_F32_STREAM_ROWS: usize = (S1_MAX_BOUNDARY_SNAPSHOTS + 2) * S1_F32_TILE_ROWS_MAX;

impl PromptArenaLayout {
    /// The layout of a window of `rows` prompt rows.
    pub fn new(geometry: &ModelGeometry, rows: usize, tile_rows: usize) -> Result<Self, String> {
        if rows == 0 || tile_rows == 0 {
            return Err("slow-link prefill arena has no rows".to_string());
        }
        // Each tile's streams are packed at their OWN tag width: BF16 (2 bytes)
        // for a tile of at least `QWEN4_F16_WMMA_MIN_TOKENS` rows on the
        // production predicate (`Gpu::qwen4_bf16_streams`), F32 below it. The
        // F32 tiles are the cut-bounded heads and tails of at most
        // `S1_F32_STREAM_ROWS` rows per window; each costs 2 bytes per element
        // more than BF16. One more tile of BF16 elements is slack: stream ops
        // validate an F32-typed view of `rows * wide` elements, which reaches
        // past a BF16 region by at most its own size.
        let bf16_bytes = 2u64;
        let wide = geometry.wide()?;
        let tagged_rows = (rows as u64)
            .checked_add(S1_F32_STREAM_ROWS as u64)
            .and_then(|total| total.checked_add(tile_rows as u64))
            .ok_or_else(|| "slow-link prefill arena rows overflow".to_string())?;
        let hc_streams = checked_mul(
            checked_mul(tagged_rows, wide, "arena row bytes")?,
            bf16_bytes,
            "arena bytes",
        )?;
        let sizes: [u64; ARENA_FIELDS.len()] = [hc_streams];
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

/// Window rows that can land in K1 short-tail tiles (at most the historic
/// `FEW_ROW_ROUTE_ROWS` rows each): the final tile of each of the
/// [`S1_MAX_BOUNDARY_SNAPSHOTS`] + 1 session-cut segments plus one short entry
/// tile of a resumed prompt, so at most 48 rows. A plan whose short tiles
/// exceed this is a pre-mutation bypass.
pub const S1_K1_TAIL_ROWS: usize =
    (S1_MAX_BOUNDARY_SNAPSHOTS + 2) * rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS;

/// The dispatch crate's route-storage dimensions of a window of `rows` rows.
pub fn route_dims(
    geometry: &ModelGeometry,
    rows: usize,
) -> hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteDims {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteDims {
        rows,
        hidden: geometry.hidden,
        n_exp: geometry.num_experts,
        mi: geometry.moe_intermediate,
        shared_intermediate: geometry.shared_intermediate,
        k1_rows: S1_K1_TAIL_ROWS,
    }
}

/// Device bytes of the dispatch crate's `PromptRouteStorage` for a window of
/// `rows` rows (saved prepared input, top-k ids and weights, route maps and the
/// widened accumulator): exactly
/// `PromptRouteLayout::total_bytes`, the one layout `PromptRouteStorage::alloc`
/// allocates from.
pub fn route_storage_bytes(geometry: &ModelGeometry, rows: usize) -> Result<u64, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        u64::try_from(layout.total_bytes())
            .map_err(|_| "slow-link prefill route storage overflows".to_string())
    })
}

/// Device bytes of the two window-wide A4 activation sidecars
/// (`int4_mmq_x_scratch`, `qwen4_moe_down_i4_scratch`) the single count barrier
/// quantizes into, from the dispatch crate's own layout.
pub fn a4_sidecar_bytes(geometry: &ModelGeometry, rows: usize) -> Result<u64, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        u64::try_from(layout.a4_sidecar_bytes())
            .map_err(|_| "slow-link prefill A4 sidecars overflow".to_string())
    })
}

/// Device bytes of the window-size grouped-scatter Gpu scratch
/// (`qwen4_moe_group_layout(rows * top-k, n_exp)`) the single count barrier
/// grows, from the dispatch crate's own layout. Excluded from
/// [`route_storage_bytes`] and [`a4_sidecar_bytes`].
pub fn group_scatter_bytes(geometry: &ModelGeometry, rows: usize) -> Result<u64, String> {
    hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&route_dims(
        geometry, rows,
    ))
    .map_err(|error| error.to_string())
    .and_then(|layout| {
        u64::try_from(layout.group_scatter_bytes())
            .map_err(|_| "slow-link prefill group scatter overflows".to_string())
    })
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
    /// The dispatch crate's `PromptRouteStorage`
    /// ([`route_storage_bytes`]: saved prepared input, top-k, route maps,
    /// widened accumulator).
    pub route_storage: u64,
    /// The window-wide A4 activation sidecars ([`a4_sidecar_bytes`]).
    pub a4_sidecars: u64,
    /// The window-size grouped-scatter Gpu scratch of the single count barrier
    /// ([`group_scatter_bytes`], `qwen4_moe_group_layout` at window slots): the
    /// FULL window figure, charged on top of the 1536-row tile workspace the
    /// base reserve already holds (the grown buffer frees the old one, so the
    /// replacement peak holds both).
    pub group_scatter: u64,
    /// The architecture-owned HC-stream arena ([`PromptArenaLayout`]).
    pub arena: u64,
    /// [`S1_MAX_BOUNDARY_SNAPSHOTS`] boundary stages of the target's session
    /// fixed parts, charged once each, plus the native-MTP entry `pending`
    /// hidden copy. The entry snapshot is the incumbent's own snapshot arena,
    /// already in the base reserve, and is never charged again.
    pub boundary_state: u64,
}

impl SlowlinkCensus {
    /// Every term, named, in log order.
    pub fn terms(&self) -> [(&'static str, u64); 7] {
        [
            ("expert ring", self.ring),
            ("expert lookahead", self.lookahead),
            ("route storage", self.route_storage),
            ("A4 sidecars", self.a4_sidecars),
            ("group scatter", self.group_scatter),
            ("HC arena", self.arena),
            ("boundary state", self.boundary_state),
        ]
    }

    /// Sum of every term, checked.
    pub fn total(&self) -> Result<u64, String> {
        self.terms()
            .into_iter()
            .try_fold(0u64, |total, (name, bytes)| {
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
    /// Layers the incumbent placement at the requested chunk would keep with
    /// the safety margin and no dedicated G2 stage charge (log only; see
    /// [`SlowlinkProfile::baseline_layers`]).
    pub baseline_layers: usize,
    /// Bytes of ONE boundary stage: the target's session fixed parts (GDN
    /// recurrent and conv, QSA selections, PLE conv, hyper feedback; no context
    /// rows, no MTP head), unpadded:
    /// [`crate::state::Qwen4State::session_fixed_stage_bytes`].
    pub fixed_state_bytes: u64,
    /// Entry `pending` hidden copy of a native-MTP request (0 for AR).
    pub pending_save_bytes: u64,
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
    /// Layers the incumbent placement at the requested prefill chunk
    /// (8192 rows on a narrow gfx1100 link) would keep with the safety margin
    /// but WITHOUT the dedicated G2 stage charge: a log-only no-stage baseline
    /// that can exceed the real incumbent's layer count. Never a measured or
    /// gating figure; the S1 floor ([`S1_MIN_RESIDENT_LAYERS`]) does not read it.
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

    /// The dispatch crate's route-storage dimensions at this window
    /// ([`route_dims`]): what the forward passes to
    /// `PromptRouteLayout::new` and `PromptRouteStorage::alloc`, the same call
    /// the census made.
    pub fn route_dims(&self) -> hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteDims {
        route_dims(&self.geometry, self.window_rows)
    }

    /// The expert page pool spec the forward constructs its ring from
    /// ([`PanelPageLayout::pool_spec`]), the same spec the census read.
    pub(crate) fn pool_spec(
        &self,
    ) -> Result<crate::gpu_forward::panel_stage::PanelPoolSpec, String> {
        self.pages.pool_spec()
    }

    /// Pinned host bytes the profile adds beside the host-mapped experts: the
    /// page pool's table staging (`PanelPoolReserve::staging`). Timeline events
    /// are pooled HIP handles ([`PanelPageLayout::timeline_events`]), no bytes.
    pub fn host_pinned_bytes(&self) -> u64 {
        self.pages.host_staging_bytes
    }

    /// The placement log line.
    pub fn describe(&self) -> String {
        format!(
            "slow-link prefill (S1): {} MiB charged [{}], {}-row tile, {}-row window, \
             {} resident layers (no-stage incumbent baseline {}), {}+{} expert pages of {} bytes ({} KiB pinned host table staging), link {:.1} GB/s",
            self.census.total().unwrap_or(0) >> 20,
            self.census.describe(),
            self.tile_rows,
            self.window_rows,
            self.resident_layers,
            self.baseline_layers,
            self.pages.active_pages,
            self.pages.pages_per_layer,
            self.pages.page_bytes,
            self.host_pinned_bytes() >> 10,
            self.link_gbps,
        )
    }
}

/// The layouts and the term-by-term census of an S1 profile for `inputs`,
/// exactly what [`admit`] charges and the forward allocates from. Pure; the
/// hold ledger reads it too, so no figure is computed twice.
pub fn census_of(
    inputs: &AdmissionInputs,
) -> Result<(PanelPageLayout, PromptArenaLayout, SlowlinkCensus), String> {
    let window_rows = inputs.window_rows;
    let pages = PanelPageLayout::new(
        inputs.geometry.num_experts,
        inputs.gate_up_layer_bytes,
        inputs.down_layer_bytes,
    )?;
    let arena = PromptArenaLayout::new(&inputs.geometry, window_rows, S1_TILE_ROWS)?;
    let census = SlowlinkCensus {
        ring: pages.ring_bytes()?,
        lookahead: pages.lookahead_bytes()?,
        route_storage: route_storage_bytes(&inputs.geometry, window_rows)?,
        a4_sidecars: a4_sidecar_bytes(&inputs.geometry, window_rows)?,
        group_scatter: group_scatter_bytes(&inputs.geometry, window_rows)?,
        arena: arena.total_bytes,
        boundary_state: checked_mul(
            S1_MAX_BOUNDARY_SNAPSHOTS as u64,
            inputs.fixed_state_bytes,
            "boundary state",
        )
        .and_then(|stages| checked_add(stages, inputs.pending_save_bytes, "boundary state"))?,
    };
    Ok((pages, arena, census))
}

/// The joint S1 placement: charge the tile, HC arena, route storage, A4 sidecars, page store
/// and boundary state together, then count the layers that remain. Refuses,
/// before anything is allocated, with the exact deficit when fewer than
/// [`S1_MIN_RESIDENT_LAYERS`] stay resident.
pub fn admit(inputs: &AdmissionInputs) -> Result<SlowlinkProfile, SlowlinkDecline> {
    let (pages, arena, census) = census_of(inputs).map_err(SlowlinkDecline::Geometry)?;
    let window_rows = inputs.window_rows;
    let charges = crate::expert_residency::PlacementCharges::new(inputs.layer_expert_bytes, false)
        .and_then(|charges| census.total().map(|bytes| charges.with_slowlink(bytes)))
        .map_err(SlowlinkDecline::Geometry)?;
    let reserve = charges
        .reserve(inputs.base_reserve)
        .map_err(SlowlinkDecline::Geometry)?;
    let layers = if inputs
        .non_expert
        .checked_add(reserve)
        .is_some_and(|need| need <= inputs.free)
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
            .ok_or_else(|| {
                SlowlinkDecline::Geometry("slow-link prefill deficit overflows".into())
            })?;
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

/// What a load knows once it has the manifest, the free device memory and the
/// two reserves, all of which the incumbent placement already computes. The
/// product loader and every example/gate build this the same way, so one
/// admission decides for all of them.
pub struct LoadFacts<'a> {
    pub config: &'a Qwen4Config,
    pub weights: &'a [hipfire_runtime::weight_manifest::WeightEntry],
    pub bytes_of: &'a dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    /// Free device bytes the placement is sized from.
    pub free: u64,
    /// [`crate::expert_residency::AutoReserve::bytes`] at [`S1_TILE_ROWS`].
    pub tile_reserve: u64,
    /// The same at the incumbent placement chunk (sizes the logged baseline
    /// layer count only).
    pub incumbent_reserve: u64,
    pub state_format: crate::Qwen4StateFormat,
    pub native_mtp: bool,
    pub window_rows: usize,
}

/// [`AdmissionInputs`] from [`LoadFacts`] and the measured narrow `link_gbps`:
/// the non-expert and per-layer expert bytes from the manifest, the packed
/// gate/up and down payload split, the exact session fixed snapshot bytes and
/// the baseline layer count.
pub fn load_admission_inputs(
    facts: &LoadFacts<'_>,
    link_gbps: f64,
) -> Result<AdmissionInputs, SlowlinkDecline> {
    use crate::expert_residency as residency;
    let geometry = ModelGeometry::from_config(facts.config).map_err(SlowlinkDecline::Geometry)?;
    let (non_expert, layer_expert_bytes) = residency::resident_split(facts.weights, facts.bytes_of)
        .map_err(SlowlinkDecline::Geometry)?;
    let baseline_layers = residency::PlacementCharges::new(layer_expert_bytes, false)
        .and_then(|charges| {
            residency::auto_placement_layers(
                facts.free,
                non_expert,
                layer_expert_bytes,
                facts.config.num_hidden_layers,
                facts.incumbent_reserve,
                &charges,
            )
        })
        .unwrap_or(0);
    let (gate_up_layer_bytes, down_layer_bytes) =
        residency::routed_layer_payload_split(facts.weights, facts.bytes_of)
            .map_err(SlowlinkDecline::Geometry)?;
    // One boundary stage = the TARGET's session fixed parts only (the head's
    // rows are not staged), exactly what `Qwen4Bundle::prompt_txn_stage_bytes`
    // allocates ([`crate::state::Qwen4State::session_fixed_stage_bytes`], the
    // unpadded sum of the same fixed-part allocator). Each new boundary stage
    // is charged once.
    let fixed_state_bytes =
        crate::state::Qwen4State::session_fixed_stage_bytes(facts.config, facts.state_format)
            .ok_or_else(|| {
                SlowlinkDecline::Geometry("session fixed stage bytes overflow".into())
            })?;
    // A native-MTP request also saves the entry `pending` hidden (one F32 wide
    // row, `hc_count * hidden * 4`) for rollback: a NEW buffer beyond the
    // existing pending owner (`Qwen4Bundle::slowlink_save_pending`).
    let pending_save_bytes = if facts.native_mtp {
        checked_mul(
            checked_mul(
                facts.config.hc_count as u64,
                facts.config.hidden_size as u64,
                "pending hidden",
            )
            .map_err(SlowlinkDecline::Geometry)?,
            4,
            "pending hidden",
        )
        .map_err(SlowlinkDecline::Geometry)?
    } else {
        0
    };
    Ok(AdmissionInputs {
        geometry,
        free: facts.free,
        non_expert,
        layer_expert_bytes,
        gate_up_layer_bytes,
        down_layer_bytes,
        trunk_layers: facts.config.num_hidden_layers,
        base_reserve: facts.tile_reserve,
        baseline_layers,
        fixed_state_bytes,
        pending_save_bytes,
        window_rows: facts.window_rows,
        link_gbps,
    })
}

/// Select the S1 profile for a load, or `Ok(None)` for the incumbent. Probe-free
/// checks run first; `link` (the host-link probe) runs only for a load that
/// passed them, and `inputs` (the memory query and census inputs) only for a
/// link that is narrow, so every other load does no S1 work and keeps its
/// placement byte for byte. A forced selector fails the load only on a capacity
/// refusal of an otherwise eligible load (exact byte deficit); every other
/// decline is the incumbent. Run it BEFORE the placement is sized: the profile
/// carries the charge ([`PlacementCharges::with_slowlink`]) and the resident
/// layers the placement must keep.
///
/// [`PlacementCharges::with_slowlink`]: crate::expert_residency::PlacementCharges::with_slowlink
pub fn select_profile(
    mode: SlowlinkMode,
    shape: &LoadShape<'_>,
    link: impl FnOnce() -> HostLink,
    inputs: impl FnOnce(f64) -> Result<AdmissionInputs, SlowlinkDecline>,
) -> Result<Option<SlowlinkProfile>, String> {
    let decline = match preflight(mode, shape) {
        Err(decline) => decline,
        Ok(()) => match link_verdict(link())
            .and_then(inputs)
            .and_then(|inputs| admit(&inputs))
        {
            Ok(profile) => {
                eprintln!("  qwen4 {}", profile.describe());
                return Ok(Some(profile));
            }
            Err(decline) => decline,
        },
    };
    if mode == SlowlinkMode::Force && decline.is_capacity() {
        return Err(format!(
            "qwen4: {SLOWLINK_PREFILL_ENV}=1 cannot be admitted: {decline}"
        ));
    }
    if decline != SlowlinkDecline::AutoNotQualified {
        eprintln!(
            "  qwen4 slow-link prefill (S1): not selected ({decline}); incumbent placement unchanged"
        );
    }
    Ok(None)
}

/// [`select_profile`] with [`load_admission_inputs`]: the one call the product
/// loader and the examples/gates make. `facts` runs only for a narrow link (it
/// queries free memory and builds the reserves), `link` only after the
/// probe-free checks passed. Returns the profile (with its resident layer
/// count) before any placement is sized.
pub fn select_load_profile<'a>(
    mode: SlowlinkMode,
    shape: &LoadShape<'_>,
    link: impl FnOnce() -> HostLink,
    facts: impl FnOnce() -> Result<LoadFacts<'a>, String>,
) -> Result<Option<SlowlinkProfile>, String> {
    select_profile(mode, shape, link, |link_gbps| {
        let facts = facts().map_err(SlowlinkDecline::Geometry)?;
        load_admission_inputs(&facts, link_gbps)
    })
}

/// The structural facts of a load that only its caller knows; the rest of a
/// [`LoadShape`] is read from the process snapshot
/// ([`LoadShape::from_process`]). Pass what is true: a load that applies a hot
/// set or `whole-first` must say so, never `false` to get a profile.
#[derive(Clone, Copy, Debug)]
pub struct StructuralShape {
    pub expert_parallel: bool,
    pub pipeline_band: bool,
    pub packed_mq4: bool,
    pub routed_experts: bool,
    pub expert_keep: bool,
    pub hotset_sidecar: bool,
    pub whole_first: bool,
}

impl<'a> LoadShape<'a> {
    /// The shape of a load whose process-level controls are read from the
    /// config snapshot exactly as the product loader reads them: an explicit
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, the `HIPFIRE_QWEN4_EXPERT_STAGE=0`
    /// opt-out and `memory.offload_exec`. `chunk_rows_env` is the raw
    /// `HIPFIRE_PREFILL_CHUNK_ROWS` value, when set.
    pub fn from_process(
        arch: &'a str,
        unified_memory: bool,
        structural: StructuralShape,
        max_seq: usize,
        chunk_rows_env: Option<&'a str>,
    ) -> Result<Self, String> {
        let explicit = crate::expert_residency::expert_vram_layers_from_env()
            .map_err(|error| format!("qwen4: {error}"))?;
        let stage = hipfire_config::developer_var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV).ok();
        Ok(Self {
            arch,
            unified_memory,
            expert_parallel: structural.expert_parallel,
            pipeline_band: structural.pipeline_band,
            packed_mq4: structural.packed_mq4,
            routed_experts: structural.routed_experts,
            expert_keep: structural.expert_keep,
            explicit_vram_layers: explicit.is_some(),
            hotset_sidecar: structural.hotset_sidecar,
            whole_first: structural.whole_first,
            stage_opt_out: stage.as_deref() == Some("0"),
            cpu_offload: hipfire_config::memory::offload_exec()
                != hipfire_config::memory::OffloadExec::Pcie,
            npu_trunk: npu_trunk_selected(
                arch,
                crate::npu_trunk::NpuTrunkMode::from_env()
                    .map_err(|error| format!("qwen4: {error}"))?,
            ),
            chunk_rows_env,
            max_seq,
        })
    }
}

/// A single-device load the S1 admission sizes: the same census inputs the
/// product loader hands the placement
/// ([`crate::expert_residency::placement_inputs`]).
pub struct SingleDeviceLoad<'a> {
    pub gpu: &'a rdna_compute::Gpu,
    pub config: &'a Qwen4Config,
    pub weights: &'a [hipfire_runtime::weight_manifest::WeightEntry],
    pub bytes_of: &'a dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    pub backend: crate::Qwen4KvBackend,
    pub max_seq: usize,
    pub vmm_granularity: usize,
    /// The native MTP head attaches (sizes the session fixed snapshot).
    pub native_mtp: bool,
    /// The head is still attached after the host-mapped placement (sizes the
    /// reserve): `native_mtp` and the host-spill policy kept it.
    pub mtp_kept: bool,
    pub max_k: usize,
    pub state_format: crate::Qwen4StateFormat,
    pub scratch_mode: crate::gpu_forward::Qwen4ForwardScratchMode,
}

/// Exact [`LoadFacts`] of a single-device discrete load: free device memory,
/// and the reserve at the S1 tile and at the incumbent placement chunk, both
/// from [`crate::expert_residency::placement_inputs`] (the product's own
/// bottom-up census; no second convention).
pub fn single_device_facts<'a>(load: &SingleDeviceLoad<'a>) -> Result<LoadFacts<'a>, String> {
    let (free, _) = load
        .gpu
        .device_mem_info()
        .map_err(|error| format!("qwen4: device memory query: {error}"))?;
    let routed_experts = !load.config.ffn.is_dense();
    let reserve_at = |chunk_rows: usize| {
        crate::expert_residency::placement_inputs(&crate::expert_residency::PlacementInputsArgs {
            gpu: load.gpu,
            config: load.config,
            weights: load.weights,
            bytes_of: load.bytes_of,
            backend: load.backend,
            use_ranges: false,
            max_seq: load.max_seq,
            vmm_granularity: load.vmm_granularity,
            mtp_kept: load.mtp_kept,
            max_k: load.max_k,
            routed_experts,
            state_format: load.state_format,
            scratch_mode: load.scratch_mode,
            ranks: 1,
            chunk_rows,
        })
        .map(|inputs| inputs.reserve.bytes())
    };
    // Beta's single incumbent helper (probes the host link once, cached).
    let placement_chunk = crate::gpu_forward::qwen4_prefill_chunk_requested(load.gpu, load.max_seq);
    Ok(LoadFacts {
        config: load.config,
        weights: load.weights,
        bytes_of: load.bytes_of,
        free: free as u64,
        tile_reserve: reserve_at(S1_TILE_ROWS)?,
        incumbent_reserve: reserve_at(placement_chunk)?,
        state_format: load.state_format,
        native_mtp: load.native_mtp,
        window_rows: S1_WINDOW_ROWS.min(load.max_seq),
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
    TooShort {
        rows: usize,
        minimum: usize,
    },
    AllRowCapture,
    ExternalCapture,
    TooManyBoundaries {
        cuts: usize,
        maximum: usize,
    },
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
            npu_trunk: false,
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
            pending_save_bytes: 4 * 2560 * 4,
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
        let arena =
            PromptArenaLayout::new(&inputs.geometry, inputs.window_rows, S1_TILE_ROWS).unwrap();
        pages.total_bytes().unwrap()
            + route_storage_bytes(&inputs.geometry, inputs.window_rows).unwrap()
            + a4_sidecar_bytes(&inputs.geometry, inputs.window_rows).unwrap()
            + group_scatter_bytes(&inputs.geometry, inputs.window_rows).unwrap()
            + arena.total_bytes
            + S1_MAX_BOUNDARY_SNAPSHOTS as u64 * inputs.fixed_state_bytes
            + inputs.pending_save_bytes
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
        for bad in [
            Some(f64::NAN),
            Some(0.0),
            Some(-3.0),
            Some(f64::INFINITY),
            None,
        ] {
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
        let cases: [(&str, LoadShape<'static>); 11] = [
            (
                "ep",
                LoadShape {
                    expert_parallel: true,
                    ..shape()
                },
            ),
            (
                "pp",
                LoadShape {
                    pipeline_band: true,
                    ..shape()
                },
            ),
            (
                "recipe",
                LoadShape {
                    packed_mq4: false,
                    ..shape()
                },
            ),
            (
                "dense",
                LoadShape {
                    routed_experts: false,
                    ..shape()
                },
            ),
            (
                "keep",
                LoadShape {
                    expert_keep: true,
                    ..shape()
                },
            ),
            (
                "layers",
                LoadShape {
                    explicit_vram_layers: true,
                    ..shape()
                },
            ),
            (
                "hotset",
                LoadShape {
                    hotset_sidecar: true,
                    ..shape()
                },
            ),
            (
                "whole-first",
                LoadShape {
                    whole_first: true,
                    ..shape()
                },
            ),
            (
                "stage off",
                LoadShape {
                    stage_opt_out: true,
                    ..shape()
                },
            ),
            (
                "cpu",
                LoadShape {
                    cpu_offload: true,
                    ..shape()
                },
            ),
            (
                "npu",
                LoadShape {
                    npu_trunk: true,
                    ..shape()
                },
            ),
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
                &LoadShape {
                    unified_memory: true,
                    ..shape()
                }
            ),
            Err(SlowlinkDecline::UnifiedMemory)
        );
        for max_seq in [0, 1, 1536] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &LoadShape { max_seq, ..shape() }),
                Err(SlowlinkDecline::ShortContext { max_seq })
            );
        }
        for max_seq in [1537, 3072] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &LoadShape { max_seq, ..shape() }),
                Ok(()),
                "{max_seq}"
            );
        }
    }

    #[test]
    fn slowlink_admission_selected_npu_truth_table() {
        use crate::npu_trunk::NpuTrunkMode::{Auto, Emulate, Off, On};
        // `auto` resolves to off outside gfx1151, so the default never selects
        // the NPU on the S1 arch; an explicit `on` or `emulate` does, anywhere.
        for arch in ["gfx1100", "gfx1201", "gfx906", "gfx1030", "gfx942"] {
            assert!(!npu_trunk_selected(arch, Off), "{arch}");
            assert!(!npu_trunk_selected(arch, Auto), "{arch}");
            assert!(npu_trunk_selected(arch, On), "{arch}");
            assert!(npu_trunk_selected(arch, Emulate), "{arch}");
        }
        assert!(npu_trunk_selected("gfx1151", Auto));
        assert!(!npu_trunk_selected("gfx1151", Off));
        // A selected NPU is an incumbent decline on an otherwise qualified
        // gfx1100 load, before any probe or reserve.
        let npu = |mode| LoadShape {
            npu_trunk: npu_trunk_selected("gfx1100", mode),
            ..shape()
        };
        assert_eq!(preflight(SlowlinkMode::Force, &npu(Auto)), Ok(()));
        for mode in [On, Emulate] {
            assert_eq!(
                preflight(SlowlinkMode::Force, &npu(mode)),
                Err(SlowlinkDecline::Unsupported("NPU trunk selected"))
            );
        }
    }

    #[test]
    fn slowlink_admission_explicit_chunk_rows_must_equal_the_tile() {
        let with = |env| LoadShape {
            chunk_rows_env: Some(env),
            ..shape()
        };
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
            assert_eq!(
                preflight(SlowlinkMode::Force, &with(env)),
                Ok(()),
                "{env:?}"
            );
        }
    }

    #[test]
    fn slowlink_admission_page_layout_is_the_pool_reserve() {
        let pages = PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER).unwrap();
        assert_eq!(pages.experts_per_page, 32);
        assert_eq!(pages.pages_per_layer, 16);
        assert_eq!(pages.total_pages(), 18);
        assert_eq!(pages.gate_up_expert_bytes, 1_740_800);
        assert_eq!(pages.down_expert_bytes, 870_400);
        // Payload bound: 32 experts' extents plus worst-case unmerged-run slack.
        let extents = 32 * 2_611_200u64;
        assert_eq!(pages.payload_bytes, extents + 64 * 2 * 255);
        let spec = pages.pool_spec().unwrap();
        assert_eq!(spec.active_slots, 2);
        assert_eq!(spec.lookahead_pages, 16);
        assert_eq!(spec.experts_per_slot, 32);
        assert_eq!(spec.num_experts, 512);
        assert_eq!(spec.payload_bytes as u64, pages.payload_bytes);
        let reserve = spec.reserve().unwrap();
        // ONE tail pad per page (rounded to 256) and two u64 pointer tables of
        // every expert per page, exactly the pool's own accounting.
        assert_eq!(
            pages.page_bytes,
            (pages.payload_bytes + 4096).div_ceil(256) * 256
        );
        assert_eq!(pages.table_bytes_per_page, 2 * 512 * 8);
        assert_eq!(reserve.page_bytes as u64, pages.page_bytes);
        assert_eq!(reserve.payload as u64, 18 * pages.page_bytes);
        assert_eq!(reserve.tables as u64, 18 * pages.table_bytes_per_page);
        assert_eq!(reserve.staging as u64, pages.host_staging_bytes);
        assert_eq!(pages.total_bytes().unwrap(), reserve.device_total as u64);
        assert_eq!(
            pages.ring_bytes().unwrap(),
            2 * pages.device_page_bytes().unwrap()
        );
        assert_eq!(
            pages.lookahead_bytes().unwrap(),
            16 * pages.device_page_bytes().unwrap()
        );
        // Plan: ~1.401 GiB of payload beside tables and pads.
        let payload = 18 * extents;
        assert!(
            payload > 1_500_000_000 && payload < 1_510_000_000,
            "{payload}"
        );
        assert_eq!(pages.timeline_events(), 4 * 2048 + 2 * 64);
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
    fn slowlink_admission_arena_is_the_hc_streams_only_and_exact() {
        let geometry = geometry();
        let arena = PromptArenaLayout::new(&geometry, S1_WINDOW_ROWS, S1_TILE_ROWS).unwrap();
        assert_eq!(arena.tiles, 11);
        assert_eq!(arena.rows, 16384);
        let wide = (geometry.hc_count * geometry.hidden) as u64;
        let rows = S1_WINDOW_ROWS as u64;
        // Saved prepared input, route maps and accumulator belong to the
        // dispatch crate's route storage, never duplicated in the arena.
        assert_eq!(ARENA_FIELDS, [ArenaField::HcStreams]);
        // Predicate pins: a tile of >= 512 rows is BF16; at most six
        // sub-512-row (F32) tiles per window; the arena changes with either.
        assert_eq!(S1_F32_TILE_ROWS_MAX, 511);
        assert_eq!(rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS, 512);
        assert_eq!(S1_F32_STREAM_ROWS, 6 * 511);
        assert_eq!(S1_F32_STREAM_ROWS, 3066);
        let bytes = (rows + 3066 + S1_TILE_ROWS as u64) * wide * 2;
        assert_eq!(arena.span(ArenaField::HcStreams).bytes, bytes);
        assert_eq!(arena.span(ArenaField::HcStreams).offset, 0);
        assert_eq!(arena.total_bytes % S1_ALIGN, 0);
        assert!(arena.total_bytes >= bytes);
        // Far below the all-F32 window, which a BF16-off load must bypass.
        assert!(arena.total_bytes < rows * wide * 4);
        assert!(arena.total_bytes - arena.span(ArenaField::HcStreams).end() < S1_ALIGN);
        // One window row more or fewer changes the allocation and the census alike.
        let shorter = PromptArenaLayout::new(&geometry, S1_WINDOW_ROWS - 1, S1_TILE_ROWS).unwrap();
        assert!(shorter.total_bytes < arena.total_bytes);
        assert!(PromptArenaLayout::new(&geometry, 0, S1_TILE_ROWS).is_err());
        assert!(PromptArenaLayout::new(&geometry, 1, 0).is_err());
        assert!(PromptArenaLayout::new(&geometry, usize::MAX / 2, S1_TILE_ROWS).is_err());
    }

    /// Hold 1 residency ledger on the REAL compact Flash-Next config (the
    /// persisted HFQ geometry): Full scratch, gfx1100 (no compact plan), a
    /// 16,640-token context, the 1,536-row tile, Q8 GDN and Q8 QSA, no MTP.
    /// Calls the actual census helpers (no reimplemented formula) and prints
    /// one structured JSON ledger (`--nocapture`). The legacy context is the
    /// discrete card's charge; the VMM 2 MiB row is conditional on the backend
    /// the load actually observes.
    #[test]
    fn slowlink_residency_hold1_ledger_for_the_real_mq4_fixture() {
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        use rdna_compute::tensor_ops::QsaKvFormat;
        let config = crate::config::compact_test_config();
        let format = crate::Qwen4StateFormat {
            qsa: QsaKvFormat::Q8,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        };
        let (max_seq, tile) = (16_640usize, S1_TILE_ROWS);

        // S1 layout: the known boundary stage and the known census total.
        let stage = crate::state::Qwen4State::session_fixed_stage_bytes(&config, format).unwrap();
        assert_eq!(stage, 34_128_016, "target-only boundary stage (no MTP)");
        let inputs = AdmissionInputs {
            geometry: ModelGeometry::from_config(&config).unwrap(),
            free: 0,
            non_expert: 0,
            layer_expert_bytes: GATE_UP_LAYER + DOWN_LAYER,
            gate_up_layer_bytes: GATE_UP_LAYER,
            down_layer_bytes: DOWN_LAYER,
            trunk_layers: config.num_hidden_layers,
            base_reserve: 0,
            baseline_layers: 0,
            fixed_state_bytes: stage,
            pending_save_bytes: 0,
            window_rows: S1_WINDOW_ROWS.min(max_seq),
            link_gbps: 7.09,
        };
        let (pages, arena, census) = census_of(&inputs).unwrap();
        let s1_total = census.total().unwrap();
        // Route storage includes D's `k1_shared` (245,760 B): 386,070,784 B,
        // charged with A4 and the grouped-scatter scratch at 471,924,992 B.
        assert_eq!(s1_total, 2_543_092_544, "S1 layout, no MTP");
        // Native MTP adds exactly the entry pending copy (hc * hidden * 4 B).
        let mtp_pending = (config.hc_count * config.hidden_size * 4) as u64;
        assert_eq!(mtp_pending, 40_960);
        let (_, _, mtp_census) = census_of(&AdmissionInputs {
            pending_save_bytes: mtp_pending,
            ..inputs
        })
        .unwrap();
        assert_eq!(mtp_census.total().unwrap(), 2_543_133_504, "S1 layout, native MTP");
        assert_eq!(census.boundary_state, 4 * 34_128_016);
        assert_eq!(pages.total_pages(), 18);

        // The incumbent's tile reserve terms from the real helpers.
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
        let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(tile, &dims)
            .expect("workspace terms");
        let workspace_total: u64 = workspace.iter().map(|(_, bytes)| *bytes).sum();
        let forward = crate::gpu_forward::qwen4_forward_device_bytes(&config, tile, Full).unwrap();
        let forward_census = crate::expert_residency::forward_census_scope(
            &config,
            tile,
            Full,
            &crate::weights::Qwen4BandScope::single(&config),
        )
        .unwrap();
        assert_eq!(forward_census.total().unwrap(), forward);
        let safety = crate::expert_residency::EXPERT_HOTSET_VRAM_MARGIN_BYTES;

        let mut contexts = Vec::new();
        for (label, backend, granularity, conditional) in [
            ("legacy", crate::Qwen4KvBackend::Legacy, 1usize, false),
            ("vmm_2mib", crate::Qwen4KvBackend::Vmm, 2usize << 20, true),
        ] {
            let census_context =
                crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
            let reserve_context = crate::Qwen4ContextCommit::for_expert_reserve(
                backend,
                false,
                max_seq,
                tile,
                granularity,
            );
            let state =
                crate::expert_residency::state_census(&config, format, &census_context, tile, Full)
                    .unwrap();
            let historical = crate::expert_residency::auto_vram_reserve(
                &config,
                &reserve_context,
                tile,
                Full,
                format.layout(),
                None,
                None,
            )
            .unwrap();
            contexts.push(serde_json::json!({
                "backend": label,
                "conditional_on_observed_backend": conditional,
                "qsa_context_bytes": state.qsa_context,
                "state_bytes": state.state,
                "forward_bytes": state.forward,
                "historical_reserve_bytes": historical,
            }));
        }

        let ledger = serde_json::json!({
            "fixture": {
                "config": "compact_test_config",
                "arch": "gfx1100",
                "scratch_mode": "Full",
                "max_seq": max_seq,
                "tile_rows": tile,
                "window_rows": inputs.window_rows,
                "state_format": { "qsa": "q8", "gdn": "q8" },
                "native_mtp": false,
            },
            "tile": {
                "forward_device_bytes": forward,
                "forward_census": {
                    "stage_shared": forward_census.stage_shared,
                    "band_local": forward_census.band_local,
                    "input_only": forward_census.input_only,
                    "output_only": forward_census.output_only,
                },
                "workspace_total_bytes": workspace_total,
                "workspace_terms": workspace
                    .iter()
                    .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                    .collect::<Vec<_>>(),
            },
            "contexts": contexts,
            "safety_margin_bytes": safety,
            "s1": {
                "total_bytes": s1_total,
                "terms": census
                    .terms()
                    .iter()
                    .map(|(name, bytes)| serde_json::json!({ "name": name, "bytes": bytes }))
                    .collect::<Vec<_>>(),
                "boundary_stage_bytes": stage,
                "boundary_stages": S1_MAX_BOUNDARY_SNAPSHOTS,
                "hc_arena_bytes": arena.total_bytes,
                "page_payload_bound_bytes": pages.payload_bytes,
                "page_bytes": pages.page_bytes,
                "page_count": pages.total_pages(),
                "pinned_host_staging_bytes": pages.host_staging_bytes,
            },
        });
        println!(
            "SLOWLINK_RESIDENCY_LEDGER {}",
            serde_json::to_string(&ledger).unwrap()
        );
    }

    /// The fixture every layout test sizes from is the real MQ4 model, not a
    /// smaller synthetic one: the H0 MQ4 receipt's layer identity is gate/up
    /// `[512, 1280, 2560]` and down `[512, 2560, 640]` (H2560, expert
    /// intermediate 640, NOT 512), 1,740,800 / 870,400 payload bytes per expert.
    #[test]
    fn slowlink_admission_fixture_is_the_h0_mq4_layer_identity() {
        let geometry = geometry();
        assert_eq!(geometry.hidden, 2560);
        assert_eq!(geometry.moe_intermediate, 640);
        assert_eq!(geometry.num_experts, 512);
        assert_eq!(geometry.experts_per_token, 10);
        assert_eq!(geometry.hc_count, 4);
        assert_eq!(geometry.shared_intermediate, 640);
        // gate/up is [n_exp, 2 * mi, hidden], down [n_exp, hidden, mi].
        assert_eq!(2 * geometry.moe_intermediate, 1280);
        assert_eq!(GATE_UP_LAYER, 891_289_600);
        assert_eq!(DOWN_LAYER, 445_644_800);
        let pages = PanelPageLayout::new(512, GATE_UP_LAYER, DOWN_LAYER).unwrap();
        assert_eq!(pages.gate_up_expert_bytes, 1_740_800);
        assert_eq!(pages.down_expert_bytes, 870_400);
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
        // The pool's own reserve is the page store: one authority.
        let reserve = profile.pages.pool_spec().unwrap().reserve().unwrap();
        assert_eq!(
            profile.pages.total_bytes().unwrap(),
            reserve.device_total as u64
        );
        assert_eq!(profile.pages.host_staging_bytes, reserve.staging as u64);
        assert_eq!(reserve.pages, 18);
        assert_eq!(
            profile.census.route_storage,
            route_storage_bytes(&profile.geometry, profile.window_rows).unwrap()
        );
        assert_eq!(
            profile.census.a4_sidecars,
            a4_sidecar_bytes(&profile.geometry, profile.window_rows).unwrap()
        );
        assert!(profile.census.a4_sidecars > 0 && profile.census.route_storage > 0);
        assert_eq!(
            profile.census.group_scatter,
            group_scatter_bytes(&profile.geometry, profile.window_rows).unwrap()
        );
        assert!(profile.census.group_scatter > 0);
        assert_eq!(profile.reserve_bytes().unwrap(), census_total(&inputs));
        assert_eq!(profile.tile_rows, 1536);
        assert!(profile.tile_rows <= 1536);
        assert_eq!(profile.window_rows, 16384);
        assert_eq!(profile.census.terms().len(), 6);
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
            Err(SlowlinkDecline::NoHostLayers {
                resident_layers: 48
            })
        ));
        assert!(!SlowlinkDecline::NoHostLayers {
            resident_layers: 48
        }
        .is_capacity());
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
        assert_eq!(
            charged.reserve(123).unwrap(),
            123 + incumbent.margin + 4 * GIB
        );
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
        // Forced: from one row past a tile up (the first genuinely multi-chunk
        // request); a request of at most one tile is a named bypass.
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(1536)),
            Err(RequestBypass::TooShort {
                rows: 1536,
                minimum: 1537
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(1537)),
            Ok(RequestVerdict {
                rows: 1537,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(3072)),
            Ok(RequestVerdict {
                rows: 3072,
                windows: 1
            })
        );
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(16384)),
            Ok(RequestVerdict {
                rows: 16384,
                windows: 1
            })
        );
        // A prompt past the window runs superwindows and says so.
        assert_eq!(
            qualify_request(SlowlinkMode::Force, &profile, &req(16385)),
            Ok(RequestVerdict {
                rows: 16385,
                windows: 2
            })
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
                    &RequestShape {
                        capture,
                        ..req(16384)
                    }
                ),
                Err(bypass)
            );
        }
        for capture in [CaptureKind::None, CaptureKind::Bounded] {
            assert!(qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape {
                    capture,
                    entry_pos: 4096,
                    ..req(16384)
                }
            )
            .is_ok());
        }
        assert_eq!(
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape {
                    session_cuts: 4,
                    ..req(16384)
                }
            )
            .unwrap()
            .windows,
            1
        );
        assert_eq!(
            qualify_request(
                SlowlinkMode::Force,
                &profile,
                &RequestShape {
                    session_cuts: 5,
                    ..req(16384)
                }
            ),
            Err(RequestBypass::TooManyBoundaries {
                cuts: 5,
                maximum: 4
            })
        );
        assert_eq!(S1_MIN_AUTO_REQUEST_ROWS, 16384);
        assert_eq!(profile.windows_for(1), 1);
        assert_eq!(profile.windows_for(32768), 2);
    }

    #[test]
    fn slowlink_admission_route_dims_are_the_census_dims() {
        let profile = profile_at_ten();
        let dims = profile.route_dims();
        assert_eq!(dims.rows, profile.window_rows);
        assert_eq!(dims.hidden, profile.geometry.hidden);
        assert_eq!(dims.n_exp, 512);
        assert_eq!(dims.mi, profile.geometry.moe_intermediate);
        assert_eq!(
            dims.shared_intermediate,
            profile.geometry.shared_intermediate
        );
        // At most (4 session-cut segments + 1 entry tile + the window tail) short tiles of 8 rows.
        assert_eq!(dims.k1_rows, 48);
        assert_eq!(S1_K1_TAIL_ROWS, 48);
        let layout =
            hipfire_dispatch::pipeline::slowlink_prefill::PromptRouteLayout::new(&dims).unwrap();
        assert_eq!(
            profile.census.route_storage,
            layout.total_bytes() as u64,
            "census and allocation read one layout"
        );
        assert_eq!(profile.census.a4_sidecars, layout.a4_sidecar_bytes() as u64);
        assert_eq!(profile.census.group_scatter, layout.group_scatter_bytes() as u64);
        // Full window grouped-scatter scratch at N=16384 slots=163840, E=512:
        // the 1536-tile's 309,248 B stays in the base reserve (replacement
        // peak holds old + new), so the full figure is charged on top.
        assert_eq!(profile.census.group_scatter, 3_278_848);
    }

    #[test]
    fn slowlink_admission_shared_helper_builds_nothing_for_a_load_that_cannot_select() {
        use std::cell::Cell;
        let facts_built = Cell::new(false);
        let facts = || -> Result<LoadFacts<'static>, String> {
            facts_built.set(true);
            Err("facts must not be built".to_string())
        };
        // Killed, unqualified-auto and ineligible loads: no probe, no facts.
        for (mode, load) in [
            (SlowlinkMode::Off, shape()),
            (SlowlinkMode::Auto, shape()),
            (
                SlowlinkMode::Force,
                LoadShape {
                    arch: "gfx1201",
                    ..shape()
                },
            ),
            (
                SlowlinkMode::Force,
                LoadShape {
                    explicit_vram_layers: true,
                    ..shape()
                },
            ),
        ] {
            let outcome = select_load_profile(mode, &load, || unreachable!("probed"), facts);
            assert_eq!(outcome, Ok(None));
            assert!(!facts_built.get());
        }
        // A measured fast link probes once and still builds no facts.
        for link in [
            HostLink::Discrete { gbps: 16.0 },
            HostLink::Discrete { gbps: 26.0 },
            HostLink::Unified,
            HostLink::Unmeasured,
        ] {
            let probes = Cell::new(0);
            let outcome = select_load_profile(
                SlowlinkMode::Force,
                &shape(),
                || {
                    probes.set(probes.get() + 1);
                    link
                },
                facts,
            );
            assert_eq!(outcome, Ok(None), "{link:?}");
            assert_eq!(probes.get(), 1);
            assert!(!facts_built.get(), "{link:?}");
        }
        // A narrow link asks for the facts, and a facts failure is a capacity
        // refusal of an eligible forced load, never a silent incumbent.
        let error = select_load_profile(
            SlowlinkMode::Force,
            &shape(),
            || HostLink::Discrete { gbps: 7.09 },
            facts,
        )
        .unwrap_err();
        assert!(facts_built.get());
        assert!(error.contains("facts must not be built"), "{error}");
    }
}
