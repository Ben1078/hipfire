// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact lane step (Flash-Next lane batching, fn-batch S2 AR rows, S5 MTP
//! verify rows).
//!
//! One trunk step over several admitted requests ("lanes"), each with its
//! own [`Qwen4State`]. Per lane, the committed state and logits must be
//! byte-identical to that request run alone on the singleton route the lane
//! takes: an AR row is `forward_token_or_argmax`, a verify block
//! `[seed, drafts..]` is the native MTP window's `verify_block` (armed GDN
//! ring, wide-hidden capture). The step is built from the singleton's own
//! typed layer operations with one rule:
//!
//! - **Stateful stages** (PLE convolution, GDN conv/recurrence/gate, QSA
//!   prologue/select/attend) always run per lane, on the lane's state, with
//!   the lane's own row count.
//! - **Row-local stages** run either per lane ([`StageShare::PerLane`], the
//!   singleton's arm by construction) or once over every lane's rows
//!   ([`StageShare::OnceAllRows`], in groups of at most [`LANE_MAX_ROWS`]).
//!   A stage moves to `OnceAllRows` only on byte evidence from the G0 oracle
//!   (`examples/fn_lane_oracle.rs --g0`): the S0 kill experiment found the
//!   few-row forward NOT bitwise equal to single rows at every N ≥ 2, so
//!   nothing is assumed row-count independent. `PerClass` executes as
//!   `PerLane`: a class-A (one-row) lane's singleton arm is the one-row
//!   kernel, and no existing kernel runs that arm over several rows.
//!
//! The GDN and QSA mixers split ([`MixerPhase`]) into their row-local input
//! projections, the per-lane state core, and the row-local output
//! projection, so the projections can share one launch while the core stays
//! per lane. Consecutive stages with the same share are emitted
//! segment-major (all of a lane's per-lane ops back to back, or all of a row
//! group's ops), so every lane sees the singleton's operation sequence, and
//! the step interpreter's fusions apply exactly where the singleton's do.
//!
//! Lane steps run eager HIP (no retained tape) and never touch the resident
//! `bundle.state` / `bundle.mtp`; QSA arenas must already cover the step
//! (mapping happens in the executor's `provision_step`, never here).
//!
//! # Prefill wave
//!
//! The same step also runs a **prefill wave** ([`forward_wave`]): every lane
//! carries at least one row ([`LaneRowKind::Prefill`]; a step is all AR or all
//! prefill), and a once-over-all-rows launch covers EVERY row of the wave in
//! one segment (not [`LANE_MAX_ROWS`] groups). Stateful stages still run per
//! lane on the lane's own rows and state, so under the invariant prefill
//! profile (`Gpu::prefill_invariant_scope`, which routes every row-count
//! decision as the canonical chunk) a lane's committed state and head bytes
//! equal that request's singleton prefill. The wave head is per lane (the
//! singleton's `execute_lm_head` forms), never grouped: [`WaveHead::Final`]
//! writes one logits row (+ top-1) per lane from the lane's last row,
//! [`WaveHead::All`] writes every row of every lane. The share policy's
//! `Head` / `Argmax` entries are not consulted by a wave.

use crate::bundle::Qwen4Bundle;
use crate::config::LayerType;
use crate::gpu_forward::{
    decode_copy, dense_ref, hyper_read_desc, invalid, layer_desc, matrix_view, ple_desc,
    program_dims, qwen4_ple_rows_f16, router_columns, view, Qwen4GpuForward,
    Qwen4GpuForwardError, Qwen4GpuForwardScratch, ScratchField, DECODE_Q8_PER_LAYER, EPSILON,
};
use crate::ple_stage::ple_async_upload;
use crate::program::{
    execute_final_hyper, execute_lm_head, seal_moe_for_layer, validate_final_hyper,
    validate_lm_head, Qwen4AttentionWeights, Qwen4FfnBinding, Qwen4LayerDescription,
    Qwen4LayerScratch, Qwen4ProgramDims,
};
use crate::state::Qwen4State;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, validate_steps, ClearOp, EmbeddingOp,
    GatedDeltaNetOp, GroupedDepthwiseOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode,
    IndexedAttentionOp, IndexedAttentionState, MixerPhase, Step,
};
use hipfire_runtime::external_rows::RowFetch;
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};

/// A stage of the lane step that the share policy places.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageName {
    /// Token embedding and the HC stream broadcast.
    Embed,
    /// PLE row conversion (the host fetch and upload are one per step).
    PleRows,
    /// Every layer's attention and MLP hyper-connection reads.
    HcRead,
    /// Every layer's attention and MLP hyper-connection writes.
    HcWrite,
    /// PLE grouped convolution (stateful).
    PleConv,
    /// GDN qkv / a / b / z projections.
    GdnInProj,
    /// GDN convolution, recurrence and gate (stateful).
    GdnCore,
    /// GDN output projection.
    GdnOutProj,
    /// QSA indexer / q / k / v projections.
    QsaInProj,
    /// QSA prologue, pooling, selection and attention (stateful).
    QsaCore,
    /// QSA output projection.
    QsaOutProj,
    /// Router, routed experts and shared expert (one sealed call).
    Moe,
    /// Final hyper-connection read.
    FinalHyper,
    /// Language-model head.
    Head,
    /// Greedy top-1.
    Argmax,
}

impl StageName {
    pub const ALL: [StageName; 15] = [
        StageName::Embed,
        StageName::PleRows,
        StageName::HcRead,
        StageName::HcWrite,
        StageName::PleConv,
        StageName::GdnInProj,
        StageName::GdnCore,
        StageName::GdnOutProj,
        StageName::QsaInProj,
        StageName::QsaCore,
        StageName::QsaOutProj,
        StageName::Moe,
        StageName::FinalHyper,
        StageName::Head,
        StageName::Argmax,
    ];

    pub fn key(self) -> &'static str {
        match self {
            StageName::Embed => "embed",
            StageName::PleRows => "ple_rows",
            StageName::HcRead => "hc_read",
            StageName::HcWrite => "hc_write",
            StageName::PleConv => "ple_conv",
            StageName::GdnInProj => "gdn_in",
            StageName::GdnCore => "gdn_core",
            StageName::GdnOutProj => "gdn_out",
            StageName::QsaInProj => "qsa_in",
            StageName::QsaCore => "qsa_core",
            StageName::QsaOutProj => "qsa_out",
            StageName::Moe => "moe",
            StageName::FinalHyper => "final_hyper",
            StageName::Head => "head",
            StageName::Argmax => "argmax",
        }
    }

    pub fn parse(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|stage| stage.key() == key)
    }

    /// Stages that read or write request state: always per lane.
    pub fn stateful(self) -> bool {
        matches!(
            self,
            StageName::PleConv | StageName::GdnCore | StageName::QsaCore
        )
    }

    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|stage| *stage == self)
            .expect("every stage is listed in ALL")
    }
}

/// Where a stage's launches run (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageShare {
    /// One launch over every lane's rows, in groups of at most
    /// [`LANE_MAX_ROWS`] rows. Admitted only with G0 byte evidence.
    OnceAllRows,
    /// One launch per lane class; executes as [`StageShare::PerLane`] with
    /// the existing kernels.
    PerClass,
    /// One launch per lane with the lane's own row count.
    PerLane,
}

impl StageShare {
    fn key(self) -> &'static str {
        match self {
            StageShare::OnceAllRows => "once",
            StageShare::PerClass => "class",
            StageShare::PerLane => "lane",
        }
    }

    fn parse(key: &str) -> Option<Self> {
        match key {
            "once" => Some(StageShare::OnceAllRows),
            "class" => Some(StageShare::PerClass),
            "lane" => Some(StageShare::PerLane),
            _ => None,
        }
    }

    fn per_lane(self) -> bool {
        self != StageShare::OnceAllRows
    }
}

/// Rows a once-over-all-rows launch covers at most: the few-row kernel arms
/// (MQ6 / Q8 / BF16 row GEMVs, the Q8 decode copies) switch above 8 rows.
pub const LANE_MAX_ROWS: usize = 8;

/// Production share policy, from the G0 oracle on gfx1151 ([`STAGE_EVIDENCE`]):
/// every row-local stage — embedding, PLE rows, HC read/write, the GDN/QSA
/// projections, the sealed MoE (the grouped few-row route over ≤ 8 AR rows is
/// byte-identical per row to the one-row indexed decode), final HC read, head
/// and argmax — produced the per-lane bytes once over all rows; the stateful
/// cores stay per lane. S0's chunk-vs-singles mismatch therefore lives in the
/// multi-row GDN/QSA cores of consecutive positions, which a lane never runs.
pub const STAGE_SHARE: [(StageName, StageShare); 15] = [
    (StageName::Embed, StageShare::OnceAllRows),
    (StageName::PleRows, StageShare::OnceAllRows),
    (StageName::HcRead, StageShare::OnceAllRows),
    (StageName::HcWrite, StageShare::OnceAllRows),
    (StageName::PleConv, StageShare::PerLane),
    (StageName::GdnInProj, StageShare::OnceAllRows),
    (StageName::GdnCore, StageShare::PerLane),
    (StageName::GdnOutProj, StageShare::OnceAllRows),
    (StageName::QsaInProj, StageShare::OnceAllRows),
    (StageName::QsaCore, StageShare::PerLane),
    (StageName::QsaOutProj, StageShare::OnceAllRows),
    (StageName::Moe, StageShare::OnceAllRows),
    (StageName::FinalHyper, StageShare::OnceAllRows),
    (StageName::Head, StageShare::OnceAllRows),
    (StageName::Argmax, StageShare::OnceAllRows),
];

/// G0 oracle receipt behind [`STAGE_SHARE`] (`fn-batch/S3-RESULT.md`): each
/// row-local stage flipped alone at k=2 and k=8, then the combined policy at
/// k=1/2/8, contexts 512/2048, per-step logits, ids and full state bytes
/// identical to the singleton.
pub const STAGE_EVIDENCE: &str = "g0 fn-batch-s3/j2-g0 2026-10-09 gfx1151";

/// The only architecture [`STAGE_EVIDENCE`] covers. Other cards bind other
/// kernels (no Q8_0 decode copies on gfx1201) and keep every stage per lane
/// until their own G0 run.
pub const STAGE_EVIDENCE_ARCH: &str = "gfx1151";

/// One share per [`StageName`], indexed in [`StageName::ALL`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneStagePolicy {
    shares: [StageShare; 15],
}

impl LaneStagePolicy {
    /// [`Self::production`] on the evidenced architecture, else every stage
    /// per lane.
    pub fn for_arch(arch: &str) -> Self {
        if arch == STAGE_EVIDENCE_ARCH {
            Self::production()
        } else {
            Self::all_per_lane()
        }
    }

    pub fn production() -> Self {
        let mut shares = [StageShare::PerLane; 15];
        for (stage, share) in STAGE_SHARE {
            shares[stage.index()] = share;
        }
        Self { shares }
    }

    pub fn all_per_lane() -> Self {
        Self {
            shares: [StageShare::PerLane; 15],
        }
    }

    pub fn share(&self, stage: StageName) -> StageShare {
        self.shares[stage.index()]
    }

    pub fn with(mut self, stage: StageName, share: StageShare) -> Result<Self, String> {
        if stage.stateful() && share != StageShare::PerLane {
            return Err(format!(
                "stage {} reads request state and always runs per lane",
                stage.key()
            ));
        }
        self.shares[stage.index()] = share;
        Ok(self)
    }

    /// `production` | `per-lane` | `<base>:<stage>=once|class|lane,...`
    /// with `<base>` one of the two named policies.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (base, overrides) = match spec.split_once(':') {
            Some((base, rest)) => (base, Some(rest)),
            None => (spec, None),
        };
        let mut policy = match base.trim() {
            "production" => Self::production(),
            "per-lane" => Self::all_per_lane(),
            other => return Err(format!("unknown lane policy base {other:?}")),
        };
        for item in overrides
            .into_iter()
            .flat_map(|rest| rest.split(','))
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            let (key, value) = item
                .split_once('=')
                .ok_or_else(|| format!("lane policy item {item:?} is not stage=share"))?;
            let stage = StageName::parse(key.trim())
                .ok_or_else(|| format!("unknown lane stage {key:?}"))?;
            let share = StageShare::parse(value.trim())
                .ok_or_else(|| format!("unknown lane share {value:?}"))?;
            policy = policy.with(stage, share)?;
        }
        Ok(policy)
    }

    pub fn describe(&self) -> String {
        StageName::ALL
            .iter()
            .map(|stage| format!("{}={}", stage.key(), self.share(*stage).key()))
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn evidence(&self) -> &'static str {
        if *self == Self::production() {
            STAGE_EVIDENCE
        } else {
            ""
        }
    }

    fn validate(&self) -> Result<(), String> {
        for stage in StageName::ALL {
            if stage.stateful() && self.share(stage) != StageShare::PerLane {
                return Err(format!("stage {} must run per lane", stage.key()));
            }
        }
        Ok(())
    }
}

/// What a lane's rows do this step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRowKind {
    /// One AR row (the lane's pending seed); no GDN row capture.
    Ar,
    /// An MTP verify block `[seed, drafts..]` of `1 + draft_len` rows
    /// (`2..=LANE_MAX_ROWS`), run under the lane's armed GDN ring exactly as
    /// the singleton `verify_block` runs it.
    Verify { draft_len: usize },
    Prefill,
}

/// One lane's rows in a step.
pub struct LaneRows<'a> {
    pub state: &'a mut Qwen4State,
    pub tokens: &'a [u32],
    pub kind: LaneRowKind,
}

/// The step's outputs; row `r` belongs to the lanes concatenated in order.
pub struct LaneOutputs<'a> {
    /// F32, at least `total_rows * vocab` elements.
    pub logits: &'a GpuTensor,
    /// Raw, at least `total_rows * 4` bytes (one i32 per row).
    pub top1: &'a GpuTensor,
    /// F32, at least `total_rows * hc_count * hidden` elements: every row's
    /// final HC streams (the trunk output before the final hyper read),
    /// exactly what the singleton's wide-hidden capture writes.
    pub wide_hidden: Option<&'a GpuTensor>,
}

/// Which logits a prefill wave writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveHead {
    /// One logits row (+ top-1) per lane, in lane order, from the lane's last
    /// row (the singleton's `Qwen4OutputRows::Final`).
    Final,
    /// Every row of every lane, in concatenated order (lab only; small totals).
    All,
}

/// One lane's rows in a prefill wave.
pub struct WaveLane<'a> {
    pub state: &'a mut Qwen4State,
    pub tokens: &'a [u32],
}

/// A prefill wave's outputs: F32 `lanes * vocab` ([`WaveHead::Final`]) or
/// `total_rows * vocab` ([`WaveHead::All`]) logits, and a Raw i32 per output
/// row for the optional top-1.
pub(crate) struct WaveOutputs<'a> {
    head: WaveHead,
    logits: &'a GpuTensor,
    top1: Option<&'a GpuTensor>,
}

/// Outputs of a lane step: one logits row and top-1 per AR row, or a prefill
/// wave's head.
pub(crate) enum StepOutputs<'a> {
    Ar(&'a LaneOutputs<'a>),
    Wave(WaveOutputs<'a>),
}

impl<'a> StepOutputs<'a> {
    /// The lane step's wide-hidden capture (verify lanes); a prefill wave has none.
    fn wide_hidden(&self) -> Option<&'a GpuTensor> {
        match self {
            Self::Ar(out) => out.wide_hidden,
            Self::Wave(_) => None,
        }
    }
}

/// A contiguous row range of the step: one lane, or a group of lanes a
/// once-over-all-rows launch covers.
#[derive(Clone, Copy, Debug)]
struct Segment {
    begin: usize,
    rows: usize,
    /// Lane whose state a projection-only mixer op names (it never reads
    /// it; the op type requires one).
    first_lane: usize,
}

fn lane_segments(rows: &[usize]) -> Vec<Segment> {
    let mut begin = 0;
    rows.iter()
        .enumerate()
        .map(|(lane, &rows)| {
            let segment = Segment {
                begin,
                rows,
                first_lane: lane,
            };
            begin += rows;
            segment
        })
        .collect()
}

/// Consecutive lanes packed into groups of at most [`LANE_MAX_ROWS`] rows (a
/// lane wider than that is its own group).
fn group_segments(rows: &[usize]) -> Vec<Segment> {
    let mut groups: Vec<Segment> = Vec::new();
    let mut begin = 0;
    for (lane, &lane_rows) in rows.iter().enumerate() {
        match groups.last_mut() {
            Some(group) if group.rows + lane_rows <= LANE_MAX_ROWS => group.rows += lane_rows,
            _ => groups.push(Segment {
                begin,
                rows: lane_rows,
                first_lane: lane,
            }),
        }
        begin += lane_rows;
    }
    groups
}

/// The once-over-all-rows groups a step of decode lanes with these per-lane
/// row counts runs ([`StageShare::OnceAllRows`] stages launch once per
/// group).
pub(crate) fn row_group_count(rows: &[usize]) -> usize {
    group_segments(rows).len()
}

/// The row views one segment's operations bind: every row-indexed scratch
/// tensor offset to the segment's first row, extending to the end of its
/// allocation (exactly what a singleton forward of these rows sees from row
/// zero). Scratch an operation consumes internally stays at row zero.
struct SegViews {
    streams: GpuTensor,
    hc_normalized: GpuTensor,
    hc_low: GpuTensor,
    hc_up: GpuTensor,
    hc_mixed: GpuTensor,
    hc_gates: GpuTensor,
    projection: GpuTensor,
    projection2: GpuTensor,
    gdn_a: GpuTensor,
    gdn_b: GpuTensor,
    gdn_gate: GpuTensor,
    gdn_beta: GpuTensor,
    gdn_recurrent_output: GpuTensor,
    gdn_z: GpuTensor,
    gdn_output: GpuTensor,
    qsa_index: GpuTensor,
    qsa_qgate: GpuTensor,
    qsa_k: GpuTensor,
    qsa_v: GpuTensor,
    qsa_output: GpuTensor,
    attention_output: GpuTensor,
    ffn_output: GpuTensor,
    router_logits: GpuTensor,
    ple_rows: GpuTensor,
    ple_query: GpuTensor,
    ple_key: GpuTensor,
    ple_value: GpuTensor,
    ple_gated: GpuTensor,
    ple_normed: GpuTensor,
    ple_output: GpuTensor,
    embeddings: GpuTensor,
    embedding_rot: GpuTensor,
    token_ids: GpuTensor,
    logits: GpuTensor,
    top1: GpuTensor,
}

/// Segment views of the scratch.  Every row-banded field comes from
/// [`Qwen4GpuForwardScratch::segment_view`]: a dedicated field is addressed at
/// its logical row width, a field of an alias set (compact scratch) at the row
/// pitch of the allocation that owns its storage, so a segment occupies the
/// same rows of the owner whichever member of the set a phase uses.
fn seg_rows(
    scratch: &Qwen4GpuForwardScratch,
    field: ScratchField,
    seg: Segment,
    logical_width: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    scratch.segment_view(field, seg.begin, seg.rows, logical_width)
}

impl SegViews {
    fn new(
        scratch: &Qwen4GpuForwardScratch,
        dims: Qwen4ProgramDims,
        vocab: usize,
        out: Option<&LaneOutputs<'_>>,
        seg: Segment,
        ple_embed_dim: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let b = seg.begin;
        let wide = dims.wide();
        let hidden = dims.hidden;
        let value_heads = dims.linear_num_value_heads;
        // A prefill wave runs its head per lane through its own views: its
        // segment views carry no logits (an empty view of a scratch tensor).
        let (logits, top1) = match out {
            Some(out) => (
                view(out.logits, b * vocab, seg.rows * vocab),
                view(
                    out.top1,
                    b * std::mem::size_of::<i32>(),
                    seg.rows * std::mem::size_of::<i32>(),
                ),
            ),
            None => (view(&scratch.token_ids, 0, 0), view(&scratch.token_ids, 0, 0)),
        };
        use ScratchField as F;
        let rows = |field: F, width: usize| seg_rows(scratch, field, seg, width);
        Ok(Self {
            streams: rows(F::Streams, wide)?,
            hc_normalized: rows(F::HcNormalized, wide)?,
            hc_low: rows(F::HcLow, dims.hc_lowrank)?,
            hc_up: rows(F::HcUp, wide)?,
            hc_mixed: rows(F::HcMixed, hidden)?,
            hc_gates: rows(F::HcGates, dims.hc_count)?,
            projection: rows(F::Projection, dims.gdn_qkv())?,
            projection2: rows(F::Projection2, dims.gdn_qkv())?,
            gdn_a: rows(F::GdnA, value_heads)?,
            gdn_b: rows(F::GdnB, value_heads)?,
            gdn_gate: rows(F::GdnGate, value_heads)?,
            gdn_beta: rows(F::GdnBeta, value_heads)?,
            gdn_recurrent_output: rows(F::GdnRecurrentOutput, dims.gdn_value())?,
            gdn_z: rows(F::GdnZ, dims.gdn_value())?,
            gdn_output: rows(F::GdnOutput, dims.gdn_value())?,
            qsa_index: rows(F::QsaIndex, dims.index_width())?,
            qsa_qgate: rows(F::QsaQgate, 2 * dims.q_width())?,
            qsa_k: rows(F::QsaK, dims.kv_width())?,
            qsa_v: rows(F::QsaV, dims.kv_width())?,
            qsa_output: rows(F::QsaOutput, dims.q_width())?,
            attention_output: rows(F::AttentionOutput, hidden)?,
            ffn_output: rows(F::FfnOutput, hidden)?,
            router_logits: matrix_view(&scratch.router_logits, seg.rows, router_columns(dims))?,
            ple_rows: rows(F::PleRows, ple_embed_dim)?,
            ple_query: rows(F::PleQuery, wide)?,
            ple_key: rows(F::PleKey, wide)?,
            ple_value: rows(F::PleValue, hidden)?,
            ple_gated: rows(F::PleGated, wide)?,
            ple_normed: rows(F::PleNormed, wide)?,
            ple_output: rows(F::PleOutput, wide)?,
            embeddings: view(&scratch.embeddings, b * hidden, seg.rows * hidden),
            embedding_rot: view(&scratch.embedding_rot, b * hidden, seg.rows * hidden),
            token_ids: view(
                &scratch.token_ids,
                b * std::mem::size_of::<i32>(),
                seg.rows * std::mem::size_of::<i32>(),
            ),
            logits,
            top1,
        })
    }

    /// The sealed-MoE / final-hyper scratch description of this segment.
    fn layer_scratch<'a>(&'a self, s: &'a Qwen4GpuForwardScratch) -> Qwen4LayerScratch<'a> {
        Qwen4LayerScratch {
            streams: &self.streams,
            rotation: &s.rotation,
            hc_normalized: &self.hc_normalized,
            hc_low: &self.hc_low,
            hc_up: &self.hc_up,
            hc_mixed: &self.hc_mixed,
            hc_gates: &self.hc_gates,
            projection: &self.projection,
            projection2: &self.projection2,
            gdn_a: &self.gdn_a,
            gdn_b: &self.gdn_b,
            gdn_gate: &self.gdn_gate,
            gdn_beta: &self.gdn_beta,
            gdn_recurrent_output: &self.gdn_recurrent_output,
            gdn_z: &self.gdn_z,
            gdn_output: &self.gdn_output,
            qsa_index: &self.qsa_index,
            qsa_qgate: &self.qsa_qgate,
            qsa_k: &self.qsa_k,
            qsa_v: &self.qsa_v,
            qsa_output: &self.qsa_output,
            attention_output: &self.attention_output,
            ffn_output: &self.ffn_output,
            moe_shared_output: &s.moe_shared_output,
            moe_router_logits: &self.router_logits,
            moe_x_rot: &s.moe_x_rot,
            moe_gate_up: &s.moe_gate_up,
            moe_scalar: &s.moe_scalar,
            moe_gate: &s.moe_gate,
            moe_up: &s.moe_up,
            moe_hidden: &s.moe_hidden,
            moe_gate_batch: &s.moe_gate_batch,
            moe_up_batch: &s.moe_up_batch,
            moe_rot_batch: &s.moe_rot_batch,
            moe_topk_indices: &s.moe_topk_indices,
            moe_topk_weights: &s.moe_topk_weights,
            moe_down_expanded: &s.moe_down_expanded,
            moe_expert_token_counts: &s.moe_expert_token_counts,
            moe_expert_offsets: &s.moe_expert_offsets,
            moe_sorted_slot_index: &s.moe_sorted_slot_index,
            moe_expert_tile_ids: &s.moe_expert_tile_ids,
            moe_inverse_perm: &s.moe_inverse_perm,
            moe_y_gate_up_grouped: &s.moe_y_gate_up_grouped,
            moe_y_down_grouped: &s.moe_y_down_grouped,
        }
    }
}

/// One operation of the layer program, placed by its stage's share.
#[derive(Clone, Copy, Debug)]
enum LaneOp {
    HcRead { layer: usize, mlp: bool },
    HcWrite { layer: usize, mlp: bool },
    PleConv,
    Gdn { layer: usize, slot: usize, phase: MixerPhase },
    Qsa { layer: usize, slot: usize, phase: MixerPhase },
    Moe { layer: usize },
}

/// The singleton layer program's operations in order, each with the stage
/// that places it. A mixer whose projections are both per lane stays the
/// whole op (the singleton's op itself); otherwise it splits into phases.
fn layer_program(
    bundle: &Qwen4Bundle,
    policy: &LaneStagePolicy,
    ple_layer: usize,
) -> Result<Vec<(LaneOp, StageName)>, Qwen4GpuForwardError> {
    let config = &bundle.config;
    let mut program = Vec::with_capacity(config.num_hidden_layers * 8 + 1);
    let mut gdn_slot = 0usize;
    let mut qsa_slot = 0usize;
    let mixer_phases = |in_stage: StageName, out_stage: StageName| -> Vec<(MixerPhase, StageName)> {
        let core = match in_stage {
            StageName::GdnInProj => StageName::GdnCore,
            _ => StageName::QsaCore,
        };
        if policy.share(in_stage).per_lane() && policy.share(out_stage).per_lane() {
            vec![(MixerPhase::Whole, core)]
        } else {
            vec![
                (MixerPhase::InProj, in_stage),
                (MixerPhase::Core, core),
                (MixerPhase::OutProj, out_stage),
            ]
        }
    };
    for layer in 0..config.num_hidden_layers {
        if layer == ple_layer {
            program.push((LaneOp::PleConv, StageName::PleConv));
        }
        program.push((LaneOp::HcRead { layer, mlp: false }, StageName::HcRead));
        match bundle
            .weights
            .layer(layer)
            .ok_or_else(|| invalid(format!("lane layer {layer} is not owned")))?
            .kind
        {
            LayerType::LinearAttention => {
                for (phase, stage) in mixer_phases(StageName::GdnInProj, StageName::GdnOutProj) {
                    program.push((LaneOp::Gdn { layer, slot: gdn_slot, phase }, stage));
                }
                gdn_slot += 1;
            }
            LayerType::FullAttention => {
                for (phase, stage) in mixer_phases(StageName::QsaInProj, StageName::QsaOutProj) {
                    program.push((LaneOp::Qsa { layer, slot: qsa_slot, phase }, stage));
                }
                qsa_slot += 1;
            }
        }
        program.push((LaneOp::HcWrite { layer, mlp: false }, StageName::HcWrite));
        program.push((LaneOp::HcRead { layer, mlp: true }, StageName::HcRead));
        program.push((LaneOp::Moe { layer }, StageName::Moe));
        program.push((LaneOp::HcWrite { layer, mlp: true }, StageName::HcWrite));
    }
    Ok(program)
}

/// QSA lengths a lane's state commits after the step (the singleton's
/// post-forward commit): `(lane, qsa slot, full, raw, pooled, selected, position)`.
type QsaCommit = (usize, usize, usize, usize, usize, usize, usize);

/// One exact trunk step over lanes (see the module docs): AR rows and MTP
/// verify blocks. Row `r` of the outputs belongs to the lanes concatenated in
/// slice order. On success every lane's host state is published exactly as
/// the singleton forward publishes it (QSA lengths, PLE history, the armed
/// ring's live slot, `position += rows`); a verify lane's PLE convolution
/// input rows stay in the forward scratch at the lane's first row until the
/// next forward ([`Qwen4Bundle::lane_ple_normed_rows`]). On failure nothing
/// host-side is published and the caller must treat the lanes' device state
/// as untrusted.
pub(crate) fn forward_lanes(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    out: &LaneOutputs<'_>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    let mut forward = bundle
        .execution
        .take()
        .ok_or_else(|| invalid("Qwen4 forward resources are not attached"))?;
    let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
    // The lane step's row-local launchers may take the gfx1151 PM row
    // kernels (`HIPFIRE_FN_LANE_PM_ROWS`, column-identical to the 1-row
    // incumbents); every other forward keeps the incumbents.
    let rows_scope = std::mem::replace(&mut gpu.lane_rows_scope, true);
    let result = forward_lanes_scoped(gpu, &mut forward, bundle, lanes, &StepOutputs::Ar(out), policy);
    gpu.lane_rows_scope = rows_scope;
    gpu.qwen4_scope = scope;
    bundle.execution = Some(forward);
    result
}

/// One exact prefill wave over `lanes` (see the module docs): the lane step
/// with prefill lanes, run under the invariant prefill profile when
/// `HIPFIRE_PREFILL_PM_INVARIANT` is set on gfx1151. QSA arenas of every
/// lane must already be mapped through its end position
/// (`Qwen4Bundle::prefill_wave` maps them). On success every lane's host
/// state is published exactly as the singleton forward publishes it; on
/// failure nothing host-side is published and the caller must treat the
/// lanes' device state as untrusted.
pub(crate) fn forward_wave(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [WaveLane<'_>],
    head: WaveHead,
    logits: &GpuTensor,
    top1: Option<&GpuTensor>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    let mut forward = bundle
        .execution
        .take()
        .ok_or_else(|| invalid("Qwen4 forward resources are not attached"))?;
    let mut rows: Vec<LaneRows<'_>> = lanes
        .iter_mut()
        .map(|lane| LaneRows {
            state: &mut *lane.state,
            tokens: lane.tokens,
            kind: LaneRowKind::Prefill,
        })
        .collect();
    let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
    let invariant = rdna_compute::prefill_invariant::prefill_pm_invariant_enabled()
        && gpu.arch == "gfx1151";
    let invariant = std::mem::replace(&mut gpu.prefill_invariant_scope, invariant);
    let result = forward_lanes_scoped(
        gpu,
        &mut forward,
        bundle,
        &mut rows,
        &StepOutputs::Wave(WaveOutputs { head, logits, top1 }),
        policy,
    );
    gpu.prefill_invariant_scope = invariant;
    gpu.qwen4_scope = scope;
    bundle.execution = Some(forward);
    result
}

fn forward_lanes_scoped(
    gpu: &mut Gpu,
    fwd: &mut Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    out: &StepOutputs<'_>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    if fwd.ep_placement().is_some() {
        return Err(invalid(
            "lane steps do not run under expert parallelism (EP forwards are one-row decode and chunk forwards only)",
        ));
    }
    if *bundle.weights.band_scope() != crate::weights::Qwen4BandScope::single(&bundle.config) {
        return Err(invalid(
            "lane steps do not run on a pipeline-parallel band (topology.pp refuses lanes)",
        ));
    }
    // Lanes are the Flash-Next route: the stage policy, row sharing and
    // decode copies are evidenced for its recipe only.
    if !bundle.config.profile.is_flash_next() {
        return Err(invalid(format!(
            "lane steps run only the Flash-Next profile, not {}",
            bundle.config.profile.id()
        )));
    }
    policy.validate().map_err(invalid)?;
    if lanes.is_empty() {
        return Err(invalid("a lane step needs at least one lane"));
    }
    let config = &bundle.config;
    let dims = program_dims(config);
    let vocab = config.vocab_size;
    let rows: Vec<usize> = lanes.iter().map(|lane| lane.tokens.len()).collect();
    let n: usize = rows.iter().sum();
    let prefill = matches!(out, StepOutputs::Wave(_));
    for (lane, rows) in lanes.iter().zip(&rows) {
        if prefill != matches!(lane.kind, LaneRowKind::Prefill) {
            return Err(invalid("a lane step is all AR/verify lanes or all prefill lanes"));
        }
        let state = &*lane.state;
        match lane.kind {
            LaneRowKind::Ar if *rows != 1 => {
                return Err(invalid("an AR lane contributes exactly one row"));
            }
            LaneRowKind::Ar if state.row_capture_armed => {
                return Err(invalid("an AR lane row never arms a GDN row capture"));
            }
            LaneRowKind::Ar => {}
            LaneRowKind::Verify { draft_len } => {
                if draft_len == 0 || *rows != draft_len + 1 || *rows > LANE_MAX_ROWS {
                    return Err(invalid(format!(
                        "a verify lane carries 1 + {draft_len} rows within {LANE_MAX_ROWS}, got {rows}"
                    )));
                }
                // The singleton's `verify_block` arms the ring for exactly
                // this shape; a lane verify never runs without it.
                if !state.row_capture_armed || state.row_capture_rows() < *rows {
                    return Err(invalid(format!(
                        "a {rows}-row verify lane needs its armed GDN ring (armed {}, {} rows)",
                        state.row_capture_armed,
                        state.row_capture_rows()
                    )));
                }
            }
            LaneRowKind::Prefill if *rows == 0 => {
                return Err(invalid("a prefill lane contributes at least one row"));
            }
            LaneRowKind::Prefill if state.row_capture_armed => {
                return Err(invalid("a prefill wave never arms a GDN row capture"));
            }
            LaneRowKind::Prefill => {}
        }
        let end = state
            .position
            .checked_add(*rows)
            .ok_or_else(|| invalid("lane position overflows"))?;
        if end > state.max_seq_len {
            return Err(invalid(format!(
                "lane end position {end} exceeds its context capacity {}",
                state.max_seq_len
            )));
        }
        if end > state.mapped_context_tokens() {
            return Err(invalid(format!(
                "lane QSA context is mapped through {} tokens, the step needs {end} (provision maps it)",
                state.mapped_context_tokens()
            )));
        }

        if lane.tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(invalid("lane token id is outside the embedding vocabulary"));
        }
    }
    if n > fwd.scratch.max_chunk {
        return Err(invalid(format!(
            "lane step of {n} rows exceeds the forward scratch ({} rows)",
            fwd.scratch.max_chunk
        )));
    }
    match out {
        StepOutputs::Ar(out) => {
            if out.logits.dtype != DType::F32 || out.logits.numel() < n * vocab {
                return Err(invalid("lane logits must be F32 with total_rows * vocab elements"));
            }
            if out.top1.dtype != DType::Raw || out.top1.buf.size() < n * std::mem::size_of::<i32>() {
                return Err(invalid("lane top1 must be Raw with one i32 per row"));
            }
        }
        StepOutputs::Wave(wave) => {
            let out_rows = match wave.head {
                WaveHead::Final => lanes.len(),
                WaveHead::All => n,
            };
            if wave.logits.dtype != DType::F32 || wave.logits.numel() < out_rows * vocab {
                return Err(invalid(format!(
                    "wave logits must be F32 with {out_rows} * vocab elements for {:?} output",
                    wave.head
                )));
            }
            if let Some(top1) = wave.top1 {
                if top1.dtype != DType::Raw
                    || top1.buf.size() < out_rows * std::mem::size_of::<i32>()
                {
                    return Err(invalid("wave top1 must be Raw with one i32 per output row"));
                }
            }
        }
    }
    if let Some(capture) = out.wide_hidden() {
        if capture.dtype != DType::F32 || capture.numel() < n * dims.wide() {
            return Err(invalid(
                "lane wide hidden capture must be F32 with total_rows * hc_count * hidden elements",
            ));
        }
    }
    // Before anything reads an expert a failed staged prefill left in a donor.
    fwd.restore_expert_stage(gpu)?;
    // The gathered prefill attention's F16 workspace, sized by the furthest
    // lane end (the singleton sizes it by its own end) before any launch.
    if prefill {
        let widest_lane = rows.iter().copied().max().unwrap_or(1);
        if gpu.route_rows(widest_lane) >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            let furthest = lanes
                .iter()
                .map(|lane| lane.state.position + lane.tokens.len())
                .max()
                .unwrap_or(0);
            if let Some(qsa) = lanes[0].state.qsa.first().filter(|qsa| qsa.f16_arena.is_none()) {
                rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(
                    gpu,
                    qsa.format,
                    config.num_key_value_heads,
                    furthest,
                )?;
            }
        }
    }

    let ple_layer = config
        .ple_layer_ids
        .first()
        .and_then(|id| id.checked_sub(1))
        .filter(|&layer| layer < config.num_hidden_layers)
        .ok_or_else(|| invalid("PLE layer id missing or outside the trunk"))?;
    let ple_geometry = config
        .ple
        .as_ref()
        .ok_or_else(|| invalid("PLE geometry missing"))?;
    let (ple_head_count, ple_row_width) = (ple_geometry.head_count, ple_geometry.row_width);
    let lane_segs = lane_segments(&rows);
    let group_segs = if prefill {
        // One launch covers every wave row.
        vec![Segment {
            begin: 0,
            rows: n,
            first_lane: 0,
        }]
    } else {
        group_segments(&rows)
    };
    let widest = group_segs.iter().map(|seg| seg.rows).max().unwrap_or(1);
    let bf16_state = if prefill {
        // One value for the whole wave: every segment's routed row count must
        // pick the same HC stream storage (always true under the invariant
        // profile; without it a wave straddling the 512-row threshold refuses).
        let state = gpu.qwen4_bf16_streams(gpu.route_rows(rows[0]));
        for seg in lane_segs.iter().chain(&group_segs) {
            if gpu.qwen4_bf16_streams(gpu.route_rows(seg.rows)) != state {
                return Err(invalid(
                    "prefill wave segments disagree on the HC stream storage (needs HIPFIRE_PREFILL_PM_INVARIANT on gfx1151, or lanes on one side of the row threshold)",
                ));
            }
        }
        state
    } else {
        // The singleton's one-row stream storage; every segment here is at
        // most LANE_MAX_ROWS rows and must agree with it.
        let state = gpu.qwen4_bf16_streams(1);
        if gpu.qwen4_bf16_streams(widest) != state {
            return Err(invalid("lane segments disagree on the HC stream storage"));
        }
        state
    };
    let segs_for = |share: StageShare| if share.per_lane() { 0..lane_segs.len() } else { lane_segs.len()..lane_segs.len() + group_segs.len() };
    let all_segs: Vec<Segment> = lane_segs.iter().chain(&group_segs).copied().collect();
    // A prefill wave's head runs per lane after the trunk (its own views), so
    // the policy's Head / Argmax segments exist only for AR steps.
    let ar_segs = |share: StageShare| if prefill { 0..0 } else { segs_for(share) };
    let ar_out: Option<&LaneOutputs<'_>> = match out {
        StepOutputs::Ar(out) => Some(*out),
        StepOutputs::Wave(_) => None,
    };
    // The row counts each segment's per-layer descriptions are built at (the
    // decode copies and the MoE symmetric capability depend on rows): an AR
    // step has one set at `widest`; a prefill wave gives every segment the
    // description its singleton forward would get at its own routed rows
    // (one set for all segments under the invariant profile).
    let seg_desc_rows: Vec<usize> = if prefill {
        all_segs.iter().map(|seg| gpu.route_rows(seg.rows)).collect()
    } else {
        vec![widest; all_segs.len()]
    };

    // Token ids: one upload for every row, in lane order.
    let mut row = 0usize;
    for lane in lanes.iter() {
        for &token in lane.tokens {
            fwd.host_token_bytes[row * 4..row * 4 + 4].copy_from_slice(&(token as i32).to_ne_bytes());
            row += 1;
        }
    }
    // PLE rows: one fetch of every lane's ids (each hashed from its own
    // history), started now so it overlaps the program build.  The lane
    // route is Flash-Next-only, which always carries PLE.
    let ple_owner = bundle
        .ple
        .as_ref()
        .ok_or_else(|| invalid("the lane forward requires a profile with PLE"))?;
    let ple_ids: Vec<u64> = lanes
        .iter()
        .flat_map(|lane| {
            lane.state
                .ple_history
                .row_ids(&ple_owner.metadata, lane.tokens)
        })
        .collect();
    let mut ple = RowFetch::begin_or_adopt(&ple_owner.rows, fwd.ple_ahead.take(), ple_ids)
        .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
    // The ids this forward uploads are not a whole-row mirror of a singleton
    // upload: no later singleton forward may claim them.
    fwd.uploaded_id_rows = 0;
    if prefill {
        // Like `lane_prefill_tile`: no lookahead names this wave's next rows,
        // and (as the singleton does for every multi-row forward) no retained
        // replay is eligible.
        fwd.set_ple_lookahead(&[]);
        gpu.replay.set_forward_eligible(false);
    }

    let attempt = (|| -> Result<Vec<QsaCommit>, Qwen4GpuForwardError> {
        let scratch = &fwd.scratch;
        let mut views = Vec::with_capacity(all_segs.len());
        for seg in &all_segs {
            views.push(SegViews::new(
                scratch,
                dims,
                vocab,
                ar_out,
                *seg,
                ple_head_count * ple_row_width,
            )?);
        }
        let seg_scratch: Vec<Qwen4LayerScratch<'_>> = views
            .iter()
            .map(|view| view.layer_scratch(scratch))
            .collect();

        // Per-layer weight descriptions, one set per distinct row count in
        // `seg_desc_rows` (AR: one set; decode copies bind for every
        // segment: none is wider than LANE_MAX_ROWS).
        let mut desc_rows: Vec<usize> = Vec::with_capacity(1);
        for &rows in &seg_desc_rows {
            if !desc_rows.contains(&rows) {
                desc_rows.push(rows);
            }
        }
        let mut desc_sets: Vec<Vec<Qwen4LayerDescription<'_>>> =
            Vec::with_capacity(desc_rows.len());
        for &rows in &desc_rows {
            let mut descs: Vec<Qwen4LayerDescription<'_>> =
                Vec::with_capacity(config.num_hidden_layers);
            for layer_index in 0..config.num_hidden_layers {
                descs.push(layer_desc(
                    &bundle.weights,
                    bundle
                        .weights
                        .layer(layer_index)
                        .ok_or_else(|| invalid(format!("lane layer {layer_index} is not owned")))?,
                    fwd.moe.get(layer_index).and_then(Option::as_ref),
                    config,
                    &fwd.decode_q8,
                    rows,
                    None,
                )?);
            }
            desc_sets.push(descs);
        }
        let seg_desc_set: Vec<usize> = seg_desc_rows
            .iter()
            .map(|rows| desc_rows.iter().position(|k| k == rows).unwrap_or(0))
            .collect();
        let ple_weights = bundle
            .weights
            .layer(ple_layer)
            .ok_or_else(|| invalid("lane PLE layer is not owned"))?
            .ple
            .as_ref()
            .ok_or_else(|| invalid("PLE weights missing"))?;
        let ple_w = ple_desc(&bundle.weights, ple_weights)?;
        let ple_rows_f16_at = |gpu: &Gpu, rows: usize| {
            qwen4_ple_rows_f16(
                gpu,
                bf16_state,
                dims.hc_count,
                config.hidden_size,
                &ple_w,
                rows,
            )
        };
        let ple_rows_f16 = if prefill {
            // One value for the whole wave (the PLE conv runs per lane on the
            // lane's own rows, like the singleton's decision at its row count).
            let first = ple_rows_f16_at(&*gpu, gpu.route_rows(rows[0]));
            for &lane_rows in &rows {
                if ple_rows_f16_at(&*gpu, gpu.route_rows(lane_rows)) != first {
                    return Err(invalid(
                        "prefill wave lanes disagree on F16 PLE row staging (needs HIPFIRE_PREFILL_PM_INVARIANT on gfx1151, or lanes on one side of the row threshold)",
                    ));
                }
            }
            first
        } else {
            if ple_rows_f16_at(&*gpu, 1) {
                return Err(invalid("lane steps do not stage F16 PLE rows"));
            }
            false
        };
        let final_hyper = hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?;
        let lm_head = dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?;
        let states: Vec<&Qwen4State> = lanes.iter().map(|lane| &*lane.state).collect();

        // Preflight the head of every segment before any launch.
        for i in segs_for(policy.share(StageName::FinalHyper)) {
            let (seg, v) = (&all_segs[i], &views[i]);
            validate_final_hyper(dims, &final_hyper, &v.streams, &v.layer_scratch(scratch), seg.rows)
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("validate lane final hyper: {error:?}"))
                })?;
        }
        for i in ar_segs(policy.share(StageName::Head)) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let hidden = view(&v.hc_mixed, 0, seg.rows * dims.hidden);
            validate_lm_head(&lm_head, &hidden, &v.logits, seg.rows, seg.rows).map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("validate lane LM head: {error:?}"))
            })?;
        }
        // A prefill wave's per-lane head outputs and row requests: Final = the
        // lane's last row only (the singleton's `requested_rows == 1`), All =
        // every row of the lane.
        let wave_head: Vec<(GpuTensor, Option<GpuTensor>, usize)> = match out {
            StepOutputs::Wave(wave) => lane_segs
                .iter()
                .enumerate()
                .map(|(lane, seg)| match wave.head {
                    WaveHead::Final => (
                        view(wave.logits, lane * vocab, vocab),
                        wave.top1.map(|top1| {
                            view(
                                top1,
                                lane * std::mem::size_of::<i32>(),
                                std::mem::size_of::<i32>(),
                            )
                        }),
                        1,
                    ),
                    WaveHead::All => (
                        view(wave.logits, seg.begin * vocab, seg.rows * vocab),
                        wave.top1.map(|top1| {
                            view(
                                top1,
                                seg.begin * std::mem::size_of::<i32>(),
                                seg.rows * std::mem::size_of::<i32>(),
                            )
                        }),
                        seg.rows,
                    ),
                })
                .collect(),
            StepOutputs::Ar(_) => Vec::new(),
        };
        for (lane, (head_logits, _, requested)) in wave_head.iter().enumerate() {
            let (seg, v) = (&lane_segs[lane], &views[lane]);
            let hidden = view(&v.hc_mixed, 0, seg.rows * dims.hidden);
            validate_lm_head(&lm_head, &hidden, head_logits, seg.rows, *requested).map_err(
                |error| {
                    Qwen4GpuForwardError::Dispatch(format!("validate wave LM head: {error:?}"))
                },
            )?;
        }

        let ctx = DispatchCtx::new(gpu);
        let program = layer_program(bundle, policy, ple_layer)?;
        let mut steps: Vec<Step<'_>> = Vec::with_capacity(program.len() * lanes.len() * 2);
        let mut qsa_commits: Vec<QsaCommit> = Vec::new();
        let mut ple_split: Option<usize> = None;
        // Runs of consecutive operations with the same share, emitted
        // segment-major.
        let mut start = 0usize;
        while start < program.len() {
            let per_lane = policy.share(program[start].1).per_lane();
            let mut end = start + 1;
            while end < program.len() && policy.share(program[end].1).per_lane() == per_lane {
                end += 1;
            }
            let share = if per_lane {
                StageShare::PerLane
            } else {
                StageShare::OnceAllRows
            };
            for i in segs_for(share) {
                let (seg, v) = (&all_segs[i], &views[i]);
                let r = seg.rows;
                let rr = if prefill { gpu.route_rows(r) } else { r };
                let descs = &desc_sets[seg_desc_set[i]];
                for &(op, _) in &program[start..end] {
                    match op {
                        LaneOp::HcRead { layer, mlp } => {
                            let read = if mlp {
                                &descs[layer].mlp_hyper.read
                            } else {
                                &descs[layer].attn_hyper.read
                            };
                            let copy = layer * DECODE_Q8_PER_LAYER + if mlp { 2 } else { 0 };
                            steps.push(Step::HyperRead(HyperReadOp {
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                input: &v.streams,
                                norm_weight: read.norm,
                                input_mix_down: decode_copy(
                                    &fwd.decode_q8,
                                    copy,
                                    rr,
                                    read.input_mix_down,
                                ),
                                input_mix_up: decode_copy(
                                    &fwd.decode_q8,
                                    copy + 1,
                                    rr,
                                    read.input_mix_up,
                                ),
                                normalized: &v.hc_normalized,
                                low: &v.hc_low,
                                up: &v.hc_up,
                                mixed: &v.hc_mixed,
                                rows: r,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                                low_rank: dims.hc_lowrank,
                            }));
                        }
                        LaneOp::HcWrite { layer, mlp } => {
                            let write = if mlp {
                                &descs[layer].mlp_hyper.write
                            } else {
                                &descs[layer].attn_hyper.write
                            };
                            steps.push(Step::HyperWrite(HyperWriteOp {
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                input: &v.streams,
                                norm_weight: write.norm,
                                block_inject: write.block_inject,
                                normalized: &v.hc_normalized,
                                mixed: if mlp {
                                    &v.ffn_output
                                } else {
                                    &v.attention_output
                                },
                                gates: &v.hc_gates,
                                output: &v.streams,
                                rows: r,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                            }));
                        }
                        LaneOp::PleConv => {
                            let state = states[seg.first_lane];
                            if ple_split.is_none() {
                                ple_split = Some(steps.len());
                            }
                            steps.push(Step::GroupedDepthwise(GroupedDepthwiseOp {
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                rows_f16: ple_rows_f16,
                                key: ple_w.key,
                                value: ple_w.value,
                                norm_key: ple_w.norm_key,
                                norm_query: ple_w.norm_query,
                                norm_conv: ple_w.norm_conv,
                                conv: ple_w.conv,
                                state: state
                                    .ple_conv
                                    .as_ref()
                                    .ok_or_else(|| invalid("PLE conv state missing"))?,
                                streams: &v.streams,
                                rows_tensor: &v.ple_rows,
                                query: &v.ple_query,
                                key_scratch: &v.ple_key,
                                value_scratch: &v.ple_value,
                                gated: &v.ple_gated,
                                normed: &v.ple_normed,
                                output: &v.ple_output,
                                rows: r,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                                embed_dim: ple_head_count * ple_row_width,
                                kernel_size: dims.ple_conv_kernel_dim,
                                dilation: config.ple_conv_dilation(),
                                epsilon: EPSILON,
                            }));
                        }
                        LaneOp::Gdn { layer, slot, phase } => {
                            let weights = match &descs[layer].attention {
                                Qwen4AttentionWeights::Linear(weights) => weights,
                                _ => return Err(invalid("GDN descriptor kind mismatch")),
                            };
                            let state = states[seg.first_lane];
                            let gdn = state
                                .gdn
                                .get(slot)
                                .ok_or_else(|| invalid("GDN state slot missing"))?;
                            let stateful = matches!(phase, MixerPhase::Whole | MixerPhase::Core);
                            steps.push(Step::GatedDeltaNet(GatedDeltaNetOp {
                                rotation: &scratch.rotation,
                                qkv: weights.qkv,
                                conv: weights.conv,
                                in_proj_a: weights.in_proj_a,
                                in_proj_b: weights.in_proj_b,
                                a_log: weights.a_log,
                                dt_bias: weights.dt_bias,
                                z: weights.z,
                                norm: weights.norm,
                                output: weights.output,
                                recurrent: &gdn.recurrent,
                                conv_state: &gdn.conv,
                                projection: &v.projection,
                                projection2: &v.projection2,
                                a: &v.gdn_a,
                                b: &v.gdn_b,
                                gate: &v.gdn_gate,
                                beta: &v.gdn_beta,
                                recurrent_output: &v.gdn_recurrent_output,
                                bf16_scratch: &scratch.gdn_bf16,
                                z_output: &v.gdn_z,
                                output_scratch: &v.gdn_output,
                                input: &v.hc_mixed,
                                output_tensor: &v.attention_output,
                                rows: r,
                                start_position: state.position,
                                key_heads: dims.linear_num_key_heads,
                                value_heads: dims.linear_num_value_heads,
                                key_dim: dims.linear_key_head_dim,
                                value_dim: dims.linear_value_head_dim,
                                conv_kernel: dims.linear_conv_kernel_dim,
                                input_width: dims.hidden,
                                row_capture: if stateful {
                                    state.gdn_row_capture(slot, r)
                                } else {
                                    None
                                },
                                zba_fold: fwd.trunk_zba.get(layer).and_then(Option::as_ref),
                                trunk_a4: fwd.trunk_a4.get(layer).copied().unwrap_or(0),
                                phase,
                            }));
                        }
                        LaneOp::Qsa { layer, slot, phase } => {
                            let weights = match &descs[layer].attention {
                                Qwen4AttentionWeights::Full(weights) => weights,
                                _ => return Err(invalid("QSA descriptor kind mismatch")),
                            };
                            let state = states[seg.first_lane];
                            let qsa = state
                                .qsa
                                .get(slot)
                                .ok_or_else(|| invalid("QSA state slot missing"))?;
                            let op = IndexedAttentionOp {
                                rotation: &scratch.rotation,
                                mode: IndexedAttentionMode::Full,
                                indexer_qk: weights.indexer_qk,
                                indexer_q_norm: weights.indexer_q_norm,
                                indexer_k_norm: weights.indexer_k_norm,
                                q: weights.q,
                                k: weights.k,
                                v: weights.v,
                                q_norm: weights.q_norm,
                                k_norm: weights.k_norm,
                                output: weights.output,
                                state: IndexedAttentionState {
                                    format: qsa.format,
                                    full_keys: &qsa.full_keys,
                                    full_values: &qsa.full_values,
                                    raw_index_keys: &qsa.raw_index_keys,
                                    pooled_keys: &qsa.pooled_keys,
                                    selected_indices: &qsa.selected_indices,
                                    f16_arena: if prefill { qsa.f16_arena_ref() } else { None },
                                    pooled_bf16: qsa.pooled_bf16_ref(),
                                    full_capacity: qsa.full_capacity,
                                    raw_capacity: qsa.raw_capacity,
                                    pooled_capacity: qsa.pooled_capacity,
                                    selected_capacity: qsa.selected_capacity,
                                    position_capacity: qsa.position_capacity,
                                    full_len: qsa.full_len,
                                    raw_len: qsa.raw_len,
                                    pooled_len: qsa.pooled_len,
                                    selected_len: qsa.selected_len,
                                    position: qsa.position,
                                },
                                input: &v.hc_mixed,
                                index_scratch: &v.qsa_index,
                                qgate_scratch: &v.qsa_qgate,
                                k_scratch: &v.qsa_k,
                                v_scratch: &v.qsa_v,
                                qsa_output: &v.qsa_output,
                                selected_scratch: &scratch.qsa_selected,
                                attention_output: &v.attention_output,
                                bf16_scratch: &scratch.gdn_bf16,
                                rows: r,
                                index_heads: dims.indexer_n_heads,
                                index_kv_heads: dims.indexer_kv_heads,
                                index_dim: dims.indexer_head_dim,
                                budget: dims.indexer_budget,
                                compress: dims.indexer_compress_ratio,
                                heads: dims.num_attention_heads,
                                kv_heads: dims.num_key_value_heads,
                                head_dim: dims.head_dim,
                                rotary_dim: config.rotary_dim(),
                                rope: config.rope(),
                                input_width: dims.hidden,
                                trunk_a4: fwd.trunk_a4.get(layer).copied().unwrap_or(0),
                                phase,
                            };
                            if matches!(phase, MixerPhase::Whole | MixerPhase::Core) {
                                let (full_len, raw_len, pooled_len, selected_len, position) =
                                    op.next_lengths().map_err(|error| {
                                        Qwen4GpuForwardError::Dispatch(format!(
                                            "derive lane QSA state: {error:?}"
                                        ))
                                    })?;
                                qsa_commits.push((
                                    seg.first_lane,
                                    slot,
                                    full_len,
                                    raw_len,
                                    pooled_len,
                                    selected_len,
                                    position,
                                ));
                            }
                            steps.push(Step::IndexedAttention(op));
                        }
                        LaneOp::Moe { layer } => {
                            // AR lanes route by their real rows (`rr == r`); a
                            // prefill wave seals at its routed rows so a 1-row
                            // lane takes the prefill MoE under the profile.
                            let Qwen4FfnBinding::Moe(moe) = descs[layer].ffn else {
                                return Err(invalid("lane steps run only the Flash-Next MoE FFN"));
                            };
                            let sealed = seal_moe_for_layer(&ctx, dims, moe, &seg_scratch[i], r, rr)
                                .map_err(|error| {
                                    Qwen4GpuForwardError::Dispatch(format!(
                                        "seal lane MoE: {error:?}"
                                    ))
                                })?;
                            steps.push(Step::Clear(ClearOp {
                                tensor: &v.ffn_output,
                                elements: r * config.hidden_size,
                            }));
                            steps.push(Step::Moe(sealed));
                        }
                    }
                }
            }
            start = end;
        }
        validate_steps(gpu, &steps).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("preflight lane program: {error:?}"))
        })?;

        // Inputs: ids, embeddings, HC stream broadcast.
        gpu.memcpy_htod_auto(
            &scratch.token_ids.buf,
            &fwd.host_token_bytes[..n * std::mem::size_of::<i32>()],
        )?;
        let embedding = bundle.weights.resident(&bundle.weights.root.embedding)?;
        let embed_share = policy.share(StageName::Embed);
        for i in segs_for(embed_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            execute_embedding(
                gpu,
                &EmbeddingOp {
                    table: embedding,
                    rotated: &v.embedding_rot,
                    token_ids: &v.token_ids,
                    output: &v.embeddings,
                    rows: seg.rows,
                    dim: config.hidden_size,
                },
            )
            .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("lane embedding: {error:?}")))?;
        }
        for i in segs_for(embed_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let streams = view(&v.streams, 0, seg.rows * dims.wide());
            if bf16_state {
                gpu.hc_streams_init_from_embed_batched_bf16(
                    &v.embeddings,
                    &streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    seg.rows as i32,
                )?;
            } else {
                gpu.hc_streams_init_from_embed_batched(
                    &v.embeddings,
                    &streams,
                    config.hidden_size as i32,
                    config.hc_count as i32,
                    seg.rows as i32,
                )?;
            }
        }

        let execute = |gpu: &mut Gpu, steps: &[Step<'_>]| {
            execute_validated_steps(gpu, &ctx, steps).map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane program: {error:?}"))
            })
        };
        let split = ple_split.ok_or_else(|| invalid("lane program has no PLE layer"))?;
        execute(gpu, &steps[..split])?;

        // PLE rows: wait for the fetch, stage, upload, convert (the first
        // layers already run on the device meanwhile).
        let lease = ple
            .wait()
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        let upload_len = lease
            .as_bytes()
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?
            .len();
        let staged = view(&scratch.ple_staged, 0, n * ple_head_count * ple_row_width);
        let buffer = fwd.host_ple.stage_buffer(gpu, upload_len)?;
        lease
            .stage_into(buffer)
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        if ple_async_upload(fwd.host_ple.enabled(), false, true, gpu.graphs.capture_mode) {
            fwd.host_ple.upload_async(gpu, &staged.buf, upload_len)?;
        } else {
            fwd.host_ple.upload_blocking(gpu, &staged.buf, upload_len)?;
        }
        lease
            .validate_after_upload()
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        let rows_share = policy.share(StageName::PleRows);
        // F16 rows are compact halves at the base of their view, so they are
        // converted only through the lane views the PLE conv reads (the
        // gather is elementwise: the bytes equal the singleton's).
        let ple_rows_segs = if ple_rows_f16 { 0..lane_segs.len() } else { segs_for(rows_share) };
        for i in ple_rows_segs {
            let (seg, v) = (&all_segs[i], &views[i]);
            let row_width = ple_head_count * ple_row_width;
            let staged_rows = view(&staged, seg.begin * row_width, seg.rows * row_width);
            let dest = view(&v.ple_rows, 0, seg.rows * row_width);
            if ple_rows_f16 {
                gpu.grouped_gather_convert_bf16_f16(
                    &staged_rows,
                    &dest,
                    seg.rows,
                    ple_head_count,
                    ple_row_width,
                )?;
            } else {
                gpu.grouped_gather_convert_bf16(
                    &staged_rows,
                    &dest,
                    seg.rows,
                    ple_head_count,
                    ple_row_width,
                )?;
            }
        }
        execute(gpu, &steps[split..])?;

        // Wide hidden capture: every row's final streams, lanes concatenated
        // (the singleton verify's `wide_hidden_capture` copy).
        if let Some(capture) = out.wide_hidden() {
            let wide_elements = n * dims.wide();
            let source = view(&scratch.streams, 0, wide_elements);
            let destination = view(capture, 0, wide_elements);
            if bf16_state {
                rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                    gpu,
                    &source,
                    &destination,
                    wide_elements,
                )?;
            } else {
                gpu.copy_d2d(&source, &destination, destination.byte_size())?;
            }
        }

        // Final HC read, LM head, greedy top-1.
        let final_share = policy.share(StageName::FinalHyper);
        for i in segs_for(final_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            execute_final_hyper(
                gpu,
                dims,
                &final_hyper,
                &v.streams,
                &seg_scratch[i],
                seg.rows,
                bf16_state,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane final hyper: {error:?}"))
            })?;
        }
        let head_share = policy.share(StageName::Head);
        for i in ar_segs(head_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let hidden = view(&v.hc_mixed, 0, seg.rows * config.hidden_size);
            let rotation = view(&scratch.rotation, 0, seg.rows * config.hidden_size);
            execute_lm_head(
                gpu,
                &lm_head,
                &hidden,
                &view(&v.logits, 0, seg.rows * vocab),
                seg.rows,
                seg.rows,
                Some(&rotation),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane LM head: {error:?}"))
            })?;
        }
        let argmax_share = policy.share(StageName::Argmax);
        for i in ar_segs(argmax_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: &view(&v.logits, 0, seg.rows * vocab),
                    indices: &v.top1,
                    rows: seg.rows,
                    vocab,
                },
            )?;
        }
        // Prefill wave head: per lane, the singleton's own LM-head / argmax
        // forms on the lane's rows (Final: `requested_rows == 1`).
        for (lane, (head_logits, head_top1, requested)) in wave_head.iter().enumerate() {
            let (seg, v) = (&lane_segs[lane], &views[lane]);
            let hidden = view(&v.hc_mixed, 0, seg.rows * config.hidden_size);
            let rotation = view(&scratch.rotation, 0, seg.rows * config.hidden_size);
            execute_lm_head(
                gpu,
                &lm_head,
                &hidden,
                head_logits,
                seg.rows,
                *requested,
                Some(&rotation),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute wave LM head: {error:?}"))
            })?;
            if let Some(top1) = head_top1 {
                argmax_f32(
                    gpu,
                    &ArgmaxF32 {
                        logits: head_logits,
                        indices: top1,
                        rows: *requested,
                        vocab,
                    },
                )?;
            }
        }
        drop(steps);
        Ok(qsa_commits)
    })();

    match attempt {
        Ok(qsa_commits) => {
            ple.complete();
            for (lane, slot, full_len, raw_len, pooled_len, selected_len, position) in qsa_commits {
                let state = lanes[lane]
                    .state
                    .qsa_mut(slot)
                    .ok_or_else(|| invalid("lane QSA state slot disappeared"))?;
                state.full_len = full_len;
                state.raw_len = raw_len;
                state.pooled_len = pooled_len;
                state.selected_len = selected_len;
                state.position = position;
                state.clamp_f16_valid();
            }
            for lane in lanes.iter_mut() {
                for &token in lane.tokens {
                    lane.state.ple_history.push(token);
                }
                lane.state.commit_row_capture(lane.tokens.len());
                lane.state.position = lane
                    .state
                    .position
                    .checked_add(lane.tokens.len())
                    .ok_or_else(|| invalid("lane position overflows at commit"))?;
            }
            Ok(())
        }
        Err(error) => {
            for lane in lanes.iter() {
                lane.state.invalidate_f16_arenas();
            }
            match ple.abort() {
                Ok(_) => Err(error),
                Err(cleanup) => Err(Qwen4GpuForwardError::Ple(format!(
                    "lane step failed: {error}; PLE cleanup failed: {cleanup}"
                ))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_policy_round_trips_through_describe_and_parse() {
        let production = LaneStagePolicy::production();
        let described = production.describe();
        let spec = format!(
            "per-lane:{}",
            described
        );
        assert_eq!(LaneStagePolicy::parse(&spec).unwrap(), production);
        assert_eq!(LaneStagePolicy::parse("production").unwrap(), production);
        assert_eq!(described.split(',').count(), StageName::ALL.len());
    }

    #[test]
    fn stateful_stages_refuse_shared_launches() {
        for stage in StageName::ALL {
            let shared = LaneStagePolicy::all_per_lane().with(stage, StageShare::OnceAllRows);
            assert_eq!(shared.is_err(), stage.stateful(), "{}", stage.key());
        }
        assert!(LaneStagePolicy::parse("per-lane:gdn_core=once").is_err());
        assert!(LaneStagePolicy::parse("per-lane:moe=once").is_ok());
        assert!(LaneStagePolicy::parse("per-lane:nope=once").is_err());
        assert!(LaneStagePolicy::parse("per-lane:moe=sometimes").is_err());
        assert!(LaneStagePolicy::parse("other").is_err());
    }

    #[test]
    fn stage_keys_are_unique_and_parse_back() {
        for stage in StageName::ALL {
            assert_eq!(StageName::parse(stage.key()), Some(stage));
        }
        let mut keys: Vec<_> = StageName::ALL.iter().map(|stage| stage.key()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), StageName::ALL.len());
    }

    #[test]
    fn production_evidence_matches_its_shares() {
        let production = LaneStagePolicy::production();
        let shared = StageName::ALL
            .iter()
            .any(|stage| production.share(*stage) == StageShare::OnceAllRows);
        // A once-over-all-rows stage ships only with a G0 receipt.
        assert_eq!(shared, !production.evidence().is_empty());
        assert!(production.validate().is_ok());
    }

    #[test]
    fn segments_cover_rows_and_cap_groups() {
        let rows = vec![1usize; 11];
        let lanes = lane_segments(&rows);
        assert_eq!(lanes.len(), 11);
        assert!(lanes.iter().enumerate().all(|(i, s)| s.begin == i && s.rows == 1 && s.first_lane == i));
        let groups = group_segments(&rows);
        assert_eq!(groups.len(), 2);
        assert_eq!((groups[0].begin, groups[0].rows, groups[0].first_lane), (0, 8, 0));
        assert_eq!((groups[1].begin, groups[1].rows, groups[1].first_lane), (8, 3, 8));
        let wide = group_segments(&[3, 9, 2]);
        assert_eq!(wide.len(), 3);
        assert_eq!((wide[1].begin, wide[1].rows), (3, 9));
        assert_eq!((wide[2].begin, wide[2].rows), (12, 2));
    }
}
