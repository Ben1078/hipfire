// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Private executable programs for sealed MoE calls.
//!
//! [`SealedMoeCall`](super::sealed_moe::SealedMoeCall) remains the complete-call
//! authority. This module only lowers an already sealed call into borrowed,
//! fixed-capacity [`Step`](super::steps::Step) values. The opaque operation
//! handles below are intentionally impossible to construct outside this
//! lowerer: a caller can submit a complete sealed call or nothing.

use super::steps::Step;
use crate::context::DispatchCtx;
use crate::cpu_exec;
use crate::cpu_split::CpuSplitExecutor;
use crate::families::moe::{
    MoeNormalization, MoeParams, MoePrefillParams, MoePrefillResolution, MoeQ8RouterPolicy,
    MoeRecipe, MoeResolution, MoeRouteCapability, MOE_BIASED_SIGMOID_MAX_K, NARROW_VERIFY_MAX_ROWS,
};
use crate::pipeline::sealed_moe::{MoeProtocol, MoeRouterInput, SealedMoeCall};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};
use smallvec::SmallVec;
use std::cell::{Cell, OnceCell, RefCell};
use std::sync::OnceLock;

const INLINE_STEP_CAPACITY: usize = 32;

/// One operation in a sealed MoE program. The lowerer alone chooses and orders
/// stages; external callers cannot create the matching sealed operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeStage {
    Normalize,
    InputBasis,
    GateSide,
    RouterProjection,
    Route,
    /// Balanced EP ownership ([`MoeRowGather::assigns_owners`]): rewrites the
    /// layer's live expert tables and slot owners from the routed ids; lowered
    /// only immediately after the stage that wrote them, before any routed
    /// table reader, and only when the installed gather asks for it.
    OwnerAssign,
    SharedGateUp,
    SharedActivation,
    SharedDown,
    Scatter,
    GateUp,
    Unscatter,
    Activation,
    Down,
    MutationFence,
    /// Expert-parallel copy-only exchange of peer-owned expanded rows
    /// ([`MoeParams::ep_row_gather`]); lowered only between `MutationFence`
    /// and `Combine` of an indexed decode that supplies a gather.
    RowGather,
    /// Owner-routed EP chunk exchange ([`MoePrefillParams::ep_tile_exchange`]): the
    /// masked copy of the tile ids (non-owned experts' tiles set to the -1 sentinel
    /// the grouped symmetric GEMMs exit on); lowered only right after the
    /// scatter of a grouped symmetric prefill that supplies an exchange.
    TileMask,
    /// Expert-parallel copy-only exchange of peer-owned grouped down rows
    /// ([`MoePrefillParams::ep_tile_exchange`]); lowered only between
    /// `MutationFence` and `Combine` of that same prefill.
    TileExchange,
    Combine,
    HostExperts,
    CpuSplice,
    /// Eager CPU-cold/GPU-hot split ([`MoeDecodeSelection::cpu_split`],
    /// [`MoePrefillSelection::cpu_split`]): the one input-readiness wait.
    /// Downloads the ORIGINAL route and rotated activation, validates every
    /// cold span and uploads the executor's separate masked ids. Lowered only
    /// after the router and the rotated input exist and strictly before the
    /// routed GPU tail, which is the only reader of the masked ids.
    PrepareSplit,
    /// Runs the cold slots' unweighted CPU arithmetic while the masked GPU
    /// tail executes, then queues their F32 slot rows behind the GPU down on
    /// the same stream. Lowered only after the routed `Down` and before the
    /// canonical `Combine`; it never completes or synchronizes the stream.
    JoinSplit,
}

fn router_shared_fuse_allowed(
    skip_routing: bool,
    use_gpu_topk: bool,
    exact_wave64_router: bool,
    batch_size: usize,
    skip_shared: bool,
    smi: usize,
    shared_down_dtype: rdna_compute::DType,
    shared_down_has_awq: bool,
    env_enabled: bool,
) -> bool {
    !skip_routing
        && use_gpu_topk
        && exact_wave64_router
        && batch_size == 1
        && !skip_shared
        && smi == 512
        && shared_down_dtype == rdna_compute::DType::MQ4G256
        && !shared_down_has_awq
        && env_enabled
}

/// Opt-in gfx1100 decode down+last-combine admission. The fused kernel is a
/// K=512 (two-group) body, so any other expert-down K (`mi`) stays on the
/// expanded down + separate combine path. `env_enabled` is evaluated last and
/// only when every structural condition holds.
#[allow(clippy::too_many_arguments)]
fn down_last_combine_allowed(
    ep_none: bool,
    is_gfx1100: bool,
    batch_size: usize,
    topk: usize,
    has_expert_dtype_tags: bool,
    routed_down: rdna_compute::DType,
    down_k: usize,
    env_enabled: impl FnOnce() -> bool,
) -> bool {
    ep_none
        && is_gfx1100
        && batch_size == 1
        && topk == 8
        && !has_expert_dtype_tags
        && routed_down == rdna_compute::DType::MQ4G256
        && down_k == 512
        && env_enabled()
}

/// Static decode choices resolved while sealing, before the call is published.
/// This record contains no borrowed call data and no executable instruction
/// sequence.
#[derive(Clone, Copy, Debug)]
pub(super) struct MoeDecodeSelection {
    pub(super) resolution: MoeResolution,
    pub(super) router_shared_fuse: bool,
    pub(super) exact_wave64_router: bool,
    pub(super) wave64_router: bool,
    pub(super) down_last_combine: bool,
    pub(super) route: Option<crate::families::moe::MoeRouteCapability>,
    pub(super) ninepath_d3: bool,
    pub(super) ninepath_d4: bool,
    pub(super) ninepath_mq3l: bool,
    pub(super) ninepath_mq4v2: bool,
    pub(super) ninepath_mq6v2: bool,
    /// The routed expert FFN runs in the arch's CPU splice after this program:
    /// no routed GPU stage may be lowered (see
    /// [`MoeParams::cpu_deferred_experts`](crate::families::moe::MoeParams::cpu_deferred_experts)).
    pub(super) cpu_deferred_experts: bool,
    /// Distinct eager CPU-cold/GPU-hot split mode (see
    /// [`RoutedExpertWeights::cpu_split`](crate::families::moe::RoutedExpertWeights::cpu_split)).
    /// Mutually exclusive with `cpu_deferred_experts`: the routed GPU tail is
    /// lowered over the executor's masked ids and the canonical combine reads
    /// the original route.
    pub(super) cpu_split: bool,
}

/// Static grouped-prefill choices resolved while sealing. It deliberately
/// contains only scalar choices and the family-owned resolution record.
pub(super) struct MoePrefillSelection {
    pub(super) resolution: MoePrefillResolution,
    pub(super) path2_m_total: usize,
    pub(super) force_mq4_grouped_fp16: bool,
    pub(super) coalesced_verify: bool,
    /// Exact architecture-selected route, or the generic grouped path.
    pub(super) route: Option<crate::families::moe::MoeRouteCapability>,
    /// The routed expert FFN runs in the arch's CPU splice after this program:
    /// no routed GPU stage may be lowered (see
    /// [`MoePrefillParams::cpu_deferred_experts`](crate::families::moe::MoePrefillParams::cpu_deferred_experts)).
    pub(super) cpu_deferred_experts: bool,
    /// Distinct eager CPU-cold/GPU-hot split mode for a window of at most
    /// [`NARROW_VERIFY_MAX_ROWS`] rows (see [`MoeDecodeSelection::cpu_split`]).
    /// Mutually exclusive with `cpu_deferred_experts`.
    pub(super) cpu_split: bool,
}

impl MoePrefillSelection {
    /// FP8 owns its grouped arm; it never enables the generic WMMA Path 2.
    pub(super) fn uses_grouped_rows(&self) -> bool {
        self.route.is_some_and(MoeRouteCapability::is_fp8_grouped) || self.resolution.use_path2
    }
}

/// The call's static kernel selection. Neither variant stores a step list or
/// a borrow of its enclosing call.
pub(super) enum MoeKernelSelection {
    Decode(MoeDecodeSelection),
    Prefill(MoePrefillSelection),
}

/// Host route produced by the CPU top-k fallback. The vectors are allocated
/// only when that route is selected, and remain execution-local.
pub(super) struct HostRoute {
    pub(super) indices: Vec<usize>,
    pub(super) weights: Vec<f32>,
}

/// Lifetime-free route metadata retained only for this execution. Tensor
/// identities are host reference identities, not device addresses.
#[derive(Clone, Copy)]
struct RouteStamp {
    invocation: u64,
    protocol: MoeProtocol,
    router: MoeRouterInput,
    n_experts: usize,
    k_top: usize,
    indices_tensor: usize,
    weights_tensor: usize,
    adopted_from: Option<u64>,
}

/// Invocation-local mutable state. It borrows the sealed call but never gets
/// stored by that call, so borrowed Steps cannot form a self-referential owner.
pub(super) struct MoeStepState<'a> {
    call: &'a SealedMoeCall<'a>,
    host_route: OnceCell<HostRoute>,
    route_stamp: Cell<Option<RouteStamp>>,
    shared_gate: Option<GpuTensor>,
    shared_up: Option<GpuTensor>,
    /// H8a: the routed target's zero fill was absorbed; the qt44 combine
    /// writes it from +0.0 and clears this.
    initial_zero: Cell<bool>,
    /// H4 level 3: the paired HC write the shared down may carry (taken by the
    /// stage), and whether it did.
    hc_write: RefCell<Option<super::qt44_qt53_prefill::HcSharedDown>>,
    hc_folded: Cell<bool>,
    /// `HIPFIRE_QWEN4_HC_ROW_FOLD`: the shared down ran the whole tail as the
    /// HC row fold, including the next HC read's norm.
    hc_row_folded: Cell<bool>,
    /// Where this invocation is in the split transaction (never read when the
    /// call is not in the split mode).
    split_phase: Cell<SplitPhase>,
    /// The post-combine parity reference re-runs the routed GPU tail over the
    /// ORIGINAL ids; while set, the masked view is not applied.
    split_reference: Cell<bool>,
}

impl<'a> MoeStepState<'a> {
    fn new(call: &'a SealedMoeCall<'a>) -> Self {
        let (shared_gate, shared_up) = match call.decode_params() {
            Some(params) => match params.shared.as_ref() {
                Some(shared) if shared.intermediate != 0 => (
                    Some(super::slice_moe_f32_view(
                        shared.gate_out,
                        0,
                        shared.intermediate,
                    )),
                    Some(super::slice_moe_f32_view(
                        shared.up_out,
                        0,
                        shared.intermediate,
                    )),
                ),
                _ => (None, None),
            },
            None => (None, None),
        };
        let route_stamp = call
            .attached_route_stamp()
            .or_else(|| call.prebound_route_stamp())
            .map(
                |(
                    invocation,
                    protocol,
                    router,
                    n_experts,
                    k_top,
                    indices_tensor,
                    weights_tensor,
                    adopted_from,
                )| RouteStamp {
                    invocation,
                    protocol,
                    router,
                    n_experts,
                    k_top,
                    indices_tensor,
                    weights_tensor,
                    adopted_from,
                },
            );
        Self {
            call,
            host_route: OnceCell::new(),
            route_stamp: Cell::new(route_stamp),
            shared_gate,
            shared_up,
            initial_zero: Cell::new(false),
            hc_write: RefCell::new(None),
            hc_folded: Cell::new(false),
            hc_row_folded: Cell::new(false),
            split_phase: Cell::new(SplitPhase::Idle),
            split_reference: Cell::new(false),
        }
    }

    fn dispatch_ctx(&self) -> &DispatchCtx {
        self.call.dispatch_ctx()
    }

    fn decode_parts(&self) -> Result<(&MoeParams<'_>, &MoeDecodeSelection), DispatchError> {
        let params = self.call.decode_params().ok_or_else(|| {
            DispatchError::Hip("sealed moe: operation requires decode operands".into())
        })?;
        let selection = match self.call.kernel_selection() {
            MoeKernelSelection::Decode(selection) => selection,
            MoeKernelSelection::Prefill(_) => {
                return Err(DispatchError::Hip(
                    "sealed moe: decode operation received prefill selection".into(),
                ));
            }
        };
        Ok((params, selection))
    }

    fn prefill_parts(
        &self,
    ) -> Result<(&MoePrefillParams<'_>, &MoePrefillSelection), DispatchError> {
        let params = self.call.prefill_params().ok_or_else(|| {
            DispatchError::Hip("sealed moe: operation requires prefill operands".into())
        })?;
        let selection = match self.call.kernel_selection() {
            MoeKernelSelection::Prefill(selection) => selection,
            MoeKernelSelection::Decode(_) => {
                return Err(DispatchError::Hip(
                    "sealed moe: prefill operation received decode selection".into(),
                ));
            }
        };
        Ok((params, selection))
    }

    /// Whether the qt44 combine of this call can take over `clear`'s zero
    /// fill (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`, exact gfx1151 / gfx1201):
    /// the combine writes every element `clear` zeroes, into that very
    /// tensor, nothing else writes it before the combine, and nothing is
    /// recorded or captured (a retained tape keeps the clear's shape).
    fn absorbs_clear(&self, gpu: &Gpu, clear: &super::layer_ops::ClearOp<'_>) -> bool {
        if !gpu.flags.qwen4_moe_combine_zinit_enabled()
            || gpu.replay.is_recording()
            || gpu.replay.retained_body_active()
            || gpu.graphs.capture_mode
        {
            return false;
        }
        let Ok((params, selection)) = self.prefill_parts() else {
            return false;
        };
        let target = params.routed_out.unwrap_or(params.x_batch);
        selection.route.is_some_and(MoeRouteCapability::is_sealed_grouped)
            // A CPU-deferred call omits the combine that would consume the
            // absorbed fill, so it must never absorb one.
            && !selection.cpu_deferred_experts
            && !selection.cpu_split
            && matches!(params.prelude.normalization, MoeNormalization::Provided)
            && super::qt44_qt53_prefill::combine_initial_zero_applies(
                gpu,
                params,
                selection.uses_grouped_rows(),
            )
            && clear.tensor.buf.as_ptr() == target.buf.as_ptr()
            && Some(clear.elements) == params.batch_size.checked_mul(params.down_m)
    }

    fn shared_views(&self) -> Result<(&GpuTensor, &GpuTensor), DispatchError> {
        match (&self.shared_gate, &self.shared_up) {
            (Some(gate), Some(up)) => Ok((gate, up)),
            _ => Err(DispatchError::Hip(
                "sealed moe: shared operation has no decode shared-expert views".into(),
            )),
        }
    }

    fn install_route_stamp(
        &self,
        n_experts: usize,
        k_top: usize,
        adopted_from: Option<u64>,
    ) -> Result<(), DispatchError> {
        if self.route_stamp.get().is_some() {
            return Err(DispatchError::Hip(
                "sealed moe: route producer ran twice in one invocation".into(),
            ));
        }
        let (indices_tensor, weights_tensor) = match self.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let params = self.decode_parts()?.0;
                (
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let params = self.prefill_parts()?.0;
                (
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
        };
        self.route_stamp.set(Some(RouteStamp {
            invocation: self.call.invocation(),
            protocol: self.call.protocol(),
            router: self.call.router_input(),
            n_experts,
            k_top,
            indices_tensor,
            weights_tensor,
            adopted_from,
        }));
        Ok(())
    }

    fn require_route_ready(&self) -> Result<(), DispatchError> {
        let stamp = self.route_stamp.get().ok_or_else(|| {
            DispatchError::Hip("sealed moe: route consumer ran before route production".into())
        })?;
        let (n_experts, k_top, indices_tensor, weights_tensor) = match self.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let params = self.decode_parts()?.0;
                (
                    params.n_exp,
                    params.k,
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let params = self.prefill_parts()?.0;
                (
                    params.n_exp,
                    params.k_top,
                    std::ptr::addr_of!(*params.topk_indices) as usize,
                    std::ptr::addr_of!(*params.topk_weights) as usize,
                )
            }
        };
        if stamp.invocation != self.call.invocation()
            || stamp.protocol != self.call.protocol()
            || stamp.router != self.call.router_input()
            || stamp.n_experts != n_experts
            || stamp.k_top != k_top
            || stamp.indices_tensor != indices_tensor
            || stamp.weights_tensor != weights_tensor
        {
            return Err(DispatchError::Hip(
                "sealed moe: route readiness stamp does not match this invocation".into(),
            ));
        }
        Ok(())
    }

    fn require_split_phase(&self, want: SplitPhase, stage: &str) -> Result<(), DispatchError> {
        let have = self.split_phase.get();
        if have != want {
            return Err(split_error(format!(
                "{stage} ran out of order (phase {have:?}, expected {want:?})"
            )));
        }
        Ok(())
    }

    /// The decode params the routed GPU tail reads. `None` outside the split
    /// mode, so the caller keeps the original call.
    ///
    /// Normal split run (`Prepared` phase): the original call with only
    /// `topk_indices` replaced by the executor's masked ids.
    ///
    /// Post-combine reference rerun: the original call (original ids) with
    /// `skip_shared = true`. The rerun replays gate/up, activation and down,
    /// and the fused routed activation sigmoids the shared selector scalar IN
    /// PLACE; the original shared stage already consumed that scalar, so a
    /// second pass would apply `sigmoid(sigmoid(x))` before the shared down.
    /// `skip_shared` keeps the shared activation out of the routed launch.
    /// Every other consumer (combine, shared, router) never takes this view.
    fn masked_decode<'p>(
        &self,
        params: &MoeParams<'p>,
        selection: &MoeDecodeSelection,
    ) -> Result<Option<MoeParams<'p>>, DispatchError> {
        if !selection.cpu_split {
            return Ok(None);
        }
        if self.split_reference.get() {
            let mut view = *params;
            view.skip_shared = true;
            return Ok(Some(view));
        }
        self.require_split_phase(SplitPhase::Prepared, "the routed GPU tail")?;
        let executor = decode_split_executor(params)?;
        Ok(Some(params.with_topk_indices(executor.masked_indices())))
    }

    /// Prefill counterpart of [`Self::masked_decode`] (the grouped scatter
    /// reads it too). The reference rerun view clears `prelude.shared`: the
    /// fused unscatter activation would otherwise sigmoid the shared selector
    /// scalar in place a second time.
    fn masked_prefill<'p>(
        &self,
        params: &MoePrefillParams<'p>,
        selection: &MoePrefillSelection,
    ) -> Result<Option<MoePrefillParams<'p>>, DispatchError> {
        if !selection.cpu_split {
            return Ok(None);
        }
        if self.split_reference.get() {
            let mut view = *params;
            view.prelude.shared = None;
            return Ok(Some(view));
        }
        self.require_split_phase(SplitPhase::Prepared, "the routed GPU tail")?;
        let executor = prefill_split_executor(params)?;
        Ok(Some(params.with_topk_indices(executor.masked_indices())))
    }
}

/// Opaque operand for one granular sealed MoE operation. The private state
/// field is the API-privacy boundary: external code cannot construct one,
/// clone one, or extract its program/state.
///
/// ```compile_fail
/// use hipfire_dispatch::pipeline::SealedMoeOp;
///
/// fn forge<'a>() -> SealedMoeOp<'a> {
///     SealedMoeOp { state: todo!() }
/// }
/// ```
pub struct SealedMoeOp<'a> {
    state: &'a MoeStepState<'a>,
}

impl<'a> SealedMoeOp<'a> {
    fn new(state: &'a MoeStepState<'a>) -> Self {
        Self { state }
    }

    pub(super) fn execute_stage(
        &self,
        gpu: &mut Gpu,
        stage: MoeStage,
    ) -> Result<(), DispatchError> {
        match stage {
            MoeStage::Normalize => self.normalize(gpu),
            MoeStage::InputBasis => self.input_basis(gpu),
            MoeStage::GateSide => self.gate_side(gpu),
            MoeStage::RouterProjection => self.router_projection(gpu),
            MoeStage::Route => self.route(gpu),
            MoeStage::OwnerAssign => self.owner_assign(gpu),
            MoeStage::SharedGateUp => self.shared_gate_up(gpu),
            MoeStage::SharedActivation => self.shared_activation(gpu),
            MoeStage::SharedDown => self.shared_down(gpu),
            MoeStage::Scatter => self.scatter(gpu),
            MoeStage::GateUp => self.gate_up(gpu),
            MoeStage::Unscatter => self.unscatter(gpu),
            MoeStage::Activation => self.activation(gpu),
            MoeStage::Down => self.down(gpu),
            MoeStage::MutationFence => self.mutation_fence(gpu),
            MoeStage::RowGather => self.row_gather(gpu),
            MoeStage::TileMask => self.tile_mask(gpu),
            MoeStage::TileExchange => self.tile_exchange(gpu),
            MoeStage::Combine => self.combine(gpu),
            MoeStage::HostExperts => self.host_experts(gpu),
            MoeStage::CpuSplice => self.cpu_splice(gpu),
            MoeStage::PrepareSplit => self.prepare_split(gpu),
            MoeStage::JoinSplit => self.join_split(gpu),
        }
    }

    pub(super) fn normalize(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                if let MoeNormalization::RmsNorm {
                    weight,
                    plain_out,
                    eps,
                } = &params.normalization
                {
                    if params.x_rot_prerotated {
                        gpu.fused_rmsnorm_rotate_mq(
                            params.x_residual,
                            weight,
                            params.x_rot_local,
                            params.hidden,
                            *eps,
                        )
                    } else {
                        gpu.rmsnorm_f32(params.x_residual, weight, plain_out, *eps)
                    }
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
            }
            MoeProtocol::GroupedPrefill => {
                let (params, _) = self.state.prefill_parts()?;
                if let MoeNormalization::RmsNorm {
                    weight,
                    plain_out,
                    eps,
                } = &params.prelude.normalization
                {
                    gpu.rmsnorm_batched(
                        params.x_batch,
                        weight,
                        plain_out,
                        params.batch_size,
                        params.gate_up_k,
                        *eps,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn input_basis(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            return super::decode_input_basis_stage(gpu, params, selection.resolution);
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection.cpu_split {
            // The CPU computes from the F32 rotated basis. The grouped stage may
            // skip it when the GPU gate/up rotates to F16 itself, so the split
            // always writes it (the same rotation the stage would have run).
            return gpu
                .rotate_x_mq_batched(
                    params.x_norm_batch,
                    params.x_rot_batch,
                    params.gate_up_k,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()));
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::input_basis(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        let input_weight = params
            .prelude
            .shared
            .as_ref()
            .map(|shared| &shared.weights.gate);
        if selection.resolution.paro_mode {
            super::prefill_input_basis_stage(gpu, params)?;
        } else if let Some(weight) = input_weight {
            if let Some(awq) = weight.awq_scale {
                gpu.rotate_x_mq_awq_batched(
                    params.x_norm_batch,
                    awq,
                    params.x_rot_batch,
                    weight.k,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            } else {
                gpu.rotate_x_mq_batched(
                    params.x_norm_batch,
                    params.x_rot_batch,
                    weight.k,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
        } else if let Some((gate_up, _)) = params.routed_experts.get(0) {
            // No-shared recipes still need the routed expert's input basis.
            gpu.rotate_x_mq_batched(
                params.x_norm_batch,
                params.x_rot_batch,
                gate_up.k,
                params.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
        Ok(())
    }

    pub(super) fn gate_side(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.decode_parts()?;
        if params.recipe == MoeRecipe::SigmoidRoutedNoShared {
            return self.router_projection(gpu);
        }
        let (shared_gate, shared_up) = self.state.shared_views()?;
        super::decode_gate_side_stage(
            self.state.dispatch_ctx(),
            gpu,
            params,
            selection.resolution,
            selection
                .resolution
                .needs_x_rot_local
                .then_some(params.x_rot_local),
            shared_gate,
            shared_up,
            selection.route,
        )
    }

    pub(super) fn router_projection(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                if matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. }) {
                    return self.gate_side(gpu);
                }
                static GEMV_ROUTER: OnceLock<crate::families::gemv::GemvFamily> = OnceLock::new();
                GEMV_ROUTER
                    .get_or_init(crate::families::gemv::GemvFamily::new)
                    .run_auto(
                        self.state.dispatch_ctx(),
                        gpu,
                        &params.router,
                        params.x_norm,
                        params.router_logits,
                    )
                    .map_err(|e| DispatchError::Hip(e.to_string()))
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if selection
                    .route
                    .is_some_and(MoeRouteCapability::is_sealed_grouped)
                {
                    return super::qt44_qt53_prefill::router_projection(gpu, params);
                }
                let prelude = &params.prelude;
                let x = match prelude.router.dtype {
                    rdna_compute::DType::Q8_0 | rdna_compute::DType::F32 => params.x_norm_batch,
                    _ => params.x_rot_batch,
                };
                let result = match prelude.q8_router_policy {
                    MoeQ8RouterPolicy::FreshFp16Wmma { enabled, scratch }
                        if prelude.router.dtype == rdna_compute::DType::Q8_0 =>
                    {
                        crate::families::gemm::run_q8_projection_fresh(
                            gpu,
                            &prelude.router.buf,
                            x,
                            prelude.router_logits,
                            prelude.router.m,
                            prelude.router.k,
                            params.batch_size,
                            scratch,
                            enabled,
                        )
                        .map_err(|e| DispatchError::Hip(e.to_string()))
                    }
                    _ => {
                        static GEMM_ROUTER: OnceLock<crate::families::gemm::GemmFamily> =
                            OnceLock::new();
                        let key = match prelude.router.dtype {
                            rdna_compute::DType::Q8_0 => {
                                crate::types::KernelKey::GemmQ8_0BatchedChunked
                            }
                            rdna_compute::DType::MQ4G256 => crate::types::KernelKey::GemmHfq4G256,
                            rdna_compute::DType::MQ4G256V2 => {
                                crate::types::KernelKey::GemmMq4G256V2
                            }
                            rdna_compute::DType::F32 => crate::types::KernelKey::GemmF32Batched,
                            _ => {
                                return Err(DispatchError::UnsupportedVariant {
                                    family: "moe",
                                    variant: "prefill-router-dtype",
                                    arch: "",
                                    quant: "",
                                });
                            }
                        };
                        GEMM_ROUTER
                            .get_or_init(crate::families::gemm::GemmFamily::new)
                            .run_key(
                                key,
                                self.state.dispatch_ctx(),
                                gpu,
                                &crate::families::gemm::GemmParams {
                                    w: &prelude.router,
                                    x,
                                    y: prelude.router_logits,
                                    batch_size: params.batch_size,
                                },
                            )
                    }
                };
                result?;
                if matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. }) {
                    super::dump_hidden_localize(
                        gpu,
                        prelude.router_logits,
                        params.batch_size,
                        0,
                        params.n_exp,
                        0,
                        "router_b",
                    );
                }
                Ok(())
            }
        }
    }

    pub(super) fn route(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::GroupedPrefill => {
                if let Some(stamp) = self.state.route_stamp.get() {
                    if stamp.adopted_from.is_some() {
                        return Ok(());
                    }
                    return Err(DispatchError::Hip(
                        "sealed moe: route producer ran twice in one invocation".into(),
                    ));
                }
                let (params, _) = self.state.prefill_parts()?;
                let scores = params.prelude.router_scores;
                let receipt = super::sealed_moe::produce_prefill_route(
                    self.state.call,
                    gpu,
                    scores,
                    params.prelude.norm_topk_prob,
                )?;
                self.state.call.validate_route_receipt(&receipt)?;
                if let super::sealed_moe::PrefillRouteMode::ProduceRoot { slot } =
                    params.prelude.route
                {
                    let proof = self
                        .state
                        .call
                        .prefill_route_producer_proof_for_receipt(&receipt)?;
                    slot.set(Some(proof));
                }
                self.state.install_route_stamp(
                    params.n_exp,
                    params.k_top,
                    receipt.adopted_from(),
                )?;
                Ok(())
            }
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if matches!(
                    self.state.call.router_input(),
                    MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
                ) {
                    if self.state.route_stamp.get().is_none() {
                        return Err(DispatchError::Hip(
                            "sealed moe: adopted route is not ready before route consumers".into(),
                        ));
                    }
                    return Ok(());
                }
                if selection.resolution.use_gpu_topk {
                    if self.state.call.router_input() == MoeRouterInput::SigmoidTopK {
                        #[cfg(feature = "deltanet")]
                        {
                            gpu.sigmoid_f32(params.router_logits)
                                .map_err(|e| DispatchError::Hip(e.to_string()))?;
                            match params.selection_bias {
                                Some(bias) => gpu.deepseek4_moe_topk_bias_aware_f32(
                                    params.router_logits,
                                    bias.bias,
                                    params.topk_indices,
                                    params.topk_weights,
                                    params.n_exp as i32,
                                    params.k as i32,
                                    bias.route_scale,
                                ),
                                None => gpu.moe_topk_renorm_k8(
                                    params.router_logits,
                                    params.topk_indices,
                                    params.topk_weights,
                                    params.n_exp,
                                    params.norm_topk_prob,
                                ),
                            }
                            .map_err(|e| DispatchError::Hip(e.to_string()))?;
                        }
                        #[cfg(not(feature = "deltanet"))]
                        return Err(DispatchError::UnsupportedVariant {
                            family: "moe",
                            variant: "sigmoid-router-requires-deltanet",
                            arch: "",
                            quant: "",
                        });
                    } else {
                        let (shared_gate, shared_up) = self.state.shared_views()?;
                        super::decode_route_gpu_stage(
                            gpu,
                            params,
                            shared_gate,
                            shared_up,
                            selection.router_shared_fuse,
                            selection.exact_wave64_router,
                            selection.wave64_router,
                            selection.route,
                        )?;
                    }
                } else {
                    super::decode_route_cpu_stage(
                        gpu,
                        params,
                        &self.state.host_route,
                        self.state.call.router_input(),
                    )?;
                }
                self.state.install_route_stamp(params.n_exp, params.k, None)
            }
        }
    }

    pub(super) fn shared_gate_up(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let (shared_gate, shared_up) = self.state.shared_views()?;
            return super::decode_gate_side_stage(
                self.state.dispatch_ctx(),
                gpu,
                params,
                selection.resolution,
                selection
                    .resolution
                    .needs_x_rot_local
                    .then_some(params.x_rot_local),
                shared_gate,
                shared_up,
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::shared_gate_up(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        super::prefill_shared_gate_up_stage(self.state.dispatch_ctx(), gpu, params)
    }

    pub(super) fn shared_activation(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let (shared_gate, shared_up) = self.state.shared_views()?;
            let target = if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
                params.x_residual
            } else {
                params.routed_out.unwrap_or(params.x_residual)
            };
            return super::decode_shared_down_stage(
                self.state.dispatch_ctx(),
                gpu,
                params,
                shared_gate,
                shared_up,
                target,
                selection.router_shared_fuse,
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::shared_activation(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        super::prefill_shared_activation_stage(gpu, params)
    }

    pub(super) fn shared_down(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let (shared_gate, shared_up) = self.state.shared_views()?;
            let target = if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
                params.x_residual
            } else {
                params.routed_out.unwrap_or(params.x_residual)
            };
            let dump = if params.skip_shared {
                None
            } else {
                super::moe_dump::shared_begin(
                    gpu,
                    params.layer_idx as usize,
                    target,
                    params.hidden,
                )?
            };
            super::decode_shared_down_stage(
                self.state.dispatch_ctx(),
                gpu,
                params,
                shared_gate,
                shared_up,
                target,
                selection.router_shared_fuse,
                selection.route,
            )?;
            return super::moe_dump::shared_end(
                gpu,
                dump,
                target,
                params.hidden,
                !std::ptr::eq(target, params.x_residual),
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            // H4 (`HIPFIRE_QWEN4_HC_FUSE` >= 3): the paired HC write rides the
            // shared down's GEMM epilogue when this call admits it.
            if let Some(hc) = self.state.hc_write.take() {
                let row_fold = hc.row_fold.is_some();
                if super::qt44_qt53_prefill::shared_down_hc(
                    gpu,
                    params,
                    &hc,
                    selection.uses_grouped_rows(),
                )? {
                    self.state.hc_folded.set(true);
                    self.state.hc_row_folded.set(row_fold);
                    return Ok(());
                }
            }
            return super::qt44_qt53_prefill::shared_down(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        super::prefill_shared_down_stage(self.state.dispatch_ctx(), gpu, params)
    }

    pub(super) fn scatter(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        self.state.require_route_ready()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if !selection.uses_grouped_rows() {
            return Err(DispatchError::Hip(
                "sealed moe: scatter is only valid for grouped prefill path 2".into(),
            ));
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::scatter(gpu, params, selection.path2_m_total);
        }
        super::prefill_scatter_stage(gpu, params, selection.path2_m_total)
    }

    pub(super) fn gate_up(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.require_route_ready()?;
        if let Ok((params, selection)) = self.state.decode_parts() {
            let view = self.state.masked_decode(params, selection)?;
            let params = view.as_ref().unwrap_or(params);
            return super::decode_gate_up_stage(
                gpu,
                params,
                selection.resolution,
                selection
                    .resolution
                    .needs_x_rot_local
                    .then_some(params.x_rot_local)
                    .unwrap_or(params.x_norm),
                selection.ninepath_d3,
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::gate_up(
                gpu,
                params,
                selection.uses_grouped_rows(),
                selection.path2_m_total,
            );
        }
        super::prefill_gate_up_stage(
            gpu,
            params,
            &selection.resolution,
            selection.path2_m_total,
            selection.force_mq4_grouped_fp16,
            selection.coalesced_verify,
        )
    }

    pub(super) fn unscatter(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if !selection.uses_grouped_rows() {
            return Err(DispatchError::Hip(
                "sealed moe: unscatter is only valid for grouped prefill path 2".into(),
            ));
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::unscatter(gpu, params, selection.path2_m_total);
        }
        super::prefill_gate_up_unscatter_stage(gpu, params, selection.path2_m_total)
    }

    pub(super) fn activation(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let view = self.state.masked_decode(params, selection)?;
            let params = view.as_ref().unwrap_or(params);
            return super::decode_activation_stage(
                gpu,
                params,
                selection.resolution,
                selection.route,
                self.state.shared_views().ok(),
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::activation(
                gpu,
                params,
                selection.uses_grouped_rows(),
            );
        }
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        super::prefill_activation_stage(gpu, params, &selection.resolution, total_slots)
    }

    pub(super) fn down(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            let view = self.state.masked_decode(params, selection)?;
            let params = view.as_ref().unwrap_or(params);
            let target = params.routed_out.unwrap_or(params.x_residual);
            // The down kernel itself folds into `target` (no expanded rows, no
            // later combine stage) on these routes; observe that boundary here.
            let fused_fold = if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial
                || selection.route.is_some()
            {
                None
            } else if selection.ninepath_d4 {
                Some("ninepath_d4")
            } else if selection.down_last_combine {
                Some("down_last_combine")
            } else if !crate::families::moe::moe_down_writes_expanded(
                params.dtypes.routed_down,
                params.expert_dtype_tags.is_some(),
            ) {
                Some("self_combining_down")
            } else {
                None
            };
            let dump = match fused_fold {
                Some(_) => super::moe_dump::fold_begin(
                    gpu,
                    params.layer_idx as usize,
                    params.hidden,
                    params.x_residual,
                    target,
                )?,
                None => None,
            };
            super::decode_down_stage(
                gpu,
                params,
                selection.resolution,
                target,
                selection.ninepath_d4,
                selection.ninepath_mq3l,
                selection.ninepath_mq4v2,
                selection.ninepath_mq6v2,
                selection.down_last_combine,
                selection.route,
            )?;
            if let Some(path) = fused_fold {
                super::moe_dump::fold_end(
                    gpu,
                    dump,
                    target,
                    params.routed_out.is_some(),
                    &super::moe_dump::FoldInfo {
                        path,
                        ep_mode: &format!("{:?}", params.ep_mode),
                        hidden: params.hidden,
                        slots: params.k,
                        expanded: None,
                        indices: params.topk_indices,
                        weights: params.topk_weights,
                        expanded_absent_reason:
                            "down kernel fused the weighted fold; no expanded rows were written",
                    },
                )?;
            }
            return Ok(());
        }
        let (params, selection) = self.state.prefill_parts()?;
        let view = self.state.masked_prefill(params, selection)?;
        let params = view.as_ref().unwrap_or(params);
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            return super::qt44_qt53_prefill::down(
                gpu,
                params,
                selection.uses_grouped_rows(),
                selection.path2_m_total,
            );
        }
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        super::prefill_down_stage(
            gpu,
            params,
            &selection.resolution,
            selection.path2_m_total,
            total_slots,
            selection.force_mq4_grouped_fp16,
            selection.coalesced_verify,
        )?;
        let compact_ep = matches!(
            &params.prelude.route,
            super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
                | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
        );
        if compact_ep && selection.uses_grouped_rows() {
            super::prefill_down_unscatter_stage(gpu, params, total_slots)?;
        }
        Ok(())
    }

    pub(super) fn combine(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if let Ok((params, selection)) = self.state.decode_parts() {
            if selection.cpu_split {
                return self.split_combine_decode(gpu, params, selection);
            }
            return super::decode_combine_stage(
                gpu,
                params,
                params.routed_out.unwrap_or(params.x_residual),
                selection.route,
            );
        }
        let (params, selection) = self.state.prefill_parts()?;
        if selection.cpu_split {
            return self.split_combine_prefill(gpu, params, selection);
        }
        if selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
        {
            // The row fold combines inside its own kernel; only the rank order
            // is owed here, and the absorbed target fill is not (the shared
            // down's rows overwrite the target).
            if self
                .state
                .hc_write
                .borrow()
                .as_ref()
                .is_some_and(|hc| hc.row_fold.is_some())
            {
                self.state.initial_zero.set(false);
                return super::qt44_qt53_prefill::combine_order_only(
                    gpu,
                    params,
                    selection.path2_m_total,
                );
            }
            return super::qt44_qt53_prefill::combine(
                gpu,
                params,
                selection.uses_grouped_rows(),
                selection.path2_m_total,
                self.state.initial_zero.take(),
            );
        }
        let canonical_slot_order = matches!(
            &params.prelude.route,
            super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
                | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
        );
        super::prefill_combine_stage(
            gpu,
            params,
            &selection.resolution,
            params.routed_out.unwrap_or(params.x_batch),
            canonical_slot_order,
        )
    }

    pub(super) fn host_experts(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, _) = self.state.decode_parts()?;
        let route = self.state.host_route.get().ok_or_else(|| {
            DispatchError::Hip("sealed moe: host experts consumed before host route".into())
        })?;
        super::decode_cpu_experts_stage(
            self.state.dispatch_ctx(),
            gpu,
            params,
            &route.indices,
            &route.weights,
        )
    }

    /// The CPU routed-expert splice, executed at the point the routed GPU tail
    /// would have run. The arch declares it (`cpu_deferred_experts`) and supplies
    /// the expert payload through `RoutedExpertWeights::cpu_splice_expert`; the
    /// seam owns the invocation order, the capture/replay refusal inside the
    /// executor and the coverage counters.
    pub(super) fn cpu_splice(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                let gpu_ref: &Gpu = &*gpu;
                cpu_exec::moe_cpu_experts(
                    gpu_ref,
                    params.hidden,
                    params.mi,
                    params.batch_size,
                    params.k,
                    params.x_rot_local,
                    params.topk_indices,
                    params.topk_weights,
                    params.x_residual,
                    params.expert_down_awq_ptrs.is_some(),
                    params.recipe.bf16_round_trip(),
                    |id| {
                        params
                            .routed_experts
                            .cpu_splice_expert(gpu_ref, id)
                            .ok_or_else(|| {
                                DispatchError::Hip(format!(
                                    "moe cpu expert splice: expert {id} has no CPU payload"
                                ))
                            })
                    },
                )
            }
            MoeProtocol::GroupedPrefill => {
                let (params, _) = self.state.prefill_parts()?;
                let gpu_ref: &Gpu = &*gpu;
                // `dim` is the model hidden size, which is exactly the routed down
                // projection's row count: `MoePrefillParams` has no `hidden` field
                // and `down_m` is built as `experts[0].down.m` (its `down_k` is
                // `mi`), matching `moe_cpu_experts`' `(dim, mi)` pair. Do not
                // "correct" this to a hidden field that does not exist.
                // The residual destination is the arch's unconditional
                // `&pbs.x_batch`: a CPU-deferred window never carries `routed_out`
                // (the sealed program refuses that combination), so the
                // `unwrap_or` below is defensive only.
                cpu_exec::moe_cpu_experts(
                    gpu_ref,
                    params.down_m,
                    params.mi,
                    params.batch_size,
                    params.k_top,
                    params.x_rot_batch,
                    params.topk_indices,
                    params.topk_weights,
                    params.routed_out.unwrap_or(params.x_batch),
                    params.expert_down_awq_ptrs.is_some(),
                    params.recipe.bf16_round_trip(),
                    |id| {
                        params
                            .routed_experts
                            .cpu_splice_expert(gpu_ref, id)
                            .ok_or_else(|| {
                                DispatchError::Hip(format!(
                                    "moe cpu expert splice: expert {id} has no CPU payload"
                                ))
                            })
                    },
                )
            }
        }
    }

    /// `PrepareSplit`: the one input-readiness wait of the eager CPU split.
    /// Every refusal (capture, retained/shadow bodies, PM4/AQL, route width,
    /// non-expanded down, shape) is raised before the executor touches any
    /// state; the executor then downloads the ORIGINAL ids/weights and the
    /// rotated activation, completes them, validates every cold span and
    /// uploads the separate masked ids. No routed GPU launch precedes it.
    pub(super) fn prepare_split(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.require_route_ready()?;
        self.state
            .require_split_phase(SplitPhase::Idle, "PrepareSplit")?;
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if !selection.cpu_split {
                    return Err(split_error(
                        "PrepareSplit lowered without the split selection",
                    ));
                }
                validate_split_decode(gpu, params, selection)?;
                let executor = decode_split_executor(params)?;
                executor.prepare(
                    &*gpu,
                    params.batch_size,
                    params.x_rot_local,
                    params.topk_indices,
                    params.topk_weights,
                    params.routed_experts,
                )?;
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if !selection.cpu_split {
                    return Err(split_error(
                        "PrepareSplit lowered without the split selection",
                    ));
                }
                validate_split_prefill(gpu, params, selection)?;
                let executor = prefill_split_executor(params)?;
                executor.prepare(
                    &*gpu,
                    params.batch_size,
                    params.x_rot_batch,
                    params.topk_indices,
                    params.topk_weights,
                    params.routed_experts,
                )?;
            }
        }
        self.state.split_phase.set(SplitPhase::Prepared);
        Ok(())
    }

    /// `JoinSplit`: the cold CPU arithmetic runs on the orchestration thread
    /// while the already queued masked GPU tail executes; the cold F32 slot
    /// rows are then queued on the same stream behind the GPU down (whose
    /// masked slots store zero). Never completes or synchronizes the stream.
    pub(super) fn join_split(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state
            .require_split_phase(SplitPhase::Prepared, "JoinSplit")?;
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, selection) = self.state.decode_parts()?;
                if !selection.cpu_split {
                    return Err(split_error("JoinSplit lowered without the split selection"));
                }
                decode_split_executor(params)?.join(
                    &*gpu,
                    params.batch_size,
                    params.down_expanded,
                    params.routed_experts,
                )?;
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                if !selection.cpu_split {
                    return Err(split_error("JoinSplit lowered without the split selection"));
                }
                prefill_split_executor(params)?.join(
                    &*gpu,
                    params.batch_size,
                    params.down_expanded,
                    params.routed_experts,
                )?;
            }
        }
        self.state.split_phase.set(SplitPhase::Joined);
        Ok(())
    }

    /// The canonical combine of a split call: the unchanged BF16 combine over
    /// the original ids/weights and the joined `down_expanded` slot rows, then
    /// (only while the lab parity record asks for it) the unmasked GPU tail is
    /// re-run into the now-dead `down_expanded`. The reference never re-combines.
    fn split_combine_decode(
        &self,
        gpu: &mut Gpu,
        params: &MoeParams<'_>,
        selection: &MoeDecodeSelection,
    ) -> Result<(), DispatchError> {
        self.state
            .require_split_phase(SplitPhase::Joined, "the canonical combine")?;
        let executor = decode_split_executor(params)?;
        let target = params.routed_out.unwrap_or(params.x_residual);
        let rows = params.batch_size;
        executor.parity_before_combine(&*gpu, rows, params.down_expanded, target)?;
        super::decode_combine_stage(gpu, params, target, selection.route)?;
        executor.parity_after_combine(&*gpu, rows, target)?;
        if executor.wants_reference() {
            self.split_reference_tail(gpu)?;
            executor.parity_reference(&*gpu, rows, params.down_expanded)?;
        }
        Ok(())
    }

    /// Prefill counterpart of [`Self::split_combine_decode`].
    fn split_combine_prefill(
        &self,
        gpu: &mut Gpu,
        params: &MoePrefillParams<'_>,
        selection: &MoePrefillSelection,
    ) -> Result<(), DispatchError> {
        self.state
            .require_split_phase(SplitPhase::Joined, "the canonical combine")?;
        // The absorbed zero fill belongs to the BF16-row grouped combine; the
        // split folds F32 slot rows and always accumulates into a filled target.
        if self.state.initial_zero.take() {
            return Err(split_error(
                "the canonical expanded combine cannot consume an absorbed target fill",
            ));
        }
        let executor = prefill_split_executor(params)?;
        let target = params.routed_out.unwrap_or(params.x_batch);
        let rows = params.batch_size;
        executor.parity_before_combine(&*gpu, rows, params.down_expanded, target)?;
        super::qt44_qt53_prefill::combine(
            gpu,
            params,
            selection.uses_grouped_rows(),
            selection.path2_m_total,
            false,
        )?;
        executor.parity_after_combine(&*gpu, rows, target)?;
        if executor.wants_reference() {
            self.split_reference_tail(gpu)?;
            executor.parity_reference(&*gpu, rows, params.down_expanded)?;
        }
        Ok(())
    }

    /// Re-run the SAME routed gate/up, activation and down stages over the
    /// ORIGINAL (unmasked) ids into `down_expanded`. Runs strictly after the
    /// canonical combine consumed the D2 rows and strictly before the shared
    /// down, so it writes only scratch that is dead at that point: routed
    /// `gate_batch`/`up_batch`/`rot_batch`, the grouped scatter/`y_*_grouped`
    /// scratch and `down_expanded`. The shared expert's hidden (`shared.rotated`
    /// and the gate/up outputs) must stay untouched: the fused routed
    /// activation / unscatter launches also activate the shared expert and
    /// sigmoid its selector scalar IN PLACE, which is NOT idempotent (a second
    /// pass yields `sigmoid(sigmoid(x))` before the shared down). The rerun
    /// therefore takes the reference view of [`Self::masked_decode`] /
    /// [`Self::masked_prefill`], which disables the shared side of those
    /// stages. No combine.
    fn split_reference_tail(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        self.state.split_reference.set(true);
        let result = self.split_reference_stages(gpu);
        self.state.split_reference.set(false);
        result
    }

    fn split_reference_stages(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let grouped = match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => false,
            MoeProtocol::GroupedPrefill => self.state.prefill_parts()?.1.uses_grouped_rows(),
        };
        if grouped {
            self.scatter(gpu)?;
        }
        self.gate_up(gpu)?;
        if grouped {
            self.unscatter(gpu)?;
        }
        self.activation(gpu)?;
        self.down(gpu)
    }

    pub(super) fn mutation_fence(&self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => super::decode_mutation_fence_stage(),
            MoeProtocol::GroupedPrefill => super::prefill_mutation_fence_stage(),
        }
    }

    pub(super) fn owner_assign(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let missing =
            || DispatchError::Hip("sealed moe: owner assignment lowered without a gather".into());
        let overflow =
            || DispatchError::Hip("sealed moe: owner assignment extent overflows".into());
        // The routed ids must exist before ownership is derived from them.
        self.state.require_route_ready()?;
        let (gather, layer_idx, slots, topk_indices) = match self.state.call.protocol() {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                (
                    params.ep_row_gather.ok_or_else(missing)?,
                    params.layer_idx,
                    params.k,
                    params.topk_indices,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let (params, _) = self.state.prefill_parts()?;
                (
                    params.ep_row_gather.ok_or_else(missing)?,
                    params.ep_layer_idx,
                    params
                        .batch_size
                        .checked_mul(params.k_top)
                        .ok_or_else(overflow)?,
                    params.topk_indices,
                )
            }
        };
        // Like the exchange it serves, the assignment is eager-only.
        if gpu.graphs.capture_mode
            || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some()
            || gpu.replay.is_recording()
        {
            return Err(DispatchError::Hip(
                "sealed moe: expert-parallel owner assignment refused while a graph capture or replay record is armed".into(),
            ));
        }
        gather
            .assign_owners(gpu, layer_idx, slots, topk_indices)
            .map_err(|error| {
                DispatchError::Hip(format!("sealed moe: EP owner assignment: {error}"))
            })
    }

    pub(super) fn row_gather(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let missing =
            || DispatchError::Hip("sealed moe: row gather lowered without a gather".into());
        let overflow = || DispatchError::Hip("sealed moe: row gather extent overflows".into());
        // The gather, the layer, the slot count and the producer's buffers, per
        // protocol: one decode row's `k` slots, or a few-row prefill's
        // `batch_size * k_top` slots in slot order.
        let (gather, layer_idx, slots, width, down_expanded, topk_indices) = match self
            .state
            .call
            .protocol()
        {
            MoeProtocol::IndexedDecode => {
                let (params, _) = self.state.decode_parts()?;
                let gather = params.ep_row_gather.ok_or_else(missing)?;
                (
                    gather,
                    params.layer_idx,
                    params.k,
                    params.hidden,
                    params.down_expanded,
                    params.topk_indices,
                )
            }
            MoeProtocol::GroupedPrefill => {
                let (params, selection) = self.state.prefill_parts()?;
                let gather = params.ep_row_gather.ok_or_else(missing)?;
                if !selection
                    .route
                    .is_some_and(MoeRouteCapability::is_sealed_grouped)
                    || !super::qt44_qt53_prefill::writes_slot_ordered_rows(
                        gpu,
                        params,
                        selection.uses_grouped_rows(),
                    )
                {
                    return Err(DispatchError::Hip(
                            "sealed moe: an EP row gather needs the grouped QT44/QT53 down to have written slot-ordered expanded rows".into(),
                        ));
                }
                // The unfused shared down runs after the combine and would
                // overwrite rows a peer may still be reading.
                let shared_out = params.shared_down_out.ok_or_else(|| {
                        DispatchError::Hip(
                            "sealed moe: a prefill EP row gather requires a separate shared-down output".into(),
                        )
                    })?;
                if shared_out.buf.as_ptr() == params.down_expanded.buf.as_ptr() {
                    return Err(DispatchError::Hip(
                        "sealed moe: the shared-down output aliases the gathered expanded rows"
                            .into(),
                    ));
                }
                (
                    gather,
                    params.ep_layer_idx,
                    params
                        .batch_size
                        .checked_mul(params.k_top)
                        .ok_or_else(overflow)?,
                    params.down_m,
                    params.down_expanded,
                    params.topk_indices,
                )
            }
        };
        // The exchange waits on peer devices and a host barrier; neither can
        // be recorded into a graph or retained body. EP is eager-only.
        if gpu.graphs.capture_mode
            || gpu.graphs.verify.capturing.is_some()
            || gpu.graphs.replay.capturing.is_some()
            || gpu.replay.is_recording()
        {
            return Err(DispatchError::Hip(
                "sealed moe: expert-parallel row gather refused while a graph capture or replay record is armed".into(),
            ));
        }
        let rows = slots.checked_mul(width).ok_or_else(overflow)?;
        let down_expanded = super::slice_moe_f32_view(down_expanded, 0, rows);
        gather
            .gather_rows(gpu, layer_idx, slots, &down_expanded, topk_indices)
            .map_err(|error| DispatchError::Hip(format!("sealed moe: EP row gather: {error}")))
    }

    pub(super) fn tile_mask(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        let tx = params.ep_tile_exchange.ok_or_else(|| {
            DispatchError::Hip("sealed moe: tile mask lowered without a tile exchange".into())
        })?;
        self.state.require_route_ready()?;
        super::qt44_qt53_prefill::tile_mask(gpu, params, tx, selection.path2_m_total)
    }

    pub(super) fn tile_exchange(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let (params, selection) = self.state.prefill_parts()?;
        let tx = params.ep_tile_exchange.ok_or_else(|| {
            DispatchError::Hip("sealed moe: tile exchange lowered without a tile exchange".into())
        })?;
        super::qt44_qt53_prefill::tile_exchange(gpu, params, tx, selection.path2_m_total)
    }
}

/// Append one borrowed instruction without allowing SmallVec to spill.
fn append_step<'a>(
    steps: &mut SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>,
    step: Step<'a>,
) -> Result<(), DispatchError> {
    if steps.len() >= INLINE_STEP_CAPACITY {
        return Err(DispatchError::Hip(
            "sealed moe: step program exceeds inline capacity".into(),
        ));
    }
    steps.push(step);
    Ok(())
}

fn lower_decode<'a>(
    state: &'a MoeStepState<'a>,
) -> Result<SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>, DispatchError> {
    let (params, selection) = state.decode_parts()?;
    if selection.cpu_split {
        if let Some(reason) = cpu_split_decode_conflict(
            params,
            selection.route,
            selection.cpu_deferred_experts,
            // The executor-dependent shape check ran at seal and runs again in
            // `validate_split_decode`; lowering needs no device object.
            None,
        ) {
            return Err(split_error(reason));
        }
    }
    let mut steps = SmallVec::new();
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Normalize))?;
    let adopted = matches!(
        state.call.router_input(),
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    // Balanced EP ownership: the stage goes right after the stage that writes
    // the routed ids and before every routed-table reader. A selection with no
    // such stage in this program (an adopted route) cannot be balanced.
    let owner_assign = params
        .ep_row_gather
        .is_some_and(|gather| gather.assigns_owners(params.layer_idx));
    match params.recipe {
        MoeRecipe::SoftmaxGatedShared { .. } => {
            if selection.resolution.needs_x_rot_local {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            }
            if !adopted {
                // Qwen's gate-side operation preserves the existing fused
                // router/selector/gate/up launch when its static quartet is
                // admitted; the route operation then owns top-k production.
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateSide))?;
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
                if owner_assign {
                    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::OwnerAssign))?;
                }
            }
            if !params.skip_shared && !params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
        }
        MoeRecipe::SigmoidRoutedNoShared => {
            if !adopted {
                append_step(
                    &mut steps,
                    Step::MoeStage(op(state), MoeStage::RouterProjection),
                )?;
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
                if owner_assign {
                    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::OwnerAssign))?;
                }
            }
            // Cohere's sigmoid route consumes the normalized natural input
            // first and transforms it only after top-eight selection.
            if selection.resolution.needs_x_rot_local {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            }
        }
    }
    if owner_assign && (adopted || !selection.resolution.use_gpu_topk) {
        return Err(DispatchError::Hip(
            "sealed moe: balanced EP ownership needs this program's GPU route stage to write the routed ids".into(),
        ));
    }
    if selection.resolution.use_gpu_topk {
        // A CPU-deferred call keeps the GPU router/rotation/shared stages, but
        // must not lower any routed GPU stage: the arch's CPU splice supplies
        // the routed contribution into the residual after this program, and the
        // expanded routed scratch is not refreshed here, so combining it would
        // fold stale rows. The mutation fence still marks the routed-stage
        // boundary the splice completes.
        if selection.cpu_split {
            // The sole input-readiness wait: after the router wrote the
            // original route and the rotated input exists, strictly before the
            // first routed GPU launch.
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::PrepareSplit),
            )?;
        }
        if !selection.cpu_deferred_experts {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
        }
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::MutationFence),
        )?;
        if selection.cpu_split {
            // The cold slot rows are queued behind the GPU down (same stream)
            // and before the canonical combine of the original route.
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::JoinSplit))?;
        }
        let down_self_combines = selection.down_last_combine
            || !crate::families::moe::moe_down_writes_expanded(
                params.dtypes.routed_down,
                params.expert_dtype_tags.is_some(),
            );
        let combine_after_down = !selection.cpu_deferred_experts
            && params.ep_mode != crate::families::moe::MoeEpMode::RootRoutedPartial
            && !selection.ninepath_d4
            && (selection
                .route
                .is_some_and(MoeRouteCapability::is_sealed_grouped)
                || !down_self_combines)
            && !params.defer_routed_combine;
        if selection.cpu_split && !combine_after_down {
            return Err(split_error(
                "requires the canonical combine over expanded slot rows (no fused, ninepath or self-combining down)",
            ));
        }
        if params.ep_row_gather.is_some() {
            // The gather fills exactly the ten expanded rows the grouped
            // indexed down wrote; every other decode lowering either folds
            // inside its down kernel or follows a different EP contract.
            let grouped = selection
                .route
                .is_some_and(MoeRouteCapability::is_sealed_grouped);
            if !grouped
                || !combine_after_down
                || params.batch_size != 1
                || params.ep_mode != crate::families::moe::MoeEpMode::None
            {
                return Err(DispatchError::Hip(
                    "sealed moe: an EP row gather requires the one-row grouped QT44/QT53 decode with its canonical combine".into(),
                ));
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::RowGather))?;
        }
        // A CPU-deferred call lowers no routed `Combine`: the splice takes its
        // place. A `shared_after_combine` recipe (Qwen4) folds the routed
        // contribution first and the shared expert second, so the splice runs
        // before `SharedDown`; the other recipes keep the splice last, exactly
        // where the arch ran it before the stage existed. The splice and
        // `SharedDown` accumulate into the same residual, so each arrangement
        // keeps its recipe's fp accumulation sequence.
        let splice_first = selection.cpu_deferred_experts && params.recipe.shared_after_combine();
        if splice_first {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
        }
        if combine_after_down {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
        }
        if params.recipe.shared_after_combine() && !params.skip_shared {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
        if selection.cpu_deferred_experts && !splice_first {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
        }
    } else {
        if params.ep_row_gather.is_some() {
            return Err(DispatchError::Hip(
                "sealed moe: an EP row gather requires GPU top-k routing".into(),
            ));
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::HostExperts))?;
        if params.recipe.shared_after_combine() && !params.skip_shared {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
    }
    Ok(steps)
}

fn lower_prefill<'a>(
    state: &'a MoeStepState<'a>,
) -> Result<SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]>, DispatchError> {
    let (params, selection) = state.prefill_parts()?;
    if selection.cpu_split {
        // Checked before any other lowering decision: the mode must never
        // fall into the generic or deferred arms by accident.
        if let Some(reason) = cpu_split_prefill_conflict(
            params,
            selection.route,
            selection.cpu_deferred_experts,
            // The executor-dependent shape check ran at seal and runs again in
            // `validate_split_prefill`; lowering needs no device object.
            None,
        ) {
            return Err(split_error(reason));
        }
    }
    let mut steps = SmallVec::new();
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Normalize))?;
    if params.ep_row_gather.is_some()
        && (!selection
            .route
            .is_some_and(MoeRouteCapability::is_sealed_grouped)
            || params.routed_out.is_some())
    {
        // The gather fills the expanded rows the grouped QT44/QT53 down wrote,
        // for the canonical combine into the residual; every other prefill
        // lowering has a different row layout or EP contract.
        return Err(DispatchError::Hip(
            "sealed moe: a prefill EP row gather requires the grouped QT44/QT53 route combining into the residual".into(),
        ));
    }
    if params.ep_tile_exchange.is_some()
        && (!selection
            .route
            .is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
            || !selection.uses_grouped_rows()
            || params.routed_out.is_some()
            || params.ep_row_gather.is_some())
    {
        // The exchange moves the BF16 grouped rows of the symmetric IU4 down or of the F16 WMMA
        // grouped down (the gfx1100 route) for the canonical combine into the residual; the
        // stages re-check the exact arm against the device (`qt44_qt53_prefill::tile_mask`).
        // Every other prefill lowering has a different row layout, a different down kernel or
        // EP contract.
        return Err(DispatchError::Hip(
            "sealed moe: a prefill EP tile exchange requires the grouped QT44/QT53 route (symmetric IU4 or F16 WMMA) combining into the residual".into(),
        ));
    }
    if selection
        .route
        .is_some_and(MoeRouteCapability::is_sealed_grouped)
    {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::RouterProjection),
        )?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::SharedGateUp),
        )?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::SharedActivation),
        )?;
        if !params.recipe.shared_after_combine() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
        // Balanced EP ownership: right after the route wrote the ids, before
        // the scatter and every routed-table reader.
        if params
            .ep_row_gather
            .is_some_and(|gather| gather.assigns_owners(params.ep_layer_idx))
        {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::OwnerAssign))?;
        }
        if selection.cpu_split {
            // Distinct eager CPU-cold/GPU-hot split. The original route feeds
            // the unchanged canonical combine and the shared expert; only the
            // routed GPU tail between `PrepareSplit` and `JoinSplit` reads the
            // executor's masked ids. `PrepareSplit` is the sole input-readiness
            // wait and precedes the first routed GPU launch; `JoinSplit` runs
            // the cold CPU arithmetic while that tail executes and queues the
            // cold F32 slot rows behind the GPU down, before the combine.
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::PrepareSplit),
            )?;
            if selection.uses_grouped_rows() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Scatter))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
            if selection.uses_grouped_rows() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Unscatter))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::MutationFence),
            )?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::JoinSplit))?;
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
            if params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
            return Ok(steps);
        }
        if selection.cpu_deferred_experts {
            // The arch's CPU splice owns the routed FFN for this narrow window;
            // lowering the grouped routed tail would read/write scratch the
            // splice does not consume and fold stale expanded rows. The fence
            // still marks the routed-stage boundary the splice completes.
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::MutationFence),
            )?;
            // The splice stands where the routed `Combine` would be: before a
            // `shared_after_combine` recipe's `SharedDown` (routed first, shared
            // second, like the GPU tail), and last for the other recipes whose
            // shared expert already ran above. Both accumulate into one
            // residual, so this keeps each recipe's fp accumulation sequence.
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
            if params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
            return Ok(steps);
        }
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Scatter))?;
        }
        if params.ep_tile_exchange.is_some() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::TileMask))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Unscatter))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
        append_step(
            &mut steps,
            Step::MoeStage(op(state), MoeStage::MutationFence),
        )?;
        if params.ep_row_gather.is_some() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::RowGather))?;
        }
        if params.ep_tile_exchange.is_some() {
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::TileExchange),
            )?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
        if params.recipe.shared_after_combine() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
        }
        return Ok(steps);
    }
    let adopted = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    match params.recipe {
        MoeRecipe::SoftmaxGatedShared { .. } => {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
            if !adopted {
                append_step(
                    &mut steps,
                    Step::MoeStage(op(state), MoeStage::RouterProjection),
                )?;
            }
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::SharedGateUp),
            )?;
            append_step(
                &mut steps,
                Step::MoeStage(op(state), MoeStage::SharedActivation),
            )?;
            if !params.recipe.shared_after_combine() {
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
        }
        MoeRecipe::SigmoidRoutedNoShared => {
            if !adopted {
                append_step(
                    &mut steps,
                    Step::MoeStage(op(state), MoeStage::RouterProjection),
                )?;
                append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
            }
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
        }
    }
    if !selection.cpu_deferred_experts {
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Scatter))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateUp))?;
        if selection.uses_grouped_rows() {
            append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Unscatter))?;
        }
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Activation))?;
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Down))?;
    }
    append_step(
        &mut steps,
        Step::MoeStage(op(state), MoeStage::MutationFence),
    )?;
    let compact_ep = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
            | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    // A CPU-deferred window must not combine: the expanded routed scratch was
    // not refreshed by this program, so folding it would publish stale rows.
    // The arch's CPU splice accumulates the routed contribution into `x_batch`.
    if !selection.cpu_deferred_experts
        && !compact_ep
        && (selection.uses_grouped_rows() || !selection.resolution.down_path0)
    {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Combine))?;
    }
    let splice_first = selection.cpu_deferred_experts && params.recipe.shared_after_combine();
    if splice_first {
        // Where the routed `Combine` would be: a `shared_after_combine` recipe
        // folds the routed contribution before the shared expert.
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
    }
    if params.recipe.shared_after_combine() {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::SharedDown))?;
    }
    if selection.cpu_deferred_experts && !splice_first {
        // The shared expert ran before the routed tail for this recipe, so the
        // splice is the last stage, as the arch's own splice was.
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::CpuSplice))?;
    }
    Ok(steps)
}

/// Lower and execute a sealed call using the shared Step interpreter.
pub(super) fn execute(gpu: &mut Gpu, call: &SealedMoeCall<'_>) -> Result<(), DispatchError> {
    // Reborrow the call at the execution lifetime. The state and every Step
    // borrow only this invocation and are dropped before the call returns.
    let state = MoeStepState::new(call);
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// The stages of the S1 layer-major PREPARE half of a sealed grouped prefill, in
/// lowering order: everything [`lower_prefill`] runs before its routed tail
/// (`Scatter` .. `Combine`) for a `shared_after_combine` recipe.  The routed tail
/// runs later, window-wide, from `slowlink_prefill`; the shared down runs after
/// it from `slowlink_prefill::finish_shared`.
pub(super) const PREPARE_STAGES: [MoeStage; 6] = [
    MoeStage::Normalize,
    MoeStage::InputBasis,
    MoeStage::RouterProjection,
    MoeStage::SharedGateUp,
    MoeStage::SharedActivation,
    MoeStage::Route,
];

/// Run [`PREPARE_STAGES`] of a sealed grouped prefill call (one tile of the S1
/// window) through the shared Step interpreter.  Refuses, before any launch,
/// every call whose lowering is not exactly the plain grouped tail followed by a
/// shared down after the combine.
pub(super) fn execute_prepare(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    let state = MoeStepState::new(call);
    if call.protocol() == MoeProtocol::IndexedDecode {
        return execute_prepare_decode(gpu, call, &state);
    }
    let (params, selection) = state.prefill_parts()?;
    if !selection
        .route
        .is_some_and(MoeRouteCapability::is_sealed_grouped)
        || selection.cpu_split
        || selection.cpu_deferred_experts
        || !params.recipe.shared_after_combine()
        || params.ep_row_gather.is_some()
        || params.ep_tile_exchange.is_some()
        || params.routed_out.is_some()
    {
        return Err(DispatchError::Hip(
            "slowlink prepare: the call is not a plain sealed grouped prefill with the shared down after the combine".into(),
        ));
    }
    let mut steps: SmallVec<[Step<'_>; INLINE_STEP_CAPACITY]> = SmallVec::new();
    for stage in PREPARE_STAGES {
        append_step(&mut steps, Step::MoeStage(SealedMoeOp::new(&state), stage))?;
    }
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// The decode (one-row, `route_rows == 1`) twin of [`execute_prepare`]: the incumbent's
/// own decode prelude up to and including the route (normalize, the optional input basis,
/// the fused router/shared gate-side stage, the GPU top-k), and nothing of its routed tail
/// or shared down.  The routed tail of a one-row tile is the K1 cohort of `slowlink_prefill`.
fn execute_prepare_decode<'a>(
    gpu: &mut Gpu,
    call: &'a SealedMoeCall<'a>,
    state: &'a MoeStepState<'a>,
) -> Result<(), DispatchError> {
    let (params, selection) = state.decode_parts()?;
    let adopted = matches!(
        call.router_input(),
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    if !selection
        .route
        .is_some_and(MoeRouteCapability::is_sealed_grouped)
        || !matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        || !params.recipe.shared_after_combine()
        || !selection.resolution.use_gpu_topk
        || adopted
        || selection.cpu_split
        || selection.cpu_deferred_experts
        || selection.down_last_combine
        || selection.ninepath_d3
        || selection.ninepath_d4
        || params.ep_row_gather.is_some()
        || params.ep_mode != crate::families::moe::MoeEpMode::None
        || params.routed_out.is_some()
        || params.defer_routed_combine
        || params.skip_shared
        || params.shared.is_none()
        || params.batch_size != 1
        || params.k != 10
    {
        return Err(DispatchError::Hip(
            "slowlink prepare: the one-row call is not a plain sealed grouped decode with the shared down after the combine".into(),
        ));
    }
    let op = |state: &'a MoeStepState<'a>| SealedMoeOp::new(state);
    let mut steps: SmallVec<[Step<'a>; INLINE_STEP_CAPACITY]> = SmallVec::new();
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Normalize))?;
    if selection.resolution.needs_x_rot_local {
        append_step(&mut steps, Step::MoeStage(op(state), MoeStage::InputBasis))?;
    }
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::GateSide))?;
    append_step(&mut steps, Step::MoeStage(op(state), MoeStage::Route))?;
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)
}

/// The MoE stages a call lowers to, in order (test probe: nothing launches).
#[cfg(test)]
pub(super) fn lowered_stages(call: &SealedMoeCall<'_>) -> Result<Vec<MoeStage>, DispatchError> {
    let state = MoeStepState::new(call);
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    Ok(steps
        .iter()
        .filter_map(|step| match step {
            Step::MoeStage(_, stage) => Some(*stage),
            _ => None,
        })
        .collect())
}

/// The split transaction's phase discipline for a call sealed in the split
/// mode (test probe: nothing launches and no executor is needed). The routed
/// GPU tail may not take the masked view before `PrepareSplit`, `PrepareSplit`
/// and `JoinSplit` each run exactly once and in order, the combine needs the
/// joined phase, and the post-combine reference keeps the original routed
/// operands with every shared-expert side effect disabled.
#[cfg(test)]
pub(super) fn split_phase_probe(call: &SealedMoeCall<'_>) -> Result<(), String> {
    let state = MoeStepState::new(call);
    if let Ok((params, selection)) = state.decode_parts() {
        if state.masked_decode(params, selection).is_ok() {
            return Err("decode tail took the masked view before PrepareSplit".into());
        }
    }
    if let Ok((params, selection)) = state.prefill_parts() {
        if state.masked_prefill(params, selection).is_ok() {
            return Err("prefill tail took the masked view before PrepareSplit".into());
        }
    }
    for want in [SplitPhase::Prepared, SplitPhase::Joined] {
        if state.require_split_phase(want, "probe").is_ok() {
            return Err(format!("{want:?} accepted from Idle"));
        }
    }
    state
        .require_split_phase(SplitPhase::Idle, "PrepareSplit")
        .map_err(|e| format!("{e:?}"))?;
    state.split_phase.set(SplitPhase::Prepared);
    if state
        .require_split_phase(SplitPhase::Idle, "PrepareSplit")
        .is_ok()
    {
        return Err("PrepareSplit ran twice".into());
    }
    if state
        .require_split_phase(SplitPhase::Joined, "Combine")
        .is_ok()
    {
        return Err("the canonical combine ran before JoinSplit".into());
    }
    state
        .require_split_phase(SplitPhase::Prepared, "JoinSplit")
        .map_err(|e| format!("{e:?}"))?;
    state.split_phase.set(SplitPhase::Joined);
    if state
        .require_split_phase(SplitPhase::Prepared, "JoinSplit")
        .is_ok()
    {
        return Err("JoinSplit ran twice".into());
    }
    state
        .require_split_phase(SplitPhase::Joined, "Combine")
        .map_err(|e| format!("{e:?}"))?;
    state.split_reference.set(true);
    // The reference rerun replays gate/up/unscatter/activation/down with the
    // ORIGINAL route, so its view must (a) exist, (b) keep every routed operand
    // of the original call, and (c) have the shared-expert side effects of
    // those stages disabled: the fused activation / unscatter launches
    // sigmoid the shared selector scalar IN PLACE, and the original call's
    // shared stage has already consumed (and will not recompute) it.
    if let Ok((params, selection)) = state.decode_parts() {
        if !super::qt44_shared_activation_in_routed(params, selection.route) {
            return Err("decode fixture does not exercise the fused shared activation".into());
        }
        let Ok(Some(view)) = state.masked_decode(params, selection) else {
            return Err("decode reference rerun has no shared-disabled view".into());
        };
        let same = [
            (view.topk_indices, params.topk_indices),
            (view.topk_weights, params.topk_weights),
            (view.expert_gate_up_ptrs, params.expert_gate_up_ptrs),
            (view.expert_down_ptrs, params.expert_down_ptrs),
            (view.gate_batch, params.gate_batch),
            (view.up_batch, params.up_batch),
            (view.rot_batch, params.rot_batch),
            (view.down_expanded, params.down_expanded),
        ];
        if !same.iter().all(|(a, b)| std::ptr::eq(*a, *b)) {
            return Err("decode reference view does not keep the original routed operands".into());
        }
        if !view.skip_shared {
            return Err("decode reference view still runs the shared expert side effects".into());
        }
        if super::qt44_shared_activation_in_routed(&view, selection.route) {
            return Err("decode reference view still fuses the shared activation".into());
        }
        if params.skip_shared {
            return Err("decode reference view mutated the original call".into());
        }
    }
    if let Ok((params, selection)) = state.prefill_parts() {
        if params.prelude.shared.is_none() {
            return Err("prefill fixture does not carry a shared expert".into());
        }
        let Ok(Some(view)) = state.masked_prefill(params, selection) else {
            return Err("prefill reference rerun has no shared-disabled view".into());
        };
        let same = [
            (view.topk_indices, params.topk_indices),
            (view.topk_weights, params.topk_weights),
            (view.expert_gate_up_ptrs, params.expert_gate_up_ptrs),
            (view.expert_down_ptrs, params.expert_down_ptrs),
            (view.gate_batch, params.gate_batch),
            (view.up_batch, params.up_batch),
            (view.rot_batch, params.rot_batch),
            (view.down_expanded, params.down_expanded),
        ];
        if !same.iter().all(|(a, b)| std::ptr::eq(*a, *b)) {
            return Err("prefill reference view does not keep the original routed operands".into());
        }
        if view.prelude.shared.is_some() {
            return Err("prefill reference view still runs the shared expert side effects".into());
        }
        if params.prelude.shared.is_none() {
            return Err("prefill reference view mutated the original call".into());
        }
    }
    Ok(())
}

/// [`execute`] for a call preceded by `Step::Clear(clear)` and, optionally,
/// followed by the HC write `hc` whose gates are ready.
///
/// H8a (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`): when the qt44 combine can start
/// from +0.0 into exactly the cleared target, the zero fill is not launched;
/// otherwise it runs here, first, exactly as the separate step would have.
/// The decision is made once from the same predicate the combine stage
/// re-checks, and the call fails loudly if an absorbed fill was not consumed
/// by the combine (the target would otherwise be left uncleared).
///
/// H4 level 3 (`HIPFIRE_QWEN4_HC_FUSE` >= 3): `hc` is offered to the shared
/// down stage, which folds it into its GEMM epilogue when it fits.  Returns
/// whether it did; the caller then skips the write.  When the stage declines
/// (or never runs), nothing was launched for it and the write runs unfused.
///
/// `HIPFIRE_QWEN4_HC_ROW_FOLD`: with `next` (the HC read right after `hc` and
/// that read's paired write, both already admitted by the step interpreter),
/// the shared down runs the whole tail as one row kernel when
/// [`super::qt44_qt53_prefill::row_fold_applies`]; the result then reports
/// `row_folded` and the read's norm + its write's gates are already done.
pub(super) fn execute_after_clear(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
    clear: &super::layer_ops::ClearOp<'_>,
    hc: Option<&super::layer_ops::HyperWriteOp<'_>>,
    next: Option<(
        &super::layer_ops::HyperReadOp<'_>,
        &super::layer_ops::HyperWriteOp<'_>,
    )>,
) -> Result<AfterClear, DispatchError> {
    let state = MoeStepState::new(call);
    let absorbed = state.absorbs_clear(gpu, clear);
    if !absorbed {
        super::layer_ops::execute_clear(gpu, clear)?;
    }
    state.initial_zero.set(absorbed);
    if let Some(write) = hc {
        let mut offered = super::qt44_qt53_prefill::HcSharedDown::offer(write);
        if let (Some(offered), Some((read, next_write))) = (offered.as_mut(), next) {
            if let Ok((params, selection)) = state.prefill_parts() {
                if selection
                    .route
                    .is_some_and(MoeRouteCapability::is_sealed_grouped)
                    && !selection.cpu_split
                    && super::qt44_qt53_prefill::row_fold_applies(
                        gpu,
                        params,
                        selection.uses_grouped_rows(),
                        offered,
                    )
                {
                    offered.row_fold = Some(super::qt44_qt53_prefill::HcRowFoldNext::offer(
                        read, next_write,
                    ));
                }
            }
        }
        *state.hc_write.borrow_mut() = offered;
    }
    let steps = match call.protocol() {
        MoeProtocol::IndexedDecode => lower_decode(&state)?,
        MoeProtocol::GroupedPrefill => lower_prefill(&state)?,
    };
    super::steps::execute_validated_steps(gpu, call.dispatch_ctx(), &steps)?;
    if state.initial_zero.get() {
        return Err(DispatchError::Hip(
            "sealed moe: the absorbed target clear was not consumed by the combine".into(),
        ));
    }
    Ok(AfterClear {
        folded: state.hc_folded.get(),
        row_folded: state.hc_row_folded.get(),
    })
}

/// What [`execute_after_clear`] did beyond the call itself.
pub(super) struct AfterClear {
    /// The shared down carried the call's HC write.
    pub folded: bool,
    /// ... and the next HC read's norm + its write's gates (the HC row fold).
    pub row_folded: bool,
}

/// Execute only the canonical slot-order combine for a validated EP call.
pub(super) fn execute_ep_slot_combine(
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    let state = MoeStepState::new(call);
    SealedMoeOp::new(&state).combine(gpu)
}

/// The specialized wire path does not apply per-projection transformations.
fn projection_has_sidecars(
    router: &crate::families::gemv::WeightRef<'_>,
    shared: &crate::families::moe::MoeSharedWeights<'_>,
) -> bool {
    [
        router,
        &shared.selector,
        &shared.gate,
        &shared.up,
        &shared.down,
    ]
    .into_iter()
    .any(|weight| weight.rotation.is_some() || weight.awq_scale.is_some())
}

/// Resolve all decode choices without a GPU. Dynamic GPU capture state remains
/// in `SealedMoeCall::validate_for_gpu` so capture refusal happens before the
/// first launch rather than during sealing.
pub(super) fn select_decode(
    ctx: &DispatchCtx,
    params: &MoeParams<'_>,
    router: MoeRouterInput,
) -> Result<MoeDecodeSelection, DispatchError> {
    super::check_moe_decode_batch_size(params.batch_size)?;
    // Qt55 decode uses the grouped protocol even for one row. An indexed
    // descriptor cannot supply its stable permutation and must fail closed.
    super::reject_fp8_moe(&params.dtypes)?;
    if params.recipe == MoeRecipe::SigmoidRoutedNoShared
        && (!super::sealed_moe::sigmoid_decode_width_ok(params)
            || params.ep_mode != crate::families::moe::MoeEpMode::None
            || params.defer_routed_combine)
    {
        return Err(DispatchError::Hip(
            "sigmoid MoE decode requires k=8 (k<=8 with a selection bias), no EP, and no deferred combine"
                .into(),
        ));
    }
    let mut resolution = MoeResolution::resolve_arch(&params.dtypes, params.k, ctx.arch.has_wmma());
    let route = (matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        && params.batch_size == 1
        && params.ep_mode == crate::families::moe::MoeEpMode::None
        && params.routed_gate_up_paro.is_none()
        && params.routed_down_paro.is_none()
        && !params.defer_routed_combine)
        .then(|| {
            super::select_grouped_route(
                &params.dtypes,
                params.n_exp,
                params.k,
                params.hidden,
                params.mi,
                params
                    .shared
                    .as_ref()
                    .map_or(0, |shared| shared.intermediate),
                params.routed_gate_up_k,
                params.routed_down_m,
                params.routed_down_k,
                params.expert_dtype_tags,
                params.expert_down_awq_ptrs.is_some()
                    || params.shared.as_ref().is_some_and(|shared| {
                        projection_has_sidecars(&params.router, &shared.weights)
                    }),
                params.route_policy,
                ctx.arch.arch(),
            )
        })
        .flatten();
    if params.route_policy.is_some() && route.is_none() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "specialized-decode-route-unsupported",
            arch: "",
            quant: "QT44/QT53",
        });
    }
    if route.is_none() {
        super::reject_mq4g128v2_moe(&params.dtypes)?;
    } else {
        resolution.use_gpu_topk = true;
        resolution.needs_x_rot_local = true;
    }
    if params.selection_bias.is_some() {
        // The bias-aware selector writes any width up to eight. At top-8 every
        // generic indexed route applies; below it, of the indexed routed
        // kernels only the uniform FWHT-basis MQ4G256 and MQ6G256 gate/up,
        // activation, down and combine read that width at run time.
        use rdna_compute::DType::{MQ4G256, MQ6G256};
        let runtime_width = matches!(
            (params.dtypes.routed_gate_up, params.dtypes.routed_down),
            (MQ4G256, MQ4G256) | (MQ6G256, MQ6G256)
        ) && !params.dtypes.routed_has_mixed_experts
            && params.expert_dtype_tags.is_none()
            && params.expert_down_awq_ptrs.is_none()
            && params.routed_gate_up_paro.is_none();
        if params.k < MOE_BIASED_SIGMOID_MAX_K && !runtime_width {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "biased-sigmoid-router-needs-runtime-width-experts",
                arch: "",
                quant: "",
            });
        }
        resolution.use_gpu_topk = true;
    }
    super::check_moe_decode_supported(
        resolution.use_gpu_topk,
        params.k,
        params.n_exp,
        !params.routed_experts.is_empty(),
    )?;
    if params.cpu_deferred_experts {
        // The arch's CPU splice consumes the GPU-produced route and the rotated
        // input, then accumulates the routed FFN into `x_residual` after this
        // program. It therefore needs the GPU top-K path (never the per-expert
        // host GEMV fallback), a rotated activation, a single-device non-EP
        // call, and a recipe whose routed combine owes no bf16 round trip.
        if !resolution.use_gpu_topk || !resolution.needs_x_rot_local {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts require the GPU top-K path with a rotated local activation"
                    .into(),
            ));
        }
        if params.ep_mode != crate::families::moe::MoeEpMode::None
            || params.routed_out.is_some()
            || params.defer_routed_combine
        {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts require a single-device call with no EP partial and no deferred routed combine"
                    .into(),
            ));
        }
        // A BF16 recipe is admitted only on the grouped QT44/QT53 route: the
        // router and shared stages this program still lowers carry its BF16
        // round trips themselves, and the omitted routed stages' round trips
        // (gate/up, activation, weighted combine) are the CPU splice's, which
        // receives the recipe flag. The generic route also rounds the rotated
        // input, which the CPU does not emulate.
        if params.recipe.bf16_round_trip()
            && !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts emulate a recipe's bf16 round trips only on the grouped QT44/QT53 route"
                    .into(),
            ));
        }
    }
    // The distinct CPU-cold/GPU-hot split of the one-row decode. Every
    // admission refusal is raised here, before the call is published.
    let cpu_split =
        match crate::families::moe::cpu_split_executor(params.routed_experts, params.batch_size) {
            Some(executor) => {
                if let Some(reason) = cpu_split_decode_conflict(
                    params,
                    route,
                    params.cpu_deferred_experts,
                    Some(executor),
                ) {
                    return Err(DispatchError::Hip(format!(
                        "sealed moe: CPU tier split: {reason}"
                    )));
                }
                if !resolution.use_gpu_topk || !resolution.needs_x_rot_local {
                    return Err(split_error(
                        "requires the GPU top-K path with a rotated local activation",
                    ));
                }
                true
            }
            None => false,
        };
    if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
        if !resolution.use_gpu_topk {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires the GPU top-K path; the CPU-top-K fallback is rejected"
                    .into(),
            ));
        }
        if params.defer_routed_combine {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires defer_routed_combine=false (weighted combine into the partial)"
                    .into(),
            ));
        }
        if params.routed_out.is_none() {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires routed_out=Some (the zeroed partial)".into(),
            ));
        }
        if !crate::families::moe::moe_down_writes_expanded(
            params.dtypes.routed_down,
            params.expert_dtype_tags.is_some(),
        ) {
            return Err(DispatchError::Hip(
                "root-routed EP decode requires an expanded-output down projection".into(),
            ));
        }
    }
    let skip_routing = matches!(
        router,
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    );
    if !resolution.use_gpu_topk {
        if skip_routing {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "cpu-topk-fallback-needs-gpu-route-for-precomputed-input",
                arch: "",
                quant: "",
            });
        }
        if params.routed_out.is_some() {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "ep-routed-out-unsupported-in-cpu-topk-fallback",
                arch: "",
                quant: "",
            });
        }
        if params.routed_experts.is_empty() {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "cpu-topk-fallback-needs-resident-experts",
                arch: "",
                quant: "",
            });
        }
    }
    if resolution.needs_x_rot_local
        && !params.x_rot_prerotated
        && resolution.routed_indexable_paro
        && params.routed_gate_up_paro.is_none()
    {
        return Err(DispatchError::Hip(
            "sealed moe: Paro decode program has no gate-up sidecar".into(),
        ));
    }

    let gfx1100_router_mode = hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64").ok();
    let exact_wave64_router = params.n_exp == 256
        && ((ctx.arch.is_gfx1100()
            && !matches!(gfx1100_router_mode.as_deref(), Some("0" | "approx")))
            || ctx.arch.is_gfx1151());
    let wave64_router = (ctx.arch.is_gfx1201()
        && hipfire_config::developer_var("HIPFIRE_GFX1201_ROUTER_W64").as_deref() != Ok("0"))
        || (ctx.arch.is_gfx1100()
            && params.n_exp == 256
            && gfx1100_router_mode.as_deref() == Some("approx"));
    let (shared_smi, shared_down_dtype, shared_down_has_awq) = params
        .shared
        .as_ref()
        .map(|shared| {
            (
                shared.intermediate,
                shared.weights.down.dtype,
                shared.weights.down.awq_scale.is_some(),
            )
        })
        .unwrap_or((0, rdna_compute::DType::F32, false));
    static ROUTER_SHARED_FUSE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_ROUTER_SHARED_FUSE").as_deref() == Ok("1")
    });
    let router_shared_fuse = router_shared_fuse_allowed(
        skip_routing,
        resolution.use_gpu_topk,
        exact_wave64_router,
        params.batch_size,
        params.skip_shared,
        shared_smi,
        shared_down_dtype,
        shared_down_has_awq,
        *ROUTER_SHARED_FUSE,
    );

    static DOWN_LAST_COMBINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let down_last_combine = down_last_combine_allowed(
        params.ep_mode == crate::families::moe::MoeEpMode::None,
        ctx.arch.is_gfx1100(),
        params.batch_size,
        params.k,
        params.expert_dtype_tags.is_some(),
        params.dtypes.routed_down,
        params.mi,
        || {
            *DOWN_LAST_COMBINE.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
            })
        },
    );

    static MOE_NINEPATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_MOE_NINEPATH").unwrap_or_default()
    });
    let ninepath_mode = MOE_NINEPATH.as_str();
    let ninepath_shape_ok = params.ep_mode == crate::families::moe::MoeEpMode::None
        && params.k == 8
        && params.batch_size == 1
        && params.hidden <= 2048
        && params.mi == 512
        && params.expert_dtype_tags.is_none()
        && params.expert_down_awq_ptrs.is_none()
        && !params.defer_routed_combine;
    let ninepath_hfq4 = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256
        && params.dtypes.routed_down == rdna_compute::DType::MQ4G256;
    let ninepath_mq3l = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ2G256Lloyd
        && params.dtypes.routed_down == rdna_compute::DType::MQ3G256Lloyd;
    let ninepath_mq4v2 = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256V2
        && params.dtypes.routed_down == rdna_compute::DType::MQ4G256V2;
    let ninepath_mq6v2 = ninepath_shape_ok
        && params.dtypes.routed_gate_up == rdna_compute::DType::MQ6G256V2
        && params.dtypes.routed_down == rdna_compute::DType::MQ6G256V2;
    let ninepath_eligible = ninepath_hfq4 || ninepath_mq3l || ninepath_mq4v2 || ninepath_mq6v2;
    resolution.check_generic_route(route)?;
    Ok(MoeDecodeSelection {
        resolution,
        router_shared_fuse: router_shared_fuse && !cpu_split,
        exact_wave64_router,
        wave64_router,
        down_last_combine: down_last_combine && !cpu_split,
        route,
        // The split's routed down always writes the expanded F32 slot rows the
        // canonical combine folds: no fused or self-combining down arm exists.
        ninepath_d3: !cpu_split && ninepath_hfq4 && matches!(ninepath_mode, "1" | "d3" | "on"),
        ninepath_d4: !cpu_split
            && ninepath_eligible
            && !matches!(ninepath_mode, "0" | "off" | "d3"),
        ninepath_mq3l: ninepath_mq3l && !cpu_split,
        ninepath_mq4v2: ninepath_mq4v2 && !cpu_split,
        ninepath_mq6v2: ninepath_mq6v2 && !cpu_split,
        cpu_deferred_experts: params.cpu_deferred_experts,
        cpu_split,
    })
}

/// Snapshot of device states that make a host-synchronous CPU split unsound.
/// Its host work cannot be captured, recorded, replayed or shadow-validated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuSplitRunGate {
    pub(super) graph_capture: bool,
    pub(super) verify_capture: bool,
    pub(super) replay_capture: bool,
    pub(super) stream_capture: bool,
    pub(super) recording: bool,
    pub(super) aql: bool,
    pub(super) pm4: bool,
    pub(super) shadow: bool,
}

impl CpuSplitRunGate {
    pub(super) fn observe(gpu: &Gpu) -> Result<Self, DispatchError> {
        let stream_capture = match gpu.active_stream.as_ref() {
            Some(stream) => gpu
                .hip
                .stream_is_capturing(stream)
                .map_err(|e| split_error(format!("stream capture query: {e}")))?,
            None => false,
        };
        Ok(Self {
            graph_capture: gpu.graphs.capture_mode,
            verify_capture: gpu.graphs.verify.capturing.is_some(),
            replay_capture: gpu.graphs.replay.capturing.is_some(),
            stream_capture,
            recording: gpu.replay.is_recording(),
            aql: gpu.replay.should_route_aql(),
            pm4: gpu.replay.should_route_pm4(),
            shadow: gpu.replay.shadow_body_route().is_some()
                || gpu.replay.state() == rdna_compute::replay::ReplayState::ShadowValidated,
        })
    }

    /// The first refused state, named for the operator.
    pub(super) fn refusal(self) -> Option<&'static str> {
        if self.graph_capture || self.verify_capture || self.replay_capture || self.stream_capture {
            Some("graph capture")
        } else if self.recording {
            Some("replay recording")
        } else if self.pm4 {
            Some("a retained PM4 body")
        } else if self.aql {
            Some("a retained AQL body")
        } else if self.shadow {
            Some("a shadow-validated retained body")
        } else {
            None
        }
    }

    pub(super) fn require_eager(self) -> Result<(), DispatchError> {
        match self.refusal() {
            None => Ok(()),
            Some(state) => Err(split_error(format!(
                "is a host-synchronous step and is refused under {state}; run eager (replay.backend=hip, memory.offload_exec=pcie)"
            ))),
        }
    }
}

/// Where one invocation is in the eager CPU split transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitPhase {
    /// Nothing has run; the routed GPU tail must not.
    Idle,
    /// `PrepareSplit` completed: the masked ids are queued on the stream.
    Prepared,
    /// `JoinSplit` completed: the cold slot rows are queued behind the GPU
    /// down, so the canonical combine may run.
    Joined,
}

/// The executor constructor's fixed route width (it refuses any other).
const SPLIT_TOP_K: usize = 10;
const _: () = assert!(crate::cpu_split::CPU_SPLIT_MAX_ROWS == NARROW_VERIFY_MAX_ROWS);

fn split_error(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("sealed moe: CPU tier split: {message}"))
}

/// The executor sealed into a decode call (the same predicate that sealed it).
fn decode_split_executor<'p>(
    params: &MoeParams<'p>,
) -> Result<&'p CpuSplitExecutor, DispatchError> {
    let experts: &'p dyn crate::families::moe::RoutedExpertWeights = params.routed_experts;
    crate::families::moe::cpu_split_executor(experts, params.batch_size)
        .ok_or_else(|| split_error("the executor sealed into this call is no longer attached"))
}

/// The executor sealed into a prefill call (the same predicate that sealed it).
fn prefill_split_executor<'p>(
    params: &MoePrefillParams<'p>,
) -> Result<&'p CpuSplitExecutor, DispatchError> {
    let experts: &'p dyn crate::families::moe::RoutedExpertWeights = params.routed_experts;
    crate::families::moe::cpu_split_executor(experts, params.batch_size)
        .ok_or_else(|| split_error("the executor sealed into this call is no longer attached"))
}

fn split_shape_conflict(
    executor: &CpuSplitExecutor,
    n_exp: usize,
    k_top: usize,
    gate_up_k: usize,
    down_m: usize,
    down_k: usize,
    mi: usize,
) -> Option<&'static str> {
    if k_top != SPLIT_TOP_K
        || n_exp != executor.placement().len()
        || gate_up_k != down_m
        || down_k != mi
        || executor.dim() != down_m
        || executor.mi() != mi
    {
        return Some(
            "the executor plan does not match this layer's expert count, top-k or hidden/intermediate geometry",
        );
    }
    None
}

/// Why a one-row decode call cannot run in the CPU split mode, or `None`.
/// One predicate for sealing, lowering and execution.
///
/// The mode keeps the router, the shared expert and the canonical combine on
/// the original grouped QT44/QT53 route unchanged: only the routed GPU tail
/// reads the executor's masked ids, the cold slots' unweighted F32 rows are
/// installed into `down_expanded`, and the unchanged BF16 combine folds the
/// original route once. Anything that changes that bracket refuses.
pub(super) fn cpu_split_decode_conflict(
    params: &MoeParams<'_>,
    route: Option<MoeRouteCapability>,
    cpu_deferred_experts: bool,
    executor: Option<&CpuSplitExecutor>,
) -> Option<&'static str> {
    if params.batch_size != 1 {
        return Some("indexed decode is one row; windows of two to four rows are the grouped prefill program");
    }
    if cpu_deferred_experts || params.cpu_deferred_experts {
        return Some("CPU tier split and CPU-deferred routed experts are mutually exclusive");
    }
    if !route.is_some_and(|route| route.is_qt44_qt53_grouped() && !route.is_slab()) {
        return Some("requires the original grouped QT44/QT53 route");
    }
    if !matches!(
        params.recipe,
        MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        }
    ) {
        return Some("requires the bf16 shared_after_combine softmax recipe");
    }
    if params.ep_mode != crate::families::moe::MoeEpMode::None
        || params.routed_out.is_some()
        || params.defer_routed_combine
        || params.ep_row_gather.is_some()
    {
        return Some("is a single-device call with the canonical routed combine (no EP, partial, deferred or gathered combine)");
    }
    if params.expert_down_awq_ptrs.is_some()
        || params.routed_gate_up_paro.is_some()
        || params.routed_down_paro.is_some()
    {
        return Some("does not admit AWQ or Paro sidecars");
    }
    if !crate::families::moe::moe_down_writes_expanded(
        params.dtypes.routed_down,
        params.expert_dtype_tags.is_some(),
    ) {
        return Some("requires an expanded-output (non self-combining) down projection");
    }
    executor.and_then(|executor| {
        split_shape_conflict(
            executor,
            params.n_exp,
            params.k,
            params.routed_gate_up_k,
            params.routed_down_m,
            params.routed_down_k,
            params.mi,
        )
    })
}

/// Prefill counterpart of [`cpu_split_decode_conflict`] for a window of one to
/// [`NARROW_VERIFY_MAX_ROWS`] rows. Wider windows never reach it.
pub(super) fn cpu_split_prefill_conflict(
    params: &MoePrefillParams<'_>,
    route: Option<MoeRouteCapability>,
    cpu_deferred_experts: bool,
    executor: Option<&CpuSplitExecutor>,
) -> Option<&'static str> {
    if params.batch_size == 0 || params.batch_size > crate::cpu_split::CPU_SPLIT_MAX_ROWS {
        return Some(
            "the window exceeds the four-row cap (wider windows are ordinary wide prefill)",
        );
    }
    if cpu_deferred_experts || params.cpu_deferred_experts {
        return Some("CPU tier split and CPU-deferred routed experts are mutually exclusive");
    }
    if !route.is_some_and(|route| route.is_qt44_qt53_grouped() && !route.is_slab()) {
        return Some("requires the original grouped QT44/QT53 route");
    }
    if !matches!(
        params.recipe,
        MoeRecipe::SoftmaxGatedShared {
            bf16_round_trip: true,
            shared_after_combine: true,
        }
    ) {
        return Some("requires the bf16 shared_after_combine softmax recipe");
    }
    if !matches!(params.prelude.normalization, MoeNormalization::Provided) {
        return Some("requires a provided natural activation (no in-program norm)");
    }
    if params.ep_row_gather.is_some()
        || params.ep_tile_exchange.is_some()
        || params.routed_out.is_some()
        || !matches!(
            params.prelude.route,
            super::sealed_moe::PrefillRouteMode::Replicated
        )
    {
        return Some(
            "is a replicated single-device mode (no EP gather/exchange/partial/root route)",
        );
    }
    if params.expert_stage_ptrs.is_some()
        || params.expert_down_awq_ptrs.is_some()
        || params.paro_gate_up.is_some()
        || params.paro_down.is_some()
    {
        return Some("does not admit a staged expert table or AWQ/Paro sidecars");
    }
    if !crate::families::moe::moe_down_writes_expanded(
        params.dtypes.routed_down,
        params.expert_dtype_tags.is_some(),
    ) {
        return Some("requires an expanded-output (non self-combining) down projection");
    }
    executor.and_then(|executor| {
        split_shape_conflict(
            executor,
            params.n_exp,
            params.k_top,
            params.gate_up_k,
            params.down_m,
            params.down_k,
            params.mi,
        )
    })
}

/// Eager-only gate of the split: it blocks on the host between the router and
/// the routed tail, so a graph capture (ordinary, specialized verify, replay
/// or raw stream), a replay recording, a retained body, retained PM4/AQL
/// transports and a shadow-validated body all refuse, before any state of this
/// call is mutated.
fn split_run_gate(gpu: &Gpu) -> Result<(), DispatchError> {
    CpuSplitRunGate::observe(gpu)?.require_eager()?;
    if gpu.replay.retained_body_active() {
        return Err(split_error(
            "is a host-synchronous step and is refused under a retained replay body; run eager (replay.backend=hip, memory.offload_exec=pcie)",
        ));
    }
    Ok(())
}

/// The expanded slot scratch must hold `rows * k * dim` F32 values.
fn split_expanded_extent(
    expanded: &GpuTensor,
    rows: usize,
    k_top: usize,
    dim: usize,
) -> Result<(), DispatchError> {
    let need = rows
        .checked_mul(k_top)
        .and_then(|slots| slots.checked_mul(dim))
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| split_error("expanded slot extent overflows"))?;
    if expanded.buf.size() < need {
        return Err(split_error(format!(
            "down_expanded holds {} bytes, the slot rows need {need}",
            expanded.buf.size()
        )));
    }
    Ok(())
}

/// Everything the split needs from the device, checked before the first
/// launch (and again by `PrepareSplit`): eager state, the route width the
/// router actually produces, and that the routed tail leaves F32 slot rows.
pub(super) fn validate_split_decode(
    gpu: &Gpu,
    params: &MoeParams<'_>,
    selection: &MoeDecodeSelection,
) -> Result<(), DispatchError> {
    split_run_gate(gpu)?;
    let executor = decode_split_executor(params)?;
    if let Some(reason) = cpu_split_decode_conflict(
        params,
        selection.route,
        selection.cpu_deferred_experts,
        Some(executor),
    ) {
        return Err(split_error(reason));
    }
    // The ordinary decode route produces exactly `rows` rows; a canonicalised
    // (prefill-invariant) route width is a different route contract.
    if gpu.route_rows(params.batch_size) != params.batch_size {
        return Err(split_error(
            "requires route rows equal to the window rows (prefill-invariant scope is refused)",
        ));
    }
    split_expanded_extent(
        params.down_expanded,
        params.batch_size,
        params.k,
        params.routed_down_m,
    )
}

/// Prefill counterpart of [`validate_split_decode`]. The routed down must be
/// the decode-kernel F32 expanded rows in slot order `[row * 10 + rank, dim]`
/// ([`super::qt44_qt53_prefill::writes_slot_ordered_rows`]), never grouped BF16
/// rows or a row-fold target; this device/width cannot satisfy that on a
/// symmetric-IU4 or WMMA grouped down, and then the call refuses rather than
/// silently changing arithmetic.
pub(super) fn validate_split_prefill(
    gpu: &Gpu,
    params: &MoePrefillParams<'_>,
    selection: &MoePrefillSelection,
) -> Result<(), DispatchError> {
    split_run_gate(gpu)?;
    let executor = prefill_split_executor(params)?;
    if let Some(reason) = cpu_split_prefill_conflict(
        params,
        selection.route,
        selection.cpu_deferred_experts,
        Some(executor),
    ) {
        return Err(split_error(reason));
    }
    if gpu.route_rows(params.batch_size) != params.batch_size {
        return Err(split_error(
            "requires route rows equal to the window rows (prefill-invariant scope is refused)",
        ));
    }
    if !super::qt44_qt53_prefill::writes_slot_ordered_rows(
        gpu,
        params,
        selection.uses_grouped_rows(),
    ) {
        return Err(split_error(
            "this device and width take a grouped BF16 down, not the F32 slot-ordered expanded rows the canonical combine and the cold-slot install require",
        ));
    }
    split_expanded_extent(
        params.down_expanded,
        params.batch_size,
        params.k_top,
        params.down_m,
    )
}

/// A grouped FP8 launcher reads one immutable contiguous block table, not the
/// legacy device pointer indirection. Prove that both tables name those same
/// live blocks before constructing the launcher's borrowed base view.
fn validate_fp8_expert_blocks(params: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    let (first_gate, first_down) = params
        .routed_experts
        .get(0)
        .ok_or_else(|| DispatchError::Hip("sealed fp8 moe: empty expert block table".into()))?;
    let gate_stride = rdna_compute::fp8_e4m3::b128_block_stride(2 * params.mi, params.gate_up_k)
        .ok_or_else(|| DispatchError::Hip("sealed fp8 moe: gate stride overflows".into()))?;
    let down_stride = rdna_compute::fp8_e4m3::b128_block_stride(params.down_m, params.down_k)
        .ok_or_else(|| DispatchError::Hip("sealed fp8 moe: down stride overflows".into()))?;
    let gate_base = first_gate.buf.buf.as_ptr() as usize;
    let down_base = first_down.buf.buf.as_ptr() as usize;
    for expert in 0..params.n_exp {
        let (gate, down) = params.routed_experts.get(expert).ok_or_else(|| {
            DispatchError::Hip("sealed fp8 moe: incomplete expert block table".into())
        })?;
        let address = |base: usize, stride: usize| {
            expert
                .checked_mul(stride)
                .and_then(|offset| base.checked_add(offset))
        };
        if Some(gate.buf.buf.as_ptr() as usize) != address(gate_base, gate_stride)
            || Some(down.buf.buf.as_ptr() as usize) != address(down_base, down_stride)
            || gate_base % 256 != 0
            || down_base % 256 != 0
            || gate.m != 2 * params.mi
            || gate.k != params.gate_up_k
            || down.m != params.down_m
            || down.k != params.down_k
            || gate.dtype != rdna_compute::DType::FP8E4M3B128
            || down.dtype != rdna_compute::DType::FP8E4M3B128
            || gate.buf.buf.size() < gate_stride
            || down.buf.buf.size() < down_stride
            || gate.rotation.is_some()
            || down.rotation.is_some()
            || gate.awq_scale.is_some()
            || down.awq_scale.is_some()
        {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "fp8-immutable-contiguous-expert-blocks",
                arch: "gfx942",
                quant: "FP8E4M3B128",
            });
        }
    }
    Ok(())
}

/// Resolve all grouped-prefill choices without a GPU or an instruction list.
///
/// The prefill [`MoePrefillResolution`] reads owner-derived residency via
/// `routed_host_mapped_projections` (tier identity first, then
/// `routed_experts.host_mapped_projections()`). A narrow MTP verify window over
/// host-mapped experts takes the indexed Path 1, except a host-mapped uniform
/// MQ4 gate_up+down on gfx1201, which is re-admitted to Path 2 (grouped-WMMA).
/// Residency comes from the packed weight owners or an explicit tier identity
/// (its layer-wide `any_host`, never expert 0) — never inferred from
/// borrowed tensor kinds — and resident experts keep the legacy route.
pub(super) fn select_prefill(
    ctx: &DispatchCtx,
    params: &MoePrefillParams<'_>,
) -> Result<MoePrefillSelection, DispatchError> {
    let fp8_route = super::select_fp8_grouped_route(
        ctx,
        &params.dtypes,
        params.route_policy,
        params.n_exp,
        params.k_top,
        params.gate_up_k,
        params.mi,
        params.prelude.shared.as_ref().map_or(0, |s| s.intermediate),
        params.gate_up_k,
        params.down_m,
        params.down_k,
        params.expert_dtype_tags,
        params.expert_down_awq_ptrs.is_some()
            || params.down_awq_scale.is_some()
            || params.paro_gate_up.is_some()
            || params.paro_down.is_some()
            || params
                .prelude
                .shared
                .as_ref()
                .is_some_and(|s| projection_has_sidecars(&params.prelude.router, &s.weights)),
    )?;
    if let Some(route) = fp8_route {
        if params.recipe
            != (MoeRecipe::SoftmaxGatedShared {
                bf16_round_trip: true,
                shared_after_combine: true,
            })
            || params.cpu_deferred_experts
            || crate::families::moe::cpu_split_executor(params.routed_experts, params.batch_size)
                .is_some()
            || params.routed_out.is_some()
            || params.expert_stage_ptrs.is_some()
            || params.ep_row_gather.is_some()
            || params.ep_tile_exchange.is_some()
            || !matches!(
                params.prelude.route,
                super::sealed_moe::PrefillRouteMode::Replicated
            )
            || crate::families::moe::routed_host_mapped(params.routed_experts)
        {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "fp8-sealed-replicated-bf16-only",
                arch: "gfx942",
                quant: "FP8E4M3B128",
            });
        }
        validate_fp8_expert_blocks(params)?;
        return Ok(MoePrefillSelection {
            resolution: MoePrefillResolution::sealed_fp8(),
            path2_m_total: params.m_total_max,
            force_mq4_grouped_fp16: false,
            coalesced_verify: false,
            route: Some(route),
            cpu_deferred_experts: false,
            cpu_split: false,
        });
    }
    let route = (matches!(params.recipe, MoeRecipe::SoftmaxGatedShared { .. })
        && params.routed_out.is_none()
        && params.paro_gate_up.is_none()
        && params.paro_down.is_none()
        && matches!(
            params.prelude.route,
            super::sealed_moe::PrefillRouteMode::Replicated
        ))
    .then(|| {
        super::select_grouped_route(
            &params.dtypes,
            params.n_exp,
            params.k_top,
            params.gate_up_k,
            params.mi,
            params
                .prelude
                .shared
                .as_ref()
                .map_or(0, |shared| shared.intermediate),
            params.gate_up_k,
            params.down_m,
            params.down_k,
            params.expert_dtype_tags,
            params.expert_down_awq_ptrs.is_some()
                || params.down_awq_scale.is_some()
                || params.prelude.shared.as_ref().is_some_and(|shared| {
                    projection_has_sidecars(&params.prelude.router, &shared.weights)
                }),
            params.route_policy,
            ctx.arch.arch(),
        )
    })
    .flatten();
    if params.route_policy.is_some() && route.is_none() {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "specialized-prefill-route-unsupported",
            arch: "",
            quant: "QT44/QT53",
        });
    }
    if route.is_none() {
        super::reject_mq4g128v2_moe(&params.dtypes)?;
    }
    let (host_gate_up, host_down) =
        crate::families::moe::routed_host_mapped_projections(params.routed_experts);
    let resolution = MoePrefillResolution::resolve_with_batch_ctx(
        &params.dtypes,
        &ctx.arch,
        &ctx.flags,
        ctx.batch_size,
        ctx.workload == crate::context::DispatchWorkload::SpeculativeVerify,
        host_gate_up || host_down,
    );
    resolution.check_generic_route(route)?;
    if route.is_some_and(MoeRouteCapability::is_slab) {
        // The slab route is a replicated, single-device lowering (resident or staged experts)
        // lowering whose K2 arm is the grouped protocol: nothing else may
        // ride on it, and a device without grouped GEMM refuses.
        if !resolution.use_path2
            || params.cpu_deferred_experts
            || params.routed_out.is_some()
            || params.ep_row_gather.is_some()
            || params.ep_tile_exchange.is_some()
            || !matches!(
                params.prelude.route,
                super::sealed_moe::PrefillRouteMode::Replicated
            )
        {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "slab-sealed-replicated-grouped-only",
                arch: if ctx.arch.is_gfx1100() {
                    "gfx1100"
                } else if ctx.arch.is_gfx1151() {
                    "gfx1151"
                } else {
                    "gfx1201"
                },
                quant: "MQ4/W3/E8P-I4 slab",
            });
        }
    }
    // The distinct CPU-cold/GPU-hot split: a window of at most four rows over
    // an attached executor. Wider windows are ordinary wide prefill. Every
    // admission refusal is raised here, before the call is published.
    let cpu_split =
        match crate::families::moe::cpu_split_executor(params.routed_experts, params.batch_size) {
            Some(executor) => {
                if let Some(reason) = cpu_split_prefill_conflict(
                    params,
                    route,
                    params.cpu_deferred_experts,
                    Some(executor),
                ) {
                    return Err(DispatchError::Hip(format!(
                        "sealed moe: CPU tier split: {reason}"
                    )));
                }
                true
            }
            None => false,
        };
    let _total_slots = params
        .batch_size
        .checked_mul(params.k_top)
        .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
    let path2_m_total = if resolution.use_path2 {
        params.m_total_max
    } else {
        0
    };
    let force_mq4_grouped_fp16 = resolution.force_mq4_grouped_fp16 || params.force_mq4_grouped_fp16;
    // Lower bound of 2 is load-bearing: a 1-row window is decode, not a verify
    // window, so it must not take the coalesced/Path-2 re-admit.
    let coalesced_verify = ctx.arch.is_gfx1201()
        && ctx.workload == crate::context::DispatchWorkload::SpeculativeVerify
        && (2..=NARROW_VERIFY_MAX_ROWS).contains(&ctx.batch_size)
        && (2..=NARROW_VERIFY_MAX_ROWS).contains(&params.batch_size)
        && route.is_none()
        && resolution.use_path2;
    let compact_ep = matches!(
        params.prelude.route,
        super::sealed_moe::PrefillRouteMode::ProduceRoot { .. }
            | super::sealed_moe::PrefillRouteMode::AdoptRoot { .. }
    );
    if compact_ep && resolution.down_path0 {
        return Err(DispatchError::Hip(
            "root-routed EP prefill requires an expanded-output down projection".into(),
        ));
    }
    if params.cpu_deferred_experts {
        // A CPU-deferred narrow window keeps the GPU router/shared stages and
        // the arch's CPU splice; it must not be an EP partial (the routed
        // contribution would be folded twice) and must not rely on the grouped
        // routed tail it is about to omit.
        if compact_ep || params.routed_out.is_some() {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts require a replicated (non-EP) prefill".into(),
            ));
        }
        // A BF16 recipe is admitted only on the grouped QT44/QT53 route: the
        // router and shared stages this program still lowers carry its BF16
        // round trips themselves, and the omitted routed stages' round trips
        // (gate/up, activation, weighted combine) are the CPU splice's, which
        // receives the recipe flag. The generic route also rounds the rotated
        // input, which the CPU does not emulate.
        if params.recipe.bf16_round_trip()
            && !route.is_some_and(MoeRouteCapability::is_qt44_qt53_grouped)
        {
            return Err(DispatchError::Hip(
                "CPU-deferred routed experts emulate a recipe's bf16 round trips only on the grouped QT44/QT53 route"
                    .into(),
            ));
        }
    }
    Ok(MoePrefillSelection {
        resolution,
        path2_m_total,
        force_mq4_grouped_fp16,
        coalesced_verify,
        route,
        cpu_deferred_experts: params.cpu_deferred_experts,
        cpu_split,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_topk_disables_router_shared_fuse() {
        assert!(!router_shared_fuse_allowed(
            false,
            false,
            true,
            1,
            false,
            512,
            rdna_compute::DType::MQ4G256,
            false,
            true,
        ));
    }

    fn last_combine(
        ep_none: bool,
        is_gfx1100: bool,
        batch_size: usize,
        topk: usize,
        has_tags: bool,
        down: rdna_compute::DType,
        down_k: usize,
    ) -> bool {
        down_last_combine_allowed(
            ep_none,
            is_gfx1100,
            batch_size,
            topk,
            has_tags,
            down,
            down_k,
            || true,
        )
    }

    #[test]
    fn down_last_combine_admits_only_k512() {
        let mq4 = rdna_compute::DType::MQ4G256;
        assert!(last_combine(true, true, 1, 8, false, mq4, 512));
        for k in [1792, 1536, 256, 0, 1024] {
            assert!(!last_combine(true, true, 1, 8, false, mq4, k), "K={k}");
        }
    }

    #[test]
    fn down_last_combine_keeps_other_predicate_restrictions() {
        let mq4 = rdna_compute::DType::MQ4G256;
        assert!(!last_combine(false, true, 1, 8, false, mq4, 512));
        assert!(!last_combine(true, false, 1, 8, false, mq4, 512));
        assert!(!last_combine(true, true, 2, 8, false, mq4, 512));
        assert!(!last_combine(true, true, 1, 4, false, mq4, 512));
        assert!(!last_combine(true, true, 1, 8, true, mq4, 512));
        assert!(!last_combine(
            true,
            true,
            1,
            8,
            false,
            rdna_compute::DType::F32,
            512
        ));
    }

    #[test]
    fn down_last_combine_env_is_last_and_short_circuited() {
        let mq4 = rdna_compute::DType::MQ4G256;
        let calls = std::cell::Cell::new(0);
        let probe = |ep_none: bool, down_k: usize, env: bool| {
            down_last_combine_allowed(ep_none, true, 1, 8, false, mq4, down_k, || {
                calls.set(calls.get() + 1);
                env
            })
        };
        assert!(!probe(true, 1792, true));
        assert!(!probe(false, 512, true));
        assert_eq!(calls.get(), 0);
        assert!(!probe(true, 512, false));
        assert_eq!(calls.get(), 1);
        assert!(probe(true, 512, true));
        assert_eq!(calls.get(), 2);
    }
}
