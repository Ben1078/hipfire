// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq

//! Borrowed Qwen4 band bindings for the generic, Gpus-owned executor.

use crate::bundle::Qwen4Bundle;
use crate::gpu_forward::{Qwen4GpuForward, Qwen4MeshHead, Qwen4MeshLayout};
use crate::program::QWEN4_STREAMS_RESOURCE;
use crate::weights::Qwen4BandScope;
use crate::Qwen4Config;
use hipfire_dispatch::pipeline::step_graph::OpId;
use hipfire_runtime::mesh_split::{CollectiveKind, PlannedCollective};
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
};
use hipfire_runtime::multi_gpu::transport::EdgeDescriptor;
use rdna_compute::Gpu;

type Result<T, E = String> = std::result::Result<T, E>;
const TRANSACTIONS_REFUSED: &str =
    "Qwen4 topology.pp does not support checkpoint, repair or session transactions yet";

/// One rank's existing owners, borrowed for submission; never owns a GPU.
pub struct Qwen4RankBand<'a> {
    pub bundle: &'a mut Qwen4Bundle,
    pub forward: &'a mut Qwen4GpuForward,
}

/// An epoch adapter. The output band may be the published root bundle; peers
/// remain in that bundle's mesh owner. Nothing is replicated or moved here.
pub struct Qwen4MeshBindings<'a> {
    ranks: Vec<Qwen4RankBand<'a>>,
    tokens: &'a [u32],
    edge_epochs: Vec<u64>,
    head: Option<Qwen4MeshHead<'a>>,
    prepared: Option<ForwardEpoch>,
    ran: [bool; 8],
    layouts: [Option<Qwen4MeshLayout>; 8],
}

fn streams_bytes(rows: usize, branches: usize, hidden: usize, dtype_bytes: usize) -> Result<usize> {
    if rows == 0 || branches == 0 || hidden == 0 || !matches!(dtype_bytes, 2 | 4) {
        return Err("invalid Qwen4 streams geometry or dtype".into());
    }
    rows.checked_mul(branches)
        .and_then(|n| n.checked_mul(hidden))
        .and_then(|n| n.checked_mul(dtype_bytes))
        .ok_or_else(|| "Qwen4 streams payload overflows".into())
}

fn validate_scopes<'s>(
    config: &Qwen4Config,
    scopes: impl IntoIterator<Item = &'s Qwen4BandScope>,
) -> Result<()> {
    let mut end = 0;
    let mut count = 0;
    for scope in scopes {
        count += 1;
        scope.validate(config)?;
        if scope.band.start != end {
            return Err("Qwen4 rank bands must tile the trunk once in layer order".into());
        }
        end = scope.band.end;
    }
    if !(2..=8).contains(&count) {
        return Err("Qwen4 topology.pp requires 2..=8 bands".into());
    }
    if end != config.num_hidden_layers {
        return Err("Qwen4 rank bands do not cover the trunk".into());
    }
    Ok(())
}

impl<'a> Qwen4MeshBindings<'a> {
    /// Validate ownership before any submission or state mutation.
    pub fn new(ranks: Vec<Qwen4RankBand<'a>>) -> Result<Self> {
        let config = &ranks
            .first()
            .ok_or("Qwen4 mesh has no bands")?
            .bundle
            .config;
        validate_scopes(
            config,
            ranks.iter().map(|rank| rank.bundle.weights.band_scope()),
        )?;
        for rank in &ranks {
            let scope = rank.bundle.weights.band_scope();
            if &rank.bundle.config != config {
                return Err("Qwen4 mesh bands have different model configurations".into());
            }
            let map = rank.bundle.state.scope_map();
            if map.band() != scope.band
                || rank.forward.band_scope() != scope
                || map.owns_input() != scope.input
                || map.owns_output() != scope.output
                || map.gdn_count() != rank.bundle.state.gdn.len()
                || map.qsa_count() != rank.bundle.state.qsa.len()
                || rank.forward.scratch.streams.dtype != ranks[0].forward.scratch.streams.dtype
            {
                return Err("Qwen4 mesh weight and state ownership disagree".into());
            }
            if rank.bundle.ep.is_some() || rank.bundle.lanes.is_some() || rank.bundle.mtp.is_some()
            {
                return Err("Qwen4 topology.pp does not admit EP, lanes or native MTP".into());
            }
        }
        Ok(Self {
            ranks,
            tokens: &[],
            edge_epochs: Vec::new(),
            head: None,
            prepared: None,
            ran: [false; 8],
            layouts: [None; 8],
        })
    }

    /// Host token storage remains immutable until submission is drained.
    pub fn set_tokens(&mut self, tokens: &'a [u32]) {
        self.tokens = tokens;
        self.prepared = None;
    }

    /// Transport epochs in directed logical band order (0->1, 1->2, ...).
    pub fn set_edge_epochs(&mut self, epochs: Vec<u64>) {
        self.edge_epochs = epochs;
        self.prepared = None;
    }

    /// Prepare the route-specific precision and exact live boundary extent.
    pub fn set_rank_layout(&mut self, owner: usize, layout: Qwen4MeshLayout) -> Result<()> {
        if owner >= self.ranks.len() {
            return Err("unknown Qwen4 mesh rank".into());
        }
        self.layouts[owner] = Some(layout);
        self.prepared = None;
        Ok(())
    }

    /// `None` commits trunk state without computing a head row.
    pub fn set_head(&mut self, head: Option<Qwen4MeshHead<'a>>) {
        self.head = head;
        self.prepared = None;
    }

    pub fn ranks(&self) -> &[Qwen4RankBand<'a>] {
        &self.ranks
    }

    pub fn ranks_mut(&mut self) -> &mut [Qwen4RankBand<'a>] {
        self.prepared = None;
        &mut self.ranks
    }

    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Qwen4 mesh epoch has not been prepared".into());
        }
        Ok(())
    }
}

impl MeshStepBindings for Qwen4MeshBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        if epoch.route == MeshRoute::Verify {
            return Err("Qwen4 topology.pp does not support verify windows yet".into());
        }
        if epoch.actual_rows == 0
            || epoch.route_rows < epoch.actual_rows
            || self.tokens.len() != epoch.actual_rows
            || (epoch.route == MeshRoute::Decode && epoch.actual_rows != 1)
            || self.edge_epochs.len() + 1 != self.ranks.len()
            || self.edge_epochs.contains(&0)
        {
            return Err("invalid Qwen4 mesh rows, tokens or transport epochs".into());
        }
        let end = epoch
            .start_position
            .checked_add(epoch.actual_rows)
            .ok_or("Qwen4 mesh position overflows")?;
        for (owner, rank) in self.ranks.iter().enumerate() {
            let layout = self.layouts[owner].ok_or("Qwen4 mesh rank layout is not prepared")?;
            let dtype_bytes = if layout.bf16_streams { 2 } else { 4 };
            if layout.route_rows != epoch.route_rows
                || layout.stream_bytes
                    != streams_bytes(
                        epoch.actual_rows,
                        rank.bundle.config.hc_count,
                        rank.bundle.config.hidden_size,
                        dtype_bytes,
                    )?
                || self.layouts[0].is_some_and(|first| first.bf16_streams != layout.bf16_streams)
            {
                return Err(
                    "Qwen4 mesh rank route or live streams extent disagrees with epoch".into(),
                );
            }
            if rank.bundle.state.position != epoch.start_position
                || end > rank.bundle.state.max_seq_len
                || epoch.actual_rows > rank.forward.scratch.max_chunk
            {
                return Err("Qwen4 mesh rank position or capacity disagrees with epoch".into());
            }
        }
        let config = &self.ranks[0].bundle.config;
        if self
            .tokens
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
        {
            return Err("Qwen4 mesh token is outside the embedding vocabulary".into());
        }
        if let Some(head) = &self.head {
            let rows = head.rows.count(epoch.actual_rows);
            let elements = rows
                .checked_mul(config.vocab_size)
                .ok_or("Qwen4 head shape overflows")?;
            if head.logits.dtype != rdna_compute::DType::F32 || head.logits.numel() != elements {
                return Err("Qwen4 mesh logits output has invalid dtype or capacity".into());
            }
            let top1_bytes = rows
                .checked_mul(std::mem::size_of::<i32>())
                .ok_or("Qwen4 argmax shape overflows")?;
            if head.top1.is_some_and(|top1| {
                top1.dtype != rdna_compute::DType::Raw || top1.buf.size() < top1_bytes
            }) {
                return Err("Qwen4 mesh argmax output has invalid dtype or capacity".into());
            }
        }
        self.ran.fill(false);
        self.prepared = Some(*epoch);
        Ok(())
    }

    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        ops: &[OpId],
    ) -> Result<()> {
        self.require_prepared(epoch)?;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        let expected = rank.forward.mesh_op_range(
            self.head.is_some(),
            self.head.as_ref().is_some_and(|head| head.top1.is_some()),
        );
        if self.ran[owner] || !ops.iter().map(|id| id.0).eq(expected) {
            return Err(
                "Qwen4 mesh band must run exactly once with its full declared op range".into(),
            );
        }
        let head = if rank.bundle.weights.band_scope().output {
            self.head.as_ref().map(|head| Qwen4MeshHead {
                logits: head.logits,
                top1: head.top1,
                rows: head.rows,
            })
        } else {
            None
        };
        rank.forward
            .forward_mesh_chunk(
                rank.bundle,
                gpu,
                self.tokens,
                epoch.route_rows,
                epoch.route,
                head,
            )
            .map_err(|error| error.to_string())?;
        self.ran[owner] = true;
        Ok(())
    }

    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>> {
        self.require_prepared(epoch)?;
        let CollectiveKind::BandXfer { src, dst } = collective.kind else {
            return Err("Qwen4 topology.pp only admits HC band transfers".into());
        };
        if src.checked_add(1) != Some(dst)
            || dst >= self.ranks.len()
            || (owner != src && owner != dst)
            || collective.operand != QWEN4_STREAMS_RESOURCE
        {
            return Err("Qwen4 HC boundary has invalid endpoints".into());
        }
        let rank = self.ranks.get(owner).ok_or("unknown Qwen4 mesh rank")?;
        let streams = &rank.forward.scratch.streams;
        let layout = self.layouts[owner].ok_or("Qwen4 mesh rank layout is not prepared")?;
        let dtype_bytes = if layout.bf16_streams { 2 } else { 4 };
        let bytes = layout.stream_bytes;
        Ok(BoundaryIo {
            view: streams,
            descriptor: EdgeDescriptor {
                epoch: *self
                    .edge_epochs
                    .get(src)
                    .ok_or("missing Qwen4 edge epoch")?,
                op: epoch.op_id,
                rows: epoch.actual_rows,
                dtype_bytes,
                bytes,
            },
        })
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        let rank = self.ranks.get(owner).ok_or("unknown Qwen4 mesh rank")?;
        if !self.ran[owner]
            || rank.bundle.state.position != epoch.start_position + epoch.actual_rows
            || rank
                .bundle
                .state
                .qsa
                .iter()
                .any(|qsa| qsa.position != rank.bundle.state.position)
        {
            return Err("Qwen4 band did not apply its local position and QSA marks".into());
        }
        Ok(())
    }

    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 mesh rank")?;
        rank.forward.set_ple_lookahead(&[]);
        drop(rank.forward.ple_ahead.take());
        rank.bundle.reset(gpu).map_err(|error| error.to_string())
    }

    fn repair_prefix_rank(
        &mut self,
        _: usize,
        _: &mut Gpu,
        _: &ForwardEpoch,
        _: usize,
        _: &[u32],
    ) -> Result<()> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn capture_session_rank(
        &mut self,
        _: usize,
        _: &mut Gpu,
        _: &ForwardEpoch,
    ) -> Result<MeshCheckpointId> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn probe_session_rank(
        &mut self,
        _: usize,
        _: &ForwardEpoch,
        _: MeshCheckpointId,
    ) -> Result<()> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn restore_session_rank(
        &mut self,
        _: usize,
        _: &mut Gpu,
        _: &ForwardEpoch,
        _: MeshCheckpointId,
    ) -> Result<()> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn checkpoint_rank(
        &mut self,
        _: usize,
        _: &mut Gpu,
        _: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn restore_rank(
        &mut self,
        _: usize,
        _: &mut Gpu,
        _: &MeshCheckpointKey,
        _: MeshCheckpointId,
    ) -> Result<()> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn validate_finalize_rank(
        &mut self,
        _: usize,
        _: &MeshCheckpointKey,
        _: MeshCheckpointId,
    ) -> Result<()> {
        Err(TRANSACTIONS_REFUSED.into())
    }
    fn finalize_rank(&mut self, _: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {
        panic!("{TRANSACTIONS_REFUSED}: finalize called without a valid checkpoint");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scopes(counts: &[usize]) -> Vec<Qwen4BandScope> {
        let total: usize = counts.iter().sum();
        let mut start = 0;
        counts
            .iter()
            .map(|&count| {
                let band = start..start + count;
                start += count;
                Qwen4BandScope {
                    input: band.start == 0,
                    output: band.end == total,
                    band,
                }
            })
            .collect()
    }

    #[test]
    fn streams_payload_is_exact_live_extent() {
        assert_eq!(streams_bytes(1, 4, 2560, 4).unwrap(), 40_960);
        assert_eq!(streams_bytes(1536, 4, 2560, 4).unwrap(), 62_914_560);
        assert_eq!(streams_bytes(4096, 4, 2560, 4).unwrap(), 167_772_160);
        assert_eq!(streams_bytes(1536, 4, 2560, 2).unwrap(), 31_457_280);
        assert!(streams_bytes(usize::MAX, 4, 2560, 4).is_err());
        assert!(streams_bytes(0, 4, 2560, 4).is_err());
        assert!(streams_bytes(1, 4, 2560, 1).is_err());
    }

    #[test]
    fn mesh_scope_tiles_and_pins() {
        let config = crate::config::compact_test_config();
        for counts in [&[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            assert!(validate_scopes(&config, &scopes(counts)).is_ok());
        }
        let mut bands = scopes(&[24, 24]);
        bands[1].band.start -= 1;
        assert!(validate_scopes(&config, &bands).is_err());
        let mut bands = scopes(&[24, 24]);
        bands[0].input = false;
        assert!(validate_scopes(&config, &bands).is_err());
        let mut bands = scopes(&[24, 24]);
        bands[1].output = false;
        assert!(validate_scopes(&config, &bands).is_err());
        let bands = scopes(&[1, 47]);
        assert!(validate_scopes(&config, &bands).is_err());
        assert!(validate_scopes(&config, &scopes(&[24, 23])).is_err());
        assert!(validate_scopes(&config, &scopes(&[48])).is_err());
    }
}
