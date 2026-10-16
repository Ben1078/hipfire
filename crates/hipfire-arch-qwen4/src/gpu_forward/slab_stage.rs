// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Long-prefill expert staging for sealed expert-slab layers.
//!
//! The legacy stage (`Qwen4ExpertStage`) borrows the two last resident trunk
//! layers as donors and copies a host layer's aggregate gate/up and down
//! tensors over them. A slab layer is one arena whose per-expert extents
//! (MQ4 or W3, by codec) differ from layer to layer, so a donor layer cannot
//! hold an arbitrary cold arena. RouteOwner3 therefore refused staging for
//! slab bundles, and every cold layer's grouped K2 GEMMs read host memory in
//! place over PCIe (measured: cold-layer expert kernels 4.8-6.3x the staged
//! legacy ones, rcpt/speed/prefill.attrib.json).
//!
//! Slab staging uses two dedicated VRAM stage buffers (one per layer parity,
//! each the largest cold arena plus its zeroed tail pad) instead of donors:
//! nothing is overwritten, so there is nothing to restore. Each cold layer's
//! pointer tables name the layer's own arena offsets rebased onto its parity's
//! stage buffer; the layer's codec tag tensor (VRAM, per layer) is untouched.
//! The copy moves the arena plus its tail pad, so the kernels read the same
//! bytes (including the W3 down over-read) they would read in place.
//! Forwards below `min_rows` rows, decode and MTP verify keep reading in place.
//!
//! Per-expert tier plans (`expert_tier_*`, aggregate MQ4 Flash-Next, one GPU)
//! reuse the same two dedicated stage buffers but stage per expert: a layer
//! with any host-tier expert copies exactly its HostMapped experts' gate/up
//! and down projections (runs of adjacent host bytes coalesced) into its
//! parity's stage buffer, followed by a zeroed tail pad, and its two 512-entry
//! pointer tables rebase only those host entries. Device-tier entries keep the
//! owner's absolute address: the model-level owners are never a copy
//! destination, so nothing is overwritten and nothing is restored. Tables stay
//! indexed by the original local expert id, so the kernels see the same ids,
//! slot order and combine order as in place; the codec table is untouched.
//! Host layers need not be a suffix: each parity's next host layer is found by
//! scanning `layer + 2, layer + 4, ...`.

use super::*;
use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_tier_owners::ExpertTierLease;
use crate::expert_tier_plan::Placement;

/// Host sources of the staged layers.
pub(super) struct SlabStageMeta {
    /// Per host layer from `Qwen4ExpertStage::first_host`: the host address of
    /// the layer's arena and the bytes to copy (arena plus tail pad).
    sources: Vec<(usize, usize)>,
    /// `Some` for a per-expert tier stage (`sources` is then empty).
    tier: Option<TierStageMeta>,
}

/// One host-to-stage copy of a tier layer.
struct StageCopy {
    /// CPU address of the pinned host owner's bytes.
    src: usize,
    /// Byte offset inside the parity's stage buffer.
    dst: usize,
    len: usize,
}

/// Staged copies of the layers with a host-tier expert.
struct TierStageMeta {
    /// Per trunk layer: its slot in `copies`/`used` and in the stage's
    /// `tables` (two per slot: gate/up, down), `None` for a layer with no
    /// host-tier expert (never staged, reads owner VRAM).
    slots: Vec<Option<usize>>,
    copies: Vec<Vec<StageCopy>>,
    /// Bytes of each slot's staged payload (the zeroed tail pad follows).
    used: Vec<usize>,
    /// Keeps both model-level owners alive while the stage exists.
    _keepalive: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

/// One expert of a tier layer, as the plan and view give it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TierStageExpert {
    pub(super) host: bool,
    /// Absolute device-visible addresses and lengths of the projections.
    pub(super) gate_up: usize,
    pub(super) gate_up_len: usize,
    pub(super) down: usize,
    pub(super) down_len: usize,
}

/// Where a tier layer's host experts sit in its stage buffer.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct TierStageLayout {
    /// `(source address, stage offset, bytes)`, each a run of adjacent or
    /// overlapping host projections copied once.
    pub(super) copies: Vec<(usize, usize, usize)>,
    /// Stage offset of each expert's gate/up and down, `None` for a device
    /// expert (its table entry stays the owner address).
    pub(super) gate_up: Vec<Option<usize>>,
    pub(super) down: Vec<Option<usize>>,
    /// End of the staged payload; the zeroed tail pad starts here.
    pub(super) used: usize,
}

const STAGE_ALIGN: usize = 256;

/// Lay out the host experts of one tier layer: every host projection must lie
/// inside `[host_base, host_base + host_bytes)`. Projections are sorted by
/// source address and merged where they touch, so a run keeps its relative
/// layout; runs are packed back to back at 256-byte alignment.
pub(super) fn tier_stage_layout(
    experts: &[TierStageExpert],
    host_base: usize,
    host_bytes: usize,
) -> Result<TierStageLayout, String> {
    let host_end = host_base
        .checked_add(host_bytes)
        .ok_or_else(|| "host owner range overflows".to_string())?;
    // (source, len, expert, is_down)
    let mut items: Vec<(usize, usize, usize, bool)> = Vec::new();
    for (eid, expert) in experts.iter().enumerate() {
        if !expert.host {
            continue;
        }
        for (src, len, down) in [
            (expert.gate_up, expert.gate_up_len, false),
            (expert.down, expert.down_len, true),
        ] {
            let end = src
                .checked_add(len)
                .ok_or_else(|| format!("expert {eid} projection overflows"))?;
            if len == 0 || src < host_base || end > host_end {
                return Err(format!(
                    "host expert {eid} extent [{src:#x}, +{len}) is outside the host owner [{host_base:#x}, +{host_bytes})"
                ));
            }
            if src % STAGE_ALIGN != 0 {
                return Err(format!("host expert {eid} source {src:#x} is not {STAGE_ALIGN}-byte aligned"));
            }
            items.push((src, len, eid, down));
        }
    }
    items.sort_unstable();
    // Runs: (source start, source end, stage offset).
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut item_run = Vec::with_capacity(items.len());
    let mut cursor = 0usize;
    for &(src, len, _, _) in &items {
        let end = src + len;
        match runs.last_mut() {
            Some(run) if src <= run.1 => {
                if end > run.1 {
                    run.1 = end;
                }
            }
            _ => {
                let dst = cursor
                    .checked_add(STAGE_ALIGN - 1)
                    .ok_or_else(|| "stage layout overflows".to_string())?
                    & !(STAGE_ALIGN - 1);
                runs.push((src, end, dst));
                // Provisional end; fixed below once the run is complete.
                cursor = dst;
            }
        }
        item_run.push(runs.len() - 1);
        let run = runs.last().expect("run just pushed or extended");
        cursor = run.2 + (run.1 - run.0);
    }
    let mut layout = TierStageLayout {
        copies: runs.iter().map(|&(src, end, dst)| (src, dst, end - src)).collect(),
        gate_up: vec![None; experts.len()],
        down: vec![None; experts.len()],
        used: cursor,
    };
    for (&(src, _, eid, down), &run) in items.iter().zip(&item_run) {
        let (run_src, _, run_dst) = runs[run];
        let offset = run_dst + (src - run_src);
        if down {
            layout.down[eid] = Some(offset);
        } else {
            layout.gate_up[eid] = Some(offset);
        }
    }
    Ok(layout)
}

/// A tier layer's little-endian device pointer table: host experts rebased to
/// `stage_base + offset`, every other entry unchanged (the owner address).
pub(super) fn tier_stage_table(
    entries: &[usize],
    offsets: &[Option<usize>],
    stage_base: usize,
) -> Result<Vec<u8>, String> {
    if entries.len() != offsets.len() {
        return Err(format!(
            "{} table entries but {} stage offsets",
            entries.len(),
            offsets.len()
        ));
    }
    let mut table = Vec::with_capacity(entries.len() * std::mem::size_of::<u64>());
    for (expert, (&entry, offset)) in entries.iter().zip(offsets).enumerate() {
        let value = match offset {
            None => entry,
            Some(offset) => stage_base
                .checked_add(*offset)
                .ok_or_else(|| format!("expert {expert} rebased entry overflows"))?,
        };
        table.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(table)
}

/// The next layer after `layer` of the same parity with a staged slot.
fn next_staged_layer(slots: &[Option<usize>], layer: usize) -> Option<usize> {
    (layer + 2..slots.len()).step_by(2).find(|&next| slots[next].is_some())
}

/// Bytes of one stage buffer: the largest cold arena plus its tail pad.
pub(super) fn stage_buffer_bytes(arena_bytes: impl IntoIterator<Item = usize>) -> Option<usize> {
    arena_bytes
        .into_iter()
        .max()
        .and_then(|largest| largest.checked_add(ARENA_TAIL_PAD_BYTES))
}

/// A layer's per-expert pointer table moved from the layer's arena to
/// `stage_base`: every entry must lie inside `[arena_base, arena_base +
/// arena_bytes)`; the result is the little-endian device pointer table.
pub(super) fn rebase_stage_entries(
    entries: &[usize],
    arena_base: usize,
    arena_bytes: usize,
    stage_base: usize,
) -> Result<Vec<u8>, String> {
    let mut table = Vec::with_capacity(entries.len() * std::mem::size_of::<u64>());
    for (expert, &entry) in entries.iter().enumerate() {
        let offset = entry
            .checked_sub(arena_base)
            .filter(|offset| *offset < arena_bytes)
            .ok_or_else(|| {
                format!(
                    "expert {expert} entry {entry:#x} is outside the arena [{arena_base:#x}, +{arena_bytes})"
                )
            })?;
        let rebased = stage_base
            .checked_add(offset)
            .ok_or_else(|| format!("expert {expert} rebased entry overflows"))?;
        table.extend_from_slice(&rebased.to_ne_bytes());
    }
    Ok(table)
}

impl Qwen4ExpertStage {
    /// Slab-arena stage, or `None` (printed) when this bundle cannot stage.
    pub(super) fn new_slab(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        first_host: usize,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            Ok(None)
        };
        if let Some(lease) = bundle.weights.expert_tiers() {
            return Self::new_tier(gpu, bundle, moe, first_host, lease);
        }
        if bundle.weights.ep_placement().is_some() {
            return off("slab arenas have no EP route");
        }
        if !moe[first_host..]
            .iter()
            .all(|layer| layer.as_ref().is_some_and(RoutedExpertWeights::host_mapped))
        {
            return off("host-mapped slab layers are not a suffix");
        }
        let config = &bundle.config;
        let mut sources = Vec::with_capacity(moe.len() - first_host);
        for (layer, runtime) in moe
            .iter()
            .enumerate()
            .skip(first_host)
            .filter_map(|(layer, runtime)| runtime.as_ref().map(|runtime| (layer, runtime)))
        {
            let Some(slab) = runtime.slab.as_ref() else {
                return off("a host layer is not a slab arena");
            };
            let arena_ref = bundle
                .weights
                .layer(layer)
                .ok_or_else(|| invalid("slab stage layer is not owned"))?
                .ffn
                .moe()
                .and_then(|weights| weights.expert_arena.as_ref())
                .ok_or_else(|| invalid("slab stage layer has no arena reference"))?;
            let owner = bundle.weights.resident(arena_ref)?;
            let Some(host) = gpu.host_bytes(owner) else {
                return off("a host slab arena has no pinned host bytes");
            };
            if host.len() < slab.arena_bytes {
                return off("a host slab arena is shorter than its plan");
            }
            // Host-mapped allocations carry a zeroed tail pad past `len()`
            // (`HOST_TAIL_PAD_BYTES`, 1 MiB >= `ARENA_TAIL_PAD_BYTES`).
            sources.push((
                host.as_ptr() as usize,
                slab.arena_bytes + ARENA_TAIL_PAD_BYTES,
            ));
        }
        let Some(stage_bytes) = stage_buffer_bytes(
            moe[first_host..]
                .iter()
                .flatten()
                .filter_map(|runtime| runtime.slab.as_ref().map(|slab| slab.arena_bytes)),
        ) else {
            return off("no slab host layers");
        };
        // Staging is an optimisation: stage buffers that do not fit VRAM leave
        // every host layer reading its arena in place.
        let mut buffers = Vec::with_capacity(2);
        for _ in 0..2 {
            match gpu.alloc_tensor(&[stage_bytes], DType::Raw) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for buffer in buffers {
                        let _ = gpu.free_tensor(buffer);
                    }
                    return off(&format!(
                        "two {} MiB stage buffers do not fit VRAM: {error}",
                        stage_bytes >> 20
                    ));
                }
            }
        }
        gpu.ensure_capture_stream()?;
        let mut stage = Self {
            min_rows: qwen4_expert_stage_min_rows(),
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors: [usize::MAX; 2],
            donor_views: Vec::new(),
            mirrors: Vec::with_capacity(2),
            tables: Vec::with_capacity(2 * sources.len()),
            compact_tables: Vec::new(),
            compact_keys: Vec::new(),
            full_keys: Vec::new(),
            compact_bound: Vec::new(),
            first_host,
            placement: None,
            bytes: [0, 0],
            copy_stream: gpu.hip.stream_create_non_blocking()?,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
            slab: Some(SlabStageMeta { sources, tier: None }),
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            stage.mirrors.extend(buffers);
            for (layer, runtime) in moe
                .iter()
                .enumerate()
                .skip(first_host)
                .filter_map(|(layer, runtime)| runtime.as_ref().map(|runtime| (layer, runtime)))
            {
                let slab = runtime.slab.as_ref().expect("checked above");
                let stage_base = stage.mirrors[layer % 2].buf.as_ptr() as usize;
                for entries in [
                    &runtime.expert_gate_up_entries,
                    &runtime.expert_down_entries,
                ] {
                    let table = rebase_stage_entries(
                        entries,
                        slab.arena_base,
                        slab.arena_bytes,
                        stage_base,
                    )
                    .map_err(|reason| {
                        invalid(format!("layer {layer} slab stage table: {reason}"))
                    })?;
                    let tensor = gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?;
                    stage.tables.push(tensor);
                    gpu.memcpy_htod_auto(
                        &stage.tables.last().expect("table just allocated").buf,
                        &table,
                    )?;
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        eprintln!(
            "  qwen4 long-prefill expert staging: forwards of >= {} rows stage slab host layers {first_host}..{} through two {} MiB VRAM stage buffers (nothing borrowed, nothing to restore; wide-hidden capture forwards {})",
            stage.min_rows,
            moe.len(),
            stage_bytes >> 20,
            if stage.capture { "stage" } else { "mapped" }
        );
        Ok(Some(stage))
    }

    /// Per-expert tier stage, or `None` (printed) when it cannot stage. Only
    /// layers with a host-tier expert are staged; the layers need not be a
    /// suffix. Device-tier entries keep their owner address in the stage
    /// tables and the model-level owners are never written.
    fn new_tier(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        first_host: usize,
        lease: &ExpertTierLease,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            Ok(None)
        };
        if bundle.weights.ep_placement().is_some() {
            return off("expert tiers have no EP route");
        }
        let (Some((alias_base, alias_bytes)), Some((cpu_base, cpu_bytes))) =
            (lease.owner_range(Placement::HostMapped), lease.host_cpu_range())
        else {
            return off("the tier plan has no pinned host owner");
        };
        if alias_bytes != cpu_bytes {
            return Err(invalid("tier host owner alias and CPU ranges differ in size"));
        }
        let config = &bundle.config;
        let mut slots = Vec::with_capacity(moe.len());
        let mut layouts: Vec<(usize, TierStageLayout)> = Vec::new();
        for (layer, runtime) in moe.iter().enumerate() {
            let Some(runtime) = runtime.as_ref() else {
                slots.push(None);
                continue;
            };
            let view = lease
                .layer(layer)
                .ok_or_else(|| invalid(format!("tier stage: layer {layer} has no tier view")))?;
            // The forward's stage selection and `before_moe` wait use the
            // runtime's layer bit; it must be the view's `any_host`.
            if view.any_host() != runtime.host_mapped() {
                return Err(invalid(format!(
                    "tier stage: layer {layer} runtime host_mapped {} disagrees with its tier view {}",
                    runtime.host_mapped(),
                    view.any_host()
                )));
            }
            if !view.any_host() {
                slots.push(None);
                continue;
            }
            let (gate_up, down) = (view.gate_up_ptrs(), view.down_ptrs());
            if gate_up.len() != config.num_experts || down.len() != config.num_experts {
                return Err(invalid(format!("tier stage: layer {layer} table length differs from num_experts")));
            }
            let len = |bytes: u64| {
                usize::try_from(bytes).map_err(|_| invalid("tier stage: expert extent overflows usize"))
            };
            let experts = (0..config.num_experts)
                .map(|eid| {
                    let entry = view
                        .entry(eid)
                        .ok_or_else(|| invalid(format!("tier stage: layer {layer} expert {eid} has no entry")))?;
                    Ok(TierStageExpert {
                        host: entry.tier == Placement::HostMapped,
                        gate_up: gate_up[eid],
                        gate_up_len: len(entry.gate_up_len)?,
                        down: down[eid],
                        down_len: len(entry.down_len)?,
                    })
                })
                .collect::<Result<Vec<_>, Qwen4GpuForwardError>>()?;
            let layout = tier_stage_layout(&experts, alias_base, alias_bytes)
                .map_err(|reason| invalid(format!("tier stage: layer {layer}: {reason}")))?;
            slots.push(Some(layouts.len()));
            layouts.push((layer, layout));
        }
        if layouts.is_empty() {
            return off("no tier layer has a host expert");
        }
        // One buffer per layer parity: the largest staged payload of that
        // parity plus the zeroed tail pad.
        let mut sizes = [ARENA_TAIL_PAD_BYTES; 2];
        for (layer, layout) in &layouts {
            let need = layout
                .used
                .checked_add(ARENA_TAIL_PAD_BYTES)
                .ok_or_else(|| invalid("tier stage buffer size overflows"))?;
            sizes[layer % 2] = sizes[layer % 2].max(need);
        }
        // Pure checked translation of every host source from the device alias
        // to the pinned CPU address, before anything is allocated.
        let cpu_end = cpu_base
            .checked_add(cpu_bytes)
            .ok_or_else(|| invalid("tier host owner CPU range overflows"))?;
        let mut copies = Vec::with_capacity(layouts.len());
        for (layer, layout) in &layouts {
            let mut layer_copies = Vec::with_capacity(layout.copies.len());
            for &(src, dst, len) in &layout.copies {
                let cpu = src
                    .checked_sub(alias_base)
                    .and_then(|offset| cpu_base.checked_add(offset))
                    .filter(|cpu| cpu.checked_add(len).is_some_and(|end| end <= cpu_end))
                    .ok_or_else(|| {
                        invalid(format!(
                            "tier stage: layer {layer} source {src:#x} (+{len}) has no CPU address in the host owner"
                        ))
                    })?;
                layer_copies.push(StageCopy { src: cpu, dst, len });
            }
            copies.push(layer_copies);
        }
        let used = layouts.iter().map(|(_, layout)| layout.used).collect::<Vec<_>>();
        gpu.ensure_capture_stream()?;
        // Staging is an optimisation: stage buffers that do not fit VRAM leave
        // every host-tier expert read in place.
        let mut buffers = Vec::with_capacity(2);
        for size in sizes {
            match gpu.alloc_tensor(&[size], DType::Raw) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for buffer in buffers {
                        let _ = gpu.free_tensor(buffer);
                    }
                    return off(&format!(
                        "stage buffers of {} and {} MiB do not fit VRAM: {error}",
                        sizes[0] >> 20,
                        sizes[1] >> 20
                    ));
                }
            }
        }
        // From here the buffers are owned by `stage.mirrors`, except while the
        // copy stream is created.
        let copy_stream = match gpu.hip.stream_create_non_blocking() {
            Ok(stream) => stream,
            Err(error) => {
                for buffer in buffers {
                    let _ = gpu.free_tensor(buffer);
                }
                return Err(error.into());
            }
        };
        let mut stage = Self {
            min_rows: qwen4_expert_stage_min_rows(),
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors: [usize::MAX; 2],
            donor_views: Vec::new(),
            mirrors: buffers,
            tables: Vec::with_capacity(2 * layouts.len()),
            compact_tables: Vec::new(),
            compact_keys: Vec::new(),
            full_keys: Vec::new(),
            compact_bound: Vec::new(),
            first_host,
            placement: None,
            bytes: [0, 0],
            copy_stream,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
            slab: Some(SlabStageMeta {
                sources: Vec::new(),
                tier: Some(TierStageMeta { slots, copies, used, _keepalive: lease.keepalive() }),
            }),
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            // `mirrors` already owns the stage buffers (see `Self { .. }`).
            for (layer, layout) in &layouts {
                let view = lease.layer(*layer).expect("checked above");
                let stage_base = stage.mirrors[layer % 2].buf.as_ptr() as usize;
                for (entries, offsets) in
                    [(view.gate_up_ptrs(), &layout.gate_up), (view.down_ptrs(), &layout.down)]
                {
                    let table = tier_stage_table(entries, offsets, stage_base)
                        .map_err(|reason| invalid(format!("tier stage: layer {layer} table: {reason}")))?;
                    let tensor = gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?;
                    stage.tables.push(tensor);
                    gpu.memcpy_htod_auto(&stage.tables.last().expect("table just allocated").buf, &table)?;
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        let host_experts = layouts
            .iter()
            .map(|(_, layout)| layout.gate_up.iter().flatten().count())
            .sum::<usize>();
        eprintln!(
            "  qwen4 long-prefill expert staging: forwards of >= {} rows stage the {host_experts} host-tier experts of {} tier layers through two VRAM stage buffers of {} and {} MiB (device-tier experts keep their owner address; owners are never written, nothing to restore; wide-hidden capture forwards {})",
            stage.min_rows,
            layouts.len(),
            sizes[0] >> 20,
            sizes[1] >> 20,
            if stage.capture { "stage" } else { "mapped" }
        );
        Ok(Some(stage))
    }

    /// The tables a staged host `layer` reads (slab arena or tier). Replaces
    /// the `layer - first_host` indexing: tier host layers are not a suffix.
    pub(super) fn slab_tables(&self, layer: usize) -> MoeStageTables<'_> {
        let index = match self.slab.as_ref().and_then(|meta| meta.tier.as_ref()) {
            Some(tier) => tier
                .slots
                .get(layer)
                .copied()
                .flatten()
                .expect("tier stage tables requested for a layer with no host-tier expert"),
            None => layer - self.first_host,
        };
        MoeStageTables {
            gate_up: &self.tables[2 * index],
            down: &self.tables[2 * index + 1],
            free: &self.free[layer % 2],
        }
    }

    /// Start of a staged forward: the first two host layers (one per parity)
    /// copy while the resident layers compute. A tier stage primes the first
    /// staged layer of each parity.
    pub(super) fn slab_prime(&self, gpu: &Gpu, layers: usize) -> Result<(), Qwen4GpuForwardError> {
        let Some(meta) = self.slab.as_ref() else {
            return Ok(());
        };
        if let Some(tier) = meta.tier.as_ref() {
            for parity in 0..2 {
                let first = tier
                    .slots
                    .iter()
                    .enumerate()
                    .skip(parity)
                    .step_by(2)
                    .find_map(|(layer, slot)| slot.map(|_| layer));
                if let Some(layer) = first.filter(|&layer| layer < layers) {
                    self.slab_enqueue(gpu, layer)?;
                }
            }
            return Ok(());
        }
        for layer in self.first_host..(self.first_host + 2).min(layers) {
            self.slab_enqueue(gpu, layer)?;
        }
        Ok(())
    }

    /// Copy host `layer`'s arena (tier: its host-tier experts) over its
    /// parity's stage buffer once the previous occupant's down released it,
    /// recording `ready`.
    fn slab_enqueue(&self, gpu: &Gpu, layer: usize) -> Result<(), Qwen4GpuForwardError> {
        let meta = self.slab.as_ref().ok_or_else(|| invalid("not a slab stage"))?;
        let parity = layer % 2;
        if let Some(tier) = meta.tier.as_ref() {
            let slot = tier
                .slots
                .get(layer)
                .copied()
                .flatten()
                .ok_or_else(|| invalid("tier stage layer has no host-tier expert"))?;
            gpu.hip.stream_wait_event(&self.copy_stream, &self.free[parity])?;
            let buffer = &self.mirrors[parity].buf;
            for copy in &tier.copies[slot] {
                // SAFETY: `copy.src` lies inside the lease's pinned host owner
                // (checked against its range in `new_tier`), which `_keepalive`
                // retains and nothing writes after materialization.
                let source = unsafe { std::slice::from_raw_parts(copy.src as *const u8, copy.len) };
                gpu.hip.memcpy_htod_async(&buffer.byte_view(copy.dst, copy.len), source, &self.copy_stream)?;
            }
            // The previous occupant may have been larger: re-zero the pad the
            // kernels' tail over-read can reach past the last staged byte.
            gpu.hip.memset_async(
                &buffer.byte_view(tier.used[slot], ARENA_TAIL_PAD_BYTES),
                0,
                ARENA_TAIL_PAD_BYTES,
                &self.copy_stream,
            )?;
            gpu.hip.event_record(&self.ready[parity], Some(&self.copy_stream))?;
            return Ok(());
        }
        let &(address, bytes) = meta
            .sources
            .get(layer - self.first_host)
            .ok_or_else(|| invalid("slab stage layer out of range"))?;
        gpu.hip.stream_wait_event(&self.copy_stream, &self.free[parity])?;
        // SAFETY: `address` is the registered host allocation of the layer's
        // arena (immutable for the bundle's lifetime) and `bytes` is the arena
        // plus `ARENA_TAIL_PAD_BYTES` of its zeroed tail slack (see `new_slab`).
        let source = unsafe { std::slice::from_raw_parts(address as *const u8, bytes) };
        gpu.hip
            .memcpy_htod_async(&self.mirrors[parity].buf, source, &self.copy_stream)?;
        gpu.hip.event_record(&self.ready[parity], Some(&self.copy_stream))?;
        Ok(())
    }

    /// After a host layer's MoE (its down recorded `free`): refill its parity
    /// for the next host layer of the same parity (slab: L+2).
    pub(super) fn slab_after_moe(
        &self,
        gpu: &Gpu,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        layer: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        if let Some(tier) = self.slab.as_ref().and_then(|meta| meta.tier.as_ref()) {
            if tier.slots.get(layer).copied().flatten().is_none() {
                return Ok(());
            }
            return match next_staged_layer(&tier.slots, layer) {
                Some(next) => self.slab_enqueue(gpu, next),
                None => Ok(()),
            };
        }
        if !moe[layer]
            .as_ref()
            .is_some_and(RoutedExpertWeights::host_mapped)
        {
            return Ok(());
        }
        match moe.get(layer + 2) {
            Some(Some(next)) if next.host_mapped() => self.slab_enqueue(gpu, layer + 2),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_moves_every_entry_by_the_same_delta() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        let entries = [arena, arena + 1280, arena + 99_999];
        let table = rebase_stage_entries(&entries, arena, 100_000, stage).unwrap();
        let decoded: Vec<usize> = table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![stage, stage + 1280, stage + 99_999]);
    }

    #[test]
    fn rebase_refuses_entries_outside_the_arena_after_an_arena_swap() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        // Below the base, at the end, and a table of a different (swapped) arena.
        assert!(rebase_stage_entries(&[arena - 8], arena, 100_000, stage).is_err());
        assert!(rebase_stage_entries(&[arena + 100_000], arena, 100_000, stage).is_err());
        let other_arena = 0x9000_0000usize;
        assert!(rebase_stage_entries(&[other_arena + 16], arena, 100_000, stage).is_err());
        assert!(rebase_stage_entries(&[arena, usize::MAX], arena, 100_000, usize::MAX - 8).is_err());
    }

    #[test]
    fn stage_buffer_holds_the_largest_arena_and_its_tail_pad() {
        assert_eq!(stage_buffer_bytes([10, 30, 20]), Some(30 + ARENA_TAIL_PAD_BYTES));
        assert_eq!(stage_buffer_bytes(std::iter::empty()), None);
        assert_eq!(stage_buffer_bytes([usize::MAX]), None);
    }

    const GU: usize = 1_740_800;
    const DN: usize = 870_400;
    const HOST: usize = 0x7000_0000;

    fn expert(host: bool, base: usize) -> TierStageExpert {
        TierStageExpert { host, gate_up: base, gate_up_len: GU, down: base + GU, down_len: DN }
    }

    fn decode(table: &[u8]) -> Vec<usize> {
        table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn tier_layout_copies_only_host_experts_and_keeps_device_entries() {
        let device = 0x2_0000_0000usize;
        // Experts 1 and 3 are host, adjacent in the host owner; 0 and 2 device.
        let experts = [
            expert(false, device),
            expert(true, HOST + 4096),
            expert(false, device + GU + DN),
            expert(true, HOST + 4096 + GU + DN),
        ];
        let layout = tier_stage_layout(&experts, HOST, 64 << 20).unwrap();
        // One coalesced run covering both experts' gate/up and down.
        assert_eq!(layout.copies, vec![(HOST + 4096, 0, 2 * (GU + DN))]);
        assert_eq!(layout.used, 2 * (GU + DN));
        assert_eq!(layout.gate_up, vec![None, Some(0), None, Some(GU + DN)]);
        assert_eq!(layout.down, vec![None, Some(GU), None, Some(GU + DN + GU)]);
        let entries: Vec<usize> = experts.iter().map(|e| e.gate_up).collect();
        let stage = 0x3_0000_0000usize;
        let table = decode(&tier_stage_table(&entries, &layout.gate_up, stage).unwrap());
        // Same length and original local-id order; device entries untouched.
        assert_eq!(table, vec![device, stage, device + GU + DN, stage + GU + DN]);
    }

    #[test]
    fn tier_layout_packs_separate_runs_aligned_and_never_reads_device_bytes() {
        let experts = [expert(true, HOST + 2 * (GU + DN)), expert(true, HOST), expert(false, 0x2_0000_0000)];
        let layout = tier_stage_layout(&experts, HOST, 64 << 20).unwrap();
        assert_eq!(layout.copies.len(), 2);
        // Sorted by source address; each run 256-aligned in the stage.
        assert_eq!(layout.copies[0], (HOST, 0, GU + DN));
        assert_eq!(layout.copies[1], (HOST + 2 * (GU + DN), GU + DN, GU + DN));
        assert!(layout.copies.iter().all(|&(src, dst, _)| src % 256 == 0 && dst % 256 == 0));
        // Expert 0 (host, later source) maps to the second run; ids keep their slots.
        assert_eq!(layout.gate_up, vec![Some(GU + DN), Some(0), None]);
    }

    #[test]
    fn tier_layout_refuses_extents_outside_the_host_owner() {
        let in_owner = expert(true, HOST);
        assert!(tier_stage_layout(&[in_owner], HOST, GU + DN - 1).is_err());
        assert!(tier_stage_layout(&[expert(true, HOST - 256)], HOST, 64 << 20).is_err());
        // A host-flagged expert whose pointer names the device owner is refused.
        assert!(tier_stage_layout(&[expert(true, 0x2_0000_0000)], HOST, 64 << 20).is_err());
        let mut zero = in_owner;
        zero.down_len = 0;
        assert!(tier_stage_layout(&[zero], HOST, 64 << 20).is_err());
        let mut unaligned = in_owner;
        unaligned.gate_up += 8;
        assert!(tier_stage_layout(&[unaligned], HOST, 64 << 20).is_err());
    }

    #[test]
    fn tier_layout_with_no_host_experts_stages_nothing() {
        let layout = tier_stage_layout(&[expert(false, 0x2_0000_0000)], HOST, 64 << 20).unwrap();
        assert!(layout.copies.is_empty());
        assert_eq!(layout.used, 0);
        assert_eq!(layout.gate_up, vec![None]);
    }

    #[test]
    fn tier_table_refuses_mismatched_offsets_and_overflow() {
        assert!(tier_stage_table(&[1, 2], &[None], 0).is_err());
        assert!(tier_stage_table(&[0], &[Some(16)], usize::MAX - 8).is_err());
    }

    #[test]
    fn next_staged_layer_walks_same_parity_only() {
        let slots = [None, Some(0), None, Some(1), Some(2), None, Some(3), None];
        assert_eq!(next_staged_layer(&slots, 1), Some(3));
        assert_eq!(next_staged_layer(&slots, 3), None);
        assert_eq!(next_staged_layer(&slots, 4), Some(6));
        assert_eq!(next_staged_layer(&slots, 6), None);
    }
}
