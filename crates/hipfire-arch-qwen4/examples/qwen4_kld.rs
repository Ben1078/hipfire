// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_kld — KLD reference builder and candidate scorer for Qwen4
//! (Qwen3.8 Flash-Next), on the saddle-quant HFKLDR v3 / HFKSEQ v2 formats.
//!
//! `ref` and `eval` run the production `Qwen4Bundle::forward_chunk` (all-row
//! logits, per-chunk `reset`) so an engine teacher and the candidate share one
//! route and differ only in weights.
//!
//! Engine references measure only differences between the supplied artifacts;
//! use the external source teacher to measure shared expert quantization error.
//! The source-precision teacher is the BF16 checkpoint itself, run by
//! `reference_oracle/kld_teacher.py` over a template reference's tokens;
//! `import` turns its logits into a reference with the same top-k, NLL and
//! plausibility gate as `ref`.
//!
//! Usage:
//!   qwen4_kld ref    --model TEACHER.hfq --slice TEXT --output REF.kldref \
//!                    [--n-ctx 512] [--top-k 256] [--max-chunks N]
//!   qwen4_kld import --logits TEACHER.logits --ref TEMPLATE.kldref \
//!                    --teacher ID_FILE --output REF.kldref [--top-k 256]
//!   qwen4_kld eval   --model CANDIDATE.hfq --ref REF.kldref --output OUT.kldseq \
//!                    [--max-chunks N] [--decode | --append-rows R[,R...] | --tail-split K]
//!                    [--score-tail K] [--ple-hot BYTES]
//!   qwen4_kld s11ref --model CANDIDATE.hfq --ref REF.s11ref \
//!                    --corpus CORPUS.json --output OUT.npy [--decode] [--ple-hot BYTES] \
//!                    [--sequence-range START:END]
//!   qwen4_kld speed --model MODEL.hfq --prompt-file PROMPT.txt --output OUT.json \
//!                    [--product-prompt] [--decode-tokens 128] [--speed-warmup 8] [--reps 2]
//!   qwen4_kld pair --base BASE.npy --cand CAND.npy --corpus CORPUS.json --output OUT.json
//! `pair` (CPU-only) reads two full-gate `s11ref` outputs (each with its
//! OUT.npy.json receipt, route `decode` or `prefill`, both arms the same)
//! over the same reference and corpus: per-corpus-sequence mean KL first,
//! then the paired candidate-minus-base 95% t interval; pass when `lo <= 0`,
//! else `hi <= +0.010`. The corpus must be a pinned gate corpus (code24: 24
//! sequences / 6120 rows; WT2: 32 sequences / 8160 rows), identified by its
//! sha256; every row and sequence is required; a partition, a partial corpus
//! or a different reference/corpus/config is refused, and OUT.json is
//! written either way. The decode `s11ref` receipt also records
//! `slowlink_prefix_routes`: the S1 route receipt of every unscored prefix
//! (`forward_chunk_final`), so a run states which prefixes took layer-major.
//! The `prefill` route scores all-row logits (N x vocab) in ONE forward per
//! sequence (`forward_chunk` tiles by the forward's prefill chunk, so the
//! receipt records every sequence's forward rows, the aggregate MoE launch
//! census delta of that forward and the resolved
//! `HIPFIRE_QWEN4_MOE_SYM_IU4` request and descriptor) and never exercises
//! S1. Its receipt-level `moe_route` is `iu4` (every sequence launched the
//! IU4 gate/up and down GEMMs and no F16 WMMA one), `f16` (the reverse) or
//! `mixed` (always a failure for pairing). `pair --moe-iu4` is the IU4
//! prefill pairing: baseline `f16` with the flag off, candidate `iu4` with
//! `HIPFIRE_QWEN4_MOE_SYM_IU4=1`, one binary and model, an otherwise
//! identical `HIPFIRE_*` environment; a requested IU4 that fell back is a
//! failure, never a pass.
//! `--product-prompt` uses the product's cold, single-user answer-mode prompt
//! renderer (no system, tools or history). Speed receipts retain prompt ids,
//! full generated ids (including warmup), and the product's terminator ids.
//! `s11ref` writes F64 KL(ref || candidate), in the reference score-index order,
//! and OUT.npy.json with model/ref/corpus/binary SHA256. S11 refs contain no
//! prefix tokens: --corpus must be the exact corpus bound by the header SHA.
//! Every corpus sequence is one cold window, including unequal assistant
//! sequences; score_positions selects role-masked next-token rows. The estimator
//! is the same top-k plus exact F64 reference residual as fp8_refs.score_rows.
//! `--sequence-range START:END` (s11ref only; excludes `--smoke-seqs`) scores corpus
//! sequences START..END (zero-based, END exclusive) in one process and writes only
//! their canonical rows to OUT.npy, with an OUT.npy.json of schema
//! `hipfire.qwen4.s11ref_klrows.partition.v1` (full sha fields plus sequence_start/end,
//! total_sequences, full_rows, row_start/end, rows, route). A partition is never a
//! full-gate PASS; the caller must tile every sequence and aggregate the rows.
//! `--decode` scores the autoregressive route instead: each chunk prefills the
//! unscored prefix, then runs one `forward_token` per scored position.
//! `--append-rows R[,R...]` scores the agent-turn append route instead: each
//! chunk runs from a fresh state as consecutive all-row forwards of at most R
//! rows on the resident state (the last one shorter), so every scored row comes
//! from an append of at most R rows. One model load scores every R listed;
//! each R writes `OUT` with `.a<R>` before its extension.
//! `--tail-split K` prefills each chunk as exactly two forwards on a fresh
//! state, rows [0, W-K) then [W-K, W) for window length W (it excludes
//! `--decode` and `--append-rows`) and implies `--score-tail K`.
//! `--score-tail K` also reports mean KLD, top-1 and row count over only the
//! scored rows inside the last K rows of each window, as a printed `tail` line
//! and in `OUT.json`. The last window row has no target, so a window scored to
//! its end yields K-1 tail rows per chunk. Scored output is otherwise unchanged.
//! `--ple-hot BYTES` attaches the PLE hot-row store (`<model>.plehot` sidecar
//! and `<model>.plehotrows` pack), verified and with BYTES of the pack
//! prewarmed, before scoring, so the identity gate covers rows read from the
//! pack.
//! Model-running commands also accept `--hot-budget-mib N [--hotset SIDECAR]`
//! to freeze the same bound expert partition as `qwen4_wide_verify`. CPU mode
//! requires an armed D2 split; PCIe mode requires it unarmed. Passback, UMA,
//! disabled CPU experts and conflicting expert VRAM-layer pins fail closed.
//! Without these flags the existing loader is unchanged. `--decode` keeps the
//! production AR n=1 route, including D2 host slots, for every scored row.
//! Reduce with `saddle-quant reduce DIR`.

use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::{ple_hot_plan, PleHotPlan, PleHotWorker, Qwen4Bundle};
use hipfire_config::memory::OffloadExec;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::ple_hot_store::{HotRowStore, PleHotRank};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, PleRowsOrigin, ResolvePolicy};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use saddle_quant::eval::estimator::{kld_from_block, nll_of, topk_block, TopKBlock};
use saddle_quant::eval::teacher::{evaluate, TeacherGate};
use saddle_quant::format::kldref::{KldRef, RefHeader, RefWriter};
use saddle_quant::format::kldseq::{self, ChunkScore};
use saddle_quant::{ArtifactId, Estimator, OracleStats, TeacherVerdict, WindowSpec};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Shared S1 receipt/KLD validation (GPU-free): the paired verdict below.
#[path = "../tests/common/slowlink_receipt.rs"]
mod slowlink_receipt;

/// Open the PLE hot-row store from `<model>.plehot` and the
/// `<rows source>.plehotrows` pack (`plan.source_path` is the model file or
/// its PLE sidecar), wait until its worker verified the pack, prewarm
/// `prewarm` bytes of it, and return it ready to attach.
#[cfg(not(unix))]
fn verified_hot_store(
    _model: &Path,
    _plan: PleHotPlan,
    _prewarm: u64,
) -> Result<PleHotWorker, String> {
    Err("--ple-hot: PLE hot-row verification and prewarm are unsupported on Windows; run this diagnostic on a Unix host".into())
}

#[cfg(unix)]
fn verified_hot_store(
    model: &Path,
    plan: PleHotPlan,
    prewarm: u64,
) -> Result<PleHotWorker, String> {
    let sidecar = PathBuf::from(format!("{}.plehot", model.display()));
    let pack = PathBuf::from(format!("{}.plehotrows", plan.source_path.display()));
    let rank = PleHotRank::open(&sidecar, plan.padded_rows, &plan.fingerprint_hex)
        .map_err(|e| format!("--ple-hot: sidecar {}: {e}", sidecar.display()))?;
    let store = HotRowStore::open(
        rank,
        &pack,
        plan.rows_per_shard,
        plan.row_encoding,
        plan.encoded_row_bytes,
        &plan.shard_offsets,
        plan.source_file_bytes,
    )?;
    let started = Instant::now();
    let handle = store.spawn_worker(&plan.source_path, plan.shard_offsets);
    let hot = PleHotWorker::new(store.clone(), handle);
    store.set_prewarm_budget(prewarm);
    loop {
        match store.verified() {
            Some(true) if store.prewarm_complete() => break,
            Some(false) => return Err("--ple-hot: the pack failed verification".into()),
            _ if started.elapsed().as_secs() > 900 => {
                return Err("--ple-hot: verification and prewarm did not finish in 900 s".into())
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
    eprintln!(
        "qwen4_kld: ple hot store verified, {:.2} of {:.2} GiB of the pack prewarmed, {:.1} s",
        store.prewarmed_bytes() as f64 / (1u64 << 30) as f64,
        store.data_bytes() as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64()
    );
    Ok(hot)
}

const USAGE: &str = "usage:\n  qwen4_kld ref    --model TEACHER.hfq --slice TEXT --output REF.kldref [--n-ctx 512] [--top-k 256] [--max-chunks N]\n  qwen4_kld import --logits TEACHER.logits --ref TEMPLATE.kldref --teacher ID_FILE --output REF.kldref [--top-k 256]\n  qwen4_kld eval   --model CANDIDATE.hfq --ref REF.kldref --output OUT.kldseq [--max-chunks N] [--decode | --append-rows R[,R...] | --tail-split K] [--score-tail K] [--ple-hot BYTES]\n  qwen4_kld s11ref --model CANDIDATE.hfq --ref REF.s11ref --corpus CORPUS.json --output OUT.npy [--decode] [--ple-hot BYTES] [--sequence-range START:END]\n  qwen4_kld pair --base BASE.npy --cand CAND.npy --corpus CORPUS.json --output OUT.json [--moe-iu4]\n  qwen4_kld speed --model MODEL.hfq --prompt-file PROMPT.txt --output OUT.json [--product-prompt] [--decode-tokens 128] [--speed-warmup 8] [--reps 2] [--max-prompt N]\n  ref/eval/s11ref/speed: [--hot-budget-mib N] [--hotset SIDECAR] (hotset requires a nonzero budget; fixed CPU/PCIe partition)";

/// How a chunk is prefilled from a fresh state.
#[derive(Clone, Copy)]
enum Prefill {
    /// One all-row forward.
    Whole,
    /// Consecutive all-row forwards of at most this many rows.
    Append(usize),
    /// Exactly two forwards: `[0, W-K)` then `[W-K, W)`.
    TailSplit(usize),
}

/// Reuse the backing allocation, but expose exactly this cold window's rows.
fn all_logits_view(backing: &GpuTensor, tokens: usize, vocab: usize) -> GpuTensor {
    backing.sub_offset(0, tokens * vocab)
}

#[derive(Default)]
struct HotsetPin {
    budget_mib: Option<u64>,
    sidecar: Option<PathBuf>,
}

impl HotsetPin {
    fn parse_budget(text: &str) -> Result<u64, String> {
        let mib: u64 = text
            .parse()
            .map_err(|_| format!("bad --hot-budget-mib '{text}'"))?;
        if mib == 0 || mib.checked_mul(1 << 20).is_none() {
            return Err("--hot-budget-mib must be nonzero and fit in bytes".into());
        }
        Ok(mib)
    }

    fn validate(&self, mode: &str) -> Result<(), String> {
        if self.sidecar.is_some() && self.budget_mib.is_none() {
            return Err("--hotset requires a nonzero --hot-budget-mib".into());
        }
        if self.budget_mib.is_some() && !matches!(mode, "ref" | "eval" | "s11ref" | "speed") {
            return Err("--hot-budget-mib and --hotset require ref, eval, s11ref or speed".into());
        }
        Ok(())
    }
}

// Keep the pinned controls identical to wide_verify, independently testable
// without initializing HIP or touching the model.
fn admit_pinned_control(
    exec: OffloadExec,
    uma: bool,
    cpu_experts_enabled: bool,
    explicit_vram_layers: bool,
) -> Result<(), String> {
    if uma {
        return Err("fixed hotset requires a discrete GPU".into());
    }
    if explicit_vram_layers {
        return Err("fixed hotset conflicts with HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS".into());
    }
    match exec {
        OffloadExec::Pcie => Ok(()),
        OffloadExec::Cpu if cpu_experts_enabled => Ok(()),
        OffloadExec::Cpu => Err("fixed CPU hotset requires CPU experts enabled".into()),
        OffloadExec::Passback => {
            Err("fixed hotset requires HIPFIRE_OFFLOAD_EXEC=cpu or pcie (not passback)".into())
        }
    }
}

fn admit_pinned_split(
    exec: OffloadExec,
    has_split_layers: bool,
    has_split_digest: bool,
) -> Result<(), String> {
    match exec {
        OffloadExec::Cpu if has_split_layers && has_split_digest => Ok(()),
        OffloadExec::Cpu => Err("requested D2 hotset split did not arm".into()),
        OffloadExec::Pcie if !has_split_layers && !has_split_digest => Ok(()),
        OffloadExec::Pcie => Err("pinned PCIe control unexpectedly armed a D2 split".into()),
        OffloadExec::Passback => Err("passback is not a pinned control".into()),
    }
}

fn partition_digest_hex(digest: [u8; 32]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("write digest to String");
    }
    hex
}

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    logits: GpuTensor,
    vocab: usize,
    arch_id: u32,
    tokenizer: Tokenizer,
}

impl Model {
    /// Same range-loaded path as the serve loader (`hipfire-loader` carriers).
    /// `ple_hot`: bytes of the PLE hot-row pack to prewarm; the store is
    /// attached verified before the first forward.
    fn load(
        path: &Path,
        n_ctx: usize,
        logit_rows: usize,
        ple_hot: Option<u64>,
        pin: &HotsetPin,
    ) -> Result<Self, String> {
        // S1 selector (`HIPFIRE_QWEN4_SLOWLINK_PREFILL` unset or `1`): the
        // shared admission-first loader runs ONLY when the same selector the
        // product uses chooses S1 for this load; otherwise (selector `0`,
        // unset and declined, fast/unified/unmeasured link, a pinned
        // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`) it returns before any
        // allocation and the loader below runs, placement and all, exactly as
        // before. Pinned hot sets and the PLE hot store are other routes: a
        // forced `1` refuses them, and with them the loader below is kept.
        let s1_mode = hipfire_arch_qwen4::slowlink_admission::SlowlinkMode::from_env();
        let hot_flags = ple_hot.is_some() || pin.budget_mib.is_some() || pin.sidecar.is_some();
        if hot_flags && s1_mode == hipfire_arch_qwen4::slowlink_admission::SlowlinkMode::Force {
            return Err("HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 excludes --ple-hot, --hotset and --hot-budget-mib".into());
        }
        // The GPU arch drives `memory.ple_format = auto` and the S1 selector;
        // initialising it allocates no model memory, so admission still
        // precedes every upload.
        let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
        if s1_mode != hipfire_arch_qwen4::slowlink_admission::SlowlinkMode::Off && !hot_flags {
            if let Some(loaded) = hipfire_arch_qwen4::slowlink_load::slowlink_load_if_selected(
                &mut gpu, path, n_ctx, None,
            )? {
                let logits = gpu
                    .zeros(&[logit_rows * loaded.vocab], DType::F32)
                    .map_err(|e| e.to_string())?;
                eprintln!(
                    "qwen4_kld: loaded {} on {} via the slowlink loader (vocab {})",
                    path.display(),
                    gpu.arch,
                    loaded.vocab
                );
                return Ok(Self {
                    gpu,
                    bundle: loaded.bundle,
                    logits,
                    vocab: loaded.vocab,
                    arch_id: loaded.arch_id,
                    tokenizer: loaded.tokenizer,
                });
            }
        }
        let mut hfq =
            HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let arch_id = hfq.arch_id;
        // The GPU arch drives `memory.ple_format = auto`; initialising it
        // allocates no model memory, so admission still precedes every upload.
        // The evaluator never derives a missing q4: derive it with
        // `hipfire ple derive` (or a serve load) first.
        let choice = resolve_and_attach_ple_rows(
            &mut hfq,
            path,
            hipfire_config::memory::ple_format(),
            &gpu.arch,
            ResolvePolicy {
                derive_missing_q4: false,
                progress: None,
            },
        )
        .map_err(|e| format!("qwen4 PLE rows: {e}"))?;
        let mut receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
            .map_err(|e| format!("qwen4 admission: {e}"))?;
        // `<model>.trunk-overlay` (QT44 trunk tensors + the A4 default), bound
        // to this base while it is open and applied by `attach_forward`.
        let trunk_overlay =
            hipfire_arch_qwen4::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
                .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
        // A rank-order table is served through its store (no shard
        // descriptors); every other source reads through 128 descriptors.
        let rank_table = receipt.ple_rows.rank_table().is_some();
        let ple_descriptors = if rank_table {
            None
        } else {
            Some(
                receipt
                    .ple_row_descriptors(&hfq)
                    .map_err(|e| format!("qwen4 PLE rows: {e}"))?,
            )
        };
        let (ple_rows_path, ple_rows_bytes) = receipt
            .ple_rows_source(&hfq)
            .map_err(|e| format!("qwen4 PLE rows: {e}"))?;
        let (rows_encoding, rows_total) = match (&receipt.ple_rows, hfq.ple_rows()) {
            (hipfire_arch_qwen4::Qwen4PleRows::Sidecar { encoding, .. }, Some(rows)) => {
                (encoding.dtype_name().to_string(), rows.header().data_bytes)
            }
            _ => {
                let descriptors = ple_descriptors.as_deref().unwrap_or_default();
                (
                    descriptors
                        .first()
                        .map_or_else(String::new, |descriptor| descriptor.dtype.clone()),
                    descriptors.iter().map(|descriptor| descriptor.length).sum(),
                )
            }
        };
        eprintln!(
            "qwen4_kld: ple rows: {} from {} ({rows_encoding}, {} order, {:.1} GB)",
            choice.format.name(),
            match &choice.origin {
                PleRowsOrigin::Embedded => "embedded model file".to_string(),
                PleRowsOrigin::Sidecar(sidecar) => sidecar.display().to_string(),
            },
            if rank_table { "rank" } else { "row" },
            rows_total as f64 / 1e9
        );
        if let Some(note) = &choice.note {
            eprintln!("qwen4_kld: ple rows: {note}");
        }
        // The loader's forward scratch plan, decided from the actual payload
        // while the mapping is readable.
        let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
            &gpu,
            &hfq,
            &receipt.config,
            &receipt.manifest,
        )
        .map_err(|e| e.to_string())?;
        // Materialize the frozen hot/cold partition before the ordinary
        // manifest is fulfilled: cold experts must never get a pooled upload.
        let tier_lease = if let Some(mib) = pin.budget_mib {
            use hipfire_arch_qwen4::expert_hotset as hotsets;
            admit_pinned_control(
                hipfire_config::memory::offload_exec(),
                gpu.is_uma(),
                hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled(),
                hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()?.is_some(),
            )?;
            let policy = match pin.sidecar.as_deref() {
                Some(path) => hotsets::HotsetPolicy::Explicit(path.to_path_buf()),
                None => hotsets::HotsetPolicy::from_env()?,
            };
            let env = hotsets::HotsetEnv {
                gpu_arch: &gpu.arch,
                unified_memory: false,
                expert_parallel: false,
                explicit_vram_layers: false,
                source: match &receipt.manifest.identity.recipe {
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => {
                        hotsets::HotsetSource::Aggregate
                    }
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => {
                        hotsets::HotsetSource::SlabArena
                    }
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => {
                        hotsets::HotsetSource::StockFp8
                    }
                    _ => {
                        hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into())
                    }
                },
            };
            let sidecar_path = hotsets::preflight(&policy, path, &env).map_err(|skip| {
                skip.log_line(&policy.sidecar_path(path))
                    .unwrap_or_else(|| skip.to_string())
            })?;
            let base = hotsets::HotsetBase::from_hfq(&hfq)?;
            let sidecar = hotsets::HotsetSidecar::load_bound(
                &sidecar_path,
                receipt.config.num_hidden_layers,
                receipt.config.num_experts,
                receipt.config.num_experts_per_tok,
                &base,
            )
            .map_err(|skip| skip.to_string())?;
            receipt
                .manifest
                .check_expert_tiers()
                .map_err(|e| e.to_string())?;
            let slabs = hipfire_arch_qwen4::artifact::hotset_aggregate_slabs(
                &hfq,
                &receipt.config,
                &receipt.manifest,
            )?;
            let plan = hipfire_arch_qwen4::expert_tier_plan::plan_expert_tiers_from_slabs(
                &slabs,
                sidecar.rank(),
                mib * (1 << 20),
            )
            .map_err(|e| e.to_string())?;
            hotsets::require_experts(plan.device_experts()).map_err(|skip| skip.to_string())?;
            plan.check_executable(&gpu.arch)
                .map_err(|e| e.to_string())?;
            hipfire_arch_qwen4::expert_residency::place_routed_experts(
                &mut receipt.manifest.weights,
                receipt.config.num_hidden_layers,
            );
            receipt.manifest = receipt
                .manifest
                .with_expert_tiers()
                .map_err(|e| e.to_string())?;
            {
                let keep: BTreeSet<_> = receipt.manifest.resident_keys().collect();
                receipt
                    .placements
                    .retain(|p| keep.contains(&(p.name.as_str(), p.layer)));
            }
            let lease = hipfire_arch_qwen4::expert_tier_owners::materialize(
                &mut gpu,
                std::sync::Arc::new(plan),
                &|offset, len| {
                    let offset = usize::try_from(offset)
                        .map_err(|_| "hotset source offset overflow".to_string())?;
                    let len = usize::try_from(len)
                        .map_err(|_| "hotset source length overflow".to_string())?;
                    hfq.data_range(offset, len).ok_or_else(|| {
                        format!("source span {offset}+{len} is outside the artifact mapping")
                    })
                },
            )
            .map_err(|e| format!("expert tier materialize: {e}"))?;
            lease.bind_hotset_sidecar_sha256(sidecar.sha256())?;
            eprintln!(
                "qwen4_kld: hotset {} sha256 {} fixed budget {mib} MiB",
                sidecar_path.display(),
                sidecar.sha256()
            );
            Some(lease)
        } else {
            None
        };
        // This is the real logical plan in BOTH controls, not the D2-only
        // split identity (which correctly remains absent in PCIe mode).
        let partition_digest = tier_lease
            .as_ref()
            .map(|lease| partition_digest_hex(lease.logical_digest()));
        if gpu.is_uma() {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let hot_plan =
            match (ple_hot, ple_descriptors.as_deref()) {
                (Some(_), Some(descriptors)) => Some(
                    ple_hot_plan(
                        receipt.ple.as_ref().ok_or(
                            "--ple-hot requires PLE hash metadata; this profile has no PLE",
                        )?,
                        descriptors,
                        &ple_rows_path,
                        ple_rows_bytes,
                    )?
                    .ok_or("--ple-hot: the PLE rows are neither Q8F16 nor Q4LM")?,
                ),
                (Some(_), None) => return Err(
                    "--ple-hot: the PLE rows are a rank-order table, which is its own hot-row pack"
                        .into(),
                ),
                (None, _) => None,
            };
        let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
        let source = HfqModelSource::from_hfq(hfq);
        // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk layers 0..N resident
        // and host-maps the rest (a slab artifact larger than one card).
        let weights = {
            let mut weights = receipt.manifest.weights.clone();
            match hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()? {
                None => {}
                Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)) => {
                    if gpu.is_uma() {
                        return Err("host-mapped experts are only for discrete GPUs".into());
                    }
                    let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
                        &mut weights,
                        layers,
                    );
                    eprintln!(
                        "qwen4_kld: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                    );
                }
                Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                    return Err("qwen4_kld takes a fixed expert layer count, not auto".into());
                }
            }
            weights
        };
        let transaction = fulfill_manifest_from_payloads(
            &weights,
            &mesh,
            receipt.config.num_hidden_layers,
            &mut gpu,
            expected,
            |entry| {
                if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                    &source,
                    &receipt.manifest.identity.recipe,
                    entry,
                ) {
                    return payload;
                }
                // The store rejects a host-mapped entry as a range: borrow it
                // from the resident source bytes, as the product loader does.
                if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped
                {
                    let (info, bytes) = source
                        .tensor_data(&entry.name)
                        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                    return Ok(SourcePayload::Borrowed { info, bytes });
                }
                source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name))
            },
        )
        .map_err(|e| format!("qwen4 manifest fulfillment: {e}"))?;
        let vocab = receipt.config.vocab_size;
        let state_format = hipfire_arch_qwen4::resolve_state_format(
            &hipfire_runtime::config::get().kv_mode,
            // Qwen3.5's DeltaNet `state_quant` knob: auto (q8) or fp32.
            &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
            &gpu,
            &receipt.config,
        )?;
        let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
        let mut bundle = Qwen4Bundle::assemble_with_ple_rows(
            receipt.config,
            &receipt.manifest.identity.recipe,
            transaction,
            &receipt.placements,
            &mut gpu,
            n_ctx,
            receipt.ple,
            receipt.ple_rows,
            state_format,
            backend,
            tier_lease,
        )
        .map_err(|e| format!("qwen4 bundle assembly: {e}"))?;
        bundle.attach_npu_trunk_identity(npu_identity);
        if let Some(overlay) = trunk_overlay {
            bundle
                .attach_trunk_overlay(overlay)
                .map_err(|e| format!("qwen4 trunk overlay: {e}"))?;
        }
        bundle
            .attach_forward(&mut gpu, n_ctx, scratch_mode)
            .map_err(|e| format!("qwen4 forward setup: {e}"))?;
        if let Some(digest) = partition_digest {
            let offload_exec = hipfire_config::memory::offload_exec();
            let has_split_layers = !bundle.split_layers().is_empty();
            let has_split_digest = bundle.split_digest().is_some();
            admit_pinned_split(offload_exec, has_split_layers, has_split_digest)?;
            eprintln!(
                "qwen4_kld: offload_exec={offload_exec} split_armed={} partition_digest={digest}",
                has_split_layers && has_split_digest
            );
            eprintln!("qwen4_kld: split_digest {:?}", bundle.split_digest());
        }
        if let (Some(prewarm), Some(plan)) = (ple_hot, hot_plan) {
            bundle
                .attach_ple_hot_store(verified_hot_store(path, plan, prewarm)?)
                .map_err(|e| format!("qwen4 PLE hot store: {e}"))?;
        }
        let logits = gpu
            .zeros(&[logit_rows * vocab], DType::F32)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "qwen4_kld: loaded {} on {} (vocab {vocab})",
            path.display(),
            gpu.arch
        );
        let resolved_format = bundle.state_format();
        eprintln!(
            "[qwen4_kld] resolved kv_mode={:?} qsa_state={:?} gdn_state={:?}",
            hipfire_runtime::config::get().kv_mode,
            resolved_format.qsa,
            resolved_format.gdn
        );
        Ok(Self {
            gpu,
            bundle,
            logits,
            vocab,
            arch_id,
            tokenizer,
        })
    }

    /// All-row logits for one chunk from a fresh state: one forward, with
    /// `Append` consecutive forwards of at most that many rows on the
    /// resident state, or with `TailSplit` exactly two forwards split that many
    /// rows before the end (each writes its own logits rows).
    fn chunk_logits(&mut self, chunk: &[u32], prefill: Prefill) -> Result<Vec<f32>, String> {
        self.bundle
            .reset(&mut self.gpu)
            .map_err(|e| e.to_string())?;
        let parts: Vec<&[u32]> = match prefill {
            Prefill::Whole => vec![chunk],
            Prefill::Append(rows) => chunk.chunks(rows).collect(),
            Prefill::TailSplit(rows) => {
                let (head, tail) = chunk.split_at(chunk.len() - rows);
                vec![head, tail]
            }
        };
        let mut at = 0;
        for part in parts {
            let logits = self
                .logits
                .sub_offset(at * self.vocab, part.len() * self.vocab);
            self.bundle
                .forward_chunk(&mut self.gpu, part, &logits, None)
                .map_err(|e| e.to_string())?;
            at += part.len();
        }
        let rows = self
            .gpu
            .download_f32(&all_logits_view(&self.logits, chunk.len(), self.vocab))
            .map_err(|e| e.to_string())?;
        if rows.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok(rows)
    }

    /// Logits rows `from..to` of one chunk through the decode route: prefill
    /// `chunk[..from]` from a fresh state, then one `forward_token` per row.
    fn decode_rows(&mut self, chunk: &[u32], from: usize, to: usize) -> Result<Vec<f32>, String> {
        let row = self.logits.sub_offset(0, self.vocab);
        self.bundle
            .reset(&mut self.gpu)
            .map_err(|e| e.to_string())?;
        if from > 0 {
            self.bundle
                .forward_chunk_final(&mut self.gpu, &chunk[..from], &row, None)
                .map_err(|e| e.to_string())?;
        }
        let mut rows = Vec::with_capacity((to - from) * self.vocab);
        for &token in &chunk[from..to] {
            self.bundle
                .forward_token(&mut self.gpu, token, &row, None)
                .map_err(|e| e.to_string())?;
            rows.extend(self.gpu.download_f32(&row).map_err(|e| e.to_string())?);
        }
        if rows.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok(rows)
    }
}

fn artifact_id(path: &Path) -> Result<ArtifactId, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
        bytes += n as u64;
    }
    Ok(ArtifactId {
        path: path.display().to_string(),
        sha256: format!("{:x}", digest.finalize()),
        bytes,
    })
}

/// Lowercase hex MD5 of a file (identity digest of the pinned references,
/// corpora and the running binary).
fn file_md5(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut md5 = slowlink_receipt::Md5::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        md5.update(&buf[..n]);
    }
    Ok(md5.finish())
}

/// Every `HIPFIRE_*` variable of the process (the configuration a pairing
/// compares between arms).
fn hipfire_env() -> serde_json::Map<String, serde_json::Value> {
    let mut env = std::collections::BTreeMap::new();
    for (key, value) in std::env::vars() {
        if key.starts_with("HIPFIRE_") {
            env.insert(key, serde_json::Value::String(value));
        }
    }
    env.into_iter().collect()
}

/// The frozen identity (C2 `moe_iu4.descriptor`) of the device's aggregate
/// symmetric IU4 MoE kernel family, or null when the arch has none.
fn moe_descriptor_json(gpu: &Gpu) -> serde_json::Value {
    use rdna_compute::gemm::MoeSymIu4DownProducer;
    let Some(kernel) = gpu.qwen4_moe_sym_kernel() else {
        return serde_json::Value::Null;
    };
    let down_producer = match kernel.down_producer {
        MoeSymIu4DownProducer::Hipcc { module } => module,
        MoeSymIu4DownProducer::Embedded { symbol, .. } => symbol,
    };
    serde_json::json!({
        "arch": kernel.arch,
        "module": kernel.module,
        "gate_up": kernel.entries[0][0],
        "down": kernel.entries[0][1],
        "entries": kernel.entries,
        "nt": kernel.tiles_per_cta[0],
        "threads": kernel.block_threads,
        "gate_producer": kernel.gate_producer.module,
        "down_producer": down_producer,
        "checker": kernel.checker.symbol,
        "c2": kernel.gate_producer.require_c2,
    })
}

/// Rows of each forward `forward_chunk` runs for a `tokens`-row sequence:
/// tiles of at most the forward's prefill chunk (`max_chunk`, 0 = no forward
/// attached, so none).
fn prefill_forward_rows(tokens: usize, max_chunk: usize) -> Vec<usize> {
    if max_chunk == 0 {
        return Vec::new();
    }
    let mut rows = Vec::new();
    let mut left = tokens;
    while left > 0 {
        let tile = left.min(max_chunk);
        rows.push(tile);
        left -= tile;
    }
    rows
}

fn engine_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Header facts that depend on the teacher source, not on the scored data.
struct RefMeta {
    arch_id: u32,
    n_ctx: usize,
    vocab: usize,
    window: WindowSpec,
    teacher: ArtifactId,
    corpus: ArtifactId,
    arch: String,
}

/// Top-k blocks plus summed target NLL for one chunk. `rows` holds only the
/// chunk's scored rows, `window.score_from..window.score_to`.
fn score_teacher_chunk(
    rows: &[f32],
    chunk: &[u32],
    meta: &RefMeta,
    top_k: usize,
    blocks: &mut Vec<TopKBlock>,
) -> Result<f64, String> {
    let mut nll = 0.0f64;
    for (row, pos) in rows
        .chunks_exact(meta.vocab)
        .zip(meta.window.score_from..meta.window.score_to)
    {
        blocks.push(topk_block(row, top_k).map_err(|e| e.to_string())?);
        nll += nll_of(row, chunk[pos + 1] as usize).map_err(|e| e.to_string())?;
    }
    Ok(nll)
}

/// Gate the teacher on its own PPL, then write the HFKLDR v3 reference.
fn write_ref(
    output: &Path,
    meta: RefMeta,
    top_k: usize,
    tokens: &[u32],
    blocks: &[TopKBlock],
    nll_sum: f64,
) -> Result<(), String> {
    let mean_nll = nll_sum / blocks.len() as f64;
    let oracle = OracleStats {
        mean_nll,
        ppl: mean_nll.exp(),
        n_scored: blocks.len(),
    };
    if let TeacherVerdict::Implausible { reason, .. } =
        evaluate(&TeacherGate::for_vocab(meta.vocab), oracle)
    {
        return Err(format!("teacher implausible: {reason}"));
    }
    let header = RefHeader {
        version: 3,
        arch_id: meta.arch_id,
        n_ctx: meta.n_ctx,
        n_chunk: tokens.len() / meta.n_ctx,
        n_vocab: meta.vocab,
        estimator: Estimator::TopK {
            k: top_k as u32,
            bias_vs_full: None,
        },
        window: meta.window,
        teacher: meta.teacher,
        corpus: meta.corpus,
        oracle,
        engine_commit: engine_commit(),
        arch: meta.arch,
    };
    let mut writer = RefWriter::create(output, header).map_err(|e| e.to_string())?;
    writer.push_tokens(tokens).map_err(|e| e.to_string())?;
    for block in blocks {
        writer
            .push_block(block.residual_logprob, &block.top)
            .map_err(|e| e.to_string())?;
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!(
        "qwen4_kld: wrote {}  teacher PPL {:.4} (mean NLL {mean_nll:.6}, {} tokens)",
        output.display(),
        oracle.ppl,
        oracle.n_scored
    );
    Ok(())
}

fn build_ref(
    model: &Path,
    slice: &Path,
    output: &Path,
    n_ctx: usize,
    top_k: usize,
    max_chunks: Option<usize>,
    pin: &HotsetPin,
) -> Result<(), String> {
    let mut m = Model::load(model, n_ctx, n_ctx, None, pin)?;
    let text =
        std::fs::read_to_string(slice).map_err(|e| format!("read {}: {e}", slice.display()))?;
    // Qwen tokenizers add no BOS: chunks are plain n_ctx windows of the stream.
    let stream = m.tokenizer.encode(&text);
    let n_chunk = (stream.len() / n_ctx).min(max_chunks.unwrap_or(usize::MAX));
    if n_chunk == 0 {
        return Err(format!(
            "slice has {} tokens, fewer than n_ctx {n_ctx}",
            stream.len()
        ));
    }
    let tokens = &stream[..n_chunk * n_ctx];
    let window = WindowSpec::legacy_half(n_ctx);
    eprintln!(
        "qwen4_kld ref: {n_chunk} chunks x {n_ctx}, scoring [{}, {})",
        window.score_from, window.score_to
    );
    let meta = RefMeta {
        arch_id: m.arch_id,
        n_ctx,
        vocab: m.vocab,
        window,
        teacher: artifact_id(model)?,
        corpus: artifact_id(slice)?,
        arch: m.gpu.arch.clone(),
    };

    // Blocks are small (n_chunk * scored * (8 + 8k) bytes), so buffer them and
    // write once the oracle statistics the header carries are known.
    let started = Instant::now();
    let mut blocks = Vec::with_capacity(n_chunk * window.scored_per_chunk());
    let mut nll_sum = 0.0f64;
    for (c, chunk) in tokens.chunks_exact(n_ctx).enumerate() {
        let rows = m.chunk_logits(chunk, Prefill::Whole)?;
        let scored = &rows[window.score_from * m.vocab..window.score_to * m.vocab];
        nll_sum += score_teacher_chunk(scored, chunk, &meta, top_k, &mut blocks)?;
        eprintln!(
            "  chunk {}/{n_chunk}  running PPL {:.4}  {:.0}s",
            c + 1,
            (nll_sum / blocks.len() as f64).exp(),
            started.elapsed().as_secs_f64()
        );
    }
    write_ref(output, meta, top_k, tokens, &blocks, nll_sum)
}

/// Build a reference from externally computed teacher logits (the source
/// checkpoint teacher, `reference_oracle/kld_teacher.py`): F32
/// `[chunk][scored position][vocab]` over the token stream of `template`.
/// `teacher` is the file that identifies the teacher weights.
fn import_ref(
    logits: &Path,
    template: &Path,
    teacher: &Path,
    output: &Path,
    top_k: usize,
) -> Result<(), String> {
    let reference = KldRef::open(template).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(template).map_err(|e| e.to_string())?;
    // SAFETY: read-only maps of files this process does not modify.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let lfile =
        std::fs::File::open(logits).map_err(|e| format!("open {}: {e}", logits.display()))?;
    let lmap = unsafe { memmap2::Mmap::map(&lfile) }.map_err(|e| e.to_string())?;
    let h = &reference.header;
    let chunk_rows = h.window.scored_per_chunk() * h.n_vocab;
    // SAFETY: any bit pattern is a valid f32; the map is page-aligned and the
    // file is little-endian F32, which is this host's layout.
    let (head, rows, tail) = unsafe { lmap.align_to::<f32>() };
    if !head.is_empty() || !tail.is_empty() || rows.is_empty() || rows.len() % chunk_rows != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of {}-row x {} chunks",
            logits.display(),
            lmap.len(),
            h.window.scored_per_chunk(),
            h.n_vocab
        ));
    }
    let n_chunk = rows.len() / chunk_rows;
    if n_chunk > h.n_chunk {
        return Err(format!(
            "{n_chunk} logit chunks but template has {}",
            h.n_chunk
        ));
    }
    let tokens = &reference.tokens(&mmap).map_err(|e| e.to_string())?[..n_chunk * h.n_ctx];
    let meta = RefMeta {
        arch_id: h.arch_id,
        n_ctx: h.n_ctx,
        vocab: h.n_vocab,
        window: h.window,
        teacher: artifact_id(teacher)?,
        corpus: h.corpus.clone(),
        arch: "pytorch-source".into(),
    };
    let mut blocks = Vec::with_capacity(n_chunk * h.window.scored_per_chunk());
    let mut nll_sum = 0.0f64;
    for (chunk_logits, chunk) in rows
        .chunks_exact(chunk_rows)
        .zip(tokens.chunks_exact(h.n_ctx))
    {
        nll_sum += score_teacher_chunk(chunk_logits, chunk, &meta, top_k, &mut blocks)?;
    }
    write_ref(output, meta, top_k, tokens, &blocks, nll_sum)
}

/// `OUT` with `.a<rows>` inserted before its extension (`out.kldseq` ->
/// `out.a145.kldseq`).
fn append_output(output: &Path, rows: usize) -> PathBuf {
    let stem = output
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name = match output.extension() {
        Some(ext) => format!("{stem}.a{rows}.{}", ext.to_string_lossy()),
        None => format!("{stem}.a{rows}"),
    };
    output.with_file_name(name)
}

fn score(
    model: &Path,
    ref_path: &Path,
    output: &Path,
    max_chunks: Option<usize>,
    decode: bool,
    append_rows: &[usize],
    tail_split: Option<usize>,
    score_tail: Option<usize>,
    ple_hot: Option<u64>,
    pin: &HotsetPin,
) -> Result<(), String> {
    let reference = KldRef::open(ref_path).map_err(|e| e.to_string())?;
    let file = std::fs::File::open(ref_path).map_err(|e| e.to_string())?;
    // SAFETY: read-only map of a file this process does not modify.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let h = &reference.header;
    let (n_ctx, window) = (h.n_ctx, h.window);
    let n_chunk = h.n_chunk.min(max_chunks.unwrap_or(usize::MAX));
    let tokens = reference.tokens(&mmap).map_err(|e| e.to_string())?;
    if tail_split.is_some_and(|k| k >= n_ctx) {
        return Err(format!(
            "--tail-split must be below the window length {n_ctx}"
        ));
    }
    let mut m = Model::load(model, n_ctx, n_ctx, ple_hot, pin)?;
    if m.vocab != h.n_vocab {
        return Err(format!(
            "candidate vocab {} != reference vocab {}",
            m.vocab, h.n_vocab
        ));
    }
    let passes: Vec<(Option<usize>, PathBuf)> = if append_rows.is_empty() {
        vec![(None, output.to_path_buf())]
    } else {
        append_rows
            .iter()
            .map(|&r| (Some(r), append_output(output, r)))
            .collect()
    };
    for (append, output) in passes {
        let prefill = match (append, tail_split) {
            (_, Some(k)) => Prefill::TailSplit(k),
            (Some(r), None) => Prefill::Append(r),
            (None, None) => Prefill::Whole,
        };
        let route = match prefill {
            _ if decode => "decode".to_string(),
            Prefill::Append(r) => format!("append <= {r} rows"),
            Prefill::TailSplit(k) => format!("tail-split at {k} rows"),
            Prefill::Whole => "all-row".to_string(),
        };
        eprintln!(
            "qwen4_kld eval: {n_chunk} chunks x {n_ctx} ({route} route); teacher PPL {:.4} ({})",
            h.oracle.ppl, h.teacher.path
        );

        let started = Instant::now();
        let mut chunks = Vec::with_capacity(n_chunk);
        let (mut kld_sum, mut nll_sum, mut top1_hits, mut scored) =
            (0.0f64, 0.0f64, 0usize, 0usize);
        // Scored rows inside the last `--score-tail` rows of the window.
        let tail_from = score_tail.map_or(usize::MAX, |k| n_ctx.saturating_sub(k));
        let (mut tail_kld_sum, mut tail_top1_hits, mut tail_scored) = (0.0f64, 0usize, 0usize);
        // Digest of every scored logits row: equal digests mean bit-identical numerics.
        let mut digest = Sha256::new();
        let (from, to) = (window.score_from, window.score_to);
        for c in 0..n_chunk {
            let chunk = &tokens[c * n_ctx..(c + 1) * n_ctx];
            let rows = if decode {
                m.decode_rows(chunk, from, to)?
            } else {
                let mut rows = m.chunk_logits(chunk, prefill)?;
                rows.truncate(to * m.vocab);
                rows.drain(..from * m.vocab);
                rows
            };
            for v in &rows {
                digest.update(v.to_le_bytes());
            }
            let mut klds = Vec::with_capacity(window.scored_per_chunk());
            let mut chunk_nll = 0.0f64;
            for (i, pos) in (from..to).enumerate() {
                let row = &rows[i * m.vocab..(i + 1) * m.vocab];
                let block = reference.block(&mmap, c, i).map_err(|e| e.to_string())?;
                let block = TopKBlock {
                    residual_logprob: block.residual_logprob,
                    top: block.top.to_vec(),
                };
                klds.push(kld_from_block(&block, row).map_err(|e| e.to_string())?);
                chunk_nll += nll_of(row, chunk[pos + 1] as usize).map_err(|e| e.to_string())?;
                let argmax = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32);
                let hit = usize::from(argmax == Some(block.top[0].0));
                top1_hits += hit;
                if pos >= tail_from {
                    tail_kld_sum += klds[i];
                    tail_top1_hits += hit;
                    tail_scored += 1;
                }
            }
            let n = klds.len();
            kld_sum += klds.iter().sum::<f64>();
            nll_sum += chunk_nll;
            scored += n;
            let mean_kld = klds.iter().sum::<f64>() / n as f64;
            klds.sort_by(f64::total_cmp);
            let p99_kld = klds[((n as f64 * 0.99).ceil() as usize).clamp(1, n) - 1];
            chunks.push(ChunkScore {
                mean_kld,
                p99_kld,
                mean_nll: chunk_nll / n as f64,
            });
            eprintln!(
                "  chunk {}/{n_chunk}  KLD {mean_kld:.6}  running KLD {:.6}  {:.0}s",
                c + 1,
                kld_sum / scored as f64,
                started.elapsed().as_secs_f64()
            );
        }
        kldseq::write(&output, &chunks).map_err(|e| e.to_string())?;
        if let Some(k) = score_tail {
            let tail = serde_json::json!({
                "rows": tail_scored,
                "mean_kld": tail_kld_sum / tail_scored as f64,
                "top1": tail_top1_hits as f64 / tail_scored as f64,
            });
            eprintln!(
                "qwen4_kld eval: tail {k} rows: mean KLD = {:.6}  top1 = {:.4}  ({tail_scored} tokens)",
                tail_kld_sum / tail_scored as f64,
                tail_top1_hits as f64 / tail_scored as f64
            );
            let receipt = serde_json::json!({
                "route": route,
                "n_chunk": n_chunk, "n_ctx": n_ctx,
                "mean_kld": kld_sum / scored as f64,
                "top1": top1_hits as f64 / scored as f64,
                "rows": scored,
                "tail_rows_requested": k,
                "tail": tail
            });
            std::fs::write(
                format!("{}.json", output.display()),
                serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())? + "\n",
            )
            .map_err(|e| e.to_string())?;
        }
        let mean_nll = nll_sum / scored as f64;
        eprintln!(
            "qwen4_kld eval: mean KLD = {:.6}  mean NLL = {mean_nll:.6}  PPL = {:.4}  top1 = {:.4}  ({scored} tokens)  logits sha256 {:x} -> {}",
            kld_sum / scored as f64,
            mean_nll.exp(),
            top1_hits as f64 / scored as f64,
            digest.finalize(),
            output.display()
        );
    }
    if let Some(store) = m.bundle.ple_hot_store() {
        eprintln!(
            "qwen4_kld eval: ple hot store served {} rows ({} pages, {} pack reads)",
            store.hot_rows(),
            store.hot_pages(),
            store.hot_reads()
        );
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct S11Header {
    schema: String,
    n_rows: usize,
    n_vocab: usize,
    k: usize,
    corpus_sha256: String,
}

struct S11Ref<'a> {
    header: S11Header,
    index: &'a [u8],
    blocks: &'a [u8],
}

impl<'a> S11Ref<'a> {
    fn parse(data: &'a [u8]) -> Result<Self, String> {
        if data.get(..8) != Some(b"S11REF\0\x01") || data.len() < 16 {
            return Err("not an S11REF v1 file".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(data[8..16].try_into().unwrap()))
            .map_err(|_| "S11REF header length overflow")?;
        let index_start = 16usize
            .checked_add(header_len)
            .ok_or("S11REF length overflow")?;
        let header: S11Header =
            serde_json::from_slice(data.get(16..index_start).ok_or("truncated S11REF header")?)
                .map_err(|e| format!("S11REF header: {e}"))?;
        if header.schema != "subq4.calib2.s11ref.v1"
            || header.n_rows == 0
            || header.k == 0
            || header.k > header.n_vocab
        {
            return Err("invalid S11REF schema or geometry".into());
        }
        let block_bytes = header
            .k
            .checked_mul(8)
            .and_then(|n| n.checked_add(8))
            .ok_or("S11REF block length overflow")?;
        let index_bytes = header
            .n_rows
            .checked_mul(12)
            .ok_or("S11REF index overflow")?;
        let blocks_start = index_start
            .checked_add(index_bytes)
            .ok_or("S11REF index overflow")?;
        let end = header
            .n_rows
            .checked_mul(block_bytes)
            .and_then(|n| blocks_start.checked_add(n))
            .ok_or("S11REF length overflow")?;
        if end != data.len() {
            return Err("S11REF size mismatch".into());
        }
        Ok(Self {
            header,
            index: &data[index_start..blocks_start],
            blocks: &data[blocks_start..],
        })
    }

    fn index(&self, row: usize) -> [u32; 3] {
        let bytes = &self.index[row * 12..(row + 1) * 12];
        std::array::from_fn(|i| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
    }

    fn block(&self, row: usize, block: &mut TopKBlock) -> Result<(), String> {
        let stride = 8 + 8 * self.header.k;
        let bytes = &self.blocks[row * stride..(row + 1) * stride];
        block.residual_logprob = f64::from_le_bytes(bytes[..8].try_into().unwrap());
        if block.residual_logprob.is_nan() || block.residual_logprob > 0.0 {
            return Err(format!("S11REF row {row}: invalid residual"));
        }
        block.top.clear();
        for pair in bytes[8..].chunks_exact(8) {
            let id = u32::from_le_bytes(pair[..4].try_into().unwrap());
            let lp = f32::from_le_bytes(pair[4..].try_into().unwrap());
            if id as usize >= self.header.n_vocab || lp.is_nan() || lp > 0.0 {
                return Err(format!("S11REF row {row}: invalid top-k entry"));
            }
            block.top.push((id, lp));
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct S11Corpus {
    schema: String,
    sequences: Vec<S11Sequence>,
}

#[derive(serde::Deserialize)]
struct S11Sequence {
    id: String,
    tokens: Vec<u32>,
    score_positions: Vec<usize>,
    reset: Option<bool>,
    stratum: String,
    role_mask: Option<Vec<u8>>,
}

fn validate_s11_corpus(reference: &S11Ref<'_>, corpus: &S11Corpus) -> Result<usize, String> {
    if corpus.schema != "hipfire.qwen4.kl_corpus.v1" || corpus.sequences.is_empty() {
        return Err("invalid S11 corpus schema or empty corpus".into());
    }
    let mut row = 0;
    let mut n_ctx = 0;
    for (seq, s) in corpus.sequences.iter().enumerate() {
        if s.tokens.len() < 2
            || s.reset == Some(false)
            || s.tokens
                .iter()
                .any(|&t| t as usize >= reference.header.n_vocab)
        {
            return Err(format!(
                "corpus sequence {}: invalid tokens or non-cold reset",
                s.id
            ));
        }
        n_ctx = n_ctx.max(s.tokens.len());
        let mut prior = None;
        for &p in &s.score_positions {
            if p >= s.tokens.len() - 1 || prior.is_some_and(|v| p <= v) {
                return Err(format!(
                    "corpus sequence {}: invalid score position {p}",
                    s.id
                ));
            }
            if s.stratum == "assistant"
                && !s
                    .role_mask
                    .as_ref()
                    .is_some_and(|mask| mask.len() == s.tokens.len() && mask[p + 1] == 1)
            {
                return Err(format!(
                    "corpus sequence {}: scored target is not assistant content",
                    s.id
                ));
            }
            if row >= reference.header.n_rows
                || reference.index(row) != [seq as u32, p as u32, s.tokens[p + 1]]
            {
                return Err(format!(
                    "corpus differs from S11REF score index at row {row}"
                ));
            }
            prior = Some(p);
            row += 1;
        }
    }
    if row != reference.header.n_rows {
        return Err(format!(
            "corpus has {row} rows, reference has {}",
            reference.header.n_rows
        ));
    }
    Ok(n_ctx)
}

fn write_kl_npy(path: &Path, rows: &[f64]) -> Result<(), String> {
    use std::io::Write;
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        rows.len()
    );
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    file.write_all(b"\x93NUMPY\x01\x00")
        .map_err(|e| e.to_string())?;
    file.write_all(&(header.len() as u16).to_le_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(header.as_bytes())
        .map_err(|e| e.to_string())?;
    for value in rows {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    file.flush().map_err(|e| e.to_string())
}

/// Parses `START:END`: zero-based corpus sequence indices, half-open
/// `[START, END)`. Bounds against the corpus are checked by `partition_rows`.
fn parse_sequence_range(text: &str) -> Result<std::ops::Range<usize>, String> {
    let bad = || {
        format!("--sequence-range {text:?}: expected START:END with START < END (zero-based, END exclusive)")
    };
    let (start, end) = text.split_once(':').ok_or_else(bad)?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(start) || !digits(end) {
        return Err(bad());
    }
    let (start, end) = (
        start.parse::<usize>().map_err(|_| bad())?,
        end.parse::<usize>().map_err(|_| bad())?,
    );
    if start >= end {
        return Err(bad());
    }
    Ok(start..end)
}

/// Canonical (full S11REF order) score-row range of the corpus sequences in
/// `range`. Rows are laid out sequence-major, so a contiguous sequence range
/// is always a contiguous row range. Refuses an out-of-bounds range and one
/// that selects no score rows; never pads or invents rows.
fn partition_rows(
    sequences: &[S11Sequence],
    range: &std::ops::Range<usize>,
) -> Result<std::ops::Range<usize>, String> {
    if range.start >= range.end || range.end > sequences.len() {
        return Err(format!(
            "--sequence-range {}:{} out of bounds for {} corpus sequences",
            range.start,
            range.end,
            sequences.len()
        ));
    }
    let rows_of =
        |seqs: &[S11Sequence]| seqs.iter().map(|s| s.score_positions.len()).sum::<usize>();
    let start = rows_of(&sequences[..range.start]);
    let rows = rows_of(&sequences[range.clone()]);
    if rows == 0 {
        return Err(format!(
            "--sequence-range {}:{} selects no score rows",
            range.start, range.end
        ));
    }
    Ok(start..start + rows)
}

/// Scores the S11 corpus. Full mode scores every sequence and writes the full
/// `hipfire.qwen4.s11ref_klrows.v1` receipt.
///
/// `--sequence-range START:END` (a partition) runs the identical loop body over
/// the borrowed slice `corpus.sequences[START..END]` and writes only the rows of
/// those sequences (canonical rows `row_start..row_end`; output allocated for
/// exactly those rows) with the distinct
/// `hipfire.qwen4.s11ref_klrows.partition.v1` receipt, never a full-gate PASS.
/// Partition equivalence: the loop body is a function of `(sequence s, reference
/// rows of s)` only. Each iteration begins with `bundle.reset` (decode route
/// directly; cold-prefill via `chunk_logits`) and feeds only `s.tokens`, so no
/// state, logits or KL of an earlier sequence is read; the logits buffer is
/// fully rewritten by this sequence's forward before it is downloaded. Model
/// load uses the full corpus `n_ctx`, so allocations match the full run. The
/// reference block index is `row_start + rows_emitted_so_far`, the same
/// canonical row the full loop reaches at that sequence. Hence each partition's
/// rows equal the corresponding slice of the full loop's rows, and fresh-process
/// partitions that tile `0..total_sequences` concatenate to the full row array.
fn score_s11ref(
    model: &Path,
    ref_path: &Path,
    corpus_path: &Path,
    output: &Path,
    decode: bool,
    ple_hot: Option<u64>,
    smoke_seqs: Option<usize>,
    sequence_range: Option<std::ops::Range<usize>>,
    pin: &HotsetPin,
) -> Result<(), String> {
    let file = std::fs::File::open(ref_path).map_err(|e| e.to_string())?;
    // SAFETY: a read-only map of the immutable reference; parsing uses LE bytes.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let reference = S11Ref::parse(&mmap)?;
    let corpus_bytes = std::fs::read(corpus_path).map_err(|e| e.to_string())?;
    let corpus_sha = format!("{:x}", Sha256::digest(&corpus_bytes));
    let corpus_md5 = slowlink_receipt::md5_hex(&corpus_bytes);
    let ref_md5 = slowlink_receipt::md5_hex(&mmap);
    if corpus_sha != reference.header.corpus_sha256 {
        return Err("corpus SHA256 differs from S11REF header".into());
    }
    let corpus: S11Corpus = serde_json::from_slice(&corpus_bytes).map_err(|e| e.to_string())?;
    let n_ctx = validate_s11_corpus(&reference, &corpus)?;
    let total_sequences = corpus.sequences.len();
    // Selected sequences and their canonical rows; validated before the GPU load.
    let (seqs, rows) = match (&sequence_range, smoke_seqs) {
        (Some(range), _) => (range.clone(), partition_rows(&corpus.sequences, range)?),
        (None, Some(n)) => (0..n.min(total_sequences), 0..reference.header.n_rows),
        (None, None) => (0..total_sequences, 0..reference.header.n_rows),
    };
    let mut m = Model::load(model, n_ctx, n_ctx, ple_hot, pin)?;
    if m.vocab != reference.header.n_vocab {
        return Err(format!(
            "candidate vocab {} != S11REF vocab {}",
            m.vocab, reference.header.n_vocab
        ));
    }
    let mut kls = Vec::with_capacity(rows.len());
    let mut block = TopKBlock {
        residual_logprob: 0.0,
        top: Vec::with_capacity(reference.header.k),
    };
    let started = Instant::now();
    let selected = &corpus.sequences[seqs.clone()];
    let mut prefix_routes: Vec<serde_json::Value> = Vec::new();
    let mut sequence_routes: Vec<serde_json::Value> = Vec::new();
    let mut censuses: Vec<slowlink_receipt::RouteCensus> = Vec::new();
    for s in selected {
        if decode {
            m.bundle.reset(&mut m.gpu).map_err(|e| e.to_string())?;
            let row = m.logits.sub_offset(0, m.vocab);
            let from = s.score_positions.first().copied().unwrap_or(s.tokens.len());
            if from > 0 {
                m.bundle
                    .forward_chunk_final(&mut m.gpu, &s.tokens[..from], &row, None)
                    .map_err(|e| e.to_string())?;
                // Whether this unscored prefix ran the S1 layer-major route.
                prefix_routes.push(serde_json::json!({
                    "id": s.id,
                    "prefix_rows": from,
                    "slowlink": m.bundle.slowlink_receipt(),
                }));
            }
            let to = s.score_positions.last().map_or(from, |p| p + 1);
            let mut positions = s.score_positions.iter().copied().peekable();
            for (p, &token) in s.tokens.iter().enumerate().take(to).skip(from) {
                m.bundle
                    .forward_token(&mut m.gpu, token, &row, None)
                    .map_err(|e| e.to_string())?;
                if positions.peek() == Some(&p) {
                    let logits = m.gpu.download_f32(&row).map_err(|e| e.to_string())?;
                    if logits.iter().any(|v| !v.is_finite()) {
                        return Err("non-finite decode logits".into());
                    }
                    reference.block(rows.start + kls.len(), &mut block)?;
                    kls.push(kld_from_block(&block, &logits).map_err(|e| e.to_string())?);
                    positions.next();
                }
            }
        } else {
            let max_chunk = m.bundle.spec_chunk_rows().unwrap_or(0);
            let forward_rows = prefill_forward_rows(s.tokens.len(), max_chunk);
            let census_before = m.gpu.moe_route_launch_census();
            let logits = m.chunk_logits(&s.tokens, Prefill::Whole)?;
            let delta = m.gpu.moe_route_launch_census().since(census_before);
            let census = slowlink_receipt::RouteCensus {
                sym_iu4_gate_up: delta.sym_iu4_gate_up,
                sym_iu4_down: delta.sym_iu4_down,
                a4_gate_producer: delta.a4_gate_producer,
                a4_down_producer: delta.a4_down_producer,
                f16_wmma_gate_up: delta.f16_wmma_gate_up,
                f16_wmma_down: delta.f16_wmma_down,
            };
            censuses.push(census);
            sequence_routes.push(serde_json::json!({
                "id": s.id,
                "tokens": s.tokens.len(),
                "prefill_chunk_rows": max_chunk,
                "forwards": forward_rows.iter().map(|&rows| serde_json::json!({
                    "rows": rows,
                    "route_rows": m.gpu.route_rows(rows),
                    "sym_iu4_applies": m.gpu.qwen4_moe_sym_iu4_applies(rows),
                })).collect::<Vec<_>>(),
                "census": census.to_value(),
            }));
            for &p in &s.score_positions {
                reference.block(rows.start + kls.len(), &mut block)?;
                kls.push(
                    kld_from_block(&block, &logits[p * m.vocab..(p + 1) * m.vocab])
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        let seq_rows = s.score_positions.len();
        let seq_mean = kls[kls.len() - seq_rows..].iter().sum::<f64>() / seq_rows.max(1) as f64;
        eprintln!(
            "  {}: {} rows scored, seq mean KLD {:.6}, {:.0}s",
            s.id,
            kls.len(),
            seq_mean,
            started.elapsed().as_secs_f64()
        );
    }
    if smoke_seqs.is_some() {
        if kls.iter().any(|v| !v.is_finite()) {
            return Err("non-finite KL rows".into());
        }
        // The prefix rows are kept (never a receipt) so two smoke runs compare byte for byte.
        write_kl_npy(output, &kls)?;
        eprintln!("qwen4_kld s11ref SMOKE (first {} sequences, not a gate, no receipt): mean KLD = {:.12}  ({} rows)",
            smoke_seqs.unwrap_or(0), kls.iter().sum::<f64>() / kls.len().max(1) as f64, kls.len());
        return Ok(());
    }
    if kls.len() != rows.len() || kls.iter().any(|v| !v.is_finite()) {
        return Err("invalid or incomplete KL rows".into());
    }
    write_kl_npy(output, &kls)?;
    let binary = std::env::current_exe().map_err(|e| e.to_string())?;
    let forward_tokens = selected
        .iter()
        .map(|s| {
            if decode {
                s.score_positions.last().map_or(s.tokens.len(), |p| p + 1)
            } else {
                s.tokens.len()
            }
        })
        .sum::<usize>();
    let route = if decode {
        "decode"
    } else {
        slowlink_receipt::KLD_PREFILL_ROUTE
    };
    let mut receipt = if sequence_range.is_some() {
        // A partition is never the full gate: distinct schema, explicit selection.
        serde_json::json!({
            "schema": "hipfire.qwen4.s11ref_klrows.partition.v1",
            "partition": true,
            "full_gate_pass": false,
            "direction": "KL(ref || hipfire)",
            "estimator": "top-k + exact reference float64 residual; saddle_quant::kld_from_block",
            "model_sha256": artifact_id(model)?.sha256,
            "ref_sha256": artifact_id(ref_path)?.sha256,
            "corpus_sha256": corpus_sha,
            "binary_sha256": artifact_id(&binary)?.sha256,
            "output_sha256": artifact_id(output)?.sha256,
            "sequence_start": seqs.start, "sequence_end": seqs.end,
            "total_sequences": total_sequences,
            "selected_sequence_ids": selected.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            "full_rows": reference.header.n_rows,
            "row_start": rows.start, "row_end": rows.end,
            "rows": kls.len(), "n_ctx": n_ctx,
            "sequences": selected.len(),
            "forward_tokens": forward_tokens,
            "route": route,
            "slowlink_prefix_routes": prefix_routes
        })
    } else {
        serde_json::json!({
            "schema": "hipfire.qwen4.s11ref_klrows.v1",
            "direction": "KL(ref || hipfire)",
            "estimator": "top-k + exact reference float64 residual; saddle_quant::kld_from_block",
            "model_sha256": artifact_id(model)?.sha256,
            "ref_sha256": artifact_id(ref_path)?.sha256,
            "corpus_sha256": corpus_sha,
            "binary_sha256": artifact_id(&binary)?.sha256,
            "output_sha256": artifact_id(output)?.sha256,
            "rows": kls.len(), "n_ctx": n_ctx,
            "sequences": total_sequences,
            "forward_tokens": forward_tokens,
            "route": route,
            "slowlink_prefix_routes": prefix_routes
        })
    };
    // Identity and route evidence shared by every route; the prefill route
    // adds the per-sequence forwards/census and the receipt-level verdict.
    let moe_route = slowlink_receipt::moe_route_verdict(&censuses);
    let mut extra = serde_json::Map::new();
    extra.insert("model_path".into(), model.display().to_string().into());
    extra.insert("ref_path".into(), ref_path.display().to_string().into());
    extra.insert("corpus_path".into(), corpus_path.display().to_string().into());
    extra.insert("binary_path".into(), binary.display().to_string().into());
    extra.insert("binary_md5".into(), file_md5(&binary)?.into());
    extra.insert("ref_md5".into(), ref_md5.into());
    extra.insert("corpus_md5".into(), corpus_md5.into());
    extra.insert("env".into(), hipfire_env().into());
    extra.insert(
        "moe_iu4".into(),
        serde_json::json!({
            "env": std::env::var(slowlink_receipt::MOE_IU4_ENV).ok(),
            "requested": m.gpu.qwen4_moe_sym_iu4_requested(),
            "arch": m.gpu.arch,
            "descriptor": moe_descriptor_json(&m.gpu),
            "route_rows_floor": slowlink_receipt::KLD_PREFILL_MIN_ROWS,
        }),
    );
    if !decode {
        extra.insert("moe_route".into(), moe_route.into());
        // The C1 census summed over every scored forward.
        let total = censuses.iter().fold(slowlink_receipt::RouteCensus::default(), |a, c| {
            slowlink_receipt::RouteCensus {
                sym_iu4_gate_up: a.sym_iu4_gate_up + c.sym_iu4_gate_up,
                sym_iu4_down: a.sym_iu4_down + c.sym_iu4_down,
                a4_gate_producer: a.a4_gate_producer + c.a4_gate_producer,
                a4_down_producer: a.a4_down_producer + c.a4_down_producer,
                f16_wmma_gate_up: a.f16_wmma_gate_up + c.f16_wmma_gate_up,
                f16_wmma_down: a.f16_wmma_down + c.f16_wmma_down,
            }
        });
        extra.insert("moe_launch_census".into(), total.to_value());
        extra.insert("sequence_routes".into(), sequence_routes.into());
    }
    receipt
        .as_object_mut()
        .ok_or("s11ref receipt is not an object")?
        .extend(extra);
    std::fs::write(
        format!("{}.json", output.display()),
        serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "qwen4_kld s11ref{} ({route}{}): mean KLD = {:.12}  ({} rows)",
        if sequence_range.is_some() {
            " PARTITION (not a full gate)"
        } else {
            ""
        },
        if decode {
            String::new()
        } else {
            format!(", moe_route {moe_route}")
        },
        kls.iter().sum::<f64>() / kls.len() as f64,
        kls.len()
    );
    Ok(())
}

/// `speed`: TTFT of one chunked prefill of a prompt file (production
/// `forward_chunk_final`, one logits row) and C1 greedy decode tok/s
/// (`forward_token_or_argmax`, the serve decode step), `reps` fresh-state
/// repetitions in one process. Rep 0 absorbs page-cache and kernel-cache
/// warmup; the receipt keeps every rep.
fn speed(
    model: &Path,
    prompt: &Path,
    output: &Path,
    decode_tokens: usize,
    warmup: usize,
    reps: usize,
    max_prompt: Option<usize>,
    product_prompt: bool,
    pin: &HotsetPin,
) -> Result<(), String> {
    let text =
        std::fs::read_to_string(prompt).map_err(|e| format!("read {}: {e}", prompt.display()))?;
    let hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let probe =
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("tokenizer: {e}"))?;
    let mut terminator_ids = vec![probe.eos_id];
    if let Some(id) = probe.eot_id {
        terminator_ids.push(id);
    }
    let mut tokens = if product_prompt {
        // Match render_qwen4_prompt_cold's AR answer-mode branch. A template
        // render error is fatal, never silently replaced by a raw completion.
        let text = hipfire_runtime::tokenizer::maybe_normalize_prompt(&text);
        let template = hfq.chat_template();
        let jinja = hipfire_config::developer_var("HIPFIRE_JINJA_CHAT")
            .ok()
            .as_deref()
            != Some("0");
        if let Some(template) = template.as_ref().filter(|_| jinja) {
            let rendered = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer: &probe,
                template,
                system: None,
                user: &text,
                enable_thinking: false,
                bos_token: None,
                reasoning_strength: None,
                // qwen_jinja_reasoning disables thinking AND omits effort
                // for the product's answer-mode request.
                reasoning_effort: None,
            }
            .render()
            .map_err(|e| format!("qwen4 Jinja render failed: {e}"))?;
            probe.encode(&rendered)
        } else {
            hipfire_runtime::prompt_frame::ChatFrame {
                tokenizer: &probe,
                system: None,
                user: &text,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
                raw: false,
            }
            .build()
        }
    } else {
        probe.encode(&text)
    };
    drop(hfq);
    drop(probe);
    if let Some(cap) = max_prompt {
        tokens.truncate(cap);
    }
    if tokens.is_empty() {
        return Err("empty prompt".into());
    }
    let n_ctx = tokens.len() + warmup + decode_tokens + 8;
    let mut m = Model::load(model, n_ctx, 1, None, pin)?;
    terminator_ids.push(m.bundle.config.eos_token_id);
    terminator_ids.sort_unstable();
    terminator_ids.dedup();
    let row = m.logits.sub_offset(0, m.vocab);
    let mut rep_rows = Vec::new();
    for rep in 0..reps {
        m.bundle.reset(&mut m.gpu).map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        m.bundle
            .forward_chunk_final(&mut m.gpu, &tokens, &row, None)
            .map_err(|e| e.to_string())?;
        let first = m.gpu.download_f32(&row).map_err(|e| e.to_string())?;
        let ttft = t0.elapsed().as_secs_f64();
        if first.iter().any(|v| !v.is_finite()) {
            return Err("non-finite prefill logits".into());
        }
        let logits_sha = {
            let mut digest = Sha256::new();
            for v in &first {
                digest.update(v.to_bits().to_le_bytes());
            }
            format!("{:x}", digest.finalize())
        };
        let mut ids = Vec::with_capacity(warmup + decode_tokens);
        let mut t1 = Instant::now();
        for i in 0..warmup + decode_tokens {
            if i == warmup {
                t1 = Instant::now();
            }
            let id = m
                .bundle
                .forward_token_or_argmax(&mut m.gpu, None, &row)
                .map_err(|e| e.to_string())?;
            ids.push(id);
        }
        m.gpu.download_f32(&row).map_err(|e| e.to_string())?;
        let decode_s = t1.elapsed().as_secs_f64();
        let tps = decode_tokens as f64 / decode_s;
        eprintln!(
            "qwen4_kld speed rep {rep}: prompt {} tokens, TTFT {:.1} ms, decode {decode_tokens} tok in {:.3} s = {tps:.2} tok/s",
            tokens.len(), ttft * 1e3, decode_s
        );
        rep_rows.push(serde_json::json!({
            "rep": rep, "ttft_ms": ttft * 1e3, "decode_tok_s": tps,
            "decode_s": decode_s, "first_ids": ids.iter().take(16).collect::<Vec<_>>(),
            "generated_ids": ids,
            "prefill_logits_sha256": logits_sha,
        }));
    }
    let binary = std::env::current_exe().map_err(|e| e.to_string())?;
    let receipt = serde_json::json!({
        "schema": "hipfire.qwen4.speed.v1",
        "model": model.display().to_string(),
        "prompt": prompt.display().to_string(),
        "prompt_sha256": artifact_id(prompt)?.sha256,
        "binary_sha256": artifact_id(&binary)?.sha256,
        "prompt_tokens": tokens.len(), "decode_tokens": decode_tokens, "warmup_tokens": warmup,
        "product_prompt": product_prompt, "prompt_ids": tokens, "terminator_ids": terminator_ids,
        "expert_vram_layers": std::env::var("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS").ok(),
        "arch": m.gpu.arch.to_string(),
        "reps": rep_rows,
    });
    std::fs::write(
        output,
        serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Little-endian f64 rows of a 1-D `.npy` as `write_kl_npy` writes them.
fn read_kl_npy(path: &Path) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" || bytes[6] != 1 {
        return Err(format!("{}: not a version-1 .npy", path.display()));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = bytes
        .get(10..10 + header_len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| format!("{}: truncated .npy header", path.display()))?;
    if !header.contains("'<f8'") || !header.contains("'fortran_order': False") {
        return Err(format!(
            "{}: not a C-order little-endian f8 array",
            path.display()
        ));
    }
    let data = &bytes[10 + header_len..];
    if data.len() % 8 != 0 {
        return Err(format!(
            "{}: payload is not a whole number of f64 rows",
            path.display()
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect())
}

/// `pair`: the paired candidate-minus-base KLD verdict over two full-gate
/// s11ref runs (decode or prefill route) on the same reference and a pinned
/// gate corpus. Per-sequence mean KL first (the corpus sequence is the unit,
/// never the token), then the paired 95% t interval of candidate minus base;
/// pass when `lo <= 0`, else `hi <= +0.010`. Every row and every corpus
/// sequence is required: a partial corpus, a partition receipt, an unpinned
/// corpus, a different reference/corpus/config or a corpus whose SHA differs
/// from the receipts is refused. `expect_iu4` adds the IU4 prefill pairing
/// ([`slowlink_receipt::kld_iu4_pair_problems`]). The report is written even
/// for a failing or refused verdict (`pass` false, `problems`).
fn pair(
    base: &Path,
    cand: &Path,
    corpus_path: &Path,
    output: &Path,
    expect_iu4: bool,
) -> Result<(), String> {
    pair_in(
        base,
        cand,
        corpus_path,
        output,
        expect_iu4,
        &slowlink_receipt::KLD_CORPORA,
    )
}

fn pair_in(
    base: &Path,
    cand: &Path,
    corpus_path: &Path,
    output: &Path,
    expect_iu4: bool,
    table: &[slowlink_receipt::KldCorpus<'_>],
) -> Result<(), String> {
    let receipt_of = |npy: &Path| -> Result<serde_json::Value, String> {
        let path = format!("{}.json", npy.display());
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))
    };
    let (rb, rc) = (receipt_of(base)?, receipt_of(cand)?);
    let corpus_bytes = std::fs::read(corpus_path).map_err(|e| e.to_string())?;
    let corpus_sha = format!("{:x}", Sha256::digest(&corpus_bytes));
    let mut problems: Vec<String> = Vec::new();
    for (arm, receipt) in [("base", &rb), ("cand", &rc)] {
        for p in slowlink_receipt::kld_receipt_problems_in(receipt, table) {
            problems.push(format!("{arm}: {p}"));
        }
        if receipt["corpus_sha256"].as_str() != Some(corpus_sha.as_str()) {
            problems.push(format!(
                "{arm}: receipt corpus_sha256 differs from the supplied corpus"
            ));
        }
    }
    problems.extend(slowlink_receipt::kld_receipts_comparable(&rb, &rc));
    if expect_iu4 {
        problems.extend(slowlink_receipt::kld_iu4_pair_problems(&rb, &rc));
    }
    let identity = slowlink_receipt::kld_corpus(table, &corpus_sha);
    let corpus_name = identity.map(|c| c.name);
    let write = |report: serde_json::Value| -> Result<(), String> {
        std::fs::write(
            output,
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| format!("write {}: {e}", output.display()))
    };
    let fail = |problems: Vec<String>| -> Result<(), String> {
        write(serde_json::json!({
            "schema": "hipfire.qwen4.slowlink_kld_pair.v1", "pass": false, "problems": problems,
            "corpus": corpus_name, "moe_iu4_pair": expect_iu4,
            "base_receipt": rb, "cand_receipt": rc,
        }))?;
        Err(format!("qwen4_kld pair refused: {}", problems.join("; ")))
    };
    let Some(identity) = identity else {
        problems.push("the supplied corpus is not a pinned gate corpus".into());
        return fail(problems);
    };
    if !problems.is_empty() {
        return fail(problems);
    }
    let corpus: S11Corpus = serde_json::from_slice(&corpus_bytes).map_err(|e| e.to_string())?;
    let counts: Vec<usize> = corpus
        .sequences
        .iter()
        .map(|s| s.score_positions.len())
        .collect();
    let ids: Vec<&str> = corpus.sequences.iter().map(|s| s.id.as_str()).collect();
    for (arm, receipt) in [("base", &rb), ("cand", &rc)] {
        for p in slowlink_receipt::kld_sequence_ids_problems(receipt, &ids) {
            problems.push(format!("{arm}: {p}"));
        }
    }
    if !problems.is_empty() {
        return fail(problems);
    }
    let (base_rows, cand_rows) = (read_kl_npy(base)?, read_kl_npy(cand)?);
    let reduce = |rows: &[f64]| slowlink_receipt::sequence_means(rows, &counts);
    let (bm, cm) = match (reduce(&base_rows), reduce(&cand_rows)) {
        (Ok(b), Ok(c)) => (b, c),
        (b, c) => {
            let mut p = Vec::new();
            p.extend(b.err());
            p.extend(c.err());
            return fail(p);
        }
    };
    if base_rows.len() as u64 != identity.rows || cand_rows.len() != base_rows.len() {
        return fail(vec![format!(
            "row vectors are {} and {} rows, {} required ({})",
            base_rows.len(),
            cand_rows.len(),
            identity.rows,
            identity.name
        )]);
    }
    if bm.len() as u64 != identity.sequences {
        return fail(vec![format!(
            "the corpus has {} sequences, {} required ({})",
            bm.len(),
            identity.sequences,
            identity.name
        )]);
    }
    let ci = match slowlink_receipt::paired_ci(&bm, &cm) {
        Ok(ci) => ci,
        Err(e) => return fail(vec![e]),
    };
    let pass = slowlink_receipt::paired_verdict(&ci);
    let per_sequence: Vec<serde_json::Value> = ids
        .iter()
        .zip(bm.iter().zip(&cm))
        .map(|(id, (b, c))| serde_json::json!({"id": id, "base": b, "cand": c, "diff": c - b}))
        .collect();
    let worst = per_sequence
        .iter()
        .max_by(|a, b| {
            a["diff"]
                .as_f64()
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&b["diff"].as_f64().unwrap_or(f64::NEG_INFINITY))
        })
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let arm_summary = |receipt: &serde_json::Value| {
        serde_json::json!({
            "route": receipt.get("route"), "moe_route": receipt.get("moe_route"),
            "moe_iu4": receipt.get("moe_iu4"), "binary_md5": receipt.get("binary_md5"),
            "top1": receipt.get("top1"), "nll": receipt.get("nll"),
        })
    };
    write(serde_json::json!({
        "schema": "hipfire.qwen4.slowlink_kld_pair.v1",
        "pass": pass, "problems": Vec::<String>::new(),
        "rule": "per-sequence means, paired candidate-minus-base 95% t interval; pass lo<=0, else hi<=0.010",
        "corpus": identity.name, "route": rb.get("route"), "moe_iu4_pair": expect_iu4,
        "sequences": ci.n, "rows": base_rows.len(),
        "mean_diff": ci.mean, "se": ci.se, "lo": ci.lo, "hi": ci.hi, "hi_limit": slowlink_receipt::KLD_HI_LIMIT,
        "base_mean": bm.iter().sum::<f64>() / bm.len() as f64,
        "cand_mean": cm.iter().sum::<f64>() / cm.len() as f64,
        "worst_sequence": worst, "per_sequence": per_sequence,
        "base": arm_summary(&rb), "cand": arm_summary(&rc),
        "corpus_sha256": corpus_sha,
        "base_receipt": rb, "cand_receipt": rc,
        "slowlink_prefix_routes": {
            "base": rb.get("slowlink_prefix_routes").cloned().unwrap_or(serde_json::Value::Null),
            "cand": rc.get("slowlink_prefix_routes").cloned().unwrap_or(serde_json::Value::Null),
        },
    }))?;
    eprintln!(
        "qwen4_kld pair ({}): mean diff {:+.6}, 95% CI [{:+.6}, {:+.6}] over {} sequences -> {}",
        identity.name,
        ci.mean,
        ci.lo,
        ci.hi,
        ci.n,
        if pass { "PASS" } else { "FAIL" }
    );
    if pass {
        Ok(())
    } else {
        Err(format!(
            "paired KLD verdict failed: lo {:+.6} > 0 and hi {:+.6} > {}",
            ci.lo,
            ci.hi,
            slowlink_receipt::KLD_HI_LIMIT
        ))
    }
}

fn run_args(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mode = args.next().ok_or(USAGE)?;
    let (mut model, mut slice, mut ref_path, mut output) = (None, None, None, None);
    let (mut logits, mut teacher, mut corpus) = (None, None, None);
    let (mut n_ctx, mut top_k, mut max_chunks, mut decode) = (512usize, 256usize, None, false);
    let (mut ple_hot, mut append_rows) = (None, Vec::new());
    let mut pin = HotsetPin::default();
    let mut smoke_seqs: Option<usize> = None;
    let mut sequence_range: Option<std::ops::Range<usize>> = None;
    let (mut pair_base, mut pair_cand): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    let mut pair_iu4 = false;
    let mut prompt_file: Option<PathBuf> = None;
    let mut product_prompt = false;
    let (mut decode_tokens, mut speed_warmup, mut reps, mut max_prompt) =
        (128usize, 8usize, 2usize, None::<usize>);
    let (mut tail_split, mut score_tail) = (None, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |v: String| v.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--slice" => slice = Some(PathBuf::from(value()?)),
            "--ref" => ref_path = Some(PathBuf::from(value()?)),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--logits" => logits = Some(PathBuf::from(value()?)),
            "--teacher" => teacher = Some(PathBuf::from(value()?)),
            "--corpus" => corpus = Some(PathBuf::from(value()?)),
            "--n-ctx" => n_ctx = number(value()?)?,
            "--top-k" => top_k = number(value()?)?,
            "--max-chunks" => max_chunks = Some(number(value()?)?),
            "--decode" => decode = true,
            "--ple-hot" => ple_hot = Some(number(value()?)? as u64),
            "--hotset" => pin.sidecar = Some(PathBuf::from(value()?)),
            "--hot-budget-mib" => pin.budget_mib = Some(HotsetPin::parse_budget(&value()?)?),
            "--append-rows" => {
                for rows in value()?.split(',') {
                    match number(rows.to_string())? {
                        0 => return Err("--append-rows: rows must be positive".into()),
                        rows => append_rows.push(rows),
                    }
                }
            }
            "--smoke-seqs" => smoke_seqs = Some(number(value()?)?),
            "--sequence-range" => sequence_range = Some(parse_sequence_range(&value()?)?),
            "--base" => pair_base = Some(PathBuf::from(value()?)),
            "--moe-iu4" => pair_iu4 = true,
            "--cand" => pair_cand = Some(PathBuf::from(value()?)),
            "--prompt-file" => prompt_file = Some(PathBuf::from(value()?)),
            "--product-prompt" => product_prompt = true,
            "--decode-tokens" => decode_tokens = number(value()?)?,
            "--speed-warmup" => speed_warmup = number(value()?)?,
            "--reps" => reps = number(value()?)?,
            "--max-prompt" => max_prompt = Some(number(value()?)?),
            "--tail-split" => match number(value()?)? {
                0 => return Err("--tail-split: rows must be positive".into()),
                rows => tail_split = Some(rows),
            },
            "--score-tail" => match number(value()?)? {
                0 => return Err("--score-tail: rows must be positive".into()),
                rows => score_tail = Some(rows),
            },
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    pin.validate(&mode)?;
    if sequence_range.is_some() {
        if mode != "s11ref" {
            return Err("--sequence-range is only for s11ref".into());
        }
        if smoke_seqs.is_some() {
            return Err("--sequence-range and --smoke-seqs are mutually exclusive".into());
        }
    }
    if mode == "s11ref"
        && (max_chunks.is_some() || slice.is_some() || logits.is_some() || teacher.is_some())
    {
        return Err("s11ref requires the full corpus; --max-chunks, --slice, --logits and --teacher are not supported (--smoke-seqs scores a prefix and is never a gate)".into());
    }
    if !append_rows.is_empty() && (mode != "eval" || decode) {
        return Err("--append-rows is an eval route and excludes --decode".into());
    }
    if tail_split.is_some() && (mode != "eval" || decode || !append_rows.is_empty()) {
        return Err("--tail-split is an eval route and excludes --decode and --append-rows".into());
    }
    if score_tail.is_some() && (mode != "eval" || decode) {
        return Err("--score-tail is an eval option and excludes --decode".into());
    }
    if pair_iu4 && mode != "pair" {
        return Err("--moe-iu4 is only for pair".into());
    }
    let score_tail = score_tail.or(tail_split);
    match (mode.as_str(), model, slice, ref_path, output) {
        ("ref", Some(model), Some(slice), None, Some(output)) => {
            build_ref(&model, &slice, &output, n_ctx, top_k, max_chunks, &pin)
        }
        ("eval", Some(model), None, Some(ref_path), Some(output)) => score(
            &model,
            &ref_path,
            &output,
            max_chunks,
            decode,
            &append_rows,
            tail_split,
            score_tail,
            ple_hot,
            &pin,
        ),
        ("s11ref", Some(model), None, Some(ref_path), Some(output)) => {
            let corpus = corpus.ok_or(
                "s11ref requires --corpus CORPUS.json (prefix tokens are not stored in S11REF)",
            )?;
            score_s11ref(
                &model,
                &ref_path,
                &corpus,
                &output,
                decode,
                ple_hot,
                smoke_seqs,
                sequence_range,
                &pin,
            )
        }
        ("speed", Some(model), None, None, Some(output)) => {
            let prompt = prompt_file.ok_or("speed requires --prompt-file")?;
            speed(
                &model,
                &prompt,
                &output,
                decode_tokens,
                speed_warmup,
                reps,
                max_prompt,
                product_prompt,
                &pin,
            )
        }
        ("pair", None, None, None, Some(output)) => {
            // CPU-only: the paired candidate-minus-base verdict over two
            // full-gate s11ref receipts (decode or prefill route).
            let (Some(base), Some(cand), Some(corpus)) = (pair_base, pair_cand, corpus) else {
                return Err("pair requires --base BASE.npy --cand CAND.npy --corpus CORPUS.json --output OUT.json".into());
            };
            pair(&base, &cand, &corpus, &output, pair_iu4)
        }
        ("ple-derive", Some(model), None, None, None) => {
            // CPU-only: writes `<model>.ple-q4` (row order) from the embedded
            // Q8 rows, bound to this exact base file; refuses an existing one.
            let report = hipfire_runtime::ple_convert::derive_q4(&model, None, &mut |p| {
                if p.total_bytes > 0 && p.done_bytes % (4 << 30) < (32 << 20) {
                    eprintln!(
                        "qwen4_kld ple-derive {}: {}/{} bytes",
                        p.phase, p.done_bytes, p.total_bytes
                    );
                }
            })?;
            eprintln!(
                "qwen4_kld ple-derive: wrote {} ({} bytes), payload sha256 {}, rows sha256 {}, {:.1} s",
                report.output.display(), report.bytes, report.payload_sha256,
                report.source_rows_sha256, report.elapsed.as_secs_f64()
            );
            Ok(())
        }
        ("import", None, None, Some(template), Some(output)) => match (logits, teacher) {
            (Some(logits), Some(teacher)) => {
                import_ref(&logits, &template, &teacher, &output, top_k)
            }
            _ => Err(USAGE.into()),
        },
        _ => Err(USAGE.into()),
    }
}

fn main() {
    if let Err(error) = run_args(std::env::args().skip(1)) {
        eprintln!("qwen4_kld: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod pinned_control_tests {
    use super::*;

    #[test]
    fn pinned_cli_budget_is_checked_without_loading_a_model() {
        assert!(HotsetPin::default().validate("import").is_ok());
        for bad in [
            "",
            "0",
            "-1",
            "bad",
            "17592186044416",
            "18446744073709551616",
        ] {
            assert!(HotsetPin::parse_budget(bad).is_err(), "{bad}");
            assert!(run_args(
                ["eval", "--hot-budget-mib", bad]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err());
        }
        let pin = HotsetPin {
            budget_mib: Some(HotsetPin::parse_budget("6000").unwrap()),
            sidecar: Some(PathBuf::from("bound.expert-hotset")),
        };
        assert_eq!(pin.budget_mib, Some(6000));
        for mode in ["ref", "eval", "s11ref", "speed"] {
            assert!(pin.validate(mode).is_ok(), "{mode}");
        }
        for mode in ["import", "ple-derive", "unknown"] {
            assert!(pin.validate(mode).is_err(), "{mode}");
        }
        for flag in ["--hotset", "--hot-budget-mib"] {
            assert!(run_args(["s11ref", flag].into_iter().map(str::to_owned))
                .unwrap_err()
                .contains("needs a value"));
        }
        assert!(run_args(
            ["s11ref", "--decode", "--hotset", "bound.expert-hotset"]
                .into_iter()
                .map(str::to_owned)
        )
        .unwrap_err()
        .contains("requires a nonzero"));
        // Accepted pin args reach existing mode validation, not the GPU.
        assert!(run_args(
            [
                "s11ref",
                "--model",
                "missing.hfq",
                "--ref",
                "ref.s11ref",
                "--output",
                "out.npy",
                "--decode",
                "--hot-budget-mib",
                "6000",
                "--hotset",
                "bound.expert-hotset"
            ]
            .into_iter()
            .map(str::to_owned)
        )
        .unwrap_err()
        .contains("--corpus"));
        assert!(run_args(
            [
                "eval",
                "--decode",
                "--append-rows",
                "1",
                "--hot-budget-mib",
                "6000"
            ]
            .into_iter()
            .map(str::to_owned)
        )
        .unwrap_err()
        .contains("excludes --decode"));
    }

    #[test]
    fn pinned_policy_requires_the_requested_execution_and_split() {
        use hipfire_config::memory::parse_offload_exec;
        let cpu = parse_offload_exec(Some(" CPU "));
        let pcie = parse_offload_exec(Some(" PCIE "));
        assert_eq!(cpu, OffloadExec::Cpu);
        assert_eq!(pcie, OffloadExec::Pcie);
        assert!(admit_pinned_control(cpu, false, true, false).is_ok());
        assert!(admit_pinned_control(cpu, false, false, false).is_err());
        assert!(admit_pinned_control(pcie, false, false, false).is_ok());
        for exec in [cpu, pcie, OffloadExec::Passback] {
            assert!(admit_pinned_control(exec, true, true, false).is_err());
            assert!(admit_pinned_control(exec, false, true, true).is_err());
            for layers in [false, true] {
                for digest in [false, true] {
                    let expected = match exec {
                        OffloadExec::Cpu => layers && digest,
                        OffloadExec::Pcie => !layers && !digest,
                        OffloadExec::Passback => false,
                    };
                    assert_eq!(admit_pinned_split(exec, layers, digest).is_ok(), expected);
                }
            }
        }
        assert!(admit_pinned_control(OffloadExec::Passback, false, true, false).is_err());
    }

    #[test]
    fn pinned_hotset_preflight_fails_closed_without_gpu() {
        use hipfire_arch_qwen4::expert_hotset as hotsets;
        let env = hotsets::HotsetEnv {
            gpu_arch: "gfx1201",
            unified_memory: false,
            expert_parallel: false,
            explicit_vram_layers: false,
            source: hotsets::HotsetSource::Aggregate,
        };
        let missing = std::env::temp_dir()
            .join(format!("qwen4-kld-{}", std::process::id()))
            .join("missing.expert-hotset");
        for policy in [
            hotsets::HotsetPolicy::Off,
            hotsets::HotsetPolicy::Explicit(PathBuf::new()),
            hotsets::HotsetPolicy::Explicit(missing),
        ] {
            assert!(hotsets::preflight(&policy, Path::new("model.hfq"), &env).is_err());
        }
        assert!(hotsets::require_experts(0).is_err());
    }

    #[test]
    fn partition_digest_preserves_every_byte_as_lowercase_hex() {
        let bytes = std::array::from_fn(|i| i as u8);
        let expected = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        assert_eq!(partition_digest_hex(bytes), expected);
        let hex = partition_digest_hex([0xab; 32]);
        assert_eq!(hex, "ab".repeat(32));
        assert_eq!(hex.len(), 64);
        assert!(hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        assert_eq!(partition_digest_hex([0; 32]), "0".repeat(64));
    }
}

#[cfg(test)]
mod s11_tests {
    use super::*;
    #[test]
    fn all_output_view_matches_each_unequal_sequence() {
        let mut storage = [0.0f32; 1479 * 8];
        let backing = GpuTensor {
            // CPU storage used only for tensor metadata, never a GPU call.
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    storage.as_mut_ptr().cast(),
                    std::mem::size_of_val(&storage),
                )
            },
            shape: vec![storage.len()],
            dtype: DType::F32,
        };
        for tokens in [166, 1479] {
            let view = all_logits_view(&backing, tokens, 8);
            assert_eq!(view.numel(), tokens * 8);
            assert_eq!(view.byte_size(), tokens * 8 * 4);
            assert_eq!(view.buf.size(), tokens * 8 * 4);
            assert_eq!(view.buf.as_ptr(), backing.buf.as_ptr());
        }
        assert_eq!(backing.numel(), 1479 * 8);
    }

    #[test]
    fn cli_rejects_missing_corpus_and_partial_rows() {
        let args = [
            "s11ref",
            "--model",
            "model.hfq",
            "--ref",
            "ref.s11ref",
            "--output",
            "out.npy",
        ];
        assert!(run_args(args.into_iter().map(str::to_string))
            .unwrap_err()
            .contains("--corpus"));
        assert!(run_args(
            args.into_iter()
                .chain(["--max-chunks", "1"])
                .map(str::to_string)
        )
        .unwrap_err()
        .contains("full corpus"));
        assert!(
            run_args(["s11ref", "--ref"].into_iter().map(str::to_string))
                .unwrap_err()
                .contains("needs a value")
        );
    }

    #[test]
    fn sequence_range_parses_start_end_exclusive() {
        assert_eq!(parse_sequence_range("0:1").unwrap(), 0..1);
        assert_eq!(parse_sequence_range("12:232").unwrap(), 12..232);
        for bad in [
            "", "3", "3:", ":3", "3:3", "4:3", "-1:3", "1:2:3", "a:2", "1:b", " 1:2", "1:2 ",
        ] {
            assert!(
                parse_sequence_range(bad).is_err(),
                "{bad:?} must be rejected"
            );
        }
    }

    fn sequences_with_rows(rows: &[usize]) -> Vec<S11Sequence> {
        rows.iter()
            .enumerate()
            .map(|(i, &n)| S11Sequence {
                id: format!("s{i}"),
                tokens: vec![0; n + 2],
                score_positions: (0..n).collect(),
                reset: Some(true),
                stratum: "text".into(),
                role_mask: None,
            })
            .collect()
    }

    #[test]
    fn partition_rows_rejects_out_of_bounds_and_empty_selections() {
        let seqs = sequences_with_rows(&[3, 0, 2, 4]);
        assert!(partition_rows(&seqs, &(0..5))
            .unwrap_err()
            .contains("out of bounds"));
        assert!(partition_rows(&seqs, &(4..5))
            .unwrap_err()
            .contains("out of bounds"));
        assert!(partition_rows(&seqs, &(9..10))
            .unwrap_err()
            .contains("out of bounds"));
        assert!(partition_rows(&seqs, &(2..2))
            .unwrap_err()
            .contains("out of bounds"));
        assert!(partition_rows(&seqs, &(1..2))
            .unwrap_err()
            .contains("no score rows"));
        assert_eq!(partition_rows(&seqs, &(0..4)).unwrap(), 0..9);
    }

    #[test]
    fn partition_rows_tile_the_full_canonical_row_order() {
        // Unequal sequences, including one with no scored rows.
        let seqs = sequences_with_rows(&[3, 0, 2, 4, 1]);
        let total: usize = seqs.iter().map(|s| s.score_positions.len()).sum();
        assert_eq!(partition_rows(&seqs, &(0..5)).unwrap(), 0..total);
        // Every contiguous split into two partitions covers each row exactly once, in order.
        for cut in 1..seqs.len() {
            let (a, b) = (
                partition_rows(&seqs, &(0..cut)),
                partition_rows(&seqs, &(cut..seqs.len())),
            );
            if let (Ok(a), Ok(b)) = (a, b) {
                assert_eq!(a.start, 0);
                assert_eq!(a.end, b.start);
                assert_eq!(b.end, total);
            }
        }
        // Per-sequence partitions (those with rows) reproduce the full row list.
        let mut covered = Vec::new();
        for i in 0..seqs.len() {
            if let Ok(r) = partition_rows(&seqs, &(i..i + 1)) {
                assert_eq!(r.len(), seqs[i].score_positions.len());
                covered.extend(r);
            }
        }
        assert_eq!(covered, (0..total).collect::<Vec<_>>());
    }

    #[test]
    fn cli_sequence_range_is_s11ref_only_and_excludes_smoke() {
        let run = |args: &[&str]| run_args(args.iter().map(|s| s.to_string())).unwrap_err();
        let base = [
            "s11ref", "--model", "m.hfq", "--ref", "r.s11ref", "--corpus", "c.json", "--output",
            "o.npy",
        ];
        let with =
            |extra: &[&'static str]| base.iter().chain(extra.iter()).copied().collect::<Vec<_>>();
        assert!(
            run(&with(&["--sequence-range", "0:2", "--smoke-seqs", "1"])).contains("--smoke-seqs")
        );
        assert!(run(&with(&["--sequence-range", "2:2"])).contains("--sequence-range"));
        assert!(run(&with(&["--sequence-range", "x"])).contains("--sequence-range"));
        assert!(run(&[
            "eval",
            "--model",
            "m.hfq",
            "--ref",
            "r.kldref",
            "--output",
            "o.kldseq",
            "--sequence-range",
            "0:1"
        ])
        .contains("only for s11ref"));
    }

    #[test]
    fn python_helper_reference_and_npy_roundtrip() {
        // Enable the external helper cross-check explicitly; no torch/GPU is
        // needed. HIPFIRE_FLOOR_ORACLE selects quality_longref.py when it is
        // not shipped beside this example. Ordinary tests need no qcal tree.
        let Some(helper) = std::env::var_os("HIPFIRE_FP8_REFS_DIR") else {
            eprintln!(
                "cross-reader test skipped: set HIPFIRE_FP8_REFS_DIR to the fp8_refs.py directory"
            );
            return;
        };
        let dir = std::env::temp_dir().join(format!("qwen4-s11ref-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let oracle = std::env::var_os("HIPFIRE_FLOOR_ORACLE")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("reference_oracle"));
        let script = r#"
import sys, json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import fp8_refs as f
import numpy as np
f.ORACLE = Path(sys.argv[2])
f.VOCAB = 8
d = Path(sys.argv[3])
sidx = np.array([[0,0,3],[0,2,5],[1,1,7]], dtype='<u4')
rows = SimpleNamespace(ids=np.array([[1,2],[3,4],[5,6]], dtype='<u4'),
    logprobs=np.log(np.array([[.4,.3],[.3,.2],[.5,.2]])).astype('<f4'),
    residual_logprob=np.log(np.array([.3,.5,.3])).astype('<f8'))
f.write_s11ref(d/'tiny.s11ref', {'corpus_sha256':'test'}, sidx, rows)
r = f.read_s11ref(d/'tiny.s11ref')
sys.path.insert(0, str(Path(sys.argv[1]).parent/'w3quality'/'kld'))
from score_kld import kld_from_block
candidate = np.array([-1.,.25,1.,-.5,.75,-.25,.5,0.], dtype=np.float32)
kl = [kld_from_block(r['ids'][i],r['logprobs'][i],r['residual'][i],candidate) for i in range(3)]
(d/'expected.json').write_text(json.dumps({'sidx':r['sidx'].tolist(),
    'ids':r['ids'].tolist(), 'lp_bits':r['logprobs'].view(np.uint32).tolist(),
    'res_bits':r['residual'].view(np.uint64).tolist(), 'kl':kl}))
"#;
        let status = std::process::Command::new("python3")
            .args(["-c", script])
            .arg(&helper)
            .arg(&oracle)
            .arg(&dir)
            .status()
            .unwrap();
        assert!(status.success());
        let data = std::fs::read(dir.join("tiny.s11ref")).unwrap();
        let r = S11Ref::parse(&data).unwrap();
        let expected: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("expected.json")).unwrap()).unwrap();
        assert_eq!(r.header.n_rows, 3);
        assert_eq!(r.header.n_vocab, 8);
        let mut block = TopKBlock {
            residual_logprob: 0.0,
            top: Vec::new(),
        };
        for row in 0..3 {
            assert_eq!(serde_json::json!(r.index(row)), expected["sidx"][row]);
            r.block(row, &mut block).unwrap();
            assert_eq!(
                block.residual_logprob.to_bits(),
                expected["res_bits"][row].as_u64().unwrap()
            );
            let kl =
                kld_from_block(&block, &[-1.0, 0.25, 1.0, -0.5, 0.75, -0.25, 0.5, 0.0]).unwrap();
            assert!((kl - expected["kl"][row].as_f64().unwrap()).abs() < 1e-12);
            for j in 0..2 {
                assert_eq!(serde_json::json!(block.top[j].0), expected["ids"][row][j]);
                assert_eq!(
                    u64::from(block.top[j].1.to_bits()),
                    expected["lp_bits"][row][j].as_u64().unwrap()
                );
            }
        }
        let corpus = S11Corpus {
            schema: "hipfire.qwen4.kl_corpus.v1".into(),
            sequences: vec![
                S11Sequence {
                    id: "a".into(),
                    tokens: vec![0, 3, 4, 5],
                    score_positions: vec![0, 2],
                    reset: Some(true),
                    stratum: "text".into(),
                    role_mask: None,
                },
                S11Sequence {
                    id: "b".into(),
                    tokens: vec![0, 1, 7],
                    score_positions: vec![1],
                    reset: Some(true),
                    stratum: "assistant".into(),
                    role_mask: Some(vec![0, 0, 1]),
                },
            ],
        };
        assert_eq!(validate_s11_corpus(&r, &corpus).unwrap(), 4);
        let mut bad = data.clone();
        bad.pop();
        assert!(S11Ref::parse(&bad).is_err());
        write_kl_npy(&dir.join("rows.npy"), &[0.1, 0.2, 0.3]).unwrap();
        let status = std::process::Command::new("python3").args(["-c",
            "import numpy as np,sys; a=np.load(sys.argv[1]); assert a.dtype==np.float64 and a.shape==(3,); np.testing.assert_array_equal(a,[.1,.2,.3])"])
            .arg(dir.join("rows.npy")).status().unwrap();
        assert!(status.success());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod slowlink_pair_tests {
    use super::*;
    use slowlink_receipt::KldCorpus;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "qwen4-kld-pair-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    /// A synthetic pinned corpus identity over a test corpus' sha.
    fn identity(sha: &str, sequences: u64) -> KldCorpus<'_> {
        KldCorpus {
            name: "synthetic",
            corpus_sha256: sha,
            corpus_md5: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ref_sha256: "2222222222222222222222222222222222222222222222222222222222222222",
            ref_md5: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            sequences,
            rows: sequences * 255,
        }
    }

    /// A decode S11 receipt of one arm over the test corpus.
    fn arm(dir: &Path, name: &str, corpus_sha: &str, rows: &[f64], sequences: usize) -> PathBuf {
        let npy = dir.join(format!("{name}.npy"));
        write_kl_npy(&npy, rows).unwrap();
        let receipt = serde_json::json!({
            "schema": "hipfire.qwen4.s11ref_klrows.v1", "direction": "KL(ref || hipfire)",
            "estimator": "e", "model_sha256": "1".repeat(64), "ref_sha256": "2".repeat(64),
            "corpus_sha256": corpus_sha, "binary_sha256": "4".repeat(64),
            "output_sha256": "5".repeat(64), "rows": rows.len(), "n_ctx": 512,
            "sequences": sequences, "forward_tokens": 1, "route": "decode",
        });
        std::fs::write(format!("{}.json", npy.display()), receipt.to_string()).unwrap();
        npy
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        F16,
        Iu4,
        Mixed,
    }

    fn census(kind: Kind, index: usize) -> serde_json::Value {
        let (iu4, f16) = match kind {
            Kind::F16 => (0, 48),
            Kind::Iu4 => (48, 0),
            Kind::Mixed if index == 0 => (48, 0),
            Kind::Mixed => (0, 48),
        };
        serde_json::json!({
            "sym_iu4_gate_up": iu4, "sym_iu4_down": iu4,
            "a4_gate_producer": iu4, "a4_down_producer": iu4,
            "f16_wmma_gate_up": f16, "f16_wmma_down": f16,
        })
    }

    /// The per-sequence censuses of `kind` summed over `sequences`.
    fn census_total(kind: Kind, sequences: usize) -> serde_json::Value {
        let mut total = serde_json::Map::new();
        for i in 0..sequences {
            for (key, value) in census(kind, i).as_object().unwrap() {
                let sum = total.get(key).and_then(|v| v.as_u64()).unwrap_or(0) + value.as_u64().unwrap();
                total.insert(key.clone(), sum.into());
            }
        }
        total.into()
    }

    /// A prefill S11 receipt of one arm: `kind` is the observed census,
    /// `requested` the flag, `binary` one hex digit repeated into the binary
    /// digests, `forwards` the number of forwards each sequence ran as.
    #[allow(clippy::too_many_arguments)]
    fn prefill_arm(
        dir: &Path,
        name: &str,
        corpus_sha: &str,
        rows: &[f64],
        sequences: usize,
        kind: Kind,
        requested: bool,
        binary: &str,
        forwards: usize,
    ) -> PathBuf {
        let npy = dir.join(format!("{name}.npy"));
        write_kl_npy(&npy, rows).unwrap();
        let forward_rows: Vec<usize> = vec![512 / forwards; forwards];
        let routes: Vec<serde_json::Value> = (0..sequences)
            .map(|i| {
                serde_json::json!({
                    "id": format!("s{i}"), "tokens": 512, "prefill_chunk_rows": 512 / forwards,
                    "forwards": forward_rows.iter().map(|&r| serde_json::json!({
                        "rows": r, "route_rows": r, "sym_iu4_applies": requested && r >= 512,
                    })).collect::<Vec<_>>(),
                    "census": census(kind, i),
                })
            })
            .collect();
        let verdict = match kind {
            Kind::F16 => "f16",
            Kind::Iu4 => "iu4",
            Kind::Mixed => "mixed",
        };
        let mut env = serde_json::Map::new();
        env.insert("HIPFIRE_KV_MODE".into(), "q8".into());
        env.insert(
            "HIPFIRE_QWEN4_MOE_SYM_IU4".into(),
            if requested { "1" } else { "0" }.into(),
        );
        let receipt = serde_json::json!({
            "schema": "hipfire.qwen4.s11ref_klrows.v1", "direction": "KL(ref || hipfire)",
            "estimator": "e", "model_sha256": "1".repeat(64), "ref_sha256": "2".repeat(64),
            "corpus_sha256": corpus_sha, "binary_sha256": binary.repeat(64),
            "output_sha256": "5".repeat(64), "rows": rows.len(), "n_ctx": 512,
            "sequences": sequences, "forward_tokens": sequences * 512, "route": "cold-prefill",
            "moe_route": verdict, "sequence_routes": routes,
            "moe_launch_census": census_total(kind, sequences),
            "moe_iu4": {
                "env": if requested { Some("1") } else { Some("0") }, "requested": requested,
                "descriptor": {
                    "arch": "gfx1100", "module": "qwen4_moe_iu4_sym_pm_gfx1100",
                    "gate_up": "g", "down": "d", "nt": 1, "threads": [64, 128],
                    "gate_producer": "mq_rotate_x_i4", "down_producer": "k3",
                    "checker": "chk", "c2": true,
                },
            },
            "env": env, "binary_path": "/bin/qwen4_kld", "binary_md5": binary.repeat(32),
            "ref_md5": "b".repeat(32), "corpus_md5": "a".repeat(32),
        });
        std::fs::write(format!("{}.json", npy.display()), receipt.to_string()).unwrap();
        npy
    }

    fn corpus(dir: &Path, sequences: usize, rows_each: usize) -> (PathBuf, String) {
        let seqs: Vec<serde_json::Value> = (0..sequences)
            .map(|i| {
                serde_json::json!({
                    "id": format!("s{i}"), "tokens": vec![1u32; rows_each + 2],
                    "score_positions": (1..=rows_each).collect::<Vec<usize>>(),
                    "reset": true, "stratum": "x", "role_mask": null,
                })
            })
            .collect();
        let bytes = serde_json::json!({"schema": "hipfire.qwen4.kl_corpus.v1", "sequences": seqs})
            .to_string()
            .into_bytes();
        let path = dir.join("corpus.json");
        std::fs::write(&path, &bytes).unwrap();
        (path, format!("{:x}", Sha256::digest(&bytes)))
    }

    fn report(out: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(out).unwrap()).unwrap()
    }

    #[test]
    fn slowlink_pair_npy_roundtrips_what_the_scorer_writes() {
        let dir = temp_dir("npy");
        let rows = [0.25, -1.5, 3.0e-9];
        write_kl_npy(&dir.join("r.npy"), &rows).unwrap();
        assert_eq!(read_kl_npy(&dir.join("r.npy")).unwrap(), rows);
        std::fs::write(dir.join("bad.npy"), b"not numpy").unwrap();
        assert!(read_kl_npy(&dir.join("bad.npy")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_pinned_corpora_are_code24_and_wt2() {
        let [code24, wt2] = slowlink_receipt::KLD_CORPORA;
        assert_eq!((code24.name, code24.sequences, code24.rows), ("code24", 24, 6120));
        assert_eq!((wt2.name, wt2.sequences, wt2.rows), ("wt2", 32, 8160));
        assert_eq!(code24.sequences * 255, code24.rows);
        assert_eq!(wt2.sequences * 255, wt2.rows);
        // The synthetic tests' corpora are not pinned gate corpora.
        let dir = temp_dir("unpinned");
        let (corpus_path, sha) = corpus(&dir, 24, 255);
        let rows: Vec<f64> = vec![0.05; 6120];
        let (b, c) = (
            arm(&dir, "base", &sha, &rows, 24),
            arm(&dir, "cand", &sha, &rows, 24),
        );
        let out = dir.join("pair.json");
        let error = pair(&b, &c, &corpus_path, &out, false).unwrap_err();
        assert!(error.contains("not a pinned gate corpus"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_verdict_uses_per_sequence_means_over_the_full_corpus() {
        let dir = temp_dir("full");
        let (corpus_path, sha) = corpus(&dir, 24, 255);
        let table = [identity(&sha, 24)];
        let base: Vec<f64> = (0..6120)
            .map(|i| 0.05 + 0.0001 * ((i / 255) % 5) as f64)
            .collect();
        let cand: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.0001 * ((i / 255) % 3) as f64)
            .collect();
        let (b, c) = (
            arm(&dir, "base", &sha, &base, 24),
            arm(&dir, "cand", &sha, &cand, 24),
        );
        let out = dir.join("pair.json");
        pair_in(&b, &c, &corpus_path, &out, false, &table).unwrap();
        let rep = report(&out);
        assert_eq!(rep["schema"], "hipfire.qwen4.slowlink_kld_pair.v1");
        assert_eq!(rep["pass"], true);
        assert_eq!(rep["sequences"], 24);
        assert_eq!(rep["rows"], 6120);
        assert_eq!(rep["per_sequence"].as_array().unwrap().len(), 24);
        assert!(rep["worst_sequence"]["diff"].as_f64().unwrap() > 0.0);
        // A clearly worse candidate fails the verdict and the command.
        let worse: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.05 + 0.0001 * ((i / 255) % 3) as f64)
            .collect();
        let w = arm(&dir, "worse", &sha, &worse, 24);
        assert!(pair_in(&b, &w, &corpus_path, &out, false, &table).is_err());
        assert_eq!(report(&out)["pass"], false);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_accepts_the_32_sequence_wt2_shape_and_refuses_the_wrong_identity() {
        let dir = temp_dir("wt2");
        let (corpus_path, sha) = corpus(&dir, 32, 255);
        let rows: Vec<f64> = (0..8160).map(|i| 0.05 + 1e-5 * ((i / 255) % 4) as f64).collect();
        let (b, c) = (
            arm(&dir, "base", &sha, &rows, 32),
            arm(&dir, "cand", &sha, &rows, 32),
        );
        let out = dir.join("pair.json");
        // The same corpus pinned as a 32-sequence corpus passes; pinned as a
        // 24-sequence corpus its receipts are a partial/oversized corpus.
        pair_in(&b, &c, &corpus_path, &out, false, &[identity(&sha, 32)]).unwrap();
        assert_eq!(report(&out)["rows"], 8160);
        let error = pair_in(&b, &c, &corpus_path, &out, false, &[identity(&sha, 24)]).unwrap_err();
        assert!(error.contains("partial corpus"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_refuses_a_partial_corpus_and_a_mismatched_reference() {
        let dir = temp_dir("partial");
        let (corpus_path, sha) = corpus(&dir, 10, 255);
        let table = [identity(&sha, 24)];
        let rows: Vec<f64> = vec![0.05; 2550];
        let (b, c) = (
            arm(&dir, "base", &sha, &rows, 10),
            arm(&dir, "cand", &sha, &rows, 10),
        );
        let out = dir.join("pair.json");
        let error = pair_in(&b, &c, &corpus_path, &out, false, &table).unwrap_err();
        assert!(error.contains("partial corpus"), "{error}");
        assert!(!std::fs::read_to_string(&out)
            .unwrap()
            .contains("\"pass\": true"));
        // Full row count but another corpus SHA than the receipts name.
        let (full_corpus, full_sha) = corpus(&dir, 24, 255);
        let table = [identity(&full_sha, 24)];
        let rows: Vec<f64> = vec![0.05; 6120];
        let (b, c) = (
            arm(&dir, "b2", "9".repeat(64).as_str(), &rows, 24),
            arm(&dir, "c2", &full_sha, &rows, 24),
        );
        let error = pair_in(&b, &c, &full_corpus, &out, false, &table).unwrap_err();
        assert!(
            error.contains("corpus_sha256") || error.contains("not the same"),
            "{error}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_iu4_prefill_passes_f16_baseline_against_iu4_candidate() {
        let dir = temp_dir("iu4ok");
        let (corpus_path, sha) = corpus(&dir, 24, 255);
        let table = [identity(&sha, 24)];
        let base: Vec<f64> = (0..6120)
            .map(|i| 0.05 + 0.0001 * ((i / 255) % 5) as f64)
            .collect();
        let cand: Vec<f64> = base.iter().map(|v| v + 0.0004).collect();
        let b = prefill_arm(&dir, "base", &sha, &base, 24, Kind::F16, false, "4", 1);
        let c = prefill_arm(&dir, "cand", &sha, &cand, 24, Kind::Iu4, true, "4", 1);
        let out = dir.join("pair.json");
        pair_in(&b, &c, &corpus_path, &out, true, &table).unwrap();
        let rep = report(&out);
        assert_eq!(rep["pass"], true);
        assert_eq!(rep["route"], "cold-prefill");
        assert_eq!(rep["moe_iu4_pair"], true);
        assert_eq!(rep["base"]["moe_route"], "f16");
        assert_eq!(rep["cand"]["moe_route"], "iu4");
        // Without --moe-iu4 the same receipts are an ordinary prefill pair.
        pair_in(&b, &c, &corpus_path, &out, false, &table).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_iu4_expected_but_observed_fallback_is_a_failure() {
        let dir = temp_dir("iu4fallback");
        let (corpus_path, sha) = corpus(&dir, 24, 255);
        let table = [identity(&sha, 24)];
        let rows: Vec<f64> = vec![0.05; 6120];
        let b = prefill_arm(&dir, "base", &sha, &rows, 24, Kind::F16, false, "4", 1);
        // The flag was set but every sequence ran the F16 route: identical
        // numbers, which is NOT a deterministic incumbent pass.
        let c = prefill_arm(&dir, "cand", &sha, &rows, 24, Kind::F16, true, "4", 1);
        let out = dir.join("pair.json");
        let error = pair_in(&b, &c, &corpus_path, &out, true, &table).unwrap_err();
        assert!(error.contains("expected IU4 but observed"), "{error}");
        assert_eq!(report(&out)["pass"], false);
        // A mixed route fails any pairing, with or without --moe-iu4.
        let m = prefill_arm(&dir, "mixed", &sha, &rows, 24, Kind::Mixed, true, "4", 1);
        for expect_iu4 in [true, false] {
            let error = pair_in(&b, &m, &corpus_path, &out, expect_iu4, &table).unwrap_err();
            assert!(error.contains("mixed"), "{error}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slowlink_pair_iu4_needs_one_binary_one_forward_and_one_env_difference() {
        let dir = temp_dir("iu4identity");
        let (corpus_path, sha) = corpus(&dir, 24, 255);
        let table = [identity(&sha, 24)];
        let rows: Vec<f64> = vec![0.05; 6120];
        let b = prefill_arm(&dir, "base", &sha, &rows, 24, Kind::F16, false, "4", 1);
        let out = dir.join("pair.json");
        // Another binary.
        let other = prefill_arm(&dir, "other", &sha, &rows, 24, Kind::Iu4, true, "7", 1);
        let error = pair_in(&b, &other, &corpus_path, &out, true, &table).unwrap_err();
        assert!(error.contains("binary_md5"), "{error}");
        // A 512-token sequence split into two 256-row forwards never reaches the IU4 floor.
        let split = prefill_arm(&dir, "split", &sha, &rows, 24, Kind::Iu4, true, "4", 2);
        let error = pair_in(&b, &split, &corpus_path, &out, true, &table).unwrap_err();
        assert!(error.contains("one 512-row forward"), "{error}");
        // Another HIPFIRE_* setting besides the flag.
        let cand = prefill_arm(&dir, "env", &sha, &rows, 24, Kind::Iu4, true, "4", 1);
        let path = format!("{}.json", cand.display());
        let mut receipt: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        receipt["env"]["HIPFIRE_KV_MODE"] = "bf16".into();
        std::fs::write(&path, receipt.to_string()).unwrap();
        let error = pair_in(&b, &cand, &corpus_path, &out, true, &table).unwrap_err();
        assert!(error.contains("only HIPFIRE_QWEN4_MOE_SYM_IU4 may differ"), "{error}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
