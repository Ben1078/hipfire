// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_wide_verify — real-model oracle for wide MTP verify windows.
//!
//! For each requested window length `n` this runs the PRODUCTION verify
//! transaction (`Qwen4Bundle::verify_block_rows` behind `SpecTarget::verify_block`:
//! snapshot, row-armed `spec_forward_rows` with `prefill_scope = false`) over a
//! block of `n` tokens and compares every row's logits BITWISE with `n`
//! sequential `Qwen4Bundle::forward_token` steps from the same resident state.
//! It then checks the post-commit state (`SpecTarget::commit_prefix`) against
//! the autoregressive state after the same number of tokens, part by part
//! (SHA-256 of every `SessionState::snapshot_parts` part), and times both
//! routes. `Qwen4Bundle::forward_chunk` is NOT the verify route (it sets
//! `prefill_scope = true`) and is not used for the comparison.
//!
//! Usage:
//!   qwen4_wide_verify --model M.hfq (--prompt-file P.txt | --prompt-files a.txt,b.txt)
//!                     --rows 1,2,...,32 --out OUT.json
//!                     [--iters 5] [--n-ctx 8192] [--block greedy|prompt]
//!                     [--accepts ends|all|none] [--arms base,wide16,wide32] [--trace]
//!                     [--hot-budget-mib N] [--hotset SIDECAR]
//!
//! `--rows` is a comma list; `a-b` ranges are accepted (`1-32`).
//! `--prompt-files` runs every prompt in one model load (`--prompt-file` is the
//! single-element alias); each prompt gets its own prefill, S0, block and AR
//! reference. The JSON is per prompt x arm x n.
//! `--arms` (default `base`): each arm sets the rdna-compute runtime verify-rows
//! override (`base` = unset, the arch table / env decides; `wideN` = N) and
//! runs the whole per-n suite in the same process. The override only takes
//! effect on gfx1151/gfx1201; the JSON records the effective verify rows.
//! `--block greedy` (default) is the AR greedy continuation of the prompt
//! (block[0] is the argmax of the prefill's final row, block[i+1] the argmax
//! after block[i]); `--block prompt` is the last max(rows) prompt tokens
//! (copy-like).
//! `--accepts ends` checks accept_len in {0, n-1}; `all` checks 0..n; `none`
//! skips every commit_prefix arm (no re-prefills; use it at deep contexts).
//! `--trace` runs per prompt x arm x n an AR window then a verify window, no
//! digests, each preceded by delimiter launches of `logit_softcap_f32` (never
//! launched by the Qwen4 forward): AR = 2 back-to-back delimiters then n
//! `forward_token`s, verify = 1 delimiter then the verify. `trace_pattern` in
//! the JSON documents it so a rocprofv3 kernel trace can be cut and labelled.
//! Every record carries the execution mode (`replay.is_recording()`,
//! `graphs.capture_mode`) read just before its windows.
//!
//! A fixed `--hot-budget-mib` uses the same frozen expert-tier partition with
//! `HIPFIRE_OFFLOAD_EXEC=cpu` (requires D2 armed) or `pcie` (requires D2 unarmed).
//! Both receipts expose its actual `partition_digest`; `split_digest` remains
//! the real D2 identity, never a substitute PCIe partition identity. Missing or
//! invalid hotsets, zero/overflowing budgets and passback fail the pinned load.
//!
//! Rewind: AR passes and the timed arms rewind with the model-owned snapshot
//! ticket (`Qwen4Bundle::snapshot` / `restore`; a verify is rolled back with its
//! own ticket). Accept arms end in a committed state with no ticket, so those
//! re-prefill the prompt; every re-prefill is checked to reproduce the
//! post-prefill state digest bit for bit, and the first ticket rewind of each
//! kind is checked the same way.
//!
//! Exit status is non-zero if any n <= 8 is not fully bitwise, any state digest
//! mismatches at n <= 8, or the run itself fails. n >= 9 mismatches are
//! reported, never fatal.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::wide_verify_oracle as oracle;
use hipfire_config::memory::OffloadExec;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::llama::argmax;
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, PleRowsOrigin, ResolvePolicy};
use hipfire_runtime::spec::{SpecScratch, SpecTarget};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::prefill_invariant::{FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: qwen4_wide_verify --model M.hfq (--prompt-file P.txt | --prompt-files a.txt,b.txt) --rows 1,2,...,32 --out OUT.json [--iters 5] [--n-ctx 8192] [--block greedy|prompt] [--accepts ends|all|none] [--arms base,wide16,wide32] [--hot-budget-mib N] [--hotset SIDECAR] [--row-capture on|off] [--trace]";

/// The delimiter kernel of `--trace`: a one-block elementwise kernel
/// (`kernels/src/logit_softcap.hip`) launched only by the Gemma/Glimmer
/// logit-softcap paths, never by any Qwen4 forward.
const DELIMITER_KERNEL: &str = "logit_softcap_f32";

fn es<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockMode {
    Greedy,
    Prompt,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Accepts {
    /// accept_len in {0, n-1}.
    Ends,
    /// accept_len in 0..n: seed-inclusive consumed prefix lengths 1..=n.
    All,
    /// No commit_prefix arms (no re-prefills).
    None,
}

impl Accepts {
    fn name(self) -> &'static str {
        match self {
            Accepts::Ends => "ends",
            Accepts::All => "all",
            Accepts::None => "none",
        }
    }
}

/// A verify-width arm: `base` leaves the runtime override unset (the arch
/// table / env decides), `wideN` sets it to N (clamped by rdna-compute).
#[derive(Clone)]
struct Arm {
    name: String,
    rows: Option<usize>,
}

impl Arm {
    fn base() -> Self {
        Arm {
            name: "base".to_string(),
            rows: None,
        }
    }
}

fn parse_arms(spec: &str) -> Result<Vec<Arm>, String> {
    let mut arms: Vec<Arm> = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let arm = if part == "base" {
            Arm::base()
        } else if let Some(n) = part
            .strip_prefix("wide")
            .and_then(|n| n.parse::<usize>().ok())
        {
            Arm {
                name: part.to_string(),
                rows: Some(n),
            }
        } else {
            return Err(format!("bad --arms entry '{part}' (base|wideN)"));
        };
        if arms.iter().any(|a| a.name == arm.name) {
            return Err(format!("duplicate --arms entry '{part}'"));
        }
        arms.push(arm);
    }
    if arms.is_empty() {
        return Err("--arms is empty".into());
    }
    Ok(arms)
}

struct Args {
    model: PathBuf,
    prompt_files: Vec<PathBuf>,
    rows: Vec<usize>,
    out: PathBuf,
    iters: usize,
    n_ctx: usize,
    block: BlockMode,
    accepts: Accepts,
    arms: Vec<Arm>,
    trace: bool,
    hot_budget_mib: Option<u64>,
    hotset: Option<PathBuf>,
    row_capture: bool,
}

fn parse_rows(spec: &str) -> Result<Vec<usize>, String> {
    let mut rows = BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("bad --rows range '{part}'"))?;
            let b: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("bad --rows range '{part}'"))?;
            if a == 0 || b < a {
                return Err(format!("bad --rows range '{part}'"));
            }
            rows.extend(a..=b);
        } else {
            let n: usize = part
                .parse()
                .map_err(|_| format!("bad --rows entry '{part}'"))?;
            if n == 0 {
                return Err("--rows entries must be >= 1".into());
            }
            rows.insert(n);
        }
    }
    if rows.is_empty() {
        return Err("--rows is empty".into());
    }
    Ok(rows.into_iter().collect())
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    parse_args_from(&argv)
}

fn parse_args_from(argv: &[String]) -> Result<Args, String> {
    let mut model = None;
    let mut prompt_files: Option<Vec<PathBuf>> = None;
    let mut rows = None;
    let mut out = None;
    let mut iters = 5usize;
    let mut n_ctx = 8192usize;
    let mut block = BlockMode::Greedy;
    let mut accepts = Accepts::Ends;
    let mut arms = vec![Arm::base()];
    let mut trace = false;
    let mut hot_budget_mib = None;
    let mut hotset = None;
    let mut row_capture = true;
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        if flag == "--trace" {
            trace = true;
            i += 1;
            continue;
        }
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?
            .as_str();
        match flag {
            "--model" => model = Some(PathBuf::from(value)),
            "--hot-budget-mib" => {
                let mib: u64 = value.parse().map_err(|_| format!("bad --hot-budget-mib '{value}'"))?;
                if mib == 0 || mib.checked_mul(1 << 20).is_none() {
                    return Err("--hot-budget-mib must be nonzero and fit in bytes".into());
                }
                hot_budget_mib = Some(mib);
            }
            "--hotset" => hotset = Some(PathBuf::from(value)),
            "--row-capture" => row_capture = match value {
                "on" => true,
                "off" => false,
                _ => return Err("--row-capture takes on|off".into()),
            },
            // `--prompt-file` is the single-element alias of `--prompt-files`.
            "--prompt-file" | "--prompt-files" => {
                prompt_files = Some(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(PathBuf::from)
                        .collect(),
                )
            }
            "--rows" => rows = Some(parse_rows(value)?),
            "--out" => out = Some(PathBuf::from(value)),
            "--iters" => {
                iters = value
                    .parse()
                    .map_err(|_| format!("bad --iters '{value}'"))?
            }
            "--n-ctx" => {
                n_ctx = value
                    .parse()
                    .map_err(|_| format!("bad --n-ctx '{value}'"))?
            }
            "--block" => {
                block = match value {
                    "greedy" => BlockMode::Greedy,
                    "prompt" => BlockMode::Prompt,
                    other => return Err(format!("--block takes greedy|prompt, got '{other}'")),
                }
            }
            "--accepts" => {
                accepts = match value {
                    "ends" => Accepts::Ends,
                    "all" => Accepts::All,
                    "none" => Accepts::None,
                    other => return Err(format!("--accepts takes ends|all|none, got '{other}'")),
                }
            }
            "--arms" => arms = parse_arms(value)?,
            other => return Err(format!("unknown argument '{other}'\n{USAGE}")),
        }
        i += 2;
    }
    if iters == 0 {
        return Err("--iters must be >= 1".into());
    }
    if hotset.is_some() && hot_budget_mib.is_none() {
        return Err("--hotset requires a nonzero --hot-budget-mib".into());
    }
    let prompt_files =
        prompt_files.ok_or_else(|| format!("--prompt-file(s) is required\n{USAGE}"))?;
    if prompt_files.is_empty() {
        return Err("--prompt-files is empty".into());
    }
    Ok(Args {
        model: model.ok_or_else(|| format!("--model is required\n{USAGE}"))?,
        prompt_files,
        rows: rows.ok_or_else(|| format!("--rows is required\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n{USAGE}"))?,
        iters,
        n_ctx,
        block,
        accepts,
        arms,
        trace,
        hot_budget_mib,
        hotset,
        row_capture,
    })
}

/// Admission is separate from GPU setup so CPU-only tests cover both controls.
fn admit_pinned_control(
    exec: OffloadExec,
    uma: bool,
    cpu_experts_enabled: bool,
) -> Result<(), String> {
    if uma {
        return Err("fixed hotset requires a discrete GPU".into());
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

#[cfg(test)]
mod pinned_control_tests {
    use super::*;

    fn cli(extra: &[&str]) -> Result<Args, String> {
        let argv: Vec<String> = [
            "--model", "model.hfq", "--prompt-file", "prompt.txt",
            "--rows", "1,2,4,8", "--out", "receipt.json",
        ]
        .into_iter()
        .chain(extra.iter().copied())
        .map(str::to_owned)
        .collect();
        parse_args_from(&argv)
    }

    #[test]
    fn pinned_cli_keeps_budget_requirements_fail_closed() {
        assert!(cli(&[]).unwrap().hot_budget_mib.is_none());
        assert!(cli(&["--hotset", "bound.expert-hotset"]).is_err());
        for budget in ["0", "-1", "bad", "17592186044416", "18446744073709551616"] {
            assert!(cli(&["--hot-budget-mib", budget]).is_err(), "{budget}");
        }
        assert!(cli(&["--hot-budget-mib"]).is_err());
        let args = cli(&[
            "--hot-budget-mib", "4096", "--hotset", "bound.expert-hotset",
        ]).unwrap();
        assert_eq!(args.hot_budget_mib, Some(4096));
        assert_eq!(args.hotset.as_deref(), Some(Path::new("bound.expert-hotset")));
    }

    #[test]
    fn pinned_controls_use_the_config_parser_and_require_real_split_status() {
        use hipfire_config::memory::parse_offload_exec;

        let pcie = parse_offload_exec(Some(" PCIE "));
        let cpu = parse_offload_exec(Some(" CPU "));
        assert_eq!(pcie, OffloadExec::Pcie);
        assert_eq!(cpu, OffloadExec::Cpu);
        assert!(admit_pinned_control(pcie, false, false).is_ok());
        assert!(admit_pinned_control(cpu, false, true).is_ok());
        assert!(admit_pinned_control(cpu, false, false).is_err());
        for exec in [pcie, cpu] {
            assert!(admit_pinned_control(exec, true, true).is_err());
            for layers in [false, true] {
                for digest in [false, true] {
                    let expected = match exec {
                        OffloadExec::Cpu => layers && digest,
                        OffloadExec::Pcie => !layers && !digest,
                        OffloadExec::Passback => unreachable!(),
                    };
                    assert_eq!(admit_pinned_split(exec, layers, digest).is_ok(), expected);
                }
            }
        }
        let passback = parse_offload_exec(Some("passback"));
        assert!(admit_pinned_control(passback, false, true).is_err());
        assert!(admit_pinned_split(passback, false, false).is_err());
    }

    #[test]
    fn pinned_hotset_preflight_rejects_disabled_empty_and_missing_paths() {
        use hipfire_arch_qwen4::expert_hotset as hotsets;

        let env = hotsets::HotsetEnv {
            gpu_arch: "gfx1201",
            unified_memory: false,
            expert_parallel: false,
            explicit_vram_layers: false,
            source: hotsets::HotsetSource::Aggregate,
        };
        let missing = std::env::temp_dir()
            .join(format!("qwen4-wide-verify-{}", std::process::id()))
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
}

// ---------------------------------------------------------------------------
// Product loader (qwen4_kld `Model::load`, minus the PLE hot store)
// ---------------------------------------------------------------------------

struct LoadInfo {
    arch: String,
    ple_line: String,
    ple_note: Option<String>,
    ple_rows_source: String,
    overlay_applied: bool,
    model_bytes: u64,
    uma: bool,
    expert_vram_layers: Option<usize>,
    offload_exec: OffloadExec,
    partition_digest: Option<String>,
    split_armed: bool,
}

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    vocab: usize,
    tokenizer: Tokenizer,
    info: LoadInfo,
}

impl Model {
    /// Same range-loaded path as the serve loader (`hipfire-loader` carriers):
    /// PLE rows sidecar (never deriving a missing q4), admission, trunk overlay,
    /// the loader's forward scratch plan, UMA mmap drop, optional
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, `attach_forward`. No MTP head, no
    /// session cache.
    fn load(path: &Path, n_ctx: usize, hot_budget_mib: Option<u64>, hotset: Option<&Path>) -> Result<Self, String> {
        let model_bytes = std::fs::metadata(path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        let mut hfq =
            HfqFile::open(path).map_err(|e| format!("open HFQM {}: {e}", path.display()))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        // The GPU arch drives `memory.ple_format = auto`; initialising it
        // allocates no model memory, so admission still precedes every upload.
        let mut gpu = Gpu::init().map_err(es)?;
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
        let overlay_applied = trunk_overlay.is_some();
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
        let ple_line = format!(
            "ple rows: {} from {} ({rows_encoding}, {} order, {:.1} GB)",
            choice.format.name(),
            match &choice.origin {
                PleRowsOrigin::Embedded => "embedded model file".to_string(),
                PleRowsOrigin::Sidecar(sidecar) => sidecar.display().to_string(),
            },
            if rank_table { "rank" } else { "row" },
            rows_total as f64 / 1e9
        );
        eprintln!("qwen4_wide_verify: {ple_line}");
        if let Some(note) = &choice.note {
            eprintln!("qwen4_wide_verify: ple rows: {note}");
        }
        // The loader's forward scratch plan, decided from the actual payload
        // while the mapping is readable.
        let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
            &gpu,
            &hfq,
            &receipt.config,
            &receipt.manifest,
        )
        .map_err(es)?;
        let offload_exec = hipfire_config::memory::offload_exec();
        // Use the same bound sidecar, slab plan and tier owners as cpu_split_parity.
        // The fixed budget freezes one partition for all AR/verify/commit passes.
        let tier_lease = if let Some(mib) = hot_budget_mib {
            use hipfire_arch_qwen4::expert_hotset as hotsets;
            admit_pinned_control(
                offload_exec,
                gpu.is_uma(),
                hipfire_dispatch::cpu_exec::moe_cpu_experts_enabled(),
            )?;
            if hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()?.is_some() {
                return Err("fixed hotset conflicts with HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS".into());
            }
            let policy = match hotset {
                Some(path) => hotsets::HotsetPolicy::Explicit(path.to_path_buf()),
                None => hotsets::HotsetPolicy::from_env()?,
            };
            let env = hotsets::HotsetEnv {
                gpu_arch: &gpu.arch,
                unified_memory: false,
                expert_parallel: false,
                explicit_vram_layers: false,
                source: match &receipt.manifest.identity.recipe {
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextPacked { .. } => hotsets::HotsetSource::Aggregate,
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::FlashNextSlab { .. } => hotsets::HotsetSource::SlabArena,
                    hipfire_arch_qwen4::weights::Qwen4TierRecipe::StockFp8 => hotsets::HotsetSource::StockFp8,
                    _ => hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into()),
                },
            };
            let sidecar_path = hotsets::preflight(&policy, path, &env)
                .map_err(|skip| skip.log_line(&policy.sidecar_path(path)).unwrap_or_else(|| skip.to_string()))?;
            let base = hotsets::HotsetBase::from_hfq(&hfq)?;
            let sidecar = hotsets::HotsetSidecar::load_bound(
                &sidecar_path, receipt.config.num_hidden_layers,
                receipt.config.num_experts, receipt.config.num_experts_per_tok, &base,
            ).map_err(|skip| skip.to_string())?;
            receipt.manifest.check_expert_tiers().map_err(es)?;
            let slabs = hipfire_arch_qwen4::artifact::hotset_aggregate_slabs(
                &hfq, &receipt.config, &receipt.manifest,
            )?;
            let plan = hipfire_arch_qwen4::expert_tier_plan::plan_expert_tiers_from_slabs(
                &slabs, sidecar.rank(), mib * (1 << 20),
            ).map_err(es)?;
            hotsets::require_experts(plan.device_experts()).map_err(|skip| skip.to_string())?;
            plan.check_executable(&gpu.arch).map_err(es)?;
            hipfire_arch_qwen4::expert_residency::place_routed_experts(
                &mut receipt.manifest.weights, receipt.config.num_hidden_layers,
            );
            receipt.manifest = receipt.manifest.with_expert_tiers().map_err(es)?;
            {
                let keep: BTreeSet<_> = receipt.manifest.resident_keys().collect();
                receipt.placements.retain(|p| keep.contains(&(p.name.as_str(), p.layer)));
            }
            let lease = hipfire_arch_qwen4::expert_tier_owners::materialize(
                &mut gpu, std::sync::Arc::new(plan), &|offset, len| {
                    let offset = usize::try_from(offset).map_err(|_| "hotset source offset overflow".to_string())?;
                    let len = usize::try_from(len).map_err(|_| "hotset source length overflow".to_string())?;
                    hfq.data_range(offset, len)
                        .ok_or_else(|| format!("source span {offset}+{len} is outside the artifact mapping"))
                },
            ).map_err(|e| format!("expert tier materialize: {e}"))?;
            lease.bind_hotset_sidecar_sha256(sidecar.sha256()).map_err(es)?;
            eprintln!("qwen4_wide_verify: hotset {} sha256 {} fixed budget {mib} MiB",
                sidecar_path.display(), sidecar.sha256());
            Some(lease)
        } else {
            None
        };
        // The lease retains the actual frozen plan's logical digest in either
        // execution mode; PCIe has no D2 split digest to use as its identity.
        let partition_digest = tier_lease.as_ref().map(|lease| {
            let mut hex = String::with_capacity(64);
            for byte in lease.logical_digest() {
                write!(&mut hex, "{byte:02x}").expect("write digest to String");
            }
            hex
        });
        let uma = gpu.is_uma();
        if uma {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().map_err(|e| format!("qwen4 mesh: {e}"))?;
        let expected = WeightOrigin::for_single(&mesh, &gpu);
        let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
        let source = HfqModelSource::from_hfq(hfq);
        // `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` keeps trunk layers 0..N resident
        // and host-maps the rest (a slab artifact larger than one card).
        let mut expert_vram_layers = None;
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
                    expert_vram_layers = Some(layers);
                    eprintln!(
                        "qwen4_wide_verify: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                    );
                }
                Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                    return Err(
                        "qwen4_wide_verify takes a fixed expert layer count, not auto".into(),
                    );
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
        let has_split_layers = !bundle.split_layers().is_empty();
        let has_split_digest = bundle.split_digest().is_some();
        let split_armed = has_split_layers && has_split_digest;
        if hot_budget_mib.is_some() {
            admit_pinned_split(offload_exec, has_split_layers, has_split_digest)?;
        }
        eprintln!("qwen4_wide_verify: offload_exec={offload_exec} split_armed={split_armed}");
        if let Some(digest) = &partition_digest {
            eprintln!("qwen4_wide_verify: partition_digest={digest}");
        }
        eprintln!("qwen4_wide_verify: split_digest {:?}", bundle.split_digest());
        eprintln!(
            "qwen4_wide_verify: loaded {} ({:.2} GB) on {} (vocab {vocab}), trunk overlay {}",
            path.display(),
            model_bytes as f64 / 1e9,
            gpu.arch,
            if overlay_applied { "applied" } else { "absent" }
        );
        let info = LoadInfo {
            arch: gpu.arch.to_string(),
            ple_line,
            ple_note: choice.note.clone(),
            ple_rows_source: format!("{} ({ple_rows_bytes} bytes)", ple_rows_path.display()),
            overlay_applied,
            model_bytes,
            uma,
            expert_vram_layers,
            offload_exec,
            partition_digest,
            split_armed,
        };
        Ok(Self {
            gpu,
            bundle,
            vocab,
            tokenizer,
            info,
        })
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

type StateDigest = Vec<(String, String)>;

/// Keys of the parts whose digests differ (a missing part counts).
fn digest_mismatches(want: &StateDigest, got: &StateDigest) -> Vec<String> {
    let mut keys = Vec::new();
    let want_map: BTreeMap<&str, &str> =
        want.iter().map(|(k, d)| (k.as_str(), d.as_str())).collect();
    let got_map: BTreeMap<&str, &str> = got.iter().map(|(k, d)| (k.as_str(), d.as_str())).collect();
    for (key, _) in want.iter().chain(got.iter()) {
        if want_map.get(key.as_str()) != got_map.get(key.as_str())
            && !keys.iter().any(|k: &String| k == key)
        {
            keys.push(key.clone());
        }
    }
    keys
}

#[derive(Default)]
struct RowCmp {
    bitwise_rows: usize,
    first_mismatch: Option<usize>,
    max_abs_diff: f64,
    mismatching: Vec<Value>,
    argmax_agree: usize,
}

/// Compare `n` rows of `got` against `reference` bit for bit.
fn compare_rows(reference: &[f32], got: &[f32], vocab: usize, n: usize) -> RowCmp {
    let mut cmp = RowCmp::default();
    for r in 0..n {
        let a = &reference[r * vocab..(r + 1) * vocab];
        let g = &got[r * vocab..(r + 1) * vocab];
        let mut differing = 0usize;
        let mut row_max = 0f64;
        for (x, y) in a.iter().zip(g) {
            if x.to_bits() != y.to_bits() {
                differing += 1;
                let d = (f64::from(*x) - f64::from(*y)).abs();
                let d = if d.is_nan() { f64::INFINITY } else { d };
                if d > row_max {
                    row_max = d;
                }
            }
        }
        if differing == 0 {
            cmp.bitwise_rows += 1;
        } else {
            if cmp.first_mismatch.is_none() {
                cmp.first_mismatch = Some(r);
            }
            cmp.mismatching.push(json!({
                "row": r,
                "differing_elements": differing,
                "max_abs_diff": row_max,
            }));
        }
        if row_max > cmp.max_abs_diff {
            cmp.max_abs_diff = row_max;
        }
        if argmax(a) == argmax(g) {
            cmp.argmax_agree += 1;
        }
    }
    cmp
}

fn median_min(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = sorted.first().copied().unwrap_or(f64::NAN);
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN);
    (median, min)
}

fn ms_json(values: &[f64]) -> Value {
    let (median, min) = median_min(values);
    json!({"median": median, "min": min, "all": values})
}

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

struct ArPass {
    tokens: Vec<u32>,
    /// Row-major `count * vocab`: row r after consuming `tokens[..=r]`.
    logits: Vec<f32>,
    /// Live-state digest after `j` tokens, for each requested `j`.
    digests: BTreeMap<usize, StateDigest>,
}

struct NOutcome {
    record: Value,
    /// Fully bitwise, every accept digest equal, no error.
    clean: bool,
    line: String,
}

struct Rig {
    m: Model,
    prompt: Vec<u32>,
    /// Final-row logits of the product prefill.
    prompt_logits: GpuTensor,
    /// One-row logits of `forward_token`.
    row: GpuTensor,
    seed_logits: Vec<f32>,
    /// Post-prefill state digest (`None` under `--trace`).
    d0: Option<StateDigest>,
    /// The resident state is no longer the post-prefill state.
    dirty: bool,
    p0: usize,
    reprefills: usize,
    rewind_checked: BTreeSet<&'static str>,
    iters: usize,
}

fn begin_arm_rewind_checks(checked: &mut BTreeSet<&'static str>) {
    // AR was checked before the arm loop; verify rewind is specific to each arm.
    checked.retain(|kind| *kind == "ar_ticket");
}

impl Rig {
    fn sync(&mut self) -> Result<(), String> {
        self.m.gpu.hip.device_synchronize().map_err(es)
    }

    fn digest(&mut self) -> Result<StateDigest, String> {
        oracle::dev_state_digest(&mut self.m.bundle, &mut self.m.gpu)
    }

    /// Start a prompt: product prefill, S0 digest (unless `--trace`).
    fn begin_prompt(&mut self, prompt: Vec<u32>, want_digest: bool) -> Result<(), String> {
        self.p0 = prompt.len();
        self.prompt = prompt;
        self.d0 = None;
        self.dirty = false;
        self.reprefills = 0;
        self.rewind_checked.clear();
        let started = Instant::now();
        self.prefill()?;
        eprintln!(
            "qwen4_wide_verify: prefilled {} prompt tokens in {:.2} s",
            self.p0,
            started.elapsed().as_secs_f64()
        );
        if want_digest {
            self.d0 = Some(self.digest()?);
        }
        Ok(())
    }

    /// Select a verify-width arm: set the runtime override (`None` = table/env)
    /// and return the verify rows the arch now resolves to.
    fn begin_arm(&mut self, arm: &Arm) -> usize {
        rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(arm.rows);
        begin_arm_rewind_checks(&mut self.rewind_checked);
        rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(&self.m.gpu.arch)
    }

    /// Launch mode just before a window: both flags must be false for an eager
    /// run (a recording replay or a hipGraph capture would change the route).
    fn exec_mode(&self) -> Value {
        let recording = self.m.gpu.replay.is_recording();
        let capture = self.m.gpu.graphs.capture_mode;
        json!({
            "replay_recording": recording,
            "graph_capture_mode": capture,
            "eager": !recording && !capture,
        })
    }

    /// The product prefill (`prefill_final`: reset, cold chunk schedule, final
    /// row only), then the seed logits.
    fn prefill(&mut self) -> Result<(), String> {
        self.m
            .bundle
            .prefill_final(&mut self.m.gpu, &self.prompt, 0, &self.prompt_logits)
            .map_err(|e| format!("prefill: {e}"))?;
        self.sync()?;
        self.seed_logits = self.m.gpu.download_f32(&self.prompt_logits).map_err(es)?;
        if self.seed_logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite prefill logits".into());
        }
        if oracle::dev_position(&self.m.bundle) != self.p0 {
            return Err(format!(
                "prefill ended at {}, expected {}",
                oracle::dev_position(&self.m.bundle),
                self.p0
            ));
        }
        Ok(())
    }

    /// Back at the post-prefill state S0 by re-prefilling, checked against the
    /// post-prefill digest.
    fn reprefill(&mut self) -> Result<(), String> {
        self.prefill()?;
        self.reprefills += 1;
        if let Some(d0) = self.d0.clone() {
            let now = self.digest()?;
            let bad = digest_mismatches(&d0, &now);
            if !bad.is_empty() {
                return Err(format!(
                    "re-prefill did not reproduce the post-prefill state (parts {bad:?}); rewind by re-prefill is not faithful"
                ));
            }
        }
        Ok(())
    }

    fn ensure_s0(&mut self) -> Result<(), String> {
        if self.dirty {
            self.reprefill()?;
            self.dirty = false;
        }
        Ok(())
    }

    /// After a ticket rewind: the first of each `kind` must reproduce S0.
    fn check_rewind(&mut self, kind: &'static str) -> Result<(), String> {
        if self.d0.is_none() || self.rewind_checked.contains(kind) {
            return Ok(());
        }
        let d0 = self.d0.clone().expect("checked above");
        let now = self.digest()?;
        let bad = digest_mismatches(&d0, &now);
        if !bad.is_empty() {
            self.dirty = true;
            return Err(format!(
                "{kind} rewind did not reproduce the post-prefill state (parts {bad:?})"
            ));
        }
        self.rewind_checked.insert(kind);
        Ok(())
    }

    /// `count` sequential `forward_token` steps from S0 under a snapshot
    /// ticket: fixed tokens, or (`fixed = None`) the greedy continuation from
    /// the prefill's argmax. Rewinds with the ticket.
    fn ar_pass(
        &mut self,
        fixed: Option<&[u32]>,
        count: usize,
        digest_counts: &BTreeSet<usize>,
    ) -> Result<ArPass, String> {
        self.ensure_s0()?;
        let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
        let result = self.ar_pass_steps(fixed, count, digest_counts);
        let restored = self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es);
        match (result, restored) {
            (Ok(pass), Ok(())) => {
                self.check_rewind("ar_ticket")?;
                Ok(pass)
            }
            (Err(error), restored) => {
                self.dirty = true;
                Err(match restored {
                    Ok(()) => error,
                    Err(restore) => format!("{error}; restore failed: {restore}"),
                })
            }
            (Ok(_), Err(restore)) => {
                self.dirty = true;
                Err(format!("AR pass restore failed: {restore}"))
            }
        }
    }

    fn ar_pass_steps(
        &mut self,
        fixed: Option<&[u32]>,
        count: usize,
        digest_counts: &BTreeSet<usize>,
    ) -> Result<ArPass, String> {
        let vocab = self.m.vocab;
        let mut tokens: Vec<u32> = match fixed {
            Some(fixed) => fixed.to_vec(),
            None => vec![argmax(&self.seed_logits)],
        };
        let mut logits = Vec::with_capacity(count * vocab);
        let mut digests = BTreeMap::new();
        for r in 0..count {
            self.m
                .bundle
                .forward_token(&mut self.m.gpu, tokens[r], &self.row, None)
                .map_err(|e| format!("AR forward_token row {r}: {e}"))?;
            let row = self.m.gpu.download_f32(&self.row).map_err(es)?;
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite AR logits at row {r}"));
            }
            if fixed.is_none() && r + 1 < count {
                tokens.push(argmax(&row));
            }
            logits.extend_from_slice(&row);
            if digest_counts.contains(&(r + 1)) {
                digests.insert(r + 1, self.digest()?);
            }
        }
        Ok(ArPass {
            tokens,
            logits,
            digests,
        })
    }

    /// Verify (production transaction), then roll back with its own ticket.
    fn verify_logits_rolled_back(
        &mut self,
        block: &[u32],
        scratch: &mut dyn SpecScratch,
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        let got = oracle::dev_verify_logits(&mut self.m.bundle, &mut self.m.gpu, block, scratch);
        match got {
            Ok(got) => {
                oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
                    .inspect_err(|_| self.dirty = true)?;
                self.check_rewind("verify_ticket")?;
                Ok(got)
            }
            Err(error) => {
                self.dirty = true;
                Err(error)
            }
        }
    }

    fn run_n(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
        accepts: Accepts,
    ) -> Result<NOutcome, String> {
        self.ensure_s0()?;
        let mut scratch = self
            .m
            .bundle
            .new_spec_scratch(&mut self.m.gpu, n)
            .map_err(|e| format!("new_spec_scratch({n}): {e}"))?;
        let outcome = self.run_n_body(n, block, ar, accepts, &mut *scratch);
        scratch.free(&mut self.m.gpu);
        outcome
    }

    fn run_n_body(
        &mut self,
        n: usize,
        block: &[u32],
        ar: &ArPass,
        accepts: Accepts,
        scratch: &mut dyn SpecScratch,
    ) -> Result<NOutcome, String> {
        let vocab = self.m.vocab;
        let ring_rows = oracle::dev_row_capture_rows(&self.m.bundle);
        let armed = oracle::dev_would_arm(&self.m.bundle, n);

        // 1. Verify logits vs AR, bitwise per row.
        let (picks, logits) = self.verify_logits_rolled_back(block, scratch)?;
        if picks.len() != n || logits.len() != n * vocab {
            return Err(format!(
                "verify returned {} picks / {} logits for {n} rows",
                picks.len(),
                logits.len()
            ));
        }
        let cmp = compare_rows(&ar.logits, &logits, vocab, n);
        let ar_argmax: Vec<u32> = (0..n)
            .map(|r| argmax(&ar.logits[r * vocab..(r + 1) * vocab]))
            .collect();
        let picks_match = picks.iter().zip(&ar_argmax).filter(|(p, a)| p == a).count();
        let bitwise_all = cmp.bitwise_rows == n;

        // 2. Post-commit state vs the AR state after accept_len + 1 tokens.
        let accepts: Vec<usize> = match (accepts, n) {
            (Accepts::None, _) => Vec::new(),
            (Accepts::All, _) => (0..n).collect(),
            (Accepts::Ends, 1) => vec![0],
            (Accepts::Ends, _) => vec![0, n - 1],
        };
        let mut accept_json = Vec::new();
        let mut state_ok_all: Option<bool> = if accepts.is_empty() { None } else { Some(true) };
        let mut accept_cells = Vec::new();
        let mut verify_repeatable = true;
        let mut exec_verify = Value::Null;
        for &accept_len in &accepts {
            self.ensure_s0()?;
            let p0 = oracle::dev_position(&self.m.bundle);
            let (_, again) = match oracle::dev_verify_logits(
                &mut self.m.bundle,
                &mut self.m.gpu,
                block,
                scratch,
            ) {
                Ok(got) => got,
                Err(error) => {
                    self.dirty = true;
                    return Err(format!("accept {accept_len}: verify: {error}"));
                }
            };
            self.dirty = true;
            if again
                .iter()
                .zip(&logits)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                verify_repeatable = false;
            }
            self.m
                .bundle
                .commit_prefix(&mut self.m.gpu, block, accept_len, p0, scratch)
                .map_err(|e| format!("accept {accept_len}: commit_prefix: {e}"))?;
            let position = oracle::dev_position(&self.m.bundle);
            let live = self.digest()?;
            let want = ar
                .digests
                .get(&(accept_len + 1))
                .ok_or_else(|| format!("no AR digest after {} tokens", accept_len + 1))?;
            let bad = digest_mismatches(want, &live);
            let ok = bad.is_empty() && position == p0 + accept_len + 1;
            state_ok_all = state_ok_all.map(|all| all && ok);
            accept_cells.push(format!("{accept_len}:{}", if ok { "Y" } else { "N" }));
            accept_json.push(json!({
                "accept_len": accept_len,
                "state_ok": ok,
                "position_after": position,
                "expected_position": p0 + accept_len + 1,
                "mismatched_parts": bad,
            }));
        }

        // 3. Timing: verify (resident, no digest, no pick readback) and the
        // n forward_token steps; restore excluded.
        self.ensure_s0()?;
        let mut verify_ms = Vec::new();
        let mut ar_ms = Vec::new();
        let mut exec_ar = Value::Null;
        for it in 0..=self.iters {
            self.sync()?;
            if it == 1 {
                exec_verify = self.exec_mode();
            }
            let start = Instant::now();
            oracle::dev_verify_resident(&mut self.m.bundle, &mut self.m.gpu, block, scratch)
                .inspect_err(|_| self.dirty = true)?;
            self.sync()?;
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, scratch)
                .inspect_err(|_| self.dirty = true)?;
            if it > 0 {
                verify_ms.push(elapsed);
            }
        }
        for it in 0..=self.iters {
            let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
            self.sync()?;
            if it == 1 {
                exec_ar = self.exec_mode();
            }
            let start = Instant::now();
            let mut step_error = None;
            for &token in block {
                if let Err(error) =
                    self.m
                        .bundle
                        .forward_token(&mut self.m.gpu, token, &self.row, None)
                {
                    step_error = Some(error.to_string());
                    break;
                }
            }
            let synced = self.sync();
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es)?;
            if let Some(error) = step_error {
                return Err(format!("timed AR forward_token: {error}"));
            }
            synced?;
            if it > 0 {
                ar_ms.push(elapsed);
            }
        }
        let (verify_med, _) = median_min(&verify_ms);
        let (ar_med, _) = median_min(&ar_ms);

        let first = cmp
            .first_mismatch
            .map_or_else(|| "-".to_string(), |r| r.to_string());
        let line = format!(
            "rows {n}: bitwise {}/{n} first_mismatch {first} max|diff| {:.3e} argmax {}/{n} state_ok[accept={}] verify_ms {verify_med:.3} ar_ms {ar_med:.3}",
            cmp.bitwise_rows,
            cmp.max_abs_diff,
            cmp.argmax_agree,
            if accept_cells.is_empty() {
                "none".to_string()
            } else {
                accept_cells.join(",")
            },
        );
        let record = json!({
            "n": n,
            "row_capture_armed": armed,
            "row_capture_ring_rows": ring_rows,
            "rows_bitwise": cmp.bitwise_rows,
            "all_rows_bitwise": bitwise_all,
            "first_mismatch_row": cmp.first_mismatch,
            "max_abs_diff": cmp.max_abs_diff,
            "mismatching_rows": cmp.mismatching,
            "argmax_agree": cmp.argmax_agree,
            "picks": picks,
            "ar_argmax": ar_argmax,
            "picks_match_ar_argmax": picks_match,
            "verify_repeatable": verify_repeatable,
            "accepts": accept_json,
            "state_ok": state_ok_all,
            "execution": {"ar": exec_ar, "verify": exec_verify},
            "verify_ms": ms_json(&verify_ms),
            "ar_ms": ms_json(&ar_ms),
        });
        Ok(NOutcome {
            record,
            clean: bitwise_all && state_ok_all.unwrap_or(true),
            line,
        })
    }

    /// `--trace` window pair for one n: the AR reference steps (delimiter
    /// twice back to back, then `n` `forward_token`s) and the verify (one
    /// delimiter, then the production verify). Restore/rollback copies land
    /// between the last forward kernel of a window and the next delimiter.
    fn trace_n(&mut self, n: usize, block: &[u32], delimiter: &GpuTensor) -> Result<Value, String> {
        self.ensure_s0()?;
        let mut scratch = self
            .m
            .bundle
            .new_spec_scratch(&mut self.m.gpu, n)
            .map_err(|e| format!("new_spec_scratch({n}): {e}"))?;
        let armed = oracle::dev_would_arm(&self.m.bundle, n);
        let mut exec_ar = Value::Null;
        let mut exec_verify = Value::Null;
        let result = (|| -> Result<(), String> {
            // AR window.
            let ticket = self.m.bundle.snapshot(&mut self.m.gpu).map_err(es)?;
            self.sync()?;
            exec_ar = self.exec_mode();
            let mut ar_error = None;
            let launched = (|| -> Result<(), String> {
                for _ in 0..2 {
                    self.m
                        .gpu
                        .logit_softcap_f32(delimiter, delimiter.numel(), 30.0)
                        .map_err(es)?;
                }
                for &token in block {
                    self.m
                        .bundle
                        .forward_token(&mut self.m.gpu, token, &self.row, None)
                        .map_err(es)?;
                }
                Ok(())
            })();
            if let Err(error) = launched {
                ar_error = Some(error);
            }
            let synced = self.sync();
            self.m.bundle.restore(&mut self.m.gpu, ticket).map_err(es)?;
            if let Some(error) = ar_error {
                return Err(format!("trace AR window: {error}"));
            }
            synced?;
            // Verify window.
            self.sync()?;
            exec_verify = self.exec_mode();
            self.m
                .gpu
                .logit_softcap_f32(delimiter, delimiter.numel(), 30.0)
                .map_err(es)?;
            oracle::dev_verify_resident(&mut self.m.bundle, &mut self.m.gpu, block, &mut *scratch)?;
            self.sync()?;
            oracle::dev_rollback_verify(&mut self.m.bundle, &mut self.m.gpu, &mut *scratch)?;
            self.sync()
        })();
        scratch.free(&mut self.m.gpu);
        if result.is_err() {
            self.dirty = true;
        }
        result?;
        Ok(json!({
            "n": n,
            "row_capture_armed": armed,
            "execution": {"ar": exec_ar, "verify": exec_verify},
        }))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Block tokens, the AR reference and whether two AR passes agree bitwise.
/// One fixed-token AR pass covers every n (AR row r does not depend on the
/// block length) and is independent of the verify-width arm (AR rows are
/// single-row forwards), so it is computed once per prompt.
fn reference_pass(
    rig: &mut Rig,
    args: &Args,
    max_rows: usize,
    digest_counts: &BTreeSet<usize>,
) -> Result<(Vec<u32>, ArPass, bool), String> {
    match args.block {
        BlockMode::Greedy => {
            let feed = rig.ar_pass(None, max_rows, &BTreeSet::new())?;
            let block = feed.tokens.clone();
            let reference = rig.ar_pass(Some(&block), max_rows, digest_counts)?;
            let repeatable = feed
                .logits
                .iter()
                .zip(&reference.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            Ok((block, reference, repeatable))
        }
        BlockMode::Prompt => {
            let block = block_tokens(rig, args, max_rows)?;
            let reference = rig.ar_pass(Some(&block), max_rows, digest_counts)?;
            let second = rig.ar_pass(Some(&block), max_rows, &BTreeSet::new())?;
            let repeatable = second
                .logits
                .iter()
                .zip(&reference.logits)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            Ok((block, reference, repeatable))
        }
    }
}

/// Warn when a `wideN` arm did not change the verify rows (non-gfx1151/1201).
fn note_arm(arm: &Arm, effective: usize) {
    let asked = arm
        .rows
        .map(|n| n.clamp(FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS));
    match asked {
        Some(asked) if asked != effective => eprintln!(
            "qwen4_wide_verify: WARNING: arm {} asked for {asked} verify rows but the arch resolves {effective} (override applies on gfx1151/gfx1201 only)",
            arm.name
        ),
        _ => eprintln!(
            "qwen4_wide_verify: arm {}: verify rows {effective}",
            arm.name
        ),
    }
}

fn run() -> Result<bool, String> {
    let args = parse_args()?;
    let max_rows = *args.rows.iter().max().expect("rows is non-empty");
    let mut prompt_texts = Vec::new();
    for path in &args.prompt_files {
        prompt_texts.push(
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?,
        );
    }

    let mut m = Model::load(&args.model, args.n_ctx, args.hot_budget_mib, args.hotset.as_deref())?;
    let prompts: Vec<Vec<u32>> = prompt_texts.iter().map(|t| m.tokenizer.encode(t)).collect();
    for (path, prompt) in args.prompt_files.iter().zip(&prompts) {
        if prompt.is_empty() {
            return Err(format!("{} encodes to zero tokens", path.display()));
        }
        if prompt.len() + max_rows + 1 > args.n_ctx {
            return Err(format!(
                "{}: {} prompt tokens + {max_rows} block rows exceeds --n-ctx {}",
                path.display(),
                prompt.len(),
                args.n_ctx
            ));
        }
    }
    let vocab = m.vocab;
    let ring_supported = oracle::dev_row_capture_supported(&m.bundle, &m.gpu);
    if args.row_capture {
        oracle::dev_arm_row_capture(&mut m.bundle, &mut m.gpu, max_rows)?;
    }
    let ring_rows = oracle::dev_row_capture_rows(&m.bundle);
    if !args.row_capture && ring_rows != 0 {
        return Err("--row-capture off requested but a capture ring is already allocated".into());
    }
    eprintln!(
        "qwen4_wide_verify: GDN row-capture ring {ring_rows} rows (route supports capture: {ring_supported})"
    );
    // Size the bundle's spec hidden rows once for the widest n, as production
    // MTP does (`new_spec_scratch(max_k + 1)`): without an NPU pipelined trunk
    // the first (n = 1) scratch would otherwise pin the capacity at one row.
    let sizing = m
        .bundle
        .new_spec_scratch(&mut m.gpu, max_rows)
        .map_err(|e| format!("new_spec_scratch({max_rows}): {e}"))?;
    sizing.free(&mut m.gpu);
    let prompt_logits = m.gpu.zeros(&[vocab], DType::F32).map_err(es)?;
    let row = m.gpu.zeros(&[vocab], DType::F32).map_err(es)?;
    let mut rig = Rig {
        m,
        prompt: Vec::new(),
        prompt_logits,
        row,
        seed_logits: Vec::new(),
        d0: None,
        dirty: false,
        p0: 0,
        reprefills: 0,
        rewind_checked: BTreeSet::new(),
        iters: args.iters,
    };

    let hipfire_env: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .collect();
    let block_mode = if args.block == BlockMode::Greedy {
        "greedy"
    } else {
        "prompt"
    };
    let model_path = args.model.display().to_string();
    let arm_names: Vec<&str> = args.arms.iter().map(|a| a.name.as_str()).collect();
    let prompt_paths: Vec<String> = args
        .prompt_files
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    eprintln!("qwen4_wide_verify: HIPFIRE_* env: {hipfire_env:?}");
    let mut identity = json!({
        "gpu_arch": rig.m.info.arch,
        "uma": rig.m.info.uma,
        "ple": rig.m.info.ple_line,
        "ple_note": rig.m.info.ple_note,
        "ple_rows_source": rig.m.info.ple_rows_source,
        "trunk_overlay_applied": rig.m.info.overlay_applied,
        "expert_vram_layers": rig.m.info.expert_vram_layers,
        "hot_budget_mib": args.hot_budget_mib,
        "hotset": args.hotset.as_ref().map(|p| p.display().to_string()),
        "offload_exec": rig.m.info.offload_exec.to_string(),
        "partition_digest": rig.m.info.partition_digest,
        "split_armed": rig.m.info.split_armed,
        "split_digest": rig.m.bundle.split_digest(),
        "model_path": model_path,
        "model_bytes": rig.m.info.model_bytes,
        "hipfire_env": hipfire_env,
        "prompt_files": prompt_paths,
        "vocab": vocab,
        "n_ctx": args.n_ctx,
        "iters": args.iters,
        "block_mode": block_mode,
        "accepts": args.accepts.name(),
        "arms": arm_names,
        "rows": args.rows,
        "row_capture_supported": ring_supported,
        "row_capture_ring_rows": ring_rows,
        "row_capture_requested": args.row_capture,
        "accepted_prefix_semantics": "accept_len in 0..n accepted drafts; seed-inclusive consumed prefix lengths 1..=n",
        "verify_route": "Qwen4Bundle::verify_block_rows via SpecTarget (spec_forward_rows, prefill_scope=false); timing uses the resident (no pick readback) variant",
    });

    let prompt_header = |index: usize, tokens: &[u32]| -> Value {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        json!({
            "prompt_index": index,
            "prompt_file": args.prompt_files[index].display().to_string(),
            "prompt_tokens": tokens.len(),
            "prompt_tokens_sha256": format!("{:x}", Sha256::digest(&bytes)),
        })
    };

    // -- trace mode ---------------------------------------------------------
    if args.trace {
        let delimiter = rig.m.gpu.zeros(&[256], DType::F32).map_err(es)?;
        eprintln!(
            "qwen4_wide_verify: trace delimiter kernel: {DELIMITER_KERNEL} (256 f32, 1 block, cap 30.0); per prompt x arm x n: AR window = 2 back-to-back delimiters then n forward_token; verify window = 1 delimiter then the verify"
        );
        // Compile the delimiter kernel outside every window.
        rig.m
            .gpu
            .logit_softcap_f32(&delimiter, delimiter.numel(), 30.0)
            .map_err(es)?;
        rig.sync()?;
        let mut windows = Vec::new();
        let mut prompt_records = Vec::new();
        let mut failed = false;
        for (pi, prompt) in prompts.iter().enumerate() {
            rig.begin_prompt(prompt.clone(), false)?;
            let block = block_tokens(&mut rig, &args, max_rows)?;
            let mut header = prompt_header(pi, prompt);
            header["block_tokens"] = json!(block);
            prompt_records.push(header);
            for arm in &args.arms {
                let effective = rig.begin_arm(arm);
                note_arm(arm, effective);
                for &n in &args.rows {
                    let mut label = json!({
                        "prompt_index": pi,
                        "prompt_tokens": prompt.len(),
                        "arm": arm.name,
                        "effective_verify_rows": effective,
                    });
                    match rig.trace_n(n, &block[..n], &delimiter) {
                        Ok(window) => {
                            println!(
                                "trace prompt {pi} arm {} rows {n}: armed {}",
                                arm.name, window["row_capture_armed"]
                            );
                            for (k, v) in window.as_object().expect("window is an object") {
                                label[k] = v.clone();
                            }
                        }
                        Err(error) => {
                            eprintln!(
                                "qwen4_wide_verify: trace prompt {pi} arm {} rows {n}: {error}",
                                arm.name
                            );
                            label["n"] = json!(n);
                            label["error"] = json!(error);
                            failed = true;
                        }
                    }
                    windows.push(label);
                }
            }
        }
        rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(None);
        identity["mode"] = json!("trace");
        identity["delimiter_kernel"] = json!(DELIMITER_KERNEL);
        identity["trace_pattern"] = json!({
            "delimiter_kernel": DELIMITER_KERNEL,
            "order": "for prompt in prompts { for arm in arms { for n in rows { AR window, then verify window } } }; each trace_windows entry is one (prompt, arm, n) in this order",
            "ar_window": "DELIM, DELIM (back to back), then n forward_token calls",
            "verify_window": "DELIM once, then the production verify (resident, no pick readback)",
            "label_rule": "a delimiter run of length 2 starts 'ar n=N'; a run of length 1 starts 'verify n=N'; windows are consecutive: ar, verify, ar, verify, ...",
            "between_windows": "snapshot/restore and rollback copies sit between the last forward kernel of a window and the next delimiter; they belong to no forward",
            "windows_per_entry": 2,
        });
        let report = json!({
            "identity": identity,
            "prompts": prompt_records,
            "trace_windows": windows,
        });
        std::fs::write(
            &args.out,
            serde_json::to_string_pretty(&report).map_err(es)?,
        )
        .map_err(|e| format!("write {}: {e}", args.out.display()))?;
        return Ok(!failed);
    }

    // -- accuracy + timing mode ---------------------------------------------
    let mut digest_counts: BTreeSet<usize> = BTreeSet::new();
    if args.accepts != Accepts::None {
        digest_counts.insert(1);
        for &n in &args.rows {
            if args.accepts == Accepts::All {
                digest_counts.extend(1..=n);
            } else {
                digest_counts.insert(n);
            }
        }
    }
    let mut ok = true;
    let mut prompt_records = Vec::new();
    for (pi, prompt) in prompts.iter().enumerate() {
        rig.begin_prompt(prompt.clone(), true)?;
        let (block, ar, ar_repeatable) = reference_pass(&mut rig, &args, max_rows, &digest_counts)?;
        if !ar_repeatable {
            eprintln!(
                "qwen4_wide_verify: WARNING: two AR passes over the same block differ bitwise"
            );
        }
        eprintln!("qwen4_wide_verify: prompt {pi} block tokens {block:?}");
        let mut arm_records = Vec::new();
        for arm in &args.arms {
            let effective = rig.begin_arm(arm);
            note_arm(arm, effective);
            let reprefills_before = rig.reprefills;
            let mut records = Vec::new();
            for &n in &args.rows {
                let tag = format!("[prompt {pi} tokens {} arm {}]", prompt.len(), arm.name);
                match rig.run_n(n, &block[..n], &ar, args.accepts) {
                    Ok(outcome) => {
                        println!("{tag} {}", outcome.line);
                        if n <= 8 && !outcome.clean {
                            ok = false;
                        }
                        let mut record = outcome.record;
                        record["arm"] = json!(arm.name);
                        record["prompt_index"] = json!(pi);
                        records.push(record);
                    }
                    Err(error) => {
                        println!("{tag} rows {n}: ERROR {error}");
                        rig.dirty = true;
                        if n <= 8 {
                            ok = false;
                        }
                        records.push(json!({
                            "n": n,
                            "arm": arm.name,
                            "prompt_index": pi,
                            "error": error,
                        }));
                    }
                }
            }
            arm_records.push(json!({
                "arm": arm.name,
                "override_rows": arm.rows,
                "effective_verify_rows": effective,
                "reprefills": rig.reprefills - reprefills_before,
                "checked_ticket_kinds": rig.rewind_checked.iter().collect::<Vec<_>>(),
                "records": records,
            }));
        }
        let mut header = prompt_header(pi, prompt);
        header["block_tokens"] = json!(block);
        header["ar_repeatable_bitwise"] = json!(ar_repeatable);
        header["post_prefill_state_parts"] = json!(rig.d0.as_ref().map_or(0, Vec::len));
        header["ar_reference_digest_counts"] = json!(digest_counts);
        header["reprefill_count"] = json!(rig.reprefills);
        header["arms"] = json!(arm_records);
        prompt_records.push(header);
    }
    rdna_compute::prefill_invariant::set_qwen4_wide_verify_rows_override(None);

    identity["mode"] = json!("oracle");
    identity["rewind"] = json!({
        "ar_passes_and_timed_arms": "snapshot/restore ticket (Qwen4Bundle::snapshot / restore)",
        "verify_arms": "the verify's own snapshot ticket (rolled back without commit)",
        "accept_arms": "re-prefill (prefill_final); a committed state has no ticket (none with --accepts none)",
        "re_prefill_checked_against_post_prefill_digest": true,
    });
    identity["shared_ar_reference"] = json!("one AR reference per prompt, shared by every arm (AR rows are single-row forwards, independent of the verify-width arm); AR timing is re-measured per arm and n");
    let report = json!({"identity": identity, "prompts": prompt_records, "ok": ok});
    std::fs::write(
        &args.out,
        serde_json::to_string_pretty(&report).map_err(es)?,
    )
    .map_err(|e| format!("write {}: {e}", args.out.display()))?;
    eprintln!("qwen4_wide_verify: wrote {}", args.out.display());
    Ok(ok)
}

/// The block tokens: `greedy` = the AR greedy continuation from the prefill
/// (block[0] = argmax of the prefill's final row); `prompt` = the last
/// `max_rows` prompt tokens (cycled when the prompt is shorter).
fn block_tokens(rig: &mut Rig, args: &Args, max_rows: usize) -> Result<Vec<u32>, String> {
    match args.block {
        BlockMode::Greedy => Ok(rig.ar_pass(None, max_rows, &BTreeSet::new())?.tokens),
        BlockMode::Prompt => {
            let len = rig.prompt.len();
            Ok((0..max_rows)
                .map(|i| {
                    if len >= max_rows {
                        rig.prompt[len - max_rows + i]
                    } else {
                        rig.prompt[i % len]
                    }
                })
                .collect())
        }
    }
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("qwen4_wide_verify: FAILED the n <= 8 contract (or a run error)");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("qwen4_wide_verify: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_rewind_checks_preserve_only_completed_reference_checks() {
        let mut checked = BTreeSet::from(["ar_ticket", "verify_ticket"]);
        begin_arm_rewind_checks(&mut checked);
        assert_eq!(checked, BTreeSet::from(["ar_ticket"]));
        let mut unchecked = BTreeSet::from(["verify_ticket"]);
        begin_arm_rewind_checks(&mut unchecked);
        assert!(unchecked.is_empty(), "never invent an AR rewind check");
    }
}
