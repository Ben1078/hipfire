// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_slowlink_gate — the S1 slow-link layer-major prefill gate harness.
//!
//! Frozen CLI (every flag required, every path absolute):
//!
//! ```text
//! qwen4_slowlink_gate --model ABS --fixture ABS.json --output ABS.json
//!     --mode primitives|state|product|perf --route incumbent|layer-major
//!     --manifest ABS.json
//! ```
//!
//! `--route` is a diagnostic force inside the same production selector and
//! safety predicate (`HIPFIRE_QWEN4_SLOWLINK_PREFILL=0|1`), not an alternate
//! implementation: a forced `layer-major` that the selector bypasses is a
//! failing check naming the bypass reason, never a silent incumbent run.
//!
//! The fixture (`hipfire.qwen4.slowlink_fixture.v1`) records the exact
//! prompt ids (512 / 1536 / 1537 / 8192 / 16384 rows, an irregular prompt
//! with original launch cuts, a resumed prompt with its entry prefix, a
//! boundary prompt with its requested boundaries and a natural-text prompt),
//! the divergent continuation ids and the named failure injections. The
//! manifest pins source base and tip, binary SHA-256, model identity, PLE
//! source/format/sidecars, kernel code-object hashes, host/device/link and
//! the memory ledger; the binary hash and model size are re-observed and a
//! mismatch fails the run before any GPU work.
//!
//! The output records the actual route, window and chunk manifest of every
//! request (the bundle's `slowlink_receipt`), fixed-state and final-logit
//! digests, generated ids and decoded text, useful/submitted bytes and the
//! copy/compute/barrier/non-expert spans, peak and resident bytes and the
//! status of EVERY required check. A required check that was not executed
//! makes the run `incomplete`, never `pass`; the exit status is 0 only for
//! `pass`.
//!
//! Modes:
//! * `primitives` — the production panel transport, dispatch and ordered
//!   combine self-test (payload/table identity, repeated epochs at the same
//!   addresses, delayed readers, within-expert panels with canaries, exact
//!   canonical combine) on the loaded bundle's admitted S1 profile.
//! * `state` — real prompts through the production prefill: route receipts
//!   (residency, tile, charged lookahead, one sweep per window, exactly one
//!   window at 16384 rows), same-route repeat digests of every fixed state,
//!   resume and launch-cut chronology, session restore and divergent
//!   continuation at the requested boundaries, named failure injections with
//!   restore/containment and a clean re-run, bounded head scoring of the
//!   last rows, and eager versus retained-body replay decode after the
//!   prefill.
//! * `product` — greedy generation with decoded text for review (attractor
//!   heuristics reject degenerate output; a human still reads the text).
//! * `perf` — fresh-process timing of the 16384-row prefill plus the
//!   pp512/pp8192 guards and decode, with the span attribution and the
//!   cross-layer event-overlap evidence.
//!
//! Decode "graph" proof: Qwen4 production has no hipGraph decode route
//! (`gpu_forward.rs`: no Qwen4 production path opens a hipGraph capture); its
//! retained body replay (`ReplayController`: first eligible forward records,
//! later ones replay) is the decode body replay, so that is the route proved
//! after the S1 prefill. A route that does not reach `Ready` or replay every
//! measured token fails the check.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::state_parity::{
    slowlink_load_bundle, slowlink_logit_rows_summary, slowlink_state_digest,
};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::session_cache::{SessionCache, SessionCacheUsage, SessionRoute};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[path = "../tests/common/slowlink_receipt.rs"]
mod slowlink_receipt;
use slowlink_receipt::*;

const USAGE: &str = "usage: qwen4_slowlink_gate --model ABS --fixture ABS.json --output ABS.json --mode primitives|state|product|perf --route incumbent|layer-major --manifest ABS.json";
const ROUTE_ENV: &str = "HIPFIRE_QWEN4_SLOWLINK_PREFILL";
const REPORT_SCHEMA: &str = "hipfire.qwen4.slowlink_gate.v1";
/// Greedy ids compared / reported for the restore and replay checks.
const RESTORE_DECODE: usize = 16;
/// Retained replay needs at most this many warm tokens before `Ready`.
const REPLAY_WARM_LIMIT: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Primitives,
    State,
    Product,
    Perf,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Primitives => "primitives",
            Self::State => "state",
            Self::Product => "product",
            Self::Perf => "perf",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    fixture: PathBuf,
    output: PathBuf,
    mode: Mode,
    route: Expect,
    manifest: PathBuf,
}

fn absolute(flag: &str, value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{flag} must be an absolute path"));
    }
    Ok(path)
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut args = args.into_iter();
    let (mut model, mut fixture, mut output, mut manifest) = (None, None, None, None);
    let (mut mode, mut route) = (None, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(absolute("--model", value()?)?),
            "--fixture" => fixture = Some(absolute("--fixture", value()?)?),
            "--output" => output = Some(absolute("--output", value()?)?),
            "--manifest" => manifest = Some(absolute("--manifest", value()?)?),
            "--mode" => {
                mode = Some(match value()?.as_str() {
                    "primitives" => Mode::Primitives,
                    "state" => Mode::State,
                    "product" => Mode::Product,
                    "perf" => Mode::Perf,
                    other => {
                        return Err(format!(
                            "--mode {other:?} is not primitives|state|product|perf"
                        ))
                    }
                })
            }
            "--route" => {
                route = Some(match value()?.as_str() {
                    "incumbent" => Expect::Incumbent,
                    "layer-major" => Expect::LayerMajor,
                    other => return Err(format!("--route {other:?} is not incumbent|layer-major")),
                })
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let args = Args {
        model: model.ok_or("--model is required")?,
        fixture: fixture.ok_or("--fixture is required")?,
        output: output.ok_or("--output is required")?,
        mode: mode.ok_or("--mode is required")?,
        route: route.ok_or("--route is required")?,
        manifest: manifest.ok_or("--manifest is required")?,
    };
    if args.mode == Mode::Primitives && args.route == Expect::Incumbent {
        return Err(
            "--mode primitives exercises the layer-major panel code; it has no incumbent arm"
                .into(),
        );
    }
    Ok(args)
}

// ── helpers ────────────────────────────────────────────────────────────

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn file_sha256(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha_u32(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// A generated sequence that fell into a short loop or collapsed to a couple
/// of tokens. A heuristic only: the decoded text is still for human review.
fn attractor(ids: &[u32]) -> Option<String> {
    if ids.len() < 8 {
        return None;
    }
    let distinct: std::collections::BTreeSet<u32> = ids.iter().copied().collect();
    if distinct.len() <= 2 {
        return Some(format!(
            "only {} distinct token(s) in {}",
            distinct.len(),
            ids.len()
        ));
    }
    for period in 1..=4usize {
        if ids.len() >= period * 8 {
            let tail = &ids[ids.len() - period * 8..];
            if tail.chunks(period).all(|c| c == &tail[..period]) {
                return Some(format!(
                    "last {} tokens repeat with period {period}",
                    period * 8
                ));
            }
        }
    }
    None
}

fn gpu_error<T>(result: Result<T, impl std::fmt::Display>, what: &str) -> Result<T, String> {
    result.map_err(|e| format!("{what}: {e}"))
}

fn sync(gpu: &Gpu) -> Result<(), String> {
    gpu_error(gpu.hip.device_synchronize(), "device synchronize")
}

/// `SessionCache::usage` as JSON, plus the summed device footprint.
fn usage_value(usage: &SessionCacheUsage) -> Value {
    let device_total = [
        usage.pool_bytes,
        usage.pending_bytes,
        usage.resident_bytes,
        usage.deferred_release_bytes,
        usage.parked_bytes,
    ]
    .into_iter()
    .fold(0u64, u64::saturating_add);
    json!({
        "pool_bytes": usage.pool_bytes, "budget_bytes": usage.budget_bytes,
        "pending_bytes": usage.pending_bytes, "resident_bytes": usage.resident_bytes,
        "deferred_release_bytes": usage.deferred_release_bytes, "parked_bytes": usage.parked_bytes,
        "evictions_total": usage.evictions_total, "device_total_bytes": device_total,
    })
}

/// Current cache occupancy and eviction counter; `Null` without a cache.
fn usage_json(cache: Option<&SessionCache>) -> Value {
    cache.map_or(Value::Null, |cache| usage_value(&cache.usage()))
}

/// Per-request cache evidence: occupancy before and after and the ACTUAL
/// evictions between them (the cache's own counter delta, never derived from
/// the occupancy change). `Null` delta when either side has no counter.
fn cache_evidence(before: &Value, after: &Value) -> Value {
    let delta = match (
        before["evictions_total"].as_u64(),
        after["evictions_total"].as_u64(),
    ) {
        (Some(b), Some(a)) => a.checked_sub(b).map_or(Value::Null, |d| json!(d)),
        _ => Value::Null,
    };
    json!({"before": before, "after": after, "evictions_delta": delta})
}

/// How the gate's phase samples are taken, recorded in every sample.
const SAMPLE_METHOD: &str = "point sample at a gate phase boundary: hipMemGetInfo (total-free) + /proc/self/status + smaps clean regular-file RSS + /proc/self/fdinfo amdgpu drm-memory-vram/gtt; observed point, not a continuous driver peak";
/// The light diagnostic sample of the non-primitive cells (no smaps).
const SAMPLE_METHOD_LIGHT: &str = "light point diagnostic at a gate phase boundary: hipMemGetInfo (total-free) + /proc/self/status + /proc/self/fdinfo amdgpu drm-memory-vram/gtt; no smaps, no sampler thread; not judged, not a continuous driver peak";

/// One labeled phase sample of this process's memory. Taken only at gate
/// phase boundaries (never inside the production prefill loops).
fn mem_sample(gpu: &Gpu, label: &str, smaps: bool) -> Value {
    let device = gpu
        .hip
        .get_vram_info()
        .ok()
        .map(|(free, total)| (free as u64, total as u64));
    let method = if smaps {
        SAMPLE_METHOD
    } else {
        SAMPLE_METHOD_LIGHT
    };
    counters_json(label, method, &read_proc_counters(smaps), device)
}

/// The sampler reads the (costly) `/proc/self/smaps` on every this-many reads.
const SAMPLER_SMAPS_EVERY: u64 = 20;

/// Background `/proc` sampler: running maxima of discrete points (process
/// status and fdinfo only; it never queries HIP).
struct ProcSampler {
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<(u64, ProcCounters)>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ProcSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new((0u64, ProcCounters::default())));
        let (thread_stop, thread_state) = (Arc::clone(&stop), Arc::clone(&state));
        let handle = std::thread::spawn(move || {
            let mut tick = 0u64;
            while !thread_stop.load(Ordering::Relaxed) {
                let counters = read_proc_counters(tick % SAMPLER_SMAPS_EVERY == 0);
                tick += 1;
                {
                    let mut guard = thread_state.lock().unwrap_or_else(|e| e.into_inner());
                    guard.0 += 1;
                    guard.1.merge_max(&counters);
                }
                std::thread::sleep(Duration::from_millis(SAMPLER_PERIOD_MS));
            }
        });
        Self {
            stop,
            state,
            handle: Some(handle),
        }
    }

    /// Stop, take one last point and return the maxima as a sample.
    fn finish(mut self) -> Value {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let last = read_proc_counters(true);
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.0 += 1;
        guard.1.merge_max(&last);
        let method = format!("background thread: /proc/self/status + fdinfo every {SAMPLER_PERIOD_MS} ms (plus smaps, the clean regular-file RSS, every {SAMPLER_SMAPS_EVERY}th read and on the final read); running maxima of discrete points, no hipMemGetInfo, not a continuous peak");
        let mut sample = counters_json(SAMPLER_LABEL, &method, &guard.1, None);
        sample["samples"] = json!(guard.0);
        sample
    }
}

struct Ctx {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    logits: GpuTensor,
    vocab: usize,
    eos: u32,
    fixture: Fixture,
    route: Expect,
    /// Labeled phase memory samples ([`mem_sample`]).
    mem: Vec<Value>,
}

impl Ctx {
    /// A heavy (smaps) phase sample; only the primitive cell takes these.
    fn sample_memory(&mut self, label: &str) {
        let sample = mem_sample(&self.gpu, label, true);
        self.mem.push(sample);
    }

    fn cache_usage(&self) -> Value {
        usage_json(self.bundle.session_cache())
    }

    /// The bundle's receipt alone (no cache evidence).
    fn bare_receipt(&self) -> Option<Value> {
        self.bundle.slowlink_receipt()
    }

    /// Receipt of the request that just ran, with its `cache_usage` evidence:
    /// the occupancy `before` it (taken by the caller before the request),
    /// the occupancy now and the actual eviction delta. `None` stays `None`.
    fn receipt(&self, before: &Value) -> Option<Value> {
        let mut receipt = self.bundle.slowlink_receipt()?;
        if let Some(object) = receipt.as_object_mut() {
            object.insert(
                "cache_usage".into(),
                cache_evidence(before, &self.cache_usage()),
            );
        }
        Some(receipt)
    }

    /// Receipt problems for one request call of `call_rows` rows. A forced
    /// layer-major call longer than one tile must run layer-major; a shorter
    /// one may be an explicit, named selector bypass.
    fn validate(
        &self,
        receipt: Option<&Value>,
        call_rows: usize,
        single_window: bool,
        overlap: bool,
    ) -> Vec<String> {
        let limits = Limits {
            require_single_window: single_window,
            require_overlap: overlap && self.route == Expect::LayerMajor,
            ..Limits::default()
        };
        match self.route {
            Expect::Incumbent => validate(receipt, Expect::Incumbent, &limits),
            Expect::LayerMajor if call_rows as u64 > MAX_TILE_ROWS => {
                validate(receipt, Expect::LayerMajor, &limits)
            }
            Expect::LayerMajor => validate_forced_short(receipt, &limits),
        }
    }
}

/// One whole request: either `prefill_final` (no cuts, no entry) or the
/// consecutive `forward_chunk_final` calls the fixture's launch cuts and
/// resume entry name. Returns each call's receipt.
fn drive(
    ctx: &mut Ctx,
    ids: &[u32],
    entry: usize,
    cuts: &[usize],
) -> Result<Vec<Option<Value>>, String> {
    gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
    let mut receipts = Vec::new();
    if entry == 0 && cuts.is_empty() {
        let before = ctx.cache_usage();
        gpu_error(
            ctx.bundle.prefill_final(&mut ctx.gpu, ids, 0, &ctx.logits),
            "prefill_final",
        )?;
        receipts.push(ctx.receipt(&before));
    } else {
        let mut points: Vec<usize> = cuts.to_vec();
        if entry > 0 {
            points.push(entry);
        }
        points.push(ids.len());
        points.sort_unstable();
        points.dedup();
        let mut start = 0;
        for end in points {
            let before = ctx.cache_usage();
            gpu_error(
                ctx.bundle
                    .forward_chunk_final(&mut ctx.gpu, &ids[start..end], &ctx.logits, None),
                &format!("forward_chunk_final {start}..{end}"),
            )?;
            receipts.push(ctx.receipt(&before));
            start = end;
        }
    }
    sync(&ctx.gpu)?;
    Ok(receipts)
}

struct Observation {
    receipts: Vec<Option<Value>>,
    state: Value,
    logits: Value,
    wall_s: f64,
}

fn observe(
    ctx: &mut Ctx,
    ids: &[u32],
    entry: usize,
    cuts: &[usize],
) -> Result<Observation, String> {
    let started = Instant::now();
    let receipts = drive(ctx, ids, entry, cuts)?;
    let wall_s = started.elapsed().as_secs_f64();
    let state = slowlink_state_digest(&ctx.gpu, &ctx.bundle)?;
    let logits = slowlink_logit_rows_summary(&ctx.gpu, &ctx.logits, 1, ctx.vocab, 16)?;
    Ok(Observation {
        receipts,
        state,
        logits,
        wall_s,
    })
}

fn receipts_json(receipts: &[Option<Value>]) -> Value {
    Value::Array(
        receipts
            .iter()
            .map(|r| r.clone().unwrap_or(Value::Null))
            .collect(),
    )
}

fn greedy(ctx: &mut Ctx, count: usize) -> Result<(Vec<u32>, &'static str), String> {
    let row = gpu_error(ctx.gpu.download_f32(&ctx.logits), "download prompt logits")?;
    if row.iter().any(|v| !v.is_finite()) {
        return Err("non-finite prefill logits".into());
    }
    let mut pending = Some(hipfire_runtime::llama::argmax(&row));
    let mut ids = Vec::with_capacity(count);
    while ids.len() < count {
        let token = gpu_error(
            ctx.bundle
                .forward_token_or_argmax(&mut ctx.gpu, pending.take(), &ctx.logits),
            "decode",
        )?;
        ids.push(token);
        if token == ctx.eos || ctx.tokenizer.is_terminator(token) {
            return Ok((ids, "stop"));
        }
    }
    Ok((ids, "length"))
}

// ── primitives ─────────────────────────────────────────────────────────

fn run_primitives(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    match ctx
        .bundle
        .slowlink_primitive_selftest(&mut ctx.gpu, &PRIMITIVE_CASES)
    {
        Err(error) => {
            // No admitted profile or a refused self-test: nothing was proved.
            out.insert("selftest_error".into(), json!(error.to_string()));
            for name in PRIMITIVE_CASES {
                checks.fail(
                    &format!("primitive.{name}"),
                    format!("self-test refused: {error}"),
                    Value::Null,
                );
            }
        }
        Ok(report) => {
            out.insert("selftest".into(), report.clone());
            let cases = report["cases"].as_array().cloned().unwrap_or_default();
            for name in PRIMITIVE_CASES {
                let key = format!("primitive.{name}");
                match cases.iter().find(|c| c["name"] == name) {
                    None => {} // missing stays missing: the run is incomplete
                    Some(c) if c["status"] == "pass" && c["mismatches"].as_u64() == Some(0) => {
                        checks.pass(&key, c.clone())
                    }
                    Some(c) => {
                        checks.fail(&key, "case did not pass with zero mismatches", c.clone())
                    }
                }
            }
        }
    }
}

// ── state ──────────────────────────────────────────────────────────────

fn same_run(a: &Observation, b: &Observation) -> Vec<String> {
    let mut bad = Vec::new();
    if a.state["combined"] != b.state["combined"] {
        bad.push("fixed-state digest differs between repeats".to_string());
    }
    if a.logits["combined"] != b.logits["combined"] {
        bad.push("final-logit digest differs between repeats".to_string());
    }
    for (x, y) in a.receipts.iter().zip(&b.receipts) {
        for key in [
            "route",
            "windows",
            "launch_cuts",
            "tile_rows",
            "resident_layers",
            "host_layers",
        ] {
            if x.as_ref().and_then(|r| r.get(key)) != y.as_ref().and_then(|r| r.get(key)) {
                bad.push(format!("receipt {key} differs between repeats"));
            }
        }
    }
    bad
}

fn prompt_check(ctx: &mut Ctx, checks: &mut Checks, report: &mut Vec<Value>, index: usize) {
    let prompt = ctx.fixture.prompts[index].clone();
    let id = prompt.id.clone();
    let mut runs: Vec<Observation> = Vec::new();
    for repeat in 0..ctx.fixture.repeats {
        match observe(ctx, &prompt.prompt_ids, prompt.entry, &prompt.launch_cuts) {
            Ok(run) => runs.push(run),
            Err(error) => {
                checks.fail(
                    &format!("state.{id}.route_receipt"),
                    format!("repeat {repeat}: {error}"),
                    Value::Null,
                );
                checks.fail(
                    &format!("state.{id}.repeat_determinism"),
                    format!("repeat {repeat}: {error}"),
                    Value::Null,
                );
                return;
            }
        }
    }
    let mut problems = Vec::new();
    for (repeat, run) in runs.iter().enumerate() {
        for (call, receipt) in run.receipts.iter().enumerate() {
            let rows = receipt
                .as_ref()
                .and_then(|r| u64_at(r, "prompt_rows"))
                .map_or(prompt.prompt_ids.len(), |n| n as usize);
            // Only the whole-prompt 16384 request has the one-window rule.
            let single = prompt.kind == "rows"
                && prompt.launch_cuts.is_empty()
                && prompt.prompt_ids.len() as u64 == TARGET_ROWS;
            for p in ctx.validate(receipt.as_ref(), rows, single, false) {
                problems.push(format!("repeat {repeat} call {call}: {p}"));
            }
        }
    }
    let detail = json!({
        "rows": prompt.prompt_ids.len(), "kind": prompt.kind, "entry": prompt.entry,
        "launch_cuts": prompt.launch_cuts,
        "wall_s": runs.iter().map(|r| r.wall_s).collect::<Vec<_>>(),
        "receipts": runs.iter().map(|r| receipts_json(&r.receipts)).collect::<Vec<_>>(),
    });
    checks.push(&format!("state.{id}.route_receipt"), problems, detail);
    let mut bad = Vec::new();
    for run in &runs[1..] {
        bad.extend(same_run(&runs[0], run));
    }
    bad.sort();
    bad.dedup();
    let digests = json!({
        "state": runs[0].state["combined"], "final_logits": runs[0].logits["combined"],
        "top1": runs[0].logits["per_row"][0]["top1"], "repeats": runs.len(),
    });
    checks.push(
        &format!("state.{id}.repeat_determinism"),
        bad,
        digests.clone(),
    );
    report.push(json!({"id": id, "digests": digests, "state": runs[0].state, "final_logits": runs[0].logits}));
    if prompt.kind == "irregular" {
        // The named launch cuts were driven as consecutive calls; each call's
        // receipt (checked above) carries its entry position.
        let mut chrono = Vec::new();
        let mut expect_entry = 0u64;
        for (call, receipt) in runs[0].receipts.iter().enumerate() {
            let entry = receipt.as_ref().and_then(|r| u64_at(r, "entry_pos"));
            let rows = receipt.as_ref().and_then(|r| u64_at(r, "prompt_rows"));
            if entry != Some(expect_entry) {
                chrono.push(format!(
                    "call {call} entry_pos {entry:?}, expected {expect_entry}"
                ));
            }
            expect_entry += rows.unwrap_or(0);
        }
        if expect_entry != prompt.prompt_ids.len() as u64 {
            chrono.push(format!(
                "calls covered {expect_entry} rows of {}",
                prompt.prompt_ids.len()
            ));
        }
        checks.push(
            &format!("state.{id}.launch_cut_chronology"),
            chrono,
            json!({"cuts": prompt.launch_cuts}),
        );
    }
    if prompt.kind == "resume" {
        let entry = runs[0]
            .receipts
            .last()
            .and_then(|r| r.as_ref())
            .and_then(|r| u64_at(r, "entry_pos"));
        let problems = if entry == Some(prompt.entry as u64) {
            Vec::new()
        } else {
            vec![format!(
                "resumed call entry_pos {entry:?} != {}",
                prompt.entry
            )]
        };
        checks.push(
            &format!("state.{id}.resume_entry"),
            problems,
            json!({"entry": prompt.entry}),
        );
    }
}

/// Cold prefill (reused = 0) of `ids`, commit, digest, logits bytes digest,
/// greedy ids.
struct Cold {
    state: Value,
    logits: Value,
    ids: Vec<u32>,
    receipt: Option<Value>,
}

fn cold_commit(ctx: &mut Ctx, ids: &[u32], reused: usize) -> Result<Cold, String> {
    let before = ctx.cache_usage();
    gpu_error(
        ctx.bundle
            .prefill_final(&mut ctx.gpu, ids, reused, &ctx.logits),
        "prefill_final",
    )?;
    let receipt = ctx.receipt(&before);
    sync(&ctx.gpu)?;
    let state = slowlink_state_digest(&ctx.gpu, &ctx.bundle)?;
    let logits = slowlink_logit_rows_summary(&ctx.gpu, &ctx.logits, 1, ctx.vocab, 16)?;
    ctx.bundle.session_commit();
    let (decoded, _) = greedy(ctx, RESTORE_DECODE)?;
    Ok(Cold {
        state,
        logits,
        ids: decoded,
        receipt,
    })
}

/// Why the boundary cases could not run: every boundary check fails (none is
/// skipped), so the run is a FAILURE, never a pass or a quiet omission.
fn fail_boundary(checks: &mut Checks, cuts: &[usize], why: &str) {
    for name in [
        "state.boundary_capture",
        "state.session_restore",
        "state.divergent_continuation",
    ] {
        checks.fail(name, why, Value::Null);
    }
    for cut in cuts {
        checks.fail(&format!("state.boundary.{cut}.restore"), why, Value::Null);
    }
}

struct BoundaryCase {
    /// Rows the cache planned to reuse (`None` if the case did not get that far).
    reused: Option<usize>,
    executed: bool,
    restore: Vec<String>,
    divergent: Vec<String>,
    detail: Value,
}

/// One requested boundary: the cache must plan exactly `cut` rows from a
/// stored snapshot; the restored request must then compute only the tail and
/// reproduce the cold recompute `cold` of the same extended prompt `ext`.
/// A plan that offers less is a failure and the restore is NOT run, so a cold
/// recompute can never pass as a restore.
fn boundary_case(
    ctx: &mut Ctx,
    cut: usize,
    ext: &[u32],
    cold: &Result<Cold, String>,
) -> BoundaryCase {
    let mut case = BoundaryCase {
        reused: None,
        executed: false,
        restore: Vec::new(),
        divergent: Vec::new(),
        detail: Value::Null,
    };
    let cold = match cold {
        Ok(cold) => cold,
        Err(e) => {
            case.restore.push(format!("cold reference: {e}"));
            return case;
        }
    };
    for p in ctx.validate(cold.receipt.as_ref(), ext.len(), false, false) {
        case.divergent.push(format!("cold reference receipt: {p}"));
    }
    // A reset drops the resident snapshot (the live rows are gone), so the
    // restore can only come from a published pool snapshot.
    if let Err(e) = gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset") {
        case.restore.push(e);
        return case;
    }
    let before_plan = ctx.cache_usage();
    let planned = ctx.bundle.session_plan(ext, SessionRoute::Ar);
    let live = planned > 0
        && ctx
            .bundle
            .session_plan_is_live(ext, SessionRoute::Ar, planned);
    case.reused = Some(planned);
    case.detail = json!({
        "cut": cut, "extended_rows": ext.len(), "planned": planned, "live": live,
        "cache_usage_before_plan": before_plan,
        "cold_receipt": cold.receipt,
    });
    if planned != cut || live {
        // No restored request runs on a bad plan: only the plan is judged.
        case.restore
            .extend(boundary_plan_problems(cut, planned, live));
        case.detail["restore_run"] = json!(false);
        case.detail["cache_usage_after"] = ctx.cache_usage();
        return case;
    }
    let warm = match cold_commit(ctx, ext, planned) {
        Ok(warm) => warm,
        Err(e) => {
            case.restore.push(format!("restored prefill: {e}"));
            case.detail["cache_usage_after"] = ctx.cache_usage();
            return case;
        }
    };
    case.executed = true;
    case.restore.extend(boundary_restore_problems(
        cut,
        planned,
        live,
        warm.receipt.as_ref(),
        ext.len(),
    ));
    let mut bad = Vec::new();
    if warm.state["combined"] != cold.state["combined"] {
        bad.push("restored state digest differs from cold");
    }
    if warm.logits["combined"] != cold.logits["combined"] {
        bad.push("restored final logits differ from cold");
    }
    if warm.ids != cold.ids {
        bad.push("restored greedy ids differ from cold");
    }
    case.divergent.extend(bad.iter().map(|b| b.to_string()));
    case.divergent.extend(
        ctx.validate(warm.receipt.as_ref(), ext.len() - planned, false, false)
            .into_iter()
            .map(|p| format!("restored receipt: {p}")),
    );
    case.detail["restore_run"] = json!(true);
    case.detail["equal"] = json!(bad.is_empty());
    case.detail["problems"] = json!(bad);
    case.detail["warm_receipt"] = json!(warm.receipt);
    case.detail["cold_ids"] = json!(cold.ids);
    case.detail["warm_ids"] = json!(warm.ids);
    case.detail["cold_state"] = cold.state["combined"].clone();
    case.detail["warm_state"] = warm.state["combined"].clone();
    case.detail["cache_usage_after"] = ctx.cache_usage();
    case
}

/// Real SessionCache boundaries through the production prefill: the cold
/// references of every cut, then ONE whole-prompt `prefill_final` of the
/// boundary prompt (the explicit marker id makes every requested cut a
/// snapshot boundary through `set_turn_end_token` + `set_turn_snapshots`, on
/// every arm), then a restore case per cut. No case is skipped.
fn boundary_cases(
    ctx: &mut Ctx,
    checks: &mut Checks,
    boundary: &FixturePrompt,
    cuts: &[usize],
    detail: &mut Vec<Value>,
) {
    let marker = ctx.fixture.boundary_token_id;
    let extended: Vec<Vec<u32>> = cuts
        .iter()
        .map(|&cut| {
            let mut ids = boundary.prompt_ids[..cut].to_vec();
            ids.extend_from_slice(&ctx.fixture.continuation_ids);
            ids
        })
        .collect();
    // Cold references first: their own captures are cleared below, so the
    // restore cases only ever read the whole-prompt prefill's snapshots.
    let mut colds: Vec<Result<Cold, String>> = Vec::new();
    for ids in &extended {
        let cold = gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")
            .and_then(|()| cold_commit(ctx, ids, 0));
        colds.push(cold);
    }
    ctx.bundle.session_clear(&mut ctx.gpu);
    let cleared = ctx.cache_usage();
    let capture = (|| -> Result<(Option<Value>, Value), String> {
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        let before = ctx.cache_usage();
        gpu_error(
            ctx.bundle
                .prefill_final(&mut ctx.gpu, &boundary.prompt_ids, 0, &ctx.logits),
            "whole-prompt prefill_final",
        )?;
        let receipt = ctx.receipt(&before);
        sync(&ctx.gpu)?;
        ctx.bundle.session_commit();
        Ok((receipt, ctx.cache_usage()))
    })();
    let layer_major = ctx.route == Expect::LayerMajor;
    let mut capture_problems = Vec::new();
    let capture_detail = match &capture {
        Err(e) => {
            capture_problems.push(format!("whole-prompt prefill: {e}"));
            json!({"cache_usage_after_clear": cleared})
        }
        Ok((receipt, published)) => {
            capture_problems.extend(ctx.validate(
                receipt.as_ref(),
                boundary.prompt_ids.len(),
                false,
                false,
            ));
            capture_problems.extend(boundary_capture_problems(
                cuts,
                receipt.as_ref(),
                layer_major,
            ));
            if published["pool_bytes"].as_u64().unwrap_or(0) == 0 {
                capture_problems.push(
                    "no snapshot was published to the pool (pool_bytes 0 after commit)".into(),
                );
            }
            json!({
                "prompt_rows": boundary.prompt_ids.len(), "marker_id": marker, "cuts": cuts,
                "receipt": receipt, "cache_usage_after_clear": cleared,
                "cache_usage_after_commit": published,
            })
        }
    };
    checks.push(
        "state.boundary_capture",
        capture_problems.clone(),
        capture_detail.clone(),
    );
    detail.push(json!({"capture": capture_detail}));
    let mut reused = Vec::new();
    let mut verified: Vec<(usize, bool)> = Vec::new();
    let mut restore_all: Vec<String> = capture_problems
        .iter()
        .map(|p| format!("capture: {p}"))
        .collect();
    let mut divergent_all = Vec::new();
    for (index, &cut) in cuts.iter().enumerate() {
        let case = boundary_case(ctx, cut, &extended[index], &colds[index]);
        if let Some(rows) = case.reused {
            reused.push((cut, rows));
        }
        checks.push(
            &format!("state.boundary.{cut}.restore"),
            case.restore.clone(),
            case.detail.clone(),
        );
        restore_all.extend(case.restore.iter().map(|p| format!("cut {cut}: {p}")));
        divergent_all.extend(case.divergent.iter().map(|p| format!("cut {cut}: {p}")));
        if !case.executed {
            divergent_all.push(format!("cut {cut}: the restored case did not run"));
        }
        verified.push((
            cut,
            case.executed && case.restore.is_empty() && case.divergent.is_empty(),
        ));
        detail.push(case.detail);
    }
    restore_all.extend(boundary_coverage_problems(cuts, &reused));
    restore_all.sort();
    restore_all.dedup();
    let largest = largest_verified_cut(&verified);
    detail.push(json!({
        "cache_usage_final": ctx.cache_usage(),
        "largest_verified_restore_cut": largest,
    }));
    checks.push(
        "state.session_restore",
        restore_all,
        json!({"boundaries": cuts, "marker_id": marker, "reused": reused,
               "largest_verified_restore_cut": largest}),
    );
    checks.push(
        "state.divergent_continuation",
        divergent_all,
        json!({"continuation_rows": ctx.fixture.continuation_ids.len(), "decode": RESTORE_DECODE}),
    );
}

/// Strict 16K boundary snapshots: ONE whole `prefill_final` of the 16384-row
/// prompt plus three continuation rows under the UNCHANGED 8192-row cadence
/// (no marker policy), then a strict pool restore at 8192 and at 16384 against
/// cold references. A cut the capture did not publish fails (the plan offers
/// less than the cut) and is never recomputed cold. No autotuning, no cache
/// policy change: a real memory or capture failure is recorded as is.
fn boundary16k_checks(ctx: &mut Ctx, checks: &mut Checks, detail: &mut Vec<Value>) {
    let fail_all = |checks: &mut Checks, why: &str| {
        for name in boundary16k_required() {
            checks.fail(&name, why, Value::Null);
        }
    };
    let Some(rows) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "rows" && p.prompt_ids.len() == 16384)
        .map(|p| p.prompt_ids.clone())
    else {
        return fail_all(checks, "the fixture has no 16384-row prompt");
    };
    let ids = match boundary16k_prompt(
        &rows,
        &ctx.fixture.continuation_ids,
        ctx.fixture.max_seq,
        RESTORE_DECODE,
    ) {
        Ok(ids) => ids,
        Err(e) => return fail_all(checks, &e),
    };
    if ctx.bundle.session_cache().is_none() {
        return fail_all(checks, "no session cache is attached to the bundle");
    }
    // Unchanged natural cadence: no message-end token, no turn snapshots.
    ctx.bundle.set_turn_end_token(None);
    ctx.bundle.set_turn_snapshots(false);
    let cuts = BOUNDARY16K_CUTS;
    let extended: Vec<Vec<u32>> = cuts
        .iter()
        .map(|&cut| ids[..cut].iter().chain(&ids[16384..]).copied().collect())
        .collect();
    let mut colds: Vec<Result<Cold, String>> = Vec::new();
    for ext in &extended {
        let cold = gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")
            .and_then(|()| cold_commit(ctx, ext, 0));
        colds.push(cold);
    }
    ctx.bundle.session_clear(&mut ctx.gpu);
    let cleared = ctx.cache_usage();
    let capture = (|| -> Result<(Option<Value>, Value), String> {
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        let before = ctx.cache_usage();
        gpu_error(
            ctx.bundle.prefill_final(&mut ctx.gpu, &ids, 0, &ctx.logits),
            "whole 16K prefill_final",
        )?;
        let receipt = ctx.receipt(&before);
        sync(&ctx.gpu)?;
        ctx.bundle.session_commit();
        Ok((receipt, ctx.cache_usage()))
    })();
    let layer_major = ctx.route == Expect::LayerMajor;
    let mut capture_problems = Vec::new();
    let capture_detail = match &capture {
        Err(e) => {
            capture_problems.push(format!("whole-prompt 16K prefill: {e}"));
            json!({"prompt_rows": ids.len(), "cuts": cuts, "cache_usage_after_clear": cleared,
                   "cache_usage_after_failure": ctx.cache_usage()})
        }
        Ok((receipt, published)) => {
            capture_problems.extend(ctx.validate(receipt.as_ref(), ids.len(), false, false));
            capture_problems.extend(boundary_capture_problems(
                &cuts,
                receipt.as_ref(),
                layer_major,
            ));
            if published["pool_bytes"].as_u64().unwrap_or(0) == 0 {
                capture_problems.push(
                    "no snapshot was published to the pool (pool_bytes 0 after commit)".into(),
                );
            }
            json!({
                "prompt_rows": ids.len(), "cuts": cuts, "cadence": "unchanged 8192-row natural cadence",
                "receipt": receipt, "cache_usage_after_clear": cleared,
                "cache_usage_after_commit": published,
                "cache_evidence": cache_evidence(&cleared, published),
            })
        }
    };
    checks.push(
        "state.boundary16k.capture",
        capture_problems.clone(),
        capture_detail.clone(),
    );
    detail.push(json!({"boundary16k_capture": capture_detail}));
    let mut verified = Vec::new();
    for (index, &cut) in cuts.iter().enumerate() {
        let case = boundary_case(ctx, cut, &extended[index], &colds[index]);
        let mut problems: Vec<String> = case.restore.clone();
        problems.extend(case.divergent.iter().cloned());
        if !case.executed {
            problems.push("the restored case did not run".to_string());
        }
        verified.push((cut, case.executed && problems.is_empty()));
        checks.push(
            &format!("state.boundary16k.{cut}.restore"),
            problems,
            case.detail.clone(),
        );
        detail.push(json!({"boundary16k_case": case.detail}));
    }
    detail.push(json!({
        "boundary16k_largest_verified_restore_cut": largest_verified_cut(&verified),
        "cache_usage_final": ctx.cache_usage(),
    }));
}

fn restore_checks(ctx: &mut Ctx, checks: &mut Checks, detail: &mut Vec<Value>) {
    marker_boundary_checks(ctx, checks, detail);
    boundary16k_checks(ctx, checks, detail);
}

fn marker_boundary_checks(ctx: &mut Ctx, checks: &mut Checks, detail: &mut Vec<Value>) {
    let Some(boundary) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "boundary")
        .cloned()
    else {
        return;
    };
    let cuts = boundary.requested_boundaries.clone();
    let marker = ctx.fixture.boundary_token_id;
    if marker as usize >= ctx.vocab {
        let why = format!(
            "boundary_token_id {marker} is outside the vocabulary of {} ids",
            ctx.vocab
        );
        return fail_boundary(checks, &cuts, &why);
    }
    if ctx.bundle.session_cache().is_none() {
        return fail_boundary(checks, &cuts, "no session cache is attached to the bundle");
    }
    // The diagnostic snapshot policy, identical on both arms: the fixture's
    // explicit marker id is the message-end token, so each marker row makes
    // the position after it a boundary.
    ctx.bundle.set_turn_end_token(Some(marker));
    ctx.bundle.set_turn_snapshots(true);
    if ctx.bundle.turn_snapshots_active() {
        boundary_cases(ctx, checks, &boundary, &cuts, detail);
    } else {
        fail_boundary(
            checks,
            &cuts,
            "the marker snapshot policy did not activate (turn_snapshots_active is false)",
        );
    }
    // The later checks run with no message-end token (and so no marker
    // boundaries), as before this case.
    ctx.bundle.set_turn_end_token(None);
    ctx.bundle.set_turn_snapshots(false);
}

fn injection_checks(ctx: &mut Ctx, checks: &mut Checks, out: &mut Vec<Value>) {
    let Some(prompt) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "resume")
        .cloned()
    else {
        return;
    };
    let entry = prompt.entry;
    for point in FAILURE_POINTS {
        let name = format!("state.inject.{point}");
        let outcome = (|| -> Result<(Vec<String>, Value), String> {
            let mut bad = Vec::new();
            // A clean reference of the whole resumed request and of the entry.
            let clean = observe(ctx, &prompt.prompt_ids, entry, &[])?;
            gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
            gpu_error(
                ctx.bundle.forward_chunk_final(
                    &mut ctx.gpu,
                    &prompt.prompt_ids[..entry],
                    &ctx.logits,
                    None,
                ),
                "entry prefix",
            )?;
            sync(&ctx.gpu)?;
            let at_entry = slowlink_state_digest(&ctx.gpu, &ctx.bundle)?;
            gpu_error(
                ctx.bundle.slowlink_inject_failure(Some(point)),
                "arm injection",
            )?;
            let failed = ctx.bundle.forward_chunk_final(
                &mut ctx.gpu,
                &prompt.prompt_ids[entry..],
                &ctx.logits,
                None,
            );
            let consumed = ctx
                .bare_receipt()
                .and_then(|r| r.get("injection_consumed").and_then(Value::as_bool))
                .unwrap_or(false);
            let _ = ctx.bundle.slowlink_inject_failure(None);
            if failed.is_ok() {
                bad.push("the injected failure did not fail the request".to_string());
            }
            if !consumed {
                bad.push("the injection was not consumed (route ignored it)".to_string());
            }
            let _ = ctx.gpu.hip.device_synchronize();
            let containment = match slowlink_state_digest(&ctx.gpu, &ctx.bundle) {
                Ok(after)
                    if after["combined"] == at_entry["combined"]
                        && after["position"] == at_entry["position"] =>
                {
                    "restored"
                }
                _ => "reset",
            };
            // Whatever the containment, a fresh clean request must reproduce
            // the clean digests: no poisoned state, no donor restore residue.
            let again = observe(ctx, &prompt.prompt_ids, entry, &[])?;
            if again.state["combined"] != clean.state["combined"]
                || again.logits["combined"] != clean.logits["combined"]
            {
                bad.push(
                    "a clean re-run after the abort does not reproduce the clean digests"
                        .to_string(),
                );
            }
            for p in ctx.validate(
                again.receipts.last().and_then(|r| r.as_ref()),
                prompt.prompt_ids.len() - entry,
                false,
                false,
            ) {
                bad.push(format!("re-run receipt: {p}"));
            }
            Ok((
                bad,
                json!({"point": point, "containment": containment, "consumed": consumed, "entry": entry}),
            ))
        })();
        match outcome {
            Ok((bad, detail)) => {
                out.push(detail.clone());
                checks.push(&name, bad, detail);
            }
            Err(e) => checks.fail(&name, e, Value::Null),
        }
    }
}

fn head_rows_check(ctx: &mut Ctx, checks: &mut Checks, out: &mut Vec<Value>) {
    let Some(prompt) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "rows" && p.prompt_ids.len() as u64 == TARGET_ROWS)
        .cloned()
    else {
        return;
    };
    let head = ctx.fixture.head_rows;
    let outcome = (|| -> Result<(Vec<String>, Value), String> {
        let body = prompt.prompt_ids.len() - head;
        let view = gpu_error(
            ctx.gpu.zeros(&[head * ctx.vocab], DType::F32),
            "head logits",
        )?;
        let mut summaries = Vec::new();
        let mut bad = Vec::new();
        for repeat in 0..ctx.fixture.repeats {
            gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
            gpu_error(
                ctx.bundle
                    .prefill_final(&mut ctx.gpu, &prompt.prompt_ids[..body], 0, &ctx.logits),
                "prefill body",
            )?;
            // All-row API over only the last `head` rows: bounded, never N×vocab.
            gpu_error(
                ctx.bundle
                    .forward_chunk(&mut ctx.gpu, &prompt.prompt_ids[body..], &view, None),
                "head forward_chunk",
            )?;
            sync(&ctx.gpu)?;
            let summary = slowlink_logit_rows_summary(&ctx.gpu, &view, head, ctx.vocab, 16)?;
            for row in summary["per_row"].as_array().into_iter().flatten() {
                if row["finite"] != true {
                    bad.push(format!(
                        "repeat {repeat}: non-finite head row {}",
                        row["row"]
                    ));
                }
            }
            summaries.push(summary);
        }
        let _ = ctx.gpu.free_tensor(view);
        if summaries
            .iter()
            .any(|s| s["combined"] != summaries[0]["combined"])
        {
            bad.push("head-row digests differ between repeats".to_string());
        }
        Ok((
            bad,
            json!({"head_rows": head, "body_rows": body, "summary": summaries[0]}),
        ))
    })();
    match outcome {
        Ok((bad, detail)) => {
            out.push(detail.clone());
            checks.push("state.head_rows_bounded", bad, detail);
        }
        Err(e) => checks.fail("state.head_rows_bounded", e, Value::Null),
    }
}

fn decode_checks(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    let Some(prompt) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "natural")
        .cloned()
    else {
        return;
    };
    let measured = ctx.fixture.decode_tokens;
    let result = (|| -> Result<(Vec<String>, Vec<String>, Value), String> {
        // Replay arm first: it fixes how many warm tokens the controller took.
        let armed = ctx.gpu.replay.configure_model_default(true);
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        drive(ctx, &prompt.prompt_ids, 0, &[])?;
        let row = gpu_error(ctx.gpu.download_f32(&ctx.logits), "download")?;
        let mut pending = Some(hipfire_runtime::llama::argmax(&row));
        let mut replay_ids = Vec::new();
        let mut warm = 0usize;
        while ctx.gpu.replay.state() != rdna_compute::replay::ReplayState::Ready
            && warm < REPLAY_WARM_LIMIT
        {
            let t = gpu_error(
                ctx.bundle
                    .forward_token_or_argmax(&mut ctx.gpu, pending.take(), &ctx.logits),
                "warm decode",
            )?;
            replay_ids.push(t);
            warm += 1;
        }
        let state_after_warm = format!("{:?}", ctx.gpu.replay.state());
        ctx.gpu.replay.begin_replay_observation_window();
        for _ in 0..measured {
            let t = gpu_error(
                ctx.bundle
                    .forward_token_or_argmax(&mut ctx.gpu, pending.take(), &ctx.logits),
                "replay decode",
            )?;
            replay_ids.push(t);
        }
        let observation = ctx.gpu.replay.replay_observation();
        let mut replay_bad = Vec::new();
        if state_after_warm != "Ready" {
            replay_bad.push(format!("retained body not Ready after {warm} warm token(s): {state_after_warm} (configure_model_default returned {armed})"));
        }
        if observation.count != measured as u64 || observation.failed {
            replay_bad.push(format!(
                "replay observed {} of {measured} measured tokens (failed={})",
                observation.count, observation.failed
            ));
        }
        // Eager arm: the same prompt, the same number of tokens.
        ctx.gpu.replay.configure_model_default(false);
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        drive(ctx, &prompt.prompt_ids, 0, &[])?;
        let (eager_ids, _) = greedy(ctx, replay_ids.len())?;
        let mut eager_bad = Vec::new();
        if eager_ids.is_empty() || eager_ids.iter().any(|&t| t as usize >= ctx.vocab) {
            eager_bad.push("eager decode produced no valid ids".to_string());
        }
        if eager_ids != replay_ids {
            replay_bad.push("retained-replay ids differ from the eager ids".to_string());
        }
        let detail = json!({
            "eager_ids": eager_ids, "replay_ids": replay_ids, "warm_tokens": warm,
            "replay_state": state_after_warm,
            "observation": {"count": observation.count, "failed": observation.failed},
            "eager_sha256": sha_u32(&eager_ids), "replay_sha256": sha_u32(&replay_ids),
            "decoded": ctx.tokenizer.decode(&eager_ids),
        });
        Ok((eager_bad, replay_bad, detail))
    })();
    match result {
        Ok((eager_bad, replay_bad, detail)) => {
            out.insert("decode".into(), detail.clone());
            checks.push("state.decode_eager", eager_bad, detail.clone());
            checks.push("state.decode_retained_replay", replay_bad, detail);
        }
        Err(e) => {
            checks.fail("state.decode_eager", e.clone(), Value::Null);
            checks.fail("state.decode_retained_replay", e, Value::Null);
        }
    }
}

fn state_required(fixture: &Fixture, route: Expect) -> Vec<String> {
    let mut names = Vec::new();
    for p in &fixture.prompts {
        names.push(format!("state.{}.route_receipt", p.id));
        names.push(format!("state.{}.repeat_determinism", p.id));
        if p.kind == "irregular" {
            names.push(format!("state.{}.launch_cut_chronology", p.id));
        }
        if p.kind == "resume" {
            names.push(format!("state.{}.resume_entry", p.id));
        }
        if p.kind == "boundary" {
            names.push("state.boundary_capture".to_string());
            for cut in &p.requested_boundaries {
                names.push(format!("state.boundary.{cut}.restore"));
            }
        }
    }
    if fixture
        .prompts
        .iter()
        .any(|p| p.kind == "rows" && p.prompt_ids.len() == 16384)
    {
        names.extend(boundary16k_required());
    }
    names.extend(
        [
            "state.session_restore",
            "state.divergent_continuation",
            "state.head_rows_bounded",
            "state.decode_eager",
            "state.decode_retained_replay",
        ]
        .map(String::from),
    );
    if route == Expect::LayerMajor {
        for point in FAILURE_POINTS {
            names.push(format!("state.inject.{point}"));
        }
    }
    names
}

fn run_state(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    let mut prompts = Vec::new();
    for index in 0..ctx.fixture.prompts.len() {
        prompt_check(ctx, checks, &mut prompts, index);
    }
    out.insert("prompts".into(), Value::Array(prompts));
    let mut restore = Vec::new();
    restore_checks(ctx, checks, &mut restore);
    out.insert("restore".into(), Value::Array(restore));
    if ctx.route == Expect::LayerMajor {
        let mut injections = Vec::new();
        injection_checks(ctx, checks, &mut injections);
        out.insert("injections".into(), Value::Array(injections));
    } else {
        for point in FAILURE_POINTS {
            checks.skip(
                &format!("state.inject.{point}"),
                "the incumbent route has no S1 transaction to inject into",
            );
        }
    }
    let mut head = Vec::new();
    head_rows_check(ctx, checks, &mut head);
    out.insert("head".into(), Value::Array(head));
    decode_checks(ctx, checks, out);
}

// ── product ────────────────────────────────────────────────────────────

fn run_product(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    let mut records = Vec::new();
    for index in 0..ctx.fixture.prompts.len() {
        let prompt = ctx.fixture.prompts[index].clone();
        let id = prompt.id.clone();
        let name = format!("product.{id}");
        let result = (|| -> Result<(Vec<String>, Value), String> {
            let obs = observe(ctx, &prompt.prompt_ids, prompt.entry, &prompt.launch_cuts)?;
            let (ids, finish) = greedy(ctx, ctx.fixture.decode_tokens)?;
            let mut bad = Vec::new();
            let single = prompt.kind == "rows"
                && prompt.launch_cuts.is_empty()
                && prompt.prompt_ids.len() as u64 == TARGET_ROWS;
            for (call, receipt) in obs.receipts.iter().enumerate() {
                let rows = receipt
                    .as_ref()
                    .and_then(|r| u64_at(r, "prompt_rows"))
                    .map_or(prompt.prompt_ids.len(), |n| n as usize);
                for p in ctx.validate(receipt.as_ref(), rows, single, false) {
                    bad.push(format!("call {call}: {p}"));
                }
            }
            if let Some(why) = attractor(&ids) {
                bad.push(format!("attractor: {why}"));
            }
            if ids.is_empty() {
                bad.push("no tokens generated".to_string());
            }
            Ok((
                bad,
                json!({
                    "id": id, "kind": prompt.kind, "prompt_rows": prompt.prompt_ids.len(),
                    "prompt_sha256": sha_u32(&prompt.prompt_ids),
                    "token_ids": ids, "finish_reason": finish, "decoded": ctx.tokenizer.decode(&ids),
                    "first_logit_digest": obs.logits["combined"], "state_digest": obs.state["combined"],
                    "receipts": receipts_json(&obs.receipts),
                }),
            ))
        })();
        match result {
            Ok((bad, record)) => {
                records.push(record.clone());
                checks.push(&name, bad, json!({"id": id}));
            }
            Err(e) => checks.fail(&name, e, Value::Null),
        }
    }
    out.insert("records".into(), Value::Array(records));
    out.insert(
        "human_review".into(),
        json!("required: read every decoded text; the attractor heuristic is not a quality gate"),
    );
}

// ── perf ───────────────────────────────────────────────────────────────

fn run_perf(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    // One untimed short prefill materializes lazy allocations and kernels.
    if let Some(warm) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "rows" && p.prompt_ids.len() == 1537)
        .cloned()
    {
        if let Err(e) = drive(ctx, &warm.prompt_ids, 0, &[]) {
            checks.fail("perf.warmup", e, Value::Null);
            return;
        }
    }
    let mut rows_out = Vec::new();
    let perf_rows = ctx.fixture.perf_rows.clone();
    for rows in perf_rows {
        let name = format!("perf.pp{rows}");
        let Some(prompt) = ctx
            .fixture
            .prompts
            .iter()
            .find(|p| p.kind == "rows" && p.prompt_ids.len() == rows)
            .cloned()
        else {
            continue;
        };
        let target = rows as u64 == TARGET_ROWS;
        let result = (|| -> Result<(Vec<String>, Value), String> {
            let mut times = Vec::new();
            let mut receipts = Vec::new();
            for _ in 0..ctx.fixture.perf_repeats {
                let started = Instant::now();
                drive(ctx, &prompt.prompt_ids, 0, &[])?;
                gpu_error(ctx.gpu.download_f32(&ctx.logits), "download prompt logits")?;
                times.push(started.elapsed().as_secs_f64());
                receipts.push(ctx.bare_receipt());
            }
            let mut bad = Vec::new();
            for r in &receipts {
                bad.extend(ctx.validate(r.as_ref(), rows, target, target));
            }
            let tok_s: Vec<f64> = times.iter().map(|t| rows as f64 / t).collect();
            Ok((
                bad,
                json!({
                    "rows": rows, "prompt_sha256": sha_u32(&prompt.prompt_ids),
                    "seconds": times, "tok_s": tok_s,
                    // The fresh-process observation is the first timed prefill;
                    // later in-process repeats are informational dispersion only.
                    "process_observation_tok_s": tok_s[0],
                    "meets_1400": target.then(|| tok_s[0] >= 1400.0),
                    "meets_2000": target.then(|| tok_s[0] >= 2000.0),
                    "receipts": receipts_json(&receipts),
                }),
            ))
        })();
        match result {
            Ok((bad, detail)) => {
                rows_out.push(detail.clone());
                checks.push(&name, bad, detail);
            }
            Err(e) => checks.fail(&name, e, Value::Null),
        }
    }
    out.insert("prefill".into(), Value::Array(rows_out));
    // Decode tok/s after the last prefill, greedy.
    let decode = (|| -> Result<Value, String> {
        let count = ctx.fixture.decode_tokens;
        let started = Instant::now();
        let (ids, finish) = greedy(ctx, count)?;
        sync(&ctx.gpu)?;
        let seconds = started.elapsed().as_secs_f64();
        Ok(
            json!({"tokens": ids.len(), "seconds": seconds, "tok_s": ids.len() as f64 / seconds, "finish": finish, "ids_sha256": sha_u32(&ids)}),
        )
    })();
    match decode {
        Ok(detail) => {
            out.insert("decode".into(), detail.clone());
            checks.pass("perf.decode", detail);
        }
        Err(e) => checks.fail("perf.decode", e, Value::Null),
    }
}

fn perf_required(fixture: &Fixture) -> Vec<String> {
    let mut names = vec!["perf.warmup".to_string()];
    for &rows in &fixture.perf_rows {
        if fixture
            .prompts
            .iter()
            .any(|p| p.kind == "rows" && p.prompt_ids.len() == rows)
        {
            names.push(format!("perf.pp{rows}"));
        }
    }
    names.push("perf.decode".into());
    names
}

fn product_required(fixture: &Fixture) -> Vec<String> {
    fixture
        .prompts
        .iter()
        .map(|p| format!("product.{}", p.id))
        .collect()
}

// ── driver ─────────────────────────────────────────────────────────────

fn write_report(path: &Path, report: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_string_pretty(report).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| format!("write {}: {e}", path.display()))
}

fn run(args: &Args) -> Result<bool, String> {
    let fixture_json = read_json(&args.fixture)?;
    let fixture = Fixture::parse(&fixture_json)?;
    let manifest = read_json(&args.manifest)?;
    // The route is forced through the one production switch, before the
    // process config snapshot is first read. A conflicting ambient value is
    // refused rather than overridden.
    let forced = if args.route == Expect::LayerMajor {
        "1"
    } else {
        "0"
    };
    if let Ok(ambient) = std::env::var(ROUTE_ENV) {
        if ambient != forced {
            return Err(format!(
                "{ROUTE_ENV}={ambient} conflicts with --route {}",
                args.route.name()
            ));
        }
    } else {
        std::env::set_var(ROUTE_ENV, forced);
    }
    let binary = std::env::current_exe().map_err(|e| e.to_string())?;
    let observed_binary = file_sha256(&binary)?;
    let model_size = std::fs::metadata(&args.model)
        .map_err(|e| format!("stat {}: {e}", args.model.display()))?
        .len();
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    // Baseline sample: after the HIP context exists, before any model allocation.
    // Only the primitive cell (Hold 1's first cell) runs the heavy telemetry:
    // the smaps/fdinfo background sampler and the required residual check.
    // Timed and quality cells keep their previous pipeline (point diagnostics).
    let heavy_memory = args.mode == Mode::Primitives;
    let mut mem_samples = vec![mem_sample(&gpu, "pre_load", heavy_memory)];
    let sampler = heavy_memory.then(ProcSampler::start);
    let identity_problems = manifest_problems(
        &manifest,
        &Observed {
            binary_sha256: observed_binary.clone(),
            model_size_bytes: model_size,
            gpu_arch: gpu.arch.clone(),
        },
    );
    let mut required = match args.mode {
        Mode::Primitives => PRIMITIVE_CASES
            .iter()
            .map(|n| format!("primitive.{n}"))
            .collect(),
        Mode::State => state_required(&fixture, args.route),
        Mode::Product => product_required(&fixture),
        Mode::Perf => perf_required(&fixture),
    };
    required.push("identity.manifest".to_string());
    required.push("load.placement".to_string());
    if sampler.is_some() {
        required.push("memory.unowned_residual".to_string());
    }
    let required_refs: Vec<&str> = required.iter().map(String::as_str).collect();
    let mut checks = Checks::new(&required_refs);
    checks.push(
        "identity.manifest",
        identity_problems.clone(),
        json!({"binary_sha256": observed_binary, "model_size_bytes": model_size, "gpu_arch": gpu.arch,
               "model_sha256_verified": false, "note": "model sha256 is pinned by the manifest and not recomputed (125 GB)"}),
    );
    let mut out = serde_json::Map::new();
    if identity_problems.is_empty() {
        let mut fixture = fixture;
        let session_cache = (args.mode == Mode::State).then_some(fixture.session_cache_bytes);
        let loaded = slowlink_load_bundle(&mut gpu, &args.model, fixture.max_seq, session_cache)?;
        let vocab = loaded.vocab;
        out.insert(
            "load".into(),
            json!({
                "num_layers": loaded.num_layers, "host_mapped_tensors": loaded.host_mapped_tensors,
                "resident_layers": loaded.resident_layers, "placement": loaded.placement,
                "slowlink_profile": loaded.slowlink_profile, "ple_format": loaded.ple_format,
                "vocab": vocab, "max_seq": fixture.max_seq,
            }),
        );
        // The placement the load actually made: a forced layer-major run needs
        // the S1 admission's joint placement (>= 10 resident layers); the
        // incumbent arm must not carry a profile.
        let mut placement_problems = Vec::new();
        match args.route {
            Expect::LayerMajor => {
                if !loaded.slowlink_profile || loaded.placement != "slowlink-profile" {
                    placement_problems.push(format!(
                        "layer-major was forced but the load did not admit the S1 profile (placement {})",
                        loaded.placement
                    ));
                }
                if (loaded.resident_layers as u64) < MIN_RESIDENT_LAYERS {
                    placement_problems.push(format!(
                        "resident layers {} < {MIN_RESIDENT_LAYERS}",
                        loaded.resident_layers
                    ));
                }
            }
            Expect::Incumbent => {
                if loaded.slowlink_profile {
                    placement_problems.push("the incumbent arm loaded an S1 profile".to_string());
                }
            }
        }
        checks.push(
            "load.placement",
            placement_problems,
            json!({"placement": loaded.placement, "resident_layers": loaded.resident_layers,
                   "host_mapped_tensors": loaded.host_mapped_tensors, "slowlink_profile": loaded.slowlink_profile}),
        );
        // Natural-text prompts are tokenized with the artifact's own tokenizer.
        for prompt in fixture.prompts.iter_mut() {
            if let Some(text) = &prompt.text {
                prompt.prompt_ids = loaded.tokenizer.encode(text);
                if prompt.prompt_ids.is_empty()
                    || prompt.prompt_ids.len() + fixture.decode_tokens + 1 > fixture.max_seq
                {
                    return Err(format!(
                        "natural prompt {} tokenizes to {} ids, outside 1..max_seq",
                        prompt.id,
                        prompt.prompt_ids.len()
                    ));
                }
            }
        }
        let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        mem_samples.push(mem_sample(&gpu, "post_load", sampler.is_some()));
        let mut ctx = Ctx {
            gpu,
            bundle: loaded.bundle,
            tokenizer: loaded.tokenizer,
            logits,
            vocab,
            eos: loaded.eos_token,
            fixture,
            route: args.route,
            mem: mem_samples,
        };
        match args.mode {
            Mode::Primitives => run_primitives(&mut ctx, &mut checks, &mut out),
            Mode::State => run_state(&mut ctx, &mut checks, &mut out),
            Mode::Product => run_product(&mut ctx, &mut checks, &mut out),
            Mode::Perf => run_perf(&mut ctx, &mut checks, &mut out),
        }
        if let Some(sampler) = sampler {
            // The primitive cell (Hold 1's first cell) only: the heavy sampler
            // and the required residual check against the manifest's census.
            ctx.sample_memory("end");
            let mut samples = std::mem::take(&mut ctx.mem);
            samples.push(sampler.finish());
            match memory_ledger(&manifest) {
                Err(e) => {
                    out.insert("memory_observations".into(), json!({"samples": samples}));
                    checks.fail("memory.unowned_residual", e, Value::Null);
                }
                Ok(ledger) => {
                    let (problems, receipt) = memory_problems(&samples, &ledger);
                    checks.push(
                        "memory.unowned_residual",
                        problems,
                        json!({
                            "observed": receipt["observed"], "residual": receipt["residual"],
                            "allowances": receipt["allowances"], "see": "evidence.memory_observations",
                        }),
                    );
                    out.insert("memory_observations".into(), receipt);
                }
            }
        } else {
            // Every other cell (state, product, timed perf): no sampler thread,
            // no smaps, no census requirement. Three light point diagnostics,
            // taken outside any timed region, are recorded and never judged.
            let mut samples = std::mem::take(&mut ctx.mem);
            samples.push(mem_sample(&ctx.gpu, "end", false));
            out.insert(
                "memory_diagnostics".into(),
                json!({"required": false, "samples": samples}),
            );
        }
        let peak = ctx
            .gpu
            .hip
            .get_vram_info()
            .ok()
            .map(|(free, total)| json!({"free": free, "total": total}));
        out.insert("vram_after".into(), peak.unwrap_or(Value::Null));
        let Ctx {
            mut gpu,
            bundle,
            logits,
            ..
        } = ctx;
        let _ = gpu.free_tensor(logits);
        if let Err(e) = bundle.free_gpu(&mut gpu) {
            checks.fail(
                "teardown",
                format!("bundle teardown failed: {e}"),
                Value::Null,
            );
        }
    }
    let summary = checks.json();
    let status = summary["status"].as_str().unwrap_or("fail").to_string();
    let report = json!({
        "schema": REPORT_SCHEMA,
        "mode": args.mode.name(),
        "route": args.route.name(),
        "status": status,
        "model": args.model, "fixture": args.fixture, "manifest": args.manifest,
        "fixture_sha256": file_sha256(&args.fixture)?,
        "manifest_sha256": file_sha256(&args.manifest)?,
        "checks": summary,
        "evidence": out,
    });
    write_report(&args.output, &report)?;
    eprintln!(
        "qwen4_slowlink_gate: {} {} -> {status}",
        args.mode.name(),
        args.route.name()
    );
    Ok(status == "pass")
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("qwen4_slowlink_gate: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            // A setup failure still leaves a report: nothing ran, so INCOMPLETE.
            let report = json!({
                "schema": REPORT_SCHEMA, "mode": args.mode.name(), "route": args.route.name(),
                "status": "incomplete", "error": e,
            });
            let _ = write_report(&args.output, &report);
            eprintln!("qwen4_slowlink_gate: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod slowlink_receipt_tests {
    use super::*;

    fn receipt() -> Value {
        json!({
            "schema": RECEIPT_SCHEMA, "route": "layer-major", "selected_by": "env1", "bypass_reason": null,
            "entry_pos": 0, "prompt_rows": 16384, "windows": [{"start": 0, "rows": 16384}],
            "launch_cuts": [1536, 3072], "tile_rows": 1536, "retained_scratch_rows": 1536,
            "sweeps": 1, "resident_layers": 10, "host_layers": 38,
            "host_expert_bytes_total": 50_803_507_200u64, "submitted_bytes": 50_803_507_200u64,
            "useful_bytes": 50_000_000_000u64, "unused_prefetched_bytes": 803_507_200u64,
            "reserve": {"tile": 1u64 << 30, "arena": 1u64 << 29, "ring": 167_772_160u64,
                        "lookahead": 1_336_000_000u64, "tables": 1u64 << 20, "total": 4_000_000_000u64},
            "spans_ms": {"wall": 11000.0, "copy": 7200.0, "compute": 3000.0, "route_barrier": 100.0, "nonexpert": 1500.0},
            "overlap": {"cross_layer_intersections": 37u64, "cross_layer_overlap_ms": 900.0,
                        "copy_idle_ms": 50.0, "page_full_stalls": 3u64, "copy_starved_stalls": 1u64},
            "peak_vram_bytes": 23_000_000_000u64,
        })
    }

    fn incumbent() -> Value {
        let mut r = receipt();
        r["route"] = json!("incumbent");
        r["selected_by"] = json!("env0");
        r["reserve"] =
            json!({"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0});
        r["overlap"] = json!({"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0, "copy_idle_ms": 0.0, "page_full_stalls": 0, "copy_starved_stalls": 0});
        r
    }

    fn single() -> Limits {
        Limits {
            require_single_window: true,
            require_overlap: true,
            ..Limits::default()
        }
    }

    fn problems(r: &Value) -> Vec<String> {
        validate(Some(r), Expect::LayerMajor, &single())
    }

    #[test]
    fn slowlink_receipt_good_layer_major_and_incumbent_pass() {
        assert_eq!(problems(&receipt()), Vec::<String>::new());
        assert_eq!(
            validate(Some(&incumbent()), Expect::Incumbent, &Limits::default()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn slowlink_receipt_missing_receipt_and_missing_route_fail_closed() {
        assert!(!validate(None, Expect::LayerMajor, &single()).is_empty());
        for key in [
            "route",
            "windows",
            "sweeps",
            "reserve",
            "spans_ms",
            "overlap",
            "launch_cuts",
            "schema",
        ] {
            let mut r = receipt();
            r.as_object_mut().unwrap().remove(key);
            assert!(!problems(&r).is_empty(), "removing {key} must fail");
        }
        let mut r = receipt();
        r["route"] = json!("maybe");
        assert!(!problems(&r).is_empty());
    }

    #[test]
    fn slowlink_receipt_forced_layer_major_that_ran_incumbent_fails_with_reason() {
        let mut r = incumbent();
        r["bypass_reason"] = json!("link is not narrow");
        let p = validate(Some(&r), Expect::LayerMajor, &Limits::default());
        assert!(p.iter().any(|m| m.contains("link is not narrow")), "{p:?}");
        // And an incumbent arm that ran layer-major is refused too.
        assert!(!validate(Some(&receipt()), Expect::Incumbent, &Limits::default()).is_empty());
        // The incumbent must not report S1 reserves.
        let mut r = incumbent();
        r["reserve"]["total"] = json!(1);
        assert!(!validate(Some(&r), Expect::Incumbent, &Limits::default()).is_empty());
    }

    #[test]
    fn slowlink_receipt_target_needs_exactly_one_window_and_one_sweep_per_window() {
        let mut r = receipt();
        r["windows"] = json!([{"start": 0, "rows": 8192}, {"start": 8192, "rows": 8192}]);
        assert!(problems(&r).iter().any(|m| m.contains("one required")));
        r["sweeps"] = json!(2);
        r["submitted_bytes"] = json!(2 * 50_803_507_200u64);
        // Two windows are consistent per se, but never a 16K target pass.
        assert!(problems(&r).iter().any(|m| m.contains("one required")));
        let mut r = receipt();
        r["sweeps"] = json!(2);
        assert!(problems(&r).iter().any(|m| m.contains("once per window")));
        let mut r = receipt();
        r["windows"] = json!([{"start": 0, "rows": 1000}]);
        assert!(problems(&r).iter().any(|m| m.contains("cover")));
    }

    #[test]
    fn slowlink_receipt_residency_tile_scratch_and_cold_bytes_are_enforced() {
        for (key, value) in [
            ("resident_layers", json!(9)),
            ("host_layers", json!(39)),
            ("tile_rows", json!(8192)),
            ("retained_scratch_rows", json!(8192)),
            ("host_expert_bytes_total", json!(57_488_179_200u64)),
        ] {
            let mut r = receipt();
            r[key] = value;
            assert!(!problems(&r).is_empty(), "{key} must be refused");
        }
        let mut r = receipt();
        r["submitted_bytes"] = json!(60_000_000_000u64);
        assert!(!problems(&r).is_empty(), "more than one sweep of bytes");
        let mut r = receipt();
        r["useful_bytes"] = json!(51_000_000_000u64);
        assert!(!problems(&r).is_empty(), "useful above submitted");
    }

    #[test]
    fn slowlink_receipt_unfunded_lookahead_reserve_is_refused() {
        let mut r = receipt();
        r["reserve"]["lookahead"] = json!(0);
        r["reserve"]["ring"] = json!(167_772_160u64);
        assert!(problems(&r).iter().any(|m| m.contains("lookahead")));
        let mut r = receipt();
        r["reserve"]["total"] = json!(10);
        assert!(problems(&r).iter().any(|m| m.contains("sum of its parts")));
    }

    #[test]
    fn slowlink_receipt_overlap_win_needs_cross_layer_event_intersections() {
        let mut r = receipt();
        r["overlap"]["cross_layer_intersections"] = json!(0);
        assert!(!overlap_claim_allowed(&r));
        assert!(problems(&r).iter().any(|m| m.contains("cross-layer")));
        assert!(overlap_claim_allowed(&receipt()));
        // Intersections without measured overlap time are not evidence either.
        let mut r = receipt();
        r["overlap"]["cross_layer_overlap_ms"] = json!(0.0);
        assert!(!overlap_claim_allowed(&r));
    }

    #[test]
    fn slowlink_receipt_unequal_manifests_cannot_claim_bitwise_transport() {
        let s1 = receipt();
        let mut incumbent8192 = incumbent();
        incumbent8192["tile_rows"] = json!(8192);
        incumbent8192["launch_cuts"] = json!([8192]);
        assert!(bitwise_transport_claim(&incumbent8192, &s1).is_err());
        // Even equal manifests on different routes are not transport-only.
        let mut other = incumbent();
        other["launch_cuts"] = s1["launch_cuts"].clone();
        assert!(bitwise_transport_claim(&other, &s1).is_err());
        assert!(bitwise_transport_claim(&s1, &s1.clone()).is_ok());
    }

    #[test]
    fn slowlink_receipt_zero_and_unexecuted_required_checks_are_incomplete() {
        let empty = Checks::new(&[]);
        assert_eq!(empty.status(), "incomplete");
        let mut checks = Checks::new(&["a", "b"]);
        checks.pass("a", Value::Null);
        assert_eq!(checks.status(), "incomplete");
        assert_eq!(checks.missing(), vec!["b".to_string()]);
        checks.skip("b", "not applicable");
        assert_eq!(
            checks.status(),
            "incomplete",
            "a skipped required check is not executed"
        );
        checks.pass("b", Value::Null);
        assert_eq!(checks.status(), "pass");
        checks.fail("b", "later failure", Value::Null);
        assert_eq!(checks.status(), "fail");
    }

    fn manifest() -> Value {
        json!({
            "source": {"base": "d8865a16b5d0e01eff712df85d2754122f6edaee", "tip": "a".repeat(40)},
            "binary": {"sha256": "b".repeat(64)},
            "model": {"path": "/m.mq4", "size_bytes": 125288544792u64, "sha256": "c".repeat(64)},
            "ple": {"source": "sidecar", "format": "q4", "sidecars": ["/m.mq4.ple-q4"]},
            "kernel": {"code_object_hashes": {"k": "d"}},
            "host": {"hostname": "hipx", "topology": "walk"},
            "device": {"uuid": "GPU-43390a851e296ee5", "arch": "gfx1100"},
            "link": {"gbps": 7.09, "topology_walk": "x4"},
            "memory_ledger": {"device_gib": 23.0, "host_unreclaimable_gib": 90.0},
        })
    }

    fn observed() -> Observed {
        Observed {
            binary_sha256: "b".repeat(64),
            model_size_bytes: 125288544792,
            gpu_arch: "gfx1100".into(),
        }
    }

    #[test]
    fn slowlink_receipt_identity_mismatch_and_missing_manifest_keys_fail_closed() {
        assert_eq!(
            manifest_problems(&manifest(), &observed()),
            Vec::<String>::new()
        );
        let mut wrong_binary = observed();
        wrong_binary.binary_sha256 = "e".repeat(64);
        assert!(manifest_problems(&manifest(), &wrong_binary)
            .iter()
            .any(|m| m.contains("binary identity")));
        let mut wrong_model = observed();
        wrong_model.model_size_bytes += 1;
        assert!(manifest_problems(&manifest(), &wrong_model)
            .iter()
            .any(|m| m.contains("model identity")));
        for path in MANIFEST_REQUIRED {
            let mut m = manifest();
            let mut node = &mut m;
            let parts: Vec<&str> = path.split('.').collect();
            for part in &parts[..parts.len() - 1] {
                node = node.get_mut(*part).unwrap();
            }
            node.as_object_mut().unwrap().remove(parts[parts.len() - 1]);
            assert!(
                !manifest_problems(&m, &observed()).is_empty(),
                "{path} must be required"
            );
        }
        let mut short = manifest();
        short["source"]["tip"] = json!("abc123");
        assert!(manifest_problems(&short, &observed())
            .iter()
            .any(|m| m.contains("40-hex")));
    }

    fn kld_receipt() -> Value {
        json!({
            "schema": "hipfire.qwen4.s11ref_klrows.v1", "rows": 6120, "sequences": 24, "route": "decode",
            "n_ctx": 512, "estimator": "e",
            "model_sha256": "1".repeat(64), "ref_sha256": "2".repeat(64), "corpus_sha256": "3".repeat(64),
            "binary_sha256": "4".repeat(64), "output_sha256": "5".repeat(64),
        })
    }

    #[test]
    fn slowlink_receipt_partial_corpus_fails_closed() {
        assert!(kld_receipt_problems(&kld_receipt()).is_empty());
        let mut partial = kld_receipt();
        partial["rows"] = json!(5000);
        assert!(!kld_receipt_problems(&partial).is_empty());
        let mut part = kld_receipt();
        part["schema"] = json!("hipfire.qwen4.s11ref_klrows.partition.v1");
        assert!(!kld_receipt_problems(&part).is_empty());
        let mut seqs = kld_receipt();
        seqs["sequences"] = json!(10);
        assert!(!kld_receipt_problems(&seqs).is_empty());
        let mut other_ref = kld_receipt();
        other_ref["ref_sha256"] = json!("9".repeat(64));
        assert!(!kld_receipts_comparable(&kld_receipt(), &other_ref).is_empty());
        assert!(kld_receipts_comparable(&kld_receipt(), &kld_receipt()).is_empty());
    }

    #[test]
    fn slowlink_receipt_paired_interval_aggregates_per_sequence_before_the_ci() {
        // 4 sequences with very different row counts: the unit is the sequence.
        let base = [0.1, 0.1, 0.1, 0.2, 0.2];
        let cand = [0.1, 0.1, 0.1, 0.2, 0.2];
        let means = sequence_means(&base, &[3, 1, 1]).unwrap();
        assert_eq!(means.len(), 3);
        assert!((means[0] - 0.1).abs() < 1e-12);
        assert!(
            sequence_means(&base, &[3, 1]).is_err(),
            "row counts must cover the rows"
        );
        assert!(
            sequence_means(&base, &[5, 0]).is_err(),
            "an empty sequence is refused"
        );
        let ci = paired_ci(&means, &sequence_means(&cand, &[3, 1, 1]).unwrap()).unwrap();
        assert_eq!((ci.mean, ci.lo, ci.hi), (0.0, 0.0, 0.0));
        assert!(paired_verdict(&ci));
        // A clearly worse candidate with a wide interval above the limit fails.
        let b: Vec<f64> = (0..24).map(|i| 0.05 + 0.001 * i as f64).collect();
        let jitter: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.05 + 0.0001 * (i % 3) as f64)
            .collect();
        let ci = paired_ci(&b, &jitter).unwrap();
        assert!(
            ci.lo > 0.0 && ci.hi > KLD_HI_LIMIT && !paired_verdict(&ci),
            "{ci:?}"
        );
        // Significantly worse but tiny passes via the hi bound.
        let tiny: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.002 + 0.0001 * (i % 3) as f64)
            .collect();
        let ci = paired_ci(&b, &tiny).unwrap();
        assert!(
            ci.lo > 0.0 && ci.hi <= KLD_HI_LIMIT && paired_verdict(&ci),
            "{ci:?}"
        );
        // An interval straddling zero passes.
        let noisy: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 2 == 0 { 0.004 } else { -0.004 })
            .collect();
        assert!(paired_verdict(&paired_ci(&b, &noisy).unwrap()));
        assert!(paired_ci(&b[..1], &noisy[..1]).is_err());
        assert!(paired_ci(&b, &noisy[..5]).is_err());
        assert!((t_975(23) - 2.069).abs() < 1e-9);
    }

    const CUTS: [usize; 5] = [3, 4, 1536, 1537, 8192];

    /// A 9000-row boundary prompt: marker id 7 at row `cut - 1` of every cut.
    fn boundary_ids() -> Vec<u32> {
        let mut ids = vec![1500u32; 9000];
        for cut in CUTS {
            ids[cut - 1] = 7;
        }
        ids
    }

    fn fixture() -> Value {
        let mut prompts = Vec::new();
        for rows in REQUIRED_PROMPT_ROWS {
            prompts.push(
                json!({"id": format!("p{rows}"), "kind": "rows", "prompt_ids": vec![7u32; rows]}),
            );
        }
        prompts.push(json!({"id": "irr", "kind": "irregular", "prompt_ids": vec![7u32; 3000], "launch_cuts": [1000, 2500]}));
        prompts.push(
            json!({"id": "res", "kind": "resume", "prompt_ids": vec![7u32; 3000], "entry": 1536}),
        );
        prompts.push(json!({"id": "bnd", "kind": "boundary", "prompt_ids": boundary_ids(), "requested_boundaries": CUTS}));
        prompts.push(json!({"id": "nat", "kind": "natural", "prompt_ids": vec![7u32; 100]}));
        json!({
            "schema": FIXTURE_SCHEMA, "max_seq": 16640, "decode_tokens": 8,
            "repeats": 3, "head_rows": 4, "prompts": prompts, "continuation_ids": [5, 6, 7],
            "failure_injections": FAILURE_POINTS, "boundary_token_id": 7,
        })
    }

    #[test]
    fn slowlink_receipt_fixture_refuses_a_narrowed_case_list_and_unbounded_head_scoring() {
        let parsed = Fixture::parse(&fixture()).unwrap();
        assert_eq!(parsed.prompts.len(), 9);
        for rows in REQUIRED_PROMPT_ROWS {
            let mut f = fixture();
            let prompts = f["prompts"].as_array_mut().unwrap();
            prompts.retain(|p| {
                p["prompt_ids"].as_array().map(|a| a.len()) != Some(rows) || p["kind"] != "rows"
            });
            assert!(Fixture::parse(&f).is_err(), "{rows}-row prompt is required");
        }
        for kind in ["irregular", "resume", "boundary", "natural"] {
            let mut f = fixture();
            f["prompts"]
                .as_array_mut()
                .unwrap()
                .retain(|p| p["kind"] != kind);
            assert!(Fixture::parse(&f).is_err(), "{kind} prompt is required");
        }
        let mut f = fixture();
        f["failure_injections"] = json!(["before_dma"]);
        assert!(Fixture::parse(&f).is_err());
        let mut f = fixture();
        f["head_rows"] = json!(1000);
        assert!(
            Fixture::parse(&f).is_err(),
            "whole-window scoring must stay bounded"
        );
        let mut f = fixture();
        f["repeats"] = json!(1);
        assert!(Fixture::parse(&f).is_err());
        let mut f = fixture();
        f["max_seq"] = json!(16000);
        assert!(Fixture::parse(&f).is_err());
    }

    #[test]
    fn slowlink_receipt_state_required_checks_cover_every_prompt_and_injection() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        let s1 = state_required(&fixture, Expect::LayerMajor);
        for point in FAILURE_POINTS {
            assert!(s1.contains(&format!("state.inject.{point}")));
        }
        assert!(s1.contains(&"state.p16384.route_receipt".to_string()));
        assert!(s1.contains(&"state.irr.launch_cut_chronology".to_string()));
        assert!(!state_required(&fixture, Expect::Incumbent)
            .contains(&"state.inject.before_dma".to_string()));
        assert_eq!(product_required(&fixture).len(), fixture.prompts.len());
        assert!(perf_required(&fixture).contains(&"perf.pp16384".to_string()));
    }

    #[test]
    fn slowlink_receipt_cli_is_frozen_and_requires_absolute_paths() {
        let ok = |extra: &[&str]| {
            let mut v: Vec<String> = [
                "--model",
                "/m",
                "--fixture",
                "/f.json",
                "--output",
                "/o.json",
                "--mode",
                "state",
                "--route",
                "layer-major",
                "--manifest",
                "/mf.json",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            v.extend(extra.iter().map(|s| s.to_string()));
            parse_args(v)
        };
        let args = ok(&[]).unwrap();
        assert_eq!((args.mode, args.route), (Mode::State, Expect::LayerMajor));
        assert!(ok(&["--surprise", "x"]).is_err());
        assert!(parse_args(
            [
                "--model",
                "rel",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "state",
                "--route",
                "incumbent",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
        assert!(parse_args(
            [
                "--model",
                "/m",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "state",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
        assert!(parse_args(
            [
                "--model",
                "/m",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "primitives",
                "--route",
                "incumbent",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
        assert!(parse_args(
            [
                "--model",
                "/m",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "perf",
                "--route",
                "bogus",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
    }

    #[test]
    fn slowlink_receipt_attractor_heuristic_flags_loops_and_collapse() {
        assert!(attractor(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).is_none());
        assert!(attractor(&[5; 20]).is_some());
        let looped: Vec<u32> = (0..40)
            .map(|i| 10 + (i % 2))
            .chain((0..16).map(|i| 100 + i))
            .collect();
        assert!(
            attractor(&looped).is_none(),
            "a loop that ended is not the tail attractor"
        );
        let tail: Vec<u32> = (0..16)
            .map(|i| 100 + i)
            .chain((0..40).map(|i| 10 + (i % 3)))
            .collect();
        assert!(attractor(&tail).is_some());
    }

    #[test]
    fn slowlink_receipt_boundary_fixture_pins_the_marker_at_every_cut_and_nowhere_else() {
        let parsed = Fixture::parse(&fixture()).unwrap();
        assert_eq!(parsed.boundary_token_id, 7);
        assert!(boundary_fixture_problems(&boundary_ids(), &CUTS, 7).is_empty());
        // The marker is required and typed.
        for bad in [json!(null), json!("7"), json!(-1), json!(1u64 << 40)] {
            let mut f = fixture();
            f["boundary_token_id"] = bad;
            assert!(Fixture::parse(&f).is_err());
        }
        let mut f = fixture();
        f.as_object_mut().unwrap().remove("boundary_token_id");
        assert!(Fixture::parse(&f).is_err(), "marker id must be explicit");
        // Inside the row-prompt id range it could collide with real ids.
        let mut f = fixture();
        f["boundary_token_id"] = json!(LCG_ID_MIN);
        assert!(Fixture::parse(&f).is_err());
        // A missing marker at cut - 1, a stray marker and a cut outside the
        // prompt are all refused; the caller's ids are never rewritten.
        for cut in CUTS {
            let mut ids = boundary_ids();
            ids[cut - 1] = 1500;
            let bad = boundary_fixture_problems(&ids, &CUTS, 7);
            assert!(bad.iter().any(|p| p.contains(&format!("boundary {cut}"))));
            let mut f = fixture();
            f["prompts"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .for_each(|p| {
                    if p["kind"] == "boundary" {
                        p["prompt_ids"] = json!(ids);
                    }
                });
            assert!(Fixture::parse(&f).is_err());
        }
        let mut ids = boundary_ids();
        ids[100] = 7;
        assert!(boundary_fixture_problems(&ids, &CUTS, 7)
            .iter()
            .any(|p| p.contains("unrequested boundary 101")));
        assert!(!boundary_fixture_problems(&ids, &[0, 9000], 7).is_empty());
    }

    #[test]
    fn slowlink_receipt_boundary_restore_must_reuse_exactly_the_cut_from_a_snapshot() {
        let receipt = |entry: u64, rows: u64| json!({"entry_pos": entry, "prompt_rows": rows});
        // 1536-row prefix + 300 continuation rows restored at 1536.
        let r = receipt(1536, 300);
        assert!(boundary_restore_problems(1536, 1536, false, Some(&r), 1836).is_empty());
        // A shallower snapshot, a cold recompute and live state never pass.
        assert!(!boundary_restore_problems(1536, 4, false, Some(&r), 1836).is_empty());
        assert!(
            !boundary_restore_problems(1536, 0, false, Some(&receipt(0, 1836)), 1836).is_empty()
        );
        assert!(!boundary_restore_problems(1536, 1536, true, Some(&r), 1836).is_empty());
        // The plan said `cut` but the request recomputed the whole prompt.
        assert!(
            !boundary_restore_problems(1536, 1536, false, Some(&receipt(0, 1836)), 1836).is_empty()
        );
        assert!(
            !boundary_restore_problems(1536, 1536, false, Some(&receipt(1536, 1836)), 1836)
                .is_empty()
        );
        assert!(!boundary_restore_problems(1536, 1536, false, None, 1836).is_empty());
        assert!(boundary_plan_problems(3, 3, false).is_empty());
        assert_eq!(boundary_plan_problems(3, 0, false).len(), 1);
    }

    #[test]
    fn slowlink_receipt_every_requested_boundary_needs_an_executed_exact_restore() {
        let all: Vec<(usize, usize)> = CUTS.iter().map(|&c| (c, c)).collect();
        assert!(boundary_coverage_problems(&CUTS, &all).is_empty());
        for skip in 0..CUTS.len() {
            let mut missing = all.clone();
            missing.remove(skip);
            let bad = boundary_coverage_problems(&CUTS, &missing);
            assert_eq!(bad.len(), 1, "{bad:?}");
            assert!(bad[0].contains("never executed"));
            let mut shallow = all.clone();
            shallow[skip].1 = 0;
            assert!(boundary_coverage_problems(&CUTS, &shallow)[0].contains("reused 0"));
        }
        // An empty case list is not coverage.
        assert_eq!(boundary_coverage_problems(&CUTS, &[]).len(), CUTS.len());
    }

    #[test]
    fn slowlink_receipt_boundary_capture_needs_every_cut_as_an_executed_exit() {
        let mut r = receipt();
        r["windows"] = json!([{"start": 0, "rows": 1537}, {"start": 1537, "rows": 7463}]);
        r["launch_cuts"] = json!([3, 4, 1536, 1537, 8192]);
        assert!(boundary_capture_problems(&CUTS, Some(&r), true).is_empty());
        r["launch_cuts"] = json!([3, 4, 1537, 8192]);
        let bad = boundary_capture_problems(&CUTS, Some(&r), true);
        assert_eq!(bad.len(), 1);
        assert!(bad[0].contains("1536"));
        // The incumbent route's exits are not receipted; only the receipt is required.
        assert!(boundary_capture_problems(&CUTS, Some(&incumbent()), false).is_empty());
        assert!(!boundary_capture_problems(&CUTS, None, false).is_empty());
    }

    #[test]
    fn slowlink_receipt_cache_usage_evidence_reports_the_actual_eviction_delta() {
        let usage = |pool: u64, evictions: u64| {
            usage_value(&SessionCacheUsage {
                pool_bytes: pool,
                budget_bytes: 2 << 30,
                pending_bytes: 11,
                resident_bytes: 22,
                deferred_release_bytes: 33,
                parked_bytes: 44,
                evictions_total: evictions,
            })
        };
        let before = usage(4096, 3);
        // Occupancy fell by two snapshots but exactly one eviction ran: the
        // delta is the counter's, not derived from the occupancy change.
        let after = usage(0, 4);
        let evidence = cache_evidence(&before, &after);
        assert_eq!(evidence["evictions_delta"], json!(1));
        assert_eq!(evidence["before"]["pool_bytes"], json!(4096));
        assert_eq!(
            evidence["after"]["device_total_bytes"],
            json!(11 + 22 + 33 + 44)
        );
        assert_eq!(
            cache_evidence(&before, &before)["evictions_delta"],
            json!(0)
        );
        // No cache, or a counter that went backwards, has no delta.
        assert!(cache_evidence(&Value::Null, &after)["evictions_delta"].is_null());
        assert!(cache_evidence(&after, &before)["evictions_delta"].is_null());
        assert!(usage_json(None).is_null());
    }

    #[test]
    fn slowlink_receipt_state_required_lists_every_boundary_case() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for route in [Expect::LayerMajor, Expect::Incumbent] {
            let names = state_required(&fixture, route);
            assert!(names.contains(&"state.boundary_capture".to_string()));
            for cut in CUTS {
                assert!(names.contains(&format!("state.boundary.{cut}.restore")));
            }
        }
    }

    #[test]
    fn slowlink_receipt_boundary16k_required_checks_and_prompt() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for route in [Expect::LayerMajor, Expect::Incumbent] {
            let names = state_required(&fixture, route);
            for name in boundary16k_required() {
                assert!(names.contains(&name), "{name}");
            }
        }
        assert_eq!(
            boundary16k_required(),
            [
                "state.boundary16k.capture",
                "state.boundary16k.8192.restore",
                "state.boundary16k.16384.restore",
            ]
        );
        let rows: Vec<u32> = (0..16384).map(|i| 1000 + i).collect();
        let ids = boundary16k_prompt(&rows, &[5, 6, 7, 8], 16640, 16).unwrap();
        // 16384 is an interior boundary of the 16387-row prompt, rows unchanged.
        assert_eq!(ids.len(), 16387);
        assert_eq!(&ids[..16384], &rows[..]);
        assert_eq!(&ids[16384..], &[5, 6, 7]);
        assert!(boundary16k_prompt(&rows[..16000], &[5, 6, 7], 16640, 16).is_err());
        assert!(boundary16k_prompt(&rows, &[5, 6], 16640, 16).is_err());
        assert!(boundary16k_prompt(&rows, &[5, 6, 7], 16400, 16).is_err());
    }

    #[test]
    fn slowlink_receipt_largest_verified_restore_cut_ignores_unverified_cases() {
        assert_eq!(largest_verified_cut(&[]), None);
        assert_eq!(
            largest_verified_cut(&[
                (3, true),
                (4, true),
                (1536, true),
                (1537, true),
                (8192, true)
            ]),
            Some(8192)
        );
        // A missed 8192 leaves the deepest verified requested cut visible.
        assert_eq!(
            largest_verified_cut(&[(3, true), (1536, true), (1537, true), (8192, false)]),
            Some(1537)
        );
        assert_eq!(largest_verified_cut(&[(3, false), (8192, false)]), None);
    }

    const GIB: u64 = 1 << 30;
    const TIB: u64 = 1 << 40;

    /// Process counters as one sample reads them: the unreclaimable RSS is
    /// `VmRSS` minus the clean regular-file RSS, derived as production does.
    /// `RssFile` is a TiB of file pages unless a test overrides it.
    fn counters(rss: u64, clean: u64, pin: u64, vram: u64, gtt: u64) -> ProcCounters {
        ProcCounters {
            vm_rss: Some(rss),
            rss_anon: Some(0),
            vm_pin: Some(pin),
            vm_hwm: Some(rss),
            rss_file: Some(TIB),
            rss_shmem: Some(0),
            reclaimable_file_clean: Some(clean),
            host_rss_unreclaimable: Some(rss.saturating_sub(clean)),
            drm_vram: Some(vram),
            drm_gtt: Some(gtt),
        }
    }

    fn mem_row(label: &str, free: u64, c: ProcCounters) -> Value {
        counters_json(label, "test", &c, Some((free, 32 * GIB)))
    }

    fn sampler_row(c: ProcCounters) -> Value {
        let mut row = counters_json(SAMPLER_LABEL, "test", &c, None);
        row["samples"] = json!(5);
        row
    }

    fn ledger_with_host(owned_host: u64) -> MemoryLedger {
        MemoryLedger {
            owned_device_bytes: 20 * GIB,
            owned_host_bytes: owned_host,
            census_basis: "test census".into(),
        }
    }

    fn ledger() -> MemoryLedger {
        ledger_with_host(80 * GIB)
    }

    /// The phase samples plus the sampler row: `c` at `post_load`,
    /// `request_after` and `end`, `s` as the sampler's maxima.
    fn memory_with(c: ProcCounters, s: ProcCounters) -> Vec<Value> {
        vec![
            // A baseline before any model mapping: no file-backed pages yet,
            // so it cannot raise the high-water RssFile of the cases below.
            mem_row(
                "pre_load",
                31 * GIB,
                ProcCounters {
                    rss_file: Some(0),
                    ..counters(GIB, 0, 0, GIB / 2, 0)
                },
            ),
            mem_row("post_load", 11 * GIB, c),
            mem_row("request_after", 11 * GIB, c),
            mem_row("end", 11 * GIB, c),
            sampler_row(s),
        ]
    }

    /// Owned 20 GiB device / 80 GiB host, a small residual on each.
    fn good_memory() -> Vec<Value> {
        memory_with(
            counters(70 * GIB, 0, 80 * GIB, 20 * GIB, 79 * GIB),
            counters(70 * GIB, 0, 80 * GIB, 20 * GIB + GIB / 2, 80 * GIB),
        )
    }

    fn swap_sampler(samples: &mut [Value], c: ProcCounters) {
        *samples.last_mut().unwrap() = sampler_row(c);
    }

    #[test]
    fn slowlink_receipt_memory_allowances_are_the_fixed_ledger_constants() {
        assert_eq!(UNOWNED_VRAM_ALLOWANCE_BYTES, 1_610_612_736);
        assert_eq!(UNOWNED_VRAM_ALLOWANCE_BYTES, 3 * (1 << 29));
        assert_eq!(UNOWNED_HOST_ALLOWANCE_BYTES, 2_147_483_648);
        assert_eq!(UNOWNED_HOST_ALLOWANCE_BYTES, 2 * GIB);
    }

    #[test]
    fn slowlink_receipt_memory_within_allowances_passes_and_subtracts_only_clean_regular_files() {
        let (bad, receipt) = memory_problems(&good_memory(), &ledger());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(receipt["residual"]["vram_bytes"], json!(GIB / 2));
        assert_eq!(receipt["residual"]["host_bytes"], json!(0));
        assert_eq!(receipt["high_water"]["rss_file"], json!(TIB));
        assert!(receipt["high_water"]["sampling"]
            .as_str()
            .unwrap()
            .contains("discrete points"));
        // Warm regular-file (HFQ) pages: a TiB of CLEAN file-backed RSS is the
        // explicit reclaimable class and is subtracted, so it passes.
        let warm = counters(TIB + 8 * GIB, TIB, 70 * GIB, 20 * GIB, 70 * GIB);
        let (bad, receipt) = memory_problems(&memory_with(warm, warm), &ledger());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(
            receipt["high_water"]["host_rss_unreclaimable"],
            json!(8 * GIB)
        );
        // Without the clean class the same RSS is a huge unowned residual.
        let dirty = counters(TIB + 8 * GIB, 0, 70 * GIB, 20 * GIB, 70 * GIB);
        let (bad, _) = memory_problems(&memory_with(dirty, dirty), &ledger());
        assert!(
            bad.iter().any(|p| p.contains("unowned host residual")),
            "{bad:?}"
        );
        // Under-use of the census is not a residual.
        let small = counters(GIB, 0, GIB, GIB, GIB);
        let (bad, receipt) = memory_problems(&memory_with(small, small), &ledger());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(receipt["residual"]["host_bytes"], json!(0));
    }

    #[test]
    fn slowlink_receipt_memory_pinned_kfd_plus_driver_anon_must_fail_the_host_cap() {
        // 52 GB of pinned KFD maps (RssFile and GTT, zero clean regular-file
        // pages) plus 4 GB of anonymous driver memory, owned census 52 GB.
        let owned = 52 * GIB;
        let kfd = |extra: u64| {
            let mut c = counters(52 * GIB + extra, 0, 52 * GIB, 20 * GIB, 52 * GIB);
            c.rss_file = Some(52 * GIB);
            c
        };
        let (bad, receipt) = memory_problems(
            &memory_with(kfd(4 * GIB), kfd(4 * GIB)),
            &ledger_with_host(owned),
        );
        assert!(
            bad.iter().any(|p| p.contains("unowned host residual")),
            "{bad:?}"
        );
        assert_eq!(receipt["residual"]["host_bytes"], json!(4 * GIB));
        // The whole RssFile is NOT subtracted: the pinned buffer objects stay counted.
        assert_eq!(receipt["high_water"]["rss_file"], json!(52 * GIB));
        // Exactly at the 2 GiB allowance passes, one byte over fails.
        let at = kfd(UNOWNED_HOST_ALLOWANCE_BYTES);
        assert!(
            memory_problems(&memory_with(at, at), &ledger_with_host(owned))
                .0
                .is_empty()
        );
        let over = kfd(UNOWNED_HOST_ALLOWANCE_BYTES + 1);
        assert!(
            !memory_problems(&memory_with(over, over), &ledger_with_host(owned))
                .0
                .is_empty()
        );
        // The pinned maps alone, no extra driver memory, are fully owned.
        let none = kfd(0);
        assert!(
            memory_problems(&memory_with(none, none), &ledger_with_host(owned))
                .0
                .is_empty()
        );
    }

    #[test]
    fn slowlink_receipt_memory_device_residual_over_the_exact_allowance_fails() {
        // Exactly at the allowance passes; one byte over fails (VRAM).
        for (extra, ok) in [
            (UNOWNED_VRAM_ALLOWANCE_BYTES, true),
            (UNOWNED_VRAM_ALLOWANCE_BYTES + 1, false),
        ] {
            let mut s = good_memory();
            swap_sampler(
                &mut s,
                counters(70 * GIB, 0, 80 * GIB, 20 * GIB + extra, 80 * GIB),
            );
            let (bad, _) = memory_problems(&s, &ledger());
            assert_eq!(bad.is_empty(), ok, "{bad:?}");
            assert!(ok || bad.iter().any(|p| p.contains("unowned VRAM residual")));
        }
        // hipMemGetInfo growth alone (small process vram) also fails.
        let host = counters(70 * GIB, 0, 80 * GIB, GIB, 79 * GIB);
        let mut s = memory_with(host, host);
        assert!(memory_problems(&s, &ledger()).0.is_empty());
        s[3]["device_free"] = json!(31 * GIB - 20 * GIB - UNOWNED_VRAM_ALLOWANCE_BYTES - 1);
        let (bad, _) = memory_problems(&s, &ledger());
        assert!(
            bad.iter().any(|p| p.contains("unowned VRAM residual")),
            "{bad:?}"
        );
    }

    #[test]
    fn slowlink_receipt_memory_missing_telemetry_fails_closed() {
        for key in [
            "drm_gtt",
            "drm_vram",
            "vm_pin",
            "vm_rss",
            "reclaimable_regular_file_clean_bytes",
            "host_rss_unreclaimable",
        ] {
            let mut s = good_memory();
            s[1][key] = Value::Null;
            let (bad, _) = memory_problems(&s, &ledger());
            assert!(
                bad.iter()
                    .any(|p| p.contains(&format!("{key} not observable"))),
                "{key}: {bad:?}"
            );
        }
        let mut s = good_memory();
        s[0]["device_free"] = Value::Null;
        assert!(!memory_problems(&s, &ledger()).0.is_empty());
        // Process GTT and the smaps class are required of the sampler too.
        for key in ["drm_gtt", "reclaimable_regular_file_clean_bytes"] {
            let mut s = good_memory();
            let mut row = sampler_row(counters(0, 0, 0, 0, 0));
            row[key] = Value::Null;
            *s.last_mut().unwrap() = row;
            assert!(!memory_problems(&s, &ledger()).0.is_empty(), "{key}");
        }
        // Missing phases or sampler are incomplete telemetry, not a pass.
        for label in MEMORY_PHASES {
            let mut s = good_memory();
            s.retain(|r| r["label"] != label);
            assert!(!memory_problems(&s, &ledger()).0.is_empty(), "{label}");
        }
        let mut s = good_memory();
        s.pop();
        assert!(!memory_problems(&s, &ledger()).0.is_empty());
        let mut s = good_memory();
        s.last_mut().unwrap()["samples"] = json!(0);
        assert!(!memory_problems(&s, &ledger()).0.is_empty());
        assert!(!memory_problems(&[], &ledger()).0.is_empty());
    }

    #[test]
    fn slowlink_receipt_memory_smaps_counts_only_clean_regular_file_mappings() {
        let map = |range: &str, path: &str, shared: u64, private: u64| {
            format!(
                "{range} r--p 00000000 08:01 99 {path}\nSize: 64 kB\nRss: 64 kB\nShared_Clean: {shared} kB\nPrivate_Clean: {private} kB\nPrivate_Dirty: 5 kB\nVmFlags: rd mr\n"
            )
        };
        let smaps = [
            map("55d0c0a00000-55d0c0a50000", "/models/q.hfq", 100, 60),
            map("7f0000000000-7f0100000000", "/dev/kfd", 0, 52_000),
            map("7f0200000000-7f0200100000", "/dev/dri/renderD128", 0, 11),
            map("7f0300000000-7f0300100000", "[heap]", 0, 7),
            map("7f0400000000-7f0400100000", "/memfd:buf (deleted)", 0, 13),
            map("7f0500000000-7f0500100000", "/dev/zero (deleted)", 0, 17),
            map("7f0600000000-7f0600100000", "/tmp/gone (deleted)", 0, 19),
            map("7f0700000000-7f0700100000", "/data/my model.bin", 4, 0),
            map("7f0800000000-7f0800100000", "", 0, 23),
        ]
        .concat();
        // 100 + 60 + 4 kB: the regular-file mappings' clean RSS only.
        assert_eq!(smaps_regular_file_clean_bytes(&smaps), Some(164 * 1024));
        assert_eq!(smaps_regular_file_clean_bytes(""), None);
        assert_eq!(
            smaps_regular_file_clean_bytes("Rss: 4 kB\nShared_Clean: 4 kB\n"),
            None
        );
        for (path, class) in [
            ("/dev/kfd", MapClass::KfdDri),
            ("/dev/dri/card0", MapClass::KfdDri),
            ("/dev/zero", MapClass::Zero),
            ("", MapClass::Anon),
            ("[stack]", MapClass::Anon),
            ("[anon:scudo]", MapClass::Anon),
            ("/memfd:x", MapClass::Anon),
            ("memfd:x", MapClass::Anon),
            ("/lib/x.so (deleted)", MapClass::Anon),
            ("/lib/x.so", MapClass::File),
        ] {
            assert_eq!(classify_mapping(path), class, "{path}");
        }
    }

    #[test]
    fn slowlink_receipt_memory_ledger_needs_the_owned_census_in_the_manifest() {
        let manifest = json!({"memory_ledger": {
            "owned_device_bytes": 21474836480u64, "owned_host_bytes": 85899345920u64,
            "census_basis": "A census + 2 GiB cache pool",
        }});
        let ledger = memory_ledger(&manifest).unwrap();
        assert_eq!(ledger.owned_device_bytes, 20 * GIB);
        assert_eq!(ledger.owned_host_bytes, 80 * GIB);
        for key in ["owned_device_bytes", "owned_host_bytes", "census_basis"] {
            let mut m = manifest.clone();
            m["memory_ledger"].as_object_mut().unwrap().remove(key);
            assert!(memory_ledger(&m).is_err(), "{key}");
        }
        let mut m = manifest.clone();
        m["memory_ledger"]["owned_host_bytes"] = json!(0);
        assert!(memory_ledger(&m).is_err());
        let mut m = manifest;
        m["memory_ledger"]["census_basis"] = json!("  ");
        assert!(memory_ledger(&m).is_err());
    }

    #[test]
    fn slowlink_receipt_memory_proc_parsers_dedupe_clients_and_fail_closed() {
        let status = "Name:\tx\nVmRSS:\t   100 kB\nRssAnon:\t    40 kB\nRssFile:\t    60 kB\nVmPin:\t     8 kB\nVmPeak:\t 999 kB\n";
        assert_eq!(status_bytes(status, "VmRSS"), Some(100 * 1024));
        assert_eq!(status_bytes(status, "RssFile"), Some(60 * 1024));
        assert_eq!(status_bytes(status, "VmPin"), Some(8 * 1024));
        assert_eq!(status_bytes(status, "VmHWM"), None);
        let client = |id: u32, extra: &str| {
            format!("pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{id}\n{extra}")
        };
        let both = client(7, "drm-memory-vram:\t2048 KiB\ndrm-memory-gtt:\t 512 KiB\n");
        let a = parse_drm_fdinfo(&both).unwrap();
        assert_eq!((a.vram, a.gtt), (Some(2 << 20), Some(512 << 10)));
        // Two fds of one client count once; a second client adds.
        let other = parse_drm_fdinfo(&client(
            9,
            "drm-memory-vram:\t1 MiB\ndrm-memory-gtt:\t1 MiB\n",
        ))
        .unwrap();
        assert_eq!(
            drm_totals(&[a.clone(), a.clone()]),
            Some((2 << 20, 512 << 10))
        );
        assert_eq!(
            drm_totals(&[a.clone(), other]),
            Some((3 << 20, (512 << 10) + (1 << 20)))
        );
        // Not amdgpu / no GTT counter / no client: not observable, never zero.
        assert!(parse_drm_fdinfo("pos:\t0\ndrm-driver:\ti915\ndrm-client-id:\t1\n").is_none());
        assert!(parse_drm_fdinfo("pos:\t0\nflags:\t02\n").is_none());
        let no_gtt = parse_drm_fdinfo(&client(5, "drm-memory-vram:\t4 KiB\n")).unwrap();
        assert_eq!(drm_totals(&[a, no_gtt]), None);
        assert_eq!(drm_totals(&[]), None);
        // Maxima: an observed value replaces an unobserved one.
        let mut max = ProcCounters::default();
        max.merge_max(&ProcCounters {
            drm_gtt: Some(5),
            ..ProcCounters::default()
        });
        max.merge_max(&ProcCounters {
            drm_gtt: Some(3),
            rss_file: Some(9),
            ..ProcCounters::default()
        });
        assert_eq!(
            (max.drm_gtt, max.rss_file, max.vm_pin),
            (Some(5), Some(9), None)
        );
    }
}
