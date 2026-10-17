// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure, GPU-free validation shared by the S1 slow-link gate
//! (`examples/qwen4_slowlink_gate.rs`), `tests/slowlink_prefill_hw.rs` and
//! the paired KLD verdict of `examples/qwen4_kld.rs`. Included with
//! `#[path = ...] mod slowlink_receipt;`.
//!
//! Everything fails closed: a missing receipt, a missing key, a partial
//! corpus, an identity mismatch or an empty required-check list is never a
//! pass. The route receipt schema is `V-RECEIPT-API.md` of the S1 plan
//! directory (`Qwen4Bundle::slowlink_receipt`).

#![allow(dead_code)]

use serde_json::{json, Value};

pub const RECEIPT_SCHEMA: &str = "hipfire.qwen4.slowlink_receipt.v1";
pub const FIXTURE_SCHEMA: &str = "hipfire.qwen4.slowlink_fixture.v1";

/// Main's rulings: at least ten resident layers, at most 38 host layers, a
/// mixer tile of at most 1,536 rows, one payload sweep per window.
pub const MIN_RESIDENT_LAYERS: u64 = 10;
pub const MAX_HOST_LAYERS: u64 = 38;
pub const MAX_TILE_ROWS: u64 = 1536;
/// 10 resident / 38 host layers of the MQ4 fixture, expert payload bytes.
pub const COLD_BYTES_LIMIT: u64 = 50_803_507_200;
/// Two 32-expert slots plus one 512-expert lookahead store of MQ4 payload
/// (~1.401 GiB); the 159 MiB ring alone is not the charged reserve.
pub const MIN_RING_LOOKAHEAD_BYTES: u64 = 1_503_238_553;
/// Rows of the one-window target fixture.
pub const TARGET_ROWS: u64 = 16_384;

pub const FAILURE_POINTS: [&str; 4] = ["before_dma", "during_panel", "after_layer", "before_commit"];
pub const REQUIRED_PROMPT_ROWS: [usize; 5] = [512, 1536, 1537, 8192, 16384];
pub const PRIMITIVE_CASES: [&str; 7] = [
    "payload_table_identity",
    "repeated_epoch_same_address",
    "delayed_reader_reuse",
    "within_expert_panel_canary",
    "canonical_combine_exact",
    "k1_tail_cohort",
    "zero_row_panel",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    Incumbent,
    LayerMajor,
}

impl Expect {
    pub fn name(self) -> &'static str {
        match self {
            Self::Incumbent => "incumbent",
            Self::LayerMajor => "layer-major",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub min_resident_layers: u64,
    pub max_host_layers: u64,
    pub max_tile_rows: u64,
    pub cold_bytes_limit: u64,
    pub min_ring_lookahead_bytes: u64,
    /// The request must have run as exactly one admitted window.
    pub require_single_window: bool,
    /// A timing/overlap claim: cross-layer L compute / L+1 H2D intersections
    /// must be observed.
    pub require_overlap: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            min_resident_layers: MIN_RESIDENT_LAYERS,
            max_host_layers: MAX_HOST_LAYERS,
            max_tile_rows: MAX_TILE_ROWS,
            cold_bytes_limit: COLD_BYTES_LIMIT,
            min_ring_lookahead_bytes: MIN_RING_LOOKAHEAD_BYTES,
            require_single_window: false,
            require_overlap: false,
        }
    }
}

const U64_KEYS: [&str; 11] = [
    "entry_pos",
    "prompt_rows",
    "tile_rows",
    "retained_scratch_rows",
    "sweeps",
    "resident_layers",
    "host_layers",
    "host_expert_bytes_total",
    "submitted_bytes",
    "useful_bytes",
    "unused_prefetched_bytes",
];

fn get<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(value, |node, key| node.get(key))
}

pub fn u64_at(value: &Value, path: &str) -> Option<u64> {
    get(value, path).and_then(Value::as_u64)
}

pub fn f64_at(value: &Value, path: &str) -> Option<f64> {
    get(value, path).and_then(Value::as_f64)
}

fn structural(receipt: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    if receipt.get("schema").and_then(Value::as_str) != Some(RECEIPT_SCHEMA) {
        bad.push(format!("receipt schema is not {RECEIPT_SCHEMA}"));
    }
    for key in ["route", "selected_by"] {
        if receipt.get(key).and_then(Value::as_str).is_none() {
            bad.push(format!("receipt key {key} missing or not a string"));
        }
    }
    match receipt.get("bypass_reason") {
        Some(Value::Null) | Some(Value::String(_)) => {}
        _ => bad.push("receipt key bypass_reason missing (string or null)".into()),
    }
    for key in U64_KEYS {
        if u64_at(receipt, key).is_none() {
            bad.push(format!("receipt key {key} missing or not an unsigned integer"));
        }
    }
    for key in ["tile", "arena", "ring", "lookahead", "tables", "total"] {
        let path = format!("reserve.{key}");
        if u64_at(receipt, &path).is_none() {
            bad.push(format!("receipt key {path} missing or not an unsigned integer"));
        }
    }
    for key in ["wall", "copy", "compute", "route_barrier", "nonexpert"] {
        let path = format!("spans_ms.{key}");
        match f64_at(receipt, &path) {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            _ => bad.push(format!("receipt key {path} missing or not a finite non-negative number")),
        }
    }
    for key in [
        "cross_layer_intersections",
        "page_full_stalls",
        "copy_starved_stalls",
    ] {
        let path = format!("overlap.{key}");
        if u64_at(receipt, &path).is_none() {
            bad.push(format!("receipt key {path} missing or not an unsigned integer"));
        }
    }
    for key in ["cross_layer_overlap_ms", "copy_idle_ms"] {
        let path = format!("overlap.{key}");
        match f64_at(receipt, &path) {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            _ => bad.push(format!("receipt key {path} missing or not a finite non-negative number")),
        }
    }
    if u64_at(receipt, "peak_vram_bytes").is_none() {
        bad.push("receipt key peak_vram_bytes missing or not an unsigned integer".into());
    }
    for key in ["windows", "launch_cuts"] {
        if receipt.get(key).and_then(Value::as_array).is_none() {
            bad.push(format!("receipt key {key} missing or not an array"));
        }
    }
    bad
}

/// `(start, rows)` of every admitted window.
fn windows(receipt: &Value) -> Result<Vec<(u64, u64)>, String> {
    receipt["windows"]
        .as_array()
        .ok_or("windows is not an array")?
        .iter()
        .map(|w| {
            Ok((
                w.get("start")
                    .and_then(Value::as_u64)
                    .ok_or("a window has no start")?,
                w.get("rows")
                    .and_then(Value::as_u64)
                    .ok_or("a window has no rows")?,
            ))
        })
        .collect()
}

/// Every reason `receipt` is not an acceptable `expect` route receipt; empty
/// means acceptable. `None` (no receipt at all) is never acceptable.
pub fn validate(receipt: Option<&Value>, expect: Expect, limits: &Limits) -> Vec<String> {
    let Some(r) = receipt else {
        return vec!["no slowlink receipt: the selector verdict is missing".into()];
    };
    let mut bad = structural(r);
    if !bad.is_empty() {
        return bad;
    }
    let route = r["route"].as_str().unwrap_or_default();
    if route != "incumbent" && route != "layer-major" {
        bad.push(format!("route {route:?} is neither incumbent nor layer-major"));
        return bad;
    }
    if !["env0", "env1", "auto"].contains(&r["selected_by"].as_str().unwrap_or_default()) {
        bad.push(format!("selected_by {:?} is not env0|env1|auto", r["selected_by"]));
    }
    let prompt_rows = u64_at(r, "prompt_rows").unwrap_or(0);
    match windows(r) {
        Err(e) => bad.push(e),
        Ok(ws) => {
            let mut cursor = 0u64;
            for (index, (start, rows)) in ws.iter().enumerate() {
                if *start != cursor || *rows == 0 {
                    bad.push(format!(
                        "window {index} starts at {start} with {rows} rows; windows must tile the request contiguously"
                    ));
                }
                cursor = start.saturating_add(*rows);
            }
            if cursor != prompt_rows {
                bad.push(format!("windows cover {cursor} rows, the request has {prompt_rows}"));
            }
            if limits.require_single_window && ws.len() != 1 {
                bad.push(format!("{} windows admitted, one required", ws.len()));
            }
            if route == "layer-major" && u64_at(r, "sweeps") != Some(ws.len() as u64) {
                bad.push(format!(
                    "sweeps {:?} != windows {}: payload must be submitted once per window",
                    u64_at(r, "sweeps"),
                    ws.len()
                ));
            }
        }
    }
    let cuts: Vec<u64> = r["launch_cuts"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    if cuts.len() != r["launch_cuts"].as_array().map_or(0, Vec::len)
        || cuts.windows(2).any(|w| w[0] >= w[1])
        || cuts.last().is_some_and(|&c| c > prompt_rows)
    {
        bad.push("launch_cuts are not strictly ascending row boundaries within the request".into());
    }
    match expect {
        Expect::Incumbent => {
            if route != "incumbent" {
                bad.push(format!("incumbent arm ran route {route}"));
            }
        }
        Expect::LayerMajor => {
            if route != "layer-major" {
                bad.push(format!(
                    "layer-major was forced but the selector ran {route} (bypass_reason {})",
                    r["bypass_reason"]
                ));
            }
        }
    }
    if route == "incumbent" {
        for path in [
            "reserve.total",
            "reserve.ring",
            "reserve.lookahead",
            "overlap.cross_layer_intersections",
            "overlap.page_full_stalls",
            "overlap.copy_starved_stalls",
        ] {
            if u64_at(r, path) != Some(0) {
                bad.push(format!("incumbent route reports nonzero S1 field {path}"));
            }
        }
        return bad;
    }
    // layer-major
    let tile = u64_at(r, "tile_rows").unwrap_or(u64::MAX);
    if tile == 0 || tile > limits.max_tile_rows {
        bad.push(format!("tile_rows {tile} outside 1..={}", limits.max_tile_rows));
    }
    let scratch = u64_at(r, "retained_scratch_rows").unwrap_or(u64::MAX);
    if scratch > limits.max_tile_rows {
        bad.push(format!(
            "retained_scratch_rows {scratch} > {}: ordinary 8192/full-prompt scratch is retained",
            limits.max_tile_rows
        ));
    }
    let resident = u64_at(r, "resident_layers").unwrap_or(0);
    let host = u64_at(r, "host_layers").unwrap_or(u64::MAX);
    if resident < limits.min_resident_layers {
        bad.push(format!("resident_layers {resident} < {}", limits.min_resident_layers));
    }
    if host > limits.max_host_layers {
        bad.push(format!("host_layers {host} > {}", limits.max_host_layers));
    }
    let host_bytes = u64_at(r, "host_expert_bytes_total").unwrap_or(u64::MAX);
    let submitted = u64_at(r, "submitted_bytes").unwrap_or(u64::MAX);
    let useful = u64_at(r, "useful_bytes").unwrap_or(u64::MAX);
    let sweeps = u64_at(r, "sweeps").unwrap_or(0);
    if host_bytes > limits.cold_bytes_limit {
        bad.push(format!(
            "host_expert_bytes_total {host_bytes} > cold limit {}",
            limits.cold_bytes_limit
        ));
    }
    if sweeps == 0 || submitted > host_bytes.saturating_mul(sweeps) {
        bad.push(format!(
            "submitted_bytes {submitted} exceeds {sweeps} sweep(s) of {host_bytes} host bytes"
        ));
    }
    if useful > submitted {
        bad.push(format!("useful_bytes {useful} > submitted_bytes {submitted}"));
    }
    if u64_at(r, "unused_prefetched_bytes").unwrap_or(u64::MAX) > submitted {
        bad.push("unused_prefetched_bytes exceeds submitted_bytes".into());
    }
    let parts: u64 = ["tile", "arena", "ring", "lookahead", "tables"]
        .iter()
        .map(|k| u64_at(r, &format!("reserve.{k}")).unwrap_or(0))
        .sum();
    if u64_at(r, "reserve.total").unwrap_or(0) < parts {
        bad.push("reserve.total is below the sum of its parts".into());
    }
    let ring_look =
        u64_at(r, "reserve.ring").unwrap_or(0) + u64_at(r, "reserve.lookahead").unwrap_or(0);
    if ring_look < limits.min_ring_lookahead_bytes {
        bad.push(format!(
            "ring+lookahead reserve {ring_look} < {}: the cross-layer lookahead store is not charged",
            limits.min_ring_lookahead_bytes
        ));
    }
    if limits.require_overlap
        && (u64_at(r, "overlap.cross_layer_intersections") == Some(0)
            || f64_at(r, "overlap.cross_layer_overlap_ms").unwrap_or(0.0) <= 0.0)
    {
        bad.push(
            "no cross-layer event intersection (copy of layer L+1 overlapping compute of layer L): an overlap win cannot be claimed"
                .into(),
        );
    }
    bad
}

/// A forced layer-major request too short to qualify (at most one tile): the
/// selector may legitimately bypass it, but only explicitly. Accepts a
/// layer-major receipt (full checks) or an incumbent receipt that names its
/// bypass reason.
pub fn validate_forced_short(receipt: Option<&Value>, limits: &Limits) -> Vec<String> {
    match receipt.and_then(|r| r.get("route")).and_then(Value::as_str) {
        Some("layer-major") => validate(receipt, Expect::LayerMajor, limits),
        _ => {
            let mut bad = validate(receipt, Expect::Incumbent, limits);
            let named = receipt
                .and_then(|r| r.get("bypass_reason"))
                .is_some_and(Value::is_string);
            if !named {
                bad.push("short forced request ran the incumbent without a named bypass_reason".into());
            }
            bad
        }
    }
}

/// A claim that the schedule overlapped must carry event evidence.
pub fn overlap_claim_allowed(receipt: &Value) -> bool {
    u64_at(receipt, "overlap.cross_layer_intersections").unwrap_or(0) > 0
        && f64_at(receipt, "overlap.cross_layer_overlap_ms").unwrap_or(0.0) > 0.0
}

/// Two arms with different launch manifests or tile sizes are not bitwise
/// transport-only comparisons. `Err` names why a "bitwise transport" claim
/// is refused.
pub fn bitwise_transport_claim(base: &Value, cand: &Value) -> Result<(), String> {
    for key in ["launch_cuts", "windows"] {
        if base.get(key) != cand.get(key) {
            return Err(format!("unequal {key}: not a bitwise transport-only comparison"));
        }
    }
    for key in ["tile_rows", "entry_pos", "prompt_rows"] {
        if u64_at(base, key) != u64_at(cand, key) {
            return Err(format!("unequal {key}: not a bitwise transport-only comparison"));
        }
    }
    if base.get("route") != cand.get("route") {
        return Err("different routes are never a bitwise transport-only comparison".into());
    }
    Ok(())
}

// ── required-check accounting ──────────────────────────────────────────

pub struct Checks {
    required: Vec<String>,
    rows: Vec<Value>,
}

impl Checks {
    pub fn new(required: &[&str]) -> Self {
        Self {
            required: required.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    pub fn pass(&mut self, name: &str, detail: Value) {
        eprintln!("qwen4_slowlink_gate: PASS {name}");
        self.rows.push(json!({"name": name, "status": "pass", "detail": detail}));
    }

    pub fn fail(&mut self, name: &str, reason: impl Into<String>, detail: Value) {
        let reason = reason.into();
        eprintln!("qwen4_slowlink_gate: FAIL {name}: {reason}");
        self.rows
            .push(json!({"name": name, "status": "fail", "reason": reason, "detail": detail}));
    }

    pub fn skip(&mut self, name: &str, reason: impl Into<String>) {
        let reason = reason.into();
        eprintln!("qwen4_slowlink_gate: SKIP {name}: {reason}");
        self.rows
            .push(json!({"name": name, "status": "skipped", "reason": reason}));
    }

    pub fn push(&mut self, name: &str, problems: Vec<String>, detail: Value) {
        if problems.is_empty() {
            self.pass(name, detail);
        } else {
            self.fail(name, problems.join("; "), detail);
        }
    }

    /// `fail` if any check failed; `incomplete` if no required check was
    /// listed or any required one has no executed (pass/fail) row; else
    /// `pass`.
    pub fn status(&self) -> &'static str {
        let executed = |name: &str| {
            self.rows.iter().any(|row| {
                row["name"] == name && (row["status"] == "pass" || row["status"] == "fail")
            })
        };
        if self.rows.iter().any(|row| row["status"] == "fail") {
            "fail"
        } else if self.required.is_empty() || self.required.iter().any(|n| !executed(n)) {
            "incomplete"
        } else {
            "pass"
        }
    }

    pub fn missing(&self) -> Vec<String> {
        self.required
            .iter()
            .filter(|name| {
                !self.rows.iter().any(|row| {
                    row["name"] == name.as_str()
                        && (row["status"] == "pass" || row["status"] == "fail")
                })
            })
            .cloned()
            .collect()
    }

    pub fn json(&self) -> Value {
        json!({
            "status": self.status(),
            "required": self.required,
            "missing": self.missing(),
            "checks": self.rows,
        })
    }
}

// ── fixture ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct FixturePrompt {
    pub id: String,
    pub kind: String,
    pub prompt_ids: Vec<u32>,
    /// Natural-text prompts: tokenized by the gate after the load
    /// (`prompt_ids` is empty until then).
    pub text: Option<String>,
    /// Resume: the first `entry` ids are a previously prefilled prefix.
    pub entry: usize,
    /// Original launch cuts (row indices) the prompt is driven through.
    pub launch_cuts: Vec<usize>,
    /// Session snapshot boundaries the fixture requests.
    pub requested_boundaries: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub max_seq: usize,
    pub decode_tokens: usize,
    pub repeats: usize,
    pub head_rows: usize,
    /// Session-cache budget of the `state` mode's bundle (bytes).
    pub session_cache_bytes: u64,
    /// Timed prefills per prompt in `perf` mode (default 1).
    pub perf_repeats: usize,
    pub prompts: Vec<FixturePrompt>,
    pub continuation_ids: Vec<u32>,
    pub failure_injections: Vec<String>,
}

fn usize_list(value: Option<&Value>, what: &str) -> Result<Vec<usize>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("{what}: not an unsigned integer list"))
            })
            .collect(),
        Some(_) => Err(format!("{what}: not a list")),
    }
}

fn u32_list(value: Option<&Value>, what: &str) -> Result<Vec<u32>, String> {
    usize_list(value, what)?
        .into_iter()
        .map(|n| u32::try_from(n).map_err(|_| format!("{what}: id {n} exceeds u32")))
        .collect()
}

impl Fixture {
    /// Parse and validate a fixture. A fixture that omits a required case
    /// (prompt rows 512/1536/1537/8192/16384, an irregular, a resume and a
    /// boundary prompt, a continuation and every named failure injection) is
    /// refused rather than silently narrowed.
    pub fn parse(value: &Value) -> Result<Self, String> {
        if value.get("schema").and_then(Value::as_str) != Some(FIXTURE_SCHEMA) {
            return Err(format!("fixture schema is not {FIXTURE_SCHEMA}"));
        }
        let usize_key = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("fixture key {key} missing or not an unsigned integer"))
        };
        let max_seq = usize_key("max_seq")?;
        let decode_tokens = usize_key("decode_tokens")?;
        let repeats = usize_key("repeats")?;
        let head_rows = usize_key("head_rows")?;
        if repeats < 2 {
            return Err("fixture repeats must be at least 2 (determinism needs a repeat)".into());
        }
        if decode_tokens == 0 || head_rows == 0 || head_rows > 8 {
            return Err("fixture decode_tokens must be > 0 and head_rows in 1..=8 (bounded head scoring)".into());
        }
        let session_cache_bytes = value
            .get("session_cache_bytes")
            .map_or(Some(2u64 << 30), Value::as_u64)
            .ok_or("fixture session_cache_bytes is not an unsigned integer")?;
        // In-process timed prefill repetitions of `perf` mode. The delivery
        // observations are fresh processes, so the default is one.
        let perf_repeats = value
            .get("perf_repeats")
            .map_or(Some(1u64), Value::as_u64)
            .filter(|&n| n >= 1)
            .ok_or("fixture perf_repeats is not a positive integer")? as usize;
        let mut prompts = Vec::new();
        for item in value
            .get("prompts")
            .and_then(Value::as_array)
            .ok_or("fixture has no prompts array")?
        {
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .ok_or("a fixture prompt has no id")?
                .to_string();
            let kind = item
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("prompt {id}: no kind"))?
                .to_string();
            if !["rows", "irregular", "resume", "boundary", "natural"].contains(&kind.as_str()) {
                return Err(format!("prompt {id}: unknown kind {kind}"));
            }
            let prompt_ids = u32_list(item.get("prompt_ids"), &format!("prompt {id} prompt_ids"))?;
            // A natural-text prompt carries `text` instead of ids: the gate
            // tokenizes it with the artifact's own tokenizer after the load.
            let text = match item.get("text") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
                Some(_) => return Err(format!("prompt {id}: text must be a non-empty string")),
            };
            if kind != "natural" && text.is_some() {
                return Err(format!("prompt {id}: only a natural prompt may carry text"));
            }
            if prompt_ids.is_empty() == text.is_none() {
                return Err(format!("prompt {id}: give exactly one of prompt_ids and text"));
            }
            let entry = item.get("entry").and_then(Value::as_u64).unwrap_or(0) as usize;
            let launch_cuts = usize_list(item.get("launch_cuts"), &format!("prompt {id} launch_cuts"))?;
            let requested_boundaries = usize_list(
                item.get("requested_boundaries"),
                &format!("prompt {id} requested_boundaries"),
            )?;
            if entry >= prompt_ids.len() && kind == "resume" {
                return Err(format!("prompt {id}: resume entry {entry} leaves no continuation"));
            }
            if launch_cuts.windows(2).any(|w| w[0] >= w[1])
                || launch_cuts.iter().any(|&c| c == 0 || c >= prompt_ids.len())
            {
                return Err(format!(
                    "prompt {id}: launch_cuts must be strictly ascending interior row indices"
                ));
            }
            if !prompt_ids.is_empty()
                && prompt_ids.len() + value["decode_tokens"].as_u64().unwrap_or(0) as usize + 1 > max_seq
            {
                return Err(format!(
                    "prompt {id}: {} rows + decode do not fit max_seq {max_seq}",
                    prompt_ids.len()
                ));
            }
            prompts.push(FixturePrompt {
                id,
                kind,
                prompt_ids,
                text,
                entry,
                launch_cuts,
                requested_boundaries,
            });
        }
        let ids: std::collections::BTreeSet<&str> = prompts.iter().map(|p| p.id.as_str()).collect();
        if ids.len() != prompts.len() {
            return Err("fixture prompt ids are not unique".into());
        }
        for rows in REQUIRED_PROMPT_ROWS {
            if !prompts.iter().any(|p| p.kind == "rows" && p.prompt_ids.len() == rows) {
                return Err(format!("fixture lacks the required {rows}-row prompt"));
            }
        }
        for kind in ["irregular", "resume", "boundary", "natural"] {
            if !prompts.iter().any(|p| p.kind == kind) {
                return Err(format!("fixture lacks a {kind} prompt"));
            }
        }
        let continuation_ids = u32_list(value.get("continuation_ids"), "continuation_ids")?;
        if continuation_ids.is_empty() {
            return Err("fixture has no continuation_ids (divergent continuation)".into());
        }
        let failure_injections: Vec<String> = value
            .get("failure_injections")
            .and_then(Value::as_array)
            .ok_or("fixture has no failure_injections")?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        for point in FAILURE_POINTS {
            if !failure_injections.iter().any(|p| p == point) {
                return Err(format!("fixture lacks the {point} failure injection"));
            }
        }
        if let Some(bad) = failure_injections.iter().find(|p| !FAILURE_POINTS.contains(&p.as_str())) {
            return Err(format!("unknown failure injection {bad}"));
        }
        Ok(Self {
            max_seq,
            decode_tokens,
            repeats,
            head_rows,
            session_cache_bytes,
            perf_repeats,
            prompts,
            continuation_ids,
            failure_injections,
        })
    }
}

// ── manifest identity ──────────────────────────────────────────────────

pub const MANIFEST_REQUIRED: [&str; 18] = [
    "source.base",
    "source.tip",
    "binary.sha256",
    "model.path",
    "model.size_bytes",
    "model.sha256",
    "ple.source",
    "ple.format",
    "ple.sidecars",
    "kernel.code_object_hashes",
    "host.hostname",
    "host.topology",
    "device.uuid",
    "device.arch",
    "link.gbps",
    "link.topology_walk",
    "memory_ledger.device_gib",
    "memory_ledger.host_unreclaimable_gib",
];

#[derive(Clone, Debug)]
pub struct Observed {
    pub binary_sha256: String,
    pub model_size_bytes: u64,
    pub gpu_arch: String,
}

fn is_hex(value: Option<&str>, len: usize) -> bool {
    value.is_some_and(|s| s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Missing keys and identity mismatches between a manifest and what the
/// process actually observes. Empty = identity holds.
pub fn manifest_problems(manifest: &Value, observed: &Observed) -> Vec<String> {
    let mut bad = Vec::new();
    for path in MANIFEST_REQUIRED {
        match get(manifest, path) {
            None | Some(Value::Null) => bad.push(format!("manifest key {path} missing")),
            Some(Value::String(s)) if s.is_empty() => bad.push(format!("manifest key {path} empty")),
            _ => {}
        }
    }
    if !bad.is_empty() {
        return bad;
    }
    for path in ["source.base", "source.tip"] {
        if !is_hex(get(manifest, path).and_then(Value::as_str), 40) {
            bad.push(format!("manifest {path} is not a full 40-hex commit"));
        }
    }
    if !is_hex(get(manifest, "binary.sha256").and_then(Value::as_str), 64) {
        bad.push("manifest binary.sha256 is not a 64-hex digest".into());
    } else if get(manifest, "binary.sha256").and_then(Value::as_str)
        != Some(observed.binary_sha256.as_str())
    {
        bad.push(format!(
            "binary identity mismatch: manifest {} != running {}",
            get(manifest, "binary.sha256").unwrap(),
            observed.binary_sha256
        ));
    }
    if !is_hex(get(manifest, "model.sha256").and_then(Value::as_str), 64) {
        bad.push("manifest model.sha256 is not a 64-hex digest".into());
    }
    if u64_at(manifest, "model.size_bytes") != Some(observed.model_size_bytes) {
        bad.push(format!(
            "model identity mismatch: manifest size {:?} != file size {}",
            u64_at(manifest, "model.size_bytes"),
            observed.model_size_bytes
        ));
    }
    if get(manifest, "device.arch").and_then(Value::as_str) != Some(observed.gpu_arch.as_str()) {
        bad.push(format!(
            "device arch mismatch: manifest {:?} != {}",
            get(manifest, "device.arch"),
            observed.gpu_arch
        ));
    }
    for path in ["memory_ledger.device_gib", "memory_ledger.host_unreclaimable_gib", "link.gbps"] {
        match f64_at(manifest, path) {
            Some(v) if v.is_finite() && v > 0.0 => {}
            _ => bad.push(format!("manifest {path} is not a positive number")),
        }
    }
    bad
}

// ── KLD receipts and the paired verdict ────────────────────────────────

pub const KLD_ROWS: u64 = 6120;
pub const KLD_SEQUENCES: u64 = 24;
pub const KLD_HI_LIMIT: f64 = 0.010;

/// A full-gate S11 decode receipt: not a partition, every row, every
/// corpus sequence, the decode route.
pub fn kld_receipt_problems(receipt: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    if receipt.get("schema").and_then(Value::as_str) != Some("hipfire.qwen4.s11ref_klrows.v1") {
        bad.push("not a full-gate s11ref receipt (schema; a partition is never the gate)".into());
    }
    if u64_at(receipt, "rows") != Some(KLD_ROWS) {
        bad.push(format!("rows {:?} != {KLD_ROWS}: partial corpus", receipt.get("rows")));
    }
    if u64_at(receipt, "sequences") != Some(KLD_SEQUENCES) {
        bad.push(format!(
            "sequences {:?} != {KLD_SEQUENCES}: partial corpus",
            receipt.get("sequences")
        ));
    }
    if receipt.get("route").and_then(Value::as_str) != Some("decode") {
        bad.push("route is not decode".into());
    }
    for key in ["model_sha256", "ref_sha256", "corpus_sha256", "binary_sha256", "output_sha256"] {
        if !is_hex(receipt.get(key).and_then(Value::as_str), 64) {
            bad.push(format!("receipt {key} is not a 64-hex digest"));
        }
    }
    bad
}

/// Same reference, corpus, context and route; different weights/binary are
/// expected (candidate versus beta).
pub fn kld_receipts_comparable(base: &Value, cand: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    for key in ["ref_sha256", "corpus_sha256", "n_ctx", "route", "rows", "sequences", "estimator"] {
        if base.get(key) != cand.get(key) {
            bad.push(format!("receipts differ in {key}: not the same ordering/reference/config"));
        }
    }
    bad
}

/// Mean KL per corpus sequence: `counts[i]` rows belong to sequence `i`, in
/// canonical row order. Aggregating first means the interval's unit is the
/// sequence, never the token.
pub fn sequence_means(rows: &[f64], counts: &[usize]) -> Result<Vec<f64>, String> {
    let total: usize = counts.iter().sum();
    if total != rows.len() {
        return Err(format!("sequence row counts sum to {total}, the rows are {}", rows.len()));
    }
    if counts.iter().any(|&c| c == 0) {
        return Err("a corpus sequence has no scored rows".into());
    }
    let mut at = 0;
    Ok(counts
        .iter()
        .map(|&c| {
            let mean = rows[at..at + c].iter().sum::<f64>() / c as f64;
            at += c;
            mean
        })
        .collect())
}

/// Two-sided 95% Student-t critical value for `df` degrees of freedom
/// (conservative above 30).
pub fn t_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060,
        2.056, 2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::INFINITY,
        1..=30 => TABLE[df - 1],
        31..=40 => 2.021,
        41..=60 => 2.000,
        61..=120 => 1.980,
        _ => 1.960,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedCi {
    pub n: usize,
    pub mean: f64,
    pub se: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Paired candidate-minus-beta interval over per-sequence means.
pub fn paired_ci(base: &[f64], cand: &[f64]) -> Result<PairedCi, String> {
    if base.len() != cand.len() || base.len() < 2 {
        return Err(format!(
            "paired interval needs equal per-sequence vectors of at least 2 ({} vs {})",
            base.len(),
            cand.len()
        ));
    }
    if base.iter().chain(cand).any(|v| !v.is_finite()) {
        return Err("non-finite per-sequence KL".into());
    }
    let diffs: Vec<f64> = cand.iter().zip(base).map(|(c, b)| c - b).collect();
    let n = diffs.len();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    let half = t_975(n - 1) * se;
    Ok(PairedCi { n, mean, se, lo: mean - half, hi: mean + half })
}

/// Pass `lo <= 0`; otherwise require `hi <= +0.010`.
pub fn paired_verdict(ci: &PairedCi) -> bool {
    ci.lo <= 0.0 || ci.hi <= KLD_HI_LIMIT
}
