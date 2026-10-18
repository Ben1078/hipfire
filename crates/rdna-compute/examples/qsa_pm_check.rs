// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Developer-only QSA gathered-route screen (`qsa-source-v1` snapshots; lab
//! feature, no production dispatch edits).
//!
//! `qsa_pm_check SNAPSHOT_DIR... --out OUT.jsonl [options]` runs three arms on
//! every real snapshot (`snapshot.json` under each directory, recursively,
//! `pre-prologue` dumps skipped):
//!
//! * `simt`: `indexed_attention_attention_batch`, the incumbent. The process
//!   MUST have `HIPFIRE_QWEN4_QSA_WMMA_GATHER=0` (checked), and the launched
//!   kernel name (`Gpu::last_launched_kernel`) must be
//!   `indexed_attention_attention_<fmt>_batched_hg4`, recorded per snapshot;
//!   any other route is rejected unless `--allow-non-hg4`.
//! * `hip`: `indexed_attention_gathered_batch(.., false)`, the hipcc gathered
//!   WMMA route, converter (K/V cache -> F16 scratch) included.
//! * `pm`: `indexed_attention_gathered_batch(.., true)`, the builder-module
//!   attend over the same hipcc converter, converter included.
//!
//! `--arms LIST` (comma list drawn from `simt`, `hip`, `pm`; default all three)
//! runs only the named arms in the GPU phases (arm runs, hashes, determinism,
//! timing, derived cases) and in the numeric passes. `pm` requires `hip` (the
//! byte comparison is PM vs hipcc). `HIPFIRE_QWEN4_QSA_WMMA_GATHER=0`, the hg4
//! kernel-name guard and every SIMT comparison (SIMT vs hip/pm/eager, saving
//! vs SIMT, kill rule) apply only when `simt` is selected, so `--arms hip,pm`
//! runs on a pack that carries the gathered module but no SIMT incumbent.
//! Arm-keyed JSON carries only the selected arms and the run header records
//! `selected_arms`; the default is the three-arm behavior. A `--reference-only`
//! pass MUST be given the same `--arms` as the GPU pass that saved the samples.
//!
//! Formats are explicit (`QsaKvFormat` F32 / FP8 / Q8 from `geometry.format`):
//! raw Q8_0 bytes (34-byte blocks: f16 scale + 32 i8 codes, head-major, 8
//! blocks per 256-wide head) are uploaded as Raw and decoded on the host for
//! the references; they are never read as F32. A Q8 header MUST carry the
//! explicit strides (`kv_row_bytes`, `kv_head_bytes`, `kv_block_elems`,
//! `kv_block_bytes`, `kv_blocks_per_head`, `qgate_row_stride_f32`,
//! `eager_row_stride_f32`, `selected_row_stride_i32`); every present stride,
//! file size and sha256 is checked against the geometry; non-finite Q/K/V
//! (rows below the visible end) and unsupported formats are rejected.
//!
//! Selected-list semantics follow the kernel: row `r` sees
//! `visible = position_start + r + 1`; its list length is
//! `min(capacity, min(budget_blocks, visible / compress) * compress +
//! visible % compress)`; entries are read in order from a `capacity`-strided
//! row; an entry is valid iff `0 <= t < min(full_capacity, visible)`;
//! duplicates count once per slot; no valid entry => output 0. Output element
//! `(r, h, c)` is `softmax(q.k / sqrt(dim)) v / (1 + exp(-gate))` with the
//! query row `[(r*heads+h)*2*dim ..][q(dim), gate(dim)]`.
//!
//! Phases: (1) GPU "hold": per snapshot, replayed arm runs over differently
//! poisoned outputs (hashes, determinism, pairwise byte comparison), sampled
//! arm rows persisted under `--work`, optional full outputs under `--outputs`,
//! and, with `--time`, converter-inclusive ABBA timing of native snapshots
//! under its own `--hold-seconds` deadline; (2) `--derived K` labeled edge
//! cases (never timed) on K snapshots; (3) CPU numerics, outside the hold: an
//! independent f64 attention from the decoded cache and a BF16-rounded-storage
//! reference (Q/K/V/gate and the unnormalized probability rounded to BF16, the
//! denominator from unrounded probabilities) over deterministic sampled rows
//! (all heads, all channels; `--full-reference` for every row). The eager
//! output is hashed for identity only and is never a numeric oracle.
//! `--reference-only` skips the GPU and recomputes phase (3) from saved arm
//! samples / full outputs.

use rdna_compute::tensor_ops::{
    indexed_attention_attention_batch, indexed_attention_gathered_batch,
    IndexedAttentionAttentionBatch, QsaKvFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Debug) -> String { format!("{e:?}") }
fn sha256(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn num(x: f64) -> Value { if x.is_finite() { json!(x) } else { json!(format!("{x}")) } }
fn i32_bytes(v: &[i32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm { Simt, Hip, Pm }
impl Arm {
    const ALL: [Arm; 3] = [Arm::Simt, Arm::Hip, Arm::Pm];
    fn key(self) -> &'static str { match self { Arm::Simt => "simt", Arm::Hip => "hip", Arm::Pm => "pm" } }
    fn idx(self) -> usize { self as usize }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimeSel { None, All, First(usize) }

struct Opts {
    dirs: Vec<PathBuf>,
    out: PathBuf,
    work: PathBuf,
    outputs: Option<PathBuf>,
    time: TimeSel,
    rounds: usize,
    calls: usize,
    warm: usize,
    hold_seconds: f64,
    layers: usize,
    replays: usize,
    sample_rows: usize,
    full_reference: bool,
    derived: usize,
    edge_rows: usize,
    edge_sample_rows: usize,
    reference_only: bool,
    allow_non_hg4: bool,
    allow_truncated: bool,
    gpu_only: bool,
    arms: [bool; 3],
}
impl Opts {
    fn has(&self, arm: Arm) -> bool { self.arms[arm.idx()] }
    /// Selected arms in canonical order (simt, hip, pm).
    fn selected(&self) -> Vec<Arm> { Arm::ALL.into_iter().filter(|a| self.has(*a)).collect() }
}

/// JSON object keyed by arm name over the selected arms only.
fn arm_json(o: &Opts, f: impl Fn(Arm) -> Value) -> serde_json::Map<String, Value> {
    o.selected().into_iter().map(|a| (a.key().to_string(), f(a))).collect()
}

fn parse_arms(list: &str) -> Result<[bool; 3]> {
    if list.trim().is_empty() { return Err("--arms: empty list (expected a comma list of simt, hip, pm)".into()) }
    let mut sel = [false; 3];
    for name in list.split(',') {
        let name = name.trim();
        let Some(arm) = Arm::ALL.into_iter().find(|a| a.key() == name) else {
            return Err(format!("--arms: unknown arm {name:?} (expected a comma list of simt, hip, pm)"));
        };
        if std::mem::replace(&mut sel[arm.idx()], true) { return Err(format!("--arms: duplicate arm {name:?}")) }
    }
    if sel[Arm::Pm.idx()] && !sel[Arm::Hip.idx()] { return Err("--arms: pm requires hip (the byte comparison is PM vs hipcc)".into()) }
    Ok(sel)
}

fn parse_args() -> Result<Opts> {
    let mut o = Opts {
        dirs: Vec::new(), out: PathBuf::new(), work: PathBuf::new(), outputs: None, time: TimeSel::None,
        rounds: 3, calls: 20, warm: 3, hold_seconds: 600.0, layers: 12, replays: 2, sample_rows: 64,
        full_reference: false, derived: 0, edge_rows: 512, edge_sample_rows: 32, reference_only: false,
        allow_non_hg4: false, allow_truncated: false, gpu_only: false, arms: [true; 3],
    };
    let mut out = None;
    let mut work = None;
    let mut it = std::env::args().skip(1);
    fn val(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
        it.next().ok_or_else(|| format!("{flag} needs a value"))
    }
    fn usz(s: String, flag: &str) -> Result<usize> { s.parse().map_err(|e| format!("{flag}: {e}")) }
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = Some(PathBuf::from(val(&mut it, &a)?)),
            "--work" => work = Some(PathBuf::from(val(&mut it, &a)?)),
            "--outputs" => o.outputs = Some(PathBuf::from(val(&mut it, &a)?)),
            "--time" => {
                let v = val(&mut it, &a)?;
                o.time = if v == "all" { TimeSel::All } else { TimeSel::First(usz(v, &a)?) };
            }
            "--rounds" => o.rounds = usz(val(&mut it, &a)?, &a)?,
            "--calls" => o.calls = usz(val(&mut it, &a)?, &a)?,
            "--warmup" => o.warm = usz(val(&mut it, &a)?, &a)?,
            "--hold-seconds" => o.hold_seconds = val(&mut it, &a)?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--layers" => o.layers = usz(val(&mut it, &a)?, &a)?,
            "--replays" => o.replays = usz(val(&mut it, &a)?, &a)?,
            "--sample-rows" => o.sample_rows = usz(val(&mut it, &a)?, &a)?,
            "--full-reference" => o.full_reference = true,
            "--derived" => o.derived = usz(val(&mut it, &a)?, &a)?,
            "--edge-rows" => o.edge_rows = usz(val(&mut it, &a)?, &a)?,
            "--edge-sample-rows" => o.edge_sample_rows = usz(val(&mut it, &a)?, &a)?,
            "--reference-only" => o.reference_only = true,
            "--allow-non-hg4" => o.allow_non_hg4 = true,
            "--allow-truncated" => o.allow_truncated = true,
            "--gpu-only" => o.gpu_only = true,
            "--arms" => o.arms = parse_arms(&val(&mut it, &a)?)?,
            s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
            _ => o.dirs.push(PathBuf::from(&a)),
        }
    }
    o.out = out.ok_or("--out OUT.jsonl required")?;
    o.work = work.unwrap_or_else(|| PathBuf::from(format!("{}.work", o.out.display())));
    if o.rounds == 0 || o.calls == 0 || o.replays == 0 || o.layers == 0 { return Err("--rounds/--calls/--replays/--layers must be >= 1".into()) }
    if o.edge_rows == 0 { return Err("--edge-rows must be >= 1".into()) }
    Ok(o)
}

fn snapshots(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "pre-prologue") { continue }
            snapshots(&p, out)?;
        } else if p.file_name().is_some_and(|n| n == "snapshot.json") {
            out.push(p);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- geometry

#[derive(Clone, Copy, Debug)]
struct Geo {
    rows: usize, position_start: usize, heads: usize, kv_heads: usize, dim: usize,
    budget_blocks: usize, compress: usize, capacity: usize, full_capacity: usize,
}
impl Geo {
    fn end(&self) -> usize { self.position_start + self.rows }
    fn visible(&self, row: usize) -> usize { self.position_start + row + 1 }
    /// The kernels' per-row selected list length.
    fn selected_len(&self, row: usize) -> usize {
        let visible = self.visible(row);
        let blocks = visible / self.compress;
        let chosen = self.budget_blocks.min(blocks);
        (chosen * self.compress + visible - blocks * self.compress).min(self.capacity)
    }
    fn json(&self, format: &str) -> Value {
        json!({"rows": self.rows, "position_start": self.position_start, "heads": self.heads,
            "kv_heads": self.kv_heads, "dim": self.dim, "budget_blocks": self.budget_blocks,
            "compress": self.compress, "capacity": self.capacity, "full_capacity": self.full_capacity,
            "format": format})
    }
}

fn fmt_name(f: QsaKvFormat) -> &'static str { f.name() }
fn fmt_tag(f: QsaKvFormat) -> &'static str {
    match f { QsaKvFormat::F32 => "f32", QsaKvFormat::Fp8 => "fp8", QsaKvFormat::Q8 => "q8", QsaKvFormat::Fp8Fnuz => "fp8_fnuz" }
}

// ------------------------------------------------------------------ decode

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((bits >> 10) & 31) as i32;
    let m = (bits & 1023) as f32;
    if e == 0 { sign * m * 2f32.powi(-24) }
    else if e == 31 { if bits & 1023 == 0 { sign * f32::INFINITY } else { f32::NAN } }
    else { sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
}
/// E4M3 (OCP FN, as the gfx12 hardware converts it): NaN is S.1111.111.
fn e4m3(code: u8) -> f32 {
    let sign = if code & 128 != 0 { -1.0f32 } else { 1.0 };
    let e = ((code >> 3) & 15) as i32;
    let m = (code & 7) as f32;
    if e == 15 && (code & 7) == 7 { f32::NAN }
    else if e == 0 { sign * m * 2f32.powi(-9) }
    else { sign * (1.0 + m / 8.0) * 2f32.powi(e - 7) }
}

/// Exact f32 decode of cache rows `[0, rows)`; rejects non-finite values.
fn decode_cache(bytes: &[u8], g: &Geo, fmt: QsaKvFormat, rows: usize) -> Result<Vec<f32>> {
    let (kvh, dim) = (g.kv_heads, g.dim);
    let rb = fmt.kv_row_bytes(kvh, dim);
    let mut out = Vec::with_capacity(rows * kvh * dim);
    for t in 0..rows {
        let row = &bytes[t * rb..(t + 1) * rb];
        match fmt {
            QsaKvFormat::F32 => {
                for c in row.chunks_exact(4) {
                    let x = f32::from_le_bytes(c.try_into().unwrap());
                    if !x.is_finite() { return Err(format!("non-finite F32 cache value at token {t}")) }
                    out.push(x);
                }
            }
            QsaKvFormat::Q8 => {
                for h in 0..kvh {
                    for b in 0..dim / 32 {
                        let base = (h * (dim / 32) + b) * 34;
                        let scale = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
                        if !scale.is_finite() { return Err(format!("non-finite Q8 scale at token {t} head {h} block {b}")) }
                        for i in 0..32 { out.push((row[base + 2 + i] as i8) as f32 * scale); }
                    }
                }
            }
            QsaKvFormat::Fp8 => {
                for h in 0..kvh {
                    let scale = f16_to_f32(u16::from_le_bytes([row[kvh * dim + 2 * h], row[kvh * dim + 2 * h + 1]]));
                    if !scale.is_finite() { return Err(format!("non-finite FP8 scale at token {t} head {h}")) }
                    for c in 0..dim {
                        let x = e4m3(row[h * dim + c]);
                        if x.is_nan() { return Err(format!("FP8 NaN code at token {t} head {h} channel {c}")) }
                        out.push(x * scale);
                    }
                }
            }
            QsaKvFormat::Fp8Fnuz => return Err("FP8 FNUZ snapshots are unsupported".into()),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- snapshot

struct Loaded {
    path: PathBuf,
    id: String,
    header: Value,
    geo: Geo,
    fmt: QsaKvFormat,
    q: Vec<f32>,
    k: Vec<u8>,
    v: Vec<u8>,
    sel: Vec<i32>,
    eager: Option<Vec<u8>>,
    kd: Vec<f32>,
    vd: Vec<f32>,
    inputs: Value,
}

fn load(path: &Path) -> Result<Loaded> {
    let ctx = |m: String| format!("{}: {m}", path.display());
    let root = path.parent().ok_or_else(|| ctx("no parent".into()))?;
    let raw = std::fs::read(path).map_err(|e| ctx(e.to_string()))?;
    let header: Value = serde_json::from_slice(&raw).map_err(|e| ctx(e.to_string()))?;
    if header["schema"] != "qsa-source-v1" { return Err(ctx("unsupported schema".into())) }
    let g = &header["geometry"];
    let n = |k: &str| -> Result<usize> {
        g[k].as_u64().map(|v| v as usize).ok_or_else(|| ctx(format!("geometry.{k} missing or not an integer")))
    };
    let geo = Geo {
        rows: n("rows")?, position_start: n("position_start")?, heads: n("heads")?, kv_heads: n("kv_heads")?,
        dim: n("dim")?, budget_blocks: n("budget_blocks")?, compress: n("compress")?,
        capacity: n("capacity")?, full_capacity: n("full_capacity")?,
    };
    let fname = g["format"].as_str().ok_or_else(|| ctx("geometry.format missing".into()))?;
    let fmt = match fname {
        "f32" => QsaKvFormat::F32,
        "fp8" => QsaKvFormat::Fp8,
        "q8" => QsaKvFormat::Q8,
        other => return Err(ctx(format!("unsupported snapshot format {other:?}"))),
    };
    if let Some(s) = header["identity"]["source_format"].as_str() {
        if s != fname { return Err(ctx(format!("identity.source_format {s} != geometry.format {fname}"))) }
    }
    if geo.rows == 0 || geo.heads == 0 || geo.kv_heads == 0 || geo.heads % geo.kv_heads != 0
        || geo.compress == 0 || geo.capacity == 0 || geo.dim == 0 || geo.position_start + geo.rows > geo.full_capacity
        || geo.position_start + geo.rows > i32::MAX as usize
    {
        return Err(ctx("inconsistent geometry".into()));
    }
    if !fmt.supports(geo.kv_heads, geo.dim) { return Err(ctx(format!("format {fname} unsupported for kv_heads {} dim {}", geo.kv_heads, geo.dim))) }
    let rb = fmt.kv_row_bytes(geo.kv_heads, geo.dim);
    let mut strides = vec![
        ("kv_row_bytes", rb), ("qgate_row_stride_f32", geo.heads * geo.dim * 2),
        ("eager_row_stride_f32", geo.heads * geo.dim), ("selected_row_stride_i32", geo.capacity),
    ];
    if fmt == QsaKvFormat::Q8 {
        strides.extend([
            ("kv_head_bytes", geo.dim / 32 * 34), ("kv_block_elems", 32), ("kv_block_bytes", 34),
            ("kv_blocks_per_head", geo.dim / 32),
        ]);
    }
    for (key, want) in &strides {
        match g.get(*key).filter(|v| !v.is_null()) {
            Some(v) => if v.as_u64() != Some(*want as u64) {
                return Err(ctx(format!("stride mismatch geometry.{key} = {v}, expected {want}")))
            },
            None => if fmt == QsaKvFormat::Q8 { return Err(ctx(format!("Q8 snapshot lacks explicit stride geometry.{key}"))) },
        }
    }
    let files = &header["files"];
    let mut hashes = serde_json::Map::new();
    let mut get = |name: &str, expect: usize, required: bool| -> Result<Option<Vec<u8>>> {
        let entry = &files[name];
        if entry.is_null() {
            return if required { Err(ctx(format!("files.{name} missing"))) } else { Ok(None) };
        }
        let b = std::fs::read(root.join(name)).map_err(|e| ctx(format!("{name}: {e}")))?;
        if b.len() != expect { return Err(ctx(format!("{name}: {} bytes, geometry/strides require {expect}", b.len()))) }
        if entry["bytes"].as_u64() != Some(b.len() as u64) { return Err(ctx(format!("{name}: header byte count {} != file {}", entry["bytes"], b.len()))) }
        let h = sha256(&b);
        if entry["sha256"] != h.as_str() { return Err(ctx(format!("hash mismatch {name}"))) }
        hashes.insert(name.into(), json!({"bytes": b.len(), "sha256": h}));
        Ok(Some(b))
    };
    let qb = get("qgate.f32", geo.rows * geo.heads * geo.dim * 2 * 4, true)?.unwrap();
    let k = get("keys.source", geo.full_capacity * rb, true)?.unwrap();
    let v = get("values.source", geo.full_capacity * rb, true)?.unwrap();
    let sb = get("selected.i32", geo.rows * geo.capacity * 4, true)?.unwrap();
    let eager = get("eager.f32", geo.rows * geo.heads * geo.dim * 4, false)?;
    let q: Vec<f32> = qb.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    if let Some(i) = q.iter().position(|x| !x.is_finite()) { return Err(ctx(format!("non-finite qgate value at element {i}"))) }
    let sel: Vec<i32> = sb.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let end = geo.end();
    let kd = decode_cache(&k, &geo, fmt, end).map_err(|e| ctx(format!("keys: {e}")))?;
    let vd = decode_cache(&v, &geo, fmt, end).map_err(|e| ctx(format!("values: {e}")))?;
    // Selection statistics (informational): entries outside the causal window
    // are masked by the kernels; duplicates count once per slot.
    let (mut invalid, mut duplicate, mut empty_rows) = (0usize, 0usize, 0usize);
    for r in 0..geo.rows {
        let len = geo.selected_len(r);
        let vis = geo.visible(r).min(geo.full_capacity);
        let list = &sel[r * geo.capacity..r * geo.capacity + len];
        let mut seen = BTreeSet::new();
        let mut valid = 0usize;
        for &t in list {
            if t < 0 || t as usize >= vis { invalid += 1 } else { valid += 1; if !seen.insert(t) { duplicate += 1 } }
        }
        empty_rows += usize::from(valid == 0);
    }
    let snap_sha = sha256(&raw);
    let stem = path.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let stem: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    Ok(Loaded {
        path: path.to_path_buf(), id: format!("{stem}-{}", &snap_sha[..8]), header: header.clone(), geo, fmt, q, k, v, sel,
        eager,
        kd, vd,
        inputs: json!({
            "snapshot_json_sha256": snap_sha, "files": Value::Object(hashes),
            "selected_stats": {"invalid_or_future_entries": invalid, "duplicate_entries": duplicate, "rows_without_valid_entry": empty_rows},
            "strides_checked": strides.iter().map(|(k, v)| json!({"key": k, "value": v})).collect::<Vec<_>>(),
        }),
    })
}

// ------------------------------------------------------------------ device

struct Case<'a> {
    q: GpuTensor, s: GpuTensor, out: GpuTensor,
    k: &'a GpuTensor, v: &'a GpuTensor,
    geo: Geo, fmt: QsaKvFormat,
}
impl<'a> Case<'a> {
    fn new(gpu: &mut Gpu, geo: Geo, fmt: QsaKvFormat, q: &[f32], sel: &[i32], k: &'a GpuTensor, v: &'a GpuTensor) -> Result<Self> {
        let sb = i32_bytes(sel);
        Ok(Case {
            q: gpu.upload_f32(q, &[q.len()]).map_err(err)?,
            s: gpu.upload_raw(&sb, &[sb.len()]).map_err(err)?,
            out: gpu.zeros(&[geo.rows * geo.heads * geo.dim], DType::F32).map_err(err)?,
            k, v, geo, fmt,
        })
    }
    fn params(&self) -> IndexedAttentionAttentionBatch<'_> {
        let g = &self.geo;
        IndexedAttentionAttentionBatch {
            q_with_gate: &self.q, full_keys: self.k, full_values: self.v, selected: &self.s, output: &self.out,
            rows: g.rows, position_start: g.position_start, n_heads: g.heads, n_kv_heads: g.kv_heads, head_dim: g.dim,
            budget_blocks: g.budget_blocks, compress: g.compress, capacity: g.capacity, full_capacity: g.full_capacity,
            format: self.fmt, shape_selected: g.capacity,
        }
    }
    fn free(self, gpu: &mut Gpu) -> Result<()> {
        for t in [self.q, self.s, self.out] { gpu.free_tensor(t).map_err(err)?; }
        Ok(())
    }
}

fn upload_cache(gpu: &mut Gpu, bytes: &[u8], fmt: QsaKvFormat) -> Result<GpuTensor> {
    if fmt == QsaKvFormat::F32 {
        let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        gpu.upload_f32(&v, &[v.len()]).map_err(err)
    } else {
        gpu.upload_raw(bytes, &[bytes.len()]).map_err(err)
    }
}

fn launch(gpu: &mut Gpu, c: &Case<'_>, arm: Arm) -> Result<()> {
    let p = c.params();
    match arm {
        Arm::Simt => indexed_attention_attention_batch(gpu, &p),
        Arm::Hip => indexed_attention_gathered_batch(gpu, &p, false),
        Arm::Pm => indexed_attention_gathered_batch(gpu, &p, true),
    }
    .map_err(|e| format!("{} arm: {e:?}", arm.key()))
}

const POISON: [i32; 3] = [0x7f, 0xff, 0xa5];

fn run(gpu: &mut Gpu, c: &Case<'_>, arm: Arm, pattern: i32) -> Result<Vec<u8>> {
    gpu.hip.memset(&c.out.buf, pattern, c.out.byte_size()).map_err(err)?;
    launch(gpu, c, arm)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let mut bytes = vec![0u8; c.out.byte_size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &c.out.buf).map_err(err)?;
    Ok(bytes)
}

struct ArmOut { bytes: Vec<u8>, sha: String, replay_sha: Vec<String>, deterministic: bool, kernel: String, nonfinite: usize }

fn nonfinite(bytes: &[u8]) -> usize {
    bytes.chunks_exact(4).filter(|c| !f32::from_le_bytes((*c).try_into().unwrap()).is_finite()).count()
}

fn check_kernel(arm: Arm, name: &str, fmt: QsaKvFormat, allow_non_hg4: bool) -> Result<()> {
    match arm {
        Arm::Simt => {
            let want = format!("indexed_attention_attention_{}_batched_hg4", fmt_tag(fmt));
            if name != want && !allow_non_hg4 {
                return Err(format!("SIMT incumbent launched {name:?}, not {want:?} (route flags: HIPFIRE_QWEN4_QSA_WMMA_GATHER must be 0; decode-sized rows take split-KV)"));
            }
        }
        Arm::Hip => if !(name.contains("gathered_wmma_f16") && !name.contains("_pm_")) {
            return Err(format!("hipcc gathered arm launched {name:?}"));
        },
        Arm::Pm => if !name.contains("_pm_") { return Err(format!("PM arm launched {name:?}")) },
    }
    Ok(())
}

fn arm_runs(gpu: &mut Gpu, c: &Case<'_>, arm: Arm, replays: usize, allow_non_hg4: bool) -> Result<ArmOut> {
    let mut first: Option<Vec<u8>> = None;
    let mut shas = Vec::new();
    let mut kernel = String::new();
    for rep in 0..replays.max(1) {
        let b = run(gpu, c, arm, POISON[rep % POISON.len()])?;
        kernel = gpu.last_launched_kernel().unwrap_or("").to_string();
        check_kernel(arm, &kernel, c.fmt, allow_non_hg4)?;
        shas.push(sha256(&b));
        if first.is_none() { first = Some(b); }
    }
    let bytes = first.unwrap();
    let nf = nonfinite(&bytes);
    Ok(ArmOut { sha: shas[0].clone(), deterministic: shas.iter().all(|s| *s == shas[0]), replay_sha: shas, kernel, nonfinite: nf, bytes })
}

fn compare(a: &[u8], b: &[u8]) -> Value {
    let (mut differing, mut max_abs, mut nonfinite_pairs) = (0usize, 0.0f64, 0usize);
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if x != y {
            differing += 1;
            let (fx, fy) = (f32::from_le_bytes(x.try_into().unwrap()), f32::from_le_bytes(y.try_into().unwrap()));
            if fx.is_finite() && fy.is_finite() { max_abs = max_abs.max((fx as f64 - fy as f64).abs()) } else { nonfinite_pairs += 1 }
        }
    }
    json!({"differing_f32": differing, "max_abs_diff_finite": num(max_abs), "differing_nonfinite": nonfinite_pairs, "equal": differing == 0})
}

// ---------------------------------------------------------------- sampling

fn sample_rows(g: &Geo, n: usize, full: bool) -> Vec<usize> {
    if full || n >= g.rows { return (0..g.rows).collect() }
    let mut s = BTreeSet::new();
    s.insert(0);
    for d in 0..4.min(g.rows) { s.insert(g.rows - 1 - d); }
    // Selection regime changes: block budget reached, selected list clamps to capacity.
    let full_budget_visible = g.budget_blocks * g.compress;
    for v in [full_budget_visible, full_budget_visible + g.compress, g.capacity] {
        if v > g.position_start {
            let r = v - 1 - g.position_start;
            for d in [-1i64, 0, 1] {
                let rr = r as i64 + d;
                if rr >= 0 && (rr as usize) < g.rows { s.insert(rr as usize); }
            }
        }
    }
    let m = n.max(2);
    for i in 0..m { s.insert(i * (g.rows - 1) / (m - 1)); }
    s.into_iter().collect()
}

fn take_rows(full: &[u8], rows: &[usize], g: &Geo) -> Vec<f32> {
    let w = g.heads * g.dim;
    let mut out = Vec::with_capacity(rows.len() * w);
    for &r in rows {
        out.extend(full[r * w * 4..(r + 1) * w * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
    }
    out
}

// --------------------------------------------------------------- reference

fn bf16r(x: f32) -> f32 {
    if !x.is_finite() { return x }
    let b = x.to_bits();
    f32::from_bits(((b.wrapping_add(0x7fff).wrapping_add((b >> 16) & 1)) >> 16) << 16)
}

struct Reference { rows: Vec<usize>, exact: Vec<f64>, bf16: Vec<f64> }

/// Independent f64 attention (and BF16-storage variant) for the given rows,
/// straight from the selected-list semantics in the module doc.
fn reference(g: &Geo, q: &[f32], kd: &[f32], vd: &[f32], sel: &[i32], rows: &[usize]) -> Reference {
    let (heads, dim, kvh) = (g.heads, g.dim, g.kv_heads);
    let group = heads / kvh;
    let kb: Vec<f32> = kd.iter().map(|&x| bf16r(x)).collect();
    let vb: Vec<f32> = vd.iter().map(|&x| bf16r(x)).collect();
    let w = heads * dim;
    let mut exact = vec![0.0f64; rows.len() * w];
    let mut bf = vec![0.0f64; rows.len() * w];
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(32).min(rows.len().max(1));
    let chunk = rows.len().div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for ((rs, ex), bs) in rows.chunks(chunk).zip(exact.chunks_mut(chunk * w)).zip(bf.chunks_mut(chunk * w)) {
            let (kb, vb) = (&kb, &vb);
            scope.spawn(move || {
                for (i, &row) in rs.iter().enumerate() {
                    let len = g.selected_len(row);
                    let vis = g.visible(row).min(g.full_capacity);
                    let valid: Vec<usize> = sel[row * g.capacity..row * g.capacity + len].iter()
                        .filter(|&&t| t >= 0 && (t as usize) < vis).map(|&t| t as usize).collect();
                    if valid.is_empty() { continue }
                    let inv = 1.0 / (dim as f64).sqrt();
                    for head in 0..heads {
                        let kv = head / group;
                        let qo = (row * heads + head) * 2 * dim;
                        let qh: Vec<f64> = q[qo..qo + dim].iter().map(|&x| bf16r(x) as f64).collect();
                        for variant in 0..2 {
                            let (kk, vv): (&[f32], &[f32]) = if variant == 0 { (kd, vd) } else { (kb.as_slice(), vb.as_slice()) };
                            let scores: Vec<f64> = valid.iter().map(|&t| {
                                let base = (t * kvh + kv) * dim;
                                let mut dot = 0.0f64;
                                for d in 0..dim {
                                    let qv = if variant == 0 { q[qo + d] as f64 } else { qh[d] };
                                    dot += qv * kk[base + d] as f64;
                                }
                                dot * inv
                            }).collect();
                            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let mut num = vec![0.0f64; dim];
                            let mut den = 0.0f64;
                            for (slot, &t) in valid.iter().enumerate() {
                                let p = (scores[slot] - m).exp();
                                den += p;
                                let pw = if variant == 0 { p } else { bf16r(p as f32) as f64 };
                                let base = (t * kvh + kv) * dim;
                                for d in 0..dim { num[d] += pw * vv[base + d] as f64; }
                            }
                            let out = if variant == 0 { &mut ex[i * w + head * dim..i * w + (head + 1) * dim] }
                                else { &mut bs[i * w + head * dim..i * w + (head + 1) * dim] };
                            for d in 0..dim {
                                let gate = q[qo + dim + d];
                                let gate = if variant == 0 { gate as f64 } else { bf16r(gate) as f64 };
                                out[d] = num[d] / den / (1.0 + (-gate).exp());
                            }
                        }
                    }
                }
            });
        }
    });
    Reference { rows: rows.to_vec(), exact, bf16: bf }
}

struct Metrics { n: usize, max_abs: f64, rms: f64, mean_abs: f64, rel_l2: f64, max_rel: f64, ref_max_abs: f64, nonfinite: usize, worst: Value }
impl Metrics {
    fn json(&self) -> Value {
        json!({"elements": self.n, "max_abs": num(self.max_abs), "rms": num(self.rms), "mean_abs": num(self.mean_abs),
            "rel_l2": num(self.rel_l2), "max_rel_where_abs_ref_ge_1e-5": num(self.max_rel),
            "ref_max_abs": num(self.ref_max_abs), "nonfinite_candidate": self.nonfinite, "worst_abs": self.worst})
    }
}
fn metrics(reference: &[f64], cand: &[f64], rows: &[usize], heads: usize, dim: usize) -> Metrics {
    let w = heads * dim;
    let (mut max_abs, mut sum_abs, mut sum_sq, mut ref_sq, mut max_rel, mut ref_max, mut nf) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    let mut worst = (0usize, 0usize, 0usize, 0.0f64, 0.0f64);
    for (i, (&r, &c)) in reference.iter().zip(cand).enumerate() {
        let d = if c.is_finite() { (c - r).abs() } else { nf += 1; f64::INFINITY };
        if d > max_abs || (d.is_infinite() && max_abs.is_finite()) { max_abs = d; worst = (rows[i / w], i % w / dim, i % dim, r, c); }
        sum_abs += d; sum_sq += d * d; ref_sq += r * r; ref_max = ref_max.max(r.abs());
        if r.abs() >= 1e-5 { max_rel = max_rel.max(d / r.abs()); }
    }
    let n = reference.len().max(1) as f64;
    let rel_l2 = if ref_sq > 0.0 { (sum_sq / ref_sq).sqrt() } else if sum_sq == 0.0 { 0.0 } else { f64::INFINITY };
    Metrics {
        n: reference.len(), max_abs, rms: (sum_sq / n).sqrt(), mean_abs: sum_abs / n, rel_l2, max_rel, ref_max_abs: ref_max, nonfinite: nf,
        worst: json!({"row": worst.0, "head": worst.1, "channel": worst.2, "reference_f64": num(worst.3), "candidate": num(worst.4)}),
    }
}

/// Numeric record body for the arms' sampled rows (None = arm not available).
fn numeric_json(g: &Geo, r: &Reference, arms: &[Option<Vec<f32>>; 3]) -> Value {
    let bf = metrics(&r.exact, &r.bf16, &r.rows, g.heads, g.dim);
    let mut out = serde_json::Map::new();
    out.insert("bf16_storage_reference_vs_f64".into(), bf.json());
    let mut gate = serde_json::Map::new();
    for arm in Arm::ALL {
        match &arms[arm.idx()] {
            None => { out.insert(arm.key().into(), Value::Null); }
            Some(s) => {
                let c: Vec<f64> = s.iter().map(|&x| x as f64).collect();
                let m = metrics(&r.exact, &c, &r.rows, g.heads, g.dim);
                let vs_bf = metrics(&r.bf16, &c, &r.rows, g.heads, g.dim);
                let pass = m.nonfinite == 0 && m.max_abs <= bf.max_abs && m.rel_l2 <= bf.rel_l2;
                let mut j = m.json();
                j["vs_bf16_storage_reference"] = vs_bf.json();
                j["no_worse_than_bf16_storage"] = json!(pass);
                out.insert(arm.key().into(), j);
                gate.insert(arm.key().into(), json!(pass));
            }
        }
    }
    out.insert("gate".into(), json!({
        "rule": "finite && max_abs <= bf16_storage max_abs && rel_l2 <= bf16_storage rel_l2, each vs the independent f64 reference over the sampled rows",
        "pass": Value::Object(gate)}));
    Value::Object(out)
}

// ------------------------------------------------------------------ timing

struct Slot { median: f64, min: f64, max: f64 }
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) * 0.5 }
}

fn time_slot(gpu: &mut Gpu, c: &Case<'_>, arm: Arm, warm: usize, calls: usize) -> Result<Slot> {
    for _ in 0..warm { launch(gpu, c, arm)?; }
    gpu.hip.device_synchronize().map_err(err)?;
    let (a, b) = (gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?);
    let mut t = Vec::with_capacity(calls);
    for _ in 0..calls {
        gpu.hip.event_record(&a, None).map_err(err)?;
        launch(gpu, c, arm)?;
        gpu.hip.event_record(&b, None).map_err(err)?;
        gpu.hip.event_synchronize(&b).map_err(err)?;
        t.push(gpu.hip.event_elapsed_ms(&a, &b).map_err(err)? as f64);
    }
    gpu.hip.event_destroy(a).map_err(err)?;
    gpu.hip.event_destroy(b).map_err(err)?;
    let (min, max) = (t.iter().copied().fold(f64::INFINITY, f64::min), t.iter().copied().fold(0.0, f64::max));
    Ok(Slot { median: median(t), min, max })
}

/// Label of the rotated ABBA palindrome over `n` selected arms
/// (`n = 3`: `[a0 a1 a2 a2 a1 a0]`).
fn abba_label(n: usize) -> String {
    let names: Vec<String> = (0..n).map(|i| format!("a{i}")).collect();
    let all: Vec<String> = names.iter().chain(names.iter().rev()).cloned().collect();
    format!("rotated ABBA palindromes [{}]", all.join(" "))
}

/// ABBA over the selected arms: each round runs a rotated palindrome
/// `[a0 a1 a2 a2 a1 a0]` (three arms), so every arm runs twice per round,
/// mirrored. Each timed call is the whole API call (converter + attend for the
/// gathered arms). Unselected arms keep an empty slot list and a 0.0 median.
/// Returns None when the hold deadline passed before the snapshot completed.
fn time_snapshot(gpu: &mut Gpu, c: &Case<'_>, o: &Opts, deadline: Instant) -> Result<Option<([Vec<Slot>; 3], [f64; 3])>> {
    let sel = o.selected();
    let n = sel.len();
    let mut slots: [Vec<Slot>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for round in 0..o.rounds {
        let order: Vec<Arm> = (0..n).map(|i| sel[(i + round) % n]).collect();
        for arm in order.iter().chain(order.iter().rev()).copied() {
            if Instant::now() > deadline { return Ok(None) }
            let s = time_slot(gpu, c, arm, o.warm, o.calls)?;
            slots[arm.idx()].push(s);
        }
    }
    let mut ms = [0.0f64; 3];
    for arm in &sel { ms[arm.idx()] = median(slots[arm.idx()].iter().map(|s| s.median).collect()); }
    Ok(Some((slots, ms)))
}

struct Timed { artifact: String, layer: i64, ctx: i64, route: String, position_start: usize, end: usize, ms: [f64; 3], snapshot: String }

/// Native route identity for aggregation: a path token `r<digits>` (the
/// r768 / r1536 / r8192 chunk-row route directories), else the header's
/// `identity.route`, else "unrouted". Routes of one artifact+ctx never mix.
fn route_key(path: &Path, header: &Value) -> String {
    for comp in path.components().rev() {
        let s = comp.as_os_str().to_string_lossy();
        for tok in s.split(|c: char| !c.is_ascii_alphanumeric()).rev() {
            if tok.len() > 1 && tok.starts_with('r') && tok[1..].bytes().all(|b| b.is_ascii_digit()) { return tok.to_string() }
        }
    }
    let r = &header["identity"]["route"];
    if r.is_null() { "unrouted".into() } else { format!("identity.route={r}") }
}

fn aggregate(timed: &[Timed], o: &Opts, total: usize, truncated: bool) -> Value {
    let mut groups: BTreeMap<(String, i64, String), Vec<&Timed>> = BTreeMap::new();
    for t in timed { groups.entry((t.artifact.clone(), t.ctx, t.route.clone())).or_default().push(t); }
    let mut out = Vec::new();
    for ((artifact, ctx, route), snaps) in groups {
        let mut layers: BTreeMap<i64, Vec<&Timed>> = BTreeMap::new();
        for t in &snaps { layers.entry(t.layer).or_default().push(t); }
        let (mut direct, mut n_complete) = ([0.0f64; 3], 0usize);
        let mut layer_json = Vec::new();
        for (layer, mut ts) in layers {
            ts.sort_by_key(|t| t.position_start);
            let dup = ts.windows(2).filter(|w| w[0].position_start == w[1].position_start).count();
            let mut next = 0usize;
            let tiled = ts.iter().all(|t| { let ok = t.position_start == next; next = t.end; ok });
            let complete = ctx > 0 && tiled && next == ctx as usize && ts.first().is_some_and(|t| t.position_start == 0);
            let sums = [0, 1, 2].map(|i| ts.iter().map(|t| t.ms[i]).sum::<f64>());
            if complete { n_complete += 1; for i in 0..3 { direct[i] += sums[i]; } }
            layer_json.push(json!({"layer": layer, "chunks": ts.len(), "duplicate_position_starts": dup, "tiles_0_to_ctx": complete,
                "sum_ms": Value::Object(arm_json(o, |a| json!(sums[a.idx()]))),
                "snapshots": ts.iter().map(|t| t.snapshot.clone()).collect::<Vec<_>>()}));
        }
        let mut g = json!({"artifact": artifact, "ctx": ctx, "route": route, "layers": layer_json, "complete_layers": n_complete,
            "requested_layers": o.layers});
        if n_complete > 0 {
            let mean = direct.map(|x| x / n_complete as f64);
            let basis = if n_complete >= o.layers { direct } else { mean.map(|x| x * o.layers as f64) };
            let extrapolated = n_complete < o.layers;
            let saving = |i: usize| basis[0] - basis[i];
            g["direct_sum_over_complete_layers_ms"] = Value::Object(arm_json(o, |a| json!(direct[a.idx()])));
            g["mean_per_complete_layer_ms"] = Value::Object(arm_json(o, |a| json!(mean[a.idx()])));
            let mut agg_ms = arm_json(o, |a| json!(basis[a.idx()]));
            agg_ms.insert("layers".into(), json!(o.layers));
            agg_ms.insert("extrapolated_from_mean_complete_layer".into(), json!(extrapolated));
            g["aggregate_ms"] = Value::Object(agg_ms);
            if o.has(Arm::Simt) {
                let mut sv = serde_json::Map::new();
                for a in [Arm::Hip, Arm::Pm] { if o.has(a) { sv.insert(a.key().into(), json!(saving(a.idx()))); } }
                g["saving_vs_simt_ms"] = Value::Object(sv);
                let mut kr = json!({"rule": "<25 ms saved per 8K prefill aggregate over the QSA layers, converter included",
                    "applies_to_ctx_8192": ctx == 8192});
                if o.has(Arm::Pm) { kr["pm_saves_ge_25ms"] = json!(saving(2) >= 25.0); }
                if o.has(Arm::Hip) { kr["hip_saves_ge_25ms"] = json!(saving(1) >= 25.0); }
                if o.has(Arm::Pm) { kr["pm_kill"] = json!(ctx == 8192 && saving(2) < 25.0); }
                g["kill_rule"] = kr;
            } else if o.has(Arm::Pm) {
                g["saving_pm_vs_hip_ms"] = json!(basis[1] - basis[2]);
            }
        }
        out.push(g);
    }
    json!({"kind": "aggregate", "timed_snapshots": timed.len(), "snapshots_total": total, "truncated_by_hold_deadline": truncated,
        "rounds": o.rounds, "calls_per_slot": o.calls, "warmup_per_slot": o.warm, "groups": out,
        "note": "timing is converter-inclusive whole-API-call GPU event time on native snapshots only; derived cases are never timed"})
}

// ------------------------------------------------------------- persistence

fn write_json(f: &mut std::fs::File, v: &Value) -> Result<()> { writeln!(f, "{v}").map_err(err) }

fn save_samples(o: &Opts, l: &Loaded, rows: &[usize], outs: &[Option<ArmOut>; 3]) -> Result<()> {
    std::fs::create_dir_all(&o.work).map_err(err)?;
    let mut arms = serde_json::Map::new();
    for arm in o.selected() {
        let s = take_rows(&outs[arm.idx()].as_ref().unwrap().bytes, rows, &l.geo);
        let b: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(o.work.join(format!("{}.{}.sample.f32", l.id, arm.key())), &b).map_err(err)?;
        arms.insert(arm.key().into(), json!(sha256(&b)));
    }
    std::fs::write(o.work.join(format!("{}.sample.json", l.id)),
        serde_json::to_vec_pretty(&json!({"rows": rows, "arm_sample_sha256": Value::Object(arms)})).map_err(err)?).map_err(err)?;
    if let Some(dir) = &o.outputs {
        std::fs::create_dir_all(dir).map_err(err)?;
        for arm in o.selected() { std::fs::write(dir.join(format!("{}.{}.f32", l.id, arm.key())), &outs[arm.idx()].as_ref().unwrap().bytes).map_err(err)?; }
    }
    Ok(())
}

// ------------------------------------------------------------------ pass 1

fn snapshot_pass(gpu: &mut Gpu, o: &Opts, l: &Loaded, index: usize, timing: bool, deadline: Instant,
                 file: &mut std::fs::File, timed: &mut Vec<Timed>, truncated: &mut bool, failures: &mut Vec<String>) -> Result<()> {
    let kdev = upload_cache(gpu, &l.k, l.fmt)?;
    let vdev = upload_cache(gpu, &l.v, l.fmt)?;
    let case = Case::new(gpu, l.geo, l.fmt, &l.q, &l.sel, &kdev, &vdev)?;
    let r: Result<()> = (|| {
        let mut outs: [Option<ArmOut>; 3] = [None, None, None];
        for arm in o.selected() { outs[arm.idx()] = Some(arm_runs(gpu, &case, arm, o.replays, o.allow_non_hg4)?); }
        let rows = sample_rows(&l.geo, o.sample_rows, o.full_reference);
        save_samples(o, l, &rows, &outs)?;
        let out = |arm: Arm| outs[arm.idx()].as_ref();
        let bytes = |arm: Arm| outs[arm.idx()].as_ref().map(|x| x.bytes.as_slice());
        let cmp = |a: Arm, b: Arm| match (bytes(a), bytes(b)) { (Some(x), Some(y)) => compare(x, y), _ => Value::Null };
        let pm_eq_hip = match (bytes(Arm::Pm), bytes(Arm::Hip)) { (Some(p), Some(h)) => Some(p == h), _ => None };
        let pm_hip = cmp(Arm::Pm, Arm::Hip);
        let mut arms = serde_json::Map::new();
        for arm in o.selected() {
            let a = out(arm).unwrap();
            arms.insert(arm.key().into(), json!({"sha256": a.sha, "replay_sha256": a.replay_sha, "deterministic": a.deterministic,
                "last_launched_kernel": a.kernel, "nonfinite_outputs": a.nonfinite, "bytes": a.bytes.len()}));
        }
        let eager = l.eager.as_ref().map(|e| json!({"sha256": sha256(e), "role": "identity only; never a numeric oracle",
            "equals_simt": bytes(Arm::Simt).map(|b| *e == b), "equals_hip": bytes(Arm::Hip).map(|b| *e == b), "equals_pm": bytes(Arm::Pm).map(|b| *e == b)}));
        let mut rec = json!({
            "kind": "snapshot", "id": l.id, "snapshot": l.path, "identity": l.header["identity"], "route_group": route_key(&l.path, &l.header),
            "geometry": l.geo.json(fmt_name(l.fmt)), "inputs": l.inputs, "arch": gpu.arch,
            "poison_output_patterns": POISON.iter().take(o.replays.max(1).min(3)).collect::<Vec<_>>(),
            "arms": Value::Object(arms), "pm_vs_hip": pm_hip,
            "pm_equals_hip": pm_eq_hip,
            "simt_vs_hip": cmp(Arm::Simt, Arm::Hip), "simt_vs_pm": cmp(Arm::Simt, Arm::Pm),
            "eager": eager, "sample_rows": rows.len(),
        });
        for arm in o.selected() {
            let a = out(arm).unwrap();
            if !a.deterministic { failures.push(format!("{}: {} arm nondeterministic across replays", l.id, arm.key())) }
            if a.nonfinite > 0 { failures.push(format!("{}: {} arm produced {} non-finite outputs", l.id, arm.key(), a.nonfinite)) }
        }
        if pm_eq_hip == Some(false) { failures.push(format!("{}: PM bytes differ from hipcc gathered", l.id)) }
        if timing {
            if Instant::now() > deadline {
                *truncated = true;
                rec["timing"] = json!({"timed": false, "reason": "hold deadline passed before this snapshot"});
            } else {
                match time_snapshot(gpu, &case, o, deadline)? {
                    Some((slots, ms)) => {
                        let sj = |a: Arm| slots[a.idx()].iter().map(|s| json!({"median_ms": s.median, "min_ms": s.min, "max_ms": s.max})).collect::<Vec<_>>();
                        rec["timing"] = json!({"timed": true, "converter_included": true, "rounds": o.rounds, "calls_per_slot": o.calls,
                            "warmup_per_slot": o.warm, "order": abba_label(o.selected().len()),
                            "slots": Value::Object(arm_json(o, |a| json!(sj(a)))),
                            "median_ms": Value::Object(arm_json(o, |a| json!(ms[a.idx()])))});
                        let id = &l.header["identity"];
                        timed.push(Timed {
                            artifact: id["artifact"].as_str().unwrap_or("?").to_string(), layer: id["layer"].as_i64().unwrap_or(-1),
                            ctx: id["ctx"].as_i64().unwrap_or(-1), position_start: l.geo.position_start, end: l.geo.end(), ms,
                            snapshot: l.path.display().to_string(), route: route_key(&l.path, &l.header),
                        });
                    }
                    None => {
                        *truncated = true;
                        rec["timing"] = json!({"timed": false, "reason": "hold deadline passed during this snapshot; partial slots discarded"});
                    }
                }
            }
        }
        write_json(file, &rec)?;
        eprintln!("{} {} pm==hip:{} simt kernel:{}", index + 1, l.id, pm_eq_hip.map_or("n/a".to_string(), |b| b.to_string()), out(Arm::Simt).map_or("n/a", |x| x.kernel.as_str()));
        Ok(())
    })();
    case.free(gpu)?;
    gpu.free_tensor(kdev).map_err(err)?;
    gpu.free_tensor(vdev).map_err(err)?;
    r
}

// --------------------------------------------------------------- derived

struct Mutation {
    label: &'static str, description: String, geo: Geo, q: Vec<f32>, sel: Vec<i32>,
    poison_cache: bool, identical_to_base: bool, extra_rows: Vec<usize>, empty_rows: Vec<usize>,
}

fn build_mutations(o: &Opts, l: &Loaded) -> (Vec<Mutation>, Geo, usize, usize) {
    let g = l.geo;
    let e = o.edge_rows.min(g.rows);
    let r0 = g.rows - e;
    let view = Geo { rows: e, position_start: g.position_start + r0, ..g };
    let w = g.heads * g.dim * 2;
    let qv: Vec<f32> = l.q[r0 * w..(r0 + e) * w].to_vec();
    let sv: Vec<i32> = l.sel[r0 * g.capacity..(r0 + e) * g.capacity].to_vec();
    let mut muts: Vec<Mutation> = Vec::new();
    let base = |label, description: String| Mutation {
        label, description, geo: view, q: qv.clone(), sel: sv.clone(), poison_cache: false, identical_to_base: false,
        extra_rows: Vec::new(), empty_rows: Vec::new(),
    };
    muts.push(base("view-base", format!("last {e} rows of the snapshot as a row view (position_start {} nonzero, original selected lists, full cache)", view.position_start)));
    // Ragged causal tails: shift the view's position by 1..3 and rebuild each
    // row's list from the real one: the real first blocks, then the newest
    // block (if the chosen count grew) and the 1..3 uncompressed tail tokens.
    for t in 1..=3usize {
        if view.end() + t > g.full_capacity { continue }
        let tv = Geo { position_start: view.position_start + t, ..view };
        let mut sel = sv.clone();
        for r in 0..e {
            let (vo, vn) = (view.visible(r), tv.visible(r));
            let chosen_o = g.budget_blocks.min(vo / g.compress);
            let (blocks_n, chosen_n) = (vn / g.compress, g.budget_blocks.min(vn / g.compress));
            let len = tv.selected_len(r);
            let row = &mut sel[r * g.capacity..(r + 1) * g.capacity];
            for (slot, entry) in row.iter_mut().enumerate().take(len).skip(chosen_o * g.compress) {
                *entry = if slot < chosen_n * g.compress {
                    ((blocks_n - 1) * g.compress + (slot - chosen_o * g.compress) % g.compress) as i32
                } else {
                    (blocks_n * g.compress + (slot - chosen_n * g.compress)) as i32
                };
            }
        }
        let label = ["tail-plus-1", "tail-plus-2", "tail-plus-3"][t - 1];
        muts.push(Mutation { label, description: format!("position_start shifted by +{t} so visible % compress != 0; tail slots rebuilt as the newest {t}.. uncompressed tokens"),
            geo: tv, q: qv.clone(), sel, poison_cache: false, identical_to_base: false, extra_rows: (e.saturating_sub(4)..e).collect(), empty_rows: Vec::new() });
    }
    if view.end() < g.full_capacity {
        let mut m = base("poison-cache-padding", "K/V rows at and beyond the visible end overwritten with 0xff (NaN scales/values); result must equal view-base bytes".into());
        m.poison_cache = true; m.identical_to_base = true;
        muts.push(m);
    }
    {
        let mut m = base("poison-selected-padding", "selected slots at and beyond each row's list length overwritten with alternating 0 / i32::MAX; result must equal view-base bytes".into());
        for r in 0..e {
            let len = view.selected_len(r);
            for slot in len..g.capacity { m.sel[r * g.capacity + slot] = if slot % 2 == 0 { 0 } else { i32::MAX }; }
        }
        m.identical_to_base = true;
        muts.push(m);
    }
    {
        let mut m = base("duplicate-selection", "slots 1, 2 and len-1 of every row duplicate slot 0 (lists with len>=4)".into());
        for r in 0..e {
            let len = view.selected_len(r);
            if len >= 4 { let s = &mut m.sel[r * g.capacity..]; let t = s[0]; s[1] = t; s[2] = t; s[len - 1] = t; m.extra_rows.push(r); }
        }
        m.extra_rows.truncate(16);
        muts.push(m);
    }
    {
        let mut m = base("invalid-future-tokens", "slots 1..4 of every row become: the row's own future token (visible), -1, full_capacity, i32::MAX (all must be masked)".into());
        for r in 0..e {
            let len = view.selected_len(r);
            if len >= 5 {
                let s = &mut m.sel[r * g.capacity..];
                s[1] = view.visible(r) as i32; s[2] = -1; s[3] = g.full_capacity as i32; s[4] = i32::MAX;
                m.extra_rows.push(r);
            }
        }
        m.extra_rows.truncate(16);
        muts.push(m);
    }
    {
        let mut m = base("empty-selection", "rows r % 7 == 3 have every list slot invalid (-1): output must be exactly 0".into());
        for r in (0..e).filter(|r| r % 7 == 3) {
            let len = view.selected_len(r);
            for slot in 0..len { m.sel[r * g.capacity + slot] = -1; }
            m.empty_rows.push(r);
        }
        m.extra_rows = m.empty_rows.iter().copied().take(16).collect();
        muts.push(m);
    }
    (muts, view, r0, e)
}

fn derived_pass(gpu: &mut Gpu, o: &Opts, l: &Loaded, file: &mut std::fs::File, failures: &mut Vec<String>) -> Result<()> {
    let g = l.geo;
    let (muts, view, r0, e) = build_mutations(o, l);
    let kdev = upload_cache(gpu, &l.k, l.fmt)?;
    let vdev = upload_cache(gpu, &l.v, l.fmt)?;
    let (mut kp, mut vp) = (None, None);
    if view.end() < g.full_capacity {
        let rb = l.fmt.kv_row_bytes(g.kv_heads, g.dim);
        let (mut kb, mut vb) = (l.k.clone(), l.v.clone());
        kb[view.end() * rb..].fill(0xff);
        vb[view.end() * rb..].fill(0xff);
        kp = Some(upload_cache(gpu, &kb, l.fmt)?);
        vp = Some(upload_cache(gpu, &vb, l.fmt)?);
    }
    let mut base_bytes: [Option<Vec<u8>>; 3] = [None, None, None];
    let mut result: Result<()> = Ok(());
    for m in &muts {
        let (k, v) = if m.poison_cache { (kp.as_ref().unwrap(), vp.as_ref().unwrap()) } else { (&kdev, &vdev) };
        let case = match Case::new(gpu, m.geo, l.fmt, &m.q, &m.sel, k, v) { Ok(c) => c, Err(e) => { result = Err(e); break } };
        let r: Result<()> = (|| {
            // Native passes demand hg4. A derived view small enough for the
            // incumbent's decode route (`few_row_limit`, as `qsa_decode_rows_admitted`)
            // legitimately launches that route's own kernel (split-KV / per-head);
            // the actual symbol is recorded per arm. Larger views still demand hg4.
            let limit = gpu.few_row_limit();
            let small_route = (1..=limit).contains(&m.geo.rows) && gpu.route_rows(m.geo.rows) <= limit;
            let admit_simt = o.allow_non_hg4 || small_route;
            let mut outs: [Option<ArmOut>; 3] = [None, None, None];
            for arm in o.selected() { outs[arm.idx()] = Some(arm_runs(gpu, &case, arm, o.replays.max(2), admit_simt)?); }
            let pm_eq_hip = match (&outs[Arm::Pm.idx()], &outs[Arm::Hip.idx()]) { (Some(p), Some(h)) => Some(p.bytes == h.bytes), _ => None };
            let simt_guard = match &outs[Arm::Simt.idx()] {
                Some(s) => json!({"native_requires_hg4": true, "small_row_decode_route_admitted": small_route, "few_row_limit": limit, "rows": m.geo.rows,
                    "simt_launched_kernel": s.kernel}),
                None => Value::Null,
            };
            let mut rec = json!({"kind": "derived", "derived_from": l.path, "id": l.id, "label": m.label, "description": m.description,
                "timed": false, "performance_claim": false, "geometry": m.geo.json(fmt_name(l.fmt)), "arch": gpu.arch,
                "simt_route_guard": simt_guard,
                "complete_cache_uploaded": true, "cache_rows": g.full_capacity, "row_view": {"first_snapshot_row": r0, "rows": e}});
            let mut arms = serde_json::Map::new();
            for arm in o.selected() {
                let x = outs[arm.idx()].as_ref().unwrap();
                arms.insert(arm.key().into(), json!({"sha256": x.sha, "replay_sha256": x.replay_sha256_json(), "deterministic": x.deterministic,
                    "last_launched_kernel": x.kernel, "nonfinite_outputs": x.nonfinite}));
                if !x.deterministic { failures.push(format!("{}:{}: {} nondeterministic", l.id, m.label, arm.key())) }
                if x.nonfinite > 0 { failures.push(format!("{}:{}: {} non-finite output", l.id, m.label, arm.key())) }
            }
            rec["arms"] = Value::Object(arms);
            rec["pm_equals_hip"] = json!(pm_eq_hip);
            if pm_eq_hip == Some(false) { failures.push(format!("{}:{}: PM differs from hipcc gathered", l.id, m.label)) }
            if m.label == "view-base" {
                base_bytes = std::array::from_fn(|i| outs[i].as_ref().map(|x| x.bytes.clone()));
            }
            if m.identical_to_base {
                let same: Vec<(Arm, bool)> = o.selected().into_iter()
                    .map(|a| (a, base_bytes[a.idx()].as_ref().is_some_and(|b| *b == outs[a.idx()].as_ref().unwrap().bytes))).collect();
                rec["identical_to_view_base"] = Value::Object(same.iter().map(|(a, s)| (a.key().to_string(), json!(*s))).collect());
                if !same.iter().all(|(_, x)| *x) { failures.push(format!("{}:{}: output changed when only unread padding was poisoned", l.id, m.label)) }
            }
            if !m.empty_rows.is_empty() {
                let w = m.geo.heads * m.geo.dim * 4;
                let zero: Vec<(Arm, bool)> = o.selected().into_iter().map(|a| {
                    let x = outs[a.idx()].as_ref().unwrap();
                    (a, m.empty_rows.iter().all(|&r| x.bytes[r * w..(r + 1) * w].iter().all(|&b| b == 0)))
                }).collect();
                let mut zj: serde_json::Map<String, Value> = zero.iter().map(|(a, z)| (a.key().to_string(), json!(*z))).collect();
                zj.insert("rows".into(), json!(m.empty_rows.len()));
                rec["empty_rows_all_zero_bits"] = Value::Object(zj);
                if !zero.iter().all(|(_, x)| *x) { failures.push(format!("{}:{}: empty selection did not yield exact zeros", l.id, m.label)) }
            }
            // CPU numerics are deferred: persist the arms' sampled rows for the
            // offline pass (`derived_numeric_pass`), which rebuilds this mutation.
            let mut rows: BTreeSet<usize> = sample_rows(&m.geo, o.edge_sample_rows, false).into_iter().collect();
            rows.extend(m.extra_rows.iter().copied());
            let rows: Vec<usize> = rows.into_iter().collect();
            let mut sample_sha = serde_json::Map::new();
            for arm in o.selected() {
                let s = take_rows(&outs[arm.idx()].as_ref().unwrap().bytes, &rows, &m.geo);
                let b: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(o.work.join(format!("{}.derived.{}.{}.sample.f32", l.id, m.label, arm.key())), &b).map_err(err)?;
                sample_sha.insert(arm.key().into(), json!(sha256(&b)));
            }
            std::fs::write(o.work.join(format!("{}.derived.{}.sample.json", l.id, m.label)),
                serde_json::to_vec_pretty(&json!({"rows": rows, "label": m.label, "view_rows": m.geo.rows, "view_position_start": m.geo.position_start,
                    "arm_sample_sha256": Value::Object(sample_sha)})).map_err(err)?).map_err(err)?;
            rec["sampled_rows"] = json!(rows);
            rec["numeric"] = Value::Null;
            rec["numeric_deferred"] = json!("a kind:\"derived_numeric\" record follows from the CPU pass over the saved samples");
            write_json(file, &rec)?;
            eprintln!("derived {} {} pm==hip:{}", l.id, m.label, pm_eq_hip.map_or("n/a".to_string(), |b| b.to_string()));
            Ok(())
        })();
        let freed = case.free(gpu);
        if let Err(e) = r { result = Err(e); break }
        if let Err(e) = freed { result = Err(e); break }
    }
    for t in [Some(kdev), Some(vdev), kp, vp].into_iter().flatten() { gpu.free_tensor(t).map_err(err)?; }
    result
}
impl ArmOut { fn replay_sha256_json(&self) -> Value { json!(self.replay_sha) } }

// ---------------------------------------------------------------- numerics

fn numeric_pass(o: &Opts, l: &Loaded, file: &mut std::fs::File) -> Result<Value> {
    let meta: Option<Value> = std::fs::read(o.work.join(format!("{}.sample.json", l.id))).ok().and_then(|b| serde_json::from_slice(&b).ok());
    let rows: Vec<usize> = match &meta {
        Some(m) => m["rows"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect()).unwrap_or_default(),
        None => sample_rows(&l.geo, o.sample_rows, o.full_reference),
    };
    if rows.is_empty() || rows.iter().any(|&r| r >= l.geo.rows) { return Err(format!("{}: invalid sample rows", l.id)) }
    let started = Instant::now();
    let r = reference(&l.geo, &l.q, &l.kd, &l.vd, &l.sel, &rows);
    let ref_secs = started.elapsed().as_secs_f64();
    let mut arms: [Option<Vec<f32>>; 3] = [None, None, None];
    let mut sources = serde_json::Map::new();
    for arm in o.selected() {
        let full = o.outputs.as_ref().map(|d| d.join(format!("{}.{}.f32", l.id, arm.key()))).filter(|p| p.exists());
        let sample = o.work.join(format!("{}.{}.sample.f32", l.id, arm.key()));
        let w = l.geo.heads * l.geo.dim;
        if let Some(p) = full {
            let b = std::fs::read(&p).map_err(err)?;
            if b.len() != l.geo.rows * w * 4 { return Err(format!("{}: saved output size mismatch", p.display())) }
            arms[arm.idx()] = Some(take_rows(&b, &rows, &l.geo));
            sources.insert(arm.key().into(), json!({"from": "full saved output", "file": p, "full_output_sha256": sha256(&b)}));
        } else if sample.exists() {
            let b = std::fs::read(&sample).map_err(err)?;
            if b.len() != rows.len() * w * 4 { return Err(format!("{}: sample size mismatch", sample.display())) }
            arms[arm.idx()] = Some(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
            sources.insert(arm.key().into(), json!({"from": "sampled rows saved by the GPU pass", "file": sample, "sample_sha256": sha256(&b)}));
        }
    }
    let ref_bytes: Vec<u8> = r.exact.iter().flat_map(|x| x.to_le_bytes()).collect();
    let rec = json!({
        "kind": "numeric", "id": l.id, "snapshot": l.path, "geometry": l.geo.json(fmt_name(l.fmt)),
        "sampling": {"policy": if o.full_reference { "every row" } else { "deterministic: row 0, last 4 rows, rows around the selected-list regime changes (budget reached, capacity clamp), and an even stride; all heads, all channels" },
            "full": o.full_reference, "rows_sampled": rows.len(), "rows_total": l.geo.rows, "rows": rows, "heads": "all", "channels": "all"},
        "reference": {"f64_exact_sha256": sha256(&ref_bytes), "seconds": ref_secs,
            "inputs": "f64 attention from the host-decoded cache (Q8/FP8/F32 exact decode), independent of every kernel and of eager.f32",
            "bf16_storage": "Q, gate, K, V and the unnormalized probability rounded to BF16 (RNE), f64 accumulation, denominator from unrounded probabilities"},
        "arm_sources": Value::Object(sources),
        "numeric": numeric_json(&l.geo, &r, &arms),
    });
    write_json(file, &rec)?;
    Ok(rec)
}

/// Offline numerics of the derived cases the GPU pass saved samples for:
/// rebuilds each mutation from the snapshot and compares the saved arm rows
/// with the f64 and BF16-storage references.
fn derived_numeric_pass(o: &Opts, l: &Loaded, file: &mut std::fs::File) -> Result<usize> {
    if !o.work.join(format!("{}.derived.view-base.sample.json", l.id)).exists() { return Ok(0) }
    let (muts, _, _, _) = build_mutations(o, l);
    let mut n = 0usize;
    for m in &muts {
        let meta_path = o.work.join(format!("{}.derived.{}.sample.json", l.id, m.label));
        let Ok(bytes) = std::fs::read(&meta_path) else { continue };
        let meta: Value = serde_json::from_slice(&bytes).map_err(err)?;
        let rows: Vec<usize> = meta["rows"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect()).unwrap_or_default();
        if rows.is_empty() || rows.iter().any(|&r| r >= m.geo.rows) { return Err(format!("{}: invalid derived sample rows", meta_path.display())) }
        if meta["view_rows"].as_u64() != Some(m.geo.rows as u64) || meta["view_position_start"].as_u64() != Some(m.geo.position_start as u64) {
            return Err(format!("{}: derived view differs from the GPU pass (rerun with the same --edge-rows)", meta_path.display()))
        }
        let w = m.geo.heads * m.geo.dim;
        let mut arms: [Option<Vec<f32>>; 3] = [None, None, None];
        for arm in o.selected() {
            let p = o.work.join(format!("{}.derived.{}.{}.sample.f32", l.id, m.label, arm.key()));
            if let Ok(b) = std::fs::read(&p) {
                if b.len() != rows.len() * w * 4 { return Err(format!("{}: derived sample size mismatch", p.display())) }
                arms[arm.idx()] = Some(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
            }
        }
        // Tail mutations see cache rows past the snapshot's original end (the
        // full cache was uploaded to the GPU): decode to the mutation's own end,
        // with the same strict finite checks on those rows.
        let need = m.geo.end();
        let ext: Option<(Vec<f32>, Vec<f32>)> = if need > l.geo.end() {
            let ctx = |what: &str, e: String| format!("{}: derived {} {what}: {e}", l.path.display(), m.label);
            Some((decode_cache(&l.k, &l.geo, l.fmt, need).map_err(|e| ctx("keys", e))?,
                  decode_cache(&l.v, &l.geo, l.fmt, need).map_err(|e| ctx("values", e))?))
        } else { None };
        let (kd, vd): (&[f32], &[f32]) = match &ext { Some((k, v)) => (k, v), None => (&l.kd, &l.vd) };
        let r = reference(&m.geo, &m.q, kd, vd, &m.sel, &rows);
        write_json(file, &json!({"kind": "derived_numeric", "id": l.id, "label": m.label, "derived_from": l.path, "timed": false,
            "performance_claim": false, "geometry": m.geo.json(fmt_name(l.fmt)), "sampled_rows": rows,
            "numeric": numeric_json(&m.geo, &r, &arms)}))?;
        n += 1;
    }
    Ok(n)
}

// -------------------------------------------------------------------- main

fn main() -> Result<()> {
    let o = parse_args()?;
    let mut paths = Vec::new();
    for d in &o.dirs { snapshots(d, &mut paths)?; }
    if paths.is_empty() { return Err("no snapshot.json found".into()) }
    if o.gpu_only && o.reference_only { return Err("--gpu-only and --reference-only are mutually exclusive".into()) }
    if !o.reference_only {
        if std::env::var_os("HIPFIRE_LOCK_DIR").is_none() { return Err("private HIPFIRE_LOCK_DIR required".into()) }
        if o.has(Arm::Simt) && std::env::var("HIPFIRE_QWEN4_QSA_WMMA_GATHER").as_deref() != Ok("0") {
            return Err("HIPFIRE_QWEN4_QSA_WMMA_GATHER=0 required so the SIMT arm is the incumbent hg4 route".into());
        }
    }
    std::fs::create_dir_all(&o.work).map_err(err)?;
    let mut file = std::fs::File::create(&o.out).map_err(err)?;
    let exe_sha = std::env::current_exe().ok().and_then(|p| std::fs::read(p).ok()).map(|b| sha256(&b));
    let mut failures: Vec<String> = Vec::new();
    let mut timed: Vec<Timed> = Vec::new();
    let mut truncated = false;
    let mut arch = String::from("none (reference-only)");
    let mut gpu_seconds = None;
    let mut gpu_opt: Option<Gpu> = None;
    if !o.reference_only {
        let mut gpu = Gpu::init().map_err(err)?;
        arch = gpu.arch.to_string();
        gpu.dpm_warmup(10.0).map_err(err)?;
        gpu_opt = Some(gpu);
    }
    let selected_arms: Vec<&str> = o.selected().iter().map(|a| a.key()).collect();
    write_json(&mut file, &json!({
        "kind": "run", "arch": arch, "args": std::env::args().collect::<Vec<_>>(), "snapshots": paths.len(),
        "binary_sha256": exe_sha, "unix_time": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).ok(),
        "env": {"HIPFIRE_QWEN4_QSA_WMMA_GATHER": std::env::var("HIPFIRE_QWEN4_QSA_WMMA_GATHER").ok(),
                "HIPFIRE_QWEN4_QSA_PM": std::env::var("HIPFIRE_QWEN4_QSA_PM").ok()},
        "arms": {"simt": "indexed_attention_attention_batch with HIPFIRE_QWEN4_QSA_WMMA_GATHER=0; launched kernel recorded per snapshot (must be ..._batched_hg4)",
                 "hip": "indexed_attention_gathered_batch(pm=false): hipcc converter + hipcc gathered WMMA attend",
                 "pm": "indexed_attention_gathered_batch(pm=true): hipcc converter + builder-module attend"},
        "selected_arms": selected_arms,
        "hold": {"seconds": o.hold_seconds, "covers": "GPU phase: init warmup, arm runs/hashes, timing, derived-case GPU invariants; CPU numerics run after it unless --gpu-only defers them to --reference-only"},
    }))?;
    if let Some(mut gpu) = gpu_opt {
        let hold_start = Instant::now();
        let deadline = hold_start + std::time::Duration::from_secs_f64(o.hold_seconds);
        for (i, path) in paths.iter().enumerate() {
            let timing = match o.time { TimeSel::None => false, TimeSel::All => true, TimeSel::First(n) => i < n };
            let l = load(path)?;
            snapshot_pass(&mut gpu, &o, &l, i, timing, deadline, &mut file, &mut timed, &mut truncated, &mut failures)?;
        }
        if o.derived > 0 {
            let n = paths.len();
            let picks: BTreeSet<usize> = (0..o.derived.min(n)).map(|i| if o.derived == 1 { 0 } else { i * (n - 1) / (o.derived.min(n) - 1).max(1) }).collect();
            for i in picks {
                let l = load(&paths[i])?;
                derived_pass(&mut gpu, &o, &l, &mut file, &mut failures)?;
            }
        }
        let used = hold_start.elapsed().as_secs_f64();
        gpu_seconds = Some(used);
        if used > o.hold_seconds { failures.push(format!("GPU phase used {used:.1}s, over the {}s hold", o.hold_seconds)); }
    }
    // CPU numerics (native and derived), skipped by --gpu-only.
    let mut gate_pass = [0usize; 3];
    let mut gate_total = 0usize;
    let mut derived_numeric = 0usize;
    if !o.gpu_only {
        for path in &paths {
            let l = load(path)?;
            let rec = numeric_pass(&o, &l, &mut file)?;
            gate_total += 1;
            for arm in Arm::ALL {
                if rec["numeric"]["gate"]["pass"][arm.key()] == true { gate_pass[arm.idx()] += 1; }
            }
            eprintln!("numeric {} simt:{} hip:{} pm:{}", l.id, rec["numeric"]["gate"]["pass"]["simt"], rec["numeric"]["gate"]["pass"]["hip"], rec["numeric"]["gate"]["pass"]["pm"]);
            derived_numeric += derived_numeric_pass(&o, &l, &mut file)?;
        }
    }
    let mut agg = aggregate(&timed, &o, paths.len(), truncated);
    agg["gpu_only"] = json!(o.gpu_only);
    agg["hold_seconds_used_by_gpu_phase"] = json!(gpu_seconds);
    agg["numeric_gate_pass_counts"] = if o.gpu_only {
        json!({"measured": false, "reason": "--gpu-only: native and derived numerics are deferred to a --reference-only pass over the saved --work samples"})
    } else {
        let mut m = serde_json::Map::new();
        m.insert("measured".into(), json!(true));
        m.insert("snapshots".into(), json!(gate_total));
        for arm in o.selected() { m.insert(arm.key().into(), json!(gate_pass[arm.idx()])); }
        m.insert("derived_cases".into(), json!(derived_numeric));
        Value::Object(m)
    };
    agg["failures"] = json!(failures);
    write_json(&mut file, &agg)?;
    println!("{}", json!({"arch": arch, "snapshots": paths.len(), "timed": timed.len(), "truncated": truncated,
        "failures": failures.len(), "out": o.out}));
    if o.time == TimeSel::All && truncated && !o.allow_truncated {
        return Err(format!("hold deadline {}s reached with {} of {} snapshots timed", o.hold_seconds, timed.len(), paths.len()));
    }
    if !failures.is_empty() { return Err(format!("{} invariant failures:\n{}", failures.len(), failures.join("\n"))) }
    Ok(())
}
