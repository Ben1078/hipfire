// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `write_i4_expert_slabs`: the DESIGN F2 owner that turns a pinned Flash-Next
//! MQ4-XTS HFQ plus per-expert MQ3-I4 / MQ2-I4 final-code records into the
//! mixed (or uniform) expert-slab artifact.
//!
//! Output layout (frozen by F2, schema [`ExpertSlabsV1`]):
//!
//! 1. every trunk expert as two rank-2 entries, layer `0..L`, expert `0..E`,
//!    gate_up then down, each with the qtype pair of its [`ExpertFormat`]
//!    (131/132, 133/134, or the copied published 44/53 slices);
//! 2. then every other source tensor in its original index order with identical
//!    name/qtype/shape/group size and byte-identical payload — the 2×`L`
//!    aggregate trunk expert tensors are the only source tensors dropped. The
//!    legacy MTP aggregate expert tensors are kept.
//!    With an expert keep map, absent trunk experts have no entries or bytes;
//!    an explicit MTP keep list compacts MTP aggregates in original-id order.
//!
//! Metadata is the source metadata object plus `qwen4.expert_slabs` and
//! `qwen4.expert_slabs_provenance`. The container is produced with
//! `hipfire_runtime::hfq::write_hfqm_package_streaming`; source payloads are
//! read with bounded positional reads (never the whole file).
//!
//! Verification is part of the writer and cannot be disabled: every record is
//! encoded and decoded back (see [`crate::i4_records`]), and after writing the
//! output is reopened with `HfqFile`, its index/spans are checked, and every
//! payload is re-read from disk and hashed against the digest of the bytes
//! the writer was given (source bytes for copied tensors and MQ4 slices,
//! encoded bytes for W3/W2 slabs). The whole-file SHA-256 comes from the same
//! pass.
//!
//! This lives in the library target (not the private `qwen4` module of the
//! `hipfire-quantize` binary) so the `qwen4_i4_slabs` bin and tests can reach it.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Instant;

use hipfire_arch_qwen4::ExpertKeep;
use hipfire_arch_qwen4::expert_slabs::{
    expert_entry_name, parse_expert_entry_name, ExpertFormat, ExpertProj, ExpertSlabsV1,
    EXPERTS_PER_LAYER, EXPERT_SLABS_METADATA_KEY, GATE_UP_SHAPE, DOWN_SHAPE, TRUNK_LAYERS,
};
use hipfire_runtime::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry};
use hipfire_runtime::mq_i4::{self, MqI4Profile};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::i4_records::{hex_lower, I4ExpertFinalCodeRecord, I4RecordShapes};

/// Metadata key for the packer provenance object.
pub const PROVENANCE_METADATA_KEY: &str = "qwen4.expert_slabs_provenance";
/// Packer identity recorded in the provenance.
pub const PACKER_NAME: &str = "hipfire-quantize/qwen4_i4_slabs";
/// Largest positional-read / hash buffer (bytes).
const IO_CHUNK: usize = 16 << 20;
/// `payload_alignment` of [`ExpertSlabsV1`].
const SLAB_ALIGNMENT: u64 = 256;
/// Source aggregate qtypes and group sizes of the MQ4-XTS trunk experts.
const MQ4_GATE_UP_GROUP: u32 = 256;
const MQ4_DOWN_GROUP: u32 = 128;

/// Failure building or verifying an expert-slab artifact.
#[derive(Debug)]
pub enum SlabError {
    /// Invalid arguments/configuration.
    Config(String),
    /// File system or container I/O.
    Io(String),
    /// The source HFQ does not have the pinned layout, or already carries slabs.
    Source(String),
    /// A calibration record failed import, identity or cross-check.
    Record {
        layer: usize,
        expert: usize,
        message: String,
    },
    /// The written artifact failed post-write verification.
    Verify(String),
}

impl fmt::Display for SlabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(m) => write!(f, "config: {m}"),
            Self::Io(m) => write!(f, "i/o: {m}"),
            Self::Source(m) => write!(f, "source: {m}"),
            Self::Record {
                layer,
                expert,
                message,
            } => write!(f, "record L{layer}/E{expert}: {message}"),
            Self::Verify(m) => write!(f, "verification failed: {m}"),
        }
    }
}

impl std::error::Error for SlabError {}

fn io_err(context: impl fmt::Display, e: impl fmt::Display) -> SlabError {
    SlabError::Io(format!("{context}: {e}"))
}

/// Lower-case dash name of a format (matches the allocation `profiles` list).
pub fn format_name(format: ExpertFormat) -> &'static str {
    match format {
        ExpertFormat::Mq3I4 => "mq3-i4",
        ExpertFormat::Mq2I4 => "mq2-i4",
        ExpertFormat::Mq4Xts => "mq4-xts",
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Trunk geometry. The FN entry point pins [`SlabGeometry::FN`]; tests use tiny
/// geometries (the written `qwen4.expert_slabs` metadata still declares the
/// frozen FN v1 value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabGeometry {
    pub layers: usize,
    pub experts: usize,
    /// `[rows, K]` of every expert's gate_up entry.
    pub gate_up: [u32; 2],
    /// `[rows, K]` of every expert's down entry.
    pub down: [u32; 2],
}

impl SlabGeometry {
    /// Flash-Next: 48 layers × 512 experts, `[1280,2560]` / `[2560,640]`.
    pub const FN: Self = Self {
        layers: TRUNK_LAYERS as usize,
        experts: EXPERTS_PER_LAYER as usize,
        gate_up: GATE_UP_SHAPE,
        down: DOWN_SHAPE,
    };

    pub fn proj_shape(&self, proj: ExpertProj) -> [u32; 2] {
        match proj {
            ExpertProj::GateUp => self.gate_up,
            ExpertProj::Down => self.down,
        }
    }

    pub fn record_shapes(&self) -> I4RecordShapes {
        I4RecordShapes {
            gate_up: self.gate_up.map(|d| d as usize),
            down: self.down.map(|d| d as usize),
        }
    }

    /// Source aggregate tensor name for a trunk layer (no expert index).
    pub fn aggregate_name(layer: usize, proj: ExpertProj) -> String {
        format!(
            "model.language_model.layers.{layer}.mlp.experts.{}",
            proj.suffix()
        )
    }

    fn mq4_group(proj: ExpertProj) -> u32 {
        match proj {
            ExpertProj::GateUp => MQ4_GATE_UP_GROUP,
            ExpertProj::Down => MQ4_DOWN_GROUP,
        }
    }

    /// Source qtype of the aggregate (`44` gate_up / `53` down).
    fn mq4_qt(proj: ExpertProj) -> u8 {
        let (gu, down) = ExpertFormat::Mq4Xts.pair();
        match proj {
            ExpertProj::GateUp => gu,
            ExpertProj::Down => down,
        }
    }

    /// Payload bytes of one expert projection in `format`.
    pub fn entry_bytes(&self, format: ExpertFormat, proj: ExpertProj) -> Result<usize, SlabError> {
        let [m, k] = self.proj_shape(proj).map(|d| d as usize);
        let (gu_qt, down_qt) = format.pair();
        let qt = match proj {
            ExpertProj::GateUp => gu_qt,
            ExpertProj::Down => down_qt,
        };
        match format {
            ExpertFormat::Mq3I4 | ExpertFormat::Mq2I4 => {
                let profile = MqI4Profile::from_quant_type(qt).expect("I4 qtype has a profile");
                mq_i4::tensor_bytes(profile, m, k)
                    .map_err(|e| SlabError::Config(format!("{} {proj:?}: {e}", format_name(format))))
            }
            ExpertFormat::Mq4Xts => {
                let group = Self::mq4_group(proj) as usize;
                let block = match proj {
                    ExpertProj::GateUp => 136,
                    ExpertProj::Down => 68,
                };
                if k == 0 || m == 0 || k % group != 0 {
                    return Err(SlabError::Config(format!(
                        "mq4-xts {proj:?} shape [{m},{k}] is not a multiple of group {group}"
                    )));
                }
                Ok(m * (k / group) * block)
            }
        }
    }

    fn validate(&self) -> Result<(), SlabError> {
        if self.layers == 0 || self.experts == 0 {
            return Err(SlabError::Config("geometry has zero layers or experts".into()));
        }
        for format in [ExpertFormat::Mq3I4, ExpertFormat::Mq2I4, ExpertFormat::Mq4Xts] {
            for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                let bytes = self.entry_bytes(format, proj)? as u64;
                if bytes % SLAB_ALIGNMENT != 0 {
                    return Err(SlabError::Config(format!(
                        "{} {proj:?} entry is {bytes} B, not a multiple of the \
                         {SLAB_ALIGNMENT}-byte slab alignment",
                        format_name(format)
                    )));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/// A parsed R1 allocation point: the per-(layer,expert) format map.
#[derive(Clone, Debug)]
pub struct Allocation {
    /// Row-major `layer * experts + expert`.
    pub formats: Vec<ExpertFormat>,
    pub file_sha256: String,
    pub point: String,
}

/// Parse `allocations[point].profile_indices[layer][expert]` (0 = MQ2-I4,
/// 1 = MQ3-I4, 2 = MQ4-XTS) from an R1 `alloc.ep2.json`.
pub fn load_allocation(
    path: &Path,
    point: &str,
    geometry: SlabGeometry,
) -> Result<Allocation, SlabError> {
    load_allocation_with_keep(path, point, geometry, None)
}

/// Accept dense allocation rows or compact rows in sorted kept-id order.
pub fn load_allocation_with_keep(
    path: &Path,
    point: &str,
    geometry: SlabGeometry,
    keep: Option<&ExpertKeep>,
) -> Result<Allocation, SlabError> {
    if keep.is_some_and(|k| k.n_layers() != geometry.layers || k.n_experts() != geometry.experts) {
        return Err(SlabError::Config("allocation keep geometry mismatch".into()));
    }
    let bytes = std::fs::read(path).map_err(|e| io_err(path.display(), e))?;
    let file_sha256 = hex_lower(&Sha256::digest(&bytes));
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|e| SlabError::Config(format!("{}: invalid JSON: {e}", path.display())))?;
    let profiles = root
        .get("profiles")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.as_str().unwrap_or("")).collect::<Vec<_>>());
    if profiles.as_deref() != Some(&["mq2-i4", "mq3-i4", "mq4-xts"][..]) {
        return Err(SlabError::Config(format!(
            "allocation `profiles` must be [\"mq2-i4\",\"mq3-i4\",\"mq4-xts\"], found {:?}",
            root.get("profiles")
        )));
    }
    let alloc = root
        .get("allocations")
        .and_then(|a| a.get(point))
        .ok_or_else(|| SlabError::Config(format!("allocation point `{point}` not found")))?;
    if alloc.get("feasible").and_then(Value::as_bool) != Some(true) {
        return Err(SlabError::Config(format!(
            "allocation point `{point}` is not feasible"
        )));
    }
    let rows = alloc
        .get("profile_indices")
        .and_then(Value::as_array)
        .ok_or_else(|| SlabError::Config("allocation lacks `profile_indices`".into()))?;
    if rows.len() != geometry.layers {
        return Err(SlabError::Config(format!(
            "profile_indices has {} layers, expected {}",
            rows.len(),
            geometry.layers
        )));
    }
    let mut formats = Vec::with_capacity(geometry.layers * geometry.experts);
    for (layer, row) in rows.iter().enumerate() {
        let row = row
            .as_array()
            .ok_or_else(|| SlabError::Config(format!("profile_indices[{layer}] is not an array")))?;
        let kept = keep.map(|k| k.kept(layer));
        let compact = kept.is_some_and(|ids| row.len() == ids.len());
        if row.len() != geometry.experts && !compact {
            return Err(SlabError::Config(format!(
                "profile_indices[{layer}] has {} experts, expected {}",
                row.len(),
                geometry.experts
            )));
        }
        for expert in 0..geometry.experts {
            if keep.is_some_and(|k| !k.is_kept(layer, expert)) {
                formats.push(ExpertFormat::Mq4Xts);
                continue;
            }
            let index = if compact { kept.unwrap().binary_search(&(expert as u16)).unwrap() } else { expert };
            let v = &row[index];
            formats.push(match v.as_u64() {
                Some(0) => ExpertFormat::Mq2I4,
                Some(1) => ExpertFormat::Mq3I4,
                Some(2) => ExpertFormat::Mq4Xts,
                _ => {
                    return Err(SlabError::Config(format!(
                        "profile_indices[{layer}][{expert}] = {v} is not 0, 1 or 2"
                    )))
                }
            });
        }
    }
    Ok(Allocation {
        formats,
        file_sha256,
        point: point.to_string(),
    })
}

/// Apply optional original-id `tiers: {layer: {expert: "w3" | "mq4"}}`.
pub fn apply_keep_tiers(root: &Value, keep: &ExpertKeep, formats: &mut [ExpertFormat]) -> Result<(), SlabError> {
    let Some(tiers) = root.get("tiers") else { return Ok(()) };
    if tiers.is_null() { return Ok(()) }
    if formats.len() != keep.n_layers() * keep.n_experts() {
        return Err(SlabError::Config("tier format geometry mismatch".into()));
    }
    let apply = |layer: usize, expert: usize, value: &Value, formats: &mut [ExpertFormat]| -> Result<(), SlabError> {
        if value.is_null() { return Ok(()) }
        if !keep.is_kept(layer, expert) {
            return Err(SlabError::Config(format!("tier for absent expert L{layer}/E{expert}")));
        }
        formats[layer * keep.n_experts() + expert] = match value.as_str() {
            Some("w3") => ExpertFormat::Mq3I4,
            Some("mq4") => ExpertFormat::Mq4Xts,
            _ => return Err(SlabError::Config(format!("invalid tier L{layer}/E{expert}: {value}"))),
        };
        Ok(())
    };
    let rows = tiers.as_object().ok_or_else(|| SlabError::Config("tiers must be an object keyed by layer id".into()))?;
    for (layer, row) in rows {
        let layer = layer.parse::<usize>().map_err(|_| SlabError::Config("invalid tier layer id".into()))?;
        if layer >= keep.n_layers() { return Err(SlabError::Config("tier layer outside geometry".into())) }
        let row = row.as_object().ok_or_else(|| SlabError::Config(format!("invalid tiers row {layer}")))?;
        for (expert, value) in row {
            let expert = expert.parse::<usize>().map_err(|_| SlabError::Config("invalid tier expert id".into()))?;
            apply(layer, expert, value, formats)?;
        }
    }
    Ok(())
}

/// A uniform format map for `geometry`.
pub fn uniform_formats(geometry: SlabGeometry, format: ExpertFormat) -> Vec<ExpertFormat> {
    vec![format; geometry.layers * geometry.experts]
}

/// Where the format map came from (recorded in provenance and receipt).
#[derive(Clone, Debug)]
pub enum AllocationSource {
    File { file_sha256: String, point: String },
    Uniform(ExpertFormat),
}

impl AllocationSource {
    fn to_json(&self) -> Value {
        match self {
            Self::File { file_sha256, point } => {
                json!({"file_sha256": file_sha256, "point": point})
            }
            Self::Uniform(f) => json!({"uniform": format_name(*f)}),
        }
    }
}

// ---------------------------------------------------------------------------
// Record providers
// ---------------------------------------------------------------------------

/// Supplies the final-code record of one (layer, expert) for one format.
/// Called concurrently from the packing workers.
pub trait ExpertRecordProvider: Sync {
    fn load(&self, layer: usize, expert: usize) -> Result<I4ExpertFinalCodeRecord, SlabError>;
    /// Identity of the record source for the provenance (root paths etc.).
    fn describe(&self) -> Value;
    /// Verification counters for the receipt.
    fn stats(&self) -> Value;
}

/// Records from `root/L{L}/E{e}.npz` (the R1 `q/<arm>` tree), with optional
/// manifest and Python-pack cross-checks.
pub struct NpzDirProvider {
    root: PathBuf,
    format: ExpertFormat,
    shapes: I4RecordShapes,
    manifests: Option<PathBuf>,
    cross_check_packs: Option<PathBuf>,
    manifest_cache: Mutex<HashMap<(PathBuf, usize), Arc<Value>>>,
    loaded: AtomicU64,
    manifest_checked: AtomicU64,
    packs_checked: AtomicU64,
}

impl NpzDirProvider {
    pub fn new(root: PathBuf, format: ExpertFormat, shapes: I4RecordShapes) -> Self {
        Self {
            root,
            format,
            shapes,
            manifests: None,
            cross_check_packs: None,
            manifest_cache: Mutex::new(HashMap::new()),
            loaded: AtomicU64::new(0),
            manifest_checked: AtomicU64::new(0),
            packs_checked: AtomicU64::new(0),
        }
    }

    /// Verify each NPZ's sha256 against `dir/L{L}/manifest.json`
    /// (`experts.E{e}.npz_sha256`) before import.
    pub fn with_manifests(mut self, dir: PathBuf) -> Self {
        self.manifests = Some(dir);
        self
    }

    /// Compare every encoded slab byte-for-byte with the Python pack
    /// `dir/L{L}/E{e}.bin`, and that file's sha256 with `dir/L{L}/manifest.json`.
    pub fn with_cross_check_packs(mut self, dir: PathBuf) -> Self {
        self.cross_check_packs = Some(dir);
        self
    }

    fn manifest(&self, dir: &Path, layer: usize) -> Result<Arc<Value>, String> {
        let key = (dir.to_path_buf(), layer);
        if let Some(m) = lock(&self.manifest_cache).get(&key) {
            return Ok(Arc::clone(m));
        }
        let path = dir.join(format!("L{layer}")).join("manifest.json");
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: invalid JSON: {e}", path.display()))?;
        let arc = Arc::new(value);
        lock(&self.manifest_cache).insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    /// `(manifest, E{e} entry)` after identity checks of the manifest header.
    fn manifest_entry(
        &self,
        dir: &Path,
        layer: usize,
        expert: usize,
        bits: u64,
    ) -> Result<Value, String> {
        let m = self.manifest(dir, layer)?;
        let at = dir.join(format!("L{layer}/manifest.json"));
        if let Some(l) = m.get("layer").and_then(Value::as_u64) {
            if l != layer as u64 {
                return Err(format!("{}: layer {l} != {layer}", at.display()));
            }
        }
        if let Some(b) = m.get("bits").and_then(Value::as_u64) {
            if b != bits {
                return Err(format!("{}: bits {b} != {bits}", at.display()));
            }
        }
        m.get("experts")
            .and_then(|e| e.get(format!("E{expert}")))
            .cloned()
            .ok_or_else(|| format!("{}: no entry experts.E{expert}", at.display()))
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ExpertRecordProvider for NpzDirProvider {
    fn load(&self, layer: usize, expert: usize) -> Result<I4ExpertFinalCodeRecord, SlabError> {
        let fail = |message: String| SlabError::Record {
            layer,
            expert,
            message,
        };
        let bits = match self.format {
            ExpertFormat::Mq3I4 => 3u64,
            ExpertFormat::Mq2I4 => 2u64,
            ExpertFormat::Mq4Xts => return Err(fail("NPZ provider cannot supply MQ4-XTS".into())),
        };
        let path = self.root.join(format!("L{layer}")).join(format!("E{expert}.npz"));
        let bytes = std::fs::read(&path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
        let rec = I4ExpertFinalCodeRecord::import_npz_bytes(
            &bytes,
            layer,
            expert,
            self.format,
            self.shapes,
        )
        .map_err(|e| fail(format!("{}: {e}", path.display())))?;
        if let Some(dir) = &self.manifests {
            let entry = self.manifest_entry(dir, layer, expert, bits).map_err(&fail)?;
            let want = entry.get("npz_sha256").and_then(Value::as_str);
            if want != Some(rec.source_npz_sha256.as_str()) {
                return Err(fail(format!(
                    "{}: npz sha256 {} != manifest {:?}",
                    path.display(),
                    rec.source_npz_sha256,
                    want
                )));
            }
            self.manifest_checked.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(dir) = &self.cross_check_packs {
            let pack_path = dir.join(format!("L{layer}")).join(format!("E{expert}.bin"));
            let pack = std::fs::read(&pack_path)
                .map_err(|e| fail(format!("{}: {e}", pack_path.display())))?;
            let mut encoded = Vec::with_capacity(rec.slab_len());
            encoded.extend_from_slice(&rec.gate_up.bytes);
            encoded.extend_from_slice(&rec.down.bytes);
            if pack != encoded {
                let at = pack.iter().zip(&encoded).position(|(a, b)| a != b);
                return Err(fail(format!(
                    "{}: Python pack differs from the Rust encoding (len {} vs {}, first diff {:?})",
                    pack_path.display(),
                    pack.len(),
                    encoded.len(),
                    at
                )));
            }
            let entry = self.manifest_entry(dir, layer, expert, bits).map_err(&fail)?;
            let pack_sha = hex_lower(&Sha256::digest(&pack));
            if entry.get("sha256").and_then(Value::as_str) != Some(pack_sha.as_str()) {
                return Err(fail(format!(
                    "{}: pack sha256 {pack_sha} != manifest {:?}",
                    pack_path.display(),
                    entry.get("sha256")
                )));
            }
            if let Some(n) = entry.get("npz_sha256").and_then(Value::as_str) {
                if n != rec.source_npz_sha256 {
                    return Err(fail(format!(
                        "{}: pack manifest npz_sha256 {n} != NPZ {}",
                        pack_path.display(),
                        rec.source_npz_sha256
                    )));
                }
            }
            self.packs_checked.fetch_add(1, Ordering::Relaxed);
        }
        self.loaded.fetch_add(1, Ordering::Relaxed);
        Ok(rec)
    }

    fn describe(&self) -> Value {
        json!({
            "format": format_name(self.format),
            "records_root": self.root.display().to_string(),
            "manifests_root": self.manifests.as_ref().map(|p| p.display().to_string()),
            "cross_check_packs_root": self.cross_check_packs.as_ref().map(|p| p.display().to_string()),
        })
    }

    fn stats(&self) -> Value {
        json!({
            "records_imported": self.loaded.load(Ordering::Relaxed),
            "npz_manifest_checks": self.manifest_checked.load(Ordering::Relaxed),
            "python_pack_cross_checks": self.packs_checked.load(Ordering::Relaxed),
        })
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Inputs of [`write_i4_expert_slabs`].
pub struct SlabWriteConfig<'a> {
    pub geometry: SlabGeometry,
    /// Pinned MQ4-XTS source HFQ.
    pub source: &'a Path,
    /// Output path; must not exist.
    pub out: &'a Path,
    /// Per-(layer,expert) format, row-major `layer * experts + expert`.
    pub formats: &'a [ExpertFormat],
    pub allocation: AllocationSource,
    /// Record provider for `ExpertFormat::Mq3I4` experts.
    pub w3: Option<&'a dyn ExpertRecordProvider>,
    /// Record provider for `ExpertFormat::Mq2I4` experts.
    pub w2: Option<&'a dyn ExpertRecordProvider>,
    /// Packing worker threads (≥ 1).
    pub threads: usize,
    /// Assert the computed source sha256 (hex).
    pub expected_source_sha256: Option<&'a str>,
    /// Print stage/progress lines to stderr.
    pub progress: bool,
}

#[derive(Clone, Debug)]
enum EntryKind {
    /// W3/W2 slab part produced by a worker.
    Packed { item: usize, proj: ExpertProj },
    /// MQ4-XTS slice of a source aggregate.
    Mq4Slice {
        item: usize,
        proj: ExpertProj,
        src_offset: u64,
    },
    /// Verbatim source tensor.
    Copy { src_offset: u64 },
    /// Compact original-id MTP aggregate slices.
    CopySlices { src_offset: u64, stride: u64, experts: Vec<u16> },
}

struct Prepared {
    gate_up: Vec<u8>,
    down: Vec<u8>,
    gate_up_sha: [u8; 32],
    down_sha: [u8; 32],
}

struct PipeState {
    next: usize,
    consumed: usize,
    abort: bool,
    ready: HashMap<usize, Result<Prepared, SlabError>>,
}

/// Ordered producer/consumer hand-off with a bounded lookahead window.
struct Pipeline {
    state: Mutex<PipeState>,
    cv: Condvar,
    n_items: usize,
    window: usize,
}

impl Pipeline {
    fn new(n_items: usize, window: usize) -> Self {
        Self {
            state: Mutex::new(PipeState {
                next: 0,
                consumed: 0,
                abort: false,
                ready: HashMap::new(),
            }),
            cv: Condvar::new(),
            n_items,
            window,
        }
    }

    fn take_work(&self) -> Option<usize> {
        let mut st = lock(&self.state);
        loop {
            if st.abort || st.next >= self.n_items {
                return None;
            }
            if st.next < st.consumed + self.window {
                let idx = st.next;
                st.next += 1;
                return Some(idx);
            }
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn publish(&self, idx: usize, res: Result<Prepared, SlabError>) {
        lock(&self.state).ready.insert(idx, res);
        self.cv.notify_all();
    }

    fn wait_ready(&self, idx: usize) -> Result<Prepared, SlabError> {
        let mut st = lock(&self.state);
        loop {
            if let Some(res) = st.ready.remove(&idx) {
                return res;
            }
            if st.abort {
                return Err(SlabError::Config("packing aborted".into()));
            }
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish_item(&self, idx: usize) {
        lock(&self.state).consumed = idx + 1;
        self.cv.notify_all();
    }

    fn abort(&self) {
        lock(&self.state).abort = true;
        self.cv.notify_all();
    }
}

struct AbortOnDrop<'a>(&'a Pipeline);

impl Drop for AbortOnDrop<'_> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn copy_span(
    file: &File,
    offset: u64,
    len: u64,
    w: &mut dyn Write,
    buf: &mut Vec<u8>,
    hasher: &mut Sha256,
) -> io::Result<()> {
    let mut done = 0u64;
    while done < len {
        let n = (len - done).min(IO_CHUNK as u64) as usize;
        if buf.len() < n {
            buf.resize(n, 0);
        }
        file.read_exact_at(&mut buf[..n], offset + done)?;
        hasher.update(&buf[..n]);
        w.write_all(&buf[..n])?;
        done += n as u64;
    }
    Ok(())
}

/// sha256 of a whole file by bounded positional reads.
fn sha256_file(path: &Path) -> Result<(String, u64), SlabError> {
    let file = File::open(path).map_err(|e| io_err(path.display(), e))?;
    let len = file.metadata().map_err(|e| io_err(path.display(), e))?.len();
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; IO_CHUNK.min(len.max(1) as usize)];
    let mut off = 0u64;
    while off < len {
        let n = (len - off).min(buf.len() as u64) as usize;
        file.read_exact_at(&mut buf[..n], off)
            .map_err(|e| io_err(path.display(), e))?;
        hasher.update(&buf[..n]);
        off += n as u64;
    }
    Ok((hex_lower(&hasher.finalize()), len))
}

/// Validate and prepare one W3/W2 expert: load, check identity, hash.
fn produce(
    cfg: &SlabWriteConfig<'_>,
    idx: usize,
) -> Result<Prepared, SlabError> {
    let layer = idx / cfg.geometry.experts;
    let expert = idx % cfg.geometry.experts;
    let format = cfg.formats[idx];
    let provider = match format {
        ExpertFormat::Mq3I4 => cfg.w3,
        ExpertFormat::Mq2I4 => cfg.w2,
        ExpertFormat::Mq4Xts => None,
    }
    .ok_or_else(|| SlabError::Config(format!("no record provider for {}", format_name(format))))?;
    let fail = |message: String| SlabError::Record {
        layer,
        expert,
        message,
    };
    let rec = provider.load(layer, expert)?;
    if (rec.layer, rec.expert, rec.format) != (layer, expert, format) {
        return Err(fail(format!(
            "provider returned L{}/E{} {:?}, wanted {:?}",
            rec.layer, rec.expert, rec.format, format
        )));
    }
    for (proj, p) in [(ExpertProj::GateUp, &rec.gate_up), (ExpertProj::Down, &rec.down)] {
        let want = cfg.geometry.entry_bytes(format, proj)?;
        let shape = cfg.geometry.proj_shape(proj).map(|d| d as usize);
        if p.bytes.len() != want || p.shape != shape {
            return Err(fail(format!(
                "{proj:?}: record is {:?}/{} B, wanted {shape:?}/{want} B",
                p.shape,
                p.bytes.len()
            )));
        }
        if p.tensor_name != expert_entry_name(layer, expert, proj) {
            return Err(fail(format!("{proj:?}: record names `{}`", p.tensor_name)));
        }
    }
    let I4ExpertFinalCodeRecord { gate_up, down, .. } = rec;
    Ok(Prepared {
        gate_up_sha: Sha256::digest(&gate_up.bytes).into(),
        down_sha: Sha256::digest(&down.bytes).into(),
        gate_up: gate_up.bytes,
        down: down.bytes,
    })
}

struct Plan {
    entries: Vec<HfqStreamEntry>,
    kinds: Vec<EntryKind>,
    /// Number of leading trunk slab entries (= 2 · layers · experts).
    n_trunk: usize,
    /// Source-order indices of copied tensors in `src.tensors()`.
    copied_src: Vec<usize>,
}

fn build_plan(
    cfg: &SlabWriteConfig<'_>,
    src: &HfqFile,
    keep: Option<&ExpertKeep>,
) -> Result<Plan, SlabError> {
    let g = cfg.geometry;
    let infos = src.tensors();
    let by_name: HashMap<&str, usize> = infos
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    if let Some(t) = infos
        .iter()
        .find(|t| parse_expert_entry_name(&t.name).is_some())
    {
        return Err(SlabError::Source(format!(
            "source already has per-expert entry `{}`",
            t.name
        )));
    }
    // Locate and validate the aggregates.
    let mut aggregate: HashMap<(usize, ExpertProj), usize> = HashMap::new();
    let mut aggregate_idx: HashSet<usize> = HashSet::new();
    for layer in 0..g.layers {
        for proj in [ExpertProj::GateUp, ExpertProj::Down] {
            let name = SlabGeometry::aggregate_name(layer, proj);
            let &i = by_name
                .get(name.as_str())
                .ok_or_else(|| SlabError::Source(format!("missing aggregate tensor `{name}`")))?;
            let t = &infos[i];
            let [m, k] = g.proj_shape(proj);
            let want_shape = vec![g.experts as u32, m, k];
            let per_expert = g.entry_bytes(ExpertFormat::Mq4Xts, proj)?;
            let want_len = g.experts * per_expert;
            if t.quant_type != SlabGeometry::mq4_qt(proj)
                || t.shape != want_shape
                || t.group_size != SlabGeometry::mq4_group(proj)
                || t.data_size != want_len
            {
                return Err(SlabError::Source(format!(
                    "aggregate `{name}` is qt{} shape {:?} group {} {} B; \
                     expected qt{} shape {want_shape:?} group {} {want_len} B",
                    t.quant_type,
                    t.shape,
                    t.group_size,
                    t.data_size,
                    SlabGeometry::mq4_qt(proj),
                    SlabGeometry::mq4_group(proj)
                )));
            }
            aggregate.insert((layer, proj), i);
            aggregate_idx.insert(i);
        }
    }
    let n_items = g.layers * g.experts;
    let mut entries = Vec::with_capacity(2 * n_items + infos.len());
    let mut kinds = Vec::with_capacity(2 * n_items + infos.len());
    for layer in 0..g.layers {
        for expert in 0..g.experts {
            if keep.is_some_and(|k| !k.is_kept(layer, expert)) {
                continue;
            }
            let item = layer * g.experts + expert;
            let format = cfg.formats[item];
            let (gu_qt, down_qt) = format.pair();
            for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                let qt = match proj {
                    ExpertProj::GateUp => gu_qt,
                    ExpertProj::Down => down_qt,
                };
                let data_len = g.entry_bytes(format, proj)? as u64;
                let (group_size, kind) = match format {
                    ExpertFormat::Mq4Xts => {
                        let agg = &infos[aggregate[&(layer, proj)]];
                        (
                            agg.group_size,
                            EntryKind::Mq4Slice {
                                item,
                                proj,
                                src_offset: agg.data_offset as u64 + expert as u64 * data_len,
                            },
                        )
                    }
                    _ => (mq_i4::GROUP as u32, EntryKind::Packed { item, proj }),
                };
                entries.push(HfqStreamEntry {
                    name: expert_entry_name(layer, expert, proj),
                    quant_type: qt,
                    shape: g.proj_shape(proj).to_vec(),
                    group_size,
                    data_len,
                });
                kinds.push(kind);
            }
        }
    }
    let n_trunk = entries.len();
    let mut copied_src = Vec::new();
    for (i, t) in infos.iter().enumerate() {
        if aggregate_idx.contains(&i) {
            continue;
        }
        let mut shape = t.shape.clone();
        let mut data_len = t.data_size as u64;
        let mut kind = EntryKind::Copy { src_offset: t.data_offset as u64 };
        if let Some(experts) = keep.and_then(ExpertKeep::mtp_kept) {
            if t.name.starts_with("mtp.") && t.name.contains(".mlp.experts.") {
                if shape.len() != 3 || shape[0] as usize != g.experts || t.data_size % g.experts != 0 {
                    return Err(SlabError::Source(format!("invalid MTP aggregate {}", t.name)));
                }
                let stride = t.data_size as u64 / g.experts as u64;
                shape[0] = experts.len() as u32;
                data_len = stride * experts.len() as u64;
                kind = EntryKind::CopySlices { src_offset: t.data_offset as u64, stride, experts: experts.to_vec() };
            }
        }
        entries.push(HfqStreamEntry {
            name: t.name.clone(),
            quant_type: t.quant_type,
            shape,
            group_size: t.group_size,
            data_len,
        });
        kinds.push(kind);
        copied_src.push(i);
    }
    Ok(Plan {
        entries,
        kinds,
        n_trunk,
        copied_src,
    })
}

fn rollup(digests: &[[u8; 32]]) -> String {
    let mut h = Sha256::new();
    for d in digests {
        h.update(d);
    }
    hex_lower(&h.finalize())
}

/// Write the mixed/uniform expert-slab artifact and verify it. Returns the JSON
/// receipt (also see the CLI's `--receipt`). A partially written output is
/// removed on failure.
pub fn write_i4_expert_slabs(cfg: &SlabWriteConfig<'_>) -> Result<Value, SlabError> {
    write_i4_expert_slabs_with_keep(cfg, None)
}

/// Pack only original-id experts present in `keep`.
pub fn write_i4_expert_slabs_with_keep(
    cfg: &SlabWriteConfig<'_>,
    keep: Option<&ExpertKeep>,
) -> Result<Value, SlabError> {
    if keep.is_some_and(|k| k.n_layers() != cfg.geometry.layers || k.n_experts() != cfg.geometry.experts) {
        return Err(SlabError::Config("keep geometry differs from slab geometry".into()));
    }
    if cfg.out.symlink_metadata().is_ok() {
        return Err(SlabError::Config(format!(
            "refusing to overwrite existing output {}",
            cfg.out.display()
        )));
    }
    let result = write_inner(cfg, keep, false);
    if result.is_err() {
        let _ = std::fs::remove_file(cfg.out);
    }
    result
}

/// Exact container size from the real pack plan, without writing an output.
pub fn size_i4_expert_slabs(cfg: &SlabWriteConfig<'_>, keep: Option<&ExpertKeep>) -> Result<Value, SlabError> {
    if keep.is_some_and(|k| k.n_layers() != cfg.geometry.layers || k.n_experts() != cfg.geometry.experts) {
        return Err(SlabError::Config("keep geometry differs from slab geometry".into()));
    }
    write_inner(cfg, keep, true)
}

fn write_inner(cfg: &SlabWriteConfig<'_>, keep: Option<&ExpertKeep>, size_only: bool) -> Result<Value, SlabError> {
    let t_total = Instant::now();
    let g = cfg.geometry;
    g.validate()?;
    if cfg.threads == 0 {
        return Err(SlabError::Config("threads must be >= 1".into()));
    }
    let n_items = g.layers * g.experts;
    if cfg.formats.len() != n_items {
        return Err(SlabError::Config(format!(
            "format map has {} entries, geometry needs {n_items}",
            cfg.formats.len()
        )));
    }
    let mut counts = [0u64; 3];
    for (item, f) in cfg.formats.iter().enumerate() {
        if keep.is_some_and(|k| !k.is_kept(item / g.experts, item % g.experts)) {
            continue;
        }
        counts[match f {
            ExpertFormat::Mq3I4 => 0,
            ExpertFormat::Mq2I4 => 1,
            ExpertFormat::Mq4Xts => 2,
        }] += 1;
    }
    if counts[0] > 0 && cfg.w3.is_none() {
        return Err(SlabError::Config(
            "format map uses MQ3-I4 but no W3 record provider was given".into(),
        ));
    }
    if counts[1] > 0 && cfg.w2.is_none() {
        return Err(SlabError::Config(
            "format map uses MQ2-I4 but no W2 record provider was given".into(),
        ));
    }

    // --- source ---------------------------------------------------------
    let src = HfqFile::open_with_reap_plan(cfg.source, None)
        .map_err(|e| io_err(cfg.source.display(), e))?;
    let mut metadata: Value = serde_json::from_str(&src.metadata_json)
        .map_err(|e| SlabError::Source(format!("metadata is not valid JSON: {e}")))?;
    let original_metadata = metadata.clone();
    {
        let obj = metadata
            .as_object_mut()
            .ok_or_else(|| SlabError::Source("metadata JSON is not an object".into()))?;
        for key in [EXPERT_SLABS_METADATA_KEY, PROVENANCE_METADATA_KEY, "qwen4.expert_keep"] {
            if obj.contains_key(key) {
                return Err(SlabError::Source(format!(
                    "source metadata already has `{key}`; refusing to re-pack"
                )));
            }
        }
    }
    let plan = build_plan(cfg, &src, keep)?;

    let t_src_hash = Instant::now();
    if cfg.progress {
        eprintln!("[i4-slabs] hashing source {}", cfg.source.display());
    }
    let (source_sha256, source_bytes) = sha256_file(cfg.source)?;
    let source_hash_s = t_src_hash.elapsed().as_secs_f64();
    if let Some(want) = cfg.expected_source_sha256 {
        if !want.eq_ignore_ascii_case(&source_sha256) {
            return Err(SlabError::Source(format!(
                "source sha256 {source_sha256} != asserted {want}"
            )));
        }
    }
    if source_bytes != src.file_len() {
        return Err(SlabError::Source(format!(
            "source size changed while packing ({source_bytes} vs {})",
            src.file_len()
        )));
    }

    // --- metadata -------------------------------------------------------
    let name_of = |p: &Path| {
        p.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let mut records = serde_json::Map::new();
    if let Some(p) = cfg.w3 {
        records.insert("mq3-i4".into(), p.describe());
    }
    if let Some(p) = cfg.w2 {
        records.insert("mq2-i4".into(), p.describe());
    }
    let format_counts = json!({"mq3-i4": counts[0], "mq2-i4": counts[1], "mq4-xts": counts[2]});
    let provenance = json!({
        "schema": "hipfire.qwen4.expert-slabs-provenance.v1",
        "packer": {"name": PACKER_NAME, "version": env!("CARGO_PKG_VERSION")},
        "source": {"file_name": name_of(cfg.source), "sha256": source_sha256, "bytes": source_bytes},
        "allocation": cfg.allocation.to_json(),
        "records": Value::Object(records),
        "format_counts": format_counts,
        "geometry": {
            "layers": g.layers, "experts_per_layer": g.experts,
            "gate_up": g.gate_up, "down": g.down,
        },
    });
    {
        let obj = metadata.as_object_mut().expect("checked above");
        obj.insert(
            EXPERT_SLABS_METADATA_KEY.into(),
            ExpertSlabsV1::fn_v1().to_json_value(),
        );
        obj.insert(PROVENANCE_METADATA_KEY.into(), provenance);
        if let Some(keep) = keep {
            obj.insert("qwen4.expert_keep".into(), keep.to_metadata_value());
        }
    }
    let metadata_json =
        serde_json::to_string(&metadata).map_err(|e| SlabError::Config(e.to_string()))?;
    if size_only {
        let index_bytes: u64 = 4 + plan.entries.iter()
            .map(|e| 2 + e.name.len() as u64 + 1 + 1 + 4 * e.shape.len() as u64 + 4 + 8)
            .sum::<u64>();
        let head = (32 + metadata_json.len() as u64 + index_bytes + 4095) & !4095;
        let routed: u64 = plan.entries[..plan.n_trunk].iter().map(|e| e.data_len).sum();
        let other: u64 = plan.entries[plan.n_trunk..].iter().map(|e| e.data_len).sum();
        return Ok(json!({"output":{"bytes":head + routed + other}, "format_counts":format_counts,
            "total_trunk_payload_bytes":routed, "non_expert":{"payload_bytes":other}}));
    }

    // --- write ----------------------------------------------------------
    let t_write = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()
        .map_err(|e| SlabError::Config(format!("thread pool: {e}")))?;
    let src_file = File::open(cfg.source).map_err(|e| io_err(cfg.source.display(), e))?;
    let work_items: Vec<usize> = (0..n_items)
        .filter(|&i| keep.map_or(true, |k| k.is_kept(i / g.experts, i % g.experts)))
        .collect();
    let pipeline = Pipeline::new(work_items.len(), cfg.threads * 4 + 4);
    let mut digests = vec![[0u8; 32]; plan.entries.len()];
    let first_error: Mutex<Option<SlabError>> = Mutex::new(None);
    let write_result = pool.in_place_scope(|scope| {
        for _ in 0..cfg.threads {
            scope.spawn(|_| {
                while let Some(work) = pipeline.take_work() {
                    let idx = work_items[work];
                    if cfg.formats[idx] == ExpertFormat::Mq4Xts {
                        continue;
                    }
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        produce(cfg, idx)
                    }))
                    .unwrap_or_else(|_| {
                        Err(SlabError::Record {
                            layer: idx / g.experts,
                            expert: idx % g.experts,
                            message: "packing worker panicked".into(),
                        })
                    });
                    pipeline.publish(idx, res);
                }
            });
        }
        let _abort = AbortOnDrop(&pipeline);
        let mut current: Option<Prepared> = None;
        let mut buf: Vec<u8> = Vec::new();
        let mut hasher = Sha256::new();
        let record_error = |e: SlabError| -> io::Error {
            let msg = e.to_string();
            lock(&first_error).get_or_insert(e);
            io::Error::other(msg)
        };
        let result = write_hfqm_package_streaming(
            cfg.out,
            src.arch_id,
            &metadata_json,
            &plan.entries,
            |i, w| {
                let len = plan.entries[i].data_len;
                match &plan.kinds[i] {
                    EntryKind::Packed { item, proj } => {
                        if *proj == ExpertProj::GateUp {
                            current = Some(pipeline.wait_ready(*item).map_err(&record_error)?);
                        }
                        let p = current
                            .as_ref()
                            .ok_or_else(|| record_error(SlabError::Config("no prepared slab".into())))?;
                        let (bytes, sha) = match proj {
                            ExpertProj::GateUp => (&p.gate_up, p.gate_up_sha),
                            ExpertProj::Down => (&p.down, p.down_sha),
                        };
                        w.write_all(bytes)?;
                        digests[i] = sha;
                        if *proj == ExpertProj::Down {
                            current = None;
                            pipeline.finish_item(work_items.binary_search(item).expect("planned item"));
                        }
                    }
                    EntryKind::Mq4Slice {
                        item,
                        proj,
                        src_offset,
                    } => {
                        hasher.reset();
                        copy_span(&src_file, *src_offset, len, w, &mut buf, &mut hasher)?;
                        digests[i] = hasher.finalize_reset().into();
                        if *proj == ExpertProj::Down {
                            pipeline.finish_item(work_items.binary_search(item).expect("planned item"));
                        }
                    }
                    EntryKind::Copy { src_offset } => {
                        hasher.reset();
                        copy_span(&src_file, *src_offset, len, w, &mut buf, &mut hasher)?;
                        digests[i] = hasher.finalize_reset().into();
                    }
                    EntryKind::CopySlices { src_offset, stride, experts } => {
                        hasher.reset();
                        for &expert in experts {
                            copy_span(&src_file, *src_offset + u64::from(expert) * stride, *stride, w, &mut buf, &mut hasher)?;
                        }
                        digests[i] = hasher.finalize_reset().into();
                    }
                }
                if cfg.progress && i < plan.n_trunk && i % (2 * g.experts) == 2 * g.experts - 1 {
                    eprintln!(
                        "[i4-slabs] layer {}/{} written ({:.0}s)",
                        i / (2 * g.experts) + 1,
                        g.layers,
                        t_write.elapsed().as_secs_f64()
                    );
                }
                Ok(())
            },
        );
        result
    });
    if let Err(e) = write_result {
        return Err(lock(&first_error)
            .take()
            .unwrap_or_else(|| io_err(cfg.out.display(), e)));
    }
    let write_s = t_write.elapsed().as_secs_f64();

    // --- verify ---------------------------------------------------------
    let t_verify = Instant::now();
    if cfg.progress {
        eprintln!("[i4-slabs] verifying {}", cfg.out.display());
    }
    let out_sha256 = verify_output(cfg, &plan, &digests, &original_metadata, &pool, keep)?;
    let verify_s = t_verify.elapsed().as_secs_f64();
    let out_bytes = std::fs::metadata(cfg.out)
        .map_err(|e| io_err(cfg.out.display(), e))?
        .len();

    // --- receipt --------------------------------------------------------
    let mut start = 0;
    let layer_rollups: Vec<String> = (0..g.layers)
        .map(|l| {
            let end = start + 2 * keep.map_or(g.experts, |k| k.kept(l).len());
            let hash = rollup(&digests[start..end]);
            start = end;
            hash
        })
        .collect();
    let trunk_bytes: u64 = plan.entries[..plan.n_trunk].iter().map(|e| e.data_len).sum();
    let mut non_expert = Sha256::new();
    for (e, d) in plan.entries[plan.n_trunk..]
        .iter()
        .zip(&digests[plan.n_trunk..])
    {
        non_expert.update((e.name.len() as u16).to_le_bytes());
        non_expert.update(e.name.as_bytes());
        non_expert.update([e.quant_type, e.shape.len() as u8]);
        for d in &e.shape {
            non_expert.update(d.to_le_bytes());
        }
        non_expert.update(e.group_size.to_le_bytes());
        non_expert.update(e.data_len.to_le_bytes());
        non_expert.update(d);
    }
    let non_expert_bytes: u64 = plan.entries[plan.n_trunk..].iter().map(|e| e.data_len).sum();
    let mut provider_stats = serde_json::Map::new();
    if let Some(p) = cfg.w3 {
        provider_stats.insert("mq3-i4".into(), p.stats());
    }
    if let Some(p) = cfg.w2 {
        provider_stats.insert("mq2-i4".into(), p.stats());
    }
    Ok(json!({
        "schema": "hipfire.qwen4.expert-slabs-receipt.v1",
        "packer": {"name": PACKER_NAME, "version": env!("CARGO_PKG_VERSION")},
        "output": {
            "path": cfg.out.display().to_string(),
            "bytes": out_bytes,
            "sha256": out_sha256,
            "tensors": plan.entries.len(),
            "trunk_slab_entries": plan.n_trunk,
        },
        "source": {
            "path": cfg.source.display().to_string(),
            "bytes": source_bytes,
            "sha256": source_sha256,
            "tensors": src.tensors().len(),
            "sha256_asserted": cfg.expected_source_sha256.is_some(),
        },
        "allocation": cfg.allocation.to_json(),
        "format_counts": format_counts,
        "total_trunk_payload_bytes": trunk_bytes,
        "layer_slab_rollups": {
            "definition": "sha256 over the concatenated per-entry payload sha256 digests (32 B each) of the layer's gate_up/down entries in index order",
            "sha256": layer_rollups,
        },
        "non_expert": {
            "tensors": plan.copied_src.len(),
            "payload_bytes": non_expert_bytes,
            "combined_sha256": hex_lower(&non_expert.finalize()),
            "combined_definition": "sha256 over, in output index order: u16le name_len, name, qt, rank, u32le dims, u32le group_size, u64le data_len, payload sha256",
        },
        "records": Value::Object(provider_stats),
        "geometry": {
            "layers": g.layers, "experts_per_layer": g.experts,
            "gate_up": g.gate_up, "down": g.down,
        },
        "threads": cfg.threads,
        "timings_s": {
            "source_sha256": source_hash_s,
            "write": write_s,
            "verify": verify_s,
            "total": t_total.elapsed().as_secs_f64(),
        },
        "verification": {
            "record_encode_decode_round_trip": "always on (every W3/W2 expert)",
            "index_order_shapes_qtypes_spans": "checked",
            "payload_sha256_vs_writer_input": "checked for every entry",
        },
    }))
}

/// One re-read unit of the verification pass.
enum Piece {
    Header(Vec<u8>),
    Entry { idx: usize, last: bool, data: Vec<u8> },
}

/// Reopen and verify the written artifact; returns the whole-file sha256.
fn verify_output(
    cfg: &SlabWriteConfig<'_>,
    plan: &Plan,
    digests: &[[u8; 32]],
    original_metadata: &Value,
    pool: &rayon::ThreadPool,
    keep: Option<&ExpertKeep>,
) -> Result<String, SlabError> {
    let vf = |m: String| SlabError::Verify(m);
    let out = HfqFile::open_with_reap_plan(cfg.out, None)
        .map_err(|e| io_err(format!("reopening {}", cfg.out.display()), e))?;
    if out.arch_id
        != HfqFile::probe_arch_id(cfg.source).map_err(|e| io_err(cfg.source.display(), e))?
    {
        return Err(vf("output arch_id differs from the source".into()));
    }
    // Metadata: every original key unchanged, plus exactly the two new keys.
    let meta: Value = serde_json::from_str(&out.metadata_json)
        .map_err(|e| vf(format!("output metadata is not JSON: {e}")))?;
    let (orig, got) = (
        original_metadata.as_object().expect("source metadata is an object"),
        meta.as_object()
            .ok_or_else(|| vf("output metadata is not an object".into()))?,
    );
    for (k, v) in orig {
        if got.get(k) != Some(v) {
            return Err(vf(format!("metadata key `{k}` changed")));
        }
    }
    if got.len() != orig.len() + 2 + usize::from(keep.is_some()) {
        return Err(vf("output metadata has unexpected extra keys".into()));
    }
    if got.get(EXPERT_SLABS_METADATA_KEY) != Some(&ExpertSlabsV1::fn_v1().to_json_value()) {
        return Err(vf(format!("`{EXPERT_SLABS_METADATA_KEY}` is not the v1 value")));
    }
    if !got.contains_key(PROVENANCE_METADATA_KEY) {
        return Err(vf(format!("`{PROVENANCE_METADATA_KEY}` missing")));
    }
    // Index order, shapes, qtypes, group sizes, contiguous aligned spans.
    let infos = out.tensors();
    if infos.len() != plan.entries.len() {
        return Err(vf(format!(
            "output has {} tensors, expected {}",
            infos.len(),
            plan.entries.len()
        )));
    }
    let data_offset = out.data_offset();
    if data_offset % SLAB_ALIGNMENT != 0 {
        return Err(vf(format!("data_offset {data_offset} is not 256-aligned")));
    }
    let mut cursor = data_offset;
    let mut spans = Vec::with_capacity(infos.len());
    for (i, (info, want)) in infos.iter().zip(&plan.entries).enumerate() {
        if info.name != want.name
            || info.quant_type != want.quant_type
            || info.shape != want.shape
            || info.group_size != want.group_size
            || info.data_size as u64 != want.data_len
        {
            return Err(vf(format!("index entry {i} `{}` differs from the plan", want.name)));
        }
        if info.data_offset as u64 != cursor {
            return Err(vf(format!(
                "entry {i} `{}` starts at {} but the previous payload ends at {cursor}",
                want.name, info.data_offset
            )));
        }
        if i < plan.n_trunk && cursor % SLAB_ALIGNMENT != 0 {
            return Err(vf(format!("slab `{}` at {cursor} is not 256-aligned", want.name)));
        }
        spans.push((cursor, want.data_len));
        cursor += want.data_len;
    }
    if cursor != out.file_len() {
        return Err(vf(format!(
            "payloads end at {cursor} but the file is {} bytes",
            out.file_len()
        )));
    }
    drop(out);

    // Re-read every byte: per-entry digests + whole-file digest in one pass.
    let file = File::open(cfg.out).map_err(|e| io_err(cfg.out.display(), e))?;
    let mut file_hasher = Sha256::new();
    let mismatch: Mutex<Option<String>> = Mutex::new(None);
    let io_failure: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|s| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Piece>(4);
        let (file, spans) = (&file, &spans);
        let io_failure = &io_failure;
        s.spawn(move || {
            let read = |off: u64, n: usize| -> io::Result<Vec<u8>> {
                let mut v = vec![0u8; n];
                file.read_exact_at(&mut v, off)?;
                Ok(v)
            };
            let run = || -> io::Result<()> {
                let mut off = 0u64;
                while off < data_offset {
                    let n = (data_offset - off).min(IO_CHUNK as u64) as usize;
                    if tx.send(Piece::Header(read(off, n)?)).is_err() {
                        return Ok(());
                    }
                    off += n as u64;
                }
                for (idx, &(start, len)) in spans.iter().enumerate() {
                    if len == 0 {
                        let _ = tx.send(Piece::Entry {
                            idx,
                            last: true,
                            data: Vec::new(),
                        });
                        continue;
                    }
                    let mut done = 0u64;
                    while done < len {
                        let n = (len - done).min(IO_CHUNK as u64) as usize;
                        let data = read(start + done, n)?;
                        done += n as u64;
                        if tx
                            .send(Piece::Entry {
                                idx,
                                last: done == len,
                                data,
                            })
                            .is_err()
                        {
                            return Ok(());
                        }
                    }
                }
                Ok(())
            };
            if let Err(e) = run() {
                *lock(io_failure) = Some(e.to_string());
            }
        });
        pool.install(|| {
            let mut entry_hasher = Sha256::new();
            for piece in rx {
                match piece {
                    Piece::Header(data) => file_hasher.update(&data),
                    Piece::Entry { idx, last, data } => {
                        rayon::join(
                            || file_hasher.update(&data),
                            || entry_hasher.update(&data),
                        );
                        if last {
                            let d: [u8; 32] = entry_hasher.finalize_reset().into();
                            if d != digests[idx] {
                                lock(&mismatch).get_or_insert(format!(
                                    "payload sha256 of `{}` differs from the bytes handed to the writer",
                                    plan.entries[idx].name
                                ));
                            }
                        }
                    }
                }
            }
        });
    });
    if let Some(e) = lock(&io_failure).take() {
        return Err(io_err(format!("re-reading {}", cfg.out.display()), e));
    }
    if let Some(m) = lock(&mismatch).take() {
        return Err(vf(m));
    }
    Ok(hex_lower(&file_hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

    fn keep_config<'a>(fx: &'a Fixture, out: &'a Path, formats: &'a [ExpertFormat], w3: &'a SynthProvider) -> SlabWriteConfig<'a> {
        SlabWriteConfig {
            geometry: TINY, source: &fx.source, out, formats,
            allocation: AllocationSource::Uniform(ExpertFormat::Mq4Xts),
            w3: Some(w3), w2: None, threads: 1,
            expected_source_sha256: None, progress: false,
        }
    }

    #[test]
    fn ragged_keep_tiers_size_and_mtp() {
        let fx = fixture();
        let out = fx.dir.join("ragged.hfq");
        let keep = ExpertKeep::new(4, vec![vec![3], vec![0, 2]], Some(vec![1, 3]), 1).unwrap();
        let mut formats = uniform_formats(TINY, ExpertFormat::Mq2I4);
        apply_keep_tiers(&json!({"tiers":{"0":{"3":"mq4"},"1":{"0":"w3","2":"mq4"}}}), &keep, &mut formats).unwrap();
        let w3 = SynthProvider::new(ExpertFormat::Mq3I4);
        let cfg = keep_config(&fx, &out, &formats, &w3);
        let size = size_i4_expert_slabs(&cfg, Some(&keep)).unwrap();
        assert!(!out.exists());
        let receipt = write_i4_expert_slabs_with_keep(&cfg, Some(&keep)).unwrap();
        assert_eq!(size["output"]["bytes"], receipt["output"]["bytes"]);
        assert_eq!(receipt["format_counts"], json!({"mq3-i4":1,"mq2-i4":0,"mq4-xts":2}));
        assert_eq!(w3.calls.load(Ordering::Relaxed), 1);
        let hfq = HfqFile::open(&out).unwrap();
        let metadata: Value = serde_json::from_str(&hfq.metadata_json).unwrap();
        assert_eq!(ExpertKeep::from_metadata(&metadata).unwrap(), Some(keep.clone()));
        let mut seen = 0;
        for t in hfq.tensors() {
            if let Some((l, e, _)) = parse_expert_entry_name(&t.name) {
                assert!(keep.is_kept(l, e));
                seen += 1;
            }
            if t.name.starts_with("mtp.") {
                let original = fx.tensors.iter().find(|src| src.name == t.name).unwrap();
                let stride = original.data.len() / 4;
                assert_eq!(t.shape[0], 2);
                let (_, bytes) = hfq.tensor_data(&t.name).unwrap();
                assert_eq!(bytes, [&original.data[stride..2*stride], &original.data[3*stride..]].concat());
            }
        }
        assert_eq!(seen, 6);
    }

    #[test]
    fn tiers_and_compact_allocation_refusals() {
        let fx = fixture();
        let keep = ExpertKeep::new(4, vec![vec![3], vec![0, 2]], None, 1).unwrap();
        let path = fx.dir.join("alloc.json");
        let root = json!({"profiles":["mq2-i4","mq3-i4","mq4-xts"],
            "allocations":{"test":{"feasible":true,"profile_indices":[[1],[2,1]]}}});
        std::fs::write(&path, root.to_string()).unwrap();
        let alloc = load_allocation_with_keep(&path, "test", TINY, Some(&keep)).unwrap();
        assert_eq!(alloc.formats[3], ExpertFormat::Mq3I4);
        assert_eq!(alloc.formats[4], ExpertFormat::Mq4Xts);
        assert_eq!(alloc.formats[6], ExpertFormat::Mq3I4);
        assert!(load_allocation(&path, "test", TINY).is_err());
        for tiers in [
            json!({"0":{"1":"w3"}}),
            json!({"0":{"3":"unknown"}}),
            json!({"2":{"0":"w3"}}),
            json!({"0":{"5":"mq4"}}),
        ] {
            assert!(apply_keep_tiers(&json!({"tiers":tiers}), &keep, &mut alloc.formats.clone()).is_err());
        }
        let out = fx.dir.join("missing-record.hfq");
        let mut w3 = SynthProvider::new(ExpertFormat::Mq3I4);
        w3.fail_at = Some((0, 3));
        let cfg = keep_config(&fx, &out, &alloc.formats, &w3);
        assert!(write_i4_expert_slabs_with_keep(&cfg, Some(&keep)).is_err());
        assert!(!out.exists());
    }

    #[test]
    fn no_keep_prechange_golden_hash() {
        // Independent reconstruction of the frozen pre-pruning MQ4 layout.
        let fx = fixture();
        let out = fx.dir.join("no-keep.hfq");
        let golden = fx.dir.join("golden.hfq");
        let formats = uniform_formats(TINY, ExpertFormat::Mq4Xts);
        let w3 = SynthProvider::new(ExpertFormat::Mq3I4);
        let cfg = keep_config(&fx, &out, &formats, &w3);
        let receipt = write_i4_expert_slabs(&cfg).unwrap();
        let (source_sha, source_bytes) = sha256_file(&fx.source).unwrap();
        let mut meta = fx.metadata.clone();
        meta[EXPERT_SLABS_METADATA_KEY] = ExpertSlabsV1::fn_v1().to_json_value();
        meta[PROVENANCE_METADATA_KEY] = json!({
            "schema":"hipfire.qwen4.expert-slabs-provenance.v1",
            "packer":{"name":PACKER_NAME,"version":env!("CARGO_PKG_VERSION")},
            "source":{"file_name":"source.hfq","sha256":source_sha,"bytes":source_bytes},
            "allocation":{"uniform":"mq4-xts"},
            "records":{"mq3-i4":{"synthetic":"mq3-i4"}},
            "format_counts":{"mq3-i4":0,"mq2-i4":0,"mq4-xts":8},
            "geometry":{"layers":2,"experts_per_layer":4,"gate_up":[64,256],"down":[128,128]}
        });
        let mut tensors = Vec::new();
        for layer in 0..2 {
            for expert in 0..4 {
                for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                    let src = fx.tensors.iter().find(|t| t.name == SlabGeometry::aggregate_name(layer, proj)).unwrap();
                    let stride = src.data.len() / 4;
                    tensors.push(mem(&expert_entry_name(layer, expert, proj), src.quant_type,
                        &TINY.proj_shape(proj), src.group_size, src.data[expert*stride..(expert+1)*stride].to_vec()));
                }
            }
        }
        for t in &fx.tensors {
            if !(0..2).any(|l| [ExpertProj::GateUp, ExpertProj::Down].iter()
                .any(|&p| t.name == SlabGeometry::aggregate_name(l, p))) {
                tensors.push(mem(&t.name, t.quant_type, &t.shape, t.group_size, t.data.clone()));
            }
        }
        write_hfqm_package_mem(&golden, 9, &meta.to_string(), &tensors).unwrap();
        assert_eq!(receipt["output"]["sha256"], sha256_file(&golden).unwrap().0);
        assert_eq!(size_i4_expert_slabs(&cfg, None).unwrap()["output"]["bytes"], receipt["output"]["bytes"]);
    }

    #[test]
    fn cpu_loader_kept_table_roundtrip() {
        use hipfire_arch_qwen4::expert_slab_table::build_expert_slab_table_with_keep;
        use hipfire_runtime::hfq::HfqTensorInfo;
        let keep = ExpertKeep::new(512, (0..48).map(|l| (0..10 + l % 3).collect()).collect(), None, 10).unwrap();
        let root = json!({"qwen4.expert_keep":keep.to_metadata_value()});
        let parsed = ExpertKeep::from_metadata(&root).unwrap().unwrap();
        let mut infos = Vec::new();
        let mut offset = 4096;
        for layer in 0..48 {
            for &expert in keep.kept(layer) {
                for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                    let len = SlabGeometry::FN.entry_bytes(ExpertFormat::Mq4Xts, proj).unwrap();
                    infos.push(HfqTensorInfo {
                        name: expert_entry_name(layer, expert as usize, proj),
                        quant_type: SlabGeometry::mq4_qt(proj),
                        shape: SlabGeometry::FN.proj_shape(proj).to_vec(),
                        group_size: SlabGeometry::mq4_group(proj),
                        data_offset: offset, data_size: len,
                    });
                    offset += len;
                }
            }
        }
        let table = build_expert_slab_table_with_keep(&ExpertSlabsV1::fn_v1(), &infos, Some(&parsed)).unwrap();
        assert_eq!(table.counts().mq4, keep.total_kept());
        for layer in 0..48 {
            for expert in 0..512 {
                let slab = table.slab(layer, expert).unwrap();
                assert_eq!(slab.gate_up.len == 0 && slab.down.len == 0, !keep.is_kept(layer, expert));
            }
        }
        infos[0].name = expert_entry_name(0, 511, ExpertProj::GateUp);
        assert!(build_expert_slab_table_with_keep(&ExpertSlabsV1::fn_v1(), &infos, Some(&parsed)).is_err());
    }

    #[test]
    fn split_source_marker_is_preserved_without_ple_rows() {
        let mut fx = fixture();
        fx.metadata["hipfire_ple_rows"] = json!({
            "version":1,"location":"sidecar","shard_count":2,
            "rows_per_shard":4,"row_width":256,"q8_rows_sha256":"ab".repeat(32)
        });
        std::fs::remove_file(&fx.source).unwrap();
        write_hfqm_package_mem(&fx.source, 9, &fx.metadata.to_string(), &fx.tensors).unwrap();
        let keep = ExpertKeep::new(4, vec![vec![3],vec![0,2]], None, 1).unwrap();
        let formats = uniform_formats(TINY, ExpertFormat::Mq4Xts);
        let w3 = SynthProvider::new(ExpertFormat::Mq3I4);
        let out = fx.dir.join("coder-base.mq3-xts");
        let cfg = keep_config(&fx, &out, &formats, &w3);
        let size = size_i4_expert_slabs(&cfg, Some(&keep)).unwrap();
        let receipt = write_i4_expert_slabs_with_keep(&cfg, Some(&keep)).unwrap();
        assert_eq!(size["output"]["bytes"], receipt["output"]["bytes"]);
        let hfq = HfqFile::open(&out).unwrap();
        let meta: Value = serde_json::from_str(&hfq.metadata_json).unwrap();
        assert_eq!(meta["hipfire_ple_rows"], fx.metadata["hipfire_ple_rows"]);
        assert!(!hfq.tensors().iter().any(|t| t.name.contains(".ngram_embedding.shard_")));
    }

    /// Tiny geometry whose every slab entry is a multiple of 256 B:
    /// gate_up `[64,256]` (R256), down `[128,128]` (R128).
    const TINY: SlabGeometry = SlabGeometry {
        layers: 2,
        experts: 4,
        gate_up: [64, 256],
        down: [128, 128],
    };

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| (self.next() >> 24) as u8).collect()
        }
    }

    /// Deterministic synthetic records (no NPZ): valid random codes, scales.
    struct SynthProvider {
        format: ExpertFormat,
        geometry: SlabGeometry,
        calls: AtomicU64,
        fail_at: Option<(usize, usize)>,
    }

    impl SynthProvider {
        fn new(format: ExpertFormat) -> Self {
            Self {
                format,
                geometry: TINY,
                calls: AtomicU64::new(0),
                fail_at: None,
            }
        }

        fn tensor(
            rng: &mut Rng,
            codec_levels: u8,
            mid: u8,
            m: usize,
            k: usize,
        ) -> (Vec<u8>, Vec<u16>) {
            let mut codes: Vec<u8> = (0..m * k)
                .map(|_| (rng.next() % (u64::from(codec_levels) + 1)) as u8)
                .collect();
            let groups = m * (k / 128);
            let mut scales: Vec<u16> = (0..groups)
                .map(|_| (((1 + rng.next() % 30) << 10) | (rng.next() % 1024)) as u16)
                .collect();
            // One zero-scale group per tensor with midpoint codes.
            scales[1] = 0;
            let (row, g) = (1 / (k / 128), 1 % (k / 128));
            for c in &mut codes[row * k + g * 128..row * k + g * 128 + 128] {
                *c = mid;
            }
            (codes, scales)
        }

        fn parts(&self, layer: usize, expert: usize) -> ((Vec<u8>, Vec<u16>), (Vec<u8>, Vec<u16>)) {
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ ((layer as u64) << 20) ^ ((expert as u64) << 8) ^ self.format.pair().0 as u64);
            let (levels, mid) = match self.format {
                ExpertFormat::Mq3I4 => (7, 4),
                _ => (3, 2),
            };
            let [gm, gk] = self.geometry.gate_up.map(|d| d as usize);
            let [dm, dk] = self.geometry.down.map(|d| d as usize);
            (
                Self::tensor(&mut rng, levels, mid, gm, gk),
                Self::tensor(&mut rng, levels, mid, dm, dk),
            )
        }
    }

    impl ExpertRecordProvider for SynthProvider {
        fn load(&self, layer: usize, expert: usize) -> Result<I4ExpertFinalCodeRecord, SlabError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail_at == Some((layer, expert)) {
                return Err(SlabError::Record {
                    layer,
                    expert,
                    message: "injected".into(),
                });
            }
            let (gu, down) = self.parts(layer, expert);
            I4ExpertFinalCodeRecord::from_parts(
                layer,
                expert,
                self.format,
                self.geometry.record_shapes(),
                gu,
                down,
                "mqi4-canonical-v1-rne-tree128",
                "uniform",
                String::new(),
            )
            .map_err(|e| SlabError::Record {
                layer,
                expert,
                message: e.to_string(),
            })
        }
        fn describe(&self) -> Value {
            json!({"synthetic": format_name(self.format)})
        }
        fn stats(&self) -> Value {
            json!({"calls": self.calls.load(Ordering::Relaxed)})
        }
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        dir: PathBuf,
        source: PathBuf,
        /// name -> (qt, shape, group, bytes) in source index order
        tensors: Vec<HfqMemTensor>,
        metadata: Value,
    }

    fn mem(name: &str, qt: u8, shape: &[u32], group: u32, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name: name.to_string(),
            quant_type: qt,
            shape: shape.to_vec(),
            group_size: group,
            data,
        }
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let mut rng = Rng(0xC0FFEE);
        let g = TINY;
        let gu_b = g.entry_bytes(ExpertFormat::Mq4Xts, ExpertProj::GateUp).unwrap();
        let dn_b = g.entry_bytes(ExpertFormat::Mq4Xts, ExpertProj::Down).unwrap();
        let mut tensors = vec![
            mem("model.embed_tokens.weight", 16, &[10, 8], 0, rng.bytes(160)),
            mem("model.language_model.norm.weight", 1, &[8], 0, rng.bytes(16)),
        ];
        for layer in 0..g.layers {
            tensors.push(mem(
                &format!("model.language_model.layers.{layer}.input_layernorm.weight"),
                1,
                &[8],
                0,
                rng.bytes(16),
            ));
            tensors.push(mem(
                &SlabGeometry::aggregate_name(layer, ExpertProj::GateUp),
                44,
                &[g.experts as u32, g.gate_up[0], g.gate_up[1]],
                256,
                rng.bytes(g.experts * gu_b),
            ));
            tensors.push(mem(
                &SlabGeometry::aggregate_name(layer, ExpertProj::Down),
                53,
                &[g.experts as u32, g.down[0], g.down[1]],
                128,
                rng.bytes(g.experts * dn_b),
            ));
            tensors.push(mem(
                &format!("model.language_model.layers.{layer}.self_attn.q_proj.weight"),
                3,
                &[16, 16],
                64,
                rng.bytes(300),
            ));
        }
        // Legacy MTP aggregates (kept) and a zero-length tensor.
        tensors.push(mem(
            "mtp.layers.0.mlp.experts.gate_up_proj",
            44,
            &[g.experts as u32, g.gate_up[0], g.gate_up[1]],
            256,
            rng.bytes(g.experts * gu_b),
        ));
        tensors.push(mem(
            "mtp.layers.0.mlp.experts.down_proj",
            53,
            &[g.experts as u32, g.down[0], g.down[1]],
            128,
            rng.bytes(g.experts * dn_b),
        ));
        tensors.push(mem("lm_head.weight", 54, &[4, 4], 0, rng.bytes(128)));
        tensors.push(mem("empty.bias", 1, &[0], 0, Vec::new()));
        let metadata = json!({
            "format": "hfq", "format_version": 1, "arch_id": 9,
            "model_type": "qwen4",
            "config": {"layers": 2, "float": 1.0e-6, "nested": {"a": [1, 2, 3]}},
            "qwen4.expert_encoding": {"symmetric": true, "payloads": "payloads"},
            "tokenizer": {"x": "y"},
        });
        let source = dir.join("source.hfq");
        write_hfqm_package_mem(&source, 9, &metadata.to_string(), &tensors).unwrap();
        Fixture {
            _dir: tmp,
            dir,
            source,
            tensors,
            metadata,
        }
    }

    fn mixed_formats(g: SlabGeometry) -> Vec<ExpertFormat> {
        (0..g.layers * g.experts)
            .map(|i| match i % 3 {
                0 => ExpertFormat::Mq3I4,
                1 => ExpertFormat::Mq2I4,
                _ => ExpertFormat::Mq4Xts,
            })
            .collect()
    }

    fn run_from(
        source: &Path,
        formats: &[ExpertFormat],
        out: &Path,
        threads: usize,
        w3: &SynthProvider,
        w2: &SynthProvider,
    ) -> Result<Value, SlabError> {
        write_i4_expert_slabs(&SlabWriteConfig {
            geometry: TINY,
            source,
            out,
            formats,
            allocation: AllocationSource::File {
                file_sha256: "ab".repeat(32),
                point: "free-test".into(),
            },
            w3: Some(w3),
            w2: Some(w2),
            threads,
            expected_source_sha256: None,
            progress: false,
        })
    }

    fn run(
        fx: &Fixture,
        formats: &[ExpertFormat],
        out: &Path,
        threads: usize,
        w3: &SynthProvider,
        w2: &SynthProvider,
    ) -> Result<Value, SlabError> {
        run_from(&fx.source, formats, out, threads, w3, w2)
    }

    #[test]
    fn mixed_artifact_layout_payloads_and_metadata() {
        let fx = fixture();
        let formats = mixed_formats(TINY);
        let out = fx.dir.join("out.hfq");
        let (w3, w2) = (
            SynthProvider::new(ExpertFormat::Mq3I4),
            SynthProvider::new(ExpertFormat::Mq2I4),
        );
        let receipt = run(&fx, &formats, &out, 3, &w3, &w2).unwrap();

        let hfq = HfqFile::open_with_reap_plan(&out, None).unwrap();
        let src = HfqFile::open_with_reap_plan(&fx.source, None).unwrap();
        let infos = hfq.tensors();
        let n_items = TINY.layers * TINY.experts;

        // Index order, names, shapes, qtypes, groups.
        let mut i = 0;
        for layer in 0..TINY.layers {
            for expert in 0..TINY.experts {
                let format = formats[layer * TINY.experts + expert];
                for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                    let t = &infos[i];
                    i += 1;
                    assert_eq!(t.name, expert_entry_name(layer, expert, proj));
                    let (gu, dn) = format.pair();
                    assert_eq!(t.quant_type, if proj == ExpertProj::GateUp { gu } else { dn });
                    assert_eq!(t.shape, TINY.proj_shape(proj).to_vec());
                    assert_eq!(
                        t.group_size,
                        match format {
                            ExpertFormat::Mq4Xts => SlabGeometry::mq4_group(proj),
                            _ => 128,
                        }
                    );
                    assert_eq!(t.data_size, TINY.entry_bytes(format, proj).unwrap());
                    assert_eq!(t.data_offset % 256, 0, "{} not 256-aligned", t.name);
                }
            }
        }
        assert_eq!(i, 2 * n_items);
        // Contiguity of the whole payload region.
        let mut cursor = hfq.data_offset() as usize;
        for t in infos {
            assert_eq!(t.data_offset, cursor, "{}", t.name);
            cursor += t.data_size;
        }
        assert_eq!(cursor as u64, hfq.file_len());
        assert_eq!(hfq.arch_id, 9);

        // Non-expert tensors: identical, in original order, aggregates absent,
        // MTP aggregates present.
        let kept: Vec<&HfqMemTensor> = fx
            .tensors
            .iter()
            .filter(|t| {
                !(0..TINY.layers).any(|l| {
                    t.name == SlabGeometry::aggregate_name(l, ExpertProj::GateUp)
                        || t.name == SlabGeometry::aggregate_name(l, ExpertProj::Down)
                })
            })
            .collect();
        assert_eq!(infos.len(), 2 * n_items + kept.len());
        for (t, want) in infos[2 * n_items..].iter().zip(&kept) {
            assert_eq!(t.name, want.name);
            assert_eq!(t.quant_type, want.quant_type);
            assert_eq!(t.shape, want.shape);
            assert_eq!(t.group_size, want.group_size);
            let (_, data) = hfq.tensor_data(&t.name).unwrap();
            assert_eq!(data, want.data.as_slice(), "{}", t.name);
        }
        assert!(hfq.find_tensor_info("mtp.layers.0.mlp.experts.gate_up_proj").is_some());
        assert!(hfq.find_tensor_info("mtp.layers.0.mlp.experts.down_proj").is_some());
        for l in 0..TINY.layers {
            for p in [ExpertProj::GateUp, ExpertProj::Down] {
                assert!(hfq
                    .find_tensor_info(&SlabGeometry::aggregate_name(l, p))
                    .is_none());
            }
        }

        // Slab payloads: MQ4 = source slice; W3/W2 decode to the record codes.
        for layer in 0..TINY.layers {
            for expert in 0..TINY.experts {
                let format = formats[layer * TINY.experts + expert];
                for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                    let name = expert_entry_name(layer, expert, proj);
                    let (info, data) = hfq.tensor_data(&name).unwrap();
                    let _ = info;
                    match format {
                        ExpertFormat::Mq4Xts => {
                            let agg = SlabGeometry::aggregate_name(layer, proj);
                            let (_, agg_data) = src.tensor_data(&agg).unwrap();
                            let n = TINY.entry_bytes(format, proj).unwrap();
                            assert_eq!(data, &agg_data[expert * n..(expert + 1) * n], "{name}");
                        }
                        _ => {
                            let provider = if format == ExpertFormat::Mq3I4 { &w3 } else { &w2 };
                            let (gu, dn) = provider.parts(layer, expert);
                            let ((codes, scales), prof, shape) = match proj {
                                ExpertProj::GateUp => (
                                    gu,
                                    MqI4Profile::from_quant_type(format.pair().0).unwrap(),
                                    TINY.gate_up,
                                ),
                                ExpertProj::Down => (
                                    dn,
                                    MqI4Profile::from_quant_type(format.pair().1).unwrap(),
                                    TINY.down,
                                ),
                            };
                            let dec = mq_i4::decode_tensor(
                                prof,
                                shape[0] as usize,
                                shape[1] as usize,
                                data,
                            )
                            .unwrap();
                            assert_eq!(dec.codes, codes, "{name}");
                            assert_eq!(dec.scale_bits, scales, "{name}");
                        }
                    }
                }
            }
        }

        // Metadata: originals unchanged + both new keys.
        let meta: Value = serde_json::from_str(&hfq.metadata_json).unwrap();
        for (k, v) in fx.metadata.as_object().unwrap() {
            assert_eq!(&meta[k], v, "metadata key {k}");
        }
        assert_eq!(
            meta[EXPERT_SLABS_METADATA_KEY],
            ExpertSlabsV1::fn_v1().to_json_value()
        );
        let prov = &meta[PROVENANCE_METADATA_KEY];
        assert_eq!(prov["packer"]["name"], PACKER_NAME);
        assert_eq!(prov["source"]["bytes"], std::fs::metadata(&fx.source).unwrap().len());
        assert_eq!(prov["allocation"]["point"], "free-test");
        assert_eq!(prov["format_counts"]["mq3-i4"], 3);
        assert_eq!(prov["format_counts"]["mq2-i4"], 3);
        assert_eq!(prov["format_counts"]["mq4-xts"], 2);
        assert_eq!(prov["records"]["mq3-i4"]["synthetic"], "mq3-i4");

        // Receipt agrees with an independent hash of the file.
        let (sha, size) = sha256_file(&out).unwrap();
        assert_eq!(receipt["output"]["sha256"], sha);
        assert_eq!(receipt["output"]["bytes"], size);
        assert_eq!(
            receipt["source"]["sha256"],
            sha256_file(&fx.source).unwrap().0
        );
        assert_eq!(receipt["format_counts"]["mq4-xts"], 2);
        assert_eq!(receipt["non_expert"]["tensors"], kept.len());
        assert_eq!(receipt["layer_slab_rollups"]["sha256"].as_array().unwrap().len(), 2);
        let trunk: usize = infos[..2 * n_items].iter().map(|t| t.data_size).sum();
        assert_eq!(receipt["total_trunk_payload_bytes"], trunk);
        assert_eq!(receipt["threads"], 3);
    }

    #[test]
    fn output_is_independent_of_thread_count_and_deterministic() {
        let fx = fixture();
        let formats = mixed_formats(TINY);
        let (w3, w2) = (
            SynthProvider::new(ExpertFormat::Mq3I4),
            SynthProvider::new(ExpertFormat::Mq2I4),
        );
        let a = fx.dir.join("a.hfq");
        let b = fx.dir.join("b.hfq");
        let ra = run(&fx, &formats, &a, 1, &w3, &w2).unwrap();
        let rb = run(&fx, &formats, &b, 7, &w3, &w2).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        assert_eq!(ra["output"]["sha256"], rb["output"]["sha256"]);
        assert_eq!(ra["layer_slab_rollups"], rb["layer_slab_rollups"]);
        assert_eq!(ra["non_expert"]["combined_sha256"], rb["non_expert"]["combined_sha256"]);
    }

    #[test]
    fn uniform_w3_w2_and_mq4_artifacts() {
        let fx = fixture();
        let (w3, w2) = (
            SynthProvider::new(ExpertFormat::Mq3I4),
            SynthProvider::new(ExpertFormat::Mq2I4),
        );
        let mut sizes = Vec::new();
        for (name, f) in [
            ("w3", ExpertFormat::Mq3I4),
            ("w2", ExpertFormat::Mq2I4),
            ("mq4", ExpertFormat::Mq4Xts),
        ] {
            let out = fx.dir.join(format!("{name}.hfq"));
            let r = run(&fx, &uniform_formats(TINY, f), &out, 2, &w3, &w2).unwrap();
            let hfq = HfqFile::open_with_reap_plan(&out, None).unwrap();
            assert_eq!(hfq.tensors()[0].quant_type, f.pair().0);
            assert_eq!(hfq.tensors()[1].quant_type, f.pair().1);
            sizes.push(r["total_trunk_payload_bytes"].as_u64().unwrap());
        }
        let per = |f: ExpertFormat| (TINY.layers * TINY.experts * (
            TINY.entry_bytes(f, ExpertProj::GateUp).unwrap()
                + TINY.entry_bytes(f, ExpertProj::Down).unwrap())) as u64;
        assert_eq!(sizes, [per(ExpertFormat::Mq3I4), per(ExpertFormat::Mq2I4), per(ExpertFormat::Mq4Xts)]);
    }

    #[test]
    fn refuses_existing_output_and_existing_slab_metadata() {
        let fx = fixture();
        let formats = mixed_formats(TINY);
        let (w3, w2) = (
            SynthProvider::new(ExpertFormat::Mq3I4),
            SynthProvider::new(ExpertFormat::Mq2I4),
        );
        let out = fx.dir.join("exists.hfq");
        std::fs::write(&out, b"keep me").unwrap();
        assert!(matches!(
            run(&fx, &formats, &out, 2, &w3, &w2),
            Err(SlabError::Config(_))
        ));
        assert_eq!(std::fs::read(&out).unwrap(), b"keep me");

        // A source that already carries slabs is refused (re-pack of a packed artifact).
        let packed = fx.dir.join("packed.hfq");
        run(&fx, &formats, &packed, 2, &w3, &w2).unwrap();
        let out2 = fx.dir.join("repack.hfq");
        let err = run_from(&packed, &formats, &out2, 2, &w3, &w2).unwrap_err();
        assert!(matches!(err, SlabError::Source(_)), "{err}");
        assert!(!out2.exists());
    }

    #[test]
    fn source_sha_assertion_and_layout_checks() {
        let fx = fixture();
        let formats = uniform_formats(TINY, ExpertFormat::Mq4Xts);
        let (w3, w2) = (
            SynthProvider::new(ExpertFormat::Mq3I4),
            SynthProvider::new(ExpertFormat::Mq2I4),
        );
        let good = sha256_file(&fx.source).unwrap().0;
        let mk = |out: &Path, sha: Option<&str>, g: SlabGeometry| {
            write_i4_expert_slabs(&SlabWriteConfig {
                geometry: g,
                source: &fx.source,
                out,
                formats: &formats,
                allocation: AllocationSource::Uniform(ExpertFormat::Mq4Xts),
                w3: Some(&w3),
                w2: Some(&w2),
                threads: 2,
                expected_source_sha256: sha,
                progress: false,
            })
        };
        let wrong = "00".repeat(32);
        let o1 = fx.dir.join("o1.hfq");
        assert!(matches!(mk(&o1, Some(&wrong), TINY), Err(SlabError::Source(_))));
        assert!(!o1.exists());
        let o2 = fx.dir.join("o2.hfq");
        mk(&o2, Some(&good), TINY).unwrap();
        // Wrong geometry (aggregate shape mismatch) is a source-layout error.
        let o3 = fx.dir.join("o3.hfq");
        let bad_geom = SlabGeometry { experts: 8, ..TINY };
        assert!(matches!(
            write_i4_expert_slabs(&SlabWriteConfig {
                geometry: bad_geom,
                formats: &uniform_formats(bad_geom, ExpertFormat::Mq4Xts),
                ..SlabWriteConfig {
                    geometry: TINY,
                    source: &fx.source,
                    out: &o3,
                    formats: &formats,
                    allocation: AllocationSource::Uniform(ExpertFormat::Mq4Xts),
                    w3: None,
                    w2: None,
                    threads: 1,
                    expected_source_sha256: None,
                    progress: false,
                }
            }),
            Err(SlabError::Source(_))
        ));
    }

    #[test]
    fn missing_provider_wrong_map_length_and_zero_threads_are_config_errors() {
        let fx = fixture();
        let formats = mixed_formats(TINY);
        let w3 = SynthProvider::new(ExpertFormat::Mq3I4);
        let o = fx.dir.join("x.hfq");
        let base = || SlabWriteConfig {
            geometry: TINY,
            source: &fx.source,
            out: &o,
            formats: &formats,
            allocation: AllocationSource::Uniform(ExpertFormat::Mq4Xts),
            w3: Some(&w3),
            w2: None,
            threads: 2,
            expected_source_sha256: None,
            progress: false,
        };
        assert!(matches!(write_i4_expert_slabs(&base()), Err(SlabError::Config(_))));
        let short = &formats[..3];
        assert!(matches!(
            write_i4_expert_slabs(&SlabWriteConfig { formats: short, w2: Some(&w3), ..base() }),
            Err(SlabError::Config(_))
        ));
        assert!(matches!(
            write_i4_expert_slabs(&SlabWriteConfig { threads: 0, w2: Some(&w3), ..base() }),
            Err(SlabError::Config(_))
        ));
        assert!(!o.exists());
    }

    #[test]
    fn provider_failure_aborts_cleanly_and_removes_the_partial_output() {
        let fx = fixture();
        let formats = uniform_formats(TINY, ExpertFormat::Mq3I4);
        let mut w3 = SynthProvider::new(ExpertFormat::Mq3I4);
        w3.fail_at = Some((1, 2));
        let w2 = SynthProvider::new(ExpertFormat::Mq2I4);
        let out = fx.dir.join("fail.hfq");
        let err = run(&fx, &formats, &out, 4, &w3, &w2).unwrap_err();
        assert!(
            matches!(err, SlabError::Record { layer: 1, expert: 2, .. }),
            "{err}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn provider_returning_the_wrong_expert_is_rejected() {
        struct Liar(SynthProvider);
        impl ExpertRecordProvider for Liar {
            fn load(&self, layer: usize, expert: usize) -> Result<I4ExpertFinalCodeRecord, SlabError> {
                self.0.load(layer, (expert + 1) % TINY.experts)
            }
            fn describe(&self) -> Value { Value::Null }
            fn stats(&self) -> Value { Value::Null }
        }
        let fx = fixture();
        let liar = Liar(SynthProvider::new(ExpertFormat::Mq3I4));
        let formats = uniform_formats(TINY, ExpertFormat::Mq3I4);
        let out = fx.dir.join("liar.hfq");
        let err = write_i4_expert_slabs(&SlabWriteConfig {
            geometry: TINY,
            source: &fx.source,
            out: &out,
            formats: &formats,
            allocation: AllocationSource::Uniform(ExpertFormat::Mq3I4),
            w3: Some(&liar),
            w2: None,
            threads: 2,
            expected_source_sha256: None,
            progress: false,
        })
        .unwrap_err();
        assert!(matches!(err, SlabError::Record { .. }), "{err}");
    }

    #[test]
    fn fn_geometry_sizes_match_the_published_contract() {
        let g = SlabGeometry::FN;
        g.validate().unwrap();
        assert_eq!(g.entry_bytes(ExpertFormat::Mq3I4, ExpertProj::GateUp).unwrap(), 1_280_000);
        assert_eq!(g.entry_bytes(ExpertFormat::Mq3I4, ExpertProj::Down).unwrap(), 640_000);
        assert_eq!(g.entry_bytes(ExpertFormat::Mq2I4, ExpertProj::GateUp).unwrap(), 870_400);
        assert_eq!(g.entry_bytes(ExpertFormat::Mq2I4, ExpertProj::Down).unwrap(), 435_200);
        assert_eq!(g.entry_bytes(ExpertFormat::Mq4Xts, ExpertProj::GateUp).unwrap(), 1_740_800);
        assert_eq!(g.entry_bytes(ExpertFormat::Mq4Xts, ExpertProj::Down).unwrap(), 870_400);
        for f in [ExpertFormat::Mq3I4, ExpertFormat::Mq2I4, ExpertFormat::Mq4Xts] {
            assert_eq!(
                g.entry_bytes(f, ExpertProj::GateUp).unwrap()
                    + g.entry_bytes(f, ExpertProj::Down).unwrap(),
                f.slab_bytes()
            );
        }
    }

    #[test]
    fn allocation_loader_validates_the_r1_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let g = SlabGeometry { layers: 2, experts: 3, ..TINY };
        let ok = json!({
            "profiles": ["mq2-i4", "mq3-i4", "mq4-xts"],
            "allocations": {
                "free-1GiB-per-rank": {"feasible": true, "profile_indices": [[0, 1, 2], [2, 1, 0]]},
                "free-9GiB-per-rank": {"feasible": false},
            }
        });
        let write = |name: &str, v: &Value| {
            let p = tmp.path().join(name);
            std::fs::write(&p, v.to_string()).unwrap();
            p
        };
        let p = write("ok.json", &ok);
        let a = load_allocation(&p, "free-1GiB-per-rank", g).unwrap();
        assert_eq!(
            a.formats,
            [
                ExpertFormat::Mq2I4, ExpertFormat::Mq3I4, ExpertFormat::Mq4Xts,
                ExpertFormat::Mq4Xts, ExpertFormat::Mq3I4, ExpertFormat::Mq2I4
            ]
        );
        assert_eq!(a.file_sha256, hex_lower(&Sha256::digest(std::fs::read(&p).unwrap())));
        assert!(load_allocation(&p, "free-9GiB-per-rank", g).is_err());
        assert!(load_allocation(&p, "missing", g).is_err());
        assert!(load_allocation(&p, "free-1GiB-per-rank", SlabGeometry { layers: 3, ..g }).is_err());
        assert!(load_allocation(&p, "free-1GiB-per-rank", SlabGeometry { experts: 4, ..g }).is_err());
        let mut bad = ok.clone();
        bad["profiles"] = json!(["mq3-i4", "mq2-i4", "mq4-xts"]);
        assert!(load_allocation(&write("p.json", &bad), "free-1GiB-per-rank", g).is_err());
        let mut bad = ok.clone();
        bad["allocations"]["free-1GiB-per-rank"]["profile_indices"][0][1] = json!(3);
        assert!(load_allocation(&write("i.json", &bad), "free-1GiB-per-rank", g).is_err());
    }

    /// Real NPZ providers with manifests and Python-pack cross-checks, on the
    /// tiny fixtures (geometry gate_up [4,256], down [8,128]).
    #[test]
    fn npz_provider_checks_manifests_and_python_packs() {
        let fx = "tests/fixtures/subq4";
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(fx);
        let tmp = tempfile::tempdir().unwrap();
        let shapes = I4RecordShapes { gate_up: [4, 256], down: [8, 128] };
        for (npz, expected, format, bits) in [
            ("tiny_w3", "tiny_w3.expected.bin", ExpertFormat::Mq3I4, 3u64),
            ("tiny_w2", "tiny_w2.expected.bin", ExpertFormat::Mq2I4, 2u64),
        ] {
            let q = tmp.path().join(format!("q-{npz}"));
            let pack = tmp.path().join(format!("pack-{npz}"));
            std::fs::create_dir_all(q.join("L3")).unwrap();
            std::fs::create_dir_all(pack.join("L3")).unwrap();
            let npz_bytes = std::fs::read(root.join(format!("{npz}.npz"))).unwrap();
            let bin = std::fs::read(root.join(expected)).unwrap();
            std::fs::write(q.join("L3/E5.npz"), &npz_bytes).unwrap();
            std::fs::write(pack.join("L3/E5.bin"), &bin).unwrap();
            let manifest = json!({
                "layer": 3, "bits": bits,
                "experts": {"E5": {
                    "sha256": hex_lower(&Sha256::digest(&bin)),
                    "npz_sha256": hex_lower(&Sha256::digest(&npz_bytes)),
                }}
            });
            std::fs::write(pack.join("L3/manifest.json"), manifest.to_string()).unwrap();
            let provider = NpzDirProvider::new(q.clone(), format, shapes)
                .with_manifests(pack.clone())
                .with_cross_check_packs(pack.clone());
            let rec = provider.load(3, 5).unwrap();
            assert_eq!([rec.gate_up.bytes, rec.down.bytes].concat(), bin);
            assert_eq!(provider.stats()["npz_manifest_checks"], 1);
            assert_eq!(provider.stats()["python_pack_cross_checks"], 1);
            assert_eq!(provider.describe()["format"], format_name(format));

            // A flipped pack byte is caught byte-for-byte.
            let mut bad = bin.clone();
            bad[7] ^= 1;
            std::fs::write(pack.join("L3/E5.bin"), &bad).unwrap();
            let err = provider.load(3, 5).unwrap_err();
            assert!(matches!(err, SlabError::Record { layer: 3, expert: 5, .. }), "{err}");
            std::fs::write(pack.join("L3/E5.bin"), &bin).unwrap();

            // A wrong npz_sha256 in the manifest is caught before import is trusted.
            let mut m = manifest.clone();
            m["experts"]["E5"]["npz_sha256"] = json!("00".repeat(32));
            std::fs::write(pack.join("L3/manifest.json"), m.to_string()).unwrap();
            let fresh = NpzDirProvider::new(q.clone(), format, shapes).with_manifests(pack.clone());
            assert!(fresh.load(3, 5).is_err());
            // Missing record file.
            assert!(fresh.load(3, 6).is_err());
        }
    }
}
