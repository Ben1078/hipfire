// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_igpu_engine_parity — GPU parity gate for `memory.offload_exec=igpu`
//! (`hipfire_dispatch::igpu_exec`), one real Flash-Next model and one prefill tile.
//!
//! The production model loads exactly like `qwen4_kld` (admission, PLE rows, trunk
//! overlay, manifest fulfilment, bundle assembly, `attach_forward`) on the model
//! device (the XTX, `Gpu::init()`), with the routed experts of every layer host
//! mapped (`--vram-layers N` keeps layers `0..N` resident; default 0 = all
//! host-mapped) and `HIPFIRE_OFFLOAD_EXEC=igpu` arming the iGPU layer engine. One
//! `forward_chunk_final` of `--rows` real tokens then runs as ONE prefill tile.
//!
//! The engine is switched into parity mode first. For every engined layer it
//! retains the exact inputs of its call (natural F32 `x_norm`, the original top-10
//! ids and weights, the combine target's input bytes), the combine target it
//! returned to the XTX, and the combine target of the SAME layer computed again
//! entirely on the gfx1151 from those inputs (separate non-pinned device buffers,
//! the same production kernels: scatter, rotate256, gate/up SiLU IU4 sym, rotate128,
//! down IU4 sym, BF16-input canonical combine).
//!
//! Per engined layer this example then
//!   1. compares the engine target with the standalone gfx1151 target BITWISE
//!      (same kernels, same inputs: any mismatching element fails the run), and
//!   2. runs the XTX's own production F16 WMMA route for that layer on the same
//!      inputs (`moe_scatter_fused_top10`, F16 rotate,
//!      `gemm_mq4g256v2_moe_grouped_top10_silu_bf16out`, fused unscatter+rotate128,
//!      `gemm_mq4g128v2_moe_grouped_top10_xf16` BF16 out, BF16-input combine), over
//!      the layer's host-mapped expert pointer tables on the model device, and
//!      reports the engine target's max abs / max rel difference against it.
//!      This is a measurement (IU4 sym vs F16 WMMA are different numeric routes),
//!      not a gate; non-finite values fail the run.
//!
//! Then the tails/edge matrix, on the SAME recorded real-model carrier (each matrix
//! layer's recorded natural `x_norm`, original top-10 ids and weights, plus its real
//! host-mapped weights): every tail width 1/2/4/8/16/64/128/511/512/513 x the
//! variants real, skew (every row on the same ten experts), repeat (duplicate experts
//! inside a row), invalid (ids -1 and >= experts, whole rows unrouted), weights
//! (zero/x64/÷64/negated) and extreme (x1e4, x1e-30, 0, x1e19 rows) runs through the
//! engine's real transport path (`igpu_parity_execute`: model-device staging, D2H,
//! iGPU kernels, H2D, zero-initial routed-only target) at ANY row count. This is a
//! parity-only entry: production admission stays at more than four rows and decode is
//! untouched. Each case's engine target must equal the standalone gfx1151 target of
//! the same inputs bitwise; rows that are unrouted or zero-weighted must be exactly
//! zero; every variant but `extreme` must stay finite; the unmodified rows must be
//! prefix-invariant across widths (the first n rows of an n-row call equal the same
//! rows of the 513-row call) and run-to-run deterministic.
//!
//! Usage:
//!   HIPFIRE_OFFLOAD_EXEC=igpu [HIPFIRE_IGPU_DEVICE=N] \
//!   cargo run --release -p hipfire-arch-qwen4 --features lab \
//!     --example qwen4_igpu_engine_parity -- MODEL.mq4 TEXT.txt \
//!     [--rows 1024] [--offset 0] [--vram-layers 0] [--layers 0,1,2|all] \
//!     [--no-matrix] [--matrix-layers 0,1,2] [--json OUT.json]
//!
//! `--rows` (default 1024) is the tile and must be >= 513 for the matrix. Exit 0 only
//! if every selected engined layer ran exactly once in the tile, its engine target
//! equals the standalone gfx1151 target bitwise, nothing is non-finite, and every
//! matrix case passes; 1 on a parity failure; 2 on an error.

use hipfire_arch_qwen4::admit_hfqm_artifact_on;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::program::grouped_m_total_bound;
use hipfire_dispatch::igpu_exec::{IgpuParityCase, IgpuParityRecord};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource, SourcePayload};
use hipfire_runtime::ple_sidecar::{resolve_and_attach_ple_rows, ResolvePolicy};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

const USAGE: &str = "usage: HIPFIRE_OFFLOAD_EXEC=igpu qwen4_igpu_engine_parity MODEL.mq4 TEXT.txt \
    [--rows 1024] [--offset 0] [--vram-layers 0] [--layers 0,1,2|all] \
    [--no-matrix] [--matrix-layers 0,1,2] [--json OUT.json]";

/// Smallest tile the iGPU engine and the symmetric IU4 route take
/// (`QWEN4_MOE_SYM_IU4_MIN_ROWS`).
const MIN_ROWS: usize = rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS;

// ── arguments ────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    text: PathBuf,
    rows: usize,
    offset: usize,
    vram_layers: usize,
    /// `None` = every engined layer.
    layers: Option<Vec<usize>>,
    /// Run the tails/edge matrix (default on).
    matrix: bool,
    /// Matrix layers; `None` = first, middle and last selected layer.
    matrix_layers: Option<Vec<usize>>,
    json: Option<PathBuf>,
}

fn parse_args() -> R<Args> {
    let mut positional = Vec::new();
    let mut rows = 1024usize;
    let mut offset = 0usize;
    let mut vram_layers = 0usize;
    let mut layers: Option<Vec<usize>> = None;
    let mut json = None;
    let mut matrix = true;
    let mut matrix_layers: Option<Vec<usize>> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value\n{USAGE}"));
        match arg.as_str() {
            "--rows" => rows = value("--rows")?.parse().map_err(|x| format!("--rows: {x}"))?,
            "--offset" => offset = value("--offset")?.parse().map_err(|x| format!("--offset: {x}"))?,
            "--vram-layers" => {
                vram_layers = value("--vram-layers")?.parse().map_err(|x| format!("--vram-layers: {x}"))?
            }
            "--layers" => {
                let list = value("--layers")?;
                layers = if list.trim() == "all" {
                    None
                } else {
                    Some(
                        list.split(',')
                            .filter(|v| !v.trim().is_empty())
                            .map(|v| v.trim().parse::<usize>().map_err(|x| format!("--layers '{v}': {x}")))
                            .collect::<R<Vec<_>>>()?,
                    )
                };
            }
            "--json" => json = Some(PathBuf::from(value("--json")?)),
            "--no-matrix" => matrix = false,
            "--matrix-layers" => {
                matrix_layers = Some(
                    value("--matrix-layers")?
                        .split(',')
                        .filter(|v| !v.trim().is_empty())
                        .map(|v| v.trim().parse::<usize>().map_err(|x| format!("--matrix-layers '{v}': {x}")))
                        .collect::<R<Vec<_>>>()?,
                )
            }
            other if other.starts_with("--") => return Err(format!("unknown argument {other}\n{USAGE}")),
            other => positional.push(other.to_string()),
        }
    }
    let [model, text] = <[String; 2]>::try_from(positional)
        .map_err(|_| format!("need MODEL and TEXT\n{USAGE}"))?;
    if rows < MIN_ROWS {
        return Err(format!(
            "--rows {rows}: the symmetric IU4 route (and so the iGPU tile) needs >= {MIN_ROWS} rows"
        ));
    }
    if matrix && rows < TAILS[TAILS.len() - 1] {
        return Err(format!(
            "--rows {rows}: the matrix slices up to {} rows of the recorded tile (use --rows >= {} or --no-matrix)",
            TAILS[TAILS.len() - 1],
            TAILS[TAILS.len() - 1]
        ));
    }
    Ok(Args {
        model: PathBuf::from(model),
        text: PathBuf::from(text),
        rows,
        offset,
        vram_layers,
        layers,
        matrix,
        matrix_layers,
        json,
    })
}

// ── model ────────────────────────────────────────────────────────────

/// Layer geometry the XTX reference needs.
#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    mi: usize,
    experts: usize,
    k_top: usize,
}

struct Model {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    dims: Dims,
    vocab: usize,
}

/// The `qwen4_kld` load (the serve loader's range-loaded path) with the routed
/// experts of layers `vram_layers..` host-mapped; `HIPFIRE_OFFLOAD_EXEC=igpu` arms the
/// engine inside `attach_forward`. `rows` is both the context and the tile bound.
fn load(path: &Path, rows: usize, vram_layers: usize) -> R<Model> {
    let mut hfq = HfqFile::open(path).map_err(|x| format!("open HFQM {}: {x}", path.display()))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|x| format!("tokenizer: {x}"))?;
    let mut gpu = Gpu::init().map_err(e)?;
    if gpu.is_uma() {
        return Err(format!(
            "model device {} is an integrated GPU: the parity needs a discrete model device (the XTX) plus a distinct gfx1151",
            gpu.arch
        ));
    }
    if gpu.arch.to_string() == "gfx1151" {
        return Err("model device 0 is the gfx1151; expose the XTX as device 0".into());
    }
    // Portable host flags must precede every expert allocation: arm the iGPU engine's
    // load (a no-op unless HIPFIRE_OFFLOAD_EXEC=igpu); `attach_forward` opens the engine.
    if hipfire_arch_qwen4::gpu_forward::igpu_engine::arm_igpu_for_load(
        &mut gpu,
        &hipfire_arch_qwen4::gpu_forward::igpu_engine::Qwen4IgpuLoadInputs::PLAIN,
    )
    .is_none()
    {
        return Err("the iGPU engine was not armed at load (see the one-line [igpu-exec] notice above)".into());
    }
    let choice = resolve_and_attach_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
        ResolvePolicy { derive_missing_q4: false, progress: None },
    )
    .map_err(|x| format!("qwen4 PLE rows: {x}"))?;
    eprintln!("qwen4_igpu_engine_parity: ple rows: {}", choice.format.name());
    let receipt =
        admit_hfqm_artifact_on(&hfq, Some(&gpu.arch)).map_err(|x| format!("qwen4 admission: {x}"))?;
    let trunk_overlay = hipfire_arch_qwen4::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
        .map_err(|x| format!("qwen4 trunk overlay: {x}"))?;
    let scratch_mode = hipfire_arch_qwen4::artifact::source_forward_scratch_mode(
        &gpu,
        &hfq,
        &receipt.config,
        &receipt.manifest,
    )
    .map_err(e)?;
    let npu_identity = hipfire_arch_qwen4::npu_trunk::model_index_identity(&hfq, &gpu.arch);
    let mesh = DeviceMesh::single().map_err(|x| format!("qwen4 mesh: {x}"))?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let mut weights = receipt.manifest.weights.clone();
    let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(&mut weights, vram_layers);
    eprintln!(
        "qwen4_igpu_engine_parity: routed experts: layers 0..{vram_layers} in VRAM, {moved} tensors host-mapped"
    );
    if moved == 0 {
        return Err("no routed expert tensor was host-mapped: the iGPU engine serves host-placed layers only".into());
    }
    // The ONE portable pinned backing the host expert aggregates become subviews of:
    // layers outside the pool are skipped by the engine (normal route), so the pool must
    // exist before the manifest uploads.
    let mut pool_sizes = Vec::new();
    for entry in &weights {
        if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped
            && (entry.name.ends_with(".mlp.experts.gate_up_proj") || entry.name.ends_with(".mlp.experts.down_proj"))
        {
            let (_, bytes) = source
                .tensor_data(&entry.name)
                .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
            pool_sizes.push(bytes.len());
        }
    }
    let capacity = gpu
        .reserve_portable_host_pool(&pool_sizes)
        .map_err(|x| format!("portable host pool ({} aggregates): {x}", pool_sizes.len()))?;
    eprintln!(
        "qwen4_igpu_engine_parity: {} host expert aggregates in one {} MiB portable pinned pool",
        pool_sizes.len(),
        capacity >> 20
    );
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if let Some(payload) =
                hipfire_arch_qwen4::expert_arena::recipe_arena_payload(&source, &receipt.manifest.identity.recipe, entry)
            {
                return payload;
            }
            // The store rejects a host-mapped entry as a range: borrow it from the
            // resident source bytes, as the product loader does.
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(|x| x.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|x| format!("qwen4 manifest fulfillment: {x}"))?;
    let dims = Dims {
        hidden: receipt.config.hidden_size,
        mi: receipt.config.moe_intermediate_size,
        experts: receipt.config.num_experts,
        k_top: receipt.config.num_experts_per_tok,
    };
    let vocab = receipt.config.vocab_size;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
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
        rows,
        receipt.ple,
        receipt.ple_rows,
        state_format,
        backend,
        None,
    )
    .map_err(|x| format!("qwen4 bundle assembly: {x}"))?;
    bundle.attach_npu_trunk_identity(npu_identity);
    if let Some(overlay) = trunk_overlay {
        bundle.attach_trunk_overlay(overlay).map_err(|x| format!("qwen4 trunk overlay: {x}"))?;
    }
    bundle
        .attach_forward(&mut gpu, rows, scratch_mode)
        .map_err(|x| format!("qwen4 forward setup: {x}"))?;
    eprintln!("qwen4_igpu_engine_parity: loaded {} on {} (vocab {vocab})", path.display(), gpu.arch);
    Ok(Model { gpu, bundle, tokenizer, dims, vocab })
}

// ── XTX F16 WMMA reference ───────────────────────────────────────────

/// Device scratch of the XTX production F16 WMMA grouped route, sized once for the
/// tile (every record of one tile has the same rows).
struct XtxRoute {
    rows: usize,
    grouped_rows: usize,
    ids: GpuTensor,
    weights: GpuTensor,
    x_norm: GpuTensor,
    target: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    y_gate_up: GpuTensor,
    y_down: GpuTensor,
    order: GpuTensor,
}

fn alloc_bytes(gpu: &mut Gpu, bytes: usize) -> R<GpuTensor> {
    gpu.alloc_tensor(&[bytes.div_ceil(4).max(1)], DType::F32).map_err(e)
}

fn upload(gpu: &Gpu, tensor: &GpuTensor, bytes: &[u8]) -> R<()> {
    if bytes.len() > tensor.buf.size() {
        return Err(format!("upload of {} bytes exceeds the {}-byte scratch", bytes.len(), tensor.buf.size()));
    }
    gpu.memcpy_htod_auto(&tensor.buf, bytes).map_err(e)
}

impl XtxRoute {
    fn new(gpu: &mut Gpu, dims: Dims, rows: usize) -> R<Self> {
        if !gpu.qwen4_moe_f16_wmma_grouped_applies(rows, dims.mi, dims.hidden) {
            return Err(format!(
                "the XTX ({}) does not take the F16 WMMA grouped route at {rows} rows (mi {}, hidden {}): no F16 reference",
                gpu.arch, dims.mi, dims.hidden
            ));
        }
        let slots = rows * dims.k_top;
        let grouped_rows = grouped_m_total_bound(slots, dims.experts).map_err(|x| format!("grouped bound: {x:?}"))?;
        Ok(Self {
            rows,
            grouped_rows,
            ids: alloc_bytes(gpu, slots * 4)?,
            weights: alloc_bytes(gpu, slots * 4)?,
            x_norm: alloc_bytes(gpu, rows * dims.hidden * 4)?,
            target: alloc_bytes(gpu, rows * dims.hidden * 4)?,
            counts: alloc_bytes(gpu, dims.experts * 4)?,
            offsets: alloc_bytes(gpu, (dims.experts + 1) * 4)?,
            sorted: alloc_bytes(gpu, grouped_rows * 4)?,
            tiles: alloc_bytes(gpu, grouped_rows * 4)?,
            inverse: alloc_bytes(gpu, slots * 4)?,
            // BF16 grouped SwiGLU activation [G, mi] and BF16 grouped down rows [G, hidden].
            y_gate_up: alloc_bytes(gpu, grouped_rows * dims.mi * 2)?,
            y_down: alloc_bytes(gpu, grouped_rows * dims.hidden * 2)?,
            order: alloc_bytes(gpu, slots * 8)?,
        })
    }

    /// The layer's combine target on the XTX's own production F16 WMMA route, from
    /// the engine call's recorded inputs; the same sequence the pcie mode runs on a
    /// gfx1100 (`qt44_qt53_prefill`: SwiGLU in the gate/up epilogue, fused unscatter +
    /// rotate128, WMMA down with BF16 rows, BF16-input combine).
    fn run(
        &self,
        gpu: &mut Gpu,
        dims: Dims,
        tables: (&GpuTensor, &GpuTensor),
        rec: &IgpuParityRecord,
    ) -> R<Vec<u8>> {
        let (rows, slots) = (self.rows, self.rows * dims.k_top);
        if rec.rows != rows || rec.hidden != dims.hidden || rec.k_top != dims.k_top {
            return Err(format!(
                "layer {}: record geometry rows {} hidden {} k_top {} != tile rows {rows} hidden {} k_top {}",
                rec.layer, rec.rows, rec.hidden, rec.k_top, dims.hidden, dims.k_top
            ));
        }
        for (what, got, want) in [
            ("x_norm", rec.x_norm.len(), rows * dims.hidden * 4),
            ("topk_ids", rec.topk_ids.len(), slots * 4),
            ("topk_weights", rec.topk_weights.len(), slots * 4),
            ("engine_target", rec.engine_target.len(), rows * dims.hidden * 4),
            ("igpu_standalone_target", rec.igpu_standalone_target.len(), rows * dims.hidden * 4),
        ] {
            if got != want {
                return Err(format!("layer {}: record {what} is {got} bytes, expected {want}", rec.layer));
            }
        }
        upload(gpu, &self.ids, &rec.topk_ids)?;
        upload(gpu, &self.weights, &rec.topk_weights)?;
        upload(gpu, &self.x_norm, &rec.x_norm)?;
        if rec.initial_zero {
            // The combine starts from +0.0; `target_in` was not read.
            upload(gpu, &self.target, &vec![0u8; rows * dims.hidden * 4])?;
        } else {
            if rec.target_in.len() != rows * dims.hidden * 4 {
                return Err(format!(
                    "layer {}: record target_in is {} bytes, expected {}",
                    rec.layer,
                    rec.target_in.len(),
                    rows * dims.hidden * 4
                ));
            }
            upload(gpu, &self.target, &rec.target_in)?;
        }
        let g = self.grouped_rows;
        gpu.moe_scatter_fused_top10(
            &self.ids,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.tiles,
            &self.inverse,
            slots,
            dims.experts,
            g,
            hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M,
        )
        .map_err(e)?;
        let x_f16 = gpu.rotate_x_mq_batched_f16(&self.x_norm, dims.hidden, rows).map_err(e)?;
        gpu.gemm_mq4g256v2_moe_grouped_top10_silu_bf16out(
            tables.0,
            &self.tiles,
            &self.sorted,
            &x_f16,
            &self.y_gate_up,
            2 * dims.mi,
            dims.hidden,
            dims.k_top,
            g,
            rows,
        )
        .map_err(e)?;
        let down_in = gpu
            .moe_unscatter_rotate128_f16(&self.y_gate_up, &self.sorted, dims.mi, g, slots)
            .map_err(e)?;
        gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
            tables.1,
            &self.tiles,
            &self.sorted,
            &down_in,
            &self.y_down,
            dims.hidden,
            dims.mi,
            g,
            true,
        )
        .map_err(e)?;
        gpu.moe_down_combine_grouped_top10_bf16in(
            &self.y_down,
            &self.inverse,
            &self.ids,
            &self.weights,
            &self.target,
            &self.order,
            dims.hidden,
            g,
            rows,
            rec.initial_zero,
            dims.experts,
        )
        .map_err(e)?;
        gpu.hip.device_synchronize().map_err(e)?;
        let mut out = gpu.download_raw_bytes(&self.target).map_err(e)?;
        out.truncate(rows * dims.hidden * 4);
        Ok(out)
    }

    fn free(self, gpu: &mut Gpu) -> R<()> {
        for t in [
            self.ids,
            self.weights,
            self.x_norm,
            self.target,
            self.counts,
            self.offsets,
            self.sorted,
            self.tiles,
            self.inverse,
            self.y_gate_up,
            self.y_down,
            self.order,
        ] {
            gpu.free_tensor(t).map_err(e)?;
        }
        Ok(())
    }
}

// ── comparisons ──────────────────────────────────────────────────────

fn f32_at(bytes: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]])
}

/// Bitwise comparison of two equal-length F32 rows-by-`hidden` buffers.
struct Bitwise {
    elems: u64,
    mismatches: u64,
    first: Option<(usize, usize, u32, u32)>,
}

fn bitwise(a: &[u8], b: &[u8], hidden: usize) -> Bitwise {
    let n = a.len() / 4;
    let mut out = Bitwise { elems: n as u64, mismatches: 0, first: None };
    for i in 0..n {
        let (x, y) = (f32_at(a, i).to_bits(), f32_at(b, i).to_bits());
        if x != y {
            out.mismatches += 1;
            out.first.get_or_insert((i / hidden, i % hidden, x, y));
        }
    }
    out
}

/// Engine target `a` against the XTX F16 WMMA target `b`, F32. `rel` is
/// `|a-b| / max(|b|, 1e-30)`; `scaled` is `max_abs / max|b|` (immune to near-zero
/// reference elements); non-finite values on either side are counted, not ranked.
struct XtxDelta {
    elems: u64,
    max_abs: f64,
    max_abs_at: (usize, usize),
    max_rel: f64,
    max_rel_at: (usize, usize),
    mean_abs: f64,
    ref_absmax: f64,
    differing: u64,
    nonfinite: u64,
}

fn xtx_delta(a: &[u8], b: &[u8], hidden: usize) -> XtxDelta {
    let n = a.len() / 4;
    let mut d = XtxDelta {
        elems: n as u64,
        max_abs: 0.0,
        max_abs_at: (0, 0),
        max_rel: 0.0,
        max_rel_at: (0, 0),
        mean_abs: 0.0,
        ref_absmax: 0.0,
        differing: 0,
        nonfinite: 0,
    };
    let mut sum = 0.0f64;
    for i in 0..n {
        let (x, y) = (f32_at(a, i), f32_at(b, i));
        if !x.is_finite() || !y.is_finite() {
            d.nonfinite += 1;
            continue;
        }
        let (x, y) = (x as f64, y as f64);
        let abs = (x - y).abs();
        d.ref_absmax = d.ref_absmax.max(y.abs());
        sum += abs;
        if x.to_bits() != y.to_bits() {
            d.differing += 1;
        }
        let at = (i / hidden, i % hidden);
        if abs > d.max_abs {
            d.max_abs = abs;
            d.max_abs_at = at;
        }
        let rel = abs / y.abs().max(1e-30);
        if rel > d.max_rel {
            d.max_rel = rel;
            d.max_rel_at = at;
        }
    }
    d.mean_abs = sum / n.max(1) as f64;
    d
}

/// Elements of the engine target that differ from its input bytes (all non-zero
/// elements under `initial_zero`): a nonzero count proves the engine wrote the target.
fn written_elems(rec: &IgpuParityRecord) -> u64 {
    let n = rec.engine_target.len() / 4;
    (0..n)
        .filter(|&i| {
            let out = f32_at(&rec.engine_target, i).to_bits();
            if rec.initial_zero {
                out != 0 && out != 0x8000_0000
            } else {
                out != f32_at(&rec.target_in, i).to_bits()
            }
        })
        .count() as u64
}

fn finite_count(bytes: &[u8]) -> u64 {
    (0..bytes.len() / 4).filter(|&i| f32_at(bytes, i).is_finite()).count() as u64
}

// ── tails / edge matrix ──────────────────────────────────────────────

/// Row counts of the matrix: every tail width the planner review names (the 513
/// case is the largest, so the recorded tile must hold at least that many rows).
const TAILS: [usize; 10] = [1, 2, 4, 8, 16, 64, 128, 511, 512, 513];

/// Input variants applied to the first `n` rows of a layer's recorded real-model
/// tile (its natural F32 `x_norm`, original top-10 ids and weights).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// Unmodified recorded rows.
    Real,
    /// Every row routes to row 0's ten experts (heavily repeated experts across rows).
    Skew,
    /// Slots 1 and 3 repeat the expert of slots 0 and 2 inside a row.
    Repeat,
    /// Invalid slots (id -1 and id >= experts) in some rows, every slot invalid in others.
    Invalid,
    /// Router weights zeroed, scaled up, scaled down and negated by row.
    Weights,
    /// Rows scaled by 1e4, 1e-30, 0 and 1e19 (the last overflows F32 downstream).
    Extreme,
}

const VARIANTS: [Variant; 6] =
    [Variant::Real, Variant::Skew, Variant::Repeat, Variant::Invalid, Variant::Weights, Variant::Extreme];

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Real => "real",
            Variant::Skew => "skew",
            Variant::Repeat => "repeat",
            Variant::Invalid => "invalid",
            Variant::Weights => "weights",
            Variant::Extreme => "extreme",
        }
    }

    /// `Extreme` may legitimately overflow; every other variant keeps finite inputs
    /// and so must stay finite.
    fn finite_required(self) -> bool {
        self != Variant::Extreme
    }
}

fn get_i32(b: &[u8], i: usize) -> i32 {
    i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}

fn put_i32(b: &mut [u8], i: usize, v: i32) {
    b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_f32(b: &mut [u8], i: usize, v: f32) {
    b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
}

/// One matrix case and the rows whose result is exactly zero by construction.
struct Built {
    case: IgpuParityCase,
    zero_rows: Vec<usize>,
}

fn build_case(rec: &IgpuParityRecord, dims: Dims, n: usize, v: Variant) -> Built {
    let (h, k, experts) = (dims.hidden, dims.k_top, dims.experts as i32);
    let mut x = rec.x_norm[..n * h * 4].to_vec();
    let mut ids = rec.topk_ids[..n * k * 4].to_vec();
    let mut w = rec.topk_weights[..n * k * 4].to_vec();
    let mut zero_rows = Vec::new();
    match v {
        Variant::Real => {}
        Variant::Skew => {
            let row0: Vec<i32> = (0..k).map(|j| get_i32(&ids, j)).collect();
            for r in 0..n {
                for (j, &id) in row0.iter().enumerate() {
                    put_i32(&mut ids, r * k + j, id);
                }
            }
        }
        Variant::Repeat => {
            for r in 0..n {
                let (a, b) = (get_i32(&ids, r * k), get_i32(&ids, r * k + 2));
                put_i32(&mut ids, r * k + 1, a);
                put_i32(&mut ids, r * k + 3, b);
            }
        }
        Variant::Invalid => {
            for r in 0..n {
                if r % 3 == 0 {
                    put_i32(&mut ids, r * k + 1, -1);
                    put_i32(&mut ids, r * k + 7, -1);
                }
                if r % 7 == 0 {
                    put_i32(&mut ids, r * k + 4, experts);
                }
            }
            // Rows with every slot invalid: nothing is routed, the target stays +0.
            if n > 1 {
                for j in 0..k {
                    put_i32(&mut ids, k + j, -1);
                }
                zero_rows.push(1);
            }
            if n > 2 {
                for j in 0..k {
                    put_i32(&mut ids, (n - 1) * k + j, experts + 5);
                }
                zero_rows.push(n - 1);
            }
        }
        Variant::Weights => {
            for r in 0..n {
                for j in 0..k {
                    let old = f32_at(&w, r * k + j);
                    let new = match r % 4 {
                        0 => 0.0,
                        1 => old * 64.0,
                        2 => old / 64.0,
                        _ => -old,
                    };
                    put_f32(&mut w, r * k + j, new);
                }
                if r % 4 == 0 {
                    zero_rows.push(r);
                }
            }
        }
        Variant::Extreme => {
            for r in 0..n {
                let scale = match r % 4 {
                    0 => 1.0e4f32,
                    1 => 1.0e-30,
                    2 => 0.0,
                    _ => 1.0e19,
                };
                for c in 0..h {
                    let old = f32_at(&x, r * h + c);
                    put_f32(&mut x, r * h + c, old * scale);
                }
            }
        }
    }
    Built {
        case: IgpuParityCase { rows: n, x_norm: x, topk_ids: ids, topk_weights: w },
        zero_rows,
    }
}

fn all_zero_row(bytes: &[u8], row: usize, hidden: usize) -> bool {
    (0..hidden).all(|c| f32_at(bytes, row * hidden + c) == 0.0)
}

fn first_text(first: Option<(usize, usize, u32, u32)>) -> String {
    first
        .map(|(r, c, a, b)| format!(" first@({r},{c}) {a:#010x} vs {b:#010x}"))
        .unwrap_or_default()
}

/// The matrix of one layer: every tail width x every variant through the engine's
/// real transport path (`igpu_parity_execute`, parity-only: production admission
/// stays at more than four rows), engine target against the standalone gfx1151
/// target of the same inputs, bitwise; the zero-by-construction rows; then, on the
/// unmodified rows, row-prefix invariance across widths and run-to-run determinism.
fn run_matrix(m: &mut Model, layer: usize, rec: &IgpuParityRecord) -> R<Vec<LayerOut>> {
    let dims = m.dims;
    let h = dims.hidden;
    let widest = TAILS[TAILS.len() - 1];
    if rec.rows < widest {
        return Err(format!("matrix layer {layer}: the recorded tile has {} rows, the matrix needs {widest}", rec.rows));
    }
    let mut outs = Vec::new();
    let mut real: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for &n in &TAILS {
        for &v in &VARIANTS {
            let built = build_case(rec, dims, n, v);
            let out = m
                .bundle
                .igpu_parity_execute(&mut m.gpu, layer, &built.case)
                .map_err(|x| format!("matrix L{layer} rows {n} {}: {x}", v.name()))?;
            let want = n * h * 4;
            if out.rows != n || out.engine_target.len() != want || out.igpu_standalone_target.len() != want {
                return Err(format!(
                    "matrix L{layer} rows {n} {}: outcome rows {} targets {}/{} bytes, expected {n} rows {want} bytes",
                    v.name(),
                    out.rows,
                    out.engine_target.len(),
                    out.igpu_standalone_target.len()
                ));
            }
            let cmp = bitwise(&out.engine_target, &out.igpu_standalone_target, h);
            let nonfinite = (n * h) as u64 - finite_count(&out.engine_target);
            let zero_ok = built.zero_rows.iter().all(|&r| all_zero_row(&out.engine_target, r, h));
            let nonzero = (0..n * h).filter(|&i| f32_at(&out.engine_target, i) != 0.0).count();
            let finite_ok = !v.finite_required() || nonfinite == 0;
            let wrote_ok = v != Variant::Real || nonzero > 0;
            let ok = cmp.mismatches == 0 && zero_ok && finite_ok && wrote_ok;
            outs.push(LayerOut {
                ok,
                line: format!(
                    "M L{layer:02} rows {n:3} {:7} {} engine==igpu {} (mismatch {}/{}{}) nonfinite {nonfinite} zero_rows {}/{} nonzero {nonzero}",
                    v.name(),
                    if ok { "ok  " } else { "FAIL" },
                    if cmp.mismatches == 0 { "BITWISE" } else { "DIFFERS" },
                    cmp.mismatches,
                    cmp.elems,
                    first_text(cmp.first),
                    if zero_ok { built.zero_rows.len() } else { 0 },
                    built.zero_rows.len()
                ),
                json: json!({
                    "kind": "case",
                    "layer": layer,
                    "rows": n,
                    "variant": v.name(),
                    "ok": ok,
                    "engine_vs_igpu_mismatch_elems": cmp.mismatches,
                    "engine_vs_igpu_first_mismatch": cmp.first.map(|(r, c, a, b)| json!({"row": r, "col": c, "engine_bits": a, "igpu_bits": b})),
                    "elems": cmp.elems,
                    "nonfinite": nonfinite,
                    "zero_rows": built.zero_rows,
                    "zero_rows_ok": zero_ok,
                    "nonzero_elems": nonzero,
                }),
            });
            if v == Variant::Real {
                real.insert(n, out.engine_target);
            }
        }
    }
    // Row-prefix invariance: the first n rows of an n-row call equal the same rows of the
    // widest call (no hidden row-size arithmetic switching).
    let full = real.get(&widest).ok_or("matrix: widest real case missing")?.clone();
    for (&n, bytes) in &real {
        let cmp = bitwise(bytes, &full[..n * h * 4], h);
        let ok = cmp.mismatches == 0;
        outs.push(LayerOut {
            ok,
            line: format!(
                "M L{layer:02} rows {n:3} prefix-invariant vs rows {widest} {} (mismatch {}/{}{})",
                if ok { "ok  BITWISE" } else { "FAIL DIFFERS" },
                cmp.mismatches,
                cmp.elems,
                first_text(cmp.first)
            ),
            json: json!({
                "kind": "prefix_invariance",
                "layer": layer,
                "rows": n,
                "against_rows": widest,
                "ok": ok,
                "mismatch_elems": cmp.mismatches,
                "elems": cmp.elems,
            }),
        });
    }
    // Run-to-run determinism of the widest real case.
    let built = build_case(rec, dims, widest, Variant::Real);
    let again = m
        .bundle
        .igpu_parity_execute(&mut m.gpu, layer, &built.case)
        .map_err(|x| format!("matrix L{layer} rows {widest} repeat: {x}"))?;
    let cmp = bitwise(&again.engine_target, &full, h);
    let ok = cmp.mismatches == 0;
    outs.push(LayerOut {
        ok,
        line: format!(
            "M L{layer:02} rows {widest:3} repeat-run {} (mismatch {}/{}{})",
            if ok { "ok  BITWISE" } else { "FAIL DIFFERS" },
            cmp.mismatches,
            cmp.elems,
            first_text(cmp.first)
        ),
        json: json!({
            "kind": "determinism",
            "layer": layer,
            "rows": widest,
            "ok": ok,
            "mismatch_elems": cmp.mismatches,
            "elems": cmp.elems,
        }),
    });
    Ok(outs)
}

// ── run ──────────────────────────────────────────────────────────────

struct LayerOut {
    ok: bool,
    line: String,
    json: Value,
}

fn real_main() -> R<bool> {
    let args = parse_args()?;
    let mode = hipfire_config::memory::offload_exec();
    if mode != hipfire_config::memory::OffloadExec::Igpu {
        return Err(format!(
            "offload_exec is '{mode}': run with HIPFIRE_OFFLOAD_EXEC=igpu (and HIPFIRE_IGPU_DEVICE=N if the gfx1151 is not auto-picked)"
        ));
    }
    let text = std::fs::read_to_string(&args.text).map_err(|x| format!("read {}: {x}", args.text.display()))?;
    let mut m = load(&args.model, args.rows, args.vram_layers)?;
    let stream = m.tokenizer.encode(&text);
    if stream.len() < args.offset + args.rows {
        return Err(format!(
            "TEXT has {} tokens, --offset {} + --rows {} needs {}",
            stream.len(),
            args.offset,
            args.rows,
            args.offset + args.rows
        ));
    }
    let tokens = &stream[args.offset..args.offset + args.rows];

    let engined = m.bundle.igpu_engined_layers();
    if engined.is_empty() {
        return Err(
            "the iGPU engine is not armed (attach_forward fell back to the normal path); see the one-line arming refusal above"
                .into(),
        );
    }
    let selected: Vec<usize> = match &args.layers {
        None => engined.clone(),
        Some(list) => {
            if let Some(bad) = list.iter().find(|l| !engined.contains(l)) {
                return Err(format!("--layers: layer {bad} is not engined (engined: {engined:?})"));
            }
            list.clone()
        }
    };
    if selected.is_empty() {
        return Err("no layer selected".into());
    }
    // Layers of the tails/edge matrix: first, middle and last selected unless named.
    let matrix_layers: Option<Vec<usize>> = if args.matrix {
        let list = match &args.matrix_layers {
            Some(list) => list.clone(),
            None => {
                let n = selected.len();
                let mut v = vec![selected[0]];
                if n > 2 {
                    v.push(selected[n / 2]);
                }
                if n > 1 {
                    v.push(selected[n - 1]);
                }
                v
            }
        };
        if let Some(bad) = list.iter().find(|l| !selected.contains(l)) {
            return Err(format!("--matrix-layers: layer {bad} is not a selected engined layer ({selected:?})"));
        }
        Some(list)
    } else {
        None
    };
    eprintln!(
        "qwen4_igpu_engine_parity: {} engined layers, {} selected, tile {} rows at token offset {}",
        engined.len(),
        selected.len(),
        args.rows,
        args.offset
    );
    m.bundle.igpu_parity_set(true, Some(selected.as_slice())).map_err(|x| format!("parity mode: {x}"))?;

    // One prefill tile of real tokens on a fresh state.
    m.bundle.reset(&mut m.gpu).map_err(e)?;
    let logits = m.gpu.zeros(&[m.vocab], DType::F32).map_err(e)?;
    m.bundle.forward_chunk_final(&mut m.gpu, tokens, &logits, None).map_err(|x| format!("tile forward: {x}"))?;
    m.gpu.hip.device_synchronize().map_err(e)?;
    let final_logits = m.gpu.download_f32(&logits).map_err(e)?;
    if final_logits.iter().any(|v| !v.is_finite()) {
        return Err("the tile's final logits are non-finite".into());
    }
    let records = m.bundle.igpu_take_parity_records();

    let mut by_layer: BTreeMap<usize, Vec<IgpuParityRecord>> = BTreeMap::new();
    for rec in records {
        by_layer.entry(rec.layer).or_default().push(rec);
    }
    let route = XtxRoute::new(&mut m.gpu, m.dims, args.rows)?;
    let mut outs: Vec<LayerOut> = Vec::new();
    for &layer in &selected {
        let recs = by_layer.get(&layer).map(Vec::as_slice).unwrap_or(&[]);
        if recs.len() != 1 {
            outs.push(LayerOut {
                ok: false,
                line: format!(
                    "L{layer:02} FAIL: {} parity records (want exactly 1 per layer: the tile must be one engine call; rows seen {:?})",
                    recs.len(),
                    recs.iter().map(|r| r.rows).collect::<Vec<_>>()
                ),
                json: json!({"layer": layer, "records": recs.len(), "ok": false}),
            });
            continue;
        }
        let rec = &recs[0];
        let hidden = rec.hidden;
        let cmp = bitwise(&rec.engine_target, &rec.igpu_standalone_target, hidden);
        let tables = m
            .bundle
            .igpu_xtx_expert_tables(layer)
            .ok_or_else(|| format!("layer {layer}: no model-device expert pointer tables"))?;
        let xtx = route.run(&mut m.gpu, m.dims, tables, rec)?;
        let d = xtx_delta(&rec.engine_target, &xtx, hidden);
        let written = written_elems(rec);
        let finite_ok = finite_count(&rec.engine_target) == cmp.elems
            && finite_count(&rec.igpu_standalone_target) == cmp.elems
            && d.nonfinite == 0;
        let ok = cmp.mismatches == 0 && finite_ok && written > 0 && rec.rows == args.rows;
        let first = cmp
            .first
            .map(|(r, c, a, b)| format!(" first@({r},{c}) engine {a:#010x} igpu {b:#010x}"))
            .unwrap_or_default();
        outs.push(LayerOut {
            ok,
            line: format!(
                "L{layer:02} {} rows {} engine==igpu {} (mismatch {}/{}{first}) written {written} | vs XTX F16 WMMA: max_abs {:.6e}@({},{}) max_rel {:.6e}@({},{}) scaled {:.6e} mean_abs {:.6e} differing {}/{} nonfinite {}",
                if ok { "ok  " } else { "FAIL" },
                rec.rows,
                if cmp.mismatches == 0 { "BITWISE" } else { "DIFFERS" },
                cmp.mismatches,
                cmp.elems,
                d.max_abs,
                d.max_abs_at.0,
                d.max_abs_at.1,
                d.max_rel,
                d.max_rel_at.0,
                d.max_rel_at.1,
                d.max_abs / d.ref_absmax.max(1e-30),
                d.mean_abs,
                d.differing,
                d.elems,
                d.nonfinite
            ),
            json: json!({
                "layer": layer,
                "ok": ok,
                "rows": rec.rows,
                "initial_zero": rec.initial_zero,
                "engine_vs_igpu_bitwise": cmp.mismatches == 0,
                "engine_vs_igpu_mismatch_elems": cmp.mismatches,
                "engine_vs_igpu_first_mismatch": cmp.first.map(|(r, c, a, b)| json!({"row": r, "col": c, "engine_bits": a, "igpu_bits": b})),
                "elems": cmp.elems,
                "written_elems": written,
                "xtx_f16_wmma": {
                    "max_abs": d.max_abs,
                    "max_abs_at": [d.max_abs_at.0, d.max_abs_at.1],
                    "max_rel": d.max_rel,
                    "max_rel_at": [d.max_rel_at.0, d.max_rel_at.1],
                    "scaled_max_abs": d.max_abs / d.ref_absmax.max(1e-30),
                    "mean_abs": d.mean_abs,
                    "ref_absmax": d.ref_absmax,
                    "differing_elems": d.differing,
                    "nonfinite": d.nonfinite,
                },
            }),
        });
    }
    route.free(&mut m.gpu)?;

    let matrix_outs = match &matrix_layers {
        Some(list) => {
            let mut all = Vec::new();
            for &layer in list {
                let rec = by_layer
                    .get(&layer)
                    .and_then(|recs| recs.first())
                    .ok_or_else(|| format!("matrix layer {layer}: no recorded tile inputs"))?;
                all.extend(run_matrix(&mut m, layer, rec)?);
            }
            all
        }
        None => Vec::new(),
    };
    m.bundle.igpu_parity_set(false, None).map_err(|x| format!("parity mode off: {x}"))?;

    for out in outs.iter().chain(matrix_outs.iter()) {
        println!("{}", out.line);
    }
    let stray: Vec<usize> = by_layer.keys().copied().filter(|l| !selected.contains(l)).collect();
    let tile_ok = !outs.is_empty() && outs.iter().all(|o| o.ok) && stray.is_empty();
    let matrix_ok = matrix_outs.iter().all(|o| o.ok);
    let all_ok = tile_ok && matrix_ok;
    let worst = |key: &str| {
        outs.iter()
            .filter_map(|o| o.json["xtx_f16_wmma"][key].as_f64())
            .fold(0.0f64, f64::max)
    };
    let (worst_abs, worst_rel, worst_scaled) = (worst("max_abs"), worst("max_rel"), worst("scaled_max_abs"));
    println!(
        "TILE {}: {} layers, {} bitwise-equal engine vs standalone gfx1151; worst vs XTX F16 WMMA: max_abs {worst_abs:.6e} max_rel {worst_rel:.6e} scaled {worst_scaled:.6e}{}",
        if tile_ok { "PASS" } else { "FAIL" },
        outs.len(),
        outs.iter().filter(|o| o.ok).count(),
        if stray.is_empty() { String::new() } else { format!("; records for unselected layers {stray:?}") }
    );
    if matrix_layers.is_some() {
        println!(
            "MATRIX {}: {} cases ({} ok) over layers {:?}, tails {TAILS:?}",
            if matrix_ok { "PASS" } else { "FAIL" },
            matrix_outs.len(),
            matrix_outs.iter().filter(|o| o.ok).count(),
            matrix_layers.as_deref().unwrap_or_default()
        );
    }
    println!("SUMMARY {}", if all_ok { "PASS" } else { "FAIL" });
    if let Some(path) = &args.json {
        let doc = json!({
            "tool": "qwen4_igpu_engine_parity",
            "model": args.model.display().to_string(),
            "text": args.text.display().to_string(),
            "arch": m.gpu.arch.to_string(),
            "rows": args.rows,
            "offset": args.offset,
            "vram_layers": args.vram_layers,
            "engined_layers": engined,
            "pass": all_ok,
            "tile_pass": tile_ok,
            "matrix_pass": matrix_ok,
            "worst_vs_xtx": {"max_abs": worst_abs, "max_rel": worst_rel, "scaled_max_abs": worst_scaled},
            "layers": outs.iter().map(|o| o.json.clone()).collect::<Vec<_>>(),
            "matrix_layers": matrix_layers,
            "matrix": matrix_outs.iter().map(|o| o.json.clone()).collect::<Vec<_>>(),
        });
        std::fs::write(path, serde_json::to_vec_pretty(&doc).map_err(e)?)
            .map_err(|x| format!("write {}: {x}", path.display()))?;
    }
    Ok(all_ok)
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(m) => {
            eprintln!("qwen4_igpu_engine_parity: {m}");
            std::process::exit(2);
        }
    }
}
