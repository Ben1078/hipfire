// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! claude: Qwen3.5-VL image prompts on the batched (chunked) prefill path.
//!
//! An image request differs from a text request in exactly two places:
//!
//! 1. the `<|image_pad|>` rows carry the vision tower's merged embeddings
//!    instead of the token-embedding row of `<|image_pad|>`;
//! 2. rotary phases are 3D `(t, h, w)` (interleaved mrope). Text rows use the
//!    same value on all three axes, which is plain 1D RoPE.
//!
//! The per-token VL body (`forward_scratch_{embed_,}mrope`) handles both but
//! runs ~40 tok/s. This module lets every `forward_prefill_batch*` chunk take
//! the same two inputs so the VL prompt runs through the text prefill (and the
//! MTP prompt fill) unchanged otherwise:
//!
//! * [`overwrite_image_rows`] copies the visual embeddings over the image rows
//!   of `pbs.x_batch` right after the token-embedding kernel;
//! * [`rope_rows`] hands `batch_chunk_full_attn_prepare` a `[n][3]` phase table
//!   for chunks that start below the end of the image, which then rotates with
//!   the batched mrope kernel (`rope_mrope_halfsplit_batched_f32`, the twin of
//!   the per-token kernel the slow path uses).
//!
//! Phases are shifted by `-rope_delta` for the whole request: RoPE depends only
//! on the phase DIFFERENCE between query and key, and a constant shift of all
//! three axes of every row leaves every `q·k` unchanged, while making every row
//! after the image — and every generated token — rotate at its physical KV
//! slot. The text decode paths (MTP verify/draft, AR, graphs) therefore need no
//! mrope awareness at all. (HF resumes decode at `slot + rope_delta`; we resume
//! at `slot` with the prompt shifted by the same amount — identical attention
//! scores up to fp32 rounding of the absolute angle.)
//!
//! Several images are several spans; the shift is the request's total
//! `-rope_delta`, so rows after the LAST image sit at their slot.
//!
//! The context is a thread-local installed by the VL generate body around the
//! whole spec turn and removed afterwards. It is keyed by absolute KV slots:
//! the device phase table covers every slot up to the KV capacity (explicit
//! phases for the prompt, slot phases after it), so any re-prefill of the
//! conversation from slot 0 — spec prefix realign, terminal repair, checkpoint
//! resume — gets the same embeddings and phases as the first prefill. Chunks
//! that start past the last image (every decode/verify forward) never see it.
//! Paths that cannot honour it (per-token fallback, captured prefill, MoE FA
//! prep, pre-embedded bands) refuse overlapping rows with an error instead of
//! silently computing text semantics.

use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::RefCell;

use super::config::{MropeCtx, Qwen35Config};
use super::weights::{DeltaNetState, LayerWeights, Qwen35Weights};
use hipfire_runtime::llama;

/// `CLAUDE_VL_VMM_GROW` (default on): the mrope per-token forwards grow the
/// VMM KV mapping like their 1D twins. `=0` restores the old (faulting) path.
pub fn claude_vl_vmm_grow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_VL_VMM_GROW").as_deref() != Ok("0"))
}

/// Whether the batched prefill can host an `n`-row VL prompt on this model:
/// every layer batch-admissible (dense full attention — the MoE FA prep has no
/// mrope arm), a KV tier with batched FA keys, and no TriAttention compaction.
/// `Err` carries the reason for the log line; the caller keeps the per-token
/// VL path then.
pub fn batched_supported(
    gpu: &Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    dn_state: &DeltaNetState,
    kv_cache: &llama::KvCache,
    n: usize,
) -> Result<(), String> {
    if weights
        .layers
        .iter()
        .any(|lw| matches!(lw, LayerWeights::FullAttnMoe(_)))
    {
        return Err("MoE full-attention layers".into());
    }
    if !(kv_cache.quant_q8
        || kv_cache.quant_asym4
        || kv_cache.quant_asym3
        || kv_cache.quant_asym2
        || kv_cache.quant_fp8)
    {
        return Err("KV tier has no batched attention".into());
    }
    if kv_cache.compact_offset != 0 {
        return Err("KV cache is compacted".into());
    }
    if !super::prefill::prefill_batch_pbs_eligible(
        weights,
        config,
        dn_state,
        n,
        gpu.arch.as_str(),
        true,
    ) {
        return Err("batched prefill not eligible for this model/config".into());
    }
    Ok(())
}

/// Inputs for one image request's batched prefill.
pub struct VlPrefillInputs {
    /// Image row runs `(first_slot, rows)`, ascending and disjoint. `embeds`
    /// holds their rows back to back in the same order.
    pub spans: Vec<(usize, usize)>,
    /// Merged visual embeddings, `[sum(rows)][dim]` f32.
    pub embeds: Vec<f32>,
    /// Model width (`config.dim`).
    pub dim: usize,
    /// Rotary phases `(t, h, w)` for rows `[0, positions.len())`, already
    /// shifted; rows past the end rotate at their slot. Empty = slot phases.
    pub positions: Vec<[i32; 3]>,
    /// `config.mrope_section`.
    pub section: [usize; 3],
}

impl VlPrefillInputs {
    /// Single image from the per-request mrope context the per-token path
    /// consumes. `mrope == None` (1D fallback) keeps slot phases and only
    /// splices the embeddings.
    pub fn new(
        image_start: usize,
        embeds: Vec<f32>,
        dim: usize,
        mrope: Option<&MropeCtx>,
        section: [usize; 3],
    ) -> Self {
        let rows = if dim == 0 { 0 } else { embeds.len() / dim };
        let positions = match mrope {
            Some(mc) => {
                debug_assert_eq!(mc.base, 0, "VL batched prefill models one turn from slot 0");
                shift_positions(&mc.positions, mc.rope_delta)
            }
            None => Vec::new(),
        };
        Self {
            spans: if rows > 0 { vec![(image_start, rows)] } else { Vec::new() },
            embeds,
            dim,
            positions,
            section,
        }
    }

    /// Number of spliced visual rows.
    pub fn n_visual(&self) -> usize {
        self.spans.iter().map(|&(_, rows)| rows).sum()
    }

    /// First slot from which every row rotates at its own slot (the shifted
    /// table is the identity there). Chunks starting at or past it take the
    /// ordinary 1D rope path.
    pub fn rope_end(&self) -> usize {
        self.positions
            .iter()
            .enumerate()
            .rposition(|(slot, p)| {
                let s = slot as i32;
                p[0] != s || p[1] != s || p[2] != s
            })
            .map(|last| last + 1)
            .unwrap_or(0)
    }
}

/// HF `get_rope_index` phases shifted by `-rope_delta` (see the module docs).
pub fn shift_positions(hf: &[[i32; 3]], rope_delta: i32) -> Vec<[i32; 3]> {
    let shift = -rope_delta;
    hf.iter()
        .map(|p| [p[0] + shift, p[1] + shift, p[2] + shift])
        .collect()
}

struct Active {
    inputs: VlPrefillInputs,
    /// First embedding row of each span in `inputs.embeds`.
    span_row0: Vec<usize>,
    rope_end: usize,
    /// Device phase table `[table_rows][3]` i32 (F32-sized elements): the
    /// explicit phases, then slot phases up to `table_rows`.
    pos3: Option<GpuTensor>,
    table_rows: usize,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

/// Install the request's VL prefill context. `table_rows` is the number of KV
/// slots any prefill of this conversation may touch (the KV capacity): the
/// device phase table covers all of them, so a replay of prompt + generated
/// tokens from slot 0 always finds its phases. Replaces (and frees) a stale
/// context left by an aborted request.
pub fn install(gpu: &mut Gpu, inputs: VlPrefillInputs, table_rows: usize) -> HipResult<()> {
    uninstall(gpu)?;
    if inputs.dim == 0 || inputs.embeds.len() != inputs.n_visual() * inputs.dim {
        return Err(HipError::new(
            0,
            "VL prefill: embedding rows do not match the image spans",
        ));
    }
    if inputs
        .spans
        .windows(2)
        .any(|w| w[0].0 + w[0].1 > w[1].0)
    {
        return Err(HipError::new(0, "VL prefill: image spans overlap or are unsorted"));
    }
    let mut span_row0 = Vec::with_capacity(inputs.spans.len());
    let mut row = 0usize;
    for &(_, rows) in &inputs.spans {
        span_row0.push(row);
        row += rows;
    }
    let rope_end = inputs.rope_end();
    let table_rows = table_rows.max(inputs.positions.len());
    let pos3 = if rope_end > 0 {
        let mut flat: Vec<i32> = Vec::with_capacity(table_rows * 3);
        for p in &inputs.positions {
            flat.extend_from_slice(p);
        }
        for slot in inputs.positions.len()..table_rows {
            let s = slot as i32;
            flat.extend_from_slice(&[s, s, s]);
        }
        let t = gpu.alloc_tensor(&[flat.len()], DType::F32)?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        if let Err(e) = gpu.hip.memcpy_htod(&t.buf, bytes) {
            let _ = gpu.free_tensor(t);
            return Err(e);
        }
        Some(t)
    } else {
        None
    };
    ACTIVE.with(|a| {
        *a.borrow_mut() = Some(Active {
            inputs,
            span_row0,
            rope_end,
            pos3,
            table_rows,
        })
    });
    Ok(())
}

/// Remove the context and free its device table. Idempotent.
pub fn uninstall(gpu: &mut Gpu) -> HipResult<()> {
    let old = ACTIVE.with(|a| a.borrow_mut().take());
    if let Some(Active { pos3: Some(t), .. }) = old {
        gpu.free_tensor(t)?;
    }
    Ok(())
}

/// Whether a VL prefill context is installed on this thread.
pub fn is_active() -> bool {
    ACTIVE.with(|a| a.borrow().is_some())
}

/// Overwrite the image rows of a chunk's `x_batch` (rows `[start, start+n)`)
/// with the visual embeddings. No-op without a context or without overlap.
/// Same transport as `MaskEmbedOverride` (ordered H2D after the embed kernel).
pub(crate) fn overwrite_image_rows(
    gpu: &mut Gpu,
    x_batch: &GpuTensor,
    start: usize,
    n: usize,
) -> HipResult<()> {
    ACTIVE.with(|a| {
        let a = a.borrow();
        let Some(a) = a.as_ref() else {
            return Ok(());
        };
        let dim = a.inputs.dim;
        let end = start.saturating_add(n);
        for (&(img0, rows), &row0) in a.inputs.spans.iter().zip(&a.span_row0) {
            let lo = start.max(img0);
            let hi = end.min(img0 + rows);
            if lo >= hi {
                continue;
            }
            let first = row0 + (lo - img0);
            let src = &a.inputs.embeds[first * dim..(first + hi - lo) * dim];
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
            gpu.hip
                .memcpy_htod_offset(&x_batch.buf, (lo - start) * dim * 4, bytes)?;
        }
        Ok(())
    })
}

/// For a chunk `[start, start+n)` that starts below the end of the explicit
/// phases: a device view of its `[n][3]` phases plus the mrope section.
/// `None` → ordinary 1D slot phases.
pub(crate) fn rope_rows(start: usize, n: usize) -> HipResult<Option<(GpuTensor, [usize; 3])>> {
    ACTIVE.with(|a| {
        let a = a.borrow();
        let Some(a) = a.as_ref() else {
            return Ok(None);
        };
        if start >= a.rope_end || n == 0 {
            return Ok(None);
        }
        let Some(t) = a.pos3.as_ref() else {
            return Ok(None);
        };
        if start + n > a.table_rows {
            return Err(HipError::new(
                0,
                &format!(
                    "VL prefill: chunk [{start}, {}) runs past the {}-row mrope table",
                    start + n,
                    a.table_rows
                ),
            ));
        }
        Ok(Some((t.sub_offset(start * 3, n * 3), a.inputs.section)))
    })
}

/// True when rows `[start, start+n)` need VL handling (image embeddings or
/// explicit phases).
pub(crate) fn overlaps(start: usize, n: usize) -> bool {
    ACTIVE.with(|a| {
        let a = a.borrow();
        let Some(a) = a.as_ref() else {
            return false;
        };
        let end = start.saturating_add(n);
        let image_hit = a
            .inputs
            .spans
            .iter()
            .any(|&(img0, rows)| rows > 0 && start < img0 + rows && end > img0);
        image_hit || (start < a.rope_end && n > 0)
    })
}

/// Error when a path that cannot splice VL inputs would process VL rows.
pub(crate) fn refuse_overlap(start: usize, n: usize, what: &str) -> HipResult<()> {
    if overlaps(start, n) {
        return Err(HipError::new(
            0,
            &format!(
                "VL prefill rows [{start}, {}) reached the {what} path, which cannot splice image embeddings / mrope phases",
                start.saturating_add(n)
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs_for(positions: Vec<[i32; 3]>) -> VlPrefillInputs {
        VlPrefillInputs {
            spans: vec![(2, 4)],
            embeds: vec![0.0; 4 * 8],
            dim: 8,
            positions,
            section: [11, 11, 10],
        }
    }

    #[test]
    fn rope_end_is_last_non_identity_row() {
        // 2 text, 4 image (2x2 merged grid), 3 text; HF phases shifted by
        // -rope_delta (= 2) so the trailing text rows are the identity.
        let hf = vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 3, 2],
            [2, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ];
        let rope_delta = 7 - 9; // max+1 - n_tokens
        let shifted = shift_positions(&hf, rope_delta);
        assert_eq!(shifted[6], [6, 6, 6]);
        assert_eq!(shifted[8], [8, 8, 8]);
        let inp = inputs_for(shifted);
        assert_eq!(inp.n_visual(), 4);
        assert_eq!(inp.rope_end(), 6);
    }

    #[test]
    fn identity_table_needs_no_rope_rows() {
        let inp = inputs_for((0..9).map(|s| [s, s, s]).collect());
        assert_eq!(inp.rope_end(), 0);
        let inp = inputs_for(Vec::new());
        assert_eq!(inp.rope_end(), 0);
    }

    #[test]
    fn new_shifts_every_axis_by_minus_rope_delta() {
        // HF get_rope_index for [text, 2x2 merged image, text]: the image
        // spans 2 phases for 4 tokens, so rope_delta = max+1-n = 4-6 = -2.
        let hf = vec![[0, 0, 0], [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [3, 3, 3]];
        let mc = MropeCtx {
            base: 0,
            positions: hf,
            rope_delta: -2,
            section: [11, 11, 10],
        };
        let inp = VlPrefillInputs::new(1, vec![0.0; 4 * 8], 8, Some(&mc), [11, 11, 10]);
        assert_eq!(
            inp.positions,
            vec![[2, 2, 2], [3, 3, 3], [3, 3, 4], [3, 4, 3], [3, 4, 4], [5, 5, 5]]
        );
        // The trailing text row (and every decode token) sits at its slot.
        assert_eq!(inp.rope_end(), 5);
        assert_eq!(inp.n_visual(), 4);
        assert_eq!(inp.spans, vec![(1, 4)]);
    }

    #[test]
    fn two_images_shift_by_the_total_delta() {
        // [t, img 2x2, t, img 2x2, t]: each image spans 2 phases for 4 rows.
        use hipfire_arch_qwen35_vl_free_mrope as m;
        let (hf, delta) = m::positions(11, &[(1, 4, 4, 4), (6, 4, 4, 4)], 2);
        assert_eq!(delta, -4);
        let shifted = shift_positions(&hf, delta);
        // Rows after the last image are the identity; rows between are not.
        assert_eq!(shifted[10], [10, 10, 10]);
        assert_ne!(shifted[5], [5, 5, 5]);
        let inp = VlPrefillInputs {
            spans: vec![(1, 4), (6, 4)],
            embeds: vec![0.0; 8 * 8],
            dim: 8,
            positions: shifted,
            section: [11, 11, 10],
        };
        assert_eq!(inp.n_visual(), 8);
        assert_eq!(inp.rope_end(), 10);
    }

    /// Minimal copy of `hipfire_arch_qwen35_vl::mrope::build_mrope_positions`
    /// (that crate depends on this one) for the multi-span test above.
    mod hipfire_arch_qwen35_vl_free_mrope {
        pub fn positions(
            n: usize,
            spans: &[(usize, usize, usize, usize)],
            merge: usize,
        ) -> (Vec<[i32; 3]>, i32) {
            let mut out = Vec::new();
            let mut cursor = 0i32;
            let mut tok = 0usize;
            for &(start, _len, gh, gw) in spans {
                while tok < start {
                    out.push([cursor; 3]);
                    cursor += 1;
                    tok += 1;
                }
                let (lh, lw) = (gh / merge, gw / merge);
                for hh in 0..lh {
                    for ww in 0..lw {
                        out.push([cursor, cursor + hh as i32, cursor + ww as i32]);
                    }
                }
                tok += lh * lw;
                cursor += (gh.max(gw) / merge) as i32;
            }
            while tok < n {
                out.push([cursor; 3]);
                cursor += 1;
                tok += 1;
            }
            let max = out.iter().flat_map(|p| p.iter().copied()).max().unwrap_or(0);
            (out, max + 1 - n as i32)
        }
    }
}
