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
//! The context is a thread-local installed by the VL generate body around the
//! prefill and removed afterwards. It is keyed by absolute KV slots, so chunks
//! that do not overlap the prompt's image/rope region (every decode/verify
//! forward) never see it. Paths that cannot honour it (per-token fallback,
//! captured prefill, MoE FA prep, pre-embedded bands) refuse overlapping rows
//! with an error instead of silently computing text semantics.

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
    /// Absolute KV slot of the first `<|image_pad|>` row.
    pub image_start: usize,
    /// Merged visual embeddings, `[n_visual][dim]` f32.
    pub embeds: Vec<f32>,
    /// Model width (`config.dim`).
    pub dim: usize,
    /// Rotary phases `(t, h, w)` for rows `[0, positions.len())`; rows past the
    /// end rotate at their slot. Empty = 1D slot phases everywhere.
    pub positions: Vec<[i32; 3]>,
    /// `config.mrope_section`.
    pub section: [usize; 3],
}

impl VlPrefillInputs {
    /// Build the request's inputs from the per-request mrope context (the same
    /// one the per-token path consumes), shifting every phase by `-rope_delta`
    /// so rows after the image rotate at their slot (see the module docs).
    /// `mrope == None` (1D fallback) keeps slot phases and only splices the
    /// embeddings.
    pub fn new(
        image_start: usize,
        embeds: Vec<f32>,
        dim: usize,
        mrope: Option<&MropeCtx>,
        section: [usize; 3],
    ) -> Self {
        let positions = match mrope {
            Some(mc) => {
                debug_assert_eq!(mc.base, 0, "VL batched prefill models one turn from slot 0");
                let shift = -mc.rope_delta;
                mc.positions
                    .iter()
                    .map(|p| [p[0] + shift, p[1] + shift, p[2] + shift])
                    .collect()
            }
            None => Vec::new(),
        };
        Self {
            image_start,
            embeds,
            dim,
            positions,
            section,
        }
    }

    /// Number of spliced visual rows.
    pub fn n_visual(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.embeds.len() / self.dim
        }
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

struct Active {
    inputs: VlPrefillInputs,
    n_visual: usize,
    rope_end: usize,
    /// Device copy of `inputs.positions` (`len * 3` i32, F32-sized elements).
    pos3: Option<GpuTensor>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

/// Install the request's VL prefill context (uploads the phase table once).
/// Replaces (and frees) a stale context left by an aborted request.
pub fn install(gpu: &mut Gpu, inputs: VlPrefillInputs) -> HipResult<()> {
    uninstall(gpu)?;
    if inputs.dim == 0 || inputs.embeds.len() % inputs.dim != 0 {
        return Err(HipError::new(
            0,
            "VL prefill: embedding buffer is not a whole number of rows",
        ));
    }
    let n_visual = inputs.n_visual();
    let rope_end = inputs.rope_end();
    let pos3 = if rope_end > 0 {
        let flat: Vec<i32> = inputs.positions.iter().flat_map(|p| p.iter().copied()).collect();
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
            n_visual,
            rope_end,
            pos3,
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
        let img0 = a.inputs.image_start;
        let lo = start.max(img0);
        let hi = start.saturating_add(n).min(img0 + a.n_visual);
        if lo >= hi {
            return Ok(());
        }
        let dim = a.inputs.dim;
        let src = &a.inputs.embeds[(lo - img0) * dim..(hi - img0) * dim];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        gpu.hip
            .memcpy_htod_offset(&x_batch.buf, (lo - start) * dim * 4, bytes)
    })
}

/// For a chunk `[start, start+n)` that starts below the end of the explicit
/// phase table: a device view of its `[n][3]` phases plus the mrope section.
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
        if start + n > a.inputs.positions.len() {
            return Err(HipError::new(
                0,
                &format!(
                    "VL prefill: chunk [{start}, {}) runs past the {}-row mrope table",
                    start + n,
                    a.inputs.positions.len()
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
        let img0 = a.inputs.image_start;
        let image_hit = start < img0 + a.n_visual && end > img0 && a.n_visual > 0;
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
            image_start: 2,
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
        let shifted: Vec<[i32; 3]> = hf
            .iter()
            .map(|p| [p[0] - rope_delta, p[1] - rope_delta, p[2] - rope_delta])
            .collect();
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
    }
}
