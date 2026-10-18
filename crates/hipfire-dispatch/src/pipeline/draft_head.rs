// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative-decode draft head: ranks the vocabulary for one hidden row and
//! returns the draft token.
//!
//! Drafts only steer acceptance; the target verifies every emitted token, so a
//! draft may rank a lower-bit copy of the language head.  With re-scoring the
//! copy's best 8 are re-scored exactly against the source head, which makes
//! the draft the source head's own argmax whenever the copy ranks it there.

use super::layer_ops::{execute_argmax, hip, project_weight};
use crate::families::gemv::WeightRef;
use crate::types::DispatchError;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Ranking-copy policy: an optional lower-bit copy of the language head
/// that drafts rank the vocabulary with, and whether its top 8 are
/// re-scored exactly against the source head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftHeadPolicy {
    pub copy: Option<DType>,
    pub rescore: bool,
}

impl DraftHeadPolicy {
    /// `mq2`..`mq6`, optional `r` suffix = rescore; anything else = no copy.
    pub fn parse(choice: &str) -> Self {
        let copy = match choice.trim_end_matches('r') {
            "mq6" => Some(DType::MQ6G256V2),
            "mq5" => Some(DType::MQ5G256V2),
            "mq4" => Some(DType::MQ4G256V2),
            "mq3" => Some(DType::MQ3G256V2),
            "mq2" => Some(DType::MQ2G256V2),
            _ => None,
        };
        Self {
            copy,
            rescore: choice.ends_with('r'),
        }
    }
}

/// Row layout drafts rank: `[0, front) ++ [special, vocab) ++ [front, special)`.
///
/// BPE merge order puts the frequent tokens first, so with re-scoring a draft
/// ranks only the first two ranges (`front` plus the control ids from
/// `special` on) while no recent input token fell in `[front, special)`.
#[derive(Clone, Copy, Debug)]
pub struct DraftHeadLayout {
    pub vocab: usize,
    pub hidden: usize,
    pub front: usize,
    pub special: usize,
    /// Drafts that rank the whole vocabulary after an input id in
    /// `[front, special)`.
    pub full_hold: u32,
}

/// See [`DraftHead::request_state`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftHeadRequestState {
    pub full_steps: u32,
    pub margin: f32,
}

impl Default for DraftHeadRequestState {
    fn default() -> Self {
        Self {
            full_steps: 0,
            margin: f32::INFINITY,
        }
    }
}

pub struct DraftHead {
    /// Lower-bit ranking copy, rows in the layout order (`None` = rank the
    /// source head).
    copy: Option<GpuTensor>,
    /// Top-8 re-score scratch; present only when re-scoring.
    partial: Option<GpuTensor>,
    logits: GpuTensor,
    /// Draft token id, then (with re-scoring) its f32 logit margin, the 8
    /// re-scored candidates and their exact logits.
    top1: GpuTensor,
    rotation: GpuTensor,
    /// 0 = whole vocabulary always, rows in token order.
    front: usize,
    special: usize,
    vocab: usize,
    hidden: usize,
    full_hold: u32,
    full_steps: u32,
    margin: f32,
    /// Last draft's re-scored `(token, exact logit)` candidates (re-scoring
    /// only; zeroed otherwise).
    candidates: [(u32, f32); 8],
}

/// Ranked row count and the `(front, special, tail)` order the re-score
/// kernel maps ranked rows back to token ids with.
fn ranking(
    front: usize,
    special: usize,
    vocab: usize,
    full_steps: u32,
) -> (usize, (usize, usize, usize)) {
    match (front, full_steps) {
        (0, _) => (vocab, (vocab, vocab, 0)),
        (front, 0) => (front + vocab - special, (front, special, vocab - special)),
        (front, _) => (vocab, (front, special, vocab - special)),
    }
}

/// What [`DraftHead::new`] builds for a `head_dtype` head: the ranking-copy
/// format, whether the copy's top 8 are re-scored, and the layout front.
fn plan(
    head_dtype: DType,
    layout: DraftHeadLayout,
    policy: DraftHeadPolicy,
) -> (Option<DType>, bool, usize) {
    // Only a bigger head is worth a smaller copy.
    let copy_format = policy.copy.filter(|&format| {
        matches!(
            head_dtype,
            DType::Q8_0 | DType::BF16 | DType::MQ6G256V2 | DType::MQ5G256V2
        ) && format != head_dtype
    });
    // The re-score kernel reads Q8_0 or MQ6G256V2 rows at K = 2560.
    let rescore = policy.rescore
        && copy_format.is_some()
        && matches!(head_dtype, DType::Q8_0 | DType::MQ6G256V2)
        && layout.hidden == 2560;
    let front = if rescore && layout.front < layout.special {
        layout.front
    } else {
        0
    };
    (copy_format, rescore, front)
}

impl DraftHead {
    /// `(resident, load scratch)` device bytes [`Self::new`] takes for a
    /// `head_dtype` head: what the head keeps, and the most it holds on top
    /// of that while building — the row-chunked F32 requant scratch
    /// (`Gpu::REQUANT_G256_CHUNK_ROWS` rows), which goes back to the device,
    /// not the pool, before `new` returns.
    pub fn device_bytes(
        head_dtype: DType,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Option<(usize, usize)> {
        let (resident, scratch) = Self::allocation_sizes(head_dtype, layout, policy)?;
        let total = |sizes: Vec<usize>| {
            sizes
                .into_iter()
                .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        };
        Some((total(resident)?, total(scratch)?))
    }

    /// [`Self::device_bytes`] per allocation: `(resident, load scratch)`, one
    /// entry per device allocation [`Self::new`] makes, in allocation order
    /// (the requantized ranking copy, the re-score partial scratch, the F32
    /// logits, the top-8 output, the F32 rotation row) and, for the scratch,
    /// the one F32 requant chunk. Zero-byte entries are omitted.
    pub fn allocation_sizes(
        head_dtype: DType,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Option<(Vec<usize>, Vec<usize>)> {
        let (copy_format, rescore, _) = plan(head_dtype, layout, policy);
        let mut resident = Vec::with_capacity(5);
        let mut scratch = Vec::new();
        if let Some(format) = copy_format {
            let (values, copy) = Gpu::requant_g256_bytes(layout.vocab, layout.hidden, format)?;
            resident.push(copy);
            scratch.push(values);
        }
        if rescore {
            resident.push(Gpu::TOPK8_PARTIAL_BYTES);
        }
        resident.push(layout.vocab.checked_mul(std::mem::size_of::<f32>())?);
        resident.push(Gpu::TOPK8_OUT_BYTES);
        resident.push(layout.hidden.checked_mul(std::mem::size_of::<f32>())?);
        resident.retain(|&bytes| bytes > 0);
        scratch.retain(|&bytes| bytes > 0);
        Some((resident, scratch))
    }

    pub fn new(
        gpu: &mut Gpu,
        head: &GpuTensor,
        layout: DraftHeadLayout,
        policy: DraftHeadPolicy,
    ) -> Result<Self, DispatchError> {
        let (copy_format, rescore, front) = plan(head.dtype, layout, policy);
        let mut owned: Vec<GpuTensor> = Vec::with_capacity(5);
        let allocated = (|| -> hip_bridge::HipResult<()> {
            if let Some(format) = copy_format {
                // Requantized straight into the layout order: no unordered
                // copy beside the ordered one.
                let (front, special, vocab) = (front, layout.special, layout.vocab);
                let ranges: &[(usize, usize)] = if front == 0 {
                    &[(0, vocab)]
                } else {
                    &[(0, front), (special, vocab), (front, special)]
                };
                owned.push(gpu.requant_g256_rows(head, vocab, layout.hidden, format, ranges)?);
            }
            if rescore {
                owned.push(gpu.zeros(&[Gpu::TOPK8_PARTIAL_BYTES], DType::Raw)?);
            }
            owned.push(gpu.zeros(&[layout.vocab], DType::F32)?);
            owned.push(gpu.zeros(&[Gpu::TOPK8_OUT_BYTES], DType::Raw)?);
            owned.push(gpu.zeros(&[layout.hidden], DType::F32)?);
            Ok(())
        })();
        if let Err(error) = allocated {
            for tensor in owned {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(DispatchError::Hip(error.to_string()));
        }
        let mut owned = owned.into_iter();
        let mut next = || owned.next().expect("draft head allocation count");
        let copy = copy_format.map(|_| next());
        let partial = rescore.then(&mut next);
        Ok(Self {
            copy,
            partial,
            logits: next(),
            top1: next(),
            rotation: next(),
            front,
            special: layout.special,
            vocab: layout.vocab,
            hidden: layout.hidden,
            full_hold: layout.full_hold,
            full_steps: 0,
            margin: f32::INFINITY,
            candidates: [(0, f32::NEG_INFINITY); 8],
        })
    }

    /// An input id in `[front, special)` ranks the whole vocabulary for
    /// `full_hold` drafts.
    pub fn observe(&mut self, token: u32) {
        if (self.front..self.special).contains(&(token as usize)) {
            self.full_steps = self.full_hold;
        }
    }

    /// Request-local draft policy: the full-vocabulary hold left and the
    /// last draft's margin. Ranking weights and scratch are not included.
    pub fn request_state(&self) -> DraftHeadRequestState {
        DraftHeadRequestState {
            full_steps: self.full_steps,
            margin: self.margin,
        }
    }

    pub fn set_request_state(&mut self, state: DraftHeadRequestState) {
        self.full_steps = state.full_steps;
        self.margin = state.margin;
    }

    /// The policy a freshly built head starts from.
    pub fn reset_request_state(&mut self) {
        self.set_request_state(DraftHeadRequestState::default());
    }

    /// Rank `hidden` (`[hidden]` F32) and return the draft token id.
    pub fn draft(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
    ) -> Result<u32, DispatchError> {
        let top1 = self.top1.sub_offset(0, Gpu::TOPK8_OUT_BYTES);
        self.draft_enqueue(gpu, head, hidden, &top1)?;
        let mut top = [0u8; Gpu::TOPK8_OUT_BYTES];
        let read = if self.partial.is_some() { top.len() } else { 8 };
        hip(gpu.hip.memcpy_dtoh(&mut top[..read], &top1.buf))?;
        self.draft_finish(&top[..read])
    }

    /// The device work of [`Self::draft`]: rank `hidden` (`[hidden]` F32)
    /// into `top1_out` (a Raw tensor of at least `Gpu::TOPK8_OUT_BYTES`
    /// bytes): projection, then the re-score or the argmax. Counts down the
    /// full-vocabulary hold exactly as `draft` does; the result is parsed by
    /// [`Self::draft_finish`] once the caller has downloaded `top1_out`.
    pub fn draft_enqueue(
        &mut self,
        gpu: &mut Gpu,
        head: &GpuTensor,
        hidden: &GpuTensor,
        top1_out: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let (ranked, order) = ranking(self.front, self.special, self.vocab, self.full_steps);
        self.full_steps = self.full_steps.saturating_sub(1);
        let ranking_head = self.copy.as_ref().unwrap_or(head);
        project_weight(
            gpu,
            &WeightRef {
                buf: ranking_head,
                dtype: ranking_head.dtype,
                m: ranked,
                k: self.hidden,
                row_stride: ranking_head
                    .dtype
                    .row_bytes(self.hidden)
                    .unwrap_or(self.hidden * ranking_head.dtype.size()),
                rotation: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            },
            hidden,
            &self.logits,
            1,
            Some(&self.rotation),
        )?;
        if let Some(partial) = self.partial.as_ref() {
            // An MQ6 head reads the ranking copy's FWHT-rotated input.
            let x = if head.dtype == DType::Q8_0 {
                hidden
            } else {
                &self.rotation
            };
            hip(gpu.topk8_rescore_k2560(&self.logits, ranked, order, head, x, partial, top1_out))?;
        } else {
            execute_argmax(gpu, &self.logits, top1_out, 1, ranked)?;
        }
        Ok(())
    }

    /// Parse a downloaded `top1_out` of [`Self::draft_enqueue`] (its first
    /// `Gpu::TOPK8_OUT_BYTES` bytes with re-scoring, 8 bytes without): the
    /// draft token, the margin and the re-scored candidates, as `draft`.
    pub fn draft_finish(&mut self, top: &[u8]) -> Result<u32, DispatchError> {
        let need = if self.partial.is_some() {
            Gpu::TOPK8_OUT_BYTES
        } else {
            8
        };
        if top.len() < need {
            return Err(DispatchError::Hip(format!(
                "draft result holds {} bytes, expected {need}",
                top.len()
            )));
        }
        let word = |i: usize| [top[4 * i], top[4 * i + 1], top[4 * i + 2], top[4 * i + 3]];
        let token = u32::from_ne_bytes(word(0));
        self.margin = if self.partial.is_some() {
            for (w, candidate) in self.candidates.iter_mut().enumerate() {
                *candidate = (
                    u32::from_ne_bytes(word(2 + w)),
                    f32::from_ne_bytes(word(10 + w)),
                );
            }
            f32::from_ne_bytes(word(1))
        } else {
            f32::INFINITY
        };
        if token as usize >= self.vocab {
            return Err(DispatchError::Hip(format!(
                "draft token {token} is outside vocab {}",
                self.vocab
            )));
        }
        Ok(token)
    }

    /// Exact logit margin of the last draft over its runner-up among the
    /// re-scored candidates (infinite without re-scoring).
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// The last draft's 8 re-scored candidates as `(token, exact logit)`:
    /// the support a sampled draft is drawn from. `None` without re-scoring,
    /// where [`Self::logits`] is the whole vocabulary in token order (the
    /// reordered layout is used only with re-scoring).
    pub fn rescored_candidates(&self) -> Option<&[(u32, f32); 8]> {
        self.partial.as_ref().map(|_| &self.candidates)
    }

    /// Logits of the last draft, rows in the ranking layout.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        let tensors = [
            Some(self.logits),
            Some(self.top1),
            Some(self.rotation),
            self.copy,
            self.partial,
        ];
        for tensor in tensors.into_iter().flatten() {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_copy_tier_and_rescore_suffix() {
        let parse = DraftHeadPolicy::parse;
        assert_eq!(
            parse("mq2r"),
            DraftHeadPolicy {
                copy: Some(DType::MQ2G256V2),
                rescore: true
            }
        );
        assert_eq!(
            parse("mq6"),
            DraftHeadPolicy {
                copy: Some(DType::MQ6G256V2),
                rescore: false
            }
        );
        assert_eq!(
            parse("off"),
            DraftHeadPolicy {
                copy: None,
                rescore: false
            }
        );
        assert_eq!(
            parse("mq9r"),
            DraftHeadPolicy {
                copy: None,
                rescore: true
            }
        );
    }

    #[test]
    fn ranking_orders_front_special_tail_and_holds_full_vocab() {
        let (vocab, front, special) = (1000, 600, 900);
        assert_eq!(ranking(0, special, vocab, 3), (vocab, (vocab, vocab, 0)));
        assert_eq!(
            ranking(front, special, vocab, 0),
            (front + vocab - special, (front, special, vocab - special))
        );
        assert_eq!(
            ranking(front, special, vocab, 1),
            (vocab, (front, special, vocab - special))
        );
    }
}
