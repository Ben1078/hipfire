// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-layer expert slab arenas (subq4 K4 loader).
//!
//! A version-1 expert-slab artifact stores the 512 experts of a trunk layer
//! back to back (`gate_up`, `down` per expert, 256 B aligned), so a layer's
//! experts are one contiguous byte range of the file. The loader moves that
//! range to the device (or to pinned host memory) as ONE tensor, the *arena*,
//! and addresses every expert by its offset inside it. This module is the pure
//! plan: which bytes each layer's arena covers and where each expert's two
//! payloads sit in it. Nothing here touches a GPU.
//!
//! The plan is part of the recipe identity ([`crate::weights::Qwen4TierRecipe::FlashNextSlab`]),
//! so the arena extents the manifest declares are exactly the bytes the table
//! validated.

use crate::expert_slab_table::ExpertSlabTable;
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};

/// Suffix of the synthetic per-layer arena entry name
/// (`<prefix>.layers.{L}.mlp.experts.slab_arena`).
pub const ARENA_SUFFIX: &str = hipfire_runtime::weight_store::SLAB_ARENA_SUFFIX;

/// Zeroed bytes every arena allocation (device or host-mapped) carries past its
/// last entry; the resident upload allocates and zeroes them
/// (`hipfire_runtime::weight_store`), the host-mapped upload's own tail is
/// larger.
pub const ARENA_TAIL_PAD_BYTES: usize = hipfire_runtime::weight_store::SLAB_ARENA_TAIL_PAD_BYTES;

/// Codec id of an MQ4 symmetric (qt44/qt53) expert in the per-layer codec table.
pub const CODEC_MQ4: u8 = 0;
/// Codec id of an MQ3-I4 (qt131/qt132) expert in the per-layer codec table.
pub const CODEC_W3: u8 = 1;

/// One expert inside its layer's arena. Offsets are relative to the arena base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaExpert {
    pub format: ExpertFormat,
    pub gate_up_offset: usize,
    pub down_offset: usize,
}

impl ArenaExpert {
    /// The per-layer codec table byte of this expert's format.
    pub fn codec(self) -> Result<u8, ExpertArenaError> {
        match self.format {
            ExpertFormat::Mq4Xts => Ok(CODEC_MQ4),
            ExpertFormat::Mq3I4 => Ok(CODEC_W3),
            ExpertFormat::Mq2I4 => Err(ExpertArenaError::UnsupportedFormat(ExpertFormat::Mq2I4)),
        }
    }
}

/// One trunk layer's arena: the absolute file range and its 512 experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertArenaLayer {
    /// Absolute `data_offset` of the arena's first byte in the artifact.
    pub file_offset: usize,
    /// Physical arena bytes: exactly the sum of the experts' payload bytes.
    pub bytes: usize,
    pub experts: Vec<ArenaExpert>,
}

impl ExpertArenaLayer {
    /// Per-expert codec bytes (`0` = MQ4, `1` = W3). Fails closed on a W2 expert.
    pub fn codec_table(&self) -> Result<Vec<u8>, ExpertArenaError> {
        self.experts.iter().map(|expert| expert.codec()).collect()
    }

    /// `(mq4, w3)` expert counts.
    pub fn codec_counts(&self) -> (usize, usize) {
        let w3 = self
            .experts
            .iter()
            .filter(|expert| expert.gate_up_offset != expert.down_offset && expert.format == ExpertFormat::Mq3I4)
            .count();
        (self.experts.iter().filter(|expert| expert.gate_up_offset != expert.down_offset).count() - w3, w3)
    }
}

/// The 48 trunk layer arenas of one artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertArenaPlan {
    layers: Vec<ExpertArenaLayer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertArenaError {
    /// Only MQ4 symmetric and MQ3-I4 experts have kernels.
    UnsupportedFormat(ExpertFormat),
    /// The span table is not contiguous at `(layer, expert)`.
    Noncontiguous { layer: usize, expert: usize },
    /// A layer arena is larger than the 32-bit offsets of the device tables.
    TooLarge { layer: usize, bytes: usize },
}

impl std::fmt::Display for ExpertArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(format) => write!(
                f,
                "expert format {format:?} has no gfx1201 kernels; only MQ4-XTS (44,53) and MQ3-I4 (131,132) experts are admitted"
            ),
            Self::Noncontiguous { layer, expert } => write!(
                f,
                "layer {layer} expert {expert} is not contiguous with the previous payload"
            ),
            Self::TooLarge { layer, bytes } => {
                write!(f, "layer {layer} arena of {bytes} bytes exceeds 4 GiB")
            }
        }
    }
}

impl std::error::Error for ExpertArenaError {}

impl ExpertArenaPlan {
    /// Derive the arenas from a validated span table. The table's builder
    /// guarantees back-to-back slabs; this re-checks it per layer because the
    /// arena is one read of `[first.gate_up.offset, last.down.end)`.
    pub fn from_table(table: &ExpertSlabTable) -> Result<Self, ExpertArenaError> {
        let mut layers = Vec::with_capacity(TRUNK_LAYERS as usize);
        for layer in 0..TRUNK_LAYERS as usize {
            let first = table
                .slab(layer, 0)
                .ok_or(ExpertArenaError::Noncontiguous { layer, expert: 0 })?;
            let base = first.gate_up.offset;
            let mut cursor = base;
            let mut experts = Vec::with_capacity(EXPERTS_PER_LAYER as usize);
            for expert in 0..EXPERTS_PER_LAYER as usize {
                let slab = table
                    .slab(layer, expert)
                    .ok_or(ExpertArenaError::Noncontiguous { layer, expert })?;
                if slab.gate_up.offset != cursor
                    || slab.gate_up.end() != Some(slab.down.offset)
                {
                    return Err(ExpertArenaError::Noncontiguous { layer, expert });
                }
                experts.push(ArenaExpert {
                    format: slab.format,
                    gate_up_offset: slab.gate_up.offset - base,
                    down_offset: slab.down.offset - base,
                });
                cursor = slab
                    .down
                    .end()
                    .ok_or(ExpertArenaError::Noncontiguous { layer, expert })?;
            }
            let bytes = cursor - base;
            if bytes > u32::MAX as usize {
                return Err(ExpertArenaError::TooLarge { layer, bytes });
            }
            layers.push(ExpertArenaLayer {
                file_offset: base,
                bytes,
                experts,
            });
        }
        Ok(Self { layers })
    }

    pub fn layers(&self) -> &[ExpertArenaLayer] {
        &self.layers
    }

    pub fn layer(&self, layer: usize) -> Option<&ExpertArenaLayer> {
        self.layers.get(layer)
    }

    /// Total physical arena bytes (the routed-expert bytes admission must use).
    pub fn physical_bytes(&self) -> u64 {
        self.layers.iter().map(|layer| layer.bytes as u64).sum()
    }

    /// Synthetic manifest/entry name of `layer`'s arena under `layer_prefix`
    /// (`model.language_model.layers.{L}`).
    pub fn arena_name(layer_prefix: &str) -> String {
        format!("{layer_prefix}{ARENA_SUFFIX}")
    }

    /// Refuse anything the gfx1201 sealed route cannot execute.
    pub fn check_executable(&self) -> Result<(), ExpertArenaError> {
        for layer in &self.layers {
            layer.codec_table()?;
        }
        Ok(())
    }
}

/// [`arena_payload`] under a manifest's recipe: `None` unless the recipe is
/// [`crate::weights::Qwen4TierRecipe::FlashNextSlab`] and `entry` is one of its
/// layer arenas. The one call every fulfillment callback of a slab-capable
/// loader makes before its ordinary tensor read.
pub fn recipe_arena_payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    recipe: &crate::weights::Qwen4TierRecipe,
    entry: &hipfire_runtime::weight_manifest::WeightEntry,
) -> Option<Result<hipfire_runtime::model_source::SourcePayload<'a>, String>> {
    match recipe {
        crate::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } => {
            arena_payload(source, arenas, entry)
        }
        _ => None,
    }
}

/// The fulfillment payload of a manifest entry that is a layer slab arena, or
/// `None` when `entry` is an ordinary index tensor (the caller then reads it by
/// name as before).
///
/// Resident arenas stream through a checked source range (no host copy). A
/// host-mapped arena needs the bytes in hand, so it is served as an owned copy
/// of the mapped span; the mapping must therefore still be alive.
pub fn arena_payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    plan: &ExpertArenaPlan,
    entry: &hipfire_runtime::weight_manifest::WeightEntry,
) -> Option<Result<hipfire_runtime::model_source::SourcePayload<'a>, String>> {
    use hipfire_runtime::model_source::{SourcePayload, TensorInfo};
    use hipfire_runtime::weight_manifest::WeightResidency;
    if !entry.name.ends_with(ARENA_SUFFIX) {
        return None;
    }
    let result = (|| {
        let layer = entry
            .layer
            .and_then(|layer| plan.layer(layer))
            .ok_or_else(|| format!("slab arena entry '{}' names no trunk layer", entry.name))?;
        if entry.logical_shape.as_slice() != [layer.bytes] {
            return Err(format!(
                "slab arena entry '{}' declares {:?}, plan has {} bytes",
                entry.name, entry.logical_shape, layer.bytes
            ));
        }
        if entry.residency == WeightResidency::HostMapped {
            let bytes = source
                .span_bytes(layer.file_offset, layer.bytes)
                .ok_or_else(|| {
                    format!(
                        "slab arena '{}': the container mapping is unavailable",
                        entry.name
                    )
                })?
                .to_vec();
            return Ok(SourcePayload::Owned {
                info: TensorInfo {
                    name: entry.name.clone(),
                    dtype: "RAW".to_string(),
                    shape: vec![layer.bytes],
                    quant_type: 0xFF,
                    data_offset: layer.file_offset,
                    data_size: layer.bytes,
                },
                bytes,
            });
        }
        source
            .span_range(layer.file_offset, layer.bytes, "RAW", vec![layer.bytes])
            .map(SourcePayload::Range)
            .map_err(|error| error.to_string())
    })();
    Some(result)
}

/// Highest byte (exclusive) the MQ3-I4 NT1 kernels read relative to the base of
/// one `[m, k]` W3 tensor (`[codes: m*k*3/8][scales: m*(k/128) f16]`,
/// `qwen4_moe_sym_w3.rs`): the maximum over the K loop's loads, issued for the
/// last row of the tensor and the worst case of the down entry's odd-epoch tail
/// (the tail's extra `load_set(1)` is counted whether or not the branch takes
/// it).
///
/// A K128 epoch is a 48-byte code group per row; each lane loads one unaligned
/// dword per 8-code chunk at byte `48p + 12i + 6hi (+3)` of the epoch pair at
/// trip `c` (`96c` bytes on, `p` the epoch parity, `i < 4`, `hi < 2`), and one
/// dword scale pair at `4c` bytes into the row's scale run, for `c` up to the
/// trip count `(k/128 - 1) / 2` (the final `load_set` runs after the last
/// advance).
pub fn w3_max_read_end(m: usize, k: usize) -> usize {
    let row_bytes = k * 3 / 8;
    let scale_row_bytes = k / 64;
    let scale_plane = m * row_bytes;
    let last_trip = (k / 128 - 1) / 2;
    let last_row = m - 1;
    let code_end = last_row * row_bytes + 96 * last_trip + 48 + 12 * 3 + 6 + 3 + 4;
    let scale_end = scale_plane + last_row * scale_row_bytes + 4 * last_trip + 4;
    code_end.max(scale_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_slabs::ExpertFormat;

    /// A 512-expert layer of `formats` (cycled), contiguous from `base`.
    fn layer(base: usize, formats: &[ExpertFormat]) -> ExpertArenaLayer {
        let mut cursor = 0usize;
        let experts = (0..EXPERTS_PER_LAYER as usize)
            .map(|expert| {
                let format = formats[expert % formats.len()];
                let entry = ArenaExpert {
                    format,
                    gate_up_offset: cursor,
                    down_offset: cursor + format.gate_up_bytes(),
                };
                cursor += format.slab_bytes();
                entry
            })
            .collect();
        ExpertArenaLayer {
            file_offset: base,
            bytes: cursor,
            experts,
        }
    }

    #[test]
    fn codec_table_is_per_expert_and_refuses_w2() {
        let mixed = layer(0, &[ExpertFormat::Mq4Xts, ExpertFormat::Mq3I4, ExpertFormat::Mq3I4]);
        let table = mixed.codec_table().unwrap();
        assert_eq!(table.len(), 512);
        assert_eq!(&table[..4], &[CODEC_MQ4, CODEC_W3, CODEC_W3, CODEC_MQ4]);
        let w2 = layer(0, &[ExpertFormat::Mq3I4, ExpertFormat::Mq2I4]);
        assert_eq!(
            w2.codec_table(),
            Err(ExpertArenaError::UnsupportedFormat(ExpertFormat::Mq2I4))
        );
    }

    #[test]
    fn w3_read_model_overreads_the_scale_plane_of_a_k640_down_by_two_bytes() {
        // Down [2560, 640]: 5 epochs, trips (5-1)/2 = 2, scale run 10 B per row,
        // the final scale dword at byte 8 of the last row.
        let size = ExpertFormat::Mq3I4.down_bytes();
        assert_eq!(size, 640_000);
        assert_eq!(w3_max_read_end(2560, 640), size + 2);
        // Gate/up K2560 (20 epochs, 40 B scale rows): its last scale dword
        // (bytes 36..40) stays inside the tensor.
        assert!(w3_max_read_end(1280, 2560) <= ExpertFormat::Mq3I4.gate_up_bytes() + 1);
    }

    #[test]
    fn arena_tail_covers_the_last_experts_down_overread() {
        assert!(ARENA_TAIL_PAD_BYTES >= 256);
        // The last expert of the arena is W3 (a hot MQ4 expert elsewhere), and
        // the whole layer is W3: both put a W3 down tensor at the arena end.
        for formats in [
            vec![ExpertFormat::Mq3I4],
            vec![ExpertFormat::Mq4Xts, ExpertFormat::Mq3I4],
            vec![ExpertFormat::Mq3I4, ExpertFormat::Mq4Xts, ExpertFormat::Mq4Xts, ExpertFormat::Mq3I4],
        ] {
            let layer = layer(0, &formats);
            let last = layer.experts.last().unwrap();
            if last.format != ExpertFormat::Mq3I4 {
                continue;
            }
            let readable = layer.bytes + ARENA_TAIL_PAD_BYTES - last.down_offset;
            let [m, k] = [2560usize, 640];
            assert!(
                readable >= w3_max_read_end(m, k),
                "arena end + pad leaves {readable} B of the last down tensor, kernel reads {}",
                w3_max_read_end(m, k)
            );
            // Without the pad the last row's scale dword overreads.
            assert!(layer.bytes - last.down_offset < w3_max_read_end(m, k));
        }
    }

    #[test]
    fn arenas_of_different_sizes_keep_their_own_extents() {
        let small = layer(0, &[ExpertFormat::Mq3I4]);
        let large = layer(small.bytes, &[ExpertFormat::Mq4Xts]);
        assert_ne!(small.bytes, large.bytes);
        assert_eq!(small.codec_counts(), (0, 512));
        assert_eq!(large.codec_counts(), (512, 0));
    }
}
