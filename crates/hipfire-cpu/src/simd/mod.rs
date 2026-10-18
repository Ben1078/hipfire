// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! SIMD kernels for the CPU matmul core.
//!
//! `std::simd` is nightly-only (`portable_simd`) and this crate targets stable,
//! so the vector kernels are `core::arch` intrinsics behind `#[target_feature]`
//! with runtime detection, plus the plain scalar path as both the fallback and
//! the reference the vector path is tested against.
//!
//! The vector path is a *throughput* choice, not a numerical one: both compute
//! ordinary `f32` sequences and are compared within a relative tolerance (see
//! `simd::tests`), never bit-for-bit — the offload feature's contract is
//! llama.cpp-level coherence, not device parity, and the vector path is what
//! makes the CPU side competitive with reading the same bytes over PCIe.
//!
//! Parallelism stays in [`crate::gemv`] (rayon over output rows); the kernels
//! here compute a single row so the two compose without either owning the other.
//!
//! Every format the crate decodes has a kernel ([`x86`]), so [`row_dot_enabled`]
//! is a pure CPU-feature question — see `docs/quant-formats/cpu-simd-coverage.md`
//! for the per-format map.

#[cfg(target_arch = "x86_64")]
mod x86;

use crate::quant::CpuQuant;

/// Whether the CPU reports AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_available() -> bool {
    false
}

/// Whether the CPU additionally reports F16C, which the kernels over an `fp16`
/// header, `fp16` weights or an `fp16` codebook need to widen those values to
/// `f32`.
///
/// Every AVX2 part in practice has F16C, but "in practice" is not a hardware
/// guarantee and the conversion is the only `fp16` step in those kernels, so it
/// is detected rather than assumed.
#[cfg(target_arch = "x86_64")]
pub fn avx2_f16c_available() -> bool {
    avx2_available() && std::arch::is_x86_feature_detected!("f16c")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn avx2_f16c_available() -> bool {
    false
}

/// Whether `q`'s vector kernel widens `fp16` metadata, weights or codebook
/// values with `vcvtph2ps`, i.e. needs F16C on top of AVX2 + FMA (see
/// [`avx2_f16c_available`]).
///
/// The match is deliberately exhaustive — the lists below name every
/// [`CpuQuant`] — so a new variant does not compile until its kernel's feature
/// requirement is stated here, next to the kernel arm in [`row_dot_avx2`].
fn needs_f16c(q: CpuQuant) -> bool {
    match q {
        // Kernels that widen fp16 metadata with `vcvtph2ps`.
        CpuQuant::Mq4G256V2
        | CpuQuant::Mq4G128V2
        | CpuQuant::Mq6G256V2
        | CpuQuant::Mq5G256V2
        | CpuQuant::Mq2G256V2
        | CpuQuant::Mq3G256V2
        | CpuQuant::Mq4CG256
        | CpuQuant::Tq2G128
        | CpuQuant::Bq1G128
        | CpuQuant::Mq2G256Lloyd
        | CpuQuant::Mq2G256LloydU
        | CpuQuant::Mq3G256Lloyd
        | CpuQuant::Mq4G256Lloyd
        | CpuQuant::Q8F16
        | CpuQuant::F16
        | CpuQuant::Mfp4G32E8
        // Planar slab formats widen their f16 scale plane with `vcvtph2ps`.
        | CpuQuant::Mq3I4G128R256
        | CpuQuant::Mq3I4G128R128
        | CpuQuant::E8pI4G128R256
        | CpuQuant::E8pI4G128R128 => true,
        // Kernels over a plain f32 header, or over weights that need no fp16
        // step at all.
        CpuQuant::Mq4G256
        | CpuQuant::Hfq4G256
        | CpuQuant::Hfq6G256
        | CpuQuant::Mq6G256
        | CpuQuant::Mq5G256
        | CpuQuant::Hfq3G256
        | CpuQuant::Mq3G256
        | CpuQuant::Hfq2G256
        | CpuQuant::Mq2G256
        | CpuQuant::Hfq4G128
        | CpuQuant::Hfq3G128
        | CpuQuant::Hfq2G128
        | CpuQuant::F32
        | CpuQuant::Bf16 => false,
    }
}

/// Whether `q` has a vector kernel on this CPU: AVX2 + FMA, plus F16C for the
/// kernels [`needs_f16c`] names.
fn features_for(q: CpuQuant) -> bool {
    if needs_f16c(q) {
        avx2_f16c_available()
    } else {
        avx2_available()
    }
}

/// The row-dot kernel an unforced GEMV runs for `q` on this CPU, named by the
/// instruction sets it uses: `"avx2-fma-f16c"` or `"avx2-fma"` for the vector
/// kernel, `"scalar"` for the fallback. Load logs print it so every run states
/// which CPU arithmetic path its host-executed experts took.
pub fn row_dot_kernel(q: CpuQuant) -> &'static str {
    match (row_dot_enabled(q, None), needs_f16c(q)) {
        (false, _) => "scalar",
        (true, true) => "avx2-fma-f16c",
        (true, false) => "avx2-fma",
    }
}

/// Whether the vector row dot can run for format `q` on this CPU, honouring a
/// forced `requested` (see [`use_avx2`]).
///
/// Resolved once per GEMV call rather than per row (`m` is thousands on every
/// real shape).
pub fn row_dot_enabled(q: CpuQuant, requested: Option<bool>) -> bool {
    use_avx2(features_for(q), requested)
}

/// Pure dispatch predicate. `requested` forces the decision (`Some(true)` /
/// `Some(false)`, for tests and for the SIMD-vs-scalar comparison), `None`
/// follows `available`.
///
/// A forced `Some(true)` on hardware without AVX2 would execute undefined
/// instructions, so the force is honoured only when `available` says the feature
/// is present: a forced request on unsupported hardware falls back to scalar
/// rather than faulting, which is also what lets the predicate be unit-tested on
/// any runner.
pub fn use_avx2(available: bool, requested: Option<bool>) -> bool {
    match requested {
        Some(true) => available,
        Some(false) => false,
        None => available,
    }
}

/// Which cached-`Σx` reduction style a format's row dot can reuse.
///
/// The styles are numerically distinct reductions of the same sums, so a
/// prepared `Σx` is only valid for the format that produced it — never reused
/// across styles. See [`prepare_activation_sums`] and
/// [`row_dot_with_activation_sums`].
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ActivationSumKind {
    /// qt 13 `Mq4G256`: [`x86`]'s hand-unrolled nibble path, `(x0+x1)+(x2+x3)`
    /// per 32-wide block.
    Mq4Quartets,
    /// qt 15 `Mq6G256`: even/odd 8-lane chunk chains.
    Mq6Pairs,
    /// qt 135 / 136 E8P-I4: one sum per 8-element word (`k / 8` per row), each
    /// in `chunk_sums8` order — the fixed `hadd`/permute reduction of eight
    /// words at a time, shared with the unprepared E8P row dot.
    E8pWords,
}

#[cfg(target_arch = "x86_64")]
impl ActivationSumKind {
    /// How many `f32` sums one activation row of length `k` prepares: one per
    /// 256-element group for the MQ kinds, one per 8-element word for
    /// [`Self::E8pWords`].
    pub(crate) fn len(self, k: usize) -> usize {
        match self {
            ActivationSumKind::Mq4Quartets | ActivationSumKind::Mq6Pairs => k / 256,
            ActivationSumKind::E8pWords => k / 8,
        }
    }
}

/// The cached-`Σx` style `q` uses, or `None` when `q` has no prepared row dot.
///
/// MQ4, MQ6 and both E8P formats reuse prepared sums. The other codebook formats
/// have no affine zero term and do not need `Σx`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn activation_sum_kind(q: CpuQuant) -> Option<ActivationSumKind> {
    match q {
        CpuQuant::Mq4G256 => Some(ActivationSumKind::Mq4Quartets),
        CpuQuant::Mq6G256 => Some(ActivationSumKind::Mq6Pairs),
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => Some(ActivationSumKind::E8pWords),
        _ => None,
    }
}

/// Precompute the `Σx` sums for `kind` into `out` — `out.len() ==
/// kind.len(k)` ([`ActivationSumKind::len`]): one `f32` per 256 activation
/// elements for the MQ kinds, one per 8-element word for E8P.
///
/// Each style reproduces its row dot's original reduction order exactly, so a
/// [`row_dot_with_activation_sums`] row is bit-identical to the unprepared
/// [`row_dot_avx2`] row. The caller resolves the feature decision once per job
/// ([`row_dot_enabled`]) before calling either prepared entry point.
#[cfg(target_arch = "x86_64")]
pub(crate) fn prepare_activation_sums(kind: ActivationSumKind, x: &[f32], out: &mut [f32]) {
    debug_assert!(
        x.len()
            >= out.len()
                * match kind {
                    ActivationSumKind::Mq4Quartets | ActivationSumKind::Mq6Pairs => 256,
                    ActivationSumKind::E8pWords => 8,
                },
        "prepare_activation_sums: activation shorter than the sums it covers"
    );
    unsafe {
        match kind {
            ActivationSumKind::Mq4Quartets => {
                x86::activation_sums_mq4(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
            ActivationSumKind::Mq6Pairs => {
                x86::activation_sums_mq6(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
            ActivationSumKind::E8pWords => {
                x86::activation_sums_e8p(x.as_ptr(), out.as_mut_ptr(), out.len())
            }
        }
    }
}

/// One weight row dotted with `x`, reusing the prepared `Σx` in `sums`
/// ([`prepare_activation_sums`]) instead of recomputing it.
///
/// Only the formats [`activation_sum_kind`] names have cached sums, and the
/// caller has already resolved the feature decision ([`row_dot_enabled`]), so
/// `q` is AVX2-enabled and `sums.len() >= kind.len(k)`. The dot FMAs, group
/// accumulation and final scaling are the same as [`row_dot_avx2`].
///
/// `scales_row` is the row's f16 scale bytes for the planar formats (as in
/// [`row_dot_avx2`]) and empty for the row formats.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_with_activation_sums(
    q: CpuQuant,
    row: &[u8],
    scales_row: &[u8],
    k: usize,
    x: &[f32],
    sums: &[f32],
) -> f32 {
    debug_assert!(
        activation_sum_kind(q).is_some_and(|kind| sums.len() >= kind.len(k)),
        "row_dot_with_activation_sums({q:?}): sums shorter than the shape"
    );
    debug_assert!(
        !q.is_planar()
            || (row.len() >= crate::gemv::row_bytes(q, k)
                && scales_row.len() >= 2 * (k / crate::quant::PLANAR_GROUP)),
        "row_dot_with_activation_sums({q:?}): planar row or scales shorter than the shape"
    );
    debug_assert!(x.len() >= k, "row_dot_with_activation_sums({q:?}): short x");
    unsafe {
        match q {
            CpuQuant::Mq4G256 => {
                x86::mq4g256_row_dot_cached(row.as_ptr(), k, x.as_ptr(), sums.as_ptr())
            }
            CpuQuant::Mq6G256 => {
                x86::mq6g256_row_dot_cached(row.as_ptr(), k, x.as_ptr(), sums.as_ptr())
            }
            CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => x86::e8p_row_dot_cached(
                row.as_ptr(),
                scales_row.as_ptr(),
                k,
                x.as_ptr(),
                sums.as_ptr(),
            ),
            _ => unreachable!("row_dot_with_activation_sums: {q:?} has no cached Σx"),
        }
    }
}

/// Whether `q` has a two-row prepared kernel
/// ([`row_pair_dot_with_activation_sums`]): the E8P formats, whose row dot is
/// bound by the table decode and shares each activation load across two rows.
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_row_pair_kernel(q: CpuQuant) -> bool {
    match q {
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => true,
        CpuQuant::Mq4G256
        | CpuQuant::Mq6G256
        | CpuQuant::Mq3I4G128R256
        | CpuQuant::Mq3I4G128R128
        | CpuQuant::Mq4G256V2
        | CpuQuant::Mq4G128V2
        | CpuQuant::Hfq4G256
        | CpuQuant::Hfq6G256
        | CpuQuant::Mq5G256
        | CpuQuant::Hfq3G256
        | CpuQuant::Mq3G256
        | CpuQuant::Hfq2G256
        | CpuQuant::Mq2G256
        | CpuQuant::Hfq4G128
        | CpuQuant::Hfq3G128
        | CpuQuant::Hfq2G128
        | CpuQuant::Mq4CG256
        | CpuQuant::Tq2G128
        | CpuQuant::Bq1G128
        | CpuQuant::Mfp4G32E8
        | CpuQuant::Mq6G256V2
        | CpuQuant::Mq5G256V2
        | CpuQuant::Mq3G256V2
        | CpuQuant::Mq2G256V2
        | CpuQuant::Mq2G256Lloyd
        | CpuQuant::Mq2G256LloydU
        | CpuQuant::Mq3G256Lloyd
        | CpuQuant::Mq4G256Lloyd
        | CpuQuant::F16
        | CpuQuant::Bf16
        | CpuQuant::F32
        | CpuQuant::Q8F16 => false,
    }
}

/// Two weight rows dotted with the same `x` in one pass, reusing the prepared
/// `Σx` in `sums`. Each row's result is bit-identical to
/// [`row_dot_with_activation_sums`] on that row.
///
/// Only formats with [`has_row_pair_kernel`] are legal; the caller has resolved
/// the feature decision ([`row_dot_enabled`]) and `sums.len() >= kind.len(k)`.
/// `srow0` / `srow1` are the rows' scale bytes ([`row_dot_avx2`]'s
/// `scales_row`).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn row_pair_dot_with_activation_sums(
    q: CpuQuant,
    row0: &[u8],
    srow0: &[u8],
    row1: &[u8],
    srow1: &[u8],
    k: usize,
    x: &[f32],
    sums: &[f32],
) -> [f32; 2] {
    debug_assert!(
        has_row_pair_kernel(q),
        "row_pair_dot_with_activation_sums({q:?}): no pair kernel"
    );
    debug_assert!(
        activation_sum_kind(q).is_some_and(|kind| sums.len() >= kind.len(k)),
        "row_pair_dot_with_activation_sums({q:?}): sums shorter than the shape"
    );
    debug_assert!(
        row0.len() >= crate::gemv::row_bytes(q, k)
            && row1.len() >= crate::gemv::row_bytes(q, k)
            && srow0.len() >= 2 * (k / crate::quant::PLANAR_GROUP)
            && srow1.len() >= 2 * (k / crate::quant::PLANAR_GROUP),
        "row_pair_dot_with_activation_sums({q:?}): row or scales shorter than the shape"
    );
    debug_assert!(
        x.len() >= k,
        "row_pair_dot_with_activation_sums({q:?}): short x"
    );
    unsafe {
        match q {
            CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => x86::e8p_row_pair_dot_cached(
                row0.as_ptr(),
                srow0.as_ptr(),
                row1.as_ptr(),
                srow1.as_ptr(),
                k,
                x.as_ptr(),
                sums.as_ptr(),
            ),
            _ => unreachable!("row_pair_dot_with_activation_sums: {q:?} has no pair kernel"),
        }
    }
}

/// One weight row dotted with a pre-rotated activation on the AVX2 path.
///
/// This is the single format→kernel map, and like [`features_for`] it is
/// exhaustive: a new [`CpuQuant`] variant does not compile until its kernel and
/// its feature requirement are both stated. The caller has already resolved the
/// feature decision ([`row_dot_enabled`]) — which is why this takes no flags —
/// so `row` must be at least `(k / q.group_elems()) * q.group_bytes()` bytes and
/// `x.len() >= k`. Rotation is the caller's business, exactly as in
/// [`crate::quant::decode_group_codes`].
///
/// `scales_row` is the row's f16 scale bytes for the planar formats
/// ([`CpuQuant::is_planar`]; `2 * (k / 128)` of them, `row` is then exactly the
/// row's codes) and empty for every other format. The planar kernels read only
/// within the two slices.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_dot_avx2(q: CpuQuant, row: &[u8], scales_row: &[u8], k: usize, x: &[f32]) -> f32 {
    debug_assert!(features_for(q), "row_dot_avx2({q:?}): features absent");
    debug_assert!(
        row.len() >= crate::gemv::row_bytes(q, k) && x.len() >= k,
        "row_dot_avx2({q:?}): row or activation shorter than the shape"
    );
    debug_assert!(
        !q.is_planar() || scales_row.len() >= 2 * (k / crate::quant::PLANAR_GROUP),
        "row_dot_avx2({q:?}): scales row shorter than the shape"
    );
    let (p, xp) = (row.as_ptr(), x.as_ptr());
    // SAFETY: the caller established this format's CPU features through
    // `row_dot_enabled`, and the bounds are the row/activation lengths checked
    // above — the kernels' own contract.
    match q {
        // Flat f32 header, 256-element group.
        CpuQuant::Mq4G256 => unsafe { x86::mq4g256_row_dot(p, k, xp) },
        CpuQuant::Hfq4G256 => unsafe { x86::hfq4g256_row_dot(p, k, xp) },
        CpuQuant::Hfq6G256 => unsafe { x86::hfq6g256_row_dot(p, k, xp) },
        CpuQuant::Mq6G256 => unsafe { x86::mq6g256_row_dot(p, k, xp) },
        CpuQuant::Mq5G256 => unsafe { x86::mq5g256_row_dot(p, k, xp) },
        CpuQuant::Hfq3G256 => unsafe { x86::hfq3g256_row_dot(p, k, xp) },
        CpuQuant::Mq3G256 => unsafe { x86::mq3g256_row_dot(p, k, xp) },
        CpuQuant::Hfq2G256 => unsafe { x86::hfq2g256_row_dot(p, k, xp) },
        CpuQuant::Mq2G256 => unsafe { x86::mq2g256_row_dot(p, k, xp) },
        // Flat f32 header, 128-element block.
        CpuQuant::Hfq4G128 => unsafe { x86::hfq4g128_row_dot(p, k, xp) },
        CpuQuant::Hfq3G128 => unsafe { x86::hfq3g128_row_dot(p, k, xp) },
        CpuQuant::Hfq2G128 => unsafe { x86::hfq2g128_row_dot(p, k, xp) },
        // fp16 header.
        CpuQuant::Mq4CG256 => unsafe { x86::mq4cg256_row_dot(p, k, xp) },
        CpuQuant::Tq2G128 => unsafe { x86::tq2g128_row_dot(p, k, xp) },
        CpuQuant::Bq1G128 => unsafe { x86::bq1g128_row_dot(p, k, xp) },
        CpuQuant::Mfp4G32E8 => unsafe { x86::mfp4g32_e8_row_dot(p, k, xp) },
        // V2 family: per-128 fp16 header.
        CpuQuant::Mq4G256V2 => unsafe { x86::mq4g256v2_row_dot(p, k, xp) },
        // Per-128 fp16 header over a 128-element row-local group.
        CpuQuant::Mq4G128V2 => unsafe { x86::mq4g128v2_row_dot(p, k, xp) },
        // Planar slab formats: codes plane row + f16 scale plane row.
        CpuQuant::Mq3I4G128R256 | CpuQuant::Mq3I4G128R128 => unsafe {
            x86::w3_row_dot(p, scales_row.as_ptr(), k, xp)
        },
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => unsafe {
            x86::e8p_row_dot(p, scales_row.as_ptr(), k, xp)
        },
        CpuQuant::Mq6G256V2 => unsafe { x86::mq6g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq5G256V2 => unsafe { x86::mq5g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq3G256V2 => unsafe { x86::mq3g256v2_row_dot(p, k, xp) },
        CpuQuant::Mq2G256V2 => unsafe { x86::mq2g256v2_row_dot(p, k, xp) },
        // Lloyd-Max codebooks.
        CpuQuant::Mq2G256Lloyd => unsafe { x86::mq2g256lloyd_row_dot(p, k, xp) },
        CpuQuant::Mq2G256LloydU => unsafe { x86::mq2g256lloydu_row_dot(p, k, xp) },
        CpuQuant::Mq3G256Lloyd => unsafe { x86::mq3g256lloyd_row_dot(p, k, xp) },
        CpuQuant::Mq4G256Lloyd => unsafe { x86::mq4g256lloyd_row_dot(p, k, xp) },
        // Element formats.
        CpuQuant::F16 => unsafe { x86::f16_row_dot(p, k, xp) },
        CpuQuant::Bf16 => unsafe { x86::bf16_row_dot(p, k, xp) },
        CpuQuant::F32 => unsafe { x86::f32_row_dot(p, k, xp) },
        CpuQuant::Q8F16 => unsafe { x86::q8f16_row_dot(p, k, xp) },
    }
}

/// On a non-x86_64 target there are no vector kernels at all
/// ([`avx2_available`] is `false`), so [`row_dot_enabled`] never lets a caller
/// reach this.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn row_dot_avx2(
    _q: CpuQuant,
    _row: &[u8],
    _scales_row: &[u8],
    _k: usize,
    _x: &[f32],
) -> f32 {
    unreachable!("row_dot_avx2 is reachable only where AVX2 was detected")
}

#[cfg(test)]
mod tests;
