// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-side helpers for the Qwen4 MoE CPU expert splice.
//!
//! A host-mapped routed-expert layer keeps its experts in two aggregate
//! allocations (gate/up and down), `experts` equal-stride extents each. The CPU
//! splice multiplies straight out of the pinned host bytes, so it needs, per
//! expert, one borrowed `&[u8]` into each aggregate. [`CpuExpertOwners`] caches
//! the host address of the two OWNING allocations once (a registry lookup per
//! aggregate, not per token) and hands out checked sub-slices.
//!
//! [`CpuSpliceAdmission`] is the pure arming decision: the splice runs only for
//! host-mapped, supported layers with no EP / slab / tier / AWQ route that
//! owns the weights differently.

use super::*;

/// Largest verify window (rows) that counts as "narrow" for the CPU splice.
pub(super) const NARROW_VERIFY_MAX_ROWS: usize =
    hipfire_dispatch::families::moe::NARROW_VERIFY_MAX_ROWS;

/// Whether the CPU expert splice may arm for one layer. Every field is an
/// input; [`Self::armed`] is the single decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuSpliceAdmission {
    /// `memory.offload_exec=cpu` selected the CPU splice.
    pub enabled: bool,
    /// The layer's routed-expert aggregates are host-mapped (not VRAM-resident).
    pub host_mapped: bool,
    /// The layer is routed through expert parallelism.
    pub ep: bool,
    /// The layer is a sealed expert-slab arena.
    pub slab: bool,
    /// The layer is governed by a per-expert tier plan.
    pub tier: bool,
    /// The layer carries AWQ scales (a bare CPU gemv would ignore them).
    pub awq: bool,
    /// Both aggregate dtypes have a CPU executor.
    pub supported: bool,
}

impl CpuSpliceAdmission {
    /// Armed only when enabled, host-mapped and supported, with no exclusion.
    pub(super) fn armed(&self) -> bool {
        self.enabled
            && self.host_mapped
            && self.supported
            && !self.ep
            && !self.slab
            && !self.tier
            && !self.awq
    }
}

/// `[expert_idx * stride, (expert_idx + 1) * stride)` inside an aggregate of
/// `experts` extents, `None` when the index is out of range or the arithmetic
/// overflows. The only place the per-expert offset is computed.
fn expert_range(
    expert_idx: usize,
    experts: usize,
    stride: usize,
) -> Option<std::ops::Range<usize>> {
    if expert_idx >= experts {
        return None;
    }
    let start = expert_idx.checked_mul(stride)?;
    let end = start.checked_add(stride)?;
    Some(start..end)
}

/// Cached host views of one layer's two aggregate expert allocations.
///
/// The private runtime borrows these addresses under the bundle's ownership:
/// aggregate weights outlive the forward and are not replaced after attachment.
/// As with `slab_stage`'s host sources, only `Gpu::host_bytes` of the owning
/// tensors supplies host addresses; borrowed device views are never re-tagged.
pub(super) struct CpuExpertOwners {
    gate_up_ptr: usize,
    gate_up_len: usize,
    down_ptr: usize,
    down_len: usize,
    experts: usize,
    gate_up_stride: usize,
    down_stride: usize,
}

impl CpuExpertOwners {
    /// Resolve the OWNING host-mapped tensors' pinned host bytes and check that
    /// `experts` extents of the given strides fit inside them. `gate_up` and
    /// `down` must be the aggregate owners, never borrowed sub-views: a view's
    /// `host_bytes` lookup fails (its pointer is not a registry key), which is
    /// reported as `None` rather than retagged.
    pub(super) fn new(
        gpu: &Gpu,
        gate_up: &GpuTensor,
        down: &GpuTensor,
        experts: usize,
        gate_up_stride: usize,
        down_stride: usize,
    ) -> Option<Self> {
        let gate_up_bytes = gpu.host_bytes(gate_up)?;
        let down_bytes = gpu.host_bytes(down)?;
        Self::from_slices(
            gate_up_bytes,
            down_bytes,
            experts,
            gate_up_stride,
            down_stride,
        )
    }

    /// Validate extents and cache the slices' addresses. Zero experts or a zero
    /// stride is refused (an empty expert is not a CPU-multipliable payload).
    fn from_slices(
        gate_up: &[u8],
        down: &[u8],
        experts: usize,
        gate_up_stride: usize,
        down_stride: usize,
    ) -> Option<Self> {
        if experts == 0 || gate_up_stride == 0 || down_stride == 0 {
            return None;
        }
        let gate_up_total = experts.checked_mul(gate_up_stride)?;
        let down_total = experts.checked_mul(down_stride)?;
        if gate_up_total > gate_up.len() || down_total > down.len() {
            return None;
        }
        Some(Self {
            gate_up_ptr: gate_up.as_ptr() as usize,
            gate_up_len: gate_up.len(),
            down_ptr: down.as_ptr() as usize,
            down_len: down.len(),
            experts,
            gate_up_stride,
            down_stride,
        })
    }

    /// Checked byte ranges of `expert_idx` in the gate/up and down aggregates.
    fn ranges(
        &self,
        expert_idx: usize,
    ) -> Option<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        let gate_up = expert_range(expert_idx, self.experts, self.gate_up_stride)?;
        let down = expert_range(expert_idx, self.experts, self.down_stride)?;
        if gate_up.end > self.gate_up_len || down.end > self.down_len {
            return None;
        }
        Some((gate_up, down))
    }

    /// Borrowed CPU payload for one expert, `None` for an out-of-range expert or
    /// a dtype without a CPU executor.
    pub(super) fn payload(
        &self,
        expert_idx: usize,
        gate_up_dtype: DType,
        down_dtype: DType,
    ) -> Option<hipfire_dispatch::cpu_exec::CpuMoeExpert<'_>> {
        let (gate_up_range, down_range) = self.ranges(expert_idx)?;
        let gate_up_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(gate_up_dtype)?;
        let down_quant = hipfire_dispatch::cpu_exec::cpu_quant_for(down_dtype)?;
        // SAFETY: `gate_up_ptr`/`down_ptr` are the addresses of the OWNING
        // host-mapped allocations' pinned bytes (`Gpu::host_bytes`), of
        // `*_len` bytes, and `ranges` has checked both extents lie inside
        // them. The allocations live in the bundle's weights, which outlive
        // the private runtime holding this cache (the runtime is torn down
        // first, then the weights free their host allocations), and no
        // aggregate owner is replaced after the forward attaches — the same
        // cached-address pattern as the host arena sources in `slab_stage.rs`.
        // The returned slices are shared reads only and borrow `self`, so they
        // cannot outlive this cache.
        let (gate_up, down) = unsafe {
            (
                std::slice::from_raw_parts(
                    (self.gate_up_ptr + gate_up_range.start) as *const u8,
                    gate_up_range.len(),
                ),
                std::slice::from_raw_parts(
                    (self.down_ptr + down_range.start) as *const u8,
                    down_range.len(),
                ),
            )
        };
        Some(hipfire_dispatch::cpu_exec::CpuMoeExpert {
            gate_up_quant,
            gate_up,
            down_quant,
            down,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real per-expert extents (QT44 / QT53 strides), 512 experts.
    const STRIDE_A: usize = 1_740_800;
    const STRIDE_B: usize = 870_400;
    const EXPERTS: usize = 512;

    fn admission(
        enabled: bool,
        host_mapped: bool,
        ep: bool,
        slab: bool,
        tier: bool,
        awq: bool,
        supported: bool,
    ) -> CpuSpliceAdmission {
        CpuSpliceAdmission {
            enabled,
            host_mapped,
            ep,
            slab,
            tier,
            awq,
            supported,
        }
    }

    #[test]
    fn narrow_verify_window_is_four_rows() {
        assert_eq!(NARROW_VERIFY_MAX_ROWS, 4);
    }

    #[test]
    fn arming_decision_table() {
        // (name, admission, armed)
        let table = [
            (
                "enabled host supported",
                admission(true, true, false, false, false, false, true),
                true,
            ),
            (
                "disabled",
                admission(false, true, false, false, false, false, true),
                false,
            ),
            (
                "ep",
                admission(true, true, true, false, false, false, true),
                false,
            ),
            (
                "slab",
                admission(true, true, false, true, false, false, true),
                false,
            ),
            (
                "tier",
                admission(true, true, false, false, true, false, true),
                false,
            ),
            (
                "awq",
                admission(true, true, false, false, false, true, true),
                false,
            ),
            (
                "unsupported",
                admission(true, true, false, false, false, false, false),
                false,
            ),
            (
                "resident",
                admission(true, false, false, false, false, false, true),
                false,
            ),
            (
                "all exclusions",
                admission(true, true, true, true, true, true, true),
                false,
            ),
            ("nothing set", CpuSpliceAdmission::default(), false),
        ];
        for (name, a, armed) in table {
            assert_eq!(a.armed(), armed, "{name}");
        }
    }

    #[test]
    fn arming_is_exhaustive_over_all_inputs() {
        for bits in 0u32..128 {
            let b = |i: u32| bits & (1 << i) != 0;
            let a = admission(b(0), b(1), b(2), b(3), b(4), b(5), b(6));
            // Exactly one assignment arms: enabled | host_mapped | supported,
            // no exclusion bit.
            assert_eq!(a.armed(), bits == 0b100_0011, "bits={bits:#09b}");
        }
    }

    #[test]
    fn default_pcie_disabled_arms_no_layer() {
        // Default (PCIe) mode: the splice is not selected, so none of the 48
        // layers arms, however host-mapped and supported they are.
        const LAYERS: usize = 48;
        let disabled = admission(false, true, false, false, false, false, true);
        assert_eq!((0..LAYERS).filter(|_| disabled.armed()).count(), 0);
        // Control: the same layers arm once enabled.
        let enabled = admission(true, true, false, false, false, false, true);
        assert_eq!((0..LAYERS).filter(|_| enabled.armed()).count(), LAYERS);
    }

    #[test]
    fn expert_range_ids_0_1_511_at_real_strides() {
        for stride in [STRIDE_A, STRIDE_B] {
            for idx in [0usize, 1, 511] {
                let r = expert_range(idx, EXPERTS, stride)
                    .unwrap_or_else(|| panic!("stride {stride} idx {idx}"));
                assert_eq!(r.start, idx * stride, "stride {stride} idx {idx}");
                assert_eq!(r.end, (idx + 1) * stride, "stride {stride} idx {idx}");
                assert_eq!(r.len(), stride, "stride {stride} idx {idx}");
                assert!(r.end <= EXPERTS * stride, "stride {stride} idx {idx}");
            }
            // Last expert ends exactly at the aggregate end.
            assert_eq!(
                expert_range(EXPERTS - 1, EXPERTS, stride).unwrap().end,
                EXPERTS * stride
            );
            assert!(expert_range(EXPERTS, EXPERTS, stride).is_none());
        }
        assert_eq!(
            expert_range(511, EXPERTS, STRIDE_A).unwrap().start,
            889_548_800
        );
        assert_eq!(
            expert_range(511, EXPERTS, STRIDE_B).unwrap().start,
            444_774_400
        );
        assert_eq!(EXPERTS * STRIDE_A, 891_289_600);
        assert_eq!(EXPERTS * STRIDE_B, 445_644_800);
    }

    #[test]
    fn expert_range_rejects_out_of_range_and_overflow() {
        assert!(expert_range(512, EXPERTS, STRIDE_A).is_none());
        assert!(expert_range(usize::MAX, EXPERTS, STRIDE_A).is_none());
        assert!(expert_range(0, 0, STRIDE_A).is_none());
        // idx * stride overflows.
        assert!(expert_range(3, usize::MAX, usize::MAX / 2).is_none());
        // start fits but start + stride overflows.
        assert!(expert_range(1, 2, usize::MAX - 1).is_none());
    }

    // Small fake layout: 512 experts, stride 64 (gate/up) and 32 (down), each
    // expert filled with a byte derived from its id so the resolved subspan
    // identity is observable. No large allocation.
    const FAKE_GU: usize = 64;
    const FAKE_DN: usize = 32;

    fn fake_layout() -> (Vec<u8>, Vec<u8>) {
        let mut gate_up = Vec::with_capacity(EXPERTS * FAKE_GU);
        let mut down = Vec::with_capacity(EXPERTS * FAKE_DN);
        for e in 0..EXPERTS {
            gate_up.extend(std::iter::repeat((e % 251) as u8 + 1).take(FAKE_GU));
            down.extend(std::iter::repeat(0x80 | (e % 127) as u8).take(FAKE_DN));
        }
        (gate_up, down)
    }

    #[test]
    fn payload_resolves_exact_subspans_for_ids_0_1_511() {
        let (gate_up, down) = fake_layout();
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        for idx in [0usize, 1, 511] {
            let p = owners
                .payload(idx, DType::MQ4G256, DType::MQ4G256)
                .unwrap_or_else(|| panic!("idx {idx}"));
            assert_eq!(p.gate_up.len(), FAKE_GU, "idx {idx}");
            assert_eq!(p.down.len(), FAKE_DN, "idx {idx}");
            assert_eq!(
                p.gate_up.as_ptr() as usize,
                gate_up.as_ptr() as usize + idx * FAKE_GU,
                "idx {idx}"
            );
            assert_eq!(
                p.down.as_ptr() as usize,
                down.as_ptr() as usize + idx * FAKE_DN,
                "idx {idx}"
            );
            assert_eq!(
                p.gate_up,
                &gate_up[idx * FAKE_GU..(idx + 1) * FAKE_GU],
                "idx {idx}"
            );
            assert_eq!(
                p.down,
                &down[idx * FAKE_DN..(idx + 1) * FAKE_DN],
                "idx {idx}"
            );
        }
        // Distinct experts resolve to distinct, adjacent, non-overlapping bytes.
        let p0 = owners.payload(0, DType::MQ4G256, DType::MQ4G256).unwrap();
        let p1 = owners.payload(1, DType::MQ4G256, DType::MQ4G256).unwrap();
        assert_ne!(p0.gate_up[0], p1.gate_up[0]);
        assert_eq!(
            p1.gate_up.as_ptr() as usize - p0.gate_up.as_ptr() as usize,
            FAKE_GU
        );
    }

    #[test]
    fn payload_accepts_aggregate_longer_than_experts_times_stride() {
        // A mapped allocation may carry slack past the last expert.
        let (mut gate_up, mut down) = fake_layout();
        gate_up.extend([0xEE; 17]);
        down.extend([0xEE; 5]);
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        let p = owners.payload(511, DType::MQ4G256, DType::MQ4G256).unwrap();
        assert_eq!(p.gate_up.len(), FAKE_GU);
        assert_eq!(p.down.len(), FAKE_DN);
        assert!(owners
            .payload(512, DType::MQ4G256, DType::MQ4G256)
            .is_none());
    }

    #[test]
    fn payload_rejects_out_of_range_expert() {
        let (gate_up, down) = fake_layout();
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        for idx in [EXPERTS, EXPERTS + 1, usize::MAX] {
            assert!(
                owners
                    .payload(idx, DType::MQ4G256, DType::MQ4G256)
                    .is_none(),
                "idx {idx}"
            );
        }
    }

    #[test]
    fn payload_rejects_untyped_dtype() {
        let (gate_up, down) = fake_layout();
        let owners =
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).unwrap();
        // `Raw` is an untyped byte blob, never a CPU-executed weight dtype.
        assert!(owners.payload(0, DType::Raw, DType::MQ4G256).is_none());
        assert!(owners.payload(0, DType::MQ4G256, DType::Raw).is_none());
        assert!(owners.payload(0, DType::Raw, DType::Raw).is_none());
    }

    #[test]
    fn from_slices_rejects_short_owner() {
        let (gate_up, down) = fake_layout();
        // One byte short in either aggregate.
        assert!(CpuExpertOwners::from_slices(
            &gate_up[..gate_up.len() - 1],
            &down,
            EXPERTS,
            FAKE_GU,
            FAKE_DN
        )
        .is_none());
        assert!(CpuExpertOwners::from_slices(
            &gate_up,
            &down[..down.len() - 1],
            EXPERTS,
            FAKE_GU,
            FAKE_DN
        )
        .is_none());
        // Empty owners.
        assert!(CpuExpertOwners::from_slices(&[], &[], EXPERTS, FAKE_GU, FAKE_DN).is_none());
        // Strides larger than the owner can hold.
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU + 1, FAKE_DN).is_none()
        );
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN + 1).is_none()
        );
        // Exactly sized owners are accepted.
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, FAKE_DN).is_some());
    }

    #[test]
    fn from_slices_rejects_overflow_and_zero_geometry() {
        let (gate_up, down) = fake_layout();
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, usize::MAX, FAKE_GU, FAKE_DN).is_none()
        );
        assert!(
            CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, usize::MAX, FAKE_DN).is_none()
        );
        assert!(CpuExpertOwners::from_slices(
            &gate_up,
            &down,
            EXPERTS,
            FAKE_GU,
            usize::MAX / 2 + 1
        )
        .is_none());
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, 0, FAKE_GU, FAKE_DN).is_none());
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, 0, FAKE_DN).is_none());
        assert!(CpuExpertOwners::from_slices(&gate_up, &down, EXPERTS, FAKE_GU, 0).is_none());
    }
}
