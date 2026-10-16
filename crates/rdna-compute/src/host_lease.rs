// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Keepalive lease on one host-mapped OWNER allocation.
//!
//! A host-mapped weight allocation has exactly one owner: the `Gpu` that
//! `hipHostMalloc`ed it (its registry frees it). The owner is either one
//! standalone allocation or a pooled backing
//! (`Gpu::reserve_portable_host_pool`) that many weight tensors are aligned
//! subviews of. When a second device engine (`memory.offload_exec=igpu`) reads
//! the same pinned bytes it holds a [`HostMappedLease`] on the owner; the `Gpu`
//! then refuses to `hipHostFree` it (`Gpu::free_tensor`, unload) while any lease
//! clone is alive. A holder that cannot prove its device drained simply never
//! drops its lease, so the allocation is retained instead of freed under an
//! in-flight read. The lease does not own the bytes and never frees them.

use std::sync::Arc;

/// Identity of one leased owner. The owning `Gpu` keeps only a `Weak` of it, so
/// "leased" means "a strong clone exists".
#[derive(Debug)]
pub(crate) struct HostLeaseToken {
    pub(crate) key: usize,
    pub(crate) host_ptr: usize,
    pub(crate) bytes: usize,
    /// Invariant: the owner was allocated `hipHostMallocMapped | Portable`.
    /// `Gpu::host_mapped_view` is the only production constructor and refuses
    /// every non-portable owner, so this is always `true` there; a second
    /// device (`Gpu::adopt_host_mapped_view`) re-checks it and fails closed.
    pub(crate) portable: bool,
}

/// A shared, clonable keepalive on one host-mapped owner of a `Gpu`
/// (`Gpu::host_mapped_view`). Dropping every clone releases the claim.
#[derive(Clone, Debug)]
pub struct HostMappedLease(pub(crate) Arc<HostLeaseToken>);

impl HostMappedLease {
    /// The device-visible base address of the owner (the registry key).
    pub fn key(&self) -> usize {
        self.0.key
    }

    /// The `hipHostMalloc` base: where a second device obtains its own alias
    /// (`hipHostGetDevicePointer` while it is the current device).
    pub fn host_ptr(&self) -> *mut u8 {
        self.0.host_ptr as *mut u8
    }

    /// Byte length of the whole owner (the pool capacity for a pooled view).
    pub fn len(&self) -> usize {
        self.0.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.0.bytes == 0
    }

    /// Live clones of this lease, this one included.
    pub fn holders(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Whether the owner is `hipHostMallocPortable` (see [`HostLeaseToken`]).
    pub fn portable(&self) -> bool {
        self.0.portable
    }
}

/// One tensor's extent inside a leased host-mapped owner: the lease on the whole
/// owner (the pooled backing shared by every subview, or one standalone
/// allocation), the byte `offset` of the tensor inside it and its logical `len`.
/// A second device maps the owner base once (`hipHostGetDevicePointer` of
/// [`HostMappedLease::host_ptr`] with that device current) and adds `offset`; an
/// interior pointer is never passed to HIP.
#[derive(Clone, Debug)]
pub struct HostMappedView {
    pub(crate) lease: HostMappedLease,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl HostMappedView {
    /// The shared lease on the owning allocation.
    pub fn lease(&self) -> &HostMappedLease {
        &self.lease
    }

    /// `hipHostMalloc` base of the owning allocation.
    pub fn owner_host_base(&self) -> *mut u8 {
        self.lease.host_ptr()
    }

    /// Byte length of the owning allocation.
    pub fn owner_len(&self) -> usize {
        self.lease.len()
    }

    /// Byte offset of this tensor inside the owner.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Logical byte length of this tensor.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `other` is a view of the same owner allocation.
    pub fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.lease.0, &other.lease.0)
    }

    /// Validate that adopting this view as a tensor of `shape`/`dtype` on a second
    /// device is in-bounds, and return the byte extent. Pure (no device access):
    /// the owner must be portable, the extent non-empty and wholly inside the
    /// owner (no overflow), and for plain float dtypes the shape must cover the
    /// extent exactly (quantized/raw dtypes carry byte-level layouts the shape
    /// alone cannot size, so only the overflow of the shape product is checked).
    pub(crate) fn checked_adopt_len(
        &self,
        shape: &[usize],
        dtype: crate::DType,
    ) -> Result<usize, String> {
        if !self.lease.portable() {
            return Err("owner is not hipHostMallocPortable".to_string());
        }
        if self.len == 0 {
            return Err("view extent is empty".to_string());
        }
        let end = self
            .offset
            .checked_add(self.len)
            .ok_or_else(|| "view extent offset+len overflows".to_string())?;
        if end > self.lease.len() {
            return Err(format!(
                "view extent [{}, {end}) exceeds the owner length {}",
                self.offset,
                self.lease.len()
            ));
        }
        let numel = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| format!("shape {shape:?} overflows usize"))?;
        if matches!(
            dtype,
            crate::DType::F32 | crate::DType::F16 | crate::DType::BF16
        ) && numel.checked_mul(dtype.size()) != Some(self.len)
        {
            return Err(format!(
                "shape {shape:?} of {dtype:?} does not cover the {}-byte view extent",
                self.len
            ));
        }
        Ok(self.len)
    }

    /// Host address of this tensor's first byte (owner base + offset), for the
    /// CPU-executed path; the owner stays allocated while a clone lives.
    pub(crate) fn host_addr(&self) -> Option<usize> {
        (self.lease.0.host_ptr).checked_add(self.offset)
    }
}

/// The single pooled host backing reserved by `Gpu::reserve_portable_host_pool`:
/// one `hipHostMalloc(MAPPED | PORTABLE)` allocation that the matching weight
/// uploads carve aligned subviews from (bump cursor, no reuse).
#[derive(Debug)]
pub(crate) struct HostPool {
    /// `hipHostMalloc` base (what `hipHostFree` takes).
    pub(crate) host_ptr: usize,
    /// Device-visible alias of the base on the owning device.
    pub(crate) dev_key: usize,
    pub(crate) capacity: usize,
    pub(crate) cursor: usize,
    /// Subviews still registered.
    pub(crate) live: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(owner_len: usize, offset: usize, len: usize, portable: bool) -> HostMappedView {
        HostMappedView {
            lease: HostMappedLease(Arc::new(HostLeaseToken {
                key: 0x1000,
                host_ptr: 0x2000,
                bytes: owner_len,
                portable,
            })),
            offset,
            len,
        }
    }

    #[test]
    fn adopt_extent_is_bounded_by_the_owner() {
        use crate::DType;
        assert_eq!(view(8192, 4096, 4096, true).checked_adopt_len(&[4096], DType::Raw), Ok(4096));
        assert!(view(8192, 4097, 4096, true).checked_adopt_len(&[4096], DType::Raw).is_err());
        assert!(view(8192, usize::MAX, 2, true).checked_adopt_len(&[2], DType::Raw).is_err());
        assert!(view(8192, 0, 0, true).checked_adopt_len(&[0], DType::Raw).is_err());
        assert!(view(8192, 0, 16, false).checked_adopt_len(&[16], DType::Raw).is_err());
        // Float shapes must cover the extent exactly; quantized shapes only must not overflow.
        assert_eq!(view(8192, 0, 16, true).checked_adopt_len(&[2, 2], DType::F32), Ok(16));
        assert!(view(8192, 0, 16, true).checked_adopt_len(&[2, 3], DType::F32).is_err());
        assert_eq!(view(8192, 0, 16, true).checked_adopt_len(&[4, 7], DType::MQ4G256V2), Ok(16));
        assert!(view(8192, 0, 16, true)
            .checked_adopt_len(&[usize::MAX, 2], DType::MQ4G256V2)
            .is_err());
    }

    #[test]
    fn clones_keep_one_owner_and_count_holders() {
        let first = view(8192, 0, 4096, true);
        let second = first.clone();
        assert!(first.same_owner(&second));
        assert_eq!(first.lease().holders(), 2);
        assert_eq!(second.host_addr(), Some(0x2000));
        drop(second);
        assert_eq!(first.lease().holders(), 1);
        assert!(!first.same_owner(&view(8192, 0, 4096, true)));
    }
}
