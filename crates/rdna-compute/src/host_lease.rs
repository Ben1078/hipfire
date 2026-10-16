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
