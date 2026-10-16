// SPDX-License-Identifier: Apache-2.0
//! Fixed-state staging for a session boundary between pipeline banks.
use super::*;
use hipfire_runtime::session_cache::StatePart;
use rdna_compute::DType;

pub(crate) struct PairSessionStage {
    pub(crate) fixed: Vec<GpuTensor>,
    meta: Vec<u8>,
    rows: Vec<usize>,
    pub(crate) position: Option<usize>,
}

impl PairSessionStage {
    pub(crate) fn copy(&self, gpu: &mut Gpu, index: usize, src: &GpuTensor) -> Result<(), BundleError> {
        self.copy_parts(gpu, index, [src])
    }

    pub(crate) fn copy_parts<const N: usize>(
        &self, gpu: &mut Gpu, index: usize, src: [&GpuTensor; N],
    ) -> Result<(), BundleError> {
        let dst = self.fixed.get(index..index + N).ok_or_else(|| BundleError::Forward("paired snapshot fixed index mismatch".into()))?;
        for (src, dst) in src.iter().zip(dst) {
            if src.byte_size() != dst.byte_size() || !src.byte_size().is_multiple_of(4) {
                return Err(BundleError::Forward("paired snapshot state must use equal aligned device copies".into()));
            }
        }
        // A copy kernel is stream-ordered even on the default stream;
        // copy_d2d's legacy-stream fallback would synchronize the host.
        let regions: [_; N] = std::array::from_fn(|i| rdna_compute::tensor_ops::CopyRegion {
            dst: &dst[i].buf, dst_offset: 0, src: &src[i].buf, src_offset: 0,
            bytes: src[i].byte_size(),
        });
        rdna_compute::tensor_ops::copy_regions(gpu, &regions).map_err(BundleError::Hip)
    }

    pub(crate) fn configure(
        &mut self, state: &Qwen4State, history: crate::ple::PleHistory,
        commits: &[(usize, usize, usize, usize, usize)],
    ) -> Result<(), BundleError> {
        let (meta, layout) = state.session_parts_at(self.position.expect("armed stage"), history, Some(commits)).map_err(BundleError::State)?;
        debug_assert_eq!(self.fixed.len(), layout.fixed.len());
        for (saved, part) in self.fixed.iter().zip(&layout.fixed) {
            debug_assert_eq!(saved.byte_size(), part.bytes);
        }
        self.meta = meta;
        #[cfg(debug_assertions)]
        {
            // The ordinary restore parser must reconstruct exactly the staged
            // boundary layout. This is host-only; never drain the HIP stream.
            let sequential = state.session_layout_for_meta(&self.meta).map_err(BundleError::State)?;
            assert_layout_shape(
                self.fixed.iter().map(GpuTensor::byte_size),
                layout.rows.iter().map(|stream| (stream.row_bytes, stream.rows)),
                sequential.fixed.iter().map(|part| part.bytes),
                sequential.rows.iter().map(|stream| (stream.row_bytes, stream.rows)),
            );
        }
        self.rows = layout.rows.iter().map(|stream| stream.rows).collect();
        Ok(())
    }

    pub(super) fn snapshot_parts<'a>(&'a self, state: &'a Qwen4State) -> Result<SnapshotParts<'a>, StateError> {
        Ok(self.parts(state.session_layout_for_meta(&self.meta)?))
    }

    pub(super) fn parts<'a>(&'a self, mut live: StateLayout<'a>) -> SnapshotParts<'a> {
        debug_assert_eq!(self.fixed.len(), live.fixed.len());
        debug_assert_eq!(self.rows.len(), live.rows.len());
        for (saved, part) in self.fixed.iter().zip(&mut live.fixed) {
            debug_assert_eq!(saved.byte_size(), part.bytes);
            *part = StatePart { buf: &saved.buf, offset: 0, bytes: saved.byte_size() };
        }
        for (&rows, stream) in self.rows.iter().zip(&mut live.rows) {
            debug_assert!(rows <= stream.rows);
            stream.rows = rows;
        }
        SnapshotParts { meta: self.meta.clone(), layout: live }
    }
}

#[cfg(debug_assertions)]
fn assert_layout_shape(
    fixed: impl Iterator<Item = usize>, rows: impl Iterator<Item = (usize, usize)>,
    sequential_fixed: impl Iterator<Item = usize>,
    sequential_rows: impl Iterator<Item = (usize, usize)>,
) {
    assert!(fixed.eq(sequential_fixed), "paired snapshot fixed layout differs from sequential");
    assert!(rows.eq(sequential_rows), "paired snapshot row layout differs from sequential");
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::assert_layout_shape;

    #[test]
    fn staged_layout_matches_sequential_without_device_buffers() {
        let fixed = [64, 128, 32, 16];
        let rows = [(8, 8192), (8, 8192), (4, 8192), (4, 512)];
        assert_layout_shape(fixed.into_iter(), rows.into_iter(), fixed.into_iter(), rows.into_iter());
    }

    #[test]
    #[should_panic(expected = "row layout differs")]
    fn pair_end_rows_cannot_be_published_at_a_boundary() {
        assert_layout_shape([64].into_iter(), [(8, 16384)].into_iter(),
            [64].into_iter(), [(8, 8192)].into_iter());
    }
}

impl Qwen4Bundle {
    pub(crate) fn prepare_pair_session_stage(
        &mut self, gpu: &mut Gpu, position: usize,
    ) -> Result<Option<PairSessionStage>, BundleError> {
        if self.session.is_none() || self.session_next_boundary() != Some(position) {
            return Ok(None);
        }
        let (_, layout) = self.state.session_parts().map_err(BundleError::State)?;
        let mut stage = self.pair_session_stage.take().unwrap_or(PairSessionStage {
            fixed: Vec::new(), meta: Vec::new(), rows: Vec::new(), position: None,
        });
        if stage.fixed.is_empty() {
            for part in &layout.fixed {
                match gpu.zeros(&[part.bytes], DType::Raw) {
                    Ok(tensor) => stage.fixed.push(tensor),
                    Err(error) => {
                        for tensor in stage.fixed { let _ = gpu.free_tensor(tensor); }
                        return Err(BundleError::Hip(error));
                    }
                }
            }
        }
        debug_assert_eq!(stage.fixed.len(), layout.fixed.len());
        stage.position = Some(position);
        // Feedback is an unchanged fixed part on this route, but is still
        // captured: the cache stores it even on profiles without PLE.
        let hyper_feedback = self
            .state
            .hyper_feedback
            .as_ref()
            .expect("hyper_feedback is input-band state");
        stage.copy(gpu, stage.fixed.len() - 1, hyper_feedback)?;
        Ok(Some(stage))
    }

    /// Use the retained canonical prefix, without allocating another token copy.
    /// MTP calls this only after A's head append and before B's append.
    pub(crate) fn publish_pair_session_boundary(&mut self, gpu: &mut Gpu, position: usize) -> Result<(), BundleError> {
        if self.session_next_boundary() != Some(position) { return Ok(()); }
        let result = self
            .ep_refuse("session_at_boundary outside a replicated call")
            .and_then(|()| {
                self.with_session(|session, bundle| session.at_turn_boundary(gpu, bundle, position))
                    .unwrap_or(Ok(()))
                    .map_err(BundleError::Forward)
            });
        if let Some(stage) = &mut self.pair_session_stage { stage.position = None; }
        result
    }

    pub(crate) fn pair_boundary_is_ar(&self) -> bool {
        self.session.as_ref().and_then(SessionDriver::turn_route) == Some(SessionRoute::Ar)
    }
}
