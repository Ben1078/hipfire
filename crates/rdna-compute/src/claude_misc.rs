//! claude r2: small helpers for the MTP KV-only forward (C5).
use crate::{Gpu, GpuTensor};
use hip_bridge::{HipResult, KernargBlob};

const CONCAT2_SRC: &str = include_str!("../../../kernels/src/claude_concat2.hip");

impl Gpu {
    /// `out[r] = [a[r] | b[r]]` for `n` rows of `dim` floats each (one launch).
    pub fn claude_concat2_rows_f32(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        assert!(a.numel() >= n * dim && b.numel() >= n * dim && out.numel() >= n * 2 * dim);
        self.bind_thread()?;
        if !self.functions.contains_key("claude_concat2_rows_f32") {
            self.ensure_kernel("claude_concat2", CONCAT2_SRC, "claude_concat2_rows_f32")?;
        }
        let mut k = KernargBlob::new();
        k.push_ptr(a.buf.as_ptr());
        k.push_ptr(b.buf.as_ptr());
        k.push_ptr(out.buf.as_ptr());
        k.push_i32(n as i32);
        k.push_i32(dim as i32);
        let gx = ((2 * dim + 255) / 256) as u32;
        self.launch_kernel_blob("claude_concat2_rows_f32", [gx, n as u32, 1], [256, 1, 1], 0, k.as_mut_slice())
    }
}
