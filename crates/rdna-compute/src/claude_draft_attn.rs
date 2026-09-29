// claude v3: route the plain full-causal single-row Q8 decode attention
// (`KernelKey::AttnFlashQ8_0` with no output gate -- in production the MTP
// draft head) to the WMMA split-K `claude_attn4` kernel + 8-warp reduce
// instead of `attention_flash_q8_0_tile` + `attention_flash_q8_0_reduce`.
//
// Called from the `(None, None)` arm of `KernelKey::AttnFlashQ8_0` in
// `hipfire-dispatch/src/families/attention.rs`; that arm falls back to the
// stock `attention_flash_q8_0` whenever this returns `Ok(false)`.
//
// Conventions shared with the stock decode path (checked against
// `attention_flash_q8_0_tile.hip` / `attention_flash_q8_0_reduce.hip` and
// `claude_attn4.hip` / `claude_attn_reduce.hip`):
// - `pos_buf[0]` (i32) is the current token's absolute position; both kernel
//   pairs attend keys `[0, pos_buf[0] + 1)` at physical row == position
//   (window <= 0: no ring). claude_attn4 reads `positions[0]` for row 0, so
//   the 4-byte `pos_buf` doubles as its `[1]` positions tensor.
// - `seq_len_hint` (host) only sizes the launch: the stock direct launch uses
//   `ceil(seq_len_hint / tile)` tiles, claude_attn4 uses
//   `ceil(seq_len_hint / span)` spans; both require seq_len_hint >= pos + 1
//   (the dispatch family passes `pos + 1`) and ignore spans past the device
//   position.
// - `partials` is the decode scratch sized `n_heads * ceil(max_seq / tile) *
//   (2 + head_dim)`; claude_attn4 checks it holds `n_heads *
//   ceil(seq_len_hint / span) * (2 + head_dim)` and widens the span (or
//   declines) when it does not.
//
// Declines (stock path runs) under hipGraph capture (`graphs.capture_mode`)
// and redline recording (`replay.is_recording()`): the stock launch retains a
// position-independent `max_tiles` grid there, while claude_attn4's grid is
// sized from `seq_len_hint`, so it must never be baked into a replayed graph.
// `CLAUDE_DRAFT_ATTN4=0` disables the route (read once). `CLAUDE_ATTN4=0`
// (the verify-path kill switch inside `claude_attn4_ok`) also disables it.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{DeviceBuffer, HipResult};

/// `CLAUDE_DRAFT_ATTN5` != "0" (default on), read once: route the draft head
/// through `claude_attn5` before `claude_attn4`.
pub fn claude_draft_attn5_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_DRAFT_ATTN5").as_deref() != Ok("0"))
}

/// `CLAUDE_DRAFT_ATTN4` != "0" (default on), read once.
pub fn claude_draft_attn4_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_DRAFT_ATTN4").as_deref() != Ok("0"))
}

impl Gpu {
    /// claude v3: single-row full-causal Q8_0 decode attention through
    /// `claude_attn4` (rows = 1). Same arguments as [`Gpu::attention_flash_q8_0`];
    /// `Ok(true)` = launched (output written), `Ok(false)` = not applicable,
    /// caller must run the stock path.
    #[allow(clippy::too_many_arguments)]
    pub fn claude_draft_attn4(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        pos_buf: &DeviceBuffer,
        seq_len_hint: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        partials: &GpuTensor,
    ) -> HipResult<bool> {
        if !claude_draft_attn4_enabled()
            || self.arch.as_str() != "gfx1100"
            || n_heads != 24
            || n_kv_heads != 4
            || head_dim != 256
            || self.graphs.capture_mode
            || self.replay.is_recording()
            || seq_len_hint == 0
            || seq_len_hint > max_seq
            || pos_buf.size() < 4
            || q.numel() < n_heads * head_dim
            || out.numel() < n_heads * head_dim
        {
            return Ok(false);
        }
        // Non-owning [1] i32 view of the decode position slot.
        let positions = GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(pos_buf.as_ptr(), pos_buf.size()) },
            shape: vec![1],
            dtype: DType::F32,
        };
        // claude v4: prefer claude_attn5 (QT=1 variant, ~1.6x attn4 at 40K;
        // CLAUDE_ATTN5=0 or CLAUDE_DRAFT_ATTN5=0 keeps attn4). Its fixed grid
        // ignores seq_len_hint; it needs n_heads * nblk * 260 partial floats.
        if claude_draft_attn5_enabled()
            && self.claude_attn5(
                q,
                k_cache,
                v_cache,
                out,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len_hint,
                1,
                partials,
            )?
        {
            return Ok(true);
        }
        self.claude_attn4(
            q,
            k_cache,
            v_cache,
            out,
            &positions,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len_hint,
            1,
            partials,
        )
    }
}
