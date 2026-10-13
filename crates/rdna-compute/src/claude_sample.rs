//! claude r2 (`CLAUDE_MTP_GPU_SAMPLE`): on-device sampling for the sampled MTP
//! draft and the verify's residual acceptance. See
//! `kernels/src/claude_mtp_sample.hip` for the algorithm and the argument that
//! it draws from the same distributions as the host `sample_categorical` /
//! `sample_residual` / `mtp_sampled_accept`.
use crate::claude_doorbell::{ClaudeDoorbell, DoorbellTicket};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;

const SRC: &str = include_str!("../../../kernels/src/claude_mtp_sample.hip");
/// Elements per block chunk (256 threads x 16-element runs in the final scan).
const CHUNK: usize = 4096;
/// Max uniforms per accept (d + 1 <= 9).
pub const CLAUDE_ACCEPT_MAX_D: usize = 8;

/// Device scratch shared by the sampling kernels (tiny).
pub struct ClaudeSampleScratch {
    part: GpuTensor,
    counter: GpuTensor,
}

impl ClaudeSampleScratch {
    pub fn new(gpu: &mut Gpu) -> HipResult<Self> {
        let part = gpu.zeros(&[3 * 1024], DType::F32)?;
        let counter = gpu.zeros(&[4], DType::F32)?;
        Ok(Self { part, counter })
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.part);
        let _ = gpu.free_tensor(self.counter);
    }
}

const SMX_SRC: &str = include_str!("../../../kernels/src/claude_softmax_mb.hip");

/// Scratch of the multi-block softmax + nucleus (`claude_softmax_topp_mb`)
/// for up to `rows` rows.
pub struct ClaudeSoftmaxScratch {
    part: GpuTensor,
    /// Per-block top-64 candidate values, `[rows][<=1024 blocks][64]`.
    cand: GpuTensor,
    counter: GpuTensor,
    rows: usize,
}

impl ClaudeSoftmaxScratch {
    pub fn new(gpu: &mut Gpu, rows: usize) -> HipResult<Self> {
        Ok(Self {
            part: gpu.zeros(&[3 * rows * 1024], DType::F32)?,
            cand: gpu.zeros(&[rows * 1024 * 64], DType::F32)?,
            counter: gpu.zeros(&[rows.max(4)], DType::F32)?,
            rows,
        })
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.part);
        let _ = gpu.free_tensor(self.cand);
        let _ = gpu.free_tensor(self.counter);
    }
}

/// `CLAUDE_SOFTMAX_MB` (default on): the GPU-sampled MTP path computes its
/// softmax + nucleus rows with the multi-block kernels instead of the
/// one-workgroup-per-row `softmax_temp_topp_batched_f32`.
pub fn claude_softmax_mb_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_SOFTMAX_MB").as_deref() != Ok("0"))
}

/// Result of an on-device accept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaudeAcceptResult {
    pub committed: Vec<u32>,
    pub accept_count: usize,
    pub hit_eos: bool,
    /// Uniforms consumed (tests + bonus draw), for advancing the host RNG.
    pub n_consumed: usize,
}

fn grid_for(n: usize) -> u32 {
    let g = (n + CHUNK - 1) / CHUNK;
    assert!(g >= 1 && g <= 1024, "claude sampler: row of {n} needs {g} blocks (max 1024)");
    g as u32
}

/// `CLAUDE_MTP_GPU_SAMPLE` (default on; needs the doorbell): sampled MTP
/// draft + accept on the device.
pub fn claude_mtp_gpu_sample_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_MTP_GPU_SAMPLE").as_deref() != Ok("0"))
}

impl Gpu {
    /// Multi-block `softmax_temp_topp_batched_into_f32`: same outputs (probs,
    /// AR-law truncation tau, kept mass Z per row) and definitions, see
    /// `kernels/src/claude_softmax_mb.hip`. `top_k` is the AR candidate cap
    /// (`llama::ar_candidate_cap`); `<= 0` or `> 64` means 64.
    #[allow(clippy::too_many_arguments)]
    pub fn claude_softmax_topp_mb(
        &mut self,
        sc: &ClaudeSoftmaxScratch,
        logits: &GpuTensor,
        probs: &GpuTensor,
        tau: &GpuTensor,
        z: &GpuTensor,
        n: usize,
        rows: usize,
        temp: f32,
        top_p: f32,
        top_k: usize,
        min_p: f32,
    ) -> HipResult<()> {
        assert!(rows >= 1 && rows <= sc.rows, "claude_softmax_topp_mb: {rows} rows > scratch {}", sc.rows);
        self.bind_thread()?;
        if !self.functions.contains_key("claude_smx_max") {
            for f in ["claude_smx_max", "claude_smx_exp", "claude_smx_norm", "claude_smx_kept"] {
                self.ensure_kernel("claude_softmax_mb", SMX_SRC, f)?;
            }
        }
        let g = grid_for(n);
        let inv_t = 1.0f32 / temp;
        let part = sc.part.buf.as_ptr() as *mut u8;
        let (pmax, psum, kpart) = unsafe {
            (
                part as *mut c_void,
                part.add(4 * rows * 1024) as *mut c_void,
                part.add(8 * rows * 1024) as *mut c_void,
            )
        };
        let grid = [g, rows as u32, 1];
        let mut b = KernargBlob::new();
        b.push_ptr(logits.buf.as_ptr());
        b.push_i32(n as i32);
        b.push_f32(inv_t);
        b.push_ptr(pmax);
        self.launch_kernel_blob("claude_smx_max", grid, [256, 1, 1], 0, b.as_mut_slice())?;
        let mut b = KernargBlob::new();
        b.push_ptr(logits.buf.as_ptr());
        b.push_ptr(probs.buf.as_ptr());
        b.push_i32(n as i32);
        b.push_f32(inv_t);
        b.push_ptr(pmax);
        b.push_ptr(psum);
        self.launch_kernel_blob("claude_smx_exp", grid, [256, 1, 1], 0, b.as_mut_slice())?;
        let mut b = KernargBlob::new();
        b.push_ptr(probs.buf.as_ptr());
        b.push_i32(n as i32);
        b.push_ptr(psum);
        b.push_ptr(sc.cand.buf.as_ptr());
        b.push_ptr(sc.counter.buf.as_ptr());
        b.push_ptr(tau.buf.as_ptr());
        b.push_ptr(z.buf.as_ptr());
        b.push_f32(top_p);
        b.push_i32(top_k as i32);
        b.push_f32(min_p);
        self.launch_kernel_blob("claude_smx_norm", grid, [256, 1, 1], 0, b.as_mut_slice())?;
        let mut b = KernargBlob::new();
        b.push_ptr(probs.buf.as_ptr());
        b.push_i32(n as i32);
        b.push_ptr(tau.buf.as_ptr());
        b.push_ptr(kpart);
        b.push_ptr(sc.counter.buf.as_ptr());
        b.push_ptr(z.buf.as_ptr());
        b.push_f32(top_p);
        b.push_i32(top_k as i32);
        b.push_f32(min_p);
        self.launch_kernel_blob("claude_smx_kept", grid, [256, 1, 1], 0, b.as_mut_slice())
    }

    fn claude_sample_kernels(&mut self) -> HipResult<()> {
        if !self.functions.contains_key("claude_mtp_sample_row") {
            self.ensure_kernel("claude_mtp_sample", SRC, "claude_mtp_sample_row")?;
            self.ensure_kernel("claude_mtp_sample", SRC, "claude_mtp_sample_accept")?;
        }
        Ok(())
    }

    /// Draw one index from the truncated nucleus of `probs[0..n]` (tau / z as
    /// written by `softmax_temp_topp_batched_into_f32`, 1-element views) with
    /// uniform `u`. `out` (4 words) receives `[idx, q(idx), max p, token]`,
    /// token = `vocab_map[idx]` if a map is given. With a doorbell the same
    /// words are rung through the next slot (returned ticket); `pos` / `tok`
    /// store the next draft position / the token into device memory.
    #[allow(clippy::too_many_arguments)]
    pub fn claude_mtp_sample_row(
        &mut self,
        sc: &ClaudeSampleScratch,
        db: Option<&mut ClaudeDoorbell>,
        probs: &GpuTensor,
        n: usize,
        tau: &GpuTensor,
        z: &GpuTensor,
        u: f32,
        out: &GpuTensor,
        vocab_map: *const c_void,
        pos: Option<(*mut c_void, i32)>,
        tok_dst: *mut c_void,
    ) -> HipResult<Option<DoorbellTicket>> {
        self.bind_thread()?;
        self.claude_sample_kernels()?;
        let (slot, seq, ticket) = match db {
            Some(db) => {
                let (a, s, t) = self.claude_doorbell_reserve(db, out, 4);
                (a, s, Some(t))
            }
            None => (0u64, 0u32, None),
        };
        let (pd, pv) = pos.unwrap_or((std::ptr::null_mut(), 0));
        let mut b = KernargBlob::new();
        b.push_ptr(probs.buf.as_ptr());
        b.push_i32(n as i32);
        b.push_ptr(tau.buf.as_ptr());
        b.push_ptr(z.buf.as_ptr());
        b.push_f32(u);
        b.push_ptr(sc.part.buf.as_ptr());
        b.push_ptr(sc.counter.buf.as_ptr());
        b.push_ptr(out.buf.as_ptr());
        b.push_ptr(vocab_map);
        b.push_u64(slot);
        b.push_u32(seq);
        b.push_ptr(pd);
        b.push_i32(pv);
        b.push_ptr(tok_dst);
        self.launch_kernel_blob(
            "claude_mtp_sample_row",
            [grid_for(n), 1, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )?;
        Ok(ticket)
    }

    /// The verify's residual acceptance on the device for `d` drafts
    /// (`d <= 8`): target rows `tprobs [d+1 x v]` (+ tau/z `[d+1]`), draft
    /// rows `dprobs [d x vd]` (+ tau/z `[d]`), draft sample outputs `dout
    /// [d x 4]` from `claude_mtp_sample_row`, `inv_map [v]` (full id -> draft
    /// index or -1) when the draft vocab is compressed, host uniforms
    /// `u[0..=d]`. `out` needs 4 + d + 1 words.
    #[allow(clippy::too_many_arguments)]
    pub fn claude_mtp_sample_accept(
        &mut self,
        sc: &ClaudeSampleScratch,
        db: Option<&mut ClaudeDoorbell>,
        tprobs: &GpuTensor,
        v: usize,
        ttau: &GpuTensor,
        tz: &GpuTensor,
        dprobs: &GpuTensor,
        vd: usize,
        dtau: &GpuTensor,
        dz: &GpuTensor,
        dout: &GpuTensor,
        inv_map: *const c_void,
        d: usize,
        eos: u32,
        u: &[f32],
        out: &GpuTensor,
    ) -> HipResult<Option<DoorbellTicket>> {
        assert!(d <= CLAUDE_ACCEPT_MAX_D && u.len() == d + 1);
        self.bind_thread()?;
        self.claude_sample_kernels()?;
        let (slot, seq, ticket) = match db {
            Some(db) => {
                let (a, s, t) = self.claude_doorbell_reserve(db, out, 4 + d + 1);
                (a, s, Some(t))
            }
            None => (0u64, 0u32, None),
        };
        let mut b = KernargBlob::new();
        b.push_ptr(tprobs.buf.as_ptr());
        b.push_i32(v as i32);
        b.push_ptr(ttau.buf.as_ptr());
        b.push_ptr(tz.buf.as_ptr());
        b.push_ptr(dprobs.buf.as_ptr());
        b.push_i32(vd as i32);
        b.push_ptr(dtau.buf.as_ptr());
        b.push_ptr(dz.buf.as_ptr());
        b.push_ptr(dout.buf.as_ptr());
        b.push_ptr(inv_map);
        b.push_i32(d as i32);
        b.push_u32(eos);
        for i in 0..9 {
            b.push_f32(if i < u.len() { u[i] } else { 0.0 });
        }
        b.push_ptr(sc.part.buf.as_ptr());
        b.push_ptr(sc.counter.buf.as_ptr());
        b.push_ptr(out.buf.as_ptr());
        b.push_u64(slot);
        b.push_u32(seq);
        self.launch_kernel_blob(
            "claude_mtp_sample_accept",
            [grid_for(v), 1, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )?;
        Ok(ticket)
    }
}

/// Decode the accept kernel's words `[n_commit, accept_count, hit_eos,
/// n_consumed, tokens...]`.
pub fn claude_decode_accept(w: &[u32]) -> ClaudeAcceptResult {
    let n = w[0] as usize;
    ClaudeAcceptResult {
        committed: w[4..4 + n].to_vec(),
        accept_count: w[1] as usize,
        hit_eos: w[2] != 0,
        n_consumed: w[3] as usize,
    }
}
