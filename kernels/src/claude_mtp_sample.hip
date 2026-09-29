// claude r2 (CLAUDE_MTP_GPU_SAMPLE): on-device sampling for the sampled MTP
// draft and the residual acceptance of the verify, with the result rung
// through the MTP doorbell slot (see claude_mtp_doorbell.hip).
//
// Both kernels consume the rows written by softmax_temp_topp_batched_f32
// (probs p, nucleus threshold tau, kept mass Z) and apply the SAME truncation
// as the host `apply_topp_trunc`:  q(p) = tau <= 0 ? p : (p >= tau ? p * (1/max(Z, FLT_MIN)) : 0).
//
// Sampling from a row of non-negative weights w with a uniform u in [0,1):
// multi-block, one pass. Block b owns the contiguous chunk
// [b*chunk, min(n, (b+1)*chunk)) and writes its partial sum (fixed tree order);
// the last block to finish (atomic ticket, self-resetting counter) sums the
// partials in block order, picks the first block whose inclusive cumulative
// weight exceeds u*total, then scans that chunk (16 contiguous elements per
// thread, run sums prefix-summed by thread 0) for the first element whose
// cumulative weight exceeds the target. This is inverse-CDF sampling of the
// same categorical distribution w_i / sum(w) that the host's sequential
// `sample_categorical` / `sample_residual` draw from; only the float
// summation order differs (a boundary shift of ~1e-7 relative), so the same u
// selects the same index except within rounding distance of a boundary.
// If rounding leaves the target past the last crossing, the last element
// with positive weight in the scanned range is taken (the host would return
// n-1, which may carry zero weight).
#include <hip/hip_runtime.h>
#include <float.h>

#define CS_BLOCK 256
#define CS_RUN 16
#define CS_CHUNK (CS_BLOCK * CS_RUN)

__device__ __forceinline__ float cs_trunc(float p, float tau, float inv_z) {
    return tau <= 0.0f ? p : (p >= tau ? p * inv_z : 0.0f);
}

__device__ __forceinline__ float cs_inv_z(float z) {
    return 1.0f / fmaxf(z, FLT_MIN);
}

// Doorbell publish of `n` words from shared/registers: payload then seq.
__device__ __forceinline__ void cs_ring(unsigned long long slot_addr, unsigned int seq,
                                        const unsigned int* words, int n) {
    if (!slot_addr) return;
    unsigned int* slot = (unsigned int*)slot_addr;
    for (int i = 0; i < n; i++)
        __hip_atomic_store(slot + 1 + i, words[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __threadfence_system();
    __hip_atomic_store(slot, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Block-wide reduction helpers (fixed order: LDS tree).
__device__ __forceinline__ float cs_block_sum(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = CS_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float cs_block_max(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = CS_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// (max value, first index) with strict '>' in index order.
__device__ __forceinline__ void cs_block_argmax(float& v, int& idx, float* rv, int* ri) {
    const int t = threadIdx.x;
    rv[t] = v; ri[t] = idx;
    __syncthreads();
    for (int s = CS_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
            const float ov = rv[t + s]; const int oi = ri[t + s];
            if (ov > rv[t] || (ov == rv[t] && oi < ri[t])) { rv[t] = ov; ri[t] = oi; }
        }
        __syncthreads();
    }
    v = rv[0]; idx = ri[0];
    __syncthreads();
}

// Weight of element i in the mode of the accept kernel.
struct CsRow {
    const float* p; float tau; float inv_z;          // target (or draft) row
    const float* dp; float dtau; float dinv_z;        // residual: draft row (null otherwise)
    const int* inv_map;                               // residual: full id -> draft idx (null = identity)
};

__device__ __forceinline__ float cs_weight(const CsRow& r, int i) {
    const float pt = cs_trunc(r.p[i], r.tau, r.inv_z);
    if (!r.dp) return pt;
    float pd = 0.0f;
    if (r.inv_map) {
        const int j = r.inv_map[i];
        if (j >= 0) pd = cs_trunc(r.dp[j], r.dtau, r.dinv_z);
    } else {
        pd = cs_trunc(r.dp[i], r.dtau, r.dinv_z);
    }
    const float d = pt - pd;
    return d > 0.0f ? d : 0.0f;
}

// The last block's selection: given partial sums part_sum[0..G), total and
// target, find the element. Returns the index (all threads get it).
__device__ int cs_select(const CsRow& r, int n, int G, const float* part_sum, float target,
                         float* red, int* sh_i) {
    const int t = threadIdx.x;
    __shared__ int s_blk;
    __shared__ float s_base;
    if (t == 0) {
        float cum = 0.0f;
        int bsel = -1;
        float base = 0.0f;
        int last_pos = -1;
        float base_last = 0.0f;
        for (int b = 0; b < G; b++) {
            const float w = part_sum[b];
            if (w > 0.0f) { last_pos = b; base_last = cum; }
            if (bsel < 0 && target < cum + w && w > 0.0f) { bsel = b; base = cum; }
            cum += w;
        }
        if (bsel < 0) { bsel = last_pos < 0 ? G - 1 : last_pos; base = base_last; }
        s_blk = bsel; s_base = base;
    }
    __syncthreads();
    const int chunk = (n + G - 1) / G;
    const int lo = s_blk * chunk;
    const int hi = min(n, lo + chunk);
    // contiguous runs of CS_RUN elements per thread
    const int rlo = lo + t * CS_RUN;
    float s = 0.0f;
    for (int e = 0; e < CS_RUN; e++) {
        const int i = rlo + e;
        if (i < hi) s += cs_weight(r, i);
    }
    red[t] = s;
    __syncthreads();
    if (t == 0) {
        float cum = s_base;
        int tsel = -1;
        float tbase = 0.0f;
        int tlast = -1;
        float tbase_last = 0.0f;
        for (int k = 0; k < CS_BLOCK; k++) {
            const float w = red[k];
            if (w > 0.0f) { tlast = k; tbase_last = cum; }
            if (tsel < 0 && w > 0.0f && target < cum + w) { tsel = k; tbase = cum; }
            cum += w;
        }
        if (tsel < 0) { tsel = tlast < 0 ? 0 : tlast; tbase = tbase_last; }
        // scan the selected run sequentially
        const int base_i = lo + tsel * CS_RUN;
        float acc = tbase;
        int pick = -1, lastnz = -1;
        for (int e = 0; e < CS_RUN; e++) {
            const int i = base_i + e;
            if (i >= hi) break;
            const float w = cs_weight(r, i);
            if (w > 0.0f) lastnz = i;
            acc += w;
            if (pick < 0 && w > 0.0f && target < acc) pick = i;
        }
        if (pick < 0) pick = lastnz >= 0 ? lastnz : max(lo, min(base_i, hi - 1));
        sh_i[0] = pick;
    }
    __syncthreads();
    const int pick = sh_i[0];
    __syncthreads();
    return pick;
}

// ---------------------------------------------------------------------------
// Draft step: sample one index from the truncated nucleus of `probs` (n wide)
// with the host-drawn uniform `u`. out = [idx, q(idx) bits, max_i p_i bits,
// token] (token = vocab_map[idx] when a map is given). Rings the doorbell with
// the same 4 words; optional stores: *pos_dst = pos_val (next draft position),
// *tok_dst = token (verify token staging).
extern "C" __global__ __launch_bounds__(CS_BLOCK)
void claude_mtp_sample_row(
    const float* __restrict__ probs, int n,
    const float* __restrict__ tau_p, const float* __restrict__ z_p, float u,
    float* __restrict__ part, unsigned int* __restrict__ counter,
    unsigned int* __restrict__ out, const unsigned int* __restrict__ vocab_map,
    unsigned long long slot_addr, unsigned int seq,
    int* __restrict__ pos_dst, int pos_val, int* __restrict__ tok_dst)
{
    __shared__ float red[CS_BLOCK];
    __shared__ int sh_i[2];
    __shared__ unsigned int s_ticket;
    const int t = threadIdx.x, b = blockIdx.x, G = gridDim.x;
    const float tau = tau_p[0];
    const float inv_z = cs_inv_z(z_p[0]);
    CsRow r = {probs, tau, inv_z, nullptr, 0.0f, 0.0f, nullptr};
    const int chunk = (n + G - 1) / G;
    const int lo = b * chunk, hi = min(n, lo + chunk);
    float s = 0.0f, m = 0.0f;
    for (int i = lo + t; i < hi; i += CS_BLOCK) {
        const float p = probs[i];
        m = fmaxf(m, p);
        s += cs_trunc(p, tau, inv_z);
    }
    s = cs_block_sum(s, red);
    m = cs_block_max(m, red);
    if (t == 0) {
        part[2 * b] = s;
        part[2 * b + 1] = m;
        __threadfence();
        s_ticket = atomicAdd(counter, 1u);
    }
    __syncthreads();
    if (s_ticket != (unsigned int)(G - 1)) return;
    __threadfence();
    __shared__ float s_psum[1024];
    __shared__ float s_pmax[1024];
    __shared__ float s_total, s_top;
    // all threads stage the partials (one global round trip), thread 0 then
    // folds them in block order from LDS
    for (int k = t; k < G; k += CS_BLOCK) {
        s_psum[k] = __hip_atomic_load(&part[2 * k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_pmax[k] = __hip_atomic_load(&part[2 * k + 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (t == 0) {
        float tot = 0.0f, top = 0.0f;
        for (int k = 0; k < G; k++) {
            tot += s_psum[k];
            top = fmaxf(top, s_pmax[k]);
        }
        s_total = tot; s_top = top;
        *counter = 0u;   // self-reset for the next launch (stream-ordered)
    }
    __syncthreads();
    const float target = u * s_total;
    const int idx = cs_select(r, n, G, s_psum, target, red, sh_i);
    if (t == 0) {
        const float q = cs_trunc(probs[idx], tau, inv_z);
        const unsigned int tok = vocab_map ? vocab_map[idx] : (unsigned int)idx;
        unsigned int w[4] = {(unsigned int)idx, __float_as_uint(q), __float_as_uint(s_top), tok};
        out[0] = w[0]; out[1] = w[1]; out[2] = w[2]; out[3] = w[3];
        if (pos_dst) *pos_dst = pos_val;
        if (tok_dst) *tok_dst = (int)tok;
        cs_ring(slot_addr, seq, w, 4);
    }
}

// ---------------------------------------------------------------------------
// Verify: the residual acceptance of `mtp_sampled_accept`, entirely on the
// device. Every block evaluates the (cheap) accept chain redundantly with the
// host-drawn uniforms U[0..=d]:
//   k = 0..d-1: q_t = trunc_t(k)[cand_k], p_d = max(q_d(k), FLT_MIN), u = U[j++];
//               u * p_d <= q_t -> accept (EOS candidate: stop, no bonus);
//               else reject: bonus ~ (trunc_t(k) - trunc_d(k))+ with U[j++]
//   all accepted (no EOS): bonus ~ trunc_t(d) with U[j++]
// then all blocks cooperate on the one bonus draw (same sampler as above;
// residual with zero mass falls back to the first argmax of trunc_t(k), as
// the host's sample_residual does).
// out = [n_commit, accept_count, hit_eos, n_consumed, tok_0 .. tok_{n_commit-1}],
// also rung through the doorbell (4 + n_commit words).
struct CsU { float u[9]; };

extern "C" __global__ __launch_bounds__(CS_BLOCK)
void claude_mtp_sample_accept(
    const float* __restrict__ tprobs, int V,
    const float* __restrict__ ttau, const float* __restrict__ tzz,
    const float* __restrict__ dprobs, int Vd,
    const float* __restrict__ dtau, const float* __restrict__ dzz,
    const unsigned int* __restrict__ dout, const int* __restrict__ inv_map,
    int d, unsigned int eos, CsU U,
    float* __restrict__ part, unsigned int* __restrict__ counter,
    unsigned int* __restrict__ out,
    unsigned long long slot_addr, unsigned int seq)
{
    __shared__ float red[CS_BLOCK];
    __shared__ int redi[CS_BLOCK];
    __shared__ int sh_i[2];
    __shared__ unsigned int s_ticket;
    const int t = threadIdx.x, b = blockIdx.x, G = gridDim.x;

    // ---- accept chain (identical in every thread) ----
    int acc = 0, used = 0, hit = 0, mode = 0, row = -1;   // mode 0 none, 1 residual, 2 target
    float ub = 0.0f;
    for (int k = 0; k < d; k++) {
        const unsigned int cand = dout[4 * k + 3];
        const float pd = fmaxf(__uint_as_float(dout[4 * k + 1]), FLT_MIN);
        const float qt = cs_trunc(tprobs[(size_t)k * V + cand], ttau[k], cs_inv_z(tzz[k]));
        const float u = U.u[used++];
        if (u * pd <= qt) {
            acc++;
            if (cand == eos) { hit = 1; break; }
        } else {
            mode = 1; row = k; ub = U.u[used++];
            break;
        }
    }
    if (mode == 0 && !hit) { mode = 2; row = d; ub = U.u[used++]; }

    if (mode == 0) {
        // EOS accepted: no bonus draw.
        if (b == 0 && t == 0) {
            unsigned int w[4 + 9];
            w[0] = (unsigned int)acc; w[1] = (unsigned int)acc; w[2] = 1u; w[3] = (unsigned int)used;
            for (int k = 0; k < acc; k++) w[4 + k] = dout[4 * k + 3];
            for (int k = 0; k < 4 + acc; k++) out[k] = w[k];
            cs_ring(slot_addr, seq, w, 4 + acc);
        }
        return;
    }

    CsRow r;
    r.p = tprobs + (size_t)row * V;
    r.tau = ttau[row];
    r.inv_z = cs_inv_z(tzz[row]);
    if (mode == 1) {
        r.dp = dprobs + (size_t)row * Vd;
        r.dtau = dtau[row];
        r.dinv_z = cs_inv_z(dzz[row]);
        r.inv_map = inv_map;
    } else {
        r.dp = nullptr; r.dtau = 0.0f; r.dinv_z = 0.0f; r.inv_map = nullptr;
    }

    // ---- pass 1: partial weight sums (+ residual fallback argmax of trunc_t) ----
    const int chunk = (V + G - 1) / G;
    const int lo = b * chunk, hi = min(V, lo + chunk);
    float s = 0.0f, am_v = -1.0f;
    int am_i = 0x7fffffff;
    for (int i = lo + t; i < hi; i += CS_BLOCK) {
        s += cs_weight(r, i);
        if (mode == 1) {
            const float q = cs_trunc(r.p[i], r.tau, r.inv_z);
            if (q > am_v || (q == am_v && i < am_i)) { am_v = q; am_i = i; }
        }
    }
    s = cs_block_sum(s, red);
    if (mode == 1) cs_block_argmax(am_v, am_i, red, redi);
    if (t == 0) {
        part[3 * b] = s;
        part[3 * b + 1] = am_v;
        part[3 * b + 2] = __int_as_float(am_i);
        __threadfence();
        s_ticket = atomicAdd(counter, 1u);
    }
    __syncthreads();
    if (s_ticket != (unsigned int)(G - 1)) return;
    __threadfence();

    __shared__ float s_psum[1024];
    __shared__ float s_amv[1024];
    __shared__ int s_ami[1024];
    __shared__ float s_total;
    __shared__ int s_am;
    for (int k = t; k < G; k += CS_BLOCK) {
        s_psum[k] = __hip_atomic_load(&part[3 * k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_amv[k] = __hip_atomic_load(&part[3 * k + 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_ami[k] = __float_as_int(__hip_atomic_load(&part[3 * k + 2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
    __syncthreads();
    if (t == 0) {
        float tot = 0.0f, bv = -1.0f;
        int bi = 0x7fffffff;
        for (int k = 0; k < G; k++) {
            tot += s_psum[k];
            const float v = s_amv[k];
            const int i = s_ami[k];
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        s_total = tot; s_am = bi;
        *counter = 0u;
    }
    __syncthreads();
    int bonus;
    if (mode == 1 && !(s_total > 0.0f)) {
        bonus = s_am;   // host: sum <= 0 -> argmax_u32(target row)
    } else {
        bonus = cs_select(r, V, G, s_psum, ub * s_total, red, sh_i);
    }
    if (t == 0) {
        unsigned int w[4 + 9];
        const int n_commit = acc + 1;
        w[0] = (unsigned int)n_commit; w[1] = (unsigned int)acc;
        w[2] = ((unsigned int)bonus == eos) ? 1u : 0u; w[3] = (unsigned int)used;
        for (int k = 0; k < acc; k++) w[4 + k] = dout[4 * k + 3];
        w[4 + acc] = (unsigned int)bonus;
        for (int k = 0; k < 4 + n_commit; k++) out[k] = w[k];
        cs_ring(slot_addr, seq, w, 4 + n_commit);
    }
}
