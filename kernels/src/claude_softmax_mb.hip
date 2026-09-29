// claude r2: multi-block softmax_temp_topp (same outputs and definitions as
// softmax_temp_topp_batched_f32 in softmax_temp_batched.hip, which runs one
// 1024-thread workgroup per row: ~250 us for a 248320-wide row on gfx1100).
// Each row is split over G blocks (grid (G, rows)); four launches:
//   claude_smx_max   : per-block max of logits * inv_t
//   claude_smx_exp   : global max (partials folded in block order, identical in
//                      every block), probs = expf(s - max), per-block sums
//   claude_smx_norm  : global sum (folded in block order), p = e * (1/sum);
//                      pmax = 1/sum (the max element's e is expf(0) = 1);
//                      512-bin histogram (mass, count) of the p >= pmax * 1e-5
//                      tail (block LDS histogram, then float atomics into the
//                      row histogram); the last block of the row derives
//                      tau = max(tau_p, tau_k, min_p * pmax) by the original
//                      top-down scan and clears the row histogram
//   claude_smx_kept  : Z = sum of p >= tau (partials folded in block order)
// Disabled nucleus (top_p >= 1, top_k <= 0, min_p <= 0): tau = 0, Z = 1.
// Float sums use a different association than the single-block kernel, so p
// differs in the last bits (and a prob sitting exactly on a histogram bucket
// edge can move one bucket); the histogram itself was never order-exact (LDS
// float atomics in the original too). The definitions of p, tau, Z are unchanged.
#include <hip/hip_runtime.h>

#define SMX_B 256
#define SMX_NBINS 512
#define SMX_MAXG 1024

__device__ __forceinline__ float smx_bsum(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = SMX_B / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float smx_bmax(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = SMX_B / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// Stage G partials into LDS with all threads (one round trip), then thread 0
// folds them in block order. op 0 = sum, 1 = max.
__device__ __forceinline__ float smx_fold(const float* part, int G, int op, float* stage) {
    for (int k = threadIdx.x; k < G; k += SMX_B) stage[k] = part[k];
    __syncthreads();
    __shared__ float s_r;
    if (threadIdx.x == 0) {
        float r = op ? -1e30f : 0.0f;
        for (int k = 0; k < G; k++) r = op ? fmaxf(r, stage[k]) : r + stage[k];
        s_r = r;
    }
    __syncthreads();
    const float r = s_r;
    __syncthreads();
    return r;
}

extern "C" __global__ __launch_bounds__(SMX_B)
void claude_smx_max(const float* __restrict__ logits, int n, float inv_t, float* __restrict__ pmax_part)
{
    __shared__ float red[SMX_B];
    const int G = gridDim.x, b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const float* in = logits + (size_t)row * n;
    const int chunk = (n + G - 1) / G, lo = b * chunk, hi = min(n, lo + chunk);
    float m = -1e30f;
    for (int i = lo + t; i < hi; i += SMX_B) m = fmaxf(m, in[i] * inv_t);
    m = smx_bmax(m, red);
    if (t == 0) pmax_part[row * G + b] = m;
}

extern "C" __global__ __launch_bounds__(SMX_B)
void claude_smx_exp(const float* __restrict__ logits, float* __restrict__ probs, int n, float inv_t,
                    const float* __restrict__ pmax_part, float* __restrict__ psum_part)
{
    __shared__ float red[SMX_B];
    __shared__ float stage[SMX_MAXG];
    const int G = gridDim.x, b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const float mx = smx_fold(pmax_part + row * G, G, 1, stage);
    const float* in = logits + (size_t)row * n;
    float* out = probs + (size_t)row * n;
    const int chunk = (n + G - 1) / G, lo = b * chunk, hi = min(n, lo + chunk);
    float s = 0.0f;
    for (int i = lo + t; i < hi; i += SMX_B) {
        const float e = expf(in[i] * inv_t - mx);
        out[i] = e;
        s += e;
    }
    s = smx_bsum(s, red);
    if (t == 0) psum_part[row * G + b] = s;
}

extern "C" __global__ __launch_bounds__(SMX_B)
void claude_smx_norm(float* __restrict__ probs, int n, const float* __restrict__ psum_part,
                     float* __restrict__ ghist, unsigned int* __restrict__ counter,
                     float* __restrict__ tau_out, float* __restrict__ z_out,
                     float top_p, int top_k, float min_p)
{
    __shared__ float stage[SMX_MAXG];
    __shared__ float shist[SMX_NBINS];
    __shared__ float scnt[SMX_NBINS];
    __shared__ unsigned int s_ticket;
    const int G = gridDim.x, b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const float sum = smx_fold(psum_part + row * G, G, 0, stage);
    for (int k = t; k < SMX_NBINS; k += SMX_B) { shist[k] = 0.0f; scnt[k] = 0.0f; }
    __syncthreads();
    const float inv_sum = 1.0f / sum;
    const float pmax = inv_sum;   // the max element's e is expf(0) = 1
    const bool disabled = (top_p >= 1.0f && top_k <= 0 && min_p <= 0.0f);
    float* out = probs + (size_t)row * n;
    const int chunk = (n + G - 1) / G, lo = b * chunk, hi = min(n, lo + chunk);
    const float floor_p = pmax * 1e-5f;
    const float inv_pmax_bins = (pmax > 0.0f) ? ((float)SMX_NBINS / pmax) : 0.0f;
    for (int i = lo + t; i < hi; i += SMX_B) {
        const float p = out[i] * inv_sum;
        out[i] = p;
        if (!disabled && p >= floor_p) {
            int bin = (int)(p * inv_pmax_bins);
            if (bin >= SMX_NBINS) bin = SMX_NBINS - 1;
            atomicAdd(&shist[bin], p);
            atomicAdd(&scnt[bin], 1.0f);
        }
    }
    __syncthreads();
    if (disabled) {
        if (b == 0 && t == 0) { tau_out[row] = 0.0f; z_out[row] = 1.0f; }
        return;
    }
    float* gh = ghist + (size_t)row * (2 * SMX_NBINS);
    for (int k = t; k < SMX_NBINS; k += SMX_B) {
        if (shist[k] != 0.0f) atomicAdd(&gh[k], shist[k]);
        if (scnt[k] != 0.0f) atomicAdd(&gh[SMX_NBINS + k], scnt[k]);
    }
    __threadfence();
    __syncthreads();
    if (t == 0) s_ticket = atomicAdd(&counter[row], 1u);
    __syncthreads();
    if (s_ticket != (unsigned int)(G - 1)) return;
    __threadfence();
    for (int k = t; k < SMX_NBINS; k += SMX_B) {
        shist[k] = __hip_atomic_load(&gh[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        scnt[k] = __hip_atomic_load(&gh[SMX_NBINS + k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        gh[k] = 0.0f;               // self-clear for the next launch
        gh[SMX_NBINS + k] = 0.0f;
    }
    __syncthreads();
    if (t == 0) {
        float cum_mass = 0.0f, cum_cnt = 0.0f, tau_p = 0.0f, tau_k = 0.0f;
        bool found_p = (top_p >= 1.0f);
        bool found_k = (top_k <= 0);
        for (int bb = SMX_NBINS - 1; bb >= 0 && !(found_p && found_k); --bb) {
            cum_mass += shist[bb];
            cum_cnt += scnt[bb];
            if (!found_p && cum_mass >= top_p) { tau_p = ((float)bb / (float)SMX_NBINS) * pmax; found_p = true; }
            if (!found_k && cum_cnt >= (float)top_k) { tau_k = ((float)bb / (float)SMX_NBINS) * pmax; found_k = true; }
        }
        const float tau_minp = (min_p > 0.0f) ? (min_p * pmax) : 0.0f;
        tau_out[row] = fmaxf(fmaxf(tau_p, tau_k), tau_minp);
        counter[row] = 0u;
    }
}

extern "C" __global__ __launch_bounds__(SMX_B)
void claude_smx_kept(const float* __restrict__ probs, int n, const float* __restrict__ tau_in,
                     float* __restrict__ kpart, unsigned int* __restrict__ counter,
                     float* __restrict__ z_out, float top_p, int top_k, float min_p)
{
    __shared__ float red[SMX_B];
    __shared__ float stage[SMX_MAXG];
    __shared__ unsigned int s_ticket;
    const int G = gridDim.x, b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    if (top_p >= 1.0f && top_k <= 0 && min_p <= 0.0f) return;   // Z = 1 written by norm
    const float tau = tau_in[row];
    const float* in = probs + (size_t)row * n;
    const int chunk = (n + G - 1) / G, lo = b * chunk, hi = min(n, lo + chunk);
    float kept = 0.0f;
    for (int i = lo + t; i < hi; i += SMX_B) {
        const float p = in[i];
        if (p >= tau) kept += p;
    }
    kept = smx_bsum(kept, red);
    if (t == 0) {
        kpart[row * G + b] = kept;
        __threadfence();
        s_ticket = atomicAdd(&counter[row], 1u);
    }
    __syncthreads();
    if (s_ticket != (unsigned int)(G - 1)) return;
    __threadfence();
    const float z = smx_fold(kpart + row * G, G, 0, stage);
    if (t == 0) {
        z_out[row] = z;
        counter[row] = 0u;
    }
}
