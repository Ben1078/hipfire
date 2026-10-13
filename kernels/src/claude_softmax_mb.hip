// claude r2: multi-block softmax_temp_topp (same outputs and definitions as
// softmax_temp_topp_batched_f32 in softmax_temp_batched.hip, which runs one
// 1024-thread workgroup per row: ~250 us for a 248320-wide row on gfx1100).
// Each row is split over G blocks (grid (G, rows)); four launches:
//   claude_smx_max   : per-block max of logits * inv_t
//   claude_smx_exp   : global max (partials folded in block order, identical in
//                      every block), probs = expf(s - max), per-block sums
//   claude_smx_norm  : global sum (folded in block order), p = e * (1/sum);
//                      pmax = 1/sum (the max element's e is expf(0) = 1).
//                      Truncation = the AR sampler's law, exactly as the
//                      single-block kernel (upstream 7a0c49444):
//                        cap = top_k (the caller's AR candidate cap;
//                              top_k <= 0 or > 64 -> 64), capped at n;
//                        the cap largest probabilities, sorted descending;
//                        min_p > 0: cap shrinks to the first rank >= 1 below
//                              min_p * p_max;
//                        nucleus over the pool's OWN mass: shortest prefix
//                              whose cumulative mass reaches top_p * mass;
//                        tau = prob of the last kept token (>= FLT_MIN).
//                      Each block radix-selects the exact top-cap VALUES of
//                      its chunk into `cand` (the global top-cap is contained
//                      in the union of the per-block top-caps; only values
//                      matter for tau / Z); the last block of the row
//                      radix-selects the exact top-cap of that union and
//                      derives tau.
//   claude_smx_kept  : Z = sum of p >= tau (partials folded in block order)
// Float sums use a different association than the single-block kernel, so p
// (and hence tau / Z) can differ in the last bits; the definitions are the same.
#include <hip/hip_runtime.h>

#define SMX_B 256
#define SMX_MAXG 1024
#define SMX_CAP 64       // max AR candidate cap (llama::ar_candidate_cap)
#define SMX_PER_T 16     // chunk <= 4096 = SMX_B * SMX_PER_T (host CHUNK)

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

// One 8-bit digit pass of an exact radix select over non-negative float bit
// patterns (MSB first). Every thread adds its candidates' digits to `hist`
// before the call; thread 0 picks the digit holding the `*s_need`-th largest
// and leaves in *s_need how many are still to take at the new prefix.
__device__ __forceinline__ void smx_pick_digit(const unsigned int* hist, unsigned int* s_digit,
                                               unsigned int* s_need)
{
    if (threadIdx.x == 0) {
        unsigned int nd = *s_need, above = 0u;
        int d = 255;
        for (; d > 0; --d) {
            if (above + hist[d] >= nd) break;
            above += hist[d];
        }
        *s_digit = (unsigned int)d;
        *s_need = nd - above;
    }
}

extern "C" __global__ __launch_bounds__(SMX_B)
void claude_smx_norm(float* __restrict__ probs, int n, const float* __restrict__ psum_part,
                     float* __restrict__ cand, unsigned int* __restrict__ counter,
                     float* __restrict__ tau_out, float* __restrict__ z_out,
                     float top_p, int top_k, float min_p)
{
    __shared__ float stage[SMX_MAXG];
    __shared__ unsigned int s_hist[256];
    __shared__ unsigned int s_digit, s_need, s_cnt, s_ngt, s_neq, s_ticket;
    __shared__ float s_cand[SMX_CAP];
    const int G = gridDim.x, b = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const float sum = smx_fold(psum_part + row * G, G, 0, stage);
    const float inv_sum = 1.0f / sum;
    const float pmax = inv_sum;   // the max element's e is expf(0) = 1
    float* out = probs + (size_t)row * n;
    const int chunk = (n + G - 1) / G, lo = b * chunk, hi = min(n, lo + chunk);
    int cap = (top_k <= 0 || top_k > SMX_CAP) ? SMX_CAP : top_k;
    if (cap > n) cap = n;
    (void)z_out;   // Z is written by claude_smx_kept

    // ---- normalize; this thread's probs stay in registers ----
    float pv[SMX_PER_T];
    bool ok[SMX_PER_T];
#pragma unroll
    for (int k = 0; k < SMX_PER_T; ++k) {
        const int i = lo + t + k * SMX_B;
        ok[k] = i < hi;
        float p = 0.0f;
        if (ok[k]) {
            p = out[i] * inv_sum;
            out[i] = p;
        }
        pv[k] = p;
    }

    // ---- per block: the exact top-cap_b probability values of the chunk ----
    const int len_b = max(0, hi - lo);
    const int cap_b = min(cap, len_b);
    float* cb = cand + ((size_t)row * G + b) * SMX_CAP;
    if (cap_b > 0) {
        // Tokens below pmax * 2^-20 would only pile atomics on a few low bins;
        // skip them when at least cap_b tokens of the chunk clear the floor
        // (then none of them can be among the chunk's top cap_b).
        const float floor_p = pmax * (1.0f / 1048576.0f);
        if (t == 0) { s_cnt = 0u; s_need = (unsigned int)cap_b; s_ngt = 0u; s_neq = 0u; }
        __syncthreads();
        unsigned int c = 0u;
#pragma unroll
        for (int k = 0; k < SMX_PER_T; ++k) c += (ok[k] && pv[k] >= floor_p) ? 1u : 0u;
        atomicAdd(&s_cnt, c);
        __syncthreads();
        const unsigned int floor_bits = (s_cnt >= (unsigned int)cap_b) ? __float_as_uint(floor_p) : 0u;
        unsigned int prefix = 0u, mask = 0u;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int k = t; k < 256; k += SMX_B) s_hist[k] = 0u;
            __syncthreads();
#pragma unroll
            for (int k = 0; k < SMX_PER_T; ++k) {
                const unsigned int bits = __float_as_uint(pv[k]);
                if (ok[k] && bits >= floor_bits && (bits & mask) == prefix)
                    atomicAdd(&s_hist[(bits >> shift) & 255u], 1u);
            }
            __syncthreads();
            smx_pick_digit(s_hist, &s_digit, &s_need);
            __syncthreads();
            prefix |= s_digit << shift;
            mask |= 255u << shift;
            __syncthreads();
        }
        const float vk = __uint_as_float(prefix);   // exact cap_b-th largest of the chunk
        const unsigned int ties = s_need;
        const unsigned int n_gt = (unsigned int)cap_b - ties;
#pragma unroll
        for (int k = 0; k < SMX_PER_T; ++k) {
            if (!ok[k]) continue;
            if (pv[k] > vk) {
                const unsigned int slot = atomicAdd(&s_ngt, 1u);
                cb[slot] = pv[k];
            } else if (pv[k] == vk) {
                const unsigned int slot = atomicAdd(&s_neq, 1u);
                if (slot < ties) cb[n_gt + slot] = pv[k];
            }
        }
    }
    // Unused slots read as probability 0: they can only be picked when the
    // row's real values run out, which cap <= n rules out.
    for (int k = cap_b + t; k < SMX_CAP; k += SMX_B) cb[k] = 0.0f;
    __threadfence();
    __syncthreads();
    if (t == 0) s_ticket = atomicAdd(&counter[row], 1u);
    __syncthreads();
    if (s_ticket != (unsigned int)(G - 1)) return;
    __threadfence();

    // ---- last block of the row: exact top-cap of the union, AR's cut ----
    float* rc = cand + (size_t)row * G * SMX_CAP;
    const int m_all = G * SMX_CAP;
    if (t == 0) { s_need = (unsigned int)cap; s_ngt = 0u; s_neq = 0u; }
    __syncthreads();
    unsigned int prefix = 0u, mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int k = t; k < 256; k += SMX_B) s_hist[k] = 0u;
        __syncthreads();
        for (int k = t; k < m_all; k += SMX_B) {
            const unsigned int bits = __float_as_uint(
                __hip_atomic_load(&rc[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
            if ((bits & mask) == prefix) atomicAdd(&s_hist[(bits >> shift) & 255u], 1u);
        }
        __syncthreads();
        smx_pick_digit(s_hist, &s_digit, &s_need);
        __syncthreads();
        prefix |= s_digit << shift;
        mask |= 255u << shift;
        __syncthreads();
    }
    const float vk = __uint_as_float(prefix);   // exact cap-th largest prob of the row
    const unsigned int ties = s_need;
    const unsigned int n_gt = (unsigned int)cap - ties;
    for (int k = t; k < m_all; k += SMX_B) {
        const float p = __hip_atomic_load(&rc[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (p > vk) {
            const unsigned int slot = atomicAdd(&s_ngt, 1u);
            s_cand[slot] = p;
        } else if (p == vk) {
            const unsigned int slot = atomicAdd(&s_neq, 1u);
            if (slot < ties) s_cand[n_gt + slot] = p;
        }
    }
    __syncthreads();
    if (t == 0) {
        // Insertion sort, descending (values only: ids never reach tau / Z).
        for (int a = 1; a < cap; ++a) {
            const float p = s_cand[a];
            int j = a;
            while (j > 0 && s_cand[j - 1] < p) { s_cand[j] = s_cand[j - 1]; --j; }
            s_cand[j] = p;
        }
        int kept_cap = cap;
        if (min_p > 0.0f) {
            const float floor_min = fminf(min_p, 1.0f) * s_cand[0];
            for (int a = 1; a < cap; ++a) {
                if (s_cand[a] < floor_min) { kept_cap = a; break; }
            }
        }
        float mass = 0.0f;
        for (int a = 0; a < kept_cap; ++a) mass += s_cand[a];
        const float threshold = fminf(fmaxf(top_p, 0.0f), 1.0f) * mass;
        float cum = 0.0f;
        int m = kept_cap;
        for (int a = 0; a < kept_cap; ++a) {
            cum += s_cand[a];
            if (cum >= threshold) { m = a + 1; break; }
        }
        // A zero (underflowed) last kept prob must not read as "no cut".
        tau_out[row] = fmaxf(s_cand[m - 1], 1.17549435e-38f);
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
    (void)top_p; (void)top_k; (void)min_p;   // tau always set by claude_smx_norm
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
