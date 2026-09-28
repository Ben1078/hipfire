// claude: two-stage full-vocab reductions (argmax, top-2 + logsumexp).
// The stock kernels scan a 248K-vocab row with one workgroup (~250-480 us);
// these spread each row over NB blocks. Ties resolve to the lowest index.
#include <hip/hip_runtime.h>

#define CLAUDE_RED_THREADS 256

__device__ __forceinline__ bool claude_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// ---- argmax ---------------------------------------------------------------
// stage 1: grid (NB, B). partial[b*NB + blk] = (max, idx)
extern "C" __global__ void claude_argmax_s1(
    const float* __restrict__ data, float* __restrict__ pv, int* __restrict__ pi, int n, int nb)
{
    __shared__ float sv[CLAUDE_RED_THREADS];
    __shared__ int si[CLAUDE_RED_THREADS];
    const int b = blockIdx.y, blk = blockIdx.x, t = threadIdx.x;
    const int chunk = (n + nb - 1) / nb;
    const int lo = blk * chunk, hi = min(n, lo + chunk);
    const float* row = data + (long long)b * n;
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int i = lo + t; i < hi; i += CLAUDE_RED_THREADS) {
        const float v = row[i];
        if (claude_better(v, i, bv, bi)) { bv = v; bi = i; }
    }
    sv[t] = bv; si[t] = bi;
    __syncthreads();
    for (int s = CLAUDE_RED_THREADS / 2; s > 0; s >>= 1) {
        if (t < s && claude_better(sv[t + s], si[t + s], sv[t], si[t])) { sv[t] = sv[t + s]; si[t] = si[t + s]; }
        __syncthreads();
    }
    if (t == 0) { pv[b * nb + blk] = sv[0]; pi[b * nb + blk] = si[0]; }
}

// stage 2: grid (B), block nb (<= 256, power of two)
extern "C" __global__ void claude_argmax_s2(
    const float* __restrict__ pv, const int* __restrict__ pi, int* __restrict__ result, int nb)
{
    __shared__ float sv[CLAUDE_RED_THREADS];
    __shared__ int si[CLAUDE_RED_THREADS];
    const int b = blockIdx.x, t = threadIdx.x;
    sv[t] = t < nb ? pv[b * nb + t] : -INFINITY;
    si[t] = t < nb ? pi[b * nb + t] : 0x7fffffff;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && claude_better(sv[t + s], si[t + s], sv[t], si[t])) { sv[t] = sv[t + s]; si[t] = si[t + s]; }
        __syncthreads();
    }
    if (t == 0) result[b] = si[0];
}

// ---- top-2 + logsumexp ----------------------------------------------------
struct ClaudeTop2 { float v0; int i0; float v1; int i1; };

__device__ __forceinline__ void claude_top2_push(ClaudeTop2& a, float v, int i) {
    if (claude_better(v, i, a.v0, a.i0)) { a.v1 = a.v0; a.i1 = a.i0; a.v0 = v; a.i0 = i; }
    else if (claude_better(v, i, a.v1, a.i1)) { a.v1 = v; a.i1 = i; }
}

// stage 1: grid (NB, B). part[(b*NB+blk)*6] = v0,i0,v1,i1,lmax,lsum
extern "C" __global__ void claude_top2lse_s1(
    const float* __restrict__ data, float* __restrict__ part, int n, int nb)
{
    __shared__ ClaudeTop2 st[CLAUDE_RED_THREADS];
    __shared__ float smax[CLAUDE_RED_THREADS];
    __shared__ float ssum[CLAUDE_RED_THREADS];
    const int b = blockIdx.y, blk = blockIdx.x, t = threadIdx.x;
    const int chunk = (n + nb - 1) / nb;
    const int lo = blk * chunk, hi = min(n, lo + chunk);
    const float* row = data + (long long)b * n;
    ClaudeTop2 a = {-INFINITY, 0x7fffffff, -INFINITY, 0x7fffffff};
    float m = -INFINITY, sum = 0.0f;
    for (int i = lo + t; i < hi; i += CLAUDE_RED_THREADS) {
        const float v = row[i];
        claude_top2_push(a, v, i);
        if (v > m) { sum = sum * expf(m - v) + 1.0f; m = v; }
        else sum += expf(v - m);
    }
    st[t] = a; smax[t] = m; ssum[t] = sum;
    __syncthreads();
    for (int s = CLAUDE_RED_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) {
            ClaudeTop2 x = st[t];
            const ClaudeTop2 y = st[t + s];
            claude_top2_push(x, y.v0, y.i0);
            claude_top2_push(x, y.v1, y.i1);
            st[t] = x;
            const float m1 = smax[t], m2 = smax[t + s];
            const float mm = fmaxf(m1, m2);
            const float s1 = (m1 == -INFINITY) ? 0.0f : ssum[t] * expf(m1 - mm);
            const float s2 = (m2 == -INFINITY) ? 0.0f : ssum[t + s] * expf(m2 - mm);
            smax[t] = mm; ssum[t] = s1 + s2;
        }
        __syncthreads();
    }
    if (t == 0) {
        float* p = part + (long long)(b * nb + blk) * 6;
        p[0] = st[0].v0; p[1] = __int_as_float(st[0].i0);
        p[2] = st[0].v1; p[3] = __int_as_float(st[0].i1);
        p[4] = smax[0]; p[5] = ssum[0];
    }
}

// stage 2: grid (B), block 1 warp-ish (serial merge of nb partials, nb small)
extern "C" __global__ void claude_top2lse_s2(
    const float* __restrict__ part, int* __restrict__ top_idx, float* __restrict__ top_logp, int nb, int k)
{
    const int b = blockIdx.x;
    if (threadIdx.x != 0) return;
    ClaudeTop2 a = {-INFINITY, 0x7fffffff, -INFINITY, 0x7fffffff};
    float m = -INFINITY, sum = 0.0f;
    for (int j = 0; j < nb; j++) {
        const float* p = part + (long long)(b * nb + j) * 6;
        claude_top2_push(a, p[0], __float_as_int(p[1]));
        claude_top2_push(a, p[2], __float_as_int(p[3]));
        const float pm = p[4], ps = p[5];
        if (pm == -INFINITY) continue;
        if (pm > m) { sum = sum * expf(m - pm) + ps; m = pm; }
        else sum += ps * expf(pm - m);
    }
    const float lse = m + logf(sum);
    top_idx[b * k + 0] = a.i0; top_logp[b * k + 0] = a.v0 - lse;
    if (k > 1) { top_idx[b * k + 1] = a.i1; top_logp[b * k + 1] = a.v1 - lse; }
}
