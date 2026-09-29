// claude C6: claude_top2lse_s2 (kernels/src/claude_reduce.hip) with the same
// serial merges, restructured so that only the dependent parts stay serial.
//
// The original stage 2 runs on one thread: 128 iterations, each doing two
// top-2 pushes and one online-logsumexp step `if (pm > m) { sum = sum *
// expf(m - pm) + ps; m = pm; } else sum += ps * expf(pm - m);` (~24 us, bound
// by per-iteration load latency and the expf in the dependency chain). The
// top-2 pushes and the logsumexp never read each other's state, and each
// expf argument depends only on the running max m. So, with every
// floating-point operation and its order unchanged:
//   1. each thread copies one partial into LDS, as six arrays (exact copies);
//   2. thread 0 walks j = 0..NB-1 through the original max chain and records,
//      per partial, which branch the original takes and its expf argument
//      (`m - pm` or `pm - m`: the same subtraction on the same operands);
//      concurrently thread 32 (another wave) runs the original top-2 push
//      sequence in the same j order;
//   3. every thread evaluates the expf of its partial (same function, same
//      input => same value);
//   4. thread 0 replays the sum recurrence in the original order with the
//      same expressions (`sum * e + ps`, `sum += ps * e`), then
//      `lse = m + logf(sum)` and the outputs as before.
// No tree merge anywhere, so the logsumexp (p_min) rounding and the top-2
// tie-breaks are those of the original kernel. The fixed trip count (NB = 128,
// the launcher's CLAUDE_RED_NB) and the array layout let the serial loops be
// unrolled with their LDS loads issued ahead of the dependency chain.
//
// Separate translation unit so claude_reduce.hip (and every kernel compiled
// from it) stays byte-identical. Grid (B), block NB threads.
#include <hip/hip_runtime.h>

#define CLAUDE_TOP2_NB 128

__device__ __forceinline__ bool claude_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

struct ClaudeTop2 { float v0; int i0; float v1; int i1; };

__device__ __forceinline__ void claude_top2_push(ClaudeTop2& a, float v, int i) {
    if (claude_better(v, i, a.v0, a.i0)) { a.v1 = a.v0; a.i1 = a.i0; a.v0 = v; a.i0 = i; }
    else if (claude_better(v, i, a.v1, a.i1)) { a.v1 = v; a.i1 = i; }
}

extern "C" __global__ __launch_bounds__(CLAUDE_TOP2_NB)
void claude_top2lse_s2_lds(
    const float* __restrict__ part, int* __restrict__ top_idx, float* __restrict__ top_logp, int nb, int k)
{
    __shared__ float s_v0[CLAUDE_TOP2_NB];
    __shared__ int s_i0[CLAUDE_TOP2_NB];
    __shared__ float s_v1[CLAUDE_TOP2_NB];
    __shared__ int s_i1[CLAUDE_TOP2_NB];
    __shared__ float s_pm[CLAUDE_TOP2_NB];
    __shared__ float s_ps[CLAUDE_TOP2_NB];
    __shared__ float s_arg[CLAUDE_TOP2_NB];   // expf argument (m - pm or pm - m)
    __shared__ float s_e[CLAUDE_TOP2_NB];     // expf(s_arg)
    __shared__ int s_case[CLAUDE_TOP2_NB];    // 0 skip (-inf), 1 new max, 2 accumulate
    __shared__ ClaudeTop2 s_top;
    __shared__ float s_max;
    if (nb != CLAUDE_TOP2_NB) return;   // host launches only nb == 128
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    {
        const float* p = part + (long long)(b * nb + t) * 6;
        s_v0[t] = p[0];
        s_i0[t] = __float_as_int(p[1]);
        s_v1[t] = p[2];
        s_i1[t] = __float_as_int(p[3]);
        s_pm[t] = p[4];
        s_ps[t] = p[5];
    }
    __syncthreads();

    if (t == 0) {
        // The original running-max chain; the expf is deferred.
        float m = -INFINITY;
#pragma unroll 16
        for (int j = 0; j < CLAUDE_TOP2_NB; j++) {
            const float pm = s_pm[j];
            if (pm == -INFINITY) { s_case[j] = 0; continue; }
            if (pm > m) { s_case[j] = 1; s_arg[j] = m - pm; m = pm; }
            else { s_case[j] = 2; s_arg[j] = pm - m; }
        }
        s_max = m;
    } else if (t == 32) {
        // The original top-2 push sequence (independent of the logsumexp).
        ClaudeTop2 a = {-INFINITY, 0x7fffffff, -INFINITY, 0x7fffffff};
#pragma unroll 16
        for (int j = 0; j < CLAUDE_TOP2_NB; j++) {
            claude_top2_push(a, s_v0[j], s_i0[j]);
            claude_top2_push(a, s_v1[j], s_i1[j]);
        }
        s_top = a;
    }
    __syncthreads();
    if (s_case[t] != 0) s_e[t] = expf(s_arg[t]);
    __syncthreads();
    if (t != 0) return;

    // The original sum recurrence, same order, same expressions.
    float sum = 0.0f;
#pragma unroll 16
    for (int j = 0; j < CLAUDE_TOP2_NB; j++) {
        const int c = s_case[j];
        if (c == 0) continue;
        const float ps = s_ps[j];
        if (c == 1) sum = sum * s_e[j] + ps;
        else sum += ps * s_e[j];
    }
    const float m = s_max;
    const float lse = m + logf(sum);
    const ClaudeTop2 a = s_top;
    top_idx[b * k + 0] = a.i0; top_logp[b * k + 0] = a.v0 - lse;
    if (k > 1) { top_idx[b * k + 1] = a.i1; top_logp[b * k + 1] = a.v1 - lse; }
}
