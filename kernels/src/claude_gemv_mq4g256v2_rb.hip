// claude: MQ4G256V2 GEMV for small batches (speculative verify).
// One warp computes R weight rows for B input rows. Each lane loads its
// 8 x values per group once per b and reuses them across the R rows, so
// x cache traffic matches the single-row GEMV while weights are read once.
// y[b*M + row] (+)= sum_k W[row,k] * x[b*K + k]. x is FWHT-rotated f32.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef CLAUDE_RB_R
#define CLAUDE_RB_R 4
#endif
#ifndef CLAUDE_RB_BMAX
#define CLAUDE_RB_BMAX 4
#endif
#ifndef CLAUDE_RB_KERNEL
#define CLAUDE_RB_KERNEL claude_gemv_mq4g256v2_rb
#endif

__launch_bounds__(32, 8)
extern "C" __global__ void CLAUDE_RB_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int B)
{
    const int row0 = blockIdx.x * CLAUDE_RB_R;
    const int tid = threadIdx.x;
    const int gpr = K >> 8;
    const long long row_bytes = (long long)gpr * 136;

    float acc[CLAUDE_RB_R][CLAUDE_RB_BMAX];
#pragma unroll
    for (int r = 0; r < CLAUDE_RB_R; r++)
#pragma unroll
        for (int b = 0; b < CLAUDE_RB_BMAX; b++) acc[r][b] = 0.0f;

    int nrows = M - row0;
    if (nrows > CLAUDE_RB_R) nrows = CLAUDE_RB_R;
    const char* base = A + (long long)row0 * row_bytes;

    for (int g = 0; g < gpr; g++) {
        // Weights first (all rows), so loads overlap before the FMA burst.
        unsigned int pk[CLAUDE_RB_R];
        float sc[CLAUDE_RB_R], zp[CLAUDE_RB_R];
#pragma unroll
        for (int r = 0; r < CLAUDE_RB_R; r++) {
            if (r < nrows) {
                const char* gp = base + r * row_bytes + (long long)g * 136;
                const unsigned int hs = *(const unsigned int*)(gp + ((tid < 16) ? 0 : 4));
                sc[r] = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
                zp[r] = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
                pk[r] = *(const unsigned int*)(gp + 8 + tid * 4);
            } else {
                sc[r] = 0.0f; zp[r] = 0.0f; pk[r] = 0u;
            }
        }
        // Dequantize once per group, reuse across all B inputs.
        float wv[CLAUDE_RB_R][8];
#pragma unroll
        for (int r = 0; r < CLAUDE_RB_R; r++)
#pragma unroll
            for (int j = 0; j < 8; j++)
                wv[r][j] = sc[r] * (float)((pk[r] >> (4 * j)) & 0xFu) + zp[r];
        const int xo = g * 256 + tid * 8;
#pragma unroll
        for (int b = 0; b < CLAUDE_RB_BMAX; b++) {
            if (b < B) {
                const float4 xa = *(const float4*)(x + (long long)b * K + xo);
                const float4 xb = *(const float4*)(x + (long long)b * K + xo + 4);
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int r = 0; r < CLAUDE_RB_R; r++) {
                    float s = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; j++) s += wv[r][j] * xv[j];
                    acc[r][b] += s;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < CLAUDE_RB_R; r++) {
#pragma unroll
        for (int b = 0; b < CLAUDE_RB_BMAX; b++) {
            float v = acc[r][b];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
            if (tid == 0 && r < nrows && b < B) {
#ifdef CLAUDE_RB_RESIDUAL
                y[(long long)b * M + row0 + r] += v;
#else
                y[(long long)b * M + row0 + r] = v;
#endif
            }
        }
    }
}
