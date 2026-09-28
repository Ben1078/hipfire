// claude: small-batch MQ4G256V2 GEMV on int8 activations (speculative verify).
//
// claude_q8x_quant: x f32 [B x K] -> per-lane packed int8 (even/odd element
//   order to match the nibble layout) + one f32 scale per 32 elements.
// claude_q8x_gemv: one warp per weight row, four groups loaded ahead, and
//   v_dot4_i32_iu8 (unsigned 4-bit weights x signed int8 activations).
//   sum_k W x ~= d * (sc * sum(q * xq) + zp * sum(xq)) per 8-element lane chunk.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef CLAUDE_Q8X_BMAX
#define CLAUDE_Q8X_BMAX 4
#endif
#ifndef CLAUDE_Q8X_KERNEL
#define CLAUDE_Q8X_KERNEL claude_q8x_gemv
#endif
#ifndef CLAUDE_Q8X_MULTI_KERNEL
#define CLAUDE_Q8X_MULTI_KERNEL claude_q8x_gemv_multi
#endif

#ifdef CLAUDE_Q8X_QUANT
// grid (K/256, B), block 32. Lane t owns elements g*256 + t*8 .. +7.
extern "C" __global__ void claude_q8x_quant(
    const float* __restrict__ x,
    unsigned int* __restrict__ xq,   // [B][K/256][32] uint2
    float* __restrict__ dx,          // [B][K/32]
    int K)
{
    const int g = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const float* xp = x + (long long)b * K + g * 256 + t * 8;
    const float4 va = *(const float4*)xp;
    const float4 vb = *(const float4*)(xp + 4);
    const float v[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
    // 32-element block = 4 lanes (4k..4k+3).
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    const float d = amax * (1.0f / 127.0f);
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q[8];
#pragma unroll
    for (int j = 0; j < 8; j++) q[j] = (int)rintf(v[j] * id);
    const unsigned int e = (unsigned)(q[0] & 0xFF) | ((unsigned)(q[2] & 0xFF) << 8)
                         | ((unsigned)(q[4] & 0xFF) << 16) | ((unsigned)(q[6] & 0xFF) << 24);
    const unsigned int o = (unsigned)(q[1] & 0xFF) | ((unsigned)(q[3] & 0xFF) << 8)
                         | ((unsigned)(q[5] & 0xFF) << 16) | ((unsigned)(q[7] & 0xFF) << 24);
    uint2* xq2 = (uint2*)xq;
    xq2[((long long)b * (K >> 8) + g) * 32 + t] = make_uint2(e, o);
    if ((t & 3) == 0) dx[(long long)b * (K >> 5) + g * 8 + (t >> 2)] = d;
}
#endif

#ifdef CLAUDE_Q8X_GEMV
#ifndef CLAUDE_Q8X_U
#define CLAUDE_Q8X_U 4
#endif
#ifndef CLAUDE_Q8X_OCC
#define CLAUDE_Q8X_OCC 8
#endif
__device__ __forceinline__ float claude_q8x_lane(
    unsigned int pk, float sc, float zp, uint2 xv, float d)
{
    const int ev = (int)(pk & 0x0F0F0F0Fu);
    const int od = (int)((pk >> 4) & 0x0F0F0F0Fu);
    int s = __builtin_amdgcn_sudot4(false, ev, true, (int)xv.x, 0, false);
    s = __builtin_amdgcn_sudot4(false, od, true, (int)xv.y, s, false);
    int sx = __builtin_amdgcn_sudot4(true, (int)xv.x, false, 0x01010101, 0, false);
    sx = __builtin_amdgcn_sudot4(true, (int)xv.y, false, 0x01010101, sx, false);
    return d * (sc * (float)s + zp * (float)sx);
}

__device__ __forceinline__ void claude_q8x_row(
    const char* __restrict__ A,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    float* __restrict__ y,
    int M, int K, int B, int row)
{
    const int t = threadIdx.x;
    const int gpr = K >> 8;
    const int dpr = K >> 5;
    const char* rp = A + (long long)row * gpr * 136;
    const uint2* xq2 = (const uint2*)xq;
    const int hoff = (t < 16) ? 0 : 4;

    float acc[CLAUDE_Q8X_BMAX];
#pragma unroll
    for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) acc[b] = 0.0f;

    int g = 0;
    for (; g + CLAUDE_Q8X_U <= gpr; g += CLAUDE_Q8X_U) {
        unsigned int hs[CLAUDE_Q8X_U], pk[CLAUDE_Q8X_U];
#pragma unroll
        for (int u = 0; u < CLAUDE_Q8X_U; u++) {
            const char* gp = rp + (g + u) * 136;
            hs[u] = *(const unsigned int*)(gp + hoff);
            pk[u] = *(const unsigned int*)(gp + 8 + t * 4);
        }
        float sc[CLAUDE_Q8X_U], zp[CLAUDE_Q8X_U];
#pragma unroll
        for (int u = 0; u < CLAUDE_Q8X_U; u++) {
            sc[u] = __half2float(__ushort_as_half((unsigned short)(hs[u] & 0xFFFFu)));
            zp[u] = __half2float(__ushort_as_half((unsigned short)(hs[u] >> 16)));
        }
#pragma unroll
        for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) {
            if (b < B) {
                const uint2* xb = xq2 + ((long long)b * gpr + g) * 32 + t;
                const float* db = dx + (long long)b * dpr + g * 8 + (t >> 2);
                float s = 0.0f;
#pragma unroll
                for (int u = 0; u < CLAUDE_Q8X_U; u++)
                    s += claude_q8x_lane(pk[u], sc[u], zp[u], xb[u * 32], db[u * 8]);
                acc[b] += s;
            }
        }
    }
    for (; g < gpr; g++) {
        const char* gp = rp + g * 136;
        const unsigned int hs = *(const unsigned int*)(gp + hoff);
        const unsigned int pk = *(const unsigned int*)(gp + 8 + t * 4);
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
#pragma unroll
        for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) {
            if (b < B)
                acc[b] += claude_q8x_lane(pk, sc, zp,
                    xq2[((long long)b * gpr + g) * 32 + t],
                    dx[(long long)b * dpr + g * 8 + (t >> 2)]);
        }
    }

#pragma unroll
    for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) {
        float v = acc[b];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (t == 0 && b < B) {
#ifdef CLAUDE_Q8X_RESIDUAL
            y[(long long)b * M + row] += v;
#else
            y[(long long)b * M + row] = v;
#endif
        }
    }
}

__launch_bounds__(32, CLAUDE_Q8X_OCC)
extern "C" __global__ void CLAUDE_Q8X_KERNEL(
    const char* __restrict__ A,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    float* __restrict__ y,
    int M, int K, int B)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    claude_q8x_row(A, xq, dx, y, M, K, B, row);
}

// Up to four weights sharing one activation in a single launch; grid = sum(M).
__launch_bounds__(32, CLAUDE_Q8X_OCC)
extern "C" __global__ void CLAUDE_Q8X_MULTI_KERNEL(
    const char* __restrict__ A0, float* __restrict__ y0, int M0,
    const char* __restrict__ A1, float* __restrict__ y1, int M1,
    const char* __restrict__ A2, float* __restrict__ y2, int M2,
    const char* __restrict__ A3, float* __restrict__ y3, int M3,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    int K, int B)
{
    int row = blockIdx.x;
    if (row < M0) { claude_q8x_row(A0, xq, dx, y0, M0, K, B, row); return; }
    row -= M0;
    if (row < M1) { claude_q8x_row(A1, xq, dx, y1, M1, K, B, row); return; }
    row -= M1;
    if (row < M2) { claude_q8x_row(A2, xq, dx, y2, M2, K, B, row); return; }
    row -= M2;
    if (row < M3) { claude_q8x_row(A3, xq, dx, y3, M3, K, B, row); }
}
#endif

#ifdef CLAUDE_Q8X_GEMV_LDS
// Same math as CLAUDE_Q8X_GEMV, but a block of W warps (one weight row each)
// stages int8 activations in LDS per chunk of KC groups, cutting activation
// cache traffic by W.
#ifndef CLAUDE_Q8X_W
#define CLAUDE_Q8X_W 8
#endif
#ifndef CLAUDE_Q8X_KC
#define CLAUDE_Q8X_KC 16
#endif
__device__ __forceinline__ float claude_q8x_lane2(
    unsigned int pk, float sc, float zp, uint2 xv, float d)
{
    const int ev = (int)(pk & 0x0F0F0F0Fu);
    const int od = (int)((pk >> 4) & 0x0F0F0F0Fu);
    int s = __builtin_amdgcn_sudot4(false, ev, true, (int)xv.x, 0, false);
    s = __builtin_amdgcn_sudot4(false, od, true, (int)xv.y, s, false);
    int sx = __builtin_amdgcn_sudot4(true, (int)xv.x, false, 0x01010101, 0, false);
    sx = __builtin_amdgcn_sudot4(true, (int)xv.y, false, 0x01010101, sx, false);
    return d * (sc * (float)s + zp * (float)sx);
}

__launch_bounds__(32 * CLAUDE_Q8X_W)
extern "C" __global__ void CLAUDE_Q8X_KERNEL(
    const char* __restrict__ A,
    const unsigned int* __restrict__ xq,
    const float* __restrict__ dx,
    float* __restrict__ y,
    int M, int K, int B)
{
    __shared__ uint2 sx[CLAUDE_Q8X_BMAX * CLAUDE_Q8X_KC * 32];
    __shared__ float sd[CLAUDE_Q8X_BMAX * CLAUDE_Q8X_KC * 8];
    const int warp = threadIdx.x >> 5;
    const int t = threadIdx.x & 31;
    const int row = blockIdx.x * CLAUDE_Q8X_W + warp;
    const bool active = row < M;
    const int gpr = K >> 8;
    const int dpr = K >> 5;
    const char* rp = A + (long long)(active ? row : 0) * gpr * 136;
    const uint2* xq2 = (const uint2*)xq;
    const int hoff = (t < 16) ? 0 : 4;

    float acc[CLAUDE_Q8X_BMAX];
#pragma unroll
    for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) acc[b] = 0.0f;

    for (int c0 = 0; c0 < gpr; c0 += CLAUDE_Q8X_KC) {
        const int kc = min(CLAUDE_Q8X_KC, gpr - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < B * kc * 32; i += 32 * CLAUDE_Q8X_W) {
            const int b = i / (kc * 32), r = i - b * (kc * 32);
            sx[b * (CLAUDE_Q8X_KC * 32) + r] = xq2[((long long)b * gpr + c0) * 32 + r];
        }
        for (int i = threadIdx.x; i < B * kc * 8; i += 32 * CLAUDE_Q8X_W) {
            const int b = i / (kc * 8), r = i - b * (kc * 8);
            sd[b * (CLAUDE_Q8X_KC * 8) + r] = dx[(long long)b * dpr + c0 * 8 + r];
        }
        __syncthreads();
        if (active) {
            int gl = 0;
            for (; gl + 4 <= kc; gl += 4) {
                unsigned int hs[4], pk[4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const char* gp = rp + (c0 + gl + u) * 136;
                    hs[u] = *(const unsigned int*)(gp + hoff);
                    pk[u] = *(const unsigned int*)(gp + 8 + t * 4);
                }
                float sc[4], zp[4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    sc[u] = __half2float(__ushort_as_half((unsigned short)(hs[u] & 0xFFFFu)));
                    zp[u] = __half2float(__ushort_as_half((unsigned short)(hs[u] >> 16)));
                }
#pragma unroll
                for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) {
                    if (b < B) {
                        float s = 0.0f;
#pragma unroll
                        for (int u = 0; u < 4; u++)
                            s += claude_q8x_lane2(pk[u], sc[u], zp[u],
                                sx[b * (CLAUDE_Q8X_KC * 32) + (gl + u) * 32 + t],
                                sd[b * (CLAUDE_Q8X_KC * 8) + (gl + u) * 8 + (t >> 2)]);
                        acc[b] += s;
                    }
                }
            }
            for (; gl < kc; gl++) {
                const char* gp = rp + (c0 + gl) * 136;
                const unsigned int hs = *(const unsigned int*)(gp + hoff);
                const unsigned int pk = *(const unsigned int*)(gp + 8 + t * 4);
                const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
                const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
#pragma unroll
                for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) {
                    if (b < B)
                        acc[b] += claude_q8x_lane2(pk, sc, zp,
                            sx[b * (CLAUDE_Q8X_KC * 32) + gl * 32 + t],
                            sd[b * (CLAUDE_Q8X_KC * 8) + gl * 8 + (t >> 2)]);
                }
            }
        }
    }

#pragma unroll
    for (int b = 0; b < CLAUDE_Q8X_BMAX; b++) {
        float v = acc[b];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down(v, off);
        if (t == 0 && active && b < B) {
#ifdef CLAUDE_Q8X_RESIDUAL
            y[(long long)b * M + row] += v;
#else
            y[(long long)b * M + row] = v;
#endif
        }
    }
}
#endif
