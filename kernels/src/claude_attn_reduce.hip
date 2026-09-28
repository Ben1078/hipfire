// claude: split-K attention reduce with 8 warps per (head, row) instead of 1,
// single pass with online max. Same math as attention_flash_asym_reduce_batched
// (partials hold ln-domain tile max, tile sum, unnormalized acc). head_dim 256.
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(256) __global__ void claude_attn_reduce(
    const float* __restrict__ partials,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads, int head_dim, int tile_size, int max_tiles, int batch_offset)
{
    __shared__ float sm[8];
    __shared__ float sl[8];
    __shared__ float sg[8][256];
    const int h = blockIdx.x, local_bid = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int stride = 2 + head_dim;
    const int seq_len = positions[batch_offset + local_bid] + 1;
    const int n_tiles = (seq_len + tile_size - 1) / tile_size;
    const float* base = partials + ((long long)local_bid * n_heads + h) * max_tiles * stride;
    const int d0 = lane * 8;
    float m = -1e30f, l = 0.0f, g[8];
#pragma unroll
    for (int i = 0; i < 8; i++) g[i] = 0.0f;
    for (int tile = warp; tile < n_tiles; tile += 8) {
        const float* p = base + (long long)tile * stride;
        const float ts = p[1];
        if (ts <= 0.0f) continue;
        const float tm = p[0];
        if (tm > m) {
            const float c = expf(m - tm);
            l *= c;
#pragma unroll
            for (int i = 0; i < 8; i++) g[i] *= c;
            m = tm;
        }
        const float c = expf(tm - m);
        l += ts * c;
        const float4 a = *(const float4*)(p + 2 + d0);
        const float4 b = *(const float4*)(p + 2 + d0 + 4);
        g[0] += a.x * c; g[1] += a.y * c; g[2] += a.z * c; g[3] += a.w * c;
        g[4] += b.x * c; g[5] += b.y * c; g[6] += b.z * c; g[7] += b.w * c;
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int i = 0; i < 8; i++) sg[warp][d0 + i] = g[i];
    __syncthreads();
    const int d = threadIdx.x;
    float M = -1e30f;
#pragma unroll
    for (int w = 0; w < 8; w++) if (sl[w] > 0.0f) M = fmaxf(M, sm[w]);
    float L = 0.0f, G = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; w++) {
        if (sl[w] <= 0.0f) continue;
        const float c = expf(sm[w] - M);
        L += sl[w] * c;
        G += sg[w][d] * c;
    }
    out[(long long)local_bid * n_heads * head_dim + h * head_dim + d] = L > 0.0f ? G / L : 0.0f;
}
