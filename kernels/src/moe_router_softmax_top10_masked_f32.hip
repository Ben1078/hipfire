// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Original-id E512/top10 router. Kept logits alone participate in selection.
#include <hip/hip_runtime.h>
#include <math.h>

__device__ __forceinline__ float keep_router_bf16(float v) {
    if (!isfinite(v)) return v;
    unsigned u = __float_as_uint(v);
    return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}

extern "C" __global__ void moe_router_softmax_top10_masked_f32(
    const float* __restrict__ logits,
    int* __restrict__ indices,
    float* __restrict__ weights,
    const unsigned char* __restrict__ keep,
    int round_logits
) {
    __shared__ float values[512];
    __shared__ unsigned char eligible[512];
    __shared__ float maxima[256];
    __shared__ int ids[256];
    __shared__ float selected[10];
    const int t = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 512;
    float a = keep[t] ? logits[base + t] : -INFINITY;
    float b = keep[t + 256] ? logits[base + t + 256] : -INFINITY;
    if (round_logits) { a = keep_router_bf16(a); b = keep_router_bf16(b); }
    values[t] = a;
    values[t + 256] = b;
    eligible[t] = keep[t];
    eligible[t + 256] = keep[t + 256];
    __syncthreads();
    for (int rank = 0; rank < 10; ++rank) {
        a = values[t]; b = values[t + 256];
        maxima[t] = fmaxf(a, b);
        ids[t] = eligible[t] && (!eligible[t + 256] || a >= b) ? t
            : eligible[t + 256] ? t + 256 : 0x7fffffff;
        __syncthreads();
        for (int width = 128; width; width >>= 1) {
            if (t < width) {
                const float other = maxima[t + width];
                const int id = ids[t + width];
                if (other > maxima[t] || (other == maxima[t] && id < ids[t])) {
                    maxima[t] = other; ids[t] = id;
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            indices[(size_t)blockIdx.x * 10 + rank] = ids[0];
            selected[rank] = maxima[0];
            values[ids[0]] = -INFINITY;
            eligible[ids[0]] = 0;
        }
        __syncthreads();
    }
    if (t == 0) {
        // Compute solely from selected logits: the same float operations and
        // order as a sliced router, independent of discarded probabilities.
        float p[10];
        float sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            p[rank] = expf(selected[rank] - selected[0]);
            sum += p[rank];
        }
        const float inverse = 1.0f / sum;
        for (int rank = 0; rank < 10; ++rank) {
            float w = p[rank] * inverse;
            weights[(size_t)blockIdx.x * 10 + rank] = round_logits ? keep_router_bf16(w) : w;
        }
    }
}
