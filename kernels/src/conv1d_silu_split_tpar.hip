// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Token-parallel prefill sibling of `conv1d_silu_split_f32`
// (conv1d_silu_split.hip): 4-tap causal depthwise conv + SiLU + Q/K/V split
// over a whole prefill chunk.
//
// The incumbent walks all n_tokens of one channel in one thread, and each
// iteration waits on its own global load (a full memory round trip per
// token), so a 1024-token chunk costs ~1024 load latencies regardless of
// bandwidth. Every output only needs x[t-3..t] of its own channel, so here a
// thread owns one channel for a TT-token slice (grid.y = token slices,
// TT = 16: 75 us vs 279 us for the incumbent at n = 1024 on gfx1100) and
// issues all TT+3 loads before computing; the halo comes from the input, or
// from the conv state for the first slice.
//
// State: only slice 0 reads and writes a channel's ring buffer (it reads the
// old state as its halo first, then writes the new state from the input's
// last three rows), so no other thread races with the update. The new state is
// the last three values of [s2, s1, s0, x_0 .. x_{n-1}], matching the
// incumbent's per-token shift for any n >= 1.
//
// Numerics are bit-identical to the incumbent's compiled DAG (gfx1100 ISA):
//   y = fma(s2, w0, fma(s1, w1, fma(x, w3, s0 * w2)))
//   result = y / (1.0f + expf(-y))      (same expression, same flags)
// with s0/s1/s2 = x[t-1]/x[t-2]/x[t-3].
//
// Layout: input [n_tokens][2*k_dim + v_dim]; weight [channel][4];
// state [channel][3] (lane 0 only: this sibling serves grid.y-sliced prefill,
// not the batched-decode lane form); q/k out [n][k_dim], v out [n][v_dim].

#include <hip/hip_runtime.h>

#ifndef CONV_TPAR_TT
#define CONV_TPAR_TT 16
#endif

extern "C" __global__ __launch_bounds__(256) void conv1d_silu_split_f32_tpar(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int k_dim,
    int v_dim,
    int n_tokens
) {
    constexpr int TT = CONV_TPAR_TT;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_channels = 2 * k_dim + v_dim;
    if (c >= n_channels) return;
    const int t0 = blockIdx.y * TT;
    if (t0 >= n_tokens) return;
    const int tn = min(TT, n_tokens - t0);

    const float w0 = weight[c * 4 + 0];
    const float w1 = weight[c * 4 + 1];
    const float w2 = weight[c * 4 + 2];
    const float w3 = weight[c * 4 + 3];

    // Window: h[0..2] = x[t0-3], x[t0-2], x[t0-1]; h[3 + i] = x[t0 + i].
    float h[TT + 3];
    if (t0 == 0) {
        h[2] = state[c * 3 + 0];
        h[1] = state[c * 3 + 1];
        h[0] = state[c * 3 + 2];
    } else {
        // t0 >= TT >= 3, so the halo lies inside the input.
        h[0] = input[(size_t)(t0 - 3) * n_channels + c];
        h[1] = input[(size_t)(t0 - 2) * n_channels + c];
        h[2] = input[(size_t)(t0 - 1) * n_channels + c];
    }
    if (tn == TT) {
#pragma unroll
        for (int i = 0; i < TT; ++i) h[3 + i] = input[(size_t)(t0 + i) * n_channels + c];
    } else {
#pragma unroll
        for (int i = 0; i < TT; ++i)
            h[3 + i] = (i < tn) ? input[(size_t)(t0 + i) * n_channels + c] : 0.0f;
    }

    float* dst;
    int dst_stride, dc;
    if (c < k_dim) {
        dst = q_out; dst_stride = k_dim; dc = c;
    } else if (c < 2 * k_dim) {
        dst = k_out; dst_stride = k_dim; dc = c - k_dim;
    } else {
        dst = v_out; dst_stride = v_dim; dc = c - 2 * k_dim;
    }

#pragma unroll
    for (int i = 0; i < TT; ++i) {
        if (i < tn) {
            const float x = h[3 + i];
            const float s0 = h[2 + i];
            const float s1 = h[1 + i];
            const float s2 = h[i];
            const float y = __builtin_fmaf(s2, w0, __builtin_fmaf(s1, w1, __builtin_fmaf(x, w3, __fmul_rn(s0, w2))));
            const float result = y / (1.0f + expf(-y));
            dst[(size_t)(t0 + i) * dst_stride + dc] = result;
        }
    }

    if (t0 == 0) {
        // New ring = last three of [s2, s1, s0, x_0 .. x_{n-1}].
        float r0, r1, r2;  // x[n-1], x[n-2], x[n-3] in sequence terms
        if (n_tokens >= 3) {
            if (n_tokens <= TT) {
                r0 = h[3 + n_tokens - 1];
                r1 = h[3 + n_tokens - 2];
                r2 = h[3 + n_tokens - 3];
            } else {
                r0 = input[(size_t)(n_tokens - 1) * n_channels + c];
                r1 = input[(size_t)(n_tokens - 2) * n_channels + c];
                r2 = input[(size_t)(n_tokens - 3) * n_channels + c];
            }
        } else {
            // n_tokens in {1, 2}: h[3 + n - 1 - j] for j = 0..2 still indexes
            // the window (h[0..2] hold the old state).
            r0 = h[3 + n_tokens - 1];
            r1 = h[3 + n_tokens - 2];
            r2 = h[3 + n_tokens - 3];
        }
        state[c * 3 + 0] = r0;
        state[c * 3 + 1] = r1;
        state[c * 3 + 2] = r2;
    }
}
