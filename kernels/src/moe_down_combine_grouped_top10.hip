// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed top-10 BF16 grouped-down combine.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// `inverse_perm[token * 10 + rank]` maps a route slot to the sorted grouped
// row.  The grouped path intentionally mirrors
// moe_down_combine_top10_batched: route ranks are selected by ascending expert
// ID, and residual, weight, output, product, and every accumulation are each
// rounded through the source BF16 contract before the next rank.
static __device__ __forceinline__ float moe_down_widen(float v) { return v; }
static __device__ __forceinline__ float moe_down_widen(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

template <typename T>
__device__ __forceinline__ void moe_down_combine_grouped(
    const T* __restrict__ grouped_down,     // [grouped_rows * H], unweighted
    const int* __restrict__ inverse_perm,    // [N * 10], flat -> grouped row
    const int* __restrict__ topk_indices,     // [N * 10], expert IDs
    const float* __restrict__ topk_weights,   // [N * 10]
    float* __restrict__ residual,             // [N * H], in-place +=
    int hidden,
    int grouped_rows,
    int tokens,
    int n_experts
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || column >= hidden) return;

    const size_t route_base = (size_t)token * 10;
    __hip_bfloat16 accumulated = moe_top10_bf16_round(
        residual[(size_t)token * hidden + column]);
    // Ranks whose id is outside [0, n_experts) (the router's -1 marker) are
    // unrouted: never selected, contributing neither weight nor a grouped row.
    int used_ranks = 0;
    for (int rank = 0; rank < 10; ++rank) {
        const int expert = topk_indices[route_base + rank];
        if (expert < 0 || expert >= n_experts) used_ranks |= 1 << rank;
    }
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1;
        int selected_expert = 0x7fffffff;
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) != 0) continue;
            const int expert = topk_indices[route_base + rank];
            if (expert < selected_expert) {
                selected_expert = expert;
                selected_rank = rank;
            }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;

        const __hip_bfloat16 weight =
            moe_top10_bf16_round(topk_weights[route_base + selected_rank]);
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float raw_output =
            (grouped_row >= 0 && grouped_row < grouped_rows)
                ? moe_down_widen(grouped_down[(size_t)grouped_row * hidden + column])
                : 0.0f;
        const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
        const __hip_bfloat16 weighted = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(output) * moe_top10_bf16_to_f32(weight));
        accumulated = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
    }
    residual[(size_t)token * hidden + column] =
        moe_top10_bf16_to_f32(accumulated);
}

__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_grouped_top10(
    const float* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, int hidden, int grouped_rows, int tokens, int n_experts) {
    moe_down_combine_grouped(grouped_down, inverse_perm, topk_indices, topk_weights,
                             residual, hidden, grouped_rows, tokens, n_experts);
}

// The BF16-input combine's rank order, once per token: the ten route ranks
// in ascending expert ID (moe_down_combine_grouped's selection) as (grouped
// row or -1, BF16-rounded weight bits).  Grid [ceil(tokens / 64)], block 64.
extern "C" __global__ __launch_bounds__(64) void moe_combine_order_top10(
    const int* __restrict__ inverse_perm, const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights, int2* __restrict__ order, int grouped_rows,
    int tokens, int n_experts) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    const size_t route_base = (size_t)token * 10;
    int expert[10];
    int used_ranks = 0;
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        expert[rank] = topk_indices[route_base + rank];
        // Id outside [0, n_experts): unrouted, never selected (order slot stays (-1, 0)).
        if (expert[rank] < 0 || expert[rank] >= n_experts) used_ranks |= 1 << rank;
    }
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        int selected_rank = -1, selected_expert = 0x7fffffff;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) {
            if (!(used_ranks & (1 << rank)) && expert[rank] < selected_expert) {
                selected_expert = expert[rank];
                selected_rank = rank;
            }
        }
        if (selected_rank < 0) {  // no rank left: adds a zero (the sum is unchanged)
            order[route_base + c] = make_int2(-1, 0);
            continue;
        }
        used_ranks |= 1 << selected_rank;
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float weight =
            moe_top10_bf16_to_f32(moe_top10_bf16_round(topk_weights[route_base + selected_rank]));
        order[route_base + c] = make_int2(
            (grouped_row >= 0 && grouped_row < grouped_rows) ? grouped_row : -1,
            __float_as_int(weight));
    }
}

// Grouped down stored as BF16 bits (the values the BF16 round above yields),
// eight columns per thread, in moe_combine_order_top10's order: all ten
// ranks' 16-byte slices are loaded before the in-order accumulation, and each
// column runs the exact moe_down_combine_grouped arithmetic.
// Grid [ceil(hidden / 512), tokens], block 64, hidden % 8 == 0.
//
// `initial_zero`: the destination was just cleared to +0.0 by a ClearOp and
// carries no residual, so the accumulator starts as a register +0.0 and the
// destination is write-only (no residual load; the clear is then redundant).
// +0.0 is exactly what the cleared buffer would have read back, and
// bf16_round(+0.0) = +0.0, so every round that follows sees identical bits
// and the stores are bytewise those of clear + the `false` instantiation.
// Only the `_zinit` symbol instantiates it; the original symbol is `false`.
template <bool initial_zero>
static __device__ __forceinline__ void moe_down_combine_grouped_bf16in(
    const unsigned short* __restrict__ grouped_down, const int2* __restrict__ order,
    float* __restrict__ residual, int hidden, int tokens) {
    const int token = blockIdx.y;
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (token >= tokens || col0 >= hidden) return;
    const size_t route_base = (size_t)token * 10;
    uint4 rows[10];
    float wts[10];
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const int2 o = order[route_base + c];
        wts[c] = __int_as_float(o.y);
        rows[c] = o.x >= 0 ? *(const uint4*)(grouped_down + (size_t)o.x * hidden + col0)
                           : make_uint4(0, 0, 0, 0);
    }
    float* res = residual + (size_t)token * hidden + col0;
    float acc[8];
    if (initial_zero) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
    } else {
        const float4 r0 = *(const float4*)res, r1 = *(const float4*)(res + 4);
        acc[0] = r0.x; acc[1] = r0.y; acc[2] = r0.z; acc[3] = r0.w;
        acc[4] = r1.x; acc[5] = r1.y; acc[6] = r1.z; acc[7] = r1.w;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        __hip_bfloat16 accumulated = moe_top10_bf16_round(acc[e]);
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            const unsigned int w = (&rows[c].x)[e >> 1];
            const float raw_output = __uint_as_float((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
            const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
            const __hip_bfloat16 weighted = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(output) * wts[c]);
            accumulated = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
        }
        acc[e] = moe_top10_bf16_to_f32(accumulated);
    }
    *(float4*)res = make_float4(acc[0], acc[1], acc[2], acc[3]);
    *(float4*)(res + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

extern "C" __global__ __launch_bounds__(64) void moe_down_combine_grouped_top10_bf16in(
    const unsigned short* __restrict__ grouped_down, const int2* __restrict__ order,
    float* __restrict__ residual, int hidden, int tokens) {
    moe_down_combine_grouped_bf16in<false>(grouped_down, order, residual, hidden, tokens);
}

// Same arguments and launch as the symbol above; `residual` is write-only.
extern "C" __global__ __launch_bounds__(64) void moe_down_combine_grouped_top10_bf16in_zinit(
    const unsigned short* __restrict__ grouped_down, const int2* __restrict__ order,
    float* __restrict__ residual, int hidden, int tokens) {
    moe_down_combine_grouped_bf16in<true>(grouped_down, order, residual, hidden, tokens);
}

// ---------------------------------------------------------------------------
// S1 slow-link layer-major prefill: the ordered combine of bounded expert
// panels.  The accumulator `acc` is the F32 widening of the running BF16 sum
// per token/column (+0.0 at the start of the layer, no residual).  Every valid
// route slot of a token is applied exactly once, in exactly the canonical
// order of moe_combine_order_top10 (ascending expert id, ties by rank), with
// exactly the arithmetic of moe_down_combine_grouped_top10_bf16in (product and
// each sum rounded to BF16), but across several launches: one launch per
// panel of grouped rows, panels visited in ascending grouped-row order.
//
// A slot belongs to a panel iff its id is valid and
//   * its grouped row is valid (0 <= row < grouped_rows) and lies inside
//     [row_base, row_base + panel_rows)  -- stable grouping puts a slot's row
//     inside its own expert's rows, so ascending panels preserve the canonical
//     order even when one expert's rows are split across panels; or
//   * its grouped row is invalid (the canonical kernel then multiplies +0 by
//     the weight) and its id lies in the panel's ownership range
//     [own_lo, own_hi); the ownership ranges of the panels of a layer
//     partition [0, n_experts) in panel order.
// Slots of other panels are SKIPPED, not added as zeros: a skipped term leaves
// the accumulator untouched, while an extra +0 term would turn a -0 into +0.
// Unrouted ranks (ids outside [0, n_experts)) are never applied here; the
// canonical kernel's trailing (10 - valid) +0 products run once per layer in
// slowlink_combine_pad.  No atomics, no completion-order sums.
// Grid [ceil(hidden / 512), tokens], block 64, hidden % 8 == 0.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(64) void slowlink_combine_panel(
    const unsigned short* __restrict__ y,         // [panel_rows, hidden] BF16 bits, panel-local rows
    const int* __restrict__ inverse_perm,         // [tokens * 10] flat slot -> window grouped row
    const int* __restrict__ topk,                 // [tokens * 10] expert ids
    const float* __restrict__ weights,            // [tokens * 10]
    float* __restrict__ acc,                      // [tokens, hidden] widened BF16 accumulator
    int hidden, int tokens, int n_experts, int grouped_rows,
    int row_base, int panel_rows, int own_lo, int own_hi) {
    const int token = blockIdx.y;
    if (token >= tokens) return;
    const size_t rb = (size_t)token * 10;
    int ids[10];
    int rows[10];
    int used = 0, mine = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const int id = topk[rb + r];
        const int row = inverse_perm[rb + r];
        ids[r] = id;
        rows[r] = row;
        if (id < 0 || id >= n_experts) {
            used |= 1 << r;
            continue;
        }
        const bool row_ok = row >= 0 && row < grouped_rows;
        const bool owned = row_ok ? (row >= row_base && row - row_base < panel_rows)
                                  : (id >= own_lo && id < own_hi);
        if (owned) mine |= 1 << r;
    }
    if (mine == 0) return;  // uniform per block
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (col0 >= hidden) return;
    used |= ~mine;
    float* a = acc + (size_t)token * hidden + col0;
    float v[8];
    {
        const float4 a0 = *(const float4*)a, a1 = *(const float4*)(a + 4);
        v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w;
        v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    }
    for (int count = 0; count < 10; ++count) {
        int sel = -1, sel_id = 0x7fffffff;
#pragma unroll
        for (int r = 0; r < 10; ++r) {
            if (!(used & (1 << r)) && ids[r] < sel_id) {
                sel_id = ids[r];
                sel = r;
            }
        }
        if (sel < 0) break;
        used |= 1 << sel;
        const float wt = moe_top10_bf16_to_f32(moe_top10_bf16_round(weights[rb + sel]));
        const int row = rows[sel];
        uint4 packed = make_uint4(0, 0, 0, 0);
        if (row >= 0 && row < grouped_rows)
            packed = *(const uint4*)(y + (size_t)(row - row_base) * hidden + col0);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const unsigned int w = (&packed.x)[e >> 1];
            const float raw_output = __uint_as_float((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
            const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
            const __hip_bfloat16 weighted = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(output) * wt);
            v[e] = moe_top10_bf16_to_f32(moe_top10_bf16_round(
                v[e] + moe_top10_bf16_to_f32(weighted)));
        }
    }
    *(float4*)a = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)(a + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// The canonical kernel's terminal products, once per layer: every token adds
// (10 - valid) +0 products (bf16(bf16(+0) * bf16(+0)) = +0) after its last
// valid rank.  Only a -0 accumulator changes.  Same grid as the panel kernel.
extern "C" __global__ __launch_bounds__(64) void slowlink_combine_pad(
    const int* __restrict__ topk, float* __restrict__ acc, int hidden, int tokens,
    int n_experts) {
    const int token = blockIdx.y;
    if (token >= tokens) return;
    const size_t rb = (size_t)token * 10;
    int valid = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const int id = topk[rb + r];
        valid += (id >= 0 && id < n_experts) ? 1 : 0;
    }
    if (valid >= 10) return;  // uniform per block
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (col0 >= hidden) return;
    float* a = acc + (size_t)token * hidden + col0;
    float v[8];
    {
        const float4 a0 = *(const float4*)a, a1 = *(const float4*)(a + 4);
        v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w;
        v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    }
    const float zero = moe_top10_bf16_to_f32(moe_top10_bf16_round(0.0f * 0.0f));
    for (int k = valid; k < 10; ++k) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            v[e] = moe_top10_bf16_to_f32(moe_top10_bf16_round(v[e] + zero));
    }
    *(float4*)a = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)(a + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// S1 K1-cohort support: the ids of a short-tail tile restricted to one expert binding
// [lo, hi); every other id becomes -1 (the K1 kernels' no-load exit), so a K1 launch against
// one staged binding computes exactly the slots whose expert that binding holds.
extern "C" __global__ __launch_bounds__(256) void slowlink_mask_ids(
    const int* __restrict__ ids, int* __restrict__ out, int n, int lo, int hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int v = ids[i];
    out[i] = (v >= lo && v < hi) ? v : -1;
}
