// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Quad-per-row prefill sibling of `gated_delta_net_q8_fast` (the DPP-reduce +
// one-token-prefetch build used by batched prefill), EF-residual form only.
//
// The incumbent gives every state row the whole wave: lane l holds columns l,
// l+32, l+64, l+96 of four rows, so each of a row's two dot products per token
// ends in a five-step cross-lane tree (permlanex16, row_shl 8/4/2/1) plus a
// readfirstlane. Per token a wave spends ~150 VALU slots (and ~50 dependency
// stalls) on four rows.
//
// Here four lanes share a row (lane 4g + j of a wave: row g of the wave's 8,
// column class j). Lane j owns the incumbent's per-lane partials p[l] for
// l = j + 4i (i = 0..7), i.e. columns l + 32m, so the incumbent's tree levels
// that pair l with l+16, l+8 and l+4 are in-lane adds and only the last two
// (l with l+2, then l+1) are DPP row_shl adds inside the quad; the result is
// broadcast with quad_perm. The arithmetic is the incumbent's compiled DAG
// (gfx1100 ISA), bit for bit:
//   partial p[l] = fma(k3, s3, fma(k2, s2, fma(k0, s0, k1 * s1)))
//                  with s_m = S[row][l + 32m], k_m = k[l + 32m]
//   tree         a[i] = p[i] + p[i+16]; b[i] = a[i] + a[i+8]; c[i] = b[i] + b[i+4];
//                d[i] = c[i] + c[i+2]; kv = d[0] + d[1]
//   delta        = fma(-alpha, kv, v_row) * beta,   alpha = expf(gate)
//   update       s = fma(alpha, s, k * delta)
//   output       the same partials/tree with q and the updated state.
// K/Q of the head are staged per token in LDS as float4 {x[l], x[l+32],
// x[l+64], x[l+96]}; the next token's K/Q/V/gate/beta are prefetched one token
// ahead into registers.
//
// Commit seams: the incumbent is launched once per `seg`-row segment
// (prefill commit_stride 512): it dequantizes the Q8 state at segment start and
// requantizes it at segment end (deterministic RN, f16 error-feedback
// residual). This kernel runs every segment in one launch and does the same
// requant -> dequant round trip in registers at each interior seam: the next
// launch's dequant scale * (float)(int8)qf is exact, and the EF residual is
// carried as the same f16 value. Only the final state is written (Q8, scale,
// EF), identical to the incumbent's last launch. seg <= 0: one segment.
//
// Grid: (n_heads, 128 / 8), block 32 (one wave, 8 rows).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <type_traits>
#include <stdint.h>

namespace gdn_qr {

constexpr int HD = 128;

// DPP helpers (gfx10+ encodings): row_shl:k = 0x100 + k, quad_perm = 0x00..0xff.
__device__ __forceinline__ float dpp_row_shl1(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x101, 0xf, 0xf, true));
}
__device__ __forceinline__ float dpp_row_shl2(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x102, 0xf, 0xf, true));
}
// quad_perm [0,0,0,0]: broadcast quad lane 0.
__device__ __forceinline__ float dpp_quad_bcast0(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x00, 0xf, 0xf, true));
}
// quad_perm [1,0,3,2] and [2,3,0,1]: xor-1 / xor-2 within the quad.
__device__ __forceinline__ float dpp_quad_xor1(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, true));
}
__device__ __forceinline__ float dpp_quad_xor2(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, true));
}

// Incumbent partial for one l: s_m = S[row][l + 32m], x_m = x[l + 32m].
__device__ __forceinline__ float partial(const float4 x, const float s0, const float s1, const float s2,
                                         const float s3) {
    return __builtin_fmaf(x.w, s3, __builtin_fmaf(x.z, s2, __builtin_fmaf(x.x, s0, __fmul_rn(x.y, s1))));
}

// Levels l+16, l+8, l+4 in-lane; l+2, l+1 across the quad. Valid in quad lane 0.
__device__ __forceinline__ float tree(const float (&p)[8]) {
    const float a0 = __fadd_rn(p[0], p[4]);
    const float a1 = __fadd_rn(p[1], p[5]);
    const float a2 = __fadd_rn(p[2], p[6]);
    const float a3 = __fadd_rn(p[3], p[7]);
    const float b0 = __fadd_rn(a0, a2);
    const float b1 = __fadd_rn(a1, a3);
    const float c = __fadd_rn(b0, b1);
    const float d = __fadd_rn(c, dpp_row_shl2(c));
    return __fadd_rn(d, dpp_row_shl1(d));
}

}  // namespace gdn_qr

extern "C" __launch_bounds__(32) __global__ void gated_delta_net_q8_quadrow(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int seg,
    __half* __restrict__ s_ef_residual) {
    using namespace gdn_qr;
    const int h = blockIdx.x;
    const int L = threadIdx.x;
    const int g = L >> 2;
    const int j = L & 3;
    const int row = blockIdx.y * 8 + g;
    if (h >= n_heads) return;
    const int stride = n_heads * HD;
    if (seg <= 0 || seg > n_tokens) seg = n_tokens;

    __shared__ float4 kq[2][2][32];  // [buffer][k|q][l]

    // ---- this lane's 32 state values S[i][m] = S[row][j + 4i + 32m] and EF.
    signed char* const q8row = s_q8 + ((size_t)h * HD + row) * HD;
    __half* const efrow = s_ef_residual + ((size_t)h * HD + row) * HD;
    float S[8][4];
    __half ef[8][4];
    {
        const float scale = s_scales[h * HD + row];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int c = j + 4 * i + 32 * m;
                S[i][m] = scale * (float)q8row[c];
                ef[i][m] = efrow[c];
            }
    }

    float pk[4], pq[4], pv = 0.0f, pg = 0.0f, pb = 0.0f;
    auto fetch = [&](int t) {
        const float* kt = k + (size_t)t * stride + h * HD;
        const float* qt = q + (size_t)t * stride + h * HD;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            pk[m] = kt[L + 32 * m];
            pq[m] = qt[L + 32 * m];
        }
        pv = v[(size_t)t * stride + h * HD + row];
        pg = gate[(size_t)t * n_heads + h];
        pb = beta[(size_t)t * n_heads + h];
    };
    if (n_tokens > 0) fetch(0);
    kq[0][0][L] = make_float4(pk[0], pk[1], pk[2], pk[3]);
    kq[0][1][L] = make_float4(pq[0], pq[1], pq[2], pq[3]);
    float v_row = pv, gate_raw = pg, beta_v = pb;
    asm volatile("" ::: "memory");

    // Incumbent requant (deterministic RN + EF) of this row; the quad's four
    // lanes hold the row's 128 values. Returns the scale; qf lands in S when
    // DEQUANT (continuing from the next launch's dequant).
    auto requant = [&](auto dequant_tag, float (&qfo)[8][4]) -> float {
        constexpr bool DEQUANT = decltype(dequant_tag)::value;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int m = 0; m < 4; ++m) S[i][m] += __half2float(ef[i][m]);
        float my_max = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int m = 0; m < 4; ++m) my_max = fmaxf(my_max, fabsf(S[i][m]));
        my_max = fmaxf(my_max, dpp_quad_xor1(my_max));
        my_max = fmaxf(my_max, dpp_quad_xor2(my_max));
        const float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
        const float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float qf = fminf(fmaxf(rintf(S[i][m] * inv_s), -128.0f), 127.0f);
                ef[i][m] = __float2half(S[i][m] - qf * scale);
                qfo[i][m] = qf;
                if (DEQUANT) S[i][m] = scale * (float)(signed char)qf;
            }
        return scale;
    };

    for (int t = 0; t < n_tokens; ++t) {
        const int buf = t & 1;
        const bool has_next = t + 1 < n_tokens;
        if (has_next) fetch(t + 1);

        const float alpha = expf(gate_raw);

        float4 K[8], Q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) K[i] = kq[buf][0][j + 4 * i];
#pragma unroll
        for (int i = 0; i < 8; ++i) Q[i] = kq[buf][1][j + 4 * i];

        float p[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = partial(K[i], S[i][0], S[i][1], S[i][2], S[i][3]);
        const float kv = dpp_quad_bcast0(tree(p));

        const float delta = __fmul_rn(__builtin_fmaf(-alpha, kv, v_row), beta_v);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            S[i][0] = __builtin_fmaf(alpha, S[i][0], __fmul_rn(K[i].x, delta));
            S[i][1] = __builtin_fmaf(alpha, S[i][1], __fmul_rn(K[i].y, delta));
            S[i][2] = __builtin_fmaf(alpha, S[i][2], __fmul_rn(K[i].z, delta));
            S[i][3] = __builtin_fmaf(alpha, S[i][3], __fmul_rn(K[i].w, delta));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = partial(Q[i], S[i][0], S[i][1], S[i][2], S[i][3]);
        const float out = tree(p);
        if (j == 0) output[(size_t)t * stride + h * HD + row] = out;

        if ((t + 1) % seg == 0 && has_next) {
            float qf[8][4];
            (void)requant(std::true_type{}, qf);
        }

        if (has_next) {
            asm volatile("" ::: "memory");
            kq[buf ^ 1][0][L] = make_float4(pk[0], pk[1], pk[2], pk[3]);
            kq[buf ^ 1][1][L] = make_float4(pq[0], pq[1], pq[2], pq[3]);
            asm volatile("" ::: "memory");
            v_row = pv;
            gate_raw = pg;
            beta_v = pb;
        }
    }

    // ---- final commit (the incumbent's last-launch requant).
    float qf[8][4];
    const float scale = requant(std::false_type{}, qf);
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int c = j + 4 * i + 32 * m;
            q8row[c] = (signed char)qf[i][m];
            efrow[c] = ef[i][m];
        }
    if (j == 0) s_scales[h * HD + row] = scale;
}
