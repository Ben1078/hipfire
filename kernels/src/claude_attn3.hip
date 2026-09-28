// claude v3: GQA-shared, key-parallel, LDS-staged speculative-verify attention
// for q8_0 KV caches with head_dim 256 (Qwen3.8-27B: 24 q heads / 4 kv heads).
//
// One block = (kv head, span of `span` keys). It stages the span tile by tile
// (32 keys) in LDS: K as raw int8 + f32 block scales, V likewise, and holds the
// 24 query vectors (6 heads x up to 4 verify rows, pre-scaled by scale_attn)
// as f16. Each of the 8 waves owns 3 queries. In the QK phase a lane owns one
// key of the tile and forms 3 scores with v_dot2_f32_f16 (K bytes are turned
// into f16 with the 0x6400 magic-exponent trick; the 1152*sum(q) offset is
// removed per 32-block before the block scale is applied). In the PV phase a
// lane owns 8 output dims and walks the tile's keys, broadcasting the
// probabilities with ds_bpermute. Online softmax in log2 domain, as the rows
// kernel. Partials are written in the rows/reduce layout:
//   [row][head][max_tiles][2 + 256], p[0] = m*ln2, p[1] = l, then acc.
// The next tile's global loads are issued before the current tile's compute.
//
// Grid: [n_kv_heads, ceil(seq_max / span), 1]. Block: 256. Numerics: q -> f16
// (rel 5e-4), everything else exact int8 * f32 scale as the rows kernel.
#include <hip/hip_runtime.h>

#ifndef CLAUDE_A3_KERNEL
#define CLAUDE_A3_KERNEL claude_attn3
#endif
#define A3_G 6
#define A3_R 4
#define A3_NQ (A3_G * A3_R)
#define A3_WAVES 8
#define A3_QPW (A3_NQ / A3_WAVES)
#define A3_TILE 32
#define A3_HD 256
#define A3_KSTRIDE 304
#define A3_VSTRIDE 288
#define A3_NEG (-1e30f)
#ifndef A3_QK_UNROLL
#define A3_QK_UNROLL 2
#endif
#ifndef A3_PV_UNROLL
#define A3_PV_UNROLL 4
#endif
#define A3_PRAGMA(x) _Pragma(#x)
#define A3_UNROLL(n) A3_PRAGMA(unroll n)

typedef _Float16 a3_h2 __attribute__((ext_vector_type(2)));

// Wave32 reductions without LDS: XOR butterfly inside each 16-lane row via
// DPP row_xmask, then swap halves with permlanex16 (all lanes get the result).
template <int CTRL>
static __device__ __forceinline__ float a3_dpp_xmask(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}
static __device__ __forceinline__ float a3_half_swap(float v) {
    const int b = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(b, b, 0x76543210, 0xfedcba98, false, false));
}
static __device__ __forceinline__ float a3_wave_max(float v) {
    v = fmaxf(v, a3_dpp_xmask<0x161>(v));
    v = fmaxf(v, a3_dpp_xmask<0x162>(v));
    v = fmaxf(v, a3_dpp_xmask<0x164>(v));
    v = fmaxf(v, a3_dpp_xmask<0x168>(v));
    return fmaxf(v, a3_half_swap(v));
}
static __device__ __forceinline__ float a3_wave_sum(float v) {
    v += a3_dpp_xmask<0x161>(v);
    v += a3_dpp_xmask<0x162>(v);
    v += a3_dpp_xmask<0x164>(v);
    v += a3_dpp_xmask<0x168>(v);
    return v + a3_half_swap(v);
}
static __device__ __forceinline__ float a3_readlane(float v, int lane) {
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), lane));
}

static __device__ __forceinline__ float a3_h16_to_f32(unsigned int bits16) {
    _Float16 h;
    unsigned short s = (unsigned short)bits16;
    __builtin_memcpy(&h, &s, 2);
    return (float)h;
}

static __device__ __forceinline__ a3_h2 a3_as_h2(unsigned int w) {
    a3_h2 r;
    __builtin_memcpy(&r, &w, 4);
    return r;
}

// Load one 34-byte q8_0 block (2-byte f16 scale + 32 int8) that starts at byte
// offset 34*b inside a 16-byte aligned 272-byte head row; `b` is wave-uniform.
static __device__ __forceinline__ void a3_load_block(
    const unsigned char* __restrict__ row, int b, unsigned int w[12]
) {
#ifdef A3_SKIP_LOAD
    #pragma unroll
    for (int i = 0; i < 12; i++) w[i] = 0x01010101u * (unsigned)(b + 1);
    return;
#endif
    const unsigned char* win = row + ((34 * b) & ~15);
    const uint4 x0 = *(const uint4*)(win);
    const uint4 x1 = *(const uint4*)(win + 16);
    const uint4 x2 = *(const uint4*)(win + 32);
    w[0] = x0.x; w[1] = x0.y; w[2] = x0.z; w[3] = x0.w;
    w[4] = x1.x; w[5] = x1.y; w[6] = x1.z; w[7] = x1.w;
    w[8] = x2.x; w[9] = x2.y; w[10] = x2.z; w[11] = x2.w;
}

// Repack a loaded 48-byte window into (scale f32, 8 data dwords). `B` is the
// q8_0 block index within the head row (compile-time so no dynamic indexing).
template <int B>
static __device__ __forceinline__ void a3_repack_c(
    const unsigned int w[12], float& scale, unsigned int d[8]
) {
    constexpr int sh = (2 * B) & 15;         // block start within window
    constexpr int base = (sh + 2) >> 2;      // data starts at sh + 2
    constexpr int bs = (sh + 2) & 3;         // 0 or 2
    const unsigned int sw = w[sh >> 2] >> ((sh & 3) * 8);
    scale = a3_h16_to_f32(sw & 0xffffu);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (bs) d[j] = __builtin_amdgcn_alignbyte(w[base + j + 1], w[base + j], 2u);
        else d[j] = w[base + j];
    }
}

static __device__ __forceinline__ void a3_repack(
    const unsigned int w[12], int b, float& scale, unsigned int d[8]
) {
    switch (b) {
        case 0: a3_repack_c<0>(w, scale, d); break;
        case 1: a3_repack_c<1>(w, scale, d); break;
        case 2: a3_repack_c<2>(w, scale, d); break;
        case 3: a3_repack_c<3>(w, scale, d); break;
        case 4: a3_repack_c<4>(w, scale, d); break;
        case 5: a3_repack_c<5>(w, scale, d); break;
        case 6: a3_repack_c<6>(w, scale, d); break;
        default: a3_repack_c<7>(w, scale, d); break;
    }
}

extern "C" __launch_bounds__(256, 2)
__global__ void CLAUDE_A3_KERNEL(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads, int n_kv_heads, float scale_attn,
    int span, int max_tiles, int batch_offset, int rows_valid
) {
    __shared__ __attribute__((aligned(16))) unsigned char s_k[A3_TILE * A3_KSTRIDE];
    __shared__ __attribute__((aligned(16))) unsigned char s_v[A3_TILE * A3_VSTRIDE];
    __shared__ __attribute__((aligned(16))) unsigned short s_q[A3_NQ * A3_HD];
    __shared__ float s_qsum[A3_NQ * 8];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int kv_h = blockIdx.x;
    const int span_idx = blockIdx.y;
    const int span_base = span_idx * span;
    if (kv_h >= n_kv_heads) return;

    int seq[A3_R];
    int seq_max = 0;
    #pragma unroll
    for (int r = 0; r < A3_R; r++) {
        seq[r] = (r < rows_valid) ? (positions[batch_offset + r] + 1) : 0;
        seq_max = max(seq_max, seq[r]);
    }
    if (span_base >= seq_max) return;

    // Queries -> f16 (pre-scaled). qi = h_local * A3_R + r.
    const int q_dim = n_heads * A3_HD;
    for (int e = tid; e < A3_NQ * A3_HD; e += 256) {
        const int qi = e >> 8;
        const int d = e & 255;
        const int hl = qi / A3_R;
        const int r = qi - hl * A3_R;
        const int row = (r < rows_valid) ? r : 0;
        const float v = q[(size_t)row * q_dim + (kv_h * A3_G + hl) * A3_HD + d] * scale_attn;
        const _Float16 h = (_Float16)v;
        unsigned short bits;
        __builtin_memcpy(&bits, &h, 2);
        s_q[e] = bits;
    }
    __syncthreads();
    if (tid < A3_NQ * 8) {
        const int qi = tid >> 3;
        const int b = tid & 7;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) s += a3_h16_to_f32(s_q[qi * A3_HD + b * 32 + i]);
        s_qsum[tid] = s;
    }

    // Per-wave query bookkeeping.
    int qseq[A3_QPW];
    #pragma unroll
    for (int j = 0; j < A3_QPW; j++) {
        const int qi = wave * A3_QPW + j;
        const int r = qi % A3_R;
        qseq[j] = seq[r];
    }
    float m_run[A3_QPW], l_run[A3_QPW], acc[A3_QPW][8];
    #pragma unroll
    for (int j = 0; j < A3_QPW; j++) {
        m_run[j] = A3_NEG;
        l_run[j] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[j][i] = 0.0f;
    }

    const int span_end = min(span_base + span, seq_max);
    const int n_tiles = (span_end - span_base + A3_TILE - 1) / A3_TILE;
    const int per_pos = n_kv_heads * 8 * 34;   // 1088 bytes per position
    const int b_stage = __builtin_amdgcn_readfirstlane(wave);   // staging: wave = q8_0 block (uniform), lane = key

    unsigned int wk[12], wv[12];
    bool have = false;
    {
        const int t = span_base + lane;
        have = t < span_end;
        if (have) {
            a3_load_block(k_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wk);
            a3_load_block(v_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wv);
        }
    }

    for (int tile = 0; tile < n_tiles; tile++) {
        const int tile_base = span_base + tile * A3_TILE;
        if (tile > 0) __syncthreads();   // previous tile fully consumed
        // Stage this tile from the prefetched registers.
        {
            unsigned char* krow = s_k + lane * A3_KSTRIDE;
            unsigned char* vrow = s_v + lane * A3_VSTRIDE;
            float ks = 0.0f, vs = 0.0f;
            unsigned int dk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            unsigned int dv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (have) {
                a3_repack(wk, b_stage, ks, dk);
                a3_repack(wv, b_stage, vs, dv);
            }
            *(float*)(krow + 4 * b_stage) = ks;
            *(float*)(vrow + 4 * b_stage) = vs;
            *(uint4*)(krow + 32 + 32 * b_stage) = make_uint4(dk[0], dk[1], dk[2], dk[3]);
            *(uint4*)(krow + 48 + 32 * b_stage) = make_uint4(dk[4], dk[5], dk[6], dk[7]);
            *(uint4*)(vrow + 32 + 32 * b_stage) = make_uint4(dv[0], dv[1], dv[2], dv[3]);
            *(uint4*)(vrow + 48 + 32 * b_stage) = make_uint4(dv[4], dv[5], dv[6], dv[7]);
        }
        // Prefetch the next tile.
        if (tile + 1 < n_tiles) {
            const int t = tile_base + A3_TILE + lane;
            have = t < span_end;
            if (have) {
                a3_load_block(k_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wk);
                a3_load_block(v_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wv);
            }
        }
        __syncthreads();

        // ── QK: lane owns key `lane` of the tile ──
        const int t_key = tile_base + lane;
        float s[A3_QPW];
#ifdef A3_SKIP_QK
        #pragma unroll
        for (int j = 0; j < A3_QPW; j++) s[j] = (float)lane * 1e-3f;
        if (false)
#endif
        {
            const unsigned char* krow = s_k + lane * A3_KSTRIDE;
            float ks[8];
            {
                const float4 a = *(const float4*)(krow);
                const float4 c = *(const float4*)(krow + 16);
                ks[0] = a.x; ks[1] = a.y; ks[2] = a.z; ks[3] = a.w;
                ks[4] = c.x; ks[5] = c.y; ks[6] = c.z; ks[7] = c.w;
            }
            #pragma unroll
            for (int j = 0; j < A3_QPW; j++) s[j] = 0.0f;
            A3_UNROLL(A3_QK_UNROLL)
            for (int b = 0; b < 8; b++) {
                const uint4 x0 = *(const uint4*)(krow + 32 + 32 * b);
                const uint4 x1 = *(const uint4*)(krow + 48 + 32 * b);
                const unsigned int kw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                a3_h2 kh[16];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const unsigned int u = kw[i] ^ 0x80808080u;
                    kh[2 * i]     = a3_as_h2(__builtin_amdgcn_perm(0x64646464u, u, 0x04010400u));
                    kh[2 * i + 1] = a3_as_h2(__builtin_amdgcn_perm(0x64646464u, u, 0x04030402u));
                }
                #pragma unroll
                for (int j = 0; j < A3_QPW; j++) {
                    const int qi = wave * A3_QPW + j;
                    const unsigned short* qb = s_q + qi * A3_HD + b * 32;
                    const uint4 y0 = *(const uint4*)(qb);
                    const uint4 y1 = *(const uint4*)(qb + 8);
                    const uint4 y2 = *(const uint4*)(qb + 16);
                    const uint4 y3 = *(const uint4*)(qb + 24);
                    const unsigned int qw[16] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w,
                                                 y2.x, y2.y, y2.z, y2.w, y3.x, y3.y, y3.z, y3.w};
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 16; i++)
                        dot = __builtin_amdgcn_fdot2(kh[i], a3_as_h2(qw[i]), dot, false);
                    dot -= 1152.0f * s_qsum[qi * 8 + b];
                    s[j] = fmaf(ks[b], dot, s[j]);
                }
            }
        }
        // ── online softmax per query (wave-level) ──
        float p[A3_QPW];
        #pragma unroll
        for (int j = 0; j < A3_QPW; j++) {
            float sv = (t_key < qseq[j]) ? s[j] : A3_NEG;
            const float mt = a3_wave_max(sv);
            const float m_new = fmaxf(m_run[j], mt);
            const float pv = (sv > -1e29f) ? __builtin_amdgcn_exp2f(sv - m_new) : 0.0f;
            const float lt = a3_wave_sum(pv);
            const float alpha = __builtin_amdgcn_exp2f(m_run[j] - m_new);
            l_run[j] = l_run[j] * alpha + lt;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[j][i] *= alpha;
            m_run[j] = m_new;
            p[j] = pv;
        }
        // ── PV: lane owns dims 8*lane .. +8 ──
        const int n_keys = min(A3_TILE, span_end - tile_base);
#ifdef A3_SKIP_PV
        if (false)
#endif
        A3_UNROLL(A3_PV_UNROLL)
        for (int t = 0; t < n_keys; t++) {
            const unsigned char* vrow = s_v + t * A3_VSTRIDE;
            const float vs = *(const float*)(vrow + 4 * (lane >> 2));
            const uint2 vv = *(const uint2*)(vrow + 32 + 8 * lane);
            float pv[A3_QPW];
            #pragma unroll
            for (int j = 0; j < A3_QPW; j++) pv[j] = a3_readlane(p[j], t) * vs;
            float vd[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                vd[i]     = (float)(signed char)(vv.x >> (8 * i));
                vd[4 + i] = (float)(signed char)(vv.y >> (8 * i));
            }
            #pragma unroll
            for (int j = 0; j < A3_QPW; j++) {
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[j][i] = fmaf(pv[j], vd[i], acc[j][i]);
            }
        }
    }

    // ── partials ──
    #pragma unroll
    for (int j = 0; j < A3_QPW; j++) {
        const int qi = wave * A3_QPW + j;
        const int hl = qi / A3_R;
        const int r = qi - hl * A3_R;
        if (r >= rows_valid) continue;
        const int head = kv_h * A3_G + hl;
        float* pp = partials
                  + ((long long)r * n_heads + head) * max_tiles * (2 + A3_HD)
                  + (long long)span_idx * (2 + A3_HD);
        if (lane == 0) {
            pp[0] = m_run[j] * 0.69314718f;
            pp[1] = l_run[j];
        }
        *(float4*)(pp + 2 + 8 * lane)     = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
        *(float4*)(pp + 2 + 8 * lane + 4) = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
    }
}
