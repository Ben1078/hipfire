// claude v3: WMMA split-K speculative-verify attention for q8_0 KV caches
// (Qwen3.8-27B shape only: 24 q heads / 4 kv heads / head_dim 256), built on
// the layout conventions of attention_q8_0_fa2_gqa.gfx11.hip (FA2 prefill):
//   K plane  f16 [32 keys][256 dims] as swizzled dword pairs,
//   V plane  f16 V^T [256 dims][16 key pairs], both dequantized once per tile;
//   QK: S^T = K * Q^T on wave32 WMMA (A = K fragment, B = Q fragment);
//   PV: O^T = V^T * P^T (A = V^T fragment, B = replicated P row).
// One block = (kv head, span of keys); 8 waves. The 24 query vectors
// (6 heads x up to 4 verify rows, qi = h_local*4 + r) form two 16-row WMMA
// tiles. Wave w owns query tile (w & 1) and output dims [64*(w>>1), +64).
// QK is split by dim chunk across the 4 waves of a query tile and the
// partial scores are summed through LDS (reusing the K plane), so each
// wave holds only its 4 Q fragments in VGPRs and every WMMA is unique. Raw K/V bytes of the next tile are
// prefetched into registers before the current tile's compute.
// Partials (rows/reduce layout): [row][head][max_tiles][2+256], p[0]=m
// (natural log units), p[1]=l, then the unnormalized O. Numerics: Q/K/V/P in
// f16, f32 accumulation (FA2 class).
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef CLAUDE_A4_KERNEL
#define CLAUDE_A4_KERNEL claude_attn4
#endif
#define A4_KT 32

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef union { uint32_t u; _Float16 h[2]; } a4_dw_t;
typedef union { uint32_t u; half2_t h2; } a4_pk_t;

// lane ^ 16 exchange on the VALU (permlanex16 with identity selects).
static __device__ __forceinline__ float a4_xor16(float v) {
    const int b = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(b, b, 0x76543210, 0xfedcba98, false, false));
}

__device__ __forceinline__ int a4_swiz(int c) {
    const int r = ((c & 1) << 3) | ((c & 2) << 1) | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
}

// 48-byte aligned window covering q8_0 block `b` (wave-uniform) of a 272-byte head row.
static __device__ __forceinline__ void a4_load_window(
    const unsigned char* __restrict__ row, int b, uint32_t w[12]
) {
    const unsigned char* win = row + ((34 * b) & ~15);
    const uint4 x0 = *(const uint4*)(win);
    const uint4 x1 = *(const uint4*)(win + 16);
    const uint4 x2 = *(const uint4*)(win + 32);
    w[0] = x0.x; w[1] = x0.y; w[2] = x0.z; w[3] = x0.w;
    w[4] = x1.x; w[5] = x1.y; w[6] = x1.z; w[7] = x1.w;
    w[8] = x2.x; w[9] = x2.y; w[10] = x2.z; w[11] = x2.w;
}
template <int B>
static __device__ __forceinline__ void a4_repack_c(const uint32_t w[12], uint32_t& su, uint32_t d[8]) {
    constexpr int sh = (2 * B) & 15;
    constexpr int base = (sh + 2) >> 2;
    constexpr int bs = (sh + 2) & 3;
    su = (w[sh >> 2] >> ((sh & 3) * 8)) & 0xffffu;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (bs) d[j] = __builtin_amdgcn_alignbyte(w[base + j + 1], w[base + j], 2u);
        else d[j] = w[base + j];
    }
}
static __device__ __forceinline__ void a4_repack(const uint32_t w[12], int b, uint32_t& su, uint32_t d[8]) {
    switch (b) {
        case 0: a4_repack_c<0>(w, su, d); break;
        case 1: a4_repack_c<1>(w, su, d); break;
        case 2: a4_repack_c<2>(w, su, d); break;
        case 3: a4_repack_c<3>(w, su, d); break;
        case 4: a4_repack_c<4>(w, su, d); break;
        case 5: a4_repack_c<5>(w, su, d); break;
        case 6: a4_repack_c<6>(w, su, d); break;
        default: a4_repack_c<7>(w, su, d); break;
    }
}

// Two codes (bytes of tb picked by sel, interleaved with 0x64) times the f16
// scale pair s2: RNE(s*c) per half (FA2's fa2_pk_deq2).
__device__ __forceinline__ uint32_t a4_pk_deq2(uint32_t tb, uint32_t sel, uint32_t s2) {
    a4_pk_t a, s;
    a.u = __builtin_amdgcn_perm(0x64646464u, tb, sel);
    s.u = s2;
    const half2_t bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    a.h2 = (a.h2 - bias) * s.h2;
    return a.u;
}

__device__ __forceinline__ half16_t a4_kfrag(const uint32_t* Kdw, int key, int dc) {
    const int rot = a4_swiz(key & 15);
    const int base = key * 128;
    half16_t kf;
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int off = base + ((dc * 8 + i + rot) & 127);
        const int32x2_t pair = *(const int32x2_t*)(Kdw + off);
        a4_dw_t d0, d1;
        d0.u = (uint32_t)pair[0];
        d1.u = (uint32_t)pair[1];
        kf[2 * i] = d0.h[0];
        kf[2 * i + 1] = d0.h[1];
        kf[2 * i + 2] = d1.h[0];
        kf[2 * i + 3] = d1.h[1];
    }
    return kf;
}

__device__ __forceinline__ half16_t a4_vfrag(const uint32_t* Vdw, int d, int sub) {
    const int rot = a4_swiz(d & 15);
    const int vbase = d * 16;
    const int ps = sub * 8;
    half16_t vf;
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int off = vbase + ((ps + i + rot) & 15);
        const int32x2_t pair = *(const int32x2_t*)(Vdw + off);
        a4_dw_t d0, d1;
        d0.u = (uint32_t)pair[0];
        d1.u = (uint32_t)pair[1];
        vf[2 * i] = d0.h[0];
        vf[2 * i + 1] = d0.h[1];
        vf[2 * i + 2] = d1.h[0];
        vf[2 * i + 3] = d1.h[1];
    }
    return vf;
}

extern "C" __launch_bounds__(256, 2)
__global__ void CLAUDE_A4_KERNEL(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads, int n_kv_heads, float scale_attn,
    int span, int max_tiles, int batch_offset, int rows_valid
) {
    __shared__ __attribute__((aligned(16))) uint32_t Kdw[A4_KT * 128];
    __shared__ __attribute__((aligned(16))) uint32_t Vdw[256 * 16];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const int kv_h = blockIdx.x;
    const int span_idx = blockIdx.y;
    const int span_base = span_idx * span;
    if (kv_h >= n_kv_heads || n_heads != 24 || n_kv_heads != 4) return;

    int seq_max = 0, seq_min = 0x7fffffff;
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        if (r < rows_valid) {
            const int s = positions[batch_offset + r] + 1;
            seq_max = max(seq_max, s);
            seq_min = min(seq_min, s);
        }
    }
    if (span_base >= seq_max) return;
    const int span_end = min(span_base + span, seq_max);
    const int n_tiles = (span_end - span_base + A4_KT - 1) / A4_KT;
    const int gmax = seq_max - 1;
    const int gmin = seq_min - 1;

    // ---- wave roles ----
    const int qt = wave & 1;            // query tile: queries qt*16 .. +16
    const int dg = wave >> 1;           // dims [64*dg, +64) = dc 4*dg .. +4
    const int qi = qt * 16 + ml;        // query index (hl*4 + r)
    const int hl = qi >> 2;
    const int r = qi & 3;
    const bool qok = (qi < 24) && (r < rows_valid);
    const int head = kv_h * 6 + hl;
    const int pos_ml = qok ? positions[batch_offset + r] : 0;
    const int q_dim = n_heads * 256;

    // Q fragments (f16) for this lane pair's query, only this wave's 4 dim chunks.
    half16_t qf[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int dc = dg * 4 + i;
        half16_t f;
        if (qok) {
            const float* qp = q + (size_t)r * q_dim + head * 256 + dc * 16;
            const float4 a = *(const float4*)(qp);
            const float4 b = *(const float4*)(qp + 4);
            const float4 c = *(const float4*)(qp + 8);
            const float4 d = *(const float4*)(qp + 12);
            f[0] = (_Float16)a.x; f[1] = (_Float16)a.y; f[2] = (_Float16)a.z; f[3] = (_Float16)a.w;
            f[4] = (_Float16)b.x; f[5] = (_Float16)b.y; f[6] = (_Float16)b.z; f[7] = (_Float16)b.w;
            f[8] = (_Float16)c.x; f[9] = (_Float16)c.y; f[10] = (_Float16)c.z; f[11] = (_Float16)c.w;
            f[12] = (_Float16)d.x; f[13] = (_Float16)d.y; f[14] = (_Float16)d.z; f[15] = (_Float16)d.w;
        } else {
            #pragma unroll
            for (int e = 0; e < 16; e++) f[e] = (_Float16)0.0f;
        }
        qf[i] = f;
    }

    float8_t Ofr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) Ofr[i] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_old = -INFINITY;
    float l_val = 0.0f;

    // ---- staging: lane = key within tile, wave = q8_0 block (uniform) ----
    const int b_stage = __builtin_amdgcn_readfirstlane(wave);
    const int per_pos = 1088;
    uint32_t wk[12], wv[12];
    bool have = false;
    {
        const int t = span_base + lane;
        have = t < span_end;
        if (have) {
            a4_load_window(k_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wk);
            a4_load_window(v_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wv);
        }
    }

    for (int tile = 0; tile < n_tiles; tile++) {
        const int ktile = span_base + tile * A4_KT;
        if (tile > 0) __syncthreads();
        // ---- fill K plane (FA2 layout: pair p of key k at K[k][(p + S(k&15)) & 127]) ----
        {
            uint32_t su = 0, uc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (have) a4_repack(wk, b_stage, su, uc);
            const int k = lane;
            const int rot = a4_swiz(k & 15);
            const int base = k * 128;
            const int pb = b_stage * 16;
            const uint32_t s2 = su | (su << 16);
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t tb = uc[w] ^ 0x80808080u;
                int32x2_t v;
                v[0] = (int32_t)a4_pk_deq2(tb, 0x04010400u, s2);
                v[1] = (int32_t)a4_pk_deq2(tb, 0x04030402u, s2);
                *(int32x2_t*)(Kdw + base + ((pb + w * 2 + rot) & 127)) = v;
            }
        }
        // ---- fill V^T plane: dword (key 2m, key 2m+1) for dim d at V_T[d][(m + S(d&15)) & 15] ----
        {
            uint32_t su = 0, uc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (have) a4_repack(wv, b_stage, su, uc);
            const uint32_t s2 = su | (su << 16);
            const int m = lane >> 1;
            const int odd = lane & 1;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t tb = uc[w] ^ 0x80808080u;
                // own dequantized dims (4 per code dword): pairs (d0,d1) and (d2,d3)
                const uint32_t p01 = a4_pk_deq2(tb, 0x04010400u, s2);
                const uint32_t p23 = a4_pk_deq2(tb, 0x04030402u, s2);
                // partner lane (lane ^ 1) values via DPP quad_perm [1,0,3,2]
                const uint32_t q01 = (uint32_t)__builtin_amdgcn_mov_dpp((int)p01, 0xB1, 0xf, 0xf, true);
                const uint32_t q23 = (uint32_t)__builtin_amdgcn_mov_dpp((int)p23, 0xB1, 0xf, 0xf, true);
                const int d0 = b_stage * 32 + w * 4;
                // even lane stores dims d0, d0+2 (low halves); odd lane stores d0+1, d0+3 (high halves).
                // pair dword = (even key value, odd key value).
                uint32_t dwA, dwB;
                int dA, dB;
                if (odd == 0) {
                    dwA = __builtin_amdgcn_perm(q01, p01, 0x05040100u);   // (p01.lo, q01.lo)
                    dwB = __builtin_amdgcn_perm(q23, p23, 0x05040100u);
                    dA = d0; dB = d0 + 2;
                } else {
                    dwA = __builtin_amdgcn_perm(p01, q01, 0x07060302u);   // (q01.hi, p01.hi)
                    dwB = __builtin_amdgcn_perm(p23, q23, 0x07060302u);
                    dA = d0 + 1; dB = d0 + 3;
                }
                Vdw[dA * 16 + ((m + a4_swiz(dA & 15)) & 15)] = dwA;
                Vdw[dB * 16 + ((m + a4_swiz(dB & 15)) & 15)] = dwB;
            }
        }
        // ---- prefetch next tile ----
        if (tile + 1 < n_tiles) {
            const int t = ktile + A4_KT + lane;
            have = t < span_end;
            if (have) {
                a4_load_window(k_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wk);
                a4_load_window(v_cache + (size_t)t * per_pos + kv_h * 272, b_stage, wv);
            }
        }
        __syncthreads();

        // ---- QK partial over this wave's 4 dim chunks (S^T = K * Q^T) ----
        float8_t sacc0 = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float8_t sacc1 = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const bool do0 = ktile <= gmax;
        const bool do1 = ktile + 16 <= gmax;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dc = dg * 4 + i;
            if (do0) sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a4_kfrag(Kdw, ml, dc), qf[i], sacc0);
            if (do1) sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a4_kfrag(Kdw, 16 + ml, dc), qf[i], sacc1);
            asm volatile("" ::: "memory");
        }
        // ---- exchange partial S through the K plane (free after all QK reads) ----
        // S[qt][dg][lane][16 floats], 4-float chunks XOR-swizzled by lane for
        // conflict-free b128 stores/loads.
        __syncthreads();
        {
            float* srow = (float*)Kdw + ((qt * 4 + dg) * 32 + lane) * 16;
            const int sw = (lane >> 1) & 3;
            *(float4*)(srow + 4 * (0 ^ sw)) = (float4){sacc0[0], sacc0[1], sacc0[2], sacc0[3]};
            *(float4*)(srow + 4 * (1 ^ sw)) = (float4){sacc0[4], sacc0[5], sacc0[6], sacc0[7]};
            *(float4*)(srow + 4 * (2 ^ sw)) = (float4){sacc1[0], sacc1[1], sacc1[2], sacc1[3]};
            *(float4*)(srow + 4 * (3 ^ sw)) = (float4){sacc1[4], sacc1[5], sacc1[6], sacc1[7]};
        }
        __syncthreads();
        {
            const int sw = (lane >> 1) & 3;
            sacc0 = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            sacc1 = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const float* srow = (const float*)Kdw + ((qt * 4 + g) * 32 + lane) * 16;
                const float4 a = *(const float4*)(srow + 4 * (0 ^ sw));
                const float4 b = *(const float4*)(srow + 4 * (1 ^ sw));
                const float4 c = *(const float4*)(srow + 4 * (2 ^ sw));
                const float4 d = *(const float4*)(srow + 4 * (3 ^ sw));
                sacc0[0] += a.x; sacc0[1] += a.y; sacc0[2] += a.z; sacc0[3] += a.w;
                sacc0[4] += b.x; sacc0[5] += b.y; sacc0[6] += b.z; sacc0[7] += b.w;
                sacc1[0] += c.x; sacc1[1] += c.y; sacc1[2] += c.z; sacc1[3] += c.w;
                sacc1[4] += d.x; sacc1[5] += d.y; sacc1[6] += d.z; sacc1[7] += d.w;
            }
        }
        // ---- softmax + PV per subtile (FA2 order) ----
        #pragma unroll 1
        for (int sub = 0; sub < 2; ++sub) {
            const bool dos = (sub == 0) ? do0 : do1;
            if (!dos) continue;
            float8_t sacc = (sub == 0) ? sacc0 : sacc1;
            const int kt_sub = ktile + sub * 16;
            const bool full = (kt_sub + 15) <= gmin;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key_g = kt_sub + 2 * j + half;
                const bool ok = qok && (full || key_g <= pos_ml);
                sacc[j] = ok ? sacc[j] * scale_attn : -INFINITY;
            }
            float mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < 8; ++j) mx = fmaxf(mx, sacc[j]);
            mx = fmaxf(mx, a4_xor16(mx));
            const float m_new = fmaxf(m_old, mx);
            const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
            m_old = m_new;
            float earr[8];
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e = (m_new == -INFINITY) ? 0.0f : __expf(sacc[j] - m_new);
                earr[j] = e;
                sum += e;
            }
            sum += a4_xor16(sum);
            l_val = l_val * alpha + sum;
            half16_t prow;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float peer = a4_xor16(earr[j]);
                const float e0 = (half == 0) ? earr[j] : peer;
                const float e1 = (half == 0) ? peer : earr[j];
                prow[2 * j] = (_Float16)e0;
                prow[2 * j + 1] = (_Float16)e1;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) Ofr[i] *= alpha;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = (dg * 4 + i) * 16 + ml;
                Ofr[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a4_vfrag(Vdw, d, sub), prow, Ofr[i]);
                asm volatile("" ::: "memory");
            }
        }
    }

    // ---- partials ----
    if (qok) {
        float* pp = partials
                  + ((long long)r * n_heads + head) * max_tiles * (2 + 256)
                  + (long long)span_idx * (2 + 256);
        if (dg == 0 && half == 0) {
            pp[0] = m_old;
            pp[1] = l_val;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dbase = (dg * 4 + i) * 16 + half;
            #pragma unroll
            for (int j = 0; j < 8; ++j) pp[2 + dbase + 2 * j] = Ofr[i][j];
        }
    }
}
