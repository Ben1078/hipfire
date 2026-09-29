// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// claude fa2 (gfx1100/gfx1151): latency-optimized compute waves for the
// GQA-fused Q8 FA2 prefill (derived from attention_q8_0_fa2_gqa.gfx11.hip,
// production recipe: Q16 tile, KT32, warp-specialized K/V fill). The fill
// waves, LDS planes/swizzle, barriers and epilogue are the upstream ones
// verbatim; only the six compute waves' tile loop changes:
//
//  CL_PIPE   Software-pipelined operand fetch. The upstream QK loop issues the
//            dim-chunk Q load (global, L2) and the K fragment (LDS) and then
//            waits vmcnt(0)/lgkmcnt(0) before every WMMA pair, and PV waits
//            lgkmcnt(0) before every WMMA: each chunk exposes a full memory
//            round trip. Here Q is prefetched CL_QD chunks ahead (rotating
//            registers; the next tile's first chunks are issued before B2),
//            K and V fragments one chunk ahead.
//  CL_XLANE  Lane-pair (xor 16) exchanges via v_permlanex16 (one VALU op)
//            instead of __shfl_xor (ds_bpermute + LDS round trip; ten
//            serialized round trips per 16-key subtile upstream). The P row
//            is exchanged as four packed f16 dwords + v_perm.
//  CL_LAZY   Exact lazy O rescale: skip the 128-register O*=alpha when no lane
//            of the wave raised its running max (alpha == 1.0f exactly).
//  CL_SM32   Online softmax over the whole 32-key tile (one max/alpha/sum per
//            tile instead of two) with log2-domain scores (scale*log2e folded
//            into one multiply, exp2 directly). Changes rounding (not
//            bit-exact vs upstream; same f16-P class).
//
// With CL_SM32=0 every arithmetic op on valid rows matches upstream in value
// and order, so the output is bit-identical to upstream (bench checks).
// Rows past batch_size load a clamped valid row (unconditional loads) and are
// never stored.

#include <hip/hip_runtime.h>

#ifndef CL_PIPE
#define CL_PIPE 1
#endif
#ifndef CL_QD
#define CL_QD 2
#endif
#ifndef CL_XLANE
#define CL_XLANE 1
#endif
#ifndef CL_LAZY
#define CL_LAZY 1
#endif
#ifndef CL_SM32
#define CL_SM32 0
#endif
// 32 query positions per workgroup (requires CL_FILL2, no CL_QLDS): twelve
// compute waves (6 heads x 2 position halves) + the two fill waves share each
// K/V tile, halving fill VALU and barrier syncs per FLOP. 448 threads,
// <= 216 VGPRs -> 7 waves/SIMD -> 2 workgroups per WGP.
#ifndef CL_Q32
#define CL_Q32 0
#endif
#if CL_Q32 && (!CL_B128 || CL_QLDS)
#error "CL_Q32 requires CL_B128 and no CL_QLDS"
#endif
#define CL_NCW (CL_Q32 ? 12 : 6)
#define CL_QPOS (CL_Q32 ? 32 : 16)
#define CL_THREADS (CL_NCW * 32 + 64)
// Fill-wave issue priority (s_setprio 0..3; 0 = hardware default).
#ifndef CL_HPRIO
#define CL_HPRIO 0
#endif
#ifndef CL_WAVES_PER_EU
#define CL_WAVES_PER_EU (CL_Q32 ? 7 : 6)
#endif
// K plane layout: 1 = padded rows (130 dwords per key; row ml starts at bank
// 2*ml, so the 16 unique rows of a b64 fragment read cover all banks once and
// every chunk address is base + immediate); 0 = upstream rotation swizzle.
#ifndef CL_KPAD
#define CL_KPAD 1
#endif
// Fill v2 (requires CL_KPAD): helper lanes own one key (K) / four keys of one
// 32-dim block (V); SGPR-base + 32-bit offset loads; OOB keys read a clamped
// valid row (always masked, so no zeroing); V^T plane padded (row stride 18
// dwords + 16 per 32-row block) so both the b64 fill stores and the PV
// fragment reads are bank-conflict-free and all read offsets are immediates.
#ifndef CL_FILL2
#define CL_FILL2 1
#endif
// Zero K/V of keys >= seq_len in fill v2 (upstream semantics). Those keys are
// always masked (P = 0), but the WMMA result is not invariant to the finite
// values under P = 0 lanes (measured: 1e-7 rel drift vs upstream), so the
// tail tile zero-fills them; full tiles take a uniform fast path.
#ifndef CL_ZERO_OOB
#define CL_ZERO_OOB 1
#endif
#if CL_FILL2 && !CL_KPAD
#error "CL_FILL2 requires CL_KPAD"
#endif
// b128 fragment layout (requires CL_FILL2): K row stride 132 dwords and V^T
// row stride 20 dwords (+16 per 32-row block), both 16-byte aligned, so every
// fragment is two ds_load_b128 with immediate offsets (no address VALU) and
// each 8-lane quarter-wave pass covers all 32 banks once.
#ifndef CL_B128
#define CL_B128 1
#endif
#if CL_B128 && !CL_FILL2
#error "CL_B128 requires CL_FILL2"
#endif
// Q chunks staged in LDS (requires CL_B128): the first CL_QLDS of the 16 Q dim
// chunks of every compute wave are copied to LDS once per workgroup (512 B per
// wave-chunk, halves XOR-swizzled by row bit 2 so 8-lane b128 passes are
// conflict-free) and read from there each tile; the rest stay global
// reloads. 9 chunks fill the 64 KiB workgroup limit (2 WGs per WGP).
#ifndef CL_QLDS
#define CL_QLDS 0
#endif
#if CL_QLDS && !CL_B128
#error "CL_QLDS requires CL_B128"
#endif
// Timing-only knockouts (wrong results; never used by the launcher):
// CL_KO_FILL helpers skip loads/dequant/stores, CL_KO_BAR drops the two
// per-tile barriers, CL_KO_SM replaces the softmax with P = f16(S),
// CL_KO_QK / CL_KO_PV skip the WMMAs, CL_KO_QLOAD reuses one Q chunk.
#ifndef CL_KO_FILL
#define CL_KO_FILL 0
#endif
#ifndef CL_KO_BAR
#define CL_KO_BAR 0
#endif
#ifndef CL_KO_SM
#define CL_KO_SM 0
#endif
#ifndef CL_KO_QK
#define CL_KO_QK 0
#endif
#ifndef CL_KO_PV
#define CL_KO_PV 0
#endif
#ifndef CL_KO_COMPUTE
#define CL_KO_COMPUTE 0
#endif
#ifndef CL_KO_QLOAD
#define CL_KO_QLOAD 0
#endif
#if CL_KO_BAR
#define CL_TILE_BARRIER() do {} while (0)
#else
#define CL_TILE_BARRIER() __syncthreads()
#endif
#ifndef CLAUDE_FA2_ENTRY
#define CLAUDE_FA2_ENTRY attention_q8_0_fa2_gqa_claude_gfx11
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef uint32_t __attribute__((ext_vector_type(8))) uint8v_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2_dw_t;

typedef union {
    uint32_t u;
    half2_t h2;
} fa2_pk_t;

__device__ __forceinline__ int fa2_swiz(int c)
{
    const int r = ((c & 1) << 3) | ((c & 2) << 1)
                | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
}

// ---- Warp-specialized fill (upstream FA2_FILL helpers, verbatim) ----
constexpr int FA2_FILL_HT = 64;
constexpr int FA2_FILL_HK = 32 * 8 / 64;
constexpr int FA2_FILL_HV = 32 * 16 / 64;
constexpr int FA2_ROW_B = 4 * (256 / 32) * 34;
#if CL_KPAD && CL_B128
constexpr int CL_KROW = 132;
#elif CL_KPAD
constexpr int CL_KROW = 130;
#else
constexpr int CL_KROW = 128;
#endif
constexpr int CL_KPLANE = 32 * CL_KROW;           // K plane dwords
#if CL_FILL2
// V^T row d (dim) at d*VROW + 16*(d>>5); 16 key-pair dwords + pad.
#if CL_B128
constexpr int CL_VROW = 20;
#else
constexpr int CL_VROW = 18;
#endif
__device__ __forceinline__ constexpr int cl_vrow(int d) { return d * CL_VROW + 16 * (d >> 5); }
constexpr int CL_VPLANE = 256 * CL_VROW + 16 * 7;  // 4720 (18) / 5232 (20) dwords
#else
constexpr int CL_VPLANE = 32 * 128;
#endif
constexpr int CL_QSTAGE = CL_QLDS * 6 * 128;        // Q stage dwords
constexpr int CL_LDS_DWORDS = CL_KPLANE + CL_VPLANE + CL_QSTAGE;
// Causal-bound mailbox: two dwords of V-plane padding / tail (transient).
constexpr int CL_MAILBOX = CL_KPLANE + CL_VPLANE - 2;

__device__ __forceinline__ uint32_t fa2_pk_deq2(uint32_t tb, uint32_t sel, uint32_t s2)
{
    fa2_pk_t a, s;
    a.u = __builtin_amdgcn_perm(0x64646464u, tb, sel);
    s.u = s2;
    const half2_t bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    a.h2 = (a.h2 - bias) * s.h2;
    return a.u;
}

__device__ __forceinline__ void fa2_fill_k_load(
    const unsigned char* k_cache, int ktile, int seq_len, int kv_blk, int idx,
    uint32_t& su, uint32_t* uc)
{
    const int gk = ktile + (idx >> 3);
    const bool ok = gk < seq_len;
    const unsigned char* blk = k_cache
        + (unsigned long long)min(gk, seq_len - 1) * (unsigned long long)FA2_ROW_B
        + (unsigned long long)(kv_blk + (idx & 7)) * 34ULL;
    const uint32_t s = (uint32_t)blk[0] | ((uint32_t)blk[1] << 8);
    su = ok ? s : 0u;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t u;
        __builtin_memcpy(&u, blk + 2 + w * 4, 4);
        uc[w] = ok ? u : 0u;
    }
}

__device__ __forceinline__ void fa2_fill_k_store(uint32_t* Kdw, int idx, uint32_t su,
                                                 const uint32_t* uc)
{
    const int k = idx >> 3;
    const int pb = (idx & 7) * 16;
    const uint32_t s2 = su | (su << 16);
#if CL_KPAD
    const int base = k * CL_KROW + pb;
#else
    const int rot = fa2_swiz(k & 15);
    const int base = k * 128;
#endif
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t tb = uc[w] ^ 0x80808080u;
        int32x2_t v;
        v[0] = (int32_t)fa2_pk_deq2(tb, 0x04010400u, s2);
        v[1] = (int32_t)fa2_pk_deq2(tb, 0x04030402u, s2);
#if CL_KPAD
        *(int32x2_t*)(Kdw + base + w * 2) = v;
#else
        *(int32x2_t*)(Kdw + base + ((pb + w * 2 + rot) & 127)) = v;
#endif
    }
}

__device__ __forceinline__ void fa2_fill_v_load(
    const unsigned char* v_cache, int ktile, int seq_len, int kv_blk, int slot,
    uint32_t& s01, uint32_t* u)
{
    const int pair = slot >> 2;
    const int w_base = (slot & 3) * 2;
    const int k0 = ktile + 2 * (pair >> 3);
    const unsigned long long boff = (unsigned long long)(kv_blk + (pair & 7)) * 34ULL;
    const bool ok0 = k0 < seq_len;
    const bool ok1 = k0 + 1 < seq_len;
    const unsigned char* blk0 = v_cache
        + (unsigned long long)min(k0, seq_len - 1) * (unsigned long long)FA2_ROW_B + boff;
    const unsigned char* blk1 = v_cache
        + (unsigned long long)min(k0 + 1, seq_len - 1) * (unsigned long long)FA2_ROW_B + boff;
    const uint32_t s0 = ok0 ? ((uint32_t)blk0[0] | ((uint32_t)blk0[1] << 8)) : 0u;
    const uint32_t s1 = ok1 ? ((uint32_t)blk1[0] | ((uint32_t)blk1[1] << 8)) : 0u;
    s01 = s0 | (s1 << 16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint32_t u0, u1;
        __builtin_memcpy(&u0, blk0 + 2 + (w_base + i) * 4, 4);
        __builtin_memcpy(&u1, blk1 + 2 + (w_base + i) * 4, 4);
        u[i] = ok0 ? u0 : 0u;
        u[2 + i] = ok1 ? u1 : 0u;
    }
}

__device__ __forceinline__ void fa2_fill_v_store(uint32_t* Vdw, int slot, uint32_t s01,
                                                 const uint32_t* u)
{
    const int pair = slot >> 2;
    const int w_base = (slot & 3) * 2;
    const int m = pair >> 3;
    const int b = pair & 7;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int w = w_base + i;
        const uint32_t xl =
            __builtin_amdgcn_perm(u[2 + i], u[i], 0x05010400u) ^ 0x80808080u;
        const uint32_t xh =
            __builtin_amdgcn_perm(u[2 + i], u[i], 0x07030602u) ^ 0x80808080u;
        uint32_t dw[4];
        dw[0] = fa2_pk_deq2(xl, 0x04010400u, s01);
        dw[1] = fa2_pk_deq2(xl, 0x04030402u, s01);
        dw[2] = fa2_pk_deq2(xh, 0x04010400u, s01);
        dw[3] = fa2_pk_deq2(xh, 0x04030402u, s01);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int d = b * 32 + w * 4 + c;
            Vdw[d * 16 + ((m + fa2_swiz(d & 15)) & 15)] = dw[c];
        }
    }
}

#if CL_FILL2
// ---- Fill v2 ----
// K: helper lane h (0..63) owns key h&31 of the tile, blocks (h>>5)+2t, t<4.
// Raw: s[t] = f16 scale bits, u[8t..8t+7] = the 32 code bytes.
__device__ __forceinline__ void cl_k2_load(const unsigned char* __restrict__ k_cache,
                                           int ktile, int seq_len, int h,
                                           uint32_t kv_boff, uint32_t* s, uint32_t* u)
{
    const int g = min(ktile + (h & 31), seq_len - 1);
    const uint32_t roff = (uint32_t)g * (uint32_t)FA2_ROW_B + kv_boff + (uint32_t)(h >> 5) * 34u;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const unsigned char* blk = k_cache + (roff + (uint32_t)t * 68u);
        s[t] = (uint32_t)*(const uint16_t*)blk;
        __builtin_memcpy(u + 8 * t, blk + 2, 32);
#if CL_ZERO_OOB
        if (ktile + 32 > seq_len && ktile + (h & 31) >= seq_len) {
            s[t] = 0u;
#pragma unroll
            for (int w = 0; w < 8; ++w) u[8 * t + w] = 0u;
        }
#endif
    }
}

__device__ __forceinline__ void cl_k2_store(uint32_t* Kdw, int h, const uint32_t* s,
                                            const uint32_t* u)
{
    const int k = h & 31;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int base = k * CL_KROW + ((h >> 5) + 2 * t) * 16;
        const uint32_t s2 = s[t] | (s[t] << 16);
#if CL_B128
        typedef uint32_t __attribute__((ext_vector_type(4))) u4_t;
#pragma unroll
        for (int w = 0; w < 8; w += 2) {
            const uint32_t t0 = u[8 * t + w] ^ 0x80808080u;
            const uint32_t t1 = u[8 * t + w + 1] ^ 0x80808080u;
            u4_t v;
            v[0] = fa2_pk_deq2(t0, 0x04010400u, s2);
            v[1] = fa2_pk_deq2(t0, 0x04030402u, s2);
            v[2] = fa2_pk_deq2(t1, 0x04010400u, s2);
            v[3] = fa2_pk_deq2(t1, 0x04030402u, s2);
            *(u4_t*)(Kdw + base + w * 2) = v;
        }
#else
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t tb = u[8 * t + w] ^ 0x80808080u;
            int32x2_t v;
            v[0] = (int32_t)fa2_pk_deq2(tb, 0x04010400u, s2);
            v[1] = (int32_t)fa2_pk_deq2(tb, 0x04030402u, s2);
            *(int32x2_t*)(Kdw + base + w * 2) = v;
        }
#endif
    }
}

// V: helper lane h owns keys 4*(h&7)+j (j<4) of 32-dim block q = h>>3.
__device__ __forceinline__ void cl_v2_load(const unsigned char* __restrict__ v_cache,
                                           int ktile, int seq_len, int h,
                                           uint32_t kv_boff, uint32_t* s, uint32_t* u)
{
    const uint32_t boff = kv_boff + (uint32_t)(h >> 3) * 34u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int g = min(ktile + 4 * (h & 7) + j, seq_len - 1);
        const unsigned char* blk = v_cache + ((uint32_t)g * (uint32_t)FA2_ROW_B + boff);
        s[j] = (uint32_t)*(const uint16_t*)blk;
        __builtin_memcpy(u + 8 * j, blk + 2, 32);
#if CL_ZERO_OOB
        if (ktile + 32 > seq_len && ktile + 4 * (h & 7) + j >= seq_len) {
            s[j] = 0u;
#pragma unroll
            for (int w = 0; w < 8; ++w) u[8 * j + w] = 0u;
        }
#endif
    }
}

// V^T plane: dword (d, m) = (key 2m, key 2m+1) at dim d. Lane (mm, q) writes
// key pairs 2mm, 2mm+1 (one b64) for the 32 dims of block q. Sixteen lanes
// (mm 0..7 x q parity) hit sixteen distinct even banks per store.
__device__ __forceinline__ void cl_v2_store(uint32_t* Vdw, int h, const uint32_t* s,
                                            const uint32_t* u)
{
    const int mm = h & 7;
    const int q = h >> 3;
    const uint32_t sab = s[0] | (s[1] << 16);
    const uint32_t scd = s[2] | (s[3] << 16);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t xa = u[w] ^ 0x80808080u;
        const uint32_t xb = u[8 + w] ^ 0x80808080u;
        const uint32_t xc = u[16 + w] ^ 0x80808080u;
        const uint32_t xd = u[24 + w] ^ 0x80808080u;
        const uint32_t lab = __builtin_amdgcn_perm(xb, xa, 0x05010400u);
        const uint32_t hab = __builtin_amdgcn_perm(xb, xa, 0x07030602u);
        const uint32_t lcd = __builtin_amdgcn_perm(xd, xc, 0x05010400u);
        const uint32_t hcd = __builtin_amdgcn_perm(xd, xc, 0x07030602u);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t sel = (c & 1) ? 0x04030402u : 0x04010400u;
            int32x2_t v;
            v[0] = (int32_t)fa2_pk_deq2(c < 2 ? lab : hab, sel, sab);
            v[1] = (int32_t)fa2_pk_deq2(c < 2 ? lcd : hcd, sel, scd);
            *(int32x2_t*)(Vdw + cl_vrow(q * 32 + w * 4 + c) + 2 * mm) = v;
        }
    }
}
#endif

// ---- Compute-wave helpers ----

// Pipeline stage fence: keeps prefetched loads and WMMAs of one chunk inside
// their stage (the machine scheduler otherwise regroups the two independent
// WMMA chains and keeps every operand alive -> spills).
#ifndef CL_SCHED_FENCE
#define CL_SCHED_FENCE 1
#endif
#if CL_SCHED_FENCE
#define CL_FENCE() do { asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0); } while (0)
#else
#define CL_FENCE() asm volatile("" ::: "memory")
#endif

// Lane-pair exchange (lane ^ 16).
__device__ __forceinline__ uint32_t cl_x16u(uint32_t v)
{
#if CL_XLANE
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
#else
    return (uint32_t)__shfl_xor((int)v, 16, 32);
#endif
}
__device__ __forceinline__ float cl_x16(float v)
{
    return __uint_as_float(cl_x16u(__float_as_uint(v)));
}

// Q dim chunk dc for this lane's query row (two aligned 16-byte loads).
__device__ __forceinline__ half16_t cl_q(const _Float16* __restrict__ q16,
                                         unsigned qq_base, int dc)
{
    const half8_t* qd = (const half8_t*)(q16 + qq_base + (unsigned)(dc * 16));
    const half8_t lo = qd[0];
    const half8_t hi = qd[1];
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                   12, 13, 14, 15);
}

// K fragment (WMMA A): key row at dword `kbase`, dim chunk dc; `rot` is the
// row's swizzle (identical for keys ml and 16+ml). Four aligned b64s.
__device__ __forceinline__ half16_t cl_kfrag(const uint32_t* Kdw, int kbase, int rot, int dc)
{
#if CL_B128
    (void)rot;
    typedef uint32_t __attribute__((ext_vector_type(4))) u4_t;
    const u4_t a = *(const u4_t*)(Kdw + kbase + dc * 8);
    const u4_t b = *(const u4_t*)(Kdw + kbase + dc * 8 + 4);
    return __builtin_bit_cast(half16_t, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
#endif
    uint8v_t r;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
#if CL_KPAD
        (void)rot;
        const int off = kbase + dc * 8 + i;
#else
        const int off = kbase + ((dc * 8 + i + rot) & 127);
#endif
        const int32x2_t pair = *(const int32x2_t*)(Kdw + off);
        r[i] = (uint32_t)pair[0];
        r[i + 1] = (uint32_t)pair[1];
    }
    return __builtin_bit_cast(half16_t, r);
}

#if CL_QLDS
// Q fragment for dim chunk c from the LDS stage (two b128; qo0/qo1 are the
// lane's swizzled half offsets).
__device__ __forceinline__ half16_t cl_qlds(const uint32_t* Qs, int qo0, int qo1, int c)
{
    typedef uint32_t __attribute__((ext_vector_type(4))) u4_t;
    const u4_t a = *(const u4_t*)(Qs + c * 128 + qo0);
    const u4_t b = *(const u4_t*)(Qs + c * 128 + qo1);
    return __builtin_bit_cast(half16_t, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
}
#endif

// V^T fragment (WMMA A): dim row dc*16+ml, keys ps..ps+15 of the plane; the
// per-lane offsets vl[i] = ml*16 + ((ps + 2i + rot) & 15) are dc-invariant,
// so each chunk is an immediate offset (dc*256 dwords).
__device__ __forceinline__ half16_t cl_vfrag(const uint32_t* Vdw, const int* vl, int dc)
{
#if CL_B128
    typedef uint32_t __attribute__((ext_vector_type(4))) u4_t;
    const int o = vl[0] + dc * (16 * CL_VROW) + 16 * (dc >> 1);
    const u4_t a = *(const u4_t*)(Vdw + o);
    const u4_t b = *(const u4_t*)(Vdw + o + 4);
    return __builtin_bit_cast(half16_t, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
#endif
    uint8v_t r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#if CL_FILL2
        const int32x2_t pair = *(const int32x2_t*)(Vdw + vl[i] + dc * 288 + 16 * (dc >> 1));
#else
        const int32x2_t pair = *(const int32x2_t*)(Vdw + vl[i] + dc * 256);
#endif
        r[2 * i] = (uint32_t)pair[0];
        r[2 * i + 1] = (uint32_t)pair[1];
    }
    return __builtin_bit_cast(half16_t, r);
}

// Build the replicated half16 P row (keys 0..15 of a subtile) from this lane's
// eight parity probabilities: RNE f16 per value (upstream cast), four packed
// dwords exchanged with lane^16, interleaved by v_perm.
__device__ __forceinline__ half16_t cl_prow(const float* e, uint32_t sel_lo, uint32_t sel_hi)
{
    uint8v_t pr;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        fa2_dw_t a;
        a.h[0] = (_Float16)e[2 * i];
        a.h[1] = (_Float16)e[2 * i + 1];
        const uint32_t b = cl_x16u(a.u);
        pr[2 * i] = __builtin_amdgcn_perm(b, a.u, sel_lo);
        pr[2 * i + 1] = __builtin_amdgcn_perm(b, a.u, sel_hi);
    }
    return __builtin_bit_cast(half16_t, pr);
}

// PV for one 16-key subtile: Ofr[dc] += V^T(dc) * P, V fragments prefetched one
// chunk ahead. `v0` is chunk 0, already issued by the caller.
__device__ __forceinline__ void cl_pv(const uint32_t* Vdw, const int* vl, half16_t v0,
                                      half16_t prow, float8_t* Ofr)
{
#if CL_PIPE
    half16_t vb[2];
    vb[0] = v0;
#pragma unroll
    for (int dc = 0; dc < 16; ++dc) {
        if (dc + 1 < 16)
            vb[(dc + 1) & 1] = cl_vfrag(Vdw, vl, dc + 1);
#if CL_KO_PV
        asm volatile("" :: "v"(vb[dc & 1]), "v"(prow));
#else
        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vb[dc & 1], prow, Ofr[dc]);
#endif
        CL_FENCE();
    }
#else
    Ofr[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v0, prow, Ofr[0]);
    CL_FENCE();
#pragma unroll
    for (int dc = 1; dc < 16; ++dc) {
        const half16_t vf = cl_vfrag(Vdw, vl, dc);
        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf, prow, Ofr[dc]);
        CL_FENCE();
    }
#endif
}

#if CL_Q32
// PV for the lower position half's last tile: keys past that half's causal
// end are zeroed in the V^T fragment registers (upstream's 16-row workgroup
// zero-fills them in its plane; here they hold the upper half's valid keys).
// P is already 0 for them, but the WMMA sum is not bit-invariant to finite
// operands under zero P, so the operand must match upstream exactly.
__device__ __forceinline__ void cl_pv_masked(const uint32_t* Vdw, const int* vl, half16_t v0,
                                             half16_t prow, float8_t* Ofr, uint8v_t vm)
{
    half16_t vb[2];
    vb[0] = __builtin_bit_cast(half16_t, __builtin_bit_cast(uint8v_t, v0) & vm);
#pragma unroll
    for (int dc = 0; dc < 16; ++dc) {
        if (dc + 1 < 16)
            vb[(dc + 1) & 1] = __builtin_bit_cast(
                half16_t, __builtin_bit_cast(uint8v_t, cl_vfrag(Vdw, vl, dc + 1)) & vm);
        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vb[dc & 1], prow, Ofr[dc]);
        CL_FENCE();
    }
}

// Dword mask for keys kbase+0..15 of a subtile: half h of dword j is kept iff
// key kbase+2j+h < lim.
__device__ __forceinline__ uint8v_t cl_keymask(int kbase, int lim)
{
    uint8v_t m;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        m[j] = (kbase + 2 * j < lim ? 0x0000ffffu : 0u) | (kbase + 2 * j + 1 < lim ? 0xffff0000u : 0u);
    return m;
}
#endif

// Shared workgroup body (direct only: writes normalized O to `out`).
__device__ __forceinline__ void cl_fa2_body(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base)
{
    constexpr int KT = 32;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + CL_KPLANE;

    // ---- Per-WG causal bounds (upstream mailbox) ----
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (tid < CL_QPOS) {
        const int qr = q_base + tid;
        if (qr < batch_size) {
            const int p = positions[qr];
            wmx = p;
            wmn = p;
        }
    }
#if CL_Q32
    // Per 16-position half first (lanes 0-15 / 16-31), then the workgroup.
    wmx = max(wmx, __shfl_xor(wmx, 8, 32));
    wmx = max(wmx, __shfl_xor(wmx, 4, 32));
    wmx = max(wmx, __shfl_xor(wmx, 2, 32));
    wmx = max(wmx, __shfl_xor(wmx, 1, 32));
    wmn = min(wmn, __shfl_xor(wmn, 8, 32));
    wmn = min(wmn, __shfl_xor(wmn, 4, 32));
    wmn = min(wmn, __shfl_xor(wmn, 2, 32));
    wmn = min(wmn, __shfl_xor(wmn, 1, 32));
    // Mailbox: pad dwords 16..19 of V^T rows 254 and 255 (never filled).
    uint32_t* mb = LDS + CL_KPLANE + cl_vrow(254) + 16;
    uint32_t* mb2 = LDS + CL_KPLANE + cl_vrow(255) + 16;
    if (tid == 0) {
        mb[0] = (uint32_t)wmx;   // lower half max
        mb[1] = (uint32_t)wmn;   // lower half min
    }
    if (tid == 16) {
        mb[2] = (uint32_t)wmx;   // upper half max
        mb[3] = (uint32_t)wmn;   // upper half min
    }
    wmx = max(wmx, __shfl_xor(wmx, 16, 32));
    wmn = min(wmn, __shfl_xor(wmn, 16, 32));
    if (tid == 0) {
        mb2[0] = (uint32_t)wmx;
        mb2[1] = (uint32_t)wmn;
    }
    __syncthreads();
    const int gmax = __builtin_amdgcn_readfirstlane((int)mb2[0]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)mb2[1]);
    const int hsel = (__builtin_amdgcn_readfirstlane(wave) / 6) & 1;
    const int my_gmax = __builtin_amdgcn_readfirstlane((int)mb[2 * hsel]);
    const int my_gmin = __builtin_amdgcn_readfirstlane((int)mb[2 * hsel + 1]);
#else
    wmx = max(wmx, __shfl_xor(wmx, 16, 32));
    wmx = max(wmx, __shfl_xor(wmx, 8, 32));
    wmx = max(wmx, __shfl_xor(wmx, 4, 32));
    wmx = max(wmx, __shfl_xor(wmx, 2, 32));
    wmx = max(wmx, __shfl_xor(wmx, 1, 32));
    wmn = min(wmn, __shfl_xor(wmn, 16, 32));
    wmn = min(wmn, __shfl_xor(wmn, 8, 32));
    wmn = min(wmn, __shfl_xor(wmn, 4, 32));
    wmn = min(wmn, __shfl_xor(wmn, 2, 32));
    wmn = min(wmn, __shfl_xor(wmn, 1, 32));
    if (tid == 0) {
        LDS[CL_MAILBOX] = (uint32_t)wmx;
        LDS[CL_MAILBOX + 1] = (uint32_t)wmn;
    }
    __syncthreads();
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[CL_MAILBOX]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[CL_MAILBOX + 1]);
    const int my_gmax = gmax;
    const int my_gmin = gmin;
#endif
    const int seq_len = gmax + 1;

    // Role split on the scalar wave index. Each role's state lives only in
    // its own branch (the epilogue is inside the compute branch), so the
    // helpers' raw-byte registers never overlap the 128-register O.
    const bool fill_compute = __builtin_amdgcn_readfirstlane(wave) < CL_NCW;
    if (fill_compute) {
        // Tile-0 K: one slot per thread (all eight waves).
        {
            uint32_t su;
            uint32_t uc[8];
            if (CL_THREADS == 256 || tid < 256) {
                fa2_fill_k_load(k_cache, 0, seq_len, kv_blk, tid, su, uc);
                fa2_fill_k_store(Kdw, tid, su, uc);
            }
        }
        const int h_ml = 6 * kv_h + (CL_Q32 ? wave % 6 : wave);
        const int qr_ml = q_base + (CL_Q32 ? 16 * (wave / 6) : 0) + ml;
        const bool qok_ml = qr_ml < batch_size;
        float8_t Ofr[16];
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float m_old = -INFINITY;
        float l_val = 0.0f;
        // Loads use a clamped valid row (rows >= batch_size are computed as a
        // duplicate and never stored), so no exec-masked loads.
        const int qr_c = min(qr_ml, batch_size - 1);
        const unsigned qq = (unsigned)qr_c * 6144u + (unsigned)h_ml * 256u;
        const int pos_ml = positions[qr_c];
        const int rot = fa2_swiz(ml);
        const int kb0 = ml * CL_KROW;
        const int kb1 = (16 + ml) * CL_KROW;
        int vl0[4], vl1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#if CL_FILL2
            vl0[i] = ml * CL_VROW + 2 * i;
            vl1[i] = ml * CL_VROW + 8 + 2 * i;
#else
            vl0[i] = ml * 16 + ((2 * i + rot) & 15);
            vl1[i] = ml * 16 + ((8 + 2 * i + rot) & 15);
#endif
        }
        const uint32_t sel_lo = half ? 0x01000504u : 0x05040100u;
        const uint32_t sel_hi = half ? 0x03020706u : 0x07060302u;
#if CL_SM32
        const float scale_l2 = scale_attn * 1.4426950408889634f;
#endif

        constexpr int QB = CL_PIPE ? CL_QD + 1 : 1;
        half16_t qbuf[QB];
#if CL_QLDS
        // Stage Q chunks [0, CL_QLDS) of this wave's 16 rows (lane writes the
        // half `half` of row ml); only this wave reads them back.
        uint32_t* Qs = LDS + CL_KPLANE + CL_VPLANE + wave * (CL_QLDS * 128);
        const int qsw = (ml >> 2) & 1;
        const int qo0 = ml * 8 + 4 * qsw;
        const int qo1 = ml * 8 + 4 * (qsw ^ 1);
        {
            typedef uint32_t __attribute__((ext_vector_type(4))) u4_t;
#pragma unroll
            for (int c = 0; c < CL_QLDS; ++c) {
                const u4_t v = *(const u4_t*)(q16 + qq + (unsigned)(c * 16 + half * 8));
                *(u4_t*)(Qs + c * 128 + ml * 8 + 4 * (half ^ qsw)) = v;
            }
        }
#endif
#if CL_PIPE
#pragma unroll
        for (int p = 0; p < CL_QD; ++p)
            if (p >= CL_QLDS)
                qbuf[p] = cl_q(q16, qq, p);
#endif
        __syncthreads();  // tile-0 K staged (pairs with the helpers' barrier)

        for (int tile = 0; ; ++tile) {
            const int ktile = tile * KT;
            if (ktile >= seq_len)
                break;
            // Per 16-position group tile range (upstream's 16-row workgroup
            // bounds): groups past their own causal end only join barriers.
            const bool active = ktile <= my_gmax;
            const bool do1 = ktile + 16 <= my_gmax;
#if CL_Q32
            if (!active) {
                CL_TILE_BARRIER();
                CL_TILE_BARRIER();
                continue;
            }
            // Lower group's last tile when the workgroup extends past it.
            const bool vmask_on = my_gmax < gmax && ktile + 32 > my_gmax + 1;
#else
            (void)active;
#endif

#if CL_KO_COMPUTE
            CL_TILE_BARRIER();
            CL_TILE_BARRIER();
            continue;
#endif
            // ---- Phase A: S^T = K * Q^T for both 16-key subtiles ----
            float8_t s0 = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t s1 = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#if CL_PIPE
            {
                half16_t ka[2], kb[2];
                ka[0] = cl_kfrag(Kdw, kb0, rot, 0);
                kb[0] = cl_kfrag(Kdw, kb1, rot, 0);
#if CL_QLDS
                half16_t qlb[2];
                qlb[0] = cl_qlds(Qs, qo0, qo1, 0);
#endif
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
#if !CL_KO_QLOAD
                    if (dc + CL_QD < 16 && dc + CL_QD >= CL_QLDS)
                        qbuf[(dc + CL_QD) % QB] = cl_q(q16, qq, dc + CL_QD);
#endif
#if CL_QLDS
                    if (dc + 1 < CL_QLDS)
                        qlb[(dc + 1) & 1] = cl_qlds(Qs, qo0, qo1, dc + 1);
#endif
                    if (dc + 1 < 16) {
                        ka[(dc + 1) & 1] = cl_kfrag(Kdw, kb0, rot, dc + 1);
                        kb[(dc + 1) & 1] = cl_kfrag(Kdw, kb1, rot, dc + 1);
                    }
#if CL_KO_QLOAD
                    const half16_t qf = qbuf[dc & 1];
#elif CL_QLDS
                    const half16_t qf = dc < CL_QLDS ? qlb[dc & 1] : qbuf[dc % QB];
#else
                    const half16_t qf = qbuf[dc % QB];
#endif
#if CL_KO_QK
                    asm volatile("" :: "v"(ka[dc & 1]), "v"(kb[dc & 1]), "v"(qf));
#else
                    s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ka[dc & 1], qf, s0);
                    s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kb[dc & 1], qf, s1);
#endif
                    CL_FENCE();
                }
            }
#else
            for (int dc = 0; dc < 16; ++dc) {
                const half16_t qf = cl_q(q16, qq, dc);
                s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    cl_kfrag(Kdw, kb0, rot, dc), qf, s0);
                s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    cl_kfrag(Kdw, kb1, rot, dc), qf, s1);
                CL_FENCE();
            }
#endif
            // Pin both score chains here: the WMMA builtins are pure, so LLVM
            // otherwise sinks the sub-1 chain past B1 into its (conditional)
            // consumer and keeps all sixteen Q chunks alive across the PV.
            asm volatile("" : "+v"(s0), "+v"(s1));
            CL_TILE_BARRIER();  // B1: V(t) in plane; K(t) consumed

            // ---- Phase B: online softmax + PV ----
#if CL_SM32
            {
                // One online-softmax step over all 32 keys, log2 domain.
                const half16_t v00 = cl_vfrag(Vdw, vl0, 0);
                const bool full = (ktile + 31) <= my_gmin;
                if (full) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        s0[j] *= scale_l2;
                        s1[j] *= scale_l2;
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int k0 = ktile + 2 * j + half;
                        s0[j] = (k0 <= pos_ml) ? s0[j] * scale_l2 : -INFINITY;
                        s1[j] = (k0 + 16 <= pos_ml) ? s1[j] * scale_l2 : -INFINITY;
                    }
                }
                float mx = fmaxf(s0[0], s1[0]);
#pragma unroll
                for (int j = 1; j < 8; ++j)
                    mx = fmaxf(mx, fmaxf(s0[j], s1[j]));
                mx = fmaxf(mx, cl_x16(mx));
                const float m_new = fmaxf(m_old, mx);
                const float alpha =
                    (m_old == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m_old - m_new);
#if CL_LAZY
                const bool resc = __builtin_amdgcn_ballot_w32(m_new != m_old) != 0u;
#endif
                m_old = m_new;
                const float m_use = (m_new == -INFINITY) ? 0.0f : m_new;
                float e0[8], e1[8];
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    e0[j] = __builtin_amdgcn_exp2f(s0[j] - m_use);
                    e1[j] = __builtin_amdgcn_exp2f(s1[j] - m_use);
                    sum += e0[j] + e1[j];
                }
                sum += cl_x16(sum);
                l_val = l_val * alpha + sum;
                const half16_t p0 = cl_prow(e0, sel_lo, sel_hi);
                const half16_t p1 = cl_prow(e1, sel_lo, sel_hi);
#if CL_LAZY
                if (resc)
#endif
                {
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        Ofr[dc] *= alpha;
                }
#if CL_Q32
                if (vmask_on) {
                    cl_pv_masked(Vdw, vl0, v00, p0, Ofr, cl_keymask(ktile, my_gmax + 1));
                    if (do1)
                        cl_pv_masked(Vdw, vl1, cl_vfrag(Vdw, vl1, 0), p1, Ofr,
                                     cl_keymask(ktile + 16, my_gmax + 1));
                } else
#endif
                {
                cl_pv(Vdw, vl0, v00, p0, Ofr);
                if (do1)
                    cl_pv(Vdw, vl1, cl_vfrag(Vdw, vl1, 0), p1, Ofr);
                }
            }
#else
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                if (sub == 1 && !do1)
                    break;
                const int* vl = sub ? vl1 : vl0;
                const half16_t v00 = cl_vfrag(Vdw, vl, 0);
                float8_t sacc = sub ? s1 : s0;
#if CL_KO_SM
                {
                    half16_t prow;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        prow[2 * j] = (_Float16)sacc[j];
                        prow[2 * j + 1] = (_Float16)sacc[j];
                    }
                    cl_pv(Vdw, vl, v00, prow, Ofr);
                    continue;
                }
#endif
                const int kt_sub = ktile + sub * 16;
                const bool full = (kt_sub + 15) <= my_gmin;
                if (full) {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        sacc[j] = sacc[j] * scale_attn;
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key_g = kt_sub + 2 * j + half;
                        sacc[j] = (key_g <= pos_ml) ? sacc[j] * scale_attn : -INFINITY;
                    }
                }
                float mx = -INFINITY;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    mx = fmaxf(mx, sacc[j]);
                mx = fmaxf(mx, cl_x16(mx));
                const float m_new = fmaxf(m_old, mx);
                const float alpha =
                    (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
#if CL_LAZY
                const bool resc = __builtin_amdgcn_ballot_w32(m_new != m_old) != 0u;
#endif
                m_old = m_new;
                float earr[8];
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = (m_new == -INFINITY) ? 0.0f : __expf(sacc[j] - m_new);
                    earr[j] = e;
                    sum += e;
                }
                sum += cl_x16(sum);
                l_val = l_val * alpha + sum;
                const half16_t prow = cl_prow(earr, sel_lo, sel_hi);
#if CL_LAZY
                if (resc)
#endif
                {
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        Ofr[dc] *= alpha;
                }
#if CL_Q32
                if (vmask_on)
                    cl_pv_masked(Vdw, vl, v00, prow, Ofr, cl_keymask(kt_sub, my_gmax + 1));
                else
#endif
                cl_pv(Vdw, vl, v00, prow, Ofr);
            }
#endif
#if CL_PIPE
            // Next tile's first Q chunks (Q is tile-invariant): in flight
            // across B2.
#pragma unroll
            for (int p = 0; p < CL_QD; ++p)
                if (p >= CL_QLDS)
                    qbuf[p] = cl_q(q16, qq, p);
#endif
            CL_TILE_BARRIER();  // B2: K(t+1) in plane; V(t) consumed
        }

        // ---- Epilogue (upstream) ----
        if (qok_ml) {
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            const unsigned bo =
                (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
            for (int dc = 0; dc < 16; ++dc) {
                const unsigned bd = bo + (unsigned)(dc * 16 + half);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    out[bd + (unsigned)(j * 2)] = Ofr[dc][j] * inv;
            }
        }
    } else {
#if CL_HPRIO
        __builtin_amdgcn_s_setprio(CL_HPRIO);
#endif
#if CL_FILL2
        const int h = tid - CL_NCW * 32;
        const uint32_t kv_boff = (uint32_t)kv_blk * 34u;
        uint32_t vs[4], vu[32], ks[4], ku[32];
        cl_v2_load(v_cache, 0, seq_len, h, kv_boff, vs, vu);
        {
            uint32_t su;
            uint32_t uc[8];
            if (CL_THREADS == 256 || tid < 256) {
                fa2_fill_k_load(k_cache, 0, seq_len, kv_blk, tid, su, uc);
                fa2_fill_k_store(Kdw, tid, su, uc);
            }
        }
        __syncthreads();  // tile-0 K staged
        for (int tile = 0; ; ++tile) {
            const int ktile = tile * KT;
            if (ktile >= seq_len)
                break;
            const bool next = ktile + KT < seq_len;
            if (next && CL_KO_FILL != 1)
                cl_k2_load(k_cache, ktile + KT, seq_len, h, kv_boff, ks, ku);
#if CL_KO_FILL != 1
            cl_v2_store(Vdw, h, vs, vu);
#endif
            CL_TILE_BARRIER();  // B1
            if (next && CL_KO_FILL != 1) {
                cl_v2_load(v_cache, ktile + KT, seq_len, h, kv_boff, vs, vu);
                cl_k2_store(Kdw, h, ks, ku);
            }
            CL_TILE_BARRIER();  // B2
        }
#else
        const int hl = tid - 6 * 32;
        uint32_t rv_s[FA2_FILL_HV];
        uint32_t rv_u[FA2_FILL_HV * 4];
#pragma unroll
        for (int s = 0; s < FA2_FILL_HV; ++s)
            fa2_fill_v_load(v_cache, 0, seq_len, kv_blk, hl + s * FA2_FILL_HT,
                            rv_s[s], rv_u + s * 4);
        {
            uint32_t su;
            uint32_t uc[8];
            if (CL_THREADS == 256 || tid < 256) {
                fa2_fill_k_load(k_cache, 0, seq_len, kv_blk, tid, su, uc);
                fa2_fill_k_store(Kdw, tid, su, uc);
            }
        }
        __syncthreads();  // tile-0 K staged
        uint32_t rk_s[FA2_FILL_HK];
        uint32_t rk_u[FA2_FILL_HK * 8];
#if CL_KO_FILL == 3
#pragma unroll
        for (int t = 0; t < FA2_FILL_HK; ++t) {
            rk_s[t] = 0x2000u + (uint32_t)t;
#pragma unroll
            for (int w = 0; w < 8; ++w)
                rk_u[t * 8 + w] = 0x01020304u * (uint32_t)(w + 1) + (uint32_t)hl;
        }
#endif
        for (int tile = 0; ; ++tile) {
            const int ktile = tile * KT;
            if (ktile >= seq_len)
                break;
            const bool next = ktile + KT < seq_len;
            if (next && (CL_KO_FILL == 0 || CL_KO_FILL == 2)) {
#pragma unroll
                for (int t = 0; t < FA2_FILL_HK; ++t)
                    fa2_fill_k_load(k_cache, ktile + KT, seq_len, kv_blk,
                                    hl + t * FA2_FILL_HT, rk_s[t], rk_u + t * 8);
            }
#if CL_KO_FILL == 0 || CL_KO_FILL == 3
#pragma unroll
            for (int s = 0; s < FA2_FILL_HV; ++s)
                fa2_fill_v_store(Vdw, hl + s * FA2_FILL_HT, rv_s[s], rv_u + s * 4);
#elif CL_KO_FILL == 2
#pragma unroll
            for (int s = 0; s < FA2_FILL_HV; ++s)
                asm volatile("" :: "v"(rv_s[s]), "v"(rv_u[s * 4]), "v"(rv_u[s * 4 + 1]), "v"(rv_u[s * 4 + 2]), "v"(rv_u[s * 4 + 3]));
#endif
            CL_TILE_BARRIER();  // B1
            if (next && CL_KO_FILL != 1) {
#if CL_KO_FILL != 3
#pragma unroll
                for (int s = 0; s < FA2_FILL_HV; ++s)
                    fa2_fill_v_load(v_cache, ktile + KT, seq_len, kv_blk,
                                    hl + s * FA2_FILL_HT, rv_s[s], rv_u + s * 4);
#endif
#if CL_KO_FILL == 2
#pragma unroll
                for (int t = 0; t < FA2_FILL_HK; ++t) {
                    asm volatile("" :: "v"(rk_s[t]), "v"(rk_u[t * 8]), "v"(rk_u[t * 8 + 1]), "v"(rk_u[t * 8 + 2]), "v"(rk_u[t * 8 + 3]));
                    asm volatile("" :: "v"(rk_u[t * 8 + 4]), "v"(rk_u[t * 8 + 5]), "v"(rk_u[t * 8 + 6]), "v"(rk_u[t * 8 + 7]));
                }
#else
#pragma unroll
                for (int t = 0; t < FA2_FILL_HK; ++t)
                    fa2_fill_k_store(Kdw, hl + t * FA2_FILL_HT, rk_s[t], rk_u + t * 8);
#endif
            }
            CL_TILE_BARRIER();  // B2
        }
#endif
    }

}

// Same kernarg ABI as attention_q8_0_fa2_gqa_gfx11 (q16 pre-converted by the
// upstream attention_fa2_q_preconvert_gfx11). Grid [ceil(batch/16), 4, 1],
// block 256, CL_LDS_DWORDS*4 B dynamic LDS (37,824 with CL_B128, 35,520 with
// CL_FILL2, 33,024 with CL_KPAD only; the launcher passes cl_lds_bytes).
extern "C" __global__ __launch_bounds__(CL_THREADS, CL_WAVES_PER_EU) void CLAUDE_FA2_ENTRY(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn)
{
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * CL_QPOS;
    if (q_base >= batch_size)
        return;
    cl_fa2_body(q16, k_cache, v_cache, out, positions, batch_size, scale_attn, kv_h,
                q_base);
}
