// claude v4: claude_attn5 -- WMMA split-K speculative-verify attention for
// q8_0 KV caches, Qwen3.8-27B shape only (24 q heads / 4 kv heads / head_dim
// 256), 1..8 verify rows. Successor of claude_attn4.hip (same numerics class:
// Q/K/V/P in f16, f32 accumulation), restructured for bandwidth:
//
//  * One block = (kv head, span of keys), 4*QT waves, QT = ceil(6*rows/16)
//    16-query WMMA tiles (query qi = h_local*rows + r).
//  * Wave w = (qt = w>>2, ks = (w>>1)&1, d = w&1): query tile qt, key half ks
//    of every 32-key tile (its own online-softmax stream), dim half d.
//    QK: S^T = K * Q^T over the wave's 128 dims (Q fragments stay in VGPRs);
//    the two dim halves exchange partial S once per tile through LDS.
//    PV: O^T = V^T * P^T for the wave's 128 dims. No per-tile redundant
//    reductions beyond the d-pair softmax.
//  * K plane (f16 [key][256], rows padded to 528 B) and V^T plane (f16
//    [dim][32 keys], 16-byte granules XOR-swizzled) are single-buffered but
//    filled in alternating phases: phase A fills V(t) while QK(t) reads K(t);
//    phase B fills K(t+1) while PV(t) reads V(t). Two barriers per tile.
//  * Raw q8 bytes are prefetched two tiles ahead into registers.
//  * Lazy rescale: the running max only moves when a tile exceeds it by
//    A5_THRESH (log2 units), so P <= 2^A5_THRESH (exact algebra, f16-safe).
//  * The two key-half streams are merged in LDS at the end; one partial per
//    (row, head, span): [m (log2 units), l, pad, pad, O[256] in WMMA C order]
//    reduced by claude_attn5_reduce.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define A5_KT 32
#ifndef A5_FD
#define A5_FD 2            // LDS fragment prefetch distance (chunks)
#endif
#define A5_KROW 132
#define A5_PSTRIDE 260
#ifndef A5_THRESH
#define A5_THRESH 8.0f
#endif

typedef _Float16 a5_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 a5_h2 __attribute__((ext_vector_type(2)));
typedef float a5_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t a5_u4 __attribute__((ext_vector_type(4)));
typedef uint32_t a5_u4u __attribute__((ext_vector_type(4), aligned(4)));
typedef uint32_t a5_u8 __attribute__((ext_vector_type(8)));
typedef uint32_t a5_u4h8 __attribute__((ext_vector_type(4), aligned(8)));
typedef union { uint32_t u; a5_h2 h2; } a5_pk;
// compile-time bool usable where a runtime bool is also accepted
struct a5_true { constexpr operator bool() const { return true; } };

#ifndef A5_NO_SB
#define A5_SB() asm volatile("" ::: "memory")
#else
#define A5_SB()
#endif
// Workgroup barrier that only orders LDS: __syncthreads() in WGP mode waits
// for every outstanding global load (vmcnt(0)) and invalidates L0, which
// would serialize the register prefetch of the next tiles.
#if defined(A5_NO_BAR)
#define A5_BAR() asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory")
#elif !defined(A5_SYNC_FULL)
// sched_barrier on both sides: otherwise the scheduler hoists uses of the
// just-issued prefetch (e.g. the next tile's scale extraction) above the
// barrier, which forces s_waitcnt vmcnt(0) on it.
#define A5_BAR() do { __builtin_amdgcn_sched_barrier(0); \
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); \
    __builtin_amdgcn_sched_barrier(0); } while (0)
#else
#define A5_BAR() __syncthreads()
#endif
#ifdef A5_PROF
__device__ uint32_t a5_prof[4096 * 16];
static __device__ __forceinline__ uint32_t a5_clk() {
    uint32_t x;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES, 0, 20)" : "=s"(x) :: "memory");
    return x;
}
#define A5_PT(i) do { const uint32_t n_ = a5_clk(); pacc[i] += (n_ - plast) & 0xfffffu; plast = n_; } while (0)
#define A5_VMW() asm volatile("s_waitcnt vmcnt(3)" ::: "memory")
#else
#define A5_PT(i) do {} while (0)
#define A5_VMW() do {} while (0)
#endif
static __device__ __forceinline__ a5_u8 a5_kfrag(const uint32_t* p) {
    const a5_u4 lo = *(const a5_u4*)(p);
    const a5_u4 hi = *(const a5_u4*)(p + 4);
    return (a5_u8){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
}
static __device__ __forceinline__ a5_u8 a5_vfrag(const uint32_t* p0, const uint32_t* p1) {
    const a5_u4 lo = *(const a5_u4*)(p0);
    const a5_u4 hi = *(const a5_u4*)(p1);
    return (a5_u8){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
}
static __device__ __forceinline__ uint32_t a5_x16(uint32_t v) {
    return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210, 0xfedcba98, false, false);
}
static __device__ __forceinline__ float a5_x16f(float v) {
    return __builtin_bit_cast(float, a5_x16(__builtin_bit_cast(uint32_t, v)));
}

// Two codes (bytes of tb picked by sel, interleaved with 0x64) times the f16
// scale pair s2: RNE(s*c) per half (same arithmetic as claude_attn4 / FA2).
static __device__ __forceinline__ uint32_t a5_deq2(uint32_t tb, uint32_t sel, uint32_t s2) {
    a5_pk a, s;
    a.u = __builtin_amdgcn_perm(0x64646464u, tb, sel);
    s.u = s2;
    const a5_h2 bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    a.h2 = (a.h2 - bias) * s.h2;
    return a.u;
}

// q8_0 block b (34 B) of a 272-byte head row: codes via two b128 loads at
// 34b+2 (2-byte aligned for even b; gfx11 global loads accept this, and it
// streams at full rate) + the f16 scale as a u16 load. w[8] = scale.
typedef uint32_t a5_u4h __attribute__((ext_vector_type(4), aligned(2)));
static __device__ __forceinline__ void a5_load(const unsigned char* __restrict__ row, int b, uint32_t w[9]) {
    const unsigned char* p = row + 34 * b;
    const a5_u4h x0 = *(const a5_u4h*)(p + 2);
    const a5_u4h x1 = *(const a5_u4h*)(p + 18);
    // full dword (scale + first two codes): a 16-bit load would be a d16
    // load into half a VGPR, and the allocator then reuses the other half
    // as an f16 temporary, forcing a wait on the prefetch mid-softmax.
    typedef uint32_t u32a2 __attribute__((aligned(2)));
    w[8] = *(const u32a2*)p;
    w[0] = x0.x; w[1] = x0.y; w[2] = x0.z; w[3] = x0.w;
    w[4] = x1.x; w[5] = x1.y; w[6] = x1.z; w[7] = x1.w;
}
static __device__ __forceinline__ void a5_zero(uint32_t w[9]) {
    #pragma unroll
    for (int i = 0; i < 9; i++) w[i] = 0u;
}
// scale pair + 8 code dwords (code ^ 0x80 applied) of the loaded block.
static __device__ __forceinline__ void a5_codes(const uint32_t w[9], int b, uint32_t& s2, uint32_t c[8]) {
    (void)b;
    // Pin the prefetched registers here: without this the compiler hoists the
    // scale/code arithmetic into the previous phase, right behind the loads,
    // which forces an early s_waitcnt vmcnt on the prefetch.
    uint32_t x[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        x[i] = w[i];
        asm volatile("" : "+v"(x[i]));
    }
    s2 = __builtin_amdgcn_perm(x[8], x[8], 0x01000100u);
    #pragma unroll
    for (int j = 0; j < 8; j++) c[j] = x[j] ^ 0x80808080u;
}

// K unit (key = lane, block b) -> K plane row granules 4b..4b+3.
static __device__ __forceinline__ void a5_fill_k(uint32_t* __restrict__ kdst, int b, const uint32_t w[9]) {
    uint32_t s2, c[8];
    a5_codes(w, b, s2, c);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a5_u4 v;
        v.x = a5_deq2(c[2 * i], 0x04010400u, s2);
        v.y = a5_deq2(c[2 * i], 0x04030402u, s2);
        v.z = a5_deq2(c[2 * i + 1], 0x04010400u, s2);
        v.w = a5_deq2(c[2 * i + 1], 0x04030402u, s2);
        *(a5_u4*)(kdst + 4 * i) = v;
    }
}

// V unit -> V^T plane. vb[jj] = per-lane dword offsets (row 32b + odd, its
// 2-dword pad, swizzled key-pair column) for (j & 3) == jj; dims 32b + 4j +
// {0,2} + odd go to rows at +64j and +64j+32 dwords.
static __device__ __forceinline__ void a5_fill_v(uint32_t* __restrict__ vdst, const int vb[4], uint32_t selv,
                                                 int b, const uint32_t w[9]) {
    uint32_t s2, c[8];
    a5_codes(w, b, s2, c);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint32_t p01 = a5_deq2(c[j], 0x04010400u, s2);
        const uint32_t p23 = a5_deq2(c[j], 0x04030402u, s2);
        const uint32_t q01 = (uint32_t)__builtin_amdgcn_mov_dpp((int)p01, 0xB1, 0xf, 0xf, true);
        const uint32_t q23 = (uint32_t)__builtin_amdgcn_mov_dpp((int)p23, 0xB1, 0xf, 0xf, true);
        vdst[vb[j & 3] + 64 * j] = __builtin_amdgcn_perm(q01, p01, selv);
        vdst[vb[j & 3] + 64 * j + 32] = __builtin_amdgcn_perm(q23, p23, selv);
    }
}

template <int QT>
struct A5Cfg {
    static constexpr int NW = 4 * QT;
    static constexpr int NT = 32 * NW;
    static constexpr int UPT = (QT == 1) ? 2 : 1;      // fill units per thread per tensor
    static constexpr int K_DW = A5_KT * A5_KROW;        // 4224
    static constexpr int V_DW = 256 * 16 + 48;          // 4144 (4*sigma(b) pad per 32 rows, sigma <= 11)
    static constexpr int S_DW = NW * 256;
    static constexpr int MERGE_DW = QT * (2048 + 64);   // one d-round of ks=1 streams
    static constexpr int MAIN_DW = K_DW + V_DW + S_DW;
    static constexpr int LDS_DW = MAIN_DW > MERGE_DW ? MAIN_DW : MERGE_DW;
};

template <int QT>
static __device__ __forceinline__ void a5_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads, float scale_log2, int n_blk, int max_spans, int batch_offset, int rows,
    uint32_t* __restrict__ lds)
{
    using C = A5Cfg<QT>;
    uint32_t* Kl = lds;
    uint32_t* Vl = lds + C::K_DW;
    float* Sx = (float*)(lds + C::K_DW + C::V_DW);

    const int tid = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const int kv_h = blockIdx.x;
    const int span_idx = blockIdx.y;   // this block owns 32-key tiles span_idx + n_blk * t

    int seq_max = 0, seq_min = 0x7fffffff;
    for (int r = 0; r < rows; r++) {
        const int s = positions[batch_offset + r] + 1;
        seq_max = max(seq_max, s);
        seq_min = min(seq_min, s);
    }
    // uniform by construction; tell the compiler so loop control stays scalar
    seq_max = __builtin_amdgcn_readfirstlane(seq_max);
    seq_min = __builtin_amdgcn_readfirstlane(seq_min);
    const int total_tiles = (seq_max + A5_KT - 1) / A5_KT;
    if (span_idx >= total_tiles) return;
    const int n_tiles = (total_tiles - span_idx + n_blk - 1) / n_blk;
    const int span_end = seq_max;
    const int gmin = seq_min - 1;

    // ---- compute roles ----
    const int d = wave & 1;
    const int ks = (wave >> 1) & 1;
    const int qt = wave >> 2;
    const int qi = qt * 16 + ml;
    const bool qok = qi < 6 * rows;
    // wave-uniform: this query tile has at least one valid query (a QT=2
    // launch serves 1..=2 rows with the qt=1 waves only filling planes)
    const bool active = qt * 16 < 6 * rows;
    const int hl = qok ? qi / rows : 0;
    const int r = qok ? qi - hl * rows : 0;
    const int head = kv_h * 6 + hl;
    const int pos_q = qok ? positions[batch_offset + r] : -1;
    const int q_dim = n_heads * 256;

    a5_h16 qf[8];

    a5_f8 O[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) O[i] = (a5_f8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // per-lane LDS offsets (dwords)
    const int k_rd = (ks * 16 + ml) * A5_KROW + d * 64;                  // + 8*c
    const int y = (ml >> 2) & 3;
    // V^T row r at 16 r + 4 sigma(r >> 5) dwords, sigma(b) = (b & 3) + 8 (b >> 2)
    // (16-byte aligned rows; conflict-free fill writes and fragment reads),
    // 16-byte granules XOR (r >> 2) & 3. Rows of this wave: b = 4d + (c >> 1).
    const int v_rd0 = (d * 128 + ml) * 16 + 32 * d + (((2 * ks) ^ y) << 2);      // + 256 c + 4 (c >> 1)
    const int v_rd1 = (d * 128 + ml) * 16 + 32 * d + (((2 * ks + 1) ^ y) << 2);
    float* sx_own = Sx + wave * 256;
    float* sx_par = Sx + (wave ^ 1) * 256;
    const uint32_t sel_e0 = half ? 0x01000504u : 0x05040100u;
    const uint32_t sel_e1 = half ? 0x03020706u : 0x07060302u;

    // ---- fill roles: lane -> (block fb = (lane>>1)&7, key fk); adjacent lanes
    // hold keys 2m, 2m+1 of the same block. QT==1: 4 waves cover 32 keys with
    // two units each (keys fk and fk+16).
    const bool filler = (QT <= 2) || wave < 8;   // compile-time true for QT <= 2
    const int odd = lane & 1;
    const int fb = (lane >> 1) & 7;
    const int fk = (C::UPT == 2) ? (wave * 4 + 2 * (lane >> 4) + odd)
                                 : ((wave & 7) * 4 + 2 * (lane >> 4) + odd);
    const uint32_t selv = odd ? 0x03020706u : 0x05040100u;
    int vb[C::UPT][4];
    #pragma unroll
    for (int u = 0; u < C::UPT; u++) {
        const int m2 = (fk + 16 * u) >> 1;
        #pragma unroll
        for (int jj = 0; jj < 4; jj++)
            vb[u][jj] = (32 * fb + odd) * 16 + 4 * ((fb & 3) + 8 * (fb >> 2)) + (((m2 >> 2) ^ jj) << 2) + (m2 & 3);
    }
    const size_t head_off = (size_t)kv_h * 272;

    uint32_t sink = 0;
    auto load_tile = [&](const unsigned char* __restrict__ cache, int t, uint32_t (&dst)[C::UPT][9]) __attribute__((always_inline)) {
        const int key0 = (span_idx + n_blk * t) * A5_KT + fk;
#ifdef A5_NO_LOAD
        #pragma unroll
        for (int u = 0; u < C::UPT; u++)
            #pragma unroll
            for (int i = 0; i < 9; i++) dst[u][i] = (i == 8) ? 0x2000u : (uint32_t)(key0 * 9 + i) * 0x01010101u;
        return;
#endif
        // Branch-free: keys past the end are clamped to the last valid row
        // (their scores are masked, and P = 0 times a finite V row adds 0),
        // so the waitcnt pass can track the prefetch precisely.
        #pragma unroll
        for (int u = 0; u < C::UPT; u++) {
            const int key = min(key0 + 16 * u, span_end - 1);
            a5_load(cache + (size_t)key * 1088 + head_off, fb, dst[u]);
        }
    };
    auto fill_k = [&](const uint32_t (&src)[C::UPT][9]) __attribute__((always_inline)) {
#ifdef A5_NO_FILL
        #pragma unroll
        for (int u = 0; u < C::UPT; u++)
            #pragma unroll
            for (int i = 0; i < 9; i++) sink ^= src[u][i];
        return;
#endif
        #pragma unroll
        for (int u = 0; u < C::UPT; u++) a5_fill_k(Kl + (fk + 16 * u) * A5_KROW + 16 * fb, fb, src[u]);
    };
    auto fill_v = [&](const uint32_t (&src)[C::UPT][9]) __attribute__((always_inline)) {
#ifdef A5_NO_FILL
        #pragma unroll
        for (int u = 0; u < C::UPT; u++)
            #pragma unroll
            for (int i = 0; i < 9; i++) sink += src[u][i];
        return;
#endif
        #pragma unroll
        for (int u = 0; u < C::UPT; u++) a5_fill_v(Vl, vb[u], selv, fb, src[u]);
    };

#ifndef A5_DEPTH
#define A5_DEPTH 1
#endif
    constexpr int D = A5_DEPTH;
    // raw prefetch ring: V(t) lives in rv[t % D], K(t) in rk[t % D]
    uint32_t rv[D][C::UPT][9], rk[D][C::UPT][9];
    // ---- prologue: K(0) into the plane; V(0..D-1), K(1..D) in flight ----
    if (filler) {
        load_tile(k_cache, 0, rk[0]);
        #pragma unroll
        for (int i = 0; i < D; i++) if (i < n_tiles) load_tile(v_cache, i, rv[i]);
    }
    // Q fragments (f16) for this wave's 8 dim chunks; loaded after the first
    // K/V tile loads are in flight so the two latencies overlap.
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a5_h16 f;
        if (qok) {
            const float* qp = q + (size_t)r * q_dim + head * 256 + (d * 8 + i) * 16;
            const float4 a = *(const float4*)(qp);
            const float4 b = *(const float4*)(qp + 4);
            const float4 c = *(const float4*)(qp + 8);
            const float4 e = *(const float4*)(qp + 12);
            f[0] = (_Float16)a.x; f[1] = (_Float16)a.y; f[2] = (_Float16)a.z; f[3] = (_Float16)a.w;
            f[4] = (_Float16)b.x; f[5] = (_Float16)b.y; f[6] = (_Float16)b.z; f[7] = (_Float16)b.w;
            f[8] = (_Float16)c.x; f[9] = (_Float16)c.y; f[10] = (_Float16)c.z; f[11] = (_Float16)c.w;
            f[12] = (_Float16)e.x; f[13] = (_Float16)e.y; f[14] = (_Float16)e.z; f[15] = (_Float16)e.w;
        } else {
            #pragma unroll
            for (int e = 0; e < 16; e++) f[e] = (_Float16)0.0f;
        }
        qf[i] = f;
    }

    if (filler) {
        fill_k(rk[0]);
        #pragma unroll
        for (int i = 1; i <= D; i++) if (i < n_tiles) load_tile(k_cache, i, rk[i % D]);
    }
    A5_BAR();

    // lv/lk: issue the V(t+D) / K(t+1+D) prefetch; kf: fill K(t+1).
    // Compile-time true in the steady state (std::true_type), so the waitcnt
    // pass sees straight-line loads; runtime bools in the tail.
#ifdef A5_PROF
    uint32_t pacc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t plast = a5_clk();
    uint32_t pn = 0;
#endif
    auto tile = [&](int t, uint32_t (&rv_s)[C::UPT][9], uint32_t (&rk_s)[C::UPT][9],
                    auto lv, auto lk, auto kf) __attribute__((always_inline)) {
        const bool pv = lv, pk = lk, fkk = kf;
        const int kt = (span_idx + n_blk * t) * A5_KT;
        // ---- phase A: fill V(t); QK(t) ----
        A5_PT(9);
        if (pv && pk) A5_VMW();
        A5_PT(0);
        if (filler) {
            fill_v(rv_s);
            if (pv) load_tile(v_cache, t + D, rv_s);
        }
        A5_PT(1);
        a5_f8 sacc = (a5_f8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (active) {
        A5_SB();
        {
            // two fragment slots, each consumed before it is refilled, so the
            // allocator keeps both live and each WMMA waits only for its own
            // fragment (lgkmcnt(2)), not for the one behind it.
            a5_u8 ka = a5_kfrag(Kl + k_rd);
            a5_u8 kb = a5_kfrag(Kl + k_rd + 8);
            sacc = (a5_f8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int c = 0; c < 8; c += 2) {
#ifdef A5_NO_WMMA
                sacc[c] += __builtin_bit_cast(float, ka[c]) + __builtin_bit_cast(float, kb[c]);
#else
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(a5_h16, ka), qf[c], sacc);
#endif
                if (c + 2 < 8) ka = a5_kfrag(Kl + k_rd + 8 * (c + 2));
#ifndef A5_NO_WMMA
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(a5_h16, kb), qf[c + 1], sacc);
#endif
                if (c + 3 < 8) kb = a5_kfrag(Kl + k_rd + 8 * (c + 3));
            }
#if !defined(A5_NO_WMMA) && !defined(A5_NO_IGLP)
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 1);
            #pragma unroll
            for (int c = 0; c < 6; c++) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
#endif
        }
        *(float4*)(sx_own + lane * 4) = (float4){sacc[0], sacc[1], sacc[2], sacc[3]};
        *(float4*)(sx_own + 128 + lane * 4) = (float4){sacc[4], sacc[5], sacc[6], sacc[7]};
        }
        A5_PT(2);
        A5_BAR();
        A5_PT(3);
        if (pv && pk) A5_VMW();
        A5_PT(4);
        // ---- phase B: fill K(t+1); softmax; PV(t) ----
        if (filler && fkk) {
            fill_k(rk_s);
            if (pk) load_tile(k_cache, t + 1 + D, rk_s);
        }
        A5_PT(5);
        if (active) {
        {
            const float4 a = *(const float4*)(sx_par + lane * 4);
            const float4 b = *(const float4*)(sx_par + 128 + lane * 4);
            sacc[0] += a.x; sacc[1] += a.y; sacc[2] += a.z; sacc[3] += a.w;
            sacc[4] += b.x; sacc[5] += b.y; sacc[6] += b.z; sacc[7] += b.w;
        }
        const int kbase = kt + ks * 16;
        if (kbase + 15 > gmin) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (kbase + 2 * j + half > pos_q) sacc[j] = -INFINITY;
            }
        }
        float mx = sacc[0];
        #pragma unroll
        for (int j = 1; j < 8; j++) mx = fmaxf(mx, sacc[j]);
        mx = fmaxf(mx, a5_x16f(mx)) * scale_log2;
        if (__builtin_amdgcn_ballot_w32(mx > m_run + A5_THRESH)) {
            const float mn = fmaxf(m_run, mx);
            const float alpha = (mn == -INFINITY) ? 1.0f : __builtin_amdgcn_exp2f(m_run - mn);
            m_run = mn;
            l_run *= alpha;
            #pragma unroll
            for (int c = 0; c < 8; c++) O[c] *= alpha;
        }
        const float mu = (m_run == -INFINITY) ? 0.0f : m_run;
        float p[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            p[j] = __builtin_amdgcn_exp2f(__builtin_fmaf(sacc[j], scale_log2, -mu));
            l_run += p[j];
        }
        uint32_t dw[4], xw[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            a5_pk pk;
            pk.h2 = (a5_h2){(_Float16)p[2 * i], (_Float16)p[2 * i + 1]};
            dw[i] = pk.u;
            xw[i] = a5_x16(dw[i]);
        }
        a5_u8 pf;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            pf[2 * i] = __builtin_amdgcn_perm(xw[i], dw[i], sel_e0);
            pf[2 * i + 1] = __builtin_amdgcn_perm(xw[i], dw[i], sel_e1);
        }
        const a5_h16 pfrag = __builtin_bit_cast(a5_h16, pf);
        A5_PT(6);
        A5_SB();
        {
            a5_u8 va = a5_vfrag(Vl + v_rd0, Vl + v_rd1);
            a5_u8 vb2 = a5_vfrag(Vl + v_rd0 + 256, Vl + v_rd1 + 256);
            #pragma unroll
            for (int c = 0; c < 8; c += 2) {
#ifdef A5_NO_WMMA
                O[c][c & 7] += __builtin_bit_cast(float, va[c & 7]) + (float)pfrag[c];
                O[c + 1][c & 7] += __builtin_bit_cast(float, vb2[c & 7]);
#else
                O[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(a5_h16, va), pfrag, O[c]);
#endif
                if (c + 2 < 8) {
                    const int o = 256 * (c + 2) + 4 * ((c + 2) >> 1);
                    va = a5_vfrag(Vl + v_rd0 + o, Vl + v_rd1 + o);
                }
#ifndef A5_NO_WMMA
                O[c + 1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(a5_h16, vb2), pfrag, O[c + 1]);
#endif
                if (c + 3 < 8) {
                    const int o = 256 * (c + 3) + 4 * ((c + 3) >> 1);
                    vb2 = a5_vfrag(Vl + v_rd0 + o, Vl + v_rd1 + o);
                }
            }
#if !defined(A5_NO_WMMA) && !defined(A5_NO_IGLP)
            // schedule: frag 0,1 reads, then WMMA c / reads of frag c+2
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
            #pragma unroll
            for (int c = 0; c < 6; c++) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
#endif
        }
        }
        A5_PT(7);
        A5_BAR();
        A5_PT(8);
#ifdef A5_PROF
        pn++;
#endif
    };

    {
        using T_ = a5_true;
        int t = 0;
        // steady state: groups of D tiles whose prefetches are all in range
        #pragma unroll 1
        for (; t + 2 * D < n_tiles; t += D) {
            #pragma unroll
            for (int i = 0; i < D; i++) tile(t + i, rv[i], rk[(i + 1) % D], T_{}, T_{}, T_{});
        }
        // tail: at most 2D tiles, runtime-guarded prefetches
        #pragma unroll
        for (int i = 0; i < 2 * D; i++) {
            const int tt = t + i;
            if (tt < n_tiles)
                tile(tt, rv[i % D], rk[(i + 1) % D], tt + D < n_tiles, tt + 1 + D < n_tiles, tt + 1 < n_tiles);
        }
    }

#ifdef A5_PROF
    if (lane == 0) {
        uint32_t* pp = a5_prof + ((size_t)(blockIdx.y * gridDim.x + blockIdx.x) * C::NW + wave) * 16;
        #pragma unroll
        for (int i = 0; i < 10; i++) pp[i] = pacc[i];
        pp[10] = pn;
    }
#endif
    // ---- merge the two key-half streams, write partials ----
    l_run += a5_x16f(l_run);
#ifdef A5_NO_FILL
    if (sink == 0x12345678u) l_run += 1.0f;
#endif
    float* mg = (float*)lds;
    #pragma unroll 1
    for (int round = 0; round < 2; round++) {
        float* slot = mg + qt * (2048 + 64);
        if (ks == 1 && d == round) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                *(float4*)(slot + (c * 2 + 0) * 128 + lane * 4) = (float4){O[c][0], O[c][1], O[c][2], O[c][3]};
                *(float4*)(slot + (c * 2 + 1) * 128 + lane * 4) = (float4){O[c][4], O[c][5], O[c][6], O[c][7]};
            }
            *(float2*)(slot + 2048 + lane * 2) = (float2){m_run, l_run};
        }
        A5_BAR();
        if (ks == 0 && d == round) {
            const float2 ml2 = *(const float2*)(slot + 2048 + lane * 2);
            const float mo = ml2.x, lo = ml2.y;
            const float mn = fmaxf(m_run, mo);
            const float a0 = (m_run == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m_run - mn);
            const float a1 = (mo == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(mo - mn);
            if (qok) {
                float* pp = partials + (((size_t)r * n_heads + head) * max_spans + span_idx) * A5_PSTRIDE;
                if (d == 0 && half == 0) {
                    pp[0] = mn;
                    pp[1] = l_run * a0 + lo * a1;
                }
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    const float4 x = *(const float4*)(slot + (c * 2 + 0) * 128 + lane * 4);
                    const float4 z = *(const float4*)(slot + (c * 2 + 1) * 128 + lane * 4);
                    float4 w0, w1;
                    w0.x = O[c][0] * a0 + x.x * a1; w0.y = O[c][1] * a0 + x.y * a1;
                    w0.z = O[c][2] * a0 + x.z * a1; w0.w = O[c][3] * a0 + x.w * a1;
                    w1.x = O[c][4] * a0 + z.x * a1; w1.y = O[c][5] * a0 + z.y * a1;
                    w1.z = O[c][6] * a0 + z.z * a1; w1.w = O[c][7] * a0 + z.w * a1;
                    // WMMA C order: 8 floats of (chunk 8d+c, half) contiguous
                    float* dst = pp + 4 + ((d * 8 + c) * 2 + half) * 8;
                    *(float4*)(dst) = w0;
                    *(float4*)(dst + 4) = w1;
                }
            }
        }
        if (round == 0) A5_BAR();
    }
}

#define A5_KERNEL(NAME, QT)                                                                          \
extern "C" __launch_bounds__(128 * QT) __global__ void NAME(                                          \
    const float* __restrict__ q, const unsigned char* __restrict__ k_cache,                           \
    const unsigned char* __restrict__ v_cache, float* __restrict__ partials,                          \
    const int* __restrict__ positions, int n_heads, float scale_log2, int n_blk, int max_spans,       \
    int batch_offset, int rows) {                                                                     \
    __shared__ __attribute__((aligned(16))) uint32_t lds[A5Cfg<QT>::LDS_DW];                          \
    a5_body<QT>(q, k_cache, v_cache, partials, positions, n_heads, scale_log2, n_blk, max_spans,      \
                batch_offset, rows, lds);                                                             \
}
A5_KERNEL(claude_attn5_q1, 1)
A5_KERNEL(claude_attn5_q2, 2)
A5_KERNEL(claude_attn5_q3, 3)

// Split-K reduce for claude_attn5 partials: one block per (head, row), 8
// warps over the spans (warp w: spans w, w+8, ..., all loads of a warp issued
// together: 5 per round covers the default 36 spans in one round trip), lane =
// 8 consecutive WMMA-C-order floats, then a cross-warp merge in LDS. exp2
// domain (m in log2 units).
#define A5_RU 5
extern "C" __launch_bounds__(256) __global__ void claude_attn5_reduce(
    const float* __restrict__ partials, float* __restrict__ out, const int* __restrict__ positions,
    int n_heads, int n_blk, int max_spans, int batch_offset)
{
    __shared__ float sm[8], sl[8];
    __shared__ __attribute__((aligned(16))) float sg[8][256];
    const int h = blockIdx.x, row = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int seq_len = positions[batch_offset + row] + 1;
    const int n = min(n_blk, (seq_len + A5_KT - 1) / A5_KT);
    const float* base = partials + ((size_t)row * n_heads + h) * max_spans * A5_PSTRIDE;
    float m = -INFINITY, l = 0.0f, g[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) g[i] = 0.0f;
    for (int s0 = warp; s0 < n; s0 += 8 * A5_RU) {
        float2 mlv[A5_RU];
        float4 a[A5_RU], b[A5_RU];
        #pragma unroll
        for (int u = 0; u < A5_RU; u++) {
            const int s = s0 + 8 * u;
            if (s < n) {
                const float* p = base + (size_t)s * A5_PSTRIDE;
                mlv[u] = *(const float2*)p;
                a[u] = *(const float4*)(p + 4 + lane * 8);
                b[u] = *(const float4*)(p + 8 + lane * 8);
            } else {
                mlv[u] = (float2){-INFINITY, 0.0f};
                a[u] = (float4){0.f, 0.f, 0.f, 0.f};
                b[u] = a[u];
            }
        }
        float bm = -INFINITY;
        #pragma unroll
        for (int u = 0; u < A5_RU; u++) if (mlv[u].y > 0.0f) bm = fmaxf(bm, mlv[u].x);
        if (bm > m) {
            const float c = __builtin_amdgcn_exp2f(m - bm);
            l *= c;
            #pragma unroll
            for (int i = 0; i < 8; i++) g[i] *= c;
            m = bm;
        }
        #pragma unroll
        for (int u = 0; u < A5_RU; u++) {
            if (!(mlv[u].y > 0.0f)) continue;
            const float c = __builtin_amdgcn_exp2f(mlv[u].x - m);
            l += mlv[u].y * c;
            g[0] += a[u].x * c; g[1] += a[u].y * c; g[2] += a[u].z * c; g[3] += a[u].w * c;
            g[4] += b[u].x * c; g[5] += b[u].y * c; g[6] += b[u].z * c; g[7] += b[u].w * c;
        }
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    *(float4*)(&sg[warp][lane * 8]) = (float4){g[0], g[1], g[2], g[3]};
    *(float4*)(&sg[warp][lane * 8 + 4]) = (float4){g[4], g[5], g[6], g[7]};
    __syncthreads();
    // thread t -> output dim t; C-order index of dim t: chunk c = t>>4, j = (t&15)>>1, half = t&1
    const int t = threadIdx.x;
    const int ci = ((t >> 4) * 2 + (t & 1)) * 8 + ((t & 15) >> 1);
    float M = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 8; w++) if (sl[w] > 0.0f) M = fmaxf(M, sm[w]);
    float L = 0.0f, G = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        if (!(sl[w] > 0.0f)) continue;
        const float c = __builtin_amdgcn_exp2f(sm[w] - M);
        L += sl[w] * c;
        G += sg[w][ci] * c;
    }
    out[((size_t)row * n_heads + h) * 256 + t] = L > 0.0f ? G / L : 0.0f;
}
