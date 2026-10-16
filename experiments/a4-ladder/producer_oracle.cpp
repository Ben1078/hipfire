// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.
//
// Standalone producer-candidate correctness probe for the fused A4 ladder
// (IU4_A4_CANDIDATES = 1/2/4/8). It compiles the PRODUCTION emitter
// `kernels/src/block_i4_128_quant.hip` (included below, unmodified) and, on
// one device and one set of inputs, runs:
//
//   producer : emit_iu4_sidecar_from_producer8   (lane p holds x[8p..8p+7] of a
//              256-wide group, two 128-blocks per wave; the real C2 one-pass
//              path when the build selects it)
//   generic  : quantize_block_i4_128_wave<true>  (same IU4_A4_CANDIDATES, lane l
//              owns x[4l..4l+3] of one 128-block)
//   full8    : quantize_block_i4_128_wave<false> (the original 8-point grid)
//
// Hard checks (any failure => exit 1):
//   * producer bytes == generic bytes for every 72-byte block (d, s, qs);
//     this is the bit-identity claim of the one-pass/remap emit.
//   * IU4_A4_CANDIDATES == 8: generic == full8 and producer == full8 (the
//     8-candidate producer equals the standalone full8 reference).
//   * host checks on EVERY output (producer, generic, full8), none of which
//     depend on a host reduction order matching the device:
//       - amax == 0  -> d == 1.0f, s == 0, all codes 0
//       - d finite and >= 0; d is a member of the host-computed candidate set
//         (the exact device float expressions), or the 1.0f sentinel when no
//         candidate's block mse is below 1e30 (N>1 only; N==1 is d = amax/7);
//       - every emitted code == clamp(rintf(x / d), -8, 7) computed on the
//         host with IEEE float division (exact, no reduction involved);
//       - s == exact integer sum of the decoded codes; code*d finite;
//       - the chosen candidate's host (double) mse is within 1e-3 relative of
//         the host-best candidate of the set (skipped, and counted, where the
//         device float mse is degenerate: mse < 1e-33 or >= 1e29);
//     plus: for N < 8 the full8 result is not worse than the producer result.
//   * guard blocks after the output are untouched, no block left unwritten.
//
// Byte equality is only asserted between DEVICE results. Host values are used
// only where they are order-free (amax, codes for a given d, integer sums) or
// compared with a relative tolerance (mse). No byte claim is made between host
// and device reductions.
//
// Build (one binary per variant; see run_producer_oracle.sh):
//   hipcc -O3 -std=c++17 --offload-arch=gfx1201 -mcode-object-version=6 \
//         -fuse-cuid=none -DIU4_A4_CANDIDATES=2 -o po_1201_c2 producer_oracle.cpp
//   hipcc ... --offload-arch=gfx1151 -DIU4_A4_CANDIDATES=2 \
//         -DHIPFIRE_IU4_ONEPASS=1 ...           # real gfx1151 C2 path
// Optional: -DPO_SLAB=1 (one-pass C2 only) checks the `_b1s` slab layout.
// Do NOT pass -march=native / -ffast-math: the host checks assume IEEE float.

#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1
#if defined(PO_SLAB)
#define HIPFIRE_IU4_SLAB 1
#endif
#include "../../kernels/src/block_i4_128_quant.hip"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__) && !defined(__gfx1100__) && !defined(__gfx1151__)
#error "producer_oracle targets gfx1201 / gfx1151 / gfx1100 (the arches whose producers use _wave<true>)"
#endif

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

// ------------------------------------------------------------------ device --

// out[0]: bit0 = HIPFIRE_IU4_C2_ONEPASS, bit1 = HIPFIRE_IU4_SLAB
// out[1]: IU4_A4_CANDIDATES, out[2]: device-pass arch (1201 / 1151 / 1100)
extern "C" __global__ __launch_bounds__(32) void po_config(int* out) {
    int v = 0;
#if defined(HIPFIRE_IU4_C2_ONEPASS)
    v |= 1;
#endif
#if defined(HIPFIRE_IU4_SLAB)
    v |= 2;
#endif
    int arch = 0;
#if defined(__gfx1201__)
    arch = 1201;
#elif defined(__gfx1151__)
    arch = 1151;
#elif defined(__gfx1100__)
    arch = 1100;
#endif
    if (threadIdx.x == 0) {
        out[0] = v;
        out[1] = IU4_A4_CANDIDATES;
        out[2] = arch;
    }
}

// Producer ownership: lane p holds x[8p..8p+7] of the 256-wide group
// (half h = p >> 4). X is [N tokens][K]; Y is [K/128][N] blocks.
extern "C" __global__ __launch_bounds__(32) void po_producer(
    const float* __restrict__ X, block_i4_128* __restrict__ Y, int K, int N) {
    const int group = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    const float4* p = reinterpret_cast<const float4*>(
        X + (long long)token * K + (long long)group * 256 + lane * 8);
    const float4 a = p[0];
    const float4 b = p[1];
    emit_iu4_sidecar_from_producer8(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w,
                                    Y, group, token, N, lane);
}

// Standalone ownership: lane l owns x[4l..4l+3] of 128-block `kblock`.
extern "C" __global__ __launch_bounds__(32) void po_ref_generic(
    const float* __restrict__ X, block_i4_128* __restrict__ Y, int K, int N) {
    const int kblock = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    const float4 v = *reinterpret_cast<const float4*>(
        X + (long long)token * K + (long long)kblock * 128 + lane * 4);
    const float xv[4] = {v.x, v.y, v.z, v.w};
    quantize_block_i4_128_wave<true>(xv, Y + (long long)kblock * N + token, lane);
}

extern "C" __global__ __launch_bounds__(32) void po_ref_full8(
    const float* __restrict__ X, block_i4_128* __restrict__ Y, int K, int N) {
    const int kblock = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    const float4 v = *reinterpret_cast<const float4*>(
        X + (long long)token * K + (long long)kblock * 128 + lane * 4);
    const float xv[4] = {v.x, v.y, v.z, v.w};
    quantize_block_i4_128_wave<false>(xv, Y + (long long)kblock * N + token, lane);
}

// -------------------------------------------------------------------- host --

#define HIP_CHECK(expr)                                                              \
    do {                                                                             \
        hipError_t e_ = (expr);                                                      \
        if (e_ != hipSuccess) {                                                      \
            fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,          \
                    hipGetErrorString(e_));                                          \
            exit(2);                                                                 \
        }                                                                            \
    } while (0)

static constexpr int BLK_ELEMS = 128;
static constexpr int GUARD_BLOCKS = 64;
static constexpr unsigned char POISON = 0xA5;

static uint32_t fbits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
static float bitsf(uint32_t u) { float v; memcpy(&v, &u, 4); return v; }

// ---- deterministic RNG (splitmix64) ----
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double u01() { return ((next() >> 11) + 1) * (1.0 / 9007199254740993.0); }  // (0,1]
    double gauss() {
        const double r = std::sqrt(-2.0 * std::log(u01()));
        return r * std::cos(6.283185307179586 * u01());
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }
};

static float bf16_rt(float v) {
    uint32_t u = fbits(v);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    return bitsf(u);
}

struct Blk {
    std::string kind;
    std::array<float, BLK_ELEMS> x;
};

// ---- host reference math (IEEE float; mirrors the device expressions) ----

// Candidate set exactly as the device builds it (float arithmetic, same
// association). N==1: d = amax/7. N==8: the full grid. N==4/2: multiplier set.
static int host_candidates(int n, float amax, float out[8]) {
    if (n == 1) { out[0] = amax / 7.0f; return 1; }
    if (n == 8) {
        for (int ci = 0; ci < 8; ++ci)
            out[ci] = (amax / 7.0f) * 0.5f * (1.0f + (float)ci / 7.0f);
        return 8;
    }
    const float base = (amax / 7.0f) * 0.5f;
    if (n == 2) {
        out[0] = base * 0x1.b6db6ep+0f;
        out[1] = base * 2.0f;
        return 2;
    }
    out[0] = base * 1.0f;
    out[1] = base * 0x1.492492p+0f;
    out[2] = base * 0x1.b6db6ep+0f;
    out[3] = base * 2.0f;
    return 4;
}

static int host_code(float x, float d) {
    float q = std::nearbyintf(x / d);
    q = std::fmin(std::fmax(q, -8.0f), 7.0f);  // NaN -> -8, as fmaxf/fminf on device
    return (int)q;
}

static double host_mse(const float* x, float d) {
    double m = 0.0;
    for (int e = 0; e < BLK_ELEMS; ++e) {
        const int q = host_code(x[e], d);
        const double err = (double)x[e] - (double)q * (double)d;
        m += err * err;
    }
    return m;
}

static float host_amax(const float* x) {
    float a = 0.0f;
    for (int e = 0; e < BLK_ELEMS; ++e) a = std::fmax(a, std::fabs(x[e]));
    return a;
}

static int nib_decode(const block_i4_128& b, int k) {
    const unsigned char byte = b.qs[k >> 1];
    const int v = (k & 1) ? (byte >> 4) : (byte & 15);
    return v >= 8 ? v - 16 : v;
}

// ---- reporting ----
struct Counter { long eval = 0, fail = 0; };
static std::map<std::string, Counter> g_hard;
static std::map<std::string, long> g_info;
static int g_examples = 4;

static void chk(const std::string& name, bool ok, const Blk& b, size_t i, const char* fmt, ...) {
    Counter& c = g_hard[name];
    c.eval++;
    if (ok) return;
    c.fail++;
    if (c.fail <= g_examples) {
        char msg[512];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);
        fprintf(stderr, "FAIL %-40s block %zu kind=%s : %s\n  x[0..7] bits:", name.c_str(), i,
                b.kind.c_str(), msg);
        for (int e = 0; e < 8; ++e) fprintf(stderr, " %08x", fbits(b.x[e]));
        fprintf(stderr, "\n");
    }
}

// Host checks for one emitted block. `nset` is the candidate count this
// emitter is supposed to search. Returns the emitted candidate index (-1 for
// sentinel/zero) and the host mse of the emitted d (NaN if not computed).
static int check_host(const char* who, const block_i4_128& o, const Blk& b, size_t i, int nset,
                      double* mse_out) {
    const std::string w = who;
    const float* x = b.x.data();
    const float amax = host_amax(x);
    const float d = o.d;
    *mse_out = std::nan("");

    int sum = 0;
    bool deq_ok = true;
    int dec[BLK_ELEMS];
    for (int e = 0; e < BLK_ELEMS; ++e) {
        dec[e] = nib_decode(o, e);
        sum += dec[e];
        if (!std::isfinite((float)dec[e] * d)) deq_ok = false;
    }
    chk(w + ":d_finite_nonneg", std::isfinite(d) && d >= 0.0f, b, i, "d=%a bits=%08x", d, fbits(d));
    chk(w + ":s_equals_code_sum", o.s == sum, b, i, "s=%d codesum=%d", o.s, sum);
    chk(w + ":dequant_finite", deq_ok, b, i, "d=%a", d);

    if (amax == 0.0f) {
        bool zero_codes = true;
        for (int e = 0; e < BLK_ELEMS; ++e) zero_codes &= (dec[e] == 0);
        chk(w + ":zero_block", fbits(d) == fbits(1.0f) && o.s == 0 && zero_codes, b, i,
            "d=%a s=%d zero_codes=%d", d, o.s, (int)zero_codes);
        return -1;
    }

    // d must come from the candidate set (or the sentinel for N>1).
    float cand[8];
    const int n = host_candidates(nset, amax, cand);
    int idx = -1;
    for (int c = 0; c < n; ++c)
        if (fbits(cand[c]) == fbits(d)) { idx = c; break; }
    double mse_c[8];
    double minmse = 1.0e300, maxmse = 0.0;
    for (int c = 0; c < n; ++c) {
        mse_c[c] = host_mse(x, cand[c]);
        minmse = std::min(minmse, mse_c[c]);
        maxmse = std::max(maxmse, mse_c[c]);
    }
    bool sentinel = false;
    if (idx < 0) {
        sentinel = nset > 1 && fbits(d) == fbits(1.0f);
        chk(w + ":d_in_candidate_set", sentinel, b, i, "d=%a bits=%08x amax=%a N=%d", d, fbits(d),
            amax, nset);
        if (sentinel) {
            chk(w + ":sentinel_only_if_all_mse_ge_1e30", minmse >= 1.0e30 * (1.0 - 1.0e-3), b, i,
                "host min mse=%g", minmse);
            *mse_out = host_mse(x, 1.0f);
        }
    } else {
        chk(w + ":d_in_candidate_set", true, b, i, " ");
        if (nset > 1) {
            *mse_out = mse_c[idx];
            chk(w + ":selected_mse_below_sentinel", mse_c[idx] < 1.0e30 * (1.0 + 1.0e-3), b, i,
                "host mse=%g idx=%d", mse_c[idx], idx);
            if (minmse < 1.0e29 && maxmse >= 1.0e-33) {
                chk(w + ":selected_mse_near_best", mse_c[idx] <= minmse * (1.0 + 1.0e-3), b, i,
                    "host mse(sel idx %d)=%.9g best=%.9g", idx, mse_c[idx], minmse);
            } else {
                g_info[w + ":mse_near_best_skipped_degenerate"]++;
            }
        }
    }

    // Codes for the emitted d: exact, no reduction involved.
    int first_bad = -1;
    for (int e = 0; e < BLK_ELEMS; ++e) {
        if (host_code(x[e], d) != dec[e]) { first_bad = e; break; }
    }
    chk(w + ":codes_equal_host_rint_clamp", first_bad < 0, b, i,
        "elem %d x=%a d=%a emitted=%d host=%d", first_bad, first_bad >= 0 ? x[first_bad] : 0.0f, d,
        first_bad >= 0 ? dec[first_bad] : 0, first_bad >= 0 ? host_code(x[first_bad], d) : 0);
    return idx;
}

static bool same_block(const block_i4_128& a, const block_i4_128& b) {
    return memcmp(&a, &b, sizeof(block_i4_128)) == 0;
}

static void describe_diff(char* buf, size_t n, const block_i4_128& a, const block_i4_128& b) {
    int nd = 0, first = -1;
    for (int k = 0; k < 64; ++k)
        if (a.qs[k] != b.qs[k]) { if (first < 0) first = k; ++nd; }
    snprintf(buf, n, "d %08x/%08x s %d/%d qs_diff_bytes=%d first@%d %02x/%02x", fbits(a.d), fbits(b.d),
             a.s, b.s, nd, first, first >= 0 ? a.qs[first] : 0, first >= 0 ? b.qs[first] : 0);
}

// ---- input generation ----
static void add(std::vector<Blk>& v, const std::string& kind, const std::array<float, BLK_ELEMS>& x) {
    v.push_back(Blk{kind, x});
}
static std::array<float, BLK_ELEMS> filled(float val) {
    std::array<float, BLK_ELEMS> a;
    a.fill(val);
    return a;
}

static void build_edge_blocks(std::vector<Blk>& v, Rng& rng) {
    // zeros
    add(v, "zero_pos", filled(0.0f));
    add(v, "zero_neg", filled(-0.0f));
    {
        auto a = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) a[i] = (i & 1) ? -0.0f : 0.0f;
        add(v, "zero_mixed_sign", a);
    }
    // single nonzero element at lane-boundary indices (producer lane = e/8, half = e/64)
    for (int idx : {0, 1, 3, 4, 7, 8, 15, 16, 63, 64, 65, 120, 127}) {
        auto a = filled(0.0f);
        a[idx] = (idx & 1) ? -0.3125f : 5.0f;
        add(v, "single_nonzero", a);
        auto n = filled(-0.0f);
        n[idx] = -7.0f;
        add(v, "single_neg_with_negzero", n);
    }
    // +-amax saturation
    add(v, "all_pos_amax", filled(3.5f));
    add(v, "all_neg_amax", filled(-3.5f));
    {
        auto a = filled(0.0f), b = filled(0.0f), c = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) {
            a[i] = (i & 1) ? -2.0f : 2.0f;
            b[i] = (i & 1) ? 0.0f : ((i & 2) ? -6.0f : 6.0f);
            c[i] = (i < 64) ? 1.0f : -1.0f;
        }
        add(v, "alt_pm_amax", a);
        add(v, "pm_amax_with_zero", b);
        add(v, "half_pos_half_neg_amax", c);
    }
    {
        auto a = filled(0.0f), b = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) {
            a[i] = 0.5f + (float)(i % 7) / 16.0f;
            b[i] = -(0.5f + (float)(i % 7) / 16.0f);
        }
        a[0] = -1.0f;    // min == -amax, rest positive: d0 saturates to -8
        b[127] = 1.0f;   // max == +amax, rest negative: d0 saturates to 7
        add(v, "neg_amax_rest_pos", a);
        add(v, "pos_amax_rest_neg", b);
    }
    // distinct per-element ramps: any lane/element permutation changes bytes
    {
        auto a = filled(0.0f), b = filled(0.0f), c = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) {
            a[i] = ((float)i - 63.5f) / 16.0f;
            b[i] = (float)(i + 1);
            c[i] = ((i & 1) ? -1.0f : 1.0f) * std::ldexp(1.0f, (i % 24) - 12);
        }
        add(v, "ramp_symmetric", a);
        add(v, "ramp_positive", b);
        add(v, "ramp_pow2_signed", c);
    }
    // rint ties: amax = 7*s so d1 = s exactly and x/d1 hits half-integers
    for (float s : {1.0f, 0x1p-10f, 0x1p+20f, 0x1p-40f, 0x1p+60f}) {
        auto a = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i)
            a[i] = (((i / 7) & 1) ? -1.0f : 1.0f) * ((float)(i % 7) + 0.5f) * s;
        a[127] = 7.0f * s;
        add(v, "tie_half_d1", a);
    }
    // just around the half-integer tie (the one-pass |y-rint(y)| <= 0.5-2^-18 test)
    const float factors[] = {1.0f,
                             std::nextafter(1.0f, 2.0f),
                             std::nextafter(1.0f, 0.0f),
                             1.0f + 0x1p-22f, 1.0f - 0x1p-22f,
                             1.0f + 0x1p-19f, 1.0f - 0x1p-19f,
                             1.0f + 0x1p-18f, 1.0f - 0x1p-18f,
                             1.0f + 0x1p-17f, 1.0f - 0x1p-17f};
    for (float f : factors) {
        auto a = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i)
            a[i] = (((i / 5) & 1) ? -1.0f : 1.0f) * ((float)(i % 7) + 0.5f) * f;
        a[127] = 7.0f;
        add(v, "near_half_d1", a);
    }
    // half-integer ties of every candidate d of the N=8/4/2 sets at amax = 7
    {
        std::vector<float> ds;
        for (int n : {8, 4, 2}) {
            float c[8];
            const int m = host_candidates(n, 7.0f, c);
            for (int k = 0; k < m; ++k) ds.push_back(c[k]);
        }
        for (float dc : ds) {
            const int kmax = std::max(0, (int)(6.99f / dc - 0.5f));
            auto a = filled(0.0f), b = filled(0.0f);
            for (int i = 0; i < BLK_ELEMS; ++i) {
                const float sgn = ((i / 3) & 1) ? -1.0f : 1.0f;
                a[i] = sgn * ((float)(i % (kmax + 1)) + 0.5f) * dc;
                b[i] = (i & 1) ? std::nextafter(a[i], 0.0f) : std::nextafter(a[i], sgn * 16.0f);
            }
            a[127] = 7.0f;
            b[127] = 7.0f;
            add(v, "near_half_candidate_exact", a);
            add(v, "near_half_candidate_ulp", b);
        }
    }
    // large outlier at lane-boundary elements over a small ramp
    for (int e : {0, 3, 4, 7, 8, 15, 16, 63, 64, 65, 100, 127}) {
        auto a = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) a[i] = ((float)((i * 37) % 101) - 50.0f) / 64.0f;
        a[e] = (e & 1) ? -1000.0f : 1000.0f;
        add(v, "outlier_at_element", a);
    }
    // huge finite (float mse overflows -> d = 1 sentinel path on N>1)
    for (double sc : {1.0e15, 1.0e30, 2.0e37}) {
        auto a = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) a[i] = (float)(rng.gauss() * sc);
        add(v, "huge_gauss", a);
    }
    {
        auto a = filled(0.0f), b = filled(0.0f), c = filled(0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i) {
            a[i] = (i & 1) ? -FLT_MAX : FLT_MAX;
            c[i] = (float)(rng.gauss() * 1.0e-20);
        }
        b[5] = FLT_MAX;
        c[0] = FLT_MAX;
        add(v, "huge_alt_fltmax", a);
        add(v, "huge_single_fltmax", b);
        add(v, "huge_with_tiny", c);
    }
    // subnormal / tiny (the one-pass 2^-100 amax gate)
    {
        const float dmin = std::numeric_limits<float>::denorm_min();
        const float lim = 0x1p-100f;
        auto a = filled(0.0f), b = filled(0.0f), c = filled(0.0f), d = filled(0.0f), e = filled(0.0f);
        a[0] = dmin;
        for (int i = 0; i < BLK_ELEMS; ++i) {
            b[i] = (float)rng.range(-100, 100) * dmin;
            c[i] = (float)(rng.gauss() * FLT_MIN);
            d[i] = (float)(rng.gauss() * 1.0e-35);
            e[i] = std::fmax(-lim, std::fmin(lim, (float)(rng.gauss() * 0.25) * lim));
        }
        e[9] = lim;  // amax == 2^-100 exactly (the gate is >=)
        auto f = e;
        const float below = std::nextafter(lim, 0.0f);
        for (int i = 0; i < BLK_ELEMS; ++i)
            if (std::fabs(f[i]) > below) f[i] = (f[i] < 0) ? -below : below;
        add(v, "subnormal_single", a);
        add(v, "subnormal_ramp", b);
        add(v, "tiny_flt_min_gauss", c);
        add(v, "tiny_1e-35_gauss", d);
        add(v, "amax_at_2^-100", e);
        add(v, "amax_below_2^-100", f);
    }
}

static void build_random_block(std::vector<Blk>& v, Rng& rng, int r) {
    std::array<float, BLK_ELEMS> a;
    const double sc = std::ldexp(1.0, rng.range(-20, 20));
    const char* kind = "";
    switch (r % 10) {
        case 0: kind = "bulk_gauss";
            for (auto& e : a) e = (float)(rng.gauss() * sc);
            break;
        case 1: kind = "bulk_uniform";
            for (auto& e : a) e = (float)((2.0 * rng.u01() - 1.0) * sc);
            break;
        case 2: kind = "bulk_laplace";
            for (auto& e : a) {
                const double u = rng.u01() - 0.5;
                e = (float)(-sc * (u < 0 ? -1.0 : 1.0) * std::log(1.0 - 2.0 * std::fabs(u) + 1e-300));
            }
            break;
        case 3: kind = "bulk_sparse";
            for (auto& e : a) e = rng.u01() < 0.8 ? 0.0f : (float)(rng.gauss() * sc);
            break;
        case 4: kind = "bulk_outlier";
            for (auto& e : a) e = (float)(rng.gauss() * sc);
            for (int k = rng.range(1, 3); k > 0; --k)
                a[rng.range(0, 127)] = (float)(rng.gauss() * sc * (10.0 + 1.0e4 * rng.u01()));
            break;
        case 5: kind = "bulk_int_grid";
            for (auto& e : a) e = (float)rng.range(-7, 7) * (float)sc;
            break;
        case 6: kind = "bulk_half_grid";
            for (auto& e : a) e = ((float)rng.range(-6, 6) + 0.5f) * (float)sc;
            a[rng.range(0, 127)] = 7.0f * (float)sc;
            break;
        case 7: kind = "bulk_bf16_gauss";
            for (auto& e : a) e = bf16_rt((float)(rng.gauss() * sc));
            break;
        case 8: kind = "bulk_bf16_outlier";
            for (auto& e : a) e = bf16_rt((float)(rng.gauss() * sc));
            a[rng.range(0, 127)] = bf16_rt((float)(rng.gauss() * sc * 300.0));
            break;
        default: kind = "bulk_two_scale";
            for (int i = 0; i < BLK_ELEMS; ++i)
                a[i] = (float)(rng.gauss() * sc * ((i & 4) ? 1.0e-3 : 1.0));
            break;
    }
    add(v, kind, a);
}

// ---- slab decode (HIPFIRE_IU4_SLAB producer layout -> canonical blocks) ----
// Per K128 block index kb = 2*group + half: planes [d 4N][s 4N][qs 0..31: 32N][qs 32..63: 32N].
static void unslab(const unsigned char* raw, std::vector<block_i4_128>& out, int nkb, int N) {
    for (int kb = 0; kb < nkb; ++kb) {
        const unsigned char* base = raw + (size_t)kb * N * 72;
        for (int t = 0; t < N; ++t) {
            block_i4_128& o = out[(size_t)kb * N + t];
            memcpy(&o.d, base + (size_t)t * 4, 4);
            memcpy(&o.s, base + 4 * (size_t)N + (size_t)t * 4, 4);
            memcpy(o.qs, base + 8 * (size_t)N + (size_t)t * 32, 32);
            memcpy(o.qs + 32, base + 40 * (size_t)N + (size_t)t * 32, 32);
        }
    }
}

static bool guard_intact(const std::vector<unsigned char>& raw, size_t nb) {
    for (size_t i = nb * 72; i < raw.size(); ++i)
        if (raw[i] != POISON) return false;
    return true;
}

int main(int argc, char** argv) {
    int random_blocks = 8192, tokens = 7, device = 0;
    uint64_t seed = 0x4134ull;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--random")) random_blocks = atoi(next("--random"));
        else if (!strcmp(argv[i], "--tokens")) tokens = atoi(next("--tokens"));
        else if (!strcmp(argv[i], "--device")) device = atoi(next("--device"));
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(next("--seed"), nullptr, 0);
        else if (!strcmp(argv[i], "--examples")) g_examples = atoi(next("--examples"));
        else {
            fprintf(stderr,
                    "usage: %s [--random N=8192] [--tokens T=7] [--device D=0] [--seed S] "
                    "[--examples K=4]\n", argv[0]);
            return 2;
        }
    }
    if (tokens < 1 || random_blocks < 0) { fprintf(stderr, "bad arguments\n"); return 2; }

    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    const std::string arch = prop.gcnArchName;

    // ---- which path did the device pass compile? ----
    int* d_cfg = nullptr;
    HIP_CHECK(hipMalloc(&d_cfg, 3 * sizeof(int)));
    HIP_CHECK(hipMemset(d_cfg, 0, 3 * sizeof(int)));
    hipLaunchKernelGGL(po_config, dim3(1), dim3(32), 0, 0, d_cfg);
    HIP_CHECK(hipGetLastError());
    int cfg[3] = {0, 0, 0};
    HIP_CHECK(hipMemcpy(cfg, d_cfg, sizeof cfg, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_cfg));
    const bool dev_onepass = cfg[0] & 1;
    const bool dev_slab = cfg[0] & 2;
    const int N = cfg[1];  // IU4_A4_CANDIDATES as compiled for the device
    const int dev_arch = cfg[2];

    const bool arch_1201 = arch.rfind("gfx1201", 0) == 0;
    const bool arch_1151 = arch.rfind("gfx1151", 0) == 0;
#if defined(PO_SLAB)
    constexpr bool want_slab = true;
#else
    constexpr bool want_slab = false;
#endif
    bool config_ok = true;
    bool expect_onepass = false;
    if (arch_1201) expect_onepass = (N == 2);
#if defined(HIPFIRE_IU4_ONEPASS)
    if (arch_1151) expect_onepass = (N == 2);
#endif
    if (arch_1201 && dev_arch != 1201) config_ok = false;
    if (arch_1151 && dev_arch != 1151) config_ok = false;
    if (N != IU4_A4_CANDIDATES) config_ok = false;
    if (dev_onepass != expect_onepass) config_ok = false;
    if (dev_slab != want_slab) config_ok = false;

    const char* path = dev_onepass ? "one-pass C2 (emit_iu4_sidecar_c2_gfx12, DPP row_xmask)"
                                   : "remap (shfl) + quantize_block_i4_128_wave<true>";
    printf("producer_oracle: device %d arch=%s device-pass-arch=%d candidates=%d slab=%d\n", device,
           arch.c_str(), dev_arch, N, (int)dev_slab);
    printf("producer_oracle: producer path = %s\n", path);
    if (!config_ok) {
        fprintf(stderr,
                "FAIL config: device path does not match the build flags (device onepass=%d, expected "
                "%d; gfx1151 C2 needs -DHIPFIRE_IU4_ONEPASS=1 -DIU4_A4_CANDIDATES=2; slab device=%d "
                "wanted=%d)\n",
                (int)dev_onepass, (int)expect_onepass, (int)dev_slab, (int)want_slab);
    }

    // ---- inputs ----
    Rng rng(seed);
    std::vector<Blk> blocks;
    build_edge_blocks(blocks, rng);
    const size_t n_edge = blocks.size();
    for (int r = 0; r < random_blocks; ++r) build_random_block(blocks, rng, r);
    const int per_group = 2 * tokens;  // 2 K128 blocks per producer group
    const int G = (int)((blocks.size() + per_group - 1) / per_group);
    while ((int)blocks.size() < G * per_group) build_random_block(blocks, rng, (int)blocks.size());
    const size_t nb = blocks.size();  // == G * 2 * tokens; block i = (kblock i / T, token i % T)
    const int K = G * 256;
    printf("producer_oracle: %zu edge + %d random blocks (+pad) = %zu blocks, K=%d tokens=%d groups=%d\n",
           n_edge, random_blocks, nb, K, tokens, G);

    std::vector<float> hx((size_t)tokens * K);
    for (size_t i = 0; i < nb; ++i) {
        const int kb = (int)(i / tokens), t = (int)(i % tokens);
        memcpy(&hx[(size_t)t * K + (size_t)kb * BLK_ELEMS], blocks[i].x.data(), BLK_ELEMS * sizeof(float));
    }

    // ---- device buffers: [nb blocks][GUARD blocks poison] ----
    const size_t ybytes = (nb + GUARD_BLOCKS) * 72;
    float* dX = nullptr;
    unsigned char *dYp = nullptr, *dYr = nullptr, *dYf = nullptr;
    HIP_CHECK(hipMalloc(&dX, hx.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dYp, ybytes));
    HIP_CHECK(hipMalloc(&dYr, ybytes));
    HIP_CHECK(hipMalloc(&dYf, ybytes));
    HIP_CHECK(hipMemcpy(dX, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dYp, POISON, ybytes));
    HIP_CHECK(hipMemset(dYr, POISON, ybytes));
    HIP_CHECK(hipMemset(dYf, POISON, ybytes));

    hipLaunchKernelGGL(po_producer, dim3(G, tokens), dim3(32), 0, 0, dX,
                       reinterpret_cast<block_i4_128*>(dYp), K, tokens);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(po_ref_generic, dim3(2 * G, tokens), dim3(32), 0, 0, dX,
                       reinterpret_cast<block_i4_128*>(dYr), K, tokens);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(po_ref_full8, dim3(2 * G, tokens), dim3(32), 0, 0, dX,
                       reinterpret_cast<block_i4_128*>(dYf), K, tokens);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned char> rawP(ybytes), rawR(ybytes), rawF(ybytes);
    HIP_CHECK(hipMemcpy(rawP.data(), dYp, ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(rawR.data(), dYr, ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(rawF.data(), dYf, ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dYp));
    HIP_CHECK(hipFree(dYr));
    HIP_CHECK(hipFree(dYf));

    std::vector<block_i4_128> P(nb), R(nb), F(nb);
    if (dev_slab) unslab(rawP.data(), P, 2 * G, tokens);
    else memcpy(P.data(), rawP.data(), nb * 72);
    memcpy(R.data(), rawR.data(), nb * 72);
    memcpy(F.data(), rawF.data(), nb * 72);

    {
        Blk none{"guard", {}};
        chk("guard:producer_untouched", guard_intact(rawP, nb), none, 0, "producer wrote past the output");
        chk("guard:generic_untouched", guard_intact(rawR, nb), none, 0, "generic wrote past the output");
        chk("guard:full8_untouched", guard_intact(rawF, nb), none, 0, "full8 wrote past the output");
    }

    // ---- per-block checks ----
    struct KindStat { long blocks = 0, prod_ne_ref = 0, prod_ne_full8 = 0; };
    std::map<std::string, KindStat> kinds;
    long hist[8] = {0};
    long sentinel_blocks = 0, zero_blocks = 0, nonzero_blocks = 0;
    long p_ne_f = 0, excess_n = 0;
    double excess_sum = 0.0, excess_max = 0.0;

    for (size_t i = 0; i < nb; ++i) {
        const Blk& b = blocks[i];
        KindStat& ks = kinds[b.kind];
        ks.blocks++;

        char diff[200];
        const bool pr = same_block(P[i], R[i]);
        describe_diff(diff, sizeof diff, P[i], R[i]);
        chk("producer==generic_wave<true>(bytes)", pr, b, i, "%s", diff);
        if (!pr) ks.prod_ne_ref++;

        if (N == 8) {
            describe_diff(diff, sizeof diff, R[i], F[i]);
            chk("generic_wave<true>(N=8)==full8(bytes)", same_block(R[i], F[i]), b, i, "%s", diff);
            const bool pf = same_block(P[i], F[i]);
            describe_diff(diff, sizeof diff, P[i], F[i]);
            chk("producer(N=8)==standalone_full8(bytes)", pf, b, i, "%s", diff);
            if (!pf) ks.prod_ne_full8++;
        }

        double mseP, mseR, mseF;
        const int ip = check_host("producer", P[i], b, i, N, &mseP);
        check_host("generic", R[i], b, i, N, &mseR);
        check_host("full8", F[i], b, i, 8, &mseF);

        if (host_amax(b.x.data()) == 0.0f) { zero_blocks++; continue; }
        nonzero_blocks++;
        if (ip >= 0) hist[ip]++;
        else if (N > 1 && fbits(P[i].d) == fbits(1.0f)) sentinel_blocks++;

        if (N > 1 && N < 8) {
            if (fbits(P[i].d) != fbits(F[i].d)) p_ne_f++;
            const bool deg = !(mseP == mseP) || !(mseF == mseF) || mseP >= 1.0e29 || mseF >= 1.0e29 ||
                             std::max(mseP, mseF) < 1.0e-33;
            if (!deg) {
                chk("full8_not_worse_than_producer(host mse)", mseF <= mseP * (1.0 + 1.0e-3), b, i,
                    "full8 mse=%.9g producer mse=%.9g", mseF, mseP);
                if (mseF > 0.0) {
                    const double ex = mseP / mseF - 1.0;
                    excess_sum += ex;
                    excess_max = std::max(excess_max, ex);
                    excess_n++;
                }
            }
        }
    }

    // ---- summary ----
    printf("\n-- checks (eval / fail) --\n");
    long total_fail = 0;
    for (auto& kv : g_hard) {
        printf("  %-52s %9ld / %ld\n", kv.first.c_str(), kv.second.eval, kv.second.fail);
        total_fail += kv.second.fail;
    }
    for (auto& kv : g_info) printf("  info %-47s %9ld\n", kv.first.c_str(), kv.second);

    printf("\n-- producer candidate selection (N=%d, %ld nonzero blocks, %ld zero blocks) --\n", N,
           nonzero_blocks, zero_blocks);
    if (N > 1) {
        for (int c = 0; c < N; ++c)
            printf("  candidate %d chosen: %ld (%.2f%%)\n", c, hist[c],
                   nonzero_blocks ? 100.0 * hist[c] / nonzero_blocks : 0.0);
        printf("  sentinel d=1 chosen: %ld\n", sentinel_blocks);
    }
    if (N > 1 && N < 8)
        printf("  producer d != full8 d: %ld / %ld blocks; host mse excess vs full8: mean %.4g%% max %.4g%% "
               "over %ld blocks\n",
               p_ne_f, nonzero_blocks, excess_n ? 100.0 * excess_sum / excess_n : 0.0, 100.0 * excess_max,
               excess_n);

    bool any_kind_mismatch = false;
    for (auto& kv : kinds)
        if (kv.second.prod_ne_ref || kv.second.prod_ne_full8) {
            if (!any_kind_mismatch) printf("\n-- kinds with byte mismatches --\n");
            any_kind_mismatch = true;
            printf("  %-32s blocks=%ld producer!=generic=%ld producer!=full8=%ld\n", kv.first.c_str(),
                   kv.second.blocks, kv.second.prod_ne_ref, kv.second.prod_ne_full8);
        }

    const bool pass = total_fail == 0 && config_ok;
    printf("\nPRODUCER_ORACLE %s arch=%s candidates=%d onepass=%d slab=%d blocks=%zu fails=%ld\n",
           pass ? "PASS" : "FAIL", arch.c_str(), N, (int)dev_onepass, (int)dev_slab, nb, total_fail);
    return pass ? 0 : 1;
}
