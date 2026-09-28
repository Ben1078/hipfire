#include <hip/hip_runtime.h>

// claude: fused_silu_mul_mq_rotate_awq + int8 activation quant (claude_q8x).
//
// GENERATED body: kernels/src/fused_silu_mul_mq_rotate_awq.hip exactly as the
// host compiles it (kernels::FUSED_SILU_MUL_MQ_ROTATE_AWQ_SRC: HIPFIRE_SILU_HIN
// / IU4_SIDECAR / KERNEL undefined), preprocessor-resolved with unifdef so
// every arithmetic expression is textually identical (x_rot is bit-identical).
// Additions only: the `xq` / `dx` arguments and the claude_q8_emit8 epilogue
// (b = blockIdx.y, g = group = blockIdx.x, t = tid).

// claude: int8 activation quant of one lane's 8 consecutive values (elements
// g*256 + t*8 .. +7 of row b) for the small-batch verify GEMV claude_q8x_gemv*.
// Same expressions, 4-lane block reduction, packing and stores as
// claude_q8x_quant (claude_q8x.hip) applied to the f32 values this lane stores
// to x_rot, so xq/dx are bit-identical to running that kernel afterwards.
// Lanes 4k..4k+3 (one 32-element block) must be converged in the same wave.
__device__ __forceinline__ void claude_q8_emit8(
    float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7,
    unsigned int* __restrict__ xq,   // [B][K/256][32] uint2
    float* __restrict__ dx,          // [B][K/32]
    int K, int b, int g, int t)
{
    const float v[8] = {f0, f1, f2, f3, f4, f5, f6, f7};
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
    // 32-element block = 4 lanes (4k..4k+3).
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    const float d = amax * (1.0f / 127.0f);
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q[8];
#pragma unroll
    for (int j = 0; j < 8; j++) q[j] = (int)rintf(v[j] * id);
    const unsigned int e = (unsigned)(q[0] & 0xFF) | ((unsigned)(q[2] & 0xFF) << 8)
                         | ((unsigned)(q[4] & 0xFF) << 16) | ((unsigned)(q[6] & 0xFF) << 24);
    const unsigned int o = (unsigned)(q[1] & 0xFF) | ((unsigned)(q[3] & 0xFF) << 8)
                         | ((unsigned)(q[5] & 0xFF) << 16) | ((unsigned)(q[7] & 0xFF) << 24);
    uint2* xq2 = (uint2*)xq;
    xq2[((long long)b * (K >> 8) + g) * 32 + t] = make_uint2(e, o);
    if ((t & 3) == 0) dx[(long long)b * (K >> 5) + g * 8 + (t >> 2)] = d;
}

// AWQ-aware variant of `fused_silu_mul_mq_rotate`.
//
// Phase A Stage A — F2 follow-up. Original kernel is used as the
// down_proj / w_down input stage (silu(gate)*up → FWHT). Stage A's AWQ
// whitelist excluded down_proj because no AWQ-aware silu_mul_rotate
// existed for that path. This kernel completes the AWQ math
// `(W·s) · (silu(g)*u / s) = W · silu(g)*u` by dividing each element by
// `awq_scale[i]` AFTER the silu*up reduction but BEFORE the signs1 gather
// and FWHT. Exact mirror of `fused_rmsnorm_mq_rotate_awq` (derived from fused_rmsnorm_mq_rotate.hip via -DHIPFIRE_RMSNORM_AWQ) Phase 1c.
//
// Backward compatibility: dispatched ONLY when the upcoming linear
// (down_proj / w_down) carries an `awq_scale`. Otherwise the standard
// kernel runs. Both produce identical output on .hfq files without AWQ.
//
// Structure matches `fused_silu_mul_mq_rotate` exactly — only phase 1
// differs (extra AWQ divide between silu*up and signs1 gather).
//
// awq_scale: 1D FP32 buffer length K (host-side F16 → F32 conversion
// happens in the loader; see hfq.rs::load_awq_scale).
//
// HIPFIRE_IU4_SIDECAR: C2 producer path — same as the plain silu twin.
//
// HIPFIRE_SILU_HIN (IU4 sidecar only): F1-lite h-producer. Phase 1 reads
// h = SILU_MUL(g, u), already formed in FP32 by the gate/up GEMM epilogue
// (`gemm_mq4g256v2_gate_up_silu_iu4_*`), from one stream instead of forming
// it from gate + up. Everything from the AWQ divide on is unchanged, so the
// x_rot / block_i4_128 outputs are bit-identical to the gate/up twin's.

#define HIPFIRE_SILU_MQ_ROTATE_KERNEL claude_silu_rotate_q8

extern "C" __launch_bounds__(32, 16)
__global__ void HIPFIRE_SILU_MQ_ROTATE_KERNEL(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ awq_scale,   // [K], FP32
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    unsigned int* __restrict__ xq,      // claude: [N][K/256][32] uint2
    float* __restrict__ dx,             // claude: [N][K/32]
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    // Batch row: grid.y is the batch dim. For decode grid=[groups_total, 1, 1]
    // and blockIdx.y = 0 (byte-identical to pre-batching). For prefill,
    // grid=[groups_total, N, 1]; gate, up, and x_rot are [N × K] row-major.
    const long long batch_off = (long long)blockIdx.y * K;
    const float* gate_b  = gate  + batch_off;
    const float* up_b    = up    + batch_off;
          float* x_rot_b = x_rot + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // Phase 1: read gate + up, compute silu(gate)*up, AWQ divide, signs1.
    // silu(z) = z / (1 + exp(-z)). AWQ divide happens BEFORE signs1+FWHT
    // to mirror the offline weight-side pre-scaling W' = W · diag(s).
    // awq_scale is indexed in the unrotated feature basis (1D, length K).
    // The F1-lite GEMM epilogue spells SILU_MUL identically.
    #define SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))
    float g0 = gate_b[base];     float u0 = up_b[base];
    float g1 = gate_b[base + 1]; float u1 = up_b[base + 1];
    float g2 = gate_b[base + 2]; float u2 = up_b[base + 2];
    float g3 = gate_b[base + 3]; float u3 = up_b[base + 3];
    float g4 = gate_b[base + 4]; float u4 = up_b[base + 4];
    float g5 = gate_b[base + 5]; float u5 = up_b[base + 5];
    float g6 = gate_b[base + 6]; float u6 = up_b[base + 6];
    float g7 = gate_b[base + 7]; float u7 = up_b[base + 7];

    float v0 = (SILU_MUL(g0, u0) / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (SILU_MUL(g1, u1) / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (SILU_MUL(g2, u2) / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (SILU_MUL(g3, u3) / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (SILU_MUL(g4, u4) / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (SILU_MUL(g5, u5) / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (SILU_MUL(g6, u6) / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (SILU_MUL(g7, u7) / awq_scale[base + 7]) * signs1[d0 + 7];
    #undef SILU_MUL

    // Local butterfly: strides 1, 2, 4 — identical to mq_rotate_x.
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // Wave butterfly via ds_swizzle — same as mq_rotate_x.
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    float f0 = v0 * s * signs2[d0    ];
    float f1 = v1 * s * signs2[d0 + 1];
    float f2 = v2 * s * signs2[d0 + 2];
    float f3 = v3 * s * signs2[d0 + 3];
    float f4 = v4 * s * signs2[d0 + 4];
    float f5 = v5 * s * signs2[d0 + 5];
    float f6 = v6 * s * signs2[d0 + 6];
    float f7 = v7 * s * signs2[d0 + 7];
    x_rot_b[base    ] = f0;
    x_rot_b[base + 1] = f1;
    x_rot_b[base + 2] = f2;
    x_rot_b[base + 3] = f3;
    x_rot_b[base + 4] = f4;
    x_rot_b[base + 5] = f5;
    x_rot_b[base + 6] = f6;
    x_rot_b[base + 7] = f7;
    claude_q8_emit8(f0, f1, f2, f3, f4, f5, f6, f7,
                    xq, dx, K, (int)blockIdx.y, group, tid);
}
