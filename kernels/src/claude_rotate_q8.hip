#include <hip/hip_runtime.h>

// claude: rotate_x_mq_awq + int8 activation quant (claude_q8x).
//
// Body: kernels/src/rotate_x_mq_awq.hip (the runtime source, no conditionals)
// with every arithmetic expression unchanged; the final `v * s * signs2`
// products are bound to f0..f7 before the (unchanged) x_out stores so the
// claude_q8_emit8 epilogue (b = blockIdx.y, g = group = blockIdx.x, t = tid)
// quantizes exactly the stored values. Additions: `xq` / `dx` arguments.

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

// AWQ-aware variant of `mq_rotate_x` (in `gemv_mq4g256.hip`).
//
// Phase A Stage A — F2 follow-up. The original `mq_rotate_x` is used as the
// input-rotation stage before the post-projection linear (o_proj / out_proj),
// which Stage A's AWQ whitelist excluded because no AWQ-aware rotate existed
// for that path. This kernel completes the AWQ math `(W·s) · (x/s) = W·x`
// by dividing each input element by `awq_scale[i]` BEFORE the FWHT — exact
// mirror of `fused_rmsnorm_mq_rotate_awq` (derived from fused_rmsnorm_mq_rotate.hip via -DHIPFIRE_RMSNORM_AWQ) Phase 1c.
//
// Backward compatibility: this kernel is dispatched ONLY when the upcoming
// linear carries an `awq_scale` GpuTensor. Otherwise the standard
// `mq_rotate_x` runs. Both produce identical output on .hfq files that
// don't carry AWQ scales.
//
// Layout: identical to `mq_rotate_x` aside from the extra divide. See that
// kernel's preamble for FWHT details.
//
// awq_scale: 1D FP32 buffer length K (host-side F16 → F32 conversion happens
// in the loader; see hfq.rs::load_awq_scale).

extern "C" __launch_bounds__(32, 16)
__global__ void claude_rotate_q8(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ awq_scale,   // [K], FP32
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    unsigned int* __restrict__ xq,      // claude: [N][K/256][32] uint2
    float* __restrict__ dx,             // claude: [N][K/32]
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    const float* x_in_b  = x_in  + batch_off;
          float* x_out_b = x_out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    // AWQ divide + signs1 gather. The divide happens BEFORE the FWHT to
    // mirror the offline weight side `W' = W · diag(s)` — quantize-time
    // pre-scaling cancels the runtime per-channel divide so that
    // `(W·s) · (x/s) = W·x`. awq_scale is 1D length-K, indexed in the
    // unrotated x basis (not FWHT-rotated).
    float v0 = (x_in_b[base    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x_in_b[base + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x_in_b[base + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x_in_b[base + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x_in_b[base + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x_in_b[base + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x_in_b[base + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x_in_b[base + 7] / awq_scale[base + 7]) * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4
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

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle.
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
    const float f0 = v0 * s * signs2[d0    ];
    const float f1 = v1 * s * signs2[d0 + 1];
    const float f2 = v2 * s * signs2[d0 + 2];
    const float f3 = v3 * s * signs2[d0 + 3];
    const float f4 = v4 * s * signs2[d0 + 4];
    const float f5 = v5 * s * signs2[d0 + 5];
    const float f6 = v6 * s * signs2[d0 + 6];
    const float f7 = v7 * s * signs2[d0 + 7];
    x_out_b[base    ] = f0;
    x_out_b[base + 1] = f1;
    x_out_b[base + 2] = f2;
    x_out_b[base + 3] = f3;
    x_out_b[base + 4] = f4;
    x_out_b[base + 5] = f5;
    x_out_b[base + 6] = f6;
    x_out_b[base + 7] = f7;
    claude_q8_emit8(f0, f1, f2, f3, f4, f5, f6, f7,
                    xq, dx, K, (int)blockIdx.y, group, tid);
}
