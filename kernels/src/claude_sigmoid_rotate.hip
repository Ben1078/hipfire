#include <hip/hip_runtime.h>

// claude C3: attention output gate + FWHT rotate (+ int8 quant) in ONE launch.
// Replaces
//   sigmoid_mul_f32(attn, gate)            in place: attn[i] = attn[i] * (1/(1+expf(-gate[i])))
//   + one of
//   claude_rotate_q8(attn, awq)            (CLAUDE_SR_AWQ=1, CLAUDE_SR_Q8=1: MTP verify FA wo input)
//   mq_rotate_x(attn)                      (CLAUDE_SR_AWQ=0, CLAUDE_SR_Q8=0: MTP draft head wo input)
//   rotate_x_mq_awq(attn, awq)             (CLAUDE_SR_AWQ=1, CLAUDE_SR_Q8=0)
// Bit-exact contract: attn (rows < N), x_rot and (q8) xq / dx equal what the
// two-launch sequence stores. Every element's gate is computed by the same
// statements as sigmoid_mul_f32 (`s = 1.0f / (1.0f + expf(-g))`, `out * s`)
// and written back in place (the old store, kept so the attn buffer state
// matches); the rotate is the original kernel's body verbatim with x_in
// taken from those registers (an exact f32 round trip): ((x / awq) or x) *
// signs1, three local and five ds_swizzle butterfly stages, v * 0.0625 *
// signs2, the x_rot stores and (q8) the claude_q8_emit8 epilogue.
//
// NOTE: sigmoid_mul_f32 runs over the whole scratch tensor (out.numel()); this
// kernel covers rows 0..N-1 only. Rows >= N of the verify scratch are dead
// (every attend writes rows 0..N-1 before any read).
//
// Grid: [K/256, N, 1]. Block: [32, 1, 1] (one group per wave, lane = tid).

#ifndef CLAUDE_SR_KERNEL
#define CLAUDE_SR_KERNEL claude_sigmoid_rotate_q8
#endif
#ifndef CLAUDE_SR_AWQ
#define CLAUDE_SR_AWQ 1
#endif
#ifndef CLAUDE_SR_Q8
#define CLAUDE_SR_Q8 1
#endif

#if CLAUDE_SR_Q8
// claude: int8 activation quant of one lane's 8 consecutive values (elements
// g*256 + t*8 .. +7 of row b), verbatim from claude_rotate_q8.hip.
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
#endif

// sigmoid_mul_f32's per-element statements; `o` is the value stored back.
#define CLAUDE_SIGMOID_MUL(o, idx) do { \
    float g = gate_b[(idx)]; \
    float s = 1.0f / (1.0f + expf(-g)); \
    (o) = attn_b[(idx)] * s; \
    attn_b[(idx)] = (o); \
} while (0)

extern "C" __launch_bounds__(32, 16)
__global__ void CLAUDE_SR_KERNEL(
    float* __restrict__ attn,              // [N][K] in/out
    const float* __restrict__ gate,        // [N][K]
#if CLAUDE_SR_AWQ
    const float* __restrict__ awq_scale,   // [K], FP32
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_out,             // [N][K] rotated
#if CLAUDE_SR_Q8
    unsigned int* __restrict__ xq,         // [N][K/256][32] uint2
    float* __restrict__ dx,                // [N][K/32]
#endif
    int K
) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    const long long batch_off = (long long)blockIdx.y * K;
    float* attn_b = attn + batch_off;
    const float* gate_b = gate + batch_off;
          float* x_out_b = x_out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float x0, x1, x2, x3, x4, x5, x6, x7;
    CLAUDE_SIGMOID_MUL(x0, base    );
    CLAUDE_SIGMOID_MUL(x1, base + 1);
    CLAUDE_SIGMOID_MUL(x2, base + 2);
    CLAUDE_SIGMOID_MUL(x3, base + 3);
    CLAUDE_SIGMOID_MUL(x4, base + 4);
    CLAUDE_SIGMOID_MUL(x5, base + 5);
    CLAUDE_SIGMOID_MUL(x6, base + 6);
    CLAUDE_SIGMOID_MUL(x7, base + 7);

#if CLAUDE_SR_AWQ
    // rotate_x_mq_awq / claude_rotate_q8: AWQ divide + signs1 gather.
    float v0 = (x0 / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x1 / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x2 / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x3 / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x4 / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x5 / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x6 / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x7 / awq_scale[base + 7]) * signs1[d0 + 7];
#else
    // mq_rotate_x: signs1 gather.
    float v0 = x0 * signs1[d0];
    float v1 = x1 * signs1[d0 + 1];
    float v2 = x2 * signs1[d0 + 2];
    float v3 = x3 * signs1[d0 + 3];
    float v4 = x4 * signs1[d0 + 4];
    float v5 = x5 * signs1[d0 + 5];
    float v6 = x6 * signs1[d0 + 6];
    float v7 = x7 * signs1[d0 + 7];
#endif

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
#if CLAUDE_SR_Q8
    claude_q8_emit8(f0, f1, f2, f3, f4, f5, f6, f7,
                    xq, dx, K, (int)blockIdx.y, group, tid);
#endif
}
#undef CLAUDE_SIGMOID_MUL
