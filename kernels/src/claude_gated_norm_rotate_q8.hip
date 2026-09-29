#include <hip/hip_runtime.h>

// claude C2: GDN output gated RMSNorm + AWQ FWHT rotate + int8 activation
// quant in ONE launch (MTP verify, gfx1100, 2..8 rows). Replaces, per GDN
// layer:
//   gated_norm_f32     grid [n_heads, N], block 32   -> dn_normed  (f32)
//   claude_rotate_q8   grid [K/256,  N], block 32   -> x_rot (f32) + xq/dx
// Bit-exact contract: normed_out, x_rot, xq and dx equal what that two-launch
// sequence stores.
//
// Phase A = gated_norm_f32 (kernels/src/gated_norm.hip) verbatim: one wave32
// per head (lane == its tid), per-lane sum of squares over i = lane, lane+32,
// .., the xor-shuffle reduction, rsqrtf(sum/head_dim + eps), then
// normed*silu(z) with z/(1+expf(-z)). The value is stored to normed_out (the
// old dn_normed store, kept so the buffer state matches) and to LDS. The two
// heads of one 256-group are the two waves of the block (head_dim == 128, the
// layout of gated_norm_mq_rotate_f16_batched_gfx1100).
//
// Phase B = claude_rotate_q8 (kernels/src/claude_rotate_q8.hip) verbatim with
// x_in read from LDS instead of global memory (an exact f32 round trip):
// (x / awq) * signs1, the three local and five ds_swizzle butterfly stages,
// v * 0.0625 * signs2, the x_rot stores and the claude_q8_emit8 epilogue
// (same expressions, 4-lane block reduction, packing and stores as
// claude_q8x_quant). Wave 0 owns the group, lane = tid, as in the original.
//
// Grid: [K/256, N, 1]. Block: [64, 1, 1]. LDS: 256 floats.

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

extern "C" __launch_bounds__(64, 8)
__global__ void claude_gated_norm_rotate_q8(
    const float* __restrict__ x,           // [N][K] GDN attention output
    const float* __restrict__ z,           // [N][K] gate
    const float* __restrict__ weight,      // [head_dim] gated-norm weight
    const float* __restrict__ awq_scale,   // [K], FP32
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ normed_out,        // [N][K] (the old gated_norm_f32 output)
    float* __restrict__ x_out,             // [N][K] rotated (the old claude_rotate_q8 x_rot)
    unsigned int* __restrict__ xq,         // [N][K/256][32] uint2
    float* __restrict__ dx,                // [N][K/32]
    int n_heads, int head_dim, float eps, int K)
{
    if (head_dim != 128) return;
    const int wave = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x;
    const int groups_total = K / 256;
    if (group >= groups_total) return;

    __shared__ float normalized[256];

    // ---- Phase A: gated_norm_f32 for head h = 2*group + wave ----
    {
        const int h = group * 2 + wave;
        const int tid = lane;
        if (h < n_heads) {
            const long long batch_off = (long long)blockIdx.y * n_heads * head_dim;
            const float* xh = x + batch_off + h * head_dim;
            const float* zh = z + batch_off + h * head_dim;
            float* oh = normed_out + batch_off + h * head_dim;
            float* lh = normalized + wave * 128;

            float sq_sum = 0.0f;
            for (int i = tid; i < head_dim; i += 32)
                sq_sum += xh[i] * xh[i];
            for (int o = 16; o > 0; o >>= 1)
                sq_sum += __shfl_xor(sq_sum, o);
            float inv_rms = rsqrtf(sq_sum / (float)head_dim + eps);

            for (int i = tid; i < head_dim; i += 32) {
                float normed = xh[i] * inv_rms * weight[i];
                float z_val = zh[i];
                float silu_z = z_val / (1.0f + expf(-z_val));
                const float val = normed * silu_z;
                oh[i] = val;
                lh[i] = val;
            }
        }
    }
    __syncthreads();
    if (wave != 0) return;

    // ---- Phase B: claude_rotate_q8 on the group, x_in from LDS ----
    const int tid = lane;
    const long long batch_off = (long long)blockIdx.y * K;
    float* x_out_b = x_out + batch_off;
    const float* x_in_b = normalized;   // group-local: index d0 + j

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float v0 = (x_in_b[d0    ] / awq_scale[base    ]) * signs1[d0    ];
    float v1 = (x_in_b[d0 + 1] / awq_scale[base + 1]) * signs1[d0 + 1];
    float v2 = (x_in_b[d0 + 2] / awq_scale[base + 2]) * signs1[d0 + 2];
    float v3 = (x_in_b[d0 + 3] / awq_scale[base + 3]) * signs1[d0 + 3];
    float v4 = (x_in_b[d0 + 4] / awq_scale[base + 4]) * signs1[d0 + 4];
    float v5 = (x_in_b[d0 + 5] / awq_scale[base + 5]) * signs1[d0 + 5];
    float v6 = (x_in_b[d0 + 6] / awq_scale[base + 6]) * signs1[d0 + 6];
    float v7 = (x_in_b[d0 + 7] / awq_scale[base + 7]) * signs1[d0 + 7];

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
