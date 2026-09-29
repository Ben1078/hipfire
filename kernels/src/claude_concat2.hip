// claude r2 C5: [a | b] row concat in one launch (the MTP KV-only forward's
// eh_proj input), replacing two copy_rows_strided_f32 launches. Pure copy.
// Grid: [ceil(2*dim / 256), n, 1]. Block: [256, 1, 1].
#include <hip/hip_runtime.h>

extern "C" __global__ void claude_concat2_rows_f32(
    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out,
    int n, int dim)
{
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n || c >= 2 * dim) return;
    out[(long long)r * 2 * dim + c] = c < dim ? a[(long long)r * dim + c] : b[(long long)r * dim + (c - dim)];
}
