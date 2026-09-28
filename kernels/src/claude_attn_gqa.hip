// claude: speculative-verify attention for a q8_0 KV cache, head_dim 256 (v3).
// One block per (kv head, T-token tile) covers all G q-heads of the group and
// up to RMAX verify rows (C = G*RMAX query vectors): each KV byte is read once
// per group. fp16 operands with f32 accumulation via v_dot2_f32_f16.
//   Phase B (scores): 8 lanes per token, lane j owns 32-dim block j; raw int8 K
//     -> fp16 (exact), 16 fdot2 per query vector, block scale applied once.
//   Phase C (softmax per query vector, log2 domain) -> P as fp16 token pairs.
//   Phase D (P.V): thread d owns dim d; fdot2 over (token t, t+1) pairs.
// Partials: stock layout [row][head][tile][2+256], ln-domain max.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef CLAUDE_G
#define CLAUDE_G 6
#endif
#ifndef CLAUDE_RMAX
#define CLAUDE_RMAX 4
#endif
#ifndef CLAUDE_T
#define CLAUDE_T 64
#endif
#ifndef CLAUDE_NT
#define CLAUDE_NT 4
#endif
#ifndef CLAUDE_ATTN_GQA_KERNEL
#define CLAUDE_ATTN_GQA_KERNEL claude_attn_gqa
#endif
#define CLAUDE_C (CLAUDE_G * CLAUDE_RMAX)
#define CLAUDE_QS 20             // half2 per 32-dim block row (16 + pad)
#define CLAUDE_PS (CLAUDE_C + 4) // half2 per token-pair row

typedef _Float16 claude_h2 __attribute__((ext_vector_type(2)));

// Two signed int8 (low bytes of a and b) -> exact fp16 pair, via the
// 0x6400|(x+128) trick (fp16 1024+x+128) minus 1152.
__device__ __forceinline__ claude_h2 claude_i8x2_to_h2(unsigned int lo_byte, unsigned int hi_byte) {
    const unsigned int bits = (((lo_byte ^ 0x80u) & 0xFFu) | (((hi_byte ^ 0x80u) & 0xFFu) << 16)) | 0x64006400u;
    claude_h2 h = __builtin_bit_cast(claude_h2, bits);
    const claude_h2 bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    return h - bias;
}

extern "C" __launch_bounds__(256) __global__ void CLAUDE_ATTN_GQA_KERNEL(
    const float* __restrict__ q,
    const unsigned char* __restrict__ kc,
    const unsigned char* __restrict__ vc,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads, int n_kv_heads, float scale_log2, int max_tiles, int rows)
{
    __shared__ claude_h2 q_s[CLAUDE_C][8][CLAUDE_QS];
    __shared__ float s_s[CLAUDE_T][CLAUDE_C + 1];
    __shared__ claude_h2 p_s[CLAUDE_T / 2][CLAUDE_PS];
    __shared__ float m_s[CLAUDE_C];
    __shared__ float l_s[CLAUDE_C];
    __shared__ float corr_s[CLAUDE_C];

    const int kvh = blockIdx.x, tile = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int seq[CLAUDE_RMAX];
    int seq_max = 0;
#pragma unroll
    for (int r = 0; r < CLAUDE_RMAX; r++) {
        seq[r] = r < rows ? positions[r] + 1 : 0;
        seq_max = max(seq_max, seq[r]);
    }
    const int tile0 = tile * CLAUDE_T * CLAUDE_NT;
    if (tile0 >= seq_max) return;
    const int per_pos = n_kv_heads * 8 * 34;
    const int q_dim = n_heads * 256;

    // Q -> fp16 pairs: q_s[c][block][pair] = (q[2p], q[2p+1]) within the block.
    for (int i = tid; i < CLAUDE_C * 128; i += 256) {
        const int c = i >> 7, pr = i & 127;
        const int g = c / CLAUDE_RMAX, r = c % CLAUDE_RMAX;
        const int rr = r < rows ? r : 0;
        const float* qp = q + (long long)rr * q_dim + (kvh * CLAUDE_G + g) * 256 + 2 * pr;
        const claude_h2 v = {(_Float16)qp[0], (_Float16)qp[1]};
        q_s[c][pr >> 4][pr & 15] = v;
    }
    __syncthreads();

    if (tid < CLAUDE_C) { m_s[tid] = -INFINITY; l_s[tid] = 0.0f; }
    const int d = tid;
    const int vj = d >> 5, vi = d & 31;
    float acc[CLAUDE_C];
#pragma unroll
    for (int c = 0; c < CLAUDE_C; c++) acc[c] = 0.0f;
    const int grp = tid >> 3;
    const int j8 = tid & 7;
    for (int st = 0; st < CLAUDE_NT; st++) {
    const int t0 = tile0 + st * CLAUDE_T;
    if (t0 >= seq_max) break;
    __syncthreads();
    // ---- Phase B: scores ------------------------------------------------
    for (int tb = 0; tb < CLAUDE_T; tb += 32) {
        const int tl = tb + grp;
        const int t = t0 + tl;
        float part[CLAUDE_C];
#pragma unroll
        for (int c = 0; c < CLAUDE_C; c++) part[c] = 0.0f;
#ifndef CLAUDE_SKIP_B
        if (t < seq_max) {
#else
        if (false) {
#endif
            const unsigned char* kb = kc + (long long)t * per_pos + (kvh * 8 + j8) * 34;
            const float sc = (float)*((const _Float16*)kb);
            const unsigned short* k16 = (const unsigned short*)(kb + 2);
            claude_h2 kh[16];
#pragma unroll
            for (int w = 0; w < 16; w++) {
                const unsigned int v = k16[w];
                kh[w] = claude_i8x2_to_h2(v & 0xFFu, v >> 8);
            }
#pragma unroll
            for (int c = 0; c < CLAUDE_C; c++) {
                const claude_h2* qr = &q_s[c][j8][0];
                float acc = 0.0f;
#pragma unroll
                for (int w = 0; w < 16; w++) acc = __builtin_amdgcn_fdot2(qr[w], kh[w], acc, false);
                part[c] = acc * sc;
            }
        }
#pragma unroll
        for (int c = 0; c < CLAUDE_C; c++) {
            float v = part[c];
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 4);
            part[c] = v;
        }
#pragma unroll
        for (int c = 0; c < CLAUDE_C; c++) {
            if ((c & 7) == j8) {
                const int r = c % CLAUDE_RMAX;
                s_s[tl][c] = (t < seq[r]) ? part[c] * scale_log2 : -INFINITY;
            }
        }
    }
    __syncthreads();

    // ---- Phase C: softmax stats per query vector; P -> fp16 pairs --------
    for (int c = warp; c < CLAUDE_C; c += 8) {
        float mt = -INFINITY;
        for (int i = lane; i < CLAUDE_T; i += 32) mt = fmaxf(mt, s_s[i][c]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor(mt, off));
        const float m_old = m_s[c];
        const float m_new = fmaxf(m_old, mt);
        float sum = 0.0f;
        for (int i = lane; i < CLAUDE_T; i += 32) {
            const float p = (m_new == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(s_s[i][c] - m_new);
            s_s[i][c] = p;
            sum += p;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);
        if (lane == 0) {
            const float corr = (m_old == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(m_old - m_new);
            corr_s[c] = corr;
            l_s[c] = l_s[c] * corr + sum;
            m_s[c] = m_new;
        }
    }
    __syncthreads();
    for (int i = tid; i < (CLAUDE_T / 2) * CLAUDE_C; i += 256) {
        const int tp = i / CLAUDE_C, c = i - tp * CLAUDE_C;
        const claude_h2 v = {(_Float16)s_s[2 * tp][c], (_Float16)s_s[2 * tp + 1][c]};
        p_s[tp][c] = v;
    }
    __syncthreads();

    // ---- Phase D: P.V, thread d owns output dim d -----------------------
#pragma unroll
    for (int c = 0; c < CLAUDE_C; c++) acc[c] *= corr_s[c];
    const int tl_end = min(CLAUDE_T, seq_max - t0);
    const unsigned char* vbase = vc + (long long)t0 * per_pos + (kvh * 8 + vj) * 34;
#ifdef CLAUDE_SKIP_D
    for (int tp = 0; tp < 0; tp += 4) {
#else
    for (int tp = 0; 2 * tp < tl_end; tp += 4) {
#endif
        // four token pairs per iteration (8 tokens): all V loads issued first
        claude_h2 vv[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float a = 0.0f, b = 0.0f;
            const int ta = 2 * (tp + u), tb = ta + 1;
            if (ta < tl_end) {
                const unsigned char* va = vbase + (long long)ta * per_pos;
                a = (float)*((const _Float16*)va) * (float)(signed char)va[2 + vi];
            }
            if (tb < tl_end) {
                const unsigned char* vb2 = vbase + (long long)tb * per_pos;
                b = (float)*((const _Float16*)vb2) * (float)(signed char)vb2[2 + vi];
            }
            const claude_h2 hv = {(_Float16)a, (_Float16)b};
            vv[u] = hv;
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const claude_h2* pr = &p_s[tp + u][0];
#pragma unroll
            for (int c = 0; c < CLAUDE_C; c++) acc[c] = __builtin_amdgcn_fdot2(pr[c], vv[u], acc[c], false);
        }
    }

    }  // sub-tiles
    __syncthreads();

    // ---- Phase E: partials ------------------------------------------------
#pragma unroll
    for (int c = 0; c < CLAUDE_C; c++) {
        const int g = c / CLAUDE_RMAX, r = c % CLAUDE_RMAX;
        if (r >= rows || tile0 >= seq[r]) continue;
        float* p = partials + (((long long)r * n_heads + kvh * CLAUDE_G + g) * max_tiles + tile) * 258;
        if (tid == 0) { p[0] = m_s[c] * 0.69314718f; p[1] = l_s[c]; }
        p[2 + d] = acc[c];
    }
}
