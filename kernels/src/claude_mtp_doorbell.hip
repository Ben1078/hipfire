// claude r2: MTP host-sync helpers (CLAUDE_MTP_DOORBELL).
//
// claude_mtp_publish: the "doorbell" readback. One thread copies n_words
// 32-bit words from device memory `src` into a pinned, device-mapped,
// fine-grained host slot (slot[1..=n_words]) and then, after a system-scope
// release fence, stores the sequence number into slot[0]. The host spins on
// slot[0] == seq and then reads the payload: no blit dispatch, no null-stream
// barrier, no hipMemcpy wake-up. All host-slot stores are system-scope
// atomics so they bypass/write through the GPU caches; program order on one
// thread plus the release (and the explicit __threadfence_system) makes every
// payload word visible before the sequence word.
// Optional device-side extras, done before the fence (stream-ordered for the
// kernels that follow):
//   pos_dst != 0 : *pos_dst = pos_val    (the next draft step's position)
//   tok_dst != 0 : *tok_dst = tok_map ? tok_map[src[tok_word]] : src[tok_word]
//                  (the candidate token staged into the verify token list)
//
// claude_mtp_stage: per-cycle scalars from kernel arguments, one launch
// (replaces hipStreamWriteValue32 at ~6.6 us GPU each and the blocking H2D
// uploads of the verify prologue):
//   ramp_dst != 0 : ramp_dst[i] = ramp_base + i, i < n_ramp  (verify positions)
//   tok_dst  != 0 : *tok_dst = tok_val                      (verify tokens[0])
//   pos_dst  != 0 : *pos_dst = pos_val                      (draft step-0 position)
#include <hip/hip_runtime.h>

// The host slot address is passed as an integer, not a pointer argument: the
// HIP runtime inspects pointer arguments and a pinned-host pointer there makes
// it treat the dispatch as touching host memory (measured: ~30 us of extra
// GPU idle before the NEXT batch's second kernel on gfx1100 / ROCm 7.2).
extern "C" __global__ void claude_mtp_publish(
    const unsigned int* __restrict__ src, int n_words,
    unsigned long long slot_addr, unsigned int seq,
    int* __restrict__ pos_dst, int pos_val,
    const unsigned int* __restrict__ tok_map, int tok_word, int* __restrict__ tok_dst)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int* slot = (unsigned int*)slot_addr;
    for (int i = 0; i < n_words; i++) {
        const unsigned int v = src[i];
        __hip_atomic_store(slot + 1 + i, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (pos_dst) *pos_dst = pos_val;
    if (tok_dst) {
        unsigned int id = src[tok_word];
        if (tok_map) id = tok_map[id];
        *tok_dst = (int)id;
    }
    __threadfence_system();
    __hip_atomic_store(slot, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

extern "C" __global__ void claude_mtp_stage(
    int* __restrict__ ramp_dst, int ramp_base, int n_ramp,
    int* __restrict__ tok_dst, int tok_val,
    int* __restrict__ pos_dst, int pos_val)
{
    const int t = threadIdx.x;
    if (ramp_dst && t < n_ramp) ramp_dst[t] = ramp_base + t;
    if (t == 0) {
        if (tok_dst) *tok_dst = tok_val;
        if (pos_dst) *pos_dst = pos_val;
    }
}
