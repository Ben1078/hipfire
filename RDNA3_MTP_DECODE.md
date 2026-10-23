# RDNA3 MTP decode branch (`rdna3-mtp-decode-v8.2`)

This branch is a fork of [hipfire](https://github.com/warpfront/hipfire) at upstream commit
`fde06275` (2026-09-23). On top of that base it adds the changes that make Qwen3.8-27B
(`mq4-xts` trunk + MTP head) decode at about 100 tok/s with MTP K=3 speculative decoding on a single
Radeon RX 7900 XTX (gfx1100). Internally this build was called "v8.2".

It is published as a reference for the upstreaming discussion in warpfront/hipfire. It is **not** an
official hipfire release, and it is not a drop-in replacement for current hipfire. Upstream has moved
on a long way since `fde06275`. See [Caveats](#caveats).

## Results

### A/B against hipfire v0.4.2.1 (2026-10-23)

Full report: [warpfront/hipfire#947](https://github.com/warpfront/hipfire/issues/947).

Both engines ran on the same RX 7900 XTX (ROCm 7.2.4) with the same model file, one engine at a time,
as a resident daemon. Model: the 2026-09-23 test SKU of `qwen3.8-27b.mq4-xts` plus the registry
`qwen3.8-27b.mtp`. MTP K=3, DFlash off, q8 KV on the VMM backend. Requests were streaming chat
completions with thinking off, greedy (`temperature 0`, `repeat_penalty 1.0`) and `max_tokens 1024`.
Prompts were synthetic documents with 7 planted facts. Each value is the mean of 2 reps (the 100K row
ran once). Throughput is the daemon-reported `prefill_tok_s` / `decode_tok_s`.

| row | prompt tokens | v0.4.2.1 decode | this branch decode | v0.4.2.1 prefill | this branch prefill | v0.4.2.1 / branch prefill |
|---|---:|---:|---:|---:|---:|---:|
| 9K cold | 9,103 | 66.8 | **101.6** | 2,291 | 2,407 | 0.95x |
| 39K cold | 39,305 | 58.9 | **97.2** | 1,811 | 2,050 | 0.88x |
| 100K | 101,300 | 42.8 | **90.0** | 1,214 | 1,452 | 0.84x |

- **Decode:** 102 / 97 / 90 tok/s on this branch vs 67 / 59 / 43 tok/s on v0.4.2.1 at 9K / 39K /
  100K, i.e. 1.5x / 1.65x / 2.1x.
- **Prefill:** v0.4.2.1 reaches 0.84-0.95x of this branch's prefill rate.
- **Facts:** the two engines are equal. Both found 7/7 planted facts at 9K and 39K. At 100K both
  found 5/7 and missed the same two.
- **Where the gap comes from:** v0.4.2.1 accepts as many or more tokens per MTP window (tau 1.71 /
  1.81 / 1.79 vs 1.61 / 1.67 / 1.76). The decode gap is GPU time per verify window: 40.6 / 47.7 /
  65.2 ms on v0.4.2.1 vs 25.7 / 27.5 / 30.7 ms here. Upstream's window time grows about 0.27 ms
  per 1K tokens of context; on this branch it grows about 0.05 ms per 1K.

Settings that still differed (details in #947):

- This branch drafted over a compressed 32,768-token draft vocabulary (worth +5-10% decode in its own
  A/B, with tau unchanged).
- p_min was 0.6 here and 0.0 upstream.
- Prefill batch was 1024 here.
- The vision tower was loaded on this branch.
- `HIPFIRE_GRAPH=1` was set here.
- The runs used a resident daemon and ran one engine after the other. This is not the fresh-process
  ABBA that upstream's benchmarking methodology asks for.

### Progress inside the fork

These numbers come from the fork's own gate runs. All rows used the same harness and prompts (39K cold
/ 39K cached follow-up / 8K, greedy, K=3, p_min 0.6). They are not directly comparable with the
table above.

| step | main additions | 39K prefill tok/s | decode tok/s (39K cold / follow-up / 8K) | 100K decode |
|---|---|---:|---|---:|
| v3.1 | int8 verify GEMV v1, WMMA verify attention (attn4), FA2 > 32K, async MTP step | 1,491 | 77.9 / 81.3 / 84.3 | - |
| v4 | MTP head GEMMs, prefill kernels, launch fusions, q8x v2/v3/v4, FA2 v3 body | 1,857 | 89.2 / 89.5 / 97.2 | 78.3 |
| v6 | doorbell, GPU sampling, compressed draft vocabulary | - | 92.5 / 97.3 / 102.4 | 77.3 |
| v7 | attn5 verify attention (1..8 rows) | - | 97.5 / 96.4 / 102.5 | 86.8 |
| v8 | builder V2C prefill GEMMs, prefill keepalive | 2,023 | 97.4 / 96.4 / 102.5 | 86.7 |
| v8.2 | ports of upstream fixes (tool-call bytes, tokenizer, spec terminal states, sampler law) | 2,051 | 100.6 / 98.7 / 103.1 | 95.3 |

## What is in the branch

All new kernels and routes are gated to exact gfx1100. Other GPUs, and gfx1100 configurations that a
route does not cover, take the stock upstream path. Each switch is an environment variable. Most of
them use a `CLAUDE_` prefix, named after the AI assistant the code was written with. They do not go
through `hipfire_config` and are not listed in `docs/env-vars.md`.

| piece | main switch(es) | default |
|---|---|---|
| int8-activation small-batch GEMV for the 2..8-row MQ4G256V2 verify projections (`claude_q8x`, `v_dot4_i32_iu8`, one launch per projection group, quant fused into the producers, 2..4-row verify lm_head) | `CLAUDE_XBATCH`, `CLAUDE_Q8X_V2`, `CLAUDE_Q8_FUSE`, `CLAUDE_LMHEAD_Q8X` | on |
| WMMA split-K verify attention for 1..8 rows (`claude_attn4` -> `claude_attn5`), also used for the MTP draft head; 8-warp split-K reduce | `CLAUDE_ATTN5`, `CLAUDE_ATTN4`, `CLAUDE_DRAFT_ATTN4/5`, `CLAUDE_ATTN_REDUCE` | on |
| two-stage argmax / top-2 + logsumexp over the vocabulary | `CLAUDE_FAST_REDUCE`, `CLAUDE_TOP2_LDS` | on |
| stream-ordered MTP step, doorbell readbacks, on-device sampling for sampled MTP, bit-exact launch fusions (C1-C6, C5) | `CLAUDE_MTP_ASYNC`, `CLAUDE_MTP_DOORBELL`, `CLAUDE_MTP_GPU_SAMPLE`, `CLAUDE_KV_PAIR`, ... | on |
| MTP head: batched MQ4 GEMMs instead of the scalar kernel, KV-only fill without Q, last-token logits only on the final fill chunk | `CLAUDE_MTP_HEAD_GEMM`, `CLAUDE_MTP_FILL_SKIP_Q` | on |
| compressed MTP draft vocabulary: `claude_mtp_draft_vocab` tool (hipfire-quantize) that byte-slices the trunk lm_head, loader override, fast compressed draft step | `CLAUDE_MTP_SIDECAR=<file.mtp>`, `CLAUDE_CVS_FAST` | opt-in (needs a sidecar you build) |
| MTP prompt fill: route fix (hidden capture was treated as verify) and the (token p, hidden p-1) pairing | `CLAUDE_HIDDEN_PREFILL_LEGACY`, `CLAUDE_MTP_FILL_LEGACY` (restore old) | on |
| accepted-row MTP KV refresh; no host sync after DeltaNet snapshot copies | `CLAUDE_MTP_ACCEPT_REFRESH=1`, `CLAUDE_NO_BULK_SYNC=1` | opt-in (used in the A/B) |
| FA2 prefill attention above 32K context, plus a faster bit-exact FA2 body | `CLAUDE_FA2_MAX_CTX`, `CLAUDE_FA2_VARIANT` | on |
| prefill: V2D iu4 GEMM, grouped SET launches, token-parallel conv1d, quad-row GDN, upstream builder V2C code object, post-kernel keepalive | `CLAUDE_IU4_V2D`, `CLAUDE_IU4_GROUP`, `CLAUDE_CONV_TPAR`, `CLAUDE_GDN_QUADROW`, `CLAUDE_IU4_PM`, `CLAUDE_PREFILL_KEEPALIVE` | on |
| GpuPool drain-and-retry on out-of-memory | - | on |
| max_tokens clamped to the KV room left, so large budgets keep the cache and the MTP drafter | `CLAUDE_MAX_TOKENS_CLAMP`, `CLAUDE_CLAMP_MARGIN`, `CLAUDE_CTX_CLAMP_MIN` | on |
| vision (Qwen3.5/3.8 VL): VMM page-fault fix, batched VL prefill, image turns through the text/MTP path | `CLAUDE_VL_V2`, `CLAUDE_VL_FAST_PREFILL`, `CLAUDE_VL_MTP` | on |
| experiments left off | `CLAUDE_VERIFY_PRESTAGE=1`, `CLAUDE_GDN_CHUNK_ADMIT=1` (not bit-exact), `CLAUDE_ATTN_GQA=1` | off |

The branch also carries a few upstream commits and ports. Three commits by Kaden Schutt are
cherry-picked from upstream: the gfx1100 prefill levers, the exact-gfx1100 FA2 variant, and the exact
tool-call argument bytes. The last four code commits port upstream fixes: HF tokenizer parity, emitting
each speculative token once, MTP/DFlash terminal states, and the AR-equivalent sampled truncation law.

The configuration used for the A/B was:

```
HIPFIRE_PREFILL_MAX_BATCH=1024 HIPFIRE_GRAPH=1 HIPFIRE_PAGE_EVICTION=1 HIPFIRE_QWEN_PREAD_LOAD=1 \
CLAUDE_MTP_ACCEPT_REFRESH=1 CLAUDE_NO_BULK_SYNC=1 CLAUDE_MTP_SIDECAR=<compressed .mtp> \
hipfire serve --model <qwen3.8-27b.mq4-xts> --kv-backend vmm --kv-mode q8
# config.toml: speculation mtp, mtp_k=3, dflash off
```

The build is the usual hipfire build: `cargo build --release -p hipfire-cli -p hipfire-daemon`, then
`scripts/compile-kernels.sh gfx1100` (otherwise the kernels JIT on first use).

To build the compressed draft vocabulary, run `claude_mtp_draft_vocab rank` with a token-frequency
corpus of your own (JSONL, `{"cat", "text"}` per document), then `claude_mtp_draft_vocab build`. The
corpus and the 32K sidecar used for the measurements are not part of this branch.

## What upstream has adopted since

As of 2026-10-23 (upstream `master` = v0.4.2.1, plus `beta`), upstream has its own versions of:

- the MTP prompt-fill workload route (v0.4.0; opt-out `HIPFIRE_MTP_OWN_PREFILL=1`);
- the MTP fill off-by-one, i.e. the (token p, hidden p-1) head pairing (v0.4.0);
- fast FA2 prefill attention above 32K on gfx1100 (up to 262,144 context);
- GpuPool drain-and-retry on out-of-memory (v0.4.2);
- max_tokens fitting when max_tokens is omitted (an explicit oversized value is refused by design);
- the tool-call, tokenizer and speculative-emission fixes, which this branch ported from upstream in
  the first place.

Not in upstream:

- **int8-activation verify GEMV.** Upstream runs 4-row MTP verify projections on f16 WMMA tiles. This
  is the largest single piece of the gap: about 10 ms per MTP window at any context. It is proposed
  in [warpfront/hipfire#946](https://github.com/warpfront/hipfire/issues/946). A port onto upstream
  `beta` exists as an opt-in route on branch
  [`feature/gfx1100-q8x-verify-gemv`](https://github.com/Ben1078/hipfire/tree/feature/gfx1100-q8x-verify-gemv).
  In a fresh-process ABBA it gives +10-13% median decode.
- **WMMA split-K verify attention.** Upstream's gfx1100 `VerifyAttn` twins keep the scalar tile's
  arithmetic. This is the largest piece at long context: about 20 ms per window at 100K.
- **Compressed draft vocabulary for the 27B MTP head.** Upstream's loader supports the layout, but its
  27B sidecar is full-vocabulary.
- **Decode-path work:** the stream-ordered MTP step, doorbell readbacks, GPU sampling, launch fusions
  and the prefill keepalive.

## Caveats

- **Tested on one machine only:** one RX 7900 XTX (gfx1100, 24 GB) on Linux with ROCm 7.2.4, and
  essentially one model (Qwen3.8-27B mq4 + MTP head). Other GPUs, other models and other ROCm
  versions are untested. The gfx1100 routes decline and fall back to stock paths outside the shapes
  they were written for, but nothing beyond that has been validated.
- **AI-agent-assisted:** most of the code and the investigation were done with an AI coding
  assistant (Claude). The `Co-Authored-By` trailers say which commits. The author validated the
  results on the hardware above (standalone parity checks, byte-identity A/Bs where a change is meant
  to be exact, end-to-end runs).
- **Old base:** the branch is based on `fde06275` (2026-09-23), thousands of commits behind
  current upstream `master` and `beta`. It is not meant as a drop-in or a maintained alternative to
  hipfire. Anything worth keeping should go upstream as separate, flagged PRs; the
  `feature/gfx1100-q8x-verify-gemv` port is the first.
- **Numerics:** several default-on routes are not byte-identical to upstream. The int8 activations
  carry q8_1-class error, and the WMMA verify attention uses f16 operands with f32 accumulation
  (FA2-class). These were checked against reference implementations and end-to-end outputs, not
  against upstream's KLD gates. Exact (bit-identical) changes are marked as such in their commits.
  MTP K>3 was not validated.
- **Not included:** local benchmark harnesses, profiling logs, the draft-vocabulary ranking corpus and
  the compressed sidecar.

## License

This branch keeps hipfire's license unchanged: Apache-2.0, with the MIT-licensed files identified by
their SPDX headers. See `LICENSE`, `LICENSE-APACHE`, `LICENSE-MIT` and `NOTICE`. The changes in this
branch are offered under the same terms (Apache-2.0).

## Credits

hipfire is written by Kaden Schutt and the hipfire contributors. Everything here builds directly on
their engine: the kernels, the Qwen3.5/3.8 MTP and DFlash speculative decoding, the VMM KV backend,
and the builder code objects. Three commits in this branch are their upstream commits, and the v8.2
fixes are ports of upstream commits. Thanks to the hipfire authors for an engine that made this work
possible.
