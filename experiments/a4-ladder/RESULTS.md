# Fused C2 A4 ladder study — closing report

## Status (supersedes the recommendation below)

**The prefill-timing section of this report is STALE: it was measured with `bench_qwen35_mq4`, q8 KV — not product.** `bench_qwen35_mq4` is not the product path and these timings used q8/q8 KV, not the product-default `kv_cache = auto` (fp8 on gfx1201). The old R9700 pp8192 C2 control (~3086 tok/s) is therefore confounded against the product rate (~5100 tok/s). Do not cite any tok/s, throughput-cost percentage, or the performance-based recommendation in this file as current. Product timing is being redone with `hipfire bench <canonical 27B> --backend noslots --workload stateless`, product-default KV; this file will be updated from that run.

**Unchanged:** the KLD results (canonical 27B MQ4-XTS, `eval_hipfire`, pinned fp8/q8) stand, the default stays `2`, and any default flip is non-exact and requires Main's word.

## Original recommendation (stale; timing basis is not product)

~~**Keep C2 as the default. Do not flip to the existing four- or eight-candidate implementations.**~~ Retained only as history: it rested on the stale `bench_qwen35_mq4`/q8 timings below and is not a current recommendation.

Fused8 improves R9700 WT2/code24 KLD by **−0.000999 / −0.002220**, recovering only **42.6%** of WMM4's WT2 standalone gain (115.7% on code24). The existing four-candidate subset worsens both KLD means. (The previously reported 15.02% / 12.52% R9700 and 4.31% / 2.71% Halo prefill costs came from the stale timing and are not current.)

The branch leaves all defaults unchanged. This is non-exact and nothing has been merged or re-pinned; any future default flip still requires Main's word. The conclusion is about the existing generic widened emitters, not an unmeasured redesigned multi-candidate one-pass emitter.

## Branch and switch

- Published branch: `research/a4-fused-ladder`, based on origin/beta `4b68ffde8`.
- Switch implementation: `bec0e6d52`; measured source/build commit: `d35ae715dbaeb9240b148a3f739a343838ba6bf3`.
- Worktree: `/home/kaden/qcal/release-0.4.3/a4-fused-ladder` on hiptrx; clean detached measured commit on hipx, fetched through Git (not a copied source tree).
- New gfx1201 switch: `HIPFIRE_G12_A4_CANDIDATES=4|8` / TOML `kernel.g12_a4_candidates`; default remains `2`. Accepted values are `1,2,4,8`.
- Halo uses existing `HIPFIRE_GFX11_A4_CANDIDATES=4|8`; default remains `2`.
- Existing `HIPFIRE_G12_A4C2=0` kill switch is unchanged. Standalone quantization remains eight candidates.
- Grid index i means multiplier `1+i/7` of `(amax/7)*0.5`. Tested compact4 is the pre-existing `{0,2,5,7}` subset, **not** an exhaustively selected optimal compact ladder.

## Quality — canonical 27B MQ4-XTS, R9700

Model: `qwen3.8-27b.mq4-xts`, 14987185152 bytes, SHA256 `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`. `scripts/check_fixture.sh` verified the fixture on both hosts. WT2 reference SHA256 `8c545178fb43647499c4c7bdd33178c24fd402615c641add9ed51a0c9a43234a`; code24 `43ab7591f2acf6b5b4058b77253109230d96591be02469ddef55235102ec7582`.

`eval_hipfire --scoring-mode prefill --max-chunks 24 --kv-mode fp8 --kv-v q8`, `HIPFIRE_GRAPH=0`, `HIPFIRE_NORMALIZE_PROMPT=0`, fresh HOME and empty kernel cache per corpus/process. Fabric correctness P0, caps ≤600 seconds. First matrix on card B (j1098); independent full8 repeat and standalone diagnostic on card E (j1105).

| Arm | WT2 KLD | code24 KLD | Δ WT2 vs default | Δ code24 vs default |
|---|---:|---:|---:|---:|
| C2 default {5,7} | 0.069571107 | 0.052118753 | +0.000000000 | +0.000000000 |
| Fused4 {0,2,5,7} | 0.069768324 | 0.053332816 | +0.000197217 | +0.001214064 |
| Fused8 {0..7} | 0.068572044 | 0.049899241 | -0.000999063 | -0.002219512 |
| Standalone8 diagnostic | 0.067223761 | 0.050200889 | -0.002347346 | -0.001917864 |

- Unset control and explicit `=2` reproduce both AGENTS.md byte pins: WT2 `ff88c49910be618a98a917afc40d4b01ceeb7c6299c62f1c553444c11b402841`; code24 `9b0616316f50e9a654690d06d594d57466430d0a4a67041ec4ff0128d830910e`.
- Fused8 repeats byte-for-byte across cards B/E: WT2 `236ef7ca8f016a31b1006fb08fd6e7c61936d0013dafe6fdd52b8ab345d882b9`; code24 `83802661d39c727d905346a0fbf76888154b571e22c612893de4f85916592d61`.
- Both widened candidates satisfy the +0.010 one-sided KLD cap, but compact4 is not a quality improvement.
- The standalone diagnostic reproduces WMM4's KLD means and sequence hashes. Its command disables producer fusion, SiLU fusion, FP8 stream, and CB verify IU4. Therefore the standalone-vs-fused contrast is **not only a candidate-ladder intervention**; the remaining WT2 gap is not assigned to a specific route difference here.
- No Halo KLD claim is made; the quality study is the canonical R9700 pin comparison.

## Prefill throughput — exclusive P1 ABBA (STALE: `bench_qwen35_mq4`, q8 KV: not product)

> **STALE / not product.** Every number in this section and its table came from `bench_qwen35_mq4` with q8/q8 KV. Superseded by product `hipfire bench` timing (pending). Kept only as provenance of the old run.

Every comparison is two ABBA blocks in one exclusive P1 fabric hold, four fresh processes per arm, three prefill passes per process. Discard pass 1 (JIT); retain two warm samples. Report the geometric mean of per-process medians. Prompt IDs are literally `0..pp-1`; generation and decode warmup are disabled. GPU DPM warmup is three seconds outside the timed forward call. The timed region is production `forward_prefill_batch` plus device synchronization.

**Timing KV is q8/q8 on both architectures**, because `bench_qwen35_mq4` does not support fp8 KV. Quality uses the pinned fp8/q8 setting above. Thus these timing numbers must not be presented as fp8-KV timings. All arms use identical KV and prompt settings within each comparison. Control rates are paired per hold, not pooled across holds.

| GPU | Candidates | pp | C2 tok/s | Widened tok/s | Δ tok/s | Two ABBA block deltas | Process spread C2 / widened | Variability note |
|---|---:|---:|---:|---:|---:|---|---|---|
| R9700 | 4 | 512 | 3701.8 | 3340.2 | -9.77% | -10.04%, -9.50% | 2.48% / 1.01% | larger than descriptive noise |
| R9700 | 4 | 8192 | 3095.4 | 2906.4 | -6.10% | -6.64%, -5.56% | 2.74% / 0.25% | larger than descriptive noise |
| R9700 | 8 | 512 | 3651.1 | 3102.6 | -15.02% | -14.12%, -15.92% | 4.28% / 0.52% | larger than descriptive noise |
| R9700 | 8 | 8192 | 3086.1 | 2699.7 | -12.52% | -13.69%, -11.33% | 4.21% / 1.22% | larger than descriptive noise |
| Halo | 4 | 512 | 1296.3 | 1281.8 | -1.12% | -1.09%, -1.15% | 0.19% / 0.34% | larger than descriptive noise |
| Halo | 4 | 8192 | 1189.0 | 1176.4 | -1.06% | -1.81%, -0.31% | 3.89% / 1.28% | below descriptive noise |
| Halo | 8 | 512 | 1298.1 | 1242.1 | -4.31% | -3.11%, -5.50% | 0.25% / 5.52% | larger than descriptive noise |
| Halo | 8 | 8192 | 1196.3 | 1163.9 | -2.71% | -2.82%, -2.59% | 5.31% / 3.71% | below descriptive noise |

All eight cells completed their schedules: 64 fresh processes total, 128 retained warm samples. No process/outlier was discarded after looking at results. Halo pp512 fused8 has one slower process (1196.5 vs 1253.0–1265.0 tok/s); it remains included. Halo pp8192 first-control transients make both comparisons noisy; do not claim a statistically established ~1–3% regression from those cells. The noise column is descriptive, not a confidence interval or significance test. R9700's 6–15% losses are clear in both ABBA blocks. The pre-registered negligible-cost criterion was ≤1% median slowdown.

Timing admission IDs: R9700 j1099/j1100/j1101/j1102; Halo j1495/j1496/j1497/j1498. Original queued Halo pp8192 jobs j1493/j1494 were cancelled **before admission**, then resubmitted with 28 GiB after measuring the R9700 long-context footprint. All timing holds remained P1.

### What the widening changes

For 4/8, the producer calls the existing remap-plus-generic quantizer, rather than C2's one-pass emitter. The exact `-DIU4_A4_CANDIDATES=2` gate in `a4_slab_active()` also disables the C2-only gfx1201 slab path. The grouped symmetric IU4 MoE route is likewise C2-only (not relevant to this dense fixture). Accordingly, these are **end-to-end costs of the available widened route**, not an isolated measurement of extra candidate MSE instructions. No kernel arithmetic or default route was rewritten for this switch.

### Memory

Selected discrete R9700 counter: DRM card0 / fabric A / BDF `0000:03:00.0`. Total observed peak VRAM is 15.89 GiB at pp512 and 24.85 GiB at pp8192. The three-pass bench retains replaced KV allocations. Initial R9700 timing declarations were 24 GiB; long-context observed use exceeded that estimate by 0.85 GiB, though the runs held the 32-GiB card exclusively. Reproduction instructions now declare 28 GiB for pp8192.

Halo is DRM card1 / BDF `0000:c0:00.0`; its active GTT plateaus are approximately 16.31 / 23.29 GiB at pp512/pp8192. Some raw initial maxima (up to 68.43 GiB) already existed in the pre-process baseline and must not be attributed to this job. Full baseline/peak/delta counters and host RSS are retained; unrelated-card/global GTT counters are not treated as this process's memory. Halo pp8192 holds declared 28 GiB.

## Exercised checks and provenance

- Integrated release build of `eval_hipfire` and `bench_qwen35_mq4`, with `--features deltanet`, succeeded; cargo was masked with `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`.
- Four focused gfx1201 selector tests passed; 212 hipfire-config tests passed. Existing unrelated compiler warnings were emitted.
- Production-source producer oracle: all 2/4/8 variants passed on gfx1201 and gfx1151, 8302 blocks per variant (107 edge cases +8192 random +padding). Producer bytes equal generic quantizer bytes; candidate8 additionally equals standalone-full8 bytes. Guards, scale membership, code packing/sums, and host MSE checks passed. C2 exercised the one-pass path on both architectures; 4/8 exercised remap.
- Evaluator MD5: `6f8b02a9687c87d68ee70054e55fb45d`; SHA256 `b7ce5c050b9c4841ad48e134b5a0a561f0fb8b7c0e767715f163e6f012ce728b`.
- Benchmark MD5: `bcf2b78bad455491a29deaad346c82e4`; SHA256 `2893e0c2f4c01a3b84c94832b092426a0b864b77fdb3c5c204b0b68390c2e830`. The same binary was transferred to Halo.

## Receipts and reproduction

KEEP directory on hiptrx: `/home/kaden/qcal/release-0.4.3/a4-ladder-runs/`.

- `quality/*/quality-*.json`, corpus command/log/kldseq files, `quality-comparison.json`, `full8-repeat-comparison.json`.
- `timing-<r9700|halo>-c<4|8>-pp<512|8192>/{manifest.json,raw.jsonl,summary.json}`; `timing-comparison.json`, `true-memory.json`, `device-counter-map.json`.
- Full Halo process logs, atlas rows and caches remain at the same run paths on **hipx**; only structured receipts were collected to hiptrx. Do not recursively copy run HOME directories: their models symlink points at the canonical store.
- `oracle-*.receipt.json`, `oracle-builds.json`, `build-tests.receipt.json`, admission receipts and exact `*.command.txt` launch commands.
- Reusable commands and harness details: `experiments/a4-ladder/README.md`, `runner.py`, `producer_oracle.cpp`, `run_producer_oracle.sh` on the research branch.

**Decision: retain default 2; research switch only.** (Quality results stand; timing evidence above is stale/not product.)
