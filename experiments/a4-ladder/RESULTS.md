# Fused C2 A4 ladder study — closing report

## Recommendation

**Keep C2 as the default, now based on corrected product-path evidence appended below.**

The old timing table is explicitly **stale bench_qwen35_mq4, q8 KV: not product** and is superseded by the product timing section at the end of this report. The corrected C2 baseline is 5002.7 tok/s at pp8192, near Kaden's ~5100 reference. Product ABBA finds fused8/C2 ratios **0.8250 at pp512 / 0.7936 at pp8192** (−17.50% / −20.64%); fused4 ratios are **0.8905 / 0.8871** (−10.95% / −11.29%). The R9700 result is not close, so no new Halo run was made.

The pinned `eval_hipfire` quality results stand: fused8 improves R9700 WT2/code24 KLD by −0.000999 / −0.002220, recovering 42.6% of WMM4's WT2 standalone gain (115.7% on code24). Defaults remain unchanged; no default flip or re-pin is authorized.

## Branch and switch

- Published branch: `research/a4-fused-ladder`; initial study base was origin/beta `4b68ffde8`. Corrected product timing is rebased onto `18b216848` (details appended below).
- Original switch implementation: `bec0e6d52` (rebased as `23c94df30`); original quality/old-example build commit: `d35ae715dbaeb9240b148a3f739a343838ba6bf3`. Corrected product binary build commit: `d0937deebcd64fe79d66f206d57a74826dfbfd13`.
- Worktree: `/home/kaden/qcal/release-0.4.3/a4-fused-ladder` on hiptrx. The old Halo study used a clean detached Git worktree on hipx; no corrected Halo product run was made.
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

## Historical timing — stale bench_qwen35_mq4, q8 KV: not product

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

All eight historical cells completed their schedules: 64 fresh processes total, 128 retained warm samples. No process/outlier was discarded after looking at results. These statistics describe the stale example only, not the product. Halo pp512 fused8 includes its slower process; Halo pp8192 comparisons are noisy. No product speed verdict follows from this table.

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

**The original timing verdict is superseded by the corrected product-path section below. Default remains 2.**

## Corrected product timing — supersedes the entire old timing table

### Rebase and baseline first

Rebased `research/a4-fused-ladder` onto beta **`18b2168482dffe9ba82188352e4406b3b1b48084`**. The old published tip is preserved as `backup/a4-fused-ladder-3d2b0222a-pre-rebase-20261016`. One changelog conflict was resolved by retaining both beta entries and the study entry; defaults and kernel arithmetic were not changed.

Built release `hipfire` and `daemon` at source commit **`d0937deebcd64fe79d66f206d57a74826dfbfd13`**, with GPUs hidden:

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 cargo build --release -p hipfire-cli -p hipfire-daemon -j8
```

The first P1 timing hold, **j1293 on R9700 card B**, ran only the unset-C2 baseline at both lengths using the actual standard product command:

```sh
hipfire bench /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xts \
  --backend noslots --workload stateless \
  --prompt-file experiments/a4-ladder/prompts/product-pp8192.txt \
  --max-tokens 1 --runs 6 --json
```

`HIPFIRE_DAEMON_BIN` pointed at the rebuilt daemon. KV, graph, normalization, and speculation were not overridden. The daemon acknowledged **effective KV `fp8`, backend `vmm`, automatic backend request**, gfx1201. This is `native-generate-v1`, not `--matrix` or an in-process example.

| Initial product C2 baseline | Exact formatted prompt tokens | Median of last five samples, tok/s |
|---|---:|---:|
| pp512 | 512 | 3800.0 |
| pp8192 | 8192 | **5002.7** |

The pp8192 samples after the initial JIT-bearing sample were 4992.4, 5012.6, 5010.9, 5002.7, 4992.1 tok/s. The median is **1.91% below ~5100**, satisfying the requested near-5100 baseline before any candidate timing began. The stale ~3086 result is not used as a baseline or denominator. Product logs explicitly show `prefill_chunk: requested=8192 admitted=8192 commit_stride=512`.

Prompt bodies were generated with the model's **embedded tokenizer and Jinja template**, not a guessed token count: 500 / 8180 `hello` words plus the product's closed-think framing produce exactly 512 / 8192 tokens. Both the initial baseline and every ABBA process confirmed the exact formatted length. The native bench includes the required `--backend noslots --workload stateless` flags; in this source version those selectors are consumed by the concurrency sweep, while the standard native-generate path itself is sequential with fresh requests. No concurrency sweep was substituted.

### Matched product ABBA — one exclusive P1 hold

Only after that baseline passed, hold **j1295**, on the **same R9700 card B**, ran both widened candidates and the common C2 control. Each of two blocks at each length was:

```text
C2  C8  C4  C4  C8  C2
```

Each candidate/control pair is an ABBA subsequence. There were **24 fresh CLI/daemon processes**, four per arm per length, all in this one hold. Every process used a fresh HOME/cache, the same exact prompt file, the pinned rebuilt binaries, and the same product-default KV/graph/normalization/speculation. The only arithmetic selector varied was `HIPFIRE_G12_A4_CANDIDATES`: unset, 8, or 4. All 24 processes were valid and all schedules completed.

For clock stabilization the ABBA cohort also set `HIPFIRE_DPM_WARMUP_SECS=10`, which the daemon runs before its loaded acknowledgement, outside the measured prefill. Each CLI invocation performed its ordinary Hello/16 warmup and six measured prompt requests. The first prompt sample was discarded for long-shape JIT; the per-process value is the median of the last five. The table reports geometric means of the four process medians. No process or retained sample was removed post hoc.

| R9700 product, default auto→fp8 KV | C2 tok/s | Candidate tok/s | Candidate / C2 | Throughput Δ | Two block ratios |
|---|---:|---:|---:|---:|---|
| C8, pp512 | 3796.3 | 3132.1 | **0.825033** | **−17.497%** | 0.828844, 0.821239 |
| C8, pp8192 | 4992.0 | 3961.5 | **0.793575** | **−20.643%** | 0.798921, 0.788264 |
| C4, pp512 | 3796.3 | 3380.4 | **0.890458** | **−10.954%** | 0.906918, 0.874297 |
| C4, pp8192 | 4992.0 | 4428.4 | **0.887106** | **−11.289%** | 0.886391, 0.887822 |

C2 process spreads were 1.43% / 0.96% at pp512/pp8192. C8 spreads were 1.22% / 1.98%; C4 spreads were 4.75% / 1.35%. Both block ratios are well below 1 for every comparison; no statistical-significance claim is needed to distinguish these losses from negligible cost. **The R9700 result is not close, so Halo was skipped as instructed.** The old Halo numbers remain stale/not-product, not a replacement product result.

All 24 effective identities matched: `kv_mode=fp8`, `kv_backend=vmm`, automatic backend request, no KV warning, gfx1201; no graph/KV/normalize override or `--spec off` was used. Product-default DFlash auto loaded the paired draft. Max-tokens 1 yielded zero speculative decode windows; this was not forced into a different AR-only load. The raw samples' prefill-rate × TTFT products match the full prompt lengths to rounding, providing no evidence of a cached-prefix shortcut.

### Corrected disposition and identity

**Do not flip to these existing widened routes.** C8's pinned KLD improvement remains real, but its corrected product prefill cost is ~17.5–20.6%, not negligible. C4 both worsens the pinned KLD means and costs ~11% product throughput. The generic widened emitter still disables the C2-only one-pass/slab route; these are costs of the available end-to-end implementation, not a claim that an unmeasured redesigned one-pass C8 kernel must be this slow.

- CLI SHA256: `2f4d41bceb850d3c183290040e82f28ffecaefd69b04412d76a5f2f5c34f04fd`.
- Daemon SHA256: `376781125d4ce2b5088818700cddfa914bda16ef771484ab610d5bb56169728d`.
- Paired draft SHA256, verified from the canonical file: `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc`.
- Target SHA256 remains the canonical `3e38ccba…f7c6f8ae` pin above.
- Clean protocol/source commit recorded by the ABBA harness: `722ebbaae36e601ce3c78912b9c93f5eb9fb6411`. Changes after the binary-build commit were Python protocol, prompt fixtures, and documentation only.
- Both corrected timing holds used P1, exclusive card B, with 30 GiB declared device memory. No new Halo job was submitted.

Corrected receipts are under `/home/kaden/qcal/release-0.4.3/a4-ladder-runs/product/`:

- `rebase-build.log`, `rebase-record.txt`, `rebase-binaries.sha256`.
- `prompt-fixtures.json`, `baseline-initial/{pp512,pp8192}/{command.json,result.json,stdout.json,stderr.log}`, `baseline-initial-summary.json`, `baseline-approved.json`, and baseline admission receipt.
- `abba-r9700/{manifest.json,raw.jsonl,summary.json}` and its per-process raw stdout/stderr/commands; `abba-r9700.admission.json` preserves the fabric receipt.
- Reproduction: branch `experiments/a4-ladder/product_runner.py` and README. The runner refuses non-fp8 effective KV, wrong prompt length/hash, rejected selectors, incomplete schedules, or stray owned daemon processes.

**Corrected decision: retain default C2; the non-exact research switch stays opt-in. KLD evidence stands; all old timing is superseded.**
