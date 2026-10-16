# A4 fused-ladder study harness

> **`runner.py timing` is stale / not product:** it drives `bench_qwen35_mq4` with q8 KV and literal token ids. Its tok/s are not product numbers (see RESULTS.md status); use `product_runner.py` below. The `quality` subcommand is unaffected.

## Product timing (`product_runner.py`)

Times the shipped CLI: `hipfire bench MODEL --backend noslots --workload stateless --prompt-file FILE
--max-tokens 1 --runs 6 --json` (not `--matrix`, not `bench_qwen35_mq4`). Every process is a fresh CLI +
fresh stdio daemon (`HIPFIRE_DAEMON_BIN` pinned), fresh `mkhome.sh` HOME, empty `HIPFIRE_KERNEL_CACHE`,
own process group (killed as a group on timeout/SIGTERM; no `pkill`). Inherited `HIPFIRE_*` (except
`HIPFIRE_DEVICES`/`LOCK_DIR`/`FABRIC_ADMISSION`), `HIP_*`/`HSA_*`/`ROCR_*`/`GPU_*`/`AMD_*` (except
visible-device vars) are scrubbed and recorded. Product defaults stay: no `HIPFIRE_KV*`/`GRAPH*`/
`NORMALIZE*`, no `--spec`, no DPM override (use a recorded `--env HIPFIRE_DPM_WARMUP_SECS=N` only if the
parent orders it). Only `HIPFIRE_G12_A4_CANDIDATES` varies: unset (default C2) / `4` / `8`.
`--backend/--workload` are recorded flags only (they matter just with `--concurrency`).

Prompts: `prompts/product-pp512.txt` (500 words) and `product-pp8192.txt` (8180 words), pinned by sha256;
the closed-think template adds 12 tokens so the daemon must report `prompt_tokens` exactly 512 / 8192.

```
python3 experiments/a4-ladder/product_runner.py baseline|abba --cli target/release/hipfire --daemon target/release/daemon \
  --model <canonical model> --prompt512 experiments/a4-ladder/prompts/product-pp512.txt \
  --prompt8192 experiments/a4-ladder/prompts/product-pp8192.txt --out <new dir>   # abba: --candidates 4,8 --blocks 2
python3 experiments/a4-ladder/product_runner.py summary <out>
```

`baseline` runs only unset C2 at 512 and 8192 (one process each); exit 4 if pp8192 is outside 4590..5610
tok/s (+-10% of ~5100) so the parent can explain before ABBA. `abba` runs, per candidate and length,
`--blocks` (default 2) palindromic `C2 cand cand C2` blocks = 4 fresh processes per arm per comparison
length, block-major. Both require `FABRIC_JOB` and stop at `--budget-secs` (1740).

Metric: per process, `samples.prefill` sample 0 is discarded (the Hello/16 warmup does not JIT the
long-prompt shapes), value = median of the last 5 (`prefill_tok_s` = daemon prefill_tokens / prefill_s);
arm = geomean of process medians; block ratio = geomean(cand procs) / geomean(C2 procs) in the block;
reported: block ratios, their geomean/median, arm spread/CV. No significance claim. A process is invalid
unless: exit 0, `native-generate-v1`, `gpu.arch` = `gfx1201`, effective `kv_mode` = `fp8`, exact
`prompt_tokens`, `prompt_md5` = file md5, 6 samples, and `prefill_tok_s*ttft_ms` (rows actually
prefilled) >= 99% of the prompt for every sample (a cached prefix would shrink it). Per process the dir
holds raw `stdout.json`, `stderr.txt`, `command.json` (argv, scrubbed/recorded env); `manifest.json`
records binary/model/prompt sha256, source HEAD/diff hash; `raw.jsonl` rows embed the full bench JSON and
effective `kv_mode`/`kv_backend*`/speculation fields.


`runner.py` (Python 3, stdlib only) is the quality and timing harness for the widened fused C2 A4
candidate ladder. It never opens a fabric hold itself: `quality` and `timing` refuse to run unless
`FABRIC_JOB` is set, i.e. they were launched by `fabric-run.sh`. Default behaviour is unchanged; the
only study knob is the A4 candidate selector, chosen by `--arch`:

| `--arch` | selector env key | expected `gpu.arch` | quality KV | timing KV |
|---|---|---|---|---|
| `r9700` | `HIPFIRE_G12_A4_CANDIDATES` (new) | `gfx1201` | `fp8` / V `q8` | `q8` / `q8` |
| `halo` | `HIPFIRE_GFX11_A4_CANDIDATES` (existing) | `gfx1151` | `q8` / `q8` | `q8` / `q8` |

`--env-key KEY` overrides the key. Arms are mapped on the command line, `--arm LABEL=VALUE`
(repeatable); `VALUE` is exported verbatim, and `@unset` leaves the selector unset (the shipped
default). Study control = `--arm control=@unset`, null check = `--arm explicit2=2`.

Timing uses `q8`/`q8` KV on both architectures because `bench_qwen35_mq4` only builds q8/asym KV
caches (no fp8). All arms in a timing comparison share the identical cache setting, and the manifest
records it. Quality keeps the pinned R9700 fp8/q8 setting.

## Build

Run in `/home/kaden/qcal/release-0.4.3/a4-fused-ladder`:

```
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CARGO_BUILD_JOBS=8 cargo build --release --features deltanet -p hipfire-runtime --example eval_hipfire --example bench_qwen35_mq4
```

Binaries: `target/release/examples/eval_hipfire`, `target/release/examples/bench_qwen35_mq4`
(the bench is a stub without `--features deltanet`). Verify the fixture outside any hold with
`scripts/check_fixture.sh <model>`; it must print sha256
`3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`, 14987185152 bytes.

## Quality (P0, fabric `--class correct`, <=600 s)

One arm per job, both corpora (WT2 then code24), a fresh HOME (`mkhome.sh`) and empty kernel cache
per corpus, `HIPFIRE_GRAPH=0`, `HIPFIRE_NORMALIZE_PROMPT=0`, `--scoring-mode prefill --max-chunks 24`.
The model sha256 and both reference sha256 are checked first (model/reference mismatch aborts).
`--validate-pin` additionally requires the `.kldseq` bytes to equal the shipped default pin
(WT2 `ff88c499...2841`, code24 `9b061631...910e`); use it for the unset control and for `explicit2`.

```
fabric-run.sh --class correct --card <r9700 card> --mem-gb 24 --max-secs 600 --wait-secs 14400 --priority P0 \
  --owner A4LadderStudy -- python3 experiments/a4-ladder/runner.py quality --arch r9700 \
  --arm control=@unset --validate-pin --model <canonical .mq4-xts> \
  --bin target/release/examples/eval_hipfire --out <runs>/quality/control
```

Per-arm output `<out>/quality-<label>.json`: model/binary/reference hashes, source provenance
(`git HEAD`, `git diff HEAD` hash, untracked-file hashes, hashes of the bench/eval/feature_flags/
compiler sources), each corpus' command, recorded environment, `.kldseq` sha256, KLD (mean of the 24
chunk means), mean p99, mean NLL, per-chunk values, `pin_match`, route-evidence log lines
(`A4_CANDIDATES`, `per-kernel flags`, `unknown HIPFIRE_*`, fallback/WARN lines), kernel-cache size.
A runtime warning that the selector value is unknown ("using default") fails the job.

After all arms ran:

```
python3 experiments/a4-ladder/runner.py quality-compare <runs>/quality/*/quality-*.json --control control --out compare.json
```

prints absolute KLD, delta vs control, whether the bytes equal the control, the recovered fraction
`-(delta) / gain` of the WMM4 standalone gains (WT2 0.0023473457, code24 0.0019178636; assumes the
gains are KLD reductions), and the one-sided `+0.010` cap check.

## Timing (P1, fabric `--class timing`, <=1800 s, one hold per comparison)

All selected arms run inside the one hold you launch. `--control` (default first `--arm`) and
`--candidates` (default all other `--arm`s) select the arms; `--lengths` (default `512,8192`) and
`--blocks` (default 2) select how much fits in a hold, e.g. one candidate and one length for Halo
pp8192. `plan` prints the schedule without touching files:

```
python3 experiments/a4-ladder/runner.py plan --arch halo --arm control=@unset --arm fused8=8 --lengths 8192 --blocks 2
```

Schedule: palindromic ABBA per block, `A B B A` for two arms, `A B C C B A` for three; every pair of
arms is an ABBA subsequence of the block. Two blocks = four fresh processes per arm per length.
Each process gets its own fresh HOME and empty `HIPFIRE_KERNEL_CACHE` (JIT happens inside every
process, so timings are cold-process timings with first sample discarded), `HIPFIRE_GRAPH=0`,
`HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_KV_MODE=q8`, `HIPFIRE_KV_V=q8`, and
`HIPFIRE_DPM_WARMUP_SECS=3` (change with `--dpm-warmup-secs`, `0` disables).

Command per process: `bench_qwen35_mq4 <model> --prefill N --prefill-runs 3 --gen 0 --warmup 0
--emit-atlas atlas.jsonl`. The prompt is the bench's literal tokens `0..N-1`; spec decode is not
involved. Prefill-code facts the harness relies on (read from `bench_qwen35_mq4.rs`):

- Each run re-creates the DeltaNet state and KV cache (untimed), then times
  `forward_prefill_batch` + `device_synchronize`.
- Only the stderr `run N: X.Xms` lines carry every sample. `PREFILL_SUMMARY`, the atlas row, and
  `SUMMARY` report only the **last** sample, and the bench states that run 1 includes kernel JIT. So
  the harness parses all `run N:` lines, explicitly discards sample 1 (stored as
  `discarded_first_ms`), and uses samples 2..N (two warm samples at the default `--prefill-runs 3`).
  `--prefill-runs` must be >= 3. The atlas last-sample value is cross-checked against the last run
  line.
- `--warmup 0` removes the decode warmup tokens; `--gen 0` makes the bench return right after the
  prefill (after its post-prefill DPM warm-up, if enabled). The bench retains replaced KV caches.
  Measured R9700 total peak VRAM is 15.89 GiB at pp512 and 24.85 GiB at pp8192
  with three samples/process; declare 28 GiB for pp8192 (including Halo UMA admission).
- `HIPFIRE_PREFILL_REUSE_PBS` is unset, as in the bench default; `--env KEY=VALUE` adds extra
  child environment to every arm (recorded in the manifest).

```
fabric-run.sh --class timing --card <card> --mem-gb 28 --max-secs 1800 --wait-secs 14400 --priority P1 \
  --owner A4LadderStudy -- python3 experiments/a4-ladder/runner.py timing --arch r9700 \
  --arm control=@unset --arm fused8=8 --lengths 8192 --blocks 2 \
  --model <canonical .mq4-xts> --bin target/release/examples/bench_qwen35_mq4 --out <runs>/timing/r9700-c8-pp8192
```

Model sha256 is not recomputed inside the hold by default (it costs about a minute of the 1800 s):
the digest passed by `--model-sha256` (default: canonical) is recorded as `expected_sha256` with
`sha256_verified: null`; add `--hash-model` to recompute and enforce it. The model size is always
checked. `--budget-secs` (default 1740) stops launching a process that cannot finish inside the hold,
and a SIGTERM from the fabric kills the child, writes the partial summary and exits non-zero.
Status is `complete` only if every scheduled process ran and was valid; exit code 3 otherwise. By
default the first invalid process (non-zero exit, wrong run-line count, wrong gpu arch, wrong KV
mode, rejected selector value, atlas mismatch) stops the run; `--keep-going` continues and lists them
under `invalid_processes`.

Outputs under `--out`:

- `manifest.json`: arms, selector key/values, schedule, KV, prompt rule, binary and source hashes,
  model record, fabric env, extra env.
- `raw.jsonl`: one row per process: all `samples_ms`, `discarded_first_ms`, retained ms and tok/s,
  the process median, wall time, exit code, host peak RSS, per-card `mem_info_vram_used` /
  `mem_info_gtt_used` baseline/peak/delta/total from sysfs (the Halo UMA shows in GTT), parsed bench
  evidence (GPU arch, KV mode, weights-load seconds, DPM warm-up headers/seconds, next token), atlas
  row, route-evidence log lines, kernel-cache file count/bytes, `proc/<tag>/` directory.
- `summary.json` (also rebuilt by `runner.py summarize <out>/raw.jsonl`): per length and arm, the
  process medians, their geomean, min/max, spread and CV, plus per-pair (control vs each candidate)
  per-block ratios, geomean/median ratio, `median_slowdown_pct`, `noise_pct`,
  `delta_resolved_above_noise`, `within_1pct_median_slowdown`, `abba_complete`.

Statistics: process value = median of its retained tok/s; arm-in-block value = geomean of its
process medians; block ratio = candidate / control; `noise_pct` = max(|first-control/last-control
- 1| over blocks, half the block-ratio range). A delta whose magnitude is not above `noise_pct` is
reported as unresolved, not as a win or loss. The "<= 1% median slowdown" criterion is
`within_1pct_median_slowdown` and must be read together with `noise_pct`.

## Route evidence limits

The runtime logs only per-kernel/module flags, not the global `-DIU4_A4_CANDIDATES` value, so the
logs prove only that the selector was accepted (an unknown value logs a warning and falls back to 2,
which fails the job). Compile-flag evidence beyond that must come from the runtime change itself
(for example a route/flag log line, which the harness captures automatically when it matches
`A4_CANDIDATES`). The generic widened path disables the C2-specific slabs, so a timing delta on an
arm that does not use C2 on a given architecture is a route difference or a no-op, not ladder cost;
report such cells as no-op.

## Files

- `runner.py`: all subcommands (`plan`, `quality`, `timing`, `summarize`, `quality-compare`).
- `README.md`: this file.
- Host paths are parameters: `--bin`, `--model`, `--out`, `--repo`, `--cwd`, `--mkhome` (or
  `A4_MKHOME`), `--ref CORPUS=PATH`. Defaults are the current hipx/hiptrx paths.
