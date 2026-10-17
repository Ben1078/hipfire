#!/usr/bin/env bash
# hold1-correct.sh — HOLD 1 (real schedule correctness), P1 correct, <= 1500 s. NOT SUBMITTED by this seat.
#   fabric-run.sh --class correct --card xtx --mem-gb $(read DEV) --hostmem-gb $(read HOST) --hostmem-lock --max-secs 1500 \
#                 --priority P1 --owner Main.SlowLinkS1 -- .../holds/hold1-correct.sh
#   hold1-correct.sh --print-memory   prints "22.4 54" (fabric exclusive device cap GiB, host GiB) from the declared final ledger-hold1.json (refuses if absent or not final)
#
# Every GPU cell runs under the external VRAM sampler (vram_sampler.py: sysfs mem_info_vram_used/total only, requested 20 ms, started
# before the command, stopped after the drain) next to the old 2 s host sampler (sample_mem.py, information only).
# Cells, in order; a failure stops every dependent cell (small primitive kill first):
#   prim         qwen4_slowlink_gate --mode primitives --route layer-major   (7 production primitive cases on real host experts; identity/
#                load/teardown must pass; its memory census/residual/peak are INFORMATION only, never a pass/fail term)
#   state        qwen4_slowlink_gate --mode state --route layer-major        (real 16K one-window schedule, irregular/tail/resume/
#                boundary/abort fixtures, same-route repeat digests, eager + retained-body replay decode, head rows, injections)
#   headroom-state  CPU: check_headroom.py on the state cell: min OBSERVED free VRAM over the WHOLE run >= 536,870,912 B from BOTH the
#                external 20 ms sysfs samples (total - used, each sample's own total) AND the in-process hipMemGetInfo points pre_load/
#                post_load/end of gate.json evidence.memory_diagnostics.samples (each its own device_total; the two totals are never
#                required to be equal). Missing/invalid/partial telemetry is INCOMPLETE (never zero-filled, never a pass). Owned census,
#                residual and the prim peak are INFO only (source-owned 11-resident core 24,850,537,964 B + nominal diag 217,812,992 B =
#                25,068,350,956 B is header arithmetic, not a driver figure). Failure/INCOMPLETE blocks every later cell.
#   greedy-beta  beta qwen4_greedy_ids on prompts.short.jsonl (beta AR, immutable reference binary; the beta example loader
#                needs a fixed expert layer count: HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10, the S1 minimum residency)
#   greedy-s1    candidate qwen4_greedy_ids (--receipt) on the SAME bytes with HIPFIRE_QWEN4_SLOWLINK_PREFILL=1
#   greedy-cmp   CPU: exact ids + the route proof from the receipts (verify_greedy.py)
# TIMING (every cell keeps its full case list; no cell is dropped, no beta opt-out): each cap = (cold model LOAD budget 110 s +
# a work ALLOWANCE) * 1.30, rounded. The 110 s budget uses Main's ~80-110 s load range; j1759 did NOT complete a load measurement:
# its 80 s cap expired after ~82 s elapsed while still loading. Work allowances are SOURCE-COUNT allowances, NOT measurements:
# prim 25 s (7 primitive cases), state 440 s (x3 repeats of 8 token-id prompts (44,661 rows = 133,983 prefill rows) + the natural-text prompt
# (~2.4 KB) + 27x32 decode steps, the marker-boundary and 16K boundary restores, 4 failure injections, eager+retained decode;
# the S1 throughput this assumes is not measured), greedy beta/s1 45 s each. Caps 180 + 720 + 210 + 210 = 1320 s, + 45 s finalize = 1365 s
# of the 1500 s hold, leaving 135 s for the CPU checks and the 90 s-bounded VRAM drain after each cell. A cell's min start budget is its FULL cap (180/720/210/210 s):
# the 30% margin is never silently truncated; with less than the full cap left the cell is INCOMPLETE (not a narrowed run). A capped
# (timeout) cell fails; a missing cell or time is INCOMPLETE and no promotion.
# ADVANCE only when every check passed, residency >= 10, tile <= 1536, bounded charged lookahead, one submitted payload sweep per
# window and no donor restore; a missing time/case is INCOMPLETE and no promotion. Fail first -> diagnose on CPU.
HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HOLDS/hold-lib.sh"
[ "${1:-}" = "--print-memory" ] && hold_memory 1
hold_init hold1 1500
S=$STAGE
REQUIRED="prim state headroom-state greedy-beta greedy-s1 greedy-cmp"
GATE=$S/bin-s1/qwen4_slowlink_gate

cell prim 180 180 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/prim/gate.json" \
  --mode primitives --route layer-major --manifest "$S/manifests/manifest-hold1.json"
cell state 720 720 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/state/gate.json" \
  --mode state --route layer-major --manifest "$S/manifests/manifest-hold1.json"
# PRODUCT headroom verdict (CPU) over the whole state run; the primitive cell's peak is recorded as information only.
headroom headroom-state "$OUT/state" "$OUT/prim"
cell greedy-beta 210 210 HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10 -- "$S/bin-beta/qwen4_greedy_ids" --model "$MODEL" --input "$S/fixtures/prompts.short.jsonl" \
  --output "$OUT/greedy-beta/ids.jsonl" --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
cell greedy-s1 210 210 HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 -- "$S/bin-s1/qwen4_greedy_ids" --model "$MODEL" \
  --input "$S/fixtures/prompts.short.jsonl" --output "$OUT/greedy-s1/ids.jsonl" --receipt "$OUT/greedy-s1/receipts.jsonl" \
  --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
check greedy-cmp python3 "$HOLDS/verify_greedy.py" "$OUT/greedy-beta/ids.jsonl" "$OUT/greedy-s1/ids.jsonl" "$OUT/greedy-s1/receipts.jsonl"
finish_summary
