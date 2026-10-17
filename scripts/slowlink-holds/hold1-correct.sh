#!/usr/bin/env bash
# hold1-correct.sh — HOLD 1 (real schedule correctness), P0 correct, <= 600 s. NOT SUBMITTED by this seat.
#   fabric-run.sh --class correct --card xtx --mem-gb $(read DEV) --hostmem-gb $(read HOST) --hostmem-lock --max-secs 600 \
#                 --priority P0 --owner Main.SlowLinkS1 -- .../holds/hold1-correct.sh
#   hold1-correct.sh --print-memory   prints "<device GiB> <host GiB>" from the declared final ledger-hold1.json (refuses if absent or not final)
#
# Cells, in order; a failure stops every dependent cell (small primitive kill first):
#   prim         qwen4_slowlink_gate --mode primitives --route layer-major   (7 production primitive cases on real host experts)
#   residual-prim  CPU: check_residual.py on the first cell: sysfs vram/gtt before-after + VmRSS/VmPin/RssAnon samples vs the DECLARED
#                numeric ledger (device_owned_bytes = committed core + sequential diagnostic; host_owned_bytes; the 2 GiB session-cache
#                budget and the pending ceiling are recorded, never added) with the fixed allowances 1,610,612,736 B device /
#                2,147,483,648 B host (declared limits, NOT measurements); fails closed over the allowance and blocks every later cell
#   state        qwen4_slowlink_gate --mode state --route layer-major        (real 16K one-window schedule, irregular/tail/resume/
#                boundary/abort fixtures, same-route repeat digests, eager + retained-body replay decode, head rows, injections)
#   greedy-beta  beta qwen4_greedy_ids on prompts.short.jsonl (beta AR, immutable reference binary; the beta example loader
#                needs a fixed expert layer count: HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10, the S1 minimum residency)
#   greedy-s1    candidate qwen4_greedy_ids (--receipt) on the SAME bytes with HIPFIRE_QWEN4_SLOWLINK_PREFILL=1
#   greedy-cmp   CPU: exact ids + the route proof from the receipts (verify_greedy.py)
# ADVANCE only when every check passed, residency >= 10, tile <= 1536, bounded charged lookahead, one submitted payload sweep per
# window and no donor restore; a missing time/case is INCOMPLETE and no promotion. Fail first -> diagnose on CPU.
HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HOLDS/hold-lib.sh"
[ "${1:-}" = "--print-memory" ] && hold_memory 1
hold_init hold1 600
S=$STAGE
REQUIRED="prim residual-prim state greedy-beta greedy-s1 greedy-cmp"
GATE=$S/bin-s1/qwen4_slowlink_gate

cell prim 80 60 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/prim/gate.json" \
  --mode primitives --route layer-major --manifest "$S/manifests/manifest-hold1.json"
# FIRST cell memory receipt: measured peak vs the DECLARED numeric ledger; fails closed over the allowance and blocks the rest.
[ -r "$S/ledger-hold1.json" ] || die "no declared ledger $S/ledger-hold1.json (final owned totals are provided by S1Integrate)"
residual residual-prim "$OUT/prim" "$S/ledger-hold1.json"
cell state 260 150 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/state/gate.json" \
  --mode state --route layer-major --manifest "$S/manifests/manifest-hold1.json"
cell greedy-beta 100 70 HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10 -- "$S/bin-beta/qwen4_greedy_ids" --model "$MODEL" --input "$S/fixtures/prompts.short.jsonl" \
  --output "$OUT/greedy-beta/ids.jsonl" --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
cell greedy-s1 100 70 HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 -- "$S/bin-s1/qwen4_greedy_ids" --model "$MODEL" \
  --input "$S/fixtures/prompts.short.jsonl" --output "$OUT/greedy-s1/ids.jsonl" --receipt "$OUT/greedy-s1/receipts.jsonl" \
  --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
check greedy-cmp python3 "$HOLDS/verify_greedy.py" "$OUT/greedy-beta/ids.jsonl" "$OUT/greedy-s1/ids.jsonl" "$OUT/greedy-s1/receipts.jsonl"
finish_summary
