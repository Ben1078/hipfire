#!/usr/bin/env bash
# hold1b-correct.sh (R8) — VERSIONED, HARNESS-ONLY repair of HOLD 1b (Hold 1 completion: headroom RECHECK on the immutable 1a RAW state data + beta/S1
# greedy), P0 correct, hard 1320 s. NOT SUBMITTED by this seat.
#   hold1b-correct.sh EXEC_JSON EXPECTED_SHA256            the H1a handoff is NOT an argument: it is the one the pinned execution JSON binds
#   hold1b-correct.sh --print-memory EXEC_JSON EXPECTED_SHA256   prints exactly "22.4 67" (fabric exclusive device GiB, effective R8 host GiB); the
#                                                          physical 24 GiB card is named on stderr only. Works outside a fabric job.
#   EXEC_JSON = the immutable external execution-h1b-r8.json; EXPECTED_SHA256 = its sha256, supplied INDEPENDENTLY in the submitted argv.
#
# WHY: j1934 failed the immutable ff799 beta reference at its first p512 prefill with a hipMalloc OOM (S1/compare never ran). H1a j1923 PASSED and its
# raw seal stays valid; runtime 40a and beta ff799 are unchanged. The approved repair changes ONLY the beta reference arm's two quality pins:
#   HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=5   (was 10)      HIPFIRE_PREFILL_CHUNK_ROWS=4096   (the old ACTUAL chunk; requested 8192 had been admitted at 4096)
# N=5 alone could let the admission ladder grow the ACTUAL chunk to 8192, which changes the arithmetic class (irregular 4100: [4096, 4] -> [4100]); the
# explicit 4096 preserves the old actual geometry. It is a QUALITY-REFERENCE pin, never H2 policy. The ceiling can downshift, so the receipt requires
# the beta stderr's own `qwen4 prefill chunk: 4096 rows` line. The candidate arm keeps ONLY HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 (no resident/chunk pin).
# Both arms keep HIPFIRE_OFFLOAD_EXEC=pcie (the real operational GPU / no-CPU control) and the recorded D2 metadata pin HIPFIRE_QWEN4_D2_DEFAULT=0
# (absent from ff799 source: metadata only). Same seven rows, order, scope, input adapter, seven records, up-to-32-token options (--max-new-tokens 32:
# generation stops early on EOS), S1 product tile 768, budgets and drains as R7: beta 600 + S1 390 = 990 s, + 2 x 15 s kill graces + 2 x 90 s VRAM drains
# + 45 s finalize + 30 s CPU = 1275 s, hard maximum 1320 s. No narrowing, no cut, no retry.
#
# HOW: the frozen R7 sources are NOT edited and NOT restaged. The execution JSON binds the exact R7 helper directory (complete file map), the R8 files
# (this directory), the saved diff, the CPU/source proof and a SEPARATE effective memory ledger (effective 67 GiB host capacity bound; the R7 stage ledger
# stays the immutable base provenance). R8_HOLDS is this directory (new bridge only). HOLDS (set by hold-lib.sh) stays the TRUTHFUL R7 directory, so the
# original sampler, evidence collector, headroom judge, beta-input adapter and verify_greedy.py remain the frozen R7 files. No R7 library is sourced
# before the bridge's `resolve` has validated the execution JSON against its independently supplied sha256 and the actual R7/R8 file inventories.
# Rows (unchanged names/order; a failure stops every dependent row):
#   a-handoff       CPU: h1b_bridge.py verify = the UNCHANGED original R7 h1_handoff.verify on the immutable H1a handoff + the execution/run binding
#   headroom-state  CPU RECHECK on the 1a RAW state data (R7 check_headroom.py via hold-lib headroom())
#   beta-input      CPU: R7 beta_input.py verify
#   greedy-beta     beta qwen4_greedy_ids, beta-only pins resident=5 AND chunk=4096 (+ H1 pcie pin)
#   greedy-s1       candidate qwen4_greedy_ids --receipt on the canonical input, SLOWLINK_PREFILL=1 only (+ H1 pcie pin)
#   greedy-cmp      CPU: R7 verify_greedy.py
#   receipt-1b      CPU: h1b_bridge.py receipt1b (schema s1.hold1b_receipt.v2)
# PASS of 1b with the verified 1a is the Hold 1 ADVANCE candidate (scope h1_complete_on_verified_1a, advance true). H2 only after this repaired H1b PASS.
# LAB-LINK POLICY: each cell is ONE process, ONE model load; a device bus loss: STOP and report INCOMPLETE (lab), NO automatic retry.
R8_HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
usage() { echo "usage: hold1b-correct.sh EXEC_JSON EXPECTED_SHA256 | hold1b-correct.sh --print-memory EXEC_JSON EXPECTED_SHA256" >&2; exit 2; }
if [ "${1:-}" = "--print-memory" ]; then
  [ "$#" = 3 ] || usage
  exec python3 "$R8_HOLDS/h1b_bridge.py" print-memory "$2" "$3"
fi
[ "$#" = 2 ] || usage
EXEC_JSON=$(readlink -f "$1")
EXEC_SHA=$2
RESOLVED=$(python3 "$R8_HOLDS/h1b_bridge.py" resolve "$EXEC_JSON" "$EXEC_SHA") || { echo "FATAL: execution JSON/inventory validation failed (no GPU, nothing sourced)" >&2; exit 2; }
mapfile -t RESOLVED_LINES <<<"$RESOLVED"
[ "${#RESOLVED_LINES[@]}" = 3 ] || { echo "FATAL: resolve yielded ${#RESOLVED_LINES[@]} fields, expected 3" >&2; exit 2; }
R7_HOLDS=$(readlink -f "${RESOLVED_LINES[0]}")
EXEC_STAGE=$(readlink -f "${RESOLVED_LINES[1]}")
A_OUT=$(readlink -f "${RESOLVED_LINES[2]}")
. "$R7_HOLDS/hold-lib.sh"
[ "$HOLDS" = "$R7_HOLDS" ] || { echo "FATAL: hold-lib resolved HOLDS=$HOLDS, not the bound base R7 directory $R7_HOLDS" >&2; exit 2; }
. "$HOLDS/hold1-common.sh"
[ "$(readlink -f "$STAGE")" = "$EXEC_STAGE" ] || { echo "FATAL: S1_STAGE/STAGE $STAGE is not the stage the execution JSON binds ($EXEC_STAGE)" >&2; exit 2; }
# Fresh unique OUT, validated BEFORE the frozen hold_init (which mkdirs OUT and only afterwards rejects an existing hold.jsonl, so an existing directory such
# as the sealed 1a A/state could otherwise be written into): OUT must be explicit, absolute, normalized, NONEXISTENT (no directory, file or symlink) and not
# inside or equal to the bound H1a run directory, stage, base R7 directory or R8 directory. echo+exit, never the base die() (it writes FATAL.txt into OUT).
out_refuse() { echo "FATAL: OUT $1 (a fresh explicit absolute nonexistent run directory outside the bound H1a/stage/R7/R8 is required; nothing was created)" >&2; exit 2; }
[ -n "${OUT:-}" ] || out_refuse "is unset"
case "$OUT" in /*) ;; *) out_refuse "'$OUT' is not absolute" ;; esac
[ "$(readlink -m -- "$OUT")" = "$OUT" ] || out_refuse "'$OUT' is not normalized (a symlinked or dotted path)"
{ [ ! -e "$OUT" ] && [ ! -L "$OUT" ]; } || out_refuse "'$OUT' already exists"
for protected in "$(dirname "$A_OUT")" "$EXEC_STAGE" "$R7_HOLDS" "$R8_HOLDS"; do
  protected=$(readlink -m -- "$protected")
  case "$OUT/" in "$protected"/*) out_refuse "'$OUT' is inside or equal to the bound directory $protected" ;; esac
done
HOLD_SCOPE=h1_complete_on_verified_1a
HOLD_ADVANCE=true
hold_init hold1b 1320
export OUT  # the bridge's verify/receipt bind the fresh OUT (and the actual FABRIC_JOB) from the environment and the hold.jsonl hold_init just wrote
S=$STAGE
REQUIRED="a-handoff headroom-state beta-input greedy-beta greedy-s1 greedy-cmp receipt-1b"
[ -f "$A_OUT" ] || die "the bound H1a handoff $A_OUT is missing"
A_DIR=$(dirname "$A_OUT")
[ -z "${S1_HOLD1A_OUT:-}" ] || [ "$(readlink -f "$S1_HOLD1A_OUT")" = "$A_OUT" ] || [ "$(readlink -f "$S1_HOLD1A_OUT")" = "$A_DIR" ] \
  || die "S1_HOLD1A_OUT names another 1a than the execution JSON binds; the H1a handoff comes only from the pinned execution JSON"

# Nothing below this verification starts a GPU process unless the immutable 1a handoff and the execution/run binding are intact.
check a-handoff python3 "$R8_HOLDS/h1b_bridge.py" verify "$S" "$A_OUT" "$HOLDS" "$EXEC_JSON" "$EXEC_SHA" \
  || { [ -n "$BLOCKED" ] || BLOCKED="a-handoff failed: the 1a handoff or the R8 execution binding is missing, changed, partial or of another identity (no GPU)"; }
headroom headroom-state "$A_DIR/state" "$A_DIR/prim"
check beta-input python3 "$HOLDS/beta_input.py" verify "$S/fixtures/prompts.short.jsonl" "$S/fixtures/prompts.short.beta.jsonl" \
  "$S/fixtures/beta-input-proof.json" \
  || { [ -n "$BLOCKED" ] || BLOCKED="beta-input failed: the beta greedy input is not the canonical input minus only resume_at"; }
cell greedy-beta 600 600 "$H1_OFFLOAD_PIN" HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=5 HIPFIRE_PREFILL_CHUNK_ROWS=4096 -- "$S/bin-beta/qwen4_greedy_ids" --model "$MODEL" \
  --input "$S/fixtures/prompts.short.beta.jsonl" --output "$OUT/greedy-beta/ids.jsonl" --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
cell greedy-s1 390 390 "$H1_OFFLOAD_PIN" HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 -- "$S/bin-s1/qwen4_greedy_ids" --model "$MODEL" \
  --input "$S/fixtures/prompts.short.jsonl" --output "$OUT/greedy-s1/ids.jsonl" --receipt "$OUT/greedy-s1/receipts.jsonl" \
  --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
check greedy-cmp python3 "$HOLDS/verify_greedy.py" "$OUT/greedy-beta/ids.jsonl" "$OUT/greedy-s1/ids.jsonl" "$OUT/greedy-s1/receipts.jsonl"
check receipt-1b python3 "$R8_HOLDS/h1b_bridge.py" receipt1b "$S" "$OUT" "$A_OUT" "$HOLDS" "$EXEC_JSON" "$EXEC_SHA"
finish_summary
