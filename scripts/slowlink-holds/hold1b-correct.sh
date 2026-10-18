#!/usr/bin/env bash
# hold1b-correct.sh — HOLD 1b (Hold 1 completion: headroom RECHECK on the immutable 1a RAW state data + beta/S1 greedy), P1 correct, hard 1320 s.
# NOT SUBMITTED by this seat.
#   fabric-run.sh --class correct --card xtx --mem-gb $(read DEV) --hostmem-gb $(read HOST) --hostmem-lock --max-secs 1320 \
#                 --priority P1 --owner Main.SlowLinkS1 -- .../holds/hold1b-correct.sh [A_OUT]
#   A_OUT = the 1a run directory (or its handoff-1a.json); the argument, else S1_HOLD1A_OUT. Neither: FATAL before any GPU cell.
#   hold1b-correct.sh --print-memory   prints "22.4 54" (fabric exclusive device cap GiB, host GiB) from the declared final ledger-hold1.json (refuses if absent or not final)
#   hold1b-correct.sh --print-memory also names the separate PHYSICAL 24 GiB card on stderr (stdout stays the two figures "22.4 54").
#   Both halves are reserved at the SAME already-approved conservative 22.4 / 54: the host figure is the SHARED SEQUENTIAL ceiling (the immutable
#   beta arm 55,740,268,544 B + the 2 GiB allowance, the max of the arms, never their sum), NOT a sum of the per-cell S1 host owners' actuals.
#
# 1b runs only on an IMMUTABLE, verified 1a handoff. Before ANY GPU cell, `a-handoff` (h1_handoff.py verify) requires that the sealed 1a
# (handoff-1a.json, hold1a, qualifies, advance false) still matches the staged identity (same runtime source/tree, gate binary, kernel hashes,
# model, fixtures, profiles, bin-s1/bin-beta files, harness files and declared harness commit/tree), that every bound closed prim/state cell file
# is unchanged (size + sha256: the raw state gate.json, the external 20 ms sampler files, baseline.json, ...), and that prim and state both
# completed PASS from their closed files (a COMPLETED gate report; a partial progress sidecar never qualifies). A missing, changed, tampered,
# partial or wrong-identity 1a sets BLOCKED, so every later row is `skipped` and no GPU process starts. The verification output
# (A input paths, sha256s and identity) is the a-handoff row's stdout.txt and is repeated in 1b's own receipt.
#
# LAB-LINK POLICY (marginal XTX OCuLink x4 lab link): each cell is ONE process with ONE model load; the submitter SYNCs after shipping artifacts and
# runs with the kernel watcher on. A device bus loss in any cell: STOP, report the hold as INCOMPLETE (lab), not a code-quality failure; NO
# automatic retry (Main power-cycles the host; any further attempt is Main's decision).
#
# Every GPU cell runs under the external VRAM sampler next to the old 2 s host sampler (information only) and carries the H1-only explicit pin
# HIPFIRE_OFFLOAD_EXEC=pcie on BOTH greedy arms (hold1-common.sh; recorded in CELL/spawn-env.json and in the receipt), not a global helper pin.
# Rows, in order; a failure stops every dependent row:
#   a-handoff       CPU: verify the immutable 1a handoff (above). Failure blocks everything after it.
#   headroom-state  CPU RECHECK (1a already judged the same raw data right after its state cell as its REQUIRED state-headroom row; this is the
#                   unchanged independent re-judgment on the sealed copy): check_headroom.py on the 1a RAW state data ($A_OUT/state, the prim cell recorded as INFO only): min OBSERVED free VRAM over
#                   the WHOLE 1a state run >= 536,870,912 B from BOTH the external 20 ms sysfs samples (total - used, each sample's own total) AND
#                   the in-process hipMemGetInfo points pre_load/post_load/end of the COMPLETED state gate.json (each its own device_total).
#                   Missing/invalid/partial telemetry is INCOMPLETE (never zero-filled, never a pass); the progress sidecar is forensic only. The
#                   primitive peak is non-gating. No duplicate GPU state run exists. Failure/INCOMPLETE blocks every later row.
#   beta-input      CPU: prompts.short.beta.jsonl is prompts.short.jsonl minus ONLY resume_at (same records, order, ids, token-id arrays, messages); the
#                   stored proof must equal the identity recomputed from the two staged files. Failure blocks every later row.
#   greedy-beta     beta qwen4_greedy_ids on prompts.short.beta.jsonl (beta AR, immutable reference binary; the beta example loader needs a fixed
#                   expert layer count: HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10, a QUALITY-REFERENCE pin of the immutable beta arm only; it is NOT an S1
#                   resident floor and does not constrain the S1 arm's adaptive count, whose arithmetic is identical at any resident count). The immutable beta parser REJECTS the
#                   candidate input's `resume_at` key, so it reads the metadata-only adapter file; receipt label "beta cold AR vs S1 resumed".
#                   No beta binary change.
#   greedy-s1       candidate qwen4_greedy_ids (--receipt) on the canonical prompts.short.jsonl (the resume record resumes) with
#                   HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 at the product purpose's 768-row tile
#   greedy-cmp      CPU: exact ids + the route proof from the receipts (verify_greedy.py; the product tile 768 exactly; the natural record's token
#                   ids are resolved at the GPU load and compared as both arms' prompt_tokens)
#   receipt-1b      CPU: h1_handoff.py receipt1b writes 1b's own receipt-1b.json: the re-verified 1a handoff and bound files, the headroom verdict taken
#                   from the 1a raw data, both greedy arms' closed files with the pin applied, the hashes of BOTH greedy input files and the
#                   token-identity adapter proof, and the comparison output.
# PASS of 1b with a verified 1a is the Hold 1 ADVANCE candidate (hold-summary.json scope h1_complete_on_verified_1a, advance true); the 1a summary alone
# never advances. Coverage, cases and the greedy env policy are unchanged except the added explicit H1 pin.
# BUDGET (Main ruling, nothing cut): GPU caps greedy-beta 600 + greedy-s1 390 = 990 s, + 2 x 15 s timeout kill graces (hold-lib.sh KILL_GRACE, one
# per cell) + 2 x 90 s VRAM drains + 45 s finalize + 30 s CPU = 1275 s, hard maximum 1320 s (P1 correct). A cell's min start budget is its FULL cap (INCOMPLETE otherwise, never a narrowed run); a capped (timeout)
# cell fails with its raw stdout/stderr/rc/telemetry/progress sidecar preserved (collect_evidence.py); a missing cell or time is INCOMPLETE and no
# promotion. No automatic rerun exists. No GPU or submission until Main's readiness (the Mq2 drop regression report and the final seal).
HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HOLDS/hold-lib.sh"
. "$HOLDS/hold1-common.sh"
[ "${1:-}" = "--print-memory" ] && hold_memory 1
HOLD_SCOPE=h1_complete_on_verified_1a
HOLD_ADVANCE=true
hold_init hold1b 1320
S=$STAGE
REQUIRED="a-handoff headroom-state beta-input greedy-beta greedy-s1 greedy-cmp receipt-1b"
A_OUT=${1:-${S1_HOLD1A_OUT:-}}
[ -n "$A_OUT" ] || die "no 1a handoff: give the 1a run directory (or its handoff-1a.json) as the argument or S1_HOLD1A_OUT; 1b never runs without a verified immutable 1a"
A_OUT=$(readlink -f "$A_OUT")
if [ -f "$A_OUT" ]; then A_DIR=$(dirname "$A_OUT"); else A_DIR=$A_OUT; fi

# Nothing below this verification starts a GPU process unless the immutable 1a handoff is intact.
check a-handoff python3 "$HOLDS/h1_handoff.py" verify "$S" "$A_OUT" \
  || { [ -n "$BLOCKED" ] || BLOCKED="a-handoff failed: the 1a handoff is missing, changed, partial or of another identity (no GPU)"; }
# PRODUCT headroom verdict (CPU) over the whole 1a state run (raw data, verified just above); the primitive cell is recorded as information only.
headroom headroom-state "$A_DIR/state" "$A_DIR/prim"
check beta-input python3 "$HOLDS/beta_input.py" verify "$S/fixtures/prompts.short.jsonl" "$S/fixtures/prompts.short.beta.jsonl" \
  "$S/fixtures/beta-input-proof.json" \
  || { [ -n "$BLOCKED" ] || BLOCKED="beta-input failed: the beta greedy input is not the canonical input minus only resume_at"; }
cell greedy-beta 600 600 "$H1_OFFLOAD_PIN" HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10 -- "$S/bin-beta/qwen4_greedy_ids" --model "$MODEL" \
  --input "$S/fixtures/prompts.short.beta.jsonl" --output "$OUT/greedy-beta/ids.jsonl" --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
cell greedy-s1 390 390 "$H1_OFFLOAD_PIN" HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 -- "$S/bin-s1/qwen4_greedy_ids" --model "$MODEL" \
  --input "$S/fixtures/prompts.short.jsonl" --output "$OUT/greedy-s1/ids.jsonl" --receipt "$OUT/greedy-s1/receipts.jsonl" \
  --max-new-tokens 32 --max-seq "$MAX_SEQ" --thinking off
check greedy-cmp python3 "$HOLDS/verify_greedy.py" "$OUT/greedy-beta/ids.jsonl" "$OUT/greedy-s1/ids.jsonl" "$OUT/greedy-s1/receipts.jsonl"
check receipt-1b python3 "$HOLDS/h1_handoff.py" receipt1b "$S" "$OUT" "$A_OUT"
finish_summary
