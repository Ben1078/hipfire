#!/usr/bin/env bash
# hold1a-correct.sh — HOLD 1a (PARTIAL Hold 1: primitives + real schedule state), P1 correct, hard 2640 s. NOT SUBMITTED by this seat.
#   fabric-run.sh --class correct --card xtx --mem-gb $(read DEV) --hostmem-gb $(read HOST) --hostmem-lock --max-secs 2640 \
#                 --priority P1 --owner Main.SlowLinkS1 -- .../holds/hold1a-correct.sh
#   hold1a-correct.sh --print-memory   prints "22.4 54" (fabric exclusive device cap GiB, host GiB) from the declared final ledger-hold1.json (refuses if absent or not final)
#   hold1a-correct.sh --print-memory also names the separate PHYSICAL 24 GiB card on stderr (stdout stays the two figures "22.4 54").
#   Both halves (1a and 1b) are reserved at the SAME already-approved conservative 22.4 / 54: the host figure is the SHARED SEQUENTIAL ceiling (the
#   immutable beta arm 55,740,268,544 B + the 2 GiB allowance, the max of the arms, never their sum), NOT a sum of the per-cell S1 host owners'
#   actuals (product 55,077,640,192 B, primitive_selftest 55,076,327,424 B including each purpose's 2 GiB bounded host snapshot cap, one cell at a time).
#
# Hold 1 is TWO runs (Main ruling): 1a = prim + state + the immediate state-headroom judgment (this script), 1b = a RE-CHECK of headroom on the sealed
# 1a RAW data + beta/S1 greedy (hold1b-correct.sh). A 1a PASS is
# PARTIAL Hold 1: hold-summary.json says scope partial_h1_prim_state and advance false; nothing advances on it. Its only product is the sealed
# immutable handoff A_OUT/handoff-1a.json (h1_handoff.py seal: the staged identity + every closed prim/state cell file by sha256, including the
# raw state gate.json, the external 20 ms sampler files and baseline.json; never the run root logs/summary, never itself) that 1b must verify before
# ANY GPU cell. 1a's REQUIRED list is prim, state and state-headroom: the immediate headroom judgment on the state cell's own raw data is a REQUIRED
# pass row (a headroom failure or INCOMPLETE fails/blocks 1a at once, so no handoff is sealed and 1b is never started on it); the seal is an extra
# CPU check row (a failing seal fails the hold).
#
# LAB-LINK POLICY (marginal XTX OCuLink x4 lab link): each cell is ONE process with ONE model load (no repeated load inside a process; no
# retry cell exists); the submitter SYNCs after shipping artifacts and runs with the kernel watcher on. A device bus loss in any cell:
# STOP, report the hold as INCOMPLETE (lab), not a code-quality failure and never a quality claim from the interrupted cell; NO automatic
# retry (Main power-cycles the host; any further attempt is Main's decision).
#
# Every GPU cell runs under the external VRAM sampler (vram_sampler.py: sysfs mem_info_vram_used/total only, requested 20 ms, started
# before the command, stopped after the drain) next to the old 2 s host sampler (sample_mem.py, information only). Every cell carries the
# H1-only explicit pin HIPFIRE_OFFLOAD_EXEC=pcie (hold1-common.sh; recorded in CELL/spawn-env.json), not a global helper pin.
# Cells, in order; a failure stops every dependent cell (small primitive kill first):
#   prim         qwen4_slowlink_gate --mode primitives --route layer-major   (7 production primitive cases on real host experts at the
#                primitive_selftest purpose's 512-row tile: kernel COVERAGE only, never a product-shape claim; identity/load/teardown must
#                pass; its memory census/residual/peak are INFORMATION only, never a pass/fail term)
#   state        qwen4_slowlink_gate --mode state --route layer-major        (PRODUCT purpose, tile 768: real 16K one-window schedule, irregular/
#                tail/resume/boundary/abort fixtures, same-route repeat digests, eager + retained-body replay decode, head rows, injections).
#                (the product VRAM headroom verdict is taken on THIS cell's raw data by the next row, no second GPU state run)
#   state-headroom CPU, REQUIRED: the EXISTING headroom judge (hold-lib.sh headroom -> check_headroom.py, threshold unchanged 536,870,912 B = 512 MiB)
#                on the state cell's RAW data immediately after the cell, BEFORE the seal: min OBSERVED free VRAM over the whole state run from BOTH the
#                external 20 ms sampler AND the in-process hipMemGetInfo pre_load/post_load/end points of the COMPLETED state gate.json. The prim cell
#                is passed as INFO only: the primitive peak/census is diagnostic and NEVER a headroom gate. rc 1 fails the hold, rc 2 (missing/invalid
#                telemetry) is INCOMPLETE; either blocks the seal. CPU only (a few seconds inside the 30 s CPU allowance); no added GPU cell, case, cut,
#                cap or drain. 1b re-runs the same judge on the SEALED copy of the same raw files (h1_handoff.py verify binds them) as a recheck.
#   handoff-seal CPU: h1_handoff.py seal (only when prim, state and state-headroom all passed; a partial progress sidecar never qualifies a cell)
# PROFILES: one manifest carries both admitted purpose profiles (memory_ledger.profiles.product / .primitive_selftest). The prim cell loads the
# primitive_selftest purpose (tile 512, diagnostic reserve); the state cell is the product purpose at tile 768 with no diagnostic reserve. The
# generic 1536-row limit remains only a historical upper bound of the receipt limits. Coverage, cases and env policy are unchanged.
# WORKLOAD (CPU source counts, NOT measurements). prim: 7 primitive cases. state (product tile 768): prefill = 288,182 token-id rows
# + 5N natural-text prompt tokens over 87 API prefill calls (5 of them natural): the prompt loop 133,983 id rows + 3N natural (x3 repeats
# of the 8 token-id prompts and the natural-text prompt), the extra marker prefill 23,272, the 16K boundary prefill 40,975, the failure
# injections 40,800 (upper bound) and the head rows 49,152 (N = the natural prompt's token count, resolved by the tokenizer at the load).
# Decode: at most 300 steps = cold_commit greedy 16 each for the 7 boundaries, cold + restored (7 x 2 x 16 = 224 forwards) + the natural
# replay / eager decode (<= 76). The older "27x32 decode steps" count was wrong. The 16K window is NOT the 768-row mixer tile: each selected
# API call of <= 16K rows is one sweep unless more than 4 snapshot cuts split it, and the 9,000-row marker CAPTURE and the 8,492-row marker
# COLD call carry 5 marker cuts each, i.e. 2 sweeps each (56 potential S1 windows for the non-natural calls). Exact per-case counts: the
# histogram supplied by the budget owner (not restated here).
# BUDGET (Main ruling, nothing cut or narrowed): GPU caps prim 420 + state 1860 = 2280 s, + 2 x 15 s timeout kill graces (hold-lib.sh KILL_GRACE, one per
# cell) + 2 x 90 s VRAM drains + 45 s finalize + 30 s CPU = 2565 s worst case, + the existing 75 s contingency = hard maximum 2640 s (P1 correct;
# 44 min, not a clamp to 2400). A cell's min start budget is its FULL cap: with less than that left it is INCOMPLETE (not a narrowed run); a
# capped (timeout) cell fails with its raw stdout/stderr/rc/telemetry/progress sidecar preserved (collect_evidence.py); a missing cell or time is
# INCOMPLETE; no automatic rerun exists.
# PRIM CAP RATIONALE (a budget, NOT a measurement of this build): the prim cell is ONE process whose WHOLE 420 s cap holds the model load, all 7
# primitive cases and teardown; no load-only timer exists, so every load second is counted. The only completed cold-load figure is the OLD non-Q4L
# loader's 194.3 s measured on this host (cold proxy; Q4L is not in this build and no completed cold load of this build was ever measured);
# plus ~18.5 s of old-work case/control overhead (an inference) = 212.8 s; with a 30% planning margin 276.64 s < 420 s. The earlier 80-110 s range
# was a warm/part-warm planning figure, NOT a cold-load number, and the former 180 s cap (j1892: rc 124 at 181 s with load_start but no load_end)
# could not hold even the cold proxy. The model files are prewarmed OUTSIDE the hold (nice 19 / ionice -c3, no GPU, no locking; not charged here).
# The measured load duration (same-run monotonic load_start -> load_end, null when censored) is recorded NON-GATING by collect_evidence.py in
# CELL/load_phase.json; no verdict reads it.
# No GPU or submission until Main's readiness (the Mq2 drop regression report and the final ledger/identity seal).
HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HOLDS/hold-lib.sh"
. "$HOLDS/hold1-common.sh"
[ "${1:-}" = "--print-memory" ] && hold_memory 1
HOLD_SCOPE=partial_h1_prim_state
HOLD_ADVANCE=false
hold_init hold1a 2640
S=$STAGE
REQUIRED="prim state state-headroom"
GATE=$S/bin-s1/qwen4_slowlink_gate

cell prim 420 420 "$H1_OFFLOAD_PIN" -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/prim/gate.json" \
  --mode primitives --route layer-major --manifest "$S/manifests/manifest-hold1.json"
cell state 1860 1860 "$H1_OFFLOAD_PIN" -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/state/gate.json" \
  --mode state --route layer-major --manifest "$S/manifests/manifest-hold1.json"
# PRODUCT headroom verdict (CPU, existing judge) on the state cell's raw data right now; the primitive cell is recorded as information only.
# A failure/INCOMPLETE sets BLOCKED inside headroom(), so the seal below is skipped and the hold fails/is incomplete.
headroom state-headroom "$OUT/state" "$OUT/prim"
# Seal ONLY the immutable closed cell files (the collectors finished inside cell()) and the staged identity; the run root logs and the summary
# (which the EXIT trap / finish_summary may still append to) are not bound.
check handoff-seal python3 "$HOLDS/h1_handoff.py" seal "$S" "$OUT"
finish_summary
