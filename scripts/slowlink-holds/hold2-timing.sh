#!/usr/bin/env bash
# hold2-timing.sh — HOLD 2 (matched full-path A/B + baseline code24), P1 timing, <= 1800 s. NOT SUBMITTED by this seat.
# STATUS: INFORMATIONAL pre-repair draft, NOT an approved timing handoff. It was written for the legacy `perf` mode and the legacy H2 anchor
# shape; it is not the official CLI timing path and is not updated for the per-purpose (product 768 / primitive_selftest 512) profiles,
# whose product timing is Main's decision. Its --print-memory reads the same per-purpose ledger (ledger_profiles.py). Do not submit it.
# R5 D2 RULE: every cell below runs with HIPFIRE_QWEN4_D2_DEFAULT=0 (hold-lib.sh cell_env), i.e. the recorded NON-DEFAULT arm. The old H2
# no-env-default acceptance ("shipped default auto placement, no env pin", HOLD2-PP16384-ACCEPTANCE.json) is BLOCKED by the current rule: a future H2 result
# is reported as NON-DEFAULT (D2 off) until Main lifts the ban, and is never labelled a shipped-default measurement.
#   fabric-run.sh --class timing --card xtx --mem-gb DEV --hostmem-gb HOST --hostmem-lock --max-secs 1800 --priority P1 \
#                 --owner Main.SlowLinkS1 -- .../holds/hold2-timing.sh        (hold2-timing.sh --print-memory: derived ledger)
# Requires hold 1 ADVANCE (the hold script refuses without HOLD1_SUMMARY pointing at the passing hold1b hold-summary.json whose advance is true).
#
# Cells (fresh process each; fresh-process repetitions, not --reps inside one loaded model):
#   perf-{inc,s1}-{1,2,3}  qwen4_slowlink_gate --mode perf; the incumbent arm is the SAME candidate binary forced to --route incumbent
#                          (kill switch 0 -> the contemporary incumbent policy), alternating arm order across the three pairs
#                          (inc,s1) (s1,inc) (inc,s1). The fixture carries the byte-identical prompt ids (md5 in identity.json).
#                          pp512 / pp8192 guards and a greedy decode sample are inside perf.
#   code24-beta            (beta example loader: fixed HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10) qwen4_kld s11ref --decode, full code24 (24 sequences, 6120 rows) on the IMMUTABLE beta reference binary
#                          against the existing fp8 reference; gets all remaining time. If it cannot fit the remaining budget the
#                          cell is INCOMPLETE with exact unfinished evidence; it is NEVER reduced to a subset or replaced by an
#                          old-binary result.
#   anchor-beta-16k        ONE immutable beta `hipfire bench MODEL --matrix --pp 16384 --max-seq 16640 --ctx 128 --tg 1 --runs 1 --json`
#                          (beta CLI + beta daemon, pinned by bin-beta/pins.json sha256; the shipped product carrier path with
#                          daemon synthetic ids 10+(i%1000); --max-seq fixes the load context at 16640 exactly as the gate fixture,
#                          --ctx 128 keeps the decode row short; the CLI's own untimed warm prefill is included in the cell's time)
#   anchor-inc-16k         the candidate gate on gate-fixture-anchor.json (the SAME ids, context, state format, PLE flags), --route
#                          incumbent with perf_repeats 2 (warm-matched ANCHOR ONLY: the SECOND full prefill is compared to the beta
#                          singleton, because the beta CLI warms with a full 16K prefill); anchor-compare (CPU) requires the two within the declared tolerance (ANCHOR_TOL, with ANCHOR_TOL_RATIONALE; no default) and records the
#                          timing-bracket differences. The anchor is IN ADDITION to the natural 3+3 arms, never a substitute.
#   Forcing HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10 is forbidden in every timing/anchor cell (it is a quality-reference-only pin).
# Verdict is produced on the CPU afterwards from the cell outputs (perf receipts: target S1 median >= 1400 tok/s with all three
# observations and dispersion; >= 10% over the incumbent; same-route first-logit/state/generated-id hashes equal across the three
# S1 runs). Report tok/s against BOTH 1400 and 2000. If floor + ordered work cannot reach 1400, record the miss; default stays off.
HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HOLDS/hold-lib.sh"
[ "${1:-}" = "--print-memory" ] && hold_memory 2
hold_init hold2 1800
S=$STAGE
GATE=$S/bin-s1/qwen4_slowlink_gate
[ -n "${HOLD1_SUMMARY:-}" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['status']=='pass' and d.get('advance') is True else 1)" "$HOLD1_SUMMARY" \
  || die "hold 1 did not ADVANCE (set HOLD1_SUMMARY to the passing hold1b hold-summary.json with advance true; a 1a summary is partial and never advances); a timing hold never defends unproven output"
for f in "$REFS/code24.fp8.s11ref:$CODE24_REF_MD5" "$REFS/code24.corpus.json:$CODE24_CORPUS_MD5"; do
  [ "$(md5sum "${f%%:*}" | cut -d' ' -f1)" = "${f##*:}" ] || die "${f%%:*} md5 differs from the pinned reference"
done
REQUIRED="perf-inc-1 perf-s1-1 anchor-beta-16k anchor-inc-16k anchor-compare perf-s1-2 perf-inc-2 perf-inc-3 perf-s1-3 code24-beta"
perf() { # ARM PAIR
  cell "perf-$1-$2" 330 200 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" \
    --output "$OUT/perf-$1-$2/gate.json" --mode perf --route "$( [ "$1" = s1 ] && echo layer-major || echo incumbent )" \
    --manifest "$S/manifests/manifest-hold2.json"
}
# The anchor tolerance is DECLARED by Main with its noise rationale (e.g. from the dispersion of repeated incumbent runs); no default.
[ -n "${ANCHOR_TOL:-}" ] && [ -n "${ANCHOR_TOL_RATIONALE:-}" ] || die "ANCHOR_TOL and ANCHOR_TOL_RATIONALE must be declared before the hold (no default noise tolerance)"
# The beta anchor needs the pinned beta CLI + daemon (immutable binaries): refuse without matching pins.
[ -x "$S/bin-beta/hipfire" ] && [ -x "$S/bin-beta/daemon" ] && [ -r "$S/bin-beta/pins.json" ] || die "bin-beta/{hipfire,daemon,pins.json} missing: no beta anchor without the pinned beta CLI and daemon"
python3 - "$S/bin-beta" <<'PY' || die "beta CLI/daemon sha256 differ from bin-beta/pins.json"
import hashlib, json, os, sys
d = sys.argv[1]
pins = json.load(open(os.path.join(d, "pins.json")))
for name in ("hipfire", "daemon"):
    h = hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest()
    if pins.get(name) != h:
        sys.exit(f"{name}: {h} != pinned {pins.get(name)}")
PY
# A failed arm blocks the rest: a timing hold never continues past a failure.
perf inc 1; perf s1 1
cell anchor-beta-16k 330 200 HIPFIRE_DAEMON_BIN="$S/bin-beta/daemon" -- "$S/bin-beta/hipfire" bench "$MODEL" --matrix \
  --pp 16384 --max-seq 16640 --ctx 128 --tg 1 --runs 1 --json
cell anchor-inc-16k 300 180 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture-anchor.json" \
  --output "$OUT/anchor-inc-16k/gate.json" --mode perf --route incumbent --manifest "$S/manifests/manifest-hold2.json"
check anchor-compare python3 "$HOLDS/anchor_compare.py" "$OUT/anchor-beta-16k/stdout.txt" "$OUT/anchor-inc-16k/gate.json" "$ANCHOR_TOL" "$ANCHOR_TOL_RATIONALE"
perf s1 2;  perf inc 2
perf inc 3; perf s1 3
cell code24-beta 1400 600 HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10 -- "$S/bin-beta/qwen4_kld" s11ref --model "$MODEL" --ref "$REFS/code24.fp8.s11ref" \
  --corpus "$REFS/code24.corpus.json" --output "$OUT/code24-beta/code24.npy" --decode
finish_summary
