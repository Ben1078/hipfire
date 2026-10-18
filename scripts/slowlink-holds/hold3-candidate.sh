#!/usr/bin/env bash
# hold3-candidate.sh — HOLD 3 (candidate code24 + final regression closure), P1 timing, <= 1800 s. NOT SUBMITTED by this seat.
#   fabric-run.sh --class timing --card xtx --mem-gb DEV --hostmem-gb HOST --hostmem-lock --max-secs 1800 --priority P1 \
#                 --owner Main.SlowLinkS1 -- .../holds/hold3-candidate.sh     (hold3-candidate.sh --print-memory: derived ledger)
# Requires hold 2 outputs: HOLD2_OUT (its run dir; code24-beta/code24.npy{,.json} is the paired beta row) and its hold-summary.
# A code change after hold 1 invalidates the affected hardware proof and still consumes the used hold; a changed binary,
# a timeout or a missing full corpus is INCOMPLETE, not permission for a fourth hold (Main must approve any further hold).
#
# Cells:
#   code24-s1   candidate qwen4_kld s11ref --decode, full code24, S1 forced on, SAME config/refs as hold 2's beta row
#   kld-pair    CPU: qwen4_kld pair (per-sequence means, paired candidate-minus-beta 95% t interval; lo<=0 else hi<=+0.010;
#               all 24 sequences / 6120 rows, same reference/corpus/config identity)
#   product     qwen4_slowlink_gate --mode product --route layer-major (natural-text greedy ids + decoded text; a human reads the
#               text, the attractor heuristic is only a floor)
#   state       qwen4_slowlink_gate --mode state --route layer-major (an independent same-route determinism observation, abort/
#               session/graph proofs on the final binary); with hold 1 and the hold 2 S1 runs this is the >= 3rd observation
# Automatic-vs-forced route equality is proved only after enablement is proposed (the selector is default-OFF until promotion):
# not a cell of this hold.
HOLDS=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
. "$HOLDS/hold-lib.sh"
[ "${1:-}" = "--print-memory" ] && hold_memory 3
hold_init hold3 1800
S=$STAGE
GATE=$S/bin-s1/qwen4_slowlink_gate
: "${HOLD2_OUT:?HOLD2_OUT (hold 2 run dir)}"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); r={c['name']:c['status'] for c in d['cells']}; sys.exit(0 if r.get('code24-beta')=='pass' else 1)" \
  "$HOLD2_OUT/hold-summary.json" || die "hold 2 has no passing code24-beta row to pair against"
BASE=$HOLD2_OUT/code24-beta/code24.npy
for f in "$BASE" "$BASE.json"; do [ -s "$f" ] || die "$f missing"; done
for f in "$REFS/code24.fp8.s11ref:$CODE24_REF_MD5" "$REFS/code24.corpus.json:$CODE24_CORPUS_MD5"; do
  [ "$(md5sum "${f%%:*}" | cut -d' ' -f1)" = "${f##*:}" ] || die "${f%%:*} md5 differs from the pinned reference"
done
REQUIRED="code24-s1 kld-pair product state"
cell code24-s1 1300 700 HIPFIRE_QWEN4_SLOWLINK_PREFILL=1 -- "$S/bin-s1/qwen4_kld" s11ref --model "$MODEL" \
  --ref "$REFS/code24.fp8.s11ref" --corpus "$REFS/code24.corpus.json" --output "$OUT/code24-s1/code24.npy" --decode
check kld-pair "$S/bin-s1/qwen4_kld" pair --base "$BASE" --cand "$OUT/code24-s1/code24.npy" \
  --corpus "$REFS/code24.corpus.json" --output "$OUT/kld-pair.json"
cell product 260 150 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/product/gate.json" \
  --mode product --route layer-major --manifest "$S/manifests/manifest-hold3.json"
cell state 300 180 -- "$GATE" --model "$MODEL" --fixture "$S/fixtures/gate-fixture.json" --output "$OUT/state/gate.json" \
  --mode state --route layer-major --manifest "$S/manifests/manifest-hold3.json"
finish_summary
