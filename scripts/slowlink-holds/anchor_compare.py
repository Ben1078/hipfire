#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""anchor_compare.py BETA_BENCH.json GATE_PERF.json TOLERANCE "NOISE RATIONALE"

Hold 2's incumbent anchor: ONE immutable beta CLI+daemon `hipfire bench ... --pp 16384 --json` process versus the candidate
binary's `qwen4_slowlink_gate --mode perf --route incumbent` on the anchor fixture (the same synthetic ids 10+(i%1000), the same
max_seq 16640, state format and PLE startup flags). Both are the contemporary auto 8192-chunk incumbent, so they must agree within
the DECLARED noise tolerance (no default: the caller states the number and the noise rationale, both recorded); a larger gap means the candidate binary's incumbent arm is not the beta incumbent
and no A/B against it may be claimed.

Known timing-bracket differences, recorded in the output, never corrected for: the beta bench brackets the prefill with
device_synchronize and includes the final-logit allocation/free; the gate includes the state reset and the final-logit D2H. Both
include the output head.

WARM-MATCHED, ANCHOR ONLY: the beta CLI runs one untimed full-length warm prefill and then one timed run, so the anchor fixture has
perf_repeats 2 and this script compares the gate's SECOND full prefill (the first is its warm) to the beta singleton. The natural
timed arms of hold 2 are unaffected: each of them reports the FIRST (cold) prefill of its process."""
import json
import sys


def main(argv):
    if len(argv) != 5 or not argv[4].strip():
        sys.exit(__doc__)
    tol = float(argv[3])
    if not 0 < tol < 1:
        sys.exit("tolerance must be a fraction in (0, 1)")
    beta = json.load(open(argv[1]))
    gate = json.load(open(argv[2]))
    rows = [r for r in beta.get("prefill", []) if r.get("tokens") == 16384]
    if len(rows) != 1 or len(rows[0].get("samples", [])) != 1:
        sys.exit(f"beta bench JSON must hold exactly one pp16384 sample, got {rows!r}")
    beta_tok_s = float(rows[0]["samples"][0])
    prefill = [p for p in gate.get("evidence", {}).get("prefill", []) if p.get("rows") == 16384]
    if gate.get("status") != "pass" or len(prefill) != 1:
        sys.exit(f"gate perf report is not a passing run with one pp16384 row (status {gate.get('status')!r})")
    samples = prefill[0].get("tok_s", [])
    if len(samples) != 2:
        sys.exit(f"anchor fixture must time exactly two full prefills (perf_repeats 2), got {len(samples)}")
    gate_tok_s = float(samples[1])  # the SECOND prefill: the first was the warm
    if not (beta_tok_s > 0 and gate_tok_s > 0):
        sys.exit("non-positive tok/s")
    ratio = gate_tok_s / beta_tok_s
    out = {"beta_tok_s": beta_tok_s, "gate_incumbent_tok_s": gate_tok_s, "ratio": ratio, "tolerance": tol, "tolerance_rationale": argv[4],
           "gate_first_cold_tok_s": float(samples[0]), "scope": "ANCHOR ONLY, warm-matched (gate second prefill vs beta timed run)",
           "within_tolerance": abs(ratio - 1.0) <= tol,
           "bracket_differences": ["beta: device_synchronize bracket incl. final-logit alloc/free; gate: incl. reset + final-logit D2H",
                                   "beta: untimed full-length warm then 1 timed; gate: 1,537-row warm, full warm prefill, then the compared second prefill",
                                   "both include the output head"]}
    print(json.dumps(out, indent=1))
    return 0 if out["within_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
