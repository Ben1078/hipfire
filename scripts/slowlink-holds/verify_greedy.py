#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""verify_greedy.py BETA_IDS.jsonl S1_IDS.jsonl S1_RECEIPTS.jsonl

Exact greedy equality between the beta autoregressive arm and the S1 candidate (full token_ids, finish_reason, prompt
token count, record order and ids), plus the route proof from the candidate's receipts: every prefill call longer than
the 1,536-row route threshold (unchanged) ran layer-major on a placement whose resident and host layers close on the 48-layer trunk with at
least one host layer (NO resident floor and no host cap: the count is adaptive, ten preferred; every call of one load reports the same split), a tile of EXACTLY
768 rows (the product purpose's tile), retained scratch of at most 768 rows, contiguous windows and one payload sweep per window.
Exit 0 only when everything holds; the md5 of both id files is printed so the evidence names the exact bytes compared.

Comparison label (printed): "beta cold AR vs S1 resumed". The beta arm reads prompts.short.beta.jsonl, the candidate canonical
prompts.short.jsonl minus ONLY the `resume_at` key (beta_input.py: the immutable beta parser rejects it), so the beta arm prefills the
resume prompt's identical token ids cold while the candidate resumes. Both arms' prompt_tokens (the natural-text record included,
tokenized at the GPU load) are compared below."""
import hashlib
import json
import sys

ROUTE_MIN_ROWS = 1536  # prefill calls longer than this must have taken the S1 layer-major route (the historical threshold, unchanged)
PRODUCT_TILE = 768  # the product purpose's tile (ledger_profiles.PROFILE_TILE['product']); the primitive purpose's 512 is never a greedy-arm tile
TRUNK_LAYERS = 48  # model data: resident + host layers of every layer-major receipt
LABEL = "beta cold AR vs S1 resumed (beta input = canonical minus only resume_at)"


def lines(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw, [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    beta_raw, beta = lines(argv[1])
    s1_raw, s1 = lines(argv[2])
    _, receipts = lines(argv[3])
    bad = []
    splits = set()
    print(LABEL)
    print("beta md5", hashlib.md5(beta_raw).hexdigest(), "s1 md5", hashlib.md5(s1_raw).hexdigest())
    if not beta or len(beta) != len(s1) or len(receipts) != len(s1):
        bad.append(f"record counts differ or are empty: beta {len(beta)} s1 {len(s1)} receipts {len(receipts)}")
    for b, s, r in zip(beta, s1, receipts):
        label = json.dumps(b.get("id"))
        for key in ("id", "token_ids", "finish_reason", "prompt_tokens"):
            if b.get(key) != s.get(key):
                bad.append(f"{label}: {key} differs")
        if r.get("id") != s.get("id"):
            bad.append(f"{label}: receipt id {r.get('id')!r} does not match")
        calls = r.get("slowlink") or []
        if not calls:
            bad.append(f"{label}: no route receipt")
        for i, c in enumerate(calls):
            rows = c.get("prompt_rows", 0)
            if rows > ROUTE_MIN_ROWS:
                if c.get("route") != "layer-major":
                    bad.append(f"{label} call {i}: {rows} rows ran {c.get('route')} (bypass: {c.get('bypass_reason')})")
                    continue
                res, host = c.get("resident_layers"), c.get("host_layers")
                if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (res, host)) or res + host != TRUNK_LAYERS or host < 1:
                    bad.append(f"{label} call {i}: residency {res}/{host} does not close on {TRUNK_LAYERS} trunk layers with a host layer")
                else:
                    splits.add((res, host))
                if c.get("tile_rows") != PRODUCT_TILE or c.get("retained_scratch_rows", 1 << 30) > PRODUCT_TILE:
                    bad.append(f"{label} call {i}: tile/scratch rows {c.get('tile_rows')}/{c.get('retained_scratch_rows')}")
                ws = c.get("windows", [])
                cursor = 0
                for w in ws:
                    if w.get("start") != cursor:
                        bad.append(f"{label} call {i}: windows not contiguous")
                        break
                    cursor += w.get("rows", 0)
                if cursor != rows or c.get("sweeps") != len(ws):
                    bad.append(f"{label} call {i}: windows cover {cursor}/{rows} rows, sweeps {c.get('sweeps')} for {len(ws)} window(s)")
    if len(splits) > 1:
        bad.append(f"layer-major receipts report different resident/host splits {sorted(splits)}: one load has one placement")
    if bad:
        print("FAIL")
        for line in bad:
            print(" ", line)
        return 1
    print(f"PASS: {len(s1)} records, exact ids, layer-major receipts valid at the {PRODUCT_TILE}-row product tile ({LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
