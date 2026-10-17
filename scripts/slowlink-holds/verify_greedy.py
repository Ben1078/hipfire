#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""verify_greedy.py BETA_IDS.jsonl S1_IDS.jsonl S1_RECEIPTS.jsonl

Exact greedy equality between the beta autoregressive arm and the S1 candidate (full token_ids, finish_reason, prompt
token count, record order and ids), plus the route proof from the candidate's receipts: every prefill call longer than
one 1,536-row tile ran layer-major with at least ten resident layers, at most 38 host layers, a tile of at most 1,536
rows, contiguous windows and one payload sweep per window. Exit 0 only when everything holds; the md5 of both id files
is printed so the evidence names the exact bytes compared."""
import hashlib
import json
import sys

TILE = 1536


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
            if rows > TILE:
                if c.get("route") != "layer-major":
                    bad.append(f"{label} call {i}: {rows} rows ran {c.get('route')} (bypass: {c.get('bypass_reason')})")
                    continue
                if c.get("resident_layers", 0) < 10 or c.get("host_layers", 99) > 38:
                    bad.append(f"{label} call {i}: residency {c.get('resident_layers')}/{c.get('host_layers')}")
                if not 0 < c.get("tile_rows", 0) <= TILE or c.get("retained_scratch_rows", 1 << 30) > TILE:
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
    if bad:
        print("FAIL")
        for line in bad:
            print(" ", line)
        return 1
    print(f"PASS: {len(s1)} records, exact ids, layer-major receipts valid")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
