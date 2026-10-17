#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_fixtures.py OUTDIR NATURAL_TEXT_FILE

CPU-only, deterministic. Writes the byte-identical prompt fixtures every compared cell consumes:

  OUTDIR/gate-fixture.json    hipfire.qwen4.slowlink_fixture.v1 (qwen4_slowlink_gate)
  OUTDIR/gate-fixture-anchor.json  the same fixture with the 16384-row prompt replaced by the shipped `hipfire bench` synthetic ids
                              (10 + (i % 1000), daemon bench_prefill) perf_rows [16384] and perf_repeats 2 (second sample = warm-matched): the incumbent anchor of hold 2
  OUTDIR/prompts.short.jsonl  qwen4_greedy_ids inputs for hold 1 (512, 1536, 1537, 8192 rows + an irregular resumed prefix)
  OUTDIR/prompts.full.jsonl   the same plus the 16384-row prompt
  OUTDIR/fixtures.md5         md5 of every file (record these in every cell's evidence)

Row prompts are LCG ids in [1000, 100000), the generator `tests/session_cache_hw.rs::prompt` uses, so they are random
token ids (a numerics/state fixture, not readable text). The boundary prompt is the same LCG stream with the explicit
marker id `boundary_token_id` (7) written at row cut-1 of each requested boundary (3, 4, 1536, 1537, 8192): metadata-only
insertion, recorded in the fixture and checked by the gate. The natural-text prompt carries `text` and is tokenized by the
gate with the artifact's own tokenizer; read its decoded output by eye (`--mode product`)."""
import hashlib
import json
import os
import sys

MAX_SEQ = 16640
BOUNDARY_TOKEN_ID = 7  # outside the LCG range [1000, 100000)
BOUNDARY_CUTS = (3, 4, 1536, 1537, 8192)


def prompt(seed, n):
    s, out = seed, []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append(1000 + (s >> 33) % 99_000)
    return out


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    out, text_file = argv[1], argv[2]
    os.makedirs(out, exist_ok=True)
    text = open(text_file, encoding="utf-8").read().strip()
    if len(text) < 2000:
        sys.exit("natural text must be at least 2000 characters")
    prompts = []
    for rows in (512, 1536, 1537, 8192, 16384):
        prompts.append({"id": f"p{rows}", "kind": "rows", "prompt_ids": prompt(0x5100 + rows, rows)})
    prompts.append({"id": "irregular", "kind": "irregular", "prompt_ids": prompt(0x5201, 4100), "launch_cuts": [1000, 2500, 4001]})
    prompts.append({"id": "resume", "kind": "resume", "prompt_ids": prompt(0x5202, 3400), "entry": 1537})
    # Synthetic state fixture, identical on every arm: BOUNDARY_TOKEN_ID (outside the LCG id range, so it occurs only
    # where placed) sits at row cut-1 of every requested boundary. The gate sets it as the message-end token
    # (set_turn_end_token + set_turn_snapshots), so each cut becomes a real SessionCache snapshot boundary. The ids are
    # carried as written; the gate verifies them and never rewrites them.
    boundary = prompt(0x5203, 9000)
    for cut in BOUNDARY_CUTS:
        boundary[cut - 1] = BOUNDARY_TOKEN_ID
    prompts.append({"id": "boundary", "kind": "boundary", "prompt_ids": boundary, "requested_boundaries": list(BOUNDARY_CUTS)})
    prompts.append({"id": "natural", "kind": "natural", "text": text})
    fixture = {
        "schema": "hipfire.qwen4.slowlink_fixture.v1",
        "max_seq": MAX_SEQ,
        "decode_tokens": 32,
        "repeats": 3,
        "perf_repeats": 1,
        "head_rows": 4,
        "session_cache_bytes": 2 << 30,
        "prompts": prompts,
        "continuation_ids": prompt(0x5300, 300),
        "failure_injections": ["before_dma", "during_panel", "after_layer", "before_commit"],
        "boundary_token_id": BOUNDARY_TOKEN_ID,
    }
    with open(os.path.join(out, "gate-fixture.json"), "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")

    anchor = json.loads(json.dumps(fixture))
    for p in anchor["prompts"]:
        if p["id"] == "p16384":
            p["prompt_ids"] = [10 + (i % 1000) for i in range(16384)]
    anchor["perf_rows"] = [16384]
    # warm-matched ANCHOR ONLY: the beta bench does one full-length warm prefill and then one timed run, so the anchor
    # times two full prefills and anchor_compare uses the SECOND. The natural 3+3 arms keep perf_repeats 1 (first cold).
    anchor["perf_repeats"] = 2
    with open(os.path.join(out, "gate-fixture-anchor.json"), "w") as f:
        json.dump(anchor, f, indent=1)
        f.write("\n")

    def records(include_16k):
        rows = []
        for p in prompts:
            if p["kind"] == "natural" or (p["id"] == "p16384" and not include_16k):
                continue
            if p["kind"] == "boundary":
                continue
            rec = {"id": p["id"], "prompt_ids": p["prompt_ids"]}
            if p["kind"] == "resume":
                rec["resume_at"] = p["entry"]
            rows.append(rec)
        rows.append({"id": "natural", "messages": [{"role": "user", "content": text[:6000]}]})
        return rows

    for name, full in (("prompts.short.jsonl", False), ("prompts.full.jsonl", True)):
        with open(os.path.join(out, name), "w") as f:
            for rec in records(full):
                f.write(json.dumps(rec) + "\n")
    with open(os.path.join(out, "fixtures.md5"), "w") as f:
        for name in sorted(os.listdir(out)):
            if name != "fixtures.md5":
                f.write(f"{hashlib.md5(open(os.path.join(out, name), 'rb').read()).hexdigest()}  {name}\n")
    print(open(os.path.join(out, "fixtures.md5")).read(), end="")


if __name__ == "__main__":
    main(sys.argv)
