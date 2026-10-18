#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""beta_input.py write CANONICAL BETA PROOF | verify CANONICAL BETA [PROOF]

Metadata-only input adapter for the immutable beta reference arm of Hold 1 (the `greedy-beta` cell, label "beta cold AR vs S1 resumed").

The candidate's canonical greedy input (prompts.short.jsonl) carries `resume_at` on its resume record (the S1 candidate resumes that
prompt from a cached prefix). The immutable beta qwen4_greedy_ids parser REJECTS any key outside {id, messages, prompt_ids, tools,
category}, so it cannot read that record. The beta input is therefore generated FROM the candidate canonical file by deleting exactly one
key, `resume_at`, from the records that carry it, and nothing else: every record, its order, its id, its key order, its prompt_ids array and
its messages are byte-for-byte the canonical values. No prompt is dropped or rewritten, the beta binary is not modified and no environment
opt-out is involved; the beta arm simply prefills the same token ids cold (autoregressive) while the candidate resumes. This adapter is
evidence-labelled, never a quality claim about the resume path itself.

verify refuses unless: the record counts, order and ids are identical; every beta record equals its canonical record minus ONLY `resume_at`
(same keys in the same order, equal values); every beta key is one the immutable parser allows; no canonical key outside the allowed set
plus `resume_at` exists (it would be silently dropped); and per record the token-id arrays (count and sha256 of the compact JSON array) and
messages (sha256 of the canonical JSON) are identical. The proof (schema s1.beta_input_adapter.v1) records both file hashes and that per-record
identity. A natural-text record has no token array at generation time: its identity is its identical messages; the tokenizer resolves its
ids at the GPU load and verify_greedy.py later compares both arms' prompt_tokens (no token array is invented here)."""
import hashlib
import json
import os
import sys

SCHEMA = "s1.beta_input_adapter.v1"
LABEL = "beta cold AR vs S1 resumed"
ALLOWED_BETA_KEYS = ("id", "messages", "prompt_ids", "tools", "category")
STRIPPED_KEYS = ("resume_at",)


def bad(msg):
    sys.exit(f"beta input adapter: {msg}")


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            bad(f"a JSON record repeats the key {k!r}")
        out[k] = v
    return out


def read_records(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        bad(f"{path} unreadable: {e}")
    records = []
    for n, line in enumerate(raw.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line, object_pairs_hook=_pairs)
        except ValueError as e:
            bad(f"{path} line {n} is not JSON: {e}")
        if not isinstance(rec, dict):
            bad(f"{path} line {n} is not an object")
        records.append(rec)
    if not records:
        bad(f"{path} has no records")
    return raw, records


def beta_view(rec):
    """The canonical record minus ONLY the stripped keys (key order preserved)."""
    return {k: v for k, v in rec.items() if k not in STRIPPED_KEYS}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _file_digest(path, raw, records):
    return {"name": os.path.basename(path), "sha256": hashlib.sha256(raw).hexdigest(), "md5": hashlib.md5(raw).hexdigest(),
            "bytes": len(raw), "records": len(records)}


def write_beta(canonical, beta):
    """Generate the beta input from the canonical file (strip only resume_at)."""
    _, records = read_records(canonical)
    with open(beta, "w") as f:
        for rec in records:
            f.write(json.dumps(beta_view(rec)) + "\n")


def verify(canonical, beta):
    """Proof dict, or exit with the first violated rule."""
    c_raw, c = read_records(canonical)
    b_raw, b = read_records(beta)
    if len(c) != len(b):
        bad(f"record counts differ: canonical {len(c)} beta {len(b)} (no record may be dropped)")
    rows = []
    for i, (cr, br) in enumerate(zip(c, b)):
        label = json.dumps(cr.get("id"))
        if cr.get("id") != br.get("id"):
            bad(f"record {i}: id {br.get('id')!r} != canonical {cr.get('id')!r} (order/ids must be identical)")
        extra = sorted(k for k in cr if k not in ALLOWED_BETA_KEYS and k not in STRIPPED_KEYS)
        if extra:
            bad(f"{label}: canonical keys {extra} are neither allowed by the immutable beta parser nor stripped by this adapter")
        unknown = sorted(k for k in br if k not in ALLOWED_BETA_KEYS)
        if unknown:
            bad(f"{label}: beta keys {unknown} are rejected by the immutable beta parser")
        if list(br) != [k for k in cr if k not in STRIPPED_KEYS] or br != beta_view(cr):
            bad(f"{label}: beta record is not the canonical record minus only {list(STRIPPED_KEYS)} (a field, its order or a value differs)")
        stripped = {k: cr[k] for k in STRIPPED_KEYS if k in cr}
        row = {"id": cr.get("id"), "stripped": stripped, "keys": list(br)}
        if "prompt_ids" in cr:
            ids = json.dumps(cr["prompt_ids"], separators=(",", ":"))
            if json.dumps(br["prompt_ids"], separators=(",", ":")) != ids:
                bad(f"{label}: prompt_ids differ")
            row["prompt_ids"] = {"count": len(cr["prompt_ids"]), "sha256": _sha(ids)}
        if "messages" in cr:
            msg = json.dumps(cr["messages"], sort_keys=True, separators=(",", ":"))
            if json.dumps(br["messages"], sort_keys=True, separators=(",", ":")) != msg:
                bad(f"{label}: messages differ")
            row["messages"] = {"sha256": _sha(msg)}
            if "prompt_ids" not in cr:
                row["token_ids"] = "resolved by the tokenizer at the GPU load; both arms' prompt_tokens are compared later by verify_greedy.py"
        rows.append(row)
    return {"schema": SCHEMA, "label": LABEL, "stripped_keys": list(STRIPPED_KEYS), "allowed_beta_keys": list(ALLOWED_BETA_KEYS),
            "canonical": _file_digest(canonical, c_raw, c), "beta": _file_digest(beta, b_raw, b), "records": rows,
            "note": "the beta input is the candidate canonical input minus only resume_at; no record, field order, token id or message differs"}


def main(argv):
    if len(argv) == 5 and argv[1] == "write":
        write_beta(argv[2], argv[3])
        proof = verify(argv[2], argv[3])
        with open(argv[4], "w") as f:
            json.dump(proof, f, indent=1)
            f.write("\n")
        return 0
    if len(argv) in (4, 5) and argv[1] == "verify":
        proof = verify(argv[2], argv[3])
        if len(argv) == 5:
            try:
                with open(argv[4]) as f:
                    stored = json.load(f)
            except (OSError, ValueError) as e:
                bad(f"stored proof {argv[4]} unreadable: {e}")
            if stored != proof:
                bad(f"stored proof {argv[4]} differs from the recomputed identity proof")
        print(json.dumps(proof, indent=1))
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
