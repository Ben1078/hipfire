#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q1_judge.py -- CPU-only Q1 quality judge for the native `qwen4_slowlink_quality` dumps.  Python stdlib only.

    q1_judge.py q1a --plan PLAN --expected-plan-sha256 SHA --baseline-dir D --s1a-dir D --score-dir D [--s1b-dir D] --out FILE
    q1_judge.py f32 --plan PLAN --expected-plan-sha256 SHA --beta-f32-dir D --s1-f32-dir D --score-f32-dir D
                    --q1a-summary FILE --expected-q1a-summary-sha256 SHA --out FILE

STATUS.  SOURCE-ONLY: never executed against real native output by its author.  No GPU, no model, no journalctl.

WHAT IS TRUSTED AND WHAT IS RECOMPUTED.  The KL numbers are the NATIVE ones (`kl.jsonl` / `kl-summary.json` of the native `score` / `score-f32`
phases): this file contains NO probability / softmax math.  It re-verifies everything else from the dumps themselves: every phase directory is
`complete:true` and bound to the plan sha, the binary sha, the model pin and the effective environment / load facts of its route; every cell's
`.logits.f32le` stream sha256, rows-file sha256, per-row logits sha256, `bin_offset`, and the top-1/top-2 logit bits at the named index are
re-read from the bytes; prefill call structure and receipt checks are checked per route; prompt / teacher hashes are recomputed from the PLAN
arrays; and every native KL row is cross-checked against BOTH rows files (position, next token, top-1 ids, margins).  What this file computes
itself: counts, bit-exact row comparisons, determinism, the descriptive block-bootstrap intervals and the verdict.

PRIMARY GATE (Q1).  Per-context mean KL(beta||S1) <= 0.010 nats for EACH of ctx16k and ctx4k AND for the pooled mean (equal weights).  The
95% intervals (deterministic stratified moving-block bootstrap: block length 32, 512 positions per context, 10000 replicates, splitmix64 seeded
from a fixed label, linear-interpolated percentiles) are DESCRIPTIVE ONLY: positions inside a context are one serially correlated teacher-forced
sequence and there are only two contexts, so the interval is neither an iid interval nor a generalisation claim, and its upper bound never gates.
Top-1 agreement is reported with counts / rates / intervals; NO threshold is invented (Q2 pending) and `human_review_required` is always true.

VERDICT.  FAIL = (provenance clean and primary data valid and the gate fails or an infinite-KL row exists) or (provenance clean and determinism
data valid and not every one of the 7 cases is bit-identical).  PASS = no issue anywhere, gate passes, determinism identical.  Everything else is
INCOMPLETE (missing / null / duplicate rows, bad provenance, determinism pending).  `default_on_recommended` is ALWAYS false.  The resume context
class is REPORTED (`resume_disposition`) and never converted into a pass/fail, a "copy bug" or a "rounding benign" call; `f32_exoneration` stays
NONE_UNTIL_Q1F32.

EXIT CODES.  0 PASS (f32: REPORTED) / 1 FAIL / 3 INCOMPLETE / 2 refused (bad args, plan sha mismatch, unreadable plan; nothing written)."""
import argparse
import hashlib
import json
import math
import os
import re
import stat as statmod
import struct
import sys

SCHEMA_Q1A = "hipfire.q1.q1a_summary.v1"
SCHEMA_F32 = "hipfire.q1.q1f32_summary.v1"
SCHEMA_DET = "hipfire.q1.determinism_check.v1"
PLAN_SCHEMA = "hipfire.qwen4.slowlink_quality.plan.v1"
SUMMARY_SCHEMA = "hipfire.qwen4.slowlink_quality.phase_summary.v1"
KL_SUMMARY_SCHEMA = "hipfire.qwen4.slowlink_quality.kl_summary.v1"
KL_ROW_SCHEMA = "hipfire.qwen4.slowlink_quality.kl_row.v1"

GATE_NATS = 0.010
BOOT_BLOCK = 32
BOOT_REPS = 10000
BOOT_LABEL = "hipfire.q1.moving-block-bootstrap.v1"
BOOT_CONFIDENCE = 0.95
KL_NEG_TOLERANCE = 1e-9
SUM_REL_TOLERANCE = 1e-9
MAX_JSON_BYTES = 512 << 20

CI_NOTE = ("DESCRIPTIVE ONLY. Positions within a context are one contiguous, serially correlated teacher-forced sequence and there are only two "
           "contexts: this stratified moving-block bootstrap interval is NOT an iid confidence interval and says nothing about generalisation "
           "to other prompts. It never gates; the verdict uses the point means only.")
TOP1_POLICY = {"threshold": None, "q2_pending": True, "human_review_required": True,
               "note": "No top-1 agreement threshold has been agreed (Q2 pending); counts and rates are reported for human review only."}

PHASES = ("baseline", "s1a", "s1b", "beta-f32", "s1-f32")
ARM = {"baseline": "baseline", "beta-f32": "baseline", "s1a": "s1", "s1b": "s1", "s1-f32": "s1"}
STATE_F32 = {"baseline": False, "s1a": False, "s1b": False, "beta-f32": True, "s1-f32": True}
STATE_QUANT_ENV = "HIPFIRE_STATE_QUANT"

Q8_PAIRS = ("primary-ctx16k", "primary-ctx4k", "resume-s1split", "resume-s1cold", "resume-s1cold-vs-s1split", "resume-betasplit",
            "resume-s1split-vs-betasplit", "resume-s1cold-vs-betasplit")
RESUME_PAIRS = tuple(p for p in Q8_PAIRS if p.startswith("resume-"))
F32_PAIR = "resume-f32-s1split-vs-betacold"
Q8_CONTRAST_PAIR = "resume-s1split"


class JudgeError(Exception):
    """The judge cannot even start (bad plan, plan sha mismatch, unreadable summary)."""


# ----------------------------------------------------------------------------------------------------------------------------------
# strict JSON, hashing, files
# ----------------------------------------------------------------------------------------------------------------------------------
def _no_dup(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _bad_const(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _parse_float(text):
    v = float(text)
    if not math.isfinite(v):
        raise ValueError(f"non-finite JSON number {text}")
    return v


def loads_strict(data):
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    return json.loads(data, object_pairs_hook=_no_dup, parse_constant=_bad_const, parse_float=_parse_float)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def seal(doc):
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    body["receipt_sha256"] = sha256_bytes(canon(body))
    return body


def seal_ok(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("receipt_sha256"), str):
        return False
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    return sha256_bytes(canon(body)) == doc["receipt_sha256"]


def u32_sha256(ids):
    return sha256_bytes(struct.pack(f"<{len(ids)}I", *ids))


def read_regular(path, limit=MAX_JSON_BYTES):
    st = os.lstat(path)
    if not statmod.S_ISREG(st.st_mode):
        raise JudgeError(f"{path} is not a regular file")
    with open(path, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise JudgeError(f"{path} is larger than {limit} bytes")
    return data


def read_json_file(path):
    try:
        raw = read_regular(path)
        doc = loads_strict(raw)
    except (OSError, ValueError) as e:
        raise JudgeError(f"{path}: {e}") from e
    return doc, sha256_bytes(raw)


def read_jsonl_file(path):
    """(records, sha256). Every non-empty line is one strict JSON object."""
    try:
        raw = read_regular(path)
    except OSError as e:
        raise JudgeError(f"{path}: {e}") from e
    out = []
    for n, line in enumerate(raw.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            rec = loads_strict(line)
        except ValueError as e:
            raise JudgeError(f"{path} line {n}: {e}") from e
        if not isinstance(rec, dict):
            raise JudgeError(f"{path} line {n}: not an object")
        out.append(rec)
    return out, sha256_bytes(raw)


def write_json_new(path, doc):
    """Create `path` (never overwrite), fsync, return the sha256 of the bytes written."""
    data = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256_bytes(data)


def _is_int(v):
    return type(v) is int


def _is_num(v):
    return type(v) in (int, float) and math.isfinite(v)


def margin_witness_equal(a, b):
    """The `top12_margin` DISPLAY witness copied by the native `score` into kl.jsonl (`reference_margin` / `candidate_margin`) against the rows
    file. serde_json without float_roundtrip can re-parse an f64 decimal one ULP away, so equal or an immediate IEEE-754 neighbour (math.nextafter)
    is accepted; 2+ ULP, a non-finite value, a bool or a non-number is rejected. Used for these two metadata fields ONLY: never for KL values or the
    gate, logit bytes, hashes, ids, positions or counts (those stay exact; the native KL sums have their own named SUM_REL_TOLERANCE)."""
    if not (_is_num(a) and _is_num(b)):
        return False
    a, b = float(a), float(b)
    return a == b or b == math.nextafter(a, math.inf) or b == math.nextafter(a, -math.inf)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Issues:
    """Categorised problems. Any entry means: not PASS."""

    CATS = ("provenance", "primary", "resume", "determinism", "f32")

    def __init__(self):
        self.by = {c: [] for c in self.CATS}

    def add(self, cat, msg):
        self.by[cat].append(msg)

    def any(self, *cats):
        return any(self.by[c] for c in (cats or self.CATS))

    def dump(self):
        return {c: list(v) for c, v in self.by.items()}


# ----------------------------------------------------------------------------------------------------------------------------------
# plan view -> expected cells
# ----------------------------------------------------------------------------------------------------------------------------------
def load_plan(path, expected_sha256):
    """(plan dict, sha256). The sha is checked against the independently supplied one; the structure is the minimum the judge needs."""
    if not HEX64.match(expected_sha256 or ""):
        raise JudgeError("--expected-plan-sha256 must be 64 lowercase hex")
    try:
        raw = read_regular(path)
    except OSError as e:
        raise JudgeError(f"plan {path}: {e}") from e
    sha = sha256_bytes(raw)
    if sha != expected_sha256:
        raise JudgeError(f"plan sha256 {sha} != expected {expected_sha256}")
    try:
        plan = loads_strict(raw)
    except ValueError as e:
        raise JudgeError(f"plan {path}: {e}") from e
    problems = []
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise JudgeError(f"plan schema is not {PLAN_SCHEMA}")
    try:
        ex = plan["expect"]
        for k in ("resident_layers", "baseline_tile_rows", "s1_tile_rows", "s1_layer_major_min_rows"):
            if not _is_int(ex[k]):
                problems.append(f"expect.{k} is not an integer")
        if not isinstance(ex["env"], dict) or not isinstance(ex["baseline_env"], dict) or not isinstance(ex["env_absent"], list):
            problems.append("expect.env / baseline_env / env_absent have the wrong type")
        art = plan["artifact"]
        if not (isinstance(art["path"], str) and _is_int(art["size"]) and HEX64.match(art["sha256"])):
            problems.append("artifact {path,size,sha256} malformed")
        det = plan["determinism"]
        if len(det["case_ids"]) != 7 or len(det["cases"]) != 7 or [c["id"] for c in det["cases"]] != det["case_ids"]:
            problems.append("determinism.case_ids/cases must be 7 in the same order")
        if not _is_int(det["max_new_tokens"]):
            problems.append("determinism.max_new_tokens malformed")
        prim = plan["primary"]
        if not _is_int(prim["heldout_rows"]) or not _is_int(prim["greedy_new_tokens"]) or len(prim["contexts"]) != 2:
            problems.append("primary malformed (2 contexts required)")
        for c in prim["contexts"]:
            if not all(isinstance(c[k], list) for k in ("prefix_ids", "heldout_ids")) or not all(HEX64.match(c[k]) for k in ("prefix_sha256", "heldout_sha256")):
                problems.append(f"primary context {c.get('id')} malformed")
            if len(c["heldout_ids"]) != prim["heldout_rows"]:
                problems.append(f"primary context {c['id']}: heldout_ids length != heldout_rows")
        res = plan["resume"]
        if not (isinstance(res["prompt_ids"], list) and isinstance(res["teacher_ids"], list) and _is_int(res["split_at"])
                and HEX64.match(res["prompt_sha256"]) and HEX64.match(res["teacher_sha256"])):
            problems.append("resume malformed")
    except (KeyError, TypeError) as e:
        raise JudgeError(f"plan is missing/has a malformed field: {e!r}") from e
    if problems:
        raise JudgeError("plan: " + "; ".join(problems))
    return plan, sha


def chunk_ranges(n, chunk):
    out, s = [], 0
    while s < n:
        e = min(s + chunk, n)
        out.append((s, e))
        s = e
    return out


def expected_effective_env(plan, phase):
    ex = plan["expect"]
    env = dict(ex["env"])
    if ARM[phase] == "baseline":
        env.update(ex["baseline_env"])
    if STATE_F32[phase]:
        env[STATE_QUANT_ENV] = "f32"
    return env


def expected_specs(plan, phase):
    """Ordered {cell name: spec}. spec: kind, context, case, prompt_tokens, prompt_sha256|None, teacher_sha256|None, feed|None, nexts|None, max_new|None,
    rows|None (None = greedy), calls, split_at."""
    arm = ARM[phase]
    ex = plan["expect"]
    specs = {}
    res = plan["resume"]
    prim = plan["primary"]
    det = plan["determinism"]

    def primary_calls(n):
        return chunk_ranges(n, ex["baseline_tile_rows"]) if arm == "baseline" else [(0, n)]

    def add_det():
        for case in det["cases"]:
            ids = case.get("prompt_ids")
            n = len(ids) if ids is not None else case["expected_prompt_tokens"]
            at = case.get("resume_at")
            specs[f"det-{case['id']}"] = {
                "kind": "determinism", "context": None, "case": case["id"], "prompt_tokens": n,
                "prompt_sha256": u32_sha256(ids) if ids is not None else None, "teacher_sha256": None, "feed": None, "nexts": None,
                "max_new": det["max_new_tokens"], "rows": None, "calls": [(0, at), (at, n)] if at else [(0, n)], "split_at": None,
                "retained": case.get("retained_ids")}

    def add_primary():
        for c in prim["contexts"]:
            n = len(c["prefix_ids"])
            specs[f"primary-{c['id']}-teacher"] = {
                "kind": "primary-teacher", "context": c["id"], "case": None, "prompt_tokens": n, "prompt_sha256": u32_sha256(c["prefix_ids"]),
                "teacher_sha256": u32_sha256(c["heldout_ids"]), "feed": c["heldout_ids"][:-1], "nexts": list(c["heldout_ids"]),
                "max_new": None, "rows": len(c["heldout_ids"]), "calls": primary_calls(n), "split_at": None}
            specs[f"primary-{c['id']}-greedy"] = {
                "kind": "primary-greedy", "context": c["id"], "case": None, "prompt_tokens": n, "prompt_sha256": u32_sha256(c["prefix_ids"]),
                "teacher_sha256": None, "feed": None, "nexts": None, "max_new": prim["greedy_new_tokens"], "rows": None,
                "calls": primary_calls(n), "split_at": None}

    def resume(split):
        n = len(res["prompt_ids"])
        specs["resume-split" if split else "resume-cold"] = {
            "kind": "resume", "context": None, "case": None, "prompt_tokens": n, "prompt_sha256": u32_sha256(res["prompt_ids"]),
            "teacher_sha256": u32_sha256(res["teacher_ids"]), "feed": list(res["teacher_ids"]), "nexts": list(res["teacher_ids"]) + [None],
            "max_new": None, "rows": len(res["teacher_ids"]) + 1,
            "calls": [(0, res["split_at"]), (res["split_at"], n)] if split else [(0, n)], "split_at": res["split_at"] if split else None}

    if phase == "baseline":
        add_primary()
        resume(False)
        resume(True)
    elif phase == "s1a":
        add_det()
        add_primary()
        resume(True)
        resume(False)
    elif phase == "s1b":
        add_det()
    elif phase == "beta-f32":
        resume(False)
    elif phase == "s1-f32":
        resume(True)
    else:
        raise JudgeError(f"unknown phase {phase}")
    return specs


# ----------------------------------------------------------------------------------------------------------------------------------
# one phase directory: summary + every cell re-verified from the bytes
# ----------------------------------------------------------------------------------------------------------------------------------
def _check_receipt_checks(rec, spec, arm, ex, name, issues, where):
    pre = rec.get("prefill")
    if not isinstance(pre, dict):
        issues.add("provenance", f"{where}: cell {name} has no prefill record")
        return
    calls = pre.get("calls")
    got = [(c.get("start"), c.get("end")) for c in calls] if isinstance(calls, list) and all(isinstance(c, dict) for c in calls) else None
    if got != spec["calls"]:
        issues.add("provenance", f"{where}: cell {name} prefill calls {got} != expected {spec['calls']}")
        return
    if isinstance(calls, list) and any(c.get("rows") != c["end"] - c["start"] for c in calls):
        issues.add("provenance", f"{where}: cell {name} prefill call rows != end-start")
    receipts, checks = pre.get("receipts"), pre.get("receipt_checks")
    if not (isinstance(receipts, list) and isinstance(checks, list) and len(receipts) == len(checks) == len(spec["calls"])):
        issues.add("provenance", f"{where}: cell {name} receipts/receipt_checks do not match the {len(spec['calls'])} prefill calls")
        return
    tile = ex["baseline_tile_rows"] if arm == "baseline" else ex["s1_tile_rows"]
    for (s, e), chk in zip(spec["calls"], checks):
        if not isinstance(chk, dict):
            issues.add("provenance", f"{where}: cell {name} receipt check is not an object")
            continue
        bad = []
        if chk.get("entry_pos") != s or chk.get("prompt_rows") != e - s:
            bad.append("entry_pos/prompt_rows")
        if chk.get("tile_rows") != tile or chk.get("retained_scratch_rows") != tile:
            bad.append(f"tile {chk.get('tile_rows')}/{chk.get('retained_scratch_rows')} != {tile}")
        if arm == "baseline":
            if chk.get("route") != "incumbent" or chk.get("selected_by") != "env0":
                bad.append(f"route {chk.get('route')!r}/selected_by {chk.get('selected_by')!r} != incumbent/env0")
        else:
            if chk.get("selected_by") not in ("env1", "auto"):
                bad.append(f"selected_by {chk.get('selected_by')!r}")
            if chk.get("route") == "layer-major":
                if chk.get("status") != "ok" or chk.get("resident_layers") != ex["resident_layers"]:
                    bad.append(f"layer-major status {chk.get('status')!r} resident_layers {chk.get('resident_layers')!r}")
            elif chk.get("route") == "incumbent":
                if e - s >= ex["s1_layer_major_min_rows"]:
                    bad.append(f"a {e - s}-row call took the incumbent route")
            else:
                bad.append(f"route {chk.get('route')!r}")
        if bad:
            issues.add("provenance", f"{where}: cell {name} receipt check: " + "; ".join(bad))


def verify_cell(dirpath, name, rec, spec, arm, ex, issues, where):
    """Re-read the cell from its bytes. Returns {"rec", "rows": [row dicts]} or None when it cannot be used."""
    n0 = sum(len(v) for v in issues.by.values())

    def bad(msg):
        issues.add("provenance", f"{where}: cell {name}: {msg}")

    for key in ("kind", "prompt_tokens", "prompt_sha256", "rows", "vocab", "bin", "bin_bytes", "rows_file", "rows_file_sha256", "stream_sha256", "final_position"):
        if key not in rec:
            bad(f"record lacks {key}")
            return None
    if rec.get("cell") != name:
        bad("record cell name differs")
    if rec["kind"] != spec["kind"]:
        bad(f"kind {rec['kind']!r} != {spec['kind']!r}")
    if spec["context"] is not None and rec.get("context") != spec["context"]:
        bad(f"context {rec.get('context')!r} != {spec['context']!r}")
    if rec.get("arm") != arm:
        bad(f"arm {rec.get('arm')!r} != {arm!r}")
    if rec["prompt_tokens"] != spec["prompt_tokens"]:
        bad(f"prompt_tokens {rec['prompt_tokens']} != {spec['prompt_tokens']}")
    if spec["prompt_sha256"] is not None and rec["prompt_sha256"] != spec["prompt_sha256"]:
        bad("prompt_sha256 differs from the plan arrays")
    if spec["teacher_sha256"] is not None and rec.get("teacher_sha256") != spec["teacher_sha256"]:
        bad("teacher_sha256 differs from the plan arrays")
    if spec["feed"] is not None and rec.get("teacher_feed_tokens") != len(spec["feed"]):
        bad(f"teacher_feed_tokens {rec.get('teacher_feed_tokens')} != {len(spec['feed'])}")
    if spec["case"] is not None and rec.get("case_id") != spec["case"]:
        bad(f"case_id {rec.get('case_id')!r} != {spec['case']!r}")
    if spec["kind"] == "resume" and rec.get("split_at") != spec["split_at"]:
        bad(f"split_at {rec.get('split_at')!r} != {spec['split_at']!r}")
    _check_receipt_checks(rec, spec, arm, ex, name, issues, where)

    rows_n, vocab = rec["rows"], rec["vocab"]
    if not (_is_int(rows_n) and _is_int(vocab) and rows_n > 0 and vocab > 1):
        bad("rows/vocab malformed")
        return None
    ids = None
    if spec["rows"] is None:                                         # greedy
        ids = rec.get("token_ids")
        if not (isinstance(ids, list) and all(_is_int(t) and t >= 0 for t in ids)):
            bad("greedy cell without integer token_ids")
            return None
        if rec.get("token_ids_sha256") != u32_sha256(ids):
            bad("token_ids_sha256 does not match token_ids")
        fin = rec.get("finish_reason")
        if fin == "length":
            if len(ids) != spec["max_new"]:
                bad(f"finish length with {len(ids)} ids (max_new {spec['max_new']})")
        elif fin == "stop":
            if not 1 <= len(ids) <= spec["max_new"]:
                bad(f"finish stop with {len(ids)} ids")
        else:
            bad(f"finish_reason {fin!r}")
        if rows_n != len(ids) + 1:
            bad(f"rows {rows_n} != ids+1 ({len(ids) + 1})")
        nexts = list(ids) + [None]
        feed = list(ids)
    else:
        if rows_n != spec["rows"]:
            bad(f"rows {rows_n} != {spec['rows']}")
        nexts, feed = spec["nexts"], spec["feed"]
    if rec["final_position"] != rec["prompt_tokens"] + rows_n - 1:
        bad(f"final_position {rec['final_position']} != prompt_tokens+rows-1")
    if rec["bin"] != f"{name}.logits.f32le" or rec["rows_file"] != f"{name}.rows.jsonl":
        bad("bin/rows_file names are not the canonical ones")
        return None
    bin_path, rows_path = os.path.join(dirpath, rec["bin"]), os.path.join(dirpath, rec["rows_file"])
    rowbytes = vocab * 4
    try:
        size = os.lstat(bin_path).st_size
    except OSError as e:
        bad(f"bin unreadable: {e}")
        return None
    if size != rows_n * rowbytes or rec["bin_bytes"] != size:
        bad(f"bin size {size} / bin_bytes {rec['bin_bytes']} != rows*vocab*4 = {rows_n * rowbytes}")
        return None
    try:
        rows, rows_sha = read_jsonl_file(rows_path)
    except JudgeError as e:
        bad(str(e))
        return None
    if rows_sha != rec["rows_file_sha256"]:
        bad("rows file sha256 differs from the cell record")
    if len(rows) != rows_n:
        bad(f"rows file has {len(rows)} lines, record says {rows_n}")
        return None
    stream = hashlib.sha256()
    try:
        with open(bin_path, "rb") as f:
            for i, row in enumerate(rows):
                buf = f.read(rowbytes)
                if len(buf) != rowbytes:
                    bad(f"short read at row {i}")
                    return None
                stream.update(buf)
                if row.get("row") != i or row.get("cell") != name:
                    bad(f"row {i}: row/cell fields differ")
                if row.get("bin_offset") != i * rowbytes:
                    bad(f"row {i}: bin_offset {row.get('bin_offset')} != {i * rowbytes}")
                if row.get("logits_sha256") != sha256_bytes(buf):
                    bad(f"row {i}: logits_sha256 differs from the bytes")
                if row.get("context_position") != rec["prompt_tokens"] + i:
                    bad(f"row {i}: context_position {row.get('context_position')} != {rec['prompt_tokens'] + i}")
                want_in = None if i == 0 else feed[i - 1]
                if row.get("input_token") != want_in:
                    bad(f"row {i}: input_token {row.get('input_token')!r} != {want_in!r}")
                if row.get("next_token") != nexts[i]:
                    bad(f"row {i}: next_token {row.get('next_token')!r} != {nexts[i]!r}")
                for which in ("top1", "top2"):
                    idx, bits = row.get(f"{which}_id"), row.get(f"{which}_logit_bits")
                    if not (_is_int(idx) and 0 <= idx < vocab and _is_int(bits)):
                        bad(f"row {i}: {which} id/bits malformed")
                    elif struct.unpack_from("<I", buf, 4 * idx)[0] != bits:
                        bad(f"row {i}: {which}_logit_bits differ from the dumped logit at {which}_id {idx}")
                m = row.get("top12_margin")
                if not _is_num(m) or m < 0:
                    bad(f"row {i}: top12_margin {m!r} is not a finite non-negative number")
                if not (_is_int(row.get("neg_inf_count")) and row["neg_inf_count"] >= 0):
                    bad(f"row {i}: neg_inf_count malformed")
            if f.read(1):
                bad("bin has trailing bytes")
    except OSError as e:
        bad(f"bin unreadable: {e}")
        return None
    if stream.hexdigest() != rec["stream_sha256"]:
        bad("stream sha256 differs from the bytes")
    if sum(len(v) for v in issues.by.values()) != n0:
        return {"rec": rec, "rows": rows, "ok": False}
    return {"rec": rec, "rows": rows, "ok": True}


def _load_summary(dirpath, phase, plan, plan_sha, issues):
    where = phase
    path = os.path.join(dirpath, "phase-summary.json")
    try:
        summary, sha = read_json_file(path)
    except JudgeError as e:
        issues.add("provenance", f"{where}: phase-summary.json unreadable (an incomplete phase has none): {e}")
        return None, None
    ex, art = plan["expect"], plan["artifact"]

    def bad(msg):
        issues.add("provenance", f"{where}: phase-summary: {msg}")

    if summary.get("schema") != SUMMARY_SCHEMA:
        bad(f"schema {summary.get('schema')!r}")
    if summary.get("complete") is not True:
        bad("complete is not true")
    if summary.get("phase") != phase or summary.get("arm") != ARM[phase] or summary.get("state_f32") is not STATE_F32[phase]:
        bad(f"phase/arm/state_f32 {summary.get('phase')!r}/{summary.get('arm')!r}/{summary.get('state_f32')!r} are not "
            f"{phase!r}/{ARM[phase]!r}/{STATE_F32[phase]!r}")
    pl = summary.get("plan") or {}
    if pl.get("sha256") != plan_sha:
        bad("plan sha256 differs from the supplied plan")
    binary = summary.get("binary") or {}
    mcontent = (summary.get("manifest") or {}).get("content") or {}
    if not HEX64.match(str(binary.get("sha256"))) or binary.get("sha256") != (mcontent.get("binary") or {}).get("sha256"):
        bad("binary sha256 missing or differs from the manifest content")
    model = summary.get("model") or {}
    if model.get("pinned_sha256") != art["sha256"] or model.get("hashed_here") is not False or model.get("path") != art["path"]:
        bad("model pin/path does not match the plan artifact (or the model was hashed)")
    if (model.get("stat") or {}).get("size") != art["size"]:
        bad("model stat size differs from the plan artifact size")
    env = summary.get("environment")
    if not isinstance(env, dict):
        bad("no environment record")
    else:
        for name, want in expected_effective_env(plan, phase).items():
            if env.get(name) != want:
                bad(f"environment {name} = {env.get(name)!r}, expected {want!r}")
        if ARM[phase] == "s1":
            for name in ex["env_absent"]:
                if env.get(name) is not None:
                    bad(f"environment {name} = {env.get(name)!r}, must be absent")
    load = summary.get("load")
    if not isinstance(load, dict):
        bad("no load facts")
    else:
        arm = ARM[phase]
        want_place = "explicit-env" if arm == "baseline" else "slowlink-profile"
        if load.get("placement") != want_place:
            bad(f"placement {load.get('placement')!r} != {want_place!r}")
        if load.get("resident_layers") != ex["resident_layers"]:
            bad(f"resident_layers {load.get('resident_layers')!r} != {ex['resident_layers']}")
        if not (_is_int(load.get("host_mapped_tensors")) and load["host_mapped_tensors"] > 0):
            bad("host_mapped_tensors is not > 0")
        want_scratch = ex["baseline_tile_rows"] if arm == "baseline" else ex["s1_tile_rows"]
        if load.get("scratch_rows") != want_scratch:
            bad(f"scratch_rows {load.get('scratch_rows')!r} != {want_scratch}")
        if load.get("ple_format") != ex["ple_format"]:
            bad(f"ple_format {load.get('ple_format')!r} != {ex['ple_format']!r}")
        sf = load.get("state_format") or {}
        if sf.get("gdn") != ("F32" if STATE_F32[phase] else "Q8") or sf.get("qsa_kv") != "Q8":
            bad(f"state_format {sf!r}")
        if arm == "s1":
            prof = load.get("profile") or {}
            if prof.get("tile_rows") != ex["s1_tile_rows"] or prof.get("purpose") != "product" or prof.get("resident_layers") != ex["resident_layers"]:
                bad(f"S1 profile {prof!r}")
    return summary, sha


def load_phase(dirpath, phase, plan, plan_sha, issues):
    """Verified phase data or None. Every problem lands in issues['provenance']."""
    where = phase
    if not os.path.isdir(dirpath) or os.path.islink(dirpath):
        issues.add("provenance", f"{where}: {dirpath} is not a real directory")
        return None
    summary, summary_sha = _load_summary(dirpath, phase, plan, plan_sha, issues)
    if summary is None:
        return None
    specs = expected_specs(plan, phase)
    expected_names = list(specs)
    if summary.get("cell_order") != expected_names:
        issues.add("provenance", f"{where}: cell_order {summary.get('cell_order')} != expected {expected_names}")
    try:
        recs, _ = read_jsonl_file(os.path.join(dirpath, "cells.jsonl"))
    except JudgeError as e:
        issues.add("provenance", f"{where}: {e}")
        return None
    names = [r.get("cell") for r in recs]
    if names != expected_names:
        issues.add("provenance", f"{where}: cells.jsonl cells {names} != expected {expected_names}")
    by_name = {}
    for r in recs:
        if r.get("cell") in by_name:
            issues.add("provenance", f"{where}: duplicate cell {r.get('cell')}")
        by_name[r.get("cell")] = r
    done = summary.get("cells")
    if not isinstance(done, list) or [d.get("cell") for d in done] != expected_names:
        issues.add("provenance", f"{where}: summary cells list differs from the expected cells")
    else:
        for d in done:
            r = by_name.get(d["cell"])
            if r is not None and (d.get("rows") != r.get("rows") or d.get("stream_sha256") != r.get("stream_sha256")
                                  or d.get("prompt_tokens") != r.get("prompt_tokens") or d.get("finish_reason") != r.get("finish_reason")):
                issues.add("provenance", f"{where}: summary entry for {d['cell']} differs from cells.jsonl")
    cells = {}
    for name, spec in specs.items():
        rec = by_name.get(name)
        if rec is None:
            issues.add("provenance", f"{where}: cell {name} missing")
            continue
        got = verify_cell(dirpath, name, rec, spec, ARM[phase], plan["expect"], issues, where)
        if got is not None:
            cells[name] = got
    return {"dir": dirpath, "phase": phase, "summary": summary, "summary_sha256": summary_sha, "cells": cells, "specs": specs}


def provenance_core(pd):
    """Fields that must be identical between two phases of the same binary/plan/route (s1a vs s1b)."""
    s = pd["summary"]
    load = dict(s.get("load") or {})
    prof = dict(load.get("profile") or {})
    return {
        "plan_sha256": (s.get("plan") or {}).get("sha256"),
        "binary_sha256": (s.get("binary") or {}).get("sha256"),
        "source": ((s.get("manifest") or {}).get("content") or {}).get("source"),
        "model": {k: (s.get("model") or {}).get(k) for k in ("path", "pinned_sha256")} | {"stat": {k: v for k, v in ((s.get("model") or {}).get("stat") or {}).items() if k != "ctime_sec" and k != "ctime_nsec"}},
        "environment": s.get("environment"),
        "load": {k: load.get(k) for k in ("selector", "placement", "slowlink_profile", "resident_layers", "host_mapped_tensors", "scratch_rows", "ple_format", "state_format", "vocab", "eos_token", "num_layers", "arch_id", "gpu_arch")},
        "profile": {k: prof.get(k) for k in ("tile_rows", "resident_layers", "purpose")},
    }


# ----------------------------------------------------------------------------------------------------------------------------------
# native scoring outputs (kl.jsonl / kl-summary.json)
# ----------------------------------------------------------------------------------------------------------------------------------
def q8_pair_specs(plan):
    ids = [c["id"] for c in plan["primary"]["contexts"]]
    rows = plan["primary"]["heldout_rows"]
    resume_rows = len(plan["resume"]["teacher_ids"]) + 1
    out = {}
    for cid in ids:
        cell = f"primary-{cid}-teacher"
        out[f"primary-{cid}"] = (("baseline", cell), ("s1a", cell), rows)
    out["resume-s1split"] = (("baseline", "resume-cold"), ("s1a", "resume-split"), resume_rows)
    out["resume-s1cold"] = (("baseline", "resume-cold"), ("s1a", "resume-cold"), resume_rows)
    out["resume-s1cold-vs-s1split"] = (("s1a", "resume-cold"), ("s1a", "resume-split"), resume_rows)
    out["resume-betasplit"] = (("baseline", "resume-cold"), ("baseline", "resume-split"), resume_rows)
    out["resume-s1split-vs-betasplit"] = (("baseline", "resume-split"), ("s1a", "resume-split"), resume_rows)
    out["resume-s1cold-vs-betasplit"] = (("baseline", "resume-split"), ("s1a", "resume-cold"), resume_rows)
    return out


def f32_pair_specs(plan):
    return {F32_PAIR: (("beta-f32", "resume-cold"), ("s1-f32", "resume-split"), len(plan["resume"]["teacher_ids"]) + 1)}



def load_score(score_dir, specs, scope, phases, issues, cat_of):
    """Native KL pairs, cross-checked against the verified phase data. Returns {pair: report} for every usable pair.
    cat_of(pair) -> the Issues category a problem of that pair belongs to."""
    out = {}
    try:
        summary, summary_sha = read_json_file(os.path.join(score_dir, "kl-summary.json"))
        klrows, kl_sha = read_jsonl_file(os.path.join(score_dir, "kl.jsonl"))
    except JudgeError as e:
        for pair in specs:
            issues.add(cat_of(pair), f"score: {e}")
        return out, None
    if summary.get("scope") != scope:
        for pair in specs:
            issues.add(cat_of(pair), f"score: kl-summary scope {summary.get('scope')!r} != {scope!r}")
    pairs = summary.get("pairs")
    got_names = [p.get("pair") if isinstance(p, dict) else None for p in pairs] if isinstance(pairs, list) else None
    if got_names != list(specs):
        for pair in specs:
            issues.add(cat_of(pair), f"score: kl-summary pairs {got_names!r} != {list(specs)}")
        return out, {"kl_summary_sha256": summary_sha, "kl_jsonl_sha256": kl_sha}
    by_pair = {}
    for r in klrows:
        by_pair.setdefault(r.get("pair"), []).append(r)
    for extra in sorted(set(by_pair) - set(specs), key=str):
        issues.add("provenance", f"score: kl.jsonl carries the unexpected pair {extra!r}")
    for psum in pairs:
        name = psum["pair"]
        ref_key, cand_key, want_rows = specs[name]
        cat = cat_of(name)

        def bad(msg, name=name, cat=cat):
            issues.add(cat, f"pair {name}: {msg}")

        ref, cand = (phases.get(ref_key[0]) or {}).get("cells", {}).get(ref_key[1]), (phases.get(cand_key[0]) or {}).get("cells", {}).get(cand_key[1])
        if ref is None or cand is None or not ref["ok"] or not cand["ok"]:
            bad(f"reference {ref_key} or candidate {cand_key} cell is missing/unverified")
            continue
        if psum.get("schema") != KL_SUMMARY_SCHEMA:
            bad(f"summary schema {psum.get('schema')!r}")
        if (psum.get("reference") or {}).get("cell") != ref_key[1] or (psum.get("reference") or {}).get("stream_sha256") != ref["rec"]["stream_sha256"]:
            bad("reference cell/stream sha256 differs from the phase data")
        if (psum.get("candidate") or {}).get("cell") != cand_key[1] or (psum.get("candidate") or {}).get("stream_sha256") != cand["rec"]["stream_sha256"]:
            bad("candidate cell/stream sha256 differs from the phase data")
        if ref["rec"]["prompt_sha256"] != cand["rec"]["prompt_sha256"] or ref["rec"].get("teacher_sha256") != cand["rec"].get("teacher_sha256"):
            bad("reference/candidate prompt or teacher sha256 differ")
        if ref["rec"]["vocab"] != cand["rec"]["vocab"]:
            bad("vocab differs")
        rows = by_pair.get(name, [])
        if psum.get("rows") != want_rows or psum.get("expected_rows") != want_rows or psum.get("complete") is not True or len(rows) != want_rows:
            bad(f"rows {psum.get('rows')}/{psum.get('expected_rows')}/complete {psum.get('complete')!r}/kl.jsonl {len(rows)} != {want_rows}")
            continue
        if len(ref["rows"]) != want_rows or len(cand["rows"]) != want_rows:
            bad("phase cells do not have the pair's row count")
            continue
        kls, trace, agree, infinite, bit_same = [], [], 0, 0, 0
        for i, (k, r, c) in enumerate(zip(rows, ref["rows"], cand["rows"])):
            if k.get("schema") != KL_ROW_SCHEMA or k.get("row") != i:
                bad(f"row {i}: schema/row index")
            if k.get("context_position") != r["context_position"] or k.get("context_position") != c["context_position"] or k.get("next_token") != r["next_token"] \
                    or r["next_token"] != c["next_token"]:
                bad(f"row {i}: position/next_token differ between kl.jsonl and the rows files")
            if k.get("reference_top1_id") != r["top1_id"] or k.get("candidate_top1_id") != c["top1_id"]:
                bad(f"row {i}: top-1 ids differ from the rows files")
            if not margin_witness_equal(k.get("reference_margin"), r["top12_margin"]) or not margin_witness_equal(k.get("candidate_margin"), c["top12_margin"]):
                bad(f"row {i}: margins differ from the rows files (beyond the 1-ULP JSON re-parse witness tolerance)")
            match = r["top1_id"] == c["top1_id"]
            if k.get("top1_match") is not match:
                bad(f"row {i}: top1_match disagrees with the ids")
            agree += match
            same = r["logits_sha256"] == c["logits_sha256"]
            bit_same += same
            status, kl = k.get("kl_status"), k.get("kl_nats")
            if status == "ok":
                if not _is_num(kl) or kl < -KL_NEG_TOLERANCE:
                    bad(f"row {i}: kl_nats {kl!r} is not a finite non-negative number")
                else:
                    kls.append(float(kl))
            elif status == "infinite":
                if kl is not None:
                    bad(f"row {i}: infinite row carries kl_nats {kl!r}")
                infinite += 1
            else:
                bad(f"row {i}: kl_status {status!r}")
            trace.append({"row": i, "context_position": r["context_position"], "next_token": r["next_token"], "kl_nats": kl, "kl_status": status,
                          "reference_top1_id": r["top1_id"], "candidate_top1_id": c["top1_id"], "top1_match": match,
                          "reference_margin": r["top12_margin"], "candidate_margin": c["top12_margin"], "bit_identical": same})
        if psum.get("top1_agree_rows") != agree:
            bad(f"top1_agree_rows {psum.get('top1_agree_rows')} != recount {agree}")
        if psum.get("infinite_rows") != infinite:
            bad(f"infinite_rows {psum.get('infinite_rows')} != recount {infinite}")
        report = {"pair": name, "rows": want_rows, "infinite_rows": infinite, "top1_agree_rows": agree, "top1_agreement": agree / want_rows,
                  "bit_identical_rows": bit_same, "bit_identical_all": bit_same == want_rows,
                  "reference": {"cell": ref_key[1], "phase": ref_key[0], "stream_sha256": ref["rec"]["stream_sha256"]},
                  "candidate": {"cell": cand_key[1], "phase": cand_key[0], "stream_sha256": cand["rec"]["stream_sha256"]},
                  "kl_values": kls, "trace": trace}
        if infinite == 0 and len(kls) == want_rows:
            sum_native, mean_native = psum.get("kl_sum_nats"), psum.get("kl_mean_nats")
            fs = math.fsum(kls)
            if not (_is_num(sum_native) and _is_num(mean_native)) or abs(fs - sum_native) > SUM_REL_TOLERANCE * max(abs(fs), 1e-300) + 1e-300 \
                    or abs(fs / want_rows - mean_native) > SUM_REL_TOLERANCE * max(abs(mean_native), 1e-300) + 1e-300:
                bad(f"native kl_sum/kl_mean {sum_native!r}/{mean_native!r} disagree with the kl.jsonl rows (sum {fs!r})")
            else:
                mx = max(kls)
                row_of_max = kls.index(mx)
                if psum.get("kl_max_row") != row_of_max and abs((psum.get("kl_max_nats") or 0.0) - mx) > SUM_REL_TOLERANCE * max(mx, 1e-300):
                    bad("kl_max_nats/kl_max_row disagree with the kl.jsonl rows")
                report.update({"kl_sum_nats": sum_native, "kl_mean_nats": mean_native, "kl_max_nats": psum.get("kl_max_nats"), "kl_max_row": psum.get("kl_max_row")})
        elif infinite:
            report.update({"kl_sum_nats": None, "kl_mean_nats": None})
        out[name] = report
    return out, {"kl_summary_sha256": summary_sha, "kl_jsonl_sha256": kl_sha, "summary": summary}


# ----------------------------------------------------------------------------------------------------------------------------------
# descriptive stratified moving-block bootstrap (deterministic)
# ----------------------------------------------------------------------------------------------------------------------------------
MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, m):
        limit = (1 << 64) - ((1 << 64) % m)
        while True:
            x = self.next()
            if x < limit:
                return x % m


def seed_from_label(label):
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def percentile(sorted_vals, p):
    pos = (len(sorted_vals) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def stratified_block_bootstrap(strata, label=BOOT_LABEL, block=BOOT_BLOCK, reps=BOOT_REPS, confidence=BOOT_CONFIDENCE):
    """strata: {name: [(kl, top1_match 0/1), ...]} in row order. Every replicate resamples each stratum independently with overlapping moving
    blocks of `block` consecutive positions (start uniform in [0, n-block]; the last block of a stratum is truncated to n); the SAME block draws
    serve the KL mean and the top-1 rate. Returns {name|'pooled': {'kl': (lo, hi), 'top1': (lo, hi)}} (pooled = equal weight over all rows)."""
    names = list(strata)
    prefix = {}
    for nm in names:
        pk, pt = [0.0], [0.0]
        for kl, t in strata[nm]:
            pk.append(pk[-1] + kl)
            pt.append(pt[-1] + t)
        prefix[nm] = (pk, pt, len(strata[nm]))
    rng = SplitMix64(seed_from_label(label))
    sums = {nm: ([], []) for nm in names}
    pooled_k, pooled_t = [], []
    total_n = sum(prefix[nm][2] for nm in names)
    for _ in range(reps):
        tk = tt = 0.0
        for nm in names:
            pk, pt, n = prefix[nm]
            if n < block:
                raise ValueError(f"stratum {nm} has {n} < block {block} positions")
            sk = st = 0.0
            for b in range(-(-n // block)):
                s = rng.below(n - block + 1)
                take = min(block, n - b * block)
                sk += pk[s + take] - pk[s]
                st += pt[s + take] - pt[s]
            sums[nm][0].append(sk / n)
            sums[nm][1].append(st / n)
            tk += sk
            tt += st
        pooled_k.append(tk / total_n)
        pooled_t.append(tt / total_n)
    alpha = (1.0 - confidence) / 2.0
    out = {}
    for nm in names:
        a, b = sorted(sums[nm][0]), sorted(sums[nm][1])
        out[nm] = {"kl": (percentile(a, alpha), percentile(a, 1 - alpha)), "top1": (percentile(b, alpha), percentile(b, 1 - alpha))}
    a, b = sorted(pooled_k), sorted(pooled_t)
    out["pooled"] = {"kl": (percentile(a, alpha), percentile(a, 1 - alpha)), "top1": (percentile(b, alpha), percentile(b, 1 - alpha))}
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# primary gate, resume report, determinism
# ----------------------------------------------------------------------------------------------------------------------------------
def primary_gate(plan, pairs, issues):
    ids = [c["id"] for c in plan["primary"]["contexts"]]
    reps = [pairs.get(f"primary-{cid}") for cid in ids]
    base = {"threshold_nats": GATE_NATS, "applies_to": "each context mean and the pooled mean", "top1_policy": TOP1_POLICY,
            "ci": {"method": "stratified moving-block bootstrap (descriptive, not iid, not a generalisation claim)", "block_length": BOOT_BLOCK,
                   "positions_per_context": plan["primary"]["heldout_rows"], "replicates": BOOT_REPS, "seed_label": BOOT_LABEL,
                   "confidence": BOOT_CONFIDENCE, "descriptive_only": True, "gates": False, "note": CI_NOTE}}
    if issues.any("primary") or any(r is None for r in reps):
        return {**base, "status": "INCOMPLETE", "contexts": {}, "pooled": None}
    ctx = {}
    for cid, r in zip(ids, reps):
        mean = r.get("kl_mean_nats")
        ctx[cid] = {"rows": r["rows"], "kl_mean_nats": mean, "kl_max_nats": r.get("kl_max_nats"), "kl_max_row": r.get("kl_max_row"),
                    "infinite_rows": r["infinite_rows"], "top1_agree_rows": r["top1_agree_rows"], "top1_agreement": r["top1_agreement"],
                    "bit_identical_rows": r["bit_identical_rows"], "pass": mean is not None and mean <= GATE_NATS}
    infinite = sum(r["infinite_rows"] for r in reps)
    total_rows = sum(r["rows"] for r in reps)
    if infinite == 0:
        pooled_mean = (reps[0]["kl_sum_nats"] + reps[1]["kl_sum_nats"]) / total_rows
        strata = {cid: list(zip(r["kl_values"], [1.0 if t["top1_match"] else 0.0 for t in r["trace"]])) for cid, r in zip(ids, reps)}
        ci = stratified_block_bootstrap(strata)
        for cid in ids:
            ctx[cid]["kl_mean_ci95"] = list(ci[cid]["kl"])
            ctx[cid]["top1_agreement_ci95"] = list(ci[cid]["top1"])
        agree = sum(r["top1_agree_rows"] for r in reps)
        pooled = {"rows": total_rows, "kl_mean_nats": pooled_mean, "kl_mean_ci95": list(ci["pooled"]["kl"]), "infinite_rows": 0,
                  "top1_agree_rows": agree, "top1_agreement": agree / total_rows, "top1_agreement_ci95": list(ci["pooled"]["top1"]),
                  "pass": pooled_mean <= GATE_NATS}
    else:
        agree = sum(r["top1_agree_rows"] for r in reps)
        pooled = {"rows": total_rows, "kl_mean_nats": None, "infinite_rows": infinite, "top1_agree_rows": agree, "top1_agreement": agree / total_rows,
                  "pass": False}
    ok = infinite == 0 and all(c["pass"] for c in ctx.values()) and pooled["pass"]
    return {**base, "status": "PASS" if ok else "FAIL", "contexts": ctx, "pooled": pooled}


def resume_report(pairs, issues):
    names = RESUME_PAIRS
    if issues.any("resume") or any(n not in pairs for n in names):
        return {"status": "INCOMPLETE", "disposition": None, "pairs": {}, "f32_exoneration": "NONE_UNTIL_Q1F32", "human_review_required": True}
    out = {}
    all_bit = True
    for n in names:
        r = pairs[n]
        out[n] = {k: v for k, v in r.items() if k not in ("kl_values",)}
        out[n]["rows_kl_ge_gate"] = [t["row"] for t in r["trace"] if t["kl_nats"] is None or t["kl_nats"] >= GATE_NATS]
        out[n]["mean_kl_ge_gate"] = r.get("kl_mean_nats") is None or r["kl_mean_nats"] >= GATE_NATS
        out[n]["max_kl_ge_gate"] = r.get("kl_max_nats") is None or r["kl_max_nats"] >= GATE_NATS
        all_bit = all_bit and r["bit_identical_all"]
    disposition = "RESUME_BIT_IDENTICAL_ALL_PAIRS" if all_bit else "WAIT_F32_REVIEW"
    flagged = [n for n in names if out[n]["mean_kl_ge_gate"] or out[n]["max_kl_ge_gate"]]
    return {"status": "REPORTED", "disposition": disposition, "pairs": out, "f32_exoneration": "NONE_UNTIL_Q1F32", "human_review_required": True,
            "gate_nats_for_flagging": GATE_NATS, "pairs_with_kl_ge_gate": flagged,
            "note": "The resume class is reported only. Every pair's KL, margins and traces are emitted; pairs with mean/max KL >= 0.010 are flagged. "
                    "No copy-bug / rounding-benign call is made and the f32 diagnostic (Q1f32) has not exonerated anything."}


def compare_determinism(plan, a, b, issues):
    """Exact s1a-vs-s1b comparison of the 7 greedy cases (both phase dirs already verified)."""
    cases = []
    cats = issues
    ca, cb = provenance_core(a), provenance_core(b)
    prov_equal = ca == cb
    if not prov_equal:
        diff = sorted(k for k in ca if ca[k] != cb[k])
        cats.add("determinism", f"s1a and s1b provenance differs in {diff}")
    for cid in plan["determinism"]["case_ids"]:
        name = f"det-{cid}"
        xa, xb = a["cells"].get(name), b["cells"].get(name)
        if xa is None or xb is None or not xa["ok"] or not xb["ok"]:
            cats.add("determinism", f"case {cid}: a verified cell is missing in s1a or s1b")
            continue
        ra, rb = xa["rec"], xb["rec"]
        rows_a, rows_b = xa["rows"], xb["rows"]
        first = next((i for i, (p, q) in enumerate(zip(rows_a, rows_b)) if p["logits_sha256"] != q["logits_sha256"]), None)
        pa, pb = ra["prefill"], rb["prefill"]
        entry = {
            "case": cid, "cell": name,
            "token_ids_equal": ra["token_ids"] == rb["token_ids"],
            "finish_reason": [ra["finish_reason"], rb["finish_reason"]], "finish_equal": ra["finish_reason"] == rb["finish_reason"],
            "rows": [len(rows_a), len(rows_b)], "rows_equal": len(rows_a) == len(rows_b),
            "first_differing_row": first if first is not None else (None if len(rows_a) == len(rows_b) else min(len(rows_a), len(rows_b))),
            "row_hashes_equal": len(rows_a) == len(rows_b) and first is None,
            "stream_sha256": [ra["stream_sha256"], rb["stream_sha256"]], "stream_equal": ra["stream_sha256"] == rb["stream_sha256"],
            "rows_file_equal": ra["rows_file_sha256"] == rb["rows_file_sha256"],
            "prefill_calls_equal": pa["calls"] == pb["calls"], "receipt_checks_equal": pa["receipt_checks"] == pb["receipt_checks"],
            "prompt_sha256_equal": ra["prompt_sha256"] == rb["prompt_sha256"],
            "retained_ids_match": [ra.get("retained_ids_match"), rb.get("retained_ids_match")]}
        entry["identical"] = all(entry[k] for k in ("token_ids_equal", "finish_equal", "rows_equal", "row_hashes_equal", "stream_equal", "rows_file_equal",
                                                    "prefill_calls_equal", "receipt_checks_equal", "prompt_sha256_equal"))
        cases.append(entry)
    complete = len(cases) == 7
    all_identical = complete and prov_equal and all(c["identical"] for c in cases)
    if not complete or not prov_equal:
        status = "INCOMPLETE"
    else:
        status = "PASS" if all_identical else "FAIL"
    return {"schema": SCHEMA_DET, "status": status, "all_identical": all_identical, "provenance_identical": prov_equal, "cases": cases,
            "retained_ids_note": "retained_ids_match is reported, never enforced"}


# ----------------------------------------------------------------------------------------------------------------------------------
# Q1a
# ----------------------------------------------------------------------------------------------------------------------------------
def _model_identity(pd):
    s = pd["summary"]
    return {"path": (s.get("model") or {}).get("path"), "pinned_sha256": (s.get("model") or {}).get("pinned_sha256"),
            "stat": {k: v for k, v in ((s.get("model") or {}).get("stat") or {}).items() if not k.startswith("ctime")}}


def judge_q1a(plan_path, expected_plan_sha256, baseline_dir, s1a_dir, score_dir, s1b_dir=None):
    plan, plan_sha = load_plan(plan_path, expected_plan_sha256)
    issues = Issues()
    phases = {"baseline": load_phase(baseline_dir, "baseline", plan, plan_sha, issues),
              "s1a": load_phase(s1a_dir, "s1a", plan, plan_sha, issues)}
    stage = "final" if s1b_dir else "resume"
    if s1b_dir:
        phases["s1b"] = load_phase(s1b_dir, "s1b", plan, plan_sha, issues)
    for a, b in (("baseline", "s1a"),):
        if phases[a] is not None and phases[b] is not None:
            pa, pb = provenance_core(phases[a]), provenance_core(phases[b])
            for k in ("plan_sha256", "binary_sha256", "source", "model"):
                if pa[k] != pb[k]:
                    issues.add("provenance", f"{a} and {b} differ in {k}")
    specs = q8_pair_specs(plan)
    pairs, score_meta = load_score(score_dir, specs, "full", phases, issues,
                                   lambda p: "primary" if p.startswith("primary-") else "resume")
    primary = primary_gate(plan, pairs, issues)
    resume = resume_report(pairs, issues)
    determinism = {"schema": SCHEMA_DET, "status": "PENDING_S1B", "all_identical": None, "cases": []}
    if s1b_dir:
        if phases["s1a"] is not None and phases.get("s1b") is not None:
            determinism = compare_determinism(plan, phases["s1a"], phases["s1b"], issues)
        else:
            determinism = {"schema": SCHEMA_DET, "status": "INCOMPLETE", "all_identical": None, "cases": []}
            issues.add("determinism", "s1a or s1b phase data is unusable")

    reasons = []
    clean = not issues.any("provenance")
    prim_fail = clean and primary["status"] == "FAIL"
    det_fail = clean and determinism["status"] == "FAIL"
    if prim_fail:
        reasons.append("primary gate failed: " + json.dumps({k: v["kl_mean_nats"] for k, v in primary["contexts"].items()} | {"pooled": primary["pooled"]["kl_mean_nats"]}))
    if det_fail:
        reasons.append("determinism: s1a and s1b are not bit-identical in all 7 cases")
    if prim_fail or det_fail:
        verdict = "FAIL"
    elif issues.any() or primary["status"] != "PASS" or determinism["status"] != "PASS" or resume["status"] != "REPORTED":
        verdict = "INCOMPLETE"
        if determinism["status"] == "PENDING_S1B":
            reasons.append("determinism pending (s1b has not run)")
        for cat, msgs in issues.dump().items():
            reasons.extend(f"{cat}: {m}" for m in msgs[:20])
    else:
        verdict = "PASS"
    identity = None
    if phases["baseline"] is not None:
        b = phases["baseline"]
        rc = b["cells"].get("resume-cold")
        identity = {"plan_sha256": plan_sha, "binary_sha256": (b["summary"].get("binary") or {}).get("sha256"),
                    "source": provenance_core(b)["source"], "model": _model_identity(b),
                    "resume": {"prompt_sha256": rc["rec"]["prompt_sha256"] if rc else None, "teacher_sha256": rc["rec"].get("teacher_sha256") if rc else None,
                               "vocab": rc["rec"]["vocab"] if rc else None}}
    doc = {
        "schema": SCHEMA_Q1A, "stage": stage, "verdict": verdict, "verdict_reasons": reasons,
        "primary_gate_status": primary["status"], "determinism_status": determinism["status"], "resume_disposition": resume.get("disposition"),
        "default_on_recommended": False, "human_review_required": True, "f32_exoneration": "NONE_UNTIL_Q1F32",
        "qualification_complete": verdict == "PASS" and resume.get("disposition") == "RESUME_BIT_IDENTICAL_ALL_PAIRS",
        "overall_S1": ("FAIL" if verdict == "FAIL" else "BLOCKED_F32_REVIEW" if resume.get("disposition") == "WAIT_F32_REVIEW"
                       else "INCOMPLETE" if verdict == "INCOMPLETE" else "Q1A_CLEAN_HUMAN_REVIEW_PENDING"),
        "report_label": "A PASS verdict means only: primary gate + determinism passed. While the q8 resume disposition is WAIT_F32_REVIEW this is NOT an "
                        "unqualified quality PASS (overall_S1 BLOCKED_F32_REVIEW); the Main resume verdict decides S1. Top-1 agreement is human-reviewed (Q2 pending).",
        "identity": identity,
        "primary": primary, "resume": resume, "determinism": determinism,
        "phases": {k: {"dir": v["dir"], "phase_summary_sha256": v["summary_sha256"], "cells": {n: {"stream_sha256": c["rec"]["stream_sha256"], "rows": c["rec"]["rows"]} for n, c in v["cells"].items()}}
                   for k, v in phases.items() if v is not None},
        "score": {k: v for k, v in (score_meta or {}).items() if k != "summary"},
        "issues": issues.dump(),
    }
    return seal(doc)


# ----------------------------------------------------------------------------------------------------------------------------------
# Q1f32
# ----------------------------------------------------------------------------------------------------------------------------------
def _row_view(trace_row):
    return {k: trace_row[k] for k in ("row", "context_position", "next_token", "kl_nats", "reference_top1_id", "candidate_top1_id", "top1_match",
                                       "reference_margin", "candidate_margin", "bit_identical")}


def judge_f32(plan_path, expected_plan_sha256, beta_dir, s1f32_dir, score_dir, q1a_summary_path, expected_q1a_sha256):
    plan, plan_sha = load_plan(plan_path, expected_plan_sha256)
    if not HEX64.match(expected_q1a_sha256 or ""):
        raise JudgeError("--expected-q1a-summary-sha256 must be 64 lowercase hex")
    prior, prior_sha = read_json_file(q1a_summary_path)
    if prior_sha != expected_q1a_sha256:
        raise JudgeError(f"Q1a summary sha256 {prior_sha} != expected {expected_q1a_sha256}")
    issues = Issues()
    if prior.get("schema") != SCHEMA_Q1A or not seal_ok(prior):
        issues.add("f32", "the Q1a summary is not a sealed hipfire.q1.q1a_summary.v1 document")
    if prior.get("stage") != "final":
        issues.add("f32", f"the Q1a summary stage is {prior.get('stage')!r}, not 'final'")
    phases = {"beta-f32": load_phase(beta_dir, "beta-f32", plan, plan_sha, issues),
              "s1-f32": load_phase(s1f32_dir, "s1-f32", plan, plan_sha, issues)}
    pair, meta = load_score(score_dir, f32_pair_specs(plan), "f32-pair-only", phases, issues, lambda p: "f32")
    ident = prior.get("identity") or {}
    if phases["beta-f32"] is not None and phases["s1-f32"] is not None:
        pa, pb = provenance_core(phases["beta-f32"]), provenance_core(phases["s1-f32"])
        for k in ("plan_sha256", "binary_sha256", "source", "model"):
            if pa[k] != pb[k]:
                issues.add("provenance", f"beta-f32 and s1-f32 differ in {k}")
        bf = phases["beta-f32"]
        if ident.get("plan_sha256") != plan_sha or ident.get("binary_sha256") != pa["binary_sha256"] or ident.get("source") != pa["source"] \
                or ident.get("model") != _model_identity(bf):
            issues.add("f32", "plan/binary/source/model identity differs from the sealed Q1a summary")
        rc = bf["cells"].get("resume-cold")
        if rc is not None and (ident.get("resume") or {}).get("prompt_sha256") != rc["rec"]["prompt_sha256"] \
                or rc is not None and (ident.get("resume") or {}).get("teacher_sha256") != rc["rec"].get("teacher_sha256"):
            issues.add("f32", "resume prompt/teacher sha256 differs from the sealed Q1a summary")
    q8 = ((prior.get("resume") or {}).get("pairs") or {}).get(Q8_CONTRAST_PAIR)
    f32 = pair.get(F32_PAIR)
    if q8 is None:
        issues.add("f32", f"the Q1a summary has no {Q8_CONTRAST_PAIR} resume pair")
    contrast = None
    if q8 is not None and f32 is not None and not issues.any():
        qt, ft = q8["trace"], f32["trace"]
        if len(qt) != len(ft) or any(a["context_position"] != b["context_position"] or a["next_token"] != b["next_token"] for a, b in zip(qt, ft)):
            issues.add("f32", "q8 and f32 resume pairs do not cover the same rows")
        else:
            q_mean, f_mean = q8.get("kl_mean_nats"), f32.get("kl_mean_nats")
            contrast = {
                "q8_pair": Q8_CONTRAST_PAIR, "f32_pair": F32_PAIR,
                "kl_mean_nats": {"q8": q_mean, "f32": f_mean, "f32_minus_q8": (f_mean - q_mean) if None not in (q_mean, f_mean) else None,
                                 "f32_over_q8": (f_mean / q_mean) if q_mean not in (None, 0) and f_mean is not None else None},
                "top1_agree_rows": {"q8": q8["top1_agree_rows"], "f32": f32["top1_agree_rows"], "rows": f32["rows"]},
                "row0": {"q8": _row_view(qt[0]), "f32": _row_view(ft[0]),
                         "top1_agreement_q8": qt[0]["top1_match"], "top1_agreement_f32": ft[0]["top1_match"]},
                "bit_identical": {"q8": q8["bit_identical_all"], "f32": f32["bit_identical_all"]},
                "margins": [{"row": a["row"], "q8_reference": a["reference_margin"], "q8_candidate": a["candidate_margin"],
                             "f32_reference": b["reference_margin"], "f32_candidate": b["candidate_margin"]} for a, b in zip(qt, ft)],
                "trace_f32": [_row_view(t) for t in ft]}
    verdict = "REPORTED" if contrast is not None and not issues.any() else "INCOMPLETE"
    doc = {
        "schema": SCHEMA_F32, "verdict": verdict,
        "verdict_reasons": [] if verdict == "REPORTED" else [f"{c}: {m}" for c, ms in issues.dump().items() for m in ms[:20]],
        "diagnostic_only": True, "performance_claim": False, "default_on_recommended": False, "human_review_required": True,
        "qualification_complete": False, "overall_S1": "F32_DIAGNOSTIC_HUMAN_REVIEW",
        "note": "Float state (f32) changes the reference: this contrast is a diagnostic. It is not a performance claim and makes no automatic copy-bug / rounding call.",
        "bound_q1a_summary": {"path": q1a_summary_path, "sha256": prior_sha, "verdict": prior.get("verdict"), "stage": prior.get("stage"),
                              "receipt_sha256": prior.get("receipt_sha256")},
        "plan_sha256": plan_sha,
        "f32_pair": {k: v for k, v in (f32 or {}).items() if k != "kl_values"},
        "contrast": contrast,
        "phases": {k: {"dir": v["dir"], "phase_summary_sha256": v["summary_sha256"]} for k, v in phases.items() if v is not None},
        "score": {k: v for k, v in (meta or {}).items() if k != "summary"},
        "issues": issues.dump(),
    }
    return seal(doc)


# ----------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------
EXIT = {"PASS": 0, "REPORTED": 0, "FAIL": 1, "INCOMPLETE": 3}


def _emit(doc):
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def _sha_arg(text):
    if not HEX64.match(text):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description="Q1 CPU judge for the native qwen4_slowlink_quality dumps")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("q1a")
    a.add_argument("--plan", required=True)
    a.add_argument("--expected-plan-sha256", required=True, type=_sha_arg)
    a.add_argument("--baseline-dir", required=True)
    a.add_argument("--s1a-dir", required=True)
    a.add_argument("--score-dir", required=True)
    a.add_argument("--s1b-dir")
    a.add_argument("--out", required=True)
    f = sub.add_parser("f32")
    f.add_argument("--plan", required=True)
    f.add_argument("--expected-plan-sha256", required=True, type=_sha_arg)
    f.add_argument("--beta-f32-dir", required=True)
    f.add_argument("--s1-f32-dir", required=True)
    f.add_argument("--score-f32-dir", required=True)
    f.add_argument("--q1a-summary", required=True)
    f.add_argument("--expected-q1a-summary-sha256", required=True, type=_sha_arg)
    f.add_argument("--out", required=True)
    args = p.parse_args(argv)
    for path in [args.plan, args.out] + [getattr(args, k) for k in ("baseline_dir", "s1a_dir", "score_dir", "s1b_dir", "beta_f32_dir", "s1_f32_dir", "score_f32_dir", "q1a_summary") if hasattr(args, k)]:
        if path is not None and not os.path.isabs(path):
            sys.stderr.write(f"q1_judge: {path} is not an absolute path\n")
            return 2
    if os.path.lexists(args.out):
        sys.stderr.write(f"q1_judge: --out {args.out} exists (never overwritten)\n")
        return 2
    try:
        if args.cmd == "q1a":
            doc = judge_q1a(args.plan, args.expected_plan_sha256, args.baseline_dir, args.s1a_dir, args.score_dir, args.s1b_dir)
        else:
            doc = judge_f32(args.plan, args.expected_plan_sha256, args.beta_f32_dir, args.s1_f32_dir, args.score_f32_dir, args.q1a_summary,
                            args.expected_q1a_summary_sha256)
    except JudgeError as e:
        sys.stderr.write(f"q1_judge: refused: {e}\n")
        return 2
    sha = write_json_new(args.out, doc)
    _emit({"verdict": doc["verdict"], "out": args.out, "file_sha256": sha, "reasons": doc["verdict_reasons"]})
    return EXIT[doc["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
