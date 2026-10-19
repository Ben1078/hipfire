#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only NATIVE -> PYTHON interop proof for Q1.   Python stdlib only.   Run:

    Q1_NATIVE_BINARY=/abs/path/to/target/release/examples/qwen4_slowlink_quality python3 test_q1_native_bridge.py

WHAT THIS PROVES.  The REAL compiled `qwen4_slowlink_quality` binary (never a Python stand-in) is run on small TEST-ONLY dumps and its REAL output is fed to
the REAL q1_judge functions / CLI:

  * `--phase check-plan`   : the native plan validator accepts the synthetic plan; its per-phase cell names/order equal the judge's `expected_specs`.
  * `--phase score`        : q8 baseline / s1a / s1b fixtures -> `kl.jsonl`, `kl-summary.json`, `determinism.json` (native writes, native hash-verifies the dumps).
  * `--phase score-f32`    : the f32 pair alone, with the actual flags; it equals the f32 pair of the full `score`.
  * q1_judge q1a / f32     : accept exactly those native files; counts, means, intervals, determinism and the resume class are asserted.
  * rejection              : a corrupted row byte, a mismatched prompt / source / binary / plan, a missing summary and a non-fresh output are refused
                             (natively where native checks it, by the judge where only the judge checks it).

WHAT IS SYNTHETIC (TEST ONLY).  Vocabulary 4, token ids < 4, a native-valid plan with the real counts (16384 / 4097 prefix + 512 held-out, 7 determinism
cases of 32 new tokens with the natural case expected at 594 tokens, 3400-token resume prompt split at 1537 with 31 shared teacher ids) whose ids are
hash noise.  The dumps carry 512 rows per primary teacher cell and 32 per resume cell.  Not real corpora, no teacher, no model, no GPU: this file
NEVER runs a GPU phase (baseline / s1a / s1b / beta-f32 / s1-f32), never passes --model / --manifest, and fails (never skips) when Q1_NATIVE_BINARY is
missing.  The dump writer below follows the Rust writer field for field (the key sets are asserted against both the Rust source text and the key sets of
the retained j1968 receipts)."""
import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1_judge as J  # noqa: E402

NATIVE = os.environ.get("Q1_NATIVE_BINARY", "")
RUST_SOURCE = os.path.normpath(os.path.join(HERE, "..", "..", "crates", "hipfire-arch-qwen4", "examples", "qwen4_slowlink_quality.rs"))

VOCAB = 4
HELD = 512
PRIMARY = (("ctx16k", 16384), ("ctx4k", 4097))
RESUME_N, SPLIT_AT, TEACHER_N = 3400, 1537, 31
MIN_LM_ROWS = 1537
ARM = J.ARM
GPU_PHASES = frozenset(J.PHASES)
FORBIDDEN_FLAGS = frozenset(("--model", "--manifest"))
PINNED = {"HIPFIRE_STATE_QUANT": "q8", "HIPFIRE_KV_MODE": "q8", "HIPFIRE_GRAPH": "0", "HIPFIRE_NORMALIZE_PROMPT": "0",
          "HIPFIRE_CHAT_CURRENT_DATE": "2026-10-17", "HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_PLE_FORMAT": "q4", "HIPFIRE_QWEN4_D2_DEFAULT": "0"}
BASELINE_ENV = {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "9", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096"}
S1_ABSENT = ["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"]
MODEL_PATH = "/synthetic/TEST-ONLY-model.mq4"
MODEL_SIZE = 125_000_000_000
MODEL_SHA = hashlib.sha256(b"TEST-ONLY synthetic model pin").hexdigest()
BIN_SHA = hashlib.sha256(b"TEST-ONLY synthetic bridge binary").hexdigest()
SOURCE = {"commit": "c" * 40, "note": "TEST-ONLY synthetic bridge fixture"}

CELL_SCHEMA = "hipfire.qwen4.slowlink_quality.cell.v1"
ROW_SCHEMA = "hipfire.qwen4.slowlink_quality.row.v1"

REF = [struct.unpack("<f", struct.pack("<f", math.log(p)))[0] for p in (0.5, 0.25, 0.125, 0.125)]
CAND = [struct.unpack("<f", struct.pack("<f", math.log(p)))[0] for p in (0.25, 0.5, 0.125, 0.125)]
PROBE_ROW = 3
PROBE_KL = 0.25 * math.log(2.0)         # KL((1/2,1/4,1/8,1/8) || (1/4,1/2,1/8,1/8)) = 1/2 ln 2 + 1/4 ln(1/2)

# Key sets of the retained j1968 S1 receipts (`greedy-s1/receipts.jsonl`, sha256 e5e5818c...63b3f): a layer-major call and the one-row-short bypass.
LM_RECEIPT_KEYS = frozenset("""bypass_reason discarded_group_payload_bytes entry_pos host_expert_bytes_total host_layers injection_consumed launch_cuts overlap
peak_vram_bytes prompt_rows reserve resident_layers retained_scratch_rows route schema selected_by session_cuts spans_ms status submitted_bytes sweeps
tile_rows timeline_overflow unused_prefetched_bytes useful_bytes vram_measurement windows""".split())
INCUMBENT_RECEIPT_KEYS = frozenset("""bypass_reason entry_pos host_expert_bytes_total host_layers injection_consumed launch_cuts overlap peak_vram_bytes
prompt_rows reserve resident_layers retained_scratch_rows route schema selected_by spans_ms submitted_bytes sweeps tile_rows unused_prefetched_bytes
useful_bytes windows""".split())
CHECK_KEYS = frozenset("route selected_by entry_pos prompt_rows tile_rows retained_scratch_rows resident_layers host_layers sweeps status bypass_reason".split())
ROW_KEYS = frozenset("""schema cell row input_token context_position next_token logits_sha256 bin_offset top1_id top1_logit top1_logit_bits top2_id top2_logit
top2_logit_bits top12_margin neg_inf_count""".split())
CELL_BASE_KEYS = frozenset("""schema phase arm cell kind context prompt_tokens prompt_sha256 prefill rows vocab bin bin_bytes rows_file rows_file_sha256
stream_sha256 rows_seconds cell_seconds final_position""".split())
CELL_TEACHER_KEYS = frozenset(("teacher_feed_tokens", "teacher_sha256"))
CELL_GREEDY_KEYS = frozenset(("token_ids", "token_ids_sha256", "finish_reason", "text"))
CELL_DET_KEYS = frozenset(("case_id", "resume_at", "retained_ids"))
CELL_PRIMARY_KEYS = frozenset(("context_id", "prefix_tokens", "prefix_sha256", "heldout_sha256", "sources"))
CELL_RESUME_KEYS = frozenset(("split_at",))
PREFILL_KEYS = frozenset(("structure", "calls", "receipts", "receipt_checks", "seconds"))
SUMMARY_KEYS = frozenset("schema complete phase arm state_f32 plan manifest binary model environment load cells cell_order wall_seconds".split())
LOAD_KEYS = frozenset("""selector placement slowlink_profile resident_layers host_mapped_tensors scratch_rows ple_format state_format vocab eos_token
num_layers arch_id gpu_arch profile""".split())
PROFILE_KEYS = frozenset("""tile_rows window_rows resident_layers baseline_layers checkpoint_stride_rows link_gbps device_snapshot_budget_bytes
purpose""".split())
STAT_KEYS = frozenset("dev ino size mtime_sec mtime_nsec ctime_sec ctime_nsec ctime_is_metadata_only".split())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_next(x):
    """The next f32 above x (bit pattern + 1; x > 0)."""
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", x))[0] + 1))[0]


def serde_parse(text):
    """serde_json 1.0.x WITHOUT `float_roundtrip` (de.rs `f64_from_parts`): the decimal significand as a u64 rounded to f64, then ONE multiply/divide by
    an exactly-representable power of ten. This is the parser model of the native binary (cargo tree: no float_roundtrip in this build)."""
    head, _, exp10 = text.partition("e")
    sign = -1.0 if head.startswith("-") else 1.0
    ip, _, fp = head.lstrip("-").partition(".")
    digits = (ip + fp).lstrip("0") or "0"
    exponent = int(exp10 or 0) - len(fp)
    assert int(digits) < 2 ** 64, text
    f = float(int(digits))
    power = float("1e%d" % abs(exponent))
    return sign * (f * power if exponent >= 0 else f / power)


def noise(label, i, j):
    h = hashlib.sha256(f"{label}|{i}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 63 - 1.0


def ids4(label, n):
    """n token ids, all < VOCAB (hash noise, roughly uniform)."""
    return [int.from_bytes(hashlib.sha256(f"{label}:{k}".encode()).digest()[:4], "big") % VOCAB for k in range(n)]


def u32sha(ids):
    return sha(struct.pack(f"<{len(ids)}I", *ids))


def chunk_ranges(n, chunk):
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


# ----------------------------------------------------------------------------------------------------------------------------------
# the native-valid synthetic plan
# ----------------------------------------------------------------------------------------------------------------------------------
DET = (("p512", 512, None), ("p1536", 1536, None), ("p1537", 1537, None), ("p8192", 8192, None), ("irregular", 4100, None),
       ("resume", RESUME_N, SPLIT_AT), ("natural", 594, "messages"))
GREEDY_SHAPE = {"p1537": (7, "stop"), "irregular": (2, "stop")}


def det_greedy(cid):
    n, finish = GREEDY_SHAPE.get(cid, (32, "length"))
    ids = ids4("greedy:" + cid, n)
    if finish == "stop":
        ids[-1] = 3
    return ids, finish


def make_plan():
    cases = []
    for cid, n, mode in DET:
        case = {"id": cid, "expected_prompt_tokens": n}
        if mode == "messages":
            case["messages"] = [{"role": "user", "content": "TEST-ONLY synthetic natural prompt"}]
        else:
            case["prompt_ids"] = ids4("det:" + cid, n)
            if mode is not None:
                case["resume_at"] = mode
        cases.append(case)
    kept, _ = det_greedy("p1537")
    cases[2]["retained_ids"] = kept                                  # matches what the dumps hold
    irregular, _ = det_greedy("irregular")
    cases[4]["retained_ids"] = [(irregular[0] + 1) % VOCAB, irregular[1]]   # differs: reported, never enforced
    contexts = []
    for cid, n in PRIMARY:
        prefix, held = ids4(cid + ":prefix", n), ids4(cid + ":held", HELD)
        contexts.append({
            "id": cid, "prefix_ids": prefix, "heldout_ids": held, "prefix_sha256": u32sha(prefix), "heldout_sha256": u32sha(held),
            "sources": [
                {"label": f"TEST-ONLY-{cid}-a", "path": f"/synthetic/{cid}-a.json", "sha256": sha(f"{cid}-a".encode()), "concat_offset": 0, "token_count": n},
                {"label": f"TEST-ONLY-{cid}-b", "path": f"/synthetic/{cid}-b.json", "sha256": sha(f"{cid}-b".encode()), "concat_offset": n, "token_count": HELD}]})
    retained = ids4("resume:retained", TEACHER_N + 1)
    prompt, teacher = ids4("resume:prompt", RESUME_N), retained[:TEACHER_N]
    return {
        "schema": J.PLAN_SCHEMA, "max_seq": 16897,
        "artifact": {"path": MODEL_PATH, "size": MODEL_SIZE, "sha256": MODEL_SHA},
        "expect": {"resident_layers": 9, "baseline_tile_rows": 4096, "s1_tile_rows": 768, "s1_layer_major_min_rows": MIN_LM_ROWS, "ple_format": "q4",
                   "env": dict(PINNED), "env_absent": list(S1_ABSENT), "baseline_env": dict(BASELINE_ENV)},
        "determinism": {"case_ids": [c["id"] for c in cases], "max_new_tokens": 32, "thinking": False, "cases": cases},
        "primary": {"heldout_rows": HELD, "greedy_new_tokens": 32, "contexts": contexts},
        "resume": {"prompt_ids": prompt, "prompt_sha256": u32sha(prompt), "split_at": SPLIT_AT, "teacher_ids": teacher, "teacher_sha256": u32sha(teacher),
                   "retained_s1_ids": retained},
    }


# ----------------------------------------------------------------------------------------------------------------------------------
# synthetic dumps, written the way the Rust writer writes them
# ----------------------------------------------------------------------------------------------------------------------------------
def lm_receipt(entry, rows):
    cuts = list(range(768, rows, 768)) + [rows]
    return {
        "schema": "hipfire.qwen4.slowlink_receipt.v1", "route": "layer-major", "selected_by": "env1", "bypass_reason": None, "status": "ok",
        "entry_pos": entry, "prompt_rows": rows, "windows": [{"start": 0, "rows": rows}], "launch_cuts": cuts, "session_cuts": [],
        "tile_rows": 768, "retained_scratch_rows": 768, "sweeps": 1, "resident_layers": 9, "host_layers": 39,
        "host_expert_bytes_total": 52140441600, "submitted_bytes": 52140441600, "useful_bytes": 20101017600, "unused_prefetched_bytes": 32039424000,
        "discarded_group_payload_bytes": 0,
        "reserve": {"tile": 947662596, "arena": 887535872, "ring": 167190528, "lookahead": 1337524224, "tables": 147456, "total": 3476572740},
        "spans_ms": {"wall": 7608.303968, "copy": 7547.773855, "compute": 229.995087, "route_barrier": 552.874143, "nonexpert": 0.0},
        "overlap": {"cross_layer_intersections": 38, "cross_layer_overlap_ms": 47.222015, "copy_idle_ms": 57.441674, "page_full_stalls": 646,
                    "copy_starved_stalls": 595},
        "timeline_overflow": 0, "peak_vram_bytes": 23024631808,
        "vram_measurement": {"method": "hipMemGetInfo_post_request_sample", "continuous_peak": False}, "injection_consumed": False}


def incumbent_receipt(selected_by, bypass, entry, rows, tile):
    return {
        "schema": "hipfire.qwen4.slowlink_receipt.v1", "route": "incumbent", "selected_by": selected_by, "bypass_reason": bypass,
        "entry_pos": entry, "prompt_rows": rows, "windows": [{"start": 0, "rows": rows}], "launch_cuts": [rows], "tile_rows": tile,
        "retained_scratch_rows": tile, "sweeps": 0, "resident_layers": 0, "host_layers": 0, "host_expert_bytes_total": 0, "submitted_bytes": 0,
        "useful_bytes": 0, "unused_prefetched_bytes": 0,
        "reserve": {"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0},
        "spans_ms": {"wall": 0.0, "copy": 0.0, "compute": 0.0, "route_barrier": 0.0, "nonexpert": 0.0},
        "overlap": {"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0, "copy_idle_ms": 0.0, "page_full_stalls": 0, "copy_starved_stalls": 0},
        "peak_vram_bytes": 0, "injection_consumed": False}


def receipt_for(arm, entry, rows):
    if arm == "baseline":
        return incumbent_receipt("env0", None, entry, rows, 4096)
    if rows >= MIN_LM_ROWS:
        return lm_receipt(entry, rows)
    return incumbent_receipt("env1", f"TooShort {{ rows: {rows}, minimum: {MIN_LM_ROWS} }}", entry, rows, 768)


def check_of(receipt):
    return {k: receipt.get(k) for k in sorted(CHECK_KEYS)}


def phase_cells(plan, phase):
    """Ordered cell descriptors of a phase (the native order)."""
    arm = ARM[phase]
    prim, res = plan["primary"], plan["resume"]
    cells = []

    def det(case):
        cid, n, at = case["id"], case["expected_prompt_tokens"], case.get("resume_at")
        ids, finish = det_greedy(cid)
        extra = {"case_id": cid, "resume_at": at, "retained_ids": case.get("retained_ids")}
        if case.get("retained_ids") is not None:
            extra["retained_ids_match"] = case["retained_ids"] == ids
        cells.append({"name": "det-" + cid, "kind": "determinism", "context": None, "n": n,
                      "prompt_sha": u32sha(case["prompt_ids"]) if "prompt_ids" in case else sha(b"TEST-ONLY rendered natural prompt"),
                      "calls": [(0, at), (at, n)] if at else [(0, n)], "structure": "direct-consecutive-calls" if at else "prefill_final",
                      "greedy": (ids, finish), "teacher": None, "extra": extra})

    def primary(ctx):
        n = len(ctx["prefix_ids"])
        calls = chunk_ranges(n, 4096) if arm == "baseline" else [(0, n)]
        structure = "direct-single-call" if len(calls) == 1 else "direct-consecutive-calls"
        extra = {"context_id": ctx["id"], "prefix_tokens": n, "prefix_sha256": ctx["prefix_sha256"], "heldout_sha256": ctx["heldout_sha256"],
                 "sources": ctx["sources"]}
        cells.append({"name": f"primary-{ctx['id']}-teacher", "kind": "primary-teacher", "context": ctx["id"], "n": n, "prompt_sha": u32sha(ctx["prefix_ids"]),
                      "calls": calls, "structure": structure, "greedy": None, "extra": extra,
                      "teacher": {"feed": ctx["heldout_ids"][:-1], "nexts": list(ctx["heldout_ids"]), "sha": ctx["heldout_sha256"]}})
        cells.append({"name": f"primary-{ctx['id']}-greedy", "kind": "primary-greedy", "context": ctx["id"], "n": n, "prompt_sha": u32sha(ctx["prefix_ids"]),
                      "calls": calls, "structure": structure, "greedy": (ids4(f"greedy:primary-{ctx['id']}", 32), "length"), "teacher": None,
                      "extra": extra})

    def resume(split):
        n = len(res["prompt_ids"])
        cells.append({"name": "resume-split" if split else "resume-cold", "kind": "resume", "context": None, "n": n, "prompt_sha": u32sha(res["prompt_ids"]),
                      "calls": [(0, res["split_at"]), (res["split_at"], n)] if split else [(0, n)],
                      "structure": "direct-consecutive-calls" if split else "direct-single-call", "greedy": None,
                      "extra": {"split_at": res["split_at"] if split else None},
                      "teacher": {"feed": list(res["teacher_ids"]), "nexts": list(res["teacher_ids"]) + [None], "sha": res["teacher_sha256"]}})

    if phase == "baseline":
        for ctx in prim["contexts"]:
            primary(ctx)
        resume(False)
        resume(True)
    elif phase == "s1a":
        for case in plan["determinism"]["cases"]:
            det(case)
        for ctx in prim["contexts"]:
            primary(ctx)
        resume(True)
        resume(False)
    elif phase == "s1b":
        for case in plan["determinism"]["cases"]:
            det(case)
    elif phase == "beta-f32":
        resume(False)
    elif phase == "s1-f32":
        resume(True)
    else:
        raise AssertionError(f"not a GPU phase: {phase}")
    return cells


def base_rows(fam, n):
    return [[f32(3.0 * noise("base:" + fam, i, j)) for j in range(VOCAB)] for i in range(n)]


def perturb(rows, label, eps):
    return [[f32(v + eps * noise(label, i, j)) for j, v in enumerate(row)] for i, row in enumerate(rows)]


def resume_rows(phase, name):
    """Known construction: only PROBE_ROW carries the analytic distributions; s1a cold == baseline cold bit for bit; the f32 pair differs ONLY at PROBE_ROW."""
    cold = base_rows("resume", TEACHER_N + 1)
    cold[PROBE_ROW] = list(REF)
    cold[7] = [f32_next(1.0), 1.0, -2.0, -3.0]      # top-1/top-2 are ADJACENT f32 values: margin 2^-23, dyadic, a 17-digit decimal
    cold[9] = [0.5, 0.5, -1.0, -1.0]                # an exact tie: margin 0.0, the lower index wins
    if phase == "baseline":
        return cold if name == "resume-cold" else perturb(cold, "betasplit", 0.0005)
    if phase == "s1a":
        if name == "resume-cold":
            return [list(r) for r in cold]
        rows = perturb(cold, "s1split", 0.002)
        rows[PROBE_ROW] = list(CAND)
        return rows
    f32cold = perturb(cold, "f32cold", 0.0003)
    f32cold[PROBE_ROW] = list(REF)
    if phase == "beta-f32":
        return f32cold
    rows = [list(r) for r in f32cold]
    rows[PROBE_ROW] = list(CAND)
    return rows


def cell_logits(phase, cell, flip):
    if cell["kind"] == "resume":
        return resume_rows(phase, cell["name"])
    n = (len(cell["greedy"][0]) if cell["greedy"] is not None else len(cell["teacher"]["feed"])) + 1
    rows = base_rows(cell["name"], n)
    if ARM[phase] == "s1":
        rows = perturb(rows, "s1:" + cell["name"], 0.002)
    if flip and phase == "s1b" and cell["name"] == "det-" + flip[0]:
        rows[flip[1]][0] = f32(rows[flip[1]][0] + 0.5)
    return rows


def row_stats(row):
    order = sorted(range(VOCAB), key=lambda j: (-row[j], j))
    t1, t2 = order[0], order[1]
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]  # noqa: E731
    return {"top1_id": t1, "top1_logit": row[t1], "top1_logit_bits": bits(row[t1]), "top2_id": t2, "top2_logit": row[t2],
            "top2_logit_bits": bits(row[t2]), "top12_margin": float(row[t1]) - float(row[t2]), "neg_inf_count": 0}


def write_cell(out, phase, cell, flip):
    arm, name = ARM[phase], cell["name"]
    rows = cell_logits(phase, cell, flip)
    if cell["greedy"] is not None:
        ids, finish = cell["greedy"]
        feed, nexts = list(ids), list(ids) + [None]
    else:
        feed, nexts = cell["teacher"]["feed"], cell["teacher"]["nexts"]
    assert len(rows) == len(feed) + 1 == len(nexts), (name, len(rows), len(feed), len(nexts))
    stream, lines, offset = hashlib.sha256(), [], 0
    with open(os.path.join(out, f"{name}.logits.f32le"), "xb") as fb:
        for i, row in enumerate(rows):
            buf = struct.pack(f"<{VOCAB}f", *row)
            fb.write(buf)
            stream.update(buf)
            lines.append(json.dumps({"schema": ROW_SCHEMA, "cell": name, "row": i, "input_token": None if i == 0 else feed[i - 1],
                                     "context_position": cell["n"] + i, "next_token": nexts[i], "logits_sha256": sha(buf), "bin_offset": offset,
                                     **row_stats(row)}))
            offset += len(buf)
    rows_bytes = ("\n".join(lines) + "\n").encode()
    with open(os.path.join(out, f"{name}.rows.jsonl"), "xb") as fr:
        fr.write(rows_bytes)
    receipts = [receipt_for(arm, s, e - s) for s, e in cell["calls"]]
    rec = {"schema": CELL_SCHEMA, "phase": phase, "arm": arm, "cell": name, "kind": cell["kind"], "context": cell["context"], "prompt_tokens": cell["n"],
           "prompt_sha256": cell["prompt_sha"],
           "prefill": {"structure": cell["structure"], "calls": [{"start": s, "end": e, "rows": e - s} for s, e in cell["calls"]],
                       "receipts": receipts, "receipt_checks": [check_of(r) for r in receipts], "seconds": 1.25},
           "rows": len(rows), "vocab": VOCAB, "bin": f"{name}.logits.f32le", "bin_bytes": len(rows) * VOCAB * 4, "rows_file": f"{name}.rows.jsonl",
           "rows_file_sha256": sha(rows_bytes), "stream_sha256": stream.hexdigest(), "rows_seconds": 0.5, "cell_seconds": 1.75,
           "final_position": cell["n"] + len(rows) - 1}
    if cell["teacher"] is not None:
        rec["teacher_feed_tokens"] = len(feed)
        rec["teacher_sha256"] = cell["teacher"]["sha"]
    if cell["greedy"] is not None:
        rec.update({"token_ids": ids, "token_ids_sha256": u32sha(ids), "finish_reason": finish, "text": "TEST-ONLY"})
    rec.update(cell["extra"])
    return rec


def load_facts(phase):
    arm = ARM[phase]
    profile = None if arm == "baseline" else {"tile_rows": 768, "window_rows": 768, "resident_layers": 9, "baseline_layers": 5, "checkpoint_stride_rows": 768,
                                              "link_gbps": 12.5, "device_snapshot_budget_bytes": 1 << 30, "purpose": "product"}
    return {"selector": "off" if arm == "baseline" else "force", "placement": "explicit-env" if arm == "baseline" else "slowlink-profile",
            "slowlink_profile": arm == "s1", "resident_layers": 9, "host_mapped_tensors": 117, "scratch_rows": 4096 if arm == "baseline" else 768,
            "ple_format": "q4", "state_format": {"gdn": "F32" if J.STATE_F32[phase] else "Q8", "qsa_kv": "Q8"}, "vocab": VOCAB, "eos_token": 3,
            "num_layers": 48, "arch_id": 1, "gpu_arch": "gfx1100", "profile": profile}


def write_phase(out_dir, plan, plan_sha, phase, flip=None):
    """One TEST-ONLY phase directory as the native GPU phase writes it (phase-summary.json LAST). `out_dir` must be absent or empty."""
    os.makedirs(out_dir)
    recs, done = [], []
    for cell in phase_cells(plan, phase):
        rec = write_cell(out_dir, phase, cell, flip)
        recs.append(rec)
        done.append({"cell": rec["cell"], "rows": rec["rows"], "stream_sha256": rec["stream_sha256"], "prompt_tokens": rec["prompt_tokens"],
                     "finish_reason": rec.get("finish_reason")})
    with open(os.path.join(out_dir, "cells.jsonl"), "xb") as f:
        for rec in recs:
            f.write((json.dumps(rec) + "\n").encode())
    env = dict(PINNED)                                   # independent of the judge: pinned class flags (+ the baseline N9 / 4096, + f32 state)
    if ARM[phase] == "baseline":
        env.update(BASELINE_ENV)
    if J.STATE_F32[phase]:
        env["HIPFIRE_STATE_QUANT"] = "f32"
    if ARM[phase] == "s1":
        for name in plan["expect"]["env_absent"]:
            env[name] = None
    ex = plan["expect"]
    art = plan["artifact"]
    model_tuple = {"path": art["path"], "dev": 64769, "ino": 4242, "size": art["size"], "mtime_sec": 1_760_000_000, "mtime_nsec": 123456789,
                   "sha256": art["sha256"]}
    summary = {
        "schema": J.SUMMARY_SCHEMA, "complete": True, "phase": phase, "arm": ARM[phase], "state_f32": J.STATE_F32[phase],
        "plan": {"path": "/synthetic/plan.json", "sha256": plan_sha, "max_seq": plan["max_seq"],
                 "expect": {"resident_layers": ex["resident_layers"], "baseline_tile_rows": ex["baseline_tile_rows"], "s1_tile_rows": ex["s1_tile_rows"],
                            "s1_layer_major_min_rows": ex["s1_layer_major_min_rows"], "ple_format": ex["ple_format"]}},
        "manifest": {"path": "/synthetic/manifest.json", "sha256": sha(b"TEST-ONLY manifest"),
                     "content": {"schema": "hipfire.qwen4.slowlink_quality.manifest.v1", "binary": {"sha256": BIN_SHA, "path": "/synthetic/bin"},
                                 "source": dict(SOURCE), "launch": {"note": "TEST-ONLY"}, "model": model_tuple}},
        "binary": {"path": "/synthetic/bin", "sha256": BIN_SHA},
        "model": {"path": art["path"], "pinned_sha256": art["sha256"], "hashed_here": False,
                  "stat": {"dev": 64769, "ino": 4242, "size": art["size"], "mtime_sec": 1_760_000_000, "mtime_nsec": 123456789, "ctime_sec": 1_760_000_100,
                           "ctime_nsec": 5, "ctime_is_metadata_only": True}},
        "environment": env, "load": load_facts(phase), "cells": done, "cell_order": [r["cell"] for r in recs], "wall_seconds": 3.5}
    with open(os.path.join(out_dir, "phase-summary.json"), "xb") as f:
        f.write(json.dumps(summary, indent=1).encode())


# ----------------------------------------------------------------------------------------------------------------------------------
# running the REAL native binary (CPU phases only)
# ----------------------------------------------------------------------------------------------------------------------------------
def scrubbed_env(home):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HIPFIRE_", "HIP_", "ROCR_", "HSA_", "GPU_", "AMD_"))}
    env["HOME"] = home
    return env


class Native:
    def __init__(self, home):
        self.home = home

    def run(self, *argv):
        """The real binary on a CPU phase. A GPU phase or a model/manifest flag is refused HERE, before anything is spawned."""
        phase = argv[argv.index("--phase") + 1] if "--phase" in argv else None
        if phase in GPU_PHASES:
            raise AssertionError(f"the bridge never runs the GPU phase {phase}")
        if FORBIDDEN_FLAGS & set(argv):
            raise AssertionError("the bridge never passes --model / --manifest")
        return subprocess.run([NATIVE, *argv], capture_output=True, text=True, timeout=300, env=scrubbed_env(self.home), cwd=self.home)

    def ok(self, *argv):
        p = self.run(*argv)
        if p.returncode != 0:
            raise AssertionError(f"native {argv[:2]} exited {p.returncode}: {p.stderr[-2000:]}")
        return p


# ----------------------------------------------------------------------------------------------------------------------------------
# reading dumps / native output back, and an independent KL
# ----------------------------------------------------------------------------------------------------------------------------------
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_cell(dirpath, name):
    rec = next(r for r in read_jsonl(os.path.join(dirpath, "cells.jsonl")) if r["cell"] == name)
    meta = read_jsonl(os.path.join(dirpath, rec["rows_file"]))
    with open(os.path.join(dirpath, rec["bin"]), "rb") as f:
        data = f.read()
    logits = [list(struct.unpack_from(f"<{VOCAB}f", data, i * VOCAB * 4)) for i in range(rec["rows"])]
    return rec, meta, logits


def naive_kl(ref, cand):
    """KL(ref||cand) in f64 from the dumped f32 logits: an independent implementation (plain max-shifted softmax, fsum)."""
    def logp(row):
        m = max(row)
        z = math.log(math.fsum(math.exp(float(v) - m) for v in row))
        return [float(v) - m - z for v in row]
    lp, lq = logp(ref), logp(cand)
    return math.fsum(math.exp(a) * (a - b) for a, b in zip(lp, lq))


def top1(row):
    return max(range(VOCAB), key=lambda j: (row[j], -j))


def ulp_distance(a, b):
    ia, ib = (struct.unpack("<q", struct.pack("<d", v))[0] for v in (a, b))
    return abs(ia - ib)


def json_keys_in_rust_source(keys, source):
    return sorted(k for k in keys if f'"{k}"' not in source)


# ----------------------------------------------------------------------------------------------------------------------------------
# the world: plan + five synthetic phase dirs + the REAL native score outputs + the REAL judge outputs
# ----------------------------------------------------------------------------------------------------------------------------------
WORLD = {}


def setUpModule():
    if not NATIVE or not os.path.isabs(NATIVE):
        raise RuntimeError("Q1_NATIVE_BINARY must be set to the ABSOLUTE path of the compiled qwen4_slowlink_quality binary (this test never skips)")
    if not (os.path.isfile(NATIVE) and os.access(NATIVE, os.X_OK)):
        raise RuntimeError(f"Q1_NATIVE_BINARY {NATIVE} is not an executable regular file")
    root = tempfile.mkdtemp(prefix="q1-native-bridge-")
    WORLD["root"] = root
    plan = make_plan()
    plan_path = os.path.join(root, "plan.json")
    data = json.dumps(plan, indent=1).encode()
    with open(plan_path, "xb") as f:
        f.write(data)
    plan_sha = sha(data)
    WORLD.update({"plan": plan, "plan_path": plan_path, "plan_sha": plan_sha})
    dirs = {}
    for phase in J.PHASES:
        dirs[phase] = os.path.join(root, phase)
        write_phase(dirs[phase], plan, plan_sha, phase)
    WORLD["dirs"] = dirs
    native = Native(root)
    WORLD["native"] = native
    base = ["--plan", plan_path]
    q8 = ["--baseline-dir", dirs["baseline"], "--s1a-dir", dirs["s1a"]]
    f32d = ["--beta-f32-dir", dirs["beta-f32"], "--s1-f32-dir", dirs["s1-f32"]]
    WORLD["score"] = os.path.join(root, "native-score")
    native.ok("--phase", "score", *base, *q8, "--s1b-dir", dirs["s1b"], "--out-dir", WORLD["score"])
    WORLD["score_full_f32"] = os.path.join(root, "native-score-with-f32")
    native.ok("--phase", "score", *base, *q8, *f32d, "--out-dir", WORLD["score_full_f32"])
    WORLD["score_f32"] = os.path.join(root, "native-score-f32")
    native.ok("--phase", "score-f32", *base, *f32d, "--out-dir", WORLD["score_f32"])
    # the REAL judge, through its CLI: q1a (with s1b), then f32 bound to the sealed q1a summary
    WORLD["q1a_out"] = os.path.join(root, "q1a-summary.json")
    WORLD["q1a_rc"], WORLD["q1a_stdout"] = run_judge(["q1a", "--plan", plan_path, "--expected-plan-sha256", plan_sha, "--baseline-dir", dirs["baseline"],
                                                       "--s1a-dir", dirs["s1a"], "--score-dir", WORLD["score"], "--s1b-dir", dirs["s1b"], "--out", WORLD["q1a_out"]])
    WORLD["q1a"], WORLD["q1a_sha"] = J.read_json_file(WORLD["q1a_out"])
    WORLD["f32_out"] = os.path.join(root, "f32-summary.json")
    WORLD["f32_rc"], WORLD["f32_stdout"] = run_judge(["f32", "--plan", plan_path, "--expected-plan-sha256", plan_sha, "--beta-f32-dir", dirs["beta-f32"],
                                                       "--s1-f32-dir", dirs["s1-f32"], "--score-f32-dir", WORLD["score_f32"], "--q1a-summary", WORLD["q1a_out"],
                                                       "--expected-q1a-summary-sha256", WORLD["q1a_sha"], "--out", WORLD["f32_out"]])
    WORLD["f32"], _ = J.read_json_file(WORLD["f32_out"])


def tearDownModule():
    if WORLD.get("root"):
        shutil.rmtree(WORLD["root"], ignore_errors=True)


def run_judge(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = J.main(argv)
    return rc, out.getvalue() + err.getvalue()


def load_kl(score_dir):
    summary, _ = J.read_json_file(os.path.join(score_dir, "kl-summary.json"))
    rows, _ = J.read_jsonl_file(os.path.join(score_dir, "kl.jsonl"))
    return summary, rows


def variant(src, name):
    dst = os.path.join(WORLD["root"], name)
    shutil.copytree(src, dst)
    return dst


def native_score_args(baseline, s1a, out, s1b=None):
    args = ["--phase", "score", "--plan", WORLD["plan_path"], "--baseline-dir", baseline, "--s1a-dir", s1a]
    if s1b:
        args += ["--s1b-dir", s1b]
    return args + ["--out-dir", out]


def judge_q1a_argv(baseline, s1a, score, out, s1b=None, plan_sha=None):
    argv = ["q1a", "--plan", WORLD["plan_path"], "--expected-plan-sha256", plan_sha or WORLD["plan_sha"], "--baseline-dir", baseline, "--s1a-dir", s1a,
            "--score-dir", score, "--out", out]
    return argv + (["--s1b-dir", s1b] if s1b else [])


# ----------------------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------------------
class NativePlanAndSchema(unittest.TestCase):
    def test_help_and_check_plan_agree_with_the_judge(self):
        native, plan = WORLD["native"], WORLD["plan"]
        help_out = native.ok("--help").stdout
        for phase in ("baseline", "s1a", "s1b", "beta-f32", "s1-f32", "score", "score-f32", "check-plan"):
            self.assertIn(phase, help_out)
        report = json.loads(native.ok("--phase", "check-plan", "--plan", WORLD["plan_path"]).stdout)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["plan_sha256"], WORLD["plan_sha"])
        self.assertEqual((report["max_seq"], report["required_max_seq"]), (16897, 16897))
        self.assertEqual((report["primary_rows"], report["resume_rows"]), (HELD, TEACHER_N + 1))
        self.assertEqual(sorted(report["cells"]), sorted(J.PHASES))
        for phase in J.PHASES:
            ours = [c["name"] for c in phase_cells(plan, phase)]
            self.assertEqual(report["cells"][phase], ours, phase)
            self.assertEqual(report["cells"][phase], list(J.expected_specs(plan, phase)), phase)
        self.assertEqual(len(report["cells"]["s1a"]), 7 + 4 + 2)
        self.assertEqual(report["cells"]["s1a"][:7], report["cells"]["s1b"])
        self.assertEqual(report["cells"]["baseline"][-2:], ["resume-cold", "resume-split"])

    def test_native_rejects_plans_that_break_the_contract(self):
        native = WORLD["native"]
        for mutate, needle in ((lambda p: p["resume"].__setitem__("split_at", 1536), "resume.split_at is 1536"),
                               (lambda p: p.__setitem__("max_seq", 16896), "max_seq 16896 is below the required"),
                               (lambda p: p["primary"]["contexts"][0]["heldout_ids"].pop(), "heldout_ids has 511 ids"),
                               (lambda p: p["expect"].__setitem__("s1_tile_rows", 1024), "s1_tile_rows")):
            plan = json.loads(json.dumps(WORLD["plan"]))
            mutate(plan)
            path = os.path.join(WORLD["root"], f"bad-plan-{abs(hash(needle))}.json")
            with open(path, "w") as f:
                json.dump(plan, f)
            p = native.run("--phase", "check-plan", "--plan", path)
            self.assertNotEqual(p.returncode, 0, needle)
            self.assertIn(needle, p.stderr)

    def test_dump_schema_matches_the_rust_writer(self):
        src = pathlib.Path(RUST_SOURCE).read_text() if os.path.isfile(RUST_SOURCE) else None
        self.assertIsNotNone(src, f"native source {RUST_SOURCE} not found next to this script")
        plan, dirs = WORLD["plan"], WORLD["dirs"]
        s1a = {r["cell"]: r for r in read_jsonl(os.path.join(dirs["s1a"], "cells.jsonl"))}
        base = {r["cell"]: r for r in read_jsonl(os.path.join(dirs["baseline"], "cells.jsonl"))}
        want = {
            "det-p1537": CELL_BASE_KEYS | CELL_GREEDY_KEYS | CELL_DET_KEYS | {"retained_ids_match"},
            "det-p512": CELL_BASE_KEYS | CELL_GREEDY_KEYS | CELL_DET_KEYS,
            "primary-ctx16k-teacher": CELL_BASE_KEYS | CELL_TEACHER_KEYS | CELL_PRIMARY_KEYS,
            "primary-ctx4k-greedy": CELL_BASE_KEYS | CELL_GREEDY_KEYS | CELL_PRIMARY_KEYS,
            "resume-split": CELL_BASE_KEYS | CELL_TEACHER_KEYS | CELL_RESUME_KEYS,
            "resume-cold": CELL_BASE_KEYS | CELL_TEACHER_KEYS | CELL_RESUME_KEYS}
        emitted = set()
        for name, keys in want.items():
            rec = s1a[name]
            self.assertEqual(set(rec), keys, name)
            self.assertEqual(set(rec["prefill"]), PREFILL_KEYS, name)
            emitted |= set(rec) | set(rec["prefill"])
            for check in rec["prefill"]["receipt_checks"]:
                self.assertEqual(set(check), CHECK_KEYS)
        # raw receipts: the key sets of the retained j1968 layer-major call and the bypass
        for rec in list(s1a.values()) + list(base.values()):
            for receipt in rec["prefill"]["receipts"]:
                lm = receipt["route"] == "layer-major"
                self.assertEqual(set(receipt), LM_RECEIPT_KEYS if lm else INCUMBENT_RECEIPT_KEYS, (rec["cell"], receipt["route"]))
        self.assertEqual(s1a["det-p512"]["prefill"]["receipts"][0]["route"], "incumbent")        # 512 < 1537: the retained TooShort bypass shape
        self.assertEqual(s1a["det-p1537"]["prefill"]["receipts"][0]["route"], "layer-major")
        summary = json.loads(pathlib.Path(dirs["s1a"], "phase-summary.json").read_text())
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(set(summary["load"]), LOAD_KEYS)
        self.assertEqual(set(summary["load"]["profile"]), PROFILE_KEYS)
        self.assertEqual(set(summary["model"]["stat"]), STAT_KEYS)
        row = read_jsonl(os.path.join(dirs["s1a"], "resume-cold.rows.jsonl"))[0]
        self.assertEqual(set(row), ROW_KEYS)
        emitted |= SUMMARY_KEYS | LOAD_KEYS | PROFILE_KEYS | STAT_KEYS | ROW_KEYS | CHECK_KEYS | {"plan", "path", "sha256", "max_seq", "expect"}
        missing = json_keys_in_rust_source(emitted, src)
        self.assertEqual(missing, [], f"keys this writer emits that the Rust source never names: {missing}")
        for schema in (CELL_SCHEMA, ROW_SCHEMA, J.SUMMARY_SCHEMA, J.KL_SUMMARY_SCHEMA, J.KL_ROW_SCHEMA, J.PLAN_SCHEMA):
            self.assertIn(f'"{schema}"', src)
        # the native 'greedy' rows are ids+1, teacher cells 512 / 32
        self.assertEqual(s1a["det-p1537"]["rows"], 8)
        self.assertEqual(s1a["det-irregular"]["rows"], 3)
        self.assertEqual(s1a["primary-ctx16k-teacher"]["rows"], HELD)
        self.assertEqual(s1a["resume-cold"]["rows"], TEACHER_N + 1)
        self.assertEqual([c["end"] - c["start"] for c in base["primary-ctx16k-teacher"]["prefill"]["calls"]], [4096] * 4)
        self.assertEqual([c["end"] - c["start"] for c in base["primary-ctx4k-teacher"]["prefill"]["calls"]], [4096, 1])
        self.assertEqual(len(plan["primary"]["contexts"][0]["prefix_ids"]), 16384)

    def test_the_bridge_refuses_gpu_phases_and_model_flags(self):
        native = WORLD["native"]
        for phase in J.PHASES:
            with self.assertRaises(AssertionError):
                native.run("--phase", phase, "--plan", WORLD["plan_path"])
        with self.assertRaises(AssertionError):
            native.run("--phase", "score", "--model", "/x")


class NativeScoreOutput(unittest.TestCase):
    def test_full_score_files_and_counts(self):
        summary, rows = load_kl(WORLD["score"])
        plan = WORLD["plan"]
        self.assertEqual(summary["scope"], "full")
        self.assertEqual([p["pair"] for p in summary["pairs"]], list(J.q8_pair_specs(plan)))
        self.assertEqual(len(rows), 2 * HELD + 6 * (TEACHER_N + 1))
        for pair in summary["pairs"]:
            want = HELD if pair["pair"].startswith("primary-") else TEACHER_N + 1
            self.assertEqual((pair["rows"], pair["expected_rows"], pair["complete"], pair["infinite_rows"]), (want, want, True, 0), pair["pair"])
            self.assertEqual(pair["schema"], J.KL_SUMMARY_SCHEMA)
        self.assertEqual({r["schema"] for r in rows}, {J.KL_ROW_SCHEMA})
        # native determinism.json (only with --s1b-dir): all 7 identical, same order as the plan
        det, _ = J.read_json_file(os.path.join(WORLD["score"], "determinism.json"))
        self.assertTrue(det["all_identical"])
        self.assertEqual([c["case"] for c in det["cases"]], plan["determinism"]["case_ids"])
        self.assertTrue(all(c["identical"] and c["token_ids_equal"] and c["first_differing_row"] is None for c in det["cases"]))

    def test_every_native_kl_equals_an_independent_f64_recomputation(self):
        summary, rows = load_kl(WORLD["score"])
        dirs = WORLD["dirs"]
        specs = J.q8_pair_specs(WORLD["plan"])
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair"], []).append(r)
        for pair in summary["pairs"]:
            (rphase, rcell), (cphase, ccell), n = specs[pair["pair"]]
            _, rmeta, rlog = read_cell(dirs[rphase], rcell)
            _, cmeta, clog = read_cell(dirs[cphase], ccell)
            kls, agree = [], 0
            for i, k in enumerate(by_pair[pair["pair"]]):
                want = naive_kl(rlog[i], clog[i])
                self.assertEqual(k["kl_status"], "ok")
                self.assertAlmostEqual(k["kl_nats"], want, delta=1e-12, msg=f"{pair['pair']} row {i}")
                self.assertEqual((k["row"], k["context_position"], k["next_token"]), (i, rmeta[i]["context_position"], rmeta[i]["next_token"]))
                self.assertEqual((k["reference_top1_id"], k["candidate_top1_id"]), (top1(rlog[i]), top1(clog[i])))
                agree += top1(rlog[i]) == top1(clog[i])
                kls.append(k["kl_nats"])
            self.assertEqual(len(kls), n)
            self.assertEqual(pair["top1_agree_rows"], agree)
            self.assertAlmostEqual(pair["kl_sum_nats"], math.fsum(kls), delta=1e-12)
            self.assertAlmostEqual(pair["kl_mean_nats"], math.fsum(kls) / n, delta=1e-12)
            self.assertEqual(pair["kl_max_nats"], max(kls))
            self.assertEqual(kls[pair["kl_max_row"]], max(kls))

    def test_known_kl_identity_and_probe_row(self):
        summary, rows = load_kl(WORLD["score"])
        pairs = {p["pair"]: p for p in summary["pairs"]}
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair"], []).append(r)
        # identity: s1a cold == baseline cold bit for bit -> KL exactly 0 on every row
        ident = pairs["resume-s1cold"]
        self.assertEqual(ident["kl_sum_nats"], 0.0)
        self.assertEqual(ident["kl_max_nats"], 0.0)
        self.assertEqual(ident["top1_agree_rows"], TEACHER_N + 1)
        self.assertTrue(all(k["kl_nats"] == 0.0 and k["top1_match"] for k in by_pair["resume-s1cold"]))
        # the analytic probe row: KL((1/2,1/4,1/8,1/8) || (1/4,1/2,1/8,1/8)) = (ln 2) / 4
        probe = by_pair["resume-s1split"][PROBE_ROW]
        self.assertAlmostEqual(probe["kl_nats"], PROBE_KL, delta=1e-6)
        self.assertFalse(probe["top1_match"])
        self.assertEqual((probe["reference_top1_id"], probe["candidate_top1_id"]), (0, 1))
        self.assertAlmostEqual(probe["reference_margin"], math.log(2.0), delta=1e-6)
        self.assertGreater(pairs["resume-s1split"]["kl_max_nats"], 0.17)
        self.assertEqual(pairs["resume-s1split"]["kl_max_row"], PROBE_ROW)
        # a candidate whose probe row equals the reference's costs nothing there
        self.assertEqual(by_pair["resume-s1cold-vs-s1split"][PROBE_ROW]["kl_nats"], probe["kl_nats"])
        self.assertLess(by_pair["resume-betasplit"][PROBE_ROW]["kl_nats"], 1e-5)

    def test_native_margins_stay_within_one_ulp_of_the_rows_files(self):
        """serde_json without `float_roundtrip` parses a long decimal with at most a double rounding, so the margin native echoes may differ from the
        rows file by at most 1 ulp (the judge accepts exactly <= 1 ulp for these copied margins, never 2 ulp or more)."""
        mism = margin_mismatches()
        self.assertEqual([m for m in mism if m[3] > 1], [], "a margin differs by more than 1 ulp")

    def test_the_fixture_margins_are_nontrivial_and_native_follows_the_serde_parser_model(self):
        """Not uniform-only: margins span many distinct values, include the adjacent-f32 dyadic 2^-23 and an exact tie 0.0, and a large share are the
        long decimals on which the non-roundtrip parser moves the value. Native's echoed margin is the exact value or the parser-model value, and the model
        differs from the exact value on at least one row (so the 1-ulp copy difference is really exercised, not assumed)."""
        _, baseline_meta, _ = read_cell(WORLD["dirs"]["baseline"], "resume-cold")
        self.assertEqual(baseline_meta[7]["top12_margin"], 2.0 ** -23)
        self.assertEqual((baseline_meta[7]["top1_id"], baseline_meta[7]["top2_id"]), (0, 1))
        self.assertEqual(baseline_meta[9]["top12_margin"], 0.0)
        self.assertEqual((baseline_meta[9]["top1_id"], baseline_meta[9]["top2_id"]), (0, 1))
        specs = J.q8_pair_specs(WORLD["plan"])
        _, rows = load_kl(WORLD["score"])
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair"], []).append(r)
        seen, model_differs, native_differs, native_off_model = set(), 0, 0, []
        for pair, (rk, ck, _) in specs.items():
            _, rmeta, _ = read_cell(WORLD["dirs"][rk[0]], rk[1])
            _, cmeta, _ = read_cell(WORLD["dirs"][ck[0]], ck[1])
            for i, k in enumerate(by_pair[pair]):
                for key, meta in (("reference_margin", rmeta[i]["top12_margin"]), ("candidate_margin", cmeta[i]["top12_margin"])):
                    seen.add(meta)
                    model = serde_parse(repr(meta))
                    model_differs += model != meta
                    native_differs += k[key] != meta
                    if k[key] not in (meta, model):
                        native_off_model.append((pair, i, key))
        self.assertGreater(len(seen), 500)
        self.assertGreater(sum(1 for m in seen if len(repr(m).replace("-", "").replace(".", "").lstrip("0")) >= 16), 100)
        self.assertGreater(model_differs, 0, "the fixture never exercises the parser difference")
        self.assertEqual(native_off_model, [], "native echoed a margin that is neither the exact value nor the serde_json model value")
        # with this build (no float_roundtrip) native differs exactly where the model does
        self.assertLessEqual(native_differs, 2 * model_differs)

    def test_the_judge_accepted_native_output_that_carries_one_ulp_margin_differences(self):
        mism = margin_mismatches()
        q = WORLD["q1a"]
        self.assertEqual((q["verdict"], q["issues"]), ("PASS", {c: [] for c in J.Issues.CATS}))
        # mismatches (if this build has any) are all exactly 1 ulp and none stopped the judge
        self.assertTrue(all(m[3] == 1 for m in mism), mism[:3])
        self.assertEqual(q["primary_gate_status"], "PASS")
        self.assertEqual(WORLD["f32"]["verdict"], "REPORTED")

    def test_f32_pair_alone_equals_the_f32_pair_of_the_full_score(self):
        alone, alone_rows = load_kl(WORLD["score_f32"])
        full, full_rows = load_kl(WORLD["score_full_f32"])
        self.assertEqual(alone["scope"], "f32-pair-only")
        self.assertEqual(full["scope"], "full")
        self.assertEqual([p["pair"] for p in alone["pairs"]], [J.F32_PAIR])
        self.assertEqual(len(alone_rows), TEACHER_N + 1)
        self.assertEqual([p["pair"] for p in full["pairs"]], list(J.q8_pair_specs(WORLD["plan"])) + [J.F32_PAIR])
        self.assertEqual(alone["pairs"][0], full["pairs"][-1])
        self.assertEqual(alone_rows, [r for r in full_rows if r["pair"] == J.F32_PAIR])
        self.assertEqual(sorted(os.listdir(WORLD["score_f32"])), ["kl-summary.json", "kl.jsonl"])
        # f32: bit-identical except the probe row
        rows = alone_rows
        self.assertAlmostEqual(rows[PROBE_ROW]["kl_nats"], PROBE_KL, delta=1e-6)
        self.assertTrue(all(r["kl_nats"] == 0.0 for i, r in enumerate(rows) if i != PROBE_ROW))
        self.assertEqual(alone["pairs"][0]["top1_agree_rows"], TEACHER_N)
        _, rmeta, rlog = read_cell(WORLD["dirs"]["beta-f32"], "resume-cold")
        _, cmeta, clog = read_cell(WORLD["dirs"]["s1-f32"], "resume-split")
        self.assertEqual([naive_kl(a, b) for a, b in zip(rlog, clog)].count(0.0), TEACHER_N)

    def test_native_refusals(self):
        native, dirs = WORLD["native"], WORLD["dirs"]
        base = ["--plan", WORLD["plan_path"]]
        # a non-fresh output is refused and left untouched
        p = native.run("--phase", "score-f32", *base, "--beta-f32-dir", dirs["beta-f32"], "--s1-f32-dir", dirs["s1-f32"], "--out-dir", WORLD["score_f32"])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("the directory is not empty", p.stderr)
        self.assertEqual(sorted(os.listdir(WORLD["score_f32"])), ["kl-summary.json", "kl.jsonl"])
        # score-f32 takes no q8 directory; score needs the q8 pair
        p = native.run("--phase", "score-f32", *base, "--beta-f32-dir", dirs["beta-f32"], "--s1-f32-dir", dirs["s1-f32"], "--baseline-dir", dirs["baseline"],
                       "--out-dir", os.path.join(WORLD["root"], "x1"))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--baseline-dir does not apply to --phase score-f32", p.stderr)
        p = native.run("--phase", "score-f32", *base, "--beta-f32-dir", dirs["beta-f32"], "--out-dir", os.path.join(WORLD["root"], "x2"))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--s1-f32-dir is required", p.stderr)
        # an incomplete phase (no phase-summary.json) is refused
        broken = variant(dirs["baseline"], "no-summary-baseline")
        os.remove(os.path.join(broken, "phase-summary.json"))
        p = native.run(*native_score_args(broken, dirs["s1a"], os.path.join(WORLD["root"], "x3")))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("phase-summary.json", p.stderr)
        # a reference/candidate pair on different prompts is refused
        other = variant(dirs["s1a"], "other-prompt-s1a")
        lines = read_jsonl(os.path.join(other, "cells.jsonl"))
        for rec in lines:
            if rec["cell"] == "resume-split":
                rec["prompt_sha256"] = sha(b"another prompt")
        with open(os.path.join(other, "cells.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in lines))
        p = native.run(*native_score_args(dirs["baseline"], other, os.path.join(WORLD["root"], "x4")))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("prompt_sha256 differs between the reference and candidate cells", p.stderr)
        # one corrupted logits byte is caught by the native row hash
        corrupt = variant(dirs["s1a"], "corrupt-s1a")
        path = os.path.join(corrupt, "resume-split.logits.f32le")
        data = bytearray(pathlib.Path(path).read_bytes())
        data[5 * VOCAB * 4 + 1] ^= 0x01
        pathlib.Path(path).write_bytes(bytes(data))
        p = native.run(*native_score_args(dirs["baseline"], corrupt, os.path.join(WORLD["root"], "x5")))
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("does not match its recorded hash", p.stderr)


def margin_mismatches():
    out = []
    specs = J.q8_pair_specs(WORLD["plan"])
    _, rows = load_kl(WORLD["score"])
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    for pair, (rk, ck, _) in specs.items():
        _, rmeta, _ = read_cell(WORLD["dirs"][rk[0]], rk[1])
        _, cmeta, _ = read_cell(WORLD["dirs"][ck[0]], ck[1])
        for i, k in enumerate(by_pair[pair]):
            for key, meta in (("reference_margin", rmeta[i]["top12_margin"]), ("candidate_margin", cmeta[i]["top12_margin"])):
                if k[key] != meta:
                    out.append((pair, i, key, ulp_distance(k[key], meta)))
    return out


class JudgeAcceptsNativeOutput(unittest.TestCase):
    def test_cli_exit_and_files(self):
        self.assertEqual(WORLD["q1a_rc"], 0, WORLD["q1a_stdout"])
        self.assertEqual(WORLD["f32_rc"], 0, WORLD["f32_stdout"])
        self.assertTrue(J.seal_ok(WORLD["q1a"]))
        self.assertTrue(J.seal_ok(WORLD["f32"]))
        self.assertEqual(json.loads(WORLD["q1a_stdout"])["verdict"], "PASS")
        self.assertEqual(json.loads(WORLD["f32_stdout"])["verdict"], "REPORTED")

    def test_q1a_summary(self):
        q = WORLD["q1a"]
        self.assertEqual(q["issues"], {c: [] for c in J.Issues.CATS}, q["verdict_reasons"])
        self.assertEqual((q["schema"], q["stage"], q["verdict"]), (J.SCHEMA_Q1A, "final", "PASS"))
        self.assertEqual((q["primary_gate_status"], q["determinism_status"]), ("PASS", "PASS"))
        self.assertFalse(q["default_on_recommended"])
        self.assertTrue(q["human_review_required"])
        self.assertEqual(q["f32_exoneration"], "NONE_UNTIL_Q1F32")
        # the q8 resume pairs differ bit-wise (the probe row), so the class is awaiting the f32 review, not qualified
        self.assertEqual(q["resume_disposition"], "WAIT_F32_REVIEW")
        self.assertEqual(q["overall_S1"], "BLOCKED_F32_REVIEW")
        self.assertFalse(q["qualification_complete"])
        self.assertEqual(q["identity"]["plan_sha256"], WORLD["plan_sha"])
        self.assertEqual(q["identity"]["binary_sha256"], BIN_SHA)
        self.assertEqual(q["identity"]["resume"]["prompt_sha256"], WORLD["plan"]["resume"]["prompt_sha256"])
        self.assertEqual(q["identity"]["resume"]["vocab"], VOCAB)
        self.assertEqual(sorted(q["phases"]), ["baseline", "s1a", "s1b"])
        self.assertEqual(q["phases"]["s1a"]["cells"]["primary-ctx16k-teacher"]["rows"], HELD)
        # the sealed document binds the native score files it was computed from
        self.assertEqual(q["score"]["kl_summary_sha256"], J.sha256_file(os.path.join(WORLD["score"], "kl-summary.json")))
        self.assertEqual(q["score"]["kl_jsonl_sha256"], J.sha256_file(os.path.join(WORLD["score"], "kl.jsonl")))

    def test_primary_counts_means_and_intervals_come_from_the_native_numbers(self):
        q = WORLD["q1a"]["primary"]
        summary, rows = load_kl(WORLD["score"])
        pairs = {p["pair"]: p for p in summary["pairs"]}
        self.assertEqual(q["status"], "PASS")
        self.assertEqual(q["threshold_nats"], 0.010)
        self.assertTrue(q["ci"]["descriptive_only"] and not q["ci"]["gates"])
        self.assertEqual((q["ci"]["block_length"], q["ci"]["positions_per_context"], q["ci"]["replicates"]), (32, HELD, 10000))
        sums = 0.0
        for cid, _ in PRIMARY:
            ctx, native = q["contexts"][cid], pairs[f"primary-{cid}"]
            self.assertEqual(ctx["rows"], HELD)
            self.assertEqual(ctx["infinite_rows"], 0)
            self.assertEqual(ctx["kl_mean_nats"], native["kl_mean_nats"])
            self.assertEqual(ctx["kl_max_nats"], native["kl_max_nats"])
            self.assertEqual(ctx["top1_agree_rows"], native["top1_agree_rows"])
            self.assertAlmostEqual(ctx["top1_agreement"], native["top1_agree_rows"] / HELD)
            self.assertTrue(ctx["pass"])
            self.assertLess(ctx["kl_mean_nats"], 0.010)
            lo, hi = ctx["kl_mean_ci95"]
            self.assertTrue(math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi)
            tlo, thi = ctx["top1_agreement_ci95"]
            self.assertTrue(0.0 <= tlo <= thi <= 1.0)
            sums += native["kl_sum_nats"]
        pooled = q["pooled"]
        self.assertEqual(pooled["rows"], 2 * HELD)
        self.assertAlmostEqual(pooled["kl_mean_nats"], sums / (2 * HELD), delta=1e-15)
        self.assertTrue(pooled["pass"])
        self.assertEqual(pooled["top1_agree_rows"], sum(pairs[f"primary-{c}"]["top1_agree_rows"] for c, _ in PRIMARY))

    def test_resume_class_is_reported_with_the_known_rows(self):
        r = WORLD["q1a"]["resume"]
        self.assertEqual((r["status"], r["disposition"]), ("REPORTED", "WAIT_F32_REVIEW"))
        self.assertEqual(sorted(r["pairs"]), sorted(J.RESUME_PAIRS))
        ident = r["pairs"]["resume-s1cold"]
        self.assertTrue(ident["bit_identical_all"])
        self.assertEqual((ident["kl_sum_nats"], ident["kl_max_nats"], ident["bit_identical_rows"]), (0.0, 0.0, TEACHER_N + 1))
        split = r["pairs"]["resume-s1split"]
        self.assertFalse(split["bit_identical_all"])
        self.assertEqual(split["rows"], TEACHER_N + 1)
        self.assertEqual(split["kl_max_row"], PROBE_ROW)
        self.assertEqual(split["rows_kl_ge_gate"], [PROBE_ROW])
        self.assertAlmostEqual(split["trace"][PROBE_ROW]["kl_nats"], PROBE_KL, delta=1e-6)
        self.assertFalse(split["trace"][PROBE_ROW]["top1_match"])
        self.assertAlmostEqual(split["trace"][PROBE_ROW]["reference_margin"], math.log(2.0), delta=1e-6)
        # mean 0.173/32 < 0.010 but the max is past the gate: flagged by max, not by mean
        self.assertFalse(split["mean_kl_ge_gate"])
        self.assertTrue(split["max_kl_ge_gate"])
        self.assertEqual(sorted(r["pairs_with_kl_ge_gate"]), ["resume-s1cold-vs-s1split", "resume-s1split", "resume-s1split-vs-betasplit"])
        self.assertTrue(r["human_review_required"])
        self.assertEqual(r["f32_exoneration"], "NONE_UNTIL_Q1F32")

    def test_determinism_matches_the_native_determinism_json(self):
        d = WORLD["q1a"]["determinism"]
        native, _ = J.read_json_file(os.path.join(WORLD["score"], "determinism.json"))
        self.assertEqual((d["status"], d["all_identical"], d["provenance_identical"]), ("PASS", True, True))
        self.assertEqual([c["case"] for c in d["cases"]], WORLD["plan"]["determinism"]["case_ids"])
        self.assertTrue(all(c["identical"] for c in d["cases"]))
        self.assertEqual(native["all_identical"], d["all_identical"])
        for ours, theirs in zip(d["cases"], native["cases"]):
            self.assertEqual(ours["case"], theirs["case"])
            self.assertEqual(ours["stream_sha256"], theirs["stream_sha256"])
            self.assertEqual(ours["rows"], theirs["rows"])
        # the retained-ids comparison the native writer recorded is carried through (reported, never enforced)
        by_case = {c["case"]: c for c in d["cases"]}
        self.assertEqual(by_case["p1537"]["retained_ids_match"], [True, True])
        self.assertEqual(by_case["irregular"]["retained_ids_match"], [False, False])
        self.assertEqual(by_case["p512"]["retained_ids_match"], [None, None])

    def test_f32_summary_and_contrast(self):
        f = WORLD["f32"]
        q = WORLD["q1a"]
        self.assertEqual(f["issues"], {c: [] for c in J.Issues.CATS}, f["verdict_reasons"])
        self.assertEqual((f["schema"], f["verdict"]), (J.SCHEMA_F32, "REPORTED"))
        self.assertTrue(f["diagnostic_only"] and not f["performance_claim"] and not f["default_on_recommended"])
        self.assertEqual(f["bound_q1a_summary"]["sha256"], WORLD["q1a_sha"])
        self.assertEqual(f["bound_q1a_summary"]["verdict"], "PASS")
        pair = f["f32_pair"]
        self.assertEqual((pair["pair"], pair["rows"], pair["infinite_rows"]), (J.F32_PAIR, TEACHER_N + 1, 0))
        self.assertEqual(pair["bit_identical_rows"], TEACHER_N)
        self.assertEqual(pair["top1_agree_rows"], TEACHER_N)
        self.assertAlmostEqual(pair["kl_sum_nats"], PROBE_KL, delta=1e-6)
        self.assertAlmostEqual(pair["kl_mean_nats"], PROBE_KL / (TEACHER_N + 1), delta=1e-7)
        c = f["contrast"]
        q8 = q["resume"]["pairs"]["resume-s1split"]
        self.assertEqual(c["kl_mean_nats"]["q8"], q8["kl_mean_nats"])
        self.assertEqual(c["kl_mean_nats"]["f32"], pair["kl_mean_nats"])
        self.assertAlmostEqual(c["kl_mean_nats"]["f32_minus_q8"], pair["kl_mean_nats"] - q8["kl_mean_nats"], delta=1e-15)
        self.assertAlmostEqual(c["kl_mean_nats"]["f32_over_q8"], pair["kl_mean_nats"] / q8["kl_mean_nats"], delta=1e-12)
        self.assertEqual(c["top1_agree_rows"], {"q8": q8["top1_agree_rows"], "f32": TEACHER_N, "rows": TEACHER_N + 1})
        self.assertEqual(c["bit_identical"], {"q8": False, "f32": False})
        self.assertEqual(len(c["margins"]), TEACHER_N + 1)
        self.assertEqual(len(c["trace_f32"]), TEACHER_N + 1)
        self.assertEqual(c["row0"]["f32"]["row"], 0)
        self.assertEqual(c["row0"]["f32"]["kl_nats"], 0.0)       # the probe is row 3, row 0 is bit-identical in f32
        self.assertTrue(c["row0"]["top1_agreement_f32"])
        self.assertEqual(c["margins"][PROBE_ROW]["f32_reference"], c["trace_f32"][PROBE_ROW]["reference_margin"])
        self.assertAlmostEqual(c["trace_f32"][PROBE_ROW]["kl_nats"], PROBE_KL, delta=1e-6)
        self.assertEqual(f["score"]["kl_summary_sha256"], J.sha256_file(os.path.join(WORLD["score_f32"], "kl-summary.json")))


class JudgeRejects(unittest.TestCase):
    def out(self, name):
        return os.path.join(WORLD["root"], name)

    def test_a_flipped_determinism_row_fails_natively_and_in_the_judge(self):
        plan, psha, dirs, native = WORLD["plan"], WORLD["plan_sha"], WORLD["dirs"], WORLD["native"]
        flipped = os.path.join(WORLD["root"], "s1b-flipped")
        write_phase(flipped, plan, psha, "s1b", flip=("p1537", 2))
        score = os.path.join(WORLD["root"], "native-score-flip")
        native.ok(*native_score_args(dirs["baseline"], dirs["s1a"], score, s1b=flipped))
        det, _ = J.read_json_file(os.path.join(score, "determinism.json"))
        self.assertFalse(det["all_identical"])
        bad = [c for c in det["cases"] if not c["identical"]]
        self.assertEqual([c["case"] for c in bad], ["p1537"])
        self.assertEqual(bad[0]["first_differing_row"], 2)
        out = self.out("flip-q1a.json")
        rc, text = run_judge(judge_q1a_argv(dirs["baseline"], dirs["s1a"], score, out, s1b=flipped))
        self.assertEqual(rc, 1, text)
        doc, _ = J.read_json_file(out)
        self.assertEqual((doc["verdict"], doc["determinism_status"]), ("FAIL", "FAIL"))
        case = next(c for c in doc["determinism"]["cases"] if c["case"] == "p1537")
        self.assertFalse(case["identical"])
        self.assertEqual(case["first_differing_row"], 2)
        self.assertTrue(all(c["identical"] for c in doc["determinism"]["cases"] if c["case"] != "p1537"))

    def test_corrupt_row_bytes_make_the_judge_incomplete(self):
        dirs = WORLD["dirs"]
        corrupt = variant(dirs["s1a"], "judge-corrupt-s1a")
        path = os.path.join(corrupt, "primary-ctx4k-teacher.logits.f32le")
        data = bytearray(pathlib.Path(path).read_bytes())
        data[17 * VOCAB * 4 + 2] ^= 0x10
        pathlib.Path(path).write_bytes(bytes(data))
        out = self.out("corrupt-q1a.json")
        rc, text = run_judge(judge_q1a_argv(dirs["baseline"], corrupt, WORLD["score"], out, s1b=dirs["s1b"]))
        doc, _ = J.read_json_file(out)
        self.assertEqual((rc, doc["verdict"]), (3, "INCOMPLETE"), text)
        joined = " ".join(doc["issues"]["provenance"])
        self.assertIn("primary-ctx4k-teacher", joined)
        self.assertIn("logits_sha256 differs from the bytes", joined)

    def test_mismatched_source_binary_or_plan_is_refused(self):
        dirs = WORLD["dirs"]
        for label, mutate, needle in (
                ("source", lambda s: s["manifest"]["content"]["source"].__setitem__("commit", "e" * 40), "differ in source"),
                ("binary", lambda s: (s["binary"].__setitem__("sha256", sha(b"other")), s["manifest"]["content"]["binary"].__setitem__("sha256", sha(b"other"))),
                 "differ in binary_sha256"),
                ("plan", lambda s: s["plan"].__setitem__("sha256", sha(b"other plan")), "plan sha256 differs")):
            d = variant(dirs["s1a"], f"judge-{label}-s1a")
            path = os.path.join(d, "phase-summary.json")
            summary = json.loads(pathlib.Path(path).read_text())
            mutate(summary)
            pathlib.Path(path).write_text(json.dumps(summary, indent=1))
            out = self.out(f"{label}-q1a.json")
            rc, text = run_judge(judge_q1a_argv(dirs["baseline"], d, WORLD["score"], out, s1b=dirs["s1b"]))
            doc, _ = J.read_json_file(out)
            self.assertEqual((rc, doc["verdict"]), (3, "INCOMPLETE"), label)
            self.assertIn(needle, " ".join(doc["issues"]["provenance"]), label)
        # a plan whose sha256 is not the expected one never starts the judge and writes nothing
        out = self.out("wrongplan-q1a.json")
        rc, text = run_judge(judge_q1a_argv(dirs["baseline"], dirs["s1a"], WORLD["score"], out, s1b=dirs["s1b"], plan_sha="0" * 64))
        self.assertEqual(rc, 2, text)
        self.assertIn("plan sha256", text)
        self.assertFalse(os.path.exists(out))

    def test_native_score_that_names_another_plan_dump_is_not_silently_accepted(self):
        """A score computed from DIFFERENT dumps than the ones judged: the stream hashes recorded in kl-summary.json no longer match the phase data."""
        plan, psha, dirs, native = WORLD["plan"], WORLD["plan_sha"], WORLD["dirs"], WORLD["native"]
        other = os.path.join(WORLD["root"], "s1a-regenerated")
        shutil.copytree(dirs["s1a"], other)
        # rewrite resume-split with a different (still self-consistent) row set and refresh the cell record + summary
        cells = read_jsonl(os.path.join(other, "cells.jsonl"))
        cell = next(c for c in phase_cells(plan, "s1a") if c["name"] == "resume-split")
        os.remove(os.path.join(other, "resume-split.logits.f32le"))
        os.remove(os.path.join(other, "resume-split.rows.jsonl"))
        original = resume_rows
        try:
            globals()["resume_rows"] = lambda phase, name: perturb(original(phase, name), "regen", 0.01)
            rec = write_cell(other, "s1a", cell, None)
        finally:
            globals()["resume_rows"] = original
        cells = [rec if c["cell"] == "resume-split" else c for c in cells]
        pathlib.Path(other, "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells))
        summary = json.loads(pathlib.Path(other, "phase-summary.json").read_text())
        for entry in summary["cells"]:
            if entry["cell"] == "resume-split":
                entry["stream_sha256"] = rec["stream_sha256"]
        pathlib.Path(other, "phase-summary.json").write_text(json.dumps(summary, indent=1))
        out = self.out("regen-q1a.json")
        rc, text = run_judge(judge_q1a_argv(dirs["baseline"], other, WORLD["score"], out, s1b=dirs["s1b"]))
        doc, _ = J.read_json_file(out)
        self.assertEqual((rc, doc["verdict"]), (3, "INCOMPLETE"), text)
        self.assertIn("candidate cell/stream sha256 differs from the phase data", " ".join(doc["issues"]["resume"]))
        # while a native score of the regenerated dump is a coherent, different result
        score = os.path.join(WORLD["root"], "native-score-regen")
        native.ok(*native_score_args(dirs["baseline"], other, score))
        a, _ = load_kl(WORLD["score"])
        b, _ = load_kl(score)
        pa = {p["pair"]: p for p in a["pairs"]}["resume-s1split"]
        pb = {p["pair"]: p for p in b["pairs"]}["resume-s1split"]
        self.assertNotEqual(pa["candidate"]["stream_sha256"], pb["candidate"]["stream_sha256"])


if __name__ == "__main__":
    unittest.main()
