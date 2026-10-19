#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for q1_judge.py.  Run:  python3 test_q1_judge.py

Everything here is SYNTHETIC: a tiny vocabulary (8 logits), tiny plans (40 held-out rows, 20-token resume prompt) and a test-only KL writer that
stands in for the native `score` phase.  This module is ALSO the shared fixture library of test_q1_hold.py (`make_plan`, `write_phase`,
`write_score`, `fake_native_main` = a Python script that plays the native binary).  No GPU, no model, no journalctl; nothing here is a production
backend."""
import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import struct
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1_judge as J  # noqa: E402

VOCAB = 8
HELD = 40
PINNED = {"HIPFIRE_STATE_QUANT": "q8", "HIPFIRE_KV_MODE": "q8", "HIPFIRE_GRAPH": "0", "HIPFIRE_NORMALIZE_PROMPT": "0",
          "HIPFIRE_CHAT_CURRENT_DATE": "2026-10-17", "HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_PLE_FORMAT": "q4", "HIPFIRE_QWEN4_D2_DEFAULT": "0"}
BASELINE_ENV = {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "9", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096"}
S1_ABSENT = ["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"]
SELECTOR_ENV = "HIPFIRE_QWEN4_SLOWLINK_PREFILL"
MODEL_SHA = hashlib.sha256(b"synthetic-model-pin").hexdigest()
BIN_SHA = hashlib.sha256(b"synthetic-binary").hexdigest()
SOURCE = {"commit": "c" * 40, "example_sha256": "d" * 64, "note": "synthetic"}
DEFAULT_KNOBS = {"eps_s1": 0.01, "eps_split": 0.0, "exact": False, "negzero": False, "s1b_flip": None, "kl_const": None}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fsha(path):
    return sha(pathlib.Path(path).read_bytes())


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def noise(label, i, j):
    h = hashlib.sha256(f"{label}|{i}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 63 - 1.0


def fake_ids(label, n, mod=50000):
    return [int.from_bytes(hashlib.sha256(f"{label}:{k}".encode()).digest()[:4], "big") % mod for k in range(n)]


# ----------------------------------------------------------------------------------------------------------------------------------
# synthetic plan / phase dumps / native score
# ----------------------------------------------------------------------------------------------------------------------------------
def make_plan(model_path="/synthetic/model.mq4", model_size=1234, model_sha=MODEL_SHA):
    cases = []
    for k in range(7):
        cid = f"case{k}"
        if k == 6:
            cases.append({"id": cid, "messages": [{"role": "user", "content": "hello"}], "expected_prompt_tokens": 9})
            continue
        n = 12 + k
        c = {"id": cid, "prompt_ids": fake_ids(cid, n), "expected_prompt_tokens": n}
        if k == 5:
            c["resume_at"] = 5
        cases.append(c)
    contexts = []
    for cid, n in (("ctx16k", 12), ("ctx4k", 13)):
        prefix, held = fake_ids(cid + "p", n), fake_ids(cid + "h", HELD)
        contexts.append({"id": cid, "prefix_ids": prefix, "heldout_ids": held, "prefix_sha256": J.u32_sha256(prefix),
                         "heldout_sha256": J.u32_sha256(held), "sources": []})
    prompt, teacher = fake_ids("resume", 20), fake_ids("teach", 5)
    return {
        "schema": J.PLAN_SCHEMA, "max_seq": 16897,
        "artifact": {"path": model_path, "size": model_size, "sha256": model_sha},
        "expect": {"resident_layers": 9, "baseline_tile_rows": 4096, "s1_tile_rows": 768, "s1_layer_major_min_rows": 10, "ple_format": "q4",
                   "env": dict(PINNED), "env_absent": list(S1_ABSENT), "baseline_env": dict(BASELINE_ENV)},
        "determinism": {"case_ids": [c["id"] for c in cases], "max_new_tokens": 6, "thinking": False, "cases": cases},
        "primary": {"heldout_rows": HELD, "greedy_new_tokens": 6, "contexts": contexts},
        "resume": {"prompt_ids": prompt, "prompt_sha256": J.u32_sha256(prompt), "split_at": 7, "teacher_ids": teacher,
                   "teacher_sha256": J.u32_sha256(teacher), "retained_s1_ids": teacher + [7]},
    }


def greedy_ids(name):
    if name.startswith("det-case2"):
        return fake_ids(name, 3), "stop"
    return fake_ids(name, 6), "length"


def family(name):
    return "resume" if name.startswith("resume-") else name


def cell_rows(phase, name, n_rows, knobs):
    route = J.ARM[phase]
    rows = []
    for i in range(n_rows):
        row = [f32(3.0 * noise("base:" + family(name), i, j)) for j in range(VOCAB)]
        if route == "s1" and not knobs["exact"]:
            row = [f32(v + knobs["eps_s1"] * noise("s1:" + name, i, j)) for j, v in enumerate(row)]
        if name == "resume-split" and not knobs["exact"]:
            row = [f32(v + knobs["eps_split"] * noise("split:" + route, i, j)) for j, v in enumerate(row)]
        if knobs["negzero"] and name.startswith("resume-") and i == 0:
            row[0] = -0.0 if (phase == "s1a" and name == "resume-split") else 0.0
        if knobs["s1b_flip"] and phase == "s1b" and name == "det-" + knobs["s1b_flip"] and i == 2:
            row[0] = f32(row[0] + 0.5)
        rows.append(row)
    return rows


def row_stats(row):
    order = sorted(range(VOCAB), key=lambda j: (-row[j], j))
    t1, t2 = order[0], order[1]
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]  # noqa: E731
    return {"top1_id": t1, "top1_logit": row[t1], "top1_logit_bits": bits(row[t1]), "top2_id": t2, "top2_logit": row[t2],
            "top2_logit_bits": bits(row[t2]), "top12_margin": row[t1] - row[t2], "neg_inf_count": 0}


def receipt_for(arm, ex, s, e):
    if arm == "baseline":
        return {"route": "incumbent", "selected_by": "env0", "entry_pos": s, "prompt_rows": e - s, "tile_rows": 4096, "retained_scratch_rows": 4096,
                "resident_layers": None, "host_layers": None, "sweeps": None, "status": None, "bypass_reason": None}
    lm = (e - s) >= ex["s1_layer_major_min_rows"]
    return {"route": "layer-major" if lm else "incumbent", "selected_by": "env1", "entry_pos": s, "prompt_rows": e - s, "tile_rows": 768,
            "retained_scratch_rows": 768, "resident_layers": 9 if lm else 0, "host_layers": 3, "sweeps": 4, "status": "ok" if lm else None,
            "bypass_reason": None if lm else "TooShort"}


def write_cell(out_dir, plan, phase, name, spec, knobs):
    arm, ex = J.ARM[phase], plan["expect"]
    if spec["rows"] is None:
        ids, finish = greedy_ids(name)
        n_rows, feed, nexts = len(ids) + 1, list(ids), list(ids) + [None]
    else:
        ids, finish, n_rows, feed, nexts = None, None, spec["rows"], spec["feed"], spec["nexts"]
    rows = cell_rows(phase, name, n_rows, knobs)
    stream = hashlib.sha256()
    lines, offset = [], 0
    with open(os.path.join(out_dir, f"{name}.logits.f32le"), "xb") as fb:
        for i, row in enumerate(rows):
            buf = struct.pack(f"<{VOCAB}f", *row)
            fb.write(buf)
            stream.update(buf)
            st = row_stats(row)
            lines.append(json.dumps({"schema": "hipfire.qwen4.slowlink_quality.row.v1", "cell": name, "row": i,
                                     "input_token": None if i == 0 else feed[i - 1], "context_position": spec["prompt_tokens"] + i,
                                     "next_token": nexts[i], "logits_sha256": sha(buf), "bin_offset": offset, **st}))
            offset += len(buf)
    rows_bytes = ("\n".join(lines) + "\n").encode()
    with open(os.path.join(out_dir, f"{name}.rows.jsonl"), "xb") as fr:
        fr.write(rows_bytes)
    calls = spec["calls"]
    checks = [receipt_for(arm, ex, s, e) for s, e in calls]
    rec = {"schema": "hipfire.qwen4.slowlink_quality.cell.v1", "phase": phase, "arm": arm, "cell": name, "kind": spec["kind"], "context": spec["context"],
           "prompt_tokens": spec["prompt_tokens"], "prompt_sha256": spec["prompt_sha256"] or sha(b"messages:" + name.encode()),
           "prefill": {"structure": "synthetic", "calls": [{"start": s, "end": e, "rows": e - s} for s, e in calls],
                       "receipts": [dict(c) for c in checks], "receipt_checks": checks, "seconds": 0.1},
           "rows": n_rows, "vocab": VOCAB, "bin": f"{name}.logits.f32le", "bin_bytes": n_rows * VOCAB * 4, "rows_file": f"{name}.rows.jsonl",
           "rows_file_sha256": sha(rows_bytes), "stream_sha256": stream.hexdigest(), "rows_seconds": 0.1, "cell_seconds": 0.2,
           "final_position": spec["prompt_tokens"] + n_rows - 1}
    if spec["kind"] == "resume":
        rec["split_at"] = spec["split_at"]
    if spec["feed"] is not None:
        rec["teacher_feed_tokens"] = len(spec["feed"])
        rec["teacher_sha256"] = spec["teacher_sha256"]
    if ids is not None:
        rec.update({"token_ids": ids, "token_ids_sha256": J.u32_sha256(ids), "finish_reason": finish, "text": "x"})
    if spec["case"] is not None:
        rec.update({"case_id": spec["case"], "resume_at": spec["calls"][0][1] if len(spec["calls"]) > 1 else None, "retained_ids": None})
    if spec["context"] is not None:
        rec["context_id"] = spec["context"]
    return rec


def load_facts(plan, phase):
    arm = J.ARM[phase]
    profile = None if arm == "baseline" else {"tile_rows": 768, "resident_layers": 9, "purpose": "product", "window_rows": 1}
    return {"selector": "off" if arm == "baseline" else "force", "placement": "explicit-env" if arm == "baseline" else "slowlink-profile",
            "slowlink_profile": arm == "s1", "resident_layers": 9, "host_mapped_tensors": 5, "scratch_rows": 4096 if arm == "baseline" else 768,
            "ple_format": "q4", "state_format": {"gdn": "F32" if J.STATE_F32[phase] else "Q8", "qsa_kv": "Q8"}, "vocab": VOCAB, "eos_token": 2,
            "num_layers": 4, "arch_id": 1, "gpu_arch": "gfx1100", "profile": profile}


def write_phase(out_dir, plan, plan_sha, phase, knobs=None, binary_sha=BIN_SHA, manifest_sha="e" * 64, source=None, model_sha=None):
    """One synthetic native GPU phase directory (phase-summary.json LAST). `out_dir` must be absent or empty, like the native one."""
    knobs = {**DEFAULT_KNOBS, **(knobs or {})}
    if os.path.isdir(out_dir) and [n for n in os.listdir(out_dir) if n != "fake-run.json"]:
        raise RuntimeError(f"{out_dir} is not empty")
    os.makedirs(out_dir, exist_ok=True)
    specs = J.expected_specs(plan, phase)
    done, names, recs = [], [], []
    for name, spec in specs.items():
        rec = write_cell(out_dir, plan, phase, name, spec, knobs)
        recs.append(rec)
        done.append({"cell": name, "rows": rec["rows"], "stream_sha256": rec["stream_sha256"], "prompt_tokens": rec["prompt_tokens"],
                     "finish_reason": rec.get("finish_reason")})
        names.append(name)
    with open(os.path.join(out_dir, "cells.jsonl"), "xb") as f:
        for rec in recs:
            f.write((json.dumps(rec) + "\n").encode())
    env = dict(J.expected_effective_env(plan, phase))
    if J.ARM[phase] == "s1":
        for n in plan["expect"]["env_absent"]:
            env[n] = None
    art = plan["artifact"]
    summary = {
        "schema": J.SUMMARY_SCHEMA, "complete": True, "phase": phase, "arm": J.ARM[phase], "state_f32": J.STATE_F32[phase],
        "plan": {"path": "/synthetic/plan.json", "sha256": plan_sha},
        "manifest": {"path": "/synthetic/manifest.json", "sha256": manifest_sha,
                     "content": {"schema": "m", "binary": {"sha256": binary_sha}, "source": source or SOURCE, "launch": {}}},
        "binary": {"path": "/synthetic/bin", "sha256": binary_sha},
        "model": {"path": art["path"], "pinned_sha256": model_sha or art["sha256"], "hashed_here": False,
                  "stat": {"dev": 1, "ino": 2, "size": art["size"], "mtime_sec": 3, "mtime_nsec": 4, "ctime_sec": 5, "ctime_nsec": 6,
                           "ctime_is_metadata_only": True}},
        "environment": env, "load": load_facts(plan, phase), "cells": done, "cell_order": names, "wall_seconds": 1.0}
    with open(os.path.join(out_dir, "phase-summary.json"), "xb") as f:
        f.write(json.dumps(summary, indent=1).encode())


def read_cell(dirpath, name):
    with open(os.path.join(dirpath, "cells.jsonl")) as f:
        rec = next(json.loads(line) for line in f if json.loads(line)["cell"] == name)
    with open(os.path.join(dirpath, rec["rows_file"])) as f:
        meta = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(dirpath, rec["bin"]), "rb") as f:
        data = f.read()
    logits = [list(struct.unpack_from(f"<{VOCAB}f", data, i * VOCAB * 4)) for i in range(rec["rows"])]
    return rec, meta, logits


def kl_div(ref, cand):
    def logp(row):
        m = max(row)
        z = math.log(sum(math.exp(v - m) for v in row))
        return [v - m - z for v in row]
    lp, lq = logp(ref), logp(cand)
    return math.fsum(math.exp(a) * (a - b) for a, b in zip(lp, lq))


def write_score(out_dir, plan, dirs, specs, scope, kl_const=None):
    """Test-only stand-in for the native score phase: kl.jsonl + kl-summary.json in the native format."""
    os.makedirs(out_dir)
    kl_lines, summaries = [], []
    for name, (rk, ck, want) in specs.items():
        rrec, rmeta, rlog = read_cell(dirs[rk[0]], rk[1])
        crec, cmeta, clog = read_cell(dirs[ck[0]], ck[1])
        kls, infinite, agree = [], 0, 0
        for i in range(want):
            if kl_const and name in kl_const:
                kl = float("inf") if kl_const[name] == "inf" else float(kl_const[name])
            else:
                kl = kl_div(rlog[i], clog[i])
            finite = math.isfinite(kl)
            if finite:
                kls.append(kl)
            else:
                infinite += 1
            match = rmeta[i]["top1_id"] == cmeta[i]["top1_id"]
            agree += match
            kl_lines.append(json.dumps({"schema": J.KL_ROW_SCHEMA, "pair": name, "row": i, "context_position": rmeta[i]["context_position"],
                                        "next_token": rmeta[i]["next_token"], "kl_nats": kl if finite else None, "kl_status": "ok" if finite else "infinite",
                                        "reference_top1_id": rmeta[i]["top1_id"], "candidate_top1_id": cmeta[i]["top1_id"], "top1_match": match,
                                        "reference_margin": rmeta[i]["top12_margin"], "candidate_margin": cmeta[i]["top12_margin"]}))
        mx = max(kls) if kls else None
        summaries.append({"schema": J.KL_SUMMARY_SCHEMA, "pair": name, "reference": {"cell": rk[1], "stream_sha256": rrec["stream_sha256"]},
                          "candidate": {"cell": ck[1], "stream_sha256": crec["stream_sha256"]}, "rows": want, "expected_rows": want, "complete": True,
                          "infinite_rows": infinite, "kl_sum_nats": math.fsum(kls) if kls else None,
                          "kl_mean_nats": math.fsum(kls) / want if infinite == 0 else None, "kl_max_nats": mx,
                          "kl_max_row": kls.index(mx) if kls else None, "top1_agree_rows": agree})
    with open(os.path.join(out_dir, "kl.jsonl"), "x") as f:
        f.write("\n".join(kl_lines) + "\n")
    with open(os.path.join(out_dir, "kl-summary.json"), "x") as f:
        json.dump({"scope": scope, "pairs": summaries}, f, indent=1)


def write_plan_file(path, plan):
    data = json.dumps(plan, indent=1).encode()
    with open(path, "xb") as f:
        f.write(data)
    return sha(data)


def build_world(root, knobs=None, s1b=True, plan=None):
    """plan + baseline + s1a (+ s1b) + q8 score dirs under root. Returns the paths."""
    os.makedirs(root)
    plan = plan or make_plan()
    plan_path = os.path.join(root, "plan.json")
    psha = write_plan_file(plan_path, plan)
    knobs = {**DEFAULT_KNOBS, **(knobs or {})}
    w = {"root": root, "plan": plan, "plan_path": plan_path, "plan_sha": psha, "baseline": os.path.join(root, "baseline"),
         "s1a": os.path.join(root, "s1a"), "s1b": os.path.join(root, "s1b"), "score": os.path.join(root, "score")}
    for ph in ("baseline", "s1a") + (("s1b",) if s1b else ()):
        write_phase(w[ph], plan, psha, ph, knobs)
    write_score(w["score"], plan, {"baseline": w["baseline"], "s1a": w["s1a"]}, J.q8_pair_specs(plan), "full", knobs["kl_const"])
    return w


def build_f32(root, plan, psha, knobs=None, source=None):
    knobs = {**DEFAULT_KNOBS, **(knobs or {})}
    w = {"beta": os.path.join(root, "beta-f32"), "s1": os.path.join(root, "s1-f32"), "score": os.path.join(root, "score-f32")}
    write_phase(w["beta"], plan, psha, "beta-f32", knobs, source=source)
    write_phase(w["s1"], plan, psha, "s1-f32", knobs, source=source)
    write_score(w["score"], plan, {"beta-f32": w["beta"], "s1-f32": w["s1"]}, J.f32_pair_specs(plan), "f32-pair-only", knobs["kl_const"])
    return w


# ----------------------------------------------------------------------------------------------------------------------------------
# the fake native binary (used by test_q1_hold.py through a tiny launcher script)
# ----------------------------------------------------------------------------------------------------------------------------------
def _argmap(argv):
    out = {}
    for i, tok in enumerate(argv):
        if tok.startswith("--") and i + 1 < len(argv):
            out[tok] = argv[i + 1]
    return out


def _fail_mode(phase):
    for item in os.environ.get("Q1FAKE_FAIL", "").split(","):
        if "=" in item:
            p, mode = item.split("=", 1)
            if p == phase:
                return mode
    return None


def _log(phase):
    path = os.environ.get("Q1FAKE_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"phase": phase, "pid": os.getpid()}) + "\n")


def fake_native_main(argv):
    a = _argmap(argv)
    phase = a.get("--phase")
    _log(phase)
    knobs = {**DEFAULT_KNOBS, **json.loads(os.environ.get("Q1FAKE_KNOBS", "{}"))}
    mode = _fail_mode(phase)
    if mode == "crash":
        sys.stderr.write("synthetic crash\n")
        return 1
    if mode == "oom":
        sys.stderr.write("hipErrorOutOfMemory: out of memory while loading\n")
        return 1
    if mode == "hang":
        time.sleep(120)
        return 0
    with open(a["--plan"], "rb") as f:
        plan_bytes = f.read()
    plan = json.loads(plan_bytes)
    psha = sha(plan_bytes)
    if phase == "check-plan":
        print(json.dumps({"plan": a["--plan"], "plan_sha256": psha, "status": "valid", "max_seq": plan["max_seq"],
                          "cells": {p: list(J.expected_specs(plan, p)) for p in J.PHASES}, "primary_rows": plan["primary"]["heldout_rows"],
                          "resume_rows": len(plan["resume"]["teacher_ids"]) + 1}))
        return 0
    if phase in ("score", "score-f32"):
        if mode == "score-crash":
            return 1
        if phase == "score":
            dirs = {"baseline": a["--baseline-dir"], "s1a": a["--s1a-dir"]}
            write_score(a["--out-dir"], plan, dirs, J.q8_pair_specs(plan), "full", knobs["kl_const"])
        else:
            dirs = {"beta-f32": a["--beta-f32-dir"], "s1-f32": a["--s1-f32-dir"]}
            write_score(a["--out-dir"], plan, dirs, J.f32_pair_specs(plan), "f32-pair-only", knobs["kl_const"])
        return 0
    # ---- GPU phase: assert everything the real native asserts that the controller is responsible for
    with open(a["--manifest"], "rb") as f:
        mbytes = f.read()
    manifest = json.loads(mbytes)
    problems = []
    me = os.path.realpath(sys.argv[0])
    with open(me, "rb") as f:
        if sha(f.read()) != manifest["binary"]["sha256"]:
            problems.append("executable sha256 != manifest.binary.sha256")
    st = os.stat(plan["artifact"]["path"])
    mm = manifest["model"]
    if (mm["dev"], mm["ino"], mm["size"], mm["mtime_sec"], mm["mtime_nsec"]) != (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns // 10 ** 9, st.st_mtime_ns % 10 ** 9):
        problems.append("manifest.model tuple != live stat")
    if a["--model"] != plan["artifact"]["path"]:
        problems.append("--model is not the plan artifact")
    eff = J.expected_effective_env(plan, phase)
    for k, v in eff.items():
        if os.environ.get(k) != v:
            problems.append(f"env {k}={os.environ.get(k)!r} expected {v!r}")
    arm = J.ARM[phase]
    if os.environ.get(SELECTOR_ENV) != ("0" if arm == "baseline" else "1"):
        problems.append(f"selector {os.environ.get(SELECTOR_ENV)!r}")
    if arm == "s1":
        for k in plan["expect"]["env_absent"]:
            if k in os.environ:
                problems.append(f"{k} must be unset on S1 routes")
    allowed = set(eff) | {SELECTOR_ENV, "HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_DEVICES", "HIPFIRE_NO_REGISTRY_FETCH",
                          "HIPFIRE_FABRIC_ADMISSION", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"}
    for k in os.environ:
        if k.startswith(("HIPFIRE_", "HIP_", "ROCR_", "ROC_", "HSA_", "GPU_", "AMD_")) and k not in allowed:
            problems.append(f"unexpected env {k}")
    if problems:
        sys.stderr.write("fake native refused: " + "; ".join(problems) + "\n")
        return 1
    out_dir = a["--out-dir"]
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "fake-run.json"), "w") as f:
        json.dump({"phase": phase, "pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0), "cwd": os.getcwd(), "argv": argv,
                   "env_names": sorted(os.environ), "home": os.environ.get("HOME")}, f)
    if mode == "nosummary":
        return 0
    if mode == "watcher":
        with open(os.environ["Q1FAKE_WATCHER_LOG"], "a") as f:
            f.write("amdgpu 0000:67:00.0: GPU reset begin\n")
        time.sleep(120)
        return 0
    if mode == "churn":
        link = plan["artifact"]["path"] + ".lnk"
        os.link(plan["artifact"]["path"], link)
        os.unlink(link)
    if mode == "touch_model":
        os.utime(plan["artifact"]["path"], (1, 1))
    if mode == "tamper_source":
        with open(os.environ["Q1FAKE_TAMPER_FILE"], "ab") as f:
            f.write(b"tampered\n")
    write_phase_into_existing(out_dir, plan, psha, phase, knobs, manifest, sha(mbytes), a["--manifest"])
    return 0


def write_phase_into_existing(out_dir, plan, psha, phase, knobs, manifest, manifest_sha, manifest_path):
    write_phase(out_dir, plan, psha, phase, knobs, binary_sha=manifest["binary"]["sha256"], manifest_sha=manifest_sha, source=manifest["source"])
    # the real native echoes the manifest path; the synthetic writer used a placeholder, so patch the summary BEFORE anyone reads it
    path = os.path.join(out_dir, "phase-summary.json")
    with open(path) as f:
        doc = json.load(f)
    doc["manifest"]["path"] = manifest_path
    doc["model"]["stat"].update({"dev": manifest["model"]["dev"], "ino": manifest["model"]["ino"], "mtime_sec": manifest["model"]["mtime_sec"],
                                 "mtime_nsec": manifest["model"]["mtime_nsec"]})
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


# ----------------------------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------------------------
_COUNTER = [0]


def edit_json(path, fn):
    with open(path) as f:
        doc = json.load(f)
    fn(doc)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def edit_jsonl(path, fn):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = fn(rows)
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="q1j-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def world(self, **kw):
        _COUNTER[0] += 1
        return build_world(os.path.join(self.tmp, f"w{_COUNTER[0]}"), **kw)

    def judge(self, w, with_s1b=True):
        return J.judge_q1a(w["plan_path"], w["plan_sha"], w["baseline"], w["s1a"], w["score"], w["s1b"] if with_s1b else None)


class SplitMixAndBootstrap(unittest.TestCase):
    def test_splitmix64_known_vector(self):
        r = J.SplitMix64(0)
        self.assertEqual(r.next(), 0xE220A8397B1DCDAF)
        self.assertEqual(r.next(), 0x6E789E6AA1B965F4)

    def test_below_is_bounded_and_deterministic(self):
        a, b = J.SplitMix64(7), J.SplitMix64(7)
        va = [a.below(481) for _ in range(500)]
        self.assertEqual(va, [b.below(481) for _ in range(500)])
        self.assertTrue(all(0 <= v < 481 for v in va))
        self.assertGreater(len(set(va)), 100)

    def test_bootstrap_is_deterministic_and_stratified(self):
        strata = {"a": [(0.0, 1.0)] * 40, "b": [(0.02, 0.0)] * 40}
        x = J.stratified_block_bootstrap(strata, reps=200)
        y = J.stratified_block_bootstrap(strata, reps=200)
        self.assertEqual(x, y)
        self.assertEqual(x["a"]["kl"], (0.0, 0.0))
        for v in x["b"]["kl"]:
            self.assertAlmostEqual(v, 0.02)
        for v in x["pooled"]["kl"]:
            self.assertAlmostEqual(v, 0.01)
        for v in x["pooled"]["top1"]:
            self.assertAlmostEqual(v, 0.5)

    def test_bootstrap_uses_moving_blocks_not_iid_points(self):
        # a strongly autocorrelated series (first half 0, second half 1): iid resampling would give a narrow interval around 0.5, moving blocks of
        # 32 out of 64 give a much wider one
        series = [(0.0 if i < 32 else 1.0, 0.0) for i in range(64)]
        wide = J.stratified_block_bootstrap({"a": series}, reps=400)["a"]["kl"]
        self.assertGreater(wide[1] - wide[0], 0.3)

    def test_bootstrap_refuses_short_strata(self):
        with self.assertRaises(ValueError):
            J.stratified_block_bootstrap({"a": [(0.0, 1.0)] * 10}, reps=10)

    def test_label_seed_is_stable(self):
        self.assertEqual(J.seed_from_label(J.BOOT_LABEL), J.seed_from_label(J.BOOT_LABEL))
        self.assertNotEqual(J.seed_from_label(J.BOOT_LABEL), J.seed_from_label(J.BOOT_LABEL + "x"))


class MarginWitness(Base):
    def test_unit_one_ulp_accepted_two_rejected(self):
        v = 1.2345
        up = math.nextafter(v, math.inf)
        up2 = math.nextafter(up, math.inf)
        dn = math.nextafter(v, -math.inf)
        dn2 = math.nextafter(dn, -math.inf)
        for a, b in ((v, v), (v, up), (v, dn), (up, v), (dn, v)):
            self.assertTrue(J.margin_witness_equal(a, b), (a, b))
        for a, b in ((v, up2), (v, dn2), (up2, v), (v, v + 1e-9)):
            self.assertFalse(J.margin_witness_equal(a, b), (a, b))

    def test_unit_zero_and_nonfinite_and_non_numbers(self):
        self.assertTrue(J.margin_witness_equal(0.0, 0.0))
        self.assertTrue(J.margin_witness_equal(0.0, 5e-324))
        self.assertTrue(J.margin_witness_equal(0.0, -5e-324))
        self.assertFalse(J.margin_witness_equal(0.0, 1e-323))
        for bad in (float("nan"), float("inf"), float("-inf"), True, False, None, "1.0", [1.0]):
            self.assertFalse(J.margin_witness_equal(bad, 1.0), bad)
            self.assertFalse(J.margin_witness_equal(1.0, bad), bad)
        self.assertFalse(J.margin_witness_equal(float("inf"), float("inf")))
        self.assertFalse(J.margin_witness_equal(True, 1))

    def bump(self, w, field, n):
        def f(rows):
            for r in rows:
                if r["pair"] == "primary-ctx16k" and r["row"] == 0:
                    v = r[field]
                    for _ in range(n):
                        v = math.nextafter(v, math.inf)
                    r[field] = v
            return rows
        edit_jsonl(os.path.join(w["score"], "kl.jsonl"), f)

    def test_kl_jsonl_margin_one_ulp_is_accepted_for_both_sides(self):
        for field in ("reference_margin", "candidate_margin"):
            w = self.world()
            self.bump(w, field, 1)
            doc = self.judge(w)
            self.assertEqual(doc["verdict"], "PASS", doc["issues"])

    def test_kl_jsonl_margin_two_ulp_is_rejected_for_both_sides(self):
        for field in ("reference_margin", "candidate_margin"):
            w = self.world()
            self.bump(w, field, 2)
            doc = self.judge(w)
            self.assertEqual(doc["verdict"], "INCOMPLETE")
            self.assertTrue(any("margins differ" in m for m in doc["issues"]["primary"]))

    def test_the_tolerance_does_not_leak_to_ids_or_hashes(self):
        w = self.world()
        def flip(rows):
            for r in rows:
                if r["pair"] == "primary-ctx16k" and r["row"] == 0:
                    r["reference_top1_id"] += 1
            return rows
        edit_jsonl(os.path.join(w["score"], "kl.jsonl"), flip)
        self.assertEqual(self.judge(w)["verdict"], "INCOMPLETE")


class PlanAndSpecs(Base):
    def test_expected_cells_follow_the_native_order(self):
        plan = make_plan()
        self.assertEqual(list(J.expected_specs(plan, "baseline")), ["primary-ctx16k-teacher", "primary-ctx16k-greedy", "primary-ctx4k-teacher",
                                                                    "primary-ctx4k-greedy", "resume-cold", "resume-split"])
        self.assertEqual(list(J.expected_specs(plan, "s1a"))[:7], [f"det-case{k}" for k in range(7)])
        self.assertEqual(list(J.expected_specs(plan, "s1a"))[-2:], ["resume-split", "resume-cold"])
        self.assertEqual(list(J.expected_specs(plan, "s1b")), [f"det-case{k}" for k in range(7)])
        self.assertEqual(list(J.expected_specs(plan, "beta-f32")), ["resume-cold"])
        self.assertEqual(list(J.expected_specs(plan, "s1-f32")), ["resume-split"])

    def test_primary_calls_per_arm_and_resume_structure(self):
        plan = make_plan()
        self.assertEqual(J.expected_specs(plan, "baseline")["primary-ctx16k-teacher"]["calls"], [(0, 12)])
        self.assertEqual(J.chunk_ranges(16384, 4096), [(0, 4096), (4096, 8192), (8192, 12288), (12288, 16384)])
        self.assertEqual(J.chunk_ranges(4097, 4096), [(0, 4096), (4096, 4097)])
        s = J.expected_specs(plan, "s1a")
        self.assertEqual(s["resume-split"]["calls"], [(0, 7), (7, 20)])
        self.assertEqual(s["resume-cold"]["calls"], [(0, 20)])
        self.assertEqual(s["det-case5"]["calls"], [(0, 5), (5, 17)])

    def test_plan_sha_must_match_the_independent_value(self):
        w = self.world()
        with self.assertRaises(J.JudgeError):
            J.load_plan(w["plan_path"], "0" * 64)
        with self.assertRaises(J.JudgeError):
            J.load_plan(w["plan_path"], "nothex")

    def test_u32_hash_is_little_endian(self):
        self.assertEqual(J.u32_sha256([1, 2]), sha(b"\x01\x00\x00\x00\x02\x00\x00\x00"))


class Q1aVerdicts(Base):
    def test_pass_with_all_required_pairs_primary_rows_and_seven_cases(self):
        w = self.world()
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "PASS", doc["issues"])
        self.assertEqual(doc["stage"], "final")
        self.assertFalse(doc["default_on_recommended"])
        self.assertTrue(doc["human_review_required"])
        self.assertEqual(doc["f32_exoneration"], "NONE_UNTIL_Q1F32")
        self.assertEqual(doc["primary"]["contexts"]["ctx16k"]["rows"], HELD)
        self.assertEqual(doc["primary"]["contexts"]["ctx4k"]["rows"], HELD)
        self.assertEqual(sorted(doc["resume"]["pairs"]), sorted(J.RESUME_PAIRS))
        self.assertEqual(len(doc["resume"]["pairs"]), 6)
        for p in doc["resume"]["pairs"].values():
            self.assertEqual(p["rows"], 6)
            self.assertEqual(len(p["trace"]), 6)
            self.assertEqual({"reference_margin", "candidate_margin", "kl_nats", "top1_match"} <= set(p["trace"][0]), True)
        self.assertEqual(len(doc["determinism"]["cases"]), 7)
        self.assertTrue(doc["determinism"]["all_identical"])
        self.assertTrue(J.seal_ok(doc))

    def test_ci_block_is_descriptive_and_reported_for_each_context_and_pooled(self):
        doc = self.judge(self.world())
        ci = doc["primary"]["ci"]
        self.assertTrue(ci["descriptive_only"])
        self.assertFalse(ci["gates"])
        self.assertEqual((ci["block_length"], ci["replicates"]), (32, 10000))
        for key in ("ctx16k", "ctx4k"):
            c = doc["primary"]["contexts"][key]
            self.assertLessEqual(c["kl_mean_ci95"][0], c["kl_mean_ci95"][1])
            self.assertLessEqual(c["top1_agreement_ci95"][0], c["top1_agreement_ci95"][1])
        self.assertEqual(doc["primary"]["pooled"]["rows"], 2 * HELD)
        self.assertIsNone(doc["primary"]["top1_policy"]["threshold"])
        self.assertTrue(doc["primary"]["top1_policy"]["q2_pending"])

    def test_judge_is_deterministic(self):
        w = self.world()
        a, b = self.judge(w), self.judge(w)
        self.assertEqual(a, b)

    def test_resume_stage_without_s1b_is_incomplete_not_failed(self):
        doc = self.judge(self.world(), with_s1b=False)
        self.assertEqual(doc["stage"], "resume")
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertEqual(doc["determinism_status"], "PENDING_S1B")
        self.assertEqual(doc["primary_gate_status"], "PASS")
        self.assertIsNotNone(doc["resume_disposition"])

    def test_primary_numeric_fail_still_reports_resume_and_final_with_s1b(self):
        w = self.world(knobs={"kl_const": {"primary-ctx16k": 0.0, "primary-ctx4k": 0.02}})
        early = self.judge(w, with_s1b=False)
        self.assertEqual(early["verdict"], "FAIL")
        self.assertEqual(early["resume"]["status"], "REPORTED")
        final = self.judge(w)
        self.assertEqual(final["verdict"], "FAIL")
        self.assertEqual(final["stage"], "final")
        self.assertEqual(final["determinism_status"], "PASS")

    def test_pooled_mean_cannot_hide_a_failed_context(self):
        w = self.world(knobs={"kl_const": {"primary-ctx16k": 0.0, "primary-ctx4k": 0.016}})
        doc = self.judge(w)
        self.assertLessEqual(doc["primary"]["pooled"]["kl_mean_nats"], J.GATE_NATS)
        self.assertTrue(doc["primary"]["pooled"]["pass"])
        self.assertFalse(doc["primary"]["contexts"]["ctx4k"]["pass"])
        self.assertTrue(doc["primary"]["contexts"]["ctx16k"]["pass"])
        self.assertEqual(doc["primary"]["status"], "FAIL")
        self.assertEqual(doc["verdict"], "FAIL")

    def test_each_context_must_pass_on_its_own_point_mean(self):
        w = self.world(knobs={"kl_const": {"primary-ctx16k": 0.0099, "primary-ctx4k": 0.004}})
        doc = self.judge(w)
        self.assertTrue(doc["primary"]["contexts"]["ctx16k"]["pass"])
        self.assertEqual(doc["verdict"], "PASS", doc["issues"])

    def test_infinite_kl_row_fails_never_passes(self):
        w = self.world(knobs={"kl_const": {"primary-ctx16k": "inf"}})
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "FAIL")
        self.assertEqual(doc["primary"]["contexts"]["ctx16k"]["infinite_rows"], HELD)
        self.assertIsNone(doc["primary"]["pooled"]["kl_mean_nats"])

    def test_null_kl_on_an_ok_row_is_incomplete(self):
        w = self.world()
        def drop(rows):
            for r in rows:
                if r["pair"] == "primary-ctx16k" and r["row"] == 3:
                    r["kl_nats"] = None
            return rows
        edit_jsonl(os.path.join(w["score"], "kl.jsonl"), drop)
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(doc["issues"]["primary"])

    def test_missing_primary_row_is_incomplete(self):
        w = self.world()
        edit_jsonl(os.path.join(w["score"], "kl.jsonl"), lambda rows: [r for r in rows if not (r["pair"] == "primary-ctx4k" and r["row"] == 7)])
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertEqual(doc["primary_gate_status"], "INCOMPLETE")

    def test_all_required_pairs_must_be_present(self):
        w = self.world()
        edit_json(os.path.join(w["score"], "kl-summary.json"), lambda d: d.update(pairs=[p for p in d["pairs"] if p["pair"] != "resume-betasplit"]))
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(any("kl-summary pairs" in m for m in doc["issues"]["resume"]))

    def test_resume_row_count_is_32_style_and_enforced(self):
        w = self.world()
        edit_jsonl(os.path.join(w["score"], "kl.jsonl"), lambda rows: [r for r in rows if not (r["pair"] == "resume-s1split" and r["row"] == 5)])
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(doc["issues"]["resume"])

    def test_wrong_scope_is_flagged(self):
        w = self.world()
        edit_json(os.path.join(w["score"], "kl-summary.json"), lambda d: d.update(scope="f32-pair-only"))
        self.assertEqual(self.judge(w)["verdict"], "INCOMPLETE")

    def test_flipped_logit_byte_breaks_stream_and_row_hashes(self):
        w = self.world()
        p = os.path.join(w["baseline"], "primary-ctx16k-teacher.logits.f32le")
        data = bytearray(pathlib.Path(p).read_bytes())
        data[5] ^= 0xFF
        pathlib.Path(p).write_bytes(bytes(data))
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(any("stream sha256" in m for m in doc["issues"]["provenance"]))
        self.assertTrue(any("logits_sha256" in m for m in doc["issues"]["provenance"]))

    def test_row_offsets_and_hashes_are_verified(self):
        w = self.world()
        def bump(rows):
            rows[1]["bin_offset"] += 4
            return rows
        edit_jsonl(os.path.join(w["s1a"], "resume-cold.rows.jsonl"), bump)
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(any("bin_offset" in m for m in doc["issues"]["provenance"]))

    def test_top_logit_bits_are_checked_against_the_bytes(self):
        w = self.world()
        def bump(rows):
            rows[0]["top1_logit_bits"] ^= 1
            return rows
        edit_jsonl(os.path.join(w["s1a"], "primary-ctx4k-greedy.rows.jsonl"), bump)
        self.assertEqual(self.judge(w)["verdict"], "INCOMPLETE")

    def test_plan_sha_mismatch_in_a_phase_summary_is_incomplete(self):
        w = self.world()
        edit_json(os.path.join(w["s1a"], "phase-summary.json"), lambda d: d["plan"].update(sha256="1" * 64))
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(any("plan sha256" in m for m in doc["issues"]["provenance"]))

    def test_binary_identity_must_match_between_baseline_and_s1a(self):
        w = self.world()
        def swap(d):
            d["binary"]["sha256"] = "2" * 64
            d["manifest"]["content"]["binary"]["sha256"] = "2" * 64
        edit_json(os.path.join(w["s1a"], "phase-summary.json"), swap)
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(any("differ in binary_sha256" in m for m in doc["issues"]["provenance"]))

    def test_environment_and_placement_are_checked_per_route(self):
        w = self.world()
        edit_json(os.path.join(w["baseline"], "phase-summary.json"), lambda d: d["environment"].update(HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS="12"))
        self.assertEqual(self.judge(w)["verdict"], "INCOMPLETE")
        w2 = self.world()
        edit_json(os.path.join(w2["s1a"], "phase-summary.json"), lambda d: d["load"].update(placement="explicit-env"))
        self.assertEqual(self.judge(w2)["verdict"], "INCOMPLETE")
        w3 = self.world()
        edit_json(os.path.join(w3["s1a"], "phase-summary.json"), lambda d: d["environment"].update(HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS="9"))
        self.assertEqual(self.judge(w3)["verdict"], "INCOMPLETE")

    def test_incomplete_phase_without_summary_is_never_used(self):
        w = self.world()
        os.unlink(os.path.join(w["s1a"], "phase-summary.json"))
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")

    def test_prefill_structure_is_checked(self):
        w = self.world()
        def one_call(recs):
            for r in recs:
                if r["cell"] == "resume-split":
                    r["prefill"]["calls"] = r["prefill"]["calls"][:1]
            return recs
        edit_jsonl(os.path.join(w["s1a"], "cells.jsonl"), one_call)
        self.assertEqual(self.judge(w)["verdict"], "INCOMPLETE")

    def test_s1_receipt_must_be_layer_major_for_long_calls(self):
        w = self.world()
        def incumbent(recs):
            for r in recs:
                if r["cell"] == "resume-cold":
                    r["prefill"]["receipt_checks"][0]["route"] = "incumbent"
            return recs
        edit_jsonl(os.path.join(w["s1a"], "cells.jsonl"), incumbent)
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertTrue(any("incumbent route" in m for m in doc["issues"]["provenance"]))

    def test_prompt_and_teacher_hash_come_from_the_plan_arrays(self):
        w = self.world()
        edit_jsonl(os.path.join(w["baseline"], "cells.jsonl"),
                   lambda recs: [dict(r, teacher_sha256="3" * 64) if r["cell"] == "resume-cold" else r for r in recs])
        self.assertEqual(self.judge(w)["verdict"], "INCOMPLETE")

    def test_resume_disposition_exact_vs_review_and_negative_zero(self):
        exact = self.judge(self.world(knobs={"exact": True}))
        self.assertEqual(exact["verdict"], "PASS", exact["issues"])
        self.assertEqual(exact["resume_disposition"], "RESUME_BIT_IDENTICAL_ALL_PAIRS")
        for p in exact["resume"]["pairs"].values():
            self.assertTrue(p["bit_identical_all"])
        loose = self.judge(self.world())
        self.assertEqual(loose["resume_disposition"], "WAIT_F32_REVIEW")
        self.assertEqual(loose["verdict"], "PASS")   # resume divergence alone never changes pass/fail
        self.assertEqual(loose["f32_exoneration"], "NONE_UNTIL_Q1F32")
        self.assertFalse(loose["qualification_complete"])
        self.assertEqual(loose["overall_S1"], "BLOCKED_F32_REVIEW")
        self.assertEqual(exact["overall_S1"], "Q1A_CLEAN_HUMAN_REVIEW_PENDING")
        self.assertTrue(exact["qualification_complete"])
        self.assertEqual(exact["resume"]["pairs_with_kl_ge_gate"], [])
        for p in loose["resume"]["pairs"].values():
            self.assertEqual(len(p["trace"]), 6)
            self.assertIn("rows_kl_ge_gate", p)
        flagged = self.judge(self.world(knobs={"kl_const": {"primary-ctx16k": 0.0, "primary-ctx4k": 0.0, "resume-s1split": 0.02}}))
        self.assertIn("resume-s1split", flagged["resume"]["pairs_with_kl_ge_gate"])
        self.assertEqual(flagged["verdict"], "PASS")
        neg = self.judge(self.world(knobs={"exact": True, "negzero": True}))
        self.assertEqual(neg["resume_disposition"], "WAIT_F32_REVIEW")
        p = neg["resume"]["pairs"]["resume-s1split"]
        self.assertEqual(p["kl_max_nats"], 0.0)
        self.assertFalse(p["bit_identical_all"])
        self.assertEqual(p["bit_identical_rows"], 5)

    def test_resume_trace_carries_all_rows_and_margins(self):
        doc = self.judge(self.world())
        p = doc["resume"]["pairs"]["resume-s1cold-vs-s1split"]
        self.assertEqual([t["row"] for t in p["trace"]], list(range(6)))
        self.assertEqual(p["trace"][-1]["next_token"], None)
        self.assertTrue(all(isinstance(t["reference_margin"], float) and isinstance(t["candidate_margin"], float) for t in p["trace"]))


class Determinism(Base):
    def test_flipped_row_in_s1b_is_a_fail_with_the_row_located(self):
        w = self.world(knobs={"s1b_flip": "case3"})
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "FAIL", doc["issues"])
        self.assertEqual(doc["determinism_status"], "FAIL")
        by = {c["case"]: c for c in doc["determinism"]["cases"]}
        self.assertEqual(len(by), 7)
        self.assertFalse(by["case3"]["identical"])
        self.assertEqual(by["case3"]["first_differing_row"], 2)
        self.assertFalse(by["case3"]["stream_equal"])
        self.assertTrue(all(by[k]["identical"] for k in by if k != "case3"))

    def test_missing_case_in_s1b_is_incomplete(self):
        w = self.world()
        edit_jsonl(os.path.join(w["s1b"], "cells.jsonl"), lambda recs: [r for r in recs if r["cell"] != "det-case4"])
        doc = self.judge(w)
        self.assertEqual(doc["verdict"], "INCOMPLETE")
        self.assertNotEqual(doc["determinism_status"], "PASS")

    def test_stop_case_with_three_ids_is_compared_too(self):
        doc = self.judge(self.world())
        c2 = next(c for c in doc["determinism"]["cases"] if c["case"] == "case2")
        self.assertEqual(c2["finish_reason"], ["stop", "stop"])
        self.assertEqual(c2["rows"], [4, 4])
        self.assertTrue(c2["identical"])

    def test_s1b_provenance_difference_is_not_a_pass(self):
        w = self.world()
        edit_json(os.path.join(w["s1b"], "phase-summary.json"), lambda d: d["manifest"]["content"].update(source={"commit": "other"}))
        doc = self.judge(w)
        self.assertNotEqual(doc["verdict"], "PASS")


class F32(Base):
    def prior(self, w, **doc_edit):
        doc = self.judge(w)
        path = os.path.join(w["root"], "q1a-summary.json")
        J.write_json_new(path, doc)
        return path, fsha(path), doc

    def f32world(self, w, knobs=None, source=None):
        return build_f32(w["root"], w["plan"], w["plan_sha"], knobs, source)

    def test_contrast_is_reported_against_the_sealed_q1a_summary(self):
        w = self.world()
        path, psha, _ = self.prior(w)
        f = self.f32world(w)
        doc = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, psha)
        self.assertEqual(doc["verdict"], "REPORTED", doc["issues"])
        self.assertTrue(doc["diagnostic_only"])
        self.assertFalse(doc["performance_claim"])
        self.assertFalse(doc["default_on_recommended"])
        self.assertEqual(doc["bound_q1a_summary"]["sha256"], psha)
        c = doc["contrast"]
        self.assertEqual(len(c["margins"]), 6)
        self.assertEqual(c["q8_pair"], "resume-s1split")
        self.assertEqual(c["f32_pair"], J.F32_PAIR)
        self.assertIn("row0", c)
        self.assertIn("top1_agree_rows", c)

    def test_a_failed_sealed_final_q1a_is_still_accepted(self):
        w = self.world(knobs={"kl_const": {"primary-ctx16k": 0.0, "primary-ctx4k": 0.05}})
        path, psha, q1a = self.prior(w)
        self.assertEqual(q1a["verdict"], "FAIL")
        f = self.f32world(w)
        doc = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, psha)
        self.assertEqual(doc["verdict"], "REPORTED", doc["issues"])
        self.assertEqual(doc["bound_q1a_summary"]["verdict"], "FAIL")

    def test_resume_stage_summary_is_not_a_final_summary(self):
        w = self.world()
        doc = self.judge(w, with_s1b=False)
        path = os.path.join(w["root"], "q1a-resume.json")
        J.write_json_new(path, doc)
        f = self.f32world(w)
        out = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, fsha(path))
        self.assertEqual(out["verdict"], "INCOMPLETE")

    def test_wrong_expected_summary_sha_is_refused(self):
        w = self.world()
        path, _, _ = self.prior(w)
        f = self.f32world(w)
        with self.assertRaises(J.JudgeError):
            J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, "0" * 64)

    def test_edited_summary_fails_the_seal(self):
        w = self.world()
        path, _, original = self.prior(w)
        self.assertEqual(original["verdict"], "PASS")        # the edit below must REALLY change a sealed field
        edit_json(path, lambda d: d.update(verdict="FAIL"))
        self.assertNotEqual(json.loads(pathlib.Path(path).read_text())["verdict"], original["verdict"])
        f = self.f32world(w)
        out = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, fsha(path))
        self.assertEqual(out["verdict"], "INCOMPLETE")
        self.assertTrue(any("sealed" in m for m in out["issues"]["f32"]))

    def test_source_identity_must_match_q1a(self):
        w = self.world()
        path, psha, _ = self.prior(w)
        f = self.f32world(w, source={"commit": "different"})
        out = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, psha)
        self.assertEqual(out["verdict"], "INCOMPLETE")

    def test_f32_phases_must_really_be_f32(self):
        w = self.world()
        path, psha, _ = self.prior(w)
        f = self.f32world(w)
        edit_json(os.path.join(f["beta"], "phase-summary.json"), lambda d: d["environment"].update(HIPFIRE_STATE_QUANT="q8"))
        out = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, psha)
        self.assertEqual(out["verdict"], "INCOMPLETE")

    def test_score_f32_must_be_the_f32_pair_only(self):
        w = self.world()
        path, psha, _ = self.prior(w)
        f = self.f32world(w)
        edit_json(os.path.join(f["score"], "kl-summary.json"), lambda d: d.update(scope="full"))
        out = J.judge_f32(w["plan_path"], w["plan_sha"], f["beta"], f["s1"], f["score"], path, psha)
        self.assertEqual(out["verdict"], "INCOMPLETE")


class Cli(Base):
    def run_main(self, argv):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = J.main(argv)
        return rc, buf.getvalue(), err.getvalue()

    def argv(self, w, out, s1b=True):
        a = ["q1a", "--plan", w["plan_path"], "--expected-plan-sha256", w["plan_sha"], "--baseline-dir", w["baseline"], "--s1a-dir", w["s1a"],
             "--score-dir", w["score"], "--out", out]
        return a + (["--s1b-dir", w["s1b"]] if s1b else [])

    def test_exit_codes_and_never_overwrite(self):
        w = self.world()
        out = os.path.join(self.tmp, "o1.json")
        rc, stdout, _ = self.run_main(self.argv(w, out))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout)["verdict"], "PASS")
        self.assertEqual(json.loads(pathlib.Path(out).read_text())["default_on_recommended"], False)
        rc2, _, err = self.run_main(self.argv(w, out))
        self.assertEqual(rc2, 2)
        rc3, _, _ = self.run_main(self.argv(w, os.path.join(self.tmp, "o2.json"), s1b=False))
        self.assertEqual(rc3, 3)

    def test_fail_exit_code_and_plan_sha_refusal(self):
        w = self.world(knobs={"kl_const": {"primary-ctx4k": 0.5}})
        rc, _, _ = self.run_main(self.argv(w, os.path.join(self.tmp, "f.json")))
        self.assertEqual(rc, 1)
        bad = dict(w, plan_sha="0" * 64)
        rc2, _, err = self.run_main(self.argv(bad, os.path.join(self.tmp, "g.json")))
        self.assertEqual(rc2, 2)
        self.assertIn("refused", err)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "g.json")))

    def test_relative_paths_are_refused(self):
        w = self.world()
        rc, _, _ = self.run_main(self.argv(w, "relative.json"))
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
