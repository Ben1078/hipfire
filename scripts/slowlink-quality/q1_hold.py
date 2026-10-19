#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q1_hold.py -- Q1 production-route quality hold controller (ONE held command per part).  Python stdlib only.

    q1_hold.py preflight|run --part a|f32 --manifest M --expected-manifest-sha256 SHA
               --expected-controller-sha256 SHA --expected-judge-sha256 SHA --h2-helper H2_OFFICIAL.py --expected-h2-sha256 SHA
               --expected-binary-sha256 SHA --expected-plan-sha256 SHA --expected-native-source-sha256 SHA
               [--frontend-scheduler-id ID] [--report FILE]  (preflight)  |  --out FRESH_DIR  (run)

STATUS.  SOURCE-ONLY: never executed against the real native binary, a GPU or a model by its author.  The CPU tests drive a FAKE native process
(a Python script that writes tiny synthetic phase dumps); nothing here is a production backend for the tests.

WHAT ONE `run` IS.  One admitted fabric hold = the whole PART, each GPU phase in its OWN fresh native process (exactly one model load per process,
same binary / plan / artifact / max_seq):
    part a   : baseline -> s1a -> native CPU `score` (q8 pairs only) -> resume.json IMMEDIATELY (stage 'resume') -> fresh s1b ->
               Python determinism check + final judge (q1a-summary.json, the sealed Q1a summary).  The q8 KL is computed ONCE, by the native
               `score`, before s1b; nothing recomputes it afterwards.
    part f32 : beta-f32 -> s1-f32 -> native CPU `score-f32` (the f32 resume pair only, no q8 directory) -> contrast bound to the sealed Q1a
               summary sha (q1f32-summary.json).  No q8 model process is ever re-run.
The native `check-plan` runs first (CPU).  Nothing is submitted, retried or defaulted: a dropped phase closes the hold INCOMPLETE (OOM/headroom is
FAIL; a bus drop is INCOMPLETE), with no bypass and no fabricated receipt.

EXTERNAL BINDING.  The sha256 of the manifest bytes, of this controller, of q1_judge.py, of the frozen H2 helper, of the native binary, of the PLAN
and of the native example source are ALL supplied on the command line (independently of the manifest) and must equal the files AND the manifest's
own claims.  q1_judge.py and h2_official.py are imported only after their sha256 matched.  H2 helpers reused (generic, unchanged): strict JSON,
atomic files, cached-input stat gate, device identity / idle / drain, watcher identity + scan, own-process-group spawn / terminate / reap.

THE WATCHER IS NOT OURS.  The outer fabric wrapper (R7 run-with-watcher.sh) starts the kernel watcher in its own session BEFORE any lock and kills its
recorded group on exit.  This controller never starts a watcher or sampler: it only READS new bytes of manifest.watcher.log_path (and the target
card's sysfs vendor/device/unique_id) about every 250 ms while a child runs.  A fatal line, a lost device or an unreadable log aborts ONLY the
recorded own child group (SIGTERM, grace, SIGKILL, reap), closes that phase and stops the hold: no retry.

ISOLATION.  Each process gets a private HOME / HIPFIRE_HOME / lock dir / kernel hot cache under OUT/runs/<phase>/; the installed kernel pack is
read-only and re-checked (names/modes/hashes); HIPFIRE_HOME/models is ONE symlink to the canonical models dir; only small seed files are copied;
the child cwd is that private run dir (never a source tree); every HIPFIRE_/HIP_/ROCR_/ROC_/HSA_/GPU_/AMD_ variable of the parent is dropped.

MANIFEST hipfire.q1_hold.manifest.v1 -- see Q1-HARNESS-CONTRACT.json (holds/).  EXIT CODES: 0 PASS|REPORTED, 1 FAIL, 2 refused before any mutation,
3 INCOMPLETE, 4 aborted by signal."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import stat as statmod
import sys
import tempfile
import threading
import time

SCHEMA_MANIFEST = "hipfire.q1_hold.manifest.v1"
SCHEMA_PREFLIGHT = "hipfire.q1_hold.preflight.v1"
SCHEMA_PHASE = "hipfire.q1_hold.phase_receipt.v1"
SCHEMA_STEP = "hipfire.q1_hold.step_receipt.v1"
SCHEMA_RESULT = "hipfire.q1_hold.hold_result.v1"
SCHEMA_NATIVE_MANIFEST = "hipfire.qwen4.slowlink_quality.manifest.v1"
CONTROLLER_VERSION = "q1_hold.v1"

PART_PHASES = {"a": ("baseline", "s1a", "s1b"), "f32": ("beta-f32", "s1-f32")}
PART_MAX_HOLD = {"a": 1800, "f32": 900}
GPU_PHASES = ("baseline", "s1a", "s1b", "beta-f32", "s1-f32")
ROUTE = {"baseline": "baseline", "beta-f32": "baseline", "s1a": "s1", "s1b": "s1", "s1-f32": "s1"}
STATE_F32 = {"baseline": False, "s1a": False, "s1b": False, "beta-f32": True, "s1-f32": True}
MIN_MAX_SEQ = 16897

PINNED_ENV = {
    "HIPFIRE_STATE_QUANT": "q8",
    "HIPFIRE_KV_MODE": "q8",
    "HIPFIRE_GRAPH": "0",
    "HIPFIRE_NORMALIZE_PROMPT": "0",
    "HIPFIRE_CHAT_CURRENT_DATE": "2026-10-17",
    "HIPFIRE_OFFLOAD_EXEC": "pcie",
    "HIPFIRE_PLE_FORMAT": "q4",
    "HIPFIRE_QWEN4_D2_DEFAULT": "0",
}
BASELINE_EXTRA = {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "9", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096"}
S1_MUST_UNSET = ["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"]
SELECTOR_ENV = "HIPFIRE_QWEN4_SLOWLINK_PREFILL"
SELECTOR = {"baseline": "0", "s1": "1"}
STATE_QUANT_ENV = "HIPFIRE_STATE_QUANT"
CLASS_ENV = {"pinned": PINNED_ENV, "baseline_extra": BASELINE_EXTRA, "s1_must_be_unset": S1_MUST_UNSET, "selector": SELECTOR}

ENV_STRIP_PREFIXES = ("HIPFIRE_", "HIP_", "ROCR_", "ROC_", "HSA_", "GPU_", "AMD_")
ENV_KEEP_EXACT = ("HIPFIRE_FABRIC_ADMISSION",)
MAX_SEED_BYTES = 1 << 20
MAX_ARTIFACT_BYTES = 4 << 30
MAX_STDERR_SCAN = 8 << 20
HOLD_CLAMP_SECONDS = 1800

STATUS_OK = "OK"
VERDICT_EXIT = {"PASS": 0, "REPORTED": 0, "FAIL": 1, "INCOMPLETE": 3, "ABORTED": 4}
BUS_STATUSES = ("DEVICE_LOST", "WATCHER_FATAL", "WATCHER_UNREADABLE")

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
HEXID = re.compile(r"^0x[0-9a-f]{4}$")

HERE = os.path.dirname(os.path.realpath(__file__))
H2 = None           # the frozen h2_official helper module, bound by bind_helpers() after its sha256 matched
JUDGE = None        # q1_judge.py of this directory, bound by bind_helpers() after its sha256 matched
STOP = threading.Event()


class Refusal(Exception):
    """Refused BEFORE any mutation (no OUT dir, no process, no GPU)."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


# ----------------------------------------------------------------------------------------------------------------------------------
# host abstraction (clocks / sleep / sysfs / env; CPU tests inject a fake)
# ----------------------------------------------------------------------------------------------------------------------------------
class Host:
    def __init__(self, environ=None):
        self._environ = dict(os.environ if environ is None else environ)

    def environ(self):
        return dict(self._environ)

    def monotonic_ns(self):
        return time.monotonic_ns()

    def sleep(self, seconds):
        time.sleep(seconds)

    def boot_id(self):
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()

    def read_text(self, path):
        with open(path) as f:
            return f.read()

    def stop_requested(self):
        return STOP.is_set()


# ----------------------------------------------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------------------------------------------
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


def _is_int(v):
    return type(v) is int


def _is_num(v):
    return type(v) in (int, float) and v == v and v not in (float("inf"), float("-inf"))


def _is_str(v):
    return isinstance(v, str) and v != ""


def _is_abs(v):
    return _is_str(v) and os.path.isabs(v) and os.path.normpath(v) == v and "\x00" not in v


def _is_hex64(v):
    return isinstance(v, str) and HEX64.match(v) is not None


def _keys(obj, where, required, optional, errs):
    if not isinstance(obj, dict):
        errs.append(f"{where}: must be an object")
        return False
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj if k not in required and k not in optional]
    if missing:
        errs.append(f"{where}: missing keys {missing}")
    if extra:
        errs.append(f"{where}: unknown keys {extra}")
    return not missing


def nested(a, b):
    """True when a == b or either path lies inside the other."""
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(doc):
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def load_module(name, path, expected_sha256):
    """Import `path` as module `name` ONLY after its raw bytes hashed to the independently supplied sha256."""
    if not _is_abs(path):
        raise Refusal([f"{name}: path {path!r} must be a normalized absolute path"])
    try:
        got = sha256_file(path)
    except OSError as e:
        raise Refusal([f"{name}: {path} unreadable: {e}"])
    if got != expected_sha256:
        raise Refusal([f"{name}: {path} sha256 {got} != the supplied {expected_sha256}"])
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bind_helpers(h2_path, expected_h2_sha256, judge_path, expected_judge_sha256):
    global H2, JUDGE
    H2 = load_module("h2_official", h2_path, expected_h2_sha256)
    JUDGE = load_module("q1_judge", judge_path, expected_judge_sha256)


# ----------------------------------------------------------------------------------------------------------------------------------
# manifest (pure validation; no filesystem)
# ----------------------------------------------------------------------------------------------------------------------------------
def validate_manifest(m, part=None):
    errs = []
    required = ("schema", "label", "part", "identity", "harness", "binary", "plan", "native_source", "native_manifest_template", "target",
                "cached_inputs", "artifacts", "pack_dirs", "class_env", "budget", "watcher", "oom_stderr_regex")
    if not _keys(m, "manifest", required, ("home_seed_files", "prior_q1a"), errs):
        return errs
    if m["schema"] != SCHEMA_MANIFEST:
        errs.append(f"schema must be {SCHEMA_MANIFEST}")
    if not _is_str(m["label"]):
        errs.append("label must be a non-empty string")
    if m["part"] not in PART_PHASES:
        errs.append(f"part must be one of {sorted(PART_PHASES)}")
        return errs
    if part is not None and m["part"] != part:
        errs.append(f"manifest.part {m['part']!r} != --part {part!r}")
    pt = m["part"]
    ident = m["identity"]
    if _keys(ident, "identity", ("runtime",), (), errs) and _keys(ident["runtime"], "identity.runtime", ("commit", "tree"), (), errs):
        for k in ("commit", "tree"):
            if not (isinstance(ident["runtime"][k], str) and HEX40.match(ident["runtime"][k])):
                errs.append(f"identity.runtime.{k} must be 40 lowercase hex")
    if _keys(m["harness"], "harness", ("q1_hold_sha256", "q1_judge_sha256", "h2_official_sha256"), (), errs):
        for k, v in m["harness"].items():
            if not _is_hex64(v):
                errs.append(f"harness.{k} must be 64 lowercase hex")
    arts = {}
    for key in ("binary", "plan", "native_source"):
        e = m[key]
        if _keys(e, key, ("path", "size_bytes", "sha256"), (), errs):
            if not _is_abs(e["path"]):
                errs.append(f"{key}.path must be a normalized absolute path")
            if not (_is_int(e["size_bytes"]) and e["size_bytes"] > 0):
                errs.append(f"{key}.size_bytes must be a positive integer")
            if not _is_hex64(e["sha256"]):
                errs.append(f"{key}.sha256 must be 64 lowercase hex")
            arts[key] = e
    t = m["target"]
    if _keys(t, "target", ("card", "uuid", "bdf", "vendor_id", "device_id", "model_path", "model_size", "models_dir", "vram_used_path",
                           "vram_total_path", "visible_devices_env", "absent_paths"), (), errs):
        for k in ("card", "uuid"):
            if not _is_str(t[k]):
                errs.append(f"target.{k} must be a non-empty string")
        if not (isinstance(t["bdf"], str) and BDF.match(t["bdf"])):
            errs.append("target.bdf must look like 0000:03:00.0 (lowercase)")
        for k in ("vendor_id", "device_id"):
            if not (isinstance(t[k], str) and HEXID.match(t[k])) or t[k] == "0xffff":
                errs.append(f"target.{k} must be a lowercase 0x%04x id and not 0xffff")
        for k in ("model_path", "models_dir", "vram_used_path", "vram_total_path"):
            if not _is_abs(t[k]):
                errs.append(f"target.{k} must be a normalized absolute path")
        if not (_is_int(t["model_size"]) and t["model_size"] > 0):
            errs.append("target.model_size must be a positive integer")
        if _is_abs(t["model_path"]) and _is_abs(t["models_dir"]) and os.path.dirname(t["model_path"]) != t["models_dir"]:
            errs.append("target.model_path must be a direct child of target.models_dir")
        if _is_abs(t["vram_used_path"]) and _is_abs(t["vram_total_path"]):
            if os.path.basename(t["vram_used_path"]) != "mem_info_vram_used" or os.path.basename(t["vram_total_path"]) != "mem_info_vram_total":
                errs.append("target.vram_*_path must be mem_info_vram_used / mem_info_vram_total")
            if os.path.dirname(t["vram_used_path"]) != os.path.dirname(t["vram_total_path"]):
                errs.append("target.vram_used_path and vram_total_path must share one device dir")
            elif isinstance(t["bdf"], str) and os.path.basename(os.path.dirname(t["vram_used_path"])) != t["bdf"]:
                errs.append("the vram sysfs dir must be named after target.bdf")
        vis = t["visible_devices_env"]
        if _keys(vis, "target.visible_devices_env", ("HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"), (), errs):
            for k, v in vis.items():
                if not _is_str(v):
                    errs.append(f"target.visible_devices_env.{k} must be a non-empty string")
        ap = t["absent_paths"]
        if not (isinstance(ap, list) and ap and all(_is_abs(p) for p in ap) and len(set(ap)) == len(ap)):
            errs.append("target.absent_paths must be a non-empty list of unique absolute paths")
    ci = m["cached_inputs"]
    cached = {}
    if not isinstance(ci, list) or not ci:
        errs.append("cached_inputs must be a non-empty list")
    else:
        for i, e in enumerate(ci):
            if _keys(e, f"cached_inputs[{i}]", ("path", "size_bytes", "device", "inode", "mtime_ns", "ctime_ns", "sha256", "cached_receipt_path"), (), errs):
                if not _is_abs(e["path"]) or not _is_abs(e["cached_receipt_path"]):
                    errs.append(f"cached_inputs[{i}]: path and cached_receipt_path must be absolute")
                for k in ("size_bytes", "device", "inode", "mtime_ns", "ctime_ns"):
                    if not (_is_int(e[k]) and e[k] >= 0):
                        errs.append(f"cached_inputs[{i}].{k} must be a JSON INTEGER >= 0 (19-digit ns must stay exact)")
                if not _is_hex64(e["sha256"]):
                    errs.append(f"cached_inputs[{i}].sha256 must be 64 lowercase hex")
                cached[e["path"]] = e
        if len(cached) != len(ci):
            errs.append("cached_inputs paths must be unique")
        if isinstance(t, dict) and _is_abs(t.get("model_path")):
            for need in (t["model_path"], t["model_path"] + ".ple-q4"):
                if need not in cached:
                    errs.append(f"cached_inputs must bind {need}")
            me = cached.get(t["model_path"])
            if me is not None and _is_int(me.get("size_bytes")) and me["size_bytes"] != t.get("model_size"):
                errs.append("cached_inputs model size_bytes != target.model_size")
    art = m["artifacts"]
    art_by_path = {}
    if not isinstance(art, list) or not art:
        errs.append("artifacts must be a non-empty list")
    else:
        for i, e in enumerate(art):
            if _keys(e, f"artifacts[{i}]", ("path", "size_bytes", "sha256"), (), errs):
                if not _is_abs(e["path"]):
                    errs.append(f"artifacts[{i}].path must be absolute")
                if not (_is_int(e["size_bytes"]) and 0 < e["size_bytes"] <= MAX_ARTIFACT_BYTES):
                    errs.append(f"artifacts[{i}].size_bytes must be an integer in 1..{MAX_ARTIFACT_BYTES} (an owned file, never the model)")
                if not _is_hex64(e["sha256"]):
                    errs.append(f"artifacts[{i}].sha256 must be 64 lowercase hex")
                art_by_path[e["path"]] = e
        if len(art_by_path) != len(art):
            errs.append("artifacts paths must be unique")
        if isinstance(t, dict) and _is_abs(t.get("models_dir")):
            for p in art_by_path:
                if p in cached or p == t.get("model_path") or p.startswith(t["models_dir"] + os.sep):
                    errs.append(f"artifact {p} is a model-side file: payloads are never hashed")
    for key, e in arts.items():
        got = art_by_path.get(e["path"]) if isinstance(e.get("path"), str) else None
        if got is None:
            errs.append(f"{key}.path must be a declared artifact")
        elif got.get("sha256") != e.get("sha256") or got.get("size_bytes") != e.get("size_bytes"):
            errs.append(f"{key} size/sha256 differ from its artifact entry")
    packs = m["pack_dirs"]
    if not (isinstance(packs, list) and packs and all(_is_abs(p) for p in packs) and len(set(packs)) == len(packs)):
        errs.append("pack_dirs must be a non-empty list of unique absolute directories")
    elif len(art_by_path) == len(art if isinstance(art, list) else []):
        for p in packs:
            if not any(a.startswith(p + os.sep) for a in art_by_path):
                errs.append(f"pack_dirs {p} has no declared artifact files")
    if m["class_env"] != CLASS_ENV:
        errs.append(f"class_env must equal the frozen class {json.dumps(CLASS_ENV, sort_keys=True)}")
    b = m["budget"]
    keys = ("phase_cap_seconds", "kill_grace_seconds", "drain_seconds", "cpu_cap_seconds", "finalize_seconds", "preflight_seconds",
            "min_phase_seconds", "maximum_hold_seconds")
    if _keys(b, "budget", keys, (), errs):
        caps = b["phase_cap_seconds"]
        if not isinstance(caps, dict) or sorted(caps) != sorted(PART_PHASES[pt]):
            errs.append(f"budget.phase_cap_seconds must have exactly the keys {list(PART_PHASES[pt])}")
        elif not all(_is_num(v) and v > 0 for v in caps.values()):
            errs.append("budget.phase_cap_seconds values must be positive numbers")
        for k in keys[1:]:
            if not (_is_num(b[k]) and b[k] > 0):
                errs.append(f"budget.{k} must be a positive number")
        if all(_is_num(b[k]) for k in keys[1:]):
            if b["maximum_hold_seconds"] > PART_MAX_HOLD[pt]:
                errs.append(f"budget.maximum_hold_seconds exceeds {PART_MAX_HOLD[pt]} (part {pt}); the {HOLD_CLAMP_SECONDS} s fabric clamp must never squeeze a hold")
            if isinstance(caps, dict) and caps and all(_is_num(v) for v in caps.values()) and b["min_phase_seconds"] > min(caps.values()):
                errs.append("budget.min_phase_seconds exceeds a phase cap")
            if b["maximum_hold_seconds"] < b["preflight_seconds"] + b["min_phase_seconds"] + b["kill_grace_seconds"] + b["drain_seconds"] + b["finalize_seconds"]:
                errs.append("budget.maximum_hold_seconds cannot hold even one minimal phase")
    w = m["watcher"]
    if _keys(w, "watcher", ("log_path", "fatal_regex"), (), errs):
        if not _is_abs(w["log_path"]):
            errs.append("watcher.log_path must be absolute")
        for key in ("fatal_regex",):
            if not _is_str(w[key]):
                errs.append(f"watcher.{key} must be a non-empty string")
            else:
                try:
                    re.compile(w[key])
                except re.error as e:
                    errs.append(f"watcher.{key} does not compile: {e}")
    if not _is_str(m["oom_stderr_regex"]):
        errs.append("oom_stderr_regex must be a non-empty string")
    else:
        try:
            re.compile(m["oom_stderr_regex"])
        except re.error as e:
            errs.append(f"oom_stderr_regex does not compile: {e}")
    tpl = m["native_manifest_template"]
    if _keys(tpl, "native_manifest_template", ("binary", "source", "model"), (), errs):
        if isinstance(tpl["binary"], dict) and arts.get("binary") and tpl["binary"].get("sha256") != arts["binary"].get("sha256"):
            errs.append("native_manifest_template.binary.sha256 != binary.sha256")
        elif not isinstance(tpl["binary"], dict):
            errs.append("native_manifest_template.binary must be an object")
        src = tpl["source"]
        if not (isinstance(src, dict) and src):
            errs.append("native_manifest_template.source must be a non-empty object")
        else:
            if arts.get("native_source") and src.get("example_sha256") != arts["native_source"].get("sha256"):
                errs.append("native_manifest_template.source.example_sha256 != native_source.sha256")
            if isinstance(ident, dict) and isinstance(ident.get("runtime"), dict) and src.get("runtime_commit") != ident["runtime"].get("commit"):
                errs.append("native_manifest_template.source.runtime_commit != identity.runtime.commit")
        if _keys(tpl["model"], "native_manifest_template.model", ("path", "size", "sha256"), (), errs) and isinstance(t, dict):
            me = cached.get(t.get("model_path"))
            if tpl["model"]["path"] != t.get("model_path") or tpl["model"]["size"] != t.get("model_size") or (me is not None and tpl["model"]["sha256"] != me.get("sha256")):
                errs.append("native_manifest_template.model must equal target.model_path / model_size / the cached model sha256")
    if pt == "f32":
        if "prior_q1a" not in m:
            errs.append("part f32 REQUIRES prior_q1a (the sealed Q1a summary)")
        elif _keys(m["prior_q1a"], "prior_q1a", ("summary_path", "summary_sha256"), (), errs):
            if not _is_abs(m["prior_q1a"]["summary_path"]):
                errs.append("prior_q1a.summary_path must be absolute")
            if not _is_hex64(m["prior_q1a"]["summary_sha256"]):
                errs.append("prior_q1a.summary_sha256 must be 64 lowercase hex")
    elif "prior_q1a" in m:
        errs.append("part a must not carry prior_q1a")
    if "home_seed_files" in m:
        seeds = m["home_seed_files"]
        if not isinstance(seeds, list):
            errs.append("home_seed_files must be a list")
        else:
            dests = []
            for i, e in enumerate(seeds):
                if not _keys(e, f"home_seed_files[{i}]", ("source", "dest", "size_bytes", "sha256"), (), errs):
                    continue
                if not _is_abs(e["source"]):
                    errs.append(f"home_seed_files[{i}].source must be absolute")
                d = e["dest"]
                if not (_is_str(d) and not os.path.isabs(d) and os.path.normpath(d) == d and not d.startswith("..")):
                    errs.append(f"home_seed_files[{i}].dest must be a normalized relative path inside HIPFIRE_HOME")
                elif d.split(os.sep)[0] == "models":
                    errs.append(f"home_seed_files[{i}].dest {d!r} collides with the harness-owned models symlink")
                dests.append(d)
                if not (_is_int(e["size_bytes"]) and 0 < e["size_bytes"] <= MAX_SEED_BYTES):
                    errs.append(f"home_seed_files[{i}].size_bytes must be an integer in 1..{MAX_SEED_BYTES}")
                if not _is_hex64(e["sha256"]):
                    errs.append(f"home_seed_files[{i}].sha256 must be 64 lowercase hex")
            if len(set(dests)) != len(dests):
                errs.append("home_seed_files dests must be unique")
    return errs


def load_manifest(path, expected_sha256, part=None):
    """(manifest, raw sha256). The raw bytes must hash to the independently supplied sha256 BEFORE they are parsed."""
    if not _is_hex64(expected_sha256):
        raise Refusal(["--expected-manifest-sha256 is missing or not 64 lowercase hex"])
    try:
        with open(path, "rb") as f:
            raw = f.read((16 << 20) + 1)
    except OSError as e:
        raise Refusal([f"manifest {path} unreadable: {e}"])
    if len(raw) > 16 << 20:
        raise Refusal([f"manifest {path} is larger than 16 MiB"])
    got = sha256_bytes(raw)
    if got != expected_sha256:
        raise Refusal([f"manifest raw-bytes sha256 {got} != the independently supplied {expected_sha256}"])
    try:
        m = H2.loads_strict(raw)
    except ValueError as e:
        raise Refusal([f"manifest {path} is not strict JSON: {e}"])
    errs = validate_manifest(m, part)
    if errs:
        raise Refusal(["manifest invalid: " + " | ".join(errs)])
    return m, got


# ----------------------------------------------------------------------------------------------------------------------------------
# filesystem admission (hashes ONLY of declared owned files; NEVER the model)
# ----------------------------------------------------------------------------------------------------------------------------------
def _lstat_regular(path, where, errs):
    try:
        st = os.lstat(path)
    except OSError as e:
        errs.append(f"{where}: {path} unreadable: {e}")
        return None
    if not statmod.S_ISREG(st.st_mode):
        errs.append(f"{where}: {path} is not a regular file (symlinks/dirs are refused)")
        return None
    return st


def walk_pack(pack_dir, errs):
    files = []
    if not os.path.isdir(pack_dir) or os.path.islink(pack_dir):
        errs.append(f"pack_dir {pack_dir} is not a real directory")
        return files
    for root, dirs, names in os.walk(pack_dir, followlinks=False):
        for n in sorted(dirs):
            if os.path.islink(os.path.join(root, n)):
                errs.append(f"pack_dir contains a symlinked directory {os.path.join(root, n)}")
        for n in sorted(names):
            p = os.path.join(root, n)
            if not statmod.S_ISREG(os.lstat(p).st_mode):
                errs.append(f"pack_dir contains a non-regular entry {p}")
            else:
                files.append(p)
    if not files:
        errs.append(f"pack_dir {pack_dir} has no files")
    return files


def check_artifacts(m, errs):
    """Strictly re-hash exactly the declared owned artifacts. Returns [{path,size_bytes,sha256,..stat}]."""
    out = []
    declared = {e["path"] for e in m["artifacts"]}
    for pack in m["pack_dirs"]:
        on_disk = set(walk_pack(pack, errs))
        in_manifest = {p for p in declared if p.startswith(pack + os.sep)}
        for p in sorted(on_disk - in_manifest):
            errs.append(f"undeclared file in pack_dir: {p}")
        for p in sorted(in_manifest - on_disk):
            errs.append(f"declared pack file missing: {p}")
    for e in m["artifacts"]:
        where = f"artifact {e['path']}"
        st = _lstat_regular(e["path"], where, errs)
        if st is None:
            continue
        if st.st_size != e["size_bytes"]:
            errs.append(f"{where}: size {st.st_size} != manifest {e['size_bytes']}")
            continue
        try:
            got = sha256_file(e["path"])
        except OSError as ex:
            errs.append(f"{where}: unreadable: {ex}")
            continue
        if got != e["sha256"]:
            errs.append(f"{where}: sha256 {got} != manifest {e['sha256']}")
        out.append({"path": e["path"], "size_bytes": st.st_size, "sha256": got,
                    **{k: v for k, v in H2.stat_tuple(st).items() if k != "size_bytes"}})
    return out


def check_packs_readonly(m, errs):
    """Every pack dir, every directory below and every file in them is read-only (no write bit; no effective write access when not root)."""
    out = {}
    check_access = os.geteuid() != 0
    for pack in m["pack_dirs"]:
        n0 = len(errs)
        paths = [pack]
        for root, dirs, names in os.walk(pack, followlinks=False):
            paths.extend(os.path.join(root, n) for n in sorted(dirs))
            paths.extend(os.path.join(root, n) for n in sorted(names))
        n_dirs = n_files = 0
        for p in paths:
            try:
                st = os.lstat(p)
            except OSError as e:
                errs.append(f"pack path {p} unreadable: {e}")
                continue
            if statmod.S_ISLNK(st.st_mode):
                continue
            if statmod.S_ISDIR(st.st_mode):
                n_dirs += 1
            else:
                n_files += 1
            if st.st_mode & 0o222:
                errs.append(f"pack path {p} is writable by mode ({statmod.filemode(st.st_mode)}): the installed pack must be chmod a-w")
            elif check_access and os.access(p, os.W_OK):
                errs.append(f"pack path {p} has effective write access: the installed pack must be read-only")
        out[pack] = {"verified": len(errs) == n0, "dirs": n_dirs, "files": n_files, "effective_access_checked": check_access}
    return out


def check_seed_files(m, errs):
    out = []
    for e in m.get("home_seed_files", []):
        where = f"home_seed_files {e['source']}"
        st = _lstat_regular(e["source"], where, errs)
        if st is None:
            continue
        if st.st_size != e["size_bytes"]:
            errs.append(f"{where}: size {st.st_size} != {e['size_bytes']}")
            continue
        got = sha256_file(e["source"])
        if got != e["sha256"]:
            errs.append(f"{where}: sha256 mismatch")
        out.append({"source": e["source"], "dest": e["dest"], "sha256": got, **H2.stat_tuple(st)})
    return out


def snapshot(m):
    """Stat-only picture of every bound input (cached, owned, seeds, packs, absent paths)."""
    snap = {"cached": {}, "artifacts": {}, "absent": {}, "seeds": {}}
    for e in m["cached_inputs"]:
        try:
            snap["cached"][e["path"]] = H2.stat_tuple(os.stat(e["path"]))
        except OSError as ex:
            snap["cached"][e["path"]] = f"unreadable: {ex}"
    for e in m["artifacts"]:
        try:
            snap["artifacts"][e["path"]] = H2.stat_tuple(os.lstat(e["path"]))
        except OSError as ex:
            snap["artifacts"][e["path"]] = f"unreadable: {ex}"
    for e in m.get("home_seed_files", []):
        try:
            snap["seeds"][e["source"]] = H2.stat_tuple(os.lstat(e["source"]))
        except OSError as ex:
            snap["seeds"][e["source"]] = f"unreadable: {ex}"
    for p in m["target"]["absent_paths"]:
        snap["absent"][p] = os.path.lexists(p)
    snap["packs"] = {p: H2.pack_snapshot(p) for p in m["pack_dirs"]}
    return snap


def snapshot_diff(snap0, snap1):
    """Sections that changed. The cached model/PLE tuple is gated on size/device/inode/mtime_ns ONLY (ctime is metadata: a hardlink
    create/delete on the same inode moves ctime without touching content); owned artifacts / packs / seeds are strict."""
    changed = []
    for sect in ("cached", "artifacts", "seeds", "absent", "packs"):
        if sect == "cached":
            diff = {p: H2.cached_gate(v) for p, v in snap1[sect].items()} != {p: H2.cached_gate(v) for p, v in snap0[sect].items()}
        else:
            diff = snap1[sect] != snap0[sect]
        if diff:
            changed.append(sect)
    return changed


def inputs_strict_changed(m, snap0):
    """Post-phase integrity: stat snapshot AND a fresh strict re-hash of every owned artifact / pack file."""
    snap1 = snapshot(m)
    changed = snapshot_diff(snap0, snap1)
    rehash_errs = []
    check_artifacts(m, rehash_errs)
    if rehash_errs and "artifacts" not in changed:
        changed.append("artifacts")
    return changed, snap1, rehash_errs


def plan_binding_errors(m, plan, plan_sha):
    """Cross-checks of the PLAN against the manifest and the frozen class (the native check-plan validates the rest)."""
    errs = []
    t, art = m["target"], plan["artifact"]
    if plan_sha != m["plan"]["sha256"]:
        errs.append("plan sha256 != manifest.plan.sha256")
    if art["path"] != t["model_path"] or art["size"] != t["model_size"]:
        errs.append("plan.artifact path/size != target.model_path/model_size")
    me = next((e for e in m["cached_inputs"] if e["path"] == t["model_path"]), None)
    if me is None or art["sha256"] != me["sha256"]:
        errs.append("plan.artifact.sha256 != the cached model sha256")
    ex = plan["expect"]
    if ex["env"] != PINNED_ENV:
        errs.append("plan.expect.env must equal the 8 pinned class flags at their q8 values")
    if ex["baseline_env"] != BASELINE_EXTRA:
        errs.append("plan.expect.baseline_env must equal the two baseline extras")
    if not set(S1_MUST_UNSET) <= set(ex["env_absent"]):
        errs.append("plan.expect.env_absent must name the layer override")
    if ex["ple_format"] != PINNED_ENV["HIPFIRE_PLE_FORMAT"]:
        errs.append("plan.expect.ple_format != q4")
    if not (_is_int(plan.get("max_seq")) and plan["max_seq"] >= MIN_MAX_SEQ):
        errs.append(f"plan.max_seq must be an integer >= {MIN_MAX_SEQ}")
    return errs


def check_device(host, m):
    dev = H2.device_state(host, m)
    try:
        used = int(host.read_text(m["target"]["vram_used_path"]).strip())
        err = None
    except (OSError, ValueError) as e:
        used, err = None, str(e)
    return dev, used, err


def prior_q1a_errors(m, plan_sha):
    """part f32: the sealed Q1a summary must be the manifest's, a final stage, and carry this hold's plan/binary/source/model identity."""
    errs = []
    pq = m["prior_q1a"]
    try:
        with open(pq["summary_path"], "rb") as f:
            raw = f.read((64 << 20) + 1)
    except OSError as e:
        return [f"prior Q1a summary unreadable: {e}"], None
    if sha256_bytes(raw) != pq["summary_sha256"]:
        return ["prior Q1a summary sha256 != manifest.prior_q1a.summary_sha256"], None
    try:
        doc = H2.loads_strict(raw)
    except ValueError as e:
        return [f"prior Q1a summary is not strict JSON: {e}"], None
    if not isinstance(doc, dict) or doc.get("schema") != JUDGE.SCHEMA_Q1A or not JUDGE.seal_ok(doc):
        errs.append("prior Q1a summary is not a sealed q1a summary")
        return errs, doc
    if doc.get("stage") != "final":
        errs.append(f"prior Q1a summary stage is {doc.get('stage')!r}, not 'final'")
    ident = doc.get("identity") or {}
    tpl = m["native_manifest_template"]
    if ident.get("plan_sha256") != plan_sha or ident.get("binary_sha256") != m["binary"]["sha256"] or ident.get("source") != tpl["source"]:
        errs.append("prior Q1a plan/binary/source identity differs from this hold's")
    mod = ident.get("model") or {}
    if mod.get("path") != m["target"]["model_path"] or mod.get("pinned_sha256") != tpl["model"]["sha256"]:
        errs.append("prior Q1a model identity differs from this hold's")
    return errs, doc


def preflight_checks(args, m, msha, host, plan_out=None):
    """Admission checks (no mutation). Returns (report dict, errs). `plan_out` receives the loaded plan dict."""
    errs = []
    report = {"schema": SCHEMA_PREFLIGHT, "controller_version": CONTROLLER_VERSION, "ok": False, "errors": errs, "part": m["part"], "manifest_sha256": msha}
    own = sha256_file(os.path.realpath(__file__))
    report["controller_sha256"] = own
    for label, want in (("controller", args.expected_controller_sha256), ("manifest.harness.q1_hold_sha256", m["harness"]["q1_hold_sha256"])):
        if own != want:
            errs.append(f"this controller's sha256 {own} != {label} {want}")
    judge_sha = sha256_file(os.path.join(HERE, "q1_judge.py"))
    h2_sha = sha256_file(args.h2_helper)
    report["judge_sha256"], report["h2_helper_sha256"] = judge_sha, h2_sha
    if judge_sha != args.expected_judge_sha256 or judge_sha != m["harness"]["q1_judge_sha256"]:
        errs.append("q1_judge.py sha256 differs from --expected-judge-sha256 / manifest.harness.q1_judge_sha256")
    if h2_sha != args.expected_h2_sha256 or h2_sha != m["harness"]["h2_official_sha256"]:
        errs.append("h2 helper sha256 differs from --expected-h2-sha256 / manifest.harness.h2_official_sha256")
    for flag, got, label in (("--expected-binary-sha256", m["binary"]["sha256"], "manifest.binary.sha256"),
                             ("--expected-plan-sha256", m["plan"]["sha256"], "manifest.plan.sha256"),
                             ("--expected-native-source-sha256", m["native_source"]["sha256"], "manifest.native_source.sha256")):
        want = getattr(args, flag[2:].replace("-", "_"))
        if want != got:
            errs.append(f"{flag} {want} != {label} {got}")
    t = m["target"]
    if not os.path.isdir(t["models_dir"]):
        errs.append(f"models_dir {t['models_dir']} is not a directory")
    elif os.path.realpath(t["models_dir"]) != os.path.realpath(os.path.dirname(t["model_path"])):
        errs.append("models_dir does not resolve to the model's directory")
    for p in t["absent_paths"]:
        if os.path.lexists(p):
            errs.append(f"expected-absent path now exists: {p}")
    if not os.path.isfile(t["model_path"]):
        errs.append(f"model_path {t['model_path']} missing")
    report["cached_inputs"] = H2.check_cached_inputs(m, errs)
    report["artifacts"] = check_artifacts(m, errs)
    report["pack_readonly"] = check_packs_readonly(m, errs)
    report["home_seed_files"] = check_seed_files(m, errs)
    if not os.access(m["binary"]["path"], os.X_OK):
        errs.append("the native binary is not executable")
    report["device"], report["device_vram_used"], vram_err = check_device(host, m)
    if not report["device"]["ok"]:
        errs.append("target device check failed: " + "; ".join(report["device"]["problems"]))
    if vram_err:
        errs.append(f"vram_used unreadable: {vram_err}")
    try:
        ident = H2.watcher_identity(m)
        report["watcher_identity"] = ident
        if not statmod.S_ISREG(os.stat(m["watcher"]["log_path"]).st_mode):
            errs.append("watcher.log_path does not resolve to a regular file")
    except OSError as e:
        errs.append(f"watcher log unreadable: {e}")
    try:
        plan, plan_sha = JUDGE.load_plan(m["plan"]["path"], args.expected_plan_sha256)
        errs.extend(plan_binding_errors(m, plan, plan_sha))
        report["plan_sha256"] = plan_sha
        if plan_out is not None:
            plan_out["plan"], plan_out["sha"] = plan, plan_sha
        if m["part"] == "f32":
            perrs, _ = prior_q1a_errors(m, plan_sha)
            errs.extend(perrs)
    except JudgeErrorTypes() as e:
        errs.append(f"plan: {e}")
    b = m["budget"]
    report["budget"] = {"maximum_hold_seconds": b["maximum_hold_seconds"], "phase_cap_seconds": dict(b["phase_cap_seconds"]),
                        "sum_of_phase_caps": sum(b["phase_cap_seconds"].values()),
                        "caps_are_watchdogs_not_a_promise_they_all_fit": True}
    fenv = host.environ()
    report["fabric"] = {"job": fenv.get("FABRIC_JOB"), "cards": fenv.get("FABRIC_CARDS"), "note": "preflight is usable outside a fabric job and records null; only `run` REQUIRES the actual FABRIC_JOB/FABRIC_CARDS"}
    report["ok"] = not errs
    return report


def JudgeErrorTypes():
    return (JUDGE.JudgeError, OSError, ValueError, KeyError, TypeError)


# ----------------------------------------------------------------------------------------------------------------------------------
# per-process environment, private tree, native manifest
# ----------------------------------------------------------------------------------------------------------------------------------
def sanitized_env(parent_env):
    """(env, unset names): FABRIC_* and the fabric admission token are kept; every other HIPFIRE_/HIP_/ROCR_/ROC_/HSA_/GPU_/AMD_ var is dropped."""
    env, unset = {}, []
    for k, v in parent_env.items():
        if k in ENV_KEEP_EXACT or k.startswith("FABRIC_"):
            env[k] = v
        elif k.startswith(ENV_STRIP_PREFIXES):
            unset.append(k)
        else:
            env[k] = v
    return env, sorted(unset)


def gpu_env(parent_env, run_dir, m, phase):
    """(env, unset, applied). The whole launch class of ONE phase."""
    home = os.path.join(run_dir, "home")
    hh = os.path.join(home, ".hipfire")
    vis = m["target"]["visible_devices_env"]
    route = ROUTE[phase]
    applied = {
        "HOME": home, "HIPFIRE_HOME": hh,
        "HIPFIRE_LOCK_DIR": os.path.join(run_dir, "lockdir"), "HIPFIRE_KERNEL_CACHE": os.path.join(run_dir, "kcache"),
        "HIPFIRE_DEVICES": vis["HIPFIRE_DEVICES"], "ROCR_VISIBLE_DEVICES": vis["ROCR_VISIBLE_DEVICES"], "HIP_VISIBLE_DEVICES": vis["HIP_VISIBLE_DEVICES"],
        "HIPFIRE_NO_REGISTRY_FETCH": "1",
        SELECTOR_ENV: SELECTOR[route],
    }
    applied.update(PINNED_ENV)
    if STATE_F32[phase]:
        applied[STATE_QUANT_ENV] = "f32"
    if route == "baseline":
        applied.update(BASELINE_EXTRA)
    env, unset = sanitized_env(parent_env)
    env.update(applied)
    if route == "s1":
        for name in S1_MUST_UNSET:
            if name in env:
                raise RuntimeError(f"internal: {name} survived sanitising on an S1 route")
    return env, unset, dict(sorted(applied.items()))


def cpu_env(parent_env, run_dir):
    env, unset = sanitized_env(parent_env)
    env["HOME"] = os.path.join(run_dir, "home")
    return env, unset


def prepare_run_dir(ctx, run_dir, gpu):
    """Fresh private tree. The hot cache is the ONLY place a kernel JIT may write; HIPFIRE_HOME/models is one symlink to the canonical models dir."""
    m = ctx.m
    hh = os.path.join(run_dir, "home", ".hipfire")
    os.makedirs(hh)
    if not gpu:
        return {"home": os.path.join(run_dir, "home")}
    os.mkdir(os.path.join(run_dir, "lockdir"))
    hot = os.path.join(run_dir, "kcache")
    os.mkdir(hot)
    if not os.access(hot, os.W_OK):
        raise RuntimeError(f"private hot cache {hot} is not writable")
    models = os.path.realpath(os.path.dirname(m["target"]["model_path"]))
    os.symlink(models, os.path.join(hh, "models"))
    seeds = []
    for e in m.get("home_seed_files", []):
        with open(e["source"], "rb") as f:
            data = f.read(MAX_SEED_BYTES + 1)
        if sha256_bytes(data) != e["sha256"] or len(data) != e["size_bytes"]:
            raise RuntimeError(f"seed file {e['source']} changed since preflight")
        dest = os.path.join(hh, e["dest"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "xb") as f:
            f.write(data)
        seeds.append({"dest": e["dest"], "sha256": e["sha256"], "size_bytes": e["size_bytes"]})
    return {"hipfire_home": hh, "models_symlink": os.path.join(hh, "models"), "models_symlink_target": models, "seeded": seeds,
            "hot_cache": {"path": hot, "private": True, "writable": True}}


def model_stat_block(m):
    """Live stat of the model against the manifest tuple (ctime is NOT bound). Returns (native model block, errors)."""
    mp = m["target"]["model_path"]
    me = next(e for e in m["cached_inputs"] if e["path"] == mp)
    errs = []
    try:
        st = os.stat(mp)
    except OSError as e:
        return None, [f"model {mp} unreadable: {e}"]
    cur = H2.stat_tuple(st)
    for k in H2.CACHED_GATE_KEYS:
        if me[k] != cur[k]:
            errs.append(f"model {k} manifest {me[k]} != live {cur[k]}")
    sec, nsec = divmod(st.st_mtime_ns, 10 ** 9)
    return {"path": mp, "dev": st.st_dev, "ino": st.st_ino, "size": st.st_size, "mtime_sec": sec, "mtime_nsec": nsec,
            "sha256": me["sha256"]}, errs


def native_manifest(ctx, phase, argv, applied, unset):
    m = ctx.m
    tpl = m["native_manifest_template"]
    model, errs = model_stat_block(m)
    if errs:
        raise RuntimeError("; ".join(errs))
    return {
        "schema": SCHEMA_NATIVE_MANIFEST,
        "binary": dict(tpl["binary"]),
        "source": dict(tpl["source"]),
        "launch": {"controller": CONTROLLER_VERSION, "controller_sha256": ctx.controller_sha256, "hold_label": m["label"], "part": ctx.part,
                   "phase": phase, "route": ROUTE[phase], "command": argv, "command_text": " ".join(shlex.quote(a) for a in argv),
                   "env": {"set": applied, "unset": unset}, "fabric": ctx.fabric, "frontend_scheduler_id": ctx.frontend_id,
                   "device": {k: m["target"][k] for k in ("card", "uuid", "bdf", "vendor_id", "device_id")},
                   "locks": "external fabric lock; private HIPFIRE_LOCK_DIR", "run_dir": os.path.join(ctx.out, "runs", phase)},
        "model": model,
    }


def write_new_bytes(path, data):
    H2.write_atomic(path, data)


# ----------------------------------------------------------------------------------------------------------------------------------
# hold context
# ----------------------------------------------------------------------------------------------------------------------------------
class HoldCtx:
    def __init__(self, args, m, msha, plan, plan_sha, host, out, hold_start_ns, report_sha):
        self.args, self.m, self.msha, self.plan, self.plan_sha, self.host, self.out = args, m, msha, plan, plan_sha, host, out
        self.part = m["part"]
        self.hold_start_ns = hold_start_ns
        self.preflight_sha = report_sha
        env = host.environ()
        self.fabric = {"job": env.get("FABRIC_JOB"), "cards": env.get("FABRIC_CARDS"),
                       "env": {k: v for k, v in sorted(env.items()) if k.startswith("FABRIC_")},
                       "hipfire_fabric_admission": env.get("HIPFIRE_FABRIC_ADMISSION")}
        self.frontend_id = args.frontend_scheduler_id
        self.controller_sha256 = sha256_file(os.path.realpath(__file__))
        self.watcher_ident = H2.watcher_identity(m)
        self.watcher_offset = self.watcher_ident["start_size_bytes"]
        self.snap0 = snapshot(m)
        self.receipts = {}            # name -> sealed receipt
        self.files = {}               # name -> {"path", "sha256"}
        self.judge_docs = {}          # "resume"/"final"/"f32" -> judge document
        self.phase_dirs = {p: os.path.join(out, "phases", p) for p in GPU_PHASES}
        self.aborted = False
        self.halt = False             # a bus drop / signal: nothing further may start

    def remaining_s(self):
        return self.m["budget"]["maximum_hold_seconds"] - (self.host.monotonic_ns() - self.hold_start_ns) / 1e9

    def event(self, name, **kw):
        line = {"t_utc": utc_now(), "t_hold_s": (self.host.monotonic_ns() - self.hold_start_ns) / 1e9, "event": name, **kw}
        fd = os.open(os.path.join(self.out, "events.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (json.dumps(line, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def record_file(self, name, path):
        self.files[name] = {"path": path, "sha256": sha256_file(path)}
        return self.files[name]["sha256"]


def seal_write(ctx, name, rel, doc):
    """Seal `doc`, write it (never overwrite) at OUT/<rel>, remember it, return the sealed doc."""
    doc = H2.seal(doc)
    path = os.path.join(ctx.out, rel)
    H2.write_json_atomic(path, doc)
    ctx.receipts[name] = doc
    ctx.record_file(name, path)
    return doc


# ----------------------------------------------------------------------------------------------------------------------------------
# one managed child (GPU phase or CPU step)
# ----------------------------------------------------------------------------------------------------------------------------------
def effective_cap(ctx, phase_cap, tail_cpu):
    b = ctx.m["budget"]
    reserve = b["kill_grace_seconds"] + b["drain_seconds"] + b["finalize_seconds"] + (b["cpu_cap_seconds"] if tail_cpu else 0)
    avail = ctx.remaining_s() - reserve
    return min(phase_cap, avail), ("total" if avail < phase_cap else "phase"), avail


def start_gate(ctx, label, gpu, cap, bound_by):
    """Reasons the child cannot start now ([] = may start)."""
    why = []
    b = ctx.m["budget"]
    if ctx.halt:
        why.append("HALTED: an earlier bus drop / signal stopped the hold")
    if ctx.host.stop_requested():
        why.append("SIGNAL: stop requested")
    floor = b["min_phase_seconds"] if gpu else 1
    if cap < floor:
        why.append(f"BUDGET: effective cap {cap:.1f}s (bound by {bound_by}) is below the minimum {floor}s")
    dev, used, err = check_device(ctx.host, ctx.m)
    if not dev["ok"]:
        why.append("DEVICE: " + "; ".join(dev["problems"]))
    elif gpu:
        idle = H2.wait_idle(ctx.host, ctx.m, 0)
        if not idle["idle"]:
            why.append(f"BUSY: card vram_used {idle['last_used']} is not < {H2.IDLE_MAX} before spawn")
    return why


def run_child(ctx, label, argv, env, run_dir, cap_s):
    """Managed own-group child with the shared health check. Returns (run dict, watcher accumulator, health_aborts)."""
    m, host = ctx.m, ctx.host
    b = m["budget"]
    acc = {"fatal": False, "matches": [], "match_count": 0, "error": None}

    def absorb(w):
        acc["fatal"] = acc["fatal"] or w["fatal"]
        acc["match_count"] += w.get("match_count", 0)
        acc["matches"].extend(w["matches"][:max(0, H2.MAX_LOG_MATCHES - len(acc["matches"]))])
        if w["error"] and not acc["error"]:
            acc["error"] = w["error"]

    def health():
        causes = []
        dev = H2.device_state(host, m)
        if not dev["ok"]:
            causes.append({"status": "DEVICE_LOST", "problems": dev["problems"]})
        ctx.watcher_offset, w = H2.watcher_scan(m, ctx.watcher_offset, ctx.watcher_ident)
        absorb(w)
        if w["fatal"]:
            causes.append({"status": "WATCHER_FATAL", "matches": w["matches"], "match_count": w.get("match_count", 0)})
        if w["error"]:
            causes.append({"status": "WATCHER_UNREADABLE", "error": w["error"]})
        return causes

    run = H2.spawn_and_wait(argv, env, run_dir, os.path.join(run_dir, "stdout.log"), os.path.join(run_dir, "stderr.log"),
                            os.path.join(run_dir, "process.json"), cap_s, b["kill_grace_seconds"], host, health=health)
    return run, acc, absorb


def child_failures(run, acc, dev_after, drain):
    ha = run.get("health_abort")
    failures = []
    if not run["spawned"]:
        failures.append("SPAWN_ERROR")
    if ha:
        failures.append(ha["causes"][0]["status"])
    if run["aborted"]:
        failures.append("ABORTED_SIGNAL")
    if run["timed_out"]:
        failures.append("TIMEOUT")
    if run["spawned"] and not run["aborted"] and not run["timed_out"] and not ha and run["returncode"] != 0:
        failures.append("EXIT_NONZERO")
    if run["leftover_after_exit"]:
        failures.append("LIFECYCLE_LEFTOVER")
    if not dev_after["ok"] and "DEVICE_LOST" not in failures:
        failures.append("DEVICE_LOST")
    if acc["error"] and "WATCHER_UNREADABLE" not in failures:
        failures.append("WATCHER_UNREADABLE")
    if acc["fatal"] and "WATCHER_FATAL" not in failures:
        failures.append("WATCHER_FATAL")
    if drain is not None and not drain["idle"]:
        failures.append("DRAIN_TIMEOUT")
    return failures


def classify(failures, stderr_text, oom_rx):
    """BUS (always incomplete) / OOM (fail, no bypass) / OTHER (incomplete) / None."""
    if not failures:
        return None
    if any(f in BUS_STATUSES for f in failures):
        return "BUS"
    if oom_rx.search(stderr_text):
        return "OOM"
    return "OTHER"


def read_tail(path, limit=MAX_STDERR_SCAN):
    with open(path, "rb") as f:
        return f.read(limit).decode("utf-8", errors="replace")


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def process_block(run):
    proc = run.get("process", {})
    elapsed = None
    if "end_monotonic_ns" in run and "start_monotonic_ns" in proc:
        elapsed = (run["end_monotonic_ns"] - proc["start_monotonic_ns"]) / 1e9
    return {**proc, "returncode": run["returncode"], "signal": run["signal"], "timed_out": run["timed_out"], "aborted": run["aborted"],
            "end_monotonic_ns": run.get("end_monotonic_ns"), "elapsed_seconds": elapsed, "spawn_error": run.get("spawn_error"),
            "start_new_session": True, "close_fds": True}


def check_native_phase(ctx, phase, out_dir, manifest_sha):
    """A completed GPU phase leaves phase-summary.json LAST with complete:true. Returns (ok, info)."""
    info = {"summary_path": os.path.join(out_dir, "phase-summary.json")}
    try:
        with open(info["summary_path"], "rb") as f:
            raw = f.read(JUDGE.MAX_JSON_BYTES)
        doc = H2.loads_strict(raw)
    except (OSError, ValueError) as e:
        info["error"] = f"phase-summary.json unreadable: {e}"
        return False, info
    info["summary_sha256"] = sha256_bytes(raw)
    problems = []
    if not isinstance(doc, dict):
        problems.append("not an object")
    else:
        if doc.get("complete") is not True:
            problems.append("complete is not true")
        if doc.get("phase") != phase:
            problems.append(f"phase {doc.get('phase')!r} != {phase!r}")
        if (doc.get("plan") or {}).get("sha256") != ctx.plan_sha:
            problems.append("plan sha256 differs")
        if (doc.get("binary") or {}).get("sha256") != ctx.m["binary"]["sha256"]:
            problems.append("binary sha256 differs")
        if (doc.get("manifest") or {}).get("sha256") != manifest_sha:
            problems.append("manifest sha256 differs from the manifest this controller wrote")
    info["problems"] = problems
    info["complete"] = not problems
    return not problems, info


def run_gpu_phase(ctx, phase, tail_cpu=False):
    """ONE fresh native process = one model load. Always writes a sealed OUT/receipts/<phase>.json (even when the phase was not started)."""
    m, host, b = ctx.m, ctx.host, ctx.m["budget"]
    cap_cfg = b["phase_cap_seconds"][phase]
    cap, bound_by, avail = effective_cap(ctx, cap_cfg, tail_cpu)
    run_dir = os.path.join(ctx.out, "runs", phase)
    out_dir = ctx.phase_dirs[phase]
    head = {"schema": SCHEMA_PHASE, "part": ctx.part, "phase": phase, "route": ROUTE[phase], "state_f32": STATE_F32[phase],
            "manifest_sha256": ctx.msha, "preflight_sha256": ctx.preflight_sha, "controller_sha256": ctx.controller_sha256,
            "fabric": ctx.fabric, "frontend_scheduler_id": ctx.frontend_id,
            "caps": {"phase_cap_seconds": cap_cfg, "effective_cap_seconds": cap, "bound_by": bound_by, "available_after_reserve_seconds": avail,
                     "remaining_hold_seconds": ctx.remaining_s(), "kill_grace_seconds": b["kill_grace_seconds"], "drain_seconds": b["drain_seconds"],
                     "tail_cpu_reserved": tail_cpu},
            "native_out_dir": out_dir, "run_dir": run_dir}
    why = start_gate(ctx, phase, True, cap, bound_by)
    if not why:
        changed, _, _ = inputs_strict_changed(m, ctx.snap0)
        if changed:
            why.append(f"INPUT_CHANGED before start: {changed}")
    if why:
        ctx.event("phase_not_started", phase=phase, why=why)
        rec = seal_write(ctx, phase, f"receipts/{phase}.json", {**head, "started": False, "status": "NOT_STARTED", "failures": ["NOT_STARTED"], "why": why,
                                                                 "failure_class": "OTHER", "ok": False})
        if any(w.startswith("SIGNAL") for w in why):
            ctx.aborted = True
        ctx.halt = ctx.halt or any(w.startswith(("SIGNAL", "DEVICE")) for w in why)
        return rec
    ctx.event("phase_start", phase=phase, effective_cap_seconds=cap)
    try:
        prep = prepare_run_dir(ctx, run_dir, True)
        env, unset, applied = gpu_env(host.environ(), run_dir, m, phase)
        argv = [m["binary"]["path"], "--phase", phase, "--plan", m["plan"]["path"], "--manifest", os.path.join(run_dir, "native-manifest.json"),
                "--model", m["target"]["model_path"], "--out-dir", out_dir]
        nm = native_manifest(ctx, phase, argv, applied, unset)
        nm_bytes = json.dumps(nm, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        write_new_bytes(os.path.join(run_dir, "native-manifest.json"), nm_bytes)
        nm_sha = sha256_bytes(nm_bytes)
        with open(os.path.join(run_dir, "command.txt"), "x") as f:
            f.write(" ".join(shlex.quote(a) for a in argv) + "\n")
        write_new_bytes(os.path.join(run_dir, "env.json"), json.dumps({"set": applied, "unset": unset, "fabric": ctx.fabric}, indent=1, sort_keys=True).encode() + b"\n")
        baseline_idle = H2.wait_idle(host, m, 0)
        run, acc, absorb = run_child(ctx, phase, argv, env, run_dir, cap)
    except BaseException as e:  # noqa: BLE001 -- the phase must always close with a sealed receipt
        return seal_write(ctx, phase, f"receipts/{phase}.json", {**head, "started": False, "status": "HARNESS_EXCEPTION", "failures": ["HARNESS_EXCEPTION"],
                                                                 "exception": f"{type(e).__name__}: {e}", "failure_class": "OTHER", "ok": False})
    ha = run.get("health_abort")
    drain = H2.wait_idle(host, m, b["drain_seconds"]) if run["spawned"] else None
    dev_after = H2.device_state(host, m)
    ctx.watcher_offset, w_last = H2.watcher_scan(m, ctx.watcher_offset, ctx.watcher_ident)
    absorb(w_last)
    acc["identity"] = ctx.watcher_ident
    failures = child_failures(run, acc, dev_after, drain)
    changed, snap1, rehash_errs = inputs_strict_changed(m, ctx.snap0)
    if changed:
        failures.append("INPUT_CHANGED")
    stderr_text = read_tail(os.path.join(run_dir, "stderr.log"))
    native = None
    if run["spawned"] and run["returncode"] == 0 and not run["timed_out"] and not run["aborted"] and not ha:
        ok_native, native = check_native_phase(ctx, phase, out_dir, nm_sha)
        if not ok_native:
            failures.append("INVALID_OUTPUT")
    status = failures[0] if failures else STATUS_OK
    fclass = classify(failures, stderr_text, ctx.oom_rx)
    stdout_raw = read_bytes(os.path.join(run_dir, "stdout.log"))
    rec = {**head, "started": True, "argv": argv, "command_text": " ".join(shlex.quote(a) for a in argv),
           "env": {"set": applied, "unset": unset}, "home": prep, "native_manifest": {"path": os.path.join(run_dir, "native-manifest.json"), "sha256": nm_sha},
           "binary_sha256": m["binary"]["sha256"], "baseline_idle": {"idle": baseline_idle["idle"], "vram_used": baseline_idle["last_used"]},
           "process": process_block(run), "lifecycle": {"events": run["events"], "leftover_after_cli_exit": run["leftover_after_exit"],
                                                        "group_after_cleanup": run.get("group_after_cleanup", [])},
           "drain": drain, "device_after": dev_after, "watcher": acc, "health_abort": ha,
           "integrity": {"changed_sections": changed, "rehash_errors": rehash_errs[:20], "cached_inputs_after": snap1["cached"],
                         "cached_gate": "size/device/inode/mtime_ns + pinned sha; ctime informational only",
                         "owned_artifacts_and_packs_rehashed": True},
           "stdout": {"size_bytes": len(stdout_raw), "sha256": sha256_bytes(stdout_raw)},
           "stderr": {"size_bytes": os.path.getsize(os.path.join(run_dir, "stderr.log")), "sha256": sha256_file(os.path.join(run_dir, "stderr.log"))},
           "native_phase": native, "failures": failures, "status": status, "failure_class": fclass, "ok": status == STATUS_OK}
    if fclass == "BUS" or "ABORTED_SIGNAL" in failures:
        ctx.halt = True
    if "ABORTED_SIGNAL" in failures:
        ctx.aborted = True
    ctx.event("phase_end", phase=phase, status=status, failure_class=fclass)
    return seal_write(ctx, phase, f"receipts/{phase}.json", rec)


def run_cpu_step(ctx, name, argv, out_dir, expect_files):
    """A native CPU phase (check-plan / score / score-f32) as a managed own-group child, receipt OUT/receipts/<name>.json."""
    m, host, b = ctx.m, ctx.host, ctx.m["budget"]
    cap = min(b["cpu_cap_seconds"], ctx.remaining_s() - b["finalize_seconds"])
    run_dir = os.path.join(ctx.out, "runs", name)
    head = {"schema": SCHEMA_STEP, "part": ctx.part, "step": name, "argv": argv, "run_dir": run_dir, "out_dir": out_dir, "fabric": ctx.fabric,
            "caps": {"cpu_cap_seconds": b["cpu_cap_seconds"], "effective_cap_seconds": cap, "remaining_hold_seconds": ctx.remaining_s()}}
    why = start_gate(ctx, name, False, cap, "total")
    if why:
        ctx.event("step_not_started", step=name, why=why)
        if any(w.startswith("SIGNAL") for w in why):
            ctx.aborted = True
        ctx.halt = ctx.halt or any(w.startswith(("SIGNAL", "DEVICE")) for w in why)
        return seal_write(ctx, name, f"receipts/{name}.json", {**head, "started": False, "status": "NOT_STARTED", "failures": ["NOT_STARTED"], "why": why,
                                                                "failure_class": "OTHER", "ok": False})
    ctx.event("step_start", step=name)
    try:
        prepare_run_dir(ctx, run_dir, False)
        env, unset = cpu_env(host.environ(), run_dir)
        with open(os.path.join(run_dir, "command.txt"), "x") as f:
            f.write(" ".join(shlex.quote(a) for a in argv) + "\n")
        run, acc, absorb = run_child(ctx, name, argv, env, run_dir, cap)
    except BaseException as e:  # noqa: BLE001
        return seal_write(ctx, name, f"receipts/{name}.json", {**head, "started": False, "status": "HARNESS_EXCEPTION", "failures": ["HARNESS_EXCEPTION"],
                                                                "exception": f"{type(e).__name__}: {e}", "failure_class": "OTHER", "ok": False})
    dev_after = H2.device_state(host, m)
    ctx.watcher_offset, w_last = H2.watcher_scan(m, ctx.watcher_offset, ctx.watcher_ident)
    absorb(w_last)
    failures = child_failures(run, acc, dev_after, None)
    changed, _, rehash_errs = inputs_strict_changed(m, ctx.snap0)
    if changed:
        failures.append("INPUT_CHANGED")
    missing = []
    if not failures:
        missing = [f for f in expect_files if not os.path.isfile(os.path.join(out_dir, f))]
        if missing:
            failures.append("INVALID_OUTPUT")
    stderr_text = read_tail(os.path.join(run_dir, "stderr.log"))
    status = failures[0] if failures else STATUS_OK
    fclass = classify(failures, stderr_text, ctx.oom_rx)
    stdout_raw = read_bytes(os.path.join(run_dir, "stdout.log"))
    rec = {**head, "started": True, "process": process_block(run), "lifecycle": {"events": run["events"], "leftover": run["leftover_after_exit"]},
           "device_after": dev_after, "watcher": acc, "health_abort": run.get("health_abort"), "missing_outputs": missing,
           "integrity": {"changed_sections": changed, "rehash_errors": rehash_errs[:20]},
           "stdout": {"size_bytes": len(stdout_raw), "sha256": sha256_bytes(stdout_raw), "text": stdout_raw.decode("utf-8", errors="replace")[:65536]},
           "stderr": {"sha256": sha256_file(os.path.join(run_dir, "stderr.log"))}, "failures": failures, "status": status, "failure_class": fclass,
           "ok": status == STATUS_OK}
    if fclass == "BUS" or "ABORTED_SIGNAL" in failures:
        ctx.halt = True
    if "ABORTED_SIGNAL" in failures:
        ctx.aborted = True
    ctx.event("step_end", step=name, status=status)
    return seal_write(ctx, name, f"receipts/{name}.json", rec)


def check_plan_step(ctx):
    """Native `--phase check-plan`: the plan must be valid and name exactly the cells the judge expects."""
    argv = [ctx.m["binary"]["path"], "--phase", "check-plan", "--plan", ctx.m["plan"]["path"]]
    rec = run_cpu_step(ctx, "check-plan", argv, ctx.out, ())
    if not rec["ok"]:
        return rec, [f"check-plan step {rec['status']}"]
    errs = check_plan_output(rec["stdout"]["text"], ctx.plan, ctx.plan_sha)
    return rec, errs


def check_plan_output(text, plan, plan_sha):
    errs = []
    try:
        rep = H2.loads_strict(text)
    except ValueError as e:
        return [f"check-plan stdout is not strict JSON: {e}"]
    if not isinstance(rep, dict) or rep.get("status") != "valid":
        return [f"check-plan status {rep.get('status') if isinstance(rep, dict) else rep!r} != valid"]
    if rep.get("plan_sha256") != plan_sha:
        errs.append("check-plan plan_sha256 differs")
    if not (_is_int(rep.get("max_seq")) and rep["max_seq"] >= MIN_MAX_SEQ):
        errs.append("check-plan max_seq below the minimum")
    for phase in GPU_PHASES:
        want = list(JUDGE.expected_specs(plan, phase))
        if (rep.get("cells") or {}).get(phase) != want:
            errs.append(f"check-plan cells[{phase}] differ from the judge's expected cells")
    return errs


# ----------------------------------------------------------------------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------------------------------------------------------------------
def write_judge_doc(ctx, name, rel, doc):
    path = os.path.join(ctx.out, rel)
    JUDGE.write_json_new(path, doc)
    ctx.judge_docs[name] = doc
    ctx.record_file(name, path)
    return path


def run_judge(ctx, name, rel, fn):
    """Run an in-process judge call; a JudgeError is recorded, never raised."""
    try:
        doc = fn()
    except JudgeErrorTypes() as e:
        ctx.event("judge_failed", judge=name, error=f"{type(e).__name__}: {e}")
        ctx.receipts[name] = {"judge": name, "status": "JUDGE_REFUSED", "error": f"{type(e).__name__}: {e}"}
        return None
    write_judge_doc(ctx, name, rel, doc)
    ctx.event("judge_done", judge=name, verdict=doc["verdict"])
    return doc


def resume_announcement(doc, path, sha):
    prim = doc["primary"]
    return {"event": "RESUME_READY", "resume_json": path, "resume_json_sha256": sha, "verdict_so_far": doc["verdict"], "stage": doc["stage"],
            "primary_gate_status": doc["primary_gate_status"], "resume_disposition": doc["resume_disposition"],
            "f32_exoneration": doc["f32_exoneration"], "determinism_status": doc["determinism_status"],
            "qualification_complete": doc["qualification_complete"], "overall_S1": doc["overall_S1"],
            "resume_pairs_with_kl_ge_gate": doc["resume"].get("pairs_with_kl_ge_gate"),
            "resume_pair_kl_means": {k: v.get("kl_mean_nats") for k, v in doc["resume"].get("pairs", {}).items()},
            "primary_context_means": {k: v["kl_mean_nats"] for k, v in prim["contexts"].items()},
            "pooled_mean": (prim["pooled"] or {}).get("kl_mean_nats"), "default_on_recommended": False}


def run_part_a(ctx):
    plan, m = ctx.m["plan"]["path"], ctx.m
    d = ctx.phase_dirs
    base = run_gpu_phase(ctx, "baseline")
    if not base["ok"]:
        return
    s1a = run_gpu_phase(ctx, "s1a", tail_cpu=True)
    if not s1a["ok"]:
        return
    score_dir = os.path.join(ctx.out, "score")
    run_cpu_step(ctx, "score", [m["binary"]["path"], "--phase", "score", "--plan", plan, "--baseline-dir", d["baseline"], "--s1a-dir", d["s1a"],
                                "--out-dir", score_dir], score_dir, ("kl.jsonl", "kl-summary.json"))
    resume = run_judge(ctx, "resume", "resume.json",
                       lambda: JUDGE.judge_q1a(plan, ctx.plan_sha, d["baseline"], d["s1a"], score_dir))
    if resume is not None:
        sha = ctx.files["resume"]["sha256"]
        emit(resume_announcement(resume, ctx.files["resume"]["path"], sha))
        ctx.event("resume_emitted", verdict=resume["verdict"], resume_disposition=resume["resume_disposition"])
    if ctx.halt or ctx.aborted:
        return
    s1b = run_gpu_phase(ctx, "s1b")
    final = run_judge(ctx, "q1a_summary", "q1a-summary.json",
                      lambda: JUDGE.judge_q1a(plan, ctx.plan_sha, d["baseline"], d["s1a"], score_dir, d["s1b"] if s1b["ok"] else None))
    if final is not None:
        det_path = os.path.join(ctx.out, "determinism.json")
        JUDGE.write_json_new(det_path, final["determinism"])
        ctx.record_file("determinism", det_path)
    return


def run_part_f32(ctx):
    m = ctx.m
    plan, d = m["plan"]["path"], ctx.phase_dirs
    beta = run_gpu_phase(ctx, "beta-f32")
    if not beta["ok"]:
        return
    s1 = run_gpu_phase(ctx, "s1-f32", tail_cpu=True)
    if not s1["ok"]:
        return
    score_dir = os.path.join(ctx.out, "score-f32")
    run_cpu_step(ctx, "score-f32", [m["binary"]["path"], "--phase", "score-f32", "--plan", plan, "--beta-f32-dir", d["beta-f32"], "--s1-f32-dir", d["s1-f32"],
                                   "--out-dir", score_dir], score_dir, ("kl.jsonl", "kl-summary.json"))
    pq = m["prior_q1a"]
    run_judge(ctx, "q1f32_summary", "q1f32-summary.json",
              lambda: JUDGE.judge_f32(plan, ctx.plan_sha, d["beta-f32"], d["s1-f32"], score_dir, pq["summary_path"], pq["summary_sha256"]))


def hold_verdict(ctx):
    """(verdict, reasons). OOM/headroom on any phase = FAIL; bus drop = INCOMPLETE; a judge FAIL stands; anything missing = INCOMPLETE."""
    reasons = []
    bad = [(n, r) for n, r in ctx.receipts.items() if isinstance(r, dict) and r.get("schema") in (SCHEMA_PHASE, SCHEMA_STEP) and not r.get("ok")]
    for n, r in bad:
        reasons.append(f"{n}: {r.get('status')} ({r.get('failure_class')}) {r.get('failures')}")
    if ctx.aborted:
        return "ABORTED", reasons + ["aborted by signal"]
    judge_key = "q1a_summary" if ctx.part == "a" else "q1f32_summary"
    jdoc = ctx.judge_docs.get(judge_key)
    if any(r.get("failure_class") == "OOM" for _, r in bad):
        return "FAIL", reasons + ["OOM/headroom failure: no bypass, no retry"]
    if jdoc is not None and jdoc["verdict"] == "FAIL":
        return "FAIL", reasons + list(jdoc["verdict_reasons"])
    if bad:
        return "INCOMPLETE", reasons
    if jdoc is None:
        return "INCOMPLETE", reasons + ["no final judge document"]
    if jdoc["verdict"] in ("PASS", "REPORTED"):
        return jdoc["verdict"], reasons
    return "INCOMPLETE", reasons + list(jdoc["verdict_reasons"])


def process_identities(ctx):
    """Every GPU phase was a distinct fresh process: pids / start ticks / process groups must all differ."""
    ids = []
    for phase in GPU_PHASES:
        r = ctx.receipts.get(phase)
        if r and r.get("started"):
            p = r["process"]
            ids.append({"phase": phase, "pid": p.get("pid"), "pgid": p.get("pgid"), "starttime_ticks": p.get("starttime_ticks")})
    distinct = len({(i["pid"], i["starttime_ticks"]) for i in ids}) == len(ids) and len({i["pgid"] for i in ids}) == len(ids)
    return ids, distinct


def finalize(ctx):
    verdict, reasons = hold_verdict(ctx)
    ids, distinct = process_identities(ctx)
    if not distinct and verdict == "PASS":
        verdict, reasons = "INCOMPLETE", reasons + ["GPU phase processes were not all distinct"]
    now = ctx.host.monotonic_ns()
    doc = {
        "schema": SCHEMA_RESULT, "controller_version": CONTROLLER_VERSION, "part": ctx.part, "label": ctx.m["label"], "verdict": verdict, "reasons": reasons,
        "default_on_recommended": False, "human_review_required": True,
        "manifest_sha256": ctx.msha, "plan_sha256": ctx.plan_sha, "binary_sha256": ctx.m["binary"]["sha256"], "controller_sha256": ctx.controller_sha256,
        "fabric": ctx.fabric, "frontend_scheduler_id": ctx.frontend_id,
        "phases": {n: {"status": r.get("status"), "failure_class": r.get("failure_class"), "ok": r.get("ok"),
                       "elapsed_seconds": (r.get("process") or {}).get("elapsed_seconds"), "drain": (r.get("drain") or {}).get("last_used") if isinstance(r.get("drain"), dict) else None,
                       "effective_cap_seconds": (r.get("caps") or {}).get("effective_cap_seconds")}
                   for n, r in ctx.receipts.items() if isinstance(r, dict) and r.get("schema") in (SCHEMA_PHASE, SCHEMA_STEP)},
        "judge": {n: {"verdict": d["verdict"], "stage": d.get("stage")} for n, d in ctx.judge_docs.items()},
        "processes": ids, "processes_distinct": distinct,
        "files": ctx.files,
        "hold": {"maximum_hold_seconds": ctx.m["budget"]["maximum_hold_seconds"], "used_seconds": (now - ctx.hold_start_ns) / 1e9,
                 "sum_of_phase_caps": sum(ctx.m["budget"]["phase_cap_seconds"].values()),
                 "note": "phase caps are watchdogs; their sum is not a promise that all of them fit"},
    }
    doc = H2.seal(doc)
    H2.write_json_atomic(os.path.join(ctx.out, "hold-result.json"), doc)
    ctx.event("hold_closed", verdict=verdict)
    return doc


# ----------------------------------------------------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------------------------------------------------
def _check_cli(args):
    errs = []
    for flag in ("expected_manifest_sha256", "expected_controller_sha256", "expected_judge_sha256", "expected_h2_sha256", "expected_binary_sha256",
                 "expected_plan_sha256", "expected_native_source_sha256"):
        if not _is_hex64(getattr(args, flag)):
            errs.append(f"--{flag.replace('_', '-')} must be 64 lowercase hex")
    for flag in ("manifest", "h2_helper"):
        if not _is_abs(getattr(args, flag)):
            errs.append(f"--{flag.replace('_', '-')} must be a normalized absolute path")
    if args.cmd == "run" and not _is_abs(args.out):
        errs.append("--out must be a normalized absolute path")
    if args.cmd == "preflight" and args.report is not None and not _is_abs(args.report):
        errs.append("--report must be a normalized absolute path")
    if errs:
        raise Refusal(errs)


def _admit(args, host):
    """Everything strictly before any mutation. Returns (m, msha, report, plan_info)."""
    _check_cli(args)
    bind_helpers(args.h2_helper, args.expected_h2_sha256, os.path.join(HERE, "q1_judge.py"), args.expected_judge_sha256)
    m, msha = load_manifest(args.manifest, args.expected_manifest_sha256, args.part)
    plan_info = {}
    report = preflight_checks(args, m, msha, host, plan_info)
    return m, msha, report, plan_info


def _out_errors(args, m, report):
    errs = []
    out = args.out
    if os.path.lexists(out):
        errs.append(f"--out {out} already exists (it must be a FRESH directory)")
    if not os.path.isdir(os.path.dirname(out)):
        errs.append(f"parent of --out {out} does not exist")
    protected = [args.manifest, m["plan"]["path"], m["binary"]["path"], m["native_source"]["path"], args.h2_helper, HERE, m["watcher"]["log_path"],
                 os.path.realpath(m["watcher"]["log_path"]), m["target"]["models_dir"]] + list(m["pack_dirs"]) + [e["path"] for e in m["cached_inputs"]]
    if m["part"] == "f32":
        protected.append(m["prior_q1a"]["summary_path"])
    wt = m["native_manifest_template"]["source"].get("worktree")
    if isinstance(wt, str):
        protected.append(wt)
    for p in protected:
        if nested(out, p):
            errs.append(f"--out {out} is nested with {p}: the OUT tree must be private and never inside an input or a source tree")
    return errs


def cmd_preflight(args, host):
    m, msha, report, plan_info = _admit(args, host)
    if report["ok"]:
        # native check-plan in a scratch dir (CPU only), so `preflight` reports the plan verdict too
        with tempfile.TemporaryDirectory(prefix="q1-check-plan-") as scratch:
            from types import SimpleNamespace
            os.mkdir(os.path.join(scratch, "receipts"))
            ctx = SimpleNamespace(m=m, host=host, out=scratch, part=m["part"], fabric=None, halt=False, aborted=False, receipts={}, files={},
                                  hold_start_ns=host.monotonic_ns(), watcher_ident=H2.watcher_identity(m), plan=plan_info["plan"], plan_sha=plan_info["sha"],
                                  oom_rx=re.compile(m["oom_stderr_regex"]), snap0=snapshot(m), controller_sha256=report["controller_sha256"])
            ctx.watcher_offset = ctx.watcher_ident["start_size_bytes"]
            ctx.remaining_s = lambda: m["budget"]["maximum_hold_seconds"]
            ctx.event = lambda *a, **k: None
            ctx.record_file = lambda *a, **k: None
            argv = [m["binary"]["path"], "--phase", "check-plan", "--plan", m["plan"]["path"]]
            rec = run_cpu_step(ctx, "check-plan", argv, scratch, ())
            errs = [f"check-plan step {rec['status']}"] if not rec["ok"] else check_plan_output(rec["stdout"]["text"], plan_info["plan"], plan_info["sha"])
            report["check_plan"] = {"ok": not errs, "errors": errs, "status": rec["status"]}
            report["errors"].extend(errs)
            report["ok"] = not report["errors"]
    report = H2.seal(report)
    if args.report:
        H2.write_json_atomic(args.report, report)
    emit(report)
    return 0 if report["ok"] else 2


def run_hold(args, host):
    """The whole part. Raises Refusal BEFORE any mutation; otherwise returns the sealed hold-result document."""
    hold_start = host.monotonic_ns()
    m, msha, report, plan_info = _admit(args, host)
    errs = list(report["errors"])
    errs.extend(H2.check_fabric(host.environ(), m))
    errs.extend(_out_errors(args, m, report))
    if errs:
        raise Refusal(errs)
    out = args.out
    os.mkdir(out)
    for sub in ("phases", "runs", "receipts"):
        os.mkdir(os.path.join(out, sub))
    H2.write_json_atomic(os.path.join(out, "preflight.json"), H2.seal(report))
    ctx = HoldCtx(args, m, msha, plan_info["plan"], plan_info["sha"], host, out, hold_start, sha256_file(os.path.join(out, "preflight.json")))
    ctx.oom_rx = re.compile(m["oom_stderr_regex"])
    ctx.event("hold_start", part=ctx.part, manifest_sha256=msha, fabric_job=ctx.fabric["job"], frontend_scheduler_id=ctx.frontend_id)
    try:
        rec, perrs = check_plan_step(ctx)
        if perrs or not rec["ok"]:
            ctx.event("check_plan_failed", errors=perrs)
            ctx.receipts["check-plan"] = dict(ctx.receipts.get("check-plan", rec), plan_errors=perrs, ok=False)
        elif ctx.part == "a":
            run_part_a(ctx)
        else:
            run_part_f32(ctx)
    except BaseException as e:  # noqa: BLE001 -- the hold must always close with a result
        ctx.event("harness_exception", error=f"{type(e).__name__}: {e}")
        ctx.receipts["harness"] = {"schema": SCHEMA_STEP, "step": "harness", "status": "HARNESS_EXCEPTION", "failures": ["HARNESS_EXCEPTION"],
                                   "failure_class": "OTHER", "ok": False, "exception": f"{type(e).__name__}: {e}"}
        if isinstance(e, KeyboardInterrupt):
            ctx.aborted = True
    return finalize(ctx)


def cmd_run(args, host):
    old = {}
    handler = lambda signum, frame: STOP.set()  # noqa: E731
    STOP.clear()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old[sig] = signal.signal(sig, handler)
    try:
        try:
            doc = run_hold(args, host)
        except Refusal as e:
            emit({"status": "REFUSED", "errors": e.errors, "mutated": False})
            return 2
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
    emit({"event": "HOLD_CLOSED", "verdict": doc["verdict"], "reasons": doc["reasons"], "hold_result": os.path.join(args.out, "hold-result.json"),
          "hold_result_sha256": sha256_file(os.path.join(args.out, "hold-result.json")), "files": doc["files"], "default_on_recommended": False})
    return VERDICT_EXIT[doc["verdict"]]


def _sha_arg(text):
    if not HEX64.match(text):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return text


def build_parser():
    p = argparse.ArgumentParser(description="Q1 production-route quality hold controller (one held command per part)")
    p.add_argument("cmd", choices=("preflight", "run"))
    p.add_argument("--part", required=True, choices=sorted(PART_PHASES))
    p.add_argument("--manifest", required=True)
    for flag, what in (("manifest", "the RAW manifest bytes"), ("controller", "q1_hold.py"), ("judge", "q1_judge.py"), ("h2", "the frozen h2_official.py"),
                       ("binary", "the native executable"), ("plan", "PLAN.json"), ("native-source", "qwen4_slowlink_quality.rs")):
        p.add_argument(f"--expected-{flag}-sha256", required=True, type=_sha_arg, metavar="SHA", help=f"sha256 of {what}, supplied independently")
    p.add_argument("--h2-helper", required=True, help="absolute path of the frozen h2_official.py (generic helpers)")
    p.add_argument("--frontend-scheduler-id", default=None, help="OPTIONAL opaque frontend scheduler id, recorded as given (null when absent)")
    p.add_argument("--report", default=None, help="preflight only: fresh path for the preflight JSON")
    p.add_argument("--out", default=None, help="run only: FRESH output directory")
    return p


def main(argv=None, host=None):
    args = build_parser().parse_args(argv)
    host = host or Host()
    try:
        if args.cmd == "run" and args.out is None:
            raise Refusal(["run requires --out"])
        if args.cmd == "preflight":
            try:
                return cmd_preflight(args, host)
            except Refusal as e:
                emit({"status": "REFUSED", "errors": e.errors, "mutated": False})
                return 2
        return cmd_run(args, host)
    except Refusal as e:
        emit({"status": "REFUSED", "errors": e.errors, "mutated": False})
        return 2


if __name__ == "__main__":
    sys.exit(main())
