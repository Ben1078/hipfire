#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""h2_official.py -- official H2 (pp16384 shipped-CLI timing) runner, validator and judge.  Python stdlib only.

    h2_official.py preflight      --manifest M --expected-manifest-sha256 M_SHA [--report FILE]
    h2_official.py run            --manifest M --expected-manifest-sha256 M_SHA --pair 1|2|3 --cell 1|2 --out FRESH_DIR
                                  [--previous PAIR1.json [PAIR2.json]] [--first-cell CELL1.json --first-cell-sha256 SHA]
    h2_official.py assemble-pair  --manifest M --expected-manifest-sha256 M_SHA --cells CELL1.json CELL2.json --cell-sha256 S1 S2
                                  [--previous PAIR1.json [PAIR2.json]] --out PAIR.json
    h2_official.py judge          --manifest M --expected-manifest-sha256 M_SHA --pairs PAIR1.json [PAIR2.json [PAIR3.json]] --out RESULT.json

STATUS OF THIS FILE.  SOURCE-ONLY.  Nothing here has been executed on a GPU; the CPU tests drive fake CLIs only.  The 350 tok/s used to size the
hold budget below is an INFERENCE planning floor, never a measurement.  An H2 verdict needs SIX ACTUAL held single-process runs (one fabric job each);
nothing in this file can manufacture, replay or retroactively declare one.

--expected-manifest-sha256 is REQUIRED on every command: the sha256 of the RAW manifest bytes, supplied independently of the manifest (pinned
inline in the submitted argv).  A missing, malformed or different value refuses before preflight/run/assemble/judge touches anything (run: no OUT
dir, no spawn; assemble-pair: no output; judge: UNKNOWN_INVALID, no receipt is read).  Neither the manifest nor this file can vouch for its own hash.

WHAT IS MEASURED.  Per fresh process, exactly this official command (argv[0] is the arm's own CLI):

    hipfire bench MODEL --matrix --pp 16384 --tg 1 --ctx 1 --runs 3 --warmups 1 --max-seq 16416 --json

one untimed warm prefill plus three timed prefills (three samples of `prefill[0].samples`, tokens/second).  `--ctx 1 --tg 1` is a minimal
AUXILIARY decode row only, identical in both arms; it is never a headline.  Three fresh processes per arm in the global order
A1 B1 | B2 A2 | A3 B3 give nine raw samples per arm.  The verdict flattens all nine samples per arm and takes the median of those nine
(NEVER a median of per-process medians); PASS needs candidate >= 2000 tok/s AND candidate >= 1.10 x incumbent.  1400 tok/s is reported as
information only.  Both arms carry HIPFIRE_QWEN4_D2_DEFAULT=0 as the lab COMPATIBILITY POLICY: explicitly NON-DEFAULT, never a default-D2 qualification.  Actual D2 control/consumption is source-bound (older branches may lack the selector); the env value alone never proves D2 is functionally disabled.

SIX SINGLE-PROCESS HOLDS.  One `run` invocation is ONE held command that spawns EXACTLY ONE fresh official CLI (+ its daemon).  The six cells
A1 B1 B2 A2 A3 B3 are six separate holds (six fabric jobs, six fresh OUT dirs, six receipts).  The fabric clamps a timing hold at 1800 s, and one
pp16384 cold cell (cold load + 4 x 16384-token prefills + auxiliary + JIT) is budgeted at cell_cap 1200 s, so two processes can never fit one
hold.  Budget (manifest `budget`, every key REQUIRED): single-hold minimum = cell_cap + kill_grace + drain + finalize + preflight; the planned
values are 1200+15+90+45+30 = 1380 with maximum_hold_seconds 1440.  A manifest whose maximum_hold_seconds exceeds 1800 is refused, so the backend
clamp can never silently squeeze a hold.  The hold clock starts when `run` starts (preflight is inside the hold); BEFORE spawning, the FULL cap +
grace + drain + finalize must fit into what remains (the cap is never shrunk).  Each hold has its own clock; nothing is chained across holds by time.
Every cell receipt records the actual FABRIC_JOB of its own process (a missing FABRIC_JOB / FABRIC_CARDS refuses; the six jobs must be distinct).

PAIRS.  A logical pair is two cell receipts.  cell 1 = ORDER[pair][0], cell 2 = ORDER[pair][1] (pair1 A,B  pair2 B,A  pair3 A,B).  Cell 2 REQUIRES
--first-cell PATH and --first-cell-sha256 SHA (the sha256 of cell 1's cell.json, supplied independently of the receipts): cell 1 must hash to that
value, be an OK measurement of this manifest/pair/order, and have its raw stdout re-validated; its binding is sealed into cell 2.  A failed cell 1
leaves no valid binding, so no further GPU cell runs after a failure (no automatic rerun).  `assemble-pair` is PURE CPU: it re-hashes both cell files
against the independently supplied hashes, revalidates both cells exactly like the judge (raw stdout, artifacts, config, argv, identity, order,
status, chain, fabric jobs, separate processes/OUT dirs) and writes the pair receipt the judge consumes.  Missing, duplicate, swapped, mixed or edited
cells refuse.  A pair receipt always holds two cells; if cell 2 failed the pair is PAIR_INCOMPLETE (and stops the chain).  A failed cell 1 has no
pair (cell 2 cannot have run).  `run --previous` takes the FULL earlier pair receipts and re-verifies them for the chain.

ARMS.  The manifest names arm A and arm B and gives each a role (`candidate` or `incumbent`).  Each arm has its own CLI, its own daemon
(reached ONLY through the supported $HIPFIRE_HOME/bin/daemon symlink; HIPFIRE_DAEMON_BIN is never set), its own kernel pack staged beside the
real daemon, and its own config.toml bytes.  Role config contract (parsed with tomllib, `schema_version` optional and then ==1):
    incumbent:  {memory: {ple_format: "q4"}}
    candidate:  {memory: {ple_format: "q4"}, developer: {qwen4_slowlink_prefill: true}}
The S1 switch is never an environment variable.  The route proof is: typed config + successful load bound to the source identity
(+ an OPTIONAL manifest `route_evidence` stderr regex; an observed N/tile is recorded only when that regex captures it).

PACKS ARE READ-ONLY, THE HOT CACHE IS PRIVATE.  Preflight requires BOTH installed pack dirs, every directory below them and every file in them to be
read-only (no write permission bit; and, when not root, no effective write access): the harness never chmods anything.  A missing-metadata kernel
JIT therefore cannot write the pack and goes to the cell's PRIVATE writable HIPFIRE_KERNEL_CACHE (<cell>/kcache, fresh per cell, never shared).  The
packs are snapshotted (names/modes/dir mtimes) and re-checked after the cell; a change is INPUT_CHANGED.  No performance env pin is ever set.

NO MODEL PAYLOAD IS EVER READ OR HASHED.  The model and its `.ple-q4` sidecar are bound by os.stat (size/dev/ino/mtime_ns, python ints
-- 19-digit nanoseconds never pass through a float) against the manifest AND against a cached sha256 receipt.  ctime_ns is recorded (the actual
observed value is emitted in every report/cell) but is INFORMATIONAL ONLY for cached inputs: a hardlink create/delete on the same inode changes
ctime without touching content, so the cached identity tuple is path/device/inode/size/mtime_ns/sha256 and a ctime-only difference never gates.
(Owned artifacts, packs, seeds are unchanged: they are re-hashed / re-statted strictly.)  Expected sidecars
(`target.absent_paths`, the `.expert-hotset` / `.trunk-overlay` of the model) must stay ABSENT; nothing is created or removed.

MANIFEST `s1.h2_official.v2` (every key REQUIRED unless marked optional; unknown keys are rejected):
  schema, label
  identity: {candidate:{commit,tree}, baseline:{commit,tree}}                 40-hex git ids
  harness:  {h2_official_sha256}                                             sha256 of THIS file; refused when it differs
  target:   {card, uuid, bdf, vendor_id, device_id, model_path, model_size, models_dir, vram_used_path, vram_total_path,
             visible_devices_env:{HIPFIRE_DEVICES, ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES}, absent_paths:[...]}
             vram_*_path live in /sys/bus/pci/devices/<bdf>/ ; vendor/device/unique_id are read from that directory (read-only).
  cached_inputs: [{path,size_bytes,device,inode,mtime_ns,ctime_ns,sha256,cached_receipt_path}]   MUST contain model_path and model_path+".ple-q4"
        cached receipt (strict JSON object) = {"path","size_bytes","device","inode","mtime_ns","ctime_ns","sha256"}; every field except the
        informational ctime_ns (which must still be a present JSON integer) must equal the entry
  artifacts: [{path,size_bytes,sha256}]   owned files hashed at preflight: both arms' cli+daemon, EVERY file below each pack_dir (and nothing
        else there), memory.ledger_path, h1b.receipt_path.  Never the model.
  arms: {A:{role,cli,daemon,pack_dir,config_toml}, B:{...}}   pack_dir == dirname(daemon)/kernels/compiled/<arch>
  budget: {cell_cap_seconds, kill_grace_seconds, drain_seconds, finalize_seconds, preflight_seconds, maximum_hold_seconds}   maximum <= 1800
  memory: {device_gib, host_gib, ledger_path, ledger_sha256}
  h1b: {receipt_path, receipt_sha256, accepted_status}   receipt JSON must carry status == accepted_status and advance is true
  watcher: {log_path, fatal_regex}   the external parent watcher's log (frozen R7 wrapper); this harness only READS it.  log_path is ONE campaign
        symlink: the parent points it at the current fresh wrapper log only INSIDE the admitted held command and records the target externally.
        The harness records the watcher identity (realpath/device/inode/start size) it saw, reads only NEW bytes, and fails the cell closed when the
        log is missing, shrinks or changes identity.  It never starts a second watcher, never calls journalctl or a GPU API; the judge never rereads
        the log (the receipt carries the watcher failure status).
  home_seed_files (optional): [{source,dest,size_bytes,sha256}]   small exact files (<= 1 MiB) copied byte-exact into each fresh $HIPFIRE_HOME
  route_evidence (optional): {stderr_regex, required, incumbent_forbidden_regex}   regex may name groups `n` and `tile`

LIFECYCLE.  The cell is a Popen(start_new_session=True, close_fds=True); pid/pgid/starttime are recorded to process.json BEFORE waiting.
stdout/stderr are regular files.  Timeout / signal -> SIGTERM to OWN group, bounded grace, SIGKILL to OWN group, reap.  After the CLI exits
the same group is swept (a daemon the CLI failed to reap is terminated and the cell is LIFECYCLE_LEFTOVER).  The card must be idle
(< 2 GiB) before spawn and drain back below it afterwards.  WHILE the CLI runs, the wait loop calls a health check about every 250 ms (target PCI
vendor/device/unique_id sysfs reads + NEW bytes of the external watcher log): a vanished/0xffff/changed device (DEVICE_LOST), a fatal watcher line
(WATCHER_FATAL) or an unreadable/shrunk/retargeted watcher log (WATCHER_UNREADABLE) terminates ONLY the recorded own group at once, preserves the
cause/matches in the cell receipt (`health_abort`, `watcher`) and closes the cell failed with no retry.  The same checks run again after the CLI
exits and after the drain.  A full-cap wait on a lost bus is therefore bounded to the health cadence.
No pattern kills, no automatic reruns, nothing is zero-filled: a missing/failed cell is UNKNOWN.

EXIT CODES.  preflight 0 ok / 2 not ok.  run 0 CELL_OK / 2 refused before any mutation / 3 cell closed failed / 4 aborted by signal.
assemble-pair 0 PAIR_COMPLETE / 2 refused (nothing written) / 3 PAIR_INCOMPLETE written.
judge 0 PASS / 1 FAIL or FAIL_EARLY_ORDER_STATISTIC / 3 UNKNOWN_*."""
import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import stat as statmod
import subprocess
import sys
import threading
import time
import tomllib
from fractions import Fraction

SCHEMA_MANIFEST = "s1.h2_official.v2"
SCHEMA_PREFLIGHT = "s1.h2_official.preflight.v1"
SCHEMA_CELL = "s1.h2_official.cell.v2"
SCHEMA_PAIR = "s1.h2_official.pair.v2"
SCHEMA_RESULT = "s1.h2_official.result.v1"

PP = 16384
TG = 1
CTX = 1
RUNS = 3
WARMUPS = 1
MAX_SEQ = 16416
PROTOCOL = "synthetic-pp-tg-matrix-v1"
PREFILL_ROUTE = "bench_prefill (synthetic ids, no speculative prompt fill)"
SAMPLES_PER_PROCESS = 3
PROCESSES_PER_ARM = 3
SAMPLES_PER_ARM = SAMPLES_PER_PROCESS * PROCESSES_PER_ARM
TARGET_TOK_S = 2000
RATIO_NUM, RATIO_DEN = 11, 10          # 1.10 as an exact rational
FLOOR_TOK_S = 1400                      # informational only
IDLE_MAX = 2 << 30
ORDER = {1: ("A", "B"), 2: ("B", "A"), 3: ("A", "B")}
GLOBAL_ORDER = ("A1", "B1", "B2", "A2", "A3", "B3")
ROLES = ("candidate", "incumbent")
MAX_ARTIFACT_BYTES = 4 << 30
MAX_SEED_BYTES = 1 << 20
MAX_SMALL_JSON = 1 << 20
ENV_STRIP_PREFIXES = ("HIPFIRE_", "HIP_", "ROCR_", "ROC_", "HSA_", "GPU_", "AMD_")
ENV_KEEP_EXACT = ("HIPFIRE_FABRIC_ADMISSION",)
D2_PIN = ("HIPFIRE_QWEN4_D2_DEFAULT", "0")
POLL_S = 0.05
HEALTH_INTERVAL_S = 0.25
MAX_LOG_MATCHES = 5
HOLD_CLAMP_SECONDS = 1800               # the fabric hard-clamps a timing job; a manifest above it is refused so the clamp can never squeeze a hold
CELLS = (1, 2)

STATUS_OK = "OK"
CELL_STATUSES = ("OK", "SPAWN_ERROR", "ABORTED_SIGNAL", "TIMEOUT", "EXIT_NONZERO", "LIFECYCLE_LEFTOVER", "INVALID_OUTPUT",
                 "ROUTE_EVIDENCE_MISSING", "ROUTE_CONTAMINATION", "DEVICE_LOST", "WATCHER_FATAL", "WATCHER_UNREADABLE", "DRAIN_TIMEOUT",
                 "INPUT_CHANGED", "NOT_STARTED", "HARNESS_EXCEPTION")

STOP = threading.Event()


class H2Error(Exception):
    """A strict-contract violation."""


class Refusal(H2Error):
    """`run` refused BEFORE any mutation (no out dir, no process, no GPU)."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


# ----------------------------------------------------------------------------------------------------------------------------------
# host abstraction (every clock / sleep / sysfs / env read of the real run path goes through here so CPU tests can inject)
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
# strict JSON / hashing / atomic files
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
    """One JSON document; duplicate keys, NaN/Infinity and overflowing floats are errors. Integers stay python ints."""
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


def read_small(path, limit=MAX_SMALL_JSON):
    with open(path, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise H2Error(f"{path} is larger than {limit} bytes")
    return data


def seal(doc):
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    body["receipt_sha256"] = sha256_bytes(canon(body))
    return body


def seal_ok(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("receipt_sha256"), str):
        return False
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    return sha256_bytes(canon(body)) == doc["receipt_sha256"]


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, data):
    """Create `path` atomically and NEVER overwrite (link fails when it exists)."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)
    finally:
        os.unlink(tmp)
    _fsync_dir(d)


def write_json_atomic(path, doc):
    write_atomic(path, json.dumps(doc, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def stat_tuple(st):
    return {"size_bytes": st.st_size, "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


# Cached model/PLE identity gate: ctime_ns is deliberately NOT a member (informational only; see the module docstring).
CACHED_GATE_KEYS = ("size_bytes", "device", "inode", "mtime_ns")


def cached_gate(t):
    """The gating subset of a cached-input stat dict; a non-dict value (an 'unreadable: ...' string) is returned unchanged so it never matches."""
    return {k: t[k] for k in CACHED_GATE_KEYS if k in t} if isinstance(t, dict) else t


# ----------------------------------------------------------------------------------------------------------------------------------
# manifest validation (pure; no filesystem)
# ----------------------------------------------------------------------------------------------------------------------------------
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
HEXID = re.compile(r"^0x[0-9a-f]{4}$")


def _is_int(v):
    return type(v) is int


def _is_num(v):
    return type(v) in (int, float) and math.isfinite(v)


def _is_str(v):
    return isinstance(v, str) and v != ""


def _is_abs(v):
    return _is_str(v) and os.path.isabs(v) and os.path.normpath(v) == v and "\x00" not in v


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


def single_hold_min(b):
    """Minimum maximum_hold_seconds of ONE single-process hold: full cell cap + kill grace + drain + finalize + preflight."""
    return b["cell_cap_seconds"] + b["kill_grace_seconds"] + b["drain_seconds"] + b["finalize_seconds"] + b["preflight_seconds"]


def expected_config(role):
    cfg = {"memory": {"ple_format": "q4"}}
    if role == "candidate":
        cfg["developer"] = {"qwen4_slowlink_prefill": True}
    return cfg


def check_config_text(text, role, where, errs):
    try:
        parsed = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, TypeError) as e:
        errs.append(f"{where}: not TOML: {e}")
        return
    ver = parsed.pop("schema_version", 1)
    if not _is_int(ver) or ver != 1:
        errs.append(f"{where}: schema_version must be 1")
    if parsed != expected_config(role):
        errs.append(f"{where}: parsed config {parsed!r} != the {role} contract {expected_config(role)!r}")


def validate_manifest(m):
    """List of errors (empty = structurally valid). No filesystem access."""
    errs = []
    if not _keys(m, "manifest", ("schema", "label", "identity", "harness", "target", "cached_inputs", "artifacts", "arms", "budget",
                                "memory", "h1b", "watcher"), ("home_seed_files", "route_evidence"), errs):
        return errs
    if m["schema"] != SCHEMA_MANIFEST:
        errs.append(f"schema must be {SCHEMA_MANIFEST}")
    if not _is_str(m["label"]):
        errs.append("label must be a non-empty string")
    ident = m["identity"]
    if _keys(ident, "identity", ("candidate", "baseline"), (), errs):
        for side in ("candidate", "baseline"):
            if _keys(ident[side], f"identity.{side}", ("commit", "tree"), (), errs):
                for k in ("commit", "tree"):
                    if not (isinstance(ident[side][k], str) and HEX40.match(ident[side][k])):
                        errs.append(f"identity.{side}.{k} must be 40 lowercase hex")
    if _keys(m["harness"], "harness", ("h2_official_sha256",), (), errs):
        if not (isinstance(m["harness"]["h2_official_sha256"], str) and HEX64.match(m["harness"]["h2_official_sha256"])):
            errs.append("harness.h2_official_sha256 must be 64 lowercase hex")
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
    # cached inputs
    ci = m["cached_inputs"]
    ci_keys = ("path", "size_bytes", "device", "inode", "mtime_ns", "ctime_ns", "sha256", "cached_receipt_path")
    paths = []
    if not isinstance(ci, list) or not ci:
        errs.append("cached_inputs must be a non-empty list")
    else:
        for i, e in enumerate(ci):
            if _keys(e, f"cached_inputs[{i}]", ci_keys, (), errs):
                if not _is_abs(e["path"]) or not _is_abs(e["cached_receipt_path"]):
                    errs.append(f"cached_inputs[{i}]: path and cached_receipt_path must be absolute")
                for k in ("size_bytes", "device", "inode", "mtime_ns", "ctime_ns"):
                    if not (_is_int(e[k]) and e[k] >= 0):
                        errs.append(f"cached_inputs[{i}].{k} must be a JSON INTEGER >= 0 (floats/strings are refused: 19-digit ns must stay exact)")
                if not (isinstance(e["sha256"], str) and HEX64.match(e["sha256"])):
                    errs.append(f"cached_inputs[{i}].sha256 must be 64 lowercase hex")
                paths.append(e["path"])
        if len(set(paths)) != len(paths):
            errs.append("cached_inputs paths must be unique")
        if isinstance(t, dict) and _is_abs(t.get("model_path")):
            for need in (t["model_path"], t["model_path"] + ".ple-q4"):
                if need not in paths:
                    errs.append(f"cached_inputs must bind {need}")
            for e in ci:
                if isinstance(e, dict) and e.get("path") == t["model_path"] and _is_int(e.get("size_bytes")) and e["size_bytes"] != t.get("model_size"):
                    errs.append("cached_inputs model size_bytes != target.model_size")
    # artifacts
    art = m["artifacts"]
    art_paths = []
    if not isinstance(art, list) or not art:
        errs.append("artifacts must be a non-empty list")
    else:
        for i, e in enumerate(art):
            if _keys(e, f"artifacts[{i}]", ("path", "size_bytes", "sha256"), (), errs):
                if not _is_abs(e["path"]):
                    errs.append(f"artifacts[{i}].path must be absolute")
                if not (_is_int(e["size_bytes"]) and 0 < e["size_bytes"] <= MAX_ARTIFACT_BYTES):
                    errs.append(f"artifacts[{i}].size_bytes must be an integer in 1..{MAX_ARTIFACT_BYTES} (an owned file, never the model)")
                if not (isinstance(e["sha256"], str) and HEX64.match(e["sha256"])):
                    errs.append(f"artifacts[{i}].sha256 must be 64 lowercase hex")
                art_paths.append(e["path"])
        if len(set(art_paths)) != len(art_paths):
            errs.append("artifacts paths must be unique")
        if isinstance(t, dict) and _is_abs(t.get("models_dir")):
            for p in art_paths:
                if p in paths or (isinstance(p, str) and (p == t.get("model_path") or p.startswith(t["models_dir"] + os.sep))):
                    errs.append(f"artifact {p} is a model-side file: payloads are never hashed")
    # arms
    arms = m["arms"]
    if _keys(arms, "arms", ("A", "B"), (), errs):
        roles = []
        for label in ("A", "B"):
            a = arms[label]
            if not _keys(a, f"arms.{label}", ("role", "cli", "daemon", "pack_dir", "config_toml"), (), errs):
                continue
            roles.append(a["role"])
            if a["role"] not in ROLES:
                errs.append(f"arms.{label}.role must be one of {ROLES}")
            for k in ("cli", "daemon", "pack_dir"):
                if not _is_abs(a[k]):
                    errs.append(f"arms.{label}.{k} must be absolute")
            for k in ("cli", "daemon"):
                if _is_abs(a[k]) and a[k] not in art_paths:
                    errs.append(f"arms.{label}.{k} must be a declared artifact")
            if _is_abs(a["daemon"]) and _is_abs(a["pack_dir"]):
                if os.path.dirname(os.path.dirname(a["pack_dir"])) != os.path.join(os.path.dirname(a["daemon"]), "kernels") \
                        or os.path.basename(os.path.dirname(a["pack_dir"])) != "compiled":
                    errs.append(f"arms.{label}.pack_dir must be dirname(daemon)/kernels/compiled/<arch>")
            if a["role"] in ROLES:
                if not isinstance(a["config_toml"], str):
                    errs.append(f"arms.{label}.config_toml must be a string")
                else:
                    check_config_text(a["config_toml"], a["role"], f"arms.{label}.config_toml", errs)
        if sorted(roles) != sorted(ROLES):
            errs.append("exactly one arm must be the candidate and one the incumbent")
        if isinstance(arms.get("A"), dict) and isinstance(arms.get("B"), dict):
            for k in ("cli", "daemon", "pack_dir"):
                if arms["A"].get(k) == arms["B"].get(k):
                    errs.append(f"arms A and B must not share {k}")
    # budget
    b = m["budget"]
    if _keys(b, "budget", ("cell_cap_seconds", "kill_grace_seconds", "drain_seconds", "finalize_seconds", "preflight_seconds",
                           "maximum_hold_seconds"), (), errs):
        for k, v in b.items():
            if not (_is_num(v) and v > 0):
                errs.append(f"budget.{k} must be a positive number")
        if all(_is_num(b[k]) for k in b):
            if b["maximum_hold_seconds"] < single_hold_min(b):
                errs.append("budget.maximum_hold_seconds cannot hold even one full single-process hold (cap + grace + drain + finalize + preflight)")
            if b["maximum_hold_seconds"] > HOLD_CLAMP_SECONDS:
                errs.append(f"budget.maximum_hold_seconds exceeds the fabric's {HOLD_CLAMP_SECONDS} s timing-job clamp: the clamp must never squeeze a hold")
    # memory
    mem = m["memory"]
    if _keys(mem, "memory", ("device_gib", "host_gib", "ledger_path", "ledger_sha256"), (), errs):
        for k in ("device_gib", "host_gib"):
            if not (_is_num(mem[k]) and mem[k] > 0):
                errs.append(f"memory.{k} must be a positive number")
        if not _is_abs(mem["ledger_path"]):
            errs.append("memory.ledger_path must be absolute")
        elif mem["ledger_path"] not in art_paths:
            errs.append("memory.ledger_path must be a declared artifact")
        if not (isinstance(mem["ledger_sha256"], str) and HEX64.match(mem["ledger_sha256"])):
            errs.append("memory.ledger_sha256 must be 64 lowercase hex")
        elif _is_abs(mem["ledger_path"]):
            for e in art if isinstance(art, list) else []:
                if isinstance(e, dict) and e.get("path") == mem["ledger_path"] and e.get("sha256") != mem["ledger_sha256"]:
                    errs.append("memory.ledger_sha256 differs from the artifact entry")
    h1b = m["h1b"]
    if _keys(h1b, "h1b", ("receipt_path", "receipt_sha256", "accepted_status"), (), errs):
        if not _is_abs(h1b["receipt_path"]):
            errs.append("h1b.receipt_path must be absolute")
        elif h1b["receipt_path"] not in art_paths:
            errs.append("h1b.receipt_path must be a declared artifact")
        else:
            for e in art:
                if isinstance(e, dict) and e.get("path") == h1b["receipt_path"] and e.get("sha256") != h1b["receipt_sha256"]:
                    errs.append("h1b.receipt_sha256 differs from the artifact entry")
        if not (isinstance(h1b["receipt_sha256"], str) and HEX64.match(h1b["receipt_sha256"])):
            errs.append("h1b.receipt_sha256 must be 64 lowercase hex")
        if not _is_str(h1b["accepted_status"]):
            errs.append("h1b.accepted_status must be a non-empty string")
    w = m["watcher"]
    if _keys(w, "watcher", ("log_path", "fatal_regex"), (), errs):
        if not _is_abs(w["log_path"]):
            errs.append("watcher.log_path must be absolute")
        if not _is_str(w["fatal_regex"]):
            errs.append("watcher.fatal_regex must be a non-empty string")
        else:
            try:
                re.compile(w["fatal_regex"])
            except re.error as e:
                errs.append(f"watcher.fatal_regex does not compile: {e}")
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
                elif d == "config.toml" or d.split(os.sep)[0] in ("bin", "models"):
                    errs.append(f"home_seed_files[{i}].dest {d!r} collides with a harness-owned entry")
                dests.append(d)
                if not (_is_int(e["size_bytes"]) and 0 < e["size_bytes"] <= MAX_SEED_BYTES):
                    errs.append(f"home_seed_files[{i}].size_bytes must be an integer in 1..{MAX_SEED_BYTES}")
                if not (isinstance(e["sha256"], str) and HEX64.match(e["sha256"])):
                    errs.append(f"home_seed_files[{i}].sha256 must be 64 lowercase hex")
            if len(set(dests)) != len(dests):
                errs.append("home_seed_files dests must be unique")
    if "route_evidence" in m:
        r = m["route_evidence"]
        if _keys(r, "route_evidence", ("stderr_regex", "required"), ("incumbent_forbidden_regex",), errs):
            if not isinstance(r["required"], bool):
                errs.append("route_evidence.required must be a boolean")
            for k in ("stderr_regex", "incumbent_forbidden_regex"):
                if k in r and r[k] is not None:
                    if not _is_str(r[k]):
                        errs.append(f"route_evidence.{k} must be a non-empty string or null")
                    else:
                        try:
                            re.compile(r[k])
                        except re.error as e:
                            errs.append(f"route_evidence.{k} does not compile: {e}")
            if not _is_str(r["stderr_regex"]):
                errs.append("route_evidence.stderr_regex must be a non-empty string")
    return errs


def load_manifest(path, expected_sha256):
    """(manifest, raw_sha256). The sha256 of the RAW manifest bytes MUST equal `expected_sha256`, a value supplied independently of the manifest
    (the manifest and this harness cannot vouch for themselves). The comparison happens before the bytes are parsed or trusted.
    Raises H2Error with every structural error."""
    if not (isinstance(expected_sha256, str) and HEX64.match(expected_sha256)):
        raise H2Error("expected manifest sha256 is missing or not 64 lowercase hex: an independently supplied value is REQUIRED")
    try:
        raw = read_small(path, 16 << 20)
    except (OSError, H2Error) as e:
        raise H2Error(f"manifest {path} unreadable: {e}")
    got = sha256_bytes(raw)
    if got != expected_sha256:
        raise H2Error(f"manifest raw-bytes sha256 {got} != the independently supplied expected {expected_sha256}")
    try:
        m = loads_strict(raw)
    except ValueError as e:
        raise H2Error(f"manifest {path} is not strict JSON: {e}")
    errs = validate_manifest(m)
    if errs:
        raise H2Error("manifest invalid: " + " | ".join(errs))
    return m, got


def arm_label_for_role(m, role):
    for label in ("A", "B"):
        if m["arms"][label]["role"] == role:
            return label
    raise H2Error(f"no arm has role {role}")


def expected_argv(m, arm):
    return [m["arms"][arm]["cli"], "bench", m["target"]["model_path"], "--matrix", "--pp", str(PP), "--tg", str(TG), "--ctx", str(CTX),
            "--runs", str(RUNS), "--warmups", str(WARMUPS), "--max-seq", str(MAX_SEQ), "--json"]


def config_bytes(m, arm):
    return m["arms"][arm]["config_toml"].encode("utf-8")


def self_sha256():
    return sha256_file(os.path.realpath(__file__))


# ----------------------------------------------------------------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------------------------------------------------------------
def build_env(parent_env, cell_dir, target):
    """(env, unset_names, set_dict). Only routing metadata + noRegistryFetch + the D2 pin are applied. No S1/placement/chunk/stage/KV/graph/
    offload/PLE env is ever set, and every HIPFIRE_/HIP_/ROCR_/ROC_/HSA_/GPU_/AMD_ variable inherited from the parent (except the fabric
    admission one) is removed."""
    home = os.path.join(cell_dir, "home")
    hh = os.path.join(home, ".hipfire")
    vis = target["visible_devices_env"]
    applied = {
        "HOME": home,
        "HIPFIRE_HOME": hh,
        "HIPFIRE_LOCK_DIR": os.path.join(cell_dir, "lockdir"),
        "HIPFIRE_KERNEL_CACHE": os.path.join(cell_dir, "kcache"),
        "HIPFIRE_DEVICES": vis["HIPFIRE_DEVICES"],
        "ROCR_VISIBLE_DEVICES": vis["ROCR_VISIBLE_DEVICES"],
        "HIP_VISIBLE_DEVICES": vis["HIP_VISIBLE_DEVICES"],
        "HIPFIRE_NO_REGISTRY_FETCH": "1",
        D2_PIN[0]: D2_PIN[1],
    }
    env, unset = {}, []
    for k, v in parent_env.items():
        if k in ENV_KEEP_EXACT or k.startswith("FABRIC_"):
            env[k] = v
        elif k.startswith(ENV_STRIP_PREFIXES):
            unset.append(k)
        else:
            env[k] = v
    env.update(applied)
    for k in env:
        if k.startswith(ENV_STRIP_PREFIXES) and k not in applied and k not in ENV_KEEP_EXACT:
            raise H2Error(f"internal: env {k} survived sanitising")
    return env, sorted(unset), dict(sorted(applied.items()))


# ----------------------------------------------------------------------------------------------------------------------------------
# preflight (real files; sha256 ONLY of declared owned artifacts; NEVER the model)
# ----------------------------------------------------------------------------------------------------------------------------------
def _lstat_regular(path, where, errs, follow=False):
    try:
        st = os.stat(path) if follow else os.lstat(path)
    except OSError as e:
        errs.append(f"{where}: {path} unreadable: {e}")
        return None
    if not statmod.S_ISREG(st.st_mode):
        errs.append(f"{where}: {path} is not a regular file" + ("" if follow else " (symlinks/dirs are refused)"))
        return None
    return st


def check_cached_inputs(m, errs):
    """os.stat/receipt metadata only. Returns [{path, stat..., receipt_sha256}]."""
    out = []
    for e in m["cached_inputs"]:
        where = f"cached_inputs {e['path']}"
        st = _lstat_regular(e["path"], where, errs, follow=True)
        if st is None:
            continue
        cur = stat_tuple(st)
        for k in CACHED_GATE_KEYS:
            if e[k] != cur[k]:
                errs.append(f"{where}: {k} manifest {e[k]} != current {cur[k]}")
        receipt_sha = None
        try:
            raw = read_small(e["cached_receipt_path"])
            rec = loads_strict(raw)
            receipt_sha = sha256_bytes(raw)
            if not isinstance(rec, dict):
                raise ValueError("receipt is not an object")
            want = ("path", "size_bytes", "device", "inode", "mtime_ns", "ctime_ns", "sha256")
            if sorted(rec) != sorted(want):
                errs.append(f"{where}: receipt keys {sorted(rec)} != {sorted(want)}")
            else:
                for k in want:
                    if k == "ctime_ns":
                        if not _is_int(rec[k]) or rec[k] < 0:
                            errs.append(f"{where}: receipt ctime_ns {rec[k]!r} must be a JSON INTEGER >= 0 (informational, not compared)")
                    elif type(rec[k]) is not type(e[k]) or rec[k] != e[k]:
                        errs.append(f"{where}: receipt {k} {rec[k]!r} != manifest {e[k]!r}")
        except (OSError, ValueError, H2Error) as ex:
            errs.append(f"{where}: cached receipt {e['cached_receipt_path']}: {ex}")
        out.append({"path": e["path"], **cur, "sha256": e["sha256"], "receipt_sha256": receipt_sha})
    return out


def _walk_pack(pack_dir, errs):
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
    """Hash exactly the declared owned artifacts. Returns [{path,size_bytes,sha256,device,inode,mtime_ns,ctime_ns}]."""
    out = []
    declared = {e["path"] for e in m["artifacts"]}
    for label in ("A", "B"):
        pack = m["arms"][label]["pack_dir"]
        on_disk = set(_walk_pack(pack, errs))
        in_manifest = {p for p in declared if p.startswith(pack + os.sep)}
        for p in sorted(on_disk - in_manifest):
            errs.append(f"arm {label}: undeclared file in pack_dir: {p}")
        for p in sorted(in_manifest - on_disk):
            errs.append(f"arm {label}: declared pack file missing: {p}")
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
        out.append({"path": e["path"], "size_bytes": st.st_size, "sha256": got, **{k: v for k, v in stat_tuple(st).items() if k != "size_bytes"}})
    return out


def check_packs_readonly(m, errs):
    """Both installed pack dirs, every directory below them and every file in them must be READ-ONLY: no write permission bit, and (when not root)
    no effective write access. The harness never chmods anything. Returns {arm: {verified, dirs, files, effective_access_checked}}."""
    out = {}
    check_access = os.geteuid() != 0
    for label in ("A", "B"):
        pack = m["arms"][label]["pack_dir"]
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
                errs.append(f"arm {label}: pack path {p} unreadable: {e}")
                continue
            if statmod.S_ISLNK(st.st_mode):
                continue                                   # symlinks are refused elsewhere
            if statmod.S_ISDIR(st.st_mode):
                n_dirs += 1
            else:
                n_files += 1
            if st.st_mode & 0o222:
                errs.append(f"arm {label}: pack path {p} is writable by mode ({statmod.filemode(st.st_mode)}): the installed pack must be chmod a-w")
            elif check_access and os.access(p, os.W_OK):
                errs.append(f"arm {label}: pack path {p} has effective write access: the installed pack must be read-only")
        out[label] = {"verified": len(errs) == n0, "dirs": n_dirs, "files": n_files, "effective_access_checked": check_access}
    return out


def check_h1b(m, errs):
    h = m["h1b"]
    try:
        doc = loads_strict(read_small(h["receipt_path"], 64 << 20))
    except (OSError, ValueError, H2Error) as e:
        errs.append(f"h1b receipt unreadable: {e}")
        return
    if not isinstance(doc, dict) or doc.get("status") != h["accepted_status"] or doc.get("advance") is not True:
        errs.append(f"h1b receipt must carry status == {h['accepted_status']!r} and advance is true")


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
        out.append({"source": e["source"], "dest": e["dest"], "sha256": got, **stat_tuple(st)})
    return out


def device_dir(m):
    return os.path.dirname(m["target"]["vram_used_path"])


def device_state(host, m):
    """Read-only identity of the target card. ok False on a vanished device or an 0xffff / mismatching vendor or device id."""
    d = device_dir(m)
    t = m["target"]
    info = {"ok": True, "problems": []}
    for name, want in (("vendor", t["vendor_id"]), ("device", t["device_id"])):
        try:
            got = host.read_text(os.path.join(d, name)).strip().lower()
        except OSError as e:
            info["ok"] = False
            info["problems"].append(f"{name} unreadable: {e}")
            continue
        info[name] = got
        if got == "0xffff" or got != want:
            info["ok"] = False
            info["problems"].append(f"{name} {got} != {want}")
    try:
        uid = host.read_text(os.path.join(d, "unique_id")).strip()
        info["unique_id"] = uid
        if uid != t["uuid"].removeprefix("GPU-"):
            info["ok"] = False
            info["problems"].append(f"unique_id {uid} != target uuid {t['uuid']}")
    except OSError as e:
        info["ok"] = False
        info["problems"].append(f"unique_id unreadable: {e}")
    return info


def preflight(manifest_path, expected_manifest_sha256, host=None):
    """(report, manifest|None). Never touches a GPU, never reads a model body."""
    host = host or Host()
    report = {"schema": SCHEMA_PREFLIGHT, "ok": False, "errors": [], "expected_manifest_sha256": expected_manifest_sha256}
    errs = report["errors"]
    try:
        m, msha = load_manifest(manifest_path, expected_manifest_sha256)
    except H2Error as e:
        errs.append(str(e))
        return report, None
    report["manifest_sha256"] = msha
    hsha = self_sha256()
    report["harness_sha256"] = hsha
    if hsha != m["harness"]["h2_official_sha256"]:
        errs.append(f"harness sha256 {hsha} != manifest.harness.h2_official_sha256")
    t = m["target"]
    if not os.path.isdir(t["models_dir"]):
        errs.append(f"models_dir {t['models_dir']} is not a directory")
    for p in t["absent_paths"]:
        if os.path.lexists(p):
            errs.append(f"expected-absent path now exists: {p}")
    for k in ("model_path",):
        if not os.path.isfile(t[k]):
            errs.append(f"{k} {t[k]} missing")
    report["cached_inputs"] = check_cached_inputs(m, errs)
    report["artifacts"] = check_artifacts(m, errs)
    report["pack_readonly"] = check_packs_readonly(m, errs)
    report["home_seed_files"] = check_seed_files(m, errs)
    check_h1b(m, errs)
    for label in ("A", "B"):
        if not os.access(m["arms"][label]["cli"], os.X_OK):
            errs.append(f"arm {label} cli is not executable")
        if not os.access(m["arms"][label]["daemon"], os.X_OK):
            errs.append(f"arm {label} daemon is not executable")
    report["device"] = device_state(host, m)
    if not report["device"]["ok"]:
        errs.append("target device check failed: " + "; ".join(report["device"]["problems"]))
    try:
        report["device_vram_used"] = int(host.read_text(t["vram_used_path"]).strip())
    except (OSError, ValueError) as e:
        report["device_vram_used"] = None
        errs.append(f"vram_used unreadable: {e}")
    report["roles"] = {label: m["arms"][label]["role"] for label in ("A", "B")}
    report["argv"] = {label: expected_argv(m, label) for label in ("A", "B")}
    report["config_sha256"] = {label: sha256_bytes(config_bytes(m, label)) for label in ("A", "B")}
    report["ok"] = not errs
    return report, m


# ----------------------------------------------------------------------------------------------------------------------------------
# process group lifecycle
# ----------------------------------------------------------------------------------------------------------------------------------
def proc_stat(pid):
    with open(f"/proc/{pid}/stat") as f:
        data = f.read()
    rp = data.rindex(")")
    comm = data[data.index("(") + 1:rp]
    rest = data[rp + 2:].split()
    return {"pid": pid, "comm": comm, "state": rest[0], "ppid": int(rest[1]), "pgrp": int(rest[2]), "session": int(rest[3]),
            "starttime": int(rest[19])}


def group_members(pgid):
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            st = proc_stat(int(name))
        except (OSError, ValueError, IndexError):
            continue
        if st["pgrp"] == pgid and st["session"] == pgid:
            out.append(st)
    return out


def live_members(pgid):
    return [s for s in group_members(pgid) if s["state"] not in ("Z", "X")]


def _brief(members):
    return [{"pid": s["pid"], "comm": s["comm"], "state": s["state"], "starttime": s["starttime"]} for s in members]


def _signal_group(pgid, sig, events, why, host):
    members = live_members(pgid)
    if not members:
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    events.append({"t_mono_ns": host.monotonic_ns(), "signal": signal.Signals(sig).name, "why": why, "members": _brief(members)})
    return True


def _await_gone(proc, pgid, seconds, host):
    end = host.monotonic_ns() + int(seconds * 1e9)
    while True:
        proc.poll()
        if proc.returncode is not None and not live_members(pgid):
            return True
        if host.monotonic_ns() >= end:
            return False
        host.sleep(POLL_S)


def terminate_group(proc, pgid, grace_s, why, events, host):
    """SIGTERM own group, bounded grace, SIGKILL own group, reap. Never signals anything outside the recorded PGID."""
    _signal_group(pgid, signal.SIGTERM, events, why, host)
    if _await_gone(proc, pgid, grace_s, host):
        return
    _signal_group(pgid, signal.SIGKILL, events, why + " (grace expired)", host)
    if not _await_gone(proc, pgid, max(grace_s, 1.0), host):
        events.append({"t_mono_ns": host.monotonic_ns(), "error": "group still has live members after SIGKILL", "members": _brief(live_members(pgid))})
    if proc.returncode is None:
        try:
            proc.wait(timeout=max(grace_s, 1.0))
        except subprocess.TimeoutExpired:
            events.append({"t_mono_ns": host.monotonic_ns(), "error": "leader not reapable after SIGKILL"})


def spawn_and_wait(argv, env, cwd, stdout_path, stderr_path, process_path, cap_s, grace_s, host, health=None, health_interval_s=HEALTH_INTERVAL_S):
    """Run one CLI cell. Returns a dict (never raises for a CLI failure).

    `health` (optional) is called about every `health_interval_s` WHILE the CLI runs. It returns a list of cause dicts (empty = healthy,
    each {"status": DEVICE_LOST|WATCHER_FATAL|WATCHER_UNREADABLE, ...}). A non-empty answer terminates ONLY the recorded own group at once
    (SIGTERM, bounded grace, SIGKILL) and is preserved in res["health_abort"]; the cell is then never valid."""
    res = {"argv": argv, "cwd": cwd, "spawned": False, "timed_out": False, "aborted": False, "returncode": None, "signal": None,
           "events": [], "leftover_after_exit": [], "health_abort": None}
    fo = open(stdout_path, "xb")
    try:
        fe = open(stderr_path, "xb")
    except BaseException:
        fo.close()
        raise
    proc = None
    pgid = None
    try:
        t0 = host.monotonic_ns()
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, env=env, cwd=cwd, start_new_session=True,
                                    close_fds=True)
        except OSError as e:
            res["spawn_error"] = f"{type(e).__name__}: {e}"
            return res
        res["spawned"] = True
        pgid = os.getpgid(proc.pid)
        try:
            starttime = proc_stat(proc.pid)["starttime"]
        except (OSError, ValueError, IndexError):
            starttime = None
        res["process"] = {"pid": proc.pid, "pgid": pgid, "starttime_ticks": starttime, "start_monotonic_ns": t0}
        write_json_atomic(process_path, res["process"])
        deadline = t0 + int(cap_s * 1e9)
        reason = None
        next_health = t0 + int(health_interval_s * 1e9)
        while True:
            if proc.poll() is not None:
                break
            if host.stop_requested():
                reason = "signal"
            elif host.monotonic_ns() >= deadline:
                reason = "timeout"
            if reason is None and health is not None and host.monotonic_ns() >= next_health:
                next_health = host.monotonic_ns() + int(health_interval_s * 1e9)
                causes = health()
                if causes:
                    res["health_abort"] = {"t_mono_ns": host.monotonic_ns(), "causes": causes}
                    reason = "health"
            if reason:
                res["timed_out"] = reason == "timeout"
                res["aborted"] = reason == "signal"
                terminate_group(proc, pgid, grace_s, f"cell {reason}", res["events"], host)
                break
            host.sleep(POLL_S)
        res["end_monotonic_ns"] = host.monotonic_ns()
        rc = proc.returncode
        res["returncode"] = rc
        if rc is not None and rc < 0:
            res["signal"] = -rc
        # The CLI is gone; whatever is left in ITS group (a daemon it failed to reap) is terminated here.
        left = live_members(pgid)
        if left:
            res["leftover_after_exit"] = _brief(left)
            terminate_group(proc, pgid, grace_s, "leftover after CLI exit", res["events"], host)
        res["group_after_cleanup"] = _brief(live_members(pgid))
        return res
    finally:
        if proc is not None and pgid is not None and (proc.poll() is None or live_members(pgid)):
            terminate_group(proc, pgid, grace_s, "harness unwinding", res["events"], host)
        fo.close()
        fe.close()


def wait_idle(host, m, deadline_s):
    """Poll card vram_used until < 2 GiB. deadline_s == 0 is a single strict look."""
    path = m["target"]["vram_used_path"]
    t0 = host.monotonic_ns()
    samples = []
    while True:
        try:
            used = int(host.read_text(path).strip())
            err = None
        except (OSError, ValueError) as e:
            used, err = None, str(e)
        elapsed = (host.monotonic_ns() - t0) / 1e9
        samples.append({"t_s": elapsed, "used": used, **({"error": err} if err else {})})
        if used is not None and used < IDLE_MAX:
            return {"idle": True, "last_used": used, "samples": samples}
        if elapsed >= deadline_s:
            return {"idle": False, "last_used": used, "samples": samples}
        host.sleep(1.0)


# ----------------------------------------------------------------------------------------------------------------------------------
# bench JSON validation (shared by `run` and `judge`; the judge re-validates the RAW stdout bytes)
# ----------------------------------------------------------------------------------------------------------------------------------
def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise H2Error("median of nothing")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def pstdev(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _finite_pos(v):
    return type(v) in (int, float) and math.isfinite(v) and v > 0


def validate_bench(raw, m):
    """{ok, errors, samples, reported_median, durations_s, decode, ...}; ok False carries NO samples (nothing is zero-filled)."""
    errs = []
    res = {"ok": False, "errors": errs, "samples": None}
    try:
        doc = loads_strict(raw)
    except (ValueError, UnicodeDecodeError) as e:
        errs.append(f"stdout is not exactly one strict JSON document: {e}")
        return res
    if not isinstance(doc, dict):
        errs.append("stdout JSON is not an object")
        return res
    if doc.get("protocol") != PROTOCOL:
        errs.append(f"protocol {doc.get('protocol')!r} != {PROTOCOL}")
    if doc.get("model") != m["target"]["model_path"]:
        errs.append(f"model {doc.get('model')!r} != {m['target']['model_path']}")
    if doc.get("prefill_route") != PREFILL_ROUTE:
        errs.append(f"prefill_route {doc.get('prefill_route')!r} != {PREFILL_ROUTE!r}")
    if not (_is_int(doc.get("runs")) and doc["runs"] == RUNS):
        errs.append(f"runs {doc.get('runs')!r} != {RUNS}")
    loaded = doc.get("loaded")
    if not isinstance(loaded, dict):
        errs.append("loaded is not an object")
        loaded = {}
    evidence = []
    if "max_seq" in loaded:
        evidence.append(("loaded.max_seq", loaded["max_seq"]))
    ms = doc.get("max_seq")
    if isinstance(ms, dict):
        if "effective" in ms and ms["effective"] is not None:
            evidence.append(("max_seq.effective", ms["effective"]))
        if "requested" in ms and ms["requested"] is not None:
            evidence.append(("max_seq.requested", ms["requested"]))
        if "source" in ms and ms["source"] != "flag":
            errs.append(f"max_seq.source {ms['source']!r} != 'flag'")
    elif ms is not None:
        evidence.append(("max_seq", ms))
    for name, v in evidence:
        if not (_is_int(v) and v == MAX_SEQ):
            errs.append(f"{name} {v!r} != {MAX_SEQ}")
    if not evidence:
        errs.append("no max_seq evidence (loaded.max_seq / max_seq.effective) in the report")
    if doc.get("redline_pm4") not in (False, None):
        errs.append(f"redline_pm4 {doc.get('redline_pm4')!r} claims a redline run")
    if doc.get("spec_requested") not in (None, False):
        errs.append(f"spec_requested {doc.get('spec_requested')!r} claims speculation")
    for scope, d in (("report", doc), ("loaded", loaded)):
        for k, v in d.items():
            if any(s in k.lower() for s in ("prompt_cache", "prefix_cache", "cached_prefix", "resume_at")) and v not in (None, False, 0, "", [], {}):
                errs.append(f"{scope}.{k} claims a prompt/prefix cache ({v!r})")
    if not isinstance(doc.get("warnings"), list):
        errs.append("warnings is not a list")
    if doc.get("sustained") != []:
        errs.append("sustained must be an empty list")
    # prefill
    prefill = doc.get("prefill")
    samples = None
    reported = None
    if not (isinstance(prefill, list) and len(prefill) == 1 and isinstance(prefill[0], dict)):
        errs.append("prefill must be a one-element array")
    else:
        row = prefill[0]
        if not (_is_int(row.get("tokens")) and row["tokens"] == PP):
            errs.append(f"prefill tokens {row.get('tokens')!r} != {PP}")
        s = row.get("samples")
        if not (isinstance(s, list) and len(s) == SAMPLES_PER_PROCESS and all(_finite_pos(v) for v in s)):
            errs.append(f"prefill samples must be {SAMPLES_PER_PROCESS} finite positive numbers (got {s!r})")
        else:
            samples = [float(v) for v in s]
        stats = row.get("stats")
        if not isinstance(stats, dict) or not all(k in stats for k in ("median", "mean", "min", "max", "stdev")):
            errs.append("prefill stats must carry median/mean/min/max/stdev")
        elif samples is not None:
            reported = stats["median"]
            if not _finite_pos(reported) or reported != median(samples):
                errs.append(f"reported median {reported!r} != median of the samples {median(samples)!r}")
            if stats["min"] != min(samples) or stats["max"] != max(samples):
                errs.append("reported min/max differ from the samples")
            if not (_is_num(stats["mean"]) and math.isclose(stats["mean"], sum(samples) / len(samples), rel_tol=1e-9)):
                errs.append("reported mean differs from the samples")
    # auxiliary decode
    decode_info = None
    dec = doc.get("decode")
    if not (isinstance(dec, list) and len(dec) == 1 and isinstance(dec[0], dict)):
        errs.append("decode must be a one-element array (auxiliary --ctx 1 --tg 1 row)")
    else:
        d = dec[0]
        ds = d.get("samples")
        if not (_is_int(d.get("context")) and d["context"] == CTX and _is_int(d.get("tokens")) and d["tokens"] == TG):
            errs.append(f"decode row context/tokens {d.get('context')!r}/{d.get('tokens')!r} != {CTX}/{TG}")
        if not (isinstance(ds, list) and len(ds) == RUNS and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in ds)):
            errs.append("decode samples must be three finite non-negative numbers")
        else:
            decode_info = {"samples": [float(v) for v in ds], "median": median([float(v) for v in ds]), "all_positive": all(v > 0 for v in ds),
                           "role": "auxiliary, never a headline"}
    res.update({
        "protocol": doc.get("protocol"), "model": doc.get("model"), "runs": doc.get("runs"), "warnings": doc.get("warnings"),
        "max_seq_report": doc.get("max_seq"), "loaded_max_seq": loaded.get("max_seq"), "load_speculation": doc.get("load_speculation"),
        "kv_mode": doc.get("kv_mode"), "kv_backend": doc.get("kv_backend"), "topology": doc.get("topology"),
        "loaded_sha256": sha256_bytes(canon(loaded)) if isinstance(loaded, dict) else None, "gpu_sha256": sha256_bytes(canon(doc.get("gpu"))),
        "decode": decode_info, "reported_median": reported,
    })
    if not errs:
        res["ok"] = True
        res["samples"] = samples
        res["durations_s"] = [PP / v for v in samples]
    return res


def route_evidence(m, role, stderr_text):
    re_cfg = m.get("route_evidence")
    out = {"declared": re_cfg is not None, "failures": []}
    if re_cfg is None:
        out["note"] = ("no stderr pattern declared: the route proof is the typed config + successful load bound to the source identity; "
                       "N/tile are not observed and never invented")
        out.update({"matched": None, "n": None, "tile": None})
        return out
    if role == "candidate":
        mt = re.search(re_cfg["stderr_regex"], stderr_text, re.MULTILINE)
        out["matched"] = bool(mt)
        out["n"] = out["tile"] = None
        if mt:
            gd = mt.groupdict()
            for key in ("n", "tile"):
                v = gd.get(key)
                out[key] = int(v) if v is not None and v.isdigit() else None
            out["line"] = mt.group(0)[:300]
        elif re_cfg["required"]:
            out["failures"].append("ROUTE_EVIDENCE_MISSING")
    else:
        forbid = re_cfg.get("incumbent_forbidden_regex")
        out["matched"] = None
        out["n"] = out["tile"] = None
        if forbid:
            mt = re.search(forbid, stderr_text, re.MULTILINE)
            out["forbidden_matched"] = bool(mt)
            if mt:
                out["line"] = mt.group(0)[:300]
                out["failures"].append("ROUTE_CONTAMINATION")
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# watcher
# ----------------------------------------------------------------------------------------------------------------------------------
def watcher_identity(m):
    """What the parent's campaign symlink resolves to RIGHT NOW (stat follows it): recorded in every cell receipt, and a later change is fatal."""
    path = m["watcher"]["log_path"]
    st = os.stat(path)
    return {"log_path": path, "realpath": os.path.realpath(path), "device": st.st_dev, "inode": st.st_ino, "start_size_bytes": st.st_size}


def watcher_scan(m, offset, ident=None):
    """(new_offset, {fatal, matches, error}). A missing, shrunk or re-targeted (identity changed) log is an error, never silence."""
    path = m["watcher"]["log_path"]
    try:
        st = os.stat(path)
        size = st.st_size
        if ident is not None and (st.st_dev, st.st_ino) != (ident["device"], ident["inode"]):
            return offset, {"fatal": False, "matches": [], "error": f"watcher log identity changed (dev/ino {st.st_dev}/{st.st_ino} != "
                                                                       f"{ident['device']}/{ident['inode']}): the log is stale or re-targeted"}
        if size < offset:
            return offset, {"fatal": False, "matches": [], "error": f"watcher log shrank ({size} < {offset})"}
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(size - offset)
    except OSError as e:
        return offset, {"fatal": False, "matches": [], "error": str(e)}
    rx = re.compile(m["watcher"]["fatal_regex"])
    matches = [ln[:300] for ln in data.decode("utf-8", errors="replace").splitlines() if rx.search(ln)]
    return offset + len(data), {"fatal": bool(matches), "matches": matches[:MAX_LOG_MATCHES], "match_count": len(matches), "error": None}


# ----------------------------------------------------------------------------------------------------------------------------------
# input snapshot (stat-only recheck after every cell)
# ----------------------------------------------------------------------------------------------------------------------------------
def snapshot(m):
    snap = {"cached": {}, "artifacts": {}, "absent": {}, "seeds": {}}
    for e in m["cached_inputs"]:
        try:
            snap["cached"][e["path"]] = stat_tuple(os.stat(e["path"]))
        except OSError as ex:
            snap["cached"][e["path"]] = f"unreadable: {ex}"
    for e in m["artifacts"]:
        try:
            snap["artifacts"][e["path"]] = stat_tuple(os.lstat(e["path"]))
        except OSError as ex:
            snap["artifacts"][e["path"]] = f"unreadable: {ex}"
    for e in m.get("home_seed_files", []):
        try:
            snap["seeds"][e["source"]] = stat_tuple(os.lstat(e["source"]))
        except OSError as ex:
            snap["seeds"][e["source"]] = f"unreadable: {ex}"
    for p in m["target"]["absent_paths"]:
        snap["absent"][p] = os.path.lexists(p)
    snap["packs"] = {label: pack_snapshot(m["arms"][label]["pack_dir"]) for label in ("A", "B")}
    return snap


def pack_snapshot(pack_dir):
    """Names, modes and (directory) mtimes of the pack tree: any created/removed/chmod-ed entry changes it."""
    out = []
    try:
        for root, dirs, names in os.walk(pack_dir, followlinks=False):
            for n in sorted(dirs + names):
                st = os.lstat(os.path.join(root, n))
                out.append([os.path.relpath(os.path.join(root, n), pack_dir), st.st_mode, st.st_mtime_ns if statmod.S_ISDIR(st.st_mode) else None])
        st = os.lstat(pack_dir)
        out.append([".", st.st_mode, st.st_mtime_ns])
    except OSError as e:
        return f"unreadable: {e}"
    return sorted(out, key=lambda e: e[0])



# ----------------------------------------------------------------------------------------------------------------------------------
# cell execution
# ----------------------------------------------------------------------------------------------------------------------------------
class HoldCtx:
    """State of ONE single-process hold (one `run` invocation = one held command = one fresh official CLI process)."""

    def __init__(self, m, msha, hsha, host, out, pair, position, hold_start_ns, boot_id, preflight_report, previous_chain, first):
        self.m, self.msha, self.hsha, self.host, self.out, self.pair, self.position = m, msha, hsha, host, out, pair, position
        self.hold_start_ns, self.boot_id, self.preflight = hold_start_ns, boot_id, preflight_report
        self.previous_chain, self.first = list(previous_chain), first
        self.preflight_sha = None
        self.remaining_at_start_s = None
        self.watcher_ident = watcher_identity(m)
        self.watcher_offset = self.watcher_ident["start_size_bytes"]
        self.snap0 = snapshot(m)

    def remaining_s(self):
        used = (self.host.monotonic_ns() - self.hold_start_ns) / 1e9
        return self.m["budget"]["maximum_hold_seconds"] - used

    def need_s(self):
        b = self.m["budget"]
        return b["cell_cap_seconds"] + b["kill_grace_seconds"] + b["drain_seconds"] + b["finalize_seconds"]


def start_gate(ctx):
    """Reasons the cell cannot start now ([] = may start). Budget uses the FULL cap + grace + drain + finalize; the cap is never shrunk."""
    why = []
    rem, need = ctx.remaining_s(), ctx.need_s()
    if rem < need:
        why.append(f"BUDGET: {rem:.1f}s of hold remain, a full cell needs {need:.1f}s (cap+grace+drain+finalize)")
    dev = device_state(ctx.host, ctx.m)
    if not dev["ok"]:
        why.append("DEVICE: " + "; ".join(dev["problems"]))
    else:
        idle = wait_idle(ctx.host, ctx.m, 0)
        if not idle["idle"]:
            why.append(f"BUSY: card vram_used {idle['last_used']} is not < {IDLE_MAX} before spawn")
    if ctx.host.stop_requested():
        why.append("SIGNAL: stop requested")
    return why


def _file_info(path):
    return {"path": path, "size_bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def _pack_files(m, a):
    return {e["path"]: e["sha256"] for e in sorted(m["artifacts"], key=lambda x: x["path"]) if e["path"].startswith(a["pack_dir"] + os.sep)}


def _tree_stats(path):
    files = size = 0
    for root, _, names in os.walk(path, followlinks=False):
        for n in names:
            try:
                st = os.lstat(os.path.join(root, n))
            except OSError:
                continue
            files += 1
            size += st.st_size
    return {"files_after": files, "bytes_after": size}


def _prepare_cell_dir(ctx, cell_dir, arm):
    """Fresh private tree of this cell. Returns (home_info, hot_cache_info). The hot cache is the ONLY place a JIT may write."""
    m = ctx.m
    a = m["arms"][arm]
    hh = os.path.join(cell_dir, "home", ".hipfire")
    os.mkdir(cell_dir)
    os.makedirs(os.path.join(hh, "bin"))
    os.mkdir(os.path.join(cell_dir, "lockdir"))
    hot = os.path.join(cell_dir, "kcache")
    os.mkdir(hot)
    if not os.access(hot, os.W_OK):
        raise H2Error(f"private hot cache {hot} is not writable")
    cfg = config_bytes(m, arm)
    with open(os.path.join(hh, "config.toml"), "xb") as f:
        f.write(cfg)
        f.flush()
        os.fsync(f.fileno())
    os.symlink(os.path.realpath(a["daemon"]), os.path.join(hh, "bin", "daemon"))
    os.symlink(m["target"]["models_dir"], os.path.join(hh, "models"))
    seeds = []
    for e in m.get("home_seed_files", []):
        with open(e["source"], "rb") as f:
            data = f.read(MAX_SEED_BYTES + 1)
        if sha256_bytes(data) != e["sha256"] or len(data) != e["size_bytes"]:
            raise H2Error(f"seed file {e['source']} changed since preflight")
        dest = os.path.join(hh, e["dest"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "xb") as f:
            f.write(data)
        seeds.append({"dest": e["dest"], "sha256": e["sha256"], "size_bytes": e["size_bytes"]})
    home = {"path": os.path.join(hh, "config.toml"), "sha256": sha256_bytes(cfg), "bytes": len(cfg),
            "daemon_symlink": os.path.join(hh, "bin", "daemon"), "daemon_symlink_target": os.path.realpath(a["daemon"]),
            "models_symlink": os.path.join(hh, "models"), "models_symlink_target": m["target"]["models_dir"], "seeded": seeds}
    return home, {"path": hot, "private": True, "writable": True}


def _hold_block(ctx):
    """This hold's own clock and budget (never chained to any other hold)."""
    b = ctx.m["budget"]
    now = ctx.host.monotonic_ns()
    return {"boot_id": ctx.boot_id, "hold_start_monotonic_ns": ctx.hold_start_ns, "hold_end_monotonic_ns": now,
            "used_seconds_at_end": (now - ctx.hold_start_ns) / 1e9, "maximum_hold_seconds": b["maximum_hold_seconds"],
            "required_cell_seconds": ctx.need_s(), "required_single_hold_seconds": single_hold_min(b),
            "remaining_at_start_seconds": ctx.remaining_at_start_s, "clamp_seconds": HOLD_CLAMP_SECONDS, "budget": dict(b)}


def _cell_header(ctx, arm):
    """Identity/binding fields common to every cell receipt (the executed one and the exception one)."""
    m = ctx.m
    role = m["arms"][arm]["role"]
    argv = expected_argv(m, arm)
    cell_id = f"{arm}{ctx.pair}"
    return {
        "schema": SCHEMA_CELL, "manifest_sha256": ctx.msha, "harness_sha256": ctx.hsha, "label": m["label"],
        "pair": ctx.pair, "position": ctx.position, "arm": arm, "cell_id": cell_id, "role": role, "cell_dir_name": cell_id,
        "source": dict(m["identity"]["candidate" if role == "candidate" else "baseline"]),
        "argv": argv, "command_text": " ".join(shlex.quote(x) for x in argv),
        "previous_pairs": [{"pair": i, "receipt_sha256": s} for i, s in enumerate(ctx.previous_chain, 1)],
        "first_cell": ctx.first, "hold": _hold_block(ctx), "preflight_sha256": ctx.preflight_sha,
    }


def execute_cell(ctx, arm):
    """The ONE fresh CLI process of this hold. Returns the SEALED cell receipt (also written to <cell>/cell.json)."""
    m, host = ctx.m, ctx.host
    cell_id = f"{arm}{ctx.pair}"
    a = m["arms"][arm]
    role = a["role"]
    cell_dir = os.path.join(ctx.out, cell_id)
    prep, hot = _prepare_cell_dir(ctx, cell_dir, arm)
    env, unset, applied = build_env(host.environ(), cell_dir, m["target"])
    argv = expected_argv(m, arm)
    with open(os.path.join(cell_dir, "command.txt"), "x") as f:
        f.write(" ".join(shlex.quote(x) for x in argv) + "\n")
    baseline = wait_idle(host, m, 0)
    stdout_path = os.path.join(cell_dir, "stdout.json")
    stderr_path = os.path.join(cell_dir, "stderr.log")
    b = m["budget"]
    wat_acc = {"fatal": False, "matches": [], "match_count": 0, "error": None}

    def absorb(w):
        wat_acc["fatal"] = wat_acc["fatal"] or w["fatal"]
        wat_acc["match_count"] += w.get("match_count", 0)
        wat_acc["matches"].extend(w["matches"][:max(0, MAX_LOG_MATCHES - len(wat_acc["matches"]))])
        if w["error"] and not wat_acc["error"]:
            wat_acc["error"] = w["error"]

    def health():
        causes = []
        dev = device_state(host, m)
        if not dev["ok"]:
            causes.append({"status": "DEVICE_LOST", "problems": dev["problems"]})
        ctx.watcher_offset, w = watcher_scan(m, ctx.watcher_offset, ctx.watcher_ident)
        absorb(w)
        if w["fatal"]:
            causes.append({"status": "WATCHER_FATAL", "matches": w["matches"], "match_count": w.get("match_count", 0)})
        if w["error"]:
            causes.append({"status": "WATCHER_UNREADABLE", "error": w["error"]})
        return causes

    run = spawn_and_wait(argv, env, cell_dir, stdout_path, stderr_path, os.path.join(cell_dir, "process.json"), b["cell_cap_seconds"],
                         b["kill_grace_seconds"], host, health=health)
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
    drain = wait_idle(host, m, b["drain_seconds"]) if run["spawned"] else None
    dev_after = device_state(host, m)
    ctx.watcher_offset, w_last = watcher_scan(m, ctx.watcher_offset, ctx.watcher_ident)
    absorb(w_last)
    wat = wat_acc
    wat["identity"] = ctx.watcher_ident
    snap1 = snapshot(m)
    changed = []
    for sect in ("cached", "artifacts", "seeds", "absent", "packs"):
        if sect == "cached":
            diff = {p: cached_gate(v) for p, v in snap1[sect].items()} != {p: cached_gate(v) for p, v in ctx.snap0[sect].items()}
        else:
            diff = snap1[sect] != ctx.snap0[sect]
        if diff:
            changed.append(sect)
    # output
    with open(stdout_path, "rb") as f:
        stdout_raw = f.read()
    with open(stderr_path, "rb") as f:
        stderr_raw = f.read()
    bench = None
    route = route_evidence(m, role, stderr_raw.decode("utf-8", errors="replace"))
    if run["spawned"] and run["returncode"] == 0 and not run["timed_out"] and not run["aborted"] and not ha:
        bench = validate_bench(stdout_raw, m)
        if not bench["ok"]:
            failures.append("INVALID_OUTPUT")
        failures.extend(route["failures"])
    if not dev_after["ok"] and "DEVICE_LOST" not in failures:
        failures.append("DEVICE_LOST")
    if wat["error"] and "WATCHER_UNREADABLE" not in failures:
        failures.append("WATCHER_UNREADABLE")
    if wat["fatal"] and "WATCHER_FATAL" not in failures:
        failures.append("WATCHER_FATAL")
    if drain is not None and not drain["idle"]:
        failures.append("DRAIN_TIMEOUT")
    if changed:
        failures.append("INPUT_CHANGED")
    status = failures[0] if failures else STATUS_OK
    proc = run.get("process", {})
    elapsed = None
    if "end_monotonic_ns" in run and "start_monotonic_ns" in proc:
        elapsed = (run["end_monotonic_ns"] - proc["start_monotonic_ns"]) / 1e9
    rec = {
        **_cell_header(ctx, arm),
        "env": {"set": applied, "unset": unset, "fabric_job": env.get("FABRIC_JOB"), "fabric_cards": env.get("FABRIC_CARDS"),
                "non_default_d2_off": True, "arm_note": ("HIPFIRE_QWEN4_D2_DEFAULT=0 is the lab compatibility policy on BOTH arms (explicitly "
                                                         "NON-DEFAULT, no default-D2 qualification); actual D2 control/consumption is source-bound "
                                                         "(older branches may lack the selector): the env value alone never proves D2 is functionally disabled")},
        "home": prep,
        "cli": _file_info(a["cli"]), "daemon": _file_info(a["daemon"]),
        "pack_dir": a["pack_dir"], "pack_files": _pack_files(m, a),
        "pack_readonly": dict(ctx.preflight["pack_readonly"][arm]),
        "hot_cache": {**hot, **_tree_stats(hot["path"])},
        "cached_inputs_after": {p: v for p, v in snap1["cached"].items()},
        "integrity": {"changed_sections": changed, "cached_inputs_unchanged": "cached" not in changed,
                      "artifacts_unchanged": "artifacts" not in changed, "absent_paths_absent": not any(snap1["absent"].values()),
                      "pack_dirs_unchanged": "packs" not in changed},
        "process": {**proc, "returncode": run["returncode"], "signal": run["signal"], "timed_out": run["timed_out"], "aborted": run["aborted"],
                    "end_monotonic_ns": run.get("end_monotonic_ns"), "elapsed_seconds": elapsed, "spawn_error": run.get("spawn_error"),
                    "start_new_session": True, "close_fds": True},
        "lifecycle": {"events": run["events"], "leftover_after_cli_exit": run["leftover_after_exit"],
                      "group_after_cleanup": run.get("group_after_cleanup", [])},
        "baseline": {"idle": baseline["idle"], "vram_used": baseline["last_used"]},
        "drain": drain, "device_after": dev_after, "watcher": wat, "health_abort": ha, "route_evidence": route,
        "stdout": {"path": "stdout.json", "size_bytes": len(stdout_raw), "sha256": sha256_bytes(stdout_raw)},
        "stderr": {"path": "stderr.log", "size_bytes": len(stderr_raw), "sha256": sha256_bytes(stderr_raw)},
        "bench": bench, "failures": failures, "status": status, "measurement_valid": status == STATUS_OK,
    }
    rec = seal(rec)
    write_json_atomic(os.path.join(cell_dir, "cell.json"), rec)
    return rec


def _write_failure_record(ctx, arm, exc):
    """The harness itself failed after the OUT dir was created: close the hold with a sealed, never-valid receipt (manifest-derived bindings are
    marked observed=False; nothing is claimed about a process)."""
    m = ctx.m
    a = m["arms"][arm]
    art = {e["path"]: e for e in m["artifacts"]}
    env = ctx.host.environ()
    cell_dir = os.path.join(ctx.out, f"{arm}{ctx.pair}")
    rec = {
        **_cell_header(ctx, arm),
        "env": {"set": None, "unset": [], "fabric_job": env.get("FABRIC_JOB"), "fabric_cards": env.get("FABRIC_CARDS"), "non_default_d2_off": True},
        "home": {"sha256": sha256_bytes(config_bytes(m, arm)), "observed": False},
        "cli": {"path": a["cli"], "size_bytes": art[a["cli"]]["size_bytes"], "sha256": art[a["cli"]]["sha256"], "observed": False},
        "daemon": {"path": a["daemon"], "size_bytes": art[a["daemon"]]["size_bytes"], "sha256": art[a["daemon"]]["sha256"], "observed": False},
        "pack_dir": a["pack_dir"], "pack_files": _pack_files(m, a),
        "process": {}, "stdout": None, "stderr": None, "bench": None,
        "failures": ["HARNESS_EXCEPTION"], "status": "HARNESS_EXCEPTION", "measurement_valid": False,
        "exception": f"{type(exc).__name__}: {exc}",
    }
    rec = seal(rec)
    os.makedirs(cell_dir, exist_ok=True)
    write_json_atomic(os.path.join(cell_dir, "cell.json"), rec)
    return rec


# ----------------------------------------------------------------------------------------------------------------------------------
# pair loading (shared by `run --previous`, `assemble-pair` and `judge`)
# ----------------------------------------------------------------------------------------------------------------------------------
def load_pair_receipt(path, msha, hsha, errs):
    try:
        doc = loads_strict(read_small(path, 256 << 20))
    except (OSError, ValueError, H2Error) as e:
        errs.append(f"pair {path}: unreadable/not strict JSON: {e}")
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_PAIR:
        errs.append(f"pair {path}: schema != {SCHEMA_PAIR}")
        return None
    if not seal_ok(doc):
        errs.append(f"pair {path}: receipt_sha256 does not match its content")
        return None
    if doc.get("manifest_sha256") != msha:
        errs.append(f"pair {path}: manifest_sha256 differs from the supplied manifest")
    if doc.get("harness_sha256") != hsha:
        errs.append(f"pair {path}: harness_sha256 differs from this harness")
    if doc.get("closed") is not True:
        errs.append(f"pair {path}: not closed")
    return doc


# ----------------------------------------------------------------------------------------------------------------------------------
# run ONE cell = ONE hold = ONE fresh official CLI process
# ----------------------------------------------------------------------------------------------------------------------------------
def check_fabric(env, m):
    errs = []
    job, cards = env.get("FABRIC_JOB"), env.get("FABRIC_CARDS")
    if not job or not cards:
        errs.append("not inside a fabric job (FABRIC_JOB / FABRIC_CARDS unset)")
        return errs
    names = cards.split(",")
    if len(names) != 1:
        errs.append(f"FABRIC_CARDS={cards} names more than one card")
    elif names[0] not in (m["target"]["card"], m["target"]["uuid"]):
        errs.append(f"FABRIC_CARDS={cards} does not name the target card {m['target']['card']} ({m['target']['uuid']})")
    return errs


def run_cell(manifest_path, expected_manifest_sha256, pair, cell, out, previous=(), first_cell=None, first_cell_sha256=None, host=None):
    """Runs ONE cell (one hold, ONE fresh official CLI process). Raises Refusal BEFORE any mutation; otherwise returns the sealed cell receipt
    (written to OUT/<cell_id>/cell.json), which may be a failed one. Cell 2 additionally needs the first cell's path and its independently
    supplied file sha256."""
    host = host or Host()
    hold_start = host.monotonic_ns()            # the hold clock starts BEFORE preflight: preflight is part of this hold's budget
    if pair not in ORDER:
        raise Refusal([f"--pair must be 1, 2 or 3 (got {pair})"])
    if cell not in CELLS:
        raise Refusal([f"--cell must be 1 or 2 (got {cell})"])
    shape = []
    if cell == 1 and (first_cell is not None or first_cell_sha256 is not None):
        shape.append("--first-cell/--first-cell-sha256 belong to cell 2 only: cell 1 is the first process of its pair")
    if cell == 2 and (first_cell is None or first_cell_sha256 is None):
        shape.append("cell 2 REQUIRES --first-cell PATH and --first-cell-sha256 SHA (independently supplied) to bind cell 1")
    if cell == 2 and first_cell_sha256 is not None and not (isinstance(first_cell_sha256, str) and HEX64.match(first_cell_sha256)):
        shape.append("--first-cell-sha256 must be 64 lowercase hex")
    if shape:
        raise Refusal(shape)
    report, m = preflight(manifest_path, expected_manifest_sha256, host)
    if not report["ok"]:
        raise Refusal(report["errors"])
    msha, hsha = report["manifest_sha256"], report["harness_sha256"]
    arm = ORDER[pair][cell - 1]
    out = os.path.abspath(out)
    errs = []
    if os.path.lexists(out):
        errs.append(f"--out {out} already exists (it must be a FRESH directory)")
    if not os.path.isdir(os.path.dirname(out)):
        errs.append(f"parent of --out {out} does not exist")
    env = host.environ()
    errs.extend(check_fabric(env, m))
    ids = new_ids()
    if len(previous) != pair - 1:
        errs.append(f"pair {pair} needs exactly {pair - 1} --previous receipts (pairs 1..{pair - 1} in order), got {len(previous)}")
    else:
        verify_previous(m, msha, hsha, previous, ids, errs)
    first = None
    if cell == 2:
        first = verify_first_cell(m, msha, hsha, first_cell, first_cell_sha256, pair, ids, errs)
    boot = host.boot_id()
    if ids["boot"] is not None and ids["boot"] != boot:
        errs.append("this hold started in a different boot than the earlier holds")
    job = env.get("FABRIC_JOB")
    if job in ids["jobs"]:
        errs.append(f"FABRIC_JOB {job} already ran an earlier cell: every cell is its own hold (one fabric job per process)")
    if hold_start <= ids["last_hold_end"]:
        errs.append("this hold started before the previous hold ended: the six holds are sequential and independent")
    for prior in ids["outs"]:
        if out == prior or out.startswith(prior + os.sep):
            errs.append(f"--out {out} is an earlier cell's OUT dir (or inside it): every cell needs its own FRESH OUT")
    if errs:
        raise Refusal(errs)
    try:
        ctx = HoldCtx(m, msha, hsha, host, out, pair, cell - 1, hold_start, boot, report, ids["chain"], first)
    except OSError as e:
        raise Refusal([f"watcher log unreadable: {e}"])
    gate = start_gate(ctx)
    if gate:
        raise Refusal(gate)
    ctx.remaining_at_start_s = ctx.remaining_s()
    os.mkdir(out)
    write_json_atomic(os.path.join(out, "preflight.json"), report)
    ctx.preflight_sha = sha256_file(os.path.join(out, "preflight.json"))
    try:
        return execute_cell(ctx, arm)
    except BaseException as e:  # noqa: BLE001 -- the hold must always close with a sealed receipt
        return _write_failure_record(ctx, arm, e)


# ----------------------------------------------------------------------------------------------------------------------------------
# cell/pair receipt validation (shared by `run --previous`, `assemble-pair` and `judge`), pair assembly, judge
# ----------------------------------------------------------------------------------------------------------------------------------
def new_ids():
    """Cross-cell identity ledger shared by `run --previous`, `assemble-pair` and `judge`: nothing may repeat between the six holds."""
    return {"procs": set(), "homes": set(), "kcaches": set(), "jobs": [], "outs": [], "last_start": -1, "last_hold_end": -1, "boot": None, "chain": []}


def _expected_env_ok(env_rec, m, errs, where):
    applied = env_rec.get("set")
    if not isinstance(applied, dict):
        errs.append(f"{where}: env.set missing")
        return
    allowed = {"HOME", "HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
               "HIPFIRE_NO_REGISTRY_FETCH", D2_PIN[0]}
    if set(applied) != allowed:
        errs.append(f"{where}: applied env keys {sorted(applied)} != {sorted(allowed)}")
        return
    if applied[D2_PIN[0]] != D2_PIN[1] or applied["HIPFIRE_NO_REGISTRY_FETCH"] != "1":
        errs.append(f"{where}: D2_DEFAULT must be 0 and NO_REGISTRY_FETCH 1")
    for k, v in m["target"]["visible_devices_env"].items():
        if applied.get(k) != v:
            errs.append(f"{where}: {k} {applied.get(k)!r} != manifest {v!r}")
    if env_rec.get("non_default_d2_off") is not True:
        errs.append(f"{where}: non_default_d2_off must be true")


def _check_hold(m, cell, w, ids, errs):
    """The cell's own hold: its budget equals the manifest's, it fit, and it is strictly later than the previous hold."""
    h = cell.get("hold")
    if not isinstance(h, dict):
        errs.append(f"{w}: hold block missing")
        return
    b = m["budget"]
    need = b["cell_cap_seconds"] + b["kill_grace_seconds"] + b["drain_seconds"] + b["finalize_seconds"]
    if (h.get("budget") != b or h.get("maximum_hold_seconds") != b["maximum_hold_seconds"] or h.get("required_cell_seconds") != need
            or h.get("required_single_hold_seconds") != single_hold_min(b) or h.get("clamp_seconds") != HOLD_CLAMP_SECONDS):
        errs.append(f"{w}: hold budget differs from the manifest budget")
    hs, he = h.get("hold_start_monotonic_ns"), h.get("hold_end_monotonic_ns")
    if not (_is_int(hs) and _is_int(he) and he >= hs):
        errs.append(f"{w}: hold start/end monotonic ns missing or inconsistent")
    else:
        if hs <= ids["last_hold_end"]:
            errs.append(f"{w}: hold does not start strictly after the previous hold ended (every cell is its own sequential hold)")
        ids["last_hold_end"] = max(ids["last_hold_end"], he)
        if (he - hs) / 1e9 > b["maximum_hold_seconds"]:
            errs.append(f"{w}: hold used {(he - hs) / 1e9:.1f}s, more than maximum_hold_seconds {b['maximum_hold_seconds']}")
    rem = h.get("remaining_at_start_seconds")
    if not _is_num(rem) or rem < need:
        errs.append(f"{w}: the full single-cell budget did not fit before spawn (remaining {rem!r} < {need})")
    if not _is_str(h.get("boot_id")):
        errs.append(f"{w}: hold boot_id missing")
    elif ids["boot"] is None:
        ids["boot"] = h["boot_id"]
    elif h["boot_id"] != ids["boot"]:
        errs.append(f"{w}: hold boot_id differs from the earlier holds")


def check_cell_binding(m, msha, hsha, cell, pair_no, position, arm, cell_dir, errs, ids):
    """Binds a cell receipt to the manifest and RE-VALIDATES its raw stdout (read from `cell_dir`, the directory holding the cell's cell.json).
    Returns the validated sample dict or None."""
    cid = f"{arm}{pair_no}"
    w = f"cell {cid}"
    if not isinstance(cell, dict) or cell.get("schema") != SCHEMA_CELL:
        errs.append(f"{w}: schema != {SCHEMA_CELL}")
        return None
    if not seal_ok(cell):
        errs.append(f"{w}: receipt_sha256 does not match")
        return None
    a = m["arms"][arm]
    role = a["role"]
    exp = {"manifest_sha256": msha, "harness_sha256": hsha, "pair": pair_no, "position": position, "arm": arm, "cell_id": cid, "role": role,
           "argv": expected_argv(m, arm), "source": dict(m["identity"]["candidate" if role == "candidate" else "baseline"]),
           "cell_dir_name": cid, "previous_pairs": [{"pair": i, "receipt_sha256": s} for i, s in enumerate(ids["chain"], 1)]}
    for k, v in exp.items():
        if cell.get(k) != v:
            errs.append(f"{w}: {k} {cell.get(k)!r} != expected {v!r}")
    cell_dir = os.path.abspath(cell_dir)
    if os.path.basename(cell_dir) != cid:
        errs.append(f"{w}: the cell receipt sits in directory {os.path.basename(cell_dir)!r}, not {cid!r}")
    out_dir = os.path.dirname(cell_dir)
    if out_dir in ids["outs"]:
        errs.append(f"{w}: shares its OUT dir {out_dir} with another cell (every cell is its own hold with its own fresh OUT)")
    ids["outs"].append(out_dir)
    first = cell.get("first_cell")
    if position == 0:
        if first is not None:
            errs.append(f"{w}: the first cell of a pair must not carry a first_cell binding")
    elif not (isinstance(first, dict) and sorted(first) == ["cell_id", "fabric_job", "file_sha256", "out_dir", "path", "receipt_sha256"]
              and all(_is_str(v) for v in first.values())):
        errs.append(f"{w}: the second cell must carry a complete first_cell binding")
    home = cell.get("home") or {}
    if home.get("sha256") != sha256_bytes(config_bytes(m, arm)):
        errs.append(f"{w}: config.toml hash != the manifest's {role} config")
    for key, art_path in (("cli", a["cli"]), ("daemon", a["daemon"])):
        want = next(e for e in m["artifacts"] if e["path"] == art_path)
        got = cell.get(key) or {}
        if got.get("path") != art_path or got.get("sha256") != want["sha256"] or got.get("size_bytes") != want["size_bytes"]:
            errs.append(f"{w}: {key} binding differs from the manifest artifact")
    want_pack = {e["path"]: e["sha256"] for e in m["artifacts"] if e["path"].startswith(a["pack_dir"] + os.sep)}
    if cell.get("pack_files") != want_pack:
        errs.append(f"{w}: pack file hashes differ from the manifest artifacts")
    job = (cell.get("env") or {}).get("fabric_job")
    if not _is_str(job):
        errs.append(f"{w}: the actual FABRIC_JOB of this hold is not recorded")
    elif job in ids["jobs"]:
        errs.append(f"{w}: FABRIC_JOB {job} was already used by another cell (one distinct hold per cell)")
    else:
        ids["jobs"].append(job)
    _check_hold(m, cell, w, ids, errs)
    if cell.get("status") != STATUS_OK or cell.get("measurement_valid") is not True:
        return None       # a failed/incomplete cell is UNKNOWN; its raw numbers are never used
    if home.get("daemon_symlink_target") not in (a["daemon"], os.path.realpath(a["daemon"])) or home.get("models_symlink_target") != m["target"]["models_dir"]:
        errs.append(f"{w}: daemon/models symlink targets differ from the manifest")
    _expected_env_ok(cell.get("env") or {}, m, errs, w)
    want_cached = {e["path"]: {k: e[k] for k in CACHED_GATE_KEYS} for e in m["cached_inputs"]}
    after = cell.get("cached_inputs_after")
    if not isinstance(after, dict) or {p: cached_gate(v) for p, v in after.items()} != want_cached:
        errs.append(f"{w}: cached input stat after the cell != the manifest binding")
    integ = cell.get("integrity") or {}
    for k in ("cached_inputs_unchanged", "artifacts_unchanged", "absent_paths_absent", "pack_dirs_unchanged"):
        if integ.get(k) is not True:
            errs.append(f"{w}: integrity.{k} is not true")
    if (cell.get("pack_readonly") or {}).get("verified") is not True:
        errs.append(f"{w}: the installed pack dir was not verified read-only")
    hc = cell.get("hot_cache") or {}
    kc = hc.get("path")
    applied = (cell.get("env") or {}).get("set") or {}
    hp = home.get("path")
    if not kc or kc != applied.get("HIPFIRE_KERNEL_CACHE") or hc.get("private") is not True or hc.get("writable") is not True:
        errs.append(f"{w}: the hot cache is not the private writable HIPFIRE_KERNEL_CACHE of this cell")
    elif kc in ids["kcaches"]:
        errs.append(f"{w}: hot cache {kc} is shared with another cell")
    elif not hp or os.path.dirname(kc) != os.path.dirname(os.path.dirname(os.path.dirname(hp))):
        errs.append(f"{w}: the hot cache is not inside the cell's own private directory")
    ids["kcaches"].add(kc)
    proc = cell.get("process") or {}
    key = (proc.get("pid"), proc.get("starttime_ticks"))
    if key[0] is None or key in ids["procs"]:
        errs.append(f"{w}: process identity {key} missing or reused")
    ids["procs"].add(key)
    if not hp or hp in ids["homes"]:
        errs.append(f"{w}: HOME config path missing or shared with another cell")
    ids["homes"].add(hp)
    start = proc.get("start_monotonic_ns")
    hold = cell.get("hold") or {}
    if not _is_int(start) or start <= ids["last_start"]:
        errs.append(f"{w}: start time is not strictly after the previous cell")
    else:
        ids["last_start"] = start
        if _is_int(hold.get("hold_start_monotonic_ns")) and _is_int(hold.get("hold_end_monotonic_ns")) \
                and not hold["hold_start_monotonic_ns"] <= start <= hold["hold_end_monotonic_ns"]:
            errs.append(f"{w}: the process did not run inside its own hold")
    # raw stdout re-validation (never trust embedded numbers)
    path = os.path.join(cell_dir, "stdout.json")
    try:
        raw = read_small(path, 64 << 20)
    except (OSError, H2Error) as e:
        errs.append(f"{w}: raw stdout {path} unreadable: {e}")
        return None
    if sha256_bytes(raw) != (cell.get("stdout") or {}).get("sha256"):
        errs.append(f"{w}: stdout sha256 differs from the receipt")
        return None
    val = validate_bench(raw, m)
    if not val["ok"]:
        errs.append(f"{w}: raw stdout fails validation: {val['errors']}")
        return None
    emb = (cell.get("bench") or {}).get("samples")
    if emb != val["samples"]:
        errs.append(f"{w}: embedded samples differ from the raw stdout")
        return None
    if cell.get("route_evidence", {}).get("failures"):
        errs.append(f"{w}: route evidence failures {cell['route_evidence']['failures']}")
        return None
    return {"cell_id": cid, "arm": arm, "samples": val["samples"], "durations_s": val["durations_s"], "reported_median": val["reported_median"],
            "decode": val["decode"], "warnings": val["warnings"], "elapsed_seconds": proc.get("elapsed_seconds"),
            "route_evidence": cell.get("route_evidence"), "load_speculation": val["load_speculation"], "max_seq_report": val["max_seq_report"]}


def check_pair_receipt(m, msha, hsha, doc, path, pair_no, ids, errs):
    """Revalidates a pair receipt against the two cell FILES it binds (hash, content, raw stdout, chain, jobs, separate OUT dirs). Appends to
    `errs`. Returns {arm: process-dict} when BOTH cells are valid OK measurements, else None (incomplete, or errors appended)."""
    w = f"pair {pair_no}"
    pair_dir = os.path.dirname(os.path.abspath(path))
    order = ORDER[pair_no]
    exp = {"pair": pair_no, "order": [f"{a}{pair_no}" for a in order], "roles": {a: m["arms"][a]["role"] for a in ("A", "B")}, "label": m["label"],
           "previous_pairs": [{"pair": i, "receipt_sha256": s} for i, s in enumerate(ids["chain"], 1)]}
    for k, v in exp.items():
        if doc.get(k) != v:
            errs.append(f"{w}: {k} {doc.get(k)!r} != expected {v!r}")
    cells, files = doc.get("cells"), doc.get("cell_files")
    if not (isinstance(cells, list) and isinstance(files, list) and len(cells) == 2 and len(files) == 2):
        errs.append(f"{w}: needs exactly two cell receipts and two cell file bindings (one per single-process hold)")
        return None
    loaded = []
    for position, arm in enumerate(order):
        fe = files[position]
        if not (isinstance(fe, dict) and sorted(fe) == ["cell_id", "file_sha256", "receipt_sha256", "relpath"] and _is_str(fe.get("relpath"))
                and not os.path.isabs(fe["relpath"]) and isinstance(fe.get("file_sha256"), str) and isinstance(fe.get("receipt_sha256"), str)):
            errs.append(f"{w}: cell_files[{position}] is malformed")
            return None
        if fe["cell_id"] != f"{arm}{pair_no}":
            errs.append(f"{w}: cell_files[{position}] is {fe['cell_id']!r}, expected {arm}{pair_no} (cells are swapped or mixed)")
            return None
        cpath = os.path.normpath(os.path.join(pair_dir, fe["relpath"]))
        try:
            raw = read_small(cpath, 64 << 20)
        except (OSError, H2Error) as e:
            errs.append(f"{w}: cell file {cpath} unreadable: {e}")
            return None
        if sha256_bytes(raw) != fe["file_sha256"]:
            errs.append(f"{w}: cell file {cpath} sha256 differs from the pair's binding (the cell receipt was edited)")
            return None
        try:
            cdoc = loads_strict(raw)
        except ValueError as e:
            errs.append(f"{w}: cell file {cpath} is not strict JSON: {e}")
            return None
        if cdoc != cells[position]:
            errs.append(f"{w}: the embedded cell {position + 1} differs from its cell file")
            return None
        if not isinstance(cdoc, dict) or cdoc.get("receipt_sha256") != fe["receipt_sha256"]:
            errs.append(f"{w}: cell file {cpath} receipt_sha256 differs from the pair's binding")
            return None
        got = check_cell_binding(m, msha, hsha, cdoc, pair_no, position, arm, os.path.dirname(cpath), errs, ids)
        loaded.append((cdoc, fe, got))
    (c0, f0, g0), (c1, f1, g1) = loaded
    if f0["file_sha256"] == f1["file_sha256"] or c0.get("receipt_sha256") == c1.get("receipt_sha256"):
        errs.append(f"{w}: the two cells are the same receipt")
    first = c1.get("first_cell")
    if isinstance(first, dict):
        for k, want in (("file_sha256", f0["file_sha256"]), ("receipt_sha256", c0.get("receipt_sha256")), ("cell_id", c0.get("cell_id")),
                        ("fabric_job", (c0.get("env") or {}).get("fabric_job"))):
            if first.get(k) != want:
                errs.append(f"{w}: cell 2 first_cell.{k} does not bind this cell 1")
    if c0.get("status") != STATUS_OK:
        errs.append(f"{w}: cell 2 exists although cell 1 is {c0.get('status')}: no GPU cell may run after a failure")
    complete = g0 is not None and g1 is not None
    want_status = "PAIR_COMPLETE" if complete else "PAIR_INCOMPLETE"
    if doc.get("status") != want_status:
        errs.append(f"{w}: status {doc.get('status')!r} != {want_status!r} computed from the validated cells")
    return {order[0]: g0, order[1]: g1} if complete else None


def verify_previous(m, msha, hsha, paths, ids, errs):
    """The earlier pairs of the chain: FULL pair receipts, each complete and fully revalidated; extends ids['chain']."""
    for i, p in enumerate(paths, 1):
        d = load_pair_receipt(p, msha, hsha, errs)
        if d is None:
            continue
        if d.get("pair") != i:
            errs.append(f"--previous #{i} is pair {d.get('pair')}")
            continue
        if d.get("status") != "PAIR_COMPLETE":
            errs.append(f"--previous pair {i} is {d.get('status')}: a failed pair stops the chain, no automatic rerun")
            continue
        if check_pair_receipt(m, msha, hsha, d, p, i, ids, errs) is None:
            errs.append(f"--previous pair {i} does not revalidate as a complete pair")
            continue
        ids["chain"].append(d["receipt_sha256"])


def verify_first_cell(m, msha, hsha, path, supplied_sha256, pair_no, ids, errs):
    """Binds cell 2 to cell 1: the file must hash to the INDEPENDENTLY supplied sha256 and be an OK, fully revalidated cell 1 of this pair.
    Returns the first_cell binding dict (None + errors on any mismatch)."""
    n0 = len(errs)
    path = os.path.abspath(path)
    try:
        raw = read_small(path, 64 << 20)
    except (OSError, H2Error) as e:
        errs.append(f"--first-cell {path} unreadable: {e}")
        return None
    got = sha256_bytes(raw)
    if got != supplied_sha256:
        errs.append(f"--first-cell {path} sha256 {got} != the independently supplied --first-cell-sha256 {supplied_sha256}")
        return None
    try:
        doc = loads_strict(raw)
    except ValueError as e:
        errs.append(f"--first-cell {path} is not strict JSON: {e}")
        return None
    arm = ORDER[pair_no][0]
    sample = check_cell_binding(m, msha, hsha, doc, pair_no, 0, arm, os.path.dirname(path), errs, ids)
    if len(errs) > n0:
        return None
    if sample is None:
        errs.append(f"--first-cell {path} is {doc.get('status')}: a failed first cell leaves no valid binding, no further GPU cell may run")
        return None
    return {"path": path, "file_sha256": got, "receipt_sha256": doc["receipt_sha256"], "cell_id": doc["cell_id"],
            "fabric_job": doc["env"]["fabric_job"], "out_dir": os.path.dirname(os.path.dirname(path))}


def assemble_pair(manifest_path, expected_manifest_sha256, cell_paths, cell_sha256s, previous, out, host=None):
    """PURE CPU. Binds the two cell receipts of ONE logical pair into the pair receipt the judge consumes. Raises Refusal (nothing written) on any
    mismatch; otherwise writes OUT (never overwriting) and returns the sealed pair receipt (PAIR_COMPLETE or PAIR_INCOMPLETE)."""
    errs = []
    try:
        m, msha = load_manifest(manifest_path, expected_manifest_sha256)
    except H2Error as e:
        raise Refusal([str(e)])
    hsha = self_sha256()
    if hsha != m["harness"]["h2_official_sha256"]:
        errs.append("harness sha256 differs from manifest.harness.h2_official_sha256")
    if len(cell_paths) != 2 or len(cell_sha256s) != 2:
        raise Refusal(errs + ["assemble-pair needs exactly two --cells and two --cell-sha256 (cell 1 then cell 2): a missing cell is refused"])
    for s in cell_sha256s:
        if not (isinstance(s, str) and HEX64.match(s)):
            errs.append("--cell-sha256 values must be 64 lowercase hex")
    out = os.path.abspath(out)
    if os.path.lexists(out):
        errs.append(f"--out {out} already exists")
    if not os.path.isdir(os.path.dirname(out)):
        errs.append(f"parent of --out {out} does not exist")
    paths = [os.path.abspath(p) for p in cell_paths]
    if paths[0] == paths[1] or os.path.realpath(paths[0]) == os.path.realpath(paths[1]) or cell_sha256s[0] == cell_sha256s[1]:
        errs.append("duplicate cell: the same cell receipt was given twice")
    docs = []
    for i, p in enumerate(paths):
        try:
            raw = read_small(p, 64 << 20)
            got = sha256_bytes(raw)
            if got != cell_sha256s[i]:
                errs.append(f"cell {i + 1} file {p}: sha256 {got} != the independently supplied {cell_sha256s[i]}")
            docs.append(loads_strict(raw))
        except (OSError, ValueError, H2Error) as e:
            errs.append(f"cell {i + 1} file {p} unreadable/not strict JSON: {e}")
    if errs or len(docs) != 2 or not all(isinstance(d, dict) for d in docs):
        raise Refusal(errs or ["cell receipts must be JSON objects"])
    pair_no = docs[0].get("pair")
    if pair_no not in ORDER or docs[1].get("pair") != pair_no:
        raise Refusal([f"mixed pairs: the cells are pair {docs[0].get('pair')!r} and {docs[1].get('pair')!r}"])
    for i in (0, 1):
        if docs[i].get("position") != i:
            errs.append(f"cell file {i + 1} is position {docs[i].get('position')!r}, expected {i} (swapped or duplicate cells)")
    if errs:
        raise Refusal(errs)
    ids = new_ids()
    if len(previous) != pair_no - 1:
        raise Refusal([f"pair {pair_no} needs exactly {pair_no - 1} --previous receipts (pairs 1..{pair_no - 1} in order), got {len(previous)}"])
    verify_previous(m, msha, hsha, previous, ids, errs)
    if errs:
        raise Refusal(errs)
    pair_dir = os.path.dirname(out)
    order = ORDER[pair_no]
    bad = [d for d in docs if d.get("status") != STATUS_OK]
    failures = bad[0].get("failures") if bad else None
    doc = {
        "schema": SCHEMA_PAIR, "manifest_sha256": msha, "harness_sha256": hsha, "label": m["label"], "pair": pair_no,
        "order": [f"{a}{pair_no}" for a in order], "roles": {a: m["arms"][a]["role"] for a in ("A", "B")},
        "previous_pairs": [{"pair": i, "receipt_sha256": s} for i, s in enumerate(ids["chain"], 1)],
        "cells": docs,
        "cell_files": [{"cell_id": docs[i].get("cell_id"), "relpath": os.path.relpath(paths[i], pair_dir), "file_sha256": cell_sha256s[i],
                        "receipt_sha256": docs[i].get("receipt_sha256")} for i in (0, 1)],
        "closed": True,
        "stop_reason": (f"{bad[0].get('cell_id')} {bad[0].get('status')}: " + ",".join(str(f) for f in (failures if isinstance(failures, list) else []))
                        if bad else None),
        "status": "PAIR_COMPLETE" if not bad else "PAIR_INCOMPLETE",
        "fabric_jobs": [(d.get("env") or {}).get("fabric_job") for d in docs],
        "assembled_from": "two single-process holds (one cell receipt each); this receipt was produced by pure-CPU assemble-pair",
        "non_default": ("HIPFIRE_QWEN4_D2_DEFAULT=0 on both arms is the lab compatibility policy: explicitly NON-DEFAULT, no default-D2 qualification; "
                        "actual D2 control/consumption is source-bound, never inferred from the env value alone"),
    }
    check_pair_receipt(m, msha, hsha, doc, out, pair_no, ids, errs)
    if errs:
        raise Refusal(errs)
    doc = seal(doc)
    write_json_atomic(out, doc)
    return doc


def arm_summary(per_process):
    flat = [s for p in per_process for s in p["samples"]]
    summary = {"processes": [{"cell_id": p["cell_id"], "samples": p["samples"], "durations_s": p["durations_s"], "reported_median": p["reported_median"],
                              "decode": p["decode"], "elapsed_seconds": p["elapsed_seconds"], "warnings": p["warnings"]} for p in per_process],
               "samples_flat": flat, "sample_count": len(flat)}
    if flat:
        summary.update({"median_all_samples": median(flat), "min": min(flat), "max": max(flat), "pstdev": pstdev(flat),
                        "durations_s_flat": [PP / v for v in flat],
                        "median_rule": "median over ALL flattened samples; never a median of per-process medians"})
    return summary


def early_bounds(cand_known, inc_known):
    """Order-statistic proof after two pairs (6 known per arm, 3 unknown positive finite values each, final median = 5th smallest of 9)."""
    c, i = sorted(cand_known), sorted(inc_known)
    if len(c) != 6 or len(i) != 6:
        raise H2Error("early bound needs exactly 6 known samples per arm")
    cand_upper, cand_lower = c[4], c[1]
    inc_upper, inc_lower = i[4], i[1]
    gate1 = Fraction(cand_upper) < TARGET_TOK_S
    gate2 = Fraction(cand_upper) * RATIO_DEN < Fraction(inc_lower) * RATIO_NUM
    return {
        "candidate_known_sorted": c, "incumbent_known_sorted": i,
        "candidate_final_median_max": cand_upper, "candidate_final_median_min": cand_lower,
        "incumbent_final_median_max": inc_upper, "incumbent_final_median_min": inc_lower,
        "cannot_reach_2000": gate1, "cannot_reach_1p10_x_incumbent": gate2,
        "fails_early": bool(gate1 or gate2),
        "proof": ("final median of 9 = 5th smallest. With 6 known sorted k1..k6 and 3 unknown values, the 5th smallest is at most k5 (all unknowns "
                  "larger) and at least k2 (all unknowns smaller). Candidate max possible median = candidate k5 = %r; incumbent min possible median = "
                  "incumbent k2 = %r. 2000 gate unreachable: %s. 1.10 gate unreachable (candidate k5 < 1.10 x incumbent k2): %s."
                  % (cand_upper, inc_lower, gate1, gate2)),
    }


def judge(manifest_path, expected_manifest_sha256, pair_paths, host=None):
    host = host or Host()
    result = {"schema": SCHEMA_RESULT, "verdict": None, "errors": [], "non_default": "HIPFIRE_QWEN4_D2_DEFAULT=0 on both arms is the lab compatibility policy: "
              "explicitly NON-DEFAULT, no default-D2 qualification; actual D2 control/consumption is source-bound, never inferred from the env value alone; "
              "decode is the minimal auxiliary --ctx 1 --tg 1 row", "gates": {
                  "candidate_min_tok_s": TARGET_TOK_S, "min_ratio_over_incumbent": "1.10", "informational_floor_tok_s": FLOOR_TOK_S}}
    errs = result["errors"]
    result["expected_manifest_sha256"] = expected_manifest_sha256
    try:
        m, msha = load_manifest(manifest_path, expected_manifest_sha256)
    except H2Error as e:
        errs.append(str(e))
        result["verdict"] = "UNKNOWN_INVALID"
        return seal(result)
    hsha = self_sha256()
    result.update({"manifest_sha256": msha, "harness_sha256": hsha, "label": m["label"]})
    if hsha != m["harness"]["h2_official_sha256"]:
        errs.append("harness sha256 differs from manifest.harness.h2_official_sha256")
    if not (1 <= len(pair_paths) <= 3):
        errs.append(f"--pairs needs 1..3 closed pair receipts, got {len(pair_paths)}")
    ids = new_ids()
    cand_label, inc_label = arm_label_for_role(m, "candidate"), arm_label_for_role(m, "incumbent")
    complete = []              # [(pair_no, {arm: process-dict})]
    incomplete_notes = []
    expect_no = 1
    stopped = False
    for path in pair_paths:
        doc = load_pair_receipt(path, msha, hsha, errs)
        if doc is None:
            stopped = True
            continue
        if stopped:
            errs.append(f"pair receipt {path} follows an incomplete pair (no pair may run after a failed one)")
            continue
        if doc.get("pair") != expect_no:
            errs.append(f"pair receipts must be unique and ordered 1,2,3: got pair {doc.get('pair')} where {expect_no} was expected")
            continue
        procs = check_pair_receipt(m, msha, hsha, doc, path, expect_no, ids, errs)
        if procs is not None:
            complete.append((expect_no, procs))
            ids["chain"].append(doc["receipt_sha256"])
        else:
            incomplete_notes.append({"pair": expect_no, "status": doc.get("status"), "stop_reason": doc.get("stop_reason")})
            stopped = True
        expect_no += 1
    result["pairs_complete"] = [p for p, _ in complete]
    result["pairs_incomplete"] = incomplete_notes
    result["fabric_jobs"] = list(ids["jobs"])
    if errs:
        result["verdict"] = "UNKNOWN_INVALID"
        result["note"] = "binding/validation errors: nothing from these receipts is trusted for a verdict"
        return seal(result)
    cand_procs = [procs[cand_label] for _, procs in complete]
    inc_procs = [procs[inc_label] for _, procs in complete]
    # global order is implied by the per-pair ORDER; keep it explicit in the result
    result["global_order_executed"] = [f"{arm}{n}" for n, _ in complete for arm in ORDER[n]]
    result["candidate"] = arm_summary(cand_procs)
    result["incumbent"] = arm_summary(inc_procs)
    result["candidate_arm"], result["incumbent_arm"] = cand_label, inc_label
    n_done = len(complete)
    if n_done == PROCESSES_PER_ARM:
        c_med = result["candidate"]["median_all_samples"]
        i_med = result["incumbent"]["median_all_samples"]
        g_target = Fraction(c_med) >= TARGET_TOK_S
        g_ratio = Fraction(c_med) * RATIO_DEN >= Fraction(i_med) * RATIO_NUM
        result["all9"] = {"candidate_median": c_med, "incumbent_median": i_med, "ratio": c_med / i_med,
                          "ratio_exact": f"{Fraction(c_med) / Fraction(i_med)}", "gate_2000_tok_s": bool(g_target),
                          "gate_1p10_over_incumbent": bool(g_ratio), "informational_1400_tok_s": bool(Fraction(c_med) >= FLOOR_TOK_S),
                          "actual_holds": len(ids["jobs"])}
        result["verdict"] = "PASS" if (g_target and g_ratio) else "FAIL"
        if result["verdict"] == "FAIL":
            result["failed_gates"] = [n for n, ok in (("gate_2000_tok_s", g_target), ("gate_1p10_over_incumbent", g_ratio)) if not ok]
        return seal(result)
    result["all9"] = None
    if n_done == 2:
        bounds = early_bounds(result["candidate"]["samples_flat"], result["incumbent"]["samples_flat"])
        result["early_bounds"] = bounds
        if bounds["fails_early"]:
            result["verdict"] = "FAIL_EARLY_ORDER_STATISTIC"
            result["note"] = "remaining three processes per arm cannot rescue the gate; the third pair MUST NOT run. No all-nine median exists."
            return seal(result)
    result["verdict"] = "UNKNOWN_INCOMPLETE"
    result["note"] = (f"{n_done} complete pair(s): fewer than three fresh processes per arm is never a PASS and never zero-filled"
                      + ("" if n_done == 2 else "; early failure needs six known samples per arm"))
    return seal(result)


# ----------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------
def _emit(doc):
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def cmd_preflight(args, host):
    report, _ = preflight(args.manifest, args.expected_manifest_sha256, host)
    if args.report:
        write_json_atomic(args.report, report)
    _emit(report)
    return 0 if report["ok"] else 2


def cmd_run(args, host):
    old = {}
    handler = lambda signum, frame: STOP.set()  # noqa: E731
    STOP.clear()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old[sig] = signal.signal(sig, handler)
    try:
        try:
            rec = run_cell(args.manifest, args.expected_manifest_sha256, args.pair, args.cell, args.out, args.previous or (), args.first_cell,
                           args.first_cell_sha256, host)
        except Refusal as e:
            _emit({"status": "REFUSED", "errors": e.errors, "mutated": False})
            return 2
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
    cell_json = os.path.join(os.path.abspath(args.out), rec["cell_id"], "cell.json")
    _emit({"status": "CELL_OK" if rec["status"] == STATUS_OK else "CELL_FAILED", "cell_status": rec["status"], "failures": rec["failures"],
           "pair": rec["pair"], "cell": args.cell, "cell_id": rec["cell_id"], "fabric_job": rec["env"]["fabric_job"],
           "receipt_sha256": rec["receipt_sha256"], "cell_json": cell_json, "cell_file_sha256": sha256_file(cell_json)})
    if "ABORTED_SIGNAL" in rec["failures"]:
        return 4
    return 0 if rec["status"] == STATUS_OK else 3


def cmd_assemble(args, host):
    try:
        doc = assemble_pair(args.manifest, args.expected_manifest_sha256, args.cells, args.cell_sha256, args.previous or (), args.out, host)
    except Refusal as e:
        _emit({"status": "REFUSED", "errors": e.errors, "mutated": False})
        return 2
    _emit({"status": doc["status"], "pair": doc["pair"], "stop_reason": doc["stop_reason"], "receipt_sha256": doc["receipt_sha256"],
           "pair_json": os.path.abspath(args.out), "pair_file_sha256": sha256_file(args.out), "fabric_jobs": doc["fabric_jobs"]})
    return 0 if doc["status"] == "PAIR_COMPLETE" else 3


def cmd_judge(args, host):
    result = judge(args.manifest, args.expected_manifest_sha256, args.pairs, host)
    write_json_atomic(args.out, result)
    _emit({"verdict": result["verdict"], "result": os.path.abspath(args.out), "errors": result["errors"]})
    v = result["verdict"]
    if v == "PASS":
        return 0
    return 1 if v in ("FAIL", "FAIL_EARLY_ORDER_STATISTIC") else 3


def _sha256_arg(text):
    if not HEX64.match(text):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return text


def main(argv=None, host=None):
    p = argparse.ArgumentParser(description="official H2 pp16384 harness (six single-process holds)")
    sub = p.add_subparsers(dest="cmd", required=True)
    expected = dict(required=True, type=_sha256_arg, metavar="M_SHA",
                    help="sha256 of the RAW manifest bytes, supplied independently of the manifest (REQUIRED, checked before anything else)")
    a = sub.add_parser("preflight")
    a.add_argument("--manifest", required=True)
    a.add_argument("--expected-manifest-sha256", **expected)
    a.add_argument("--report")
    b = sub.add_parser("run", help="ONE cell = ONE hold = ONE fresh official CLI process")
    b.add_argument("--manifest", required=True)
    b.add_argument("--expected-manifest-sha256", **expected)
    b.add_argument("--pair", required=True, type=int, choices=(1, 2, 3))
    b.add_argument("--cell", required=True, type=int, choices=CELLS)
    b.add_argument("--out", required=True)
    b.add_argument("--previous", nargs="*", default=[])
    b.add_argument("--first-cell", help="cell 2 only: path of cell 1's cell.json")
    b.add_argument("--first-cell-sha256", type=_sha256_arg, help="cell 2 only: sha256 of cell 1's cell.json, supplied independently")
    d = sub.add_parser("assemble-pair", help="pure CPU: bind the two cell receipts of one pair")
    d.add_argument("--manifest", required=True)
    d.add_argument("--expected-manifest-sha256", **expected)
    d.add_argument("--cells", required=True, nargs=2, metavar=("CELL1", "CELL2"))
    d.add_argument("--cell-sha256", required=True, nargs=2, type=_sha256_arg, metavar=("S1", "S2"))
    d.add_argument("--previous", nargs="*", default=[])
    d.add_argument("--out", required=True)
    c = sub.add_parser("judge")
    c.add_argument("--manifest", required=True)
    c.add_argument("--expected-manifest-sha256", **expected)
    c.add_argument("--pairs", required=True, nargs="+")
    c.add_argument("--out", required=True)
    args = p.parse_args(argv)
    host = host or Host()
    return {"preflight": cmd_preflight, "run": cmd_run, "assemble-pair": cmd_assemble, "judge": cmd_judge}[args.cmd](args, host)


if __name__ == "__main__":
    sys.exit(main())
