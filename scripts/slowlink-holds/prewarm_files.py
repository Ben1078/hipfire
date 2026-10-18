#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prewarm_files.py --inputs PLAN.json --out REPORT.json

CPU-only page-cache warmer, run OUTSIDE a GPU hold (the parent invokes it as `nice -n19 ionice -c3 python3 prewarm_files.py ...`; the low
priority comes from that invocation, nothing here changes scheduling). It streams every file of the plan once through one reusable buffer
and discards the bytes: no hashing of payload, no copy, no mlock, no fadvise, no GPU, no temp file. The cache stays reclaimable and is not
guaranteed resident afterwards; the report is data, not a residency proof.

PLAN.json schema `s1.prewarm_inputs.v1`: {"schema": ..., "files": [{"role", "path", "size_bytes", "inode", "mtime_ns"(exact decimal STRING),
optional "sha256"(cached, informational, never recomputed)}, ...], ...any parent provenance...}. The parent resolves the actual file set;
this helper guesses nothing. EVERY listed file is required: a missing file, a declared (size, inode, mtime_ns) that does not match the opened
file, a short read, or a file that changed while it was read fails the run (exit 1) and the report says so (status "failed", the failing
entry, files already read, files not attempted). Success is only status "complete" with every file read in full. Symlinks are followed;
identity is taken from the OPENED descriptor (fstat), and re-checked on it after the read plus against the path (replacement race).

Exit codes: 0 complete; 1 file failure; 2 bad plan/arguments; 130 interrupted. A report is written (atomically) at start as status
"incomplete" so a killed run can never leave a stale success."""
import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from datetime import datetime, timezone

SCHEMA_IN = "s1.prewarm_inputs.v1"
SCHEMA_OUT = "s1.prewarm_report.v1"
BUF_BYTES = 4 << 20  # one reusable buffer
MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Active(file)", "Inactive(file)", "Mlocked")
DECIMAL = re.compile(r"0|[1-9][0-9]*")
HEX64 = re.compile(r"[0-9a-f]{64}")


class PlanError(Exception):
    pass


class WarmError(Exception):
    def __init__(self, kind, message, record):
        super().__init__(message)
        self.kind = kind
        self.record = record


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def write_atomic(path, obj):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def meminfo():
    """Selected /proc/meminfo fields in kB; an unreadable file is recorded, never hidden."""
    try:
        out = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in MEMINFO_KEYS:
                    out[key + "_kb"] = int(rest.split()[0])
        return out
    except (OSError, ValueError, IndexError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def current_nice():
    try:
        return os.nice(0)
    except (OSError, AttributeError):
        return None


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def int_field(entry, key, where):
    v = entry.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise PlanError(f"{where}: {key} must be a non-negative JSON integer, got {v!r}")
    return v


def parse_entry(entry, i):
    where = f"files[{i}]"
    if not isinstance(entry, dict):
        raise PlanError(f"{where}: not an object")
    for key in ("role", "path"):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise PlanError(f"{where}: {key} must be a non-empty string")
    mtime = entry.get("mtime_ns")
    # An exact decimal STRING: a JSON number would already have been through a float or lost its text form.
    if not isinstance(mtime, str) or not DECIMAL.fullmatch(mtime):
        raise PlanError(f"{where}: mtime_ns must be an exact canonical decimal string (ASCII digits, no sign/space/leading zero), got {mtime!r}")
    if "sha256" in entry and (not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"])):
        raise PlanError(f"{where}: sha256, if present, must be 64 lowercase hex digits")
    return {"role": entry["role"], "path": entry["path"], "size_bytes": int_field(entry, "size_bytes", where),
            "inode": int_field(entry, "inode", where), "mtime_ns": mtime,
            "declared": dict(entry)}


def load_plan(path):
    """-> (plan_meta, entries). plan_meta keeps every top-level key except `files` verbatim, plus the plan file's own sha256."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise PlanError(f"cannot read inputs {path}: {e}")
    try:
        plan = json.loads(raw)
    except ValueError as e:
        raise PlanError(f"inputs {path} is not JSON: {e}")
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA_IN:
        raise PlanError(f"inputs schema must be {SCHEMA_IN!r}")
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise PlanError("files must be a non-empty list")
    entries = [parse_entry(e, i) for i, e in enumerate(files)]
    roles = [e["role"] for e in entries]
    dup = sorted({r for r in roles if roles.count(r) > 1})
    if dup:
        raise PlanError(f"duplicate roles: {dup}")
    meta = {k: v for k, v in plan.items() if k != "files"}
    meta["inputs_path"] = os.path.abspath(path)
    meta["inputs_sha256"] = sha_bytes(raw)
    return meta, entries


def identity(st):
    return {"size_bytes": st.st_size, "inode": st.st_ino, "mtime_ns": str(st.st_mtime_ns)}


def warm_file(entry, buf):
    """Read one file fully. Returns the receipt record; raises WarmError (with the partial record) on any violation."""
    rec = {"role": entry["role"], "path": entry["path"], "canonical_path": None, "status": "failed",
           "declared": entry["declared"], "actual_before": None, "actual_after": None, "bytes_read": 0, "elapsed_s": None}
    t0 = time.perf_counter_ns()

    def fail(kind, message):
        rec["elapsed_s"] = (time.perf_counter_ns() - t0) / 1e9
        rec["failure"] = {"kind": kind, "message": message}
        raise WarmError(kind, f"{entry['role']} ({entry['path']}): {message}", rec)

    try:
        rec["canonical_path"] = os.path.realpath(entry["path"], strict=True)
        fd = os.open(entry["path"], os.O_RDONLY)
    except OSError as e:
        fail("missing_or_unopenable", f"{type(e).__name__}: {e}")
    try:
        st = os.fstat(fd)
        rec["actual_before"] = identity(st)
        if not stat.S_ISREG(st.st_mode):
            fail("not_regular_file", f"mode {st.st_mode:o} is not a regular file")
        want = {"size_bytes": entry["size_bytes"], "inode": entry["inode"], "mtime_ns": entry["mtime_ns"]}
        if rec["actual_before"] != want:
            fail("identity_mismatch", f"declared {want} but opened file is {rec['actual_before']}")
        view = memoryview(buf)
        total = 0
        try:
            while True:
                n = os.readv(fd, [view])
                if n == 0:
                    break
                total += n
        except OSError as e:
            rec["bytes_read"] = total
            fail("read_error", f"after {total} bytes: {type(e).__name__}: {e}")
        rec["bytes_read"] = total
        st2 = os.fstat(fd)
        rec["actual_after"] = identity(st2)
        if total != entry["size_bytes"]:
            fail("short_read" if total < entry["size_bytes"] else "grew_during_read",
                 f"read {total} bytes, declared {entry['size_bytes']}")
        if rec["actual_after"] != rec["actual_before"]:
            fail("changed_during_read", f"identity {rec['actual_before']} became {rec['actual_after']}")
        try:
            sp = os.stat(entry["path"])
        except OSError as e:
            fail("path_gone_after_read", f"{type(e).__name__}: {e}")
        if (sp.st_dev, sp.st_ino) != (st2.st_dev, st2.st_ino):
            fail("path_replaced_during_read", "path no longer names the file that was read")
    finally:
        os.close(fd)
    rec["status"] = "complete"
    rec["elapsed_s"] = (time.perf_counter_ns() - t0) / 1e9
    return rec


def run(plan_path, out_path):
    started = utc_now()
    t0 = time.perf_counter_ns()
    report = {"schema": SCHEMA_OUT, "status": "incomplete", "started_utc": started, "finished_utc": None, "wall_s": None,
              "nice": current_nice(), "buffer_bytes": BUF_BYTES, "input_plan": None, "files": [], "total_bytes_read": 0,
              "mem_before": meminfo(), "mem_after": None, "failure": None}
    write_atomic(out_path, report)

    def finish(status, code, failure=None):
        report["status"] = status
        report["failure"] = failure
        report["finished_utc"] = utc_now()
        report["wall_s"] = (time.perf_counter_ns() - t0) / 1e9
        report["mem_after"] = meminfo()
        report["total_bytes_read"] = sum(f["bytes_read"] for f in report["files"])
        write_atomic(out_path, report)
        return code

    try:
        meta, entries = load_plan(plan_path)
    except PlanError as e:
        return finish("failed", 2, {"kind": "bad_plan", "message": str(e)})
    report["input_plan"] = meta
    report["files"] = [{"role": e["role"], "path": e["path"], "status": "not_attempted", "declared": e["declared"], "bytes_read": 0}
                       for e in entries]
    buf = bytearray(BUF_BYTES)
    try:
        for i, entry in enumerate(entries):
            report["files"][i] = warm_file(entry, buf)
    except WarmError as e:
        report["files"][i] = e.record
        return finish("failed", 1, {"kind": e.kind, "role": entry["role"], "message": str(e)})
    except KeyboardInterrupt:
        report["files"][i]["status"] = "interrupted"
        return finish("failed", 130, {"kind": "interrupted", "role": entries[i]["role"], "message": "KeyboardInterrupt"})
    return finish("complete", 0)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv[1:])
    code = run(args.inputs, args.out)
    print(f"prewarm_files: {'complete' if code == 0 else 'FAILED'} (exit {code}); report {args.out}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
