#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""collect_evidence.py CELL_DIR NAME [RC [DRAINED]]

CPU-only FORENSIC index of one finished cell directory (run by hold-lib.sh cell() after the command, the drain and the sampler stop; also
usable by hand on an old run directory). Writes CELL_DIR/evidence.json (schema s1.cell_evidence.v1) atomically. It reads files and never
changes or removes one: the raw stdout/stderr/rc/command, the host sampler mem.csv, the external 20 ms VRAM sampler files, the meminfo
snapshots, status.txt, the gate report and the gate's durable progress sidecar stay exactly as the cell left them, on a pass, a failure,
a timeout (rc 124, or 137 after the kill grace) and a load failure alike.

Gate progress sidecar (qwen4_slowlink_gate):
  path        <--output>.progress.jsonl next to the gate report, i.e. CELL_DIR/gate.json.progress.jsonl for the hold cells
  schema      one JSON object per line, each {schema: "hipfire.qwen4.slowlink_gate.progress.v1", run_id, seq (contiguous from 0 per run),
              event, utc, unix_ms, mono_ns, ...fields}; events include run_start, plan, memory (pre_load/post_load/end points, plus
              load_failed / run_failed when a failure leaves the GPU owned), case_start, repeat_start, repeat_end (raw receipts, digests,
              wall time, cache occupancy), check rows, abort/load_error (error text) and final (report digest). Each line is written with
              sync_data when observed, so a SIGTERM, timeout or crash keeps it.
  readability a complete line that is not a valid record is listed in corrupt_lines; an unterminated last line is torn_tail; both are
              reported, neither is interpreted. readable is true only when the file opens and holds at least one valid record.
The sidecar is EVIDENCE ONLY. This script reports its summary under forensic_only/promotable false with a status that is `fail` when a
failed check row was durably recorded and `incomplete` otherwise: it NEVER says pass, and no verdict anywhere reads it. A cell passes only
from its process exit status; the Hold 1 headroom pass additionally requires a COMPLETED gate report (check_headroom.py) and the whole-run
external samples. A partial sidecar can never stand in for either.

Load phase timing (CELL_DIR/load_phase.json, schema s1.load_phase.v1, NON-GATING data, written atomically next to evidence.json): the elapsed
seconds between the SAME run's monotonic load_start and load_end progress records when both are present and ordered. A run with no load_end
(timeout / crash during the load: right-censored), no marker at all, a marker with a bad mono_ns/seq or duplicated, an end before its start, or
a start and end from different run_ids is reported with an explicit status and reason and elapsed_s = null: never 0, never a fabricated
completion. Statuses: complete, censored, missing, malformed, reversed, cross_run, no_sidecar, unreadable. Like evidence.json it is forensic
only: no cell status, check, headroom verdict or hold summary reads it, and the raw sidecar/gate records are never changed."""
import hashlib
import json
import os
import sys

SCHEMA = "s1.cell_evidence.v1"
PROGRESS_SCHEMA = "hipfire.qwen4.slowlink_gate.progress.v1"
PROGRESS_SUFFIX = ".progress.jsonl"
REPORT_SCHEMA = "hipfire.qwen4.slowlink_gate.v1"
LOAD_PHASE_FILE = "load_phase.json"
LOAD_PHASE_SCHEMA = "s1.load_phase.v1"
COMPLETED_STATUSES = ("pass", "fail")
TIMEOUT_RCS = (124, 137)  # timeout(1): 124 cap expired, 137 killed after the grace
SKIP_DIRS = ("home", "lockdir", "kcache")  # per-cell scratch (HOME, lock dir, kernel cache): listed, never hashed
RAW_FILES = ("command.txt", "stdout.txt", "stderr.txt", "rc.txt", "status.txt", "spawn-env.json", "baseline.json", "meminfo-before.txt",
             "meminfo-after.txt", "mem.csv", "vram.csv", "vram-events.jsonl", "vram-summary.json", "vram.ready", "vram-sampler.log")
PASS_POLICY = ("forensic only: neither this file nor the progress sidecar is read by any pass/fail decision; a cell passes from its process "
               "exit status and the hold's required checks, and the headroom verdict needs a completed gate report plus the whole-run external samples")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path):
    return {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def valid_mono(rec):
    v = rec.get("mono_ns")
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def marker(rec):
    return {"run_id": rec["run_id"], "seq": rec["seq"], "mono_ns": rec.get("mono_ns"), "utc": rec.get("utc")}


def load_phase_of(records, run_id, corrupt, torn_tail):
    """Load duration from the SAME run's monotonic load_start/load_end (qwen4_slowlink_gate progress records). NON-GATING data: elapsed_s is
    set only for a complete, strictly ordered pair; every other case has an explicit status + reason and elapsed_s None (never 0).

    Segment rule: the ACTIVE segment is every record from the LAST run_start of `run_id` onward (all records when that run has no run_start).
    Everything before that anchor is history and is ignored, complete or not. Inside the active segment a load marker carrying another run_id
    is contamination: cross_run, never borrowed. Otherwise the run's own markers decide: one start and one end with mono_ns AND seq strictly
    increasing = complete; own start only = censored; own end only or no marker = missing."""
    anchor = max((i for i, r in enumerate(records) if r["event"] == "run_start" and r["run_id"] == run_id), default=0)
    mk = {ev: [r for r in records[anchor:] if r["event"] == ev] for ev in ("load_start", "load_end")}
    s, e = ([r for r in mk[ev] if r["run_id"] == run_id] for ev in ("load_start", "load_end"))
    foreign = sorted({r["run_id"] for ev in mk for r in mk[ev] if r["run_id"] != run_id})
    out = {"status": None, "reason": None, "elapsed_s": None, "run_id": run_id,
           "load_start": marker(s[0]) if s else None, "load_end": marker(e[0]) if e else None,
           "sidecar_corrupt_lines": corrupt, "sidecar_torn_tail": torn_tail}

    def done(status, reason):
        out.update({"status": status, "reason": reason})
        return out

    if foreign:
        return done("cross_run", f"load marker(s) from run(s) {foreign} inside the active segment of run {run_id}; never borrowed")
    if len(s) > 1 or len(e) > 1:
        return done("malformed", f"{len(s)} load_start and {len(e)} load_end records in run {run_id}; exactly one of each is required")
    for name, group in (("load_start", s), ("load_end", e)):
        if group and not valid_mono(group[0]):
            return done("malformed", f"{name} (seq {group[0]['seq']}) has no valid non-negative integer mono_ns")
    if s and e:
        if not (e[0]["mono_ns"] > s[0]["mono_ns"] and e[0]["seq"] > s[0]["seq"]):
            return done("reversed", "load_end is not strictly after load_start (mono_ns and seq must both increase; equal or reversed)")
        out["elapsed_s"] = (e[0]["mono_ns"] - s[0]["mono_ns"]) / 1e9
        return done("complete", "same-run monotonic load_start to load_end")
    if e:
        return done("missing", "load_end recorded without a load_start")
    damage = "; the sidecar has corrupt/torn lines that could hide a marker" if corrupt or torn_tail else ""
    if s:
        return done("censored", "load_start recorded but no load_end in this run: the load did not complete (timeout, crash or still loading); "
                                "right-censored, not measured" + damage)
    return done("missing", "no load_start or load_end record in this run" + damage)


def read_progress(path):
    """Lenient summary of a (possibly truncated, torn or corrupted) progress sidecar. Never `pass`."""
    out = {"path": os.path.basename(path), "present": os.path.isfile(path), "forensic_only": True, "promotable": False}
    if not out["present"]:
        out.update({"readable": False, "status": "incomplete", "note": "no progress sidecar (the gate never opened its output, or the cell is not a gate cell)"})
        return out
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        out.update({"readable": False, "status": "incomplete", "error": f"unreadable: {e}"})
        return out
    out.update({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    segments = data.split(b"\n")
    tail = segments.pop()
    torn_tail = bool(tail)
    records, corrupt = [], []
    for i, line in enumerate(segments, 1):
        try:
            rec = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            corrupt.append(i)
            continue
        if isinstance(rec, dict) and rec.get("schema") == PROGRESS_SCHEMA and isinstance(rec.get("seq"), int) \
                and isinstance(rec.get("event"), str) and isinstance(rec.get("run_id"), str):
            records.append(rec)
        else:
            corrupt.append(i)
    run_id = None
    for rec in reversed(records):
        if rec["event"] == "run_start":
            run_id = rec["run_id"]
            break
    if run_id is None and records:
        run_id = records[-1]["run_id"]
    run = [r for r in records if r["run_id"] == run_id]
    events, memory_phases, failed_checks, cases = {}, [], [], {}
    error, final = None, None
    for r in run:
        ev = r["event"]
        events[ev] = events.get(ev, 0) + 1
        if ev == "memory" and isinstance(r.get("phase"), str):
            memory_phases.append(r["phase"])
        elif ev == "check" and r.get("status") == "fail" and isinstance(r.get("name"), str):
            failed_checks.append(r["name"])
        elif ev in ("abort", "load_error"):
            error = r.get("error")
        elif ev == "final":
            final = {"status": r.get("status"), "report_sha256": r.get("report_sha256"), "utc": r.get("utc")}
        elif ev in ("repeat_start", "repeat_end") and isinstance(r.get("case"), str):
            c = cases.setdefault(r["case"], {"repeats_completed": 0, "repeats_failed": 0, "in_flight_repeat": None})
            if ev == "repeat_start":
                c["in_flight_repeat"] = r.get("repeat")
            else:
                if c["in_flight_repeat"] == r.get("repeat"):
                    c["in_flight_repeat"] = None
                c["repeats_completed" if r.get("status") == "ok" else "repeats_failed"] += 1
    last = run[-1] if run else None
    out.update({
        "readable": bool(records), "records_valid": len(records), "run_id": run_id, "records_in_run": len(run),
        "run_start_seen": any(r["event"] == "run_start" for r in run),
        "seq_contiguous": all(r["seq"] == i for i, r in enumerate(run)),
        "corrupt_lines": corrupt, "torn_tail": torn_tail,
        "last_event": last["event"] if last else None, "last_utc": last.get("utc") if last else None,
        "last_mono_ns": last.get("mono_ns") if last else None,
        "events": events, "memory_phases": memory_phases, "failed_checks": failed_checks, "cases": cases,
        "error": error, "final_recorded": final,
        "status": "fail" if failed_checks else "incomplete",
        "note": "evidence only: this status can never be pass; a completed run's verdict is the atomically written gate report alone",
        "load_phase": load_phase_of(records, run_id, corrupt, torn_tail),
    })
    return out


def read_report(path):
    out = {"path": os.path.basename(path), "present": os.path.isfile(path)}
    if not out["present"]:
        out.update({"readable": False, "completed": False,
                    "note": "no gate report: the report is written once, atomically, at the end of a run (timeout, crash and load failure leave none)"})
        return out
    out.update(file_row(path))
    try:
        with open(path) as f:
            rep = json.load(f)
    except (OSError, ValueError) as e:
        out.update({"readable": False, "completed": False, "error": f"not valid JSON: {e}"})
        return out
    if not isinstance(rep, dict):
        out.update({"readable": False, "completed": False, "error": "not a JSON object"})
        return out
    status = rep.get("status")
    out.update({"readable": True, "schema": rep.get("schema"), "status": status,
                "completed": rep.get("schema") == REPORT_SCHEMA and status in COMPLETED_STATUSES,
                "error": rep.get("error"),
                "progress": rep.get("progress") if isinstance(rep.get("progress"), dict) else None})
    return out


def read_spawn_env(path):
    """The spawn env pins actually applied to the cell (written by hold-lib.sh cell()); null when absent or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def int_or_none(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def write_json_atomic(path, obj):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def collect(cell, name, rc=None, drained=None):
    files, scratch = {}, []
    for entry in sorted(os.listdir(cell)):
        path = os.path.join(cell, entry)
        if entry in (LOAD_PHASE_FILE, "evidence.json") or entry.startswith((LOAD_PHASE_FILE + ".tmp", "evidence.json.tmp")):
            continue  # derived outputs of this collector, rewritten on every run
        if os.path.isdir(path):
            if entry in SKIP_DIRS:
                scratch.append(entry)
            continue
        if os.path.isfile(path):
            files[entry] = file_row(path)
    rc_file = None
    try:
        with open(os.path.join(cell, "rc.txt")) as f:
            rc_file = int_or_none(f.read().strip())
    except OSError:
        pass
    rc = int_or_none(rc) if rc is not None else rc_file
    sidecars = sorted(n for n in files if n.endswith(PROGRESS_SUFFIX))
    reports = sorted(n[: -len(PROGRESS_SUFFIX)] for n in sidecars)
    gate = reports[0] if reports else "gate.json"
    ev = {
        "schema": SCHEMA, "cell": name, "rc": rc, "rc_file": rc_file, "timed_out": rc in TIMEOUT_RCS,
        "drained": None if drained is None else str(drained).lower() == "true",
        "raw_files": {n: ({"present": True, **files[n]} if n in files else {"present": False}) for n in RAW_FILES},
        "files": files, "scratch_dirs_not_hashed": scratch,
        "gate_report": read_report(os.path.join(cell, gate)),
        "progress_sidecar": read_progress(os.path.join(cell, gate + PROGRESS_SUFFIX)),
        "other_progress_sidecars": [n for n in sidecars if n != gate + PROGRESS_SUFFIX],
        "partial_gate_files": sorted(n for n in files if n.startswith(gate + ".tmp")),
        "spawn_env": read_spawn_env(os.path.join(cell, "spawn-env.json")),
        "pass_policy": PASS_POLICY,
    }
    side = ev["progress_sidecar"]
    if "load_phase" in side:
        phase = side["load_phase"]
    else:
        absent = not side["present"]
        phase = {"status": "no_sidecar" if absent else "unreadable", "reason": side.get("note") or side.get("error"), "elapsed_s": None,
                 "run_id": None, "load_start": None, "load_end": None}
    ev["load_phase"] = {"file": LOAD_PHASE_FILE, "status": phase["status"], "elapsed_s": phase["elapsed_s"], "non_gating": True}
    write_json_atomic(os.path.join(cell, LOAD_PHASE_FILE),
                      {"schema": LOAD_PHASE_SCHEMA, "cell": name, "non_gating": True, "sidecar": side["path"], **phase})
    write_json_atomic(os.path.join(cell, "evidence.json"), ev)
    return ev


def main(argv):
    if len(argv) not in (3, 4, 5) or not os.path.isdir(argv[1]):
        sys.exit(__doc__)
    ev = collect(argv[1], argv[2], argv[3] if len(argv) > 3 else None, argv[4] if len(argv) > 4 else None)
    side = ev["progress_sidecar"]
    print(f"evidence {ev['cell']}: rc={ev['rc']} timed_out={ev['timed_out']} gate_report_completed={ev['gate_report']['completed']} "
          f"progress_sidecar present={side['present']} readable={side['readable']} (forensic only) "
          f"load_phase={ev['load_phase']['status']} elapsed_s={ev['load_phase']['elapsed_s']} (non-gating)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
