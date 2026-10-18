#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_headroom.py CELL_DIR [--info-cell DIR ...]

Hold 1's PRODUCT-state VRAM HEADROOM check (replaces the retired residual-prim ownership check). It judges ONE question on the
product `state` cell, from two independent observation sources, BOTH required:

  external  CELL_DIR/vram.csv + vram-summary.json + vram-events.jsonl (vram_sampler.py: sysfs mem_info_vram_used/total at a requested
            20 ms, CLOCK_MONOTONIC stamps, started before the GPU command and stopped after the command and the drain).
            external_min_free = min over EVERY sample of (total - used) across the whole run.
  hip       CELL_DIR/gate.json  (a COMPLETED gate report: schema hipfire.qwen4.slowlink_gate.v1, status pass|fail; an `incomplete`
            report or only the <output>.progress.jsonl sidecar is INCOMPLETE, never a pass) evidence.memory_diagnostics.samples, the
            in-process hipMemGetInfo points `pre_load`, `post_load` and `end` (each with a positive device_total and a device_free in
            0..total). hip_min_free = min over those labelled points of device_free.

PASS only when external_min_free >= MIN_FREE_BYTES (512 MiB = 536,870,912 B) AND hip_min_free >= MIN_FREE_BYTES. The threshold is a
fixed constant, not an option. One byte below fails. Missing, unparseable, partial or inconsistent telemetry is never zero-filled and
never passes: it is INCOMPLETE (exit 2). A value really observed below the threshold is FAIL (exit 1), even if the other source is
incomplete. A telemetry source is VALID only when: the sampler summary says finished with 0 read/parse errors and the requested
interval is 20 ms; the summary sample count equals the CSV rows and its min/max equal the values recomputed from the CSV; every row has
total > 0 and 0 <= used <= total with non-decreasing timestamps; the command_start and command_end marks exist in order; there is a
sample at or before command_start and one at or after command_end; no gap between consecutive samples from the last sample at or
before command_start to the first at or after command_end exceeds MAX_GAP_NS (100 ms = 5 requested 20 ms periods; the actual max gap and its
ratio to the requested interval are reported; a longer gap makes the telemetry INCOMPLETE, a coverage-quality verdict, never a product
failure unless a free-VRAM threshold breach was also observed); and
the gate report carries all three phase points.

Everything else (peak used, baseline, the sample spacing, the optional --info-cell diagnostic cells such as the primitive cell, the
census/ledger) is INFORMATION ONLY: recorded in the JSON, never gated, never classified as an attribution. Output is JSON on
stdout and one summary line on stderr; exit 0 pass, 1 fail, 2 incomplete."""
import csv
import json
import os
import statistics
import sys

MIN_FREE_BYTES = 536_870_912
REQUESTED_INTERVAL_MS = 20
MAX_GAP_NS = 5 * REQUESTED_INTERVAL_MS * 1_000_000  # 100 ms = 5 requested periods: coverage quality only (> => INCOMPLETE, never a product fail)
PHASES = ("pre_load", "post_load", "end")
REPORT_SCHEMA = "hipfire.qwen4.slowlink_gate.v1"
COMPLETED_STATUSES = ("pass", "fail")


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def load_external(cell):
    """Returns (info dict, reasons). reasons empty = valid. info always carries what could be read."""
    why, info = [], {}
    try:
        with open(os.path.join(cell, "vram-summary.json")) as f:
            summ = json.load(f)
    except (OSError, ValueError) as e:
        return info, [f"external vram-summary.json unreadable: {e}"]
    if not isinstance(summ, dict) or summ.get("finished") is not True:
        return info, ["external sampler did not finish (no clean stop)"]
    errs = summ.get("errors") or {}
    if errs.get("count") != 0:
        why.append(f"external sampler recorded {errs.get('count')!r} read/parse errors")
    if summ.get("requested_interval_ms") != REQUESTED_INTERVAL_MS:
        why.append(f"external requested interval {summ.get('requested_interval_ms')!r} ms != {REQUESTED_INTERVAL_MS} ms")
    try:
        with open(os.path.join(cell, "vram.csv")) as f:
            rows = [(int(r["t_mono_ns"]), int(r["used"]), int(r["total"])) for r in csv.DictReader(f)]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return info, why + [f"external vram.csv unreadable: {e}"]
    if not rows:
        return info, why + ["external vram.csv has no samples"]
    if summ.get("samples") != len(rows):
        why.append(f"summary samples {summ.get('samples')!r} != csv rows {len(rows)}")
    bad = [i for i, (_, u, t) in enumerate(rows) if t <= 0 or u < 0 or u > t]
    if bad:
        why.append(f"{len(bad)} sample(s) with total <= 0 or used outside 0..total (first row {bad[0]})")
    ts = [r[0] for r in rows]
    if any(b < a for a, b in zip(ts, ts[1:])):
        why.append("external sample timestamps not monotonic")
    free = [t - u for _, u, t in rows]
    used = [u for _, u, _ in rows]
    totals = [t for _, _, t in rows]
    ext_min = min(free)
    if summ.get("min_free_bytes") != ext_min or summ.get("max_used_bytes") != max(used):
        why.append("summary min_free/max_used differ from the csv recomputation")
    ev = {}
    try:
        with open(os.path.join(cell, "vram-events.jsonl")) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    ev.setdefault(e["event"], e["t_mono_ns"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        why.append(f"external vram-events.jsonl unreadable: {e}")
    start, end = ev.get("command_start"), ev.get("command_end")
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    min_at = rows[free.index(ext_min)][0]
    info = {"samples": len(rows), "min_free_bytes": ext_min, "min_free_at_ms_from_first_sample": (min_at - ts[0]) / 1e6,
            "max_used_bytes": max(used), "first_used_bytes": used[0], "min_total_bytes": min(totals), "max_total_bytes": max(totals),
            "requested_interval_ms": summ.get("requested_interval_ms"), "median_gap_ms": statistics.median(gaps) / 1e6 if gaps else None,
            "max_gap_ms": max(gaps) / 1e6 if gaps else None, "overruns": summ.get("overruns"),
            "span_s": (ts[-1] - ts[0]) / 1e9, "marks": ev}
    if start is None or end is None or not start < end:
        why.append("command_start/command_end marks missing or out of order")
    else:
        before = [i for i, t in enumerate(ts) if t <= start]
        after = [i for i, t in enumerate(ts) if t >= end]
        if not before:
            why.append("no external sample at or before command_start")
        if not after:
            why.append("no external sample at or after command_end")
        if before and after:
            lo, hi = before[-1], after[0]
            worst = max((ts[i + 1] - ts[i] for i in range(lo, hi)), default=0)
            info["max_gap_in_run_ms"] = worst / 1e6
            info["max_gap_in_run_over_requested_interval"] = worst / (REQUESTED_INTERVAL_MS * 1e6)
            if worst > MAX_GAP_NS:
                why.append(f"external coverage gap {worst / 1e6:.3f} ms ({worst / (REQUESTED_INTERVAL_MS * 1e6):.2f}x the requested "
                           f"{REQUESTED_INTERVAL_MS} ms) > {MAX_GAP_NS / 1e6:.0f} ms inside the command window")
    return info, why


def load_hip(cell):
    why = []
    try:
        with open(os.path.join(cell, "gate.json")) as f:
            rep = json.load(f)
        samples = rep["evidence"]["memory_diagnostics"]["samples"]
        if not isinstance(samples, list):
            raise TypeError("samples is not a list")
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {}, [f"gate.json evidence.memory_diagnostics.samples unreadable: {e!r}"]
    # Only a COMPLETED gate report (the file the gate writes once, atomically, at the end of a run: schema + status pass|fail) can carry the
    # in-process points of a pass. An aborted run's `incomplete` report, and the durable <output>.progress.jsonl sidecar (forensic only, never
    # read here), make the telemetry INCOMPLETE; a breach observed in the samples still fails.
    if rep.get("schema") != REPORT_SCHEMA or rep.get("status") not in COMPLETED_STATUSES:
        why.append(f"gate.json is not a completed gate report (schema {rep.get('schema')!r}, status {rep.get('status')!r}); "
                   f"the progress sidecar is forensic only")
    points = {}
    for ph in PHASES:
        rows = [s for s in samples if isinstance(s, dict) and s.get("label") == ph]
        if not rows:
            why.append(f"hip sample {ph} missing")
            continue
        for s in rows:
            f, t = s.get("device_free"), s.get("device_total")
            if not (_int(f) and _int(t)) or t <= 0 or f < 0 or f > t:
                why.append(f"hip sample {ph} has no valid positive device_total / device_free in 0..total ({f!r}, {t!r})")
            else:
                points.setdefault(ph, []).append({"free": f, "total": t, "used": t - f})
    flat = [p["free"] for v in points.values() for p in v]
    info = {"points": points, "min_free_bytes": min(flat) if flat else None}
    return info, why


def cell_info(cell):
    """Information-only digest of a diagnostic cell: never gated, never raises."""
    info, why = load_external(cell)
    out = {"external": {k: info.get(k) for k in ("samples", "min_free_bytes", "max_used_bytes", "first_used_bytes", "span_s")}}
    if why:
        out["external_unavailable_or_invalid"] = why
    return out


def evaluate(cell, info_cells=()):
    ext, ext_why = load_external(cell)
    hip, hip_why = load_hip(cell)
    below = []
    if ext.get("min_free_bytes") is not None and ext["min_free_bytes"] < MIN_FREE_BYTES:
        below.append(f"external min free {ext['min_free_bytes']} B < {MIN_FREE_BYTES} B")
    if hip.get("min_free_bytes") is not None and hip["min_free_bytes"] < MIN_FREE_BYTES:
        below.append(f"hip min free {hip['min_free_bytes']} B < {MIN_FREE_BYTES} B")
    reasons = below + ext_why + hip_why
    status = "fail" if below else ("incomplete" if reasons else "pass")
    return {"status": status, "reasons": reasons, "gate": {"min_free_bytes_required": MIN_FREE_BYTES,
            "external_min_free_bytes": ext.get("min_free_bytes"), "hip_min_free_bytes": hip.get("min_free_bytes"),
            "both_sources_and_all_phases_required": True},
            "external": ext, "hip": hip,
            "info_only": {"diagnostic_cells": {os.path.basename(os.path.normpath(c)): cell_info(c) for c in info_cells},
                          "note": "peak/used/baseline/ledger/census values are recorded, never gated and never an attribution"}}


def main(argv):
    args, info_cells, i = [], [], 1
    while i < len(argv):
        if argv[i] == "--info-cell" and i + 1 < len(argv):
            info_cells.append(argv[i + 1])
            i += 2
        else:
            args.append(argv[i])
            i += 1
    if len(args) != 1:
        sys.exit(__doc__)
    out = evaluate(args[0], info_cells)
    print(json.dumps(out, indent=1))
    print("headroom %s: %s" % (out["status"], "; ".join(out["reasons"]) or "external and hip minima >= 512 MiB"), file=sys.stderr)
    return {"pass": 0, "fail": 1}.get(out["status"], 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
