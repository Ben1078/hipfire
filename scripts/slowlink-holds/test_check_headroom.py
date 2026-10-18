#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of check_headroom.py and vram_sampler.py on synthetic temp CSV / gate JSON / fake sysfs files. No GPU, no model.

    python3 scripts/slowlink-holds/test_check_headroom.py -v
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_headroom as ch  # noqa: E402

MIN = ch.MIN_FREE_BYTES
TOTAL = 24 * 2**30
MS = 1_000_000


def poke(path, text):
    """Rewrite a fake sysfs counter in place: one same-length write at offset 0 (the sampler keeps its descriptors open)."""
    with open(path, "r+b") as f:
        f.write(text.encode())


def rjson(path):
    with open(path) as f:
        return json.load(f)


def wjson(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def wtext(path, text):
    with open(path, "w") as f:
        f.write(text)


def write_external(cell, free_at=None, floor_free=4 * 2**30, n=300, finished=True, interval=20, errors=0, events=True,
                   cover=True, max_gap_ns=None, summary=True):
    """Synthetic sampler output: n samples 20 ms apart; the sample at index `free_at` has free = floor_free-style override."""
    os.makedirs(cell, exist_ok=True)
    # a fixture is built from scratch: nothing from an earlier write_external() on the same cell may survive (e.g. events=False)
    for name in ("vram.csv", "vram-events.jsonl", "vram-summary.json", "vram.ready"):
        try:
            os.remove(os.path.join(cell, name))
        except FileNotFoundError:
            pass
    t0 = 1_000_000_000_000
    rows = []
    for i in range(n):
        free = floor_free
        if free_at is not None and i == free_at[0]:
            free = free_at[1]
        rows.append((t0 + i * 20 * MS, TOTAL - free, TOTAL))
    if max_gap_ns is not None:  # shift every row after the midpoint so the one gap between consecutive samples is max_gap_ns
        rows = [(t + (max_gap_ns - 20 * MS if i >= n // 2 else 0), u, tt) for i, (t, u, tt) in enumerate(rows)]
    with open(os.path.join(cell, "vram.csv"), "w") as f:
        f.write("t_mono_ns,used,total\n")
        for r in rows:
            f.write("%d,%d,%d\n" % r)
    start = rows[5][0] + 1 if cover else rows[0][0] - 10
    end = rows[-5][0] - 1 if cover else rows[-1][0] + 10
    if events:
        with open(os.path.join(cell, "vram-events.jsonl"), "w") as f:
            f.write(json.dumps({"event": "command_start", "t_mono_ns": start}) + "\n")
            f.write(json.dumps({"event": "command_end", "t_mono_ns": end}) + "\n")
    if summary:
        free = [t - u for _, u, t in rows]
        wjson(os.path.join(cell, "vram-summary.json"),
              {"finished": finished, "requested_interval_ms": interval, "samples": len(rows), "errors": {"count": errors, "first": []},
               "min_free_bytes": min(free), "max_used_bytes": max(u for _, u, _ in rows)})


def write_gate(cell, free=None, drop=None, total=TOTAL, mutate=None, status="pass", schema=None):
    """gate.json with the three light phase points; free = {label: bytes} overrides the default 4 GiB."""
    free = free or {}
    samples = []
    for ph in ch.PHASES:
        if ph == drop:
            continue
        samples.append({"label": ph, "method": "test", "device_free": free.get(ph, 4 * 2**30), "device_total": total})
    if mutate:
        mutate(samples)
    os.makedirs(cell, exist_ok=True)
    wjson(os.path.join(cell, "gate.json"),
          {"schema": schema or ch.REPORT_SCHEMA, "status": status,
           "evidence": {"memory_diagnostics": {"required": False, "samples": samples}}})


class Headroom(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cell = os.path.join(self.tmp.name, "state")

    def tearDown(self):
        self.tmp.cleanup()

    def run_ok(self, **gate):
        write_gate(self.cell, **gate)
        return ch.evaluate(self.cell)

    def test_pass_at_exactly_512_mib_both_sources(self):
        write_external(self.cell, free_at=(100, MIN))
        r = self.run_ok(free={"post_load": MIN})
        self.assertEqual(r["status"], "pass", r["reasons"])
        self.assertEqual(r["gate"]["external_min_free_bytes"], MIN)
        self.assertEqual(r["gate"]["hip_min_free_bytes"], MIN)

    def test_external_one_byte_below_fails(self):
        write_external(self.cell, free_at=(100, MIN - 1))
        r = self.run_ok()
        self.assertEqual(r["status"], "fail")
        self.assertTrue(any("external min free" in x for x in r["reasons"]))

    def test_hip_one_byte_below_fails_for_each_phase(self):
        for ph in ch.PHASES:
            write_external(self.cell)
            r = self.run_ok(free={ph: MIN - 1})
            self.assertEqual(r["status"], "fail", ph)
            self.assertTrue(any("hip min free" in x for x in r["reasons"]), ph)

    def test_external_dip_between_hip_points_still_fails(self):
        write_external(self.cell, free_at=(150, 1))  # hip points alone would pass
        self.assertEqual(self.run_ok()["status"], "fail")

    def test_missing_external_telemetry_is_incomplete_not_pass(self):
        write_gate(self.cell)
        r = ch.evaluate(self.cell)
        self.assertEqual(r["status"], "incomplete")
        self.assertIsNone(r["gate"]["external_min_free_bytes"])

    def test_missing_hip_telemetry_is_incomplete(self):
        write_external(self.cell)
        os.makedirs(self.cell, exist_ok=True)
        self.assertEqual(ch.evaluate(self.cell)["status"], "incomplete")

    def test_each_hip_phase_is_required(self):
        for ph in ch.PHASES:
            write_external(self.cell)
            r = self.run_ok(drop=ph)
            self.assertEqual(r["status"], "incomplete", ph)
            self.assertTrue(any(ph in x for x in r["reasons"]), ph)

    def test_hip_values_never_zero_filled(self):
        for bad in (None, 0, "1", True, -1):
            write_external(self.cell)

            def m(samples, bad=bad):
                samples[1]["device_free"] = bad
            r = self.run_ok(mutate=m)
            self.assertNotEqual(r["status"], "pass", repr(bad))
        write_external(self.cell)
        r = self.run_ok(total=0)
        self.assertEqual(r["status"], "incomplete")  # no positive total

    def test_external_sampler_must_have_finished_with_no_errors_at_20ms(self):
        for kw in ({"finished": False}, {"errors": 3}, {"interval": 50}, {"summary": False}, {"events": False}):
            write_external(self.cell, **kw)
            self.assertEqual(self.run_ok()["status"], "incomplete", kw)

    def test_external_must_cover_the_command_window(self):
        write_external(self.cell, cover=False)  # marks outside the sampled span
        self.assertEqual(self.run_ok()["status"], "incomplete")
        write_external(self.cell, max_gap_ns=1500 * MS)  # a 1.5 s hole inside the window
        r = self.run_ok()
        self.assertEqual(r["status"], "incomplete")
        self.assertTrue(any("coverage gap" in x for x in r["reasons"]))

    def test_coverage_gap_bound_is_exactly_100_ms_five_requested_periods(self):
        self.assertEqual(ch.MAX_GAP_NS, 100 * MS)
        write_external(self.cell, max_gap_ns=100 * MS)  # exactly 5 x 20 ms: passes
        r = self.run_ok()
        self.assertEqual(r["status"], "pass", r["reasons"])
        self.assertEqual(r["external"]["max_gap_in_run_ms"], 100.0)
        self.assertEqual(r["external"]["max_gap_in_run_over_requested_interval"], 5.0)
        write_external(self.cell, max_gap_ns=100 * MS + 1)  # one nanosecond more: INCOMPLETE, not a product failure
        r = self.run_ok()
        self.assertEqual(r["status"], "incomplete")
        self.assertTrue(any("coverage gap" in x for x in r["reasons"]))
        self.assertEqual(r["external"]["max_gap_in_run_ms"], 100.000001)
        # a gap never turns into a product failure by itself, but a really observed breach still fails
        write_external(self.cell, max_gap_ns=100 * MS + 1, free_at=(100, MIN - 1))
        self.assertEqual(self.run_ok()["status"], "fail")

    def test_summary_must_match_csv(self):
        write_external(self.cell)
        p = os.path.join(self.cell, "vram-summary.json")
        s = rjson(p)
        s["min_free_bytes"] += 1
        wjson(p, s)
        self.assertEqual(self.run_ok()["status"], "incomplete")

    def test_diagnostic_prim_peak_is_recorded_but_never_gates(self):
        write_external(self.cell)
        prim = os.path.join(self.tmp.name, "prim")
        write_external(prim, free_at=(100, 1))  # prim free is 1 B: far below the threshold
        write_gate(self.cell)
        r = ch.evaluate(self.cell, [prim])
        self.assertEqual(r["status"], "pass", r["reasons"])
        self.assertEqual(r["info_only"]["diagnostic_cells"]["prim"]["external"]["min_free_bytes"], 1)
        # an unreadable diagnostic cell is information, not a failure
        r = ch.evaluate(self.cell, [os.path.join(self.tmp.name, "absent")])
        self.assertEqual(r["status"], "pass")

    def test_exit_codes(self):
        write_external(self.cell)
        write_gate(self.cell)
        run = lambda: subprocess.run([sys.executable, os.path.join(HERE, "check_headroom.py"), self.cell], capture_output=True, text=True)
        self.assertEqual(run().returncode, 0)
        write_external(self.cell, free_at=(100, MIN - 1))
        self.assertEqual(run().returncode, 1)
        os.remove(os.path.join(self.cell, "gate.json"))
        write_external(self.cell)
        self.assertEqual(run().returncode, 2)

    def test_only_a_completed_gate_report_can_pass(self):
        write_external(self.cell)
        for status in ("incomplete", "running", None):
            with self.subTest(status=status):
                write_gate(self.cell, status=status)
                r = ch.evaluate(self.cell)
                self.assertEqual(r["status"], "incomplete", r)
                self.assertTrue(any("completed gate report" in x for x in r["reasons"]), r["reasons"])
        write_gate(self.cell, schema="x")
        self.assertEqual(ch.evaluate(self.cell)["status"], "incomplete")
        for status in ("pass", "fail"):  # a completed report (whatever the gate verdict) carries valid points; the headroom verdict is its own
            write_gate(self.cell, status=status)
            self.assertEqual(ch.evaluate(self.cell)["status"], "pass", status)

    def test_a_breach_in_an_incomplete_report_still_fails(self):
        write_external(self.cell)
        write_gate(self.cell, status="incomplete", free={"post_load": MIN - 1})
        self.assertEqual(ch.evaluate(self.cell)["status"], "fail")

    def test_progress_sidecar_is_forensic_only_and_never_read(self):
        write_external(self.cell)
        os.makedirs(self.cell, exist_ok=True)
        points = [{"schema": "hipfire.qwen4.slowlink_gate.progress.v1", "run_id": "r", "seq": i, "event": "memory", "phase": ph,
                   "device_free": 4 * 2**30, "device_total": TOTAL} for i, ph in enumerate(ch.PHASES)]
        wtext(os.path.join(self.cell, "gate.json.progress.jsonl"), "".join(json.dumps(p) + "\n" for p in points))
        # no gate report at all: the sidecar's perfect points cannot stand in for it
        r = ch.evaluate(self.cell)
        self.assertEqual(r["status"], "incomplete", r)
        self.assertIsNone(r["gate"]["hip_min_free_bytes"])
        # a completed report decides alone: a sidecar full of breaching points changes nothing
        write_gate(self.cell)
        bad = [dict(p, device_free=1) for p in points]
        wtext(os.path.join(self.cell, "gate.json.progress.jsonl"), "".join(json.dumps(p) + "\n" for p in bad))
        self.assertEqual(ch.evaluate(self.cell)["status"], "pass")


class SamplerIntegration(unittest.TestCase):
    """vram_sampler.py against fake sysfs files: ready handshake, marks, clean stop, and the checker accepting its output."""

    # sysfs counters are rewritten IN PLACE (same length, one write at offset 0) because the sampler keeps its descriptors open

    def test_sampler_roundtrip_and_error_is_not_zero_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev, cell = os.path.join(tmp, "dev"), os.path.join(tmp, "cell")
            os.makedirs(dev)
            wtext(os.path.join(dev, "mem_info_vram_total"), str(TOTAL))
            wtext(os.path.join(dev, "mem_info_vram_used"), str(TOTAL - 4 * 2**30))
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "vram_sampler.py"), dev, cell, "20"])
            try:
                end = time.monotonic() + 10
                while not os.path.exists(os.path.join(cell, "vram.ready")):
                    self.assertLess(time.monotonic(), end, "sampler never became ready")
                    time.sleep(0.01)
                p.send_signal(signal.SIGUSR1)
                time.sleep(0.4)
                poke(os.path.join(dev, "mem_info_vram_used"), str(TOTAL - MIN))  # exactly the threshold
                time.sleep(0.4)
                p.send_signal(signal.SIGUSR2)
                time.sleep(0.1)
            finally:
                p.terminate()
            self.assertEqual(p.wait(timeout=10), 0)
            s = rjson(os.path.join(cell, "vram-summary.json"))
            self.assertTrue(s["finished"])
            self.assertEqual(s["requested_interval_ms"], 20.0)
            self.assertEqual(s["errors"]["count"], 0)
            self.assertEqual(s["min_free_bytes"], MIN)
            write_gate(cell)
            r = ch.evaluate(cell)
            self.assertEqual(r["status"], "pass", r["reasons"])
            self.assertEqual(r["gate"]["external_min_free_bytes"], MIN)

            # an unparseable sysfs value is an error count and never a zero/free value: the checker cannot pass it
            cell2 = os.path.join(tmp, "cell2")
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "vram_sampler.py"), dev, cell2, "20"])
            try:
                end = time.monotonic() + 10
                while not os.path.exists(os.path.join(cell2, "vram.ready")):
                    self.assertLess(time.monotonic(), end)
                    time.sleep(0.01)
                p.send_signal(signal.SIGUSR1)
                time.sleep(0.2)
                poke(os.path.join(dev, "mem_info_vram_used"), "junkjunkjun")
                time.sleep(0.2)
                p.send_signal(signal.SIGUSR2)
                time.sleep(0.1)
            finally:
                p.terminate()
            p.wait(timeout=10)
            s2 = rjson(os.path.join(cell2, "vram-summary.json"))
            self.assertGreater(s2["errors"]["count"], 0)
            write_gate(cell2)
            self.assertEqual(ch.evaluate(cell2)["status"], "incomplete")

    def test_sampler_refuses_missing_sysfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, os.path.join(HERE, "vram_sampler.py"), os.path.join(tmp, "none"), os.path.join(tmp, "c")],
                               capture_output=True)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(os.path.exists(os.path.join(tmp, "c", "vram.ready")))


if __name__ == "__main__":
    unittest.main()
