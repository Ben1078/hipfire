#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the NON-GATING load-phase timing that collect_evidence.py writes to CELL/load_phase.json: elapsed seconds between the
same run's monotonic load_start and load_end progress records when both exist and are ordered; explicit status/reason with elapsed_s null for
a censored (j1892: load_start, no load_end), missing, malformed, reversed or cross-run marker; the raw sidecar/gate files and the cell verdict
inputs never change. The only non-synthetic input is the bundled load-phase-censored-fixture.jsonl, a byte-for-byte copy of the REAL j1892 censored
progress sidecar (6 events, sha256-pinned, a log not a model; required, never skipped). Everything else is a tiny synthetic temp file or a mock:
no GPU, no model file, no staged binary is ever read or executed.

    python3 scripts/slowlink-holds/test_load_phase.py -v
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_evidence as ce  # noqa: E402

J1892_SIDECAR = os.path.join(HERE, "load-phase-censored-fixture.jsonl")
J1892_SHA256 = "a200d07491199e18bb59429cda07df968bb99745f839053456e4b7fdf6101890"
J1892_RUN = "4083-1792359734857"
J1892_LOAD_START_MONO = 535497991


DEFAULT = object()


def rec(seq, event, run="r1", mono=DEFAULT, **fields):
    """One synthetic progress record (SYNTHETIC: mono_ns defaults to a made-up value derived from seq)."""
    r = {"schema": ce.PROGRESS_SCHEMA, "run_id": run, "seq": seq, "event": event, "utc": "2026-01-01T00:00:00.000Z", "unix_ms": 1,
         "mono_ns": 1_000_000_000 * (seq + 1) if mono is DEFAULT else mono, **fields}
    if mono is False:
        del r["mono_ns"]
    return r


def jsonl(records):
    return "".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records)


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cell = os.path.join(tmp.name, "prim")
        os.makedirs(os.path.join(self.cell, "home"))

    def put(self, name, data):
        with open(os.path.join(self.cell, name), "wb" if isinstance(data, bytes) else "w") as f:
            f.write(data)

    def phase(self, text, torn=""):
        """Collect a cell whose sidecar holds `text` (+ an unterminated tail) and return (load_phase.json content, evidence)."""
        self.put("rc.txt", "0\n")
        self.put("gate.json.progress.jsonl", text + torn)
        ev = ce.collect(self.cell, "prim", "0", "true")
        with open(os.path.join(self.cell, ce.LOAD_PHASE_FILE)) as f:
            return json.load(f), ev

    def assertNotMeasured(self, lp, status):
        self.assertEqual(lp["status"], status)
        self.assertIsNone(lp["elapsed_s"])  # null, never 0 and never a fabricated duration
        self.assertTrue(lp["reason"])


class Elapsed(Base):
    def test_complete_pair_gives_elapsed_seconds_from_monotonic_ns(self):
        lp, ev = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=2_000_000_000), rec(2, "load_end", mono=196_300_000_000)]))
        self.assertEqual(lp["status"], "complete")
        self.assertAlmostEqual(lp["elapsed_s"], 194.3, places=9)
        self.assertEqual((lp["schema"], lp["cell"], lp["non_gating"], lp["run_id"]), (ce.LOAD_PHASE_SCHEMA, "prim", True, "r1"))
        self.assertEqual((lp["load_start"]["seq"], lp["load_end"]["seq"]), (1, 2))
        self.assertEqual(lp["sidecar"], "gate.json.progress.jsonl")
        self.assertEqual(ev["load_phase"], {"file": "load_phase.json", "status": "complete", "elapsed_s": lp["elapsed_s"], "non_gating": True})

    def test_the_markers_do_not_need_to_be_adjacent(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=5_000_000_000), rec(2, "memory", phase="pre_load"),
                                  rec(3, "check", name="x", status="pass"), rec(4, "load_end", mono=7_500_000_000), rec(5, "final")]))
        self.assertEqual((lp["status"], lp["elapsed_s"]), ("complete", 2.5))

    def test_the_last_run_is_used_and_an_earlier_complete_run_is_not_borrowed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start", run="old"), rec(1, "load_start", run="old"), rec(2, "load_end", run="old"),
                                  rec(0, "run_start", run="new"), rec(1, "load_start", run="new", mono=3_000_000_000)]))
        self.assertEqual(lp["run_id"], "new")
        self.assertEqual(lp["load_start"]["run_id"], "new")
        self.assertNotMeasured(lp, "censored")


class Censored(Base):
    def test_load_start_without_load_end_is_right_censored_not_zero(self):
        lp, ev = self.phase(jsonl([rec(0, "run_start"), rec(1, "plan"), rec(2, "load_start")]))
        self.assertNotMeasured(lp, "censored")
        self.assertIn("right-censored", lp["reason"])
        self.assertIsNotNone(lp["load_start"])
        self.assertIsNone(lp["load_end"])
        self.assertIsNone(ev["load_phase"]["elapsed_s"])

    def test_a_torn_or_corrupt_sidecar_is_made_visible_on_a_censored_load(self):
        torn = json.dumps(rec(3, "load_end"))[:25]  # a load_end cut mid-write: never interpreted
        lp, ev = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start"), "{not json"]), torn=torn)
        self.assertNotMeasured(lp, "censored")
        self.assertEqual((lp["sidecar_corrupt_lines"], lp["sidecar_torn_tail"]), ([3], True))
        self.assertIn("could hide a marker", lp["reason"])
        self.assertEqual((ev["progress_sidecar"]["corrupt_lines"], ev["progress_sidecar"]["torn_tail"]), ([3], True))

    def test_a_clean_censored_sidecar_does_not_claim_damage(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start")]))
        self.assertNotIn("could hide", lp["reason"])
        self.assertEqual((lp["sidecar_corrupt_lines"], lp["sidecar_torn_tail"]), ([], False))


class Missing(Base):
    def test_no_marker_at_all_is_missing(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "plan"), rec(2, "load_error", error="boom")]))
        self.assertNotMeasured(lp, "missing")
        self.assertIsNone(lp["load_start"])
        self.assertIsNone(lp["load_end"])

    def test_load_end_without_load_start_is_missing(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_end")]))
        self.assertNotMeasured(lp, "missing")
        self.assertIn("without a load_start", lp["reason"])

    def test_an_empty_sidecar_is_missing_and_the_absent_sidecar_is_no_sidecar(self):
        lp, ev = self.phase("")
        self.assertNotMeasured(lp, "missing")
        self.assertFalse(ev["progress_sidecar"]["readable"])
        os.remove(os.path.join(self.cell, "gate.json.progress.jsonl"))
        ev = ce.collect(self.cell, "prim", "1", "true")
        with open(os.path.join(self.cell, ce.LOAD_PHASE_FILE)) as f:
            lp = json.load(f)
        self.assertNotMeasured(lp, "no_sidecar")
        self.assertEqual(ev["load_phase"]["status"], "no_sidecar")
        self.assertEqual(lp["sidecar"], "gate.json.progress.jsonl")


class Malformed(Base):
    def test_a_bad_monotonic_timestamp_is_malformed(self):
        for label, bad in (("string", "12"), ("float", 1.5), ("negative", -1), ("bool", True), ("null", None), ("absent", False)):
            with self.subTest(label):
                lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=bad), rec(2, "load_end", mono=9_000_000_000)]))
                self.assertNotMeasured(lp, "malformed")
                self.assertIn("mono_ns", lp["reason"])

    def test_a_bad_load_end_timestamp_on_a_censored_looking_run_is_malformed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=1_000), rec(2, "load_end", mono="later")]))
        self.assertNotMeasured(lp, "malformed")
        self.assertEqual(lp["load_end"]["mono_ns"], "later")

    def test_duplicate_markers_in_one_run_are_malformed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start"), rec(2, "load_start"), rec(3, "load_end")]))
        self.assertNotMeasured(lp, "malformed")
        self.assertIn("exactly one", lp["reason"])
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start"), rec(2, "load_end"), rec(3, "load_end")]))
        self.assertNotMeasured(lp, "malformed")

    def test_a_record_that_fails_the_schema_is_a_corrupt_line_never_a_marker(self):
        no_run = {k: v for k, v in rec(2, "load_end").items() if k != "run_id"}
        lp, ev = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start"), no_run]))
        self.assertNotMeasured(lp, "censored")  # the schema-less line is not trusted as a load_end
        self.assertEqual(ev["progress_sidecar"]["corrupt_lines"], [3])
        self.assertIn("could hide a marker", lp["reason"])


class Reversed(Base):
    def test_load_end_before_load_start_by_monotonic_clock_is_reversed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=9_000_000_000), rec(2, "load_end", mono=3_000_000_000)]))
        self.assertNotMeasured(lp, "reversed")

    def test_load_end_recorded_before_load_start_by_seq_is_reversed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_end", mono=9_000_000_000), rec(2, "load_start", mono=3_000_000_000)]))
        self.assertNotMeasured(lp, "reversed")

    def test_equal_monotonic_timestamps_are_never_a_fake_zero_duration(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=5_000_000_000), rec(2, "load_end", mono=5_000_000_000)]))
        self.assertNotMeasured(lp, "reversed")
        self.assertIn("strictly after", lp["reason"])

    def test_equal_seq_is_never_a_measured_duration(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start"), rec(1, "load_start", mono=1_000_000_000), rec(1, "load_end", mono=9_000_000_000)]))
        self.assertNotMeasured(lp, "reversed")


class CrossRun(Base):
    """Segment rule: the active segment starts at the LAST run_start of the selected run; history before it is ignored (never borrowed), a
    load marker with a foreign run_id INSIDE the active segment is cross_run."""

    def test_a_start_from_before_the_new_run_start_is_history_not_borrowed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start", run="a"), rec(1, "load_start", run="a", mono=1_000_000_000),
                                  rec(0, "run_start", run="b"), rec(1, "load_end", run="b", mono=9_000_000_000)]))
        self.assertNotMeasured(lp, "missing")
        self.assertEqual(lp["run_id"], "b")
        self.assertIsNone(lp["load_start"])
        self.assertIn("without a load_start", lp["reason"])

    def test_an_end_from_before_the_new_run_start_is_history_not_borrowed(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start", run="b"), rec(1, "load_end", run="b", mono=1), rec(0, "run_start", run="a"),
                                  rec(1, "load_start", run="a", mono=1_000_000_000)]))
        self.assertNotMeasured(lp, "censored")
        self.assertEqual(lp["run_id"], "a")
        self.assertIsNone(lp["load_end"])

    def test_a_complete_earlier_run_does_not_give_a_markerless_new_run_a_measurement(self):
        lp, _ = self.phase(jsonl([rec(0, "run_start", run="a"), rec(1, "load_start", run="a"), rec(2, "load_end", run="a"),
                                  rec(0, "run_start", run="b"), rec(1, "plan", run="b")]))
        self.assertNotMeasured(lp, "missing")
        self.assertEqual(lp["run_id"], "b")
        self.assertIsNone(lp["load_start"])
        self.assertIsNone(lp["load_end"])

    def test_a_foreign_marker_inside_the_active_segment_is_cross_run_even_with_an_own_complete_pair(self):
        for label, foreign in (("start", rec(2, "load_start", run="a", mono=2_000_000_000)), ("end", rec(2, "load_end", run="a", mono=8_000_000_000))):
            with self.subTest(label):
                lp, _ = self.phase(jsonl([rec(0, "run_start", run="b"), rec(1, "load_start", run="b", mono=1_000_000_000), foreign,
                                          rec(3, "load_end", run="b", mono=9_000_000_000)]))
                self.assertNotMeasured(lp, "cross_run")
                self.assertEqual(lp["run_id"], "b")
                self.assertIn("['a']", lp["reason"])

    def test_a_foreign_marker_is_never_borrowed_to_complete_or_explain_a_missing_own_marker(self):
        cases = {
            "own start, foreign end": [rec(0, "run_start", run="b"), rec(1, "load_start", run="b"), rec(2, "load_end", run="a", mono=9_000_000_000)],
            "own end, foreign start": [rec(0, "run_start", run="b"), rec(1, "load_start", run="a", mono=1), rec(2, "load_end", run="b", mono=9_000_000_000)],
            "markerless, foreign marker": [rec(0, "run_start", run="b"), rec(1, "load_start", run="a")],
        }
        for label, records in cases.items():
            with self.subTest(label):
                lp, _ = self.phase(jsonl(records))
                self.assertNotMeasured(lp, "cross_run")
                self.assertEqual(lp["run_id"], "b")

    def test_foreign_markers_in_the_active_segment_without_any_run_start_are_cross_run(self):
        lp, _ = self.phase(jsonl([rec(1, "load_start", run="a", mono=1_000_000_000), rec(2, "load_end", run="b", mono=9_000_000_000)]))
        self.assertNotMeasured(lp, "cross_run")  # no run_start anchor: every record is active; the last record's run_id is selected
        self.assertEqual(lp["run_id"], "b")


class Integration(Base):
    def test_collect_writes_load_phase_json_without_touching_any_raw_file_or_verdict_input(self):
        gate = json.dumps({"schema": ce.REPORT_SCHEMA, "status": "pass", "progress": {"records_before_final": 3}})
        side = jsonl([rec(0, "run_start"), rec(1, "load_start", mono=1_000_000_000), rec(2, "load_end", mono=4_000_000_000)])
        for name, data in (("gate.json", gate), ("gate.json.progress.jsonl", side), ("rc.txt", "0\n"), ("stdout.txt", "out"), ("stderr.txt", "err")):
            self.put(name, data)
        before = {n: open(os.path.join(self.cell, n), "rb").read() for n in sorted(os.listdir(self.cell)) if os.path.isfile(os.path.join(self.cell, n))}
        ev = ce.collect(self.cell, "prim", "0", "true")
        after = {n: open(os.path.join(self.cell, n), "rb").read() for n in before}
        self.assertEqual(after, before)  # raw records, progress and gate report byte-identical
        self.assertTrue(ev["gate_report"]["completed"])  # original gate/cell verdict inputs unchanged
        self.assertEqual(ev["gate_report"]["status"], "pass")
        self.assertEqual(ev["progress_sidecar"]["status"], "incomplete")  # the sidecar can still never say pass
        self.assertFalse(ev["progress_sidecar"]["promotable"])
        self.assertEqual((ev["rc"], ev["timed_out"]), (0, False))
        self.assertEqual(ev["progress_sidecar"]["load_phase"]["elapsed_s"], 3.0)
        with open(os.path.join(self.cell, ce.LOAD_PHASE_FILE)) as f:
            self.assertEqual(json.load(f)["elapsed_s"], 3.0)
        self.assertFalse([n for n in os.listdir(self.cell) if ".tmp" in n])
        self.assertNotIn(ce.LOAD_PHASE_FILE, ev["files"])  # a derived output, like evidence.json: not indexed, so a re-run is stable
        again = ce.collect(self.cell, "prim", "0", "true")
        self.assertEqual(again, ev)

    def test_a_timeout_with_a_censored_load_keeps_rc_124_and_reports_null_elapsed(self):
        self.put("rc.txt", "124\n")
        self.put("gate.json.progress.jsonl", jsonl([rec(0, "run_start"), rec(1, "load_start")]))
        ev = ce.collect(self.cell, "prim", "124", "true")
        self.assertTrue(ev["timed_out"])
        self.assertFalse(ev["gate_report"]["completed"])
        self.assertEqual((ev["load_phase"]["status"], ev["load_phase"]["elapsed_s"]), ("censored", None))

    def test_the_command_line_prints_the_load_phase_and_still_exits_0(self):
        self.put("gate.json.progress.jsonl", jsonl([rec(0, "run_start"), rec(1, "load_start", mono=0), rec(2, "load_end", mono=2_000_000_000)]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(ce.main(["collect_evidence.py", self.cell, "prim", "0", "true"]), 0)
        self.assertIn("load_phase=complete elapsed_s=2.0 (non-gating)", out.getvalue())

    def test_no_verdict_code_reads_the_load_phase(self):
        for name in ("hold-lib.sh", "check_headroom.py", "h1_handoff.py", "hold1a-correct.sh", "hold1b-correct.sh"):
            with open(os.path.join(HERE, name)) as f:
                body = f.read()
            code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
            self.assertNotIn("load_phase", code, name)

    def test_this_test_module_never_loads_a_model_or_a_gpu_runtime(self):
        with open(os.path.abspath(__file__)) as f:
            imports = [l.split()[1] for l in f if l.startswith(("import ", "from "))]
        for mod in imports:
            self.assertFalse(mod.lower().startswith(("torch", "hip", "rocm", "cupy", "pyamdgpu")), mod)

    def test_an_existing_but_unreadable_sidecar_is_an_explicit_unreadable_status(self):
        # mock-based (root-safe, no permission assumptions): opening the sidecar raises OSError; every other open is real
        self.put("rc.txt", "0\n")
        self.put("gate.json.progress.jsonl", jsonl([rec(0, "run_start"), rec(1, "load_start"), rec(2, "load_end")]))
        real_open = open

        def deny(path, *args, **kwargs):
            if str(path).endswith(ce.PROGRESS_SUFFIX):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", deny), mock.patch.object(ce, "sha256_file", lambda p: "0" * 64):
            ev = ce.collect(self.cell, "prim", "0", "true")
        side = ev["progress_sidecar"]
        self.assertTrue(side["present"])
        self.assertFalse(side["readable"])
        self.assertIn("unreadable", side["error"])
        self.assertEqual(side["status"], "incomplete")
        with open(os.path.join(self.cell, ce.LOAD_PHASE_FILE)) as f:
            lp = json.load(f)
        self.assertNotMeasured(lp, "unreadable")  # neither no_sidecar nor a fabricated completed duration
        self.assertIn("Permission denied", lp["reason"])
        self.assertTrue(lp["non_gating"])
        self.assertEqual(ev["load_phase"]["status"], "unreadable")


class RealJ1892Sidecar(Base):
    """The bundled copy of the REAL censored sidecar of hold1a j1892 (timeout rc 124 at 181 s: load_start at mono 535497991, never a load_end)."""

    def test_the_real_artifact_is_the_pinned_one(self):
        with open(J1892_SIDECAR, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), J1892_SHA256)
        with open(J1892_SIDECAR, "rb") as f:
            events = [json.loads(l)["event"] for l in f.read().splitlines()]
        self.assertEqual(events, ["run_start", "plan", "memory", "check", "profile", "load_start"])

    def test_the_real_censored_load_is_reported_censored_with_null_elapsed(self):
        shutil.copy(J1892_SIDECAR, os.path.join(self.cell, "gate.json.progress.jsonl"))
        self.put("rc.txt", "124\n")
        before = open(os.path.join(self.cell, "gate.json.progress.jsonl"), "rb").read()
        ev = ce.collect(self.cell, "prim", "124", "true")
        self.assertEqual(open(os.path.join(self.cell, "gate.json.progress.jsonl"), "rb").read(), before)
        with open(os.path.join(self.cell, ce.LOAD_PHASE_FILE)) as f:
            lp = json.load(f)
        self.assertNotMeasured(lp, "censored")
        self.assertEqual(lp["run_id"], J1892_RUN)
        self.assertEqual((lp["load_start"]["seq"], lp["load_start"]["mono_ns"]), (5, J1892_LOAD_START_MONO))
        self.assertIsNone(lp["load_end"])
        self.assertEqual((lp["sidecar_corrupt_lines"], lp["sidecar_torn_tail"]), ([], False))
        self.assertTrue(ev["timed_out"])
        self.assertFalse(ev["gate_report"]["present"])
        side = ev["progress_sidecar"]
        self.assertEqual((side["records_valid"], side["last_event"], side["status"]), (6, "load_start", "incomplete"))
        self.assertTrue(side["seq_contiguous"])


if __name__ == "__main__":
    unittest.main()
