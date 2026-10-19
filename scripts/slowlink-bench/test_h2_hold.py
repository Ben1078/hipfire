#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for h2_hold.py (the literal H2 hold dispatcher).  Run:  python3 test_h2_hold.py

A callable FIXTURE runner stands in for the core subprocess: it records the exact argv and writes SYNTHETIC cell/pair/judge documents only.  Nothing
here touches a GPU, a model, journalctl or the real core CLI.  The real h2_official.load_manifest runs (only validate_manifest is stubbed for
the minimal fixture manifest, except in the test that proves the real validator is consulted).  The 87 core tests live in test_h2_official.py and
are unchanged."""
import ast
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_hold  # noqa: E402

CELL_IDS = ["A1", "B1", "B2", "A2", "A3", "B3"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(pathlib.Path(p).read_bytes())


def opt(argv, name):
    return argv[argv.index(name) + 1]


def opts(argv, name):
    out = []
    for tok in argv[argv.index(name) + 1:]:
        if tok.startswith("--"):
            break
        out.append(tok)
    return out


class FakeRunner:
    """Records the exact argv of every call and writes SYNTHETIC documents only."""

    def __init__(self, rig):
        self.rig = rig
        self.calls = []
        self.during = []
        self.env = []
        self.run_ids = []
        self.plan = {}      # (pair, cell) -> {"mode": ok|fail|signal|badjson|wrongjob|raise}
        self.judge = {}     # pair -> (verdict, errors, rc)
        self.raw_stdout = {}

    def __call__(self, argv):
        self.calls.append(list(argv))
        self.env.append(dict(os.environ))
        return {"run": self._run, "assemble-pair": self._assemble, "judge": self._judge}[argv[2]](argv)

    def _run(self, argv):
        pair, cell, out = int(opt(argv, "--pair")), int(opt(argv, "--cell")), opt(argv, "--out")
        cid = h2_hold.cell_id_for(pair, cell)
        self.run_ids.append(cid)
        binding = self.rig.prefix(pair, cell) + ".watcher-binding-before.json"
        self.during.append({"readlink": os.readlink(self.rig.link), "realpath": os.path.realpath(self.rig.link), "binding_exists": os.path.exists(binding),
                            "out_exists": os.path.lexists(out)})
        mode = self.plan.get((pair, cell), {}).get("mode", "ok")
        if mode == "raise":
            raise RuntimeError("fixture runner exploded")
        cell_json = os.path.join(out, cid, "cell.json")
        os.makedirs(os.path.dirname(cell_json))
        status = "OK" if mode in ("ok", "badjson", "wrongjob") else "EXIT_NONZERO"
        pathlib.Path(cell_json).write_text(json.dumps({"synthetic": True, "cell_id": cid, "status": status}) + "\n")
        if mode == "badjson":
            self.raw_stdout[(pair, cell)] = b"this is not json\n"
            return 0, self.raw_stdout[(pair, cell)], b"warn\n"
        doc = {"status": "CELL_OK" if status == "OK" else "CELL_FAILED", "cell_status": status, "failures": [] if status == "OK" else ["ABORTED_SIGNAL"],
               "pair": pair, "cell": cell, "cell_id": cid, "fabric_job": "other-job" if mode == "wrongjob" else os.environ["FABRIC_JOB"],
               "receipt_sha256": "ab" * 32, "cell_json": cell_json, "cell_file_sha256": sha_file(cell_json)}
        raw = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
        self.raw_stdout[(pair, cell)] = raw
        return {"ok": 0, "wrongjob": 0, "fail": 3, "signal": 4}[mode], raw, b""

    def _assemble(self, argv):
        out = opt(argv, "--out")
        pair = int(os.path.basename(out)[len("pair"):-len(".json")])
        pathlib.Path(out).write_text(json.dumps({"synthetic": True, "pair": pair, "status": "PAIR_COMPLETE"}) + "\n")
        return 0, (json.dumps({"status": "PAIR_COMPLETE", "pair": pair, "pair_json": out}) + "\n").encode(), b""

    def _judge(self, argv):
        out = opt(argv, "--out")
        n = len(opts(argv, "--pairs"))
        verdict, errors, rc = self.judge.get(n, ("PASS", [], 0) if n == 3 else ("UNKNOWN_INCOMPLETE", [], 3))
        pathlib.Path(out).write_text(json.dumps({"synthetic": True, "verdict": verdict, "errors": errors}) + "\n")
        return rc, (json.dumps({"verdict": verdict, "result": out, "errors": errors}) + "\n").encode(), b""


class Rig:
    def __init__(self, case, patch_validate=True):
        self.core = h2_hold.load_core()
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="rig-", dir=case.tmp))
        self.campaign = os.path.join(self.tmp, "campaign")
        self.wdir = os.path.join(self.tmp, "watch")
        self.outs = os.path.join(self.tmp, "outs")
        for d in (self.campaign, self.wdir, self.outs):
            os.mkdir(d)
        self.placeholder = os.path.join(self.wdir, "placeholder.log")
        pathlib.Path(self.placeholder).write_text("placeholder\n")
        self.link = os.path.join(self.wdir, "campaign-watcher.log")
        os.symlink(self.placeholder, self.link)
        manifest = {"label": "fixture", "harness": {"h2_official_sha256": sha_file(h2_hold.core_path())}, "watcher": {"log_path": self.link}}
        self.manifest = os.path.join(self.tmp, "manifest.json")
        pathlib.Path(self.manifest).write_bytes(json.dumps(manifest, sort_keys=True).encode())
        self.msha = sha_file(self.manifest)
        self.dsha = sha_file(os.path.realpath(h2_hold.__file__))
        self.env_before = None
        self._patcher = None
        if patch_validate:
            self._patcher = mock.patch.object(self.core, "validate_manifest", return_value=[])
            self._patcher.start()
            case.addCleanup(self.close)

    def close(self):
        """Undo the validate_manifest stub (idempotent). Each rig must unwind its own stub: a still-active stub from an earlier rig of the same
        test would otherwise mask the real validator in a later rig that asked for it."""
        if self._patcher is not None:
            self._patcher.stop()
            self._patcher = None

    def out(self, pair, cell):
        return os.path.join(self.outs, f"hold-{pair}{cell}")

    def prefix(self, pair, cell):
        return os.path.join(self.wdir, f"wrap-{pair}{cell}")

    def pair_file(self, pair):
        return os.path.join(self.campaign, f"pair{pair}.json")

    def cell_json(self, pair, cell):
        return os.path.join(self.out(pair, cell), h2_hold.cell_id_for(pair, cell), "cell.json")

    def argv_for(self, pair, cell, over=None, wrapper_log=True):
        """The held command's argv; creates the wrapper's <prefix>.dmesg.log the way the real wrapper does."""
        if wrapper_log:
            pathlib.Path(self.prefix(pair, cell) + ".dmesg.log").write_text("wrapper dmesg log\n")
        flags = [("--manifest", self.manifest), ("--expected-manifest-sha256", self.msha), ("--expected-dispatcher-sha256", self.dsha),
                 ("--pair", str(pair)), ("--cell", str(cell)), ("--out", self.out(pair, cell)), ("--watcher-prefix", self.prefix(pair, cell)),
                 ("--campaign", self.campaign)]
        over = over or {}
        argv = []
        for k, v in flags:
            argv += [k, over.get(k, v)]
        return argv

    def invoke(self, argv, runner, job=None, environ=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if environ is not None:
                rc = h2_hold.dispatch(argv, runner=runner, core=self.core, environ=environ)
            else:
                pair, cell = opt(argv, "--pair"), opt(argv, "--cell")
                with mock.patch.dict(os.environ, {"FABRIC_JOB": job or f"test-job-{pair}{cell}", "FABRIC_CARDS": "card0"}):
                    self.env_before = dict(os.environ)
                    rc = h2_hold.dispatch(argv, runner=runner, core=self.core)
        self.stdout = buf.getvalue()
        return rc

    def hold(self, pair, cell, runner, **kw):
        return self.invoke(self.argv_for(pair, cell), runner, **kw)

    def snapshot(self):
        items = []
        for root, dirs, files in os.walk(self.tmp, followlinks=False):
            for n in dirs + files:
                p = os.path.join(root, n)
                st = os.lstat(p)
                items.append((p, "L:" + os.readlink(p) if os.path.islink(p) else (st.st_size, st.st_mtime_ns)))
        return sorted(items)

    def stop(self):
        return json.loads(pathlib.Path(os.path.join(self.campaign, "STOP.json")).read_text())

    def expected_run(self, pair, cell, first_sha=None):
        argv = [sys.executable, h2_hold.core_path(), "run", "--manifest", self.manifest, "--expected-manifest-sha256", self.msha,
                "--pair", str(pair), "--cell", str(cell), "--out", self.out(pair, cell)]
        if pair > 1:
            argv += ["--previous"] + [self.pair_file(i) for i in range(1, pair)]
        if cell == 2:
            argv += ["--first-cell", self.cell_json(pair, 1), "--first-cell-sha256", first_sha]
        return argv

    def advance(self, runner, upto):
        """Run every hold of pairs 1..upto-1 successfully."""
        for pair in range(1, upto):
            for cell in (1, 2):
                assert self.hold(pair, cell, runner) == 0, self.stdout


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="h2hold-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def rig(self, **kw):
        r = Rig(self, **kw)
        r.runner = FakeRunner(r)
        return r


class SixHoldTests(Base):
    def test_abbaab_six_holds_one_run_each_exact_argv_and_chain(self):
        r = self.rig()
        fr = r.runner
        first_sha = {}
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("no spawn is allowed besides the injected runner")):
            for pair in (1, 2, 3):
                n0 = len(fr.calls)
                self.assertEqual(r.hold(pair, 1, fr), 0, r.stdout)
                self.assertEqual(len(fr.calls), n0 + 1, "cell 1 is exactly one core run: no assemble, no judge")
                first_sha[pair] = sha_file(r.cell_json(pair, 1))          # computed natively by the test
                self.assertEqual(fr.calls[n0], r.expected_run(pair, 1))
                self.assertFalse(os.path.lexists(r.pair_file(pair)))
                self.assertEqual(r.hold(pair, 2, fr), 0, r.stdout)
                self.assertEqual(len(fr.calls), n0 + 4, "cell 2 = one run + assemble-pair + judge")
                self.assertEqual(fr.calls[n0 + 1], r.expected_run(pair, 2, first_sha[pair]))
                second_sha = sha_file(r.cell_json(pair, 2))
                prev = [r.pair_file(i) for i in range(1, pair)]
                self.assertEqual(fr.calls[n0 + 2], [sys.executable, h2_hold.core_path(), "assemble-pair", "--manifest", r.manifest,
                                                    "--expected-manifest-sha256", r.msha, "--cells", r.cell_json(pair, 1), r.cell_json(pair, 2),
                                                    "--cell-sha256", first_sha[pair], second_sha] + (["--previous"] + prev if prev else [])
                                 + ["--out", r.pair_file(pair)])
                self.assertEqual(fr.calls[n0 + 3], [sys.executable, h2_hold.core_path(), "judge", "--manifest", r.manifest,
                                                    "--expected-manifest-sha256", r.msha, "--pairs"] + prev + [r.pair_file(pair)]
                                 + ["--out", os.path.join(r.campaign, f"judge-pair{pair}.json")])
        self.assertEqual(fr.run_ids, CELL_IDS)
        self.assertEqual([c[2] for c in fr.calls].count("run"), 6)
        self.assertEqual(len(fr.calls), 12)
        verdicts = [json.loads(pathlib.Path(os.path.join(r.campaign, f"judge-pair{p}.json")).read_text())["verdict"] for p in (1, 2, 3)]
        self.assertEqual(verdicts, ["UNKNOWN_INCOMPLETE", "UNKNOWN_INCOMPLETE", "PASS"])
        self.assertFalse(os.path.lexists(os.path.join(r.campaign, "STOP.json")))
        for pair in (1, 2, 3):
            for cell in (1, 2):
                self.assertTrue(os.path.isfile(r.prefix(pair, cell) + ".watcher-binding-before.json"))
        # the six holds carry six distinct actual job ids recorded in their ledgers
        jobs = [json.loads(pathlib.Path(h2_hold.ledger_path(r.campaign, p, c)).read_text())["fabric_job"] for p in (1, 2, 3) for c in (1, 2)]
        self.assertEqual(jobs, [f"test-job-{p}{c}" for p in (1, 2, 3) for c in (1, 2)])

    def test_cell1_ledger_carries_native_sha_and_cell2_uses_it(self):
        r = self.rig()
        self.assertEqual(r.hold(1, 1, r.runner), 0)
        led = json.loads(pathlib.Path(h2_hold.ledger_path(r.campaign, 1, 1)).read_text())
        self.assertEqual(led["cell_file_sha256"], sha_file(r.cell_json(1, 1)))
        self.assertEqual(r.hold(1, 2, r.runner), 0)
        run2 = r.runner.calls[1]
        self.assertEqual(opt(run2, "--first-cell-sha256"), sha_file(r.cell_json(1, 1)))

    def test_environment_is_inherited_unchanged_without_performance_pins(self):
        r = self.rig()
        self.assertEqual(r.hold(1, 1, r.runner, job="actual-fixture-job-77"), 0)
        seen = r.runner.env[0]
        self.assertEqual(seen, r.env_before)
        self.assertEqual(seen["FABRIC_JOB"], "actual-fixture-job-77")
        added = set(seen) - set(r.env_before)
        self.assertFalse(added)
        self.assertEqual(r.runner.calls[0].count("run"), 1)


class StopTests(Base):
    def test_early_order_statistic_fail_stops_and_blocks_next_without_spawn(self):
        r = self.rig()
        fr = r.runner
        fr.judge[2] = ("FAIL_EARLY_ORDER_STATISTIC", [], 1)
        r.advance(fr, 2)
        self.assertEqual(r.hold(2, 1, fr), 0)
        self.assertEqual(r.hold(2, 2, fr), 1)
        stop = r.stop()
        self.assertEqual((stop["stage"], stop["verdict"], stop["exit_code"], stop["pair"]), ("judge", "FAIL_EARLY_ORDER_STATISTIC", 1, 2))
        self.assertTrue(stop["no_automatic_retry"])
        n, link = len(fr.calls), os.readlink(r.link)
        argv = r.argv_for(3, 1)
        snap = r.snapshot()
        self.assertEqual(r.invoke(argv, fr), 2)
        self.assertEqual(len(fr.calls), n, "a stopped campaign spawns nothing")
        self.assertEqual(os.readlink(r.link), link)
        self.assertEqual(r.snapshot(), snap)
        self.assertIn("STOP.json", r.stdout)

    def test_judge_fail_after_three_pairs_stops_with_exit_1(self):
        r = self.rig()
        r.runner.judge[3] = ("FAIL", [], 1)
        r.advance(r.runner, 3)
        self.assertEqual(r.hold(3, 1, r.runner), 0)
        self.assertEqual(r.hold(3, 2, r.runner), 1)
        self.assertEqual(r.stop()["verdict"], "FAIL")

    def test_unacceptable_judge_outcomes_stop_with_exit_3(self):
        cases = [("unknown with errors", ("UNKNOWN_INCOMPLETE", ["bad binding"], 3)),
                 ("invalid", ("UNKNOWN_INVALID", ["e"], 3)),
                 ("pass before pair 3", ("PASS", [], 0)),
                 ("return code disagrees with verdict", ("UNKNOWN_INCOMPLETE", [], 0)),
                 ("unknown verdict", ("WHATEVER", [], 3))]
        for name, outcome in cases:
            with self.subTest(name):
                r = self.rig()
                r.runner.judge[1] = outcome
                self.assertEqual(r.hold(1, 1, r.runner), 0)
                self.assertEqual(r.hold(1, 2, r.runner), 3)
                self.assertEqual(r.stop()["stage"], "judge")
                n = len(r.runner.calls)
                self.assertEqual(r.hold(2, 1, r.runner), 2)
                self.assertEqual(len(r.runner.calls), n)

    def test_failed_run_stops_preserves_raw_and_receipt_and_blocks_cell2(self):
        r = self.rig()
        fr = r.runner
        fr.plan[(1, 1)] = {"mode": "fail"}
        self.assertEqual(r.hold(1, 1, fr), 3)
        stop = r.stop()
        cell_json = r.cell_json(1, 1)
        self.assertEqual((stop["stage"], stop["core_returncode"], stop["exit_code"]), ("run", 3, 3))
        self.assertEqual(stop["receipt"], {"cell_json": cell_json, "present": True, "sha256": sha_file(cell_json)})
        self.assertEqual(pathlib.Path(stop["raw"]["stdout"]).read_bytes(), fr.raw_stdout[(1, 1)])
        self.assertTrue(os.path.isfile(stop["raw"]["stderr"]))
        self.assertEqual(stop["argv"], r.expected_run(1, 1))
        self.assertFalse(os.path.lexists(h2_hold.ledger_path(r.campaign, 1, 1)), "a failed first cell advances nothing")
        n = len(fr.calls)
        self.assertEqual(n, 1, "no automatic retry")
        self.assertEqual(r.hold(1, 2, fr), 2)
        self.assertEqual(len(fr.calls), n)

    def test_signal_abort_returns_4_and_stops(self):
        r = self.rig()
        r.runner.plan[(1, 1)] = {"mode": "signal"}
        self.assertEqual(r.hold(1, 1, r.runner), 4)
        self.assertEqual(r.stop()["stage"], "run")

    def test_non_json_or_mismatching_core_stdout_stops(self):
        for mode in ("badjson", "wrongjob"):
            with self.subTest(mode):
                r = self.rig()
                r.runner.plan[(1, 1)] = {"mode": mode}
                self.assertEqual(r.hold(1, 1, r.runner), 3)
                stop = r.stop()
                self.assertEqual(stop["stage"], "run")
                self.assertEqual(pathlib.Path(stop["raw"]["stdout"]).read_bytes(), r.runner.raw_stdout[(1, 1)])
                self.assertFalse(os.path.lexists(h2_hold.ledger_path(r.campaign, 1, 1)))

    def test_runner_exception_after_admission_closes_campaign_and_keeps_evidence(self):
        r = self.rig()
        r.runner.plan[(1, 1)] = {"mode": "raise"}
        self.assertEqual(r.hold(1, 1, r.runner), 3)
        stop = r.stop()
        self.assertEqual(stop["stage"], "harness_exception")
        self.assertIn("fixture runner exploded", stop["reason"])
        self.assertEqual(stop["receipt"]["present"], False)
        self.assertTrue(os.path.isfile(r.prefix(1, 1) + ".watcher-binding-before.json"), "the binding record is never cleaned up")
        self.assertEqual(r.hold(1, 1, r.runner), 2)


class RefusalTests(Base):
    """Every refusal exits 2 BEFORE any mutation: no file created, the watcher link untouched, no runner call."""

    def check(self, name, prepare, needle=None):
        with self.subTest(name), contextlib.closing(self.rig(**getattr(prepare, "rig_kw", {}))) as r:
            spec = prepare(r) or {}
            pair, cell = spec.get("pair", 1), spec.get("cell", 1)
            argv = r.argv_for(pair, cell, spec.get("over"), wrapper_log=spec.get("wrapper_log", True))
            if spec.get("after_argv"):
                spec["after_argv"](r)
            n, snap = len(r.runner.calls), r.snapshot()
            rc = r.invoke(argv, r.runner, environ=spec.get("environ"))
            self.assertEqual(rc, 2, r.stdout)
            self.assertEqual(len(r.runner.calls), n)
            self.assertEqual(r.snapshot(), snap)
            self.assertIn('"mutated": false', r.stdout)
            if needle:
                self.assertIn(needle, r.stdout)

    def test_refusals(self):
        def run_cell1(r, pair=1):
            assert r.hold(pair, 1, r.runner) == 0

        def no_job(r):
            return {"environ": {"PATH": "/usr/bin"}}

        def bad_msha(r):
            return {"over": {"--expected-manifest-sha256": "0" * 64}}

        def bad_dsha(r):
            return {"over": {"--expected-dispatcher-sha256": "0" * 64}}

        def regular_link(r):
            os.unlink(r.link)
            pathlib.Path(r.link).write_text("a regular file\n")

        def chained_link(r):
            other = os.path.join(r.wdir, "other.log")
            pathlib.Path(other).write_text("o\n")
            os.unlink(r.placeholder)
            os.symlink(other, r.placeholder)

        def missing_link(r):
            os.unlink(r.link)

        def out_exists(r):
            os.mkdir(r.out(1, 1))

        def stop_present(r):
            pathlib.Path(os.path.join(r.campaign, "STOP.json")).write_text("{}\n")

        def binding_exists(r):
            pathlib.Path(r.prefix(1, 1) + ".watcher-binding-before.json").write_text("{}\n")

        def no_wrapper_log(r):
            return {"wrapper_log": False}

        def wrapper_log_symlink(r):
            def after(r):
                p = r.prefix(1, 1) + ".dmesg.log"
                os.unlink(p)
                os.symlink(r.placeholder, p)
            return {"after_argv": after}

        def out_in_campaign(r):
            return {"over": {"--out": os.path.join(r.campaign, "inside")}}

        def out_is_watcher_dir(r):
            return {"over": {"--out": r.wdir}}

        def relative_manifest(r):
            return {"over": {"--manifest": "manifest.json"}}

        def no_campaign(r):
            return {"over": {"--campaign": os.path.join(r.tmp, "nope")}}

        def no_previous_pair(r):
            return {"pair": 2, "cell": 1}

        def incomplete_previous_pair(r):
            pathlib.Path(r.pair_file(1)).write_text(json.dumps({"pair": 1, "status": "PAIR_INCOMPLETE"}) + "\n")
            return {"pair": 2, "cell": 1}

        def previous_pair_is_symlink(r):
            real = os.path.join(r.tmp, "real-pair1.json")
            pathlib.Path(real).write_text(json.dumps({"pair": 1, "status": "PAIR_COMPLETE"}) + "\n")
            os.symlink(real, r.pair_file(1))
            return {"pair": 2, "cell": 1}

        def cell2_without_first(r):
            return {"pair": 1, "cell": 2}

        def tampered_first(r):
            run_cell1(r)
            with open(r.cell_json(1, 1), "ab") as f:
                f.write(b" \n")
            return {"pair": 1, "cell": 2}

        def first_cell_missing(r):
            run_cell1(r)
            os.unlink(r.cell_json(1, 1))
            return {"pair": 1, "cell": 2}

        def cell2_reuses_first_out(r):
            run_cell1(r)
            return {"pair": 1, "cell": 2, "over": {"--out": r.out(1, 1)}}

        def cell2_out_inside_first_out(r):
            run_cell1(r)
            return {"pair": 1, "cell": 2, "over": {"--out": os.path.join(r.out(1, 1), "again")}}

        def pair_output_exists(r):
            pathlib.Path(r.pair_file(1)).write_text("{}\n")

        def judge_output_exists(r):
            pathlib.Path(os.path.join(r.campaign, "judge-pair1.json")).write_text("{}\n")

        def ledger_exists(r):
            pathlib.Path(h2_hold.ledger_path(r.campaign, 1, 1)).write_text("{}\n")

        def raw_output_exists(r):
            pathlib.Path(h2_hold.raw_paths(r.campaign, 1, 1, "run")[0]).write_text("{}\n")

        def real_validator(r):
            return {}
        real_validator.rig_kw = {"patch_validate": False}

        self.check("missing FABRIC_JOB", no_job, "FABRIC_JOB")
        self.check("bad manifest sha", bad_msha, "independently supplied")
        self.check("bad dispatcher sha", bad_dsha, "dispatcher source sha256")
        self.check("watcher path is a regular file", regular_link, "not a symlink")
        self.check("watcher path is a chain of symlinks", chained_link, "chain of symlinks")
        self.check("watcher path missing", missing_link, "watcher.log_path")
        self.check("out exists", out_exists, "already exists")
        self.check("campaign STOP.json present", stop_present, "campaign is stopped")
        self.check("binding file reused", binding_exists, "binding file")
        self.check("wrapper watcher log missing", no_wrapper_log, "wrapper watcher log")
        self.check("wrapper watcher log is a symlink", wrapper_log_symlink, "not a regular file")
        self.check("out nested in campaign", out_in_campaign, "nested")
        self.check("out contains the watcher dir", out_is_watcher_dir, "nested")
        self.check("relative manifest path", relative_manifest, "canonical absolute")
        self.check("campaign missing", no_campaign, "campaign")
        self.check("pair 2 without pair1.json", no_previous_pair, "previous pair 1")
        self.check("previous pair not complete", incomplete_previous_pair, "PAIR_COMPLETE")
        self.check("previous pair is a symlink", previous_pair_is_symlink, "regular file")
        self.check("cell 2 without cell 1", cell2_without_first, "ledger")
        self.check("first cell tampered", tampered_first, "sha256")
        self.check("first cell file missing", first_cell_missing, "first cell file")
        self.check("cell 2 reuses cell 1 OUT", cell2_reuses_first_out, "already exists")
        self.check("cell 2 OUT inside cell 1 OUT", cell2_out_inside_first_out, "nested")
        self.check("pair output exists", pair_output_exists, "already exists")
        self.check("judge output exists", judge_output_exists, "already exists")
        self.check("ledger exists", ledger_exists, "already exists")
        self.check("raw output exists", raw_output_exists, "already exists")
        self.check("real validator is consulted", real_validator, "manifest invalid")

    def test_argparse_requires_every_flag(self):
        r = self.rig()
        full = r.argv_for(1, 1)
        for flag in ("--manifest", "--expected-manifest-sha256", "--expected-dispatcher-sha256", "--pair", "--cell", "--out", "--watcher-prefix",
                     "--campaign"):
            with self.subTest(flag):
                argv = list(full)
                i = argv.index(flag)
                del argv[i:i + 2]
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                    h2_hold.dispatch(argv, runner=r.runner, core=r.core, environ={"FABRIC_JOB": "x"})
                self.assertEqual(cm.exception.code, 2)
        self.assertEqual(r.runner.calls, [])


class WatcherTests(Base):
    def test_atomic_repoint_only_inside_held_call_and_binding_records_actual_job(self):
        r = self.rig()
        before = os.readlink(r.link)
        self.assertEqual(before, r.placeholder)
        argv = r.argv_for(1, 1)
        self.assertEqual(os.readlink(r.link), before, "nothing is repointed outside the held call")
        self.assertEqual(r.invoke(argv, r.runner, job="fixture-actual-job-9"), 0)
        log = r.prefix(1, 1) + ".dmesg.log"
        during = r.runner.during[0]
        self.assertEqual((during["readlink"], during["realpath"], during["binding_exists"], during["out_exists"]), (log, log, True, False))
        self.assertEqual(os.readlink(r.link), log)
        binding = json.loads(pathlib.Path(r.prefix(1, 1) + ".watcher-binding-before.json").read_text())
        st = os.stat(log)
        self.assertEqual(binding["fabric_job"], "fixture-actual-job-9")
        self.assertEqual(binding["target"], log)
        self.assertEqual(binding["readlink"], log)
        self.assertEqual(binding["realpath"], log)
        self.assertEqual(binding["previous_target"], before)
        self.assertEqual(binding["target_stat"]["inode"], st.st_ino)
        self.assertEqual(binding["target_stat"]["device"], st.st_dev)
        self.assertEqual(binding["target_stat"]["size_bytes"], st.st_size)
        self.assertTrue(binding["utc"].endswith("+00:00"))
        self.assertEqual((binding["pair"], binding["cell"], binding["cell_id"]), (1, 1, "A1"))
        self.assertEqual(binding["dispatcher_sha256"], r.dsha)
        self.assertEqual(binding["manifest_sha256"], r.msha)
        self.assertFalse([n for n in os.listdir(r.wdir) if n.endswith(".tmp")], "the owned temp symlink never lingers")
        # the next held command repoints again to its OWN wrapper log; the earlier log is untouched
        self.assertEqual(r.hold(1, 2, r.runner), 0)
        self.assertEqual(os.readlink(r.link), r.prefix(1, 2) + ".dmesg.log")
        self.assertEqual(pathlib.Path(log).read_text(), "wrapper dmesg log\n")

    def test_no_second_watcher_or_journal_or_sampler(self):
        r = self.rig()
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("spawn")), mock.patch.object(subprocess, "run", side_effect=AssertionError("spawn")), \
                mock.patch.object(os, "system", side_effect=AssertionError("spawn")):
            self.assertEqual(r.hold(1, 1, r.runner), 0)
            self.assertEqual(r.hold(1, 2, r.runner), 0)
        for argv in r.runner.calls:
            self.assertIn(argv[2], ("run", "assemble-pair", "judge"))
            self.assertEqual(argv[:2], [sys.executable, h2_hold.core_path()])


class SourceTests(Base):
    def test_core_is_the_one_beside_the_dispatcher_and_order_matches(self):
        core = h2_hold.load_core()
        here = os.path.dirname(os.path.realpath(h2_hold.__file__))
        self.assertEqual(os.path.realpath(core.__file__), os.path.join(here, "h2_official.py"))
        self.assertEqual(h2_hold.core_path(), os.path.join(here, "h2_official.py"))
        self.assertEqual({k: tuple(v) for k, v in core.ORDER.items()}, h2_hold.ORDER)
        self.assertEqual(list(core.GLOBAL_ORDER), CELL_IDS)

    def test_source_has_no_env_overrides_spawn_options_or_environment_writes(self):
        src = pathlib.Path(os.path.realpath(h2_hold.__file__)).read_text()
        tree = ast.parse(src)
        doc_node = tree.body[0].value
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n is not doc_node]
        self.assertFalse([s for s in strings if s.startswith(("HIPFIRE_", "HIP_", "ROCR_", "HSA_", "GPU_", "AMD_"))])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertNotIn("start_new_session", [k.arg for k in node.keywords])
                self.assertNotIn("env", [k.arg for k in node.keywords], "the child inherits the fabric environment unchanged")
                self.assertNotIn("preexec_fn", [k.arg for k in node.keywords])
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.assertNotEqual(ast.unparse(node.value), "os.environ")
        self.assertNotIn("setsid", src)
        self.assertNotIn("killall", src)
        self.assertNotIn("pkill", src)

    def test_dispatcher_hash_guard_reads_its_own_bytes(self):
        r = self.rig()
        self.assertEqual(r.dsha, sha_file(os.path.realpath(h2_hold.__file__)))
        self.assertEqual(r.hold(1, 1, r.runner), 0)


class ProductionRunnerTests(Base):
    """Real (CPU-only) child processes; the child is a trivial python -c, never the core."""

    def test_child_shares_group_and_session_and_inherits_env(self):
        code = "import os;print(os.getpgrp(), os.getsid(0), os.environ.get('H2HOLD_PROBE'))"
        with mock.patch.dict(os.environ, {"H2HOLD_PROBE": "inherited"}):
            rc, out, err = h2_hold.run_subprocess([sys.executable, "-c", code])
        self.assertEqual(rc, 0)
        pgid, sid, probe = out.decode().split()
        self.assertEqual((int(pgid), int(sid), probe), (os.getpgrp(), os.getsid(0), "inherited"))

    def test_returns_nonzero_exit_and_both_streams_untouched(self):
        code = "import sys;sys.stdout.write('o\\x00k');sys.stderr.write('e');sys.exit(3)"
        rc, out, err = h2_hold.run_subprocess([sys.executable, "-c", code])
        self.assertEqual((rc, out, err), (3, b"o\x00k", b"e"))

    def test_sigterm_is_forwarded_to_own_child_and_the_child_is_awaited(self):
        old = signal.getsignal(signal.SIGTERM)
        timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
        self.addCleanup(timer.cancel)
        timer.start()
        rc, out, err = h2_hold.run_subprocess([sys.executable, "-c", "import time;time.sleep(60)"])
        self.assertEqual(rc, -signal.SIGTERM)
        self.assertIs(signal.getsignal(signal.SIGTERM), old, "the previous handler is restored")


if __name__ == "__main__":
    unittest.main()
