#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for q1_hold.py.  Run:  python3 test_q1_hold.py     (needs the frozen h2_official.py: env Q1_H2_HELPER or the default path)

The native binary is FAKED: a tiny launcher script (executable) calls test_q1_judge.fake_native_main, which asserts the launch class the real native
asserts (environment, selector, executable sha, model stat) and writes SYNTHETIC tiny dumps.  The real controller, the real q1_judge and the real
frozen h2 helper run unchanged.  A fake sysfs card and a fake watcher log stand in for the hardware; there is no GPU, no model payload, no journalctl,
no production environment backend, and the controller's working directory is never a source tree."""
import contextlib
import io
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1_hold as Q  # noqa: E402
import q1_judge as J  # noqa: E402
import test_q1_judge as T  # noqa: E402

H2_PATH = os.environ.get("Q1_H2_HELPER", "/home/kaden/qcal/release-0.4.3/slowlink-prefill/holds/harness-h2-r4-prep/scripts/slowlink-bench/h2_official.py")
FROZEN_H2_SHA = "97c4509b9313c5d29ac124208b06b24a54791e130c195e6bf63e98b5f969813e"
BDF = "0000:67:00.0"
UUID = "GPU-43390a851e296ee5"


def sha(data):
    return T.sha(data)


def sha_file(path):
    return sha(pathlib.Path(path).read_bytes())


class SkewHost(Q.Host):
    """Adds `skew_ns` to the monotonic clock once `trigger_path` exists (i.e. once the check-plan step has closed): deterministic budget exhaustion."""

    def __init__(self, environ, trigger_path, skew_ns):
        super().__init__(environ)
        self.trigger_path, self.skew_ns = trigger_path, skew_ns

    def monotonic_ns(self):
        base = time.monotonic_ns()
        if os.path.exists(self.trigger_path):
            return base + self.skew_ns
        return base


class Rig:
    """A complete synthetic input world: fake card, model + PLE sidecar with cached receipts, fake native binary, read-only kernel pack, plan,
    watcher log and manifests."""

    def __init__(self, root):
        self.root = root
        for d in ("work", "logs", "models", "bin", "src", "receipts", f"sys/{BDF}"):
            os.makedirs(os.path.join(root, d))
        sysd = os.path.join(root, "sys", BDF)
        for name, text in (("vendor", "0x1002\n"), ("device", "0x744c\n"), ("unique_id", UUID.removeprefix("GPU-") + "\n"),
                           ("mem_info_vram_used", "0\n"), ("mem_info_vram_total", "25753026560\n")):
            with open(os.path.join(sysd, name), "w") as f:
                f.write(text)
        self.model = os.path.join(root, "models", "model.mq4")
        self.ple = self.model + ".ple-q4"
        with open(self.model, "wb") as f:
            f.write(b"M" * 1234)
        with open(self.ple, "wb") as f:
            f.write(b"P" * 77)
        self.cached = []
        for path, digest, name in ((self.model, T.MODEL_SHA, "model.json"), (self.ple, sha(b"ple-pin"), "ple.json")):
            st = os.stat(path)
            rec = {"path": path, "size_bytes": st.st_size, "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns,
                   "ctime_ns": st.st_ctime_ns, "sha256": digest}
            rp = os.path.join(root, "receipts", name)
            with open(rp, "w") as f:
                json.dump(rec, f)
            self.cached.append({**rec, "cached_receipt_path": rp})
        self.binary = os.path.join(root, "bin", "native")
        with open(self.binary, "w") as f:
            f.write(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {HERE!r})\nimport test_q1_judge as T\nsys.exit(T.fake_native_main(sys.argv[1:]))\n")
        os.chmod(self.binary, 0o755)
        self.pack = os.path.join(root, "bin", "kernels", "compiled", "gfx1100")
        os.makedirs(self.pack)
        for name in ("k1.hsaco", "k2.hsaco", "index.json"):
            with open(os.path.join(self.pack, name), "wb") as f:
                f.write(("pack-" + name).encode() * 10)
            os.chmod(os.path.join(self.pack, name), 0o444)
        os.chmod(self.pack, 0o555)
        self.source = os.path.join(root, "src", "qwen4_slowlink_quality.rs")
        with open(self.source, "w") as f:
            f.write("// synthetic native source\n")
        self.plan_obj = T.make_plan(self.model, 1234, T.MODEL_SHA)
        self.plan = os.path.join(root, "plan.json")
        self.plan_sha = T.write_plan_file(self.plan, self.plan_obj)
        self.watch = os.path.join(root, "wrapper.dmesg.log")
        pathlib.Path(self.watch).write_bytes(b"")
        self.worktree = os.path.join(root, "wt")
        self.n = 0

    def cleanup(self):
        for dp, dn, fn in os.walk(self.root):
            os.chmod(dp, 0o755)
        shutil.rmtree(self.root, True)

    def artifact(self, path):
        with open(path, "rb") as f:
            data = f.read()
        return {"path": path, "size_bytes": len(data), "sha256": sha(data)}

    def manifest_obj(self, part="a", prior=None, caps=None, maximum=None, min_phase=None):
        bsha = sha_file(self.binary)
        ssha = sha_file(self.source)
        arts = [self.artifact(self.binary), self.artifact(self.plan), self.artifact(self.source)]
        arts += [self.artifact(os.path.join(self.pack, n)) for n in sorted(os.listdir(self.pack))]
        default_caps = {"a": {"baseline": 30, "s1a": 30, "s1b": 30}, "f32": {"beta-f32": 30, "s1-f32": 30}}[part]
        m = {
            "schema": Q.SCHEMA_MANIFEST, "label": f"q1 synthetic part {part}", "part": part,
            "identity": {"runtime": {"commit": "40a4e3bee206a2bd96867b8fbcdec770fa2f0ce5", "tree": "c8321cbd114cba12dae66445ecf7157471568647"}},
            "harness": {"q1_hold_sha256": sha_file(os.path.realpath(Q.__file__)), "q1_judge_sha256": sha_file(os.path.join(HERE, "q1_judge.py")),
                        "h2_official_sha256": sha_file(H2_PATH)},
            "binary": self.artifact(self.binary), "plan": self.artifact(self.plan), "native_source": self.artifact(self.source),
            "native_manifest_template": {"binary": {"sha256": bsha, "path": self.binary},
                                         "source": {"example_sha256": ssha, "runtime_commit": "40a4e3bee206a2bd96867b8fbcdec770fa2f0ce5",
                                                    "worktree": self.worktree},
                                         "model": {"path": self.model, "size": 1234, "sha256": T.MODEL_SHA}},
            "target": {"card": "xtx", "uuid": UUID, "bdf": BDF, "vendor_id": "0x1002", "device_id": "0x744c", "model_path": self.model, "model_size": 1234,
                       "models_dir": os.path.dirname(self.model),
                       "vram_used_path": os.path.join(self.root, "sys", BDF, "mem_info_vram_used"),
                       "vram_total_path": os.path.join(self.root, "sys", BDF, "mem_info_vram_total"),
                       "visible_devices_env": {"HIPFIRE_DEVICES": UUID, "ROCR_VISIBLE_DEVICES": UUID, "HIP_VISIBLE_DEVICES": "0"},
                       "absent_paths": [self.model + ".expert-hotset", self.model + ".trunk-overlay"]},
            "cached_inputs": self.cached, "artifacts": arts, "pack_dirs": [self.pack],
            "class_env": json.loads(json.dumps(Q.CLASS_ENV)),
            "budget": {"phase_cap_seconds": caps or default_caps, "kill_grace_seconds": 2, "drain_seconds": 3, "cpu_cap_seconds": 30,
                       "finalize_seconds": 5, "preflight_seconds": 5, "min_phase_seconds": min_phase or 1, "maximum_hold_seconds": maximum or 120},
            "watcher": {"log_path": self.watch, "fatal_regex": "(?i)GPU reset|ring .*timeout"},
            "oom_stderr_regex": "(?i)out of memory|hipErrorOutOfMemory",
        }
        if prior:
            m["prior_q1a"] = {"summary_path": prior[0], "summary_sha256": prior[1]}
        return m

    def write_manifest(self, m):
        self.n += 1
        path = os.path.join(self.root, f"manifest-{self.n}.json")
        data = json.dumps(m, indent=1).encode()
        with open(path, "wb") as f:
            f.write(data)
        return path, sha(data)

    def argv(self, cmd, manifest, out=None, part="a"):
        mpath, msha = manifest
        a = [cmd, "--part", part, "--manifest", mpath, "--expected-manifest-sha256", msha,
             "--expected-controller-sha256", sha_file(os.path.realpath(Q.__file__)), "--expected-judge-sha256", sha_file(os.path.join(HERE, "q1_judge.py")),
             "--h2-helper", H2_PATH, "--expected-h2-sha256", sha_file(H2_PATH), "--expected-binary-sha256", sha_file(self.binary),
             "--expected-plan-sha256", self.plan_sha, "--expected-native-source-sha256", sha_file(self.source)]
        if out:
            a += ["--out", out]
        return a

    def env(self, name, fabric=True, fail="", knobs=None):
        e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "12", "HIPFIRE_POLLUTION": "1", "HIP_FOO": "x",
             "ROC_BAR": "y", "Q1FAKE_LOG": os.path.join(self.root, "logs", name + ".log"), "Q1FAKE_FAIL": fail, "Q1FAKE_KNOBS": json.dumps(knobs or {}),
             "Q1FAKE_WATCHER_LOG": self.watch, "Q1FAKE_TAMPER_FILE": self.source}
        if fabric:
            e.update({"FABRIC_JOB": "job-q1", "FABRIC_CARDS": "xtx"})
        return e

    def phases(self, name):
        p = os.path.join(self.root, "logs", name + ".log")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [json.loads(line)["phase"] for line in f if line.strip()]

    def pids(self, name):
        with open(os.path.join(self.root, "logs", name + ".log")) as f:
            return {json.loads(line)["phase"]: json.loads(line)["pid"] for line in f if line.strip()}


def run_main(argv, host):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = Q.main(argv, host)
    return rc, buf.getvalue()


def jload(path):
    with open(path) as f:
        return json.load(f)


class HoldBase(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="q1h-"))
        self.rig = Rig(self.root)
        self.addCleanup(self.rig.cleanup)
        self.addCleanup(Q.STOP.clear)
        self.m_a = self.rig.write_manifest(self.rig.manifest_obj("a"))

    def out(self, name):
        return os.path.join(self.root, "work", name)

    def run_a(self, name, fail="", knobs=None, fabric=True, manifest=None, host=None, part="a"):
        manifest = manifest or self.m_a
        host = host or Q.Host(self.rig.env(name, fabric, fail, knobs))
        rc, text = run_main(self.rig.argv("run", manifest, self.out(name), part), host)
        return rc, text

    def rj(self, name, rel):
        return jload(os.path.join(self.out(name), rel))


class HappyPath(HoldBase):
    def test_part_a_end_to_end(self):
        rc, text = self.run_a("o1")
        self.assertEqual(rc, 0, text[-3000:])
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline", "s1a", "score", "s1b"])
        res = self.rj("o1", "hold-result.json")
        self.assertEqual(res["verdict"], "PASS", res["reasons"])
        self.assertFalse(res["default_on_recommended"])
        self.assertTrue(res["processes_distinct"])
        self.assertTrue(J.seal_ok(res))
        self.assertLess(text.index("RESUME_READY"), text.index("HOLD_CLOSED"))
        events = [json.loads(line) for line in pathlib.Path(os.path.join(self.out("o1"), "events.jsonl")).read_text().splitlines() if line.strip()]
        names = [e["event"] for e in events]
        s1b_start = next(i for i, e in enumerate(events) if e["event"] == "phase_start" and e.get("phase") == "s1b")
        self.assertLess(names.index("resume_emitted"), s1b_start)
        final = self.rj("o1", "q1a-summary.json")
        self.assertEqual((final["stage"], final["verdict"]), ("final", "PASS"))
        self.assertEqual(res["files"]["q1a_summary"]["sha256"], sha_file(os.path.join(self.out("o1"), "q1a-summary.json")))
        self.assertTrue(os.path.exists(os.path.join(self.out("o1"), "determinism.json")))
        resume = self.rj("o1", "resume.json")
        self.assertEqual(resume["stage"], "resume")
        self.assertEqual(resume["determinism_status"], "PENDING_S1B")
        self.assertEqual(resume["resume_disposition"], "WAIT_F32_REVIEW")
        self.assertEqual(resume["overall_S1"], "BLOCKED_F32_REVIEW")
        self.assertFalse(resume["qualification_complete"])

    def test_one_fresh_model_process_per_gpu_phase_with_private_cwd(self):
        rc, _ = self.run_a("o1")
        self.assertEqual(rc, 0)
        runs = {}
        for ph in ("baseline", "s1a", "s1b"):
            runs[ph] = self.rj("o1", f"phases/{ph}/fake-run.json")
            self.assertEqual(os.path.realpath(runs[ph]["cwd"]), os.path.join(self.out("o1"), "runs", ph))
            self.assertEqual(runs[ph]["pgid"], runs[ph]["sid"])           # own session / process group
            self.assertEqual(runs[ph]["pgid"], runs[ph]["pid"])
            self.assertTrue(runs[ph]["home"].startswith(os.path.join(self.out("o1"), "runs", ph)))
            self.assertFalse(os.path.realpath(runs[ph]["cwd"]).startswith(HERE))
            self.assertFalse(os.path.realpath(runs[ph]["cwd"]).startswith(self.rig.worktree))
        self.assertEqual(len({r["pid"] for r in runs.values()}), 3)
        log = [p for p in self.rig.phases("o1") if p in ("baseline", "s1a", "s1b")]
        self.assertEqual(log, ["baseline", "s1a", "s1b"])                  # each GPU phase spawned exactly once, no retry

    def test_class_environment_per_route(self):
        rc, _ = self.run_a("o1")
        self.assertEqual(rc, 0)
        base = self.rj("o1", "runs/baseline/env.json")
        s1a = self.rj("o1", "runs/s1a/env.json")
        s1b = self.rj("o1", "runs/s1b/env.json")
        for k, v in T.PINNED.items():
            self.assertEqual(base["set"][k], v)
            self.assertEqual(s1a["set"][k], v)
        self.assertEqual(base["set"]["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"], "9")
        self.assertEqual(base["set"]["HIPFIRE_PREFILL_CHUNK_ROWS"], "4096")
        self.assertEqual(base["set"][T.SELECTOR_ENV], "0")
        for e in (s1a, s1b):
            self.assertNotIn("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", e["set"])
            self.assertNotIn("HIPFIRE_PREFILL_CHUNK_ROWS", e["set"])
            self.assertEqual(e["set"][T.SELECTOR_ENV], "1")
        for e in (base, s1a, s1b):
            for stripped in ("HIPFIRE_POLLUTION", "HIP_FOO", "ROC_BAR", "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"):
                self.assertIn(stripped, e["unset"])
            self.assertEqual(e["fabric"]["job"], "job-q1")
            self.assertEqual(e["set"]["HIPFIRE_DEVICES"], UUID)
            self.assertEqual(e["set"]["HIP_VISIBLE_DEVICES"], "0")

    def test_native_manifest_and_receipts_carry_real_identity(self):
        rc, _ = self.run_a("o1")
        self.assertEqual(rc, 0)
        nm = self.rj("o1", "runs/s1a/native-manifest.json")
        self.assertEqual(nm["schema"], Q.SCHEMA_NATIVE_MANIFEST)
        st = os.stat(self.rig.model)
        self.assertEqual(nm["model"]["mtime_sec"] * 10 ** 9 + nm["model"]["mtime_nsec"], st.st_mtime_ns)
        self.assertEqual((nm["model"]["dev"], nm["model"]["ino"], nm["model"]["size"]), (st.st_dev, st.st_ino, st.st_size))
        self.assertEqual(nm["model"]["sha256"], T.MODEL_SHA)
        self.assertEqual(nm["binary"]["sha256"], sha_file(self.rig.binary))
        self.assertEqual(nm["launch"]["fabric"]["job"], "job-q1")
        self.assertIsNone(nm["launch"]["frontend_scheduler_id"])
        self.assertEqual(nm["launch"]["phase"], "s1a")
        rec = self.rj("o1", "receipts/s1a.json")
        self.assertTrue(rec["ok"])
        self.assertTrue(rec["process"]["start_new_session"])
        self.assertEqual(rec["integrity"]["changed_sections"], [])
        self.assertEqual(rec["caps"]["bound_by"], "phase")
        self.assertIsNotNone(rec["drain"])
        self.assertIsNotNone(rec["watcher"]["identity"])
        self.assertEqual(rec["native_phase"]["complete"], True)
        self.assertEqual(rec["fabric"]["job"], "job-q1")

    def test_score_runs_q8_only_before_s1b(self):
        rc, _ = self.run_a("o1")
        self.assertEqual(rc, 0)
        cmd = pathlib.Path(os.path.join(self.out("o1"), "runs", "score", "command.txt")).read_text()
        self.assertIn("--baseline-dir", cmd)
        self.assertIn("--s1a-dir", cmd)
        self.assertNotIn("--s1b-dir", cmd)
        self.assertNotIn("--beta-f32-dir", cmd)

    def test_frontend_scheduler_id_is_recorded_when_given(self):
        host = Q.Host(self.rig.env("o1"))
        argv = self.rig.argv("run", self.m_a, self.out("o1")) + ["--frontend-scheduler-id", "frontend-77"]
        rc, _ = run_main(argv, host)
        self.assertEqual(rc, 0)
        self.assertEqual(self.rj("o1", "hold-result.json")["frontend_scheduler_id"], "frontend-77")
        self.assertEqual(self.rj("o1", "runs/baseline/native-manifest.json")["launch"]["frontend_scheduler_id"], "frontend-77")

    def test_primary_numeric_fail_still_runs_fresh_s1b_and_final(self):
        rc, _ = self.run_a("o1", knobs={"kl_const": {"primary-ctx16k": 0.0, "primary-ctx4k": 0.02}})
        self.assertEqual(rc, 1)
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline", "s1a", "score", "s1b"])
        res = self.rj("o1", "hold-result.json")
        self.assertEqual(res["verdict"], "FAIL")
        final = self.rj("o1", "q1a-summary.json")
        self.assertEqual((final["stage"], final["verdict"], final["determinism_status"]), ("final", "FAIL", "PASS"))
        self.assertEqual(self.rj("o1", "resume.json")["verdict"], "FAIL")      # resume still emitted immediately

    def test_determinism_fail_in_s1b(self):
        rc, _ = self.run_a("o1", knobs={"s1b_flip": "case3"})
        self.assertEqual(rc, 1)
        self.assertEqual(self.rj("o1", "q1a-summary.json")["determinism_status"], "FAIL")

    def test_rerun_into_the_same_out_is_refused(self):
        rc, _ = self.run_a("o1")
        self.assertEqual(rc, 0)
        before = sorted(os.listdir(self.out("o1")))
        rc2, text = self.run_a("o1")
        self.assertEqual(rc2, 2)
        self.assertIn("REFUSED", text)
        self.assertEqual(sorted(os.listdir(self.out("o1"))), before)


class FailureModes(HoldBase):
    def test_oom_is_a_fail_without_retry_or_bypass(self):
        rc, _ = self.run_a("o1", fail="baseline=oom")
        self.assertEqual(rc, 1)
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline"])
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual((rec["status"], rec["failure_class"]), ("EXIT_NONZERO", "OOM"))
        self.assertEqual(self.rj("o1", "hold-result.json")["verdict"], "FAIL")
        self.assertFalse(os.path.exists(os.path.join(self.out("o1"), "phases", "s1a")))

    def test_crash_is_incomplete_and_never_retried(self):
        rc, _ = self.run_a("o1", fail="s1a=crash")
        self.assertEqual(rc, 3)
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline", "s1a"])
        self.assertEqual(self.rj("o1", "receipts/s1a.json")["failure_class"], "OTHER")
        self.assertFalse(os.path.exists(os.path.join(self.out("o1"), "resume.json")))
        self.assertEqual(self.rj("o1", "hold-result.json")["verdict"], "INCOMPLETE")

    def test_timeout_reaps_the_own_group_and_records_the_cap(self):
        m = self.rig.write_manifest(self.rig.manifest_obj("a", caps={"baseline": 30, "s1a": 2, "s1b": 30}))
        rc, _ = self.run_a("o1", fail="s1a=hang", manifest=m)
        self.assertEqual(rc, 3)
        rec = self.rj("o1", "receipts/s1a.json")
        self.assertEqual(rec["status"], "TIMEOUT")
        self.assertTrue(rec["process"]["timed_out"])
        self.assertEqual(rec["caps"]["effective_cap_seconds"], 2)
        pid = self.rig.pids("o1")["s1a"]
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline", "s1a"])

    def test_watcher_fatal_aborts_only_the_own_group_and_stops_the_hold(self):
        rc, _ = self.run_a("o1", fail="baseline=watcher")
        self.assertEqual(rc, 3)
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual((rec["status"], rec["failure_class"]), ("WATCHER_FATAL", "BUS"))
        self.assertTrue(rec["watcher"]["matches"])
        self.assertIsNotNone(rec["health_abort"])
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline"])
        with self.assertRaises(ProcessLookupError):
            os.kill(self.rig.pids("o1")["baseline"], 0)

    def test_signal_forwards_to_the_own_group_and_closes_aborted(self):
        log = os.path.join(self.root, "logs", "o1.log")

        def waiter():
            for _ in range(400):
                if os.path.exists(log) and "baseline" in pathlib.Path(log).read_text():
                    time.sleep(0.5)
                    Q.STOP.set()
                    return
                time.sleep(0.05)

        t = threading.Thread(target=waiter)
        t.start()
        rc, _ = self.run_a("o1", fail="baseline=hang")
        t.join()
        self.assertEqual(rc, 4)
        res = self.rj("o1", "hold-result.json")
        self.assertEqual(res["verdict"], "ABORTED")
        self.assertEqual(self.rj("o1", "receipts/baseline.json")["status"], "ABORTED_SIGNAL")
        with self.assertRaises(ProcessLookupError):
            os.kill(self.rig.pids("o1")["baseline"], 0)
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline"])

    def test_missing_phase_summary_is_never_ignored(self):
        rc, _ = self.run_a("o1", fail="baseline=nosummary")
        self.assertEqual(rc, 3)
        self.assertEqual(self.rj("o1", "receipts/baseline.json")["status"], "INVALID_OUTPUT")
        self.assertEqual(self.rig.phases("o1"), ["check-plan", "baseline"])

    def test_ctime_hardlink_churn_is_metadata_only(self):
        before = os.stat(self.rig.model).st_ctime_ns
        rc, _ = self.run_a("o1", fail="baseline=churn")
        self.assertEqual(rc, 0, self.rj("o1", "hold-result.json")["reasons"])
        self.assertNotEqual(os.stat(self.rig.model).st_ctime_ns, before)       # the churn really moved ctime
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual(rec["integrity"]["changed_sections"], [])
        self.assertEqual(rec["status"], "OK")

    def test_model_mtime_change_is_an_input_change(self):
        rc, _ = self.run_a("o1", fail="baseline=touch_model")
        self.assertEqual(rc, 3)
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual(rec["status"], "INPUT_CHANGED")
        self.assertIn("cached", rec["integrity"]["changed_sections"])

    def test_owned_artifact_change_is_strict(self):
        rc, _ = self.run_a("o1", fail="baseline=tamper_source")
        self.assertEqual(rc, 3)
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual(rec["status"], "INPUT_CHANGED")
        self.assertIn("artifacts", rec["integrity"]["changed_sections"])

    def test_budget_gate_refuses_to_start_a_phase_that_cannot_get_its_minimum(self):
        m = self.rig.write_manifest(self.rig.manifest_obj("a", maximum=60, min_phase=20))
        host = SkewHost(self.rig.env("o1"), os.path.join(self.out("o1"), "receipts", "check-plan.json"), 100 * 10 ** 9)
        rc, _ = self.run_a("o1", manifest=m, host=host)
        self.assertEqual(rc, 3)
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual(rec["status"], "NOT_STARTED")
        self.assertTrue(any(w.startswith("BUDGET") for w in rec["why"]))
        self.assertEqual(self.rig.phases("o1"), ["check-plan"])

    def test_caps_are_watchdogs_and_the_total_cap_can_bind(self):
        m = self.rig.write_manifest(self.rig.manifest_obj("a", caps={"baseline": 100, "s1a": 100, "s1b": 100}, maximum=60))
        rc, _ = self.run_a("o1", manifest=m)
        self.assertEqual(rc, 0, self.rj("o1", "hold-result.json")["reasons"])
        rec = self.rj("o1", "receipts/baseline.json")
        self.assertEqual(rec["caps"]["bound_by"], "total")
        self.assertLessEqual(rec["caps"]["effective_cap_seconds"], 50)
        self.assertLess(rec["caps"]["effective_cap_seconds"], rec["caps"]["phase_cap_seconds"])
        self.assertTrue(self.rj("o1", "hold-result.json")["hold"]["sum_of_phase_caps"] > 60)


class Admission(HoldBase):
    def assert_refused(self, name, argv, host):
        rc, text = run_main(argv, host)
        self.assertEqual(rc, 2, text[-2000:])
        self.assertIn("REFUSED", text)
        self.assertFalse(os.path.exists(self.out(name)))
        self.assertEqual(self.rig.phases(name), [])

    def with_flag(self, argv, flag, value):
        a = list(argv)
        a[a.index(flag) + 1] = value
        return a

    def test_every_external_sha_is_enforced_before_any_mutation(self):
        base = self.rig.argv("run", self.m_a, self.out("o1"))
        for flag in ("--expected-manifest-sha256", "--expected-controller-sha256", "--expected-judge-sha256", "--expected-h2-sha256",
                     "--expected-binary-sha256", "--expected-plan-sha256", "--expected-native-source-sha256"):
            self.assert_refused("o1", self.with_flag(base, flag, "0" * 64), Q.Host(self.rig.env("o1")))

    def test_part_must_match_the_manifest(self):
        self.assert_refused("o1", self.rig.argv("run", self.m_a, self.out("o1"), part="f32"), Q.Host(self.rig.env("o1")))

    def test_real_fabric_job_and_matching_card_are_required_for_run(self):
        self.assert_refused("o1", self.rig.argv("run", self.m_a, self.out("o1")), Q.Host(self.rig.env("o1", fabric=False)))
        env = self.rig.env("o2")
        env["FABRIC_CARDS"] = "other-card"
        self.assert_refused("o2", self.rig.argv("run", self.m_a, self.out("o2")), Q.Host(env))

    def test_out_must_be_fresh_and_never_nested_with_inputs(self):
        os.mkdir(self.out("exists"))
        rc, text = run_main(self.rig.argv("run", self.m_a, self.out("exists")), Q.Host(self.rig.env("exists")))
        self.assertEqual(rc, 2)
        for bad in (os.path.join(self.rig.root, "models", "inside"), os.path.join(self.rig.pack, "inside"), os.path.join(self.rig.worktree, "out"),
                    os.path.join(HERE, "inside-out")):
            rc, text = run_main(self.rig.argv("run", self.m_a, bad), Q.Host(self.rig.env("n")))
            self.assertEqual(rc, 2, bad)
            self.assertFalse(os.path.exists(bad))

    def test_preflight_works_outside_fabric_and_records_null_job(self):
        host = Q.Host(self.rig.env("pre", fabric=False))
        rc, text = run_main(self.rig.argv("preflight", self.m_a), host)
        self.assertEqual(rc, 0, text[-3000:])
        doc = json.loads(text)
        self.assertTrue(doc["ok"])
        self.assertIsNone(doc["fabric"]["job"])
        self.assertTrue(doc["check_plan"]["ok"])
        self.assertEqual(self.rig.phases("pre"), ["check-plan"])
        self.assertTrue(J.seal_ok(doc))

    def test_preflight_reports_a_bad_artifact(self):
        with open(self.rig.source, "a") as f:
            f.write("changed\n")
        rc, text = run_main(self.rig.argv("preflight", self.m_a), Q.Host(self.rig.env("pre", fabric=False)))
        self.assertEqual(rc, 2)

    def test_writable_pack_is_refused(self):
        os.chmod(os.path.join(self.rig.pack, "k1.hsaco"), 0o644)
        rc, text = run_main(self.rig.argv("preflight", self.m_a), Q.Host(self.rig.env("pre", fabric=False)))
        self.assertEqual(rc, 2)
        self.assertIn("writable by mode", text)

    def test_undeclared_pack_file_is_refused(self):
        os.chmod(self.rig.pack, 0o755)
        with open(os.path.join(self.rig.pack, "extra.hsaco"), "wb") as f:
            f.write(b"x")
        os.chmod(os.path.join(self.rig.pack, "extra.hsaco"), 0o444)
        os.chmod(self.rig.pack, 0o555)
        rc, text = run_main(self.rig.argv("preflight", self.m_a), Q.Host(self.rig.env("pre", fabric=False)))
        self.assertEqual(rc, 2)
        self.assertIn("undeclared", text)

    def test_missing_card_or_wrong_device_is_refused(self):
        with open(os.path.join(self.rig.root, "sys", BDF, "device"), "w") as f:
            f.write("0xffff\n")
        rc, text = run_main(self.rig.argv("preflight", self.m_a), Q.Host(self.rig.env("pre", fabric=False)))
        self.assertEqual(rc, 2)

    def test_busy_card_is_refused_at_phase_start(self):
        with open(os.path.join(self.rig.root, "sys", BDF, "mem_info_vram_used"), "w") as f:
            f.write(str(8 << 30) + "\n")
        rc, _ = self.run_a("o1")
        self.assertEqual(rc, 3)
        self.assertEqual(self.rj("o1", "receipts/baseline.json")["status"], "NOT_STARTED")
        self.assertEqual(self.rig.phases("o1"), ["check-plan"])


class ManifestValidation(HoldBase):
    def errs(self, mutate, part="a", prior=None):
        m = self.rig.manifest_obj(part, prior=prior)
        mutate(m)
        return Q.validate_manifest(m)

    def test_baseline_manifest_is_valid(self):
        self.assertEqual(Q.validate_manifest(self.rig.manifest_obj("a")), [])
        self.assertEqual(Q.validate_manifest(self.rig.manifest_obj("f32", prior=("/x/summary.json", "a" * 64))), [])

    def test_class_env_cannot_be_changed_by_the_manifest(self):
        self.assertTrue(self.errs(lambda m: m["class_env"]["pinned"].update(HIPFIRE_KV_MODE="f16")))
        self.assertTrue(self.errs(lambda m: m["class_env"]["baseline_extra"].update(HIPFIRE_PREFILL_CHUNK_ROWS="768")))
        self.assertTrue(self.errs(lambda m: m["class_env"].update(s1_must_be_unset=[])))

    def test_budget_rules(self):
        self.assertTrue(self.errs(lambda m: m["budget"].update(maximum_hold_seconds=1801)))
        self.assertTrue(self.errs(lambda m: m["budget"]["phase_cap_seconds"].pop("s1b")))
        self.assertTrue(self.errs(lambda m: m["budget"]["phase_cap_seconds"].update(extra=5)))
        self.assertTrue(self.errs(lambda m: m["budget"].update(maximum_hold_seconds=901), part="f32", prior=("/x/s.json", "a" * 64)))
        self.assertTrue(self.errs(lambda m: m["budget"].update(min_phase_seconds=999)))
        self.assertTrue(self.errs(lambda m: m["budget"].pop("cpu_cap_seconds")))

    def test_part_prior_and_unknown_keys(self):
        self.assertTrue(self.errs(lambda m: m.update(prior_q1a={"summary_path": "/x", "summary_sha256": "a" * 64})))
        self.assertTrue(self.errs(lambda m: None, part="f32"))
        self.assertTrue(self.errs(lambda m: m.update(surprise=1)))
        self.assertTrue(self.errs(lambda m: m.update(schema="other")))
        self.assertTrue(Q.validate_manifest(dict(self.rig.manifest_obj("a"), part="z")))

    def test_identity_and_template_bindings(self):
        self.assertTrue(self.errs(lambda m: m["native_manifest_template"]["binary"].update(sha256="b" * 64)))
        self.assertTrue(self.errs(lambda m: m["native_manifest_template"]["source"].update(example_sha256="b" * 64)))
        self.assertTrue(self.errs(lambda m: m["native_manifest_template"]["source"].pop("runtime_commit")))
        self.assertTrue(self.errs(lambda m: m["native_manifest_template"]["model"].update(sha256="b" * 64)))
        self.assertTrue(self.errs(lambda m: m["identity"]["runtime"].update(commit="xyz")))
        self.assertTrue(self.errs(lambda m: m["harness"].update(q1_judge_sha256="zz")))

    def test_inputs_and_artifacts_shape(self):
        self.assertTrue(self.errs(lambda m: m["cached_inputs"].pop()))                      # the PLE sidecar must be bound
        self.assertTrue(self.errs(lambda m: m["cached_inputs"][0].update(mtime_ns=1.5)))     # floats never carry 19-digit ns
        self.assertTrue(self.errs(lambda m: m["artifacts"].append({"path": self.rig.model, "size_bytes": 5, "sha256": "a" * 64})))
        self.assertTrue(self.errs(lambda m: m["artifacts"].pop(0)))                          # the binary must be a declared artifact
        self.assertTrue(self.errs(lambda m: m["pack_dirs"].append("/nonexistent/other-pack")))
        self.assertTrue(self.errs(lambda m: m["watcher"].update(fatal_regex="(")))
        self.assertTrue(self.errs(lambda m: m.update(oom_stderr_regex="(")))
        self.assertTrue(self.errs(lambda m: m["target"].update(bdf="BAD")))

    def test_frozen_class_constants(self):
        self.assertEqual(Q.PINNED_ENV, T.PINNED)
        self.assertEqual(Q.BASELINE_EXTRA, T.BASELINE_ENV)
        self.assertEqual(Q.S1_MUST_UNSET, T.S1_ABSENT)
        self.assertEqual(Q.PART_MAX_HOLD, {"a": 1800, "f32": 900})
        self.assertEqual(Q.PART_PHASES["a"], ("baseline", "s1a", "s1b"))
        self.assertEqual(Q.PART_PHASES["f32"], ("beta-f32", "s1-f32"))

    def test_default_on_is_never_recommended(self):
        src = pathlib.Path(os.path.join(HERE, "q1_hold.py")).read_text()
        self.assertNotIn('"default_on_recommended": True', src)

    def test_env_builder_substitutes_f32_state_only_in_the_f32_phases(self):
        m = self.rig.manifest_obj("a")
        parent = {"PATH": "/bin", "HIPFIRE_POLLUTION": "1", "FABRIC_JOB": "j", "HIPFIRE_FABRIC_ADMISSION": "tok"}
        env, unset, applied = Q.gpu_env(parent, "/r", m, "beta-f32")
        self.assertEqual(applied["HIPFIRE_STATE_QUANT"], "f32")
        self.assertEqual(applied["HIPFIRE_KV_MODE"], "q8")
        self.assertEqual(applied["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"], "9")
        self.assertEqual(env["HIPFIRE_FABRIC_ADMISSION"], "tok")
        self.assertIn("HIPFIRE_POLLUTION", unset)
        env, unset, applied = Q.gpu_env(parent, "/r", m, "s1-f32")
        self.assertEqual(applied["HIPFIRE_STATE_QUANT"], "f32")
        self.assertNotIn("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", env)
        env, unset, applied = Q.gpu_env(parent, "/r", m, "s1a")
        self.assertEqual(applied["HIPFIRE_STATE_QUANT"], "q8")

    def test_classification(self):
        import re
        rx = re.compile("(?i)out of memory")
        self.assertEqual(Q.classify([], "", rx), None)
        self.assertEqual(Q.classify(["EXIT_NONZERO"], "Out Of Memory", rx), "OOM")
        self.assertEqual(Q.classify(["EXIT_NONZERO"], "boom", rx), "OTHER")
        self.assertEqual(Q.classify(["WATCHER_FATAL", "EXIT_NONZERO"], "out of memory", rx), "BUS")


class F32Part(HoldBase):
    def q1a_prior(self, knobs=None):
        rc, _ = self.run_a("q1a", knobs=knobs)
        self.assertIn(rc, (0, 1))
        path = os.path.join(self.out("q1a"), "q1a-summary.json")
        return path, sha_file(path)

    def f32_manifest(self, prior):
        return self.rig.write_manifest(self.rig.manifest_obj("f32", prior=prior))

    def run_f32(self, name, manifest, fail="", knobs=None):
        host = Q.Host(self.rig.env(name, True, fail, knobs))
        return run_main(self.rig.argv("run", manifest, self.out(name), part="f32"), host)

    def test_f32_part_reports_a_contrast_bound_to_the_sealed_q1a_summary(self):
        prior = self.q1a_prior()
        rc, text = self.run_f32("f1", self.f32_manifest(prior))
        self.assertEqual(rc, 0, text[-3000:])
        self.assertEqual(self.rig.phases("f1"), ["check-plan", "beta-f32", "s1-f32", "score-f32"])
        beta = self.rj("f1", "runs/beta-f32/env.json")["set"]
        s1 = self.rj("f1", "runs/s1-f32/env.json")["set"]
        self.assertEqual(beta["HIPFIRE_STATE_QUANT"], "f32")
        self.assertEqual(s1["HIPFIRE_STATE_QUANT"], "f32")
        self.assertEqual(beta["HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS"], "9")
        self.assertEqual(beta[T.SELECTOR_ENV], "0")
        self.assertNotIn("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", s1)
        self.assertEqual(s1[T.SELECTOR_ENV], "1")
        doc = self.rj("f1", "q1f32-summary.json")
        self.assertEqual(doc["verdict"], "REPORTED", doc["issues"])
        self.assertEqual(doc["bound_q1a_summary"]["sha256"], prior[1])
        self.assertFalse(doc["qualification_complete"])
        self.assertTrue(doc["diagnostic_only"])
        res = self.rj("f1", "hold-result.json")
        self.assertEqual(res["verdict"], "REPORTED")
        self.assertFalse(res["default_on_recommended"])
        cmd = pathlib.Path(os.path.join(self.out("f1"), "runs", "score-f32", "command.txt")).read_text()
        self.assertIn("--beta-f32-dir", cmd)
        self.assertNotIn("--baseline-dir", cmd)
        self.assertNotIn("--s1a-dir", cmd)
        pids = self.rig.pids("f1")
        self.assertNotEqual(pids["beta-f32"], pids["s1-f32"])

    def test_a_failed_but_sealed_final_q1a_is_accepted(self):
        prior = self.q1a_prior(knobs={"kl_const": {"primary-ctx16k": 0.0, "primary-ctx4k": 0.05}})
        self.assertEqual(jload(prior[0])["verdict"], "FAIL")
        rc, text = self.run_f32("f1", self.f32_manifest(prior))
        self.assertEqual(rc, 0, text[-3000:])
        self.assertEqual(self.rj("f1", "q1f32-summary.json")["bound_q1a_summary"]["verdict"], "FAIL")

    def test_f32_refuses_missing_or_invalid_prior_data(self):
        prior = self.q1a_prior()
        self.assertEqual(self.run_f32("r1", self.f32_manifest((prior[0], "0" * 64)))[0], 2)               # wrong sha
        resume = os.path.join(self.out("q1a"), "resume.json")                                               # stage 'resume' is not final
        self.assertEqual(self.run_f32("r2", self.f32_manifest((resume, sha_file(resume))))[0], 2)
        self.assertEqual(self.run_f32("r3", self.f32_manifest((os.path.join(self.root, "nope.json"), "a" * 64)))[0], 2)
        edited = os.path.join(self.root, "edited.json")
        doc = jload(prior[0])
        self.assertEqual(doc["verdict"], "PASS")         # the edit must REALLY change a sealed field
        doc["verdict"] = "FAIL"
        with open(edited, "w") as f:
            json.dump(doc, f)
        self.assertEqual(self.run_f32("r4", self.f32_manifest((edited, sha_file(edited))))[0], 2)         # seal broken
        for name in ("r1", "r2", "r3", "r4"):
            self.assertFalse(os.path.exists(self.out(name)))
        no_prior = self.rig.write_manifest(self.rig.manifest_obj("f32"))
        self.assertEqual(self.run_f32("r5", no_prior)[0], 2)

    def test_f32_phase_failure_stops_without_a_q8_rerun(self):
        prior = self.q1a_prior()
        rc, _ = self.run_f32("f1", self.f32_manifest(prior), fail="s1-f32=oom")
        self.assertEqual(rc, 1)
        self.assertEqual(self.rig.phases("f1"), ["check-plan", "beta-f32", "s1-f32"])
        self.assertNotIn("baseline", self.rig.phases("f1"))
        self.assertEqual(self.rj("f1", "hold-result.json")["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
