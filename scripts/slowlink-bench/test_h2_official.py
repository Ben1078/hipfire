#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of h2_official.py. No GPU, no model, no daemon: the "CLI" is a tiny fake script (TEST ONLY) written into a tmp tree; the card is
a fake sysfs directory of plain files. The real run path (Popen / process groups / sysfs reads / receipts) is exercised; only the clock is
injected where a test needs to move time.  Every `run` call below is ONE hold = ONE fake-CLI process with its OWN fake FABRIC_JOB (the rig hands out
a fresh job id per host); a logical pair is two such calls plus the pure-CPU assemble-pair.  The fake processes exist ONLY in these tests.

    python3 -m unittest test_h2_official            (from this directory; the parent runs it once after landing)
"""
import contextlib
import copy
import hashlib
import io
import itertools
import json
import os
import pathlib
import random
import shutil
import signal
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h2_official as h2  # noqa: E402

GIB = 1 << 30
MODEL_BYTES = (b"FAKE-MODEL-PAYLOAD-NEVER-HASHED" * 400)
PLE_BYTES = b"FAKE-PLE" * 300
MTIME_19 = 1_700_000_000_123_456_789       # 19-digit ns: any float round trip would corrupt it
BDF = "0000:03:00.0"
UUID = "GPU-1234abcd"

FAKE_CLI = r'''#!@PY@
import json, math, os, pathlib, signal, subprocess, sys, time
ARM = "@ARM@"
state = os.environ["H2_FAKE_STATE"]
os.makedirs(state, exist_ok=True)
idx_file = os.path.join(state, "calls")
n = len(pathlib.Path(idx_file).read_text().split()) if os.path.exists(idx_file) else 0
with open(idx_file, "a") as f:
    f.write(ARM + "\n")
plan = json.loads(os.environ["H2_FAKE_PLAN"])[n]
mode = plan.get("mode", "ok")

def stats(v):
    s = sorted(v)
    mean = sum(v) / len(v)
    var = sum((x - mean) ** 2 for x in v) / len(v)
    return {"median": s[len(s) // 2], "mean": mean, "min": s[0], "max": s[-1], "stdev": math.sqrt(var)}

hh = os.environ.get("HIPFIRE_HOME", "")
report = {
    "argv": sys.argv, "cwd": os.getcwd(), "pgid": os.getpgid(0), "pid": os.getpid(),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("HIP", "ROC", "HSA", "GPU", "AMD", "FABRIC", "HOME"))},
    "daemon_real": os.path.realpath(os.path.join(hh, "bin", "daemon")) if hh else None,
    "models_real": os.path.realpath(os.path.join(hh, "models")) if hh else None,
    "config": pathlib.Path(os.path.join(hh, "config.toml")).read_text() if hh else None,
    "hh_entries": sorted(os.listdir(hh)) if hh else None,
    "lockdir": os.path.isdir(os.environ.get("HIPFIRE_LOCK_DIR", "/nonexistent")),
    "kcache": os.path.isdir(os.environ.get("HIPFIRE_KERNEL_CACHE", "/nonexistent")),
    "call": n,
}
sys.stderr.write("FAKE-ENV " + json.dumps(report) + "\n")
if plan.get("stderr"):
    sys.stderr.write(plan["stderr"] + "\n")
sys.stderr.flush()
for path, text, how in plan.get("write", []):
    with open(path, how) as f:
        f.write(text)
for path, ns in plan.get("utime", []):
    os.utime(path, ns=(ns, ns))
for path in plan.get("create", []):
    pathlib.Path(path).write_text("x")
for name in plan.get("kcache_write", []):
    pathlib.Path(os.path.join(os.environ["HIPFIRE_KERNEL_CACHE"], name)).write_text("jit")
for path, mode in plan.get("chmod", []):
    os.chmod(path, mode)
for link, target in plan.get("relink", []):
    os.unlink(link)
    os.symlink(target, link)
for path in plan.get("try_write", []):
    try:
        with open(path, "w") as f:
            f.write("x")
        sys.stderr.write("TRYWRITE ok " + path + "\n")
    except OSError as e:
        sys.stderr.write("TRYWRITE err " + type(e).__name__ + " " + path + "\n")

def child(ignore_term):
    code = "import signal,time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") + "time.sleep(300)\n"
    # same process group on purpose: a daemon the CLI failed to reap. posix_spawn returns a bare pid (no Popen object to leave unreaped).
    pid = os.posix_spawn(sys.executable, [sys.executable, "-c", code], os.environ)
    pathlib.Path(os.path.join(state, "child%d.pid" % n)).write_text(str(pid))
    return pid

if mode == "exit3":
    sys.stderr.write("fake failure\n")
    sys.exit(3)
if mode == "badjson":
    print("this is not json")
    sys.exit(0)
if mode == "sleep":
    child(False)
    time.sleep(300)
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child(True)
    time.sleep(300)
if mode == "leak":
    child(False)
if mode == "raw":
    sys.stdout.write(plan["raw"])
    sys.exit(0)
s = plan["samples"]
d = plan.get("decode", [50.0, 50.0, 50.0])
doc = {
    "protocol": "synthetic-pp-tg-matrix-v1", "model": sys.argv[2], "loaded": {"max_seq": 16416, "kv_mode": "q8"}, "gpu": {"arch": "gfx1100"},
    "max_seq": {"effective": 16416, "requested": 16416, "source": "flag", "run_need": 16416, "model_limit": None},
    "redline_pm4": False, "kv_mode": "q8", "spec_requested": None,
    "load_speculation": {"mtp_mode": None, "dflash_mode": None, "dspark_mode": None, "ngram_draft": None, "mtp_effective": None},
    "prefill_route": "bench_prefill (synthetic ids, no speculative prompt fill)", "runs": 3, "warnings": [],
    "prefill": [{"tokens": 16384, "stats": stats(s), "samples": s, "power": []}],
    "decode": [{"context": 1, "tokens": 1, "stats": stats(d), "samples": d, "power": []}],
    "sustained": [], "topology": None,
}
doc.update(plan.get("patch", {}))
print(json.dumps(doc, indent=2))
'''


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return False
    return data[data.rindex(")") + 2:].split()[0] not in ("Z", "X")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read_bytes(path):
    return pathlib.Path(path).read_bytes()


def read_text(path):
    return pathlib.Path(path).read_text()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def write_text(path, text):
    pathlib.Path(path).write_text(text)


def write_bytes(path, data):
    pathlib.Path(path).write_bytes(data)


class ClockHost(h2.Host):
    """Injected clock: monotonic time only advances through sleep() (plus a tiny real sleep so subprocesses make progress)."""

    def __init__(self, environ, start_ns=10 ** 15):
        super().__init__(environ)
        self.now = start_ns

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1e9)
        time.sleep(0.01)


class StopHost(h2.Host):
    """stop_requested() turns true after N polls."""

    def __init__(self, environ, after):
        super().__init__(environ)
        self.after, self.calls = after, 0

    def stop_requested(self):
        self.calls += 1
        return self.calls > self.after


class Rig:
    def __init__(self, tmp, roles=None, budget=None, route_evidence=None, seeds=None):
        self.tmp = tmp
        self.jobs = itertools.count(1)                 # one fresh fake FABRIC_JOB per host == per held command
        self.roles = roles or {"A": "incumbent", "B": "candidate"}
        r = self.root = os.path.join(tmp, "rig")
        os.makedirs(r)
        self.models = os.path.join(r, "models")
        os.mkdir(self.models)
        self.model = os.path.join(self.models, "model.mq4")
        self.ple = self.model + ".ple-q4"
        for p, data in ((self.model, MODEL_BYTES), (self.ple, PLE_BYTES)):
            with open(p, "wb") as f:
                f.write(data)
            os.utime(p, ns=(MTIME_19, MTIME_19))
        self.sysdir = os.path.join(r, "sys", BDF)
        os.makedirs(self.sysdir)
        for name, val in (("mem_info_vram_used", "1000"), ("mem_info_vram_total", str(32 * GIB)), ("vendor", "0x1002"), ("device", "0x744c"),
                          ("unique_id", UUID.removeprefix("GPU-"))):
            self.write_sys(name, val)
        self.watch = os.path.join(r, "watch.log")
        with open(self.watch, "w") as f:
            f.write("old line before the hold\n")
        self.state = os.path.join(r, "fakestate")
        self.arms = {}
        for label in ("A", "B"):
            d = os.path.join(r, "arms", label)
            pack = os.path.join(d, "kernels", "compiled", "gfx1100")
            os.makedirs(pack)
            cli = os.path.join(d, "hipfire")
            with open(cli, "w") as f:
                f.write(FAKE_CLI.replace("@PY@", sys.executable).replace("@ARM@", label))
            os.chmod(cli, 0o755)
            daemon = os.path.join(d, "daemon")
            with open(daemon, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(daemon, 0o755)
            for name in ("pack.hsaco", "index.json"):
                with open(os.path.join(pack, name), "wb") as f:
                    f.write((label + name).encode() * 7)
            self.arms[label] = {"cli": cli, "daemon": daemon, "pack": pack}
        self.ledger = os.path.join(r, "ledger.json")
        with open(self.ledger, "w") as f:
            f.write('{"ledger": "fake"}')
        self.h1b = os.path.join(r, "h1b.json")
        with open(self.h1b, "w") as f:
            json.dump({"status": "PASS_ADVANCE", "advance": True}, f)
        self.receipts = os.path.join(r, "receipts")
        os.mkdir(self.receipts)
        self.seed_src = os.path.join(r, "seed.bin")
        with open(self.seed_src, "wb") as f:
            f.write(b"cache-metadata")
        self.manifest_path = os.path.join(r, "manifest.json")
        self.m = self.build_manifest(budget, route_evidence, seeds)
        self.save()
        self.lock_packs()                                # the installed packs are chmod a-w, exactly like the real beta pack

    def write_sys(self, name, value):
        with open(os.path.join(self.sysdir, name), "w") as f:
            f.write(str(value) + "\n")

    def cached_entry(self, path, tag):
        st = os.stat(path)
        e = {"path": path, "size_bytes": st.st_size, "device": st.st_dev, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns,
             "ctime_ns": st.st_ctime_ns, "sha256": sha(tag.encode())}
        rp = os.path.join(self.receipts, tag + ".json")
        rec = {k: e[k] for k in ("path", "size_bytes", "device", "inode", "mtime_ns", "ctime_ns", "sha256")}
        with open(rp, "w") as f:
            json.dump(rec, f)
        e["cached_receipt_path"] = rp
        return e

    def artifact(self, path):
        data = read_bytes(path)
        return {"path": path, "size_bytes": len(data), "sha256": sha(data)}

    def build_manifest(self, budget, route_evidence, seeds):
        arts = [self.artifact(self.ledger), self.artifact(self.h1b)]
        arms = {}
        for label in ("A", "B"):
            a = self.arms[label]
            arts += [self.artifact(a["cli"]), self.artifact(a["daemon"])]
            for name in sorted(os.listdir(a["pack"])):
                arts.append(self.artifact(os.path.join(a["pack"], name)))
            role = self.roles[label]
            cfg = '[memory]\nple_format = "q4"\n'
            if role == "candidate":
                cfg += "\n[developer]\nqwen4_slowlink_prefill = true\n"
            arms[label] = {"role": role, "cli": a["cli"], "daemon": a["daemon"], "pack_dir": a["pack"], "config_toml": cfg}
        m = {
            "schema": h2.SCHEMA_MANIFEST, "label": "H2 official pp16384 NON-DEFAULT (test)",
            "identity": {"candidate": {"commit": "4" * 40, "tree": "c" * 40}, "baseline": {"commit": "0" * 40, "tree": "e" * 40}},
            "harness": {"h2_official_sha256": h2.self_sha256()},
            "target": {"card": "xtx", "uuid": UUID, "bdf": BDF, "vendor_id": "0x1002", "device_id": "0x744c", "model_path": self.model,
                       "model_size": len(MODEL_BYTES), "models_dir": self.models,
                       "vram_used_path": os.path.join(self.sysdir, "mem_info_vram_used"),
                       "vram_total_path": os.path.join(self.sysdir, "mem_info_vram_total"),
                       "visible_devices_env": {"HIPFIRE_DEVICES": UUID, "ROCR_VISIBLE_DEVICES": UUID, "HIP_VISIBLE_DEVICES": "0"},
                       "absent_paths": [self.model + ".expert-hotset", self.model + ".trunk-overlay"]},
            "cached_inputs": [self.cached_entry(self.model, "model"), self.cached_entry(self.ple, "ple")],
            "artifacts": arts,
            "arms": arms,
            "budget": budget or {"cell_cap_seconds": 20, "kill_grace_seconds": 1, "drain_seconds": 1, "finalize_seconds": 1,
                                 "preflight_seconds": 5, "maximum_hold_seconds": 600},
            "memory": {"device_gib": 31.0, "host_gib": 67.0, "ledger_path": self.ledger, "ledger_sha256": self.artifact(self.ledger)["sha256"]},
            "h1b": {"receipt_path": self.h1b, "receipt_sha256": self.artifact(self.h1b)["sha256"], "accepted_status": "PASS_ADVANCE"},
            "watcher": {"log_path": self.watch, "fatal_regex": r"amdgpu.*(GPU reset|ring .* timeout)|VM_L2_PROTECTION_FAULT"},
        }
        if route_evidence is not None:
            m["route_evidence"] = route_evidence
        if seeds:
            m["home_seed_files"] = [{"source": self.seed_src, "dest": "cache/meta.bin", "size_bytes": 14, "sha256": sha(b"cache-metadata")}]
        return m

    @property
    def msha(self):
        """sha256 of the RAW manifest bytes on disk right now (test-side, independent of the harness)."""
        with open(self.manifest_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def save(self, m=None):
        with open(self.manifest_path, "w") as f:
            json.dump(m if m is not None else self.m, f)

    def _pack_paths(self):
        for label in ("A", "B"):
            pack = self.arms[label]["pack"]
            for root, dirs, names in os.walk(pack):
                for n in dirs + names:
                    yield os.path.join(root, n), os.path.isdir(os.path.join(root, n))
            yield pack, True

    def lock_packs(self):
        for p, is_dir in self._pack_paths():
            os.chmod(p, 0o555 if is_dir else 0o444)

    def unlock_packs(self):
        for p, is_dir in self._pack_paths():
            if os.path.lexists(p) and not os.path.islink(p):
                os.chmod(p, 0o755 if is_dir else 0o644)

    def host(self, plans, cls=h2.Host, **kw):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "FABRIC_JOB": f"job-{next(self.jobs)}", "FABRIC_CARDS": "xtx", "H2_FAKE_STATE": self.state,
               "H2_FAKE_PLAN": json.dumps(plans),
               # poison that MUST be stripped / never forwarded
               "HIPFIRE_QWEN4_SLOWLINK_PREFILL": "1", "HIPFIRE_DAEMON_BIN": "/poison/daemon", "HIP_VISIBLE_DEVICES": "9",
               "HSA_OVERRIDE_GFX_VERSION": "11.0.0", "HIPFIRE_EXPERT_VRAM_LAYERS": "10", "HIPFIRE_PREFILL_CHUNK": "512", "ROC_SOMETHING": "1",
               "GPU_MAX_HW_QUEUES": "8", "AMD_LOG_LEVEL": "4", "HIPFIRE_FABRIC_ADMISSION": "keep-me", "HIPFIRE_HOME": "/poison/home"}
        env.update(kw.pop("extra_env", {}))
        return cls(env, **kw)

    def children(self):
        out = []
        if os.path.isdir(self.state):
            for n in os.listdir(self.state):
                if n.startswith("child"):
                    out.append(int(read_text(os.path.join(self.state, n))))
        return out

    def cleanup(self):
        for pid in self.children():          # only pids this rig's fake CLI recorded about itself
            if alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


def ok_plan(samples, decode=None, **extra):
    p = {"mode": "ok", "samples": samples}
    if decode is not None:
        p["decode"] = decode
    p.update(extra)
    return p


def hold_plan(roles, cand, inc, **extra):
    """cand/inc: three sample lists (one per fresh process of that arm), emitted in the global order A1 B1 B2 A2 A3 B3."""
    seq = []
    for cell in h2.GLOBAL_ORDER:
        arm, k = cell[0], int(cell[1]) - 1
        seq.append(ok_plan((cand if roles[arm] == "candidate" else inc)[k], **extra))
    return seq


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="h2test-")
        self.rigs = []

    def tearDown(self):
        for r in self.rigs:
            r.cleanup()
            r.unlock_packs()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rig(self, **kw):
        r = Rig(os.path.join(self.tmp, f"r{len(self.rigs)}"), **kw)
        self.rigs.append(r)
        return r

    @staticmethod
    def sha_file(path):
        return sha(read_bytes(path))

    @staticmethod
    def cell_file(out, rec):
        return os.path.join(out, rec["cell_id"], "cell.json")

    def calls(self, rig):
        p = os.path.join(rig.state, "calls")
        return read_text(p).split() if os.path.exists(p) else []

    def run_cell(self, rig, plans, pair, cell, out, previous=(), first=None, host=None):
        """ONE `run` invocation = ONE hold = ONE fresh fake CLI. `first` = (cell 1 path, its independently recorded sha256) for cell 2."""
        host = host or rig.host(plans)
        f_path, f_sha = first if first else (None, None)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        return h2.run_cell(rig.manifest_path, rig.msha, pair, cell, out, list(previous), f_path, f_sha, host)

    def run_pair_cells(self, rig, plans, pair, previous=(), out_root=None, assemble=True):
        """Cell 1 hold, cell 2 hold (bound to cell 1), then pure-CPU assemble-pair. Cell 2 only runs when cell 1 is OK."""
        out_root = out_root or os.path.join(rig.root, "out")
        o1, o2 = os.path.join(out_root, f"p{pair}c1"), os.path.join(out_root, f"p{pair}c2")
        r1 = self.run_cell(rig, plans, pair, 1, o1, previous)
        c1 = self.cell_file(o1, r1)
        res = {"r1": r1, "c1": c1, "o1": o1, "o2": o2, "r2": None, "c2": None, "doc": None, "path": None}
        if r1["status"] != "OK":
            return res
        r2 = self.run_cell(rig, plans, pair, 2, o2, previous, first=(c1, self.sha_file(c1)))
        c2 = self.cell_file(o2, r2)
        res.update(r2=r2, c2=c2)
        if assemble:
            path = os.path.join(out_root, f"pair{pair}", "pair.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            res["doc"] = h2.assemble_pair(rig.manifest_path, rig.msha, [c1, c2], [self.sha_file(c1), self.sha_file(c2)], list(previous), path)
            res["path"] = path
        return res

    def run_pairs(self, rig, plans, pairs=(1, 2, 3), out_root=None):
        """Each logical pair = two single-process holds + assemble. Returns (pair docs, pair.json paths)."""
        docs, paths = [], []
        for p in pairs:
            res = self.run_pair_cells(rig, plans, p, paths, out_root)
            self.assertIsNotNone(res["doc"], res["r1"]["failures"])
            docs.append(res["doc"])
            paths.append(res["path"])
        return docs, paths

    def one_cell(self, plans, rig_kw=None, host=None, rig=None):
        """A fresh rig and ONE cell-1 hold. Returns (rig, cell receipt, OUT dir)."""
        r = rig or self.rig(**(rig_kw or {}))
        out = os.path.join(r.root, "out", "p1c1")
        return r, self.run_cell(r, plans, 1, 1, out, host=host), out

    def assert_second_cell_refused(self, r, r1, o1, needle="failed first cell"):
        """After a failed cell 1 nothing more runs: cell 2 is refused before any mutation or spawn."""
        c1 = self.cell_file(o1, r1)
        o2 = os.path.join(r.root, "out", "p1c2")
        before = self.calls(r)
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, r.msha, 1, 2, o2, [], c1, self.sha_file(c1), r.host([ok_plan([2000.0] * 3)] * 3))
        self.assertIn(needle, " ".join(cm.exception.errors))
        self.assertFalse(os.path.exists(o2))
        self.assertEqual(self.calls(r), before)

    def judge(self, rig, paths):
        return h2.judge(rig.manifest_path, rig.msha, paths, rig.host([]))

    @staticmethod
    def fake_env(out, cell_id):
        for line in read_text(os.path.join(out, cell_id, "stderr.log")).splitlines():
            if line.startswith("FAKE-ENV "):
                return json.loads(line[len("FAKE-ENV "):])
        raise AssertionError("no FAKE-ENV line")


# ======================================================================================================================================
class ManifestTests(Base):
    def test_valid_manifest_has_no_errors(self):
        r = self.rig()
        self.assertEqual(h2.validate_manifest(r.m), [])

    def mutate(self, fn):
        r = self.rig()
        m = copy.deepcopy(r.m)
        fn(m)
        return h2.validate_manifest(m)

    def test_unknown_and_missing_keys_rejected(self):
        self.assertTrue(any("unknown keys" in e for e in self.mutate(lambda m: m.__setitem__("extra", 1))))
        self.assertTrue(any("missing keys" in e for e in self.mutate(lambda m: m.pop("watcher"))))
        self.assertTrue(any("missing keys" in e for e in self.mutate(lambda m: m["budget"].pop("drain_seconds"))))
        self.assertTrue(any("unknown keys" in e for e in self.mutate(lambda m: m["target"].__setitem__("model_hash", "x"))))

    def test_stat_fields_must_be_json_integers(self):
        for bad in (float(MTIME_19), str(MTIME_19), True, -1):
            errs = self.mutate(lambda m, b=bad: m["cached_inputs"][0].__setitem__("mtime_ns", b))
            self.assertTrue(any("INTEGER" in e for e in errs), (bad, errs))

    def test_candidate_and_incumbent_config_contracts(self):
        cand = "[memory]\nple_format = \"q4\"\n"
        errs = self.mutate(lambda m: m["arms"]["B"].__setitem__("config_toml", cand))      # candidate without the developer key
        self.assertTrue(any("candidate contract" in e for e in errs), errs)
        dev = cand + "[developer]\nqwen4_slowlink_prefill = true\n"
        errs = self.mutate(lambda m: m["arms"]["A"].__setitem__("config_toml", dev))       # incumbent with the developer key
        self.assertTrue(any("incumbent contract" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["arms"]["B"].__setitem__("config_toml", dev + "[kernel]\nx = 1\n"))
        self.assertTrue(any("candidate contract" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["arms"]["A"].__setitem__("config_toml", "[memory]\nple_format = \"auto\"\n"))
        self.assertTrue(any("incumbent contract" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["arms"]["A"].__setitem__("config_toml", "schema_version = 2\n" + cand))
        self.assertTrue(any("schema_version" in e for e in errs), errs)
        self.assertEqual(self.mutate(lambda m: m["arms"]["A"].__setitem__("config_toml", "schema_version = 1\n" + cand)), [])
        errs = self.mutate(lambda m: m["arms"]["A"].__setitem__("config_toml", "[memory\n"))
        self.assertTrue(any("not TOML" in e for e in errs), errs)

    def test_roles_budget_and_paths(self):
        self.assertTrue(any("exactly one arm" in e for e in self.mutate(lambda m: m["arms"]["A"].__setitem__("role", "candidate"))))
        errs = self.mutate(lambda m: m["arms"]["A"].__setitem__("pack_dir", "/elsewhere/kernels/compiled/gfx1100"))
        self.assertTrue(any("pack_dir must be dirname(daemon)" in e for e in errs), errs)
        self.assertTrue(any("budget.cell_cap_seconds" in e for e in self.mutate(lambda m: m["budget"].__setitem__("cell_cap_seconds", 0))))
        self.assertTrue(any("budget.drain_seconds" in e for e in self.mutate(lambda m: m["budget"].__setitem__("drain_seconds", True))))
        self.assertTrue(any("cannot hold even one full single-process hold" in e for e in self.mutate(lambda m: m["budget"].__setitem__("maximum_hold_seconds", 5))))
        self.assertTrue(any("must be a declared artifact" in e for e in self.mutate(lambda m: m["artifacts"].pop(2))))
        self.assertTrue(any("model-side file" in e for e in self.mutate(
            lambda m: m["artifacts"].append({"path": m["target"]["model_path"], "size_bytes": 5, "sha256": "0" * 64}))))
        self.assertTrue(any("must bind" in e for e in self.mutate(lambda m: m["cached_inputs"].pop())))
        self.assertTrue(any("model size" in e for e in self.mutate(lambda m: m["cached_inputs"][0].__setitem__("size_bytes", 1))))
        self.assertTrue(any("absent_paths" in e for e in self.mutate(lambda m: m["target"].__setitem__("absent_paths", []))))
        self.assertTrue(any("visible_devices_env" in e for e in self.mutate(lambda m: m["target"]["visible_devices_env"].pop("HIP_VISIBLE_DEVICES"))))

    def test_budget_single_hold_minimum_preflight_and_backend_clamp(self):
        planned = {"cell_cap_seconds": 1200, "kill_grace_seconds": 15, "drain_seconds": 90, "finalize_seconds": 45, "preflight_seconds": 30,
                   "maximum_hold_seconds": 1440}
        self.assertEqual(h2.single_hold_min(planned), 1380)                                   # 1200+15+90+45+30
        self.assertEqual(self.mutate(lambda m: m.__setitem__("budget", dict(planned))), [])
        self.assertEqual(self.mutate(lambda m: m.__setitem__("budget", dict(planned, maximum_hold_seconds=1380))), [])   # exactly the single-hold minimum
        self.assertEqual(self.mutate(lambda m: m.__setitem__("budget", dict(planned, maximum_hold_seconds=1800))), [])   # exactly the backend clamp
        errs = self.mutate(lambda m: m.__setitem__("budget", dict(planned, maximum_hold_seconds=1379)))
        self.assertTrue(any("cannot hold even one full single-process hold" in e for e in errs), errs)
        errs = self.mutate(lambda m: m.__setitem__("budget", dict(planned, maximum_hold_seconds=1801)))
        self.assertTrue(any("clamp" in e for e in errs), errs)
        # two cold processes can never fit one hold: the cap is never squeezed, the over-clamp manifest is simply refused
        errs = self.mutate(lambda m: m.__setitem__("budget", dict(planned, cell_cap_seconds=2400, maximum_hold_seconds=2600)))
        self.assertTrue(any("clamp" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["budget"].pop("preflight_seconds"))
        self.assertTrue(any("missing keys" in e and "preflight_seconds" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["budget"].__setitem__("preflight_seconds", 0))
        self.assertTrue(any("budget.preflight_seconds" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["budget"].__setitem__("preflight_seconds", True))
        self.assertTrue(any("budget.preflight_seconds" in e for e in errs), errs)
        errs = self.mutate(lambda m: m["budget"].__setitem__("pair_seconds", 1))
        self.assertTrue(any("unknown keys" in e for e in errs), errs)

    def test_route_evidence_and_seed_validation(self):
        errs = self.mutate(lambda m: m.__setitem__("route_evidence", {"stderr_regex": "(", "required": True}))
        self.assertTrue(any("does not compile" in e for e in errs), errs)
        errs = self.mutate(lambda m: m.__setitem__("route_evidence", {"stderr_regex": "x", "required": "yes"}))
        self.assertTrue(any("boolean" in e for e in errs), errs)
        seed = {"source": "/x/y", "dest": "bin/daemon", "size_bytes": 1, "sha256": "0" * 64}
        errs = self.mutate(lambda m: m.__setitem__("home_seed_files", [seed]))
        self.assertTrue(any("collides" in e for e in errs), errs)
        seed2 = dict(seed, dest="../escape", size_bytes=h2.MAX_SEED_BYTES + 1)
        errs = self.mutate(lambda m: m.__setitem__("home_seed_files", [seed2]))
        self.assertTrue(any("inside HIPFIRE_HOME" in e for e in errs) and any("size_bytes" in e for e in errs), errs)

    def test_strict_loader_rejects_duplicates_and_nan(self):
        with self.assertRaises(ValueError):
            h2.loads_strict('{"a": 1, "a": 2}')
        with self.assertRaises(ValueError):
            h2.loads_strict('{"a": NaN}')
        with self.assertRaises(ValueError):
            h2.loads_strict('{"a": 1e999}')
        self.assertEqual(h2.loads_strict('{"a": %d}' % MTIME_19)["a"], MTIME_19)


# ======================================================================================================================================
class PreflightTests(Base):
    def test_ok_hashes_only_declared_artifacts_never_the_model(self):
        r = self.rig()
        opened, hashed = [], []
        real_open, real_hash = open, h2.sha256_file

        def spy_open(file, *a, **k):
            if isinstance(file, (str, os.PathLike)):
                opened.append(os.fspath(file))
            return real_open(file, *a, **k)

        def spy_hash(path):
            hashed.append(path)
            return real_hash(path)

        with mock.patch("builtins.open", spy_open), mock.patch.object(h2, "sha256_file", spy_hash):
            report, m = h2.preflight(r.manifest_path, r.msha, r.host([]))
        self.assertTrue(report["ok"], report["errors"])
        self.assertNotIn(r.model, opened)
        self.assertNotIn(r.ple, opened)
        self.assertNotIn(r.model, hashed)
        self.assertNotIn(r.ple, hashed)
        declared = {e["path"] for e in r.m["artifacts"]}
        self.assertTrue(declared <= set(hashed))
        self.assertTrue(set(hashed) - declared <= {os.path.realpath(h2.__file__)}, set(hashed) - declared)
        self.assertEqual(report["argv"]["B"][:3], [r.arms["B"]["cli"], "bench", r.model])
        self.assertEqual(report["roles"], {"A": "incumbent", "B": "candidate"})
        self.assertEqual(report["cached_inputs"][0]["mtime_ns"], MTIME_19)

    def errors(self, r):
        report, _ = h2.preflight(r.manifest_path, r.msha, r.host([]))
        return report["errors"]

    def test_stat_ns_binding_and_receipt(self):
        r = self.rig()
        self.assertEqual(self.errors(r), [])
        os.utime(r.model, ns=(MTIME_19, MTIME_19 + 1))                       # one nanosecond is enough
        errs = self.errors(r)
        self.assertTrue(any("mtime_ns" in e for e in errs), errs)
        os.utime(r.model, ns=(MTIME_19, MTIME_19))
        self.assertTrue(any("ctime_ns" in e for e in self.errors(r)))        # utime moved the inode ctime: exact binding catches it
        r2 = self.rig()
        m = copy.deepcopy(r2.m)
        m["cached_inputs"][1]["inode"] += 1
        r2.save(m)
        self.assertTrue(any("inode" in e for e in self.errors(r2)))
        r3 = self.rig()
        with open(r3.m["cached_inputs"][0]["cached_receipt_path"]) as f:
            rec = json.load(f)
        rec["sha256"] = "f" * 64
        with open(r3.m["cached_inputs"][0]["cached_receipt_path"], "w") as f:
            json.dump(rec, f)
        self.assertTrue(any("receipt sha256" in e for e in self.errors(r3)))
        rec["extra"] = 1
        with open(r3.m["cached_inputs"][0]["cached_receipt_path"], "w") as f:
            json.dump(rec, f)
        self.assertTrue(any("receipt keys" in e for e in self.errors(r3)))

    def test_float_ns_in_manifest_refused_even_if_equal(self):
        r = self.rig()
        text = read_text(r.manifest_path).replace(str(MTIME_19), f"{MTIME_19}.0")
        self.assertIn(f"{MTIME_19}.0", text)
        with open(r.manifest_path, "w") as f:
            f.write(text)
        errs = self.errors(r)
        self.assertTrue(any("INTEGER" in e for e in errs), errs)

    def test_sidecar_artifact_harness_and_device_checks(self):
        r = self.rig()
        write_text(r.model + ".expert-hotset", "x")
        self.assertTrue(any("expected-absent" in e for e in self.errors(r)))
        os.unlink(r.model + ".expert-hotset")
        with open(r.arms["A"]["cli"], "a") as f:
            f.write("# tamper\n")
        errs = self.errors(r)
        self.assertTrue(any("size" in e and "artifact" in e for e in errs), errs)
        r2 = self.rig()
        r2.unlock_packs()
        with open(os.path.join(r2.arms["B"]["pack"], "undeclared.bin"), "w") as f:
            f.write("x")
        r2.lock_packs()
        self.assertTrue(any("undeclared file in pack_dir" in e for e in self.errors(r2)))
        r3 = self.rig()
        r3.unlock_packs()
        os.unlink(os.path.join(r3.arms["A"]["pack"], "pack.hsaco"))
        r3.lock_packs()
        errs = self.errors(r3)
        self.assertTrue(any("declared pack file missing" in e for e in errs), errs)
        r4 = self.rig()
        m = copy.deepcopy(r4.m)
        m["harness"]["h2_official_sha256"] = "1" * 64
        r4.save(m)
        self.assertTrue(any("harness sha256" in e for e in self.errors(r4)))
        r5 = self.rig()
        r5.write_sys("vendor", "0xffff")
        self.assertTrue(any("device check failed" in e for e in self.errors(r5)))
        r6 = self.rig()
        with open(r6.h1b, "w") as f:
            json.dump({"status": "PASS_ADVANCE", "advance": False}, f)
        errs = self.errors(r6)                                                 # h1b file changed: hash AND content refuse
        self.assertTrue(any("h1b" in e for e in errs), errs)

    def test_installed_packs_must_be_readonly_and_preflight_never_chmods(self):
        r = self.rig()
        rep, _ = h2.preflight(r.manifest_path, r.msha, r.host([]))
        self.assertTrue(rep["ok"], rep["errors"])
        for label in ("A", "B"):
            self.assertEqual((rep["pack_readonly"][label]["verified"], rep["pack_readonly"][label]["files"], rep["pack_readonly"][label]["dirs"]),
                             (True, 2, 1))
        pack_a, pack_b = r.arms["A"]["pack"], r.arms["B"]["pack"]
        hsaco = os.path.join(pack_a, "pack.hsaco")
        os.chmod(hsaco, 0o644)                                                  # a writable pack FILE
        errs = self.errors(r)
        self.assertTrue(any("writable by mode" in e and "pack.hsaco" in e for e in errs), errs)
        os.chmod(hsaco, 0o444)
        os.chmod(pack_b, 0o755)                                                 # a writable pack DIRECTORY
        errs = self.errors(r)
        self.assertTrue(any("writable by mode" in e and pack_b in e for e in errs), errs)
        self.assertEqual(stat.S_IMODE(os.stat(pack_b).st_mode), 0o755)           # preflight reported it, it did not repair it
        os.chmod(pack_b, 0o555)
        self.assertEqual(self.errors(r), [])
        self.assertEqual(stat.S_IMODE(os.stat(pack_b).st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(os.stat(hsaco).st_mode), 0o444)
        self.assertTrue(h2.preflight(r.manifest_path, r.msha, r.host([]))[0]["ok"])

    @unittest.skipIf(os.geteuid() == 0, "root bypasses permission bits: the effective-access probe is not meaningful")
    def test_readonly_pack_cannot_be_written_by_the_effective_user(self):
        r = self.rig()
        pack = r.arms["A"]["pack"]
        with self.assertRaises(PermissionError):
            write_text(os.path.join(pack, "jit-output.bin"), "x")
        with self.assertRaises(PermissionError):
            write_text(os.path.join(pack, "pack.hsaco"), "x")
        self.assertFalse(os.access(pack, os.W_OK))

    def test_symlinked_artifact_refused(self):
        r = self.rig()
        real = r.arms["A"]["cli"] + ".real"
        os.rename(r.arms["A"]["cli"], real)
        os.symlink(real, r.arms["A"]["cli"])
        errs = self.errors(r)
        self.assertTrue(any("not a regular file" in e for e in errs), errs)

    def test_cli_preflight_exit_codes_and_report(self):
        r = self.rig()
        rep = os.path.join(r.root, "pre.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = h2.main(["preflight", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--report", rep], r.host([]))
        self.assertEqual(rc, 0)
        self.assertTrue(read_json(rep)["ok"])
        r.write_sys("device", "0xffff")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h2.main(["preflight", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha], r.host([])), 2)

    # ---------------------------------------------------------------- independently supplied expected manifest sha256 (REQUIRED)
    def test_expected_manifest_sha_required_and_enforced_on_every_command(self):
        r = self.rig()
        good = r.msha
        self.assertEqual(good, sha(read_bytes(r.manifest_path)))
        # positive
        m, got = h2.load_manifest(r.manifest_path, good)
        self.assertEqual(got, good)
        report, _ = h2.preflight(r.manifest_path, good, r.host([]))
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["expected_manifest_sha256"], good)
        # wrong / malformed / missing values are refused BEFORE the manifest is parsed or trusted
        wrong = "0" * 64
        for bad in (wrong, good.upper(), good[:-1], good + "0", "", None, 5):
            with self.assertRaises(h2.H2Error, msg=repr(bad)):
                h2.load_manifest(r.manifest_path, bad)
            report, mm = h2.preflight(r.manifest_path, bad, r.host([]))
            self.assertFalse(report["ok"])
            self.assertIsNone(mm)
            self.assertNotIn("manifest_sha256", report)                  # nothing from the unverified manifest is reported
            self.assertEqual(len(report["errors"]), 1)
        report, _ = h2.preflight(r.manifest_path, wrong, r.host([]))
        self.assertIn("independently supplied expected", report["errors"][0])
        # a byte-level edit that keeps the JSON valid and equal in meaning is still refused
        with open(r.manifest_path, "ab") as f:
            f.write(b"\n")
        report, _ = h2.preflight(r.manifest_path, good, r.host([]))
        self.assertFalse(report["ok"])
        self.assertIn("!= the independently supplied expected", report["errors"][0])

    def test_cli_requires_expected_manifest_sha_before_anything(self):
        r = self.rig()
        out = os.path.join(r.root, "out", "p1c1")
        os.makedirs(os.path.dirname(out))
        res_out = os.path.join(r.root, "res.json")
        pair_out = os.path.join(r.root, "pair.json")
        cells = [os.path.join(r.root, "c1.json"), os.path.join(r.root, "c2.json")]
        cmds = (["preflight", "--manifest", r.manifest_path],
                ["run", "--manifest", r.manifest_path, "--pair", "1", "--cell", "1", "--out", out],
                ["judge", "--manifest", r.manifest_path, "--pairs", os.path.join(r.root, "p.json"), "--out", res_out],
                ["assemble-pair", "--manifest", r.manifest_path, "--cells", *cells, "--cell-sha256", "0" * 64, "1" * 64, "--out", pair_out])
        for argv in cmds:                                           # missing: argparse refuses (exit 2), nothing is created
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as cm:
                h2.main(argv, r.host([ok_plan([2000.0] * 3)] * 2))
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--expected-manifest-sha256", err.getvalue())
        for argv in cmds:                                           # malformed (not 64 lowercase hex)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                h2.main(argv + ["--expected-manifest-sha256", "abc"], r.host([]))
            self.assertEqual(cm.exception.code, 2)
        self.assertFalse(os.path.exists(out))
        self.assertFalse(os.path.exists(res_out))
        self.assertFalse(os.path.exists(pair_out))
        self.assertFalse(os.path.exists(os.path.join(r.state, "calls")))
        # wrong but well-formed
        wrong = ["--expected-manifest-sha256", "f" * 64]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h2.main(cmds[0] + wrong, r.host([])), 2)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = h2.main(cmds[1] + wrong, r.host([ok_plan([2000.0] * 3)] * 2))
        self.assertEqual(rc, 2)
        refusal = json.loads(buf.getvalue())
        self.assertEqual((refusal["status"], refusal["mutated"]), ("REFUSED", False))
        self.assertTrue(any("independently supplied expected" in e for e in refusal["errors"]), refusal)
        self.assertFalse(os.path.exists(out), "a wrong expected sha must refuse before the out dir is created")
        self.assertFalse(os.path.exists(os.path.join(r.state, "calls")), "and before any spawn")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(h2.main(cmds[3] + wrong, r.host([])), 2)
        self.assertTrue(any("independently supplied expected" in e for e in json.loads(buf.getvalue())["errors"]))
        self.assertFalse(os.path.exists(pair_out))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h2.main(cmds[2] + wrong, r.host([])), 3)
        res = read_json(res_out)
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertEqual(res["expected_manifest_sha256"], "f" * 64)
        self.assertNotIn("candidate", res)                          # no receipt was read, no verdict content
        self.assertNotIn("manifest_sha256", res)
        # the correct value runs exactly ONE process
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h2.main(cmds[1] + ["--expected-manifest-sha256", r.msha], r.host([ok_plan([2000.0] * 3)] * 2)), 0)
        self.assertTrue(os.path.exists(os.path.join(out, "A1", "cell.json")))
        self.assertEqual(self.calls(r), ["A"])

    def test_run_assemble_and_judge_refuse_wrong_external_sha_even_with_valid_receipts(self):
        r = self.rig()
        plans = hold_plan(r.roles, [[2400.0] * 3] * 3, [[1000.0] * 3] * 3)
        docs, paths = self.run_pairs(r, plans, pairs=(1, 2))
        out3 = os.path.join(r.root, "out", "p3c1")
        calls = self.calls(r)
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, "1" * 64, 3, 1, out3, paths, None, None, r.host(plans))
        self.assertIn("independently supplied expected", " ".join(cm.exception.errors))
        self.assertFalse(os.path.exists(out3))
        self.assertEqual(self.calls(r), calls)
        res3 = self.run_pair_cells(r, plans, 3, paths, assemble=False)
        pair3 = os.path.join(r.root, "out", "pair3", "pair.json")
        os.makedirs(os.path.dirname(pair3))
        cell_args = [res3["c1"], res3["c2"]], [self.sha_file(res3["c1"]), self.sha_file(res3["c2"])]
        with self.assertRaises(h2.Refusal) as cm:
            h2.assemble_pair(r.manifest_path, "1" * 64, *cell_args, paths, pair3)
        self.assertIn("independently supplied expected", " ".join(cm.exception.errors))
        self.assertFalse(os.path.exists(pair3))
        res = h2.judge(r.manifest_path, "1" * 64, paths, r.host([]))
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertNotIn("candidate", res)
        self.assertEqual(h2.judge(r.manifest_path, r.msha, paths, r.host([]))["verdict"], "UNKNOWN_INCOMPLETE")

# ======================================================================================================================================
class RunTests(Base):
    def test_cell_argv_config_env_fresh_homes_and_no_model_copy(self):
        r = self.rig()
        plans = hold_plan(r.roles, [[2200.0, 2300.0, 2400.0]] * 3, [[1900.0, 2000.0, 2100.0]] * 3)
        res = self.run_pair_cells(r, plans, 1)
        doc, cells, outs = res["doc"], [res["r1"], res["r2"]], [res["o1"], res["o2"]]
        self.assertEqual(doc["status"], "PAIR_COMPLETE", doc["stop_reason"])
        self.assertEqual(doc["order"], ["A1", "B1"])
        self.assertTrue(h2.seal_ok(doc))
        self.assertEqual(read_json(res["path"]), doc)
        self.assertEqual([c["cell_id"] for c in cells], ["A1", "B1"])
        homes, jobs = set(), []
        for c, arm, out in zip(cells, ("A", "B"), outs):
            self.assertEqual(c["status"], "OK")
            self.assertTrue(c["measurement_valid"] and h2.seal_ok(c))
            self.assertEqual(read_json(self.cell_file(out, c)), c)
            self.assertEqual(sha(read_bytes(os.path.join(out, "preflight.json"))), c["preflight_sha256"])
            self.assertEqual(sorted(os.listdir(out)), [c["cell_id"], "preflight.json"])      # ONE cell dir per hold, never the other cell's
            self.assertEqual(c["argv"], [r.arms[arm]["cli"], "bench", r.model, "--matrix", "--pp", "16384", "--tg", "1", "--ctx", "1", "--runs", "3",
                                         "--warmups", "1", "--max-seq", "16416", "--json"])
            self.assertEqual(c["home"]["sha256"], sha(r.m["arms"][arm]["config_toml"].encode()))
            self.assertEqual(c["cli"]["sha256"], r.artifact(r.arms[arm]["cli"])["sha256"])
            self.assertEqual(c["daemon"]["sha256"], r.artifact(r.arms[arm]["daemon"])["sha256"])
            self.assertEqual(c["manifest_sha256"], doc["manifest_sha256"])
            cell_dir = os.path.join(out, c["cell_id"])
            rep = self.fake_env(out, c["cell_id"])
            self.assertEqual(rep["argv"][1:], c["argv"][1:])
            self.assertEqual(rep["cwd"], cell_dir)
            self.assertEqual(rep["pgid"], rep["pid"])                       # own session/group
            self.assertEqual(rep["pgid"], c["process"]["pgid"])
            self.assertEqual(c["process"]["pid"], rep["pid"])
            self.assertEqual(rep["daemon_real"], os.path.realpath(r.arms[arm]["daemon"]))   # the supported symlink, to THIS arm's real daemon
            self.assertEqual(rep["models_real"], os.path.realpath(r.models))
            self.assertTrue(rep["lockdir"] and rep["kcache"])
            self.assertEqual(rep["config"], r.m["arms"][arm]["config_toml"])
            self.assertEqual("developer" in rep["config"], r.roles[arm] == "candidate")
            self.assertEqual(rep["hh_entries"], ["bin", "config.toml", "models"])
            env = rep["env"]
            self.assertEqual(env["HOME"], os.path.join(cell_dir, "home"))
            self.assertEqual(env["HIPFIRE_HOME"], os.path.join(cell_dir, "home", ".hipfire"))
            self.assertEqual(env["HIPFIRE_KERNEL_CACHE"], os.path.join(cell_dir, "kcache"))   # the PRIVATE per-cell writable hot cache
            self.assertEqual((c["hot_cache"]["path"], c["hot_cache"]["private"], c["hot_cache"]["writable"]), (env["HIPFIRE_KERNEL_CACHE"], True, True))
            self.assertTrue(c["pack_readonly"]["verified"])
            self.assertEqual(env["HIPFIRE_QWEN4_D2_DEFAULT"], "0")
            self.assertEqual(env["HIPFIRE_NO_REGISTRY_FETCH"], "1")
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")                # manifest value replaced the poisoned inherited 9
            self.assertEqual(env["HIPFIRE_DEVICES"], UUID)
            self.assertEqual(env["FABRIC_JOB"], c["env"]["fabric_job"])      # the receipt records the ACTUAL job of this one process
            self.assertTrue(env["FABRIC_JOB"].startswith("job-"))
            jobs.append(env["FABRIC_JOB"])
            self.assertEqual(env["HIPFIRE_FABRIC_ADMISSION"], "keep-me")
            for forbidden in ("HIPFIRE_DAEMON_BIN", "HIPFIRE_QWEN4_SLOWLINK_PREFILL", "HSA_OVERRIDE_GFX_VERSION", "HIPFIRE_EXPERT_VRAM_LAYERS",
                              "HIPFIRE_PREFILL_CHUNK", "ROC_SOMETHING", "GPU_MAX_HW_QUEUES", "AMD_LOG_LEVEL"):
                self.assertNotIn(forbidden, env)
            self.assertEqual(sorted(k for k in env if k.startswith(("HIP", "ROC", "HSA", "GPU", "AMD"))),
                             sorted(["HIPFIRE_HOME", "HIPFIRE_LOCK_DIR", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                     "HIP_VISIBLE_DEVICES", "HIPFIRE_NO_REGISTRY_FETCH", "HIPFIRE_QWEN4_D2_DEFAULT", "HIPFIRE_FABRIC_ADMISSION"]))
            homes.add(env["HOME"])
            # nothing model-sized was copied into the cell
            for root, _, names in os.walk(os.path.join(cell_dir, "home")):
                for n in names:
                    p = os.path.join(root, n)
                    if not os.path.islink(p):
                        self.assertLess(os.path.getsize(p), 4096, p)
            self.assertTrue(os.path.islink(os.path.join(cell_dir, "home", ".hipfire", "models")))
            proc = read_json(os.path.join(cell_dir, "process.json"))
            self.assertEqual(proc["pid"], c["process"]["pid"])
            self.assertEqual(proc["starttime_ticks"], c["process"]["starttime_ticks"])
        self.assertEqual(len(homes), 2)
        self.assertEqual(len(set(jobs)), 2)
        self.assertNotEqual(cells[0]["process"]["pid"], cells[1]["process"]["pid"])
        self.assertEqual(h2.expected_argv(r.m, "A"), cells[0]["argv"])
        for out in outs:                               # nothing outside the cells was written with model bytes
            for p in os.listdir(out):
                full = os.path.join(out, p)
                if os.path.isfile(full):
                    self.assertNotEqual(os.path.getsize(full), len(MODEL_BYTES))
        self.assertEqual(read_bytes(r.model), MODEL_BYTES)

    def test_three_pairs_global_order_and_judge_pass(self):
        r = self.rig()
        cand = [[2200.0, 2300.0, 2400.0], [2250.0, 2350.0, 2450.0], [2100.0, 2150.0, 2500.0]]
        inc = [[1900.0, 1950.0, 2000.0], [1850.0, 1900.0, 1950.0], [1800.0, 1850.0, 2000.0]]
        docs, paths = self.run_pairs(r, hold_plan(r.roles, cand, inc))
        self.assertTrue(all(d["status"] == "PAIR_COMPLETE" for d in docs))
        self.assertEqual(self.calls(r), ["A", "B", "B", "A", "A", "B"])             # A1 B1 | B2 A2 | A3 B3: six single-process holds
        self.assertEqual([c["cell_id"] for d in docs for c in d["cells"]], list(h2.GLOBAL_ORDER))
        res = self.judge(r, paths)
        self.assertEqual(res["verdict"], "PASS", res)
        self.assertEqual(res["global_order_executed"], list(h2.GLOBAL_ORDER))
        flat = sorted(s for k in cand for s in k)
        self.assertEqual(res["candidate"]["median_all_samples"], flat[4])
        self.assertEqual(res["candidate"]["sample_count"], 9)
        self.assertEqual(res["incumbent"]["sample_count"], 9)
        self.assertTrue(res["all9"]["gate_2000_tok_s"] and res["all9"]["gate_1p10_over_incumbent"] and res["all9"]["informational_1400_tok_s"])
        self.assertEqual(res["all9"]["actual_holds"], 6)
        self.assertEqual(len(set(res["fabric_jobs"])), 6)
        self.assertTrue(h2.seal_ok(res))
        # the CLI writes the same result atomically and exits 0; it never overwrites
        out = os.path.join(r.root, "result.json")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = h2.main(["judge", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--pairs", *paths, "--out", out], r.host([]))
            self.assertEqual(rc, 0)
            with self.assertRaises(FileExistsError):
                h2.main(["judge", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--pairs", *paths, "--out", out], r.host([]))
        self.assertEqual(read_json(out)["verdict"], "PASS")

    def test_swapped_roles_map_candidate_correctly(self):
        r = self.rig(roles={"A": "candidate", "B": "incumbent"})
        cand = [[2400.0] * 3] * 3
        inc = [[1000.0] * 3] * 3
        _, paths = self.run_pairs(r, hold_plan(r.roles, cand, inc))
        res = self.judge(r, paths)
        self.assertEqual(res["verdict"], "PASS", res)
        self.assertEqual((res["candidate_arm"], res["incumbent_arm"]), ("A", "B"))
        self.assertEqual(res["candidate"]["median_all_samples"], 2400.0)

    def test_route_evidence_n_tile_observed_or_null(self):
        rx = {"stderr_regex": r"slowlink route layer-major N=(?P<n>\d+) tile=(?P<tile>\d+)", "required": True,
              "incumbent_forbidden_regex": r"slowlink route"}
        r = self.rig(route_evidence=rx)
        plans = [ok_plan([2000.0] * 3), ok_plan([2000.0] * 3, stderr="slowlink route layer-major N=7 tile=3")]
        res = self.run_pair_cells(r, plans, 1)
        c_inc, c_cand = res["r1"], res["r2"]
        self.assertEqual(c_cand["route_evidence"]["n"], 7)
        self.assertEqual(c_cand["route_evidence"]["tile"], 3)
        self.assertEqual(c_inc["route_evidence"]["forbidden_matched"], False)
        # required pattern absent -> measurement invalid, never silently accepted
        r2 = self.rig(route_evidence=rx)
        res = self.run_pair_cells(r2, [ok_plan([2000.0] * 3)] * 2, 1)
        self.assertEqual(res["doc"]["status"], "PAIR_INCOMPLETE")
        self.assertEqual(res["r1"]["status"], "OK")
        self.assertEqual(res["r2"]["status"], "ROUTE_EVIDENCE_MISSING")
        self.assertFalse(res["r2"]["measurement_valid"])
        # incumbent contamination
        r3 = self.rig(route_evidence=rx)
        res = self.run_pair_cells(r3, [ok_plan([2000.0] * 3, stderr="slowlink route leaked")] * 2, 1)
        self.assertEqual(res["r1"]["status"], "ROUTE_CONTAMINATION")
        self.assertIsNone(res["r2"])
        # no declared evidence: nothing observed, nothing invented
        r4 = self.rig()
        res = self.run_pair_cells(r4, [ok_plan([2000.0] * 3, stderr="N=9 tile=9")] * 2, 1)
        ev = res["r2"]["route_evidence"]
        self.assertEqual((ev["declared"], ev["matched"], ev["n"], ev["tile"]), (False, None, None, None))
        # a regex without capture groups: matched, N/tile stay null
        r5 = self.rig(route_evidence={"stderr_regex": "route ok", "required": True})
        res = self.run_pair_cells(r5, [ok_plan([2000.0] * 3), ok_plan([2000.0] * 3, stderr="route ok")], 1)
        ev = res["r2"]["route_evidence"]
        self.assertEqual((ev["matched"], ev["n"], ev["tile"]), (True, None, None))

    def test_home_seed_files_are_exact_copies(self):
        r = self.rig(seeds=True)
        _, c, out = self.one_cell([ok_plan([2000.0] * 3)] * 2, rig=r)
        seeded = os.path.join(out, "A1", "home", ".hipfire", "cache", "meta.bin")
        self.assertEqual(read_bytes(seeded), b"cache-metadata")
        self.assertEqual(c["home"]["seeded"][0]["sha256"], sha(b"cache-metadata"))

    # ---------------------------------------------------------------- refusals before mutation
    def refusal(self, r, host, pair=1, cell=1, previous=(), out=None, first=None):
        out = out or os.path.join(r.root, "out", f"p{pair}c{cell}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        f_path, f_sha = first if first else (None, None)
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, r.msha, pair, cell, out, list(previous), f_path, f_sha, host)
        self.assertFalse(os.path.exists(out), "a refusal must not create the out dir")
        self.assertFalse(os.path.exists(os.path.join(r.state, "calls")), "a refusal must not spawn the CLI")
        return " | ".join(cm.exception.errors)

    def test_refusals_before_spawn_or_mutation(self):
        r = self.rig()
        plans = [ok_plan([2000.0] * 3)] * 6
        r.write_sys("mem_info_vram_used", 3 * GIB)
        self.assertIn("BUSY", self.refusal(r, r.host(plans)))
        r.write_sys("mem_info_vram_used", 1000)
        self.assertIn("not inside a fabric job", self.refusal(r, r.host(plans, extra_env={"FABRIC_JOB": ""})))
        self.assertIn("not inside a fabric job", self.refusal(r, r.host(plans, extra_env={"FABRIC_CARDS": ""})))
        self.assertIn("more than one card", self.refusal(r, r.host(plans, extra_env={"FABRIC_CARDS": "xtx,other"})))
        self.assertIn("does not name the target", self.refusal(r, r.host(plans, extra_env={"FABRIC_CARDS": "other"})))
        out = os.path.join(r.root, "out", "exists")
        os.makedirs(out)
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, r.msha, 1, 1, out, [], None, None, r.host(plans))
        self.assertIn("FRESH", " ".join(cm.exception.errors))
        self.assertEqual(os.listdir(out), [])
        self.assertIn("needs exactly 1 --previous", self.refusal(r, r.host(plans), pair=2))
        self.assertIn("--pair must be", self.refusal(r, r.host(plans), pair=4))
        self.assertIn("--cell must be", self.refusal(r, r.host(plans), cell=3))
        self.assertIn("REQUIRES --first-cell", self.refusal(r, r.host(plans), cell=2))
        self.assertIn("REQUIRES --first-cell", self.refusal(r, r.host(plans), cell=2, first=("/x/cell.json", None)))
        self.assertIn("belong to cell 2 only", self.refusal(r, r.host(plans), cell=1, first=("/x/cell.json", "0" * 64)))
        r.write_sys("unique_id", "deadbeef")
        self.assertIn("unique_id", self.refusal(r, r.host(plans)))
        r.write_sys("unique_id", UUID.removeprefix("GPU-"))
        r.write_sys("vendor", "0xffff")
        self.assertIn("device check failed", self.refusal(r, r.host(plans)))
        r.write_sys("vendor", "0x1002")
        os.unlink(r.watch)
        self.assertIn("watcher log unreadable", self.refusal(r, r.host(plans)))

    def test_preflight_failure_refuses_run(self):
        r = self.rig()
        with open(r.arms["A"]["daemon"], "a") as f:
            f.write("# tamper\n")
        self.assertIn("artifact", self.refusal(r, r.host([ok_plan([1.0] * 3)])))

    def test_writable_installed_pack_refuses_run_before_spawn(self):
        r = self.rig()
        os.chmod(r.arms["B"]["pack"], 0o755)                                 # the installed pack dir is not read-only any more
        self.assertIn("writable by mode", self.refusal(r, r.host([ok_plan([1.0] * 3)])))
        os.chmod(r.arms["B"]["pack"], 0o555)
        os.chmod(os.path.join(r.arms["A"]["pack"], "index.json"), 0o666)    # ... or one of its files
        self.assertIn("writable by mode", self.refusal(r, r.host([ok_plan([1.0] * 3)])))

    def test_remaining_budget_refused_before_mutation(self):
        budget = {"cell_cap_seconds": 20, "kill_grace_seconds": 2, "drain_seconds": 3, "finalize_seconds": 5, "preflight_seconds": 10,
                  "maximum_hold_seconds": 40}           # single-hold minimum 40; one full cell (cap+grace+drain+finalize) needs 30 AFTER preflight
        r = self.rig(budget=budget)
        plans = [ok_plan([2000.0] * 3)] * 4
        real_preflight = h2.preflight

        def slow(advance_s):
            def wrapper(manifest_path, sha_, host_):
                host_.now += int(advance_s * 1e9)                    # preflight consumed hold time
                return real_preflight(manifest_path, sha_, host_)
            return wrapper

        out = os.path.join(r.root, "out", "p1c1")
        os.makedirs(os.path.dirname(out))
        with mock.patch.object(h2, "preflight", slow(10.5)):        # 29.5 s remain < 30 s: refused before any mutation
            with self.assertRaises(h2.Refusal) as cm:
                h2.run_cell(r.manifest_path, r.msha, 1, 1, out, [], None, None, r.host(plans, cls=ClockHost))
        self.assertIn("BUDGET", " ".join(cm.exception.errors))
        self.assertFalse(os.path.exists(out))
        self.assertEqual(self.calls(r), [])
        with mock.patch.object(h2, "preflight", slow(9.5)):         # 30.5 s remain: enough; the cap is never shrunk, the cell starts
            rec = h2.run_cell(r.manifest_path, r.msha, 1, 1, out, [], None, None, r.host(plans, cls=ClockHost))
        self.assertEqual(rec["status"], "OK")
        self.assertEqual(self.calls(r), ["A"])
        hold = rec["hold"]
        self.assertEqual((hold["required_cell_seconds"], hold["required_single_hold_seconds"], hold["maximum_hold_seconds"]), (30, 40, 40))
        self.assertGreaterEqual(hold["remaining_at_start_seconds"], 30)
        self.assertLess(hold["remaining_at_start_seconds"], 31)

    def test_each_hold_has_its_own_clock_and_full_budget(self):
        budget = {"cell_cap_seconds": 20, "kill_grace_seconds": 2, "drain_seconds": 3, "finalize_seconds": 5, "preflight_seconds": 5,
                  "maximum_hold_seconds": 35}                       # exactly one single-process hold: two cells could never share it
        r = self.rig(budget=budget)
        plans = [ok_plan([2000.0] * 3)] * 2
        host1 = r.host(plans, cls=ClockHost)
        o1, o2 = os.path.join(r.root, "out", "p1c1"), os.path.join(r.root, "out", "p1c2")
        rec1 = self.run_cell(r, plans, 1, 1, o1, host=host1)
        self.assertEqual(rec1["status"], "OK")
        c1 = self.cell_file(o1, rec1)
        host2 = r.host(plans, cls=ClockHost, start_ns=host1.now + 10 ** 9)       # a LATER, separate hold with its own clock
        rec2 = self.run_cell(r, plans, 1, 2, o2, first=(c1, self.sha_file(c1)), host=host2)
        self.assertEqual(rec2["status"], "OK")
        self.assertGreater(rec2["hold"]["hold_start_monotonic_ns"], rec1["hold"]["hold_end_monotonic_ns"])
        self.assertAlmostEqual(rec2["hold"]["remaining_at_start_seconds"], 35, delta=0.5)   # the FULL budget again, not 35 minus cell 1's use
        self.assertGreater(rec1["hold"]["used_seconds_at_end"], 0)
        self.assertEqual(self.calls(r), ["A", "B"])

    def test_later_pair_needs_complete_previous_and_same_hold(self):
        r = self.rig()
        plans = [ok_plan([2000.0] * 3), {"mode": "exit3"}, ok_plan([2000.0] * 3), ok_plan([2000.0] * 3)]
        res = self.run_pair_cells(r, plans, 1)
        self.assertEqual(res["doc"]["status"], "PAIR_INCOMPLETE")
        self.assertEqual(res["r2"]["status"], "EXIT_NONZERO")
        before = self.calls(r)
        out2 = os.path.join(r.root, "out", "p2c1")
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, r.msha, 2, 1, out2, [res["path"]], None, None, r.host(plans))
        self.assertIn("PAIR_INCOMPLETE", " ".join(cm.exception.errors))
        self.assertFalse(os.path.exists(out2))
        self.assertEqual(self.calls(r), before)
        # a receipt of another manifest/harness is refused
        r2 = self.rig()
        m = copy.deepcopy(r2.m)
        m["label"] = "different manifest"
        r2.save(m)
        good = self.run_pair_cells(self.rig(), [ok_plan([2000.0] * 3)] * 2, 1)
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r2.manifest_path, r2.msha, 2, 1, os.path.join(r2.root, "o2"), [good["path"]], None, None, r2.host([ok_plan([1.0] * 3)] * 2))
        self.assertIn("manifest_sha256", " ".join(cm.exception.errors))

    # ---------------------------------------------------------------- failure handling
    def test_nonzero_exit_stops_without_zero_fill_and_no_cell_two(self):
        r, rec, out = self.one_cell([{"mode": "exit3"}, ok_plan([2000.0] * 3)])
        self.assertEqual(rec["status"], "EXIT_NONZERO")
        self.assertEqual(rec["failures"], ["EXIT_NONZERO"])
        self.assertEqual(rec["process"]["returncode"], 3)
        self.assertIsNone(rec["bench"])
        self.assertFalse(rec["measurement_valid"])
        self.assertEqual(self.calls(r), ["A"])                                            # no automatic rerun, no second process
        self.assertTrue(h2.seal_ok(rec))
        self.assertTrue(os.path.exists(os.path.join(out, "A1", "stderr.log")))             # closed files preserved
        self.assert_second_cell_refused(r, rec, out)                                       # nothing resumes on the GPU after a failure

    def test_failed_second_cell_closes_pair_incomplete_and_judge_is_unknown(self):
        r = self.rig()
        res = self.run_pair_cells(r, [ok_plan([2000.0] * 3), {"mode": "exit3"}], 1)
        self.assertEqual((res["r1"]["status"], res["r2"]["status"]), ("OK", "EXIT_NONZERO"))
        doc = res["doc"]
        self.assertEqual(doc["status"], "PAIR_INCOMPLETE")
        self.assertIn("B1 EXIT_NONZERO", doc["stop_reason"])
        self.assertTrue(h2.seal_ok(doc))
        self.assertEqual(self.calls(r), ["A", "B"])
        j = self.judge(r, [res["path"]])
        self.assertEqual(j["verdict"], "UNKNOWN_INCOMPLETE")
        self.assertEqual(j["errors"], [])
        self.assertEqual(j["pairs_incomplete"][0]["pair"], 1)
        self.assertNotIn("median_all_samples", j["candidate"])
        self.assertEqual(j["candidate"]["samples_flat"], [])
        self.assertEqual(j["incumbent"]["samples_flat"], [])                               # the good cell 1 of an incomplete pair is never used

    def test_invalid_output_variants(self):
        for plan in ({"mode": "badjson"}, {"mode": "raw", "raw": "banner\n"}, {"mode": "raw", "raw": ""},
                     ok_plan([2000.0, 0.0, 2000.0]), ok_plan([2000.0, 2000.0]),
                     ok_plan([2000.0, 2000.0, 2000.0], patch={"model": "/other/model"})):
            r, rec, out = self.one_cell([plan, ok_plan([2000.0] * 3)])
            self.assertEqual(rec["status"], "INVALID_OUTPUT", plan)
            self.assertFalse(rec["measurement_valid"])
            self.assert_second_cell_refused(r, rec, out)

    def test_timeout_terminates_own_group_and_reaps_children(self):
        budget = {"cell_cap_seconds": 1, "kill_grace_seconds": 1, "drain_seconds": 1, "finalize_seconds": 1, "preflight_seconds": 1,
                  "maximum_hold_seconds": 600}
        r, c, out = self.one_cell([{"mode": "sleep"}, ok_plan([2000.0] * 3)], rig_kw={"budget": budget})
        self.assertEqual(c["status"], "TIMEOUT")
        self.assertTrue(c["process"]["timed_out"])
        self.assertEqual(c["process"]["signal"], signal.SIGTERM)
        pid = c["process"]["pid"]
        self.assertEqual(c["process"]["pgid"], pid)
        sigs = [e.get("signal") for e in c["lifecycle"]["events"]]
        self.assertEqual(sigs, ["SIGTERM"])
        members = c["lifecycle"]["events"][0]["members"]
        self.assertEqual(len(members), 2)                              # the fake CLI AND its same-group child were signalled
        self.assertTrue(r.children())
        for child in r.children():
            self.assertFalse(alive(child), "child of the cell's own group must be gone")
        self.assertFalse(alive(pid))
        self.assertEqual(c["lifecycle"]["group_after_cleanup"], [])
        self.assertEqual(self.calls(r), ["A"])
        self.assertLess(c["process"]["elapsed_seconds"], 4)

    def test_sigterm_ignored_escalates_to_sigkill_of_own_group(self):
        budget = {"cell_cap_seconds": 1, "kill_grace_seconds": 1, "drain_seconds": 1, "finalize_seconds": 1, "preflight_seconds": 1,
                  "maximum_hold_seconds": 600}
        r, c, out = self.one_cell([{"mode": "stubborn"}, ok_plan([2000.0] * 3)], rig_kw={"budget": budget})
        self.assertEqual(c["status"], "TIMEOUT")
        self.assertEqual([e.get("signal") for e in c["lifecycle"]["events"]], ["SIGTERM", "SIGKILL"])
        self.assertEqual(c["process"]["signal"], signal.SIGKILL)
        for child in r.children():
            self.assertFalse(alive(child))
        self.assertEqual(c["lifecycle"]["group_after_cleanup"], [])

    def test_unreaped_daemon_after_cli_exit_is_swept_and_flagged(self):
        r, c, out = self.one_cell([{"mode": "leak", "samples": [2000.0] * 3}, ok_plan([2000.0] * 3)])
        self.assertEqual(c["process"]["returncode"], 0)
        self.assertEqual(c["status"], "LIFECYCLE_LEFTOVER")
        self.assertFalse(c["measurement_valid"])
        self.assertEqual(len(c["lifecycle"]["leftover_after_cli_exit"]), 1)
        for child in r.children():
            self.assertFalse(alive(child), "no detached daemon may leak")
        self.assert_second_cell_refused(r, c, out)

    def test_signal_aborts_cell_and_cleans_group(self):
        r = self.rig()
        host = r.host([{"mode": "sleep"}, ok_plan([2000.0] * 3)], cls=StopHost, after=4)
        out = os.path.join(r.root, "out", "p1c1")
        c = self.run_cell(r, [], 1, 1, out, host=host)
        self.assertEqual(c["status"], "ABORTED_SIGNAL")
        self.assertTrue(c["process"]["aborted"])
        self.assertFalse(c["measurement_valid"])
        for child in r.children():
            self.assertFalse(alive(child))
        self.assertEqual(self.calls(r), ["A"])

    def test_device_loss_after_cell_stops_without_retry(self):
        r = self.rig()
        vendor = os.path.join(r.sysdir, "vendor")
        plan = [ok_plan([2000.0] * 3, write=[[vendor, "0xffff\n", "w"]]), ok_plan([2000.0] * 3)]
        _, c, out = self.one_cell(plan, rig=r)
        self.assertEqual(c["status"], "DEVICE_LOST")
        self.assertFalse(c["device_after"]["ok"])
        self.assertEqual(self.calls(r), ["A"])

    def test_fatal_watcher_line_stops_the_pair(self):
        r = self.rig()
        plan = [ok_plan([2000.0] * 3, write=[[r.watch, "amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=1\n", "a"]]), ok_plan([2000.0] * 3)]
        _, c, out = self.one_cell(plan, rig=r)
        self.assertEqual(c["status"], "WATCHER_FATAL")
        self.assertEqual(c["watcher"]["match_count"], 1)
        self.assertEqual(self.calls(r), ["A"])
        # a benign amdgpu line is not fatal, and cell 1's lines are never re-counted by cell 2's own hold
        r2 = self.rig()
        res = self.run_pair_cells(r2, [ok_plan([2000.0] * 3, write=[[r2.watch, "amdgpu: all fine\n", "a"]]), ok_plan([2000.0] * 3)], 1)
        self.assertEqual(res["doc"]["status"], "PAIR_COMPLETE")
        self.assertEqual(res["r1"]["watcher"]["match_count"], 0)
        self.assertIsNone(res["r1"]["watcher"]["error"])
        self.assertEqual(res["r2"]["watcher"]["match_count"], 0)
        # the fatal line of the first hold's log does not poison the next hold's offset
        r3 = self.rig()
        fatal = "amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=1\n"
        write_text(r3.watch, "old\n" + fatal)                                     # already in the log BEFORE the hold: not a new line
        res = self.run_pair_cells(r3, [ok_plan([2000.0] * 3)] * 2, 1)
        self.assertEqual(res["doc"]["status"], "PAIR_COMPLETE")

    def hung_cell(self, r, write, extra=None, refusal_needle="failed first cell"):
        """A cell whose fake CLI (plus a same-group child) hangs far past the 20 s cap after injecting `write`; the harness must abort it early."""
        t0 = time.monotonic()
        _, c, out = self.one_cell([{"mode": "sleep", "write": write, **(extra or {})}, ok_plan([2000.0] * 3)], rig=r)
        self.assertLess(time.monotonic() - t0, 8, "the health poll must end the cell long before the 20 s cap")
        self.assertFalse(c["process"]["timed_out"])
        self.assertFalse(c["measurement_valid"])
        self.assertIsNone(c["bench"])
        self.assertEqual(self.calls(r), ["A"])
        events = c["lifecycle"]["events"]
        self.assertEqual(events[0]["signal"], "SIGTERM")                                             # only the recorded own group was signalled
        self.assertEqual(len(events[0]["members"]), 2)                                               # the fake CLI AND its same-group child
        self.assertEqual(c["process"]["pgid"], c["process"]["pid"])
        self.assertTrue(r.children())
        for child in r.children():
            self.assertFalse(alive(child))
        self.assertFalse(alive(c["process"]["pid"]))
        self.assertEqual(c["lifecycle"]["group_after_cleanup"], [])
        self.assertIsNotNone(c["health_abort"])
        self.assert_second_cell_refused(r, c, out, needle=refusal_needle)                            # the next cell never starts
        return c

    def test_device_change_mid_run_terminates_own_group_and_preserves_cause(self):
        r = self.rig()
        # the device stays 0xffff, so the (stronger) preflight device check refuses cell 2 before the failed-first-cell binding is reached
        c = self.hung_cell(r, [[os.path.join(r.sysdir, "device"), "0xffff\n", "w"]], refusal_needle="target device check failed")
        self.assertEqual(c["status"], "DEVICE_LOST")
        cause = c["health_abort"]["causes"][0]
        self.assertEqual(cause["status"], "DEVICE_LOST")
        self.assertTrue(any("device 0xffff" in p for p in cause["problems"]), cause)
        self.assertFalse(c["device_after"]["ok"])
        self.assertIn("DEVICE_LOST", c["failures"])

    def test_fatal_watcher_line_mid_run_terminates_own_group_and_preserves_matches(self):
        r = self.rig()
        line = "amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=7\n"
        c = self.hung_cell(r, [[r.watch, line, "a"]])
        self.assertEqual(c["status"], "WATCHER_FATAL")
        cause = c["health_abort"]["causes"][0]
        self.assertEqual(cause["status"], "WATCHER_FATAL")
        self.assertEqual(cause["matches"], [line.strip()])
        self.assertEqual(c["watcher"]["matches"], [line.strip()])
        self.assertEqual(c["watcher"]["match_count"], 1)                                             # seen once, not re-counted by the final scan
        self.assertTrue(c["watcher"]["fatal"])

    def test_unreadable_watcher_mid_run_terminates_own_group(self):
        r = self.rig()
        c = self.hung_cell(r, [[r.watch, "", "w"]])                                                  # log truncated below the recorded offset
        self.assertEqual(c["status"], "WATCHER_UNREADABLE")
        self.assertIn("shrank", c["health_abort"]["causes"][0]["error"])
        self.assertIn("shrank", c["watcher"]["error"])

    def test_watcher_symlink_is_followed_recorded_and_a_retarget_mid_run_is_fatal(self):
        r = self.rig()
        link = os.path.join(r.root, "campaign-watch.log")                                            # the parent's ONE campaign symlink
        os.symlink(r.watch, link)
        other = os.path.join(r.root, "other-watch.log")
        write_text(other, "some other wrapper log\n" * 10)                                           # longer: a shrink check cannot be what trips
        m = copy.deepcopy(r.m)
        m["watcher"]["log_path"] = link
        r.save(m)
        c = self.hung_cell(r, [], extra={"relink": [[link, other]]})                                 # re-pointed INSIDE the held command
        self.assertEqual(c["status"], "WATCHER_UNREADABLE")
        self.assertIn("identity changed", c["health_abort"]["causes"][0]["error"])
        ident = c["watcher"]["identity"]
        self.assertEqual((ident["log_path"], ident["realpath"]), (link, os.path.realpath(r.watch)))
        # a stable symlink works and its resolved identity is recorded in the receipt
        r2 = self.rig()
        link2 = os.path.join(r2.root, "campaign-watch.log")
        os.symlink(r2.watch, link2)
        m2 = copy.deepcopy(r2.m)
        m2["watcher"]["log_path"] = link2
        r2.save(m2)
        _, ok, _ = self.one_cell([ok_plan([2000.0] * 3)] * 2, rig=r2)
        self.assertEqual(ok["status"], "OK")
        st = os.stat(r2.watch)
        self.assertEqual((ok["watcher"]["identity"]["device"], ok["watcher"]["identity"]["inode"]), (st.st_dev, st.st_ino))
        self.assertEqual(ok["watcher"]["identity"]["realpath"], os.path.realpath(r2.watch))

    def test_health_callback_cadence_and_own_group_only(self):
        r = self.rig()
        host = ClockHost({})
        calls = []

        def health():
            calls.append(host.now)
            return [{"status": "DEVICE_LOST", "problems": ["injected"]}] if len(calls) == 4 else []

        so, se, sp = (os.path.join(r.root, n) for n in ("cad.out", "cad.err", "cad.proc"))
        res = h2.spawn_and_wait([sys.executable, "-c", "import time; time.sleep(300)"], dict(os.environ), r.root, so, se, sp, 60, 1, host, health=health)
        self.assertEqual(len(calls), 4)
        gaps = [b - a for a, b in zip(calls, calls[1:])]
        self.assertTrue(all(abs(g - int(h2.HEALTH_INTERVAL_S * 1e9)) <= int(h2.POLL_S * 1e9) for g in gaps), gaps)
        self.assertEqual(res["health_abort"]["causes"][0]["status"], "DEVICE_LOST")
        self.assertFalse(res["timed_out"])
        self.assertFalse(res["aborted"])
        self.assertEqual([e["signal"] for e in res["events"]], ["SIGTERM"])
        self.assertEqual(res["events"][0]["members"][0]["pid"], res["process"]["pid"])
        self.assertEqual(res["group_after_cleanup"], [])
        self.assertFalse(alive(res["process"]["pid"]))

    def test_drain_timeout_when_vram_stays_busy(self):
        r = self.rig()
        used = os.path.join(r.sysdir, "mem_info_vram_used")
        _, c, out = self.one_cell([ok_plan([2000.0] * 3, write=[[used, str(5 * GIB) + "\n", "w"]]), ok_plan([2000.0] * 3)], rig=r)
        self.assertEqual(c["status"], "DRAIN_TIMEOUT")
        self.assertFalse(c["drain"]["idle"])
        self.assertEqual(c["drain"]["last_used"], 5 * GIB)
        self.assert_second_cell_refused(r, c, out)

    def test_input_change_during_cell_invalidates_it(self):
        r = self.rig()
        _, c, _ = self.one_cell([ok_plan([2000.0] * 3, utime=[[r.model, MTIME_19 + 5]]), ok_plan([2000.0] * 3)], rig=r)
        self.assertEqual(c["status"], "INPUT_CHANGED")
        self.assertFalse(c["integrity"]["cached_inputs_unchanged"])
        r2 = self.rig()
        _, c2, _ = self.one_cell([ok_plan([2000.0] * 3, create=[r2.model + ".trunk-overlay"]), ok_plan([2000.0] * 3)], rig=r2)
        self.assertEqual(c2["status"], "INPUT_CHANGED")
        self.assertFalse(c2["integrity"]["absent_paths_absent"])

    def test_pack_permission_change_during_cell_is_input_change(self):
        r = self.rig()
        hsaco = os.path.join(r.arms["A"]["pack"], "pack.hsaco")
        _, c, _ = self.one_cell([ok_plan([2000.0] * 3, chmod=[[hsaco, 0o644]]), ok_plan([2000.0] * 3)], rig=r)
        self.assertEqual(c["status"], "INPUT_CHANGED")
        self.assertFalse(c["integrity"]["pack_dirs_unchanged"])
        self.assertIn("packs", c["integrity"]["changed_sections"])

    @unittest.skipIf(os.geteuid() == 0, "root can write a chmod a-w pack: the read-only proof needs an unprivileged user")
    def test_pack_stays_readonly_and_jit_writes_only_to_the_private_hot_cache(self):
        r = self.rig()
        pack_a = r.arms["A"]["pack"]
        probe = os.path.join(pack_a, "jit-metadata.bin")
        plans = [ok_plan([2000.0] * 3, kcache_write=["jit1.hsaco", "jit2.hsaco"], try_write=[probe]),
                 ok_plan([2000.0] * 3, kcache_write=["jit3.hsaco"])]
        res = self.run_pair_cells(r, plans, 1)
        c1, c2 = res["r1"], res["r2"]
        self.assertEqual(c1["status"], "OK", c1["failures"])
        self.assertEqual(c2["status"], "OK", c2["failures"])
        self.assertIn("TRYWRITE err PermissionError", read_text(os.path.join(res["o1"], "A1", "stderr.log")))   # the pack rejected the write
        self.assertFalse(os.path.exists(probe))
        k1, k2 = c1["hot_cache"], c2["hot_cache"]
        self.assertEqual(k1["path"], os.path.join(res["o1"], "A1", "kcache"))
        self.assertEqual(k2["path"], os.path.join(res["o2"], "B1", "kcache"))
        self.assertNotEqual(k1["path"], k2["path"])                                                           # private per cell, never shared
        self.assertEqual((k1["files_after"], k2["files_after"]), (2, 1))
        self.assertTrue(os.path.exists(os.path.join(k1["path"], "jit1.hsaco")))
        self.assertFalse(os.path.exists(os.path.join(k2["path"], "jit1.hsaco")))
        self.assertTrue(c1["integrity"]["pack_dirs_unchanged"] and c1["integrity"]["artifacts_unchanged"])
        self.assertEqual(c1["pack_readonly"], {"verified": True, "dirs": 1, "files": 2, "effective_access_checked": True})
        self.assertEqual(self.judge(r, [res["path"]])["errors"], [])

    def test_exception_after_out_is_created_closes_the_hold_with_a_sealed_failure_receipt(self):
        r = self.rig()
        out = os.path.join(r.root, "out", "p1c1")
        with mock.patch.object(h2, "_prepare_cell_dir", side_effect=OSError("disk full")):
            rec = self.run_cell(r, [ok_plan([2000.0] * 3)] * 2, 1, 1, out)
        self.assertEqual((rec["status"], rec["failures"]), ("HARNESS_EXCEPTION", ["HARNESS_EXCEPTION"]))
        self.assertTrue(h2.seal_ok(rec))
        self.assertFalse(rec["measurement_valid"])
        self.assertIn("disk full", rec["exception"])
        self.assertEqual(read_json(self.cell_file(out, rec)), rec)
        self.assertEqual(self.calls(r), [])                                                  # nothing was ever spawned, nothing is claimed
        errs, ids = [], h2.new_ids()
        got = h2.check_cell_binding(r.m, r.msha, h2.self_sha256(), rec, 1, 0, "A", os.path.dirname(self.cell_file(out, rec)), errs, ids)
        self.assertIsNone(got)                                                               # a failed (UNKNOWN) cell, never a measurement
        self.assertEqual(errs, [])
        self.assert_second_cell_refused(r, rec, out)

    def test_spawn_error_is_recorded(self):
        r = self.rig()
        os.chmod(r.arms["A"]["cli"], 0o644)           # not executable: preflight refuses before anything is spawned
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, r.msha, 1, 1, os.path.join(r.root, "o"), [], None, None, r.host([ok_plan([1.0] * 3)]))
        self.assertIn("not executable", " ".join(cm.exception.errors))
        res = h2.spawn_and_wait(["/nonexistent/cli"], {}, r.root, os.path.join(r.root, "so"), os.path.join(r.root, "se"),
                                os.path.join(r.root, "sp"), 1, 1, h2.Host({}))
        self.assertFalse(res["spawned"])
        self.assertIn("spawn_error", res)

    def test_cli_run_exit_codes_one_cell(self):
        r = self.rig()
        out = os.path.join(r.root, "out", "p1c1")
        os.makedirs(os.path.dirname(out))
        argv = ["run", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--pair", "1", "--cell", "1", "--out", out]
        plans = [ok_plan([2000.0] * 3), {"mode": "exit3"}]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = h2.main(argv, r.host(plans))
        self.assertEqual(rc, 0)
        summary = json.loads(buf.getvalue())
        self.assertEqual((summary["status"], summary["cell_id"], summary["cell"]), ("CELL_OK", "A1", 1))
        self.assertEqual(summary["cell_file_sha256"], self.sha_file(summary["cell_json"]))
        self.assertTrue(summary["fabric_job"].startswith("job-"))
        self.assertEqual(self.calls(r), ["A"])                                              # exactly one process for one invocation
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = h2.main(argv, r.host(plans))                                          # out exists now
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(buf.getvalue())["status"], "REFUSED")
        self.assertEqual(self.calls(r), ["A"])
        out2 = os.path.join(r.root, "out", "p1c1-failed")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = h2.main(["run", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--pair", "1", "--cell", "1", "--out", out2],
                         r.host(plans))
        self.assertEqual(rc, 3)
        self.assertEqual(json.loads(buf.getvalue())["cell_status"], "EXIT_NONZERO")

    def test_cli_cell2_assemble_pair_and_judge_end_to_end(self):
        r = self.rig()
        plans = hold_plan(r.roles, [[2400.0] * 3] * 3, [[1000.0] * 3] * 3)
        os.makedirs(os.path.join(r.root, "out"))
        o1, o2 = os.path.join(r.root, "out", "p1c1"), os.path.join(r.root, "out", "p1c2")
        base = ["--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha]

        def cli(argv, host):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = h2.main(argv, host)
            return rc, json.loads(buf.getvalue())

        rc, out1 = cli(["run", *base, "--pair", "1", "--cell", "1", "--out", o1], r.host(plans))
        self.assertEqual(rc, 0)
        rc, refused = cli(["run", *base, "--pair", "1", "--cell", "2", "--out", o2], r.host(plans))         # cell 2 without the independent binding
        self.assertEqual((rc, refused["status"]), (2, "REFUSED"))
        self.assertTrue(any("REQUIRES --first-cell" in e for e in refused["errors"]), refused)
        self.assertFalse(os.path.exists(o2))
        self.assertEqual(self.calls(r), ["A"])
        rc, out2 = cli(["run", *base, "--pair", "1", "--cell", "2", "--out", o2, "--first-cell", out1["cell_json"],
                        "--first-cell-sha256", out1["cell_file_sha256"]], r.host(plans))
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls(r), ["A", "B"])
        self.assertNotEqual(out1["fabric_job"], out2["fabric_job"])
        pair_path = os.path.join(r.root, "out", "pair1.json")
        asm = ["assemble-pair", *base, "--cells", out1["cell_json"], out2["cell_json"], "--cell-sha256", out1["cell_file_sha256"],
               out2["cell_file_sha256"], "--out", pair_path]
        with mock.patch.object(h2.subprocess, "Popen", side_effect=AssertionError("assemble-pair is pure CPU and must never spawn")):
            rc, asm_out = cli(asm, r.host([]))
        self.assertEqual((rc, asm_out["status"]), (0, "PAIR_COMPLETE"))
        self.assertEqual(asm_out["pair_file_sha256"], self.sha_file(pair_path))
        self.assertEqual(asm_out["fabric_jobs"], [out1["fabric_job"], out2["fabric_job"]])
        rc, again = cli(asm, r.host([]))                                                                      # never overwrites
        self.assertEqual((rc, again["status"]), (2, "REFUSED"))
        rc, j = cli(["judge", *base, "--pairs", pair_path, "--out", os.path.join(r.root, "res.json")], r.host([]))
        self.assertEqual((rc, j["verdict"]), (3, "UNKNOWN_INCOMPLETE"))


# ======================================================================================================================================
class HoldSplitTests(Base):
    """Six single-process holds, logical pair receipts, strict cell-1 binding."""

    def test_exactly_one_spawn_per_invocation_and_fresh_separate_cells(self):
        r = self.rig()
        plans = hold_plan(r.roles, [[2400.0] * 3] * 3, [[1000.0] * 3] * 3)
        o1, o2 = os.path.join(r.root, "out", "p1c1"), os.path.join(r.root, "out", "p1c2")
        rec1 = self.run_cell(r, plans, 1, 1, o1)
        self.assertEqual(self.calls(r), ["A"])                                      # ONE invocation = ONE fresh CLI/daemon
        self.assertEqual(sorted(os.listdir(o1)), ["A1", "preflight.json"])
        c1 = self.cell_file(o1, rec1)
        sha1 = self.sha_file(c1)
        rec2 = self.run_cell(r, plans, 1, 2, o2, first=(c1, sha1))
        self.assertEqual(self.calls(r), ["A", "B"])
        self.assertEqual(sorted(os.listdir(o2)), ["B1", "preflight.json"])
        self.assertEqual(self.sha_file(c1), sha1)                                   # cell 2 never mutated cell 1's data
        self.assertEqual([rec1["position"], rec2["position"]], [0, 1])
        self.assertNotEqual(rec1["process"]["pid"], rec2["process"]["pid"])
        self.assertNotEqual(rec1["home"]["path"], rec2["home"]["path"])
        self.assertNotEqual(rec1["hot_cache"]["path"], rec2["hot_cache"]["path"])
        self.assertNotEqual(rec1["env"]["fabric_job"], rec2["env"]["fabric_job"])   # distinct ACTUAL holds, no retroactive fictitious one
        self.assertIsNone(rec1["first_cell"])
        fc = rec2["first_cell"]
        self.assertEqual((fc["file_sha256"], fc["receipt_sha256"], fc["cell_id"], fc["fabric_job"]),
                         (sha1, rec1["receipt_sha256"], "A1", rec1["env"]["fabric_job"]))
        self.assertEqual(rec2["hold"]["boot_id"], rec1["hold"]["boot_id"])
        self.assertGreater(rec2["hold"]["hold_start_monotonic_ns"], rec1["hold"]["hold_end_monotonic_ns"])
        self.assertEqual(rec1["hold"]["budget"], r.m["budget"])
        pair_path = os.path.join(r.root, "out", "pair1", "pair.json")
        os.makedirs(os.path.dirname(pair_path))
        c2 = self.cell_file(o2, rec2)
        with mock.patch.object(h2.subprocess, "Popen", side_effect=AssertionError("assemble-pair is pure CPU and must never spawn")), \
                mock.patch.object(h2, "spawn_and_wait", side_effect=AssertionError("assemble-pair is pure CPU and must never spawn")):
            doc = h2.assemble_pair(r.manifest_path, r.msha, [c1, c2], [sha1, self.sha_file(c2)], [], pair_path)
        self.assertEqual(doc["status"], "PAIR_COMPLETE")
        self.assertEqual(doc["fabric_jobs"], [rec1["env"]["fabric_job"], rec2["env"]["fabric_job"]])
        self.assertEqual([f["relpath"] for f in doc["cell_files"]],
                         [os.path.relpath(c1, os.path.dirname(pair_path)), os.path.relpath(c2, os.path.dirname(pair_path))])
        self.assertEqual(doc["cells"], [rec1, rec2])
        self.assertEqual(self.calls(r), ["A", "B"])
        j = self.judge(r, [pair_path])
        self.assertEqual((j["verdict"], j["errors"], len(j["fabric_jobs"])), ("UNKNOWN_INCOMPLETE", [], 2))

    def test_second_cell_requires_independent_first_cell_binding(self):
        r = self.rig()
        plans = [ok_plan([2400.0] * 3)] * 12
        o1 = os.path.join(r.root, "out", "p1c1")
        rec1 = self.run_cell(r, plans, 1, 1, o1)
        c1 = self.cell_file(o1, rec1)
        sha1 = self.sha_file(c1)
        before = self.calls(r)
        o2 = os.path.join(r.root, "out", "p1c2")

        def refuse(needle, first, host=None, out=o2):
            f_path, f_sha = first if first else (None, None)
            with self.assertRaises(h2.Refusal) as cm:
                h2.run_cell(r.manifest_path, r.msha, 1, 2, out, [], f_path, f_sha, host or r.host(plans))
            self.assertIn(needle, " | ".join(cm.exception.errors))
            self.assertFalse(os.path.exists(out))
            self.assertEqual(self.calls(r), before)

        refuse("REQUIRES --first-cell", None)
        refuse("REQUIRES --first-cell", (c1, None))
        refuse("REQUIRES --first-cell", (None, sha1))
        refuse("must be 64 lowercase hex", (c1, "xyz"))
        refuse("independently supplied", (c1, "0" * 64))                          # a hash the receipts did not supply
        refuse("unreadable", (os.path.join(r.root, "no-such-cell.json"), sha1))
        # a copy of cell 1 edited after the fact: the independently recorded hash no longer matches
        shutil.copytree(o1, o1 + "-edited")
        edited = os.path.join(o1 + "-edited", "A1", "cell.json")
        write_bytes(edited, read_bytes(edited) + b"\n")
        refuse("independently supplied", (edited, sha1))
        # edited AND the attacker supplies the matching hash: the seal / bindings still refuse
        doc = read_json(edited)
        doc["argv"][-1] = "--nope"
        write_text(edited, json.dumps(doc))
        refuse("receipt_sha256 does not match", (edited, self.sha_file(edited)))
        write_text(edited, json.dumps(h2.seal(doc)))
        refuse("argv", (edited, self.sha_file(edited)))
        # a first cell that is not cell 1 of THIS pair/order: the second cell of the same pair, or a cell of another manifest
        rec2 = self.run_cell(r, plans, 1, 2, o2, first=(c1, sha1))
        c2 = self.cell_file(o2, rec2)
        before = self.calls(r)
        refuse("position", (c2, self.sha_file(c2)), out=os.path.join(r.root, "out", "p1c2-again"))
        other = self.rig()
        o_other = os.path.join(other.root, "out", "p1c1")
        rec_other = self.run_cell(other, plans, 1, 1, o_other)
        c_other = self.cell_file(o_other, rec_other)
        refuse("manifest_sha256", (c_other, self.sha_file(c_other)), out=os.path.join(r.root, "out", "p1c2-other"))
        # the same fabric job / the first cell's OUT dir / another boot are never a second hold
        refuse("already ran an earlier cell", (c1, sha1), host=r.host(plans, extra_env={"FABRIC_JOB": rec1["env"]["fabric_job"]}),
               out=os.path.join(r.root, "out", "p1c2-samejob"))
        refuse("earlier cell's OUT dir", (c1, sha1), out=os.path.join(o1, "nested"))
        with mock.patch.object(h2.Host, "boot_id", lambda self: "some-other-boot"):
            refuse("different boot", (c1, sha1), out=os.path.join(r.root, "out", "p1c2-boot"))
        with self.assertRaises(h2.Refusal) as cm:                                      # the exact first OUT is not fresh
            h2.run_cell(r.manifest_path, r.msha, 1, 2, o1, [], c1, sha1, r.host(plans))
        self.assertIn("FRESH", " ".join(cm.exception.errors))
        self.assertEqual(self.calls(r), before)

    def fixture(self):
        """Pair 1 = x1+x2, an extra cell-1 run y1, pair 2 = z1+z2 chained on the assembled pair 1 (all real single-process holds)."""
        r = self.rig()
        plans = [ok_plan([2400.0] * 3)] * 16
        f = {"r": r}

        def run(name, pair, cell, previous=(), first=None):
            out = os.path.join(r.root, "out", name)
            rec = self.run_cell(r, plans, pair, cell, out, previous, first)
            path = self.cell_file(out, rec)
            f[name] = (rec, path, self.sha_file(path))
            self.assertEqual(rec["status"], "OK", rec["failures"])
            return f[name]

        x1 = run("x1", 1, 1)
        run("x2", 1, 2, first=(x1[1], x1[2]))
        run("y1", 1, 1)
        f["pair1"] = os.path.join(r.root, "out", "pair1", "pair.json")
        os.makedirs(os.path.dirname(f["pair1"]))
        h2.assemble_pair(r.manifest_path, r.msha, [f["x1"][1], f["x2"][1]], [f["x1"][2], f["x2"][2]], [], f["pair1"])
        z1 = run("z1", 2, 1, previous=[f["pair1"]])
        run("z2", 2, 2, previous=[f["pair1"]], first=(z1[1], z1[2]))
        return f

    def edited(self, f, name, mutate, reseal=True):
        """A copy of cell `name`'s OUT dir with the cell receipt edited (and resealed). Returns (cell.json path, its sha256)."""
        rec, path, _ = f[name]
        out = os.path.dirname(os.path.dirname(path))
        dst = tempfile.mkdtemp(prefix=f"edit-{name}-", dir=os.path.join(f["r"].root, "out"))
        shutil.rmtree(dst)
        shutil.copytree(out, dst)
        p = os.path.join(dst, rec["cell_id"], "cell.json")
        doc = read_json(p)
        mutate(doc)
        write_text(p, json.dumps(h2.seal(doc) if reseal else doc))
        return p, self.sha_file(p)

    def test_assemble_pair_rejects_missing_duplicate_swapped_mixed_and_edited_cells(self):
        f = self.fixture()
        r = f["r"]
        (x1, cx1, sx1), (x2, cx2, sx2), (y1, cy1, sy1) = f["x1"], f["x2"], f["y1"]
        (z1, cz1, sz1), (z2, cz2, sz2) = f["z1"], f["z2"]
        count = [0]

        def refuse(paths, needle, shas=None, previous=(), manifest_sha=None):
            count[0] += 1
            out = os.path.join(r.root, "out", f"asm{count[0]}", "pair.json")
            os.makedirs(os.path.dirname(out))
            shas = shas if shas is not None else [self.sha_file(p) if os.path.exists(p) else "0" * 64 for p in paths]
            with self.assertRaises(h2.Refusal) as cm:
                h2.assemble_pair(r.manifest_path, manifest_sha or r.msha, list(paths), list(shas), list(previous), out)
            self.assertIn(needle, " | ".join(cm.exception.errors))
            self.assertFalse(os.path.exists(out), "a refused assembly must write nothing")

        calls = self.calls(r)
        # missing
        refuse([cx1], "exactly two", shas=[sx1])
        refuse([cx1, cx2], "exactly two", shas=[sx1])
        refuse([cx1, os.path.join(r.root, "no-such-cell.json")], "unreadable", shas=[sx1, sx2])
        # duplicate (same path, or identical bytes elsewhere)
        refuse([cx1, cx1], "duplicate cell")
        shutil.copytree(os.path.dirname(os.path.dirname(cx1)), os.path.join(r.root, "out", "x1-copy"))
        refuse([cx1, os.path.join(r.root, "out", "x1-copy", "A1", "cell.json")], "duplicate cell")
        # swapped
        refuse([cx2, cx1], "swapped")
        # mixed pairs / mixed runs of one pair
        refuse([cx1, cz2], "mixed pairs", previous=[f["pair1"]])
        refuse([cy1, cx2], "does not bind this cell 1")                                  # cell 2 was bound to x1, not to y1
        refuse([cx1, cz2], "mixed pairs")
        # wrong independently supplied hash / wrong manifest sha
        refuse([cx1, cx2], "independently supplied", shas=[sx1, "0" * 64])
        refuse([cx1, cx2], "independently supplied", shas=[sy1, sx2])
        refuse([cx1, cx2], "independently supplied expected", manifest_sha="1" * 64)
        # an edited cell 2 (hash recomputed by the attacker): seal, argv, binding, process, job, OUT all still refuse
        p, s = self.edited(f, "x2", lambda d: d["argv"].__setitem__(-1, "--nope"), reseal=False)
        refuse([cx1, p], "receipt_sha256 does not match", shas=[sx1, s])
        p, s = self.edited(f, "x2", lambda d: d["argv"].__setitem__(-1, "--nope"))
        refuse([cx1, p], "argv", shas=[sx1, s])

        def alter_binding(d):
            d["first_cell"]["file_sha256"] = "a" * 64
        p, s = self.edited(f, "x2", alter_binding)                                       # altered first-cell binding
        refuse([cx1, p], "first_cell.file_sha256 does not bind this cell 1", shas=[sx1, s])
        p, s = self.edited(f, "x2", lambda d: d["first_cell"].__setitem__("fabric_job", "job-forged"))
        refuse([cx1, p], "first_cell.fabric_job does not bind this cell 1", shas=[sx1, s])

        def same_process(d):
            d["process"]["pid"], d["process"]["starttime_ticks"] = x1["process"]["pid"], x1["process"]["starttime_ticks"]
        p, s = self.edited(f, "x2", same_process)
        refuse([cx1, p], "reused", shas=[sx1, s])
        p, s = self.edited(f, "x2", lambda d: d["env"].__setitem__("fabric_job", x1["env"]["fabric_job"]))
        refuse([cx1, p], "already used by another cell", shas=[sx1, s])
        p, s = self.edited(f, "x2", lambda d: d["env"].pop("fabric_job"))
        refuse([cx1, p], "FABRIC_JOB of this hold is not recorded", shas=[sx1, s])
        p, s = self.edited(f, "x2", lambda d: d["hold"].__setitem__("hold_start_monotonic_ns", 1))
        refuse([cx1, p], "strictly after the previous hold", shas=[sx1, s])
        shutil.copytree(os.path.dirname(cx2), os.path.join(os.path.dirname(os.path.dirname(cx1)), "B1"))   # cell 2's data inside cell 1's OUT dir
        shared = os.path.join(os.path.dirname(os.path.dirname(cx1)), "B1", "cell.json")
        refuse([cx1, shared], "shares its OUT dir", shas=[sx1, self.sha_file(shared)])
        # cell 1 edited to a failure while cell 2 still carries a binding to the good one: nothing may follow a failure
        p, s = self.edited(f, "x1", lambda d: (d.__setitem__("status", "EXIT_NONZERO"), d.__setitem__("measurement_valid", False)))
        refuse([p, cx2], "cell 2 exists although cell 1 is EXIT_NONZERO", shas=[s, sx2])
        # chain: pair 2 needs exactly the one earlier pair, complete and in order
        refuse([cz1, cz2], "needs exactly 1 --previous")
        # an existing OUT is never overwritten
        out = os.path.join(r.root, "out", "exists.json")
        write_text(out, "x")
        with self.assertRaises(h2.Refusal) as cm:
            h2.assemble_pair(r.manifest_path, r.msha, [cx1, cx2], [sx1, sx2], [], out)
        self.assertIn("already exists", " ".join(cm.exception.errors))
        self.assertEqual(read_text(out), "x")
        self.assertEqual(self.calls(r), calls)                                               # not one more process for any of it
        # and the good assemblies still work
        ok2 = os.path.join(r.root, "out", "pair2", "pair.json")
        os.makedirs(os.path.dirname(ok2))
        doc = h2.assemble_pair(r.manifest_path, r.msha, [cz1, cz2], [sz1, sz2], [f["pair1"]], ok2)
        self.assertEqual((doc["status"], doc["previous_pairs"][0]["pair"]), ("PAIR_COMPLETE", 1))
        self.assertEqual(doc["previous_pairs"][0]["receipt_sha256"], read_json(f["pair1"])["receipt_sha256"])
        self.assertEqual(self.judge(r, [f["pair1"], ok2])["errors"], [])

    def test_pair_chain_abbaab_common_manifest_distinct_jobs_and_sequential_holds(self):
        r = self.rig()
        docs, paths = self.run_pairs(r, hold_plan(r.roles, [[2400.0] * 3] * 3, [[1000.0] * 3] * 3))
        cells = [c for d in docs for c in d["cells"]]
        self.assertEqual([c["cell_id"] for c in cells], list(h2.GLOBAL_ORDER))
        self.assertEqual([c["arm"] for c in cells], list("ABBAAB"))
        self.assertEqual({c["manifest_sha256"] for c in cells} | {d["manifest_sha256"] for d in docs}, {r.msha})   # ONE common manifest sha
        jobs = [c["env"]["fabric_job"] for c in cells]
        self.assertEqual(len(set(jobs)), 6)
        for a, b in zip(cells, cells[1:]):
            self.assertLess(a["hold"]["hold_end_monotonic_ns"], b["hold"]["hold_start_monotonic_ns"])
            self.assertLess(a["process"]["start_monotonic_ns"], b["process"]["start_monotonic_ns"])
        for c in cells:
            self.assertEqual(c["hold"]["budget"], r.m["budget"])
            self.assertGreaterEqual(c["hold"]["remaining_at_start_seconds"], c["hold"]["required_cell_seconds"])
            self.assertLessEqual(c["hold"]["used_seconds_at_end"], c["hold"]["maximum_hold_seconds"])
        self.assertEqual(docs[0]["previous_pairs"], [])
        self.assertEqual(docs[1]["previous_pairs"], [{"pair": 1, "receipt_sha256": docs[0]["receipt_sha256"]}])
        self.assertEqual(docs[2]["previous_pairs"], [{"pair": 1, "receipt_sha256": docs[0]["receipt_sha256"]},
                                                     {"pair": 2, "receipt_sha256": docs[1]["receipt_sha256"]}])
        for d in docs:
            for c in d["cells"]:
                self.assertEqual(c["previous_pairs"], d["previous_pairs"])
        # the chain cannot be reordered, skipped or reused at run time
        calls = self.calls(r)
        out3 = os.path.join(r.root, "out", "p3c1-x")
        for previous, needle in (([paths[1], paths[0]], "--previous #1 is pair 2"), ([paths[0]], "needs exactly 2"), ([], "needs exactly 2")):
            with self.assertRaises(h2.Refusal) as cm:
                h2.run_cell(r.manifest_path, r.msha, 3, 1, out3, previous, None, None, r.host([ok_plan([2000.0] * 3)] * 2))
            self.assertIn(needle, " ".join(cm.exception.errors))
            self.assertFalse(os.path.exists(out3))
        self.assertEqual(self.calls(r), calls)
        # all-nine judge and the early order-statistic verdict keep their semantics over the logical pair receipts
        res = self.judge(r, paths)
        self.assertEqual((res["verdict"], res["errors"]), ("PASS", []))
        self.assertEqual(res["fabric_jobs"], jobs)
        self.assertEqual(res["all9"]["actual_holds"], 6)
        early = self.judge(r, paths[:2])
        self.assertEqual((early["verdict"], len(early["fabric_jobs"])), ("UNKNOWN_INCOMPLETE", 4))
        self.assertIsNone(early["all9"])

    def test_early_logical_fail_after_two_pairs_uses_four_holds_and_never_fabricates_a_median(self):
        r = self.rig()
        docs, paths = self.run_pairs(r, hold_plan(r.roles, [[1000.0] * 3] * 3, [[900.0] * 3] * 3), pairs=(1, 2))
        res = self.judge(r, paths)
        self.assertEqual(res["verdict"], "FAIL_EARLY_ORDER_STATISTIC")
        self.assertEqual(len(res["fabric_jobs"]), 4)
        self.assertIsNone(res["all9"])
        self.assertTrue(res["early_bounds"]["cannot_reach_2000"])
        self.assertEqual(res["candidate"]["sample_count"], 6)

    def test_no_resumed_gpu_after_failure(self):
        r = self.rig()
        plans = [ok_plan([2000.0] * 3), {"mode": "exit3"}, ok_plan([2000.0] * 3), ok_plan([2000.0] * 3)]
        res = self.run_pair_cells(r, plans, 1)
        self.assertEqual(self.calls(r), ["A", "B"])
        out = os.path.join(r.root, "out", "p2c1")
        with self.assertRaises(h2.Refusal) as cm:
            h2.run_cell(r.manifest_path, r.msha, 2, 1, out, [res["path"]], None, None, r.host(plans))
        self.assertIn("PAIR_INCOMPLETE", " ".join(cm.exception.errors))
        self.assertFalse(os.path.exists(out))
        self.assertEqual(self.calls(r), ["A", "B"])
        r2, rec, o1 = self.one_cell([{"mode": "exit3"}, ok_plan([2000.0] * 3)])
        self.assert_second_cell_refused(r2, rec, o1)
        self.assertEqual(self.calls(r2), ["A"])                                      # nothing ran after the failed cell 1

    def test_bad_budgets_fail_before_any_spawn_and_planned_budget_runs(self):
        planned = {"cell_cap_seconds": 1200, "kill_grace_seconds": 15, "drain_seconds": 90, "finalize_seconds": 45, "preflight_seconds": 30,
                   "maximum_hold_seconds": 1440}
        no_preflight = {k: v for k, v in planned.items() if k != "preflight_seconds"}
        for budget, needle in ((dict(planned, maximum_hold_seconds=1801), "clamp"),
                               (dict(planned, cell_cap_seconds=1800, maximum_hold_seconds=2000), "clamp"),
                               (dict(planned, maximum_hold_seconds=1379), "cannot hold even one full single-process hold"),
                               (no_preflight, "preflight_seconds")):
            r = self.rig(budget=budget)
            out = os.path.join(r.root, "out", "p1c1")
            os.makedirs(os.path.dirname(out))
            with self.assertRaises(h2.Refusal) as cm:
                h2.run_cell(r.manifest_path, r.msha, 1, 1, out, [], None, None, r.host([ok_plan([2000.0] * 3)]))
            self.assertIn(needle, " ".join(cm.exception.errors))
            self.assertFalse(os.path.exists(out))
            self.assertEqual(self.calls(r), [])
        r = self.rig(budget=planned)                                              # 1200+15+90+45+30 = 1380 <= 1440 <= 1800
        _, rec, _ = self.one_cell([ok_plan([2000.0] * 3)] * 2, rig=r)
        self.assertEqual(rec["status"], "OK")
        self.assertEqual((rec["hold"]["required_single_hold_seconds"], rec["hold"]["maximum_hold_seconds"], rec["hold"]["clamp_seconds"]),
                         (1380, 1440, 1800))

    def test_watcher_failure_closes_the_hold_cleanly_and_blocks_cell_two(self):
        r = self.rig()
        line = "amdgpu 0000:03:00.0: VM_L2_PROTECTION_FAULT address 0x1\n"
        _, c, out = self.one_cell([{"mode": "sleep", "write": [[r.watch, line, "a"]]}, ok_plan([2000.0] * 3)], rig=r)
        self.assertEqual(c["status"], "WATCHER_FATAL")
        self.assertEqual(c["lifecycle"]["group_after_cleanup"], [])
        self.assertFalse(alive(c["process"]["pid"]))
        for child in r.children():
            self.assertFalse(alive(child))
        self.assertTrue(h2.seal_ok(c))
        self.assert_second_cell_refused(r, c, out)


# ======================================================================================================================================
class BenchValidationTests(Base):
    def setUp(self):
        super().setUp()
        self.r = self.rig()
        self.m = self.r.m

    def doc(self, **patch):
        s = [2000.0, 2100.0, 2200.0]
        d = {
            "protocol": h2.PROTOCOL, "model": self.m["target"]["model_path"], "loaded": {"max_seq": 16416}, "gpu": {},
            "max_seq": {"effective": 16416, "requested": 16416, "source": "flag", "run_need": 16416, "model_limit": None},
            "redline_pm4": False, "kv_mode": None, "spec_requested": None, "load_speculation": {}, "prefill_route": h2.PREFILL_ROUTE,
            "runs": 3, "warnings": [],
            "prefill": [{"tokens": 16384, "stats": {"median": 2100.0, "mean": sum(s) / 3, "min": 2000.0, "max": 2200.0, "stdev": 81.6}, "samples": s,
                         "power": []}],
            "decode": [{"context": 1, "tokens": 1, "stats": {}, "samples": [50.0, 51.0, 52.0], "power": []}], "sustained": [], "topology": None}
        d.update(patch)
        return d

    def val(self, d):
        raw = d if isinstance(d, (bytes, str)) else json.dumps(d)
        return h2.validate_bench(raw.encode() if isinstance(raw, str) else raw, self.m)

    def assertBad(self, d, needle):
        res = self.val(d)
        self.assertFalse(res["ok"], needle)
        self.assertIsNone(res["samples"])                     # nothing zero-filled
        self.assertTrue(any(needle in e for e in res["errors"]), (needle, res["errors"]))

    def test_valid_document_and_durations(self):
        res = self.val(self.doc())
        self.assertTrue(res["ok"], res["errors"])
        self.assertEqual(res["samples"], [2000.0, 2100.0, 2200.0])
        self.assertEqual(res["reported_median"], 2100.0)
        self.assertAlmostEqual(res["durations_s"][0], 16384 / 2000.0)
        self.assertEqual(res["decode"]["samples"], [50.0, 51.0, 52.0])

    def test_max_seq_forms(self):
        self.assertTrue(self.val(self.doc(max_seq=16416, loaded={}))["ok"])                         # plain int form
        self.assertTrue(self.val(self.doc(max_seq={"effective": None}, loaded={"max_seq": 16416}))["ok"])
        self.assertBad(self.doc(loaded={"max_seq": 16640}), "loaded.max_seq")
        self.assertBad(self.doc(max_seq={"effective": 16640, "requested": 16416, "source": "flag"}), "max_seq.effective")
        self.assertBad(self.doc(max_seq={"effective": 16416, "requested": 16416, "source": "run_shape"}), "source")
        self.assertBad(self.doc(max_seq=None, loaded={}), "no max_seq evidence")

    def test_sample_and_stats_strictness(self):
        base = self.doc()
        for bad in ([2000.0, 0.0, 2200.0], [2000.0, -1.0, 2200.0], [2000.0, 2100.0], [2000.0] * 4, [2000.0, "x", 2200.0], [2000.0, None, 2200.0],
                    [2000.0, True, 2200.0]):
            d = copy.deepcopy(base)
            d["prefill"][0]["samples"] = bad
            self.assertBad(d, "samples must be")
        d = copy.deepcopy(base)
        d["prefill"][0]["stats"]["median"] = 2150.0
        self.assertBad(d, "reported median")
        d = copy.deepcopy(base)
        d["prefill"][0]["stats"]["max"] = 2300.0
        self.assertBad(d, "min/max")
        d = copy.deepcopy(base)
        d["prefill"][0]["stats"].pop("stdev")
        self.assertBad(d, "stats must carry")
        d = copy.deepcopy(base)
        d["prefill"][0]["tokens"] = 8192
        self.assertBad(d, "prefill tokens")
        d = copy.deepcopy(base)
        d["prefill"].append(copy.deepcopy(d["prefill"][0]))
        self.assertBad(d, "one-element")
        self.assertBad(self.doc(prefill=[]), "one-element")

    def test_nan_infinity_duplicates_and_non_pure_stdout(self):
        good = json.dumps(self.doc())
        self.assertBad(good.replace("2100.0, 2200.0], \"power\"", "NaN, 2200.0], \"power\""), "strict JSON")
        self.assertBad(good.replace("2100.0, 2200.0], \"power\"", "Infinity, 2200.0], \"power\""), "strict JSON")
        self.assertBad(good.replace("2100.0, 2200.0], \"power\"", "1e999, 2200.0], \"power\""), "strict JSON")
        self.assertBad(good[:-1] + ', "runs": 3}', "strict JSON")              # duplicate key
        self.assertBad("progress line\n" + good, "strict JSON")
        self.assertBad(good + "\ntrailer", "strict JSON")
        self.assertBad("", "strict JSON")
        self.assertBad("[1, 2]", "not an object")
        self.assertTrue(self.val("\n  " + good + "\n")["ok"])

    def test_protocol_binding_and_no_claims(self):
        self.assertBad(self.doc(protocol="other"), "protocol")
        self.assertBad(self.doc(model="/other/model"), "model")
        self.assertBad(self.doc(prefill_route="spec"), "prefill_route")
        self.assertBad(self.doc(runs=2), "runs")
        self.assertBad(self.doc(redline_pm4=True), "redline")
        self.assertBad(self.doc(spec_requested="mtp"), "speculation")
        self.assertBad(self.doc(sustained=[{"x": 1}]), "sustained")
        self.assertBad(self.doc(prompt_cache_hit=True), "prompt/prefix cache")
        self.assertBad(self.doc(loaded={"max_seq": 16416, "prefix_cache": {"tokens": 12}}), "prompt/prefix cache")
        self.assertBad(self.doc(loaded=None), "loaded is not an object")

    def test_auxiliary_decode_row(self):
        d = self.doc()
        d["decode"][0]["context"] = 128
        self.assertBad(d, "decode row")
        d = self.doc()
        d["decode"][0]["samples"] = [1.0, 2.0]
        self.assertBad(d, "decode samples")
        d = self.doc()
        d["decode"] = []
        self.assertBad(d, "decode must be")
        d = self.doc()
        d["decode"][0]["samples"] = [0.0, 0.0, 0.0]               # auxiliary: recorded, not a prefill failure
        res = self.val(d)
        self.assertTrue(res["ok"])
        self.assertFalse(res["decode"]["all_positive"])


# ======================================================================================================================================
class JudgeTests(Base):
    def hold(self, cand, inc, pairs=(1, 2, 3), **kw):
        r = self.rig(**kw)
        docs, paths = self.run_pairs(r, hold_plan(r.roles, cand, inc), pairs=pairs)
        for d in docs:
            self.assertEqual(d["status"], "PAIR_COMPLETE", d["stop_reason"])
        return r, docs, paths

    def test_flat_nine_median_is_not_median_of_process_medians(self):
        cand = [[1.0, 1.0, 5000.0], [1.0, 1.0, 5000.0], [3000.0, 3000.0, 3000.0]]        # process medians 1, 1, 3000 -> median-of-medians = 1
        inc = [[2000.0] * 3] * 3
        r, _, paths = self.hold(cand, inc)
        res = self.judge(r, paths)
        flat = sorted(s for k in cand for s in k)
        self.assertEqual(res["candidate"]["median_all_samples"], 3000.0)
        self.assertEqual(flat[4], 3000.0)
        self.assertEqual(h2.median([p["reported_median"] for p in res["candidate"]["processes"]]), 1.0)
        self.assertNotEqual(res["candidate"]["median_all_samples"], 1.0)
        self.assertEqual(res["verdict"], "PASS")
        self.assertEqual(len(res["candidate"]["samples_flat"]), 9)
        self.assertEqual(len(res["candidate"]["durations_s_flat"]), 9)
        self.assertEqual([p["cell_id"] for p in res["candidate"]["processes"]], ["B1", "B2", "B3"])
        self.assertEqual([p["cell_id"] for p in res["incumbent"]["processes"]], ["A1", "A2", "A3"])

    def test_gate_boundaries_are_exact(self):
        # exactly 1.10 x incumbent and exactly 2000: PASS (>=)
        r, _, paths = self.hold([[2200.0] * 3] * 3, [[2000.0] * 3] * 3)
        res = self.judge(r, paths)
        self.assertEqual(res["verdict"], "PASS", res)
        self.assertEqual(res["all9"]["ratio_exact"], "11/10")
        # one ulp-ish under the ratio gate
        r, _, paths = self.hold([[2200.0] * 3] * 3, [[2000.5] * 3] * 3)
        res = self.judge(r, paths)
        self.assertEqual((res["verdict"], res["failed_gates"]), ("FAIL", ["gate_1p10_over_incumbent"]))
        # target miss with 1400 floor and ratio satisfied
        r, _, paths = self.hold([[1999.0] * 3] * 3, [[1000.0] * 3] * 3)
        res = self.judge(r, paths)
        self.assertEqual((res["verdict"], res["failed_gates"]), ("FAIL", ["gate_2000_tok_s"]))
        self.assertTrue(res["all9"]["informational_1400_tok_s"])
        self.assertTrue(res["all9"]["gate_1p10_over_incumbent"])
        r, _, paths = self.hold([[1300.0] * 3] * 3, [[1000.0] * 3] * 3)
        res = self.judge(r, paths)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertFalse(res["all9"]["informational_1400_tok_s"])

    def test_partial_holds_are_never_pass(self):
        great = [[5000.0] * 3] * 3
        poor = [[1000.0] * 3] * 3
        r, _, paths = self.hold(great, poor)
        res = self.judge(r, paths[:1])
        self.assertEqual(res["verdict"], "UNKNOWN_INCOMPLETE")
        self.assertIsNone(res["all9"])
        self.assertNotIn("early_bounds", res)                        # one pair: 3 known samples, never an early call
        res = self.judge(r, paths[:2])
        self.assertEqual(res["verdict"], "UNKNOWN_INCOMPLETE")       # 6 known, thresholds still reachable
        self.assertIn("early_bounds", res)
        self.assertFalse(res["early_bounds"]["fails_early"])
        self.assertIsNone(res["all9"])

    def test_early_fail_after_two_pairs_target_and_ratio(self):
        r, _, paths = self.hold([[1000.0] * 3] * 3, [[900.0] * 3] * 3)
        res = self.judge(r, paths[:2])
        self.assertEqual(res["verdict"], "FAIL_EARLY_ORDER_STATISTIC")
        self.assertTrue(res["early_bounds"]["cannot_reach_2000"])
        self.assertIsNone(res["all9"])
        self.assertIn("5th smallest", res["early_bounds"]["proof"])
        # the full three pairs of the same data are a plain FAIL with a real all-nine median
        self.assertEqual(self.judge(r, paths)["verdict"], "FAIL")
        # only the ratio gate is unreachable
        r, _, paths = self.hold([[2100.0] * 3] * 3, [[2000.0] * 3] * 3)
        res = self.judge(r, paths[:2])
        self.assertEqual(res["verdict"], "FAIL_EARLY_ORDER_STATISTIC")
        self.assertFalse(res["early_bounds"]["cannot_reach_2000"])
        self.assertTrue(res["early_bounds"]["cannot_reach_1p10_x_incumbent"])
        # a candidate that is slow in only some samples is NOT an early fail: k5 decides
        cand = [[1000.0, 2500.0, 2500.0], [2500.0, 2500.0, 2500.0], [2500.0] * 3]
        r, _, paths = self.hold(cand, [[1000.0] * 3] * 3)
        res = self.judge(r, paths[:2])
        self.assertEqual(res["verdict"], "UNKNOWN_INCOMPLETE")

    def test_early_bound_is_exact_against_adversarial_remaining_values(self):
        rng = random.Random(20261019)
        extremes = [1e-9, 1500.0, 2500.0, 1e12]
        for trial in range(6):
            cand = [rng.choice([900.0, 1500.0, 1999.0, 2000.0, 2100.0, 2600.0, 3000.0]) for _ in range(6)]
            inc = [rng.choice([800.0, 1200.0, 1800.0, 2000.0, 2300.0]) for _ in range(6)]
            b = h2.early_bounds(cand, inc)
            can_pass_some = False
            all_fail = True
            for cu in itertools.product(extremes, repeat=3):
                c9 = sorted(cand + list(cu))
                self.assertLessEqual(b["candidate_final_median_min"], c9[4])
                self.assertGreaterEqual(b["candidate_final_median_max"], c9[4])
                for iu in itertools.product(extremes, repeat=3):
                    i9 = sorted(inc + list(iu))
                    self.assertLessEqual(b["incumbent_final_median_min"], i9[4])
                    self.assertGreaterEqual(b["incumbent_final_median_max"], i9[4])
                    passes = c9[4] >= 2000 and c9[4] * 10 >= i9[4] * 11
                    can_pass_some |= passes
                    all_fail &= not passes
            self.assertEqual(b["fails_early"], all_fail, (cand, inc, b))       # fails early IFF no completion of the nine can pass
            self.assertEqual(b["fails_early"], not can_pass_some)
            self.assertEqual(b["candidate_final_median_max"], sorted(cand)[4])
            self.assertEqual(b["incumbent_final_median_min"], sorted(inc)[1])
        with self.assertRaises(h2.H2Error):
            h2.early_bounds([1.0] * 5, [1.0] * 6)

    # ---------------------------------------------------------------- binding / tamper
    def copy_and_edit(self, r, paths, edit, rebind=True):
        """Copies the WHOLE out tree (cell OUT dirs + pair dirs) to a fresh root, applies edit(pair_docs, out_root) to the embedded cells / raw files,
        then rewrites every cell file and pair receipt CONSISTENTLY (reseal, file hashes, cell-2 first-cell binding, chain) so that only the
        validators themselves can catch the tamper; `rebind=False` keeps the edited bindings. Returns the new pair.json paths."""
        out_root = os.path.dirname(os.path.dirname(paths[0]))
        root = tempfile.mkdtemp(prefix="tampered-", dir=r.root)       # unique and owned by this call (several tamper cases per test)
        dst = os.path.join(root, "out")
        shutil.copytree(out_root, dst, symlinks=True)
        new = [os.path.join(dst, os.path.relpath(p, out_root)) for p in paths]
        docs = [read_json(p) for p in new]
        edit(docs, dst)
        chain = []
        for d, p in zip(docs, new):
            files = d["cell_files"]
            for j, c in enumerate(d["cells"]):
                if rebind:
                    c["previous_pairs"] = [{"pair": i, "receipt_sha256": s} for i, s in enumerate(chain, 1)]
                    if j == 1:
                        first = d["cells"][0]
                        c["first_cell"].update({"file_sha256": files[0]["file_sha256"], "receipt_sha256": first["receipt_sha256"],
                                                "cell_id": first["cell_id"], "fabric_job": first["env"].get("fabric_job") or "none"})
                sealed = h2.seal(c)
                d["cells"][j] = sealed
                cp = os.path.normpath(os.path.join(os.path.dirname(p), files[j]["relpath"]))
                if not os.path.exists(cp) or read_json(cp) != sealed:       # untouched cells keep their ORIGINAL bytes (and hashes)
                    write_text(cp, json.dumps(sealed))
                files[j]["file_sha256"] = self.sha_file(cp)
                files[j]["receipt_sha256"] = sealed["receipt_sha256"]
            if rebind:
                d["previous_pairs"] = [{"pair": i, "receipt_sha256": s} for i, s in enumerate(chain, 1)]
            sealed_pair = h2.seal(d)
            if read_json(p) != sealed_pair:
                write_text(p, json.dumps(sealed_pair))
            chain.append(sealed_pair["receipt_sha256"])
        return new

    def test_tampered_receipts_are_invalid_not_trusted(self):
        r, _, paths = self.hold([[2400.0] * 3] * 3, [[1000.0] * 3] * 3)
        # the untouched copy is valid (the helper itself introduces no error)
        self.assertEqual(self.judge(r, self.copy_and_edit(r, paths, lambda docs, root: None))["verdict"], "PASS")
        # a pair receipt edited without resealing
        new = self.copy_and_edit(r, paths, lambda docs, root: None)
        d = read_json(new[0])
        d["cells"][0]["argv"][-1] = "--nope"
        write_text(new[0], json.dumps(d))
        res = self.judge(r, new)
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertTrue(any("receipt_sha256" in e for e in res["errors"]), res["errors"])

        def bad_argv(docs, root):
            docs[1]["cells"][0]["argv"] = docs[1]["cells"][0]["argv"][:-1] + ["--tg", "0", "--json"]
        res = self.judge(r, self.copy_and_edit(r, paths, bad_argv))
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertTrue(any("argv" in e for e in res["errors"]))

        def bad_config(docs, root):
            docs[0]["cells"][1]["home"]["sha256"] = "0" * 64
        res = self.judge(r, self.copy_and_edit(r, paths, bad_config))
        self.assertTrue(any("config.toml hash" in e for e in res["errors"]))

        def bad_source(docs, root):
            docs[2]["cells"][0]["source"]["tree"] = "a" * 40
        res = self.judge(r, self.copy_and_edit(r, paths, bad_source))
        self.assertTrue(any("source" in e for e in res["errors"]))

        def bad_env(docs, root):
            docs[0]["cells"][0]["env"]["set"]["HIPFIRE_QWEN4_D2_DEFAULT"] = "1"
        res = self.judge(r, self.copy_and_edit(r, paths, bad_env))
        self.assertTrue(any("D2_DEFAULT" in e for e in res["errors"]))

        def extra_env(docs, root):
            docs[0]["cells"][0]["env"]["set"]["HIPFIRE_QWEN4_SLOWLINK_PREFILL"] = "1"
        res = self.judge(r, self.copy_and_edit(r, paths, extra_env))
        self.assertTrue(any("applied env keys" in e for e in res["errors"]))

        def bad_cli(docs, root):
            docs[0]["cells"][1]["cli"]["sha256"] = "0" * 64
        res = self.judge(r, self.copy_and_edit(r, paths, bad_cli))
        self.assertTrue(any("cli binding" in e for e in res["errors"]))

        def bad_cached(docs, root):
            docs[1]["cells"][1]["cached_inputs_after"][r.model]["mtime_ns"] += 1
        res = self.judge(r, self.copy_and_edit(r, paths, bad_cached))
        self.assertTrue(any("cached input stat" in e for e in res["errors"]))

        def bad_pack(docs, root):
            k = next(iter(docs[0]["cells"][0]["pack_files"]))
            docs[0]["cells"][0]["pack_files"][k] = "0" * 64
        res = self.judge(r, self.copy_and_edit(r, paths, bad_pack))
        self.assertTrue(any("pack file hashes" in e for e in res["errors"]))

    def test_tampered_holds_jobs_hot_cache_packs_and_chain_are_invalid(self):
        r, _, paths = self.hold([[2400.0] * 3] * 3, [[1000.0] * 3] * 3)

        def verdict_has(edit, needle, rebind=True):
            res = self.judge(r, self.copy_and_edit(r, paths, edit, rebind=rebind))
            self.assertEqual(res["verdict"], "UNKNOWN_INVALID", needle)
            self.assertTrue(any(needle in e for e in res["errors"]), (needle, res["errors"]))

        verdict_has(lambda docs, root: docs[0]["cells"][0]["hold"].__setitem__("maximum_hold_seconds", 5000), "hold budget differs from the manifest")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["hold"].__setitem__("clamp_seconds", 3600), "hold budget differs from the manifest")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["hold"].__setitem__("hold_end_monotonic_ns",
                                                                                docs[0]["cells"][0]["hold"]["hold_end_monotonic_ns"] + 2000 * 10 ** 9),
                    "more than maximum_hold_seconds")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["hold"].__setitem__("remaining_at_start_seconds", 1), "full single-cell budget did not fit")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["hold"].__setitem__("boot_id", "another-boot"), "boot_id differs")
        verdict_has(lambda docs, root: docs[0]["cells"][0].pop("hold"), "hold block missing")
        # distinct actual holds: a reused / missing FABRIC_JOB invalidates the receipts
        verdict_has(lambda docs, root: docs[1]["cells"][0]["env"].__setitem__("fabric_job", docs[0]["cells"][0]["env"]["fabric_job"]), "already used by another cell")
        verdict_has(lambda docs, root: docs[0]["cells"][1]["env"].__setitem__("fabric_job", docs[0]["cells"][0]["env"]["fabric_job"]), "already used by another cell")
        verdict_has(lambda docs, root: docs[2]["cells"][1]["env"].pop("fabric_job"), "FABRIC_JOB of this hold is not recorded")
        # hot cache must be the private writable per-cell one; packs must have been verified read-only and unchanged

        def shared_cache(docs, root):
            k = docs[0]["cells"][0]["hot_cache"]["path"]
            docs[1]["cells"][0]["hot_cache"]["path"] = k
            docs[1]["cells"][0]["env"]["set"]["HIPFIRE_KERNEL_CACHE"] = k
        verdict_has(shared_cache, "is shared with another cell")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["hot_cache"].__setitem__("writable", False), "not the private writable HIPFIRE_KERNEL_CACHE")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["hot_cache"].__setitem__("path", "/tmp/shared-global-cache"),
                    "not the private writable HIPFIRE_KERNEL_CACHE")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["pack_readonly"].__setitem__("verified", False), "not verified read-only")
        verdict_has(lambda docs, root: docs[0]["cells"][0]["integrity"].__setitem__("pack_dirs_unchanged", False), "integrity.pack_dirs_unchanged is not true")
        # chain / first-cell bindings (rebind=False keeps the tampered bindings)
        verdict_has(lambda docs, root: docs[2]["cells"][0].__setitem__("previous_pairs", []), "previous_pairs", rebind=False)
        verdict_has(lambda docs, root: docs[2].__setitem__("previous_pairs", docs[2]["previous_pairs"][:1]), "previous_pairs", rebind=False)
        verdict_has(lambda docs, root: docs[1]["cells"][1]["first_cell"].__setitem__("file_sha256", "b" * 64), "does not bind this cell 1", rebind=False)
        verdict_has(lambda docs, root: docs[1]["cells"][1].__setitem__("first_cell", None), "complete first_cell binding", rebind=False)
        verdict_has(lambda docs, root: docs[0]["cells"][0].__setitem__("first_cell", dict(docs[0]["cells"][1]["first_cell"])),
                    "must not carry a first_cell binding", rebind=False)
        # cell files: swapped, mixed between pairs, edited or missing after assembly

        def swap_files(docs, root):
            docs[0]["cell_files"].reverse()
            docs[0]["cells"].reverse()
        verdict_has(swap_files, "swapped or mixed", rebind=False)

        def mixed_pair_cell(docs, root):
            docs[1]["cell_files"][0] = dict(docs[0]["cell_files"][0], relpath=docs[0]["cell_files"][0]["relpath"])
            docs[1]["cells"][0] = docs[0]["cells"][0]
        verdict_has(mixed_pair_cell, "swapped or mixed", rebind=False)
        new = self.copy_and_edit(r, paths, lambda docs, root: None)
        cell_file = os.path.normpath(os.path.join(os.path.dirname(new[0]), read_json(new[0])["cell_files"][0]["relpath"]))
        with open(cell_file, "a") as f:
            f.write("\n")
        res = self.judge(r, new)
        self.assertTrue(any("sha256 differs from the pair's binding" in e for e in res["errors"]), res["errors"])
        new = self.copy_and_edit(r, paths, lambda docs, root: None)
        d = read_json(new[0])
        d["cells"][1]["argv"][-1] = "--nope"                                  # embedded copy edited, pair resealed, cell FILE untouched
        write_text(new[0], json.dumps(h2.seal(d)))
        res = self.judge(r, new)
        self.assertTrue(any("embedded cell 2 differs from its cell file" in e for e in res["errors"]), res["errors"])
        new = self.copy_and_edit(r, paths, lambda docs, root: None)
        os.unlink(os.path.normpath(os.path.join(os.path.dirname(new[1]), read_json(new[1])["cell_files"][1]["relpath"])))
        res = self.judge(r, new)
        self.assertTrue(any("cell file" in e and "unreadable" in e for e in res["errors"]), res["errors"])

    def test_raw_stdout_is_revalidated_and_must_match(self):
        r, _, paths = self.hold([[2400.0] * 3] * 3, [[1000.0] * 3] * 3)

        def stdout_changed(docs, root):
            p = os.path.join(root, "p1c1", "A1", "stdout.json")
            with open(p, "a") as f:
                f.write("\n")
        res = self.judge(r, self.copy_and_edit(r, paths, stdout_changed))
        self.assertTrue(any("stdout sha256" in e for e in res["errors"]), res["errors"])

        def stdout_forged(docs, root):
            p = os.path.join(root, "p1c1", "A1", "stdout.json")
            doc = read_json(p)
            doc["prefill"][0]["samples"] = [9999.0, 9999.0, 9999.0]       # forged rates; median now inconsistent
            raw = json.dumps(doc).encode()
            write_bytes(p, raw)
            docs[0]["cells"][0]["stdout"]["sha256"] = sha(raw)
        res = self.judge(r, self.copy_and_edit(r, paths, stdout_forged))
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertTrue(any("fails validation" in e for e in res["errors"]), res["errors"])

        def embedded_forged(docs, root):
            docs[0]["cells"][0]["bench"]["samples"] = [1.0, 2.0, 3.0]
        res = self.judge(r, self.copy_and_edit(r, paths, embedded_forged))
        self.assertTrue(any("embedded samples differ" in e for e in res["errors"]), res["errors"])

        def missing_stdout(docs, root):
            os.unlink(os.path.join(root, "p2c2", "A2", "stdout.json"))
        res = self.judge(r, self.copy_and_edit(r, paths, missing_stdout))
        self.assertTrue(any("unreadable" in e for e in res["errors"]), res["errors"])

    def test_pair_order_uniqueness_and_manifest_binding(self):
        r, _, paths = self.hold([[2400.0] * 3] * 3, [[1000.0] * 3] * 3)
        for bad in ([paths[1], paths[0]], [paths[0], paths[0]], [paths[0], paths[2]], [paths[1]]):
            res = self.judge(r, bad)
            self.assertEqual(res["verdict"], "UNKNOWN_INVALID", bad)
            self.assertTrue(any("unique and ordered" in e for e in res["errors"]), res["errors"])
        res = self.judge(r, paths + paths[:1])
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        # other manifest
        m = copy.deepcopy(r.m)
        m["label"] = "other"
        r.save(m)
        res = self.judge(r, paths)
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertTrue(any("manifest_sha256" in e for e in res["errors"]))

    def test_failed_pair_blocks_any_later_pair_and_is_incomplete(self):
        r = self.rig()
        plans = [ok_plan([2400.0] * 3), ok_plan([1000.0] * 3), ok_plan([2400.0] * 3), {"mode": "exit3"}] + [ok_plan([2400.0] * 3)] * 2
        docs, paths = self.run_pairs(r, plans, pairs=(1,))
        res2 = self.run_pair_cells(r, plans, 2, paths)                      # cell 1 (B2) is fine, cell 2 (A2) fails: the pair closes incomplete
        self.assertEqual(res2["doc"]["status"], "PAIR_INCOMPLETE")
        res = self.judge(r, [paths[0], res2["path"]])
        self.assertEqual(res["verdict"], "UNKNOWN_INCOMPLETE")
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["pairs_complete"], [1])
        self.assertEqual(res["pairs_incomplete"][0]["pair"], 2)
        calls = self.calls(r)
        with self.assertRaises(h2.Refusal) as cm:                         # and no further hold may run
            h2.run_cell(r.manifest_path, r.msha, 3, 1, os.path.join(r.root, "out", "p3c1"), [paths[0], res2["path"]], None, None, r.host(plans))
        self.assertIn("PAIR_INCOMPLETE", " ".join(cm.exception.errors))
        self.assertEqual(self.calls(r), calls)

    def test_duplicate_process_identity_is_rejected(self):
        r, _, paths = self.hold([[2400.0] * 3] * 3, [[1000.0] * 3] * 3)

        def dup(docs, root):
            docs[1]["cells"][0]["process"]["pid"] = docs[0]["cells"][0]["process"]["pid"]
            docs[1]["cells"][0]["process"]["starttime_ticks"] = docs[0]["cells"][0]["process"]["starttime_ticks"]
        res = self.judge(r, self.copy_and_edit(r, paths, dup))
        self.assertTrue(any("reused" in e for e in res["errors"]), res["errors"])

        def same_home(docs, root):
            docs[1]["cells"][1]["home"]["path"] = docs[0]["cells"][0]["home"]["path"]
        res = self.judge(r, self.copy_and_edit(r, paths, same_home))
        self.assertTrue(any("shared with another cell" in e for e in res["errors"]), res["errors"])

        def time_travel(docs, root):
            docs[1]["cells"][0]["process"]["start_monotonic_ns"] = 1
        res = self.judge(r, self.copy_and_edit(r, paths, time_travel))
        self.assertTrue(any("strictly after" in e for e in res["errors"]), res["errors"])

        def shared_out_dir(docs, root):                                       # the two cells of pair 1 report from ONE OUT dir
            old = os.path.join(root, "p1c2", "B1")
            new_dir = os.path.join(root, "p1c1", "B1")
            shutil.copytree(old, new_dir)
            docs[0]["cell_files"][1]["relpath"] = os.path.relpath(os.path.join(new_dir, "cell.json"), os.path.join(root, "pair1"))
        res = self.judge(r, self.copy_and_edit(r, paths, shared_out_dir))
        self.assertTrue(any("shares its OUT dir" in e for e in res["errors"]), res["errors"])

    def test_judge_rejects_swapped_arm_roles(self):
        r, _, paths = self.hold([[2400.0] * 3] * 3, [[1000.0] * 3] * 3)

        def swap(docs, root):
            for d in docs:
                for c in d["cells"]:
                    c["role"] = "candidate" if c["role"] == "incumbent" else "incumbent"
        res = self.judge(r, self.copy_and_edit(r, paths, swap))
        self.assertEqual(res["verdict"], "UNKNOWN_INVALID")
        self.assertTrue(any("role" in e for e in res["errors"]))

    def test_judge_exit_codes(self):
        r, _, paths = self.hold([[1000.0] * 3] * 3, [[900.0] * 3] * 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(h2.main(["judge", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--pairs", *paths[:2], "--out",
                                      os.path.join(r.root, "j1.json")], r.host([])), 1)
            self.assertEqual(h2.main(["judge", "--manifest", r.manifest_path, "--expected-manifest-sha256", r.msha, "--pairs", *paths[:1], "--out",
                                      os.path.join(r.root, "j2.json")], r.host([])), 3)


# ======================================================================================================================================
class UnitTests(Base):
    def test_build_env_strips_and_applies_only_routing_metadata(self):
        r = self.rig()
        parent = {"PATH": "/bin", "LANG": "C", "FABRIC_JOB": "j", "HIPFIRE_FABRIC_ADMISSION": "x", "HIPFIRE_ANYTHING": "1", "HIP_FOO": "1",
                  "ROCR_VISIBLE_DEVICES": "7", "ROC_X": "1", "HSA_X": "1", "GPU_X": "1", "AMD_X": "1", "HIPFIRE_DAEMON_BIN": "/d", "HOME": "/h"}
        env, unset, applied = h2.build_env(parent, "/cell", r.m["target"])
        self.assertEqual(unset, sorted(["HIPFIRE_ANYTHING", "HIP_FOO", "ROCR_VISIBLE_DEVICES", "ROC_X", "HSA_X", "GPU_X", "AMD_X", "HIPFIRE_DAEMON_BIN"]))
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["FABRIC_JOB"], "j")
        self.assertEqual(env["HIPFIRE_FABRIC_ADMISSION"], "x")
        self.assertEqual(env["HOME"], "/cell/home")
        self.assertEqual(env["HIPFIRE_HOME"], "/cell/home/.hipfire")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], UUID)
        self.assertEqual(env["HIPFIRE_QWEN4_D2_DEFAULT"], "0")
        self.assertNotIn("HIPFIRE_DAEMON_BIN", env)
        self.assertEqual(sorted(applied), sorted(["HIPFIRE_DEVICES", "HIPFIRE_HOME", "HIPFIRE_KERNEL_CACHE", "HIPFIRE_LOCK_DIR",
                                                  "HIPFIRE_NO_REGISTRY_FETCH", "HIPFIRE_QWEN4_D2_DEFAULT", "HOME", "HIP_VISIBLE_DEVICES",
                                                  "ROCR_VISIBLE_DEVICES"]))

    def test_write_atomic_never_overwrites(self):
        p = os.path.join(self.tmp, "x.json")
        h2.write_atomic(p, b"one")
        with self.assertRaises(FileExistsError):
            h2.write_atomic(p, b"two")
        self.assertEqual(read_bytes(p), b"one")
        self.assertEqual([n for n in os.listdir(self.tmp) if n.startswith(".x.json")], [])

    def test_seal_roundtrip_detects_any_edit(self):
        d = h2.seal({"a": 1, "b": [1.5, 2]})
        self.assertTrue(h2.seal_ok(d))
        d["b"][0] = 1.6
        self.assertFalse(h2.seal_ok(d))

    def test_median_and_expected_argv(self):
        self.assertEqual(h2.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(h2.median([1.0, 3.0]), 2.0)
        r = self.rig()
        self.assertEqual(h2.expected_argv(r.m, "B")[1:], ["bench", r.model, "--matrix", "--pp", "16384", "--tg", "1", "--ctx", "1", "--runs", "3",
                                                         "--warmups", "1", "--max-seq", "16416", "--json"])
        self.assertEqual(h2.GLOBAL_ORDER, tuple(f"{a}{p}" for p in (1, 2, 3) for a in h2.ORDER[p]))

    def test_wait_idle_threshold_and_group_members(self):
        r = self.rig()
        m = r.m
        host = r.host([], cls=ClockHost)
        self.assertTrue(h2.wait_idle(host, m, 0)["idle"])
        r.write_sys("mem_info_vram_used", 2 * GIB)                 # the threshold is strictly below 2 GiB
        res = h2.wait_idle(host, m, 3)
        self.assertFalse(res["idle"])
        self.assertGreaterEqual(len(res["samples"]), 3)
        r.write_sys("mem_info_vram_used", "garbage")
        res = h2.wait_idle(host, m, 0)
        self.assertFalse(res["idle"])
        self.assertIn("error", res["samples"][0])                  # unreadable is recorded, never treated as idle
        self.assertEqual(h2.live_members(2 ** 22 + 12345), [])

    def test_watcher_scan_identity_shrink_and_missing_log_fail_closed(self):
        r = self.rig()
        ident = h2.watcher_identity(r.m)
        self.assertEqual((ident["inode"], ident["device"]), (os.stat(r.watch).st_ino, os.stat(r.watch).st_dev))
        off, w = h2.watcher_scan(r.m, ident["start_size_bytes"], ident)
        self.assertEqual((w["error"], w["fatal"]), (None, False))
        with open(r.watch, "a") as f:
            f.write("amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout\n")
        off, w = h2.watcher_scan(r.m, off, ident)
        self.assertTrue(w["fatal"] and w["match_count"] == 1)
        other = os.path.join(r.root, "other.log")
        write_text(other, "x" * 500)
        os.unlink(r.watch)
        os.symlink(other, r.watch)                                   # stale / re-targeted log
        _, w = h2.watcher_scan(r.m, off, ident)
        self.assertIn("identity changed", w["error"])
        os.unlink(r.watch)                                            # missing
        _, w = h2.watcher_scan(r.m, off, ident)
        self.assertIsNotNone(w["error"])
        with self.assertRaises(OSError):
            h2.watcher_identity(r.m)

    def test_pack_snapshot_detects_mode_and_entry_changes(self):
        r = self.rig()
        pack = r.arms["A"]["pack"]
        before = h2.pack_snapshot(pack)
        self.assertEqual(before, h2.pack_snapshot(pack))
        os.chmod(os.path.join(pack, "pack.hsaco"), 0o644)
        self.assertNotEqual(before, h2.pack_snapshot(pack))
        os.chmod(os.path.join(pack, "pack.hsaco"), 0o444)
        self.assertEqual(before, h2.pack_snapshot(pack))
        r.unlock_packs()
        write_text(os.path.join(pack, "new.bin"), "x")
        r.lock_packs()
        self.assertNotEqual(before, h2.pack_snapshot(pack))

    def test_single_hold_minimum_and_new_ids_are_fresh(self):
        b = {"cell_cap_seconds": 1200, "kill_grace_seconds": 15, "drain_seconds": 90, "finalize_seconds": 45, "preflight_seconds": 30,
             "maximum_hold_seconds": 1440}
        self.assertEqual(h2.single_hold_min(b), 1380)
        a, c = h2.new_ids(), h2.new_ids()
        a["jobs"].append("x")
        self.assertEqual(c["jobs"], [])


if __name__ == "__main__":
    unittest.main()
