#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""gpu_oracle.py -- external QSA matched-input GPU oracle (stdlib only).

Runs the PREBUILT rdna-compute test binary with the exact filter
`tensor_ops::tests::qsa_gathered_wmma_matches_per_head_kernel --exact
--nocapture --test-threads=1` inside ONE admitted `correct`-class P1 fabric
hold on hipx/xtx. Reuses the frozen R7 kernel-watcher wrapper
(run-with-watcher.sh) and the H2 safety patterns (exact-group cleanup only,
private HOME/locks/kernel-cache, device/BDF checks, MemAvailable gate, no
pattern kills, no automatic retry). The H2 helper itself has NO single-binary
oracle API, so it is referenced as a reviewed safety pattern only -- this file
invokes exactly: R7 wrapper -> fabric-run.sh -> `gpu_oracle.py cell`.

Hold env pins HIPFIRE_FN_QSA_GATHER_GFX1100=1 (gfx1100 gather is default-off),
HIPFIRE_QWEN4_QSA_PM=1, HIPFIRE_QWEN4_QSA_WMMA_GATHER=1, HIPFIRE_QWEN4_F16_WMMA=1
(normally default-on) and HIPFIRE_QWEN4_D2_DEFAULT=0. The matched fixture only
proves the compiled PM-vs-HIP twin on identical synthetic inputs forced inside
the test; it never claims env-only production routing on the real 4K context.
The existing qual kernel pack may be staged read-only beside the hold (bound by
actual source identity); no new kernel compile or workaround is performed here.

This file is external Python SOURCE and must reach the hold by git deployment
from the allowed source branch, never rsync. The binary under test is labelled
honestly: 8be42771a plus the diagnostic format-only npu overlay (manifest
overlay.kind diagnostic-format-only, exact_clean_claim false), NEVER an
exact-clean 8be42771a build claim.

Subcommands (each requires --expected-manifest-sha256 pinned inline):
  preflight --manifest M --expected-manifest-sha256 S [--report FILE]
  launch    --manifest M --expected-manifest-sha256 S --name NAME
  cell      --manifest M --expected-manifest-sha256 S --out DIR   (inside hold only)
  wait      --manifest M --expected-manifest-sha256 S --name NAME [--max-wait-secs N]

Exit codes: 0 PASS-or-ok / 1 FAIL (raw preserved) / 2 REFUSED before any
mutation (launch/preflight) or admission failure / 3 INCOMPLETE (bus drop, no
retry) / 4 aborted-by-signal inside cell / 75 wait timeout.
No cargo is ever invoked in the hold; no model payload exists anywhere here.
JIT compilation of the already-embedded kernels may occur inside the hold into
the cell-private kcache. SOURCE-ONLY in this slice: nothing here has launched.
"""
import argparse
import ctypes
import importlib.util
import datetime
import hashlib
import json
import os
import re
import select
import shlex
import signal
import stat as statmod
import subprocess
import sys
import time

SCHEMA_MANIFEST = "s1.qsa_oracle.preparation.v1"
SCHEMA_RESULT = "s1.qsa_oracle.cell-result.v1"
SCHEMA_LAUNCH = "s1.qsa_oracle.launch.v1"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2
EXIT_INCOMPLETE = 3
EXIT_SIGNAL = 4
EXIT_WAIT_TIMEOUT = 75

FILTER = "tensor_ops::tests::qsa_gathered_wmma_matches_per_head_kernel"
FILTER_ARGS = ["--exact", "--nocapture", "--test-threads=1"]

STRIP_PREFIXES = ("HIPFIRE_", "HIP_", "ROCR_", "ROC_", "HSA_", "GPU_", "AMD_")
KEEP_ENV = ("FABRIC_JOB", "FABRIC_CARDS", "HIPFIRE_FABRIC_ADMISSION")


def eprint(*a):
    print(*a, file=sys.stderr)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_canonical_abs(p):
    return (isinstance(p, str) and p != "" and "\x00" not in p
            and os.path.isabs(p) and os.path.normpath(p) == p)


def nested(a, b):
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, data):
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(os.path.dirname(path) or ".")


def load_manifest(mpath, expected_sha):
    errs = []
    if not is_canonical_abs(mpath):
        return None, None, ["manifest path must be canonical absolute"]
    try:
        with open(mpath, "rb") as f:
            raw = f.read()
    except OSError as e:
        return None, None, ["cannot read manifest: %s" % e]
    got = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha or ""):
        return None, None, ["--expected-manifest-sha256 must be 64 hex chars"]
    if got != expected_sha:
        return None, None, ["manifest sha mismatch: file=%s expected=%s" % (got, expected_sha)]
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return None, None, ["manifest is not strict JSON: %s" % e]
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_MANIFEST:
        return None, None, ["manifest schema must be %s" % SCHEMA_MANIFEST]
    return doc, raw, errs


def git_out(wt, *args):
    try:
        r = subprocess.run(["git", "-C", wt] + list(args),
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "git failed: %s" % e
    if r.returncode != 0:
        return None, "git %s: %s" % (" ".join(args), (r.stderr or "").strip()[:300])
    return (r.stdout or "").strip(), None


def load_safety(m):
    spec = importlib.util.spec_from_file_location("s1_qsa_frozen_h2", m["safety"]["h2_official"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def admission_checks(m):
    """Read-only admission checks. Returns list of error strings (empty = ok)."""
    errs = []
    src = m.get("source", {})
    binn = m.get("binary", {})
    ora = m.get("oracle", {})
    ov = m.get("overlay", {})
    fab = m.get("fabric", {})
    run = m.get("runtime", {})
    wat = m.get("watcher", {})
    saf = m.get("safety", {})
    pk = m.get("pack", {})
    if "model_path" in m or "model" in m:
        errs.append("manifest must carry no model payload reference at all")

    opath = ora.get("path", "")
    if not is_canonical_abs(opath) or not os.path.isfile(opath):
        errs.append("oracle.path missing or not canonical: %r" % opath)
    elif ora.get("gpu_oracle_sha256", "") != sha256_file(opath):
        errs.append("oracle file hash differs from manifest oracle.gpu_oracle_sha256")

    wrap = wat.get("run_with_watcher", "")
    if not is_canonical_abs(wrap) or not os.path.isfile(wrap):
        errs.append("watcher wrapper missing: %r" % wrap)
    elif wat.get("run_with_watcher_sha256", "") != sha256_file(wrap):
        errs.append("R7 wrapper hash differs from manifest watcher.run_with_watcher_sha256")

    href = saf.get("h2_official", "")
    if not is_canonical_abs(href) or not os.path.isfile(href):
        errs.append("H2 safety reference missing: %r" % href)
    elif saf.get("h2_official_sha256", "") != sha256_file(href):
        errs.append("H2 helper hash differs from manifest safety.h2_official_sha256")

    frun = fab.get("fabric_run", "")
    if not is_canonical_abs(frun) or not os.path.isfile(frun):
        errs.append("fabric-run.sh missing: %r" % frun)
    elif not os.access(frun, os.X_OK):
        errs.append("fabric-run.sh not executable: %r" % frun)

    wt = src.get("worktree", "")
    if not is_canonical_abs(wt) or not os.path.realpath(wt).startswith("/home/kaden/qcal/release-0.4.3/"):
        return errs + ["source worktree is outside the allowed release root"]
    if not is_canonical_abs(wt) or not os.path.isdir(wt):
        errs.append("source worktree missing: %r" % wt)
    else:
        tip, e1 = git_out(wt, "rev-parse", "HEAD")
        tree, e2 = git_out(wt, "rev-parse", "HEAD^{tree}")
        dirty, e3 = git_out(wt, "status", "--porcelain")
        if e1:
            errs.append("source tip unreadable: %s" % e1)
        elif tip != src.get("tip"):
            errs.append("source tip %s != manifest %s" % (tip, src.get("tip")))
        if e2:
            errs.append("source tree unreadable: %s" % e2)
        elif tree != src.get("tree"):
            errs.append("source tree %s != manifest %s" % (tree, src.get("tree")))
        if e3:
            errs.append("source cleanliness unreadable: %s" % e3)
        elif dirty != "":
            errs.append("source worktree not clean: %r" % dirty[:200])
        anchor = os.path.join(wt, "crates", "rdna-compute", "src", "tensor_ops.rs")
        try:
            with open(anchor, "r", encoding="utf-8", errors="strict") as f:
                found = False
                for line in f:
                    if "fn qsa_gathered_wmma_matches_per_head_kernel" in line:
                        found = True
                        break
            if not found:
                errs.append("test anchor fn missing in deployed tensor_ops.rs")
        except OSError as e:
            errs.append("cannot read deployed tensor_ops.rs: %s" % e)
    if ov:
        if ov.get("kind") != "diagnostic-format-only":
            errs.append("overlay.kind must be diagnostic-format-only, never exact-clean")
        if ov.get("exact_clean_claim") is not False:
            errs.append("overlay.exact_clean_claim must be false explicitly")
        for field in ("overlay_sha256", "head_sha256", "provenance"):
            if not ov.get(field):
                errs.append("overlay.%s required when overlay present" % field)
        if ov.get("overlay_sha256") and not re.fullmatch(r"[0-9a-f]{64}", ov.get("overlay_sha256")):
            errs.append("overlay.overlay_sha256 must be 64-hex")
        if ov.get("head_sha256") and not re.fullmatch(r"[0-9a-f]{64}", ov.get("head_sha256")):
            errs.append("overlay.head_sha256 must be 64-hex")

    art = binn.get("artifact", "")
    if not is_canonical_abs(art) or not os.path.isfile(art):
        errs.append("binary artifact missing: %r" % art)
    else:
        try:
            st = os.stat(art)
        except OSError as e:
            errs.append("cannot stat artifact: %s" % e)
            st = None
        if st is not None:
            if st.st_size != binn.get("size_bytes"):
                errs.append("artifact size %d != manifest %s" % (st.st_size, binn.get("size_bytes")))
            if st.st_dev != binn.get("device"):
                errs.append("artifact device %d != manifest %s" % (st.st_dev, binn.get("device")))
            if st.st_ino != binn.get("inode"):
                errs.append("artifact inode %d != manifest %s" % (st.st_ino, binn.get("inode")))
            if st.st_mtime_ns != binn.get("mtime_ns"):
                errs.append("artifact mtime_ns %d != manifest %s" % (st.st_mtime_ns, binn.get("mtime_ns")))
            if statmod.S_IMODE(st.st_mode) != 0o555:
                errs.append("artifact mode %o is not 0555" % statmod.S_IMODE(st.st_mode))
            if sha256_file(art) != binn.get("sha256"):
                errs.append("artifact sha256 differs from manifest binary.sha256")
    stage = binn.get("stage_symlink", "")
    if not is_canonical_abs(stage) or not os.path.islink(stage):
        errs.append("stage symlink missing/not-a-link: %r" % stage)
    elif os.path.realpath(stage) != os.path.realpath(art):
        errs.append("stage symlink does not resolve to artifact")

    if pk:
        for field in ("path", "sha256", "source_commit"):
            if not pk.get(field):
                errs.append("pack.%s required when pack section present" % field)
        ppath = pk.get("path", "")
        if ppath and (not is_canonical_abs(ppath) or not os.path.isfile(ppath)):
            errs.append("pack.path missing or not canonical: %r" % ppath)
        elif ppath and pk.get("sha256", "") != sha256_file(ppath):
            errs.append("pack tarball hash differs from manifest pack.sha256")
        if pk.get("source_commit") and not re.fullmatch(r"[0-9a-f]{40}", pk.get("source_commit")):
            errs.append("pack.source_commit must be 40-hex when present")
    if m.get("test", {}).get("filter") != FILTER:
        errs.append("test.filter is not the frozen exact filter")
    if m.get("test", {}).get("args") != FILTER_ARGS:
        errs.append("test.args are not the frozen --exact --nocapture --test-threads=1")

    for k, want in (("class", "correct"), ("card", "xtx"), ("priority", "P1"),
                    ("uuid", fab.get("uuid")), ("bdf", fab.get("bdf"))):
        if not fab.get(k):
            errs.append("fabric.%s missing" % k)
    if fab.get("class") != "correct":
        errs.append("fabric.class must be correct (not timing)")
    if fab.get("card") != "xtx":
        errs.append("fabric.card must be xtx only")
    if fab.get("priority") != "P1":
        errs.append("fabric.priority must be P1")
    try:
        if not (float(fab.get("mem_gb", 0)) == 22.4):
            errs.append("fabric.mem_gb must be 22.4")
    except (TypeError, ValueError):
        errs.append("fabric.mem_gb not numeric")
    try:
        if int(fab.get("max_secs", 0)) > 900 or int(fab.get("max_secs", 0)) <= 0:
            errs.append("fabric.max_secs must be within (0,900]")
    except (TypeError, ValueError):
        errs.append("fabric.max_secs not an integer")
    if run.get("cargo_forbidden") is not True:
        errs.append("runtime.cargo_forbidden must be true")

    rr = m.get("run_root", "")
    if not is_canonical_abs(rr):
        errs.append("run_root must be canonical absolute")
    return errs


def cmd_preflight(args):
    m, raw, errs = load_manifest(args.manifest, args.expected_manifest_sha256)
    if errs:
        eprint("PREFLIGHT_REFUSED")
        for e in errs:
            eprint("  - %s" % e)
        return EXIT_REFUSED
    errs = admission_checks(m)
    if errs:
        eprint("PREFLIGHT_REFUSED")
        for e in errs:
            eprint("  - %s" % e)
        return EXIT_REFUSED
    rep = {"schema": "s1.qsa_oracle.preflight.v1", "status": "PREFLIGHT_OK",
           "manifest_sha256": args.expected_manifest_sha256,
           "source": m["source"], "overlay": m.get("overlay", {}),
           "binary_sha256": m["binary"]["sha256"],
           "checked_utc": utc_now()}
    if args.report:
        if os.path.exists(args.report) or os.path.islink(args.report):
            eprint("refusing existing --report: %s" % args.report)
            return EXIT_REFUSED
        write_atomic(args.report, (json.dumps(rep, indent=2, sort_keys=True) + "\n").encode())
    print("PREFLIGHT_OK manifest=%s tip=%s binary=%s" % (
        args.expected_manifest_sha256[:12], m["source"]["tip"][:12],
        m["binary"]["sha256"][:12]))
    return EXIT_OK


def cmd_launch(args):
    m, raw, errs = load_manifest(args.manifest, args.expected_manifest_sha256)
    if errs:
        eprint("LAUNCH_REFUSED")
        for e in errs:
            eprint("  - %s" % e)
        return EXIT_REFUSED
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.name or ""):
        eprint("LAUNCH_REFUSED: --name must match [A-Za-z0-9._-]{1,64}")
        return EXIT_REFUSED
    errs = admission_checks(m)
    if errs:
        eprint("LAUNCH_REFUSED")
        for e in errs:
            eprint("  - %s" % e)
        return EXIT_REFUSED
    root = os.path.join(m["run_root"], args.name)
    for p in (root, os.path.join(root, "watch"), os.path.join(root, "watch.exit"),
              os.path.join(root, "cell"), os.path.join(root, "launch.json")):
        if os.path.exists(p) or os.path.islink(p):
            eprint("LAUNCH_REFUSED: refusing existing run path: %s" % p)
            return EXIT_REFUSED
    for other in (args.manifest, m["oracle"]["path"], m["binary"]["artifact"],
                  m["binary"]["stage_symlink"], m["source"]["worktree"]):
        if nested(root, other) or nested(other, root):
            eprint("LAUNCH_REFUSED: run dir nested with owned input: %s" % other)
            return EXIT_REFUSED
    os.makedirs(root)
    write_atomic(os.path.join(root, "manifest.submitted.json"), raw)
    argv_text = " ".join(shlex.quote(x) for x in
                         [sys.executable, m["oracle"]["path"], "launch",
                          "--manifest", args.manifest,
                          "--expected-manifest-sha256", args.expected_manifest_sha256,
                          "--name", args.name]) + "\n"
    write_atomic(os.path.join(root, "submitted-argv.txt"), argv_text.encode())
    md5line = "%s  %s\n%s  %s\n" % (
        hashlib.md5(open(m["oracle"]["path"], "rb").read()).hexdigest(), m["oracle"]["path"],
        hashlib.md5(raw).hexdigest(), args.manifest)
    write_atomic(os.path.join(root, "scripts.md5"), md5line.encode())

    fab = m["fabric"]
    inner = ("for f in /proc/$$/fd/*; do n=${f##*/}; "
             "[ \"$n\" -gt 2 ] 2>/dev/null && eval \"exec $n>&-\"; done; "
             "exec env OUT_PREFIX=\"$1/watch\" PYTHONDONTWRITEBYTECODE=1 "
             "bash \"$2\" \"$3\" --class correct --card xtx --mem-gb 22.4 "
             "--hostmem-gb 8 --hostmem-lock --max-secs \"$4\" --priority P1 "
             "--owner S1Qsa -- python3 \"$5\" cell --manifest \"$6\" "
             "--expected-manifest-sha256 \"$7\" --out \"$1/cell\"")
    cmd = ["nohup", "setsid", "bash", "-c", inner, "launcher", root,
           m["watcher"]["run_with_watcher"], fab["fabric_run"],
           str(int(fab["max_secs"])), m["oracle"]["path"],
           args.manifest, args.expected_manifest_sha256]
    logf = open(os.path.join(root, "launcher.log"), "xb")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                stdout=logf, stderr=subprocess.STDOUT,
                                close_fds=True)
    finally:
        try:
            logf.flush()
            os.fsync(logf.fileno())
        finally:
            logf.close()
    time.sleep(1)
    if proc.poll() is not None and proc.returncode != 0:
        eprint("LAUNCH_REFUSED: detached launcher died immediately rc=%s" % proc.returncode)
        return EXIT_REFUSED
    try:
        pgid = os.getpgid(proc.pid)
    except OSError as e:
        eprint("LAUNCH_REFUSED: launcher not live: %s" % e)
        return EXIT_REFUSED
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot_id = f.read().strip()
    except OSError:
        boot_id = "unknown"
    launch = {"schema": SCHEMA_LAUNCH, "name": args.name, "pid": proc.pid,
              "pgid": pgid, "boot_id": boot_id, "launch_unix": time.time(),
              "starttime_ticks": load_safety(m).proc_stat(proc.pid)["starttime"],
              "manifest_sha256": args.expected_manifest_sha256,
              "watch_exit": os.path.join(root, "watch.exit")}
    waiter_log = open(os.path.join(root, "waiter.log"), "xb")
    try:
        waiter = subprocess.Popen(
            ["nohup", "setsid", sys.executable, m["oracle"]["path"], "wait",
             "--manifest", args.manifest, "--expected-manifest-sha256", args.expected_manifest_sha256,
             "--name", args.name],
            stdin=subprocess.DEVNULL, stdout=waiter_log, stderr=subprocess.STDOUT, close_fds=True)
        launch["waiter"] = {"pid": waiter.pid,
                            "starttime_ticks": load_safety(m).proc_stat(waiter.pid)["starttime"]}
    finally:
        waiter_log.close()
    blob = (json.dumps(launch, indent=2, sort_keys=True) + "\n").encode()
    write_atomic(os.path.join(root, "launch.json"), blob)
    sys.stdout.write(blob.decode())
    return EXIT_OK


def sysfs_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def mem_available_kib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def vram_used_bytes(bdf):
    t = sysfs_text("/sys/bus/pci/devices/%s/mem_info_vram_used" % bdf)
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def build_cell_env(m, out):
    env = {}
    for k, v in os.environ.items():
        if k in KEEP_ENV or k == "PATH" or k == "LD_LIBRARY_PATH":
            env[k] = v
            continue
        if k.startswith(STRIP_PREFIXES):
            continue
        env[k] = v
    fab = m["fabric"]
    env.update({
        "HOME": os.path.join(out, "home"),
        "HIPFIRE_HOME": os.path.join(out, "home", ".hipfire"),
        "HIPFIRE_LOCK_DIR": os.path.join(out, "lockdir"),
        "HIPFIRE_KERNEL_CACHE": os.path.join(out, "kcache"),
        "HIPFIRE_DEVICES": fab["uuid"],
        "ROCR_VISIBLE_DEVICES": fab["uuid"],
        "HIP_VISIBLE_DEVICES": "0",
        "HIPFIRE_NO_REGISTRY_FETCH": "1",
        "HIPFIRE_QWEN4_D2_DEFAULT": "0",
        "HIPFIRE_FN_QSA_GATHER_GFX1100": "1",
        "HIPFIRE_QWEN4_QSA_PM": "1",
        "HIPFIRE_QWEN4_QSA_WMMA_GATHER": "1",
        "HIPFIRE_QWEN4_F16_WMMA": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def classify_output(text, rc):
    m_run = re.search(r"running (\d+) test", text)
    running = int(m_run.group(1)) if m_run else None
    m_res = re.search(r"test result:\s*(\w+)\.\s*(\d+) passed;\s*(\d+) failed;\s*(\d+) ignored;", text)
    status = m_res.group(1) if m_res else None
    passed = int(m_res.group(2)) if m_res else None
    failed = int(m_res.group(3)) if m_res else None
    ignored = int(m_res.group(4)) if m_res else None
    detail = {"rc": rc, "running": running, "status": status,
              "passed": passed, "failed": failed, "ignored": ignored}
    one_pass = (rc == 0 and running == 1 and status == "ok" and passed == 1
                and failed == 0 and ignored == 0)
    if re.search(r"(?i)\bskip(?:ped)?\b|no GPU|cannot open|No such device|no device", text):
        return "FAIL", "SKIP_REJECTED: test returned without exercising the GPU", detail
    if one_pass:
        return "PASS", "one exact test passed", detail
    if re.search(r"^test \S+ ignored", text, re.M) or "running 0 tests" in text:
        return "FAIL", "SKIP_REJECTED: test did not execute (ignored/0 tests)", detail
    if status == "FAILED" or (failed is not None and failed > 0):
        return "FAIL", "assertion failure: compiled test failed", detail
    if m_res is None:
        return "FAIL", "NO_RESULT: no test-result summary in raw output", detail
    return "FAIL", "NOT_ONE_PASS: expected exactly 1 passed/0 failed/0 ignored", detail


def cmd_cell(args):
    m, raw, errs = load_manifest(args.manifest, args.expected_manifest_sha256)
    if not is_canonical_abs(args.out):
        eprint("CELL_REFUSED: --out must be canonical absolute")
        return EXIT_REFUSED
    if os.path.exists(args.out) or os.path.islink(args.out):
        eprint("CELL_REFUSED: refusing existing --out")
        return EXIT_REFUSED
    if errs:
        eprint("CELL_ADMISSION_FAIL")
        for e in errs:
            eprint("  - %s" % e)
        return EXIT_FAIL
    errs = admission_checks(m)
    if errs:
        eprint("CELL_ADMISSION_FAIL: " + "; ".join(errs))
        return EXIT_REFUSED
    helper = load_safety(m)
    errs = helper.check_fabric(os.environ, {"target": m["fabric"]})
    if errs:
        eprint("CELL_REFUSED: " + "; ".join(errs))
        return EXIT_REFUSED
    os.makedirs(args.out)
    for sub in ("home", "lockdir", "kcache"):
        os.makedirs(os.path.join(args.out, sub))
    fab = m["fabric"]
    bdf = fab["bdf"]
    hosts = {"vendor": sysfs_text("/sys/bus/pci/devices/%s/vendor" % bdf),
             "device": sysfs_text("/sys/bus/pci/devices/%s/device" % bdf),
             "unique_id": sysfs_text("/sys/bus/pci/devices/%s/unique_id" % bdf),
             "vram_used_before": vram_used_bytes(bdf),
             "mem_available_kib_before": mem_available_kib()}
    pre = {"label": "qsa-oracle", "host": os.uname().nodename,
           "fabric_job": os.environ.get("FABRIC_JOB", ""),
           "fabric_cards": os.environ.get("FABRIC_CARDS", ""),
           "boot_id": sysfs_text("/proc/sys/kernel/random/boot_id"),
           "start_unix": time.time(), "device": hosts,
           "binary_sha256": m["binary"]["sha256"]}
    write_atomic(os.path.join(args.out, "cell-preflight.json"),
                 (json.dumps(pre, indent=2, sort_keys=True) + "\n").encode())
    if (hosts["vendor"] != fab.get("vendor") or hosts["device"] != fab.get("device")):
        write_atomic(os.path.join(args.out, "XTX_DROP"),
                     ("XTX_DROP pre: vendor=%s device=%s bdf=%s\n" % (
                         hosts["vendor"], hosts["device"], bdf)).encode())
        write_atomic(os.path.join(args.out, "cell-result.json"),
                     (json.dumps({"schema": SCHEMA_RESULT, "verdict": "INCOMPLETE",
                                  "reason": "bus drop at preflight",
                                  "device": hosts}, indent=2) + "\n").encode())
        with open(os.path.join(args.out, "exit"), "w") as f:
            f.write("3\n")
        return EXIT_INCOMPLETE
    if hosts["vram_used_before"] is None or hosts["vram_used_before"] >= 2 * 1024**3:
        eprint("CELL_FAIL: card not idle: vram used %s" % hosts["vram_used_before"])
        with open(os.path.join(args.out, "exit"), "w") as f:
            f.write("1\n")
        return EXIT_FAIL
    if hosts["mem_available_kib_before"] is None or hosts["mem_available_kib_before"] < 80 * 1024**2:
        eprint("CELL_FAIL: MemAvailable %s KiB < 80 GiB" % hosts["mem_available_kib_before"])
        with open(os.path.join(args.out, "exit"), "w") as f:
            f.write("1\n")
        return EXIT_FAIL

    env = build_cell_env(m, args.out)
    with open(os.path.join(args.out, "env-applied.txt"), "w") as f:
        for k in sorted(env):
            if k.startswith(("HIPFIRE_", "HIP_", "ROCR_")) or k == "HOME":
                f.write("%s=%s\n" % (k, env[k]))
    argv = [m["binary"]["stage_symlink"], FILTER] + list(FILTER_ARGS)
    with open(os.path.join(args.out, "command.txt"), "w") as f:
        f.write(" ".join(shlex.quote(x) for x in argv) + "\n")
    out_path = os.path.join(args.out, "raw_stdout.txt")
    err_path = os.path.join(args.out, "raw_stderr.log")
    health_manifest = {
        "target": {"vram_used_path": "/sys/bus/pci/devices/%s/mem_info_vram_used" % bdf,
                   "vendor_id": fab["vendor"], "device_id": fab["device"], "uuid": fab["uuid"]},
        "watcher": {"log_path": os.path.join(os.path.dirname(args.out), "watch.dmesg.log"),
                    "fatal_regex": m["watcher"]["fatal_regex"]},
    }
    watcher_identity = helper.watcher_identity(health_manifest)
    watcher_offset = [0]
    def health():
        causes = []
        device = helper.device_state(helper.Host(), health_manifest)
        if not device["ok"]:
            causes.append({"status": "DEVICE_LOST", "device": device})
        watcher_offset[0], scan = helper.watcher_scan(health_manifest, watcher_offset[0], watcher_identity)
        if scan["fatal"] or scan["error"]:
            causes.append({"status": "WATCHER_FATAL" if scan["fatal"] else "WATCHER_UNREADABLE",
                           "watcher": scan})
        return causes
    old_handlers = {sig: signal.signal(sig, lambda *_: helper.STOP.set())
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        lifecycle = helper.spawn_and_wait(
            argv, env, args.out, out_path, err_path, os.path.join(args.out, "process.json"),
            int(m["runtime"].get("cell_cap_seconds", 600)), 10, helper.Host(), health=health)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    final_health = health()
    if final_health and not lifecycle.get("health_abort"):
        lifecycle["health_abort"] = {"causes": final_health}
    write_atomic(os.path.join(args.out, "lifecycle.json"),
                 (json.dumps(lifecycle, indent=2, sort_keys=True) + "\n").encode())
    rc = lifecycle.get("returncode")
    try:
        with open(out_path, "rb") as f:
            raw_out = f.read()
    except OSError:
        raw_out = b""
    try:
        with open(err_path, "rb") as f:
            raw_err = f.read()
    except OSError:
        raw_err = b""
    text = (raw_out + b"\n" + raw_err).decode("utf-8", errors="replace")
    vendor2 = sysfs_text("/sys/bus/pci/devices/%s/vendor" % bdf)
    for _ in range(90):
        u = vram_used_bytes(bdf)
        if u is not None and u < 2 * 1024**3:
            break
        time.sleep(1)
    post = {"end_unix": time.time(), "vendor_after": vendor2,
            "vram_used_after": vram_used_bytes(bdf),
            "mem_available_kib_after": mem_available_kib()}
    write_atomic(os.path.join(args.out, "postflight.json"),
                 (json.dumps(post, indent=2, sort_keys=True) + "\n").encode())
    if vendor2 != fab.get("vendor") or lifecycle.get("health_abort"):
        write_atomic(os.path.join(args.out, "XTX_DROP"),
                     ("XTX_DROP post: vendor=%s bdf=%s\n" % (vendor2, bdf)).encode())
        verdict, reason, detail = "INCOMPLETE", "bus drop after cell", {}
    else:
        verdict, reason, detail = classify_output(text, rc if rc is not None else 127)
        if (not lifecycle.get("spawned") or lifecycle.get("timed_out") or lifecycle.get("aborted")
                or lifecycle.get("leftover_after_exit") or lifecycle.get("group_after_cleanup")
                or post["vram_used_after"] is None or post["vram_used_after"] >= 2 * 1024**3):
            verdict, reason = "FAIL", "process lifecycle or VRAM drain failed"
        if os.path.exists(os.path.join(args.out, "TIMEOUT")) and verdict == "PASS":
            verdict, reason = "FAIL", "TIMEOUT flag present; PASS refused"
    result = {"schema": SCHEMA_RESULT, "verdict": verdict, "reason": reason,
              "rc": rc, "detail": detail, "checked_utc": utc_now(),
              "raw_stdout": out_path, "raw_stderr": err_path}
    write_atomic(os.path.join(args.out, "cell-result.json"),
                 (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    code = {"PASS": EXIT_OK, "FAIL": EXIT_FAIL, "INCOMPLETE": EXIT_INCOMPLETE}.get(verdict, EXIT_INCOMPLETE)
    with open(os.path.join(args.out, "exit"), "w") as f:
        f.write("%d\n" % code)
    print("%s: %s" % (verdict, reason))
    return code


def cmd_wait(args):
    m, raw, errs = load_manifest(args.manifest, args.expected_manifest_sha256)
    if errs:
        eprint("WAIT_REFUSED")
        for e in errs:
            eprint("  - %s" % e)
        return EXIT_REFUSED
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.name or ""):
        eprint("WAIT_REFUSED: bad --name")
        return EXIT_REFUSED
    root = os.path.join(m["run_root"], args.name)
    watch_exit = os.path.join(root, "watch.exit")
    if not os.path.isdir(root):
        eprint("WAIT_REFUSED: unknown run: %s" % root)
        return EXIT_REFUSED
    maxw = int(args.max_wait_secs)
    start = time.monotonic()
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    try:
        if libc.inotify_add_watch(fd, os.fsencode(root), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")
        while not os.path.exists(watch_exit):
            remaining = maxw - (time.monotonic() - start)
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                print("WAIT_TIMEOUT %s" % args.name)
                return EXIT_WAIT_TIMEOUT
            os.read(fd, 1024 * 1024)
    finally:
        os.close(fd)
    try:
        with open(watch_exit) as f:
            wrc = f.read().strip()
    except OSError:
        wrc = "?"
    print("exit=%s" % wrc)
    flog = os.path.join(root, "watch.fabric.log")
    try:
        with open(flog, errors="replace") as f:
            blob = f.read()
        print(blob[-2000:])
    except OSError:
        print("(watch.fabric.log unreadable)")
    cellres = os.path.join(root, "cell", "cell-result.json")
    verdict, reason = "INCOMPLETE", "cell result missing"
    try:
        with open(cellres) as f:
            res = json.load(f)
        verdict, reason = res.get("verdict", "INCOMPLETE"), res.get("reason", "")
        print("cell: %s: %s (rc=%s detail=%s)" % (
            verdict, reason, res.get("rc"), json.dumps(res.get("detail", {}), sort_keys=True)))
    except (OSError, ValueError) as e:
        print("cell-result unreadable: %s" % e)
    if os.path.exists(os.path.join(root, "cell", "XTX_DROP")):
        print("XTX_DROP PRESENT")
        verdict = "INCOMPLETE"
    if wrc != "0" and verdict == "PASS":
        verdict, reason = "FAIL", "watcher/fabric command did not exit cleanly"
    try:
        r = subprocess.run(["pgrep", "-af", root], capture_output=True,
                           text=True, timeout=30)
        hits = [l for l in (r.stdout or "").splitlines() if "pgrep" not in l]
        if hits:
            verdict, reason = "FAIL", "post-job owned processes remain"
            print("post-job leftovers:")
            for l in hits[:20]:
                print("  " + l)
        else:
            print("post-job: no process for %s" % args.name)
    except (OSError, subprocess.TimeoutExpired) as e:
        print("post-job pgrep unavailable: %s" % e)
    print(json.dumps({"name": args.name, "watch_exit": wrc, "verdict": verdict,
                      "reason": reason}, sort_keys=True))
    return {"PASS": EXIT_OK, "FAIL": EXIT_FAIL}.get(verdict, EXIT_INCOMPLETE)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpu_oracle.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("preflight")
    a.add_argument("--manifest", required=True)
    a.add_argument("--expected-manifest-sha256", required=True)
    a.add_argument("--report", default=None)
    b = sub.add_parser("launch")
    b.add_argument("--manifest", required=True)
    b.add_argument("--expected-manifest-sha256", required=True)
    b.add_argument("--name", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--manifest", required=True)
    c.add_argument("--expected-manifest-sha256", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("wait")
    d.add_argument("--manifest", required=True)
    d.add_argument("--expected-manifest-sha256", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--max-wait-secs", default="1800")
    args = p.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "launch":
        return cmd_launch(args)
    if args.cmd == "cell":
        return cmd_cell(args)
    if args.cmd == "wait":
        return cmd_wait(args)
    return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
