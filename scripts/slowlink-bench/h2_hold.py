#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""h2_hold.py -- literal, versioned Python command dispatcher for ONE official H2 hold.  Python stdlib only.

    h2_hold.py --manifest M --expected-manifest-sha256 M_SHA --expected-dispatcher-sha256 D_SHA --pair 1|2|3 --cell 1|2
               --out FRESH_DIR --watcher-prefix FRESH_PREFIX --campaign CAMPAIGN_DIR

WHY THIS FILE EXISTS.  The first operational caller (job j1970) aborted BEFORE the core harness, any model or any GPU because a parent shell/JS string
escaping produced a control character and an invalid multi-line embedded Python.  Raw failure preserved.  This dispatcher replaces that fragile inline
timing call with ordinary, tested, versioned source: the held command is just `python3 h2_hold.py <flags>`.  No code is generated, escaped or embedded.

AUTHORIZATION / STATUS.  Main authorized an H2-ONLY run.  The raw H1b FAIL is RETAINED and is NOT turned into a quality PASS by anything here; the
manifest's h1b section is checked by the unchanged core exactly as before.  This file adds NO metric, threshold, default, math or runtime behaviour:
the official core (h2_official.py, same directory, byte-frozen) alone measures, validates, assembles and judges.  SOURCE-ONLY until its CPU tests
and AST gate run; nothing here has executed a GPU hold.

WHAT ONE INVOCATION DOES (one invocation == one admitted fabric hold == one cell):
  1. ADMISSION, strictly BEFORE any mutation (a refusal exits 2 and writes nothing, repoints nothing, spawns nothing):
       - all paths canonical absolute; this file's own sha256 == --expected-dispatcher-sha256 (supplied independently in the submitted argv);
       - the core is imported ONLY from this file's own directory; h2_official.load_manifest(M, --expected-manifest-sha256) is the ACTUAL manifest
         check, and manifest.harness.h2_official_sha256 must equal the core's own self hash (the core's own guard is unchanged and runs again);
       - the actual FABRIC_JOB exists (no invented production ids);
       - campaign STOP.json absent; --out, the watcher binding file and every dispatcher-owned campaign output are FRESH (never overwritten);
       - --out is not nested with the campaign, the manifest, the watcher paths or any earlier cell's OUT;
       - pair N needs campaign/pair1.json..pair<N-1>.json, each strict JSON with status PAIR_COMPLETE;
       - cell 2 needs cell 1's ledger and cell.json; this file hashes cell 1's cell.json NATIVELY and passes that as --first-cell-sha256
         (the core re-validates raw stdout, manifest, order, chain);
       - manifest.watcher.log_path is ONE campaign SYMLINK (never a regular file, never a chain of links) and the wrapper's
         <watcher-prefix>.dmesg.log already exists as a regular file.
  2. INSIDE the held command only: atomically replace an owned temp symlink so the campaign watcher link points at <watcher-prefix>.dmesg.log,
     assert readlink/realpath/identity and record <watcher-prefix>.watcher-binding-before.json (native dev/inode/size, real FABRIC_JOB, UTC).
     The parent collects the final target hash after this wrapper exits; the core tracks watcher identity during the cell.
  3. Invoke EXACTLY ONE fresh core `run` subprocess (same process group, environment inherited unchanged, no extra variable, no new session).
     No journalctl, no sampler, no second watcher, no GPU API, no retry.
  4. Cell 1: ledger campaign/pair<P>-cell1.hold.json (cell file sha256 computed natively) and stop; no assembly.
     Cell 2: ledger, then pure-CPU `assemble-pair` (campaign/pair<P>.json) and `judge` over the previous pairs plus this one
     (campaign/judge-pair<P>.json).  UNKNOWN_INCOMPLETE with no errors after pair 1 or 2 is allowed; PASS is allowed only after pair 3.
     FAIL / FAIL_EARLY_ORDER_STATISTIC creates STOP.json and exits 1; any other judge outcome creates STOP.json and exits 3.
  A failed run/assemble/judge writes STOP.json (fail closed, with the raw return code and raw stdout/stderr/receipt paths) and exits nonzero; every
  later held command refuses at admission, so no later GPU cell can start.  Raw receipts and raw core output are never altered; nothing is zero-filled
  and an all-nine median is never manufactured: only the core judge may state one.

EXIT CODES.  0 hold done / 1 judge FAIL or FAIL_EARLY (STOP written) / 2 refused before any mutation / 3 STOP written (run, assemble, judge or
harness failure) / 4 core `run` aborted by signal (STOP written).

SIGNALS.  The core child shares this process group (fabric SIGTERM reaches both).  This process never detaches the child and never dies before
it: SIGTERM/SIGINT/SIGHUP are forwarded to the OWN child only, then the child is awaited.  No pattern kills.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import stat as statmod
import subprocess
import sys
import threading

SCHEMA_STOP = "s1.h2_hold.stop.v1"
SCHEMA_BINDING = "s1.h2_hold.watcher-binding-before.v1"
SCHEMA_LEDGER = "s1.h2_hold.cell-ledger.v1"
DISPATCHER_VERSION = "h2_hold.r3.v1"

ORDER = {1: ("A", "B"), 2: ("B", "A"), 3: ("A", "B")}
MAX_JSON = 256 << 20

EXIT_OK = 0
EXIT_JUDGE_FAIL = 1
EXIT_REFUSED = 2
EXIT_STOPPED = 3
EXIT_SIGNAL = 4

JUDGE_FAIL_VERDICTS = ("FAIL", "FAIL_EARLY_ORDER_STATISTIC")
JUDGE_RC = {"PASS": 0, "FAIL": 1, "FAIL_EARLY_ORDER_STATISTIC": 1, "UNKNOWN_INCOMPLETE": 3}


class Refused(Exception):
    """Admission refused BEFORE any mutation."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


class HoldError(Exception):
    """A failure after admission: closes the hold with STOP.json."""


# ----------------------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def is_canonical(path):
    return (isinstance(path, str) and path != "" and "\x00" not in path and os.path.isabs(path) and os.path.normpath(path) == path)


def nested(a, b):
    """True when a == b or either path lies inside the other."""
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def emit(doc):
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


_CORE = None


def core_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "h2_official.py")


def load_core():
    """The actual official core, imported ONLY from this file's own directory (never from sys.path)."""
    global _CORE
    if _CORE is None:
        spec = importlib.util.spec_from_file_location("h2_official", core_path())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CORE = mod
    return _CORE


def read_json_file(core, path, what, errs):
    """Strict-JSON object from a REGULAR (non-symlink) file, or None with an error appended."""
    try:
        st = os.lstat(path)
    except OSError as e:
        errs.append(f"{what} {path} unreadable: {e}")
        return None
    if not statmod.S_ISREG(st.st_mode):
        errs.append(f"{what} {path} is not a regular file")
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON:
            errs.append(f"{what} {path} is larger than {MAX_JSON} bytes")
            return None
        doc = core.loads_strict(raw)
    except (OSError, ValueError) as e:
        errs.append(f"{what} {path} is not readable strict JSON: {e}")
        return None
    if not isinstance(doc, dict):
        errs.append(f"{what} {path} is not a JSON object")
        return None
    return doc


# ----------------------------------------------------------------------------------------------------------------------------------
# paths owned by the dispatcher
# ----------------------------------------------------------------------------------------------------------------------------------
def cell_id_for(pair, cell):
    return f"{ORDER[pair][cell - 1]}{pair}"


def cell_json_path(out, cell_id):
    return os.path.join(out, cell_id, "cell.json")


def ledger_path(campaign, pair, cell):
    return os.path.join(campaign, f"pair{pair}-cell{cell}.hold.json")


def raw_paths(campaign, pair, cell, stage):
    return (os.path.join(campaign, f"pair{pair}-cell{cell}.{stage}.stdout.json"), os.path.join(campaign, f"pair{pair}-cell{cell}.{stage}.stderr.log"))


def pair_path(campaign, pair):
    return os.path.join(campaign, f"pair{pair}.json")


def judge_path(campaign, pair):
    return os.path.join(campaign, f"judge-pair{pair}.json")


# ----------------------------------------------------------------------------------------------------------------------------------
# admission (NO mutation of any kind)
# ----------------------------------------------------------------------------------------------------------------------------------
def admit(args, core, environ):
    errs = []
    pair, cell = args.pair, args.cell
    for flag, val in (("--manifest", args.manifest), ("--out", args.out), ("--watcher-prefix", args.watcher_prefix), ("--campaign", args.campaign)):
        if not is_canonical(val):
            errs.append(f"{flag} {val!r} must be a canonical absolute path")
    if errs:
        raise Refused(errs)
    own = sha256_file(os.path.realpath(__file__))
    if own != args.expected_dispatcher_sha256:
        raise Refused([f"dispatcher source sha256 {own} != the independently supplied --expected-dispatcher-sha256"])
    if os.path.realpath(getattr(core, "__file__", "")) != core_path():
        raise Refused(["the core is not the h2_official.py beside this dispatcher"])
    if {k: tuple(v) for k, v in core.ORDER.items()} != ORDER:
        raise Refused(["core ORDER differs from the dispatcher's fixed A1 B1 B2 A2 A3 B3 order"])
    try:
        m, msha = core.load_manifest(args.manifest, args.expected_manifest_sha256)
    except core.H2Error as e:
        raise Refused([str(e)])
    core_sha = core.self_sha256()
    if core_sha != m["harness"]["h2_official_sha256"]:
        errs.append(f"core sha256 {core_sha} != manifest.harness.h2_official_sha256")
    job = environ.get("FABRIC_JOB")
    if not job:
        errs.append("FABRIC_JOB is not set: the hold must run inside an actual fabric job")
    if os.path.realpath(args.manifest) != args.manifest:
        errs.append(f"--manifest {args.manifest} is not a real (symlink-free) path")

    campaign, out, prefix = args.campaign, args.out, args.watcher_prefix
    if not os.path.isdir(campaign) or os.path.islink(campaign) or os.path.realpath(campaign) != campaign:
        errs.append(f"--campaign {campaign} must be an existing real directory")
    stop = os.path.join(campaign, "STOP.json")
    if os.path.lexists(stop):
        errs.append(f"campaign {stop} exists: the campaign is stopped, no further hold may start")

    cell_id = cell_id_for(pair, cell)
    expected_out_json = cell_json_path(out, cell_id)

    # --out: fresh, own parent, not nested with anything protected
    if os.path.lexists(out):
        errs.append(f"--out {out} already exists: it must be a FRESH directory")
    parent = os.path.dirname(out)
    if not os.path.isdir(parent) or os.path.realpath(parent) != parent:
        errs.append(f"parent of --out {out} must be an existing real directory")
    protected = [("campaign", campaign), ("manifest", args.manifest)]

    # earlier pairs
    previous = []
    for i in range(1, pair):
        p = pair_path(campaign, i)
        doc = read_json_file(core, p, f"previous pair {i}", errs)
        if doc is not None:
            if doc.get("pair") != i or doc.get("status") != "PAIR_COMPLETE":
                errs.append(f"previous pair {i} {p} is not a PAIR_COMPLETE pair-{i} receipt")
            protected.append((f"previous pair {i}", p))
        previous.append(p)

    # earlier cells (ledgers): their OUT dirs are protected
    prior = [(i, c) for i in range(1, pair) for c in (1, 2)] + ([(pair, 1)] if cell == 2 else [])
    first = None
    for i, c in prior:
        lp = ledger_path(campaign, i, c)
        if not os.path.lexists(lp):
            if (i, c) == (pair, 1):
                errs.append(f"cell 2 needs cell 1's ledger {lp}: the first cell of this pair has not completed")
            continue
        led = read_json_file(core, lp, f"ledger pair{i} cell{c}", errs)
        if led is None:
            continue
        led_out, led_json = led.get("out"), led.get("cell_json")
        if not (is_canonical(led_out) and is_canonical(led_json)) or led.get("pair") != i or led.get("cell") != c:
            errs.append(f"ledger {lp} is malformed")
            continue
        protected.append((f"earlier cell pair{i} cell{c} OUT", led_out))
        if (i, c) == (pair, 1):
            exp_json = cell_json_path(led_out, cell_id_for(pair, 1))
            try:
                st = os.lstat(led_json)
                if led_json != exp_json or not statmod.S_ISREG(st.st_mode):
                    raise OSError("first cell.json is not the expected regular file")
                native = sha256_file(led_json)
            except OSError as e:
                errs.append(f"first cell file {led_json} unusable: {e}")
                continue
            if native != led.get("cell_file_sha256"):
                errs.append(f"first cell file {led_json} sha256 {native} != the ledger's {led.get('cell_file_sha256')}")
                continue
            first = {"path": led_json, "sha256": native, "out": led_out}

    # watcher
    wl = m["watcher"]["log_path"]
    target = prefix + ".dmesg.log"
    binding = prefix + ".watcher-binding-before.json"
    wdir = os.path.dirname(wl)
    if os.path.realpath(wdir) != wdir or not os.path.isdir(wdir):
        errs.append(f"watcher link directory {wdir} must be an existing real directory")
    try:
        lst = os.lstat(wl)
    except OSError as e:
        lst = None
        errs.append(f"manifest watcher.log_path {wl} must exist as a campaign symlink: {e}")
    previous_target = None
    if lst is not None:
        if not statmod.S_ISLNK(lst.st_mode):
            errs.append(f"manifest watcher.log_path {wl} is not a symlink (a regular file is refused)")
        else:
            previous_target = os.readlink(wl)
            if os.path.islink(os.path.join(wdir, previous_target)):
                errs.append(f"manifest watcher.log_path {wl} is a chain of symlinks, not a single symlink")
    if wl == target:
        errs.append("--watcher-prefix log equals the campaign watcher link")
    try:
        tst = os.lstat(target)
        if not statmod.S_ISREG(tst.st_mode):
            errs.append(f"wrapper watcher log {target} is not a regular file")
        elif os.path.realpath(target) != target:
            errs.append(f"wrapper watcher log {target} is not a real (symlink-free) path")
    except OSError as e:
        errs.append(f"wrapper watcher log {target} must already exist: {e}")
    if os.path.lexists(binding):
        errs.append(f"watcher binding file {binding} already exists: it must be fresh")
    protected += [("watcher link", wl), ("wrapper watcher log", target), ("watcher binding file", binding)]
    for what, p in protected:
        if nested(out, p):
            errs.append(f"--out {out} is nested with protected path ({what}) {p}")

    # dispatcher-owned outputs must be fresh
    owned = [ledger_path(campaign, pair, cell), pair_path(campaign, pair), judge_path(campaign, pair)]
    for stage in ("run", "assemble", "judge"):
        owned += list(raw_paths(campaign, pair, cell, stage))
    for p in owned:
        if os.path.lexists(p):
            errs.append(f"dispatcher output {p} already exists: nothing is overwritten")
    if cell == 2 and first is None and not errs:
        errs.append("cell 2 has no usable first cell")
    if errs:
        raise Refused(errs)
    return {"pair": pair, "cell": cell, "cell_id": cell_id, "arm": ORDER[pair][cell - 1], "manifest": args.manifest, "msha": msha,
            "expected_msha": args.expected_manifest_sha256, "dispatcher_sha": own, "campaign": campaign, "out": out, "stop": stop,
            "cell_json": expected_out_json, "fabric_job": job, "fabric_cards": environ.get("FABRIC_CARDS"), "previous": previous, "first": first,
            "watcher_link": wl, "watcher_previous_target": previous_target, "watcher_target": target, "binding": binding}


# ----------------------------------------------------------------------------------------------------------------------------------
# watcher binding (mutation: only after admission, only inside the held command)
# ----------------------------------------------------------------------------------------------------------------------------------
def bind_watcher(adm, core):
    link, target = adm["watcher_link"], adm["watcher_target"]
    d = os.path.dirname(link)
    tmp = os.path.join(d, f".{os.path.basename(link)}.h2hold{os.getpid()}.tmp")
    os.symlink(target, tmp)                       # FileExistsError on a stale owned temp: fail closed
    try:
        os.replace(tmp, link)                     # atomic repoint
    except BaseException:
        os.unlink(tmp)
        raise
    fsync_dir(d)
    got, real = os.readlink(link), os.path.realpath(link)
    if got != target or real != target:
        raise HoldError(f"watcher link {link} readlink/realpath ({got} / {real}) != {target}")
    st_t, st_l, st_via = os.stat(target), os.lstat(link), os.stat(link)
    if (st_via.st_dev, st_via.st_ino) != (st_t.st_dev, st_t.st_ino):
        raise HoldError("watcher link does not resolve to the wrapper log's file identity")
    doc = {"schema": SCHEMA_BINDING, "dispatcher_version": DISPATCHER_VERSION, "utc": utc_now(), "fabric_job": adm["fabric_job"],
           "fabric_cards": adm["fabric_cards"], "pair": adm["pair"], "cell": adm["cell"], "cell_id": adm["cell_id"], "manifest_sha256": adm["msha"],
           "dispatcher_sha256": adm["dispatcher_sha"], "out": adm["out"], "watcher_link": link,
           "previous_target": adm["watcher_previous_target"], "target": target, "readlink": got, "realpath": real,
           "target_stat": {"device": st_t.st_dev, "inode": st_t.st_ino, "size_bytes": st_t.st_size, "mtime_ns": st_t.st_mtime_ns},
           "link_inode": st_l.st_ino}
    core.write_json_atomic(adm["binding"], doc)


# ----------------------------------------------------------------------------------------------------------------------------------
# runner (the only subprocess seam)
# ----------------------------------------------------------------------------------------------------------------------------------
def run_subprocess(argv):
    """Production runner: ONE plain child in THIS process group (no new session), environment inherited unchanged. Signals received here are
    forwarded to the OWN child only and the child is always awaited. Returns (returncode, stdout bytes, stderr bytes)."""
    holder, pending, old = [], [], {}

    def handler(signum, frame):
        pending.append(signum)
        for proc in holder:
            if proc.poll() is None:
                try:
                    proc.send_signal(signum)
                except ProcessLookupError:
                    pass

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old[sig] = signal.signal(sig, handler)
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        holder.append(proc)
        for signum in list(pending):               # a signal that landed between handler install and Popen return
            proc.send_signal(signum)
        out, err = proc.communicate()
        return proc.returncode, out, err
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def core_argv(sub, *rest):
    return [sys.executable, core_path(), sub] + list(rest)


def run_argv(adm):
    argv = core_argv("run", "--manifest", adm["manifest"], "--expected-manifest-sha256", adm["expected_msha"], "--pair", str(adm["pair"]),
                     "--cell", str(adm["cell"]), "--out", adm["out"])
    if adm["previous"]:
        argv += ["--previous"] + adm["previous"]
    if adm["cell"] == 2:
        argv += ["--first-cell", adm["first"]["path"], "--first-cell-sha256", adm["first"]["sha256"]]
    return argv


# ----------------------------------------------------------------------------------------------------------------------------------
# results / STOP
# ----------------------------------------------------------------------------------------------------------------------------------
def write_stop(core, adm, stage, reason, **extra):
    doc = {"schema": SCHEMA_STOP, "dispatcher_version": DISPATCHER_VERSION, "utc": utc_now(), "stage": stage, "reason": reason, "pair": adm["pair"],
           "cell": adm["cell"], "cell_id": adm["cell_id"], "fabric_job": adm["fabric_job"], "manifest_sha256": adm["msha"],
           "dispatcher_sha256": adm["dispatcher_sha"], "out": adm["out"], "campaign": adm["campaign"], "no_automatic_retry": True}
    doc.update(extra)
    try:
        core.write_json_atomic(adm["stop"], doc)
    except FileExistsError:
        pass                                      # already stopped: the first STOP is the record
    return doc


def fail(core, adm, stage, reason, code, **extra):
    doc = write_stop(core, adm, stage, reason, exit_code=code, **extra)
    emit({"status": "STOPPED", "stage": stage, "reason": reason, "stop": adm["stop"], "pair": adm["pair"], "cell": adm["cell"], "exit_code": code})
    return code


def save_raw(core, adm, stage, stdout, stderr):
    so, se = raw_paths(adm["campaign"], adm["pair"], adm["cell"], stage)
    core.write_atomic(so, bytes(stdout or b""))
    core.write_atomic(se, bytes(stderr or b""))
    return {"stdout": so, "stderr": se}


def parse_object(core, raw):
    try:
        doc = core.loads_strict(bytes(raw or b""))
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def receipt_info(path):
    info = {"cell_json": path, "present": os.path.lexists(path)}
    if info["present"]:
        try:
            info["sha256"] = sha256_file(path)
        except OSError as e:
            info["sha256_error"] = str(e)
    return info


def write_ledger(core, adm, stdout_doc, native_sha):
    doc = {"schema": SCHEMA_LEDGER, "dispatcher_version": DISPATCHER_VERSION, "utc": utc_now(), "pair": adm["pair"], "cell": adm["cell"],
           "cell_id": adm["cell_id"], "out": adm["out"], "cell_json": adm["cell_json"], "cell_file_sha256": native_sha,
           "receipt_sha256": stdout_doc.get("receipt_sha256"), "fabric_job": adm["fabric_job"], "manifest_sha256": adm["msha"],
           "dispatcher_sha256": adm["dispatcher_sha"], "watcher_binding": adm["binding"]}
    core.write_json_atomic(ledger_path(adm["campaign"], adm["pair"], adm["cell"]), doc)


# ----------------------------------------------------------------------------------------------------------------------------------
# execution of an admitted hold
# ----------------------------------------------------------------------------------------------------------------------------------
def execute(adm, core, runner):
    pair, cell = adm["pair"], adm["cell"]
    try:
        bind_watcher(adm, core)
    except (OSError, HoldError) as e:
        return fail(core, adm, "watcher_binding", f"{type(e).__name__}: {e}", EXIT_STOPPED)

    # --- exactly one fresh core `run` ---
    argv = run_argv(adm)
    rc, so, se = runner(argv)
    raw = save_raw(core, adm, "run", so, se)
    doc = parse_object(core, so)
    evidence = {"argv": argv, "core_returncode": rc, "raw": raw, "receipt": receipt_info(adm["cell_json"])}
    code = rc if rc in (3, 4) else EXIT_STOPPED
    if rc != 0 or doc is None:
        return fail(core, adm, "run", f"core run returned {rc}" + ("" if doc is not None else " with non-JSON stdout"), code, **evidence)
    bad = []
    if doc.get("status") != "CELL_OK" or doc.get("cell_status") != "OK":
        bad.append(f"core reported {doc.get('status')}/{doc.get('cell_status')}")
    if doc.get("pair") != pair or doc.get("cell") != cell or doc.get("cell_id") != adm["cell_id"]:
        bad.append("core reported a different pair/cell")
    if doc.get("cell_json") != adm["cell_json"]:
        bad.append(f"core cell_json {doc.get('cell_json')!r} != {adm['cell_json']}")
    if doc.get("fabric_job") != adm["fabric_job"]:
        bad.append(f"core fabric_job {doc.get('fabric_job')!r} != {adm['fabric_job']!r}")
    native = None
    try:
        if not statmod.S_ISREG(os.lstat(adm["cell_json"]).st_mode):
            bad.append("cell.json is not a regular file")
        else:
            native = sha256_file(adm["cell_json"])
    except OSError as e:
        bad.append(f"cell.json unreadable: {e}")
    if native is not None and native != doc.get("cell_file_sha256"):
        bad.append(f"native cell.json sha256 {native} != core cell_file_sha256 {doc.get('cell_file_sha256')!r}")
    if bad:
        return fail(core, adm, "run", "; ".join(bad), EXIT_STOPPED, **evidence)
    write_ledger(core, adm, doc, native)
    if cell == 1:
        emit({"status": "CELL1_OK", "pair": pair, "cell": 1, "cell_id": adm["cell_id"], "cell_json": adm["cell_json"], "cell_file_sha256": native,
              "fabric_job": adm["fabric_job"]})
        return EXIT_OK

    # --- cell 2: pure-CPU assemble-pair, then judge ---
    first = adm["first"]
    try:
        first_now = sha256_file(first["path"])
    except OSError as e:
        return fail(core, adm, "assemble", f"first cell file became unreadable: {e}", EXIT_STOPPED, **evidence)
    if first_now != first["sha256"]:
        return fail(core, adm, "assemble", f"first cell file changed since admission ({first_now} != {first['sha256']})", EXIT_STOPPED, **evidence)
    pair_file = pair_path(adm["campaign"], pair)
    aargv = core_argv("assemble-pair", "--manifest", adm["manifest"], "--expected-manifest-sha256", adm["expected_msha"],
                      "--cells", first["path"], adm["cell_json"], "--cell-sha256", first["sha256"], native)
    if adm["previous"]:
        aargv += ["--previous"] + adm["previous"]
    aargv += ["--out", pair_file]
    arc, aso, ase = runner(aargv)
    araw = save_raw(core, adm, "assemble", aso, ase)
    adoc = parse_object(core, aso)
    aev = dict(evidence, assemble_argv=aargv, assemble_returncode=arc, assemble_raw=araw)
    if arc != 0 or adoc is None or adoc.get("status") != "PAIR_COMPLETE" or adoc.get("pair") != pair or adoc.get("pair_json") != pair_file:
        return fail(core, adm, "assemble", f"assemble-pair returned {arc} / {None if adoc is None else adoc.get('status')}", EXIT_STOPPED, **aev)
    if not os.path.isfile(pair_file) or os.path.islink(pair_file):
        return fail(core, adm, "assemble", f"pair receipt {pair_file} missing after assemble-pair", EXIT_STOPPED, **aev)

    jfile = judge_path(adm["campaign"], pair)
    jargv = core_argv("judge", "--manifest", adm["manifest"], "--expected-manifest-sha256", adm["expected_msha"],
                      "--pairs", *(adm["previous"] + [pair_file]), "--out", jfile)
    jrc, jso, jse = runner(jargv)
    jraw = save_raw(core, adm, "judge", jso, jse)
    jev = dict(aev, judge_argv=jargv, judge_returncode=jrc, judge_raw=jraw, judge_result=jfile)
    errs = []
    result = read_json_file(core, jfile, "judge result", errs)
    if result is None:
        return fail(core, adm, "judge", "judge result unreadable: " + " | ".join(errs), EXIT_STOPPED, **jev)
    verdict, jerrors = result.get("verdict"), result.get("errors")
    if verdict in JUDGE_FAIL_VERDICTS:
        return fail(core, adm, "judge", f"judge verdict {verdict}: no later GPU hold may run", EXIT_JUDGE_FAIL, verdict=verdict, **jev)
    if JUDGE_RC.get(verdict) is None or jerrors != [] or jrc != JUDGE_RC[verdict]:
        return fail(core, adm, "judge", f"judge outcome not accepted: verdict {verdict!r}, errors {jerrors!r}, return {jrc}", EXIT_STOPPED,
                    verdict=verdict, **jev)
    if (verdict == "PASS") != (pair == 3):
        return fail(core, adm, "judge", f"judge verdict {verdict} is not valid after pair {pair}", EXIT_STOPPED, verdict=verdict, **jev)
    emit({"status": "HOLD_COMPLETE", "pair": pair, "cell": 2, "verdict": verdict, "pair_json": pair_file, "judge_result": jfile,
          "fabric_job": adm["fabric_job"]})
    return EXIT_OK


# ----------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------
def _sha256_arg(text):
    if not (len(text) == 64 and all(c in "0123456789abcdef" for c in text)):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex characters")
    return text


def build_parser():
    p = argparse.ArgumentParser(description="dispatcher for ONE official H2 hold (one cell, one fresh core run)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--expected-manifest-sha256", required=True, type=_sha256_arg)
    p.add_argument("--expected-dispatcher-sha256", required=True, type=_sha256_arg)
    p.add_argument("--pair", required=True, type=int, choices=(1, 2, 3))
    p.add_argument("--cell", required=True, type=int, choices=(1, 2))
    p.add_argument("--out", required=True)
    p.add_argument("--watcher-prefix", required=True)
    p.add_argument("--campaign", required=True)
    return p


def dispatch(argv=None, runner=None, core=None, environ=None):
    args = build_parser().parse_args(argv)
    core = core or load_core()
    environ = os.environ if environ is None else environ
    runner = runner or run_subprocess
    try:
        adm = admit(args, core, environ)
    except Refused as e:
        emit({"status": "REFUSED", "errors": e.errors, "mutated": False})
        return EXIT_REFUSED
    try:
        return execute(adm, core, runner)
    except Exception as e:  # noqa: BLE001 -- any failure after admission closes the campaign
        return fail(core, adm, "harness_exception", f"{type(e).__name__}: {e}", EXIT_STOPPED, receipt=receipt_info(adm["cell_json"]))


def main(argv=None):
    return dispatch(argv)


if __name__ == "__main__":
    sys.exit(main())
