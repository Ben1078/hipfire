#!/usr/bin/env python3
"""Pinned host->device copy VRAM-residue diagnostic (standalone: stdlib + ctypes).

Where does the free-VRAM drop around the first host-link copy come from
(one-time copy-path state, per-stream, per-size, or event creation)?  Only
pinned H2D copies are issued, hipMemGetInfo is read after every step, nothing
is timed, and nothing is asserted: no threshold, arch gate or assumed constant.
The final summary only reports actual baseline/post free-memory deltas.

HIP ABIs / flags mirror crates/hip-bridge/src/ffi.rs + lib.rs:
  hipHostMalloc(flags=1 portable) CPU-zeroed; raw hipMalloc/hipFree (no pool);
  hipStreamCreateWithFlags(0x1 nonblocking); hipMemcpyAsync / hipMemcpy kind 1
  (HostToDevice); hipEventCreate / hipEventCreateWithFlags(0x2 disable timing) /
  hipEventRecord / hipEventSynchronize / hipEventDestroy (untimed markers, no
  hipEventElapsedTime); hipMemGetInfo; hipDeviceGetPCIBusId;
  hipRuntimeGetVersion.  GPU_PINNED_MIN_XFER_SIZE=100000 is armed like
  HipRuntime::load unless preset or HIPFIRE_PINNED_XFER_DEFAULT=0.
  HSA_USERPTR_FOR_PAGED_MEM is never forced.  No pageable copies.

Parent (never loads HIP) runs 4 FRESH sequential children ("arms"), each with a
different FIRST copy size so a small-copy/runtime-init threshold effect is not
confounded with size order:
  arm1 async nonblocking, 64 MiB first, then 4 KiB (SDMA env inherited)
  arm2 async nonblocking, 4 KiB first, then 64 MiB (SDMA env inherited)
  arm3 sync hipMemcpy (no stream), 64 MiB first, then 4 KiB (inherited)
  arm4 async nonblocking, 64 MiB first, then 4 KiB, HSA_ENABLE_SDMA=1 set
       before dlopen
Each child after hipInit/hipSetDevice: init(baseline, identity, baseline_recheck),
s1_first_<size> (hipHostMalloc + CPU zero, hipMalloc, stream[async], H2D, sync,
cleanup), events (fresh nonblocking stream; per default/disable_timing pair
create, hipEventRecord, hipEventSynchronize, destroy, sampled each; untimed; AFTER the first
no-event session), s2_repeat_<size> (same first size, fresh stream/buffers),
s3_other_<size>.  JSONL on stdout; exit 0 only if all four arms completed.
Any HIP error/timeout/nonzero child exit stops the remaining arms.  Bounds:
child <= 10 s, whole protocol < 55 s; VRAM < 1 GiB, host < 1 GiB.

Run: python3 pinned_copy_residue.py [--device N]
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

PROTOCOL = "pinned_copy_residue/2"

TOTAL_BUDGET_S = 55.0
CHILD_TIMEOUT_S = 10.0
KILL_RESERVE_S = 3.5        # TERM grace + KILL reap + pipe drain + slack
MIN_CHILD_TIMEOUT_S = 3.0   # fail closed instead of starting a mode that cannot finish
GRACE_S = 1.0

HOST_MALLOC_PORTABLE = 1
STREAM_NON_BLOCKING = 1
EVENT_DISABLE_TIMING = 0x2
MEMCPY_H2D = 1
PINNED_XFER_DEFAULT_MB = "100000"
KIB = 1024
MIB = 1024 * 1024

MODES = [
    {"name": "arm1_async_nb_64m_first_inherited_sdma", "copy": "async", "sdma": None, "first": 64 * MIB, "other": 4 * KIB},
    {"name": "arm2_async_nb_4k_first_inherited_sdma", "copy": "async", "sdma": None, "first": 4 * KIB, "other": 64 * MIB},
    {"name": "arm3_sync_hipMemcpy_64m_first_inherited_sdma", "copy": "sync", "sdma": None, "first": 64 * MIB, "other": 4 * KIB},
    {"name": "arm4_async_nb_64m_first_hsa_enable_sdma_1", "copy": "async", "sdma": "1", "first": 64 * MIB, "other": 4 * KIB},
]
MODE_BY_NAME = {m["name"]: m for m in MODES}

EXIT_HIP_ERROR, EXIT_QUARANTINE, EXIT_EXCEPTION, EXIT_SETUP = 3, 4, 5, 6

# The ONLY environment variables ever recorded (null = unset).  No prefix sweep.
ENV_ALLOWLIST = (
    "HSA_ENABLE_SDMA", "HSA_USERPTR_FOR_PAGED_MEM", "GPU_PINNED_MIN_XFER_SIZE",
    "HIPFIRE_PINNED_XFER_DEFAULT", "HIPFIRE_QWEN4_D2_DEFAULT",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
    "HSA_XNACK", "HIP_HOST_COHERENT", "ROCM_PATH", "HIPFIRE_ROCM_PATH", "LD_LIBRARY_PATH",
)


def emit(record):
    sys.stdout.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def allowlisted_env(env):
    return {k: env.get(k) for k in ENV_ALLOWLIST}


# =========================================================================
# Child: the only code that loads HIP.
# =========================================================================

class HipFailure(Exception):
    def __init__(self, call, rc, text):
        super().__init__("%s rc=%s %s" % (call, rc, text))
        self.call, self.rc, self.text = call, rc, text
        self.emitted = False


class QuarantineAbort(Exception):
    """Async work could not be proven drained; pointers are retained, never freed."""


QUARANTINE = []  # pointers retained until process end


class Hip:
    """Checked ctypes wrapper over libamdhip64; every call's hipError_t is checked."""

    SIGS = {
        "hipInit": [ctypes.c_uint],
        "hipSetDevice": [ctypes.c_int],
        "hipMemGetInfo": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
        "hipHostMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        "hipHostFree": [ctypes.c_void_p],
        "hipMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "hipFree": [ctypes.c_void_p],
        "hipMemcpyAsync": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p],
        "hipMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint],
        "hipStreamCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
        "hipStreamSynchronize": [ctypes.c_void_p],
        "hipStreamDestroy": [ctypes.c_void_p],
        "hipEventCreate": [ctypes.POINTER(ctypes.c_void_p)],
        "hipEventCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
        "hipEventDestroy": [ctypes.c_void_p],
        "hipEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
        "hipEventSynchronize": [ctypes.c_void_p],
        "hipRuntimeGetVersion": [ctypes.POINTER(ctypes.c_int)],
        "hipDeviceGetPCIBusId": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
        "hipGetErrorString": [ctypes.c_int],
    }

    def __init__(self, library):
        self.lib = ctypes.CDLL(library)  # kept for the object's lifetime; OSError carries the dlerror text
        self.fn = {}
        for name, argtypes in self.SIGS.items():
            func = getattr(self.lib, name)  # AttributeError names a missing symbol
            func.argtypes = argtypes
            func.restype = ctypes.c_char_p if name == "hipGetErrorString" else ctypes.c_int
            self.fn[name] = func

    def call(self, name, *args):
        rc = self.fn[name](*args)
        if rc != 0:
            text = self.fn["hipGetErrorString"](rc)
            raise HipFailure(name, rc, text.decode("utf-8", "replace") if text else "")

    def out_handle(self, name, *args):
        out = ctypes.c_void_p(None)
        self.call(name, ctypes.byref(out), *args)
        if out.value is None:
            raise HipFailure(name, -1, "success with NULL handle")
        return out.value

    def mem_info(self):
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        self.call("hipMemGetInfo", ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value


class Recorder:
    """One JSONL phase record per hipMemGetInfo sample (actual free/total, no timing)."""

    def __init__(self, hip, mode):
        self.hip, self.mode = hip, mode
        self.seq = 0
        self.baseline = None
        self.prev = None

    def sample(self, session, phase, checkpoint=False, **extra):
        free, total = self.hip.mem_info()
        if session == "init" and phase == "baseline":
            self.baseline = free
        record = {
            "event": "phase", "mode": self.mode, "seq": self.seq, "session": session, "phase": phase,
            "checkpoint": checkpoint, "free_bytes": free, "total_bytes": total,
            # positive = free VRAM went DOWN relative to the reference sample
            "consumed_vs_baseline_bytes": None if self.baseline is None else self.baseline - free,
            "consumed_vs_prev_bytes": None if self.prev is None else self.prev - free,
        }
        record.update(extra)
        self.seq += 1
        self.prev = free
        emit(record)


def emit_hip_error(mode, session, phase, exc):
    emit({"event": "hip_error", "mode": mode, "session": session, "phase": phase,
          "call": exc.call, "rc": exc.rc, "hip_error_string": exc.text})
    exc.emitted = True


def quarantine(mode, session, why, pointers):
    """Retain pointers until process end; NEVER free memory a copy may still use."""
    QUARANTINE.extend((kind, ptr) for kind, ptr in pointers if ptr is not None)
    emit({"event": "quarantine", "mode": mode, "session": session, "reason": why,
          "retained": [{"kind": k, "pointer": p} for k, p in QUARANTINE]})
    raise QuarantineAbort(why)


def release(hip, mode, session, host, dev, stream, pending, sync_failed):
    """Drain before free.  Raises QuarantineAbort if undrained, else HipFailure on the first free error."""
    pointers = (("host_pinned", host), ("device", dev), ("stream", stream))
    if pending:
        if sync_failed:
            quarantine(mode, session, "hipStreamSynchronize failed; copy may be in flight", pointers)
        try:
            hip.call("hipStreamSynchronize", stream)
        except HipFailure as exc:
            emit_hip_error(mode, session, "cleanup_sync", exc)
            quarantine(mode, session, "cleanup hipStreamSynchronize failed; copy may be in flight", pointers)
    first = None
    for call, value in (("hipStreamDestroy", stream), ("hipFree", dev), ("hipHostFree", host)):
        if value is None:
            continue
        try:
            hip.call(call, value)
        except HipFailure as exc:
            emit_hip_error(mode, session, "cleanup_" + call, exc)
            first = first or exc
    if first:
        raise first


def run_session(hip, rec, cfg, session, size):
    host = dev = stream = None
    pending = False       # an async copy may be enqueued and undrained
    sync_failed = False
    copy_unproven = False  # synchronous hipMemcpy returned an error
    primary = None
    try:
        host = hip.out_handle("hipHostMalloc", size, HOST_MALLOC_PORTABLE)
        ctypes.memset(host, 0, size)
        rec.sample(session, "host_malloc_zero", size_bytes=size)
        dev = hip.out_handle("hipMalloc", size)
        rec.sample(session, "device_malloc", size_bytes=size)
        if cfg["copy"] == "async":
            stream = hip.out_handle("hipStreamCreateWithFlags", STREAM_NON_BLOCKING)
            rec.sample(session, "stream_create_nonblocking")
            pending = True  # set before the call: a failing enqueue is not assumed to be a no-op
            hip.call("hipMemcpyAsync", dev, host, size, MEMCPY_H2D, stream)
            rec.sample(session, "h2d_enqueued", op="hipMemcpyAsync", size_bytes=size)
            try:
                hip.call("hipStreamSynchronize", stream)
            except HipFailure:
                sync_failed = True
                raise
            pending = False
            rec.sample(session, "stream_sync")
        else:
            try:
                hip.call("hipMemcpy", dev, host, size, MEMCPY_H2D)
            except HipFailure:
                copy_unproven = True
                raise
            rec.sample(session, "h2d_sync_returned", op="hipMemcpy", size_bytes=size)
    except HipFailure as exc:
        primary = exc
        emit_hip_error(rec.mode, session, "session_body", exc)
    if copy_unproven:  # a failed hipMemcpy cannot prove a partial transfer is inactive: never free
        quarantine(rec.mode, session, "hipMemcpy failed; partial transfer cannot be proven inactive",
                   (("host_pinned", host), ("device", dev), ("stream", stream)))
    try:
        release(hip, rec.mode, session, host, dev, stream, pending, sync_failed)
    except QuarantineAbort:
        raise
    except HipFailure as exc:
        primary = primary or exc
    if primary:
        raise primary
    rec.sample(session, "cleanup", checkpoint=True, size_bytes=size)


def run_events(hip, rec):
    """Event objects on a fresh nonblocking stream, with memInfo samples; untimed markers only.

    Per default / disable_timing pair: create+sample each, hipEventRecord+sample each,
    hipEventSynchronize+sample each, destroy+sample; then destroy the stream+sample.
    hipEventElapsedTime is never called.  Once a record is issued and its
    synchronize has not succeeded, completion cannot be proven: events and stream
    are quarantined (retained, never destroyed) and the child exits.
    """
    session = "events"
    stream = None
    live = []        # created, not yet destroyed
    unproven = []    # record issued, hipEventSynchronize not yet successful
    primary = None
    try:
        rec.sample(session, "event_baseline")
        stream = hip.out_handle("hipStreamCreateWithFlags", STREAM_NON_BLOCKING)
        rec.sample(session, "event_stream_create_nonblocking")
        for label, flags in (("default", None), ("disable_timing", EVENT_DISABLE_TIMING)):
            pair = []
            for index in (1, 2):
                if flags is None:
                    event = hip.out_handle("hipEventCreate")
                else:
                    event = hip.out_handle("hipEventCreateWithFlags", flags)
                pair.append(event)
                live.append(event)
                rec.sample(session, "event_create_%s_%d" % (label, index))
            for index, event in enumerate(pair, 1):
                unproven.append(event)  # before the call: a failing record is not assumed inert
                hip.call("hipEventRecord", event, stream)
                rec.sample(session, "event_record_%s_%d" % (label, index))
                hip.call("hipEventSynchronize", event)
                unproven.remove(event)
                rec.sample(session, "event_sync_%s_%d" % (label, index))
            while pair:
                event = pair.pop()
                hip.call("hipEventDestroy", event)
                live.remove(event)
            rec.sample(session, "event_destroy_%s_pair" % label, checkpoint=True)
        hip.call("hipStreamDestroy", stream)
        stream = None
        rec.sample(session, "event_stream_destroy", checkpoint=True)
    except HipFailure as exc:
        primary = exc
        emit_hip_error(rec.mode, session, "events_body", exc)
    if unproven:
        quarantine(rec.mode, session, "recorded event not proven complete; events and stream retained",
                   [("event", e) for e in live] + [("stream", stream)])
    for event in list(live):  # all recorded events were synchronized, so destroying leftovers is safe
        try:
            hip.call("hipEventDestroy", event)
        except HipFailure as exc:
            emit_hip_error(rec.mode, session, "cleanup_hipEventDestroy", exc)
            primary = primary or exc
    if stream is not None:
        try:
            hip.call("hipStreamDestroy", stream)
        except HipFailure as exc:
            emit_hip_error(rec.mode, session, "cleanup_hipStreamDestroy", exc)
            primary = primary or exc
    if primary:
        raise primary


def arm_parent_death_signal(mode, parent_pid):
    """PR_SET_PDEATHSIG=SIGKILL so a SIGKILLed parent cannot orphan this process group; failure is fatal."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, "prctl(PR_SET_PDEATHSIG) failed: " + os.strerror(code))
        if parent_pid and os.getppid() != parent_pid:
            raise OSError(0, "parent already gone (getppid=%d expected=%d)" % (os.getppid(), parent_pid))
    except (OSError, AttributeError) as exc:
        emit({"event": "setup_error", "mode": mode, "error": "pdeathsig: %s" % exc,
              "errno": getattr(exc, "errno", None)})
        os._exit(EXIT_SETUP)


def child_main(mode, device, parent_pid):
    cfg = MODE_BY_NAME[mode]
    arm_parent_death_signal(mode, parent_pid)

    if cfg["sdma"] is not None and os.environ.get("HSA_ENABLE_SDMA") != cfg["sdma"]:
        emit({"event": "setup_error", "mode": mode, "error": "HSA_ENABLE_SDMA must be set before dlopen"})
        os._exit(EXIT_SETUP)
    library = ctypes.util.find_library("amdhip64")
    if library is None:
        emit({"event": "setup_error", "mode": mode, "error": "ctypes.util.find_library('amdhip64') returned None"})
        os._exit(EXIT_SETUP)
    emit({"event": "child_start", "mode": mode, "pid": os.getpid(), "copy": cfg["copy"],
          "device_index": device, "library": library, "env": allowlisted_env(os.environ)})
    try:
        hip = Hip(library)  # dlopen happens here, after the env is final
    except (OSError, AttributeError) as exc:
        emit({"event": "setup_error", "mode": mode, "library": library, "error": str(exc)})
        os._exit(EXIT_SETUP)

    rec = Recorder(hip, mode)
    session = "init"
    try:
        hip.call("hipInit", 0)
        hip.call("hipSetDevice", device)
        rec.sample("init", "baseline", checkpoint=True)
        # Identity AFTER the baseline so it cannot perturb it; baseline_recheck shows any effect.
        buf = ctypes.create_string_buffer(64)
        hip.call("hipDeviceGetPCIBusId", buf, 64, device)
        version = ctypes.c_int(0)
        hip.call("hipRuntimeGetVersion", ctypes.byref(version))
        emit({"event": "identity", "mode": mode, "device_index": device,
              "pci_bus_id": buf.value.decode("ascii", "replace").lower(), "hip_runtime_version_raw": version.value})
        rec.sample("init", "baseline_recheck", checkpoint=True)
        label = {4 * KIB: "4k", 64 * MIB: "64m"}
        for session, size in (("s1_first_" + label[cfg["first"]], cfg["first"]), ("events", None),
                              ("s2_repeat_" + label[cfg["first"]], cfg["first"]),
                              ("s3_other_" + label[cfg["other"]], cfg["other"])):
            if size is None:
                run_events(hip, rec)
            else:
                run_session(hip, rec, cfg, session, size)
        emit({"event": "child_done", "mode": mode, "ok": True})
        return 0
    except QuarantineAbort:
        sys.stdout.flush()
        os._exit(EXIT_QUARANTINE)  # skip library teardown: retained pointers stay untouched
    except HipFailure as exc:
        if not exc.emitted:
            emit_hip_error(mode, session, "child", exc)
        emit({"event": "child_done", "mode": mode, "ok": False, "call": exc.call, "rc": exc.rc})
        sys.stdout.flush()
        os._exit(EXIT_HIP_ERROR)
    except Exception:  # noqa: BLE001 - keep the actual error for the parent
        emit({"event": "child_exception", "mode": mode, "session": session, "traceback": traceback.format_exc()})
        sys.stdout.flush()
        os._exit(EXIT_EXCEPTION)


# =========================================================================
# Parent: never loads HIP; spawns bounded fresh children.
# =========================================================================

class ParentInterrupted(Exception):
    def __init__(self, signum):
        super().__init__("signal %d" % signum)
        self.signum = signum


def _raise_interrupt(signum, _frame):
    raise ParentInterrupted(signum)


def kill_group(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def terminate_group(proc):
    """Group TERM, grace, group SIGKILL (always, even if the leader exited), reap.  True if reaped."""
    kill_group(proc, signal.SIGTERM)
    reaped = False
    try:
        proc.wait(timeout=GRACE_S)
        reaped = True
    except subprocess.TimeoutExpired:
        pass
    kill_group(proc, signal.SIGKILL)  # stragglers keep the group alive after the leader is gone
    if not reaped:
        try:
            proc.wait(timeout=GRACE_S)
            reaped = True
        except subprocess.TimeoutExpired:
            pass
    return reaped


def child_env(cfg):
    env = dict(os.environ)
    if "GPU_PINNED_MIN_XFER_SIZE" in env:
        note = "preset"
    elif env.get("HIPFIRE_PINNED_XFER_DEFAULT") == "0":
        note = "unset (HIPFIRE_PINNED_XFER_DEFAULT=0)"
    else:
        env["GPU_PINNED_MIN_XFER_SIZE"] = PINNED_XFER_DEFAULT_MB  # as HipRuntime::load does
        note = "armed %s like HipRuntime::load" % PINNED_XFER_DEFAULT_MB
    if cfg["sdma"] is not None:
        env["HSA_ENABLE_SDMA"] = cfg["sdma"]
    # HSA_USERPTR_FOR_PAGED_MEM is intentionally untouched.
    return env, note


def run_child(cfg, device, timeout_s, parent_pid):
    env, xfer_note = child_env(cfg)
    argv = [sys.executable, "-u", os.path.abspath(__file__), "--child", cfg["name"],
            "--device", str(device), "--parent-pid", str(parent_pid)]
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            close_fds=True, start_new_session=True, env=env)
    timed_out = unreaped = False
    out = err = b""
    try:
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as first:
            timed_out = True
            unreaped = not terminate_group(proc)
            out, err = first.stdout or b"", first.stderr or b""
            try:
                out, err = proc.communicate(timeout=GRACE_S)
            except subprocess.TimeoutExpired as second:
                out, err = second.stdout or out, second.stderr or err
    except ParentInterrupted:
        terminate_group(proc)
        try:
            out, err = proc.communicate(timeout=GRACE_S)
        except subprocess.TimeoutExpired as drain:
            out, err = drain.stdout or b"", drain.stderr or b""
        emit({"event": "interrupted_child_partial", "mode": cfg["name"],
              "stdout": out.decode("utf-8", "replace"), "stderr": err.decode("utf-8", "replace")[-8000:]})
        raise
    records, unparsed = [], []
    for line in out.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        (records if isinstance(obj, dict) else unparsed).append(obj if isinstance(obj, dict) else line)
    return {"mode": cfg["name"], "returncode": proc.returncode, "timed_out": timed_out, "unreaped": unreaped,
            "timeout_s": timeout_s, "xfer_note": xfer_note, "records": records, "unparsed": unparsed,
            "stderr": err.decode("utf-8", "replace")}


ERROR_EVENTS = ("hip_error", "quarantine", "child_exception", "setup_error")


def summarize(child):
    records = child["records"]
    phases = [r for r in records if r.get("event") == "phase"]
    errors = [r for r in records if r.get("event") in ERROR_EVENTS]
    done = next((r for r in records if r.get("event") == "child_done"), None)
    out = {
        "mode": child["mode"], "returncode": child["returncode"], "timed_out": child["timed_out"],
        "completed": bool(done and done.get("ok") is True and child["returncode"] == 0
                          and not child["timed_out"] and not errors),
        "phases_recorded": len(phases), "errors": errors, "gpu_pinned_min_xfer_size": child["xfer_note"],
    }
    base = next((r for r in phases if r["session"] == "init" and r["phase"] == "baseline"), None)
    if base is None:
        return out
    final = phases[-1]
    lowest = min(r["free_bytes"] for r in phases)
    out.update({
        "baseline_free_bytes": base["free_bytes"],
        "final_free_bytes": final["free_bytes"],
        "final_phase": "%s/%s" % (final["session"], final["phase"]),
        "final_minus_baseline_consumed_bytes": base["free_bytes"] - final["free_bytes"],
        "min_free_bytes": lowest,
        "peak_consumed_vs_baseline_bytes": base["free_bytes"] - lowest,
        "checkpoints": [{"phase": "%s/%s" % (r["session"], r["phase"]), "free_bytes": r["free_bytes"],
                         "consumed_vs_baseline_bytes": r["consumed_vs_baseline_bytes"]}
                        for r in phases if r["checkpoint"]],
    })
    return out


def parent_main(device):
    t0 = time.monotonic()
    deadline = t0 + TOTAL_BUDGET_S
    pid = os.getpid()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)
    emit({"event": "parent_start", "protocol": PROTOCOL, "pid": pid, "device_index": device,
          "parent_loads_hip": False, "modes": [m["name"] for m in MODES],
          "budget_s": TOTAL_BUDGET_S, "child_timeout_s": CHILD_TIMEOUT_S, "env": allowlisted_env(os.environ)})
    summaries, stopped, code = [], None, 0
    try:
        for cfg in MODES:
            remaining = deadline - time.monotonic()
            timeout_s = min(CHILD_TIMEOUT_S, remaining - KILL_RESERVE_S)
            if timeout_s < MIN_CHILD_TIMEOUT_S:
                stopped, code = "budget exhausted before %s (%.1fs left)" % (cfg["name"], remaining), 2
                break
            child = run_child(cfg, device, round(timeout_s, 2), pid)
            for record in child["records"]:
                emit(record)
            for line in child["unparsed"]:
                emit({"event": "unparsed_child_stdout", "mode": cfg["name"], "line": line})
            emit({"event": "child_exit", "mode": cfg["name"], "returncode": child["returncode"],
                  "timed_out": child["timed_out"], "unreaped": child["unreaped"], "timeout_s": child["timeout_s"],
                  "stderr": child["stderr"][-8000:]})
            summary = summarize(child)
            summaries.append(summary)
            if not summary["completed"]:
                stopped = "mode %s did not complete; remaining modes not run" % cfg["name"]
                code = 2 if child["timed_out"] or child["unreaped"] else 1
                break
    except ParentInterrupted as exc:
        stopped, code = "parent interrupted by signal %d; child group terminated" % exc.signum, 128 + exc.signum

    complete = stopped is None and len(summaries) == len(MODES)
    run = [s["mode"] for s in summaries]
    emit({"event": "summary", "protocol": PROTOCOL, "status": "complete" if complete else "incomplete",
          "stopped_reason": stopped, "modes_run": run,
          "modes_not_run": [m["name"] for m in MODES if m["name"] not in run],
          "wall_s_total": round(time.monotonic() - t0, 3), "modes": summaries,
          "note": "consumed = reference_free - free (positive = less free VRAM); actual hipMemGetInfo values only"})
    return 0 if complete else code


def main():
    parser = argparse.ArgumentParser(description="Pinned H2D copy-path VRAM residue diagnostic (run once, <=60s hold).")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--child", choices=sorted(MODE_BY_NAME), help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child_main(args.child, args.device, args.parent_pid)
    return parent_main(args.device)


if __name__ == "__main__":
    sys.exit(main())
