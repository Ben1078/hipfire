#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""h1_handoff.py seal STAGE A_OUT | verify STAGE A_PATH | receipt1b STAGE B_OUT A_PATH

CPU-only. Hold 1 is two halves: 1a (prim + state + the immediate required state-headroom row, hold1a-correct.sh) and 1b (a headroom
RECHECK on the sealed 1a RAW state data + beta/S1 greedy, hold1b-correct.sh). 1a PASS is PARTIAL Hold 1, never an advance; 1b may run only on an IMMUTABLE, verified 1a handoff.

seal STAGE A_OUT   (end of 1a, after both cell collectors finished)
    Writes A_OUT/handoff-1a.json (schema s1.hold1a_handoff.v1) atomically, ONLY when prim and state both completed PASS. It binds
      * the staged identity (stage_identity): manifest-hold1.json sha256 and its runtime source base/tip/tree, gate binary sha256, kernel
        code-object hashes, model path/size/sha256, harness commit/tree as declared, the profile ledgers (sha256 + tiles), the manifest-bound
        greedy inputs (canonical + beta file hashes + adapter proof), every file of bin-s1 and bin-beta, every fixture file, target.json,
        ledger-hold1.json and every top-level harness file; and
      * every CLOSED CELL FILE of prim and state by size and sha256: command, stdout/stderr, rc, spawn-env.json (the env pins actually applied),
        evidence.json, the raw gate.json, the progress sidecar (forensic, bound, never used for a verdict), baseline.json, meminfo snapshots,
        mem.csv and the external 20 ms sampler files (vram.csv, vram-events.jsonl, vram-summary.json).
    It never hashes the run root (hold.jsonl, cells.jsonl, hold-summary.json, FATAL.txt: the EXIT trap and finish_summary may append to
    them) and never hashes itself, so there is no self-hash cycle; the handoff file's own sha256 is recorded by the verifier.
    A cell qualifies only from its closed files: rc 0, no status.txt (no lifecycle/sampler problem), a COMPLETED gate report (schema, status
    pass, the expected mode and layer-major route, manifest_sha256 and fixture_sha256 equal to the staged files, BOTH evidence.s1_profile.expected_*
    (built-in intent) AND admitted_* (null until a bundle actually loaded, then its read-back) equal to the manifest's profile: primitive_selftest
    512 for prim, product 768 for state; an expected-only report with null admitted never qualifies; an aborted/setup-failed `incomplete` report
    has no evidence keys and is rejected without touching them; the `checks` block {status, required, missing, checks[{name,status,detail}]} must
    be self-consistent (status recomputed from the rows, every required check passed, none failed) and carry exactly one PASS `load.placement`
    row whose detail is the load read-back (placement slowlink-profile, S1 profile admitted, >= 10 resident layers, expected_*/admitted_*
    equal to the sealed profile) even when a forged root status says pass), the H1 env pin HIPFIRE_OFFLOAD_EXEC=pcie in
    spawn-env.json, and an evidence.json showing a completed report. A partial progress sidecar or a perfect-looking sampler can never qualify
    a cell.

verify STAGE A_PATH   (start of 1b, before ANY GPU cell)
    A_PATH is the 1a run directory or its handoff-1a.json (S1_HOLD1A_OUT or the CLI argument). Recomputes the staged identity and requires it
    equal to the sealed one (same runtime/tree/binaries/harness/model/fixtures/profiles), re-hashes every bound cell file, re-derives
    qualification from the files, and prints the verification receipt (schema s1.hold1b_a_verification.v1: handoff path+sha256, every bound
    file's absolute path/size/sha256, the identity, the A input paths/hashes/proof). Exit non-zero on the first-class problem list: missing,
    changed, tampered, partial or mismatched A blocks 1b.

receipt1b STAGE B_OUT A_PATH   (end of 1b)
    Writes B_OUT/receipt-1b.json (schema s1.hold1b_receipt.v1): the re-verified A handoff and its bound files, the headroom verdict taken from
    the immutable 1a RAW state data, both greedy arms' closed files with the spawn env pin actually applied (HIPFIRE_OFFLOAD_EXEC=pcie on BOTH the
    beta and the S1 arm), the hashes of both greedy input files and the token-identity adapter proof, and the exact comparison output."""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beta_input  # noqa: E402

HANDOFF_SCHEMA = "s1.hold1a_handoff.v1"
VERIFY_SCHEMA = "s1.hold1b_a_verification.v1"
RECEIPT_SCHEMA = "s1.hold1b_receipt.v1"
HANDOFF_NAME = "handoff-1a.json"
RECEIPT_NAME = "receipt-1b.json"
REPORT_SCHEMA = "hipfire.qwen4.slowlink_gate.v1"
OFFLOAD_KEY, OFFLOAD_VALUE = "HIPFIRE_OFFLOAD_EXEC", "pcie"
MANIFEST = os.path.join("manifests", "manifest-hold1.json")
SCOPE = "partial_h1_prim_state"

CELLS = {"prim": {"mode": "primitives", "purpose": "primitive_selftest"}, "state": {"mode": "state", "purpose": "product"}}
CELL_FILES = ("command.txt", "stdout.txt", "stderr.txt", "rc.txt", "spawn-env.json", "evidence.json", "baseline.json",
              "meminfo-before.txt", "meminfo-after.txt", "mem.csv", "vram.csv", "vram-events.jsonl", "vram-summary.json", "gate.json",
              "gate.json.progress.jsonl")
REQUIRED_FILES = {"prim": ("gate.json", "rc.txt", "spawn-env.json", "evidence.json"),
                  "state": ("gate.json", "rc.txt", "spawn-env.json", "evidence.json", "baseline.json", "vram.csv", "vram-events.jsonl",
                            "vram-summary.json")}
GREEDY_ARMS = {"greedy-beta": ("ids.jsonl",), "greedy-s1": ("ids.jsonl", "receipts.jsonl")}
GREEDY_FILES = ("command.txt", "stdout.txt", "stderr.txt", "rc.txt", "spawn-env.json", "evidence.json", "baseline.json", "meminfo-before.txt",
                "meminfo-after.txt", "mem.csv", "vram.csv", "vram-events.jsonl", "vram-summary.json")


def bad(msg):
    sys.exit(f"h1 handoff: {msg}")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path):
    return {"bytes": os.path.getsize(path), "sha256": sha_file(path)}


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def dir_hashes(d):
    if not os.path.isdir(d):
        return {}
    return {n: sha_file(os.path.join(d, n)) for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n))}


def write_atomic(path, obj):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stage_identity(stage):
    """Everything the two halves must share, read from the staged files (never from an argument)."""
    mp = os.path.join(stage, MANIFEST)
    man = jload(mp)
    if not isinstance(man, dict):
        bad(f"{mp} missing or not a JSON object")
    profiles = (man.get("memory_ledger") or {}).get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        bad(f"{mp} has no memory_ledger.profiles")
    harness_dir = HERE
    harness = {n: sha_file(os.path.join(harness_dir, n)) for n in sorted(os.listdir(harness_dir))
               if os.path.isfile(os.path.join(harness_dir, n)) and not n.endswith(".pyc")}
    return {
        "manifest": {"path": MANIFEST, "sha256": sha_file(mp)},
        "source": man.get("source"),
        "binary_sha256": (man.get("binary") or {}).get("sha256"),
        "kernel_code_object_hashes": (man.get("kernel") or {}).get("code_object_hashes"),
        "model": man.get("model"),
        "harness_declared": man.get("harness"),
        "profiles": {"sha256": hashlib.sha256(canon(profiles).encode()).hexdigest(),
                     "tiles": {k: v.get("tile_rows") for k, v in sorted(profiles.items())}},
        "inputs": man.get("inputs"),
        "bin_s1": dir_hashes(os.path.join(stage, "bin-s1")),
        "bin_beta": dir_hashes(os.path.join(stage, "bin-beta")),
        "fixtures": dir_hashes(os.path.join(stage, "fixtures")),
        "target_sha256": sha_file(os.path.join(stage, "target.json")) if os.path.isfile(os.path.join(stage, "target.json")) else None,
        "ledger_sha256": sha_file(os.path.join(stage, "ledger-hold1.json")) if os.path.isfile(os.path.join(stage, "ledger-hold1.json")) else None,
        "harness_files": harness,
    }


def identity_diff(sealed, now):
    out = []
    for key in sorted(set(sealed) | set(now)):
        a, b = sealed.get(key), now.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict) and key in ("bin_s1", "bin_beta", "fixtures", "harness_files"):
            names = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
            out.append(f"{key} differs in {names}")
        else:
            out.append(f"{key} differs")
    return out


PLACEMENT = "load.placement"
PLACEMENT_NAME = "slowlink-profile"  # the loaded placement a forced layer-major run must report
MIN_RESIDENT_LAYERS = 10


def checks_problems(name, rep, purpose, want_tile):
    """(placement detail, problems) from the report's `checks` block exactly as the gate serializes it (Checks::json):
    {status, required: [names], missing: [names], checks: [{name, status: pass|fail|skipped, detail, reason?}]}. Never trusts the root status:
    the block must be self-consistent (the status recomputed from the rows), every required check executed and passed, `load.placement` required
    and passed exactly once, and its detail the actual load read-back: placement 'slowlink-profile', the S1 profile admitted, >= 10 resident layers,
    expected_* and admitted_* non-null and equal to the sealed profile (and to evidence.s1_profile). A missing or failed placement row is refused
    even when a forged root status says pass."""
    block = rep.get("checks")
    if not isinstance(block, dict):
        return None, [f"{name}: report has no checks block (an aborted or setup-failed report carries none)"]
    problems = []
    required, rows = block.get("required"), block.get("checks")
    if not isinstance(required, list) or not required or any(not isinstance(n, str) for n in required):
        problems.append(f"{name}: checks.required is not a non-empty list of names")
        required = []
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("name"), str) or not isinstance(r.get("status"), str)
                                         for r in rows):
        return None, problems + [f"{name}: checks.checks is not a list of {{name, status}} rows"]
    if PLACEMENT not in required:
        problems.append(f"{name}: {PLACEMENT} is not among the required checks")
    executed = {r["name"] for r in rows if r["status"] in ("pass", "fail")}
    missing = [n for n in required if n not in executed]
    failed = [r["name"] for r in rows if r["status"] == "fail"]
    recomputed = "fail" if failed else ("incomplete" if not required or missing else "pass")
    if block.get("status") != "pass" or recomputed != "pass" or rep.get("status") != "pass":
        problems.append(f"{name}: checks status {block.get('status')!r} / recomputed {recomputed!r} / root {rep.get('status')!r} is not a consistent "
                        f"pass (failed {failed}, missing {missing})")
    if block.get("missing") != []:
        problems.append(f"{name}: checks.missing {block.get('missing')!r} is not empty")
    for n in required:
        if not any(r["name"] == n and r["status"] == "pass" for r in rows):
            problems.append(f"{name}: required check {n!r} has no pass row")
    placement = [r for r in rows if r["name"] == PLACEMENT]
    if len(placement) != 1:
        problems.append(f"{name}: {len(placement)} {PLACEMENT} rows (exactly one pass row is required)")
        return None, problems
    row = placement[0]
    if row["status"] != "pass":
        problems.append(f"{name}: {PLACEMENT} status {row['status']!r} is not pass")
    detail = row.get("detail")
    if not isinstance(detail, dict):
        return None, problems + [f"{name}: {PLACEMENT} row has no detail object"]
    if detail.get("placement") != PLACEMENT_NAME or detail.get("slowlink_profile") is not True:
        problems.append(f"{name}: {PLACEMENT} placement {detail.get('placement')!r} / slowlink_profile {detail.get('slowlink_profile')!r} is not "
                        f"the admitted S1 profile ({PLACEMENT_NAME!r}, true)")
    resident = detail.get("resident_layers")
    if not isinstance(resident, int) or isinstance(resident, bool) or resident < MIN_RESIDENT_LAYERS:
        problems.append(f"{name}: {PLACEMENT} resident_layers {resident!r} < {MIN_RESIDENT_LAYERS}")
    for prefix in ("expected", "admitted"):
        got = (detail.get(f"{prefix}_purpose"), detail.get(f"{prefix}_tile_rows"))
        if got != (purpose, want_tile) or want_tile is None:
            problems.append(f"{name}: {PLACEMENT} {prefix} profile {got[0]!r}/{got[1]!r} != {purpose!r}/{want_tile!r} of the sealed manifest")
    prof = (rep.get("evidence") or {}).get("s1_profile") if isinstance(rep.get("evidence"), dict) else None
    if isinstance(prof, dict) and (prof.get("admitted_purpose"), prof.get("admitted_tile_rows")) != \
            (detail.get("admitted_purpose"), detail.get("admitted_tile_rows")):
        problems.append(f"{name}: evidence.s1_profile admitted differs from the {PLACEMENT} read-back")
    return detail, problems


def cell_result(a_out, name, ident):
    """(info, problems) of one cell directory, derived from its closed files only."""
    spec, problems = CELLS[name], []
    d = os.path.join(a_out, name)
    if not os.path.isdir(d):
        return None, [f"{name}: cell directory {d} missing"]
    files = {fn: file_row(os.path.join(d, fn)) for fn in CELL_FILES if os.path.isfile(os.path.join(d, fn))}
    for fn in REQUIRED_FILES[name]:
        if fn not in files:
            problems.append(f"{name}: required file {fn} missing")
    if os.path.exists(os.path.join(d, "status.txt")):
        problems.append(f"{name}: status.txt present (a lifecycle, drain or sampler problem was recorded)")
    rc = None
    try:
        with open(os.path.join(d, "rc.txt")) as f:
            rc = int(f.read().strip())
    except (OSError, ValueError):
        pass
    if rc != 0:
        problems.append(f"{name}: rc {rc!r} is not 0")
    rep = jload(os.path.join(d, "gate.json"))
    info = {"rc": rc, "files": files}
    if not isinstance(rep, dict):
        problems.append(f"{name}: gate.json missing or not a completed report (a timeout/crash leaves only the forensic progress sidecar)")
    else:
        info["gate_status"] = rep.get("status")
        if rep.get("schema") != REPORT_SCHEMA or rep.get("status") != "pass":
            problems.append(f"{name}: gate report schema/status {rep.get('schema')!r}/{rep.get('status')!r} is not a completed pass")
        if rep.get("mode") != spec["mode"] or rep.get("route") != "layer-major":
            problems.append(f"{name}: gate mode/route {rep.get('mode')!r}/{rep.get('route')!r} != {spec['mode']!r}/'layer-major'")
        if rep.get("manifest_sha256") != ident["manifest"]["sha256"]:
            problems.append(f"{name}: gate manifest_sha256 {rep.get('manifest_sha256')!r} != the staged manifest {ident['manifest']['sha256']}")
        if rep.get("fixture_sha256") != ident["fixtures"].get("gate-fixture.json"):
            problems.append(f"{name}: gate fixture_sha256 {rep.get('fixture_sha256')!r} != the staged gate-fixture.json")
        prof = (rep.get("evidence") or {}).get("s1_profile") if isinstance(rep.get("evidence"), dict) else None
        want_tile = ident["profiles"]["tiles"].get(spec["purpose"])
        info["expected_purpose"] = prof.get("expected_purpose") if isinstance(prof, dict) else None
        info["expected_tile_rows"] = prof.get("expected_tile_rows") if isinstance(prof, dict) else None
        info["admitted_purpose"] = prof.get("admitted_purpose") if isinstance(prof, dict) else None
        info["admitted_tile_rows"] = prof.get("admitted_tile_rows") if isinstance(prof, dict) else None
        # expected_* is only what the binary is built to admit (known at startup); admitted_* is null until a bundle actually loaded and is then
        # the read-back of the loaded bundle's own profile. A cell qualifies only when BOTH equal the sealed manifest profile: an expected-only
        # report (admitted null) never qualifies.
        for prefix in ("expected", "admitted"):
            got = (info[f"{prefix}_purpose"], info[f"{prefix}_tile_rows"])
            if got != (spec["purpose"], want_tile) or want_tile is None:
                problems.append(f"{name}: {prefix} profile {got[0]!r}/{got[1]!r} != {spec['purpose']!r}/{want_tile!r} of the sealed manifest"
                                + (" (admitted is null until the actual bundle load is read back)" if prefix == "admitted" and got == (None, None) else ""))
        info["load_placement"], check_problems = checks_problems(name, rep, spec["purpose"], want_tile)
        problems += check_problems
    se = jload(os.path.join(d, "spawn-env.json"))
    pin = (se.get("set") or {}).get(OFFLOAD_KEY) if isinstance(se, dict) and isinstance(se.get("set"), dict) else None
    info["spawn_env_pin"] = {OFFLOAD_KEY: pin}
    if pin != OFFLOAD_VALUE:
        problems.append(f"{name}: spawn env {OFFLOAD_KEY} {pin!r} != {OFFLOAD_VALUE!r}")
    ev = jload(os.path.join(d, "evidence.json"))
    if not isinstance(ev, dict) or not (ev.get("gate_report") or {}).get("completed") or ev.get("timed_out"):
        problems.append(f"{name}: evidence.json does not show a completed, non-timed-out gate report")
    return info, problems


def seal(stage, a_out):
    ident = stage_identity(stage)
    cells, problems = {}, []
    for name in CELLS:
        info, p = cell_result(a_out, name, ident)
        cells[name] = info
        problems += p
    if problems:
        bad("refusing to seal 1a: " + "; ".join(problems))
    handoff = {
        "schema": HANDOFF_SCHEMA, "hold": "hold1a", "scope": SCOPE, "advance": False, "qualifies": True,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "a_out": os.path.abspath(a_out),
        "identity": ident, "cells": cells,
        "bound": "closed cell files only (command/stdout/stderr/rc/spawn-env/evidence/gate.json/progress sidecar/baseline/meminfo/mem.csv/"
                 "vram sampler files); never the run root logs or summary, never this file",
        "policy": "1a PASS is PARTIAL Hold 1, not an advance; the progress sidecar is bound but forensic only",
    }
    write_atomic(os.path.join(a_out, HANDOFF_NAME), handoff)
    print(os.path.join(a_out, HANDOFF_NAME))


def verify(stage, a_path):
    path = a_path if os.path.isfile(a_path) else os.path.join(a_path, HANDOFF_NAME)
    if not os.path.isfile(path):
        bad(f"no 1a handoff at {path}: 1b needs the sealed immutable 1a handoff (S1_HOLD1A_OUT or the CLI path)")
    handoff = jload(path)
    if not isinstance(handoff, dict):
        bad(f"{path} is not a JSON object")
    a_out = os.path.dirname(os.path.abspath(path))
    problems = []
    if handoff.get("schema") != HANDOFF_SCHEMA or handoff.get("hold") != "hold1a":
        problems.append(f"schema/hold {handoff.get('schema')!r}/{handoff.get('hold')!r} is not a hold1a handoff")
    if handoff.get("qualifies") is not True or handoff.get("advance") is not False or handoff.get("scope") != SCOPE:
        problems.append("handoff does not state qualifies true, advance false and the partial scope")
    sealed = handoff.get("identity")
    if not isinstance(sealed, dict):
        bad("; ".join(problems + ["handoff has no identity"]))
    ident = stage_identity(stage)
    problems += [f"stage identity changed since 1a: {d}" for d in identity_diff(sealed, ident)]
    receipt_cells = {}
    bound_cells = handoff.get("cells") if isinstance(handoff.get("cells"), dict) else {}
    for name in CELLS:
        bound = (bound_cells.get(name) or {}).get("files") if isinstance(bound_cells.get(name), dict) else None
        if not isinstance(bound, dict):
            problems.append(f"{name}: no bound files in the handoff")
            continue
        for fn in REQUIRED_FILES[name]:
            if fn not in bound:
                problems.append(f"{name}: required file {fn} is not bound by the handoff")
        for fn, row in sorted(bound.items()):
            p = os.path.join(a_out, name, fn)
            if fn not in CELL_FILES:
                problems.append(f"{name}: handoff binds an unknown file {fn}")
            elif not os.path.isfile(p):
                problems.append(f"{name}: bound file {fn} is missing")
            elif file_row(p) != row:
                problems.append(f"{name}: bound file {fn} changed since the seal (size/sha256 differ)")
        info, p = cell_result(a_out, name, ident)
        problems += p
        if info is not None:
            receipt_cells[name] = {k: v for k, v in info.items() if k != "files"}
            receipt_cells[name]["files"] = {fn: {"path": os.path.join(a_out, name, fn), **row} for fn, row in sorted(bound.items())}
    if problems:
        bad("1a handoff refused: " + "; ".join(problems))
    return {"schema": VERIFY_SCHEMA, "handoff": {"path": path, "sha256": sha_file(path)}, "a_out": a_out,
            "identity_sha256": hashlib.sha256(canon(sealed).encode()).hexdigest(), "identity": sealed,
            "a_inputs": sealed.get("inputs"), "cells": receipt_cells,
            "note": "verified from the closed cell files only; the progress sidecars are bound and forensic only"}


def receipt1b(stage, b_out, a_path):
    verification = verify(stage, a_path)  # re-verified now: the receipt never trusts an earlier check
    problems = []
    inputs = (verification["identity"].get("inputs") or {})
    fx = os.path.join(stage, "fixtures")
    canonical, beta = os.path.join(fx, "prompts.short.jsonl"), os.path.join(fx, "prompts.short.beta.jsonl")
    proof = beta_input.verify(canonical, beta)
    if (inputs.get("greedy_canonical") or {}).get("sha256") != proof["canonical"]["sha256"] or \
            (inputs.get("greedy_beta") or {}).get("sha256") != proof["beta"]["sha256"]:
        problems.append("the staged greedy inputs differ from the hashes the manifest binds")
    stored = jload(os.path.join(fx, "beta-input-proof.json"))
    if stored != proof:
        problems.append("beta-input-proof.json differs from the identity proof recomputed from the staged inputs")
    arms = {}
    for arm, outputs in GREEDY_ARMS.items():
        d = os.path.join(b_out, arm)
        row = {"files": {}}
        names = GREEDY_FILES + outputs
        for fn in names:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                row["files"][fn] = {"path": p, **file_row(p)}
        for fn in ("rc.txt", "spawn-env.json", "evidence.json") + outputs:
            if fn not in row["files"]:
                problems.append(f"{arm}: closed file {fn} missing")
        try:
            with open(os.path.join(d, "rc.txt")) as f:
                row["rc"] = int(f.read().strip())
        except (OSError, ValueError):
            row["rc"] = None
        if row["rc"] != 0 or os.path.exists(os.path.join(d, "status.txt")):
            problems.append(f"{arm}: rc {row['rc']!r} or a recorded lifecycle problem (status.txt)")
        se = jload(os.path.join(d, "spawn-env.json"))
        pin = (se.get("set") or {}).get(OFFLOAD_KEY) if isinstance(se, dict) and isinstance(se.get("set"), dict) else None
        row["spawn_env_pin"] = {OFFLOAD_KEY: pin}
        if pin != OFFLOAD_VALUE:
            problems.append(f"{arm}: spawn env {OFFLOAD_KEY} {pin!r} != {OFFLOAD_VALUE!r} (both arms carry the H1 pin)")
        arms[arm] = row
    hr_path = os.path.join(b_out, "headroom-state", "stdout.txt")
    hr = jload(hr_path)
    if not isinstance(hr, dict) or hr.get("status") != "pass":
        problems.append("headroom-state is not a recorded pass")
    cmp_path = os.path.join(b_out, "greedy-cmp", "stdout.txt")
    cmp_text = open(cmp_path).read() if os.path.isfile(cmp_path) else ""
    if "PASS:" not in cmp_text:
        problems.append("greedy-cmp has no PASS line")
    if problems:
        bad("refusing 1b receipt: " + "; ".join(problems))
    receipt = {
        "schema": RECEIPT_SCHEMA, "hold": "hold1b", "advance_candidate": True,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "a_handoff": verification,
        "headroom_state": {"source": "the immutable 1a RAW state cell (no second GPU state run)", "path": hr_path, **file_row(hr_path),
                           "status": hr["status"], "gate": hr.get("gate")},
        "greedy_inputs": {"canonical": {"path": canonical, **file_row(canonical)}, "beta": {"path": beta, **file_row(beta)},
                          "adapter_proof": {"path": os.path.join(fx, "beta-input-proof.json"), **file_row(os.path.join(fx, "beta-input-proof.json"))},
                          "proof": proof},
        "greedy_arms": arms,
        "greedy_cmp": {"path": cmp_path, **file_row(cmp_path)},
        "label": beta_input.LABEL,
    }
    write_atomic(os.path.join(b_out, RECEIPT_NAME), receipt)
    print(os.path.join(b_out, RECEIPT_NAME))


def main(argv):
    if len(argv) == 4 and argv[1] == "seal":
        seal(argv[2], argv[3])
        return 0
    if len(argv) == 4 and argv[1] == "verify":
        print(json.dumps(verify(argv[2], argv[3]), indent=1))
        return 0
    if len(argv) == 5 and argv[1] == "receipt1b":
        receipt1b(argv[2], argv[3], argv[4])
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
