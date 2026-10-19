#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""h1b_bridge.py print-memory EXEC_JSON EXPECTED_SHA256
              | resolve EXEC_JSON EXPECTED_SHA256
              | verify STAGE A_PATH BASE_HARNESS EXEC_JSON EXPECTED_SHA256
              | receipt1b STAGE B_OUT A_PATH BASE_HARNESS EXEC_JSON EXPECTED_SHA256

CPU-only bridge of the versioned, harness-only Hold 1b repair (R8). It NEVER seals, never rewrites the H1a handoff and never edits the frozen R7
harness: the unchanged original R7 h1_handoff.py is imported by a normal `import` from its REAL actual path (the directory the independently
pinned execution JSON names, verified file by file first; no monkeypatching of HERE or anything else) and its `verify` is called unchanged on the
immutable H1a handoff. Its complete v1 verification, INCLUDING the original R7 identity (every harness file hash), is retained verbatim next to the
separately named effective R8 execution identity.

The external, immutable `execution-h1b-r8.json` (schema s1.hold1b_execution.v1; its sha256 is supplied INDEPENDENTLY on the command line, never read
from a file the run could rewrite) binds: the base stage and H1a handoff, the exact base R7 directory with its COMPLETE actual helper-file map and
commit/tree, the effective R8 commit/tree/parent, this directory's four files, the saved diff, the CPU/source proof, the reference source/binary
identities, the approved delta pins, and a SEPARATE effective memory ledger (the R7 stage ledger stays the immutable H1a/base provenance and is never
mutated). Fresh OUT and the actual FABRIC_JOB are bound at verify/receipt time from the environment and hold.jsonl that hold_init wrote for THIS job.

  print-memory   static validation only (works outside a fabric job; claims no run binding). stdout EXACTLY "22.4 67\\n"; stderr names the separate
                 physical 24 GiB card. Refuses a ledger/terms/hash mismatch and any effective figure equal to the base R7 host figure.
  resolve        static validation only; prints three lines for the shell: base R7 directory, base stage, H1a handoff path.
  verify         full static + stage/handoff binding + run binding + unchanged original verify; prints schema s1.hold1b_a_verification.v2 JSON.
  receipt1b      re-does verify, requires the a-handoff row's stdout to equal it, preserves EVERY original receipt guard, additionally requires the
                 exact approved pins (beta 5 AND chunk 4096, rejected on S1), the observed beta `qwen4 prefill chunk: 4096 rows` line and the up-to-32 /
                 EOS id semantics, and writes receipt-1b.json (schema s1.hold1b_receipt.v2) atomically.
No seal operation exists here."""
import hashlib
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))

EXEC_SCHEMA = "s1.hold1b_execution.v1"
VERIFY_SCHEMA = "s1.hold1b_a_verification.v2"
RECEIPT_SCHEMA = "s1.hold1b_receipt.v2"
LEDGER_SCHEMA = "s1.hold1b_effective_memory_ledger.v1"
PROOF_SCHEMA = "s1.beta_math_proof.v1"
RECEIPT_NAME = "receipt-1b.json"
R8_FILES = ("hold1b-correct.sh", "h1b_bridge.py", "prove_beta_math.py", "test_h1_handoff.py")
BETA_COMMIT = "ff799626eb3dddf8ce827a90d88c990fc0eb4aa2"

RESIDENT_KEY, CHUNK_KEY, SLOWLINK_KEY = "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS", "HIPFIRE_PREFILL_CHUNK_ROWS", "HIPFIRE_QWEN4_SLOWLINK_PREFILL"
BETA_PINS = {RESIDENT_KEY: "5", CHUNK_KEY: "4096"}
CANDIDATE_PINS = {SLOWLINK_KEY: "1"}
ACTUAL_CHUNK_ROWS = 4096
MAX_NEW_TOKENS = 32
D2_NOTE = "absent_in_ff799_preserved_metadata_only"
DEVICE_GIB, HOST_GIB = 22.4, 67
GIB = 1 << 30
PHYSICAL_CARD_BYTES = 24 * GIB
# Effective (R8) capacity ledger: source-quoted constants of the contract. A capacity BOUND, not an RSS measurement or a requested placement.
LEDGER_FIXED_TERMS = {"external_ple_staging": 83_886_080, "ple_cache": 268_435_456, "g2_mirrors": 2_678_063_104, "max_live_construction_vec": 445_644_800,
                      "forward_ple": 41_943_040, "forward_tokens": 32_768, "mesh_tokens": 32_768}
LEDGER_HOST = {"host_weight_layers": 49, "owners": 98, "payload_bytes": 65_509_785_600, "guard_bytes": 102_760_448, "host_owned_bytes": 65_612_546_048,
               "fixed_bytes": 3_518_038_016, "beta_upper_bytes": 69_130_584_064, "candidate_upper_bytes": 68_467_955_712,
               "sequential_max_bytes": 69_130_584_064, "allowance_bytes": 2_147_483_648, "needed_bytes": 71_278_067_712, "reserved_bytes": 71_940_702_208,
               "residual_bytes": 662_634_496}
LEDGER_BASIS = "capacity_bound_not_rss_measurement"

HEX40, HEX64 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")
CHUNK_LINE = re.compile(r"^\s*qwen4 prefill chunk: (\d+) rows \(requested (\d+)\)\s*$")
ROUTED_LINE = re.compile(r"^qwen4_greedy_ids: routed experts: layers 0\.\.(\d+) in VRAM, (\d+) tensors host-mapped\s*$")
DONOR_RE = re.compile(r"through resident layers (\d+) and (\d+)")
GREEDY_ROWS_AFTER_A = ("headroom-state", "beta-input", "greedy-beta", "greedy-s1", "greedy-cmp", "receipt-1b")


def bad(msg):
    sys.exit(f"h1b bridge: {msg}")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            bad(f"a JSON object repeats the key {k!r}")
        out[k] = v
    return out


def jstrict(path, what):
    try:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("utf-8"), object_pairs_hook=_pairs)
    except (OSError, ValueError) as e:
        bad(f"{what} {path} unreadable or not JSON: {e}")


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def req(obj, keys, where):
    if not isinstance(obj, dict):
        bad(f"{where} is not an object")
    missing, extra = sorted(set(keys) - set(obj)), sorted(set(obj) - set(keys))
    if missing or extra:
        bad(f"{where} keys differ (missing {missing}, unexpected {extra})")
    return obj


def apath(v, where):
    if not isinstance(v, str) or not v.startswith("/") or "\x00" in v:
        bad(f"{where} {v!r} is not an absolute path")
    return v


def hx(v, pat, where):
    if not isinstance(v, str) or not pat.fullmatch(v):
        bad(f"{where} {v!r} is not a lowercase hex identity")
    return v


def file_ref(obj, where):
    req(obj, ("path", "sha256"), where)
    apath(obj["path"], f"{where}.path")
    hx(obj["sha256"], HEX64, f"{where}.sha256")
    return obj


def bound_file(obj, where):
    """A {path, sha256} reference whose file must hash to exactly its sha256."""
    file_ref(obj, where)
    if not os.path.isfile(obj["path"]):
        bad(f"{where} file {obj['path']} is missing")
    got = sha_file(obj["path"])
    if got != obj["sha256"]:
        bad(f"{where} {obj['path']} sha256 {got} != bound {obj['sha256']}")
    return obj


def inventory(directory):
    """Every top-level file of a harness directory by sha256 (never a .pyc), exactly as the original stage_identity hashes its HERE."""
    return {n: sha_file(os.path.join(directory, n)) for n in sorted(os.listdir(directory))
            if os.path.isfile(os.path.join(directory, n)) and not n.endswith(".pyc")}


def inventory_diff(want, got):
    return sorted(n for n in set(want) | set(got) if want.get(n) != got.get(n))


# ------------------------------------------------------------------------------------------------ execution JSON
def load_execution(path, expected_sha):
    if not isinstance(expected_sha, str) or not HEX64.fullmatch(expected_sha):
        bad(f"expected execution sha256 {expected_sha!r} is not 64 lowercase hex (it must be supplied independently)")
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        bad(f"execution JSON {path} unreadable: {e}")
    got = sha_bytes(raw)
    if got != expected_sha:
        bad(f"execution JSON {path} sha256 {got} != the independently supplied {expected_sha}")
    try:
        exe = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except ValueError as e:
        bad(f"execution JSON is not JSON: {e}")
    req(exe, ("schema", "stage", "h1a_handoff", "base_r7", "execution_r8", "reference", "delta", "memory"), "execution")
    if exe["schema"] != EXEC_SCHEMA:
        bad(f"execution schema {exe['schema']!r} != {EXEC_SCHEMA}")
    req(exe["stage"], ("path", "manifest_sha256"), "execution.stage")
    apath(exe["stage"]["path"], "execution.stage.path")
    hx(exe["stage"]["manifest_sha256"], HEX64, "execution.stage.manifest_sha256")
    file_ref(exe["h1a_handoff"], "execution.h1a_handoff")
    r7 = req(exe["base_r7"], ("path", "commit", "tree", "runtime_commit", "runtime_tree", "files", "ledger"), "execution.base_r7")
    apath(r7["path"], "execution.base_r7.path")
    for key in ("commit", "tree", "runtime_commit", "runtime_tree"):
        hx(r7[key], HEX40, f"execution.base_r7.{key}")
    if not isinstance(r7["files"], dict) or not r7["files"] or any(not isinstance(k, str) or not HEX64.fullmatch(str(v)) for k, v in r7["files"].items()):
        bad("execution.base_r7.files is not a complete {name: sha256} map")
    led = req(r7["ledger"], ("path", "sha256", "fabric_reservation_device_gib", "host_reservation_gib"), "execution.base_r7.ledger")
    apath(led["path"], "execution.base_r7.ledger.path")
    hx(led["sha256"], HEX64, "execution.base_r7.ledger.sha256")
    if led["fabric_reservation_device_gib"] != DEVICE_GIB or not is_int(led["host_reservation_gib"]) or led["host_reservation_gib"] <= 0:
        bad("execution.base_r7.ledger device/host figures are not the recorded R7 ledger figures")
    r8 = req(exe["execution_r8"], ("path", "commit", "tree", "parent_commit", "files", "diff", "proof", "memory_ledger"), "execution.execution_r8")
    apath(r8["path"], "execution.execution_r8.path")
    for key in ("commit", "tree", "parent_commit"):
        hx(r8[key], HEX40, f"execution.execution_r8.{key}")
    if not isinstance(r8["files"], dict) or sorted(r8["files"]) != sorted(R8_FILES) or any(not HEX64.fullmatch(str(v)) for v in r8["files"].values()):
        bad(f"execution.execution_r8.files must be exactly {sorted(R8_FILES)} with sha256 values")
    diff = req(r8["diff"], ("command", "from", "to", "path", "sha256"), "execution.execution_r8.diff")
    if not isinstance(diff["command"], str) or not diff["command"].strip():
        bad("execution.execution_r8.diff.command is empty")
    hx(diff["from"], HEX40, "diff.from")
    hx(diff["to"], HEX40, "diff.to")
    file_ref({"path": diff["path"], "sha256": diff["sha256"]}, "execution.execution_r8.diff")
    file_ref(r8["proof"], "execution.execution_r8.proof")
    file_ref(r8["memory_ledger"], "execution.execution_r8.memory_ledger")
    ref = req(exe["reference"], ("beta_source_path", "beta_source_commit", "beta_binary_sha256"), "execution.reference")
    apath(ref["beta_source_path"], "execution.reference.beta_source_path")
    hx(ref["beta_binary_sha256"], HEX64, "execution.reference.beta_binary_sha256")
    if ref["beta_source_commit"] != BETA_COMMIT:
        bad(f"execution.reference.beta_source_commit {ref['beta_source_commit']!r} is not the immutable beta {BETA_COMMIT}")
    delta = req(exe["delta"], ("quality_reference_only", "beta", "candidate", "actual_chunk_rows", "max_new_tokens", "d2_default"), "execution.delta")
    if delta["quality_reference_only"] is not True or delta["beta"] != BETA_PINS or delta["candidate"] != CANDIDATE_PINS or \
            delta["actual_chunk_rows"] != ACTUAL_CHUNK_ROWS or delta["max_new_tokens"] != MAX_NEW_TOKENS or delta["d2_default"] != D2_NOTE:
        bad("execution.delta is not the approved quality-reference-only delta (beta resident 5 + chunk 4096; candidate SLOWLINK_PREFILL=1 only; up to 32 tokens)")
    mem = req(exe["memory"], ("device_gib", "host_gib"), "execution.memory")
    if mem["device_gib"] != DEVICE_GIB or mem["host_gib"] != HOST_GIB or not is_int(mem["host_gib"]):
        bad(f"execution.memory {mem!r} is not the effective {DEVICE_GIB} {HOST_GIB} GiB reservation")
    if mem["host_gib"] == led["host_reservation_gib"]:
        bad("the effective host reservation equals the base R7 host figure: the effective and base ledgers must stay separate")
    if r8["parent_commit"] != r7["runtime_commit"] or diff["from"] != r8["parent_commit"] or diff["to"] != r8["commit"]:
        bad("execution diff endpoints are not parent(runtime) -> R8 commit")
    if r8["commit"] == r7["commit"] or r8["commit"] == r7["runtime_commit"]:
        bad("the effective R8 commit equals a base commit")
    return exe, got


def validate_ledger(led, device_gib, host_gib):
    """The effective R8 ledger's terms and arithmetic. Returns its summary for the verification/receipt."""
    req(led, ("schema", "final", "fabric_reservation_device_gib", "host_reservation_gib", "physical_card_bytes", "host", "basis"), "effective ledger")
    if led["schema"] != LEDGER_SCHEMA or led["final"] is not True or led["basis"] != LEDGER_BASIS:
        bad("effective ledger schema/final/basis is not the final capacity-bound ledger")
    if led["fabric_reservation_device_gib"] != device_gib or led["host_reservation_gib"] != host_gib or not is_int(led["host_reservation_gib"]):
        bad(f"effective ledger reservation {led['fabric_reservation_device_gib']!r}/{led['host_reservation_gib']!r} != {device_gib}/{host_gib}")
    if led["physical_card_bytes"] != PHYSICAL_CARD_BYTES or not is_int(led["physical_card_bytes"]):
        bad(f"effective ledger physical_card_bytes {led['physical_card_bytes']!r} != the physical 24 GiB card {PHYSICAL_CARD_BYTES}")
    if device_gib * GIB > led["physical_card_bytes"]:
        bad("the fabric exclusive device cap exceeds the physical card")
    h = req(led["host"], tuple(LEDGER_HOST) + ("fixed_terms",), "effective ledger host")
    for key, want in LEDGER_HOST.items():
        if not is_int(h[key]) or h[key] != want:
            bad(f"effective ledger host.{key} {h[key]!r} != {want}")
    terms = h["fixed_terms"]
    req(terms, tuple(LEDGER_FIXED_TERMS), "effective ledger host.fixed_terms")
    for key, want in LEDGER_FIXED_TERMS.items():
        if not is_int(terms[key]) or terms[key] != want:
            bad(f"effective ledger fixed term {key} {terms[key]!r} != {want}")
    # the arithmetic, recomputed (never trusting a stored total)
    if h["payload_bytes"] + h["guard_bytes"] != h["host_owned_bytes"] or h["host_weight_layers"] * 2 != h["owners"]:
        bad("effective ledger host weight terms do not add up (payload + guards, 49 layers x 2 owners)")
    if sum(terms.values()) != h["fixed_bytes"] or h["host_owned_bytes"] + h["fixed_bytes"] != h["beta_upper_bytes"]:
        bad("effective ledger beta terms do not add up")
    if h["sequential_max_bytes"] != max(h["beta_upper_bytes"], h["candidate_upper_bytes"]) or \
            h["needed_bytes"] != h["sequential_max_bytes"] + h["allowance_bytes"] or h["reserved_bytes"] != host_gib * GIB or \
            h["residual_bytes"] != h["reserved_bytes"] - h["needed_bytes"] or h["residual_bytes"] < 0:
        bad("effective ledger sequential-maximum/allowance/reservation arithmetic is inconsistent (a MAX of the arms, never a sum)")
    return {"host_reservation_gib": host_gib, "fabric_reservation_device_gib": device_gib, "physical_card_bytes": led["physical_card_bytes"],
            "needed_bytes": h["needed_bytes"], "reserved_bytes": h["reserved_bytes"], "residual_bytes": h["residual_bytes"], "basis": led["basis"]}


def validate_proof(proof):
    if not isinstance(proof, dict) or proof.get("schema") != PROOF_SCHEMA:
        bad(f"the CPU/source proof is not a {PROOF_SCHEMA} object")
    if (proof.get("beta_source") or {}).get("commit") != BETA_COMMIT:
        bad("the CPU/source proof is not about the immutable beta source commit")
    pins = proof.get("pins") or {}
    if pins.get("beta") != BETA_PINS or pins.get("required_actual_chunk_rows") != ACTUAL_CHUNK_ROWS or pins.get("quality_reference_only") is not True:
        bad("the CPU/source proof does not state the approved pins / actual chunk 4096 / quality-reference-only")
    donors = proof.get("donors") or {}
    if donors.get("n5") != [4, 3] or donors.get("n10") != [8, 9]:
        bad(f"the CPU/source proof donor parity order {donors.get('n5')!r}/{donors.get('n10')!r} is not N5 [4, 3] / N10 [8, 9]")
    claims = proof.get("claims") or {}
    if claims.get("measured_fit") is not False or claims.get("generated_output_equality") is not False or claims.get("gpu_executed") is not False:
        bad("the CPU/source proof claims a measured fit / output equality / GPU run it cannot have")
    if not all(c.get("same_class_per_layer_and_forward") is True for c in proof.get("cases", [])) or len(proof.get("cases", [])) != 6:
        bad("the CPU/source proof does not carry the six fixed same-class cases")
    if (proof.get("negative_8192_ceiling") or {}).get("class_changed") is not True:
        bad("the CPU/source proof lacks the class-changing 8192-ceiling negative case")
    return {"schema": proof["schema"], "beta_source_commit": proof["beta_source"]["commit"], "digest_fixed_cases_sha256": proof.get("digest_fixed_cases_sha256")}


def check_static(exe_path, expected_sha):
    """Everything provable from the independently pinned JSON and the files it names, before ANY helper is imported or sourced."""
    exe, sha = load_execution(exe_path, expected_sha)
    r7, r8 = exe["base_r7"], exe["execution_r8"]
    base = os.path.realpath(r7["path"])
    if not os.path.isdir(base):
        bad(f"base R7 directory {r7['path']} is missing")
    got = inventory(base)
    if got != r7["files"]:
        bad(f"base R7 directory differs from the bound complete helper-file map in {inventory_diff(r7['files'], got)}")
    if os.path.realpath(r8["path"]) != HERE:
        bad(f"this bridge runs from {HERE} but the execution JSON binds the R8 directory {r8['path']}")
    got = inventory(HERE)
    if got != r8["files"]:
        bad(f"the R8 directory differs from the bound file hashes in {inventory_diff(r8['files'], got)}")
    bound_file({"path": r8["diff"]["path"], "sha256": r8["diff"]["sha256"]}, "saved diff")
    proof_summary = validate_proof(jstrict(bound_file(r8["proof"], "CPU/source proof")["path"], "CPU/source proof"))
    ledger = jstrict(bound_file(r8["memory_ledger"], "effective memory ledger")["path"], "effective memory ledger")
    ledger_summary = validate_ledger(ledger, exe["memory"]["device_gib"], exe["memory"]["host_gib"])
    return exe, sha, proof_summary, ledger_summary


# ------------------------------------------------------------------------------------------------ original R7 helper (normal import)
_ORIG = {}


def load_orig(base_dir):
    base = os.path.realpath(base_dir)
    if base in _ORIG:
        return _ORIG[base]
    if base not in sys.path:
        sys.path.insert(0, base)
    orig = importlib.import_module("h1_handoff")
    if os.path.realpath(orig.__file__) != os.path.join(base, "h1_handoff.py") or os.path.realpath(orig.HERE) != base:
        bad(f"the imported h1_handoff ({orig.__file__}) is not the original R7 helper of {base}")
    _ORIG[base] = orig
    return orig


# ------------------------------------------------------------------------------------------------ run binding / verify
def run_binding(exe, a_out, initial):
    """Fresh OUT and the actual FABRIC_JOB, tied to the hold.jsonl that THIS job's hold_init wrote (never fabricated before the queue)."""
    out, job = os.environ.get("OUT"), os.environ.get("FABRIC_JOB")
    if not out or not job:
        bad("no run binding: OUT and FABRIC_JOB must both be set (verify/receipt run inside the actual fabric job)")
    out = os.path.realpath(apath(out, "OUT"))
    if not os.path.isdir(out):
        bad(f"OUT {out} is not a directory")
    for label, other in (("the sealed 1a run directory", a_out), ("the stage", exe["stage"]["path"]), ("the base R7 directory", exe["base_r7"]["path"]),
                         ("the R8 directory", exe["execution_r8"]["path"])):
        other = os.path.realpath(other)
        if out == other or out.startswith(other + os.sep):
            bad(f"OUT {out} is inside {label}: a fresh unique run directory is required")
    if os.path.exists(os.path.join(out, RECEIPT_NAME)):
        bad(f"OUT {out} already has {RECEIPT_NAME}: not a fresh run")
    if initial:
        stale = [n for n in GREEDY_ROWS_AFTER_A if os.path.exists(os.path.join(out, n))]
        if stale:
            bad(f"OUT {out} already holds rows {stale} before the a-handoff row finished: not a fresh run")
    try:
        with open(os.path.join(out, "hold.jsonl")) as f:
            events = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as e:
        bad(f"OUT {out} has no readable hold.jsonl written by hold_init: {e}")
    starts = [e for e in events if isinstance(e, dict) and e.get("event") == "hold_start"]
    if len(starts) != 1 or events[0] is not starts[0]:
        bad("hold.jsonl does not begin with exactly one hold_start (not a fresh run)")
    start = starts[0]
    if start.get("hold") != "hold1b" or os.path.realpath(str(start.get("out"))) != out or str(start.get("fabric_job")) != job:
        bad(f"hold.jsonl hold_start {start!r} does not name hold1b / OUT {out} / FABRIC_JOB {job}")
    return {"out": out, "fabric_job": job}


def stage_binding(exe, stage, a_path):
    if os.path.realpath(stage) != os.path.realpath(exe["stage"]["path"]):
        bad(f"stage {stage} is not the bound stage {exe['stage']['path']}")
    mp = os.path.join(stage, "manifests", "manifest-hold1.json")
    if not os.path.isfile(mp) or sha_file(mp) != exe["stage"]["manifest_sha256"]:
        bad("the staged manifest-hold1.json differs from the sha256 the execution JSON binds")
    handoff = os.path.realpath(exe["h1a_handoff"]["path"])
    given = os.path.realpath(a_path if os.path.isfile(a_path) else os.path.join(a_path, "handoff-1a.json"))
    if given != handoff:
        bad(f"A path {a_path} is not the bound H1a handoff {exe['h1a_handoff']['path']}")
    if not os.path.isfile(handoff) or sha_file(handoff) != exe["h1a_handoff"]["sha256"]:
        bad("the H1a handoff differs from the sha256 the execution JSON binds (H1a is immutable)")
    return handoff


def base_print_memory(base_dir, stage, exe):
    ledger = os.path.join(stage, "ledger-hold1.json")
    r = subprocess.run([sys.executable, os.path.join(base_dir, "ledger_profiles.py"), "print-memory", ledger], capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=120)
    want = f"{DEVICE_GIB} {exe['base_r7']['ledger']['host_reservation_gib']}\n"
    if r.returncode != 0 or r.stdout != want:
        bad(f"the base R7 ledger prints {r.stdout!r} (rc {r.returncode}), not the bound {want!r}")
    return r.stdout.strip()


def _verify(stage, a_path, base_harness, execution_json, expected_sha, initial):
    exe, exec_sha, proof_summary, ledger_summary = check_static(execution_json, expected_sha)
    r7, r8 = exe["base_r7"], exe["execution_r8"]
    base = os.path.realpath(r7["path"])
    if os.path.realpath(base_harness) != base:
        bad(f"base harness {base_harness} is not the bound base R7 directory {r7['path']}")
    handoff = stage_binding(exe, stage, a_path)
    run = run_binding(exe, os.path.dirname(handoff), initial)
    orig = load_orig(base)
    v1 = orig.verify(stage, a_path)  # the unchanged original R7 verification (exits non-zero on any problem)
    ident = v1["identity"]
    problems = []
    if os.path.realpath(v1["handoff"]["path"]) != handoff or v1["handoff"]["sha256"] != exe["h1a_handoff"]["sha256"]:
        problems.append("the original verification names another handoff than the bound one")
    if ident["manifest"]["sha256"] != exe["stage"]["manifest_sha256"]:
        problems.append("sealed identity manifest differs from the bound manifest")
    if ident["harness_files"] != r7["files"]:
        problems.append(f"sealed identity harness_files differs from the bound base R7 map in {inventory_diff(r7['files'], ident['harness_files'])}")
    src = ident.get("source") or {}
    if (src.get("tip"), src.get("tree")) != (r7["runtime_commit"], r7["runtime_tree"]):
        problems.append("sealed runtime source tip/tree differs from the bound runtime commit/tree")
    decl = ident.get("harness_declared") or {}
    if (decl.get("commit"), decl.get("tree")) != (r7["commit"], r7["tree"]):
        problems.append("declared harness commit/tree differs from the bound base R7 commit/tree")
    ledger_path = os.path.join(stage, "ledger-hold1.json")
    if ident.get("ledger_sha256") != r7["ledger"]["sha256"] or not os.path.isfile(ledger_path) or sha_file(ledger_path) != r7["ledger"]["sha256"]:
        problems.append("the base R7 stage ledger differs from the immutable base ledger the execution JSON binds")
    if (ident.get("bin_beta") or {}).get("qwen4_greedy_ids") != exe["reference"]["beta_binary_sha256"]:
        problems.append("the staged beta qwen4_greedy_ids differs from the bound reference binary sha256")
    if problems:
        bad("execution binding refused: " + "; ".join(problems))
    base_memory = base_print_memory(base, stage, exe)
    verification = {
        "schema": VERIFY_SCHEMA,
        "run": run,
        "base_r7": {"path": r7["path"], "commit": r7["commit"], "tree": r7["tree"], "runtime_commit": r7["runtime_commit"], "runtime_tree": r7["runtime_tree"],
                    "harness_files": r7["files"], "harness_files_sha256": sha_bytes(canon(r7["files"]).encode()),
                    "ledger": {**r7["ledger"], "print_memory": base_memory, "role": "immutable H1a/base provenance only; never mutated, never the R8 reservation"},
                    "verification_v1": v1},
        "execution_r8": {"execution_json": {"path": os.path.realpath(execution_json), "sha256": exec_sha},
                         "path": r8["path"], "commit": r8["commit"], "tree": r8["tree"], "parent_commit": r8["parent_commit"], "files": r8["files"],
                         "diff": r8["diff"], "proof": {**r8["proof"], **proof_summary},
                         "memory_ledger": {**r8["memory_ledger"], "summary": ledger_summary, "role": "effective R8 capacity ledger (separate from base_r7.ledger)"},
                         "memory": exe["memory"]},
        "stage": exe["stage"], "h1a_handoff": exe["h1a_handoff"], "reference": exe["reference"], "delta": exe["delta"],
        "a_out": v1["a_out"],
        "note": "the unchanged original R7 verify result is retained in base_r7.verification_v1; the effective R8 execution identity is separate",
    }
    return json.loads(json.dumps(verification))


def verify(stage, a_path, base_harness, execution_json, expected_sha256):
    return _verify(stage, a_path, base_harness, execution_json, expected_sha256, True)


# ------------------------------------------------------------------------------------------------ receipt
def parse_beta_stderr(text):
    """The beta arm's OWN observed chunk/placement lines. The source-derived proof never substitutes for these."""
    chunk = [CHUNK_LINE.match(l) for l in text.splitlines() if "qwen4 prefill chunk:" in l]
    if len(chunk) != 1 or chunk[0] is None:
        bad(f"beta stderr has {len(chunk)} unambiguous `qwen4 prefill chunk: N rows (requested M)` line(s); exactly one is required")
    actual, requested = int(chunk[0].group(1)), int(chunk[0].group(2))
    if actual != ACTUAL_CHUNK_ROWS or requested != ACTUAL_CHUNK_ROWS:
        bad(f"beta observed actual prefill chunk {actual} (requested {requested}); exactly {ACTUAL_CHUNK_ROWS} is required (a downshift or growth changes the arithmetic class)")
    routed = [ROUTED_LINE.match(l) for l in text.splitlines() if "routed experts: layers 0.." in l]
    if len(routed) != 1 or routed[0] is None or int(routed[0].group(1)) != int(BETA_PINS[RESIDENT_KEY]):
        bad(f"beta stderr does not show exactly one `routed experts: layers 0..{BETA_PINS[RESIDENT_KEY]} in VRAM` line")
    donors = [DONOR_RE.search(l) for l in text.splitlines() if "long-prefill expert staging" in l]
    seen = None
    if donors and all(d is not None for d in donors):
        seen = sorted({(int(d.group(1)), int(d.group(2))) for d in donors})
        if len(seen) != 1 or sorted(seen[0]) != [3, 4]:
            bad(f"beta stage donors {seen} are not the N5 donor pair {{3, 4}}")
    return {"actual_chunk_rows": actual, "requested_chunk_rows": requested, "resident_expert_layers": int(routed[0].group(1)),
            "host_mapped_tensors": int(routed[0].group(2)), "stage_donors_observed": list(seen[0]) if seen else None}


def parse_command(path, arm, outputs_name):
    try:
        with open(path) as f:
            tokens = shlex.split(f.read())
    except (OSError, ValueError) as e:
        return None, [f"{arm}: command.txt unreadable: {e}"]
    problems = []

    def value_of(flag):
        return tokens[tokens.index(flag) + 1] if flag in tokens and tokens.index(flag) + 1 < len(tokens) else None
    if value_of("--max-new-tokens") != str(MAX_NEW_TOKENS):
        problems.append(f"{arm}: command --max-new-tokens {value_of('--max-new-tokens')!r} != {MAX_NEW_TOKENS} (the original UP-TO-32 option)")
    if value_of("--thinking") != "off":
        problems.append(f"{arm}: command --thinking {value_of('--thinking')!r} != 'off'")
    if not (value_of("--input") or "").endswith(outputs_name):
        problems.append(f"{arm}: command --input {value_of('--input')!r} is not {outputs_name}")
    return tokens, problems


def ids_semantics(path, arm):
    """Up to 32 generated ids per record; EOS/terminator stops early (finish_reason stop, at least the stop token), a full run is length with exactly 32."""
    problems, rows = [], []
    try:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError) as e:
        return None, [f"{arm}: ids.jsonl unreadable: {e}"]
    if len(records) != 7:
        problems.append(f"{arm}: {len(records)} id records, not all 7")
    for r in records:
        ids, fin = r.get("token_ids"), r.get("finish_reason")
        if not isinstance(ids, list) or any(not is_int(t) for t in ids) or len(ids) > MAX_NEW_TOKENS:
            problems.append(f"{arm}: record {r.get('id')!r} has token_ids that are not at most {MAX_NEW_TOKENS} ints")
            continue
        if fin == "length":
            if len(ids) != MAX_NEW_TOKENS:
                problems.append(f"{arm}: record {r.get('id')!r} finish_reason length with {len(ids)} ids (must be exactly {MAX_NEW_TOKENS})")
        elif fin == "stop":
            if not 1 <= len(ids) <= MAX_NEW_TOKENS:
                problems.append(f"{arm}: record {r.get('id')!r} finish_reason stop with {len(ids)} ids")
        else:
            problems.append(f"{arm}: record {r.get('id')!r} finish_reason {fin!r} is neither stop nor length")
        rows.append({"id": r.get("id"), "n_tokens": len(ids), "finish_reason": fin, "prompt_tokens": r.get("prompt_tokens")})
    return rows, problems


def receipt1b(stage, b_out, a_path, base_harness, execution_json, expected_sha256):
    verification = _verify(stage, a_path, base_harness, execution_json, expected_sha256, False)  # re-verified now: the receipt never trusts an earlier check
    orig = load_orig(verification["base_r7"]["path"])
    exe, _, _, _ = check_static(execution_json, expected_sha256)
    problems = []
    if os.path.realpath(b_out) != verification["run"]["out"]:
        problems.append(f"B_OUT {b_out} is not the bound run OUT {verification['run']['out']}")
    initial_path = os.path.join(b_out, "a-handoff", "stdout.txt")
    initial = orig.jload(initial_path)
    if not isinstance(initial, dict) or canon(initial) != canon(verification):
        problems.append("the a-handoff row's initial verification stdout differs from the re-verification now (execution/run/handoff binding must be identical)")
    inputs = (verification["base_r7"]["verification_v1"]["identity"].get("inputs") or {})
    fx = os.path.join(stage, "fixtures")
    canonical, beta = os.path.join(fx, "prompts.short.jsonl"), os.path.join(fx, "prompts.short.beta.jsonl")
    proof = orig.beta_input.verify(canonical, beta)
    if (inputs.get("greedy_canonical") or {}).get("sha256") != proof["canonical"]["sha256"] or \
            (inputs.get("greedy_beta") or {}).get("sha256") != proof["beta"]["sha256"]:
        problems.append("the staged greedy inputs differ from the hashes the manifest binds")
    stored = orig.jload(os.path.join(fx, "beta-input-proof.json"))
    if stored != proof:
        problems.append("beta-input-proof.json differs from the identity proof recomputed from the staged inputs")
    arms, beta_observed, greedy_semantics = {}, None, {}
    for arm, outputs in orig.GREEDY_ARMS.items():
        d = os.path.join(b_out, arm)
        row = {"files": {}}
        for fn in orig.GREEDY_FILES + outputs:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                row["files"][fn] = {"path": p, **orig.file_row(p)}
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
        se = orig.jload(os.path.join(d, "spawn-env.json"))
        pins = se.get("set") if isinstance(se, dict) and isinstance(se.get("set"), dict) else {}
        row["spawn_env_pin"] = {key: pins.get(key) for key, _ in orig.ENV_PINS}
        row["arm"] = "non_default_d2_off" if pins.get(orig.D2_KEY) == orig.D2_VALUE else None
        for key, value in orig.ENV_PINS:
            if pins.get(key) != value:
                problems.append(f"{arm}: spawn env {key} {pins.get(key)!r} != {value!r} (both arms carry the H1 pin and the D2 non-default pin)")
        # --- R8 additions: the approved delta is exact on the beta arm and absent from the candidate arm
        if arm == "greedy-beta":
            row["approved_pins"] = {k: pins.get(k) for k in BETA_PINS}
            for key, value in BETA_PINS.items():
                if pins.get(key) != value:
                    problems.append(f"{arm}: spawn env {key} {pins.get(key)!r} != {value!r} (the approved quality-reference pins are exactly EXPERT_VRAM_LAYERS=5 and PREFILL_CHUNK_ROWS=4096)")
            if SLOWLINK_KEY in pins:
                problems.append(f"{arm}: the beta arm carries {SLOWLINK_KEY}")
        else:
            row["approved_pins"] = {SLOWLINK_KEY: pins.get(SLOWLINK_KEY)}
            for key in BETA_PINS:
                if key in pins:
                    problems.append(f"{arm}: the S1 candidate arm carries the beta-only pin {key}={pins[key]!r} (no resident count or chunk pin on S1)")
            if pins.get(SLOWLINK_KEY) != "1":
                problems.append(f"{arm}: spawn env {SLOWLINK_KEY} {pins.get(SLOWLINK_KEY)!r} != '1'")
        tokens, p = parse_command(os.path.join(d, "command.txt"), arm, "prompts.short.beta.jsonl" if arm == "greedy-beta" else "prompts.short.jsonl")
        problems += p
        if tokens is not None:
            for key, value in (BETA_PINS if arm == "greedy-beta" else CANDIDATE_PINS).items():
                if f"{key}={value}" not in tokens:
                    problems.append(f"{arm}: command.txt does not show {key}={value}")
            if arm == "greedy-s1":
                if "--receipt" not in tokens:
                    problems.append(f"{arm}: command.txt has no --receipt")
                for key in BETA_PINS:
                    if any(t.startswith(f"{key}=") for t in tokens):
                        problems.append(f"{arm}: command.txt carries {key}")
        rows, p = ids_semantics(os.path.join(d, "ids.jsonl"), arm)
        problems += p
        greedy_semantics[arm] = {"max_new_tokens": MAX_NEW_TOKENS, "records": rows,
                                 "rule": "up to 32 ids; finish_reason stop (EOS/terminator) may be shorter, length is exactly 32; no 32 generated tokens are assumed"}
        if arm == "greedy-beta":
            try:
                with open(os.path.join(d, "stderr.txt"), errors="replace") as f:
                    beta_observed = parse_beta_stderr(f.read())
            except OSError as e:
                problems.append(f"{arm}: stderr.txt unreadable: {e}")
            except SystemExit as e:  # parse_beta_stderr refuses with its own message; the receipt reports it with every other problem
                problems.append(str(e))
        arms[arm] = row
    hr_path = os.path.join(b_out, "headroom-state", "stdout.txt")
    hr = orig.jload(hr_path)
    if not isinstance(hr, dict) or hr.get("status") != "pass":
        problems.append("headroom-state is not a recorded pass")
    cmp_path = os.path.join(b_out, "greedy-cmp", "stdout.txt")
    cmp_text = ""
    if os.path.isfile(cmp_path):
        with open(cmp_path) as f:
            cmp_text = f.read()
    if "PASS:" not in cmp_text:
        problems.append("greedy-cmp has no PASS line")
    elif not re.search(r"^PASS: 7 records\b", cmp_text, re.M):
        problems.append("greedy-cmp PASS line does not cover all 7 records")
    if problems:
        bad("refusing 1b receipt: " + "; ".join(problems))
    receipt = {
        "schema": RECEIPT_SCHEMA, "hold": "hold1b", "advance_candidate": True,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "a_handoff": verification,
        "headroom_state": {"source": "the immutable 1a RAW state cell (no second GPU state run)", "path": hr_path, **orig.file_row(hr_path),
                           "status": hr["status"], "gate": hr.get("gate")},
        "greedy_inputs": {"canonical": {"path": canonical, **orig.file_row(canonical)}, "beta": {"path": beta, **orig.file_row(beta)},
                          "adapter_proof": {"path": os.path.join(fx, "beta-input-proof.json"), **orig.file_row(os.path.join(fx, "beta-input-proof.json"))},
                          "proof": proof},
        "greedy_arms": arms,
        "greedy_cmp": {"path": cmp_path, **orig.file_row(cmp_path)},
        "label": orig.beta_input.LABEL,
        "base_r7": verification["base_r7"],
        "execution_r8": verification["execution_r8"],
        "delta": {**exe["delta"], "label": "quality-reference pins of the immutable beta arm only; never H2 policy and never an S1 resident floor",
                  "d2_note": "HIPFIRE_QWEN4_D2_DEFAULT is absent from ff799; the required env 0 is recorded metadata only; OFFLOAD_EXEC=pcie is the operational control"},
        "proof": verification["execution_r8"]["proof"],
        "effective_memory_ledger": verification["execution_r8"]["memory_ledger"],
        "base_memory_ledger": verification["base_r7"]["ledger"],
        "initial_verify": {"path": initial_path, **orig.file_row(initial_path), "verification_sha256": sha_bytes(canon(verification).encode())},
        "run": verification["run"],
        "beta_observed": {**beta_observed, "source_line": "qwen4 prefill chunk: 4096 rows (requested 4096)",
                          "note": "observed in the beta arm's own stderr; the source proof never substitutes for it"},
        "greedy_semantics": greedy_semantics,
    }
    orig.write_atomic(os.path.join(b_out, RECEIPT_NAME), receipt)
    print(os.path.join(b_out, RECEIPT_NAME))


# ------------------------------------------------------------------------------------------------ CLI
def main(argv):
    if len(argv) == 4 and argv[1] == "print-memory":
        exe, _, _, ledger = check_static(argv[2], argv[3])
        print(f"{exe['memory']['device_gib']} {exe['memory']['host_gib']}")
        print(f"print-memory: fabric exclusive device {exe['memory']['device_gib']} GiB (the physical card is a SEPARATE 24 GiB, card_bytes "
              f"{ledger['physical_card_bytes']}); effective R8 host {exe['memory']['host_gib']} GiB = capacity bound {ledger['needed_bytes']} B needed, "
              f"{ledger['reserved_bytes']} B reserved, {ledger['residual_bytes']} B residual (NOT an RSS measurement); base R7 host "
              f"{exe['base_r7']['ledger']['host_reservation_gib']} GiB stays separate provenance", file=sys.stderr)
        return 0
    if len(argv) == 4 and argv[1] == "resolve":
        exe, _, _, _ = check_static(argv[2], argv[3])
        print(exe["base_r7"]["path"])
        print(exe["stage"]["path"])
        print(exe["h1a_handoff"]["path"])
        return 0
    if len(argv) == 7 and argv[1] == "verify":
        print(json.dumps(verify(*argv[2:]), indent=1))
        return 0
    if len(argv) == 8 and argv[1] == "receipt1b":
        receipt1b(*argv[2:])
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
