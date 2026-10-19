#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the versioned harness-only Hold 1b repair (R8): h1b_bridge.py (execution binding, unchanged original R7 verify, receipt v2),
hold1b-correct.sh (R8 entry structure) and prove_beta_math.py (source-attested same-class oracle). Synthetic temp files only: no GPU, no model,
no staged binary is ever executed; the ORIGINAL R7 helpers (h1_handoff, World fixture of test_hold1_split.py) are imported from their real path
with a normal import and never monkeypatched. The only mocks are for the oracle's `git rev-parse` of a synthetic source tree.

    python3 test_h1_handoff.py -v            (S1_R7_HOLDS selects the frozen R7 helper directory when it is not at a default location)
    S1_BETA_SOURCE=/path/to/ff799/worktree   enables the real-source oracle tests (otherwise they report skipped, never a fake pass)
"""
import contextlib
import copy
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True  # importing the frozen R7 helpers must not add anything to their directory
HERE = os.path.dirname(os.path.realpath(__file__))
R7_DEFAULTS = ("/home/kaden/qcal/release-0.4.3/slowlink-prefill/wt-hold1-r7-harness/scripts/slowlink-holds",
               "/home/kaden/qcal/release-0.4.3/slowlink-prefill/wt-hold1-r7-894be98a4/scripts/slowlink-holds")
BETA_DEFAULT = "/home/kaden/qcal/release-0.4.3/slowlink-prefill/wt-beta-anchor"


def find_r7():
    for cand in ([os.environ["S1_R7_HOLDS"]] if os.environ.get("S1_R7_HOLDS") else []) + list(R7_DEFAULTS):
        if os.path.isfile(os.path.join(cand, "h1_handoff.py")):
            return os.path.realpath(cand)
    sys.exit("test_h1_handoff: the frozen R7 helper directory is not found (set S1_R7_HOLDS)")


R7 = find_r7()
BETA_SOURCE = os.environ.get("S1_BETA_SOURCE") or (BETA_DEFAULT if os.path.isdir(BETA_DEFAULT) else None)
sys.path.insert(0, HERE)
sys.path.insert(0, R7)
import h1_handoff as h1  # noqa: E402  (the ORIGINAL R7 module, imported from its real path)
import h1b_bridge as br  # noqa: E402
import ledger_testlib as lt  # noqa: E402
import prove_beta_math as pbm  # noqa: E402
import test_hold1_split as base  # noqa: E402  (the original World fixture)

SCRIPT = os.path.join(HERE, "hold1b-correct.sh")
BRIDGE = os.path.join(HERE, "h1b_bridge.py")


def sha(path):
    return h1.sha_file(path)


def rb(path):
    """Read bytes with the handle closed (never a bare open().read())."""
    return pathlib.Path(path).read_bytes()


def rt(path):
    return pathlib.Path(path).read_text()


def jdump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def proof_obj():
    cases = [pbm.case_row(name, length, 4096, 4096, 10, 5)[0] for name, length in pbm.CASES]
    return {"schema": br.PROOF_SCHEMA, "beta_source": {"commit": br.BETA_COMMIT}, "pins": {"beta": dict(br.BETA_PINS), "quality_reference_only": True,
                                                                                         "required_actual_chunk_rows": 4096},
            "donors": {"n5": pbm.donors(5), "n10": pbm.donors(10)}, "claims": {"measured_fit": False, "generated_output_equality": False, "gpu_executed": False},
            "cases": cases, "negative_8192_ceiling": {"class_changed": True}, "digest_fixed_cases_sha256": "d" * 64}


def ledger_obj():
    return {"schema": br.LEDGER_SCHEMA, "final": True, "fabric_reservation_device_gib": 22.4, "host_reservation_gib": 67, "physical_card_bytes": 25769803776,
            "host": {**br.LEDGER_HOST, "fixed_terms": dict(br.LEDGER_FIXED_TERMS)}, "basis": br.LEDGER_BASIS}


class Constants(unittest.TestCase):
    """The contract's source-quoted numbers, written independently of the bridge constants."""

    def test_effective_ledger_arithmetic_is_the_contract_arithmetic(self):
        self.assertEqual(65_509_785_600 + 102_760_448, 65_612_546_048)
        fixed = 83_886_080 + 268_435_456 + 2_678_063_104 + 445_644_800 + 41_943_040 + 32_768 + 32_768
        self.assertEqual(fixed, 3_518_038_016)
        self.assertEqual(65_612_546_048 + fixed, 69_130_584_064)
        self.assertEqual(max(69_130_584_064, 68_467_955_712) + 2 * 2**30, 71_278_067_712)  # sequential MAX + 2 GiB, never an arm sum
        self.assertEqual(67 * 2**30, 71_940_702_208)
        self.assertEqual(71_940_702_208 - 71_278_067_712, 662_634_496)
        self.assertEqual(49 * 2, 98)
        for key, want in (("payload_bytes", 65_509_785_600), ("guard_bytes", 102_760_448), ("host_owned_bytes", 65_612_546_048), ("fixed_bytes", 3_518_038_016),
                          ("beta_upper_bytes", 69_130_584_064), ("candidate_upper_bytes", 68_467_955_712), ("sequential_max_bytes", 69_130_584_064),
                          ("needed_bytes", 71_278_067_712), ("reserved_bytes", 71_940_702_208), ("residual_bytes", 662_634_496)):
            self.assertEqual(br.LEDGER_HOST[key], want, key)
        self.assertEqual((br.DEVICE_GIB, br.HOST_GIB, br.PHYSICAL_CARD_BYTES), (22.4, 67, 24 * 2**30))

    def test_approved_pins_and_chunk(self):
        self.assertEqual(br.BETA_PINS, {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "5", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096"})
        self.assertEqual(br.CANDIDATE_PINS, {"HIPFIRE_QWEN4_SLOWLINK_PREFILL": "1"})
        self.assertEqual((br.ACTUAL_CHUNK_ROWS, br.MAX_NEW_TOKENS), (4096, 32))


class Fixture(base.World):
    """The ORIGINAL World (a real sealed H1a over a synthetic stage; the real R7 harness directory) plus the R8 execution binding."""

    def setUp(self):
        super().setUp()
        self.seal()
        self.handoff = os.path.join(self.a, h1.HANDOFF_NAME)
        self.out = os.path.join(self.tmp, "run-out")
        os.makedirs(self.out)
        self.job = "j-test"
        self.write_hold_jsonl()
        patcher = mock.patch.dict(os.environ, {"OUT": self.out, "FABRIC_JOB": self.job})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ext = os.path.join(self.tmp, "ext")
        os.makedirs(self.ext)
        self.proof_path, self.ledger_path, self.diff_path = (os.path.join(self.ext, n) for n in ("proof.json", "ledger.json", "diff.patch"))
        jdump(self.proof_path, proof_obj())
        jdump(self.ledger_path, ledger_obj())
        with open(self.diff_path, "w") as f:
            f.write("diff --git a/scripts/slowlink-holds-h1b-r8/h1b_bridge.py b/scripts/slowlink-holds-h1b-r8/h1b_bridge.py\n")
        self.exe = self.build_exec()
        self.publish()

    # ---- builders
    def write_hold_jsonl(self, **over):
        event = {"ts": "t", "event": "hold_start", "hold": "hold1b", "out": self.out, "fabric_job": self.job, "budget_s": 1275}
        event.update(over)
        with open(os.path.join(self.out, "hold.jsonl"), "w") as f:
            f.write(json.dumps(event) + "\n")

    def base_memory(self):
        r = subprocess.run([sys.executable, os.path.join(R7, "ledger_profiles.py"), "print-memory", os.path.join(self.stage, "ledger-hold1.json")],
                           capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        dev, host = r.stdout.split()
        return float(dev), int(host)

    def build_exec(self):
        dev, host = self.base_memory()
        ledger = os.path.join(self.stage, "ledger-hold1.json")
        return {
            "schema": br.EXEC_SCHEMA,
            "stage": {"path": self.stage, "manifest_sha256": sha(os.path.join(self.stage, h1.MANIFEST))},
            "h1a_handoff": {"path": self.handoff, "sha256": sha(self.handoff)},
            "base_r7": {"path": R7, "commit": "9" * 40, "tree": "8" * 40, "runtime_commit": lt.TIP, "runtime_tree": lt.TREE, "files": br.inventory(R7),
                        "ledger": {"path": ledger, "sha256": sha(ledger), "fabric_reservation_device_gib": dev, "host_reservation_gib": host}},
            "execution_r8": {"path": HERE, "commit": "7" * 40, "tree": "6" * 40, "parent_commit": lt.TIP, "files": br.inventory(HERE),
                             "diff": {"command": "git diff <40a> <r8> -- scripts/slowlink-holds-h1b-r8", "from": lt.TIP, "to": "7" * 40,
                                      "path": self.diff_path, "sha256": sha(self.diff_path)},
                             "proof": {"path": self.proof_path, "sha256": sha(self.proof_path)},
                             "memory_ledger": {"path": self.ledger_path, "sha256": sha(self.ledger_path)}},
            "reference": {"beta_source_path": "/beta/source", "beta_source_commit": br.BETA_COMMIT,
                          "beta_binary_sha256": sha(os.path.join(self.stage, "bin-beta", "qwen4_greedy_ids"))},
            "delta": {"quality_reference_only": True, "beta": dict(br.BETA_PINS), "candidate": dict(br.CANDIDATE_PINS), "actual_chunk_rows": 4096,
                      "max_new_tokens": 32, "d2_default": br.D2_NOTE},
            "memory": {"device_gib": 22.4, "host_gib": 67},
        }

    def write_exec(self, exe, name):
        """Write an execution JSON variant WITHOUT touching the default exec_path/exec_sha."""
        path = os.path.join(self.ext, name)
        jdump(path, exe)
        return path, sha(path)

    def publish(self, exe=None, name="execution-h1b-r8.json"):
        exe = self.exe if exe is None else exe
        self.exec_path, self.exec_sha = self.write_exec(exe, name)
        return self.exec_path, self.exec_sha

    def mutated(self, mutate, name="mutated.json"):
        exe = copy.deepcopy(self.exe)
        mutate(exe)
        return self.write_exec(exe, name)

    def verify(self, exec_path=None, exec_sha=None, stage=None, a_path=None, base=R7):
        # explicit None checks: an empty-string argument must reach the bridge (an `or` fallback would silently substitute the good value)
        return br.verify(self.stage if stage is None else stage, self.a if a_path is None else a_path, base,
                         self.exec_path if exec_path is None else exec_path, self.exec_sha if exec_sha is None else exec_sha)

    def refused(self, fragment, **kw):
        self.assertRefused(lambda: self.verify(**kw), fragment)

    def dir_state(self, d):
        return {os.path.relpath(os.path.join(r, n), d): sha(os.path.join(r, n)) for r, _, fs in os.walk(d) for n in fs}


class Verify(Fixture):
    def test_valid_r7_seal_verifies_to_v2_with_separate_identities_and_no_h1a_rewrite(self):
        before = self.dir_state(self.a)
        v = self.verify()
        self.assertEqual(self.dir_state(self.a), before, "H1a is immutable: nothing under it is written, resealed or added")
        self.assertEqual(v["schema"], br.VERIFY_SCHEMA)
        self.assertEqual(v["run"], {"out": os.path.realpath(self.out), "fabric_job": self.job})
        v1 = v["base_r7"]["verification_v1"]
        self.assertEqual(v1["schema"], h1.VERIFY_SCHEMA)
        self.assertEqual(v1, h1.verify(self.stage, self.a), "the FULL unchanged original verification is retained verbatim")
        self.assertEqual(v1["identity"]["harness_files"], br.inventory(R7))  # the ORIGINAL R7 identity
        self.assertEqual(v["base_r7"]["harness_files"], br.inventory(R7))
        self.assertEqual(v["execution_r8"]["files"], br.inventory(HERE))
        self.assertNotEqual(v["execution_r8"]["files"], v["base_r7"]["harness_files"])
        self.assertEqual(v["execution_r8"]["commit"], "7" * 40)
        self.assertEqual(v["base_r7"]["commit"], "9" * 40)
        # base ledger vs effective ledger are separate and never mixed
        dev, host = self.base_memory()
        self.assertEqual(v["base_r7"]["ledger"]["print_memory"], f"{dev} {host}")
        self.assertEqual(v["base_r7"]["ledger"]["sha256"], sha(os.path.join(self.stage, "ledger-hold1.json")))
        self.assertEqual(v["execution_r8"]["memory_ledger"]["sha256"], sha(self.ledger_path))
        self.assertEqual(v["execution_r8"]["memory"], {"device_gib": 22.4, "host_gib": 67})
        self.assertNotEqual(host, 67)
        self.assertEqual(v["execution_r8"]["memory_ledger"]["summary"]["reserved_bytes"], 71_940_702_208)
        self.assertEqual(v["delta"]["beta"], br.BETA_PINS)
        self.assertEqual(v["execution_r8"]["execution_json"]["sha256"], self.exec_sha)
        self.assertEqual(v["execution_r8"]["proof"]["beta_source_commit"], br.BETA_COMMIT)

    def test_the_original_module_is_imported_from_its_real_path_unpatched(self):
        orig = br.load_orig(R7)
        self.assertIs(orig, h1)
        self.assertEqual(os.path.realpath(orig.__file__), os.path.join(R7, "h1_handoff.py"))
        self.assertEqual(os.path.realpath(orig.HERE), R7)
        self.assertEqual(orig.RECEIPT_SCHEMA, "s1.hold1b_receipt.v1")  # the original receipt writer is untouched and not used
        import ast
        tree = ast.parse(rt(BRIDGE))
        patched = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                   if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in ("orig", "h1_handoff", "h1")]
        self.assertEqual(patched, [], "the bridge never assigns attributes of the original module (no monkeypatching)")

    def test_cli_verify_prints_the_same_json_inside_the_run(self):
        r = subprocess.run([sys.executable, BRIDGE, "verify", self.stage, self.handoff, R7, self.exec_path, self.exec_sha], capture_output=True, text=True,
                           env={**os.environ, "OUT": self.out, "FABRIC_JOB": self.job})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), self.verify())

    def test_independent_expected_sha_is_mandatory_and_exact(self):
        for bad_sha, fragment in (("0" * 64, "!= the independently supplied"), ("", "not 64 lowercase hex"), (self.exec_sha.upper(), "not 64 lowercase hex"),
                                  ("zz", "not 64 lowercase hex")):
            with self.subTest(bad_sha):
                self.refused(fragment, exec_sha=bad_sha)
        with open(self.exec_path, "a") as f:
            f.write(" ")
        self.refused("!= the independently supplied")
        with self.assertRaises(SystemExit):
            br.verify(self.stage, self.a, R7, os.path.join(self.ext, "missing.json"), self.exec_sha)

    def test_malformed_execution_json_is_refused_closed(self):
        cases = {
            "schema": lambda e: e.update({"schema": "x"}),
            "missing section": lambda e: e.pop("delta"),
            "extra top key": lambda e: e.update({"surprise": 1}),
            "stage manifest sha": lambda e: e["stage"].update({"manifest_sha256": "g" * 64}),
            "relative stage path": lambda e: e["stage"].update({"path": "stage"}),
            "base commit not 40 hex": lambda e: e["base_r7"].update({"commit": "abc"}),
            "base files empty": lambda e: e["base_r7"].update({"files": {}}),
            "r8 files extra": lambda e: e["execution_r8"]["files"].update({"extra.py": "0" * 64}),
            "r8 files missing": lambda e: e["execution_r8"]["files"].pop("prove_beta_math.py"),
            "reference commit": lambda e: e["reference"].update({"beta_source_commit": "0" * 40}),
            "diff endpoints": lambda e: e["execution_r8"]["diff"].update({"from": "1" * 40}),
            "parent is not the runtime commit": lambda e: e["execution_r8"].update({"parent_commit": "2" * 40}),
            "r8 equals base": lambda e: e["execution_r8"].update({"commit": e["base_r7"]["commit"]}),
            "delta beta resident 10": lambda e: e["delta"]["beta"].update({"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "10"}),
            "delta beta chunk 8192": lambda e: e["delta"]["beta"].update({"HIPFIRE_PREFILL_CHUNK_ROWS": "8192"}),
            "delta chunk pin missing": lambda e: e["delta"]["beta"].pop("HIPFIRE_PREFILL_CHUNK_ROWS"),
            "delta candidate resident pin": lambda e: e["delta"]["candidate"].update({"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "5"}),
            "delta not quality-only": lambda e: e["delta"].update({"quality_reference_only": False}),
            "delta max new tokens forced 64": lambda e: e["delta"].update({"max_new_tokens": 64}),
            "delta actual chunk 2048": lambda e: e["delta"].update({"actual_chunk_rows": 2048}),
            "memory effective 55": lambda e: e["memory"].update({"host_gib": 55}),
            "memory effective 54": lambda e: e["memory"].update({"host_gib": 54}),
            "memory device 24": lambda e: e["memory"].update({"device_gib": 24}),
            "memory host equals base host": lambda e: e["memory"].update({"host_gib": e["base_r7"]["ledger"]["host_reservation_gib"]}),
        }
        for label, mutate in cases.items():
            with self.subTest(label):
                path, digest = self.mutated(mutate)
                self.refused("", exec_path=path, exec_sha=digest)

    def test_tampered_or_missing_proof_diff_ledger_are_refused(self):
        for label, path, fragment in (("proof", self.proof_path, "CPU/source proof"), ("diff", self.diff_path, "saved diff"),
                                      ("ledger", self.ledger_path, "effective memory ledger")):
            with self.subTest(label):
                original = rb(path)
                with open(path, "ab") as f:
                    f.write(b" ")
                self.refused(fragment)
                os.remove(path)
                self.refused("is missing")
                with open(path, "wb") as f:
                    f.write(original)
                self.verify()

    def test_proof_content_is_checked(self):
        mutators = {
            "wrong schema": lambda p: p.update({"schema": "x"}),
            "wrong commit": lambda p: p["beta_source"].update({"commit": "0" * 40}),
            "chunk 8192": lambda p: p["pins"].update({"required_actual_chunk_rows": 8192}),
            "beta pins differ": lambda p: p["pins"]["beta"].update({"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "10"}),
            "donor parity [3,4]": lambda p: p["donors"].update({"n5": [3, 4]}),
            "n10 donors": lambda p: p["donors"].update({"n10": [9, 8]}),
            "measured fit claim": lambda p: p["claims"].update({"measured_fit": True}),
            "output equality claim": lambda p: p["claims"].update({"generated_output_equality": True}),
            "case missing": lambda p: p["cases"].pop(),
            "case class differs": lambda p: p["cases"][0].update({"same_class_per_layer_and_forward": False}),
            "negative missing": lambda p: p.update({"negative_8192_ceiling": {"class_changed": False}}),
        }
        for label, mutate in mutators.items():
            with self.subTest(label):
                proof = proof_obj()
                mutate(proof)
                jdump(self.proof_path, proof)
                self.exe["execution_r8"]["proof"]["sha256"] = sha(self.proof_path)
                self.publish()
                self.refused("proof")
        jdump(self.proof_path, proof_obj())
        self.exe["execution_r8"]["proof"]["sha256"] = sha(self.proof_path)
        self.publish()
        self.verify()

    def test_effective_ledger_terms_and_hashes_are_checked(self):
        mutators = {
            "schema": lambda l: l.update({"schema": "x"}),
            "not final": lambda l: l.update({"final": False}),
            "effective 55": lambda l: l.update({"host_reservation_gib": 55}),
            "device 22.5": lambda l: l.update({"fabric_reservation_device_gib": 22.5}),
            "physical card 25 GiB": lambda l: l.update({"physical_card_bytes": 25 * 2**30}),
            "basis measured": lambda l: l.update({"basis": "rss_measurement"}),
            "fixed term +1": lambda l: l["host"]["fixed_terms"].update({"g2_mirrors": l["host"]["fixed_terms"]["g2_mirrors"] + 1}),
            "fixed term missing": lambda l: l["host"]["fixed_terms"].pop("mesh_tokens"),
            "fixed term extra": lambda l: l["host"]["fixed_terms"].update({"free_lunch": 1}),
            "host key extra": lambda l: l["host"].update({"extra": 1}),
            "sum of arms instead of max": lambda l: l["host"].update({"sequential_max_bytes": l["host"]["beta_upper_bytes"] + l["host"]["candidate_upper_bytes"]}),
            "allowance 1": lambda l: l["host"].update({"allowance_bytes": 1}),
            "residual wrong": lambda l: l["host"].update({"residual_bytes": 0}),
            "49 layers become 48": lambda l: l["host"].update({"host_weight_layers": 48}),
            "payload +1": lambda l: l["host"].update({"payload_bytes": l["host"]["payload_bytes"] + 1}),
            "bool in int slot": lambda l: l["host"].update({"owners": True}),
        }
        for key in br.LEDGER_HOST:
            mutators[f"host.{key} +1"] = (lambda k: lambda l: l["host"].update({k: l["host"][k] + 1}))(key)
        for label, mutate in mutators.items():
            with self.subTest(label):
                led = ledger_obj()
                mutate(led)
                jdump(self.ledger_path, led)
                self.exe["execution_r8"]["memory_ledger"]["sha256"] = sha(self.ledger_path)
                self.publish()
                self.refused("")
        jdump(self.ledger_path, ledger_obj())
        self.exe["execution_r8"]["memory_ledger"]["sha256"] = sha(self.ledger_path)
        self.publish()
        self.verify()

    def test_base_and_r8_inventories_are_exact(self):
        name = sorted(self.exe["base_r7"]["files"])[0]
        for label, mutate, fragment in (
                ("base file hash", lambda e: e["base_r7"]["files"].update({name: "0" * 64}), "base R7 directory differs"),
                ("base file removed from the map", lambda e: e["base_r7"]["files"].pop(name), "base R7 directory differs"),
                ("base file added to the map", lambda e: e["base_r7"]["files"].update({"ghost.py": "0" * 64}), "base R7 directory differs"),
                ("r8 file hash", lambda e: e["execution_r8"]["files"].update({"h1b_bridge.py": "0" * 64}), "R8 directory differs"),
                ("r8 path elsewhere", lambda e: e["execution_r8"].update({"path": self.tmp}), "binds the R8 directory"),
                ("base path elsewhere", lambda e: e["base_r7"].update({"path": self.tmp}), "base R7 directory"),
        ):
            with self.subTest(label):
                path, digest = self.mutated(mutate)
                self.refused(fragment, exec_path=path, exec_sha=digest)

    def test_stage_and_handoff_binding(self):
        for label, mutate, fragment in (
                ("other stage", lambda e: e["stage"].update({"path": self.tmp}), "not the bound stage"),
                ("manifest sha", lambda e: e["stage"].update({"manifest_sha256": "0" * 64}), "manifest-hold1.json differs"),
                ("handoff sha", lambda e: e["h1a_handoff"].update({"sha256": "0" * 64}), "H1a handoff differs"),
                ("handoff path", lambda e: e["h1a_handoff"].update({"path": os.path.join(self.tmp, "other.json")}), "not the bound H1a handoff"),
        ):
            with self.subTest(label):
                path, digest = self.mutated(mutate)
                self.refused(fragment, exec_path=path, exec_sha=digest)
        self.refused("is not the bound base R7 directory", base=self.tmp)
        self.verify(a_path=self.handoff)  # the handoff file works as well as the run directory

    def test_runtime_model_fixture_kernel_binary_profile_changes_are_refused(self):
        # (a) any staged change that moves the manifest breaks the manifest binding of the independently pinned JSON
        manifest = os.path.join(self.stage, h1.MANIFEST)
        original = rt(manifest)

        def edit_manifest(fn):
            man = json.loads(original)
            fn(man)
            with open(manifest, "w") as f:
                json.dump(man, f)
        edits = {
            "runtime source tip": lambda m: m["source"].update({"tip": "e" * 40}),
            "runtime tree": lambda m: m["source"].update({"tree": "e" * 40}),
            "model sha": lambda m: m["model"].update({"sha256": "f" * 64}),
            "model path": lambda m: m["model"].update({"path": "/other.mq4"}),
            "kernel code objects": lambda m: m["kernel"]["code_object_hashes"].update({"k": "e" * 64}),
            "S1 product profile tile": lambda m: m["memory_ledger"]["profiles"]["product"].update({"tile_rows": 512}),
            "declared harness commit": lambda m: m["harness"].update({"commit": "5" * 40}),
            "greedy inputs binding": lambda m: m["inputs"]["greedy_canonical"].update({"sha256": "0" * 64}),
        }
        for label, fn in edits.items():
            with self.subTest("manifest edit: " + label):
                edit_manifest(fn)
                self.refused("manifest-hold1.json differs")
                # (b) even with the JSON re-bound to the edited manifest, the unchanged original verify refuses the sealed-identity change
                exe = copy.deepcopy(self.exe)
                exe["stage"]["manifest_sha256"] = sha(manifest)
                path, digest = self.write_exec(exe, "rebound.json")
                self.assertRaises(SystemExit, lambda: self.verify(exec_path=path, exec_sha=digest))
                with open(manifest, "w") as f:
                    f.write(original)
        self.verify()
        # (c) staged binaries, fixtures, ledger and target are in the sealed identity itself
        for rel in ("bin-s1/qwen4_slowlink_gate", "bin-s1/qwen4_greedy_ids", "bin-beta/qwen4_greedy_ids", "fixtures/gate-fixture.json",
                    "fixtures/prompts.short.jsonl", "fixtures/prompts.short.beta.jsonl", "target.json", "ledger-hold1.json"):
            with self.subTest("staged file: " + rel):
                path = os.path.join(self.stage, rel)
                original_bytes = rb(path)
                with open(path, "ab") as f:
                    f.write(b" ")
                self.assertRaises(SystemExit, lambda: self.verify())
                with open(path, "wb") as f:
                    f.write(original_bytes)
        self.verify()

    def test_bound_identity_cross_checks(self):
        for label, mutate, fragment in (
                ("runtime commit", lambda e: (e["base_r7"].update({"runtime_commit": "3" * 40}), e["execution_r8"].update({"parent_commit": "3" * 40}),
                                              e["execution_r8"]["diff"].update({"from": "3" * 40})), "runtime source tip/tree"),
                ("runtime tree", lambda e: e["base_r7"].update({"runtime_tree": "3" * 40}), "runtime source tip/tree"),
                ("base harness commit", lambda e: e["base_r7"].update({"commit": "4" * 40}), "declared harness commit/tree"),
                ("base harness tree", lambda e: e["base_r7"].update({"tree": "4" * 40}), "declared harness commit/tree"),
                ("base ledger sha", lambda e: e["base_r7"]["ledger"].update({"sha256": "0" * 64}), "base R7 stage ledger"),
                ("beta binary sha", lambda e: e["reference"].update({"beta_binary_sha256": "0" * 64}), "reference binary"),
                ("base printed host figure", lambda e: e["base_r7"]["ledger"].update({"host_reservation_gib": e["base_r7"]["ledger"]["host_reservation_gib"] + 1}),
                 "base R7 ledger prints"),
        ):
            with self.subTest(label):
                path, digest = self.mutated(mutate)
                self.refused(fragment, exec_path=path, exec_sha=digest)

    def test_h1a_tamper_and_partial_a_still_block(self):
        for rel in ("state/gate.json", "state/vram.csv", "prim/gate.json", "state/rc.txt"):
            path = os.path.join(self.a, rel)
            original = rb(path)
            with open(path, "ab") as f:
                f.write(b" ")
            self.refused("")
            with open(path, "wb") as f:
                f.write(original)
        os.remove(os.path.join(self.a, "state", "vram-summary.json"))
        self.refused("")

    def test_fresh_out_and_actual_fabric_job_binding(self):
        def hold(**over):
            self.write_hold_jsonl(**over)
        with mock.patch.dict(os.environ):
            os.environ.pop("OUT")
            self.refused("no run binding")
        with mock.patch.dict(os.environ):
            os.environ.pop("FABRIC_JOB")
            self.refused("no run binding")
        with mock.patch.dict(os.environ, {"FABRIC_JOB": "j-other"}):
            self.refused("does not name hold1b")
        with mock.patch.dict(os.environ, {"OUT": os.path.join(self.tmp, "nope")}):
            self.refused("is not a directory")
        with mock.patch.dict(os.environ, {"OUT": os.path.join(self.a, "state")}):
            self.refused("inside the sealed 1a run directory")
        with mock.patch.dict(os.environ, {"OUT": self.stage}):
            self.refused("inside the stage")
        hold(out=os.path.join(self.tmp, "another-out"))
        self.refused("does not name hold1b")
        hold(hold="hold1a")
        self.refused("does not name hold1b")
        hold(fabric_job="j-not-this-job")
        self.refused("does not name hold1b")
        hold(event="not_a_start")
        self.refused("exactly one hold_start")
        with open(os.path.join(self.out, "hold.jsonl"), "a") as f:
            f.write(json.dumps({"event": "hold_start", "hold": "hold1b", "out": self.out, "fabric_job": self.job}) + "\n")
        self.write_hold_jsonl()
        with open(os.path.join(self.out, "hold.jsonl"), "a") as f:
            f.write(json.dumps({"event": "hold_start", "hold": "hold1b", "out": self.out, "fabric_job": self.job}) + "\n")
        self.refused("exactly one hold_start")
        os.remove(os.path.join(self.out, "hold.jsonl"))
        self.refused("no readable hold.jsonl")
        self.write_hold_jsonl()
        os.makedirs(os.path.join(self.out, "greedy-beta"))  # an OUT that already holds later rows is not fresh
        self.refused("not a fresh run")
        shutil.rmtree(os.path.join(self.out, "greedy-beta"))
        with open(os.path.join(self.out, br.RECEIPT_NAME), "w") as f:
            f.write("{}")
        self.refused("already has receipt-1b.json")
        os.remove(os.path.join(self.out, br.RECEIPT_NAME))
        # a numeric fabric job id (hold_init's jlog json-decodes it) is the same job
        with mock.patch.dict(os.environ, {"FABRIC_JOB": "1934"}):
            self.write_hold_jsonl(fabric_job=1934)
            self.verify()


class PrintMemory(Fixture):
    def run_cli(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith("FABRIC_") and k != "OUT"}
        e.update(env or {})
        return subprocess.run([sys.executable, BRIDGE, *args], capture_output=True, text=True, env=e)

    def test_print_memory_is_exactly_22_4_67_outside_a_fabric_job(self):
        r = self.run_cli("print-memory", self.exec_path, self.exec_sha)
        self.assertEqual((r.returncode, r.stdout), (0, "22.4 67\n"), r.stderr)
        self.assertIn("SEPARATE 24 GiB", r.stderr)
        self.assertIn("25769803776", r.stderr)
        self.assertIn("NOT an RSS measurement", r.stderr)
        dev, host = self.base_memory()
        self.assertIn(f"base R7 host {host} GiB stays separate", r.stderr)
        self.assertNotEqual(r.stdout, f"{dev} {host}\n")

    def test_print_memory_refuses_effective_55_and_hash_or_term_mismatch(self):
        r = self.run_cli("print-memory", self.exec_path, "0" * 64)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "")
        for label, mutate in (("effective 55", lambda e: e["memory"].update({"host_gib": 55})),
                              ("ledger hash", lambda e: e["execution_r8"]["memory_ledger"].update({"sha256": "0" * 64}))):
            path, digest = self.mutated(mutate)
            r = self.run_cli("print-memory", path, digest)
            self.assertNotEqual(r.returncode, 0, label)
            self.assertEqual(r.stdout, "", label)
        led = ledger_obj()
        led["host"]["fixed_terms"]["ple_cache"] += 1
        jdump(self.ledger_path, led)
        self.exe["execution_r8"]["memory_ledger"]["sha256"] = sha(self.ledger_path)
        self.publish()
        r = self.run_cli("print-memory", self.exec_path, self.exec_sha)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "")
        self.assertIn("fixed term ple_cache", r.stderr)

    def test_resolve_prints_exactly_the_three_bound_paths(self):
        r = self.run_cli("resolve", self.exec_path, self.exec_sha)
        self.assertEqual(r.stdout.splitlines(), [R7, self.stage, self.handoff], r.stderr)

    def test_the_bridge_has_no_seal_operation(self):
        r = self.run_cli("seal", self.stage, self.a)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("def seal", rt(BRIDGE))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "a-out", "handoff-1a.json.tmp")))


class Receipt(Fixture):
    ARMS = ("greedy-beta", "greedy-s1")
    BETA_STDERR = ("qwen4_greedy_ids: routed experts: layers 0..5 in VRAM, 88 tensors host-mapped\n"
                   "  qwen4 host link: device 0 discrete, 7.1 GB/s H2D (narrow)\n"
                   "  qwen4 prefill chunk: 4096 rows (requested 4096)\n"
                   "  qwen4 long-prefill expert staging: forwards of >= 768 rows stage host expert layers 5..48 through resident layers 4 and 3 "
                   "(borrowed donor VRAM, no dedicated stage buffers; prefill chunk 4096)\n")

    def records(self, n=7, stop_first=True, tokens=32):
        out = []
        for i in range(n):
            if i == 0 and stop_first:
                out.append({"id": f"r{i}", "token_ids": [11, 12, 13, 14, 15], "finish_reason": "stop", "prompt_tokens": 10 + i})
            else:
                out.append({"id": f"r{i}", "token_ids": list(range(tokens)), "finish_reason": "length", "prompt_tokens": 10 + i})
        return out

    def command(self, arm, extra_env=None, drop=(), max_new="32", thinking="off", input_name=None):
        env = {"HIPFIRE_OFFLOAD_EXEC": "pcie"}
        env.update(br.BETA_PINS if arm == "greedy-beta" else br.CANDIDATE_PINS)
        env.update(extra_env or {})
        env = {k: v for k, v in env.items() if k not in drop and v is not None}
        name = input_name or ("prompts.short.beta.jsonl" if arm == "greedy-beta" else "prompts.short.jsonl")
        tokens = ["env", *[f"{k}={v}" for k, v in env.items()], "setsid", "-w", "timeout", "-k", "15", "600",
                  os.path.join(self.stage, "bin-beta" if arm == "greedy-beta" else "bin-s1", "qwen4_greedy_ids"), "--model", "/m.mq4",
                  "--input", os.path.join(self.stage, "fixtures", name), "--output", os.path.join(self.out, arm, "ids.jsonl")]
        if arm == "greedy-s1":
            tokens += ["--receipt", os.path.join(self.out, arm, "receipts.jsonl")]
        tokens += ["--max-new-tokens", max_new, "--max-seq", "16640", "--thinking", thinking]
        return shlex.join(tokens) + "\n"

    def build_b(self, beta_env=None, s1_env=None, beta_stderr=None, cmp_text="PASS: 7 records, exact ids, layer-major receipts valid\n", headroom="pass",
                beta_ids=None, s1_ids=None, beta_cmd=None, s1_cmd=None, initial=True, beta_rc=0, s1_rc=0, arm_status=None, skip_file=None):
        for n in os.listdir(self.out):
            if n != "hold.jsonl":
                p = os.path.join(self.out, n)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        os.makedirs(os.path.join(self.out, "a-handoff"))
        if initial is True:
            with open(os.path.join(self.out, "a-handoff", "stdout.txt"), "w") as f:
                f.write(json.dumps(self.verify(), indent=1))
        elif initial is not None:
            with open(os.path.join(self.out, "a-handoff", "stdout.txt"), "w") as f:
                f.write(initial)
        for arm in self.ARMS:
            d = os.path.join(self.out, arm)
            os.makedirs(d)
            pins = {"HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_QWEN4_D2_DEFAULT": "0", **(br.BETA_PINS if arm == "greedy-beta" else br.CANDIDATE_PINS)}
            for k, v in ((beta_env if arm == "greedy-beta" else s1_env) or {}).items():
                pins.pop(k, None)
                if v is not None:
                    pins[k] = v
            files = {"command.txt": (beta_cmd if arm == "greedy-beta" else s1_cmd) or self.command(arm), "stdout.txt": "", "evidence.json": "{}",
                     "rc.txt": f"{beta_rc if arm == 'greedy-beta' else s1_rc}\n", "spawn-env.json": json.dumps({"set": pins}),
                     "stderr.txt": (beta_stderr if beta_stderr is not None else self.BETA_STDERR) if arm == "greedy-beta" else "s1 stderr\n",
                     "ids.jsonl": "".join(json.dumps(r) + "\n" for r in ((beta_ids if arm == "greedy-beta" else s1_ids) or self.records()))}
            if arm == "greedy-s1":
                files["receipts.jsonl"] = "{}\n"
            if arm_status and arm_status[0] == arm:
                files["status.txt"] = arm_status[1]
            for fn, data in files.items():
                if skip_file != (arm, fn):
                    with open(os.path.join(d, fn), "w") as f:
                        f.write(data)
        os.makedirs(os.path.join(self.out, "headroom-state"))
        jdump(os.path.join(self.out, "headroom-state", "stdout.txt"), {"status": headroom, "gate": {"min_free_bytes_required": 536870912}})
        os.makedirs(os.path.join(self.out, "greedy-cmp"))
        with open(os.path.join(self.out, "greedy-cmp", "stdout.txt"), "w") as f:
            f.write(cmp_text)
        return self.out

    def make(self, b=None, **kw):
        b = b or self.build_b(**kw)
        with contextlib.redirect_stdout(io.StringIO()):
            br.receipt1b(self.stage, b, self.a, R7, self.exec_path, self.exec_sha)
        with open(os.path.join(b, br.RECEIPT_NAME)) as f:
            return json.load(f)

    def refuse(self, fragment="refusing 1b receipt", **kw):
        self.assertRefused(lambda: self.make(**kw), fragment)

    def test_valid_receipt_v2_binds_everything(self):
        before = self.dir_state(self.a)
        r = self.make()
        self.assertEqual(self.dir_state(self.a), before)
        self.assertEqual((r["schema"], r["hold"], r["advance_candidate"]), (br.RECEIPT_SCHEMA, "hold1b", True))
        self.assertEqual(r["a_handoff"]["schema"], br.VERIFY_SCHEMA)
        self.assertEqual(r["a_handoff"]["base_r7"]["verification_v1"], h1.verify(self.stage, self.a))
        self.assertEqual(r["a_handoff"]["base_r7"]["harness_files"], br.inventory(R7))
        self.assertEqual(r["base_r7"]["commit"], "9" * 40)
        self.assertEqual(r["execution_r8"]["commit"], "7" * 40)
        self.assertEqual(r["execution_r8"]["execution_json"]["sha256"], self.exec_sha)
        self.assertEqual(r["effective_memory_ledger"]["summary"]["reserved_bytes"], 71_940_702_208)
        self.assertEqual(r["effective_memory_ledger"]["sha256"], sha(self.ledger_path))
        self.assertEqual(r["base_memory_ledger"]["sha256"], sha(os.path.join(self.stage, "ledger-hold1.json")))
        self.assertNotEqual(r["effective_memory_ledger"]["sha256"], r["base_memory_ledger"]["sha256"])
        self.assertEqual(r["delta"]["beta"], br.BETA_PINS)
        self.assertTrue(r["delta"]["quality_reference_only"])
        self.assertEqual(r["proof"]["sha256"], sha(self.proof_path))
        self.assertEqual(r["run"], {"out": os.path.realpath(self.out), "fabric_job": self.job})
        init = os.path.join(self.out, "a-handoff", "stdout.txt")
        self.assertEqual(r["initial_verify"]["sha256"], sha(init))
        self.assertEqual(r["beta_observed"]["actual_chunk_rows"], 4096)
        self.assertEqual((r["beta_observed"]["resident_expert_layers"], r["beta_observed"]["stage_donors_observed"]), (5, [4, 3]))
        self.assertEqual(r["greedy_arms"]["greedy-beta"]["approved_pins"], br.BETA_PINS)
        self.assertEqual(r["greedy_arms"]["greedy-s1"]["approved_pins"], {"HIPFIRE_QWEN4_SLOWLINK_PREFILL": "1"})
        for arm in self.ARMS:  # every original common pin stays
            self.assertEqual(r["greedy_arms"][arm]["spawn_env_pin"], {"HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_QWEN4_D2_DEFAULT": "0"})
            self.assertEqual((r["greedy_arms"][arm]["arm"], r["greedy_arms"][arm]["rc"]), ("non_default_d2_off", 0))
        self.assertIn("receipts.jsonl", r["greedy_arms"]["greedy-s1"]["files"])
        gi = r["greedy_inputs"]
        self.assertEqual(gi["canonical"]["sha256"], sha(os.path.join(self.stage, "fixtures", "prompts.short.jsonl")))
        self.assertEqual(gi["beta"]["sha256"], sha(os.path.join(self.stage, "fixtures", "prompts.short.beta.jsonl")))
        self.assertEqual(gi["adapter_proof"]["sha256"], sha(os.path.join(self.stage, "fixtures", "beta-input-proof.json")))
        self.assertEqual(r["headroom_state"]["source"], "the immutable 1a RAW state cell (no second GPU state run)")
        self.assertEqual(r["greedy_cmp"]["sha256"], sha(os.path.join(self.out, "greedy-cmp", "stdout.txt")))
        self.assertEqual(r["label"], h1.beta_input.LABEL)
        sem = r["greedy_semantics"]["greedy-beta"]
        self.assertEqual(sem["max_new_tokens"], 32)
        self.assertEqual([x["n_tokens"] for x in sem["records"]], [5] + [32] * 6)  # EOS stop shorter than 32 is legal; no 32 is fabricated

    def test_the_original_receipt_schema_name_is_not_reused(self):
        r = self.make()
        self.assertNotEqual(r["schema"], h1.RECEIPT_SCHEMA)
        self.assertFalse(any(os.path.exists(os.path.join(self.out, n)) for n in ("receipt-1b.json.tmp",)))

    def test_exact_beta_pins_both_required_and_rejected_on_s1(self):
        R = br.RESIDENT_KEY
        C = br.CHUNK_KEY
        cases = {
            "beta resident 10 (the failed reference)": {"beta_env": {R: "10"}, "beta_cmd": self.command("greedy-beta", {R: "10"})},
            "beta resident pin missing": {"beta_env": {R: None}, "beta_cmd": self.command("greedy-beta", drop=(R,))},
            "beta chunk pin missing": {"beta_env": {C: None}, "beta_cmd": self.command("greedy-beta", drop=(C,))},
            "beta chunk 8192": {"beta_env": {C: "8192"}, "beta_cmd": self.command("greedy-beta", {C: "8192"})},
            "beta chunk 2048": {"beta_env": {C: "2048"}, "beta_cmd": self.command("greedy-beta", {C: "2048"})},
            "beta env set but command lacks it": {"beta_cmd": self.command("greedy-beta", drop=(C,))},
            "beta carries slowlink": {"beta_env": {br.SLOWLINK_KEY: "1"}},
            "S1 carries the resident pin": {"s1_env": {R: "5"}},
            "S1 carries the chunk pin": {"s1_env": {C: "4096"}},
            "S1 command carries the chunk pin": {"s1_cmd": self.command("greedy-s1", {C: "4096"})},
            "S1 lacks slowlink": {"s1_env": {br.SLOWLINK_KEY: None}, "s1_cmd": self.command("greedy-s1", drop=(br.SLOWLINK_KEY,))},
            "S1 slowlink 0": {"s1_env": {br.SLOWLINK_KEY: "0"}},
        }
        for label, kw in cases.items():
            with self.subTest(label):
                self.refuse(**kw)
        self.make()  # and the exact pair passes

    def test_actual_chunk_is_observed_in_beta_stderr_exactly_once_and_exactly_4096(self):
        ok = "  qwen4 prefill chunk: 4096 rows (requested 4096)"
        line = self.BETA_STDERR
        cases = {
            "downshift actual 2048": line.replace(ok, "  qwen4 prefill chunk: 2048 rows (requested 4096)"),
            "actual 1536": line.replace(ok, "  qwen4 prefill chunk: 1536 rows (requested 4096)"),
            "grown actual 8192": line.replace(ok, "  qwen4 prefill chunk: 8192 rows (requested 8192)"),
            "actual 4096 but requested 8192 (the failed run's shape)": line.replace(ok, "  qwen4 prefill chunk: 4096 rows (requested 8192)"),
            "missing line": line.replace(ok + "\n", ""),
            "ambiguous (two lines)": line + ok + "\n",
            "two lines disagree": line + "  qwen4 prefill chunk: 2048 rows (requested 4096)\n",
            "malformed": line.replace(ok, "  qwen4 prefill chunk: 4096 rows"),
            "resident 10 observed": line.replace("layers 0..5 in VRAM, 88", "layers 0..10 in VRAM, 78"),
            "routed line missing": line.replace("qwen4_greedy_ids: routed experts: layers 0..5 in VRAM, 88 tensors host-mapped\n", ""),
            "N5 donors [8, 9]": line.replace("layers 4 and 3", "layers 8 and 9"),
            "empty stderr": "",
        }
        for label, text in cases.items():
            with self.subTest(label):
                self.refuse(beta_stderr=text)
        self.make(beta_stderr=line.replace("layers 4 and 3", "layers 3 and 4"))  # parity order is the proof's concern; the donor SET {3, 4} is observed
        self.make(beta_stderr=line.replace("  qwen4 long-prefill expert staging", "  (no staging line)"))

    def test_every_original_common_pin_and_guard_is_preserved(self):
        pin = "HIPFIRE_OFFLOAD_EXEC"
        d2 = "HIPFIRE_QWEN4_D2_DEFAULT"
        cases = {
            "beta offload pin missing": {"beta_env": {pin: None}},
            "beta offload off": {"beta_env": {pin: "off"}},
            "S1 offload cpu": {"s1_env": {pin: "cpu"}},
            "S1 offload missing": {"s1_env": {pin: None}},
            "beta d2 missing": {"beta_env": {d2: None}},
            "S1 d2 missing": {"s1_env": {d2: None}},
            "beta d2 1": {"beta_env": {d2: "1"}},
            "S1 d2 1": {"s1_env": {d2: "1"}},
            "beta rc 1": {"beta_rc": 1},
            "beta timeout rc": {"beta_rc": 124},
            "S1 rc 1": {"s1_rc": 1},
            "beta lifecycle status.txt": {"arm_status": ("greedy-beta", "lifecycle: leftover='x' drained=false\n")},
            "S1 lifecycle status.txt": {"arm_status": ("greedy-s1", "lifecycle: leftover='x' drained=false\n")},
            "beta ids missing": {"skip_file": ("greedy-beta", "ids.jsonl")},
            "S1 receipts missing": {"skip_file": ("greedy-s1", "receipts.jsonl")},
            "S1 rc.txt missing": {"skip_file": ("greedy-s1", "rc.txt")},
            "beta spawn-env missing": {"skip_file": ("greedy-beta", "spawn-env.json")},
            "evidence missing": {"skip_file": ("greedy-beta", "evidence.json")},
            "compare FAIL": {"cmp_text": "FAIL\n"},
            "compare not all 7 records": {"cmp_text": "PASS: 6 records, exact ids\n"},
            "compare empty": {"cmp_text": ""},
            "headroom incomplete": {"headroom": "incomplete"},
            "headroom fail": {"headroom": "fail"},
        }
        for label, kw in cases.items():
            with self.subTest(label):
                self.refuse(**kw)

    def test_input_and_adapter_guards_are_preserved(self):
        self.build_b()
        fx = os.path.join(self.stage, "fixtures")
        for rel in ("beta-input-proof.json", "prompts.short.beta.jsonl", "prompts.short.jsonl"):
            with self.subTest(rel):
                path = os.path.join(fx, rel)
                original = rb(path)
                with open(path, "ab") as f:
                    f.write(b" \n" if rel.endswith("jsonl") else b" ")
                self.assertRaises(SystemExit, lambda: self.make(b=self.out))
                with open(path, "wb") as f:
                    f.write(original)
        self.make(b=self.out)

    def test_command_and_id_semantics_up_to_32_with_eos(self):
        cases = {
            "beta max-new-tokens 64": {"beta_cmd": self.command("greedy-beta", max_new="64")},
            "S1 max-new-tokens 31": {"s1_cmd": self.command("greedy-s1", max_new="31")},
            "S1 thinking on": {"s1_cmd": self.command("greedy-s1", thinking="on")},
            "beta reads the canonical input": {"beta_cmd": self.command("greedy-beta", input_name="prompts.short.jsonl")},
            "S1 reads the beta adapter": {"s1_cmd": self.command("greedy-s1", input_name="prompts.short.beta.jsonl")},
            "S1 without --receipt": {"s1_cmd": self.command("greedy-s1").replace("--receipt", "--nothing")},
            "33 ids": {"beta_ids": self.records(tokens=33)},
            "length with 31 ids (fabricated stop)": {"s1_ids": self.records(tokens=31)},
            "stop with 0 ids": {"beta_ids": [{**r, "token_ids": [], "finish_reason": "stop"} if i == 0 else r for i, r in enumerate(self.records())]},
            "unknown finish reason": {"beta_ids": [{**r, "finish_reason": "eos"} if i == 0 else r for i, r in enumerate(self.records())]},
            "six records": {"beta_ids": self.records(n=6)},
            "non-int token": {"s1_ids": [{**r, "token_ids": ["a"]} if i == 1 else r for i, r in enumerate(self.records())]},
        }
        for label, kw in cases.items():
            with self.subTest(label):
                self.refuse(**kw)
        r = self.make(beta_ids=self.records(stop_first=False), s1_ids=self.records(stop_first=True))  # all-32 and EOS-shortened are both legal
        self.assertEqual([x["n_tokens"] for x in r["greedy_semantics"]["greedy-s1"]["records"]][:2], [5, 32])

    def test_initial_a_handoff_stdout_must_equal_the_reverification(self):
        good = json.dumps(self.verify(), indent=1)
        mutate = json.loads(good)
        mutate["execution_r8"]["commit"] = "1" * 40
        other_run = json.loads(good)
        other_run["run"]["fabric_job"] = "j-other"
        other_exec = json.loads(good)
        other_exec["execution_r8"]["execution_json"]["sha256"] = "0" * 64
        for label, text in (("missing stdout", None), ("not json", "FATAL"), ("different execution identity", json.dumps(mutate)),
                            ("different fabric job", json.dumps(other_run)), ("different execution json hash", json.dumps(other_exec)),
                            ("v1 verification instead (an unbound old verify)", json.dumps(h1.verify(self.stage, self.a)))):
            with self.subTest(label):
                self.refuse(initial=text)
        self.make()

    def test_receipt_reverifies_h1a_and_the_binding_now(self):
        self.build_b()
        with open(os.path.join(self.a, "state", "vram.csv"), "a") as f:
            f.write("tamper\n")
        self.assertRefused(lambda: self.make(b=self.out), "changed since the seal")
        self.setUp()  # fresh world
        self.build_b()
        with mock.patch.dict(os.environ, {"FABRIC_JOB": "j-other"}):
            self.assertRefused(lambda: self.make(b=self.out), "does not name hold1b")
        other = os.path.join(self.tmp, "elsewhere")
        os.makedirs(other)
        self.assertRefused(lambda: self.make(b=other), "")
        with open(os.path.join(self.out, "greedy-cmp", "stdout.txt"), "w") as f:
            f.write("PASS: 7 records\n")
        self.make(b=self.out)
        self.assertRefused(lambda: self.make(b=self.out), "already has receipt-1b.json")  # a finished OUT is never reused

    def test_receipt_never_calls_the_original_receipt_writer(self):
        src = rt(BRIDGE)
        self.assertNotIn("orig.receipt1b", src)
        self.assertNotIn(".receipt1b(stage", src.replace("def receipt1b(stage", ""))
        self.assertNotIn("orig.seal", src)
        self.assertNotIn("orig.RECEIPT_SCHEMA", src)


class Structure(unittest.TestCase):
    def text(self, name=None):
        with open(os.path.join(HERE, name or "hold1b-correct.sh")) as f:
            return f.read()

    def code(self):
        return [l for l in self.text().splitlines() if l.strip() and not l.lstrip().startswith("#")]

    def test_dedicated_directory_has_exactly_the_four_files_and_nothing_of_r7_is_copied(self):
        names = sorted(n for n in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, n)) and not n.endswith(".pyc"))
        self.assertEqual(names, sorted(br.R8_FILES))
        for n in br.R8_FILES:
            self.assertNotIn(n, os.listdir(R7) if n != "hold1b-correct.sh" else [], n)  # only the entry script shares a name with an R7 file
        self.assertTrue(os.access(SCRIPT, os.X_OK))

    def test_bash_syntax_and_python_ast(self):
        r = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        import ast
        for n in ("h1b_bridge.py", "prove_beta_math.py", "test_h1_handoff.py"):
            ast.parse(self.text(n))

    def test_required_rows_scope_and_budgets_are_unchanged_from_r7(self):
        m = re.search(r'^REQUIRED="([^"]*)"', self.text(), re.M)
        self.assertEqual(m.group(1).split(), ["a-handoff", "headroom-state", "beta-input", "greedy-beta", "greedy-s1", "greedy-cmp", "receipt-1b"])
        body = "\n".join(self.code())
        self.assertIn("HOLD_SCOPE=h1_complete_on_verified_1a", body)
        self.assertIn("HOLD_ADVANCE=true", body)
        self.assertIn("hold_init hold1b 1320", body)
        self.assertIn("cell greedy-beta 600 600 ", body)
        self.assertIn("cell greedy-s1 390 390 ", body)
        self.assertEqual(600 + 390 + 2 * 15 + 2 * 90 + 45 + 30, 1275)
        self.assertLessEqual(1275, 1320)
        self.assertEqual(len([l for l in self.code() if l.startswith("cell ")]), 2)
        for forbidden in ("cell prim", "cell state", "--mode state", "--mode primitives"):
            self.assertNotIn(forbidden, body)
        self.assertEqual(len([l for l in self.code() if l.startswith("headroom ")]), 1)
        self.assertIn('headroom headroom-state "$A_DIR/state" "$A_DIR/prim"', body)

    def test_cells_carry_exactly_the_approved_pins_and_same_options(self):
        lines = self.text().splitlines()
        beta = next(i for i, l in enumerate(lines) if l.startswith("cell greedy-beta "))
        s1 = next(i for i, l in enumerate(lines) if l.startswith("cell greedy-s1 "))
        beta_cmd, s1_cmd = " ".join(lines[beta:beta + 3]), " ".join(lines[s1:s1 + 3])
        for needle in ('"$H1_OFFLOAD_PIN"', "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=5", "HIPFIRE_PREFILL_CHUNK_ROWS=4096", '"$S/bin-beta/qwen4_greedy_ids"',
                       '--input "$S/fixtures/prompts.short.beta.jsonl"', "--max-new-tokens 32", "--thinking off", '--max-seq "$MAX_SEQ"'):
            self.assertIn(needle, beta_cmd, needle)
        self.assertNotIn("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=10", self.text().replace("(was 10)", ""))
        for needle in ('"$H1_OFFLOAD_PIN"', "HIPFIRE_QWEN4_SLOWLINK_PREFILL=1", '"$S/bin-s1/qwen4_greedy_ids"', '--input "$S/fixtures/prompts.short.jsonl"',
                       '--receipt "$OUT/greedy-s1/receipts.jsonl"', "--max-new-tokens 32", "--thinking off"):
            self.assertIn(needle, s1_cmd, needle)
        for forbidden in ("EXPERT_VRAM_LAYERS", "PREFILL_CHUNK_ROWS", "D2_DEFAULT"):
            self.assertNotIn(forbidden, s1_cmd, forbidden)
        self.assertNotIn("D2_DEFAULT", beta_cmd)  # the D2 metadata pin stays the shared cell_env pin, not a per-cell override
        self.assertNotIn("--max-new-tokens 33", self.text())

    def test_libraries_are_sourced_only_from_the_validated_r7_path_after_resolve(self):
        code = self.code()
        resolve = next(i for i, l in enumerate(code) if 'h1b_bridge.py" resolve' in l)
        lib = next(i for i, l in enumerate(code) if l.startswith('. "$R7_HOLDS/hold-lib.sh"'))
        common = next(i for i, l in enumerate(code) if l.startswith('. "$HOLDS/hold1-common.sh"'))
        init = next(i for i, l in enumerate(code) if l.startswith("hold_init "))
        self.assertLess(resolve, lib)
        self.assertLess(lib, common)
        self.assertLess(common, init)
        self.assertEqual(len([l for l in code if l.startswith(". ")]), 2)
        body = "\n".join(code)
        self.assertNotIn('. "$R8_HOLDS', body)
        self.assertNotIn("source ", body)
        self.assertIn('[ "$HOLDS" = "$R7_HOLDS" ]', body)
        self.assertIn('[ "$(readlink -f "$STAGE")" = "$EXEC_STAGE" ]', body)
        self.assertIn("export OUT", body)
        # R7 helpers stay the frozen R7 files (HOLDS truthful); only the bridge is R8
        for helper in ("check_headroom", "beta_input.py", "verify_greedy.py"):
            for l in code:
                if helper in l and "python3" in l:
                    self.assertIn('"$HOLDS/', l)
        self.assertEqual(len([l for l in code if "h1b_bridge.py" in l]), 4)  # --print-memory, resolve, verify, receipt1b
        self.assertNotIn("h1_handoff.py", "\n".join(l for l in code if "python3" in l))
        self.assertIn("S1_HOLD1A_OUT names another 1a", body)

    def test_the_handoff_is_verified_before_any_gpu_cell_and_blocks_on_failure(self):
        code = self.code()
        verify = next(i for i, l in enumerate(code) if l.startswith("check a-handoff"))
        first_cell = next(i for i, l in enumerate(code) if l.startswith("cell "))
        headroom = next(i for i, l in enumerate(code) if l.startswith("headroom headroom-state"))
        self.assertLess(verify, headroom)
        self.assertLess(headroom, first_cell)
        self.assertIn("BLOCKED=", "\n".join(code[verify:headroom]))
        self.assertIn('verify "$S" "$A_OUT" "$HOLDS" "$EXEC_JSON" "$EXEC_SHA"', code[verify])
        self.assertIn("beta-input", "\n".join(code[headroom:first_cell]))
        self.assertTrue(code[-1].startswith("finish_summary"))
        self.assertIn('receipt1b "$S" "$OUT" "$A_OUT" "$HOLDS" "$EXEC_JSON" "$EXEC_SHA"', "\n".join(code))

    def test_print_memory_and_bad_sha_run_before_anything_is_sourced(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = os.path.join(tmp, "execution.json")
            with open(bogus, "w") as f:
                f.write("{}")
            r = subprocess.run(["bash", SCRIPT, bogus, "0" * 64], capture_output=True, text=True, env={**os.environ, "S1_STAGE": tmp})
            self.assertEqual(r.returncode, 2)
            self.assertIn("nothing sourced", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "runs")))
            r = subprocess.run(["bash", SCRIPT, "--print-memory", bogus], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("usage", r.stderr)
            r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)


class ShellPrintMemory(Fixture):
    def test_shell_print_memory_is_exactly_22_4_67(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FABRIC_") and k != "OUT"}
        r = subprocess.run(["bash", SCRIPT, "--print-memory", self.exec_path, self.exec_sha], capture_output=True, text=True, env=env)
        self.assertEqual((r.returncode, r.stdout), (0, "22.4 67\n"), r.stderr)
        self.assertIn("SEPARATE 24 GiB", r.stderr)
        r = subprocess.run(["bash", SCRIPT, "--print-memory", self.exec_path, "1" * 64], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "")

    def test_shell_refuses_before_any_stage_use_when_the_execution_is_tampered(self):
        with open(self.diff_path, "a") as f:
            f.write("tampered\n")
        env = {**os.environ, "S1_STAGE": self.stage}
        r = subprocess.run(["bash", SCRIPT, self.exec_path, self.exec_sha], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("nothing sourced", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.stage, "runs")))


class ShellFreshOut(Fixture):
    """The shell entry validates OUT BEFORE the frozen hold_init (which mkdirs OUT): protected directories must never be written."""

    def run_entry(self, out):
        env = {k: v for k, v in os.environ.items() if k != "OUT"}
        env["S1_STAGE"] = self.stage
        if out is not None:
            env["OUT"] = out
        return subprocess.run(["bash", SCRIPT, self.exec_path, self.exec_sha], capture_output=True, text=True, env=env, timeout=120)

    def snapshot(self):
        return {d: self.dir_state(d) for d in (self.a, self.stage, R7, HERE)} | {"tmp": sorted(os.listdir(self.tmp))}

    def test_unsafe_out_is_refused_before_anything_is_created_or_written(self):
        link_new = os.path.join(self.tmp, "link-to-new")
        os.symlink(os.path.join(self.tmp, "link-target-new"), link_new)
        link_existing = os.path.join(self.tmp, "link-to-existing")
        os.symlink(self.out, link_existing)
        existing_file = os.path.join(self.tmp, "afile")
        with open(existing_file, "w"):
            pass
        cases = {
            "unset": None,
            "empty": "",
            "relative": "relative-out",
            "existing empty directory": self.out,
            "existing H1a state cell": os.path.join(self.a, "state"),
            "existing H1a prim cell": os.path.join(self.a, "prim"),
            "the H1a run directory itself": self.a,
            "nonexistent inside the H1a run directory": os.path.join(self.a, "new-out"),
            "nonexistent inside an H1a cell": os.path.join(self.a, "state", "new-out"),
            "the stage": self.stage,
            "nonexistent inside the stage": os.path.join(self.stage, "runs", "x"),
            "the base R7 directory": R7,
            "nonexistent inside the base R7 directory": os.path.join(R7, "runs-x"),
            "the R8 directory": HERE,
            "nonexistent inside the R8 directory": os.path.join(HERE, "runs-x"),
            "dangling symlink": link_new,
            "symlink to an existing directory": link_existing,
            "existing file": existing_file,
            "dotted path": os.path.join(self.tmp, "x", "..", "dotted-new"),
            "trailing slash": os.path.join(self.tmp, "slash-new") + "/",
            "symlinked parent": os.path.join(link_existing, "child-new"),
        }
        for label, out in cases.items():
            with self.subTest(label):
                before = self.snapshot()
                r = self.run_entry(out)
                self.assertEqual(r.returncode, 2, (r.stdout, r.stderr))
                self.assertIn("nothing was created", r.stderr)
                self.assertNotIn("FATAL.txt", r.stderr)
                self.assertEqual(self.snapshot(), before, "no file under A/stage/R7/R8/tmp may change; no FATAL.txt, hold.jsonl or directory is created")
                self.assertFalse(os.path.exists(os.path.join(self.a, "state", "hold.jsonl")))
                self.assertFalse(os.path.exists(os.path.join(self.out, "FATAL.txt")))

    def test_a_safe_fresh_out_passes_the_guard_and_proceeds_to_the_frozen_hold_init(self):
        out = os.path.join(self.tmp, "fresh-run-out")
        r = self.run_entry(out)
        # hold_init then refuses (not inside a fabric job / no target): the guard itself did not trip
        self.assertNotIn("a fresh explicit absolute nonexistent run directory", r.stderr)

    def test_the_guard_precedes_hold_init_in_the_script(self):
        with open(SCRIPT) as f:
            code = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        guard = next(i for i, l in enumerate(code) if l.startswith("out_refuse()"))
        init = next(i for i, l in enumerate(code) if l.startswith("hold_init "))
        self.assertLess(guard, init)
        body = "\n".join(code[guard:init])
        for needle in ('[ -n "${OUT:-}" ]', 'case "$OUT" in /*)', "readlink -m", '[ ! -L "$OUT" ]', "exit 2"):
            self.assertIn(needle, body)
        self.assertNotIn("die ", body.replace("# ", ""))
        self.assertNotIn("mkdir", body)


class Oracle(unittest.TestCase):
    """The same-class source oracle's model (pure functions) and, when the exact ff799 source is available, its real attestation."""

    def test_donor_parity_n5_is_4_3_not_3_4_and_n10_is_8_9(self):
        self.assertEqual(pbm.donors(5), [4, 3])
        self.assertNotEqual(pbm.donors(5), [3, 4])
        self.assertEqual(pbm.donors(10), [8, 9])

    def test_chunk_ladder_and_schedules(self):
        self.assertEqual(pbm.rungs(4096), [4096, 2048, 1536])
        self.assertEqual(pbm.rungs(8192), [8192, 4096, 2048, 1536])
        self.assertEqual(pbm.schedule(4100, 4096), [4096, 4])
        self.assertEqual(pbm.schedule(4100, 8192), [4100])
        self.assertEqual(pbm.schedule(8192, 4096), [4096, 4096])
        for n in (512, 1536, 1537, 3400, 582):
            self.assertEqual(pbm.schedule(n, 4096), [n])

    def test_kernel_classes_by_actual_rows(self):
        gate = lambda r: pbm.kernel_class(r)[0]  # noqa: E731
        down = lambda r: pbm.kernel_class(r)[1]  # noqa: E731
        self.assertEqual(gate(1), "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched")
        self.assertEqual(gate(2), "gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4")
        self.assertEqual(gate(8), gate(2))
        self.assertEqual(gate(9), "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4")
        self.assertEqual(gate(511), gate(9))
        self.assertEqual(down(9), "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151")
        self.assertEqual(down(2), "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded")
        for rows in (512, 582, 768, 4096):
            self.assertEqual(pbm.kernel_class(rows), ("gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out", "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out",
                                                      "f16_dequant_f32_wmma_acc_bf16_rows"))
        self.assertEqual(pbm.kernel_class(511)[2], "f32_dequant_f32_arith")

    def test_fixed_six_cases_are_same_class_for_n10_and_n5_at_actual_4096(self):
        for name, length in pbm.CASES:
            with self.subTest(name):
                row, t10, t5 = pbm.case_row(name, length, 4096, 4096, 10, 5)
                self.assertTrue(row["same_class_per_layer_and_forward"])
                self.assertEqual(pbm.class_projection(t10), pbm.class_projection(t5))
                self.assertEqual(row["singleton_continuation_opportunities"]["count_upper_bound"], 32)
                self.assertEqual(row["singleton_continuation_opportunities"]["applies_to_layers"], list(range(48)))
        # ownership, base, donor and copy events are the ONLY permitted differences
        _, t10, t5 = pbm.case_row("p8192", 8192, 4096, 4096, 10, 5)
        l10, l5 = t10[0]["layers"], t5[0]["layers"]
        self.assertNotEqual([x["owner"] for x in l10], [x["owner"] for x in l5])
        self.assertEqual([x["tuple"] for x in l10], [x["tuple"] for x in l5])
        self.assertEqual(l5[7]["copy"], {"read": "byte_copied_donor_stage", "donor": pbm.donors(5)[7 % 2]})
        self.assertEqual(l10[7]["copy"], {"read": "own_resident_bytes", "donor": None})
        _, s10, s5 = pbm.case_row("p512", 512, 4096, 4096, 10, 5)
        self.assertEqual(s5[0]["layers"][7]["copy"], {"read": "host_mapped_alias", "donor": None})  # below 768 rows: the mapped alias, same kernel

    def test_irregular_4100_is_not_the_same_class_with_an_8192_ceiling(self):
        a = [pbm.kernel_class(r) for r in pbm.schedule(4100, 4096)]
        b = [pbm.kernel_class(r) for r in pbm.schedule(4100, 8192)]
        self.assertNotEqual(a, b)
        self.assertEqual(a[1][2], "f32_dequant_f32_arith")  # the four-row F32 tail
        self.assertEqual(b[0][2], "f16_dequant_f32_wmma_acc_bf16_rows")
        with self.assertRaises(SystemExit):
            pbm.case_row("negative", 4100, 8192, 4096, 10, 5)  # N5 at a grown chunk against the N10 reference: refused

    def test_source_counted_relief_and_growth(self):
        per_layer = pbm.EXPERTS * (pbm.GATE_UP_BYTES + pbm.DOWN_BYTES)
        self.assertEqual(per_layer, 1_336_934_400)
        self.assertEqual(5 * per_layer, 6_684_672_000)
        self.assertEqual(5 * per_layer + 10 * 2**20, 6_695_157_760)
        self.assertEqual((pbm.DEVICE_RELIEF_BYTES, pbm.HOST_GROWTH_BYTES), (6_684_672_000, 6_695_157_760))

    def test_pins_cover_every_consulted_source_file_and_anchors_name_pinned_files(self):
        self.assertEqual(pbm.BETA_COMMIT, "ff799626eb3dddf8ce827a90d88c990fc0eb4aa2")
        for _, rel, first, last, needles in pbm.ANCHORS:
            self.assertIn(rel, pbm.PINS)
            self.assertLessEqual(first, last)
            self.assertTrue(needles)
        for c in pbm.MODEL_CONSTANT_ANCHORS.values():
            names = {a[0] for a in pbm.ANCHORS}
            for n in c.split(","):
                self.assertIn(n, names)

    def test_source_drift_fails_closed_on_a_synthetic_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(pbm.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="0" * 40 + "\n", stderr="")):
                self.assertRaises(SystemExit, lambda: pbm.attest_source(tmp))  # wrong commit
            with mock.patch.object(pbm.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=pbm.BETA_COMMIT + "\n", stderr="")):
                with self.assertRaises(SystemExit) as cm:
                    pbm.attest_source(tmp)  # right commit, pinned files absent
                self.assertIn("missing", str(cm.exception))
                first = next(iter(pbm.PINS))
                os.makedirs(os.path.dirname(os.path.join(tmp, first)))
                with open(os.path.join(tmp, first), "w") as f:
                    f.write("drifted")
                with self.assertRaises(SystemExit) as cm:
                    pbm.attest_source(tmp)
                self.assertIn("source drift", str(cm.exception))

    @unittest.skipUnless(BETA_SOURCE and os.path.isdir(BETA_SOURCE), "exact ff799 source not available (set S1_BETA_SOURCE)")
    def test_real_source_attestation_full_proof_and_anchor_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "proof.json")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pbm.main(["prove_beta_math.py", BETA_SOURCE, out]), 0)
            proof = json.loads(rt(out))
            self.assertEqual(proof["schema"], br.PROOF_SCHEMA)
            br.validate_proof(proof)  # the bridge accepts exactly what the oracle emits
            self.assertEqual(proof["donors"], {"n10": [8, 9], "n5": [4, 3], "n5_is_not": [3, 4], "rule": "[first_host-2, first_host-1] stably sorted by layer % 2"})
            self.assertEqual(len(proof["cases"]), 6)
            self.assertTrue(proof["natural"]["conditional"])
            self.assertEqual(proof["natural"]["assumed_prompt_tokens"], 582)
            self.assertEqual(proof["general_admissible_natural_length_property"]["count"], pbm.MAX_SEQ)
            self.assertTrue(proof["negative_8192_ceiling"]["class_changed"])
            self.assertEqual(proof["negative_8192_ceiling"]["chunk4096_rows"], [4096, 4])
            self.assertEqual(proof["negative_8192_ceiling"]["chunk8192_rows"], [4100])
            self.assertFalse(proof["d2_default"]["selector_present_in_beta_source"])
            self.assertEqual(proof["memory"]["device_payload_relief_bytes"], 6_684_672_000)
            self.assertEqual(proof["memory"]["host_growth_bytes_including_10_pads"], 6_695_157_760)
            self.assertEqual((proof["claims"]["measured_fit"], proof["claims"]["generated_output_equality"], proof["claims"]["generated_32_tokens"]),
                             (False, False, False))
            # anchor drift: a copy of the pinned files with one asserted expression removed (pins re-computed so only the ANCHOR can catch it)
            copy_root = os.path.join(tmp, "src")
            for rel in pbm.PINS:
                os.makedirs(os.path.dirname(os.path.join(copy_root, rel)), exist_ok=True)
                shutil.copy(os.path.join(BETA_SOURCE, rel), os.path.join(copy_root, rel))
            target = os.path.join(copy_root, pbm.GF)
            text = rt(target)
            self.assertIn("donors.sort_by_key(|layer| layer % 2);", text)
            with open(target, "w") as f:
                f.write(text.replace("donors.sort_by_key(|layer| layer % 2);", "donors.sort_by_key(|layer| layer % 3);"))
            pins = dict(pbm.PINS)
            pins[pbm.GF] = sha(target)
            with mock.patch.object(pbm, "PINS", pins), mock.patch.object(pbm.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=pbm.BETA_COMMIT + "\n", stderr="")):
                with self.assertRaises(SystemExit) as cm:
                    pbm.attest_source(copy_root)
                self.assertIn("source anchor stage_donor_parity", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
