#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the declared-target inputs: stage_build.py pack-architecture derivation, make_manifest.py arch/reservation parsing and
hold-lib.sh load_target / hold_memory. Synthetic temp files only: no GPU, no model, no staged binary is ever executed.

    python3 scripts/slowlink-holds/test_target_inputs.py -v
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_manifest as mm  # noqa: E402
import stage_build as sb  # noqa: E402
import ledger_testlib as lt  # noqa: E402

TIP = "a" * 40
BASE = "b" * 40
TREE = "c" * 40
LIB = os.path.join(HERE, "hold-lib.sh")


def wjson(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def add(tar, name, data=b"x", type=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type = type
    info.linkname = linkname
    if type == tarfile.REGTYPE:
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    else:
        tar.addfile(info)


def modules_members(arch, modules=("alpha", "beta_2"), object_ok=True):
    """(name, bytes) members of one coherent pack: manifest.json + <arch>/<module>.{hsaco,hash,index.json}."""
    out = [("manifest.json", json.dumps({"commit": TIP, "arch": arch, "format": 1, "modules": len(modules)}).encode())]
    for mod in modules:
        obj = f"object-{arch}-{mod}".encode()
        index = {"module": mod, "arch": arch, "object_sha256": hashlib.sha256(obj if object_ok else b"other").hexdigest(),
                 "source_sha256": "d" * 64}
        out += [(f"{arch}/{mod}.hsaco", obj), (f"{arch}/{mod}.hash", b"h"), (f"{arch}/{mod}.index.json", json.dumps(index).encode())]
    return out


class Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def pack(self, members, name="pack.tar.gz", extra=()):
        p = self.path(name)
        with tarfile.open(p, "w:gz") as tar:
            for n, data in members:
                add(tar, n, data)
            for args in extra:
                add(tar, *args[:1], **args[1])
        return p


class SafeMembers(Tmp):
    def arch_of(self, members, extra=()):
        with tarfile.open(self.pack(members, extra=extra), "r:*") as tar:
            keep, arch = sb.safe_members(tar)
        return arch, sorted(n for _, n in keep)

    def refused(self, members, extra=(), fragment=""):
        with self.assertRaises(SystemExit) as cm:
            self.arch_of(members, extra)
        self.assertIn(fragment, str(cm.exception))

    def test_arch_is_derived_from_the_members_not_a_constant(self):
        for arch in ("gfx1100", "gfx90a", "gfx1201"):
            with self.subTest(arch=arch):
                got, names = self.arch_of(modules_members(arch))
                self.assertEqual(got, arch)
                self.assertIn("manifest.json", names)
                self.assertTrue(all(n == "manifest.json" or n.startswith(arch + "/") for n in names))

    def test_directory_entry_of_the_arch_is_allowed(self):
        self.assertEqual(self.arch_of(modules_members("gfx90a"), extra=[("gfx90a", {"type": tarfile.DIRTYPE})])[0], "gfx90a")

    def test_mixed_architecture_directories_rejected(self):
        self.refused(modules_members("gfx1100") + [("gfx90a/m.hsaco", b"x")], fragment="mixes architecture")

    def test_second_empty_architecture_directory_rejected(self):
        self.refused(modules_members("gfx1100"), extra=[("gfx90a", {"type": tarfile.DIRTYPE})], fragment="beside architecture")

    def test_unsafe_paths_rejected(self):
        for name, fragment in (("../gfx1100/m.hsaco", "unsafe tar member path"), ("/gfx1100/m.hsaco", "unsafe tar member path"),
                               ("gfx1100/../m.hsaco", "unsafe tar member path"), ("gfx1100/sub/m.hsaco", "outside manifest.json"),
                               ("gfx1100/m.txt", "not <module>"), ("gfx1100/bad-name.hsaco", "not <module>"),
                               ("top.hsaco", "outside manifest.json"), ("a b/m.hsaco", "outside manifest.json")):
            with self.subTest(name=name):
                self.refused([("manifest.json", b"{}"), (name, b"x")], fragment=fragment)

    def test_non_regular_members_rejected(self):
        self.refused([("manifest.json", b"{}"), ("gfx1100/m.hsaco", b"x")],
                     extra=[("gfx1100/link.hsaco", {"type": tarfile.SYMTYPE, "linkname": "/etc/passwd"})], fragment="not a regular file")

    def test_duplicate_member_rejected(self):
        self.refused(modules_members("gfx1100") + [("gfx1100/alpha.hsaco", b"again")], fragment="duplicate")

    def test_pack_without_modules_rejected(self):
        self.refused([("manifest.json", b"{}")], fragment="no <arch>/<module> members")


@unittest.skipUnless(hasattr(tarfile, "data_filter"), "needs tarfile data filter")
class StageBuild(Tmp):
    def setUp(self):
        super().setUp()
        rel = self.path("target", "release")
        for item in sb.BINARIES["s1"]:
            p = os.path.join(rel, item)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write("#!/bin/sh\n# synthetic, never executed\n")
            os.chmod(p, 0o755)
        self.stage = self.path("stage")
        os.makedirs(self.stage)

    def stage_pack(self, pack, tip=TIP):
        argv = ["stage_build.py", "--stage", self.stage, "--arm", "s1", "--source-base", BASE, "--source-tip", tip, "--source-tree", TREE,
                "--target-dir", self.path("target"), "--kernel-pack", pack]
        with contextlib.redirect_stdout(io.StringIO()):
            sb.main(argv)

    def identity(self):
        with open(os.path.join(self.stage, "bin-s1", "BUILD-IDENTITY.json")) as f:
            return json.load(f)

    def test_actual_gfx1100_pack_is_still_sealed(self):
        self.stage_pack(self.pack(modules_members("gfx1100")))
        bi = self.identity()
        self.assertEqual(bi["kernel_arch"], "gfx1100")
        self.assertEqual(sorted(bi["kernel_code_object_hashes"]), ["alpha", "beta_2"])
        self.assertEqual(bi["kernel_code_object_hashes"]["alpha"], hashlib.sha256(b"object-gfx1100-alpha").hexdigest())
        self.assertTrue(os.path.isfile(os.path.join(self.stage, "bin-s1", "kernels", "compiled", "gfx1100", "alpha.hsaco")))

    def test_other_coherent_arch_pack_accepted_by_metadata(self):
        self.stage_pack(self.pack(modules_members("gfx90a")))
        bi = self.identity()
        self.assertEqual(bi["kernel_arch"], "gfx90a")
        self.assertEqual(bi["pack"]["manifest"]["arch"], "gfx90a")
        compiled = os.path.join(self.stage, "bin-s1", "kernels", "compiled")
        self.assertEqual(os.listdir(compiled), ["gfx90a"])

    def test_manifest_arch_must_match_member_directory(self):
        members = modules_members("gfx90a")
        members[0] = ("manifest.json", json.dumps({"commit": TIP, "arch": "gfx1100", "format": 1, "modules": 2}).encode())
        with self.assertRaises(SystemExit) as cm:
            self.stage_pack(self.pack(members))
        self.assertIn("manifest.arch", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.stage, "bin-s1")))

    def test_index_arch_must_match_pack_arch(self):
        members = modules_members("gfx90a")
        index = json.loads(dict(members)["gfx90a/alpha.index.json"])
        index["arch"] = "gfx1100"
        members = [(n, json.dumps(index).encode() if n == "gfx90a/alpha.index.json" else d) for n, d in members]
        with self.assertRaises(SystemExit) as cm:
            self.stage_pack(self.pack(members))
        self.assertIn("alpha.index.json names", str(cm.exception))

    def test_code_object_hash_mismatch_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            self.stage_pack(self.pack(modules_members("gfx90a", object_ok=False)))
        self.assertIn("object sha256", str(cm.exception))

    def test_mixed_pack_rejected_before_staging(self):
        with self.assertRaises(SystemExit) as cm:
            self.stage_pack(self.pack(modules_members("gfx1100") + [("gfx90a/extra.hsaco", b"x")]))
        self.assertIn("mixes architecture", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.stage, "bin-s1")))


class MakeManifestInputs(unittest.TestCase):
    def bi(self, arch="gfx1100", **extra):
        return {"pack": {"manifest": {"arch": arch}}, **extra}

    def test_pack_arch_comes_from_the_sealed_identity(self):
        for arch in ("gfx1100", "gfx90a"):
            self.assertEqual(mm.pack_arch(self.bi(arch)), arch)
            self.assertEqual(mm.pack_arch(self.bi(arch, kernel_arch=arch)), arch)

    def test_pack_arch_rejects_missing_malformed_or_disagreeing(self):
        for bi in ({}, {"pack": {}}, {"pack": None}, self.bi(None), self.bi(""), self.bi("gfx/1100"), self.bi("../x"), self.bi(1100),
                   self.bi("gfx1100", kernel_arch="gfx90a")):
            with self.subTest(bi=bi), self.assertRaises(SystemExit):
                mm.pack_arch(bi)

    def test_fabric_reservation_is_the_ledger_field(self):
        self.assertEqual(mm.fabric_reservation_gib({"fabric_reservation_device_gib": 22.4}), 22.4)
        self.assertEqual(mm.fabric_reservation_gib({"fabric_reservation_device_gib": 18}), 18)

    def test_fabric_reservation_missing_or_invalid_rejected(self):
        for v in (None, 0, -1, True, "22.4", float("nan"), float("inf"), [22.4]):
            with self.subTest(v=v), self.assertRaises(SystemExit):
                mm.fabric_reservation_gib({"fabric_reservation_device_gib": v})
        with self.assertRaises(SystemExit):
            mm.fabric_reservation_gib({"card_bytes": 25769803776})


class ManifestIdentity(unittest.TestCase):
    GATE = "e" * 64

    def bi(self, **changes):
        out = {"schema": "s1.build_identity.v1", "arm": "s1", "source": {"base": BASE, "tip": TIP, "tree": TREE},
               "binary_sha256": {"qwen4_slowlink_gate": self.GATE}}
        out.update(changes)
        return out

    def test_supplied_base_tip_must_match_and_the_tree_comes_from_the_identity(self):
        self.assertEqual(mm.check_build_identity(self.bi(), BASE, TIP, self.GATE), TREE)
        for base, tip in (("d" * 40, TIP), (BASE, "d" * 40)):
            with self.subTest(base=base, tip=tip), self.assertRaises(SystemExit):
                mm.check_build_identity(self.bi(), base, tip, self.GATE)

    def test_identity_shape_and_gate_hash_are_enforced(self):
        for bi in (self.bi(arm="beta"), self.bi(schema="x"), self.bi(source=None), self.bi(source={"base": BASE, "tip": TIP}),
                   self.bi(source={"base": BASE, "tip": TIP, "tree": "T" * 40}), self.bi(binary_sha256={}), []):
            with self.subTest(bi=bi), self.assertRaises(SystemExit):
                mm.check_build_identity(bi, BASE, TIP, self.GATE)
        with self.assertRaises(SystemExit):
            mm.check_build_identity(self.bi(), BASE, TIP, "f" * 64)

    def test_ledger_provenance_is_preserved_and_must_name_the_same_build(self):
        prov = {"census_source_commit": TIP, "census_source_tree": TREE, "census_source_base": BASE, "binary_sha256": self.GATE,
                "harness_commit": "9" * 40}
        self.assertEqual(mm.check_ledger_provenance(prov, BASE, TIP, TREE, self.GATE), prov)
        self.assertEqual(mm.check_ledger_provenance({"model_config_sha256": "x"}, BASE, TIP, TREE, self.GATE), {"model_config_sha256": "x"})
        for key in ("census_source_commit", "census_source_tree", "census_source_base", "binary_sha256", "runtime_commit", "runtime_tree"):
            with self.subTest(key=key), self.assertRaises(SystemExit):
                mm.check_ledger_provenance({key: "0" * 40}, BASE, TIP, TREE, self.GATE)
        for bad in (None, {}, []):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                mm.check_ledger_provenance(bad, BASE, TIP, TREE, self.GATE)

    def test_shipped_ledger_template_is_final_and_the_nonfinal_copy_is_refused_at_the_seal(self):
        led = lt.template()
        got = mm.lp.check_ledger(led)
        self.assertEqual(got["owned"], {"product": 24054058348, "primitive_selftest": 23988959724})
        self.assertEqual(got["peak_owned"], 24054058348)
        self.assertEqual(led["device_owned_bytes"], 24054058348)
        self.assertEqual(got["host_peak"], 55077640192)
        self.assertEqual(led["host_reservation_owned_bytes"], 55740268544)
        self.assertEqual(led["host_total_with_allowance_bytes"], 57887752192)
        self.assertEqual(led["fabric_hostmem_bytes"] - led["host_total_with_allowance_bytes"], 94306304)
        self.assertEqual(mm.fabric_reservation_gib(led), 22.4)
        self.assertEqual(led["card_bytes"], 24 * 2**30)
        self.assertTrue(led["census_final"])
        # the legacy N10 unit fixture explicitly binds the no-loss row: the arithmetic passes, the PRODUCTION seal refuses it
        self.assertEqual(mm.check_ledger_arithmetic(led, canonical=None)["peak_owned"], 24054058348)
        with self.assertRaises(SystemExit) as cm:
            mm.check_ledger_arithmetic(led)
        self.assertIn("canonical_observation", str(cm.exception))
        nonfinal = lt.nonfinal_template()
        mm.lp.check_ledger(nonfinal, final=False)
        with self.assertRaises(SystemExit):  # the seal refuses a non-final ledger
            mm.check_ledger_arithmetic(nonfinal)
        final = lt.production_ledger()  # canonical post_initialization_actual: actual N9 reference
        got9 = mm.check_ledger_arithmetic(final)
        self.assertEqual(got9["peak_owned"], 24054058348 - 1337982976)
        self.assertEqual(got9["canonical_observation"], "post_initialization_actual")

    def test_arithmetic_rejects_a_term_that_does_not_sum(self):
        led = lt.final_ledger()
        led["host_census"]["product"]["terms"][-1]["bytes"] += 1
        with self.assertRaises(SystemExit):
            mm.check_ledger_arithmetic(led)


class HoldLib(Tmp):
    GOOD = {"schema": "s1.hold_target.v1", "card": "xtx", "uuid": "GPU-43390a851e296ee5", "bdf": "0000:67:00.0",
            "model": {"path": "/models/some model.mq4", "size_bytes": 125288544792}}
    LEDGER = None  # per-purpose ledgers come from ledger_testlib.final_ledger()

    def run_lib(self, snippet, env=None):
        e = {k: v for k, v in os.environ.items() if k not in ("MODEL", "S1_STAGE")}
        e.update({"S1_STAGE": self.tmp, **(env or {})})
        return subprocess.run(["bash", "-c", f'source "{LIB}"; {snippet}'], capture_output=True, text=True, env=e, timeout=60)

    def target(self, **changes):
        t = json.loads(json.dumps(self.GOOD))
        for k, v in changes.items():
            if v is KeyError:
                t.pop(k, None)
            else:
                t[k] = v
        wjson(self.path("target.json"), t)

    def test_target_is_loaded_from_the_declaration(self):
        self.target(card="other", uuid="GPU-0123abcd", bdf="0000:03:00.1", model={"path": "/m/x.gguf", "size_bytes": 42})
        r = self.run_lib('load_target; printf "%s|%s|%s|%s|%s" "$CARD" "$UUID" "$BDF" "$MODEL" "$MODEL_SIZE"')
        self.assertEqual((r.returncode, r.stdout), (0, "other|GPU-0123abcd|0000:03:00.1|/m/x.gguf|42"), r.stderr)

    def test_declared_xtx_target_and_env_model_does_not_override(self):
        self.target()
        r = self.run_lib('load_target; printf "%s|%s|%s|%s|%s" "$CARD" "$UUID" "$BDF" "$MODEL" "$MODEL_SIZE"', env={"MODEL": "/evil"})
        self.assertEqual((r.returncode, r.stdout), (0, "xtx|GPU-43390a851e296ee5|0000:67:00.0|/models/some model.mq4|125288544792"), r.stderr)

    def test_no_target_strings_compiled_into_the_library(self):
        with open(LIB) as f:
            text = f.read()
        for literal in ("GPU-43390a851e296ee5", "0000:67:00.0", "125288544792", "qwen3.8-flash"):
            self.assertNotIn(literal, text)

    def test_missing_target_file_fails_closed(self):
        r = self.run_lib("load_target; echo reached")
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("reached", r.stdout)
        self.assertIn("target.json", r.stderr)

    def test_malformed_targets_fail_closed(self):
        bad = {
            "no schema": {"schema": KeyError}, "wrong schema": {"schema": "v2"}, "no card": {"card": KeyError}, "card comma": {"card": "a,b"},
            "empty card": {"card": ""}, "uuid prefix": {"uuid": "43390a851e296ee5"}, "uuid junk": {"uuid": "GPU-xyz"},
            "bdf short": {"bdf": "67:00.0"}, "bdf traversal": {"bdf": "../../x"}, "no model": {"model": KeyError},
            "model not object": {"model": "/m"}, "relative path": {"model": {"path": "m.mq4", "size_bytes": 1}},
            "path newline": {"model": {"path": "/m\nx", "size_bytes": 1}}, "size zero": {"model": {"path": "/m", "size_bytes": 0}},
            "size bool": {"model": {"path": "/m", "size_bytes": True}}, "size string": {"model": {"path": "/m", "size_bytes": "9"}},
            "size missing": {"model": {"path": "/m"}},
        }
        for label, changes in bad.items():
            with self.subTest(label):
                self.target(**changes)
                r = self.run_lib("load_target; echo reached")
                self.assertEqual(r.returncode, 2, r.stdout)
                self.assertNotIn("reached", r.stdout)

    def test_non_json_and_non_object_targets_fail_closed(self):
        for text in ("", "not json", "[]", "null"):
            with self.subTest(text=text):
                with open(self.path("target.json"), "w") as f:
                    f.write(text)
                r = self.run_lib("load_target; echo reached")
                self.assertEqual(r.returncode, 2)
                self.assertNotIn("reached", r.stdout)

    def memory(self, **changes):
        led = lt.final_ledger()
        for k, v in changes.items():
            if v is KeyError:
                led.pop(k, None)
            else:
                led[k] = v
        wjson(self.path("ledger-hold1.json"), led)
        return self.run_lib("hold_memory 1")

    def test_hold_memory_prints_the_ledger_reservation(self):
        r = self.memory()
        self.assertEqual((r.returncode, r.stdout.split()), (0, ["22.4", "54"]), r.stderr)
        r = self.memory(fabric_reservation_device_gib=18)
        self.assertEqual((r.returncode, r.stdout.split()), (0, ["18", "54"]), r.stderr)

    def test_hold_memory_needs_an_explicit_valid_reservation(self):
        for v in (KeyError, None, 0, -3, True, "22.4"):
            with self.subTest(v=v):
                r = self.memory(fabric_reservation_device_gib=v)
                self.assertEqual(r.returncode, 2)
                self.assertEqual(r.stdout, "")

    def test_final_per_purpose_ledger_prints_22_4_and_54_with_the_physical_card_separate(self):
        r = self.memory()
        self.assertEqual((r.returncode, r.stdout.split()), (0, ["22.4", "54"]), r.stderr)
        self.assertIn("SEPARATE 24 GiB", r.stderr)
        self.assertIn("host 54 GiB", r.stderr)
        self.assertIn("product 24054058348", r.stderr)
        self.assertIn("primitive_selftest 23988959724", r.stderr)

    def test_the_real_n9_template_prints_22_4_and_55_gib(self):
        # the shipped final template: canonical N9 (40 host layers): S1 host census 56,416,671,744 + 2 GiB allowance -> 55 GiB; owned N9 figures
        wjson(self.path("ledger-hold1.json"), lt.shipped_template())
        r = self.run_lib("hold_memory 1")
        self.assertEqual((r.returncode, r.stdout.split()), (0, ["22.4", "55"]), r.stderr)
        self.assertIn("host 55 GiB", r.stderr)
        self.assertIn("product 22716075372", r.stderr)
        self.assertIn("primitive_selftest 22650976748", r.stderr)

    def test_hold_memory_refuses_the_nonfinal_template_and_flat_or_wrong_profile_ledgers(self):
        wjson(self.path("ledger-hold1.json"), lt.nonfinal_template())
        r = self.run_lib("hold_memory 1")
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertIn("not final", r.stderr)
        flat = lt.final_ledger()
        flat.update({"tile_rows": 768, "device_weights_bytes": 1})
        flat_r = self.memory(**{k: flat[k] for k in ("tile_rows", "device_weights_bytes")})
        self.assertEqual((flat_r.returncode, flat_r.stdout), (2, ""))
        bad = lt.final_ledger()
        bad["profiles"]["product"]["tile_rows"] = 512
        wjson(self.path("ledger-hold1.json"), bad)
        r = self.run_lib("hold_memory 1")
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertIn("768", r.stderr)


if __name__ == "__main__":
    unittest.main()
