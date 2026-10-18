#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the split Hold 1: hold1a-correct.sh / hold1b-correct.sh structure (required lists, status scope, budgets, the H1-only
explicit HIPFIRE_OFFLOAD_EXEC=pcie pin), the sealed immutable 1a handoff (h1_handoff.py seal/verify/receipt1b: valid, tamper, partial,
runtime/fixture/profile mismatch) and the 1b block-before-any-GPU rule. Synthetic temp files only: no GPU, no model, no staged binary is
ever executed.

    python3 scripts/slowlink-holds/test_hold1_split.py -v
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_evidence as ce  # noqa: E402
import h1_handoff as h1  # noqa: E402
import ledger_testlib as lt  # noqa: E402
import make_fixtures  # noqa: E402
import make_manifest as mm  # noqa: E402
import check_headroom as ch  # noqa: E402
import test_check_headroom as tch  # noqa: E402

LIB = os.path.join(HERE, "hold-lib.sh")


def text(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


def code_lines(name):
    """Non-comment lines of a shell script."""
    return [l for l in text(name).splitlines() if l.strip() and not l.lstrip().startswith("#")]


def required(name):
    m = re.search(r'^REQUIRED="([^"]*)"', text(name), re.M)
    return m.group(1).split()


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Structure(unittest.TestCase):
    def test_obsolete_single_entry_is_gone_and_both_halves_exist(self):
        self.assertFalse(os.path.exists(os.path.join(HERE, "hold1-correct.sh")))
        for name in ("hold1a-correct.sh", "hold1b-correct.sh"):
            self.assertTrue(os.access(os.path.join(HERE, name), os.X_OK), name)
        for name in os.listdir(HERE):
            if name.endswith((".sh", ".py", ".json")) and name != "test_hold1_split.py":
                self.assertNotIn("hold1-correct", text(name), name)

    def test_required_lists_and_status_scope_are_split(self):
        self.assertEqual(required("hold1a-correct.sh"), ["prim", "state", "state-headroom"])
        b = required("hold1b-correct.sh")
        self.assertEqual(b, ["a-handoff", "headroom-state", "beta-input", "greedy-beta", "greedy-s1", "greedy-cmp", "receipt-1b"])
        self.assertFalse(set(required("hold1a-correct.sh")) & set(b))
        a_code, b_code = "\n".join(code_lines("hold1a-correct.sh")), "\n".join(code_lines("hold1b-correct.sh"))
        self.assertIn("HOLD_SCOPE=partial_h1_prim_state", a_code)
        self.assertIn("HOLD_ADVANCE=false", a_code)
        self.assertIn("HOLD_SCOPE=h1_complete_on_verified_1a", b_code)
        self.assertIn("HOLD_ADVANCE=true", b_code)
        # 1a runs the state-headroom judgment (and no 1b headroom row name) but no greedy cell; 1b runs no prim or state cell
        for forbidden in ("headroom-state", "greedy-", "beta"):
            self.assertNotIn(forbidden, a_code)
        for forbidden in ("cell prim", "cell state", "--mode state", "--mode primitives"):
            self.assertNotIn(forbidden, b_code)

    def test_1a_runs_the_required_state_headroom_row_after_state_and_before_the_seal(self):
        lines = code_lines("hold1a-correct.sh")
        state = next(i for i, l in enumerate(lines) if l.startswith("cell state "))
        rows = [i for i, l in enumerate(lines) if l.startswith("headroom ")]
        self.assertEqual(len(rows), 1, "exactly one headroom invocation in 1a")
        seal = next(i for i, l in enumerate(lines) if l.startswith("check handoff-seal"))
        self.assertLess(state, rows[0])
        self.assertLess(rows[0], seal)
        # immediately after the state cell and immediately before the seal: no other row or cell in between
        self.assertEqual(rows[0], state + 2, "the headroom row follows the state cell (continuation line) with nothing between")
        self.assertEqual(seal, rows[0] + 1)
        # the exact invocation: state cell raw data judged, the primitive cell recorded as INFO only
        self.assertEqual(lines[rows[0]], 'headroom state-headroom "$OUT/state" "$OUT/prim"')
        self.assertIn("state-headroom", required("hold1a-correct.sh"))
        self.assertEqual(len([l for l in lines if l.startswith("cell ")]), 2, "no GPU cell was added")
        # the primitive cell is never the judged cell
        self.assertNotIn('headroom state-headroom "$OUT/prim"', "\n".join(lines))
        # the judge, its threshold and its info-only handling are the existing ones
        self.assertEqual(ch.MIN_FREE_BYTES, 512 * 2**20)
        self.assertEqual(ch.MIN_FREE_BYTES, 536_870_912)
        self.assertIn("536,870,912", text("hold-lib.sh"))
        self.assertIn('--info-cell "$c"', text("hold-lib.sh"))
        self.assertEqual(text("hold-lib.sh").count("python3 \"$HOLDS/check_headroom.py\""), 1)

    def test_1b_keeps_the_headroom_recheck_on_the_sealed_1a_raw_data(self):
        b = "\n".join(code_lines("hold1b-correct.sh"))
        self.assertIn('headroom headroom-state "$A_DIR/state" "$A_DIR/prim"', b)
        self.assertIn("headroom-state", required("hold1b-correct.sh"))
        self.assertEqual(len([l for l in code_lines("hold1b-correct.sh") if l.startswith("headroom ")]), 1)
        self.assertNotIn("state-headroom", b)

    def test_budgets_follow_the_main_ruling(self):
        a, b = "\n".join(code_lines("hold1a-correct.sh")), "\n".join(code_lines("hold1b-correct.sh"))
        self.assertIn("hold_init hold1a 2880", a)
        self.assertIn("cell prim 660 660 ", a)
        self.assertIn("cell state 1860 1860 ", a)
        self.assertEqual(len([l for l in a.splitlines() if l.startswith("cell ")]), 2)
        lib = text("hold-lib.sh")
        self.assertIn("KILL_GRACE=15", lib)
        self.assertEqual(lib.count("timeout -k $KILL_GRACE"), 2)  # command.txt record + the real spawn: one kill grace per cell
        self.assertIn("await_idle 90", lib)
        grace, drain, finalize, cpu, contingency = 15, 90, 45, 30, 75
        worst_a = 660 + 1860 + 2 * grace + 2 * drain + finalize + cpu
        self.assertEqual(worst_a, 2805)
        self.assertEqual(worst_a + contingency, 2880)
        self.assertLessEqual(worst_a, 2880)
        self.assertEqual(660, 600 + 60)  # at least 600 s complete LOAD + 60 s case/control allowance, whole load counted inside the cap
        self.assertEqual(round(123 - 104.5, 1), 18.5)  # old 123 s primitive cell minus its 104.5 s load: an old-build proxy, not a measurement
        self.assertLess(18.5, 60)
        self.assertIn("hold_init hold1b 1320", b)
        self.assertIn("cell greedy-beta 600 600 ", b)
        self.assertIn("cell greedy-s1 390 390 ", b)
        worst_b = 600 + 390 + 2 * grace + 2 * drain + finalize + cpu  # both kill graces counted, as for 1a
        self.assertEqual(worst_b, 1275)
        self.assertLessEqual(worst_b, 1320)
        self.assertIn("FINALIZE_S=45", text("hold-lib.sh"))
        for name in ("hold1a-correct.sh", "hold1b-correct.sh"):
            self.assertIn('[ "${1:-}" = "--print-memory" ] && hold_memory 1', text(name))

    def test_1a_header_states_the_budget_basis_and_makes_no_measured_load_claim(self):
        head = "\n".join(l for l in text("hold1a-correct.sh").splitlines() if l.startswith("#"))
        flat = re.sub(r"\s*\n#\s*", " ", "\n" + head).replace("\n", " ")
        for needle in ("660", "2805", "2880", "2 x 15 s", "75 s contingency", "at least 600 s for the LOAD", "60 s case/control allowance",
                       "123 s primitive cell", "104.5 s load", "~18.5 s", "NOT a current measurement", "NOT a measurement of this build",
                       "194.3 s", "MODEL ALONE", "not the whole model + PLE", "Q4L is not in this build", "no completed load of this build has been measured",
                       "122.2 GiB", "page caching cannot keep all of it resident", "no separate prewarm", "no hook and no IO reader",
                       "same-run monotonic load_start -> load_end", "NON-GATING", "load_phase.json", "null when censored"):
            self.assertIn(needle, flat, needle)
        for stale in ("hard 2400", "hard 2640", "prim 180", "prim 420", "2565", "276.64", "cold proxy", "ionice", "nice 19", "prewarmed OUTSIDE"):
            self.assertNotIn(stale, flat, stale)
        # the old 80-110 s range may only appear as a retracted warm figure, never as a budget input or something to rely on
        self.assertIn("80-110 s range is a retracted warm/part-warm planning figure and is not a budget input", flat)
        code = "\n".join(code_lines("hold1a-correct.sh"))
        self.assertNotRegex(code, r"(?i)prewarm|ionice|readahead|vmtouch|\bcat\b.*\bmodel|dd if=")

    def test_the_h1_pin_is_exactly_pcie_on_every_h1_gpu_cell_and_nowhere_global(self):
        common = "\n".join(code_lines("hold1-common.sh"))
        self.assertEqual(common.strip(), 'H1_OFFLOAD_PIN="HIPFIRE_OFFLOAD_EXEC=pcie"')
        for name, cells in (("hold1a-correct.sh", ("prim", "state")), ("hold1b-correct.sh", ("greedy-beta", "greedy-s1"))):
            body = text(name)
            for cell in cells:
                line = [l for l in body.splitlines() if l.startswith(f"cell {cell} ")]
                self.assertEqual(len(line), 1, (name, cell))
                self.assertIn('"$H1_OFFLOAD_PIN"', line[0], (name, cell))
            self.assertEqual(len([l for l in code_lines(name) if l.startswith("cell ")]), len(cells))
            self.assertIn('. "$HOLDS/hold1-common.sh"', body)
        for name in ("hold-lib.sh", "hold2-timing.sh", "hold3-candidate.sh"):
            self.assertNotIn("HIPFIRE_OFFLOAD_EXEC", "\n".join(code_lines(name)), name)
            self.assertNotIn("hold1-common", "\n".join(code_lines(name)), name)
        self.assertNotIn("=off", text("hold1-common.sh"))

    def run_bash(self, script, env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("HIPFIRE_", "HIP_", "ROCR_", "HSA_"))}
        e.update(env or {})
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e, timeout=60)

    def test_cell_env_strips_a_foreign_pin_and_applies_the_h1_pin_only_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = f'source "{LIB}"; S1_STAGE="{tmp}"; UUID=GPU-1; '
            plain = self.run_bash(base + f'cell_env "{tmp}/c1"; printf "%s\\n" "${{E[@]}}"', env={"HIPFIRE_OFFLOAD_EXEC": "cpu"})
            self.assertEqual(plain.returncode, 0, plain.stderr)
            self.assertNotIn("HIPFIRE_OFFLOAD_EXEC=cpu", plain.stdout.split("\n"))
            self.assertIn("HIPFIRE_OFFLOAD_EXEC", plain.stdout.split("\n"))  # the `-u` of the foreign value
            self.assertFalse(any(l.startswith("HIPFIRE_OFFLOAD_EXEC=") for l in plain.stdout.splitlines()))
            pinned = self.run_bash(base + f'. "{HERE}/hold1-common.sh"; cell_env "{tmp}/c2" "$H1_OFFLOAD_PIN"; printf "%s\\n" "${{E[@]}}"',
                                   env={"HIPFIRE_OFFLOAD_EXEC": "cpu"})
            self.assertIn("HIPFIRE_OFFLOAD_EXEC=pcie", pinned.stdout.splitlines())
            # R5 D2 rule: EVERY cell (plain or H1-pinned) carries the non-default D2 pin, whatever the ambient environment says
            for out in (plain, pinned):
                self.assertIn("HIPFIRE_QWEN4_D2_DEFAULT=0", out.stdout.splitlines())
            ambient = self.run_bash(base + f'cell_env "{tmp}/c3"; printf "%s\\n" "${{E[@]}}"', env={"HIPFIRE_QWEN4_D2_DEFAULT": "1"})
            self.assertIn("HIPFIRE_QWEN4_D2_DEFAULT=0", ambient.stdout.splitlines())
            self.assertNotIn("HIPFIRE_QWEN4_D2_DEFAULT=1", ambient.stdout.splitlines())

    def test_block_before_any_gpu_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (f'source "{LIB}"; OUT="{tmp}"; CELLS_JSON="{tmp}/cells.jsonl"; : > "$CELLS_JSON"; T0=$(date +%s); BUDGET=1000; '
                      f'BLOCKED=""; FAILED=(); check a-handoff false || {{ [ -n "$BLOCKED" ] || BLOCKED="a-handoff failed"; }}; '
                      f'cell greedy-beta 600 600 -- touch "{tmp}/ran"; echo "rc=$?"')
            r = self.run_bash(script, env={"S1_STAGE": tmp})
            self.assertIn("rc=1", r.stdout, r.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "ran")))
            self.assertTrue(open(os.path.join(tmp, "greedy-beta", "status.txt")).read().startswith("skipped"))

    def test_1b_script_verifies_the_handoff_before_its_first_cell(self):
        lines = code_lines("hold1b-correct.sh")
        verify = next(i for i, l in enumerate(lines) if l.startswith("check a-handoff"))
        first_cell = next(i for i, l in enumerate(lines) if l.startswith("cell "))
        headroom = next(i for i, l in enumerate(lines) if l.startswith("headroom headroom-state"))
        self.assertLess(verify, headroom)
        self.assertLess(headroom, first_cell)
        body = "\n".join(lines)
        self.assertIn('A_OUT=${1:-${S1_HOLD1A_OUT:-}}', body)
        self.assertIn("BLOCKED=", body.split("check a-handoff")[1].split("headroom headroom-state")[0])
        self.assertIn('headroom headroom-state "$A_DIR/state" "$A_DIR/prim"', body)  # the 1a RAW data, no second state cell
        self.assertIn("check beta-input", body)
        self.assertLess(body.index("check beta-input"), body.index("cell greedy-beta"))
        self.assertIn('--input "$S/fixtures/prompts.short.beta.jsonl"', body)

    def summary(self, scope, advance, rows, required_names):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "cells.jsonl"), "w") as f:
                for name, status in rows:
                    f.write(json.dumps({"name": name, "status": status, "why": "", "rc": 0, "seconds": 1, "leftover": ""}) + "\n")
            script = (f'source "{LIB}"; CELLS_JSON="{tmp}/cells.jsonl"; OUT="{tmp}"; HOLD=h; REQUIRED="{required_names}"; '
                      f'HOLD_SCOPE={scope}; HOLD_ADVANCE={advance}; finish_summary')
            subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**os.environ, "S1_STAGE": tmp}, timeout=60)
            with open(os.path.join(tmp, "hold-summary.json")) as f:
                return json.load(f)

    def test_1a_pass_is_partial_and_never_an_advance(self):
        s = self.summary("partial_h1_prim_state", "false", [("prim", "pass"), ("state", "pass"), ("handoff-seal", "pass")], "prim state")
        self.assertEqual((s["status"], s["scope"], s["advance"]), ("pass", "partial_h1_prim_state", False))
        s = self.summary("h1_complete_on_verified_1a", "true", [("a-handoff", "pass"), ("greedy-cmp", "pass")], "a-handoff greedy-cmp")
        self.assertEqual((s["status"], s["advance"]), ("pass", True))
        s = self.summary("h1_complete_on_verified_1a", "true", [("a-handoff", "pass"), ("greedy-cmp", "skipped")], "a-handoff greedy-cmp")
        self.assertEqual((s["status"], s["advance"]), ("incomplete", False))
        s = self.summary("h1_complete_on_verified_1a", "true", [("a-handoff", "fail")], "a-handoff greedy-cmp")
        self.assertEqual((s["status"], s["advance"]), ("fail", False))

    def run_1a_tail(self, state_free=None, state_gate=True, prim_free=None, state_cell_status="pass"):
        """Run the 1a tail rows for real (the existing headroom() judge, the real finish_summary) on synthetic raw cells; the seal is a marker."""
        with tempfile.TemporaryDirectory() as tmp:
            state, prim = os.path.join(tmp, "state"), os.path.join(tmp, "prim")
            tch.write_external(state, free_at=(100, state_free) if state_free is not None else None)
            if state_gate:
                tch.write_gate(state)
            tch.write_external(prim, free_at=(100, prim_free) if prim_free is not None else None)
            tch.write_gate(prim)
            script = (f'source "{LIB}"; OUT="{tmp}"; CELLS_JSON="{tmp}/cells.jsonl"; : > "$CELLS_JSON"; T0=$(date +%s); BUDGET=2355; '
                      f'BLOCKED=""; FAILED=(); HOLD=hold1a; REQUIRED="prim state state-headroom"; HOLD_SCOPE=partial_h1_prim_state; HOLD_ADVANCE=false; '
                      f'cell_row prim pass "" 0 1 ""; cell_row state {state_cell_status} "" 0 1 ""; '
                      f'[ "{state_cell_status}" = pass ] || BLOCKED="state failed"; '
                      f'headroom state-headroom "$OUT/state" "$OUT/prim"; check handoff-seal touch "{tmp}/sealed"; finish_summary')
            r = self.run_bash(script, env={"S1_STAGE": tmp})
            with open(os.path.join(tmp, "hold-summary.json")) as f:
                summary = json.load(f)
            with open(os.path.join(tmp, "cells.jsonl")) as f:
                rows = [json.loads(l) for l in f if l.strip()]
            hr = os.path.join(tmp, "state-headroom", "stdout.txt")
            verdict = json.load(open(hr)) if os.path.exists(hr) and os.path.getsize(hr) else None  # the judge's JSON, read before the temp dir goes away
            return r, summary, [(x["name"], x["status"]) for x in rows], os.path.exists(os.path.join(tmp, "sealed")), verdict

    def test_state_headroom_pass_is_required_and_precedes_the_seal(self):
        r, summary, rows, sealed, _ = self.run_1a_tail(state_free=ch.MIN_FREE_BYTES)  # exactly 512 MiB passes (threshold unchanged)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(rows, [("prim", "pass"), ("state", "pass"), ("state-headroom", "pass"), ("handoff-seal", "pass")])
        self.assertTrue(sealed)
        self.assertEqual((summary["status"], summary["scope"], summary["advance"]), ("pass", "partial_h1_prim_state", False))
        self.assertEqual(summary["required"], ["prim", "state", "state-headroom"])

    def test_state_headroom_fail_fails_1a_and_blocks_the_seal(self):
        r, summary, rows, sealed, _ = self.run_1a_tail(state_free=ch.MIN_FREE_BYTES - 1)  # one byte below 512 MiB
        self.assertEqual(r.returncode, 1)
        self.assertEqual(rows, [("prim", "pass"), ("state", "pass"), ("state-headroom", "fail"), ("handoff-seal", "skipped")])
        self.assertFalse(sealed)
        self.assertEqual((summary["status"], summary["advance"]), ("fail", False))

    def test_state_headroom_incomplete_telemetry_is_not_a_pass_and_blocks_the_seal(self):
        r, summary, rows, sealed, _ = self.run_1a_tail(state_gate=False)  # no completed state gate.json: INCOMPLETE (rc 2), never zero-filled
        self.assertEqual(r.returncode, 1)
        self.assertEqual(rows, [("prim", "pass"), ("state", "pass"), ("state-headroom", "incomplete"), ("handoff-seal", "skipped")])
        self.assertFalse(sealed)
        self.assertEqual((summary["status"], summary["advance"]), ("incomplete", False))

    def test_a_failed_state_cell_skips_headroom_and_seal_and_the_hold_is_not_a_pass(self):
        r, summary, rows, sealed, _ = self.run_1a_tail(state_cell_status="fail")
        self.assertEqual(rows, [("prim", "pass"), ("state", "fail"), ("state-headroom", "skipped"), ("handoff-seal", "skipped")])
        self.assertFalse(sealed)
        self.assertEqual(summary["status"], "fail")

    def test_primitive_headroom_never_gates_1a(self):
        r, summary, rows, sealed, verdict = self.run_1a_tail(state_free=ch.MIN_FREE_BYTES, prim_free=1)  # prim free is 1 B: information only
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(rows[2], ("state-headroom", "pass"))
        self.assertTrue(sealed)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(verdict["info_only"]["diagnostic_cells"]["prim"]["external"]["min_free_bytes"], 1)

    def test_a_missing_state_headroom_row_is_incomplete_never_a_pass(self):
        s = self.summary("partial_h1_prim_state", "false", [("prim", "pass"), ("state", "pass"), ("handoff-seal", "pass")], "prim state state-headroom")
        self.assertEqual((s["status"], s["missing"], s["advance"]), ("incomplete", ["state-headroom"], False))
        s = self.summary("partial_h1_prim_state", "false", [("prim", "pass"), ("state", "pass"), ("state-headroom", "skipped")],
                         "prim state state-headroom")
        self.assertEqual(s["status"], "incomplete")

    def test_h2_needs_the_advance_flag_of_the_1b_summary(self):
        body = "\n".join(code_lines("hold2-timing.sh"))
        self.assertIn("d.get('advance') is True", body)


class World(unittest.TestCase):
    """A synthetic stage and a synthetic 1a run directory whose closed cell files are real (collect_evidence.py writes evidence.json)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.stage = os.path.join(self.tmp, "stage")
        self.a = os.path.join(self.tmp, "a-out")
        for d in ("manifests", "bin-s1", "bin-beta", "fixtures"):
            os.makedirs(os.path.join(self.stage, d))
        os.makedirs(self.a)
        for rel, data in (("bin-s1/qwen4_slowlink_gate", b"gate"), ("bin-s1/qwen4_greedy_ids", b"greedy-s1"), ("bin-s1/BUILD-IDENTITY.json", b"{}"),
                          ("bin-beta/qwen4_greedy_ids", b"greedy-beta")):
            with open(os.path.join(self.stage, rel), "wb") as f:
                f.write(data)
        natural = os.path.join(self.tmp, "natural.txt")
        with open(natural, "w") as f:
            f.write("natural text sentence. " * 200)
        with contextlib.redirect_stdout(io.StringIO()):
            make_fixtures.main(["make_fixtures.py", os.path.join(self.stage, "fixtures"), natural])
        with open(os.path.join(self.stage, "target.json"), "w") as f:
            json.dump({"schema": "s1.hold_target.v1"}, f)
        with open(os.path.join(self.stage, "ledger-hold1.json"), "w") as f:
            json.dump(lt.final_ledger(), f)
        self.write_manifest()
        for name in h1.CELLS:
            self.write_cell(name)

    # ---- builders ----
    def write_manifest(self, tip=lt.TIP, product_tile=768):
        inputs = mm.beta_input_binding(self.stage, "m" * 64, {"model_config_sha256": "c" * 64})
        man = {"source": {"base": lt.BASE, "tip": tip, "tree": lt.TREE}, "binary": {"sha256": sha(os.path.join(self.stage, "bin-s1", "qwen4_slowlink_gate"))},
               "kernel": {"code_object_hashes": {"k": "d" * 64}}, "model": {"path": "/m.mq4", "size_bytes": 1, "sha256": "m" * 64},
               "harness": {"commit": "9" * 40, "tree": "8" * 40},
               "memory_ledger": {"profiles": {"product": {"tile_rows": product_tile}, "primitive_selftest": {"tile_rows": 512}}},
               "inputs": inputs}
        with open(os.path.join(self.stage, "manifests", "manifest-hold1.json"), "w") as f:
            json.dump(man, f)

    @staticmethod
    def checks_block(purpose, tile):
        """The report's `checks` exactly as the gate serializes Checks::json: {status, required, missing, checks: [{name, status, detail}]}."""
        slack = lt.POST_INIT - lt.needed(9)
        detail = {"placement": "slowlink-profile", "resident_layers": 9, "host_mapped_tensors": 39, "slowlink_profile": True,
                  "expected_purpose": purpose, "expected_tile_rows": tile, "admitted_purpose": purpose, "admitted_tile_rows": tile,
                  "admitted_device_snapshot_budget_bytes": slack, "admission": lt.witness(lt.POST_INIT, lt.PRE_INIT, 9)}
        return {"status": "pass", "required": ["identity.manifest", "load.placement", "progress.durability"], "missing": [],
                "checks": [{"name": "identity.manifest", "status": "pass", "detail": {"manifest": "ok"}},
                           {"name": "load.placement", "status": "pass", "detail": detail},
                           {"name": "progress.durability", "status": "pass", "detail": None}]}

    def gate(self, name, **over):
        spec = h1.CELLS[name]
        tile = 768 if spec["purpose"] == "product" else 512
        rep = {"schema": h1.REPORT_SCHEMA, "status": "pass", "mode": spec["mode"], "route": "layer-major",
               "manifest_sha256": sha(os.path.join(self.stage, h1.MANIFEST)),
               "fixture_sha256": sha(os.path.join(self.stage, "fixtures", "gate-fixture.json")),
               "checks": self.checks_block(spec["purpose"], tile),
               "evidence": self.profile(spec["purpose"], tile, spec["purpose"], tile)}
        rep.update(over)
        return rep

    @staticmethod
    def profile(expected_purpose, expected_tile, admitted_purpose, admitted_tile):
        """report.evidence with the gate's s1_profile: expected_* is built-in intent, admitted_* is the read-back of the loaded bundle (null before)."""
        return {"s1_profile": {"expected_purpose": expected_purpose, "expected_tile_rows": expected_tile,
                               "admitted_purpose": admitted_purpose, "admitted_tile_rows": admitted_tile}}

    def write_cell(self, name, rc=0, pin="pcie", gate=None, status_txt=None, sidecar_only=False, d2="0", **gate_over):
        d = os.path.join(self.a, name)
        shutil.rmtree(d, ignore_errors=True)  # a rebuilt cell never inherits files of its earlier incarnation
        os.makedirs(d)

        def put(fn, data):
            with open(os.path.join(d, fn), "w") as f:
                f.write(data)
        for fn in ("command.txt", "stdout.txt", "stderr.txt", "mem.csv", "meminfo-before.txt", "meminfo-after.txt"):
            put(fn, f"{name} {fn}\n")
        put("rc.txt", f"{rc}\n")
        put("baseline.json", json.dumps({"vram_used": 1, "gtt_used": 2}))
        put("vram.csv", "t_mono_ns,used,total\n1,2,3\n")
        put("vram-events.jsonl", json.dumps({"event": "command_start", "t_mono_ns": 1}) + "\n")
        put("vram-summary.json", json.dumps({"finished": True}))
        put("spawn-env.json", json.dumps({"schema": "s1.cell_spawn_env.v1", "unset": [],
                                          "set": {**({"HIPFIRE_OFFLOAD_EXEC": pin} if pin else {}), **({"HIPFIRE_QWEN4_D2_DEFAULT": d2} if d2 else {})}}))
        record = {"schema": ce.PROGRESS_SCHEMA, "run_id": "r", "seq": 0, "event": "run_start", "utc": "u", "unix_ms": 1, "mono_ns": 1}
        put("gate.json.progress.jsonl", json.dumps(record) + "\n" + json.dumps({**record, "seq": 1, "event": "final", "status": "pass"}) + "\n")
        if not sidecar_only:
            put("gate.json", json.dumps(gate if gate is not None else self.gate(name, **gate_over)))
        if status_txt:
            put("status.txt", status_txt)
        ce.collect(d, name, str(rc), "true")

    def seal(self):
        with contextlib.redirect_stdout(io.StringIO()):
            h1.seal(self.stage, self.a)

    def sealed(self):
        self.seal()
        return h1.verify(self.stage, self.a)

    def assertRefused(self, fn, fragment):
        with self.assertRaises(SystemExit) as cm:
            fn()
        self.assertIn(fragment, str(cm.exception))


class Handoff(World):
    def test_valid_handoff_seals_and_verifies_from_the_directory_or_the_file(self):
        v = self.sealed()
        self.assertEqual(v["schema"], h1.VERIFY_SCHEMA)
        path = os.path.join(self.a, h1.HANDOFF_NAME)
        self.assertEqual(v["handoff"], {"path": path, "sha256": sha(path)})
        self.assertEqual(h1.verify(self.stage, path)["identity_sha256"], v["identity_sha256"])
        handoff = json.load(open(path))
        self.assertEqual((handoff["hold"], handoff["scope"], handoff["advance"], handoff["qualifies"]), ("hold1a", h1.SCOPE, False, True))
        state = v["cells"]["state"]["files"]
        for fn in ("gate.json", "vram.csv", "vram-events.jsonl", "vram-summary.json", "baseline.json"):
            self.assertEqual(state[fn]["path"], os.path.join(self.a, "state", fn))
            self.assertEqual(state[fn]["sha256"], sha(os.path.join(self.a, "state", fn)))
        self.assertIn("gate.json.progress.jsonl", state)  # bound, but never used for a verdict
        placement = v["cells"]["state"]["load_placement"]
        self.assertEqual((placement["placement"], placement["slowlink_profile"], placement["admitted_tile_rows"]), ("slowlink-profile", True, 768))
        self.assertEqual(v["cells"]["state"]["spawn_env_pin"], {"HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_QWEN4_D2_DEFAULT": "0"})
        self.assertEqual(v["cells"]["state"]["arm"], "non_default_d2_off")
        # the adaptive placement (nine layers, ten preferred) is carried verbatim in the verified read-back
        self.assertEqual((placement["resident_layers"], placement["admission"]["resident_layers"]), (9, 9))
        self.assertEqual(placement["admission"]["preferred_deficit_bytes"], 50_000_000)
        self.assertEqual((v["cells"]["state"]["admitted_purpose"], v["cells"]["state"]["admitted_tile_rows"]), ("product", 768))
        self.assertEqual((v["cells"]["prim"]["admitted_purpose"], v["cells"]["prim"]["admitted_tile_rows"]), ("primitive_selftest", 512))
        # the A input paths/hashes/proof are carried for the 1b receipt
        inputs = v["a_inputs"]
        self.assertEqual(inputs["greedy_canonical"]["sha256"], sha(os.path.join(self.stage, "fixtures", "prompts.short.jsonl")))
        self.assertEqual(inputs["greedy_beta"]["sha256"], sha(os.path.join(self.stage, "fixtures", "prompts.short.beta.jsonl")))
        self.assertEqual(inputs["beta_adapter"]["stripped_keys"], ["resume_at"])
        # the run root (logs/summary the EXIT trap may append) is neither bound nor needed
        with open(os.path.join(self.a, "hold.jsonl"), "w") as f:
            f.write("late root log\n")
        with open(os.path.join(self.a, "hold-summary.json"), "w") as f:
            f.write("{}")
        h1.verify(self.stage, self.a)

    def test_a_cell_that_did_not_complete_pass_is_never_sealed(self):
        cases = {
            "rc 1": lambda: self.write_cell("state", rc=1),
            "timeout rc": lambda: self.write_cell("state", rc=124),
            "status.txt recorded": lambda: self.write_cell("state", status_txt="lifecycle: leftover='x' drained=false\n"),
            "incomplete report": lambda: self.write_cell("state", status="incomplete"),
            "fail report": lambda: self.write_cell("state", status="fail"),
            "wrong schema": lambda: self.write_cell("state", schema="x"),
            "wrong mode": lambda: self.write_cell("state", mode="product"),
            "wrong route": lambda: self.write_cell("state", route="incumbent"),
            "wrong product tile": lambda: self.write_cell("state", evidence=self.profile("product", 768, "product", 512)),
            "primitive purpose on state": lambda: self.write_cell("state", evidence=self.profile("product", 768, "primitive_selftest", 768)),
            "expected-only (admitted null after a failed load)": lambda: self.write_cell("state", evidence=self.profile("product", 768, None, None)),
            "admitted purpose null": lambda: self.write_cell("prim", evidence=self.profile("primitive_selftest", 512, None, 512)),
            "expected wrong, admitted right": lambda: self.write_cell("state", evidence=self.profile("product", 512, "product", 768)),
            "no s1_profile at all": lambda: self.write_cell("state", evidence={}),
            "aborted/setup-failed report (no checks, evidence or hashes)": lambda: self.write_cell(
                "state", gate={"schema": h1.REPORT_SCHEMA, "status": "incomplete", "mode": "state", "route": "layer-major", "error": "load failed"}),
            "wrong manifest": lambda: self.write_cell("prim", manifest_sha256="0" * 64),
            "wrong fixture": lambda: self.write_cell("prim", fixture_sha256="0" * 64),
            "pin off": lambda: self.write_cell("state", pin="off"),
            "pin missing": lambda: self.write_cell("prim", pin=None),
            "d2 pin missing": lambda: self.write_cell("state", d2=None),
            "d2 default left on": lambda: self.write_cell("prim", d2="1"),
            "d2 pin empty": lambda: self.write_cell("state", d2=""),
        }
        for label, build in cases.items():
            with self.subTest(label):
                build()
                self.assertRefused(self.seal, "refusing to seal 1a")
                self.assertFalse(os.path.exists(os.path.join(self.a, h1.HANDOFF_NAME)))
                for name in h1.CELLS:  # restore
                    self.write_cell(name)

    def test_forged_root_pass_cannot_hide_a_missing_failed_or_inconsistent_checks_block(self):
        def row(b):
            return next(r for r in b["checks"] if r["name"] == "load.placement")

        def detail(b):
            return row(b)["detail"]

        mutators = {
            "placement row absent (still required, root forged pass)": lambda b: b["checks"].remove(row(b)),
            "placement row failed (root and block forged pass)": lambda b: row(b).update({"status": "fail", "reason": "declined"}),
            "placement row skipped": lambda b: row(b).update({"status": "skipped", "reason": "x"}),
            "placement neither required nor present": lambda b: (b["checks"].remove(row(b)), b["required"].remove("load.placement")),
            "placement not required but present": lambda b: b["required"].remove("load.placement"),
            "placement twice": lambda b: b["checks"].append(copy.deepcopy(row(b))),
            "placement is not the S1 profile": lambda b: detail(b).update({"placement": "incumbent"}),
            "slowlink_profile false": lambda b: detail(b).update({"slowlink_profile": False}),
            "resident layers disagree with the witness": lambda b: detail(b).update({"resident_layers": 10}),
            "admission witness missing": lambda b: detail(b).pop("admission"),
            "admission witness null": lambda b: detail(b).update({"admission": None}),
            "admission witness nonmaximal": lambda b: detail(b).update({"admission": lt.witness(lt.PRE_INIT + lt.LAYER, lt.PRE_INIT + lt.LAYER, 9)}),
            "admission witness oversubscribed": lambda b: detail(b).update(
                {"resident_layers": 10, "admission": lt.witness(lt.POST_INIT, lt.PRE_INIT, 10), "admitted_device_snapshot_budget_bytes": 0}),
            "admission witness for all-resident": lambda b: detail(b).update(
                {"resident_layers": 48, "admission": lt.witness(lt.needed(48) + 5, None, 48), "admitted_device_snapshot_budget_bytes": 5}),
            "snapshot allowance differs from the quote slack": lambda b: detail(b).update({"admitted_device_snapshot_budget_bytes": 1}),
            "witness tampered (needed bytes)": lambda b: detail(b)["admission"].update({"needed_bytes": detail(b)["admission"]["needed_bytes"] + 1}),
            "witness forged preferred deficit": lambda b: detail(b)["admission"].update({"preferred_deficit_bytes": 0}),
            "witness basis is the gate's pre_load": lambda b: detail(b)["admission"].update({"memory_basis": "pre_load"}),
            "resident layers bool": lambda b: detail(b).update({"resident_layers": True}),
            "placement detail absent": lambda b: row(b).update({"detail": None}),
            "expected null in the placement row": lambda b: detail(b).update({"expected_purpose": None, "expected_tile_rows": None}),
            "admitted null in the placement row": lambda b: detail(b).update({"admitted_purpose": None, "admitted_tile_rows": None}),
            "admitted tile wrong in the placement row": lambda b: detail(b).update({"admitted_tile_rows": 512}),
            "admitted purpose wrong in the placement row": lambda b: detail(b).update({"admitted_purpose": "primitive_selftest"}),
            "another check failed under a forged pass": lambda b: b["checks"][0].update({"status": "fail", "reason": "x"}),
            "a required check was never executed": lambda b: b["required"].append("state.digests"),
            "a required check only skipped": lambda b: (b["required"].append("state.digests"),
                                                        b["checks"].append({"name": "state.digests", "status": "skipped", "reason": "x"})),
            "block status not pass": lambda b: b.update({"status": "incomplete"}),
            "block status fail": lambda b: b.update({"status": "fail"}),
            "missing not empty": lambda b: b.update({"missing": ["x"]}),
            "required empty": lambda b: b.update({"required": []}),
            "rows malformed": lambda b: b.update({"checks": [{"status": "pass"}]}),
            "rows not a list": lambda b: b.update({"checks": {}}),
        }
        for label, mutate in mutators.items():
            with self.subTest(label):
                block = self.checks_block("product", 768)
                mutate(block)
                self.write_cell("state", checks=block)  # the report's root status stays a forged pass
                self.assertRefused(self.seal, "refusing to seal 1a")
                self.assertFalse(os.path.exists(os.path.join(self.a, h1.HANDOFF_NAME)))
        self.write_cell("state")
        self.write_cell("state", checks="not a dict")
        self.assertRefused(self.seal, "no checks block")
        self.write_cell("state")
        self.seal()

    def test_a_partial_progress_sidecar_never_qualifies_a_cell(self):
        self.write_cell("state", sidecar_only=True)  # a timeout/crash: sidecar (even claiming a final pass) but no completed gate report
        self.assertRefused(self.seal, "gate.json missing")
        self.write_cell("state")
        self.write_cell("prim", sidecar_only=True)
        self.assertRefused(self.seal, "gate.json missing")

    def test_missing_cell_and_missing_raw_files_are_refused(self):
        os.rename(os.path.join(self.a, "state"), os.path.join(self.a, "state-moved"))
        self.assertRefused(self.seal, "cell directory")
        os.rename(os.path.join(self.a, "state-moved"), os.path.join(self.a, "state"))
        for fn in ("vram.csv", "vram-events.jsonl", "vram-summary.json", "baseline.json"):
            path = os.path.join(self.a, "state", fn)
            data = open(path).read()
            os.remove(path)
            self.assertRefused(self.seal, fn)
            with open(path, "w") as f:
                f.write(data)
        self.seal()

    def test_tampered_or_removed_bound_files_block_1b(self):
        self.seal()
        for rel in ("state/gate.json", "state/vram.csv", "state/vram-summary.json", "state/baseline.json", "prim/gate.json", "state/rc.txt",
                    "state/gate.json.progress.jsonl", "state/spawn-env.json"):
            path = os.path.join(self.a, rel)
            original = open(path, "rb").read()
            with open(path, "ab") as f:
                f.write(b" ")
            self.assertRefused(lambda: h1.verify(self.stage, self.a), "changed since the seal")
            with open(path, "wb") as f:
                f.write(original)
            h1.verify(self.stage, self.a)
            data = original
            os.remove(path)
            self.assertRefused(lambda: h1.verify(self.stage, self.a), "missing")
            with open(path, "wb") as f:
                f.write(data)
        h1.verify(self.stage, self.a)

    def test_a_forged_handoff_cannot_launder_a_failed_cell(self):
        self.seal()
        self.write_cell("state", rc=1)  # the closed files now say failed; rebind them so the file hashes agree
        path = os.path.join(self.a, h1.HANDOFF_NAME)
        handoff = json.load(open(path))
        handoff["cells"]["state"]["files"] = {fn: h1.file_row(os.path.join(self.a, "state", fn)) for fn in handoff["cells"]["state"]["files"]
                                              if os.path.isfile(os.path.join(self.a, "state", fn))}
        with open(path, "w") as f:
            json.dump(handoff, f)
        self.assertRefused(lambda: h1.verify(self.stage, self.a), "rc 1")

    def test_handoff_header_must_say_partial_not_advance(self):
        self.seal()
        path = os.path.join(self.a, h1.HANDOFF_NAME)
        original = json.load(open(path))
        for key, value in (("advance", True), ("qualifies", False), ("scope", "full"), ("hold", "hold1b"), ("schema", "x")):
            handoff = dict(original)
            handoff[key] = value
            with open(path, "w") as f:
                json.dump(handoff, f)
            self.assertRefused(lambda: h1.verify(self.stage, self.a), "handoff")
        with open(path, "w") as f:
            f.write("not json")
        self.assertRefused(lambda: h1.verify(self.stage, self.a), "not a JSON object")
        os.remove(path)
        self.assertRefused(lambda: h1.verify(self.stage, self.a), "no 1a handoff")

    def test_runtime_binary_fixture_profile_or_harness_mismatch_blocks_1b(self):
        self.seal()
        stage = self.stage

        def mutate_file(rel, data=b" "):
            path = os.path.join(stage, rel)
            original = open(path, "rb").read()
            with open(path, "ab") as f:
                f.write(data)
            return lambda: open(path, "wb").write(original)

        for rel in ("bin-s1/qwen4_slowlink_gate", "bin-s1/qwen4_greedy_ids", "bin-s1/BUILD-IDENTITY.json", "bin-beta/qwen4_greedy_ids",
                    "fixtures/gate-fixture.json", "fixtures/prompts.short.jsonl", "fixtures/prompts.short.beta.jsonl", "target.json",
                    "ledger-hold1.json"):
            with self.subTest(rel):
                restore = mutate_file(rel)
                self.assertRefused(lambda: h1.verify(stage, self.a), "stage identity changed since 1a")
                restore()
                h1.verify(stage, self.a)
        # a different runtime source tip (and so a different manifest), and another product tile in the sealed profiles
        original = open(os.path.join(stage, h1.MANIFEST)).read()
        self.write_manifest(tip="e" * 40)
        self.assertRefused(lambda: h1.verify(stage, self.a), "manifest differs")
        self.write_manifest(product_tile=512)
        self.assertRefused(lambda: h1.verify(stage, self.a), "profiles differs")
        with open(os.path.join(stage, h1.MANIFEST), "w") as f:
            f.write(original)
        h1.verify(stage, self.a)
        # an extra file in a hashed directory is also a different identity
        with open(os.path.join(stage, "bin-s1", "extra"), "w") as f:
            f.write("x")
        self.assertRefused(lambda: h1.verify(stage, self.a), "bin_s1 differs in ['extra']")
        os.remove(os.path.join(stage, "bin-s1", "extra"))
        # the harness itself (a copy standing in for the harness directory)
        harness = os.path.join(self.tmp, "harness")
        os.makedirs(harness)
        with open(os.path.join(harness, "a.py"), "w") as f:
            f.write("1")
        saved = h1.HERE
        h1.HERE = harness
        try:
            self.seal()
            h1.verify(stage, self.a)
            with open(os.path.join(harness, "a.py"), "w") as f:
                f.write("2")
            self.assertRefused(lambda: h1.verify(stage, self.a), "harness_files differs in ['a.py']")
        finally:
            h1.HERE = saved

    def test_gate_report_of_another_manifest_or_fixture_after_a_restage_is_refused(self):
        self.seal()
        # restaging the fixtures changes the staged identity even though every A file is intact
        path = os.path.join(self.stage, "fixtures", "gate-fixture.json")
        with open(path, "a") as f:
            f.write(" ")
        self.assertRefused(lambda: h1.verify(self.stage, self.a), "fixtures differs")


class Receipt(World):
    def build_b(self, beta_pin="pcie", s1_pin="pcie", cmp_pass=True, headroom="pass", beta_d2="0", s1_d2="0"):
        b = os.path.join(self.tmp, "b-out")
        os.makedirs(b, exist_ok=True)
        for arm, outputs in h1.GREEDY_ARMS.items():
            d = os.path.join(b, arm)
            os.makedirs(d, exist_ok=True)
            for fn in outputs + ("command.txt", "stdout.txt", "stderr.txt", "evidence.json"):
                with open(os.path.join(d, fn), "w") as f:
                    f.write(f"{arm} {fn}\n")
            with open(os.path.join(d, "rc.txt"), "w") as f:
                f.write("0\n")
            pin = beta_pin if arm == "greedy-beta" else s1_pin
            d2 = beta_d2 if arm == "greedy-beta" else s1_d2
            with open(os.path.join(d, "spawn-env.json"), "w") as f:
                json.dump({"set": {**({"HIPFIRE_OFFLOAD_EXEC": pin} if pin else {}), **({"HIPFIRE_QWEN4_D2_DEFAULT": d2} if d2 else {})}}, f)
        os.makedirs(os.path.join(b, "headroom-state"), exist_ok=True)
        with open(os.path.join(b, "headroom-state", "stdout.txt"), "w") as f:
            json.dump({"status": headroom, "gate": {"min_free_bytes_required": 536870912}}, f)
        os.makedirs(os.path.join(b, "greedy-cmp"), exist_ok=True)
        with open(os.path.join(b, "greedy-cmp", "stdout.txt"), "w") as f:
            f.write("PASS: 7 records\n" if cmp_pass else "FAIL\n")
        return b

    def make(self, b):
        with contextlib.redirect_stdout(io.StringIO()):
            h1.receipt1b(self.stage, b, self.a)
        with open(os.path.join(b, h1.RECEIPT_NAME)) as f:
            return json.load(f)

    def test_1b_receipt_binds_a_the_inputs_the_proof_and_both_arm_pins(self):
        self.seal()
        r = self.make(self.build_b())
        self.assertEqual(r["schema"], h1.RECEIPT_SCHEMA)
        self.assertEqual(r["a_handoff"]["handoff"]["sha256"], sha(os.path.join(self.a, h1.HANDOFF_NAME)))
        self.assertEqual(r["headroom_state"]["source"], "the immutable 1a RAW state cell (no second GPU state run)")
        gi = r["greedy_inputs"]
        self.assertEqual(gi["canonical"]["sha256"], sha(os.path.join(self.stage, "fixtures", "prompts.short.jsonl")))
        self.assertEqual(gi["beta"]["sha256"], sha(os.path.join(self.stage, "fixtures", "prompts.short.beta.jsonl")))
        self.assertEqual(gi["adapter_proof"]["sha256"], sha(os.path.join(self.stage, "fixtures", "beta-input-proof.json")))
        self.assertEqual(gi["proof"]["stripped_keys"] if "stripped_keys" in gi["proof"] else None, ["resume_at"])
        for arm in ("greedy-beta", "greedy-s1"):
            self.assertEqual(r["greedy_arms"][arm]["spawn_env_pin"], {"HIPFIRE_OFFLOAD_EXEC": "pcie", "HIPFIRE_QWEN4_D2_DEFAULT": "0"})
            self.assertEqual(r["greedy_arms"][arm]["arm"], "non_default_d2_off")
            self.assertEqual(r["greedy_arms"][arm]["rc"], 0)
        self.assertIn("receipts.jsonl", r["greedy_arms"]["greedy-s1"]["files"])

    def test_1b_receipt_refuses_a_missing_pin_on_either_arm_a_bad_compare_or_headroom(self):
        self.seal()
        for kwargs in ({"beta_pin": None}, {"s1_pin": None}, {"beta_pin": "off"}, {"s1_pin": "cpu"}, {"cmp_pass": False}, {"headroom": "incomplete"},
                       {"beta_d2": None}, {"s1_d2": None}, {"beta_d2": "1"}, {"s1_d2": "1"}):
            with self.subTest(kwargs):
                self.assertRefused(lambda: self.make(self.build_b(**kwargs)), "refusing 1b receipt")

    def test_1b_receipt_refuses_a_changed_a_or_changed_inputs(self):
        self.seal()
        b = self.build_b()
        with open(os.path.join(self.a, "state", "vram.csv"), "a") as f:
            f.write("tamper\n")
        self.assertRefused(lambda: self.make(b), "changed since the seal")


if __name__ == "__main__":
    unittest.main()
