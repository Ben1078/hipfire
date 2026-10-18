#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the R5 adaptive admission witness (admission_witness.py) and of the ledger/manifest handling of the REFERENCE resident
count (ledger_profiles.py). Synthetic accounting only (ledger_testlib.witness / quote_entries): nothing here is the real model's quote, a real
emit or a frozen SHA. Run: python3 -m unittest test_admission_adaptive"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import admission_witness as aw  # noqa: E402
import ledger_profiles as lp  # noqa: E402
import ledger_testlib as lt  # noqa: E402

N9 = lambda: lt.witness(lt.POST_INIT, lt.PRE_INIT, 9)  # noqa: E731
N10 = lambda: lt.witness(lt.PRE_INIT, lt.PRE_INIT, 10)  # noqa: E731


def base_free(n, extra):
    return lt.needed(n) + extra


def problems(w, **kw):
    return aw.witness_problems(w, check_actual=False, **kw)


def actual(w):
    return {"resident": w["resident_layers"], "actual_trunk": w["trunk_layers"], "snapshot_budget": w["slack_bytes"]}


class Witness(unittest.TestCase):
    def test_n9_n10_and_n0_are_accepted_when_maximal(self):
        n9 = N9()
        self.assertEqual(n9["observed_initialization_bytes"], 153_092_096)  # the measured interval, never an assumed constant
        self.assertEqual(n9["preferred_deficit_bytes"], 50_000_000)  # ten is preferred: a deficit is information, not a rejection
        self.assertEqual(problems(n9), [])
        self.assertEqual(aw.witness_problems(n9, **actual(n9)), [])
        n10 = N10()
        self.assertEqual((n10["observed_initialization_bytes"], n10["preferred_deficit_bytes"]), (0, 0))
        self.assertEqual(problems(n10), [])
        n0 = lt.witness(base_free(0, 100), None, 0)  # N0 is valid when the base fits and host layers remain
        self.assertEqual(problems(n0), [])
        self.assertEqual(n0["slack_bytes"], 100)

    def test_exact_and_one_byte_boundaries(self):
        for extra, n in ((0, 9), (lt.LAYER - 1, 9), (lt.LAYER, 10), (lt.LAYER + 1, 10)):
            self.assertEqual(problems(lt.witness(base_free(9, extra), None, n)), [], (extra, n))
        # one byte short of the next layer's cost keeps nine; the exact cost makes nine non-maximal
        self.assertTrue(any("not maximal" in p for p in problems(lt.witness(base_free(9, lt.LAYER), None, 9))))
        # one byte below nine layers' cost does not fit nine
        self.assertTrue(any("oversubscribed" in p for p in problems(lt.witness(base_free(9, 0) - 1, None, 9))))
        # counts above the preferred ten are data when the actual free supports them
        self.assertEqual(problems(lt.witness(base_free(11, 5), None, 11)), [])

    def test_base_that_does_not_fit_has_no_valid_count(self):
        for n in (0, 1):
            w = lt.witness(base_free(0, 0) - 1, None, n)
            self.assertTrue(any("oversubscribed" in p for p in problems(w)), n)

    def test_all_resident_bypass_is_not_an_s1_claim(self):
        w = lt.witness(base_free(48, 5), None, 48)
        self.assertTrue(any("no host layer" in p for p in problems(w)))
        w = lt.witness(base_free(49, 5), None, 49)
        self.assertTrue(any("no host layer" in p for p in problems(w)))

    def test_a_forged_count_that_fits_nothing_is_oversubscribed(self):
        w = lt.witness(lt.POST_INIT, lt.PRE_INIT, 10)  # ten claimed at the post-initialization sample: 50,000,000 B short
        self.assertLess(w["slack_bytes"], 0)
        self.assertTrue(any("oversubscribed" in p for p in problems(w)))

    def test_missing_null_and_tampered_witnesses_are_refused(self):
        self.assertTrue(aw.witness_problems(None))
        self.assertTrue(aw.witness_problems("x"))
        self.assertTrue(aw.witness_problems([]))
        w = N9()
        self.assertEqual(len(aw.ALL_KEYS), 14)
        for key in aw.ALL_KEYS:
            gone = {k: v for k, v in w.items() if k != key}
            self.assertTrue(problems(gone), f"removing {key}")
            nulled = {**w, key: None}
            if key in aw.OPTIONAL_KEYS:
                self.assertTrue(problems(nulled), f"nulling {key} breaks the measured interval")
            else:
                self.assertTrue(problems(nulled), f"nulling {key}")
        for key in aw.INT_KEYS:
            for step in (1, -1):
                if key == "trunk_layers":
                    continue  # the trunk is tied to the loaded model, not to the scalars (see the bundle-facts test)
                self.assertTrue(problems({**w, key: w[key] + step}), f"{key} {step:+}")
            self.assertTrue(problems({**w, key: True}), f"{key} bool")
            self.assertTrue(problems({**w, key: float(w[key])}), f"{key} float")
        self.assertTrue(problems({**w, "preferred_deficit_bytes": 0}))  # a forged zero deficit
        self.assertTrue(problems({**w, "observed_initialization_bytes": 0}))  # a forged loss-free interval
        self.assertTrue(problems({**w, "free_before_initialization_bytes": lt.PRE_INIT + 1}))
        self.assertTrue(problems({**w, "memory_basis": "pre_load"}))
        self.assertTrue(problems({**w, "initialization_charged_via_free": False}))
        self.assertTrue(problems({**w, "initialization_charged_via_free": "true"}))
        forged = {**w, "resident_layers": 10, "needed_bytes": lt.needed(10), "slack_bytes": 0}
        self.assertTrue(any("oversubscribed" in p for p in problems(forged)))

    def test_the_initialization_interval_is_explicit_and_never_invented(self):
        gained = lt.witness(lt.PRE_INIT, lt.POST_INIT, 10)  # the fresh free ROSE: actual stays authoritative, no positive tax is invented
        self.assertEqual(gained["observed_initialization_bytes"], 0)
        self.assertEqual(problems(gained), [])
        self.assertEqual(problems({**gained, "observed_initialization_bytes": None}), [])
        self.assertTrue(problems({**gained, "observed_initialization_bytes": 1}))
        nobefore = {**N9(), "free_before_initialization_bytes": None}
        self.assertTrue(problems(nobefore))  # observed must then be null too
        self.assertEqual(problems({**nobefore, "observed_initialization_bytes": None}), [])

    def test_the_bundles_own_facts_must_equal_the_witness(self):
        w = N9()
        good = actual(w)
        self.assertEqual(aw.witness_problems(w, **good), [])
        for key, bad in (("resident", 10), ("resident", None), ("actual_trunk", 47), ("actual_trunk", None),
                         ("snapshot_budget", w["slack_bytes"] + 1), ("snapshot_budget", None)):
            self.assertTrue(aw.witness_problems(w, **{**good, key: bad}), (key, bad))

    def test_cli_verifies_a_witness_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.json")
            with open(path, "w") as f:
                json.dump(N9(), f)
            ok = subprocess.run([sys.executable, os.path.join(HERE, "admission_witness.py"), "verify", path], capture_output=True, text=True)
            self.assertEqual((ok.returncode, ok.stdout.strip()), (0, "admission witness: ok"))
            with open(path, "w") as f:
                json.dump({**N9(), "resident_layers": 8}, f)
            bad = subprocess.run([sys.executable, os.path.join(HERE, "admission_witness.py"), "verify", path], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("admission witness:", bad.stderr)


def refused(test, led, fragment, final=True):
    with test.assertRaises(SystemExit) as cm:
        lp.check_ledger(led, final=final)
    test.assertIn(fragment, str(cm.exception))


def refused_production(test, led, fragment):
    """The PRODUCTION rule (the seal's `canonical=post_initialization_actual`): requirements that exist only for production sealing (the raw
    host_owners echo, the preserved source_observation_label, the canonical binding) are asserted through THIS helper; the plain `refused` is the
    permissive legacy-fixture path (canonical=None) and must never be expected to refuse them."""
    with test.assertRaises(SystemExit) as cm:
        lp.check_ledger(led, final=True, canonical=lp.OBS_POST)
    test.assertIn(fragment, str(cm.exception))


class LedgerReference(unittest.TestCase):
    def test_reference_count_is_data_not_a_floor_ten_is_not_required(self):
        for ref in (0, 1, 9, 10):
            led = lt.final_ledger(ref)
            lp.check_ledger(led)
            self.assertEqual({led["profiles"][n]["resident_expert_layers"] for n in lp.PROFILES}, {ref})
            self.assertEqual(led["device_weights"]["resident_expert_bytes"], ref * led["device_weights"]["expert_layer_bytes"])

    def test_a_reference_count_that_leaves_no_host_layer_or_is_not_a_count_is_refused(self):
        led = lt.final_ledger()
        led["fixture"]["resident_expert_layers"] = lp.TRUNK_LAYERS
        refused(self, led, "reference count")
        for bad in (True, -1, "10", 10.5):
            led = lt.final_ledger()
            led["profiles"]["product"]["resident_expert_layers"] = bad
            refused(self, led, "resident_expert_layers")
        led = lt.final_ledger()
        led["profiles"]["product"]["resident_expert_layers"] = 9  # disagrees with the shared reference count
        refused(self, led, "fixture reference")

    def test_projection_role_must_say_reference_everywhere(self):
        for path in (("fixture",), ("device_weights",), ("device_projection",), ("profiles", "product"), ("profiles", "primitive_selftest")):
            led = lt.final_ledger()
            node = led
            for part in path:
                node = node[part]
            node["projection_role"] = "observed"
            with self.assertRaises(SystemExit):
                lp.check_ledger(led)
            led = lt.final_ledger()
            node = led
            for part in path:
                node = node[part]
            del node["projection_role"]
            with self.assertRaises(SystemExit):
                lp.check_ledger(led)
        led = lt.final_ledger()
        led["device_projection"]["memory_basis"] = "post_initialization_actual"
        refused(self, led, "reference_pre_initialization")

    def test_a_final_ledger_needs_the_four_bound_quotes_and_the_runtime_commit(self):
        led = lt.final_ledger()
        got = lp.check_ledger(led)
        self.assertEqual(len(led["admission_quotes"]["entries"]), 4)
        self.assertEqual(got["peak_owned"], 24054058348)
        led["admission_quotes"]["entries"] = []
        refused(self, led, "no admission_quotes.entries")
        led = lt.final_ledger()
        led["admission_quotes"]["entries"].pop(0)
        refused(self, led, "lacks the")
        led = lt.final_ledger()
        led["admission_quotes"]["source_commit"] = "d" * 40
        refused(self, led, "source_commit")
        led = lt.final_ledger()
        led["admission_quotes"]["schema"] = "OLD"
        refused(self, led, "admission_quotes")
        led = lt.final_ledger()
        del led["admission_quotes"]
        refused(self, led, "missing root keys")

    def test_each_quote_is_reverified_independently(self):
        for index in range(4):
            for edit in (lambda w: w.update({"resident_layers": w["resident_layers"] + 1}),
                         lambda w: w.update({"slack_bytes": w["slack_bytes"] + 1}),
                         lambda w: w.update({"memory_basis": "pre_load"}),
                         lambda w: w.update({"preferred_deficit_bytes": w["preferred_deficit_bytes"] + 1})):
                led = lt.final_ledger()
                edit(led["admission_quotes"]["entries"][index]["witness"])
                with self.assertRaises(SystemExit):
                    lp.check_ledger(led)

    def test_quote_structure_rules(self):
        entries = lambda led: led["admission_quotes"]["entries"]  # noqa: E731
        led = lt.final_ledger()
        entries(led)[0]["tile_rows"] = 512
        refused(self, led, "tile_rows")
        led = lt.final_ledger()
        entries(led)[1]["memory_observation"] = lp.OBS_POST
        refused(self, led, "repeats")
        led = lt.final_ledger()
        entries(led)[0]["purpose"] = "other"
        refused(self, led, "purpose")
        # reference observation: no initialization loss; post observation: a measured before sample is mandatory
        led = lt.final_ledger()
        w = entries(led)[1]["witness"]  # product / reference
        w["free_before_initialization_bytes"] = w["free_at_admission_bytes"] + 7
        w["observed_initialization_bytes"] = 7
        refused(self, led, "must equal free at admission")
        led = lt.final_ledger()
        w = entries(led)[0]["witness"]  # product / post
        w["free_before_initialization_bytes"] = None
        w["observed_initialization_bytes"] = None
        refused(self, led, "before sample is missing")
        # one quote, two free samples: the scalars must not differ between the observations of one purpose
        led = lt.final_ledger()
        entries(led)[0]["witness"]["joint_reserve_bytes"] += 1
        with self.assertRaises(SystemExit):
            lp.check_ledger(led)
        # the reference observation selects the profile's reference count; the post count cannot exceed it
        led = lt.final_ledger()
        entries(led)[1]["witness"] = lt.witness(lt.needed(9) + lt.REFERENCE_SLACK, lt.needed(9) + lt.REFERENCE_SLACK, 9)
        entries(led)[1]["host_owners"] = lt.host_owners(9)
        entries(led)[0]["witness"] = lt.witness(lt.needed(8) + 1, lt.needed(8) + 1 + lt.INIT_LOSS, 8)
        entries(led)[0]["host_owners"] = lt.host_owners(8)
        refused(self, led, "reference count")

    def test_nonfinal_ledger_may_hold_no_quotes_only_when_pending_names_them(self):
        led = lt.draft_ledger()
        lp.check_ledger(led, final=False)
        led["pending_final"] = ["something else"]
        refused(self, led, "admission_quotes", final=False)
        # a FINAL-marked ledger (census_final true, nothing pending) without entries is refused as final
        led = lt.draft_ledger()
        led["census_final"], led["pending_final"] = True, []
        refused(self, led, "no admission_quotes.entries")

    def test_manifest_seals_reference_profiles_and_quotes(self):
        import make_manifest as mm
        led = lt.final_ledger()
        self.assertTrue(all(led["profiles"][n]["projection_role"] == "reference" for n in lp.PROFILES))
        self.assertIn("projection_role", lp.PROFILE_KEYS)
        sealed = {n: copy.deepcopy({k: led["profiles"][n][k] for k in lp.PROFILE_KEYS}) for n in lp.PROFILES}
        for n in lp.PROFILES:
            self.assertEqual(sealed[n]["projection_role"], "reference")
        self.assertIs(mm.lp, lp)

class CanonicalBinding(unittest.TestCase):
    """The production ledger's reference derives from the ACTUAL post-initialization quote (N9 at the j1811 free); the earlier no-loss N10 row is a
    counterfactual kept as information only and can never be bound as the canonical reference."""

    def prod(self):
        return lt.production_ledger()

    def test_final_n9_post_initialization_reference_is_accepted_for_production(self):
        led = self.prod()
        got = lp.check_ledger(led, canonical=lp.OBS_POST)
        self.assertEqual(got["canonical_observation"], lp.OBS_POST)
        self.assertEqual(led["admission_quotes"]["canonical_observation"], "post_initialization_actual")
        self.assertEqual({led["profiles"][n]["resident_expert_layers"] for n in lp.PROFILES}, {9})
        self.assertEqual(led["fixture"]["resident_expert_layers"], 9)
        self.assertEqual(led["device_projection"]["preload_free_bytes"], lt.POST_INIT)  # the j1811 fresh free 25,560,088,576
        self.assertEqual(led["device_projection"]["memory_basis"], "post_initialization_actual")
        by_obs = {e["memory_observation"]: e["witness"] for e in led["admission_quotes"]["entries"] if e["purpose"] == "product"}
        self.assertEqual((by_obs[lp.OBS_POST]["resident_layers"], by_obs[lp.OBS_POST]["free_at_admission_bytes"]), (9, 25_560_088_576))
        self.assertEqual(by_obs[lp.OBS_POST]["observed_initialization_bytes"], 153_092_096)
        # the old pre-initialization N10 row is kept, but only as the counterfactual
        self.assertEqual((by_obs[lp.OBS_REFERENCE]["resident_layers"], by_obs[lp.OBS_REFERENCE]["free_at_admission_bytes"]), (10, 25_713_180_672))
        self.assertEqual(got["owned"]["product"], 24054058348 - 1337982976)

    def test_binding_the_profile_to_the_old_pre_init_n10_row_is_refused_in_production(self):
        # (a) the ledger declares the no-loss row canonical: the production requirement refuses it
        legacy = lt.final_ledger()  # explicit unit-fixture choice: canonical reference at N10
        lp.check_ledger(legacy)  # accepted as a fixture with no production requirement
        with self.assertRaises(SystemExit) as cm:
            lp.check_ledger(legacy, canonical=lp.OBS_POST)
        self.assertIn("counterfactual", str(cm.exception))
        # (b) the ledger says post-initialization but its profiles still carry the old N10 reference
        led = self.prod()
        lt.retarget(led, 10)
        with self.assertRaises(SystemExit) as cm:
            lp.check_ledger(led, canonical=lp.OBS_POST)
        self.assertIn("reference count", str(cm.exception))
        self.assertIn("counterfactual", str(cm.exception))
        # (c) labels are not swapped: the N9 post row's figures cannot stand in for the counterfactual row or vice versa
        led = self.prod()
        for e in led["admission_quotes"]["entries"]:
            e["memory_observation"] = lp.OBS_REFERENCE if e["memory_observation"] == lp.OBS_POST else lp.OBS_POST
        with self.assertRaises(SystemExit):
            lp.check_ledger(led, canonical=lp.OBS_POST)

    def test_canonical_field_is_required_valid_and_non_null_when_quotes_exist(self):
        led = self.prod()
        del led["admission_quotes"]["canonical_observation"]
        refused(self, led, "canonical_observation is missing")
        for bad in ("post_initialization", "pre_load", 9, True):
            led = self.prod()
            led["admission_quotes"]["canonical_observation"] = bad
            refused(self, led, "canonical_observation")
        led = self.prod()
        led["admission_quotes"]["canonical_observation"] = None
        refused(self, led, "canonical_observation is null")
        # a draft without entries may only leave it null
        draft = lt.draft_ledger()
        self.assertIsNone(draft["admission_quotes"]["canonical_observation"])
        lp.check_ledger(draft, final=False)
        draft["admission_quotes"]["canonical_observation"] = lp.OBS_POST
        refused(self, draft, "no quote entries back it", final=False)

    def test_projection_is_made_at_the_canonical_quotes_own_free_and_basis(self):
        led = self.prod()
        led["device_projection"]["memory_basis"] = "reference_pre_initialization"
        refused(self, led, "post_initialization_actual")
        led = self.prod()
        led["device_projection"]["preload_free_bytes"] += 1
        refused(self, led, "canonical")
        # the legacy fixture's projection is likewise tied to ITS canonical row (the pre-initialization free), never to the other row
        legacy = lt.final_ledger()
        legacy["device_projection"]["preload_free_bytes"] = lt.POST_INIT
        refused(self, legacy, "canonical")

    def test_each_purposes_canonical_count_must_equal_its_profile_reference_count(self):
        for index in (0, 2):  # product / primitive_selftest post rows
            led = self.prod()
            w = led["admission_quotes"]["entries"][index]["witness"]
            self.assertEqual(w["resident_layers"], 9)
            led["admission_quotes"]["entries"][index]["witness"] = lt.witness(lt.needed(8) + lt.POST_SLACK, lt.needed(8) + lt.POST_SLACK + lt.INIT_LOSS, 8)
            with self.assertRaises(SystemExit):
                lp.check_ledger(led)

    def test_a_no_loss_n10_counterfactual_is_information_not_a_constraint(self):
        led = self.prod()
        for e in led["admission_quotes"]["entries"]:
            if e["memory_observation"] == lp.OBS_REFERENCE:
                # a differently-sized counterfactual is still accepted when it is a valid maximal quote not below the post count
                e["witness"] = lt.witness(lt.needed(11) + 5, lt.needed(11) + 5, 11)
                e["host_owners"] = lt.host_owners(11)
        lp.check_ledger(led, canonical=lp.OBS_POST)

    def test_cli_production_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, bad = os.path.join(tmp, "p.json"), os.path.join(tmp, "l.json")
            with open(ok, "w") as f:
                json.dump(self.prod(), f)
            with open(bad, "w") as f:
                json.dump(lt.final_ledger(), f)
            run = lambda path: subprocess.run([sys.executable, os.path.join(HERE, "ledger_profiles.py"), "check", "--production", path],  # noqa: E731
                                              capture_output=True, text=True)
            good = run(ok)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)["canonical_observation"], "post_initialization_actual")
            self.assertNotEqual(run(bad).returncode, 0)



class HostBinding(unittest.TestCase):
    """The production host census is bound to the SAME canonical quote through its raw `host_owners` echo (synthetic fixture data: 49 weight layers,
    per-layer payload/guard from ledger_testlib; none of it is a validator rule)."""

    def prod(self):
        return lt.production_ledger()

    @staticmethod
    def row(led, purpose="product", obs=lp.OBS_POST):
        return next(e for e in led["admission_quotes"]["entries"] if e["purpose"] == purpose and e["memory_observation"] == obs)

    @staticmethod
    def terms(led, purpose="product"):
        return {t["name"]: t for t in led["host_census"][purpose]["terms"]}

    def test_production_n9_host_payload_and_pad_from_the_canonical_quote_are_accepted(self):
        led = self.prod()
        got = lp.check_ledger(led, canonical=lp.OBS_POST)
        sel = self.row(led)["host_owners"]["selected"]
        self.assertEqual((sel["resident_layers"], sel["host_layers"]), (9, 40))
        self.assertEqual((sel["host_payload_bytes"], sel["host_owner_pad_bytes"]), (40 * 1_336_934_400, 40 * 2_097_152))
        self.assertEqual(self.terms(led)[lp.HOST_PAYLOAD_TERM]["bytes"], 53_477_376_000)
        self.assertEqual(self.terms(led)[lp.HOST_PAD_TERM]["bytes"], 83_886_080)
        # the honest N9 totals: S1 host census 56,416,671,744 (> beta 55,740,268,544) + 2 GiB allowance = 58,564,155,392 -> 55 GiB, slack 491,644,928
        self.assertEqual(got["host_peak"], 56_416_671_744)
        self.assertEqual(led["host_reservation_owned_bytes"], 56_416_671_744)
        self.assertEqual(got["host_total"], 58_564_155_392)
        self.assertEqual((got["host_gib"], led["fabric_hostmem_gib"], led["fabric_hostmem_bytes"]), (55, 55, 59_055_800_320))
        self.assertEqual(led["host_slack_bytes"], 491_644_928)
        self.assertEqual(led["host_census"]["primitive_selftest"]["total_bytes"], 56_415_358_976)
        # the source labels are preserved and mapped, never swapped
        self.assertEqual(self.row(led)["source_observation_label"], "actual_post_initialization")
        self.assertEqual(self.row(led, obs=lp.OBS_REFERENCE)["source_observation_label"], "reference_pre_equals_post")

    def test_stale_n10_host_terms_under_an_n9_canonical_are_rejected(self):
        for purpose in lp.PROFILES:
            led = self.prod()
            n10 = lt.host_block(10)
            for t in led["host_census"][purpose]["terms"]:
                if t["name"] == lp.HOST_PAYLOAD_TERM:
                    t["bytes"] = n10["host_payload_bytes"]
                elif t["name"] == lp.HOST_PAD_TERM:
                    t["bytes"] = n10["host_owner_pad_bytes"]
            led["host_census"][purpose]["total_bytes"] = sum(t["bytes"] for t in led["host_census"][purpose]["terms"])
            refused(self, led, "stale")
        # the whole host section left at the carried-forward R4 N=10 arithmetic (as the draft template holds it) is stale too
        led = self.prod()
        old = lt.final_ledger()  # the legacy N10 fixture: 39 host layers, the carried-forward R4 host arithmetic
        for key in ("host_census", "host_owned_bytes", "host_reservation_owned_bytes", "host_total_with_allowance_bytes", "fabric_hostmem_gib",
                    "fabric_hostmem_bytes", "host_slack_bytes"):
            led[key] = old[key]
        refused(self, led, "stale")
        # a single-byte drift in either term is refused as well
        for term in (lp.HOST_PAYLOAD_TERM, lp.HOST_PAD_TERM):
            led = self.prod()
            self.terms(led)[term]["bytes"] += 1
            led["host_census"]["product"]["total_bytes"] += 1
            with self.assertRaises(SystemExit):
                lp.check_ledger(led, canonical=lp.OBS_POST)

    def test_tampered_selected_count_host_layer_closure_and_payload_plus_pad_are_rejected(self):
        cases = {
            "selected resident count": (lambda ho: ho["selected"].update({"resident_layers": 10, "host_layers": 39}), "selected.resident_layers"),
            "host layer closure": (lambda ho: ho["selected"].update({"host_layers": 39}), "weight_layers_with_experts"),
            "weight layers below trunk": (lambda ho: ho.update({"weight_layers_with_experts": 47}), "trunk_layers"),
            "total != payload + pad": (lambda ho: ho["selected"].update({"host_total_bytes": ho["selected"]["host_total_bytes"] + 1}), "payload"),
            "preferred closure": (lambda ho: ho["preferred"].update({"host_layers": 40}), "weight_layers_with_experts"),
            "preferred count": (lambda ho: ho["preferred"].update({"resident_layers": 9, "host_layers": 40}), "preferred"),
            "missing selected key": (lambda ho: ho["selected"].pop("host_owner_entries"), "exactly"),
            "extra key": (lambda ho: ho.update({"extra": 1}), "raw object"),
            "non-integer payload": (lambda ho: ho["selected"].update({"host_payload_bytes": 1.5}), "exactly"),
        }
        for label, (mutate, fragment) in cases.items():
            led = self.prod()
            mutate(self.row(led)["host_owners"])
            with self.subTest(label):
                refused(self, led, fragment)
        # payload moved without moving the total (or vice versa) cannot match the census terms and the closure at once
        led = self.prod()
        sel = self.row(led)["host_owners"]["selected"]
        sel["host_payload_bytes"] += 5
        sel["host_total_bytes"] += 5
        refused(self, led, "stale")

    def test_the_counterfactual_row_never_drives_the_canonical_host_totals(self):
        led = self.prod()
        cf = self.row(led, obs=lp.OBS_REFERENCE)["host_owners"]
        self.assertEqual(cf["selected"]["resident_layers"], 10)  # N10 counterfactual: 39 host layers
        # a counterfactual row that stays self-consistent but differs in its payload/pad figures changes nothing: it never drives the host totals
        cf["selected"]["host_payload_bytes"] += 5
        cf["selected"]["host_total_bytes"] += 5
        cf["selected"]["host_owner_entries"] += 7
        lp.check_ledger(led, canonical=lp.OBS_POST)
        # ...but an INCONSISTENT counterfactual echo is still refused as a malformed emit
        led = self.prod()
        self.row(led, obs=lp.OBS_REFERENCE)["host_owners"]["selected"]["host_total_bytes"] += 1
        refused(self, led, "payload")
        # and binding the host census to the counterfactual's N10 payload is exactly what is refused
        led = self.prod()
        n10 = lt.host_block(10)
        for t in led["host_census"]["product"]["terms"]:
            if t["name"] == lp.HOST_PAYLOAD_TERM:
                t["bytes"] = self.row(led, obs=lp.OBS_REFERENCE)["host_owners"]["selected"]["host_payload_bytes"]
        self.assertEqual(self.terms(led)[lp.HOST_PAYLOAD_TERM]["bytes"], n10["host_payload_bytes"])
        led["host_census"]["product"]["total_bytes"] = sum(t["bytes"] for t in led["host_census"]["product"]["terms"])
        refused(self, led, "stale")

    def test_production_requires_the_host_echo_and_the_source_label_legacy_fixtures_stay_permissive(self):
        led = self.prod()
        for e in led["admission_quotes"]["entries"]:
            if e["memory_observation"] == lp.OBS_POST:
                del e["host_owners"]
        refused_production(self, led, "no host_owners")  # the PRODUCTION seal requires the raw echo on the canonical rows
        lp.check_ledger(led)  # canonical=None: the permissive legacy-fixture path accepts a ledger lacking the echo
        led = self.prod()
        del self.row(led)["source_observation_label"]
        refused_production(self, led, "source_observation_label")
        lp.check_ledger(led)  # permissive without the production requirement
        # legacy N10 fixture: echo present and consistent at its own canonical (reference) row
        legacy = lt.final_ledger()
        lp.check_ledger(legacy)
        self.assertEqual(self.terms(legacy)[lp.HOST_PAYLOAD_TERM]["bytes"], 52_140_441_600)
        self.assertEqual(legacy["host_owned_bytes"], 55_077_640_192)

    def test_source_label_mapping_never_swaps_rows(self):
        led = self.prod()
        self.row(led)["source_observation_label"] = "reference_pre_equals_post"  # an N9 post row carrying the no-loss label
        refused(self, led, "source_observation_label")
        led = self.prod()
        self.row(led)["source_observation_label"] = "unknown_label"
        refused(self, led, "source_observation_label")
        led = self.prod()
        self.row(led, obs=lp.OBS_REFERENCE)["source_observation_label"] = "actual_post_initialization"
        refused(self, led, "source_observation_label")

    def test_weight_layer_count_and_entries_per_layer_are_fixture_data_not_rules(self):
        # a different weight-layer count / entries-per-layer closes as long as the raw echo is self-consistent and the census follows it
        led = self.prod()
        for e in led["admission_quotes"]["entries"]:
            ho = e["host_owners"]
            for block in (ho["selected"], ho["preferred"]):
                block["host_owner_entries"] = block["host_layers"] * 3
            self.assertGreaterEqual(ho["weight_layers_with_experts"], e["witness"]["trunk_layers"])
        lp.check_ledger(led, canonical=lp.OBS_POST)

    def test_manifest_seals_the_raw_host_echo_and_the_n9_host_totals(self):
        led = self.prod()
        self.assertEqual(led["admission_quotes"]["entries"][0]["host_owners"]["selected"]["host_total_bytes"], 40 * (1_336_934_400 + 2_097_152))
        self.assertTrue(all("host_owners" in e and "source_observation_label" in e for e in led["admission_quotes"]["entries"]))


if __name__ == "__main__":
    unittest.main()
