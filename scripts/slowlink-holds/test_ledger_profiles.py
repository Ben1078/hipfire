#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the per-purpose ledger (ledger_profiles.py, make_manifest.py), the beta input adapter (beta_input.py,
make_fixtures.py) and the forensic evidence index (collect_evidence.py, hold-lib.sh finish_summary). Synthetic temp files only: no GPU,
no model, no staged binary is ever executed.

    python3 scripts/slowlink-holds/test_ledger_profiles.py -v
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beta_input  # noqa: E402
import collect_evidence as ce  # noqa: E402
import ledger_profiles as lp  # noqa: E402
import ledger_testlib as lt  # noqa: E402
import make_fixtures  # noqa: E402
import make_manifest as mm  # noqa: E402

LIB = os.path.join(HERE, "hold-lib.sh")


def refused(test, led, fragment, final=True):
    with test.assertRaises(SystemExit) as cm:
        lp.check_ledger(led, final=final)
    test.assertIn(fragment, str(cm.exception))


class Profiles(unittest.TestCase):
    def test_valid_product_768_and_primitive_512_ledger_is_accepted(self):
        led = lt.final_ledger()
        got = lp.check_ledger(led)
        self.assertEqual(got["owned"], {"product": 24054058348, "primitive_selftest": 23988959724})
        self.assertEqual(got["peak_owned"], 24054058348)
        self.assertEqual((got["host_peak"], got["host_total"], got["host_gib"]), (55077640192, 57887752192, 54))
        self.assertEqual([led["profiles"][n]["tile_rows"] for n in lp.PROFILES], [768, 512])
        for name in lp.PROFILES:
            self.assertEqual(set(led["profiles"][name]), set(lp.PROFILE_KEYS))

    def test_shipped_template_is_the_final_r5_production_ledger_bound_to_the_canonical_n9_quote(self):
        led = lt.shipped_template()
        got = lp.check_ledger(led, canonical=lp.OBS_POST)  # the PRODUCTION rule: final, reference derived from the actual post-initialization quote
        self.assertTrue(led["census_final"])
        self.assertEqual(led["pending_final"], [])
        self.assertEqual(got["canonical_observation"], "post_initialization_actual")
        q = led["admission_quotes"]
        self.assertEqual((q["schema"], len(q["entries"])), ("SLOWLINK_ADMISSION_QUOTE_R5", 4))
        self.assertEqual(q["source_commit"], led["provenance"]["census_source_commit"])
        self.assertEqual(led["provenance"]["census_source_commit"], "40a4e3bee206a2bd96867b8fbcdec770fa2f0ce5")
        by = {(e["purpose"], e["memory_observation"]): e for e in q["entries"]}
        for purpose in lp.PROFILES:
            post, cf = by[(purpose, lp.OBS_POST)], by[(purpose, lp.OBS_REFERENCE)]
            self.assertEqual((post["witness"]["resident_layers"], post["witness"]["free_at_admission_bytes"]), (9, 25_560_088_576))
            self.assertEqual((cf["witness"]["resident_layers"], cf["witness"]["free_at_admission_bytes"]), (10, 25_713_180_672))  # counterfactual
            self.assertEqual(post["source_observation_label"], "actual_post_initialization")
            self.assertEqual(cf["source_observation_label"], "reference_pre_equals_post")
            self.assertEqual(led["profiles"][purpose]["resident_expert_layers"], 9)
        # the host census derives from the canonical N9 quote (40 host layers, 55 GiB), never from the N10 counterfactual
        self.assertEqual((got["host_peak"], got["host_total"], got["host_gib"]), (56_416_671_744, 58_564_155_392, 55))
        self.assertEqual(got["owned"], {"product": 22_716_075_372, "primitive_selftest": 22_650_976_748})
        # every resident-count-dependent figure is declared REFERENCE data, never observed or a target
        self.assertEqual(led["fixture"]["projection_role"], "reference")
        self.assertEqual(led["device_weights"]["projection_role"], "reference")
        self.assertEqual(led["device_projection"]["projection_role"], "reference")
        self.assertEqual(led["device_projection"]["memory_basis"], "post_initialization_actual")
        self.assertEqual(led["device_projection"]["preload_free_bytes"], 25_560_088_576)
        for name in lp.PROFILES:
            self.assertEqual(led["profiles"][name]["projection_role"], "reference")

    def test_explicit_draft_is_nonfinal_names_what_is_missing_and_never_passes_production(self):
        led = lt.draft_ledger()
        lp.check_ledger(led, final=False)
        refused(self, led, "not final")
        self.assertFalse(led["census_final"])
        names = " ".join(led["pending_final"])
        for what in ("admission_quotes", "census_source_commit"):
            self.assertIn(what, names)
        self.assertEqual(led["admission_quotes"]["entries"], [])  # nothing invented
        self.assertIsNone(led["admission_quotes"]["source_commit"])
        self.assertIsNone(led["admission_quotes"]["canonical_observation"])
        for key in lp.PROVENANCE_RUNTIME:
            self.assertNotIn(key, led["provenance"])
        with self.assertRaises(SystemExit):
            lp.check_ledger(led, canonical=lp.OBS_POST)

    def test_bound_final_ledger_carries_the_tile_terms_and_the_synthetic_quotes(self):
        led = lt.template()
        got = lp.check_ledger(led)  # final=True: no pending field, every sum exact, runtime source ids and quotes bound
        self.assertTrue(led["census_final"])
        self.assertEqual(led["pending_final"], [])
        self.assertEqual(got["owned"], {"product": 24054058348, "primitive_selftest": 23988959724})
        for name, tile_terms in (("product", 10), ("primitive_selftest", 10)):
            terms = led["profiles"][name]["s1_tile_census_terms"]
            self.assertEqual(len(terms), tile_terms)
            self.assertEqual(sum(t["bytes"] for t in terms), led["profiles"][name]["s1_tile_census_bytes"])
            self.assertEqual([t["name"] for t in terms].count(lp.COLD_F16_TERM), 1)
        by = {n: {t["name"]: t["bytes"] for t in led["profiles"][n]["s1_tile_census_terms"]} for n in lp.PROFILES}
        self.assertEqual((by["product"]["forward_bytes"], by["product"]["workspace_total_bytes"]), (1011232772, 23870984))
        self.assertEqual((by["primitive_selftest"]["forward_bytes"], by["primitive_selftest"]["workspace_total_bytes"]), (735014276, 15974664))
        for name in lp.PROFILES:
            self.assertEqual((by[name]["state_bytes"], by[name]["qsa_context_bytes"]), (68514080, 281149440))
            self.assertEqual([by[name][k] for k in ("qsa_selector_bytes", "mtp_bytes", "gather_bytes", "runtime_bytes")], [0, 0, 0, 0])
            self.assertEqual(len(led["profiles"][name]["s1_extras_terms"]), 8)
        for key in lp.PROVENANCE_RUNTIME:
            self.assertRegex(led["provenance"][key], r"^[0-9a-f]{40}$")

    PAD = "allocation_backing_padding_bytes"
    SNAP = "bounded_host_snapshot_cap_bytes"

    def test_r4_physical_backing_keeps_the_cold_f16_payload_once_and_names_the_padding(self):
        led = lt.template()
        want = {"product": (77638656, 8988416, 2761329452, 2537898560, 24054058348, 48509588, 0),
                "primitive_selftest": (85125120, 7939840, 2484701100, 2531607104, 23988959724, 113608212, 217821184)}
        for name, (tile_pad, extras_pad, tile_total, extras_total, owned, slack, diag) in want.items():
            p = led["profiles"][name]
            tile = {t["name"]: t["bytes"] for t in p["s1_tile_census_terms"]}
            extras = {t["name"]: t["bytes"] for t in p["s1_extras_terms"]}
            # the validator's cold term is the PAYLOAD bound, exactly once, and the padding is one separately named allocation term (no extra F16 term)
            self.assertEqual([t["name"] for t in p["s1_tile_census_terms"]].count(lp.COLD_F16_TERM), 1)
            self.assertEqual(tile[lp.COLD_F16_TERM], 1298923520)
            self.assertEqual(p["f16_shadow_cold_bound_bytes"], 1298923520)
            self.assertNotIn(lp.COLD_F16_TERM, extras)
            self.assertEqual((tile[self.PAD], extras[self.PAD]), (tile_pad, extras_pad))
            self.assertEqual([n for n in list(tile) + list(extras) if "f16" in n.lower()], [lp.COLD_F16_TERM])
            self.assertEqual((p["s1_tile_census_bytes"], p["s1_extras_bytes"]), (tile_total, extras_total))
            self.assertEqual(sum(v for k, v in tile.items() if k != self.PAD) + tile_pad, tile_total)
            self.assertEqual(sum(v for k, v in extras.items() if k != self.PAD) + extras_pad, extras_total)
            self.assertEqual(p["device_weights_bytes"], 18754830336)
            self.assertEqual((p["device_owned_bytes"], p["sequential_diag_bytes"]), (owned, diag))
            self.assertEqual(led["device_projection"]["by_purpose"][name]["slack_over_safety_bytes"], slack)
            # the 1 GiB runtime budget and the 512 MiB safety margin are unchanged
            self.assertEqual((p["runtime_budget_bytes"], p["safety_margin_bytes"]), (1 << 30, 512 << 20))
        # physical weights, payload kept separately inside the extra model metadata
        w = led["device_weights"]
        self.assertEqual((w["non_expert_bytes"], w["expert_layer_bytes"], w["resident_expert_bytes"]), (5375000576, 1337982976, 13379829760))
        self.assertEqual(w["non_expert_bytes"] + w["resident_expert_bytes"], 18754830336)
        meta = w["model_payload_metadata"]
        self.assertEqual((meta["non_expert_payload_bytes"], meta["expert_layer_payload_bytes"]), (5364055296, 1336934400))
        self.assertEqual(meta["total_payload_bytes"], 18733399296)
        self.assertEqual(meta["total_payload_bytes"], meta["non_expert_payload_bytes"] + 10 * meta["expert_layer_payload_bytes"])
        self.assertEqual(w["non_expert_bytes"] - meta["non_expert_payload_bytes"], meta["non_expert_backing_padding_bytes"])
        self.assertEqual(w["expert_layer_bytes"] - meta["expert_layer_payload_bytes"], meta["expert_layer_backing_padding_bytes"])
        self.assertEqual(led["device_owned_bytes"], 24054058348)

    def test_r4_cold_f16_backing_or_a_second_f16_term_is_still_refused(self):
        led = lt.final_ledger()
        for t in led["profiles"]["product"]["s1_tile_census_terms"]:
            if t["name"] == lp.COLD_F16_TERM:
                t["bytes"] = 1300234240  # the modelled BACKING can never stand in for the validated cold payload bound
        refused(self, led, lp.COLD_F16_TERM)
        led = lt.final_ledger()
        for t in led["profiles"]["primitive_selftest"]["s1_tile_census_terms"]:
            if t["name"] == self.PAD:
                t["name"] = "f16_shadow_backing_padding_bytes"
        refused(self, led, "F16")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_extras_terms"].append({"name": "f16_backing_padding_bytes", "bytes": 0})
        refused(self, led, "F16")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_tile_census_terms"].append({"name": self.PAD, "bytes": 0})
        refused(self, led, "repeats the term name")
        led = lt.final_ledger()
        for t in led["profiles"]["product"]["s1_tile_census_terms"]:
            if t["name"] == self.PAD:
                t["bytes"] -= 1
        refused(self, led, "s1_tile_census_terms sum")

    def test_r4_host_census_carries_the_bounded_snapshot_cap_and_the_beta_arm_still_dominates(self):
        led = lt.template()
        want = {"product": 55077640192, "primitive_selftest": 55076327424}
        for name, total in want.items():
            node = led["host_census"][name]
            terms = {t["name"]: t["bytes"] for t in node["terms"]}
            self.assertEqual(terms[self.SNAP], 2 << 30)
            self.assertEqual(node["total_bytes"], total)
            self.assertEqual(total, sum(terms.values()))
            self.assertLess(total, led["host_reservation"]["total_bytes"])
        self.assertEqual(led["host_owned_bytes"], 55077640192)
        self.assertEqual(led["host_reservation"]["total_bytes"], 55740268544)  # beta G2 unchanged and still the sequential max
        self.assertEqual(led["host_reservation_owned_bytes"], 55740268544)
        self.assertEqual(led["host_total_with_allowance_bytes"], 57887752192)
        self.assertEqual(led["fabric_hostmem_bytes"] - led["host_total_with_allowance_bytes"], 94306304)
        self.assertEqual((led["fabric_hostmem_gib"], led["fabric_reservation_device_gib"]), (54, 22.4))
        # the 2 GiB device session cache stays a nominal, uncommitted cap and the pending ceiling stays recorded (now inside the host cap)
        self.assertEqual(led["session_cache_budget_cap_bytes_not_committed"], 2 << 30)
        self.assertEqual(led["pending_working_set_ceiling_bytes_not_charged"], 620021760)
        self.assertIn("INSIDE the bounded host snapshot cap", led["pending_ceiling_note"])

    def test_r4_data_names_the_r4_source_as_reference_only_and_no_r3_identity_remains(self):
        led = lt.shipped_template()
        prov = led["provenance"]
        self.assertEqual((prov["r4_reference_census_source_commit"], prov["r4_reference_census_source_tree"],
                          prov["r4_reference_census_source_base"]),
                         ("32c507c7b5cecab23cf6e14b001d4d87898c0416", "83a3e87203e8af97ba6d1f97bb3712a69e2ff5bf",
                          "0a3b67cb12ef63b60e7ea40a66689b99e3dcf981"))
        self.assertIn("SOURCE-CENSUS-R5.json", led["source_emit"])
        self.assertIn("40a4e3bee206a2bd96867b8fbcdec770fa2f0ce5", led["source_emit"])
        self.assertFalse(os.path.exists(os.path.join(HERE, "SOURCE-CENSUS-R4.json")))  # the old census file is not part of the R5 harness seed
        for name in ("ledger-template.json", "ledger-static.json"):
            with open(os.path.join(HERE, name)) as f:
                body = f.read()
            for r3 in ("SOURCE-CENSUS-R3", "847b9d945de943e968226f509adafe105c7342a5", "e2511cdd4ca75f36f35acd5bc4bb6b752cdf8912",
                       "5855c27d52bacf519221ed444f37f06d4553034cb256c00238e850bf87e17a42", "156567700", "228104212"):
                self.assertNotIn(r3, body, (name, r3))
        # the real runtime identity is still supplied (and bound) externally: nothing here is a fabricated identity
        self.assertNotIn("binary_sha256", prov)
        self.assertNotIn("runtime_commit", prov)

    def test_explicit_nonfinal_copy_is_refused_where_final_is_needed_and_says_what_is_missing(self):
        led = lt.nonfinal_template()
        lp.check_ledger(led, final=False)
        self.assertFalse(led["census_final"])
        self.assertTrue(any("s1_tile_census_terms" in p for p in led["pending_final"]))
        refused(self, led, "not final")
        # the only tile term carried is the cold F16 bound; the unfilled remainder is the real getter remainder, withdrawn, not invented
        for name, rest in (("product", 1462405932), ("primitive_selftest", 1185777580)):
            p = led["profiles"][name]
            self.assertEqual(p["s1_tile_census_terms"], [{"name": lp.COLD_F16_TERM, "bytes": lp.COLD_F16_BYTES}])
            self.assertEqual(p["s1_tile_census_bytes"] - lp.COLD_F16_BYTES, rest)
            self.assertEqual(lt.tile_remainder(name), rest)
        # a non-final ledger may not under-fill the tile terms unless pending_final says so
        led["pending_final"] = ["provenance only"]
        refused(self, led, "sum to", final=False)
        # a FINAL ledger never under-fills, even with a pending list withdrawn
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_tile_census_terms"] = [t for t in led["profiles"]["product"]["s1_tile_census_terms"]
                                                              if t["name"] == lp.COLD_F16_TERM]
        refused(self, led, "s1_tile_census_terms sum")
        # a final ledger without the runtime source identity is refused
        led = lt.final_ledger()
        del led["provenance"]["census_source_commit"]
        refused(self, led, "census_source_commit")

    def test_wrong_product_tile_512_cannot_pass_as_the_768_profile(self):
        led = lt.final_ledger()
        led["profiles"]["product"]["tile_rows"] = 512
        refused(self, led, "768")
        # the whole primitive profile presented as the product profile
        led = lt.final_ledger()
        led["profiles"]["product"] = copy.deepcopy(led["profiles"]["primitive_selftest"])
        led["profiles"]["product"]["purpose"] = "product"
        refused(self, led, "768")
        # and the product's numbers under the primitive name
        led = lt.final_ledger()
        led["profiles"]["primitive_selftest"] = copy.deepcopy(led["profiles"]["product"])
        led["profiles"]["primitive_selftest"]["purpose"] = "primitive_selftest"
        refused(self, led, "512")
        # the historical generic 1536-row bound is not an acceptable profile tile
        led = lt.final_ledger()
        led["profiles"]["product"]["tile_rows"] = lp.GENERIC_MAX_TILE
        refused(self, led, "768")

    def test_static_derivation_matches_the_template(self):
        led = lt.template()
        with open(os.path.join(HERE, "ledger-static.json")) as f:
            static = json.load(f)
        dev = static["device"]
        self.assertEqual(dev["weights"]["total_bytes"], led["profiles"]["product"]["device_weights_bytes"])
        for name in lp.PROFILES:
            p, s = led["profiles"][name], dev["profiles"][name]
            self.assertEqual(s["tile_rows"], p["tile_rows"])
            self.assertEqual(s["tile_census_bytes"], p["s1_tile_census_bytes"])
            self.assertEqual(s["tile_census_non_f16_bytes"], p["s1_tile_census_bytes"] - lp.COLD_F16_BYTES)
            self.assertEqual(s["tile_census_independent_terms"], {lp.COLD_F16_TERM: lp.COLD_F16_BYTES})
            self.assertEqual(s["extras"]["total_bytes"], p["s1_extras_bytes"])
            self.assertEqual(sum(v for k, v in s["extras"].items() if k != "total_bytes"), p["s1_extras_bytes"])
            norm = lambda n: n.lower().replace(" ", "_").removesuffix("_bytes")  # the final template spells owner terms with spaces
            self.assertEqual({norm(k): v for k, v in s["extras"].items() if k != "total_bytes"}, {norm(t["name"]): t["bytes"] for t in p["s1_extras_terms"]})
            self.assertEqual(s["core_bytes"], p["device_core_bytes"])
            self.assertEqual(s["sequential_diag_bytes"], p["sequential_diag_bytes"])
            self.assertEqual(s["owned_bytes"], p["device_owned_bytes"])
            self.assertEqual(static["host"]["s1"][name]["total_bytes"], led["host_census"][name]["total_bytes"])
            self.assertEqual(static["device_projection"][name]["slack_over_safety_bytes"],
                             led["device_projection"]["by_purpose"][name]["slack_over_safety_bytes"])
        self.assertEqual(dev["root_owned_bytes"], led["device_owned_bytes"])
        self.assertEqual(static["host"]["reservation"]["sequential_max_bytes"], led["host_reservation_owned_bytes"])
        self.assertEqual(static["host"]["reservation"]["total_bytes"], led["host_total_with_allowance_bytes"])
        self.assertEqual(static["host"]["sequential_beta"]["total_bytes"], led["host_reservation"]["total_bytes"])
        self.assertEqual(static["device_projection"]["preload_free_bytes"], led["device_projection"]["preload_free_bytes"])

    def test_missing_wrong_or_aliased_profiles_are_refused(self):
        for name in lp.PROFILES:
            led = lt.final_ledger()
            del led["profiles"][name]
            refused(self, led, "profiles keys")
        led = lt.final_ledger()
        led["profiles"]["extra"] = copy.deepcopy(led["profiles"]["product"])
        refused(self, led, "profiles keys")
        led = lt.final_ledger()
        led["profiles"]["product"]["purpose"] = "primitive_selftest"
        refused(self, led, "purpose")
        for key in ("tile_rows", "device_weights_bytes", "s1_tile_terms", "s1_extras_terms", "sequential_diag_bytes", "projection"):
            led = lt.final_ledger()
            led[key] = 1
            refused(self, led, "flat")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_tile_terms"] = []  # an alias beside the exact key
        refused(self, led, "unexpected")
        led = lt.final_ledger()
        del led["profiles"]["product"]["s1_tile_census_terms"]
        refused(self, led, "missing")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_extras_terms"] = {"ring": 1}  # the retired dict form
        refused(self, led, "array")

    def test_inconsistent_root_peak_is_refused(self):
        peak = 24054058348
        for bad in (peak + 217821184, 23988959724, peak - 1, 0):
            led = lt.final_ledger()
            led["device_owned_bytes"] = bad
            refused(self, led, "MAX")
        led = lt.final_ledger()
        led["profiles"]["product"]["sequential_diag_bytes"] = 217821184  # the product purpose carries no diagnostic
        refused(self, led, "sequential_diag_bytes")
        led = lt.final_ledger()
        led["profiles"]["primitive_selftest"]["sequential_diag_bytes"] = 0
        refused(self, led, "sequential_diag_bytes")
        led = lt.final_ledger()
        led["profiles"]["product"]["device_core_bytes"] += 1
        refused(self, led, "core")
        led = lt.final_ledger()
        led["profiles"]["product"]["device_owned_bytes"] += 1
        refused(self, led, "owned")

    def test_duplicated_or_shadow_terms_are_refused(self):
        led = lt.final_ledger()
        terms = led["profiles"]["product"]["s1_tile_census_terms"]
        terms.append(next(dict(t) for t in terms if t["name"] == lp.COLD_F16_TERM))  # the cold F16 counted twice
        refused(self, led, "repeats the term name")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_tile_census_terms"][1]["name"] = "forward_f16_shadow_bytes"
        refused(self, led, "F16")
        led = lt.final_ledger()
        led["profiles"]["primitive_selftest"]["s1_extras_terms"].append({"name": lp.COLD_F16_TERM, "bytes": 0})
        refused(self, led, "F16")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_tile_census_terms"] = [t for t in led["profiles"]["product"]["s1_tile_census_terms"]
                                                              if t["name"] != lp.COLD_F16_TERM]
        refused(self, led, lp.COLD_F16_TERM)
        led = lt.final_ledger()
        led["profiles"]["product"]["f16_shadow_cold_bound_bytes"] += 1
        refused(self, led, "f16_shadow_cold_bound_bytes")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_extras_terms"][0]["bytes"] += 1
        refused(self, led, "s1_extras_terms sum")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_tile_census_terms"][1]["bytes"] -= 1
        refused(self, led, "s1_tile_census_terms sum")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_extras_terms"][1]["bytes"] = -1
        refused(self, led, "u64")
        led = lt.final_ledger()
        led["profiles"]["product"]["s1_extras_terms"][1]["extra"] = 1
        refused(self, led, "exactly")
        # a repeated JSON key is a duplicated shadow field
        base = lt.final_ledger()
        text = json.dumps(base).replace(f'"hold": {json.dumps(base["hold"])}', f'"hold": {json.dumps(base["hold"])}, "hold": "dup"', 1)
        self.assertEqual(text.count('"hold": "dup"'), 1)  # the shadow key really is in the document the guard must refuse
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "led.json")
            with open(path, "w") as f:
                f.write(text)
            with self.assertRaises(SystemExit) as cm:
                lp.load_ledger(path)
            self.assertIn("repeats the key", str(cm.exception))

    def test_host_census_follows_each_purposes_tile_and_the_reservation_is_a_max_not_a_sum(self):
        led = lt.final_ledger()
        hc = led["host_census"]["product"]
        for term in hc["terms"]:
            if term["name"] == "forward_ple_stage_bytes":
                term["bytes"] = 1536 * lp.PLE_ROW_BYTES  # the old 1536-row stage
        hc["total_bytes"] = sum(t["bytes"] for t in hc["terms"])
        refused(self, led, "forward_ple_stage_bytes")
        led = lt.final_ledger()
        beta, host = led["host_reservation"]["total_bytes"], led["host_owned_bytes"]
        led["host_reservation_owned_bytes"] = beta + host  # an arm SUM
        led["host_total_with_allowance_bytes"] = beta + host + led["host_allowance_bytes"]
        refused(self, led, "sequential MAX")
        led = lt.final_ledger()
        led["host_owned_bytes"] = led["host_census"]["primitive_selftest"]["total_bytes"]
        refused(self, led, "MAX")
        led = lt.final_ledger()
        led["host_slack_bytes"] += 1
        refused(self, led, "host_slack_bytes")
        led = lt.final_ledger()
        led["fabric_reservation_device_gib"] = 25
        refused(self, led, "exceeds the physical card")

    def test_final_needs_a_bound_runtime_identity(self):
        led = lt.final_ledger()
        del led["provenance"]["census_source_tree"]
        refused(self, led, "census_source_tree")
        led = lt.final_ledger()
        led["provenance"]["census_source_commit"] = "not-hex"
        refused(self, led, "census_source_commit")

    def test_print_memory_prints_22_4_and_54(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger-hold1.json")
            with open(path, "w") as f:
                json.dump(lt.final_ledger(), f)
            r = subprocess.run([sys.executable, os.path.join(HERE, "ledger_profiles.py"), "print-memory", path], capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stdout.split()), (0, ["22.4", "54"]), r.stderr)
            self.assertIn("SEPARATE 24 GiB", r.stderr)
            with open(path, "w") as f:
                json.dump(lt.nonfinal_template(), f)
            r = subprocess.run([sys.executable, os.path.join(HERE, "ledger_profiles.py"), "print-memory", path], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(r.stdout, "")
            with open(path, "w") as f:  # the shipped final template itself prints
                json.dump(lt.template(), f)
            r = subprocess.run([sys.executable, os.path.join(HERE, "ledger_profiles.py"), "print-memory", path], capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stdout.split()), (0, ["22.4", "54"]), r.stderr)


class ManifestSeal(unittest.TestCase):
    """make_manifest.py end to end on a synthetic stage: both profiles sealed verbatim, nothing flat, inputs bound."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stage = os.path.join(tmp.name, "stage")
        os.makedirs(os.path.join(self.stage, "bin-s1"))
        os.makedirs(os.path.join(self.stage, "fixtures"))
        gate = os.path.join(self.stage, "bin-s1", "qwen4_slowlink_gate")
        with open(gate, "wb") as f:
            f.write(b"synthetic, never executed")
        self.gate_sha = hashlib.sha256(b"synthetic, never executed").hexdigest()
        self.write_identity(self.gate_sha)
        with open(os.path.join(HERE, "model-identity.json")) as f:
            cached = json.load(f)
        ident = {"host": "hipx", "uuid": "GPU-1", "pci_walk": [{"bdf": "0000:00:00.0"}], "upstream_bottleneck_gbps": 7.0,
                 "upstream_bottleneck_bdf": "0000:00:00.0"}
        for key in ("model", "ple_sidecar"):
            ident[key] = {f: cached[key][f] for f in ("path", "size_bytes", "inode", "mtime_ns")}
        self.write("prep-identity.json", ident)
        text = os.path.join(tmp.name, "natural.txt")
        with open(text, "w") as f:
            f.write("natural text sentence. " * 200)
        with contextlib.redirect_stdout(io.StringIO()):
            make_fixtures.main(["make_fixtures.py", os.path.join(self.stage, "fixtures"), text])
        self.led = lt.production_ledger()  # the production seal binds canonical_observation post_initialization_actual (N9)
        self.write("ledger-hold1.json", self.led)

    def write(self, name, obj):
        with open(os.path.join(self.stage, name), "w") as f:
            json.dump(obj, f)

    def write_identity(self, gate_sha, tip=lt.TIP):
        self.write(os.path.join("bin-s1", "BUILD-IDENTITY.json"),
                   {"schema": "s1.build_identity.v1", "arm": "s1", "source": {"base": lt.BASE, "tip": tip, "tree": lt.TREE},
                    "binary_sha256": {"qwen4_slowlink_gate": gate_sha}, "kernel_code_object_hashes": {"k": "d" * 64},
                    "pack": {"manifest": {"arch": "gfx1100"}}})

    def seal(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mm.main(["make_manifest.py", self.stage, "hold1", lt.BASE, lt.TIP, "sidecar", "--stdout", *extra])
        return json.loads(out.getvalue())

    def test_both_profiles_are_sealed_verbatim_and_nothing_is_flat(self):
        m = self.seal()
        ml = m["memory_ledger"]
        self.assertEqual(set(ml["profiles"]), {"product", "primitive_selftest"})
        for name in lp.PROFILES:
            self.assertEqual(ml["profiles"][name], {k: self.led["profiles"][name][k] for k in lp.PROFILE_KEYS})
        self.assertEqual((ml["profiles"]["product"]["tile_rows"], ml["profiles"]["primitive_selftest"]["tile_rows"]), (768, 512))
        self.assertEqual(ml["device_owned_bytes"], max(p["device_owned_bytes"] for p in ml["profiles"].values()))
        self.assertEqual(ml["device_owned_bytes"], 24054058348 - 1337982976)  # N9: one resident layer fewer than the carried-forward N10 arithmetic
        for flat in lp.FLAT_ROOT_KEYS + ("owned_device_bytes", "s1_tile_terms"):
            self.assertNotIn(flat, ml)
        # N9: one more host layer than the carried-forward N10 arithmetic (+1,336,934,400 payload + 2,097,152 guard per layer): the S1 host census now
        # exceeds the beta arm, so it is the sequential maximum; + the 2 GiB allowance = 58,564,155,392 B, 55 GiB (slack 491,644,928)
        self.assertEqual(ml["owned_host_bytes"], 56416671744)
        self.assertEqual(ml["reservation_owned_host_bytes"], 56416671744)
        self.assertEqual(ml["host_unreclaimable_gib"], 58564155392 / 2**30)
        self.assertEqual(ml["fabric_exclusive_device_gib"], 22.4)
        self.assertEqual((ml["device_gib"], ml["physical_card_gib"]), (24.0, 24.0))
        self.assertEqual(m["source"]["tree"], lt.TREE)
        self.assertEqual(ml["ledger_provenance"]["census_source_commit"], lt.TIP)
        # the reference projection and the Main-bound R5 quotes ride into the manifest verbatim (the gate reads the reference role from the profiles)
        self.assertEqual(ml["admission_quotes"], self.led["admission_quotes"])
        self.assertEqual(ml["device_projection"]["projection_role"], "reference")
        for name in lp.PROFILES:
            self.assertEqual(ml["profiles"][name]["projection_role"], "reference")

    def test_both_beta_inputs_and_their_identity_proof_are_bound(self):
        m = self.seal()
        fx = os.path.join(self.stage, "fixtures")
        sha = lambda n: hashlib.sha256(open(os.path.join(fx, n), "rb").read()).hexdigest()
        inp = m["inputs"]
        self.assertEqual(inp["greedy_canonical"]["sha256"], sha("prompts.short.jsonl"))
        self.assertEqual(inp["greedy_beta"]["sha256"], sha("prompts.short.beta.jsonl"))
        self.assertNotEqual(inp["greedy_canonical"]["sha256"], inp["greedy_beta"]["sha256"])
        self.assertEqual(inp["beta_adapter"]["proof_sha256"], sha("beta-input-proof.json"))
        self.assertEqual(inp["beta_adapter"]["stripped_keys"], ["resume_at"])
        self.assertEqual(inp["model_sha256"], m["model"]["sha256"])

    def test_nonfinal_wrong_tile_unbound_identity_and_tampered_inputs_are_refused(self):
        led = lt.production_ledger()
        led["census_final"], led["pending_final"] = False, ["x"]
        self.write("ledger-hold1.json", led)
        with self.assertRaises(SystemExit):
            self.seal()
        led = lt.production_ledger()
        led["profiles"]["product"]["tile_rows"] = 512
        self.write("ledger-hold1.json", led)
        with self.assertRaises(SystemExit):
            self.seal()
        led = lt.production_ledger()
        led["provenance"]["census_source_commit"] = "0" * 40
        self.write("ledger-hold1.json", led)
        with self.assertRaises(SystemExit):
            self.seal()
        self.write("ledger-hold1.json", lt.production_ledger())
        self.seal()
        # the seal refuses a ledger bound to the no-loss counterfactual (the old pre-initialization N10 row) as the canonical reference
        self.write("ledger-hold1.json", lt.final_ledger())
        with self.assertRaises(SystemExit) as cm:
            self.seal()
        self.assertIn("canonical_observation", str(cm.exception))
        self.write("ledger-hold1.json", lt.production_ledger())
        beta = os.path.join(self.stage, "fixtures", "prompts.short.beta.jsonl")
        with open(beta, "a") as f:
            f.write("\n")
        with open(beta, "rb") as f:
            raw = f.read()
        with open(beta, "wb") as f:
            f.write(raw.replace(b'"id": "p512"', b'"id": "p513"', 1))
        with self.assertRaises(SystemExit):
            self.seal()

    def test_binary_not_matching_the_identity_is_refused(self):
        self.write_identity("f" * 64)
        with self.assertRaises(SystemExit):
            self.seal()

    def test_the_real_bound_template_seals_with_its_raw_quotes_and_host_echoes_verbatim(self):
        # the shipped final template (real raw quotes, N9 canonical, 55 GiB) with only the unit-test runtime identity swapped in
        led = lt.bound_template_ledger()
        self.write("ledger-hold1.json", led)
        m = self.seal()
        ml = m["memory_ledger"]
        self.assertEqual(ml["admission_quotes"], led["admission_quotes"])
        self.assertEqual(ml["admission_quotes"]["canonical_observation"], "post_initialization_actual")
        self.assertEqual({ml["profiles"][n]["resident_expert_layers"] for n in lp.PROFILES}, {9})
        self.assertEqual((ml["owned_host_bytes"], ml["reservation_owned_host_bytes"]), (56416671744, 56416671744))
        self.assertEqual(ml["host_unreclaimable_gib"], 58564155392 / 2**30)
        self.assertEqual(ml["device_projection"]["memory_basis"], "post_initialization_actual")
        self.assertEqual(ml["device_owned_bytes"], 22716075372)


class BetaAdapter(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fx = os.path.join(tmp.name, "fx")
        text = os.path.join(tmp.name, "natural.txt")
        with open(text, "w") as f:
            f.write("natural text sentence. " * 200)
        with contextlib.redirect_stdout(io.StringIO()):
            make_fixtures.main(["make_fixtures.py", self.fx, text])
        self.canon = os.path.join(self.fx, "prompts.short.jsonl")
        self.beta = os.path.join(self.fx, "prompts.short.beta.jsonl")

    def records(self, path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_the_beta_input_is_the_canonical_input_minus_only_resume_at(self):
        c, b = self.records(self.canon), self.records(self.beta)
        self.assertEqual(len(c), 7)
        self.assertEqual([r["id"] for r in c], [r["id"] for r in b])
        stripped = [r["id"] for r in c if "resume_at" in r]
        self.assertEqual(stripped, ["resume"])
        self.assertEqual([r for r in c if r["id"] == "resume"][0]["resume_at"], 1537)
        for cr, br in zip(c, b):
            self.assertEqual(br, {k: v for k, v in cr.items() if k != "resume_at"})
            self.assertEqual(list(br), [k for k in cr if k != "resume_at"])
            self.assertTrue(set(br) <= set(beta_input.ALLOWED_BETA_KEYS))
        proof = json.load(open(os.path.join(self.fx, "beta-input-proof.json")))
        self.assertEqual(proof, beta_input.verify(self.canon, self.beta))
        by_id = {r["id"]: r for r in proof["records"]}
        self.assertEqual(by_id["resume"]["stripped"], {"resume_at": 1537})
        self.assertEqual(by_id["resume"]["prompt_ids"]["count"], 3400)
        self.assertEqual(by_id["p8192"]["stripped"], {})
        self.assertNotIn("prompt_ids", by_id["natural"])  # no token array is invented for the natural record
        self.assertIn("tokenizer", by_id["natural"]["token_ids"])
        self.assertEqual(proof["canonical"]["sha256"], hashlib.sha256(open(self.canon, "rb").read()).hexdigest())
        self.assertEqual(proof["beta"]["sha256"], hashlib.sha256(open(self.beta, "rb").read()).hexdigest())
        md5 = open(os.path.join(self.fx, "fixtures.md5")).read()
        self.assertIn("prompts.short.beta.jsonl", md5)
        self.assertIn("beta-input-proof.json", md5)

    def rewrite_beta(self, fn):
        recs = self.records(self.beta)
        fn(recs)
        with open(self.beta, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def test_any_other_difference_is_refused(self):
        cases = {
            "dropped record": lambda r: r.pop(),
            "reordered": lambda r: r.reverse(),
            "changed id": lambda r: r[0].__setitem__("id", "x"),
            "changed token": lambda r: r[0]["prompt_ids"].__setitem__(0, r[0]["prompt_ids"][0] + 1),
            "truncated ids": lambda r: r[1].__setitem__("prompt_ids", r[1]["prompt_ids"][:-1]),
            "changed message": lambda r: r[-1]["messages"][0].__setitem__("content", "other"),
            "extra key": lambda r: r[0].__setitem__("category", "x"),
            "resume_at kept": lambda r: [x.__setitem__("resume_at", 1537) for x in r if x["id"] == "resume"],
            "key order": lambda r: r.__setitem__(0, {"prompt_ids": r[0]["prompt_ids"], "id": r[0]["id"]}),
        }
        for label, fn in cases.items():
            with self.subTest(label):
                beta_input.write_beta(self.canon, self.beta)
                self.rewrite_beta(fn)
                with self.assertRaises(SystemExit):
                    beta_input.verify(self.canon, self.beta)

    def test_a_canonical_key_the_adapter_would_silently_drop_is_refused(self):
        recs = self.records(self.canon)
        recs[0]["temperature"] = 0
        with open(self.canon, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with self.assertRaises(SystemExit) as cm:
            beta_input.write_beta(self.canon, self.beta)
            beta_input.verify(self.canon, self.beta)
        self.assertIn("temperature", str(cm.exception))


class Evidence(unittest.TestCase):
    PROGRESS = "hipfire.qwen4.slowlink_gate.progress.v1"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.cell = os.path.join(self.tmp, "state")
        os.makedirs(os.path.join(self.cell, "home"))

    def put(self, name, data):
        mode = "wb" if isinstance(data, bytes) else "w"
        with open(os.path.join(self.cell, name), mode) as f:
            f.write(data)

    def record(self, seq, event, run="r1", **fields):
        return json.dumps({"schema": self.PROGRESS, "run_id": run, "seq": seq, "event": event, "utc": "2026-10-18T00:00:00.000Z",
                           "unix_ms": 1, "mono_ns": seq, **fields})

    def raw_snapshot(self):
        # evidence.json and load_phase.json are the collector's derived outputs (rewritten on every run); every other file is raw and must not change
        return {n: open(os.path.join(self.cell, n), "rb").read() for n in sorted(os.listdir(self.cell))
                if os.path.isfile(os.path.join(self.cell, n)) and not n.startswith(("evidence.json", "load_phase.json"))}

    def test_timeout_with_a_torn_and_corrupt_sidecar_keeps_every_raw_file_and_never_passes(self):
        self.put("stdout.txt", "out")
        self.put("stderr.txt", "err")
        self.put("rc.txt", "124\n")
        self.put("command.txt", "cmd")
        self.put("mem.csv", "t,v\n1,2\n")
        self.put("vram.csv", "t_mono_ns,used,total\n")
        self.put("gate.json.tmp-123", "{partial")
        lines = [self.record(0, "run_start"), self.record(1, "plan", required=["a", "b"]),
                 self.record(2, "memory", phase="pre_load"), "{not json", self.record(3, "final", status="pass", report_sha256="0" * 64)]
        self.put("gate.json.progress.jsonl", "\n".join(lines) + "\n" + self.record(4, "repeat_start", case="x", repeat=0)[:20])
        before = self.raw_snapshot()
        ev = ce.collect(self.cell, "state", "124", "true")
        self.assertEqual(self.raw_snapshot(), before)  # nothing was changed, removed or rewritten
        self.assertTrue(ev["timed_out"])
        self.assertEqual(ev["rc"], 124)
        self.assertFalse(ev["gate_report"]["present"])
        self.assertFalse(ev["gate_report"]["completed"])
        side = ev["progress_sidecar"]
        self.assertTrue(side["present"] and side["readable"])
        self.assertEqual((side["torn_tail"], side["corrupt_lines"], side["records_valid"]), (True, [4], 4))
        self.assertEqual(side["memory_phases"], ["pre_load"])
        self.assertEqual(side["status"], "incomplete")  # a recorded final/pass event is never a pass
        self.assertFalse(side["promotable"])
        self.assertTrue(side["forensic_only"])
        self.assertEqual(ev["partial_gate_files"], ["gate.json.tmp-123"])
        for name in ("stdout.txt", "stderr.txt", "rc.txt", "command.txt", "mem.csv", "vram.csv"):
            self.assertTrue(ev["raw_files"][name]["present"], name)
        self.assertFalse(ev["raw_files"]["vram-summary.json"]["present"])
        self.assertEqual(ev["scratch_dirs_not_hashed"], ["home"])
        self.assertEqual(json.load(open(os.path.join(self.cell, "evidence.json")))["schema"], ce.SCHEMA)

    def test_a_durably_recorded_failed_check_is_fail_never_pass(self):
        self.put("rc.txt", "1\n")
        self.put("gate.json.progress.jsonl", "\n".join([self.record(0, "run_start"),
                                                         self.record(1, "check", name="x.y", status="fail")]) + "\n")
        side = ce.collect(self.cell, "state", "1", "true")["progress_sidecar"]
        self.assertEqual((side["status"], side["failed_checks"]), ("fail", ["x.y"]))

    def test_load_failure_error_text_and_missing_sidecar_are_reported(self):
        self.put("rc.txt", "1\n")
        self.put("gate.json.progress.jsonl", "\n".join([self.record(0, "run_start"),
                                                         self.record(1, "load_error", error="hipMalloc failed")]) + "\n")
        side = ce.collect(self.cell, "prim", "1", "true")["progress_sidecar"]
        self.assertEqual(side["error"], "hipMalloc failed")
        self.assertEqual(side["last_event"], "load_error")
        os.remove(os.path.join(self.cell, "gate.json.progress.jsonl"))
        ev = ce.collect(self.cell, "prim", "1", "true")
        self.assertFalse(ev["progress_sidecar"]["present"])
        self.assertFalse(ev["progress_sidecar"]["readable"])

    def test_completed_report_is_recognised_and_other_runs_in_the_sidecar_are_ignored(self):
        self.put("rc.txt", "0\n")
        self.put("gate.json", json.dumps({"schema": ce.REPORT_SCHEMA, "status": "pass", "progress": {"records_before_final": 2}}))
        self.put("gate.json.progress.jsonl", "\n".join([self.record(0, "run_start", run="old"), self.record(0, "run_start", run="new"),
                                                         self.record(1, "plan", run="new")]) + "\n")
        ev = ce.collect(self.cell, "state", "0", "true")
        self.assertTrue(ev["gate_report"]["completed"])
        self.assertEqual((ev["progress_sidecar"]["run_id"], ev["progress_sidecar"]["records_in_run"]), ("new", 2))
        self.assertFalse(ev["timed_out"])
        self.put("gate.json", json.dumps({"schema": ce.REPORT_SCHEMA, "status": "incomplete", "error": "aborted"}))
        self.assertFalse(ce.collect(self.cell, "state", "1", "true")["gate_report"]["completed"])

    def test_summary_status_never_reads_the_evidence(self):
        out = os.path.join(self.tmp, "run")
        os.makedirs(os.path.join(out, "state"))
        with open(os.path.join(out, "state", "evidence.json"), "w") as f:
            json.dump({"progress_sidecar": {"status": "incomplete"}}, f)
        with open(os.path.join(out, "cells.jsonl"), "w") as f:
            f.write(json.dumps({"name": "state", "status": "pass", "why": "", "rc": 0, "seconds": 1, "leftover": ""}) + "\n")
        script = f'source "{LIB}"; CELLS_JSON="{out}/cells.jsonl"; OUT="{out}"; HOLD=hold1; REQUIRED="state"; finish_summary'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**os.environ, "S1_STAGE": self.tmp}, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary = json.load(open(os.path.join(out, "hold-summary.json")))
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["evidence"], {"state": "state/evidence.json"})


if __name__ == "__main__":
    unittest.main()
