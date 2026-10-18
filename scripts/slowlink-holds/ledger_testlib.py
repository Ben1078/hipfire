# SPDX-License-Identifier: Apache-2.0
"""Test support for the per-purpose ledger tests (CPU only, synthetic data). Imported by test_target_inputs.py and test_ledger_profiles.py."""
import copy
import os

import ledger_profiles as lp

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "ledger-template.json")
TIP = "a" * 40
BASE = "b" * 40
TREE = "c" * 40


def template():
    """The shipped ledger-template.json: the FINAL, source-filled ledger (every tile-census term from the CPU getter emit, exact extras terms,
    source ids), parsed with the duplicate-key guard."""
    return lp.load_ledger(TEMPLATE)


def nonfinal_template():
    """An explicit NON-FINAL copy of the shipped template, for the non-final refusal / under-fill / print-memory tests: every tile-census term
    except the cold F16 weight shadow is removed (so the terms sum to LESS than the total, the real getter remainder is absent, not invented),
    the runtime source identity is withdrawn, and a truthful pending_final names what is missing."""
    led = copy.deepcopy(template())
    for name in lp.PROFILES:
        p = led["profiles"][name]
        p["s1_tile_census_terms"] = [t for t in p["s1_tile_census_terms"] if t["name"] == lp.COLD_F16_TERM]
    for key in lp.PROVENANCE_RUNTIME:
        led["provenance"].pop(key, None)
    led["census_final"] = False
    led["pending_final"] = [f"profiles.{name}.s1_tile_census_terms: only {lp.COLD_F16_TERM} is carried; the remaining terms are not filled"
                            for name in lp.PROFILES] + ["provenance.census_source_commit / census_source_tree / census_source_base: not supplied"]
    return led


def tile_remainder(name):
    """Census bytes of one purpose that the non-final copy leaves unfilled (total minus the cold F16 term)."""
    return template()["profiles"][name]["s1_tile_census_bytes"] - lp.COLD_F16_BYTES


def final_ledger():
    """A FINAL ledger for tests: the exact real terms and figures of the shipped template with only the runtime identity replaced by a
    synthetic one (the unit-test stage identity TIP/TREE/BASE, never a real build)."""
    led = copy.deepcopy(template())
    led["provenance"].update({"census_source_commit": TIP, "census_source_tree": TREE, "census_source_base": BASE})
    return led
