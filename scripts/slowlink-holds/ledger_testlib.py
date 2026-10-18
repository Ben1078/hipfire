# SPDX-License-Identifier: Apache-2.0
"""Test support for the per-purpose ledger tests (CPU only). Imported by test_target_inputs.py, test_ledger_profiles.py, test_admission_adaptive.py
and test_hold1_split.py.

R5: the shipped ledger-template.json is now the FINAL production ledger Main bound: canonical observation post_initialization_actual, actual N9 at
the j1811 free 25,560,088,576, four raw SLOWLINK_ADMISSION_QUOTE_R5 CPU quotes with host echoes, runtime 40a4e3bee. It is the ACTUAL BASE the
fixtures retarget from (`retarget` moves every resident-count-dependent figure from the template's own count to `n`; nothing assumes ten):

  shipped_template()      the template exactly as shipped (real quotes, real runtime identity)
  bound_template_ledger() the template with only the unit-test runtime identity (TIP/TREE/BASE), for the seal tests; real quotes kept
  final_ledger(n, canon)  SYNTHETIC quote variants of the template retargeted to `n` (legacy default: the N10 no-loss row canonical)
  production_ledger()     the same with canonical post_initialization_actual at the template's actual N9 (synthetic round-number witnesses)
  draft_ledger()          an explicit NON-FINAL draft (no quotes, no runtime identity, pending_final names them) for the non-final cases
  nonfinal_template()     the template with its tile terms withdrawn and pending_final naming them

Synthetic witnesses are round-number test data (see `witness`), never the real model's quote."""
import copy
import os

import ledger_profiles as lp

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "ledger-template.json")
TIP = "a" * 40
BASE = "b" * 40
TREE = "c" * 40

# Synthetic accounting (test data, not the real model's quote). The two free samples are the j1811 progress samples (before / after the host
# link and runtime initialization); the other scalars are round numbers chosen so that ten resident layers is the maximal count at the
# reference sample and nine at the post-initialization sample.
PRE_INIT = 25_713_180_672
POST_INIT = 25_560_088_576
INIT_LOSS = PRE_INIT - POST_INIT  # 153,092,096
NON_EXPERT = 8_589_934_592
LAYER = 1_342_177_280
RESERVE = 3_598_381_184
TRUNK = 48
REFERENCE_SLACK = 103_092_096  # < LAYER: one more layer does not fit at the reference sample


def needed(n):
    return NON_EXPERT + n * LAYER + RESERVE


def witness(free, before, n, trunk=TRUNK):
    """The 14 receipt witness scalars of a quote at `n` resident layers and `free` fresh free bytes (before: the earlier sample or None)."""
    need, need10 = needed(n), needed(10)
    return {
        "free_before_initialization_bytes": before,
        "free_at_admission_bytes": free,
        "observed_initialization_bytes": None if before is None else max(before - free, 0),
        "non_expert_bytes": NON_EXPERT,
        "resident_layer_bytes": LAYER,
        "joint_reserve_bytes": RESERVE,
        "trunk_layers": trunk,
        "resident_layers": n,
        "preferred_resident_layers": 10,
        "needed_bytes": need,
        "slack_bytes": free - need,
        "preferred_deficit_bytes": max(need10 - free, 0),
        "memory_basis": "post_initialization_actual",
        "initialization_charged_via_free": True,
    }

POST_SLACK = LAYER - 50_000_000  # 1,292,177,280: at the post-initialization sample ref_n fits and one more layer is 50,000,000 B short

# Host-owner fixture data (SYNTHETIC test arithmetic of the R4 N=10 template, not a validator rule): 49 weight layers (48 trunk + 1 MTP), the
# per-layer expert payload and the per-layer 2 MiB host-map guard that reproduce the template's 39 x payload 52,140,441,600 and 39 x pad 81,788,928.
WEIGHT_LAYERS = 49
EXPERT_PAYLOAD = 1_336_934_400
HOST_PAD_PER_LAYER = 2_097_152
LABEL_OF = {lp.OBS_POST: "actual_post_initialization", lp.OBS_REFERENCE: "reference_pre_equals_post"}


def host_block(n):
    """The raw quote's host_owners.selected|preferred block at `n` resident layers (host_owner_entries is emit data: two per layer here)."""
    hl = WEIGHT_LAYERS - n
    payload, pad = hl * EXPERT_PAYLOAD, hl * HOST_PAD_PER_LAYER
    return {"resident_layers": n, "host_layers": hl, "host_owner_entries": 2 * hl, "host_payload_bytes": payload,
            "host_owner_pad_bytes": pad, "host_total_bytes": payload + pad}


def host_owners(n):
    return {"weight_layers_with_experts": WEIGHT_LAYERS, "selected": host_block(n), "preferred": host_block(10)}


def quote_entries(ref_n=10, canonical=None):
    """Four synthetic entries (both purposes x both memory observations).

    canonical == OBS_REFERENCE (the legacy unit-fixture choice): the no-loss reference row selects `ref_n` layers and the post-initialization row
    (the j1811 interval lost) one fewer.
    canonical == OBS_POST (the PRODUCTION choice): the post-initialization row selects `ref_n` layers at free needed(ref_n) + POST_SLACK and the
    no-loss row is the COUNTERFACTUAL that selects ref_n + 1 at free + INIT_LOSS (N9 / N10 at the j1811 samples)."""
    if canonical == lp.OBS_POST:
        post_free = needed(ref_n) + POST_SLACK
        cf_free = post_free + INIT_LOSS
        post = witness(post_free, cf_free, ref_n)
        cf = witness(cf_free, cf_free, ref_n + 1)
    else:
        free_ref = needed(ref_n) + REFERENCE_SLACK
        post = witness(free_ref - INIT_LOSS, free_ref, ref_n - 1) if ref_n >= 1 else witness(free_ref, free_ref + INIT_LOSS, 0)
        cf = witness(free_ref, free_ref, ref_n)
    entries = []
    for purpose, tile in (("product", 768), ("primitive_selftest", 512)):
        for obs, w in ((lp.OBS_POST, post), (lp.OBS_REFERENCE, cf)):
            entries.append({"purpose": purpose, "memory_observation": obs, "tile_rows": tile, "source_observation_label": LABEL_OF[obs],
                            "witness": copy.deepcopy(w), "host_owners": host_owners(w["resident_layers"])})
    return entries


def shipped_template():
    """The shipped ledger-template.json exactly as it is: the FINAL R5 production ledger (duplicate-key guarded)."""
    return lp.load_ledger(TEMPLATE)


def bound_template_ledger():
    """The shipped final template with ONLY the unit-test runtime identity swapped in (the seal tests stage BUILD-IDENTITY with TIP/TREE/BASE).
    The real raw quotes, host echoes and canonical binding are kept untouched."""
    led = copy.deepcopy(shipped_template())
    led["provenance"].update({"census_source_commit": TIP, "census_source_tree": TREE, "census_source_base": BASE})
    led["admission_quotes"]["source_commit"] = TIP
    return led


def draft_ledger():
    """An explicit NON-FINAL draft derived from the template: quotes, canonical binding and runtime identity withdrawn, pending_final names them."""
    led = copy.deepcopy(shipped_template())
    led["census_final"] = False
    led["pending_final"] = ["admission_quotes: the fresh SLOWLINK_ADMISSION_QUOTE_R5 entries are not bound yet",
                            "provenance.census_source_commit / census_source_tree / census_source_base: the runtime source is not frozen yet"]
    led["admission_quotes"].update({"canonical_observation": None, "source_commit": None, "entries": []})
    for key in lp.PROVENANCE_RUNTIME:
        led["provenance"].pop(key, None)
    return led


def retarget(led, n):
    """Move every resident-count-dependent REFERENCE figure of a ledger from 10 to `n` layers (weights, core, owned, root peak, slack)."""
    layer = led["device_weights"]["expert_layer_bytes"]
    delta = (n - led["fixture"]["resident_expert_layers"]) * layer
    led["fixture"]["resident_expert_layers"] = n
    led["device_weights"]["resident_expert_layers"] = n
    led["device_weights"]["resident_expert_bytes"] = n * layer
    meta = led["device_weights"].get("model_payload_metadata")
    if isinstance(meta, dict):  # the logical-payload record follows the count too
        meta["resident_expert_payload_bytes"] = n * meta["expert_layer_payload_bytes"]
        meta["total_payload_bytes"] = meta["non_expert_payload_bytes"] + meta["resident_expert_payload_bytes"]
    for name in lp.PROFILES:
        p = led["profiles"][name]
        p["resident_expert_layers"] = n
        for key in ("device_weights_bytes", "device_core_bytes", "device_owned_bytes"):
            p[key] += delta
        led["device_projection"]["by_purpose"][name]["slack_over_safety_bytes"] -= delta
    led["device_owned_bytes"] = max(led["profiles"][name]["device_owned_bytes"] for name in lp.PROFILES)
    retarget_host(led, n)
    return led


def retarget_host(led, n):
    """Move the host census to `n` resident layers honestly: pinned expert owners and the host-map pad follow the host layer count, and every
    derived host figure (purpose totals, peak, the sequential reservation max, allowance total, whole-GiB fabric host, slack) is recomputed."""
    block = host_block(n)
    for name in lp.PROFILES:
        node = led["host_census"][name]
        for t in node["terms"]:
            if t["name"] == lp.HOST_PAYLOAD_TERM:
                t["bytes"] = block["host_payload_bytes"]
            elif t["name"] == lp.HOST_PAD_TERM:
                t["bytes"] = block["host_owner_pad_bytes"]
        node["total_bytes"] = sum(t["bytes"] for t in node["terms"])
    peak = max(led["host_census"][name]["total_bytes"] for name in lp.PROFILES)
    led["host_owned_bytes"] = peak
    led["host_reservation_owned_bytes"] = max(led["host_reservation"]["total_bytes"], peak)
    total = led["host_reservation_owned_bytes"] + led["host_allowance_bytes"]
    led["host_total_with_allowance_bytes"] = total
    gib = -(-total // lp.GIB)
    led["fabric_hostmem_gib"] = gib
    led["fabric_hostmem_bytes"] = gib * lp.GIB
    led["host_slack_bytes"] = led["fabric_hostmem_bytes"] - total


def final_ledger(ref_n=10, canonical=lp.OBS_REFERENCE):
    """A FINAL SYNTHETIC ledger for tests: the shipped reference figures (moved to `ref_n` reference layers), the unit-test runtime identity and
    four synthetic admission-quote entries. Never a real emit or a real build. The legacy unit fixture EXPLICITLY chooses its canonical
    observation (default: the no-loss reference row at N10, so the R4 figures its tests assert stay exact); `production_ledger()` is the
    production binding (post_initialization_actual)."""
    led = copy.deepcopy(shipped_template())  # the actual base count (9) is read from the template: ref_n only retargets away from it
    if ref_n != led["fixture"]["resident_expert_layers"]:
        retarget(led, ref_n)
    led["census_final"] = True
    led["pending_final"] = []
    led["provenance"].update({"census_source_commit": TIP, "census_source_tree": TREE, "census_source_base": BASE})
    entries = quote_entries(ref_n, canonical)
    led["admission_quotes"] = {"schema": lp.QUOTE_SCHEMA, "canonical_observation": canonical, "source_commit": TIP, "entries": entries}
    # the device projection is made at the canonical quote's own free memory and carries that observation's basis
    free = next(e["witness"]["free_at_admission_bytes"] for e in entries if e["memory_observation"] == canonical)
    proj = led["device_projection"]
    proj["preload_free_bytes"] = free
    proj["memory_basis"] = lp.CANON_BASIS[canonical]
    for name in lp.PROFILES:
        p = led["profiles"][name]
        proj["by_purpose"][name]["slack_over_safety_bytes"] = free - p["device_owned_bytes"] - p["runtime_budget_bytes"] - p["safety_margin_bytes"]
    return led


def production_ledger(ref_n=9):
    """The PRODUCTION-bound synthetic ledger: canonical_observation post_initialization_actual with the actual N9 reference (weights, owned bytes,
    projection free, count derived from the post-initialization quote) and the N10 no-loss row only as the counterfactual."""
    return final_ledger(ref_n, lp.OBS_POST)


def template():
    """The FINAL synthetic ledger (alias kept for the tests written against the R4 final template)."""
    return final_ledger()


def nonfinal_template():
    """An explicit NON-FINAL copy of the shipped template, for the non-final refusal / under-fill / print-memory tests: every tile-census term
    except the cold F16 weight shadow is removed (so the terms sum to LESS than the total, the real getter remainder is absent, not invented) and
    pending_final names exactly that."""
    led = copy.deepcopy(shipped_template())
    for name in lp.PROFILES:
        p = led["profiles"][name]
        p["s1_tile_census_terms"] = [t for t in p["s1_tile_census_terms"] if t["name"] == lp.COLD_F16_TERM]
    led["census_final"] = False
    led["pending_final"] = [
        f"profiles.{name}.s1_tile_census_terms: only {lp.COLD_F16_TERM} is carried; the remaining terms are not filled" for name in lp.PROFILES]
    return led


def tile_remainder(name):
    """Census bytes of one purpose that the non-final copy leaves unfilled (total minus the cold F16 term)."""
    return shipped_template()["profiles"][name]["s1_tile_census_bytes"] - lp.COLD_F16_BYTES
