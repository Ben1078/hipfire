#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ledger_profiles.py print-memory LEDGER | check [--nonfinal | --production] LEDGER

The ONE validator of the declared per-purpose memory ledger (ledger-holdN.json, template ledger-template.json). It is imported by
make_manifest.py (which seals the profiles into manifest.memory_ledger) and called by hold-lib.sh (`--print-memory`), so there is no second
control plane for the arithmetic. CPU integer arithmetic of declared inputs only; nothing here is a GPU measurement or a gate.

The ledger carries TWO full source profiles, one per admitted purpose, under `profiles` (no flat profile fields at the root, no aliases):

  product             tile 768, no diagnostic reserve   (every product cell: state, headroom, greedy, perf)
  primitive_selftest  tile 512, diagnostic reserve > 0  (the primitive cell: kernel correctness coverage only)

Each profile has EXACTLY these keys: purpose, tile_rows, projection_role, resident_expert_layers, window_rows, page_count, device_weights_bytes,
s1_tile_census_bytes, s1_tile_census_terms, f16_shadow_cold_bound_bytes, s1_extras_bytes, s1_extras_terms, device_core_bytes,
sequential_diag_bytes, device_owned_bytes, runtime_budget_bytes, safety_margin_bytes. The two `*_terms` are ARRAYS of
{"name": non-empty string, "bytes": u64} with unique names whose checked sum is the total. The cold BF16->F16 weight shadow
(f16_weight_shadow_bytes, 1,298,923,520 B) is exactly ONE tile-census term, appears in no extras term and no other term name mentions
f16; f16_shadow_cold_bound_bytes states it and must equal that one term. device_core = weights + tile census + extras (the diagnostic is NOT
in core); device_owned = core + sequential diagnostic; the root device_owned_bytes is the MAX of the two purposes' owned bytes (they load
sequentially, so the physical peak is NOT product core + the primitive diagnostic). R4 data: every device figure is the PHYSICAL backing of the
canonical arena plan (a model over the queried granule/observed capacity, NOT a per-allocation physical or recovery measurement): the cold F16 term
stays the 1,298,923,520 B PAYLOAD bound (exactly once, validated below, no second F16 term), device_weights is the physical backing (the model
payload is recorded separately in device_weights.model_payload_metadata, not validated here) and every rounding is a named
allocation_backing_padding_bytes term of s1_tile_census_terms and of s1_extras_terms, so the sums stay exact. Each host census also carries the
2 GiB bounded_host_snapshot_cap_bytes term. No rule below changed. The root keeps only the shared physical card, the
separate fabric reservation, the host reservation (a sequential MAX, never an arm sum) and the provenance. Host census is per purpose
(`host_census`): the forward PLE stage and the two token Vecs follow each purpose's own tile.

R5 ADAPTIVE ADMISSION. The runtime keeps the largest resident layer count that actually fits (ten preferred, never a floor), so every figure
that depends on the resident count here is REFERENCE data (`projection_role` "reference"): `resident_expert_layers` is any count in
0..47 (not a fixed ten), shared by the fixture, `device_weights` and both profiles, and `device_projection` is the reference projection at its
own free (the CANONICAL observation's: see below). A FINAL ledger additionally carries `admission_quotes`: the fresh SLOWLINK_ADMISSION_QUOTE_R5
emit's scalar witnesses (the receipt witness keys) for BOTH purposes at BOTH memory observations (`post_initialization_actual`: the measured
before/after interval; `reference_pre_initialization`: before == after), each re-verified here by admission_witness.py (selected fits, one more does
not, a host layer remains) with the purpose's non-expert/layer/reserve scalars identical across the two observations and the post-initialization
count not above the no-loss count. `admission_quotes.canonical_observation` names the ONE observation the reference figures (profile counts,
weights, owned bytes, device projection free, host census) derive from; the PRODUCTION ledger binds `post_initialization_actual` (e.g. N9 at the
j1811 free 25,560,088,576) and the production seal (make_manifest, `check --production`) refuses anything else. The other row (e.g. the N10
no-initialization-loss sample) is a COUNTERFACTUAL kept as information only, never swapped in or relabelled. Main binds those entries and the runtime source
commit (admission_quotes.source_commit == provenance.census_source_commit) from the fresh emit; this file invents neither.

A ledger is FINAL (census_final true, pending_final empty, every sum exact, provenance bound; the shipped ledger-template.json is a NON-FINAL draft
until Main binds the R5 emit) or a NON-FINAL draft (census_final false, a non-empty pending_final naming exactly what is missing; a
profile's tile terms may then sum to less than their total, and `admission_quotes.entries` may be empty only when pending_final names admission_quotes; the tests build such a copy explicitly). Nothing that gates a hold or a seal
accepts a non-final ledger. Exit status: a refusal exits non-zero with the reason on stderr."""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import admission_witness as aw  # noqa: E402

GIB = 1 << 30
U64 = (1 << 64) - 1

PRODUCT = "product"
PRIMITIVE = "primitive_selftest"
PROFILES = (PRODUCT, PRIMITIVE)
PROFILE_TILE = {PRODUCT: 768, PRIMITIVE: 512}
GENERIC_MAX_TILE = 1536  # the historical generic upper bound only; never a profile's expected tile
TRUNK_LAYERS = 48  # model data: a reference resident count must leave at least one host layer
REFERENCE_ROLE = "reference"
QUOTE_SCHEMA = "SLOWLINK_ADMISSION_QUOTE_R5"
OBS_POST, OBS_REFERENCE = "post_initialization_actual", "reference_pre_initialization"
OBSERVATIONS = (OBS_POST, OBS_REFERENCE)
WINDOW_ROWS = 16384
PAGE_COUNT = 18
COLD_F16_TERM = "f16_weight_shadow_bytes"
COLD_F16_BYTES = 1298923520
RUNTIME_BUDGET_BYTES = 1 << 30
SAFETY_MARGIN_BYTES = 512 << 20
PLE_ROW_BYTES = 16 * 160 * 2  # forward PLE stage bytes per tile row (2560 x 2)
TOKEN_ROW_BYTES = 4
BETA_CHUNK_ROWS = 8192  # the immutable beta arm's authoritative requested chunk ceiling

PROFILE_KEYS = ("purpose", "tile_rows", "projection_role", "resident_expert_layers", "window_rows", "page_count", "device_weights_bytes",
                "s1_tile_census_bytes", "s1_tile_census_terms", "f16_shadow_cold_bound_bytes", "s1_extras_bytes", "s1_extras_terms",
                "device_core_bytes", "sequential_diag_bytes", "device_owned_bytes", "runtime_budget_bytes", "safety_margin_bytes")

ROOT_KEYS = ("hold", "census_final", "pending_final", "census_role", "source_emit", "route", "fixture", "device_weights", "profiles", "admission_quotes",
             "device_owned_bytes", "device_owned_note", "device_projection", "card_bytes", "physical_card_gib",
             "fabric_reservation_device_gib", "card_bytes_note", "session_cache_budget_cap_bytes_not_committed", "session_cache_note",
             "pending_working_set_ceiling_bytes_not_charged", "pending_ceiling_note", "host_census", "host_owned_bytes", "host_owned_note",
             "host_reservation_owned_bytes", "host_reservation", "host_reservation_note", "host_allowance_bytes",
             "host_total_with_allowance_bytes", "fabric_hostmem_gib", "fabric_hostmem_bytes", "host_slack_bytes",
             "allowances_are_declared_not_measured", "census_basis", "provenance", "provenance_note")

# Fields of the retired FLAT (single-profile) ledger. They are refused at the root, so a stale or aliased ledger can never be read as a profile.
FLAT_ROOT_KEYS = ("tile_rows", "resident_expert_layers", "device_weights_bytes", "device_weights_terms", "s1_tile_census_bytes",
                  "s1_tile_terms", "s1_tile_census_terms", "s1_extras_bytes", "s1_extras_terms", "device_core_bytes",
                  "sequential_diag_bytes", "runtime_overhead_budget_bytes", "runtime_budget_bytes", "safety_margin_bytes",
                  "host_owned_terms", "host_reservation_terms", "projection", "route_layout")

BETA_TERMS = ("beta_old_max_bytes", "beta_forward_ple_stage_bytes", "beta_forward_token_stage_bytes", "beta_forward_mesh_token_stage_bytes")
PROVENANCE_RUNTIME = ("census_source_commit", "census_source_tree", "census_source_base")


def bad(msg):
    sys.exit(f"ledger: {msg}")


def _uint(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= U64


def _pos(v):
    return _uint(v) and v > 0


def _gib(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _dup_guard(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            bad(f"JSON repeats the key {k!r} (a duplicated shadow field)")
        out[k] = v
    return out


def load_ledger(path):
    """The ledger JSON, refusing a missing file, invalid JSON, a non-object and any repeated object key."""
    try:
        with open(path) as f:
            led = json.load(f, object_pairs_hook=_dup_guard)
    except (OSError, ValueError) as e:
        bad(f"{path} unreadable or not JSON: {e}")
    if not isinstance(led, dict):
        bad(f"{path} is not a JSON object")
    return led


def terms_array(where, value):
    """{name: bytes} of an array of {"name","bytes"} terms; unique names, u64 bytes, no other fields."""
    if not isinstance(value, list) or not value:
        bad(f"{where} is not a non-empty array of {{name, bytes}} terms")
    seen = {}
    for i, t in enumerate(value):
        if not isinstance(t, dict) or set(t) != {"name", "bytes"}:
            bad(f"{where}[{i}] is not exactly {{\"name\", \"bytes\"}}")
        name, b = t["name"], t["bytes"]
        if not isinstance(name, str) or not name.strip():
            bad(f"{where}[{i}].name is not a non-empty string")
        if not _uint(b):
            bad(f"{where}[{i}].bytes {b!r} is not a u64 integer")
        if name in seen:
            bad(f"{where} repeats the term name {name!r} (a duplicated shadow term)")
        seen[name] = b
    return seen


def fabric_reservation_gib(led):
    """The ledger's explicit fabric exclusive-use device reservation in GiB (a finite number > 0); anything else is an error."""
    v = led.get("fabric_reservation_device_gib")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        sys.exit(f"declared ledger needs fabric_reservation_device_gib as a finite number > 0 (got {v!r})")
    return v


def _pending_covers(pending, prefix):
    return any(isinstance(p, str) and p.startswith(prefix) for p in pending)


def check_profile(name, p, final, pending):
    """One purpose profile; returns its owned device bytes. Non-final ledgers may under-fill the tile terms only if pending_final says so."""
    where = f"profiles.{name}"
    if not isinstance(p, dict):
        bad(f"{where} is not an object")
    unexpected, missing = sorted(set(p) - set(PROFILE_KEYS)), sorted(set(PROFILE_KEYS) - set(p))
    if unexpected or missing:
        bad(f"{where} keys: unexpected {unexpected}, missing {missing} (exact per-purpose key set, no aliases)")
    if p["purpose"] != name:
        bad(f"{where}.purpose {p['purpose']!r} is not {name!r}")
    want_tile = PROFILE_TILE[name]
    if p["tile_rows"] != want_tile or isinstance(p["tile_rows"], bool):
        bad(f"{where}.tile_rows {p['tile_rows']!r} != {want_tile}, the tile of the {name} purpose (a profile of another tile cannot stand in for it)")
    for key, want in (("window_rows", WINDOW_ROWS), ("page_count", PAGE_COUNT),
                      ("runtime_budget_bytes", RUNTIME_BUDGET_BYTES), ("safety_margin_bytes", SAFETY_MARGIN_BYTES)):
        if p[key] != want or isinstance(p[key], bool):
            bad(f"{where}.{key} {p[key]!r} != {want}")
    if p["projection_role"] != REFERENCE_ROLE:
        bad(f"{where}.projection_role {p['projection_role']!r} is not {REFERENCE_ROLE!r}: the sealed census is a reference projection, not an observation or a target")
    if not _uint(p["resident_expert_layers"]) or p["resident_expert_layers"] >= TRUNK_LAYERS:
        bad(f"{where}.resident_expert_layers {p['resident_expert_layers']!r} is not a reference count in 0..{TRUNK_LAYERS - 1} "
            f"(zero is valid, ten is only preferred; a count leaving no host layer is not an S1 placement)")
    for key in ("device_weights_bytes", "s1_tile_census_bytes", "s1_extras_bytes", "device_core_bytes", "device_owned_bytes"):
        if not _pos(p[key]):
            bad(f"{where}.{key} {p[key]!r} is not a positive integer")
    diag = p["sequential_diag_bytes"]
    if not _uint(diag):
        bad(f"{where}.sequential_diag_bytes {diag!r} is not a u64 integer")
    if name == PRODUCT and diag != 0:
        bad(f"{where}.sequential_diag_bytes {diag}: the product purpose carries no diagnostic reserve")
    if name == PRIMITIVE and diag == 0:
        bad(f"{where}.sequential_diag_bytes is 0: the primitive self-test reserve is missing")
    tile_terms = terms_array(f"{where}.s1_tile_census_terms", p["s1_tile_census_terms"])
    extras_terms = terms_array(f"{where}.s1_extras_terms", p["s1_extras_terms"])
    if tile_terms.get(COLD_F16_TERM) != COLD_F16_BYTES:
        bad(f"{where}.s1_tile_census_terms needs exactly one {COLD_F16_TERM} term of {COLD_F16_BYTES} B (got {tile_terms.get(COLD_F16_TERM)!r})")
    for label, terms in (("s1_tile_census_terms", tile_terms), ("s1_extras_terms", extras_terms)):
        for n in terms:
            if "f16" in n.lower() and not (label == "s1_tile_census_terms" and n == COLD_F16_TERM):
                bad(f"{where}.{label} term {n!r} names F16: the cold F16 shadow is one tile-census term and is never charged twice")
    if p["f16_shadow_cold_bound_bytes"] != COLD_F16_BYTES or isinstance(p["f16_shadow_cold_bound_bytes"], bool):
        bad(f"{where}.f16_shadow_cold_bound_bytes {p['f16_shadow_cold_bound_bytes']!r} != {COLD_F16_BYTES}, the one {COLD_F16_TERM} term")
    tile_sum, extras_sum = sum(tile_terms.values()), sum(extras_terms.values())
    if extras_sum != p["s1_extras_bytes"]:
        bad(f"{where}.s1_extras_terms sum to {extras_sum}, not s1_extras_bytes {p['s1_extras_bytes']}")
    if tile_sum != p["s1_tile_census_bytes"]:
        if final or tile_sum > p["s1_tile_census_bytes"] or not _pending_covers(pending, f"{where}.s1_tile_census_terms"):
            bad(f"{where}.s1_tile_census_terms sum to {tile_sum}, not s1_tile_census_bytes {p['s1_tile_census_bytes']}"
                + ("" if final else " (a non-final ledger may under-fill them only when pending_final names that profile)"))
    core = p["device_weights_bytes"] + p["s1_tile_census_bytes"] + p["s1_extras_bytes"]
    if p["device_core_bytes"] != core:
        bad(f"{where}.device_core_bytes {p['device_core_bytes']} != weights + tile census + extras {core}")
    if p["device_owned_bytes"] != p["device_core_bytes"] + diag:
        bad(f"{where}.device_owned_bytes {p['device_owned_bytes']} != core + sequential diag {p['device_core_bytes'] + diag}")
    return p["device_owned_bytes"]


def check_admission_quotes(led, final, pending, profiles, canonical=None):
    """The SLOWLINK_ADMISSION_QUOTE_R5 scalar witnesses Main binds from the fresh emit, re-verified by admission_witness.py (no second policy: selected
    count fits, one more does not, a host layer remains). FINAL: exactly one entry per purpose x memory observation, `source_commit` equal to
    provenance.census_source_commit. A non-final ledger may carry none only when pending_final names admission_quotes.

    CANONICAL OBSERVATION. `admission_quotes.canonical_observation` names the ONE observation the sealed reference figures (profiles'
    resident_expert_layers, weights, owned bytes, the device projection and so the host census) are derived from: the PRODUCTION ledger binds
    `post_initialization_actual` (the actual fresh free after initialization, e.g. N9); the other observation (the earlier no-initialization-loss
    sample, e.g. N10) is a COUNTERFACTUAL kept as information only, never swapped in and never relabelled. The canonical witness count must equal
    every profile's reference count. `canonical` (the production seal passes `post_initialization_actual`) additionally REQUIRES that binding;
    unit fixtures may declare the other. Null is allowed only for a non-final draft with no entries. Returns {purpose: canonical witness} (empty
    when nothing is bound)."""
    q = led["admission_quotes"]
    if not isinstance(q, dict) or q.get("schema") != QUOTE_SCHEMA:
        bad(f"admission_quotes must be an object with schema {QUOTE_SCHEMA!r}")
    entries = q.get("entries")
    if not isinstance(entries, list):
        bad("admission_quotes.entries is not a list")
    if "canonical_observation" not in q:
        bad("admission_quotes.canonical_observation is missing (the observation the sealed reference figures derive from)")
    chosen = q["canonical_observation"]
    if chosen is not None and chosen not in OBSERVATIONS:
        bad(f"admission_quotes.canonical_observation {chosen!r} is not one of {list(OBSERVATIONS)} or null")
    if canonical is not None and chosen != canonical:
        bad(f"admission_quotes.canonical_observation {chosen!r} != the required {canonical!r}: the production ledger's reference figures must derive from the "
            f"actual post-initialization quote; the {OBS_REFERENCE} row is a counterfactual and cannot be bound as the reference")
    if not entries:
        if final:
            bad("final ledger has no admission_quotes.entries: Main binds the fresh SLOWLINK_ADMISSION_QUOTE_R5 emit for both purposes at both "
                "memory observations (nothing is invented or hand-computed here)")
        if not _pending_covers(pending, "admission_quotes"):
            bad("a non-final ledger without admission_quotes.entries must name admission_quotes in pending_final")
        if chosen is not None:
            bad(f"admission_quotes.canonical_observation {chosen!r} is bound but no quote entries back it")
        return {}, {}
    if chosen is None:
        bad("admission_quotes has entries but canonical_observation is null: name the observation the reference derives from")
    seen, rows = {}, {}
    for i, e in enumerate(entries):
        where = f"admission_quotes.entries[{i}]"
        if not isinstance(e, dict) or not {"purpose", "memory_observation", "tile_rows", "witness"} <= set(e):
            bad(f"{where} needs purpose, memory_observation, tile_rows and witness")
        purpose, obs = e["purpose"], e["memory_observation"]
        if purpose not in PROFILES or obs not in OBSERVATIONS:
            bad(f"{where} purpose {purpose!r} / memory_observation {obs!r} is not one of {sorted(PROFILES)} / {list(OBSERVATIONS)}")
        if e["tile_rows"] != PROFILE_TILE[purpose] or isinstance(e["tile_rows"], bool):
            bad(f"{where}.tile_rows {e['tile_rows']!r} != {PROFILE_TILE[purpose]}, the tile of the {purpose} purpose")
        if (purpose, obs) in seen:
            bad(f"{where} repeats {purpose}/{obs}")
        problems = aw.witness_problems(e["witness"], check_actual=False)
        if problems:
            bad(f"{where} ({purpose}/{obs}): " + "; ".join(problems))
        w = e["witness"]
        before, after = w["free_before_initialization_bytes"], w["free_at_admission_bytes"]
        if obs == OBS_REFERENCE and before != after:
            bad(f"{where} is the reference observation: free before initialization {before!r} must equal free at admission {after} (no initialization loss)")
        if obs == OBS_POST and before is None:
            bad(f"{where} is the post-initialization observation: the measured before sample is missing (a loss is never assumed)")
        label = e.get("source_observation_label")
        if label is not None and SOURCE_LABELS.get(label) != obs:
            bad(f"{where}.source_observation_label {label!r} maps to {SOURCE_LABELS.get(label)!r}, not the entry's memory_observation {obs!r} "
                f"(rows are never swapped; the mapping is {SOURCE_LABELS})")
        if canonical is not None and label is None:
            bad(f"{where} ({purpose}/{obs}): production sealing requires source_observation_label (the raw emit's original label, preserved)")
        if e.get("host_owners") is not None:
            _host_echo(f"admission_quotes.entries[{i}] ({purpose}/{obs}) host_owners", e["host_owners"], w)
        seen[(purpose, obs)] = w
        rows[(purpose, obs)] = e
    canon, canon_rows = {}, {}
    for purpose in PROFILES:
        post, ref = seen.get((purpose, OBS_POST)), seen.get((purpose, OBS_REFERENCE))
        if final and (post is None or ref is None):
            bad(f"final ledger admission_quotes lacks the {purpose} entry for "
                + ("both observations" if post is None and ref is None else (OBS_POST if post is None else OBS_REFERENCE)))
        if post is not None and ref is not None:
            for key in ("non_expert_bytes", "resident_layer_bytes", "joint_reserve_bytes", "trunk_layers"):
                if post[key] != ref[key]:
                    bad(f"admission_quotes {purpose}: {key} differs between the two memory observations ({post[key]} vs {ref[key]}): one quote, two free samples")
            if post["resident_layers"] > ref["resident_layers"]:
                bad(f"admission_quotes {purpose}: the post-initialization count {post['resident_layers']} exceeds the counterfactual no-loss count "
                    f"{ref['resident_layers']}")
        w = seen.get((purpose, chosen))
        if w is None:
            bad(f"admission_quotes {purpose}: no entry for the canonical observation {chosen!r}")
        if w["resident_layers"] != profiles[purpose]["resident_expert_layers"]:
            bad(f"admission_quotes {purpose}: the canonical {chosen} quote selects {w['resident_layers']} layers but the profile's reference count is "
                f"{profiles[purpose]['resident_expert_layers']} (the reference must derive from the canonical quote; the other observation is a "
                f"counterfactual)")
        canon[purpose] = w
        canon_rows[purpose] = rows[(purpose, chosen)]
    if final:
        prov = led["provenance"]
        if not isinstance(prov, dict) or q.get("source_commit") != prov.get("census_source_commit"):
            bad(f"admission_quotes.source_commit {q.get('source_commit')!r} != provenance.census_source_commit "
                f"{prov.get('census_source_commit') if isinstance(prov, dict) else prov!r}: the quotes must come from the runtime source the census describes")
    return canon, canon_rows



# Metadata-only mapping of the raw source emit's observation labels to the ledger's observations (the original label is preserved on the entry as
# `source_observation_label`; a row is never relabelled to the other observation).
SOURCE_LABELS = {"reference_pre_equals_post": OBS_REFERENCE, "actual_post_initialization": OBS_POST}
HOST_SELECTED_KEYS = ("resident_layers", "host_layers", "host_owner_entries", "host_payload_bytes", "host_owner_pad_bytes", "host_total_bytes")
HOST_PAYLOAD_TERM, HOST_PAD_TERM = "pinned_expert_owners_bytes", "host_map_owner_pad_bytes"


def _host_block(where, block, weight_layers):
    """One raw `host_owners.selected|preferred` block: exact u64 keys, closure on the weight layers, total == payload + pad."""
    if not isinstance(block, dict) or set(block) != set(HOST_SELECTED_KEYS) or not all(_uint(block[k]) for k in HOST_SELECTED_KEYS):
        bad(f"{where} must be an object of exactly the u64 keys {list(HOST_SELECTED_KEYS)}")
    if block["host_layers"] + block["resident_layers"] != weight_layers:
        bad(f"{where}: host_layers {block['host_layers']} + resident_layers {block['resident_layers']} != weight_layers_with_experts {weight_layers}")
    if block["host_total_bytes"] != block["host_payload_bytes"] + block["host_owner_pad_bytes"]:
        bad(f"{where}: host_total_bytes {block['host_total_bytes']} != payload {block['host_payload_bytes']} + pad {block['host_owner_pad_bytes']}")
    return block


def _host_echo(where, ho, w):
    """Validate one entry's raw `host_owners` echo against ITS OWN witness (every row, canonical or counterfactual): the raw object keys,
    weight_layers_with_experts >= the witness trunk, selected/preferred closure and total == payload + pad, preferred count == the witness preferred
    count, selected.resident_layers == the row's witness count. Returns the selected block."""
    if not isinstance(ho, dict) or set(ho) != {"weight_layers_with_experts", "selected", "preferred"}:
        bad(f"{where} must be the raw object {{weight_layers_with_experts, selected, preferred}}")
    wl = ho["weight_layers_with_experts"]
    if not _pos(wl) or wl < w["trunk_layers"]:
        bad(f"{where}.weight_layers_with_experts {wl!r} is not a positive count >= the witness trunk_layers {w['trunk_layers']}")
    sel = _host_block(f"{where}.selected", ho["selected"], wl)
    _host_block(f"{where}.preferred", ho["preferred"], wl)
    if ho["preferred"]["resident_layers"] != w["preferred_resident_layers"]:
        bad(f"{where}.preferred.resident_layers {ho['preferred']['resident_layers']} != the witness preferred count {w['preferred_resident_layers']}")
    if sel["resident_layers"] != w["resident_layers"]:
        bad(f"{where}.selected.resident_layers {sel['resident_layers']} != the witness count {w['resident_layers']}")
    return sel


def check_host_binding(led, canon, canon_rows, production):
    """The host census is bound to the SAME canonical quote: each purpose's canonical entry carries the raw quote's `host_owners` unchanged
    (validated per row by _host_echo); host_census[purpose] terms pinned_expert_owners_bytes / host_map_owner_pad_bytes equal the CANONICAL row's
    selected payload / pad EXACTLY (a stale host census of another resident count is refused). Nothing assumes entries per layer, a quant format
    or 49 layers: those are the emit's data. The counterfactual row is validated for self-consistency but never drives the host totals.
    PRODUCTION requires the echo; a legacy fixture without `host_owners` stays permissive."""
    for purpose in PROFILES:
        row = canon_rows.get(purpose)
        if row is None:
            continue
        ho = row.get("host_owners")
        if ho is None:
            if production:
                bad(f"admission_quotes {purpose}: the canonical entry carries no host_owners (production sealing binds the host census to the canonical quote)")
            continue
        sel = ho["selected"]
        terms = {t["name"]: t["bytes"] for t in led["host_census"][purpose]["terms"]}
        for term, want in ((HOST_PAYLOAD_TERM, sel["host_payload_bytes"]), (HOST_PAD_TERM, sel["host_owner_pad_bytes"])):
            if terms.get(term) != want:
                bad(f"host_census.{purpose}.terms {term} {terms.get(term)!r} != the canonical quote's selected {want} ({sel['resident_layers']} resident / "
                    f"{sel['host_layers']} host layers): a host census of another resident count is stale")


CANON_BASIS = {OBS_POST: "post_initialization_actual", OBS_REFERENCE: "reference_pre_initialization"}




def check_ledger(led, final=True, canonical=None):
    """Full validation; returns {"owned": {profile: bytes}, "peak_owned": int, "host_peak": int, "host_total": int, "host_gib": int}."""
    if not isinstance(led, dict):
        bad("not an object")
    flat = [k for k in FLAT_ROOT_KEYS if k in led]
    if flat:
        bad(f"flat (single-profile) fields {flat} at the root: the ledger carries only profiles.product and profiles.primitive_selftest")
    absent = [k for k in ROOT_KEYS if k not in led]
    if absent:
        bad(f"missing root keys {absent}")
    cf, pending = led["census_final"], led["pending_final"]
    if not isinstance(cf, bool):
        bad("census_final is not a boolean")
    if not isinstance(pending, list) or any(not isinstance(x, str) or not x.strip() for x in pending):
        bad("pending_final is not a list of non-empty strings")
    if cf and pending:
        bad(f"census_final is true but pending_final is not empty: {pending}")
    if not cf and not pending:
        bad("census_final is false but pending_final names nothing")
    if final and not cf:
        bad("ledger is not final (census_final false; pending_final: " + "; ".join(pending) + ")")
    if led["route"] != "aggregate":
        bad(f"route {led['route']!r} is not 'aggregate'")
    fx = led["fixture"]
    if not isinstance(fx, dict):
        bad("fixture is not an object")
    for key, want in (("window_rows", WINDOW_ROWS), ("page_count", PAGE_COUNT)):
        if fx.get(key) != want:
            bad(f"fixture.{key} {fx.get(key)!r} != {want}")
    ref = fx.get("resident_expert_layers")
    if not _uint(ref) or ref >= TRUNK_LAYERS:
        bad(f"fixture.resident_expert_layers {ref!r} is not a reference count in 0..{TRUNK_LAYERS - 1}")
    if fx.get("projection_role") != REFERENCE_ROLE:
        bad(f"fixture.projection_role {fx.get('projection_role')!r} is not {REFERENCE_ROLE!r}")
    for stale in ("tile_rows",):
        if stale in fx:
            bad(f"fixture.{stale} is a per-purpose profile field, not a shared fixture field")
    cold = fx.get("cold_f16")
    if not isinstance(cold, dict) or cold.get("bytes") != COLD_F16_BYTES or cold.get("tile_census_term") != COLD_F16_TERM \
            or cold.get("tile_census_term_count") != 1 or cold.get("extras_term_count") != 0:
        bad(f"fixture.cold_f16 must state bytes {COLD_F16_BYTES}, tile_census_term {COLD_F16_TERM!r}, tile_census_term_count 1 and extras_term_count 0")
    # shared weights
    w = led["device_weights"]
    if not isinstance(w, dict) or not all(_pos(w.get(k)) for k in ("non_expert_bytes", "expert_layer_bytes")) or not _uint(w.get("resident_expert_bytes")):
        bad("device_weights needs positive non_expert_bytes and expert_layer_bytes and an integer resident_expert_bytes")
    if w.get("projection_role") != REFERENCE_ROLE:
        bad(f"device_weights.projection_role {w.get('projection_role')!r} is not {REFERENCE_ROLE!r}")
    if w.get("resident_expert_layers") != ref or w["resident_expert_bytes"] != ref * w["expert_layer_bytes"]:
        bad(f"device_weights resident_expert_bytes {w['resident_expert_bytes']} != the reference {ref} x expert_layer_bytes {w['expert_layer_bytes']} "
            f"(resident_expert_layers {w.get('resident_expert_layers')!r} must equal fixture.resident_expert_layers)")
    weights = w["non_expert_bytes"] + w["resident_expert_bytes"]
    profiles = led["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILES):
        bad(f"profiles keys {sorted(profiles) if isinstance(profiles, dict) else profiles!r} != exactly {sorted(PROFILES)}")
    owned = {}
    for name in PROFILES:
        owned[name] = check_profile(name, profiles[name], final, pending)
        if profiles[name]["device_weights_bytes"] != weights:
            bad(f"profiles.{name}.device_weights_bytes {profiles[name]['device_weights_bytes']} != device_weights {weights}")
        if profiles[name]["resident_expert_layers"] != ref:
            bad(f"profiles.{name}.resident_expert_layers {profiles[name]['resident_expert_layers']!r} != fixture reference {ref}")
    peak = max(owned.values())
    if led["device_owned_bytes"] != peak or isinstance(led["device_owned_bytes"], bool):
        bad(f"root device_owned_bytes {led['device_owned_bytes']!r} != the MAX {peak} of the per-purpose owned bytes "
            f"(sequential purposes; never product core + the primitive diagnostic)")
    # shared physical card and the separate fabric reservation
    card = led["card_bytes"]
    if not _pos(card) or not _gib(led["physical_card_gib"]) or led["physical_card_gib"] * GIB != card:
        bad(f"card_bytes {card!r} is not physical_card_gib {led['physical_card_gib']!r} GiB")
    fab = fabric_reservation_gib(led)
    if fab * GIB > card:
        bad(f"fabric_reservation_device_gib {fab} exceeds the physical card {card} B")
    if peak > card:
        bad(f"sequential owned peak {peak} exceeds the physical card {card}")
    # projection: slack on top of runtime + safety, per purpose
    proj = led["device_projection"]
    pre = proj.get("preload_free_bytes") if isinstance(proj, dict) else None
    by = proj.get("by_purpose") if isinstance(proj, dict) else None
    if not _pos(pre) or not isinstance(by, dict) or set(by) != set(PROFILES):
        bad("device_projection needs preload_free_bytes and by_purpose for exactly both purposes")
    canon, canon_rows = check_admission_quotes(led, final, pending, profiles, canonical)
    chosen = led["admission_quotes"]["canonical_observation"]
    if proj.get("projection_role") != REFERENCE_ROLE:
        bad("device_projection must declare projection_role 'reference': the projection is reference data, never an observation")
    want_basis = CANON_BASIS.get(chosen)
    if (want_basis is not None and proj.get("memory_basis") != want_basis) or (want_basis is None and proj.get("memory_basis") not in CANON_BASIS.values()):
        bad(f"device_projection.memory_basis {proj.get('memory_basis')!r} != {want_basis!r}, the basis of the canonical observation {chosen!r}: the "
            f"projection is made at the canonical quote's own free memory (the other observation is a counterfactual)")
    for name, w in canon.items():
        if pre != w["free_at_admission_bytes"]:
            bad(f"device_projection.preload_free_bytes {pre} != the canonical {chosen} quote's free at admission {w['free_at_admission_bytes']} ({name})")
    for name in PROFILES:
        p = profiles[name]
        slack = pre - p["device_owned_bytes"] - p["runtime_budget_bytes"] - p["safety_margin_bytes"]
        row = by[name]
        if not isinstance(row, dict) or row.get("slack_over_safety_bytes") != slack or slack < 0:
            bad(f"device_projection.by_purpose.{name}.slack_over_safety_bytes "
                f"{row.get('slack_over_safety_bytes') if isinstance(row, dict) else row!r} "
                f"!= preload free - owned - runtime - safety = {slack} (must be >= 0)")
    # host census per purpose: the forward stages follow each purpose's own tile
    hc = led["host_census"]
    if not isinstance(hc, dict) or set(hc) != set(PROFILES):
        bad("host_census needs exactly both purposes")
    host_totals = {}
    for name in PROFILES:
        node, tile = hc[name], profiles[name]["tile_rows"]
        terms = terms_array(f"host_census.{name}.terms", node.get("terms") if isinstance(node, dict) else None)
        total = sum(terms.values())
        if not isinstance(node, dict) or node.get("total_bytes") != total:
            bad(f"host_census.{name}.total_bytes {node.get('total_bytes') if isinstance(node, dict) else node!r} != its terms' sum {total}")
        for key, want in (("forward_ple_stage_bytes", tile * PLE_ROW_BYTES), ("forward_token_stage_bytes", tile * TOKEN_ROW_BYTES),
                          ("forward_mesh_token_stage_bytes", tile * TOKEN_ROW_BYTES)):
            if terms.get(key) != want:
                bad(f"host_census.{name}.terms {key} {terms.get(key)!r} != tile {tile} x {want // tile} = {want}")
        host_totals[name] = total
    check_host_binding(led, canon, canon_rows, canonical is not None)
    host_peak = max(host_totals.values())
    if led["host_owned_bytes"] != host_peak or isinstance(led["host_owned_bytes"], bool):
        bad(f"host_owned_bytes {led['host_owned_bytes']!r} != the MAX {host_peak} of the per-purpose host census")
    # shared host reservation: the sequential MAX of the beta arm and the S1 purposes, never their sum
    hr = led["host_reservation"]
    if not isinstance(hr, dict):
        bad("host_reservation is not an object")
    beta = terms_array("host_reservation.terms", hr.get("terms") if isinstance(hr, dict) else None)
    if set(beta) != set(BETA_TERMS):
        bad(f"host_reservation.terms names {sorted(beta)} != {sorted(BETA_TERMS)}")
    beta_total = sum(beta.values())
    if hr.get("total_bytes") != beta_total or hr.get("beta_chunk_rows_ceiling") != BETA_CHUNK_ROWS:
        bad(f"host_reservation total_bytes {hr.get('total_bytes')!r} / beta_chunk_rows_ceiling {hr.get('beta_chunk_rows_ceiling')!r} "
            f"!= the terms' sum {beta_total} / {BETA_CHUNK_ROWS}")
    if beta["beta_forward_ple_stage_bytes"] != BETA_CHUNK_ROWS * PLE_ROW_BYTES or \
            beta["beta_forward_token_stage_bytes"] != BETA_CHUNK_ROWS * TOKEN_ROW_BYTES or \
            beta["beta_forward_mesh_token_stage_bytes"] != BETA_CHUNK_ROWS * TOKEN_ROW_BYTES:
        bad(f"host_reservation beta forward stages do not follow the {BETA_CHUNK_ROWS}-row ceiling")
    reserved = max(beta_total, host_peak)
    if led["host_reservation_owned_bytes"] != reserved or isinstance(led["host_reservation_owned_bytes"], bool):
        bad(f"host_reservation_owned_bytes {led['host_reservation_owned_bytes']!r} != the sequential MAX {reserved} of the beta arm {beta_total} "
            f"and the S1 purposes {host_peak} (a maximum, never an arm sum)")
    if not _pos(led["host_allowance_bytes"]):
        bad("host_allowance_bytes is not a positive integer")
    total = reserved + led["host_allowance_bytes"]
    if led["host_total_with_allowance_bytes"] != total:
        bad(f"host_total_with_allowance_bytes {led['host_total_with_allowance_bytes']!r} != reservation + allowance {total}")
    fab_host = led["fabric_hostmem_bytes"]
    if not _pos(fab_host) or not _gib(led["fabric_hostmem_gib"]) or led["fabric_hostmem_gib"] * GIB != fab_host:
        bad(f"fabric_hostmem_bytes {fab_host!r} is not fabric_hostmem_gib {led['fabric_hostmem_gib']!r} GiB")
    if total > fab_host:
        bad(f"host total {total} exceeds the fabric host reservation {fab_host}")
    if led["host_slack_bytes"] != fab_host - total:
        bad(f"host_slack_bytes {led['host_slack_bytes']!r} != fabric host - total {fab_host - total}")
    host_gib = math.ceil(total / GIB)
    if led["fabric_hostmem_gib"] != host_gib:
        bad(f"fabric_hostmem_gib {led['fabric_hostmem_gib']!r} != the total rounded up to whole GiB {host_gib}")
    if led["allowances_are_declared_not_measured"] is not True:
        bad("allowances_are_declared_not_measured must be true")
    if not isinstance(led["census_basis"], str) or not led["census_basis"].strip():
        bad("census_basis is empty")
    prov = led["provenance"]
    if not isinstance(prov, dict):
        bad("provenance is not an object")
    if final:
        for key in PROVENANCE_RUNTIME:
            if not isinstance(prov.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", prov[key]):
                bad(f"final ledger provenance.{key} {prov.get(key)!r} is not a 40-hex identity of the runtime source the census describes")
    return {"owned": owned, "peak_owned": peak, "host_peak": host_peak, "host_total": total, "host_gib": host_gib, "canonical_observation": chosen}


def print_memory(path):
    """`--print-memory`: "<fabric device GiB> <host GiB>" on stdout; the separate physical card on stderr. Refuses anything not final."""
    led = load_ledger(path)
    got = check_ledger(led, final=True)
    fab = fabric_reservation_gib(led)
    print(fab, got["host_gib"])
    print(f"print-memory: fabric exclusive device {fab} GiB (the physical card is a SEPARATE {led['card_bytes'] / GIB:g} GiB, card_bytes "
          f"{led['card_bytes']}); sequential owned peak {got['peak_owned']} B "
          f"(product {got['owned'][PRODUCT]}, primitive_selftest {got['owned'][PRIMITIVE]}); "
          f"host {got['host_gib']} GiB (sequential reservation {led['host_reservation_owned_bytes']} + allowance "
          f"{led['host_allowance_bytes']} = {got['host_total']} B)", file=sys.stderr)


def main(argv):
    if len(argv) == 3 and argv[1] == "print-memory":
        print_memory(argv[2])
        return 0
    if len(argv) == 4 and argv[1] == "check" and argv[2] == "--production":
        # the production seal's rule: final, and the reference derived from the ACTUAL post-initialization quote
        print(json.dumps(check_ledger(load_ledger(argv[3]), final=True, canonical=OBS_POST), indent=1))
        return 0
    if len(argv) in (3, 4) and argv[1] == "check" and (len(argv) == 3 or argv[2] == "--nonfinal"):
        path = argv[-1]
        got = check_ledger(load_ledger(path), final=len(argv) == 3)
        print(json.dumps(got, indent=1))
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
