#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""admission_witness.py verify WITNESS.json [--resident N --trunk T --snapshot-budget B]

CPU-only INDEPENDENT verifier of the scalar S1 admission witness (R5 adaptive admission). The runtime selects the resident layer count with ONE
shared Rust projection (`project_at_tile`, post-initialization `PlacementBudget`); the gate serializes the scalars of that same quote as
`load.placement.detail.admission`, and the ledger carries the same scalars per purpose and memory observation (SLOWLINK_ADMISSION_QUOTE_R5
emit). This module re-derives every figure from the scalars ALONE with exact integer arithmetic and holds NO selection policy of its own: it
only proves that the claimed count is the maximal count that fits (the same check `admission_witness_problems` makes in
tests/common/slowlink_receipt.rs). It never recomputes a projection from a model and never hardcodes a resident floor: ten resident layers is
PREFERRED (information only, the deficit is reported), the claimed count may be anything from 0 up to the trunk minus one.

The witness (exactly these keys, no others required, extras ignored):
  free_before_initialization_bytes (u64|null)   free device bytes before the host link / runtime initialization (measured interval start)
  free_at_admission_bytes                        FRESH post-initialization hipMemGetInfo free: the initialization is charged ONCE through it
  observed_initialization_bytes (u64|null)       measured interval loss before - after (0 when the free rose; null when no before sample)
  non_expert_bytes, resident_layer_bytes (BACKED layer cost), joint_reserve_bytes (the ENTIRE reserve incl. base, S1, granule, safety once)
  trunk_layers, resident_layers, preferred_resident_layers (10)
  needed_bytes = non_expert + resident_layers * resident_layer + joint_reserve
  slack_bytes  = free_at_admission - needed (>= 0)
  preferred_deficit_bytes = max(0, needed(10) - free_at_admission)   (information, never a rejection)
  memory_basis "post_initialization_actual", initialization_charged_via_free true

Accepted iff: every field present and typed; needed re-derives; the count FITS (needed <= free); ONE MORE does NOT fit (slack < layer cost);
at least one host layer remains (count < trunk: an all-resident load bypasses S1 and never claims a profile); the preferred deficit and
the observed initialization interval re-derive; and, when supplied, the count/trunk/snapshot allowance equal the bundle's own."""
import json
import sys

PREFERRED_RESIDENT_LAYERS = 10
MEMORY_BASIS = "post_initialization_actual"
U64 = (1 << 64) - 1

INT_KEYS = ("free_at_admission_bytes", "non_expert_bytes", "resident_layer_bytes", "joint_reserve_bytes", "trunk_layers", "resident_layers",
            "preferred_resident_layers", "needed_bytes", "slack_bytes", "preferred_deficit_bytes")
OPTIONAL_KEYS = ("free_before_initialization_bytes", "observed_initialization_bytes")
ALL_KEYS = INT_KEYS + OPTIONAL_KEYS + ("memory_basis", "initialization_charged_via_free")


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def witness_problems(w, resident=None, actual_trunk=None, snapshot_budget=None, check_actual=True):
    """Every reason `w` is not a proof of a valid, maximal, actual placement (empty list = proven). With check_actual the bundle's own
    resident/trunk/snapshot figures must be supplied and equal; ledger entries (no bundle) pass check_actual=False."""
    if not isinstance(w, dict):
        return ["admission witness missing: the actual placement is not proved maximal-fits"]
    bad = []
    for key in INT_KEYS:
        v = w.get(key)
        if not _int(v) or abs(v) > U64 or (v < 0 and key != "slack_bytes"):
            bad.append(f"admission.{key} {v!r} missing or not an integer (non-negative except slack_bytes)")
    for key in OPTIONAL_KEYS:
        if key not in w or not (w[key] is None or (_int(w[key]) and 0 <= w[key] <= U64)):
            bad.append(f"admission.{key} missing (unsigned integer or null)")
    if w.get("memory_basis") != MEMORY_BASIS:
        bad.append(f"admission.memory_basis {w.get('memory_basis')!r} is not {MEMORY_BASIS!r}: the quote must use the fresh post-initialization free")
    if w.get("initialization_charged_via_free") is not True:
        bad.append("admission.initialization_charged_via_free is not true: initialization is charged once, through the lower actual free")
    if bad:
        return bad
    free, non_expert, layer, reserve = (w["free_at_admission_bytes"], w["non_expert_bytes"], w["resident_layer_bytes"], w["joint_reserve_bytes"])
    trunk, n = w["trunk_layers"], w["resident_layers"]
    if w["preferred_resident_layers"] != PREFERRED_RESIDENT_LAYERS:
        bad.append(f"admission.preferred_resident_layers {w['preferred_resident_layers']} != {PREFERRED_RESIDENT_LAYERS}")
    if layer == 0:
        return bad + ["admission.resident_layer_bytes is 0: maximality is unprovable"]

    def needed(count):
        return non_expert + count * layer + reserve

    if needed(n) != w["needed_bytes"]:
        bad.append(f"admission.needed_bytes {w['needed_bytes']} != non_expert + {n} x resident_layer + reserve = {needed(n)}")
    if needed(n) > free:
        bad.append(f"admission oversubscribed: {n} resident layers need {needed(n)} B but only {free} B were free at admission")
    elif free - needed(n) != w["slack_bytes"]:
        bad.append(f"admission.slack_bytes {w['slack_bytes']} != free {free} - needed {needed(n)}")
    if n >= trunk:
        bad.append(f"admission claims {n} resident of {trunk} trunk layers: no host layer remains, so an S1 profile must not be claimed")
    if needed(n + 1) <= free:
        bad.append(f"admission is not maximal: {n + 1} resident layers would also fit ({needed(n + 1)} B <= {free} B free)")
    deficit = max(0, needed(PREFERRED_RESIDENT_LAYERS) - free)
    if deficit != w["preferred_deficit_bytes"]:
        bad.append(f"admission.preferred_deficit_bytes {w['preferred_deficit_bytes']} != the {PREFERRED_RESIDENT_LAYERS}-layer deficit {deficit} "
                   f"(information only, never a rejection)")
    before, observed = w["free_before_initialization_bytes"], w["observed_initialization_bytes"]
    if before is None:
        ok = observed is None
    elif before >= free:
        ok = observed == before - free
    else:  # the fresh free rose: the actual free stays authoritative and no positive loss is invented
        ok = observed in (None, 0)
    if not ok:
        bad.append(f"admission.observed_initialization_bytes {observed!r} != the measured interval before {before!r} - after {free}")
    if check_actual:
        if resident != n:
            bad.append(f"admission.resident_layers {n} != the bundle's actual resident layers {resident!r}")
        if actual_trunk != trunk:
            bad.append(f"admission.trunk_layers {trunk} != the loaded model's trunk layers {actual_trunk!r}")
        if snapshot_budget != w["slack_bytes"]:
            bad.append(f"the profile's snapshot allowance {snapshot_budget!r} != the quote's slack {w['slack_bytes']} at the selected count")
    return bad


def main(argv):
    args = argv[1:]
    if len(args) < 2 or args[0] != "verify":
        sys.exit(__doc__)
    opts = {"--resident": None, "--trunk": None, "--snapshot-budget": None}
    rest = args[2:]
    while rest:
        if rest[0] not in opts or len(rest) < 2:
            sys.exit(__doc__)
        opts[rest[0]] = int(rest[1])
        rest = rest[2:]
    with open(args[1]) as f:
        w = json.load(f)
    check_actual = any(v is not None for v in opts.values())
    problems = witness_problems(w, opts["--resident"], opts["--trunk"], opts["--snapshot-budget"], check_actual=check_actual)
    if problems:
        sys.exit("admission witness: " + "; ".join(problems))
    print("admission witness: ok")


if __name__ == "__main__":
    main(sys.argv)
