#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_residual.py CELL_DIR LEDGER.json

Fail-closed memory residual gate of Hold 1's FIRST cell (and any cell passed to it).

LEDGER.json is the DECLARED numeric ledger (ledger-template.json): the exact source-owned census (`device_owned_bytes`,
`host_owned_bytes`), the committed device core + sequential diagnostic (`device_owned_bytes`; the 2 GiB session-cache budget and the pending ceiling are recorded, never added), and the two fixed ALLOWANCES (1,610,612,736 B device,
2,147,483,648 B host). The allowances are declared limits, NOT measurements. A ledger with `census_final` false or any null owned
term is refused (exit 2): no hold is judged against a ledger nobody finished.

Measured from the cell's own samples (hold-lib.sh `cell` writes CELL_DIR/baseline.json; sample_mem.py writes mem.csv with RAW
components, never summed):
  device peak delta   = max(vram_used) - baseline vram_used          (card-wide sysfs, includes the driver/runtime)
Host residual = MAX of three INDEPENDENT terms, each minus `host_owned_bytes` (never a sum: VmPin, RssAnon and the KFD file-backed
pinned pages can describe the SAME physical pages):
  rss term   : VmRSS - clean RSS of REGULAR-FILE mappings (smaps; /dev/kfd, /dev/dri, /dev/zero, memfd, anon, [..] and (deleted)
               maps are never subtracted, so pinned-as-anon and device maps stay counted; only the reclaimable clean page-cache
               part of ordinary files is removed - NOT all of RssFile, because a KFD-pinned BO can be file-backed and non-reclaimable)
  pin term   : VmPin
  gtt term   : gtt_used - baseline gtt_used
The method and every raw component of the peak sample are recorded. Residual = measured - declared owned; the gate FAILS when it
exceeds the allowance. Output is JSON on stdout; exit 0 pass, 1 fail, 2 refused."""
import csv
import json
import sys


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    cell, ledger_path = argv[1], argv[2]
    led = json.load(open(ledger_path))
    need = ["device_owned_bytes", "device_weights_bytes", "s1_tile_census_bytes", "s1_extras_bytes", "sequential_diag_bytes",
            "safety_margin_bytes", "device_allowance_bytes", "card_bytes", "host_owned_bytes", "host_allowance_bytes", "census_basis"]
    if not led.get("census_final") or any(led.get(k) in (None, "") for k in need):
        print(json.dumps({"status": "refused", "reason": "ledger is not final (census_final false or a null term)"}))
        return 2
    base = json.load(open(f"{cell}/baseline.json"))
    rows = list(csv.DictReader(open(f"{cell}/mem.csv")))
    if len(rows) < 2:
        print(json.dumps({"status": "refused", "reason": "fewer than 2 memory samples: the cell was too short to measure"}))
        return 2

    def col(name):
        return [int(r[name] or 0) for r in rows if r.get(name) not in (None, "")]

    vram = col("vram_used")
    vmrss, vmpin, fclean = col("proc_vmrss_kb"), col("proc_vmpin_kb"), col("file_clean_kb")
    if not vmrss or max(vmrss) == 0:
        print(json.dumps({"status": "refused", "reason": "no per-process samples (the process was never seen)"}))
        return 2
    dev_peak = max(vram) - int(base["vram_used"])
    gtt_delta = [g - int(base["gtt_used"]) for g in col("gtt_used")]
    rss_term = [(r - c) * 1024 for r, c in zip(vmrss, fclean)]
    pin_term = [p * 1024 for p in vmpin]
    peak_i = max(range(len(rows)), key=lambda i: max(rss_term[i], pin_term[i], gtt_delta[i]))
    terms = {"rss_minus_reclaimable_regular_file_clean": max(rss_term), "vmpin": max(pin_term), "gtt_delta": max(gtt_delta)}
    host_peak = max(terms.values())
    dev_declared = led["device_owned_bytes"]  # committed core + sequential diagnostic; cache budget/pending are never added
    host_declared = led["host_owned_bytes"]
    split = [led["device_weights_bytes"], led["s1_tile_census_bytes"], led["s1_extras_bytes"], led["sequential_diag_bytes"]]
    if sum(split) != led["device_owned_bytes"]:
        print(json.dumps({"status": "refused", "reason": f"ledger split {sum(split)} != device_owned_bytes {led['device_owned_bytes']}"}))
        return 2
    # Main-approved guard (first primitive; safety_margin_bytes is an admission safety margin, no device allocation, not added twice to the committed fit): max device - weights <= (tile + extras + diag) + safety + driver allowance. The stricter
    # total residual below (max device - device_owned <= allowance) is retained and still decides.
    guard_budget = sum(split[1:]) + led["safety_margin_bytes"] + led["device_allowance_bytes"]
    guard_lhs = dev_peak - led["device_weights_bytes"]
    dev_resid = dev_peak - dev_declared
    host_resid = host_peak - host_declared
    raw = {k: rows[peak_i][k] for k in rows[0]}
    out = {"device": {"measured_peak_delta_bytes": dev_peak, "declared_owned_bytes": dev_declared, "residual_bytes": dev_resid,
                      "allowance_bytes": led["device_allowance_bytes"], "within_allowance": dev_resid <= led["device_allowance_bytes"],
                      "scratch_driver_guard": {"max_device_minus_weights_bytes": guard_lhs, "budget_bytes": guard_budget,
                                               "terms": {"s1_tile_census_bytes": split[1], "s1_extras_bytes": split[2],
                                                         "sequential_diag_bytes": split[3], "safety_margin_bytes": led["safety_margin_bytes"],
                                                         "driver_allowance_bytes": led["device_allowance_bytes"]},
                                               "within_budget": guard_lhs <= guard_budget}},
           "host": {"method": "MAX of (VmRSS - reclaimable regular-file clean RSS), VmPin, GTT delta; each minus host_owned; never summed",
                    "terms_peak_bytes": terms, "measured_peak_bytes": host_peak, "raw_components_at_peak_sample": raw,
                    "declared_owned_bytes": host_declared,
                    "residual_bytes": host_resid, "allowance_bytes": led["host_allowance_bytes"],
                    "within_allowance": host_resid <= led["host_allowance_bytes"]},
           "reported": {"vmrss_peak_bytes": max(vmrss) * 1024, "vmpin_peak_bytes": max(vmpin) * 1024, "samples": len(rows)},
           "allowances_are_declared_not_measured": True,
           "ledger_provenance": led.get("provenance")}
    out["status"] = ("pass" if out["device"]["within_allowance"] and out["device"]["scratch_driver_guard"]["within_budget"]
                     and out["host"]["within_allowance"] else "fail")
    print(json.dumps(out, indent=1))
    return 0 if out["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
