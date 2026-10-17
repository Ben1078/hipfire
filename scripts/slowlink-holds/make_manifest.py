#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_manifest.py STAGE HOLDN BASE_SHA TIP_SHA PLE_SOURCE [--identity PATH] [--link-gbps X [--link-source TEXT]] [--out PATH | --stdout]

Builds the qwen4_slowlink_gate manifest (tests/common/slowlink_receipt.rs MANIFEST_REQUIRED) for hold N. Default output is
STAGE/manifests/manifest-holdN.json (--out PATH elsewhere, --stdout to print). Everything is READ, nothing is guessed:
  STAGE/prep-identity.json    passive host/PCI/stat identity (prep_identity.py over read-only ssh; default for --identity)
  STAGE/ledger-holdN.json     the FINAL declared ledger
  STAGE/bin-s1/BUILD-IDENTITY.json  kernel code-object hashes;  bin-s1/qwen4_slowlink_gate is hashed (the process the gate re-hashes)
  holds/model-identity.json   CACHED model + PLE-sidecar sha256 (no 125 GB hash is computed), valid only while the live size,
                              inode and mtime_ns in prep-identity.json equal the recorded ones.
link.gbps: without --link-gbps it is the passive sysfs theoretical payload rate (link.kind sysfs_theoretical, NOT a measurement);
with --link-gbps it is kind declared_measured_reference and --link-source names where that figure was measured.
A missing file, a stale cached hash, or a non-numeric link figure is an error."""
import argparse
import hashlib
import json
import os
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def need(path, what):
    if not os.path.exists(path):
        sys.exit(f"missing {what}: {path}")
    return path


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("stage", "hold", "base", "tip", "ple_source"):
        ap.add_argument(name)
    ap.add_argument("--identity")
    ap.add_argument("--link-gbps", type=float)
    ap.add_argument("--link-source", help="provenance of --link-gbps (e.g. the historical H0 reference); copied to link.measurement_source")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--out")
    out.add_argument("--stdout", action="store_true")
    a = ap.parse_args(argv[1:])
    stage, hold, base, tip, ple_source = a.stage, a.hold, a.base, a.tip, a.ple_source
    ident = json.load(open(need(a.identity or os.path.join(stage, "prep-identity.json"), "prep-identity.json (prep_identity.py)")))
    here = os.path.dirname(os.path.abspath(__file__))
    cached = json.load(open(need(os.path.join(here, "model-identity.json"), "cached model-identity.json")))
    for key, live_key in (("model", "model"), ("ple_sidecar", "ple_sidecar")):
        for field in ("path", "size_bytes", "inode", "mtime_ns"):
            if cached[key][field] != ident[live_key][field]:
                sys.exit(f"cached {key} hash is stale: {field} cached {cached[key][field]!r} != live {ident[live_key][field]!r}")
    bottleneck = ident.get("upstream_bottleneck_gbps")
    if not isinstance(bottleneck, (int, float)) or isinstance(bottleneck, bool):
        sys.exit("prep identity upstream_bottleneck_gbps is not numeric")
    if a.link_gbps is not None:
        link, link_kind = a.link_gbps, "declared_measured_reference"
    else:
        link, link_kind = float(bottleneck), "sysfs_theoretical"  # passive PCIe payload rate from negotiated speed/width: NOT a measurement
    if a.link_source and a.link_gbps is None:
        sys.exit("--link-source needs --link-gbps")
    led = json.load(open(need(os.path.join(stage, f"ledger-{hold}.json"), "declared memory ledger")))
    if not led.get("census_final"):
        sys.exit("declared ledger is not final")
    dev = led["card_bytes"]
    # Host reservation: host_reservation_owned_bytes when declared (the largest sequential arm), else the S1 census; host_owned_bytes
    # stays the S1 census (the strict first-cell driver check and owned_host_bytes).
    reserved = led.get("host_reservation_owned_bytes", led["host_owned_bytes"])
    if not isinstance(reserved, int) or isinstance(reserved, bool) or reserved < led["host_owned_bytes"]:
        sys.exit(f"host_reservation_owned_bytes {reserved!r} is not an integer >= host_owned_bytes {led['host_owned_bytes']}")
    ledger = [dev / 2**30, (reserved + led["host_allowance_bytes"]) / 2**30]
    owned_device = led["device_owned_bytes"]  # core + sequential diagnostic only
    owned_host = led["host_owned_bytes"]
    split_keys = ("device_weights_bytes", "s1_tile_census_bytes", "s1_extras_bytes", "sequential_diag_bytes", "safety_margin_bytes")
    split = {k: led.get(k) for k in split_keys}
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in split.values()):
        sys.exit("declared ledger needs positive integer " + ", ".join(split_keys))
    if sum(split[k] for k in split_keys[:4]) != owned_device:
        sys.exit(f"ledger split {sum(split[k] for k in split_keys[:4])} != device_owned_bytes {owned_device}")
    basis = led.get("census_basis")
    if not (isinstance(basis, str) and basis.strip()) or owned_device <= 0 or owned_host <= 0:
        sys.exit("declared ledger needs a non-empty census_basis and positive owned device/host bytes")
    bi = json.load(open(need(os.path.join(stage, "bin-s1", "BUILD-IDENTITY.json"), "BUILD-IDENTITY.json")))
    if "kernel_code_object_hashes" not in bi:
        sys.exit("BUILD-IDENTITY.json has no kernel_code_object_hashes")
    model_sha = cached["model"]["sha256"]
    walk = ident["pci_walk"]
    manifest = {
        "schema": "hipfire.qwen4.slowlink_manifest.v1",
        "source": {"base": base, "tip": tip, "branch": "feat/slowlink-s1"},
        "binary": {"path": os.path.join(stage, "bin-s1", "qwen4_slowlink_gate"),
                   "sha256": sha(os.path.join(stage, "bin-s1", "qwen4_slowlink_gate"))},
        "model": {"path": cached["model"]["path"], "size_bytes": cached["model"]["size_bytes"], "sha256": model_sha,
                  "sha256_source": "cached receipts (model-identity.json); live size/inode/mtime_ns match"},
        "ple": {"source": ple_source, "format": "q4", "sidecars": [cached["ple_sidecar"]["path"]],
                "sidecar_identity": [{k: cached["ple_sidecar"][k] for k in ("path", "sha256", "size_bytes", "inode", "mtime_ns")}]},
        "kernel": {"code_object_hashes": bi["kernel_code_object_hashes"]},
        "host": {"hostname": ident["host"], "topology": walk},
        "device": {"uuid": ident["uuid"], "arch": "gfx1100"},
        "link": {"gbps": link, "kind": link_kind, "measurement_source": a.link_source, "topology_walk": walk, "upstream_bottleneck_gbps": float(bottleneck),
                 "upstream_bottleneck_bdf": ident.get("upstream_bottleneck_bdf")},
        "memory_ledger": {**split, "owned_device_bytes": owned_device, "owned_host_bytes": owned_host, "reservation_owned_host_bytes": reserved,
                          "sequential_arm_host_census": led.get("sequential_arm_host_census"), "census_basis": basis,
                          "device_gib": float(ledger[0]), "host_unreclaimable_gib": float(ledger[1]), "allowances_are_declared_not_measured": True,
                          "ledger_provenance": led.get("provenance")},
    }
    text = json.dumps(manifest, indent=1) + "\n"
    if a.stdout:
        sys.stdout.write(text)
        return
    dest = a.out or os.path.join(stage, "manifests", f"manifest-{hold}.json")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w") as f:
        f.write(text)
    print(dest)


if __name__ == "__main__":
    main(sys.argv)
