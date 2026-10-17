#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prep_identity.py BDF MODEL SIDECAR > prep-identity.json

PASSIVE host identity for the manifest, gathered without a GPU process, a fabric hold or any hashing of the 125 GB model: the
hostname, the card's unique_id and PCI topology walk (every bridge from the root port to the card, numeric payload GB/s per link,
sysfs reads only) and os.stat of the model and its PLE sidecar. Run it ON the XTX host over read-only ssh:
    ssh hipx python3 - 0000:67:00.0 MODEL SIDECAR < prep_identity.py > STAGE/prep-identity.json
It is NOT the hold's identity.json (the hold writes that itself); it only lets make_manifest.py build the manifest before the hold."""
import json
import os
import sys
import time


def rd(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def speed(v):
    try:
        return float((v or "").split()[0])
    except (ValueError, IndexError):
        return None


def gbps(node):  # PCIe payload GB/s per link: GT/s * width * encoding / 8
    s, w = speed(node["speed"]), int(node["width"] or 0)
    if not s or not w:
        return None
    return s * w * (128 / 130 if s >= 8 else 0.8) / 8


def stat(path):
    st = os.stat(path)
    return {"path": path, "size_bytes": st.st_size, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "nlink": st.st_nlink}


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    bdf, model, sidecar = argv[1:]
    parts = os.path.realpath("/sys/bus/pci/devices/" + bdf).split("/")
    cur = "/sys/devices/" + parts[3]
    walk = []
    for part in parts[4:]:
        cur += "/" + part
        node = {"bdf": part, "speed": rd(cur + "/current_link_speed"), "width": rd(cur + "/current_link_width"),
                "max_speed": rd(cur + "/max_link_speed"), "max_width": rd(cur + "/max_link_width")}
        node["gbps"] = gbps(node)
        walk.append(node)
    rates = [(n["gbps"], n["bdf"]) for n in walk if n["gbps"]]
    if not rates:
        sys.exit(f"no PCIe link rates readable under {bdf}")
    value, at = min(rates)
    uid = rd(f"/sys/bus/pci/devices/{bdf}/unique_id")
    ident = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
             "uuid": "GPU-" + uid if uid else None, "card_bdf": bdf, "pci_walk": walk,
             "upstream_bottleneck_gbps": value, "upstream_bottleneck_bdf": at,
             "model": stat(model), "ple_sidecar": stat(sidecar), "kernel": os.uname().release,
             "note": "passive sysfs/stat read; the model is NOT hashed here"}
    if not ident["uuid"]:
        sys.exit(f"no unique_id for {bdf}")
    json.dump(ident, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
