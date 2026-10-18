#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_manifest.py STAGE HOLDN BASE_SHA TIP_SHA PLE_SOURCE [--identity PATH] [--link-gbps X [--link-source TEXT]]
                    [--harness-commit SHA --harness-tree SHA] [--out PATH | --stdout]

Builds the qwen4_slowlink_gate manifest (tests/common/slowlink_receipt.rs MANIFEST_REQUIRED) for hold N. Default output is
STAGE/manifests/manifest-holdN.json (--out PATH elsewhere, --stdout to print). Everything is READ, nothing is guessed:
  STAGE/prep-identity.json    passive host/PCI/stat identity (prep_identity.py over read-only ssh; default for --identity)
  STAGE/ledger-holdN.json     the FINAL declared per-purpose ledger (ledger-template.json, validated by ledger_profiles.py: BOTH purpose profiles
                              profiles.product (tile 768) and profiles.primitive_selftest (tile 512) with exact term arrays, core/owned/peak sums;
                              a non-final or flat ledger is refused)
  STAGE/fixtures/prompts.short.jsonl, prompts.short.beta.jsonl, beta-input-proof.json   the canonical greedy input, the metadata-only beta
                              input (canonical minus ONLY resume_at) and their identity proof; recomputed here and bound as manifest.inputs
  STAGE/bin-s1/BUILD-IDENTITY.json  runtime source base/tip/tree (BASE_SHA/TIP_SHA must equal it), kernel code-object hashes;
                              bin-s1/qwen4_slowlink_gate is hashed (the process the gate re-hashes) and must equal the recorded hash
  holds/model-identity.json   CACHED model + PLE-sidecar sha256 (no 125 GB hash is computed), valid only while the live size,
                              inode and mtime_ns in prep-identity.json equal the recorded ones.
link.gbps: without --link-gbps it is the passive sysfs theoretical payload rate (link.kind sysfs_theoretical, NOT a measurement);
with --link-gbps it is kind declared_measured_reference and --link-source names where that figure was measured.
memory_ledger records BOTH purpose profiles verbatim (memory_ledger.profiles.product / .primitive_selftest, the exact key set the gate's
receipt requires; no flat profile fields, no aliases), the root device_owned_bytes = the MAX of the per-purpose owned bytes, the physical card
(declared card_bytes, device_gib), the separate fabric exclusive reservation (the ledger's explicit fabric_reservation_device_gib, never a
constant), the shared host reservation (the sequential max, never an arm sum), the product headroom threshold
(check_headroom.MIN_FREE_BYTES = 536,870,912) and Main's policy; the owned census is information only. device.arch is the architecture of
the sealed candidate kernel pack (BUILD-IDENTITY.json
pack.manifest.arch, validated by stage_build.py): staging metadata, not a runtime capability claim.
A missing file, a stale cached hash, a non-numeric link figure, a supplied base/tip that differs from BUILD-IDENTITY.json source{base,tip},
a staged gate whose sha256 differs from BUILD-IDENTITY.json, or a ledger provenance naming another build is an error. The manifest records
source.tree from BUILD-IDENTITY.json and preserves ledger.provenance; --harness-commit/--harness-tree (supplied together, externally: a harness
commit cannot name its own tree) are recorded as manifest.harness. Nothing is invented or sealed here."""
import argparse
import hashlib
import json
import copy
import os
import re
import sys

import beta_input
import ledger_profiles as lp
from check_headroom import MIN_FREE_BYTES


def fabric_reservation_gib(led):
    """The ledger's explicit fabric exclusive-use device reservation in GiB (a finite number > 0); anything else is an error."""
    return lp.fabric_reservation_gib(led)


def pack_arch(bi):
    """The kernel-pack architecture sealed into BUILD-IDENTITY.json (pack.manifest.arch); kernel_arch, when present, must agree."""
    try:
        arch = bi["pack"]["manifest"]["arch"]
    except (KeyError, TypeError):
        sys.exit("BUILD-IDENTITY.json has no pack.manifest.arch")
    if not isinstance(arch, str) or not re.fullmatch(r"[A-Za-z0-9_]+", arch):
        sys.exit(f"BUILD-IDENTITY.json pack.manifest.arch {arch!r} is not a single architecture name")
    if "kernel_arch" in bi and bi["kernel_arch"] != arch:
        sys.exit(f"BUILD-IDENTITY.json kernel_arch {bi['kernel_arch']!r} != pack.manifest.arch {arch!r}")
    return arch


def hex40(what, value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        sys.exit(f"{what} {value!r} is not 40 lowercase hex characters")
    return value


def check_build_identity(bi, base, tip, gate_sha):
    """The staged candidate's BUILD-IDENTITY.json must be an s1 identity whose source base/tip equal the SUPPLIED ones and whose recorded
    qwen4_slowlink_gate sha256 equals the hash of the staged file; returns the runtime source tree it records."""
    if not isinstance(bi, dict) or bi.get("schema") != "s1.build_identity.v1" or bi.get("arm") != "s1":
        sys.exit("BUILD-IDENTITY.json is not an s1.build_identity.v1 identity for arm s1")
    src = bi.get("source")
    if not isinstance(src, dict):
        sys.exit("BUILD-IDENTITY.json has no source{base,tip,tree}")
    for field, supplied in (("base", base), ("tip", tip)):
        if src.get(field) != supplied:
            sys.exit(f"BUILD-IDENTITY.json source.{field} {src.get(field)!r} != supplied {supplied!r}: refusing to label this binary set")
    tree = hex40("BUILD-IDENTITY.json source.tree", src.get("tree"))
    recorded = bi.get("binary_sha256")
    recorded = recorded.get("qwen4_slowlink_gate") if isinstance(recorded, dict) else None
    if recorded != gate_sha:
        sys.exit(f"BUILD-IDENTITY.json binary_sha256.qwen4_slowlink_gate {recorded!r} != staged file sha256 {gate_sha}")
    return tree


# ledger.provenance fields supplied at the seal that name the runtime source the census describes (checked when present).
PROVENANCE_BINDINGS = (("census_source_commit", "tip"), ("census_source_tree", "tree"), ("census_source_base", "base"), ("source_tip", "tip"),
                       ("source_tree", "tree"), ("source_base", "base"), ("runtime_commit", "tip"), ("runtime_tree", "tree"),
                       ("binary_sha256", "binary"))


def check_ledger_provenance(prov, base, tip, tree, gate_sha):
    """The ledger's provenance is PRESERVED verbatim in the manifest (including externally supplied runtime/harness commit and tree
    fields); every runtime-identity field it carries must equal the staged BUILD-IDENTITY/binary, so a census can never be attached to a
    different build. Absent optional fields are not invented."""
    if not isinstance(prov, dict) or not prov:
        sys.exit("declared ledger needs a non-empty provenance object")
    actual = {"tip": tip, "tree": tree, "base": base, "binary": gate_sha}
    for key, kind in PROVENANCE_BINDINGS:
        if key in prov and prov[key] != actual[kind]:
            sys.exit(f"ledger provenance.{key} {prov[key]!r} != staged {kind} {actual[kind]!r}: refusing to attach this census")
    return prov


def check_ledger_arithmetic(led, final=True, canonical=lp.OBS_POST):
    """Self-consistency of the declared per-purpose ledger (ledger_profiles.check_ledger: exact profile key sets and tiles, term arrays
    against their totals, core/owned/peak sums, host census/reservation). final=True (the seal) also refuses a non-final ledger, and the
    seal requires the PRODUCTION binding: the reference figures derive from the ACTUAL post-initialization quote (canonical_observation
    post_initialization_actual); a ledger bound to the no-loss counterfactual row is refused. Unit fixtures that exercise only the arithmetic pass
    canonical=None."""
    return lp.check_ledger(led, final=final, canonical=canonical)


def beta_input_binding(stage, model_sha, prov):
    """Bind the canonical greedy input, the metadata-only beta input and their identity proof: the proof is RECOMPUTED from the two staged
    files (never trusted) and must equal the staged beta-input-proof.json; the returned record carries both file hashes."""
    fixtures = os.path.join(stage, "fixtures")
    canon = need(os.path.join(fixtures, "prompts.short.jsonl"), "canonical greedy input (make_fixtures.py)")
    beta = need(os.path.join(fixtures, "prompts.short.beta.jsonl"), "beta greedy input (make_fixtures.py)")
    proof_path = need(os.path.join(fixtures, "beta-input-proof.json"), "beta input identity proof (make_fixtures.py)")
    proof = beta_input.verify(canon, beta)
    with open(proof_path) as f:
        if json.load(f) != proof:
            sys.exit(f"{proof_path} differs from the identity proof recomputed from the staged canonical and beta inputs")
    return {"greedy_canonical": proof["canonical"], "greedy_beta": proof["beta"],
            "beta_adapter": {"schema": proof["schema"], "label": proof["label"], "stripped_keys": proof["stripped_keys"],
                             "allowed_beta_keys": proof["allowed_beta_keys"], "proof_sha256": sha(proof_path), "records": proof["records"]},
            "model_sha256": model_sha, "model_config_sha256": (prov or {}).get("model_config_sha256")}




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
    ap.add_argument("--harness-commit", type=lambda v: hex40("--harness-commit", v),
                    help="40-hex commit of the harness checkout, supplied externally and recorded as manifest.harness (needs --harness-tree)")
    ap.add_argument("--harness-tree", type=lambda v: hex40("--harness-tree", v), help="40-hex git tree of the harness commit")
    a = ap.parse_args(argv[1:])
    if bool(a.harness_commit) != bool(a.harness_tree):
        sys.exit("--harness-commit and --harness-tree are supplied together or not at all")
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
    led = lp.load_ledger(need(os.path.join(stage, f"ledger-{hold}.json"), "declared memory ledger"))
    # BOTH purpose profiles (exact key sets, tiles 768/512, term arrays, core/owned sums, the root MAX peak, the shared host maximum) are
    # validated before anything is sealed; a non-final, flat or single-profile ledger is refused here.
    check_ledger_arithmetic(led)
    card_gib = led["card_bytes"] / 2**30
    reserved = led["host_reservation_owned_bytes"]
    host_gib = (reserved + led["host_allowance_bytes"]) / 2**30
    # the profiles are sealed verbatim with the exact key set the gate's receipt reads; no flat fields, no aliases
    profiles = {name: copy.deepcopy({k: led["profiles"][name][k] for k in lp.PROFILE_KEYS}) for name in lp.PROFILES}
    basis = led["census_basis"]
    bi = json.load(open(need(os.path.join(stage, "bin-s1", "BUILD-IDENTITY.json"), "BUILD-IDENTITY.json")))
    if "kernel_code_object_hashes" not in bi:
        sys.exit("BUILD-IDENTITY.json has no kernel_code_object_hashes")
    arch = pack_arch(bi)
    gate_path = os.path.join(stage, "bin-s1", "qwen4_slowlink_gate")
    gate_sha = sha(gate_path)
    # The supplied base/tip must be the ones this binary set was STAGED from, and the staged gate must be the hashed one: a manifest can
    # never label a binary with a source it was not built from. The runtime source tree comes from BUILD-IDENTITY, never from an argument.
    tree = check_build_identity(bi, base, tip, gate_sha)
    provenance = check_ledger_provenance(led.get("provenance"), base, tip, tree, gate_sha)
    harness = {"commit": a.harness_commit, "tree": a.harness_tree,
               "supplied": "externally (the harness commit cannot contain its own commit or tree)"} if a.harness_commit else None
    fabric_gib = fabric_reservation_gib(led)
    model_sha = cached["model"]["sha256"]
    inputs = beta_input_binding(stage, model_sha, provenance)
    walk = ident["pci_walk"]
    manifest = {
        "schema": "hipfire.qwen4.slowlink_manifest.v1",
        "source": {"base": base, "tip": tip, "tree": tree, "branch": "feat/slowlink-s1"},
        "binary": {"path": gate_path, "sha256": gate_sha},
        "model": {"path": cached["model"]["path"], "size_bytes": cached["model"]["size_bytes"], "sha256": model_sha,
                  "sha256_source": "cached receipts (model-identity.json); live size/inode/mtime_ns match"},
        "ple": {"source": ple_source, "format": "q4", "sidecars": [cached["ple_sidecar"]["path"]],
                "sidecar_identity": [{k: cached["ple_sidecar"][k] for k in ("path", "sha256", "size_bytes", "inode", "mtime_ns")}]},
        "kernel": {"code_object_hashes": bi["kernel_code_object_hashes"]},
        "host": {"hostname": ident["host"], "topology": walk},
        "device": {"uuid": ident["uuid"], "arch": arch},
        "link": {"gbps": link, "kind": link_kind, "measurement_source": a.link_source, "topology_walk": walk, "upstream_bottleneck_gbps": float(bottleneck),
                 "upstream_bottleneck_bdf": ident.get("upstream_bottleneck_bdf")},
        "memory_ledger": {"profiles": profiles,
                          # the two purposes load as sequential fresh processes: the root owned bytes are the MAX of the per-purpose owned
                          # bytes, never product core + the primitive diagnostic
                          "device_owned_bytes": led["device_owned_bytes"], "device_owned_note": led["device_owned_note"],
                          # reference projection (never observed, never a target) and the Main-bound R5 admission quotes (scalar witnesses of the
                          # shared quote per purpose and memory observation, re-verified by ledger_profiles.check_admission_quotes)
                          "device_projection": led["device_projection"], "admission_quotes": led["admission_quotes"],
                          "owned_host_bytes": led["host_owned_bytes"], "reservation_owned_host_bytes": reserved,
                          "host_census_by_purpose": led["host_census"], "sequential_arm_host_census": led["host_reservation"],
                          "census_basis": basis,
                          # device_gib is the declared PHYSICAL card figure; the fabric exclusive-use reservation is the ledger's separate explicit field.
                          "device_gib": float(card_gib), "physical_card_gib": float(card_gib), "fabric_exclusive_device_gib": fabric_gib,
                          "host_unreclaimable_gib": float(host_gib), "allowances_are_declared_not_measured": True,
                          "census_role": "information only (source/header arithmetic; not a driver figure, no owned+allowance fit is judged)",
                          "product_headroom": {"min_free_bytes": MIN_FREE_BYTES, "scope": "Hold 1 PRODUCT state cell (profile product, tile 768), whole run",
                                               "sources": ["external sysfs mem_info_vram_used/total at a requested 20 ms",
                                                           "in-process hipMemGetInfo pre_load/post_load/end (gate evidence.memory_diagnostics.samples)"],
                                               "each_source_uses_its_own_reported_total": True,
                                               "policy": "Main: both observed minima of free VRAM >= 512 MiB in BOTH domains over a COMPLETED gate report; missing, "
                                                         "partial or invalid telemetry is INCOMPLETE (the progress sidecar is forensic only, never a pass); the primitive "
                                                         "peak is non-gating; census/residual/peak and attribution fields are information, never classified as a measurement"},
                          "ledger_provenance": provenance},
        "inputs": inputs,
        "harness": harness,
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
