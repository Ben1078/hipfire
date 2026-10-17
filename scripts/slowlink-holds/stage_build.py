#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""stage_build.py --stage STAGE --arm s1|beta --source-base BASE --source-tip TIP --source-tree TREE
                  --target-dir CARGO_TARGET --kernel-pack PACK.tar.gz [--build-command CMD ...]

Stages ALREADY BUILT release binaries and an official kernel pack into STAGE/bin-ARM/ and writes BUILD-IDENTITY.json.
It builds nothing, runs no binary, copies no model or HOME.

Binaries (CARGO_TARGET/release/...):
  s1    examples/{qwen4_slowlink_gate,qwen4_greedy_ids,qwen4_kld,qwen4_parity} + daemon
  beta  examples/{qwen4_greedy_ids,qwen4_kld} + hipfire + daemon, and bin-beta/pins.json = {"hipfire": sha256, "daemon": sha256}
They are copied flat into STAGE/bin-ARM/ (basename), mode 0755.

Kernel pack (scripts/build-kernel-pack.sh layout): a tarball holding `manifest.json` and `<arch>/<module>.{hsaco,hash,index.json}`.
The architecture is DERIVED from the pack: every module member lives under ONE top-level directory <arch>, and manifest.arch must name
exactly that directory (a mixed or mismatched pack is refused). This is staging metadata only; which architectures the runtime has
kernels for is the runtime's / pack registry's truth, not this script's.
Required: manifest.commit == --source-tip, manifest.arch == <arch>, every module has all three files and nothing else is in the
pack, every index names its module/arch and its object_sha256 equals the SHA-256 of the copied .hsaco. Members are validated
(regular files only, no duplicate, absolute or `..` paths, only manifest.json and <arch>/* entries) and extracted with tarfile's `data`
filter into a scratch directory inside STAGE, then installed EXE-relative at STAGE/bin-ARM/kernels/compiled/<arch> (the runtime's
exe-neighbour probe, rdna-compute kernel_pack::pack_arch_dir).

BUILD-IDENTITY.json: source{base,tip,tree}, kernel_arch (the validated pack arch), binary_sha256{basename: sha}, kernel_code_object_hashes{module: SHA-256 of the actual
.hsaco}, kernel_source_hashes{module: index.source_sha256} (a SOURCE hash, never labelled a code-object hash),
pack{archive_sha256, manifest}, plus arm and optional build commands. An existing bin-ARM is replaced ONLY when it carries a valid
s1.build_identity.v1 for this arm with the SAME source tip (restaged atomically); one with a different tip, a malformed identity or
no BUILD-IDENTITY.json is refused untouched."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time

PACK_FORMAT = 1
BINARIES = {
    "s1": ["examples/qwen4_slowlink_gate", "examples/qwen4_greedy_ids", "examples/qwen4_kld", "examples/qwen4_parity", "daemon"],
    "beta": ["examples/qwen4_greedy_ids", "examples/qwen4_kld", "hipfire", "daemon"],
}
PINNED = {"beta": ["hipfire", "daemon"]}
SUFFIXES = (".hsaco", ".hash", ".index.json")
MODULE = re.compile(r"^[A-Za-z0-9_]+$")
ARCH_NAME = MODULE  # a single path component naming the pack architecture directory


def die(msg):
    sys.exit(f"stage_build: {msg}")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hex_arg(name, value, length):
    if not re.fullmatch(r"[0-9a-f]{%d}" % length, value or ""):
        die(f"--{name} must be {length} lowercase hex characters")
    return value


def module_of(name):
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None, None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_members(tar):
    """Validated members and the pack architecture: returns ([(member, name)], arch). manifest.json plus <arch>/<module>.<suffix>
    regular files only, all under ONE <arch> directory (a second architecture directory is refused)."""
    keep, seen, dirs, arch = [], set(), set(), None
    for m in tar.getmembers():
        name = m.name[2:] if m.name.startswith("./") else m.name
        name = name.rstrip("/")
        if m.isdir() and name in ("", "."):
            continue
        if not (m.isreg() or m.isdir()):
            die(f"unsafe tar member (not a regular file): {m.name!r}")
        if name.startswith("/") or ".." in name.split("/"):
            die(f"unsafe tar member path: {m.name!r}")
        if name in seen:
            die(f"duplicate tar member: {m.name!r}")
        seen.add(name)
        if m.isdir():
            if not ARCH_NAME.match(name):
                die(f"pack directory is not a single <arch> component: {m.name!r}")
            dirs.add(name)
            continue
        if name == "manifest.json":
            keep.append((m, name))
            continue
        parts = name.split("/")
        if len(parts) != 2 or not ARCH_NAME.match(parts[0]):
            die(f"pack member outside manifest.json and <arch>/: {m.name!r}")
        if arch is None:
            arch = parts[0]
        elif parts[0] != arch:
            die(f"pack mixes architecture directories {arch!r} and {parts[0]!r}: {m.name!r}")
        module, suffix = module_of(parts[1])
        if module is None or not MODULE.match(module):
            die(f"pack member is not <module>.{{hsaco,hash,index.json}}: {m.name!r}")
        keep.append((m, name))
    if arch is None:
        die("pack has no <arch>/<module> members")
    stray = sorted(dirs - {arch})
    if stray:
        die(f"pack has directories {stray} beside architecture {arch!r}")
    return keep, arch


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--arm", required=True, choices=("s1", "beta"))
    ap.add_argument("--source-base", required=True)
    ap.add_argument("--source-tip", required=True)
    ap.add_argument("--source-tree", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--kernel-pack", required=True)
    ap.add_argument("--build-command", action="append", default=[], help="recorded verbatim (optional, repeatable)")
    a = ap.parse_args(argv[1:])
    base = hex_arg("source-base", a.source_base, 40)
    tip = hex_arg("source-tip", a.source_tip, 40)
    tree = hex_arg("source-tree", a.source_tree, 40)
    if not hasattr(tarfile, "data_filter"):
        die("this Python's tarfile has no `data` extraction filter (need >= 3.12, or 3.11.4+)")

    stage = os.path.abspath(a.stage)
    final = os.path.join(stage, f"bin-{a.arm}")
    sealed = os.path.join(final, "BUILD-IDENTITY.json")
    if os.path.lexists(final):
        # Never delete unknown contents: an existing bin-ARM must carry a valid identity for the SAME source tip.
        if not os.path.isdir(final) or os.path.islink(final) or not os.path.isfile(sealed):
            die(f"{final} exists without a BUILD-IDENTITY.json: refusing to rename or delete unknown contents")
        try:
            prior = load_json(sealed)
            prior_tip = prior["source"]["tip"]
            ok = prior.get("schema") == "s1.build_identity.v1" and prior.get("arm") == a.arm
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            die(f"{sealed} is unreadable or malformed: refusing to overwrite")
        if not ok:
            die(f"{sealed} is not an s1.build_identity.v1 identity for arm {a.arm}: refusing to overwrite")
        if prior_tip != tip:
            die(f"{final} is already sealed for source tip {prior_tip}; refusing to overwrite it with {tip}")

    # 1. binaries: exist, regular, executable
    rel = os.path.join(os.path.abspath(a.target_dir), "release")
    sources = []
    for item in BINARIES[a.arm]:
        path = os.path.join(rel, item)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            die(f"missing or non-executable release binary: {path}")
        sources.append((os.path.basename(item), path))
    names = [n for n, _ in sources]
    if len(set(names)) != len(names):
        die("binary basenames collide")

    # 2. pack: validate and extract into scratch inside STAGE (never /tmp: RAM-backed)
    pack = os.path.abspath(a.kernel_pack)
    if not os.path.isfile(pack):
        die(f"kernel pack not found: {pack}")
    scratch = os.path.join(stage, f".stage-build-{a.arm}-{os.getpid()}")
    if os.path.exists(scratch):
        die(f"scratch exists: {scratch}")
    os.makedirs(scratch)
    try:
        with tarfile.open(pack, "r:*") as tar:
            members, arch = safe_members(tar)
            if not any(n == "manifest.json" for _, n in members):
                die("pack has no manifest.json")
            for m, _ in members:
                tar.extract(m, scratch, filter="data")
        manifest = load_json(os.path.join(scratch, "manifest.json"))
        if not isinstance(manifest, dict):
            die("pack manifest.json is not an object")
        if manifest.get("commit") != tip:
            die(f"pack manifest.commit {manifest.get('commit')!r} != source tip {tip}")
        if manifest.get("arch") != arch:
            die(f"pack manifest.arch {manifest.get('arch')!r} != pack member architecture {arch!r}")
        if manifest.get("format") != PACK_FORMAT:
            die(f"pack manifest.format {manifest.get('format')!r} != {PACK_FORMAT}")
        arch_dir = os.path.join(scratch, arch)
        if not os.path.isdir(arch_dir):
            die(f"pack has no {arch}/ directory")
        files = sorted(os.listdir(arch_dir))
        modules = sorted({module_of(f)[0] for f in files})
        if len(modules) != manifest.get("modules"):
            die(f"pack holds {len(modules)} modules; manifest.modules = {manifest.get('modules')}")
        code_hashes, source_hashes = {}, {}
        for mod in modules:
            for suffix in SUFFIXES:
                if mod + suffix not in files:
                    die(f"module {mod} missing {mod}{suffix}")
            index = load_json(os.path.join(arch_dir, mod + ".index.json"))
            if not isinstance(index, dict):
                die(f"{mod}.index.json is not an object")
            if index.get("module") != mod or index.get("arch") != arch:
                die(f"{mod}.index.json names module {index.get('module')!r} arch {index.get('arch')!r} (pack arch {arch!r})")
            actual = sha_file(os.path.join(arch_dir, mod + ".hsaco"))
            if index.get("object_sha256") != actual:
                die(f"{mod}: copied object sha256 {actual} != index.object_sha256 {index.get('object_sha256')}")
            code_hashes[mod] = actual
            source_hashes[mod] = index["source_sha256"]
        archive_sha = sha_file(pack)

        # 3. stage the arm directory beside the final one, then swap it in
        building = os.path.join(stage, f".bin-{a.arm}.new-{os.getpid()}")
        os.makedirs(os.path.join(building, "kernels", "compiled"))
        binary_sha = {}
        for name, path in sources:
            dest = os.path.join(building, name)
            shutil.copyfile(path, dest)
            os.chmod(dest, 0o755)
            binary_sha[name] = sha_file(dest)
            if binary_sha[name] != sha_file(path):
                die(f"{name}: copy differs from source")
        shutil.copytree(arch_dir, os.path.join(building, "kernels", "compiled", arch))
        if a.arm in PINNED:
            pins = {n: binary_sha[n] for n in PINNED[a.arm]}
            with open(os.path.join(building, "pins.json"), "w") as f:
                json.dump(pins, f, indent=1, sort_keys=True)
                f.write("\n")
        identity = {
            "schema": "s1.build_identity.v1",
            "arm": a.arm,
            "source": {"base": base, "tip": tip, "tree": tree},
            "kernel_arch": arch,
            "binary_sha256": binary_sha,
            "kernel_code_object_hashes": code_hashes,
            "kernel_source_hashes": source_hashes,
            "pack": {"archive_sha256": archive_sha, "archive": os.path.basename(pack), "manifest": manifest},
            "staged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build": {"target_dir": os.path.abspath(a.target_dir), "commands": a.build_command},
        }
        with open(os.path.join(building, "BUILD-IDENTITY.json"), "w") as f:
            json.dump(identity, f, indent=1)
            f.write("\n")
        old = None
        if os.path.exists(final):
            old = os.path.join(stage, f".bin-{a.arm}.old-{os.getpid()}")
            os.rename(final, old)
        os.rename(building, final)
        if old:
            shutil.rmtree(old)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        leftover = os.path.join(stage, f".bin-{a.arm}.new-{os.getpid()}")
        if os.path.exists(leftover):
            shutil.rmtree(leftover, ignore_errors=True)
    print(json.dumps({"staged": final, "binaries": binary_sha, "kernel_modules": len(code_hashes), "pack_sha256": archive_sha}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
