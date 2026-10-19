#!/usr/bin/env python3
"""Create a SYMLINK-ONLY executable stage from artifact roots, plus a read-only identity manifest.

Local and remote use the same file (rsync the artifacts first, then run this on the host that will run the cells):

  gap_stage.py --out STAGE_ROOT --cli ARTIFACT/hipfire --daemon ARTIFACT/daemon \
      --kernels ARTIFACT/kernels/compiled/gfx1100 --pack-manifest ARTIFACT/<pack>.manifest.json \
      [--pack-tar-sha256 HEX] [--probe NAME=ARTIFACT/example-binary ...]

Creates (STAGE_ROOT must not exist; its parent must):
  STAGE_ROOT/bin/hipfire, bin/daemon, bin/<probe NAME>   symlinks to the canonical artifact files
  STAGE_ROOT/bin/kernels/compiled/gfx1100                symlink to the canonical pack arch dir
  STAGE_ROOT/STAGE-MANIFEST.json                         0444: per-entry readlink, canonical target, ino, size,
                                                         mtime_ns, sha256, md5; pack tree sha256 + per-file stats;
                                                         pack manifest source hash (sha256/md5, commit, modules)
No payload or source is copied and the model is never hashed. The stage dirs are made read-only afterwards.
The cells (gap-cell.sh) and the plan generator (gap_plan.py) re-verify every one of these fields.
"""
import argparse
import json
import os
import re
import socket
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_lib as L  # noqa: E402

RESERVED = {"hipfire", "daemon", "kernels"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="stage root; must not exist")
    ap.add_argument("--cli", required=True)
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--kernels", required=True, help="pack arch dir holding the *.hsaco/.hash/.index.json files")
    ap.add_argument("--pack-manifest", required=True, help="the pack's manifest json (its sha256 is pinned)")
    ap.add_argument("--pack-tar-sha256", default=None, help="sha256 of the pack tarball, recorded verbatim")
    ap.add_argument("--probe", action="append", default=[], metavar="NAME=PATH",
                    help="optional extra executable (e.g. native quality probe) to symlink into bin/")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if os.path.lexists(out):
        L.die(f"refusing existing stage {out}")
    if not os.path.isdir(os.path.dirname(out)):
        L.die(f"parent of {out} does not exist")

    exes = {"hipfire": a.cli, "daemon": a.daemon}
    for kv in a.probe:
        name, sep, path = kv.partition("=")
        if not sep or not re.match(r"^[A-Za-z0-9_.-]+$", name) or name in RESERVED or name in exes:
            L.die(f"bad --probe {kv!r}")
        exes[name] = path
    canon = {}
    for name, path in exes.items():
        c = os.path.realpath(path)
        st = os.stat(c)
        if not stat.S_ISREG(st.st_mode) or not os.access(c, os.X_OK):
            L.die(f"{name}: {path} is not an executable regular file")
        canon[name] = c
    kdir = os.path.realpath(a.kernels)
    if not os.path.isdir(kdir):
        L.die(f"--kernels {a.kernels} is not a directory")
    pm_path = os.path.realpath(a.pack_manifest)
    with open(pm_path) as f:
        pm = json.load(f)
    if pm.get("arch") != L.ARCH:
        L.die(f"pack manifest arch {pm.get('arch')!r} != {L.ARCH}")

    os.mkdir(out)
    kroot = os.path.join(out, "bin", "kernels", "compiled")
    os.makedirs(kroot)
    for name, c in canon.items():
        os.symlink(c, os.path.join(out, "bin", name))
    os.symlink(kdir, os.path.join(kroot, L.ARCH))

    cap = L.stage_capture(os.path.join(out, "bin"), hashes=True)
    if pm.get("modules") != cap["kernels"]["hsaco_count"]:
        L.die(f"pack manifest modules={pm.get('modules')} but the pack dir holds {cap['kernels']['hsaco_count']} .hsaco")
    p_sha, p_md5, p_size = L.hash_file(pm_path)
    pst = os.stat(pm_path)
    manifest = {
        "schema": L.SCHEMA_STAGE,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "stage_root": out,
        "sources": canon,
        "capture": cap,
        "pack_manifest": {
            "canonical": pm_path, "sha256": p_sha, "md5": p_md5, "size": p_size, "ino": pst.st_ino,
            "mtime_ns": pst.st_mtime_ns, "commit": pm.get("commit"), "tag": pm.get("tag"),
            "modules": pm.get("modules"), "kernel_cache_abi": pm.get("kernel_cache_abi"),
            "extra_flags": pm.get("extra_flags"), "code_object_version": pm.get("code_object_version"),
        },
        "pack_tar_sha256": a.pack_tar_sha256,
        "model_hashed": False,
    }
    mpath = os.path.join(out, "STAGE-MANIFEST.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    os.chmod(mpath, 0o444)
    for d in (kroot, os.path.dirname(kroot), os.path.join(out, "bin"), out):
        os.chmod(d, 0o555)
    print(json.dumps({"stage_root": out, "bin": os.path.join(out, "bin"), "manifest": mpath,
                      "kernels_tree_sha256": cap["kernels"]["tree_sha256"],
                      "hipfire_sha256": cap["entries"]["hipfire"]["sha256"],
                      "daemon_sha256": cap["entries"]["daemon"]["sha256"],
                      "pack_manifest_sha256": p_sha}, indent=1))


if __name__ == "__main__":
    main()
