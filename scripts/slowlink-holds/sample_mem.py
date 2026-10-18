#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""sample_mem.py EXE_SUBSTRING PCI_BDF OUT.csv [INTERVAL_S]

Per-sample memory components of one cell, written raw (never summed into a verdict here). Runs until killed. For every process
whose command line contains EXE_SUBSTRING (this sampler excluded) it reads /proc/PID/status (VmRSS, VmPin, RssAnon, RssFile) and
/proc/PID/smaps, classifying every mapping by its path:

  kfd_dri   /dev/kfd*, /dev/dri/*            (device / pinned-BO maps)
  zero      /dev/zero*                        (anonymous shared)
  anon      no path, [heap], [stack], [anon:*], memfd:*, (deleted)
  file      any other absolute path (regular file maps)

and records, per class, Rss and the CLEAN part (Shared_Clean + Private_Clean). `file_clean_kb` is the clean RSS of regular-file
mappings only, i.e. the page-cache-backed part that is reclaimable. Device sysfs counters (vram_used, gtt_used) are sampled with it.
All sizes in kB except the sysfs columns (bytes). INFORMATION ONLY: nothing gates on this 2 s file (the retired residual checker is gone);
the gating VRAM observation is the independent 20 ms vram_sampler.py read by check_headroom.py."""
import os
import sys
import time

COLS = ["t", "vram_used", "gtt_used", "proc_vmrss_kb", "proc_vmpin_kb", "proc_rssanon_kb", "proc_rssfile_kb",
        "file_rss_kb", "file_clean_kb", "kfd_dri_rss_kb", "zero_rss_kb", "anon_rss_kb", "mem_available_kb", "mlocked_kb"]


def classify(path):
    if path.startswith("/dev/kfd") or path.startswith("/dev/dri"):
        return "kfd_dri"
    if path.startswith("/dev/zero"):
        return "zero"
    if not path or path.startswith("[") or path.startswith("/memfd:") or path.startswith("memfd:") or "(deleted)" in path:
        return "anon"
    return "file" if path.startswith("/") else "anon"


def read_int(path):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return 0


def pids(pattern, me):
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            cmd = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if pattern in cmd and "sample_mem.py" not in cmd:
            out.append(int(name))
    return out


def status_fields(pid):
    f = {"VmRSS": 0, "VmPin": 0, "RssAnon": 0, "RssFile": 0}
    try:
        for line in open(f"/proc/{pid}/status"):
            key, _, rest = line.partition(":")
            if key in f:
                f[key] = int(rest.split()[0])
    except OSError:
        pass
    return f


def smaps(pid):
    acc = {c: [0, 0] for c in ("file", "kfd_dri", "zero", "anon")}
    cur = "anon"
    try:
        for line in open(f"/proc/{pid}/smaps"):
            head = line.split(None, 5)
            if len(head) >= 5 and "-" in head[0] and not line.startswith(("Rss", "Pss", "Shared", "Private", "Size")):
                cur = classify(head[5].strip() if len(head) == 6 else "")
            elif line.startswith("Rss:"):
                acc[cur][0] += int(line.split()[1])
            elif line.startswith(("Shared_Clean:", "Private_Clean:")):
                acc[cur][1] += int(line.split()[1])
    except OSError:
        pass
    return acc


def main(argv):
    if len(argv) not in (4, 5):
        sys.exit(__doc__)
    pattern, bdf, out = argv[1], argv[2], argv[3]
    interval = float(argv[4]) if len(argv) == 5 else 2.0
    base = f"/sys/bus/pci/devices/{bdf}/"
    me = os.getpid()
    with open(out, "w", buffering=1) as f:
        f.write(",".join(COLS) + "\n")
        while True:
            tot = {"VmRSS": 0, "VmPin": 0, "RssAnon": 0, "RssFile": 0}
            cls = {c: [0, 0] for c in ("file", "kfd_dri", "zero", "anon")}
            for pid in pids(pattern, me):
                for k, v in status_fields(pid).items():
                    tot[k] += v
                for c, (rss, clean) in smaps(pid).items():
                    cls[c][0] += rss
                    cls[c][1] += clean
            mem = {}
            try:
                for line in open("/proc/meminfo"):
                    k, _, v = line.partition(":")
                    if k in ("MemAvailable", "Mlocked"):
                        mem[k] = int(v.split()[0])
            except OSError:
                pass
            row = [int(time.time()), read_int(base + "mem_info_vram_used"), read_int(base + "mem_info_gtt_used"),
                   tot["VmRSS"], tot["VmPin"], tot["RssAnon"], tot["RssFile"],
                   cls["file"][0], cls["file"][1], cls["kfd_dri"][0], cls["zero"][0], cls["anon"][0],
                   mem.get("MemAvailable", 0), mem.get("Mlocked", 0)]
            f.write(",".join(str(x) for x in row) + "\n")
            time.sleep(interval)


if __name__ == "__main__":
    main(sys.argv)
