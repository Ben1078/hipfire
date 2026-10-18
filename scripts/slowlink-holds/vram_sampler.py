#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vram_sampler.py DEVICE_DIR CELL_DIR [INTERVAL_MS]

Independent EXTERNAL card-wide VRAM sampler of one hold cell. It reads exactly two sysfs files of the card and nothing else (no
/proc, no smaps, no HIP, no process attach): DEVICE_DIR/mem_info_vram_used and DEVICE_DIR/mem_info_vram_total
(DEVICE_DIR = /sys/bus/pci/devices/<BDF>). INTERVAL_MS is the REQUESTED period (default 20); the achieved spacing is recorded.

It is started BEFORE the GPU command and stopped AFTER the command and the VRAM drain (hold-lib.sh `cell`). Files written to CELL_DIR:
  vram.csv            t_mono_ns,used,total      one row per successful sample; t_mono_ns = CLOCK_MONOTONIC (time.monotonic_ns) taken
                                                immediately after the two reads; used/total are the raw sysfs byte counts
  vram.ready          created after the FIRST valid sample is flushed (the cell waits for it before it launches the GPU command)
  vram-events.jsonl   one line per mark, {"event", "t_mono_ns"}: SIGUSR1 = command_start, SIGUSR2 = command_end
  vram-summary.json   written ONCE at exit (SIGTERM/SIGINT): "finished": true, requested_interval_ms, samples, errors{count,first},
                      min_free_bytes (= total - used), min_used/max_used_bytes, min/max_total_bytes, max_gap_ns, median_gap_ns,
                      overruns, the marks, the monotonic start/stop. A killed sampler (SIGKILL) leaves no summary: the checker then
                      reports the telemetry INCOMPLETE.
A read or parse error is counted and recorded (first 20 kept), never replaced by a value and never zero-filled; the file is reopened.
After SIGTERM one last sample is taken, so the final row is later than the command_end mark. Exit 0 on a clean stop, 2 when a sysfs file
cannot be opened at start (no vram.ready is written)."""
import json
import os
import signal
import statistics
import sys
import time

MAX_ERRORS_KEPT = 20


class Sampler:
    def __init__(self, dev, out, interval_ms):
        self.dev, self.out, self.interval_ms = dev, out, interval_ms
        self.interval_ns = int(interval_ms * 1_000_000)
        self.stop = False
        self.events = []
        self.errors = {"count": 0, "first": []}
        self.fds = {}
        self.times = []
        self.min_free = self.max_used = self.min_used = self.min_total = self.max_total = None
        self.samples = 0
        self.overruns = 0
        self.ev_file = open(os.path.join(out, "vram-events.jsonl"), "a", buffering=1)

    def error(self, what):
        self.errors["count"] += 1
        if len(self.errors["first"]) < MAX_ERRORS_KEPT:
            self.errors["first"].append({"t_mono_ns": time.monotonic_ns(), "error": what})

    def open_one(self, name):
        try:
            self.fds[name] = os.open(os.path.join(self.dev, name), os.O_RDONLY)
            return True
        except OSError as e:
            self.error(f"open {name}: {e}")
            return False

    def read_one(self, name):
        if name not in self.fds and not self.open_one(name):
            return None
        try:
            return int(os.pread(self.fds[name], 64, 0).decode().strip())
        except (OSError, ValueError) as e:
            self.error(f"read {name}: {e}")
            try:
                os.close(self.fds.pop(name))
            except (KeyError, OSError):
                pass
            return None

    def mark(self, name):
        t = time.monotonic_ns()
        self.events.append({"event": name, "t_mono_ns": t})
        self.ev_file.write(json.dumps({"event": name, "t_mono_ns": t}) + "\n")

    def sample(self, csv):
        used = self.read_one("mem_info_vram_used")
        total = self.read_one("mem_info_vram_total")
        t = time.monotonic_ns()
        if used is None or total is None:
            return False
        csv.write(f"{t},{used},{total}\n")
        self.samples += 1
        self.times.append(t)
        free = total - used
        self.min_free = free if self.min_free is None else min(self.min_free, free)
        self.max_used = used if self.max_used is None else max(self.max_used, used)
        self.min_used = used if self.min_used is None else min(self.min_used, used)
        self.min_total = total if self.min_total is None else min(self.min_total, total)
        self.max_total = total if self.max_total is None else max(self.max_total, total)
        return True

    def close(self):
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()
        self.ev_file.close()

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGUSR1, lambda *_: self.mark("command_start"))
        signal.signal(signal.SIGUSR2, lambda *_: self.mark("command_end"))
        started = time.monotonic_ns()
        wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ready = False
        with open(os.path.join(self.out, "vram.csv"), "w") as csv:
            csv.write("t_mono_ns,used,total\n")
            nxt = started
            while not self.stop:
                if self.sample(csv) and not ready:
                    csv.flush()
                    open(os.path.join(self.out, "vram.ready"), "w").close()
                    ready = True
                nxt += self.interval_ns
                now = time.monotonic_ns()
                if nxt < now:
                    self.overruns += 1
                    nxt = now
                else:
                    time.sleep((nxt - now) / 1e9)
            self.sample(csv)  # one sample strictly after the stop request (and therefore after command_end)
        stopped = time.monotonic_ns()
        gaps = [b - a for a, b in zip(self.times, self.times[1:])]
        summary = {
            "schema": "hipfire.slowlink.vram_sampler.v1", "finished": True, "device_dir": self.dev,
            "files": ["mem_info_vram_used", "mem_info_vram_total"], "requested_interval_ms": self.interval_ms,
            "started_utc": wall, "started_mono_ns": started, "stopped_mono_ns": stopped,
            "samples": self.samples, "errors": self.errors, "overruns": self.overruns,
            "min_free_bytes": self.min_free, "max_used_bytes": self.max_used, "min_used_bytes": self.min_used,
            "min_total_bytes": self.min_total, "max_total_bytes": self.max_total,
            "max_gap_ns": max(gaps) if gaps else None, "median_gap_ns": int(statistics.median(gaps)) if gaps else None,
            "events": self.events,
        }
        tmp = os.path.join(self.out, "vram-summary.json.tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=1)
        os.replace(tmp, os.path.join(self.out, "vram-summary.json"))
        return 0


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    dev, out = argv[1], argv[2]
    interval_ms = float(argv[3]) if len(argv) == 4 else 20.0
    if interval_ms <= 0:
        sys.exit("INTERVAL_MS must be positive")
    os.makedirs(out, exist_ok=True)
    s = Sampler(dev, out, interval_ms)
    try:
        for name in ("mem_info_vram_used", "mem_info_vram_total"):
            if not s.open_one(name):
                with open(os.path.join(out, "vram-summary.json"), "w") as f:
                    json.dump({"schema": "hipfire.slowlink.vram_sampler.v1", "finished": True, "samples": 0, "errors": s.errors,
                               "requested_interval_ms": interval_ms}, f, indent=1)
                return 2
        return s.run()
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
