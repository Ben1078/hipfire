#!/usr/bin/env bash
# hold-lib.sh (S1 slow-link prefill, slice V): shared by hold1-correct.sh, hold2-timing.sh and hold3-candidate.sh.
# NOT submitted by this seat: Main owns submission through hipx ~/qcal/locks/fabric/fabric-run.sh (card xtx,
# GPU-43390a851e296ee5). These are the frozen INNER scripts; they refuse to run outside a fabric job on exactly the XTX.
#
# Staged BEFORE the hold (the hold never builds, copies a model or purges a cache). STAGE (default $HOLDS/stage):
#   bin-s1/    candidate set (stage_build.py): qwen4_slowlink_gate qwen4_greedy_ids qwen4_kld qwen4_parity daemon kernels/compiled/gfx1100 BUILD-IDENTITY.json
#   bin-beta/  immutable beta reference set built from landed beta ff799626eb3dddf8ce827a90d88c990fc0eb4aa2: qwen4_greedy_ids qwen4_kld
#              hipfire daemon (the beta CLI+daemon of the hold-2 anchor) pins.json (sha256 of hipfire and daemon) BUILD-IDENTITY.json
#   fixtures/  gate-fixture.json (hipfire.qwen4.slowlink_fixture.v1; perf_repeats 1) prompts.short.jsonl prompts.full.jsonl
#   manifests/ manifest-hold1.json manifest-hold2.json manifest-hold3.json (gate manifest: source base+tip, binary sha256, model,
#              PLE source/format/sidecars, kernel code-object hashes, host/device/link, memory_ledger)
#   ledger-hold{1,2,3}.json  the FINAL numeric ledger (ledger-template.json; sealed by S1Integrate with the final tip and release binary
#              SHA): the whole exclusive XTX is reserved (--mem-gb 24); the device ledger is an ACCOUNTING FIT (committed core + the
#              sequential diagnostic + the fixed driver allowance must fit 24 GiB); the 2 GiB session-cache budget and the pending
#              working-set ceiling are recorded but neither committed nor added; host = host_owned + the fixed host allowance, ceilinged.
#              The allowances are declared, not measured. `--print-memory` refuses when the file is missing or not final.
# Fabric wrapper shape (Main submits):
#   fabric-run.sh --class correct|timing --card xtx --mem-gb DEV --hostmem-gb HOST --hostmem-lock --max-secs 600|1800
#                 --priority P0|P1 --owner Main.SlowLinkS1 -- <this hold's script>
#
# Every cell: one fresh process group, own HOME/lock/kernel cache, isolated env (HIPFIRE_*/HIP_*/HSA_*/ROCR_* stripped, then
# pinned), raw stdout/stderr/rc preserved, VRAM and process group drained before the next cell. A failed cell stops every cell
# that depends on it (recorded as skipped); a cell that cannot start inside the remaining budget is INCOMPLETE, never reduced.
set -uo pipefail

HOLDS=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
STAGE=${S1_STAGE:-$HOLDS/stage}
CARD=xtx
UUID=GPU-43390a851e296ee5
BDF=0000:67:00.0
MODEL=${MODEL:-/home/kaden/pm-wave/fngptq/models/qwen3.8-flash-next-gptq3.mq4}
MODEL_SIZE=125288544792
REFS=${REFS:-/home/kaden/pm-wave/consumer24/gates/refs}
CODE24_REF_MD5=56d93a4931631e306244e95ea64aecfc
CODE24_CORPUS_MD5=6145e25c0858e31e08527c1e5278c074
PLE_FORMAT=${PLE_FORMAT:-q4}
KV_MODE=${KV_MODE:-q8}
STATE_QUANT=${STATE_QUANT:-q8}
MAX_SEQ=${MAX_SEQ:-16640}
IDLE_MAX=$((2 << 30))
KILL_GRACE=15
FINALIZE_S=45

# hold_memory N: `--print-memory` handler (call before anything else). Main's ruling: the hold reserves the WHOLE exclusive XTX
# (device 24 GiB, always) and the ledger is an ACCOUNTING FIT, not a declaration: owned device bytes (committed core + the sequential
# diagnostic) + the fixed driver allowance must fit 24 GiB, asserted here. The 2 GiB session-cache budget and the pending
# working-set ceiling are recorded in the ledger but never added. Host = host_owned + the fixed host allowance, rounded UP to whole GiB.
# Reads ONLY a FINAL numeric ledger ($STAGE/ledger-holdN.json, template ledger-template.json); refuses otherwise.
hold_memory() {
  local f=$STAGE/ledger-hold$1.json
  [ -r "$f" ] || { echo "no final memory ledger $f (S1Integrate provides host_owned/census_basis/provenance): refusing" >&2; exit 2; }
  python3 - "$f" <<'PY' || exit 2
import json, math, sys
led = json.load(open(sys.argv[1]))
need = ["device_owned_bytes", "device_weights_bytes", "s1_tile_census_bytes", "s1_extras_bytes", "sequential_diag_bytes", "safety_margin_bytes", "device_allowance_bytes", "card_bytes", "host_owned_bytes", "host_allowance_bytes", "census_basis"]
if not led.get("census_final") or any(led.get(k) in (None, "") for k in need):
    sys.exit("ledger is not final (census_final false or a null term)")
parts = [led["device_weights_bytes"], led["s1_tile_census_bytes"], led["s1_extras_bytes"], led["sequential_diag_bytes"]]
if sum(parts) != led["device_owned_bytes"] or led["safety_margin_bytes"] is None:
    sys.exit(f"ledger split {sum(parts)} != device_owned_bytes {led['device_owned_bytes']}")
fit = led["device_owned_bytes"] + led["device_allowance_bytes"]  # safety_margin_bytes: admission margin, no allocation, not added here
if fit > led["card_bytes"]:
    sys.exit(f"accounting does not fit the card: {fit} > {led['card_bytes']}")
# The reservation may exceed the S1 census (sequential arm, e.g. the shipped-G2 beta baseline): host_reservation_owned_bytes when the
# ledger declares it, else host_owned_bytes. host_owned_bytes itself stays the S1 census for the strict first-cell driver check.
reserved = led.get("host_reservation_owned_bytes", led["host_owned_bytes"])
if not isinstance(reserved, int) or reserved < led["host_owned_bytes"]:
    sys.exit(f"host_reservation_owned_bytes {reserved!r} is not an integer >= host_owned_bytes {led['host_owned_bytes']}")
print(24, math.ceil((reserved + led["host_allowance_bytes"]) / 2**30))
PY
  exit 0
}

# hold_init NAME MAXSECS: guards, OUT, identity.
hold_init() {
  HOLD=$1; MAXS=$2
  T0=$(date +%s); BUDGET=$((MAXS - FINALIZE_S))
  : "${OUT:=$HOLDS/runs/$HOLD-$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$OUT" || exit 2
  [ -e "$OUT/hold.jsonl" ] && { echo "FATAL: $OUT already used" >&2; exit 2; }
  CELLS_RAN=(); CELLS_JSON="$OUT/cells.jsonl"; : > "$CELLS_JSON"; FAILED=(); BLOCKED=""
  [ -n "${FABRIC_JOB:-}" ] && [ -n "${FABRIC_CARDS:-}" ] || die "not inside a fabric job (FABRIC_JOB/FABRIC_CARDS unset)"
  case ",$FABRIC_CARDS," in *,xtx,*|*,"$UUID",*) ;; *) die "FABRIC_CARDS=$FABRIC_CARDS does not name the XTX" ;; esac
  [ "${FABRIC_CARDS//[^,]/}" = "" ] || die "FABRIC_CARDS=$FABRIC_CARDS names more than one card"
  [ "$(cat /sys/bus/pci/devices/$BDF/unique_id 2>/dev/null)" = "${UUID#GPU-}" ] || die "pci $BDF unique_id != $UUID"
  [ -r "$MODEL" ] && [ "$(stat -c %s "$MODEL")" = "$MODEL_SIZE" ] || die "model $MODEL missing or not $MODEL_SIZE bytes"
  [ -d "$STAGE" ] || die "stage dir $STAGE missing (inputs and binaries are staged before the hold)"
  for t in qwen4_slowlink_gate qwen4_greedy_ids qwen4_kld; do [ -x "$STAGE/bin-s1/$t" ] || die "$STAGE/bin-s1/$t missing"; done
  jlog hold_start "hold=$HOLD" "out=$OUT" "fabric_job=$FABRIC_JOB" "budget_s=$BUDGET"
  identity
}

left() { echo $((BUDGET - ($(date +%s) - T0))); }
vram_used() { cat "/sys/bus/pci/devices/$BDF/mem_info_vram_used" 2>/dev/null || echo 99999999999; }
await_idle() { local end=$((SECONDS + $1)); while :; do (( $(vram_used) < IDLE_MAX )) && return 0; (( SECONDS >= end )) && return 1; sleep 2; done; }

jlog() { # EVENT [key=value ...]
  python3 - "$OUT/hold.jsonl" "$@" <<'PY'
import json, sys, time
path, ev, *kv = sys.argv[1:]
o = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": ev}
for a in kv:
    k, _, v = a.partition("=")
    try: o[k] = json.loads(v)
    except ValueError: o[k] = v
with open(path, "a") as f: f.write(json.dumps(o) + "\n")
print(json.dumps(o), file=sys.stderr)
PY
}
die() { echo "FATAL: $1" >&2; [ -n "${OUT:-}" ] && { echo "FATAL: $1" > "$OUT/FATAL.txt"; jlog fatal "msg=$1" 2>/dev/null; finish_summary; }; exit "${2:-2}"; }

# identity.json: source/binary/model/host/link topology walk (upstream bottleneck, not the card's own x16 label).
identity() {
  python3 - "$OUT/identity.json" "$STAGE" "$MODEL" "$BDF" "$UUID" <<'PY'
import hashlib, json, os, sys, time
out, stage, model, bdf, uuid = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
def speed(v):
    try: return float(v.split()[0])
    except Exception: return None
walk = []
path = os.path.realpath("/sys/bus/pci/devices/" + bdf)
parts = path.split("/")
cur = "/sys/devices/" + parts[3]
for part in parts[4:]:
    cur += "/" + part
    walk.append({"bdf": part, "speed": rd(cur + "/current_link_speed"), "width": rd(cur + "/current_link_width"),
                 "max_speed": rd(cur + "/max_link_speed"), "max_width": rd(cur + "/max_link_width")})
def gbps(n):  # PCIe payload GB/s per link: GT/s * width * encoding / 8
    s, w = speed(n["speed"] or ""), int(n["width"] or 0)
    if not s or not w: return None
    enc = 128 / 130 if s >= 8 else 0.8
    return s * w * enc / 8
rates = [(gbps(n), n["bdf"]) for n in walk if gbps(n)]
bott_gbps, bott_bdf = min(rates) if rates else (None, None)
ident = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename, "uuid": uuid, "card_bdf": bdf,
         "pci_walk": walk, "upstream_bottleneck_gbps": bott_gbps, "upstream_bottleneck_bdf": bott_bdf,
         "note": "topology walk of every bridge from root port to the card; card-own link label is not the verdict",
         "model": {"path": model, "bytes": os.stat(model).st_size, "mtime": os.stat(model).st_mtime,
                   "sha256": "pinned by the manifest; 125 GB, not recomputed in the hold"},
         "meminfo_start": {l.split(":")[0]: l.split(":")[1].strip() for l in open("/proc/meminfo")}}
bins = {}
for d in ("bin-s1", "bin-beta"):
    p = os.path.join(stage, d)
    if os.path.isdir(p):
        bins[d] = {n: sha(os.path.join(p, n)) for n in sorted(os.listdir(p)) if os.path.isfile(os.path.join(p, n))}
ident["binaries_sha256"] = bins
fx = os.path.join(stage, "fixtures")
ident["fixtures_sha256"] = {n: sha(os.path.join(fx, n)) for n in sorted(os.listdir(fx))} if os.path.isdir(fx) else {}
json.dump(ident, open(out, "w"), indent=1)
PY
  [ -s "$OUT/identity.json" ] || die "identity.json not written"
}

# cell_env DIR [KEY=VAL ...]: sets E (the env prefix) for one cell.
cell_env() {
  local d=$1; shift; local v
  mkdir -p "$d/home/.hipfire" "$d/lockdir" "$d/kcache"
  printf '[memory]\nmax_seq = %s\nple_format = "%s"\n' "$MAX_SEQ" "$PLE_FORMAT" > "$d/home/.hipfire/config.toml"
  ln -s /home/kaden/.hipfire/models "$d/home/.hipfire/models" 2>/dev/null
  E=(env)
  for v in $(compgen -e); do
    case "$v" in FABRIC_*|HIPFIRE_FABRIC_ADMISSION) ;; HIPFIRE_*|HIP_*|ROCR_*|ROC_*|HSA_*|GPU_*|AMD_*) E+=(-u "$v");; esac
  done
  E+=(HOME="$d/home" HIPFIRE_HOME="$d/home/.hipfire" HIPFIRE_LOCK_DIR="$d/lockdir" HIPFIRE_KERNEL_CACHE="${KCACHE:-$d/kcache}"
      HIPFIRE_NO_REGISTRY_FETCH=1 HIPFIRE_PLE_FORMAT="$PLE_FORMAT" HIPFIRE_DEVICES="$UUID" ROCR_VISIBLE_DEVICES="$UUID"
      HIPFIRE_KV_MODE="$KV_MODE" HIPFIRE_STATE_QUANT="$STATE_QUANT" HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_CHAT_CURRENT_DATE=2026-10-17 "$@")
}

# cell NAME CAP_S MIN_S [KEY=VAL ...] -- cmd args...   (dependencies: set BLOCKED to a reason to skip all later cells)
cell() {
  local name=$1 cap=$2 min=$3; shift 3
  local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  local d=$OUT/$name rem; rem=$(left)
  mkdir -p "$d"
  if [ -n "$BLOCKED" ]; then
    echo "skipped: $BLOCKED" > "$d/status.txt"; cell_row "$name" skipped "$BLOCKED" 0 0 ""; return 1
  fi
  if (( rem < min )); then
    echo "incomplete: only ${rem}s left, needs >= ${min}s" > "$d/status.txt"
    cell_row "$name" incomplete "needs ${min}s, ${rem}s left" 0 0 ""; BLOCKED="$name could not start (INCOMPLETE)"; return 1
  fi
  (( cap > rem )) && cap=$rem
  (( $(vram_used) < IDLE_MAX )) || die "$name: vram_used=$(vram_used) before start"
  cell_env "$d" "${extra[@]}"
  printf '%q ' "${E[@]}" setsid -w timeout -k $KILL_GRACE "$cap" "$@" > "$d/command.txt"; echo >> "$d/command.txt"
  cp /proc/meminfo "$d/meminfo-before.txt"
  printf '{"vram_used": %s, "gtt_used": %s}\n' "$(vram_used)" "$(cat /sys/bus/pci/devices/$BDF/mem_info_gtt_used 2>/dev/null || echo 0)" > "$d/baseline.json"
  # Raw per-sample components (never summed): sysfs vram/gtt and, per process of this cell, VmRSS/VmPin/RssAnon/RssFile plus smaps
  # classes (regular-file clean/total, kfd/dri, zero, anon). check_residual.py applies the declared method.
  python3 "$HOLDS/sample_mem.py" "$1" "$BDF" "$d/mem.csv" 2 > /dev/null 2>&1 &
  local samp=$! w0=$SECONDS
  setsid -w "${E[@]}" timeout -k $KILL_GRACE "$cap" "$@" > "$d/stdout.txt" 2> "$d/stderr.txt" < /dev/null &
  local pid=$!; wait "$pid"; local rc=$?
  kill "$samp" 2>/dev/null; wait "$samp" 2>/dev/null
  echo "$rc" > "$d/rc.txt"; cp /proc/meminfo "$d/meminfo-after.txt"
  local leftover; leftover=$(pgrep -g "$pid" 2>/dev/null | tr '\n' ' ')
  [ -z "$leftover" ] || { kill -KILL -- "-$pid" 2>/dev/null; sleep 2; }
  local drained=true; await_idle 90 || drained=false
  local status=pass
  [ "$rc" = 0 ] || status=fail
  { [ -z "$leftover" ] && [ "$drained" = true ]; } || { status=fail; echo "lifecycle: leftover='$leftover' drained=$drained" >> "$d/status.txt"; }
  cell_row "$name" "$status" "rc=$rc" "$rc" "$((SECONDS - w0))" "$leftover"
  jlog cell "name=$name" "status=$status" "rc=$rc" "seconds=$((SECONDS - w0))" "drained=$drained"
  if [ "$status" != pass ]; then FAILED+=("$name"); BLOCKED="$name failed (rc=$rc); dependent cells are not run"; return 1; fi
  return 0
}
cell_row() { python3 - "$CELLS_JSON" "$@" <<'PY'
import json, sys
p, name, status, why, rc, secs, left = sys.argv[1:]
open(p, "a").write(json.dumps({"name": name, "status": status, "why": why, "rc": int(rc), "seconds": int(secs), "leftover": left}) + "\n")
PY
}

# check NAME CMD...: a CPU verification step (never a GPU process); failure fails the hold but does not block later cells.
check() {
  local name=$1; shift; local d=$OUT/$name; mkdir -p "$d"
  if [ -n "$BLOCKED" ]; then echo "skipped: $BLOCKED" > "$d/status.txt"; cell_row "$name" skipped "$BLOCKED" 0 0 ""; return 1; fi
  if "$@" > "$d/stdout.txt" 2> "$d/stderr.txt"; then cell_row "$name" pass "" 0 0 ""; jlog check "name=$name" "status=pass"; return 0; fi
  cell_row "$name" fail "$(head -c 300 "$d/stderr.txt" | tr '\n' ' ')" 1 0 ""; jlog check "name=$name" "status=fail"; FAILED+=("$name"); return 1
}

# residual NAME CELL_DIR LEDGER_JSON: the fail-closed memory residual gate (check_residual.py) as a check; a failure blocks every
# later cell. Refuses (counts as fail) when the ledger is not final.
residual() {
  local name=$1 d=$2 led=$3
  check "$name" python3 "$HOLDS/check_residual.py" "$d" "$led" || { BLOCKED="$name failed: residual over the allowance or ledger not final"; return 1; }
}

# finish_summary: exit 0 only when every listed cell/check passed (any skipped/incomplete/failed row is not a pass).
finish_summary() {
  python3 - "$CELLS_JSON" "$OUT/hold-summary.json" "${HOLD:-?}" "${REQUIRED:-}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()] if sys.argv[1] else []
required = [r for r in sys.argv[4].split() if r]
seen = {r["name"]: r["status"] for r in rows}
missing = [n for n in required if n not in seen]
if any(r["status"] == "fail" for r in rows): status = "fail"
elif missing or any(r["status"] in ("skipped", "incomplete") for r in rows) or not required: status = "incomplete"
else: status = "pass"
json.dump({"hold": sys.argv[3], "status": status, "required": required, "missing": missing, "cells": rows},
          open(sys.argv[2], "w"), indent=1)
print("hold %s: %s" % (sys.argv[3], status))
sys.exit(0 if status == "pass" else 1)
PY
}
