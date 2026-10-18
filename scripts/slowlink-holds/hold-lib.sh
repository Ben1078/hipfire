#!/usr/bin/env bash
# hold-lib.sh (S1 slow-link prefill, slice V): shared by hold1a-correct.sh, hold1b-correct.sh, hold2-timing.sh and hold3-candidate.sh.
# NOT submitted by this seat: Main owns submission through hipx ~/qcal/locks/fabric/fabric-run.sh (the declared target card in
# $STAGE/target.json). These are the frozen INNER scripts; they refuse to run outside a fabric job on exactly that declared target.
#
# Staged BEFORE the hold (the hold never builds, copies a model or purges a cache). STAGE (default $HOLDS/stage):
#   bin-s1/    candidate set (stage_build.py): qwen4_slowlink_gate qwen4_greedy_ids qwen4_kld qwen4_parity daemon kernels/compiled/<pack arch> BUILD-IDENTITY.json
#   bin-beta/  immutable beta reference set built from landed beta ff799626eb3dddf8ce827a90d88c990fc0eb4aa2: qwen4_greedy_ids qwen4_kld
#              hipfire daemon (the beta CLI+daemon of the hold-2 anchor) pins.json (sha256 of hipfire and daemon) BUILD-IDENTITY.json
#   fixtures/  gate-fixture.json (hipfire.qwen4.slowlink_fixture.v1; perf_repeats 1) prompts.short.jsonl prompts.full.jsonl
#   target.json  the DECLARED target (REQUIRED, no fallback; loaded by hold_init before any guard): {"schema": "s1.hold_target.v1",
#              "card": fabric card name, "uuid": "GPU-<hex>", "bdf": "<domain>:<bus>:<dev>.<fn>", "model": {"path": absolute path,
#              "size_bytes": positive integer}}. The card/UUID/BDF/model guards below are exact to this declaration; no target string is
#              compiled into these scripts.
#   manifests/ manifest-hold1.json manifest-hold2.json manifest-hold3.json (gate manifest: source base+tip, binary sha256, model,
#              PLE source/format/sidecars, kernel code-object hashes, host/device/link, memory_ledger)
#   ledger-hold{1,2,3}.json  the FINAL per-purpose numeric ledger (ledger-template.json, validated by ledger_profiles.py; copied by Main at the
#              seal; the harness/runtime commits and trees and the binary sha256 are recorded EXTERNALLY): INFORMATION about the
#              source-owned census (10 resident experts, aggregate route) of BOTH admitted purposes (profiles.product tile 768, profiles.
#              primitive_selftest tile 512) and the shared host reservation; it gates nothing. The whole
#              exclusive device is reserved at the ledger's explicit fabric_reservation_device_gib (the physical card_bytes is separate); the
#              product question of Hold 1 is the observed free-VRAM headroom of the PRODUCT cell (check_headroom.py: >= 512 MiB from the
#              external 20 ms sysfs sampler AND the in-process hipMemGetInfo points of a COMPLETED gate report, BOTH sources required).
#              The 2 GiB device session-cache budget (a nominal best-effort cap, not committed) is recorded, and the pending working-set ceiling
#              lies inside each purpose's 2 GiB bounded host snapshot cap, neither added to the device census; host = the sequential max host reservation + the
#              fixed host allowance, ceilinged. The allowances are declared, not measured. `--print-memory` refuses when the file is missing,
#              not final, flat or inconsistent.
#   fixtures/ also: prompts.short.beta.jsonl + beta-input-proof.json (the metadata-only beta input: prompts.short.jsonl minus ONLY resume_at,
#              beta_input.py); the manifest binds both input hashes and the identity proof.
# Fabric wrapper shape (Main submits):
#   fabric-run.sh --class correct|timing --card CARD --mem-gb DEV --hostmem-gb HOST --hostmem-lock --max-secs 600|1800
#                 --priority P0|P1 --owner Main.SlowLinkS1 -- <this hold's script>
#
# Every cell: one fresh process group, own HOME/lock/kernel cache, isolated env (HIPFIRE_*/HIP_*/HSA_*/ROCR_* stripped, then
# pinned), raw stdout/stderr/rc preserved, VRAM and process group drained before the next cell. A failed cell stops every cell
# that depends on it (recorded as skipped); a cell that cannot start inside the remaining budget is INCOMPLETE, never reduced.
# Per-cell evidence (collect_evidence.py, run after every started cell whether it passed, failed, timed out (rc 124/137) or failed to load):
# CELL/evidence.json lists the preserved raw files (command.txt, stdout.txt, stderr.txt, rc.txt, status.txt, mem.csv, meminfo-*.txt, the
# vram sampler files) with sizes and sha256, the gate report state and the gate's durable progress sidecar CELL/gate.json.progress.jsonl
# (schema hipfire.qwen4.slowlink_gate.progress.v1: one JSON object per line {schema, run_id, seq, event, utc, unix_ms, mono_ns, ...}, written with
# sync_data per record; torn tail / corrupt lines are reported, never interpreted). The gate report is written once, atomically, at the end of
# a run; a timeout, crash or load failure leaves the sidecar (and any gate.json.tmp-*) as the only partial evidence. The sidecar and
# evidence.json are FORENSIC ONLY: no cell status, check, headroom verdict or hold summary status ever reads them, and a partial sidecar can
# never be a pass. Nothing is removed or rewritten, and there is no automatic rerun.
#
# LAB-LINK POLICY (XTX OCuLink x4 marginal lab link; Main ruling after xbd-1): one model load per process (never a repeated load inside one
# process); SYNC after shipping an artifact to the host; the submitter must have the kernel watcher on. A new device bus loss in any cell:
# STOP and report the run as INCOMPLETE (lab hazard evidence, not a code-quality gate failure; no quality claim from an interrupted cell);
# NEVER an automatic retry, and no extra hold. Main power-cycles the host.
set -uo pipefail

HOLDS=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
STAGE=${S1_STAGE:-$HOLDS/stage}
# The declared target (target.json, loaded by load_target in hold_init): card, UUID, PCI BDF, model path and exact byte size.
CARD=""
UUID=""
BDF=""
MODEL=""
MODEL_SIZE=""
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

# hold_memory N: `--print-memory` handler (call before anything else). Prints "<fabric device GiB> <host GiB>" on stdout ("22.4 54" for
# the aggregate-route ledger) and, on stderr only, a line naming the separate PHYSICAL 24 GiB card and both purposes' owned bytes.
# Device = the ledger's explicit fabric_reservation_device_gib (the fabric's exclusive-use reservation for the declared target: a finite
# number > 0, no built-in default; the physical card_bytes is separate; the whole product run is judged by the observed headroom check, not
# by a declared owned+allowance fit). The device owned census (10 resident) is per purpose, INFORMATION from source/header arithmetic,
# never a driver figure and never gated here: product tile 768 owned 24,054,058,348 B (no diagnostic), primitive_selftest tile 512 owned
# 23,988,959,724 B (core 23,771,138,540 + diagnostic 217,821,184); the root owned figure is their MAX (sequential purposes), not a sum. These are
# PHYSICAL-backing figures of the canonical arena plan (a model over the queried granule/observed capacity, NOT a per-allocation physical or
# recovery measurement): weights 18,754,830,336 (payload 18,733,399,296 recorded separately), the cold F16 PAYLOAD term 1,298,923,520 charged
# once, and every rounding in the named allocation_backing_padding_bytes terms; slack over the 1 GiB runtime + 512 MiB safety: product 48,509,588,
# primitive 113,608,212 (information).
# Host = the sequential maximum host reservation (host_reservation_owned_bytes = max(the shipped-G2 beta arm at its 8192-row source
# capacity upper bound, the S1 purposes), unchanged: 55,740,268,544) + the fixed host allowance, rounded UP to whole GiB
# (55,740,268,544 + 2,147,483,648 = 57,887,752,192 -> 54 GiB = 57,982,058,496). Each S1 purpose's host census now includes the 2 GiB TOTAL bounded host
# snapshot cap (pending, resident, parked and quarantined snapshots; product 55,077,640,192, primitive 55,076,327,424), still below the beta arm. The 2 GiB
# device session-cache cap is nominal (not committed; the runtime derives the actual budget) and the 620,021,760 B pending ceiling lies inside the host cap.
# Reads ONLY a FINAL per-purpose ledger ($STAGE/ledger-holdN.json, template ledger-template.json), validated by ledger_profiles.py (both
# profiles, exact key sets and tiles, term arrays, sums, sequential peak); a missing, non-final, flat or inconsistent ledger is refused.
hold_memory() {
  local f=$STAGE/ledger-hold$1.json
  [ -r "$f" ] || { echo "no final memory ledger $f (S1Integrate provides the final per-purpose ledger): refusing" >&2; exit 2; }
  python3 "$HOLDS/ledger_profiles.py" print-memory "$f" || exit 2
  exit 0
}

# load_target: read the REQUIRED $STAGE/target.json (schema s1.hold_target.v1; see the header) into CARD UUID BDF MODEL MODEL_SIZE.
# Missing, unreadable or malformed data is fatal; there is no default and no environment override.
load_target() {
  local f=$STAGE/target.json out
  [ -r "$f" ] || die "declared target $f missing or unreadable (required; no fallback)"
  out=$(python3 - "$f" <<'PY'
import json, re, sys
def bad(msg):
    sys.exit(f"target.json: {msg}")
try:
    with open(sys.argv[1]) as fh:
        t = json.load(fh)
except (OSError, ValueError) as e:
    bad(f"unreadable or not JSON: {e}")
if not isinstance(t, dict) or t.get("schema") != "s1.hold_target.v1":
    bad("schema must be s1.hold_target.v1")
def text(v, what, pat):
    if not isinstance(v, str) or not re.fullmatch(pat, v):
        bad(f"{what} {v!r} is missing or malformed")
    return v
card = text(t.get("card"), "card", r"[A-Za-z0-9._-]+")
uuid = text(t.get("uuid"), "uuid", r"GPU-[0-9a-fA-F]+")
bdf = text(t.get("bdf"), "bdf", r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
m = t.get("model")
if not isinstance(m, dict):
    bad("model must be an object {path, size_bytes}")
path = text(m.get("path"), "model.path", r"/[^\x00-\x1f\x7f]*")
size = m.get("size_bytes")
if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
    bad(f"model.size_bytes {size!r} is not a positive integer")
print("\n".join([card, uuid, bdf, path, str(size)]))
PY
  ) || die "declared target $f is malformed"
  local -a v
  mapfile -t v <<<"$out"
  [ "${#v[@]}" = 5 ] || die "declared target $f yielded ${#v[@]} fields, expected 5"
  CARD=${v[0]}; UUID=${v[1]}; BDF=${v[2]}; MODEL=${v[3]}; MODEL_SIZE=${v[4]}
}

# hold_init NAME MAXSECS: guards, OUT, identity.
hold_init() {
  HOLD=$1; MAXS=$2
  T0=$(date +%s); BUDGET=$((MAXS - FINALIZE_S))
  : "${OUT:=$HOLDS/runs/$HOLD-$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$OUT" || exit 2
  [ -e "$OUT/hold.jsonl" ] && { echo "FATAL: $OUT already used" >&2; exit 2; }
  CELLS_RAN=(); CELLS_JSON="$OUT/cells.jsonl"; : > "$CELLS_JSON"; FAILED=(); BLOCKED=""
  load_target
  [ -n "${FABRIC_JOB:-}" ] && [ -n "${FABRIC_CARDS:-}" ] || die "not inside a fabric job (FABRIC_JOB/FABRIC_CARDS unset)"
  case ",$FABRIC_CARDS," in *,"$CARD",*|*,"$UUID",*) ;; *) die "FABRIC_CARDS=$FABRIC_CARDS does not name the declared target card $CARD ($UUID)" ;; esac
  [ "${FABRIC_CARDS//[^,]/}" = "" ] || die "FABRIC_CARDS=$FABRIC_CARDS names more than one card"
  [ "$(cat /sys/bus/pci/devices/$BDF/unique_id 2>/dev/null)" = "${UUID#GPU-}" ] || die "pci $BDF unique_id != $UUID"
  [ -r "$MODEL" ] && [ "$(stat -c %s "$MODEL")" = "$MODEL_SIZE" ] || die "model $MODEL missing or not $MODEL_SIZE bytes"
  [ -d "$STAGE" ] || die "stage dir $STAGE missing (inputs and binaries are staged before the hold)"
  for t in qwen4_slowlink_gate qwen4_greedy_ids qwen4_kld; do [ -x "$STAGE/bin-s1/$t" ] || die "$STAGE/bin-s1/$t missing"; done
  trap vram_stop EXIT
  jlog hold_start "hold=$HOLD" "out=$OUT" "fabric_job=$FABRIC_JOB" "budget_s=$BUDGET"
  identity
}

left() { echo $((BUDGET - ($(date +%s) - T0))); }
vram_used() { cat "/sys/bus/pci/devices/$BDF/mem_info_vram_used" 2>/dev/null || echo 99999999999; }
await_idle() { local end=$((SECONDS + $1)); while :; do (( $(vram_used) < IDLE_MAX )) && return 0; (( SECONDS >= end )) && return 1; sleep 2; done; }

# External VRAM sampler lifecycle (vram_sampler.py). One sampler at a time (cells are sequential); the EXIT trap in hold_init stops a leftover.
VRAM_PID=""
VRAM_SAMPLE_MS=20
# vram_start CELL_DIR: start BEFORE the GPU command; success only when the first valid sample is flushed (vram.ready) within 10 s.
vram_start() {
  local dir=$1 end=$((SECONDS + 10))
  python3 "$HOLDS/vram_sampler.py" "/sys/bus/pci/devices/$BDF" "$dir" "$VRAM_SAMPLE_MS" > "$dir/vram-sampler.log" 2>&1 &
  VRAM_PID=$!
  until [ -e "$dir/vram.ready" ]; do
    kill -0 "$VRAM_PID" 2>/dev/null || return 1
    (( SECONDS >= end )) && return 1
    sleep 0.05
  done
}
# vram_mark SIGNAL EVENT CELL_DIR: ask the sampler to stamp EVENT with its own CLOCK_MONOTONIC and wait (<= 5 s) until it is recorded.
vram_mark() {
  local end=$((SECONDS + 5))
  [ -n "$VRAM_PID" ] && kill -"$1" "$VRAM_PID" 2>/dev/null || return 1
  until grep -q "\"$2\"" "$3/vram-events.jsonl" 2>/dev/null; do
    (( SECONDS >= end )) && return 1
    sleep 0.01
  done
}
# vram_stop: SIGTERM, then wait for the sampler to write its summary (clean exit required). Idempotent.
vram_stop() {
  [ -n "$VRAM_PID" ] || return 0
  local pid=$VRAM_PID; VRAM_PID=""
  kill -TERM "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
}

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

# cell_env DIR [KEY=VAL ...]: sets E (the env prefix) for one cell. It strips every HIPFIRE_*/HIP_*/ROCR_*/ROC_*/HSA_*/GPU_*/AMD_* variable
# (except the fabric admission ones) and sets a fixed pin list that deliberately has NO HIPFIRE_OFFLOAD_EXEC: H2/H3 keep the default (D2
# pairing). The Hold 1-only pin is a per-cell extra KEY=VAL supplied by hold1a/hold1b through hold1-common.sh, never a global helper pin.
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
  # The spawn env actually applied to this cell (the unset list and every KEY=VAL), persisted as closed evidence next to command.txt.
  python3 - "$d/spawn-env.json" "${E[@]}" <<'PY'
import json, sys
out, argv = sys.argv[1], sys.argv[2:]
unset, pins, i = [], {}, 1  # argv[0] is `env`
while i < len(argv):
    if argv[i] == "-u" and i + 1 < len(argv):
        unset.append(argv[i + 1])
        i += 2
    else:
        k, _, v = argv[i].partition("=")
        pins[k] = v
        i += 1
with open(out, "w") as f:
    json.dump({"schema": "s1.cell_spawn_env.v1", "unset": sorted(unset), "set": pins}, f, indent=1)
    f.write("\n")
PY
  printf '%q ' "${E[@]}" setsid -w timeout -k $KILL_GRACE "$cap" "$@" > "$d/command.txt"; echo >> "$d/command.txt"
  cp /proc/meminfo "$d/meminfo-before.txt"
  printf '{"vram_used": %s, "gtt_used": %s}\n' "$(vram_used)" "$(cat /sys/bus/pci/devices/$BDF/mem_info_gtt_used 2>/dev/null || echo 0)" > "$d/baseline.json"
  # INFORMATION ONLY slow host sampling (2 s; never gates): sysfs vram/gtt and, per process of this cell, VmRSS/VmPin/RssAnon/RssFile
  # plus smaps classes. The gating VRAM observation is the independent external 20 ms sampler below.
  # External VRAM sampler (vram_sampler.py: only sysfs mem_info_vram_used/total at 20 ms), started BEFORE the GPU command, ready
  # before launch, stopped after the command and the drain. A cell whose sampler cannot start is INCOMPLETE and never launched.
  if ! vram_start "$d"; then
    vram_stop; echo "incomplete: vram sampler did not become ready" > "$d/status.txt"
    cell_row "$name" incomplete "vram sampler not ready" 0 0 ""; BLOCKED="$name could not start (INCOMPLETE: vram sampler)"; return 1
  fi
  python3 "$HOLDS/sample_mem.py" "$1" "$BDF" "$d/mem.csv" 2 > /dev/null 2>&1 &
  local samp=$! w0=$SECONDS
  if ! vram_mark USR1 command_start "$d"; then
    kill "$samp" 2>/dev/null; wait "$samp" 2>/dev/null; vram_stop
    echo "incomplete: vram sampler could not mark command_start" > "$d/status.txt"
    cell_row "$name" incomplete "vram sampler mark failed" 0 0 ""; BLOCKED="$name could not start (INCOMPLETE: vram sampler)"; return 1
  fi
  setsid -w "${E[@]}" timeout -k $KILL_GRACE "$cap" "$@" > "$d/stdout.txt" 2> "$d/stderr.txt" < /dev/null &
  local pid=$!; wait "$pid"; local rc=$?
  # command_end is marked the instant the command's process returns; the sampler keeps running through the leftover kill and the drain
  vram_mark USR2 command_end "$d" || echo "vram sampler: command_end mark not recorded" >> "$d/status.txt"
  kill "$samp" 2>/dev/null; wait "$samp" 2>/dev/null
  echo "$rc" > "$d/rc.txt"; cp /proc/meminfo "$d/meminfo-after.txt"
  local leftover; leftover=$(pgrep -g "$pid" 2>/dev/null | tr '\n' ' ')
  [ -z "$leftover" ] || { kill -KILL -- "-$pid" 2>/dev/null; sleep 2; }
  local drained=true; await_idle 90 || drained=false
  vram_stop || echo "vram sampler: did not stop cleanly" >> "$d/status.txt"
  # Forensic evidence index (never a verdict, runs for pass, fail, timeout and load-failure alike; the raw files are already preserved).
  python3 "$HOLDS/collect_evidence.py" "$d" "$name" "$rc" "$drained" > "$d/evidence.log" 2>&1 \
    || echo "evidence: collector failed; the raw files in this directory are preserved as they are" >> "$d/status.txt"
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

# headroom NAME CELL_DIR [INFO_CELL_DIR ...]: the PRODUCT-state VRAM headroom check (check_headroom.py) as a check. Requires the cell's
# external 20 ms sysfs telemetry AND its gate.json evidence.memory_diagnostics.samples pre_load/post_load/end: both minima of free VRAM
# must be >= 536,870,912 B. rc 0 pass, 1 fail (an observed minimum below the threshold), 2 INCOMPLETE (missing/invalid telemetry, never
# zero-filled). Any non-pass blocks every later cell. INFO_CELL_DIRs (e.g. the primitive cell) are recorded in the JSON, never gated.
headroom() {
  local name=$1 d=$2; shift 2
  local info=() c; for c in "$@"; do info+=(--info-cell "$c"); done
  local dd=$OUT/$name; mkdir -p "$dd"
  if [ -n "$BLOCKED" ]; then echo "skipped: $BLOCKED" > "$dd/status.txt"; cell_row "$name" skipped "$BLOCKED" 0 0 ""; return 1; fi
  python3 "$HOLDS/check_headroom.py" "$d" "${info[@]}" > "$dd/stdout.txt" 2> "$dd/stderr.txt"; local rc=$?
  local why; why=$(head -c 300 "$dd/stderr.txt" | tr '\n' ' ')
  case $rc in
    0) cell_row "$name" pass "" 0 0 ""; jlog check "name=$name" "status=pass"; return 0 ;;
    1) cell_row "$name" fail "$why" 1 0 ""; jlog check "name=$name" "status=fail"; FAILED+=("$name")
       [ -n "$BLOCKED" ] || BLOCKED="$name failed: free VRAM headroom below 512 MiB"; return 1 ;;
    *) cell_row "$name" incomplete "$why" "$rc" 0 ""; jlog check "name=$name" "status=incomplete"
       [ -n "$BLOCKED" ] || BLOCKED="$name INCOMPLETE: headroom telemetry missing or invalid"; return 1 ;;
  esac
}

# finish_summary: exit 0 only when every listed cell/check passed (any skipped/incomplete/failed row is not a pass).
finish_summary() {
  python3 - "$CELLS_JSON" "$OUT/hold-summary.json" "${HOLD:-?}" "${REQUIRED:-}" "${HOLD_SCOPE:-}" "${HOLD_ADVANCE:-false}" <<'PY'
import json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()] if sys.argv[1] else []
required = [r for r in sys.argv[4].split() if r]
seen = {r["name"]: r["status"] for r in rows}
missing = [n for n in required if n not in seen]
if any(r["status"] == "fail" for r in rows): status = "fail"
elif missing or any(r["status"] in ("skipped", "incomplete") for r in rows) or not required: status = "incomplete"
else: status = "pass"
# Forensic index only (collect_evidence.py): per-cell evidence.json, which lists the raw files and the gate progress sidecar. Nothing in
# it feeds `status` above; a partial progress sidecar never makes a cell or the hold pass.
out_dir = os.path.dirname(os.path.abspath(sys.argv[2]))
evidence = {r["name"]: r["name"] + "/evidence.json" for r in rows if os.path.isfile(os.path.join(out_dir, r["name"], "evidence.json"))}
json.dump({"hold": sys.argv[3], "status": status, "required": required, "missing": missing, "cells": rows, "evidence": evidence,
           "scope": sys.argv[5], "advance": status == "pass" and sys.argv[6] == "true",
           "scope_note": "scope names what a PASS proves; advance is true only for a hold whose entry declares HOLD_ADVANCE=true (1b); a 1a PASS is partial Hold 1 and never an advance",
           "evidence_note": "forensic only; the status above is computed from the cell/check rows, never from the evidence files or the progress sidecars"},
          open(sys.argv[2], "w"), indent=1)
print("hold %s: %s" % (sys.argv[3], status))
sys.exit(0 if status == "pass" else 1)
PY
}
