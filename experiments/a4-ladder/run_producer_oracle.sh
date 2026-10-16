#!/usr/bin/env bash
# Build + run the producer-candidate correctness probe (producer_oracle.cpp).
#
#   run_producer_oracle.sh <gfx1201|gfx1151> [outdir]
#
# One binary per variant. A variant is "<N>[o][s]":
#   N  IU4_A4_CANDIDATES (1, 2, 4 or 8)
#   o  also -DHIPFIRE_IU4_ONEPASS=1 (gfx1151 C2: the real one-pass path)
#   s  also -DPO_SLAB=1 (one-pass C2 slab layout; needs the one-pass path)
# Defaults: gfx1201 -> "2 4 8" (N=2 is one-pass C2 on gfx1201 by itself);
#           gfx1151 -> "2o 2 4 8" (2o = one-pass C2, 2 = remap + generic).
# Override with PO_VARIANTS="2 4 8", e.g. PO_VARIANTS="2 2s 4 8" on gfx1201.
#
# Env: HIPCC (default hipcc), PO_BLOCKS (random blocks, default 8192),
#      PO_TOKENS (default 7), PO_DEVICE (default 0), PO_SEED,
#      PO_TIMEOUT (per-binary run seconds, default 120),
#      PO_JOBS (parallel builds, default 1).
# Exit: 0 all variants PASS; 1 any probe FAIL; 2 build/setup failure.
set -uo pipefail

ARCH="${1:?usage: $0 <gfx1201|gfx1151> [outdir]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${2:-$HERE/build-$ARCH}"
HIPCC="${HIPCC:-hipcc}"
BLOCKS="${PO_BLOCKS:-8192}"
TOKENS="${PO_TOKENS:-7}"
DEVICE="${PO_DEVICE:-0}"
RUN_TIMEOUT="${PO_TIMEOUT:-120}"
JOBS="${PO_JOBS:-1}"

case "$ARCH" in
    gfx1201) DEFAULT_VARIANTS="2 4 8" ;;
    gfx1151) DEFAULT_VARIANTS="2o 2 4 8" ;;
    *) echo "unsupported arch '$ARCH' (gfx1201 | gfx1151)" >&2; exit 2 ;;
esac
VARIANTS="${PO_VARIANTS:-$DEFAULT_VARIANTS}"
mkdir -p "$OUT"

variant_flags() {  # $1 = variant -> compile defines on stdout
    local v="$1" n="${1%%[!0-9]*}" rest="${1#"${1%%[!0-9]*}"}"
    case "$n" in 1|2|4|8) ;; *) return 1 ;; esac
    local flags="-DIU4_A4_CANDIDATES=$n"
    [[ "$rest" == *o* ]] && flags="$flags -DHIPFIRE_IU4_ONEPASS=1"
    [[ "$rest" == *s* ]] && flags="$flags -DPO_SLAB=1"
    [[ "${rest//[os]/}" == "" ]] || return 1
    printf '%s' "$flags"
}

build_one() {  # $1 = variant
    local v="$1" flags bin
    flags="$(variant_flags "$v")" || { echo "bad variant '$v'" >&2; return 2; }
    bin="$OUT/producer_oracle_${ARCH}_c${v}"
    # Production hipcc flags (compiler.rs core_hipcc_flags) minus --genco: this
    # is a full host+device binary. Keep host FP strict: no -march=native/-ffast-math.
    echo "[build $v] $HIPCC -O3 -std=c++17 --offload-arch=$ARCH -mcode-object-version=6 -fuse-cuid=none $flags -o $bin"
    "$HIPCC" -O3 -std=c++17 "--offload-arch=$ARCH" -mcode-object-version=6 -fuse-cuid=none \
        $flags -o "$bin" "$HERE/producer_oracle.cpp" >"$OUT/build_${v}.log" 2>&1 || {
        echo "[build $v] FAILED, see $OUT/build_${v}.log" >&2
        return 2
    }
}

# ---- builds ----
build_fail=0
pids=()
for v in $VARIANTS; do
    if (( JOBS > 1 )); then
        build_one "$v" & pids+=($!)
        while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
    else
        build_one "$v" || build_fail=1
    fi
done
for p in "${pids[@]:-}"; do [[ -n "$p" ]] && { wait "$p" || build_fail=1; }; done
if (( build_fail )); then echo "build failures; not running" >&2; exit 2; fi

# ---- runs ----
rc=0
for v in $VARIANTS; do
    bin="$OUT/producer_oracle_${ARCH}_c${v}"
    log="$OUT/run_${v}.log"
    args=(--device "$DEVICE" --random "$BLOCKS" --tokens "$TOKENS")
    [[ -n "${PO_SEED:-}" ]] && args+=(--seed "$PO_SEED")
    echo "[run $v] timeout $RUN_TIMEOUT $bin ${args[*]}"
    timeout "$RUN_TIMEOUT" "$bin" "${args[@]}" >"$log" 2>&1
    st=$?
    cat "$log"
    if (( st == 0 )); then
        echo "[run $v] PASS"
    else
        echo "[run $v] exit $st (1 = check failure, 2 = HIP/setup error, 124 = timeout)" >&2
        rc=1
    fi
done
grep -h '^PRODUCER_ORACLE' "$OUT"/run_*.log || true
exit $rc
