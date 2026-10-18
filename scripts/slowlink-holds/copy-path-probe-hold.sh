#!/usr/bin/env bash
# Main-authorized ONE model-free XTX copy-state probe (P0, <=60 s).
# This wrapper starts before fabric; the raw HIP probe has no D2/model route.
set -u
set +m
: "${OUT_PREFIX:?Set a fresh absolute output prefix}"
: "${TARGET_GPU_UUID:?Set the approved XTX visible-device UUID}"
HERE=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$(dirname -- "$OUT_PREFIX")"
if [ -e "$OUT_PREFIX.exit" ] || [ -e "$OUT_PREFIX.fabric.log" ]; then
  echo "Refusing to reuse probe output prefix: $OUT_PREFIX" >&2
  exit 2
fi
# Own session/PGID; close every non-stdio descriptor BEFORE exec and BEFORE locks.
setsid bash -c 'for f in /proc/$$/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-"; done; exec journalctl -k -f -n 0 -o short-iso --no-pager' >"$OUT_PREFIX.dmesg.log" 2>&1 </dev/null &
WPID=$!
printf '%s\n' "$WPID" >"$OUT_PREFIX.watcher-pgid"
finish() {
  local rc=$?
  trap - EXIT
  kill -TERM -- "-$WPID" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-$WPID" 2>/dev/null || true
  wait "$WPID" 2>/dev/null || true
  pgrep -af '[j]ournalctl -k -f' >"$OUT_PREFIX.watchers-after.txt" || true
  printf '%s\n' "$rc" >"$OUT_PREFIX.exit"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
sleep 1
if ! kill -0 -- "-$WPID" 2>/dev/null; then
  echo 'Kernel watcher failed before fabric; probe not submitted' >&2
  exit 2
fi
CMD=(/home/kaden/qcal/locks/fabric/fabric-run.sh --class correct --card xtx --mem-gb 1 --hostmem-gb 1 --hostmem-lock --max-secs 60 --priority P0 --owner Main.SlowLinkS1.CopyPath -- env PYTHONDONTWRITEBYTECODE=1 "ROCR_VISIBLE_DEVICES=$TARGET_GPU_UUID" HIP_VISIBLE_DEVICES=0 python3 "$HERE/pinned_copy_residue.py" --device 0)
printf '%q ' "${CMD[@]}" >"$OUT_PREFIX.command.txt"
printf '\n' >>"$OUT_PREFIX.command.txt"
"${CMD[@]}" >"$OUT_PREFIX.fabric.log" 2>&1
RC=$?
exit "$RC"
