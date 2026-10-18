#!/usr/bin/env bash
# Generic kernel-watcher wrapper for ONE fabric submission:
#   OUT_PREFIX=/fresh/absolute/prefix run-with-watcher.sh COMMAND [ARG...]
# The kernel watcher starts BEFORE COMMAND (hence before any fabric lock), in its own session/PGID, with every
# fd > 2 closed so it can never inherit a fabric lock. Only the watcher group this script launched is ever signalled.
# Writes: $OUT_PREFIX.{command.txt,dmesg.log,watcher.pgid,fabric.log,watchers-after.txt,exit}.
set -u
set +m
if [ -z "${OUT_PREFIX:-}" ]; then
  echo "Usage: OUT_PREFIX=/fresh/absolute/prefix $0 COMMAND [ARG...]" >&2
  exit 2
fi
case "$OUT_PREFIX" in
  /*) ;;
  *) echo "OUT_PREFIX must be an absolute path: $OUT_PREFIX" >&2; exit 2 ;;
esac
case "$OUT_PREFIX" in
  */) echo "OUT_PREFIX must not end with '/': $OUT_PREFIX" >&2; exit 2 ;;
esac
if [ "$#" -eq 0 ]; then
  echo "No COMMAND given" >&2
  exit 2
fi
for p in "$OUT_PREFIX" "$OUT_PREFIX.exit" "$OUT_PREFIX.fabric.log" "$OUT_PREFIX.dmesg.log" \
  "$OUT_PREFIX.watcher.pgid" "$OUT_PREFIX.watchers-after.txt" "$OUT_PREFIX.command.txt"; do
  if [ -e "$p" ] || [ -L "$p" ]; then
    echo "Refusing to reuse output prefix (exists: $p)" >&2
    exit 2
  fi
done
CMD=("$@")
mkdir -p -- "$(dirname -- "$OUT_PREFIX")" || exit 2
WPID=
finish() {
  local rc=$?
  trap - EXIT
  trap '' TERM INT HUP
  if [ -n "$WPID" ]; then
    kill -TERM -- "-$WPID" 2>/dev/null || true
    for _ in $(seq 20); do
      kill -0 -- "-$WPID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 -- "-$WPID" 2>/dev/null; then
      kill -KILL -- "-$WPID" 2>/dev/null || true
    fi
    wait "$WPID" 2>/dev/null || true
  fi
  pgrep -af '[j]ournalctl -k -f' >"$OUT_PREFIX.watchers-after.txt" 2>&1 || true
  printf '%s\n' "$rc" >"$OUT_PREFIX.exit"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
# Own session/PGID; close every non-stdio descriptor BEFORE exec and BEFORE any lock is taken.
setsid bash -c 'for f in /proc/$$/fd/*; do n=${f##*/}; [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-"; done; exec journalctl -k -f -n 0 -o short-iso --no-pager' >"$OUT_PREFIX.dmesg.log" 2>&1 </dev/null &
WPID=$!
printf '%s\n' "$WPID" >"$OUT_PREFIX.watcher.pgid"
sleep 1
read -r WPGID WSTAT < <(ps -o pgid= -o stat= -p "$WPID" 2>/dev/null) || true
if [ "${WPGID:-}" != "$WPID" ] || [ -z "${WSTAT:-}" ] || [ "${WSTAT#Z}" != "$WSTAT" ] || ! kill -0 -- "-$WPID" 2>/dev/null; then
  echo "Kernel watcher not live as its own PGID before command; command not started" >&2
  exit 2
fi
printf '%q ' "${CMD[@]}" >"$OUT_PREFIX.command.txt"
printf '\n' >>"$OUT_PREFIX.command.txt"
"${CMD[@]}" >"$OUT_PREFIX.fabric.log" 2>&1
RC=$?
exit "$RC"
