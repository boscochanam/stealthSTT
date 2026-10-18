#!/usr/bin/env bash
# Start/stop the upload portal on demand (pidfile in data/). Not a permanent service:
# it exits by itself after portal.idle_shutdown_minutes with no requests.
# Runs under a memory cap so an overload kills only the portal, not the laptop
# (7 GB RAM; the always-on stack uses ~3.4 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
PID=data/portal.pid
LOG=data/portal.log
mkdir -p data

running() { [[ -f $PID ]] && kill -0 "$(cat $PID)" 2>/dev/null; }

start() {
  if running; then echo "already running (pid $(cat $PID))"; return; fi
  avail=$(free -m | awk '/^Mem:/{print $7}')
  if (( avail < 2500 )); then echo "only ${avail} MB RAM available (need ~2500): close something first"; exit 1; fi
  setsid nohup systemd-run --user --scope --quiet -p MemoryMax=3G -p MemorySwapMax=512M \
    .venv/bin/silent-hermes portal >>"$LOG" 2>&1 &
  echo $! >"$PID"
  echo "started pid $(cat $PID); log $LOG"
}

stop() {
  if running; then kill "$(cat $PID)"; sleep 1; echo "stopped"; else echo "not running"; fi
  rm -f "$PID"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) running && echo "running (pid $(cat $PID))" || echo "not running" ;;
  *) echo "usage: $0 start|stop|restart|status"; exit 1 ;;
esac
