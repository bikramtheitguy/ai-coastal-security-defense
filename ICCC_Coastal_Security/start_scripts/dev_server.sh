#!/usr/bin/env bash
# Start/stop the backend (serving the built frontend) in the background. Usage: dev_server.sh start|stop|restart [--fresh]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
PIDFILE="$DATA_DIR/server.pid"
stop() { if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; sleep 1; fi; }
start() {
  mkdir -p "$DATA_DIR"
  if [ "${1:-}" = "--fresh" ]; then rm -f "$DATA_DIR"/iccc_poc.db*; fi
  cd "$ROOT/backend"
  nohup "$PY" -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" > "$DATA_DIR/server.log" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 60); do curl -sf "http://localhost:${PORT:-8000}/api/public/info" >/dev/null && { echo "ICCC POC running on http://localhost:${PORT:-8000}"; return 0; }; sleep 1; done
  echo "Server did not start; see $DATA_DIR/server.log"; tail -20 "$DATA_DIR/server.log"; return 1
}
case "${1:-start}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  restart) stop; start "${2:-}" ;;
esac
