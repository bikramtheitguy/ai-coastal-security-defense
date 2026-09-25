#!/usr/bin/env bash
# One-command local run without containers (SQLite). Requires Python 3.11+ and Node.js 20+.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "» Creating Python virtual environment (.venv)"
  python3 -m venv .venv
fi
echo "» Installing backend dependencies"
.venv/bin/pip install -q -r backend/requirements-dev.txt
if [ ! -f frontend/out/index.html ] || [ "${REBUILD_FRONTEND:-0}" = "1" ]; then
  echo "» Building frontend (static export)"
  (cd frontend && npm ci --no-audit --no-fund && NEXT_TELEMETRY_DISABLED=1 npx next build)
fi
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
echo "» Starting on http://localhost:${PORT:-8000}  (data in $DATA_DIR; Ctrl+C to stop)"
cd backend
exec ../.venv/bin/python -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
