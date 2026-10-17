#!/usr/bin/env bash
# Runs the backend API suite (SQLite; set TEST_DATABASE_URL for PostgreSQL) and the Playwright browser suite.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
echo "» Backend API tests"
"$PY" -m pytest "$ROOT/tests/backend" -q -p no:warnings
if [ "${SKIP_E2E:-0}" != "1" ]; then
  echo "» Building frontend for browser tests"
  (cd "$ROOT/frontend" && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && NEXT_TELEMETRY_DISABLED=1 npx next build >/dev/null)
  echo "» Browser (Playwright) tests"
  (cd "$ROOT/tests/e2e" && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && PYTHON="$PY" npx playwright test)
fi
