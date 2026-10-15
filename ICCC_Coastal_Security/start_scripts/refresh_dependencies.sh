#!/usr/bin/env bash
# Generate reviewable dependency locks. No install scripts, commits or pushes.
# Run with uv 0.13.1 and npm 10 on Node 22; requires registry access.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
uv pip compile backend/requirements.in --python-version 3.11 --universal --generate-hashes --no-header --output-file backend/requirements.txt
uv pip compile backend/requirements-dev.in --constraint backend/requirements.txt --python-version 3.11 --universal --generate-hashes --no-header --output-file backend/requirements-dev.txt
(cd frontend && npm install --package-lock-only --ignore-scripts --no-audit --no-fund)
echo "Review the lock diff and run ICCC POC CI before accepting these files."
