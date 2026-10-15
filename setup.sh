#!/usr/bin/env bash
# Prepare directories only; dependency installation is explicit in the installation guide.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$PROJECT_DIR/setup_project.py"
