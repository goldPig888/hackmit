#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Run ./scripts/01_create_env.sh first." >&2; exit 1; }
"$PY" -m pip install -e '.[dev]'
echo "Installed ARGUS core and test dependencies. Next: ./scripts/04_run_demo.sh"
