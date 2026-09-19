#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Run ./scripts/01_create_env.sh then ./scripts/02_install.sh first." >&2; exit 1; }
cd "$ROOT"
"$PY" -m argus demo --output runs/demo
