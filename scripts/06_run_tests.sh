#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Run setup scripts first." >&2; exit 1; }
cd "$ROOT"
"$PY" -m pytest -q
