#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SOURCE="${1:-0}"
[[ -x "$PY" ]] || { echo "Run setup scripts first." >&2; exit 1; }
cd "$ROOT"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m argus camera --source "$SOURCE" --output runs/camera
