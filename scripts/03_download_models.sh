#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "Run setup scripts first." >&2; exit 1; }
mkdir -p "$ROOT/models"
"$PY" - <<PY
from ultralytics import YOLO
YOLO("yolo11n.pt")
print("YOLO11n is cached by Ultralytics; it will be reused by camera mode.")
PY
