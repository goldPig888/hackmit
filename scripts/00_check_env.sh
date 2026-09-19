#!/usr/bin/env bash
set -euo pipefail

echo "HALO environment check"
command -v python3 >/dev/null || { echo "Python 3 is required." >&2; exit 1; }
python3 - <<'PY'
import sys
print(f"Python: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
PY
echo "Ready. Next: ./scripts/01_create_env.sh"
