#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "HALO Quick Cloudflare Tunnel"
echo "=========================================="
echo "This will create a temporary public URL for local testing"
echo "Press Ctrl+C to stop the tunnel"
echo "=========================================="
echo ""

# Start a quick tunnel (no config needed)
cloudflared tunnel --url http://localhost:8080