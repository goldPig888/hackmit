#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TUNNEL_DIR="$ROOT/.cloudflare"

if [ ! -f "$TUNNEL_DIR/config.yml" ]; then
    echo "Tunnel config not found. Run bash scripts/setup_cloudflare_tunnel.sh first."
    exit 1
fi

echo "Starting Cloudflare tunnel..."
echo "Tunnel will be accessible via the hostname configured in $TUNNEL_DIR/config.yml"
echo "Press Ctrl+C to stop the tunnel"
echo ""

cloudflared tunnel --config "$TUNNEL_DIR/config.yml" run