#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "HALO Cloudflare Tunnel Setup"
echo "=========================================="

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install cloudflared
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        sudo dpkg -i cloudflared-linux-amd64.deb
    else
        echo "Please install cloudflared manually from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
        exit 1
    fi
fi

# Create tunnel directory
TUNNEL_DIR="$ROOT/.cloudflare"
mkdir -p "$TUNNEL_DIR"

# Check if tunnel already exists
if [ -f "$TUNNEL_DIR/tunnel.json" ]; then
    echo "Tunnel already exists at $TUNNEL_DIR/tunnel.json"
    echo "To recreate, remove the file and run this script again."
    read -p "Do you want to use existing tunnel? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm "$TUNNEL_DIR/tunnel.json"
    else
        TUNNEL_ID=$(jq -r '.tunnel' "$TUNNEL_DIR/tunnel.json")
        echo "Using existing tunnel: $TUNNEL_ID"
    fi
fi

# Create new tunnel if needed
if [ ! -f "$TUNNEL_DIR/tunnel.json" ]; then
    echo "Creating new Cloudflare tunnel..."
    cloudflared tunnel create halo-monitoring --config "$TUNNEL_DIR/config.yml"
    TUNNEL_ID=$(jq -r '.tunnel' "$TUNNEL_DIR/tunnel.json")
    echo "Created tunnel: $TUNNEL_ID"
fi

# Create config file
cat > "$TUNNEL_DIR/config.yml" <<EOF
tunnel: $(jq -r '.tunnel' "$TUNNEL_DIR/tunnel.json")
credentials-file: $TUNNEL_DIR/credentials.json

ingress:
  - hostname: halo-monitoring.your-domain.com
    service: http://localhost:8080
  - service: http_status:404
EOF

echo "=========================================="
echo "Tunnel setup complete!"
echo "=========================================="
echo "Tunnel ID: $(jq -r '.tunnel' "$TUNNEL_DIR/tunnel.json")"
echo "Config file: $TUNNEL_DIR/config.yml"
echo ""
echo "Next steps:"
echo "1. Add a custom domain in Cloudflare dashboard"
echo "2. Update the hostname in $TUNNEL_DIR/config.yml"
echo "3. Run: bash scripts/start_tunnel.sh"
echo "=========================================="