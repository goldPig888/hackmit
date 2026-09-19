#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "HALO Remote Monitoring Setup"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e '.[dashboard]' --quiet

# Install opencv if not present
if ! python -c "import cv2" 2>/dev/null; then
    echo "Installing opencv-python for video processing..."
    pip install opencv-python --quiet
fi

# Create necessary directories
mkdir -p runs/overlays
mkdir -p dashboard_static

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Choose an option:"
echo "1. Start demo dashboard (simulated data)"
echo "2. Start camera dashboard (requires camera)"
echo "3. Start video overlay only"
echo "4. Start Cloudflare tunnel"
echo "5. Start quick Cloudflare tunnel (temporary)"
echo ""
read -p "Enter option (1-5): " choice

case $choice in
    1)
        echo "Starting demo dashboard..."
        python scripts/run_dashboard.py --source demo --port 8080
        ;;
    2)
        echo "Starting camera dashboard..."
        python scripts/run_dashboard.py --source 0 --port 8080
        ;;
    3)
        echo "Starting video overlay..."
        python scripts/video_overlay.py --source 0
        ;;
    4)
        echo "Starting Cloudflare tunnel..."
        bash scripts/start_tunnel.sh
        ;;
    5)
        echo "Starting quick Cloudflare tunnel..."
        bash scripts/start_quick_tunnel.sh
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac