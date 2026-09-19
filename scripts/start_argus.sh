#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default values
SOURCE="demo"
PORT=8080
ENABLE_DASHBOARD=true
ENABLE_OVERLAY=false
ENABLE_TUNNEL=false
TUNNEL_TYPE="quick"
ENABLE_VISION=false
ENABLE_ADVANCED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-dashboard)
            ENABLE_DASHBOARD=false
            shift
            ;;
        --overlay)
            ENABLE_OVERLAY=true
            shift
            ;;
        --tunnel)
            ENABLE_TUNNEL=true
            shift
            ;;
        --permanent-tunnel)
            ENABLE_TUNNEL=true
            TUNNEL_TYPE="permanent"
            shift
            ;;
        --vision)
            ENABLE_VISION=true
            shift
            ;;
        --advanced)
            ENABLE_ADVANCED=true
            shift
            ;;
        --help)
            echo "ARGUS Unified Start Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --source SOURCE         Video source (demo, 0, /path/to/video.mp4) [default: demo]"
            echo "  --port PORT              Dashboard server port [default: 8080]"
            echo "  --no-dashboard          Disable dashboard server"
            echo "  --overlay               Enable video overlay"
            echo "  --tunnel                Enable Cloudflare quick tunnel"
            echo "  --permanent-tunnel      Enable Cloudflare permanent tunnel"
            echo "  --vision                Install vision dependencies (opencv, ultralytics)"
            echo "  --advanced              Install advanced pipeline dependencies (scipy)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Start dashboard with demo data"
            echo "  $0 --source 0 --overlay          # Start dashboard + overlay with camera"
            echo "  $0 --source 0 --overlay --tunnel # Start everything with camera + tunnel"
            echo "  $0 --permanent-tunnel           # Start dashboard with permanent tunnel"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ARGUS Unified Start"
echo "=========================================="
echo "Configuration:"
echo "  Source: $SOURCE"
echo "  Port: $PORT"
echo "  Dashboard: $ENABLE_DASHBOARD"
echo "  Video Overlay: $ENABLE_OVERLAY"
echo "  Cloudflare Tunnel: $ENABLE_TUNNEL ($TUNNEL_TYPE)"
echo "  Vision Dependencies: $ENABLE_VISION"
echo "  Advanced Pipeline: $ENABLE_ADVANCED"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install base dependencies
echo "Installing base dependencies..."
pip install -e '.[dashboard]' --quiet

# Install vision dependencies if requested
if [ "$ENABLE_VISION" = true ]; then
    echo "Installing vision dependencies..."
    pip install opencv-python ultralytics --quiet
fi

# Install advanced pipeline dependencies if requested
if [ "$ENABLE_ADVANCED" = true ]; then
    echo "Installing advanced pipeline dependencies..."
    pip install scipy --quiet
fi

# Install opencv if overlay is enabled and not already installed
if [ "$ENABLE_OVERLAY" = true ] && ! python -c "import cv2" 2>/dev/null; then
    echo "Installing opencv-python for video overlay..."
    pip install opencv-python --quiet
fi

# Create necessary directories
mkdir -p runs/overlays
mkdir -p dashboard_static

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up background processes..."
    if [ ! -z "${DASHBOARD_PID:-}" ]; then
        kill $DASHBOARD_PID 2>/dev/null || true
    fi
    if [ ! -z "${OVERLAY_PID:-}" ]; then
        kill $OVERLAY_PID 2>/dev/null || true
    fi
    if [ ! -z "${TUNNEL_PID:-}" ]; then
        kill $TUNNEL_PID 2>/dev/null || true
    fi
    # Clean up temporary files
    rm -f /tmp/cloudflare_output.log
    # Kill any remaining background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start dashboard server
if [ "$ENABLE_DASHBOARD" = true ]; then
    echo "Starting dashboard server on port $PORT..."
    python scripts/run_dashboard.py --source "$SOURCE" --port "$PORT" &
    DASHBOARD_PID=$!
    
    # Wait for dashboard to start
    sleep 3
    
    # Check if dashboard is running
    if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
        echo "ERROR: Dashboard server failed to start"
        exit 1
    fi
    
    echo "Dashboard server started (PID: $DASHBOARD_PID)"
    echo "Access at: http://localhost:$PORT"
fi

# Start video overlay if enabled
if [ "$ENABLE_OVERLAY" = true ]; then
    echo "Starting video overlay..."
    python scripts/video_overlay.py --source "$SOURCE" &
    OVERLAY_PID=$!
    echo "Video overlay started (PID: $OVERLAY_PID)"
fi

# Start Cloudflare tunnel if enabled
if [ "$ENABLE_TUNNEL" = true ]; then
    echo "Starting Cloudflare tunnel ($TUNNEL_TYPE)..."
    
    if [ "$TUNNEL_TYPE" = "quick" ]; then
        # Start quick tunnel in background
        bash scripts/start_quick_tunnel.sh > /tmp/cloudflare_output.log 2>&1 &
        TUNNEL_PID=$!
        
        # Wait a moment and show the URL
        sleep 5
        if [ -f /tmp/cloudflare_output.log ]; then
            echo "Cloudflare tunnel output:"
            head -5 /tmp/cloudflare_output.log
        fi
    else
        # Check if tunnel is configured
        if [ ! -f ".cloudflare/config.yml" ]; then
            echo "ERROR: Permanent tunnel not configured. Run setup_cloudflare_tunnel.sh first"
            exit 1
        fi
        bash scripts/start_tunnel.sh &
        TUNNEL_PID=$!
    fi
    
    echo "Cloudflare tunnel started (PID: $TUNNEL_PID)"
fi

echo "=========================================="
echo "ARGUS System Started Successfully"
echo "=========================================="

if [ "$ENABLE_DASHBOARD" = true ]; then
    echo "Dashboard: http://localhost:$PORT"
fi

if [ "$ENABLE_TUNNEL" = true ]; then
    echo "Cloudflare tunnel: Running (check terminal for URL)"
fi

echo "Press Ctrl+C to stop all services"
echo "=========================================="

# Wait for all background processes
wait