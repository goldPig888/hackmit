# ARGUS Remote Monitoring System

Complete remote monitoring solution with Cloudflare tunnel, live web dashboard, and video overlay capabilities.

## 🚀 Quick Start

### Option 1: Demo Mode (Recommended for Testing)
```bash
cd /Users/mayespinola/Documents/hackmit
bash scripts/start_monitoring.sh
# Select option 1 for demo mode
```

### Option 2: Camera Mode
```bash
bash scripts/start_monitoring.sh
# Select option 2 for camera mode
```

### Option 3: Video Overlay Only
```bash
bash scripts/start_monitoring.sh
# Select option 3 for video overlay
```

## 🌐 Cloudflare Tunnel Setup

### Quick Tunnel (Temporary URL)
```bash
bash scripts/start_quick_tunnel.sh
```
This creates a temporary public URL for testing without configuration.

### Permanent Tunnel Setup
```bash
# Setup tunnel
bash scripts/setup_cloudflare_tunnel.sh

# Start tunnel
bash scripts/start_tunnel.sh
```

The permanent tunnel requires:
1. Cloudflare account
2. Custom domain configured in Cloudflare dashboard
3. Update hostname in `.cloudflare/config.yml`

## 📊 Live Dashboard Features

### Real-Time Data Streaming
- **Server-Sent Events (SSE)** for real-time updates
- **WebSocket support** for bidirectional communication
- **REST API endpoints** for historical data

### Dashboard Components
- **Status Bar**: FPS, active objects, total detections, risk level
- **Detection Feed**: Real-time detection events with risk scores
- **Hazard Alerts**: Synchronized threat warnings
- **Mobile Responsive**: Optimized for phone viewing

### API Endpoints
- `GET /` - Main dashboard HTML
- `GET /api/status` - Current system status
- `GET /api/detections?limit=10` - Recent detections
- `GET /api/hazards?limit=10` - Recent hazards
- `GET /stream` - SSE stream for real-time updates
- `GET /ws` - WebSocket endpoint

## 🎥 Video Overlay Features

### Real-Time Overlays
- **Detection Bounding Boxes**: Color-coded by risk level
- **Risk Assessment**: Real-time risk scores and TTC
- **Trajectory Prediction**: Predicted object paths
- **Hazard Warnings**: Coordinated threat alerts
- **System Info**: FPS, detection counts, time

### Controls
- `q` - Quit
- `s` - Save current frame
- `t` - Toggle trajectory overlays

### Usage
```bash
# Basic usage
python scripts/video_overlay.py --source 0

# With options
python scripts/video_overlay.py --source video.mp4 --no-trajectories --no-hazards
```

## 📱 Mobile Access

### Access Dashboard on Phone
1. Start the dashboard: `bash scripts/start_monitoring.sh` (option 1 or 2)
2. Start Cloudflare tunnel: `bash scripts/start_quick_tunnel.sh`
3. Open the provided URL on your phone
4. Dashboard is mobile-optimized with touch-friendly interface

### Mobile Features
- Responsive layout for different screen sizes
- Touch-optimized event lists
- Reduced font sizes for small screens
- Connection status indicator
- Real-time updates via SSE

## 🔧 Configuration

### Dashboard Server Configuration
Edit `scripts/run_dashboard.py` to customize:
- Port (default: 8080)
- Data stream settings
- Detection thresholds
- Risk calculation parameters

### Video Overlay Configuration
Edit `scripts/video_overlay.py` to customize:
- Overlay colors and styles
- Trajectory prediction settings
- Risk display options
- Hazard warning styles

### Cloudflare Tunnel Configuration
Edit `.cloudflare/config.yml` to customize:
- Tunnel hostname
- Service endpoints
- Ingress rules

## 🧪 Testing

### Test Dashboard Locally
```bash
# Start demo dashboard
.venv/bin/python scripts/run_dashboard.py --source demo --port 8080

# Test endpoints
curl http://localhost:8080/api/status
curl http://localhost:8080/api/detections
curl http://localhost:8080/api/hazards
```

### Test Video Overlay
```bash
# Test with mock detector
.venv/bin/python scripts/video_overlay.py --source 0

# Test with video file
.venv/bin/python scripts/video_overlay.py --source path/to/video.mp4
```

### Test Cloudflare Tunnel
```bash
# Start quick tunnel
bash scripts/start_quick_tunnel.sh

# Access the provided URL from your phone
# Should see the same dashboard as local access
```

## 📊 Data Flow

```
Camera/Video → Detection → ARGUS Pipeline → Risk Assessment
                                             ↓
                                     Data Streamer
                                             ↓
        ┌────────────────┬────────────────────┴────────────────┐
        ↓                ↓                                     ↓
   Dashboard Server  Video Overlay                    Cloudflare Tunnel
        ↓                ↓                                     ↓
   Web Browser    Local Display                     Remote Phone Access
```

## 🎯 Use Cases

### 1. Local Development
- Run demo mode for testing
- Use video overlay for local debugging
- Access dashboard on localhost:8080

### 2. Remote Monitoring
- Start camera dashboard with real detection
- Enable Cloudflare tunnel for phone access
- Monitor from anywhere with internet

### 3. Field Testing
- Use video overlay for real-time feedback
- Record overlay frames for analysis
- Monitor detection performance remotely

### 4. Integration Testing
- Test custom detectors with live data
- Validate risk assessment algorithms
- Test hazard detection patterns

## 🔍 Troubleshooting

### Dashboard Not Accessible
- Check if port 8080 is available
- Verify firewall settings
- Check dashboard server logs

### Cloudflare Tunnel Issues
- Verify cloudflared installation
- Check Cloudflare account status
- Ensure custom domain is configured

### Video Overlay Problems
- Check camera/video source availability
- Verify opencv-python installation
- Test with demo mode first

### Mobile Access Issues
- Ensure Cloudflare tunnel is running
- Check phone internet connection
- Verify URL is correct

## 📈 Performance

### Expected Performance
- **Dashboard**: <100ms latency for updates
- **Video Overlay**: 15-30 FPS depending on hardware
- **API Endpoints**: <50ms response time
- **SSE Streaming**: Real-time with minimal delay

### Optimization Tips
- Use demo mode for testing without camera
- Reduce detection frequency for lower CPU usage
- Adjust video resolution for better performance
- Use lightweight detection models

## 🚀 Next Steps

1. **Customize Dashboard**: Modify HTML/CSS in `src/argus/dashboard/server.py`
2. **Add Custom Detectors**: Integrate your weapons detection model
3. **Configure Alerts**: Set up risk threshold notifications
4. **Add Recording**: Implement video recording with overlays
5. **Deploy to Server**: Move to cloud hosting for 24/7 monitoring

## 📝 API Examples

### Get Current Status
```bash
curl http://localhost:8080/api/status
```

### Get Recent Detections
```bash
curl http://localhost:8080/api/detections?limit=5
```

### Stream Real-Time Events
```bash
curl -N http://localhost:8080/stream
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🎨 Customization

### Dashboard Styling
Edit the CSS in `src/argus/dashboard/server.py`:
- Color schemes
- Layout configurations
- Mobile breakpoints
- Animation effects

### Data Processing
Modify `scripts/run_dashboard.py`:
- Detection processing logic
- Risk calculation methods
- Hazard detection parameters
- Status update frequency

### Overlay Appearance
Edit `scripts/video_overlay.py`:
- Bounding box styles
- Text formatting
- Color schemes
- Information density

This system provides a complete solution for remote ARGUS monitoring with real-time visualization, mobile access, and comprehensive testing capabilities.