"""HALO Live Dashboard Server."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from aiohttp import web, WSMsgType
from aiohttp.web import Application, Request, Response, WebSocketResponse

from .data_streamer import DataStreamer, DetectionEvent, HazardEvent, SystemStatus
from .phone_receiver import PhoneStreamReceiver


class DashboardServer:
    """Live dashboard server for HALO monitoring."""

    def __init__(self, data_streamer: DataStreamer, host: str = "0.0.0.0", port: int = 8080):
        """Initialize dashboard server.
        
        Args:
            data_streamer: Data streamer instance
            host: Host to bind to
            port: Port to bind to
        """
        self.data_streamer = data_streamer
        self.host = host
        self.port = port
        self.app: Application = web.Application()
        self._setup_routes()
        self.websocket_clients: set[WebSocketResponse] = set()
        self.phone_receiver = PhoneStreamReceiver()
        self.world_state: dict = {"ready": False}

    def _setup_routes(self) -> None:
        """Setup HTTP routes."""
        self.app.add_routes([
            web.get('/', self.redirect_demo),
            web.get('/demo', self.serve_dashboard),
            web.get('/api/status', self.get_status),
            web.get('/api/detections', self.get_detections),
            web.get('/api/hazards', self.get_hazards),
            web.get('/stream', self.sse_stream),
            web.get('/video', self.video_stream),
            web.get('/frame.jpg', self.latest_frame_jpeg),
            web.get('/ws', self.websocket_handler),
            web.get('/phone', self.serve_phone_page),
            web.get('/ingest', self.ingest_handler),
            web.get('/api/world', self.get_world_state),
            web.static('/static', self._get_static_dir())
        ])

    def _get_static_dir(self) -> Path:
        """Get static files directory."""
        static_dir = Path(__file__).parent.parent.parent.parent / "dashboard_static"
        static_dir.mkdir(exist_ok=True)
        return static_dir

    async def redirect_demo(self, request: Request) -> Response:
        """Redirect root to the demo console."""
        raise web.HTTPFound('/demo')

    async def serve_dashboard(self, request: Request) -> Response:
        """Serve the predictive safety console (falls back to legacy dashboard)."""
        static_dir = self._get_static_dir()
        for name in ("console.html", "index.html"):
            html_path = static_dir / name
            if html_path.exists():
                return web.FileResponse(html_path)

        # Generate default HTML if not exists
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')

    def set_world_state(self, state: dict) -> None:
        """Store the latest console world-state snapshot (called by CV loop)."""
        self.world_state = state

    async def get_world_state(self, request: Request) -> Response:
        """Latest unified world-state snapshot for the console UI."""
        return web.json_response(self.world_state)

    async def get_status(self, request: Request) -> Response:
        """Get current system status."""
        status = await self.data_streamer.get_current_status()
        return web.json_response(status or {})

    async def get_detections(self, request: Request) -> Response:
        """Get recent detections."""
        limit = int(request.query.get('limit', 10))
        detections = await self.data_streamer.get_recent_detections(limit)
        return web.json_response(detections)

    async def get_hazards(self, request: Request) -> Response:
        """Get recent hazards."""
        limit = int(request.query.get('limit', 10))
        hazards = await self.data_streamer.get_recent_hazards(limit)
        return web.json_response(hazards)

    async def sse_stream(self, request: Request) -> Response:
        """Server-Sent Events stream for real-time updates."""
        response = web.StreamResponse()
        response.content_type = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
        
        await response.prepare(request)
        
        try:
            async for event in self.data_streamer.event_generator():
                await response.write(event.encode('utf-8'))
                await response.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            pass
        
        return response

    async def websocket_handler(self, request: Request) -> WebSocketResponse:
        """WebSocket handler for real-time updates."""
        ws = WebSocketResponse()
        await ws.prepare(request)
        self.websocket_clients.add(ws)
        
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    # Handle client messages if needed
                    pass
                elif message.type == WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
        finally:
            self.websocket_clients.discard(ws)
        
        return ws

    async def serve_phone_page(self, request: Request) -> Response:
        """Serve the iPhone camera+IMU streaming page."""
        html_path = self._get_static_dir() / "phone_stream.html"
        if html_path.exists():
            return web.FileResponse(html_path)
        return web.Response(text="phone_stream.html not found", status=404)

    async def ingest_handler(self, request: Request) -> WebSocketResponse:
        """WebSocket endpoint receiving phone camera frames and IMU data.

        Binary messages: 8-byte BE double timestamp (ms) + JPEG bytes.
        Text messages: JSON {"type": "imu", ...}.
        """
        ws = WebSocketResponse(max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        self.phone_receiver.connected = True
        print("Phone stream connected")

        try:
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    self.phone_receiver.push_frame_message(message.data)
                elif message.type == WSMsgType.TEXT:
                    self.phone_receiver.push_imu_message(message.data)
                elif message.type == WSMsgType.ERROR:
                    print(f'Ingest WebSocket error: {ws.exception()}')
        finally:
            self.phone_receiver.connected = False
            print("Phone stream disconnected")

        return ws

    async def video_stream(self, request: Request) -> Response:
        """MJPEG stream of processed (annotated) frames from the phone pipeline."""
        response = web.StreamResponse()
        response.content_type = 'multipart/x-mixed-replace; boundary=frame'

        await response.prepare(request)

        boundary = b'--frame\r\n'

        try:
            while True:
                frame_data = (self.phone_receiver.processed_frame()
                              or self.phone_receiver.latest_raw_frame())
                if not frame_data:
                    await asyncio.sleep(0.2)
                    continue

                await response.write(boundary)
                await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                await response.write(frame_data)
                await response.write(b'\r\n')

                await asyncio.sleep(0.08)  # ~12 FPS, matches phone send rate
        except (ConnectionResetError, asyncio.CancelledError):
            pass

        return response

    async def latest_frame_jpeg(self, request: Request) -> Response:
        """Single latest frame as JPEG — polled by the console canvas renderer."""
        frame = (self.phone_receiver.processed_frame()
                 or self.phone_receiver.latest_raw_frame())
        if not frame:
            return web.Response(status=204)
        return web.Response(
            body=frame, content_type='image/jpeg',
            headers={'Cache-Control': 'no-store, no-cache'})

    async def broadcast_to_websockets(self, message: dict) -> None:
        """Broadcast message to all connected WebSocket clients.
        
        Args:
            message: Message to broadcast
        """
        message_str = json.dumps(message)
        disconnected = set()
        
        for ws in self.websocket_clients:
            try:
                await ws.send_str(message_str)
            except Exception:
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected

    def _generate_dashboard_html(self) -> str:
        """Generate default dashboard HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HALO Live Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e6ed;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid #1a2332;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #00ff88;
            font-size: 2em;
            margin-bottom: 10px;
        }
        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .status-card {
            background: #121826;
            border: 1px solid #1a2332;
            border-radius: 8px;
            padding: 15px;
        }
        .status-card h3 {
            color: #4a9eff;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .status-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
        }
        .status-card .label {
            font-size: 0.8em;
            color: #8892b0;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: #121826;
            border: 1px solid #1a2332;
            border-radius: 8px;
            padding: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .status-bar { grid-template-columns: 1fr 1fr; }
            .header h1 { font-size: 1.5em; }
            .status-card .value { font-size: 1.4em; }
        }
        @media (max-width: 480px) {
            .status-bar { grid-template-columns: 1fr; }
            .header h1 { font-size: 1.2em; }
            .status-card .value { font-size: 1.2em; }
            .panel { padding: 15px; }
        }
        .panel h2 {
            color: #4a9eff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .event-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .event-item {
            padding: 10px;
            border-bottom: 1px solid #1a2332;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .event-item:last-child { border-bottom: none; }
        .event-info {
            flex: 1;
        }
        .event-type {
            font-weight: bold;
            color: #00ff88;
        }
        .event-time {
            font-size: 0.8em;
            color: #8892b0;
        }
        .event-details {
            font-size: 0.9em;
            color: #a0aab5;
        }
        .risk-high { color: #ff4444; }
        .risk-medium { color: #ffaa00; }
        .risk-low { color: #00ff88; }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            background: #121826;
            border: 1px solid #1a2332;
            z-index: 1000;
        }
        @media (max-width: 480px) {
            .connection-status {
                top: 5px;
                right: 5px;
                font-size: 0.7em;
                padding: 3px 8px;
            }
        }
        .connected { color: #00ff88; }
        .disconnected { color: #ff4444; }
        
        /* Mobile touch optimizations */
        @media (max-width: 768px) {
            .event-item {
                padding: 8px;
                font-size: 0.9em;
            }
            .event-details {
                font-size: 0.8em;
            }
            .panel h2 {
                font-size: 1em;
            }
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>HALO Live Dashboard</h1>
            <p>Real-time safety monitoring system</p>
        </div>
        
        <div class="status-bar">
            <div class="status-card">
                <h3>FPS</h3>
                <div class="value" id="fps">0</div>
                <div class="label">Frames per second</div>
            </div>
            <div class="status-card">
                <h3>Active Objects</h3>
                <div class="value" id="activeObjects">0</div>
                <div class="label">Tracked objects</div>
            </div>
            <div class="status-card">
                <h3>Total Detections</h3>
                <div class="value" id="totalDetections">0</div>
                <div class="label">Since start</div>
            </div>
            <div class="status-card">
                <h3>Risk Level</h3>
                <div class="value" id="riskLevel">LOW</div>
                <div class="label">Current threat level</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h2>Recent Detections</h2>
                <div class="event-list" id="detectionList">
                    <div class="event-item">
                        <div class="event-info">
                            <div class="event-type">Waiting for data...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>Hazard Alerts</h2>
                <div class="event-list" id="hazardList">
                    <div class="event-item">
                        <div class="event-info">
                            <div class="event-type">No hazards detected</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // SSE connection
        const eventSource = new EventSource('/stream');
        const connectionStatus = document.getElementById('connectionStatus');
        
        eventSource.onopen = () => {
            connectionStatus.textContent = 'Connected';
            connectionStatus.className = 'connection-status connected';
        };
        
        eventSource.onerror = () => {
            connectionStatus.textContent = 'Disconnected';
            connectionStatus.className = 'connection-status disconnected';
        };
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'status':
                    updateStatus(data.data);
                    break;
                case 'detection':
                    addDetection(data.data);
                    break;
                case 'hazard':
                    addHazard(data.data);
                    break;
            }
        };
        
        function updateStatus(status) {
            document.getElementById('fps').textContent = status.fps.toFixed(1);
            document.getElementById('activeObjects').textContent = status.active_objects;
            document.getElementById('totalDetections').textContent = status.total_detections;
            document.getElementById('riskLevel').textContent = status.risk_level;
            
            // Update risk level color
            const riskElement = document.getElementById('riskLevel');
            riskElement.className = 'value';
            if (status.risk_level === 'HIGH') {
                riskElement.classList.add('risk-high');
            } else if (status.risk_level === 'MEDIUM') {
                riskElement.classList.add('risk-medium');
            } else {
                riskElement.classList.add('risk-low');
            }
        }
        
        function addDetection(detection) {
            const list = document.getElementById('detectionList');
            const item = document.createElement('div');
            item.className = 'event-item';
            
            const riskClass = detection.risk_score > 0.7 ? 'risk-high' : 
                           detection.risk_score > 0.4 ? 'risk-medium' : 'risk-low';
            
            item.innerHTML = `
                <div class="event-info">
                    <div class="event-type ${riskClass}">${detection.label} - ${detection.object_id}</div>
                    <div class="event-details">
                        Confidence: ${(detection.confidence * 100).toFixed(1)}% | 
                        Risk: ${(detection.risk_score * 100).toFixed(1)}% |
                        ${detection.direction ? 'Direction: ' + detection.direction : ''}
                    </div>
                </div>
                <div class="event-time">${new Date(detection.timestamp_s * 1000).toLocaleTimeString()}</div>
            `;
            
            // Remove placeholder if exists
            const placeholder = list.querySelector('.event-type');
            if (placeholder && placeholder.textContent.includes('Waiting for data')) {
                list.innerHTML = '';
            }
            
            list.insertBefore(item, list.firstChild);
            
            // Keep only last 20 items
            while (list.children.length > 20) {
                list.removeChild(list.lastChild);
            }
        }
        
        function addHazard(hazard) {
            const list = document.getElementById('hazardList');
            const item = document.createElement('div');
            item.className = 'event-item';
            
            const severityClass = hazard.severity > 0.7 ? 'risk-high' : 
                               hazard.severity > 0.4 ? 'risk-medium' : 'risk-low';
            
            item.innerHTML = `
                <div class="event-info">
                    <div class="event-type ${severityClass}">${hazard.hazard_type.toUpperCase()}</div>
                    <div class="event-details">
                        ${hazard.description} | 
                        Severity: ${(hazard.severity * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="event-time">${new Date(hazard.timestamp_s * 1000).toLocaleTimeString()}</div>
            `;
            
            // Remove placeholder if exists
            const placeholder = list.querySelector('.event-type');
            if (placeholder && placeholder.textContent.includes('No hazards')) {
                list.innerHTML = '';
            }
            
            list.insertBefore(item, list.firstChild);
            
            // Keep only last 20 items
            while (list.children.length > 20) {
                list.removeChild(list.lastChild);
            }
        }
    </script>
</body>
</html>"""

    async def start(self) -> None:
        """Start the dashboard server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"Dashboard server running on http://{self.host}:{self.port}")
        print(f"Static files: {self._get_static_dir()}")

    async def stop(self) -> None:
        """Stop the dashboard server."""
        pass