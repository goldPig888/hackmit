"""Real-time data streaming for ARGUS dashboard."""

from __future__ import annotations

import json
import asyncio
import time
from typing import AsyncGenerator, Callable, Any
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime


@dataclass
class DetectionEvent:
    """Real-time detection event."""
    object_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    timestamp_s: float
    risk_score: float | None = None
    ttc_s: float | None = None
    direction: str | None = None


@dataclass
class HazardEvent:
    """Real-time hazard event."""
    hazard_type: str
    severity: float
    description: str
    involved_objects: list[str]
    timestamp_s: float


@dataclass
class SystemStatus:
    """System status information."""
    fps: float
    active_objects: int
    total_detections: int
    risk_level: str
    uptime_s: float
    timestamp: str


class DataStreamer:
    """Manages real-time data streaming for dashboard."""

    def __init__(self, max_history: int = 100):
        """Initialize data streamer.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self.max_history = max_history
        self.detection_history: deque[DetectionEvent] = deque(maxlen=max_history)
        self.hazard_history: deque[HazardEvent] = deque(maxlen=max_history)
        self.current_status: SystemStatus | None = None
        self.subscribers: set[Callable] = set()
        self.start_time = time.time()
        self._lock = asyncio.Lock()

    async def add_detection(self, detection: DetectionEvent) -> None:
        """Add detection event and notify subscribers.
        
        Args:
            detection: Detection event to add
        """
        async with self._lock:
            self.detection_history.append(detection)
            detection_data = {
                "object_id": detection.object_id,
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "timestamp_s": detection.timestamp_s,
                "risk_score": detection.risk_score,
                "ttc_s": detection.ttc_s,
                "direction": detection.direction
            }
            await self._notify_subscribers("detection", detection_data)

    async def add_hazard(self, hazard: HazardEvent) -> None:
        """Add hazard event and notify subscribers.
        
        Args:
            hazard: Hazard event to add
        """
        async with self._lock:
            self.hazard_history.append(hazard)
            hazard_data = {
                "hazard_type": hazard.hazard_type,
                "severity": hazard.severity,
                "description": hazard.description,
                "involved_objects": hazard.involved_objects,
                "timestamp_s": hazard.timestamp_s
            }
            await self._notify_subscribers("hazard", hazard_data)

    async def update_status(self, status: SystemStatus) -> None:
        """Update system status and notify subscribers.
        
        Args:
            status: System status to update
        """
        async with self._lock:
            self.current_status = status
            status_data = {
                "fps": status.fps,
                "active_objects": status.active_objects,
                "total_detections": status.total_detections,
                "risk_level": status.risk_level,
                "uptime_s": status.uptime_s,
                "timestamp": status.timestamp
            }
            await self._notify_subscribers("status", status_data)

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to data updates.
        
        Args:
            callback: Async callback function to call on updates
        """
        self.subscribers.add(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from data updates.
        
        Args:
            callback: Callback function to remove
        """
        self.subscribers.discard(callback)

    async def _notify_subscribers(self, event_type: str, data: dict) -> None:
        """Notify all subscribers of new data.
        
        Args:
            event_type: Type of event (detection, hazard, status)
            data: Event data
        """
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Notify all subscribers concurrently
        tasks = []
        for callback in self.subscribers:
            try:
                task = callback(message)
                if asyncio.iscoroutine(task):
                    tasks.append(task)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_recent_detections(self, limit: int = 10) -> list[dict]:
        """Get recent detection events.
        
        Args:
            limit: Number of recent events to return
            
        Returns:
            List of recent detection events
        """
        async with self._lock:
            recent = list(self.detection_history)[-limit:]
            return [asdict(event) for event in recent]

    async def get_recent_hazards(self, limit: int = 10) -> list[dict]:
        """Get recent hazard events.
        
        Args:
            limit: Number of recent events to return
            
        Returns:
            List of recent hazard events
        """
        async with self._lock:
            recent = list(self.hazard_history)[-limit:]
            return [asdict(event) for event in recent]

    async def get_current_status(self) -> dict | None:
        """Get current system status.
        
        Returns:
            Current system status or None
        """
        async with self._lock:
            return asdict(self.current_status) if self.current_status else None

    async def event_generator(self) -> AsyncGenerator[str, None]:
        """Generate SSE (Server-Sent Events) stream.
        
        Yields:
            SSE-formatted event strings
        """
        event_queue: asyncio.Queue = asyncio.Queue()
        
        async def subscriber(message: dict) -> None:
            await event_queue.put(message)
        
        self.subscribe(subscriber)
        
        try:
            while True:
                try:
                    # Wait for event with timeout
                    message = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    
                    # Format as SSE
                    event_data = json.dumps(message)
                    yield f"data: {event_data}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    
        finally:
            self.unsubscribe(subscriber)

    def clear_history(self) -> None:
        """Clear all event history."""
        self.detection_history.clear()
        self.hazard_history.clear()