#!/usr/bin/env python3
"""HALO Dashboard with real-time monitoring and video overlay."""

import sys
import asyncio
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from halo import HaloPipeline
from halo.dashboard import DashboardServer, DataStreamer
from halo.dashboard.data_streamer import DetectionEvent, HazardEvent, SystemStatus
from halo.detectors import YOLODetector
from halo.hazards import SyncHazardDetector
from halo.models import Detection


class DashboardRunner:
    """Run HALO with live dashboard."""

    def __init__(self, source: str = "0", port: int = 8080):
        """Initialize dashboard runner.
        
        Args:
            source: Video source (camera index or file path)
            port: Dashboard server port
        """
        self.source = source
        self.port = port
        self.data_streamer = DataStreamer()
        self.dashboard_server = DashboardServer(self.data_streamer, port=port)
        self.pipeline = HaloPipeline()
        self.sync_detector = SyncHazardDetector()
        self.total_detections = 0
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()

    async def run_with_camera(self) -> None:
        """Run with camera input."""
        try:
            import cv2
        except ImportError as error:
            raise SystemExit("Camera mode requires opencv-python. Install with: pip install opencv-python") from error

        # Initialize YOLO detector
        try:
            detector = YOLODetector()
        except ImportError as error:
            print("YOLO detector not available, using mock detector for demo")
            detector = self._create_mock_detector()

        capture = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not capture.isOpened():
            raise SystemExit(f"Could not open camera/video source: {self.source}")

        print(f"Starting dashboard on port {self.port}")
        print(f"Camera source: {self.source}")
        print("Press 'q' to quit")

        # Start dashboard server
        await self.dashboard_server.start()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                self.frame_count += 1
                timestamp = time.time() - self.start_time

                # Get detections
                detections = detector.detect(frame, timestamp_s=timestamp)
                self.total_detections += len(detections)

                # Process detections through pipeline
                track_states = []
                for detection in detections:
                    assessment = self.pipeline.update(detection, rider_speed_mps=4.0)
                    
                    # Stream detection event
                    detection_event = DetectionEvent(
                        object_id=detection.object_id,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                        timestamp_s=detection.timestamp_s,
                        risk_score=assessment.risk if assessment else None,
                        ttc_s=assessment.ttc_s if assessment else None,
                        direction=assessment.direction if assessment else None
                    )
                    await self.data_streamer.add_detection(detection_event)

                    if assessment:
                        # Convert to track state for hazard detection
                        import numpy as np
                        mock_track_state = self._create_mock_track_state(assessment)
                        track_states.append(mock_track_state)

                # Detect synchronized hazards
                if track_states:
                    hazards = self.sync_detector.update(track_states)
                    for hazard in hazards:
                        hazard_event = HazardEvent(
                            hazard_type=hazard.hazard_type,
                            severity=hazard.severity,
                            description=hazard.description,
                            involved_objects=hazard.involved_objects,
                            timestamp_s=hazard.timestamp_s
                        )
                        await self.data_streamer.add_hazard(hazard_event)

                # Update system status periodically
                if time.time() - self.last_fps_update > 1.0:
                    fps = self.frame_count / (time.time() - self.last_fps_update)
                    risk_level = self._calculate_risk_level()
                    
                    status = SystemStatus(
                        fps=fps,
                        active_objects=len(self.pipeline.tracks),
                        total_detections=self.total_detections,
                        risk_level=risk_level,
                        uptime_s=time.time() - self.start_time,
                        timestamp=time.strftime("%H:%M:%S")
                    )
                    await self.data_streamer.update_status(status)
                    
                    self.frame_count = 0
                    self.last_fps_update = time.time()

                # Display frame with overlays
                frame = self._overlay_detections(frame, detections)
                cv2.imshow("HALO Dashboard", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            capture.release()
            cv2.destroyAllWindows()
            detector.cleanup()

    async def run_demo(self) -> None:
        """Run with simulated data."""
        print(f"Starting dashboard on port {self.port}")
        print("Running in demo mode with simulated data")

        # Start dashboard server
        await self.dashboard_server.start()

        # Simulate data stream
        try:
            while True:
                await asyncio.sleep(0.1)
                timestamp = time.time() - self.start_time
                
                # Simulate detection
                import random
                if random.random() > 0.7:  # 30% chance of detection
                    self.total_detections += 1
                    
                    detection_event = DetectionEvent(
                        object_id=f"simulated-{self.total_detections}",
                        label=random.choice(["car", "person", "bicycle"]),
                        confidence=random.uniform(0.7, 0.95),
                        bbox=(random.uniform(100, 500), random.uniform(100, 400), 
                              random.uniform(50, 150), random.uniform(50, 150)),
                        timestamp_s=timestamp,
                        risk_score=random.uniform(0.0, 0.9),
                        ttc_s=random.uniform(1.0, 10.0) if random.random() > 0.5 else None,
                        direction=random.choice(["left", "right", "center"])
                    )
                    await self.data_streamer.add_detection(detection_event)

                # Simulate hazard occasionally
                if random.random() > 0.9:  # 10% chance of hazard
                    hazard_event = HazardEvent(
                        hazard_type=random.choice(["convergence", "coordinated_movement"]),
                        severity=random.uniform(0.5, 0.9),
                        description="Simulated coordinated threat",
                        involved_objects=[f"obj-{i}" for i in range(random.randint(2, 4))],
                        timestamp_s=timestamp
                    )
                    await self.data_streamer.add_hazard(hazard_event)

                # Update status
                if time.time() - self.last_fps_update > 1.0:
                    fps = self.frame_count / (time.time() - self.last_fps_update)
                    risk_level = self._calculate_risk_level()
                    
                    status = SystemStatus(
                        fps=fps,
                        active_objects=random.randint(0, 5),
                        total_detections=self.total_detections,
                        risk_level=risk_level,
                        uptime_s=time.time() - self.start_time,
                        timestamp=time.strftime("%H:%M:%S")
                    )
                    await self.data_streamer.update_status(status)
                    
                    self.frame_count = 0
                    self.last_fps_update = time.time()
                
                self.frame_count += 1

        except asyncio.CancelledError:
            print("Demo stopped")

    def _create_mock_detector(self):
        """Create a mock detector for demo purposes."""
        class MockDetector:
            def __init__(self):
                self.count = 0
            
            def detect(self, frame, timestamp_s):
                from halo.models import Detection
                self.count += 1
                return [Detection(
                    object_id=f"mock-{self.count}",
                    label="car",
                    confidence=0.9,
                    bbox=(100 + self.count * 10, 200, 80, 60),
                    timestamp_s=timestamp_s
                )]
            
            def get_supported_classes(self):
                return {"car", "person"}
            
            def cleanup(self):
                pass
        
        return MockDetector()

    def _create_mock_track_state(self, assessment):
        """Create mock track state from assessment."""
        import numpy as np
        from halo.models import TrackState
        
        return TrackState(
            object_id=assessment.object_id,
            label=assessment.label,
            confidence=0.9,
            position_m=np.array([1.0, 5.0]),
            velocity_mps=np.array([-0.5, -1.0]),
            position_sigma_m=0.5,
            ttc_s=assessment.ttc_s if assessment.ttc_s else 5.0,
            closing=assessment.risk > 0.3,
            timestamp_s=time.time() - self.start_time
        )

    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level."""
        if self.total_detections > 10:
            return "HIGH"
        elif self.total_detections > 5:
            return "MEDIUM"
        else:
            return "LOW"

    def _overlay_detections(self, frame, detections):
        """Overlay detection information on frame."""
        import cv2
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            color = (0, 255, 136)  # Green
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Draw label
            label = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(frame, label, (int(x), int(y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add system info
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        cv2.putText(frame, f"Total: {self.total_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        
        return frame


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HALO Live Dashboard")
    parser.add_argument("--source", default="demo", help="Video source (demo, camera index, or video file)")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard server port")
    args = parser.parse_args()

    runner = DashboardRunner(source=args.source, port=args.port)

    try:
        if args.source == "demo":
            await runner.run_demo()
        else:
            await runner.run_with_camera()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")


if __name__ == "__main__":
    asyncio.run(main())