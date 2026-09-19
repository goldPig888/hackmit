#!/usr/bin/env python3
"""HALO Video Overlay - Real-time overlays on video feed."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from halo import HaloPipeline
from halo.detectors import YOLODetector
from halo.hazards import SyncHazardDetector
from halo.models import Detection, TrackState
import cv2
import time
import numpy as np


class VideoOverlay:
    """Real-time video overlay system for HALO."""

    def __init__(self, source: str = "0", show_trajectories: bool = True, 
                 show_risk: bool = True, show_hazards: bool = True):
        """Initialize video overlay system.
        
        Args:
            source: Video source (camera index or file path)
            show_trajectories: Show predicted trajectories
            show_risk: Show risk assessments
            show_hazards: Show hazard warnings
        """
        self.source = source
        self.show_trajectories = show_trajectories
        self.show_risk = show_risk
        self.show_hazards = show_hazards
        
        self.pipeline = HaloPipeline()
        self.sync_detector = SyncHazardDetector()
        self.detector = None
        self.total_detections = 0
        self.start_time = time.time()

    def initialize_detector(self) -> None:
        """Initialize the detection model."""
        try:
            self.detector = YOLODetector()
            print("YOLO detector initialized successfully")
        except ImportError:
            print("YOLO not available, using mock detector")
            self.detector = self._create_mock_detector()

    def run(self) -> None:
        """Run the video overlay system."""
        self.initialize_detector()
        
        capture = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not capture.isOpened():
            raise SystemExit(f"Could not open camera/video source: {self.source}")

        print(f"Starting video overlay with source: {self.source}")
        print("Press 'q' to quit, 's' to save frame, 't' to toggle trajectories")

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp = time.time() - self.start_time
                
                # Get detections
                detections = self.detector.detect(frame, timestamp_s=timestamp)
                self.total_detections += len(detections)

                # Process detections
                track_states = []
                overlay_data = []
                
                for detection in detections:
                    assessment = self.pipeline.update(detection, rider_speed_mps=4.0)
                    
                    overlay_data.append({
                        'detection': detection,
                        'assessment': assessment
                    })
                    
                    if assessment:
                        import numpy as np
                        mock_track_state = self._create_mock_track_state(assessment)
                        track_states.append(mock_track_state)

                # Detect hazards
                hazards = []
                if track_states:
                    hazards = self.sync_detector.update(track_states)

                # Apply overlays
                frame = self._apply_overlays(frame, overlay_data, hazards, timestamp)

                # Display frame
                cv2.imshow("HALO Video Overlay", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame)
                elif key == ord('t'):
                    self.show_trajectories = not self.show_trajectories
                    print(f"Trajectories: {'ON' if self.show_trajectories else 'OFF'}")

        finally:
            capture.release()
            cv2.destroyAllWindows()
            if self.detector:
                self.detector.cleanup()

    def _apply_overlays(self, frame, overlay_data, hazards, timestamp):
        """Apply all visual overlays to the frame."""
        # Apply detection overlays
        for data in overlay_data:
            detection = data['detection']
            assessment = data['assessment']
            
            frame = self._overlay_detection(frame, detection, assessment)
            
            if self.show_trajectories and assessment:
                frame = self._overlay_trajectory(frame, assessment)
            
            if self.show_risk and assessment:
                frame = self._overlay_risk(frame, assessment)

        # Apply hazard overlays
        if self.show_hazards and hazards:
            frame = self._overlay_hazards(frame, hazards)

        # Apply system info overlay
        frame = self._overlay_system_info(frame, timestamp)

        return frame

    def _overlay_detection(self, frame, detection, assessment):
        """Overlay detection bounding box and label."""
        x, y, w, h = detection.bbox
        
        # Color based on risk
        if assessment and assessment.risk > 0.7:
            color = (0, 0, 255)  # Red for high risk
        elif assessment and assessment.risk > 0.4:
            color = (0, 165, 255)  # Orange for medium risk
        else:
            color = (0, 255, 136)  # Green for low risk
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        
        # Draw label background
        label = f"{detection.label} {detection.confidence:.2f}"
        if assessment:
            label += f" R:{assessment.risk:.2f}"
        
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (int(x), int(y - label_height - 10)), 
                     (int(x + label_width), int(y)), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (int(x), int(y - 5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

    def _overlay_trajectory(self, frame, assessment):
        """Overlay predicted trajectory."""
        try:
            from halo.trajectory import predict_path
            import numpy as np
            
            # Get predicted path
            times, path = predict_path(
                assessment.closest_distance_m * np.array([0.5, 1.0]),  # Approximate position
                np.array([-1.0, -2.0])  # Approximate velocity
            )
            
            # Convert to image coordinates (simplified)
            frame_height, frame_width = frame.shape[:2]
            scale_x = frame_width / 20.0  # Assume 20m field of view
            scale_y = frame_height / 15.0  # Assume 15m field of view
            
            # Draw trajectory line
            points = []
            for i, point in enumerate(path[:10]):  # Show first 10 points
                screen_x = int(frame_width / 2 + point[0] * scale_x)
                screen_y = int(frame_height - point[1] * scale_y)
                points.append((screen_x, screen_y))
            
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], False, (255, 0, 255), 2)
                
                # Draw arrow at end
                if len(points) >= 2:
                    cv2.arrowedLine(frame, points[-2], points[-1], (255, 0, 255), 3)
        
        except Exception as e:
            print(f"Trajectory overlay error: {e}")
        
        return frame

    def _overlay_risk(self, frame, assessment):
        """Overlay risk assessment information."""
        # Risk indicator
        risk_text = f"Risk: {assessment.risk:.2f}"
        risk_color = (0, 0, 255) if assessment.risk > 0.7 else \
                    (0, 165, 255) if assessment.risk > 0.4 else (0, 255, 136)
        
        cv2.putText(frame, risk_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # TTC if available
        if assessment.ttc_s:
            ttc_text = f"TTC: {assessment.ttc_s:.1f}s"
            cv2.putText(frame, ttc_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Direction
        direction_text = f"Dir: {assessment.direction.upper()}"
        cv2.putText(frame, direction_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def _overlay_hazards(self, frame, hazards):
        """Overlay hazard warnings."""
        y_offset = 180
        
        for hazard in hazards:
            # Hazard warning box
            color = (0, 0, 255) if hazard.severity > 0.7 else \
                   (0, 165, 255) if hazard.severity > 0.4 else (255, 165, 0)
            
            warning_text = f"WARNING: {hazard.hazard_type.upper()}: {hazard.description}"
            
            # Draw warning background
            (text_width, text_height), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (10, y_offset - text_height - 5), 
                         (10 + text_width + 10, y_offset + 5), color, -1)
            
            # Draw warning text
            cv2.putText(frame, warning_text, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 30
        
        return frame

    def _overlay_system_info(self, frame, timestamp):
        """Overlay system information."""
        # FPS
        fps = 1.0 / (time.time() - self.start_time + 0.001) if timestamp > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        
        # Detection count
        cv2.putText(frame, f"Detections: {self.total_detections}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
        
        # Time
        time_text = f"Time: {timestamp:.1f}s"
        cv2.putText(frame, time_text, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Active tracks
        active_tracks = len(self.pipeline.tracks)
        cv2.putText(frame, f"Tracks: {active_tracks}", (frame.shape[1] - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def _save_frame(self, frame):
        """Save current frame to disk."""
        timestamp = int(time.time())
        filename = f"runs/overlays/frame_{timestamp}.png"
        
        import os
        os.makedirs("runs/overlays", exist_ok=True)
        
        cv2.imwrite(filename, frame)
        print(f"Frame saved: {filename}")

    def _create_mock_detector(self):
        """Create mock detector for testing."""
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
                    bbox=(100 + (self.count % 10) * 50, 200, 80, 60),
                    timestamp_s=timestamp_s
                )]
            
            def get_supported_classes(self):
                return {"car", "person"}
            
            def cleanup(self):
                pass
        
        return MockDetector()

    def _create_mock_track_state(self, assessment):
        """Create mock track state."""
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HALO Video Overlay")
    parser.add_argument("--source", default="0", help="Video source (camera index or video file)")
    parser.add_argument("--no-trajectories", action="store_true", help="Hide trajectory overlays")
    parser.add_argument("--no-risk", action="store_true", help="Hide risk overlays")
    parser.add_argument("--no-hazards", action="store_true", help="Hide hazard overlays")
    args = parser.parse_args()

    overlay = VideoOverlay(
        source=args.source,
        show_trajectories=not args.no_trajectories,
        show_risk=not args.no_risk,
        show_hazards=not args.no_hazards
    )

    try:
        overlay.run()
    except KeyboardInterrupt:
        print("\nVideo overlay stopped")


if __name__ == "__main__":
    main()