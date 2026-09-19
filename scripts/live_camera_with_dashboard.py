#!/usr/bin/env python3
"""Live camera with integrated dashboard server for phone viewing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import time
import json
import asyncio
import threading
from datetime import datetime
from typing import Optional

from argus import (
    AdvancedArgusPipeline, AdvancedPipelineConfig,
    ArgusPipeline, Detection
)
from argus.detectors import YOLODetector
from argus.dashboard import DataStreamer, DashboardServer


class LiveCameraWithDashboard:
    """Live camera feed with integrated dashboard server for phone viewing."""

    def __init__(self, camera_source: int = 0, port: int = 8080,
                 use_advanced: bool = True, enable_overlay: bool = True):
        """Initialize live camera with dashboard.
        
        Args:
            camera_source: Camera source (0 = default camera)
            port: Dashboard server port
            use_advanced: Use advanced pipeline with IMU compensation
            enable_overlay: Enable visual overlays on camera feed
        """
        self.camera_source = camera_source
        self.port = port
        self.use_advanced = use_advanced
        self.enable_overlay = enable_overlay
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_source}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize ARGUS pipeline
        if use_advanced:
            config = AdvancedPipelineConfig(
                enable_imu_compensation=True,
                enable_visual_odometry=False,
                enable_ekf=True,
                enable_cpa_collision=True
            )
            self.pipeline = AdvancedArgusPipeline(config)
        else:
            self.pipeline = ArgusPipeline()
        
        # Initialize detector
        try:
            self.detector = YOLODetector(model_name="yolov8n.pt")
            print("✓ YOLO detector loaded")
        except Exception as e:
            print(f"Warning: Could not load YOLO detector: {e}")
            self.detector = None
        
        # Initialize data streamer and dashboard server
        self.data_streamer = DataStreamer()
        self.dashboard_server = DashboardServer(self.data_streamer, port=port)
        
        # Shared frame buffer for video streaming
        self.current_frame = None
        self.current_frame_lock = threading.Lock()
        
        # Motion testing state
        self.motion_test_active = False
        self.motion_test_type = None
        self.test_start_time = None
        self.motion_history = []
        
        # Simulated IMU for motion testing
        self.simulated_yaw = 0.0
        self.simulated_yaw_rate = 0.0
        self.simulated_speed = 1.5
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        # Colors for overlays
        self.colors = {
            'car': (0, 255, 0),
            'person': (255, 0, 0),
            'bicycle': (0, 165, 255),
            'motorcycle': (255, 255, 0),
            'truck': (0, 0, 255),
            'bus': (128, 0, 128),
            'default': (255, 255, 255)
        }
        
        self.risk_colors = {
            'safe': (0, 255, 0),
            'low': (0, 255, 255),
            'medium': (0, 165, 255),
            'high': (0, 0, 255),
            'critical': (0, 0, 128)
        }

    def process_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        """Process a frame through ARGUS pipeline.
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'detections': [],
            'risk_assessments': [],
            'object_states': [],
            'ego_state': None,
            'system_status': {}
        }
        
        # Run detection
        if self.detector:
            try:
                detections = self.detector.detect(frame, timestamp)
                results['detections'] = detections
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Process detections through pipeline
        for detection in results['detections']:
            # Simulate IMU data
            imu_data = self._simulate_imu_data(timestamp)
            
            # Update pipeline
            if self.use_advanced:
                risk_assessment = self.pipeline.update(detection, imu_data=imu_data)
            else:
                risk_assessment = self.pipeline.update(detection, rider_speed_mps=1.5)
            
            if risk_assessment:
                results['risk_assessments'].append(risk_assessment)
        
        # Get object states (advanced pipeline only)
        if self.use_advanced:
            results['object_states'] = self.pipeline.get_all_object_states()
            results['ego_state'] = self.pipeline.get_ego_state()
            results['system_status'] = self.pipeline.get_system_status()
        
        return results

    def _simulate_imu_data(self, timestamp: float) -> tuple:
        """Simulate IMU data for testing with hand motion."""
        # Update simulated motion based on test type
        if self.motion_test_active:
            if self.motion_test_type == "approaching":
                self.simulated_speed = min(3.0, self.simulated_speed + 0.01)
                self.simulated_yaw_rate = 0.0
            elif self.motion_test_type == "turning":
                self.simulated_yaw_rate = 0.5 * np.sin(timestamp * 2.0)
                self.simulated_yaw += self.simulated_yaw_rate * 0.033
                self.simulated_speed = 1.5
            elif self.motion_test_type == "lateral":
                self.simulated_yaw_rate = 0.0
                self.simulated_speed = 1.5
            else:
                self.simulated_yaw_rate = 0.0
                self.simulated_speed = 1.5
        else:
            self.simulated_yaw_rate = 0.0
            self.simulated_speed = 1.5
        
        gyro = np.array([0.01, 0.02, self.simulated_yaw_rate])
        
        if self.motion_test_type == "approaching":
            accel = np.array([0.1, 0.2 + (self.simulated_speed - 1.5) * 0.5, 9.8])
        elif self.motion_test_type == "lateral":
            accel = np.array([0.1 + np.sin(timestamp) * 0.3, 0.2, 9.8])
        else:
            accel = np.array([0.1, 0.2, 9.8])
        
        cy = np.cos(self.simulated_yaw / 2)
        sy = np.sin(self.simulated_yaw / 2)
        quaternion = np.array([cy, 0.0, 0.0, sy])
        
        return (gyro, accel, quaternion)

    def draw_overlays(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw ARGUS overlays on frame."""
        overlay_frame = frame.copy()
        
        # Draw detection boxes
        for detection in results['detections']:
            x, y, w, h = detection.bbox
            label = detection.label
            confidence = detection.confidence
            
            color = self.colors.get(label.lower(), self.colors['default'])
            
            cv2.rectangle(overlay_frame, (int(x), int(y)), 
                        (int(x + w), int(y + h)), color, 2)
            
            label_text = f"{label}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(overlay_frame, (int(x), int(y - text_height - 10)),
                        (int(x + text_width), int(y)), color, -1)
            cv2.putText(overlay_frame, label_text, (int(x), int(y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw risk assessments
        for risk in results['risk_assessments']:
            for detection in results['detections']:
                if detection.object_id == risk.object_id:
                    x, y, w, h = detection.bbox
                    
                    risk_level = 'safe'
                    if risk.risk > 0.7:
                        risk_level = 'high'
                    elif risk.risk > 0.4:
                        risk_level = 'medium'
                    elif risk.risk > 0.2:
                        risk_level = 'low'
                    
                    risk_color = self.risk_colors.get(risk_level, self.risk_colors['safe'])
                    
                    cv2.rectangle(overlay_frame, (int(x), int(y + h)),
                                (int(x + w), int(y + h + 30)), risk_color, -1)
                    
                    risk_text = f"Risk: {risk.risk:.2f}"
                    cv2.putText(overlay_frame, risk_text, (int(x + 5), int(y + h + 20)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    if risk.ttc_s:
                        ttc_text = f"TTC: {risk.ttc_s:.1f}s"
                        cv2.putText(overlay_frame, ttc_text, (int(x + 5), int(y + h + 45)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    break
        
        # Draw system info
        cv2.putText(overlay_frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        detection_count = len(results['detections'])
        cv2.putText(overlay_frame, f"Detections: {detection_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.use_advanced and results['system_status']:
            status = results['system_status']
            cv2.putText(overlay_frame, f"Objects: {status['num_tracked_objects']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay_frame, f"Ego Speed: {status['ego_speed']:.1f} m/s", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if status['is_turning']:
                cv2.putText(overlay_frame, f"Turning: {status['turn_direction']}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw motion test info
        if self.motion_test_active:
            elapsed = time.time() - self.test_start_time if self.test_start_time else 0
            cv2.putText(overlay_frame, f"Motion Test: {self.motion_test_type}", (10, overlay_frame.shape[0] - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(overlay_frame, f"Elapsed: {elapsed:.1f}s", (10, overlay_frame.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(overlay_frame, "Press 's' to stop", (10, overlay_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return overlay_frame

    def update_dashboard(self, results: dict, overlay_frame: np.ndarray) -> None:
        """Update dashboard with current data."""
        # Store current frame for video streaming
        with self.current_frame_lock:
            self.current_frame = overlay_frame.copy()
        
        # Update data streamer
        self.data_streamer.add_detection(results['detections'])
        
        for risk in results['risk_assessments']:
            self.data_streamer.add_risk_event({
                'object_id': risk.object_id,
                'risk': risk.risk,
                'direction': risk.direction,
                'ttc_s': risk.ttc_s,
                'timestamp': time.time()
            })
        
        status = {
            'fps': self.fps,
            'active_objects': len(results['object_states']) if self.use_advanced else len(results['detections']),
            'total_detections': self.frame_count,
            'risk_level': self._get_overall_risk(results['risk_assessments']),
            'uptime_s': time.time() - self.start_time,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'camera_active': True,
            'motion_test_active': self.motion_test_active,
            'motion_test_type': self.motion_test_type
        }
        
        self.data_streamer.update_status(status)

    def _get_overall_risk(self, risk_assessments: list) -> str:
        """Get overall risk level."""
        if not risk_assessments:
            return "LOW"
        
        max_risk = max(r.risk for r in risk_assessments)
        
        if max_risk > 0.7:
            return "HIGH"
        elif max_risk > 0.4:
            return "MEDIUM"
        elif max_risk > 0.2:
            return "LOW"
        else:
            return "SAFE"

    def start_motion_test(self, test_type: str) -> None:
        """Start a motion test."""
        self.motion_test_active = True
        self.motion_test_type = test_type
        self.test_start_time = time.time()
        self.motion_history = []
        print(f"Starting motion test: {test_type}")

    def stop_motion_test(self) -> None:
        """Stop current motion test."""
        if self.motion_test_active:
            elapsed = time.time() - self.test_start_time
            print(f"Motion test stopped: {self.motion_test_type} ({elapsed:.1f}s)")
            print(f"Motion history: {len(self.motion_history)} samples")
        
        self.motion_test_active = False
        self.motion_test_type = None
        self.test_start_time = None

    def run_camera_loop(self) -> None:
        """Run the camera processing loop."""
        print("Starting camera loop...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                timestamp = time.time()
                
                # Process frame
                results = self.process_frame(frame, timestamp)
                
                # Draw overlays
                if self.enable_overlay:
                    display_frame = self.draw_overlays(frame, results)
                else:
                    display_frame = frame
                
                # Update dashboard
                self.update_dashboard(results, display_frame)
                
                # Record motion data if test active
                if self.motion_test_active and results['system_status']:
                    self.motion_history.append({
                        'timestamp': timestamp,
                        'ego_speed': results['system_status']['ego_speed'],
                        'ego_yaw': results['system_status']['ego_yaw'],
                        'objects': results['system_status']['num_tracked_objects']
                    })
                
                # Calculate FPS
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
                
                # Display frame locally
                cv2.imshow('ARGUS Live Camera', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.start_motion_test("approaching")
                elif key == ord('2'):
                    self.start_motion_test("turning")
                elif key == ord('3'):
                    self.start_motion_test("lateral")
                elif key == ord('s'):
                    self.stop_motion_test()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Camera loop stopped")

    def run(self) -> None:
        """Run the complete system with dashboard server."""
        print("=" * 60)
        print("Live Camera with Dashboard Server")
        print("=" * 60)
        print(f"Camera source: {self.camera_source}")
        print(f"Advanced pipeline: {self.use_advanced}")
        print(f"Overlays enabled: {self.enable_overlay}")
        print(f"Dashboard port: {self.port}")
        print()
        print("Dashboard will be available at: http://localhost:8080")
        print()
        print("Controls:")
        print("  'q' - Quit")
        print("  '1' - Start approaching motion test")
        print("  '2' - Start turning motion test")
        print("  '3' - Start lateral motion test")
        print("  's' - Stop motion test")
        print("=" * 60)
        print()
        
        # Start dashboard server in background thread
        import aiohttp
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        def run_server():
            aiohttp.web.run_app(self.dashboard_server.app, host='0.0.0.0', port=self.port)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print("Dashboard server started on background thread")
        print("Starting camera loop...")
        print()
        
        # Run camera loop
        self.run_camera_loop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live camera with dashboard server")
    parser.add_argument("--camera", type=int, default=0, help="Camera source (default: 0)")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--no-advanced", action="store_true", help="Use basic pipeline")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlays")
    
    args = parser.parse_args()
    
    try:
        system = LiveCameraWithDashboard(
            camera_source=args.camera,
            port=args.port,
            use_advanced=not args.no_advanced,
            enable_overlay=not args.no_overlay
        )
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()