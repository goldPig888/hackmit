#!/usr/bin/env python3
"""HALO live pipeline fed by an iPhone camera + IMU stream.

Start this, expose the dashboard over HTTPS (Cloudflare tunnel), then open
https://<tunnel-host>/phone on the iPhone and tap Start. The phone streams
JPEG frames and DeviceMotion/DeviceOrientation data over /ingest; this loop
decodes frames, runs detection, and feeds the REAL IMU into the advanced
pipeline (rotation compensation, world-frame tracking, CPA).

Dashboard (also shows annotated video at /video):
    http://localhost:8080
"""

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from aiohttp import web

from halo import AdvancedHaloPipeline, AdvancedPipelineConfig
from halo.dashboard import DataStreamer, DashboardServer
from halo.dashboard.data_streamer import DetectionEvent, HazardEvent, SystemStatus
from halo.dashboard.world_state import WorldStateBuilder
from halo.detectors import YOLODetector
from halo.haptics import HapticPublisher
from halo.imu.camera_pose import CameraIntrinsics
from halo.pose_strike import PoseStrike


class IPhoneHaloProcessor:
    """CV loop: phone frames -> detection -> advanced pipeline with real IMU."""

    def __init__(self, server, streamer, loop, model_path: str,
                 hfov_deg: float, output_dir: Path):
        self.server = server
        self.receiver = server.phone_receiver
        self.streamer = streamer
        self.loop = loop
        self.output_dir = output_dir

        config = AdvancedPipelineConfig(
            enable_imu_compensation=True,
            enable_visual_odometry=False,   # real IMU makes this unnecessary
            enable_ekf=True,
            enable_cpa_collision=True,
        )
        self.pipeline = AdvancedHaloPipeline(config)
        self.world_builder = WorldStateBuilder()
        tracker_name = os.environ.get("HALO_TRACKER", "botsort_reid")
        tracker_cfg = Path(__file__).resolve().parent.parent / "config" / f"{tracker_name}.yaml"
        self.detector = YOLODetector(
            model_path=model_path,
            tracker_config=str(tracker_cfg) if tracker_cfg.exists() else "bytetrack.yaml")
        self.haptics = HapticPublisher(output_dir)
        self.pose_strike = PoseStrike() if os.environ.get("HALO_POSE", "1") != "0" else None
        self._record = os.environ.get("HALO_RECORD", "1") != "0"
        self._run_file = None

        self.hfov_deg = hfov_deg
        self._intrinsics_set = False

        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self._last_status_push = 0.0
        self._last_cleanup = 0.0
        self._clear_requested = threading.Event()

    def _record_frame(self, state: dict, ts_s: float) -> None:
        """Append compact per-frame features to runs/run_*.jsonl for /training replay."""
        if not self._record or not state.get("tracks"):
            return
        try:
            if self._run_file is None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._run_file = self.output_dir / f"run_{int(time.time())}.jsonl"
            rec = {
                "t": float(ts_s),
                "context": {"density": state.get("scene", {}).get("density"),
                            "environment": state.get("scene", {}).get("environment")},
                "tracks": [{
                    "id": tr["id"], "position": tr["position"],
                    "closing_rate": tr["closing_rate"],
                    "bearing_rate": tr["bearing_rate"],
                    "area_rate": tr.get("area_rate", 0.0),
                    "tcpa": tr["tcpa"], "dcpa": tr["dcpa"],
                    "will_collide": tr["will_collide"],
                    "evidence": tr.get("evidence", {}),
                    "halo_action": tr["att_state"] if tr["att_state"] != "REFLEX" else "WARN",
                } for tr in state["tracks"]],
            }
            with self._run_file.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def request_clear_subjects(self) -> None:
        """Schedule a reset; the CV thread performs it between frames."""
        self._clear_requested.set()

    def _clear_subjects(self) -> None:
        self.pipeline.reset()
        self.detector.reset()
        if self.pose_strike:
            self.pose_strike.reset()
        self.world_builder.reset()
        self._intrinsics_set = False
        self._last_cleanup = 0.0
        self._clear_requested.clear()
        print("Subjects cleared; acquiring a fresh tracking epoch.")

    def run(self) -> None:
        print("Waiting for phone stream... (open /phone on the iPhone)")
        while True:
            item = self.receiver.take_latest_frame()
            if item is None:
                time.sleep(0.01)
                continue
            try:
                if self._clear_requested.is_set():
                    self._clear_subjects()
                self._process(item[0], item[1])
            except Exception as e:
                print(f"Frame processing error: {e}")

    def _process(self, ts_s: float, jpeg: bytes) -> None:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._ensure_intrinsics(frame)

        detections = self.detector.detect(frame, ts_s)
        # Pose reflex path: strike-motion evidence, only on close-range persons
        strike_ev = (self.pose_strike.update(frame, detections, ts_s)
                     if self.pose_strike else {})
        imu = self.receiver.imu_tuple_at(ts_s)
        imu_status = "imu-ok" if imu else "NO-IMU"

        assessments = []
        for det in detections:
            if imu is None:
                continue  # advanced pipeline needs a camera pose
            risk = self.pipeline.update(det, imu_data=imu)
            if risk is not None:
                assessments.append((det, risk))
                event = self.haptics.publish(risk)
                if event:
                    self.world_builder.notify_haptic(event.direction, event.intensity, ts_s)
                    print(f"RISK {risk.direction:>6} {risk.risk:.0%} "
                          f"{risk.object_id} closest={risk.closest_distance_m:.2f}m")

        self._push_detections(detections, assessments)
        self._draw_and_publish(frame, detections, assessments, imu_status)

        state = self.world_builder.update(
            pipeline=self.pipeline,
            detections=detections,
            assessments=[a for _, a in assessments],
            receiver=self.receiver,
            fps=self.fps,
            haptic_endpoint_set=self.haptics.endpoint is not None,
            frame_size=(frame.shape[1], frame.shape[0]),
            ts_s=ts_s,
            brightness=float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()),
            strike_evidence=strike_ev,
        )
        self.server.set_world_state(state)
        self._record_frame(state, ts_s)

        # Reflex path: immediate-motion threats bypass risk gating entirely —
        # strong directional haptic on every reflex frame.
        threat = state.get("threat") or {}
        if threat.get("att_state") == "REFLEX":
            event = self.haptics.publish_reflex(
                threat.get("direction", "center"), threat.get("id", "?"))
            if event:
                self.world_builder.notify_haptic(event.direction, event.intensity, ts_s)

        self.frame_count += 1
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0.0

        # cleanup_old_objects must use the phone clock, not wall time
        if ts_s - self._last_cleanup > 5.0:
            self.pipeline.world_tracker.cleanup_old_objects(ts_s)
            self._last_cleanup = ts_s

        if time.time() - self._last_status_push > 1.0:
            self._push_status(assessments)
            self._last_status_push = time.time()

    def _ensure_intrinsics(self, frame: np.ndarray) -> None:
        if self._intrinsics_set:
            return
        h, w = frame.shape[:2]
        fx = fy = w / (2.0 * np.tan(np.radians(self.hfov_deg) / 2.0))
        self.pipeline.camera_pose_estimator.intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0, width=w, height=h)
        print(f"Intrinsics set for {w}x{h}, hfov={self.hfov_deg}deg, f={fx:.0f}px")
        self._intrinsics_set = True

    def _push_detections(self, detections, assessments) -> None:
        risk_by_id = {a.object_id: (d, a) for d, a in assessments}
        for det in detections:
            pair = risk_by_id.get(det.object_id)
            event = DetectionEvent(
                object_id=det.object_id,
                label=det.label,
                confidence=det.confidence,
                bbox=det.bbox,
                timestamp_s=det.timestamp_s,
                risk_score=pair[1].risk if pair else None,
                ttc_s=pair[1].ttc_s if pair else None,
                direction=pair[1].direction if pair else None,
            )
            self._fire(self.streamer.add_detection(event))
        for det, risk in assessments:
            if risk.risk > 0.4:
                self._fire(self.streamer.add_hazard(HazardEvent(
                    hazard_type="collision",
                    severity=risk.risk,
                    description=f"{risk.label or det.label} {risk.direction} "
                                f"CPA={risk.closest_distance_m:.2f}m",
                    involved_objects=[risk.object_id],
                    timestamp_s=det.timestamp_s,
                )))

    def _push_status(self, assessments) -> None:
        max_risk = max((a.risk for _, a in assessments), default=0.0)
        risk_level = ("HIGH" if max_risk > 0.7 else
                      "MEDIUM" if max_risk > 0.4 else
                      "LOW" if max_risk > 0.2 else "SAFE")
        self._fire(self.streamer.update_status(SystemStatus(
            fps=self.fps,
            active_objects=len(self.pipeline.get_all_object_states()),
            total_detections=self.frame_count,
            risk_level=risk_level,
            uptime_s=time.time() - self.start_time,
            timestamp=time.strftime("%H:%M:%S"),
        )))

    def _fire(self, coro) -> None:
        try:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        except RuntimeError:
            pass

    def _draw_and_publish(self, frame, detections, assessments, imu_status) -> None:
        risk_by_id = {a.object_id: a for _, a in assessments}
        for det in detections:
            x, y, w, h = det.bbox
            risk = risk_by_id.get(det.object_id)
            if risk is None:
                color = (0, 255, 0)
                text = f"{det.label} {det.confidence:.2f}"
            else:
                color = ((0, 0, 255) if risk.risk > 0.7 else
                         (0, 165, 255) if risk.risk > 0.4 else (0, 255, 255))
                text = f"{det.label} risk={risk.risk:.2f} {risk.direction}"
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(frame, text, (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS {self.fps:.1f} | {imu_status}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Bake a prominent status card into the JPEG itself. This is separate
        # from the browser canvas so it remains visible in /video and on the
        # console if a browser layout or canvas rendering issue occurs.
        featured_det = None
        featured_risk = None
        if assessments:
            featured_det, featured_risk = max(assessments, key=lambda pair: pair[1].risk)
        elif detections:
            featured_det = max(detections, key=lambda item: item.confidence)
        if featured_det is not None:
            risk_value = float(featured_risk.risk) if featured_risk else 0.0
            is_conflict = risk_value >= 0.30
            is_high = risk_value >= 0.70
            accent = (0, 0, 235) if is_high else ((0, 185, 255) if is_conflict else (255, 220, 0))
            title = "PREDICTED CONFLICT" if is_conflict else "TRACKING"
            direction = featured_risk.direction.upper() if featured_risk else "PATH ACTIVE"
            card_w = min(frame.shape[1] - 24, max(360, int(frame.shape[1] * 0.52)))
            card_h = min(frame.shape[0] - 40, max(108, int(frame.shape[0] * 0.18)))
            overlay = frame.copy()
            cv2.rectangle(overlay, (12, 38), (12 + card_w, 38 + card_h), (10, 12, 18), -1)
            cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
            cv2.rectangle(frame, (12, 38), (12 + card_w, 38 + card_h), accent, 3)
            cv2.putText(frame, title, (28, 72), cv2.FONT_HERSHEY_DUPLEX, 0.78, accent, 2)
            label = featured_det.label.upper()
            cv2.putText(frame, f"{label}  {direction}  {risk_value:.0%}", (28, 108),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (245, 245, 245), 2)
            if featured_risk and featured_risk.cpa_result.time_to_cpa is not None:
                cv2.putText(frame, f"CPA {featured_risk.cpa_result.time_to_cpa:.1f}s", (28, 137),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, accent, 2)

        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.receiver.set_processed_frame(enc.tobytes())


async def main_async(args) -> None:
    streamer = DataStreamer()
    server = DashboardServer(streamer, port=args.port)
    runner = web.AppRunner(server.app)
    await runner.setup()
    try:
        await web.TCPSite(runner, server.host, args.port).start()
    except OSError as error:
        await runner.cleanup()
        if error.errno in {48, 98, 10048}:
            raise RuntimeError(
                f"Port {args.port} is already in use. HALO is likely still running; "
                f"open http://localhost:{args.port}/demo, stop that process, or restart on "
                f"another port with --port {args.port + 1}.") from None
        raise

    loop = asyncio.get_running_loop()
    processor = IPhoneHaloProcessor(
        server=server,
        streamer=streamer,
        loop=loop,
        model_path=args.model,
        hfov_deg=args.hfov,
        output_dir=args.output,
    )
    server.set_clear_subjects_handler(processor.request_clear_subjects)

    print("=" * 60)
    print("HALO iPhone Stream")
    print("=" * 60)
    print(f"Dashboard:  http://localhost:{args.port}")
    print(f"Phone page: http://localhost:{args.port}/phone")
    print()
    print("iOS requires HTTPS for camera+motion. Expose this server with the")
    print("Cloudflare tunnel, then open https://<tunnel-host>/phone on the")
    print("iPhone and tap Start HALO.")
    print("=" * 60)

    await loop.run_in_executor(None, processor.run)


def main() -> None:
    parser = argparse.ArgumentParser(description="HALO iPhone camera stream")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--hfov", type=float, default=75.0,
                        help="Approximate horizontal FOV of the streamed image (deg)")
    parser.add_argument("--output", type=Path, default=Path("runs/iphone"))
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nStopped")
    except RuntimeError as error:
        print(f"\nCannot start HALO: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
