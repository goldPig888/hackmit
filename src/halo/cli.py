from __future__ import annotations

import argparse
import time
from pathlib import Path

from .haptics import HapticPublisher
from .models import Detection
from .pipeline import HaloPipeline


def demo(output: Path) -> None:
    pipeline, publisher = HaloPipeline(), HapticPublisher(output)
    print("HALO simulated crossing scenario (rider: 4 m/s, car approaches rear-left)")
    for frame in range(45):
        timestamp = frame / 10
        # Object grows and moves rightward toward the rider's projected path.
        size = 30 + frame * 2.7
        detection = Detection("car-12", "car", 0.94, (260 + frame * 8, 360 - frame * 2, size, size * .72), timestamp)
        result = pipeline.update(detection, rider_speed_mps=4.0, yaw_rate_rps=0.0)
        if result:
            event = publisher.publish(result)
            print(f"t={timestamp:>4.1f}s  {result.object_id:7} risk={result.risk:.0%}  side={result.direction:6} "
                  f"closest={result.closest_distance_m:.2f}m  warning={event.intensity if event else '-'}")
    print(f"Events: {output / 'haptics.jsonl'}")


def camera(source: str, output: Path) -> None:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Camera mode requires ./scripts/02_install.sh") from error
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera/video source: {source}")
    model, pipeline, publisher = YOLO("yolo11n.pt"), HaloPipeline(), HapticPublisher(output)
    started = time.monotonic()
    allowed = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        now = time.monotonic() - started
        if result.boxes.id is not None:
            names = result.names
            for box, track_id, cls, confidence in zip(result.boxes.xywh.cpu().tolist(), result.boxes.id.int().cpu().tolist(), result.boxes.cls.int().cpu().tolist(), result.boxes.conf.cpu().tolist()):
                label = names[cls]
                if label not in allowed:
                    continue
                x, y, w, h = box
                assessment = pipeline.update(Detection(f"{label}-{track_id}", label, confidence, (x-w/2, y-h/2, w, h), now))
                if assessment:
                    publisher.publish(assessment)
        cv2.imshow("HALO — press q to stop", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="HALO safety prototype")
    sub = parser.add_subparsers(required=True, dest="command")
    demo_parser = sub.add_parser("demo")
    demo_parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    camera_parser = sub.add_parser("camera")
    camera_parser.add_argument("--source", default="0")
    camera_parser.add_argument("--output", type=Path, default=Path("runs/camera"))
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output)
    else:
        camera(args.source, args.output)
