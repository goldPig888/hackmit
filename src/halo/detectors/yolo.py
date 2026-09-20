"""YOLO detector implementation for HALO."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Detection

from .base import BaseDetector


class YOLODetector(BaseDetector):
    """YOLO-based detector using ultralytics package.
    
    Wraps YOLO for compatibility with HALO detector plugin system.
    """

    def __init__(self, model_path: str = "yolo11n.pt", tracker_config: str = "bytetrack.yaml",
                 allowed_classes: set[str] | None = None, device: str | None = None):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            tracker_config: Path to tracker configuration
            allowed_classes: Set of class names to detect (None = all classes)
        """
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("YOLO detector requires ultralytics package. Install with: pip install ultralytics") from error
        
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.allowed_classes = allowed_classes or {
            "car", "truck", "bus", "motorcycle", "bicycle", "person",
            # hand-held objects worth flagging when carried by a person
            "scissors", "knife", "baseball bat", "tennis racket",
            "bottle", "umbrella",
        }
        self._supported_classes = set(self.model.names.values())
        # Prefer Apple-Silicon GPU; ~2x faster than CPU for n-class models.
        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        # Compact per-epoch display IDs: ByteTrack's internal counter keeps
        # climbing forever, so map raw IDs onto 0..N and restart on reset().
        self._id_map: dict[int, int] = {}
        # Selective ReID: only the closest N detections (by bbox height) — or
        # detections showing suspicious image motion (fast expansion / fast
        # displacement) — get appearance embeddings. Env-tunable for demos.
        self._reid_top_n = int(os.environ.get("HALO_REID_TOPN", "4"))
        self._reid_min_h = int(os.environ.get("HALO_REID_MINH", "60"))
        self._reid_extra = int(os.environ.get("HALO_REID_EXTRA", "4"))

    def _patch_selective_reid(self) -> None:
        """Wrap the BoT-SORT encoder so only near or suspicious subjects get ReID.

        In `model: auto` mode the encoder just converts backbone features —
        masking keeps appearance matching focused on close subjects (big
        bboxes) plus anything whose box grew or moved fast vs last frame,
        which are the image-space signatures of approach/lunging. Distant
        steady tracks run IoU+Kalman only.
        """
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        if not trackers:
            return
        tracker = trackers[0]
        encoder = getattr(tracker, "encoder", None)
        if encoder is None or getattr(tracker, "_halo_selective", False):
            return
        top_n, min_h, extra = self._reid_top_n, self._reid_min_h, self._reid_extra
        state = {"prev": None}

        def _iou(a, b):
            ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
            bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
            iw = max(0.0, min(ax1 + a[2], bx1 + b[2]) - max(ax1, bx1))
            ih = max(0.0, min(ay1 + a[3], by1 + b[3]) - max(ay1, by1))
            inter = iw * ih
            return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-6)

        def selective(feats, dets, _enc=encoder):
            out = _enc(feats, dets)
            if len(out) != len(dets):
                return out
            order = sorted(range(len(dets)), key=lambda i: -float(dets[i][3]))
            keep = {i for i in order[:top_n] if float(dets[i][3]) >= min_h}
            prev = state["prev"]
            if prev is not None:
                for i, d in enumerate(dets):
                    best, bi = 0.2, -1
                    for j, p in enumerate(prev):
                        ov = _iou(d, p)
                        if ov > best:
                            best, bi = ov, j
                    if bi < 0:
                        continue
                    p = prev[bi]
                    growth = float(d[3]) / max(float(p[3]), 1e-3)
                    move = ((float(d[0]) - float(p[0])) ** 2
                            + (float(d[1]) - float(p[1])) ** 2) ** 0.5
                    if growth >= 1.15 or move >= 90:
                        keep.add(i)
            state["prev"] = [tuple(d[:4]) for d in dets]
            if len(keep) > top_n + extra:      # bound the compute budget
                keep = set(sorted(keep, key=lambda i: -float(dets[i][3]))[:top_n + extra])
            return [f if i in keep else None for i, f in enumerate(out)]

        tracker.encoder = selective
        tracker._halo_selective = True

    def reset(self) -> None:
        """Restart tracking epoch — next detection becomes ID 0."""
        self._id_map.clear()
        try:
            from ultralytics.trackers.basetrack import BaseTrack
            BaseTrack.reset_id()
            if getattr(self.model, "predictor", None) is not None:
                self.model.predictor = None
        except Exception:
            pass

    def detect(self, frame, timestamp_s: float) -> list[Detection]:
        """Process frame with YOLO detection and tracking.
        
        Args:
            frame: Input frame (numpy array)
            timestamp_s: Current timestamp in seconds
            
        Returns:
            List of Detection objects
        """
        import numpy as np
        
        from ..models import Detection
        
        result = self.model.track(frame, persist=True, tracker=self.tracker_config,
                                  verbose=False, device=self.device)[0]
        self._patch_selective_reid()
        detections = []

        # which live tracks hold an appearance embedding this frame
        reid_ids: set[int] = set()
        try:
            trk = self.model.predictor.trackers[0]
            reid_ids = {
                t.track_id for t in getattr(trk, "tracked_stracks", [])
                if getattr(t, "curr_feat", None) is not None
                or getattr(t, "smooth_feat", None) is not None
            }
        except Exception:
            pass

        if result.boxes.id is not None:
            names = result.names
            for box, track_id, cls, confidence in zip(
                result.boxes.xywh.cpu().tolist(),
                result.boxes.id.int().cpu().tolist(),
                result.boxes.cls.int().cpu().tolist(),
                result.boxes.conf.cpu().tolist()
            ):
                label = names[cls]
                if label not in self.allowed_classes:
                    continue
                
                x, y, w, h = box
                compact_id = self._id_map.setdefault(track_id, len(self._id_map))
                object_id = f"{label}-{compact_id}"
                detections.append(Detection(
                    object_id=object_id,
                    label=label,
                    confidence=float(confidence),
                    bbox=(float(x - w/2), float(y - h/2), float(w), float(h)),
                    timestamp_s=timestamp_s,
                    reid=track_id in reid_ids,
                ))
        
        return detections

    def get_supported_classes(self) -> set[str]:
        """Return classes this YOLO model can detect."""
        return self._supported_classes

    def cleanup(self) -> None:
        """Release YOLO model resources."""
        # YOLO models are automatically cleaned up by Python garbage collection
        pass