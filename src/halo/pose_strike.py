"""Pose-based strike-motion evidence for HALO's reflex path.

Runs a lightweight pose model on a small budget: only close-range persons
(bbox height above a fraction of frame height — a strike only matters at
arm's length), at most MAX_SUBJECTS per frame. Extracts wrist speed and
elbow-extension rate from COCO keypoints and emits a 0..1
"rapid strike-like motion" evidence value per track.

This detects the observable event — fast limb extension toward the wearer —
not intent. The value feeds score_track() as the 'strike_motion' channel.
"""

from __future__ import annotations

import math
from collections import deque

# COCO-17 indices
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10

_MAX_SUBJECTS = 2          # pose is expensive on CPU — spend it on the closest
_MIN_BBOX_FRAC = 0.22      # bbox height / frame height — farther people can't reach us
_WINDOW_S = 0.45           # short window: a strike is a burst, not a trend
_CONF_MIN = 0.3


class PoseStrike:
    """Per-track strike-motion evidence from pose keypoints."""

    def __init__(self, model_path: str = "yolo11n-pose.pt", device: str | None = None):
        self._model_path = model_path
        self._model = None
        self._disabled = False
        self._hist: dict[str, deque] = {}
        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._device = device

    def _ensure_model(self):
        if self._disabled:
            return None
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self._model_path)
            except Exception as e:
                print(f"PoseStrike disabled: {e}")
                self._disabled = True
        return self._model

    def update(self, frame, detections, ts_s: float) -> dict[str, float]:
        """Return {object_id: strike_motion 0..1} for close-range persons."""
        fh = frame.shape[0]
        persons = [d for d in detections
                   if d.label == "person" and d.bbox[3] > _MIN_BBOX_FRAC * fh]
        persons.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        persons = persons[:_MAX_SUBJECTS]
        if not persons:
            self._prune(set(), ts_s)
            return {}
        model = self._ensure_model()
        if model is None:
            return {}

        out: dict[str, float] = {}
        live = set()
        for det in persons:
            live.add(det.object_id)
            x, y, w, h = det.bbox
            x0, y0 = max(int(x - 0.1 * w), 0), max(int(y - 0.05 * h), 0)
            x1 = min(int(x + 1.1 * w), frame.shape[1])
            y1 = min(int(y + 1.05 * h), frame.shape[0])
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            try:
                res = model(crop, verbose=False, device=self._device)[0]
            except Exception:
                continue
            if res.keypoints is None or len(res.keypoints.xy) == 0:
                continue
            # largest person in the crop (should be our subject)
            kxy = res.keypoints.xy.cpu().numpy()
            kconf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
            idx = max(range(len(kxy)),
                      key=lambda i: float(kconf[i].sum()) if kconf is not None else 0)
            kp, kc = kxy[idx], (kconf[idx] if kconf is not None else None)

            def pt(i):
                if kc is not None and kc[i] < _CONF_MIN:
                    return None
                return (float(kp[i][0]) + x0, float(kp[i][1]) + y0)

            sample = {
                "ts": ts_s,
                "wl": pt(_L_WRIST), "wr": pt(_R_WRIST),
                "el_l": _elbow_angle(pt(_L_SHOULDER), pt(_L_ELBOW), pt(_L_WRIST)),
                "el_r": _elbow_angle(pt(_R_SHOULDER), pt(_R_ELBOW), pt(_R_WRIST)),
            }
            hist = self._hist.setdefault(det.object_id, deque(maxlen=20))
            hist.append(sample)
            out[det.object_id] = self._score(hist, ts_s)

        self._prune(live, ts_s)
        return out

    def _score(self, hist: deque, ts_s: float) -> float:
        recent = [s for s in hist if ts_s - s["ts"] < _WINDOW_S]
        if len(recent) < 2:
            return 0.0
        s0, s1 = recent[0], recent[-1]
        dt = max(s1["ts"] - s0["ts"], 1e-3)
        wrist_speed = 0.0
        for side in ("wl", "wr"):
            if s0[side] and s1[side]:
                wrist_speed = max(wrist_speed,
                                  math.hypot(s1[side][0] - s0[side][0],
                                             s1[side][1] - s0[side][1]) / dt)
        ext_rate = 0.0
        for side in ("el_l", "el_r"):
            if s0[side] is not None and s1[side] is not None:
                ext_rate = max(ext_rate, (s1[side] - s0[side]) / dt)
        # wrist burst ~600+ px/s at phone resolution; elbow extends ~2-4 rad/s
        return (0.55 * _clamp01(wrist_speed / 600.0)
                + 0.45 * _clamp01(ext_rate / 2.5))

    def _prune(self, live: set[str], ts_s: float) -> None:
        for oid in list(self._hist):
            h = self._hist[oid]
            if oid not in live and (not h or ts_s - h[-1]["ts"] > 2.0):
                self._hist.pop(oid, None)

    def reset(self) -> None:
        self._hist.clear()


def _elbow_angle(shoulder, elbow, wrist) -> float | None:
    """Angle at the elbow in radians (pi = fully extended)."""
    if not shoulder or not elbow or not wrist:
        return None
    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    n1 = math.hypot(*v1) or 1e-6
    n2 = math.hypot(*v2) or 1e-6
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.acos(cos)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
