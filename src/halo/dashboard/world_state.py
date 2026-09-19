"""Adapter: unified world-state snapshot for the predictive safety console.

Translates existing pipeline state (ObjectState, EgoTracker,
CollisionRiskAssessment, RotationCompensator) into the JSON model the console
UI consumes. No safety math lives here beyond thin derivations (TTC estimate,
counterfactual path comparison); authoritative risk comes from the pipeline's
own CollisionRiskAssessor.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

_INTENSITY = {"weak": 0.35, "medium": 0.65, "strong": 1.0}
_GHOST_TIMES = (0.5, 1.0, 1.5, 2.0)


def _euler_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) radians from a rotation matrix. Display only."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def _direction_label(bearing_rel: float) -> str:
    """Convert relative bearing (rad, ego-relative) to a compass-style label."""
    deg = math.degrees(bearing_rel)
    deg = (deg + 180.0) % 360.0 - 180.0
    a = abs(deg)
    side = "RIGHT" if deg > 0 else "LEFT"
    if a < 30:
        return "AHEAD"
    if a > 150:
        return "BEHIND"
    if a < 70:
        return f"FRONT {side}"
    if a > 110:
        return f"REAR {side}"
    return side


class WorldStateBuilder:
    """Builds unified console state; owns event history and per-track trails."""

    def __init__(self, horizon_s: float = 4.0) -> None:
        self.horizon_s = horizon_s
        self._pos_hist: dict[str, deque] = {}
        self._px_hist: dict[str, deque] = {}
        self._area_hist: dict[str, deque] = {}
        self._events: deque = deque(maxlen=120)
        self._event_seq = 0
        self._risk_hist: deque = deque(maxlen=200)
        self._known_ids: set[str] = set()
        self._closing: dict[str, bool] = {}
        self._conflict: dict[str, bool] = {}
        self._cpa_flagged: dict[str, bool] = {}
        self._was_turning = False
        self._haptic = {"left": 0.0, "right": 0.0, "center": 0.0,
                        "direction": None, "intensity": None, "ts": 0.0}

    # ---- inputs from the processing loop ----

    def notify_haptic(self, direction: str, intensity: str, ts_s: float) -> None:
        v = _INTENSITY.get(intensity, 0.5)
        self._haptic = {
            "left": v if direction == "left" else (v * 0.5 if direction == "center" else 0.0),
            "right": v if direction == "right" else (v * 0.5 if direction == "center" else 0.0),
            "center": v if direction == "center" else 0.0,
            "direction": direction, "intensity": intensity, "ts": ts_s,
        }
        self._emit("HAPTIC ISSUED", ts_s, f"{direction} / {intensity}")

    def reset(self) -> None:
        """Clear visual histories, selected threats, events, and haptic state."""
        self._pos_hist.clear()
        self._px_hist.clear()
        self._area_hist.clear()
        self._events.clear()
        self._risk_hist.clear()
        self._known_ids.clear()
        self._closing.clear()
        self._conflict.clear()
        self._cpa_flagged.clear()
        self._was_turning = False
        self._haptic = {"left": 0.0, "right": 0.0, "center": 0.0,
                        "direction": None, "intensity": None, "ts": 0.0}

    def _emit(self, etype: str, ts: float, label: str,
              object_id: Optional[str] = None, risk: Optional[float] = None) -> None:
        self._event_seq += 1
        self._events.append({"id": self._event_seq, "ts": float(ts), "type": etype,
                             "label": label, "object_id": object_id, "risk": risk})

    # ---- projection helpers ----

    def _world_to_pixel(self, pipeline, point: np.ndarray) -> Optional[list[float]]:
        """Project a world-frame point back through the current camera pose."""
        pose = pipeline.camera_pose_estimator.get_current_pose()
        if pose is None:
            return None
        x, y, z = float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)
        h = math.hypot(x, y)
        r_w = np.array([x, y, z]) / max(h, 1e-6)
        r_c = pose.rotation_matrix.T @ r_w
        if r_c[2] < 0.05:
            return None
        uv = pipeline.camera_pose_estimator.intrinsics.K @ (r_c / r_c[2])
        if not np.all(np.isfinite(uv[:2])):
            return None
        return [float(uv[0]), float(uv[1])]

    # ---- main entry ----

    def update(self, pipeline, detections, assessments, receiver,
               fps: float, haptic_endpoint_set: bool, frame_size, ts_s: float) -> dict:
        """Produce one console state snapshot. Call once per processed frame."""
        ego = pipeline.ego_tracker.get_current_state()
        objects = pipeline.get_all_object_states()
        bbox_by_id = {d.object_id: list(d.bbox) for d in detections}
        risk_by_id = {}
        for a in pipeline.get_risk_assessments():
            if a.object_id not in risk_by_id or a.probability > risk_by_id[a.object_id].probability:
                risk_by_id[a.object_id] = a

        # --- events from state transitions ---
        for obj in objects:
            if obj.object_id not in self._known_ids:
                self._emit("TRACK ACQUIRED", ts_s, f"{obj.label} {obj.object_id}", obj.object_id)
                self._known_ids.add(obj.object_id)
        stale = self._known_ids - {o.object_id for o in objects}
        self._known_ids -= stale
        for sid in stale:
            self._pos_hist.pop(sid, None)
            self._px_hist.pop(sid, None)
            self._area_hist.pop(sid, None)

        turning = bool(pipeline.ego_tracker.is_turning())
        if turning != self._was_turning:
            self._emit("RIDER TURN STARTED" if turning else "RIDER TURN ENDED",
                       ts_s, pipeline.ego_tracker.get_turn_direction() if turning else "straight")
        self._was_turning = turning

        # --- rider ---
        _, ego_path = pipeline.ego_tracker.get_predicted_path(self.horizon_s, 0.2)
        straight_ego = type(ego)(
            position=ego.position.copy(), velocity=ego.velocity.copy(),
            yaw=ego.yaw, yaw_rate=0.0, speed=ego.speed,
            timestamp_s=ego.timestamp_s, confidence=ego.confidence)
        straight_path = straight_ego.predict_future_path(self.horizon_s, 0.2)[1]

        tracks = []
        primary = None
        max_risk = 0.0
        funnel = {"perceived": len(detections), "tracked": len(objects),
                  "moving": 0, "closing": 0, "conflict": 0}

        for obj in objects:
            oid = obj.object_id
            pos = np.asarray(obj.position, dtype=float)
            vel = np.asarray(obj.velocity, dtype=float)
            speed = float(np.linalg.norm(vel[:2]))

            # histories
            self._pos_hist.setdefault(oid, deque(maxlen=80)).append((ts_s, float(pos[0]), float(pos[1])))
            bbox = bbox_by_id.get(oid)
            if bbox:
                cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                self._px_hist.setdefault(oid, deque(maxlen=40)).append((ts_s, cx, cy))
                self._area_hist.setdefault(oid, deque(maxlen=40)).append((ts_s, bbox[2] * bbox[3]))

            # motion channels
            bearing_rate = None
            vel_rate = pipeline.rotation_compensator.get_object_velocity(oid)
            if vel_rate is not None:
                bearing_rate = float(vel_rate[0])
            rel = pipeline.world_tracker.get_relative_motion(oid)
            closing_rate = float(rel.get("closing_rate", 0.0)) if "closing_rate" in rel else 0.0
            is_closing = closing_rate < -0.3
            is_moving = speed > 0.3
            funnel["moving"] += is_moving
            funnel["closing"] += is_closing
            if is_closing and not self._closing.get(oid):
                self._emit("CLOSING DETECTED", ts_s, f"{closing_rate:.1f} m/s", oid)
            self._closing[oid] = is_closing

            # prediction
            pred = pipeline.world_tracker.get_object_predictions(oid, self.horizon_s, 0.2)
            pred_times, pred_pos = [], []
            if pred is not None:
                pred_times = [float(t) for t in pred[0]]
                pred_pos = [[float(p[0]), float(p[1])] for p in pred[1]]
            px_pred = []
            if pred is not None:
                for p in pred[1]:
                    px = self._world_to_pixel(pipeline, p)
                    px_pred.append(px if px else None)

            # Phone pose calibration can temporarily make world re-projection
            # unavailable. Keep the visual prediction legible by extrapolating
            # the measured image-center motion in that case; this affects only
            # rendering, never the collision calculation.
            if pred is not None and bbox and (not px_pred or not any(px_pred)):
                history = self._px_hist.get(oid)
                if history and len(history) >= 2:
                    t0, u0, v0 = history[0]
                    t1, u1, v1 = history[-1]
                    dt = max(t1 - t0, 1e-3)
                    du, dv = (u1 - u0) / dt, (v1 - v0) / dt
                    px_pred = [[float(u1 + du * t), float(v1 + dv * t)] for t in pred[0]]

            # risk / CPA from the pipeline's own assessor
            assess = risk_by_id.get(oid)
            risk = float(assess.probability) if assess else 0.0
            tcpa = float(assess.cpa_result.time_to_cpa) if assess else None
            dcpa = float(assess.cpa_result.distance_at_cpa) if assess else None
            will_collide = bool(assess.cpa_result.will_collide) if assess else False
            conflict_type = assess.conflict_type if assess else "safe_pass"
            factors = assess.contributing_factors if assess else {}
            funnel["conflict"] += will_collide
            if will_collide and not self._conflict.get(oid):
                self._emit("PATH CONFLICT", ts_s, f"tCPA {tcpa:.1f}s", oid, risk)
            self._conflict[oid] = will_collide
            if will_collide and tcpa is not None and tcpa < 1.5 and not self._cpa_flagged.get(oid):
                self._emit("CPA IMMINENT", ts_s, f"dCPA {dcpa:.2f}m", oid, risk)
                self._cpa_flagged[oid] = True
            if not will_collide:
                self._cpa_flagged[oid] = False

            # conflict point = object predicted position at t_cpa
            conflict_point = None
            if will_collide and tcpa is not None and pred_times:
                idx = min(range(len(pred_times)), key=lambda i: abs(pred_times[i] - tcpa))
                conflict_point = pred_pos[idx]

            # sigma from covariance
            sigma = float(np.sqrt(max(np.trace(np.asarray(obj.covariance)[:3, :3]), 0.0) / 3.0))

            # ttc estimate from range/closing rate (derived, not authoritative)
            dist = float(np.linalg.norm(pos[:2]))
            ttc = dist / abs(closing_rate) if closing_rate < -0.05 else None

            # frame expansion rate for "why" evidence
            expanding = False
            ah = self._area_hist.get(oid)
            if ah and len(ah) >= 3:
                t0, a0 = ah[0]
                t1, a1 = ah[-1]
                if t1 > t0 and a0 > 0:
                    expanding = (a1 / a0 - 1.0) / (t1 - t0) > 0.15

            # ego-motion diagnostic channels
            raw_px_rate = None
            ph = self._px_hist.get(oid)
            if ph and len(ph) >= 2:
                (t0, u0, v0), (t1, u1, v1) = ph[0], ph[-1]
                if t1 > t0:
                    raw_px_rate = math.hypot(u1 - u0, v1 - v0) / (t1 - t0)

            # pixel trails (world history reprojected = stabilized; raw = px history)
            px_hist_pts = [[u, v] for _, u, v in ph] if ph else []
            stab_hist_pts = []
            for _, hx, hy in list(self._pos_hist.get(oid, [])):
                px = self._world_to_pixel(pipeline, np.array([hx, hy, 0.0]))
                stab_hist_pts.append(px)

            bearing_rel = (obj.bearing - ego.yaw + math.pi) % (2 * math.pi) - math.pi

            tracks.append({
                "id": oid, "cls": obj.label, "confidence": float(obj.confidence),
                "bbox": bbox,
                "position": [float(pos[0]), float(pos[1])],
                "velocity": [float(vel[0]), float(vel[1])], "speed": speed,
                "bearing": float(obj.bearing), "bearing_rate": bearing_rate,
                "bearing_rel": float(bearing_rel),
                "closing_rate": closing_rate,
                "history": [[hx, hy] for _, hx, hy in self._pos_hist.get(oid, [])],
                "predicted_times": pred_times, "predicted_path": pred_pos,
                "pixel_history": px_hist_pts, "pixel_predicted": px_pred,
                "pixel_history_stab": [p for p in stab_hist_pts],
                "sigma": sigma,
                "ttc": ttc, "tcpa": tcpa, "dcpa": dcpa,
                "risk": risk,
                "risk_level": assess.risk_level.value if assess else "safe",
                "conflict_type": conflict_type, "will_collide": will_collide,
                "conflict_point": conflict_point,
                "direction": assess.direction if assess else "center",
                "direction_label": _direction_label(bearing_rel),
                "factors": {k: float(v) for k, v in factors.items()},
                "expanding": expanding,
                "raw_px_rate": raw_px_rate,
            })
            # Always expose one primary track to the visual console. Risk only
            # controls its warning severity; it should not make the live
            # overlay vanish between otherwise valid detection updates.
            if primary is None or risk > max_risk:
                max_risk = risk
                primary = tracks[-1]

        # --- counterfactual: rider goes straight instead of current action ---
        counterfactual = {"current_cpa": None, "straight_cpa": None,
                          "current_risk": 0.0, "straight_risk": 0.0, "delta": 0.0}
        if primary is not None and primary["predicted_path"]:
            n = min(len(primary["predicted_path"]), len(straight_path))
            if n:
                op = np.asarray(primary["predicted_path"][:n])
                sp = np.asarray([[p[0], p[1]] for p in straight_path[:n]])
                straight_cpa = float(np.min(np.linalg.norm(op - sp, axis=1)))
                current_cpa = primary["dcpa"]
                safe = pipeline.cpa_detector.safe_distance
                straight_risk = max(0.0, 1.0 - straight_cpa / (2.0 * safe))
                counterfactual = {
                    "current_cpa": current_cpa, "straight_cpa": straight_cpa,
                    "current_risk": primary["risk"], "straight_risk": straight_risk,
                    "delta": primary["risk"] - straight_risk,
                }

        # --- system / status ---
        st = receiver.stats if receiver else {}
        imu_ok = (st.get("last_imu_ts") is not None and st.get("last_frame_ts") is not None
                  and abs(st["last_imu_ts"] - st["last_frame_ts"]) < 1.5)
        cam_ok = bool(receiver.connected) if receiver else False
        pose = pipeline.camera_pose_estimator.get_current_pose()
        cy, cp, cr = _euler_from_matrix(pose.rotation_matrix) if pose is not None else (0.0, 0.0, 0.0)

        self._risk_hist.append((ts_s, max_risk))

        threat = None
        if primary is not None:
            threat = {
                "id": primary["id"], "cls": primary["cls"],
                "direction_label": primary["direction_label"],
                "risk": primary["risk"], "ttc": primary["ttc"],
                "tcpa": primary["tcpa"], "dcpa": primary["dcpa"],
                "closing_rate": primary["closing_rate"],
                "confidence": float(risk_by_id[primary["id"]].confidence) if primary["id"] in risk_by_id else 0.0,
                "why": [
                    {"label": "closing rapidly", "active": primary["closing_rate"] < -0.5},
                    {"label": "bearing stable", "active": primary["bearing_rate"] is not None and abs(primary["bearing_rate"]) < 0.05},
                    {"label": "rider turning", "active": turning},
                    {"label": "predicted paths overlap", "active": primary["will_collide"] or (primary["dcpa"] or 99) < 2.0},
                    {"label": "expanding in frame", "active": primary["expanding"]},
                ],
            }

        return {
            "ready": True,
            "timestamp": float(ts_s),
            "system": {
                "fps": float(fps), "camera": cam_ok, "imu": bool(imu_ok),
                "tracking": len(tracks),
                "haptic_endpoint": bool(haptic_endpoint_set),
                "world_model": "STABLE" if (cam_ok and imu_ok) else "DEGRADED",
                "frame_size": list(frame_size) if frame_size else None,
            },
            "camera_pose": {"yaw": cy, "pitch": cp, "roll": cr,
                            "confidence": float(pose.confidence) if pose else 0.0},
            "rider_state": {
                "position": [0.0, 0.0],
                "velocity": [float(ego.velocity[0]), float(ego.velocity[1])],
                "speed": float(ego.speed), "heading": float(ego.yaw),
                "yaw_rate": float(ego.yaw_rate), "turning": turning,
                "turn_direction": pipeline.ego_tracker.get_turn_direction(),
                "predicted_path": [[float(p[0]), float(p[1])] for p in ego_path],
                "straight_path": [[float(p[0]), float(p[1])] for p in straight_path],
            },
            "tracks": tracks,
            "primary_threat_id": primary["id"] if primary else None,
            "threat": threat,
            "haptic": dict(self._haptic),
            "counterfactual": counterfactual,
            "funnel": funnel,
            "events": list(self._events)[-40:],
            "risk_history": [[t, r] for t, r in self._risk_hist],
            "horizon_s": self.horizon_s,
        }
