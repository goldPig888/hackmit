"""State featurization for the attention policy.

The RL agent never sees pixels. It sees the world-state HALO already
computes: geometry, kinematics, expansion, CPA, persistence, co-movement,
strike evidence, plus scene context — one vector per subject track.
"""

from __future__ import annotations

import math

import numpy as np

# Action space — REFLEX deliberately excluded: the deterministic reflex
# rules bypass the policy entirely for immediate-motion evidence.
ACTIONS = ["OBSERVE", "ATTEND", "WARN"]
OBSERVE, ATTEND, WARN = 0, 1, 2

FEATURE_DIM = 15
FEATURE_NAMES = [
    "distance", "closing_rate", "heading_toward", "bearing_rate",
    "area_rate", "inv_tcpa", "inv_dcpa", "will_collide",
    "persistence", "comovement", "strike_motion",
    "density_crowded", "density_normal", "density_sparse", "outdoor",
]


def featurize(track: dict, scene: dict) -> np.ndarray:
    """World-state track + scene dict -> FEATURE_DIM vector."""
    pos = track.get("position") or [0.0, 0.0]
    d = math.hypot(pos[0], pos[1])
    ev = track.get("evidence") or {}
    tcpa = track.get("tcpa")
    dcpa = track.get("dcpa")
    density = (scene or {}).get("density", "normal")
    return np.array([
        min(d / 20.0, 1.0),
        max(-1.0, min(1.0, (track.get("closing_rate") or 0.0) / -8.0)),
        float(ev.get("heading_toward", 0.0)),
        max(-1.0, min(1.0, abs(track.get("bearing_rate") or 0.0) / 0.3)),
        max(-1.0, min(1.0, (track.get("area_rate") or 0.0) / 1.5)),
        1.0 / max(tcpa, 0.1) if tcpa else 0.0,
        1.0 / max(dcpa, 0.1) if dcpa else 0.0,
        1.0 if track.get("will_collide") else 0.0,
        float(ev.get("persistence", 0.0)),
        float(ev.get("persistent_comovement", 0.0)),
        float(ev.get("strike_motion", 0.0)),
        1.0 if density == "crowded" else 0.0,
        1.0 if density == "normal" else 0.0,
        1.0 if density == "sparse" else 0.0,
        1.0 if (scene or {}).get("environment") == "outdoor" else 0.0,
    ], dtype=np.float32)


def featurize_raw(d: float, closing: float, heading: float, bearing_rate: float,
                  area_rate: float, tcpa: float | None, dcpa: float | None,
                  will_collide: bool, persistence: float, comovement: float,
                  strike: float, density: str, environment: str) -> np.ndarray:
    """Same layout for the simulator, which doesn't emit track dicts."""
    return featurize(
        {"position": [0.0, d], "closing_rate": closing, "area_rate": area_rate,
         "bearing_rate": bearing_rate, "tcpa": tcpa, "dcpa": dcpa,
         "will_collide": will_collide,
         "evidence": {"heading_toward": heading, "persistence": persistence,
                      "persistent_comovement": comovement,
                      "strike_motion": strike}},
        {"density": density, "environment": environment})


# Reward table (kept simple and legible — shown verbatim on /training)
REWARD_RULES = {
    "correct": 2.0,
    "early_warning": 1.0,
    "unnecessary_attend": -1.0,
    "false_warn": -3.0,
    "missed_warn": -6.0,
    "nag": -0.1,
}


def reward(action: int, label: int, *, early: bool = False,
           repeated_bother: bool = False) -> float:
    """Score one action against the episode's ground-truth label."""
    if action == label:
        r = REWARD_RULES["correct"]
        if action == WARN and early:
            r += REWARD_RULES["early_warning"]
    elif action == ATTEND and label == OBSERVE:
        r = REWARD_RULES["unnecessary_attend"]
    elif action == WARN and label < WARN:
        r = REWARD_RULES["false_warn"]
    elif action < WARN and label == WARN:
        r = REWARD_RULES["missed_warn"]
    else:  # ATTEND when WARN needed — under-attended but not fully missed
        r = REWARD_RULES["missed_warn"] * 0.5
    if repeated_bother and action > label:
        r += REWARD_RULES["nag"]
    return r
