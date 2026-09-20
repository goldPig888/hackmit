"""Context-conditioned attention layer.

Sits between perception and alerting: every tracked object gets an attention
score each frame, then a state — OBSERVE / ATTEND / WARN / REFLEX. The slow
path is the weighted score with scene-conditioned thresholds; the reflex path
bypasses it for fast, close-range approach events.

No learned model here — the linear score is intentional: interpretable,
tunable on-device, and the reasons it emits are literally the weighted terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# State ordering (higher wins)
OBSERVE, ATTEND, WARN, REFLEX = "OBSERVE", "ATTEND", "WARN", "REFLEX"

# Context-conditioned profiles: not just thresholds — the channel weights
# themselves change with the scene. Crowds suppress proximity and reward
# motion channels; sparse scenes reward persistence and co-movement.
# Keys: d=proximity v=approach h=heading p=unusual t=persistence c=comovement
#       s=strike-like motion (pose path)
_PROFILES = {
    ("crowded", "indoor"):  {"w": {"d": 0.08, "v": 0.28, "h": 0.16, "p": 0.22, "t": 0.08, "c": 0.05, "s": 0.13},
                             "watch": 0.65, "alert": 0.84},
    ("crowded", "outdoor"): {"w": {"d": 0.10, "v": 0.27, "h": 0.15, "p": 0.18, "t": 0.08, "c": 0.08, "s": 0.14},
                             "watch": 0.60, "alert": 0.82},
    ("normal", "indoor"):   {"w": {"d": 0.20, "v": 0.24, "h": 0.14, "p": 0.14, "t": 0.12, "c": 0.06, "s": 0.10},
                             "watch": 0.55, "alert": 0.85},
    ("normal", "outdoor"):  {"w": {"d": 0.20, "v": 0.24, "h": 0.14, "p": 0.12, "t": 0.12, "c": 0.10, "s": 0.08},
                             "watch": 0.50, "alert": 0.80},
    ("sparse", "indoor"):   {"w": {"d": 0.18, "v": 0.20, "h": 0.12, "p": 0.12, "t": 0.18, "c": 0.14, "s": 0.08},
                             "watch": 0.50, "alert": 0.82},
    ("sparse", "outdoor"):  {"w": {"d": 0.15, "v": 0.22, "h": 0.12, "p": 0.10, "t": 0.18, "c": 0.15, "s": 0.08},
                             "watch": 0.40, "alert": 0.64},
}

# Reflex fast-path: a genuinely fast, close-range approach. Fires without
# waiting for the temporal filter / attention accumulation.
_REFLEX_CLOSING_MS = -3.0     # m/s toward wearer
_REFLEX_RANGE_M = 6.0         # m
_REFLEX_PX_RATE = 120.0       # px/s raw image-space motion
_REFLEX_TTC_S = 1.5           # s
# Strike-proxy variant: at arm's length, a punch shows as sudden expansion +
# fast image motion even when the torso is barely approaching. Noisier —
# kept strict so waving doesn't trip it.
_REFLEX_STRIKE_M = 3.0        # m
_REFLEX_STRIKE_PX = 250.0     # px/s
_REFLEX_STRIKE_AREA = 0.8     # bbox area growth /s (~doubling in ~0.9s)


@dataclass
class AttentionResult:
    attention: float
    state: str
    reflex: bool
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass
class SceneContext:
    """Full scene profile — drives weight selection, not just thresholds."""
    density: str
    environment: str
    people_count: int = 0
    vehicle_count: int = 0
    ego_motion: str = "stationary"        # stationary | walking | riding
    proximity_baseline: float = 0.0       # mean 1/dist of people — ambient crowding
    background_motion: float = 0.0        # mean track speed — ambient activity


def build_context(n_people: int, n_tracks: int, ego_speed: float,
                  brightness: float | None = None, has_vehicle: bool = False,
                  vehicle_count: int = 0, mean_person_dist: float | None = None,
                  mean_track_speed: float = 0.0) -> SceneContext:
    """Estimate the scene profile from cheap observables."""
    if n_people >= 6 or n_tracks >= 10:
        density = "crowded"
    elif n_tracks <= 2:
        density = "sparse"
    else:
        density = "normal"
    if has_vehicle:
        environment = "outdoor"
    elif brightness is not None:
        environment = "outdoor" if brightness > 110 else "indoor"
    else:
        environment = "indoor"
    ego_motion = "riding" if ego_speed > 4.0 else "walking" if ego_speed > 0.5 else "stationary"
    proximity_baseline = (1.0 / max(mean_person_dist, 0.5)) if mean_person_dist else 0.0
    return SceneContext(density=density, environment=environment,
                        people_count=n_people, vehicle_count=vehicle_count,
                        ego_motion=ego_motion,
                        proximity_baseline=proximity_baseline,
                        background_motion=mean_track_speed)


def classify_scene(n_people: int, n_tracks: int, ego_speed: float,
                   brightness: float | None = None,
                   has_vehicle: bool = False) -> tuple[str, str]:
    """Density + environment pair (kept for callers/tests)."""
    ctx = build_context(n_people, n_tracks, ego_speed, brightness, has_vehicle)
    return ctx.density, ctx.environment


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_track(
    *,
    distance_m: float,
    closing_rate: float,
    velocity: tuple[float, float],
    position: tuple[float, float],
    bearing_rate: float | None,
    duration_s: float,
    expanding: bool,
    raw_px_rate: float,
    area_rate: float = 0.0,
    strike_motion: float = 0.0,
    ttc: float | None,
    will_collide: bool,
    label: str,
    density: str,
    environment: str,
    ego_moving: bool,
) -> AttentionResult:
    """Score one track. Returns attention in [0,1], a state, and evidence."""
    d = max(distance_m, 0.1)

    # --- reflex fast path ---
    # (a) fast approach at close range — lunges, runners, vehicles
    fast_approach = (
        closing_rate < _REFLEX_CLOSING_MS
        and d < _REFLEX_RANGE_M
        and (expanding or raw_px_rate > _REFLEX_PX_RATE)
        and (ttc is None or ttc < _REFLEX_TTC_S)
    )
    # (b) sudden close-range burst — strike proxy when torso is static.
    # A punch on an already-tracked person mostly makes the box grow fast,
    # so strong normalized area growth alone is sufficient evidence.
    sudden_close = (
        d < _REFLEX_STRIKE_M
        and (area_rate > _REFLEX_STRIKE_AREA
             or strike_motion > 0.7
             or (expanding and raw_px_rate > _REFLEX_STRIKE_PX))
    )
    if fast_approach or sudden_close:
        reasons = (
            ["sudden close-range motion",
             f"very close ({d:.1f} m)"]
            + (["rapid strike-like motion"] if strike_motion > 0.7 else [])
            + (["image expanding fast"] if area_rate > _REFLEX_STRIKE_AREA
               or expanding else ["fast image motion"])
            if sudden_close and not fast_approach else
            ["rapid approach at close range",
             f"closing {abs(closing_rate):.1f} m/s",
             "image expanding fast" if expanding else "fast image motion",
             f"est. contact {ttc:.1f}s" if ttc else "contact imminent"])
        return AttentionResult(attention=1.0, state=REFLEX, reflex=True,
                               reasons=reasons,
                               evidence={"rapid_approach": 1.0,
                                         "strike_motion": max(strike_motion, _clamp01(area_rate)),
                                         "rapid_expansion": _clamp01(area_rate)})

    # --- weighted attention features ---
    proximity = _clamp01(1.0 - d / 20.0)                       # near matters
    approach = _clamp01(-closing_rate / 8.0)                   # m/s toward us
    # heading alignment: is its velocity pointed at the wearer?
    vx, vy = velocity
    px, py = position
    spd = math.hypot(vx, vy)
    heading_toward = 0.0
    if spd > 0.3 and d > 0.1:
        heading_toward = _clamp01((-(vx * px + vy * py)) / (spd * d))
    unusual = _clamp01(_clamp01(raw_px_rate / 200.0)
                       + _clamp01(area_rate / 1.2) * 0.5
                       + (0.4 if expanding else 0.0))
    persistence = _clamp01(duration_s / 15.0)
    # co-movement: stays near while we move and its bearing barely changes
    comovement = 0.0
    if (ego_moving and duration_s > 4.0 and d < 12.0
            and bearing_rate is not None and abs(bearing_rate) < 0.05):
        comovement = 0.8 if density == "sparse" else 0.35

    profile = _PROFILES.get((density, environment), _PROFILES[("normal", "indoor")])
    w = profile["w"]
    attention = (w["d"] * proximity + w["v"] * approach + w["h"] * heading_toward
                 + w["p"] * unusual + w["t"] * persistence
                 + w["c"] * comovement + w["s"] * strike_motion)
    attention = _clamp01(attention)

    evidence = {
        "proximity": proximity,
        "rapid_approach": approach,
        "heading_toward": heading_toward,
        "unusual_motion": unusual,
        "rapid_expansion": _clamp01(area_rate / 1.2),
        "persistence": persistence,
        "persistent_comovement": comovement,
        "strike_motion": strike_motion,
        "trajectory_conflict": 1.0 if will_collide else 0.0,
    }

    th = {"watch": profile["watch"], "alert": profile["alert"]}
    if will_collide or attention >= th["alert"]:
        state = WARN
    elif attention >= th["watch"]:
        state = ATTEND
    else:
        state = OBSERVE

    reasons = []
    if approach > 0.4:
        reasons.append(f"approaching at {abs(closing_rate):.1f} m/s")
    if proximity > 0.6:
        reasons.append(f"close range ({d:.1f} m)")
    if heading_toward > 0.6:
        reasons.append("heading toward wearer")
    if expanding:
        reasons.append(f"expanding in frame ({area_rate:.1f}/s)")
    if bearing_rate is not None and abs(bearing_rate) < 0.05 and approach > 0.2:
        reasons.append("bearing stable")
    if comovement > 0.5:
        reasons.append("persistent co-movement")
    if strike_motion > 0.4:
        reasons.append("strike-like limb motion")
    if will_collide:
        reasons.append("predicted paths overlap")

    return AttentionResult(attention=attention, state=state,
                           reflex=False, reasons=reasons, evidence=evidence)
