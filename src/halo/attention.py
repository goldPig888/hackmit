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

# Scene-conditioned thresholds: how much attention is needed to promote.
# Density raises/lowers the bar; outdoors generally lowers it further
# (approaches arrive faster and from farther).
_THRESHOLDS = {
    "crowded": {"indoor":  {"watch": 0.65, "alert": 0.88},
                "outdoor": {"watch": 0.60, "alert": 0.85}},
    "normal":  {"indoor":  {"watch": 0.55, "alert": 0.85},
                "outdoor": {"watch": 0.50, "alert": 0.80}},
    "sparse":  {"indoor":  {"watch": 0.50, "alert": 0.82},
                "outdoor": {"watch": 0.40, "alert": 0.75}},
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


def classify_scene(n_people: int, n_tracks: int, ego_speed: float,
                   brightness: float | None = None,
                   has_vehicle: bool = False) -> tuple[str, str]:
    """Scene context: (density, environment).

    Density from track counts; environment inferred from mean frame
    luminance and vehicle-class presence — cheap proxies, good enough for
    threshold selection.
    """
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
    return density, environment


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
             or (expanding and raw_px_rate > _REFLEX_STRIKE_PX))
    )
    if fast_approach or sudden_close:
        reasons = (
            ["sudden close-range motion",
             f"very close ({d:.1f} m)",
             "image expanding fast",
             "fast image motion"]
            if sudden_close and not fast_approach else
            ["rapid approach at close range",
             f"closing {abs(closing_rate):.1f} m/s",
             "image expanding fast" if expanding else "fast image motion",
             f"est. contact {ttc:.1f}s" if ttc else "contact imminent"])
        return AttentionResult(attention=1.0, state=REFLEX, reflex=True,
                               reasons=reasons)

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
    unusual = (_clamp01(raw_px_rate / 200.0)
               + _clamp01(area_rate / 1.2) * 0.5
               + (0.4 if expanding else 0.0))
    persistence = _clamp01(duration_s / 15.0)
    # co-movement: stays near while we move and its bearing barely changes
    comovement = 0.0
    if (ego_moving and duration_s > 4.0 and d < 12.0
            and bearing_rate is not None and abs(bearing_rate) < 0.05):
        comovement = 0.8 if density == "sparse" else 0.35

    w = {"d": 0.28, "v": 0.26, "h": 0.16, "p": 0.10, "t": 0.10, "c": 0.10}
    attention = (w["d"] * proximity + w["v"] * approach + w["h"] * heading_toward
                 + w["p"] * _clamp01(unusual) + w["t"] * persistence
                 + w["c"] * comovement)
    attention = _clamp01(attention)

    th = _THRESHOLDS[density][environment]
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
    if will_collide:
        reasons.append("predicted paths overlap")

    return AttentionResult(attention=attention, state=state,
                           reflex=False, reasons=reasons)
