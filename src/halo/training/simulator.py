"""Tiny scenario simulator for the attention policy.

Generates kinematic episodes: the wearer at origin, one subject agent plus
ambient agents (which only shape scene context). Because the simulator knows
the subject's true future path, it can label each timestep with the action a
perfect attention policy would take — using TRUE minimum future distance,
not HALO's estimated CPA.

Labels: WARN if the subject will get within ~1.6m soon, ATTEND if it closes
meaningfully, OBSERVE otherwise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .features import OBSERVE, ATTEND, WARN, featurize_raw

SCENARIOS = ["normal_crowd", "fast_approach", "safe_close_pass",
             "converging_path", "persistent_person", "sudden_close"]

_WARN_DIST = 1.6      # true future closest-approach that merits warning
_ATTEND_DIST = 5.0    # closes meaningfully but never gets dangerous
_DT = 0.1
_STEPS = 40           # 4s episodes


@dataclass
class Step:
    t: float
    subject_pos: tuple[float, float]
    subject_vel: tuple[float, float]
    wearer_pos: tuple[float, float]
    wearer_vel: tuple[float, float]
    ambient: list[tuple[float, float]]
    density: str
    environment: str
    label: int
    state: object  # np.ndarray
    min_future_dist: float
    closing_rate: float
    area_rate: float


@dataclass
class Episode:
    scenario: str
    steps: list[Step] = field(default_factory=list)


def _area_from_dist(d: float) -> float:
    """Approximate bbox area for a person at distance d (arbitrary units)."""
    return (8.0 / max(d, 0.4)) ** 2


def _ambient_positions(rng, n: int, t: float) -> list[tuple[float, float]]:
    return [(math.sin(i * 2.4 + t * 0.15) * (4 + i % 3),
             math.cos(i * 1.7 + t * 0.1) * (5 + i % 4)) for i in range(n)]


def sample_episode(rng: random.Random, scenario: str | None = None) -> Episode:
    scenario = scenario or rng.choice(SCENARIOS)
    density = "crowded" if scenario in ("normal_crowd", "fast_approach",
                                        "sudden_close") else rng.choice(
        ["sparse", "normal"])
    environment = "indoor" if density == "crowded" else rng.choice(
        ["indoor", "outdoor"])
    n_ambient = rng.randint(10, 18) if density == "crowded" else rng.randint(0, 3)

    # wearer motion
    w_speed = rng.uniform(0.8, 1.6) if environment == "indoor" else rng.uniform(1.2, 3.0)
    w_dir = rng.uniform(-0.3, 0.3)

    # subject start + velocity per scenario
    bearing = rng.uniform(-1.0, 1.0)
    dist = rng.uniform(6.0, 14.0)
    sx, sy = math.sin(bearing) * dist, math.cos(bearing) * dist
    if scenario == "fast_approach":
        spd = rng.uniform(3.0, 6.0)
        vx, vy = -math.sin(bearing) * spd * 0.9, -math.cos(bearing) * spd
    elif scenario == "safe_close_pass":
        spd = rng.uniform(1.0, 2.0)
        vx, vy = -math.sin(bearing + 0.6) * spd, -math.cos(bearing + 0.6) * spd
    elif scenario == "converging_path":
        spd = rng.uniform(1.5, 3.0)
        vx, vy = -math.sin(bearing) * spd, -math.cos(bearing) * spd * 0.6
    elif scenario == "persistent_person":
        spd = w_speed * rng.uniform(0.9, 1.1)   # co-mover
        vx, vy = w_dir * spd, -abs(math.cos(bearing)) * 0.05
        sy = -abs(sy)  # starts behind
    elif scenario == "sudden_close":
        spd = 0.0  # static until the burst
        vx = vy = 0.0
        dist = rng.uniform(2.0, 4.0)
        sx, sy = math.sin(bearing) * dist, math.cos(bearing) * dist
    else:  # normal_crowd
        spd = rng.uniform(0.3, 1.2)
        vx, vy = math.sin(rng.uniform(0, 6.28)) * spd, math.cos(rng.uniform(0, 6.28)) * spd

    # pre-roll subject path
    xs, ys = [sx], [sy]
    vx_t, vy_t = vx, vy
    burst_at = rng.randint(int(_STEPS * 0.45), int(_STEPS * 0.7))
    for k in range(1, _STEPS):
        if scenario == "sudden_close" and k == burst_at:
            spd = rng.uniform(4.0, 7.0)
            vx_t, vy_t = -sx / dist * spd, -sy / dist * spd
        xs.append(xs[-1] + vx_t * _DT)
        ys.append(ys[-1] + vy_t * _DT)

    ep = Episode(scenario=scenario)
    duration = 0.0
    for k in range(_STEPS):
        t = k * _DT
        wx, wy = w_dir * w_speed * t, w_speed * t
        dx, dy = xs[k] - wx, ys[k] - wy
        d = math.hypot(dx, dy)
        rel_vx = (xs[k] - xs[k - 1]) / _DT - w_dir * w_speed if k else 0.0
        rel_vy = (ys[k] - ys[k - 1]) / _DT - w_speed if k else 0.0
        closing = (dx * rel_vx + dy * rel_vy) / max(d, 0.1)
        a_now = _area_from_dist(d)
        a_prev = _area_from_dist(math.hypot(xs[k - 1] - wx + w_dir * w_speed * _DT,
                                            ys[k - 1] - wy + w_speed * _DT)) if k else a_now
        area_rate = (a_now - a_prev) / (a_prev * _DT) if a_prev > 0 else 0.0
        min_future = min(math.hypot(xs[j] - w_dir * w_speed * j * _DT,
                                    ys[j] - w_speed * j * _DT)
                         for j in range(k, _STEPS))
        spd_rel = math.hypot(rel_vx, rel_vy)
        heading = (-(dx * rel_vx + dy * rel_vy) / (spd_rel * d)
                   if spd_rel > 0.1 and d > 0.1 else 0.0)
        duration += _DT
        persistence = min(duration / 15.0, 1.0)
        comovement = 0.8 if (scenario == "persistent_person"
                             and abs(math.atan2(dx, dy)) < 2.0) else 0.0
        strike = (0.75 if scenario == "sudden_close" and k >= burst_at
                  and area_rate > 0.5 else 0.0)

        # label from the subject's TRUE future
        approaching_soon = closing < -0.4 or heading > 0.6
        if min_future < _WARN_DIST and (approaching_soon or area_rate > 0.4):
            label = WARN
        elif min_future < _ATTEND_DIST or closing < -0.5 or comovement > 0.5:
            label = ATTEND
        else:
            label = OBSERVE

        state = featurize_raw(
            d=d, closing=closing, heading=max(0.0, heading),
            bearing_rate=0.02 if comovement else 0.15,
            area_rate=area_rate, tcpa=(d / abs(closing) if closing < -0.1 else None),
            dcpa=min_future, will_collide=min_future < 1.2,
            persistence=persistence, comovement=comovement,
            strike=strike, density=density, environment=environment)

        ep.steps.append(Step(
            t=t, subject_pos=(xs[k] - wx, ys[k] - wy),
            subject_vel=(rel_vx, rel_vy), wearer_pos=(wx, wy),
            wearer_vel=(w_dir * w_speed, w_speed),
            ambient=_ambient_positions(rng, n_ambient, t),
            density=density, environment=environment, label=label,
            state=state, min_future_dist=min_future,
            closing_rate=closing, area_rate=area_rate))
    return ep


def holdout_set(rng: random.Random, n: int = 30) -> list[Episode]:
    """Fixed evaluation set — same scenarios every call for comparability."""
    return [sample_episode(rng) for _ in range(n)]
