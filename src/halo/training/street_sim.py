"""Street-level simulation on real OSM geometry (MIT campus area).

Agents move along actual road polylines at real posted speed limits;
pedestrians occasionally cross at real marked crossings. The wearer walks
a route through the map and the same world-space reasoning HALO uses —
predicted paths, CPA, direction labels — drives warnings.

Also accumulates per-crossing near-miss statistics so the page can rank
which intersections *observed in simulation* are most dangerous, alongside
known real-world hotspots.
"""
from __future__ import annotations

import json
import math
import random
import threading
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parents[3] / "data" / "mit_map.json"

DRIVE_TYPES = {"primary", "secondary", "tertiary", "residential",
               "unclassified", "service", "living_street"}
DEFAULT_SPEED = {"primary": 11.0, "secondary": 9.0, "tertiary": 8.0,
                 "residential": 6.5, "unclassified": 6.5, "service": 4.5,
                 "living_street": 3.0}

# known Cambridge hotspots (real crash history), local meters
HOTSPOTS = [
    {"name": "Massachusetts Ave × Vassar St", "pos": [-272, -490],
     "note": "cycling fatality corridor"},
    {"name": "Massachusetts Ave × Memorial Dr", "pos": [-180, -700],
     "note": "high-speed riverfront junction"},
]

SNAP = 8.0  # endpoint snapping grid (m)


def _snap(p):
    return (round(p[0] / SNAP), round(p[1] / SNAP))


def _seg_len(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


class Agent:
    __slots__ = ("id", "cls", "way", "seg", "t", "dir", "speed", "pos",
                 "heading", "crossing", "cross_target", "state", "track")

    def __init__(self, aid, cls, way, seg, t, direction, speed):
        self.id, self.cls = aid, cls
        self.way, self.seg, self.t, self.dir = way, seg, t, direction
        self.speed = speed
        self.pos = self._interp()
        a, b = way["pts"][seg], way["pts"][seg + 1]
        self.heading = math.atan2(b[0] - a[0], b[1] - a[1])
        if direction < 0:
            self.heading += math.pi
        self.crossing = False
        self.cross_target = None
        self.state = "OBSERVE"
        self.track = []

    def _interp(self):
        a, b = self.way["pts"][self.seg], self.way["pts"][self.seg + 1]
        return [a[0] + (b[0] - a[0]) * self.t, a[1] + (b[1] - a[1]) * self.t]


class StreetSim:
    """Continuous sim on the MIT road graph; stepped on a timer thread."""

    def __init__(self, map_path=MAP_PATH, n_cars=26, n_peds=40, n_bikes=8):
        m = json.loads(Path(map_path).read_text())
        self.ways = [w for w in m["ways"] if len(w["pts"]) > 1]
        self.crossings = [{"pos": c["pos"], "score": 0.0, "events": 0}
                          for c in m["crossings"]]
        self.drive_ways = [w for w in self.ways if w["type"] in DRIVE_TYPES]
        self.walk_ways = [w for w in self.ways
                          if w["type"] in {"footway", "path", "residential",
                                           "pedestrian", "living_street",
                                           "tertiary", "secondary"}]
        # endpoint → way index for graph connectivity
        self.ends = {}
        for w in self.ways:
            self.ends.setdefault(_snap(w["pts"][0]), []).append((w, 0))
            self.ends.setdefault(_snap(w["pts"][-1]), []).append((w, len(w["pts"]) - 2))

        # each crossing's nearest drive-road segment, precomputed once
        for c in self.crossings:
            road = min(self.drive_ways,
                       key=lambda w: self._dist_to_way(c["pos"], w))
            c["road"], c["seg"] = road, self._nearest_seg(c["pos"], road)

        self.rng = random.Random(7)
        self.t = 0.0
        self.lock = threading.Lock()
        self.events = []
        self.warnings = 0
        self.near_misses = 0
        self.agents = []
        for i in range(n_cars):
            self.agents.append(self._spawn(f"car-{i}", "car"))
        for i in range(n_bikes):
            self.agents.append(self._spawn(f"bike-{i}", "bicycle"))
        for i in range(n_peds):
            self.agents.append(self._spawn(f"person-{i}", "person"))
        self.wearer = self._spawn("you", "you", speed=1.5)
        # keep the action near the wearer: seed a bubble, re-seed wanderers
        for ag in self.agents:
            if math.hypot(ag.pos[0] - self.wearer.pos[0],
                          ag.pos[1] - self.wearer.pos[1]) > 150:
                self._respawn_near(ag)

    def _respawn_near(self, ag, rmin=20, rmax=130):
        """Move an agent onto a way inside the wearer's attention bubble."""
        wx, wy = self.wearer.pos
        pool = self.drive_ways if ag.cls in ("car", "bicycle") else self.walk_ways
        near = [w for w in pool if any(
            rmin < math.hypot(p[0] - wx, p[1] - wy) < rmax for p in w["pts"])]
        if not near:
            return
        w = self.rng.choice(near)
        ag.way = w
        ag.seg = self.rng.randrange(len(w["pts"]) - 1)
        ag.t, ag.dir = self.rng.random(), self.rng.choice([1, -1])
        ag.pos = ag._interp()
        ag.track.clear()

    # -------------------------------------------------- spawning / routing
    def _spawn(self, aid, cls, speed=None):
        if cls in ("car",):
            way = self.rng.choice(self.drive_ways)
            spd = way["speed"] or DEFAULT_SPEED.get(way["type"], 7)
            speed = speed or spd * self.rng.uniform(0.8, 1.0)
        elif cls == "bicycle":
            way = self.rng.choice(self.drive_ways)
            speed = speed or self.rng.uniform(4, 6.5)
        else:
            way = self.rng.choice(self.walk_ways or self.ways)
            speed = speed or self.rng.uniform(1.1, 1.7)
        seg = self.rng.randrange(len(way["pts"]) - 1)
        return Agent(aid, cls, way, seg, self.rng.random(),
                     self.rng.choice([1, -1]), speed)

    def _next_way(self, ag):
        end_pt = ag.way["pts"][-1] if ag.dir > 0 else ag.way["pts"][0]
        cands = [w for w, _ in self.ends.get(_snap(end_pt), []) if w is not ag.way]
        pool = self.drive_ways if ag.cls in ("car", "you", "bicycle") else None
        if pool is not None:
            cands = [w for w in cands if w in pool] or cands
        if not cands:
            return None
        same = [w for w in cands if w.get("name") and w["name"] == ag.way.get("name")]
        nxt = self.rng.choice(same or cands)
        # enter from whichever end touches
        if _snap(nxt["pts"][0]) == _snap(end_pt):
            return nxt, 0, 1
        if _snap(nxt["pts"][-1]) == _snap(end_pt):
            return nxt, len(nxt["pts"]) - 2, -1
        return None

    # -------------------------------------------------- stepping
    def step(self, dt=0.1):
        with self.lock:
            self.t += dt
            for ag in self.agents + [self.wearer]:
                self._advance(ag, dt)
            # keep the scene populated near the wearer
            if self.rng.random() < 0.3:
                for ag in self.agents:
                    if math.hypot(ag.pos[0] - self.wearer.pos[0],
                                  ag.pos[1] - self.wearer.pos[1]) > 180:
                        self._respawn_near(ag)
            self._ped_crossing_logic(dt)
            threat = self._assess()
            for c in self.crossings:
                c["score"] *= (1 - 0.02 * dt)  # slow decay
            return threat

    def _advance(self, ag, dt):
        if ag.crossing:
            dx = ag.cross_target[0] - ag.pos[0]
            dy = ag.cross_target[1] - ag.pos[1]
            d = math.hypot(dx, dy)
            if d < 0.6:
                ag.crossing = False
            else:
                step = min(1.4 * dt, d)
                ag.pos[0] += dx / d * step
                ag.pos[1] += dy / d * step
                ag.heading = math.atan2(dx, dy)
            self._record(ag)
            return
        pts = ag.way["pts"]
        remaining = ag.speed * dt
        while remaining > 0:
            a, b = pts[ag.seg], pts[ag.seg + 1]
            L = _seg_len(a, b)
            travel = L * (1 - ag.t) if ag.dir > 0 else L * ag.t
            if travel <= 1e-6:
                # next segment / next way
                nseg = ag.seg + ag.dir
                if 0 <= nseg < len(pts) - 1:
                    ag.seg = nseg
                    ag.t = 0.0 if ag.dir > 0 else 1.0
                    continue
                nxt = self._next_way(ag)
                if nxt is None:
                    if ag is self.wearer:
                        self._respawn_wearer(ag)
                    else:
                        new = self._spawn(ag.id, ag.cls)
                        ag.way, ag.seg, ag.t, ag.dir, ag.speed = \
                            new.way, new.seg, new.t, new.dir, new.speed
                    pts = ag.way["pts"]
                    continue
                ag.way, ag.seg, ag.dir = nxt
                ag.t = 0.0 if ag.dir > 0 else 1.0
                pts = ag.way["pts"]
                continue
            move = min(remaining, travel)
            ag.t += (move / L) * ag.dir
            remaining -= move
        a, b = pts[ag.seg], pts[ag.seg + 1]
        nx = a[0] + (b[0] - a[0]) * ag.t
        ny = a[1] + (b[1] - a[1]) * ag.t
        ag.heading = math.atan2(nx - ag.pos[0], ny - ag.pos[1])
        ag.pos = [nx, ny]
        self._record(ag)

    def _record(self, ag):
        ag.track.append(tuple(ag.pos))
        if len(ag.track) > 60:
            ag.track.pop(0)

    def _respawn_wearer(self, ag):
        """Wearer loops back to a road near map center."""
        cands = sorted(self.drive_ways,
                       key=lambda w: min(math.hypot(p[0], p[1]) for p in w["pts"]))
        ag.way = self.rng.choice(cands[:8])
        ag.seg, ag.t, ag.dir = 0, 0.0, 1
        ag.track.clear()

    # -------------------------------------------------- ped crossings
    def _ped_crossing_logic(self, dt):
        for ag in self.agents:
            if ag.cls != "person" or ag.crossing:
                continue
            for c in self.crossings:
                d = math.hypot(ag.pos[0] - c["pos"][0], ag.pos[1] - c["pos"][1])
                if d < 4 and self.rng.random() < 0.25 * dt:
                    road, i = c["road"], c["seg"]
                    a, b = road["pts"][i], road["pts"][i + 1]
                    h = math.atan2(b[0] - a[0], b[1] - a[1]) + math.pi / 2
                    ag.crossing = True
                    ag.cross_target = [c["pos"][0] + math.sin(h) * 14,
                                       c["pos"][1] + math.cos(h) * 14]
                    break

    def _dist_to_way(self, p, w):
        return min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in w["pts"])

    def _nearest_seg(self, p, w):
        return min(range(len(w["pts"]) - 1),
                   key=lambda i: math.hypot(p[0] - w["pts"][i][0],
                                            p[1] - w["pts"][i][1]))

    # -------------------------------------------------- HALO reasoning
    def _predict(self, ag, horizon=4.0, step=0.5):
        """Predicted path: continue along the way polyline."""
        pts = ag.way["pts"]
        path = [list(ag.pos)]
        seg, t, d = ag.seg, ag.t, ag.dir
        rem = ag.speed * horizon
        while rem > 0:
            a, b = pts[seg], pts[seg + 1]
            L = _seg_len(a, b)
            t += (ag.speed * step) / max(L, 1e-6) * d
            if not (0 <= t <= 1):
                seg += d
                if not (0 <= seg < len(pts) - 1):
                    break
                t = 0.0 if d > 0 else 1.0
                continue
            path.append([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            rem -= ag.speed * step
        return path

    def _assess(self):
        wx, wy = self.wearer.pos
        wh = self.wearer.heading
        best = None
        for ag in self.agents:
            rx, ry = ag.pos[0] - wx, ag.pos[1] - wy
            dist = math.hypot(rx, ry)
            rvx = math.sin(ag.heading) * ag.speed - math.sin(wh) * self.wearer.speed
            rvy = math.cos(ag.heading) * ag.speed - math.cos(wh) * self.wearer.speed
            # linear CPA
            denom = rvx * rvx + rvy * rvy
            tcpa = max(0.0, -(rx * rvx + ry * rvy) / denom) if denom > 1e-6 else 99
            dcpa = math.hypot(rx + rvx * tcpa, ry + rvy * tcpa)
            # path-overlap: does the agent's predicted path pass within 2.5m?
            overlap = False
            if dist < 40:
                for p in self._predict(ag):
                    if math.hypot(p[0] - wx, p[1] - wy) < 2.5:
                        overlap = True
                        break
            # bearing vs wearer heading → direction label
            br = math.atan2(rx, ry) - wh
            while br > math.pi: br -= 2 * math.pi
            while br < -math.pi: br += 2 * math.pi
            side = "LEFT" if math.sin(br) < 0 else "RIGHT"
            label = ("FRONT" if abs(br) < 0.6 else
                     "BEHIND" if abs(br) > 2.5 else
                     ("BEHIND " if abs(br) > 1.9 else "") + side)

            if dist < 3 or (dcpa < 2.0 and tcpa < 3) or (overlap and tcpa < 5):
                ag.state = "WARN"
            elif dcpa < 6 and tcpa < 8 or dist < 10:
                ag.state = "ATTEND"
            else:
                ag.state = "OBSERVE"

            score = (10 if ag.state == "WARN" else 2 if ag.state == "ATTEND" else 0) \
                - dist * 0.05
            if best is None or score > best["score"]:
                best = {"score": score, "agent": ag, "dist": dist,
                        "tcpa": tcpa, "dcpa": dcpa, "label": label,
                        "overlap": overlap}

            # near-miss stats at crossings: car close to a crossing ped
            if ag.cls == "car" and dist < 60:
                for ped in self.agents:
                    if ped.cls == "person" and ped.crossing:
                        pd = math.hypot(ag.pos[0] - ped.pos[0],
                                        ag.pos[1] - ped.pos[1])
                        if pd < 8:
                            cx = min(self.crossings,
                                     key=lambda c: math.hypot(
                                         ped.pos[0] - c["pos"][0],
                                         ped.pos[1] - c["pos"][1]), default=None)
                            if cx and math.hypot(ped.pos[0] - cx["pos"][0],
                                                 ped.pos[1] - cx["pos"][1]) < 10:
                                cx["score"] += 0.05
                                cx["events"] += 1

        if best and best["agent"].state == "WARN":
            self.warnings += 1
            if best["dist"] < 2.5:
                self.near_misses += 1
            if not self.events or self.t - self.events[-1]["ts"] > 2:
                self.events.append({"ts": self.t, "type": "WARN",
                                    "label": f'{best["agent"].id} · {best["label"]} · '
                                             f'{best["tcpa"]:.1f}s'})
                self.events = self.events[-40:]
        return best

    # -------------------------------------------------- output
    def map_data(self):
        """Static geometry — fetched once by the page."""
        return {"ways": self.ways,
                "crossings": [c["pos"] for c in self.crossings],
                "hotspots": HOTSPOTS}

    def snapshot(self):
        with self.lock:
            th = getattr(self, "_last_threat", None)
            return {
                "t": round(self.t, 1),
                "agents": [{
                    "id": a.id, "cls": a.cls, "pos": a.pos,
                    "heading": a.heading, "speed": round(a.speed, 1),
                    "state": a.state, "crossing": a.crossing,
                    "predicted": self._predict(a) if a.state != "OBSERVE" else [],
                    "track": a.track[-30:],
                } for a in self.agents],
                "wearer": {"pos": self.wearer.pos, "heading": self.wearer.heading,
                           "speed": self.wearer.speed,
                           "predicted": self._predict(self.wearer)},
                "threat": th and {
                    "id": th["agent"].id, "cls": th["agent"].cls,
                    "state": th["agent"].state, "dist": round(th["dist"], 1),
                    "tcpa": round(th["tcpa"], 1) if th["tcpa"] < 90 else None,
                    "dcpa": round(th["dcpa"], 1), "label": th["label"],
                    "overlap": th["overlap"],
                },
                "stats": {"warnings": self.warnings,
                          "near_misses": self.near_misses,
                          "agents": len(self.agents),
                          "attend": sum(1 for a in self.agents if a.state == "ATTEND"),
                          "warn": sum(1 for a in self.agents if a.state == "WARN")},
                "danger_ranking": sorted(
                    [{"pos": c["pos"], "score": round(c["score"], 2),
                      "events": c["events"]} for c in self.crossings if c["score"] > 0.1],
                    key=lambda c: -c["score"])[:5],
                "events": self.events[-10:],
            }

    def run_forever(self, dt=0.1):
        import time
        while True:
            self._last_threat = self.step(dt)
            time.sleep(dt)


_SIM = None
_SIM_THREAD = None


def get_sim():
    global _SIM, _SIM_THREAD
    if _SIM is None:
        _SIM = StreetSim()
        _SIM_THREAD = threading.Thread(target=_SIM.run_forever, daemon=True)
        _SIM_THREAD.start()
    return _SIM
