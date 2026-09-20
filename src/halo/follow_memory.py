"""Long-horizon follower memory for HALO.

ReID keeps identity persistent; this module keeps *episodic* memory —
who has been near the wearer, for how long, across how many reappearances —
and decides when a pattern crosses from coincidence into "following".

The decision is deterministic. An optional LLM narrator (Llama API via
HALO_LLM_KEY) only writes the human-readable sentence; if the API is
unavailable a template is used. Safety logic never depends on the LLM.
"""
from __future__ import annotations

import math
import os
import threading
import urllib.request
import json

# thresholds (seconds / counts) — env-tunable for demos
FOLLOW_COMOVE_S = float(os.environ.get("HALO_FOLLOW_S", 45.0))
FOLLOW_REAPPEARS = int(os.environ.get("HALO_FOLLOW_REAP", 2))
FOLLOW_NEAR_M = float(os.environ.get("HALO_FOLLOW_NEAR", 18.0))
FOLLOW_GAP_S = float(os.environ.get("HALO_FOLLOW_GAP", 10.0))
WEARER_MOVING_MS = 0.4       # wearer must be moving for co-movement to count
REARM_GAP_S = 120.0          # re-alert allowed after this silence


class FollowMemory:
    def __init__(self):
        self._subjects = {}      # oid -> episode dict
        self._alerts = {}        # oid -> last alert ts
        self._narrations = {}    # oid -> llm/template sentence
        self._pending = set()    # oids with a narration request in flight

    # -------------------------------------------------------------
    def update(self, tracks: list[dict], wearer_speed: float, ts: float,
               wearer_pos=(0.0, 0.0)) -> list[dict]:
        """Fold one frame of world-state tracks into episodic memory.

        Returns active follower alerts (possibly empty).
        """
        seen_now = set()
        for t in tracks:
            oid = t.get("id")
            if t.get("cls") != "person" or not oid:
                continue
            pos = t.get("position") or [0, 0]
            dist = math.hypot(pos[0] - wearer_pos[0], pos[1] - wearer_pos[1])
            ep = self._subjects.setdefault(oid, {
                "first": ts, "last": ts, "seen_s": 0.0, "comove_s": 0.0,
                "reappears": 0, "reid": False, "min_dist": dist,
                "prev_ts": ts,
            })
            dt = min(ts - ep["prev_ts"], 0.5)      # clamp gaps
            ep["prev_ts"] = ts
            ep["seen_s"] += max(dt, 0)
            if ts - ep["last"] > FOLLOW_GAP_S:
                ep["reappears"] += 1             # lost then refound — ReID win
            ep["last"] = ts
            ep["reid"] = ep["reid"] or bool(t.get("reid"))
            ep["min_dist"] = min(ep["min_dist"], dist)
            near = dist < FOLLOW_NEAR_M
            moving = wearer_speed > WEARER_MOVING_MS
            if near and moving:
                ep["comove_s"] += max(dt, 0)
            seen_now.add(oid)

        active = []
        for oid, ep in self._subjects.items():
            following = (ep["comove_s"] >= FOLLOW_COMOVE_S
                         or (ep["reappears"] >= FOLLOW_REAPPEARS
                             and ep["comove_s"] >= FOLLOW_COMOVE_S * 0.45))
            if not following:
                continue
            last_alert = self._alerts.get(oid)
            fresh = last_alert is None or ts - last_alert > REARM_GAP_S
            if fresh:
                self._alerts[oid] = ts
                self._narrate_async(oid, ep)
            active.append({
                "id": oid,
                "comove_s": round(ep["comove_s"], 0),
                "reappears": ep["reappears"],
                "min_dist": round(ep["min_dist"], 1),
                "reid": ep["reid"],
                "narration": self._narrations.get(oid),
                "fresh": fresh,
            })
        # prune stale episodes
        for oid in [o for o, ep in self._subjects.items() if ts - ep["last"] > 300]:
            self._subjects.pop(oid, None)
        return active

    # -------------------------------------------------------------
    def _narrate_async(self, oid: str, ep: dict) -> None:
        if oid in self._pending:
            return
        self._pending.add(oid)

        def run():
            try:
                self._narrations[oid] = narrate_follower(oid, ep)
            finally:
                self._pending.discard(oid)

        threading.Thread(target=run, daemon=True).start()

    def reset(self) -> None:
        self._subjects.clear()
        self._alerts.clear()
        self._narrations.clear()
        self._pending.clear()


# ---------------------------------------------------------------
def narrate_follower(oid: str, ep: dict) -> str:
    """One calm sentence describing the pattern. LLM if configured,
    deterministic template otherwise."""
    key = os.environ.get("HALO_LLM_KEY", "")
    base = os.environ.get("HALO_LLM_BASE", "https://api.meta.ai/v1")
    model = os.environ.get("HALO_LLM_MODEL", "muse-spark-1.3")
    facts = (f"subject id {oid}; moving near wearer for "
             f"{ep['comove_s']:.0f}s; reappeared {ep['reappears']} time(s) "
             f"after losing sight; closest approach {ep['min_dist']:.1f}m; "
             f"identity confirmed by ReID: {ep['reid']}")
    if key:
        try:
            body = json.dumps({
                "model": model,
                "max_tokens": 800,          # muse-spark reasons internally; leave headroom
                "reasoning_effort": "minimal",
                "messages": [
                    {"role": "system", "content":
                     "You are HALO's memory module for a wearable safety "
                     "system. Given tracking facts about a possibly "
                     "following person, write ONE calm factual sentence to "
                     "the wearer, second person, no alarm language, under "
                     "25 words."},
                    {"role": "user", "content": facts},
                ],
            }).encode()
            req = urllib.request.Request(
                f"{base}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                out = json.loads(r.read())
                msg = (out["choices"][0]["message"].get("content") or "").strip()
                if msg:
                    return msg
        except Exception:
            pass
    # deterministic fallback
    mins = ep["comove_s"] / 60
    if ep["reappears"] >= FOLLOW_REAPPEARS:
        return (f"{oid} has reappeared near you {ep['reappears']}× and kept "
                f"pace for {ep['comove_s']:.0f}s — worth a glance over your shoulder.")
    return (f"{oid} has stayed within {FOLLOW_NEAR_M:.0f}m of you for "
            f"{mins:.1f} min while you move — persistent co-movement.")
