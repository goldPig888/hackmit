"use strict";
/* HALO Predictive Safety Console — visualization only.
   All safety math happens server-side; this file renders /api/world snapshots. */

const POLL_MS = 100;
const GHOST_TIMES = [0.5, 1.0, 1.5, 2.0];
const DEMO = new URLSearchParams(location.search).has("demo");

const S = {
    prev: null, curr: null, currAt: 0, prevAt: 0,
    snapshots: [],            // ring buffer for replay
    linked: null,             // cross-view highlighted track id
    stab: true,               // stabilized vs raw trail toggle
    replay: null,             // {idx} while replaying
    lastEventId: 0,
    worldScale: 30,
};

const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const deg = r => r * 180 / Math.PI;

function riskColor(r, a = 1) {
    if (r > 0.7) return `rgba(244,63,94,${a})`;
    if (r > 0.4) return `rgba(251,191,36,${a})`;
    return `rgba(52,211,153,${a})`;
}
const CLS_COLORS = {
    car: "#fbbf24", truck: "#fb923c", bus: "#fb923c", motorcycle: "#f472b6",
    bicycle: "#a78bfa", person: "#34d399",
    scissors: "#f43f5e", knife: "#f43f5e",
    "baseball bat": "#fb923c", "tennis racket": "#fb923c",
    bottle: "#94a3b8", umbrella: "#94a3b8",
};
const clsColor = c => CLS_COLORS[c] || "#22d3ee";
const SHARP_OBJ = new Set(["scissors", "knife"]);
const heldTag = t => t.held_object
    ? " ·" + (SHARP_OBJ.has(t.held_object) ? "✂" : "") + t.held_object.toUpperCase()
    : "";

/* ================= MOCK ADAPTER (clearly isolated demo data) ================ */
/* Used only with ?demo — generates a synthetic but schema-complete state so
   the full console is visible without the phone backend. */
const SCENARIO = new URLSearchParams(location.search).get("demo") || "car";

// virtual rear-facing camera used to render the simulated feed in ?demo
const SIM_CAM = { W: 640, H: 480, F: 300, CAMH: 1.3, HORIZON: 0.42 };
const SIM_DIMS = { person: [0.5, 1.7], car: [0.8, 0.55], truck: [1.2, 1.4],
                   bus: [1.2, 1.4], motorcycle: [0.6, 1.2], bicycle: [0.5, 1.1] };

function simProject(wx, wy, yaw, pitch) {
    const V = yaw + Math.PI;                       // rear camera looks backward
    const depth = wx * Math.sin(V) + wy * Math.cos(V);
    const lat = -(wx * Math.cos(V) - wy * Math.sin(V)); // mirrored: behind-left → left
    if (depth < 0.4) return null;
    const horizon = SIM_CAM.H * SIM_CAM.HORIZON + pitch * SIM_CAM.F;
    const sx = SIM_CAM.W / 2 + (lat / depth) * SIM_CAM.F;
    const feetY = horizon + (SIM_CAM.CAMH / depth) * SIM_CAM.F;
    return { sx, feetY, depth, lat };
}

const Mock = {
    t0: performance.now() / 1000,
    // world → 640x480 rear-camera image: fills bbox + pixel trails so the
    // normal overlay pipeline renders on the simulated scene for free
    project(tracks, yaw, pitch) {
        for (const t of tracks) {
            const p = simProject(t.position[0], t.position[1], yaw, pitch);
            const dims = SIM_DIMS[t.cls] || [0.6, 0.9];
            if (!p || Math.abs(p.lat / p.depth) > 1.6) { t.bbox = null; t.pixel_predicted = []; t.pixel_history = []; t.pixel_history_stab = []; continue; }
            const wp = (dims[0] / p.depth) * SIM_CAM.F, hp = (dims[1] / p.depth) * SIM_CAM.F;
            t.bbox = [p.sx - wp / 2, p.feetY - hp, wp, hp];
            const mid = q => [q.sx, q.feetY - (dims[1] / q.depth) * SIM_CAM.F * 0.55];
            t.pixel_predicted = (t.predicted_path || []).map(pt => {
                const q = simProject(pt[0], pt[1], yaw, pitch);
                return q && q.depth > 0.4 ? mid(q) : null;
            });
            t.pixel_history = (t.history || []).map(pt => {
                const q = simProject(pt[0], pt[1], yaw, pitch);
                return q && q.depth > 0.4 ? mid(q) : null;
            });
            t.pixel_history_stab = t.pixel_history;
        }
    },
    ambient(scene, n, t) {
        // background pedestrians milling — the "perceived but unattended" layer
        return Array.from({ length: n }, (_, i) => {
            const a = (i / n) * Math.PI * 2 + Math.sin(t * 0.1 + i) * 0.1;
            const d = 6 + (i % 4) * 2.5;
            const x = Math.sin(a) * d, y = Math.cos(a) * d;
            return {
                id: `person-${20 + i}`, cls: "person", confidence: 0.8,
                bbox: [100 + (i % 5) * 100, 400 + (i % 3) * 60, 42, 110],
                position: [x, y], velocity: [Math.sin(i) * 0.3, Math.cos(i) * 0.3],
                speed: 0.4, bearing: Math.atan2(x, y), bearing_rate: 0.005,
                bearing_rel: Math.atan2(x, y), closing_rate: 0.0,
                history: [], predicted_times: Array.from({ length: 21 }, (_, k) => k * 0.2),
                predicted_path: Array.from({ length: 21 }, (_, k) => [x + Math.sin(i) * 0.3 * k * 0.2, y + Math.cos(i) * 0.3 * k * 0.2]),
                pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
                sigma: 0.5, ttc: null, tcpa: null, dcpa: null, risk: 0.04,
                risk_level: "safe", conflict_type: "safe_pass", will_collide: false,
                conflict_point: null, direction: "center", direction_label: "AROUND",
                factors: {}, expanding: false, area_rate: 0.02, raw_px_rate: 8,
            evidence: { proximity: 0.4, persistence: 0.3, rapid_approach: 0,
                        heading_toward: 0, unusual_motion: 0.05, persistent_comovement: 0 },
                attention: 0.12, att_state: "OBSERVE", reasons: [],
            };
        });
    },
    state() {
        const t = performance.now() / 1000 - this.t0;
        const cyc = (t % 16) / 16;                       // 16s threat cycle
        const yaw = 0.35 * Math.sin(t * 0.5);            // rider weaving
        const carD = lerp(22, 4, Math.min(cyc * 1.3, 1)); // car closing from rear-left
        const carB = -2.4 + 0.2 * Math.sin(t * 0.3);
        const carX = Math.sin(carB) * carD, carY = Math.cos(carB) * carD;
        const colliding = carD < 9;
        const risk = clamp(1 - carD / 12, 0, 1) * (colliding ? 1 : 0.55);

        const mkPath = (x0, y0, vx, vy, n = 21) =>
            Array.from({ length: n }, (_, i) => [x0 + vx * i * 0.2, y0 + vy * i * 0.2]);
        const carPath = mkPath(carX, carY, -carX / carD * 6, -carY / carD * 6);
        const egoPath = mkPath(0, 0, Math.sin(yaw) * 1.5, Math.cos(yaw) * 1.5);
        const straightPath = mkPath(0, 0, Math.sin(yaw) * 1.5, Math.cos(yaw) * 1.5);

        const tracks = [{
            id: "car-7", cls: "car", confidence: 0.94,
            bbox: [420 - carD * 8, 300, 90 + (22 - carD) * 8, 60 + (22 - carD) * 5],
            position: [carX, carY], velocity: [-carX / carD * 6, -carY / carD * 6],
            speed: 6, bearing: Math.atan2(carX, carY), bearing_rate: -0.02,
            bearing_rel: Math.atan2(carX, carY) - yaw, closing_rate: -6 * (1 - carD / 30),
            history: mkPath(carX * 1.15, carY * 1.15, 0, 0).map((p, i) => [lerp(carX * 1.15, carX, i / 20), lerp(carY * 1.15, carY, i / 20)]),
            predicted_times: Array.from({ length: 21 }, (_, i) => i * 0.2),
            predicted_path: carPath,
            pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
            sigma: 0.6 + (22 - carD) * 0.02, ttc: carD / 6, tcpa: carD / 6.5,
            dcpa: colliding ? 1.2 : 6.0, risk,
            risk_level: risk > 0.7 ? "high" : risk > 0.4 ? "medium" : "low",
            conflict_type: colliding ? "collision" : "safe_pass",
            will_collide: colliding, conflict_point: colliding ? carPath[Math.min(8, carPath.length - 1)] : null,
            direction: "left", direction_label: "BEHIND LEFT",
            factors: {}, expanding: carD < 15, area_rate: carD < 15 ? 0.4 : 0.05, raw_px_rate: 40,
            attention: clamp(risk + 0.2, 0, 1),
            att_state: colliding ? "WARN" : (risk > 0.35 ? "ATTEND" : "OBSERVE"),
            reasons: colliding ? ["approaching wearer", "predicted paths overlap"] : ["approaching wearer"],
            evidence: { rapid_approach: clamp(-(carD - 22) / 18, 0, 1), proximity: clamp(1 - carD / 20, 0, 1),
                        heading_toward: 0.7, trajectory_conflict: colliding ? 1 : 0.2,
                        rapid_expansion: carD < 15 ? 0.5 : 0.1, strike_motion: 0, persistence: 0.3 },
        }, {
            id: "person-12", cls: "person", confidence: 0.88, bbox: [80, 320, 40, 110],
            position: [8, 12], velocity: [0, 0.4], speed: 0.4,
            bearing: Math.atan2(8, 12), bearing_rate: 0.01, bearing_rel: Math.atan2(8, 12) - yaw,
            closing_rate: 0.1, history: mkPath(8.5, 12.5, 0, 0),
            predicted_times: Array.from({ length: 21 }, (_, i) => i * 0.2),
            predicted_path: mkPath(8, 12, 0, 0.4),
            pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
            sigma: 0.5, ttc: null, tcpa: null, dcpa: null, risk: 0.05,
            risk_level: "safe", conflict_type: "safe_pass", will_collide: false,
            conflict_point: null, direction: "right", direction_label: "FRONT RIGHT",
            factors: {}, expanding: false, raw_px_rate: 5,
            attention: 0.15, att_state: "OBSERVE", reasons: [],
        }];

        // -------- scenario lab: ?demo=car|crowded|sparse|lunge|comove|closepass|conflict --------
        let density = "normal", environment = "outdoor";
        if (SCENARIO === "closepass") {
            // A: RC vehicle sweeps past VERY close behind-left but parallel —
            // looks scary, predicted path is safe → tracks, never buzzes.
            density = "sparse"; environment = "outdoor";
            tracks.length = 0;
            const ph = (t % 10) / 10;                 // 10s pass cycle
            const cx = lerp(-6, -0.9, Math.min(ph * 1.6, 1));  // sweeps to 0.9m lateral
            const cy = lerp(-14, 10, ph);             // behind → ahead, parallel
            const near = Math.hypot(cx, cy) < 4;
            tracks.push({
                reid: true, id: "car-1", cls: "car", confidence: 0.9,
                bbox: [200, 380, 140, 100],
                position: [cx, cy], velocity: [0.05, 6.0], speed: 6.0,
                bearing: Math.atan2(cx, cy), bearing_rate: 0.02,
                bearing_rel: Math.atan2(cx, cy) - yaw, closing_rate: 0.1,
                history: mkPath(cx, cy - 8, 0, 6),
                predicted_times: Array.from({ length: 21 }, (_, i) => i * 0.2),
                predicted_path: mkPath(cx, cy, 0.05, 6.0),
                pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
                sigma: 0.4, ttc: null, tcpa: 2.4, dcpa: 0.9,
                risk: near ? 0.18 : 0.08,             // close but SAFE — risk stays low
                risk_level: "low", conflict_type: "safe_pass",
                will_collide: false, conflict_point: null,
                direction: "left", direction_label: near ? "LEFT" : "BEHIND LEFT",
                factors: {}, expanding: near, area_rate: near ? 0.3 : 0.1, raw_px_rate: 60,
                attention: near ? 0.45 : 0.2,
                att_state: near ? "ATTEND" : "OBSERVE",
                reasons: near ? ["close range (0.9 m)"] : [],
                evidence: { proximity: near ? 0.9 : 0.4, trajectory_conflict: 0,
                            rapid_approach: 0.05, heading_toward: 0.1 },
            });
        } else if (SCENARIO === "conflict") {
            // B: vehicle still FAR away — but the wearer is turning into its
            // path. HALO predicts the intersection before they're close.
            density = "sparse"; environment = "outdoor";
            tracks.length = 0;
            const turning = Math.sin(t * 0.5) > 0.2;   // periodic left turn
            const cx = -7, cy = -16;                    // 17m behind-left
            tracks.push({
                reid: true, id: "car-2", cls: "car", confidence: 0.87,
                bbox: [160, 400, 90, 60],
                position: [cx, cy], velocity: [0.5, 5.0], speed: 5.0,
                bearing: Math.atan2(cx, cy), bearing_rate: -0.01,
                bearing_rel: Math.atan2(cx, cy) - yaw, closing_rate: -1.2,
                history: mkPath(cx, cy - 10, 0, 5),
                predicted_times: Array.from({ length: 21 }, (_, i) => i * 0.2),
                predicted_path: mkPath(cx, cy, 0.5, 5.0),
                pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
                sigma: 0.5, ttc: 14, tcpa: 1.8, dcpa: turning ? 0.7 : 6.0,
                risk: turning ? 0.82 : 0.25,
                risk_level: turning ? "high" : "low",
                conflict_type: turning ? "collision" : "safe_pass",
                will_collide: turning,
                conflict_point: turning ? [-2, -6] : null,
                direction: "left", direction_label: "BEHIND LEFT",
                factors: {}, expanding: false, area_rate: 0.05, raw_px_rate: 25,
                attention: turning ? 0.9 : 0.35,
                att_state: turning ? "WARN" : "ATTEND",
                reasons: turning
                    ? ["predicted paths overlap", "bearing stable", "rider turning"]
                    : ["persistent approach"],
                evidence: turning
                    ? { trajectory_conflict: 1.0, heading_toward: 0.7, rapid_approach: 0.3, proximity: 0.2 }
                    : { heading_toward: 0.6, persistence: 0.5, proximity: 0.15 },
            });
        } else if (SCENARIO === "crowded") {
            density = "crowded"; environment = "indoor";
            tracks.push(...this.ambient("crowded", 16, t));
        } else if (SCENARIO === "sparse" || SCENARIO === "comove") {
            density = "sparse"; environment = "indoor";
            tracks.length = 0;
            // one person co-moving behind the wearer: attention, not alarm
            const bx = -2 + Math.sin(yaw) * 0.5, by = -7;   // behind-left, ~7m
            tracks.push({
                id: "person-4", cls: "person", confidence: 0.9, bbox: [180, 500, 55, 150],
                position: [bx, by], velocity: [0, 0], speed: 1.4,
                bearing: Math.atan2(bx, by), bearing_rate: 0.01,
                bearing_rel: Math.atan2(bx, by), closing_rate: 0.0,
                history: mkPath(bx * 1.3, by * 1.3, 0, 0),
                predicted_times: Array.from({ length: 21 }, (_, i) => i * 0.2),
                predicted_path: mkPath(bx, by, Math.sin(yaw) * 1.4, Math.cos(yaw) * 1.4),
                pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
                sigma: 0.4, ttc: null, tcpa: null, dcpa: null, risk: 0.1,
                risk_level: "low", conflict_type: "safe_pass", will_collide: false,
                conflict_point: null, direction: "left", direction_label: "BEHIND LEFT",
                factors: {}, expanding: false, raw_px_rate: 12,
                attention: 0.62, att_state: "ATTEND",
                reasons: ["persistent co-movement", "bearing stable", "tracking 12.0s"],
                evidence: { persistent_comovement: 0.8, persistence: 0.8, proximity: 0.55,
                            heading_toward: 0.2, rapid_approach: 0.05, unusual_motion: 0.1 },
            });
        } else if (SCENARIO === "lunge") {
            density = "sparse"; environment = "indoor";
            // person lunges toward the wearer every 8s — reflex path demo
            const ph = (t % 8) / 8;
            const lunging = ph > 0.55 && ph < 0.85;
            const pd = lunging ? lerp(7, 2.2, (ph - 0.55) / 0.3) : lerp(10, 7, ph / 0.55);
            const px2 = Math.sin(0.5) * pd, py2 = Math.cos(0.5) * pd;
            const pClose = lunging ? -6.5 : -0.4;
            tracks.push({
                id: "person-9", cls: "person", confidence: 0.91,
                bbox: [260 - (10 - pd) * 14, 380, 60 + (10 - pd) * 10, 160 + (10 - pd) * 16],
                position: [px2, py2], velocity: [pClose * Math.sin(0.5), pClose * Math.cos(0.5)],
                speed: Math.abs(pClose), bearing: Math.atan2(px2, py2), bearing_rate: 0.01,
                bearing_rel: Math.atan2(px2, py2), closing_rate: pClose,
                history: mkPath(px2 * 1.4, py2 * 1.4, 0, 0),
                predicted_times: Array.from({ length: 21 }, (_, i) => i * 0.2),
                predicted_path: mkPath(px2, py2, pClose * Math.sin(0.5), pClose * Math.cos(0.5)),
                pixel_history: [], pixel_predicted: [], pixel_history_stab: [],
                sigma: 0.5, ttc: pd / Math.abs(pClose), tcpa: pd / Math.abs(pClose),
                dcpa: 0.8, risk: lunging ? 0.85 : 0.15,
                risk_level: lunging ? "high" : "low",
                conflict_type: lunging ? "collision" : "safe_pass",
                will_collide: lunging, conflict_point: lunging ? [0.3, 0.4] : null,
                direction: "right", direction_label: "FRONT RIGHT",
                factors: {}, expanding: lunging, area_rate: lunging ? 1.1 : 0.05,
                raw_px_rate: lunging ? 260 : 15,
                attention: lunging ? 1.0 : 0.2,
                att_state: lunging ? "REFLEX" : "OBSERVE",
                evidence: lunging
                    ? { rapid_approach: 0.95, rapid_expansion: 0.9, strike_motion: 0.72,
                        heading_toward: 0.85, proximity: 0.7, trajectory_conflict: 0.8 }
                    : { proximity: 0.3, persistence: 0.4 },
                reasons: lunging
                    ? ["rapid approach at close range", "closing 6.5 m/s", "image expanding fast", `est. contact ${(pd / 6.5).toFixed(1)}s`]
                    : [],
            });
        }

        const attCount = tracks.filter(x => x.att_state !== "OBSERVE").length;
        const watchCount = tracks.filter(x => x.att_state === "WARN" || x.att_state === "REFLEX").length;
        const threatCount = tracks.filter(x => x.att_state === "REFLEX" || x.will_collide).length;

        this.project(tracks, yaw, 0.1);

        const prim = tracks.find(x => x.att_state === "REFLEX")
            || tracks.find(x => x.will_collide)
            || tracks.reduce((a, b) => (b.attention > (a?.attention ?? -1) ? b : a), null);
        const threatTrack = prim && prim.att_state !== "OBSERVE" ? prim : null;

        return {
            ready: true, timestamp: t,
            system: { fps: 10, camera: true, imu: true, tracking: tracks.length,
                      haptic_endpoint: false, world_model: "STABLE", frame_size: [SIM_CAM.W, SIM_CAM.H] },
            camera_pose: { yaw, pitch: 0.1, roll: 0, confidence: 0.9 },
            scene: { density, environment,
                     people: tracks.filter(x => x.cls === "person").length,
                     vehicles: tracks.filter(x => x.cls !== "person").length,
                     ego_motion: "walking",
                     reflex: tracks.some(x => x.att_state === "REFLEX") },
            rider_state: { position: [0, 0], velocity: [Math.sin(yaw) * 1.5, Math.cos(yaw) * 1.5],
                           speed: 1.5, heading: yaw, yaw_rate: 0.3 * Math.cos(t * 0.5),
                           turning: Math.abs(0.3 * Math.cos(t * 0.5)) > 0.12,
                           turn_direction: yaw > 0 ? "left" : "right",
                           predicted_path: egoPath, straight_path: straightPath },
            tracks,
            primary_threat_id: threatTrack ? threatTrack.id : null,
            threat: threatTrack ? { id: threatTrack.id, cls: threatTrack.cls,
                      direction_label: threatTrack.direction_label,
                      att_state: threatTrack.att_state, attention: threatTrack.attention,
                      reasons: threatTrack.reasons, evidence: threatTrack.evidence,
                      secondary: (() => {
                          const s = tracks.find(x => x !== threatTrack && x.att_state !== "OBSERVE");
                          return s ? { id: s.id, cls: s.cls, att_state: s.att_state,
                                       attention: s.attention, direction_label: s.direction_label,
                                       reasons: s.reasons } : null;
                      })(),
                      risk: threatTrack.risk,
                      ttc: threatTrack.ttc, tcpa: threatTrack.tcpa,
                      dcpa: threatTrack.dcpa,
                      closing_rate: threatTrack.closing_rate, confidence: 0.82,
                      why: [
                          { label: "closing rapidly", active: threatTrack.closing_rate < -0.5 },
                          { label: "bearing stable", active: true },
                          { label: "rider turning", active: Math.abs(0.3 * Math.cos(t * 0.5)) > 0.12 },
                          { label: "predicted paths overlap", active: threatTrack.will_collide },
                          { label: "expanding in frame", active: threatTrack.expanding }]}
                      : null,
            haptic: { left: threatTrack && threatTrack.risk > 0.5 ? threatTrack.risk : 0,
                      right: 0, center: 0,
                      direction: threatTrack ? threatTrack.direction : null,
                      intensity: threatTrack && threatTrack.att_state === "REFLEX" ? "strong" : "medium", ts: t },
            counterfactual: { current_cpa: threatTrack ? threatTrack.dcpa : null,
                              straight_cpa: 5.5,
                              current_risk: threatTrack ? threatTrack.risk : 0,
                              straight_risk: 0.2,
                              delta: threatTrack ? threatTrack.risk - 0.2 : 0 },
            funnel: { perceived: tracks.length, attended: attCount,
                      watch: watchCount, threat: threatCount },
            events: [
                { id: 1, ts: t - 6, type: "TRACK ACQUIRED", label: threatTrack ? `${threatTrack.cls} ${threatTrack.id}` : "—" },
                { id: 2, ts: t - 4, type: "CLOSING DETECTED", label: "-5.4 m/s" },
                { id: 3, ts: t - 2, type: "RIDER TURN STARTED", label: "left" },
                ...(threatTrack && threatTrack.att_state === "REFLEX"
                    ? [{ id: 4, ts: t - 0.5, type: "REFLEX EVENT", label: "rapid approach", risk: 1 }]
                    : threatTrack && threatTrack.will_collide
                        ? [{ id: 4, ts: t - 1, type: "PATH CONFLICT", label: "tCPA 1.2s", risk: threatTrack.risk }] : []),
            ],
            risk_history: Array.from({ length: 80 }, (_, i) => [t - 8 + i * 0.1, clamp(1 - (lerp(30, threatTrack ? Math.hypot(...threatTrack.position) : 30, i / 79)) / 12, 0, 1)]),
            horizon_s: 4,
        };
    }
};
/* ================= end mock adapter ======================================== */

/* ---------------- data ---------------- */
async function poll() {
    try {
        const r = await fetch("/api/world");
        const s = await r.json();
        if (s && s.ready) ingest(s);
    } catch (e) { /* server down — keep last frame */ }
}

function ingest(s) {
    S.prev = S.curr; S.prevAt = S.currAt;
    S.curr = s; S.currAt = performance.now();
    S.snapshots.push(s);
    if (S.snapshots.length > 80) S.snapshots.shift();
}

function activeState() {
    if (DEMO) return Mock.state();
    if (S.replay) return S.snapshots[S.replay.idx] || S.curr;
    return S.curr;
}

function interpTracks(s) {
    if (!s || !s.tracks) return [];
    const now = performance.now();
    S.seen = S.seen || {};
    const cur = new Set();
    for (const t of s.tracks) { cur.add(t.id); S.seen[t.id] = { t, at: now }; }
    let out;
    if (!S.prev || DEMO || S.replay) {
        out = s.tracks.slice();
    } else {
        const a = clamp((now - S.currAt) / POLL_MS, 0, 1);
        const prevById = Object.fromEntries((S.prev.tracks || []).map(t => [t.id, t]));
        out = s.tracks.map(t => {
            const p = prevById[t.id];
            if (!p) return t;
            return { ...t,
                position: [lerp(p.position[0], t.position[0], a), lerp(p.position[1], t.position[1], a)],
                risk: lerp(p.risk || 0, t.risk || 0, a) };
        });
    }
    // hold recently-lost tracks as fading ghosts (~0.9s) so a single missed
    // detection doesn't make a box blink out and back
    for (const [id, g] of Object.entries(S.seen)) {
        if (cur.has(id)) continue;
        const age = now - g.at;
        if (age > 900) { delete S.seen[id]; continue; }
        out.push({ ...g.t, fading: 1 - age / 900 });
    }
    return out;
}

/* ---------------- camera feed + overlay ---------------- */
const camCv = $("camOverlay"), camWrap = $("camWrap");
const cctx = camCv.getContext("2d");
let camBmp = null, camFetchBusy = false;

async function decodeCameraFrame(blob) {
    // createImageBitmap is fast, but Safari can reject a valid JPEG during a
    // transient decoder reset. Falling back to HTMLImageElement keeps the live
    // overlay alive instead of leaving a black canvas until the next reload.
    if (window.createImageBitmap) {
        try { return await createImageBitmap(blob); } catch (_) { /* fallback below */ }
    }
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(blob);
        const image = new Image();
        image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
        image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("JPEG decode failed")); };
        image.src = url;
    });
}

async function pollFrame() {
    if (camFetchBusy) return;
    camFetchBusy = true;
    try {
        const r = await fetch("/frame.jpg", { cache: "no-store" });
        if (r.status === 200) {
            const bmp = await decodeCameraFrame(await r.blob());
            if (camBmp && typeof camBmp.close === "function") camBmp.close();
            camBmp = bmp;
        }
    } catch (e) { /* transient — keep last bitmap */ }
    camFetchBusy = false;
}
setInterval(pollFrame, 90);
pollFrame();

function camRect(fw, fh) {
    const r = camWrap.getBoundingClientRect();
    const sc = Math.min(r.width / fw, r.height / fh);
    const w = fw * sc, h = fh * sc;
    return { x: (r.width - w) / 2, y: (r.height - h) / 2, w, h, sc };
}

// synthetic ego-view: perspective ground grid + track sprites, so ?demo
// renders a plausible rear-camera feed instead of a black panel
function drawSimScene(m, s, tracks) {
    const toScr = p => [m.x + p[0] * m.sc, m.y + p[1] * m.sc];
    const yaw = s.camera_pose.yaw, pitch = s.camera_pose.pitch;
    const horizon = SIM_CAM.H * SIM_CAM.HORIZON + pitch * SIM_CAM.F;
    const [hx0, hy] = toScr([0, horizon]);

    cctx.save();
    cctx.beginPath(); cctx.rect(m.x, m.y, m.w, m.h); cctx.clip();

    // sky / ground
    const sky = cctx.createLinearGradient(0, m.y, 0, hy);
    sky.addColorStop(0, "#0d1319"); sky.addColorStop(1, "#080c10");
    cctx.fillStyle = sky; cctx.fillRect(m.x, m.y, m.w, Math.max(0, hy - m.y));
    const gnd = cctx.createLinearGradient(0, hy, 0, m.y + m.h);
    gnd.addColorStop(0, "#0d1013"); gnd.addColorStop(1, "#060809");
    cctx.fillStyle = gnd; cctx.fillRect(m.x, Math.max(m.y, hy), m.w, m.h - Math.max(0, hy - m.y));

    // range rings on the ground plane
    cctx.strokeStyle = "rgba(90,180,200,0.10)"; cctx.lineWidth = 1;
    for (const d of [0.8, 1.2, 2, 3, 4, 5, 7, 10, 14, 20, 30]) {
        const y = horizon + (SIM_CAM.CAMH / d) * SIM_CAM.F;
        if (y < horizon + 2 || y > SIM_CAM.H + 20) continue;
        const [lx, ly] = toScr([0, y]);
        cctx.beginPath(); cctx.moveTo(m.x, ly); cctx.lineTo(m.x + m.w, ly); cctx.stroke();
        if ([1, 2, 5, 10, 20].includes(d)) {
            cctx.fillStyle = "rgba(90,180,200,0.28)"; cctx.font = "8px monospace";
            cctx.fillText(`${d}m`, m.x + 4, ly - 2);
        }
    }
    // radial lanes converging on the wearer
    for (const lat of [-8, -6, -4, -2, 0, 2, 4, 6, 8]) {
        const near = SIM_CAM.W / 2 + (lat / 0.6) * SIM_CAM.F;
        const far = SIM_CAM.W / 2 + (lat / 30) * SIM_CAM.F;
        const [ax, ay] = toScr([near, horizon + (SIM_CAM.CAMH / 0.6) * SIM_CAM.F]);
        const [bx, by] = toScr([far, horizon + (SIM_CAM.CAMH / 30) * SIM_CAM.F]);
        cctx.beginPath(); cctx.moveTo(ax, ay); cctx.lineTo(bx, by); cctx.stroke();
    }
    // horizon glow
    cctx.strokeStyle = "rgba(34,211,238,0.25)"; cctx.lineWidth = 1.5;
    cctx.beginPath(); cctx.moveTo(m.x, hy); cctx.lineTo(m.x + m.w, hy); cctx.stroke();

    // sprites, far → near (painter's order)
    const visible = tracks.filter(t => t.bbox)
        .map(t => ({ t, p: simProject(t.position[0], t.position[1], yaw, pitch) }))
        .filter(o => o.p).sort((a, b) => b.p.depth - a.p.depth);
    for (const { t, p } of visible) {
        const [bx, by] = toScr([t.bbox[0], t.bbox[1]]);
        const bw = t.bbox[2] * m.sc, bh = t.bbox[3] * m.sc;
        const col = clsColor(t.cls);
        // ground shadow
        const [fx, fy] = toScr([p.sx, p.feetY]);
        cctx.fillStyle = "rgba(0,0,0,0.5)";
        cctx.beginPath(); cctx.ellipse(fx, fy, bw * 0.55, bw * 0.14, 0, 0, Math.PI * 2); cctx.fill();

        if (t.cls === "person") {
            // capsule body + head
            cctx.fillStyle = "rgba(24,32,42,0.95)";
            cctx.strokeStyle = col; cctx.lineWidth = 1;
            cctx.beginPath();
            cctx.roundRect(bx + bw * 0.15, by + bh * 0.18, bw * 0.7, bh * 0.82, bw * 0.3);
            cctx.fill(); cctx.stroke();
            cctx.beginPath();
            cctx.arc(bx + bw / 2, by + bh * 0.11, Math.max(2, bw * 0.16), 0, Math.PI * 2);
            cctx.fill(); cctx.stroke();
        } else {
            // vehicle: body + cabin + wheels + headlights (facing camera)
            cctx.fillStyle = "rgba(26,34,44,0.97)";
            cctx.strokeStyle = col; cctx.lineWidth = 1;
            cctx.beginPath();
            cctx.roundRect(bx, by + bh * 0.35, bw, bh * 0.55, bw * 0.12);
            cctx.fill(); cctx.stroke();
            cctx.beginPath();
            cctx.roundRect(bx + bw * 0.18, by + bh * 0.08, bw * 0.64, bh * 0.34, bw * 0.08);
            cctx.fill(); cctx.stroke();
            cctx.fillStyle = "#05070a";
            cctx.beginPath(); cctx.ellipse(bx + bw * 0.16, by + bh * 0.9, bw * 0.13, bh * 0.1, 0, 0, Math.PI * 2); cctx.fill();
            cctx.beginPath(); cctx.ellipse(bx + bw * 0.84, by + bh * 0.9, bw * 0.13, bh * 0.1, 0, 0, Math.PI * 2); cctx.fill();
            // headlight glow
            cctx.fillStyle = "rgba(255,220,140,0.85)";
            for (const lx of [bx + bw * 0.2, bx + bw * 0.8]) {
                cctx.beginPath(); cctx.arc(lx, by + bh * 0.55, Math.max(1.5, bw * 0.05), 0, Math.PI * 2); cctx.fill();
            }
        }
    }

    // vignette
    const vg = cctx.createRadialGradient(m.x + m.w / 2, m.y + m.h / 2, m.h * 0.35,
                                         m.x + m.w / 2, m.y + m.h / 2, m.w * 0.75);
    vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.45)");
    cctx.fillStyle = vg; cctx.fillRect(m.x, m.y, m.w, m.h);
    cctx.restore();
}

function drawCamera(s, tracks) {
    const r = camWrap.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    // Keep canvas backing pixels aligned with its CSS box on Retina displays.
    const dpr = window.devicePixelRatio || 1;
    camCv.width = Math.round(r.width * dpr); camCv.height = Math.round(r.height * dpr);
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cctx.fillStyle = "#000"; cctx.fillRect(0, 0, r.width, r.height);
    $("camNoFeed").classList.toggle("show", !camBmp && !DEMO);

    const fs = s.system && s.system.frame_size;
    const fw = camBmp ? camBmp.width : (fs ? fs[0] : 0);
    const fh = camBmp ? camBmp.height : (fs ? fs[1] : 0);
    if (!fw || !fh) return;

    const m = camRect(fw, fh);
    if (camBmp) cctx.drawImage(camBmp, m.x, m.y, m.w, m.h);
    else if (DEMO) drawSimScene(m, s, tracks);
    const toScr = p => [m.x + p[0] * m.sc, m.y + p[1] * m.sc];
    const now = performance.now() / 1000;

    for (const t of tracks) {
        const primary = t.id === s.primary_threat_id || t.att_state === "REFLEX";
        const linked = t.id === S.linked;
        const col = primary ? riskColor(t.att_state === "REFLEX" ? 1 : t.risk)
            : (linked ? "rgba(34,211,238,1)"
            : (t.reid ? "rgba(244,114,182,0.95)" : "rgba(74,90,110,0.9)"));

        // trails
        const trail = (S.stab ? t.pixel_history_stab : t.pixel_history) || [];
        const pts = trail.filter(p => p);
        if (pts.length > 1) {
            for (let i = 1; i < pts.length; i++) {
                const a = (i / pts.length) * (primary ? 0.75 : 0.4);
                cctx.strokeStyle = primary ? riskColor(t.risk, a) : `rgba(120,140,160,${a})`;
                cctx.lineWidth = primary ? 2 : 1;
                cctx.beginPath();
                cctx.moveTo(...toScr(pts[i - 1]));
                cctx.lineTo(...toScr(pts[i]));
                cctx.stroke();
            }
        }

        // ghost boxes at +0.5/+1/+1.5/+2s
        if (t.pixel_predicted && t.predicted_times) {
            const d0 = Math.hypot(...t.position) || 1;
            GHOST_TIMES.forEach(gt => {
                let idx = t.predicted_times.findIndex(x => x >= gt);
                if (idx < 0) idx = t.predicted_times.length - 1;
                if (idx >= t.predicted_path.length) return;
                const px = t.pixel_predicted[idx];
                if (!px || !t.bbox) return;
                const dT = Math.hypot(...t.predicted_path[idx]) || 1;
                const k = clamp(d0 / dT, 0.3, 3);
                const [gx, gy] = toScr(px);
                const gw = t.bbox[2] * k * m.sc, gh = t.bbox[3] * k * m.sc;
                cctx.strokeStyle = primary ? riskColor(t.risk, 0.55) : "rgba(120,140,160,0.35)";
                cctx.setLineDash([4, 4]);
                cctx.lineWidth = 1;
                cctx.strokeRect(gx - gw / 2, gy - gh / 2, gw, gh);
                cctx.setLineDash([]);
                cctx.fillStyle = cctx.strokeStyle;
                cctx.font = "9px monospace";
                cctx.fillText(`+${gt}s`, gx + gw / 2 + 3, gy);
            });

            // motion vector: center -> +1s position
            const idx1 = t.predicted_times.findIndex(x => x >= 1.0);
            const px1 = t.pixel_predicted[idx1 >= 0 ? Math.min(idx1, t.predicted_path.length - 1) : t.predicted_times.length - 1];
            if (px1 && t.bbox) {
                const c0 = toScr([t.bbox[0] + t.bbox[2] / 2, t.bbox[1] + t.bbox[3] / 2]);
                const c1 = toScr(px1);
                cctx.strokeStyle = col; cctx.lineWidth = primary ? 2 : 1;
                cctx.beginPath(); cctx.moveTo(...c0); cctx.lineTo(...c1); cctx.stroke();
            }
        }

        // bbox outline (thin — server already draws base boxes)
        if (t.bbox) {
            const fade = t.fading != null ? t.fading : 1;
            cctx.globalAlpha = fade;
            const [bx, by] = toScr([t.bbox[0], t.bbox[1]]);
            const bw = t.bbox[2] * m.sc, bh = t.bbox[3] * m.sc;
            cctx.strokeStyle = col; cctx.lineWidth = primary ? 4 : (linked ? 2.5 : 1.5);
            cctx.strokeRect(bx, by, bw, bh);
            if (primary || linked || t.reid || t.held_object) {
                cctx.fillStyle = col; cctx.font = "bold 14px monospace";
                const label = `${t.cls.toUpperCase()} #${t.id.split("-").pop()}${primary ? " ⚠" : ""}${t.reid ? " ·REID" : ""}${heldTag(t)}`;
                cctx.fillText(label, bx, by - 6);
            }
            if (t.closing_rate < -0.3) {
                cctx.fillStyle = riskColor(t.risk || 0.3, 0.9);
                cctx.font = "9px monospace";
                cctx.fillText(`▼ ${Math.abs(t.closing_rate).toFixed(1)} m/s`, bx, by + bh + 12);
            }
            t._hit = [bx, by, bw, bh];
            cctx.globalAlpha = 1;
        }
    }

    const p = s.camera_pose || { yaw: 0, pitch: 0, roll: 0 };
    $("camHud").textContent =
        `YAW ${deg(p.yaw).toFixed(0)}° · PITCH ${deg(p.pitch).toFixed(0)}° · ROLL ${deg(p.roll).toFixed(0)}°`;
}

/* ---------------- world view ---------------- */
const wCv = $("worldCanvas"), wctx = wCv.getContext("2d");

// scroll = zoom into the wearer, double-click = reset to auto-range
wCv.addEventListener("wheel", e => {
    e.preventDefault();
    S.worldZoom = clamp((S.worldZoom || 1) * (e.deltaY < 0 ? 1.15 : 1 / 1.15), 0.25, 10);
}, { passive: false });
wCv.addEventListener("dblclick", () => { S.worldZoom = 1; });

function drawWorld(s, tracks) {
    const r = wCv.getBoundingClientRect();
    wCv.width = r.width * devicePixelRatio; wCv.height = r.height * devicePixelRatio;
    wctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    wctx.clearRect(0, 0, r.width, r.height);

    const heading = s.rider_state.heading;
    const egoX = r.width / 2, egoY = r.height * 0.62;

    // dynamic range
    let maxD = 14;
    for (const t of tracks) {
        maxD = Math.max(maxD, Math.hypot(...t.position));
        for (const p of t.predicted_path || []) maxD = Math.max(maxD, Math.hypot(p[0], p[1]));
    }
    // user zoom on top of auto-range: zoom>1 shrinks the visible range so
    // close-in detail (the strike range) fills the panel
    const target = clamp(maxD * 1.25 / (S.worldZoom || 1), 2, 60);
    // Range used to ease at 8% per animation frame, leaving the radius far
    // behind after a subject appeared/disappeared or the user zoomed. Snap
    // big changes; ease small sensor jitter quickly without visual wobble.
    const change = Math.abs(target - S.worldScale) / Math.max(S.worldScale, 1);
    S.worldScale = change > 0.30 ? target : lerp(S.worldScale, target, 0.30);
    const scale = (Math.min(r.width, r.height) * 0.42) / S.worldScale;
    $("worldScale").textContent = `range ${S.worldScale.toFixed(0)} m` +
        ((S.worldZoom || 1) !== 1 ? ` · zoom ${S.worldZoom.toFixed(1)}×` : "");

    const w2s = p => {
        const d = Math.hypot(p[0], p[1]);
        const rel = Math.atan2(p[0], p[1]) - heading;
        return [egoX + Math.sin(rel) * d * scale, egoY - Math.cos(rel) * d * scale];
    };

    // grid + range rings
    wctx.strokeStyle = "rgba(22,34,47,0.9)"; wctx.lineWidth = 1;
    const step = 5 * scale;
    for (let x = egoX % step; x < r.width; x += step) { wctx.beginPath(); wctx.moveTo(x, 0); wctx.lineTo(x, r.height); wctx.stroke(); }
    for (let y = egoY % step; y < r.height; y += step) { wctx.beginPath(); wctx.moveTo(0, y); wctx.lineTo(r.width, y); wctx.stroke(); }
    const ringStep = S.worldScale <= 6 ? 1 : S.worldScale <= 15 ? 2 : 5;
    for (let ring = ringStep; ring <= S.worldScale; ring += ringStep) {
        wctx.strokeStyle = "rgba(34,60,80,0.7)";
        wctx.beginPath(); wctx.arc(egoX, egoY, ring * scale, 0, Math.PI * 2); wctx.stroke();
        wctx.fillStyle = "rgba(74,90,110,0.8)"; wctx.font = "9px monospace";
        wctx.fillText(`${ring}m`, egoX + ring * scale + 3, egoY - 2);
    }

    const pulse = 0.6 + 0.4 * Math.sin(performance.now() / 180);
    // Spotlight: while anyone is attended (headed for the wearer or otherwise
    // relevant), fade the background so attention lands on them.
    const spotlight = tracks.some(t => t.att_state && t.att_state !== "OBSERVE");

    // risk fields: translucent occupancy regions along predicted paths
    for (const t of tracks) {
        if (!t.predicted_path || !t.predicted_path.length) continue;
        const col = t.id === s.primary_threat_id ? riskColor(t.risk) : "rgba(120,140,160,";
        t.predicted_path.forEach((p, i) => {
            const tt = (t.predicted_times && t.predicted_times[i]) ?? i * 0.2;
            const rad = (0.6 + (t.sigma || 0.5) * 0.5 + tt * 0.35) * scale;
            const [sx, sy] = w2s(p);
            const g = wctx.createRadialGradient(sx, sy, 0, sx, sy, rad);
            const base = t.id === s.primary_threat_id ? riskColor(t.risk, 0.10 + 0.05 * pulse) : `rgba(120,140,160,${spotlight ? 0.02 : 0.05})`;
            g.addColorStop(0, base); g.addColorStop(1, "rgba(0,0,0,0)");
            wctx.fillStyle = g;
            wctx.beginPath(); wctx.arc(sx, sy, rad, 0, Math.PI * 2); wctx.fill();
        });
    }

    // history trails + predicted ribbons + center lines
    for (const t of tracks) {
        const primary = t.id === s.primary_threat_id || t.att_state === "REFLEX";
        const base = primary ? riskColor(t.att_state === "REFLEX" ? Math.max(t.risk, 0.9) : t.risk) : clsColor(t.cls);
        const alpha = primary ? 1 : spotlight
            ? (t.att_state === "ATTEND" ? 0.28 : 0.1)
            : (t.att_state === "ATTEND" ? 0.7 : 0.4);

        if (t.history && t.history.length > 1) {
            for (let i = 1; i < t.history.length; i++) {
                wctx.strokeStyle = hexA(base, (i / t.history.length) * 0.7 * alpha);
                wctx.lineWidth = primary ? 2 : 1;
                wctx.beginPath();
                wctx.moveTo(...w2s(t.history[i - 1])); wctx.lineTo(...w2s(t.history[i]));
                wctx.stroke();
            }
        }

        const path = t.predicted_path || [];
        if (path.length > 1) {
            // uncertainty ribbon
            const left = [], right = [];
            for (let i = 0; i < path.length; i++) {
                const tt = (t.predicted_times && t.predicted_times[i]) ?? i * 0.2;
                const w = (0.25 + (t.sigma || 0.4) * 0.4 + tt * 0.5);
                const a = i + 1 < path.length ? path[i + 1] : path[i];
                const b = i > 0 ? path[i - 1] : path[i];
                let dx = a[0] - b[0], dy = a[1] - b[1];
                const n = Math.hypot(dx, dy) || 1; dx /= n; dy /= n;
                left.push([path[i][0] - dy * w, path[i][1] + dx * w]);
                right.push([path[i][0] + dy * w, path[i][1] - dx * w]);
            }
            wctx.fillStyle = hexA(base, primary ? 0.16 : 0.07);
            wctx.beginPath();
            wctx.moveTo(...w2s(left[0]));
            for (const p of left) wctx.lineTo(...w2s(p));
            for (const p of right.reverse()) wctx.lineTo(...w2s(p));
            wctx.closePath(); wctx.fill();

            // dashed center line + time labels
            wctx.strokeStyle = hexA(base, 0.85 * alpha); wctx.lineWidth = primary ? 2 : 1;
            wctx.setLineDash([5, 4]);
            wctx.beginPath();
            wctx.moveTo(...w2s(path[0]));
            for (const p of path) wctx.lineTo(...w2s(p));
            wctx.stroke(); wctx.setLineDash([]);

            [0.5, 1, 2, 3].forEach(gt => {
                let idx = t.predicted_times.findIndex(x => x >= gt);
                if (idx < 0) return;
                const [sx, sy] = w2s(path[idx]);
                wctx.fillStyle = hexA(base, 0.9 * alpha); wctx.font = "9px monospace";
                wctx.fillText(`+${gt}s`, sx + 4, sy - 3);
                wctx.beginPath(); wctx.arc(sx, sy, 2, 0, Math.PI * 2); wctx.fill();
            });
        }
    }

    // rider paths: current intent vs counterfactual straight
    const rp = s.rider_state.predicted_path || [];
    if (rp.length > 1) {
        wctx.strokeStyle = "rgba(34,211,238,0.95)"; wctx.lineWidth = 2.5;
        wctx.beginPath(); wctx.moveTo(...w2s(rp[0]));
        for (const p of rp) wctx.lineTo(...w2s(p));
        wctx.stroke();
        wctx.fillStyle = "rgba(34,211,238,0.9)"; wctx.font = "9px monospace";
        const end = w2s(rp[rp.length - 1]);
        wctx.fillText("ME · CURRENT PATH", end[0] + 6, end[1]);
    }
    const sp = s.rider_state.straight_path || [];
    if (sp.length > 1 && s.rider_state.turning) {
        wctx.strokeStyle = "rgba(160,170,185,0.6)"; wctx.lineWidth = 1.5;
        wctx.setLineDash([6, 5]);
        wctx.beginPath(); wctx.moveTo(...w2s(sp[0]));
        for (const p of sp) wctx.lineTo(...w2s(p));
        wctx.stroke(); wctx.setLineDash([]);
        const end = w2s(sp[sp.length - 1]);
        wctx.fillStyle = "rgba(160,170,185,0.8)"; wctx.font = "9px monospace";
        wctx.fillText("IF STRAIGHT", end[0] + 6, end[1]);
    }

    // conflict markers
    for (const t of tracks) {
        if (t.will_collide && t.conflict_point) {
            const [sx, sy] = w2s(t.conflict_point);
            const sz = 10 * pulse;
            wctx.strokeStyle = riskColor(t.risk, 0.95); wctx.lineWidth = 3;
            wctx.beginPath();
            wctx.moveTo(sx - sz, sy - sz); wctx.lineTo(sx + sz, sy + sz);
            wctx.moveTo(sx + sz, sy - sz); wctx.lineTo(sx - sz, sy + sz);
            wctx.stroke();
            wctx.fillStyle = riskColor(t.risk, 1); wctx.font = "bold 11px monospace";
            wctx.fillText(`T-${(t.tcpa || 0).toFixed(1)}s · ${(t.dcpa || 0).toFixed(1)}m`, sx + sz + 5, sy + 4);
        }
    }

    // objects
    for (const t of tracks) {
        const primary = t.id === s.primary_threat_id || t.att_state === "REFLEX";
        const linked = t.id === S.linked;
        const fade = t.fading != null ? t.fading : 1;
        wctx.globalAlpha = fade;
        const [sx, sy] = w2s(t.position);
        const rad = primary ? 8 : 5;
        wctx.fillStyle = primary ? riskColor(t.att_state === "REFLEX" ? 1 : t.risk) : hexA(clsColor(t.cls), spotlight ? (t.att_state === "ATTEND" ? 0.35 : 0.15) : (t.att_state === "ATTEND" ? 0.85 : 0.6));
        wctx.beginPath(); wctx.arc(sx, sy, rad, 0, Math.PI * 2); wctx.fill();
        if (primary || linked) {
            wctx.strokeStyle = primary ? riskColor(t.risk, 0.5 * pulse + 0.3) : "rgba(34,211,238,0.8)";
            wctx.lineWidth = 2;
            wctx.beginPath(); wctx.arc(sx, sy, rad + 5 + 3 * pulse, 0, Math.PI * 2); wctx.stroke();
        } else if (t.reid) {
            wctx.strokeStyle = "rgba(244,114,182,0.9)";
            wctx.lineWidth = 1.5;
            wctx.beginPath(); wctx.arc(sx, sy, rad + 4, 0, Math.PI * 2); wctx.stroke();
        }
        // velocity vector
        wctx.strokeStyle = hexA(clsColor(t.cls), 0.9); wctx.lineWidth = 1.5;
        wctx.beginPath(); wctx.moveTo(sx, sy);
        const vEnd = w2s([t.position[0] + t.velocity[0], t.position[1] + t.velocity[1]]);
        wctx.lineTo(...vEnd); wctx.stroke();
        // label
        wctx.fillStyle = primary ? "#fff" : (t.reid ? "rgba(244,114,182,0.95)" : "rgba(200,214,229,0.85)");
        wctx.font = (primary ? "bold " : "") + "10px monospace";
        wctx.fillText(`${t.cls.toUpperCase()} #${t.id.split("-").pop()}${heldTag(t)}`, sx + rad + 4, sy + 3);
        t._whit = [sx, sy];
        wctx.globalAlpha = 1;
    }

    // wearer
    wctx.save();
    wctx.translate(egoX, egoY);
    wctx.fillStyle = "#22d3ee";
    wctx.beginPath(); wctx.moveTo(0, -12); wctx.lineTo(8, 8); wctx.lineTo(-8, 8); wctx.closePath(); wctx.fill();
    wctx.strokeStyle = "rgba(34,211,238,0.35)"; wctx.lineWidth = 1;
    wctx.beginPath(); wctx.arc(0, 0, 18, 0, Math.PI * 2); wctx.stroke();
    wctx.restore();
    wctx.fillStyle = "rgba(34,211,238,1)"; wctx.font = "bold 11px monospace";
    wctx.fillText("ME", egoX + 14, egoY + 12);
    wctx.fillStyle = "rgba(34,211,238,0.78)"; wctx.font = "9px monospace";
    wctx.fillText("CURRENT", egoX + 14, egoY + 25);
}

function hexA(hex, a) {
    if (hex.startsWith("rgba")) return hex.replace(/[\d.]+\)$/, `${a})`);
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/* ---------------- timeline ---------------- */
const tlCv = $("timelineCanvas"), tlctx = tlCv.getContext("2d");

function drawTimeline(s) {
    const r = tlCv.getBoundingClientRect();
    tlCv.width = r.width * devicePixelRatio; tlCv.height = r.height * devicePixelRatio;
    tlctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    tlctx.clearRect(0, 0, r.width, r.height);

    const now = s.timestamp;
    const t0 = now - 4, t1 = now + 3;
    const x = t => ((t - t0) / (t1 - t0)) * r.width;

    // future region
    tlctx.fillStyle = "rgba(34,211,238,0.04)";
    tlctx.fillRect(x(now), 0, r.width - x(now), r.height);

    // grid
    tlctx.strokeStyle = "rgba(22,34,47,0.8)"; tlctx.lineWidth = 1;
    for (let t = Math.ceil(t0); t <= t1; t++) {
        tlctx.beginPath(); tlctx.moveTo(x(t), 0); tlctx.lineTo(x(t), r.height); tlctx.stroke();
    }

    // risk history
    const hist = s.risk_history || [];
    if (hist.length > 1) {
        tlctx.strokeStyle = "rgba(251,191,36,0.9)"; tlctx.lineWidth = 1.5;
        tlctx.beginPath();
        hist.forEach(([t, v], i) => {
            const px = x(t), py = r.height - 12 - v * (r.height - 30);
            i ? tlctx.lineTo(px, py) : tlctx.moveTo(px, py);
        });
        tlctx.stroke();
    }

    // events
    for (const e of s.events || []) {
        if (e.ts < t0 || e.ts > t1) continue;
        const ex = x(e.ts);
        const warn = /CONFLICT|CPA|HAPTIC/.test(e.type);
        tlctx.strokeStyle = warn ? "rgba(244,63,94,0.9)" : "rgba(34,211,238,0.6)";
        tlctx.lineWidth = warn ? 2 : 1;
        tlctx.beginPath(); tlctx.moveTo(ex, r.height - 26); tlctx.lineTo(ex, r.height - 8); tlctx.stroke();
        tlctx.fillStyle = warn ? "rgba(244,63,94,0.9)" : "rgba(120,140,160,0.9)";
        tlctx.font = "8px monospace";
        tlctx.save(); tlctx.translate(ex + 2, r.height - 28); tlctx.rotate(-0.5);
        tlctx.fillText(e.type.slice(0, 14), 0, 0); tlctx.restore();
    }

    // NOW line
    tlctx.strokeStyle = "rgba(34,211,238,0.9)"; tlctx.lineWidth = 1.5;
    tlctx.beginPath(); tlctx.moveTo(x(now), 0); tlctx.lineTo(x(now), r.height); tlctx.stroke();
    tlctx.fillStyle = "rgba(34,211,238,0.9)"; tlctx.font = "9px monospace";
    tlctx.fillText("NOW", x(now) + 4, 11);
    tlctx.fillStyle = "rgba(74,90,110,0.9)";
    tlctx.fillText("-4s", 4, 11);
    tlctx.fillText("+3s predicted", r.width - 84, 11);
}

/* ---------------- DOM updates ---------------- */
function updateDom(s, tracks) {
    const sys = s.system;
    $("stFps").textContent = sys.fps.toFixed(1);
    $("stTrk").textContent = sys.tracking;
    setPill("stCam", sys.camera, "OK", "OFF");
    setPill("stImu", sys.imu, "LOCK", "NO DATA");
    setPill("stHap", sys.haptic_endpoint, "ESP32", "LOCAL");
    const wm = $("stWm");
    wm.querySelector("b").textContent = `WORLD MODEL: ${sys.world_model}`;
    wm.className = `pill wm ${sys.world_model === "STABLE" ? "stable" : "degraded"}`;

    // funnel: perception -> attention -> watch -> threat
    const f = s.funnel || {};
    const funnelKeys = f.attended !== undefined
        ? ["perceived", "attended", "watch", "threat"]
        : ["perceived", "tracked", "moving", "closing", "conflict"];
    $("funnel").innerHTML = funnelKeys
        .map((k, i, arr) =>
            `<div class="f-step ${k === "threat" || k === "conflict" ? (f[k] > 0 ? "hot" : "") : ""}"><b>${f[k] ?? 0}</b> ${k.toUpperCase()}</div>` +
            (i < arr.length - 1 ? `<div class="f-arrow">→</div>` : "")).join("");

    // scene + reflex mode
    const scene = s.scene || {};
    const sceneLabel = scene.environment
        ? `${scene.environment.toUpperCase()} · ${(scene.density || "--").toUpperCase()}`
        : (scene.density || "--").toUpperCase();
    $("stScene").querySelector("b").textContent =
        sceneLabel + (scene.people != null ? ` · ${scene.people}p` : "")
        + (scene.ego_motion && scene.ego_motion !== "stationary"
            ? ` · ${scene.ego_motion.toUpperCase()}` : "");
    const reflexOn = Boolean(scene.reflex);
    document.body.classList.toggle("reflex", reflexOn);
    reflexAlert(reflexOn);
    // reflex onset: hard red flash + freeze the evidence bars ~2.5s so the
    // judge can read *why* it fired — attention accumulates, reflex reacts
    if (reflexOn && !S._reflexOn) {
        const fl = $("reflexFlash");
        if (fl) { fl.classList.remove("on"); void fl.offsetWidth; fl.classList.add("on"); }
        if (s.threat) S.evHold = { until: performance.now() + 2500, ev: s.threat.evidence };
    }
    S._reflexOn = reflexOn;

    // threat rail
    const th = s.threat;
    if (th) {
        $("thId").textContent = `${(th.cls || "").toUpperCase()} ${th.id}${heldTag(th)}`;
        $("thDir").textContent =
            `${th.att_state || "—"} · ${th.direction_label || "—"}`;
        const pct = Math.round((th.risk || 0) * 100);
        $("thRisk").textContent = pct;
        const rb = document.querySelector(".risk-big");
        rb.className = "risk-big" + (pct > 70 ? " high" : pct > 40 ? " med" : "");
        $("thRiskBar").style.width = pct + "%";
        $("thRiskBar").style.background = riskColor(th.risk);
        // sparkline history: attention (cyan) + risk (red), last ~5s
        S.spark = S.spark || [];
        S.spark.push({ t: performance.now(), a: th.attention || 0, r: th.risk || 0 });
        while (S.spark.length > 120) S.spark.shift();
        drawSpark();
        $("thTtc").textContent = th.ttc != null ? th.ttc.toFixed(1) + " s" : "—";
        $("thTcpa").textContent = th.tcpa != null && isFinite(th.tcpa) ? th.tcpa.toFixed(1) + " s" : "—";
        $("thDcpa").textContent = th.dcpa != null ? th.dcpa.toFixed(1) + " m" : "—";
        $("thClose").textContent = th.closing_rate != null ? th.closing_rate.toFixed(1) + " m/s" : "—";
        $("thConf").textContent = th.confidence != null ? (th.confidence * 100).toFixed(0) + "%" : "—";
        const whyHtml = (th.why || []).map(w =>
            `<li class="${w.active ? "on" : ""}">${w.label}</li>`);
        for (const reason of th.reasons || []) {
            if (!whyHtml.some(h => h.includes(reason)))
                whyHtml.push(`<li class="on">${reason}</li>`);
        }
        $("whyList").innerHTML = whyHtml.join("");

        // evidence breakdown — attention as auditable channel scores, lerped
        // so bars visibly rise/fall; frozen briefly after a reflex
        const held = S.evHold && performance.now() < S.evHold.until ? S.evHold : null;
        const ev = held ? held.ev : (th.evidence || {});
        S.evDisp = S.evDisp || {};
        for (const k of Object.keys(S.evDisp)) if (!(k in ev)) delete S.evDisp[k];
        const evRows = Object.entries(ev)
            .sort((a, b) => b[1] - a[1]).slice(0, 6)
            .map(([k, v]) => {
                S.evDisp[k] = (S.evDisp[k] || 0) + (v - (S.evDisp[k] || 0)) * 0.25;
                const dv = Math.max(0, Math.min(1, S.evDisp[k]));
                return `<div class="ev-row"><span>${k.replace(/_/g, " ")}</span>` +
                    `<div class="ev-bar"><div style="width:${Math.round(dv * 100)}%"></div></div>` +
                    `<b>${dv.toFixed(2)}</b></div>`;
            });
        $("evBars").innerHTML = evRows.join("")
            + (held ? `<div class="ev-hold">REFLEX SNAPSHOT</div>` : "");

        const sec = th.secondary;
        $("thSec").textContent = sec
            ? `${sec.att_state} · ${sec.cls} ${sec.id}`
            : "—";
    } else {
        $("thId").textContent = "—"; $("thDir").textContent = "NO ACTIVE THREAT";
        $("thRisk").textContent = "0"; $("thRiskBar").style.width = "0%";
        $("whyList").innerHTML = `<li class="dim">no active threat</li>`;
        $("evBars").innerHTML = "";
        $("thSec").textContent = "—";
        if (S.spark) { S.spark.length = 0; drawSpark(); }
    }

    // episodic memory: persistent-follower alerts surface here regardless of
    // whether the follower is currently the primary threat
    const followers = (s.memory && s.memory.followers) || [];
    const fol = followers[0];
    $("thMemRow").style.display = fol ? "" : "none";
    $("thMemNote").style.display = fol ? "" : "none";
    if (fol) {
        $("thMem").textContent = `FOLLOWER? ${fol.id} · ${(fol.comove_s / 60).toFixed(1)} min`;
        $("thMemNote").textContent = fol.narration
            || `${fol.reappears} reappearance(s) · closest ${fol.min_dist}m`;
    }

    // This card deliberately sits on top of the camera, independent of whether
    // a world point can be reprojected into pixel coordinates.
    const alert = $("cameraAlert");
    const alertTrack = tracks.find(t => t.id === s.primary_threat_id);
    // Keep the large camera annotation present for the selected track. It
    // changes vocabulary/color with risk instead of flashing away between
    // low-risk frames and the next risk assessment.
    const showAlert = Boolean((s.system && (s.system.camera || s.system.frame_size)) || (th && alertTrack));
    alert.hidden = !showAlert;
    if (showAlert) {
        if (!th || !alertTrack) {
            alert.classList.remove("reflex", "high", "track");
            alert.classList.add("track");
            $("cameraAlertTitle").textContent = "SCANNING · LIVE";
            alert.querySelector(".camera-alert-kicker").textContent = "NO TRACKED SUBJECTS";
            $("cameraAlertMeta").textContent = "CAMERA ACTIVE · WAITING FOR DETECTIONS";
        } else {
        const reflex = th.att_state === "REFLEX" || (scene.reflex && alertTrack.att_state === "REFLEX");
        const high = th.risk >= 0.70;
        const conflict = th.risk >= 0.30 || reflex;
        alert.classList.toggle("reflex", reflex);
        alert.classList.toggle("high", high && !reflex);
        alert.classList.toggle("track", !conflict);
        $("cameraAlertTitle").textContent = reflex
            ? `IMMEDIATE THREAT · ${(th.direction_label || "").toUpperCase()}`
            : `${(th.direction_label || "TRACKING").toUpperCase()} · ${Math.round(th.risk * 100)}%`;
        const cpa = reflex
            ? "RAPID APPROACH DETECTED"
            : (th.tcpa != null && isFinite(th.tcpa)
                ? `${th.tcpa.toFixed(1)}s TO CPA`
                : (conflict ? "TRACKING CONFLICT" : "PREDICTED PATH ACTIVE"));
        const label = (alertTrack && alertTrack.cls ? alertTrack.cls.toUpperCase() : th.cls || "OBJECT").toUpperCase();
        alert.querySelector(".camera-alert-kicker").textContent =
            reflex ? "HALO REFLEX" : (conflict ? "PREDICTED CONFLICT" : "TRACKING");
        $("cameraAlertMeta").textContent = `${label} · ${cpa}`;
        }
    }

    // haptics (decay after 1.5s)
    const hp = s.haptic || {};
    const stale = (s.timestamp - (hp.ts || 0)) > 1.5 ? 0 : 1;
    $("hapL").style.width = (hp.left || 0) * 100 * stale + "%";
    $("hapR").style.width = (hp.right || 0) * 100 * stale + "%";

    // counterfactual
    const cf = s.counterfactual || {};
    $("cfCur").textContent = cf.current_cpa != null ? cf.current_cpa.toFixed(1) + " m" : "—";
    $("cfStr").textContent = cf.straight_cpa != null ? cf.straight_cpa.toFixed(1) + " m" : "—";
    const dEl = $("cfDelta");
    if (cf.delta != null) {
        dEl.textContent = (cf.delta >= 0 ? "+" : "") + (cf.delta * 100).toFixed(0) + "%";
        dEl.style.color = cf.delta > 0.1 ? "var(--red)" : "var(--txt)";
    } else dEl.textContent = "—";

    // ego-motion diagnostic (primary track)
    const pt = tracks.find(t => t.id === s.primary_threat_id) || tracks[0];
    $("dgRaw").style.width = clamp(((pt && pt.raw_px_rate) || 0) / 300 * 100, 0, 100) + "%";
    $("dgCam").style.width = clamp(Math.abs(deg(s.rider_state.yaw_rate)) / 120 * 100, 0, 100) + "%";
    $("dgStab").style.width = clamp(Math.abs(deg((pt && pt.bearing_rate) || 0)) / 60 * 100, 0, 100) + "%";

    // events feed
    for (const e of s.events || []) {
        if (e.id <= S.lastEventId) continue;
        S.lastEventId = e.id;
        const div = document.createElement("div");
        div.className = "ev" + (/CONFLICT|CPA|HAPTIC/.test(e.type) ? " warn" : "");
        div.innerHTML = `<span class="t">T+${e.ts.toFixed(1)}</span><span class="k">${e.type}</span><span class="d">${e.label || ""}</span>`;
        const feed = $("eventFeed");
        feed.insertBefore(div, feed.firstChild);
        while (feed.children.length > 30) feed.removeChild(feed.lastChild);
    }
}

function setPill(id, ok, okText, badText) {
    const el = $(id);
    el.querySelector("b").textContent = ok ? okText : badText;
    el.className = "pill " + (ok ? "ok" : "bad");
}

/* ---------------- threat sparkline ---------------- */
// tiny 5s attention/risk history — a single number becomes a trend
function drawSpark() {
    const cv = $("thSpark"); if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 160, h = cv.clientHeight || 30;
    if (cv.width !== Math.round(w * dpr)) { cv.width = w * dpr; cv.height = h * dpr; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pts = S.spark || [];
    if (pts.length < 2) return;
    const t0 = pts[0].t, span = Math.max(performance.now() - t0, 1);
    const draw = (key, color) => {
        ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        ctx.beginPath();
        pts.forEach((p, i) => {
            const x = (p.t - t0) / span * (w - 2) + 1;
            const y = h - 3 - p[key] * (h - 6);
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        });
        ctx.stroke();
    };
    draw("a", "rgba(34,211,238,0.85)");   // attention
    draw("r", "rgba(244,63,94,0.9)");     // risk
    ctx.fillStyle = "rgba(148,163,184,0.7)"; ctx.font = "7px monospace";
    ctx.fillText("ATT", 2, 8); ctx.fillStyle = "rgba(244,63,94,0.8)";
    ctx.fillText("RISK", 2, h - 2);
}

/* ---------------- reflex audio alert ---------------- */
let _audioCtx = null, _wasReflex = false;
document.addEventListener("click", () => { _audioCtx && _audioCtx.resume(); });

function reflexAlert(on) {
    if (on && !_wasReflex) {
        try {
            _audioCtx = _audioCtx || new (window.AudioContext || window.webkitAudioContext)();
            if (_audioCtx.state === "suspended") _audioCtx.resume();
            const osc = _audioCtx.createOscillator(), g = _audioCtx.createGain();
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(660, _audioCtx.currentTime);
            osc.frequency.setValueAtTime(880, _audioCtx.currentTime + 0.12);
            g.gain.setValueAtTime(0.18, _audioCtx.currentTime);
            g.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + 0.55);
            osc.connect(g); g.connect(_audioCtx.destination);
            osc.start(); osc.stop(_audioCtx.currentTime + 0.55);
        } catch (e) { /* audio unavailable — visual alert still fires */ }
    }
    _wasReflex = on;
}

/* ---------------- interactions ---------------- */
camCv.addEventListener("mousemove", e => {
    const s = activeState(); if (!s) return;
    const r = camCv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    S.linked = null;
    for (const t of s.tracks || []) {
        if (t._hit && mx >= t._hit[0] && mx <= t._hit[0] + t._hit[2] &&
            my >= t._hit[1] && my <= t._hit[1] + t._hit[3]) S.linked = t.id;
    }
});
wCv.addEventListener("mousemove", e => {
    const s = activeState(); if (!s) return;
    const r = wCv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    S.linked = null;
    for (const t of s.tracks || []) {
        if (t._whit && Math.hypot(mx - t._whit[0], my - t._whit[1]) < 22) S.linked = t.id;
    }
});

// panel expand: ⤢ button toggles, Esc closes
document.querySelectorAll(".expand-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const p = $(btn.dataset.panel);
        const was = p.classList.contains("expanded");
        document.querySelectorAll("section.expanded").forEach(x => x.classList.remove("expanded"));
        if (!was) p.classList.add("expanded");
        document.body.classList.toggle("has-expanded",
            Boolean(document.querySelector("section.expanded")));
    });
});
document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
        document.querySelectorAll("section.expanded").forEach(x => x.classList.remove("expanded"));
        document.body.classList.remove("has-expanded");
    }
});

// -------- spoken alerts (browser TTS — zero hardware needed for the demo) ----
S.audio = false;
S.spoken = { follower: {}, reflexAt: 0 };
function speak(text) {
    if (!S.audio || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05; u.volume = 0.9;
    speechSynthesis.speak(u);
}
$("audioToggle").addEventListener("click", () => {
    S.audio = !S.audio;
    $("audioToggle").textContent = S.audio ? "🔊 AUDIO" : "🔇 AUDIO";
    if (S.audio) { speechSynthesis.cancel(); speak("HALO audio armed"); }
});

function audioAlerts(s) {
    const followers = (s.memory && s.memory.followers) || [];
    for (const fol of followers) {
        const seen = S.spoken.follower[fol.id];
        if (!seen) {
            S.spoken.follower[fol.id] = { narrated: false };
            speak(fol.narration || `Possible follower: ${fol.id.replace("-", " ")}`);
        } else if (!seen.narrated && fol.narration) {
            seen.narrated = true;
            speak(fol.narration);
        }
    }
    const now = performance.now();
    if ((s.tracks || []).some(t => t.att_state === "REFLEX") && now - S.spoken.reflexAt > 3000) {
        S.spoken.reflexAt = now;
        const th = s.threat;
        speak(`Reflex${th && th.direction_label ? " — " + th.direction_label.toLowerCase() : ""}`);
    }
}

$("stabToggle").addEventListener("click", () => {
    S.stab = !S.stab;
    $("stabToggle").textContent = S.stab ? "STAB" : "RAW";
    $("stabTag").classList.toggle("off", !S.stab);
});

$("replayBtn").addEventListener("click", () => {
    if (S.replay) { S.replay = null; $("replayBtn").classList.remove("active"); return; }
    if (!S.snapshots.length) return;
    S.replay = { idx: 0 };
    $("replayBtn").classList.add("active");
});

$("clearSubjects").addEventListener("click", async () => {
    const button = $("clearSubjects");
    button.disabled = true; button.classList.add("clearing"); button.textContent = "CLEARING…";
    try {
        const response = await fetch("/api/clear-subjects", { method: "POST" });
        if (!response.ok) throw new Error("clear request failed");
        S.prev = null; S.curr = null; S.snapshots = []; S.linked = null; S.lastEventId = 0;
        S.seen = {}; S.evHold = null; S.spark = []; S.evDisp = {};
    } catch (error) {
        console.error("Could not clear subjects", error);
    } finally {
        setTimeout(() => {
            button.disabled = false; button.classList.remove("clearing"); button.textContent = "CLEAR SUBJECTS";
        }, 450);
    }
});

/* A canvas does not repaint itself when CSS Grid finishes a late layout pass.
   Safari commonly does that after the first live state arrives, which used to
   leave a one-pixel-looking camera/world panel until a manual reload. */
let layoutRenderQueued = false;
function redrawAfterLayout() {
    if (layoutRenderQueued) return;
    layoutRenderQueued = true;
    requestAnimationFrame(() => {
        layoutRenderQueued = false;
        const s = activeState();
        if (!s || !s.ready) return;
        try {
            const tracks = interpTracks(s);
            drawCamera(s, tracks);
            drawWorld(s, tracks);
            drawTimeline(s);
            updateDom(s, tracks);
        } catch (e) { console.error("layout redraw error:", e); }
    });
}

if (window.ResizeObserver) {
    const layoutObserver = new ResizeObserver(redrawAfterLayout);
    [$("camWrap"), $("worldCanvas"), $("timelineCanvas")].forEach(node => layoutObserver.observe(node));
}
window.addEventListener("resize", redrawAfterLayout);
window.addEventListener("pageshow", () => {
    redrawAfterLayout();
    poll();
});

/* ---------------- main loop ---------------- */
function frame() {
    const s = activeState();
    const ready = s && s.ready;
    $("emptyState").hidden = ready || DEMO;
    // The JPEG stream is independent of the world model. Paint it immediately
    // rather than making the live camera wait for its first tracking snapshot.
    if (!ready && (camBmp || DEMO)) {
        drawCamera(s || { system: {}, camera_pose: { yaw: 0, pitch: 0, roll: 0 } }, []);
    }
    if (ready) {
        try {
            const tracks = interpTracks(s);
            drawCamera(s, tracks);
            drawWorld(s, tracks);
            drawTimeline(s);
            updateDom(s, tracks);
            audioAlerts(s);
            if (S.replay) {
                S.replay.idx += 1;
                if (S.replay.idx >= S.snapshots.length) {
                    S.replay = null; $("replayBtn").classList.remove("active");
                }
            }
        } catch (e) { console.error("frame render error:", e); }
    }
    requestAnimationFrame(frame);
}

if (DEMO) $("demoBadge").hidden = false;
setInterval(poll, POLL_MS);
poll();
frame();
