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
};
const clsColor = c => CLS_COLORS[c] || "#22d3ee";

/* ================= MOCK ADAPTER (clearly isolated demo data) ================ */
/* Used only with ?demo — generates a synthetic but schema-complete state so
   the full console is visible without the phone backend. */
const Mock = {
    t0: performance.now() / 1000,
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
            direction: "left", direction_label: "REAR LEFT",
            factors: {}, expanding: carD < 15, raw_px_rate: 40,
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
        }];

        return {
            ready: true, timestamp: t,
            system: { fps: 10, camera: true, imu: true, tracking: 2,
                      haptic_endpoint: false, world_model: "STABLE", frame_size: [1280, 720] },
            camera_pose: { yaw, pitch: 0.1, roll: 0, confidence: 0.9 },
            rider_state: { position: [0, 0], velocity: [Math.sin(yaw) * 1.5, Math.cos(yaw) * 1.5],
                           speed: 1.5, heading: yaw, yaw_rate: 0.3 * Math.cos(t * 0.5),
                           turning: Math.abs(0.3 * Math.cos(t * 0.5)) > 0.12,
                           turn_direction: yaw > 0 ? "left" : "right",
                           predicted_path: egoPath, straight_path: straightPath },
            tracks,
            primary_threat_id: "car-7",
            threat: { id: "car-7", cls: "car", direction_label: "REAR LEFT", risk,
                      ttc: carD / 6, tcpa: carD / 6.5, dcpa: colliding ? 1.2 : 6,
                      closing_rate: -6 * (1 - carD / 30), confidence: 0.82,
                      why: [
                          { label: "closing rapidly", active: true },
                          { label: "bearing stable", active: true },
                          { label: "rider turning", active: Math.abs(0.3 * Math.cos(t * 0.5)) > 0.12 },
                          { label: "predicted paths overlap", active: colliding },
                          { label: "expanding in frame", active: carD < 15 }]},
            haptic: { left: risk > 0.5 ? risk : 0, right: 0, center: 0,
                      direction: "left", intensity: "strong", ts: t },
            counterfactual: { current_cpa: colliding ? 1.2 : 6, straight_cpa: 5.5,
                              current_risk: risk, straight_risk: 0.2, delta: risk - 0.2 },
            funnel: { perceived: 2, tracked: 2, moving: 2, closing: 1, conflict: colliding ? 1 : 0 },
            events: [
                { id: 1, ts: t - 6, type: "TRACK ACQUIRED", label: "car car-7" },
                { id: 2, ts: t - 4, type: "CLOSING DETECTED", label: "-5.4 m/s" },
                { id: 3, ts: t - 2, type: "RIDER TURN STARTED", label: "left" },
                ...(colliding ? [{ id: 4, ts: t - 1, type: "PATH CONFLICT", label: "tCPA 1.2s", risk }] : []),
            ],
            risk_history: Array.from({ length: 80 }, (_, i) => [t - 8 + i * 0.1, clamp(1 - (lerp(30, carD, i / 79)) / 12, 0, 1)]),
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
    if (!S.prev || DEMO || S.replay) return s.tracks;
    const a = clamp((performance.now() - S.currAt) / POLL_MS, 0, 1);
    const prevById = Object.fromEntries((S.prev.tracks || []).map(t => [t.id, t]));
    return s.tracks.map(t => {
        const p = prevById[t.id];
        if (!p) return t;
        return { ...t,
            position: [lerp(p.position[0], t.position[0], a), lerp(p.position[1], t.position[1], a)],
            risk: lerp(p.risk || 0, t.risk || 0, a) };
    });
}

/* ---------------- camera feed + overlay ---------------- */
const camCv = $("camOverlay"), camWrap = $("camWrap");
const cctx = camCv.getContext("2d");
let camBmp = null, camFetchBusy = false;

async function pollFrame() {
    if (camFetchBusy) return;
    camFetchBusy = true;
    try {
        const r = await fetch("/frame.jpg", { cache: "no-store" });
        if (r.status === 200) {
            const bmp = await createImageBitmap(await r.blob());
            if (camBmp) camBmp.close();
            camBmp = bmp;
        }
    } catch (e) { /* transient — keep last bitmap */ }
    camFetchBusy = false;
}
setInterval(pollFrame, 90);

function camRect(fw, fh) {
    const r = camWrap.getBoundingClientRect();
    const sc = Math.min(r.width / fw, r.height / fh);
    const w = fw * sc, h = fh * sc;
    return { x: (r.width - w) / 2, y: (r.height - h) / 2, w, h, sc };
}

function drawCamera(s, tracks) {
    const r = camWrap.getBoundingClientRect();
    // Keep canvas backing pixels aligned with its CSS box on Retina displays.
    const dpr = window.devicePixelRatio || 1;
    camCv.width = Math.round(r.width * dpr); camCv.height = Math.round(r.height * dpr);
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cctx.fillStyle = "#000"; cctx.fillRect(0, 0, r.width, r.height);
    $("camNoFeed").classList.toggle("show", !camBmp && !DEMO);

    const fs = s.system.frame_size;
    const fw = camBmp ? camBmp.width : (fs ? fs[0] : 0);
    const fh = camBmp ? camBmp.height : (fs ? fs[1] : 0);
    if (!fw || !fh) return;

    const m = camRect(fw, fh);
    if (camBmp) cctx.drawImage(camBmp, m.x, m.y, m.w, m.h);
    const toScr = p => [m.x + p[0] * m.sc, m.y + p[1] * m.sc];
    const now = performance.now() / 1000;

    for (const t of tracks) {
        const primary = t.id === s.primary_threat_id;
        const linked = t.id === S.linked;
        const col = primary ? riskColor(t.risk) : (linked ? "rgba(34,211,238,1)" : "rgba(74,90,110,0.9)");

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
            const [bx, by] = toScr([t.bbox[0], t.bbox[1]]);
            const bw = t.bbox[2] * m.sc, bh = t.bbox[3] * m.sc;
            cctx.strokeStyle = col; cctx.lineWidth = primary ? 4 : (linked ? 2.5 : 1.5);
            cctx.strokeRect(bx, by, bw, bh);
            if (primary || linked) {
                cctx.fillStyle = col; cctx.font = "bold 14px monospace";
                const label = `${t.cls.toUpperCase()} #${t.id.split("-").pop()}${primary ? " ⚠" : ""}`;
                cctx.fillText(label, bx, by - 6);
            }
            if (t.closing_rate < -0.3) {
                cctx.fillStyle = riskColor(t.risk || 0.3, 0.9);
                cctx.font = "9px monospace";
                cctx.fillText(`▼ ${Math.abs(t.closing_rate).toFixed(1)} m/s`, bx, by + bh + 12);
            }
            t._hit = [bx, by, bw, bh];
        }
    }

    const p = s.camera_pose;
    $("camHud").textContent =
        `YAW ${deg(p.yaw).toFixed(0)}° · PITCH ${deg(p.pitch).toFixed(0)}° · ROLL ${deg(p.roll).toFixed(0)}°`;
}

/* ---------------- world view ---------------- */
const wCv = $("worldCanvas"), wctx = wCv.getContext("2d");

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
    const target = clamp(maxD * 1.25, 12, 60);
    S.worldScale = lerp(S.worldScale, target, 0.08);
    const scale = (Math.min(r.width, r.height) * 0.42) / S.worldScale;
    $("worldScale").textContent = `range ${S.worldScale.toFixed(0)} m`;

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
    for (let ring = 5; ring <= S.worldScale; ring += 5) {
        wctx.strokeStyle = "rgba(34,60,80,0.7)";
        wctx.beginPath(); wctx.arc(egoX, egoY, ring * scale, 0, Math.PI * 2); wctx.stroke();
        wctx.fillStyle = "rgba(74,90,110,0.8)"; wctx.font = "9px monospace";
        wctx.fillText(`${ring}m`, egoX + ring * scale + 3, egoY - 2);
    }

    const pulse = 0.6 + 0.4 * Math.sin(performance.now() / 180);

    // risk fields: translucent occupancy regions along predicted paths
    for (const t of tracks) {
        if (!t.predicted_path || !t.predicted_path.length) continue;
        const col = t.id === s.primary_threat_id ? riskColor(t.risk) : "rgba(120,140,160,";
        t.predicted_path.forEach((p, i) => {
            const tt = (t.predicted_times && t.predicted_times[i]) ?? i * 0.2;
            const rad = (0.6 + (t.sigma || 0.5) * 0.5 + tt * 0.35) * scale;
            const [sx, sy] = w2s(p);
            const g = wctx.createRadialGradient(sx, sy, 0, sx, sy, rad);
            const base = t.id === s.primary_threat_id ? riskColor(t.risk, 0.10 + 0.05 * pulse) : "rgba(120,140,160,0.05)";
            g.addColorStop(0, base); g.addColorStop(1, "rgba(0,0,0,0)");
            wctx.fillStyle = g;
            wctx.beginPath(); wctx.arc(sx, sy, rad, 0, Math.PI * 2); wctx.fill();
        });
    }

    // history trails + predicted ribbons + center lines
    for (const t of tracks) {
        const primary = t.id === s.primary_threat_id;
        const base = primary ? riskColor(t.risk) : clsColor(t.cls);
        const alpha = primary ? 1 : 0.55;

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
        const primary = t.id === s.primary_threat_id;
        const linked = t.id === S.linked;
        const [sx, sy] = w2s(t.position);
        const rad = primary ? 8 : 5;
        wctx.fillStyle = primary ? riskColor(t.risk) : hexA(clsColor(t.cls), 0.85);
        wctx.beginPath(); wctx.arc(sx, sy, rad, 0, Math.PI * 2); wctx.fill();
        if (primary || linked) {
            wctx.strokeStyle = primary ? riskColor(t.risk, 0.5 * pulse + 0.3) : "rgba(34,211,238,0.8)";
            wctx.lineWidth = 2;
            wctx.beginPath(); wctx.arc(sx, sy, rad + 5 + 3 * pulse, 0, Math.PI * 2); wctx.stroke();
        }
        // velocity vector
        wctx.strokeStyle = hexA(clsColor(t.cls), 0.9); wctx.lineWidth = 1.5;
        wctx.beginPath(); wctx.moveTo(sx, sy);
        const vEnd = w2s([t.position[0] + t.velocity[0], t.position[1] + t.velocity[1]]);
        wctx.lineTo(...vEnd); wctx.stroke();
        // label
        wctx.fillStyle = primary ? "#fff" : "rgba(200,214,229,0.85)";
        wctx.font = (primary ? "bold " : "") + "10px monospace";
        wctx.fillText(`${t.cls.toUpperCase()} #${t.id.split("-").pop()}`, sx + rad + 4, sy + 3);
        t._whit = [sx, sy];
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

    // funnel
    const f = s.funnel || {};
    $("funnel").innerHTML = ["perceived", "tracked", "moving", "closing", "conflict"]
        .map((k, i, arr) =>
            `<div class="f-step ${k === "conflict" && f[k] > 0 ? "hot" : ""}"><b>${f[k] ?? 0}</b> ${k.toUpperCase()}</div>` +
            (i < arr.length - 1 ? `<div class="f-arrow">→</div>` : "")).join("");

    // threat rail
    const th = s.threat;
    if (th) {
        $("thId").textContent = `${(th.cls || "").toUpperCase()} ${th.id}`;
        $("thDir").textContent = th.direction_label || "—";
        const pct = Math.round((th.risk || 0) * 100);
        $("thRisk").textContent = pct;
        const rb = document.querySelector(".risk-big");
        rb.className = "risk-big" + (pct > 70 ? " high" : pct > 40 ? " med" : "");
        $("thRiskBar").style.width = pct + "%";
        $("thRiskBar").style.background = riskColor(th.risk);
        $("thTtc").textContent = th.ttc != null ? th.ttc.toFixed(1) + " s" : "—";
        $("thTcpa").textContent = th.tcpa != null && isFinite(th.tcpa) ? th.tcpa.toFixed(1) + " s" : "—";
        $("thDcpa").textContent = th.dcpa != null ? th.dcpa.toFixed(1) + " m" : "—";
        $("thClose").textContent = th.closing_rate != null ? th.closing_rate.toFixed(1) + " m/s" : "—";
        $("thConf").textContent = th.confidence != null ? (th.confidence * 100).toFixed(0) + "%" : "—";
        $("whyList").innerHTML = (th.why || []).map(w =>
            `<li class="${w.active ? "on" : ""}">${w.label}</li>`).join("");
    } else {
        $("thId").textContent = "—"; $("thDir").textContent = "NO ACTIVE THREAT";
        $("thRisk").textContent = "0"; $("thRiskBar").style.width = "0%";
        $("whyList").innerHTML = `<li class="dim">no active threat</li>`;
    }

    // This card deliberately sits on top of the camera, independent of whether
    // a world point can be reprojected into pixel coordinates.
    const alert = $("cameraAlert");
    const alertTrack = tracks.find(t => t.id === s.primary_threat_id);
    // Keep the large camera annotation present for the selected track. It
    // changes vocabulary/color with risk instead of flashing away between
    // low-risk frames and the next risk assessment.
    const showAlert = Boolean(th && alertTrack);
    alert.hidden = !showAlert;
    if (showAlert) {
        const high = th.risk >= 0.70;
        const conflict = th.risk >= 0.30;
        alert.classList.toggle("high", high);
        alert.classList.toggle("track", !conflict);
        $("cameraAlertTitle").textContent = `${(th.direction_label || "TRACKING").toUpperCase()} · ${Math.round(th.risk * 100)}%`;
        const cpa = th.tcpa != null && isFinite(th.tcpa)
            ? `${th.tcpa.toFixed(1)}s TO CPA`
            : (conflict ? "TRACKING CONFLICT" : "PREDICTED PATH ACTIVE");
        const label = (alertTrack && alertTrack.cls ? alertTrack.cls.toUpperCase() : th.cls || "OBJECT").toUpperCase();
        alert.querySelector(".camera-alert-kicker").textContent = conflict ? "PREDICTED CONFLICT" : "TRACKING";
        $("cameraAlertMeta").textContent = `${label} · ${cpa}`;
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
    } catch (error) {
        console.error("Could not clear subjects", error);
    } finally {
        setTimeout(() => {
            button.disabled = false; button.classList.remove("clearing"); button.textContent = "CLEAR SUBJECTS";
        }, 450);
    }
});

/* ---------------- main loop ---------------- */
function frame() {
    const s = activeState();
    const ready = s && s.ready;
    $("emptyState").hidden = ready || DEMO;
    if (ready) {
        try {
            const tracks = interpTracks(s);
            drawCamera(s, tracks);
            drawWorld(s, tracks);
            drawTimeline(s);
            updateDom(s, tracks);
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
