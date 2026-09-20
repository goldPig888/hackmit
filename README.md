# HALO

**HALO is a live predictive safety and attention prototype.** It turns an iPhone camera and motion stream into stabilized tracks, approximate rider-relative world state, future paths, collision reasoning, attention evidence, directional haptics, and a live visual console.

```text
iPhone camera + IMU
        ↓
pose correction → YOLO11n + tracking → world-frame EKF
        ↓
object and rider path prediction → closest point of approach
        ↓
context-aware attention + reflex → video / console / haptics
```

> HALO is a hackathon research prototype, not a certified safety system, violence detector, or security product. Its distances, risk values, thresholds, and motion labels are estimates that require calibration and evaluation before real-world safety use.

## Features

| Feature | What it means | Live status |
|---|---|---|
| Predictive personal-safety layer | Forecasts interactions instead of only labeling nearby objects. | Active |
| iPhone sensing and transport | Safari streams JPEG frames plus orientation, gyro, and acceleration over `/ingest`. | Active |
| YOLO11n + persistent subjects | Detects road users and people; iPhone mode prefers BoT-SORT + selective ReID, with ByteTrack fallback. | Active |
| IMU ego-motion compensation | Stabilizes camera rays so turning the phone is not mistaken for subject motion. | Active |
| Stabilized 3D world model | Tracks approximate `[X, Y, Z, Vx, Vy, Vz]` object state with an EKF. | Active |
| Future path prediction | Predicts straight constant-velocity object paths and a curved rider arc for four seconds. | Active |
| CPA conflict reasoning | Separates a close safe pass from predicted path overlap using time and distance at closest approach. | Active |
| Multiple independent cues | Combines depth, closing rate, bearing, expansion/looming, trajectory overlap, and uncertainty instead of relying on one signal. | Active / experimental by module |
| Two-path attention | Uses a slow contextual `OBSERVE → ATTEND → WARN` path plus a deterministic high-priority `REFLEX` path. | Active |
| Explainable behavior evidence | Shows proximity, rapid approach, heading, unusual motion, persistence, co-movement, strike-like motion, and path-conflict evidence. | Active |
| Compute-bounded pose reflex | Runs pose only for up to two close-range people to calculate wrist-speed and elbow-extension evidence. | Active |
| Collective hazard patterns | Provides convergence, coordinated-motion, encirclement, flanking, trap, and ambush analysis modules. These describe geometry, not intent. | Available module; not yet wired into the live iPhone alert policy |
| Directional haptics and explainable console | Emits directional JSONL/ESP32 haptics and renders video, world paths, evidence, replay, zoom, fullscreen, and clear-subject controls. | Active |
| Adaptive learning | Includes a simulation/PPO Training Lab for the slow attention policy; deterministic REFLEX remains outside the learned policy. | Training Lab / experimental |

HALO labels **observable motion and geometric interaction patterns only**. It does not identify people, infer intent, or determine whether someone is dangerous.

## Quick start

```bash
./scripts/00_check_env.sh
./scripts/01_create_env.sh
./scripts/02_install.sh
./scripts/04_run_demo.sh
```

The synthetic demo requires no phone, model, or hardware. It writes haptic events to `runs/demo/haptics.jsonl`.

```bash
./scripts/06_run_tests.sh
```

## iPhone live mode

Install optional vision dependencies and cache model weights:

```bash
./scripts/03_download_models.sh
```

Start the live pipeline:

```bash
.venv/bin/python scripts/live_iphone_camera.py
```

On the Mac, open:

- Console: `http://localhost:8080/demo`
- Processed video: `http://localhost:8080/video`
- Phone streaming page: `http://localhost:8080/phone`
- Training Lab: `http://localhost:8080/training`

iOS requires HTTPS for camera and motion permission. A Cloudflare quick-tunnel launcher is included:

```bash
./scripts/start_halo.sh --iphone --tunnel --vision
```

If port 8080 is occupied:

```bash
.venv/bin/python scripts/live_iphone_camera.py --port 8081
```

Keep the phone preview upright as mounted. The receiver corrects phone-to-camera rotation using the screen orientation; use the phone-page rotate control if the preview is sideways. `--hfov` configures assumed horizontal field of view (default: 75°).

## Dashboard controls

- `⤢`: expand a camera or predicted-world view; `Esc` closes it.
- Mouse wheel on the world view: zoom; double-click: return to auto-range.
- `STAB`: switch raw/stabilized motion trails.
- `CLEAR SUBJECTS`: reset tracker IDs, EKF, pose history, dashboard history, and haptic state; the next frame starts a new tracking epoch.
- One primary report is shown at a time. `REFLEX` outranks `WARN`, `ATTEND`, and `OBSERVE`; a secondary elevated subject may also be shown.

## How it works

### Sensing and pose

The iPhone sends JPEG frames plus motion messages. Browser orientation creates a device-to-earth rotation, then HALO applies the orientation-specific device-to-camera rotation:

\[
R_{earth\leftarrow camera}=R_{earth\leftarrow device}R_{device\leftarrow camera}
\]

For camera pixel \(p=[u,v,1]^T\), HALO computes a camera ray and stabilizes it in the world frame:

\[
r_c=K^{-1}p,qquad r_w=R_{earth\leftarrow camera}r_c
\]

The newest frame is always used and older buffered frames are dropped, preventing latency from accumulating.

### World state

Monocular depth is estimated from semantic object height and bounding-box height:

\[
d\approx\frac{H_{typical}f_y}{h_{bbox}}
\]

HALO estimates a state of position and velocity:

\[
[X,Y,Z,V_x,V_y,V_z]^T
\]

An EKF combines noisy bearing/elevation/depth observations with temporal prediction. Coordinates are rider-relative: +Y forward, +X right, +Z up.

### Prediction and conflict

Objects use constant-velocity prediction:

\[
p_o(t+\tau)=p_o(t)+v_o(t)\tau
\]

The rider uses a bicycle-model arc when turning. HALO samples both trajectories for four seconds and finds:

\[
t_{CPA}=\arg\min_\tau\|p_o(\tau)-p_r(\tau)\|,qquad
d_{CPA}=\min_\tau\|p_o(\tau)-p_r(\tau)\|
\]

That prevents a false warning when something is nearby but will safely pass outside the rider’s path.

### Attention and reflex

Collision probability is not the only signal. HALO aggregates explainable evidence:

```text
proximity · rapid approach · heading toward wearer · unusual image motion
rapid expansion · persistence · co-movement · strike-like limb motion
trajectory conflict
```

Weights change across six profiles: sparse/normal/crowded × indoor/outdoor. Crowded scenes reduce the importance of proximity; sparse outdoor scenes increase persistence and co-movement influence.

`REFLEX` is separate from ordinary collision-risk gating. It can trigger for rapid close-range motion, strong image expansion, or close-range pose evidence. It labels an **observable motion pattern**, never a person’s intent.

## Outputs

- **Haptics:** JSONL events plus an optional ESP32 HTTP POST.
- **Video:** annotated latest JPEG at `/frame.jpg` and MJPEG at `/video`.
- **Console:** unified `/api/world` snapshot rendered at `/demo`.
- **Training Lab:** simulation and attention-policy experimentation at `/training`; deterministic reflex always remains outside the learned policy.

To POST haptics to an ESP32:

```bash
HALO_HAPTIC_URL=http://192.168.4.1/haptic \
  .venv/bin/python scripts/live_iphone_camera.py
```

Example payload:

```json
{
  "direction": "left",
  "risk": 0.82,
  "ttc_s": 1.9,
  "conflict_s": 1.6,
  "object_id": "car-12",
  "intensity": "strong"
}
```

## Project layout

- `scripts/` — setup, demo, live launch, testing, and utility scripts.
- `dashboard_static/` — console, phone streaming page, and Training Lab UI.
- `src/halo/` — perception, IMU, tracking, EKF, collision, attention, pose, haptic, dashboard, and training modules.
- `config/` — ByteTrack and BoT-SORT-ReID configurations.
- `tests/` — deterministic core checks.
- `runs/` — generated recordings and output, ignored by Git.

## Research lineage and limits

HALO takes its baseline from Lim et al.’s smartphone-based motorcycle collision-warning work: monocular TTC, temporal filtering, trajectory prediction, and IMU-derived rider direction. HALO extends this with modern tracking, full phone-pose stabilization, approximate world-frame EKF state, curved rider paths, CPA path conflict, context-conditioned attention, pose evidence, explainability, and directional haptics. [Lim et al., 2021](https://www.sciopen.com/article/10.1108/JICV-11-2020-0014) · [Lim’s SUTD thesis](https://repository.sutd.edu.sg/esploro/outputs/graduate/Advanced-forward-collision-warning-system-for/9910477209846)

Before any safety claim, record and label scenarios, calibrate camera geometry, evaluate false alerts and misses across contexts, and test the haptic policy with users. HALO must not be used to determine a person’s identity, intent, or dangerousness.
