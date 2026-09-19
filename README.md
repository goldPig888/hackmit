# HALO — Predictive Micromobility Safety Prototype

HALO turns rear-camera detections and phone motion into interpretable collision-risk events:

`detection → temporal state → TTC gate → future paths → conflict → directional haptic event`

This is a hackathon prototype, **not a certified safety system**. Its risk thresholds must not be used to control a vehicle or replace rider awareness.

## Run it in order

```bash
./scripts/00_check_env.sh
./scripts/01_create_env.sh
./scripts/02_install.sh
./scripts/04_run_demo.sh
```

The demo needs no camera, YOLO model, or hardware. It simulates an approaching, crossing car and writes newline-delimited haptic events to `runs/demo/haptics.jsonl`.

For a real video/camera, first install the optional vision dependencies and cache the model:

```bash
./scripts/03_download_models.sh
./scripts/05_run_camera.sh path/to/rear-camera.mp4
# or
./scripts/05_run_camera.sh 0
```

## iPhone live stream (real IMU)

The advanced pipeline's camera-rotation compensation needs real IMU data. An iPhone provides it via Safari — no app install:

```bash
.venv/bin/python scripts/live_iphone_camera.py          # starts server on :8080
./scripts/start_halo.sh --iphone --tunnel              # or via the launcher + HTTPS tunnel
```

Then open `https://<tunnel-host>/phone` on the iPhone (iOS requires HTTPS for camera + motion access) and tap **Start HALO**. The page streams JPEG frames + gyro/accel/orientation over `/ingest`; the Mac runs detection and feeds the real pose into `AdvancedHaloPipeline`. Watch annotated output at `http://localhost:8080/video`.

Calibration notes: keep the preview on the phone page upright as held (use the rotate buttons if it looks sideways — the camera extrinsic assumes display-upright pixels). `--hfov` tunes the assumed lens FOV (default 75°).

To forward output to an ESP32 HTTP endpoint:

```bash
HALO_HAPTIC_URL=http://192.168.4.1/haptic ./scripts/05_run_camera.sh 0
```

## Repository layout

- `scripts/` — executable, numbered Bash entry points.
- `src/halo/` — reusable library; it has no coupling to a camera or ESP32.
- `tests/` — deterministic checks of TTC, conflict, and directional risk behavior.
- `runs/` — generated recordings, ignored by Git.

## Input coordinate convention

The core world model uses meters: rider is at `(0, 0)`, forward is positive `y`, and right is positive `x`. Camera detections are in pixels and are mapped into this local frame by `ImageGroundProjector`. That mapping is deliberately conservative: calibrate it before interpreting values as physical measurements.

## Haptic event contract

```json
{"direction":"left","risk":0.82,"ttc_s":1.9,"conflict_s":1.6,"object_id":"car-12"}
```

`direction` is the side where the threat lies (`left`, `right`, or `center`). The HTTP transport uses a POST of this JSON; without `HALO_HAPTIC_URL`, events are safely recorded locally.
