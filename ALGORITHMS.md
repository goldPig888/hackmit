# HALO — Algorithm & Novelty Reference

Technical reference for every calculation in HALO, organized by pipeline stage.
Each section gives the math, the code that implements it, and why the approach
was chosen over the standard alternative.

```
pixels ──► ego-motion compensation ──► stabilized tracking ──► prediction
                                                                  │
              ┌───────────────────────────────────────────────────┘
              ▼
     CPA / TTC / conflict ──► risk ──► attention ──► directional haptics
```

---

## 1. Perception → stabilized world frame

### 1.1 Simple ground projection (baseline pipeline)

`src/halo/geometry.py` — `ImageGroundProjector`

Bottom-of-bbox is treated as the ground contact point:

```
forward = (720 − (cy + h/2)) / pixels_per_meter
lateral = (cx − width/2)     / pixels_per_meter
```

Deliberately approximate — the code comments state a calibrated homography
should replace it before values are treated as metric truth.

### 1.2 Rotational decoupling (advanced pipeline) — core novelty

`src/halo/imu/camera_pose.py`, `src/halo/imu/rotation_compensator.py`

Each detection becomes a unit ray in a stabilized world frame:

```
r_w = R_t · K⁻¹ · p_t
```

- `p_t`  — homogeneous pixel observation
- `K⁻¹`  — inverse camera intrinsics (`CameraIntrinsics.K_inv`)
- `R_t`  — camera rotation from IMU quaternion
- `r_w`  — world ray; converted to bearing `atan2(x, y)` and elevation
  `atan2(z, √(x²+y²))`

**Why it matters:** pixel-space trackers cannot distinguish "the car moved
right" from "you rotated the phone left." Tracking *bearing and elevation*
instead of pixels makes object-motion inference invariant to camera rotation
by construction. The inverse map `p = K·Rᵀ·r_w` (in
`RotationCompensator.bearing_to_pixel`) reprojects stabilized tracks back to
pixels for the video overlay.

### 1.3 Quaternion integration

`src/halo/imu/imu_processor.py`

```
q̇ = ½ · q ⊗ ω        (ω as pure quaternion [0, ωx, ωy, ωz])
q_{t+dt} = normalize(q + q̇·dt)
```

Full Hamilton quaternion multiply implemented explicitly
(`_quaternion_multiply`), plus stationary-window gyro/accel bias calibration
(`calibrate`) and a sliding smoothing window (`get_smoothed_reading`).

### 1.4 Semantic monocular depth

`src/halo/tracking/world_tracker.py` — `_estimate_depth_from_bbox`

```
depth = h_typical(label) · f_y / bbox_height_px
```

Class-conditioned pinhole depth — person 1.7 m, car 1.5 m, truck 2.5 m,
bus 3.0 m, bicycle 1.0 m, motorcycle 1.2 m — clamped to [1, 50] m.
Metric-ish depth from a single camera with no depth network, using the
detected class's expected physical size.

---

## 2. Temporal filtering — three tiers

### 2.1 Constant-velocity Kalman (baseline)

`src/halo/kalman.py` — `ConstantVelocityKalman`

State `[x, y, vx, vy]`, position-only measurements, and a proper
piecewise-constant-white-acceleration process noise matrix:

```
Q = q · [[dt⁴/4, 0, dt³/2, 0],
         [0, dt⁴/4, 0, dt³/2],
         [dt³/2, 0, dt²,  0],
         [0, dt³/2, 0, dt² ]]
```

Position uncertainty is exposed as `σ = √(tr(P_pos)/2)` and consumed
downstream by the conflict model.

### 2.2 Extended Kalman Filter (advanced)

`src/halo/filters/extended_kalman.py`

State `[X, Y, Z, Vx, Vy, Vz]`; measurement is **spherical**
`[bearing, elevation, distance]`:

```
bearing   = atan2(X, Y)
elevation = atan2(Z, √(X²+Y²))
distance  = √(X²+Y²+Z²)
```

The 3×6 measurement Jacobian is fully hand-derived — e.g.
`∂bearing/∂X = Y/(X²+Y²)`, `∂elevation/∂Z = √(X²+Y²)/(X²+Y²+Z²)` — with
singularity guards near the origin. Necessary because spherical measurements
are nonlinear functions of the Cartesian state.

### 2.3 Unscented Kalman Filter (alternative)

`src/halo/filters/unscented_kalman.py`

Sigma-point propagation through the same nonlinear models — no Jacobians.
Escape hatch for when bearing measurements are too nonlinear for EKF
linearization (close range, large uncertainty). Standard Van der Merwe
weights `Wm`, `Wc` with `α, β, κ` parameterization.

---

## 3. Looming TTC — depth-free time-to-contact

`src/halo/ttc.py` — `ExpansionTTC`

Apparent bbox area scales as `A ∝ 1/d²`. Differentiating:

```
Ȧ = −2A·ḋ/d   ⟹   TTC = d/|ḋ| = 2A/Ȧ
```

A closed-form time-to-contact from **image expansion alone** — no depth, no
velocity estimate, no camera model. Implemented as a finite-difference rate
over an 8-sample window, gated by a minimum expansion rate, clamped to
[0.05, 30] s. This is the "looming" cue biological vision systems use.

---

## 4. Motion models & tracking in the stabilized frame

### 4.1 Two different motion models, one frame

`src/halo/trajectory.py`, `src/halo/tracking/ego_tracker.py`

- **Object**: constant velocity — `p(t) = p₀ + v·t`
- **Rider (ego)**: bicycle/arc model. For yaw rate ω and speed v, turning
  radius `R = v/ω`:

  ```
  x(t) = R·(1 − cos ωt)
  y(t) = R·sin ωt
  ```

  degenerating to a straight line when `|ω| < 1e-4`.

Pairing a curved ego path against a linear object path makes conflict search
meaningful *while the rider is turning* — a TTC computed on the assumption
of straight-line rider motion misses collisions that happen mid-turn.

### 4.2 Ego-motion separation & bearing rates

`src/halo/tracking/world_tracker.py`, `src/halo/imu/rotation_compensator.py`

- `v_object = v_observed − v_ego`
- Closing rate: `r̂ · v_rel` (radial component; negative = approaching)
- Bearing rate from stabilized history: `Δθ/Δt` — invariant to camera
  rotation, so it measures true lateral drift.

### 4.3 Velocity smoothing & covariance growth

`world_tracker.py::_update_existing_object`

- Exponential velocity smoothing: `v ← 0.3·v_meas + 0.7·v_prev`
- Confidence grows with track age; covariance inflates `×(1 + 0.1·dt)`
  between observations.

---

## 5. CPA — closest point of approach

`src/halo/collision/cpa_detector.py`

Closed form over relative kinematics `r(t) = r₀ + v_rel·t`:

```
t_CPA = −(r₀ · v_rel) / ‖v_rel‖²
d_CPA = ‖r₀ + v_rel · t_CPA‖
```

`will_collide = (0 < t_CPA < horizon) ∧ (d_CPA < safe_distance)`.

**Why not TTC:** time-to-collision answers "when do we hit" and presupposes
a hit. CPA answers "will we hit, when, and by how much" — a fast car passing
3 m wide produces a correctly-safe `d_CPA` instead of a scary TTC.

`calculate_trajectory_cpa` does the same on *predicted trajectory pairs*
(argmin of `‖p_obj(tᵢ) − p_ego(tᵢ)‖` over sampled times) for when the
constant-velocity assumption breaks, and `assess_miss_distance` classifies
the outcome: `collision / near_miss / safe_pass / moving_away`.

### 5.1 Constant-bearing detection

`calculate_bearing_rate_conflict` — the maritime collision signature:

```
|θ̇| ≈ 0   while   range decreasing  ⟹  collision course
```

An object looming (expanding bbox) with near-zero bearing drift is on a
collision course — exactly the threat pattern humans detect worst.
(Note: `bearing_rate` is currently stubbed at 0 pending history wiring.)

---

## 6. Risk composition

### 6.1 Baseline — interpretable product form

`src/halo/risk.py`, `src/halo/trajectory.py::closest_conflict`

```
P_conflict = exp(−½ (d_min/σ)²)     σ = σ₀ + 0.15·step   (grows with horizon)
risk       = P_conflict · e^(−t/τ₀) · e^(−d/d₀) · class_mult · gate · conf
```

- Conflict probability is a **Gaussian overlap** between predicted paths,
  and σ grows per prediction step — the model discounts its own distant
  future predictions. Uncertainty-aware by construction.
- Temporal `e^(−t/τ₀)` (τ₀=2.5 s) and spatial `e^(−d/d₀)` (d₀=1.2 m)
  decay terms.
- A hard-ish gate (`closing ? 1.0 : 0.35`) requires radial closure or real
  predicted overlap — non-approaching objects can't reach high risk.

### 6.2 Advanced — CPA-driven levels

`src/halo/collision/collision_risk.py`

`probability = base · dist_factor · time_factor · speed_factor · class_factor`
→ bucketed into SAFE/LOW/MEDIUM/HIGH/CRITICAL, with an explicit
`contributing_factors` dict (distance, speed, heading alignment, CPA
distance, time-to-CPA, class, ego-turning). Explainability is part of the
output, not an afterthought.

---

## 7. Attention layer — two-path alerting

`src/halo/attention.py`

### 7.1 Reflex fast path

```
closing < −3 m/s  ∧  range < 6 m  ∧  (expanding ∨ px_rate > 120 px/s)
∧  TTC < 1.5 s   ⟹   REFLEX (attention = 1.0)
```

Bypasses the slow accumulator entirely — modeled on reflexive response,
where latency costs more than false positives.

### 7.2 Slow path — interpretable weighted score

```
attention = 0.28·proximity + 0.26·approach + 0.16·heading_toward
          + 0.10·unusual   + 0.10·persistence + 0.10·comovement
```

- `heading_toward = clamp(−(v·p)/(|v||p|))` — dot-product alignment of the
  object's velocity with the line to the wearer.
- **Co-movement term**: ego moving ∧ tracked > 4 s ∧ < 12 m ∧
  `|bearing_rate| < 0.05` ⟹ "this object is pacing me" — tailing detection
  that no per-frame risk score captures.
- **Scene-conditioned thresholds**: crowded scenes raise the alert bar
  (0.88) to suppress alert fatigue; sparse scenes lower it (0.78).
- Every score returns a `reasons` list — the alert explains itself.

---

## 8. Visual-inertial odometry

`src/halo/vio/visual_odometry.py`, `src/halo/vio/vio_fusion.py`

- ORB features + pyramidal Lucas–Kanade optical flow, **masked against YOLO
  detections** — ego-motion is estimated only from background, so the very
  objects being tracked can't corrupt the motion estimate.
- Homography → SVD → nearest rotation matrix (`U·Vᵀ`, det correction).
- Complementary fusion `R = α·R_IMU + (1−α)·R_vision` (α=0.7), then SVD
  re-orthogonalization — the nearest valid rotation in Frobenius norm.
  (Proper version would slerp quaternions; noted in code.)

---

## 9. Collective / synchronized hazard reasoning

`src/halo/hazards/`

Beyond single-object safety:

- **Convergence** (`sync_detector.py`): ≥2 closing objects within 5 m;
  severity scales with count.
- **Coordinated movement**: pairwise velocity correlation over a sliding
  window > 0.7 while both are closing.
- **Encirclement**: sort object bearings around the rider and find the
  **maximum angular gap** — if the largest uncovered sector is < 90°, the
  rider is surrounded. Circular gap statistics applied to threat detection.
- **Pattern analyzer** (`pattern_analyzer.py`):
  - *Ambush* — a close "distraction" object while flankers close in.
  - *Trap* — objects occupying ≥3 of 4 quadrants (blocked escape routes).
  - *Flanking* — symmetric closing from left and right simultaneously,
    confidence scaled by symmetry.

---

## 10. Output contract

`src/halo/risk.py::intensity`, `src/halo/haptics.py`

Risk → intensity bands (`weak < 0.55 < medium < 0.75 < strong`) →
directional haptic event `{direction, risk, ttc_s, conflict_s, object_id}`
logged to `haptics.jsonl` and optionally POSTed to an ESP32 endpoint.

---

## Novelty summary (for writeups)

1. **Rotational decoupling** — tracking in IMU-stabilized bearing space
   (`r_w = R·K⁻¹·p`) makes object-motion inference invariant to handheld
   camera rotation, the failure mode of pixel-space trackers.
2. **Three independent depth-weak cues fused** — looming TTC (`2A/Ȧ`),
   semantic pinhole depth (`h·f/px`), bearing rates — no stereo, no depth
   model.
3. **CPA + constant-bearing reasoning** instead of TTC thresholds —
   separates "fast approach that misses" from "slow drift that hits."
4. **Self-discounting predictions** — Gaussian path overlap with σ growing
   per step; risk honestly degrades with horizon.
5. **Two-path attention** — reflex fast-path vs. interpretable weighted
   score with scene-adaptive thresholds and co-movement (tailing)
   detection. Every alert carries its reasons.
6. **Collective threat reasoning** — pairwise velocity correlation and
   angular-gap encirclement analysis beyond single-object alerts.

## Honest limitations

- Monocular depth is class-heuristic and clamped to [1, 50] m.
- VIO rotation fusion averages matrices (SVD re-orthogonalized), not
  geodesic/quaternion slerp.
- `calculate_bearing_rate_conflict` stubs `bearing_rate = 0` — needs
  bearing-history wiring to activate.
- UKF `α = 1e-3` produces a very tight sigma-point spread.
- Camera intrinsics are defaults (f=800 px); real calibration would
  tighten everything downstream.
- Hackathon prototype — thresholds are not safety-certified.
