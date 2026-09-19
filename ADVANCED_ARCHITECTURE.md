# Advanced HALO Architecture - Camera-Frame Realignment

## Overview

This document describes the advanced camera-frame realignment architecture implemented to address the fundamental issue you identified: **distinguishing camera motion from object motion**.

## The Problem

In the original HALO implementation:
- Objects were tracked in raw pixel coordinates
- When you rotate your phone left, a stationary car appears to move right
- The system would incorrectly conclude "CAR MOVED RIGHT! Increased risk!"
- This makes tracking unreliable for handheld/backpack-mounted cameras

## The Solution

**Never predict danger in raw camera coordinates.**

Instead, implement the architecture:
```
Perception → Ego-motion compensation → World-space tracking → Prediction
```

## New Components

### 1. IMU Integration (`src/halo/imu/`)

**IMUProcessor**: Process gyroscope and accelerometer data
- Quaternion integration for orientation tracking
- Bias calibration
- Smoothing over time windows

**CameraPoseEstimator**: Estimate camera pose from IMU
- Convert quaternion to rotation matrix
- Estimate confidence based on angular velocity
- Transform pixel observations to world rays

**RotationCompensator**: Stabilize observations
- **Key operation**: `r_t^w = R_t * K^-1 * p_t`
- Transform pixel observations to stable world frame
- Track bearing and elevation (not pixels)
- Calculate bearing rates for motion analysis

### 2. Stabilized World Frame Tracking (`src/halo/tracking/`)

**WorldFrameTracker**: Track objects in HALO frame
- Coordinate frame: +Y = forward, +X = right, +Z = up
- Depth estimation from bbox size
- Ego-motion separation: `v_object = v_observed - v_ego`
- Persistent tracking in stable coordinates

**ObjectState**: Object state in HALO frame
- Position: [X, Y, Z] (meters)
- Velocity: [Vx, Vy, Vz] (m/s)
- Bearing/elevation (radians)
- Relative depth estimate
- Covariance matrix for uncertainty

**EgoTracker**: Track rider/wearer motion
- Bicycle model for trajectory prediction
- IMU-based heading estimation
- GPS support (optional)
- Turn detection

### 3. CPA-Based Collision Detection (`src/halo/collision/`)

**CPADetector**: Closest Point of Approach calculation
- **Math**: `t_CPA = -(r_0 · v_rel) / ||v_rel||^2`
- **Math**: `d_CPA = ||r_0 + v_rel * t_CPA||`
- Much better than simple TTC
- Handles vehicles that will safely pass beside you

**CollisionRiskAssessor**: Comprehensive risk assessment
- Combines CPA with speed, distance, time, class factors
- Risk levels: SAFE, LOW, MEDIUM, HIGH, CRITICAL
- Threat direction determination
- Contributing factor analysis

### 4. Extended Kalman Filter (`src/halo/filters/`)

**ExtendedKalmanFilter**: Nonlinear state estimation
- State: [X, Y, Z, Vx, Vy, Vz]
- Measurement: [bearing, elevation, distance]
- Jacobian-based linearization
- Uncertainty propagation

**UnscentedKalmanFilter**: Alternative UKF implementation
- Sigma point propagation
- No Jacobian calculation needed
- More robust for highly nonlinear systems

### 5. Visual-Inertial Odometry (`src/halo/vio/`)

**VisualOdometryEstimator**: Visual ego-motion from background features
- ORB feature detection
- Lucas-Kanade optical flow
- Mask out YOLO detections (track only background)
- Homography-based rotation estimation

**VIOFusion**: Fuse IMU and visual estimates
- Complementary filter: `R_camera = α * R_IMU + (1-α) * R_vision`
- Confidence weighting
- Temporal smoothing

### 6. Advanced Pipeline (`src/halo/advanced_pipeline.py`)

**AdvancedHaloPipeline**: Complete advanced system
- Configurable features (IMU, VIO, EKF, CPA)
- Full integration of all components
- Backward-compatible with legacy API
- System status monitoring

## Key Improvements

### Before (Original)
```python
YOLO → ByteTrack → pixel coords → simple projection → Kalman → risk
```
- Tracks in camera coordinates
- Camera rotation = object motion
- Simple TTC threshold
- No uncertainty quantification

### After (Advanced)
```python
YOLO → ByteTrack → IMU compensation → world frame → EKF → CPA → risk
```
- Tracks in stable HALO frame
- Camera rotation ≠ object motion
- CPA-based collision reasoning
- Explicit uncertainty propagation

## Architecture Diagram

```
                  FRAME t
                     │
       ┌─────────────┴──────────────┐
       ↓                            ↓
    YOLO                         IMU
       ↓                    gyro + accel
   ByteTrack                       │
       ↓                            │
object observations                  │
       │                            │
       └─────────────┬──────────────┘
                     ↓
            CAMERA POSE ESTIMATOR
                     │
               Rcw(t), tcw(t)
                     │
                     ↓
            EGO-MOTION REMOVAL
                     │
                     ↓
            STABILIZED TRACKS
                     │
           ┌─────────┴─────────┐
           ↓                   ↓
     relative depth       optical flow
           │                   │
           └─────────┬─────────┘
                     ↓
              WORLD STATE
                     │
       [X,Y,Z,Vx,Vy,Vz,Σ]
                     │
                EKF / UKF
                     ↓
          FUTURE TRAJECTORIES
            ↙             ↘
       OTHER OBJECT       WEARER
            ╲             ╱
             ╲           ╱
              ↓         ↓
           CLOSEST APPROACH
                  +
                 TTC
                  +
            UNCERTAINTY
                  ↓
               RISK
                  ↓
        directional haptics
```

## Usage

### Basic Usage
```python
from halo import AdvancedHaloPipeline, AdvancedPipelineConfig

# Create configuration
config = AdvancedPipelineConfig(
    enable_imu_compensation=True,
    enable_visual_odometry=False,  # Optional, computationally intensive
    enable_ekf=True,
    enable_cpa_collision=True
)

# Initialize pipeline
pipeline = AdvancedHaloPipeline(config)

# Update with detection and IMU data
risk_assessment = pipeline.update(
    detection=detection,
    imu_data=(gyro, accel, quaternion)
)
```

### Testing
```bash
# Test all advanced components
.venv/bin/python scripts/test_advanced_pipeline.py

# Expected output: 7 passed, 0 failed
```

### Install Advanced Dependencies
```bash
# Add scipy for advanced math operations
pip install scipy

# Or use the unified script
bash scripts/start_halo.sh --advanced
```

## Mathematical Foundation

### Camera Rotation Compensation
The key operation that solves the fundamental issue:

```
r_t^w = R_t * K^-1 * p_t
```

Where:
- `p_t` = pixel observation
- `K^-1` = inverse camera intrinsics
- `R_t` = camera rotation from IMU
- `r_t^w` = world ray in stable frame

### CPA Calculation
Collision prediction using closest point of approach:

```
t_CPA = -(r_0 · v_rel) / ||v_rel||^2
d_CPA = ||r_0 + v_rel * t_CPA||
```

Where:
- `r_0` = initial relative position
- `v_rel` = relative velocity
- `t_CPA` = time to closest approach
- `d_CPA` = distance at closest approach

### EKF Measurement Model
Nonlinear measurement for bearing, elevation, distance:

```
bearing = atan2(X, Y)
elevation = atan2(Z, sqrt(X^2 + Y^2))
distance = sqrt(X^2 + Y^2 + Z^2)
```

## Benefits

1. **Robust to camera motion**: Distinguishes camera rotation from object motion
2. **Stable tracking**: Objects tracked in world frame, not camera frame
3. **Better collision prediction**: CPA handles safe passage cases
4. **Uncertainty quantification**: EKF provides state uncertainty
5. **Future-ready**: Foundation for VIO and full SLAM

## Limitations

- **Monocular depth**: Approximate from bbox size, not metric
- **IMU required**: Requires phone IMU data (not all devices)
- **Computational cost**: EKF and VIO add processing overhead
- **Calibration needed**: Camera intrinsics should be calibrated

## Future Work

1. **Add actual camera calibration** for accurate intrinsics
2. **Implement quaternion interpolation** for VIO fusion
3. **Add depth estimation** (e.g., Depth Anything) for better depth
4. **Full SLAM** for metric 6DOF pose estimation
5. **Bearing rate detection** for constant-bearing collision patterns

## Testing Results

All 7 advanced pipeline tests passing:
- ✓ IMU Integration
- ✓ Rotation Compensation
- ✓ World Frame Tracking
- ✓ Ego Tracking
- ✓ CPA Collision Detection
- ✓ Extended Kalman Filter
- ✓ Full Advanced Pipeline

## Comparison to Lim et al. Paper

| Aspect | Lim et al. | HALO Advanced |
|--------|-----------|----------------|
| Frame | 2D pixels | Stabilized world frame |
| Ego motion | IMU lean only | IMU + optional VIO |
| Collision | TTC + direction | CPA + uncertainty |
| Prediction | LSTM trajectory | EKF state estimation |
| Applicability | Motorcycle (rigid) | Pedestrian/cyclist (dynamic) |

The advanced architecture maintains the scientific foundation while modernizing for dynamic camera use cases.