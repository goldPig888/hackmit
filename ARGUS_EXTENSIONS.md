# ARGUS Extensions - Custom Detection & Testing Framework

I've successfully extended your ARGUS system with a comprehensive plugin architecture for custom detection models, testing tools, and model fine-tuning utilities. Here's what has been added:

## 🎯 New Features Overview

### 1. Custom Detection Model Plugin Architecture

**Location:** `src/argus/detectors/`

- **Base Detector Interface** (`base.py`): Abstract base class that all custom detectors must implement
- **Detector Registry** (`registry.py`): Central registry for managing and instantiating custom detectors
- **YOLO Integration** (`yolo.py`): Wrapper for YOLO models using ultralytics
- **Weapons Detection Template** (`weapons.py`): Ready-to-use template for your custom weapons detection model

**Usage Example:**
```python
from argus.detectors import DetectorRegistry, WeaponsDetector

# Register your custom detector
DetectorRegistry.register("weapons", WeaponsDetector)

# Create and use detector
detector = DetectorRegistry.create("weapons", model_path="my_model.pt")
detections = detector.detect(frame, timestamp_s=0.0)
```

### 2. Component-Level Testing Tools

**Location:** `src/argus/testing/`

- **Component Tester** (`component_tester.py`): Test individual pipeline components in isolation
- **Mock Data Generators** (`mock_data.py`): Generate synthetic detection and IMU data for testing
- **Pipeline Tester** (`pipeline_tester.py`): Test the full ARGUS pipeline with custom detectors

**Available Tests:**
- Geometry projection testing
- Kalman filter tracking
- TTC estimation
- Trajectory prediction
- Full component suite

**Run Component Tests:**
```bash
.venv/bin/python scripts/test_components.py
```

### 3. Synchronized Hazard Detection

**Location:** `src/argus/hazards/`

- **Sync Hazard Detector** (`sync_detector.py`): Detect coordinated threats like:
  - Multiple vehicles converging on rider
  - Coordinated movement patterns
  - Encirclement scenarios
  - Ambush patterns

- **Pattern Analyzer** (`pattern_analyzer.py`): Advanced pattern recognition for:
  - Ambush scenarios (distraction + flanking)
  - Trap scenarios (blocking escape routes)
  - Flanking maneuvers

**Usage Example:**
```python
from argus.hazards import SyncHazardDetector

detector = SyncHazardDetector()
hazards = detector.update(track_states)
for hazard in hazards:
    print(f"{hazard.hazard_type}: {hazard.description} (severity: {hazard.severity})")
```

### 4. Model Fine-Tuning & Evaluation

**Location:** `src/argus/fine_tuning/`

- **Model Trainer** (`model_trainer.py`): Training utilities with:
  - Base trainer class for framework-agnostic training
  - YOLO-specific trainer using ultralytics
  - Training configuration management
  - Checkpoint saving and callbacks

- **Model Evaluator** (`evaluation.py`): Comprehensive evaluation with:
  - Precision, recall, F1 score metrics
  - Per-class metrics
  - Inference time measurement
  - Model comparison utilities

- **Data Loader** (`data_loader.py`): Dataset management with:
  - Support for YOLO, COCO, Pascal VOC formats
  - Auto-format detection
  - Dataset statistics
  - Train/val/test splitting

**Training Commands:**
```bash
# Prepare dataset
.venv/bin/python scripts/train_model.py prepare /path/to/dataset --output /path/to/output

# Train YOLO model
.venv/bin/python scripts/train_model.py train-yolo --dataset data.yaml --model-name weapons_detector

# Evaluate model
.venv/bin/python scripts/train_model.py evaluate model.pt /path/to/test_data
```

## 🚀 New Scripts

### `scripts/test_components.py`
Test individual ARGUS pipeline components:
- Geometry projection
- Kalman filter tracking
- TTC estimation
- Trajectory prediction
- Full component suite

### `scripts/test_custom_detectors.py`
Test custom detection models:
- Detector registry functionality
- YOLO detector integration
- Weapons detector integration
- Pipeline with custom detectors
- Synchronized hazard detection

### `scripts/train_model.py`
Model training and evaluation CLI:
- Dataset preparation
- YOLO model training
- Custom model training
- Model evaluation

## 📁 New Directory Structure

```
src/argus/
├── detectors/           # Custom detection models
│   ├── __init__.py
│   ├── base.py         # Base detector interface
│   ├── registry.py     # Detector registry
│   ├── yolo.py         # YOLO integration
│   └── weapons.py      # Weapons detection template
├── testing/            # Testing utilities
│   ├── __init__.py
│   ├── component_tester.py
│   ├── mock_data.py
│   └── pipeline_tester.py
├── hazards/            # Hazard detection
│   ├── __init__.py
│   ├── sync_detector.py
│   └── pattern_analyzer.py
└── fine_tuning/        # Model training
    ├── __init__.py
    ├── model_trainer.py
    ├── evaluation.py
    └── data_loader.py
```

## 🔧 Integration with Existing ARGUS

The new extensions are fully compatible with your existing ARGUS pipeline:

```python
from argus import ArgusPipeline, BaseDetector
from argus.detectors import DetectorRegistry

# Register your custom detector
DetectorRegistry.register("my_custom_detector", MyCustomDetector)

# Use with existing pipeline
pipeline = ArgusPipeline()
detector = DetectorRegistry.create("my_custom_detector")
detections = detector.detect(frame, timestamp_s=0.0)

# Process detections through ARGUS pipeline
for detection in detections:
    risk_assessment = pipeline.update(detection, rider_speed_mps=4.0)
    if risk_assessment:
        # Handle risk assessment
        pass
```

## 🎯 Next Steps for Your Use Case

### 1. Implement Your Weapons Detection Model

Edit `src/argus/detectors/weapons.py`:
```python
def detect(self, frame, timestamp_s: float) -> list[Detection]:
    # Replace with your actual model inference
    results = self._model.infer(frame)
    detections = []
    for result in results:
        if result.confidence >= self.confidence_threshold:
            detections.append(Detection(
                object_id=f"weapon-{result.id}",
                label=result.label,
                confidence=result.confidence,
                bbox=result.bbox,
                timestamp_s=timestamp_s
            ))
    return detections
```

### 2. Train/Fine-tune Your Model

```bash
# Prepare your dataset
.venv/bin/python scripts/train_model.py prepare /path/to/weapons_dataset --output data/weapons

# Train your model
.venv/bin/python scripts/train_model.py train-yolo \
    --dataset data/weapons/data.yaml \
    --model-name weapons_detector \
    --base-model yolo11n.pt \
    --epochs 50

# Evaluate
.venv/bin/python scripts/train_model.py evaluate runs/training/weapons_detector_*/best.pt data/weapons/test
```

### 3. Test Your Integration

```bash
# Test component by component
.venv/bin/python scripts/test_components.py

# Test custom detectors
.venv/bin/python scripts/test_custom_detectors.py
```

### 4. Integrate Sync Hazard Detection

Add to your main pipeline:
```python
from argus.hazards import SyncHazardDetector

# Initialize sync hazard detector
sync_detector = SyncHazardDetector()

# In your main loop:
track_states = [...]  # Get from your pipeline
hazards = sync_detector.update(track_states)
for hazard in hazards:
    print(f"SYNC HAZARD: {hazard.hazard_type} - {hazard.description}")
```

## 🧪 Testing Results

All new components have been tested and are working:

- ✅ Component testing suite passes
- ✅ Custom detector registry works
- ✅ Weapons detector template integrates
- ✅ Pipeline testing with custom detectors works
- ✅ Sync hazard detection initialized

## 📝 Key Design Principles

1. **Modularity**: Each component can be used independently or together
2. **Extensibility**: Easy to add new detectors, patterns, or training methods
3. **Compatibility**: Works with existing ARGUS pipeline without modifications
4. **Testability**: Comprehensive testing tools for development and validation
5. **Production-Ready**: Includes error handling, cleanup, and resource management

## 🌐 Remote Monitoring System

### Live Dashboard
- **Real-time web dashboard** with Server-Sent Events streaming
- **Mobile-responsive design** optimized for phone access
- **Cloudflare tunnel integration** for remote access
- **REST API endpoints** for data access and integration

### Video Overlay
- **Real-time video overlays** with detection boxes and risk scores
- **Trajectory visualization** showing predicted paths
- **Hazard warning overlays** for coordinated threats
- **System information display** (FPS, counts, timing)

### Cloudflare Tunnel
- **Quick tunnel setup** for temporary public URLs
- **Permanent tunnel configuration** for custom domains
- **Secure remote access** from any device
- **Mobile phone compatibility** for field monitoring

## 🚀 Quick Start for Remote Monitoring

```bash
# Start demo dashboard with simulated data
bash scripts/start_monitoring.sh
# Select option 1

# Start Cloudflare tunnel for phone access
bash scripts/start_quick_tunnel.sh

# Access the provided URL on your phone
```

## 📱 Mobile Access Features

- **Responsive layout** adapts to different screen sizes
- **Touch-optimized interface** for mobile interaction
- **Real-time updates** via SSE streaming
- **Connection status indicator** for monitoring
- **Low-bandwidth optimization** for cellular networks

This framework gives you a solid foundation to incorporate your custom weapons detection model, test it thoroughly, and integrate it with the sophisticated ARGUS safety system. The plugin architecture allows you to easily add more custom detectors (sync hazards, other specialized models) following the same pattern. The remote monitoring system enables real-time observation from your phone anywhere with internet access.