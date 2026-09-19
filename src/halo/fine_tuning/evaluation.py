"""Model evaluation utilities for custom detection models."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass
from collections import defaultdict

if TYPE_CHECKING:
    from ..detectors.base import BaseDetector


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    mAP: float
    accuracy: float
    confusion_matrix: dict
    per_class_metrics: dict[str, dict[str, float]]
    inference_time_ms: float


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    dataset_path: str
    total_images: int
    total_detections: int
    metrics: EvaluationMetrics
    errors: list[str]
    timestamp: str


class ModelEvaluator:
    """Evaluate custom detection models on test datasets."""

    def __init__(self, output_dir: str | Path = "runs/evaluation"):
        """Initialize model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, detector: BaseDetector, test_data_path: str, 
                 iou_threshold: float = 0.5, confidence_threshold: float = 0.5) -> EvaluationResult:
        """Evaluate detector on test dataset.
        
        Args:
            detector: Detector to evaluate
            test_data_path: Path to test dataset
            iou_threshold: IoU threshold for matching detections
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            EvaluationResult with detailed metrics
        """
        import time
        from datetime import datetime
        
        test_path = Path(test_data_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {test_data_path}")
        
        # Load ground truth annotations
        ground_truth = self._load_ground_truth(test_path)
        
        # Run inference
        predictions = []
        inference_times = []
        
        try:
            for image_path in self._get_test_images(test_path):
                # Load image (implementation depends on detector type)
                image = self._load_image(image_path)
                
                start_time = time.time()
                detections = detector.detect(image, timestamp_s=0.0)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                predictions.extend(detections)
        finally:
            detector.cleanup()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            ground_truth, 
            predictions, 
            iou_threshold, 
            confidence_threshold
        )
        
        metrics.inference_time_ms = float(np.mean(inference_times)) if inference_times else 0.0
        
        result = EvaluationResult(
            model_name=type(detector).__name__,
            dataset_path=str(test_path),
            total_images=len(list(self._get_test_images(test_path))),
            total_detections=len(predictions),
            metrics=metrics,
            errors=[],
            timestamp=datetime.now().isoformat()
        )
        
        self._save_evaluation_result(result)
        return result

    def _load_ground_truth(self, dataset_path: Path) -> dict:
        """Load ground truth annotations.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dictionary mapping image IDs to ground truth annotations
        """
        # Placeholder implementation - adapt to your annotation format
        # Common formats: COCO JSON, YOLO txt, Pascal VOC XML
        
        ground_truth = {}
        
        # Example for YOLO format
        for label_file in dataset_path.rglob("*.txt"):
            if label_file.name == "classes.txt":
                continue
                
            image_id = label_file.stem
            annotations = []
            
            with label_file.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = parts[:5]
                        annotations.append({
                            "class_id": int(class_id),
                            "bbox": [float(x_center), float(y_center), float(width), float(height)],
                            "confidence": 1.0  # Ground truth has perfect confidence
                        })
            
            ground_truth[image_id] = annotations
        
        return ground_truth

    def _get_test_images(self, dataset_path: Path) -> list[Path]:
        """Get list of test images.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            List of image paths
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        return [p for p in dataset_path.rglob("*") if p.suffix.lower() in image_extensions]

    def _load_image(self, image_path: Path):
        """Load image for inference.
        
        Args:
            image_path: Path to image
            
        Returns:
            Loaded image (format depends on detector)
        """
        try:
            import cv2
            return cv2.imread(str(image_path))
        except ImportError:
            # Fallback to PIL if opencv not available
            from PIL import Image
            return Image.open(image_path)

    def _calculate_metrics(self, ground_truth: dict, predictions: list, 
                          iou_threshold: float, confidence_threshold: float) -> EvaluationMetrics:
        """Calculate evaluation metrics.
        
        Args:
            ground_truth: Ground truth annotations
            predictions: Model predictions
            iou_threshold: IoU threshold for matching
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            EvaluationMetrics object
        """
        # Filter predictions by confidence
        filtered_predictions = [p for p in predictions if p.confidence >= confidence_threshold]
        
        # Calculate per-class metrics
        per_class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        # Match predictions to ground truth (simplified - real implementation needs proper IoU)
        for pred in filtered_predictions:
            # In real implementation, calculate IoU with ground truth
            # and count as TP if IoU > threshold
            per_class_metrics[pred.label]["tp"] += 1
        
        # Count false negatives (ground truth not matched)
        for image_id, annotations in ground_truth.items():
            for ann in annotations:
                # Simplified - in real implementation, check if this was matched
                class_name = f"class_{ann['class_id']}"  # Convert to class name
                per_class_metrics[class_name]["fn"] += 1
        
        # Calculate aggregate metrics
        total_tp = sum(m["tp"] for m in per_class_metrics.values())
        total_fp = sum(m["fp"] for m in per_class_metrics.values())
        total_fn = sum(m["fn"] for m in per_class_metrics.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-class metrics
        for class_name, counts in per_class_metrics.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0.0
            
            per_class_metrics[class_name] = {
                "precision": class_precision,
                "recall": class_recall,
                "f1_score": class_f1
            }
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mAP=0.0,  # Would need proper implementation
            accuracy=0.0,  # Not typically used for detection
            confusion_matrix={},  # Would need proper implementation
            per_class_metrics=dict(per_class_metrics),
            inference_time_ms=0.0  # Set by caller
        )

    def _save_evaluation_result(self, result: EvaluationResult) -> None:
        """Save evaluation result to file.
        
        Args:
            result: Evaluation result to save
        """
        timestamp = result.timestamp.replace(":", "-").replace(".", "-")
        output_path = self.output_dir / f"evaluation_{timestamp}.json"
        
        with output_path.open("w") as f:
            json.dump({
                "model_name": result.model_name,
                "dataset_path": result.dataset_path,
                "total_images": result.total_images,
                "total_detections": result.total_detections,
                "metrics": {
                    "precision": result.metrics.precision,
                    "recall": result.metrics.recall,
                    "f1_score": result.metrics.f1_score,
                    "mAP": result.metrics.mAP,
                    "accuracy": result.metrics.accuracy,
                    "per_class_metrics": result.metrics.per_class_metrics,
                    "inference_time_ms": result.metrics.inference_time_ms
                },
                "errors": result.errors,
                "timestamp": result.timestamp
            }, f, indent=2)

    def compare_models(self, detector1: BaseDetector, detector2: BaseDetector, 
                      test_data_path: str) -> dict:
        """Compare two detectors on the same test set.
        
        Args:
            detector1: First detector
            detector2: Second detector
            test_data_path: Path to test dataset
            
        Returns:
            Dictionary with comparison results
        """
        result1 = self.evaluate(detector1, test_data_path)
        result2 = self.evaluate(detector2, test_data_path)
        
        comparison = {
            "model1": result1.model_name,
            "model2": result2.model_name,
            "metrics_comparison": {
                "precision": {
                    "model1": result1.metrics.precision,
                    "model2": result2.metrics.precision,
                    "difference": result2.metrics.precision - result1.metrics.precision
                },
                "recall": {
                    "model1": result1.metrics.recall,
                    "model2": result2.metrics.recall,
                    "difference": result2.metrics.recall - result1.metrics.recall
                },
                "f1_score": {
                    "model1": result1.metrics.f1_score,
                    "model2": result2.metrics.f1_score,
                    "difference": result2.metrics.f1_score - result1.metrics.f1_score
                },
                "inference_time_ms": {
                    "model1": result1.metrics.inference_time_ms,
                    "model2": result2.metrics.inference_time_ms,
                    "speedup": result1.metrics.inference_time_ms / result2.metrics.inference_time_ms if result2.metrics.inference_time_ms > 0 else 0
                }
            }
        }
        
        return comparison