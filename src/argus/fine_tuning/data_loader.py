"""Data loading utilities for model training and evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Generator, Callable, Any
from dataclasses import dataclass
from enum import Enum


class DatasetFormat(Enum):
    """Supported dataset formats."""
    YOLO = "yolo"
    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    CUSTOM = "custom"


@dataclass
class DatasetSample:
    """Single dataset sample."""
    image_path: Path
    annotations: list[dict]
    metadata: dict[str, Any]


@dataclass
class DatasetInfo:
    """Dataset information."""
    name: str
    format: DatasetFormat
    total_samples: int
    classes: list[str]
    train_val_test_split: dict[str, int]
    statistics: dict[str, Any]


class DataLoader:
    """Load and manage datasets for training and evaluation."""

    def __init__(self, dataset_path: str | Path, format: DatasetFormat = DatasetFormat.YOLO):
        """Initialize data loader.
        
        Args:
            dataset_path: Path to dataset
            format: Dataset format
        """
        self.dataset_path = Path(dataset_path)
        self.format = format
        self.classes: list[str] = []
        self.dataset_info: DatasetInfo | None = None

    def load_dataset(self) -> DatasetInfo:
        """Load dataset and return information.
        
        Returns:
            DatasetInfo with dataset details
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Auto-detect format if not specified
        if self.format == DatasetFormat.CUSTOM:
            self.format = self._detect_format()
        
        # Load based on format
        if self.format == DatasetFormat.YOLO:
            self._load_yolo_dataset()
        elif self.format == DatasetFormat.COCO:
            self._load_coco_dataset()
        elif self.format == DatasetFormat.PASCAL_VOC:
            self._load_pascal_voc_dataset()
        else:
            raise ValueError(f"Unsupported dataset format: {self.format}")
        
        # Calculate dataset statistics
        self.dataset_info = self._calculate_dataset_info()
        return self.dataset_info

    def _detect_format(self) -> DatasetFormat:
        """Auto-detect dataset format from directory structure.
        
        Returns:
            Detected dataset format
        """
        # Check for YOLO format
        if (self.dataset_path / "data.yaml").exists() or (self.dataset_path / "yaml").exists():
            return DatasetFormat.YOLO
        
        # Check for COCO format
        if any(self.dataset_path.rglob("*.json")):
            for json_file in self.dataset_path.rglob("*.json"):
                try:
                    with json_file.open("r") as f:
                        data = json.load(f)
                        if "images" in data and "annotations" in data:
                            return DatasetFormat.COCO
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Check for Pascal VOC format
        if any(self.dataset_path.rglob("*.xml")):
            return DatasetFormat.PASCAL_VOC
        
        return DatasetFormat.CUSTOM

    def _load_yolo_dataset(self) -> None:
        """Load YOLO format dataset."""
        # Load classes from data.yaml or classes.txt
        yaml_path = self.dataset_path / "data.yaml"
        classes_path = self.dataset_path / "classes.txt"
        
        if yaml_path.exists():
            try:
                import yaml
                with yaml_path.open("r") as f:
                    data = yaml.safe_load(f)
                    self.classes = data.get("names", [])
            except ImportError:
                # If yaml not available, try classes.txt
                pass
        
        if not self.classes and classes_path.exists():
            with classes_path.open("r") as f:
                self.classes = [line.strip() for line in f if line.strip()]

    def _load_coco_dataset(self) -> None:
        """Load COCO format dataset."""
        # Find annotations JSON
        for json_file in self.dataset_path.rglob("*.json"):
            try:
                with json_file.open("r") as f:
                    data = json.load(f)
                    if "categories" in data:
                        self.classes = [cat["name"] for cat in data["categories"]]
                        break
            except (json.JSONDecodeError, KeyError):
                continue

    def _load_pascal_voc_dataset(self) -> None:
        """Load Pascal VOC format dataset."""
        # Pascal VOC doesn't have a central class file
        # Classes are typically defined in the annotation files
        # This is a simplified approach
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                       "bus", "car", "cat", "chair", "cow",
                       "diningtable", "dog", "horse", "motorbike", "person",
                       "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def _calculate_dataset_info(self) -> DatasetInfo:
        """Calculate dataset information.
        
        Returns:
            DatasetInfo object
        """
        # Count samples in each split
        splits = {}
        for split in ["train", "val", "test"]:
            split_path = self.dataset_path / split
            if split_path.exists():
                # Count images
                image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
                splits[split] = len([p for p in split_path.rglob("*") if p.suffix.lower() in image_extensions])
            else:
                splits[split] = 0
        
        total_samples = sum(splits.values())
        
        return DatasetInfo(
            name=self.dataset_path.name,
            format=self.format,
            total_samples=total_samples,
            classes=self.classes,
            train_val_test_split=splits,
            statistics={
                "num_classes": len(self.classes),
                "class_distribution": self._calculate_class_distribution()
            }
        )

    def _calculate_class_distribution(self) -> dict[str, int]:
        """Calculate class distribution across dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {cls: 0 for cls in self.classes}
        
        # Count annotations based on format
        if self.format == DatasetFormat.YOLO:
            for txt_file in self.dataset_path.rglob("*.txt"):
                if txt_file.name == "classes.txt":
                    continue
                try:
                    with txt_file.open("r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id < len(self.classes):
                                    distribution[self.classes[class_id]] += 1
                except (ValueError, IndexError):
                    continue
        
        return distribution

    def get_train_samples(self) -> Generator[DatasetSample, None, None]:
        """Get training samples.
        
        Yields:
            DatasetSample objects for training
        """
        train_path = self.dataset_path / "train"
        if not train_path.exists():
            return
        
        for sample in self._get_samples_from_split(train_path):
            yield sample

    def get_val_samples(self) -> Generator[DatasetSample, None, None]:
        """Get validation samples.
        
        Yields:
            DatasetSample objects for validation
        """
        val_path = self.dataset_path / "val"
        if not val_path.exists():
            return
        
        for sample in self._get_samples_from_split(val_path):
            yield sample

    def get_test_samples(self) -> Generator[DatasetSample, None, None]:
        """Get test samples.
        
        Yields:
            DatasetSample objects for testing
        """
        test_path = self.dataset_path / "test"
        if not test_path.exists():
            return
        
        for sample in self._get_samples_from_split(test_path):
            yield sample

    def _get_samples_from_split(self, split_path: Path) -> Generator[DatasetSample, None, None]:
        """Get samples from a specific split directory.
        
        Args:
            split_path: Path to split directory
            
        Yields:
            DatasetSample objects
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        for image_path in split_path.rglob("*"):
            if image_path.suffix.lower() not in image_extensions:
                continue
            
            # Load annotations based on format
            annotations = self._load_annotations_for_image(image_path)
            
            yield DatasetSample(
                image_path=image_path,
                annotations=annotations,
                metadata={
                    "split": split_path.name,
                    "image_size": self._get_image_size(image_path)
                }
            )

    def _load_annotations_for_image(self, image_path: Path) -> list[dict]:
        """Load annotations for a specific image.
        
        Args:
            image_path: Path to image
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        if self.format == DatasetFormat.YOLO:
            # YOLO format: .txt file with same name as image
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists():
                with txt_path.open("r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = parts[:5]
                            annotations.append({
                                "class_id": int(class_id),
                                "bbox": [float(x_center), float(y_center), float(width), float(height)],
                                "confidence": 1.0
                            })
        
        # Add other format implementations as needed
        
        return annotations

    def _get_image_size(self, image_path: Path) -> tuple[int, int]:
        """Get image dimensions.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (width, height)
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except ImportError:
            # Fallback to opencv if PIL not available
            try:
                import cv2
                img = cv2.imread(str(image_path))
                if img is not None:
                    return (img.shape[1], img.shape[0])  # (width, height)
            except ImportError:
                pass
        
        return (0, 0)

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                      test_ratio: float = 0.1, shuffle: bool = True) -> dict[str, list[Path]]:
        """Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Dictionary mapping split names to lists of image paths
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [p for p in self.dataset_path.rglob("*") if p.suffix.lower() in image_extensions]
        
        if shuffle:
            random.shuffle(all_images)
        
        # Calculate split points
        total = len(all_images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        return {
            "train": all_images[:train_end],
            "val": all_images[train_end:val_end],
            "test": all_images[val_end:]
        }

    def create_yolo_structure(self, output_path: str | Path) -> None:
        """Create YOLO-compatible directory structure.
        
        Args:
            output_path: Path where to create the structure
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard YOLO directories
        for split in ["train", "val", "test"]:
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Create classes file
        if self.classes:
            classes_file = output_path / "classes.txt"
            with classes_file.open("w") as f:
                for cls in self.classes:
                    f.write(f"{cls}\n")
        
        # Create data.yaml
        yaml_content = f"""
path: {output_path}
train: train/images
val: val/images
test: test/images

nc: {len(self.classes)}
names: {self.classes}
"""
        
        yaml_file = output_path / "data.yaml"
        with yaml_file.open("w") as f:
            f.write(yaml_content.strip())