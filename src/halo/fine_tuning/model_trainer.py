"""Model training utilities for custom detection models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    base_model: str
    dataset_path: str
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: tuple[int, int] = (640, 640)
    device: str = "cuda"  # or "cpu"
    augmentation: bool = True
    early_stopping_patience: int = 10
    save_best_only: bool = True


@dataclass
class TrainingResult:
    """Result of model training."""
    model_path: str
    training_time: float
    final_loss: float
    best_loss: float
    epochs_completed: int
    metrics: dict[str, float]
    artifacts: dict[str, str]


class ModelTrainer:
    """Base trainer class for custom detection models.
    
    Provides framework-agnostic training utilities and can be extended
    for specific frameworks (PyTorch, TensorFlow, etc.).
    """

    def __init__(self, config: TrainingConfig, output_dir: str | Path = "runs/training"):
        """Initialize model trainer.
        
        Args:
            config: Training configuration
            output_dir: Directory to save training outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{config.model_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_history: list[dict] = []
        self.callbacks: list[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        """Add a training callback.
        
        Args:
            callback: Function to call during training (on_epoch_end, etc.)
        """
        self.callbacks.append(callback)

    def prepare_dataset(self) -> dict[str, Any]:
        """Prepare dataset for training.
        
        Returns:
            Dictionary with dataset information
        """
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Basic dataset validation
        dataset_info = {
            "path": str(dataset_path),
            "exists": True,
            "size": sum(1 for _ in dataset_path.rglob("*") if _.is_file()),
            "structure": self._analyze_dataset_structure(dataset_path)
        }
        
        return dataset_info

    def _analyze_dataset_structure(self, dataset_path: Path) -> dict:
        """Analyze dataset structure.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            "has_train": (dataset_path / "train").exists(),
            "has_val": (dataset_path / "val").exists(),
            "has_test": (dataset_path / "test").exists(),
            "has_annotations": any(
                dataset_path.rglob("*.xml") or 
                dataset_path.rglob("*.json") or 
                dataset_path.rglob("*.txt")
            )
        }
        return structure

    def train(self) -> TrainingResult:
        """Train the model (base implementation).
        
        This is a template method - override in framework-specific subclasses.
        
        Returns:
            TrainingResult with training outcomes
        """
        import time
        start_time = time.time()
        
        # Validate dataset
        dataset_info = self.prepare_dataset()
        self._save_config()
        self._save_dataset_info(dataset_info)
        
        # Placeholder for actual training - override in subclasses
        # In a real implementation, this would:
        # 1. Load the base model
        # 2. Prepare data loaders
        # 3. Set up optimizers and schedulers
        # 4. Run training loop
        # 5. Save checkpoints and metrics
        
        training_time = time.time() - start_time
        
        # Placeholder result
        result = TrainingResult(
            model_path=str(self.experiment_dir / "best_model.pt"),
            training_time=training_time,
            final_loss=0.0,
            best_loss=0.0,
            epochs_completed=0,
            metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0},
            artifacts={"config": str(self.experiment_dir / "config.json")}
        )
        
        self._save_training_result(result)
        return result

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.experiment_dir / "config.json"
        with config_path.open("w") as f:
            json.dump({
                "model_name": self.config.model_name,
                "base_model": self.config.base_model,
                "dataset_path": self.config.dataset_path,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "image_size": self.config.image_size,
                "device": self.config.device,
                "augmentation": self.config.augmentation,
                "early_stopping_patience": self.config.early_stopping_patience
            }, f, indent=2)

    def _save_dataset_info(self, dataset_info: dict) -> None:
        """Save dataset information."""
        dataset_path = self.experiment_dir / "dataset_info.json"
        with dataset_path.open("w") as f:
            json.dump(dataset_info, f, indent=2)

    def _save_training_result(self, result: TrainingResult) -> None:
        """Save training result."""
        result_path = self.experiment_dir / "training_result.json"
        with result_path.open("w") as f:
            json.dump({
                "model_path": result.model_path,
                "training_time": result.training_time,
                "final_loss": result.final_loss,
                "best_loss": result.best_loss,
                "epochs_completed": result.epochs_completed,
                "metrics": result.metrics,
                "artifacts": result.artifacts
            }, f, indent=2)

    def log_epoch(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
        """Log training epoch information.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            metrics: Dictionary of metrics
        """
        epoch_log = {
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.training_history.append(epoch_log)
        
        # Call callbacks
        for callback in self.callbacks:
            callback(epoch_log)

    def save_checkpoint(self, epoch: int, model_state: dict, is_best: bool = False) -> Path:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            model_state: Model state dictionary
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_best or not self.config.save_best_only:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            # In real implementation, save actual model state
            # torch.save(model_state, checkpoint_path)
            
            if is_best:
                best_path = checkpoint_dir / "best_model.pt"
                # shutil.copy(checkpoint_path, best_path)
        
        return checkpoint_dir / f"epoch_{epoch}.pt"


class YOLOTrainer(ModelTrainer):
    """YOLO-specific trainer using ultralytics framework."""

    def train(self) -> TrainingResult:
        """Train YOLO model.
        
        Returns:
            TrainingResult with training outcomes
        """
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("YOLO training requires ultralytics package") from error
        
        import time
        start_time = time.time()
        
        # Load base model
        model = YOLO(self.config.base_model)
        
        # Train the model
        results = model.train(
            data=self.config.dataset_path,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            imgsz=self.config.image_size[0],
            lr0=self.config.learning_rate,
            device=self.config.device,
            project=str(self.output_dir),
            name=self.config.model_name,
            exist_ok=True,
            augment=self.config.augmentation,
            patience=self.config.early_stopping_patience
        )
        
        training_time = time.time() - start_time
        
        # Extract metrics from results
        metrics = {
            "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
            "mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
            "recall": float(results.results_dict.get('metrics/recall(B)', 0))
        }
        
        result = TrainingResult(
            model_path=str(results.save_dir / "weights" / "best.pt"),
            training_time=training_time,
            final_loss=float(results.results_dict.get('train/loss', 0)),
            best_loss=float(results.results_dict.get('train/loss', 0)),
            epochs_completed=self.config.epochs,
            metrics=metrics,
            artifacts={
                "config": str(self.experiment_dir / "config.json"),
                "results_dir": str(results.save_dir)
            }
        )
        
        self._save_training_result(result)
        return result