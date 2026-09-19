#!/usr/bin/env python3
"""Model training and fine-tuning utilities for ARGUS."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from argus.fine_tuning import ModelTrainer, YOLOTrainer, TrainingConfig, DataLoader, DatasetFormat


def prepare_dataset(dataset_path: str, output_path: str | None = None):
    """Prepare dataset for training.
    
    Args:
        dataset_path: Path to raw dataset
        output_path: Path to save prepared dataset structure
    """
    print(f"Preparing dataset from: {dataset_path}")
    
    loader = DataLoader(dataset_path)
    dataset_info = loader.load_dataset()
    
    print(f"  Dataset name: {dataset_info.name}")
    print(f"  Format: {dataset_info.format}")
    print(f"  Total samples: {dataset_info.total_samples}")
    print(f"  Classes: {dataset_info.classes}")
    print(f"  Split: {dataset_info.train_val_test_split}")
    print(f"  Statistics: {dataset_info.statistics}")
    
    if output_path:
        print(f"\nCreating YOLO structure at: {output_path}")
        loader.create_yolo_structure(output_path)
        print("  YOLO structure created successfully!")
    
    return dataset_info


def train_yolo_model(config: TrainingConfig):
    """Train YOLO model with given configuration.
    
    Args:
        config: Training configuration
    """
    print(f"Training YOLO model: {config.model_name}")
    print(f"  Base model: {config.base_model}")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    trainer = YOLOTrainer(config)
    result = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"  Model saved to: {result.model_path}")
    print(f"  Training time: {result.training_time:.2f}s")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Best loss: {result.best_loss:.4f}")
    print(f"  Epochs completed: {result.epochs_completed}")
    print(f"  Metrics: {result.metrics}")
    
    return result


def train_custom_model(config: TrainingConfig):
    """Train custom model with given configuration.
    
    Args:
        config: Training configuration
    """
    print(f"Training custom model: {config.model_name}")
    print(f"  Base model: {config.base_model}")
    print(f"  Dataset: {config.dataset_path}")
    
    trainer = ModelTrainer(config)
    result = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"  Model saved to: {result.model_path}")
    print(f"  Training time: {result.training_time:.2f}s")
    
    return result


def evaluate_model(model_path: str, test_data_path: str):
    """Evaluate trained model on test dataset.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test dataset
    """
    print(f"Evaluating model: {model_path}")
    print(f"  Test dataset: {test_data_path}")
    
    from argus.fine_tuning import ModelEvaluator
    from argus.detectors import YOLODetector
    
    # Create detector with trained model
    detector = YOLODetector(model_path=model_path)
    
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(detector, test_data_path)
    
    print(f"\nEvaluation completed!")
    print(f"  Total images: {result.total_images}")
    print(f"  Total detections: {result.total_detections}")
    print(f"  Precision: {result.metrics.precision:.4f}")
    print(f"  Recall: {result.metrics.recall:.4f}")
    print(f"  F1 Score: {result.metrics.f1_score:.4f}")
    print(f"  Inference time: {result.metrics.inference_time_ms:.2f}ms")
    
    return result


def main():
    """Main training CLI."""
    parser = argparse.ArgumentParser(description="ARGUS Model Training Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset for training")
    prepare_parser.add_argument("dataset_path", help="Path to raw dataset")
    prepare_parser.add_argument("--output", help="Output path for prepared dataset")
    
    # Train YOLO command
    yolo_parser = subparsers.add_parser("train-yolo", help="Train YOLO model")
    yolo_parser.add_argument("--model-name", default="weapons_detector", help="Model name")
    yolo_parser.add_argument("--base-model", default="yolo11n.pt", help="Base YOLO model")
    yolo_parser.add_argument("--dataset", required=True, help="Path to dataset (data.yaml)")
    yolo_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    yolo_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    yolo_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    yolo_parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    # Train custom model command
    custom_parser = subparsers.add_parser("train-custom", help="Train custom model")
    custom_parser.add_argument("--model-name", required=True, help="Model name")
    custom_parser.add_argument("--base-model", required=True, help="Base model path")
    custom_parser.add_argument("--dataset", required=True, help="Path to dataset")
    custom_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    custom_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    # Evaluate model command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("model_path", help="Path to trained model")
    eval_parser.add_argument("test_data", help="Path to test dataset")
    
    args = parser.parse_args()
    
    try:
        if args.command == "prepare":
            prepare_dataset(args.dataset_path, args.output)
        
        elif args.command == "train-yolo":
            config = TrainingConfig(
                model_name=args.model_name,
                base_model=args.base_model,
                dataset_path=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device
            )
            train_yolo_model(config)
        
        elif args.command == "train-custom":
            config = TrainingConfig(
                model_name=args.model_name,
                base_model=args.base_model,
                dataset_path=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            train_custom_model(config)
        
        elif args.command == "evaluate":
            evaluate_model(args.model_path, args.test_data)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()