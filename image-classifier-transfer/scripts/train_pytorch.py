#!/usr/bin/env python
# Copyright 2025 Brandon Davis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
import json
from datetime import datetime
import torch

from src.config import TransferLearningClassifierConfig
from src.trainer_pytorch import TransferLearningTrainerPyTorch


def main():
    parser = argparse.ArgumentParser(description='Train PyTorch Transfer Learning image classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--model_save_path', type=str, default='models/transfer_learning_pytorch.pth',
                       help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Input image size (height width)')
    parser.add_argument('--base_model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 
                               'densenet121', 'densenet169', 'efficientnet_b0', 'efficientnet_b1',
                               'efficientnet_b2', 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
                       help='Base pre-trained model to use')
    parser.add_argument('--fine_tune_layers', type=int, default=10,
                       help='Number of layers to fine-tune (0 to disable fine-tuning)')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning phase')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--dense_units', type=int, nargs='+', default=[512, 256],
                       help='Dense layer units (e.g., --dense_units 512 256)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--class_weights', action='store_true',
                       help='Use class weights for imbalanced datasets')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory for logs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup logging
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/training_pytorch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"PyTorch Transfer Learning Training")
    print(f"==================================")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Base model: {args.base_model}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print()
    
    # Create configuration
    config = TransferLearningClassifierConfig(
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.num_epochs,
        early_stopping_patience=args.patience,
        base_model_name=args.base_model,
        fine_tune_layers=args.fine_tune_layers,
        fine_tune_learning_rate=args.fine_tune_lr,
        validation_split=args.validation_split,
        mixed_precision=args.mixed_precision,
        class_weights=args.class_weights,
        model_save_path=args.model_save_path,
        log_dir=args.log_dir,
        head_dropout_rate=args.dropout_rate,
        dense_units=args.dense_units,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        num_workers=args.num_workers
    )
    
    # Create trainer
    trainer = TransferLearningTrainerPyTorch(config)
    
    # Train model
    logging.info("Starting PyTorch transfer learning training...")
    logging.info(f"Configuration: {config}")
    
    results = trainer.train(args.data_path)
    
    # Save model
    logging.info(f"Saving model to {args.model_save_path}")
    trainer.save_model(args.model_save_path)
    
    # Save training history
    history_path = f"{args.log_dir}/training_history_pytorch.json"
    with open(history_path, 'w') as f:
        json.dump(results['training_history'], f, indent=2)
    
    # Plot training history
    try:
        plot_path = f"{args.log_dir}/training_history_pytorch.png"
        trainer.plot_training_history(save_path=plot_path)
    except Exception as e:
        logging.warning(f"Could not save training plot: {e}")
    
    # Print final results
    print("\nTraining Complete!")
    print("==================")
    print(f"Test Accuracy: {results['metrics']['test_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['metrics']['top5_accuracy']:.4f}")
    print(f"Best Val Accuracy: {results['metrics']['best_val_accuracy']:.4f}")
    print(f"Model Parameters: {results['metrics']['model_parameters']:,}")
    print(f"Trainable Parameters: {results['metrics']['trainable_parameters']:,}")
    print(f"Classes: {results['metrics']['num_classes']}")
    print(f"Device: {results['metrics']['device']}")
    print(f"Base Model: {results['metrics']['base_model']}")
    print(f"Fine-tuned: {results['metrics']['fine_tuned']}")


if __name__ == "__main__":
    main()