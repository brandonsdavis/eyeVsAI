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

from src.classifier import TransferLearningClassifier
from src.trainer import TransferLearningTrainer
from src.config import TransferLearningClassifierConfig


def main():
    parser = argparse.ArgumentParser(description='Train Transfer Learning image classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--model_save_path', type=str, default='models/transfer_learning_classifier.h5',
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
                       choices=['resnet50', 'vgg16', 'efficientnet_b0', 'efficientnet_b1', 'mobilenet_v2', 'inception_v3'],
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
    parser.add_argument('--use_xla', action='store_true',
                       help='Enable XLA compilation')
    parser.add_argument('--class_weights', action='store_true',
                       help='Use class weights for imbalanced datasets')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory for logs')
    parser.add_argument('--cache_dataset', action='store_true',
                       help='Cache dataset for faster training')
    
    args = parser.parse_args()
    
    # Setup logging
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    print(f"Starting Transfer Learning training...")
    print(f"Base model: {args.base_model}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"XLA compilation: {args.use_xla}")
    
    # Create configuration with minimal essential parameters
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
        use_xla=args.use_xla,
        class_weights=args.class_weights,
        cache_dataset=args.cache_dataset,
        model_save_path=args.model_save_path,
        log_dir=args.log_dir
    )
    
    # Create model and trainer
    model = TransferLearningClassifier(config=config)
    trainer = TransferLearningTrainer(model, config)
    
    # Train model
    logging.info("Starting transfer learning training...")
    logging.info(f"Configuration: {config}")
    
    results = trainer.train(args.data_path)
    
    # Save model
    logging.info(f"Saving model to {args.model_save_path}")
    model.save_model(
        args.model_save_path,
        model=results['model'],
        class_names=results['class_names'],
        accuracy=results['metrics']['test_accuracy'],
        training_history=results['training_history']
    )
    
    # Save training history
    history_path = f"{args.log_dir}/training_history.json"
    trainer.save_training_history(history_path)
    
    # Plot training history
    try:
        plot_path = f"{args.log_dir}/training_history.png"
        trainer.plot_training_history(save_path=plot_path)
    except Exception as e:
        logging.warning(f"Could not save training plot: {e}")
    
    logging.info(f"Transfer learning training completed successfully!")
    logging.info(f"Test accuracy: {results['metrics']['test_accuracy']:.2f}")
    logging.info(f"Best validation accuracy: {results['metrics']['best_val_accuracy']:.2f}")
    logging.info(f"Model parameters: {results['metrics']['model_parameters']:,}")
    logging.info(f"Classes trained: {results['metrics']['num_classes']}")
    logging.info(f"Base model: {results['metrics']['base_model']}")
    logging.info(f"Fine-tuned: {results['metrics']['fine_tuned']}")


if __name__ == "__main__":
    main()