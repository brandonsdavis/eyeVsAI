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
import torch
from datetime import datetime

from src.classifier import DeepLearningV1Classifier
from src.trainer import DeepLearningV1Trainer
from src.config import DeepLearningV1Config


def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning v1 image classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--model_save_path', type=str, default='models/deep_v1_classifier.pth',
                       help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 128],
                       help='Input image size (height width)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory for logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
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
    
    # Create configuration
    config = DeepLearningV1Config(
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        dropout_rate=args.dropout_rate,
        model_save_path=args.model_save_path,
        log_dir=args.log_dir
    )
    
    # Create model and trainer
    model = DeepLearningV1Classifier(config=config)
    trainer = DeepLearningV1Trainer(model, config)
    
    # Train model
    logging.info("Starting training...")
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
    trainer.save_training_history(f"{args.log_dir}/training_history.json")
    
    logging.info(f"Training completed. Test accuracy: {results['metrics']['test_accuracy']:.2f}%")
    logging.info(f"Best validation accuracy: {results['metrics']['best_val_accuracy']:.2f}%")
    logging.info(f"Training time: {results['metrics']['training_time']:.2f} seconds")


if __name__ == "__main__":
    main()