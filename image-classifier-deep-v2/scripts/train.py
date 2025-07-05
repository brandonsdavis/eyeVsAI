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

from src.classifier import DeepLearningV2Classifier
from src.trainer import MemoryEfficientTrainingManager
from src.config import DeepLearningV2Config


def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning v2 image classifier with advanced features')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--model_save_path', type=str, default='models/deep_v2_classifier.pth',
                       help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience')
    parser.add_argument('--image_size', type=int, nargs=2, default=[96, 96],
                       help='Input image size (height width)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=0.3,
                       help='Probability of applying mixup')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                       help='Label smoothing factor')
    parser.add_argument('--attention_reduction', type=int, default=8,
                       help='Channel attention reduction ratio')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory for logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--memory_efficient', action='store_true',
                       help='Enable memory efficient training')
    
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
    config = DeepLearningV2Config(
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        label_smoothing=args.label_smoothing,
        attention_reduction_ratio=args.attention_reduction,
        memory_efficient=args.memory_efficient,
        model_save_path=args.model_save_path,
        log_dir=args.log_dir
    )
    
    # Create model and trainer
    model = DeepLearningV2Classifier(config=config)
    trainer = MemoryEfficientTrainingManager(model, config)
    
    # Train model
    logging.info("Starting advanced training with memory efficiency...")
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
    trainer.save_training_history(f"{args.log_dir}/training_history.json")
    
    logging.info(f"Advanced training completed successfully!")
    logging.info(f"Test accuracy: {results['metrics']['test_accuracy']:.2f}%")
    logging.info(f"Best validation accuracy: {results['metrics']['best_val_accuracy']:.2f}%")
    logging.info(f"Model parameters: {results['metrics']['model_parameters']:,}")
    logging.info(f"Classes trained: {results['metrics']['num_classes']}")


if __name__ == "__main__":
    main()