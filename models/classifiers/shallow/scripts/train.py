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
from datetime import datetime

from src.classifier import ShallowImageClassifier
from src.trainer import ShallowLearningTrainer  
from src.config import ShallowLearningConfig


def main():
    parser = argparse.ArgumentParser(description='Train shallow learning image classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--model_save_path', type=str, default='models/shallow_classifier.pkl',
                       help='Path to save trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (not used for shallow learning)')
    parser.add_argument('--n_components_svd', type=int, default=100,
                       help='Number of SVD components for dimensionality reduction')
    parser.add_argument('--use_pca', action='store_true',
                       help='Use PCA instead of SVD')
    parser.add_argument('--n_components_pca', type=int, default=50,
                       help='Number of PCA components')
    parser.add_argument('--svm_C', type=float, default=1.0,
                       help='SVM regularization parameter')
    parser.add_argument('--log_dir', type=str, default='logs/',
                       help='Directory for logs')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{args.log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create configuration
    config = ShallowLearningConfig(
        batch_size=args.batch_size,
        n_components_svd=args.n_components_svd,
        use_pca=args.use_pca,
        n_components_pca=args.n_components_pca,
        svm_C=args.svm_C,
        model_save_path=args.model_save_path,
        log_dir=args.log_dir
    )
    
    # Create model and trainer
    model = ShallowImageClassifier(config=config)
    trainer = ShallowLearningTrainer(model, config)
    
    # Train model
    logging.info("Starting training...")
    results = trainer.train(args.data_path)
    
    # Save model
    logging.info(f"Saving model to {args.model_save_path}")
    model.save_model(args.model_save_path)
    
    # Save training history
    trainer.save_training_history(f"{args.log_dir}/training_history.json")
    
    logging.info(f"Training completed. Final validation accuracy: {results['metrics']['val_accuracy']:.4f}")
    

if __name__ == "__main__":
    main()