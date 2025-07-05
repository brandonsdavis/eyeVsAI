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
sys.path.append(str(Path(__file__).parent.parent.parent / "ml_models_core" / "src"))

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from base_trainer import BaseTrainer
from .models import TransferLearningModel
from .data_loader import create_memory_efficient_datasets
from .config import TransferLearningClassifierConfig


class TransferLearningTrainer(BaseTrainer):
    """Trainer for transfer learning models using TensorFlow/Keras."""
    
    def __init__(self, model, config: TransferLearningClassifierConfig):
        super().__init__(model, config.to_dict())
        self.config = config
        self.tf_model = None
        self.class_names = None
        self.class_weights = None
        
        # Training history
        self.training_history = None
        self.fine_tune_history = None
        
    def train(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Train the transfer learning model with two-phase training.
        
        Args:
            data_path: Path to the training data directory
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        self.logger.info(f"Starting transfer learning training from {data_path}")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset, class_names, class_weights = create_memory_efficient_datasets(
            data_path, self.config
        )
        
        self.class_names = class_names
        self.class_weights = class_weights
        num_classes = len(class_names)
        
        # Update model with discovered classes
        if hasattr(self.model, 'class_names'):
            self.model.class_names = class_names
            self.model.num_classes = num_classes
        
        # Create TensorFlow model
        self.tf_model = TransferLearningModel(self.config, num_classes)
        model = self.tf_model.build_model()
        
        # Phase 1: Train with frozen base model
        self.logger.info("Phase 1: Training with frozen base model...")
        phase1_epochs = max(1, self.config.epochs // 3)
        
        phase1_callbacks = self.tf_model.get_callbacks(log_dir=f"{self.config.log_dir}/phase1")
        
        # Train phase 1
        self.training_history = model.fit(
            train_dataset,
            epochs=phase1_epochs,
            validation_data=val_dataset,
            callbacks=phase1_callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        if self.config.fine_tune_layers > 0:
            self.logger.info("Phase 2: Fine-tuning with unfrozen layers...")
            
            # Prepare for fine-tuning
            self.tf_model.prepare_for_fine_tuning()
            
            # Fine-tuning callbacks
            phase2_callbacks = self.tf_model.get_callbacks(log_dir=f"{self.config.log_dir}/phase2")
            
            # Continue training with fine-tuning
            phase2_epochs = self.config.epochs - phase1_epochs
            
            self.fine_tune_history = model.fit(
                train_dataset,
                epochs=phase1_epochs + phase2_epochs,
                initial_epoch=phase1_epochs,
                validation_data=val_dataset,
                callbacks=phase2_callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        # Evaluate on test set
        test_results = self._evaluate_model(model, test_dataset)
        
        # Calculate dataset sizes
        train_size = tf.data.experimental.cardinality(train_dataset).numpy() * self.config.batch_size
        val_size = tf.data.experimental.cardinality(val_dataset).numpy() * self.config.batch_size
        test_size = tf.data.experimental.cardinality(test_dataset).numpy() * self.config.batch_size
        
        # Collect metrics
        metrics = {
            'train_samples': int(train_size),
            'val_samples': int(val_size),
            'test_samples': int(test_size),
            'test_accuracy': test_results['accuracy'],
            'test_loss': test_results['loss'],
            'num_classes': num_classes,
            'model_parameters': model.count_params(),
            'trainable_parameters': sum(tf.size(p) for p in model.trainable_variables),
            'base_model': self.config.base_model_name,
            'fine_tuned': self.config.fine_tune_layers > 0
        }
        
        # Get best validation accuracy
        if self.training_history:
            metrics['best_val_accuracy_phase1'] = max(self.training_history.history['val_accuracy'])
        
        if self.fine_tune_history:
            metrics['best_val_accuracy_phase2'] = max(self.fine_tune_history.history['val_accuracy'])
            metrics['best_val_accuracy'] = metrics['best_val_accuracy_phase2']
        else:
            metrics['best_val_accuracy'] = metrics.get('best_val_accuracy_phase1', 0.0)
        
        self.log_metrics(metrics, phase="final")
        
        # Store the trained model in our wrapper
        if hasattr(self.model, 'model'):
            self.model.model = model
        else:
            self.model = model
        
        return {
            'model': model,
            'metrics': metrics,
            'class_names': class_names,
            'class_weights': class_weights,
            'training_history': self.training_history.history if self.training_history else None,
            'fine_tune_history': self.fine_tune_history.history if self.fine_tune_history else None,
            'test_results': test_results
        }
    
    def _evaluate_model(self, model: keras.Model, test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Evaluate the model on test dataset."""
        self.logger.info("Evaluating model on test dataset...")
        
        # Basic evaluation
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        
        # Detailed predictions for classification report
        predictions = []
        true_labels = []
        
        for batch_images, batch_labels in test_dataset:
            batch_predictions = model.predict(batch_images, verbose=0)
            predictions.extend(np.argmax(batch_predictions, axis=1))
            true_labels.extend(batch_labels.numpy())
        
        # Classification report
        if len(set(true_labels)) > 1:  # Only if we have multiple classes
            report = classification_report(
                true_labels, 
                predictions, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
        else:
            report = {}
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Test Loss: {test_loss:.4f}")
        self.logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'predictions': predictions,
            'true_labels': true_labels,
            'classification_report': report
        }
    
    def evaluate(self, X_test: tf.data.Dataset, y_test: Optional[tf.Tensor] = None) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.tf_model is None or self.tf_model.model is None:
            raise ValueError("Model not trained yet")
        
        if isinstance(X_test, tf.data.Dataset):
            # Evaluate on dataset
            loss, accuracy = self.tf_model.model.evaluate(X_test, verbose=0)
            return {
                'test_accuracy': float(accuracy),
                'test_loss': float(loss)
            }
        else:
            # Evaluate on tensors
            if y_test is None:
                raise ValueError("y_test required when X_test is not a dataset")
            
            loss, accuracy = self.tf_model.model.evaluate(X_test, y_test, verbose=0)
            return {
                'test_accuracy': float(accuracy),
                'test_loss': float(loss)
            }
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        if self.tf_model is None or self.tf_model.model is None:
            raise ValueError("No model to save")
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path.parent / f"model_epoch_{epoch}.h5"
        self.tf_model.model.save(model_path)
        
        # Save metadata
        checkpoint_data = {
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config.to_dict(),
            'class_names': self.class_names,
            'class_weights': self.class_weights,
            'model_path': str(model_path)
        }
        
        if self.training_history:
            checkpoint_data['training_history'] = self.training_history.history
        
        if self.fine_tune_history:
            checkpoint_data['fine_tune_history'] = self.fine_tune_history.history
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Load model
        model_path = checkpoint_data.get('model_path')
        if model_path and Path(model_path).exists():
            model = keras.models.load_model(model_path)
            
            # Recreate TensorFlow model wrapper
            if 'class_names' in checkpoint_data:
                num_classes = len(checkpoint_data['class_names'])
                self.tf_model = TransferLearningModel(self.config, num_classes)
                self.tf_model.model = model
                
                self.class_names = checkpoint_data['class_names']
                self.class_weights = checkpoint_data.get('class_weights')
                
                # Update wrapper model
                if hasattr(self.model, 'model'):
                    self.model.model = model
                    self.model.class_names = self.class_names
                    self.model.num_classes = num_classes
                else:
                    self.model = model
        
        return checkpoint_data
    
    def save_training_history(self, history_path: str) -> None:
        """Save training history to file."""
        history_data = {}
        
        if self.training_history:
            history_data['phase1'] = self.training_history.history
        
        if self.fine_tune_history:
            history_data['phase2'] = self.fine_tune_history.history
        
        history_data['config'] = self.config.to_dict()
        history_data['class_names'] = self.class_names
        
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training history."""
        if not self.training_history and not self.fine_tune_history:
            self.logger.warning("No training history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine histories
        all_history = {}
        if self.training_history:
            for key, values in self.training_history.history.items():
                all_history[key] = values
        
        if self.fine_tune_history:
            for key, values in self.fine_tune_history.history.items():
                if key in all_history:
                    all_history[key].extend(values)
                else:
                    all_history[key] = values
        
        epochs = range(1, len(all_history['loss']) + 1)
        
        # Training & Validation Loss
        ax1.plot(epochs, all_history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in all_history:
            ax1.plot(epochs, all_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Training & Validation Accuracy
        ax2.plot(epochs, all_history['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in all_history:
            ax2.plot(epochs, all_history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # Learning Rate (if available)
        if 'lr' in all_history:
            ax3.plot(epochs, all_history['lr'], 'g-', label='Learning Rate')
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
        else:
            ax3.axis('off')
        
        # Mark phase transition
        if self.training_history and self.fine_tune_history:
            phase1_epochs = len(self.training_history.history['loss'])
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=phase1_epochs, color='orange', linestyle='--', 
                          label='Fine-tuning Start', alpha=0.7)
                if ax != ax3 or 'lr' in all_history:
                    ax.legend()
        
        # Additional metrics
        if len(all_history) > 4:
            other_metrics = [k for k in all_history.keys() 
                           if k not in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'lr']]
            if other_metrics:
                metric = other_metrics[0]
                ax4.plot(epochs, all_history[metric], 'purple', label=metric)
                ax4.set_title(f'Additional Metric: {metric}')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel(metric)
                ax4.legend()
            else:
                ax4.axis('off')
        else:
            ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()