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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import gc
from typing import Dict, Any, Tuple, List
from sklearn.metrics import classification_report

from base_trainer import BaseTrainer
from .model import DeepLearningV2
from .data_loader import create_memory_efficient_loaders, mixup_data, mixup_criterion
from .config import DeepLearningV2Config


class MemoryEfficientTrainingManager(BaseTrainer):
    """Memory-efficient training manager with gradient accumulation and advanced techniques."""
    
    def __init__(self, model, config: DeepLearningV2Config):
        super().__init__(model, config.to_dict())
        self.config = config  # Keep typed config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.accumulation_steps = config.accumulation_steps
        self.patience_counter = 0
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        # Memory monitoring
        self.memory_usage = []
    
    def monitor_memory(self):
        """Monitor memory usage."""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        
        process = psutil.Process()
        memory_info['ram_usage'] = process.memory_info().rss / 1024**3
        
        return memory_info
    
    def train(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Train the deep learning v2 model with advanced techniques.
        
        Args:
            data_path: Path to the training data directory
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        # Create memory-efficient data loaders
        self.logger.info(f"Loading data from {data_path}")
        train_loader, val_loader, test_loader, class_names = create_memory_efficient_loaders(data_path, self.config)
        
        # Update model with discovered classes
        num_classes = len(class_names)
        if hasattr(self.model, 'class_names'):
            self.model.class_names = class_names
            self.model.num_classes = num_classes
        
        # Create new model with correct number of classes if needed
        if self.model.model is None or self.model.model.num_classes != num_classes:
            self.logger.info(f"Creating model for {num_classes} classes")
            pytorch_model = DeepLearningV2(
                num_classes=num_classes,
                input_channels=self.config.input_channels,
                dropout_rates=self.config.dropout_rates,
                attention_reduction=self.config.attention_reduction_ratio,
                spatial_kernel=self.config.spatial_attention_kernel,
                residual_dropout=self.config.residual_dropout
            ).to(self.device)
            
            if hasattr(self.model, 'model'):
                self.model.model = pytorch_model
            else:
                self.model = pytorch_model
        else:
            pytorch_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        # Train the model
        self.logger.info("Starting memory-efficient training...")
        best_val_accuracy = self._train_memory_efficient(pytorch_model, train_loader, val_loader)
        
        # Evaluate on test set
        test_accuracy, predictions, targets, probabilities = self._evaluate_model_advanced(pytorch_model, test_loader)
        
        # Log final metrics
        metrics = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'test_accuracy': test_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'num_classes': num_classes,
            'epochs_trained': len(self.train_losses),
            'model_parameters': sum(p.numel() for p in pytorch_model.parameters())
        }
        self.log_metrics(metrics, phase="final")
        
        return {
            'model': pytorch_model,
            'metrics': metrics,
            'class_names': class_names,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates,
                'best_val_accuracy': self.best_val_accuracy
            },
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities
        }
    
    def _train_memory_efficient(self, model, train_loader, val_loader):
        """Memory-efficient training loop with advanced techniques."""
        self.logger.info(f"Starting memory-efficient training for {self.config.num_epochs} epochs...")
        self.logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        self.logger.info(f"Effective batch size: {train_loader.batch_size * self.accumulation_steps}")
        self.logger.info(f"Training on {model.num_classes} classes")
        
        start_time = time.time()
        
        # Optimizers with advanced settings
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma)
        
        # Label smoothing loss
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Monitor memory at start of epoch
            memory_info = self.monitor_memory()
            self.memory_usage.append(memory_info)
            self.logger.info(f"Memory at epoch start: GPU {memory_info.get('gpu_allocated', 0):.1f}GB, "
                           f"RAM {memory_info.get('ram_usage', 0):.1f}GB")
            
            # Training with memory efficiency
            train_loss, train_acc = self._train_epoch_memory_efficient(
                model, train_loader, criterion, optimizer
            )
            
            # Validation
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            self.logger.info(f"Best Val Acc: {self.best_val_accuracy:.2f}%")
            
            # Aggressive memory cleanup after each epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Load best model
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model state")
        
        return self.best_val_accuracy
    
    def _train_epoch_memory_efficient(self, model, train_loader, criterion, optimizer):
        """Memory-efficient training with gradient accumulation and mixup."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Reset gradients
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup with probability
            if np.random.random() < self.config.mixup_prob:
                mixed_data, y_a, y_b, lam = mixup_data(data, target, self.config.mixup_alpha, self.device)
                
                output = model(mixed_data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                loss.backward()
                
                running_loss += loss.item() * self.accumulation_steps
                
                # Accuracy calculation for mixup is approximate
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (lam * (predicted == y_a).sum().item() + 
                           (1 - lam) * (predicted == y_b).sum().item())
            
            else:
                # Standard training
                output = model(data)
                loss = criterion(output, target)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                loss.backward()
                
                running_loss += loss.item() * self.accumulation_steps
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Memory cleanup every few steps
                if (batch_idx + 1) % (self.accumulation_steps * 4) == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Progress reporting (less frequent)
            if batch_idx % 100 == 0:
                memory_info = self.monitor_memory()
                self.logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * self.accumulation_steps:.4f}, '
                               f'GPU: {memory_info.get("gpu_allocated", 0):.1f}GB')
        
        # Final gradient step if needed
        if len(train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        
        return epoch_loss, epoch_accuracy
    
    def _validate_epoch(self, model, val_loader, criterion):
        """Memory-efficient validation epoch."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Memory cleanup during validation
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)
        
        # Early stopping and best model saving
        if epoch_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = epoch_accuracy
            self.best_model_state = model.state_dict().copy()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return epoch_loss, epoch_accuracy
    
    def _evaluate_model_advanced(self, model, test_loader):
        """Advanced model evaluation with detailed metrics."""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        test_accuracy = 100. * correct / total
        
        self.logger.info(f"Advanced Test Evaluation:")
        self.logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        
        return test_accuracy, all_predictions, all_targets, all_probabilities
    
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not hasattr(self.model, 'model') or self.model.model is None:
            raise ValueError("Model not trained yet")
        
        model = self.model.model if hasattr(self.model, 'model') else self.model
        model.eval()
        
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            
            accuracy = (predicted == y_test).float().mean().item()
        
        return {
            'test_accuracy': accuracy,
            'test_samples': len(y_test)
        }
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        model = self.model.model if hasattr(self.model, 'model') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates,
                'best_val_accuracy': self.best_val_accuracy
            },
            'metrics': metrics,
            'config': self.config,
            'class_names': getattr(self.model, 'class_names', [])
        }
        
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate model if needed
        if 'class_names' in checkpoint:
            num_classes = len(checkpoint['class_names'])
            model = DeepLearningV2(
                num_classes=num_classes,
                input_channels=self.config.input_channels,
                dropout_rates=self.config.dropout_rates,
                attention_reduction=self.config.attention_reduction_ratio,
                spatial_kernel=self.config.spatial_attention_kernel,
                residual_dropout=self.config.residual_dropout
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if hasattr(self.model, 'model'):
                self.model.model = model
                self.model.class_names = checkpoint['class_names']
                self.model.num_classes = num_classes
            else:
                self.model = model
        
        # Restore training history
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accuracies = history.get('train_accuracies', [])
            self.val_accuracies = history.get('val_accuracies', [])
            self.learning_rates = history.get('learning_rates', [])
            self.best_val_accuracy = history.get('best_val_accuracy', 0.0)
        
        return checkpoint