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
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core" / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix

from base_trainer import BaseTrainer
from .model import DeepLearningV1
from .data_loader import create_data_loaders
from .config import DeepLearningV1Config


class DeepLearningV1Trainer(BaseTrainer):
    """Trainer for Deep Learning v1 models."""
    
    def __init__(self, model, config: DeepLearningV1Config):
        super().__init__(model, config.to_dict())
        self.config = config  # Keep typed config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Early stopping parameters
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.wait = 0
        self.stopped_epoch = 0
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
    def train(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Train the deep learning model.
        
        Args:
            data_path: Path to the training data directory
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        # Create data loaders
        self.logger.info(f"Loading data from {data_path}")
        train_loader, val_loader, test_loader, class_names = create_data_loaders(data_path, self.config)
        
        # Update model with discovered classes
        num_classes = len(class_names)
        if hasattr(self.model, 'class_names'):
            self.model.class_names = class_names
            self.model.num_classes = num_classes
        
        # Create new model with correct number of classes if needed
        if self.model.model is None or self.model.model.num_classes != num_classes:
            self.logger.info(f"Creating model for {num_classes} classes")
            pytorch_model = DeepLearningV1(
                num_classes=num_classes,
                input_channels=self.config.input_channels,
                dropout_rate=self.config.dropout_rate
            ).to(self.device)
            
            if hasattr(self.model, 'model'):
                self.model.model = pytorch_model
            else:
                self.model = pytorch_model
        else:
            pytorch_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            pytorch_model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.config.lr_step_size, 
            gamma=self.config.lr_gamma
        )
        
        # Train model
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training
            train_loss, train_acc = self._train_epoch(pytorch_model, train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(pytorch_model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update best validation accuracy
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
            
            # Check for early stopping
            if self._check_early_stopping(val_loss, epoch):
                break
        
        training_time = time.time() - start_time
        
        # Load best model
        if self.best_model_state:
            pytorch_model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model state")
        
        # Evaluate on test set
        test_accuracy, predictions, targets = self._evaluate_model(pytorch_model, test_loader)
        
        # Log final metrics
        metrics = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'test_accuracy': test_accuracy,
            'best_val_accuracy': self.best_val_accuracy,
            'training_time': training_time,
            'num_classes': num_classes,
            'epochs_trained': epoch + 1 if self.stopped_epoch > 0 else self.config.num_epochs
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
                'val_accuracies': self.val_accuracies
            }
        }
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                self.logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        
        return epoch_loss, epoch_accuracy
    
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)
        
        return epoch_loss, epoch_accuracy
    
    def _check_early_stopping(self, val_loss, epoch):
        """Check if training should stop early."""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_model_state = (self.model.model if hasattr(self.model, 'model') else self.model).state_dict().copy()
            self.wait = 0
            self.logger.info(f"Validation loss improved to {val_loss:.4f} - saving best model")
        else:
            self.wait += 1
            self.logger.info(f"Validation loss did not improve ({self.wait}/{self.patience})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                return True
        
        return False
    
    def _evaluate_model(self, model, test_loader):
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_accuracy = 100. * correct / total
        self.logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        
        return test_accuracy, all_predictions, all_targets
    
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
                'val_accuracies': self.val_accuracies
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
            model = DeepLearningV1(
                num_classes=num_classes,
                input_channels=self.config.input_channels,
                dropout_rate=self.config.dropout_rate
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
        
        return checkpoint