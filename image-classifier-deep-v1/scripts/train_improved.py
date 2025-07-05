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
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "ml_models_core" / "src"))

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import json
from datetime import datetime
import os
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from src.model_improved import DeepLearningV1Improved
from src.config_improved import DeepLearningV1ImprovedConfig
from base_classifier import BaseImageClassifier


class ImprovedTrainer:
    """Trainer for the improved Deep Learning v1 model."""
    
    def __init__(self, config: DeepLearningV1ImprovedConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Log GPU info
        if torch.cuda.is_available():
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Create directories
        os.makedirs(Path(config.model_save_path).parent, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
    
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.config.log_dir) / f"training_improved_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _get_transforms(self):
        """Get data transforms with enhanced augmentation."""
        train_transform = transforms.Compose([
            transforms.RandomRotation(self.config.rotation_degrees),
            transforms.ColorJitter(
                brightness=self.config.color_jitter_brightness,
                contrast=self.config.color_jitter_contrast,
                saturation=self.config.color_jitter_saturation,
                hue=self.config.color_jitter_hue
            ),
            transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
            transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.normalize_mean, self.config.normalize_std),
            transforms.RandomErasing(p=self.config.random_erasing_prob) if self.config.random_erasing_prob > 0 else nn.Identity()
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.normalize_mean, self.config.normalize_std)
        ])
        
        return train_transform, val_transform
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply MixUp augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """MixUp loss calculation."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def _create_optimizer(self, model):
        """Create optimizer based on configuration."""
        # Adjust learning rate based on batch size
        lr = self.config.adjust_lr_for_batch_size(self.config.learning_rate)
        self.logger.info(f"Adjusted learning rate: {lr:.6f} (base: {self.config.learning_rate}, batch_size: {self.config.batch_size})")
        
        if self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=self.config.nesterov
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=self.config.weight_decay
            )
        else:  # adam
            return optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self, optimizer, num_batches):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            # Cosine annealing with warmup
            def lr_lambda(epoch):
                if epoch < self.config.lr_warmup_epochs:
                    return float(epoch) / float(max(1, self.config.lr_warmup_epochs))
                progress = float(epoch - self.config.lr_warmup_epochs) / float(max(1, self.config.num_epochs - self.config.lr_warmup_epochs))
                return max(self.config.lr_min / self.config.learning_rate, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=self.config.lr_gamma,
                patience=self.config.lr_step_size // 2, verbose=True
            )
        else:  # step
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply MixUp
            if self.config.use_mixup and np.random.rand() < 0.5:
                inputs, labels_a, labels_b, lam = self.mixup_data(inputs, labels, self.config.mixup_alpha)
            else:
                labels_a = labels_b = labels
                lam = 1.0
            
            # Forward pass with AMP
            if self.config.use_amp:
                with autocast():
                    outputs = model(inputs)
                    if lam == 1.0:
                        loss = criterion(outputs, labels)
                    else:
                        loss = self.mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(inputs)
                if lam == 1.0:
                    loss = criterion(outputs, labels)
                else:
                    loss = self.mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0) * self.config.gradient_accumulation_steps
            if lam == 1.0:
                running_corrects += torch.sum(preds == labels.data)
            else:
                # Convert MixUp accuracy calculation to integer
                mixup_corrects = lam * torch.sum(preds == labels_a.data) + (1 - lam) * torch.sum(preds == labels_b.data)
                running_corrects += int(mixup_corrects.item())
            total_samples += inputs.size(0)
            
            # Log progress
            if batch_idx % 50 == 0:
                self.logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                               f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.config.use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def train(self, data_path):
        """Main training loop."""
        # Prepare data
        train_transform, val_transform = self._get_transforms()
        
        # Load datasets
        full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        
        # Split into train/val
        total_size = len(full_dataset)
        val_size = int(total_size * 0.2)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # Update val dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Log dataset info
        num_classes = len(full_dataset.classes)
        self.logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Effective batch size: {self.config.get_effective_batch_size()}")
        
        # Create model
        model = DeepLearningV1Improved(
            num_classes=num_classes,
            dropout_rate=self.config.dropout_rate,
            use_residual=self.config.use_residual
        ).to(self.device)
        
        # Log model info
        model_info = model.get_model_info()
        self.logger.info(f"Model architecture: {model_info['architecture']}")
        self.logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Loss and optimizer
        if self.config.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if self.config.scheduler_type == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # Log metrics
            self.logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                           f"LR: {current_lr:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': self.config
                }, self.config.model_save_path)
                self.logger.info(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        history_path = Path(self.config.log_dir) / "training_history_improved.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train improved Deep Learning v1 model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Initial learning rate (default: 0.01)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--use_residual', action='store_true',
                       help='Use residual connections in the model')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use (default: adamw)')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--no_mixup', action='store_true',
                       help='Disable MixUp augmentation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = DeepLearningV1ImprovedConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_residual=args.use_residual,
        optimizer=args.optimizer,
        use_amp=not args.no_amp,
        use_mixup=not args.no_mixup
    )
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Effective batch size: {config.get_effective_batch_size()}")
    print(f"Adjusted learning rate: {config.adjust_lr_for_batch_size():.6f}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Use residual: {config.use_residual}")
    print(f"Use AMP: {config.use_amp}")
    print(f"Use MixUp: {config.use_mixup}")
    print()
    
    # Train model
    trainer = ImprovedTrainer(config)
    model, best_acc = trainer.train(args.data_path)
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()