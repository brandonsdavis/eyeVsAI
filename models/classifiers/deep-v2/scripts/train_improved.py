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
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core" / "src"))

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
from torch.amp import GradScaler, autocast
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR
from torchvision.transforms import autoaugment

from src.model_improved import DeepLearningV2Improved
from src.config_improved import DeepLearningV2ImprovedConfig


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model, decay=0.9999, device='cuda'):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_val = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_val.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


class ImprovedTrainer:
    """Trainer for the improved Deep Learning v2 model."""
    
    def __init__(self, config: DeepLearningV2ImprovedConfig):
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
        self.scaler = GradScaler('cuda') if config.use_amp and torch.cuda.is_available() else None
    
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
        train_transforms = [
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
            transforms.Normalize(self.config.normalize_mean, self.config.normalize_std)
        ]
        
        # Add AutoAugment if enabled
        if self.config.use_autoaugment:
            if self.config.autoaugment_policy == "imagenet":
                train_transforms.insert(0, autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET))
            elif self.config.autoaugment_policy == "cifar10":
                train_transforms.insert(0, autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.CIFAR10))
        
        # Add RandomErasing
        if self.config.random_erasing_prob > 0:
            train_transforms.append(
                transforms.RandomErasing(
                    p=self.config.random_erasing_prob,
                    scale=self.config.random_erasing_scale
                )
            )
        
        train_transform = transforms.Compose(train_transforms)
        
        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.normalize_mean, self.config.normalize_std)
        ])
        
        return train_transform, val_transform
    
    def _create_optimizer(self, model):
        """Create optimizer with proper weight decay handling."""
        # Get parameter groups
        param_groups = self.config.get_weight_decay_params(model)
        
        # Adjust learning rate based on batch size
        lr = self.config.adjust_lr_for_batch_size(self.config.learning_rate)
        self.logger.info(f"Adjusted learning rate: {lr:.6f} (base: {self.config.learning_rate}, batch_size: {self.config.batch_size})")
        
        if self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                param_groups,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                param_groups,
                lr=lr,
                betas=self.config.betas
            )
        elif self.config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                param_groups,
                lr=lr,
                momentum=self.config.momentum
            )
        else:  # adam
            return optim.Adam(
                param_groups,
                lr=lr,
                betas=self.config.betas
            )
    
    def _create_scheduler(self, optimizer, num_batches):
        """Create learning rate scheduler."""
        total_steps = self.config.num_epochs * num_batches
        
        if self.config.scheduler_type == "cosine":
            # Cosine annealing with warmup
            warmup_steps = self.config.lr_warmup_epochs * num_batches
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(self.config.lr_min / self.config.learning_rate,
                          0.5 * (1.0 + np.cos(np.pi * progress)))
            
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif self.config.scheduler_type == "plateau":
            return ReduceLROnPlateau(
                optimizer, mode='max', factor=self.config.lr_gamma,
                patience=5, verbose=True, min_lr=self.config.lr_min
            )
        else:  # step
            return StepLR(
                optimizer, step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
    
    def train_epoch(self, model, train_loader, criterion, optimizer, scheduler, epoch, ema=None):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply MixUp or CutMix
            r = np.random.rand(1)
            if r < self.config.mixup_prob and self.config.mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, self.config.mixup_alpha)
                mixed = True
            elif r < self.config.mixup_prob + self.config.cutmix_prob and self.config.cutmix_alpha > 0:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels, self.config.cutmix_alpha)
                mixed = True
            else:
                mixed = False
            
            # Forward pass with AMP
            if self.config.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    outputs = model(inputs)
                    if mixed:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                if mixed:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labels)
            
            # Gradient accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.config.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.use_amp and self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                    if self.config.gradient_clip_norm:
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if self.config.gradient_clip_norm:
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                # Update scheduler (for OneCycle)
                if self.config.scheduler_type in ["onecycle", "cosine"]:
                    scheduler.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0) * self.config.accumulation_steps
            
            if not mixed:
                running_corrects += torch.sum(preds == labels.data).item()
            else:
                running_corrects += (lam * torch.sum(preds == labels_a.data).item() + 
                                   (1 - lam) * torch.sum(preds == labels_b.data).item())
            total_samples += inputs.size(0)
            
            # Log progress
            if batch_idx % self.config.log_frequency == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                               f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion, ema=None):
        """Validate the model."""
        # Apply EMA if available
        if ema is not None:
            ema.apply_shadow()
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.config.use_amp and self.scaler is not None:
                    with autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)
        
        # Restore original weights if EMA was applied
        if ema is not None:
            ema.restore()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
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
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
        # Log dataset info
        num_classes = len(full_dataset.classes)
        self.logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        self.logger.info(f"Number of classes: {num_classes}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Effective batch size: {self.config.get_effective_batch_size()}")
        
        # Create model
        model = DeepLearningV2Improved(
            num_classes=num_classes,
            dropout_rates=self.config.dropout_rates,
            use_cbam=self.config.use_cbam,
            use_se=self.config.use_se,
            residual_dropout=self.config.residual_dropout,
            architecture=self.config.architecture
        ).to(self.device)
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
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
        
        # EMA
        ema = ModelEMA(model, decay=self.config.ema_decay) if self.config.use_ema else None
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, scheduler, epoch, ema
            )
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion, ema)
            
            # Update scheduler (for non-batch schedulers)
            if self.config.scheduler_type == "plateau":
                scheduler.step(val_acc)
            elif self.config.scheduler_type == "step":
                scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
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
                
                # Apply EMA for saving if available
                if ema is not None:
                    ema.apply_shadow()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': self.config
                }, self.config.model_save_path)
                
                if ema is not None:
                    ema.restore()
                
                self.logger.info(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_frequency == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
            
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
    parser = argparse.ArgumentParser(description='Train improved Deep Learning v2 model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Initial learning rate (default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--architecture', type=str, default='resnet',
                       choices=['resnet', 'densenet', 'hybrid'],
                       help='Model architecture (default: resnet)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                       help='Optimizer to use (default: sgd)')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--no_ema', action='store_true',
                       help='Disable exponential moving average')
    parser.add_argument('--no_cbam', action='store_true',
                       help='Disable CBAM attention')
    
    args = parser.parse_args()
    
    # Create configuration
    config = DeepLearningV2ImprovedConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        architecture=args.architecture,
        optimizer=args.optimizer,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        use_cbam=not args.no_cbam
    )
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"Architecture: {config.architecture}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Effective batch size: {config.get_effective_batch_size()}")
    print(f"Adjusted learning rate: {config.adjust_lr_for_batch_size():.6f}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Use AMP: {config.use_amp}")
    print(f"Use EMA: {config.use_ema}")
    print(f"Use CBAM: {config.use_cbam}")
    print()
    
    # Train model
    trainer = ImprovedTrainer(config)
    model, best_acc = trainer.train(args.data_path)
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()