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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import logging

from .config import TransferLearningClassifierConfig

logger = logging.getLogger(__name__)


class TransferLearningModelPyTorch(nn.Module):
    """Transfer learning model using PyTorch pre-trained networks."""
    
    def __init__(self, config: TransferLearningClassifierConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Load pre-trained model
        self.base_model, self.input_size = self._load_base_model()
        
        # Get the number of features from the base model
        if hasattr(self.base_model, 'fc'):
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove original classifier
        elif hasattr(self.base_model, 'classifier'):
            if isinstance(self.base_model.classifier, nn.Sequential):
                num_features = self.base_model.classifier[0].in_features
                self.base_model.classifier = nn.Identity()
            else:
                num_features = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported base model: {config.base_model_name}")
        
        # Build custom head
        layers = []
        input_dim = num_features
        
        # Add dense layers
        for units in self.config.dense_units:
            layers.append(nn.Linear(input_dim, units))
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.config.head_dropout_rate))
            input_dim = units
        
        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
        
        # Freeze base model if configured
        if not self.config.base_trainable:
            self._freeze_base_model()
        
        # Log model info
        self._log_model_info()
    
    def _load_base_model(self) -> Tuple[nn.Module, int]:
        """Load pre-trained base model."""
        model_name = self.config.base_model_name.lower()
        
        if model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            input_size = 224
        elif model_name == 'resnet101':
            base_model = models.resnet101(pretrained=True)
            input_size = 224
        elif model_name == 'resnet152':
            base_model = models.resnet152(pretrained=True)
            input_size = 224
        elif model_name == 'vgg16':
            base_model = models.vgg16(pretrained=True)
            input_size = 224
        elif model_name == 'vgg19':
            base_model = models.vgg19(pretrained=True)
            input_size = 224
        elif model_name == 'densenet121':
            base_model = models.densenet121(pretrained=True)
            input_size = 224
        elif model_name == 'densenet169':
            base_model = models.densenet169(pretrained=True)
            input_size = 224
        elif model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=True)
            input_size = 224
        elif model_name == 'efficientnet_b1':
            base_model = models.efficientnet_b1(pretrained=True)
            input_size = 240
        elif model_name == 'efficientnet_b2':
            base_model = models.efficientnet_b2(pretrained=True)
            input_size = 260
        elif model_name == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=True)
            input_size = 224
        elif model_name == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(pretrained=True)
            input_size = 224
        elif model_name == 'mobilenet_v3_large':
            base_model = models.mobilenet_v3_large(pretrained=True)
            input_size = 224
        else:
            raise ValueError(f"Unsupported base model: {model_name}")
        
        logger.info(f"Loaded {model_name} base model with input size {input_size}")
        return base_model, input_size
    
    def _freeze_base_model(self):
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base model parameters frozen")
    
    def unfreeze_layers(self, num_layers: int):
        """Unfreeze the last num_layers of the base model for fine-tuning."""
        # Get all layers
        all_layers = []
        for child in self.base_model.children():
            if isinstance(child, nn.Sequential):
                all_layers.extend(list(child.children()))
            else:
                all_layers.append(child)
        
        # Unfreeze last num_layers
        layers_to_unfreeze = all_layers[-num_layers:] if num_layers > 0 else []
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Unfroze last {num_layers} layers of base model")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    def forward(self, x):
        """Forward pass."""
        # Extract features from base model
        features = self.base_model(x)
        
        # Pass through custom head
        output = self.head(features)
        
        return output
    
    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Base model: {self.config.base_model_name}")
        logger.info(f"Input shape: ({self.input_size}, {self.input_size}, 3)")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def get_optimizer(self, phase: str = 'initial') -> optim.Optimizer:
        """Get optimizer for training."""
        if phase == 'initial':
            # Only train the head
            params = self.head.parameters()
            lr = self.config.learning_rate
        else:  # fine-tuning
            # Train entire model with different learning rates
            params = [
                {'params': self.base_model.parameters(), 'lr': self.config.fine_tune_learning_rate},
                {'params': self.head.parameters(), 'lr': self.config.learning_rate}
            ]
            lr = self.config.learning_rate
        
        # Create optimizer
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def get_scheduler(self, optimizer: optim.Optimizer, num_epochs: int):
        """Get learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=0
            )
        elif self.config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=num_epochs // 3, gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience, min_lr=1e-7
            )
        else:
            scheduler = None
        
        return scheduler
    
    def get_transforms(self) -> Dict[str, transforms.Compose]:
        """Get data transforms for training and validation."""
        # Normalization values for ImageNet pre-trained models
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
        
        # Validation/test transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(int(self.input_size * 1.14)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            normalize
        ])
        
        return {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        }