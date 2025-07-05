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

from dataclasses import dataclass
from typing import Tuple, Optional
from config import DeepLearningConfig


@dataclass
class DeepLearningV2ImprovedConfig(DeepLearningConfig):
    """Improved configuration for Deep Learning v2 with advanced optimization."""
    
    # Model architecture
    input_channels: int = 3
    architecture: str = "resnet"  # Options: "resnet", "densenet", "hybrid"
    use_cbam: bool = True  # Convolutional Block Attention Module
    use_se: bool = True  # Squeeze-and-Excitation
    dropout_rates: Tuple[float, float, float] = (0.4, 0.3, 0.2)  # Reduced from original
    residual_dropout: float = 0.05  # Reduced for better gradient flow
    
    # Training parameters - optimized for larger batch sizes
    learning_rate: float = 0.1  # Higher initial LR with warmup
    weight_decay: float = 5e-4  # Slightly higher for better regularization
    batch_size: int = 64  # Increased from 8 to 64
    num_epochs: int = 100  # More epochs with early stopping
    accumulation_steps: int = 1  # Reduced since we have larger batch size
    
    # Optimizer configuration
    optimizer: str = "sgd"  # Options: "sgd", "adam", "adamw", "rmsprop"
    momentum: float = 0.9
    nesterov: bool = True
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"  # Options: "cosine", "step", "plateau", "onecycle"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-5
    lr_step_size: int = 30  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler
    
    # Advanced training techniques
    mixup_alpha: float = 0.3  # Increased from 0.2
    mixup_prob: float = 0.5  # Increased probability
    cutmix_alpha: float = 1.0  # CutMix augmentation
    cutmix_prob: float = 0.5
    label_smoothing: float = 0.1  # Increased from 0.05
    
    # Data augmentation - enhanced
    rotation_degrees: int = 20  # Increased from 15
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.1  # Added for certain datasets
    random_erasing_prob: float = 0.2  # Increased from 0.05
    random_erasing_scale: Tuple[float, float] = (0.02, 0.33)
    
    # Auto augmentation
    use_autoaugment: bool = True
    autoaugment_policy: str = "imagenet"  # Options: "imagenet", "cifar10", "svhn"
    
    # Early stopping and monitoring
    patience: int = 15  # Increased patience
    min_delta: float = 0.0005  # More sensitive
    monitor_metric: str = "val_acc"  # What to monitor for early stopping
    
    # Training optimization
    gradient_clip_norm: float = 1.0
    gradient_clip_value: Optional[float] = None
    use_amp: bool = True  # Automatic Mixed Precision
    use_sync_bn: bool = False  # Synchronized BatchNorm for multi-GPU
    
    # Memory optimization
    gradient_checkpointing: bool = True  # Trade compute for memory
    pin_memory: bool = True  # Faster GPU transfer
    num_workers: int = 4  # Increased for faster data loading
    persistent_workers: bool = True  # Keep workers alive between epochs
    
    # Normalization
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Stochastic depth (for residual networks)
    stochastic_depth_prob: float = 0.1  # Drop path rate
    
    # Model EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_freq: int = 1
    
    # Paths
    model_save_path: str = "models/deep_v2_improved.pth"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    # Additional features
    save_frequency: int = 5  # Save checkpoint every N epochs
    log_frequency: int = 50  # Log every N batches
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with accumulation."""
        return self.batch_size * self.accumulation_steps
    
    def adjust_lr_for_batch_size(self, base_lr: float = 0.1, base_batch_size: int = 256) -> float:
        """Adjust learning rate based on linear scaling rule."""
        return base_lr * (self.get_effective_batch_size() / base_batch_size)
    
    def get_weight_decay_params(self, model: 'nn.Module') -> list:
        """Get parameters that should have weight decay applied."""
        # Don't apply weight decay to batch norm and bias parameters
        decay = []
        no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        return [
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]