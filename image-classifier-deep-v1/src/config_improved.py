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
from typing import Tuple
from config import DeepLearningConfig


@dataclass
class DeepLearningV1ImprovedConfig(DeepLearningConfig):
    """Improved configuration for Deep Learning v1 classifier with tunable parameters."""
    
    # Model architecture
    input_channels: int = 3
    dropout_rate: float = 0.5
    use_residual: bool = True  # Enable residual connections
    
    # Training parameters - optimized for better performance
    learning_rate: float = 0.01  # Higher initial LR
    weight_decay: float = 1e-4
    batch_size: int = 64  # Increased from 8 to 64 (configurable)
    num_epochs: int = 50  # More epochs with early stopping
    
    # Optimizer settings
    optimizer: str = "adamw"  # Options: "adam", "adamw", "sgd"
    momentum: float = 0.9  # For SGD
    nesterov: bool = True  # For SGD
    
    # Early stopping
    patience: int = 10  # Increased patience
    min_delta: float = 0.001
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"  # Options: "step", "cosine", "plateau"
    lr_step_size: int = 10  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler
    lr_min: float = 1e-6  # Minimum learning rate for cosine
    lr_warmup_epochs: int = 5  # Warmup period
    
    # Data augmentation - enhanced
    rotation_degrees: int = 15  # Increased from 10
    color_jitter_brightness: float = 0.3  # Increased from 0.2
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.1
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0  # Added vertical flip option
    random_erasing_prob: float = 0.1  # Added random erasing
    
    # MixUp augmentation
    mixup_alpha: float = 0.2  # MixUp augmentation strength
    use_mixup: bool = True
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Normalization values (ImageNet pretrained)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # GPU optimization
    use_amp: bool = True  # Automatic mixed precision
    gradient_accumulation_steps: int = 1  # For simulating larger batches
    gradient_clip: float = 1.0  # Gradient clipping
    
    # Paths
    model_save_path: str = "models/deep_v1_improved.pth"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    # Additional features
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # For faster GPU transfer
    
    def get_effective_batch_size(self):
        """Get the effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def adjust_lr_for_batch_size(self, base_lr: float = 0.01, base_batch_size: int = 64):
        """Adjust learning rate based on batch size using linear scaling rule."""
        return base_lr * (self.get_effective_batch_size() / base_batch_size)