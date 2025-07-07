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

from dataclasses import dataclass
from typing import Tuple
from config import DeepLearningConfig


@dataclass
class DeepLearningV2Config(DeepLearningConfig):
    """Configuration for Deep Learning v2 classifier with advanced features."""
    
    # Model architecture
    input_channels: int = 3
    dropout_rates: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # Multi-layer dropout
    
    # Advanced training parameters
    learning_rate: float = 0.0005  # Lower for stability
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 25
    accumulation_steps: int = 4  # Gradient accumulation
    
    # Early stopping and monitoring
    patience: int = 8
    min_delta: float = 0.001
    
    # Learning rate scheduling
    lr_step_size: int = 15
    lr_gamma: float = 0.5
    
    # Advanced augmentation
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.3  # Probability of applying mixup
    label_smoothing: float = 0.05
    
    # Data augmentation
    rotation_degrees: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    horizontal_flip_prob: float = 0.5
    random_erasing_prob: float = 0.05
    random_erasing_scale: Tuple[float, float] = (0.02, 0.20)
    
    # Attention mechanisms
    attention_reduction_ratio: int = 8  # Channel attention reduction
    spatial_attention_kernel: int = 7
    
    # Residual blocks
    residual_dropout: float = 0.1
    
    # Training optimization
    gradient_clip_norm: float = 1.0
    memory_efficient: bool = True
    pin_memory: bool = False  # Disabled for memory efficiency
    num_workers: int = 1  # Reduced for memory
    
    # Normalization values (ImageNet pretrained)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Paths
    model_save_path: str = "models/deep_v2_classifier.pth"
    log_dir: str = "logs/"