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
class DeepLearningV1Config(DeepLearningConfig):
    """Configuration for Deep Learning v1 classifier."""
    
    # Model architecture
    input_channels: int = 3
    dropout_rate: float = 0.5
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 30
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Learning rate scheduling
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    
    # Data augmentation
    rotation_degrees: int = 10
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    horizontal_flip_prob: float = 0.5
    
    # Normalization values (ImageNet pretrained)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Paths
    model_save_path: str = "models/deep_v1_classifier.pth"
    log_dir: str = "logs/"