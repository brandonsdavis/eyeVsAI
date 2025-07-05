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
from typing import Tuple, List

# Import from the ml_models_core config module (renamed to avoid circular import)
import importlib
ml_config_module = importlib.import_module("config")
TransferLearningConfig = ml_config_module.TransferLearningConfig


@dataclass
class TransferLearningClassifierConfig(TransferLearningConfig):
    """Configuration for Transfer Learning classifier with TensorFlow/Keras."""
    
    # TensorFlow-specific settings
    mixed_precision: bool = True
    use_xla: bool = True
    memory_growth: bool = True
    
    # Transfer learning specific
    base_model_name: str = "resnet50"  # resnet50, vgg16, efficientnet_b0
    base_trainable: bool = False
    fine_tune_layers: int = 10
    fine_tune_learning_rate: float = 1e-5
    
    # Data preprocessing
    rescale: float = 1.0/255.0
    validation_split: float = 0.2
    
    # Advanced training
    class_weights: bool = True
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.2
    
    # Cache and performance
    cache_dataset: bool = True
    prefetch_buffer: int = 64
    parallel_calls: int = 4
    
    # Model architecture
    global_pooling: str = "avg"  # avg, max
    dense_units: List[int] = None
    final_activation: str = "softmax"
    
    # PyTorch specific settings
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau, none
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.dense_units is None:
            self.dense_units = [512, 256]