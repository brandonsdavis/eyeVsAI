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

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class for all models."""
    
    # Common training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Data parameters - NO num_classes (determined from data)
    image_size: tuple = (224, 224)
    normalize: bool = True
    
    # Training behavior
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    save_best_only: bool = True
    verbose: int = 1
    
    # Paths
    model_save_path: str = "models/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def update(self, **kwargs) -> None:
        """Update config parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")


@dataclass
class ShallowLearningConfig(BaseConfig):
    """Configuration for shallow learning models."""
    
    # Feature extraction
    n_components_svd: int = 100
    use_pca: bool = False
    n_components_pca: int = 50
    
    # SVM parameters
    svm_kernel: str = 'rbf'
    svm_C: float = 1.0
    svm_gamma: str = 'scale'
    
    # Feature preprocessing
    flatten_images: bool = True
    feature_scaling: str = 'standard'  # 'standard', 'minmax', or None


@dataclass
class DeepLearningConfig(BaseConfig):
    """Configuration for deep learning models."""
    
    # Architecture
    hidden_layers: list = field(default_factory=lambda: [128, 64])
    activation: str = 'relu'
    dropout_rate: float = 0.5
    batch_norm: bool = True
    
    # Optimization
    optimizer: str = 'adam'
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Data augmentation
    augment_train: bool = True
    rotation_range: float = 20.0
    zoom_range: float = 0.15
    horizontal_flip: bool = True
    
    # Memory optimization
    num_workers: int = 2
    pin_memory: bool = False
    prefetch_factor: int = 2


@dataclass
class TransferLearningConfig(BaseConfig):
    """Configuration for transfer learning models."""
    
    # Base model
    base_model: str = 'resnet50'
    freeze_base_layers: bool = True
    fine_tune_at_layer: Optional[int] = None
    
    # Custom head
    head_hidden_units: list = field(default_factory=lambda: [512, 256])
    head_dropout_rate: float = 0.5
    use_batch_norm: bool = True
    
    # Two-phase training
    initial_epochs: int = 20
    fine_tune_epochs: int = 10
    fine_tune_learning_rate: float = 1e-5
    
    # Preprocessing
    preprocess_input_mode: str = 'tf'  # 'tf', 'torch', or 'caffe'