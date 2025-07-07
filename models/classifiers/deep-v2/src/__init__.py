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

"""
Deep Learning v2 Image Classifier Package

This package provides an advanced CNN implementation with attention mechanisms,
residual connections, and modern training techniques for image classification.
"""

from .classifier import DeepLearningV2Classifier
from .trainer import MemoryEfficientTrainingManager
from .model import DeepLearningV2, AttentionBlock, ResidualBlock
from .config import DeepLearningV2Config
from .data_loader import create_memory_efficient_loaders, LazyUnifiedDataset, mixup_data, mixup_criterion

__version__ = "2.0.0"
__author__ = "Brandon Davis"

__all__ = [
    'DeepLearningV2Classifier',
    'MemoryEfficientTrainingManager', 
    'DeepLearningV2',
    'AttentionBlock',
    'ResidualBlock',
    'DeepLearningV2Config',
    'create_memory_efficient_loaders',
    'LazyUnifiedDataset',
    'mixup_data',
    'mixup_criterion'
]