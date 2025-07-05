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
Deep Learning v1 Image Classifier Package

This package provides a custom CNN implementation for image classification
with PyTorch, designed for multi-class classification tasks.
"""

from .classifier import DeepLearningV1Classifier
from .trainer import DeepLearningV1Trainer
from .model import DeepLearningV1
from .config import DeepLearningV1Config
from .data_loader import create_data_loaders, UnifiedDataset, get_transforms

__version__ = "1.0.0"
__author__ = "Brandon Davis"

__all__ = [
    'DeepLearningV1Classifier',
    'DeepLearningV1Trainer', 
    'DeepLearningV1',
    'DeepLearningV1Config',
    'create_data_loaders',
    'UnifiedDataset',
    'get_transforms'
]