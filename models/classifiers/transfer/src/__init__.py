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
Transfer Learning Image Classifier Package

This package provides a transfer learning implementation using pre-trained models
from TensorFlow/Keras for image classification tasks.
"""

from .classifier import TransferLearningClassifier
from .trainer import TransferLearningTrainer
from .models import TransferLearningModel
from .config import TransferLearningClassifierConfig
from .data_loader import (
    create_memory_efficient_datasets,
    discover_classes,
    compute_class_weights,
    create_inference_dataset
)

__version__ = "1.0.0"
__author__ = "Brandon Davis"

__all__ = [
    'TransferLearningClassifier',
    'TransferLearningTrainer',
    'TransferLearningModel',
    'TransferLearningClassifierConfig',
    'create_memory_efficient_datasets',
    'discover_classes',
    'compute_class_weights',
    'create_inference_dataset'
]