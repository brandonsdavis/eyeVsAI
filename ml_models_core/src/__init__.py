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

from .base_classifier import BaseImageClassifier
from .model_registry import ModelRegistry, ModelMetadata
from .data_manager import UnifiedDatasetManager, get_dataset_manager, create_combined_classification_dataset, create_three_class_classification_dataset, create_four_class_classification_dataset
from .data_loaders import get_main_classification_data, get_pets_data, get_vegetables_data, get_street_foods_data, get_musical_instruments_data, get_unified_classification_data, get_three_class_data, get_four_class_data
from .data_utils import verify_dataset_integrity, visualize_dataset_statistics, create_dataset_report
from .utils import ModelUtils

__version__ = "0.1.0"
__all__ = [
    "BaseImageClassifier", 
    "ModelRegistry", 
    "ModelMetadata",
    "UnifiedDatasetManager",
    "get_dataset_manager",
    "create_combined_classification_dataset",
    "create_three_class_classification_dataset",
    "create_four_class_classification_dataset",
    "get_main_classification_data",
    "get_pets_data", 
    "get_vegetables_data",
    "get_street_foods_data",
    "get_musical_instruments_data",
    "get_unified_classification_data",
    "get_three_class_data",
    "get_four_class_data",
    "verify_dataset_integrity",
    "visualize_dataset_statistics", 
    "create_dataset_report",
    "ModelUtils"
]