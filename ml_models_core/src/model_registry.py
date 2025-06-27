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

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelMetadata:
    """Model metadata structure."""
    name: str
    version: str
    model_type: str  # "shallow", "deep_v1", "deep_v2", "transfer", "ensemble"
    accuracy: float
    training_date: str
    model_path: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]


class ModelRegistry:
    """Central registry for managing model metadata and versions."""
    
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self.models: Dict[str, List[ModelMetadata]] = {}
        self.load_registry()
    
    def load_registry(self) -> None:
        """Load the model registry from file."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self.models[model_name] = [
                        ModelMetadata(**version) for version in versions
                    ]
        else:
            self.models = {}
    
    def save_registry(self) -> None:
        """Save the model registry to file."""
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = [asdict(version) for version in versions]
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a new model version."""
        if metadata.name not in self.models:
            self.models[metadata.name] = []
        
        self.models[metadata.name].append(metadata)
        self.save_registry()
    
    def get_model(self, name: str, version: str = "latest") -> Optional[ModelMetadata]:
        """Get model metadata by name and version."""
        if name not in self.models:
            return None
        
        versions = self.models[name]
        if version == "latest":
            return max(versions, key=lambda x: x.training_date)
        
        for model in versions:
            if model.version == version:
                return model
        
        return None
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions."""
        return {
            name: [model.version for model in versions]
            for name, versions in self.models.items()
        }
    
    def get_best_model(self, model_type: str = None) -> Optional[ModelMetadata]:
        """Get the best performing model, optionally filtered by type."""
        all_models = []
        for versions in self.models.values():
            all_models.extend(versions)
        
        if model_type:
            all_models = [m for m in all_models if m.model_type == model_type]
        
        if not all_models:
            return None
        
        return max(all_models, key=lambda x: x.accuracy)