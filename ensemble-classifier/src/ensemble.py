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
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml-models-core', 'src'))

from base_classifier import BaseImageClassifier
from voting_strategies import VotingStrategy, MajorityVoting
from typing import List, Dict, Any
import numpy as np


class EnsembleClassifier(BaseImageClassifier):
    """Ensemble classifier that combines multiple models."""
    
    def __init__(self, models: List[BaseImageClassifier], voting_strategy: VotingStrategy = None):
        super().__init__("ensemble", "1.0.0")
        self.models = models
        self.voting_strategy = voting_strategy or MajorityVoting()
        self._is_loaded = True  # Ensemble is loaded when constituent models are loaded
    
    def load_model(self, model_path: str) -> None:
        """Load all constituent models."""
        for model in self.models:
            if not model.is_loaded:
                model.load_model(model_path)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Use the first model's preprocessing (assuming all models use similar preprocessing)."""
        if self.models:
            return self.models[0].preprocess(image)
        return image
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make ensemble predictions using the voting strategy."""
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            if model.is_loaded:
                predictions = model.predict(image)
                all_predictions.append(predictions)
        
        if not all_predictions:
            raise ValueError("No models are loaded and ready for prediction")
        
        # Combine predictions using voting strategy
        return self.voting_strategy.combine_predictions(all_predictions)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the ensemble."""
        model_metadata = []
        for model in self.models:
            model_metadata.append({
                "name": model.model_name,
                "version": model.version,
                "metadata": model.get_metadata()
            })
        
        return {
            "ensemble_version": self.version,
            "voting_strategy": self.voting_strategy.__class__.__name__,
            "constituent_models": model_metadata,
            "total_models": len(self.models)
        }
    
    def add_model(self, model: BaseImageClassifier) -> None:
        """Add a new model to the ensemble."""
        self.models.append(model)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the ensemble by name."""
        for i, model in enumerate(self.models):
            if model.model_name == model_name:
                self.models.pop(i)
                return True
        return False