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

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class VotingStrategy(ABC):
    """Base class for ensemble voting strategies."""
    
    @abstractmethod
    def combine_predictions(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions from multiple models."""
        pass


class MajorityVoting(VotingStrategy):
    """Simple majority voting strategy."""
    
    def combine_predictions(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions by taking the most common top prediction."""
        if not predictions:
            return {}
        
        # Get top prediction from each model
        top_predictions = []
        for pred_dict in predictions:
            if pred_dict:
                top_class = max(pred_dict.items(), key=lambda x: x[1])[0]
                top_predictions.append(top_class)
        
        # Count votes for each class
        vote_counts = {}
        for pred in top_predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        # Convert to probabilities
        total_votes = len(top_predictions)
        result = {class_name: count / total_votes 
                 for class_name, count in vote_counts.items()}
        
        return result


class WeightedVoting(VotingStrategy):
    """Weighted voting strategy based on model confidence."""
    
    def __init__(self, weights: List[float] = None):
        self.weights = weights
    
    def combine_predictions(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions using weighted average."""
        if not predictions:
            return {}
        
        # Use equal weights if none provided
        if self.weights is None:
            self.weights = [1.0] * len(predictions)
        
        # Get all unique class names
        all_classes = set()
        for pred_dict in predictions:
            all_classes.update(pred_dict.keys())
        
        # Weighted average
        combined = {}
        total_weight = sum(self.weights[:len(predictions)])
        
        for class_name in all_classes:
            weighted_sum = 0.0
            for i, pred_dict in enumerate(predictions):
                weight = self.weights[i] if i < len(self.weights) else 1.0
                confidence = pred_dict.get(class_name, 0.0)
                weighted_sum += weight * confidence
            
            combined[class_name] = weighted_sum / total_weight
        
        return combined


class StackingVoting(VotingStrategy):
    """Stacking ensemble strategy (placeholder for future ML-based combination)."""
    
    def __init__(self):
        # In a real implementation, this would include a meta-learner
        # For now, fall back to weighted voting
        self.fallback_strategy = WeightedVoting()
    
    def combine_predictions(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions using stacking (currently falls back to weighted voting)."""
        # TODO: Implement actual stacking with a trained meta-model
        return self.fallback_strategy.combine_predictions(predictions)