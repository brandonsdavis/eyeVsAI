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
from typing import Dict, Any, List
import numpy as np
from PIL import Image


class BaseImageClassifier(ABC):
    """Base class that all image classifiers must implement."""
    
    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image for prediction."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make predictions on the input image.
        
        Returns:
            Dict mapping class names to confidence scores.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata including performance metrics."""
        pass
    
    def predict_from_pil(self, pil_image: Image.Image) -> Dict[str, float]:
        """Convenience method to predict from PIL Image."""
        image_array = np.array(pil_image)
        return self.predict(image_array)
    
    def get_top_predictions(self, predictions: Dict[str, float], top_k: int = 3) -> List[tuple]:
        """Get top k predictions sorted by confidence."""
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:top_k]
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded