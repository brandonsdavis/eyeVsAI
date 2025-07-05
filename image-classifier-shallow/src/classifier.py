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

import numpy as np
import pickle
from typing import Dict, Any, Optional, List
import cv2

from base_classifier import BaseImageClassifier
from utils import ModelUtils
from .feature_extractor import FeatureExtractor
from .config import ShallowLearningConfig


class ShallowImageClassifier(BaseImageClassifier):
    """Memory-efficient shallow learning classifier implementing the base interface."""
    
    def __init__(self, model_name: str = "shallow-classifier", version: str = "1.0.0", 
                 config: Optional[ShallowLearningConfig] = None, class_names: Optional[List[str]] = None):
        super().__init__(model_name, version)
        self.config = config or ShallowLearningConfig()
        self.model = None
        self.feature_extractor = None
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names else None
        
    def build_model(self):
        """Build the model architecture based on configuration."""
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Load data first to determine classes.")
            
        # Model building is handled in the trainer
        # This method exists for consistency with other classifiers
        pass
        
    def load_model(self, model_path: str) -> None:
        """Load the trained model and feature extractor."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.class_names = model_data['class_names']
        self.num_classes = len(self.class_names)
        
        # Load config if saved
        if 'config' in model_data:
            self.config = model_data['config']
            
        self._is_loaded = True
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction."""
        # Resize to expected size
        image_resized = ModelUtils.resize_image(image, self.config.image_size)
        
        # Convert to RGB if needed
        if len(image_resized.shape) == 3 and image_resized.shape[2] == 4:
            image_resized = ModelUtils.convert_to_rgb(image_resized)
        elif len(image_resized.shape) == 2:
            # Convert grayscale to RGB
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
            
        # Ensure correct data type and range for feature extraction
        if image_resized.max() <= 1.0:
            image_resized = (image_resized * 255).astype(np.uint8)
            
        return image_resized
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make predictions on input image."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Preprocess image
        processed_image = self.preprocess(image)
        
        # Extract features using batch method (for single image)
        features = self.feature_extractor.extract_features_from_images([processed_image])
        
        # Scale features
        features_scaled = self.feature_extractor.scale_features(features, fit=False)
        
        # Apply PCA if available
        if self.feature_extractor.pca is not None:
            features_final = self.feature_extractor.pca.transform(features_scaled)
        else:
            features_final = features_scaled
            
        # Get predictions
        probabilities = self.model.predict_proba(features_final)[0]
        
        # Convert to class name mapping
        predictions = {}
        for i, prob in enumerate(probabilities):
            predictions[self.class_names[i]] = float(prob)
            
        return predictions
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_type": "shallow_learning",
            "algorithm": type(self.model).__name__ if self.model else "Unknown",
            "feature_extractor": {
                "basic_features": True,
                "histogram_features": True,
                "texture_features": True,
                "pca_components": self.feature_extractor.pca.n_components_ if self.feature_extractor and self.feature_extractor.pca else None
            },
            "config": self.config.to_dict(),
            "classes": self.class_names,
            "num_classes": self.num_classes,
            "version": self.version,
            "memory_efficient": True
        }
    
    def save_model(self, model_path: str, model=None, feature_extractor=None) -> None:
        """Save the trained model and feature extractor."""
        # Use provided model/extractor or use instance attributes
        model_to_save = model if model is not None else self.model
        extractor_to_save = feature_extractor if feature_extractor is not None else self.feature_extractor
        
        if model_to_save is None or extractor_to_save is None:
            raise ValueError("No model or feature extractor to save")
            
        model_data = {
            'model': model_to_save,
            'feature_extractor': extractor_to_save,
            'class_names': self.class_names,
            'config': self.config
        }
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {model_path}")