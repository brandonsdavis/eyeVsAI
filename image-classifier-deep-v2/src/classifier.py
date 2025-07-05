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

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, Optional, List

from base_classifier import BaseImageClassifier
from utils import ModelUtils
from .model import DeepLearningV2
from .config import DeepLearningV2Config


class DeepLearningV2Classifier(BaseImageClassifier):
    """Deep Learning v2 classifier implementing the base interface with advanced features."""
    
    def __init__(self, model_name: str = "deep-learning-v2", version: str = "2.0.0", 
                 config: Optional[DeepLearningV2Config] = None, class_names: Optional[List[str]] = None):
        super().__init__(model_name, version)
        self.config = config or DeepLearningV2Config()
        self.model = None
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size[0], self.config.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
        ])
    
    def build_model(self):
        """Build the model architecture based on configuration."""
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Load data first to determine classes.")
        
        self.model = DeepLearningV2(
            num_classes=self.num_classes,
            input_channels=self.config.input_channels,
            dropout_rates=self.config.dropout_rates,
            attention_reduction=self.config.attention_reduction_ratio,
            spatial_kernel=self.config.spatial_attention_kernel,
            residual_dropout=self.config.residual_dropout
        ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode by default
    
    def load_model(self, model_path: str) -> None:
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get class names and model config
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        
        # Load model configuration if available
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            num_classes = model_config.get('num_classes', self.num_classes)
            input_size = model_config.get('input_size', self.config.image_size)
            
            # Update config if needed
            if input_size != self.config.image_size:
                self.config.image_size = input_size
                # Update transform with new image size
                self.transform = transforms.Compose([
                    transforms.Resize((input_size[0], input_size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
                ])
        
        # Recreate model architecture
        self.model = DeepLearningV2(
            num_classes=self.num_classes,
            input_channels=self.config.input_channels,
            dropout_rates=self.config.dropout_rates,
            attention_reduction=self.config.attention_reduction_ratio,
            spatial_kernel=self.config.spatial_attention_kernel,
            residual_dropout=self.config.residual_dropout
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for prediction."""
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Handle different image formats
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
        
        pil_image = Image.fromarray(image)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply transforms
        tensor_image = self.transform(pil_image)
        
        return tensor_image
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make predictions on input image."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        tensor_image = self.preprocess(image)
        tensor_image = tensor_image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor_image)
            probabilities = F.softmax(outputs, dim=1)
        
        # Convert to class name mapping
        predictions = {}
        for i, prob in enumerate(probabilities[0]):
            predictions[self.class_names[i]] = float(prob.cpu())
        
        return predictions
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Make predictions on a batch of images."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess all images
        tensor_images = []
        for image in images:
            tensor_image = self.preprocess(image)
            tensor_images.append(tensor_image)
        
        # Stack into batch
        batch_tensor = torch.stack(tensor_images).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Convert to list of class name mappings
        batch_predictions = []
        for prob_vector in probabilities:
            predictions = {}
            for i, prob in enumerate(prob_vector):
                predictions[self.class_names[i]] = float(prob.cpu())
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def get_feature_maps(self, image: np.ndarray, layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Extract feature maps from intermediate layers."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        tensor_image = self.preprocess(image)
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        # Extract feature maps
        with torch.no_grad():
            feature_maps = self.model.get_feature_maps(tensor_image, layer_names)
        
        # Convert to numpy arrays
        numpy_features = {}
        for layer_name, features in feature_maps.items():
            numpy_features[layer_name] = features.cpu().numpy()
        
        return numpy_features
    
    def get_attention_weights(self, image: np.ndarray) -> Dict[str, float]:
        """Get attention weights for analysis."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        tensor_image = self.preprocess(image)
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(tensor_image)
        
        return attention_weights
    
    def analyze_confidence(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze model confidence on a batch of images."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        batch_predictions = self.predict_batch(images)
        
        # Calculate confidence statistics
        confidences = []
        max_probs = []
        entropy_scores = []
        
        for predictions in batch_predictions:
            probs = list(predictions.values())
            max_prob = max(probs)
            
            # Calculate entropy as uncertainty measure
            probs_array = np.array(probs)
            entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
            
            confidences.append(max_prob)
            max_probs.append(max_prob)
            entropy_scores.append(entropy)
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'mean_entropy': np.mean(entropy_scores),
            'predictions': batch_predictions
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        metadata = {
            "model_type": "deep_learning_v2",
            "architecture": "Advanced CNN with ResNet + Attention",
            "input_size": f"{self.config.image_size[0]}x{self.config.image_size[1]}x{self.config.input_channels}",
            "features": [
                "residual_connections", 
                "self_attention", 
                "channel_attention", 
                "spatial_attention",
                "mixup_augmentation", 
                "label_smoothing",
                "gradient_accumulation",
                "advanced_regularization"
            ],
            "classes": self.class_names,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "version": self.version,
            "config": self.config.to_dict()
        }
        
        if self.model is not None:
            model_info = self.model.get_model_info()
            metadata.update({
                "parameters": model_info["total_parameters"],
                "trainable_parameters": model_info["trainable_parameters"],
                "model_size_mb": model_info["model_size_mb"],
                "layers": model_info["layers"],
                "advanced_features": model_info["features"]
            })
        
        return metadata
    
    def save_model(self, model_path: str, model=None, class_names=None, accuracy=None, training_history=None) -> None:
        """Save the trained model."""
        # Use provided values or instance attributes
        model_to_save = model if model is not None else self.model
        class_names_to_save = class_names if class_names is not None else self.class_names
        
        if model_to_save is None:
            raise ValueError("No model to save")
        
        if class_names_to_save is None:
            raise ValueError("No class names to save")
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'class_names': class_names_to_save,
            'model_config': {
                'num_classes': len(class_names_to_save),
                'input_size': self.config.image_size,
                'architecture': 'DeepLearningV2',
                'input_channels': self.config.input_channels,
                'dropout_rates': self.config.dropout_rates,
                'attention_reduction': self.config.attention_reduction_ratio,
                'spatial_kernel': self.config.spatial_attention_kernel,
                'residual_dropout': self.config.residual_dropout,
                'features': [
                    'residual_connections', 
                    'attention_mechanisms', 
                    'advanced_training',
                    'memory_efficient_loading'
                ]
            },
            'config': self.config.to_dict()
        }
        
        # Add optional fields
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        if training_history is not None:
            checkpoint['training_history'] = training_history
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, model_path)
        print(f"Advanced model saved to {model_path}")