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
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core" / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
import json
import logging

from base_classifier import BaseImageClassifier
from utils import ModelUtils

logger = logging.getLogger(__name__)


class TransferLearningModelPyTorch(nn.Module):
    """PyTorch Transfer Learning model for image classification that matches the actual training architecture."""
    
    def __init__(self, base_model_name: str, num_classes: int, dense_units: List[int], 
                 head_dropout_rate: float = 0.5, pretrained: bool = True):
        super().__init__()
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        
        # Create base model to match the training architecture exactly
        if base_model_name == "vgg16":
            # VGG16 architecture matching the training structure
            self.base_model = models.vgg16(pretrained=pretrained)
            # Keep features, remove classifier
            self.base_model.classifier = nn.Identity()
            backbone_output_size = 25088  # VGG16 feature output size
            
        elif base_model_name in ["resnet50", "resnet101"]:
            # ResNet architecture 
            if base_model_name == "resnet50":
                self.base_model = models.resnet50(pretrained=pretrained)
            else:
                self.base_model = models.resnet101(pretrained=pretrained)
            backbone_output_size = self.base_model.fc.in_features
            # Remove the final classification layer
            self.base_model.fc = nn.Identity()
            
        elif base_model_name == "densenet121":
            self.base_model = models.densenet121(pretrained=pretrained)
            backbone_output_size = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
            
        elif base_model_name == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            backbone_output_size = self.base_model.classifier[1].in_features
            # Remove the final classification layer
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Build classification head (called "head" to match training architecture)
        head_layers = []
        input_size = backbone_output_size
        
        for i, units in enumerate(dense_units):
            head_layers.extend([
                nn.Linear(input_size, units),
                nn.BatchNorm1d(units),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout_rate)
            ])
            input_size = units
        
        # Final classification layer
        head_layers.append(nn.Linear(input_size, num_classes))
        
        self.head = nn.Sequential(*head_layers)
        
        # Global average pooling for VGG
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, x):
        # Extract features from base model
        features = self.base_model.features(x) if hasattr(self.base_model, 'features') else self.base_model(x)
        
        # Handle different architecture outputs
        if self.base_model_name == "vgg16":
            # VGG outputs need to be pooled and flattened
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif self.base_model_name in ["resnet50", "resnet101"]:
            # ResNet outputs are already pooled
            features = features.view(features.size(0), -1) if len(features.shape) > 2 else features
        elif self.base_model_name == "densenet121":
            # DenseNet needs global pooling
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        elif self.base_model_name == "efficientnet_b0":
            # EfficientNet is already handled by the model
            features = features.view(features.size(0), -1) if len(features.shape) > 2 else features
        
        # Apply classification head
        return self.head(features)


class PyTorchTransferLearningClassifier(BaseImageClassifier):
    """PyTorch-based Transfer Learning classifier implementing the base interface."""
    
    def __init__(self, model_name: str = "pytorch-transfer-learning", version: str = "1.0.0"):
        super().__init__(model_name, version)
        self.model = None
        self.class_names = None
        self.num_classes = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define transforms for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path: str) -> None:
        """Load the trained PyTorch model."""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract metadata
            self.config = checkpoint.get('config', {})
            self.class_names = checkpoint.get('class_names', [])
            self.num_classes = len(self.class_names)
            input_size = checkpoint.get('input_size', (224, 224))
            
            # Create model architecture
            base_model = self.config.get('base_model', 'resnet50')
            dense_units = self.config.get('dense_units', [512, 256])
            head_dropout_rate = self.config.get('head_dropout_rate', 0.5)
            
            self.model = TransferLearningModelPyTorch(
                base_model_name=base_model,
                num_classes=self.num_classes,
                dense_units=dense_units,
                head_dropout_rate=head_dropout_rate,
                pretrained=False  # We're loading weights, don't need pretrained
            )
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Update transforms based on input size
            if input_size != (224, 224):
                self.transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            self._is_loaded = True
            logger.info(f"PyTorch transfer learning model loaded successfully from {model_path}")
            logger.info(f"Model: {base_model}, Classes: {self.num_classes}, Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model from {model_path}: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for prediction."""
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Handle different image formats
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
        
        # Convert to PIL Image
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
        
        # Handle PIL Image input
        if isinstance(image, Image.Image):
            image = np.array(image)
        
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
            predictions[self.class_names[i]] = float(prob.cpu().numpy())
        
        return predictions
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Make predictions on a batch of images."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess all images
        tensor_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)
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
                predictions[self.class_names[i]] = float(prob.cpu().numpy())
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        metadata = {
            "model_type": "transfer_learning",
            "framework": "PyTorch",
            "architecture": f"Transfer Learning with {self.config.get('base_model', 'unknown')}",
            "input_size": f"{self.config.get('input_size', (224, 224))}",
            "features": [
                "transfer_learning",
                "pre_trained_backbone",
                "two_phase_training",
                "fine_tuning"
            ],
            "base_model": self.config.get('base_model', 'unknown'),
            "classes": self.class_names,
            "num_classes": self.num_classes,
            "version": self.version,
            "device": str(self.device),
            "config": self.config
        }
        
        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            metadata.update({
                "parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Rough estimate for float32
            })
        
        return metadata
    
    def save_model(self, model_path: str, model=None, class_names=None, accuracy=None, training_history=None) -> None:
        """Save the trained model."""
        # This method is primarily for compatibility with the base interface
        # In practice, PyTorch models are saved during training with torch.save()
        if model is None:
            model = self.model
        
        if class_names is None:
            class_names = self.class_names
        
        if model is None or class_names is None:
            raise ValueError("No model or class names to save")
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': class_names,
            'config': self.config or {},
            'input_size': self.config.get('input_size', (224, 224)) if self.config else (224, 224),
            'version': self.version,
            'framework': 'pytorch'
        }
        
        if accuracy is not None:
            checkpoint['accuracy'] = accuracy
        
        if training_history is not None:
            checkpoint['training_history'] = training_history
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, model_path)
        logger.info(f"PyTorch transfer learning model saved to {model_path}")