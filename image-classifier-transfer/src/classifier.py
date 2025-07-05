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

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
import json

from base_classifier import BaseImageClassifier
from utils import ModelUtils
from .models import TransferLearningModel
from .config import TransferLearningClassifierConfig


class TransferLearningClassifier(BaseImageClassifier):
    """Transfer learning classifier implementing the base interface."""
    
    def __init__(self, model_name: str = "transfer-learning", version: str = "1.0.0", 
                 config: Optional[TransferLearningClassifierConfig] = None, 
                 class_names: Optional[List[str]] = None):
        super().__init__(model_name, version)
        self.config = config or TransferLearningClassifierConfig()
        self.model = None
        self.tf_model = None
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names else None
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        """Configure TensorFlow settings."""
        # Enable mixed precision if requested
        if self.config.mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and self.config.memory_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        # Enable XLA compilation
        if self.config.use_xla:
            tf.config.optimizer.set_jit(True)
    
    def build_model(self):
        """Build the model architecture based on configuration."""
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Load data first to determine classes.")
        
        self.tf_model = TransferLearningModel(self.config, self.num_classes)
        self.model = self.tf_model.build_model()
    
    def load_model(self, model_path: str) -> None:
        """Load the trained model."""
        model_path = Path(model_path)
        
        if model_path.suffix == '.h5' or model_path.suffix == '.keras':
            # Load Keras model directly
            self.model = keras.models.load_model(model_path)
            
            # Try to load metadata if available
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get('class_names', [])
                    self.num_classes = len(self.class_names)
                    
                    # Update config if available
                    if 'config' in metadata:
                        config_dict = metadata['config']
                        for key, value in config_dict.items():
                            if hasattr(self.config, key):
                                setattr(self.config, key, value)
            
        else:
            # Load from checkpoint format
            with open(model_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Load model
            model_path_h5 = checkpoint_data.get('model_path')
            if model_path_h5 and Path(model_path_h5).exists():
                self.model = keras.models.load_model(model_path_h5)
                self.class_names = checkpoint_data.get('class_names', [])
                self.num_classes = len(self.class_names)
                
                # Update config
                if 'config' in checkpoint_data:
                    config_dict = checkpoint_data['config']
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
        
        if self.model is None:
            raise ValueError(f"Failed to load model from {model_path}")
        
        # Create TensorFlow model wrapper
        if self.tf_model is None and self.num_classes:
            self.tf_model = TransferLearningModel(self.config, self.num_classes)
            self.tf_model.model = self.model
        
        self._is_loaded = True
    
    def preprocess(self, image: np.ndarray) -> tf.Tensor:
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
        
        # Resize to model input size
        pil_image = pil_image.resize(self.config.image_size)
        
        # Convert to tensor and normalize
        tensor_image = tf.convert_to_tensor(np.array(pil_image), dtype=tf.float32)
        tensor_image = tensor_image / 255.0  # Normalize to [0, 1]
        
        return tensor_image
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Make predictions on input image."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        tensor_image = self.preprocess(image)
        tensor_image = tf.expand_dims(tensor_image, 0)  # Add batch dimension
        
        # Make prediction
        predictions = self.model(tensor_image, training=False)
        probabilities = tf.nn.softmax(predictions, axis=1)
        
        # Convert to class name mapping
        predictions_dict = {}
        for i, prob in enumerate(probabilities[0]):
            predictions_dict[self.class_names[i]] = float(prob.numpy())
        
        return predictions_dict
    
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
        batch_tensor = tf.stack(tensor_images)
        
        # Make predictions
        predictions = self.model(batch_tensor, training=False)
        probabilities = tf.nn.softmax(predictions, axis=1)
        
        # Convert to list of class name mappings
        batch_predictions = []
        for prob_vector in probabilities:
            predictions_dict = {}
            for i, prob in enumerate(prob_vector):
                predictions_dict[self.class_names[i]] = float(prob.numpy())
            batch_predictions.append(predictions_dict)
        
        return batch_predictions
    
    def get_feature_maps(self, image: np.ndarray, layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Extract feature maps from intermediate layers."""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        tensor_image = self.preprocess(image)
        tensor_image = tf.expand_dims(tensor_image, 0)
        
        # Create feature extraction model
        if layer_names is None:
            # Extract from all layers
            layer_names = [layer.name for layer in self.model.layers]
        
        # Get layers
        layers = []
        for name in layer_names:
            try:
                layer = self.model.get_layer(name)
                layers.append(layer.output)
            except ValueError:
                print(f"Layer {name} not found, skipping...")
                continue
        
        if not layers:
            return {}
        
        # Create feature extraction model
        feature_extractor = keras.Model(inputs=self.model.input, outputs=layers)
        
        # Extract features
        features = feature_extractor(tensor_image)
        
        # Convert to numpy arrays
        if not isinstance(features, list):
            features = [features]
        
        feature_maps = {}
        for i, feature_map in enumerate(features):
            if i < len(layer_names):
                feature_maps[layer_names[i]] = feature_map.numpy()
        
        return feature_maps
    
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
            "model_type": "transfer_learning",
            "architecture": f"Transfer Learning with {self.config.base_model_name}",
            "input_size": f"{self.config.image_size[0]}x{self.config.image_size[1]}x3",
            "features": [
                "transfer_learning",
                "pre_trained_backbone",
                "data_augmentation",
                "two_phase_training",
                "fine_tuning"
            ],
            "base_model": self.config.base_model_name,
            "classes": self.class_names,
            "num_classes": self.num_classes,
            "framework": "TensorFlow/Keras",
            "version": self.version,
            "config": self.config.to_dict()
        }
        
        if self.model is not None:
            metadata.update({
                "parameters": int(self.model.count_params()),
                "trainable_parameters": int(sum(tf.size(p) for p in self.model.trainable_variables)),
                "model_size_mb": self.model.count_params() * 4 / (1024 * 1024),  # Rough estimate
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape
            })
        
        if self.config.mixed_precision:
            metadata["features"].append("mixed_precision")
        
        if self.config.use_xla:
            metadata["features"].append("xla_compilation")
        
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
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if model_path.suffix in ['.h5', '.keras']:
            model_to_save.save(model_path)
        else:
            model_path = model_path.with_suffix('.h5')
            model_to_save.save(model_path)
        
        # Save metadata
        metadata = {
            'class_names': class_names_to_save,
            'config': self.config.to_dict(),
            'model_type': 'transfer_learning',
            'base_model': self.config.base_model_name,
            'framework': 'tensorflow',
            'version': self.version
        }
        
        if accuracy is not None:
            metadata['accuracy'] = accuracy
        
        if training_history is not None:
            metadata['training_history'] = training_history
        
        # Save metadata file
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Transfer learning model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")