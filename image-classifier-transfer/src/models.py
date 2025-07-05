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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from typing import Tuple, Dict, Any, List, Optional
import numpy as np

from .config import TransferLearningClassifierConfig


class TransferLearningModel:
    """Transfer learning model using pre-trained networks."""
    
    def __init__(self, config: TransferLearningClassifierConfig, num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        """Configure TensorFlow settings for optimal performance."""
        # Enable mixed precision if requested
        if self.config.mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and self.config.memory_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        # Enable XLA compilation
        if self.config.use_xla:
            tf.config.optimizer.set_jit(True)
            print("XLA compilation enabled")
    
    def _get_base_model(self) -> keras.Model:
        """Get the pre-trained base model."""
        input_shape = (*self.config.image_size, 3)
        
        base_models = {
            'resnet50': applications.ResNet50,
            'vgg16': applications.VGG16,
            'efficientnet_b0': applications.EfficientNetB0,
            'efficientnet_b1': applications.EfficientNetB1,
            'mobilenet_v2': applications.MobileNetV2,
            'inception_v3': applications.InceptionV3
        }
        
        if self.config.base_model_name not in base_models:
            raise ValueError(f"Unsupported base model: {self.config.base_model_name}")
        
        base_model_class = base_models[self.config.base_model_name]
        
        # Adjust input shape for InceptionV3
        if self.config.base_model_name == 'inception_v3':
            input_shape = (299, 299, 3)
        
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        base_model.trainable = self.config.base_trainable
        
        print(f"Base model: {self.config.base_model_name}")
        print(f"Input shape: {input_shape}")
        print(f"Base trainable: {self.config.base_trainable}")
        print(f"Base model parameters: {base_model.count_params():,}")
        
        return base_model
    
    def build_model(self) -> keras.Model:
        """Build the complete transfer learning model."""
        # Get base model
        self.base_model = self._get_base_model()
        
        # Build the classifier head
        inputs = keras.Input(shape=(*self.config.image_size, 3))
        
        # Preprocessing layers
        x = layers.Rescaling(self.config.rescale)(inputs)
        
        # Data augmentation during training
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Base model
        x = self.base_model(x, training=False)
        
        # Global pooling
        if self.config.global_pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif self.config.global_pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
        else:
            raise ValueError(f"Unsupported pooling: {self.config.global_pooling}")
        
        # Dense layers
        for units in self.config.dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.config.head_dropout_rate)(x)
        
        # Output layer
        if self.config.final_activation == "softmax":
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation=self.config.final_activation, dtype='float32')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self._compile_model()
        
        return self.model
    
    def _compile_model(self):
        """Compile the model with appropriate optimizer and loss."""
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', 'sparse_top_k_categorical_accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"Model compiled with {loss} loss")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum(tf.size(p) for p in self.model.trainable_variables):,}")
    
    def prepare_for_fine_tuning(self):
        """Prepare model for fine-tuning by unfreezing top layers."""
        if self.base_model is None:
            raise ValueError("Model must be built before fine-tuning")
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Freeze all layers except the top ones
        total_layers = len(self.base_model.layers)
        freeze_layers = max(0, total_layers - self.config.fine_tune_layers)
        
        for layer in self.base_model.layers[:freeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        optimizer = keras.optimizers.Adam(learning_rate=self.config.fine_tune_learning_rate)
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', 'sparse_top_k_categorical_accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        trainable_params = sum(tf.size(p) for p in self.model.trainable_variables)
        
        print(f"Fine-tuning preparation complete:")
        print(f"  Unfrozen layers: {self.config.fine_tune_layers}")
        print(f"  Frozen layers: {freeze_layers}")
        print(f"  Fine-tune learning rate: {self.config.fine_tune_learning_rate}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def get_callbacks(self, log_dir: str = "logs") -> List[keras.callbacks.Callback]:
        """Get training callbacks."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"{log_dir}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        if log_dir:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            )
        
        return callbacks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        if self.model is None:
            return {}
        
        return {
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum(tf.size(p) for p in self.model.trainable_variables),
            "base_model": self.config.base_model_name,
            "num_classes": self.num_classes,
            "input_shape": (*self.config.image_size, 3),
            "architecture": "Transfer Learning",
            "framework": "TensorFlow/Keras",
            "features": [
                "transfer_learning",
                "pre_trained_backbone",
                "data_augmentation",
                "mixed_precision" if self.config.mixed_precision else None,
                "fine_tuning_support"
            ]
        }
    
    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet")