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

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
import pickle
import gc

from base_trainer import BaseTrainer
from .classifier import ShallowImageClassifier
from .feature_extractor import FeatureExtractor
from .data_loader import load_data, load_images_batch
from .config import ShallowLearningConfig


class ShallowLearningTrainer(BaseTrainer):
    """Trainer for shallow learning models."""
    
    def __init__(self, model: ShallowImageClassifier, config: ShallowLearningConfig):
        super().__init__(model, config.to_dict())
        self.config = config  # Keep typed config
        self.feature_extractor = FeatureExtractor(n_jobs=4)  # Use 4 CPU cores for parallel processing
        
    def train(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Train the shallow learning model.
        
        Args:
            data_path: Path to the training data directory
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        # Load data paths and discover classes
        self.logger.info(f"Loading data from {data_path}")
        paths_train, labels_train, paths_val, labels_val, class_names = load_data(
            data_path, self.config
        )
        
        # Update model with discovered classes
        self.model.class_names = class_names
        self.model.num_classes = len(class_names)
        self.logger.info(f"Discovered {self.model.num_classes} classes")
        
        # Extract features from training data
        self.logger.info("Extracting features from training data...")
        train_features = self._extract_features(paths_train, labels_train)
        
        # Scale features
        self.logger.info("Scaling features...")
        train_features_scaled = self.feature_extractor.scale_features(train_features, fit=True)
        
        # Apply dimensionality reduction if configured
        if self.config.use_pca:
            self.logger.info(f"Applying PCA with {self.config.n_components_pca} components...")
            train_features_final = self.feature_extractor.apply_pca(
                train_features_scaled, n_components=self.config.n_components_pca
            )
        elif self.config.n_components_svd > 0:
            self.logger.info(f"Applying SVD with {self.config.n_components_svd} components...")
            svd = TruncatedSVD(n_components=min(self.config.n_components_svd, train_features_scaled.shape[1] - 1))
            train_features_final = svd.fit_transform(train_features_scaled)
            self.feature_extractor.svd = svd  # Store for later use
        else:
            train_features_final = train_features_scaled
            
        # Train the model
        self.logger.info("Training classifier...")
        classifier = self._create_classifier()
        classifier.fit(train_features_final, labels_train)
        
        # Store in model
        self.model.model = classifier
        self.model.feature_extractor = self.feature_extractor
        
        # Validate
        self.logger.info("Validating model...")
        val_features = self._extract_features(paths_val, labels_val)
        val_features_scaled = self.feature_extractor.scale_features(val_features, fit=False)
        
        if self.config.use_pca and self.feature_extractor.pca is not None:
            val_features_final = self.feature_extractor.pca.transform(val_features_scaled)
        elif hasattr(self.feature_extractor, 'svd') and self.feature_extractor.svd is not None:
            val_features_final = self.feature_extractor.svd.transform(val_features_scaled)
        else:
            val_features_final = val_features_scaled
            
        val_predictions = classifier.predict(val_features_final)
        val_accuracy = accuracy_score(labels_val, val_predictions)
        
        # Log metrics
        metrics = {
            'train_samples': len(labels_train),
            'val_samples': len(labels_val),
            'val_accuracy': val_accuracy,
            'num_features': train_features_final.shape[1],
            'num_classes': self.model.num_classes
        }
        self.log_metrics(metrics, phase="final")
        
        return {
            'model': classifier,
            'feature_extractor': self.feature_extractor,
            'metrics': metrics,
            'class_names': class_names
        }
        
    def _extract_features(self, paths: List[str], labels: List[int]) -> np.ndarray:
        """Extract features from image paths."""
        # Define load function for the feature extractor
        def load_func(batch_paths):
            return load_images_batch(batch_paths, image_size=self.config.image_size)
            
        # Extract features in batches
        features = self.feature_extractor.extract_features_from_paths(
            paths, load_func, batch_size=self.config.batch_size
        )
        
        return features
        
    def _create_classifier(self):
        """Create the classifier based on configuration."""
        if self.config.svm_kernel:
            # SVM is the default
            return SVC(
                kernel=self.config.svm_kernel,
                C=self.config.svm_C,
                gamma=self.config.svm_gamma,
                probability=True,
                random_state=self.config.random_seed
            )
        else:
            # Fallback to Random Forest
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_seed,
                n_jobs=-1
            )
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model.model is None:
            raise ValueError("Model not trained yet")
            
        # Make predictions
        predictions = self.model.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            'test_accuracy': accuracy,
            'test_samples': len(y_test)
        }
        
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.model,
            'feature_extractor': self.feature_extractor,
            'metrics': metrics,
            'config': self.config,
            'class_names': self.model.class_names
        }
        
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        self.model.model = checkpoint['model']
        self.feature_extractor = checkpoint['feature_extractor']
        self.model.feature_extractor = self.feature_extractor
        self.model.class_names = checkpoint['class_names']
        self.model.num_classes = len(checkpoint['class_names'])
        
        return checkpoint