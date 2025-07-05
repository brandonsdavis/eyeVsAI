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

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple
import gc


class FeatureExtractor:
    """Extract features from images for shallow learning with batch processing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def extract_basic_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract basic statistical features from a batch of images."""
        features = []
        
        for img in images:
            img_features = []
            
            # Convert to grayscale for some features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Color statistics (RGB channels)
            for channel in range(3):
                channel_data = img[:, :, channel].flatten()
                img_features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
            
            # Grayscale statistics
            gray_flat = gray.flatten()
            img_features.extend([
                np.mean(gray_flat),
                np.std(gray_flat),
                np.var(gray_flat)
            ])
            
            # Edge detection features
            edges = cv2.Canny(gray, 50, 150)
            img_features.extend([
                np.sum(edges > 0) / edges.size,  # Edge density
                np.mean(edges),
                np.std(edges)
            ])
            
            features.append(img_features)
            
        return np.array(features)
    
    def extract_histogram_features_batch(self, images: List[np.ndarray], bins: int = 16) -> np.ndarray:
        """Extract color histogram features from a batch of images."""
        features = []
        
        for img in images:
            hist_features = []
            
            # Histogram for each color channel
            for channel in range(3):
                hist, _ = np.histogram(img[:, :, channel], bins=bins, range=(0, 256))
                hist = hist / np.sum(hist)  # Normalize
                hist_features.extend(hist)
                
            features.append(hist_features)
            
        return np.array(features)
    
    def extract_texture_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract texture features from a batch of images."""
        features = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Simple texture measures
            texture_features = []
            
            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            texture_features.extend([
                np.mean(gradient_mag),
                np.std(gradient_mag),
                np.percentile(gradient_mag, 90)
            ])
            
            # Local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
            
            texture_features.extend([
                np.mean(local_var),
                np.std(local_var)
            ])
            
            features.append(texture_features)
            
        return np.array(features)
    
    def extract_features_from_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract all features from a list of images."""
        # Extract different feature types
        basic_features = self.extract_basic_features_batch(images)
        hist_features = self.extract_histogram_features_batch(images)
        texture_features = self.extract_texture_features_batch(images)
        
        # Combine features
        features = np.hstack([basic_features, hist_features, texture_features])
        
        return features
    
    def extract_features_from_paths(self, image_paths: List[str], load_func, batch_size: int = 50) -> np.ndarray:
        """Extract features from image paths using batch processing."""
        all_features = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} images)")
            
            # Load batch of images using provided load function
            batch_images = load_func(batch_paths)
            
            # Extract features for this batch
            batch_features = self.extract_features_from_images(batch_images)
            all_features.append(batch_features)
            
            # Clean up batch images from memory
            del batch_images, batch_features
            gc.collect()
            
        # Combine all batch features
        final_features = np.vstack(all_features)
        print(f"Feature extraction complete. Shape: {final_features.shape}")
        
        return final_features
    
    def apply_pca(self, features: np.ndarray, n_components: int = 50) -> np.ndarray:
        """Apply PCA for dimensionality reduction."""
        self.pca = PCA(n_components=n_components)
        reduced_features = self.pca.fit_transform(features)
        
        print(f"PCA reduced features from {features.shape[1]} to {reduced_features.shape[1]} dimensions")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return reduced_features
    
    def scale_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale features using StandardScaler."""
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)