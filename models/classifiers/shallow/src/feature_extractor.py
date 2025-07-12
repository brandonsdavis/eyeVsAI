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
import multiprocessing as mp
from functools import partial


class FeatureExtractor:
    """Extract features from images for shallow learning with batch processing."""
    
    def __init__(self, n_jobs: int = -1):
        self.scaler = StandardScaler()
        self.pca = None
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
    def extract_basic_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract basic statistical features from a batch of images."""
        if not images:
            return np.array([])
            
        # Convert to numpy array for vectorized operations
        imgs_array = np.array(images)  # Shape: (batch_size, height, width, 3)
        batch_size = imgs_array.shape[0]
        
        # Pre-allocate feature array (18 color features + 3 gray features + 3 edge features = 24)
        features = np.zeros((batch_size, 24))
        
        # Vectorized color statistics for all images at once
        for channel in range(3):
            channel_data = imgs_array[:, :, :, channel].reshape(batch_size, -1)
            start_idx = channel * 6
            
            features[:, start_idx:start_idx+6] = np.column_stack([
                np.mean(channel_data, axis=1),
                np.std(channel_data, axis=1),
                np.min(channel_data, axis=1),
                np.max(channel_data, axis=1),
                np.percentile(channel_data, 25, axis=1),
                np.percentile(channel_data, 75, axis=1)
            ])
        
        # Vectorized grayscale conversion and statistics
        gray_imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])
        gray_flat = gray_imgs.reshape(batch_size, -1)
        
        features[:, 18:21] = np.column_stack([
            np.mean(gray_flat, axis=1),
            np.std(gray_flat, axis=1),
            np.var(gray_flat, axis=1)
        ])
        
        # Edge detection (still needs individual processing due to cv2.Canny)
        for i, gray in enumerate(gray_imgs):
            edges = cv2.Canny(gray, 50, 150)
            features[i, 21:24] = [
                np.sum(edges > 0) / edges.size,
                np.mean(edges),
                np.std(edges)
            ]
            
        return features
    
    def extract_histogram_features_batch(self, images: List[np.ndarray], bins: int = 16) -> np.ndarray:
        """Extract color histogram features from a batch of images."""
        if not images:
            return np.array([])
            
        batch_size = len(images)
        features = np.zeros((batch_size, bins * 3))  # 3 channels * bins per channel
        
        # Vectorized histogram computation
        imgs_array = np.array(images)
        
        for channel in range(3):
            channel_data = imgs_array[:, :, :, channel]
            
            # Compute histograms for all images in the batch at once
            for i in range(batch_size):
                hist, _ = np.histogram(channel_data[i], bins=bins, range=(0, 256))
                hist = hist / (np.sum(hist) + 1e-8)  # Normalize with small epsilon
                start_idx = channel * bins
                features[i, start_idx:start_idx+bins] = hist
                
        return features
    
    def extract_texture_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract texture features from a batch of images."""
        if not images:
            return np.array([])
            
        batch_size = len(images)
        features = np.zeros((batch_size, 5))  # 3 gradient + 2 local variance features
        
        # Pre-compute kernel for local variance
        kernel = np.ones((5, 5), np.float32) / 25
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Gradient magnitude computation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            features[i, 0:3] = [
                np.mean(gradient_mag),
                np.std(gradient_mag),
                np.percentile(gradient_mag, 90)
            ]
            
            # Local variance computation
            gray_float = gray.astype(np.float32)
            local_mean = cv2.filter2D(gray_float, -1, kernel)
            local_var = cv2.filter2D((gray_float - local_mean)**2, -1, kernel)
            
            features[i, 3:5] = [
                np.mean(local_var),
                np.std(local_var)
            ]
            
        return features
    
    def extract_features_from_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract all features from a list of images."""
        # Extract different feature types
        basic_features = self.extract_basic_features_batch(images)
        hist_features = self.extract_histogram_features_batch(images)
        texture_features = self.extract_texture_features_batch(images)
        
        # Combine features
        features = np.hstack([basic_features, hist_features, texture_features])
        
        return features
    
    def _process_batch(self, batch_info):
        """Process a single batch of images - for multiprocessing."""
        batch_paths, load_func = batch_info
        batch_images = load_func(batch_paths)
        return self.extract_features_from_images(batch_images)
    
    def extract_features_from_paths(self, image_paths: List[str], load_func, batch_size: int = 50) -> np.ndarray:
        """Extract features from image paths using batch processing with optional multiprocessing."""
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        # Create batches
        batches = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batches.append((batch_paths, load_func))
        
        print(f"Processing {total_batches} batches with {self.n_jobs} workers")
        
        # Use multiprocessing if more than 1 job and enough batches
        if self.n_jobs > 1 and len(batches) > 1:
            try:
                with mp.Pool(processes=min(self.n_jobs, len(batches))) as pool:
                    all_features = pool.map(self._process_batch, batches)
            except Exception as e:
                print(f"Multiprocessing failed ({e}), falling back to sequential processing")
                all_features = []
                for i, batch_info in enumerate(batches):
                    print(f"Processing batch {i+1}/{total_batches}")
                    batch_features = self._process_batch(batch_info)
                    all_features.append(batch_features)
                    gc.collect()
        else:
            # Sequential processing
            all_features = []
            for i, batch_info in enumerate(batches):
                print(f"Processing batch {i+1}/{total_batches}")
                batch_features = self._process_batch(batch_info)
                all_features.append(batch_features)
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