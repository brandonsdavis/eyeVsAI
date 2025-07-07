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
from PIL import Image
import hashlib
import os
from typing import Tuple, List


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        pil_image = Image.fromarray(image)
        resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
        """Normalize image with mean and standard deviation."""
        image_float = image.astype(np.float32) / 255.0
        
        if mean is not None and std is not None:
            mean = np.array(mean)
            std = np.array(std)
            image_float = (image_float - mean) / std
        
        return image_float
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def validate_image_format(image: np.ndarray) -> bool:
        """Validate image format and dimensions."""
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False
        
        return True
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert image to RGB format."""
        if len(image.shape) == 2:  # Grayscale
            return np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            return image[:, :, :3]
        else:  # Already RGB
            return image