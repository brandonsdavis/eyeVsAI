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
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


def load_images_batch(image_paths: List[str], image_size: Tuple[int, int] = (64, 64)) -> List[np.ndarray]:
    """Load a batch of images from file paths."""
    images = []
    
    for img_path in image_paths:
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            images.append(img_array)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create blank image as fallback
            images.append(np.zeros((*image_size, 3), dtype=np.uint8))
            
    return images


def scan_dataset(dataset_path: str, max_classes: Optional[int] = None, 
                 max_images_per_class: Optional[int] = None) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan dataset directory and return image paths, labels, and class names.
    
    Args:
        dataset_path: Path to dataset directory with class subdirectories
        max_classes: Maximum number of classes to load (None for all)
        max_images_per_class: Maximum images per class (None for all)
        
    Returns:
        image_paths: List of image file paths
        labels: List of class indices
        class_names: List of class names
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
    # Get class directories
    all_class_dirs = [d for d in dataset_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
    
    if not all_class_dirs:
        raise ValueError(f"No class directories found in {dataset_path}")
        
    # Limit classes if specified
    class_dirs = sorted(all_class_dirs)[:max_classes] if max_classes else sorted(all_class_dirs)
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_dir in class_dirs:
        class_idx = class_to_idx[class_dir.name]
        
        # Get all valid image files
        image_files = []
        for ext in valid_extensions:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
        # Limit images per class if specified
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
            
        # Add to lists
        for img_file in image_files:
            image_paths.append(str(img_file))
            labels.append(class_idx)
            
    print(f"Total images found: {len(image_paths)}")
    
    return image_paths, labels, class_names


def prepare_data_splits(image_paths: List[str], labels: List[int], 
                       test_size: float = 0.2, val_size: float = 0.1, 
                       random_seed: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Split data into train, validation, and test sets.
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (paths, labels) tuples
    """
    # First split: train+val vs test
    paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_seed
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for the remaining data
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_train_val, labels_train_val, test_size=val_size_adjusted, 
        stratify=labels_train_val, random_state=random_seed
    )
    
    return {
        'train': (paths_train, labels_train),
        'val': (paths_val, labels_val),
        'test': (paths_test, labels_test)
    }


def load_data(data_path: str, config) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    """
    Load data and determine classes from directory structure.
    
    Returns:
        paths_train, labels_train, paths_val, labels_val, class_names
    """
    # Scan dataset to get paths and labels
    image_paths, labels, class_names = scan_dataset(
        data_path,
        max_classes=None,  # Load all classes
        max_images_per_class=None  # Load all images
    )
    
    # Split data
    splits = prepare_data_splits(
        image_paths, labels,
        test_size=config.test_split,
        val_size=config.validation_split,
        random_seed=config.random_seed
    )
    
    paths_train, labels_train = splits['train']
    paths_val, labels_val = splits['val']
    
    print(f"Data splits - Train: {len(paths_train)}, Val: {len(paths_val)}, Test: {len(splits['test'][0])}")
    
    return paths_train, labels_train, paths_val, labels_val, class_names