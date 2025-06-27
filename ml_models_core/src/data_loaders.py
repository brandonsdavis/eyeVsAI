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

"""
Common data loaders for all image classification models.
Provides consistent data loading interfaces across different model types.
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .data_manager import get_dataset_manager


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    X_train: np.ndarray
    X_val: np.ndarray  
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    class_names: List[str]
    class_to_idx: Dict[str, int]


class BaseImageDataset:
    """Base class for image datasets with common functionality."""
    
    def __init__(self, 
                 dataset_path: str,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Scan for images and build class mapping
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._scan_dataset()
        
    def _scan_dataset(self):
        """Scan dataset directory and build image/label lists."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Collect all class directories
        class_dirs = [d for d in self.dataset_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {self.dataset_path}")
        
        self.class_names = sorted([d.name for d in class_dirs])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Collect all images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} images across {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize
            if self.image_size:
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1]
            if self.normalize:
                image_array = image_array / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            shape = (*self.image_size, 3) if self.image_size else (224, 224, 3)
            return np.zeros(shape, dtype=np.float32)
    
    def create_data_split(self, 
                         test_size: float = 0.2,
                         val_size: float = 0.2,
                         random_state: int = 42,
                         stratify: bool = True) -> DataSplit:
        """Create train/validation/test splits."""
        
        # Load all images
        print("Loading images...")
        X = np.array([self.load_image(path) for path in self.image_paths])
        y = np.array(self.labels)
        
        # Stratify if requested and possible
        stratify_y = y if stratify else None
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        stratify_trainval = y_trainval if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_trainval
        )
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return DataSplit(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            class_names=self.class_names,
            class_to_idx=self.class_to_idx
        )


# PyTorch-specific implementations
if PYTORCH_AVAILABLE:
    class PyTorchImageDataset(Dataset):
        """PyTorch Dataset for image classification."""
        
        def __init__(self, 
                     image_paths: List[str],
                     labels: List[int],
                     transform: Optional[Callable] = None,
                     image_size: Tuple[int, int] = (224, 224)):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            self.image_size = image_size
            
            # Default transform if none provided
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                # Create blank image
                image = Image.new('RGB', self.image_size, (0, 0, 0))
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    
    def create_pytorch_dataloaders(dataset_path: str,
                                 batch_size: int = 32,
                                 image_size: Tuple[int, int] = (224, 224),
                                 test_size: float = 0.2,
                                 val_size: float = 0.2,
                                 num_workers: int = 4,
                                 augment_train: bool = True) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders for train/val/test splits."""
        
        # Load dataset
        base_dataset = BaseImageDataset(dataset_path, image_size=image_size)
        
        # Create splits (using image paths, not loaded images for PyTorch)
        X_paths = base_dataset.image_paths
        y = base_dataset.labels
        
        # Split the paths and labels
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_paths, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, 
            random_state=42, stratify=y_trainval
        )
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(0.5) if augment_train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if augment_train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if augment_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = PyTorchImageDataset(X_train, y_train, train_transform, image_size)
        val_dataset = PyTorchImageDataset(X_val, y_val, val_test_transform, image_size)
        test_dataset = PyTorchImageDataset(X_test, y_test, val_test_transform, image_size)
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=num_workers, pin_memory=True)
        }
        
        print(f"PyTorch DataLoaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples") 
        print(f"  Test: {len(test_dataset)} samples")
        print(f"  Classes: {base_dataset.class_names}")
        
        return dataloaders, base_dataset.class_names, base_dataset.class_to_idx


# TensorFlow-specific implementations  
if TENSORFLOW_AVAILABLE:
    def create_tensorflow_datasets(dataset_path: str,
                                 batch_size: int = 32,
                                 image_size: Tuple[int, int] = (224, 224),
                                 test_size: float = 0.2,
                                 val_size: float = 0.2,
                                 augment_train: bool = True):
        """Create TensorFlow datasets for train/val/test splits."""
        
        # Load and split data
        base_dataset = BaseImageDataset(dataset_path, image_size=image_size)
        data_split = base_dataset.create_data_split(test_size, val_size)
        
        # Convert to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            data_split.X_train, data_split.y_train
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            data_split.X_val, data_split.y_val
        ))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            data_split.X_test, data_split.y_test
        ))
        
        # Apply data augmentation to training set if requested
        if augment_train:
            def augment_fn(image, label):
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_contrast(image, 0.8, 1.2)
                return image, label
            
            train_dataset = train_dataset.map(augment_fn)
        
        # Configure datasets for performance
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        print(f"TensorFlow datasets created:")
        print(f"  Train: {len(data_split.X_train)} samples")
        print(f"  Val: {len(data_split.X_val)} samples")
        print(f"  Test: {len(data_split.X_test)} samples")
        print(f"  Classes: {data_split.class_names}")
        
        return datasets, data_split.class_names, data_split.class_to_idx


# Scikit-learn compatible data loading
def create_sklearn_data(dataset_path: str,
                       image_size: Tuple[int, int] = (224, 224),
                       test_size: float = 0.2,
                       val_size: float = 0.2,
                       flatten: bool = True) -> DataSplit:
    """Create scikit-learn compatible data splits."""
    
    base_dataset = BaseImageDataset(dataset_path, image_size=image_size)
    data_split = base_dataset.create_data_split(test_size, val_size)
    
    # Flatten images for traditional ML models if requested
    if flatten:
        original_shape = data_split.X_train.shape
        data_split.X_train = data_split.X_train.reshape(original_shape[0], -1)
        data_split.X_val = data_split.X_val.reshape(original_shape[0], -1) 
        data_split.X_test = data_split.X_test.reshape(original_shape[0], -1)
        
        print(f"Flattened image shape: {data_split.X_train.shape[1]} features")
    
    return data_split


# Convenience functions for getting standard datasets
def get_main_classification_data(framework: str = 'sklearn', **kwargs):
    """Get the main pets vs vegetables classification dataset."""
    manager = get_dataset_manager()
    
    # Ensure the combined dataset exists
    try:
        dataset_path = manager.get_dataset_path('combined_pets_vs_vegetables')
        if not dataset_path:
            print("Creating combined pets vs vegetables dataset...")
            from .data_manager import create_combined_classification_dataset
            dataset_path = create_combined_classification_dataset()
    except Exception as e:
        print(f"Error creating combined dataset: {e}")
        # Fallback to individual datasets
        dataset_path = manager.download_dataset('oxford_pets')
    
    # Return data in requested format
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_pets_data(framework: str = 'sklearn', **kwargs):
    """Get Oxford pets dataset."""
    manager = get_dataset_manager()
    dataset_path = manager.download_dataset('oxford_pets')
    
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_vegetables_data(framework: str = 'sklearn', **kwargs):
    """Get Kaggle vegetables dataset."""
    manager = get_dataset_manager()
    dataset_path = manager.download_dataset('kaggle_vegetables')
    
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_street_foods_data(framework: str = 'sklearn', **kwargs):
    """Get street foods dataset."""
    manager = get_dataset_manager()
    dataset_path = manager.download_dataset('street_foods')
    
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_musical_instruments_data(framework: str = 'sklearn', **kwargs):
    """Get musical instruments dataset."""
    manager = get_dataset_manager()
    dataset_path = manager.download_dataset('musical_instruments')
    
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_unified_classification_data(framework: str = 'sklearn', **kwargs):
    """Get unified dataset combining all classes from all image datasets."""
    manager = get_dataset_manager()
    
    # Ensure the unified dataset exists
    try:
        dataset_path = manager.get_dataset_path('combined_unified_classification')
        if not dataset_path:
            print("Creating unified classification dataset...")
            available_datasets = ['oxford_pets', 'kaggle_vegetables', 'street_foods', 'musical_instruments']
            dataset_path = manager.create_combined_dataset(
                dataset_names=available_datasets,
                output_name="unified_classification",
                class_mapping=None  # Keep original class names
            )
    except Exception as e:
        print(f"Error accessing unified dataset: {e}")
        # Fallback to main classification dataset
        return get_main_classification_data(framework, **kwargs)
    
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_three_class_data(framework: str = 'sklearn', **kwargs):
    """Get three-class combined dataset: pets vs vegetables vs street foods."""
    manager = get_dataset_manager()
    
    # Ensure the three-class combined dataset exists
    try:
        dataset_path = manager.get_dataset_path('combined_pets_vs_vegetables_vs_foods')
        if not dataset_path:
            print("Creating three-class combined dataset...")
            from .data_manager import create_three_class_classification_dataset
            dataset_path = create_three_class_classification_dataset()
    except Exception as e:
        print(f"Error creating three-class dataset: {e}")
        # Fallback to binary dataset
        return get_main_classification_data(framework, **kwargs)
    
    # Return data in requested format
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)


def get_four_class_data(framework: str = 'sklearn', **kwargs):
    """Get four-class combined dataset: pets vs vegetables vs street foods vs musical instruments."""
    manager = get_dataset_manager()
    
    # Ensure the four-class combined dataset exists
    try:
        dataset_path = manager.get_dataset_path('combined_pets_vs_vegetables_vs_foods_vs_instruments')
        if not dataset_path:
            print("Creating four-class combined dataset...")
            from .data_manager import create_four_class_classification_dataset
            dataset_path = create_four_class_classification_dataset()
    except Exception as e:
        print(f"Error creating four-class dataset: {e}")
        # Fallback to three-class dataset
        return get_three_class_data(framework, **kwargs)
    
    # Return data in requested format
    if framework.lower() == 'pytorch' and PYTORCH_AVAILABLE:
        return create_pytorch_dataloaders(dataset_path, **kwargs)
    elif framework.lower() == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return create_tensorflow_datasets(dataset_path, **kwargs)
    else:
        return create_sklearn_data(dataset_path, **kwargs)