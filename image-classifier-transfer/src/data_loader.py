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
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import os

from .config import TransferLearningClassifierConfig


def discover_classes(dataset_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Discover classes from dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory with class subdirectories
        
    Returns:
        class_names, class_to_idx mapping
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Get class directories and filter out empty ones
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    class_dirs = []
    
    for d in dataset_path.iterdir():
        if d.is_dir() and not d.name.startswith('.'):
            # Count images in directory
            image_count = sum(1 for f in d.iterdir() 
                            if f.suffix.lower() in valid_extensions)
            if image_count > 0:  # Only include directories with images
                class_dirs.append(d)
    
    if not class_dirs:
        raise ValueError(f"No class directories with images found in {dataset_path}")
    
    class_names = sorted([d.name for d in class_dirs])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Discovered {len(class_names)} classes: {class_names}")
    
    # Count images per class
    total_images = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_count = sum(1 for f in class_dir.iterdir() 
                         if f.suffix.lower() in valid_extensions)
        total_images += image_count
        print(f"  {class_name}: {image_count} images")
    
    print(f"Total images: {total_images}")
    
    return class_names, class_to_idx


def compute_class_weights(dataset_path: str, class_names: List[str]) -> Dict[int, float]:
    """
    Compute class weights for handling class imbalance.
    
    Args:
        dataset_path: Path to dataset directory
        class_names: List of class names
        
    Returns:
        Dictionary mapping class indices to weights
    """
    dataset_path = Path(dataset_path)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    class_counts = {}
    total_images = 0
    
    for i, class_name in enumerate(class_names):
        class_dir = dataset_path / class_name
        if class_dir.exists():
            count = sum(1 for f in class_dir.iterdir() 
                       if f.suffix.lower() in valid_extensions)
            class_counts[i] = count
            total_images += count
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_images / (len(class_names) * count)
    
    print("Class weights calculated:")
    for i, weight in class_weights.items():
        print(f"  {class_names[i]}: {weight:.3f}")
    
    return class_weights


def create_dataset_from_directory(
    dataset_path: str,
    config: TransferLearningClassifierConfig,
    class_names: List[str],
    subset: str = "training",
    validation_split: Optional[float] = None
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        config: Configuration object
        class_names: List of class names
        subset: "training", "validation", or None
        validation_split: Validation split ratio
        
    Returns:
        TensorFlow dataset
    """
    if validation_split is None:
        validation_split = config.validation_split
    
    # Use tf.keras.utils.image_dataset_from_directory for efficient loading
    dataset = keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',
        label_mode='int',
        class_names=class_names,
        color_mode='rgb',
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        seed=config.random_seed,
        validation_split=validation_split,
        subset=subset if validation_split and validation_split > 0 else None
    )
    
    return dataset


def apply_preprocessing(
    dataset: tf.data.Dataset, 
    config: TransferLearningClassifierConfig,
    is_training: bool = True
) -> tf.data.Dataset:
    """
    Apply preprocessing and augmentation to dataset.
    
    Args:
        dataset: Input dataset
        config: Configuration object
        is_training: Whether this is training data
        
    Returns:
        Preprocessed dataset
    """
    # Normalization
    normalization_layer = keras.layers.Rescaling(config.rescale)
    dataset = dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=config.parallel_calls
    )
    
    # Data augmentation for training
    if is_training:
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ])
        
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=config.parallel_calls
        )
    
    return dataset


def create_memory_efficient_datasets(
    dataset_path: str, 
    config: TransferLearningClassifierConfig
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], Dict[int, float]]:
    """
    Create memory-efficient TensorFlow datasets for training, validation, and test.
    
    Args:
        dataset_path: Path to dataset directory
        config: Configuration object
        
    Returns:
        train_dataset, val_dataset, test_dataset, class_names, class_weights
    """
    print(f"Creating memory-efficient TensorFlow datasets from {dataset_path}")
    
    # Discover classes
    class_names, class_to_idx = discover_classes(dataset_path)
    
    # Compute class weights if requested
    class_weights = None
    if config.class_weights:
        class_weights = compute_class_weights(dataset_path, class_names)
    
    # Create training and validation datasets
    train_dataset = create_dataset_from_directory(
        dataset_path, config, class_names, subset="training", 
        validation_split=config.validation_split
    )
    
    val_dataset = create_dataset_from_directory(
        dataset_path, config, class_names, subset="validation", 
        validation_split=config.validation_split
    )
    
    # For test dataset, we'll use a separate split or a portion of validation
    # Create a simple test dataset by taking a portion of validation
    val_size = tf.data.experimental.cardinality(val_dataset).numpy()
    test_size = max(1, val_size // 3)  # Take 1/3 of validation as test
    
    test_dataset = val_dataset.take(test_size)
    val_dataset = val_dataset.skip(test_size)
    
    # Apply preprocessing
    train_dataset = apply_preprocessing(train_dataset, config, is_training=True)
    val_dataset = apply_preprocessing(val_dataset, config, is_training=False)
    test_dataset = apply_preprocessing(test_dataset, config, is_training=False)
    
    # Performance optimizations
    if config.cache_dataset:
        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.cache()
        test_dataset = test_dataset.cache()
    
    # Prefetch for performance
    train_dataset = train_dataset.prefetch(config.prefetch_buffer)
    val_dataset = val_dataset.prefetch(config.prefetch_buffer)
    test_dataset = test_dataset.prefetch(config.prefetch_buffer)
    
    # Calculate dataset sizes
    train_size = tf.data.experimental.cardinality(train_dataset).numpy()
    val_size = tf.data.experimental.cardinality(val_dataset).numpy()
    test_size = tf.data.experimental.cardinality(test_dataset).numpy()
    
    print(f"\nDataset splits created:")
    print(f"  Training batches: {train_size}")
    print(f"  Validation batches: {val_size}")
    print(f"  Test batches: {test_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Image size: {config.image_size}")
    
    # Test loading a batch
    print(f"\nTesting data loading...")
    try:
        sample_batch = next(iter(train_dataset))
        print(f"✅ Successfully loaded batch: {sample_batch[0].shape}, {sample_batch[1].shape}")
        print(f"✅ Data types: {sample_batch[0].dtype}, {sample_batch[1].dtype}")
        print(f"✅ Value ranges: images [{sample_batch[0].numpy().min():.3f}, {sample_batch[0].numpy().max():.3f}]")
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
    
    return train_dataset, val_dataset, test_dataset, class_names, class_weights


def create_inference_dataset(
    images_path: str,
    config: TransferLearningClassifierConfig,
    class_names: List[str]
) -> tf.data.Dataset:
    """
    Create dataset for inference on new images.
    
    Args:
        images_path: Path to directory with images for inference
        config: Configuration object
        class_names: List of class names
        
    Returns:
        TensorFlow dataset for inference
    """
    dataset = keras.utils.image_dataset_from_directory(
        images_path,
        labels=None,
        class_names=class_names,
        color_mode='rgb',
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False
    )
    
    # Apply only normalization (no augmentation for inference)
    normalization_layer = keras.layers.Rescaling(config.rescale)
    dataset = dataset.map(
        lambda x: normalization_layer(x),
        num_parallel_calls=config.parallel_calls
    )
    
    dataset = dataset.prefetch(config.prefetch_buffer)
    
    return dataset