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
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split


class UnifiedDataset(Dataset):
    """Memory-efficient dataset wrapper for unified classification data."""
    
    def __init__(self, image_paths, labels, class_names, transform=None):
        self.image_paths = image_paths  # Store paths instead of loaded images
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image on-demand
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image from disk
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Create blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(config):
    """Get training and validation transforms."""
    transform_train = transforms.Compose([
        transforms.Resize((config.image_size[0], config.image_size[1])),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        transforms.RandomRotation(config.rotation_degrees),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((config.image_size[0], config.image_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
    ])
    
    return transform_train, transform_val


def scan_dataset(dataset_path: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan dataset directory and return image paths, labels, and class names.
    
    Args:
        dataset_path: Path to dataset directory with class subdirectories
        
    Returns:
        image_paths: List of image file paths
        labels: List of class indices
        class_names: List of class names
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Get class directories
    class_dirs = [d for d in dataset_path.iterdir() 
                 if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {dataset_path}")
    
    class_names = sorted([d.name for d in class_dirs])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in valid_extensions]
        
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"\nCollected {len(image_paths)} image paths")
    
    return image_paths, labels, class_names


def create_data_splits(image_paths: List[str], labels: List[int], 
                      train_ratio: float = 0.7, val_ratio: float = 0.15, 
                      random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train, validation, and test splits from indices.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices
    """
    # Create index splits
    indices = list(range(len(image_paths)))
    labels_array = np.array(labels)
    
    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, stratify=labels_array, random_state=random_seed
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_ratio_adjusted, 
        stratify=labels_array[train_val_indices], random_state=random_seed
    )
    
    print(f"Dataset splits:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Training: {len(train_indices)}")
    print(f"  Validation: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def create_data_loaders(dataset_path: str, config) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch data loaders for training, validation, and testing.
    
    Args:
        dataset_path: Path to dataset directory
        config: Configuration object with training parameters
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Scan dataset
    image_paths, labels, class_names = scan_dataset(dataset_path)
    
    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits(
        image_paths, labels, 
        train_ratio=0.7, val_ratio=0.15, 
        random_seed=config.random_seed
    )
    
    # Get transforms
    transform_train, transform_val = get_transforms(config)
    
    # Create datasets
    train_dataset = UnifiedDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        class_names,
        transform=transform_train
    )
    
    val_dataset = UnifiedDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        class_names,
        transform=transform_val
    )
    
    test_dataset = UnifiedDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        class_names,
        transform=transform_val
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"DataLoaders created with batch_size={config.batch_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    return train_loader, val_loader, test_loader, class_names