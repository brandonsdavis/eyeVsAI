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

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, List, Optional
import numpy as np
from pathlib import Path
import logging
import os

from .config import TransferLearningClassifierConfig

logger = logging.getLogger(__name__)


def create_pytorch_datasets(
    data_path: str,
    config: TransferLearningClassifierConfig,
    transforms_dict: Dict[str, transforms.Compose]
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Optional[torch.Tensor]]:
    """
    Create PyTorch data loaders for training, validation, and testing.
    
    Args:
        data_path: Path to dataset directory
        config: Configuration object
        transforms_dict: Dictionary of transforms for train/val/test
        
    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    # Filter out empty directories by creating a custom dataset
    class FilteredImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            """Override to filter out empty directories."""
            classes = []
            class_to_idx = {}
            valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
            
            for idx, target_class in enumerate(sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())):
                class_dir = os.path.join(directory, target_class)
                # Check if directory contains any valid images
                has_images = any(
                    Path(os.path.join(class_dir, fname)).suffix.lower() in valid_extensions
                    for fname in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, fname))
                )
                if has_images:
                    classes.append(target_class)
                    if target_class not in class_to_idx:
                        class_to_idx[target_class] = len(class_to_idx)
            
            return classes, class_to_idx
    
    # Create dataset
    full_dataset = FilteredImageFolder(
        root=data_path,
        transform=transforms_dict['train']
    )
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {class_names}")
    
    # Calculate dataset splits
    total_size = len(full_dataset)
    val_size = int(config.validation_split * total_size)
    test_size = int(config.test_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # Update transforms for validation and test sets
    val_dataset.dataset.transform = transforms_dict['val']
    test_dataset.dataset.transform = transforms_dict['test']
    
    # Calculate class weights if requested
    class_weights = None
    if config.class_weights:
        class_counts = torch.zeros(num_classes)
        for _, label in full_dataset:
            class_counts[label] += 1
        
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes
        logger.info("Calculated class weights for imbalanced dataset")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    logger.info(f"Batch counts - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_names, class_weights


class InMemoryDataset(Dataset):
    """Dataset that loads all images into memory for faster training."""
    
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        self.classes = []
        self.class_to_idx = {}
        
        # Load all images
        dataset = datasets.ImageFolder(root=data_path)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        
        logger.info("Loading dataset into memory...")
        for idx, (image, label) in enumerate(dataset):
            self.data.append(image)
            self.targets.append(label)
            
            if idx % 1000 == 0:
                logger.info(f"Loaded {idx}/{len(dataset)} images")
        
        logger.info(f"Loaded {len(self.data)} images into memory")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target