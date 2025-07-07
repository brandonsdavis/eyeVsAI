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
from typing import List, Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import random
import gc


class LazyUnifiedDataset(Dataset):
    """Ultra memory-efficient dataset with lazy loading and minimal memory footprint."""
    
    def __init__(self, dataset_info, subset_indices=None, transform=None, mixup_alpha=0.2):
        self.dataset_path = dataset_info['dataset_path']
        self.class_names = dataset_info['class_names']
        self.class_to_idx = dataset_info['class_to_idx']
        self.valid_extensions = dataset_info['valid_extensions']
        self.transform = transform
        self.mixup_alpha = mixup_alpha
        
        # Build image paths lazily only when needed
        self._image_paths = None
        self._labels = None
        self.subset_indices = subset_indices
        
        # Calculate total length without loading paths
        if subset_indices is not None:
            self._length = len(subset_indices)
        else:
            self._length = sum(dataset_info['class_counts'].values())
    
    def _load_paths_lazy(self):
        """Load image paths only when first accessed."""
        if self._image_paths is None:
            print("Loading image paths (first access)...")
            self._image_paths = []
            self._labels = []
            
            for class_dir in sorted(self.dataset_path.iterdir()):
                if not class_dir.is_dir() or class_dir.name.startswith('.'):
                    continue
                    
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    continue
                    
                class_idx = self.class_to_idx[class_name]
                
                # Load paths for this class
                image_files = [f for f in class_dir.iterdir() 
                              if f.suffix.lower() in self.valid_extensions]
                
                for img_path in image_files:
                    self._image_paths.append(str(img_path))
                    self._labels.append(class_idx)
            
            # Apply subset if specified
            if self.subset_indices is not None:
                self._image_paths = [self._image_paths[i] for i in self.subset_indices]
                self._labels = [self._labels[i] for i in self.subset_indices]
            
            print(f"Loaded {len(self._image_paths)} image paths")
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        # Lazy load paths on first access
        if self._image_paths is None:
            self._load_paths_lazy()
            
        image_path = self._image_paths[idx]
        label = self._labels[idx]
        
        try:
            # Load image from disk with memory-efficient handling
            with Image.open(image_path) as img:
                image = img.convert('RGB').copy()  # Copy to close file handle
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Create blank image if loading fails
            image = Image.new('RGB', (96, 96), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_sample_for_split(self, train_ratio=0.7, val_ratio=0.15, random_seed=42):
        """Create train/val/test splits without loading all data."""
        if self._image_paths is None:
            self._load_paths_lazy()
            
        total_samples = len(self._image_paths)
        indices = list(range(total_samples))
        
        # Shuffle for random splits
        random.seed(random_seed)
        random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices


def scan_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """
    Scan dataset directory for class information without loading all paths.
    
    Args:
        dataset_path: Path to dataset directory with class subdirectories
        
    Returns:
        Dictionary with dataset information
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
    
    print(f"Found {len(class_names)} classes")
    
    # Count images per class WITHOUT loading all paths into memory
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    total_images = 0
    class_counts = {}
    
    print("Counting images per class...")
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Count files without storing paths
        image_count = sum(1 for f in class_dir.iterdir() 
                         if f.suffix.lower() in valid_extensions)
        
        class_counts[class_name] = image_count
        total_images += image_count
        
        print(f"  {class_name}: {image_count} images")
    
    print(f"\nTotal images: {total_images}")
    
    # Store dataset info for later use
    dataset_info = {
        'dataset_path': dataset_path,
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'class_counts': class_counts,
        'total_images': total_images,
        'valid_extensions': valid_extensions
    }
    
    return dataset_info


def get_advanced_transforms(config):
    """Get advanced training and validation transforms."""
    transform_train = transforms.Compose([
        transforms.Resize((config.image_size[0], config.image_size[1])),
        transforms.RandomResizedCrop(config.image_size[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        transforms.RandomRotation(config.rotation_degrees),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        transforms.RandomErasing(
            p=config.random_erasing_prob, 
            scale=config.random_erasing_scale
        )
    ])

    transform_val = transforms.Compose([
        transforms.Resize((config.image_size[0], config.image_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
    ])
    
    return transform_train, transform_val


def create_memory_efficient_loaders(dataset_path: str, config) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create memory-efficient PyTorch data loaders with lazy loading.
    
    Args:
        dataset_path: Path to dataset directory
        config: Configuration object with training parameters
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Get dataset information without loading all paths
    dataset_info = scan_dataset_info(dataset_path)
    
    # Get transforms
    transform_train, transform_val = get_advanced_transforms(config)
    
    # Create lazy dataset - paths not loaded until first access
    print("Creating lazy dataset...")
    full_dataset = LazyUnifiedDataset(dataset_info, transform=transform_train)
    
    # Get splits without loading all data
    train_indices, val_indices, test_indices = full_dataset.get_sample_for_split(
        train_ratio=0.7, val_ratio=0.15, random_seed=config.random_seed
    )
    
    print(f"Dataset splits (calculated efficiently):")
    print(f"  Total images: {len(full_dataset)}")
    print(f"  Training: {len(train_indices)}")
    print(f"  Validation: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")
    
    # Create separate dataset instances for each split
    train_dataset = LazyUnifiedDataset(
        dataset_info, 
        subset_indices=train_indices,
        transform=transform_train,
        mixup_alpha=config.mixup_alpha
    )
    
    val_dataset = LazyUnifiedDataset(
        dataset_info, 
        subset_indices=val_indices,
        transform=transform_val
    )
    
    test_dataset = LazyUnifiedDataset(
        dataset_info, 
        subset_indices=test_indices,
        transform=transform_val
    )
    
    # Create data loaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=False
    )
    
    print(f"\nMemory-optimized DataLoaders created:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Pin memory: {config.pin_memory}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {len(dataset_info['class_names'])}")
    
    # Test loading a single batch to verify everything works
    print(f"\nTesting memory-efficient data loading...")
    try:
        sample_batch = next(iter(train_loader))
        print(f"✅ Successfully loaded batch: {sample_batch[0].shape}, {sample_batch[1].shape}")
        
        # Calculate approximate memory usage
        batch_memory_mb = (sample_batch[0].numel() * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"✅ Batch memory usage: ~{batch_memory_mb:.1f} MB")
        
        # Free the test batch
        del sample_batch
        
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
    
    # Aggressive memory cleanup
    gc.collect()
    
    return train_loader, val_loader, test_loader, dataset_info['class_names']


def mixup_data(x, y, alpha=0.2, device=None):
    """
    Apply mixup augmentation to a batch of data.
    
    Args:
        x: Input batch
        y: Labels batch  
        alpha: Mixup parameter
        device: Device to use
        
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)