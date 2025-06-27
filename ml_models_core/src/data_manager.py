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
Unified data management system for image classification project.
Handles downloading, preprocessing, and loading of multiple datasets.
"""

import os
import re
import json
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging

import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    source_type: str  # 'kaggle', 'url', 'oxford', 'local'
    source_url: str
    local_path: str
    description: str
    num_classes: int
    class_names: List[str]
    image_extensions: List[str] = None
    preprocessing_config: Dict = None
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if self.preprocessing_config is None:
            self.preprocessing_config = {
                'resize': (224, 224),
                'normalize': True,
                'augmentation': False
            }


class DatasetDownloader:
    """Handles downloading datasets from various sources."""
    
    def __init__(self, cache_dir: str = "data/downloads"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_dataset(self, dataset_id: str, target_dir: str) -> str:
        """Download dataset from Kaggle using kagglehub."""
        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        try:
            download_path = kagglehub.dataset_download(dataset_id)
            target_path = self.cache_dir / target_dir
            
            if target_path.exists():
                shutil.rmtree(target_path)
            
            shutil.copytree(download_path, target_path)
            logger.info(f"Kaggle dataset downloaded to: {target_path}")
            return str(target_path)
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset {dataset_id}: {e}")
            raise
    
    def download_oxford_pets(self, target_dir: str) -> str:
        """Download Oxford-IIIT Pet Dataset."""
        logger.info("Downloading Oxford-IIIT Pet Dataset")
        target_path = self.cache_dir / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        
        # URLs for Oxford pets dataset
        images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        
        try:
            # Download images
            images_file = target_path / "images.tar.gz"
            if not images_file.exists():
                logger.info("Downloading pet images...")
                self._download_with_progress(images_url, images_file)
                
                # Extract images
                with tarfile.open(images_file, 'r:gz') as tar:
                    tar.extractall(target_path)
                
                # Clean up tar file
                images_file.unlink()
            
            # Download annotations
            annotations_file = target_path / "annotations.tar.gz"
            if not annotations_file.exists():
                logger.info("Downloading pet annotations...")
                self._download_with_progress(annotations_url, annotations_file)
                
                # Extract annotations
                with tarfile.open(annotations_file, 'r:gz') as tar:
                    tar.extractall(target_path)
                
                # Clean up tar file
                annotations_file.unlink()
            
            logger.info(f"Oxford pets dataset downloaded to: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Failed to download Oxford pets dataset: {e}")
            raise
    
    def download_from_url(self, url: str, target_dir: str, extract: bool = True) -> str:
        """Download and optionally extract dataset from URL."""
        target_path = self.cache_dir / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        
        filename = url.split('/')[-1]
        filepath = target_path / filename
        
        try:
            if not filepath.exists():
                logger.info(f"Downloading from {url}")
                self._download_with_progress(url, filepath)
            
            if extract and (filepath.suffix in ['.zip', '.tar', '.gz']):
                logger.info(f"Extracting {filepath}")
                if filepath.suffix == '.zip':
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(target_path)
                elif '.tar' in filepath.suffixes:
                    with tarfile.open(filepath, 'r:*') as tar:
                        tar.extractall(target_path)
                
                # Remove archive after extraction
                filepath.unlink()
            
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Failed to download from {url}: {e}")
            raise
    
    def download_kaggle_pandas_dataset(self, dataset_id: str, target_dir: str) -> str:
        """Download dataset from Kaggle using pandas adapter."""
        logger.info(f"Downloading Kaggle dataset with pandas adapter: {dataset_id}")
        target_path = self.cache_dir / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load the dataset using pandas adapter
            # Set empty file_path to get the whole dataset
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                dataset_id,
                "",  # Empty file path to load the whole dataset
            )
            
            logger.info(f"Loaded pandas dataset with {len(df)} records")
            
            # Determine dataset type and create appropriate filename
            if 'street-foods' in dataset_id or 'food' in dataset_id.lower():
                csv_filename = "street_foods_data.csv"
                dataset_type = "street_foods"
            elif 'music-instruments' in dataset_id or 'instrument' in dataset_id.lower():
                csv_filename = "musical_instruments_data.csv"
                dataset_type = "musical_instruments"
            else:
                csv_filename = "dataset_data.csv"
                dataset_type = "unknown"
            
            # Save the dataframe as CSV for reference
            csv_path = target_path / csv_filename
            df.to_csv(csv_path, index=False)
            
            # Extract image information - handle different column naming patterns
            metadata = {
                'dataset_info': {
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'dataset_type': dataset_type
                },
                'image_urls': []
            }
            
            # Handle street foods dataset
            if dataset_type == "street_foods" and 'image_url' in df.columns and 'food_name' in df.columns:
                logger.info("Processing street foods images...")
                metadata['dataset_info']['categories'] = df['food_name'].unique().tolist()
                metadata['image_urls'] = df[['food_name', 'image_url']].to_dict('records')
                logger.info(f"Found {len(metadata['dataset_info']['categories'])} food categories")
                
            # Handle musical instruments dataset
            elif dataset_type == "musical_instruments":
                logger.info("Processing musical instruments data...")
                # Try common column names for instruments
                instrument_col = None
                image_col = None
                
                for col in df.columns:
                    if 'instrument' in col.lower() or 'name' in col.lower():
                        instrument_col = col
                    if 'image' in col.lower() or 'url' in col.lower():
                        image_col = col
                
                if instrument_col:
                    metadata['dataset_info']['categories'] = df[instrument_col].unique().tolist()
                    logger.info(f"Found {len(metadata['dataset_info']['categories'])} instrument categories")
                    
                    if image_col:
                        metadata['image_urls'] = df[[instrument_col, image_col]].rename(
                            columns={instrument_col: 'category', image_col: 'image_url'}
                        ).to_dict('records')
                    else:
                        # If no image URLs, create placeholder structure
                        metadata['image_urls'] = [
                            {'category': cat, 'image_url': None} 
                            for cat in metadata['dataset_info']['categories']
                        ]
                else:
                    logger.warning(f"Could not identify instrument category column in: {list(df.columns)}")
                    metadata['dataset_info']['categories'] = []
            
            else:
                logger.warning(f"Unknown dataset structure for {dataset_id}. Columns: {list(df.columns)}")
                metadata['dataset_info']['categories'] = []
            
            # Save metadata
            metadata_path = target_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Dataset metadata saved to: {metadata_path}")
            
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Failed to download Kaggle pandas dataset {dataset_id}: {e}")
            raise
    
    def _download_with_progress(self, url: str, filepath: Path):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


class ImageDatasetProcessor:
    """Processes and organizes image datasets."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def process_oxford_pets(self, raw_path: str, output_path: str) -> Dict:
        """Process Oxford pets dataset into organized structure."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_dir = raw_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Extract labels and organize images
        class_counts = {}
        image_paths = []
        labels = []
        
        for image_file in images_dir.glob("*"):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Extract breed name from filename (everything before last underscore)
                match = re.match(r"(.+)_\d+\.(jpg|jpeg|png)", image_file.name, re.IGNORECASE)
                if match:
                    breed = match.group(1)
                    
                    # Create breed directory
                    breed_dir = output_path / breed
                    breed_dir.mkdir(exist_ok=True)
                    
                    # Copy image to breed directory
                    target_file = breed_dir / image_file.name
                    if not target_file.exists():
                        shutil.copy2(image_file, target_file)
                    
                    image_paths.append(str(target_file))
                    labels.append(breed)
                    class_counts[breed] = class_counts.get(breed, 0) + 1
        
        logger.info(f"Processed Oxford pets: {len(image_paths)} images, {len(class_counts)} breeds")
        
        return {
            'num_images': len(image_paths),
            'num_classes': len(class_counts),
            'class_names': sorted(class_counts.keys()),
            'class_counts': class_counts,
            'output_path': str(output_path)
        }
    
    def process_kaggle_vegetables(self, raw_path: str, output_path: str) -> Dict:
        """Process Kaggle vegetable dataset."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # The vegetable dataset is already organized by class
        vegetable_images_dir = raw_path / "Vegetable Images"
        if not vegetable_images_dir.exists():
            # Try alternative structure
            vegetable_images_dir = raw_path
        
        class_counts = {}
        total_images = 0
        
        # Copy organized structure
        for split in ['train', 'test', 'validation']:
            split_dir = vegetable_images_dir / split
            if split_dir.exists():
                output_split_dir = output_path / split
                output_split_dir.mkdir(exist_ok=True)
                
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        output_class_dir = output_split_dir / class_name
                        output_class_dir.mkdir(exist_ok=True)
                        
                        count = 0
                        for image_file in class_dir.glob("*"):
                            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                target_file = output_class_dir / image_file.name
                                if not target_file.exists():
                                    shutil.copy2(image_file, target_file)
                                count += 1
                        
                        class_counts[class_name] = class_counts.get(class_name, 0) + count
                        total_images += count
        
        logger.info(f"Processed vegetables: {total_images} images, {len(class_counts)} classes")
        
        return {
            'num_images': total_images,
            'num_classes': len(class_counts),
            'class_names': sorted(class_counts.keys()),
            'class_counts': class_counts,
            'output_path': str(output_path)
        }
    
    def process_street_foods(self, raw_path: str, output_path: str) -> Dict:
        """Process street foods dataset with metadata."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_path = raw_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Street foods metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract food categories
        food_categories = metadata['dataset_info'].get('food_categories', [])
        
        # Create a simplified dataset structure
        # Since this dataset likely contains image URLs rather than local images,
        # we'll create a reference structure with the metadata
        
        # Create category directories for reference
        class_counts = {}
        for category in food_categories:
            category_dir = output_path / category.replace(' ', '_').replace('/', '_')
            category_dir.mkdir(exist_ok=True)
            
            # Count images for this category from metadata
            category_images = [
                item for item in metadata.get('image_urls', [])
                if item.get('food_name') == category
            ]
            
            class_counts[category] = len(category_images)
            
            # Save image URLs for this category
            if category_images:
                urls_file = category_dir / "image_urls.json"
                with open(urls_file, 'w') as f:
                    json.dump(category_images, f, indent=2)
        
        # Save complete metadata in output directory
        shutil.copy2(metadata_path, output_path / "metadata.json")
        
        total_images = sum(class_counts.values())
        
        logger.info(f"Processed street foods: {total_images} image references, {len(class_counts)} categories")
        
        return {
            'num_images': total_images,
            'num_classes': len(class_counts),
            'class_names': sorted(class_counts.keys()),
            'class_counts': class_counts,
            'output_path': str(output_path),
            'is_url_dataset': True  # Flag to indicate this dataset contains URLs, not local images
        }
    
    def process_pandas_dataset(self, raw_path: str, output_path: str, dataset_name: str) -> Dict:
        """Generic processor for pandas-based datasets (street foods, musical instruments, etc.)."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_path = raw_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract categories
        categories = metadata['dataset_info'].get('categories', [])
        
        # Create category directories for reference
        class_counts = {}
        for category in categories:
            # Clean category name for directory
            clean_category = category.replace(' ', '_').replace('/', '_').replace('\\', '_')
            category_dir = output_path / clean_category
            category_dir.mkdir(exist_ok=True)
            
            # Count items for this category from metadata
            category_items = [
                item for item in metadata.get('image_urls', [])
                if (item.get('category') == category or 
                    item.get('food_name') == category or
                    item.get('instrument_name') == category)
            ]
            
            class_counts[clean_category] = len(category_items)
            
            # Save URLs/metadata for this category
            if category_items:
                urls_file = category_dir / "image_urls.json"
                with open(urls_file, 'w') as f:
                    json.dump(category_items, f, indent=2)
        
        # Save complete metadata in output directory
        shutil.copy2(metadata_path, output_path / "metadata.json")
        
        total_items = sum(class_counts.values())
        
        logger.info(f"Processed {dataset_name}: {total_items} item references, {len(class_counts)} categories")
        
        return {
            'num_images': total_items,
            'num_classes': len(class_counts),
            'class_names': sorted(class_counts.keys()),
            'class_counts': class_counts,
            'output_path': str(output_path),
            'is_url_dataset': True  # Flag to indicate this dataset contains URLs, not local images
        }
    
    def process_kaggle_street_foods(self, raw_path: str, output_path: str) -> Dict:
        """Process street foods image dataset from Kaggle."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Navigate to the actual dataset directory
        dataset_path = raw_path / "popular_street_foods" / "dataset"
        
        if not dataset_path.exists():
            logger.error(f"Street foods dataset path not found: {dataset_path}")
            return self._process_generic_image_dataset(raw_path, output_path, "street_foods")
        
        return self._process_organized_image_dataset(dataset_path, output_path, "street_foods")
    
    def process_kaggle_musical_instruments(self, raw_path: str, output_path: str) -> Dict:
        """Process musical instruments image dataset from Kaggle."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Navigate to the actual dataset directory
        dataset_path = raw_path / "music_instruments"
        
        if not dataset_path.exists():
            logger.error(f"Musical instruments dataset path not found: {dataset_path}")
            return self._process_generic_image_dataset(raw_path, output_path, "musical_instruments")
        
        return self._process_organized_image_dataset(dataset_path, output_path, "musical_instruments")
    
    def _process_organized_image_dataset(self, dataset_path: Path, output_path: Path, dataset_name: str) -> Dict:
        """Process properly organized image dataset with class subdirectories."""
        class_counts = {}
        total_images = 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        logger.info(f"Processing organized {dataset_name} dataset from {dataset_path}")
        
        # Iterate through class directories
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Clean class name (handle special characters)
                clean_class_name = class_name.replace('(', '').replace(')', '').replace(' ', '_')
                
                # Create output directory for this class
                output_class_dir = output_path / clean_class_name
                output_class_dir.mkdir(exist_ok=True)
                
                # Count and copy images
                image_count = 0
                for image_file in class_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in valid_extensions:
                        # Copy image to output directory
                        target_file = output_class_dir / image_file.name
                        if not target_file.exists():
                            shutil.copy2(image_file, target_file)
                        image_count += 1
                
                if image_count > 0:
                    class_counts[clean_class_name] = image_count
                    total_images += image_count
                    logger.info(f"Processed {class_name} -> {clean_class_name}: {image_count} images")
        
        logger.info(f"Processed {dataset_name}: {total_images} images, {len(class_counts)} classes")
        
        return {
            'num_images': total_images,
            'num_classes': len(class_counts),
            'class_names': sorted(class_counts.keys()),
            'class_counts': class_counts,
            'output_path': str(output_path)
        }
    
    def _process_generic_image_dataset(self, raw_path: str, output_path: str, dataset_name: str) -> Dict:
        """Generic processor for image datasets organized in folders."""
        raw_path = Path(raw_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        class_counts = {}
        total_images = 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Look for images in the raw directory structure
        for item in raw_path.rglob('*'):
            if item.is_file() and item.suffix.lower() in valid_extensions:
                # Try to determine class from directory structure
                # Look for parent directory that's not the raw_path itself
                class_name = None
                for parent in item.parents:
                    if parent != raw_path and parent.parent == raw_path:
                        class_name = parent.name
                        break
                
                # If no clear class structure, use parent directory name
                if not class_name:
                    class_name = item.parent.name
                
                # Skip if class name is generic/unclear
                if class_name.lower() in ['images', 'data', 'dataset', raw_path.name.lower()]:
                    # Try using grandparent or look for meaningful directory
                    possible_parents = [p.name for p in item.parents if p != raw_path]
                    if possible_parents:
                        class_name = possible_parents[0]
                    else:
                        class_name = 'unknown'
                
                # Clean class name
                class_name = class_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                
                # Create class directory in output
                class_dir = output_path / class_name
                class_dir.mkdir(exist_ok=True)
                
                # Copy image to class directory
                target_file = class_dir / item.name
                if not target_file.exists():
                    shutil.copy2(item, target_file)
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_images += 1
        
        # If no images found in subdirectories, look for a flat structure
        if total_images == 0:
            for item in raw_path.iterdir():
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    # Use filename pattern to determine class if possible
                    class_name = dataset_name  # Default to dataset name
                    
                    # Create class directory
                    class_dir = output_path / class_name
                    class_dir.mkdir(exist_ok=True)
                    
                    # Copy image
                    target_file = class_dir / item.name
                    if not target_file.exists():
                        shutil.copy2(item, target_file)
                    
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_images += 1
        
        logger.info(f"Processed {dataset_name}: {total_images} images, {len(class_counts)} classes")
        
        return {
            'num_images': total_images,
            'num_classes': len(class_counts),
            'class_names': sorted(class_counts.keys()),
            'class_counts': class_counts,
            'output_path': str(output_path)
        }


class UnifiedDatasetManager:
    """Main class for managing all datasets."""
    
    # Predefined dataset configurations
    DATASET_CONFIGS = {
        'oxford_pets': DatasetConfig(
            name='oxford_pets',
            source_type='oxford',
            source_url='https://www.robots.ox.ac.uk/~vgg/data/pets/',
            local_path='oxford_pets',
            description='Oxford-IIIT Pet Dataset - 37 pet breeds',
            num_classes=37,
            class_names=[],  # Will be populated during processing
        ),
        'kaggle_vegetables': DatasetConfig(
            name='kaggle_vegetables',
            source_type='kaggle',
            source_url='misrakahmed/vegetable-image-dataset',
            local_path='vegetables',
            description='Kaggle Vegetable Images Dataset',
            num_classes=15,  # Approximate
            class_names=[],  # Will be populated during processing
        ),
        'imagenet_subset': DatasetConfig(
            name='imagenet_subset',
            source_type='kaggle',
            source_url='ifigotin/imagenetmini-1000',
            local_path='imagenet_mini',
            description='ImageNet Mini - 1000 classes subset',
            num_classes=1000,
            class_names=[],
        ),
        'street_foods': DatasetConfig(
            name='street_foods',
            source_type='kaggle',
            source_url='nikolasgegenava/popular-street-foods',
            local_path='street_foods',
            description='Popular Street Foods Dataset',
            num_classes=0,  # Will be determined from data
            class_names=[],  # Will be populated during processing
        ),
        'musical_instruments': DatasetConfig(
            name='musical_instruments',
            source_type='kaggle',
            source_url='nikolasgegenava/music-instruments',
            local_path='musical_instruments',
            description='Musical Instruments Dataset',
            num_classes=0,  # Will be determined from data
            class_names=[],  # Will be populated during processing
        )
    }
    
    def __init__(self, base_cache_dir: str = "data/downloads"):
        self.cache_dir = Path(base_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloader = DatasetDownloader(base_cache_dir)
        self.processor = ImageDatasetProcessor(base_cache_dir)
        
        # Load existing configs
        self.config_file = self.cache_dir / "dataset_configs.json"
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, DatasetConfig]:
        """Load dataset configurations from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                return {
                    name: DatasetConfig(**config) 
                    for name, config in data.items()
                }
        else:
            # Initialize with predefined configs
            self._save_configs(self.DATASET_CONFIGS)
            return self.DATASET_CONFIGS.copy()
    
    def _save_configs(self, configs: Dict[str, DatasetConfig]):
        """Save dataset configurations to file."""
        with open(self.config_file, 'w') as f:
            json.dump({
                name: asdict(config) 
                for name, config in configs.items()
            }, f, indent=2)
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> str:
        """Download a specific dataset."""
        if dataset_name not in self.configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.configs[dataset_name]
        dataset_path = self.cache_dir / config.local_path
        
        if dataset_path.exists() and not force_redownload:
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}")
            return str(dataset_path)
        
        # Download based on source type
        if config.source_type == 'kaggle':
            raw_path = self.downloader.download_kaggle_dataset(
                config.source_url, f"{config.local_path}_raw"
            )
        elif config.source_type == 'kaggle_pandas':
            raw_path = self.downloader.download_kaggle_pandas_dataset(
                config.source_url, f"{config.local_path}_raw"
            )
        elif config.source_type == 'oxford':
            raw_path = self.downloader.download_oxford_pets(f"{config.local_path}_raw")
        elif config.source_type == 'url':
            raw_path = self.downloader.download_from_url(
                config.source_url, f"{config.local_path}_raw"
            )
        else:
            raise ValueError(f"Unsupported source type: {config.source_type}")
        
        # Process the downloaded data
        if dataset_name == 'oxford_pets':
            result = self.processor.process_oxford_pets(raw_path, dataset_path)
        elif dataset_name == 'kaggle_vegetables':
            result = self.processor.process_kaggle_vegetables(raw_path, dataset_path)
        elif dataset_name == 'street_foods':
            result = self.processor.process_kaggle_street_foods(raw_path, dataset_path)
        elif dataset_name == 'musical_instruments':
            result = self.processor.process_kaggle_musical_instruments(raw_path, dataset_path)
        elif config.source_type == 'kaggle_pandas':
            # Generic pandas dataset processing
            result = self.processor.process_pandas_dataset(raw_path, dataset_path, dataset_name)
        else:
            # Generic processing - just copy organized structure
            if Path(raw_path).exists():
                shutil.copytree(raw_path, dataset_path, dirs_exist_ok=True)
                result = {'output_path': str(dataset_path)}
        
        # Update config with actual class information
        if 'class_names' in result:
            config.class_names = result['class_names']
            config.num_classes = result['num_classes']
            self.configs[dataset_name] = config
            self._save_configs(self.configs)
        
        logger.info(f"Dataset {dataset_name} ready at: {dataset_path}")
        return str(dataset_path)
    
    def download_all_datasets(self, force_redownload: bool = False):
        """Download all configured datasets."""
        for dataset_name in self.configs.keys():
            try:
                self.download_dataset(dataset_name, force_redownload)
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
    
    def get_dataset_path(self, dataset_name: str) -> Optional[str]:
        """Get the local path for a dataset."""
        if dataset_name not in self.configs:
            return None
        
        config = self.configs[dataset_name]
        dataset_path = self.cache_dir / config.local_path
        
        if dataset_path.exists():
            return str(dataset_path)
        return None
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get configuration info for a dataset."""
        return self.configs.get(dataset_name)
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self.configs.keys())
    
    def create_combined_dataset(self, 
                              dataset_names: List[str], 
                              output_name: str,
                              class_mapping: Optional[Dict[str, str]] = None) -> str:
        """Create a combined dataset from multiple sources."""
        output_path = self.cache_dir / f"combined_{output_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_class_names = set()
        combined_stats = {}
        
        for dataset_name in dataset_names:
            dataset_path = self.get_dataset_path(dataset_name)
            if not dataset_path:
                logger.warning(f"Dataset {dataset_name} not found, downloading...")
                dataset_path = self.download_dataset(dataset_name)
            
            config = self.get_dataset_info(dataset_name)
            logger.info(f"Adding {dataset_name} to combined dataset...")
            
            # Copy images with optional class mapping
            dataset_path = Path(dataset_path)
            for class_dir in dataset_path.iterdir():
                if class_dir.is_dir():
                    original_class = class_dir.name
                    
                    # Apply class mapping if provided
                    if class_mapping and original_class in class_mapping:
                        target_class = class_mapping[original_class]
                    else:
                        target_class = original_class
                    
                    all_class_names.add(target_class)
                    
                    target_class_dir = output_path / target_class
                    target_class_dir.mkdir(exist_ok=True)
                    
                    count = 0
                    for image_file in class_dir.glob("*"):
                        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            target_file = target_class_dir / f"{dataset_name}_{image_file.name}"
                            if not target_file.exists():
                                shutil.copy2(image_file, target_file)
                            count += 1
                    
                    combined_stats[target_class] = combined_stats.get(target_class, 0) + count
        
        # Create config for combined dataset
        combined_config = DatasetConfig(
            name=f"combined_{output_name}",
            source_type='local',
            source_url='',
            local_path=f"combined_{output_name}",
            description=f"Combined dataset from: {', '.join(dataset_names)}",
            num_classes=len(all_class_names),
            class_names=sorted(all_class_names)
        )
        
        self.configs[f"combined_{output_name}"] = combined_config
        self._save_configs(self.configs)
        
        logger.info(f"Combined dataset created: {len(all_class_names)} classes, "
                   f"{sum(combined_stats.values())} total images")
        
        return str(output_path)


def get_dataset_manager() -> UnifiedDatasetManager:
    """Get singleton instance of dataset manager."""
    if not hasattr(get_dataset_manager, '_instance'):
        get_dataset_manager._instance = UnifiedDatasetManager()
    return get_dataset_manager._instance


# Convenience functions
def download_all_datasets():
    """Download all available datasets."""
    manager = get_dataset_manager()
    manager.download_all_datasets()


def get_dataset_path(dataset_name: str) -> Optional[str]:
    """Get path to a specific dataset."""
    manager = get_dataset_manager()
    return manager.get_dataset_path(dataset_name)


def create_combined_classification_dataset() -> str:
    """Create the main combined dataset for all models to use (binary: pets vs vegetables)."""
    manager = get_dataset_manager()
    
    # Download individual datasets first
    manager.download_dataset('oxford_pets')
    manager.download_dataset('kaggle_vegetables')
    
    # Create binary classification mapping (pets vs vegetables)
    class_mapping = {}
    
    # Get pet class names and map to 'pets'
    pets_config = manager.get_dataset_info('oxford_pets')
    if pets_config and pets_config.class_names:
        for class_name in pets_config.class_names:
            class_mapping[class_name] = 'pets'
    
    # Get vegetable class names and map to 'vegetables'  
    veg_config = manager.get_dataset_info('kaggle_vegetables')
    if veg_config and veg_config.class_names:
        for class_name in veg_config.class_names:
            class_mapping[class_name] = 'vegetables'
    
    # Create combined dataset
    combined_path = manager.create_combined_dataset(
        dataset_names=['oxford_pets', 'kaggle_vegetables'],
        output_name='pets_vs_vegetables',
        class_mapping=class_mapping
    )
    
    logger.info(f"Main classification dataset created at: {combined_path}")
    return combined_path


def create_three_class_classification_dataset() -> str:
    """Create a three-class combined dataset: pets vs vegetables vs street_foods."""
    manager = get_dataset_manager()
    
    # Download individual datasets first
    manager.download_dataset('oxford_pets')
    manager.download_dataset('kaggle_vegetables')
    manager.download_dataset('street_foods')
    
    # Create three-class classification mapping
    class_mapping = {}
    
    # Get pet class names and map to 'pets'
    pets_config = manager.get_dataset_info('oxford_pets')
    if pets_config and pets_config.class_names:
        for class_name in pets_config.class_names:
            class_mapping[class_name] = 'pets'
    
    # Get vegetable class names and map to 'vegetables'  
    veg_config = manager.get_dataset_info('kaggle_vegetables')
    if veg_config and veg_config.class_names:
        for class_name in veg_config.class_names:
            class_mapping[class_name] = 'vegetables'
    
    # Get street food class names and map to 'street_foods'
    foods_config = manager.get_dataset_info('street_foods')
    if foods_config and foods_config.class_names:
        for class_name in foods_config.class_names:
            # Clean up class names for consistent mapping
            clean_name = class_name.replace(' ', '_').replace('/', '_')
            class_mapping[clean_name] = 'street_foods'
    
    # Create combined dataset
    combined_path = manager.create_combined_dataset(
        dataset_names=['oxford_pets', 'kaggle_vegetables', 'street_foods'],
        output_name='pets_vs_vegetables_vs_foods',
        class_mapping=class_mapping
    )
    
    logger.info(f"Three-class classification dataset created at: {combined_path}")
    return combined_path


def create_four_class_classification_dataset() -> str:
    """Create a four-class combined dataset: pets vs vegetables vs street_foods vs musical_instruments."""
    manager = get_dataset_manager()
    
    # Download individual datasets first
    manager.download_dataset('oxford_pets')
    manager.download_dataset('kaggle_vegetables')
    manager.download_dataset('street_foods')
    manager.download_dataset('musical_instruments')
    
    # Create four-class classification mapping
    class_mapping = {}
    
    # Get pet class names and map to 'pets'
    pets_config = manager.get_dataset_info('oxford_pets')
    if pets_config and pets_config.class_names:
        for class_name in pets_config.class_names:
            class_mapping[class_name] = 'pets'
    
    # Get vegetable class names and map to 'vegetables'  
    veg_config = manager.get_dataset_info('kaggle_vegetables')
    if veg_config and veg_config.class_names:
        for class_name in veg_config.class_names:
            class_mapping[class_name] = 'vegetables'
    
    # Get street food class names and map to 'street_foods'
    foods_config = manager.get_dataset_info('street_foods')
    if foods_config and foods_config.class_names:
        for class_name in foods_config.class_names:
            clean_name = class_name.replace(' ', '_').replace('/', '_')
            class_mapping[clean_name] = 'street_foods'
    
    # Get musical instrument class names and map to 'musical_instruments'
    instruments_config = manager.get_dataset_info('musical_instruments')
    if instruments_config and instruments_config.class_names:
        for class_name in instruments_config.class_names:
            clean_name = class_name.replace(' ', '_').replace('/', '_')
            class_mapping[clean_name] = 'musical_instruments'
    
    # Create combined dataset
    combined_path = manager.create_combined_dataset(
        dataset_names=['oxford_pets', 'kaggle_vegetables', 'street_foods', 'musical_instruments'],
        output_name='pets_vs_vegetables_vs_foods_vs_instruments',
        class_mapping=class_mapping
    )
    
    logger.info(f"Four-class classification dataset created at: {combined_path}")
    return combined_path