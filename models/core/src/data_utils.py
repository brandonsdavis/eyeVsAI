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
Data validation, verification, and utility functions.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False


def verify_dataset_integrity(dataset_path: str, 
                           expected_classes: Optional[List[str]] = None,
                           min_images_per_class: int = 10) -> Dict:
    """Verify dataset integrity and structure."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return {'valid': False, 'error': f'Dataset path does not exist: {dataset_path}'}
    
    # Scan for class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        return {'valid': False, 'error': 'No class directories found'}
    
    # Count images per class
    class_counts = {}
    corrupted_images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_count = 0
        
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                # Try to open image to verify it's not corrupted
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    image_count += 1
                except Exception as e:
                    corrupted_images.append((str(img_path), str(e)))
        
        class_counts[class_name] = image_count
    
    # Check for expected classes
    found_classes = set(class_counts.keys())
    if expected_classes:
        expected_set = set(expected_classes)
        missing_classes = expected_set - found_classes
        extra_classes = found_classes - expected_set
    else:
        missing_classes = set()
        extra_classes = set()
    
    # Check minimum images per class
    insufficient_classes = {
        cls: count for cls, count in class_counts.items() 
        if count < min_images_per_class
    }
    
    # Determine if dataset is valid
    is_valid = (
        len(corrupted_images) == 0 and
        len(missing_classes) == 0 and
        len(insufficient_classes) == 0
    )
    
    return {
        'valid': is_valid,
        'class_counts': class_counts,
        'total_images': sum(class_counts.values()),
        'num_classes': len(class_counts),
        'corrupted_images': corrupted_images,
        'missing_classes': list(missing_classes),
        'extra_classes': list(extra_classes),
        'insufficient_classes': insufficient_classes,
        'min_images_per_class': min_images_per_class
    }


def analyze_dataset_balance(class_counts: Dict[str, int]) -> Dict:
    """Analyze class balance in dataset."""
    counts = list(class_counts.values())
    
    if not counts:
        return {'balanced': False, 'error': 'No classes found'}
    
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = min(counts)
    max_count = max(counts)
    
    # Calculate imbalance ratio (max/min)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Coefficient of variation
    cv = std_count / mean_count if mean_count > 0 else float('inf')
    
    # Consider balanced if CV < 0.3 and imbalance ratio < 2
    is_balanced = cv < 0.3 and imbalance_ratio < 2.0
    
    return {
        'balanced': is_balanced,
        'mean_count': mean_count,
        'std_count': std_count,
        'min_count': min_count,
        'max_count': max_count,
        'imbalance_ratio': imbalance_ratio,
        'coefficient_of_variation': cv,
        'class_distribution': class_counts
    }


def visualize_dataset_statistics(dataset_path: str, save_path: Optional[str] = None):
    """Create visualizations of dataset statistics."""
    integrity = verify_dataset_integrity(dataset_path)
    
    if not integrity['valid']:
        print(f"Dataset integrity issues: {integrity}")
        return
    
    class_counts = integrity['class_counts']
    balance_info = analyze_dataset_balance(class_counts)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset Analysis: {Path(dataset_path).name}', fontsize=16)
    
    # 1. Class distribution bar chart
    ax1 = axes[0, 0]
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax1.bar(range(len(classes)), counts)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Images per Class')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    
    # Color bars based on count (red for low, green for high)
    max_count = max(counts)
    for bar, count in zip(bars, counts):
        color_intensity = count / max_count
        bar.set_color(plt.cm.RdYlGn(color_intensity))
    
    # 2. Class distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(counts, bins=min(10, len(classes)), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Images')
    ax2.set_ylabel('Number of Classes')
    ax2.set_title('Distribution of Class Sizes')
    ax2.axvline(np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    ax2.legend()
    
    # 3. Balance metrics
    ax3 = axes[1, 0]
    metrics = [
        f"Total Images: {integrity['total_images']}",
        f"Number of Classes: {integrity['num_classes']}",
        f"Mean per Class: {balance_info['mean_count']:.1f}",
        f"Std Dev: {balance_info['std_count']:.1f}",
        f"Imbalance Ratio: {balance_info['imbalance_ratio']:.2f}",
        f"CV: {balance_info['coefficient_of_variation']:.3f}",
        f"Balanced: {'Yes' if balance_info['balanced'] else 'No'}"
    ]
    
    ax3.text(0.1, 0.9, '\n'.join(metrics), transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Dataset Metrics')
    
    # 4. Class imbalance visualization
    ax4 = axes[1, 1]
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    classes_sorted = [item[0] for item in sorted_counts]
    counts_sorted = [item[1] for item in sorted_counts]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(counts_sorted)))
    ax4.barh(range(len(classes_sorted)), counts_sorted, color=colors)
    ax4.set_yticks(range(len(classes_sorted)))
    ax4.set_yticklabels(classes_sorted)
    ax4.set_xlabel('Number of Images')
    ax4.set_title('Classes Sorted by Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset analysis saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Path: {dataset_path}")
    print(f"  Total Images: {integrity['total_images']}")
    print(f"  Classes: {integrity['num_classes']}")
    print(f"  Balanced: {'Yes' if balance_info['balanced'] else 'No'}")
    
    if integrity['corrupted_images']:
        print(f"  Corrupted Images: {len(integrity['corrupted_images'])}")
    
    if balance_info['imbalance_ratio'] > 2.0:
        print(f"  ⚠️  High class imbalance detected (ratio: {balance_info['imbalance_ratio']:.2f})")


def calculate_dataset_checksum(dataset_path: str) -> str:
    """Calculate MD5 checksum of entire dataset."""
    dataset_path = Path(dataset_path)
    
    # Get all image files sorted by path for consistency
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if Path(file).suffix.lower() in valid_extensions:
                image_files.append(os.path.join(root, file))
    
    # Calculate combined hash
    hasher = hashlib.md5()
    
    for file_path in sorted(image_files):
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return hasher.hexdigest()


def compare_datasets(dataset1_path: str, dataset2_path: str) -> Dict:
    """Compare two datasets and return differences."""
    
    info1 = verify_dataset_integrity(dataset1_path)
    info2 = verify_dataset_integrity(dataset2_path)
    
    if not info1['valid'] or not info2['valid']:
        return {
            'comparable': False,
            'error': 'One or both datasets are invalid',
            'dataset1_valid': info1['valid'],
            'dataset2_valid': info2['valid']
        }
    
    classes1 = set(info1['class_counts'].keys())
    classes2 = set(info2['class_counts'].keys())
    
    common_classes = classes1.intersection(classes2)
    unique_to_1 = classes1 - classes2
    unique_to_2 = classes2 - classes1
    
    # Compare counts for common classes
    count_differences = {}
    for class_name in common_classes:
        diff = info1['class_counts'][class_name] - info2['class_counts'][class_name]
        if diff != 0:
            count_differences[class_name] = {
                'dataset1': info1['class_counts'][class_name],
                'dataset2': info2['class_counts'][class_name],
                'difference': diff
            }
    
    return {
        'comparable': True,
        'dataset1_summary': {
            'total_images': info1['total_images'],
            'num_classes': info1['num_classes']
        },
        'dataset2_summary': {
            'total_images': info2['total_images'],
            'num_classes': info2['num_classes']
        },
        'common_classes': list(common_classes),
        'unique_to_dataset1': list(unique_to_1),
        'unique_to_dataset2': list(unique_to_2),
        'count_differences': count_differences,
        'identical_structure': len(unique_to_1) == 0 and len(unique_to_2) == 0 and len(count_differences) == 0
    }


def sample_images_from_dataset(dataset_path: str, 
                             samples_per_class: int = 5,
                             save_dir: Optional[str] = None) -> Dict:
    """Create a sample visualization of images from each class."""
    dataset_path = Path(dataset_path)
    
    # Get class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        return {'error': 'No class directories found'}
    
    samples = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all images in class
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in valid_extensions]
        
        # Sample random images
        if len(image_files) >= samples_per_class:
            sampled = np.random.choice(image_files, samples_per_class, replace=False)
        else:
            sampled = image_files
        
        samples[class_name] = [str(f) for f in sampled]
    
    # Create visualization
    num_classes = len(samples)
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                           figsize=(samples_per_class * 2, num_classes * 2))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    elif samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (class_name, image_paths) in enumerate(samples.items()):
        for j, image_path in enumerate(image_paths):
            if j >= samples_per_class:
                break
                
            try:
                image = Image.open(image_path)
                axes[i, j].imshow(image)
                axes[i, j].set_title(f"{class_name}")
                axes[i, j].axis('off')
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f"Error loading\n{Path(image_path).name}", 
                              ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
        
        # Hide unused subplots
        for j in range(len(image_paths), samples_per_class):
            axes[i, j].axis('off')
    
    plt.suptitle(f'Sample Images from {dataset_path.name}', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f"{dataset_path.name}_samples.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample images saved to: {save_path}")
    
    plt.show()
    
    return samples


def create_dataset_report(dataset_path: str, output_dir: str):
    """Create comprehensive dataset analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = Path(dataset_path).name
    
    print(f"Creating dataset report for: {dataset_name}")
    
    # 1. Dataset integrity check
    integrity = verify_dataset_integrity(dataset_path)
    
    # 2. Balance analysis
    if integrity['valid']:
        balance_info = analyze_dataset_balance(integrity['class_counts'])
    else:
        balance_info = {'balanced': False, 'error': 'Dataset invalid'}
    
    # 3. Generate visualizations
    visualize_dataset_statistics(dataset_path, 
                                save_path=output_dir / f"{dataset_name}_statistics.png")
    
    # 4. Sample images
    sample_images_from_dataset(dataset_path, 
                             save_dir=output_dir)
    
    # 5. Calculate checksum
    checksum = calculate_dataset_checksum(dataset_path)
    
    # 6. Create summary report
    report = {
        'dataset_name': dataset_name,
        'dataset_path': str(dataset_path),
        'analysis_timestamp': str(np.datetime64('now')),
        'integrity_check': integrity,
        'balance_analysis': balance_info,
        'dataset_checksum': checksum
    }
    
    # Save report as JSON
    report_path = output_dir / f"{dataset_name}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Dataset report saved to: {output_dir}")
    print(f"Summary: {integrity['total_images']} images, {integrity['num_classes']} classes")
    
    return report