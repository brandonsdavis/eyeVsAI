#!/usr/bin/env python3
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
Setup script to download and prepare all datasets for the image classification project.
Run this script to automatically download and organize all required datasets.
"""

import sys
import argparse
from pathlib import Path

# Add ml-models-core to path
sys.path.append(str(Path(__file__).parent / "ml-models-core" / "src"))

from data_manager import get_dataset_manager, create_combined_classification_dataset, create_three_class_classification_dataset, create_four_class_classification_dataset
from data_utils import create_dataset_report


def main():
    parser = argparse.ArgumentParser(description="Setup datasets for image classification project")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["oxford_pets", "kaggle_vegetables", "street_foods", "musical_instruments", "imagenet_subset", "all"],
                       default=["oxford_pets", "kaggle_vegetables", "street_foods", "musical_instruments"],
                       help="Datasets to download")
    parser.add_argument("--create-unified", action="store_true", default=True,
                       help="Create unified dataset combining all image classes")
    parser.add_argument("--force", action="store_true", default=False,
                       help="Force re-download even if datasets exist")
    parser.add_argument("--reports", action="store_true", default=False,
                       help="Generate dataset analysis reports")
    parser.add_argument("--cache-dir", default="data/downloads",
                       help="Directory to store downloaded datasets")
    
    args = parser.parse_args()
    
    print("🚀 Setting up datasets for ML Image Classification Project")
    print("=" * 60)
    
    # Initialize dataset manager
    manager = get_dataset_manager()
    
    # Download requested datasets
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["oxford_pets", "kaggle_vegetables", "street_foods", "musical_instruments", "imagenet_subset"]
    
    print(f"📥 Downloading datasets: {', '.join(datasets_to_download)}")
    
    for dataset_name in datasets_to_download:
        try:
            print(f"\n🔄 Processing {dataset_name}...")
            dataset_path = manager.download_dataset(dataset_name, force_redownload=args.force)
            print(f"✅ {dataset_name} ready at: {dataset_path}")
            
            # Generate report if requested
            if args.reports:
                print(f"📊 Generating analysis report for {dataset_name}...")
                report_dir = Path(args.cache_dir) / "reports"
                create_dataset_report(dataset_path, report_dir)
                
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
            continue
    
    # Create unified dataset if requested
    if args.create_unified:
        try:
            print(f"\n🔗 Creating unified dataset with all classes...")
            # Get all available datasets
            available_datasets = []
            for dataset_name in datasets_to_download:
                if dataset_name != "all":
                    available_datasets.append(dataset_name)
            
            # Create unified dataset combining all classes
            unified_path = manager.create_combined_dataset(
                dataset_names=available_datasets,
                output_name="unified_classification",
                class_mapping=None  # No mapping, keep original class names
            )
            print(f"✅ Unified dataset created at: {unified_path}")
            
            if args.reports:
                print(f"📊 Generating analysis report for unified dataset...")
                report_dir = Path(args.cache_dir) / "reports"
                create_dataset_report(unified_path, report_dir)
                
        except Exception as e:
            print(f"❌ Failed to create unified dataset: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Dataset Setup Summary:")
    
    available_datasets = manager.list_available_datasets()
    for dataset_name in available_datasets:
        dataset_path = manager.get_dataset_path(dataset_name)
        if dataset_path:
            dataset_info = manager.get_dataset_info(dataset_name)
            print(f"  ✅ {dataset_name}")
            print(f"     Path: {dataset_path}")
            if dataset_info:
                print(f"     Classes: {dataset_info.num_classes}")
                print(f"     Description: {dataset_info.description}")
        else:
            print(f"  ❌ {dataset_name} (not downloaded)")
    
    print("\n Dataset setup complete!")


if __name__ == "__main__":
    main()