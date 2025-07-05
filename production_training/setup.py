#!/usr/bin/env python
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
Setup script for production training pipeline.
Installs required dependencies and prepares the environment.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages for the production training pipeline."""
    requirements = [
        "optuna>=3.0.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nSetup completed!")


def verify_datasets():
    """Verify that required datasets exist."""
    base_dir = Path("/home/brandond/Projects/pvt/personal/eyeVsAI/data/downloads")
    
    expected_datasets = [
        "combined_unified_classification",
        "vegetables", 
        "oxford_pets",
        "street_foods",
        "musical_instruments"
    ]
    
    print("\nVerifying datasets...")
    for dataset in expected_datasets:
        dataset_path = base_dir / dataset
        if dataset_path.exists():
            print(f"✓ Found {dataset}")
        else:
            print(f"✗ Missing {dataset} at {dataset_path}")
    
    print("\nDataset verification completed!")


if __name__ == "__main__":
    install_requirements()
    verify_datasets()