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
Unified CLI for training all image classifier models.

This script provides a single entry point for training any of the implemented
image classification models with their specific configurations and parameters.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os


def get_model_info():
    """Get information about available models."""
    models = {
        'shallow': {
            'name': 'Shallow Learning Classifier',
            'description': 'Traditional machine learning with feature extraction (HOG, LBP, color histograms)',
            'framework': 'scikit-learn',
            'path': 'image-classifier-shallow',
            'dependencies': ['scikit-learn', 'opencv-python', 'numpy', 'matplotlib'],
            'example': 'python train_models.py shallow --data_path /path/to/data --epochs 100'
        },
        'deep-v1': {
            'name': 'Deep Learning v1 Classifier',
            'description': 'Basic CNN with standard layers and training techniques',
            'framework': 'PyTorch',
            'path': 'image-classifier-deep-v1',
            'dependencies': ['torch', 'torchvision', 'numpy', 'matplotlib', 'Pillow'],
            'example': 'python train_models.py deep-v1 --data_path /path/to/data --epochs 30 --batch_size 16'
        },
        'deep-v2': {
            'name': 'Deep Learning v2 Classifier',
            'description': 'Advanced CNN with ResNet, attention mechanisms, and modern techniques',
            'framework': 'PyTorch',
            'path': 'image-classifier-deep-v2',
            'dependencies': ['torch', 'torchvision', 'numpy', 'matplotlib', 'Pillow', 'torchinfo'],
            'example': 'python train_models.py deep-v2 --data_path /path/to/data --epochs 25 --batch_size 8'
        },
        'transfer': {
            'name': 'Transfer Learning Classifier',
            'description': 'Transfer learning with pre-trained models (ResNet50, VGG16, EfficientNet)',
            'framework': 'TensorFlow/Keras',
            'path': 'image-classifier-transfer',
            'dependencies': ['tensorflow', 'numpy', 'matplotlib', 'Pillow'],
            'example': 'python train_models.py transfer --data_path /path/to/data --epochs 20 --base_model resnet50'
        }
    }
    return models


def check_dependencies(model_key):
    """Check if model dependencies are installed."""
    models = get_model_info()
    if model_key not in models:
        return False, []
    
    missing = []
    for dep in models[model_key]['dependencies']:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append(dep)
    
    return len(missing) == 0, missing


def install_dependencies(model_key):
    """Install dependencies for a specific model."""
    models = get_model_info()
    if model_key not in models:
        print(f"Unknown model: {model_key}")
        return False
    
    deps = models[model_key]['dependencies']
    print(f"Installing dependencies for {models[model_key]['name']}: {', '.join(deps)}")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + deps)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False


def list_models():
    """List all available models with their descriptions."""
    models = get_model_info()
    
    print("Available Image Classification Models:")
    print("=" * 60)
    
    for key, info in models.items():
        deps_ok, missing = check_dependencies(key)
        status = "✅ Ready" if deps_ok else f"❌ Missing: {', '.join(missing)}"
        
        print(f"\n{key.upper()}: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Framework: {info['framework']}")
        print(f"  Status: {status}")
        print(f"  Example: {info['example']}")


def run_model_training(model_key, args):
    """Run training for a specific model."""
    models = get_model_info()
    
    if model_key not in models:
        print(f"Error: Unknown model '{model_key}'")
        print("Available models:", ', '.join(models.keys()))
        return False
    
    # Check dependencies
    deps_ok, missing = check_dependencies(model_key)
    if not deps_ok:
        print(f"Missing dependencies for {model_key}: {', '.join(missing)}")
        print("Install them with:")
        print(f"  python train_models.py install {model_key}")
        print("Or manually:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    # Construct command
    model_path = Path(__file__).parent / models[model_key]['path']
    train_script = model_path / 'scripts' / 'train.py'
    
    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        return False
    
    # Run the training script
    cmd = [sys.executable, str(train_script)] + args
    
    print(f"Running {models[model_key]['name']} training...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Change to model directory for proper imports
        old_cwd = os.getcwd()
        os.chdir(model_path)
        
        result = subprocess.run(cmd, cwd=model_path)
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False
    finally:
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(
        description='Unified CLI for training image classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py list                    # List all available models
  python train_models.py install shallow         # Install dependencies for shallow learning
  python train_models.py shallow --data_path /path/to/data --epochs 100
  python train_models.py deep-v2 --data_path /path/to/data --epochs 25 --batch_size 8
  python train_models.py transfer --data_path /path/to/data --base_model resnet50

For model-specific help:
  python train_models.py shallow --help
  python train_models.py deep-v2 --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available models')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install dependencies for a model')
    install_parser.add_argument('model', choices=get_model_info().keys(), help='Model to install dependencies for')
    
    # Model training commands
    models = get_model_info()
    for model_key in models.keys():
        model_parser = subparsers.add_parser(model_key, help=f'Train {models[model_key]["name"]}', 
                                           add_help=False)
    
    # Parse known args to handle model-specific arguments
    args, unknown = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_models()
    
    elif args.command == 'install':
        success = install_dependencies(args.model)
        if success:
            print(f"✅ Dependencies installed successfully for {args.model}")
        else:
            print(f"❌ Failed to install dependencies for {args.model}")
    
    elif args.command in models:
        # Handle help for specific models
        if '--help' in unknown or '-h' in unknown:
            # Run the model's training script with --help
            model_path = Path(__file__).parent / models[args.command]['path']
            train_script = model_path / 'scripts' / 'train.py'
            subprocess.run([sys.executable, str(train_script), '--help'], cwd=model_path)
            return
        
        success = run_model_training(args.command, unknown)
        if not success:
            sys.exit(1)
    
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()