#!/usr/bin/env python3
"""
Test script to verify model imports work in Docker environment.
This simulates what the model_manager.py would do.
"""

import sys
import os
from pathlib import Path

# Add models to Python path (simulating Docker environment)
models_path = Path(__file__).parent / "models"
sys.path.insert(0, str(models_path))

print(f"Python path: {sys.path}")
print(f"Models path: {models_path}")
print(f"Models path exists: {models_path.exists()}")

# Test imports
try:
    from classifiers.shallow.src.classifier import ShallowImageClassifier
    print("✅ ShallowImageClassifier imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ShallowImageClassifier: {e}")

try:
    from classifiers.deep_v1.src.classifier import DeepLearningV1Classifier
    print("✅ DeepLearningV1Classifier imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DeepLearningV1Classifier: {e}")

try:
    from classifiers.deep_v2.src.classifier import DeepLearningV2Classifier
    print("✅ DeepLearningV2Classifier imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DeepLearningV2Classifier: {e}")

try:
    from classifiers.transfer.src.classifier import TransferLearningClassifier
    print("✅ TransferLearningClassifier imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TransferLearningClassifier: {e}")

# Test if we can find a model path
try:
    transfer_path = models_path / "production" / "models" / "transfer"
    print(f"\nTransfer models path: {transfer_path}")
    print(f"Transfer models path exists: {transfer_path.exists()}")
    
    if transfer_path.exists():
        print("Available transfer models:")
        for model_dir in transfer_path.iterdir():
            if model_dir.is_dir():
                print(f"  {model_dir.name}")
                
except Exception as e:
    print(f"❌ Error checking model paths: {e}")