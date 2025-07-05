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
Test script to validate the production training pipeline.
Runs a quick test with minimal epochs to ensure everything works.
"""

import json
import subprocess
import sys
from pathlib import Path
import argparse


def test_single_model(model_type: str, variation: str, dataset: str):
    """Test training a single model with minimal epochs."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type}/{variation} on {dataset}")
    print(f"{'='*60}")
    
    base_dir = Path(__file__).parent.parent
    
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "production_trainer.py"),
        "--model_type", model_type,
        "--variation", variation, 
        "--dataset", dataset,
        "--config_dir", str(base_dir / "configs"),
        "--models_dir", str(base_dir / "models" / "test"),
        "--logs_dir", str(base_dir / "logs" / "test")
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print("✓ Test PASSED")
            return True
        else:
            print("✗ Test FAILED")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Test TIMED OUT (5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Test ERROR: {e}")
        return False


def test_hyperparameter_tuning(model_type: str, variation: str, dataset: str):
    """Test hyperparameter tuning with minimal trials."""
    print(f"\n{'='*60}")
    print(f"Testing hyperparameter tuning: {model_type}/{variation} on {dataset}")
    print(f"{'='*60}")
    
    base_dir = Path(__file__).parent.parent
    
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "hyperparameter_tuner.py"),
        "--model_type", model_type,
        "--model_variation", variation,
        "--dataset", dataset,
        "--n_trials", "2",  # Minimal trials for testing
        "--config_dir", str(base_dir / "configs"),
        "--log_dir", str(base_dir / "logs" / "test")
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            print("✓ Tuning test PASSED")
            return True
        else:
            print("✗ Tuning test FAILED")
            print("STDOUT:", result.stdout[-500:])
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Tuning test TIMED OUT (10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Tuning test ERROR: {e}")
        return False


def run_quick_tests():
    """Run quick tests on a subset of models."""
    print("Running Production Pipeline Quick Tests")
    print("=" * 60)
    
    # Test configurations - one from each model type
    test_configs = [
        ("shallow", "svm_hog_lbp", "vegetables"),
        ("deep_v1", "standard", "vegetables"),
        ("deep_v2", "resnet", "vegetables"),
        ("transfer", "resnet50", "vegetables")
    ]
    
    # Test individual model training
    print("\n🧪 Testing individual model training...")
    training_results = []
    
    for model_type, variation, dataset in test_configs:
        result = test_single_model(model_type, variation, dataset)
        training_results.append((f"{model_type}/{variation}", result))
    
    # Test hyperparameter tuning (just one)
    print("\n🔍 Testing hyperparameter tuning...")
    tuning_result = test_hyperparameter_tuning("shallow", "svm_hog_lbp", "vegetables")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print("\nModel Training Tests:")
    for model_name, passed in training_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {model_name}: {status}")
    
    print(f"\nHyperparameter Tuning Test:")
    status = "✓ PASS" if tuning_result else "✗ FAIL"
    print(f"  shallow/svm_hog_lbp: {status}")
    
    # Overall result
    all_passed = all(result for _, result in training_results) and tuning_result
    
    print(f"\nOVERALL RESULT: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Production pipeline is ready to use!")
        print("You can now run the full pipeline with:")
        print("python scripts/train_all_production_models.py")
    else:
        print("\n⚠️  Please fix the failing tests before running the full pipeline.")
    
    return all_passed


def verify_configuration():
    """Verify that configuration files are valid."""
    print("Verifying configuration files...")
    
    base_dir = Path(__file__).parent.parent
    
    # Check datasets.json
    datasets_file = base_dir / "configs" / "datasets.json"
    try:
        with open(datasets_file, 'r') as f:
            datasets_config = json.load(f)
        print("✓ datasets.json is valid")
        
        # Check if dataset paths exist
        missing_datasets = []
        for name, config in datasets_config["datasets"].items():
            if not Path(config["path"]).exists():
                missing_datasets.append(name)
        
        if missing_datasets:
            print(f"⚠️  Missing datasets: {missing_datasets}")
        else:
            print("✓ All dataset paths exist")
            
    except Exception as e:
        print(f"✗ Error in datasets.json: {e}")
        return False
    
    # Check models.json
    models_file = base_dir / "configs" / "models.json"
    try:
        with open(models_file, 'r') as f:
            models_config = json.load(f)
        print("✓ models.json is valid")
    except Exception as e:
        print(f"✗ Error in models.json: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test the production training pipeline')
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite (takes longer)')
    parser.add_argument('--config_only', action='store_true',
                       help='Only verify configuration files')
    
    args = parser.parse_args()
    
    # Always verify configuration first
    if not verify_configuration():
        print("❌ Configuration verification failed!")
        return False
    
    if args.config_only:
        print("✅ Configuration verification completed!")
        return True
    
    # Run tests
    if args.full:
        print("Running full test suite...")
        # TODO: Implement full test suite
        print("Full test suite not implemented yet. Use quick tests.")
        return run_quick_tests()
    else:
        return run_quick_tests()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)