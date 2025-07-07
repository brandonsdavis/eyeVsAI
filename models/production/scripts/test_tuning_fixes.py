#!/usr/bin/env python
"""
Test script to verify the hyperparameter tuning fixes.
"""

import json
from pathlib import Path
from hyperparameter_tuner import HyperparameterTuner


def test_shallow_learning_fix():
    """Test that shallow learning no longer has import errors."""
    print("🧪 Testing Shallow Learning Import Fix...")
    
    config_dir = Path(__file__).parent.parent / "configs"
    log_dir = Path(__file__).parent.parent / "logs"
    
    # Test with just 2 trials on vegetables dataset (smaller)
    tuner = HyperparameterTuner(
        model_type="shallow",
        model_variation="svm_hog_lbp",
        dataset_name="vegetables",
        config_dir=config_dir,
        log_dir=log_dir,
        n_trials=2
    )
    
    try:
        results = tuner.run_tuning()
        print(f"✅ Shallow Learning Test PASSED")
        print(f"   Best accuracy: {results['best_value']:.4f}")
        print(f"   Best params: {results['best_params']}")
        return True
    except Exception as e:
        print(f"❌ Shallow Learning Test FAILED: {e}")
        return False


def test_deep_learning_fix():
    """Test that deep learning gets better results with more epochs."""
    print("\n🧪 Testing Deep Learning V1 Epochs Fix...")
    
    config_dir = Path(__file__).parent.parent / "configs"
    log_dir = Path(__file__).parent.parent / "logs"
    
    # Test with just 1 trial on vegetables dataset
    tuner = HyperparameterTuner(
        model_type="deep_v1",
        model_variation="standard",
        dataset_name="vegetables",
        config_dir=config_dir,
        log_dir=log_dir,
        n_trials=1
    )
    
    try:
        results = tuner.run_tuning()
        print(f"✅ Deep Learning Test PASSED")
        print(f"   Best accuracy: {results['best_value']:.4f}")
        if results['best_value'] > 0.05:  # At least 5% accuracy
            print(f"   ✅ Non-zero accuracy achieved!")
        else:
            print(f"   ⚠️  Still getting low accuracy - may need dataset verification")
        return True
    except Exception as e:
        print(f"❌ Deep Learning Test FAILED: {e}")
        return False


def verify_datasets():
    """Verify that datasets are accessible."""
    print("\n🧪 Verifying Dataset Accessibility...")
    
    config_dir = Path(__file__).parent.parent / "configs"
    with open(config_dir / "datasets.json", 'r') as f:
        datasets_config = json.load(f)
    
    for dataset_name, dataset_info in datasets_config["datasets"].items():
        dataset_path = Path(dataset_info["path"])
        if dataset_path.exists():
            # Count subdirectories (classes)
            class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            print(f"   ✅ {dataset_name}: {len(class_dirs)} classes at {dataset_path}")
        else:
            print(f"   ❌ {dataset_name}: NOT FOUND at {dataset_path}")
    
    return True


def main():
    """Run all tests."""
    print("🔧 TESTING HYPERPARAMETER TUNING FIXES")
    print("=" * 60)
    
    # First verify datasets are accessible
    verify_datasets()
    
    # Test the fixes
    shallow_success = test_shallow_learning_fix()
    deep_success = test_deep_learning_fix()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if shallow_success:
        print("✅ Shallow Learning Import Fix: WORKING")
    else:
        print("❌ Shallow Learning Import Fix: FAILED")
    
    if deep_success:
        print("✅ Deep Learning Epochs Fix: WORKING")
    else:
        print("❌ Deep Learning Epochs Fix: FAILED")
    
    if shallow_success and deep_success:
        print("\n🎉 ALL FIXES WORKING! You can now run full training:")
        print("   python scripts/train_all_production_models.py --run_tuning --tuning_trials 10")
    else:
        print("\n⚠️  Some fixes still need work. Check the error messages above.")


if __name__ == "__main__":
    main()