#!/usr/bin/env python
"""
Comprehensive training script for all ML models on the 67-class dataset.
Runs all models with realistic hyperparameters and saves results.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
DATA_PATH = "/home/brandond/Projects/pvt/personal/eyeVsAI/data/downloads/combined_unified_classification"
RESULTS_DIR = f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training configurations for each model
TRAINING_CONFIGS = {
    "shallow_learning": {
        "script": "ml_models_core/notebooks/shallow_learning_classifier.py",
        "args": [
            "--data_path", DATA_PATH,
            "--model_save_path", f"{RESULTS_DIR}/shallow_learning_model.pkl",
            "--test_size", "0.2",
            "--random_state", "42",
            "--n_components", "100",
            "--kernel", "rbf"
        ],
        "description": "Shallow Learning (HOG/LBP + SVM)"
    },
    
    "deep_learning_v1": {
        "script": "ml_models_core/notebooks/deep_learning_classifier_optimized.py",
        "args": [
            "--data_path", DATA_PATH,
            "--model_save_path", f"{RESULTS_DIR}/deep_learning_v1_model.pth",
            "--batch_size", "64",
            "--learning_rate", "0.001",
            "--num_epochs", "30",
            "--patience", "5",
            "--image_size", "128", "128",
            "--num_workers", "4",
            "--gradient_accumulation", "2"
        ],
        "description": "Deep Learning v1 (CNN with Attention)"
    },
    
    "deep_learning_v2": {
        "script": "ml_models_core/notebooks/deep_learning_classifier_v2.py",
        "args": [
            "--data_path", DATA_PATH,
            "--model_save_path", f"{RESULTS_DIR}/deep_learning_v2_model.pth",
            "--batch_size", "32",
            "--learning_rate", "0.001",
            "--num_epochs", "30",
            "--patience", "5",
            "--image_size", "224", "224",
            "--num_workers", "4",
            "--architecture", "custom_cnn",
            "--dropout", "0.5"
        ],
        "description": "Deep Learning v2 (Advanced CNN)"
    },
    
    "transfer_learning": {
        "script": "image-classifier-transfer/scripts/train_pytorch.py",
        "args": [
            "--data_path", DATA_PATH,
            "--model_save_path", f"{RESULTS_DIR}/transfer_learning_model.pth",
            "--batch_size", "32",
            "--learning_rate", "0.001",
            "--num_epochs", "20",
            "--patience", "5",
            "--base_model", "resnet50",
            "--fine_tune_layers", "10",
            "--fine_tune_lr", "0.00001",
            "--mixed_precision",
            "--class_weights",
            "--optimizer", "adamw",
            "--scheduler", "cosine"
        ],
        "description": "Transfer Learning (ResNet50 + Fine-tuning)"
    }
}

def run_model_training(model_name, config):
    """Run training for a specific model."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}: {config['description']}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Build command
    cmd = ["python", config["script"]] + config["args"]
    
    # Log command
    with open(f"{RESULTS_DIR}/{model_name}_command.txt", "w") as f:
        f.write(" ".join(cmd))
    
    # Run training
    import subprocess
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Save output
        with open(f"{RESULTS_DIR}/{model_name}_output.txt", "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
        
        # Extract metrics from output
        metrics = extract_metrics(result.stdout, model_name)
        
        training_time = time.time() - start_time
        
        return {
            "status": "success",
            "training_time": training_time,
            "metrics": metrics
        }
        
    except subprocess.CalledProcessError as e:
        # Save error output
        with open(f"{RESULTS_DIR}/{model_name}_error.txt", "w") as f:
            f.write(f"Command failed with return code {e.returncode}\n")
            f.write(f"STDOUT:\n{e.stdout}\n")
            f.write(f"STDERR:\n{e.stderr}\n")
        
        return {
            "status": "failed",
            "error": str(e),
            "training_time": time.time() - start_time
        }

def extract_metrics(output, model_name):
    """Extract metrics from training output."""
    metrics = {}
    
    # Common patterns to look for
    patterns = {
        "test_accuracy": [
            r"Test [Aa]ccuracy[:\s]+([0-9.]+)",
            r"test_acc[:\s]+([0-9.]+)",
            r"Final test accuracy[:\s]+([0-9.]+)"
        ],
        "validation_accuracy": [
            r"[Vv]al[idation]* [Aa]ccuracy[:\s]+([0-9.]+)",
            r"Best val[idation]* accuracy[:\s]+([0-9.]+)"
        ],
        "training_time": [
            r"Training time[:\s]+([0-9.]+)",
            r"Total time[:\s]+([0-9.]+)"
        ]
    }
    
    import re
    for metric_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                    break
                except:
                    pass
    
    return metrics

def main():
    print(f"Starting comprehensive model training")
    print(f"Dataset: {DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Start time: {datetime.now()}")
    
    results = {}
    
    # Run each model
    for model_name, config in TRAINING_CONFIGS.items():
        print(f"\n\nPreparing to train: {model_name}")
        
        # Check if script exists
        if not os.path.exists(config["script"]):
            print(f"WARNING: Script not found: {config['script']}")
            results[model_name] = {
                "status": "skipped",
                "reason": "script_not_found"
            }
            continue
        
        # Run training
        result = run_model_training(model_name, config)
        results[model_name] = result
        
        # Save intermediate results
        with open(f"{RESULTS_DIR}/results_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{model_name} completed in {result.get('training_time', 0):.2f} seconds")
        print(f"Status: {result['status']}")
        if result.get('metrics'):
            print(f"Metrics: {result['metrics']}")
    
    # Generate final report
    print(f"\n\n{'='*80}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*80}\n")
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Status: {result['status']}")
        if result.get('metrics'):
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value}")
        print(f"  Training time: {result.get('training_time', 0):.2f} seconds")
    
    # Save final summary
    summary = {
        "training_date": datetime.now().isoformat(),
        "dataset": DATA_PATH,
        "results": results
    }
    
    with open(f"{RESULTS_DIR}/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nAll results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()