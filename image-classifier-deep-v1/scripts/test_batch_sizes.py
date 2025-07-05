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

import subprocess
import sys
import json
from pathlib import Path
import time

def test_batch_size(batch_size, data_path, epochs=10):
    """Test training with a specific batch size."""
    print(f"\n{'='*60}")
    print(f"Testing with batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "scripts/train_improved.py",
        "--data_path", data_path,
        "--batch_size", str(batch_size),
        "--num_epochs", str(epochs),
        "--use_residual",
        "--optimizer", "adamw"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract results from output
        output_lines = result.stdout.split('\n')
        best_acc_line = [line for line in output_lines if "Best validation accuracy" in line]
        
        if best_acc_line:
            best_acc = float(best_acc_line[-1].split(':')[-1].strip())
        else:
            best_acc = 0.0
        
        training_time = time.time() - start_time
        
        # Read training history
        history_path = Path("logs/training_history_improved.json")
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0.0
                final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
        else:
            final_train_acc = final_val_acc = 0.0
        
        return {
            'batch_size': batch_size,
            'best_val_acc': best_acc,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'training_time': training_time,
            'success': True,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"Error with batch size {batch_size}: {error_msg}")
        
        return {
            'batch_size': batch_size,
            'best_val_acc': 0.0,
            'final_train_acc': 0.0,
            'final_val_acc': 0.0,
            'training_time': time.time() - start_time,
            'success': False,
            'error': error_msg
        }
    except Exception as e:
        print(f"Unexpected error with batch size {batch_size}: {str(e)}")
        
        return {
            'batch_size': batch_size,
            'best_val_acc': 0.0,
            'final_train_acc': 0.0,
            'final_val_acc': 0.0,
            'training_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test different batch sizes for Deep Learning v1 model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--batch_sizes', type=int, nargs='+', 
                       default=[32, 64, 128],
                       help='Batch sizes to test (default: 32 64 128)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per test (default: 10)')
    
    args = parser.parse_args()
    
    print("Deep Learning v1 Improved - Batch Size Testing")
    print(f"Data path: {args.data_path}")
    print(f"Batch sizes to test: {args.batch_sizes}")
    print(f"Epochs per test: {args.epochs}")
    
    # Test each batch size
    results = []
    for batch_size in args.batch_sizes:
        result = test_batch_size(batch_size, args.data_path, args.epochs)
        results.append(result)
        
        # Save intermediate results
        with open('batch_size_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH SIZE TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Batch Size':<12} {'Best Val Acc':<15} {'Final Train Acc':<17} {'Training Time':<15} {'Status':<10}")
    print(f"{'-'*80}")
    
    for result in results:
        status = "Success" if result['success'] else "Failed"
        print(f"{result['batch_size']:<12} "
              f"{result['best_val_acc']:<15.4f} "
              f"{result['final_train_acc']:<17.4f} "
              f"{result['training_time']:<15.2f}s "
              f"{status:<10}")
    
    # Find best batch size
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['best_val_acc'])
        print(f"\nBest batch size: {best_result['batch_size']} "
              f"(Validation Accuracy: {best_result['best_val_acc']:.4f})")
    
    print(f"\nDetailed results saved to: batch_size_results.json")


if __name__ == "__main__":
    main()