#!/usr/bin/env python
"""
Quick demonstration of the production training pipeline.
Trains a few select models to show the system working.
"""

import subprocess
import sys
from pathlib import Path
import json

def run_demo():
    """Run a quick demo of the production training pipeline."""
    print("🚀 Production Training Pipeline Demo")
    print("=" * 50)
    
    base_dir = Path(__file__).parent
    
    # Demo configurations - quick models to test
    demo_configs = [
        ("shallow", "svm_hog_lbp", "vegetables"),
        ("transfer", "resnet50", "vegetables"),  
        ("deep_v1", "standard", "vegetables")
    ]
    
    results = []
    
    for model_type, variation, dataset in demo_configs:
        print(f"\n📈 Training: {model_type}/{variation} on {dataset}")
        print("-" * 40)
        
        cmd = [
            sys.executable,
            str(base_dir / "scripts" / "production_trainer.py"),
            "--model_type", model_type,
            "--variation", variation,
            "--dataset", dataset,
            "--config_dir", str(base_dir / "configs"),
            "--models_dir", str(base_dir / "models" / "demo"),
            "--logs_dir", str(base_dir / "logs" / "demo")
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ SUCCESS")
                results.append({"model": f"{model_type}/{variation}", "dataset": dataset, "status": "success"})
            else:
                print("❌ FAILED")
                print("Error:", result.stderr[-200:])  # Last 200 chars
                results.append({"model": f"{model_type}/{variation}", "dataset": dataset, "status": "failed"})
                
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT (5 minutes)")
            results.append({"model": f"{model_type}/{variation}", "dataset": dataset, "status": "timeout"})
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 DEMO SUMMARY")
    print("=" * 50)
    
    for result in results:
        status_emoji = {"success": "✅", "failed": "❌", "timeout": "⏰"}[result["status"]]
        print(f"{status_emoji} {result['model']} on {result['dataset']}: {result['status'].upper()}")
    
    successful = len([r for r in results if r["status"] == "success"])
    total = len(results)
    
    print(f"\n📊 Results: {successful}/{total} models trained successfully")
    
    if successful > 0:
        print("\n🎉 Pipeline is working! You can now run:")
        print("   python scripts/train_all_production_models.py")
        print("   (This will train all 420+ model combinations)")
    else:
        print("\n⚠️  Some issues detected. Check the logs for details.")
    
    # Show where results are stored
    models_dir = base_dir / "models" / "demo"
    if models_dir.exists():
        print(f"\n📁 Demo models saved to: {models_dir}")
        
        # Show model structure
        for model_path in models_dir.rglob("training_results.json"):
            try:
                with open(model_path, 'r') as f:
                    model_result = json.load(f)
                print(f"   📄 {model_result['model_type']}/{model_result['variation']}: {model_path}")
            except:
                pass

if __name__ == "__main__":
    run_demo()