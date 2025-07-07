#!/usr/bin/env python
"""
Demonstration of the enhanced production training system with comprehensive logging.
This script shows how the training registry tracks all model metadata.
"""

import json
from pathlib import Path
from datetime import datetime

from production_trainer import ProductionTrainer
from training_registry import TrainingRegistry
from production_export import ProductionExporter


def demonstrate_enhanced_system():
    """Demonstrate the enhanced training and export system."""
    
    print("=" * 80)
    print("ENHANCED PRODUCTION TRAINING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Setup paths
    config_dir = Path(__file__).parent.parent / "configs"
    models_dir = Path(__file__).parent.parent / "models" / "demo"
    logs_dir = Path(__file__).parent.parent / "logs"
    registry_path = models_dir.parent / "demo_registry.json"
    
    print(f"\nConfiguration:")
    print(f"  Config directory: {config_dir}")
    print(f"  Models directory: {models_dir}")
    print(f"  Registry path: {registry_path}")
    
    # Initialize registry
    registry = TrainingRegistry(registry_path)
    
    # Start training session
    session_id = registry.start_training_session("demo_session")
    print(f"\nStarted training session: {session_id}")
    
    # Initialize trainer with the session
    trainer = ProductionTrainer(
        config_dir=config_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        session_id=session_id
    )
    
    print("\n" + "-" * 60)
    print("TRAINING DEMONSTRATION")
    print("-" * 60)
    
    # Train a single model as demonstration
    print("\nTraining demonstration model: transfer/resnet50 on vegetables dataset...")
    
    try:
        result = trainer.train_production_model(
            model_type="transfer",
            variation="resnet50", 
            dataset="vegetables"
        )
        
        print(f"\nTraining completed!")
        print(f"Model ID: {result.get('model_id', 'N/A')}")
        print(f"Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")
        print(f"Training time: {result.get('training_time_seconds', 0):.1f} seconds")
        
    except Exception as e:
        print(f"Training failed (expected in demo): {e}")
        print("This is normal in a demo environment without proper datasets")
    
    # End training session
    registry.end_training_session(session_id)
    
    print("\n" + "-" * 60)
    print("REGISTRY DEMONSTRATION")
    print("-" * 60)
    
    # Show registry capabilities
    summary = registry.get_training_summary()
    print(f"\nRegistry Summary:")
    print(f"  Total models: {summary['total_models']}")
    print(f"  Production ready: {summary['production_ready_models']}")
    print(f"  Pending export: {summary['models_pending_export']}")
    
    # Show models in registry
    if registry.registry["models"]:
        print(f"\nRegistered Models:")
        for model_id, model_info in registry.registry["models"].items():
            print(f"  {model_id}:")
            print(f"    Type: {model_info['model_type']}/{model_info['variation']}")
            print(f"    Dataset: {model_info['dataset']}")
            print(f"    Accuracy: {model_info['best_validation_accuracy']:.4f}")
            print(f"    Status: {model_info['training_status']}")
            print(f"    Export: {model_info['export_status']}")
    
    print("\n" + "-" * 60)
    print("EXPORT SYSTEM DEMONSTRATION")
    print("-" * 60)
    
    # Demonstrate export system
    export_dir = models_dir.parent / "demo_exports"
    exporter = ProductionExporter(registry_path, export_dir)
    
    # Check what models are available for export
    models_to_export = registry.get_models_for_export("pending")
    print(f"\nModels available for export: {len(models_to_export)}")
    
    for model_entry in models_to_export:
        model_info = model_entry["model_info"]
        print(f"  {model_entry['model_id']}: {model_info['model_type']}/{model_info['variation']} on {model_info['dataset']}")
    
    if models_to_export:
        print(f"\nExport system is ready to process {len(models_to_export)} models")
        print("To export models, run:")
        print(f"  python scripts/production_export.py --registry_path {registry_path}")
    else:
        print("\nNo models available for export (expected in demo)")
    
    print("\n" + "-" * 60)
    print("COMPREHENSIVE LOGGING DEMONSTRATION")
    print("-" * 60)
    
    # Show the comprehensive metadata that gets logged
    if registry.registry["models"]:
        model_id = list(registry.registry["models"].keys())[0]
        model_info = registry.registry["models"][model_id]
        
        print(f"\nComprehensive model metadata for {model_id}:")
        print(json.dumps({
            "identification": {
                "model_id": model_info["model_id"],
                "model_type": model_info["model_type"],
                "variation": model_info["variation"],
                "dataset": model_info["dataset"],
                "version": model_info["version"]
            },
            "performance": {
                "metrics": model_info["metrics"],
                "training_time": model_info["training_time_seconds"]
            },
            "configuration": {
                "hyperparameters": model_info["hyperparameters"],
                "final_hyperparameters": model_info["final_hyperparameters"]
            },
            "files": {
                "model_path": model_info["model_path"],
                "config_path": model_info["config_path"],
                "log_path": model_info["log_path"],
                "model_files": model_info["model_files"]
            },
            "production": {
                "training_status": model_info["training_status"],
                "export_status": model_info["export_status"],
                "production_ready": model_info["production_ready"]
            },
            "environment": model_info.get("environment", {}),
            "session": {
                "session_id": model_info["session_id"],
                "registered_at": model_info["registered_at"]
            }
        }, indent=2))
    
    print("\n" + "-" * 60)
    print("PRODUCTION WORKFLOW SUMMARY")
    print("-" * 60)
    
    print(f"""
The enhanced production training system provides:

1. COMPREHENSIVE METADATA TRACKING:
   ✓ Model identification (type, variation, dataset, version)
   ✓ Performance metrics and training time
   ✓ Complete hyperparameter configuration
   ✓ File paths for all model artifacts
   ✓ Training environment details
   ✓ Export status and production readiness

2. SESSION MANAGEMENT:
   ✓ Track training sessions with unique IDs
   ✓ Monitor progress across multiple model trainings
   ✓ Session-level statistics and summaries

3. EXPORT ORCHESTRATION:
   ✓ Identify models ready for export
   ✓ Batch export to ONNX format
   ✓ Update export status automatically
   ✓ Generate deployment packages

4. PRODUCTION DEPLOYMENT:
   ✓ All model metadata available in JSON logs
   ✓ Standardized file organization
   ✓ Version tracking and reproducibility
   ✓ Production-ready export formats

NEXT STEPS:
- Run full training: python scripts/train_all_production_models.py
- Export models: python scripts/production_export.py
- Create deployment package: python scripts/production_export.py --package production_v1

Registry location: {registry_path}
Export directory: {export_dir}
    """)
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_enhanced_system()