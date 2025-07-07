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
Production export script for trained models.
Exports models to ONNX format and updates registry.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import subprocess
import sys

from training_registry import TrainingRegistry


class ProductionExporter:
    """Export trained models to production formats."""
    
    def __init__(self, registry_path: Path, export_dir: Path = None):
        self.registry = TrainingRegistry(registry_path)
        self.export_dir = export_dir or registry_path.parent / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.export_dir / f"export_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def export_models(self, model_type: str = None, dataset: str = None, 
                     format: str = "onnx", force_export: bool = False) -> Dict[str, Any]:
        """Export models to production format."""
        
        # Get models that need export
        export_status = "pending" if not force_export else None
        models_to_export = self.registry.get_models_for_export(
            export_status=export_status,
            model_type=model_type,
            dataset=dataset
        )
        
        if not models_to_export:
            self.logger.info("No models found for export")
            return {"exported": 0, "failed": 0, "skipped": 0}
        
        self.logger.info(f"Found {len(models_to_export)} models to export")
        
        export_results = {
            "exported": 0,
            "failed": 0,
            "skipped": 0,
            "details": []
        }
        
        for model_entry in models_to_export:
            model_id = model_entry["model_id"]
            model_info = model_entry["model_info"]
            
            try:
                self.logger.info(f"Exporting {model_id}: {model_info['model_type']}/{model_info['variation']}")
                
                if format == "onnx":
                    result = self._export_to_onnx(model_id, model_info)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                if result["success"]:
                    # Update registry
                    self.registry.update_export_status(
                        model_id=model_id,
                        export_status="completed",
                        export_files=result["export_files"]
                    )
                    export_results["exported"] += 1
                    self.logger.info(f"Successfully exported {model_id}")
                else:
                    self.registry.update_export_status(
                        model_id=model_id,
                        export_status="failed"
                    )
                    export_results["failed"] += 1
                    self.logger.error(f"Failed to export {model_id}: {result['error']}")
                
                export_results["details"].append({
                    "model_id": model_id,
                    "status": "success" if result["success"] else "failed",
                    "files": result.get("export_files", {}),
                    "error": result.get("error")
                })
                
            except Exception as e:
                self.logger.error(f"Exception exporting {model_id}: {e}")
                self.registry.update_export_status(
                    model_id=model_id,
                    export_status="failed"
                )
                export_results["failed"] += 1
                export_results["details"].append({
                    "model_id": model_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save export summary
        summary_file = self.export_dir / f"export_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        self.logger.info(f"Export completed: {export_results['exported']} success, {export_results['failed']} failed")
        self.logger.info(f"Export summary saved to: {summary_file}")
        
        return export_results
    
    def _export_to_onnx(self, model_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Export a single model to ONNX format."""
        
        model_type = model_info["model_type"]
        model_path = model_info["model_path"]
        
        # Skip shallow learning models (not PyTorch)
        if model_type == "shallow":
            return {
                "success": False,
                "error": "ONNX export not supported for shallow learning models"
            }
        
        # Create export directory
        export_model_dir = self.export_dir / model_id
        export_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        onnx_path = export_model_dir / "model.onnx"
        metadata_path = export_model_dir / "metadata.json"
        
        try:
            # Create export script for this model
            script_content = self._generate_export_script(model_type, model_info, onnx_path)
            script_path = export_model_dir / "export_script.py"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run export script
            cmd = [sys.executable, str(script_path)]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0 and onnx_path.exists():
                # Create metadata file
                metadata = {
                    "model_id": model_id,
                    "model_info": model_info,
                    "export_info": {
                        "format": "onnx",
                        "exported_at": datetime.now().isoformat(),
                        "onnx_path": str(onnx_path),
                        "input_shape": [1, 3, 128, 128],
                        "input_names": ["input"],
                        "output_names": ["output"]
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return {
                    "success": True,
                    "export_files": {
                        "onnx": str(onnx_path),
                        "metadata": str(metadata_path)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Export script failed: {result.stderr}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Export script timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }
    
    def _generate_export_script(self, model_type: str, model_info: Dict[str, Any], 
                              onnx_path: Path) -> str:
        """Generate a standalone export script for the model."""
        
        model_path = model_info["model_path"]
        hyperparams = model_info.get("hyperparameters", {})
        num_classes = hyperparams.get("num_classes", 67)
        
        # Find the actual model file
        model_files = model_info.get("model_files", {}).get("pytorch_model", [])
        if model_files:
            actual_model_path = model_files[0]["path"]
        else:
            # Fallback to model.pth in model directory
            actual_model_path = str(Path(model_path) / "model.pth")
        
        script_template = f'''
import torch
import torch.onnx
import sys
from pathlib import Path

def export_model():
    try:
        # Add model paths to sys.path
        model_base_path = Path(__file__).parent.parent.parent.parent
        '''
        
        if model_type == "transfer":
            script_template += f'''
        sys.path.append(str(model_base_path / "image-classifier-transfer" / "src"))
        from models_pytorch import TransferLearningModelPyTorch
        from config import TransferLearningClassifierConfig
        
        # Create model config
        config = TransferLearningClassifierConfig(
            base_model="{hyperparams.get('base_model', 'resnet50')}",
            head_dropout_rate={hyperparams.get('head_dropout_rate', 0.5)},
            dense_units={hyperparams.get('dense_units', [512, 256])}
        )
        
        model = TransferLearningModelPyTorch(config, {num_classes})
        '''
        
        elif model_type == "deep_v1":
            script_template += f'''
        sys.path.append(str(model_base_path / "image-classifier-deep-v1" / "src"))
        from model_improved import DeepLearningV1Improved
        
        model = DeepLearningV1Improved(
            num_classes={num_classes},
            use_residual={hyperparams.get('use_residual', True)}
        )
        '''
        
        elif model_type == "deep_v2":
            script_template += f'''
        sys.path.append(str(model_base_path / "image-classifier-deep-v2" / "src"))
        from model_improved import DeepLearningV2Improved
        
        model = DeepLearningV2Improved(
            num_classes={num_classes},
            architecture="{hyperparams.get('architecture', 'resnet')}"
        )
        '''
        
        script_template += f'''
        
        # Load model weights
        model_path = "{actual_model_path}"
        if not Path(model_path).exists():
            print(f"Model file not found: {{model_path}}")
            return False
            
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Export to ONNX
        dummy_input = torch.randn(1, 3, 128, 128)
        onnx_path = "{onnx_path}"
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={{'input': {{0: 'batch_size'}}, 'output': {{0: 'batch_size'}}}}
        )
        
        print(f"Model exported to ONNX: {{onnx_path}}")
        return True
        
    except Exception as e:
        print(f"Export failed: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_model()
    sys.exit(0 if success else 1)
'''
        
        return script_template
    
    def generate_deployment_package(self, model_ids: List[str], package_name: str = None) -> Path:
        """Generate a deployment package with selected models."""
        
        if not package_name:
            package_name = f"deployment_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        package_dir = self.export_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        deployment_info = {
            "package_name": package_name,
            "created_at": datetime.now().isoformat(),
            "models": {},
            "readme": "This package contains production-ready ML models"
        }
        
        for model_id in model_ids:
            # Copy exported model files
            model_export_dir = self.export_dir / model_id
            if model_export_dir.exists():
                model_package_dir = package_dir / model_id
                model_package_dir.mkdir(exist_ok=True)
                
                # Copy ONNX and metadata files
                for file_path in model_export_dir.glob("*"):
                    if file_path.is_file() and file_path.suffix in [".onnx", ".json"]:
                        import shutil
                        shutil.copy2(file_path, model_package_dir)
                
                # Get model info from registry
                model_info = self.registry.registry["models"].get(model_id, {})
                deployment_info["models"][model_id] = {
                    "model_type": model_info.get("model_type"),
                    "variation": model_info.get("variation"),
                    "dataset": model_info.get("dataset"),
                    "accuracy": model_info.get("best_validation_accuracy", 0.0),
                    "files": {
                        "onnx": f"{model_id}/model.onnx",
                        "metadata": f"{model_id}/metadata.json"
                    }
                }
        
        # Save deployment info
        with open(package_dir / "deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Create README
        readme_content = f"""# Deployment Package: {package_name}

This package contains {len(model_ids)} production-ready ML models exported to ONNX format.

## Models Included:
"""
        for model_id, info in deployment_info["models"].items():
            readme_content += f"- **{model_id}**: {info['model_type']}/{info['variation']} on {info['dataset']} (Accuracy: {info['accuracy']:.4f})\n"
        
        readme_content += """
## Usage:
1. Load ONNX models with your preferred inference framework
2. Use metadata.json for input/output specifications
3. Preprocess inputs to match training format (128x128 RGB images)

## Files:
- `deployment_info.json`: Package metadata and model information
- `{model_id}/model.onnx`: ONNX model file
- `{model_id}/metadata.json`: Model metadata and configuration
"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Deployment package created: {package_dir}")
        return package_dir


def main():
    """Command line interface for production export."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Model Export')
    parser.add_argument('--registry_path', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/production_training/training_registry.json',
                       help='Path to training registry')
    parser.add_argument('--export_dir', type=str,
                       help='Export directory (optional)')
    parser.add_argument('--model_type', type=str,
                       help='Filter by model type')
    parser.add_argument('--dataset', type=str,
                       help='Filter by dataset')
    parser.add_argument('--format', type=str, default='onnx',
                       choices=['onnx'],
                       help='Export format')
    parser.add_argument('--force', action='store_true',
                       help='Force re-export of already exported models')
    parser.add_argument('--package', type=str,
                       help='Create deployment package with best models')
    
    args = parser.parse_args()
    
    # Create exporter
    export_dir = Path(args.export_dir) if args.export_dir else None
    exporter = ProductionExporter(
        registry_path=Path(args.registry_path),
        export_dir=export_dir
    )
    
    if args.package:
        # Create deployment package with best models
        registry = TrainingRegistry(Path(args.registry_path))
        best_models = registry.get_best_models(top_n=10)
        
        if best_models:
            package_dir = exporter.generate_deployment_package(
                model_ids=best_models,
                package_name=args.package
            )
            print(f"Deployment package created: {package_dir}")
        else:
            print("No models found for deployment package")
    else:
        # Export models
        results = exporter.export_models(
            model_type=args.model_type,
            dataset=args.dataset,
            format=args.format,
            force_export=args.force
        )
        
        print(f"Export completed:")
        print(f"  Exported: {results['exported']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Skipped: {results['skipped']}")


if __name__ == "__main__":
    main()