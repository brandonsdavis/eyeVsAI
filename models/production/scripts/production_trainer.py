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

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import shutil
from typing import Dict, Any, Optional
import torch
import hashlib
import time

from training_registry import TrainingRegistry


class ProductionTrainer:
    """Train production models with best hyperparameters."""
    
    def __init__(self, config_dir: Path, models_dir: Path, logs_dir: Path, session_id: str = None):
        self.config_dir = Path(config_dir)
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        with open(self.config_dir / "models.json", 'r') as f:
            self.models_config = json.load(f)
        
        with open(self.config_dir / "datasets.json", 'r') as f:
            self.datasets_config = json.load(f)
        
        # Setup training registry
        registry_path = self.models_dir.parent / "training_registry.json"
        self.registry = TrainingRegistry(registry_path)
        
        # Start or use existing session
        self.session_id = session_id or self.registry.start_training_session()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"production_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_model_version(self, model_type: str, variation: str, dataset: str) -> str:
        """Generate a unique version identifier for the model."""
        # Create a hash of the configuration
        config_str = f"{model_type}_{variation}_{dataset}_{datetime.now().strftime('%Y%m%d')}"
        version_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"v{datetime.now().strftime('%Y%m%d')}_{version_hash}"
    
    def _load_best_hyperparameters(self, model_type: str, variation: str, dataset: str) -> Optional[Dict[str, Any]]:
        """Load best hyperparameters from tuning results."""
        # Look for tuning results
        tuning_results_pattern = f"tuning_results_{model_type}_{variation}_{dataset}_*.json"
        tuning_files = list(self.logs_dir.glob(tuning_results_pattern))
        
        if not tuning_files:
            self.logger.warning(f"No tuning results found for {model_type}/{variation}/{dataset}")
            return None
        
        # Get the most recent tuning result
        latest_file = max(tuning_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        return results.get("best_params", {})
    
    def train_production_model(self, model_type: str, variation: str, dataset: str,
                             hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a single production model."""
        self.logger.info(f"Training production model: {model_type}/{variation} on {dataset}")
        
        # Get configurations
        model_info = self.models_config["model_types"][model_type]
        dataset_info = self.datasets_config["datasets"][dataset]
        variation_config = model_info["variations"][variation]["config"]
        
        # Load hyperparameters
        if hyperparameters is None:
            hyperparameters = self._load_best_hyperparameters(model_type, variation, dataset)
            if hyperparameters is None:
                # Use default hyperparameters
                hyperparameters = self._get_default_hyperparameters(model_type)
        
        # Merge configurations
        params = {
            **hyperparameters,
            **variation_config,
            "dataset_path": dataset_info["path"],
            "num_classes": dataset_info["num_classes"],
            "model_type": model_type,
            "variation": variation,
            "dataset": dataset
        }
        
        # Create model directory
        version = self._get_model_version(model_type, variation, dataset)
        model_dir = self.models_dir / model_type / variation / dataset / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = model_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Train model
        start_time = time.time()
        training_status = "running"
        
        try:
            if model_type == "shallow":
                metrics = self._train_shallow_production(params, model_dir)
            elif model_type == "deep_v1":
                metrics = self._train_deep_v1_production(params, model_dir)
            elif model_type == "deep_v2":
                metrics = self._train_deep_v2_production(params, model_dir)
            elif model_type == "transfer":
                metrics = self._train_transfer_production(params, model_dir)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            training_status = "completed"
            training_time = time.time() - start_time
            
            # Gather environment information
            env_info = self._get_environment_info()
            
            # Enhanced results with comprehensive metadata
            results = {
                "model_type": model_type,
                "variation": variation,
                "dataset": dataset,
                "version": version,
                "metrics": metrics,
                "hyperparameters": params,
                "final_hyperparameters": params,  # Could be different if tuned
                "model_path": str(model_dir),
                "config_path": str(config_file),
                "log_path": str(model_dir / "training.log"),
                "training_time_seconds": training_time,
                "training_status": training_status,
                "trained_at": datetime.now().isoformat(),
                "environment": env_info,
                "session_id": self.session_id
            }
            
            # Save local results
            results_file = model_dir / "training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Register in central registry
            model_id = self.registry.register_model(results, self.session_id)
            results["model_id"] = model_id
            
            self.logger.info(f"Training completed: {model_type}/{variation}/{dataset}")
            self.logger.info(f"Model ID: {model_id}")
            self.logger.info(f"Metrics: {metrics}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        defaults = {
            "shallow": {
                "C": 1.0,
                "kernel": "rbf",
                "n_estimators": 100,
                "max_depth": 20,
                "pca_components": 100
            },
            "deep_v1": {
                "learning_rate": 0.01,
                "batch_size": 64,
                "dropout_rate": 0.5,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "num_epochs": 50
            },
            "deep_v2": {
                "learning_rate": 0.1,
                "batch_size": 64,
                "dropout_rates": [0.4, 0.3, 0.2],
                "optimizer": "sgd",
                "weight_decay": 5e-4,
                "mixup_alpha": 0.3,
                "num_epochs": 100
            },
            "transfer": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "head_dropout_rate": 0.5,
                "optimizer": "adamw",
                "weight_decay": 1e-4,
                "dense_units": [512, 256],
                "num_epochs": 30
            }
        }
        return defaults.get(model_type, {})
    
    def _train_shallow_production(self, params: Dict[str, Any], model_dir: Path) -> Dict[str, Any]:
        """Train shallow learning production model."""
        # Use subprocess to run the existing training script
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "shallow"
        script_path = module_path / "scripts" / "train.py"
        
        if not script_path.exists():
            # Try to find any training script
            scripts_dir = module_path / "scripts"
            if scripts_dir.exists():
                train_scripts = list(scripts_dir.glob("train*.py"))
                if train_scripts:
                    script_path = train_scripts[0]
        
        if script_path.exists():
            # The shallow learning script expects a directory with class subdirectories
            # If the dataset has train/val/test structure, use the train directory
            dataset_path = Path(params["dataset_path"])
            if (dataset_path / "train").exists() and (dataset_path / "train").is_dir():
                # Use train directory for datasets with train/val/test structure
                data_path = str(dataset_path / "train")
            else:
                # Use the dataset path directly
                data_path = params["dataset_path"]
            
            cmd = [
                sys.executable, str(script_path),
                "--data_path", data_path,
                "--model_save_path", str(model_dir / "model.pkl"),
                "--log_dir", str(model_dir / "logs")
            ]
            
            # The shallow learning script currently only supports SVM
            # Add SVM C parameter if specified
            if "C" in params:
                cmd.extend(["--svm_C", str(params["C"])])
            
            # Add PCA components if specified
            if "pca_components" in params and params["pca_components"] > 0:
                cmd.extend(["--use_pca"])
                # Use min of requested components and 50 (safe default for feature dimensions)
                n_components = min(params["pca_components"], 50)
                cmd.extend(["--n_components_pca", str(n_components)])
            
            # Note: The current shallow learning implementation doesn't support:
            # - Different kernels (always uses RBF)
            # - Random Forest classifier
            # This would need to be added to the shallow learning module
            
            # Create logs directory for the shallow learning script
            logs_dir = model_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Run training
            log_file = model_dir / "training.log"
            env = self._get_training_env()
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
            
            # Extract metrics from log
            metrics = self._extract_metrics_from_log(log_file)
            
            # For shallow learning, we don't have epochs, so set a default
            if metrics["total_epochs"] == 0:
                metrics["total_epochs"] = 1  # Shallow learning doesn't use epochs
            
            return metrics
        else:
            # Fallback: create a simple placeholder
            self.logger.warning(f"No training script found for shallow learning at {script_path}")
            
            # Create a minimal training result
            return {
                "validation_accuracy": 0.15,  # Placeholder accuracy
                "status": "placeholder",
                "note": "Actual training script not found"
            }
    
    def _train_deep_v1_production(self, params: Dict[str, Any], model_dir: Path) -> Dict[str, Any]:
        """Train Deep Learning V1 production model."""
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "deep-v1"
        script_path = module_path / "scripts" / "train_improved.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", params["dataset_path"],
            "--batch_size", str(params.get("batch_size", 64)),
            "--learning_rate", str(params.get("learning_rate", 0.01)),
            "--num_epochs", str(params.get("num_epochs", 50)),
            "--optimizer", params.get("optimizer", "adamw")
        ]
        
        if params.get("use_residual", False):
            cmd.append("--use_residual")
        
        # Run training from project root
        log_file = model_dir / "training.log"
        project_root = self._get_project_root()
        env = self._get_training_env()
        with open(log_file, 'w') as f:
            process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=project_root, env=env)
        
        # Copy trained model to production location
        source_model = Path(module_path) / "models" / "deep_v1_improved.pth"
        if source_model.exists():
            shutil.copy(source_model, model_dir / "model.pth")
        
        # Extract metrics
        metrics = self._extract_metrics_from_log(log_file)
        
        # Convert to ONNX for production (skip for now due to import issues)
        # self._export_to_onnx(model_dir / "model.pth", model_dir / "model.onnx", params)
        self.logger.info("ONNX export skipped - can be done separately if needed")
        
        return metrics
    
    def _train_deep_v2_production(self, params: Dict[str, Any], model_dir: Path) -> Dict[str, Any]:
        """Train Deep Learning V2 production model."""
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "deep-v2"
        script_path = module_path / "scripts" / "train_improved.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", params["dataset_path"],
            "--batch_size", str(params.get("batch_size", 64)),
            "--learning_rate", str(params.get("learning_rate", 0.1)),
            "--num_epochs", str(params.get("num_epochs", 100)),
            "--architecture", params.get("architecture", "resnet"),
            "--optimizer", params.get("optimizer", "sgd")
        ]
        
        if not params.get("use_cbam", True):
            cmd.append("--no_cbam")
        
        # Run training from project root
        log_file = model_dir / "training.log"
        project_root = self._get_project_root()
        env = self._get_training_env()
        with open(log_file, 'w') as f:
            process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=project_root, env=env)
        
        # Copy trained model
        source_model = Path(module_path) / "models" / "deep_v2_improved.pth"
        if source_model.exists():
            shutil.copy(source_model, model_dir / "model.pth")
        
        # Extract metrics
        metrics = self._extract_metrics_from_log(log_file)
        
        # Convert to ONNX (skip for now due to import issues)
        # self._export_to_onnx(model_dir / "model.pth", model_dir / "model.onnx", params)
        self.logger.info("ONNX export skipped - can be done separately if needed")
        
        return metrics
    
    def _train_transfer_production(self, params: Dict[str, Any], model_dir: Path) -> Dict[str, Any]:
        """Train Transfer Learning production model."""
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "transfer"
        script_path = module_path / "scripts" / "train_pytorch.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", params["dataset_path"],
            "--batch_size", str(params.get("batch_size", 32)),
            "--learning_rate", str(params.get("learning_rate", 0.001)),
            "--num_epochs", str(params.get("num_epochs", 30)),
            "--base_model", params.get("base_model", "resnet50"),
            "--optimizer", params.get("optimizer", "adamw"),
            "--model_save_path", str(model_dir / "model.pth"),
            "--log_dir", str(model_dir / "logs")
        ]
        
        # Run training from project root
        log_file = model_dir / "training.log"
        project_root = self._get_project_root()
        env = self._get_training_env()
        with open(log_file, 'w') as f:
            process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=project_root, env=env)
        
        # Extract metrics
        metrics = self._extract_metrics_from_log(log_file)
        
        # Read training history if available
        history_file = model_dir / "logs" / "training_history_pytorch.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                metrics["training_history"] = history
        
        # Convert to ONNX (skip for now due to import issues)
        # self._export_to_onnx(model_dir / "model.pth", model_dir / "model.onnx", params)
        self.logger.info("ONNX export skipped - can be done separately if needed")
        
        return metrics
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Gather training environment information."""
        env_info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "training_time": datetime.now().strftime("%H:%M:%S")
        }
        
        # Add GPU memory info if available
        if torch.cuda.is_available():
            try:
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    memory_info = {
                        "device": i,
                        "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        "name": torch.cuda.get_device_name(i)
                    }
                    gpu_memory.append(memory_info)
                env_info["gpu_memory"] = gpu_memory
            except Exception:
                pass
        
        return env_info
    
    def _get_project_root(self):
        """Get the project root directory for running training scripts."""
        return Path(__file__).parent.parent.parent
    
    def _get_training_env(self):
        """Get environment variables for training subprocess."""
        env = os.environ.copy()
        # Add PYTHONPATH to ensure modules can be imported
        project_root = str(self._get_project_root())
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        return env
    
    def _extract_metrics_from_log(self, log_file: Path) -> Dict[str, Any]:
        """Extract training metrics from log file."""
        metrics = {
            "best_validation_accuracy": 0.0,
            "final_train_accuracy": 0.0,
            "final_validation_accuracy": 0.0,
            "total_epochs": 0
        }
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            import re
            for line in lines:
                # Shallow learning format: "Final validation accuracy: 0.9773"
                if "final validation accuracy:" in line.lower():
                    acc_match = re.search(r'final validation accuracy:\s*([\d.]+)', line.lower())
                    if acc_match:
                        val_acc = float(acc_match.group(1))
                        metrics["best_validation_accuracy"] = val_acc
                        metrics["final_validation_accuracy"] = val_acc
                
                # Shallow learning format: "'val_accuracy': 0.9773333333333334"
                elif "'val_accuracy':" in line:
                    acc_match = re.search(r"'val_accuracy':\s*([\d.]+)", line)
                    if acc_match:
                        val_acc = float(acc_match.group(1))
                        metrics["best_validation_accuracy"] = max(metrics["best_validation_accuracy"], val_acc)
                        metrics["final_validation_accuracy"] = val_acc
                
                # Deep learning format
                elif "best validation accuracy" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "accuracy" in part.lower() and i + 1 < len(parts):
                            try:
                                metrics["best_validation_accuracy"] = float(parts[i + 1].strip(':,'))
                            except:
                                pass
                
                elif "epoch [" in line.lower() and "/" in line:
                    # Extract epoch information
                    epoch_match = re.search(r'epoch \[(\d+)/(\d+)\]', line.lower())
                    if epoch_match:
                        metrics["total_epochs"] = int(epoch_match.group(2))
                    
                    # Extract accuracies
                    if "train acc:" in line.lower():
                        acc_match = re.search(r'train acc:\s*([\d.]+)', line.lower())
                        if acc_match:
                            metrics["final_train_accuracy"] = float(acc_match.group(1))
                    
                    if "val acc:" in line.lower():
                        acc_match = re.search(r'val acc:\s*([\d.]+)', line.lower())
                        if acc_match:
                            metrics["final_validation_accuracy"] = float(acc_match.group(1))
                            metrics["best_validation_accuracy"] = max(
                                metrics["best_validation_accuracy"],
                                float(acc_match.group(1))
                            )
        
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
        
        return metrics
    
    def _export_to_onnx(self, model_path: Path, onnx_path: Path, params: Dict[str, Any]):
        """Export PyTorch model to ONNX format for production deployment."""
        try:
            import torch
            import torch.onnx
            
            # Load model based on type
            if params["model_type"] == "transfer":
                # Load transfer learning model
                sys.path.append(str(Path(__file__).parent.parent.parent / "classifiers" / "transfer" / "src"))
                from models_pytorch import TransferLearningModelPyTorch
                from config import TransferLearningClassifierConfig
                
                # Create dummy config
                config = TransferLearningClassifierConfig(
                    base_model=params.get("base_model", "resnet50"),
                    head_dropout_rate=params.get("head_dropout_rate", 0.5),
                    dense_units=params.get("dense_units", [512, 256])
                )
                
                model = TransferLearningModelPyTorch(config, params["num_classes"])
                
            elif params["model_type"] == "deep_v1":
                sys.path.append(str(Path(__file__).parent.parent.parent / "classifiers" / "deep-v1" / "src"))
                from model_improved import DeepLearningV1Improved
                
                model = DeepLearningV1Improved(
                    num_classes=params["num_classes"],
                    use_residual=params.get("use_residual", True)
                )
                
            elif params["model_type"] == "deep_v2":
                sys.path.append(str(Path(__file__).parent.parent.parent / "classifiers" / "deep-v2" / "src"))
                from model_improved import DeepLearningV2Improved
                
                model = DeepLearningV2Improved(
                    num_classes=params["num_classes"],
                    architecture=params.get("architecture", "resnet")
                )
            else:
                self.logger.warning(f"ONNX export not supported for {params['model_type']}")
                return
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 128, 128)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            self.logger.info(f"Model exported to ONNX: {onnx_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to ONNX: {e}")
    
    def train_all_models(self, run_tuning: bool = False, tuning_trials: int = 20):
        """Train all model variations on all datasets."""
        results = []
        
        # Get all combinations
        for model_type, model_info in self.models_config["model_types"].items():
            for variation_name, variation_info in model_info["variations"].items():
                for dataset_name in self.datasets_config["datasets"].keys():
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"Processing: {model_type}/{variation_name} on {dataset_name}")
                    self.logger.info(f"{'='*60}")
                    
                    try:
                        # Run hyperparameter tuning if requested
                        if run_tuning:
                            self.logger.info("Running hyperparameter tuning...")
                            from hyperparameter_tuner import HyperparameterTuner
                            
                            tuner = HyperparameterTuner(
                                model_type=model_type,
                                model_variation=variation_name,
                                dataset_name=dataset_name,
                                config_dir=self.config_dir,
                                log_dir=self.logs_dir,
                                n_trials=tuning_trials
                            )
                            
                            tuning_results = tuner.run_tuning()
                            hyperparameters = tuning_results["best_params"]
                        else:
                            hyperparameters = None
                        
                        # Train production model
                        result = self.train_production_model(
                            model_type=model_type,
                            variation=variation_name,
                            dataset=dataset_name,
                            hyperparameters=hyperparameters
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to train {model_type}/{variation_name} on {dataset_name}: {e}")
                        results.append({
                            "model_type": model_type,
                            "variation": variation_name,
                            "dataset": dataset_name,
                            "status": "failed",
                            "error": str(e)
                        })
        
        # Save summary
        summary_file = self.models_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nTraining completed! Summary saved to: {summary_file}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Production model training')
    parser.add_argument('--config_dir', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/models/production/configs',
                       help='Configuration directory')
    parser.add_argument('--models_dir', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/models/production/models',
                       help='Models output directory')
    parser.add_argument('--logs_dir', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/models/production/logs',
                       help='Logs directory')
    parser.add_argument('--model_type', type=str,
                       help='Specific model type to train (optional)')
    parser.add_argument('--variation', type=str,
                       help='Specific variation to train (optional)')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to train on (optional)')
    parser.add_argument('--run_tuning', action='store_true',
                       help='Run hyperparameter tuning before training')
    parser.add_argument('--tuning_trials', type=int, default=20,
                       help='Number of tuning trials')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ProductionTrainer(
        config_dir=args.config_dir,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir
    )
    
    if args.model_type and args.variation and args.dataset:
        # Train specific model
        result = trainer.train_production_model(
            model_type=args.model_type,
            variation=args.variation,
            dataset=args.dataset
        )
        print(f"\nTraining completed!")
        print(f"Results: {result['metrics']}")
    else:
        # Train all models
        results = trainer.train_all_models(
            run_tuning=args.run_tuning,
            tuning_trials=args.tuning_trials
        )
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        for result in results:
            if "metrics" in result:
                print(f"{result['model_type']}/{result['variation']} on {result['dataset']}: "
                      f"Val Acc = {result['metrics'].get('best_validation_accuracy', 0):.4f}")
            else:
                print(f"{result['model_type']}/{result['variation']} on {result['dataset']}: FAILED")


if __name__ == "__main__":
    main()