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

import optuna
from optuna.trial import TrialState
import json
import logging
import os
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
import subprocess
import sys
import time
from datetime import datetime


class HyperparameterTuner:
    """Automated hyperparameter tuning using Optuna."""
    
    def __init__(self, model_type: str, model_variation: str, dataset_name: str,
                 config_dir: Path, log_dir: Path, n_trials: int = 20):
        self.model_type = model_type
        self.model_variation = model_variation
        self.dataset_name = dataset_name
        self.config_dir = Path(config_dir)
        self.log_dir = Path(log_dir)
        self.n_trials = n_trials
        
        # Load configurations
        with open(self.config_dir / "models.json", 'r') as f:
            self.models_config = json.load(f)
        
        with open(self.config_dir / "datasets.json", 'r') as f:
            self.datasets_config = json.load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Create study name
        self.study_name = f"{model_type}_{model_variation}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _setup_logging(self):
        """Setup logging for the tuner."""
        log_file = self.log_dir / f"tuning_{self.model_type}_{self.model_variation}_{self.dataset_name}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_objective(self) -> Callable:
        """Create the objective function for Optuna."""
        model_info = self.models_config["model_types"][self.model_type]
        dataset_info = self.datasets_config["datasets"][self.dataset_name]
        hyperparameter_space = model_info["hyperparameter_space"]
        
        def objective(trial):
            # Sample hyperparameters based on model type
            params = self._sample_hyperparameters(trial, hyperparameter_space)
            
            # Add fixed parameters
            params.update({
                "model_type": self.model_type,
                "model_variation": self.model_variation,
                "dataset_name": self.dataset_name,
                "dataset_path": dataset_info["path"],
                "num_classes": dataset_info["num_classes"]
            })
            
            # Add variation-specific config
            variation_config = model_info["variations"][self.model_variation]["config"]
            params.update(variation_config)
            
            # Train model and get validation accuracy
            val_accuracy = self._train_and_evaluate(trial, params)
            
            return val_accuracy
        
        return objective
    
    def _sample_hyperparameters(self, trial, hyperparameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from the search space."""
        params = {}
        
        for param_name, param_values in hyperparameter_space.items():
            if isinstance(param_values, list):
                # Check if all values are numbers
                if all(isinstance(v, (int, float)) for v in param_values if v is not None):
                    # Numerical parameter
                    if all(isinstance(v, int) for v in param_values if v is not None):
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            min(v for v in param_values if v is not None),
                            max(v for v in param_values if v is not None)
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            min(v for v in param_values if v is not None),
                            max(v for v in param_values if v is not None),
                            log=True if param_name in ["learning_rate", "weight_decay"] else False
                        )
                else:
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values, dict):
                # Nested parameters
                if "min" in param_values and "max" in param_values:
                    if param_values.get("type") == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, param_values["min"], param_values["max"]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_values["min"], param_values["max"],
                            log=param_values.get("log", False)
                        )
        
        return params
    
    def _train_and_evaluate(self, trial, params: Dict[str, Any]) -> float:
        """Train model with given parameters and return validation accuracy."""
        self.logger.info(f"Trial {trial.number}: Training with params: {params}")
        
        # Create unique trial directory
        trial_dir = self.log_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        
        # Save trial parameters
        with open(trial_dir / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        try:
            if self.model_type == "shallow":
                val_accuracy = self._train_shallow_model(params, trial_dir)
            elif self.model_type == "deep_v1":
                val_accuracy = self._train_deep_v1_model(params, trial_dir, trial)
            elif self.model_type == "deep_v2":
                val_accuracy = self._train_deep_v2_model(params, trial_dir, trial)
            elif self.model_type == "transfer":
                val_accuracy = self._train_transfer_model(params, trial_dir, trial)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.logger.info(f"Trial {trial.number}: Validation accuracy = {val_accuracy:.4f}")
            
            # Save trial result
            with open(trial_dir / "result.json", 'w') as f:
                json.dump({"val_accuracy": val_accuracy}, f)
            
            return val_accuracy
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            return 0.0
    
    def _train_shallow_model(self, params: Dict[str, Any], trial_dir: Path) -> float:
        """Train shallow learning model using subprocess to avoid import issues."""
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
        
        if not script_path.exists():
            self.logger.error(f"No training script found for shallow learning at {script_path}")
            return 0.0
        
        # Handle different dataset structures for shallow learning
        dataset_path = params["dataset_path"]
        if params.get("dataset_name") == "vegetables":
            # Vegetables dataset has train/test/validation structure, use train for shallow learning
            dataset_path = str(Path(dataset_path) / "train")
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", dataset_path,
            "--model_save_path", str(trial_dir / "model.pkl"),
            "--log_dir", str(trial_dir)
        ]
        
        # Add additional parameters based on what the script actually accepts
        if "C" in params:
            cmd.extend(["--svm_C", str(params["C"])])
        if "pca_components" in params:
            # Ensure PCA components doesn't exceed available features (typically 77 for HOG+LBP)
            pca_components = min(params["pca_components"], 50)  # Cap at 50 to be safe
            cmd.extend(["--n_components_pca", str(pca_components)])
            cmd.append("--use_pca")  # Enable PCA when pca_components is specified
        
        # Run training from project root with correct working directory
        log_file = trial_dir / "training.log"
        
        # Set working directory to project root so relative paths work correctly
        project_root = Path(__file__).parent.parent.parent
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=300, cwd=project_root)
            
            # Extract validation accuracy from log
            val_accuracy = self._extract_accuracy_from_log(log_file)
            return val_accuracy
            
        except subprocess.TimeoutExpired:
            self.logger.error("Shallow learning training timed out")
            return 0.0
        except Exception as e:
            self.logger.error(f"Shallow learning training failed: {e}")
            return 0.0
    
    def _train_deep_v1_model(self, params: Dict[str, Any], trial_dir: Path, trial) -> float:
        """Train Deep Learning V1 model."""
        # Prepare command
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "deep-v1"
        script_path = module_path / "scripts" / "train_improved.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", params["dataset_path"],
            "--batch_size", str(params.get("batch_size", 64)),
            "--learning_rate", str(params.get("learning_rate", 0.01)),
            "--num_epochs", str(50),  # Increased for better convergence
            "--optimizer", params.get("optimizer", "adamw")
        ]
        
        if params.get("use_residual", False):
            cmd.append("--use_residual")
        
        # Run training from project root with correct working directory
        log_file = trial_dir / "training.log"
        
        # Set working directory to project root so relative paths work correctly
        project_root = Path(__file__).parent.parent.parent
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=600, cwd=project_root)  # 10 minute timeout
        except subprocess.TimeoutExpired:
            self.logger.error(f"Deep V1 training timed out for trial")
            return 0.0
        
        # Extract validation accuracy from log
        val_accuracy = self._extract_accuracy_from_log(log_file)
        return val_accuracy
    
    def _train_deep_v2_model(self, params: Dict[str, Any], trial_dir: Path, trial) -> float:
        """Train Deep Learning V2 model."""
        # Prepare command
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "deep-v2"
        script_path = module_path / "scripts" / "train_improved.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", params["dataset_path"],
            "--batch_size", str(params.get("batch_size", 64)),
            "--learning_rate", str(params.get("learning_rate", 0.1)),
            "--num_epochs", str(50),  # Increased for better convergence
            "--architecture", params.get("architecture", "resnet"),
            "--optimizer", params.get("optimizer", "sgd")
        ]
        
        if not params.get("use_cbam", True):
            cmd.append("--no_cbam")
        
        # Run training from project root with correct working directory
        log_file = trial_dir / "training.log"
        
        # Set working directory to project root so relative paths work correctly
        project_root = Path(__file__).parent.parent.parent
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=600, cwd=project_root)  # 10 minute timeout
        except subprocess.TimeoutExpired:
            self.logger.error(f"Training timed out for trial")
            return 0.0
        
        # Extract validation accuracy
        val_accuracy = self._extract_accuracy_from_log(log_file)
        return val_accuracy
    
    def _train_transfer_model(self, params: Dict[str, Any], trial_dir: Path, trial) -> float:
        """Train Transfer Learning model."""
        # Prepare command
        module_path = Path(__file__).parent.parent.parent / "classifiers" / "transfer"
        script_path = module_path / "scripts" / "train_pytorch.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--data_path", params["dataset_path"],
            "--batch_size", str(params.get("batch_size", 32)),
            "--learning_rate", str(params.get("learning_rate", 0.001)),
            "--num_epochs", str(40),
            "--base_model", params.get("base_model", "resnet50"),
            "--optimizer", params.get("optimizer", "adamw"),
            "--model_save_path", str(trial_dir / "model.pth"),
            "--log_dir", str(trial_dir)
        ]
        
        # Run training from project root with correct working directory
        log_file = trial_dir / "training.log"
        
        # Set working directory to project root so relative paths work correctly
        project_root = Path(__file__).parent.parent.parent
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=600, cwd=project_root)  # 10 minute timeout
        except subprocess.TimeoutExpired:
            self.logger.error(f"Training timed out for trial")
            return 0.0
        
        # Extract validation accuracy
        val_accuracy = self._extract_accuracy_from_log(log_file)
        return val_accuracy
    
    def _extract_accuracy_from_log(self, log_file: Path) -> float:
        """Extract best validation accuracy from training log."""
        best_accuracy = 0.0
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Look for validation accuracy patterns
                    if "best validation accuracy" in line.lower():
                        # Extract number from line
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "accuracy" in part.lower() and i + 1 < len(parts):
                                try:
                                    acc = float(parts[i + 1].strip(':,'))
                                    best_accuracy = max(best_accuracy, acc)
                                except:
                                    pass
                    elif "final validation accuracy" in line.lower():
                        # Extract shallow learning final accuracy
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "accuracy:" in part.lower() and i + 1 < len(parts):
                                try:
                                    acc = float(parts[i + 1].strip())
                                    best_accuracy = max(best_accuracy, acc)
                                except:
                                    pass
                    elif "val acc:" in line.lower() or "val_acc:" in line.lower():
                        # Extract validation accuracy
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if ("acc:" in part.lower() or "accuracy:" in part.lower()) and i + 1 < len(parts):
                                try:
                                    acc = float(parts[i + 1].strip(','))
                                    best_accuracy = max(best_accuracy, acc)
                                except:
                                    pass
        except Exception as e:
            self.logger.error(f"Error extracting accuracy: {e}")
        
        return best_accuracy
    
    def run_tuning(self) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        self.logger.info(f"Starting hyperparameter tuning for {self.study_name}")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        study.optimize(
            self.create_objective(),
            n_trials=self.n_trials,
            timeout=None,
            catch=(Exception,)
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best validation accuracy: {best_value:.4f}")
        
        # Save study results
        results = {
            "study_name": self.study_name,
            "model_type": self.model_type,
            "model_variation": self.model_variation,
            "dataset_name": self.dataset_name,
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "datetime": datetime.now().isoformat()
        }
        
        results_file = self.log_dir / f"tuning_results_{self.study_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for models')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['shallow', 'deep_v1', 'deep_v2', 'transfer'],
                       help='Type of model to tune')
    parser.add_argument('--model_variation', type=str, required=True,
                       help='Model variation to tune')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['combined', 'vegetables', 'pets', 'street_foods', 'instruments'],
                       help='Dataset to use')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of tuning trials')
    parser.add_argument('--config_dir', type=str, 
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/models/production/configs',
                       help='Configuration directory')
    parser.add_argument('--log_dir', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/models/production/logs',
                       help='Log directory')
    
    args = parser.parse_args()
    
    # Create tuner
    tuner = HyperparameterTuner(
        model_type=args.model_type,
        model_variation=args.model_variation,
        dataset_name=args.dataset,
        config_dir=args.config_dir,
        log_dir=args.log_dir,
        n_trials=args.n_trials
    )
    
    # Run tuning
    results = tuner.run_tuning()
    
    print(f"\nTuning completed!")
    print(f"Best validation accuracy: {results['best_value']:.4f}")
    print(f"Best parameters: {results['best_params']}")


if __name__ == "__main__":
    main()