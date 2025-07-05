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
Master script to orchestrate training of all production models.

This script will:
1. Optionally run hyperparameter tuning for each model/dataset combination
2. Train production models with best hyperparameters
3. Export models in production-ready formats
4. Generate comprehensive reports
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import concurrent.futures
import logging
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ProductionPipeline:
    """Orchestrate the entire production training pipeline."""
    
    def __init__(self, base_dir: Path, parallel_jobs: int = 1):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        self.parallel_jobs = parallel_jobs
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configurations
        with open(self.config_dir / "models.json", 'r') as f:
            self.models_config = json.load(f)
        
        with open(self.config_dir / "datasets.json", 'r') as f:
            self.datasets_config = json.load(f)
    
    def _setup_logging(self):
        """Setup logging for the pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_all_combinations(self) -> List[Dict[str, str]]:
        """Get all model/dataset combinations to train."""
        combinations = []
        
        for model_type, model_info in self.models_config["model_types"].items():
            for variation_name in model_info["variations"].keys():
                for dataset_name in self.datasets_config["datasets"].keys():
                    combinations.append({
                        "model_type": model_type,
                        "variation": variation_name,
                        "dataset": dataset_name
                    })
        
        return combinations
    
    def run_hyperparameter_tuning(self, combination: Dict[str, str], n_trials: int = 10) -> Dict[str, Any]:
        """Run hyperparameter tuning for a single combination."""
        self.logger.info(f"Running hyperparameter tuning for {combination}")
        
        cmd = [
            sys.executable,
            str(self.base_dir / "scripts" / "hyperparameter_tuner.py"),
            "--model_type", combination["model_type"],
            "--model_variation", combination["variation"],
            "--dataset", combination["dataset"],
            "--n_trials", str(n_trials),
            "--config_dir", str(self.config_dir),
            "--log_dir", str(self.logs_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Tuning completed for {combination}")
            return {"status": "success", "combination": combination}
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Tuning failed for {combination}: {e}")
            return {"status": "failed", "combination": combination, "error": str(e)}
    
    def train_production_model(self, combination: Dict[str, str]) -> Dict[str, Any]:
        """Train a single production model."""
        self.logger.info(f"Training production model for {combination}")
        
        cmd = [
            sys.executable,
            str(self.base_dir / "scripts" / "production_trainer.py"),
            "--model_type", combination["model_type"],
            "--variation", combination["variation"],
            "--dataset", combination["dataset"],
            "--config_dir", str(self.config_dir),
            "--models_dir", str(self.models_dir),
            "--logs_dir", str(self.logs_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Training completed for {combination}")
            return {"status": "success", "combination": combination}
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed for {combination}: {e}")
            return {"status": "failed", "combination": combination, "error": str(e)}
    
    def run_pipeline(self, run_tuning: bool = False, tuning_trials: int = 10,
                    specific_models: List[str] = None, specific_datasets: List[str] = None):
        """Run the complete pipeline."""
        self.logger.info("Starting production training pipeline")
        
        # Get combinations to process
        all_combinations = self.get_all_combinations()
        
        # Filter if specific models or datasets requested
        if specific_models:
            all_combinations = [c for c in all_combinations if c["model_type"] in specific_models]
        if specific_datasets:
            all_combinations = [c for c in all_combinations if c["dataset"] in specific_datasets]
        
        self.logger.info(f"Processing {len(all_combinations)} model/dataset combinations")
        
        # Phase 1: Hyperparameter tuning (if requested)
        if run_tuning:
            self.logger.info("Phase 1: Hyperparameter tuning")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                tuning_futures = {
                    executor.submit(self.run_hyperparameter_tuning, combo, tuning_trials): combo
                    for combo in all_combinations
                }
                
                tuning_results = []
                for future in concurrent.futures.as_completed(tuning_futures):
                    result = future.result()
                    tuning_results.append(result)
                    
                    if result["status"] == "success":
                        self.logger.info(f"✓ Tuning completed: {result['combination']}")
                    else:
                        self.logger.error(f"✗ Tuning failed: {result['combination']}")
        
        # Phase 2: Production training
        self.logger.info("Phase 2: Production model training")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            training_futures = {
                executor.submit(self.train_production_model, combo): combo
                for combo in all_combinations
            }
            
            training_results = []
            for future in concurrent.futures.as_completed(training_futures):
                result = future.result()
                training_results.append(result)
                
                if result["status"] == "success":
                    self.logger.info(f"✓ Training completed: {result['combination']}")
                else:
                    self.logger.error(f"✗ Training failed: {result['combination']}")
        
        # Phase 3: Generate reports
        self.logger.info("Phase 3: Generating reports")
        self.generate_reports()
        
        self.logger.info("Pipeline completed!")
        
        return {
            "total_combinations": len(all_combinations),
            "successful_trainings": len([r for r in training_results if r["status"] == "success"]),
            "failed_trainings": len([r for r in training_results if r["status"] == "failed"])
        }
    
    def generate_reports(self):
        """Generate comprehensive reports of all trained models."""
        self.logger.info("Generating comprehensive reports")
        
        # Collect all results
        all_results = []
        
        for model_path in self.models_dir.rglob("training_results.json"):
            with open(model_path, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        
        if not all_results:
            self.logger.warning("No training results found")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Extract metrics
        if 'metrics' in df.columns:
            metrics_df = pd.json_normalize(df['metrics'])
            df = pd.concat([df, metrics_df], axis=1)
        
        # Save detailed CSV report
        csv_file = self.results_dir / f"all_models_report_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved detailed report to: {csv_file}")
        
        # Generate visualizations
        self._generate_visualizations(df)
        
        # Generate summary report
        self._generate_summary_report(df)
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots."""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Accuracy comparison by model type
        plt.figure(figsize=(12, 8))
        if 'best_validation_accuracy' in df.columns:
            pivot_data = df.pivot_table(
                values='best_validation_accuracy',
                index='model_type',
                columns='dataset',
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
            plt.title('Average Validation Accuracy by Model Type and Dataset')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'accuracy_heatmap.png', dpi=300)
            plt.close()
        
        # 2. Model performance distribution
        plt.figure(figsize=(12, 6))
        if 'best_validation_accuracy' in df.columns:
            df.boxplot(column='best_validation_accuracy', by='model_type', figsize=(12, 6))
            plt.title('Model Performance Distribution by Type')
            plt.suptitle('')  # Remove default title
            plt.ylabel('Validation Accuracy')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'performance_distribution.png', dpi=300)
            plt.close()
        
        # 3. Best models ranking
        plt.figure(figsize=(14, 10))
        if 'best_validation_accuracy' in df.columns:
            top_models = df.nlargest(20, 'best_validation_accuracy')
            top_models['model_name'] = top_models['model_type'] + '/' + top_models['variation'] + '/' + top_models['dataset']
            
            plt.barh(top_models['model_name'], top_models['best_validation_accuracy'])
            plt.xlabel('Validation Accuracy')
            plt.title('Top 20 Models by Validation Accuracy')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'top_models.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate a summary report."""
        summary = {
            "report_generated": datetime.now().isoformat(),
            "total_models_trained": len(df),
            "model_types": df['model_type'].unique().tolist(),
            "datasets": df['dataset'].unique().tolist(),
            "best_overall_model": None,
            "best_by_dataset": {},
            "best_by_model_type": {},
            "summary_statistics": {}
        }
        
        if 'best_validation_accuracy' in df.columns:
            # Best overall model
            best_idx = df['best_validation_accuracy'].idxmax()
            best_model = df.loc[best_idx]
            summary["best_overall_model"] = {
                "model": f"{best_model['model_type']}/{best_model['variation']}",
                "dataset": best_model['dataset'],
                "accuracy": float(best_model['best_validation_accuracy']),
                "version": best_model.get('version', 'unknown')
            }
            
            # Best by dataset
            for dataset in df['dataset'].unique():
                dataset_df = df[df['dataset'] == dataset]
                if len(dataset_df) > 0 and 'best_validation_accuracy' in dataset_df.columns:
                    best_idx = dataset_df['best_validation_accuracy'].idxmax()
                    best = dataset_df.loc[best_idx]
                    summary["best_by_dataset"][dataset] = {
                        "model": f"{best['model_type']}/{best['variation']}",
                        "accuracy": float(best['best_validation_accuracy'])
                    }
            
            # Best by model type
            for model_type in df['model_type'].unique():
                model_df = df[df['model_type'] == model_type]
                if len(model_df) > 0 and 'best_validation_accuracy' in model_df.columns:
                    best_idx = model_df['best_validation_accuracy'].idxmax()
                    best = model_df.loc[best_idx]
                    summary["best_by_model_type"][model_type] = {
                        "variation": best['variation'],
                        "dataset": best['dataset'],
                        "accuracy": float(best['best_validation_accuracy'])
                    }
            
            # Summary statistics
            summary["summary_statistics"] = {
                "mean_accuracy": float(df['best_validation_accuracy'].mean()),
                "std_accuracy": float(df['best_validation_accuracy'].std()),
                "min_accuracy": float(df['best_validation_accuracy'].min()),
                "max_accuracy": float(df['best_validation_accuracy'].max())
            }
        
        # Save summary
        summary_file = self.results_dir / f"summary_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved summary report to: {summary_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("PRODUCTION TRAINING SUMMARY")
        print("="*80)
        print(f"Total models trained: {summary['total_models_trained']}")
        
        if summary["best_overall_model"]:
            print(f"\nBest overall model: {summary['best_overall_model']['model']} "
                  f"on {summary['best_overall_model']['dataset']} "
                  f"(accuracy: {summary['best_overall_model']['accuracy']:.4f})")
        
        print("\nBest models by dataset:")
        for dataset, info in summary["best_by_dataset"].items():
            print(f"  {dataset}: {info['model']} (accuracy: {info['accuracy']:.4f})")
        
        print("\nBest models by type:")
        for model_type, info in summary["best_by_model_type"].items():
            print(f"  {model_type}: {info['variation']} on {info['dataset']} "
                  f"(accuracy: {info['accuracy']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Train all production models with automated pipeline'
    )
    parser.add_argument('--base_dir', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/production_training',
                       help='Base directory for production training')
    parser.add_argument('--run_tuning', action='store_true',
                       help='Run hyperparameter tuning before training')
    parser.add_argument('--tuning_trials', type=int, default=10,
                       help='Number of hyperparameter tuning trials per model')
    parser.add_argument('--parallel_jobs', type=int, default=1,
                       help='Number of parallel training jobs')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['shallow', 'deep_v1', 'deep_v2', 'transfer'],
                       help='Specific model types to train')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['combined', 'vegetables', 'pets', 'street_foods', 'instruments'],
                       help='Specific datasets to train on')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProductionPipeline(
        base_dir=args.base_dir,
        parallel_jobs=args.parallel_jobs
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        run_tuning=args.run_tuning,
        tuning_trials=args.tuning_trials,
        specific_models=args.models,
        specific_datasets=args.datasets
    )
    
    print(f"\nPipeline completed!")
    print(f"Total combinations: {results['total_combinations']}")
    print(f"Successful trainings: {results['successful_trainings']}")
    print(f"Failed trainings: {results['failed_trainings']}")


if __name__ == "__main__":
    main()