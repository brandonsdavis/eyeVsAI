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
Centralized training registry for tracking all trained models.
Provides comprehensive logging and model management.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
import torch
import uuid


class TrainingRegistry:
    """Centralized registry for tracking all trained models and their metadata."""
    
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new one
        self.registry = self._load_registry()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load existing registry or create a new one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                return registry
            except Exception as e:
                self.logger.warning(f"Failed to load registry, creating new one: {e}")
        
        # Create new registry structure
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_models": 0,
            "models": {},
            "sessions": {},
            "statistics": {
                "by_model_type": {},
                "by_dataset": {},
                "best_models": {}
            }
        }
    
    def _save_registry(self):
        """Save registry to disk."""
        self.registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def start_training_session(self, session_name: str = None) -> str:
        """Start a new training session and return session ID."""
        session_id = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.registry["sessions"][session_id] = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "models_trained": [],
            "total_models": 0,
            "successful_models": 0,
            "failed_models": 0
        }
        
        self._save_registry()
        self.logger.info(f"Started training session: {session_id}")
        return session_id
    
    def end_training_session(self, session_id: str):
        """Mark training session as completed."""
        if session_id in self.registry["sessions"]:
            self.registry["sessions"][session_id]["status"] = "completed"
            self.registry["sessions"][session_id]["completed_at"] = datetime.now().isoformat()
            self._save_registry()
            self.logger.info(f"Completed training session: {session_id}")
    
    def register_model(self, model_info: Dict[str, Any], session_id: str = None) -> str:
        """Register a trained model in the registry."""
        
        # Generate unique model ID
        model_id = self._generate_model_id(model_info)
        
        # Enhanced model record
        model_record = {
            "model_id": model_id,
            "registered_at": datetime.now().isoformat(),
            "session_id": session_id,
            
            # Model identification
            "model_type": model_info.get("model_type"),
            "variation": model_info.get("variation"),
            "dataset": model_info.get("dataset"),
            "version": model_info.get("version"),
            
            # Performance metrics
            "metrics": model_info.get("metrics", {}),
            "best_validation_accuracy": model_info.get("metrics", {}).get("best_validation_accuracy", 0.0),
            "training_time_seconds": model_info.get("training_time_seconds"),
            
            # Training configuration
            "hyperparameters": model_info.get("hyperparameters", {}),
            "final_hyperparameters": model_info.get("final_hyperparameters"),
            "tuning_results": model_info.get("tuning_results"),
            
            # File paths
            "model_path": str(model_info.get("model_path", "")),
            "config_path": str(model_info.get("config_path", "")),
            "log_path": str(model_info.get("log_path", "")),
            "checkpoint_path": str(model_info.get("checkpoint_path", "")),
            
            # Model file information
            "model_files": self._scan_model_files(model_info.get("model_path")),
            
            # Training environment
            "gpu_used": model_info.get("gpu_used"),
            "cuda_version": model_info.get("cuda_version"),
            "pytorch_version": model_info.get("pytorch_version"),
            
            # Status and flags
            "training_status": model_info.get("training_status", "completed"),
            "export_status": "pending",
            "production_ready": False,
            
            # Additional metadata
            "notes": model_info.get("notes", ""),
            "tags": model_info.get("tags", [])
        }
        
        # Add to registry
        self.registry["models"][model_id] = model_record
        self.registry["total_models"] = len(self.registry["models"])
        
        # Update session if provided
        if session_id and session_id in self.registry["sessions"]:
            self.registry["sessions"][session_id]["models_trained"].append(model_id)
            self.registry["sessions"][session_id]["total_models"] += 1
            if model_record["training_status"] == "completed":
                self.registry["sessions"][session_id]["successful_models"] += 1
            else:
                self.registry["sessions"][session_id]["failed_models"] += 1
        
        # Update statistics
        self._update_statistics()
        
        # Save registry
        self._save_registry()
        
        self.logger.info(f"Registered model: {model_id}")
        return model_id
    
    def _generate_model_id(self, model_info: Dict[str, Any]) -> str:
        """Generate unique model ID based on model characteristics."""
        identifier = f"{model_info.get('model_type')}_{model_info.get('variation')}_{model_info.get('dataset')}_{model_info.get('version', '')}"
        model_hash = hashlib.md5(identifier.encode()).hexdigest()[:12]
        return f"model_{model_hash}"
    
    def _scan_model_files(self, model_path: Optional[str]) -> Dict[str, Any]:
        """Scan model directory for files and their information."""
        if not model_path:
            return {}
        
        model_dir = Path(model_path)
        if not model_dir.exists():
            return {}
        
        files_info = {}
        
        # Common model files to look for
        file_patterns = {
            "pytorch_model": ["*.pth", "*.pt", "model.pth"],
            "onnx_model": ["*.onnx"],
            "config": ["config.json", "*.json"],
            "logs": ["*.log"],
            "checkpoints": ["checkpoint*.pth", "*.ckpt"],
            "results": ["*results*.json", "training_history*.json"]
        }
        
        for category, patterns in file_patterns.items():
            found_files = []
            for pattern in patterns:
                found_files.extend(list(model_dir.rglob(pattern)))
            
            if found_files:
                files_info[category] = []
                for file_path in found_files:
                    try:
                        stat = file_path.stat()
                        files_info[category].append({
                            "path": str(file_path),
                            "size_mb": stat.st_size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "exists": True
                        })
                    except Exception:
                        files_info[category].append({
                            "path": str(file_path),
                            "exists": False
                        })
        
        return files_info
    
    def _update_statistics(self):
        """Update registry statistics."""
        stats = {
            "by_model_type": {},
            "by_dataset": {},
            "best_models": {}
        }
        
        for model_id, model in self.registry["models"].items():
            # By model type
            model_type = model.get("model_type", "unknown")
            if model_type not in stats["by_model_type"]:
                stats["by_model_type"][model_type] = {"count": 0, "avg_accuracy": 0.0, "best_accuracy": 0.0}
            
            stats["by_model_type"][model_type]["count"] += 1
            accuracy = model.get("best_validation_accuracy", 0.0)
            stats["by_model_type"][model_type]["best_accuracy"] = max(
                stats["by_model_type"][model_type]["best_accuracy"], accuracy
            )
            
            # By dataset
            dataset = model.get("dataset", "unknown")
            if dataset not in stats["by_dataset"]:
                stats["by_dataset"][dataset] = {"count": 0, "avg_accuracy": 0.0, "best_accuracy": 0.0, "best_model": None}
            
            stats["by_dataset"][dataset]["count"] += 1
            if accuracy > stats["by_dataset"][dataset]["best_accuracy"]:
                stats["by_dataset"][dataset]["best_accuracy"] = accuracy
                stats["by_dataset"][dataset]["best_model"] = model_id
        
        # Find overall best models
        all_models = [(model_id, model.get("best_validation_accuracy", 0.0)) 
                     for model_id, model in self.registry["models"].items()]
        all_models.sort(key=lambda x: x[1], reverse=True)
        
        stats["best_models"] = {
            "overall_best": all_models[0][0] if all_models else None,
            "top_10": [model_id for model_id, _ in all_models[:10]]
        }
        
        self.registry["statistics"] = stats
    
    def update_export_status(self, model_id: str, export_status: str, export_files: Dict[str, str] = None):
        """Update model export status."""
        if model_id in self.registry["models"]:
            self.registry["models"][model_id]["export_status"] = export_status
            self.registry["models"][model_id]["exported_at"] = datetime.now().isoformat()
            
            if export_files:
                self.registry["models"][model_id]["export_files"] = export_files
            
            if export_status == "completed":
                self.registry["models"][model_id]["production_ready"] = True
            
            self._save_registry()
            self.logger.info(f"Updated export status for {model_id}: {export_status}")
    
    def get_models_for_export(self, export_status: str = "pending", 
                            model_type: str = None, dataset: str = None) -> List[Dict[str, Any]]:
        """Get models that need to be exported."""
        models_to_export = []
        
        for model_id, model in self.registry["models"].items():
            # Filter by export status
            if model.get("export_status") != export_status:
                continue
            
            # Filter by model type if specified
            if model_type and model.get("model_type") != model_type:
                continue
            
            # Filter by dataset if specified
            if dataset and model.get("dataset") != dataset:
                continue
            
            # Only include models that have PyTorch files
            model_files = model.get("model_files", {})
            if "pytorch_model" not in model_files or not model_files["pytorch_model"]:
                continue
            
            models_to_export.append({
                "model_id": model_id,
                "model_info": model
            })
        
        return models_to_export
    
    def get_best_models(self, by_dataset: bool = False, by_model_type: bool = False, 
                       top_n: int = 10) -> Dict[str, Any]:
        """Get best performing models."""
        if by_dataset:
            return {dataset: info["best_model"] 
                   for dataset, info in self.registry["statistics"]["by_dataset"].items()
                   if info["best_model"]}
        
        if by_model_type:
            best_by_type = {}
            for model_type in self.registry["statistics"]["by_model_type"].keys():
                type_models = [(model_id, model.get("best_validation_accuracy", 0.0))
                             for model_id, model in self.registry["models"].items()
                             if model.get("model_type") == model_type]
                type_models.sort(key=lambda x: x[1], reverse=True)
                if type_models:
                    best_by_type[model_type] = type_models[0][0]
            return best_by_type
        
        # Return top N overall
        return self.registry["statistics"]["best_models"]["top_10"][:top_n]
    
    def get_training_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get training summary for a session or overall."""
        if session_id:
            return self.registry["sessions"].get(session_id, {})
        
        return {
            "total_models": self.registry["total_models"],
            "statistics": self.registry["statistics"],
            "recent_sessions": list(self.registry["sessions"].keys())[-5:],
            "models_pending_export": len(self.get_models_for_export("pending")),
            "production_ready_models": len([m for m in self.registry["models"].values() 
                                          if m.get("production_ready", False)])
        }
    
    def export_registry_report(self, output_path: Path):
        """Export comprehensive registry report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "registry_summary": self.get_training_summary(),
            "detailed_models": self.registry["models"],
            "sessions": self.registry["sessions"],
            "statistics": self.registry["statistics"]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Exported registry report to: {output_path}")


def main():
    """Command line interface for training registry."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Registry Management')
    parser.add_argument('--registry_path', type=str,
                       default='/home/brandond/Projects/pvt/personal/eyeVsAI/production_training/training_registry.json',
                       help='Path to registry file')
    parser.add_argument('--action', type=str, required=True,
                       choices=['summary', 'export_pending', 'best_models', 'report'],
                       help='Action to perform')
    parser.add_argument('--output', type=str,
                       help='Output file path for reports')
    
    args = parser.parse_args()
    
    registry = TrainingRegistry(args.registry_path)
    
    if args.action == 'summary':
        summary = registry.get_training_summary()
        print(json.dumps(summary, indent=2))
    
    elif args.action == 'export_pending':
        pending_models = registry.get_models_for_export("pending")
        print(f"Models pending export: {len(pending_models)}")
        for model in pending_models:
            print(f"  {model['model_id']}: {model['model_info']['model_type']}/{model['model_info']['variation']}")
    
    elif args.action == 'best_models':
        best_overall = registry.get_best_models(top_n=10)
        best_by_dataset = registry.get_best_models(by_dataset=True)
        best_by_type = registry.get_best_models(by_model_type=True)
        
        print("Best Models Report:")
        print(f"Top 10 Overall: {best_overall}")
        print(f"Best by Dataset: {json.dumps(best_by_dataset, indent=2)}")
        print(f"Best by Type: {json.dumps(best_by_type, indent=2)}")
    
    elif args.action == 'report':
        output_path = Path(args.output) if args.output else Path("registry_report.json")
        registry.export_registry_report(output_path)
        print(f"Report exported to: {output_path}")


if __name__ == "__main__":
    main()