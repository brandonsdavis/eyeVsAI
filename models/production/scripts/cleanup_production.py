#!/usr/bin/env python3
"""
Production training cleanup utility.

This script provides specialized cleanup for production training artifacts:
- Failed training runs and partial models
- Hyperparameter tuning trials and logs
- Intermediate checkpoints and temporary files
- Old model versions with retention policies
- Training registry cleanup

Usage:
    python cleanup_production.py --help
    python cleanup_production.py --analyze
    python cleanup_production.py --failed-runs
    python cleanup_production.py --old-versions --keep 5
    python cleanup_production.py --tuning-logs --days 7
"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

class ProductionCleaner:
    """Specialized cleanup for production training artifacts."""
    
    def __init__(self, production_dir: Path, dry_run: bool = False):
        self.production_dir = Path(production_dir)
        self.dry_run = dry_run
        self.total_size_freed = 0
        self.files_deleted = 0
        
        # Load training registry if it exists
        self.registry_path = self.production_dir / "training_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load the training registry."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"models": {}, "sessions": {}}
    
    def _save_registry(self) -> None:
        """Save the updated training registry."""
        if not self.dry_run:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
    
    def get_file_size(self, path: Path) -> int:
        """Get size of file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0
    
    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
    
    def find_failed_runs(self) -> List[Tuple[Path, str, int]]:
        """Find failed training runs that can be cleaned up."""
        failed_runs = []
        models_dir = self.production_dir / "models"
        
        if not models_dir.exists():
            return failed_runs
        
        # Look for model directories without proper completion
        for model_type_dir in models_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
                
            for variation_dir in model_type_dir.iterdir():
                if not variation_dir.is_dir():
                    continue
                    
                for dataset_dir in variation_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                        
                    for version_dir in dataset_dir.iterdir():
                        if not version_dir.is_dir():
                            continue
                        
                        # Check if this is a failed run
                        failure_reason = self._check_failed_run(version_dir)
                        if failure_reason:
                            size = self.get_file_size(version_dir)
                            failed_runs.append((version_dir, failure_reason, size))
        
        return sorted(failed_runs, key=lambda x: x[2], reverse=True)
    
    def _check_failed_run(self, version_dir: Path) -> Optional[str]:
        """Check if a training run failed and return the reason."""
        # Check for required files
        required_files = ["model.pth", "training_results.json", "config.json"]
        missing_files = [f for f in required_files if not (version_dir / f).exists()]
        
        if missing_files:
            return f"Missing files: {', '.join(missing_files)}"
        
        # Check training results for failure indicators
        results_file = version_dir / "training_results.json"
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Check training status
            status = results.get("training_status", "unknown")
            if status == "failed":
                return "Training marked as failed"
            
            # Check for suspiciously low accuracy (likely failed)
            metrics = results.get("metrics", {})
            val_acc = metrics.get("best_validation_accuracy", 0)
            if val_acc < 0.1:  # Less than 10% accuracy is suspicious
                return f"Suspiciously low accuracy: {val_acc:.3f}"
                
        except (json.JSONDecodeError, IOError, KeyError):
            return "Corrupted training results"
        
        return None
    
    def find_old_versions(self, keep_count: int = 5) -> List[Tuple[Path, datetime, int]]:
        """Find old model versions that can be cleaned up, keeping the newest ones."""
        old_versions = []
        models_dir = self.production_dir / "models"
        
        if not models_dir.exists():
            return old_versions
        
        # Group versions by model/variation/dataset
        version_groups = {}
        
        for model_type_dir in models_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
                
            for variation_dir in model_type_dir.iterdir():
                if not variation_dir.is_dir():
                    continue
                    
                for dataset_dir in variation_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    
                    group_key = f"{model_type_dir.name}/{variation_dir.name}/{dataset_dir.name}"
                    version_groups[group_key] = []
                    
                    for version_dir in dataset_dir.iterdir():
                        if not version_dir.is_dir():
                            continue
                        
                        # Get creation time
                        try:
                            creation_time = datetime.fromtimestamp(version_dir.stat().st_ctime)
                            size = self.get_file_size(version_dir)
                            version_groups[group_key].append((version_dir, creation_time, size))
                        except OSError:
                            continue
        
        # For each group, find versions to delete (keeping newest)
        for group_key, versions in version_groups.items():
            if len(versions) <= keep_count:
                continue
            
            # Sort by creation time (newest first)
            versions.sort(key=lambda x: x[1], reverse=True)
            
            # Mark older versions for deletion
            for version_info in versions[keep_count:]:
                old_versions.append(version_info)
        
        return sorted(old_versions, key=lambda x: x[2], reverse=True)
    
    def find_tuning_logs(self, days: int = 7) -> List[Tuple[Path, int]]:
        """Find old hyperparameter tuning logs."""
        old_logs = []
        logs_dir = self.production_dir / "logs"
        
        if not logs_dir.exists():
            return old_logs
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Find tuning-related files
        for log_file in logs_dir.rglob("*"):
            if not log_file.is_file():
                continue
            
            # Check if it's a tuning log
            if any(keyword in log_file.name for keyword in ["tuning", "optuna", "trial"]):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        size = self.get_file_size(log_file)
                        old_logs.append((log_file, size))
                except OSError:
                    continue
        
        return sorted(old_logs, key=lambda x: x[1], reverse=True)
    
    def find_temp_files(self) -> List[Tuple[Path, int]]:
        """Find temporary files that can be cleaned up."""
        temp_files = []
        
        # Patterns for temporary files
        temp_patterns = [
            "**/temp_*",
            "**/tmp_*",
            "**/*.tmp",
            "**/trial_*",
            "**/.optuna_*",
        ]
        
        for pattern in temp_patterns:
            for path in self.production_dir.rglob(pattern):
                if path.is_file() or path.is_dir():
                    size = self.get_file_size(path)
                    temp_files.append((path, size))
        
        return sorted(temp_files, key=lambda x: x[1], reverse=True)
    
    def clean_failed_runs(self) -> None:
        """Clean up failed training runs."""
        failed_runs = self.find_failed_runs()
        
        if not failed_runs:
            print("✅ No failed training runs found")
            return
        
        total_size = sum(size for _, _, size in failed_runs)
        
        print(f"\n🎯 FAILED TRAINING RUNS")
        print(f"📁 Failed runs found: {len(failed_runs)}")
        print(f"💾 Total size: {self.format_size(total_size)}")
        
        # Show examples
        print(f"\n📋 Failed runs to clean:")
        for path, reason, size in failed_runs[:10]:
            relative_path = path.relative_to(self.production_dir)
            print(f"  • {relative_path} - {reason} ({self.format_size(size)})")
        
        if len(failed_runs) > 10:
            print(f"  • ... and {len(failed_runs) - 10} more")
        
        if self.dry_run:
            print(f"🔍 DRY RUN: Would delete {len(failed_runs)} failed runs")
            return
        
        response = input(f"\n❓ Delete {len(failed_runs)} failed runs? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        # Delete failed runs
        success_count = 0
        for path, reason, size in failed_runs:
            try:
                shutil.rmtree(path)
                self.total_size_freed += size
                self.files_deleted += 1
                success_count += 1
                
                # Remove from registry if present
                self._remove_from_registry(path)
                
            except (OSError, PermissionError) as e:
                print(f"⚠️  Could not delete {path}: {e}")
        
        self._save_registry()
        print(f"✅ Cleaned {success_count}/{len(failed_runs)} failed runs")
    
    def clean_old_versions(self, keep_count: int = 5) -> None:
        """Clean old model versions, keeping the newest ones."""
        old_versions = self.find_old_versions(keep_count)
        
        if not old_versions:
            print(f"✅ No old versions found (keeping {keep_count} newest per model)")
            return
        
        total_size = sum(size for _, _, size in old_versions)
        
        print(f"\n🎯 OLD MODEL VERSIONS")
        print(f"📁 Old versions found: {len(old_versions)} (keeping {keep_count} newest per model)")
        print(f"💾 Total size: {self.format_size(total_size)}")
        
        # Show examples
        print(f"\n📋 Old versions to clean:")
        for path, creation_time, size in old_versions[:10]:
            relative_path = path.relative_to(self.production_dir)
            age_days = (datetime.now() - creation_time).days
            print(f"  • {relative_path} - {age_days} days old ({self.format_size(size)})")
        
        if len(old_versions) > 10:
            print(f"  • ... and {len(old_versions) - 10} more")
        
        if self.dry_run:
            print(f"🔍 DRY RUN: Would delete {len(old_versions)} old versions")
            return
        
        response = input(f"\n❓ Delete {len(old_versions)} old versions? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        # Delete old versions
        success_count = 0
        for path, _, size in old_versions:
            try:
                shutil.rmtree(path)
                self.total_size_freed += size
                self.files_deleted += 1
                success_count += 1
                
                # Remove from registry if present
                self._remove_from_registry(path)
                
            except (OSError, PermissionError) as e:
                print(f"⚠️  Could not delete {path}: {e}")
        
        self._save_registry()
        print(f"✅ Cleaned {success_count}/{len(old_versions)} old versions")
    
    def clean_tuning_logs(self, days: int = 7) -> None:
        """Clean old hyperparameter tuning logs."""
        old_logs = self.find_tuning_logs(days)
        
        if not old_logs:
            print(f"✅ No tuning logs older than {days} days found")
            return
        
        total_size = sum(size for _, size in old_logs)
        
        print(f"\n🎯 OLD TUNING LOGS")
        print(f"📁 Old tuning logs: {len(old_logs)} (older than {days} days)")
        print(f"💾 Total size: {self.format_size(total_size)}")
        
        if self.dry_run:
            print(f"🔍 DRY RUN: Would delete {len(old_logs)} tuning logs")
            return
        
        # Delete old logs
        for path, size in old_logs:
            try:
                path.unlink()
                self.total_size_freed += size
                self.files_deleted += 1
            except (OSError, PermissionError) as e:
                print(f"⚠️  Could not delete {path}: {e}")
        
        print(f"✅ Cleaned {len(old_logs)} tuning logs")
    
    def clean_temp_files(self) -> None:
        """Clean temporary files."""
        temp_files = self.find_temp_files()
        
        if not temp_files:
            print("✅ No temporary files found")
            return
        
        total_size = sum(size for _, size in temp_files)
        
        print(f"\n🎯 TEMPORARY FILES")
        print(f"📁 Temporary files: {len(temp_files)}")
        print(f"💾 Total size: {self.format_size(total_size)}")
        
        if self.dry_run:
            print(f"🔍 DRY RUN: Would delete {len(temp_files)} temporary files")
            return
        
        # Delete temp files
        for path, size in temp_files:
            try:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                self.total_size_freed += size
                self.files_deleted += 1
            except (OSError, PermissionError) as e:
                print(f"⚠️  Could not delete {path}: {e}")
        
        print(f"✅ Cleaned {len(temp_files)} temporary files")
    
    def _remove_from_registry(self, path: Path) -> None:
        """Remove a model from the training registry."""
        path_str = str(path)
        models_to_remove = []
        
        for model_id, model_info in self.registry.get("models", {}).items():
            if model_info.get("model_path", "").startswith(path_str):
                models_to_remove.append(model_id)
        
        for model_id in models_to_remove:
            del self.registry["models"][model_id]
    
    def analyze(self) -> None:
        """Analyze production directory and show cleanup opportunities."""
        print("📊 PRODUCTION TRAINING ANALYSIS")
        print("=" * 50)
        
        # Total production size
        total_size = self.get_file_size(self.production_dir)
        print(f"📁 Total production directory size: {self.format_size(total_size)}")
        
        # Analyze different categories
        categories = {
            "Failed runs": self.find_failed_runs(),
            "Old versions (keep 5)": self.find_old_versions(5),
            "Tuning logs (>7 days)": self.find_tuning_logs(7),
            "Temporary files": self.find_temp_files(),
        }
        
        print(f"\n🎯 CLEANUP OPPORTUNITIES")
        print(f"{'Category':<20} {'Count':<8} {'Size':<12}")
        print("-" * 50)
        
        total_savings = 0
        for category, items in categories.items():
            if category == "Failed runs":
                count = len(items)
                size = sum(size for _, _, size in items)
            elif category == "Old versions (keep 5)":
                count = len(items)
                size = sum(size for _, _, size in items)
            else:
                count = len(items)
                size = sum(size for _, size in items)
            
            if count > 0:
                print(f"{category:<20} {count:<8} {self.format_size(size):<12}")
                total_savings += size
        
        if total_savings > 0:
            print(f"\n💾 Total potential savings: {self.format_size(total_savings)}")
            print(f"\n💡 SUGGESTIONS")
            print(f"• Run with --failed-runs to clean failed training attempts")
            print(f"• Run with --old-versions to clean old model versions") 
            print(f"• Run with --tuning-logs to clean old hyperparameter logs")
            print(f"• Run with --temp-files to clean temporary files")
        else:
            print(f"\n✅ Production directory is already clean!")

def main():
    parser = argparse.ArgumentParser(description='Production training cleanup utility')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze production directory and show cleanup opportunities')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be deleted without actually deleting')
    
    # Cleanup categories
    parser.add_argument('--failed-runs', action='store_true',
                       help='Clean failed training runs')
    parser.add_argument('--old-versions', action='store_true',
                       help='Clean old model versions')
    parser.add_argument('--tuning-logs', action='store_true',
                       help='Clean old hyperparameter tuning logs')
    parser.add_argument('--temp-files', action='store_true',
                       help='Clean temporary files')
    parser.add_argument('--all', action='store_true',
                       help='Clean all categories')
    
    # Options
    parser.add_argument('--keep', type=int, default=5,
                       help='Number of newest versions to keep (default: 5)')
    parser.add_argument('--days', type=int, default=7,
                       help='Age threshold in days for logs (default: 7)')
    parser.add_argument('--production-dir', type=str, default='.',
                       help='Production directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Default to analyze if no action specified
    if not any([args.analyze, args.failed_runs, args.old_versions, 
                args.tuning_logs, args.temp_files, args.all]):
        args.analyze = True
    
    # Initialize cleaner
    production_dir = Path(args.production_dir).resolve()
    if not production_dir.exists():
        print(f"❌ Production directory not found: {production_dir}")
        sys.exit(1)
    
    cleaner = ProductionCleaner(production_dir, dry_run=args.dry_run)
    
    # Execute requested actions
    if args.analyze:
        cleaner.analyze()
    
    if args.all:
        cleaner.clean_failed_runs()
        cleaner.clean_old_versions(keep_count=args.keep)
        cleaner.clean_tuning_logs(days=args.days)
        cleaner.clean_temp_files()
    else:
        if args.failed_runs:
            cleaner.clean_failed_runs()
        if args.old_versions:
            cleaner.clean_old_versions(keep_count=args.keep)
        if args.tuning_logs:
            cleaner.clean_tuning_logs(days=args.days)
        if args.temp_files:
            cleaner.clean_temp_files()
    
    # Show summary
    if cleaner.total_size_freed > 0:
        print(f"\n📊 CLEANUP SUMMARY")
        print(f"🗑️  Items deleted: {cleaner.files_deleted}")
        print(f"💾 Total space freed: {cleaner.format_size(cleaner.total_size_freed)}")

if __name__ == '__main__':
    main()