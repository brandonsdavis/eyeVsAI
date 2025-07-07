#!/usr/bin/env python3
"""
Comprehensive cleanup script for the eyeVsAI project.

This script helps manage disk space by cleaning up various types of generated files:
- Training logs and temporary files
- Model checkpoints and intermediate files  
- Downloaded datasets (with confirmation)
- Old trained models (with age-based filtering)
- Cache files and build artifacts

Usage:
    python cleanup.py --help                    # Show all options
    python cleanup.py --dry-run                 # Preview what would be deleted
    python cleanup.py --logs                    # Clean only log files
    python cleanup.py --checkpoints             # Clean training checkpoints
    python cleanup.py --old-models --days 30    # Clean models older than 30 days
    python cleanup.py --all --confirm           # Clean everything (with confirmation)
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import json

class ProjectCleaner:
    """Comprehensive cleanup utility for the eyeVsAI project."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.total_size_freed = 0
        self.files_deleted = 0
        self.dirs_deleted = 0
        
        # Define cleanup targets
        self.cleanup_patterns = {
            'logs': {
                'patterns': ['**/*.log', '**/logs/**/*'],
                'description': 'Training and debug log files',
                'paths': [
                    'logs/',
                    'models/classifiers/*/logs/',
                    'models/production/logs/',
                    '/tmp/test_logs/',
                ]
            },
            'checkpoints': {
                'patterns': ['**/checkpoints/**/*', '**/checkpoint_*.pth'],
                'description': 'Training checkpoints and intermediate saves',
                'paths': [
                    'models/checkpoints/',
                    'models/classifiers/*/checkpoints/',
                ]
            },
            'temp_models': {
                'patterns': ['**/temp_*.pth', '**/temp_*.pkl', '**/temp_*.h5'],
                'description': 'Temporary model files',
                'paths': [
                    'models/classifiers/*/models/temp_*',
                    'models/production/models/temp_*',
                ]
            },
            'cache': {
                'patterns': ['**/__pycache__/**/*', '**/*.pyc', '**/.pytest_cache/**/*'],
                'description': 'Python cache files and build artifacts',
                'paths': [
                    '**/__pycache__/',
                    '**/.pytest_cache/',
                ]
            },
            'jupyter_checkpoints': {
                'patterns': ['**/.ipynb_checkpoints/**/*'],
                'description': 'Jupyter notebook checkpoints',
                'paths': [
                    '**/.ipynb_checkpoints/',
                ]
            },
            'old_models': {
                'patterns': ['**/*.pth', '**/*.pkl', '**/*.h5'],
                'description': 'Old trained model files',
                'age_based': True,
                'paths': [
                    'models/classifiers/*/models/',
                    'models/production/models/',
                ]
            },
            'data_downloads': {
                'patterns': ['**/downloads/**/*'],
                'description': 'Downloaded dataset files (WARNING: Large files)',
                'dangerous': True,
                'paths': [
                    'models/data/downloads/',
                ]
            },
            'notebooks_data': {
                'patterns': ['**/notebooks/data/**/*'],
                'description': 'Data copied into notebook directories',
                'paths': [
                    'models/classifiers/*/notebooks/data/',
                ]
            }
        }
    
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
    
    def is_file_old(self, path: Path, days: int) -> bool:
        """Check if file is older than specified days."""
        try:
            file_time = datetime.fromtimestamp(path.stat().st_mtime)
            cutoff_time = datetime.now() - timedelta(days=days)
            return file_time < cutoff_time
        except (OSError, ValueError):
            return False
    
    def find_cleanup_targets(self, category: str, days: int = 30) -> List[Tuple[Path, int]]:
        """Find files/directories to clean for a given category."""
        if category not in self.cleanup_patterns:
            return []
        
        config = self.cleanup_patterns[category]
        targets = []
        
        # Process each pattern
        for pattern in config['patterns']:
            for path in self.project_root.rglob(pattern.lstrip('**/')):
                if path.exists() and 'venv' not in str(path):
                    # Skip if age-based filtering is enabled and file is not old enough
                    if config.get('age_based', False) and not self.is_file_old(path, days):
                        continue
                    
                    size = self.get_file_size(path)
                    targets.append((path, size))
        
        # Sort by size (largest first)
        targets.sort(key=lambda x: x[1], reverse=True)
        return targets
    
    def clean_category(self, category: str, days: int = 30, confirm: bool = False) -> bool:
        """Clean files for a specific category."""
        if category not in self.cleanup_patterns:
            print(f"❌ Unknown category: {category}")
            return False
        
        config = self.cleanup_patterns[category]
        targets = self.find_cleanup_targets(category, days)
        
        if not targets:
            print(f"✅ No {config['description'].lower()} found to clean")
            return True
        
        total_size = sum(size for _, size in targets)
        
        print(f"\n🎯 {config['description']}")
        print(f"📁 Files/directories to clean: {len(targets)}")
        print(f"💾 Total size: {self.format_size(total_size)}")
        
        # Show dangerous warning
        if config.get('dangerous', False):
            print(f"⚠️  WARNING: This will delete large dataset files!")
            print(f"⚠️  You may need to re-download data for training.")
        
        # Show some examples
        if len(targets) > 0:
            print(f"\n📋 Examples of what will be cleaned:")
            for path, size in targets[:5]:
                relative_path = path.relative_to(self.project_root)
                print(f"  • {relative_path} ({self.format_size(size)})")
            if len(targets) > 5:
                print(f"  • ... and {len(targets) - 5} more items")
        
        if self.dry_run:
            print(f"🔍 DRY RUN: Would delete {len(targets)} items ({self.format_size(total_size)})")
            return True
        
        # Confirmation for dangerous operations
        if config.get('dangerous', False) and not confirm:
            response = input(f"\n❓ Are you sure you want to delete {config['description'].lower()}? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Cancelled")
                return False
        
        # Perform cleanup
        success_count = 0
        for path, size in targets:
            try:
                if path.is_file():
                    path.unlink()
                    self.files_deleted += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    self.dirs_deleted += 1
                
                self.total_size_freed += size
                success_count += 1
                
            except (OSError, PermissionError) as e:
                print(f"⚠️  Could not delete {path}: {e}")
        
        print(f"✅ Cleaned {success_count}/{len(targets)} items ({self.format_size(sum(size for _, size in targets[:success_count]))})")
        return True
    
    def analyze_disk_usage(self) -> None:
        """Analyze current disk usage and suggest cleanup options."""
        print("📊 DISK USAGE ANALYSIS")
        print("=" * 50)
        
        # Get total project size
        total_size = self.get_file_size(self.project_root)
        print(f"📁 Total project size: {self.format_size(total_size)}")
        
        # Analyze each category
        category_sizes = {}
        for category, config in self.cleanup_patterns.items():
            targets = self.find_cleanup_targets(category)
            if targets:
                category_size = sum(size for _, size in targets)
                category_sizes[category] = (len(targets), category_size, config['description'])
        
        if category_sizes:
            print(f"\n🎯 CLEANUP OPPORTUNITIES")
            print(f"{'Category':<20} {'Files':<8} {'Size':<12} {'Description'}")
            print("-" * 70)
            
            for category, (count, size, desc) in sorted(category_sizes.items(), key=lambda x: x[1][1], reverse=True):
                danger_flag = "⚠️ " if self.cleanup_patterns[category].get('dangerous', False) else "  "
                print(f"{danger_flag}{category:<18} {count:<8} {self.format_size(size):<12} {desc}")
        
        print(f"\n💡 SUGGESTIONS")
        print(f"• Run 'python cleanup.py --logs' to clean log files")
        print(f"• Run 'python cleanup.py --checkpoints' to clean training checkpoints") 
        print(f"• Run 'python cleanup.py --cache' to clean Python cache files")
        print(f"• Run 'python cleanup.py --old-models --days 30' to clean old models")
        print(f"• Run 'python cleanup.py --dry-run --all' to preview full cleanup")
    
    def clean_all(self, days: int = 30, confirm: bool = False) -> None:
        """Clean all categories (except dangerous ones unless confirmed)."""
        print("🧹 COMPREHENSIVE CLEANUP")
        print("=" * 50)
        
        # Order categories by safety (safest first)
        safe_categories = ['cache', 'jupyter_checkpoints', 'logs', 'temp_models']
        moderate_categories = ['checkpoints', 'old_models', 'notebooks_data'] 
        dangerous_categories = ['data_downloads']
        
        all_categories = safe_categories + moderate_categories
        if confirm:
            all_categories += dangerous_categories
        
        for category in all_categories:
            if category in self.cleanup_patterns:
                self.clean_category(category, days=days, confirm=confirm)
        
        print(f"\n📊 CLEANUP SUMMARY")
        print(f"🗑️  Files deleted: {self.files_deleted}")
        print(f"📁 Directories deleted: {self.dirs_deleted}")
        print(f"💾 Total space freed: {self.format_size(self.total_size_freed)}")

def main():
    parser = argparse.ArgumentParser(
        description='Cleanup utility for the eyeVsAI project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Action flags
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze disk usage and show cleanup suggestions')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be deleted without actually deleting')
    
    # Category-specific cleanup
    parser.add_argument('--logs', action='store_true', help='Clean log files')
    parser.add_argument('--checkpoints', action='store_true', help='Clean training checkpoints')
    parser.add_argument('--temp-models', action='store_true', help='Clean temporary model files')
    parser.add_argument('--cache', action='store_true', help='Clean Python cache files')
    parser.add_argument('--jupyter-checkpoints', action='store_true', help='Clean Jupyter checkpoints')
    parser.add_argument('--old-models', action='store_true', help='Clean old model files')
    parser.add_argument('--notebooks-data', action='store_true', help='Clean notebook data copies')
    parser.add_argument('--data-downloads', action='store_true', help='Clean downloaded datasets (DANGEROUS)')
    
    # Comprehensive cleanup
    parser.add_argument('--all', action='store_true', help='Clean all categories')
    
    # Options
    parser.add_argument('--days', type=int, default=30,
                       help='Age threshold in days for old files (default: 30)')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompts for dangerous operations')
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Default to analyze if no action specified
    if not any([args.analyze, args.dry_run, args.logs, args.checkpoints, args.temp_models, 
                args.cache, args.jupyter_checkpoints, args.old_models, args.notebooks_data,
                args.data_downloads, args.all]):
        args.analyze = True
    
    # Initialize cleaner
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"❌ Project root directory not found: {project_root}")
        sys.exit(1)
    
    cleaner = ProjectCleaner(project_root, dry_run=args.dry_run)
    
    # Execute requested actions
    if args.analyze:
        cleaner.analyze_disk_usage()
    
    if args.all:
        cleaner.clean_all(days=args.days, confirm=args.confirm)
    else:
        # Individual category cleanup
        categories = []
        if args.logs: categories.append('logs')
        if args.checkpoints: categories.append('checkpoints')
        if args.temp_models: categories.append('temp_models')
        if args.cache: categories.append('cache')
        if args.jupyter_checkpoints: categories.append('jupyter_checkpoints')
        if args.old_models: categories.append('old_models')
        if args.notebooks_data: categories.append('notebooks_data')
        if args.data_downloads: categories.append('data_downloads')
        
        for category in categories:
            cleaner.clean_category(category, days=args.days, confirm=args.confirm)

if __name__ == '__main__':
    main()