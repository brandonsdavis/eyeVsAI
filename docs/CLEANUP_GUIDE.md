# Project Cleanup Guide

This guide provides comprehensive procedures for managing disk space in the eyeVsAI project by cleaning up training artifacts, logs, and temporary files.

## 📊 Current Space Usage Analysis

The project can consume significant disk space during training:

- **Total project size**: ~47GB
- **Main space consumers**:
  - **Notebook data copies**: 63.6GB (duplicated datasets)
  - **Training checkpoints**: 48.6GB (intermediate training saves)
  - **Downloaded datasets**: 4.4GB (raw data)
  - **Production models**: 2.3GB (trained models)
  - **Training logs**: 255MB (debug and training logs)

## 🛠️ Cleanup Tools

The project includes two specialized cleanup scripts:

### 1. Main Cleanup Script (`cleanup.py`)

Handles general project cleanup across all directories.

### 2. Production Cleanup Script (`models/production/scripts/cleanup_production.py`)

Specialized cleanup for production training artifacts.

## 🚀 Quick Start

### Analyze Current Usage

```bash
# Analyze entire project
python cleanup.py --analyze

# Analyze production training only
cd models/production
python scripts/cleanup_production.py --analyze
```

### Safe Quick Cleanup (Recommended)

```bash
# Clean safe items without confirmation
python cleanup.py --cache --logs --dry-run  # Preview first
python cleanup.py --cache --logs             # Execute cleanup

# Clean production failed runs and temp files
cd models/production  
python scripts/cleanup_production.py --failed-runs --temp-files
```

### Preview All Cleanup (Safe)

```bash
# See everything that would be cleaned without deleting
python cleanup.py --dry-run --all
```

## 📋 Cleanup Categories

### 🟢 Safe to Clean (No Confirmation Needed)

#### Cache Files
```bash
python cleanup.py --cache
```
- Python `__pycache__` directories
- Pytest cache files
- Build artifacts
- **Impact**: Minimal, automatically regenerated

#### Log Files  
```bash
python cleanup.py --logs
```
- Training logs (`.log` files)
- Debug output files
- **Impact**: Lose training history/debugging info

#### Jupyter Checkpoints
```bash
python cleanup.py --jupyter-checkpoints
```
- Jupyter notebook checkpoint files
- **Impact**: Lose notebook autosave history

#### Temporary Files
```bash
python cleanup.py --temp-models
cd models/production
python scripts/cleanup_production.py --temp-files
```
- Temporary model files
- Failed training artifacts
- **Impact**: None, safe to delete

### 🟡 Moderate Risk (Confirmation Required)

#### Training Checkpoints
```bash
python cleanup.py --checkpoints
```
- Intermediate training saves
- Model checkpoints during training
- **Impact**: Cannot resume interrupted training
- **Space savings**: ~48GB

#### Old Models (Age-Based)
```bash
python cleanup.py --old-models --days 30
cd models/production
python scripts/cleanup_production.py --old-versions --keep 5
```
- Models older than specified days
- Keep only newest N versions per model
- **Impact**: Lose older trained models

#### Hyperparameter Tuning Logs
```bash
cd models/production
python scripts/cleanup_production.py --tuning-logs --days 7
```
- Old Optuna trial logs
- Hyperparameter search results
- **Impact**: Lose tuning history

### 🔴 High Risk (Dangerous - Large Impact)

#### Notebook Data Copies
```bash
python cleanup.py --notebooks-data
```
- Duplicated datasets in notebook directories
- **Impact**: Notebooks may need data re-download
- **Space savings**: ~63GB

#### Downloaded Datasets
```bash
python cleanup.py --data-downloads --confirm
```
- Raw downloaded dataset files
- **Impact**: Must re-download for training
- **Space savings**: ~4.4GB

## 📖 Detailed Usage Examples

### Common Scenarios

#### 1. Weekly Maintenance
```bash
# Safe weekly cleanup
python cleanup.py --cache --logs --jupyter-checkpoints
cd models/production
python scripts/cleanup_production.py --failed-runs --temp-files --tuning-logs --days 7
```

#### 2. Major Space Recovery
```bash
# Preview major cleanup first
python cleanup.py --dry-run --checkpoints --notebooks-data --old-models --days 30

# Execute if satisfied with preview
python cleanup.py --checkpoints --notebooks-data --old-models --days 30 --confirm
```

#### 3. Pre-Training Cleanup
```bash
# Clean before starting new training to ensure space
python cleanup.py --checkpoints --old-models --days 14
cd models/production
python scripts/cleanup_production.py --old-versions --keep 3
```

#### 4. Complete Reset (Nuclear Option)
```bash
# DANGER: This removes almost everything except source code
python cleanup.py --all --confirm
cd models/production
python scripts/cleanup_production.py --all --keep 1
```

### Advanced Options

#### Selective Model Cleanup
```bash
# Keep only 3 newest versions of each model
cd models/production
python scripts/cleanup_production.py --old-versions --keep 3

# Clean models older than 14 days
python cleanup.py --old-models --days 14
```

#### Custom Age Thresholds
```bash
# Clean logs older than 3 days
python cleanup.py --logs

# Clean tuning logs older than 14 days
cd models/production
python scripts/cleanup_production.py --tuning-logs --days 14
```

#### Dry Run Everything
```bash
# See what would be deleted without doing it
python cleanup.py --dry-run --all
cd models/production
python scripts/cleanup_production.py --dry-run --all
```

## 🔧 Script Options Reference

### Main Cleanup Script (`cleanup.py`)

```bash
python cleanup.py [OPTIONS]

# Analysis
--analyze              # Show disk usage analysis

# Safety
--dry-run             # Preview without deleting

# Categories
--logs                # Clean log files
--checkpoints         # Clean training checkpoints  
--temp-models         # Clean temporary model files
--cache              # Clean Python cache files
--jupyter-checkpoints # Clean Jupyter checkpoints
--old-models         # Clean old model files
--notebooks-data     # Clean notebook data copies (DANGEROUS)
--data-downloads     # Clean downloaded datasets (DANGEROUS)

# Comprehensive
--all                # Clean all categories

# Options
--days N             # Age threshold for old files (default: 30)
--confirm            # Skip confirmation for dangerous operations
--project-root PATH  # Project root directory
```

### Production Cleanup Script

```bash
cd models/production
python scripts/cleanup_production.py [OPTIONS]

# Analysis
--analyze            # Show production cleanup opportunities

# Safety  
--dry-run           # Preview without deleting

# Categories
--failed-runs       # Clean failed training runs
--old-versions      # Clean old model versions
--tuning-logs       # Clean hyperparameter tuning logs
--temp-files        # Clean temporary files

# Comprehensive
--all              # Clean all categories

# Options
--keep N           # Number of versions to keep (default: 5)
--days N           # Age threshold for logs (default: 7)
```

## ⚠️ Important Warnings

### Before Cleaning

1. **Backup Important Models**: Ensure production models are backed up
2. **Check Running Training**: Don't clean while training is in progress
3. **Preview First**: Always use `--dry-run` for major cleanup operations
4. **Understand Impact**: Read what each category does before cleaning

### Data Recovery

If you accidentally delete important files:

1. **Check Recycle Bin/Trash** (if available)
2. **Use file recovery tools** (ext4/NTFS undelete utilities)
3. **Re-run training** for models (time-consuming but recovers everything)
4. **Re-download datasets** for data files

### Preventing Issues

1. **Regular Maintenance**: Run safe cleanup weekly
2. **Monitor Space**: Check disk usage before training
3. **Separate Important Models**: Move production models to safe location
4. **Use External Storage**: Store datasets on external drives when possible

## 📅 Recommended Cleanup Schedule

### Daily (Automated)
```bash
# Add to cron/scheduled task
python cleanup.py --cache --temp-models
```

### Weekly  
```bash
python cleanup.py --logs --jupyter-checkpoints
cd models/production
python scripts/cleanup_production.py --failed-runs --temp-files
```

### Monthly
```bash
python cleanup.py --old-models --days 30
cd models/production  
python scripts/cleanup_production.py --old-versions --keep 5 --tuning-logs --days 30
```

### As Needed (When Space Low)
```bash
python cleanup.py --checkpoints --notebooks-data --confirm
```

## 🔍 Monitoring Disk Usage

### Check Current Usage
```bash
# Project total
du -sh /path/to/eyeVsAI

# By directory
du -sh /path/to/eyeVsAI/models/*

# Largest files
find /path/to/eyeVsAI -type f -exec du -h {} \; | sort -hr | head -20
```

### Set Up Alerts
Consider setting up disk usage alerts:
- Alert when project > 50GB
- Alert when disk < 10GB free
- Weekly usage reports

## 🚨 Emergency Space Recovery

If you're completely out of space:

1. **Immediate relief** (safest):
   ```bash
   python cleanup.py --cache --logs --temp-models
   ```

2. **Moderate relief**:
   ```bash
   python cleanup.py --checkpoints
   ```

3. **Major relief** (data loss):
   ```bash
   python cleanup.py --notebooks-data --confirm
   ```

4. **Nuclear option** (rebuild everything):
   ```bash
   python cleanup.py --all --confirm
   cd models/production
   python scripts/cleanup_production.py --all --keep 1
   ```

Remember: It's always better to prevent space issues with regular maintenance than to need emergency cleanup!