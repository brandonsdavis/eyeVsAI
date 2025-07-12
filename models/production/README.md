# Production Training Pipeline

This directory contains the enterprise-grade training pipeline for all machine learning models in the Eye vs AI system. The production pipeline automates model training, hyperparameter optimization, and deployment preparation across multiple datasets and model architectures.

## Overview

The production training system provides comprehensive automation for machine learning workflows including:

- **Automated Training**: Parallel training of all model/dataset combinations
- **Hyperparameter Optimization**: Bayesian optimization using Optuna
- **Model Registry**: Centralized tracking of all trained models with metadata
- **Report Generation**: Comprehensive performance analysis and visualizations
- **Export Management**: Multi-format model exports for deployment
- **Resource Management**: GPU memory optimization and cleanup utilities

The pipeline trains **84 different model variations** across **5 datasets**, producing comprehensive model coverage:

### Model Types & Variations
- **Shallow Learning**: 2 variations (SVM, Random Forest)
- **Deep Learning v1**: 2 variations (Standard, Residual)  
- **Deep Learning v2**: 3 variations (ResNet, DenseNet, Hybrid)
- **Transfer Learning**: 7 variations (ResNet50/101, EfficientNet-B0/B1, VGG16, MobileNetV2, DenseNet121)

### Datasets
- **Combined**: 67 classes (pets + vegetables + instruments + street foods)
- **Vegetables**: 15 classes
- **Pets**: 37 classes  
- **Street Foods**: 10 classes
- **Instruments**: 30 classes

## Directory Structure

```
production/
├── configs/                    # Configuration files
│   ├── models.json            # Model type definitions and variations
│   └── datasets.json          # Dataset configurations and metadata
├── scripts/                   # Training automation scripts
│   ├── train_all_production_models.py    # Master orchestration script
│   ├── production_trainer.py             # Individual model training
│   ├── hyperparameter_tuner.py           # Automated hyperparameter optimization
│   ├── production_export.py              # Model export utilities
│   └── cleanup_production.py             # Resource management and cleanup
├── models/                    # Trained model storage
│   └── {model_type}/{variation}/{dataset}/{version}/
│       ├── model.pth          # Trained model weights
│       ├── config.json        # Training configuration
│       ├── training_results.json         # Performance metrics
│       └── training.log       # Detailed training logs
├── logs/                      # Pipeline execution logs
├── results/                   # Training reports and visualizations
│   ├── all_models_report_YYYYMMDD.csv    # Comprehensive model comparison
│   ├── summary_report_YYYYMMDD.json      # Executive summary
│   ├── game_backend_report_YYYYMMDD.json # Game-ready model registry
│   └── *.png                  # Performance visualization plots
└── training_registry.json     # Central model registry with all metadata
```

## Master Training Script

The `train_all_production_models.py` script orchestrates the complete training pipeline with comprehensive progress tracking and parallel execution support.

### Quick Start

```bash
# Train all models with default settings
python scripts/train_all_production_models.py --parallel_jobs 2

# Train with hyperparameter optimization
python scripts/train_all_production_models.py --run_tuning --tuning_trials 15 --parallel_jobs 2

# Train specific models only
python scripts/train_all_production_models.py --models deep_v1 deep_v2 --datasets pets instruments

# Generate reports from existing results
python scripts/train_all_production_models.py --generate_reports_only
```

## Pipeline Features

### Automated Hyperparameter Tuning
- **Optuna-based optimization** with configurable trials
- **Pruning** for efficiency (median pruner)
- **Parallel execution** support
- **Model-specific search spaces** defined in configs

### Production-Ready Export
- **PyTorch .pth models** for inference
- **ONNX export** (available separately - see ONNX Export section)
- **Comprehensive metadata** and versioning
- **Model performance metrics**

### Comprehensive Reporting & Model Registry
- **Centralized training registry** with complete model metadata
- **Session-based tracking** for batch training operations
- **Automatic environment logging** (GPU, CUDA, PyTorch versions)
- **Export status management** and deployment tracking
- **JSON-formatted metadata** for all models and training runs
- **Best model identification** by dataset and type

## Directory Structure

```
models/production/
├── configs/
│   ├── datasets.json          # Dataset registry
│   └── models.json            # Model configurations
├── scripts/
│   ├── hyperparameter_tuner.py    # Optuna-based tuning
│   ├── production_trainer.py      # Production model training  
│   ├── train_all_production_models.py  # Master orchestration
│   ├── training_registry.py       # Centralized model tracking
│   ├── production_export.py       # ONNX export and deployment
│   └── demo_enhanced_system.py    # System demonstration
├── models/                    # Trained production models
│   ├── shallow/
│   ├── deep_v1/
│   ├── deep_v2/
│   └── transfer/
├── logs/                      # Training logs and tuning results
├── results/                   # Reports and visualizations
├── exports/                   # ONNX models and deployment packages
├── training_registry.json    # Centralized model metadata registry
└── README.md
```

## Configuration

### Adding New Datasets
Edit `configs/datasets.json`:
```json
{
  "datasets": {
    "new_dataset": {
      "name": "New Dataset",
      "path": "/path/to/dataset",
      "description": "Description",
      "num_classes": 10,
      "domains": ["domain1"],
      "image_size": [128, 128],
      "validation_split": 0.2,
      "test_split": 0.1
    }
  }
}
```

### Adding New Model Variations
Edit `configs/models.json`:
```json
{
  "model_types": {
    "transfer": {
      "variations": {
        "new_model": {
          "name": "New Transfer Model",
          "config": {
            "base_model": "new_architecture",
            "freeze_base_epochs": 10
          }
        }
      }
    }
  }
}
```

## Usage Examples

### Basic Training
```bash
# Train all models (no tuning)
python scripts/train_all_production_models.py

# With parallel execution (faster)
python scripts/train_all_production_models.py --parallel_jobs 4
```

### Hyperparameter Tuning
```bash
# Light tuning (10 trials per model)
python scripts/train_all_production_models.py --run_tuning --tuning_trials 10

# Intensive tuning (50 trials per model)
python scripts/train_all_production_models.py --run_tuning --tuning_trials 50
```

### Selective Training
```bash
# Only deep learning models
python scripts/train_all_production_models.py --models deep_v1 deep_v2

# Only transfer learning on combined dataset
python scripts/train_all_production_models.py --models transfer --datasets combined

# Specific model/dataset combination
python scripts/production_trainer.py \
    --model_type transfer \
    --variation efficientnet_b0 \
    --dataset vegetables
```

### Individual Hyperparameter Tuning
```bash
python scripts/hyperparameter_tuner.py \
    --model_type deep_v2 \
    --model_variation resnet \
    --dataset combined \
    --n_trials 30
```

## Model Performance Tracking

Each trained model includes:
- **Training metrics** (accuracy, loss curves)
- **Validation performance** 
- **Hyperparameter configuration**
- **Model versioning** with timestamps
- **Export formats** (.pth, .onnx)

Results are saved in:
- `models/{type}/{variation}/{dataset}/{version}/`
- Comprehensive CSV reports in `results/`
- Visualization plots in `results/`

## Expected Training Times

Approximate times per model (on GPU):
- **Shallow Learning**: 2-5 minutes
- **Deep Learning V1**: 10-30 minutes  
- **Deep Learning V2**: 20-60 minutes
- **Transfer Learning**: 15-45 minutes

**Total pipeline time**: 50-200 hours (depending on tuning and parallelization)

## Monitoring Progress

Check training progress:
```bash
# View overall pipeline log
tail -f logs/pipeline_*.log

# View specific model training
tail -f logs/training_improved_*.log

# Check tuning progress
tail -f logs/tuning_*.log
```

## Best Practices

1. **Start with subset**: Test on 1-2 models first
2. **Use parallel jobs**: Speed up with `--parallel_jobs 4`
3. **Monitor GPU memory**: Some models require significant VRAM
4. **Regular checkpointing**: Models save every 5 epochs
5. **Early stopping**: Prevents overfitting with patience

## Reproducibility

The pipeline ensures reproducibility through:
- **Fixed random seeds** in configurations
- **Version tracking** for all models
- **Configuration snapshots** saved with each model
- **Complete parameter logging**

## Production Deployment

Trained models are ready for deployment:
- **ONNX format** for cross-platform inference
- **Metadata files** with class mappings
- **Performance benchmarks** for model selection
- **Standardized input preprocessing**

## Enhanced Model Registry & Export System

### Training Registry Features
The system now includes a comprehensive training registry that tracks:
- **Complete model metadata**: type, variation, dataset, version, hyperparameters
- **Performance metrics**: accuracy, training time, epoch details
- **File organization**: automatic discovery of model artifacts
- **Training environment**: GPU info, CUDA version, PyTorch version
- **Export status**: pending/completed/failed tracking
- **Session management**: batch training progress and statistics

### View Training Registry
```bash
# View all registered models
python scripts/training_registry.py --action summary

# View models ready for export
python scripts/training_registry.py --action export_pending

# View best performing models
python scripts/training_registry.py --action best_models

# Generate comprehensive report
python scripts/training_registry.py --action report --output registry_report.json
```

### Production Export System

#### Automated ONNX Export
```bash
# Export all pending models to ONNX
python scripts/production_export.py

# Export specific model type
python scripts/production_export.py --model_type transfer

# Force re-export of already exported models
python scripts/production_export.py --force

# Export specific dataset models
python scripts/production_export.py --dataset vegetables
```

#### Create Deployment Packages
```bash
# Create deployment package with best models
python scripts/production_export.py --package production_v1

# Custom package name
python scripts/production_export.py --package my_deployment
```

### Registry File Locations
- **Main Registry**: `training_registry.json` - Complete model tracking
- **Model Metadata**: `models/{type}/{variation}/{dataset}/{version}/training_results.json`
- **Export Directory**: `exports/` - ONNX models and deployment packages

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch sizes in configs
2. **Dataset not found**: Check paths in `configs/datasets.json`
3. **Import errors**: Run `python setup.py` to install dependencies
4. **Long training times**: Use fewer epochs for testing
5. **Registry not updating**: Check file permissions on training_registry.json
6. **Export failures**: Check model files exist and are accessible
7. **Missing model files**: Ensure training completed successfully

### Log Analysis
- Training logs: `logs/training_*.log`
- Tuning results: `logs/tuning_results_*.json`
- Model metadata: `models/{type}/{variation}/{dataset}/{version}/training_results.json`
- Registry data: `training_registry.json`
- Export logs: `exports/export_*.log`
- Error debugging: Check stdout/stderr in log files

## Contributing

To add new model types or datasets:
1. Update configuration files
2. Add training logic if needed
3. Test with single model first
4. Update documentation

This pipeline is designed to be **fully automated** and **easily extensible** for new datasets and model architectures.
