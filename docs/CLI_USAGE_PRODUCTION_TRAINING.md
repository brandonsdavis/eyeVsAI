# CLI Training Scripts Usage Guide

This document provides comprehensive instructions for using the production-ready command-line interface (CLI) training scripts for all image classification models.

## Overview

The project includes both individual model training scripts and a comprehensive **Production Training Pipeline** that provides automated hyperparameter tuning, model registry, and deployment preparation.

### Available Models

1. **Shallow Learning Classifier** - Traditional ML with HOG/LBP feature extraction (SVM, Random Forest)
2. **Deep Learning v1 Classifier** - Basic CNN with 3×3 kernels and optional residual connections
3. **Deep Learning v2 Classifier** - Advanced CNN with ResNet/DenseNet + CBAM attention mechanisms
4. **Transfer Learning Classifier** - Pre-trained models (ResNet, EfficientNet, VGG, MobileNet, DenseNet)

## Production Training Pipeline (Recommended)

The **production training system** provides automated training across all model types and datasets with comprehensive logging and export capabilities.

### Quick Start

1. **Train all models with hyperparameter tuning:**
```bash
cd production_training
python scripts/train_all_production_models.py --run_tuning --tuning_trials 20
```

2. **Train specific models without tuning:**
```bash
python scripts/train_all_production_models.py --models deep_v1 deep_v2
```

3. **Train on specific datasets:**
```bash
python scripts/train_all_production_models.py --datasets combined vegetables --run_tuning
```

4. **Train a single model variation:**
```bash
python scripts/production_trainer.py \
    --model_type transfer \
    --variation resnet50 \
    --dataset combined
```

### Available Datasets

Configure datasets in `configs/datasets.json`:
- **combined**: ~67 classes (pets + vegetables + instruments + street foods)
- **vegetables**: ~15 classes from Kaggle vegetables dataset
- **pets**: ~37 classes from Oxford-IIIT Pet Dataset
- **street_foods**: ~10 classes from street foods dataset
- **instruments**: ~30 classes from musical instruments dataset

### Model Variations

Configure in `configs/models.json`:

**Shallow Learning (2 variations):**
- `svm_hog_lbp`: SVM with HOG+LBP features
- `rf_hog_lbp`: Random Forest with HOG+LBP features

**Deep Learning V1 (2 variations):**
- `standard`: Basic CNN with 3×3 kernels
- `residual`: CNN with residual connections

**Deep Learning V2 (3 variations):**
- `resnet`: ResNet-based architecture with CBAM attention
- `densenet`: DenseNet-based architecture 
- `hybrid`: Hybrid ResNet+DenseNet architecture

**Transfer Learning (7 variations):**
- `resnet50`, `resnet101`: ResNet-based transfer learning
- `efficientnet_b0`, `efficientnet_b1`: EfficientNet-based
- `vgg16`: VGG-based transfer learning
- `mobilenet_v2`: MobileNet-based for efficiency
- `densenet121`: DenseNet-based transfer learning

## Hyperparameter Tuning

### Individual Model Tuning
```bash
python scripts/hyperparameter_tuner.py \
    --model_type deep_v2 \
    --model_variation resnet \
    --dataset combined \
    --n_trials 50
```

### Tuning Parameters

The system automatically tunes:

**Shallow Learning:**
- SVM: C, kernel, PCA components
- Random Forest: n_estimators, max_depth

**Deep Learning:**
- Learning rate, batch size, dropout rates
- Optimizer choice (Adam, AdamW, SGD)
- Weight decay, architecture options

**Transfer Learning:**
- Learning rate, batch size, head architecture
- Dense layer configurations, dropout rates
- Fine-tuning strategies

## Training Registry & Metadata

All trained models are automatically tracked in `training_registry.json` with:

### Comprehensive Metadata
- Model identification (type, variation, dataset, version)
- Performance metrics (accuracy, training time, epochs)
- Complete hyperparameter configuration
- File paths for all model artifacts
- Training environment details (GPU, CUDA, PyTorch versions)
- Export status and production readiness flags

### Registry Commands
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

## Production Export & Deployment

### ONNX Export
```bash
# Export all pending models to ONNX
python scripts/production_export.py

# Export specific model type
python scripts/production_export.py --model_type transfer

# Force re-export of already exported models
python scripts/production_export.py --force
```

### Create Deployment Packages
```bash
# Create deployment package with best models
python scripts/production_export.py --package production_v1

# Custom package name  
python scripts/production_export.py --package my_deployment
```

### Deployment Package Contents
- ONNX model files for cross-platform inference
- Complete metadata with class mappings and preprocessing info
- Performance benchmarks for model selection
- README with usage instructions

## Individual Model Training

### Shallow Learning Classifier
```bash
cd image-classifier-shallow
python scripts/train.py \
    --data_path /path/to/dataset \
    --classifier_type svm \
    --svm_C 1.0 \
    --svm_kernel rbf \
    --pca_components 100 \
    --model_save_path models/shallow.pkl
```

### Deep Learning V1 (Improved)
```bash
cd image-classifier-deep-v1  
python scripts/train_improved.py \
    --data_path /path/to/dataset \
    --batch_size 64 \
    --learning_rate 0.01 \
    --num_epochs 50 \
    --optimizer adamw \
    --use_residual
```

### Deep Learning V2 (Improved)
```bash
cd image-classifier-deep-v2
python scripts/train_improved.py \
    --data_path /path/to/dataset \
    --batch_size 64 \
    --learning_rate 0.1 \
    --num_epochs 100 \
    --architecture resnet \
    --optimizer sgd \
    --mixup_alpha 0.3
```

### Transfer Learning (PyTorch)
```bash
cd image-classifier-transfer
python scripts/train_pytorch.py \
    --data_path /path/to/dataset \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 40 \
    --base_model resnet50 \
    --optimizer adamw
```

## Data Format Requirements

All models expect data in the following directory structure:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── ...
```

**Supported image formats:** .jpg, .jpeg, .png, .bmp, .tiff
**Recommended image size:** 128×128 for consistency across models

## Output Files & Organization

### Production Training Outputs
```
production_training/
├── models/                    # Trained production models
│   ├── {type}/{variation}/{dataset}/{version}/
│   │   ├── model.pth         # PyTorch model file
│   │   ├── config.json       # Training configuration
│   │   ├── training_results.json  # Complete metadata
│   │   └── training.log      # Training logs
├── logs/                      # Training and tuning logs
├── exports/                   # ONNX models and deployment packages
└── training_registry.json    # Centralized model registry
```

### Individual Model Outputs
1. **Model files** (.pth, .pkl, .h5 depending on framework)
2. **Training logs** with detailed progress and metrics
3. **Configuration files** with all parameters
4. **Model metadata** for registry integration

## Environment Setup

### Production Training Environment
```bash
# Install core dependencies
pip install torch torchvision optuna scikit-learn opencv-python numpy matplotlib Pillow

# Additional dependencies for specific models
pip install tensorflow  # For TensorFlow-based transfer learning (if used)
```

### GPU Requirements
- **Recommended:** NVIDIA GPU with 8GB+ VRAM
- **Minimum:** 4GB VRAM (use smaller batch sizes)
- **CPU-only:** Supported but significantly slower

## Performance Optimization

### Memory Optimization
```bash
# Reduce batch sizes for memory-constrained environments
--batch_size 16  # For Deep Learning models
--batch_size 8   # For very limited VRAM

# Use gradient accumulation for effective larger batches
python scripts/train_improved.py --batch_size 16 --accumulation_steps 4  # Effective batch size 64
```

### Training Speed
```bash
# Use mixed precision for faster training
--use_amp  # Available in Deep Learning V1/V2

# Parallel hyperparameter tuning
python scripts/train_all_production_models.py --parallel_jobs 4
```

## Monitoring & Troubleshooting

### Monitor Training Progress
```bash
# View overall pipeline log
tail -f logs/pipeline_*.log

# View specific model training
tail -f logs/training_*.log

# Check tuning progress  
tail -f logs/tuning_*.log
```

### Common Issues & Solutions

1. **Import errors**: Ensure you're running from project root
2. **Memory errors**: Reduce batch sizes or use gradient accumulation
3. **CUDA errors**: Check GPU availability and PyTorch CUDA installation
4. **0.0% accuracy**: Verify dataset paths and format
5. **Training hangs**: Check dataset accessibility and permissions

### Error Resolution
- **Import issues**: Fixed by running from project root with correct PYTHONPATH
- **Missing dependencies**: Use individual model requirements.txt files
- **Dataset errors**: Verify directory structure and image formats

## Integration with Model Registry

### Programmatic Access
```python
from training_registry import TrainingRegistry

# Load registry
registry = TrainingRegistry("models/production/training_registry.json")

# Get training summary
summary = registry.get_training_summary()

# Get best models
best_models = registry.get_best_models(by_dataset=True)

# Get models ready for export
export_models = registry.get_models_for_export("pending")
```

### Model Deployment
```python
# Load a trained model for inference
import torch
from models.classifiers.deep_v1.src.model_improved import DeepLearningV1Improved

model = DeepLearningV1Improved(num_classes=67)
checkpoint = torch.load("models/production/models/deep_v1/standard/combined/v20250705_abc123/model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
predictions = model(input_tensor)
```

## Advanced Features

### Session Management
All training runs are organized into sessions for tracking:
```python
# Start a training session
session_id = registry.start_training_session("experiment_1")

# All models trained during this session are tracked together
# View session summary
session_summary = registry.get_training_summary(session_id)
```

### Reproducibility
- **Fixed random seeds** in configurations
- **Version tracking** for all models  
- **Configuration snapshots** saved with each model
- **Complete parameter logging** in JSON format

### Production Deployment
Trained models are deployment-ready with:
- **ONNX format** for cross-platform inference
- **Metadata files** with preprocessing requirements
- **Performance benchmarks** for model selection
- **Standardized input preprocessing** (128×128 RGB images)

## Best Practices

1. **Start small**: Test with 1-2 models before running full pipeline
2. **Use hyperparameter tuning**: Significantly improves model performance
3. **Monitor GPU memory**: Adjust batch sizes based on available VRAM
4. **Regular checkpointing**: Models save automatically every few epochs
5. **Version control**: All models are automatically versioned with timestamps

## Support & Troubleshooting

### Getting Help
- Check individual model documentation in `{model}/README.md`
- Review training logs in `logs/` directories  
- Use `--help` flag with any training script
- Examine configuration files in `production_training/configs/`

### Performance Expectations
**Approximate training times per model (on GPU):**
- Shallow Learning: 2-5 minutes
- Deep Learning V1: 10-30 minutes
- Deep Learning V2: 20-60 minutes  
- Transfer Learning: 15-45 minutes

**Expected accuracies (vegetables dataset, 3 classes):**
- Shallow Learning: 60-75%
- Deep Learning V1: 70-85%
- Deep Learning V2: 75-90%
- Transfer Learning: 80-95%

This production training system is designed to be **fully automated** and **easily extensible** for new datasets and model architectures while providing comprehensive tracking and deployment capabilities.