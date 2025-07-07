# CLI Training Scripts Usage Guide

This document provides comprehensive instructions for using the command-line interface (CLI) training scripts for all image classification models.

## Overview

The project includes both individual CLI scripts for each model and a unified CLI launcher that provides a single entry point for training any model.

### Available Models

1. **Shallow Learning Classifier** - Traditional machine learning with feature extraction
2. **Deep Learning v1 Classifier** - Basic CNN with standard layers
3. **Deep Learning v2 Classifier** - Advanced CNN with ResNet + Attention mechanisms
4. **Transfer Learning Classifier** - Pre-trained models with fine-tuning

## Unified CLI Launcher

The recommended way to train models is using the unified CLI launcher:

```bash
python train_models.py [command] [options]
```

### Quick Start

1. **List available models:**
```bash
python train_models.py list
```

2. **Install dependencies for a specific model:**
```bash
python train_models.py install shallow
python train_models.py install deep-v2
```

3. **Train a model:**
```bash
python train_models.py shallow --data_path /path/to/data --epochs 100
python train_models.py deep-v2 --data_path /path/to/data --epochs 25 --batch_size 8
python train_models.py transfer --data_path /path/to/data --base_model resnet50
```

4. **Get help for specific models:**
```bash
python train_models.py shallow --help
python train_models.py deep-v2 --help
```

## Individual CLI Scripts

Each model also has its own standalone CLI script:

```bash
# From the model's directory
cd models/classifiers/shallow && python scripts/train.py --help
cd models/classifiers/deep-v1 && python scripts/train.py --help
cd models/classifiers/deep-v2 && python scripts/train.py --help
cd models/classifiers/transfer && python scripts/train.py --help
```

## Model-Specific Usage

### Shallow Learning Classifier

**Dependencies:** scikit-learn, opencv-python, numpy, matplotlib

**Example usage:**
```bash
python train_models.py shallow \
    --data_path /path/to/dataset \
    --model_save_path models/shallow_classifier.pkl \
    --feature_types hog lbp color_histogram \
    --max_iter 1000 \
    --test_size 0.2 \
    --log_dir logs/shallow/
```

**Key parameters:**
- `--feature_types`: Feature extraction methods (hog, lbp, color_histogram, texture)
- `--max_iter`: Maximum iterations for the classifier
- `--test_size`: Test set split ratio

### Deep Learning v1 Classifier

**Dependencies:** torch, torchvision, numpy, matplotlib, Pillow

**Example usage:**
```bash
python train_models.py deep-v1 \
    --data_path /path/to/dataset \
    --model_save_path models/deep_v1_classifier.pth \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_epochs 30 \
    --patience 10 \
    --image_size 64 64 \
    --log_dir logs/deep_v1/
```

**Key parameters:**
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for optimizer
- `--num_epochs`: Number of training epochs
- `--patience`: Early stopping patience
- `--image_size`: Input image dimensions

### Deep Learning v2 Classifier

**Dependencies:** torch, torchvision, numpy, matplotlib, Pillow, torchinfo

**Example usage:**
```bash
python train_models.py deep-v2 \
    --data_path /path/to/dataset \
    --model_save_path models/deep_v2_classifier.pth \
    --batch_size 8 \
    --accumulation_steps 4 \
    --learning_rate 0.0005 \
    --num_epochs 25 \
    --patience 8 \
    --image_size 96 96 \
    --mixup_alpha 0.2 \
    --mixup_prob 0.3 \
    --label_smoothing 0.05 \
    --attention_reduction 8 \
    --memory_efficient \
    --log_dir logs/deep_v2/
```

**Key parameters:**
- `--accumulation_steps`: Gradient accumulation steps for effective larger batches
- `--mixup_alpha`: Mixup augmentation alpha parameter
- `--mixup_prob`: Probability of applying mixup
- `--label_smoothing`: Label smoothing factor
- `--attention_reduction`: Channel attention reduction ratio
- `--memory_efficient`: Enable memory efficient training

### Transfer Learning Classifier

**Dependencies:** tensorflow, numpy, matplotlib, Pillow

**Example usage:**
```bash
python train_models.py transfer \
    --data_path /path/to/dataset \
    --model_save_path models/transfer_learning_classifier.h5 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_epochs 20 \
    --patience 5 \
    --image_size 224 224 \
    --base_model resnet50 \
    --fine_tune_layers 10 \
    --fine_tune_lr 1e-5 \
    --validation_split 0.2 \
    --dense_units 512 256 \
    --mixed_precision \
    --use_xla \
    --class_weights \
    --log_dir logs/transfer/ \
    --cache_dataset
```

**Key parameters:**
- `--base_model`: Pre-trained backbone (resnet50, vgg16, efficientnet_b0, etc.)
- `--fine_tune_layers`: Number of layers to fine-tune
- `--fine_tune_lr`: Learning rate for fine-tuning phase
- `--dense_units`: Hidden layer sizes for classifier head
- `--mixed_precision`: Enable mixed precision training
- `--use_xla`: Enable XLA compilation
- `--class_weights`: Use class weights for imbalanced datasets

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

## Output Files

Each model generates the following outputs:

1. **Trained model file** (format varies by model type)
2. **Training logs** in the specified log directory
3. **Training history** (JSON format with metrics)
4. **Configuration file** with training parameters
5. **Model metadata** for the model registry

## Environment Setup

### Option 1: Install all dependencies at once
```bash
pip install torch torchvision tensorflow scikit-learn opencv-python numpy matplotlib Pillow torchinfo
```

### Option 2: Install per model as needed
```bash
python train_models.py install shallow
python train_models.py install deep-v2
python train_models.py install transfer
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the correct directory and dependencies are installed
2. **Memory errors**: Reduce batch size or enable memory efficient training
3. **CUDA errors**: Check GPU availability and memory

### Memory Optimization

For memory-constrained environments:

```bash
# Deep Learning models
--batch_size 4 --accumulation_steps 8  # Effective batch size 32
--memory_efficient                     # Enable memory optimizations

# Transfer Learning
--batch_size 16 --cache_dataset        # Cache dataset for faster access
```

### Performance Tuning

For faster training:

```bash
# Transfer Learning
--mixed_precision --use_xla            # Enable performance optimizations
--cache_dataset                        # Cache dataset in memory

# Deep Learning v2  
--memory_efficient                     # Memory optimizations don't sacrifice speed
```

## Integration with Model Registry

All trained models are automatically registered in the model registry with:
- Model metadata and configuration
- Training metrics and performance
- Model file paths and versioning
- Framework and architecture information

Access registered models programmatically:
```python
from models.core.src.model_registry import ModelRegistry

registry = ModelRegistry()
models = registry.list_models()
model_info = registry.get_model("deep-learning-v2")
```

## Advanced Usage

### Ensemble Training
Train multiple models for ensemble prediction:
```bash
python train_models.py shallow --data_path /data --model_save_path models/ensemble_shallow.pkl
python train_models.py deep-v2 --data_path /data --model_save_path models/ensemble_deep.pth
python train_models.py transfer --data_path /data --model_save_path models/ensemble_transfer.h5
```

### Cross-Validation
Each model supports different validation strategies through configuration.

### Production Deployment
All models implement the `BaseImageClassifier` interface for consistent deployment:

```python
from models.classifiers.deep_v2.src.classifier import DeepLearningV2Classifier

classifier = DeepLearningV2Classifier()
classifier.load_model("models/production/models/deep_v2_classifier.pth")
predictions = classifier.predict(image_array)
```

## Support

For issues with specific models, refer to their individual documentation in the `src/` directories. The unified CLI provides dependency checking and installation to help resolve common setup issues.