# Machine Learning Models

This directory contains the complete machine learning pipeline for the Eye vs AI image classification system. The project demonstrates various approaches to image classification, from traditional machine learning to state-of-the-art deep learning architectures.

## Overview

The models component provides a comprehensive ML training and deployment system featuring:
- Four distinct model types with different architectural approaches
- Unified command-line interface for all models
- Production-grade training pipeline with hyperparameter optimization
- Automated model evaluation and selection
- Export capabilities for cross-platform deployment

## Model Types

### 1. Shallow Learning Models

**Technical Description**: Traditional machine learning approach using handcrafted feature extraction combined with classical algorithms. Features are extracted using Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and color histograms, then classified using Support Vector Machines (SVM) or Random Forests.

**Simplified Explanation**: These models work by converting images into numerical descriptions (like counting edges and textures) and then using traditional statistical methods to classify them. Think of it as describing an image with numbers and then finding patterns in those numbers.

**Key Papers**:
- Dalal & Triggs (2005): "Histograms of Oriented Gradients for Human Detection"
- Ojala et al. (2002): "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns"

### 2. Deep Learning v1

**Technical Description**: Basic convolutional neural network architecture implementing standard CNN layers with batch normalization and dropout regularization. Features a straightforward architecture with convolutional blocks followed by fully connected layers. Includes a residual variant with skip connections for improved gradient flow.

**Simplified Explanation**: This model learns to recognize images by building up understanding layer by layer - first detecting edges, then shapes, then objects. It's like teaching a computer to see by showing it thousands of examples until it learns the patterns.

**Architecture Details**:
- 4 convolutional blocks with increasing filter sizes (32, 64, 128, 256)
- MaxPooling after each block
- Batch normalization and dropout for regularization
- Global average pooling to reduce parameters
- Fully connected classifier head

**Key Papers**:
- LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
- Simonyan & Zisserman (2014): "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- Krizhevsky, Sutskever & Hinton (2012): "ImageNet Classification with Deep Convolutional Neural Networks"

### 3. Deep Learning v2

**Technical Description**: Advanced CNN architectures incorporating modern design patterns including attention mechanisms, depthwise separable convolutions, and sophisticated regularization techniques. Implements three variants:
- **ResNet-style**: Deep residual networks with identity mappings
- **DenseNet-style**: Dense connectivity patterns for feature reuse
- **Hybrid**: Combines residual and dense connections with attention

**Simplified Explanation**: These are more sophisticated versions of deep learning that use advanced techniques to focus on important parts of images (attention) and reuse learned features efficiently. They're like having multiple experts collaborate to understand an image.

**Key Papers**:
- He et al. (2016): "Deep Residual Learning for Image Recognition"
- Huang et al. (2017): "Densely Connected Convolutional Networks"
- Vaswani et al. (2017): "Attention Is All You Need"

### 4. Transfer Learning

**Technical Description**: Leverages pre-trained models from ImageNet including VGG16, ResNet50/101, DenseNet121, MobileNetV2, and EfficientNet variants. Uses these as feature extractors with custom classification heads fine-tuned for specific datasets. Implements progressive unfreezing for optimal adaptation.

**Simplified Explanation**: Instead of learning from scratch, these models start with knowledge from millions of images and adapt it to our specific task. It's like hiring an expert photographer who just needs to learn about your specific subjects rather than learning photography from scratch.

**Available Base Models**:
- VGG16: Simple, effective architecture
- ResNet50/101: Excellent for general tasks
- DenseNet121: Efficient feature reuse
- MobileNetV2: Optimized for mobile deployment
- EfficientNet B0/B1: Best accuracy/efficiency trade-off

**Key Papers**:
- Mingsheng et al. (2016): "Deep Learning with Joint Adaptation Networks"

## Directory Structure

```
models/
├── classifiers/           # Model implementations
│   ├── shallow/          # Traditional ML models
│   ├── deep-v1/          # Basic CNN
│   ├── deep-v2/          # Advanced CNN architectures
│   └── transfer/         # Transfer learning models
├── core/                 # Base classes and utilities
├── production/           # Production training system
│   ├── configs/          # Model and dataset configurations
│   ├── models/           # Trained model storage
│   ├── results/          # Training reports and visualizations
│   └── scripts/          # Training and management scripts
├── data/                 # Dataset storage
└── train_models.py       # Unified CLI interface
```

## Production Training System

The production system (`models/production/`) provides enterprise-grade model training and management:

### Key Scripts

#### 1. train_all_production_models.py
Master orchestration script for training all model combinations.

```bash
# Basic usage
python scripts/train_all_production_models.py --parallel_jobs 2

# With hyperparameter tuning
python scripts/train_all_production_models.py --run_tuning --tuning_trials 15 --parallel_jobs 2

# Train specific models
python scripts/train_all_production_models.py --models deep_v1 deep_v2 --datasets pets instruments

# With automatic cleanup
python scripts/train_all_production_models.py --auto-cleanup --cleanup-old-versions 3
```

**Options**:
- `--parallel_jobs N`: Number of concurrent training jobs (default: 1)
- `--run_tuning`: Enable hyperparameter optimization
- `--tuning_trials N`: Number of tuning trials per model (default: 10)
- `--models [list]`: Specific models to train (shallow, deep_v1, deep_v2, transfer)
- `--datasets [list]`: Specific datasets (combined, pets, vegetables, street_foods, instruments)
- `--auto-cleanup`: Clean failed runs after training
- `--generate_reports_only`: Generate reports without training

#### 2. hyperparameter_tuner.py
Automated hyperparameter optimization using Optuna.

```bash
python scripts/hyperparameter_tuner.py \
    --model_type deep_v2 \
    --model_variation resnet \
    --dataset pets \
    --n_trials 30
```

**Features**:
- Bayesian optimization for efficient search
- Model-specific parameter ranges
- Automatic trial pruning for failed runs
- Results saved for production training

#### 3. production_trainer.py
Trains individual models with specified hyperparameters.

```bash
python scripts/production_trainer.py \
    --model_type transfer \
    --variation resnet50 \
    --dataset combined
```

**Features**:
- Loads optimal hyperparameters from tuning
- Comprehensive logging and error handling
- Automatic model versioning
- Metadata tracking for reproducibility

#### 4. production_export.py
Exports trained models to production formats.

```bash
python scripts/production_export.py \
    --registry_path training_registry.json \
    --export_format onnx
```

**Export Formats**:
- PyTorch: Native format (.pth)
- TorchScript: Optimized for deployment (.pt)
- ONNX: Cross-platform inference (.onnx)
- TensorFlow: For TF Serving (transfer models)

#### 5. cleanup_production.py
Manages model versions and cleans training artifacts.

```bash
# Remove failed training runs
python scripts/cleanup_production.py --failed-runs

# Keep only 3 newest versions per model
python scripts/cleanup_production.py --old-versions --keep 3

# Clean logs older than 7 days
python scripts/cleanup_production.py --tuning-logs --days 7
```

### Unified CLI Interface

The `train_models.py` script provides a consistent interface for all model types:

```bash
# List available models
python train_models.py list

# Install dependencies for a model
python train_models.py install deep-v2

# Train a model
python train_models.py shallow --data_path ./data/pets --epochs 100
python train_models.py deep-v1 --data_path ./data/pets --batch_size 32
python train_models.py deep-v2 --data_path ./data/pets --architecture resnet
python train_models.py transfer --data_path ./data/pets --base_model resnet50
```

## Performance Benchmarks

Based on production training results (RTX 5070 Ti, 16GB VRAM):

| Model Type | Best Accuracy | Training Time | Parameters | Inference Speed |
|------------|--------------|---------------|------------|-----------------|
| Shallow (SVM) | 71.2% | 5-10 min | N/A | 1000+ img/s |
| Deep v1 | 84.7% | 30-45 min | ~5M | 200 img/s |
| Deep v2 | 91.3% | 45-60 min | ~25M | 150 img/s |
| Transfer | 94.7% | 20-30 min | ~23M | 180 img/s |

*Accuracy on combined dataset (67 classes)

## Configuration

Model configurations are stored in `production/configs/`:

- `models.json`: Model architectures and variations
- `datasets.json`: Dataset paths and metadata

Training results are tracked in:
- `training_registry.json`: Central model registry
- `results/game_backend_report_*.json`: Game-ready model selection

## Future Production Considerations

While this implementation serves well for demonstration purposes, a production deployment would benefit from:

1. **Distributed Training**: Multi-GPU support using PyTorch DDP or Horovod
2. **Model Versioning**: Integration with MLflow or DVC for experiment tracking
3. **Continuous Training**: Automated retraining pipelines with data drift detection
4. **Model Optimization**: Quantization and pruning for edge deployment
5. **A/B Testing**: Gradual rollout with performance monitoring
6. **Hardware Acceleration**: TensorRT optimization for NVIDIA GPUs
7. **Monitoring**: Real-time inference metrics and model degradation alerts

## Development Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Install base dependencies:
```bash
pip install -r requirements.txt
```

3. Install model-specific dependencies:
```bash
python train_models.py install <model_type>
```

4. Download datasets:
```bash
python setup_data.py
```

## Testing

Run model tests:
```bash
# All tests
pytest

# Specific model tests
python test_models.py --test TestShallowLearning
python test_models.py --test TestDeepLearningV1

# Structure tests (no GPU required)
python test_structure.py
```

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**: Reduce batch size or enable memory_efficient mode
2. **Import Errors**: Ensure running from project root with correct PYTHONPATH
3. **Slow Training**: Check GPU utilization, consider reducing model complexity
4. **TensorFlow Conflicts**: Project uses tensorflow-cpu to avoid CUDA conflicts (trained on NVidia 5070 Ti that currently requires latest nightly builds of pytorch and has incompatibilities with TF GPU dependencies)

For detailed logs, check:
- Training logs: `models/production/logs/`
- Model-specific logs: `models/production/models/{type}/{variation}/{dataset}/*/training.log`