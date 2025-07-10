# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**eyeVsAI** is a production-ready machine learning image classification system that combines multiple ML model implementations with a full-stack web application. The project features:

- 4 model types: Shallow Learning (scikit-learn), Deep Learning v1/v2 (PyTorch), Transfer Learning (TensorFlow/Keras)
- Unified CLI interface for training any model
- Production training pipeline with hyperparameter tuning and model registry
- FastAPI backend with React frontend game
- Docker-based deployment infrastructure

## Common Development Commands

### Testing
```bash
# Run all model tests
python models/test_models.py -v

# Run specific test class
python models/test_models.py --test TestShallowLearning

# Run structural tests (no dependencies required)
python models/test_structure.py

# Run configuration tests
python models/test_configs.py

# Run with pytest directly
pytest
pytest -v --cov  # with coverage
```

### Linting and Code Quality
```bash
# Format code with black
black models/

# Check linting with flake8
flake8 models/ --exclude=venv,__pycache__

# Type checking with mypy
mypy models/ --ignore-missing-imports
```

### Model Training
```bash
# List all available models
python train_models.py list

# Install dependencies for a specific model
python train_models.py install deep-v2

# Train a model using unified CLI
python train_models.py shallow --data_path ./data --epochs 100
python train_models.py deep-v2 --data_path ./data --batch_size 8 --memory_efficient

# Production training pipeline
cd models/production
python scripts/train_all_production_models.py --models transfer --datasets combined --run_tuning
```

### Running Services
```bash
# Run full stack with Docker
docker-compose up --build

# Run API server in development
cd backend/api
uvicorn app.main:app --reload

# Run frontend in development
cd frontend/game
npm start
```

### Data Management
```bash
# Setup datasets
python setup_data.py

# Cleanup old models and logs
python cleanup.py --old-models --days 30 --dry-run  # preview
python cleanup.py --logs --checkpoints --confirm  # execute
```

## Code Architecture

### Directory Structure
```
models/
├── classifiers/          # Model implementations
│   ├── shallow/         # Traditional ML (SVM, Random Forest)
│   ├── deep-v1/         # Basic CNN (PyTorch)
│   ├── deep-v2/         # Advanced CNN with attention (PyTorch)
│   └── transfer/        # Transfer learning (TensorFlow/Keras)
├── core/                # Base classes and utilities
├── production/          # Production training pipeline
└── train_models.py      # Unified CLI launcher
```

### Key Design Patterns

1. **BaseImageClassifier Interface**: All models implement this interface for consistency
   - `load_model(model_path)`: Load trained model
   - `predict(image)`: Single image prediction
   - `predict_batch(images)`: Batch prediction
   - `get_metadata()`: Model configuration and performance

2. **Modular Structure**: Each model type is self-contained with:
   - `src/`: Production modules (config, model, trainer, classifier)
   - `scripts/`: CLI training scripts
   - `notebooks/`: Development notebooks

3. **Configuration Management**: Dataclass-based configs with serialization
   ```python
   @dataclass
   class DeepLearningV2Config(DeepLearningConfig):
       mixup_alpha: float = 0.2
       memory_efficient: bool = True
   ```

4. **Memory Optimization**: 
   - Lazy loading datasets
   - Gradient accumulation for effective larger batches
   - Mixed precision training
   - Aggressive memory cleanup

### Production Training Pipeline

The production system (`models/production/`) provides:
- Automated hyperparameter tuning with Optuna
- Model registry with comprehensive metadata tracking
- ONNX export for cross-platform deployment
- Parallel training across multiple models and datasets

Key files:
- `configs/models.json`: Model type definitions
- `configs/datasets.json`: Dataset configurations
- `scripts/train_all_production_models.py`: Master training pipeline
- `training_registry.json`: Centralized model tracking

### Adding New Models

To add a new model type:
1. Create directory under `models/classifiers/` following existing structure
2. Implement `BaseImageClassifier` interface in `src/classifier.py`
3. Add configuration dataclass extending base config
4. Create CLI script in `scripts/train.py`
5. Register in `train_models.py` unified launcher

## Important Development Notes

1. **Working Directory**: Always run commands from the project root directory
2. **GPU Memory**: Adjust batch sizes based on available VRAM (use `--batch_size 8` for limited memory)
3. **Dependencies**: Each model can have different dependencies - use `train_models.py install <model>` to install
4. **Testing**: Run tests before committing changes, especially `test_structure.py` for code organization
5. **Docker**: The `docker-compose.yml` runs the complete stack (API, frontend, Redis, Nginx)

## Model-Specific Optimal Settings

### Shallow Learning
```bash
python train_models.py shallow --data_path ./data --feature_types hog lbp color_histogram
```

### Deep Learning v2 (Memory Constrained)
```bash
python train_models.py deep-v2 --data_path ./data --batch_size 8 --accumulation_steps 4 --memory_efficient
```

### Transfer Learning
```bash
python train_models.py transfer --data_path ./data --base_model resnet50 --mixed_precision --fine_tune_layers 10
```

## Debugging Tips

1. **Import Errors**: Ensure running from project root with correct PYTHONPATH
2. **CUDA Errors**: Check GPU availability with `torch.cuda.is_available()`
3. **Memory Errors**: Reduce batch size or enable memory_efficient mode
4. **Training Logs**: Check `logs/` directories for detailed training progress
5. **Model Registry**: Use `training_registry.json` to track all trained models