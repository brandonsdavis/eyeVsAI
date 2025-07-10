# EyeVsAI - Production ML Image Classification Game

A comprehensive machine learning project featuring **production-ready image classification models** with a **full-stack web game** where players compete against AI models. The system includes unified CLI interfaces, comprehensive testing, OAuth authentication, leaderboards, and cloud deployment capabilities.

## 🚀 Project Overview

This project is a **complete ML system with a gamified web application** featuring:

### ML Training System
- **4 Complete Model Packages** - Shallow Learning, Deep Learning v1/v2, Transfer Learning
- **Unified CLI Interface** - Single command-line tool for training any model
- **Production Pipeline** - Automated training, hyperparameter tuning, model registry
- **MLOps Integration** - Model versioning, ONNX export, deployment tools

### Game Application
- **Full-Stack Web Game** - Players compete against AI models in image classification
- **OAuth Authentication** - Support for Google, Facebook, GitHub, Discord, Twitter, Apple
- **Real-time Leaderboards** - Global rankings with multiple time periods
- **Cryptographic Fairness** - Provable AI predictions with commitment schemes
- **Cloud Deployment** - Docker containers, S3 storage, Redis caching

## 📁 Project Structure

```
eyeVsAI/
├── models/                        # 🧠 ML Models and Training
│   ├── classifiers/               # Model implementations
│   │   ├── shallow/               # Traditional ML with feature extraction
│   │   │   ├── src/               # Extracted production modules
│   │   │   │   ├── config.py      # Configuration dataclass
│   │   │   │   ├── feature_extractor.py  # HOG, LBP, color features
│   │   │   │   ├── classifier.py  # BaseImageClassifier implementation
│   │   │   │   ├── trainer.py     # Training and evaluation
│   │   │   │   └── __init__.py    # Package exports
│   │   │   ├── scripts/           # CLI training scripts
│   │   │   │   └── train.py       # Command-line interface
│   │   │   └── notebooks/         # Updated notebooks using extracted modules
│   │   │
│   │   ├── deep-v1/               # Basic CNN implementation
│   │   │   ├── src/               # PyTorch-based modules
│   │   │   │   ├── config.py      # Deep learning configuration
│   │   │   │   ├── model.py       # CNN architecture
│   │   │   │   ├── data_loader.py # Memory-efficient data loading
│   │   │   │   ├── trainer.py     # Training with GPU optimization
│   │   │   │   ├── classifier.py  # Prediction interface
│   │   │   │   └── __init__.py    # Package exports
│   │   │   └── scripts/train.py   # CLI with GPU/CPU support
│   │   │
│   │   ├── deep-v2/               # Advanced CNN with attention
│   │   │   ├── src/               # Advanced PyTorch implementation
│   │   │   │   ├── config.py      # Advanced training configuration
│   │   │   │   ├── model.py       # ResNet + Attention mechanisms
│   │   │   │   ├── data_loader.py # Lazy loading with mixup
│   │   │   │   ├── trainer.py     # Memory-efficient training
│   │   │   │   ├── classifier.py  # Advanced prediction interface
│   │   │   │   └── __init__.py    # Package exports
│   │   │   └── scripts/train.py   # CLI with memory optimization
│   │   │
│   │   └── transfer/              # Transfer learning with pre-trained models
│   │       ├── src/               # TensorFlow/Keras modules
│   │       │   ├── config.py      # Transfer learning configuration
│   │       │   ├── models.py      # Pre-trained model integration
│   │       │   ├── data_loader.py # TensorFlow data pipeline
│   │       │   ├── trainer.py     # Two-phase training
│   │       │   ├── classifier.py  # Transfer learning interface
│   │       │   └── __init__.py    # Package exports
│   │       └── scripts/train.py   # CLI with model selection
│   │
│   ├── core/                      # Core infrastructure
│   │   └── src/                   # Base classes and utilities
│   │       ├── base_classifier.py # BaseImageClassifier interface
│   │       ├── base_trainer.py    # BaseTrainer interface
│   │       ├── base_config.py     # Configuration base classes
│   │       ├── model_registry.py  # Model metadata management
│   │       └── utils.py           # Utility functions
│   │
│   ├── production/                # Production training pipeline
│   │   ├── configs/               # Model and dataset configurations
│   │   │   ├── models.json        # Model type definitions
│   │   │   └── datasets.json      # Dataset configurations
│   │   ├── scripts/               # Training automation scripts
│   │   │   ├── production_trainer.py        # Individual model training
│   │   │   ├── hyperparameter_tuner.py      # Automated hyperparameter tuning
│   │   │   └── train_all_production_models.py # Master training pipeline
│   │   ├── models/                # Trained model outputs
│   │   ├── logs/                  # Training logs
│   │   └── results/               # Training results and reports
│   │
│   ├── data/                      # Training datasets
│   │   └── downloads/             # Downloaded dataset storage
│   │
│   └── checkpoints/               # Model checkpoints
│
├── backend/                       # 🔧 Backend Services
│   ├── api/                       # Game API server
│   │   ├── app/                   # FastAPI application
│   │   │   ├── routes/            # API endpoints
│   │   │   │   ├── auth.py        # Authentication & OAuth
│   │   │   │   ├── game.py        # Game management
│   │   │   │   └── images.py      # Image serving
│   │   │   ├── services/          # Business logic
│   │   │   │   ├── model_manager.py      # AI model integration
│   │   │   │   ├── game_backend_service.py # Game logic
│   │   │   │   ├── cache_service.py      # Redis caching
│   │   │   │   └── s3_service.py         # Cloud storage
│   │   │   ├── auth.py            # JWT & OAuth logic
│   │   │   ├── database.py        # Database setup
│   │   │   ├── db_models.py       # SQLAlchemy models
│   │   │   └── main.py            # FastAPI app
│   │   ├── requirements.txt       # Python dependencies
│   │   ├── .env.example           # Environment variables
│   │   ├── start.sh               # Startup script
│   │   └── API_DOCUMENTATION.md   # API reference
│   │
│   └── deploy/                    # Deployment configuration
│       └── nginx.conf             # Nginx configuration
│
├── frontend/                      # 🎮 Frontend Game
│   └── game/                      # Classification game interface
│       ├── src/                   # React game implementation
│       ├── Dockerfile             # Container configuration
│       └── package.json           # Node.js dependencies
│
├── docs/                          # 📚 Documentation
│   ├── CLI_USAGE.md              # Complete CLI documentation
│   ├── CLI_USAGE_PRODUCTION_TRAINING.md  # Production training guide
│   ├── TESTING_SUMMARY.md        # Testing framework guide
│   ├── CLEANUP_GUIDE.md          # Model cleanup documentation
│   └── PROJECT_COMPLETION_SUMMARY.md # Implementation overview
│
├── GAME_DESIGN.md                # Game design and architecture
├── CLAUDE.md                     # AI assistant instructions
│
├── train_models.py               # 🎯 UNIFIED CLI LAUNCHER
├── test_models.py                # Comprehensive testing suite
├── test_structure.py             # Structural validation tests
├── test_configs.py               # Configuration testing
├── docker-compose.yml            # Multi-service deployment
└── README.md                     # This file
```

## ✨ Key Features

### 🏗️ **Production-Ready Architecture**
- **Modular Design**: All notebook code extracted into reusable Python modules
- **Consistent Interfaces**: All models implement `BaseImageClassifier` for uniform API
- **Memory Optimization**: Lazy loading, gradient accumulation, GPU memory management
- **Error Handling**: Robust error management with informative logging

### 🛠️ **Unified Development Experience** 
- **Single CLI**: `train_models.py` - one interface for training any model
- **Dependency Management**: Automatic dependency checking and installation
- **Configuration**: Dataclass-based configs with serialization support
- **Testing Framework**: Comprehensive structural and unit tests

### 🚀 **MLOps Best Practices**
- **Model Registry**: Centralized metadata tracking and versioning
- **Reproducible Training**: Configuration-driven experiments
- **Quality Assurance**: 10/10 passing structural tests, 100% license coverage
- **Documentation**: Complete usage guides and API documentation

### ⚡ **Performance Optimizations**
- **Memory Efficiency**: Lazy dataset loading, aggressive memory cleanup
- **Training Acceleration**: Mixed precision, XLA compilation, gradient accumulation
- **GPU Optimization**: Automatic device detection and memory monitoring
- **Advanced Techniques**: Attention mechanisms, mixup augmentation, label smoothing

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd eyeVsAI

# Set up Python environment (recommended: Python 3.9+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ML dependencies
pip install -r requirements.txt

# Install backend dependencies
cd backend/api
pip install -r requirements.txt
cd ../..
```

### 2. List Available Models

```bash
# View all available models and their status
python train_models.py list
```

Output:
```
Available Image Classification Models:
============================================================

SHALLOW: Shallow Learning Classifier
  Description: Traditional machine learning with feature extraction (HOG, LBP, color histograms)
  Framework: scikit-learn
  Status: ✅ Ready  (or ❌ Missing: dependencies)
  Example: python train_models.py shallow --data_path /path/to/data --epochs 100

DEEP-V2: Deep Learning v2 Classifier
  Description: Advanced CNN with ResNet, attention mechanisms, and modern techniques
  Framework: PyTorch
  Status: ✅ Ready
  Example: python train_models.py deep-v2 --data_path /path/to/data --epochs 25 --batch_size 8
```

### 3. Install Dependencies for Specific Models

```bash
# Install dependencies for a specific model
python train_models.py install shallow
python train_models.py install deep-v2
python train_models.py install transfer
```

### 4. Train Models

```bash
# Train individual models
python train_models.py shallow --data_path ./data/downloads/pets --epochs 100
python train_models.py deep-v2 --data_path ./data/downloads/pets --batch_size 8
python train_models.py transfer --data_path ./data/downloads/pets --base_model resnet50

# Run production training pipeline
cd models/production
python scripts/train_all_production_models.py \
    --models shallow deep_v1 deep_v2 transfer \
    --datasets pets vegetables instruments street_foods combined \
    --parallel_jobs 4 \
    --run_tuning \
    --auto-cleanup

# Generate game backend report
python scripts/train_all_production_models.py --generate_reports_only
```

### 5. Get Model-Specific Help

```bash
# Get detailed help for any model
python train_models.py shallow --help
python train_models.py deep-v2 --help
python train_models.py transfer --help
```

## 📊 Model Implementations

| Model | Framework | Key Features | Memory Efficiency | Production Ready |
|-------|-----------|--------------|-------------------|------------------|
| **Shallow Learning** | scikit-learn | HOG, LBP, Color Histograms, Texture Features | Optimized feature extraction | ✅ |
| **Deep Learning v1** | PyTorch | Basic CNN, Standard training techniques | Batch optimization | ✅ |
| **Deep Learning v2** | PyTorch | ResNet + Attention, Advanced regularization | Gradient accumulation, Lazy loading | ✅ |
| **Transfer Learning** | TensorFlow/Keras | Pre-trained models, Two-phase training | Mixed precision, Dataset caching | ✅ |

### Model-Specific Features

#### Shallow Learning Classifier
- **Feature Types**: HOG, LBP, color histograms, texture features
- **Algorithms**: SVM, Random Forest, Logistic Regression
- **Optimization**: Feature selection, parameter tuning

#### Deep Learning v1 Classifier  
- **Architecture**: Basic CNN with standard layers
- **Training**: Standard backpropagation with optimization
- **Features**: Dropout, batch normalization, early stopping

#### Deep Learning v2 Classifier
- **Architecture**: ResNet with Attention mechanisms (channel + spatial)
- **Advanced Training**: Mixup augmentation, label smoothing, gradient accumulation
- **Memory Optimization**: Lazy loading, memory monitoring, aggressive cleanup
- **Techniques**: Self-attention, residual connections, progressive dropout

#### Transfer Learning Classifier
- **Pre-trained Models**: ResNet50, VGG16, EfficientNet, MobileNet, Inception
- **Training Strategy**: Two-phase (frozen backbone → fine-tuning)
- **Optimization**: Mixed precision, XLA compilation, class weights
- **TensorFlow Features**: tf.data pipeline, automatic augmentation

## 🧪 Testing Framework

### Comprehensive Quality Assurance

```bash
# Run structural tests (no dependencies required)
python test_structure.py

# Run full functionality tests (when dependencies available)
python test_models.py

# Run configuration-specific tests
python test_configs.py
```

## 🏗️ Development Workflow

### Using Extracted Modules in Notebooks

All notebooks have been updated to use the extracted modules:

```python
# Example from deep_learning_v2_development.ipynb
from src.config import DeepLearningV2Config
from src.classifier import DeepLearningV2Classifier
from src.trainer import MemoryEfficientTrainingManager

# Create configuration
config = DeepLearningV2Config(
    image_size=(96, 96),
    batch_size=8,
    memory_efficient=True
)

# Train model using extracted modules
classifier = DeepLearningV2Classifier(config=config)
trainer = MemoryEfficientTrainingManager(classifier, config)
results = trainer.train("/path/to/data")
```

### Production Integration

```python
# Load any trained model with consistent interface
from models.classifiers.deep_v2.src.classifier import DeepLearningV2Classifier

classifier = DeepLearningV2Classifier()
classifier.load_model("models/production/models/trained_model.pth")

# Make predictions
predictions = classifier.predict(image_array)
metadata = classifier.get_metadata()
```

## 📚 Documentation

### Complete Documentation Suite
- **[CLI_USAGE.md](CLI_USAGE.md)** - Comprehensive CLI usage guide
- **[TESTING_SUMMARY.md](TESTING_SUMMARY.md)** - Testing framework documentation  
- **[PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md)** - Implementation overview
- **Individual module documentation** - Docstrings and examples in each package

### API Documentation
All classifiers implement the `BaseImageClassifier` interface:

```python
class BaseImageClassifier:
    def load_model(self, model_path: str) -> None:
        """Load trained model from file."""
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Predict class probabilities for input image."""
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Predict for multiple images."""
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata and configuration."""
```

## ⚡ Performance Optimizations

### Memory Efficiency
- **Lazy Loading**: Image paths loaded only when accessed
- **Gradient Accumulation**: Effective large batch sizes with limited memory
- **Memory Monitoring**: Real-time GPU/RAM usage tracking
- **Automatic Cleanup**: Aggressive memory management between epochs

### Training Acceleration
- **Mixed Precision**: 16-bit training for faster computation
- **XLA Compilation**: TensorFlow graph optimization
- **Dataset Caching**: In-memory dataset caching for repeated epochs
- **Parallel Processing**: Multi-worker data loading

### Advanced Techniques
- **Attention Mechanisms**: Channel and spatial attention for better feature extraction
- **Mixup Augmentation**: Data augmentation for improved generalization
- **Label Smoothing**: Better model calibration and confidence estimates
- **Two-Phase Training**: Optimal transfer learning with frozen → fine-tuning phases

## 🔧 Configuration Management

### Environment-Agnostic Configuration

All models use dataclass-based configuration with serialization support:

```python
# Example: Deep Learning v2 Configuration
@dataclass
class DeepLearningV2Config(DeepLearningConfig):
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.3
    label_smoothing: float = 0.05
    accumulation_steps: int = 4
    memory_efficient: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)
```

### Model Registry Integration

```python
from models.core.src.model_registry import ModelRegistry, ModelMetadata

registry = ModelRegistry()
metadata = ModelMetadata(
    name="deep-learning-v2",
    version="2.0.0", 
    model_type="deep_v2",
    accuracy=0.94,
    training_date="2024-01-01",
    model_path="./models/deep_v2_classifier.pth",
    config=config.to_dict(),
    performance_metrics={"f1_score": 0.92}
)
registry.register_model(metadata)
```

## 🎮 Game Backend Setup

### 1. Database Setup

```bash
# Option A: PostgreSQL (recommended for production)
docker run -d --name eyevsai-db \
    -e POSTGRES_USER=eyevsai \
    -e POSTGRES_PASSWORD=your-password \
    -e POSTGRES_DB=eyevsai_db \
    -p 5432:5432 \
    postgres:15

# Option B: SQLite (for development)
# Automatically created when you start the backend
```

### 2. Configure Environment

```bash
cd backend/api
cp .env.example .env
# Edit .env with your configuration:
# - Database URL
# - OAuth client IDs and secrets
# - AWS credentials (optional)
# - Redis URL (optional)
```

### 3. Start Backend API

```bash
cd backend/api
./start.sh
# API will be available at http://localhost:8000
# Swagger docs at http://localhost:8000/api/docs
```

### 4. OAuth Setup

To enable social logins, configure OAuth apps:

1. **Google**: https://console.cloud.google.com/
   - Create OAuth 2.0 Client ID
   - Add redirect URI: `http://localhost:3000/auth/callback/google`

2. **Facebook**: https://developers.facebook.com/
   - Create App → Add Facebook Login
   - Valid OAuth Redirect URI: `http://localhost:3000/auth/callback/facebook`

3. **GitHub**: https://github.com/settings/developers
   - New OAuth App
   - Authorization callback URL: `http://localhost:3000/auth/callback/github`

4. Add credentials to `.env` file

## 🚀 Production Deployment

### 1. Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Services:
# - API: http://localhost:8000
# - Frontend: http://localhost:3000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
```

### 2. Model Deployment to S3

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Upload models to S3
python scripts/deploy_models_to_s3.py \
    --model_dir ./models/production/models \
    --bucket eyevsai-models

# Upload training images
python scripts/deploy_images_to_s3.py \
    --data_dir ./data/downloads \
    --bucket eyevsai-images
```

### 3. Environment Variables

```bash
# Production environment variables
DATABASE_URL=postgresql://user:pass@db-host/eyevsai_db
REDIS_URL=redis://redis-host:6379/0
SECRET_KEY=generate-with-openssl-rand-hex-32
ALLOWED_ORIGINS=https://yourdomain.com
S3_BUCKET_NAME=eyevsai-models
S3_BUCKET_IMAGES=eyevsai-images
```

### CLI Integration in Production

```bash
# Production training pipeline
python models/production/scripts/train_all_production_models.py \
    --models transfer \
    --datasets combined \
    --run_tuning \
    --parallel_jobs 2

# Individual model training
python models/classifiers/transfer/scripts/train.py \
    --data_path /production/dataset \
    --base_model resnet50 \
    --epochs 20 \
    --mixed_precision \
    --class_weights \
    --log_dir /logs/production \
    --model_save_path /models/production/transfer_model.h5
```

## 🔍 Monitoring and Logging

### Comprehensive Logging
- **Training Metrics**: Loss, accuracy, learning rate tracking
- **Memory Usage**: GPU/RAM monitoring during training
- **Model Performance**: Test accuracy, confidence analysis
- **Error Handling**: Detailed error logging with context

### Quality Metrics
- **Code Coverage**: 100% structural test coverage
- **Documentation**: Complete docstring coverage for all classes
- **Standards Compliance**: Consistent licensing and formatting
- **Performance**: Memory-efficient implementations across all models

## 🧩 Extension and Customization

### Adding New Models
1. **Create model directory** following the established structure
2. **Implement BaseImageClassifier** interface in classifier.py
3. **Add configuration dataclass** extending base configuration
4. **Create CLI script** following the template pattern
5. **Add to unified CLI** in train_models.py

### Custom Training Workflows
```python
# Example: Custom ensemble training
from train_models import get_model_info, run_model_training

models = ['shallow', 'deep-v2', 'transfer']
for model in models:
    success = run_model_training(model, ['--data_path', '/data', '--epochs', '10'])
    if success:
        print(f"✅ {model} training completed")
```

## 📊 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Create new account
- `POST /api/v1/auth/login` - Login with email/password
- `POST /api/v1/auth/guest` - Create guest session
- `GET /api/v1/auth/oauth/{provider}/authorize` - OAuth login
- `POST /api/v1/auth/oauth/callback` - OAuth callback

### Game
- `GET /api/v1/game/datasets` - Available datasets
- `POST /api/v1/game/session` - Start new game
- `POST /api/v1/game/session/{id}/round` - Get next round
- `POST /api/v1/game/round/{id}/submit` - Submit answer
- `POST /api/v1/game/session/{id}/complete` - Finish game
- `GET /api/v1/game/leaderboard/{dataset}/{difficulty}` - Rankings

## 🔒 Security Features

- **JWT Authentication**: Secure token-based auth
- **OAuth 2.0**: Industry-standard social logins
- **Password Security**: Bcrypt hashing with salt
- **Cryptographic Commitments**: Provable AI predictions
- **SQL Injection Prevention**: Parameterized queries
- **CORS Protection**: Configurable origin validation
- **Rate Limiting**: API request throttling

## 🏗️ Architecture

### Backend Stack
- **Framework**: FastAPI (Python 3.9+)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT + OAuth 2.0
- **Caching**: Redis
- **Storage**: AWS S3 or local filesystem
- **ML Models**: PyTorch, TensorFlow, scikit-learn

### Frontend Stack (Coming Soon)
- **Framework**: React 18+
- **State Management**: Redux Toolkit
- **UI Components**: Material-UI
- **API Client**: Axios with interceptors

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Follow established patterns**: Use existing modules as templates
4. **Add comprehensive tests**: Include structural and unit tests
5. **Update documentation**: Maintain documentation consistency
6. **Submit pull request**: Include testing results and documentation

## 📄 License

Licensed under the Apache License, Version 2.0. See individual module headers for specific licensing information.

---

## 🎯 Quick Reference

### Most Common Commands
```bash
# List all models and their status
python train_models.py list

# Install dependencies for a model  
python train_models.py install deep-v2

# Train with optimal settings
python train_models.py deep-v2 --data_path /data --memory_efficient

# Run all tests
python test_structure.py

# Get help for specific model
python train_models.py transfer --help
```

### Production Commands
```bash
# Train all models on all datasets
python models/production/scripts/train_all_production_models.py --parallel_jobs 4

# Generate game backend report
python models/production/scripts/train_all_production_models.py --generate_reports_only

# Start game backend
cd backend/api && ./start.sh

# Run with Docker
docker-compose up --build
```

## 📈 Performance

### Training Results
- **Best Overall Model**: shallow/rf_hog_lbp on vegetables (97.73% accuracy)
- **Best Transfer Learning**: ResNet101 across all datasets
- **142 Models Trained**: Across 5 datasets with hyperparameter tuning
- **Game-Ready**: Models categorized by difficulty for balanced gameplay
