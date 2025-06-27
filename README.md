# ML Image Classification Project

A comprehensive machine learning project demonstrating multiple classification approaches with a unified data management system and production-ready API deployment. Features automated data downloading, framework-agnostic data loaders, and an interactive web game.

## Project Structure

```
image_game/
├── image-classifier-shallow/     # Traditional ML approaches
├── image-classifier-deep-v1/     # First neural network implementation
├── image-classifier-deep-v2/     # Second neural network implementation
├── image-classifier-transfer/    # Transfer learning implementation
├── ml-models-core/              # Central model registry and utilities
├── ensemble-classifier/         # Ensemble model combining all approaches
├── classification-api/          # Python FastAPI backend
├── classification-game/         # JavaScript React frontend
├── deploy/                      # Deployment scripts and configs
├── docker-compose.yml           # Docker orchestration
└── README.md                    # This file
```

## Features

### 🤖 Multiple Model Types
- **Shallow Learning**: Traditional ML with scikit-learn
- **Deep Learning v1**: Basic CNN with PyTorch  
- **Deep Learning v2**: Advanced CNN with attention mechanisms
- **Transfer Learning**: Pre-trained models with TensorFlow
- **Ensemble**: Combining all model predictions

### 📊 Unified Data Management
- **Automated downloading** of Oxford Pets, Kaggle Vegetables, Street Foods, and Musical Instruments datasets
- **Multi-class dataset creation** for 2, 3, and 4-class classification problems
- **Data validation and integrity checking**
- **Consistent preprocessing** across all models
- **Framework-agnostic data loaders** (PyTorch, TensorFlow, scikit-learn)

### 🎮 Interactive Game
- **Challenge Mode**: Compete against AI models in identifying images
- **Upload & Test**: Upload your own images for classification  
- **Real-time Results**: Instant feedback and model confidence scores
- **Leaderboard**: Track performance against other players

### 🏗️ Production Architecture
- **FastAPI backend** with async model serving
- **React frontend** with interactive classification game
- **Docker deployment** on EC2 
- **Model registry** for version management
- **Single Python venv** with unified dependency management

## Quick Start

### 1. Environment Setup

Set up the unified Python virtual environment:

```bash
# Setup single project environment with all dependencies
./setup_environment.sh
```

This creates one unified virtual environment with all dependencies:
- PyTorch nightly with CUDA 12.8 support for deep learning models
- TensorFlow CPU for transfer learning  
- Scikit-learn for traditional ML approaches
- FastAPI and web development tools
- Jupyter notebooks with proper kernel registration

### 2. Data Setup

Download and prepare all datasets:

```bash
# Activate the virtual environment
source venv/bin/activate

# Download all datasets and create combined datasets
python setup_data.py

# Or download specific datasets
python setup_data.py --datasets oxford_pets kaggle_vegetables street_foods musical_instruments

# Create multi-class datasets
python setup_data.py --create-three-class --create-four-class

# Generate analysis reports
python setup_data.py --reports
```

### 3. Model Development

All models use the same unified environment with Jupyter kernel:

```bash
# Activate the virtual environment
source venv/bin/activate

# Start Jupyter Lab with access to all model notebooks
jupyter lab

# Or start Jupyter Notebook
jupyter notebook

# Select "ML Image Classification (Python 3.11)" kernel for all notebooks:
# - image-classifier-shallow/notebooks/shallow_learning_development.ipynb
# - image-classifier-deep-v1/notebooks/deep_learning_v1_development.ipynb
# - image-classifier-deep-v2/notebooks/deep_learning_v2_development.ipynb
# - image-classifier-transfer/notebooks/transfer_learning_development.ipynb
```

The environment stays activated in your terminal session, so you can run any Python scripts or Jupyter commands directly.

### 4. API and Game Deployment

```bash
# Start the full application stack
docker-compose up --build
```

- **API**: http://localhost:8000
- **Game**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## Architecture

### Models
- **Shallow Learning**: Traditional ML with feature extraction
- **Deep Learning v1**: Custom neural network architecture
- **Deep Learning v2**: Improved neural network with optimizations
- **Transfer Learning**: Pre-trained model with fine-tuning layers
- **Ensemble**: Combination of all models using voting strategies

### Backend (FastAPI)
- RESTful API with automatic documentation
- Async model loading and prediction
- Image upload and processing
- Game challenge management
- Model performance tracking

### Frontend (React)
- Material-UI components
- Responsive design
- Real-time predictions
- Interactive game interface
- File upload with drag-and-drop

### Infrastructure
- Docker containerization for EC2 deployment
- Nginx reverse proxy
- Simplified architecture (no Redis/Kubernetes overkill)
- Configurable deployment

## API Endpoints

### Health & Status
- `GET /api/v1/health` - Health check
- `GET /api/v1/models/status` - Model loading status
- `GET /api/v1/models` - List available models

### Predictions
- `POST /api/v1/predict` - Predict from base64 image
- `POST /api/v1/predict/upload` - Predict from uploaded file

### Game
- `POST /api/v1/game/challenge` - Create game challenge
- `POST /api/v1/game/submit` - Submit game answer

## Model Integration

### Adding Your Trained Models

1. **Place model files** in the respective directories:
   - `image-classifier-shallow/models/`
   - `image-classifier-deep-v1/models/`
   - `image-classifier-deep-v2/models/`
   - `image-classifier-transfer/models/`

2. **Implement the BaseImageClassifier interface** in each model directory:
   ```python
   from ml_models_core.base_classifier import BaseImageClassifier
   
   class YourClassifier(BaseImageClassifier):
       def load_model(self, model_path: str) -> None:
           # Load your trained model
       
       def predict(self, image: np.ndarray) -> Dict[str, float]:
           # Return class predictions
   ```

3. **Update model manager** in `classification-api/app/services/model_manager.py` to load your models

### Model Registry

The `ml-models-core` provides a centralized registry for model metadata:

```python
from ml_models_core import ModelRegistry, ModelMetadata

registry = ModelRegistry()
metadata = ModelMetadata(
    name="my-model",
    version="1.0.0",
    model_type="deep_v1",
    accuracy=0.95,
    training_date="2024-01-01",
    model_path="./models/my-model.pkl",
    config={},
    performance_metrics={"f1_score": 0.93}
)
registry.register_model(metadata)
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `API_HOST`, `API_PORT` - API server settings
- `REACT_APP_API_URL` - Frontend API endpoint
- `MODEL_PATH` - Path to model files
- `REDIS_HOST`, `REDIS_PORT` - Cache settings
- `LOG_LEVEL` - Logging configuration

### Model Configuration

Models can be configured through the registry system or individual configuration files in each model directory.

## Monitoring

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f api
docker-compose logs -f frontend
```

### Health Checks
- API: `curl http://localhost:8000/api/v1/health`
- Models: `curl http://localhost:8000/api/v1/models/status`

## Development

### Adding New Features

1. **Backend**: Add endpoints in `classification-api/app/routes/`
2. **Frontend**: Add components in `classification-game/src/components/`
3. **Models**: Extend `BaseImageClassifier` for new model types

### Testing

```bash
# Backend tests
cd classification-api
pytest

# Frontend tests
cd classification-game
npm test
```

## Deployment to EC2

1. **Launch EC2 instance** with Docker support
2. **Clone repository** on the instance
3. **Configure environment** variables for production
4. **Run deployment script**:
   ```bash
   ./deploy/start.sh
   ```
5. **Set up domain/SSL** (optional) using Let's Encrypt

### Production Considerations

- Use environment-specific `.env` files
- Configure proper CORS origins
- Set up log rotation
- Monitor resource usage
- Implement backup strategies for model files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for demonstration purposes. See individual model directories for specific licensing information.