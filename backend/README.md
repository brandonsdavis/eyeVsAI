# Eye vs AI Backend API

A FastAPI-based REST API that serves as the backend for the Eye vs AI game. The API manages game sessions, serves ML model predictions, handles user authentication, and provides leaderboard functionality.

## Overview

The backend provides a robust foundation for the image classification game, featuring:
- RESTful API design with OpenAPI documentation
- ML model serving with real-time predictions
- User authentication and session management
- Game logic and scoring systems
- Database management for user data and statistics
- Containerized deployment with Docker

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM for database operations
- **SQLite/PostgreSQL**: Database for user data and game statistics
- **Pydantic**: Data validation and settings management
- **PyTorch/scikit-learn**: ML model inference
- **Docker**: Containerized deployment
- **Uvicorn**: ASGI server for production deployment

## Project Structure

```
backend/
├── api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application entry point
│   │   ├── auth.py           # Authentication logic
│   │   ├── database.py       # Database configuration
│   │   ├── db_models.py      # SQLAlchemy models
│   │   ├── models.py         # Pydantic models (schemas)
│   │   └── routers/          # API route definitions
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile           # Container configuration
│   └── start.sh             # Application startup script
└── README.md               # This file
```

## API Endpoints

### Game Management

#### Start New Game
```http
POST /api/game/start
Content-Type: application/json

{
    "dataset": "pets",
    "difficulty": "medium",
    "player_name": "John Doe",
    "game_mode": "standard"
}
```

**Response:**
```json
{
    "game_id": "uuid-string",
    "session_id": "session-uuid",
    "ai_model": {
        "name": "ResNet50",
        "accuracy": 0.847,
        "model_type": "transfer"
    },
    "dataset_info": {
        "name": "Oxford Pets",
        "num_classes": 37,
        "description": "Pet breed classification"
    }
}
```

#### Get Next Round
```http
GET /api/game/{game_id}/next
```

**Response:**
```json
{
    "round_number": 1,
    "image_url": "/api/images/pets/123.jpg",
    "options": [
        {"id": "a", "label": "Golden Retriever"},
        {"id": "b", "label": "Labrador"},
        {"id": "c", "label": "German Shepherd"},
        {"id": "d", "label": "Beagle"}
    ],
    "time_limit": 30
}
```

#### Submit Answer
```http
POST /api/game/{game_id}/answer
Content-Type: application/json

{
    "round_number": 1,
    "selected_option": "a",
    "time_taken": 12.5
}
```

**Response:**
```json
{
    "correct": true,
    "correct_answer": "a",
    "ai_prediction": "a",
    "ai_confidence": 0.92,
    "player_score": 100,
    "ai_score": 100,
    "explanation": "Both player and AI correctly identified the Golden Retriever"
}
```

#### Get Game Results
```http
GET /api/game/{game_id}/results
```

**Response:**
```json
{
    "game_id": "uuid-string",
    "final_scores": {
        "player": 850,
        "ai": 920
    },
    "rounds_completed": 10,
    "player_accuracy": 0.8,
    "ai_accuracy": 0.9,
    "average_response_time": 8.5,
    "winner": "ai",
    "performance_breakdown": [
        {
            "round": 1,
            "player_correct": true,
            "ai_correct": true,
            "category": "Golden Retriever"
        }
    ]
}
```

### Model Management

#### Get Available Models
```http
GET /api/models?difficulty=medium&dataset=pets
```

**Response:**
```json
{
    "models": [
        {
            "id": "transfer_resnet50_pets",
            "name": "ResNet50 Transfer Learning",
            "model_type": "transfer",
            "accuracy": 0.847,
            "difficulty": "medium",
            "description": "Pre-trained ResNet50 fine-tuned on pet images"
        }
    ]
}
```

#### Get Model Information
```http
GET /api/models/{model_id}
```

### Authentication

#### User Registration
```http
POST /api/auth/register
Content-Type: application/json

{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "secure_password"
}
```

#### User Login
```http
POST /api/auth/login
Content-Type: application/json

{
    "username": "johndoe",
    "password": "secure_password"
}
```

**Response:**
```json
{
    "access_token": "jwt-token-string",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
        "id": 1,
        "username": "johndoe",
        "email": "john@example.com"
    }
}
```

### Leaderboards

#### Get Global Leaderboard
```http
GET /api/leaderboard?dataset=pets&difficulty=medium&limit=10
```

**Response:**
```json
{
    "leaderboard": [
        {
            "rank": 1,
            "username": "Alice",
            "score": 950,
            "accuracy": 0.95,
            "games_played": 25,
            "avg_response_time": 7.2
        }
    ],
    "total_players": 156,
    "your_rank": 23
}
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

### Games Table
```sql
CREATE TABLE games (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER REFERENCES users(id),
    dataset VARCHAR(50) NOT NULL,
    difficulty VARCHAR(20) NOT NULL,
    ai_model VARCHAR(100) NOT NULL,
    player_score INTEGER DEFAULT 0,
    ai_score INTEGER DEFAULT 0,
    rounds_completed INTEGER DEFAULT 0,
    game_status VARCHAR(20) DEFAULT 'active',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

### Game Rounds Table
```sql
CREATE TABLE game_rounds (
    id SERIAL PRIMARY KEY,
    game_id UUID REFERENCES games(id),
    round_number INTEGER NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    correct_answer VARCHAR(10) NOT NULL,
    player_answer VARCHAR(10),
    ai_prediction VARCHAR(10) NOT NULL,
    ai_confidence FLOAT NOT NULL,
    player_response_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Model Serving Architecture

### Model Loading
Models are loaded on application startup and cached in memory:

```python
class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models from the model registry."""
        registry_path = Path("../models/production/training_registry.json")
        with open(registry_path) as f:
            registry = json.load(f)
        
        for model_entry in registry["models"]:
            model_id = f"{model_entry['model_type']}_{model_entry['variation']}_{model_entry['dataset']}"
            self.models[model_id] = self.load_model(model_entry)
    
    def predict(self, model_id: str, image_path: str) -> Prediction:
        """Get prediction from specified model."""
        model = self.models[model_id]
        return model.predict(image_path)
```

### Prediction Pipeline
1. Image preprocessing (resize, normalize)
2. Model inference
3. Post-processing (softmax, class mapping)
4. Confidence calculation
5. Response formatting

### Performance Optimization
- Model caching in memory
- Batch prediction support
- Async processing for multiple requests
- GPU utilization when available

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///./eyevsai.db
# or for PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/eyevsai

# Authentication
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model Paths
MODELS_PATH=/app/models
MODEL_REGISTRY_PATH=/app/models/production/training_registry.json

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://eyevsai.com
```

### Application Settings
Settings managed with Pydantic:

```python
class Settings(BaseSettings):
    database_url: str = "sqlite:///./eyevsai.db"
    secret_key: str
    access_token_expire_minutes: int = 60
    
    models_path: Path = Path("../models")
    registry_path: Path = Path("../models/production/training_registry.json")
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"
```

## Deployment

### Development Setup
```bash
# Navigate to backend directory
cd backend/api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=sqlite:///./eyevsai.db
export SECRET_KEY=dev-secret-key

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build Docker image
docker build -t eyevsai-backend .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=sqlite:///./eyevsai.db \
  -e SECRET_KEY=production-secret \
  -v $(pwd)/../models:/app/models \
  eyevsai-backend
```

### Production Deployment
For production with PostgreSQL:
```bash
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/eyevsai \
  -e SECRET_KEY=production-secret \
  -e DEBUG=false \
  -v /path/to/models:/app/models \
  eyevsai-backend
```

## Security Considerations

### Authentication
- JWT tokens with expiration
- Password hashing with bcrypt
- Rate limiting on authentication endpoints
- CORS configuration for frontend origins

### Data Validation
- Pydantic models for request/response validation
- SQL injection prevention with SQLAlchemy
- Input sanitization for user data
- File upload restrictions

### Production Security
- HTTPS enforcement
- Secure headers configuration
- Database connection encryption
- Environment variable security
- API rate limiting

## Monitoring and Logging

### Application Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/eyevsai/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Performance Metrics
- Response time tracking
- Model inference latency
- Database query performance
- Memory usage monitoring
- Error rate tracking

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "models_loaded": len(model_manager.models),
        "database": "connected"
    }
```

## Testing

### Unit Tests
```python
# Test example
def test_game_creation():
    client = TestClient(app)
    response = client.post("/api/game/start", json={
        "dataset": "pets",
        "difficulty": "medium",
        "player_name": "Test Player"
    })
    assert response.status_code == 200
    assert "game_id" in response.json()
```

### Integration Tests
- Database operations
- Model predictions
- Authentication flow
- Complete game sessions

### Load Testing
```bash
# Using locust for load testing
pip install locust
locust -f tests/load_test.py --host http://localhost:8000
```

## Performance Optimization

### Database Optimization
- Connection pooling
- Query optimization with indexes
- Async database operations
- Caching frequently accessed data

### Model Serving
- Model warm-up on startup
- Batch prediction support
- GPU memory management
- Model quantization for faster inference

### API Performance
- Async request handling
- Response caching
- Compression for large responses
- CDN integration for static assets

## Future Enhancements

### Planned Features
1. **Real-time Multiplayer**: WebSocket support for live games
2. **Advanced Analytics**: Detailed player performance tracking
3. **Model A/B Testing**: Compare different AI opponents
4. **Custom Datasets**: User-uploaded image classification
5. **Tournament System**: Scheduled competitions

### Technical Improvements
1. **Microservices**: Split into game, auth, and model services
2. **Message Queues**: Async processing with Redis/RabbitMQ
3. **Caching Layer**: Redis for session and leaderboard caching
4. **Monitoring**: Prometheus metrics and Grafana dashboards
5. **CI/CD**: Automated testing and deployment pipelines

### Production Scaling
For handling increased load:
1. **Horizontal Scaling**: Multiple API instances behind load balancer
2. **Database Sharding**: Distribute data across multiple databases
3. **Model Serving**: Dedicated model inference service
4. **CDN Integration**: Global content delivery
5. **Auto-scaling**: Kubernetes deployment with HPA

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check DATABASE_URL format
   - Verify database server is running
   - Check connection permissions

2. **Model Loading Failures**
   - Verify model file paths
   - Check model registry format
   - Ensure sufficient memory

3. **Authentication Issues**
   - Verify JWT secret key
   - Check token expiration
   - Validate user credentials

4. **Performance Problems**
   - Monitor database query performance
   - Check model inference times
   - Profile memory usage

### Debug Mode
Enable debug logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## API Documentation

FastAPI automatically generates OpenAPI documentation:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc format: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Development Guidelines

### Code Style
- PEP 8 compliance
- Type hints for all functions
- Docstrings for public methods
- Black formatting
- isort import sorting

### Error Handling
- Comprehensive exception handling
- Meaningful error messages
- Proper HTTP status codes
- Logging for debugging

### Testing Requirements
- Unit tests for all endpoints
- Integration tests for workflows
- Minimum 80% code coverage
- Performance benchmarks

## Support

For issues or questions:
1. Check API documentation at `/docs`
2. Review application logs
3. Consult model serving documentation
4. Contact development team