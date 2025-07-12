# Eye vs AI - Image Classification Game

A comprehensive machine learning project featuring production-ready image classification models with a full-stack web game where players compete against AI models. The system demonstrates various ML approaches from traditional feature engineering to modern deep learning architectures.

## Project Overview

This project serves as a portfolio demonstration of modern machine learning engineering practices, featuring:

### Machine Learning Pipeline
- Four distinct model architectures with different technical approaches
- Unified command-line interface for consistent model training
- Production-grade training pipeline with hyperparameter optimization
- Comprehensive model evaluation and deployment system

### Web Application
- Interactive game where users compete against AI models
- Real-time classification challenges across multiple datasets
- Performance tracking and leaderboard functionality
- Containerized deployment architecture for AWS

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
│   │   │   │   └── trainer.py     # Training and evaluation
│   │   │   ├── scripts/           # CLI training scripts
│   │   │   └── notebooks/         # Jupyter notebooks for experimentation
│   │   │
│   │   ├── deep-v1/               # Basic CNN implementation
│   │   │   ├── src/               # PyTorch-based modules
│   │   │   │   ├── config.py      # Deep learning configuration
│   │   │   │   ├── model.py       # CNN architecture
│   │   │   │   ├── data_loader.py # Memory-efficient data loading
│   │   │   │   ├── trainer.py     # Training with GPU optimization
│   │   │   │   └── classifier.py  # Prediction interface
│   │   │   ├── scripts/           # CLI training scripts
│   │   │   └── notebooks/         # Jupyter notebooks for experimentation
│   │   │
│   │   ├── deep-v2/               # Advanced CNN with attention
│   │   │   ├── src/               # Advanced PyTorch implementation
│   │   │   │   ├── config.py      # Advanced training configuration
│   │   │   │   ├── model.py       # ResNet + Attention mechanisms
│   │   │   │   ├── data_loader.py # Lazy loading with mixup
│   │   │   │   ├── trainer.py     # Memory-efficient training
│   │   │   │   └── classifier.py  # Advanced prediction interface
│   │   │   ├── scripts/           # CLI training scripts
│   │   │   └── notebooks/         # Jupyter notebooks for experimentation
│   │   │
│   │   └── transfer/              # Transfer learning with pre-trained models
│   │       ├── src/               # TensorFlow/Keras modules
│   │       │   ├── config.py      # Transfer learning configuration
│   │       │   ├── models.py      # Pre-trained model integration
│   │       │   ├── data_loader.py # TensorFlow data pipeline
│   │       │   ├── trainer.py     # Two-phase training
│   │       │   └── classifier.py  # Transfer learning interface
│   │       ├── scripts/           # CLI training scripts
│   │       └── notebooks/         # Jupyter notebooks for experimentation
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
├── backend/                       # Backend Services
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
├── frontend/                      # Frontend Game
│   └── game/                      # Classification game interface
│       ├── src/                   # React game implementation
│       ├── Dockerfile             # Container configuration
│       └── package.json           # Node.js dependencies
│
├── GAME_DESIGN.md                # Game design and architecture
├── CLAUDE.md                     # AI assistant instructions
│
├── train_models.py               # UNIFIED CLI LAUNCHER
├── test_models.py                # Comprehensive testing suite
├── test_structure.py             # Structural validation tests
├── test_configs.py               # Configuration testing
├── docker-compose.yml            # Multi-service deployment
└── README.md                     # This file
```

## System Architecture

### Component Structure
The system is organized into three main components, each with dedicated documentation:

- **[models/](models/README.md)** - Machine learning pipeline with four model types
- **[frontend/](frontend/README.md)** - React-based game interface  
- **[backend/](backend/README.md)** - FastAPI server with model serving

### Design Philosophy
This project demonstrates modern ML engineering principles while maintaining focus on educational value and portfolio presentation. The architecture balances production-ready practices with single-server deployment simplicity.

## Quick Start

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd eyeVsAI

# Set up Python environment (Python 3.11 recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Machine Learning Models

The system includes four model types with unified CLI access:

```bash
# View available models
python train_models.py list

# Train individual models
python train_models.py shallow --data_path ./data/pets
python train_models.py deep-v1 --data_path ./data/pets
python train_models.py deep-v2 --data_path ./data/pets --architecture resnet
python train_models.py transfer --data_path ./data/pets --base_model resnet50

# Run production training pipeline
cd models/production
python scripts/train_all_production_models.py \
    --models deep_v1 deep_v2 transfer \
    --parallel_jobs 2 \
    --run_tuning
```

See the [models documentation](models/README.md) for detailed usage instructions.

## Model Types

| Model | Framework | Approach | Use Case |
|-------|-----------|----------|----------|
| **Shallow Learning** | scikit-learn | Traditional ML with feature engineering | Interpretable baseline models |
| **Deep Learning v1** | PyTorch | Basic CNN architecture | Standard deep learning approach |
| **Deep Learning v2** | PyTorch | Advanced CNN with attention | State-of-the-art techniques |
| **Transfer Learning** | TensorFlow | Pre-trained model fine-tuning | Efficient training on limited data |

Each model type includes comprehensive documentation, configuration management, and production deployment capabilities. For detailed technical specifications and usage examples, see the [models documentation](models/README.md).

## AWS Deployment Guide

This system is designed for deployment on a single AWS EC2 instance using Docker containers. The following guide covers production deployment for demonstration purposes.

### Infrastructure Requirements

**Recommended EC2 Instance**: t3.xlarge or larger
- 4+ vCPUs for parallel model training (smaller instances can be used for inference-only)
- 16+ GB RAM for model serving
- 100+ GB EBS storage for models and data

**Additional AWS Services**:
- RDS PostgreSQL (db.t3.micro for demonstration)
- S3 bucket for model storage
- ElastiCache Redis (cache.t3.micro)
- Application Load Balancer (optional)

### 1. EC2 Instance Setup

```bash
# Connect to your EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system and install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io docker-compose -y
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Clone repository
git clone <your-repo-url>
cd eyeVsAI
```

### 2. Environment Configuration

Create production environment file:

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql://username:password@your-rds-endpoint/eyevsai

# Authentication
SECRET_KEY=$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=https://yourdomain.com

# AWS Services
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_MODELS=your-models-bucket
S3_BUCKET_IMAGES=your-images-bucket

# Redis
REDIS_URL=redis://your-elasticache-endpoint:6379/0

# Frontend
REACT_APP_API_URL=https://yourdomain.com/api
EOF
```

### 3. Build and Deploy

```bash
# Build all containers
docker-compose -f docker-compose.prod.yml build

# Deploy services
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
curl http://localhost:8000/health
```

### 4. Model Training and Upload

```bash
# Train models for production
cd models/production
python scripts/train_all_production_models.py \
    --models deep_v1 deep_v2 transfer \
    --parallel_jobs 2 \
    --run_tuning

# Upload trained models to S3
aws s3 sync models/ s3://your-models-bucket/ --exclude "*.log"
```

### 5. Database Setup

```bash
# Initialize database schema
docker-compose exec backend python -c "
from app.database import create_tables
create_tables()
"

# Create initial admin user (optional)
docker-compose exec backend python -c "
from app.auth import create_user
create_user('admin@yourdomain.com', 'secure_password', is_admin=True)
"
```

### 6. SSL and Domain Configuration

For production deployment with a custom domain:

```bash
# Install certbot for SSL certificates
sudo apt install certbot nginx -y

# Configure nginx (update nginx.conf with your domain)
sudo cp deploy/nginx.conf /etc/nginx/sites-available/eyevsai
sudo ln -s /etc/nginx/sites-available/eyevsai /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com

# Start nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### 7. Monitoring and Maintenance

```bash
# View application logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Monitor system resources
htop
docker stats

# Backup database
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql

# Update application
git pull
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

### Production Considerations

**Security**:
- Configure AWS Security Groups to restrict access
- Use IAM roles instead of access keys when possible
- Enable CloudTrail for audit logging
- Set up VPC with private subnets for database

**Scalability**:
- Use Application Load Balancer for multiple instances
- Implement auto-scaling groups for traffic spikes
- Consider ECS or EKS for container orchestration
- Use CloudFront CDN for static assets

**Monitoring**:
- CloudWatch for infrastructure metrics
- Application-level logging with structured logs
- Health checks for container orchestration
- Database performance monitoring

**Cost Optimization**:
- Use Reserved Instances for predictable workloads
- Implement S3 lifecycle policies for old models
- Monitor costs with AWS Cost Explorer

### Testing Deployment

```bash
# Test API endpoints
curl https://yourdomain.com/api/health
curl https://yourdomain.com/api/models

# Test frontend
curl https://yourdomain.com/

# Load test (optional)
ab -n 100 -c 10 https://yourdomain.com/api/health
```

## Local Development

### Running Individual Components

**Backend API**:
```bash
cd backend/api
./start.sh
# Available at http://localhost:8000
```

**Frontend Game**:
```bash
cd frontend/game
npm install
npm start
# Available at http://localhost:3000
```

**Full Stack with Docker**:
```bash
docker-compose up --build
```

## Component Documentation

For detailed information about each component:

- **[Machine Learning Models](models/README.md)** - Training pipelines, model architectures, and production deployment
- **[Frontend Application](frontend/README.md)** - React game interface, components, and build process
- **[Backend API](backend/README.md)** - FastAPI server, endpoints, authentication, and model serving

## Project Features

This project demonstrates modern software engineering practices:

- **Modular Architecture**: Component-based design with clear separation of concerns
- **Production Deployment**: Docker containerization with AWS deployment guide
- **Quality Assurance**: Comprehensive testing and documentation standards
- **Scalable Design**: Architecture considerations for future growth

## License

Licensed under the Apache License, Version 2.0. See individual components for specific licensing information.
