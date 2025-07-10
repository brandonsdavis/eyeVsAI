FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML models
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for static files
RUN mkdir -p static/images

# Copy models directory (needed for imports)
COPY models /app/models

# Install the core models package
RUN cd /app/models/core && pip install -e .

# Copy application code
COPY backend/api .

# Set Python path to include models directory
ENV PYTHONPATH="/app/models:$PYTHONPATH"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]