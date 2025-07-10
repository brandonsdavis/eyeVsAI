#!/bin/bash

# Start script for EyeVsAI Backend API

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-1}

echo "Starting EyeVsAI Backend API..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Run database migrations if alembic is available
if command -v alembic &> /dev/null; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Start the API
if [ "$WORKERS" -gt 1 ]; then
    # Production mode with multiple workers
    gunicorn app.main:app \
        --workers $WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind $HOST:$PORT \
        --access-logfile - \
        --error-logfile -
else
    # Development mode with auto-reload
    python -m uvicorn app.main:app \
        --host $HOST \
        --port $PORT \
        --reload \
        --log-level info
fi