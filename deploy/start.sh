#!/bin/bash
# Copyright 2025 Brandon Davis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Deployment script for Image Classification Game

set -e

echo "Starting Image Classification Game deployment..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration before proceeding."
    echo "Then run this script again."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models data logs

# Build and start services
echo "Building and starting services..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking service health..."
if curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "API is healthy"
else
    echo "API health check failed"
    docker-compose logs api
fi

if curl -s http://localhost:3000 > /dev/null; then
    echo "Frontend is healthy"
else
    echo "Frontend health check failed"
    docker-compose logs frontend
fi

echo "Deployment complete!"
echo "Frontend: http://localhost"
echo "API: http://localhost/api/v1"
echo "API Direct: http://localhost:8000"

echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
echo "To restart: docker-compose restart"