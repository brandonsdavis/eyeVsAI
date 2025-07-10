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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging

from .routes import classification, health, auth, game, images
from .services.model_manager import ModelManager
from .database import create_tables, engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EyeVsAI Game API",
    description="Backend API for the EyeVsAI image classification game",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware for frontend integration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(game.router, prefix="/api/v1")
app.include_router(images.router, prefix="/api/v1")
app.include_router(classification.router, prefix="/api/v1")  # Legacy support

# Create static directory if it doesn't exist
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        # Create database tables
        logger.info("Creating database tables...")
        create_tables()
        
        # Initialize model manager
        logger.info("Initializing model manager...")
        await model_manager.initialize_models()
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        logger.info("Shutting down application...")
        await model_manager.cleanup()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to EyeVsAI Game API",
        "version": "2.0.0",
        "docs": "/api/docs"
    }

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "app.main:app" if reload else app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )