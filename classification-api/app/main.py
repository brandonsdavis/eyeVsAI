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

from .routes import classification, health
from .services.model_manager import ModelManager

app = FastAPI(
    title="Image Classification API",
    description="API for image classification using multiple ML models",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(classification.router, prefix="/api/v1", dependencies=[])

# Mount static files for serving uploaded images
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    await model_manager.initialize_models()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await model_manager.cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)