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

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict
import base64
import io
import time
from PIL import Image
import numpy as np

from ..models import (
    PredictionRequest, PredictionResponse, ModelInfo, ModelType,
    GameChallengeRequest, GameChallengeResponse,
    GameSubmissionRequest, GameSubmissionResponse
)
from ..services.model_manager import ModelManager
from ..services.game_service import GameService

router = APIRouter()
model_manager = ModelManager()
game_service = GameService()


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(request: PredictionRequest):
    """Predict image classification."""
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Get model and make prediction
        model = await model_manager.get_model(request.model_type.value)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        predictions = model.predict(image_array)
        top_predictions = model.get_top_predictions(predictions, request.top_k)
        
        processing_time = time.time() - start_time
        confidence_score = max(predictions.values()) if predictions else 0.0
        
        return PredictionResponse(
            predictions=predictions,
            top_predictions=top_predictions,
            model_used=f"{model.model_name}:{model.version}",
            confidence_score=confidence_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/upload")
async def predict_uploaded_image(
    file: UploadFile = File(...),
    model_type: ModelType = ModelType.ENSEMBLE,
    top_k: int = 3
):
    """Predict classification for uploaded image file."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Convert to base64 for reuse with existing prediction logic
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Use existing prediction endpoint
        request = PredictionRequest(
            image_data=image_b64,
            model_type=model_type,
            top_k=top_k
        )
        
        return await predict_image(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload prediction failed: {str(e)}")


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    models_info = await model_manager.get_models_info()
    return models_info


@router.post("/game/challenge", response_model=GameChallengeResponse)
async def create_game_challenge(request: GameChallengeRequest):
    """Create a new game challenge."""
    try:
        challenge = await game_service.create_challenge(
            request.category,
            request.difficulty
        )
        return challenge
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Challenge creation failed: {str(e)}")


@router.post("/game/submit", response_model=GameSubmissionResponse)
async def submit_game_answer(request: GameSubmissionRequest):
    """Submit answer for a game challenge."""
    try:
        result = await game_service.submit_answer(
            request.challenge_id,
            request.user_answer
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer submission failed: {str(e)}")