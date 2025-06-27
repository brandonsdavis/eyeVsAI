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

from pydantic import BaseModel
from typing import Dict, List, Optional
from enum import Enum


class ModelType(str, Enum):
    SHALLOW = "shallow"
    DEEP_V1 = "deep_v1"
    DEEP_V2 = "deep_v2"
    TRANSFER = "transfer"
    ENSEMBLE = "ensemble"


class PredictionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    model_type: Optional[ModelType] = ModelType.ENSEMBLE
    top_k: Optional[int] = 3


class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    top_predictions: List[tuple]
    model_used: str
    confidence_score: float
    processing_time: float


class ModelInfo(BaseModel):
    name: str
    version: str
    type: ModelType
    accuracy: float
    is_loaded: bool


class GameChallengeRequest(BaseModel):
    category: str  # "dogs_cats" or "fruits_vegetables"
    difficulty: str  # "easy", "medium", "hard"


class GameChallengeResponse(BaseModel):
    image_url: str
    correct_answer: str
    options: List[str]
    challenge_id: str


class GameSubmissionRequest(BaseModel):
    challenge_id: str
    user_answer: str


class GameSubmissionResponse(BaseModel):
    correct: bool
    correct_answer: str
    model_predictions: Dict[str, float]
    user_score: int
    explanation: str