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

from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid


class ModelType(str, Enum):
    SHALLOW = "shallow"
    DEEP_V1 = "deep_v1"
    DEEP_V2 = "deep_v2"
    TRANSFER = "transfer"
    ENSEMBLE = "ensemble"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GameStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


# Authentication Models
class UserRegistration(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserProfile(BaseModel):
    id: str
    email: Optional[str]
    username: Optional[str]
    display_name: Optional[str]
    avatar_url: Optional[str]
    is_guest: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    username: Optional[str] = None


# OAuth Models
class OAuthProvider(str, Enum):
    GOOGLE = "google"
    FACEBOOK = "facebook"
    GITHUB = "github"
    DISCORD = "discord"
    TWITTER = "twitter"
    APPLE = "apple"


class OAuthAuthorizationURL(BaseModel):
    authorization_url: str
    state: str


class OAuthCallback(BaseModel):
    code: str
    state: str
    provider: OAuthProvider


# Game Models
class GameDatasets(BaseModel):
    """Available datasets for the game."""
    datasets: Dict[str, Dict[str, Any]]


class GameConfiguration(BaseModel):
    dataset: str
    difficulty: DifficultyLevel
    ai_model_key: str
    total_rounds: int = 10


class GameSessionResponse(BaseModel):
    session_id: str
    dataset: str
    difficulty: str
    ai_model_key: str
    total_rounds: int
    current_round: int
    status: str
    correct_answers: int
    total_score: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class GameRoundResponse(BaseModel):
    round_id: str
    round_number: int
    image_url: str
    options: List[str]
    ai_commitment_hash: str  # Cryptographic commitment to AI's answer
    
    class Config:
        from_attributes = True


class GameRoundSubmission(BaseModel):
    round_id: str
    user_answer: str
    response_time_ms: Optional[int] = None


class GameRoundResult(BaseModel):
    round_id: str
    user_answer: str
    correct_answer: str
    user_correct: bool
    ai_answer: str
    ai_confidence: float
    ai_commitment_proof: str  # Salt to prove AI's pre-commitment
    points_earned: int
    explanation: str
    
    class Config:
        from_attributes = True


class GameSessionSummary(BaseModel):
    session_id: str
    final_score: int
    correct_answers: int
    total_rounds: int
    accuracy: float
    ai_correct_answers: int
    ai_accuracy: float
    beat_ai: bool
    average_response_time_ms: Optional[int]
    completed_at: datetime
    
    class Config:
        from_attributes = True


# Leaderboard Models
class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    display_name: str
    avatar_url: Optional[str]
    score: int
    accuracy: float
    games_played: int
    beat_ai_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class LeaderboardResponse(BaseModel):
    dataset: str
    difficulty: str
    period: str  # "daily", "weekly", "monthly", "all_time"
    entries: List[LeaderboardEntry]
    user_rank: Optional[int] = None
    total_players: int


# Legacy Models (for backward compatibility)
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


# Deprecated - keeping for backward compatibility
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