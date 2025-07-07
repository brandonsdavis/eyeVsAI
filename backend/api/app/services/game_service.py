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

import asyncio
import random
import uuid
from typing import Dict, List
import logging

from ..models import GameChallengeResponse, GameSubmissionResponse

logger = logging.getLogger(__name__)


class GameService:
    """Service for managing game challenges and scoring."""
    
    def __init__(self):
        self.active_challenges: Dict[str, dict] = {}
        self.challenge_images = {
            "dogs_cats": {
                "easy": [
                    {"url": "/static/images/golden_retriever.jpg", "answer": "Golden Retriever", "options": ["Golden Retriever", "Labrador", "Husky", "Beagle"]},
                    {"url": "/static/images/siamese_cat.jpg", "answer": "Siamese Cat", "options": ["Siamese Cat", "Persian Cat", "Maine Coon", "British Shorthair"]},
                ],
                "medium": [
                    {"url": "/static/images/border_collie.jpg", "answer": "Border Collie", "options": ["Border Collie", "Australian Shepherd", "Sheltie", "Collie"]},
                    {"url": "/static/images/ragdoll_cat.jpg", "answer": "Ragdoll Cat", "options": ["Ragdoll Cat", "Birman", "Himalayan", "Turkish Angora"]},
                ],
                "hard": [
                    {"url": "/static/images/australian_kelpie.jpg", "answer": "Australian Kelpie", "options": ["Australian Kelpie", "Australian Cattle Dog", "Kelpie Mix", "Working Kelpie"]},
                    {"url": "/static/images/oriental_shorthair.jpg", "answer": "Oriental Shorthair", "options": ["Oriental Shorthair", "Siamese", "Cornish Rex", "Devon Rex"]},
                ]
            },
            "fruits_vegetables": {
                "easy": [
                    {"url": "/static/images/apple.jpg", "answer": "Apple", "options": ["Apple", "Pear", "Peach", "Plum"]},
                    {"url": "/static/images/carrot.jpg", "answer": "Carrot", "options": ["Carrot", "Parsnip", "Sweet Potato", "Turnip"]},
                ],
                "medium": [
                    {"url": "/static/images/kiwi.jpg", "answer": "Kiwi", "options": ["Kiwi", "Passion Fruit", "Lime", "Avocado"]},
                    {"url": "/static/images/eggplant.jpg", "answer": "Eggplant", "options": ["Eggplant", "Purple Pepper", "Plum", "Purple Potato"]},
                ],
                "hard": [
                    {"url": "/static/images/dragon_fruit.jpg", "answer": "Dragon Fruit", "options": ["Dragon Fruit", "Passion Fruit", "Star Fruit", "Rambutan"]},
                    {"url": "/static/images/kohlrabi.jpg", "answer": "Kohlrabi", "options": ["Kohlrabi", "Turnip", "Rutabaga", "Radish"]},
                ]
            }
        }
    
    async def create_challenge(self, category: str, difficulty: str) -> GameChallengeResponse:
        """Create a new game challenge."""
        try:
            if category not in self.challenge_images:
                raise ValueError(f"Unknown category: {category}")
            
            if difficulty not in self.challenge_images[category]:
                raise ValueError(f"Unknown difficulty: {difficulty}")
            
            # Select random challenge from category/difficulty
            challenges = self.challenge_images[category][difficulty]
            challenge_data = random.choice(challenges)
            
            # Generate unique challenge ID
            challenge_id = str(uuid.uuid4())
            
            # Store challenge data
            self.active_challenges[challenge_id] = {
                "category": category,
                "difficulty": difficulty,
                "correct_answer": challenge_data["answer"],
                "image_url": challenge_data["url"],
                "options": challenge_data["options"]
            }
            
            return GameChallengeResponse(
                image_url=challenge_data["url"],
                correct_answer="",  # Don't reveal the answer
                options=challenge_data["options"],
                challenge_id=challenge_id
            )
            
        except Exception as e:
            logger.error(f"Failed to create challenge: {e}")
            raise
    
    async def submit_answer(self, challenge_id: str, user_answer: str) -> GameSubmissionResponse:
        """Submit an answer for a challenge."""
        try:
            if challenge_id not in self.active_challenges:
                raise ValueError(f"Challenge {challenge_id} not found")
            
            challenge = self.active_challenges[challenge_id]
            correct_answer = challenge["correct_answer"]
            is_correct = user_answer == correct_answer
            
            # Calculate score based on difficulty
            difficulty_multiplier = {"easy": 10, "medium": 20, "hard": 30}
            base_score = difficulty_multiplier.get(challenge["difficulty"], 10)
            user_score = base_score if is_correct else 0
            
            # TODO: Get actual model predictions for the challenge image
            # For now, create mock predictions
            mock_predictions = {
                correct_answer: 0.85,
                challenge["options"][1]: 0.10,
                challenge["options"][2]: 0.03,
                challenge["options"][3]: 0.02
            }
            
            # Generate explanation
            explanation = self._generate_explanation(
                correct_answer, 
                user_answer, 
                is_correct, 
                mock_predictions
            )
            
            # Cleanup challenge
            del self.active_challenges[challenge_id]
            
            return GameSubmissionResponse(
                correct=is_correct,
                correct_answer=correct_answer,
                model_predictions=mock_predictions,
                user_score=user_score,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Failed to submit answer: {e}")
            raise
    
    def _generate_explanation(self, correct_answer: str, user_answer: str, 
                            is_correct: bool, predictions: Dict[str, float]) -> str:
        """Generate explanation for the answer."""
        if is_correct:
            return f"Correct! The AI model was {predictions.get(correct_answer, 0.85):.0%} confident it was a {correct_answer}."
        else:
            model_confidence = predictions.get(correct_answer, 0.85)
            return f"Incorrect. You guessed '{user_answer}', but it was actually '{correct_answer}'. The AI model was {model_confidence:.0%} confident in the correct answer."