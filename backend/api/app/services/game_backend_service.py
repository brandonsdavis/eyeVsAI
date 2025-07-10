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

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..models import GameConfiguration

logger = logging.getLogger(__name__)


class GameBackendService:
    """Service for managing game backend operations."""
    
    def __init__(self):
        # Use Docker-compatible paths
        if Path("/app/models").exists():
            self.project_root = Path("/app")
            logger.info(f"GameBackendService: Using Docker path - project_root: {self.project_root}")
        else:
            # Fallback for local development
            self.project_root = Path(__file__).parent.parent.parent.parent.parent
            logger.info(f"GameBackendService: Using local path - project_root: {self.project_root}")
        
        self.game_report_path = self.project_root / "models" / "production" / "results"
        self.data_path = self.project_root / "models" / "data" / "downloads"
        logger.info(f"GameBackendService: game_report_path: {self.game_report_path}")
        logger.info(f"GameBackendService: path exists: {self.game_report_path.exists()}")
        self._game_config = None
        self._datasets_config = None
        self._used_images = {}  # Track used images per session
    
    async def get_available_datasets(self) -> Dict[str, Any]:
        """Get available datasets from game backend report."""
        try:
            config = await self._load_game_config()
            if not config:
                return {}
            
            return config.get("datasets", {})
        except Exception as e:
            logger.error(f"Failed to load game datasets: {e}")
            return {}
    
    async def validate_game_configuration(self, config: GameConfiguration) -> bool:
        """Validate game configuration against available datasets and models."""
        try:
            game_config = await self._load_game_config()
            if not game_config:
                return False
            
            # Check if dataset exists
            if config.dataset not in game_config.get("datasets", {}):
                return False
            
            dataset_config = game_config["datasets"][config.dataset]
            
            # Check if difficulty exists
            if config.difficulty not in dataset_config.get("difficulty_levels", {}):
                return False
            
            # Check if AI model exists in difficulty level
            difficulty_config = dataset_config["difficulty_levels"][config.difficulty]
            models = difficulty_config.get("models", [])
            
            model_keys = [model.get("model_key") for model in models]
            if config.ai_model_key not in model_keys:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate game configuration: {e}")
            return False
    
    async def get_random_image(self, dataset: str, exclude_used: bool = False, 
                             session_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a random image from the dataset."""
        try:
            logger.info(f"Getting random image for dataset: {dataset}")
            datasets_config = await self._load_datasets_config()
            if not datasets_config:
                logger.error("No datasets config loaded")
                return None
            
            # Map dataset names to config keys
            dataset_name_mapping = {
                "pets": "oxford_pets",
                "vegetables": "kaggle_vegetables",
                "street_foods": "street_foods",
                "instruments": "musical_instruments",
                "combined": "combined_unified_classification"
            }
            
            config_key = dataset_name_mapping.get(dataset)
            logger.info(f"Dataset mapping: {dataset} -> {config_key}")
            if not config_key or config_key not in datasets_config:
                logger.error(f"Config key not found: {config_key}, available keys: {list(datasets_config.keys())}")
                return None
            
            dataset_config = datasets_config[config_key]
            dataset_path = self.data_path / dataset_config.get("local_path", "")
            logger.info(f"Dataset path: {dataset_path}")
            
            if not dataset_path.exists():
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return None
            
            # Find all images in the dataset
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
            all_images = []
            
            # Check if dataset has train/validation structure
            train_path = dataset_path / "train"
            val_path = dataset_path / "validation"
            
            # Check if train directory exists and has content
            has_train_content = train_path.exists() and any(train_path.iterdir())
            
            if has_train_content:
                # Use train directory
                for class_dir in train_path.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        for img_file in class_dir.iterdir():
                            if img_file.suffix.lower() in image_extensions:
                                all_images.append({
                                    "image_path": str(img_file),
                                    "correct_answer": class_name
                                })
            else:
                # Use root directory structure
                for class_dir in dataset_path.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith('.') and class_dir.name not in ['train', 'validation', 'test']:
                        class_name = class_dir.name
                        for img_file in class_dir.iterdir():
                            if img_file.suffix.lower() in image_extensions:
                                all_images.append({
                                    "image_path": str(img_file),
                                    "correct_answer": class_name
                                })
            
            if not all_images:
                return None
            
            # Filter out used images if requested
            if exclude_used and session_id:
                used_images = self._used_images.get(session_id, set())
                available_images = [img for img in all_images if img["image_path"] not in used_images]
                
                if not available_images:
                    # Reset used images if we've used all of them
                    self._used_images[session_id] = set()
                    available_images = all_images
            else:
                available_images = all_images
            
            # Select random image
            selected_image = random.choice(available_images)
            
            # Track used image
            if session_id:
                if session_id not in self._used_images:
                    self._used_images[session_id] = set()
                self._used_images[session_id].add(selected_image["image_path"])
            
            return selected_image
            
        except Exception as e:
            logger.error(f"Failed to get random image: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def get_answer_options(self, dataset: str, correct_answer: str, 
                               num_options: int = 4) -> List[str]:
        """Get answer options for a question including the correct answer."""
        try:
            game_config = await self._load_game_config()
            if not game_config:
                return [correct_answer]
            
            dataset_config = game_config.get("datasets", {}).get(dataset, {})
            all_classes = dataset_config.get("class_names", [])
            
            if not all_classes:
                return [correct_answer]
            
            # Remove correct answer from options to choose from
            other_classes = [cls for cls in all_classes if cls != correct_answer]
            
            # Select random incorrect options
            num_incorrect = min(num_options - 1, len(other_classes))
            incorrect_options = random.sample(other_classes, num_incorrect)
            
            # Combine with correct answer
            options = [correct_answer] + incorrect_options
            
            # Shuffle to randomize position of correct answer
            random.shuffle(options)
            
            return options
            
        except Exception as e:
            logger.error(f"Failed to get answer options: {e}")
            return [correct_answer]
    
    async def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            game_config = await self._load_game_config()
            if not game_config:
                return None
            
            return game_config.get("model_descriptions", {}).get(model_key)
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    async def _load_game_config(self) -> Optional[Dict[str, Any]]:
        """Load game backend configuration from production results."""
        if self._game_config is not None:
            return self._game_config
        
        try:
            # Find the latest game backend report
            print(f"DEBUG: Looking for game backend reports in: {self.game_report_path}")
            print(f"DEBUG: Path exists: {self.game_report_path.exists()}")
            report_files = list(self.game_report_path.glob("game_backend_report_*.json"))
            print(f"DEBUG: Found {len(report_files)} report files: {report_files}")
            logger.info(f"Looking for game backend reports in: {self.game_report_path}")
            logger.info(f"Path exists: {self.game_report_path.exists()}")
            logger.info(f"Found {len(report_files)} report files: {report_files}")
            if not report_files:
                logger.warning("No game backend report found")
                return None
            
            # Get the most recent report
            latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                self._game_config = json.load(f)
            
            logger.info(f"Loaded game config from {latest_report}")
            return self._game_config
            
        except Exception as e:
            print(f"DEBUG: Exception in _load_game_config: {type(e).__name__}: {e}")
            logger.error(f"Failed to load game config: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _load_datasets_config(self) -> Optional[Dict[str, Any]]:
        """Load datasets configuration."""
        if self._datasets_config is not None:
            return self._datasets_config
        
        try:
            config_path = self.data_path / "dataset_configs.json"
            if not config_path.exists():
                logger.warning("Dataset configs not found")
                return None
            
            with open(config_path, 'r') as f:
                self._datasets_config = json.load(f)
            
            logger.info(f"Loaded datasets config from {config_path}")
            return self._datasets_config
            
        except Exception as e:
            logger.error(f"Failed to load datasets config: {e}")
            return None
    
    def clear_session_cache(self, session_id: str):
        """Clear cached data for a session."""
        if session_id in self._used_images:
            del self._used_images[session_id]
    
    async def refresh_config(self):
        """Refresh cached configuration."""
        self._game_config = None
        self._datasets_config = None
        await self._load_game_config()
        await self._load_datasets_config()