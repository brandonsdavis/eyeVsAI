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
import os
import sys
from typing import Dict, List, Optional
import logging

# Add paths to import from other modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-models-core', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ensemble-classifier', 'src'))

from base_classifier import BaseImageClassifier
from model_registry import ModelRegistry
from ensemble import EnsembleClassifier
from ..models import ModelInfo, ModelType

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and serving of ML models."""
    
    def __init__(self):
        self.models: Dict[str, BaseImageClassifier] = {}
        self.registry = ModelRegistry()
        self._initialized = False
    
    async def initialize_models(self):
        """Initialize all models asynchronously."""
        if self._initialized:
            return
        
        try:
            # TODO: Implement actual model loading
            # For now, create placeholder models
            logger.info("Initializing models...")
            
            # This would be replaced with actual model loading logic
            # self.models["shallow"] = ShallowClassifier()
            # self.models["deep_v1"] = DeepV1Classifier()
            # self.models["deep_v2"] = DeepV2Classifier()
            # self.models["transfer"] = TransferClassifier()
            # self.models["ensemble"] = EnsembleClassifier([...])
            
            self._initialized = True
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def get_model(self, model_type: str) -> Optional[BaseImageClassifier]:
        """Get a specific model by type."""
        if not self._initialized:
            await self.initialize_models()
        
        return self.models.get(model_type)
    
    async def get_models_info(self) -> List[ModelInfo]:
        """Get information about all loaded models."""
        if not self._initialized:
            await self.initialize_models()
        
        models_info = []
        for model_type, model in self.models.items():
            metadata = model.get_metadata()
            
            model_info = ModelInfo(
                name=model.model_name,
                version=model.version,
                type=ModelType(model_type),
                accuracy=metadata.get("accuracy", 0.0),
                is_loaded=model.is_loaded
            )
            models_info.append(model_info)
        
        return models_info
    
    async def reload_model(self, model_type: str) -> bool:
        """Reload a specific model."""
        try:
            if model_type in self.models:
                # TODO: Implement model reloading logic
                logger.info(f"Reloading model: {model_type}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reload model {model_type}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        self._initialized = False
        logger.info("Model manager cleaned up")