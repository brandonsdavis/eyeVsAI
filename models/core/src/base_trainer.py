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

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime


class BaseTrainer(ABC):
    """Base class for all model trainers."""
    
    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.training_history = []
        
    @abstractmethod
    def train(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            data_path: Path to the training data
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and results
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int = None, phase: str = "train") -> None:
        """Log training metrics."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "phase": phase,
            "epoch": epoch,
            "metrics": metrics
        }
        self.training_history.append(log_entry)
        
        # Log to console
        if epoch is not None:
            self.logger.info(f"Epoch {epoch} [{phase}]: {metrics}")
        else:
            self.logger.info(f"[{phase}]: {metrics}")
    
    def save_training_history(self, save_path: str) -> None:
        """Save training history to JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {save_path}")
    
    def discover_classes(self, data_path: str) -> List[str]:
        """
        Discover class names from the data directory structure.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            List of class names
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Look for subdirectories as class names
        class_dirs = [d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        class_names = sorted([d.name for d in class_dirs])
        
        if not class_names:
            raise ValueError(f"No class directories found in {data_path}")
        
        self.logger.info(f"Discovered {len(class_names)} classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
        return class_names