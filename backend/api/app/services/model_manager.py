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
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from PIL import Image
import time

# Add paths to import from models
# In Docker, models are at /app/models
if Path("/app/models").exists():
    models_path = Path("/app/models")
    project_root_for_imports = Path("/app")
else:
    # Fallback for local development
    project_root_for_imports = Path(__file__).parent.parent.parent.parent.parent
    models_path = project_root_for_imports / "models"
sys.path.insert(0, str(models_path))

# Also add the current directory to path for relative imports
sys.path.insert(0, str(project_root_for_imports))

logger = logging.getLogger(__name__)

# Import model classifiers individually
ShallowImageClassifier = None
DeepLearningV1Classifier = None
DeepLearningV2Classifier = None
TransferLearningClassifier = None
PyTorchTransferLearningClassifier = None

try:
    from classifiers.shallow.src.classifier import ShallowImageClassifier
    logger.info("✅ ShallowImageClassifier imported successfully")
except ImportError as e:
    logger.warning(f"❌ Failed to import ShallowImageClassifier: {e}")

try:
    # Handle deep-v1 with package-style imports
    import importlib
    import importlib.util
    
    # Temporarily add the classifiers path
    classifiers_path = models_path / "classifiers"
    if str(classifiers_path) not in sys.path:
        sys.path.insert(0, str(classifiers_path))
    
    # Create fake package structure for deep-v1
    if "deep_v1" not in sys.modules:
        deep_v1_package = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("deep_v1", loader=None, is_package=True)
        )
        sys.modules["deep_v1"] = deep_v1_package
        
        # Load src submodule
        src_spec = importlib.util.spec_from_loader("deep_v1.src", loader=None, is_package=True)
        src_module = importlib.util.module_from_spec(src_spec)
        sys.modules["deep_v1.src"] = src_module
        
        # Load the actual modules
        deep_v1_path = models_path / "classifiers" / "deep-v1" / "src"
        logger.info(f"Looking for deep-v1 modules at: {deep_v1_path}")
        
        # Load config
        config_spec = importlib.util.spec_from_file_location(
            "deep_v1.src.config", deep_v1_path / "config.py"
        )
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules["deep_v1.src.config"] = config_module
        config_spec.loader.exec_module(config_module)
        
        # Load model
        model_spec = importlib.util.spec_from_file_location(
            "deep_v1.src.model", deep_v1_path / "model.py"
        )
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules["deep_v1.src.model"] = model_module
        model_spec.loader.exec_module(model_module)
        
        # Load classifier
        classifier_spec = importlib.util.spec_from_file_location(
            "deep_v1.src.classifier", deep_v1_path / "classifier.py"
        )
        classifier_module = importlib.util.module_from_spec(classifier_spec)
        sys.modules["deep_v1.src.classifier"] = classifier_module
        classifier_spec.loader.exec_module(classifier_module)
        
        DeepLearningV1Classifier = classifier_module.DeepLearningV1Classifier
    
    logger.info("✅ DeepLearningV1Classifier imported successfully")
except Exception as e:
    logger.warning(f"❌ Failed to import DeepLearningV1Classifier: {e}")

try:
    # Handle deep-v2 with package-style imports
    if "deep_v2" not in sys.modules:
        deep_v2_package = importlib.util.module_from_spec(
            importlib.util.spec_from_loader("deep_v2", loader=None, is_package=True)
        )
        sys.modules["deep_v2"] = deep_v2_package
        
        # Load src submodule
        src_spec = importlib.util.spec_from_loader("deep_v2.src", loader=None, is_package=True)
        src_module = importlib.util.module_from_spec(src_spec)
        sys.modules["deep_v2.src"] = src_module
        
        # Load the actual modules
        deep_v2_path = models_path / "classifiers" / "deep-v2" / "src"
        logger.info(f"Looking for deep-v2 modules at: {deep_v2_path}")
        
        # Load config
        config_spec = importlib.util.spec_from_file_location(
            "deep_v2.src.config", deep_v2_path / "config.py"
        )
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules["deep_v2.src.config"] = config_module
        config_spec.loader.exec_module(config_module)
        
        # Load model
        model_spec = importlib.util.spec_from_file_location(
            "deep_v2.src.model", deep_v2_path / "model.py"
        )
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules["deep_v2.src.model"] = model_module
        model_spec.loader.exec_module(model_module)
        
        # Load classifier
        classifier_spec = importlib.util.spec_from_file_location(
            "deep_v2.src.classifier", deep_v2_path / "classifier.py"
        )
        classifier_module = importlib.util.module_from_spec(classifier_spec)
        sys.modules["deep_v2.src.classifier"] = classifier_module
        classifier_spec.loader.exec_module(classifier_module)
        
        DeepLearningV2Classifier = classifier_module.DeepLearningV2Classifier
    
    logger.info("✅ DeepLearningV2Classifier imported successfully")
except Exception as e:
    logger.warning(f"❌ Failed to import DeepLearningV2Classifier: {e}")

try:
    from classifiers.transfer.src.classifier import TransferLearningClassifier
    logger.info("✅ TransferLearningClassifier imported successfully")
except ImportError as e:
    logger.warning(f"❌ Failed to import TransferLearningClassifier: {e}")

try:
    from classifiers.transfer.src.pytorch_classifier import PyTorchTransferLearningClassifier
    logger.info("✅ PyTorchTransferLearningClassifier imported successfully")
except ImportError as e:
    logger.warning(f"❌ Failed to import PyTorchTransferLearningClassifier: {e}")
    PyTorchTransferLearningClassifier = None

from ..models import ModelInfo, ModelType


class ModelManager:
    """Manages loading and serving of ML models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        self._initialized = False
        if Path("/app/models").exists():
            self.project_root = Path("/app")
        else:
            self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.production_models_path = self.project_root / "models" / "production" / "models"
        self.game_config_path = self.project_root / "models" / "production" / "results"
        self._game_config = None
    
    async def initialize_models(self):
        """Initialize all models asynchronously."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing models...")
            
            # Load game configuration to see available models
            await self._load_game_config()
            
            if self._game_config:
                # Get model descriptions
                self.model_configs = self._game_config.get("model_descriptions", {})
                
                # For now, we'll lazy load models when requested
                # This avoids loading all models at startup
                logger.info(f"Found {len(self.model_configs)} model configurations")
            
            self._initialized = True
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def get_model(self, model_type: str) -> Optional[Any]:
        """Get a specific model by type."""
        if not self._initialized:
            await self.initialize_models()
        
        return self.models.get(model_type)
    
    async def get_model_by_key(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get a model by its full key (e.g., 'transfer/resnet50')."""
        if not self._initialized:
            await self.initialize_models()
        
        # Check if model is already loaded
        if model_key in self.models:
            return self.models[model_key]
        
        # Try to load the model
        model = await self._load_model(model_key)
        if model:
            self.models[model_key] = model
            return model
        
        return None
    
    async def get_models_info(self) -> List[ModelInfo]:
        """Get information about all available models."""
        if not self._initialized:
            await self.initialize_models()
        
        models_info = []
        
        # Get info from game config
        if self._game_config:
            for dataset_name, dataset_config in self._game_config.get("datasets", {}).items():
                for difficulty, diff_config in dataset_config.get("difficulty_levels", {}).items():
                    for model_data in diff_config.get("models", []):
                        model_key = model_data.get("model_key", "")
                        model_type = model_key.split("/")[0] if "/" in model_key else "unknown"
                        
                        try:
                            model_info = ModelInfo(
                                name=model_data.get("name", model_key),
                                version=model_data.get("version", "unknown"),
                                type=ModelType(model_type) if model_type in [t.value for t in ModelType] else ModelType.TRANSFER,
                                accuracy=model_data.get("accuracy", 0.0),
                                is_loaded=model_key in self.models
                            )
                            models_info.append(model_info)
                        except Exception as e:
                            logger.warning(f"Failed to create ModelInfo for {model_key}: {e}")
        
        # Remove duplicates
        seen = set()
        unique_models = []
        for model in models_info:
            key = f"{model.name}:{model.version}"
            if key not in seen:
                seen.add(key)
                unique_models.append(model)
        
        return unique_models
    
    async def reload_model(self, model_key: str) -> bool:
        """Reload a specific model."""
        try:
            if model_key in self.models:
                del self.models[model_key]
                logger.info(f"Unloaded model: {model_key}")
            
            model = await self._load_model(model_key)
            if model:
                self.models[model_key] = model
                logger.info(f"Reloaded model: {model_key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to reload model {model_key}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        self._initialized = False
        logger.info("Model manager cleaned up")
    
    def _convert_to_docker_path(self, host_path: str) -> str:
        """Convert host path to Docker container path."""
        host_path = str(host_path)
        
        # Common host path patterns that need to be converted
        patterns_to_replace = [
            ("/home/brandond/Projects/pvt/personal/eyeVsAI/models/production/models", "/app/models/production/models"),
            ("/home/brandond/Projects/pvt/personal/eyeVsAI/production_training/models", "/app/models/production/models"),
            ("/home/brandond/Projects/pvt/personal/eyeVsAI/models", "/app/models"),
            ("/home/brandond/Projects/pvt/personal/eyeVsAI", "/app")
        ]
        
        for host_pattern, docker_pattern in patterns_to_replace:
            if host_path.startswith(host_pattern):
                converted_path = host_path.replace(host_pattern, docker_pattern, 1)
                logger.info(f"Converted model path: {host_path} -> {converted_path}")
                return converted_path
        
        # If no pattern matches, assume it's already a Docker path or construct one
        if not host_path.startswith("/app"):
            logger.warning(f"Unable to convert path pattern: {host_path}")
            # Try to extract the relative part and prepend Docker base
            if "models/production/models" in host_path:
                relative_part = host_path.split("models/production/models", 1)[1]
                converted_path = f"/app/models/production/models{relative_part}"
                logger.info(f"Fallback conversion: {host_path} -> {converted_path}")
                return converted_path
        
        return host_path
    
    def _detect_model_format(self, model_path, model_data=None):
        """Detect model format from metadata or file extension with enterprise-grade priority."""
        if model_data and "model_format" in model_data:
            format_type = model_data["model_format"]
            logger.info(f"Using specified model format: {format_type}")
            return format_type
        
        # Fallback to file extension detection with enterprise priority
        model_path = Path(model_path)
        
        # If it's a directory, look for model files inside with priority order
        if model_path.is_dir():
            # Priority 1: TorchScript (.pt) - Production ready
            if (model_path / "model.pt").exists():
                return "torchscript"
            # Priority 2: ONNX (.onnx) - Cross-platform
            elif any(model_path.glob("*.onnx")):
                return "onnx"
            # Priority 3: PyTorch state dict (.pth) - Training compatibility
            elif (model_path / "model.pth").exists():
                return "pytorch"
            # Priority 4: TensorFlow formats
            elif any(model_path.glob("*.h5")) or any(model_path.glob("*.keras")):
                return "tensorflow"
            # Priority 5: sklearn
            elif any(model_path.glob("*.pkl")):
                return "sklearn"
        else:
            # Check file extension directly with priority
            if model_path.suffix == ".pt":
                return "torchscript"
            elif model_path.suffix == ".onnx":
                return "onnx"
            elif model_path.suffix == ".pth":
                return "pytorch"
            elif model_path.suffix in [".h5", ".keras"]:
                return "tensorflow"
            elif model_path.suffix == ".pkl":
                return "sklearn"
        
        logger.warning(f"Unknown model format for {model_path}")
        return "unknown"
    
    async def _load_game_config(self) -> bool:
        """Load game backend configuration."""
        try:
            # Find the latest game backend report
            report_files = list(self.game_config_path.glob("game_backend_report_*.json"))
            if not report_files:
                logger.warning("No game backend report found")
                return False
            
            latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                self._game_config = json.load(f)
            
            logger.info(f"Loaded game config from {latest_report}")
            return True
        except Exception as e:
            logger.error(f"Failed to load game config: {e}")
            return False
    
    async def _load_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Load a specific model."""
        try:
            # Parse model key (e.g., "transfer/resnet50")
            parts = model_key.split("/")
            if len(parts) != 2:
                logger.error(f"Invalid model key format: {model_key}")
                return None
            
            model_type, variation = parts
            
            # Find model path from game config
            model_path = None
            if self._game_config:
                for dataset_config in self._game_config.get("datasets", {}).values():
                    for diff_config in dataset_config.get("difficulty_levels", {}).values():
                        for model_data in diff_config.get("models", []):
                            if model_data.get("model_key") == model_key:
                                model_path = model_data.get("model_path")
                                break
            
            if not model_path:
                # Try to construct path
                model_path = self.production_models_path / model_type / variation
            else:
                # Convert host path to Docker container path
                model_path = self._convert_to_docker_path(model_path)
            
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return None
            
            # Create a wrapper that provides predict_image method
            classifier = await self._create_classifier(model_type, model_path)
            if not classifier:
                return None
            
            return {
                "classifier": classifier,
                "model_key": model_key,
                "model_type": model_type,
                "variation": variation,
                "predict_image": lambda image_path: self._predict_image(classifier, image_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return None
    
    async def _create_classifier(self, model_type: str, model_path: Path) -> Optional[Any]:
        """Create classifier instance based on model type with enterprise-grade format handling."""
        try:
            if model_type == "shallow" and ShallowImageClassifier:
                classifier = ShallowImageClassifier()
                classifier.load_model(str(model_path))
                return classifier
            elif model_type == "deep_v1" and DeepLearningV1Classifier:
                classifier = DeepLearningV1Classifier()
                classifier.load_model(str(model_path))
                return classifier
            elif model_type == "deep_v2" and DeepLearningV2Classifier:
                classifier = DeepLearningV2Classifier()
                classifier.load_model(str(model_path))
                return classifier
            elif model_type == "transfer":
                # Enterprise-grade format detection and fallback
                return await self._load_transfer_model_with_fallback(model_path)
            else:
                logger.error(f"Unknown or unavailable model type: {model_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create classifier for {model_type}: {e}")
            return None
    
    async def _load_transfer_model_with_fallback(self, model_path: Path) -> Optional[Any]:
        """Load transfer learning model with enterprise-grade format fallback."""
        model_format = self._detect_model_format(model_path)
        
        # Try formats in priority order with fallback
        fallback_formats = [
            ("torchscript", "model.pt"),
            ("onnx", "*.onnx"),
            ("pytorch", "model.pth"),
            ("tensorflow", "*.h5"),
            ("tensorflow", "*.keras")
        ]
        
        # Start with detected format, then try fallbacks
        formats_to_try = [model_format]
        for fmt, _ in fallback_formats:
            if fmt not in formats_to_try:
                formats_to_try.append(fmt)
        
        for format_type in formats_to_try:
            try:
                classifier = await self._create_transfer_classifier(model_path, format_type)
                if classifier:
                    logger.info(f"Successfully loaded transfer model with format: {format_type}")
                    return classifier
            except Exception as e:
                logger.warning(f"Failed to load transfer model with format {format_type}: {e}")
                continue
        
        logger.error(f"No suitable transfer learning classifier could be loaded from {model_path}")
        return None
    
    async def _create_transfer_classifier(self, model_path: Path, format_type: str) -> Optional[Any]:
        """Create transfer learning classifier for specific format."""
        if format_type == "torchscript":
            if not PyTorchTransferLearningClassifier:
                raise ImportError("PyTorchTransferLearningClassifier not available")
            
            # Look for TorchScript file
            torchscript_path = model_path / "model.pt" if model_path.is_dir() else model_path
            if not torchscript_path.exists():
                raise FileNotFoundError(f"TorchScript model not found: {torchscript_path}")
            
            classifier = PyTorchTransferLearningClassifier()
            classifier.load_model(str(torchscript_path))
            return classifier
            
        elif format_type == "onnx":
            if not PyTorchTransferLearningClassifier:
                raise ImportError("PyTorchTransferLearningClassifier not available for ONNX")
            
            # Look for ONNX file
            if model_path.is_dir():
                onnx_files = list(model_path.glob("*.onnx"))
                if not onnx_files:
                    raise FileNotFoundError(f"ONNX model not found in {model_path}")
                onnx_path = onnx_files[0]
            else:
                onnx_path = model_path
            
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
            # Note: ONNX loading would require additional implementation
            # For now, we'll skip ONNX and fall back to PyTorch
            raise NotImplementedError("ONNX loading not yet implemented")
            
        elif format_type == "pytorch":
            if not PyTorchTransferLearningClassifier:
                raise ImportError("PyTorchTransferLearningClassifier not available")
            
            # Look for PyTorch state dict
            pytorch_path = model_path / "model.pth" if model_path.is_dir() else model_path
            if not pytorch_path.exists():
                raise FileNotFoundError(f"PyTorch model not found: {pytorch_path}")
            
            classifier = PyTorchTransferLearningClassifier()
            classifier.load_model(str(pytorch_path))
            return classifier
            
        elif format_type == "tensorflow":
            if not TransferLearningClassifier:
                raise ImportError("TransferLearningClassifier not available")
            
            # Look for TensorFlow model files
            if model_path.is_dir():
                tf_files = list(model_path.glob("*.h5")) + list(model_path.glob("*.keras"))
                if not tf_files:
                    raise FileNotFoundError(f"TensorFlow model not found in {model_path}")
                tf_path = tf_files[0]
            else:
                tf_path = model_path
            
            if not tf_path.exists():
                raise FileNotFoundError(f"TensorFlow model not found: {tf_path}")
            
            classifier = TransferLearningClassifier()
            classifier.load_model(str(tf_path))
            return classifier
            
        else:
            raise ValueError(f"Unsupported model format: {format_type}")
    
    def validate_model_format(self, model_path: Path, expected_format: str = None) -> Dict[str, Any]:
        """Validate model format and return comprehensive validation results."""
        validation_result = {
            "path": str(model_path),
            "exists": model_path.exists(),
            "detected_format": "unknown",
            "available_formats": [],
            "metadata_files": [],
            "validation_errors": [],
            "recommendations": []
        }
        
        if not model_path.exists():
            validation_result["validation_errors"].append(f"Model path does not exist: {model_path}")
            return validation_result
        
        try:
            # Detect available formats
            if model_path.is_dir():
                # Check for different model formats in directory
                formats_found = []
                
                # TorchScript
                if (model_path / "model.pt").exists():
                    formats_found.append("torchscript")
                    if (model_path / "model.metadata.json").exists():
                        validation_result["metadata_files"].append("model.metadata.json")
                    if (model_path / "model.model_card.json").exists():
                        validation_result["metadata_files"].append("model.model_card.json")
                
                # ONNX
                onnx_files = list(model_path.glob("*.onnx"))
                if onnx_files:
                    formats_found.append("onnx")
                
                # PyTorch state dict
                if (model_path / "model.pth").exists():
                    formats_found.append("pytorch")
                
                # TensorFlow
                tf_files = list(model_path.glob("*.h5")) + list(model_path.glob("*.keras"))
                if tf_files:
                    formats_found.append("tensorflow")
                
                # sklearn
                pkl_files = list(model_path.glob("*.pkl"))
                if pkl_files:
                    formats_found.append("sklearn")
                
                validation_result["available_formats"] = formats_found
                
                # Determine primary format
                if "torchscript" in formats_found:
                    validation_result["detected_format"] = "torchscript"
                elif "onnx" in formats_found:
                    validation_result["detected_format"] = "onnx"
                elif "pytorch" in formats_found:
                    validation_result["detected_format"] = "pytorch"
                elif "tensorflow" in formats_found:
                    validation_result["detected_format"] = "tensorflow"
                elif "sklearn" in formats_found:
                    validation_result["detected_format"] = "sklearn"
            
            else:
                # Single file - detect by extension
                if model_path.suffix == ".pt":
                    validation_result["detected_format"] = "torchscript"
                    validation_result["available_formats"] = ["torchscript"]
                elif model_path.suffix == ".onnx":
                    validation_result["detected_format"] = "onnx"
                    validation_result["available_formats"] = ["onnx"]
                elif model_path.suffix == ".pth":
                    validation_result["detected_format"] = "pytorch"
                    validation_result["available_formats"] = ["pytorch"]
                elif model_path.suffix in [".h5", ".keras"]:
                    validation_result["detected_format"] = "tensorflow"
                    validation_result["available_formats"] = ["tensorflow"]
                elif model_path.suffix == ".pkl":
                    validation_result["detected_format"] = "sklearn"
                    validation_result["available_formats"] = ["sklearn"]
            
            # Validate against expected format
            if expected_format and validation_result["detected_format"] != expected_format:
                if expected_format in validation_result["available_formats"]:
                    validation_result["validation_errors"].append(
                        f"Expected format '{expected_format}' is available but not primary format"
                    )
                else:
                    validation_result["validation_errors"].append(
                        f"Expected format '{expected_format}' not found"
                    )
            
            # Generate recommendations
            if "torchscript" not in validation_result["available_formats"] and "pytorch" in validation_result["available_formats"]:
                validation_result["recommendations"].append(
                    "Consider exporting PyTorch model to TorchScript for production deployment"
                )
            
            if not validation_result["metadata_files"]:
                validation_result["recommendations"].append(
                    "Consider adding metadata files (*.metadata.json, *.model_card.json) for better model management"
                )
            
            if len(validation_result["available_formats"]) == 1:
                validation_result["recommendations"].append(
                    "Consider exporting model in multiple formats for better compatibility"
                )
            
        except Exception as e:
            validation_result["validation_errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _predict_image(self, classifier: Any, image_path: str) -> Dict[str, Any]:
        """Make prediction on an image."""
        try:
            start_time = time.time()
            
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Make prediction
            result = classifier.predict(image)
            
            # Extract top prediction
            if isinstance(result, dict) and "predictions" in result:
                predictions = result["predictions"]
                top_class = max(predictions, key=predictions.get)
                confidence = predictions[top_class]
            else:
                # Handle different result formats
                top_class = str(result)
                confidence = 1.0
            
            processing_time = time.time() - start_time
            
            return {
                "prediction": top_class,
                "confidence": float(confidence),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to predict image: {e}")
            # Return a random prediction as fallback
            return {
                "prediction": "unknown",
                "confidence": 0.5,
                "processing_time": 0.0
            }