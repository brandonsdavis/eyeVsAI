#!/usr/bin/env python
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

"""
Standalone ONNX export utility for trained models.
This script can convert PyTorch models to ONNX format for production deployment.
"""

import argparse
import torch
import torch.onnx
import json
import sys
from pathlib import Path
import logging
import importlib.util


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model_from_config(model_path: Path, config_path: Path, logger):
    """Load model based on its configuration."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_type = config.get('model_type')
    
    if model_type == 'transfer':
        return load_transfer_model(model_path, config, logger)
    elif model_type == 'deep_v1':
        return load_deep_v1_model(model_path, config, logger)
    elif model_type == 'deep_v2':
        return load_deep_v2_model(model_path, config, logger)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_transfer_model(model_path: Path, config: dict, logger):
    """Load transfer learning model."""
    try:
        # Add path to transfer learning modules
        transfer_src = Path(__file__).parent.parent.parent / "image-classifier-transfer" / "src"
        sys.path.insert(0, str(transfer_src))
        
        # Dynamic import to avoid relative import issues
        models_spec = importlib.util.spec_from_file_location(
            "models_pytorch", transfer_src / "models_pytorch.py"
        )
        models_module = importlib.util.module_from_spec(models_spec)
        models_spec.loader.exec_module(models_module)
        
        config_spec = importlib.util.spec_from_file_location(
            "config", transfer_src / "config.py"
        )
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        
        # Create model
        model_config = config_module.TransferLearningClassifierConfig(
            base_model=config.get("base_model", "resnet50"),
            head_dropout_rate=config.get("head_dropout_rate", 0.5),
            dense_units=config.get("dense_units", [512, 256])
        )
        
        model = models_module.TransferLearningModelPyTorch(model_config, config["num_classes"])
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded transfer learning model: {config.get('base_model', 'resnet50')}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load transfer model: {e}")
        raise


def load_deep_v1_model(model_path: Path, config: dict, logger):
    """Load Deep Learning V1 model."""
    try:
        # Add path to deep v1 modules
        deep_v1_src = Path(__file__).parent.parent.parent / "image-classifier-deep-v1" / "src"
        sys.path.insert(0, str(deep_v1_src))
        
        # Dynamic import
        model_spec = importlib.util.spec_from_file_location(
            "model_improved", deep_v1_src / "model_improved.py"
        )
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        
        # Create model
        model = model_module.DeepLearningV1Improved(
            num_classes=config["num_classes"],
            use_residual=config.get("use_residual", True)
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded Deep Learning V1 model")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Deep V1 model: {e}")
        raise


def load_deep_v2_model(model_path: Path, config: dict, logger):
    """Load Deep Learning V2 model."""
    try:
        # Add path to deep v2 modules
        deep_v2_src = Path(__file__).parent.parent.parent / "image-classifier-deep-v2" / "src"
        sys.path.insert(0, str(deep_v2_src))
        
        # Dynamic import
        model_spec = importlib.util.spec_from_file_location(
            "model_improved", deep_v2_src / "model_improved.py"
        )
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        
        # Create model
        model = model_module.DeepLearningV2Improved(
            num_classes=config["num_classes"],
            architecture=config.get("architecture", "resnet")
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info(f"Loaded Deep Learning V2 model: {config.get('architecture', 'resnet')}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Deep V2 model: {e}")
        raise


def export_to_onnx(model, onnx_path: Path, input_shape=(1, 3, 128, 128), logger=None):
    """Export PyTorch model to ONNX format."""
    try:
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        if logger:
            logger.info(f"Successfully exported model to ONNX: {onnx_path}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to export to ONNX: {e}")
        return False


def verify_onnx_model(onnx_path: Path, logger=None):
    """Verify the exported ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # Get input shape
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        
        # Create dummy input (replace None/dynamic dimensions with 1)
        test_shape = [1 if dim is None else dim for dim in input_shape]
        dummy_input = torch.randn(*test_shape).numpy()
        
        # Run inference
        outputs = ort_session.run(None, {input_name: dummy_input})
        
        if logger:
            logger.info(f"ONNX model verification successful")
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"ONNX model verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX format')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to PyTorch model (.pth file)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration (config.json)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for ONNX model')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the exported ONNX model')
    parser.add_argument('--input_size', type=int, nargs=2, default=[128, 128],
                       help='Input image size (height width)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    model_path = Path(args.model_path)
    config_path = Path(args.config_path)
    output_path = Path(args.output_path)
    
    # Validate input files
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = load_model_from_config(model_path, config_path, logger)
        
        # Export to ONNX
        input_shape = (1, 3, args.input_size[0], args.input_size[1])
        logger.info(f"Exporting to ONNX with input shape: {input_shape}")
        
        success = export_to_onnx(model, output_path, input_shape, logger)
        
        if success and args.verify:
            logger.info("Verifying exported ONNX model...")
            verify_onnx_model(output_path, logger)
        
        if success:
            logger.info("ONNX export completed successfully!")
            return True
        else:
            logger.error("ONNX export failed!")
            return False
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False


def export_all_models_in_directory(models_dir: Path, logger=None):
    """Export all PyTorch models in a directory to ONNX format."""
    if not logger:
        logger = setup_logging()
    
    # Find all model files
    model_files = list(models_dir.rglob("model.pth"))
    
    logger.info(f"Found {len(model_files)} models to export")
    
    success_count = 0
    for model_path in model_files:
        try:
            # Look for config file in the same directory
            config_path = model_path.parent / "config.json"
            if not config_path.exists():
                logger.warning(f"Config not found for {model_path}, skipping")
                continue
            
            # Output ONNX path
            onnx_path = model_path.with_suffix('.onnx')
            
            logger.info(f"Exporting {model_path.parent.name}")
            
            # Load and export
            model = load_model_from_config(model_path, config_path, logger)
            success = export_to_onnx(model, onnx_path, logger=logger)
            
            if success:
                success_count += 1
                logger.info(f"✓ Exported: {onnx_path}")
            else:
                logger.error(f"✗ Failed: {model_path}")
                
        except Exception as e:
            logger.error(f"Error exporting {model_path}: {e}")
    
    logger.info(f"ONNX export completed: {success_count}/{len(model_files)} successful")
    return success_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)