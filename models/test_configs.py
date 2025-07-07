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
Unit tests for configuration classes across all models.

This script tests the configuration dataclasses to ensure they:
- Instantiate correctly with default values
- Support to_dict() conversion
- Maintain proper inheritance hierarchy
- Have valid parameter ranges
"""

import unittest
import sys
from pathlib import Path
from dataclasses import is_dataclass

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class TestConfigurations(unittest.TestCase):
    """Test configuration classes from all models."""
    
    def test_shallow_learning_config(self):
        """Test ShallowLearningConfig class."""
        try:
            sys.path.append(str(project_root / "image-classifier-shallow"))
            from src.config import ShallowLearningConfig
            
            # Test instantiation
            config = ShallowLearningConfig()
            
            # Test it's a dataclass
            self.assertTrue(is_dataclass(config))
            
            # Test default values
            self.assertIsInstance(config.feature_types, list)
            self.assertGreater(len(config.feature_types), 0)
            self.assertIn('hog', config.feature_types)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            self.assertIn('feature_types', config_dict)
            self.assertIn('max_iter', config_dict)
            
            # Test parameter validation
            self.assertGreater(config.max_iter, 0)
            self.assertGreater(config.test_size, 0)
            self.assertLess(config.test_size, 1)
            
            print("✅ ShallowLearningConfig: All tests passed")
            
        except ImportError as e:
            self.skipTest(f"ShallowLearningConfig not available: {e}")
    
    def test_deep_learning_v1_config(self):
        """Test DeepLearningV1Config class."""
        try:
            sys.path.append(str(project_root / "image-classifier-deep-v1"))
            from src.config import DeepLearningV1Config
            
            # Test instantiation
            config = DeepLearningV1Config()
            
            # Test it's a dataclass
            self.assertTrue(is_dataclass(config))
            
            # Test default values
            self.assertIsInstance(config.image_size, tuple)
            self.assertEqual(len(config.image_size), 2)
            self.assertGreater(config.batch_size, 0)
            self.assertGreater(config.learning_rate, 0)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            self.assertIn('image_size', config_dict)
            self.assertIn('batch_size', config_dict)
            self.assertIn('learning_rate', config_dict)
            
            # Test parameter validation
            self.assertGreater(config.num_epochs, 0)
            self.assertGreaterEqual(config.patience, 1)
            
            print("✅ DeepLearningV1Config: All tests passed")
            
        except ImportError as e:
            self.skipTest(f"DeepLearningV1Config not available: {e}")
    
    def test_deep_learning_v2_config(self):
        """Test DeepLearningV2Config class."""
        try:
            sys.path.append(str(project_root / "image-classifier-deep-v2"))
            from src.config import DeepLearningV2Config
            
            # Test instantiation
            config = DeepLearningV2Config()
            
            # Test it's a dataclass
            self.assertTrue(is_dataclass(config))
            
            # Test advanced features
            self.assertIsInstance(config.mixup_alpha, float)
            self.assertIsInstance(config.accumulation_steps, int)
            self.assertIsInstance(config.memory_efficient, bool)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIn('mixup_alpha', config_dict)
            self.assertIn('accumulation_steps', config_dict)
            self.assertIn('attention_reduction_ratio', config_dict)
            
            # Test parameter validation
            self.assertGreater(config.accumulation_steps, 0)
            self.assertGreaterEqual(config.mixup_alpha, 0)
            self.assertLessEqual(config.mixup_prob, 1)
            
            print("✅ DeepLearningV2Config: All tests passed")
            
        except ImportError as e:
            self.skipTest(f"DeepLearningV2Config not available: {e}")
    
    def test_transfer_learning_config(self):
        """Test TransferLearningClassifierConfig class."""
        try:
            sys.path.append(str(project_root / "image-classifier-transfer"))
            from src.config import TransferLearningClassifierConfig
            
            # Test instantiation
            config = TransferLearningClassifierConfig()
            
            # Test it's a dataclass
            self.assertTrue(is_dataclass(config))
            
            # Test TensorFlow specific settings
            self.assertIsInstance(config.mixed_precision, bool)
            self.assertIsInstance(config.base_model_name, str)
            self.assertIsInstance(config.fine_tune_layers, int)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIn('base_model_name', config_dict)
            self.assertIn('mixed_precision', config_dict)
            self.assertIn('fine_tune_layers', config_dict)
            
            # Test parameter validation
            self.assertIn(config.base_model_name, ['resnet50', 'vgg16', 'efficientnet_b0'])
            self.assertGreaterEqual(config.fine_tune_layers, 0)
            self.assertGreater(config.fine_tune_learning_rate, 0)
            
            print("✅ TransferLearningClassifierConfig: All tests passed")
            
        except ImportError as e:
            self.skipTest(f"TransferLearningClassifierConfig not available: {e}")
    
    def test_config_inheritance(self):
        """Test configuration inheritance from base classes."""
        configs_tested = 0
        
        # Test shallow learning config
        try:
            sys.path.append(str(project_root / "image-classifier-shallow"))
            from src.config import ShallowLearningConfig
            
            config = ShallowLearningConfig()
            
            # Test base functionality
            self.assertTrue(hasattr(config, 'to_dict'))
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            
            configs_tested += 1
            
        except ImportError:
            pass
        
        # Test deep learning configs
        for model_name in ['image-classifier-deep-v1', 'image-classifier-deep-v2']:
            try:
                sys.path.append(str(project_root / model_name))
                if model_name == 'image-classifier-deep-v1':
                    from src.config import DeepLearningV1Config as ConfigClass
                else:
                    from src.config import DeepLearningV2Config as ConfigClass
                
                config = ConfigClass()
                
                # Test base functionality
                self.assertTrue(hasattr(config, 'to_dict'))
                config_dict = config.to_dict()
                self.assertIsInstance(config_dict, dict)
                
                # Test common deep learning attributes
                self.assertTrue(hasattr(config, 'image_size'))
                self.assertTrue(hasattr(config, 'batch_size'))
                self.assertTrue(hasattr(config, 'learning_rate'))
                
                configs_tested += 1
                
            except ImportError:
                pass
        
        # Test transfer learning config
        try:
            sys.path.append(str(project_root / "image-classifier-transfer"))
            from src.config import TransferLearningClassifierConfig
            
            config = TransferLearningClassifierConfig()
            
            # Test base functionality
            self.assertTrue(hasattr(config, 'to_dict'))
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            
            configs_tested += 1
            
        except ImportError:
            pass
        
        print(f"✅ Configuration inheritance: Tested {configs_tested} config classes")
        
        # Ensure we tested at least one config
        self.assertGreater(configs_tested, 0, "No configuration classes could be tested")
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        import json
        
        configs_tested = 0
        
        # Test each available config
        config_paths = [
            ("image-classifier-shallow", "ShallowLearningConfig"),
            ("image-classifier-deep-v1", "DeepLearningV1Config"),
            ("image-classifier-deep-v2", "DeepLearningV2Config"),
            ("image-classifier-transfer", "TransferLearningClassifierConfig"),
        ]
        
        for model_path, config_name in config_paths:
            try:
                sys.path.append(str(project_root / model_path))
                module = __import__('src.config', fromlist=[config_name])
                ConfigClass = getattr(module, config_name)
                
                # Create config
                config = ConfigClass()
                
                # Test serialization
                config_dict = config.to_dict()
                json_str = json.dumps(config_dict)
                
                # Test deserialization
                loaded_dict = json.loads(json_str)
                self.assertEqual(config_dict, loaded_dict)
                
                configs_tested += 1
                
            except (ImportError, AttributeError):
                pass
        
        print(f"✅ Configuration serialization: Tested {configs_tested} config classes")
        self.assertGreater(configs_tested, 0, "No configuration classes could be tested for serialization")


def main():
    """Run configuration tests."""
    print("Running Configuration Tests")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("Configuration Tests Summary:")
    print("- Tests configuration dataclasses")
    print("- Validates default parameters")
    print("- Checks serialization support")
    print("- Verifies inheritance structure")


if __name__ == '__main__':
    main()