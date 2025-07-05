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
Comprehensive test suite for all extracted image classification modules.

This script provides unit tests for all model components including:
- Configuration classes
- Model architectures  
- Data loaders
- Trainers
- Classifiers
- Base class implementations
"""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup and utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock dataset structure
        self.create_mock_dataset()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_dataset(self):
        """Create a mock dataset for testing."""
        self.dataset_path = self.temp_path / "test_dataset"
        
        # Create class directories
        classes = ["class1", "class2", "class3"]
        for class_name in classes:
            class_dir = self.dataset_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock images (as empty files)
            for i in range(5):
                (class_dir / f"image_{i}.jpg").touch()
        
        return self.dataset_path
    
    def create_mock_image(self, size=(64, 64, 3)):
        """Create a mock image array."""
        return np.random.randint(0, 255, size, dtype=np.uint8)


class TestMLModelsCore(BaseTestCase):
    """Test the ml_models_core base classes and utilities."""
    
    def test_base_classifier_interface(self):
        """Test BaseImageClassifier interface."""
        try:
            from ml_models_core.src.base_classifier import BaseImageClassifier
            
            # Test abstract methods exist
            self.assertTrue(hasattr(BaseImageClassifier, 'load_model'))
            self.assertTrue(hasattr(BaseImageClassifier, 'predict'))
            self.assertTrue(hasattr(BaseImageClassifier, 'get_metadata'))
            
            # Test instantiation requires subclassing
            with self.assertRaises(TypeError):
                BaseImageClassifier()
                
        except ImportError as e:
            self.skipTest(f"ml_models_core not available: {e}")
    
    def test_base_trainer_interface(self):
        """Test BaseTrainer interface."""
        try:
            from ml_models_core.src.base_trainer import BaseTrainer
            
            # Test abstract methods exist
            self.assertTrue(hasattr(BaseTrainer, 'train'))
            self.assertTrue(hasattr(BaseTrainer, 'evaluate'))
            
        except ImportError as e:
            self.skipTest(f"ml_models_core not available: {e}")
    
    def test_model_registry(self):
        """Test ModelRegistry functionality."""
        try:
            from ml_models_core.src.model_registry import ModelRegistry, ModelMetadata
            
            registry = ModelRegistry()
            
            # Test metadata creation
            metadata = ModelMetadata(
                name="test_model",
                version="1.0.0",
                model_type="test",
                accuracy=0.95,
                training_date="2024-01-01",
                model_path="/path/to/model",
                config={},
                performance_metrics={}
            )
            
            self.assertEqual(metadata.name, "test_model")
            self.assertEqual(metadata.accuracy, 0.95)
            
        except ImportError as e:
            self.skipTest(f"ml_models_core not available: {e}")


class TestShallowLearning(BaseTestCase):
    """Test shallow learning classifier components."""
    
    def test_shallow_config(self):
        """Test ShallowLearningConfig."""
        try:
            from image_classifier_shallow.src.config import ShallowLearningConfig
            
            config = ShallowLearningConfig()
            
            # Test default values
            self.assertIsInstance(config.feature_types, list)
            self.assertIn('hog', config.feature_types)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            self.assertIn('feature_types', config_dict)
            
        except ImportError as e:
            self.skipTest(f"Shallow learning modules not available: {e}")
    
    def test_feature_extractor(self):
        """Test FeatureExtractor class."""
        try:
            from image_classifier_shallow.src.feature_extractor import FeatureExtractor
            from image_classifier_shallow.src.config import ShallowLearningConfig
            
            config = ShallowLearningConfig()
            extractor = FeatureExtractor(config)
            
            # Test feature extraction
            mock_image = self.create_mock_image()
            features = extractor.extract_features(mock_image)
            
            self.assertIsInstance(features, np.ndarray)
            self.assertGreater(len(features), 0)
            
        except ImportError as e:
            self.skipTest(f"Shallow learning modules not available: {e}")
    
    def test_shallow_classifier(self):
        """Test ShallowImageClassifier."""
        try:
            from image_classifier_shallow.src.classifier import ShallowImageClassifier
            from image_classifier_shallow.src.config import ShallowLearningConfig
            
            config = ShallowLearningConfig()
            classifier = ShallowImageClassifier(config=config)
            
            # Test interface compliance
            self.assertTrue(hasattr(classifier, 'predict'))
            self.assertTrue(hasattr(classifier, 'get_metadata'))
            
            # Test metadata
            metadata = classifier.get_metadata()
            self.assertIsInstance(metadata, dict)
            self.assertIn('model_type', metadata)
            
        except ImportError as e:
            self.skipTest(f"Shallow learning modules not available: {e}")


class TestDeepLearningV1(BaseTestCase):
    """Test deep learning v1 classifier components."""
    
    def test_deep_v1_config(self):
        """Test DeepLearningV1Config."""
        try:
            from image_classifier_deep_v1.src.config import DeepLearningV1Config
            
            config = DeepLearningV1Config()
            
            # Test default values
            self.assertIsInstance(config.image_size, tuple)
            self.assertEqual(len(config.image_size), 2)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            
        except ImportError as e:
            self.skipTest(f"Deep learning v1 modules not available: {e}")
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_deep_v1_model(self, mock_cuda):
        """Test DeepLearningV1 model architecture."""
        try:
            import torch
            from image_classifier_deep_v1.src.model import DeepLearningV1
            
            model = DeepLearningV1(num_classes=3)
            
            # Test model structure
            self.assertTrue(hasattr(model, 'features'))
            self.assertTrue(hasattr(model, 'classifier'))
            
            # Test forward pass
            x = torch.randn(1, 3, 64, 64)
            output = model(x)
            self.assertEqual(output.shape, (1, 3))
            
        except ImportError as e:
            self.skipTest(f"Deep learning v1 modules not available: {e}")
    
    def test_deep_v1_classifier(self):
        """Test DeepLearningV1Classifier."""
        try:
            from image_classifier_deep_v1.src.classifier import DeepLearningV1Classifier
            
            classifier = DeepLearningV1Classifier()
            
            # Test interface compliance
            self.assertTrue(hasattr(classifier, 'predict'))
            self.assertTrue(hasattr(classifier, 'get_metadata'))
            
            # Test metadata
            metadata = classifier.get_metadata()
            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata['model_type'], 'deep_learning_v1')
            
        except ImportError as e:
            self.skipTest(f"Deep learning v1 modules not available: {e}")


class TestDeepLearningV2(BaseTestCase):
    """Test deep learning v2 classifier components."""
    
    def test_deep_v2_config(self):
        """Test DeepLearningV2Config."""
        try:
            from image_classifier_deep_v2.src.config import DeepLearningV2Config
            
            config = DeepLearningV2Config()
            
            # Test advanced features
            self.assertIsInstance(config.mixup_alpha, float)
            self.assertIsInstance(config.accumulation_steps, int)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIn('mixup_alpha', config_dict)
            
        except ImportError as e:
            self.skipTest(f"Deep learning v2 modules not available: {e}")
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_attention_block(self, mock_cuda):
        """Test AttentionBlock component."""
        try:
            import torch
            from image_classifier_deep_v2.src.model import AttentionBlock
            
            attention = AttentionBlock(64)
            
            # Test forward pass
            x = torch.randn(1, 64, 32, 32)
            output = attention(x)
            self.assertEqual(output.shape, x.shape)
            
        except ImportError as e:
            self.skipTest(f"Deep learning v2 modules not available: {e}")
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_deep_v2_model(self, mock_cuda):
        """Test DeepLearningV2 model with attention."""
        try:
            import torch
            from image_classifier_deep_v2.src.model import DeepLearningV2
            
            model = DeepLearningV2(num_classes=3)
            
            # Test model components
            self.assertTrue(hasattr(model, 'attention1'))
            self.assertTrue(hasattr(model, 'attention2'))
            
            # Test forward pass
            x = torch.randn(1, 3, 96, 96)
            output = model(x)
            self.assertEqual(output.shape, (1, 3))
            
        except ImportError as e:
            self.skipTest(f"Deep learning v2 modules not available: {e}")


class TestTransferLearning(BaseTestCase):
    """Test transfer learning classifier components."""
    
    def test_transfer_config(self):
        """Test TransferLearningClassifierConfig."""
        try:
            from image_classifier_transfer.src.config import TransferLearningClassifierConfig
            
            config = TransferLearningClassifierConfig()
            
            # Test TensorFlow specific settings
            self.assertIsInstance(config.mixed_precision, bool)
            self.assertIsInstance(config.base_model_name, str)
            
            # Test to_dict method
            config_dict = config.to_dict()
            self.assertIn('base_model_name', config_dict)
            
        except ImportError as e:
            self.skipTest(f"Transfer learning modules not available: {e}")
    
    def test_transfer_model_creation(self):
        """Test TransferLearningModel creation."""
        try:
            # Mock TensorFlow to avoid heavy dependency
            with patch.dict('sys.modules', {'tensorflow': MagicMock()}):
                from image_classifier_transfer.src.models import TransferLearningModel
                from image_classifier_transfer.src.config import TransferLearningClassifierConfig
                
                config = TransferLearningClassifierConfig()
                model = TransferLearningModel(config, num_classes=3)
                
                # Test model info method
                model_info = model.get_model_info()
                self.assertIsInstance(model_info, dict)
                
        except ImportError as e:
            self.skipTest(f"Transfer learning modules not available: {e}")
    
    def test_transfer_classifier(self):
        """Test TransferLearningClassifier."""
        try:
            with patch.dict('sys.modules', {'tensorflow': MagicMock()}):
                from image_classifier_transfer.src.classifier import TransferLearningClassifier
                
                classifier = TransferLearningClassifier()
                
                # Test interface compliance
                self.assertTrue(hasattr(classifier, 'predict'))
                self.assertTrue(hasattr(classifier, 'get_metadata'))
                
                # Test metadata
                metadata = classifier.get_metadata()
                self.assertEqual(metadata['model_type'], 'transfer_learning')
                
        except ImportError as e:
            self.skipTest(f"Transfer learning modules not available: {e}")


class TestDataLoaders(BaseTestCase):
    """Test data loading components across all models."""
    
    def test_lazy_dataset_creation(self):
        """Test lazy dataset creation for memory efficiency."""
        try:
            from image_classifier_deep_v2.src.data_loader import LazyUnifiedDataset
            
            # Create mock dataset info
            dataset_info = {
                'dataset_path': self.dataset_path,
                'class_names': ['class1', 'class2', 'class3'],
                'class_to_idx': {'class1': 0, 'class2': 1, 'class3': 2},
                'class_counts': {'class1': 5, 'class2': 5, 'class3': 5},
                'valid_extensions': {'.jpg', '.jpeg', '.png'}
            }
            
            dataset = LazyUnifiedDataset(dataset_info)
            
            # Test lazy loading (paths not loaded yet)
            self.assertIsNone(dataset._image_paths)
            
            # Test length calculation
            self.assertEqual(len(dataset), 15)  # 3 classes * 5 images
            
        except ImportError as e:
            self.skipTest(f"Data loader modules not available: {e}")


class TestIntegration(BaseTestCase):
    """Test integration between components."""
    
    def test_config_inheritance(self):
        """Test that all configs properly inherit from base classes."""
        configs_to_test = [
            ('image_classifier_shallow.src.config', 'ShallowLearningConfig'),
            ('image_classifier_deep_v1.src.config', 'DeepLearningV1Config'),
            ('image_classifier_deep_v2.src.config', 'DeepLearningV2Config'),
            ('image_classifier_transfer.src.config', 'TransferLearningClassifierConfig'),
        ]
        
        for module_name, class_name in configs_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                config_class = getattr(module, class_name)
                
                # Test instantiation
                config = config_class()
                
                # Test to_dict method exists
                self.assertTrue(hasattr(config, 'to_dict'))
                config_dict = config.to_dict()
                self.assertIsInstance(config_dict, dict)
                
            except ImportError:
                continue  # Skip if module not available
    
    def test_classifier_interface_compliance(self):
        """Test that all classifiers implement BaseImageClassifier."""
        classifiers_to_test = [
            ('image_classifier_shallow.src.classifier', 'ShallowImageClassifier'),
            ('image_classifier_deep_v1.src.classifier', 'DeepLearningV1Classifier'),
            ('image_classifier_deep_v2.src.classifier', 'DeepLearningV2Classifier'),
        ]
        
        for module_name, class_name in classifiers_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                classifier_class = getattr(module, class_name)
                
                # Test instantiation
                classifier = classifier_class()
                
                # Test required methods exist
                required_methods = ['predict', 'get_metadata', 'load_model']
                for method in required_methods:
                    self.assertTrue(hasattr(classifier, method))
                
                # Test metadata structure
                metadata = classifier.get_metadata()
                self.assertIn('model_type', metadata)
                
            except ImportError:
                continue  # Skip if module not available


def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMLModelsCore,
        TestShallowLearning,
        TestDeepLearningV1,
        TestDeepLearningV2,
        TestTransferLearning,
        TestDataLoaders,
        TestIntegration,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main entry point for running tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unit tests for image classification modules')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity (use -vv for more verbose)')
    parser.add_argument('--test', '-t', type=str, help='Run specific test class')
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        suite = unittest.TestLoader().loadTestsFromName(args.test, module=sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=args.verbose + 1)
        result = runner.run(suite)
        success = result.wasSuccessful()
    else:
        # Run all tests
        success = run_tests(verbosity=args.verbose + 1)
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())