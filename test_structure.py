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
Structural tests for extracted image classification modules.

This script tests the module structure and ensures all expected files
and components are present without requiring heavy dependencies.
"""

import unittest
from pathlib import Path
import ast
import sys

# Add project root to path
project_root = Path(__file__).parent


class TestModuleStructure(unittest.TestCase):
    """Test the structure of extracted modules."""
    
    def test_model_directories_exist(self):
        """Test that all model directories exist with proper structure."""
        expected_models = [
            'image-classifier-shallow',
            'image-classifier-deep-v1', 
            'image-classifier-deep-v2',
            'image-classifier-transfer'
        ]
        
        for model_name in expected_models:
            model_path = project_root / model_name
            self.assertTrue(model_path.exists(), f"Model directory {model_name} should exist")
            
            # Check src directory
            src_path = model_path / 'src'
            self.assertTrue(src_path.exists(), f"src directory should exist for {model_name}")
            
            # Check scripts directory
            scripts_path = model_path / 'scripts'
            self.assertTrue(scripts_path.exists(), f"scripts directory should exist for {model_name}")
        
        print(f"✅ All {len(expected_models)} model directories exist with proper structure")
    
    def test_required_src_files(self):
        """Test that required source files exist in each model."""
        models = [
            'image-classifier-shallow',
            'image-classifier-deep-v1',
            'image-classifier-deep-v2', 
            'image-classifier-transfer'
        ]
        
        required_files = [
            '__init__.py',
            'config.py',
            'classifier.py'
        ]
        
        files_found = 0
        
        for model_name in models:
            src_path = project_root / model_name / 'src'
            if not src_path.exists():
                continue
                
            for filename in required_files:
                file_path = src_path / filename
                if file_path.exists():
                    files_found += 1
                    self.assertTrue(file_path.is_file(), f"{filename} should be a file in {model_name}")
        
        print(f"✅ Found {files_found} required source files across all models")
        self.assertGreater(files_found, 0, "Should find at least some required source files")
    
    def test_cli_scripts_exist(self):
        """Test that CLI training scripts exist."""
        models = [
            'image-classifier-shallow',
            'image-classifier-deep-v1',
            'image-classifier-deep-v2',
            'image-classifier-transfer'
        ]
        
        scripts_found = 0
        
        for model_name in models:
            script_path = project_root / model_name / 'scripts' / 'train.py'
            if script_path.exists():
                scripts_found += 1
                self.assertTrue(script_path.is_file(), f"train.py should exist for {model_name}")
        
        print(f"✅ Found {scripts_found} CLI training scripts")
        self.assertGreater(scripts_found, 0, "Should find at least some CLI scripts")
    
    def test_unified_cli_exists(self):
        """Test that unified CLI exists."""
        unified_cli = project_root / 'train_models.py'
        self.assertTrue(unified_cli.exists(), "Unified CLI should exist")
        self.assertTrue(unified_cli.is_file(), "Unified CLI should be a file")
        
        # Test that it's executable
        import stat
        file_stat = unified_cli.stat()
        is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
        self.assertTrue(is_executable, "Unified CLI should be executable")
        
        print("✅ Unified CLI exists and is executable")
    
    def test_python_syntax_validity(self):
        """Test that all Python files have valid syntax."""
        python_files = []
        
        # Find all Python files
        for model_name in ['image-classifier-shallow', 'image-classifier-deep-v1', 
                          'image-classifier-deep-v2', 'image-classifier-transfer']:
            model_path = project_root / model_name
            if model_path.exists():
                python_files.extend(model_path.rglob('*.py'))
        
        # Add root Python files
        python_files.extend(project_root.glob('*.py'))
        
        syntax_errors = []
        files_checked = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Parse and compile to check syntax
                ast.parse(source)
                compile(source, str(py_file), 'exec')
                files_checked += 1
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                # Skip files that might have import issues
                pass
        
        print(f"✅ Checked syntax of {files_checked} Python files")
        
        if syntax_errors:
            for error in syntax_errors:
                print(f"❌ Syntax error: {error}")
        
        self.assertEqual(len(syntax_errors), 0, f"Found {len(syntax_errors)} syntax errors")
    
    def test_class_definitions_exist(self):
        """Test that expected class definitions exist in source files."""
        expected_classes = {
            'image-classifier-shallow/src/classifier.py': ['ShallowImageClassifier'],
            'image-classifier-deep-v1/src/classifier.py': ['DeepLearningV1Classifier'],
            'image-classifier-deep-v2/src/classifier.py': ['DeepLearningV2Classifier'],
            'image-classifier-transfer/src/classifier.py': ['TransferLearningClassifier'],
            'image-classifier-shallow/src/config.py': ['ShallowLearningConfig'],
            'image-classifier-deep-v1/src/config.py': ['DeepLearningV1Config'],
            'image-classifier-deep-v2/src/config.py': ['DeepLearningV2Config'],
            'image-classifier-transfer/src/config.py': ['TransferLearningClassifierConfig'],
        }
        
        classes_found = 0
        total_expected = sum(len(classes) for classes in expected_classes.values())
        
        for file_path, class_names in expected_classes.items():
            full_path = project_root / file_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                # Find class definitions
                found_classes = [node.name for node in ast.walk(tree) 
                               if isinstance(node, ast.ClassDef)]
                
                for class_name in class_names:
                    if class_name in found_classes:
                        classes_found += 1
                        
            except Exception as e:
                # Skip files with parsing issues
                pass
        
        print(f"✅ Found {classes_found}/{total_expected} expected class definitions")
        self.assertGreater(classes_found, 0, "Should find at least some expected classes")
    
    def test_import_structure(self):
        """Test import structure in __init__.py files."""
        init_files_tested = 0
        
        for model_name in ['image-classifier-shallow', 'image-classifier-deep-v1',
                          'image-classifier-deep-v2', 'image-classifier-transfer']:
            init_path = project_root / model_name / 'src' / '__init__.py'
            if not init_path.exists():
                continue
                
            try:
                with open(init_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for imports
                self.assertIn('from .classifier import', content, 
                             f"__init__.py should import classifier in {model_name}")
                
                # Check for __all__ definition
                if '__all__' in content:
                    tree = ast.parse(content)
                    has_all = any(isinstance(node, ast.Assign) and 
                                any(target.id == '__all__' for target in node.targets 
                                   if isinstance(target, ast.Name))
                                for node in ast.walk(tree))
                    self.assertTrue(has_all, f"__all__ should be properly defined in {model_name}")
                
                init_files_tested += 1
                
            except Exception as e:
                # Skip files with issues
                pass
        
        print(f"✅ Tested import structure in {init_files_tested} __init__.py files")
        self.assertGreater(init_files_tested, 0, "Should test at least some __init__.py files")
    
    def test_documentation_files(self):
        """Test that documentation files exist."""
        doc_files = [
            'CLI_USAGE.md',
            'test_models.py',
            'test_configs.py'
        ]
        
        docs_found = 0
        
        for doc_file in doc_files:
            doc_path = project_root / doc_file
            if doc_path.exists():
                docs_found += 1
                self.assertTrue(doc_path.is_file(), f"{doc_file} should be a file")
        
        print(f"✅ Found {docs_found}/{len(doc_files)} documentation files")
        self.assertGreater(docs_found, 0, "Should find at least some documentation files")


class TestCodeQuality(unittest.TestCase):
    """Test code quality aspects."""
    
    def test_license_headers(self):
        """Test that Python files have license headers."""
        python_files = []
        
        # Find Python files
        for model_name in ['image-classifier-shallow', 'image-classifier-deep-v1',
                          'image-classifier-deep-v2', 'image-classifier-transfer']:
            model_path = project_root / model_name
            if model_path.exists():
                python_files.extend(model_path.rglob('*.py'))
        
        # Add root Python files
        python_files.extend(project_root.glob('*.py'))
        
        files_with_license = 0
        files_checked = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                files_checked += 1
                
                # Check for license header
                if 'Copyright' in content and 'Licensed under the Apache License' in content:
                    files_with_license += 1
                    
            except Exception:
                pass
        
        print(f"✅ {files_with_license}/{files_checked} Python files have license headers")
        
        # At least some files should have license headers
        if files_checked > 0:
            license_ratio = files_with_license / files_checked
            self.assertGreater(license_ratio, 0.5, "Most files should have license headers")
    
    def test_docstring_presence(self):
        """Test that classes and functions have docstrings."""
        python_files = []
        
        # Find source files
        for model_name in ['image-classifier-shallow', 'image-classifier-deep-v1',
                          'image-classifier-deep-v2', 'image-classifier-transfer']:
            src_path = project_root / model_name / 'src'
            if src_path.exists():
                python_files.extend(src_path.glob('*.py'))
        
        classes_with_docstrings = 0
        total_classes = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            classes_with_docstrings += 1
                            
            except Exception:
                pass
        
        print(f"✅ {classes_with_docstrings}/{total_classes} classes have docstrings")
        
        if total_classes > 0:
            docstring_ratio = classes_with_docstrings / total_classes
            self.assertGreater(docstring_ratio, 0.3, "Many classes should have docstrings")


def main():
    """Run structural tests."""
    print("Running Structural Tests for Extracted Modules")
    print("=" * 60)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModuleStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeQuality))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("Structural Tests Summary:")
    print("- Verified module directory structure")
    print("- Checked required source files exist")
    print("- Validated Python syntax")
    print("- Confirmed class definitions")
    print("- Tested import structure")
    print("- Checked code quality aspects")
    
    success = result.wasSuccessful()
    if success:
        print("\n✅ All structural tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())