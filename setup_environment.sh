#!/bin/bash
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

# Setup script for ML Image Classification Project
# This script sets up a Python virtual environment for the entire project
# and registers a Jupyter kernel for notebook development

echo "Setting up ML Image Classification Project Environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch nightly first (to avoid conflicts)
echo "Installing PyTorch nightly with CUDA 12.8..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install dependencies without scikit-learn first
echo "Installing core dependencies..."
pip install numpy pandas pillow matplotlib seaborn opencv-python tqdm tensorflow-cpu jupyter ipykernel notebook fastapi uvicorn pydantic kagglehub requests typing-extensions joblib

# Install scikit-learn separately
echo "Installing scikit-learn..."
pip install scikit-learn

# Register Jupyter kernel
echo "Registering Jupyter kernel..."
if python -m ipykernel install --user --name=ml-image-classification --display-name="ML Image Classification (Python 3.11)"; then
    echo "✅ Kernel registered successfully"
else
    echo "❌ Failed to register kernel"
    exit 1
fi

# Install core module in development mode
echo "Installing ml-models-core in development mode..."
if pip install -e ml-models-core/; then
    echo "✅ Core module installed successfully"
else
    echo "❌ Failed to install core module"
    exit 1
fi

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "Jupyter kernel registered: ML Image Classification (Python 3.11)"
echo ""
echo "To use the environment:"
echo "  1. Activate: source venv/bin/activate"
echo "  2. Start Jupyter: jupyter lab"
echo "  3. Select 'ML Image Classification (Python 3.11)' kernel"
echo ""
echo "To setup datasets:"
echo "  source venv/bin/activate && python setup_data.py"