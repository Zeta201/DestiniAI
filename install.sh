#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Uninstalling incompatible packages..."
pip uninstall -y numpy torch torchvision torchaudio

echo "Installing compatible numpy version..."
pip install "numpy<2"

echo "Installing PyTorch 2.5.0 with CUDA 12.1..."
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing additional dependencies..."
pip install sentence-transformers transformers pandas scikit-learn matplotlib surprise faker

echo "Installing flask..."
pip install flask

echo "Installing torch_geometric and torch_sparse..."
pip install torch_geometric
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

echo "✅ Installation complete. Activate the environment with: source venv/bin/activate"
