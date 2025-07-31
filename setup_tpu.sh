#!/bin/bash
# TPU Environment Setup Script

echo "=== Starting TPU Setup ==="

# Update system
echo "Updating system packages..."
sudo apt-get update

# Create virtual environment
echo "Creating Python 3.10 virtual environment..."
python3.10 -m venv ~/tpu-env
source ~/tpu-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch and XLA
echo "Installing PyTorch 2.7.0 and XLA..."
pip install torch==2.7.0
pip install 'torch_xla[tpu]==2.7.0' -f https://storage.googleapis.com/libtpu-wheels/index.html

# Install project requirements
echo "Installing project requirements..."
pip install transformers==4.36.0 datasets==2.16.1 numpy==1.24.3
pip install peft==0.8.2
pip install scikit-learn matplotlib seaborn tqdm
pip install pyarrow==11.0.0 fsspec==2023.5.0

# Set environment variables
echo "Setting environment variables..."
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TORCH_COMPILE_DISABLE=1

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch_xla; print('XLA import successful')"

echo "=== TPU Setup Complete ==="
echo "To activate the environment in future sessions, run:"
echo "source ~/tpu-env/bin/activate"