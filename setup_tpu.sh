#!/bin/bash

echo "Setting up TPU environment for LoRA training..."

# Create virtual environment
python3.10 -m venv ~/tpu-env
source ~/tpu-env/bin/activate

# Install requirements
pip install --upgrade pip
pip install torch==2.7.0
pip install torch-xla[tpu]==2.7.0
pip install transformers==4.36.0
pip install datasets==2.16.1
pip install peft==0.8.2
pip install numpy==1.26.4
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install tqdm

# Set environment variables
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TORCH_COMPILE_DISABLE=1

echo "Environment setup complete!"
echo "Baseline model status:"
if [ -d "baseline_model_seed42" ]; then
    echo "✓ Baseline model found"
    ls -la baseline_model_seed42/
else
    echo "✗ Baseline model not found - please copy from fine-tuning-project"
fi
