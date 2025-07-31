# BERT LoRA Fine-tuning on Google Cloud TPU

Low-Rank Adaptation (LoRA) implementation for BERT-base model on SST-2 sentiment analysis using Google Cloud TPU v3-8.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements parameter-efficient fine-tuning of BERT using LoRA, training only 0.037% of model parameters while maintaining performance of ~90%. The implementation is optimized for Google Cloud TPUs using PyTorch/XLA.

### Key Features
* LoRA implementation via Hugging Face PEFT
* TPU-optimized distributed training
* Comprehensive evaluation metrics with visualization
* Gradient tracking for adaptation analysis
* Memory-efficient training pipeline
* Automatic mixed precision with bfloat16

## Architecture

### Model Configuration
* Base Model: bert-base-uncased (110M parameters)
* LoRA Rank: 8
* LoRA Alpha: 16
* Target Modules: Query and Value attention matrices
* Trainable Parameters: ~40K (0.037% of total)

### Training Configuration
* Learning Rate: 5e-4
* Batch Size: 128 per TPU core (1024 total)
* Optimizer: AdamW
* Scheduler: Linear warmup + cosine decay
* Mixed Precision: bfloat16
* Epochs: 3

## Requirements

### Hardware
* Google Cloud TPU v3-8 or v4-8
* Alternative: NVIDIA GPU with 16GB+ VRAM

### Software
* Python 3.10
* CUDA 11.8+ (for GPU)
* TPU VM with Ubuntu 22.04 base image

## Installation

### Local Development
```bash
git clone https://github.com/chrisfrancisque/bert-lora-tpu.git
cd bert-lora-tpu
python3.10 -m venv venv-local
source venv-local/bin/activate
pip install -r requirements.txt

TPU Setup
# Create TPU instance
gcloud compute tpus tpu-vm create bert-lora-tpu \
  --project=your-project-id \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-ubuntu2204-base

# SSH into TPU
gcloud compute tpus tpu-vm ssh bert-lora-tpu --zone=us-central1-a

# Clone and setup
git clone https://github.com/chrisfrancisque/bert-lora-tpu.git
cd bert-lora-tpu
bash setup_tpu.sh

Usage
Local Testing
python test_local.py

TPU Training
# Activate environment
source ~/tpu-env/bin/activate

# Set environment variables
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TORCH_COMPILE_DISABLE=1

# Run training
python train_lora_tpu.py

Custom Configuration
# Modify config.py
config = TrainingConfig(
    num_train_epochs=5,
    learning_rate=3e-4,
    lora_config=LoRAConfig(r=16, lora_alpha=32)
)

Troubleshooting
Common Issues
Dataset Cache Corruption
# Force fresh download
download_config = DownloadConfig(
    force_download=True,
    resume_download=False
)

PyTorch 2.7.0 Compilation Hang
export TORCH_COMPILE_DISABLE=1

TPU Permission Error
sudo pkill -f libtpu
sudo systemctl restart tpu-runtime

Distributed Training Synchronization

Ensure data loading occurs on all TPU cores
Use xm.mark_step() for proper graph execution

Memory Optimization

Reduce batch size if OOM occurs
Enable gradient checkpointing for larger models
Use bfloat16 mixed precision


