#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def verify_baseline():
    """Verify the baseline model is properly set up"""
    
    baseline_path = 'baseline_model_seed42'
    
    # Check if baseline exists
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline model not found at {baseline_path}")
        print("Please copy from fine-tuning-project:")
        print("cp -r ~/fine-tuning-project/mft_models_20250820_165656/checkpoint_50pct_baseline ./baseline_model_seed42")
        return False
    
    print(f"✓ Baseline model found at {baseline_path}")
    
    # Check files
    required_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json', 'tokenizer_config.json']
    for file in required_files:
        path = os.path.join(baseline_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # Size in MB
            print(f"  ✓ {file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {file} missing")
    
    # Try loading the model
    try:
        print("\\nLoading model...")
        model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
        tokenizer = AutoTokenizer.from_pretrained(baseline_path)
        
        # Check model architecture
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        
        # Check baseline info
        info_path = os.path.join(baseline_path, 'baseline_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"  Baseline accuracy: {info.get('warm_up_accuracy', 0):.4f}")
        
        print("\\n✅ Baseline model is ready for LoRA training!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    verify_baseline()
