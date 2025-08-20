import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import (
    get_peft_model, 
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import logging
import os

logger = logging.getLogger(__name__)

def load_baseline_model(baseline_path='baseline_model_seed42', device='cpu'):
    """Load the warmed baseline model"""
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline model not found at {baseline_path}!\\n"
            f"Please ensure the baseline model is copied from fine-tuning-project"
        )
    
    logger.info(f"Loading baseline model from {baseline_path}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)
    
    # Load baseline info if available
    import json
    baseline_info_path = os.path.join(baseline_path, 'baseline_info.json')
    if os.path.exists(baseline_info_path):
        with open(baseline_info_path, 'r') as f:
            baseline_info = json.load(f)
        logger.info(f"Baseline accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f}")
    else:
        baseline_info = {}
    
    model.to(device)
    
    # Ensure all parameters are unfrozen for LoRA
    for param in model.parameters():
        param.requires_grad = True
    
    return model, tokenizer, baseline_info

def create_lora_model(config, device='cpu', use_baseline=True):
    """Create LoRA model starting from warmed baseline"""
    
    if use_baseline:
        # Load the warmed baseline model
        logger.info("Loading warmed baseline model")
        model, tokenizer, baseline_info = load_baseline_model(
            baseline_path='baseline_model_seed42',
            device=device
        )
        logger.info(f"Starting from baseline with {baseline_info.get('warm_up_accuracy', 0):.4f} accuracy")
    else:
        # Fallback to original model (for testing)
        logger.info(f"Loading base model: {config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        baseline_info = {}

    if config.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_config.r,
            lora_alpha=config.lora_config.lora_alpha,
            lora_dropout=config.lora_config.lora_dropout,
            target_modules=config.lora_config.target_modules,
            bias=config.lora_config.bias,
            use_rslora=config.lora_config.use_rslora,
            modules_to_save=config.lora_config.modules_to_save
        )

        model = get_peft_model(model, lora_config)

        # Print trainable parameter info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"All parameters: {all_params:,}")
        logger.info(f"Percentage trainable: {trainable_percent:.4f}%")

        # Print which modules are being trained
        logger.info("Modules being trained:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: {param.shape}")
    
    return model, tokenizer, baseline_info

def get_lora_state_dict(model):
    """Extract only LoRA parameters from model"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            lora_state_dict[name] = param
    return lora_state_dict
