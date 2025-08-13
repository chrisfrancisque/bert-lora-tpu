import torch
from transformers import AutoModelForSequenceClassification
from peft import (
    get_peft_model, 
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import logging
from load_baseline import load_baseline_model  # Add this import

logger = logging.getLogger(__name__)

def create_lora_model(config, device='cpu', use_baseline=True):
    """
    Create LoRA model, optionally starting from warmed baseline.
    
    Args:
        config: Training configuration
        device: Device to load model to
        use_baseline: Whether to use the warmed baseline model
    """
    
    if use_baseline:
        # Load the warmed baseline model
        logger.info("Loading warmed baseline model from baseline_model_seed42")
        model, tokenizer, baseline_info = load_baseline_model(
            baseline_path='baseline_model_seed42',
            device=device
        )
        logger.info(f"Starting from baseline with {baseline_info.get('warm_up_accuracy', 0):.4f} accuracy")
    else:
        # Original code path for testing/comparison
        logger.info(f"Loading base model: {config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            torch_dtype=torch.float32
        )
        tokenizer = None
        baseline_info = {}

    if config.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r = config.lora_config.r,
            lora_alpha = config.lora_config.lora_alpha,
            lora_dropout = config.lora_config.lora_dropout,
            target_modules = config.lora_config.target_modules,
            bias = config.lora_config.bias,
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
    
    return model, tokenizer, baseline_info  # Return all three