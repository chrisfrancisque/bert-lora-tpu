import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo"
os.environ["XLA_USE_TORCH_COMPILE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch._dynamo.config.disable = True
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json
from tqdm import tqdm
import torch.nn.functional as F
import math

from config import TrainingConfig
from dataset_utils import load_and_prepare_dataset
from lora_model import create_lora_model, get_lora_state_dict
from evaluate import evaluate_model, visualize_metrics, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(config):
    """Create output directories"""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/gradients", exist_ok=True)
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)


def track_gradients(model, step, writer_dict):
    gradient_data = {
        'step': step,
        'lora_gradients': {},
        'base_gradients': {},
        'statistics': {}
    }

    lora_grads = []
    base_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            if "lora_" in name:
                lora_grads.append(grad_norm)
                gradient_data['lora_gradients'][name] = grad_norm
            elif param.requires_grad:
                base_grads.append(grad_norm)
                gradient_data['base_gradients'][name] = grad_norm
    
    # Calculate Statistics
    if lora_grads:
        gradient_data['statistics']['lora_mean'] = sum(lora_grads) / len(lora_grads)
        gradient_data['statistics']['lora_max'] = max(lora_grads)
        gradient_data['statistics']['lora_min'] = min(lora_grads)

    if base_grads:
        gradient_data['statistics']['base_mean'] = sum(base_grads) / len(base_grads)
        gradient_data['statistics']['base_max'] = max(base_grads)

    filename = f"{writer_dict['gradient_dir']}/gradient_step_{step}.json"
    with open(filename, 'w') as f:
        json.dump(gradient_data, f, indent=2)

    return gradient_data['statistics']


def train_on_tpu(index, config):
    # Device setup
    device = xm.xla_device()
    
    # The index is passed as the first argument by xmp.spawn
    is_master = index == 0  # Master is typically index 0

    if is_master:
        logger.info(f"Starting LoRA training on TPU core {index}")
        logger.info(f"Training configuration:")
        logger.info(f"  - Train samples: {config.train_samples}")
        logger.info(f"  - Batch size per device: {config.per_device_train_batch_size}")
        logger.info(f"  - Total batch size: {config.total_train_batch_size}")
        logger.info(f"  - Learning rate: {config.learning_rate}")
        logger.info(f"  - LoRA rank: {config.lora_config.r}")
        logger.info(f"  - LoRA alpha: {config.lora_config.lora_alpha}")
        setup_directories(config)

    # Load data on ALL processes
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    
    # Create model from warmed baseline
    model, tokenizer, baseline_info = create_lora_model(config, device='cpu', use_baseline=True)
    model = model.to(device)
    
    if is_master:
        logger.info(f"Starting from warmed baseline model")
        logger.info(f"Baseline accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f}")
        logger.info(f"Training will progress from this 48.86% baseline")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # World size is 8 for TPU v3-8
    world_size = 8
    
    # Create distributed sampler and dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=index,
        shuffle=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.per_device_train_batch_size,
        drop_last=True
    )

    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)

    writer_dict = {
        'gradient_dir': f"{config.output_dir}/gradients"
    }

    metrics_history = []
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warmup_steps

    if is_master:
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
        logger.info(f"World size: {world_size}")

    # Training Loop
    global_step = 0
    best_accuracy = baseline_info.get('warm_up_accuracy', 0.4886)

    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0

        if is_master:
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_train_epochs}")

        for step, batch in enumerate(train_device_loader):
            # Forward Pass with autocast for bfloat16
            with torch.autocast('xla', dtype=torch.bfloat16):
                outputs = model(**batch)
                logits = outputs.logits
            
            # Compute loss in float32 for numerical stability
            with torch.autocast('xla', enabled=False):
                logits_fp32 = logits.float()
                labels = batch['labels']
                loss = F.cross_entropy(logits_fp32, labels)
            
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Track gradients if needed
            if config.track_gradients and global_step % config.gradient_tracking_steps == 0:
                if is_master:
                    grad_stats = track_gradients(model, global_step, writer_dict)
                    logger.info(f"Step {global_step} gradient stats: {grad_stats}")

            # Optimizer step
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            xm.mark_step()

            # Update learning rate with cosine schedule
            if global_step < num_warmup_steps:
                lr = config.learning_rate * global_step / num_warmup_steps
            else:
                progress = (global_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            epoch_loss += loss.item()
            global_step += 1

            if is_master and step % 10 == 0:
                pbar.update(10)
                pbar.set_postfix({'loss': loss.item(), 'lr': lr})

        # End of epoch
        if is_master:
            pbar.close()
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Evaluation
        logger.info("Running evaluation")
        eval_metrics = evaluate_model(model, eval_dataset, config, device)
        xm.mark_step()
        
        # Synchronization point
        xm.rendezvous(f"eval_epoch_{epoch}_complete")

        # Only master handles metrics logging and saving
        if is_master:
            current_accuracy = eval_metrics['accuracy']
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items() 
                                   if k != 'roc_curve'])
            logger.info(f"Epoch {epoch+1} eval metrics: {metrics_str}")
            
            # Track improvement from baseline
            improvement = current_accuracy - baseline_info.get('warm_up_accuracy', 0.4886)
            logger.info(f"Improvement from baseline: +{improvement:.4f} ({improvement*100:.2f}%)")
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                logger.info(f"New best accuracy: {best_accuracy:.4f}")
            
            metrics_history.append(eval_metrics)
            save_metrics(eval_metrics, f"{config.output_dir}/metrics_epoch_{epoch+1}.json")

        # Save checkpoint
        if is_master:
            checkpoint_path = f"{config.output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save LoRA weights only
            lora_state_dict = get_lora_state_dict(model)
            cpu_state_dict = {k: v.cpu() for k, v in lora_state_dict.items()}
            torch.save(cpu_state_dict, f"{checkpoint_path}/adapter_model.bin")
            
            # Save model config
            model.save_pretrained(checkpoint_path, safe_serialization=False)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save training info
            checkpoint_info = {
                'epoch': epoch + 1,
                'accuracy': current_accuracy,
                'baseline_accuracy': baseline_info.get('warm_up_accuracy', 0.4886),
                'improvement': improvement,
                'best_accuracy': best_accuracy,
                'lora_rank': config.lora_config.r,
                'lora_alpha': config.lora_config.lora_alpha
            }
            with open(f"{checkpoint_path}/checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Synchronization after checkpoint
        xm.rendezvous(f"checkpoint_epoch_{epoch}_saved")
        xm.mark_step()

    # Generate final visualizations and summary
    if is_master:
        logger.info("Generating final visualizations...")
        visualize_metrics(metrics_history, f"{config.output_dir}/plots")
        
        final_metrics = metrics_history[-1] if metrics_history else {}
        save_metrics(final_metrics, f"{config.output_dir}/final_metrics.json")

        # Create comprehensive summary
        summary = {
            'training_config': {
                'model': 'BERT-base with warmed baseline',
                'baseline_accuracy': baseline_info.get('warm_up_accuracy', 0.4886),
                'lora_rank': config.lora_config.r,
                'lora_alpha': config.lora_config.lora_alpha,
                'target_modules': config.lora_config.target_modules,
                'learning_rate': config.learning_rate,
                'batch_size': config.total_train_batch_size,
                'epochs': config.num_train_epochs,
                'train_samples': config.train_samples
            },
            'results': {
                'final_accuracy': final_metrics.get('accuracy', 0),
                'best_accuracy': best_accuracy,
                'improvement_from_baseline': best_accuracy - baseline_info.get('warm_up_accuracy', 0.4886),
                'final_metrics': {k: v for k, v in final_metrics.items() if k != 'roc_curve'}
            },
            'metrics_history': [{k: v for k, v in m.items() if k != 'roc_curve'} 
                               for m in metrics_history]
        }
        
        with open(f"{config.output_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("LoRA Training Complete!")
        logger.info(f"Baseline accuracy: {baseline_info.get('warm_up_accuracy', 0.4886):.4f}")
        logger.info(f"Final accuracy: {final_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        logger.info(f"Total improvement: +{best_accuracy - baseline_info.get('warm_up_accuracy', 0.4886):.4f}")
        logger.info("=" * 50)


def main():
    config = TrainingConfig()
    
    # Use None for nprocs to use all available devices
    xmp.spawn(train_on_tpu, args=(config,), nprocs=None)

if __name__ == "__main__":
    main()
