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
import torch._dynamo.config as dynamo_config
import torch.nn.functional as F

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
    gradient_data ={
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
    
    #Calculate Statistics
    if lora_grads:
        gradient_data['statistics']['lora_mean'] = sum(lora_grads)/len(lora_grads)
        gradient_data['statistics']['lora_max'] = max(lora_grads)
        gradient_data['statistics']['lora_min'] = min(lora_grads)

    if base_grads:
        gradient_data['statistics']['base_mean'] = sum(base_grads)/ len(base_grads)
        gradient_data['statistics']['base_max'] = max(base_grads)

    
    filename = f"{writer_dict['gradient_dir']}/gradient_step_{step}.json"
    with open(filename, 'w') as f:
        json.dump(gradient_data, f, indent =2)

    return gradient_data['statistics']

def train_on_tpu(index, config):
    #device setup
    device = xm.xla_device()
    logger.info("Device set")
    

    is_master = index == 0



    if is_master:
        logger.info(f"Starting training on TPU core {index}")
        setup_directories(config)

    #Load data on ALL processes not just master
    train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
    logger.info("Datasets Loaded")

    model = create_lora_model(config)
    model = model.to(device)
    logger.info("Model created")


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay = config.weight_decay
    )
    logger.info("Optimizer set")



    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=8,
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
        'gradient_dir' : f"{config.output_dir}/gradients"
    }

    metrics_history = []

    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warmup_steps

    if is_master:
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

    logger.info("starting training loop")
    # Training Loop
    global_step = 0

    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0

        if is_master:
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(train_device_loader):
            
            
            logger.info("Forward Pass starting")
            #Forward Pass with autocast for bfloat16
            with torch.autocast('xla', dtype=torch.bfloat16):
                outputs = model(**batch)
                logits = outputs.logits
            
            #Compute loss in float32 for numerical stability
            with torch.autocast('xla', enabled=False):
                logits_fp32 = logits.float()
                labels = batch['labels']
                loss = F.cross_entropy(logits_fp32, labels)
            
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {global_step}, skipping batch")
                optimizer.zero_grad()
                continue

            #backward pass
            loss.backward()
            logger.info("gradient tracking starting")

            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            if config.track_gradients and global_step % config.gradient_tracking_steps == 0:
                if is_master:
                    grad_stats = track_gradients(model, global_step, writer_dict)
                    logger.info(f"Step {global_step} gradient stats: {grad_stats}")

            
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()

            #Update learning rate
            if global_step < num_warmup_steps:
                lr = config.learning_rate * global_step / num_warmup_steps
            else:
                progress = (global_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                lr = config.learning_rate * 0.5 * (1+ torch.cos(torch.tensor(3.14159 * progress)))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr.item() if hasattr(lr, 'item') else lr

            
            epoch_loss += loss.item()
            global_step += 1

            if is_master and step%10 == 0:
                pbar.update(10)
                pbar.set_postfix({'loss': loss.item(), 'lr': lr})

            logger.info("Parameters updated")
            
        if is_master:
            pbar.close()
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

            #Evaluate at end of epoch
            logger.info("Running evaluation")
            eval_metrics = evaluate_model(model, eval_dataset, config, device)

            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items() 
                                   if k != 'roc_curve'])
            
            logger.info(f"Epoch {epoch} eval metrics: {metrics_str}")

            metrics_history.append(eval_metrics)

            save_metrics(eval_metrics, f"{config.output_dir}/metrics_epoch_{epoch}.json")

            if epoch == config.num_train_epochs - 1:
                logger.info("Generating Visualizations...")
                visualize_metrics(metrics_history, f"{config.output_dir}/plots")

                save_metrics(eval_metrics, f"{config.output_dir}/final_metrics.json")

                summary = {
                    'config': {
                        'model':config.model_name,
                        'lora_rank': config.lora_config.r,
                        'lora_alpha': config.lora_config.lora_alpha,
                        'learning_rate': config.learning_rate,
                        'batch_size': config.total_train_batch_size,
                        'epochs': config.num_train_epochs
                    },
                    'final_metrics': {k: v for k,v in eval_metrics.items() if k != 'roc_curve'},
                    'metrics_history': [{k: v for k,v in m.items() if k != 'roc_curve'} for m in metrics_history]
                }

                with open(f"{config.output_dir}/training_summary.json", 'w') as f:
                    json.dump(summary, f, indent =2)


            #Save checkpoint
            checkpoint_path = f"{config.output_dir}/checkpoint_epoch_{epoch}"
            os.makedirs(checkpoint_path, exist_ok=True)

            #Save LoRA weights only
            lora_state_dict = get_lora_state_dict(model)
            xm.save(lora_state_dict, f"{checkpoint_path}/adapter_model.bin")

            #Save configuration
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    if is_master:
        logger.info("Training Completed")

def main():
    config = TrainingConfig()

    if config.use_tpu:
   
        xmp.spawn(train_on_tpu, args=(config,))
    else:
        train_on_tpu(0, config)
if __name__ =="__main__":
    main()






