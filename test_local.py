from config import TrainingConfig
from dataset_utils import load_and_prepare_dataset
from lora_model import create_lora_model
from torch.utils.data import DataLoader
import torch

config = TrainingConfig(
    use_tpu=False,
    train_samples=100,
    per_device_train_batch_size=4
)

print("Testing dataset loading...")
train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
print(f"✓ Dataset loaded: {len(train_dataset)} train samples")

print("\nTesting LoRA Model Creation with Baseline...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer, baseline_info = create_lora_model(config, device=device, use_baseline=True)
print(f"✓ Model created from baseline (accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f})")

print("\nTesting forward pass...")
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
batch = next(iter(dataloader))
batch = {k: v.to(device) for k, v in batch.items()}

outputs = model(**batch)
print(f"✓ Forward pass successful, loss: {outputs.loss.item():.4f}")

print("\n✅ All tests passed! Ready for TPU deployment")