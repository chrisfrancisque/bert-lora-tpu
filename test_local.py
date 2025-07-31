from config import TrainingConfig
from dataset_utils import load_and_prepare_dataset
from lora_model import create_lora_model
from torch.utils.data import DataLoader

config = TrainingConfig(
    use_tpu=False,
    train_samples=100,
    per_device_train_batch_size=4
)

print("Testing dataset loading...")
train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
print(f"✓ Dataset loaded: {len(train_dataset)} train samples")

print("\nTesting LoRA Model Creation...")
model = create_lora_model(config)
print("Model Created Successfully")

print("\nTesting forward pass...")
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
batch = next(iter(dataloader))

outputs = model(**batch)
print(f"Forward Pass sucessful, loss: {outputs.loss.item():.4f}")

print("All Tests passed, ready for TPU deployment")