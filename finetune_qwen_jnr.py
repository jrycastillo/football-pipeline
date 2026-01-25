"""
Fine-tune Qwen2.5-VL-3B on SoccerNet JNR Dataset
Uses LoRA for efficient training on H100 GPU.
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import warnings
warnings.filterwarnings("ignore")

# ============= Configuration =============
DATA_DIR = Path("data/soccernet_jnr/jersey-2023/train/train")
OUTPUT_DIR = Path("output_models/qwen_jnr_finetuned")
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
MAX_SAMPLES = 10000  # Limit for faster training

# ============= Dataset =============
class SoccerNetJNRDataset(Dataset):
    def __init__(self, data_dir, processor, max_samples=None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Directly scan image files - structure is images/{jersey_num}/{jersey_num}_{id}.jpg
        self.samples = []
        images_dir = self.data_dir / "images"
        
        for jersey_folder in images_dir.iterdir():
            if not jersey_folder.is_dir():
                continue
            try:
                jersey_num = int(jersey_folder.name)
                if jersey_num < 1 or jersey_num > 99:
                    continue
            except ValueError:
                continue
            
            for img_file in jersey_folder.glob("*.jpg"):
                self.samples.append({
                    "image_path": str(img_file),
                    "jersey_num": jersey_num
                })
        
        # Shuffle and limit samples
        import random
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from SoccerNet JNR")
        
        # Prompt template
        self.prompt = "What is the jersey number on this player's shirt? Reply with only the number (1-99)."
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        jersey_num = sample["jersey_num"]
        
        # Create chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self.prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(jersey_num)}
                ]
            }
        ]
        
        return {"messages": messages, "image": img, "label": jersey_num}


def collate_fn(batch, processor):
    """Custom collate for Qwen VL."""
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Format messages
    prompts = []
    for item in batch:
        text = processor.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        prompts.append(text)
    
    # Process
    inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    
    # Create labels (shift input_ids for causal LM)
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs, labels


def train():
    print("=" * 60)
    print("Fine-tuning Qwen2.5-VL-3B on SoccerNet JNR Dataset")
    print("=" * 60)
    
    # Load model and processor
    print(f"\n1. Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    # Configure LoRA
    print("\n2. Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("\n3. Loading SoccerNet JNR dataset...")
    dataset = SoccerNetJNRDataset(DATA_DIR, processor, max_samples=MAX_SAMPLES)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n4. Training for {EPOCHS} epochs...")
    model.train()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_loss = float("inf")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (inputs, labels) in enumerate(progress):
            # Move to GPU
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward
            outputs = model(**inputs)
            loss = outputs.loss / GRADIENT_ACCUMULATION
            
            # Backward
            loss.backward()
            
            # Update weights
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            progress.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION:.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(OUTPUT_DIR / "best_model")
            processor.save_pretrained(OUTPUT_DIR / "best_model")
            print(f"   ✅ Saved best model (loss: {best_loss:.4f})")
    
    # Save final model
    model.save_pretrained(OUTPUT_DIR / "final_model")
    processor.save_pretrained(OUTPUT_DIR / "final_model")
    print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
