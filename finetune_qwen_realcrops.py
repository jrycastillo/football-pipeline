"""
Fine-tune Qwen2.5-VL on Labeled Real Crops Dataset
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
DATA_DIR = Path("data/real_crops_labeled")
OUTPUT_DIR = Path("output_models/qwen_realcrops_finetuned")
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

EPOCHS = 10  # More epochs for small dataset
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-5  # Higher LR for small dataset

# ============= Dataset =============
class RealCropsDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Load labels
        with open(self.data_dir / "labels.json") as f:
            self.samples = json.load(f)
        
        # Filter valid samples
        self.samples = [s for s in self.samples if s.get("label", -1) > 0]
        print(f"Loaded {len(self.samples)} labeled real crops")
        
        self.prompt = "What is the jersey number on this player's shirt? Reply with only the number."
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_dir / "images" / sample["path"]
        img = Image.open(img_path).convert("RGB")
        jersey_num = sample["label"]
        
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
                "content": [{"type": "text", "text": str(jersey_num)}]
            }
        ]
        
        return {"messages": messages, "image": img, "label": jersey_num}


def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    
    prompts = []
    for item in batch:
        text = processor.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        prompts.append(text)
    
    inputs = processor(
        text=prompts, images=images, padding=True, return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs, labels


def train():
    print("=" * 60)
    print("Fine-tuning Qwen2.5-VL on REAL CROPS Dataset")
    print("=" * 60)
    
    print(f"\n1. Loading model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    print("\n2. Configuring LoRA...")
    lora_config = LoraConfig(
        r=32,  # Higher rank for small dataset
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\n3. Loading real crops dataset...")
    dataset = RealCropsDataset(DATA_DIR, processor)
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor), num_workers=0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n4. Training for {EPOCHS} epochs...")
    model.train()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_loss = float("inf")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (inputs, labels) in enumerate(progress):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            loss = outputs.loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            progress.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION:.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(OUTPUT_DIR / "best_model")
            processor.save_pretrained(OUTPUT_DIR / "best_model")
            print(f"   ✅ Saved best model (loss: {best_loss:.4f})")
    
    model.save_pretrained(OUTPUT_DIR / "final_model")
    processor.save_pretrained(OUTPUT_DIR / "final_model")
    print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
