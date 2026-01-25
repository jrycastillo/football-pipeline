
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vision.resnet_recognition import create_resnet34_grayscale

# Configuration
DATA_DIR = "/home/ubuntu/football/data/manual_label_crops"
BASE_MODEL_PATH = "output_models/resnet34_jnr_grayscale/best_model.pt"
OUTPUT_DIR = "output_models/resnet34_jnr_manual_finetune"
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-5 # Low LR for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ManualRealCropsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Scan for files matching crop_{id}_{label}.jpg
        # Valid labels are 0-99
        pattern = re.compile(r'crop_\d+_(\d+)\.jpg')
        
        all_files = list(self.root_dir.glob("*.jpg"))
        print(f"Found {len(all_files)} total files in {root_dir}")
        
        for f in all_files:
            match = pattern.search(f.name)
            if match:
                label_str = match.group(1)
                try:
                    label = int(label_str)
                    if 0 <= label <= 99:
                        self.samples.append((f, label))
                except ValueError:
                    pass
        
        print(f"Loaded {len(self.samples)} valid samples (0-99). Filtered out 'LABEL' and others.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Open as Grayscale to match model (L)
        img = Image.open(path).convert('L')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset & Loader
    # Same transforms as training + some variations for robustness
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Grayscale norm
    ])
    
    dataset = ManualRealCropsDataset(DATA_DIR, transform=train_transform)
    
    if len(dataset) == 0:
        print("❌ No valid samples found! Check dataset path.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = create_resnet34_grayscale(num_classes=100)
    
    if os.path.exists(BASE_MODEL_PATH):
        checkpoint = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Base weights loaded successfully.")
    else:
        print("⚠️ Base model not found! Starting from scratch (not recommended for fine-tuning).")
    
    model = model.to(DEVICE)
    model.train()
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    print(f"Starting fine-tuning for {EPOCHS} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
        # Save best specific to this fine-tuning
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, f"{OUTPUT_DIR}/best_model.pt")
            
    # Save final
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_model.pt")
    print(f"✅ Fine-tuning complete. Best model saved to {OUTPUT_DIR}/best_model.pt")

if __name__ == "__main__":
    train()
