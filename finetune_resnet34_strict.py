
import os
import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

# Add current directory to path so we can import vision module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vision.resnet_recognition import create_resnet34_grayscale

# Configuration
DATA_DIR = "/home/ubuntu/football/data/manual_label_crops"
BASE_MODEL_PATH = "output_models/resnet34_jnr_grayscale/best_model.pt"
OUTPUT_DIR = "output_models/resnet34_jnr_manual_strict"
IMAGE_SIZE = 128
BATCH_SIZE = 16 # Smaller batch for small dataset
EPOCHS = 50
LEARNING_RATE = 5e-5 # Increased from 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StrictManualCropsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Robust Regex: Matches crop_X_Y.jpg where Y is 1-2 digits
        # Explicitly excludes "LABEL" or non-digits
        name_pattern = re.compile(r'crop_\d+_(\d+)\.jpg$')
        
        all_files = list(self.root_dir.glob("*.jpg"))
        print(f"Directory scan: Found {len(all_files)} total .jpg files")
        
        ignored_count = 0
        label_dist = {}
        
        for f in all_files:
            match = name_pattern.search(f.name)
            if match:
                label_str = match.group(1)
                try:
                    label = int(label_str)
                    if 0 <= label <= 99:
                        self.samples.append((f, label))
                        label_dist[label] = label_dist.get(label, 0) + 1
                    else:
                        print(f"Skipping Out-of-Range Label: {f.name} (Label: {label})")
                        ignored_count += 1
                except ValueError:
                    ignored_count += 1
            else:
                # print(f"Skipping Invalid Format: {f.name}")
                ignored_count += 1
        
        print(f"Loaded {len(self.samples)} valid samples.")
        print(f"Ignored {ignored_count} invalid samples (including *_LABEL.jpg).")
        print(f"Unique classes found: {len(label_dist)}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L') # Force Grayscale
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset with Split
    full_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Augmentations for Train only (applied to dataset, but we split dataset object... 
        # ideally we use separate transforms for val, but for simplicity we rely on moderate augs)
        transforms.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Validation Transform (No Augmentation ideally, but using same dataset obj makes it hard.
    # We will accept minor noise in Val or implement subset wrapper)
    
    full_dataset = StrictManualCropsDataset(DATA_DIR, transform=full_transform)
    
    # 80:20 Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Split: {train_size} Train | {val_size} Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = create_resnet34_grayscale(num_classes=100)
    
    if os.path.exists(BASE_MODEL_PATH):
        checkpoint = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Base weights loaded.")
    else:
        print("⚠️ Base model NOT found.")

    model = model.to(DEVICE)
    
    # 3. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    
    print(f"Starting Training: {EPOCHS} Epochs, LR={LEARNING_RATE}")
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / train_total
        
        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / val_total
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'args': {'lr': LEARNING_RATE}
            }, f"{OUTPUT_DIR}/best_model.pt")
            print(f"⭐ New Best Model (Val Acc: {val_acc:.1f}%)")

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.1f}%")

if __name__ == "__main__":
    train()
