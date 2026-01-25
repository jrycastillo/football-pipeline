"""
Fine-tune ResNet34 (RGB) on Manual Real Crops (Strict)
- RGB input for full color context
- Loads base weights from Phase 226
- Robust validation split (80/20)
"""
import os
import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# Configuration
DATA_DIR = "/home/ubuntu/football/data/manual_label_crops"
BASE_MODEL_PATH = "output_models/resnet34_jnr_rgb/best_model.pt"
OUTPUT_DIR = "output_models/resnet34_jnr_manual_rgb_strict"
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StrictManualCropsDatasetRGB(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        name_pattern = re.compile(r'crop_\d+_(\d+)\.jpg$')
        
        all_files = list(self.root_dir.glob("*.jpg"))
        print(f"Directory scan: Found {len(all_files)} total .jpg files")
        
        ignored_count = 0
        
        for f in all_files:
            match = name_pattern.search(f.name)
            if match:
                label_str = match.group(1)
                try:
                    label = int(label_str)
                    if 0 <= label <= 99:
                        self.samples.append((f, label))
                    else:
                        ignored_count += 1
                except ValueError:
                    ignored_count += 1
            else:
                ignored_count += 1
        
        print(f"Loaded {len(self.samples)} valid samples.")
        print(f"Ignored {ignored_count} invalid samples.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def create_resnet34_rgb(num_classes=100):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = StrictManualCropsDatasetRGB(DATA_DIR, transform=train_transform)
    
    # Splitting logic with separate Transforms via re-instantiation not needed for small dataset hack
    # We will just apply val transform to val set by wrapping or hacking.
    # Actually, let's just use the dataset split and accept that `full_dataset` has train_transform
    # For validation, we manually handle it or create two datasets.
    # Creating two datasets is safer.
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Deterministic Split
    generator = torch.Generator().manual_seed(42)
    
    # Train Set
    ds_train = StrictManualCropsDatasetRGB(DATA_DIR, transform=train_transform)
    train_dataset, _ = random_split(ds_train, [train_size, val_size], generator=generator)
    
    # Val Set
    ds_val = StrictManualCropsDatasetRGB(DATA_DIR, transform=val_transform)
    _, val_dataset = random_split(ds_val, [train_size, val_size], generator=generator)
    
    print(f"Split: {len(train_dataset)} Train | {len(val_dataset)} Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model Loading
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model = create_resnet34_rgb(num_classes=100)
    
    if os.path.exists(BASE_MODEL_PATH):
        try:
            checkpoint = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Base RGB weights loaded.")
        except Exception as e:
             print(f"⚠️ Error loading weights: {e}")
    else:
        print("⚠️ Base model NOT found. Starting from scratch (ImageNet strict initialization? No, random + fc).")
        # Initialize with ImageNet if base missing?
        # Ideally we want base.
        print("Initializing with ImageNet weights as fallback...")
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 100)

    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Low LR for Fine-Tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    
    print(f"Starting Training: {EPOCHS} Epochs")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"   ✅ Saved Best ({val_acc:.1f}%)")

if __name__ == "__main__":
    train()
