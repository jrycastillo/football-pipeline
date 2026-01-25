"""
Train ResNet34 (RGB) on SoccerNet JNR Dataset
- RGB input for full color context (Red vs Green separation)
- Pretrained ImageNet weights (Standard Transfer Learning)
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
import cv2

# ============= Configuration =============
DATA_DIR = Path("data/soccernet_jnr/jersey-2023/train/train")
OUTPUT_DIR = Path("output_models/resnet34_jnr_rgb")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_CLASSES = 100  # Jersey numbers 0-99
IMAGE_SIZE = 128
NUM_WORKERS = 4

# ============= Dataset =============
class SoccerNetJNRDatasetRGB(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load ground truth: maps folder_id -> jersey_number
        gt_file = self.data_dir / "train_gt.json"
        with open(gt_file) as f:
            gt = json.load(f)
        
        # Convert string keys to int and filter valid labels (0-99)
        self.gt = {}
        for k, v in gt.items():
            if 0 <= v <= 99:  # Exclude -1 (unknown)
                self.gt[int(k)] = v
        
        print(f"Ground truth: {len(self.gt):,} valid labels (excluding unknowns)")
        
        # Scan all images recursively from images/{folder_id}/*.jpg
        self.samples = []
        images_dir = self.data_dir / "images"
        
        print("Scanning dataset (this may take a minute for 733K images)...")
        
        # Use rglob for recursive search
        all_files = list(images_dir.rglob("*.jpg"))
        print(f"Found {len(all_files):,} image files")
        
        for img_file in all_files:
            # Extract folder_id from parent folder name
            try:
                folder_id = int(img_file.parent.name)
                if folder_id in self.gt:
                    jersey_num = self.gt[folder_id]
                    self.samples.append({
                        "path": str(img_file),
                        "label": jersey_num
                    })
            except ValueError:
                continue
        
        # Shuffle
        import random
        random.seed(42)
        random.shuffle(self.samples)
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        # Compute class weights for imbalanced data
        labels = [s["label"] for s in self.samples]
        label_counts = Counter(labels)
        total = len(labels)
        self.class_weights = torch.zeros(NUM_CLASSES)
        for label, count in label_counts.items():
            # Inverse frequency weighting
            self.class_weights[label] = total / (NUM_CLASSES * count)
        
        print(f"Loaded {len(self.samples):,} samples with valid labels")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample["path"]).convert("RGB")  # Force RGB
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a blank image to avoid crashing (training loop should handle implies minor data loss is acceptable)
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
            
        label = sample["label"]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============= Model =============
def create_resnet34_rgb(num_classes=100, pretrained=True):
    """Create standard ResNet34 for RGB input."""
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


# ============= Training =============
def train():
    print("=" * 60)
    print("Training ResNet34 on SoccerNet JNR (RGB - Color Aware)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Transforms (Standard ImageNet Normalization)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05), # Added Saturation/Hue for RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    print("\n1. Loading dataset...")
    full_dataset = SoccerNetJNRDatasetRGB(DATA_DIR, transform=train_transform)
    
    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update val transform
    val_dataset.dataset.transform = val_transform # This hack might not work with Subset but relies on shared underlying dataset reference being weird.
    # Actually, random_split wraps in Subset. Subset doesn't have transform. The underlying dataset has transform.
    # To properly support separate transforms, we should create two dataset instances.
    # Re-instantiating is safer.
    
    print("   Reloading Validation Dataset separately to ensure correct transforms...")
    train_dataset_full = SoccerNetJNRDatasetRGB(DATA_DIR, transform=train_transform)
    val_dataset_full = SoccerNetJNRDatasetRGB(DATA_DIR, transform=val_transform)
    
    # Ensure consistent split/shuffle
    # We can use the same indices if we set seed
    generator = torch.Generator().manual_seed(42)
    train_ds, _ = torch.utils.data.random_split(train_dataset_full, [train_size, val_size], generator=generator)
    _, val_ds = torch.utils.data.random_split(val_dataset_full, [train_size, val_size], generator=generator)
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Model
    print("\n2. Creating model...")
    model = create_resnet34_rgb(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Loss 
    criterion = nn.CrossEntropyLoss() # Skipping class weights for now to let it learn generic features first from massive data
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Training
    print(f"\n3. Training for {EPOCHS} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*train_correct/train_total:.1f}%"})
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Results: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"   ✅ Saved Best Model ({val_acc:.1f}%)")
        
        scheduler.step(val_acc)

if __name__ == "__main__":
    train()
