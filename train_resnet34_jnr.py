"""
Train ResNet34 on SoccerNet JNR Dataset
- Grayscale input for color invariance
- Strong augmentations for domain robustness
- Pretrained ImageNet weights (adapted for 1 channel)
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
OUTPUT_DIR = Path("output_models/resnet34_jnr_grayscale")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_CLASSES = 100  # Jersey numbers 0-99
IMAGE_SIZE = 128
NUM_WORKERS = 4

# ============= Dataset =============
class SoccerNetJNRDataset(Dataset):
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
        print(f"Label distribution (top 10): {label_counts.most_common(10)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("L")  # Convert to grayscale
        label = sample["label"]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============= Model =============
def create_resnet34_grayscale(num_classes=100, pretrained=True):
    """Create ResNet34 adapted for grayscale input."""
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Modify first conv layer for 1 channel input
    # Average the pretrained RGB weights to create grayscale weights
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    if pretrained:
        # Initialize from pretrained by averaging RGB channels
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
    
    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


# ============= Training =============
def train():
    print("=" * 60)
    print("Training ResNet34 on SoccerNet JNR (Grayscale)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Dataset
    print("\n1. Loading dataset...")
    full_dataset = SoccerNetJNRDataset(DATA_DIR, transform=train_transform)
    
    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update val transform
    val_dataset.dataset.transform = val_transform
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Model
    print("\n2. Creating model...")
    model = create_resnet34_grayscale(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with class weights
    class_weights = full_dataset.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    print(f"\n3. Training for {EPOCHS} epochs...")
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(EPOCHS):
        # Train
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
            
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*train_correct/train_total:.1f}%"
            })
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.1f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "grayscale": True
                }
            }, OUTPUT_DIR / "best_model.pt")
            print(f"   ✅ Saved best model (Val Acc: {val_acc:.1f}%)")
        
        scheduler.step()
    
    # Save final model
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "config": {
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "grayscale": True
        }
    }, OUTPUT_DIR / "final_model.pt")
    
    # Save history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"   Models saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
