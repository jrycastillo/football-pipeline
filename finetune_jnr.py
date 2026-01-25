#!/usr/bin/env python3
"""
Fine-tune ResNet32 JNR Model for Domain Generalization

Strategy:
1. Use existing real crops as primary training signal (with oversampling)
2. Apply Phase 217 aligned augmentations (blur, rotation, color jitter)
3. Lower learning rate for fine-tuning (1e-4)
4. Use label smoothing to prevent overconfidence

Usage:
    python finetune_jnr.py --crops_dir output/crops --epochs 10 --output output_models/resnet32_finetuned.pt
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import cv2


# ============ ResNet32 Architecture (Must match original) ============
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet32(num_classes=100):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


# ============ Custom Transforms (Phase 217 Aligned) ============
class SmartPadResize:
    """Letterbox pad to square, then resize."""
    def __init__(self, size=224):
        self.size = size
    
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        img_padded = ImageOps.pad(img, (max_dim, max_dim), color=(0, 0, 0))
        return img_padded.resize((self.size, self.size), Image.Resampling.BILINEAR)


class GaussianBlur:
    """Light Gaussian blur to match training augmentation."""
    def __init__(self, kernel_size=3, sigma=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        if random.random() < 0.7:  # Apply 70% of the time
            img_np = np.array(img)
            img_np = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)
            return Image.fromarray(img_np)
        return img


class CLAHE:
    """Contrast enhancement matching Phase 217."""
    def __init__(self, clip_limit=1.5, tile_size=4):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
    
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_np)


# ============ Dataset ============
class JerseyNumberDataset(Dataset):
    """Dataset for jersey number crops with labels from filenames."""
    
    def __init__(self, image_paths, labels, transform=None, oversample=1):
        self.image_paths = []
        self.labels = []
        
        # Apply oversampling
        for path, label in zip(image_paths, labels):
            for _ in range(oversample):
                self.image_paths.append(path)
                self.labels.append(label)
        
        self.transform = transform
        print(f"Dataset: {len(self.image_paths)} samples ({len(image_paths)} unique, {oversample}x oversample)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid sample
            return self.__getitem__(random.randint(0, len(self) - 1))


def load_labeled_crops(crops_dir: str):
    """
    Load crops from directory. Expects filenames like:
    - {id}.jpg where id is track ID (needs manual labeling)
    - OR a labels.json file mapping filenames to jersey numbers
    
    Returns: list of (path, label) tuples
    """
    crops_dir = Path(crops_dir)
    labels_file = crops_dir / "labels.json"
    
    samples = []
    
    if labels_file.exists():
        # Use labels file
        with open(labels_file) as f:
            labels = json.load(f)
        for filename, label in labels.items():
            path = crops_dir / filename
            if path.exists() and isinstance(label, int) and 0 <= label <= 99:
                samples.append((str(path), label))
    else:
        # Try to infer labels from any existing identification
        # For now, create pseudo-labels from high-confidence predictions
        print(f"No labels.json found in {crops_dir}")
        print("Will create pseudo-labels using current model predictions...")
        
    return samples


def create_pseudo_labels(crops_dir: str, model, device, transform, confidence_threshold=0.7):
    """
    Create pseudo-labels for unlabeled crops using model predictions.
    Only use high-confidence predictions.
    """
    crops_dir = Path(crops_dir)
    model.eval()
    
    samples = []
    image_files = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
    
    print(f"Creating pseudo-labels for {len(image_files)} crops...")
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Pseudo-labeling"):
            try:
                img = Image.open(img_path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)
                
                outputs = model(img_t)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                conf_val = conf.item()
                pred_val = pred.item()
                
                # Only use high-confidence predictions
                if conf_val >= confidence_threshold:
                    samples.append((str(img_path), pred_val, conf_val))
                    
            except Exception as e:
                continue
    
    print(f"Created {len(samples)} pseudo-labels (conf >= {confidence_threshold})")
    
    # Save for future use
    labels_dict = {
        Path(p).name: {"label": l, "confidence": c} 
        for p, l, c in samples
    }
    labels_file = crops_dir / "pseudo_labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels_dict, f, indent=2)
    print(f"Saved to {labels_file}")
    
    return [(p, l) for p, l, c in samples]


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet32 JNR")
    parser.add_argument("--crops_dir", default="output/crops", help="Directory with crop images")
    parser.add_argument("--base_weights", default="output_models/resnet32_recognition_v1.pt", 
                       help="Path to base model weights")
    parser.add_argument("--output", default="output_models/resnet32_finetuned_v1.pt",
                       help="Output path for fine-tuned weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--oversample", type=int, default=50, help="Oversample factor for real crops")
    parser.add_argument("--pseudo_conf", type=float, default=0.7, 
                       help="Confidence threshold for pseudo-labeling")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    model = resnet32(num_classes=100)
    if os.path.exists(args.base_weights):
        state_dict = torch.load(args.base_weights, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded base weights from {args.base_weights}")
    else:
        print(f"Warning: Base weights not found at {args.base_weights}")
    model = model.to(device)
    
    # Phase 217 aligned transforms for training
    train_transform = transforms.Compose([
        SmartPadResize(224),
        GaussianBlur(3, 0.5),  # Match training augmentation
        CLAHE(1.5, 4),  # Match Phase 217
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.3),  # Jersey numbers can be mirrored in some angles
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transform (minimal augmentation)
    val_transform = transforms.Compose([
        SmartPadResize(224),
        CLAHE(1.5, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load or create labeled crops
    samples = load_labeled_crops(args.crops_dir)
    
    if not samples:
        # Create pseudo-labels using model predictions
        samples = create_pseudo_labels(
            args.crops_dir, model, device, val_transform, 
            confidence_threshold=args.pseudo_conf
        )
    
    if not samples:
        print("No labeled samples available. Please provide labels.json or lower --pseudo_conf")
        return
    
    # Split into train/val (80/20)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")
    
    # Create datasets
    train_paths, train_labels = zip(*train_samples) if train_samples else ([], [])
    val_paths, val_labels = zip(*val_samples) if val_samples else ([], [])
    
    train_dataset = JerseyNumberDataset(
        list(train_paths), list(train_labels), 
        transform=train_transform, 
        oversample=args.oversample
    )
    val_dataset = JerseyNumberDataset(
        list(val_paths), list(val_labels),
        transform=val_transform,
        oversample=1  # No oversampling for validation
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True) if val_samples else None
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, 
        steps_per_epoch=len(train_loader)
    )
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.output)
                print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")
        else:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}%")
            # Save every epoch if no validation
            torch.save(model.state_dict(), args.output)
    
    # Final save
    torch.save(model.state_dict(), args.output)
    print(f"\nFine-tuning complete! Model saved to {args.output}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
