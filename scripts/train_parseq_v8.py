"""
Train PARSeq JNR v7 — football jersey recognition without GT roster.

Dataset mix:
  1. SoccerNet JNR 2023 train: 1024 valid players (~733k crops) — European football
  2. Our auto-labeled crops: 65,742 crops from 9 match videos (conf >= 0.95)
     - Capped at MAX_PER_CLASS per number to prevent #36/#18 dominance
  3. Optional: Hamburg/Bayern crops if annotation CSV provided

Strategy:
  - Start from jersey-number-pipeline SoccerNet PARSeq checkpoint
  - Phase 1 (8 epochs):  encoder frozen, LR=1e-4
  - Phase 2 (30 epochs): full unfreeze, LR=5e-5 cosine
  - Heavy augmentation: motion blur, occlusion, perspective, color jitter, rotation

Output: models/parseq_local_v7.pt

Usage:
    python scripts/train_parseq_v7.py
    python scripts/train_parseq_v7.py --epochs_frozen 8 --epochs_full 30
"""
import csv, json, math, os, random, sys, time
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/ronan/work/football")
os.chdir(BASE)
sys.path.insert(0, str(BASE / "parseq"))

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from strhub.models.utils import create_model
from strhub.data.module import SceneTextDataModule
from torchvision import transforms
import string

CHARSET  = string.digits
IMG_SIZE = (32, 128)
MAX_PER_CLASS = 3000   # cap per jersey number to prevent imbalance

# ── Augmentation ──────────────────────────────────────────────────────────────

class MotionBlur:
    """Simulate motion blur from fast-moving players."""
    def __call__(self, img):
        if random.random() < 0.4:
            radius = random.choice([1, 2, 3])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class RandomOcclusion:
    """Randomly mask a horizontal band (simulates arm/ball occlusion)."""
    def __call__(self, img):
        w, h = img.size
        if random.random() < 0.3 and w >= 12 and h >= 12:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            y1 = random.randint(0, max(1, h // 2))
            y2 = y1 + random.randint(2, max(3, h // 3))
            x1 = random.randint(0, max(1, w // 3))
            x2 = random.randint(max(x1+1, 2 * w // 3), w)
            draw.rectangle([x1, y1, x2, min(y2, h)], fill=(random.randint(0,50),)*3)
        return img

AUGMENT = transforms.Compose([
    MotionBlur(),
    RandomOcclusion(),
    transforms.RandomChoice([
        transforms.GaussianBlur(3, sigma=(0.1, 2.5)),
        transforms.RandomGrayscale(p=1.0),
        transforms.Lambda(lambda x: x),
    ]),
    transforms.RandomPerspective(distortion_scale=0.30, p=0.5),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.12),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.12, 0.08), shear=8),
])

# ── Dataset ───────────────────────────────────────────────────────────────────

def torso_crop(img, top=0.15, bot=0.52):
    w, h = img.size
    y1, y2 = int(h * top), int(h * bot)
    if y2 - y1 < 6 or w < 6:
        return img
    return img.crop((0, y1, w, y2))

class JerseyDataset(Dataset):
    def __init__(self, samples, augment=False):
        # samples: list of (path, label_str)
        self.samples = samples
        self.augment = augment
        self.base_tf = SceneTextDataModule.get_transform(IMG_SIZE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img  = Image.open(path).convert("RGB")
            crop = torso_crop(img)
            if crop.size[0] < 4 or crop.size[1] < 4:
                crop = img
        except Exception:
            crop = Image.new("RGB", (64, 32), (100, 100, 100))
        if self.augment:
            crop = AUGMENT(crop)
        return self.base_tf(crop), label

def collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_soccernet(split="train"):
    """Load SoccerNet JNR 2023 crops. Returns list of (path, label_str)."""
    sn_dir = BASE / f"data/soccernet_jnr/jersey-2023/{split}"
    gt     = json.load(open(sn_dir / f"{split}_gt.json"))
    samples = []
    per_class = defaultdict(int)
    for player_id, jersey_num in gt.items():
        if jersey_num == -1 or not (1 <= jersey_num <= 99):
            continue
        label = str(jersey_num)
        img_dir = sn_dir / "images" / player_id
        if not img_dir.exists():
            continue
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        random.shuffle(imgs)
        for p in imgs:
            if per_class[label] >= MAX_PER_CLASS:
                break
            samples.append((str(p), label))
            per_class[label] += 1
    print(f"  SoccerNet {split}: {len(samples)} crops, {len(per_class)} unique numbers")
    return samples

def load_our_crops():
    """Load Hamburg/Bayern crops extracted from test_gdrive (verified labels via v6 pipeline)."""
    manifest = BASE / "data/annotations/jersey_crops_hb.csv"
    if not manifest.exists():
        print("  Hamburg/Bayern crops not yet extracted — skipping")
        return []
    rows = list(csv.DictReader(open(manifest)))
    per_class = defaultdict(list)
    for r in rows:
        p = Path(r["path"])
        if p.exists():
            per_class[r["label"]].append(str(p))

    samples = []
    for label, paths in per_class.items():
        random.shuffle(paths)
        for p in paths[:MAX_PER_CLASS]:
            samples.append((p, label))

    print(f"  Hamburg/Bayern crops: {len(samples)} crops, {len(per_class)} unique numbers")
    return samples

def load_extra_csv(csv_path):
    """Load additional annotated crops from a CSV with columns: path, label."""
    if not Path(csv_path).exists():
        return []
    rows = list(csv.DictReader(open(csv_path)))
    samples = []
    for r in rows:
        if r.get("label") in ("", "skip"):
            continue
        p = Path(r["path"])
        if not p.exists():
            p = BASE / r["path"]
        if p.exists():
            samples.append((str(p), str(r["label"])))
    print(f"  Extra CSV {csv_path}: {len(samples)} crops")
    return samples

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(checkpoint):
    ckpt      = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_key = next(iter(ckpt))
    model     = create_model("parseq", pretrained=False,
                             charset_train=CHARSET, charset_test=CHARSET,
                             max_label_length=2)
    if first_key.startswith("model."):
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded state_dict from {checkpoint}")
    else:
        skip = {"model.head.weight", "model.head.bias",
                "model.text_embed.embedding.weight", "model.pos_queries"}
        remapped = {f"model.{k}": v for k, v in ckpt.items() if f"model.{k}" not in skip}
        model.load_state_dict(remapped, strict=False)
        print(f"  Loaded backbone ({len(remapped)} weights) from {checkpoint}")
    return model

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = logits[:, :3, :11].softmax(-1)
            preds, _ = model.tokenizer.decode(probs)
            for pred, gt in zip(preds, labels):
                if pred.strip() == gt.strip():
                    correct += 1
                total += 1
    return correct / total * 100 if total else 0.0

# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Build dataset
    print("Loading datasets...")
    sn_train  = load_soccernet("train")
    our_crops = load_our_crops()
    extra     = load_extra_csv(args.extra_csv) if args.extra_csv else []

    all_samples = sn_train + our_crops + extra
    random.shuffle(all_samples)

    # 90/10 train/val split
    cut = int(len(all_samples) * 0.9)
    train_samples = all_samples[:cut]
    val_samples   = all_samples[cut:]

    print(f"\nTotal: {len(all_samples)} crops")
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    # Class counts for weighted sampler (balance training)
    label_counts = defaultdict(int)
    for _, lbl in train_samples:
        label_counts[lbl] += 1
    weights = [1.0 / label_counts[lbl] for _, lbl in train_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)

    train_ds  = JerseyDataset(train_samples, augment=True)
    val_ds    = JerseyDataset(val_samples,   augment=False)
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                           num_workers=4, collate_fn=collate, drop_last=True)
    val_ldr   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                           num_workers=2, collate_fn=collate)

    print(f"\nLoading base model from {args.checkpoint}...")
    model = load_model(args.checkpoint).to(device)

    # Phase 1: freeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    print(f"  Encoder frozen for first {args.epochs_frozen} epochs\n")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_frozen, weight_decay=0.01
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    total_epochs = args.epochs_frozen + args.epochs_full
    print(f"Training: {args.epochs_frozen} frozen + {args.epochs_full} full = {total_epochs} epochs\n")

    for epoch in range(1, total_epochs + 1):

        if epoch == args.epochs_frozen + 1:
            print(f"\nEpoch {epoch}: Unfreezing encoder — full fine-tune lr={args.lr_full:.0e}")
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=0.01)

        if epoch > args.epochs_frozen:
            progress = (epoch - args.epochs_frozen) / max(args.epochs_full, 1)
            lr = 1e-7 + 0.5 * (args.lr_full - 1e-7) * (1 + math.cos(math.pi * progress))
            for pg in opt.param_groups:
                pg["lr"] = lr

        model.train()
        total_loss = total = 0
        t0 = time.time()
        for imgs, labels in train_ldr:
            imgs = imgs.to(device)
            opt.zero_grad()
            loss_out = model.training_step((imgs, labels), 0)
            loss = loss_out if isinstance(loss_out, torch.Tensor) else loss_out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        avg_loss = total_loss / total
        val_acc  = evaluate(model, val_ldr, device)
        lr_now   = opt.param_groups[0]["lr"]
        phase    = "frz" if epoch <= args.epochs_frozen else "ful"
        marker   = " *" if val_acc > best_val else ""
        print(f"Epoch {epoch:2d}/{total_epochs} [{phase}] | "
              f"loss {avg_loss:.4f} | val_acc {val_acc:.1f}% | "
              f"lr {lr_now:.1e} | {time.time()-t0:.0f}s{marker}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out)
            print(f"  -> Saved best (val {val_acc:.1f}%) to {out}")

    print(f"\nDone. Best val accuracy: {best_val:.1f}% -> {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="jersey-number-pipeline/str/parseq/models/parseq-bb5792a6.pt",
                        help="Base checkpoint (default: SoccerNet PARSeq)")
    parser.add_argument("--output",        default="models/parseq_local_v8.pt")
    parser.add_argument("--extra_csv",     default="",
                        help="Optional extra annotated CSV (path,label columns)")
    parser.add_argument("--epochs_frozen", type=int,   default=8)
    parser.add_argument("--epochs_full",   type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr_frozen",     type=float, default=1e-4)
    parser.add_argument("--lr_full",       type=float, default=5e-5)
    args = parser.parse_args()
    train(args)
