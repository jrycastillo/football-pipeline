# JNR Model Retraining Plan

**Date:** 2026-03-27
**Author:** Babak AI Team
**Goal:** Retrain ResNet34 Jersey Number Recognition model to improve jersey match rate from 35% to 60-75%

---

## Current State

| Parameter | Value |
|-----------|-------|
| Model | ResNet34, RGB input, 128x128 |
| Classes | 100 (jersey #0-99) |
| Model size | 82MB |
| Current accuracy | 35% jersey match (9/26 detected match real lineups) |
| Training data | Unknown origin — likely public dataset, not fine-tuned on broadcast footage |
| Problem | Broadcast camera angles, small player crops (20-60px torso), motion blur, compression artifacts |

---

## Infrastructure

### Hardware (Local Workstation)

| Component | Spec | Role |
|-----------|------|------|
| CPU | Ryzen 7 5700X (8C/16T) | Data preprocessing, frame extraction |
| GPU | RTX 3070 8GB VRAM | Training (ResNet34 fits easily ~2GB) |
| RAM | 64GB | Dataset loading, batch processing |
| Storage | ~100-200GB needed for extracted crops | SSD recommended |

### Network Architecture

```
Google Drive (videos)
    → VPS (Tailscale gateway, Claude API)
    → Ryzen/3070 (processing + training)
```

- **VPS:** API gateway, receives Claude agent commands, forwards to workstation
- **Tailscale:** Mesh VPN connecting VPS ↔ workstation (secure, no port forwarding)
- **Claude (LLM):** Orchestration brain — decides what to extract, validates crop quality, generates scripts, monitors training, adjusts hyperparameters

### Agent Tools Required

| Tool | Purpose |
|------|---------|
| Google Drive API | Download videos |
| Shell execution | Run YOLO, crop, train |
| File system access | Read/write crops, labels |
| GPU monitoring | nvidia-smi, training logs |
| Claude Vision API | Annotation assist for medium-confidence crops |

---

## Phase-by-Phase Plan

### Phase 1: Infrastructure Setup (2-3 days)

| Task | Time | Details |
|------|------|---------|
| Install Tailscale on VPS + workstation | 2 hours | Mesh network, SSH tunnel |
| Set up Claude Agent SDK on VPS | 4 hours | Node.js/Python, API keys, tool definitions |
| Install training environment on workstation | 3 hours | CUDA 12.x, PyTorch 2.x, ultralytics |
| Google Drive API auth (service account) | 2 hours | OAuth2 or rclone mount |
| Test end-to-end: Claude → VPS → workstation → GPU | 4 hours | Verify latency, file transfer speed |

### Phase 2: Data Extraction (3-5 days)

The agent downloads videos and extracts jersey crops automatically.

| Task | Time | Details |
|------|------|---------|
| Download videos from GDrive | 4-8 hours | ~15GB total, depends on bandwidth |
| Run YOLO detection on all videos | 12-20 hours | ~5 videos x 50k frames each, stride=3 |
| Crop torso regions (15-45% height) | 2-3 hours | From YOLO bounding boxes, save as PNG |
| Quality filter (blur, size, occlusion) | 2-3 hours | Discard crops <20px, heavily occluded |

**Expected yield per video:**
- ~53,000 processed frames (stride=3)
- ~20 players visible per frame
- ~1,060,000 raw crops per video
- After quality filter: ~200,000-400,000 usable crops per video
- **5 videos total: ~1-2 million raw crops**

### Phase 3: Annotation (5-10 days — THE BOTTLENECK)

This is the hardest and most time-consuming phase. Three options:

#### Option A: Semi-Automatic (Recommended) — 5-7 days

1. Run current JNR model on all crops → get predictions with confidence
2. High-confidence crops (>0.70): auto-label (trust the model)
3. Medium-confidence (0.30-0.70): Claude Vision reviews crop images and suggests labels
4. Low-confidence (<0.30): cluster by track ID, manually label one crop per cluster, propagate to all crops in that track
5. Human verification: spot-check 5-10% of auto-labels

#### Option B: Fully Manual — 2-3 weeks

- Use a labeling tool (Label Studio, CVAT)
- Human annotates crops one by one
- ~50,000 crops to label = ~100 hours at 500/hour

#### Option C: Claude Vision Annotation — 7-10 days

- Send crop images to Claude Vision API in batches
- Claude reads the jersey number from each crop
- Cost: ~$0.01-0.03 per image x 50,000 images = $500-1,500
- Accuracy: ~70-80% (Claude struggles with same tiny/blurry crops the model does)

**Recommended: Option A** — uses the existing model for easy cases, Claude for medium cases, human for hard cases. Target: **50,000-100,000 labeled crops** (500-1,000 per jersey number).

### Phase 4: Dataset Preparation (1-2 days)

| Task | Time | Details |
|------|------|---------|
| Deduplicate near-identical crops | 3 hours | Perceptual hashing, remove >95% similar |
| Balance classes | 2 hours | Oversample rare numbers, undersample common |
| Train/val/test split (70/15/15) | 1 hour | Stratified by jersey number AND video source |
| Augmentation pipeline | 3 hours | Random crop, rotation, blur, brightness, compression artifacts |
| Generate manifest files | 1 hour | CSV: path, label, source_video, confidence |

**Target dataset size:** 50,000-100,000 labeled crops
- ~500-1,000 crops per jersey number (0-99)
- Some numbers will be sparse (60-99 are rare in football)

### Phase 5: Training (2-3 days)

| Task | Time | Details |
|------|------|---------|
| Baseline training (from scratch) | 6-8 hours | ResNet34, batch=64, lr=1e-3, 50 epochs |
| Fine-tune from current weights | 3-4 hours | lr=1e-4, 30 epochs, freeze early layers |
| Hyperparameter sweep | 8-12 hours | Learning rate, augmentation strength, input size (128 vs 224) |
| Evaluate on test set | 1 hour | Per-class accuracy, confusion matrix |
| Compare against current model on real videos | 4-6 hours | Run pipeline on g5718885, compare jersey accuracy |

**Training specs on RTX 3070:**

| Parameter | Value |
|-----------|-------|
| Throughput | ~200 samples/sec @ batch=64, 128x128 |
| Full training (100k samples x 50 epochs) | ~7 hours |
| VRAM usage | ~2-3GB |
| Fine-tuning (30 epochs) | ~4 hours |

### Phase 6: Validation & Deployment (1-2 days)

| Task | Time | Details |
|------|------|---------|
| Run full pipeline with new model on all test videos | 8-10 hours | Side-by-side comparison |
| Jersey match rate analysis | 2 hours | Compare detected vs real lineups |
| A/B test old vs new model | 4 hours | Same video, different weights |
| Deploy to production | 1 hour | Replace `resnet34_rgb_jnr.pt` |

---

## Overall Timeline

| Phase | Duration | Agent Automatable? |
|-------|----------|--------------------|
| 1. Infrastructure | 2-3 days | Partially (human sets up Tailscale/auth, agent tests) |
| 2. Data Extraction | 3-5 days | **Yes — fully automated** |
| 3. Annotation | 5-10 days | **Mostly — 70% auto, 20% Claude, 10% human** |
| 4. Dataset Prep | 1-2 days | **Yes — fully automated** |
| 5. Training | 2-3 days | **Yes — fully automated** |
| 6. Validation | 1-2 days | **Yes — fully automated** |
| **Total** | **2-4 weeks** | **~80% automated** |

**Critical path:** Annotation (Phase 3). Everything else is automatable.

---

## Cost Estimate

| Item | Cost |
|------|------|
| Claude API (agent orchestration) | ~$50-100/month |
| Claude Vision (annotation assist) | ~$200-500 (one-time) |
| VPS (gateway) | ~$5-20/month |
| Electricity (RTX 3070, ~2 weeks) | ~$10-20 |
| Google Drive storage | Already available |
| **Total** | **~$300-700** |

---

## Expected Improvement

| Metric | Current | Expected After Retrain |
|--------|---------|----------------------|
| Jersey match rate | 35% (9/26) | 60-75% |
| Players with correct # | 9 | 16-20 |
| Ghost players | 17 | 5-8 |

**Why not 90%+?** Broadcast footage has fundamental limitations — distant players are 15-30px wide, numbers are 5-10px tall. Even humans can't read them reliably. To reach 90%+ you would need:
- Multi-frame temporal voting (already implemented)
- Higher resolution input (224 or 256 instead of 128)
- Different architecture (CRNN or attention-based for digit recognition)
- Dedicated camera angles or higher-resolution source video

---

## Agent Capability Matrix

| Task | Can Agent Do It? | Notes |
|------|-----------------|-------|
| Download from GDrive | Yes | rclone or API |
| Run YOLO + crop | Yes | Shell commands |
| Auto-label with current model | Yes | Python script |
| Claude Vision annotation | Yes | API calls with images |
| Quality filtering | Yes | Script with blur/size checks |
| Build dataset | Yes | Python data pipeline |
| Launch training | Yes | PyTorch training script |
| Monitor training | Yes | Read logs, tensorboard |
| Evaluate results | Yes | Confusion matrix, accuracy |
| **Manual annotation review** | **No** | **Human needed for 10-20%** |

The agent can handle ~80% of the work end-to-end. The main human involvement is:
1. Initial infrastructure setup (Tailscale, auth keys)
2. Spot-checking and correcting annotations (~10-20% of labels)

---

## Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| Annotation quality too low | Model learns wrong labels | Human spot-check 10%, discard low-confidence labels |
| Class imbalance (rare jersey numbers) | Poor accuracy on #60-99 | Synthetic augmentation, oversample rare classes |
| Domain gap (training vs inference) | Model doesn't generalize | Include crops from ALL video sources in training set |
| RTX 3070 VRAM limit | Can't increase batch size | ResNet34 only uses 2-3GB — not a concern |
| Tailscale latency | Slow agent responsiveness | Agent runs local scripts, only reports back to Claude |
| Google Drive download speed | Bottleneck on data extraction | Download once, cache locally |
