# JNR Model Retraining — Phase 1 Completion Report

**Date:** 2026-03-29
**Phase:** 1 — Infrastructure Setup
**Status:** Complete

---

## Objective

Set up the training infrastructure for retraining the ResNet34 Jersey Number Recognition (JNR) model. The goal is to improve jersey match rate from 35% (9/26 correct) to 60-75% by retraining on broadcast-quality data.

---

## Infrastructure Overview

```
Mac (M4 Pro)                VPS                     Worker (WSL2, RTX 3070)
  └── Tailscale ──────────── Tailscale ──────────── Tailscale (100.76.11.68)
        │                        │                        │
   Development              API Gateway             GPU Training
   Code editing             Redis job queue         Dataset preparation
   Pipeline testing         Claude Code CLI         Jersey cropping
   Report generation        Claude Code CLI         Annotation pipeline
                                                    Model training
                                                    Claude Code CLI
```

### Worker Specifications

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 5800X (8C/16T) |
| GPU | NVIDIA GeForce RTX 3070 8GB VRAM |
| RAM | 64 GB |
| OS | Ubuntu 24.04 (WSL2) |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| Python | 3.12 |

### Network

| Node | Tailscale IP | Role |
|------|-------------|------|
| Mac | Connected | Development, orchestration |
| VPS | 100.89.153.94 | API gateway, Redis queue |
| Worker | 100.76.11.68 | GPU training, data processing |

- SSH key authentication configured (Mac → Worker, passwordless)
- VPS has Redis running via Docker for job queue
- Claude Code CLI installed on both VPS and Worker for AI-assisted orchestration

### Software Stack (All Nodes)

| Software | VPS | Worker |
|----------|-----|--------|
| Node.js | v22.22.0 | v22.22.0 |
| npm | 10.9.4 | 10.9.4 |
| Claude Code CLI | v2.1.87 | v2.1.87 |
| Python | 3.12 | 3.12 |
| PyTorch | N/A | 2.5.1+cu121 |
| CUDA | N/A | 12.1 |

---

## Tasks Completed

### 1. Tailscale Mesh Network
- Tailscale installed on all 3 nodes (Mac, VPS, Worker)
- All nodes connected and reachable via Tailscale IPs
- SSH key auth set up for Mac → Worker (no password needed)

### 2. VPS Setup
- Python venv created with all dependencies
- Redis server running (Docker container)
- Ready to serve as API gateway / job coordinator

### 3. Worker Environment
- Python 3.12 venv at `~/work/football/.venv`
- PyTorch 2.5.1 with CUDA 12.1 verified
- RTX 3070 detected and functional
- All pip packages installed (torch, torchvision, ultralytics, opencv, etc.)

### 4. Repository Sync
- Worker cloned `football` repo from GitHub
- Branch: `production-v1.0` (same as Mac)
- Latest commit: `52ced0f` — synced with Mac
- Working tree clean, no divergence

### 5. Model Transfer
- `resnet34_rgb_jnr.pt` (82 MB) → `~/work/football/models/`
- This is the current JNR model we want to improve

### 6. Training Script Transfer
- `train_resnet34_jnr_rgb.py` → `~/work/football/archive_dev_20260130_115410/`
- Original training script, uses SoccerNet JNR 2023 dataset
- Config: ResNet34, batch=64, lr=1e-4, 30 epochs, 128x128 input, 100 classes

### 7. SoccerNet JNR 2023 Dataset Transfer
- Downloaded on Mac (2,046,214 images total, 11 GB)
- Transferred compressed zips via SCP over Tailscale (train.zip: 1.2 GB, test.zip: 1.0 GB)
- Extracted on worker: `~/work/football/data/soccernet_jnr/jersey-2023/`

| Split | Images | Size | Ground Truth Labels |
|-------|--------|------|-------------------|
| Train | 733,001 | 2.9 GB | 1,024 valid (jersey 0-99) |
| Test | 564,547 | 2.3 GB | Yes |
| **Total** | **1,297,548** | **5.1 GB** | — |

### 8. Claude Code CLI Installation
- Installed Node.js v22.22.0 and Claude Code CLI v2.1.87 on both VPS and Worker
- Claude Code enables AI-assisted orchestration directly on the machines:
  - **Worker:** Claude can manage data extraction, jersey cropping, annotation, and training runs
  - **VPS:** Claude can coordinate jobs, monitor progress, and manage the pipeline
- Authentication required on first use (`claude login` or `ANTHROPIC_API_KEY` env var)

### 9. GPU Training Test
- Ran quick test: 1 epoch, 1,000 samples, batch=64
- **Result: PASSED**

| Metric | Value |
|--------|-------|
| Throughput | ~7 batches/sec (~448 samples/sec) |
| GPU VRAM used | 0.93 GB (of 8 GB available) |
| Loss (1 epoch) | 4.19 |
| Accuracy (1 epoch) | 9.9% (expected for 100 classes, 1 epoch) |

---

## Worker Directory Structure

```
~/work/football/                        # Git repo (production-v1.0)
├── .venv/                              # Python venv (PyTorch 2.5.1+cu121)
├── models/
│   └── resnet34_rgb_jnr.pt             # Current JNR model (82 MB)
├── archive_dev_20260130_115410/
│   └── train_resnet34_jnr_rgb.py       # Training script
├── data/
│   └── soccernet_jnr/jersey-2023/
│       ├── train/ (733K images, 2.9 GB)
│       │   ├── train_gt.json
│       │   └── images/{0,1,10,...}/
│       └── test/ (564K images, 2.3 GB)
│           ├── test_gt.json
│           └── images/
├── pipeline_consolidated.py            # Main pipeline
├── vision/                             # Color classifier, JNR, ByteTrack
├── stats/                              # Event logic, metrics, xG
└── ...                                 # All other repo files
```

---

## Phase 2 Plan (Next)

**Goal:** Extract jersey crops from broadcast video footage to create training data that matches real inference conditions.

The worker will:
1. **Download videos** from Google Drive (via rclone or API)
2. **Run YOLO detection** on video frames to detect players
3. **Crop torso regions** (15-45% of bounding box height) — same crop the pipeline uses during inference
4. **Quality filter** — discard crops that are too small (<20px), blurry, or heavily occluded
5. **Auto-annotate** using the current JNR model for high-confidence predictions
6. **Flag uncertain crops** for human review

Expected yield: ~200K-400K usable crops per video, 5 videos = ~1-2M total crops.

### Why Custom Crops Matter

The current model was trained on SoccerNet JNR data (studio-quality tracklets). But during inference, the pipeline crops from broadcast footage with:
- Small player sizes (20-60px torso width)
- Motion blur from fast movement
- Compression artifacts from video encoding
- Variable camera angles and distances

Training on crops that match these conditions will close the domain gap and improve real-world accuracy.

---

## Risk/Issues

| Issue | Status | Notes |
|-------|--------|-------|
| Mac Tailscale (brew version) | Resolved | Switched to App Store version |
| SCP truncated files via expect | Resolved | Set up SSH keys for reliable transfer |
| `unzip` not installed on worker | Resolved | Installed via apt |
| Models/data in wrong directory | Resolved | Moved from `~/work/babak/` to `~/work/football/` |
| Node.js not installed (VPS/Worker) | Resolved | Installed Node.js v22.22.0 via NodeSource |
| npm global install permissions (Worker) | Resolved | Used sudo for global npm install |
| WSL not auto-starting on boot | Known | Must manually run `wsl` in Windows Terminal after reboot |

---

## Timeline vs Plan

| Phase | Planned | Actual | Status |
|-------|---------|--------|--------|
| 1. Infrastructure | 2-3 days | 2 days | Complete |
| 2. Data Extraction | 3-5 days | — | Next |
| 3. Annotation | 5-10 days | — | Pending |
| 4. Dataset Prep | 1-2 days | — | Pending |
| 5. Training | 2-3 days | — | Pending |
| 6. Validation | 1-2 days | — | Pending |

**On schedule.** Phase 1 completed within the planned 2-3 day window.
