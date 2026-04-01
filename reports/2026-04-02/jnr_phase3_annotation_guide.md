# JNR Retraining — Phase 3: Annotation Guide

**Date:** 2026-04-02
**Status:** Phase 2 complete (crop extraction done), Phase 3 starting

---

## What Has Been Done So Far

### Phase 1 — Infrastructure ✅
- Worker: RTX 3070, Ubuntu 24.04 WSL2, CUDA 12.1, PyTorch installed
- VPS: Orchestrator + Claude agent, SSH access to worker via Tailscale
- All 9 match videos downloaded to worker: `~/work/football/data/videos/For data extraction/`

### Phase 2 — Data Extraction ✅
- Script: `~/work/football/scripts/extract_jersey_crops.py`
- Extracted upper body crops (0–60% bbox height) from all 9 videos
- YOLO model used: `models/yolo_player.pt` (classes 1=GK, 2=player)
- Settings: `FRAME_STRIDE=10`, `MIN_CROP_SIZE=40`, `DETECTION_CONF=0.15`, blur check disabled
- Output: `~/work/football/data/jersey_crops/{video_name}/frame_XXXXXX_crop_XXXX.jpg`
- Expected yield: ~1.5 million total crops across 9 videos

**Key lesson from Phase 2:** Original torso-only crop (15–45% height) was too small.
Fixed to upper body (0–60% height) which shows the full jersey number clearly.

---

## Phase 3 — Annotation Plan

**Goal:** Label crops with jersey numbers (0–99) to build training dataset.
**Target:** 50,000–100,000 labeled crops (500–1,000 per jersey number).

### Recommended Approach: Semi-Automatic (Option A)

Three tiers based on model confidence:

| Tier | Confidence | Action | Volume |
|------|-----------|--------|--------|
| High | > 0.70 | Auto-label (trust current model) | ~60% of crops |
| Medium | 0.30–0.70 | Claude Vision reviews and suggests label | ~25% of crops |
| Low | < 0.30 | Cluster by track ID, human labels one per cluster | ~15% of crops |

---

## Step-by-Step Instructions for the Agent

### Step 1 — Run Current JNR Model on All Crops

```bash
ssh ronan@100.76.11.68
cd ~/work/football
source .venv/bin/activate

python scripts/run_jnr_on_crops.py \
  --crops_dir data/jersey_crops \
  --model_path models/resnet34_rgb_jnr.pt \
  --output_csv data/annotations/predictions.csv
```

This produces a CSV: `image_path, predicted_jersey, confidence`

**Script to create:** `scripts/run_jnr_on_crops.py`
- Load ResNet34 model (`models/resnet34_rgb_jnr.pt`)
- Process each crop image (resize to 128x128, normalize)
- Output: path, top-1 prediction (0–99), confidence score
- Expected throughput: ~500 images/sec on RTX 3070

### Step 2 — Split by Confidence Tier

```bash
python scripts/split_by_confidence.py \
  --predictions data/annotations/predictions.csv \
  --high_conf_csv data/annotations/auto_labeled.csv \
  --medium_conf_csv data/annotations/needs_review.csv \
  --low_conf_csv data/annotations/needs_human.csv \
  --high_threshold 0.70 \
  --low_threshold 0.30
```

### Step 3 — Auto-Label High Confidence Crops

High confidence (>0.70): accept prediction as ground truth label directly.

```bash
python scripts/accept_high_confidence.py \
  --input data/annotations/auto_labeled.csv \
  --output data/annotations/labeled_final.csv
```

### Step 4 — Claude Vision for Medium Confidence

Send medium-confidence crops to Claude Vision API to read the jersey number from the image.

```bash
python scripts/annotate_with_claude_vision.py \
  --input data/annotations/needs_review.csv \
  --output data/annotations/claude_reviewed.csv \
  --batch_size 50
```

**How it works:**
- Load each crop image
- Send to Claude API with prompt: *"What jersey number is visible on this player? Reply with just the number (0-99) or 'unknown' if not readable."*
- Save Claude's answer as the label
- Cost estimate: ~$0.001 per image × 50,000 = ~$50

### Step 5 — Human Review of Low Confidence

Low confidence crops should be grouped by track ID (same player across frames).
Label one crop per track cluster → propagate label to all crops in that cluster.

Use a simple viewer script:

```bash
python scripts/review_low_confidence.py \
  --input data/annotations/needs_human.csv \
  --output data/annotations/human_labeled.csv
```

Shows crops in terminal or saves grid images to `data/annotations/review_grids/`.
Human types the jersey number for each cluster.

### Step 6 — Merge All Labels

```bash
python scripts/merge_annotations.py \
  --auto data/annotations/labeled_final.csv \
  --claude data/annotations/claude_reviewed.csv \
  --human data/annotations/human_labeled.csv \
  --output data/annotations/full_dataset.csv
```

### Step 7 — Quality Check

```bash
python scripts/check_dataset_balance.py \
  --input data/annotations/full_dataset.csv
```

Check:
- Total labeled crops
- Distribution per jersey number (0–99)
- Flag jersey numbers with < 100 crops (may need augmentation)

---

## Key Files to Read for Context

| File | Why Read It |
|------|-------------|
| `reports/2026-03-27/jnr_retraining_plan.md` | Full Phase 1–6 plan with timelines and cost estimates |
| `reports/2026-03-29/jnr_retraining_phase1_report.md` | Phase 1 completion report |
| `scripts/extract_jersey_crops.py` | Phase 2 extraction script (understand crop format) |
| `vision/resnet_recognition.py` | Current JNR model — how it loads and runs inference |
| `CLAUDE.md` | Full pipeline architecture and model details |

---

## Where Everything Lives on the Worker

```
~/work/football/
├── data/
│   ├── videos/For data extraction/   # 9 source videos (~21GB)
│   ├── jersey_crops/                 # Phase 2 output (~1.5M crops)
│   │   ├── COSAFA Cup 2024 - Angola vs Namibia .../
│   │   ├── COSAFA U17 COMOROS v ESWATINI .../
│   │   ├── The Gambia vs Madagascar .../
│   │   ├── U20 AFCON - LIBYA VS TUNISIA .../
│   │   ├── World Cup Qualifiers Benin vs Nigeria .../
│   │   ├── ASN NPFL IKORODU CITY v PLATEAU UNITED .../
│   │   ├── Argentina vs Nigeria .../
│   │   ├── Algeria vs Morocco .../
│   │   └── Tunisia vs Algeria .../
│   └── annotations/                  # Phase 3 output (create this)
├── models/
│   ├── yolo_player.pt               # Used in Phase 2
│   └── resnet34_rgb_jnr.pt          # Current JNR model for Phase 3
└── scripts/
    └── extract_jersey_crops.py      # Phase 2 script
```

---

## Scripts That Need to Be Created for Phase 3

| Script | Purpose |
|--------|---------|
| `scripts/run_jnr_on_crops.py` | Run JNR model on all crops, output confidence CSV |
| `scripts/split_by_confidence.py` | Split CSV into high/medium/low tiers |
| `scripts/accept_high_confidence.py` | Auto-label high confidence predictions |
| `scripts/annotate_with_claude_vision.py` | Claude Vision API annotation |
| `scripts/review_low_confidence.py` | Human review interface |
| `scripts/merge_annotations.py` | Combine all label sources |
| `scripts/check_dataset_balance.py` | Check class distribution |

---

## Success Criteria for Phase 3

- [ ] At least 50,000 labeled crops total
- [ ] At least 100 crops per jersey number for numbers 1–25 (most common in football)
- [ ] Less than 5% label error rate (verified by spot-check)
- [ ] Labels saved as CSV: `image_path, jersey_number, confidence_source`
  - `confidence_source` = `auto`, `claude_vision`, or `human`

Once complete, proceed to **Phase 4: Dataset Preparation** (deduplication, augmentation, train/val/test split).
