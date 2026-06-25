# SoccerNet JNR Dataset Download Instructions

**Date:** 2026-03-27
**Dataset:** SoccerNet Jersey Number Recognition (JNR) 2023
**Purpose:** Download the dataset used to train the current ResNet34 jersey recognition model

---

## Prerequisites

- Python 3.8+ with pip
- ~10-15 GB free disk space
- Stable internet connection (dataset is ~5-10 GB)

---

## Step 1: Install SoccerNet Package

```bash
pip install SoccerNet
```

**Expected output:**
```
Successfully installed SoccerNet-x.x.x
```

---

## Step 2: Download the Dataset

Create a Python script or run in a Jupyter notebook:

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

# Set your local directory (adjust path as needed)
local_path = "/Users/ronan/Babak/data/soccernet_jnr"

# Initialize downloader
downloader = SNdl(LocalDirectory=local_path)

# Download jersey-2023 dataset (all splits)
downloader.downloadDataTask(
    task="jersey-2023",
    split=["train", "test", "challenge"]
)
```

**Alternative: Download only training data (faster, smaller)**

```python
# Only download train split (~70% of data)
downloader.downloadDataTask(
    task="jersey-2023",
    split=["train"]
)
```

---

## Step 3: Verify Download

After download completes, verify the structure:

```bash
ls -lh /Users/ronan/Babak/data/soccernet_jnr/jersey-2023/
```

**Expected structure:**

```
jersey-2023/
├── train/
│   ├── train_gt.json          # Ground truth labels
│   └── images/
│       ├── 1/                  # Player ID folders
│       │   ├── 001.jpg         # Player thumbnails
│       │   ├── 002.jpg
│       │   └── ...
│       ├── 2/
│       └── ...
├── test/
│   ├── test_gt.json
│   └── images/
│       └── ...
└── challenge/
    └── images/
        └── ... (no ground truth provided)
```

---

## Step 4: Check Dataset Size

```bash
# Check total images in train split
find /Users/ronan/Babak/data/soccernet_jnr/jersey-2023/train/images -name "*.jpg" | wc -l

# Check disk usage
du -sh /Users/ronan/Babak/data/soccernet_jnr/jersey-2023/
```

**Expected:**
- **Train split:** ~500,000-600,000 images
- **Total size:** ~5-8 GB

---

## Dataset Details

| Parameter | Value |
|-----------|-------|
| **Total tracklets** | 2,853 player tracklets |
| **Total images** | ~733,000 across all splits |
| **Classes** | 100 (jersey numbers 0-99) |
| **Format** | RGB JPG images |
| **Labels** | JSON mapping player_id → jersey_number |
| **Unknown labels** | Marked as -1 (non-visible jersey) |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'SoccerNet'`

**Solution:**
```bash
pip install --upgrade SoccerNet
```

### Issue: Download is very slow

**Solution:** The dataset is hosted on remote servers. Download time depends on your internet speed:
- 100 Mbps: ~7-10 minutes
- 50 Mbps: ~15-20 minutes
- 10 Mbps: ~1-2 hours

You can download only the train split to speed up:
```python
downloader.downloadDataTask(task="jersey-2023", split=["train"])
```

### Issue: Disk space error

**Solution:** The full dataset requires ~10-15 GB. Free up space or download to an external drive:
```python
# Download to external drive
downloader = SNdl(LocalDirectory="/Volumes/ExternalDrive/soccernet_jnr")
```

### Issue: `ConnectionError` or `TimeoutError`

**Solution:**
1. Check internet connection
2. Retry the download (it should resume from where it stopped)
3. If persistent, try downloading one split at a time:
```python
# Download splits separately
downloader.downloadDataTask(task="jersey-2023", split=["train"])
# Wait for completion, then:
downloader.downloadDataTask(task="jersey-2023", split=["test"])
```

---

## Using the Dataset

Once downloaded, you can use it with the existing training script:

```bash
cd /Users/ronan/Babak
python archive_dev_20260130_115410/train_resnet34_jnr_rgb.py
```

**Note:** The training script expects the dataset at:
```
data/soccernet_jnr/jersey-2023/train/train
```

If you downloaded to a different location, update line 21 in `train_resnet34_jnr_rgb.py`:
```python
DATA_DIR = Path("/your/custom/path/jersey-2023/train/train")
```

---

## Quick Start Script

Save this as `download_soccernet_jnr.py`:

```python
#!/usr/bin/env python3
"""
Quick script to download SoccerNet JNR 2023 dataset
"""
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import os

# Configuration
LOCAL_DIR = "/Users/ronan/Babak/data/soccernet_jnr"
SPLITS = ["train", "test", "challenge"]  # or just ["train"] for faster download

def main():
    print(f"Downloading SoccerNet JNR 2023 to: {LOCAL_DIR}")
    print(f"Splits: {SPLITS}")
    print("-" * 60)

    # Create directory if needed
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Initialize downloader
    downloader = SNdl(LocalDirectory=LOCAL_DIR)

    # Download
    print("Starting download (this may take 10-60 minutes depending on connection)...")
    downloader.downloadDataTask(task="jersey-2023", split=SPLITS)

    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print(f"Dataset location: {LOCAL_DIR}/jersey-2023/")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python download_soccernet_jnr.py
```

---

## Official Resources

- **SoccerNet Website:** https://www.soccer-net.org/tasks/jersey-number-recognition
- **GitHub Repository:** https://github.com/SoccerNet/sn-jersey
- **PyPI Package:** https://pypi.org/project/SoccerNet/
- **Paper (CVPR 2024):** [A General Framework for Jersey Number Recognition in Sports Video](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)

---

## Contact

If you encounter issues with the SoccerNet package or dataset:
- Open an issue on GitHub: https://github.com/SoccerNet/sn-jersey/issues
- Contact the SoccerNet team via their website
