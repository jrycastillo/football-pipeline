# H100 Server Migration Guide

**Target Hardware:** NVIDIA H100 (80GB VRAM)
**Expected Performance:** 3-5x faster than current setup
**Date:** 2026-01-30

---

## Overview

This guide covers migrating the Football Analysis Pipeline from the current environment to an NVIDIA H100-equipped server for production deployment.

---

## H100 Specifications & Advantages

### Hardware Specs

| Feature | Specification |
|---------|--------------|
| **GPU** | NVIDIA H100 SXM5 |
| **VRAM** | 80GB HBM3 |
| **CUDA Cores** | 16,896 |
| **Tensor Cores** | 528 (4th gen) |
| **Memory Bandwidth** | 3.35 TB/s |
| **FP16 Performance** | 1,979 TFLOPS |
| **TDP** | 700W |

### Expected Performance Gains

| Task | Current | H100 | Speedup |
|------|---------|------|---------|
| **YOLO Detection** | ~3.5 FPS | ~12-15 FPS | **3-4x** |
| **Tracking** | ~4 FPS | ~15-20 FPS | **4-5x** |
| **JNR (ResNet)** | ~10 FPS | ~30-40 FPS | **3-4x** |
| **JNR (Qwen VLM)** | ~1.5 FPS | ~4-6 FPS | **3-4x** |
| **Full Match (10min)** | ~45-60 min | ~12-18 min | **3-4x** |

### Optimal Configuration for H100

```yaml
# config.yaml optimized for H100

heuristics:
  DET_IMG_SIZE: 1280      # Increased from 832 (H100 can handle it)
  VID_STRIDE: 1           # Process every frame (no skipping needed)
  JNR_STRIDE: null        # Process all frames with JNR

# Can run 5-6 parallel workers instead of 3
```

---

## Pre-Migration Checklist

### 1. Backup Current System

```bash
# Backup database
mysqldump -h <host> -u <user> -p footballgallery > backup_$(date +%Y%m%d).sql

# Backup code (if not in git)
tar -czf pipeline_backup_$(date +%Y%m%d).tar.gz \
    orchestrator.py \
    pipeline_consolidated.py \
    vision/ \
    stats/ \
    utils/ \
    config.yaml \
    .env

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### 2. Document Current Performance

```bash
# Record baseline metrics
python orchestrator.py \
    --local_video test_full_match.mp4 \
    --save_local \
    --no_db

# Note:
# - Total processing time
# - Average FPS
# - GPU memory usage
# - Any errors
```

### 3. Prepare Migration Package

```bash
# Run cleanup
./cleanup_for_production.sh

# Create deployment package
git add .
git commit -m "Production-ready: Cleaned codebase for H100 migration"
git push origin main

# Tag release
git tag -a v1.0-h100-ready -m "Production deployment ready for H100"
git push origin v1.0-h100-ready
```

---

## H100 Server Setup

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (latest)
# Check latest version: https://www.nvidia.com/Download/index.aspx
sudo apt install -y nvidia-driver-535  # Or latest version
sudo reboot

# Verify driver
nvidia-smi
# Should show H100 GPU with 80GB memory
```

### Step 2: CUDA Toolkit Installation

```bash
# Install CUDA 12.1 (or latest compatible)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Step 3: cuDNN Installation

```bash
# Download cuDNN 8.9 for CUDA 12.x from:
# https://developer.nvidia.com/cudnn

# Install
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y libcudnn8 libcudnn8-dev
```

### Step 4: Python Environment

```bash
# Install Python 3.10+
sudo apt install -y python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
# Should output: CUDA Available: True, Device: NVIDIA H100...
```

---

## Code Deployment

### Step 1: Clone Repository

```bash
# Clone from repo
git clone <repo-url> /home/ubuntu/football-pipeline
cd /home/ubuntu/football-pipeline

# Checkout production tag
git checkout v1.0-h100-ready

# Verify
git log --oneline -5
```

### Step 2: Install Dependencies

```bash
# Activate environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python check_setup.py
```

### Step 3: Transfer Models

```bash
# Create models directory
mkdir -p models

# Option A: Download from storage
scp user@backup-server:/path/to/models/*.pt models/

# Option B: Download from cloud
# gsutil cp gs://bucket/models/*.pt models/
# aws s3 cp s3://bucket/models/ models/ --recursive

# Verify models
ls -lh models/
# Should see:
# - yolo_player.pt (~50MB)
# - yolo_ball.pt (~20MB)
# - yolo_pitch.pt (~45MB)
# - resnet34_rgb_jnr.pt (~85MB)
```

### Step 4: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Update with production credentials
nano .env

# Test database connection
python test_db_connection.py
```

---

## H100-Specific Optimizations

### 1. Update config.yaml for H100

```yaml
# config.yaml - H100 Optimized

heuristics:
  # Detection - Can handle larger images
  DET_CONF: 0.10
  DET_IOU: 0.50
  DET_IMG_SIZE: 1280        # ↑ Increased from 832
  VID_STRIDE: 1             # Process every frame

  # JNR - More frequent processing
  JNR_IMG_SIZE: 224         # ↑ Increased from 160 for better accuracy
  JNR_GATE: 0.35
  JNR_STRIDE: 1             # ↑ Process every frame (was 5)

  # Tracking
  MAX_TRACK_FRAMES: null    # No limit
  STORE_IMAGES: 1
  STORE_IMAGES_UP_TO: 3000  # ↑ Increased from 1500

  # Stats
  FPS: 25
```

### 2. Optimize Systemd Service

```ini
# deployment/football-pipeline.service - H100 Optimized

[Service]
# Increase parallel workers (H100 can handle more)
ExecStart=/home/ubuntu/football-pipeline/venv/bin/python orchestrator.py \
    --poll \
    --poll_interval 30 \
    --parallel 5 \
    --locking_mode 2 \
    --tracking_mode bytetrack

# Resource limits for H100
MemoryMax=60G
CPUQuota=800%

# Environment
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
```

### 3. Enable Mixed Precision (FP16)

The pipeline already uses FP16 for VLM models. Verify it's enabled:

```python
# In vision/hybrid_recognition.py (already configured)
self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    self.vl_model_path,
    torch_dtype=torch.float16,  #  Already using FP16
    device_map="auto"
)
```

### 4. Enable Flash Attention 2

```python
# In vision/hybrid_recognition.py (already configured)
_attn_implementation="flash_attention_2" if is_cuda_available() else "eager"
```

---

## Testing on H100

### Phase 1: Smoke Test

```bash
# Quick test with 100 frames
python orchestrator.py \
    --local_video test_sample.mp4 \
    --save_local \
    --max_frames 100 \
    --no_db

# Expected: Completes in <30 seconds
# Monitor GPU:
watch -n 1 nvidia-smi
```

### Phase 2: Performance Benchmark

```bash
# Full 10-minute match
time python orchestrator.py \
    --local_video test_full_match.mp4 \
    --output_dir output/h100_benchmark \
    --locking_mode 2

# Record metrics:
# - Total time (target: <18 minutes)
# - Average FPS (target: >12 FPS)
# - GPU utilization (target: 70-90%)
# - Memory usage (target: <40GB)
```

### Phase 3: Parallel Processing

```bash
# Test with 5 parallel workers
python orchestrator.py \
    --poll \
    --parallel 5 \
    --max_videos 5 \
    --poll_interval 30

# Monitor:
# - GPU memory (should stay <70GB)
# - All workers completing successfully
# - No CUDA OOM errors
```

### Phase 4: Stress Test

```bash
# Process 20 videos back-to-back
python orchestrator.py \
    --poll \
    --parallel 5 \
    --max_videos 20

# Monitor for:
# - Memory leaks
# - Consistent performance
# - Error rates <1%
```

---

## Performance Monitoring

### GPU Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU stats every 5 seconds
while true; do
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader >> gpu_stats.log
    sleep 5
done
```

### Expected Metrics on H100

| Metric | Target Value |
|--------|--------------|
| **GPU Utilization** | 75-90% |
| **Memory Usage** | 30-50GB (with 5 workers) |
| **Processing FPS** | 12-18 FPS (detection + tracking) |
| **Full Match Time** | 12-18 minutes |
| **Throughput** | 15-20 videos/hour (with 5 workers) |
| **Power Consumption** | 400-600W |
| **Temperature** | <80°C |

---

## Optimization Tips

### 1. Batch Size Tuning

```python
# For YOLO detection, can increase batch size
# In pipeline_consolidated.py, experiment with:
results = model.predict(
    frame,
    conf=DET_CONF,
    iou=DET_IOU,
    imgsz=DET_IMG_SIZE,
    batch=4  # ↑ Increase from 1 (H100 can handle batches)
)
```

### 2. TensorRT Optimization (Optional)

For maximum performance, convert models to TensorRT:

```bash
# Export YOLO to TensorRT
yolo export model=models/yolo_player.pt format=engine device=0 half=True

# Update config to use .engine files
# DET_WEIGHTS: "models/yolo_player.engine"
```

### 3. Memory Management

```python
# In pipeline_consolidated.py, add periodic cleanup
import gc
import torch

# After processing each video
gc.collect()
torch.cuda.empty_cache()
```

---

## Troubleshooting H100

### Issue: GPU Not Detected

```bash
# Check driver
nvidia-smi

# Reinstall driver
sudo apt purge nvidia-* -y
sudo apt install nvidia-driver-535
sudo reboot
```

### Issue: CUDA Version Mismatch

```bash
# Check versions
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Should match (e.g., both 12.1)
# Reinstall PyTorch if mismatch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory (Even on H100)

```bash
# Reduce parallel workers
# In systemd service: --parallel 5 → --parallel 3

# Or reduce batch processing
# In config.yaml: DET_IMG_SIZE: 1280 → 832
```

### Issue: Lower Performance Than Expected

```bash
# Check power limit
nvidia-smi -q -d POWER

# Increase if capped
sudo nvidia-smi -pl 700  # Set to 700W (max for H100)

# Enable persistence mode
sudo nvidia-smi -pm 1

# Disable ECC (if acceptable for your use case)
sudo nvidia-smi -e 0
```

---

## Rollback Plan

If H100 migration encounters issues:

```bash
# 1. Stop service
sudo systemctl stop football-pipeline

# 2. Revert to previous configuration
git checkout <previous-version>

# 3. Restore config
cp config.yaml.backup config.yaml

# 4. Reduce parallel workers
# Edit deployment/football-pipeline.service
# --parallel 5 → --parallel 2

# 5. Restart
sudo systemctl start football-pipeline
```

---

## Success Criteria

**H100 migration is successful when:**

1. **Performance:** Full match processes in <18 minutes
2. **Throughput:** 15+ videos/hour with 5 workers
3. **Stability:** 24+ hours continuous operation without errors
4. **Utilization:** GPU at 75-90% during processing
5. **Memory:** <50GB usage with 5 parallel workers
6. **Success Rate:** >98% of videos processed successfully
7. **Quality:** Statistics match or exceed previous baseline
8. **Cost Efficiency:** 3x performance gain over previous setup

---

## Post-Migration

### Week 1: Intensive Monitoring

- [ ] Hourly performance checks
- [ ] Daily GPU health monitoring
- [ ] Thermal monitoring (ensure <80°C)
- [ ] Power consumption tracking
- [ ] Benchmark against previous system

### Optimization Opportunities

After stable operation:
- [ ] Test TensorRT conversion for YOLO models
- [ ] Experiment with larger batch sizes
- [ ] Try DET_IMG_SIZE 1536 or 1664
- [ ] Profile for further bottlenecks
- [ ] Consider model quantization (INT8)

---

## Cost-Benefit Analysis

### H100 Advantages

**3-4x faster processing**
**5x parallel workers (vs 3 previously)**
**Higher accuracy possible (larger input sizes)**
**Future-proof for larger models**
**Better energy efficiency per video**

### Considerations

**Higher power consumption (700W)**
**Requires CUDA 12.1+ and updated drivers**
**Need to tune configurations for optimal performance**

---

**Migration Status:** [ ] Not Started [→] In Progress [] Completed

**Next Steps:**
1. [ ] Run cleanup script
2. [ ] Commit to repo
3. [ ] Set up H100 server
4. [ ] Deploy code
5. [ ] Run benchmarks
6. [ ] Monitor for 48 hours
7. [ ] Optimize configuration

Last Updated: 2026-08-07
