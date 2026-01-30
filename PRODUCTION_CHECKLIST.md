# Production Deployment Checklist

**Date:** 2026-01-30
**Target:** H100 Server Migration

---

## Pre-Deployment Cleanup

### ✅ Step 1: Run Cleanup Script

```bash
# Make executable
chmod +x cleanup_for_production.sh

# Run cleanup
./cleanup_for_production.sh

# Review archived files
ls -la archive_dev_*/
```

### ✅ Step 2: Verify Essential Files Present

**Core Pipeline:**
- [ ] `orchestrator.py` - Main entry point
- [ ] `pipeline_consolidated.py` - Core processing
- [ ] `config.yaml` - Configuration
- [ ] `requirements.txt` - Dependencies
- [ ] `.env.example` - Environment template
- [ ] `.gitignore` - Git ignore rules

**Directories:**
- [ ] `vision/` - Vision components (color classifier, tracking, JNR, ball tracking)
- [ ] `stats/` - Statistics engine
- [ ] `utils/` - Utility modules (health monitor, device utils)
- [ ] `models/` - Model weights directory (create if missing)
- [ ] `deployment/` - Systemd service files
- [ ] `docs/` - Documentation

**Utilities:**
- [ ] `check_setup.py` - Installation verification
- [ ] `check_db_status.py` - Database monitoring
- [ ] `test_db_connection.py` - DB diagnostics
- [ ] `test_color_classifier.py` - Color detection testing
- [ ] `test_stats_computation.py` - Stats diagnostics
- [ ] `monitor_processing.sh` - Live monitoring dashboard
- [ ] `restart_orchestrator.sh` - Restart helper

**Documentation:**
- [ ] `README.md` - Main documentation
- [ ] `SETUP_STAGING.md` - Setup guide
- [ ] `MODEL_SETUP.md` - Model download guide
- [ ] `IMPROVEMENTS_SUMMARY.md` - Feature summary
- [ ] `PIPELINE_ISSUES_FIXED.md` - Bug fixes log
- [ ] `color_classifier_review.md` - Color detection analysis
- [ ] `docs/PIPELINE_USAGE.md` - Complete usage guide
- [ ] `docs/QUICK_REFERENCE.md` - Command cheat sheet
- [ ] `docs/PIPELINE_GUIDE.md` - Architecture guide
- [ ] `docs/TRACKER_SELECTION_GUIDE.md` - Tracking guide
- [ ] `docs/README.md` - Documentation index

---

## Critical Files That Must NOT Be Committed

These should be in `.gitignore`:

❌ `.env` - Contains secrets (only `.env.example` should be committed)
❌ `models/*.pt` - Model weights (too large, download separately)
❌ `output/` - Processing output
❌ `*.log` - Log files
❌ `archive_*/` - Archived development files
❌ `backups/` - Backup directories

---

## Pre-Commit Verification

### Step 1: Check Git Status

```bash
git status

# Verify no secrets or large files staged
git diff --cached --name-only
```

### Step 2: Verify .env Not Staged

```bash
# Should show .env in ignored files
git status --ignored | grep .env

# If .env is staged, unstage it immediately
git reset HEAD .env
```

### Step 3: Check File Sizes

```bash
# No files over 10MB should be committed
find . -type f -size +10M -not -path "./.git/*" -not -path "./models/*" -not -path "./output/*"
```

### Step 4: Verify Models Directory

```bash
# Models should NOT be in git
ls -lh models/
git check-ignore models/*.pt
# Should output model file paths (meaning they're ignored)
```

---

## Production Environment Setup

### 1. Server Requirements

**Hardware:**
- [ ] NVIDIA H100 GPU (80GB)
- [ ] 64GB+ RAM
- [ ] 500GB+ SSD storage
- [ ] High-speed network connection

**Software:**
- [ ] Ubuntu 22.04 LTS
- [ ] Python 3.9+
- [ ] CUDA 12.1+
- [ ] cuDNN 8.9+
- [ ] Git

### 2. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Verify NVIDIA drivers
nvidia-smi

# Verify CUDA
nvcc --version
```

### 3. Clone Repository

```bash
# Clone from repo
git clone <repo-url> football-pipeline
cd football-pipeline

# Verify branch
git branch
git log --oneline -5
```

### 4. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installations
python check_setup.py
```

### 5. Model Download

```bash
# Create models directory
mkdir -p models

# Download models (contact team for access)
# Place these files in models/:
# - yolo_player.pt
# - yolo_ball.pt
# - yolo_pitch.pt
# - resnet34_rgb_jnr.pt

# Verify models exist
ls -lh models/*.pt
```

### 6. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with production credentials
nano .env

# Required variables:
# - MYSQL_HOST
# - MYSQL_PORT
# - MYSQL_USER
# - MYSQL_PASSWORD
# - MYSQL_DB
# - TABLE_NAME
# - SBG_BASE
# - SBG_TOKEN

# Test database connection
python test_db_connection.py
```

### 7. Directory Structure

```bash
# Create output directory
mkdir -p output

# Set permissions
chmod 755 orchestrator.py
chmod 755 pipeline_consolidated.py
chmod 755 monitor_processing.sh
chmod 755 restart_orchestrator.sh
```

---

## Testing & Validation

### Stage 1: Quick Test (CPU Mode)

```bash
# Test with small clip, no GPU, no DB
export CUDA_VISIBLE_DEVICES=''
python orchestrator.py \
    --local_video test_videos/sample.mp4 \
    --save_local \
    --max_frames 100 \
    --no_db

# Verify output
ls -la output/
cat output/player_stats.json
```

### Stage 2: GPU Test

```bash
# Test GPU with 500 frames
unset CUDA_VISIBLE_DEVICES
python orchestrator.py \
    --local_video test_videos/sample.mp4 \
    --save_local \
    --max_frames 500 \
    --no_db

# Monitor GPU
watch -n 1 nvidia-smi
```

### Stage 3: Database Integration

```bash
# Test with database
python orchestrator.py \
    --local_video test_videos/sample.mp4 \
    --save_local \
    --max_frames 500

# Verify database entry
python check_db_status.py
```

### Stage 4: Full Match Test

```bash
# Process full match (10-minute video)
python orchestrator.py \
    --local_video test_videos/full_match.mp4 \
    --output_dir output/test_full \
    --locking_mode 2 \
    --tracking_mode bytetrack

# Validate output
python test_stats_computation.py output/test_full/player_stats.json
```

### Stage 5: Polling Mode Test

```bash
# Test polling (dry run, max 1 video)
python orchestrator.py \
    --poll \
    --parallel 1 \
    --max_videos 1 \
    --poll_interval 30

# Monitor logs
tail -f output/pipeline.log
```

---

## Production Deployment

### Option A: Systemd Service (Recommended)

```bash
# Install service
sudo ./deployment/install-service.sh

# Enable auto-start
sudo systemctl enable football-pipeline

# Start service
sudo systemctl start football-pipeline

# Verify status
sudo systemctl status football-pipeline

# View logs
sudo journalctl -u football-pipeline -f
```

### Option B: Manual Start (Development)

```bash
# Start in background
nohup python orchestrator.py \
    --poll \
    --parallel 3 \
    --poll_interval 60 \
    --locking_mode 2 \
    > logs/orchestrator.log 2>&1 &

# Save PID
echo $! > orchestrator.pid

# Monitor
tail -f logs/orchestrator.log
```

---

## Monitoring & Health Checks

### Daily Checks

```bash
# Check processing status
python check_db_status.py

# View health metrics
cat output/health_snapshot.json

# Check logs for errors
tail -100 output/pipeline.log | grep -i error

# GPU utilization
nvidia-smi
```

### Live Monitoring

```bash
# Start monitoring dashboard
./monitor_processing.sh

# Or manual monitoring
watch -n 5 'python check_db_status.py'
```

### Performance Metrics

Track these KPIs:
- **Throughput:** Videos processed per hour
- **Success Rate:** (Finished / Total) %
- **Average FPS:** Processing speed
- **GPU Utilization:** 70-90% optimal
- **Memory Usage:** <60GB for H100
- **Error Rate:** <1%

---

## Troubleshooting

### Common Issues

**GPU Out of Memory:**
```bash
# Check GPU memory
nvidia-smi

# Reduce parallel workers
# In deployment/football-pipeline.service, change --parallel 3 to --parallel 2
```

**Database Connection Failed:**
```bash
# Verify credentials
python test_db_connection.py

# Check .env
cat .env | grep MYSQL

# Test network connectivity
ping db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com
```

**Models Not Found:**
```bash
# Verify models exist
ls -lh models/*.pt

# Check paths in config.yaml
cat config.yaml | grep -A 5 "env:"
```

**Service Won't Start:**
```bash
# Check service logs
sudo journalctl -u football-pipeline -n 50

# Verify Python path
which python
/path/to/venv/bin/python --version

# Test manual start
source venv/bin/activate
python orchestrator.py --poll --parallel 1 --max_videos 1
```

---

## Rollback Procedure

If deployment fails:

```bash
# 1. Stop service
sudo systemctl stop football-pipeline

# 2. Check logs
sudo journalctl -u football-pipeline -n 100

# 3. Restore from backup
git log --oneline -10
git checkout <previous-commit>

# 4. Reinstall
pip install -r requirements.txt

# 5. Test
python check_setup.py

# 6. Restart
sudo systemctl start football-pipeline
```

---

## Post-Deployment

### Week 1: Intensive Monitoring

- [ ] Monitor every 4 hours
- [ ] Check error rates daily
- [ ] Verify all statistics are generating correctly
- [ ] Performance benchmarking

### Week 2-4: Standard Monitoring

- [ ] Daily health check
- [ ] Weekly performance review
- [ ] Review and optimize based on metrics

### Ongoing

- [ ] Monthly model updates (if applicable)
- [ ] Quarterly system updates
- [ ] Regular backup of configuration

---

## Success Criteria

✅ **Deployment is successful when:**

1. Service auto-starts on boot
2. Processing 3+ videos simultaneously without errors
3. 95%+ success rate (finished vs failed)
4. Average processing FPS > 2.5
5. GPU utilization 70-90%
6. No CUDA out of memory errors
7. All statistics generating correctly
8. Database updates working
9. Health metrics reporting
10. No manual intervention needed for 48+ hours

---

## Contact & Support

**Issues:** Check [PIPELINE_USAGE.md](docs/PIPELINE_USAGE.md) troubleshooting section

**Logs Location:**
- Pipeline: `output/pipeline.log`
- Systemd: `sudo journalctl -u football-pipeline`
- Health: `output/health_snapshot.json`

**Critical Files Backup:**
- `.env` (credentials)
- `config.yaml` (configuration)
- Model weights (keep external backup)

---

**Checklist Status:** [ ] Not Started  [→] In Progress  [✓] Completed

Last Updated: 2026-01-30
