# Staging Server Setup Guide

**Quick reference for deploying the Football Pipeline on your staging server**

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Ubuntu 20.04+ server with SSH access
- [ ] sudo/root privileges
- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] MySQL database credentials
- [ ] ScoutBridge API token
- [ ] Model files (contact team for access)

Optional but recommended:
- [ ] GPU with CUDA 11.8+ drivers (for faster processing)
- [ ] 50GB+ free disk space
- [ ] 16GB+ RAM

---

## Step-by-Step Setup

### 1. Clone Repository

```bash
# SSH into your staging server
ssh ubuntu@your-staging-server.com

# Clone repo
cd /home/ubuntu
git clone https://github.com/your-org/football-pipeline.git football
cd football
```

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required variables to set:**

```bash
# Database (REQUIRED)
MYSQL_HOST=your-db-host.ondigitalocean.com
MYSQL_PORT=25060
MYSQL_USER=scoutbridge
MYSQL_PASSWORD=YOUR_ACTUAL_PASSWORD_HERE  #  CHANGE THIS
MYSQL_DB=footballgallery
TABLE_NAME=MatchesVideoAnalysis_test

# API (REQUIRED)
SBG_BASE=https://api-staging.scoutbridge.net/football-gallery/api
SBG_TOKEN=YOUR_ACTUAL_JWT_TOKEN_HERE      #  CHANGE THIS
```

Save and exit (Ctrl+X, Y, Enter).

### 4. Download Models

```bash
# Create models directory
mkdir -p models

# Download models (get download links from team)
# Place these files in models/:
# - yolo_player.pt
# - yolo_ball.pt
# - yolo_pitch.pt
# - resnet34_rgb_jnr.pt

# Verify models exist
ls -lh models/
```

Expected output:
```
-rw-r--r-- 1 ubuntu ubuntu 137M yolo_player.pt
-rw-r--r-- 1 ubuntu ubuntu  52M yolo_ball.pt
-rw-r--r-- 1 ubuntu ubuntu 137M yolo_pitch.pt
-rw-r--r-- 1 ubuntu ubuntu  87M resnet34_rgb_jnr.pt
```

### 5. Test Pipeline

```bash
# Test with a small video (if you have one)
python orchestrator.py \
    --local_video test_videos/sample.mp4 \
    --output_dir ./test_output \
    --no_db \
    --max_frames 500

# Check output
ls -lh test_output/
cat test_output/player_stats.json | jq '.players_flat | length'
```

### 6. Test Database Connection

```bash
# Source environment
source venv/bin/activate

# Test DB connection
python -c "
from orchestrator import _conn
try:
    conn = _conn()
    print(' Database connection successful')
    conn.close()
except Exception as e:
    print(f' Database connection failed: {e}')
"
```

### 7. Install Systemd Service

```bash
# Make installer executable
chmod +x deployment/install-service.sh

# Run installer (requires sudo)
sudo deployment/install-service.sh
```

Expected output:
```
==== Football Pipeline Service Installer ====
 Creating directories...
 Installing service file...
 Reloading systemd...
 Enabling service...
 Starting service...
● football-pipeline.service - Football Analysis Pipeline
   Loaded: loaded
   Active: active (running)
```

### 8. Verify Service is Running

```bash
# Check status
sudo systemctl status football-pipeline

# View live logs
tail -f logs/pipeline.log

# Or journalctl
journalctl -u football-pipeline -f
```

You should see:
```
[poll] Starting polling loop (interval=60s, workers=3)...
[poll] Fetching from https://api-staging.scoutbridge.net/...
[poll] Found X pending videos.
```

---

## Configuration Tuning

### Adjust Parallel Workers

Based on your server specs:

| Server RAM | CPU Cores | GPU | Recommended Workers |
|------------|-----------|-----|---------------------|
| 8GB | 4 | No | 1 |
| 16GB | 8 | No | 2 |
| 32GB | 16 | No | 3 |
| 32GB+ | 16+ | Yes | 3-5 |

Edit service file:
```bash
sudo nano /etc/systemd/system/football-pipeline.service

# Change --parallel value
ExecStart=... --parallel 2

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart football-pipeline
```

### Change Tracker Mode

For faster processing (lower accuracy):
```bash
# Edit service
sudo nano /etc/systemd/system/football-pipeline.service

# Change --tracking_mode
ExecStart=... --tracking_mode bytetrack

# Reload
sudo systemctl daemon-reload
sudo systemctl restart football-pipeline
```

For best accuracy (slower):
```bash
ExecStart=... --tracking_mode botsort
```

---

## Monitoring

### View Logs

```bash
# Application log (recommended)
tail -f /home/ubuntu/football/logs/pipeline.log

# Error log
tail -f /home/ubuntu/football/logs/pipeline-error.log

# Systemd journal
journalctl -u football-pipeline -f

# Last 100 lines
journalctl -u football-pipeline -n 100
```

### Check Health Metrics

```bash
# View health snapshot
cat /home/ubuntu/football/output/health_snapshot.json | jq '.'

# Key metrics
cat output/health_snapshot.json | jq '.metrics'

# Recent errors
cat output/health_snapshot.json | jq '.recent_errors'
```

### Monitor Resources

```bash
# CPU/Memory usage
htop

# GPU usage (if available)
nvidia-smi -l 1

# Disk usage
df -h
du -sh /home/ubuntu/football/output/*
```

---

## Maintenance

### Update Code

```bash
# Stop service
sudo systemctl stop football-pipeline

# Pull latest
cd /home/ubuntu/football
git pull

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl start football-pipeline
```

### Clear Old Outputs

```bash
# Clean outputs older than 30 days
find /home/ubuntu/football/output -type d -mtime +30 -exec rm -rf {} +

# Clean logs older than 14 days
find /home/ubuntu/football/logs -name "*.log" -mtime +14 -delete
```

### Backup Database

```bash
# Dump recent results
mysqldump -h $MYSQL_HOST -P $MYSQL_PORT -u $MYSQL_USER -p$MYSQL_PASSWORD \
    $MYSQL_DB MatchesVideoAnalysis_test > backup_$(date +%Y%m%d).sql
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u football-pipeline -n 50 --no-pager

# Test manually
cd /home/ubuntu/football
source venv/bin/activate
python orchestrator.py --poll --parallel 1 --no_db
```

### Database Errors

```bash
# Verify credentials
echo $MYSQL_PASSWORD

# Test connection
mysql -h $MYSQL_HOST -P $MYSQL_PORT -u $MYSQL_USER -p$MYSQL_PASSWORD -e "SHOW DATABASES;"
```

### No Videos Being Processed

Possible causes:
1. All videos already processed (check DB)
2. API token expired
3. Network issues
4. Size filters excluding videos

Check:
```bash
# Test API
curl -H "Authorization: Bearer $SBG_TOKEN" \
     https://api-staging.scoutbridge.net/football-gallery/api/v2/files/list/video/for-match-analysis
```

### Out of Disk Space

```bash
# Check space
df -h

# Find large directories
du -sh /home/ubuntu/football/* | sort -h

# Clean old outputs
rm -rf /home/ubuntu/football/output/older_than_30_days/
```

---

## Quick Commands Reference

```bash
# Service management
sudo systemctl start football-pipeline
sudo systemctl stop football-pipeline
sudo systemctl restart football-pipeline
sudo systemctl status football-pipeline

# View logs
tail -f logs/pipeline.log
journalctl -u football-pipeline -f

# Health check
cat output/health_snapshot.json | jq '.status'

# Manual run (debug)
cd /home/ubuntu/football
source venv/bin/activate
python orchestrator.py --local_video test.mp4 --no_db

# Update service after config change
sudo systemctl daemon-reload
sudo systemctl restart football-pipeline
```

---

## Security Checklist

Before going to production:

- [ ] `.env` file has correct permissions (`chmod 600 .env`)
- [ ] `.env` is in `.gitignore` (already done)
- [ ] Database password is strong and unique
- [ ] API token is valid and not expired
- [ ] Firewall allows only necessary ports
- [ ] SSH key authentication enabled (disable password login)
- [ ] Regular backups configured
- [ ] Log rotation configured
- [ ] Monitoring/alerting set up

---

## Performance Benchmarks

**Test Environment**: Ubuntu 22.04, 16 vCPU, 32GB RAM, RTX 3090

| Video Length | Resolution | Processing Time | Throughput |
|--------------|------------|-----------------|------------|
| 5 min | 1080p | 7 min | 0.7x realtime |
| 10 min | 1080p | 14 min | 0.7x realtime |
| 45 min (half) | 720p | 35 min | 1.3x realtime |
| 90 min (full) | 1080p | 2.5 hours | 0.6x realtime |

*With ByteTrack, stride=1, 3 parallel workers*

---

## Next Steps

1. Service is running and processing videos
2. Set up monitoring dashboard (Grafana recommended)
3. Configure alerts for failures
4. Optimize based on your video characteristics
5. Deploy to production when ready

---

## Support

- **Documentation**: [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)
- **Tracker Guide**: [docs/TRACKER_SELECTION_GUIDE.md](docs/TRACKER_SELECTION_GUIDE.md)
- **Deployment**: [deployment/README.md](deployment/README.md)
- **Issues**: GitHub Issues or Slack #football-pipeline

---

**You're all set! The pipeline is now running and will automatically process incoming videos.**

*For any issues, check the logs first: `tail -f logs/pipeline.log`*
