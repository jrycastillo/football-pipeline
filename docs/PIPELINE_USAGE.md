# Football Pipeline - Complete Usage Guide

**Version:** 2026-01-30
**Pipeline:** orchestrator.py + pipeline_consolidated.py

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Command-Line Arguments](#command-line-arguments)
5. [Usage Modes](#usage-modes)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Football Analysis Pipeline processes match videos to extract:
- Player tracking and identification
- Jersey number recognition
- Team assignment
- Player statistics (passes, shots, dribbles, etc.)
- Event detection (goals, tackles, interceptions)
- xG (expected goals) calculations

### Architecture

```
orchestrator.py          → High-level coordination, DB, API polling
    ↓
pipeline_consolidated.py → Core processing (detection, tracking, JNR, stats)
    ↓
stats/event_logic.py     → Event detection and statistics calculation
```

---

## Prerequisites

### System Requirements

- **OS:** Linux/macOS (tested on Ubuntu 22.04, macOS)
- **Python:** 3.9+
- **RAM:** 16GB minimum, 32GB recommended
- **GPU:** CUDA-enabled GPU recommended (NVIDIA with 8GB+ VRAM)
  - CPU-only mode supported but slower

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd Babak

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (if not already present)
# Place models in models/ directory:
# - models/yolo_player.pt
# - models/yolo_ball.pt
# - models/yolo_pitch.pt
# - models/resnet34_rgb_jnr.pt

# 5. Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Setup

Create `.env` file with required credentials:

```bash
# Database Configuration
MYSQL_HOST=your-db-host
MYSQL_PORT=25060
MYSQL_USER=your-username
MYSQL_PASSWORD=your-password
MYSQL_DB=your-database
TABLE_NAME=MatchesVideoAnalysis_test

# ScoutBridge API
SBG_BASE=https://api-staging.scoutbridge.net/football-gallery/api
SBG_TOKEN=your-api-token

# Model paths (optional overrides)
DET_WEIGHTS=models/yolo_player.pt
BALL_MODEL_PATH=models/yolo_ball.pt
POSE_WEIGHTS=models/yolo_pitch.pt
JNR_WEIGHTS=models/resnet34_rgb_jnr.pt
```

---

## Command-Line Arguments

### orchestrator.py (High-Level Controller)

#### Basic Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--local_video` | str | - | Path to local video file or SPACES URL for single video processing |
| `--no_db` | flag | False | Skip database connections (for testing) |
| `--save_local` | flag | False | Save output to ./output folder |
| `--make_video` | flag | False | Generate annotated debug video output |
| `--output_dir` | str | - | Custom output directory path |

#### Processing Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_frames` | int | - | Limit number of frames to process (for testing) |
| `--resume_frame` | int | 0 | Start processing from specific frame index |
| `--locking_mode` | int (1-3) | 2 | Identity locking mode (see below) |
| `--jnr_stride` | int | - | Process jersey recognition every N frames |
| `--vid_stride` | int | 1 | Video frame stride (2=skip every other frame) |

#### Tracking Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tracking_mode` | str | bytetrack | Tracking backend: `bytetrack`, `botsort`, `sam2` |
| `--sam2_model` | str | large | SAM2 model size: `tiny`, `small`, `base`, `large` |

#### Polling Mode (Production)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--poll` | flag | False | Enable polling mode to fetch from ScoutBridge API |
| `--poll_interval` | int | 60 | Seconds between API polls |
| `--max_videos` | int | - | Max videos to process before stopping |
| `--min_size_mb` | float | 0 | Minimum video file size filter (MB) |
| `--max_size_mb` | float | ∞ | Maximum video file size filter (MB) |
| `--parallel` | int | 1 | Number of concurrent pipeline workers |

### pipeline_consolidated.py (Core Pipeline)

**Note:** Usually called by orchestrator.py, but can be run directly.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | - | **Required.** Video path (local file or URL) |
| `--output_dir` | str | - | **Required.** Output directory |
| `--no_video_output` | flag | False | Skip video output generation |
| `--max_frames` | int | - | Limit frames to process |
| `--locking_mode` | int (1-3) | 2 | Identity locking mode |
| `--jnr_stride` | int | - | JNR processing stride |
| `--vid_stride` | int | 1 | Video frame stride |
| `--tracking_mode` | str | bytetrack | Tracking backend (legacy) |
| `--tracker` | str | bytetrack | Strict tracker selector: `bytetrack`, `botsort` |
| `--enable_reid` | int (0/1) | - | Enable ReID features |
| `--audit_rejections` | int (0/1) | - | Enable rejection audit logging |
| `--resize_h` | int | 0 | Resize video height (0=disabled) |
| `--sam2_model` | str | large | SAM2 model variant |

---

## Usage Modes

### Mode 1: Single Video Processing (Debug/Testing)

Process a single video file locally.

```bash
# Basic usage
python orchestrator.py --local_video path/to/video.mp4 --save_local

# With debug video output
python orchestrator.py --local_video video.mp4 --save_local --make_video

# Process only first 500 frames
python orchestrator.py --local_video video.mp4 --save_local --max_frames 500

# Skip database (offline testing)
python orchestrator.py --local_video video.mp4 --save_local --no_db

# Custom output directory
python orchestrator.py --local_video video.mp4 --output_dir /path/to/output
```

**Output:**
- `output/player_stats.json` - Player statistics
- `output/player_stats.csv` - Statistics in CSV format
- `output/debug_video.mp4` - Annotated video (if --make_video)
- `output/logs/` - Processing logs

### Mode 2: Polling Mode (Production)

Continuously poll ScoutBridge API for new videos and process them.

```bash
# Basic polling (1 worker)
python orchestrator.py --poll

# Multiple parallel workers (recommended for production)
python orchestrator.py --poll --parallel 3

# Custom poll interval (every 30 seconds)
python orchestrator.py --poll --poll_interval 30 --parallel 3

# Filter by video size (50MB - 2GB)
python orchestrator.py --poll --parallel 3 --min_size_mb 50 --max_size_mb 2000

# Process max 10 videos then stop
python orchestrator.py --poll --parallel 3 --max_videos 10

# With specific tracking backend
python orchestrator.py --poll --parallel 3 --tracking_mode botsort
```

**Production Deployment:**

Use systemd service (see [deployment/football-pipeline.service](../deployment/football-pipeline.service)):

```bash
# Install service
sudo ./deployment/install-service.sh

# Start service
sudo systemctl start football-pipeline

# View logs
sudo journalctl -u football-pipeline -f
```

### Mode 3: Direct Pipeline Execution

Call `pipeline_consolidated.py` directly (advanced users).

```bash
python pipeline_consolidated.py \
    --video test_videos/match.mp4 \
    --output_dir output/match_001 \
    --locking_mode 2 \
    --tracking_mode bytetrack \
    --max_frames 1000
```

---

## Configuration

### config.yaml

Global configuration file for default settings.

```yaml
# Environment Settings
env:
  # Model paths
  DET_WEIGHTS: "models/yolo_player.pt"
  POSE_WEIGHTS: "models/yolo_pitch.pt"
  JNR_WEIGHTS: "models/resnet34_rgb_jnr.pt"
  BALL_MODEL_PATH: "models/yolo_ball.pt"

# Heuristics & Thresholds
heuristics:
  # Detection
  DET_CONF: 0.10          # Detection confidence threshold
  DET_IOU: 0.50           # IoU threshold for NMS
  DET_IMG_SIZE: 832       # Detection input size
  VID_STRIDE: 1           # Video frame stride

  # JNR
  JNR_IMG_SIZE: 160       # JNR input size
  JNR_GATE: 0.35          # JNR confidence threshold

  # Tracking
  MAX_TRACK_FRAMES: null  # Max frames to track (null=unlimited)
  STORE_IMAGES: 1         # Store crops (0/1)
  STORE_IMAGES_UP_TO: 1500  # Max frames to store crops

  # Stats
  FPS: 25                 # Video FPS
  MAX_OWNER_PX: 300       # Max distance for ball possession
  TACKLE_PX: 100          # Distance threshold for tackles
  DRIBBLE_MIN_PX: 25      # Minimum dribble distance

# Class IDs
classes:
  ball: 0
  goalkeeper: 1
  player: 2
  referee: 3
```

### Locking Modes

Controls how jersey numbers are assigned to tracks:

| Mode | Name | Behavior | Use Case |
|------|------|----------|----------|
| **1** | Instant Lock | Lock on first high-confidence detection | Fast, less accurate |
| **2** | Consecutive (Default) | Lock after N consecutive consistent detections | **Recommended** - balanced |
| **3** | Bayesian Dirichlet | Probabilistic accumulation | Most accurate, slower |

**Recommendation:** Use Mode 2 for production. Mode 3 for critical accuracy.

### Tracking Modes

| Mode | Speed | Accuracy | ReID | Description |
|------|-------|----------|------|-------------|
| **bytetrack** | ⚡⚡⚡ | ⭐⭐⭐ | No | **Default.** Fast, reliable motion-based tracking |
| **botsort** | ⚡⚡ | ⭐⭐⭐⭐ | Yes | Slower, uses appearance features for re-identification |
| **sam2** | ⚡ | ⭐⭐⭐⭐⭐ | Yes | Experimental. Segmentation-based, very slow |

**Recommendation:** Use `bytetrack` for speed, `botsort` for accuracy.

---

## Examples

### Example 1: Quick Test (First 500 Frames)

```bash
python orchestrator.py \
    --local_video test_videos/sample.mp4 \
    --save_local \
    --max_frames 500 \
    --no_db
```

**Use Case:** Verify pipeline setup, test new changes

### Example 2: Full Match Processing (Local)

```bash
python orchestrator.py \
    --local_video videos/full_match.mp4 \
    --output_dir output/match_20260130 \
    --locking_mode 2 \
    --tracking_mode bytetrack
```

**Use Case:** Process archived matches locally

### Example 3: Production Polling (3 Workers)

```bash
python orchestrator.py \
    --poll \
    --parallel 3 \
    --poll_interval 60 \
    --locking_mode 2 \
    --tracking_mode bytetrack
```

**Use Case:** Production server continuously processing new uploads

### Example 4: High-Accuracy Processing

```bash
python orchestrator.py \
    --local_video important_match.mp4 \
    --save_local \
    --locking_mode 3 \
    --tracking_mode botsort \
    --make_video
```

**Use Case:** Critical match requiring maximum accuracy

### Example 5: Fast Preview (Skip Frames)

```bash
python orchestrator.py \
    --local_video match.mp4 \
    --save_local \
    --vid_stride 2 \
    --max_frames 1000
```

**Use Case:** 2x faster processing for quick preview

### Example 6: Resume Processing

```bash
python orchestrator.py \
    --local_video match.mp4 \
    --save_local \
    --resume_frame 5000
```

**Use Case:** Continue from frame 5000 after interruption

---

## Monitoring

### Real-Time Monitoring

```bash
# Live log monitoring
tail -f output/pipeline.log

# Database status
python check_db_status.py

# Live processing dashboard
./monitor_processing.sh
```

### Health Metrics

The pipeline tracks:
- **videos_processed** - Completed videos
- **videos_failed** - Failed videos
- **db_queries** / **db_errors** - Database health
- **api_calls** / **api_errors** - API health
- **average_fps** - Processing speed

View metrics:

```bash
cat output/health_snapshot.json
```

Example output:

```json
{
  "timestamp": "2026-01-30T12:00:00",
  "metrics": {
    "videos_processed": 53,
    "videos_failed": 0,
    "videos_running": 3,
    "db_queries": 156,
    "db_errors": 0,
    "api_calls": 45,
    "api_errors": 0
  },
  "performance": {
    "average_processing_time_s": 4523.2,
    "average_fps": 3.31
  }
}
```

### Database Monitoring

```bash
# Check current status
python check_db_status.py

# Full connection test
python test_db_connection.py
```

Output:

```
================================================================================
DATABASE STATUS
================================================================================

🔄 Running: 3
✅ Finished: 53
❌ Failed: 0

⏰ Updated in last hour: 3
   • ID   124 | running    | Updated: 2026-01-30 12:30:45
   • ID   123 | running    | Updated: 2026-01-30 12:25:12
   • ID   122 | running    | Updated: 2026-01-30 12:20:33
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Option 1: Process every other frame (2x faster, uses less memory)
python orchestrator.py --local_video video.mp4 --vid_stride 2

# Option 2: Use CPU mode
export CUDA_VISIBLE_DEVICES=''
python orchestrator.py --local_video video.mp4

# Option 3: Reduce detection size in config.yaml
# DET_IMG_SIZE: 640  # instead of 832
```

#### 2. Database Connection Failed

**Error:** `MYSQL_PASSWORD not set`

**Solution:**
```bash
# Check .env file exists and has correct credentials
cat .env | grep MYSQL_PASSWORD

# Verify connection
python test_db_connection.py
```

#### 3. No Statistics Generated

**Issue:** `player_stats.json` shows all zeros

**Diagnosis:**
```bash
python test_stats_computation.py output/player_stats.json
```

**Common causes:**
- Pipeline crashed early (check logs)
- Ball tracking failed
- CUDA errors during processing

**Solution:**
```bash
# Check logs for errors
tail -100 output/pipeline.log | grep -i error

# Try CPU mode
export CUDA_VISIBLE_DEVICES=''
python orchestrator.py --local_video video.mp4 --save_local
```

#### 4. Worker Tasks Failing Silently

**Check:**
```bash
# Look for worker errors in logs
grep "Worker task failed" output/pipeline.log

# Check health metrics
cat output/health_snapshot.json | grep errors
```

#### 5. Temp Files Accumulating

**Issue:** Disk space filling up

**Solution:**
```bash
# Clean temp files
rm -rf /tmp/football_*

# Check cleanup is working
ls -lh /tmp/football_* 2>/dev/null || echo "Cleanup working correctly"
```

### Debug Mode

Enable verbose logging:

```bash
# Set logging level
export PYTHONUNBUFFERED=1

# Run with logging
python orchestrator.py --local_video video.mp4 --save_local 2>&1 | tee debug.log
```

### Performance Optimization

| Scenario | Recommended Settings |
|----------|---------------------|
| **Maximum Speed** | `--vid_stride 2 --tracking_mode bytetrack` |
| **Maximum Accuracy** | `--locking_mode 3 --tracking_mode botsort` |
| **Balanced (Default)** | `--locking_mode 2 --tracking_mode bytetrack` |
| **Low Memory** | `--vid_stride 2` + reduce DET_IMG_SIZE in config |
| **Preview/Testing** | `--max_frames 500 --vid_stride 2` |

### Getting Help

1. **Check Documentation:**
   - [README.md](../README.md) - Overview
   - [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Architecture details
   - [TRACKER_SELECTION_GUIDE.md](TRACKER_SELECTION_GUIDE.md) - Tracking options

2. **Run Tests:**
   ```bash
   python check_setup.py           # Verify installation
   python test_db_connection.py    # Test database
   python test_color_classifier.py # Test color detection
   ```

3. **View Logs:**
   ```bash
   tail -100 output/pipeline.log
   cat polling_service.log
   ```

4. **Check System Status:**
   ```bash
   # GPU status
   nvidia-smi

   # Running processes
   ps aux | grep -E "pipeline|orchestrator"

   # Disk space
   df -h
   ```

---

## Output Files

### Standard Output

After processing, output directory contains:

```
output/
├── player_stats.json       # Main statistics output
├── player_stats.csv        # Statistics in CSV format
├── match_kits.json         # Detected team colors
├── debug_video.mp4         # Annotated video (if --make_video)
├── pipeline.log            # Processing logs
└── health_snapshot.json    # Health metrics
```

### player_stats.json Structure

```json
{
  "player_1": {
    "player_name": "Unknown",
    "jersey_number": 10,
    "team": "Team A",
    "observations": 523,
    "stats": {
      "ball_touches_total": 45,
      "time_on_ball_s": 12.3,
      "total_distance": 8542.1,
      "passes_total": 32,
      "passes_successful": 28,
      "shots_on_target_total": 2,
      "shots_wide_total": 1,
      "crosses_total": 4,
      "dribbles_total": 5,
      "tackles_total": 2,
      "ball_interceptions_total": 3,
      "goals_total": 1,
      "xg_foot_no_opponent": 0.85
    }
  }
}
```

---

## API Integration

### ScoutBridge API Polling

The pipeline automatically polls for new videos when run with `--poll`:

**API Endpoint:** `{SBG_BASE}/matches-videos/tasks/queued`

**Authentication:** Bearer token from `.env`

**Response:**
```json
{
  "items": [
    {
      "id": "video_123",
      "spacesURL": "https://...",
      "fileSize": 125000000,
      "userId": "user_456"
    }
  ]
}
```

### Database Updates

Pipeline updates `MatchesVideoAnalysis_test` table:

| Field | Description |
|-------|-------------|
| `unique_id` | SHA1 hash of video URL |
| `matches_video_id` | Video ID |
| `user_id` | User ID |
| `status` | `queued`, `running`, `finished`, `failed` |
| `analysis` | JSON output (player_stats) |
| `error` | Error message if failed |
| `created_at` | Timestamp |
| `updated_at` | Last update timestamp |

---

## Best Practices

### Production Deployment

1. **Use systemd service** for automatic restart
2. **Set --parallel 3** for optimal throughput
3. **Monitor health metrics** regularly
4. **Set up log rotation** to prevent disk filling
5. **Use --locking_mode 2** for balanced accuracy/speed

### Development/Testing

1. **Use --max_frames 500** for quick iterations
2. **Enable --make_video** to visualize results
3. **Use --no_db** for offline testing
4. **Test color classifier** after changes: `python test_color_classifier.py`

### Performance Tuning

1. **GPU utilization:** Monitor with `nvidia-smi`
2. **Process in batches:** Use --max_videos to limit batch size
3. **Frame skipping:** Use --vid_stride 2 for 2x speedup
4. **Memory management:** Pipeline auto-cleans cache between videos

---

## Version History

- **2026-01-30:** Added futures tracking, graceful shutdown, color classifier fixes
- **2026-01-29:** Added health monitoring, improved error handling
- **2026-01-25:** Added Gold, Lime, Teal, Cyan colors to classifier
- **2026-01-22:** Disabled SAM2 by default for speed

---

## Support

For issues or questions:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review logs in `output/pipeline.log`
3. Run diagnostic scripts: `check_setup.py`, `test_db_connection.py`
4. Check [GitHub Issues](https://github.com/your-repo/issues)
