# Football Analysis Pipeline - Complete Guide

**Version**: Production-Clean
**Last Updated**: 2026-01-29
**Maintainer**: Football Analytics Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage Modes](#usage-modes)
7. [Pipeline Components](#pipeline-components)
8. [Output Format](#output-format)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## Overview

The Football Analysis Pipeline is an end-to-end computer vision system that processes football match videos to extract:

- **Player tracking** with persistent IDs
- **Team assignment** via jersey color analysis
- **Jersey number recognition** using hybrid ResNet + VLM
- **Ball tracking** with pitch polygon filtering
- **Event detection** (passes, tackles, interceptions, crosses, dribbles)
- **Shot analysis** with Expected Goals (xG) calculation
- **Goal detection** with automatic validation
- **Comprehensive statistics** per player and team

### Key Features

✅ **Production-ready**: Battle-tested on 1000+ match videos
✅ **Scalable**: Parallel processing support (1-16 workers)
✅ **Flexible**: Local files, streaming URLs, or cloud storage
✅ **Accurate**: State-of-the-art tracking and recognition models
✅ **Observable**: Built-in health monitoring and metrics
✅ **Configurable**: 50+ tunable parameters via config file or CLI

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│  - API Polling                                               │
│  - Job Queue Management                                      │
│  - Database Integration                                      │
│  - Health Monitoring                                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE CONSOLIDATED                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   DETECTION  │→ │   TRACKING   │→ │     STATS    │      │
│  │              │  │              │  │              │      │
│  │ • YOLOv8     │  │ • ByteTrack  │  │ • Possession │      │
│  │ • Players    │  │ • BoT-SORT   │  │ • Passes     │      │
│  │ • Ball       │  │ • SAM2       │  │ • Shots/xG   │      │
│  │ • Referees   │  │              │  │ • Events     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ TEAM ASSIGN  │  │     JNR      │  │   OUTPUT     │      │
│  │              │  │              │  │              │      │
│  │ • HSV K-Means│  │ • ResNet32   │  │ • JSON       │      │
│  │ • Color Desc │  │ • Qwen2.5-VL │  │ • CSV        │      │
│  │              │  │ • Voting     │  │ • Video      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Video file (local/URL) or API poll for pending videos
2. **Detection**: Frame-by-frame object detection (players, ball, etc.)
3. **Tracking**: Associate detections across frames with unique IDs
4. **Team Assignment**: Cluster players by jersey color (K-means)
5. **Jersey Recognition**: Identify player numbers (ResNet → Qwen verification)
6. **Statistics**: Compute events, possession, shots, xG
7. **Output**: JSON stats, CSV, optional annotated video
8. **Database**: Store results in MySQL (production mode)

---

## Installation

### Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS 11+, or Windows 10+
- **Python**: 3.8 - 3.11
- **GPU**: Optional but recommended (CUDA 11.8+ or MPS for Mac)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk**: 50GB for models + output storage

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/football-pipeline.git
cd football-pipeline
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Models

```bash
# Download pre-trained models (contact team for access)
./scripts/download_models.sh

# Or manually place models in models/ directory:
# - models/yolo_player.pt       (Player detection)
# - models/yolo_ball.pt          (Ball detection)
# - models/yolo_pitch.pt         (Pitch keypoints)
# - models/resnet34_rgb_jnr.pt   (Jersey numbers)
```

### Step 5: Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with your credentials
```

**Required variables:**
```bash
MYSQL_PASSWORD=your-database-password
SBG_TOKEN=your-api-token
```

See [.env.example](.env.example) for all options.

---

## Quick Start

### Process a Single Video

```bash
# Basic usage (local file)
python orchestrator.py --local_video path/to/match.mp4 --save_local

# With custom output directory
python orchestrator.py \
    --local_video match.mp4 \
    --output_dir ./results \
    --make_video
```

### Process from URL

```bash
# Streaming URL (DigitalOcean Spaces, S3, etc.)
python orchestrator.py \
    --local_video "https://your-cdn.com/match.mp4" \
    --save_local \
    --no_db
```

### Start Polling Service (Production)

```bash
# Poll API for pending videos, process 3 in parallel
python orchestrator.py \
    --poll \
    --poll_interval 60 \
    --parallel 3
```

### Check Results

```bash
# View JSON output
cat output/player_stats.json | jq '.players_flat[] | {jersey_number, team, passes, goals}'

# View CSV
column -s, -t < output/player_stats.csv | less -S

# Health metrics
cat output/health_snapshot.json | jq '.metrics'
```

---

## Configuration

The pipeline uses a **three-tier configuration system** (precedence: CLI > ENV > config.yaml).

### config.yaml

Main configuration file with sensible defaults:

```yaml
env:
  # Model paths
  DET_WEIGHTS: "models/yolo_player.pt"
  BALL_MODEL_PATH: "models/yolo_ball.pt"
  POSE_WEIGHTS: "models/yolo_pitch.pt"
  JNR_WEIGHTS: "models/resnet34_rgb_jnr.pt"

  # Database
  MYSQL_HOST: "your-db-host.com"
  MYSQL_PORT: 25060
  MYSQL_USER: "scoutbridge"
  MYSQL_DB: "footballgallery"
  TABLE_NAME: "MatchesVideoAnalysis_test"

  # API
  SBG_BASE: "https://api-staging.scoutbridge.net/football-gallery/api"

heuristics:
  # Detection
  DET_CONF: 0.10          # Detection confidence threshold
  DET_IOU: 0.50           # NMS IoU threshold
  DET_IMG_SIZE: 832       # Input image size for YOLO
  VID_STRIDE: 1           # Process every Nth frame (1 = all frames)

  # Jersey Recognition
  JNR_IMG_SIZE: 160       # Crop size for JNR model
  JNR_GATE: 0.35          # Confidence threshold for JNR

  # Stats
  MAX_OWNER_PX: 300       # Max distance (px) for ball ownership
  TACKLE_PX: 100          # Max distance (px) for tackle detection
  DRIBBLE_MIN_PX: 25      # Min displacement (px) for dribble
  OPP_NEAR_PX: 150        # Proximity threshold for opponent pressure

  # xG
  SPEED_PX_THR_FRAC: 0.015  # Ball speed threshold (fraction of width)
  NEAR_GOAL_FRAC: 0.50      # "Near goal" region (fraction of width)
```

### Environment Variables

Override config.yaml with environment variables:

```bash
export MYSQL_PASSWORD="secure-password"
export DET_CONF="0.25"
export VID_STRIDE="2"
export MAX_TRACK_FRAMES="5000"
```

### CLI Arguments

Highest priority - overrides both config.yaml and ENV:

```bash
python orchestrator.py \
    --local_video match.mp4 \
    --max_frames 3000 \
    --locking_mode 2 \
    --jnr_stride 5 \
    --vid_stride 2 \
    --tracking_mode bytetrack \
    --make_video
```

**Common CLI flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--local_video PATH` | Path/URL to video file | Required in local mode |
| `--poll` | Enable API polling mode | Disabled |
| `--parallel N` | Concurrent jobs | 1 |
| `--max_frames N` | Limit frames processed | Unlimited |
| `--tracking_mode` | bytetrack/botsort/sam2 | bytetrack |
| `--locking_mode` | Internal mode (1/2/3) | 2 |
| `--make_video` | Generate annotated output video | Disabled |
| `--no_db` | Skip database writes | Enabled if no creds |

See `python orchestrator.py --help` for full list.

---

## Usage Modes

### 1. Debug Mode (Single Video)

Best for: Testing, development, local analysis

```bash
python orchestrator.py \
    --local_video test_videos/match.mp4 \
    --output_dir ./debug_output \
    --no_db \
    --make_video \
    --max_frames 1000
```

**Output**: Local files in `./debug_output/`

### 2. Batch Processing

Best for: Processing multiple local files

```bash
for video in test_videos/*.mp4; do
    python orchestrator.py \
        --local_video "$video" \
        --output_dir "./output/$(basename "$video" .mp4)" \
        --no_db
done
```

### 3. API Polling (Production)

Best for: Continuous processing of incoming videos

```bash
python orchestrator.py \
    --poll \
    --poll_interval 60 \
    --parallel 3 \
    --min_size_mb 10 \
    --max_size_mb 5000 \
    --tracking_mode bytetrack
```

**How it works:**
1. Polls ScoutBridge API every 60 seconds
2. Fetches pending videos
3. Filters by size (10-5000 MB)
4. Submits to thread pool (3 workers)
5. Processes each video through full pipeline
6. Writes results to database
7. Repeats indefinitely

### 4. Systemd Service (Server Deployment)

Best for: Production servers with auto-restart

```bash
# Install service (see deployment/README.md)
sudo deployment/install-service.sh

# Monitor
journalctl -u football-pipeline -f
```

Service auto-starts on boot and restarts on failure.

---

## Pipeline Components

### 1. Detection Module

**Model**: YOLOv8x (player detection), YOLOv8m (ball)
**Classes**: Player, Goalkeeper, Ball, Referee
**Processing**: Frame-by-frame with configurable stride

**Key parameters:**
- `DET_CONF`: Confidence threshold (0.1 = recall-focused, 0.5 = precision-focused)
- `DET_IOU`: NMS IoU threshold
- `DET_IMG_SIZE`: Input resolution (higher = slower but more accurate)

### 2. Tracking Module

**Backends**: ByteTrack (default), BoT-SORT, SAM2
**Features**:
- Persistent player IDs across frames
- Ghost spawn prevention
- Stale track cleanup
- Occlusion handling

See [TRACKER_SELECTION_GUIDE.md](TRACKER_SELECTION_GUIDE.md) for choosing the right tracker.

### 3. Team Assignment

**Algorithm**: HSV color clustering (K-means, k=2)
**Input**: Jersey torso crops from first 1200 frames
**Output**: Team labels (e.g., "blue" vs "red") + player→team mapping

**Process:**
1. Extract jersey color patches (torso region)
2. Convert to HSV, filter grass/dark pixels
3. Compute median hue/saturation/value per player
4. Cluster into 2 teams using K-means
5. Label teams by dominant color

### 4. Jersey Number Recognition (JNR)

**Hybrid Two-Pass System:**

**Pass 1 - ResNet32 (Primary)**:
- Fast CNN inference on jersey crops
- Voting: 3 consecutive identical predictions = locked
- Confidence threshold: 0.80

**Pass 2 - Qwen2.5-VL (Verifier)**:
- Vision-language model for low-confidence cases
- Reviews predictions below 80% confidence
- Always reviews ambiguous numbers (11, 19, 29)

**Output**: Jersey number + confidence + source (resnet_vote/qwen)

### 5. Statistics Engine

Computes per-player and per-team statistics:

**Possession**:
- Touch frames (frames with ball control)
- Time on ball (seconds)

**Passing**:
- Total passes
- Accurate passes (same team)
- Accuracy percentage

**Defensive**:
- Tackles (close-range turnovers)
- Interceptions (cross-field turnovers)
- Challenges (tackles + interceptions)

**Attacking**:
- Crosses (wide→box passes)
- Dribbles (sustained possession with displacement)
- Shots on target / wide
- Goals

**Expected Goals (xG)**:
- Distance to goal
- Angle to goal
- Header vs foot
- Opponent pressure
- Categories: foot_no_opponent, foot_opponent_present, header_no_opponent, header_opponent_present

### 6. Goal Detection

**Heuristic-based approach:**

1. Detect high-speed ball movement (>1.5% of frame width/frame)
2. Near goal region (<50% of width from goal)
3. Ball enters goal mouth zone (12% from edge, 15-85% height)
4. Validate: Ball stays in goal or disappears (not bounced back)

**Output**: `is_goal: true/false` in shot events

---

## Output Format

### player_stats.json

Complete statistics in JSON format:

```json
{
  "players_flat": [
    {
      "team": "blue",
      "player_id": 23,
      "jersey_number": "10",
      "time_on_ball_s": 145.2,
      "touch_frames": 3630,
      "shots_on_target": 4,
      "shots_wide": 1,
      "penalty": 0,
      "crosses": 8,
      "crosses_accurate": 5,
      "dribbles": 12,
      "dribbles_successful": 10,
      "passes": 87,
      "accurate_passes_%": 82.5,
      "challenges": 6,
      "challenges_won": 5,
      "tackles": 3,
      "tackles_successful": 3,
      "ball_interceptions": 2,
      "fouls": 0,
      "goals": 1,
      "xg_foot_no_opponent": 0.35,
      "xg_header_no_opponent": 0.0,
      "xg_foot_opponent_present": 0.18,
      "xg_header_opponent_present": 0.0
    }
  ],
  "class_map": {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3},
  "team_labels": {"0": "blue", "1": "red"},
  "video_path": "match.mp4",
  "model_weights": "models/yolo_player.pt",
  "fps": 25
}
```

### player_stats.csv

Same data in CSV format for Excel/analysis tools.

### tracking.json (Optional)

Frame-by-frame tracking data (large file):

```json
{
  "frames": [
    {
      "frame_id": 0,
      "boxes": [
        {
          "id": 23,
          "cls": 2,
          "conf": 0.95,
          "xyxy": [120, 340, 180, 480],
          "team": 0,
          "jersey": "10"
        }
      ]
    }
  ]
}
```

### Health Snapshot

Operational metrics saved to `output/health_snapshot.json`:

```json
{
  "status": "healthy",
  "uptime_hours": 12.5,
  "metrics": {
    "videos_processed": 45,
    "videos_failed": 2,
    "videos_running": 1,
    "total_frames_processed": 1250000,
    "db_queries": 180,
    "api_calls": 150
  },
  "performance": {
    "avg_processing_time_s": 320.5,
    "avg_frame_rate_fps": 22.3
  }
}
```

---

## Performance Tuning

### Speed Optimization

**1. Reduce frame processing**:
```bash
--vid_stride 2  # Process every 2nd frame (2x faster)
```

**2. Limit video length**:
```bash
--max_frames 3000  # ~2 min at 25fps
```

**3. Use faster tracker**:
```bash
--tracking_mode bytetrack  # vs botsort
```

**4. Reduce detection resolution**:
```yaml
# config.yaml
DET_IMG_SIZE: 640  # Default 832
```

**5. Skip video output**:
```bash
# Don't use --make_video flag
```

### Accuracy Optimization

**1. Use best tracker**:
```bash
--tracking_mode botsort
```

**2. Increase detection confidence**:
```yaml
DET_CONF: 0.25  # Default 0.10 (more false positives)
```

**3. Process all frames**:
```bash
--vid_stride 1
```

**4. Higher resolution**:
```yaml
DET_IMG_SIZE: 1280  # Warning: Much slower
```

### Memory Optimization

**1. Reduce stored images**:
```yaml
STORE_IMAGES_UP_TO: 500  # Default 1500
```

**2. Lower parallel workers**:
```bash
--parallel 1  # Default 3
```

**3. Limit JNR sampling**:
```bash
--jnr_stride 10  # Sample every 10th frame for jersey recognition
```

### Benchmark Results

**Test video**: 1080p, 5 minutes, 25 FPS (7500 frames)

| Configuration | Time | FPS | Accuracy |
|---------------|------|-----|----------|
| **Speed-focused** (ByteTrack, stride=2) | 4 min | 31 | Good |
| **Balanced** (ByteTrack, stride=1) | 7 min | 18 | Very Good |
| **Accuracy-focused** (BoT-SORT, stride=1) | 12 min | 10 | Excellent |

*GPU: RTX 3090, CPU: Ryzen 9 5900X*

---

## Troubleshooting

### Common Issues

#### 1. "MYSQL_PASSWORD not set" Warning

**Cause**: Environment variable not configured
**Solution**:
```bash
# Add to .env file
echo "MYSQL_PASSWORD=your-password" >> .env

# Or set temporarily
export MYSQL_PASSWORD="your-password"
```

#### 2. CUDA Out of Memory

**Cause**: GPU memory exhausted (usually with BoT-SORT or large batches)
**Solutions**:
- Reduce parallel workers: `--parallel 1`
- Use ByteTrack instead: `--tracking_mode bytetrack`
- Lower detection resolution: `DET_IMG_SIZE: 640`
- Process fewer frames: `--max_frames 3000`

#### 3. No Detections / Empty Output

**Possible causes**:
- Detection confidence too high
- Video codec issues
- Incorrect model paths

**Debug**:
```bash
# Test with lower confidence
DET_CONF=0.05 python orchestrator.py --local_video test.mp4 --no_db

# Verify models exist
ls -lh models/*.pt

# Check video can be read
ffprobe test.mp4
```

#### 4. Team Assignment Failure

**Symptom**: All players labeled "unknown" team
**Causes**:
- Similar jersey colors (both teams blue)
- Poor lighting
- Not enough frames sampled

**Solutions**:
- Increase sample frames in code (line 1242: `sample_frames=2000`)
- Manually label teams post-processing
- Use better quality video

#### 5. Jersey Number Recognition Poor

**Symptoms**: Most players show "Unknown" jersey
**Causes**:
- Low video resolution
- Motion blur
- Numbers not visible (camera angle)

**Solutions**:
- Use higher resolution source video
- Reduce stride: `--jnr_stride 3` (sample more often)
- Lower gate threshold: `JNR_GATE: 0.25` (more permissive)
- Check Qwen model is loaded (requires GPU + 16GB VRAM)

#### 6. Slow Processing

**Check**:
```bash
# View health metrics
cat output/health_snapshot.json | jq '.performance'

# Monitor GPU usage
nvidia-smi -l 1

# Check CPU
htop
```

**Optimize** (see [Performance Tuning](#performance-tuning))

#### 7. Database Connection Errors

**Verify connection**:
```bash
mysql -h $MYSQL_HOST -P $MYSQL_PORT -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DB -e "SHOW TABLES;"
```

**Common fixes**:
- Check firewall allows port 25060
- Verify credentials in .env
- Test with `--no_db` flag to isolate issue

### Debug Mode

Enable verbose logging:

```bash
# Set Python logging level
export PYTHONUNBUFFERED=1

# Run with max verbosity
python orchestrator.py --local_video test.mp4 --no_db --make_video 2>&1 | tee debug.log
```

Check logs:
```bash
tail -f output/pipeline.log
grep -i "error\|warning\|failed" output/pipeline.log
```

---

## API Reference

### Orchestrator CLI

```
usage: orchestrator.py [-h] [--local_video PATH] [--no_db] [--save_local]
                       [--make_video] [--max_frames N] [--output_dir DIR]
                       [--poll] [--poll_interval SEC] [--max_videos N]
                       [--min_size_mb MB] [--max_size_mb MB]
                       [--locking_mode {1,2,3}] [--jnr_stride N]
                       [--vid_stride N] [--tracking_mode {bytetrack,botsort,sam2}]
                       [--sam2_model {tiny,small,base,large}] [--parallel N]

Football Pipeline Orchestrator

optional arguments:
  -h, --help            Show help message
  --local_video PATH    Path/URL to video file
  --no_db               Skip database operations
  --save_local          Save output to ./output folder
  --make_video          Generate annotated video
  --max_frames N        Limit frames processed
  --output_dir DIR      Output directory path

Polling mode:
  --poll                Enable API polling mode
  --poll_interval SEC   Seconds between polls (default: 60)
  --max_videos N        Max videos to process before exit
  --min_size_mb MB      Minimum video size filter
  --max_size_mb MB      Maximum video size filter
  --parallel N          Concurrent pipeline jobs (default: 1)

Pipeline configuration:
  --locking_mode {1,2,3}    Internal processing mode (default: 2)
  --jnr_stride N            JNR sampling interval
  --vid_stride N            Video frame stride
  --tracking_mode {bytetrack,botsort,sam2}    Tracker backend
  --sam2_model {tiny,small,base,large}       SAM2 model variant
```

### Health Monitor API

```python
from utils.health_monitor import get_health_monitor

health = get_health_monitor()

# Record events
health.record_video_start(video_id="abc123", user_id="user1")
health.record_video_complete(video_id="abc123", frames_processed=7500)
health.record_video_failure(video_id="xyz789", error_msg="Timeout")
health.record_db_query(success=True)
health.record_api_call(success=False)

# Get metrics
status = health.get_health_status()
print(f"Status: {status['status']}")
print(f"Videos processed: {status['metrics']['videos_processed']}")

# Save/print reports
health.save_snapshot('health.json')
health.print_summary()
```

### Direct Pipeline Usage

```python
from orchestrator import run_pipeline

success = run_pipeline(
    video_path="match.mp4",
    output_dir="./output",
    max_frames=5000,
    no_db=True,
    video_id="test123",
    user_id="user1",
    locking_mode=2,
    tracking_mode="bytetrack",
    make_video=False
)

if success:
    print("Pipeline completed successfully")
    # Results in ./output/player_stats.json
```

---

## Advanced Topics

### Custom Model Integration

Replace default models by updating paths in `config.yaml`:

```yaml
env:
  DET_WEIGHTS: "path/to/custom_yolo.pt"
  JNR_WEIGHTS: "path/to/custom_jnr.pt"
```

Models must be compatible with:
- Detection: YOLOv8 format (Ultralytics)
- JNR: PyTorch state dict with ResNet34 architecture

### Extending Statistics

Add custom stats in `stats/metrics.py`:

```python
def calculate_custom_metric(frames, ownership, team_map):
    # Your logic here
    return custom_stats_dict
```

Hook into pipeline in `pipeline_consolidated.py` (line ~1200).

### Database Schema

Results stored in `MatchesVideoAnalysis_test` table:

| Column | Type | Description |
|--------|------|-------------|
| id | INT | Auto-increment primary key |
| matches_video_id | INT | Video ID from matches table |
| user_id | VARCHAR | User who uploaded |
| unique_id | VARCHAR(40) | SHA1 hash of source URL |
| source_url | TEXT | Original video URL |
| task_id | INT | Processing task ID |
| status | ENUM | running, finished, failed |
| analysis | JSON | Full stats output |
| error | TEXT | Error message if failed |
| created_at | TIMESTAMP | Creation time |
| updated_at | TIMESTAMP | Last update time |

---

## Support & Contributing

### Getting Help

1. Check this guide and [TRACKER_SELECTION_GUIDE.md](TRACKER_SELECTION_GUIDE.md)
2. Review logs: `output/pipeline.log`
3. Check health status: `output/health_snapshot.json`
4. Search existing issues on GitHub
5. Contact the team on Slack

### Reporting Bugs

Include:
- Pipeline version
- Full command/configuration used
- Relevant log excerpts
- Video characteristics (resolution, duration, codec)
- Expected vs actual behavior

### Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## Changelog

### 2026-01-29 (Production-Clean)
- ✅ Removed hardcoded credentials
- ✅ Added health monitoring system
- ✅ Archived legacy pipeline
- ✅ Created systemd service for auto-start
- ✅ Complete documentation overhaul

### 2025-12-21 (Phase Stable)
- Hybrid JNR system (ResNet + Qwen)
- Ghost spawn prevention
- Improved team clustering

### 2025-11-15 (Phase 110)
- BoT-SORT tracker integration
- Enhanced xG model
- Goal detection heuristics

---

**Happy Analyzing! ⚽📊**
