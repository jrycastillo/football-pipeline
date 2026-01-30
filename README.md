# Football Analysis Pipeline ⚽📊

**Production-ready computer vision system for automated football match analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 What It Does

Automatically analyze football match videos to extract:

- ✅ **Player Tracking** - Persistent IDs across the entire match
- ✅ **Team Assignment** - Automatic jersey color detection
- ✅ **Jersey Numbers** - Hybrid CNN + VLM recognition
- ✅ **Ball Tracking** - Frame-by-frame ball detection
- ✅ **Match Events** - Passes, tackles, interceptions, crosses, dribbles
- ✅ **Shot Analysis** - Expected Goals (xG) calculation
- ✅ **Goal Detection** - Automatic goal validation
- ✅ **Statistics** - Comprehensive per-player and team metrics

**Input**: Match video (MP4, AVI, MOV)
**Output**: JSON/CSV with detailed statistics + optional annotated video

---

## 🚀 Quick Start

### 1. Install

```bash
# Clone repository
git clone https://github.com/your-org/football-pipeline.git
cd football-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
nano .env  # Add your database password and API token
```

### 2. Download Models

```bash
# Contact team for model access, then place in models/ directory:
# - models/yolo_player.pt
# - models/yolo_ball.pt
# - models/yolo_pitch.pt
# - models/resnet34_rgb_jnr.pt
```

### 3. Process Your First Video

```bash
# Single video analysis
python orchestrator.py \
    --local_video test_videos/match.mp4 \
    --output_dir ./results \
    --no_db \
    --make_video

# Results will be in ./results/player_stats.json
```

---

## 📚 Documentation

### Getting Started
- **[Pipeline Usage Guide](docs/PIPELINE_USAGE.md)** - Complete command-line reference and examples
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Common commands cheat sheet
- **[Setup Guide](SETUP_STAGING.md)** - Installation and staging server setup

### Advanced Topics
- **[Complete Pipeline Guide](docs/PIPELINE_GUIDE.md)** - Architecture and technical details
- **[Tracker Selection Guide](docs/TRACKER_SELECTION_GUIDE.md)** - Choosing the right tracker
- **[Deployment Guide](deployment/README.md)** - Production deployment with systemd

### Reference
- **[Color Classifier Review](color_classifier_review.md)** - Color detection analysis
- **[Pipeline Issues Fixed](PIPELINE_ISSUES_FIXED.md)** - Recent improvements
- **[Improvements Summary](IMPROVEMENTS_SUMMARY.md)** - Recent enhancements

---

## 💡 Usage Examples

### Local Video Analysis

```bash
python orchestrator.py \
    --local_video match.mp4 \
    --save_local \
    --make_video
```

### Streaming URL

```bash
python orchestrator.py \
    --local_video "https://cdn.example.com/match.mp4" \
    --output_dir ./output \
    --no_db
```

### Production Polling Service

```bash
python orchestrator.py \
    --poll \
    --poll_interval 60 \
    --parallel 3 \
    --tracking_mode bytetrack
```

### Custom Configuration

```bash
python orchestrator.py \
    --local_video match.mp4 \
    --max_frames 5000 \
    --tracking_mode botsort \
    --locking_mode 2 \
    --vid_stride 2 \
    --no_db
```

---

## 🎬 Sample Output

### JSON Statistics (player_stats.json)

```json
{
  "players_flat": [
    {
      "team": "blue",
      "jersey_number": "10",
      "passes": 87,
      "accurate_passes_%": 82.5,
      "shots_on_target": 4,
      "goals": 1,
      "xg_foot_no_opponent": 0.35,
      "dribbles": 12,
      "tackles": 3,
      "time_on_ball_s": 145.2
    }
  ]
}
```

### CSV Output (player_stats.csv)

```csv
team,jersey_number,passes,accurate_passes_%,goals,xg_foot_no_opponent
blue,10,87,82.5,1,0.35
red,7,64,75.0,2,0.52
```

---

## ⚙️ Configuration

Three-tier configuration system (CLI > ENV > config.yaml):

### config.yaml

```yaml
heuristics:
  DET_CONF: 0.10        # Detection confidence
  VID_STRIDE: 1         # Process every Nth frame
  MAX_OWNER_PX: 300     # Ball ownership distance
  JNR_GATE: 0.35        # Jersey number threshold
```

### Environment Variables (.env)

```bash
MYSQL_PASSWORD=your-password
SBG_TOKEN=your-api-token
DET_CONF=0.25
VID_STRIDE=2
```

### CLI Arguments

```bash
--tracking_mode bytetrack    # Tracker: bytetrack/botsort/sam2
--max_frames 3000             # Limit processing
--parallel 3                  # Concurrent jobs
--make_video                  # Generate annotated output
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│  ORCHESTRATOR   │  ← API Polling, Job Queue, Health Monitoring
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         PIPELINE CONSOLIDATED               │
│                                             │
│  Detection → Tracking → Team → JNR → Stats │
│  (YOLOv8)   (ByteTrack) (K-means) (ResNet) │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  OUTPUT         │  ← JSON, CSV, Annotated Video
└─────────────────┘
```

**Key Components:**
- **Detection**: YOLOv8 for players, ball, referees
- **Tracking**: ByteTrack (default), BoT-SORT, or SAM2
- **Team Assignment**: HSV color clustering
- **JNR**: Hybrid ResNet32 + Qwen2.5-VL
- **Stats**: Event detection, xG calculation, goal validation

---

## 📊 Performance

**Test Setup**: 1080p video, 5 minutes, 25 FPS (7500 frames), RTX 3090

| Mode | Time | FPS | ID Switches |
|------|------|-----|-------------|
| **Speed** (ByteTrack, stride=2) | 4 min | 31 | ~8% |
| **Balanced** (ByteTrack, stride=1) | 7 min | 18 | ~5% |
| **Accuracy** (BoT-SORT, stride=1) | 12 min | 10 | ~2% |

---

## 🔧 Troubleshooting

### Common Issues

**"MYSQL_PASSWORD not set"**
→ Add password to `.env` file

**CUDA Out of Memory**
→ Reduce `--parallel` workers or use `--tracking_mode bytetrack`

**No detections**
→ Lower `DET_CONF` in config.yaml (try 0.05)

**Slow processing**
→ Increase `--vid_stride` or use `--max_frames`

**Poor jersey recognition**
→ Lower `JNR_GATE` threshold or check video resolution

See [Troubleshooting Guide](docs/PIPELINE_GUIDE.md#troubleshooting) for detailed solutions.

---

## 🚢 Production Deployment

### Systemd Service (Auto-Start on Boot)

```bash
# Install service
sudo deployment/install-service.sh

# Monitor logs
journalctl -u football-pipeline -f

# Check status
systemctl status football-pipeline
```

See [deployment/README.md](deployment/README.md) for full guide.

### Health Monitoring

Built-in metrics tracking:

```bash
# View health snapshot
cat output/health_snapshot.json

# Metrics include:
# - Videos processed/failed/running
# - Avg processing time & frame rate
# - Database & API call stats
# - Recent errors
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **ByteTrack** by ByteDance
- **BoT-SORT** by NirAharon
- **Qwen2.5-VL** by Alibaba Cloud
- **SAM2** by Meta AI

---

## 📞 Support

- **Documentation**: [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/football-pipeline/issues)
- **Email**: support@yourorg.com
- **Slack**: #football-pipeline

---

**Made with ⚽ by the Football Analytics Team**

*Last Updated: 2026-01-29*
