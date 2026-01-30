# Football Pipeline - Quick Reference

**Quick command reference for common operations**

---

## Basic Commands

### Process Single Video

```bash
# Basic
python orchestrator.py --local_video video.mp4 --save_local

# With debug video
python orchestrator.py --local_video video.mp4 --save_local --make_video

# First 500 frames only
python orchestrator.py --local_video video.mp4 --save_local --max_frames 500
```

### Production Polling

```bash
# Start polling (3 parallel workers)
python orchestrator.py --poll --parallel 3

# Custom poll interval
python orchestrator.py --poll --parallel 3 --poll_interval 30

# Filter by size (50MB - 2GB)
python orchestrator.py --poll --parallel 3 --min_size_mb 50 --max_size_mb 2000
```

---

## Common Arguments

| Argument | Example | Description |
|----------|---------|-------------|
| `--local_video` | `video.mp4` | Process single video |
| `--poll` | - | Enable polling mode |
| `--parallel` | `3` | Number of workers (polling mode) |
| `--save_local` | - | Save to ./output |
| `--output_dir` | `/path/to/out` | Custom output location |
| `--make_video` | - | Generate debug video |
| `--max_frames` | `500` | Limit frames (testing) |
| `--vid_stride` | `2` | Skip every other frame (2x faster) |
| `--locking_mode` | `2` | Identity mode: 1=Fast, 2=Balanced, 3=Accurate |
| `--tracking_mode` | `bytetrack` | Tracker: bytetrack, botsort, sam2 |
| `--no_db` | - | Skip database (testing) |

---

## Monitoring

```bash
# View logs
tail -f output/pipeline.log

# Database status
python check_db_status.py

# Live monitor
./monitor_processing.sh

# Health metrics
cat output/health_snapshot.json
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU
export CUDA_VISIBLE_DEVICES=''
python orchestrator.py --local_video video.mp4 --save_local

# Or skip frames
python orchestrator.py --local_video video.mp4 --save_local --vid_stride 2
```

### Database Connection
```bash
# Test connection
python test_db_connection.py

# Check .env
cat .env | grep MYSQL_PASSWORD
```

### No Statistics
```bash
# Diagnose
python test_stats_computation.py output/player_stats.json

# Check logs
tail -100 output/pipeline.log | grep -i error
```

---

## Testing

```bash
# Verify setup
python check_setup.py

# Test database
python test_db_connection.py

# Test color classifier
python test_color_classifier.py

# Test stats computation
python test_stats_computation.py output/player_stats.json
```

---

## Performance Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **Fast** | `--vid_stride 2 --tracking_mode bytetrack` | Quick preview |
| **Balanced** | `--locking_mode 2 --tracking_mode bytetrack` | **Default** |
| **Accurate** | `--locking_mode 3 --tracking_mode botsort` | Critical matches |
| **Low Memory** | `--vid_stride 2` | Limited GPU memory |

---

## Production Deployment

```bash
# Install systemd service
sudo ./deployment/install-service.sh

# Start service
sudo systemctl start football-pipeline

# View logs
sudo journalctl -u football-pipeline -f

# Stop service
sudo systemctl stop football-pipeline
```

---

## Output Files

```
output/
├── player_stats.json       # Main statistics
├── player_stats.csv        # CSV format
├── match_kits.json         # Team colors
├── debug_video.mp4         # Annotated video (if --make_video)
└── pipeline.log            # Processing logs
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Credentials (DB, API tokens) |
| `config.yaml` | Default settings (thresholds, model paths) |
| `deployment/football-pipeline.service` | Systemd service config |

---

## Locking Modes

| Mode | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| **1** | ⚡⚡⚡ | ⭐⭐ | Quick tests |
| **2** | ⚡⚡ | ⭐⭐⭐ | **Production (default)** |
| **3** | ⚡ | ⭐⭐⭐⭐ | Critical accuracy |

---

## Tracking Modes

| Mode | Speed | Accuracy | ReID | Best For |
|------|-------|----------|------|----------|
| **bytetrack** | ⚡⚡⚡ | ⭐⭐⭐ | No | **Default/Production** |
| **botsort** | ⚡⚡ | ⭐⭐⭐⭐ | Yes | High accuracy |
| **sam2** | ⚡ | ⭐⭐⭐⭐⭐ | Yes | Experimental |

---

## Common Workflows

### Quick Test
```bash
python orchestrator.py \
    --local_video test.mp4 \
    --save_local \
    --max_frames 500 \
    --no_db
```

### Full Match (Local)
```bash
python orchestrator.py \
    --local_video match.mp4 \
    --output_dir output/match_001 \
    --locking_mode 2 \
    --make_video
```

### Production Server
```bash
python orchestrator.py \
    --poll \
    --parallel 3 \
    --poll_interval 60 \
    --locking_mode 2
```

### High Accuracy
```bash
python orchestrator.py \
    --local_video important.mp4 \
    --save_local \
    --locking_mode 3 \
    --tracking_mode botsort \
    --make_video
```

---

For detailed documentation, see [PIPELINE_USAGE.md](PIPELINE_USAGE.md)
