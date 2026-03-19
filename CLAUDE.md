# Football Match Analysis AI Pipeline

## What This Is

An end-to-end computer vision pipeline that processes football match videos and extracts per-player statistics: passes, tackles, shots, dribbles, xG, goals, interceptions, possession, and more.

**Input:** A football match video file (MP4, WebM, AVI)
**Output:** `player_stats.json` with per-player stats, team assignments, jersey numbers, and match events

---

## Quick Start

```bash
# Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Process a video (local, no database)
python pipeline_consolidated.py \
    --tracking_mode bytetrack \
    --output_dir output/my_run \
    --vid_stride 3

# Video path is set in config.yaml → env.SRC_VIDEO
# Or use orchestrator for API/database mode:
python orchestrator.py --local_video path/to/video.mp4 --no_db --save_local
```

---

## Architecture

```
Video → YOLO Detection → ByteTrack Tracking → ResNet34 JNR → Color Classification → Stats Engine → JSON Output
```

### Pipeline Stages (in order)

1. **YOLO Detection** — YOLOv8 detects objects per frame
   - Classes: `0=ball, 1=goalkeeper, 2=player, 3=referee`
   - Models: `models/yolo_player.pt` (persons), `models/yolo_ball.pt` (ball)
   - Config: `DET_CONF=0.10`, `DET_IMG_SIZE=640`

2. **ByteTrack Tracking** — Assigns persistent track IDs across frames
   - Implementation: `vision/custom_bytetrack.py`
   - Creates track fragments (stride=3 causes 100-1500+ fragments per player)

3. **Jersey Number Recognition (JNR)** — ResNet34 CNN identifies jersey numbers
   - Model: `models/resnet34_rgb_jnr.pt`
   - Vote-based: 3 consistent predictions at confidence >= 0.30 → locked
   - Lock condition: 3 votes + margin >= 1.0

4. **Color Classification** — HSV-based jersey color detection
   - Implementation: `vision/color_classifier.py`
   - Grass mask removes pitch green (H 35-90, S > 70)
   - Pre-mask green jersey detection (S > 80, ratio > 35%)
   - `_FOOTBALL_MERGE` dict collapses 17 HSV labels → 8 football colors
   - Torso ROI: 15-45% height (shoulders only, avoids shorts/pitch)

5. **Team Clustering** — K-means on player colors assigns teams
   - Two teams discovered from dominant jersey colors
   - Kit correction applied retroactively after discovery (~frame 500)

6. **Stats Engine** — Computes all match events
   - `stats/event_logic.py` — Pass, shot, tackle, dribble, interception detection
   - `stats/metrics.py` — StatsEngine, Phase 216 remap (track IDs → jersey numbers)
   - `stats/xg.py` — Expected goals calculation
   - `stats/post_processor.py` — Post-processing and validation

7. **Phase 216 Remap** — Maps ByteTrack fragments to final jersey numbers
   - Top-N merge (`_MERGE_TOP_N=5`): sums events from top 5 fragments per jersey
   - Handles ByteTrack fragmentation where one player gets 100-1500 track fragments

---

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `pipeline_consolidated.py` | Main pipeline — detection, tracking, JNR, color, stats | 2,320 |
| `orchestrator.py` | API orchestration, database integration, job queue | 574 |
| `stats/metrics.py` | StatsEngine, Phase 216 remap, event aggregation | ~1,500 |
| `stats/event_logic.py` | Pass/shot/tackle/dribble event detection logic | ~1,300 |
| `stats/xg.py` | Expected goals model | ~120 |
| `stats/post_processor.py` | Post-processing, validation | ~450 |
| `vision/color_classifier.py` | HSV color classification with grass masking | ~600 |
| `vision/custom_bytetrack.py` | ByteTrack tracker implementation | ~900 |
| `vision/resnet_recognition.py` | ResNet34 JNR service (vote-based) | ~450 |
| `config.yaml` | Main configuration (detection, tracking, stats thresholds) | 89 |

### Important: Two IdentityManager Classes Exist
- `vision/identity_manager.py` — **NOT used** (legacy)
- `pipeline_consolidated.py:484` — **ACTIVE** (inline class)

---

## Configuration

Three-tier precedence: **CLI args > Environment vars > config.yaml**

### config.yaml (key parameters)

```yaml
env:
  SRC_VIDEO: "test_videos/v1_1e942fd8a6344bd.webm"  # Video to process
  DET_WEIGHTS: "models/yolo_player.pt"
  BALL_MODEL_PATH: "models/yolo_ball.pt"
  JNR_WEIGHTS: "models/resnet34_rgb_jnr.pt"

heuristics:
  DET_CONF: 0.10       # Detection confidence (lower = more detections)
  DET_IMG_SIZE: 640    # YOLO input resolution
  VID_STRIDE: 3        # Process every 3rd frame
  MAX_OWNER_PX: 300    # Ball ownership distance (pixels)
  TACKLE_PX: 100       # Tackle proximity threshold (pixels)
  DRIBBLE_MIN_PX: 25   # Minimum dribble displacement
  MAX_GAP: 25          # Ball interpolation gap (frames)
```

### Event Cooldowns (in stats/event_logic.py)

| Event | Cooldown | Description |
|-------|----------|-------------|
| Tackle | 5 seconds | Per-player cooldown between tackle events |
| Shot debounce | 1 second | Minimum gap between shot detections |
| Dribble | 1 second | Per-player cooldown between dribble events |

---

## Output Files

After processing, `output_dir/` contains:

| File | Description |
|------|-------------|
| `player_stats.json` | Final per-player statistics (the main output) |
| `match_kits.json` | Discovered team colors (`{"goalkeepers": [...], "players": [...]}`) |
| `raw_tracks.json` | Raw ByteTrack tracking data |
| `debug_all_frames.json` | Frame-by-frame detection data |
| `output_video.mp4` | Annotated video (if `--make_video`) |
| `pipeline.log` | Processing log |

### player_stats.json Structure

```json
{
  "jersey_number": {
    "player_name": "Player N",
    "jersey_number": N,
    "team": "Green",
    "position": "Player",
    "stats": {
      "passes_total": 23,
      "passes_accurate": 8,
      "accurate_passes_percent": 34.8,
      "tackles_total": 1,
      "shots_on_target_total": 4,
      "goals_total": 2,
      "dribbles_total": 1,
      "xg_foot_no_opponent": 0.31,
      "ball_interceptions_total": 10,
      "total_distance": 112.79,
      "time_on_ball_s": 16.44,
      "touch_frames": 137
    }
  }
}
```

---

## GPU Support

| Platform | Flag | Speed |
|----------|------|-------|
| NVIDIA (CUDA) | Auto-detected | Fastest (H100: ~1.5h per 90-min match) |
| Apple Silicon (MPS) | Auto-detected | ~3-4x faster than CPU, ~50-60% of CUDA quality |
| CPU | Fallback | Very slow, not recommended |

MPS (Metal Performance Shaders) = Apple's GPU acceleration for M-series chips.
YOLO must be explicitly loaded with `.to(device)` — defaults to CPU otherwise.

---

## Known Issues & Limitations (as of Round 13, March 2026)

1. **Team imbalance**: Detects 13 vs 6 players instead of 11 vs 11. White team under-detected on MPS.
2. **Low tackle count**: 6 tackles detected vs ~21 expected. MPS has shorter proximity windows.
3. **Ghost players**: Some players (e.g. #35) show 1000+ observations but 0 stats — detection artifacts.
4. **Pass accuracy low**: Many players <30% accuracy due to ownership mapping noise in crowded areas.
5. **ByteTrack fragmentation**: stride=3 creates 100-1500+ micro-fragments per player. Top-N merge mitigates but doesn't fully solve.
6. **MPS vs CUDA gap**: Local MPS produces ~50-60% of H100 event counts.

---

## Development Workflow

### Branches
- `production-v1.0` — Deployment branch with all production fixes
- `football-pipeline-fixes` — Main development / PR target branch

### Processing a Video

```bash
# 1. Set video path in config.yaml (env.SRC_VIDEO)
# 2. Run pipeline
python pipeline_consolidated.py \
    --tracking_mode bytetrack \
    --output_dir output/my_run \
    --vid_stride 3

# 3. Check results
cat output/my_run/player_stats.json | python -m json.tool
```

### Test Videos Available

| Video | Size | Description |
|-------|------|-------------|
| `v1_1e942fd8a6344bd.webm` | 1.4 GB | Main test video (~105 min) |
| `69a33466fc234db.mp4` | 3.5 GB | V2 test video |
| `f561510bde5e4ca.mp4` | 2.3 GB | V3 test video |
| `clipped_ikorudo_tornadoes.mp4` | 136 MB | Short 5-min test clip |

### Reports

Historical round reports are in `reports/YYYY-MM-DD/`. The latest is:
- `reports/2026-03-18/v1_round13_report.md` — R13 results with full comparison
- `reports/2026-03-18/next_steps.md` — Prioritized improvement plan

---

## Critical Bugs Previously Fixed (do not reintroduce)

1. **Phase 216 pick-primary discards 95% of events** — Fixed with top-N merge (_MERGE_TOP_N=5)
2. **Grass mask eats green jerseys** — Fixed with pre-mask green detection (S>80, ratio>35%)
3. **White jerseys classified as Green** — Fixed with S>70 achromatic threshold + torso ROI 15-45%
4. **First-Color-Wins lock** — Fixed with settle-then-lock (store first, correct at obs #10)
5. **Early tracks locked before kit correction** — Fixed with `apply_kit_correction()` retroactive pass
6. **Pass None-gap** — A→None→B transitions now counted as passes after filtering None segments
7. **GK saves on field players** — Fixed to use `dominant_class == 1` instead of heuristic

---

## Dependencies

Key Python packages (see requirements.txt for full list):
- `ultralytics` — YOLOv8 detection
- `torch` / `torchvision` — PyTorch for ResNet JNR + MPS/CUDA
- `opencv-python` — Video I/O and image processing
- `numpy`, `scipy` — Numerical computation
- `scikit-learn` — K-means clustering for team assignment
- `mysql-connector-python` — Database integration (optional)
