# Football Analysis Pipeline

A computer vision pipeline that processes football match video and extracts
per-player statistics: passes, tackles, shots, dribbles, xG, goals,
interceptions, possession, and distance.

> **Status:** In active development — not production-ready. This is a personal /
> internal development repository. Accuracy is still being validated against
> ground-truth benchmarks, and several components are works in progress.

**Input:** a match video (MP4, WebM, AVI)
**Output:** `player_stats.json` — per-player stats, team assignments, jersey
numbers, and match events.

---

## What it does

- **Detection** — YOLOv8 detects players, goalkeepers, referees, and the ball
- **Tracking** — ByteTrack assigns persistent IDs across frames
- **Jersey numbers** — ResNet34 reads shirt numbers (vote-based locking)
- **Team assignment** — HSV color clustering discovers the two kit colors
- **ReID** — OSNet appearance embeddings (optional, `--enable_reid`)
- **Stats engine** — passes, shots, tackles, dribbles, interceptions, xG, goals

---

## Quick start

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Models — place the 5 weight files into a models/ directory (see Models section below)

# Run on a video
python pipeline_consolidated.py \
    --video path/to/match.mp4 \
    --tracking_mode bytetrack \
    --output_dir output/my_run \
    --vid_stride 3 \
    --enable_reid true

# Result
cat output/my_run/player_stats.json | python -m json.tool
```

The video path can also be set in `config.yaml` under `env.SRC_VIDEO`.

---

## Models

Model weights are not stored in this repo (too large). Place these 5 files into a
local `models/` directory — ask the maintainer for the shared model folder:

| File | Purpose |
|------|---------|
| `yolo_player.pt` | player / GK / referee detection |
| `yolo_ball.pt` | ball detection |
| `resnet34_clean.pt` | jersey number recognition |
| `legibility_resnet18.pt` | legibility gate |
| `osnet_x0_25.pth` | OSNet appearance ReID |

---

## Configuration

Three-tier precedence: **CLI args > environment vars > `config.yaml`**.

```yaml
# config.yaml (key parameters)
heuristics:
  DET_CONF: 0.10        # detection confidence
  VID_STRIDE: 3         # process every Nth frame
  NEW_TRACK_THRESH: 0.85 # track-spawn confidence (lower for non-broadcast footage)
```

Database / API credentials go in a local `.env` (see `.env.example`). The real
`.env` is gitignored and must never be committed.

---

## Architecture

```
Video → YOLO detection → ByteTrack tracking → ResNet34 jersey recognition
      → HSV team clustering → OSNet ReID → Stats engine → player_stats.json
```

| Area | File |
|------|------|
| Main pipeline | `pipeline_consolidated.py` |
| Orchestration / job queue | `orchestrator.py` |
| Event detection | `stats/event_logic.py` |
| Stats engine, identity resolution | `stats/metrics.py` |
| Expected goals | `stats/xg.py` |
| Color / team classification | `vision/color_classifier.py`, `vision/team_clustering.py` |
| Tracking | `vision/custom_bytetrack.py` |
| Jersey recognition | `vision/resnet_recognition.py` |
| Pitch projection | `vision/camera.py` |
