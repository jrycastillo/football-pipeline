# Model Setup

The pipeline loads its model weights from `models/` (relative to the repo root).
`models/` is **git-ignored** — weights are large (~350 MB core) and are stored
outside git. Download them and place them in `models/` before running.

## Required models (must match `config.yaml`)

| File | Role | config.yaml key | Approx size | Required? |
|---|---|---|---|---|
| `yolo_player.pt` | Player/GK/referee detection (YOLOv8) | `DET_WEIGHTS` | ~131 MB | **Yes** |
| `resnet34_clean.pt` | Jersey-number recognition (ResNet34) | `JNR_WEIGHTS` | ~82 MB | **Yes** |
| `nabeel_best.pt` | Ball **and** goal detection (multi-class) | `BALL_MODEL_PATH`, `GOAL_MODEL_PATH` | ~43 MB | **Yes** |
| `yolo_pitch.pt` | Pitch-keypoint homography | `POSE_WEIGHTS` | ~134 MB | Optional* |

\* **Optional** — only used with `--pitch_homography`. Without it the pipeline runs
normally; pitch homography (distance/xG geometry) just stays off. If you don't have
`yolo_pitch.pt`, leave `POSE_WEIGHTS` as-is and don't pass `--pitch_homography`.

> These are the **current** model names. Older docs referenced
> `resnet34_rgb_jnr.pt` / `yolo_ball.pt` — those are superseded. Always match the
> filenames in `config.yaml`.

## Install

```bash
mkdir -p models
# copy/download the files into models/, e.g.:
#   models/yolo_player.pt
#   models/resnet34_clean.pt
#   models/nabeel_best.pt
#   models/yolo_pitch.pt        # optional
```

Source of the weights: the team's shared storage. **For production, host them in
object storage (S3 / GCS / DigitalOcean Spaces)** and pull them in at deploy time —
not Google Drive, which isn't reliable for automated setup.

## Verify

```bash
python check_setup.py            # checks models/config are in place
# or manually:
ls -lh models/{yolo_player,resnet34_clean,nabeel_best}.pt
```

If a required file is missing or misnamed, the pipeline fails at model load with a
clear "file not found" — check the name against `config.yaml`.
