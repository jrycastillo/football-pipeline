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

## Install — production model bundle (Google Drive)

All runtime models are packaged in a single tarball, **`models_deploy.tar.gz`**
(~670 MB): the required models above plus the optional ones (homography re-id,
legibility, fallback digit reader) and a `MODELS_MANIFEST.txt` describing each.

**Download link:** https://drive.google.com/file/d/11MjXkc00fXAnXBqHYtL-WYN3zDaJs5Ml/view?usp=sharing

```bash
# On the deploy server (gdown handles Google Drive's large-file scan prompt;
# a plain wget on the share link returns the HTML warning page, not the file):
pip install gdown
gdown 11MjXkc00fXAnXBqHYtL-WYN3zDaJs5Ml -O models_deploy.tar.gz

# Verify integrity — MUST match:
sha256sum -c <<< "4511fded7d10bf8ea340116ee55b0c8c2a9cbfc4cffcaad34229799c6696c186  models_deploy.tar.gz"

# Extract into models/ (filenames then line up with config.yaml):
mkdir -p models && tar -xzf models_deploy.tar.gz -C models/
```

If you can't use `gdown`, open the link in a browser, download the tarball, and
copy it to the server, then run the `sha256sum` + `tar` steps above.

> **Production note:** Google Drive works for a manual pull like this, but for a
> fully automated deploy the sturdier option is object storage (S3 / GCS /
> DigitalOcean Spaces) — mirror the bundle there and swap the `gdown` line for a
> `aws s3 cp` / `curl`. The Drive link is the current source of record.

## Verify

```bash
python check_setup.py            # checks models/config are in place
# or manually:
ls -lh models/{yolo_player,resnet34_clean,nabeel_best}.pt
```

If a required file is missing or misnamed, the pipeline fails at model load with a
clear "file not found" — check the name against `config.yaml`.
