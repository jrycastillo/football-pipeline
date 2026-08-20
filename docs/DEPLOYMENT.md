# Deployment Guide — Football Analysis Pipeline

End-to-end setup for DevOps. Takes a fresh Linux (GPU) server to a running pipeline.

The pipeline ingests a football match video and writes per-player stats + match
events (JSON, and optionally to MySQL) plus event clips.

- **`orchestrator.py`** — production entrypoint: polls for jobs, runs the pipeline,
  writes results to the DB, uploads clips. Runs as a systemd service.
- **`pipeline_consolidated.py`** — the core pipeline; can also be run directly on a
  local file (no DB) for testing.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Ubuntu 20.04+ / Debian 11+ |
| **Python** | 3.9–3.11 (torch ≥ 2.0) |
| **GPU** | NVIDIA + CUDA drivers for production (H100/A100/RTX). CPU works but is very slow. Apple MPS supported for dev only. |
| **ffmpeg** | Required for H.264 clip encoding: `sudo apt install -y ffmpeg` |
| **MySQL** | 8.0 (managed DB is fine) — only for DB/orchestrator mode |
| **Disk** | Models ~350 MB + working space for videos/outputs (tens of GB) |

Verify GPU: `nvidia-smi` should list the card.

---

## 2. Clone (production branch)

```bash
sudo mkdir -p /home/ubuntu && cd /home/ubuntu
git clone -b production-v1.0 https://github.com/jrycastillo/football.git
cd football
```
> The systemd unit expects the repo at `/home/ubuntu/football`. To use another
> path, edit `WorkingDirectory`/`ExecStart` in `deployment/football-pipeline.service`.

---

## 3. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
For a specific CUDA build of PyTorch, install torch/torchvision from the matching
`--index-url` **before** `pip install -r requirements.txt` (see pytorch.org).

---

## 4. Models

Weights are **not** in git. Follow **[MODEL_SETUP.md](../MODEL_SETUP.md)** — place the
3 required files (+1 optional) in `models/`:

```
models/yolo_player.pt        # detection      (required)
models/resnet34_clean.pt     # jersey numbers (required)
models/nabeel_best.pt        # ball + goal    (required)
models/yolo_pitch.pt         # homography     (optional)
```

**Model bundle (all weights, ~670 MB):** https://drive.google.com/file/d/11MjXkc00fXAnXBqHYtL-WYN3zDaJs5Ml/view?usp=sharing

```bash
pip install gdown
gdown 11MjXkc00fXAnXBqHYtL-WYN3zDaJs5Ml -O models_deploy.tar.gz
sha256sum -c <<< "4511fded7d10bf8ea340116ee55b0c8c2a9cbfc4cffcaad34229799c6696c186  models_deploy.tar.gz"
mkdir -p models && tar -xzf models_deploy.tar.gz -C models/
```
See **[MODEL_SETUP.md](../MODEL_SETUP.md)** for details. For a fully automated
deploy, mirror the bundle to object storage (S3/GCS/Spaces) and pull from there.

---

## 5. Configuration & secrets

- **`config.yaml`** (in the repo) holds non-secret settings: model paths, detection
  and stats thresholds. It's tuned for H100. `config_h100.yaml` is an alternate.
- **Secrets go in `.env`** — never in `config.yaml`. `.env` overrides config
  (precedence: **CLI > env/.env > config.yaml**).

```bash
cp .env.example .env
nano .env     # set MYSQL_* (incl. MYSQL_PASSWORD) and SBG_TOKEN
```
`.env` is git-ignored. Required for DB/poll mode: `MYSQL_HOST/PORT/USER/PASSWORD/DB`,
`TABLE_NAME`, `SBG_BASE`, `SBG_TOKEN`.

---

## 6. Database (DB/orchestrator mode only)

Create the schema on your MySQL instance:
```bash
mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p "$MYSQL_DB" < db/schema.sql
```
Creates the 6 tables the front-end reads (`matches`, `ai_player_stats`, `events`,
`clips`, `roster_players`, `roster_known_stats`). Field reference:
[docs/AI_OUTPUT_SCHEMA.md](AI_OUTPUT_SCHEMA.md).

---

## 7. Smoke test (before the service)

Run one local video with no DB, to confirm models + GPU work:
```bash
source venv/bin/activate
python pipeline_consolidated.py \
  --tracking_mode bytetrack --output_dir output/smoke --vid_stride 3
# (video path from config.yaml env.SRC_VIDEO; or use the orchestrator:)
python orchestrator.py --local_video path/to/clip.mp4 --no_db --save_local --output_dir output/smoke
```
Success = `output/smoke/player_stats.json` is written and non-empty.

---

## 8. Run in production (systemd service)

```bash
sudo deployment/install-service.sh
```
This installs `deployment/football-pipeline.service`, enables auto-start on boot,
and launches:
```
python orchestrator.py --poll --poll_interval 60 --parallel 3 \
                       --locking_mode 2 --tracking_mode bytetrack
```
It polls every 60 s for new videos and processes 3 in parallel. Tune `--parallel`
to the GPU (H100 80 GB handles ~10; each worker ≈ 4 GB VRAM — see notes in
`config.yaml`).

**Manage the service:**
```bash
sudo systemctl status football-pipeline
sudo systemctl restart football-pipeline
journalctl -u football-pipeline -f          # live logs
tail -f logs/pipeline.log logs/pipeline-error.log
```

---

## 9. Monitoring
```bash
watch -n 1 nvidia-smi        # GPU/VRAM
tail -f logs/pipeline.log    # pipeline progress
```
Throughput (H100, VID_STRIDE=3, DET 640): ~1.5–2 h/match at 1 worker, ~10–15 min
with 10 parallel workers.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: models/...pt` | Model missing/misnamed — match `config.yaml` (§4) |
| Clips not produced / codec error | Install `ffmpeg` (§1) |
| DB connection refused | Check `.env` MYSQL_* + that `db/schema.sql` was applied |
| Runs on CPU (very slow) | CUDA/driver not visible — check `nvidia-smi`, reinstall torch CUDA build |
| Poll finds no videos | Check `SBG_BASE`/`SBG_TOKEN` in `.env` and the queue |

---

## 11. Security checklist (do before go-live)

- [ ] **Rotate the MySQL password** — an old one was committed to git history and is compromised.
- [ ] **Set a fresh `SBG_TOKEN` in `.env`** — `config.yaml` no longer carries a token (removed); the old one was leaked + expired.
- [ ] `.env` is present, git-ignored, and world-unreadable (`chmod 600 .env`).
- [ ] No secrets in `config.yaml` (only non-secret URLs/paths/thresholds).
- [ ] Service runs as a non-root user (`ubuntu`); `models/`, `logs/`, `output/` writable by it.
