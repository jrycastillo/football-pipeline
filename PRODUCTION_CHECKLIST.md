# Production Go-Live Checklist

Full instructions: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. This is the
tick-list to confirm a server is ready.

## Setup
- [ ] Server meets prerequisites — Ubuntu 20.04+, Python 3.9–3.11, NVIDIA+CUDA, `ffmpeg`, disk (DEPLOYMENT §1)
- [ ] Repo cloned at `/home/ubuntu/football` from branch **`production-v1.0`** (DEPLOYMENT §2)
- [ ] `venv` created + `pip install -r requirements.txt` (§3)
- [ ] Models in `models/` — `yolo_player.pt`, `resnet34_clean.pt`, `nabeel_best.pt` (+ optional `yolo_pitch.pt`) matching `config.yaml` ([MODEL_SETUP.md](MODEL_SETUP.md))
- [ ] `.env` created from `.env.example` with real `MYSQL_*` + `SBG_TOKEN` (§5)
- [ ] `db/schema.sql` applied to the MySQL instance (§6)

## Verify
- [ ] Smoke test writes a non-empty `player_stats.json` (DEPLOYMENT §7)
- [ ] `nvidia-smi` shows the pipeline using the GPU (not CPU)
- [ ] `sudo deployment/install-service.sh` → service active, logs clean (§8)

## Security (must)
- [ ] **MySQL password rotated** (old one was in git history)
- [ ] Fresh `SBG_TOKEN` in `.env` only (none in `config.yaml`)
- [ ] `chmod 600 .env`; no secrets committed
- [ ] Service runs as non-root `ubuntu`

## Essential files present (should already be, on production-v1.0)
- [ ] `orchestrator.py`, `pipeline_consolidated.py`, `config.yaml`, `requirements.txt`
- [ ] `stats/`, `vision/`, `utils/`, `db/schema.sql`, `deployment/`
