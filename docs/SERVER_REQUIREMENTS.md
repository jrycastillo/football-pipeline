# Server Requirements — for Infrastructure Estimate

Sizing info for provisioning a production server. The pipeline is **GPU-bound**.

---

## 1. Current dev/test machine (NOT production)

This is only where we test — a Windows PC with WSL, not a deployment target:

| | |
|---|---|
| OS | Windows + WSL2 (Ubuntu 24.04) |
| CPU | AMD Ryzen 7 5700X — 8 cores / 16 threads |
| RAM | 16 GB |
| GPU | NVIDIA **RTX 3080 Ti (12 GB VRAM)** |
| Disk | 1 TB |
| Python | 3.12 |

---

## 2. Production server requirements (size for this)

| Resource | Recommended | Notes |
|---|---|---|
| **GPU** | NVIDIA **H100 80GB** (ideal) — or A100 / L40S / RTX-class | Config is tuned for H100. More VRAM = more parallel workers = more throughput. This is the main cost lever. |
| **VRAM per worker** | ~4 GB each | 24 GB GPU ≈ 5–6 workers · 80 GB ≈ 10–12 workers |
| **CPU** | 16+ cores | video decode + ffmpeg encoding; scales with worker count |
| **RAM** | 32–64 GB | video buffering + parallel workers |
| **Storage** | 100 GB+ working | models ~350 MB; each match video is 1.4–3.5 GB; plus output clips/JSON |
| **OS** | Ubuntu 20.04+ | with CUDA / NVIDIA drivers + `ffmpeg` |

### Throughput (per config, on H100)
- **1 worker:** ~1.5–2 hours per 90-min match
- **10 parallel workers:** ~10–15 min per match → **~40–60 videos/hour**

Lower-cost options for smaller volume: a single **A100** or **L40S** (middle ground),
or even one **RTX 4090 / 3090-class** box for low volume.

---

## 3. External services it connects to

- **MySQL** — stores results (currently a **DigitalOcean managed MySQL**)
- **ScoutBridge API** — job queue + clip upload

---

## 4. Deployment model

- Runs as a **systemd service** (`orchestrator.py --poll`): polls for new videos,
  processes them, writes stats/events to MySQL, uploads clips.
- Full step-by-step setup: **[DEPLOYMENT.md](DEPLOYMENT.md)** (prerequisites,
  install, models, config, DB schema, service).

---

## 5. To finalize the estimate

The GPU choice depends on **target volume (videos/day)**. Give that number and it
maps directly to GPU + worker count:
- Low volume (a few matches/day) → single mid-range GPU (A100 / L40S / 4090).
- High volume (dozens/day) → H100 with 10+ parallel workers.

Models are downloaded/placed on the server (not in git) — see
[MODEL_SETUP.md](../MODEL_SETUP.md). Secrets go in a server-side `.env`.
