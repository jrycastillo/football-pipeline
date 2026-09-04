# Docker deployment (GPU droplet)

Container build for the ephemeral-droplet flow: an upload triggers a droplet,
the container processes one match, the droplet is destroyed.

## Build

Model weights are not in git, so download them into `./models` first:

```bash
pip install gdown
gdown 11MjXkc00fXAnXBqHYtL-WYN3zDaJs5Ml -O models.tar.gz
mkdir -p models && tar -xzf models.tar.gz -C models/
```

The bundle is a gzipped tar (not a zip) whose entries sit at the archive root,
so it extracts straight into `models/`. Expect 752 MB across 10 weights plus
`MODELS_MANIFEST.txt`, which lists which are required vs feature-gated.

Then build:

```bash
docker build -t scoutbridge-pipeline:v1.0 .
```

Measured result: **12.9 GB uncompressed**. Breakdown:

| Layer | Size |
|---|---|
| `/opt/conda` (from the pytorch base image) | 5.94 GB |
| our pip dependencies | 1.21 GB |
| model weights | 788 MB |
| ffmpeg + libgl apt layer | 406 MB |

The conda environment shipped inside `pytorch/pytorch` is the single largest
cost - larger than everything we add put together. Swapping to an
`nvidia/cuda:*-runtime` base and installing torch from the PyTorch wheel index
would likely save 3-4 GB, at the cost of doing the torch/CUDA install by hand.
Not attempted here: the current image is verified working, and that change
risks the failure mode this Dockerfile exists to avoid.

## Run

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v /mnt/output:/app/output \
  scoutbridge-pipeline:v1.0 \
    --local_video "<spacesURL>" \
    --video_id "<real_id>" --user_id "<real_user_id>" \
    --write_db --clip_events shot,goal,assist,goal_restart \
    --tracking_mode bytetrack
```

`--gpus all` requires the NVIDIA Container Toolkit on the host. Without it the
container starts, finds no GPU, and falls back to CPU - the run still exits 0,
just many times slower. Verify on the host before the first real match:

```bash
docker run --rm --gpus all scoutbridge-pipeline:v1.0 \
  -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Exit code 0 means success (safe to destroy the droplet); non-zero means the run
failed and the droplet should be kept for log inspection.

## Configuration

Behaviour flags (`CLIPPER_RECALL`, `CLIPPER_NO_BOXES`, `SCOREBOARD_GOALS`) are
baked into the image as ENV. They default to OFF in code, and every result we
have validated was produced with them ON, so the image sets them rather than
relying on a hand-written `.env`.

`--env-file` should therefore carry **credentials only** (DB, ScoutBridge,
Spaces). Keeping secrets out of the image means the image can live in a
registry without leaking the database password.

## Registry

At 12.9 GB this image does not fit any provider's free private tier, so the
choice is about which small bill to pay:

| Option | Cost | Notes |
|---|---|---|
| No registry - build on the snapshot droplet | $0 | Simplest. Loses versioning and reproducible pulls. |
| GHCR, paid storage | ~$3/mo | GitHub's free private-package allowance is 500 MB; beyond that storage is billed per GB. |
| DigitalOcean Container Registry, Professional | $20/mo | 100 GB. Basic (5 GB) does not fit. Same-region pulls. |

Given the image is pre-pulled into a snapshot rather than fetched per match,
transfer volume is low and storage dominates - which makes GHCR the cheaper
registry. If versioning is not wanted at all, building directly on the droplet
that becomes the snapshot is free and legitimate.

Using GHCR:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u jrycastillo --password-stdin
docker tag scoutbridge-pipeline:v1.0 ghcr.io/jrycastillo/scoutbridge-pipeline:v1.0
docker push ghcr.io/jrycastillo/scoutbridge-pipeline:v1.0
```

Alternative if everything should stay inside DigitalOcean: DigitalOcean
Container Registry. Note the free tier is 500 MB, which this image does not fit;
Basic ($5/mo, 5 GB) is tight once there is more than one tag, so Professional
($20/mo) is the realistic tier.

### Pre-pull the image into the droplet snapshot

Do not pull the image on every match. Build the droplet snapshot with the image
already pulled, so a per-match boot does not depend on registry auth and a
multi-GB download succeeding before any work starts. Refresh the snapshot when a
new image tag ships.

Tag images explicitly (`v1.0`, `v1.1`). Avoid `latest` for production runs - it
makes it impossible to tell which build produced a given match's stats.

## Verification status

Built and checked on a CPU-only VPS (no GPU available; the RTX 3080 Ti worker
was offline). What that does and does not establish:

Verified:
- Image builds clean from this Dockerfile
- `torch 2.5.1+cu121` present and `torch.version.cuda == 12.1` - the CUDA build
  survived the pip step rather than being replaced by a CPU wheel
- `cv2 5.0.0`, `ultralytics 8.4.138`, sklearn, pymysql, yaml all import
- Model weights baked in (752 MB, 11 files)
- `CLIPPER_RECALL`, `CLIPPER_NO_BOXES`, `SCOREBOARD_GOALS` all set to 1
- Entrypoint runs and exits 0

NOT verified - requires an actual GPU host:
- `torch.cuda.is_available()` (necessarily False on the build machine)
- `--gpus all` passing a GPU through
- Any real match run, so nothing about speed or the full-match OOM question

Run the toolkit check in the Run section as the first thing on the droplet. A
missing NVIDIA Container Toolkit does not fail loudly: the container starts,
finds no GPU, silently uses CPU, and still exits 0.
