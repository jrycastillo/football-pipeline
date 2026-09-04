# Docker deployment (GPU droplet)

Container build for the ephemeral-droplet flow: an upload triggers a droplet,
the container processes one match, the droplet is destroyed.

## Build

Model weights are not in git, so download them into `./models` first:

```bash
pip install gdown
gdown 11MjXkc00fXAnXBqHYtL-WYN3zDaJs5Ml -O models.zip
unzip models.zip -d models/
```

Then build:

```bash
docker build -t scoutbridge-pipeline:v1.0 .
```

Expect roughly 6-7 GB uncompressed (~3-4 GB pushed). The bulk is the CUDA
torch base image, not our code.

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

Recommended: **GitHub Container Registry (ghcr.io)** - free for private images
and tied to the repo this code already lives in.

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
