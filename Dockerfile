# ScoutBridge Pro - football analysis pipeline (GPU)
#
# Base image already ships a CUDA-enabled torch/torchvision. That is deliberate:
# building torch against CUDA by hand is the single most common way this image
# ends up silently running on CPU.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg for video I/O; libgl1 + libglib2.0-0 because opencv is imported at
# module scope and fails with "ImportError: libGL.so.1" without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies EXCEPT torch/torchvision. requirements.txt pins
# `torch>=2.0.0`, which pip would happily satisfy by pulling the CPU wheel
# over the CUDA build above - the run then "succeeds" at ~50x slower.
COPY requirements.txt .
RUN grep -viE '^(torch|torchvision)([>=<~!]|$)' requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
    && python -c "import torch; assert torch.version.cuda, 'CUDA torch was replaced by a CPU build'"

# Model weights (~670 MB). Not in git, so they must be downloaded into ./models
# before building - see docs/DOCKER.md. Baked in on purpose: an ephemeral
# droplet should not re-download weights on every match.
COPY models/ /app/models/

COPY . /app

# Behaviour flags baked in so they cannot be lost by a hand-written .env.
# Both default to OFF in code; every result we have validated was produced with
# them ON. CLIPPER_NO_BOXES also drops the player boxes from event clips.
ENV CLIPPER_RECALL=1 \
    CLIPPER_NO_BOXES=1 \
    SCOREBOARD_GOALS=1

# Fail loudly if the GPU is not visible, rather than burning droplet time on CPU.
HEALTHCHECK NONE
ENTRYPOINT ["python", "orchestrator.py"]
