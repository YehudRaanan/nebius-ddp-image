# Right-sized training image for multi-node PyTorch DDP (Nebius MLOps Ex1).
#
# Base: official PyTorch + CUDA 12.6 runtime. It bundles NCCL (so DDP and the
# NCCL init logs the rubric wants work out of the box) and is ~7-9 GB vs the
# ~20 GB NGC image — fast to build on GitHub runners and fast to pull.
#
# We pre-install the OS packages SkyPilot's Kubernetes runtime needs (rsync,
# curl, wget, netcat, fuse, sshd). Without these, SkyPilot tries to apt-get
# them at launch time; on a slow apt mirror that download times out and breaks
# the pod. Baking them in removes the runtime download entirely.
#
# train.py is NOT baked in: SkyPilot syncs it at runtime via `workdir: .`,
# so we install only the libraries train.py actually imports.
#
# Published (via GitHub Actions) to: ghcr.io/yehudraanan/nebius-trainer
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      rsync curl wget netcat-openbsd fuse \
      openssh-server openssh-client \
      git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      transformers \
      datasets \
      accelerate

CMD ["bash"]
