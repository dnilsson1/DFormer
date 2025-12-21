# Dockerfile for DFormer (GPU, PyTorch 2.1.2 + CUDA 11.8)
# Notes:
# - This image expects to be run with GPU access from the host, e.g.:
#     docker run --gpus all ...
#   and the host must have the NVIDIA Container Toolkit configured. The container
#   cannot install or provide host NVIDIA drivers; if you see "Found no NVIDIA driver"
#   that means the host or the docker runtime isn't configured with GPU support.

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential ca-certificates \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN python -m pip install --upgrade pip

# Install mmcv wheel that matches torch 2.1 + cu118 and other repo deps.
# We install packages first and then force-reinstall a pinned numpy 1.x at the end
# to ensure any transitive dependency that pulled numpy>=2 is overridden. This
# prevents the runtime error where C-extensions compiled against NumPy 1.x are
# incompatible with NumPy 2.x.
# tensorboardX: logging library, tensorboard: web interface & CLI tool
# timm: PyTorch Image Models library required for DFormerv2
RUN pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html && \
    pip install --no-cache-dir tqdm opencv-python scipy tensorboardX tensorboard tabulate easydict ftfy regex timm

# Ensure a stable NumPy 1.x is installed last (choose a specific 1.x release).
# This does a force-reinstall so it wins over any prior install. Use a 1.25.x
# release that is available for the image's Python version.
RUN pip install --no-cache-dir --upgrade --force-reinstall numpy==1.25.2

# copy repo (you can instead mount the repo at runtime for development)
COPY . /workspace

# default workdir
WORKDIR /workspace

# entrypoint is bash to allow interactive use
ENTRYPOINT ["/bin/bash"]