FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    bash \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
      torchvision==0.15.1 \
      timm==0.6.7 \
      "numpy<2" \
      Pillow \
      opencv-python-headless \
      plyfile \
      tensorboardX \
      einops \
      tqdm

WORKDIR /3dgs_pipe/thirdparty/diffmvs
