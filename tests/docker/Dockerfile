FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04 AS base

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  wget \
  unzip \
  libopenblas-dev \
  python3.8 \
  python3.8-dev \
  python3-pip \
  && \
  apt-get clean autoclean && \
  apt-get autoremove -y && \
  rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.8 -m pip install --no-cache-dir --upgrade pip

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN python3.8 -m pip install --no-cache-dir -r /tmp/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

RUN useradd -ms /bin/bash user

USER user
