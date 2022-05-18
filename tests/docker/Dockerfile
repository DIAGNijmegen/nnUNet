FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04 AS base

# This part is required to install the new (rotating) public keys for this image
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

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
