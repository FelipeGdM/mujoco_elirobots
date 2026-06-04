FROM python:3.12-trixie

RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-tools \
    wget \
    xvfb \
    fish \
    mesa-vulkan-drivers \
    ffmpeg \
    libsm6 \
    libxext6 \
    libvulkan-dev \
    wget \
    libglvnd-dev \
    libgl1 \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/usr/bin/fish", "-c"]

RUN chsh -s /usr/bin/fish root

RUN pip config set global.break-system-packages true

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install uv 

ADD pyproject.toml .

ADD uv.lock .

ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_LINK_MODE=symlink

RUN uv sync --no-install-project --inexact

# download physx GPU binary via sapien
RUN python3 -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"
