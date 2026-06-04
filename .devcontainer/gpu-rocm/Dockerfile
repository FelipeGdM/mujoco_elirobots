FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=symlink

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

RUN pip install uv 

ADD pyproject.toml .

ADD uv.lock .

RUN uv sync --no-install-project --inexact

# download physx GPU binary via sapien
RUN python3 -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"
