# syntax=docker/dockerfile:1
# Base: official PyTorch CUDA 12.4 *runtime* image.
# torch / cuda / cudnn versions are pinned by this tag; do NOT reinstall torch
# from requirements.txt (a PyPI wheel would replace the CUDA build).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=dev

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/src

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host UID/GID so mounted files stay writable.
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
 && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}"

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r /tmp/requirements.txt

USER ${USERNAME}
