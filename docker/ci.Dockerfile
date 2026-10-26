# -------------------------------------------------------------------
# uv binary
# -------------------------------------------------------------------

# Resolve uv version when the CI image is manually built.
# Pin the tag if the manual build fails.
FROM ghcr.io/astral-sh/uv:latest AS uv-bin

# -------------------------------------------------------------------
# CPU / Build / Documentation
# -------------------------------------------------------------------
# xvfb for sphinx-gallery is installed by the documentation jobs.

FROM ubuntu:22.04 AS cpu-ci

COPY --from=uv-bin /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        make \
        git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# GPU / Test
# -------------------------------------------------------------------
# The base image ships cuda-compat-12-4, which lets CUDA 12.4 run on
# older NVIDIA drivers. The NVIDIA Container Toolkit on the runner
# makes those libraries discoverable. PyTorch wheels bundle the rest.

FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS cu124-ci

COPY --from=uv-bin /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        make \
        git \
    && rm -rf /var/lib/apt/lists/*
