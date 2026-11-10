# syntax=docker/dockerfile:1
# ghcr.io/xcelsior-gpu/serverless-base:py3.12 — queue-mode custom worker base.
#
# Build context is api/ (the COPY paths below are relative to it). Publish with
# api/scripts/publish_serverless_base.sh, which smoke-tests the echo example
# against this image before pushing.
#
# There is no CUDA toolkit in this image, and the tag says so: it used to read
# "cuda12.4-py3.12" over a plain python:3.12-slim base. GPU access comes from
# the host's NVIDIA container runtime, which injects the driver; frameworks
# such as torch ship their CUDA runtime inside their wheels. A worker that
# needs the CUDA toolkit itself should start from an nvidia/cuda image and
# copy serverless/worker_sdk in, as this file does.
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="xcelsior serverless-base" \
      org.opencontainers.image.description="Base image for Xcelsior serverless custom workers: the worker SDK (queue handler, ASGI mode, /healthz, fitness checks) on Python 3.12." \
      org.opencontainers.image.source="https://github.com/xcelsior-gpu/serverless-base" \
      org.opencontainers.image.documentation="https://docs.xcelsior.ca" \
      org.opencontainers.image.url="https://xcelsior.ca" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/xcelsior

# Copy control-plane worker SDK (mounted at build time from repo root)
COPY serverless/worker_sdk ./serverless/worker_sdk
COPY serverless/__init__.py ./serverless/__init__.py

ENV PYTHONPATH=/opt/xcelsior
ENV XCELSIOR_HTTP_PORT=8080
ENV HF_HOME=/xcelsior-cache/huggingface

# Pinned so two builds of the same commit ship the same SDK dependencies.
# These were the uv.lock versions when the image was first published; the
# image does not have to track the API's lock, but bump them deliberately.
RUN pip install --no-cache-dir httpx==0.28.1 uvicorn==0.40.0 fastapi==0.135.4

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -sf http://127.0.0.1:${XCELSIOR_HTTP_PORT}/healthz || exit 1
