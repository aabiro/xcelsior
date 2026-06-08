# syntax=docker/dockerfile:1
# xcelsior/serverless-base:cuda12.4-py3.12 — queue-mode custom worker base
FROM python:3.12-slim-bookworm AS runtime

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

RUN pip install --no-cache-dir httpx uvicorn fastapi

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -sf http://127.0.0.1:${XCELSIOR_HTTP_PORT}/healthz || exit 1