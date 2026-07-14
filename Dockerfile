# syntax=docker/dockerfile:1
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Xcelsior API" \
      org.opencontainers.image.description="FastAPI backend for Xcelsior — Canadian sovereign GPU marketplace" \
      org.opencontainers.image.url="https://xcelsior.ca" \
      org.opencontainers.image.source="https://github.com/aabiro/xcelsior" \
      org.opencontainers.image.vendor="Xcelsior" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system deps for psycopg (libpq), SSH (scheduler needs ssh client), ping (health monitor),
# curl (healthcheck), cryptsetup/e2fsprogs/util-linux (LUKS volume provisioning on colocated NFS).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    libpq5 openssh-client iputils-ping tmux \
    cryptsetup e2fsprogs util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY . .
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# Ensure templates and migrations are present
RUN test -d templates && test -d migrations

ENV PYTHONUNBUFFERED=1
EXPOSE 9500 9501

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -sf http://localhost:${XCELSIOR_API_PORT:-9500}/healthz || exit 1

CMD ["gunicorn", "api:app", "-c", "gunicorn.conf.py"]
