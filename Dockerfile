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
# curl (healthcheck). Privileged LUKS/NFS tooling (cryptsetup/e2fsprogs/util-linux) deliberately
# lives in the volume-provisioner image (blueprint §19.4) — this image runs unprivileged and
# performs those operations over host SSH, so shipping the binaries here only widens the
# blast radius of a compromise.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    libpq5 openssh-client iputils-ping tmux \
    && rm -rf /var/lib/apt/lists/*

# ── Unprivileged runtime identity (blueprint §19.4 / §21.1) ───────────
# A fixed high UID/GID so host bind mounts and the named data volume can
# be granted to it deterministically across rebuilds.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid ${APP_GID} xcelsior \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home \
       --home-dir /home/xcelsior --shell /usr/sbin/nologin xcelsior \
    && mkdir -p /home/xcelsior/.ssh /data \
    && chown -R ${APP_UID}:${APP_GID} /home/xcelsior /data \
    && chmod 700 /home/xcelsior/.ssh

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY . .
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    HOME=/home/xcelsior

# Ensure templates and migrations are present
RUN test -d templates && test -d migrations

# Precompile to bytecode as root so a read-only root filesystem at runtime
# costs nothing: without this every import would recompile on each start.
RUN python -m compileall -q /app /app/.venv/lib || true

# /app is read-only to the runtime user; only the data volume is writable.
# This is what makes `read_only: true` viable in compose.
RUN chown -R root:${APP_GID} /app && chmod -R g+rX /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 9500 9501

USER ${APP_UID}:${APP_GID}

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -sf http://localhost:${XCELSIOR_API_PORT:-9500}/healthz || exit 1

CMD ["gunicorn", "api:app", "-c", "gunicorn.conf.py"]
