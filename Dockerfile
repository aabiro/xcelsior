FROM python:3.12-slim

LABEL org.opencontainers.image.title="Xcelsior API" \
      org.opencontainers.image.description="FastAPI backend for Xcelsior — Canadian sovereign GPU marketplace" \
      org.opencontainers.image.url="https://xcelsior.ca" \
      org.opencontainers.image.source="https://github.com/aabiro/xcelsior" \
      org.opencontainers.image.vendor="Xcelsior" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system deps for psycopg (libpq), SSH (scheduler needs ssh client), ping (health monitor), curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 openssh-client iputils-ping tmux curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure templates and migrations are present
RUN test -d templates && test -d migrations

ENV PYTHONUNBUFFERED=1
EXPOSE 9500 9501

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -sf http://localhost:${XCELSIOR_API_PORT:-9500}/healthz || exit 1

CMD ["gunicorn", "api:app", "-c", "gunicorn.conf.py"]
