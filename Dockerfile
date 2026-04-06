FROM python:3.12-slim

LABEL org.opencontainers.image.title="Xcelsior API" \
      org.opencontainers.image.description="FastAPI backend for Xcelsior — Canadian sovereign GPU marketplace" \
      org.opencontainers.image.url="https://xcelsior.ca" \
      org.opencontainers.image.source="https://github.com/aabiro/xcelsior" \
      org.opencontainers.image.vendor="Xcelsior" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system deps for psycopg (libpq), SSH (scheduler needs ssh client), and ping (health monitor)
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 openssh-client iputils-ping && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure templates and migrations are present
RUN test -d templates && test -d migrations

ENV PYTHONUNBUFFERED=1
EXPOSE 9500

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9500"]
