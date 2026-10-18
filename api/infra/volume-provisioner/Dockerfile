# Minimal privileged volume-provisioner image (Phase 10).
# Not used by default compose; profile-gated for operators that need
# local loop/cryptsetup instead of host SSH.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    cryptsetup \
    mount \
    util-linux \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Placeholder entrypoint: real command consumer lands with durable volume cmds.
CMD ["python", "-c", "import time; print('volume-provisioner idle — wire durable volume commands'); time.sleep(3600)"]
