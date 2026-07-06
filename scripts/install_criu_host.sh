#!/usr/bin/env bash
# Install CRIU + enable Docker experimental checkpoint on xcelsior GPU hosts.
# Requires NVIDIA driver ≥570 for gpu-criu class (CUDA context restore).
set -euo pipefail

echo "=== NVIDIA driver ==="
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader || true

echo "=== Install CRIU (Ubuntu/Debian) ==="
if command -v criu >/dev/null 2>&1; then
  criu --version | head -1
else
  sudo apt-get update
  sudo apt-get install -y criu || {
    echo "WARN: criu not in apt — build from https://github.com/checkpoint-restore/criu or use cedana"
    exit 1
  }
fi

echo "=== Docker experimental (checkpoint API) ==="
DAEMON_JSON="/etc/docker/daemon.json"
if [ -f "$DAEMON_JSON" ] && grep -q '"experimental"' "$DAEMON_JSON"; then
  echo "daemon.json already has experimental"
else
  echo 'Set {"experimental": true} in /etc/docker/daemon.json and restart docker'
fi

echo "=== Probe ==="
cd "$(dirname "$0")/.."
python -c "from criu_hosts import probe_checkpoint_stack; import json; print(json.dumps(probe_checkpoint_stack(), indent=2))"