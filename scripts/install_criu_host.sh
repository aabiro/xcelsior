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
  CRIU_SRC="${CRIU_SRC:-/mnt/storage/tools/criu-latest}"
  if [ ! -f "$CRIU_SRC/criu/criu" ]; then
    sudo apt-get update
    sudo apt-get install -y build-essential git libprotobuf-dev libprotobuf-c-dev \
      protobuf-c-compiler protobuf-compiler python3-protobuf libnl-3-dev \
      libnl-route-3-dev libcap-dev libnet1-dev libbsd-dev || true
    git clone --depth 1 https://github.com/checkpoint-restore/criu.git "$CRIU_SRC"
    make -C "$CRIU_SRC" -j"$(nproc)"
  fi
  sudo cp "$CRIU_SRC/criu/criu" /usr/local/bin/criu
  sudo setcap cap_checkpoint_restore,cap_sys_ptrace,cap_sys_admin+eip /usr/local/bin/criu
  criu --version | head -1
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