#!/usr/bin/env bash
# Start the Xcelsior worker agent locally.
# Reads configuration from .env in the project root.
set -euo pipefail

cd "$(dirname "$0")/.."

# Ensure .env exists
if [[ ! -f .env ]]; then
    echo "ERROR: .env file not found. Copy .env.example and fill in your values."
    exit 1
fi

# Source the .env manually (python-dotenv handles it too, but this ensures shell sees it)
set -a
source .env
set +a

# Validate required vars
if [[ -z "${XCELSIOR_HOST_ID:-}" ]]; then
    echo "ERROR: XCELSIOR_HOST_ID is not set in .env"
    exit 1
fi
if [[ -z "${XCELSIOR_SCHEDULER_URL:-}" ]]; then
    echo "ERROR: XCELSIOR_SCHEDULER_URL is not set in .env"
    exit 1
fi

echo "┌─────────────────────────────────────────────┐"
echo "│  Xcelsior Worker Agent                       │"
echo "│  Host ID:  ${XCELSIOR_HOST_ID}              "
echo "│  Scheduler: ${XCELSIOR_SCHEDULER_URL}       "
echo "│  Cost/hr:   \$${XCELSIOR_COST_PER_HOUR:-0.50} CAD"
echo "└─────────────────────────────────────────────┘"

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi query failed)"
else
    echo "WARNING: nvidia-smi not found — GPU detection will rely on NVML"
fi

echo ""
echo "Starting worker agent..."
exec python3 worker_agent.py "$@"
