#!/usr/bin/env bash
# Live-upstream evidence lane (rows 1/2) — runs before capture_row_evidence.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${XCELSIOR_GOAL_SCRATCH:-/tmp/grok-goal-6f86c7cfe9c2/implementer}"
mkdir -p "$SCRATCH"
cd "$ROOT"

export XCELSIOR_GOAL_SCRATCH="$SCRATCH"
export XCELSIOR_LMCACHE_MESH_HOSTS="${XCELSIOR_LMCACHE_MESH_HOSTS:-lmcache://mesh-host-a:8100,lmcache://mesh-host-b:8100}"

echo "=== fleet live: vLLM probe ===" | tee "$SCRATCH/fleet-live-proof.log"
XCELSIOR_LIVE_VLLM_START_DOCKER="${XCELSIOR_LIVE_VLLM_START_DOCKER:-0}" \
  XCELSIOR_LIVE_VLLM_MODEL="${XCELSIOR_LIVE_VLLM_MODEL:-Qwen/Qwen3-4B-AWQ}" \
  XCELSIOR_LIVE_VLLM_MODEL_PATH="${XCELSIOR_LIVE_VLLM_MODEL_PATH:-/mnt/storage/models/staging/2026H2/hf/Qwen_Qwen3-4B-AWQ}" \
  XCELSIOR_LIVE_VLLM_DRAFT_PATH="${XCELSIOR_LIVE_VLLM_DRAFT_PATH:-/mnt/storage/models/staging/2026H2/hf/AngelSlim_Qwen3-4B_eagle3}" \
  XCELSIOR_LIVE_VLLM_QUANTIZATION="${XCELSIOR_LIVE_VLLM_QUANTIZATION:-awq}" \
  python scripts/live_vllm_e2e.py 2>&1 | tee -a "$SCRATCH/fleet-live-proof.log" || true

echo "=== fleet live: wait for vLLM ready ===" | tee -a "$SCRATCH/fleet-live-proof.log"
for i in $(seq 1 40); do
  if curl -sf -m 3 http://127.0.0.1:8199/v1/models >/dev/null 2>&1; then
    echo "vLLM ready after ${i} attempts" | tee -a "$SCRATCH/fleet-live-proof.log"
    break
  fi
  sleep 5
done

echo "=== fleet live: proxy served KPI (live vLLM upstream) ===" | tee -a "$SCRATCH/fleet-live-proof.log"
XCELSIOR_LIVE_VLLM_PORT=8199 python scripts/capture_live_proxy_served_kpi.py 2>&1 | tee -a "$SCRATCH/fleet-live-proof.log" || true
python scripts/capture_live_served_kpi.py 2>&1 | tee -a "$SCRATCH/fleet-live-proof.log" || true

echo "=== fleet live: speculative proxy → live vLLM ===" | tee -a "$SCRATCH/fleet-live-proof.log"
XCELSIOR_LIVE_VLLM_PORT=8199 python scripts/capture_live_speculative_proxy.py 2>&1 | tee -a "$SCRATCH/fleet-live-proof.log" || true

echo "=== fleet live: CRIU demo ===" | tee -a "$SCRATCH/fleet-live-proof.log"
python scripts/criu_live_demo_probe.py 2>&1 | tee -a "$SCRATCH/fleet-live-proof.log" || true

echo "=== fleet live proof complete ===" | tee -a "$SCRATCH/fleet-live-proof.log"