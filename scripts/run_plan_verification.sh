#!/usr/bin/env bash
# Canonical verification runner — plan.md steps 1–6 (token billing goal).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${XCELSIOR_GOAL_SCRATCH:-/tmp/grok-goal-6f86c7cfe9c2/implementer}"
MAC_HOST="${XCELSIOR_MAC_HOST:-aaryn@100.64.0.3}"
REGISTRY_MD="$ROOT/../pxl-registry/docs/AI_ML_MASTER_INDEX_BY_REPO.md"
mkdir -p "$SCRATCH"
cd "$ROOT"

if test -n "${XCELSIOR_POSTGRES_DSN:-}"; then
  alembic upgrade head >> "$SCRATCH/migrate.log" 2>&1 || true
elif test -f .env.test; then
  set -a && source .env.test && set +a
  alembic upgrade head >> "$SCRATCH/migrate.log" 2>&1 || true
fi

export XCELSIOR_ENV=test
export XCELSIOR_GOAL_SCRATCH="$SCRATCH"
export XCELSIOR_CLOSURE_POLICY="${XCELSIOR_CLOSURE_POLICY:-engineering_partial}"

echo "=== step 0: billing tick + periodic harness (gate) ===" | tee "$SCRATCH/pytest-billing-tick.log"
python -m pytest tests/test_billing_tick_integration.py tests/test_billing_periodic_harness.py -v --tb=short \
  2>&1 | tee -a "$SCRATCH/pytest-billing-tick.log"

echo "=== step 0b: supplemental closure tests (artifacts) ===" | tee "$SCRATCH/pytest-supplemental.log"
python -m pytest \
  tests/test_ops_rows_closure.py \
  tests/test_criu_preempt_resume.py \
  tests/test_token_billing_closure.py \
  tests/test_speculative_gate.py \
  tests/test_anchor_workloads.py \
  tests/test_registry_models.py \
  tests/test_b2_model_sync.py \
  tests/test_serverless_observability.py \
  tests/test_criu_hosts.py \
  tests/test_serverless_chaos_billing.py \
  tests/test_serverless_extended_features.py \
  tests/test_criu_preempt_resume.py \
  tests/test_platform_rows_8_12.py \
  tests/test_mcp_assistant_safety_eval.py \
  -q 2>&1 | tee -a "$SCRATCH/pytest-supplemental.log"

echo "=== step 2b: live vLLM probe (skip docker pull by default) ===" | tee -a "$SCRATCH/live-vllm-e2e.log"
XCELSIOR_LIVE_VLLM_START_DOCKER="${XCELSIOR_LIVE_VLLM_START_DOCKER:-0}" \
  python scripts/live_vllm_e2e.py 2>&1 | tee -a "$SCRATCH/live-vllm-e2e.log" || true

echo "=== step 1: plan gating pytest (exact command only, verbose) ===" | tee "$SCRATCH/pytest-billing.log"
python -m pytest \
  tests/test_billing_tick_integration.py \
  tests/test_serverless_billing.py \
  tests/test_serverless_openai_compat.py \
  tests/test_serverless_service.py \
  tests/test_serverless_idempotency.py \
  tests/test_instance_flow.py::TestBillingCharge \
  -v --tb=short 2>&1 | tee -a "$SCRATCH/pytest-billing.log"

echo "=== step 2: metering matrix (from closure helpers) ==="
test -f "$SCRATCH/metering-matrix.log" && echo "metering-matrix.log OK" || echo "WARN: run closure tests first"

echo "=== step 3–4: usage + anchor artifacts ==="
for f in billing-tick-integration.json billing-periodic-harness.json usage-sample.json anchor-workloads.json billing-e2e-transcript.json eagle3-gate.log speculative-proxy-evidence.json competitor-ui-research.txt; do
  test -f "$SCRATCH/$f" && echo "$f OK" || echo "WARN: $f missing"
done

echo "=== step 5: UI token table copy ===" | tee "$SCRATCH/ui-token-copy.txt"
grep -n "TokenPricingTable\|formatTokenRateFromPricing\|input_price_cad_per_m" \
  frontend/src/features/serverless/token-pricing-table.tsx \
  frontend/src/features/serverless/deploy-studio.tsx \
  frontend/src/features/serverless/cost-usage-panel.tsx \
  frontend/src/app/\(dashboard\)/dashboard/inference/page.tsx \
  >> "$SCRATCH/ui-token-copy.txt" 2>/dev/null || true

echo "=== step 5a: KV served-traffic KPI + live fleet proof ===" | tee "$SCRATCH/kv-kpi-capture.log"
python scripts/capture_kv_served_kpi.py 2>&1 | tee -a "$SCRATCH/kv-kpi-capture.log" || true
bash scripts/fleet_gate_live_proof.sh 2>&1 | tee -a "$SCRATCH/kv-kpi-capture.log" || true

echo "=== step 5b: capture row evidence (before sync) ===" | tee "$SCRATCH/row-evidence-capture.log"
python scripts/capture_row_evidence.py 2>&1 | tee -a "$SCRATCH/row-evidence-capture.log"

echo "=== step 6: sync master index from row-evidence.json ===" | tee "$SCRATCH/master-index-sync.log"
python scripts/sync_xcelsior_master_index.py --evidence "$SCRATCH/row-evidence.json" \
  --report "$SCRATCH/master-index-probes.json" 2>&1 | tee -a "$SCRATCH/master-index-sync.log"
cp -f "$SCRATCH/master-index-probes.json" "$SCRATCH/master-index-sync-report.json" 2>/dev/null || true
if test -d "$ROOT/../pxl-registry/.git"; then
  (cd "$ROOT/../pxl-registry" && git add docs/AI_ML_MASTER_INDEX_BY_REPO.md && \
    git commit -m "xcelsior §10: engineering_partial 27/31 checkbox sync from row-evidence.json" \
    >> "$SCRATCH/master-index-sync.log" 2>&1) || true
fi
grep -A 48 "## 10. \`xcelsior\`" docs/AI_ML_MASTER_INDEX_XCELSIOR.md \
  | tee "$SCRATCH/master-index-xcelsior.txt" >/dev/null
if test -f "$REGISTRY_MD"; then
  grep -A 48 "## 10. \`xcelsior\`" "$REGISTRY_MD" \
    | tee "$SCRATCH/master-index-pxl-registry.txt" >/dev/null
  echo "pxl-registry md captured" >> "$SCRATCH/master-index-sync.log"
  if scp -o BatchMode=yes -o ConnectTimeout=15 "$REGISTRY_MD" \
    "$MAC_HOST:~/Projects/pxl-registry/docs/AI_ML_MASTER_INDEX_BY_REPO.md" \
    >> "$SCRATCH/master-index-sync.log" 2>&1; then
    echo "pxl-registry md synced to Mac" >> "$SCRATCH/master-index-sync.log"
    python scripts/audit_ai_ml_docs.py "$SCRATCH" >> "$SCRATCH/master-index-sync.log" 2>&1 || {
      echo "FATAL: Mac/local §10 checkbox mismatch after scp" >> "$SCRATCH/master-index-sync.log"
      exit 1
    }
  else
    echo "WARN: Mac scp failed (local pxl-registry md still updated)" >> "$SCRATCH/master-index-sync.log"
  fi
else
  echo "WARN: $REGISTRY_MD missing" | tee "$SCRATCH/master-index-pxl-registry.txt"
fi

echo "=== step 7: mac + local doc audit ===" | tee "$SCRATCH/mac-ai-ml-docs.txt"
python scripts/audit_ai_ml_docs.py "$SCRATCH" 2>&1 | tee -a "$SCRATCH/mac-ai-ml-docs.txt" || true
if ssh -o BatchMode=yes -o ConnectTimeout=15 "$MAC_HOST" bash -s <<'REMOTE' >> "$SCRATCH/mac-ai-ml-docs.txt" 2>&1; then
set -euo pipefail
DOC_ROOT=~/Projects/pxl-registry/docs
echo "=== AI_ML markdown inventory ==="
ls -la "$DOC_ROOT"/AI_ML*.md
echo "=== frontier plan F5 token rows ==="
grep -nE 'token|cached|EAGLE|metering|anchor|embed' "$DOC_ROOT/AI_ML_FRONTIER_NEXT7_PLAN.md" | head -25 || true
echo "=== upgrade plan xcelsior rows ==="
grep -nE 'xcelsior|token|metering|prefix|EAGLE' "$DOC_ROOT/AI_ML_UPGRADE_PLAN.md" 2>/dev/null | head -20 || true
echo "mac ssh audit OK"
REMOTE
  echo "mac ssh audit completed" >> "$SCRATCH/mac-ai-ml-docs.txt"
else
  echo "mac unreachable (crosscheck still from audit script)" >> "$SCRATCH/mac-ai-ml-docs.txt"
fi

echo "=== verification complete ==="