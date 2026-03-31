#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Xcelsior — Test Runner
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./run-tests.sh              Run full suite (excludes e2e live)
#   ./run-tests.sh quick        Run fast unit tests only (~15s)
#   ./run-tests.sh ai           Run AI assistant tests
#   ./run-tests.sh billing      Run billing + payment tests
#   ./run-tests.sh security     Run security + privacy + jurisdiction
#   ./run-tests.sh slurm        Run Slurm/HPC tests
#   ./run-tests.sh api          Run API + integration tests
#   ./run-tests.sh infra        Run scheduler, db, events, worker
#   ./run-tests.sh e2e          Run e2e tests (NOT live)
#   ./run-tests.sh live         Run live e2e (requires running server)
#   ./run-tests.sh <file>       Run a specific test file
#   ./run-tests.sh -k "pattern" Pass-through to pytest -k
#
# Options (prepend before target):
#   -v / --verbose              Show full output
#   -x / --failfast             Stop on first failure
#   -s                          Show print output (no capture)
#   --coverage                  Run with coverage report
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "venv" ]]; then
    source venv/bin/activate
fi

# ── Defaults ──────────────────────────────────────────────────────────────
PYTEST_ARGS=()
EXTRA_ARGS=()
TARGET=""

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)   EXTRA_ARGS+=("-v" "--tb=long"); shift ;;
        -x|--failfast)  EXTRA_ARGS+=("-x"); shift ;;
        -s)             EXTRA_ARGS+=("-s"); shift ;;
        --coverage)     EXTRA_ARGS+=("--cov=." "--cov-report=term-missing" "--cov-report=html"); shift ;;
        -k)             EXTRA_ARGS+=("-k" "$2"); shift 2 ;;
        -k=*)           EXTRA_ARGS+=("-k" "${1#-k=}"); shift ;;
        -*)             EXTRA_ARGS+=("$1"); shift ;;
        *)              TARGET="$1"; shift ;;
    esac
done

# ── Target mapping ───────────────────────────────────────────────────────
case "${TARGET}" in
    ""|full)
        PYTEST_ARGS=(tests/ --ignore=tests/test_e2e_live.py)
        echo "▸ Running full test suite (excluding live e2e)…"
        ;;
    quick)
        PYTEST_ARGS=(
            tests/test_api.py
            tests/test_db.py
            tests/test_scheduler.py
            tests/test_billing.py
            tests/test_security.py
        )
        echo "▸ Running quick unit tests…"
        ;;
    ai)
        PYTEST_ARGS=(tests/test_ai_assistant.py)
        echo "▸ Running AI assistant tests…"
        ;;
    billing)
        PYTEST_ARGS=(
            tests/test_billing.py
            tests/test_payment_flows.py
            tests/test_invoices_sla.py
            tests/test_bitcoin.py
        )
        echo "▸ Running billing & payment tests…"
        ;;
    security)
        PYTEST_ARGS=(
            tests/test_security.py
            tests/test_privacy.py
            tests/test_jurisdiction.py
            tests/test_verification.py
        )
        echo "▸ Running security & compliance tests…"
        ;;
    slurm|hpc)
        PYTEST_ARGS=(
            tests/test_slurm_adapter.py
            tests/test_slurm_ui.py
        )
        echo "▸ Running Slurm/HPC tests…"
        ;;
    api)
        PYTEST_ARGS=(
            tests/test_api.py
            tests/test_integration.py
            tests/test_v2_integration.py
            tests/test_persistent_auth.py
            tests/test_password_marketplace.py
        )
        echo "▸ Running API & integration tests…"
        ;;
    infra)
        PYTEST_ARGS=(
            tests/test_scheduler.py
            tests/test_db.py
            tests/test_db_migrations.py
            tests/test_events.py
            tests/test_worker_agent.py
            tests/test_nvml_telemetry.py
        )
        echo "▸ Running infrastructure tests…"
        ;;
    marketplace)
        PYTEST_ARGS=(
            tests/test_marketplace_v2.py
            tests/test_inference_store.py
            tests/test_inference_volumes.py
            tests/test_artifacts.py
            tests/test_reputation.py
        )
        echo "▸ Running marketplace tests…"
        ;;
    e2e)
        PYTEST_ARGS=(tests/test_e2e.py)
        echo "▸ Running e2e tests (offline)…"
        ;;
    live)
        PYTEST_ARGS=(tests/test_e2e_live.py)
        echo "▸ Running LIVE e2e tests (requires running server)…"
        ;;
    phases)
        PYTEST_ARGS=(
            tests/test_phase3_features.py
            tests/test_phase4_features.py
            tests/test_phase5_features.py
            tests/test_phase6_verification.py
        )
        echo "▸ Running phase feature tests…"
        ;;
    *)
        # If it's a file path, use it directly
        if [[ -f "$TARGET" ]]; then
            PYTEST_ARGS=("$TARGET")
            echo "▸ Running $TARGET…"
        elif [[ -f "tests/$TARGET" ]]; then
            PYTEST_ARGS=("tests/$TARGET")
            echo "▸ Running tests/$TARGET…"
        elif [[ -f "tests/test_${TARGET}.py" ]]; then
            PYTEST_ARGS=("tests/test_${TARGET}.py")
            echo "▸ Running tests/test_${TARGET}.py…"
        else
            echo "✗ Unknown target: $TARGET"
            echo "  Run ./run-tests.sh --help or see script header for usage."
            exit 1
        fi
        ;;
esac

# ── Run ──────────────────────────────────────────────────────────────────
python -m pytest "${PYTEST_ARGS[@]}" "${EXTRA_ARGS[@]}" -q --tb=short -p no:sugar -p no:rich
