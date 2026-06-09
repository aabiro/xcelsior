#!/usr/bin/env bash
# Switch local xcelsior-worker.service between prod and test schedulers.
#
# Usage:
#   bash scripts/switch_worker_env.sh test   # point RTX worker at localhost:9501
#   bash scripts/switch_worker_env.sh prod   # restore prod worker (xcelsior.ca)
#
# Called automatically by:
#   scripts/deploy.sh --test       (switches to test after stack is up)
#   scripts/deploy.sh --test-stop  (restores prod worker)
#   scripts/deploy.sh / --post-merge / --quick  (restores prod if was on test)
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# When invoked via sudo, keep paths under the invoking user's home (not /root).
_worker_home() {
  if [[ -n "${SUDO_USER:-}" ]]; then
    getent passwd "$SUDO_USER" | cut -d: -f6
  else
    echo "$HOME"
  fi
}
_WORKER_HOME="$(_worker_home)"
WORKER_ENV="${XCELSIOR_WORKER_ENV:-$_WORKER_HOME/.xcelsior/worker.env}"
WORKER_MODE_FILE="${XCELSIOR_WORKER_MODE_FILE:-$_WORKER_HOME/.xcelsior/worker.mode}"
PROD_BACKUP="${WORKER_ENV}.prod.bak"
TEST_OVERLAY="${PROJECT_DIR}/.env.worker.test"
SERVICE_NAME="${XCELSIOR_WORKER_SERVICE:-xcelsior-worker.service}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[worker]${NC} $1"; }
ok() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

usage() {
  echo "Usage: $0 test|prod" >&2
  exit 1
}

[[ "$MODE" == "test" || "$MODE" == "prod" ]] || usage

mkdir -p "$(dirname "$WORKER_ENV")"

_worker_sudo() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
  else
    warn "sudo required to manage $SERVICE_NAME — run: sudo systemctl restart $SERVICE_NAME"
    return 1
  fi
}

_service_active() {
  systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null
}

_stop_worker_service() {
  if _service_active; then
    log "Stopping $SERVICE_NAME..."
    _worker_sudo systemctl stop "$SERVICE_NAME" || true
  fi
}

_start_worker_service() {
  if systemctl list-unit-files "$SERVICE_NAME" 2>/dev/null | grep -q "$SERVICE_NAME"; then
    log "Starting $SERVICE_NAME..."
    _worker_sudo systemctl start "$SERVICE_NAME" && ok "Worker service started ($MODE)"
  else
    warn "No systemd unit $SERVICE_NAME — start manually: bash $SCRIPT_DIR/run_worker_test.sh"
  fi
}

_stop_local_job_containers() {
  local ids
  ids=$(docker ps -q --filter "name=xcl-" 2>/dev/null || true)
  if [[ -n "$ids" ]]; then
    log "Stopping local job containers (xcl-*)..."
    docker stop $ids >/dev/null 2>&1 || true
    ok "Job containers stopped"
  fi
}

_backup_prod_env() {
  if [[ ! -f "$PROD_BACKUP" ]]; then
    if [[ ! -f "$WORKER_ENV" ]]; then
      echo "Missing $WORKER_ENV — cannot switch worker" >&2
      exit 1
    fi
    cp "$WORKER_ENV" "$PROD_BACKUP"
    ok "Backed up prod worker env to $PROD_BACKUP"
  fi
}

_strip_worker_keys() {
  grep -v -E '^(XCELSIOR_SCHEDULER_URL|XCELSIOR_API_URL|XCELSIOR_API_TOKEN|XCELSIOR_PREFER_GVISOR|XCELSIOR_ALLOW_INSECURE_UPGRADE)=' "$1" || true
}

_apply_test_env() {
  _backup_prod_env
  local sched_url="http://localhost:9501"
  local api_url="http://localhost:9501"
  local api_token="test-token-not-for-production"
  local prefer_gvisor="false"
  local allow_insecure="1"
  if [[ -f "$TEST_OVERLAY" ]]; then
    # shellcheck disable=SC1090
    source "$TEST_OVERLAY"
    sched_url="${XCELSIOR_SCHEDULER_URL:-$sched_url}"
    api_url="${XCELSIOR_API_URL:-$api_url}"
    api_token="${XCELSIOR_API_TOKEN:-$api_token}"
    prefer_gvisor="${XCELSIOR_PREFER_GVISOR:-$prefer_gvisor}"
    allow_insecure="${XCELSIOR_ALLOW_INSECURE_UPGRADE:-$allow_insecure}"
  fi
  {
    _strip_worker_keys "$PROD_BACKUP"
    echo "XCELSIOR_SCHEDULER_URL=$sched_url"
    echo "XCELSIOR_API_URL=$api_url"
    echo "XCELSIOR_API_TOKEN=$api_token"
    echo "XCELSIOR_PREFER_GVISOR=$prefer_gvisor"
    echo "XCELSIOR_ALLOW_INSECURE_UPGRADE=$allow_insecure"
  } >"$WORKER_ENV"
  echo "test" >"$WORKER_MODE_FILE"
  ok "Worker env -> test ($sched_url)"
}

_apply_prod_env() {
  if [[ ! -f "$PROD_BACKUP" ]]; then
    warn "No $PROD_BACKUP — worker env unchanged"
    echo "prod" >"$WORKER_MODE_FILE"
    return 0
  fi
  cp "$PROD_BACKUP" "$WORKER_ENV"
  echo "prod" >"$WORKER_MODE_FILE"
  ok "Worker env -> prod (restored from backup)"
}

if [[ "$MODE" == "test" ]]; then
  _stop_local_job_containers
  _stop_worker_service
  _apply_test_env
  _start_worker_service
else
  _stop_local_job_containers
  _stop_worker_service
  _apply_prod_env
  _start_worker_service
fi