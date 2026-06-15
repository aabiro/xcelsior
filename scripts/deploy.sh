#!/usr/bin/env bash
# Xcelsior Deployment Script
# Automates deployment to production VPS (xcelsior.ca)
#
# Usage:
#   ./scripts/deploy.sh              # Full deploy (build + restart)
#   ./scripts/deploy.sh --quick      # Quick deploy (pull + restart, no rebuild)
#   ./scripts/deploy.sh --setup      # First-time setup (install deps, SSL, nginx)
#   ./scripts/deploy.sh --rollback   # Rollback to previous version

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="/opt/xcelsior"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"
DOMAIN="xcelsior.ca"
BACKUP_DIR="/opt/xcelsior-backups"

# ── Environment Selection ─────────────────────────────────────────────
# .env       = production config (deployed to VPS)
# .env.test  = test config (used locally for testing)
TARGET_ENV="${XCELSIOR_TARGET_ENV:-prod}"

resolve_env() {
    case "$TARGET_ENV" in
        prod|production)
            TARGET_ENV="prod"
            ENV_FILE="$PROJECT_DIR/.env"
            ;;
        test|testing)
            TARGET_ENV="test"
            ENV_FILE="$PROJECT_DIR/.env.test"
            ;;
        *)
            error "Unknown environment: $TARGET_ENV. Use 'prod' or 'test'."
            ;;
    esac

    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file not found: $ENV_FILE"
    fi
    log "Environment: ${BOLD}$TARGET_ENV${NC} ($ENV_FILE)"
}

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# ── Helper Functions ──────────────────────────────────────────────────
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
WORKER_MODE_FILE="${XCELSIOR_WORKER_MODE_FILE:-$HOME/.xcelsior/worker.mode}"
SSH_CONTROL_PATH="${XCELSIOR_SSH_CONTROL_PATH:-$HOME/.ssh/cm-xcelsior-%r@%h:%p}"
DEPLOY_SYNC_MODE="${XCELSIOR_DEPLOY_SYNC:-rsync}"  # rsync (fast) | tarball (legacy)
DEPLOY_TIMING="${XCELSIOR_DEPLOY_TIMING:-0}"
DEPLOY_COMPRESS_MODE="${XCELSIOR_DEPLOY_COMPRESS:-zstd}"  # zstd | gzip | none
DEPLOY_BUILD_CACHE="${XCELSIOR_DEPLOY_BUILD_CACHE:-1}"
DOCKER_BUILD_CACHE_DIR="${XCELSIOR_DOCKER_BUILD_CACHE_DIR:-/opt/xcelsior-backups/docker-build-cache}"
DEPLOY_ALLOW_MULTI_PRIMARY="${XCELSIOR_DEPLOY_ALLOW_MULTI_PRIMARY:-0}"
declare -a DEPLOY_SYNC_HOSTS=()
declare -a MIRROR_SYNC_PIDS=()
declare -gA REMOTE_DEPLOY_HASHES=()
RSYNC_COMPRESS_OPTS=()
_DEPLOY_STEP_TS=0

_deploy_pkg_binary() {
    case "$1" in
        zstd) printf '%s' "zstd" ;;
        pigz) printf '%s' "pigz" ;;
        rsync) printf '%s' "rsync" ;;
        *) error "Unknown deploy package: $1" ;;
    esac
}

_deploy_pkg_apt_name() {
    case "$1" in
        zstd|pigz|rsync) printf '%s' "$1" ;;
        *) error "Unknown deploy package: $1" ;;
    esac
}

_deploy_packages_for_compress_mode() {
    local -n _out=$1
    _out=()
    case "$DEPLOY_COMPRESS_MODE" in
        none) _out=(rsync) ;;
        gzip) _out=(pigz rsync) ;;
        zstd) _out=(zstd pigz rsync) ;;
        *) error "Unknown XCELSIOR_DEPLOY_COMPRESS=$DEPLOY_COMPRESS_MODE (use zstd, gzip, or none)" ;;
    esac
}

_apt_install_missing() {
    local pkgs=("$@")
    local missing=() pkg bin
    for pkg in "${pkgs[@]}"; do
        bin=$(_deploy_pkg_binary "$pkg")
        command -v "$bin" >/dev/null 2>&1 || missing+=("$(_deploy_pkg_apt_name "$pkg")")
    done
    [[ ${#missing[@]} -eq 0 ]] && return 0
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing deploy packages: ${missing[*]}"
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
        return 0
    fi
    if command -v brew >/dev/null 2>&1; then
        log "Installing deploy packages via brew: ${missing[*]}"
        brew install "${missing[@]}"
        return 0
    fi
    error "Missing deploy packages: ${missing[*]}. Install with apt or brew."
}

_remote_apt_install_missing() {
    local host="$1"
    shift
    local pkgs=("$@")
    ssh_cmd_host "$host" "PACKAGES='${pkgs[*]}' bash -s" <<'EOF'
set -euo pipefail
missing=()
for pkg in $PACKAGES; do
    case "$pkg" in
        zstd) command -v zstd >/dev/null 2>&1 || missing+=("zstd") ;;
        pigz) command -v pigz >/dev/null 2>&1 || missing+=("pigz") ;;
        rsync) command -v rsync >/dev/null 2>&1 || missing+=("rsync") ;;
    esac
done
if [[ ${#missing[@]} -eq 0 ]]; then
    exit 0
fi
if ! command -v apt-get >/dev/null 2>&1; then
    echo "Missing packages on remote and apt-get unavailable: ${missing[*]}" >&2
    exit 1
fi
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
EOF
}

ensure_local_deploy_tools() {
    local -a pkgs=()
    _deploy_packages_for_compress_mode pkgs
    _apt_install_missing "${pkgs[@]}"
}

ensure_remote_deploy_tools() {
    local host="${1:-$REMOTE_HOST}"
    local -a pkgs=()
    _deploy_packages_for_compress_mode pkgs
    _remote_apt_install_missing "$host" "${pkgs[@]}"
}

ensure_all_remote_deploy_tools() {
    ensure_remote_deploy_tools "$REMOTE_HOST"
    local host
    for host in "${DEPLOY_SYNC_HOSTS[@]}"; do
        [[ "$host" == "$REMOTE_HOST" ]] && continue
        ensure_remote_deploy_tools "$host"
    done
}

init_deploy_compress() {
    case "$DEPLOY_COMPRESS_MODE" in
        auto) DEPLOY_COMPRESS_MODE=zstd ;;
        zstd|gzip|none) ;;
        *) error "Unknown XCELSIOR_DEPLOY_COMPRESS=$DEPLOY_COMPRESS_MODE (use zstd, gzip, or none)" ;;
    esac
}

_parse_csv_hosts() {
    local raw="$1"
    local -n _out=$2
    _out=()
    [[ -z "$raw" ]] && return 0
    local part
    local IFS=','
    for part in $raw; do
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        [[ -n "$part" ]] && _out+=("$part")
    done
}

init_deploy_targets() {
    _parse_csv_hosts "${XCELSIOR_DEPLOY_SYNC_HOSTS:-}" DEPLOY_SYNC_HOSTS
    if [[ "$DEPLOY_ALLOW_MULTI_PRIMARY" == "1" && -n "${XCELSIOR_DEPLOY_TARGETS:-}" ]]; then
        local -a extra_targets=()
        _parse_csv_hosts "$XCELSIOR_DEPLOY_TARGETS" extra_targets
        local host
        for host in "${extra_targets[@]}"; do
            if [[ "$host" != "$REMOTE_HOST" ]]; then
                DEPLOY_SYNC_HOSTS+=("$host")
            fi
        done
    fi
}

init_rsync_compress() {
    RSYNC_COMPRESS_OPTS=()
    case "$DEPLOY_COMPRESS_MODE" in
        none) ;;
        gzip) RSYNC_COMPRESS_OPTS=(--compress) ;;
        zstd) RSYNC_COMPRESS_OPTS=(--compress --compress-choice=zstd) ;;
    esac
    if [[ ${#RSYNC_COMPRESS_OPTS[@]} -gt 0 ]]; then
        log "Rsync wire compression: ${DEPLOY_COMPRESS_MODE} (${RSYNC_COMPRESS_OPTS[*]})"
    fi
}

_select_tar_compress() {
    local -n _compressor=$1
    local -n _ext=$2
    case "$DEPLOY_COMPRESS_MODE" in
        none)
            _compressor=(cat)
            _ext=tar
            ;;
        gzip)
            _compressor=(pigz -1)
            _ext=gz
            ;;
        zstd)
            _compressor=(zstd -3 -T0)
            _ext=zst
            ;;
    esac
}

remote_docker_build_prefix() {
    if [[ "$DEPLOY_BUILD_CACHE" == "1" ]]; then
        printf 'export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 BUILDKIT_PROGRESS=plain; mkdir -p %q; ' "$DOCKER_BUILD_CACHE_DIR"
    else
        printf 'export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1; '
    fi
}

_deploy_mark() {
    [[ "$DEPLOY_TIMING" == "1" ]] || return 0
    local now
    now=$(date +%s)
    if [[ "$_DEPLOY_STEP_TS" -gt 0 ]]; then
        log "⏱ step ${1:-done} took $((now - _DEPLOY_STEP_TS))s"
    fi
    _DEPLOY_STEP_TS=$now
}

SSH_BASE_OPTS=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=accept-new
    -o ControlMaster=auto
    -o ControlPersist=15m
    -o ControlPath="$SSH_CONTROL_PATH"
    -o Compression=no
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=4
)

RSYNC_BASE_OPTS=(
    -az
    --partial
    --human-readable
    --omit-dir-times
    --no-perms
    --no-owner
    --no-group
)

RSYNC_EXCLUDES=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='.pytest_cache'
    --exclude='venv'
    --exclude='node_modules'
    --exclude='.next'
    --exclude='*.db'
    --exclude='*.db-*'
    --exclude='*.log'
    --exclude='data'
    --exclude='/artifacts'
    --exclude='.env'
    --exclude='/desktop'
    --exclude='checkpoints'
    --exclude='.cursor'
    --exclude='terminals'
)

_remote_target() {
    local host="${1:-$REMOTE_HOST}"
    printf '%s@%s' "$REMOTE_USER" "$host"
}

_rsync_shell() {
    # shellcheck disable=SC2068
    printf 'ssh'
    for opt in "${SSH_BASE_OPTS[@]}"; do
        printf ' %q' "$opt"
    done
}

_rsync_shell_for_host() {
    printf 'ssh'
    for opt in "${SSH_BASE_OPTS[@]}"; do
        printf ' %q' "$opt"
    done
}

cleanup_ssh_mux() {
    ssh -O exit -o ControlPath="$SSH_CONTROL_PATH" -i "$SSH_KEY" "$(_remote_target)" >/dev/null 2>&1 || true
}

trap cleanup_ssh_mux EXIT

switch_local_worker() {
    local mode="$1"
    if [[ ! -x "$SCRIPT_DIR/switch_worker_env.sh" ]]; then
        warn "switch_worker_env.sh missing — worker not switched to $mode"
        return 0
    fi
    log "Switching local GPU worker -> ${BOLD}$mode${NC}..."
    if bash "$SCRIPT_DIR/switch_worker_env.sh" "$mode"; then
        success "Local worker configured for $mode"
    else
        warn "Worker switch to $mode failed (sudo may be required)"
    fi
}

ensure_prod_worker_if_test() {
    if [[ -f "$WORKER_MODE_FILE" ]] && [[ "$(tr -d '[:space:]' <"$WORKER_MODE_FILE")" == "test" ]]; then
        switch_local_worker prod
    fi
}

open_ssh_mux() {
    open_ssh_mux_host "$REMOTE_HOST"
}

open_ssh_mux_host() {
    local host="${1:-$REMOTE_HOST}"
    ssh "${SSH_BASE_OPTS[@]}" "$(_remote_target "$host")" "true" >/dev/null
}

open_ssh_mux_all() {
    open_ssh_mux_host "$REMOTE_HOST"
    local host
    for host in "${DEPLOY_SYNC_HOSTS[@]}"; do
        [[ "$host" == "$REMOTE_HOST" ]] && continue
        open_ssh_mux_host "$host" &
    done
    wait
}

ssh_cmd() {
    ssh_cmd_host "$REMOTE_HOST" "$@"
}

ssh_cmd_host() {
    local host="$1"
    shift
    ssh "${SSH_BASE_OPTS[@]}" "$(_remote_target "$host")" "$@"
}

scp_file() {
    scp_file_host "$REMOTE_HOST" "$1" "$2"
}

scp_file_host() {
    local host="$1" src="$2" dest="$3"
    scp "${SSH_BASE_OPTS[@]}" "$src" "$(_remote_target "$host"):$dest"
}

rsync_to_host() {
    local host="$1" src="$2" dest="$3"
    shift 3
    rsync "${RSYNC_BASE_OPTS[@]}" "${RSYNC_COMPRESS_OPTS[@]}" "$@" \
        -e "$(_rsync_shell_for_host)" "$src" "$(_remote_target "$host"):$dest"
}

rsync_to_remote() {
    rsync_to_host "$REMOTE_HOST" "$@"
}

get_env_value() {
    local key="$1"
    python3 - "$ENV_FILE" "$key" <<'PY'
import sys

path, key = sys.argv[1], sys.argv[2]
value = ""
with open(path, "r", encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() != key:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        value = v
        break
print(value, end="")
PY
}

resolve_api_base_url() {
    local base
    base=$(get_env_value "XCELSIOR_BASE_URL")
    if [[ -z "$base" ]]; then
        base=$(get_env_value "NEXT_PUBLIC_API_URL")
    fi
    if [[ -z "$base" ]]; then
        base="https://$DOMAIN"
    fi
    printf '%s' "${base%/}"
}

resolve_api_token() {
    local token
    token=$(get_env_value "XCELSIOR_API_TOKEN")
    [[ -n "$token" ]] || error "XCELSIOR_API_TOKEN missing from $ENV_FILE (required for deployment maintenance API calls)"
    printf '%s' "$token"
}

api_request() {
    local method="$1" path="$2" body="${3:-}"
    local base token
    base=$(resolve_api_base_url)
    token=$(resolve_api_token)

    if [[ -n "$body" ]]; then
        curl -fsS -X "$method" \
            -H "Authorization: Bearer $token" \
            -H "Content-Type: application/json" \
            -d "$body" \
            "$base$path"
    else
        curl -fsS -X "$method" \
            -H "Authorization: Bearer $token" \
            "$base$path"
    fi
}

host_maintenance_summary() {
    local host_id="$1"
    api_request GET "/host/$host_id/maintenance"
}

drain_worker_host() {
    local host_id="$1"
    log "Draining worker host $host_id..."
    api_request POST "/host/$host_id/drain" >/dev/null
    success "Host $host_id marked draining"
}

undrain_worker_host() {
    local host_id="$1"
    log "Undraining worker host $host_id..."
    api_request POST "/host/$host_id/undrain" >/dev/null
    success "Host $host_id restored to schedulable state"
}

guard_worker_host() {
    local host_id="$1"
    local summary
    summary=$(host_maintenance_summary "$host_id") || error "Failed to fetch maintenance status for $host_id"

    python3 - "$host_id" "$summary" <<'PY'
import json
import sys

host_id = sys.argv[1]
summary = json.loads(sys.argv[2])
status = summary.get("status", "unknown")
interactive_count = int(summary.get("active_interactive_instances", 0) or 0)
serverless_count = int(summary.get("active_serverless_workers", 0) or 0)

print(f"Host: {host_id}")
print(f"Status: {status}")
print(f"Active interactive instances: {interactive_count}")
for item in summary.get("interactive_instances", []):
    print(f"- {item.get('job_id')} {item.get('status')} {item.get('name')}")
print(f"Active serverless workers: {serverless_count}")
for item in summary.get("serverless_workers", []):
    print(f"- {item.get('job_id')} {item.get('status')} {item.get('name')}")

if status != "draining":
    print("Unsafe: host is not drained", file=sys.stderr)
    raise SystemExit(2)
if interactive_count > 0:
    print("Unsafe: interactive instances are still active", file=sys.stderr)
    raise SystemExit(3)
if serverless_count > 0:
    print("Unsafe: serverless workers are still active", file=sys.stderr)
    raise SystemExit(4)
PY
    success "Host $host_id is safe for maintenance"
}

repair_nginx_systemd() {
    # Ensure nginx is managed by systemd (not an orphan master from a failed start).
    log "Repairing nginx systemd unit..."
    ssh_cmd << 'EOF'
set -e
if systemctl is-active --quiet nginx; then
  echo "nginx already active under systemd"
  exit 0
fi
if [ -f /var/run/nginx.pid ]; then
  sudo kill -QUIT "$(cat /var/run/nginx.pid)" 2>/dev/null || true
fi
sleep 2
sudo pkill -QUIT nginx 2>/dev/null || true
sleep 1
sudo systemctl reset-failed nginx 2>/dev/null || true
sudo systemctl start nginx
systemctl is-active --quiet nginx
EOF
    success "nginx running under systemd"
}

install_nginx_configs() {
    log "Installing nginx site configs (rsync bundle)..."
    _deploy_mark "nginx-start"

    ssh_cmd "rm -rf /tmp/xcelsior-nginx && mkdir -p /tmp/xcelsior-nginx"
    rsync_to_remote "$PROJECT_DIR/nginx/" "/tmp/xcelsior-nginx/" \
        --include='*.conf' --exclude='*'

    ssh_cmd << 'EOF'
set -e
for f in xcelsior headscale headscale-http docs-xcelsior downloads-xcelsior; do
  sudo cp "/tmp/xcelsior-nginx/${f}.conf" "/etc/nginx/sites-available/${f}"
  sudo ln -sf "/etc/nginx/sites-available/${f}" "/etc/nginx/sites-enabled/${f}"
done
sudo nginx -t
if systemctl is-active --quiet nginx; then
  sudo systemctl reload nginx
else
  sudo systemctl reset-failed nginx 2>/dev/null || true
  sudo systemctl start nginx
fi
EOF
    _deploy_mark "nginx-done"
    success "Nginx configs installed"
}

check_ssh() {
    log "Testing SSH connection to $REMOTE_HOST (mux=${SSH_CONTROL_PATH})..."
    if open_ssh_mux_all && ssh_cmd "echo 'SSH OK'" &>/dev/null; then
        success "SSH connection successful (control master ready)"
    else
        error "Cannot connect to $REMOTE_HOST. Check SSH keys and connectivity."
    fi
    ensure_all_remote_deploy_tools
    if [[ ${#DEPLOY_SYNC_HOSTS[@]} -gt 0 ]]; then
        local host
        for host in "${DEPLOY_SYNC_HOSTS[@]}"; do
            [[ "$host" == "$REMOTE_HOST" ]] && continue
            if ssh_cmd_host "$host" "echo 'SSH OK'" &>/dev/null; then
                success "Mirror host reachable: $host"
            else
                warn "Mirror host unreachable: $host (will retry during sync)"
            fi
        done
    fi
}

# ── First-Time Setup ──────────────────────────────────────────────────
setup_server() {
    log "Running first-time server setup..."
    
    ssh_cmd << 'EOF'
set -e

echo "=== Installing dependencies ==="
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx docker.io docker-compose-plugin postgresql postgresql-client zstd pigz rsync curl

echo "=== Creating directories ==="
sudo mkdir -p /opt/xcelsior /opt/xcelsior-backups /var/www/certbot
sudo chown -R $USER:$USER /opt/xcelsior /opt/xcelsior-backups

echo "=== Adding user to docker group ==="
sudo usermod -aG docker $USER

echo "=== Enabling services ==="
sudo systemctl enable nginx docker postgresql
sudo systemctl start docker postgresql

echo "=== Setup complete ==="
EOF
    success "Server dependencies installed"
}

setup_ssl() {
    log "Setting up SSL certificates..."
    
    ssh_cmd << EOF
set -e

# Install nginx config (without SSL first for certbot)
sudo tee /etc/nginx/sites-available/xcelsior > /dev/null << 'NGINX'
server {
    listen 80;
    server_name xcelsior.ca www.xcelsior.ca hs.xcelsior.ca;
    
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    
    location / {
        return 200 'Xcelsior is being configured...';
        add_header Content-Type text/plain;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/xcelsior /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Get SSL certificate
sudo certbot certonly --webroot -w /var/www/certbot \
    -d $DOMAIN -d www.$DOMAIN \
    --non-interactive --agree-tos \
    --email admin@$DOMAIN

sudo certbot certonly --webroot -w /var/www/certbot \
    -d hs.$DOMAIN \
    --non-interactive --agree-tos \
    --email admin@$DOMAIN

# Now install full nginx config
echo "Certificates installed"

# Setup auto-renewal
echo "0 3 * * * root certbot renew --quiet --post-hook 'systemctl reload nginx'" | sudo tee /etc/cron.d/certbot-renew
EOF
    install_nginx_configs
    success "SSL certificates configured"
}

setup_systemd() {
    log "Installing systemd services..."
    
    scp_file "$PROJECT_DIR/xcelsior-api.service" "/tmp/xcelsior-api.service"
    scp_file "$PROJECT_DIR/xcelsior-health.service" "/tmp/xcelsior-health.service"
    scp_file "$PROJECT_DIR/xcelsior-worker.service" "/tmp/xcelsior-worker.service"
    
    ssh_cmd << 'EOF'
sudo cp /tmp/xcelsior-*.service /etc/systemd/system/
sudo systemctl daemon-reload
EOF
    success "Systemd services installed"
}

# ── Deployment Functions ──────────────────────────────────────────────
backup_current() {
    log "Backing up current deployment..."
    
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    ssh_cmd << EOF
set -e
if [ -d "$DEPLOY_DIR" ] && [ -f "$DEPLOY_DIR/api.py" ]; then
    sudo mkdir -p $BACKUP_DIR
    sudo tar -czf "$BACKUP_DIR/xcelsior_$TIMESTAMP.tar.gz" -C /opt xcelsior
    # Keep only last 5 backups
    ls -t $BACKUP_DIR/xcelsior_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r sudo rm
    echo "Backup created: xcelsior_$TIMESTAMP.tar.gz"
else
    echo "No existing deployment to backup"
fi
EOF
    success "Backup complete"
}

sync_code_push_env() {
    sync_code_push_env_host "$REMOTE_HOST"
}

sync_code_push_env_host() {
    local host="${1:-$REMOTE_HOST}"
    log "Sending $TARGET_ENV environment config -> $host..."
    rsync_to_host "$host" "$ENV_FILE" "/tmp/xcelsior_env"
    ssh_cmd_host "$host" "cp /tmp/xcelsior_env /opt/xcelsior/.env && rm /tmp/xcelsior_env"
}

sync_code_preserve_remote_files_host() {
    local host="$1"
    ssh_cmd_host "$host" << 'EOF'
set -e
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/opt/xcelsior/$f" ] && cp "/opt/xcelsior/$f" "/tmp/xcelsior_preserve_$f" || true
done
EOF
}

sync_code_restore_preserved_files_host() {
    local host="$1"
    ssh_cmd_host "$host" << 'EOF'
set -e
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/tmp/xcelsior_preserve_$f" ] && cp "/tmp/xcelsior_preserve_$f" "/opt/xcelsior_new/$f" || true
    rm -f "/tmp/xcelsior_preserve_$f"
done
EOF
}

sync_code_rsync_host() {
    local host="$1"
    local push_env="${2:-1}"
    log "Syncing code -> $host (rsync delta)..."
    _deploy_mark "sync-$host-start"

    ssh_cmd_host "$host" << 'EOF'
set -e
sudo mkdir -p /opt/xcelsior /opt/xcelsior_new
sudo chown -R $USER:$USER /opt/xcelsior /opt/xcelsior_new 2>/dev/null || true
rm -rf /opt/xcelsior_new/*
EOF

    sync_code_preserve_remote_files_host "$host"

    rsync_to_host "$host" "$PROJECT_DIR/" "/opt/xcelsior_new/" \
        "${RSYNC_EXCLUDES[@]}" \
        --delete

    sync_code_restore_preserved_files_host "$host"

    ssh_cmd_host "$host" << 'EOF'
set -e
sudo rm -rf /opt/xcelsior
sudo mv /opt/xcelsior_new /opt/xcelsior
sudo chown -R $USER:$USER /opt/xcelsior
EOF
    if [[ "$push_env" == "1" ]]; then
        sync_code_push_env_host "$host"
    fi
    _deploy_mark "sync-$host-done"
    success "Code synced via rsync -> $host (env_push=$push_env)"
}

sync_code_rsync() {
    log "Syncing code to primary $REMOTE_HOST (rsync delta, mux SSH)..."
    _deploy_mark "sync-start"
    sync_code_rsync_host "$REMOTE_HOST" 1
    _deploy_mark "sync-done"
}

sync_code_mirror_host() {
    local host="$1"
    open_ssh_mux_host "$host" 2>/dev/null || true
    sync_code_rsync_host "$host" 0 || warn "Mirror sync failed: $host"
}

start_mirror_syncs() {
    MIRROR_SYNC_PIDS=()
    [[ ${#DEPLOY_SYNC_HOSTS[@]} -eq 0 ]] && return 0
    local host
    for host in "${DEPLOY_SYNC_HOSTS[@]}"; do
        [[ "$host" == "$REMOTE_HOST" ]] && continue
        log "Starting background mirror sync -> $host"
        sync_code_mirror_host "$host" &
        MIRROR_SYNC_PIDS+=($!)
    done
}

wait_mirror_syncs() {
    [[ ${#MIRROR_SYNC_PIDS[@]} -eq 0 ]] && return 0
    local pid rc=0
    for pid in "${MIRROR_SYNC_PIDS[@]}"; do
        wait "$pid" || rc=1
    done
    if [[ $rc -eq 0 ]]; then
        success "All mirror syncs complete (${#MIRROR_SYNC_PIDS[@]} host(s))"
    else
        warn "One or more mirror syncs failed (primary deploy unaffected)"
    fi
    return 0
}

sync_code_tarball() {
    log "Syncing code to server (legacy tarball)..."
    _deploy_mark "sync-start"
    local remote_tarball tar_compress tar_ext
    _select_tar_compress tar_compress tar_ext

    ssh_cmd '
set -e
for d in /tmp /var/tmp /opt/xcelsior-backups /opt/xcelsior-backups/staging; do
    if [ -d "$d" ]; then
        find "$d" -maxdepth 1 -type f \( -name "xcelsior_deploy*.tar.gz" -o -name "xcelsior_deploy*.tar.zst" -o -name "xcelsior_deploy*.tar" \) -mtime +2 -delete 2>/dev/null || true
    fi
done
' || true

    local tarball="/tmp/xcelsior_deploy.tar.${tar_ext}"
    log "Tarball compression: ${tar_compress[*]} (.${tar_ext})"

    tar -cf - \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='venv' \
        --exclude='node_modules' \
        --exclude='.next' \
        --exclude='*.db' \
        --exclude='*.db-*' \
        --exclude='*.log' \
        --exclude='data' \
        --exclude='/artifacts' \
        --exclude='.env' \
        --exclude='./desktop' \
        --exclude='checkpoints' \
        -C "$PROJECT_DIR" . | "${tar_compress[@]}" >"$tarball"

    remote_tarball=""
    local candidate probe_target
    for candidate in /tmp /var/tmp /opt/xcelsior-backups/staging /opt/xcelsior-backups; do
        ssh_cmd "mkdir -p '$candidate' 2>/dev/null || sudo mkdir -p '$candidate' || true; sudo chown \$USER:\$USER '$candidate' 2>/dev/null || true" || true
        probe_target="${candidate%/}/xcelsior_deploy_$(date +%s).tar.${tar_ext}"
        if scp_file "$tarball" "$probe_target" 2>/dev/null; then
            remote_tarball="$probe_target"
            break
        fi
    done
    [[ -n "$remote_tarball" ]] || error "Failed to upload deploy artifact to all staging paths (/tmp, /var/tmp, /opt/xcelsior-backups)"
    log "Using remote staging path: $remote_tarball"
    rm -f "$tarball"

    ssh_cmd "DEPLOY_ARCHIVE='$remote_tarball' DEPLOY_ARCHIVE_EXT='$tar_ext' bash -s" << 'EOF'
set -e
sudo mkdir -p /opt/xcelsior /opt/xcelsior_new
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/opt/xcelsior/$f" ] && sudo cp "/opt/xcelsior/$f" "/tmp/xcelsior_preserve_$f" || true
done
sudo rm -rf /opt/xcelsior_new/*
case "$DEPLOY_ARCHIVE_EXT" in
    zst) zstd -d -c "$DEPLOY_ARCHIVE" | sudo tar -xf - -C /opt/xcelsior_new ;;
    gz) sudo tar -xzf "$DEPLOY_ARCHIVE" -C /opt/xcelsior_new ;;
    tar) sudo tar -xf "$DEPLOY_ARCHIVE" -C /opt/xcelsior_new ;;
    *) sudo tar -xf "$DEPLOY_ARCHIVE" -C /opt/xcelsior_new ;;
esac
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/tmp/xcelsior_preserve_$f" ] && sudo cp "/tmp/xcelsior_preserve_$f" "/opt/xcelsior_new/$f" || true
    sudo rm -f "/tmp/xcelsior_preserve_$f"
done
sudo rm -rf /opt/xcelsior
sudo mv /opt/xcelsior_new /opt/xcelsior
sudo chown -R $USER:$USER /opt/xcelsior
rm -f "$DEPLOY_ARCHIVE"
EOF

    sync_code_push_env
    start_mirror_syncs
    _deploy_mark "sync-done"
    success "Code synced via tarball (env=$TARGET_ENV)"
}

sync_code() {
    case "$DEPLOY_SYNC_MODE" in
        rsync)
            set +e
            sync_code_rsync
            local rsync_rc=$?
            set -e
            if [[ $rsync_rc -ne 0 ]]; then
                warn "rsync sync failed (exit $rsync_rc) — falling back to tarball"
                sync_code_tarball
            else
                start_mirror_syncs
            fi
            ;;
        tarball|tar)
            sync_code_tarball
            ;;
        *)
            error "Unknown XCELSIOR_DEPLOY_SYNC=$DEPLOY_SYNC_MODE (use rsync or tarball)"
            ;;
    esac
}

validate_build_env() {
    # Frontend build bakes NEXT_PUBLIC_* vars at compile time — if they're
    # blank the build succeeds silently but features (analytics, Google
    # verification, Stripe, WalletConnect) will be missing at runtime.
    # Auto-discovers ALL NEXT_PUBLIC_* vars from .env so new ones are
    # never silently skipped.
    log "Validating frontend build-time env vars..."

    local missing=()
    local found=0

    # Discover every NEXT_PUBLIC_* var defined in .env
    while IFS='=' read -r var val; do
        found=$((found + 1))
        if [[ -z "$val" ]]; then
            if [[ "$var" == "NEXT_PUBLIC_API_URL" || "$var" == "NEXT_PUBLIC_APP_URL" ]]; then
                missing+=("$var (REQUIRED)")
            else
                warn "$var is empty — feature will be disabled in this build"
            fi
        else
            success "  $var = ${val:0:20}..."
        fi
    done < <(grep '^NEXT_PUBLIC_' "$ENV_FILE" 2>/dev/null)

    if [[ $found -eq 0 ]]; then
        error "No NEXT_PUBLIC_* vars found in $ENV_FILE"
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        for m in "${missing[@]}"; do
            error "Missing env var: $m"
        done
    fi
    log "Found $found NEXT_PUBLIC_* vars in .env"
    success "Build-time env vars validated"
}

verify_frontend_desktop_runtime_on_remote() {
    local missing
    missing=$(ssh_cmd "for f in \
frontend/src/lib/desktop/runtime.tsx \
frontend/src/lib/desktop/contract.ts \
frontend/src/lib/desktop/tauri.ts; do
  test -f \"/opt/xcelsior/\$f\" || echo \"\$f\"
done" || true)
    if [[ -n "$missing" ]]; then
        error "Frontend desktop runtime missing on server (rsync exclude too broad?):
$missing
Ensure scripts/deploy.sh uses --exclude='/desktop' (root-only), then rerun deploy."
    fi
    success "Frontend desktop runtime present on server"
}

# Collect --build-arg flags for every NEXT_PUBLIC_* var in .env
# so they are explicitly passed to `docker compose build` and
# guaranteed to be baked into the Next.js static output.
collect_frontend_build_args() {
    local args=()
    while IFS='=' read -r var val; do
        args+=("--build-arg" "${var}=${val}")
    done < <(grep '^NEXT_PUBLIC_' "$ENV_FILE" 2>/dev/null)
    echo "${args[@]}"
}

hash_repo_subset() {
    python3 - "$PROJECT_DIR" "$@" <<'PY'
import glob
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
patterns = sys.argv[2:] or ["."]
files: set[Path] = set()

for pattern in patterns:
    matches = glob.glob(str(root / pattern), recursive=True)
    candidate = root / pattern
    if not matches and candidate.exists():
        matches = [str(candidate)]
    for raw in matches:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_dir():
            for item in path.rglob("*"):
                if item.is_file():
                    files.add(item.resolve())
        elif path.is_file():
            files.add(path.resolve())

digest = hashlib.sha256()
for path in sorted(files):
    rel = path.relative_to(root).as_posix()
    digest.update(rel.encode())
    digest.update(b"\0")
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    digest.update(b"\0")

print(digest.hexdigest(), end="")
PY
}

frontend_build_hash() {
    local source_hash env_hash
    source_hash=$(hash_repo_subset frontend)
    env_hash=$(grep '^NEXT_PUBLIC_' "$ENV_FILE" 2>/dev/null | sort | sha256sum | cut -d' ' -f1)
    printf '%s%s' "$source_hash" "$env_hash" | sha256sum | cut -d' ' -f1
}

remote_deploy_meta_dir() {
    printf '%s' "/opt/xcelsior-backups/.deploy-meta"
}

load_remote_deploy_hash() {
    local name="$1"
    printf '%s' "${REMOTE_DEPLOY_HASHES[$name]:-}"
}

load_remote_deploy_hashes() {
    local meta line name value
    meta=$(remote_deploy_meta_dir)
    declare -gA REMOTE_DEPLOY_HASHES=()
    while IFS='|' read -r name value; do
        [[ -n "$name" ]] && REMOTE_DEPLOY_HASHES["$name"]="$value"
    done < <(ssh_cmd "META='$meta'; for n in api frontend nginx runtime; do
        printf '%s|' \"\$n\"
        cat \"\$META/\$n\" 2>/dev/null || true
        printf '\n'
    done" 2>/dev/null || true)
}

store_remote_deploy_hash() {
    local name="$1" value="$2"
    ssh_cmd "mkdir -p '$(remote_deploy_meta_dir)' && printf '%s' '$value' > '$(remote_deploy_meta_dir)/$name'"
}

store_remote_deploy_hashes() {
    local meta payload
    meta=$(remote_deploy_meta_dir)
    payload=$(printf 'api=%s\nfrontend=%s\nnginx=%s\nruntime=%s\n' \
        "$DEPLOY_API_HASH" "$DEPLOY_FRONTEND_HASH" "$DEPLOY_NGINX_HASH" "$DEPLOY_RUNTIME_HASH")
    ssh_cmd "META='$meta'; mkdir -p \"\$META\"
while IFS='=' read -r k v; do
  [[ -z \"\$k\" ]] && continue
  printf '%s' \"\$v\" > \"\$META/\$k\"
done <<'HASHES'
$payload
HASHES"
}

DEPLOY_API_HASH=""
DEPLOY_FRONTEND_HASH=""
DEPLOY_NGINX_HASH=""
DEPLOY_RUNTIME_HASH=""
DEPLOY_BUILD_API=true
DEPLOY_BUILD_FRONTEND=true
DEPLOY_INSTALL_NGINX=true
DEPLOY_RUNTIME_CHANGED=true

detect_deploy_inputs() {
    local env_rel hash_dir
    env_rel="${ENV_FILE#$PROJECT_DIR/}"
    hash_dir=$(mktemp -d)
    _deploy_mark "diff-start"

    # One SSH round-trip for all remote hashes (was 4 sequential calls).
    load_remote_deploy_hashes & local pid_remote=$!

    # Hash local inputs in parallel (CPU-bound).
    hash_repo_subset .dockerignore Dockerfile requirements.txt alembic.ini pyproject.toml "*.py" routes templates migrations \
        >"$hash_dir/api" &
    frontend_build_hash >"$hash_dir/frontend" &
    hash_repo_subset nginx >"$hash_dir/nginx" &
    hash_repo_subset docker-compose.yml "$env_rel" >"$hash_dir/runtime" &
    wait

    wait "$pid_remote" 2>/dev/null || true

    DEPLOY_API_HASH=$(cat "$hash_dir/api")
    DEPLOY_FRONTEND_HASH=$(cat "$hash_dir/frontend")
    DEPLOY_NGINX_HASH=$(cat "$hash_dir/nginx")
    DEPLOY_RUNTIME_HASH=$(cat "$hash_dir/runtime")
    rm -rf "$hash_dir"

    local prev_api_hash prev_frontend_hash prev_nginx_hash prev_runtime_hash
    prev_api_hash=$(load_remote_deploy_hash api)
    prev_frontend_hash=$(load_remote_deploy_hash frontend)
    prev_nginx_hash=$(load_remote_deploy_hash nginx)
    prev_runtime_hash=$(load_remote_deploy_hash runtime)

    [[ -n "$prev_api_hash" && "$DEPLOY_API_HASH" == "$prev_api_hash" ]] && DEPLOY_BUILD_API=false || DEPLOY_BUILD_API=true
    [[ -n "$prev_frontend_hash" && "$DEPLOY_FRONTEND_HASH" == "$prev_frontend_hash" ]] && DEPLOY_BUILD_FRONTEND=false || DEPLOY_BUILD_FRONTEND=true
    [[ -n "$prev_nginx_hash" && "$DEPLOY_NGINX_HASH" == "$prev_nginx_hash" ]] && DEPLOY_INSTALL_NGINX=false || DEPLOY_INSTALL_NGINX=true
    [[ -n "$prev_runtime_hash" && "$DEPLOY_RUNTIME_HASH" == "$prev_runtime_hash" ]] && DEPLOY_RUNTIME_CHANGED=false || DEPLOY_RUNTIME_CHANGED=true

    _deploy_mark "diff-done"
    log "Deploy diff: api_build=${DEPLOY_BUILD_API} frontend_build=${DEPLOY_BUILD_FRONTEND} nginx=${DEPLOY_INSTALL_NGINX} runtime=${DEPLOY_RUNTIME_CHANGED}"
}

persist_deploy_inputs() {
    store_remote_deploy_hashes
}

deploy_docker() {
    log "Deploying with Docker Compose ($TARGET_ENV)..."
    
    # Verify .env exists on server
    ssh_cmd "test -f /opt/xcelsior/.env" || error ".env file not found on server"

    local build_prefix
    build_prefix=$(remote_docker_build_prefix)

    if [[ "$DEPLOY_BUILD_API" == true && "$DEPLOY_BUILD_FRONTEND" == true ]]; then
        validate_build_env
        verify_frontend_desktop_runtime_on_remote
        log "Building API + frontend images in parallel on remote (BuildKit cache=${DEPLOY_BUILD_CACHE})..."
        local build_args
        build_args=$(collect_frontend_build_args)
        ssh_cmd "cd /opt/xcelsior && set -e
$build_prefix
api_pid=''
fe_pid=''
(docker compose --profile blue build api api-blue scheduler-worker bg-worker) & api_pid=\$!
(docker compose build $build_args frontend) & fe_pid=\$!
wait \"\$api_pid\"
wait \"\$fe_pid\"
" || error "Parallel docker build failed"
        success "API + frontend images built (parallel)"
    elif [[ "$DEPLOY_BUILD_API" == true ]]; then
        log "Building API + scheduler-worker + bg-worker images (BuildKit cache=${DEPLOY_BUILD_CACHE})..."
        ssh_cmd "cd /opt/xcelsior && $build_prefix docker compose --profile blue build api api-blue scheduler-worker bg-worker" || error "API/scheduler-worker/bg-worker build failed"
        success "API + scheduler-worker + bg-worker images built"
    elif [[ "$DEPLOY_BUILD_FRONTEND" == true ]]; then
        validate_build_env
        verify_frontend_desktop_runtime_on_remote
        log "Building frontend image (explicit build args, BuildKit cache=${DEPLOY_BUILD_CACHE})..."
        local build_args
        build_args=$(collect_frontend_build_args)
        ssh_cmd "cd /opt/xcelsior && $build_prefix docker compose build $build_args frontend" || error "Frontend build failed"
        success "Frontend image built"
    else
        log "Image build inputs unchanged — skipping docker image rebuilds"
    fi

    # Run Alembic migrations (P3/C8 — fatal; silent-warn hides broken schema).
    # If this fails, aborting is far safer than running new code against an
    # old schema. A failed deploy can be rolled back; a corrupted schema can't.
    log "Running database migrations..."
    ssh_cmd "cd /opt/xcelsior && docker compose run --rm api alembic upgrade head" || error "Migration failed — aborting deploy. Fix the migration then rerun scripts/deploy.sh."
    success "Migrations applied"

    # ── Blue-green zero-downtime swap ────────────────────────────────────
    # Only roll the API when the image or runtime (.env) changed. Frontend-only
    # deploys must not touch the live API, ssh-gateway, or scheduler (keeps
    # running instances + web terminals connected).
    local state_file="/opt/xcelsior/.deploy_colour"
    local final_port final_colour
    if [[ "$DEPLOY_BUILD_API" == true || "$DEPLOY_RUNTIME_CHANGED" == true ]]; then
        # State file tracks which colour is currently live.
        # Default: "green" (api on 9500).  After swap: "blue" (api-blue on 9501).
        local live_colour
        live_colour=$(ssh_cmd "cat $state_file 2>/dev/null || echo green")

        local standby_service standby_port live_service live_port
        if [[ "$live_colour" == "green" ]]; then
            live_service="api"        ; live_port=9500
            standby_service="api-blue"; standby_port=9501
        else
            live_service="api-blue"   ; live_port=9501
            standby_service="api"     ; standby_port=9500
        fi

        log "Blue-green deploy: live=$live_colour ($live_service:$live_port) → standby=$standby_service:$standby_port"

        # 1. Start the standby service on the other port
        log "Starting standby API on port $standby_port..."
        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps $standby_service" || error "Standby API ($standby_service) failed to start"

        # 2. Wait for standby to become healthy (cold start can exceed 60s)
        local standby_ok=false
        for i in {1..60}; do
            if ssh_cmd "curl -sf http://localhost:$standby_port/healthz" &>/dev/null; then
                standby_ok=true
                break
            fi
            sleep 2
        done

        if [ "$standby_ok" != true ]; then
            warn "Standby API ($standby_service:$standby_port) not healthy after 120s — aborting swap"
            ssh_cmd "cd /opt/xcelsior && docker compose --profile blue stop $standby_service 2>/dev/null" || true
            error "Blue-green swap aborted — live API ($live_service:$live_port) left running. Check standby logs and retry."
        fi
        success "Standby API ($standby_service:$standby_port) is healthy"

        # 3. Swap nginx upstream to point at the standby (now primary)
        log "Swapping nginx upstream to port $standby_port..."
        ssh_cmd "sudo sed -i \
            -e '/upstream xcelsior_api/,/}/{ \
                s/server 127.0.0.1:${standby_port} backup;/server 127.0.0.1:${standby_port};/; \
                s/server 127.0.0.1:${standby_port};/server 127.0.0.1:${standby_port};/; \
                s/server 127.0.0.1:${live_port} backup;/server 127.0.0.1:${live_port} backup;/; \
                s/server 127.0.0.1:${live_port};/server 127.0.0.1:${live_port} backup;/ \
            }' /etc/nginx/sites-available/xcelsior && sudo nginx -t && sudo nginx -s reload" \
            || error "Nginx upstream swap failed"
        success "Nginx now routing to $standby_service:$standby_port"

        # 4. Gracefully stop the old live service (30s drain via stop_grace_period)
        log "Draining old API ($live_service:$live_port)..."
        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue stop -t 30 $live_service" || warn "Old API stop returned non-zero"
        success "Old API ($live_service) stopped"

        # 5. Record the new live colour
        local new_colour
        if [[ "$live_colour" == "green" ]]; then new_colour="blue"; else new_colour="green"; fi
        ssh_cmd "echo $new_colour > $state_file"
        success "Deploy state: $new_colour is now live"

        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps scheduler-worker" || error "Scheduler-worker restart failed"
        success "Scheduler-worker restarted"

        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps bg-worker" || error "bg-worker restart failed"
        success "bg-worker restarted"

        # ssh-gateway shares API/terminal routes — restart only when API code changed.
        ssh_cmd "cd /opt/xcelsior && $build_prefix docker compose up -d --no-deps --build ssh-gateway" || error "ssh-gateway restart failed"
        success "ssh-gateway restarted"
    else
        log "API image/runtime unchanged — skipping blue-green swap (live API + ssh-gateway untouched)"
    fi

    if [[ "$DEPLOY_BUILD_FRONTEND" == true ]]; then
        ssh_cmd "cd /opt/xcelsior && docker compose up -d --no-deps frontend" || error "Frontend restart failed"
        success "Frontend restarted"
    else
        log "Frontend image unchanged — skipping frontend restart"
    fi

    log "Starting Jaeger (OTLP trace collector)..."
    ssh_cmd "cd /opt/xcelsior && docker compose up -d jaeger" || warn "Jaeger failed to start — traces will not export"
    ssh_cmd "curl -sf http://127.0.0.1:4317 >/dev/null 2>&1 || true"  # gRPC port open check via compose health
    if ssh_cmd "docker ps --format '{{.Names}}' | grep -q jaeger"; then
        success "Jaeger running (UI http://127.0.0.1:16686 on server)"
    else
        warn "Jaeger container not running — set OTEL_EXPORTER_OTLP_ENDPOINT after fixing"
    fi

    # Final health check — whichever port is now live
    final_colour=$(ssh_cmd "cat $state_file 2>/dev/null || echo green")
    if [[ "$final_colour" == "blue" ]]; then final_port=9501; else final_port=9500; fi

    log "Verifying API on port $final_port ($final_colour)..."
    local healthy=false
    for i in {1..15}; do
        if ssh_cmd "curl -sf --max-time 3 http://localhost:$final_port/healthz" 2>/dev/null | grep -q '"ok"'; then
            healthy=true
            break
        fi
        sleep 2
    done

    if [ "$healthy" = true ]; then
        success "API is healthy on port $final_port"
    else
        warn "API not healthy after 30s — fetching logs..."
        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue logs --tail=30" || true
    fi

    log "Verifying worker PATCH OAuth gate in running API container…"
    local api_container
    api_container=$(ssh_cmd "docker ps --format '{{.Names}}' | grep -E 'xcelsior-api' | head -1" 2>/dev/null || true)
    if [[ -n "$api_container" ]] && ssh_cmd "docker exec $api_container grep -q 'client_credentials' /app/routes/instances.py" 2>/dev/null; then
        success "Worker PATCH accepts OAuth client_credentials"
    else
        warn "Worker PATCH OAuth gate missing in container — jobs may stick in leased/starting"
    fi

    log "Verifying /readyz and PayPal (parallel)…"
    local readyz_out="" paypal_out=""
    readyz_out=$(ssh_cmd "curl -sf --max-time 10 http://localhost:$final_port/readyz" 2>/dev/null || true) &
    local pid_readyz=$!
    paypal_out=$(ssh_cmd "curl -sf --max-time 10 http://localhost:$final_port/api/billing/paypal/enabled" 2>/dev/null || true) &
    local pid_paypal=$!
    wait "$pid_readyz" 2>/dev/null || true
    wait "$pid_paypal" 2>/dev/null || true
    if grep -q nfs_volumes <<<"$readyz_out"; then
        success "readyz includes nfs_volumes probe"
    else
        warn "readyz missing nfs_volumes — deploy may be on an older API build"
    fi
    if grep -q '"enabled":true' <<<"$paypal_out"; then
        success "PayPal enabled on API"
    else
        warn "PayPal not enabled — check PAYPAL_CLIENT_ID/SECRET in .env"
    fi

    ssh_cmd "cd /opt/xcelsior && docker compose --profile blue ps"

    # Prune dangling images; preserve BuildKit layer cache when enabled.
    if [[ "$DEPLOY_BUILD_CACHE" == "1" ]]; then
        log "Pruning dangling Docker images (keeping BuildKit cache)..."
        ssh_cmd "docker image prune -f 2>/dev/null" || true
    else
        log "Pruning unused Docker images and build cache..."
        ssh_cmd "docker image prune -af 2>/dev/null; docker builder prune -af --keep-storage=1G 2>/dev/null" || true
    fi
    success "Docker cleanup complete"
    success "Docker deployment complete (blue-green)"
}

deploy_systemd() {
    log "Deploying with systemd..."
    
    ssh_cmd << 'EOF'
set -e
cd /opt/xcelsior

# Create venv if needed
if [ ! -d venv ]; then
    python3.12 -m venv venv || python3.11 -m venv venv || python3 -m venv venv
fi

# Install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ensure .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found"
    exit 1
fi

# Restart services
sudo systemctl restart xcelsior-api
sudo systemctl restart xcelsior-health

# Wait and check health
sleep 5
if curl -sf http://localhost:9500/healthz > /dev/null; then
    echo "API is healthy!"
else
    echo "WARNING: Health check failed"
    sudo journalctl -u xcelsior-api --no-pager -n 50
fi

sudo systemctl status xcelsior-api --no-pager
EOF
    success "Systemd deployment complete"
}

# ── Rollback ──────────────────────────────────────────────────────────
rollback() {
    log "Rolling back to previous version..."
    
    # Find latest backup
    local latest
    latest=$(ssh_cmd "ls -t /opt/xcelsior-backups/xcelsior_*.tar.gz 2>/dev/null | head -1")
    [[ -z "$latest" ]] && error "No backups found"
    log "Rolling back to: $latest"

    # Stop, restore, restart
    ssh_cmd "docker compose -f /opt/xcelsior/docker-compose.yml down 2>/dev/null || true"
    ssh_cmd "sudo rm -rf /opt/xcelsior && sudo mkdir -p /opt/xcelsior && sudo tar -xzf '$latest' -C /opt && sudo chown -R \$USER:\$USER /opt/xcelsior"
    ssh_cmd "cd /opt/xcelsior && docker compose up -d"
    success "Rollback complete"
}

# ── Local Test Deployment ──────────────────────────────────────────────
deploy_test_local() {
    log "Deploying test environment locally..."

    # Run Alembic migrations on the test database
    log "Running test database migrations..."
    (
        cd "$PROJECT_DIR"
        export XCELSIOR_POSTGRES_DSN
        XCELSIOR_POSTGRES_DSN=$(grep '^XCELSIOR_POSTGRES_DSN=' "$ENV_FILE" | cut -d= -f2-)
        source venv/bin/activate 2>/dev/null || true
        # C8 — fatal on migration failure. Silent-warn hides real bugs;
        # aborting here forces the operator to fix the migration before
        # any downstream test run looks deceptively green.
        alembic upgrade head || { echo "ERROR: alembic upgrade head failed"; exit 1; }
    ) || error "Test database migration failed — aborting."
    success "Test database migrated"

    # Start with docker compose using the test env file
    log "Starting test containers..."
    cd "$PROJECT_DIR"
    docker compose --env-file .env.test -p xcelsior-test up -d --build

    # Wait for health
    local api_port
    api_port=$(grep '^XCELSIOR_BASE_URL=' "$ENV_FILE" | grep -oP ':\K[0-9]+' || echo "9501")
    log "Waiting for test API on port $api_port..."
    for i in {1..20}; do
        if curl -sf "http://localhost:$api_port/healthz" > /dev/null 2>&1; then
            success "Test API is healthy at http://localhost:$api_port"
            break
        fi
        if [[ $i -eq 20 ]]; then
            warn "Test API not responding. Check: docker compose -p xcelsior-test logs"
        fi
        sleep 2
    done

    docker compose -p xcelsior-test ps

    log "Spot test smoke (infra)..."
    if python3 "$SCRIPT_DIR/spot_staging_smoke.py" --base-url "http://localhost:$api_port"; then
        success "Spot infra smoke passed"
    else
        warn "Spot infra smoke failed — check test API logs"
    fi

    switch_local_worker test

    success "Test environment running (worker -> test)"
    log "Run: python3 scripts/spot_preemption_live.py"
    log "Restore prod: ./scripts/deploy.sh --test-stop"
}

stop_test_local() {
    log "Stopping test environment..."
    cd "$PROJECT_DIR"
    docker compose --env-file .env.test -p xcelsior-test down --remove-orphans
    switch_local_worker prod
    success "Test environment stopped; local worker restored to prod"
}

restore_prod_worker() {
    switch_local_worker prod
}

# ── Health Check ──────────────────────────────────────────────────────
health_check() {
    log "Running health checks..."

    if ! ssh_cmd "systemctl is-active --quiet nginx"; then
        warn "nginx not active under systemd — attempting repair"
        repair_nginx_systemd || warn "nginx repair failed"
    fi
    
    # Public check via nginx / Cloudflare with a short retry window after restarts
    local public_ok=false
    for _ in {1..10}; do
        if curl -sf "https://$DOMAIN/healthz" > /dev/null; then
            public_ok=true
            break
        fi
        sleep 2
    done
    if [ "$public_ok" = true ]; then
        success "Public endpoint healthy: https://$DOMAIN/healthz"
    else
        warn "Public endpoint not responding after retries"
    fi
    
    # Remote internal checks (separate calls for reliability)
    log "Docker status:"
    ssh_cmd "cd /opt/xcelsior && docker compose ps" || true

    log "API health:"
    local live_colour live_port
    live_colour=$(ssh_cmd "tr -d '[:space:]' </opt/xcelsior/.deploy_colour 2>/dev/null || echo green")
    if [[ "$live_colour" == "blue" ]]; then live_port=9501; else live_port=9500; fi
    if ssh_cmd "curl -sf http://127.0.0.1:$live_port/healthz" &>/dev/null; then
        success "API healthy on port $live_port ($live_colour)"
    else
        warn "API not responding on port $live_port ($live_colour)"
    fi

    log "Nginx status:"
    ssh_cmd "sudo systemctl status nginx --no-pager -l | head -10" || true
}

# ── Main ──────────────────────────────────────────────────────────────
print_usage() {
    cat << EOF
${BOLD}Xcelsior Deployment Script${NC}

${CYAN}Usage:${NC}
  $0                    Full production deploy (backup, sync, build, restart)
  $0 --quick            Quick prod deploy (sync + restart, no rebuild)
  $0 --test             Deploy test stack + switch local GPU worker to test API
  $0 --test-stop        Stop test stack + restore local GPU worker to prod
  $0 --worker-prod      Restore local GPU worker to prod (no deploy)
  $0 --worker-test      Point local GPU worker at test API (stack must be up)
  $0 --setup            First-time server setup
  $0 --ssl              Setup SSL certificates
  $0 --rollback         Rollback to previous backup
  $0 --health           Run health checks
  $0 --fix-infra        Repair nginx systemd + start Jaeger on prod (no full deploy)
  $0 --post-merge       Deploy main + migrations + health + post_merge_smoke.sh
  $0 --smoke            Run post_merge_smoke.sh locally (no deploy)
  $0 --systemd          Deploy using systemd instead of Docker
  $0 --drain-host ID    Mark a worker host as draining
  $0 --undrain-host ID  Restore a drained worker host
  $0 --guard-host ID    Fail unless a drained host has no active interactive or serverless workers
  $0 --help             Show this help

${CYAN}Environment files:${NC}
  .env                  Production config → deployed to VPS
  .env.test             Test config → used locally for testing

${CYAN}Environment variables:${NC}
  XCELSIOR_DEPLOY_USER  SSH user (default: linuxuser)
  XCELSIOR_DEPLOY_HOST  VPS IP (default: 149.28.121.61)
  XCELSIOR_SSH_KEY      SSH key path (default: ~/.ssh/xcelsior)
  XCELSIOR_DEPLOY_SYNC  rsync (default, fast delta) or tarball (legacy)
  XCELSIOR_DEPLOY_COMPRESS  zstd (default) | gzip | none — installs zstd/pigz/rsync as needed
  XCELSIOR_DEPLOY_TIMING  Set to 1 to log per-step durations
  XCELSIOR_DEPLOY_SYNC_HOSTS  Comma-separated mirror hosts (rsync-only, parallel)
  XCELSIOR_DEPLOY_BUILD_CACHE  1 (default) keep BuildKit cache between deploys
  XCELSIOR_DOCKER_BUILD_CACHE_DIR  Remote cache dir (default: /opt/xcelsior-backups/docker-build-cache)
  XCELSIOR_SSH_CONTROL_PATH  SSH mux socket (default: ~/.ssh/cm-xcelsior-%r@%h:%p)

${CYAN}Examples:${NC}
  # First-time setup
  ./scripts/deploy.sh --setup
  ./scripts/deploy.sh --ssl
  ./scripts/deploy.sh

  # Normal production deployment
  ./scripts/deploy.sh

  # Quick production update (no rebuild)
  ./scripts/deploy.sh --quick

  # Run test environment locally
  ./scripts/deploy.sh --test

  # Stop test environment and restore prod worker
  ./scripts/deploy.sh --test-stop

  # Manually switch worker only
  ./scripts/deploy.sh --worker-prod
  ./scripts/deploy.sh --worker-test

  # Worker maintenance
  ./scripts/deploy.sh --drain-host gpu-worker-01
  ./scripts/deploy.sh --guard-host gpu-worker-01
  ./scripts/deploy.sh --undrain-host gpu-worker-01
EOF
}

main() {
    case "${1:-}" in
        --help|-h)
            print_usage
            exit 0
            ;;
    esac

    init_deploy_targets
    init_deploy_compress
    ensure_local_deploy_tools
    init_rsync_compress

    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║         XCELSIOR DEPLOYMENT SCRIPT             ║"
    echo "║              xcelsior.ca                       ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"

    if [[ ${#DEPLOY_SYNC_HOSTS[@]} -gt 0 ]]; then
        log "Mirror sync hosts: ${DEPLOY_SYNC_HOSTS[*]} (code-only, parallel with primary deploy)"
    fi
    
    case "${1:-}" in
        --test)
            TARGET_ENV="test"
            resolve_env
            deploy_test_local
            ;;
        --test-stop)
            TARGET_ENV="test"
            resolve_env
            stop_test_local
            ;;
        --worker-prod)
            restore_prod_worker
            ;;
        --worker-test)
            switch_local_worker test
            ;;
        --setup)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            setup_server
            setup_systemd
            success "Setup complete. Run --ssl next, then deploy."
            ;;
        --ssl)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            setup_ssl
            success "SSL setup complete. Ready for deployment."
            ;;
        --quick)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            detect_deploy_inputs
            sync_code
            if [[ "$DEPLOY_INSTALL_NGINX" == true ]]; then
                install_nginx_configs
            else
                log "Nginx configs unchanged — skipping"
            fi
            deploy_docker
            wait_mirror_syncs
            persist_deploy_inputs
            local quick_hash
            quick_hash=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
            ssh_cmd "echo '$quick_hash' | sudo tee /opt/xcelsior/.deploy_hash > /dev/null"
            ;;
        --rollback)
            check_ssh
            rollback
            health_check
            ;;
        --health)
            check_ssh
            health_check
            ;;
        --fix-infra)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            repair_nginx_systemd
            install_nginx_configs
            ssh_cmd "cd /opt/xcelsior && docker compose pull jaeger 2>/dev/null; docker compose up -d jaeger" \
                || warn "Jaeger start failed"
            health_check
            log "OTEL in API logs:"
            ssh_cmd "docker ps --format '{{.Names}}' | grep -E 'api|jaeger'" || true
            ssh_cmd "docker logs \$(docker ps --format '{{.Names}}' | grep api-blue | head -1) 2>&1 | grep OTEL | tail -3" \
                || ssh_cmd "docker logs \$(docker ps --format '{{.Names}}' | grep -E '^xcelsior-api-' | grep -v blue | head -1) 2>&1 | grep OTEL | tail -3" \
                || true
            success "Infrastructure repair complete"
            ;;
        --smoke)
            bash "$SCRIPT_DIR/post_merge_smoke.sh"
            ;;
        --post-merge)
            TARGET_ENV="prod"
            resolve_env
            ensure_prod_worker_if_test
            check_ssh
            detect_deploy_inputs
            backup_current
            sync_code
            if [[ "$DEPLOY_INSTALL_NGINX" == true ]]; then
                install_nginx_configs
            fi
            deploy_docker
            wait_mirror_syncs
            persist_deploy_inputs
            local_hash=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
            ssh_cmd "echo '$local_hash' | sudo tee /opt/xcelsior/.deploy_hash > /dev/null"
            health_check
            bash "$SCRIPT_DIR/post_merge_smoke.sh"
            success "Post-merge deploy + smoke complete"
            ;;
        --systemd)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            backup_current
            sync_code
            install_nginx_configs
            deploy_systemd
            wait_mirror_syncs
            health_check
            ;;
        --drain-host)
            TARGET_ENV="prod"
            resolve_env
            [[ -n "${2:-}" ]] || error "Missing host ID for --drain-host"
            drain_worker_host "$2"
            ;;
        --undrain-host)
            TARGET_ENV="prod"
            resolve_env
            [[ -n "${2:-}" ]] || error "Missing host ID for --undrain-host"
            undrain_worker_host "$2"
            ;;
        --guard-host)
            TARGET_ENV="prod"
            resolve_env
            [[ -n "${2:-}" ]] || error "Missing host ID for --guard-host"
            guard_worker_host "$2"
            ;;
        ""|--docker)
            TARGET_ENV="prod"
            resolve_env
            ensure_prod_worker_if_test
            check_ssh
            detect_deploy_inputs

            # ── Smart conditional steps ──
            local local_hash remote_hash
            local_hash=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
            remote_hash=$(ssh_cmd "cat /opt/xcelsior/.deploy_hash 2>/dev/null" || echo "none")

            if [[ "$local_hash" == "$remote_hash" && "$DEPLOY_BUILD_API" == false && "$DEPLOY_BUILD_FRONTEND" == false && "$DEPLOY_INSTALL_NGINX" == false && "$DEPLOY_RUNTIME_CHANGED" == false ]]; then
                log "Remote is already at ${BOLD}${local_hash:0:8}${NC} — nothing to deploy."
                health_check
                exit 0
            fi

            backup_current
            sync_code

            if [[ "$DEPLOY_INSTALL_NGINX" == true ]]; then
                install_nginx_configs
            else
                log "Nginx configs unchanged — skipping"
            fi

            deploy_docker

            wait_mirror_syncs
            persist_deploy_inputs
            ssh_cmd "echo '$local_hash' | sudo tee /opt/xcelsior/.deploy_hash > /dev/null"
            health_check
            success "Deployment complete! Visit https://$DOMAIN"
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
}

main "$@"
