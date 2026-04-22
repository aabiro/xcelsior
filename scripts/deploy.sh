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

ssh_cmd() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$REMOTE_USER@$REMOTE_HOST" "$@"
}

scp_file() {
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$1" "$REMOTE_USER@$REMOTE_HOST:$2"
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
count = int(summary.get("active_interactive_instances", 0) or 0)

print(f"Host: {host_id}")
print(f"Status: {status}")
print(f"Active interactive instances: {count}")
for item in summary.get("interactive_instances", []):
    print(f"- {item.get('job_id')} {item.get('status')} {item.get('name')}")

if status != "draining":
    print("Unsafe: host is not drained", file=sys.stderr)
    raise SystemExit(2)
if count > 0:
    print("Unsafe: interactive instances are still active", file=sys.stderr)
    raise SystemExit(3)
PY
    success "Host $host_id is safe for maintenance"
}

install_nginx_configs() {
    log "Installing nginx site configs..."

    scp_file "$PROJECT_DIR/nginx/xcelsior.conf" "/tmp/xcelsior.conf"
    scp_file "$PROJECT_DIR/nginx/headscale.conf" "/tmp/headscale.conf"
    scp_file "$PROJECT_DIR/nginx/headscale-http.conf" "/tmp/headscale-http.conf"
    scp_file "$PROJECT_DIR/nginx/docs-xcelsior.conf" "/tmp/docs-xcelsior.conf"
    scp_file "$PROJECT_DIR/nginx/downloads-xcelsior.conf" "/tmp/downloads-xcelsior.conf"

    ssh_cmd << 'EOF'
set -e
sudo cp /tmp/xcelsior.conf /etc/nginx/sites-available/xcelsior
sudo cp /tmp/headscale.conf /etc/nginx/sites-available/headscale
sudo cp /tmp/headscale-http.conf /etc/nginx/sites-available/headscale-http
sudo cp /tmp/docs-xcelsior.conf /etc/nginx/sites-available/docs-xcelsior
sudo cp /tmp/downloads-xcelsior.conf /etc/nginx/sites-available/downloads-xcelsior
sudo ln -sf /etc/nginx/sites-available/xcelsior /etc/nginx/sites-enabled/xcelsior
sudo ln -sf /etc/nginx/sites-available/headscale /etc/nginx/sites-enabled/headscale
sudo ln -sf /etc/nginx/sites-available/headscale-http /etc/nginx/sites-enabled/headscale-http
sudo ln -sf /etc/nginx/sites-available/docs-xcelsior /etc/nginx/sites-enabled/docs-xcelsior
sudo ln -sf /etc/nginx/sites-available/downloads-xcelsior /etc/nginx/sites-enabled/downloads-xcelsior
sudo nginx -t
sudo systemctl reload nginx
EOF
    success "Nginx configs installed"
}

check_ssh() {
    log "Testing SSH connection to $REMOTE_HOST..."
    if ssh_cmd "echo 'SSH OK'" &>/dev/null; then
        success "SSH connection successful"
    else
        error "Cannot connect to $REMOTE_HOST. Check SSH keys and connectivity."
    fi
}

# ── First-Time Setup ──────────────────────────────────────────────────
setup_server() {
    log "Running first-time server setup..."
    
    ssh_cmd << 'EOF'
set -e

echo "=== Installing dependencies ==="
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx docker.io docker-compose-plugin postgresql postgresql-client

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

sync_code() {
    log "Syncing code to server..."
    local remote_tarball

    # Auto-clean stale deploy archives so repeated deploys don't fill /tmp.
    ssh_cmd '
set -e
for d in /tmp /var/tmp /opt/xcelsior-backups /opt/xcelsior-backups/staging; do
    if [ -d "$d" ]; then
        find "$d" -maxdepth 1 -type f -name "xcelsior_deploy*.tar.gz" -mtime +2 -delete 2>/dev/null || true
    fi
done
' || true

    # Create tarball locally (excluding unnecessary files)
    TARBALL="/tmp/xcelsior_deploy.tar.gz"
    tar -czf "$TARBALL" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='venv' \
        --exclude='node_modules' \
        --exclude='.next' \
        --exclude='*.db' \
        --exclude='*.db-*' \
        --exclude='*.log' \
        --exclude='data/*' \
        --exclude='./artifacts/*' \
        --exclude='.env' \
        --exclude='./desktop' \
        --exclude='./desktop/*' \
        --exclude='checkpoints' \
        -C "$PROJECT_DIR" .

    # Try multiple remote staging paths in order; /tmp can be full on busy hosts.
    remote_tarball=""
    local candidate
    for candidate in /tmp /var/tmp /opt/xcelsior-backups/staging /opt/xcelsior-backups; do
        ssh_cmd "mkdir -p '$candidate' 2>/dev/null || sudo mkdir -p '$candidate' || true; sudo chown \$USER:\$USER '$candidate' 2>/dev/null || true" || true
        local probe_target
        probe_target="${candidate%/}/xcelsior_deploy_$(date +%s).tar.gz"
        if scp_file "$TARBALL" "$probe_target" 2>/dev/null; then
            remote_tarball="$probe_target"
            break
        fi
    done
    [[ -n "$remote_tarball" ]] || error "Failed to upload deploy artifact to all staging paths (/tmp, /var/tmp, /opt/xcelsior-backups)"
    log "Using remote staging path: $remote_tarball"
    rm "$TARBALL"
    
    ssh_cmd "DEPLOY_ARCHIVE='$remote_tarball' bash -s" << 'EOF'
set -e
sudo mkdir -p /opt/xcelsior /opt/xcelsior_new
# Preserve server-specific files (but NOT .env — we'll send the right one)
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/opt/xcelsior/$f" ] && sudo cp "/opt/xcelsior/$f" "/tmp/xcelsior_preserve_$f" || true
done
sudo rm -rf /opt/xcelsior_new/*
sudo tar -xzf "$DEPLOY_ARCHIVE" -C /opt/xcelsior_new
# Restore preserved files
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/tmp/xcelsior_preserve_$f" ] && sudo cp "/tmp/xcelsior_preserve_$f" "/opt/xcelsior_new/$f" || true
    sudo rm -f "/tmp/xcelsior_preserve_$f"
done
sudo rm -rf /opt/xcelsior
sudo mv /opt/xcelsior_new /opt/xcelsior
sudo chown -R $USER:$USER /opt/xcelsior
rm -f "$DEPLOY_ARCHIVE"
EOF

    # Send the correct env file as .env on the server
    log "Sending $TARGET_ENV environment config..."
    scp_file "$ENV_FILE" "/tmp/xcelsior_env"
    ssh_cmd "sudo cp /tmp/xcelsior_env /opt/xcelsior/.env && sudo chown \$USER:\$USER /opt/xcelsior/.env && rm /tmp/xcelsior_env"
    success "Code synced (env=$TARGET_ENV)"
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
    ssh_cmd "cat '$(remote_deploy_meta_dir)/$name' 2>/dev/null" || echo ""
}

store_remote_deploy_hash() {
    local name="$1" value="$2"
    ssh_cmd "mkdir -p '$(remote_deploy_meta_dir)' && printf '%s' '$value' > '$(remote_deploy_meta_dir)/$name'"
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
    local env_rel
    env_rel="${ENV_FILE#$PROJECT_DIR/}"

    DEPLOY_API_HASH=$(hash_repo_subset .dockerignore Dockerfile requirements.txt alembic.ini pyproject.toml "*.py" routes templates migrations)
    DEPLOY_FRONTEND_HASH=$(frontend_build_hash)
    DEPLOY_NGINX_HASH=$(hash_repo_subset nginx)
    DEPLOY_RUNTIME_HASH=$(hash_repo_subset docker-compose.yml "$env_rel")

    local prev_api_hash prev_frontend_hash prev_nginx_hash prev_runtime_hash
    prev_api_hash=$(load_remote_deploy_hash api)
    prev_frontend_hash=$(load_remote_deploy_hash frontend)
    prev_nginx_hash=$(load_remote_deploy_hash nginx)
    prev_runtime_hash=$(load_remote_deploy_hash runtime)

    [[ -n "$prev_api_hash" && "$DEPLOY_API_HASH" == "$prev_api_hash" ]] && DEPLOY_BUILD_API=false || DEPLOY_BUILD_API=true
    [[ -n "$prev_frontend_hash" && "$DEPLOY_FRONTEND_HASH" == "$prev_frontend_hash" ]] && DEPLOY_BUILD_FRONTEND=false || DEPLOY_BUILD_FRONTEND=true
    [[ -n "$prev_nginx_hash" && "$DEPLOY_NGINX_HASH" == "$prev_nginx_hash" ]] && DEPLOY_INSTALL_NGINX=false || DEPLOY_INSTALL_NGINX=true
    [[ -n "$prev_runtime_hash" && "$DEPLOY_RUNTIME_HASH" == "$prev_runtime_hash" ]] && DEPLOY_RUNTIME_CHANGED=false || DEPLOY_RUNTIME_CHANGED=true

    log "Deploy diff: api_build=${DEPLOY_BUILD_API} frontend_build=${DEPLOY_BUILD_FRONTEND} nginx=${DEPLOY_INSTALL_NGINX} runtime=${DEPLOY_RUNTIME_CHANGED}"
}

persist_deploy_inputs() {
    store_remote_deploy_hash api "$DEPLOY_API_HASH"
    store_remote_deploy_hash frontend "$DEPLOY_FRONTEND_HASH"
    store_remote_deploy_hash nginx "$DEPLOY_NGINX_HASH"
    store_remote_deploy_hash runtime "$DEPLOY_RUNTIME_HASH"
}

deploy_docker() {
    log "Deploying with Docker Compose ($TARGET_ENV)..."
    
    # Verify .env exists on server
    ssh_cmd "test -f /opt/xcelsior/.env" || error ".env file not found on server"

    if [[ "$DEPLOY_BUILD_API" == true ]]; then
        log "Building API + scheduler-worker + bg-worker images..."
        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue build api api-blue scheduler-worker bg-worker" || error "API/scheduler-worker/bg-worker build failed"
        success "API + scheduler-worker + bg-worker images built"
    else
        log "API build inputs unchanged — skipping api/scheduler-worker image rebuild"
    fi

    if [[ "$DEPLOY_BUILD_FRONTEND" == true ]]; then
        validate_build_env
        log "Building frontend image (explicit build args)..."
        local build_args
        build_args=$(collect_frontend_build_args)
        ssh_cmd "cd /opt/xcelsior && docker compose build $build_args frontend" || error "Frontend build failed"
        success "Frontend image built"
    else
        log "Frontend build inputs unchanged — skipping frontend image rebuild"
    fi

    # Run Alembic migrations (P3/C8 — fatal; silent-warn hides broken schema).
    # If this fails, aborting is far safer than running new code against an
    # old schema. A failed deploy can be rolled back; a corrupted schema can't.
    log "Running database migrations..."
    ssh_cmd "cd /opt/xcelsior && docker compose run --rm api alembic upgrade head" || error "Migration failed — aborting deploy. Fix the migration then rerun scripts/deploy.sh."
    success "Migrations applied"

    # ── Blue-green zero-downtime swap ────────────────────────────────────
    # State file tracks which colour is currently live.
    # Default: "green" (api on 9500).  After swap: "blue" (api-blue on 9501).
    local state_file="/opt/xcelsior/.deploy_colour"
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

    # 2. Wait for standby to become healthy
    local standby_ok=false
    for i in {1..30}; do
        if ssh_cmd "curl -sf http://localhost:$standby_port/healthz" &>/dev/null; then
            standby_ok=true
            break
        fi
        sleep 2
    done

    if [ "$standby_ok" != true ]; then
        warn "Standby API ($standby_service:$standby_port) not healthy after 60s — aborting swap"
        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue stop $standby_service 2>/dev/null" || true
        # Fall back: just restart the live service in-place
        log "Falling back to in-place restart of $live_service..."
        ssh_cmd "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps $live_service" || error "In-place API restart failed"
        local fallback_ok=false
        for i in {1..20}; do
            if ssh_cmd "curl -sf http://localhost:$live_port/healthz" &>/dev/null; then
                fallback_ok=true; break
            fi
            sleep 2
        done
        if [ "$fallback_ok" = true ]; then
            success "API is healthy (in-place restart — no zero-downtime this deploy)"
        else
            warn "API not healthy after in-place restart"
        fi
    else
        success "Standby API ($standby_service:$standby_port) is healthy"

        # 3. Swap nginx upstream to point at the standby (now primary)
        #    We use sed to reorder the upstream servers so the new primary is first.
        log "Swapping nginx upstream to port $standby_port..."
        ssh_cmd "sudo sed -i \
            -e '/upstream xcelsior_api/,/}/{ \
                s/server 127.0.0.1:${standby_port} backup;/server 127.0.0.1:${standby_port};/; \
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
    fi

    ssh_cmd "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps scheduler-worker" || error "Scheduler-worker restart failed"
    success "Scheduler-worker restarted"

    ssh_cmd "cd /opt/xcelsior && docker compose --profile blue up -d --no-deps bg-worker" || error "bg-worker restart failed"
    success "bg-worker restarted"

    # ssh-gateway shares the same codebase (ssh_gateway.py) and is
    # rebuilt from the same Dockerfile, so it must be restarted whenever
    # we ship new code. `--no-deps` keeps docker-compose from touching
    # unrelated services. `--build` ensures the image picks up the new
    # ssh_gateway.py (the main api build above doesn't target this image).
    ssh_cmd "cd /opt/xcelsior && docker compose up -d --no-deps --build ssh-gateway" || error "ssh-gateway restart failed"
    success "ssh-gateway restarted"

    ssh_cmd "cd /opt/xcelsior && docker compose up -d --no-deps frontend" || error "Frontend restart failed"
    success "Frontend restarted"

    # Final health check — whichever port is now live
    local final_port final_colour
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

    ssh_cmd "cd /opt/xcelsior && docker compose --profile blue ps"

    # Clean up dangling images and build cache to prevent disk bloat
    log "Pruning unused Docker images and build cache..."
    ssh_cmd "docker image prune -af 2>/dev/null; docker builder prune -af --keep-storage=1G 2>/dev/null" || true
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
    success "Test environment running"
}

stop_test_local() {
    log "Stopping test environment..."
    cd "$PROJECT_DIR"
    docker compose --env-file .env.test -p xcelsior-test down --remove-orphans
    success "Test environment stopped"
}

# ── Health Check ──────────────────────────────────────────────────────
health_check() {
    log "Running health checks..."
    
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
    local live_port
    live_port=$(ssh_cmd "cat /opt/xcelsior/.deploy-colour 2>/dev/null || echo green")
    if [[ "$live_port" == "blue" ]]; then live_port=9501; else live_port=9500; fi
    ssh_cmd "curl -s http://localhost:$live_port/healthz" || warn "API not responding"

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
  $0 --test             Deploy test environment locally (Docker)
  $0 --test-stop        Stop local test environment
  $0 --setup            First-time server setup
  $0 --ssl              Setup SSL certificates
  $0 --rollback         Rollback to previous backup
  $0 --health           Run health checks
  $0 --systemd          Deploy using systemd instead of Docker
  $0 --drain-host ID    Mark a worker host as draining
  $0 --undrain-host ID  Restore a drained worker host
  $0 --guard-host ID    Fail unless a drained worker host has no active interactive instances
  $0 --help             Show this help

${CYAN}Environment files:${NC}
  .env                  Production config → deployed to VPS
  .env.test             Test config → used locally for testing

${CYAN}Environment variables:${NC}
  XCELSIOR_DEPLOY_USER  SSH user (default: linuxuser)
  XCELSIOR_DEPLOY_HOST  VPS IP (default: 149.28.121.61)
  XCELSIOR_SSH_KEY      SSH key path (default: ~/.ssh/xcelsior)

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

  # Stop test environment
  ./scripts/deploy.sh --test-stop

  # Worker maintenance
  ./scripts/deploy.sh --drain-host gpu-worker-01
  ./scripts/deploy.sh --guard-host gpu-worker-01
  ./scripts/deploy.sh --undrain-host gpu-worker-01
EOF
}

main() {
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════╗"
    echo "║         XCELSIOR DEPLOYMENT SCRIPT             ║"
    echo "║              xcelsior.ca                       ║"
    echo "╚════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    case "${1:-}" in
        --help|-h)
            print_usage
            exit 0
            ;;
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
        --systemd)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            backup_current
            sync_code
            install_nginx_configs
            deploy_systemd
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

            if [[ "$DEPLOY_BUILD_API" == true || "$DEPLOY_BUILD_FRONTEND" == true ]]; then
                deploy_docker
            else
                log "Image build inputs unchanged — running migrations and container refresh only"
                # P3/C8 — fatal on migration failure; see note at deploy_docker().
                ssh_cmd "cd /opt/xcelsior && docker compose run --rm api alembic upgrade head" || error "Migration failed — aborting deploy. Fix the migration then rerun scripts/deploy.sh."
                ssh_cmd "cd /opt/xcelsior && docker compose up -d" || error "Docker up failed"
            fi

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
