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

install_nginx_configs() {
    log "Installing nginx site configs..."

    scp_file "$PROJECT_DIR/nginx/xcelsior.conf" "/tmp/xcelsior.conf"
    scp_file "$PROJECT_DIR/nginx/headscale.conf" "/tmp/headscale.conf"
    scp_file "$PROJECT_DIR/nginx/headscale-http.conf" "/tmp/headscale-http.conf"

    ssh_cmd << 'EOF'
set -e
sudo cp /tmp/xcelsior.conf /etc/nginx/sites-available/xcelsior
sudo cp /tmp/headscale.conf /etc/nginx/sites-available/headscale
sudo cp /tmp/headscale-http.conf /etc/nginx/sites-available/headscale-http
sudo ln -sf /etc/nginx/sites-available/xcelsior /etc/nginx/sites-enabled/xcelsior
sudo ln -sf /etc/nginx/sites-available/headscale /etc/nginx/sites-enabled/headscale
sudo ln -sf /etc/nginx/sites-available/headscale-http /etc/nginx/sites-enabled/headscale-http
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
        -C "$PROJECT_DIR" .
    
    scp_file "$TARBALL" "/tmp/xcelsior_deploy.tar.gz"
    rm "$TARBALL"
    
    ssh_cmd << 'EOF'
set -e
sudo mkdir -p /opt/xcelsior
# Preserve server-specific files (but NOT .env — we'll send the right one)
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/opt/xcelsior/$f" ] && sudo cp "/opt/xcelsior/$f" "/tmp/xcelsior_preserve_$f" || true
done
sudo tar -xzf /tmp/xcelsior_deploy.tar.gz -C /opt/xcelsior
# Restore preserved files
for f in docker-compose.override.yml docker-compose.prod.yml; do
    [ -f "/tmp/xcelsior_preserve_$f" ] && sudo cp "/tmp/xcelsior_preserve_$f" "/opt/xcelsior/$f" || true
    sudo rm -f "/tmp/xcelsior_preserve_$f"
done
sudo chown -R $USER:$USER /opt/xcelsior
rm /tmp/xcelsior_deploy.tar.gz
EOF

    # Send the correct env file as .env on the server
    log "Sending $TARGET_ENV environment config..."
    scp_file "$ENV_FILE" "/tmp/xcelsior_env"
    ssh_cmd "sudo cp /tmp/xcelsior_env /opt/xcelsior/.env && sudo chown \$USER:\$USER /opt/xcelsior/.env && rm /tmp/xcelsior_env"
    success "Code synced (env=$TARGET_ENV)"
}

deploy_docker() {
    log "Deploying with Docker Compose ($TARGET_ENV)..."
    
    # Verify .env exists on server
    ssh_cmd "test -f /opt/xcelsior/.env" || error ".env file not found on server"

    # Build images (separate SSH call so a timeout here doesn't leave containers down)
    log "Building Docker images..."
    ssh_cmd "cd /opt/xcelsior && docker compose build" || error "Docker build failed"
    success "Docker images built"

    # Run Alembic migrations
    log "Running database migrations..."
    ssh_cmd "cd /opt/xcelsior && docker compose run --rm api alembic upgrade head" || warn "Migration failed — may need manual attention"
    success "Migrations applied"

    # Recreate containers (fast — images already built)
    log "Restarting containers..."
    ssh_cmd "cd /opt/xcelsior && docker compose down --remove-orphans 2>/dev/null; docker compose up -d" || error "Docker up failed"
    success "Containers started"

    # Wait for health
    log "Waiting for API health..."
    local healthy=false
    for i in {1..30}; do
        if ssh_cmd "curl -sf http://localhost:9500/healthz" &>/dev/null; then
            healthy=true
            break
        fi
        sleep 2
    done

    if [ "$healthy" = true ]; then
        success "API is healthy"
    else
        warn "API not healthy after 60s — fetching logs..."
        ssh_cmd "cd /opt/xcelsior && docker compose logs --tail=30" || true
    fi

    ssh_cmd "cd /opt/xcelsior && docker compose ps"
    success "Docker deployment complete"
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
        alembic upgrade head
    )
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
    ssh_cmd "curl -s http://localhost:9500/healthz" || warn "API not responding"

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
            sync_code
            install_nginx_configs
            deploy_docker
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
        ""|--docker)
            TARGET_ENV="prod"
            resolve_env
            check_ssh
            backup_current
            sync_code
            install_nginx_configs
            deploy_docker
            health_check
            success "Deployment complete! Visit https://$DOMAIN"
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac
}

main "$@"
