#!/usr/bin/env bash
# install.sh — Bootstrap installer for Xcelsior provider nodes.
# Downloads and runs the setup wizard, or installs the worker agent directly.
#
# Usage:
#   curl -fsSL https://xcelsior.ca/install.sh | bash
#   # or with options:
#   bash install.sh --wizard          # Full interactive wizard
#   bash install.sh --agent-only      # Just install the worker agent
#
set -euo pipefail

XCELSIOR_API="${XCELSIOR_API_URL:-https://xcelsior.ca}"
CONFIG_DIR="$HOME/.xcelsior"
AGENT_PATH="$CONFIG_DIR/worker_agent.py"
ENV_FILE="$CONFIG_DIR/worker.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${CYAN}[xcelsior]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────────

check_deps() {
    for cmd in python3 docker curl; do
        command -v "$cmd" >/dev/null 2>&1 || fail "$cmd is required but not installed"
    done
    ok "Dependencies: python3, docker, curl"
}

check_nvidia() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu
        gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        ok "GPU detected: $gpu"
    else
        warn "nvidia-smi not found — GPU compute won't be available"
    fi
}

check_docker() {
    if docker info >/dev/null 2>&1; then
        ok "Docker is running"
    else
        fail "Docker is not running or current user lacks permissions. Try: sudo usermod -aG docker \$USER"
    fi

    # Check NVIDIA Container Toolkit
    if docker info 2>/dev/null | grep -qi nvidia; then
        ok "NVIDIA Container Toolkit detected"
    else
        warn "NVIDIA Container Toolkit not detected — GPU passthrough may not work"
        warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
}

# ── Network detection ─────────────────────────────────────────────────

detect_network() {
    HOST_IP=""
    if command -v tailscale >/dev/null 2>&1; then
        HOST_IP=$(tailscale ip -4 2>/dev/null || true)
        if [ -n "$HOST_IP" ]; then
            ok "Headscale mesh network: $HOST_IP"
            return
        fi
    fi

    # Fall back to public IP
    HOST_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || true)
    if [ -n "$HOST_IP" ]; then
        warn "Using public IP: $HOST_IP (mesh networking recommended for security)"
    else
        fail "Cannot determine host IP — check network connectivity"
    fi
}

# ── Agent install ────────────────────────────────────────────────────

install_agent() {
    mkdir -p "$CONFIG_DIR"

    info "Downloading worker agent..."
    # Capture response headers so we can verify the sha256 advertised by the server.
    HEADERS_FILE="$(mktemp)"
    curl -fsSL -D "$HEADERS_FILE" "$XCELSIOR_API/static/worker_agent.py" -o "$AGENT_PATH" \
        || fail "Failed to download worker agent"

    EXPECTED_SHA=$(awk 'tolower($1)=="x-xcelsior-agent-sha256:"{gsub(/\r/,"",$2); print tolower($2)}' "$HEADERS_FILE" | tail -n1)
    rm -f "$HEADERS_FILE"
    if [ -n "$EXPECTED_SHA" ]; then
        ACTUAL_SHA=$(sha256sum "$AGENT_PATH" | awk '{print $1}')
        if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
            rm -f "$AGENT_PATH"
            fail "Worker agent hash mismatch (expected $EXPECTED_SHA, got $ACTUAL_SHA)"
        fi
        ok "Worker agent sha256 verified: $ACTUAL_SHA"
    else
        warn "Server did not advertise X-Xcelsior-Agent-SHA256 — skipping hash verification"
    fi
    chmod +x "$AGENT_PATH"
    ok "Worker agent downloaded to $AGENT_PATH"

    # P2.1 — stage host-side tools that will be bind-mounted read-only into
    # every interactive container at /opt/xcelsior/bin. This lets containers
    # use rsync/git/curl without paying an apt-install tax at boot (the
    # platform hard-caps provisioning at 15 s). All best-effort — if we can't
    # stage tools, containers fall back to whatever the image ships with.
    TOOLS_DIR="${XCELSIOR_TOOLS_DIR:-/var/lib/xcelsior/tools}"
    if sudo mkdir -p "$TOOLS_DIR" 2>/dev/null; then
        _staged=0
        for bin in rsync git curl jq htop less ca-certificates; do
            src="$(command -v "$bin" 2>/dev/null || true)"
            if [ -n "$src" ] && [ -f "$src" ]; then
                if sudo cp -L "$src" "$TOOLS_DIR/" 2>/dev/null; then
                    _staged=$((_staged + 1))
                fi
            fi
        done
        if [ "$_staged" -gt 0 ]; then
            sudo chmod -R a+rx "$TOOLS_DIR" 2>/dev/null || true
            ok "Staged $_staged host tools at $TOOLS_DIR (bind-mounted into containers)"
        else
            warn "No host tools staged at $TOOLS_DIR — install rsync/git on the host for <15s container provisioning"
        fi
    else
        warn "Could not create $TOOLS_DIR (permission?) — containers will rely on image tools"
    fi

    # Prompt for credentials if env file doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        echo ""
        info "Configure your worker agent:"
        read -rp "  Host ID (hostname): " HOST_ID
        HOST_ID="${HOST_ID:-$(hostname)}"
        info "Both API keys and OAuth client credentials are supported."
        read -rp "  Auth method [api-key/oauth-client]: " AUTH_METHOD
        AUTH_METHOD="${AUTH_METHOD:-api-key}"
        API_TOKEN=""
        OAUTH_CLIENT_ID=""
        OAUTH_CLIENT_SECRET=""
        case "$AUTH_METHOD" in
            api-key|api|1)
                read -rp "  API Key: " API_TOKEN
                [ -z "$API_TOKEN" ] && fail "API key is required — create one in Dashboard -> Settings -> API & SSH at $XCELSIOR_API/dashboard/settings"
                ;;
            oauth-client|oauth|2)
                read -rp "  OAuth Client ID: " OAUTH_CLIENT_ID
                read -rp "  OAuth Client Secret: " OAUTH_CLIENT_SECRET
                [ -z "$OAUTH_CLIENT_ID" ] && fail "OAuth client ID is required"
                [ -z "$OAUTH_CLIENT_SECRET" ] && fail "OAuth client secret is required"
                ;;
            *)
                fail "Unknown auth method '$AUTH_METHOD' — use 'api-key' or 'oauth-client'"
                ;;
        esac

        detect_network

        cat > "$ENV_FILE" <<EOF
XCELSIOR_HOST_ID=$HOST_ID
XCELSIOR_SCHEDULER_URL=$XCELSIOR_API
XCELSIOR_HOST_IP=$HOST_IP
EOF
        if [ -n "$API_TOKEN" ]; then
            printf 'XCELSIOR_API_TOKEN=%s\n' "$API_TOKEN" >> "$ENV_FILE"
        fi
        if [ -n "$OAUTH_CLIENT_ID" ]; then
            printf 'XCELSIOR_OAUTH_CLIENT_ID=%s\n' "$OAUTH_CLIENT_ID" >> "$ENV_FILE"
            printf 'XCELSIOR_OAUTH_CLIENT_SECRET=%s\n' "$OAUTH_CLIENT_SECRET" >> "$ENV_FILE"
        fi
        chmod 600 "$ENV_FILE"
        ok "Configuration saved to $ENV_FILE"
    else
        ok "Existing config found at $ENV_FILE"
        # shellcheck disable=SC1090
        source "$ENV_FILE"
    fi
}

install_systemd() {
    info "Installing systemd service..."

    cat > /tmp/xcelsior-worker.service <<EOF
[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=$ENV_FILE
ExecStart=/usr/bin/python3 $AGENT_PATH
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo cp /tmp/xcelsior-worker.service /etc/systemd/system/xcelsior-worker.service
    sudo systemctl daemon-reload
    sudo systemctl enable xcelsior-worker
    sudo systemctl start xcelsior-worker
    ok "Worker agent installed and running as systemd service"
    info "View logs: journalctl -u xcelsior-worker -f"
}

# ── E4 — container registry login on host boot ─────────────────────
# Every GPU host that participates in snapshot push/pull needs to be
# logged into the platform registry before `docker push` can succeed.
# We ship a oneshot systemd unit that runs on boot (and reloads of the
# worker) to do the equivalent of:
#   docker login <registry> -u <username> -p <password>
# The credentials are read from the worker env file, which is the only
# place we persist secrets (mode 600).
install_registry_login() {
    if ! grep -qE '^(XCELSIOR_REGISTRY_URL|XCELSIOR_REGISTRY_USERNAME)=' "$ENV_FILE" 2>/dev/null; then
        warn "No registry credentials in $ENV_FILE — skipping docker login service."
        warn "Snapshots (docker commit + push) will fail until XCELSIOR_REGISTRY_URL + _USERNAME + _PASSWORD are added."
        return 0
    fi

    info "Installing registry login systemd service..."

    sudo tee /usr/local/bin/xcelsior-registry-login.sh >/dev/null <<'SHELL'
#!/usr/bin/env bash
# Oneshot registry login for the Xcelsior worker agent.
# Sourced env: XCELSIOR_REGISTRY_URL, XCELSIOR_REGISTRY_USERNAME, XCELSIOR_REGISTRY_PASSWORD.
set -eu
: "${XCELSIOR_REGISTRY_URL:?missing}"
: "${XCELSIOR_REGISTRY_USERNAME:?missing}"
: "${XCELSIOR_REGISTRY_PASSWORD:?missing}"
# Strip scheme — `docker login` takes bare host.
host="${XCELSIOR_REGISTRY_URL#https://}"
host="${host#http://}"
host="${host%%/*}"
printf '%s' "$XCELSIOR_REGISTRY_PASSWORD" | docker login "$host" \
    --username "$XCELSIOR_REGISTRY_USERNAME" --password-stdin >/dev/null
SHELL
    sudo chmod 755 /usr/local/bin/xcelsior-registry-login.sh

    sudo tee /etc/systemd/system/xcelsior-registry-login.service >/dev/null <<EOF
[Unit]
Description=Xcelsior Registry Login (docker login on boot)
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service
Before=xcelsior-worker.service

[Service]
Type=oneshot
RemainAfterExit=yes
EnvironmentFile=$ENV_FILE
ExecStart=/usr/local/bin/xcelsior-registry-login.sh
# If credentials are missing or wrong, don't block the worker from
# coming up — snapshots just won't push until creds are fixed.
SuccessExitStatus=0 1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable xcelsior-registry-login
    if sudo systemctl start xcelsior-registry-login; then
        ok "Registry login service active"
    else
        warn "Registry login failed — check: journalctl -u xcelsior-registry-login"
    fi
}

# ── Wizard mode ──────────────────────────────────────────────────────

run_wizard() {
    info "Checking for Node.js..."
    if command -v npx >/dev/null 2>&1; then
        info "Launching interactive wizard..."
        npx @xcelsior/wizard@latest
    else
        warn "Node.js not found — using direct agent install instead"
        install_agent
        install_registry_login
        install_systemd
    fi
}

# ── Main ─────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Xcelsior GPU Cloud Installer     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
    echo ""

    local mode="${1:---wizard}"

    check_deps
    check_nvidia
    check_docker

    case "$mode" in
        --wizard|-w)
            run_wizard
            ;;
        --agent-only|-a)
            install_agent
            install_registry_login
            install_systemd
            ;;
        *)
            info "Usage: bash install.sh [--wizard | --agent-only]"
            exit 1
            ;;
    esac

    echo ""
    ok "Setup complete! Your host will appear in the Xcelsior dashboard."
    info "Dashboard: $XCELSIOR_API/dashboard/hosts"
    echo ""
}

main "$@"
