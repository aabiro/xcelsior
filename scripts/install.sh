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

# ── Wizard mode ──────────────────────────────────────────────────────

run_wizard() {
    info "Checking for Node.js..."
    if command -v npx >/dev/null 2>&1; then
        info "Launching interactive wizard..."
        npx @xcelsior/wizard@latest
    else
        warn "Node.js not found — using direct agent install instead"
        install_agent
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
