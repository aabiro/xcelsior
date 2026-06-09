#!/usr/bin/env bash
# install.sh — Bootstrap installer for Xcelsior provider nodes.
#
# Usage:
#   curl -fsSL https://xcelsior.ca/install.sh | bash
#   bash install.sh --wizard          # Full interactive wizard
#   bash install.sh --agent-only      # Just install the worker agent
#   bash install.sh --uninstall       # Remove systemd units + config
#   bash install.sh --update          # Re-download agent + restart
#   bash install.sh --help
#
set -euo pipefail

XCELSIOR_API="${XCELSIOR_API_URL:-https://xcelsior.ca}"
CONFIG_DIR="$HOME/.xcelsior"
AGENT_PATH="$CONFIG_DIR/worker_agent.py"
ENV_FILE="$CONFIG_DIR/worker.env"
VENV_DIR="$CONFIG_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
AGENT_SIGNING_PUB_EMBEDDED='-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAa8ybnMNSvU/2/QRbDJ5Y1F6LzusKKu1ACYMeMhyyRXs=
-----END PUBLIC KEY-----'

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${CYAN}[xcelsior]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── Platform guard (B10) ─────────────────────────────────────────────

check_platform() {
    local os arch
    os="$(uname -s 2>/dev/null || echo unknown)"
    arch="$(uname -m 2>/dev/null || echo unknown)"
    case "$os" in
        Linux) ;;
        *) fail "Xcelsior worker install requires Linux (detected: $os)" ;;
    esac
    case "$arch" in
        x86_64|amd64|aarch64|arm64) ;;
        *) fail "Unsupported CPU architecture: $arch (need amd64 or arm64)" ;;
    esac
    if ! command -v systemctl >/dev/null 2>&1; then
        fail "systemd is required but systemctl was not found"
    fi
}

# ── Pre-flight checks ────────────────────────────────────────────────

check_deps() {
    for cmd in python3 docker curl; do
        command -v "$cmd" >/dev/null 2>&1 || fail "$cmd is required but not installed"
    done
    command -v openssl >/dev/null 2>&1 || fail "openssl is required for agent signature verification"
    ok "Dependencies: python3, docker, curl, openssl"
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

    if docker info 2>/dev/null | grep -qi nvidia; then
        ok "NVIDIA Container Toolkit detected"
    else
        warn "NVIDIA Container Toolkit not detected — GPU passthrough may not work"
        warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
}

# ── Network detection (B8) ─────────────────────────────────────────────

_is_private_ip() {
    local ip="$1"
    [[ "$ip" =~ ^10\. ]] && return 0
    [[ "$ip" =~ ^192\.168\. ]] && return 0
    [[ "$ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && return 0
    [[ "$ip" =~ ^100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\. ]] && return 0
    return 1
}

detect_network() {
    HOST_IP=""
    if command -v tailscale >/dev/null 2>&1; then
        if command -v tailscaled >/dev/null 2>&1 && systemctl is-active --quiet tailscaled 2>/dev/null; then
            if tailscale status --json 2>/dev/null | grep -q '"BackendState":"Running"'; then
                HOST_IP=$(tailscale ip -4 2>/dev/null || true)
                if [ -n "$HOST_IP" ]; then
                    ok "Headscale mesh network: $HOST_IP (authenticated)"
                    return
                fi
            else
                warn "tailscaled is running but not authenticated — falling back to public IP"
            fi
        else
            warn "tailscale CLI present but tailscaled not active — falling back to public IP"
        fi
    fi

    for provider in "https://api.ipify.org" "https://ifconfig.me/ip" "https://icanhazip.com"; do
        HOST_IP=$(curl -fsSL --max-time 5 "$provider" 2>/dev/null | tr -d '[:space:]' || true)
        if [ -n "$HOST_IP" ]; then
            if _is_private_ip "$HOST_IP"; then
                warn "Detected RFC1918/CGNAT IP $HOST_IP from $provider — mesh networking recommended"
            else
                warn "Using public IP: $HOST_IP (mesh networking recommended for security)"
            fi
            return
        fi
    done
    fail "Cannot determine host IP — check network connectivity"
}

# ── Agent signature verification (B6) ──────────────────────────────────

verify_agent_signature() {
    local file="$1"
    local sig_b64="$2"
    local pub tmp_sig tmp_pub
    pub="$(mktemp)"
    tmp_sig="$(mktemp)"
    tmp_pub="$(mktemp)"
    printf '%s\n' "$AGENT_SIGNING_PUB_EMBEDDED" > "$pub"
    echo "$sig_b64" | tr -d '[:space:]' | base64 -d > "$tmp_sig" 2>/dev/null \
        || fail "Invalid agent signature encoding"
    if ! openssl pkeyutl -verify -pubin -inkey "$pub" -sigfile "$tmp_sig" -in "$file" >/dev/null 2>&1; then
        rm -f "$pub" "$tmp_sig" "$tmp_pub"
        fail "Agent signature verification failed — refusing install (supply-chain protection)"
    fi
    rm -f "$pub" "$tmp_sig" "$tmp_pub"
    ok "Agent ed25519 signature verified"
}

download_static_file() {
    local name="$1"
    local dest="$2"
    local headers expected_sha actual_sha sig_b64
    headers="$(mktemp)"
    curl -fsSL -D "$headers" "$XCELSIOR_API/static/$name" -o "$dest" \
        || fail "Failed to download $name"
    expected_sha=$(awk 'tolower($1)=="x-xcelsior-agent-sha256:"{gsub(/\r/,"",$2); print tolower($2)}' "$headers" | tail -n1)
    sig_b64=$(awk 'tolower($1)=="x-xcelsior-agent-signature:"{sub(/^[^:]*:[[:space:]]*/,""); gsub(/\r/,""); print}' "$headers" | tail -n1)
    if [ -z "$expected_sha" ]; then
        rm -f "$headers" "$dest"
        fail "Server did not advertise X-Xcelsior-Agent-SHA256 for $name — refusing install"
    fi
    actual_sha=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$actual_sha" != "$expected_sha" ]; then
        rm -f "$headers" "$dest"
        fail "$name hash mismatch (expected $expected_sha, got $actual_sha)"
    fi
    if [ "$name" = "worker_agent.py" ]; then
        if [ -z "$sig_b64" ]; then
            rm -f "$headers" "$dest"
            fail "Server did not advertise X-Xcelsior-Agent-Signature for worker_agent.py — refusing install"
        fi
        verify_agent_signature "$dest" "$sig_b64"
    fi
    rm -f "$headers"
    ok "Downloaded $name (sha256=$actual_sha)"
}

setup_agent_venv() {
    info "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_PYTHON" -m pip install --upgrade pip wheel >/dev/null
    download_static_file "worker-requirements.txt" "$CONFIG_DIR/worker-requirements.txt"
    "$VENV_PYTHON" -m pip install -r "$CONFIG_DIR/worker-requirements.txt"
    ok "Worker dependencies installed in venv"
}

# ── Agent install (B1) ───────────────────────────────────────────────

install_agent() {
    mkdir -p "$CONFIG_DIR" "$CONFIG_DIR/lib"

    info "Downloading worker agent and dependencies..."
    download_static_file "worker_agent.py" "$AGENT_PATH"
    download_static_file "security.py" "$CONFIG_DIR/lib/security.py"
    download_static_file "nvml_telemetry.py" "$CONFIG_DIR/lib/nvml_telemetry.py"
    chmod +x "$AGENT_PATH"

    setup_agent_venv

    # B7 — host-tool staging disabled by default (dynamic binaries break in containers).
    if [ "${XCELSIOR_STAGE_HOST_TOOLS:-0}" = "1" ]; then
        warn "XCELSIOR_STAGE_HOST_TOOLS=1 — staging host binaries (may be incompatible inside containers)"
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
            [ "$_staged" -gt 0 ] && ok "Staged $_staged host tools at $TOOLS_DIR"
        fi
    else
        info "Skipping host-tool staging (use image-bundled tools or XCELSIOR_STAGE_HOST_TOOLS=1)"
    fi

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
                read -rsp "  API Key: " API_TOKEN
                echo ""
                [ -z "$API_TOKEN" ] && fail "API key is required — create one in Dashboard -> Settings -> API & SSH at $XCELSIOR_API/dashboard/settings"
                ;;
            oauth-client|oauth|2)
                read -rp "  OAuth Client ID: " OAUTH_CLIENT_ID
                read -rsp "  OAuth Client Secret: " OAUTH_CLIENT_SECRET
                echo ""
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
PYTHONPATH=$CONFIG_DIR/lib
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

    local unit_tmp
    unit_tmp="$(mktemp)"
    cat > "$unit_tmp" <<EOF
[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=$ENV_FILE
Environment=PYTHONPATH=$CONFIG_DIR/lib
ExecStart=$VENV_PYTHON $AGENT_PATH
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true
ReadWritePaths=$CONFIG_DIR /var/lib/xcelsior /tmp
MemoryMax=4G
OOMScoreAdjust=500
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/xcelsior-worker.service
    rm -f "$unit_tmp"
    sudo systemctl daemon-reload
    sudo systemctl enable xcelsior-worker
    sudo systemctl restart xcelsior-worker
    ok "Worker agent installed and running as systemd service"
    info "View logs: journalctl -u xcelsior-worker -f"
}

# ── Post-install verification (B9) ───────────────────────────────────

verify_host_online() {
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    local host_id="${XCELSIOR_HOST_ID:-}"
    [ -z "$host_id" ] && return 0

    info "Verifying host registration (up to 60s)..."
    local i status
    for i in $(seq 1 12); do
        status=$(curl -fsSL --max-time 5 "$XCELSIOR_API/host/$host_id" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('host') or {}).get('status',''))" 2>/dev/null || true)
        if [ "$status" = "online" ] || [ "$status" = "active" ]; then
            ok "Host $host_id is $status in the dashboard"
            return 0
        fi
        sleep 5
    done
    warn "Host not yet visible as online — check logs:"
    warn "  journalctl -u xcelsior-worker -n 50 --no-pager"
    return 1
}

install_registry_login() {
    if ! grep -qE '^(XCELSIOR_REGISTRY_URL|XCELSIOR_REGISTRY_USERNAME)=' "$ENV_FILE" 2>/dev/null; then
        warn "No registry credentials in $ENV_FILE — skipping docker login service."
        return 0
    fi

    info "Installing registry login systemd service..."

    sudo tee /usr/local/bin/xcelsior-registry-login.sh >/dev/null <<'SHELL'
#!/usr/bin/env bash
set -eu
: "${XCELSIOR_REGISTRY_URL:?missing}"
: "${XCELSIOR_REGISTRY_USERNAME:?missing}"
: "${XCELSIOR_REGISTRY_PASSWORD:?missing}"
host="${XCELSIOR_REGISTRY_URL#https://}"
host="${host#http://}"
host="${host%%/*}"
printf '%s' "$XCELSIOR_REGISTRY_PASSWORD" | docker login "$host" \
    --username "$XCELSIOR_REGISTRY_USERNAME" --password-stdin >/dev/null
SHELL
    sudo chmod 755 /usr/local/bin/xcelsior-registry-login.sh

    local unit_tmp
    unit_tmp="$(mktemp)"
    cat > "$unit_tmp" <<EOF
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
SuccessExitStatus=0 1

[Install]
WantedBy=multi-user.target
EOF
    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/xcelsior-registry-login.service
    rm -f "$unit_tmp"

    sudo systemctl daemon-reload
    sudo systemctl enable xcelsior-registry-login
    sudo systemctl start xcelsior-registry-login || warn "Registry login failed — see journalctl -u xcelsior-registry-login"
}

# ── Wizard mode (B5) ─────────────────────────────────────────────────

run_wizard() {
    info "Checking for Node.js..."
    if command -v npx >/dev/null 2>&1; then
        info "Launching interactive wizard..."
        if npx @xcelsior-gpu/wizard@latest; then
            return 0
        fi
        warn "Wizard failed (network, npm, or package error) — falling back to direct agent install"
    else
        warn "Node.js/npx not found — using direct agent install instead"
    fi
    install_agent
    install_registry_login
    install_systemd
    verify_host_online || true
}

# ── Lifecycle subcommands (B10) ──────────────────────────────────────

show_help() {
    cat <<EOF
Xcelsior GPU Cloud Installer

Usage:
  bash install.sh [--wizard | -w]     Interactive onboarding wizard (default)
  bash install.sh [--agent-only | -a] Install worker agent only
  bash install.sh [--update | -u]     Re-download agent + restart service
  bash install.sh [--uninstall]       Remove systemd units (keeps ~/.xcelsior)
  bash install.sh [--help | -h]       Show this help

Environment:
  XCELSIOR_API_URL   API base URL (default: https://xcelsior.ca)
  XCELSIOR_STAGE_HOST_TOOLS=1  Stage host binaries (not recommended)
EOF
}

do_uninstall() {
    info "Stopping and removing Xcelsior systemd units..."
    sudo systemctl stop xcelsior-worker xcelsior-registry-login 2>/dev/null || true
    sudo systemctl disable xcelsior-worker xcelsior-registry-login 2>/dev/null || true
    sudo rm -f /etc/systemd/system/xcelsior-worker.service \
        /etc/systemd/system/xcelsior-registry-login.service \
        /usr/local/bin/xcelsior-registry-login.sh
    sudo systemctl daemon-reload
    ok "Systemd units removed (config kept at $CONFIG_DIR)"
}

do_update() {
    install_agent
    if systemctl list-unit-files xcelsior-worker.service >/dev/null 2>&1; then
        install_systemd
        verify_host_online || true
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

    case "$mode" in
        --help|-h) show_help; exit 0 ;;
        --uninstall) check_platform; do_uninstall; exit 0 ;;
    esac

    check_platform
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
            verify_host_online || true
            ;;
        --update|-u)
            do_update
            ;;
        *)
            show_help
            exit 1
            ;;
    esac

    echo ""
    ok "Setup complete! Check the dashboard for host status."
    info "Dashboard: $XCELSIOR_API/dashboard/hosts"
    echo ""
}

main "$@"