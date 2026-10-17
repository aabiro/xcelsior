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
# Your settings stay in your home directory; systemd reads worker.env as root
# before the service starts, so the worker never needs access to your home.
CONFIG_DIR="${XCELSIOR_CONFIG_DIR:-$HOME/.xcelsior}"
ENV_FILE="$CONFIG_DIR/worker.env"
# The worker runs as a dedicated system user, from code that user owns (so the
# agent can self-update). Root-only steps (mounts, encrypted volumes) go through
# xcelsior-privd, a root helper that validates every request (issue #65).
SERVICE_USER="xcelsior"
INSTALL_DIR="${XCELSIOR_INSTALL_DIR:-/opt/xcelsior}"
LIB_DIR="$INSTALL_DIR/lib"
AGENT_PATH="$INSTALL_DIR/worker_agent.py"
VENV_DIR="$INSTALL_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
PRIVD_DIR="/usr/local/libexec/xcelsior"
PRIVD_CONF="/etc/xcelsior/privileged.conf"
AGENT_TLS_DIR="/etc/xcelsior/agent-tls"
# Host identity for agent.xcelsior.ca (issue #67): a SPIRE agent attests once with
# a single-use join token, then the worker gets hourly X.509-SVIDs (CN=<host_id>).
SPIRE_VERSION="1.11.2"
SPIRE_SHA256_AMD64="ac53853d478175f15cec8a3757d71638d0821126c0d69583ed4141399e01c49f"
SPIRE_SHA256_ARM64="256a8c027eb028c40d27b8d9620763923537102178b3abf698ea12882856cca4"
SVID_DIR="/run/xcelsior-svid"
SPIRE_SOCKET="/run/spire/agent/public/api.sock"
# Our own tailscaled instance: never touches a provider's personal Tailscale.
MESH_SOCKET="/var/run/tailscale-xcelsior.sock"
readonly WIZARD_PACKAGE="@xcelsior-gpu/wizard@0.1.7"
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
    command -v nvidia-smi >/dev/null 2>&1 \
        || fail "nvidia-smi is required — install a supported NVIDIA driver before running the worker"

    local gpu
    if ! gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null) \
        || [ -z "$gpu" ]; then
        fail "nvidia-smi could not detect a usable NVIDIA GPU"
    fi
    ok "GPU detected: $(printf '%s\n' "$gpu" | head -1)"
}

check_docker() {
    if docker info >/dev/null 2>&1; then
        ok "Docker is running"
    else
        fail "Docker is not running or current user lacks permissions. Try: sudo usermod -aG docker \$USER"
    fi

    local runtimes
    if ! runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null) \
        || ! printf '%s' "$runtimes" | grep -qi nvidia; then
        fail "NVIDIA Container Toolkit/runtime is required — install it before running the worker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
    ok "NVIDIA Container Toolkit/runtime detected"
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
    # Ed25519 requires raw input. OpenSSL before 3.5 does not infer this from
    # the key type (including Ubuntu 24.04's OpenSSL 3.0).
    if ! openssl pkeyutl -verify -rawin -pubin -inkey "$pub" -sigfile "$tmp_sig" -in "$file" >/dev/null 2>&1; then
        rm -f "$pub" "$tmp_sig" "$tmp_pub"
        fail "Agent signature verification failed — refusing install (supply-chain protection)"
    fi
    rm -f "$pub" "$tmp_sig" "$tmp_pub"
    ok "Agent ed25519 signature verified"
}

# Files whose bytes are ed25519-signed by the release pipeline (must match
# scripts/sign-static-agent.py::SIGNABLE and routes/static.py::_SIGNABLE).
# These are imported/executed with full worker-agent privileges, so every
# one of them — not just worker_agent.py itself — must be signature-checked.
SIGNABLE_FILES="worker_agent.py security.py nvml_telemetry.py worker_image_cache.py worker_nfs.py worker_nvme_cache.py worker_luks_volumes.py env_config.py criu_hosts.py egress_isolation.py worker_privileged.py worker-requirements.txt"

_is_signable() {
    local name="$1" f
    for f in $SIGNABLE_FILES; do
        [ "$f" = "$name" ] && return 0
    done
    return 1
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
    if _is_signable "$name"; then
        if [ -z "$sig_b64" ]; then
            rm -f "$headers" "$dest"
            fail "Server did not advertise X-Xcelsior-Agent-Signature for $name — refusing install"
        fi
        verify_agent_signature "$dest" "$sig_b64"
    fi
    rm -f "$headers"
    ok "Downloaded $name (sha256=$actual_sha)"
}

setup_agent_venv() {
    info "Creating Python venv at $VENV_DIR..."
    (cd / && sudo -H -u "$SERVICE_USER" python3 -m venv "$VENV_DIR")
    (cd / && sudo -H -u "$SERVICE_USER" "$VENV_PYTHON" -m pip install --upgrade pip wheel >/dev/null)
    (cd / && sudo -H -u "$SERVICE_USER" "$VENV_PYTHON" -m pip install -r "$INSTALL_DIR/worker-requirements.txt")
    ok "Worker dependencies installed in venv"
}

ensure_service_user() {
    if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
        info "Creating system user '$SERVICE_USER' (the worker runs as it, not as root or as you)..."
        sudo useradd --system --user-group --home-dir /var/lib/xcelsior --no-create-home \
            --shell /usr/sbin/nologin "$SERVICE_USER"
    fi
    if getent group docker >/dev/null 2>&1; then
        sudo usermod -aG docker "$SERVICE_USER"
    fi
    sudo install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$INSTALL_DIR" "$LIB_DIR"
    sudo install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 /var/lib/xcelsior
    # tailscale up/status without root (only when tailscale is installed)
    if command -v tailscale >/dev/null 2>&1; then
        local ts_socket
        ts_socket="$(sed -n 's/^XCELSIOR_TAILSCALE_SOCKET=//p' "$ENV_FILE" 2>/dev/null | tail -n1)"
        sudo tailscale ${ts_socket:+--socket="$ts_socket"} set --operator="$SERVICE_USER" >/dev/null 2>&1 || true
    fi
}

_env_value() {
    sed -n "s/^$1=//p" "$ENV_FILE" 2>/dev/null | tail -n1
}

install_privd() {
    local helper="$1" roots extra
    info "Installing xcelsior-privd (root helper for mounts and encrypted volumes)..."
    sudo install -d -m 0755 "$PRIVD_DIR" /etc/xcelsior
    # Root runs this copy, never the worker-owned one in $LIB_DIR.
    sudo install -o root -g root -m 0644 "$helper" "$PRIVD_DIR/worker_privileged.py"
    if ! sudo test -f "$PRIVD_CONF"; then
        roots="/exports"
        for extra in "$(_env_value XCELSIOR_NFS_PATH)" "$(_env_value XCELSIOR_NFS_EXPORT_BASE)"; do
            case "$extra" in ""|/exports|/exports/*) ;; *) roots="$roots,$extra" ;; esac
        done
        sudo tee "$PRIVD_CONF" >/dev/null <<EOF
# Root-owned settings for xcelsior-privd ($PRIVD_DIR/worker_privileged.py).
# The worker cannot change these. Edit, then nothing to restart: each request reads them.
# NFS servers allowed besides mesh addresses (100.64.0.0/10), comma-separated.
NFS_SERVERS=$(_env_value XCELSIOR_NFS_SERVER)
NFS_EXPORT_ROOTS=$roots
NFS_MOUNT_OPTS=$(_env_value XCELSIOR_NFS_MOUNT_OPTS)
VOLUME_DIR=${XCELSIOR_VOLUME_DIR:-$(_env_value XCELSIOR_VOLUME_DIR)}
VOLUME_KEY_DIR=${XCELSIOR_VOLUME_KEY_DIR:-$(_env_value XCELSIOR_VOLUME_KEY_DIR)}
EOF
        # Empty values fall back to the helper's defaults.
        sudo sed -i -E '/^(NFS_MOUNT_OPTS|VOLUME_DIR|VOLUME_KEY_DIR)=$/d' "$PRIVD_CONF"
    fi

    local unit_tmp
    unit_tmp="$(mktemp)"
    cat > "$unit_tmp" <<EOF
[Unit]
Description=Xcelsior worker privileged helper socket

[Socket]
ListenStream=/run/xcelsior-privd.sock
SocketUser=root
SocketGroup=$SERVICE_USER
SocketMode=0660
Accept=yes
MaxConnections=16

[Install]
WantedBy=sockets.target
EOF
    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/xcelsior-privd.socket
    cat > "$unit_tmp" <<EOF
[Unit]
Description=Xcelsior worker privileged helper (one request)

[Service]
Type=simple
# No mount sandboxing on purpose: mounts must land in the host namespace, where
# docker sees them. Every request is validated by the helper itself.
ExecStart=/usr/bin/python3 $PRIVD_DIR/worker_privileged.py
StandardInput=socket
StandardOutput=socket
StandardError=journal
TimeoutSec=600
EOF
    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/xcelsior-privd@.service
    rm -f "$unit_tmp"
    sudo systemctl daemon-reload
    sudo systemctl enable --now xcelsior-privd.socket >/dev/null
    ok "xcelsior-privd listening on /run/xcelsior-privd.sock (settings: $PRIVD_CONF)"
}

_set_env() {
    # Replace KEY=... in worker.env, or append it.
    local key="$1" value="$2"
    if grep -q "^$key=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s#^$key=.*#$key=$value#" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}

_api_bearer() {
    # The worker's own credential: an API key, or an OAuth client exchanged for a token.
    local token client_id client_secret
    token="$(_env_value XCELSIOR_API_TOKEN)"
    if [ -n "$token" ]; then
        printf '%s' "$token"
        return 0
    fi
    client_id="$(_env_value XCELSIOR_OAUTH_CLIENT_ID)"
    client_secret="$(_env_value XCELSIOR_OAUTH_CLIENT_SECRET)"
    [ -n "$client_id" ] && [ -n "$client_secret" ] || return 1
    curl -fsS -X POST "$XCELSIOR_API/oauth/token" \
        --data-urlencode grant_type=client_credentials \
        --data-urlencode "client_id=$client_id" --data-urlencode "client_secret=$client_secret" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
}

join_mesh() {
    local login_server="$1" auth_key="$2" host_id="$3"
    if ! command -v tailscaled >/dev/null 2>&1; then
        info "Installing Tailscale (used only for the Xcelsior private network)..."
        curl -fsSL https://tailscale.com/install.sh | sh >/dev/null
    fi
    if ! systemctl is-active --quiet tailscaled-xcelsior 2>/dev/null; then
        local unit_tmp
        unit_tmp="$(mktemp)"
        cat > "$unit_tmp" <<EOF
[Unit]
Description=Tailscale node for the Xcelsior private network (separate from any personal tailnet)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$(command -v tailscaled) --state=/var/lib/tailscale-xcelsior/tailscaled.state --socket=$MESH_SOCKET --port=41642 --tun=xcelsior0
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
        sudo install -m 0644 "$unit_tmp" /etc/systemd/system/tailscaled-xcelsior.service
        rm -f "$unit_tmp"
        sudo systemctl daemon-reload
        sudo systemctl enable --now tailscaled-xcelsior >/dev/null
        sleep 2
    fi
    sudo tailscale --socket="$MESH_SOCKET" up --login-server="$login_server" \
        --authkey="$auth_key" --hostname="$host_id" --accept-dns=false >/dev/null
    sudo tailscale --socket="$MESH_SOCKET" set --operator="$SERVICE_USER" >/dev/null 2>&1 || true
    _set_env XCELSIOR_TAILSCALE_SOCKET "$MESH_SOCKET"
    ok "Joined the Xcelsior private network as $(sudo tailscale --socket="$MESH_SOCKET" ip -4 | head -n1)"
}

install_spire_agent() {
    local address="$1" port="$2" trust_domain="$3" token="$4" bundle="$5"
    local arch sha tarball
    case "$(uname -m)" in
        x86_64) arch=amd64; sha="$SPIRE_SHA256_AMD64" ;;
        aarch64|arm64) arch=arm64; sha="$SPIRE_SHA256_ARM64" ;;
        *) fail "No SPIRE agent build for $(uname -m)" ;;
    esac
    if [ ! -x /opt/spire/bin/spire-agent ] || ! /opt/spire/bin/spire-agent --version 2>&1 | grep -q "$SPIRE_VERSION"; then
        info "Installing spire-agent $SPIRE_VERSION..."
        tarball="$(mktemp)"
        curl -fsSL -o "$tarball" \
            "https://github.com/spiffe/spire/releases/download/v$SPIRE_VERSION/spire-$SPIRE_VERSION-linux-$arch-musl.tar.gz"
        echo "$sha  $tarball" | sha256sum -c --quiet - || { rm -f "$tarball"; fail "spire-agent checksum mismatch"; }
        sudo install -d -m 0755 /opt/spire/bin
        tar -xzf "$tarball" -O "spire-$SPIRE_VERSION/bin/spire-agent" | sudo tee /opt/spire/bin/spire-agent >/dev/null
        sudo chmod 0755 /opt/spire/bin/spire-agent
        rm -f "$tarball"
    fi
    sudo install -d -m 0755 /etc/spire/agent
    sudo install -d -m 0700 /var/lib/spire/agent
    sudo install -m 0644 "$bundle" /etc/spire/agent/bootstrap.crt
    sudo tee /etc/spire/agent/agent.conf >/dev/null <<EOF
agent {
  data_dir          = "/var/lib/spire/agent"
  log_level         = "WARN"
  server_address    = "$address"
  server_port       = "$port"
  socket_path       = "$SPIRE_SOCKET"
  trust_domain      = "$trust_domain"
  trust_bundle_path = "/etc/spire/agent/bootstrap.crt"
  # Single use: after the first attestation the agent renews its own identity.
  join_token        = "$token"
}
plugins {
  NodeAttestor "join_token" { plugin_data {} }
  # On disk, so a restart renews instead of re-attesting with the used token.
  KeyManager "disk" { plugin_data { directory = "/var/lib/spire/agent" } }
  WorkloadAttestor "unix" { plugin_data {} }
}
EOF
    sudo chmod 0600 /etc/spire/agent/agent.conf
    local unit_tmp
    unit_tmp="$(mktemp)"
    cat > "$unit_tmp" <<EOF
[Unit]
Description=SPIRE agent (Xcelsior host identity)
After=network-online.target tailscaled-xcelsior.service
Wants=network-online.target

[Service]
ExecStart=/opt/spire/bin/spire-agent run -config /etc/spire/agent/agent.conf
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/spire-agent.service
    # Refreshes the worker's certificate well inside its 1 h lifetime.
    cat > "$unit_tmp" <<EOF
[Unit]
Description=Fetch the Xcelsior worker's X.509-SVID from the SPIRE agent
After=spire-agent.service
Requires=spire-agent.service

[Service]
Type=oneshot
User=$SERVICE_USER
Group=$SERVICE_USER
RuntimeDirectory=xcelsior-svid
RuntimeDirectoryPreserve=yes
ExecStart=/bin/sh -c 'mkdir -p $SVID_DIR/.next; for i in 1 2 3 4 5 6 7 8 9 10 11 12; do /opt/spire/bin/spire-agent api fetch x509 -socketPath $SPIRE_SOCKET -write $SVID_DIR/.next >/dev/null 2>&1 && mv -f $SVID_DIR/.next/svid.0.key $SVID_DIR/svid.0.key && mv -f $SVID_DIR/.next/svid.0.pem $SVID_DIR/svid.0.pem && exit 0; sleep 5; done; exit 1'
EOF
    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/xcelsior-svid.service
    cat > "$unit_tmp" <<EOF
[Unit]
Description=Refresh the Xcelsior worker's X.509-SVID

[Timer]
OnBootSec=1min
OnUnitActiveSec=20min

[Install]
WantedBy=timers.target
EOF
    sudo install -m 0644 "$unit_tmp" /etc/systemd/system/xcelsior-svid.timer
    rm -f "$unit_tmp"
    sudo systemctl daemon-reload
    sudo systemctl enable --now spire-agent >/dev/null
    sudo systemctl enable --now xcelsior-svid.timer >/dev/null
    sudo systemctl start xcelsior-svid.service || fail "No identity from the SPIRE agent (journalctl -u spire-agent -u xcelsior-svid)"
    ok "Host identity issued: $(openssl x509 -in "$SVID_DIR/svid.0.pem" -noout -subject 2>/dev/null)"
}

enroll_host() {
    local host_id scheduler bearer reply code
    host_id="$(_env_value XCELSIOR_HOST_ID)"
    scheduler="$(_env_value XCELSIOR_SCHEDULER_URL)"
    if [ -n "$(_env_value XCELSIOR_AGENT_CLIENT_CERT)" ] && [ "$scheduler" != "$XCELSIOR_API" ]; then
        # A hand-issued certificate, or a SPIRE identity that still renews: nothing to do.
        if [ "$(_env_value XCELSIOR_AGENT_CLIENT_CERT)" != "$SVID_DIR/svid.0.pem" ] \
            || sudo systemctl start xcelsior-svid.service >/dev/null 2>&1; then
            ok "Host already has a gateway identity ($scheduler)"
            return 0
        fi
        # Offline past the agent's 30-day identity: the join token is spent, so enroll again.
        warn "This host's identity can no longer be renewed (offline over 30 days?); enrolling again"
        sudo systemctl stop spire-agent >/dev/null 2>&1 || true
        sudo rm -rf /var/lib/spire/agent
    fi
    [ -n "$host_id" ] || fail "XCELSIOR_HOST_ID is missing from $ENV_FILE"
    info "Connecting $host_id to Xcelsior (private network + host identity)..."
    bearer="$(_api_bearer)" || fail "No API key or OAuth client in $ENV_FILE"
    reply="$(mktemp)"
    chmod 600 "$reply"
    code=$(curl -sS -o "$reply" -w '%{http_code}' -X POST -H "Authorization: Bearer $bearer" \
        "$XCELSIOR_API/api/hosts/$host_id/enrollment" || true)
    case "$code" in
        200) ;;
        404) rm -f "$reply"; fail "Host $host_id is not registered to this account (Dashboard -> Hosts)" ;;
        429) rm -f "$reply"; fail "Too many enrollment attempts for $host_id; try again in an hour" ;;
        503) rm -f "$reply"; fail "Xcelsior cannot connect new hosts automatically right now; contact support" ;;
        *) rm -f "$reply"; fail "Enrollment failed (HTTP ${code:-none}); retry with: bash install.sh --update" ;;
    esac
    local fields
    fields="$(mktemp -d)"
    python3 - "$reply" "$fields" <<'PY'
import json, os, sys
r = json.load(open(sys.argv[1]))
out = sys.argv[2]
values = {
    "login_server": r["mesh"]["login_server"], "auth_key": r["mesh"]["auth_key"],
    "address": r["spire"]["server_address"], "port": str(r["spire"]["server_port"]),
    "trust_domain": r["spire"]["trust_domain"], "token": r["spire"]["join_token"],
    "bundle.pem": r["spire"]["trust_bundle_pem"], "scheduler_url": r["scheduler_url"],
}
for name, value in values.items():
    fd = os.open(os.path.join(out, name), os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(value)
PY
    rm -f "$reply"
    join_mesh "$(cat "$fields/login_server")" "$(cat "$fields/auth_key")" "$host_id"
    install_spire_agent "$(cat "$fields/address")" "$(cat "$fields/port")" \
        "$(cat "$fields/trust_domain")" "$(cat "$fields/token")" "$fields/bundle.pem"
    _set_env XCELSIOR_SCHEDULER_URL "$(cat "$fields/scheduler_url")"
    _set_env XCELSIOR_AGENT_V2_PATHS 1
    _set_env XCELSIOR_AGENT_CLIENT_CERT "$SVID_DIR/svid.0.pem"
    _set_env XCELSIOR_AGENT_CLIENT_KEY "$SVID_DIR/svid.0.key"
    local mesh_ip
    mesh_ip="$(sudo tailscale --socket="$MESH_SOCKET" ip -4 2>/dev/null | head -n1 || true)"
    [ -n "$mesh_ip" ] && _set_env XCELSIOR_HOST_IP "$mesh_ip"
    rm -rf "$fields"
    ok "Enrolled: worker traffic goes to $(_env_value XCELSIOR_SCHEDULER_URL) with this host's own identity"
}

migrate_agent_tls() {
    # The worker can no longer read your home directory: move mTLS client files
    # it was pointed at into /etc/xcelsior/agent-tls (readable by root and the service group).
    local var path dest
    for var in XCELSIOR_AGENT_CLIENT_CERT XCELSIOR_AGENT_CLIENT_KEY; do
        path="$(_env_value "$var")"
        case "$path" in ""|"$AGENT_TLS_DIR"/*) continue ;; esac
        if [ ! -f "$path" ]; then
            warn "$var points at $path, which does not exist"
            continue
        fi
        dest="$AGENT_TLS_DIR/$(basename "$path")"
        sudo install -d -o root -g "$SERVICE_USER" -m 0750 "$AGENT_TLS_DIR"
        sudo install -o root -g "$SERVICE_USER" -m 0640 "$path" "$dest"
        sed -i "s#^$var=.*#$var=$dest#" "$ENV_FILE"
        ok "$var: copied to $dest"
    done
}

# ── Agent install (B1) ───────────────────────────────────────────────

install_agent() {
    mkdir -p "$CONFIG_DIR"
    ensure_service_user

    local stage
    stage="$(mktemp -d)"
    mkdir -p "$stage/lib"
    info "Downloading worker agent and dependencies..."
    download_static_file "worker_agent.py" "$stage/worker_agent.py"
    download_static_file "security.py" "$stage/lib/security.py"
    download_static_file "nvml_telemetry.py" "$stage/lib/nvml_telemetry.py"
    download_static_file "worker_image_cache.py" "$stage/lib/worker_image_cache.py"
    download_static_file "worker_nfs.py" "$stage/lib/worker_nfs.py"
    download_static_file "worker_nvme_cache.py" "$stage/lib/worker_nvme_cache.py"
    download_static_file "worker_luks_volumes.py" "$stage/lib/worker_luks_volumes.py"
    download_static_file "env_config.py" "$stage/lib/env_config.py"
    download_static_file "criu_hosts.py" "$stage/lib/criu_hosts.py"
    download_static_file "egress_isolation.py" "$stage/lib/egress_isolation.py"
    download_static_file "worker_privileged.py" "$stage/lib/worker_privileged.py"
    download_static_file "worker-requirements.txt" "$stage/worker-requirements.txt"
    sudo install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$stage/worker_agent.py" "$AGENT_PATH"
    sudo install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0644 "$stage"/lib/*.py "$LIB_DIR/"
    sudo install -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0644 "$stage/worker-requirements.txt" "$INSTALL_DIR/"

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

    migrate_agent_tls
    install_privd "$stage/lib/worker_privileged.py"
    rm -rf "$stage"
    enroll_host
    if [ -f "$CONFIG_DIR/worker_agent.py" ]; then
        info "The agent now runs from $INSTALL_DIR; the old copy in $CONFIG_DIR (worker_agent.py, lib/, venv/) is unused and can be deleted."
    fi
}

# ── Network protection for rented containers ─────────────────────────
# Rented containers reach the internet only through an allowlisting proxy on
# this machine (egress_isolation.py). Root runs the provisioner from a
# root-owned copy; the worker only reads the status file it writes, and gets
# no rentals while that status is missing or stale.

prepare_egress_proxy_access() {
    # Existing hosts may keep credentials under a private /etc/xcelsior.
    # Give only the proxy user traversal, without making that directory public.
    sudo mkdir -p /etc/xcelsior
    if ! sudo -u proxy test -x /etc/xcelsior; then
        if ! command -v setfacl >/dev/null 2>&1; then
            sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq acl >/dev/null \
                || fail "Could not install acl for the proxy's config access"
        fi
        sudo setfacl -m u:proxy:--x /etc/xcelsior
    fi
    # Ubuntu's Squid AppArmor profile permits only the packaged config/runtime
    # paths. Keep confinement enabled and grant these three exact files.
    if [ -f /etc/apparmor.d/usr.sbin.squid ]; then
        local local_profile=/etc/apparmor.d/local/usr.sbin.squid
        sudo install -d -m 0755 /etc/apparmor.d/local
        if ! sudo grep -qF '# Xcelsior tenant proxy' "$local_profile" 2>/dev/null; then
            sudo tee -a "$local_profile" >/dev/null <<'EOF'

# Xcelsior tenant proxy
/etc/xcelsior/egress/squid.conf r,
/etc/xcelsior/egress/domains.txt r,
/run/xcelsior-egress-proxy/squid.pid rwk,
EOF
        fi
        if command -v aa-status >/dev/null 2>&1 && sudo aa-status --enabled; then
            sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.squid
        fi
    fi
}

install_egress_isolation() {
    info "Setting up network protection for rented containers..."
    if ! command -v squid >/dev/null 2>&1; then
        sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq squid >/dev/null 2>&1 \
            || { sudo apt-get update -qq >/dev/null 2>&1 \
                 && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq squid >/dev/null; } \
            || fail "Could not install squid, which protects rented containers"
    fi
    # The packaged service listens on every interface; ours binds only the
    # rented containers' gateway.
    sudo systemctl disable --now squid.service >/dev/null 2>&1 || true
    sudo systemctl mask squid.service >/dev/null 2>&1 || true
    prepare_egress_proxy_access

    sudo install -d -m 0755 -o root -g root /usr/local/lib/xcelsior
    sudo install -m 0644 -o root -g root "$LIB_DIR/egress_isolation.py" \
        /usr/local/lib/xcelsior/egress_isolation.py

    local unit_tmp
    unit_tmp="$(mktemp -d)"
    cat > "$unit_tmp/xcelsior-egress.service" <<'EOF'
[Unit]
Description=Xcelsior network protection for rented containers
# The proxy starts first so the check finds it listening (the check still
# waits briefly for it); on a first install it binds once this creates the network.
After=docker.service xcelsior-egress-proxy.service
Wants=xcelsior-egress-proxy.service
Requires=docker.service
PartOf=docker.service

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /usr/local/lib/xcelsior/egress_isolation.py provision

[Install]
WantedBy=multi-user.target docker.service
EOF
    cat > "$unit_tmp/xcelsior-egress.timer" <<'EOF'
[Unit]
Description=Re-check Xcelsior network protection for rented containers

[Timer]
OnBootSec=30s
OnUnitInactiveSec=2min
AccuracySec=10s

[Install]
WantedBy=timers.target
EOF
    cat > "$unit_tmp/xcelsior-egress-proxy.service" <<'EOF'
[Unit]
Description=Xcelsior proxy for rented containers (allowlisted downloads only)
After=docker.service

[Service]
User=proxy
Group=proxy
RuntimeDirectory=xcelsior-egress-proxy
ExecStart=/usr/sbin/squid -N -d 1 -f /etc/xcelsior/egress/squid.conf
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
NoNewPrivileges=yes
CapabilityBoundingSet=
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
ReadWritePaths=/var/spool/squid

[Install]
WantedBy=multi-user.target
EOF
    local unit
    for unit in xcelsior-egress.service xcelsior-egress.timer xcelsior-egress-proxy.service; do
        sudo install -m 0644 "$unit_tmp/$unit" "/etc/systemd/system/$unit"
    done
    rm -rf "$unit_tmp"
    sudo systemctl daemon-reload
    sudo systemctl enable xcelsior-egress.service xcelsior-egress.timer xcelsior-egress-proxy.service >/dev/null 2>&1

    # The first run creates the network the proxy binds to; the proxy check
    # passes once squid is listening.
    sudo systemctl start xcelsior-egress.service >/dev/null 2>&1 || true
    sudo systemctl restart xcelsior-egress-proxy.service
    sudo systemctl start xcelsior-egress.timer
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if sudo systemctl start xcelsior-egress.service >/dev/null 2>&1; then
            ok "Network protection for rented containers is active"
            return 0
        fi
        sleep 2
    done
    warn "Network protection is not ready, so this machine will not receive rentals yet."
    warn "Details: sudo python3 /usr/local/lib/xcelsior/egress_isolation.py verify"
}

install_systemd() {
    install_egress_isolation

    info "Installing systemd service..."

    local unit_tmp groups=""
    unit_tmp="$(mktemp)"
    if getent group docker >/dev/null 2>&1; then
        groups="SupplementaryGroups=docker"
    fi
    cat > "$unit_tmp" <<EOF
[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service xcelsior-privd.socket xcelsior-svid.service
Wants=network-online.target xcelsior-privd.socket xcelsior-svid.timer
Requires=docker.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
$groups
# Read by systemd itself (as root) before the sandbox below applies.
EnvironmentFile=$ENV_FILE
Environment=PYTHONPATH=$LIB_DIR
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON $AGENT_PATH
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
# systemd creates /var/lib/xcelsior (owned by the service user) and passes it to the
# worker as \$STATE_DIRECTORY. Listing a missing path in ReadWritePaths instead made
# the first start fail with 226/NAMESPACE on a fresh host.
StateDirectory=xcelsior
StateDirectoryMode=0750
# /run/xcelsior/gpu.lock serialises exclusive-GPU jobs across restarts.
RuntimeDirectory=xcelsior
RuntimeDirectoryPreserve=yes
# Self-update replaces the agent in place.
ReadWritePaths=$INSTALL_DIR
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
    # The public /host/<id> surface is retired (410); ask the product API, as the owner.
    local host_id bearer started i row last_seen status admitted
    host_id="$(_env_value XCELSIOR_HOST_ID)"
    [ -z "$host_id" ] && return 0
    bearer="$(_api_bearer)" || return 0
    started=$(date +%s)
    info "Waiting for the first heartbeat from $host_id (up to 90s)..."
    for i in $(seq 1 18); do
        row=$(curl -fsS --max-time 5 -H "Authorization: Bearer $bearer" \
            "$XCELSIOR_API/hosts?active_only=false" 2>/dev/null | python3 -c '
import json, sys
host = next((h for h in json.load(sys.stdin).get("hosts", []) if h.get("host_id") == sys.argv[1]), {})
print(int(float(host.get("last_seen") or 0)), host.get("status", ""), str(host.get("admitted") is True).lower(), sep="|")
' "$host_id" 2>/dev/null || true)
        last_seen="${row%%|*}"
        status="$(printf '%s' "$row" | cut -d'|' -f2)"
        admitted="${row##*|}"
        if [ -n "$last_seen" ] && [ "$last_seen" -ge "$started" ] 2>/dev/null; then
            if [ "$admitted" = "true" ]; then
                ok "Host $host_id is connected, admitted and $status"
            else
                ok "Host $host_id is connected (heartbeat received)"
                info "It is pending admission review and not eligible for work yet; the dashboard shows its status"
            fi
            return 0
        fi
        sleep 5
    done
    warn "No heartbeat from $host_id yet. Check: journalctl -u xcelsior-worker -n 50 --no-pager"
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
        if npx --yes "$WIZARD_PACKAGE"; then
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
  bash install.sh [--uninstall]       Remove systemd units (keeps ~/.xcelsior, /opt/xcelsior)
  bash install.sh [--help | -h]       Show this help

Environment:
  XCELSIOR_API_URL   API base URL (default: https://xcelsior.ca)
  XCELSIOR_STAGE_HOST_TOOLS=1  Stage host binaries (not recommended)
EOF
}

do_uninstall() {
    info "Stopping and removing Xcelsior systemd units..."
    sudo systemctl stop xcelsior-worker xcelsior-registry-login xcelsior-privd.socket 2>/dev/null || true
    sudo systemctl disable xcelsior-worker xcelsior-registry-login xcelsior-privd.socket 2>/dev/null || true
    sudo rm -f /etc/systemd/system/xcelsior-worker.service \
        /etc/systemd/system/xcelsior-registry-login.service \
        /etc/systemd/system/xcelsior-privd.socket \
        /etc/systemd/system/xcelsior-privd@.service \
        /usr/local/bin/xcelsior-registry-login.sh
    sudo rm -rf "$PRIVD_DIR"
    sudo systemctl daemon-reload
    ok "Systemd units removed (config kept at $CONFIG_DIR, agent at $INSTALL_DIR, user '$SERVICE_USER' kept)"
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
    ok "Local setup complete. Check the dashboard for host admission status."
    info "Dashboard: $XCELSIOR_API/dashboard/hosts"
    echo ""
}

# Run main when executed (bash install.sh) or piped (curl ... | bash, where
# BASH_SOURCE[0] is empty), but not when sourced for its functions (the wizard).
if [[ "${BASH_SOURCE[0]:-$0}" == "$0" ]]; then
    main "$@"
fi
