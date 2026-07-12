#!/usr/bin/env bash
# Xcelsior Setup Script v1.0.0
# Automated installation and configuration for the Xcelsior GPU scheduler.
# Run with: chmod +x setup.sh && ./setup.sh

set -e  # Exit on error

# ── Colors & Formatting ───────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helper Functions ──────────────────────────────────────────────────

print_header() {
    echo -e "${CYAN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    XCELSIOR SETUP v1.0.0                      ║"
    echo "║          Distributed GPU Scheduler for Canadians              ║"
    echo "║                     Ever Upward.                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${BLUE}${BOLD}━━━ $1 ━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

ask_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local response

    if [ "$default" = "y" ]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi

    while true; do
        read -p "$(echo -e ${YELLOW}${prompt}${NC})" response
        response=${response:-$default}
        case "$response" in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

ask_input() {
    local prompt="$1"
    local default="$2"
    local response

    if [ -n "$default" ]; then
        prompt="$prompt [$default]: "
    else
        prompt="$prompt: "
    fi

    read -p "$(echo -e ${YELLOW}${prompt}${NC})" response
    echo "${response:-$default}"
}

write_worker_env_file() {
    local host_id="$1"
    local scheduler_url="$2"
    local cost_per_hour="$3"
    local api_token="${4:-}"
    local oauth_client_id="${5:-}"
    local oauth_client_secret="${6:-}"
    local host_ssh_user="${7:-}"

    echo "export XCELSIOR_HOST_ID=$host_id" > worker_env.sh
    echo "export XCELSIOR_SCHEDULER_URL=$scheduler_url" >> worker_env.sh
    echo "export XCELSIOR_COST_PER_HOUR=$cost_per_hour" >> worker_env.sh
    echo "export XCELSIOR_REPORT_INTERVAL=5" >> worker_env.sh
    if [ -n "$api_token" ]; then
        echo "export XCELSIOR_API_TOKEN=$api_token" >> worker_env.sh
    fi
    if [ -n "$oauth_client_id" ] && [ -n "$oauth_client_secret" ]; then
        echo "export XCELSIOR_OAUTH_CLIENT_ID=$oauth_client_id" >> worker_env.sh
        echo "export XCELSIOR_OAUTH_CLIENT_SECRET=$oauth_client_secret" >> worker_env.sh
    fi
    # SSH user the API will connect as for Docker-over-SSH (web terminal).
    # Must match the account whose authorized_keys contains the platform pubkey.
    if [ -n "$host_ssh_user" ]; then
        echo "export XCELSIOR_HOST_SSH_USER=$host_ssh_user" >> worker_env.sh
    fi
}

# ── System Detection ──────────────────────────────────────────────────

detect_os() {
    print_section "Detecting Operating System"

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
        print_success "Detected: $PRETTY_NAME"
    else
        print_error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi

    # Check if OS is supported
    case "$OS" in
        ubuntu|debian|centos|rhel|fedora)
            print_success "OS is supported."
            ;;
        *)
            print_warning "OS may not be fully supported. Proceeding anyway..."
            ;;
    esac
}

detect_python() {
    print_section "Detecting Python"

    # Check for Python 3.11+
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        PYTHON_VERSION=$(python3.11 --version | cut -d' ' -f2)
        print_success "Found Python $PYTHON_VERSION at $(which python3.11)"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            PYTHON_CMD="python3"
            print_success "Found Python $PYTHON_VERSION at $(which python3)"
        else
            print_error "Python 3.11+ required. Found Python $PYTHON_VERSION"
            if ask_yes_no "Would you like to install Python 3.11?" "n"; then
                install_python
            else
                exit 1
            fi
        fi
    else
        print_error "Python not found."
        if ask_yes_no "Would you like to install Python 3.11?" "y"; then
            install_python
        else
            exit 1
        fi
    fi
}

install_python() {
    print_section "Installing Python 3.11"

    case "$OS" in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
            PYTHON_CMD="python3.11"
            ;;
        centos|rhel|fedora)
            sudo yum install -y python3.11 python3.11-devel
            PYTHON_CMD="python3.11"
            ;;
        *)
            print_error "Automatic Python installation not supported for $OS"
            print_info "Please install Python 3.11+ manually and run this script again."
            exit 1
            ;;
    esac

    print_success "Python 3.11 installed successfully."
}

detect_docker() {
    print_section "Detecting Docker"

    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | sed 's/,//')
        print_success "Found Docker $DOCKER_VERSION at $(which docker)"

        # Check if user can run docker without sudo
        if docker ps &> /dev/null; then
            print_success "Docker is accessible without sudo."
            DOCKER_INSTALLED=true
        else
            print_warning "Docker requires sudo. User may need to be added to docker group."
            DOCKER_INSTALLED=true
        fi
    else
        print_warning "Docker not found."
        DOCKER_INSTALLED=false
    fi
}

install_docker() {
    print_section "Installing Docker"

    case "$OS" in
        ubuntu|debian)
            curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
            sudo sh /tmp/get-docker.sh
            sudo usermod -aG docker $USER
            rm /tmp/get-docker.sh
            ;;
        centos|rhel|fedora)
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install -y docker-ce docker-ce-cli containerd.io
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        *)
            print_error "Automatic Docker installation not supported for $OS"
            print_info "Please install Docker manually: https://docs.docker.com/engine/install/"
            return 1
            ;;
    esac

    print_success "Docker installed successfully."
    print_warning "You may need to log out and back in for docker group changes to take effect."
    DOCKER_INSTALLED=true
}

detect_nvidia() {
    print_section "Detecting NVIDIA GPU"

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
        print_success "Found GPU: $GPU_INFO"
        NVIDIA_INSTALLED=true

        # Parse GPU info
        GPU_MODEL=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
        GPU_VRAM=$(echo $GPU_INFO | cut -d',' -f2 | xargs | sed 's/ MiB//' | awk '{print $1/1024}')

        print_info "GPU Model: $GPU_MODEL"
        print_info "Total VRAM: ${GPU_VRAM} GB"
    else
        print_warning "nvidia-smi not found. NVIDIA drivers may not be installed."
        NVIDIA_INSTALLED=false
    fi
}

detect_nvidia_docker() {
    print_section "Detecting NVIDIA Container Toolkit"

    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        print_success "NVIDIA Container Toolkit is working."
        NVIDIA_DOCKER_INSTALLED=true
    else
        print_warning "NVIDIA Container Toolkit not found or not working."
        NVIDIA_DOCKER_INSTALLED=false
    fi
}

install_nvidia_docker() {
    print_section "Installing NVIDIA Container Toolkit"

    case "$OS" in
        ubuntu|debian)
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-docker.list
            sudo apt-get update
            sudo apt-get install -y nvidia-container-toolkit
            sudo systemctl restart docker
            ;;
        centos|rhel|fedora)
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
                sudo tee /etc/yum.repos.d/nvidia-docker.repo
            sudo yum install -y nvidia-container-toolkit
            sudo systemctl restart docker
            ;;
        *)
            print_error "Automatic NVIDIA Container Toolkit installation not supported for $OS"
            print_info "Please install manually: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            return 1
            ;;
    esac

    print_success "NVIDIA Container Toolkit installed successfully."
    NVIDIA_DOCKER_INSTALLED=true
}

# ── Installation ──────────────────────────────────────────────────────

create_venv() {
    print_section "Creating Python Virtual Environment"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists."
        if ask_yes_no "Do you want to recreate it?" "n"; then
            rm -rf venv
        else
            print_info "Using existing virtual environment."
            return 0
        fi
    fi

    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created."
}

install_dependencies() {
    print_section "Installing Python Dependencies"

    source venv/bin/activate

    print_info "Upgrading pip..."
    pip install --quiet --upgrade pip

    print_info "Installing requirements..."
    pip install --quiet -r requirements.txt

    print_success "All Python dependencies installed."
}

configure_environment() {
    print_section "Configuring Environment"

    if [ -f ".env" ]; then
        print_warning ".env file already exists."
        if ask_yes_no "Do you want to reconfigure?" "n"; then
            rm .env
        else
            print_info "Keeping existing .env file."
            return 0
        fi
    fi

    cp .env.example .env
    print_success ".env file created from template."

    # Generate local control-plane token
    print_info "Generating local API token..."
    API_TOKEN=$($PYTHON_CMD -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s/^XCELSIOR_API_TOKEN=.*/XCELSIOR_API_TOKEN=$API_TOKEN/" .env
    print_success "Local API token generated and saved."

    echo -e "\n${GREEN}${BOLD}Your Local API Token:${NC} ${CYAN}$API_TOKEN${NC}"
    echo -e "${YELLOW}Save this token. It secures the local API and workers can use it directly. OAuth client credentials are also supported once the API is running.${NC}\n"

    # Ask for optional configuration
    if ask_yes_no "Would you like to configure email alerts?" "n"; then
        configure_email_alerts
    fi

    if ask_yes_no "Would you like to configure Telegram alerts?" "n"; then
        configure_telegram_alerts
    fi
}

configure_email_alerts() {
    echo ""
    print_info "Email Alert Configuration"

    SMTP_HOST=$(ask_input "SMTP Host" "smtp.gmail.com")
    SMTP_PORT=$(ask_input "SMTP Port" "587")
    SMTP_USER=$(ask_input "SMTP Username")
    SMTP_PASS=$(ask_input "SMTP Password")
    EMAIL_FROM=$(ask_input "From Email" "$SMTP_USER")
    EMAIL_TO=$(ask_input "To Email" "$SMTP_USER")

    sed -i "s/^XCELSIOR_SMTP_HOST=.*/XCELSIOR_SMTP_HOST=$SMTP_HOST/" .env
    sed -i "s/^XCELSIOR_SMTP_PORT=.*/XCELSIOR_SMTP_PORT=$SMTP_PORT/" .env
    sed -i "s/^XCELSIOR_SMTP_USER=.*/XCELSIOR_SMTP_USER=$SMTP_USER/" .env
    sed -i "s/^XCELSIOR_SMTP_PASS=.*/XCELSIOR_SMTP_PASS=$SMTP_PASS/" .env
    sed -i "s/^XCELSIOR_EMAIL_FROM=.*/XCELSIOR_EMAIL_FROM=$EMAIL_FROM/" .env
    sed -i "s/^XCELSIOR_EMAIL_TO=.*/XCELSIOR_EMAIL_TO=$EMAIL_TO/" .env

    print_success "Email alerts configured."
}

configure_telegram_alerts() {
    echo ""
    print_info "Telegram Alert Configuration"
    print_info "Create a bot at https://t.me/BotFather to get a token."

    TG_TOKEN=$(ask_input "Telegram Bot Token")
    TG_CHAT_ID=$(ask_input "Telegram Chat ID")

    sed -i "s/^XCELSIOR_TG_TOKEN=.*/XCELSIOR_TG_TOKEN=$TG_TOKEN/" .env
    sed -i "s/^XCELSIOR_TG_CHAT_ID=.*/XCELSIOR_TG_CHAT_ID=$TG_CHAT_ID/" .env

    print_success "Telegram alerts configured."
}

# ── SSH Key Setup ─────────────────────────────────────────────────────

detect_host_ssh_user() {
    # Determine the correct SSH user for Docker-over-SSH access.
    # Prefer SUDO_USER (the real user behind sudo), fall back to current user.
    # Never use root — the platform SSH key should be authorized under the
    # provider's regular account.
    if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
        HOST_SSH_USER="$SUDO_USER"
    elif [ "$(whoami)" != "root" ]; then
        HOST_SSH_USER="$(whoami)"
    else
        HOST_SSH_USER="${USER:-root}"
    fi
    # Resolve the SSH home directory for this user
    HOST_SSH_HOME=$(getent passwd "$HOST_SSH_USER" 2>/dev/null | cut -d: -f6)
    if [ -z "$HOST_SSH_HOME" ]; then
        HOST_SSH_HOME="$HOME"
    fi
}

setup_platform_ssh_keypair() {
    # Generate the platform Ed25519 keypair used by the API container to SSH
    # into GPU hosts for Docker-over-SSH (web terminal). Idempotent — skips
    # if the key already exists.
    print_section "Platform SSH Key Setup"

    local key_path="${HOME}/.ssh/xcelsior"

    if [ -f "$key_path" ]; then
        print_success "Platform SSH keypair already exists: $key_path"
    else
        print_info "Generating platform SSH keypair..."
        mkdir -p "${HOME}/.ssh"
        chmod 700 "${HOME}/.ssh"
        ssh-keygen -t ed25519 -f "$key_path" -N "" -C "xcelsior" -q
        chmod 600 "$key_path"
        print_success "Platform SSH keypair generated: $key_path"
    fi

    local pub_key
    pub_key=$(cat "${key_path}.pub" 2>/dev/null || true)
    if [ -z "$pub_key" ]; then
        print_error "Could not read public key at ${key_path}.pub"
        return 1
    fi

    echo -e "\n${CYAN}Platform public key (for provider hosts):${NC}"
    echo -e "  ${YELLOW}${pub_key}${NC}\n"
}

authorize_platform_ssh_key() {
    # Add the platform SSH public key to the current host's authorized_keys
    # so the API container can SSH in for Docker-over-SSH (web terminal).
    # Uses the detected HOST_SSH_USER's home directory.
    detect_host_ssh_user

    local key_path="${HOME}/.ssh/xcelsior.pub"
    if [ ! -f "$key_path" ]; then
        print_warning "Platform public key not found at $key_path — skipping authorized_keys setup."
        print_info "Run the scheduler setup first to generate the keypair, or use the provider wizard."
        return 0
    fi

    local pub_key
    pub_key=$(cat "$key_path")

    local ssh_dir="${HOST_SSH_HOME}/.ssh"
    local auth_keys="${ssh_dir}/authorized_keys"

    # Ensure .ssh directory exists with correct permissions
    if [ ! -d "$ssh_dir" ]; then
        mkdir -p "$ssh_dir"
        chmod 700 "$ssh_dir"
        # If running as root for another user, fix ownership
        if [ "$(whoami)" = "root" ] && [ "$HOST_SSH_USER" != "root" ]; then
            chown "$HOST_SSH_USER:$HOST_SSH_USER" "$ssh_dir"
        fi
    fi

    # Check if key is already authorized
    if [ -f "$auth_keys" ] && grep -qF "$pub_key" "$auth_keys"; then
        print_success "Platform SSH key already authorized for user '$HOST_SSH_USER'"
    else
        echo "$pub_key" >> "$auth_keys"
        chmod 600 "$auth_keys"
        # Fix ownership if needed
        if [ "$(whoami)" = "root" ] && [ "$HOST_SSH_USER" != "root" ]; then
            chown "$HOST_SSH_USER:$HOST_SSH_USER" "$auth_keys"
        fi
        print_success "Platform SSH key authorized for user '$HOST_SSH_USER'"
    fi

    print_info "Web terminal will connect as: $HOST_SSH_USER"
}

# ── Tailscale Mesh Networking ─────────────────────────────────────────

install_tailscale() {
    # Install Tailscale using the official installer. Supports Debian/Ubuntu,
    # Fedora/RHEL, Arch, and other distros via https://tailscale.com/install.sh.
    print_section "Installing Tailscale"

    if command -v tailscale &>/dev/null; then
        print_success "Tailscale is already installed: $(tailscale version 2>/dev/null | head -1)"
        return 0
    fi

    print_info "Installing Tailscale via official installer..."
    if curl -fsSL https://tailscale.com/install.sh | sh; then
        print_success "Tailscale installed successfully"
    else
        print_error "Tailscale installation failed"
        print_info "You can install it manually: https://tailscale.com/download"
        return 1
    fi
}

setup_tailscale_networking() {
    # Offer to install and enable Tailscale for mesh networking.
    # Mesh networking lets remote GPU hosts be reachable from the API
    # container without port forwarding or public IPs.

    local ts_installed=false
    if command -v tailscale &>/dev/null; then
        ts_installed=true
    fi

    if [ "$ts_installed" = "false" ]; then
        echo ""
        print_info "Tailscale mesh networking allows the API to reach GPU hosts"
        print_info "behind NAT/firewalls without port forwarding."
        if ask_yes_no "Install Tailscale for mesh networking? (recommended)" "y"; then
            install_tailscale
            ts_installed=true
        else
            print_warning "Skipping Tailscale — GPU hosts must have a publicly reachable SSH port."
            return 0
        fi
    fi

    if [ "$ts_installed" = "true" ]; then
        # Check if already connected
        if tailscale status &>/dev/null 2>&1; then
            local ts_ip
            ts_ip=$(tailscale ip -4 2>/dev/null || true)
            if [ -n "$ts_ip" ]; then
                print_success "Tailscale mesh active — IP: $ts_ip"
                TAILSCALE_IP="$ts_ip"
                return 0
            fi
        fi

        # Start tailscaled if not running
        if ! systemctl is-active tailscaled &>/dev/null 2>&1; then
            print_info "Starting tailscaled service..."
            sudo systemctl enable tailscaled 2>/dev/null || true
            sudo systemctl start tailscaled 2>/dev/null || true
        fi

        echo ""
        print_info "Tailscale needs to be connected to your tailnet."
        print_info "If you use Headscale, provide the login server URL."
        echo ""

        local login_server
        login_server=$(ask_input "Headscale login server URL (leave empty for Tailscale SaaS)" "")

        local up_cmd="sudo tailscale up --hostname ${DEFAULT_HOST_ID:-xcelsior-worker}"
        if [ -n "$login_server" ]; then
            up_cmd="$up_cmd --login-server $login_server"
        fi

        print_info "Running: $up_cmd"
        print_info "Follow the authentication URL if prompted..."
        echo ""
        eval "$up_cmd" || {
            print_warning "Tailscale login may require manual auth — check the URL above."
        }

        # Verify connection
        sleep 2
        local ts_ip
        ts_ip=$(tailscale ip -4 2>/dev/null || true)
        if [ -n "$ts_ip" ]; then
            print_success "Tailscale connected — IP: $ts_ip"
            TAILSCALE_IP="$ts_ip"
        else
            print_warning "Tailscale not yet connected — complete auth and restart the worker."
        fi
    fi
}

# ── Setup Modes ───────────────────────────────────────────────────────

choose_setup_mode() {
    print_section "Setup Mode Selection"

    echo -e "${CYAN}Choose your setup mode:${NC}"
    echo ""
    echo "  1. Single machine (scheduler + worker)"
    echo "     → Everything runs on this machine"
    echo "     → Requires: GPU, Docker, NVIDIA drivers"
    echo ""
    echo "  2. Scheduler only"
    echo "     → API server and scheduler logic only"
    echo "     → No GPU required"
    echo ""
    echo "  3. Worker only"
    echo "     → GPU worker that reports to existing scheduler"
    echo "     → Requires: GPU, Docker, NVIDIA drivers"
    echo ""

    while true; do
        SETUP_MODE=$(ask_input "Select mode [1-3]" "1")
        case "$SETUP_MODE" in
            1|2|3) break;;
            *) echo "Invalid choice. Please enter 1, 2, or 3.";;
        esac
    done
}

setup_single_machine() {
    print_section "Single Machine Setup"

    # Verify GPU is available
    if [ "$NVIDIA_INSTALLED" != "true" ]; then
        print_error "NVIDIA GPU not detected. This mode requires a GPU."
        print_info "Please install NVIDIA drivers and run this script again."
        exit 1
    fi

    # Install Docker if needed
    if [ "$DOCKER_INSTALLED" != "true" ]; then
        if ask_yes_no "Docker is required. Install it now?" "y"; then
            install_docker
        else
            exit 1
        fi
    fi

    # Install NVIDIA Docker if needed
    if [ "$NVIDIA_DOCKER_INSTALLED" != "true" ]; then
        if ask_yes_no "NVIDIA Container Toolkit is required. Install it now?" "y"; then
            install_nvidia_docker
        else
            exit 1
        fi
    fi

    # Generate platform SSH keypair (scheduler side)
    setup_platform_ssh_keypair

    # Authorize the platform key on this host (also the GPU host)
    authorize_platform_ssh_key

    # Tailscale mesh networking
    setup_tailscale_networking

    # Register local GPU host
    register_local_host

    print_success "Single machine setup complete."
    print_info "You can now start the API server and worker agent."
}

setup_scheduler_only() {
    print_section "Scheduler Only Setup"

    # Generate platform SSH keypair (needed for web terminal to SSH into provider hosts)
    setup_platform_ssh_keypair

    print_success "Scheduler setup complete."
    print_info "You can now start the API server."
    print_info "Configure workers to connect to: http://$(hostname -I | awk '{print $1}'):8000"
}

setup_worker_only() {
    print_section "Worker Only Setup"

    # Install cryptsetup for encrypted workspace support (LUKS2)
    if ! command -v cryptsetup &>/dev/null; then
        print_info "Installing cryptsetup for encrypted workspace support..."
        sudo apt-get install -y cryptsetup-bin || print_warning "cryptsetup install failed — encrypted workspaces will be unavailable"
    fi

    # Verify GPU is available
    if [ "$NVIDIA_INSTALLED" != "true" ]; then
        print_error "NVIDIA GPU not detected. This mode requires a GPU."
        print_info "Please install NVIDIA drivers and run this script again."
        exit 1
    fi

    # Install Docker if needed
    if [ "$DOCKER_INSTALLED" != "true" ]; then
        if ask_yes_no "Docker is required. Install it now?" "y"; then
            install_docker
        else
            exit 1
        fi
    fi

    # Install NVIDIA Docker if needed
    if [ "$NVIDIA_DOCKER_INSTALLED" != "true" ]; then
        if ask_yes_no "NVIDIA Container Toolkit is required. Install it now?" "y"; then
            install_nvidia_docker
        else
            exit 1
        fi
    fi

    # Authorize the platform SSH key on this host for web terminal access
    authorize_platform_ssh_key

    # Tailscale mesh networking
    setup_tailscale_networking

    # Configure worker
    configure_worker

    print_success "Worker setup complete."
}

register_local_host() {
    print_section "Registering Local GPU Host"

    # Get host info
    DEFAULT_HOST_ID=$(hostname | tr '[:upper:]' '[:lower:]')
    HOST_ID=$(ask_input "Host ID" "$DEFAULT_HOST_ID")

    DEFAULT_IP=$(hostname -I | awk '{print $1}')
    HOST_IP=$(ask_input "Host IP" "$DEFAULT_IP")

    COST_PER_HOUR=$(ask_input "Cost per hour (USD)" "0.50")

    # Register via CLI
    source venv/bin/activate
    python cli.py host-add \
        --id "$HOST_ID" \
        --ip "$HOST_IP" \
        --gpu "$GPU_MODEL" \
        --vram "$GPU_VRAM" \
        --free-vram "$GPU_VRAM" \
        --rate "$COST_PER_HOUR"

    print_success "Host registered: $HOST_ID"

    # Save worker config (detect SSH user for web terminal)
    detect_host_ssh_user
    write_worker_env_file "$HOST_ID" "http://localhost:8000" "$COST_PER_HOUR" "$API_TOKEN" "" "" "$HOST_SSH_USER"
    print_success "Worker configuration saved to worker_env.sh"
}

configure_worker() {
    print_section "Worker Configuration"

    # Get host info
    DEFAULT_HOST_ID=$(hostname | tr '[:upper:]' '[:lower:]')
    HOST_ID=$(ask_input "Host ID" "$DEFAULT_HOST_ID")

    SCHEDULER_URL=$(ask_input "Scheduler URL" "http://192.168.1.1:8000")

    echo ""
    print_info "Worker authentication supports either an API token or OAuth client credentials."
    AUTH_METHOD=$(ask_input "Auth method (api-token/oauth-client)" "api-token")
    WORKER_API_TOKEN=""
    WORKER_OAUTH_CLIENT_ID=""
    WORKER_OAUTH_CLIENT_SECRET=""
    case "$AUTH_METHOD" in
        api-token|api|token)
            print_warning "You'll need the scheduler bearer token from your local .env or setup output."
            WORKER_API_TOKEN=$(ask_input "API Token")
            ;;
        oauth-client|oauth|client)
            print_info "Create a confidential OAuth client with grant type client_credentials after the scheduler is running."
            WORKER_OAUTH_CLIENT_ID=$(ask_input "OAuth Client ID")
            WORKER_OAUTH_CLIENT_SECRET=$(ask_input "OAuth Client Secret")
            ;;
        *)
            print_error "Unknown auth method: $AUTH_METHOD"
            exit 1
            ;;
    esac

    COST_PER_HOUR=$(ask_input "Cost per hour (USD)" "0.50")

    # Save worker config (detect SSH user for web terminal)
    detect_host_ssh_user
    write_worker_env_file \
        "$HOST_ID" \
        "$SCHEDULER_URL" \
        "$COST_PER_HOUR" \
        "$WORKER_API_TOKEN" \
        "$WORKER_OAUTH_CLIENT_ID" \
        "$WORKER_OAUTH_CLIENT_SECRET" \
        "$HOST_SSH_USER"
    print_success "Worker configuration saved to worker_env.sh"

    # Register with scheduler
    if ask_yes_no "Would you like to register this host with the scheduler now?" "y"; then
        source worker_env.sh
        source venv/bin/activate

        python - <<EOF
import requests
import os

url = os.environ['XCELSIOR_SCHEDULER_URL'] + '/host'
headers = {}
api_token = os.environ.get('XCELSIOR_API_TOKEN', '').strip()
oauth_client_id = os.environ.get('XCELSIOR_OAUTH_CLIENT_ID', '').strip()
oauth_client_secret = os.environ.get('XCELSIOR_OAUTH_CLIENT_SECRET', '').strip()

if api_token:
    headers['Authorization'] = f"Bearer {api_token}"
elif oauth_client_id and oauth_client_secret:
    token_resp = requests.post(
        os.environ['XCELSIOR_SCHEDULER_URL'].rstrip('/') + '/oauth/token',
        data={
            'grant_type': 'client_credentials',
            'client_id': oauth_client_id,
            'client_secret': oauth_client_secret,
            'scope': 'api',
        },
        timeout=10,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get('access_token', '')
    if not access_token:
        raise RuntimeError('OAuth token response missing access_token')
    headers['Authorization'] = f"Bearer {access_token}"
else:
    raise RuntimeError('No worker auth configured in worker_env.sh')

data = {
    'host_id': os.environ['XCELSIOR_HOST_ID'],
    'ip': '$(hostname -I | awk '{print $1}')',
    'gpu_model': '$GPU_MODEL',
    'total_vram_gb': float('$GPU_VRAM'),
    'free_vram_gb': float('$GPU_VRAM'),
    'cost_per_hour': float(os.environ['XCELSIOR_COST_PER_HOUR'])
}

try:
    response = requests.put(url, json=data, headers=headers, timeout=10)
    response.raise_for_status()
    print("Host registered successfully!")
except Exception as e:
    print(f"Failed to register host: {e}")
    print("You can register manually later using the CLI or API.")
EOF
    fi
}

# ── Testing & Verification ────────────────────────────────────────────

run_tests() {
    print_section "Running Verification Tests"

    source venv/bin/activate

    print_info "Running unit tests..."
    python -m pytest test_scheduler.py test_api.py -v --tb=short 2>&1 | tail -20
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_warning "Some tests failed. This may be normal if you haven't configured all features."
    fi
}

# ── Next Steps ────────────────────────────────────────────────────────

show_next_steps() {
    print_section "Setup Complete!"

    echo -e "${GREEN}${BOLD}✓ Xcelsior has been successfully installed!${NC}\n"

    case "$SETUP_MODE" in
        1)  # Single machine
            echo -e "${CYAN}${BOLD}Next Steps:${NC}\n"
            echo -e "1. ${BOLD}Start the API server:${NC}"
            echo -e "   ${YELLOW}source venv/bin/activate${NC}"
            echo -e "   ${YELLOW}uvicorn api:app --host 0.0.0.0 --port 8000${NC}"
            echo ""
            echo -e "2. ${BOLD}In another terminal, start the worker agent:${NC}"
            echo -e "   ${YELLOW}source worker_env.sh${NC}"
            echo -e "   ${YELLOW}source venv/bin/activate${NC}"
            echo -e "   ${YELLOW}python worker_agent.py${NC}"
            echo ""
            echo -e "3. ${BOLD}Submit your first job:${NC}"
            echo -e "   ${YELLOW}python cli.py run --model bert-base-uncased --vram 4.0${NC}"
            echo ""
            echo -e "4. ${BOLD}Check job status:${NC}"
            echo -e "   ${YELLOW}python cli.py jobs${NC}"
            echo ""
            echo -e "5. ${BOLD}Visit the dashboard:${NC}"
            echo -e "   ${YELLOW}http://localhost:8000/dashboard${NC}"
            ;;
        2)  # Scheduler only
            echo -e "${CYAN}${BOLD}Next Steps:${NC}\n"
            echo -e "1. ${BOLD}Start the API server:${NC}"
            echo -e "   ${YELLOW}source venv/bin/activate${NC}"
            echo -e "   ${YELLOW}uvicorn api:app --host 0.0.0.0 --port 8000${NC}"
            echo ""
            echo -e "   For production, use gunicorn:"
            echo -e "   ${YELLOW}gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000${NC}"
            echo ""
            echo -e "2. ${BOLD}Configure workers to connect to:${NC}"
            echo -e "   ${CYAN}http://$(hostname -I | awk '{print $1}'):8000${NC}"
            echo ""
            echo -e "3. ${BOLD}Your Local API Token:${NC}"
            echo -e "   ${MAGENTA}$API_TOKEN${NC}"
            echo -e "   ${YELLOW}Workers can use this token directly, or you can create OAuth client credentials later from the dashboard or API.${NC}"
            echo ""
            echo -e "4. ${BOLD}Visit the dashboard:${NC}"
            echo -e "   ${YELLOW}http://$(hostname -I | awk '{print $1}'):8000/dashboard${NC}"
            ;;
        3)  # Worker only
            echo -e "${CYAN}${BOLD}Next Steps:${NC}\n"
            echo -e "1. ${BOLD}Start the worker agent:${NC}"
            echo -e "   ${YELLOW}source worker_env.sh${NC}"
            echo -e "   ${YELLOW}source venv/bin/activate${NC}"
            echo -e "   ${YELLOW}python worker_agent.py${NC}"
            echo ""
            echo -e "2. ${BOLD}Check worker status on scheduler:${NC}"
            echo -e "   Visit the scheduler dashboard or use:"
            echo -e "   ${YELLOW}python cli.py hosts${NC}"
            ;;
    esac

    echo ""
    echo -e "${CYAN}${BOLD}Useful Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}python cli.py --help${NC}        - Show all CLI commands"
    echo -e "  ${YELLOW}python cli.py jobs${NC}          - List all jobs"
    echo -e "  ${YELLOW}python cli.py hosts${NC}         - List all hosts"
    echo -e "  ${YELLOW}python cli.py run MODEL VRAM${NC} - Submit a new job"
    echo ""
    echo -e "${CYAN}${BOLD}Documentation:${NC}"
    echo ""
    echo -e "  ${YELLOW}README.md${NC}                   - Complete documentation"
    echo -e "  ${YELLOW}http://localhost:8000/docs${NC}  - Interactive API documentation"
    echo ""
    echo -e "${GREEN}${BOLD}Ever upward. Xcelsior.${NC}"
    echo ""
}

# ── Main Execution ────────────────────────────────────────────────────

main() {
    print_header

    # System detection
    detect_os
    detect_python
    detect_docker
    detect_nvidia

    if [ "$DOCKER_INSTALLED" = "true" ] && [ "$NVIDIA_INSTALLED" = "true" ]; then
        detect_nvidia_docker
    fi

    # Choose setup mode
    choose_setup_mode

    # Create virtual environment and install dependencies
    create_venv
    install_dependencies

    # Configure environment
    configure_environment

    # Setup based on mode
    case "$SETUP_MODE" in
        1) setup_single_machine;;
        2) setup_scheduler_only;;
        3) setup_worker_only;;
    esac

    # Run tests
    if ask_yes_no "Would you like to run verification tests?" "y"; then
        run_tests
    fi

    # Show next steps
    show_next_steps
}

# Run main function
main
