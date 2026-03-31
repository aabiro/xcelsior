#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Xcelsior — Dev Environment Launcher
# ─────────────────────────────────────────────────────────────────────────────
# Starts backend + frontend, opens browser, and tails logs.
#
# Usage:
#   ./run-dev.sh            Start everything (backend + frontend + browser)
#   ./run-dev.sh backend    Start backend only
#   ./run-dev.sh frontend   Start frontend only
#   ./run-dev.sh stop       Kill running backend/frontend processes
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ───────────────────────────────────────────────────────────────
BACKEND_PORT="${XCELSIOR_API_PORT:-9500}"
FRONTEND_PORT="${XCELSIOR_FRONTEND_PORT:-3000}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
LOG_DIR="$SCRIPT_DIR/.dev-logs"

# ── Colors ───────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}▸${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn()    { echo -e "${YELLOW}⚠${NC} $1"; }
error()   { echo -e "${RED}✗${NC} $1"; }

# ── Helpers ──────────────────────────────────────────────────────────────
open_browser() {
    local url="$1"
    sleep 3  # give servers a moment
    if command -v xdg-open &>/dev/null; then
        xdg-open "$url" &>/dev/null &
    elif command -v open &>/dev/null; then
        open "$url" &
    elif command -v wslview &>/dev/null; then
        wslview "$url" &
    else
        warn "Could not detect browser — open manually: ${BOLD}${url}${NC}"
        return
    fi
    success "Opened ${BOLD}${url}${NC} in browser"
}

check_port() {
    local port="$1" name="$2"
    if port_is_open "$port"; then
        warn "${name} port ${port} already in use"
        lsof -i :"$port" -sTCP:LISTEN -t 2>/dev/null | head -3 | while read pid; do
            warn "  PID $pid: $(ps -p "$pid" -o comm= 2>/dev/null || echo 'unknown')"
        done
        return 1
    fi
    return 0
}

port_is_open() {
    local port="$1"
    # Try multiple detection methods
    (echo >/dev/tcp/localhost/"$port") 2>/dev/null && return 0
    lsof -i :"$port" -sTCP:LISTEN &>/dev/null 2>&1 && return 0
    ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    return 1
}
would this be why i dont see the canada map animations version as well?wait_for_port() {
    local port="$1" name="$2" timeout="${3:-30}"
    local elapsed=0
    while ! port_is_open "$port"; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [[ $elapsed -ge $timeout ]]; then
            error "${name} failed to start within ${timeout}s"
            error "Check logs: cat .dev-logs/$(echo "$name" | tr '[:upper:]' '[:lower:]').log"
            return 1
        fi
    done
    success "${name} is up on port ${port} (${elapsed}s)"
}

stop_services() {
    info "Stopping Xcelsior dev services…"
    local killed=0

    # Kill backend (uvicorn on BACKEND_PORT)
    local backend_pids
    backend_pids=$(lsof -i :"$BACKEND_PORT" -sTCP:LISTEN -t 2>/dev/null || true)
    if [[ -n "$backend_pids" ]]; then
        echo "$backend_pids" | xargs kill 2>/dev/null && killed=$((killed + 1))
        success "Stopped backend (port ${BACKEND_PORT})"
    fi

    # Kill frontend — find by process name since lsof may not detect Node
    local fe_pids
    fe_pids=$(pgrep -f "next dev.*--port ${FRONTEND_PORT}" 2>/dev/null || true)
    if [[ -z "$fe_pids" ]]; then
        fe_pids=$(pgrep -f "next dev" 2>/dev/null || true)
    fi
    if [[ -z "$fe_pids" ]]; then
        fe_pids=$(lsof -i :"$FRONTEND_PORT" -sTCP:LISTEN -t 2>/dev/null || true)
    fi
    if [[ -n "$fe_pids" ]]; then
        echo "$fe_pids" | xargs kill 2>/dev/null && killed=$((killed + 1))
        success "Stopped frontend (port ${FRONTEND_PORT})"
    fi

    # Also kill via PID files
    for svc in backend frontend; do
        if [[ -f "$LOG_DIR/${svc}.pid" ]]; then
            local pid
            pid=$(cat "$LOG_DIR/${svc}.pid")
            kill "$pid" 2>/dev/null && { killed=$((killed + 1)); success "Stopped ${svc} (PID ${pid})"; } || true
            rm -f "$LOG_DIR/${svc}.pid"
        fi
    done

    if [[ $killed -eq 0 ]]; then
        info "No running services found"
    fi
    exit 0
}

start_backend() {
    info "Starting backend on port ${BACKEND_PORT}…"

    # Activate venv
    if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "venv" ]]; then
        source venv/bin/activate
    fi

    # Ensure .env exists
    if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
        warn "No .env found — copying .env.example"
        cp .env.example .env
    fi

    mkdir -p "$LOG_DIR"
    uvicorn api:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload \
        --log-level info \
        > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$LOG_DIR/backend.pid"

    wait_for_port "$BACKEND_PORT" "Backend" 20
}

start_frontend() {
    info "Starting frontend on port ${FRONTEND_PORT}…"

    # Set API URL for local dev
    export NEXT_PUBLIC_API_URL="$BACKEND_URL"

    mkdir -p "$LOG_DIR"

    # Install deps if needed
    if [[ ! -d "frontend/node_modules" ]]; then
        info "Installing frontend dependencies…"
        (cd frontend && npm install) >> "$LOG_DIR/frontend.log" 2>&1
        success "Dependencies installed"
    fi

    (cd frontend && npm run dev -- --port "$FRONTEND_PORT") \
        > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$LOG_DIR/frontend.pid"

    wait_for_port "$FRONTEND_PORT" "Frontend" 30
}

# ── Banner ───────────────────────────────────────────────────────────────
print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "  ╔══════════════════════════════════════════════╗"
    echo "  ║          XCELSIOR  DEV  ENVIRONMENT          ║"
    echo "  ╚══════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  Backend  → ${BOLD}${BACKEND_URL}${NC}  (API docs: ${BACKEND_URL}/docs)"
    echo -e "  Frontend → ${BOLD}${FRONTEND_URL}${NC}"
    echo -e "  Logs     → ${BOLD}.dev-logs/${NC}"
    echo ""
    echo -e "  ${YELLOW}Press Ctrl+C to stop all services${NC}"
    echo ""
}

# ── Cleanup on exit ──────────────────────────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down…"
    # Kill child processes
    if [[ -f "$LOG_DIR/backend.pid" ]]; then
        kill "$(cat "$LOG_DIR/backend.pid")" 2>/dev/null && success "Backend stopped"
        rm -f "$LOG_DIR/backend.pid"
    fi
    if [[ -f "$LOG_DIR/frontend.pid" ]]; then
        kill "$(cat "$LOG_DIR/frontend.pid")" 2>/dev/null && success "Frontend stopped"
        rm -f "$LOG_DIR/frontend.pid"
    fi
    # Also clean up any children
    jobs -p 2>/dev/null | xargs kill 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Main ─────────────────────────────────────────────────────────────────
TARGET="${1:-all}"

case "$TARGET" in
    stop)
        stop_services
        ;;
    backend)
        check_port "$BACKEND_PORT" "Backend" || exit 1
        start_backend
        print_banner
        tail -f "$LOG_DIR/backend.log"
        ;;
    frontend)
        start_frontend
        open_browser "$FRONTEND_URL"
        print_banner
        tail -f "$LOG_DIR/frontend.log"
        ;;
    all|"")
        check_port "$BACKEND_PORT" "Backend" || exit 1
        check_port "$FRONTEND_PORT" "Frontend" || exit 1

        start_backend
        start_frontend
        open_browser "$FRONTEND_URL"
        print_banner

        # Tail both logs interleaved
        tail -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log"
        ;;
    *)
        error "Unknown target: $TARGET"
        echo "Usage: ./run-dev.sh [all|backend|frontend|stop]"
        exit 1
        ;;
esac
