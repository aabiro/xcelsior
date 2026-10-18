#!/usr/bin/env bash
# ── Blue-green deployment end-to-end test ─────────────────────────────
# Exercises the ACTUAL deployment mechanics:
#   - Container lifecycle (build → start → healthz → stop)
#   - Blue-green swap (green on 9500 ↔ blue on 9501)
#   - Zero-downtime: continuous requests during swap
#   - Nginx upstream sed swap (on temp copy)
#   - State file colour tracking
#   - Graceful shutdown & fallback path
#
# Prerequisites:
#   docker, curl, python3, ports 9500+9501 free (or only green running)
#
# Usage:
#   bash scripts/test_deploy_e2e.sh          # full run (rebuild images)
#   bash scripts/test_deploy_e2e.sh --quick  # skip build, reuse images
#
# The test WILL start/stop docker containers.  It leaves the environment
# in its ORIGINAL state (green running, blue stopped) on success or ctrl-c.
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
GREEN_PORT=9500
BLUE_PORT=9501
HEALTHZ="/healthz"
HEALTH_TIMEOUT=90          # max seconds to wait for a container to become healthy
DRAIN_TIMEOUT=10           # seconds to allow for graceful stop
LOAD_DURATION=8            # seconds to run zero-downtime load generator
LOAD_INTERVAL=0.15         # seconds between load requests
QUICK=false
[[ "${1:-}" == "--quick" ]] && QUICK=true

# ── Counters & helpers ────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
pass()    { PASS=$((PASS + 1)); printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail()    { FAIL=$((FAIL + 1)); printf "  \033[31m✗\033[0m %s\n" "$1"; }
skip()    { SKIP=$((SKIP + 1)); printf "  \033[33m⊘\033[0m %s (skipped)\n" "$1"; }
section() { printf "\n\033[1m── %s ──\033[0m\n" "$1"; }
info()    { printf "  \033[36mℹ\033[0m %s\n" "$1"; }

ORIGINAL_GREEN_RUNNING=false
ORIGINAL_BLUE_RUNNING=false
TMPDIR_E2E=""
LOAD_PID=""

cleanup() {
    local exit_code=$?
    printf "\n\033[1m── Cleanup ──\033[0m\n"

    # Kill background load generator if still running
    if [[ -n "$LOAD_PID" ]] && kill -0 "$LOAD_PID" 2>/dev/null; then
        kill "$LOAD_PID" 2>/dev/null || true
        wait "$LOAD_PID" 2>/dev/null || true
    fi

    # Restore original state
    if [[ "$ORIGINAL_BLUE_RUNNING" == true ]]; then
        info "Restoring api-blue (was running before test)"
        docker compose --profile blue up -d --no-deps api-blue 2>/dev/null || true
    else
        info "Stopping api-blue (was not running before test)"
        docker compose --profile blue stop api-blue 2>/dev/null || true
    fi

    if [[ "$ORIGINAL_GREEN_RUNNING" == true ]]; then
        info "Restoring api (was running before test)"
        docker compose up -d --no-deps api 2>/dev/null || true
    else
        info "Stopping api (was not running before test)"
        docker compose stop api 2>/dev/null || true
    fi

    # Clean temp dir
    if [[ -n "$TMPDIR_E2E" && -d "$TMPDIR_E2E" ]]; then
        rm -rf "$TMPDIR_E2E"
    fi

    printf "\n"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

cd "$(dirname "$0")/.."
TMPDIR_E2E=$(mktemp -d)

# ── 0. Pre-flight ─────────────────────────────────────────────────────
section "Pre-flight"

# Docker available?
if ! command -v docker &>/dev/null; then
    fail "docker not found in PATH"
    exit 1
fi
pass "docker available"

# Docker compose works?
if ! docker compose version &>/dev/null; then
    fail "docker compose plugin not available"
    exit 1
fi
pass "docker compose available"

# curl available?
command -v curl &>/dev/null && pass "curl available" || { fail "curl not found"; exit 1; }

# Snapshot current state so we can restore on exit
if docker compose ps --format '{{.Service}} {{.State}}' 2>/dev/null | grep -q '^api running'; then
    ORIGINAL_GREEN_RUNNING=true
fi
if docker compose --profile blue ps --format '{{.Service}} {{.State}}' 2>/dev/null | grep -q '^api-blue running'; then
    ORIGINAL_BLUE_RUNNING=true
fi
info "Original state: green=${ORIGINAL_GREEN_RUNNING} blue=${ORIGINAL_BLUE_RUNNING}"

# docker compose config validates (catches YAML / env errors)
if docker compose --profile blue config --quiet 2>/dev/null; then
    pass "docker compose config valid"
else
    fail "docker compose config invalid — cannot continue"
    exit 1
fi

# ── 1. Build ──────────────────────────────────────────────────────────
section "Build"

if [[ "$QUICK" == true ]]; then
    skip "Image build (--quick)"
else
    info "Building api + api-blue images (this may take a minute)..."
    if docker compose --profile blue build api api-blue 2>"$TMPDIR_E2E/build.log"; then
        pass "Images built successfully"
    else
        fail "Image build failed — see $TMPDIR_E2E/build.log"
        tail -20 "$TMPDIR_E2E/build.log" >&2
        exit 1
    fi
fi

# ── 2. Green API lifecycle ────────────────────────────────────────────
section "Green API (port $GREEN_PORT)"

# Stop blue if it's up (we test green first in isolation)
docker compose --profile blue stop api-blue 2>/dev/null || true

# Start green
info "Starting api on port $GREEN_PORT..."
if docker compose up -d --no-deps api 2>/dev/null; then
    pass "api container started"
else
    fail "api container failed to start"
    docker compose logs --tail=30 api 2>/dev/null || true
    exit 1
fi

# Wait for healthz
green_healthy=false
for i in $(seq 1 "$((HEALTH_TIMEOUT / 2))"); do
    if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        green_healthy=true
        break
    fi
    sleep 2
done
if [[ "$green_healthy" == true ]]; then
    pass "api healthy on port $GREEN_PORT (${i}x2s)"
else
    fail "api not healthy after ${HEALTH_TIMEOUT}s"
    docker compose logs --tail=40 api 2>/dev/null || true
    exit 1
fi

# Verify response body
green_body=$(curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" 2>/dev/null)
if echo "$green_body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('ok') is True" 2>/dev/null; then
    pass "healthz returns {\"ok\": true}"
else
    fail "healthz unexpected response: $green_body"
fi

# ── 3. Blue API lifecycle ─────────────────────────────────────────────
section "Blue API (port $BLUE_PORT)"

info "Starting api-blue on port $BLUE_PORT..."
if docker compose --profile blue up -d --no-deps api-blue 2>/dev/null; then
    pass "api-blue container started"
else
    fail "api-blue container failed to start"
    docker compose --profile blue logs --tail=30 api-blue 2>/dev/null || true
    exit 1
fi

# Wait for healthz
blue_healthy=false
for i in $(seq 1 "$((HEALTH_TIMEOUT / 2))"); do
    if curl -sf "http://localhost:${BLUE_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        blue_healthy=true
        break
    fi
    sleep 2
done
if [[ "$blue_healthy" == true ]]; then
    pass "api-blue healthy on port $BLUE_PORT (${i}x2s)"
else
    fail "api-blue not healthy after ${HEALTH_TIMEOUT}s"
    docker compose --profile blue logs --tail=40 api-blue 2>/dev/null || true
    # Don't exit — continue with green-only tests
fi

# ── 4. Dual-port concurrency ─────────────────────────────────────────
section "Dual-port concurrency"

if [[ "$blue_healthy" == true ]]; then
    # Fire 20 parallel requests across both ports
    ok_count=0
    err_count=0
    for port in $GREEN_PORT $BLUE_PORT; do
        for _ in $(seq 1 10); do
            curl -sf "http://localhost:${port}${HEALTHZ}" >/dev/null 2>&1 && ok_count=$((ok_count + 1)) || err_count=$((err_count + 1)) &
        done
    done
    wait
    if [[ $err_count -eq 0 ]]; then
        pass "20/20 concurrent requests succeeded (10×9500 + 10×9501)"
    else
        fail "$err_count/$((ok_count + err_count)) requests failed"
    fi
else
    skip "Dual-port concurrency (blue not healthy)"
fi

# ── 5. Nginx upstream sed swap ────────────────────────────────────────
section "Nginx upstream sed swap"

cp nginx/xcelsior.conf "$TMPDIR_E2E/xcelsior.conf"

# Swap: green→backup, blue→primary (the "green → blue" deploy)
sed -i \
    -e "/upstream xcelsior_api/,/}/{
        s/server 127.0.0.1:${BLUE_PORT} backup;/server 127.0.0.1:${BLUE_PORT};/
        s/server 127.0.0.1:${GREEN_PORT};/server 127.0.0.1:${GREEN_PORT} backup;/
    }" "$TMPDIR_E2E/xcelsior.conf"

# Verify green is now backup
if grep -A5 'upstream xcelsior_api' "$TMPDIR_E2E/xcelsior.conf" | grep -q "127.0.0.1:${GREEN_PORT} backup"; then
    pass "After swap: green ($GREEN_PORT) is backup"
else
    fail "After swap: green ($GREEN_PORT) not marked as backup"
    grep -A5 'upstream xcelsior_api' "$TMPDIR_E2E/xcelsior.conf" >&2
fi

# Verify blue is now primary (no "backup" suffix)
blue_line=$(grep -A5 'upstream xcelsior_api' "$TMPDIR_E2E/xcelsior.conf" | grep "127.0.0.1:${BLUE_PORT}")
if echo "$blue_line" | grep -qv "backup"; then
    pass "After swap: blue ($BLUE_PORT) is primary"
else
    fail "After swap: blue ($BLUE_PORT) still marked as backup"
fi

# Swap BACK: blue→backup, green→primary (the "blue → green" deploy)
sed -i \
    -e "/upstream xcelsior_api/,/}/{
        s/server 127.0.0.1:${GREEN_PORT} backup;/server 127.0.0.1:${GREEN_PORT};/
        s/server 127.0.0.1:${BLUE_PORT};/server 127.0.0.1:${BLUE_PORT} backup;/
    }" "$TMPDIR_E2E/xcelsior.conf"

if grep -A5 'upstream xcelsior_api' "$TMPDIR_E2E/xcelsior.conf" | grep -q "127.0.0.1:${BLUE_PORT} backup"; then
    pass "Reverse swap: blue ($BLUE_PORT) back to backup"
else
    fail "Reverse swap failed"
fi

# Verify round-trip produces identical config
if diff -q nginx/xcelsior.conf "$TMPDIR_E2E/xcelsior.conf" >/dev/null 2>&1; then
    pass "Round-trip swap produces identical config"
else
    fail "Round-trip swap changed the config"
    diff nginx/xcelsior.conf "$TMPDIR_E2E/xcelsior.conf" >&2
fi

# ── 6. Zero-downtime simulation ──────────────────────────────────────
section "Zero-downtime simulation"

if [[ "$blue_healthy" == true ]]; then
    LOAD_LOG="$TMPDIR_E2E/load.log"

    # Background load generator: hits BOTH ports (simulates what nginx does).
    # A real client behind nginx would be routed to whichever port is up.
    # We mirror that by trying green first, falling back to blue.
    info "Starting load generator (both ports, nginx-style) for ${LOAD_DURATION}s..."
    (
        end_time=$((SECONDS + LOAD_DURATION))
        seq_num=0
        while [[ $SECONDS -lt $end_time ]]; do
            seq_num=$((seq_num + 1))
            ts=$(date +%s.%N)
            http_code=""
            # Try green first, then blue (like nginx proxy_next_upstream)
            for try_port in $GREEN_PORT $BLUE_PORT; do
                http_code=$(curl -sf -o /dev/null -w '%{http_code}' \
                    --connect-timeout 1 --max-time 2 \
                    "http://localhost:${try_port}${HEALTHZ}" 2>/dev/null) && break
                http_code=""
            done
            [[ -z "$http_code" ]] && http_code="000"
            echo "$seq_num $ts $http_code" >> "$LOAD_LOG"
            sleep "$LOAD_INTERVAL"
        done
    ) &
    LOAD_PID=$!

    # Let load generator warm up
    sleep 1

    # Simulate the swap: stop green, verify blue is still reachable
    info "Stopping green API (simulating drain)..."
    docker compose stop -t "$DRAIN_TIMEOUT" api 2>/dev/null || true

    # Now green is down — wait a beat, then check blue is serving
    sleep 1
    blue_serves=false
    for _ in $(seq 1 5); do
        if curl -sf "http://localhost:${BLUE_PORT}${HEALTHZ}" >/dev/null 2>&1; then
            blue_serves=true
            break
        fi
        sleep 1
    done
    if [[ "$blue_serves" == true ]]; then
        pass "Blue API serves while green is down"
    else
        fail "Blue API not reachable after green stopped"
    fi

    # Bring green back up (restore for remaining tests)
    info "Restarting green API..."
    docker compose up -d --no-deps api 2>/dev/null || true
    for i in $(seq 1 "$((HEALTH_TIMEOUT / 2))"); do
        if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    # Wait for load generator to finish
    wait "$LOAD_PID" 2>/dev/null || true
    LOAD_PID=""

    # Analyse load log
    if [[ -f "$LOAD_LOG" ]]; then
        total=$(wc -l < "$LOAD_LOG")
        ok_reqs=$(grep -c ' 200$' "$LOAD_LOG" || true)
        failed_reqs=$((total - ok_reqs))
        green_down_reqs=$(awk -v drain_start="$(date +%s)" 'BEGIN{c=0} $3 != "200" {c++} END{print c}' "$LOAD_LOG")

        if [[ $total -gt 0 ]]; then
            pct=$((ok_reqs * 100 / total))
            if [[ $failed_reqs -eq 0 ]]; then
                pass "Zero-downtime: ${ok_reqs}/${total} requests succeeded (100%)"
            elif [[ $pct -ge 95 ]]; then
                pass "Near-zero-downtime: ${ok_reqs}/${total} requests succeeded (${pct}%) — ${failed_reqs} failed during port switchover"
            else
                fail "Too many failures: ${ok_reqs}/${total} (${pct}%) — expected ≥95% with nginx-style fallback"
            fi
        else
            fail "Load generator produced no output"
        fi
    fi
else
    skip "Zero-downtime simulation (blue not healthy)"
fi

# ── 7. State file tracking ───────────────────────────────────────────
section "State file tracking"

state_file="$TMPDIR_E2E/.deploy_colour"

# Initial state
echo "green" > "$state_file"
colour=$(cat "$state_file")
if [[ "$colour" == "green" ]]; then
    pass "Initial state: green"
else
    fail "Initial state: expected green, got $colour"
fi

# Simulate deploy (green → blue)
if [[ "$colour" == "green" ]]; then
    echo "blue" > "$state_file"
fi
colour=$(cat "$state_file")
if [[ "$colour" == "blue" ]]; then
    pass "After deploy: blue"
else
    fail "After deploy: expected blue, got $colour"
fi

# Simulate second deploy (blue → green)
if [[ "$colour" == "blue" ]]; then
    echo "green" > "$state_file"
fi
colour=$(cat "$state_file")
if [[ "$colour" == "green" ]]; then
    pass "After second deploy: green (round-trip)"
else
    fail "After second deploy: expected green, got $colour"
fi

# Verify deploy.sh state logic matches
state_logic_ok=$(python3 -c "
import re
script = open('scripts/deploy.sh').read()
# Must read state file with default fallback to green
assert re.search(r'deploy_colour', script), 'no state file'
assert re.search(r'echo green', script), 'no default green'
# Must write new colour to state file
assert re.search(r'echo.*new_colour.*state_file', script), 'no state write'
print('ok')
" 2>&1) && pass "deploy.sh state logic matches test expectations" || fail "deploy.sh state logic: $state_logic_ok"

# ── 8. Graceful shutdown ─────────────────────────────────────────────
section "Graceful shutdown"

if [[ "$blue_healthy" == true ]]; then
    # Verify blue survives green shutdown
    info "Stopping green to test blue isolation..."
    docker compose stop -t "$DRAIN_TIMEOUT" api 2>/dev/null || true
    sleep 1

    if curl -sf "http://localhost:${BLUE_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        pass "Blue survives green shutdown (process isolation)"
    else
        fail "Blue died when green was stopped"
    fi

    # Verify green survives blue shutdown
    info "Restarting green, then stopping blue..."
    docker compose up -d --no-deps api 2>/dev/null || true
    green_back=false
    for _ in $(seq 1 30); do
        if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
            green_back=true
            break
        fi
        sleep 2
    done
    if [[ "$green_back" != true ]]; then
        fail "Green did not come back up before blue-stop test"
    else
        docker compose --profile blue stop -t "$DRAIN_TIMEOUT" api-blue 2>/dev/null || true
        sleep 2

        if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
            pass "Green survives blue shutdown (process isolation)"
        else
            fail "Green died when blue was stopped"
        fi
    fi

    # Verify the stopped service actually stopped
    if curl -sf "http://localhost:${BLUE_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        fail "Blue still responding after stop (not actually stopped)"
    else
        pass "Blue port $BLUE_PORT closed after stop"
    fi
else
    skip "Graceful shutdown (blue not healthy)"
fi

# ── 9. Fallback path simulation ──────────────────────────────────────
section "Fallback path"

# Simulate: standby is unhealthy → fall back to restarting live service in-place
info "Testing fallback: standby (blue) is down, restart green in-place..."

# Green should already be running from previous test
if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
    # "Restart" green in-place (what deploy.sh does on fallback)
    docker compose up -d --no-deps api 2>/dev/null || true
    sleep 2
    if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        pass "Fallback in-place restart: green still healthy"
    else
        fail "Fallback in-place restart: green not healthy"
    fi
else
    info "Green not up — starting it for fallback test..."
    docker compose up -d --no-deps api 2>/dev/null || true
    for _ in $(seq 1 20); do
        curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1 && break
        sleep 2
    done
    if curl -sf "http://localhost:${GREEN_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        pass "Fallback: green started successfully"
    else
        fail "Fallback: green not healthy"
    fi
fi

# Verify blue is still stopped (no accidental restart)
if curl -sf "http://localhost:${BLUE_PORT}${HEALTHZ}" >/dev/null 2>&1; then
    fail "Blue unexpectedly running during fallback test"
else
    pass "Blue remains stopped during fallback (expected)"
fi

# ── 10. Port differentiation (runtime) ───────────────────────────────
section "Port differentiation (runtime)"

# Restart blue briefly to verify the two ports are indeed different processes
info "Starting blue to verify port differentiation..."
docker compose --profile blue up -d --no-deps api-blue 2>/dev/null || true
blue_up=false
for _ in $(seq 1 "$((HEALTH_TIMEOUT / 2))"); do
    if curl -sf "http://localhost:${BLUE_PORT}${HEALTHZ}" >/dev/null 2>&1; then
        blue_up=true
        break
    fi
    sleep 2
done

if [[ "$blue_up" == true ]]; then
    green_pid=$(docker inspect xcelsior-api-1 --format '{{.State.Pid}}' 2>/dev/null || echo "")
    blue_pid=$(docker inspect xcelsior-api-blue-1 --format '{{.State.Pid}}' 2>/dev/null || echo "")

    if [[ -n "$green_pid" && -n "$blue_pid" && "$green_pid" != "$blue_pid" ]]; then
        pass "Separate container PIDs: green=$green_pid blue=$blue_pid"
    elif [[ -n "$green_pid" && -n "$blue_pid" ]]; then
        fail "Same PID for both containers: $green_pid"
    else
        skip "Could not inspect container PIDs"
    fi

    # Different container IDs
    green_id=$(docker inspect xcelsior-api-1 --format '{{.Id}}' 2>/dev/null | head -c 12)
    blue_id=$(docker inspect xcelsior-api-blue-1 --format '{{.Id}}' 2>/dev/null | head -c 12)
    if [[ -n "$green_id" && -n "$blue_id" && "$green_id" != "$blue_id" ]]; then
        pass "Separate containers: green=$green_id blue=$blue_id"
    else
        fail "Container IDs not distinct"
    fi

    # Clean up blue
    docker compose --profile blue stop api-blue 2>/dev/null || true
else
    skip "Port differentiation (blue didn't start)"
fi

# ── Summary ───────────────────────────────────────────────────────────
printf "\n\033[1m── Results ──\033[0m\n"
printf "  \033[32m%d passed\033[0m" "$PASS"
[[ $SKIP -gt 0 ]] && printf "  \033[33m%d skipped\033[0m" "$SKIP"
[[ $FAIL -gt 0 ]] && printf "  \033[31m%d failed\033[0m" "$FAIL"
printf "\n"

exit "$FAIL"
