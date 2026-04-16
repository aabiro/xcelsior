#!/usr/bin/env bash
# ── Zero-downtime blue-green deployment tests ─────────────────────────
# Run locally before deploying:  bash scripts/test_deploy.sh
# Validates: YAML syntax, env parity, nginx config, Dockerfile, deploy script
set -euo pipefail

PASS=0
FAIL=0
WARN=0

pass() { PASS=$((PASS + 1)); printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { FAIL=$((FAIL + 1)); printf "  \033[31m✗\033[0m %s\n" "$1"; }
warn() { WARN=$((WARN + 1)); printf "  \033[33m!\033[0m %s\n" "$1"; }
section() { printf "\n\033[1m── %s ──\033[0m\n" "$1"; }

cd "$(dirname "$0")/.."

# ── docker-compose.yml ────────────────────────────────────────────────
section "docker-compose.yml"

# 1. Valid YAML
if python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))" 2>/dev/null; then
    pass "Valid YAML"
else
    fail "Invalid YAML"
fi

# 2. Env var parity between api and api-blue
parity_result=$(python3 -c "
import yaml, sys
d = yaml.safe_load(open('docker-compose.yml'))
api = set(d['services']['api']['environment'].keys())
blue = set(d['services']['api-blue']['environment'].keys())
missing = api - blue
extra = blue - api
if missing: print(f'MISSING: {sorted(missing)}'); sys.exit(1)
if extra: print(f'EXTRA: {sorted(extra)}'); sys.exit(1)
print(f'{len(api)} vars, parity OK')
" 2>&1) && pass "Env parity: $parity_result" || fail "Env parity: $parity_result"

# 3. Port differentiation
ports_ok=$(python3 -c "
import yaml
d = yaml.safe_load(open('docker-compose.yml'))
ap = d['services']['api']['environment']['XCELSIOR_API_PORT']
bp = d['services']['api-blue']['environment']['XCELSIOR_API_PORT']
if ap == bp: print('SAME'); exit(1)
print(f'api={ap} blue={bp}')
" 2>&1) && pass "Port differentiation: $ports_ok" || fail "Port differentiation: $ports_ok"

# 4. api-blue has profiles: ["blue"]
has_profile=$(python3 -c "
import yaml
d = yaml.safe_load(open('docker-compose.yml'))
p = d['services']['api-blue'].get('profiles', [])
assert 'blue' in p, f'profiles={p}'
print('profiles=[blue]')
" 2>&1) && pass "api-blue $has_profile" || fail "api-blue missing profile: $has_profile"

# 5. Both services have healthcheck
for svc in api api-blue; do
    has_hc=$(python3 -c "
import yaml
d = yaml.safe_load(open('docker-compose.yml'))
hc = d['services']['$svc'].get('healthcheck')
assert hc and 'test' in hc, 'no healthcheck'
print('healthcheck OK')
" 2>&1) && pass "$svc $has_hc" || fail "$svc healthcheck: $has_hc"
done

# 6. Both services have stop_grace_period
for svc in api api-blue; do
    has_sgp=$(python3 -c "
import yaml
d = yaml.safe_load(open('docker-compose.yml'))
sgp = d['services']['$svc'].get('stop_grace_period')
assert sgp, 'no stop_grace_period'
print(f'stop_grace_period={sgp}')
" 2>&1) && pass "$svc $has_sgp" || fail "$svc $has_sgp"
done

# 7. Volumes match
vols_ok=$(python3 -c "
import yaml
d = yaml.safe_load(open('docker-compose.yml'))
av = d['services']['api'].get('volumes', [])
bv = d['services']['api-blue'].get('volumes', [])
assert av == bv, f'api={av} blue={bv}'
print(f'{len(av)} volumes match')
" 2>&1) && pass "Volumes: $vols_ok" || fail "Volumes: $vols_ok"

# 8. docker compose config validates (if docker available)
if command -v docker &>/dev/null; then
    if docker compose --profile blue config --quiet 2>/dev/null; then
        pass "docker compose config validates"
    else
        fail "docker compose config validation failed"
    fi
else
    warn "docker not available — skipping compose config validation"
fi

# ── Dockerfile ────────────────────────────────────────────────────────
section "Dockerfile"

# 9. Uses gunicorn (not raw uvicorn)
if grep -q 'CMD.*gunicorn' Dockerfile; then
    pass "CMD uses gunicorn"
else
    fail "CMD does not use gunicorn"
fi

# 10. Has HEALTHCHECK
if grep -q '^HEALTHCHECK' Dockerfile; then
    pass "HEALTHCHECK instruction present"
else
    fail "HEALTHCHECK instruction missing"
fi

# 11. curl installed (needed for HEALTHCHECK)
if grep -q 'apt-get.*curl' Dockerfile; then
    pass "curl in apt-get install"
else
    fail "curl not installed — HEALTHCHECK will fail"
fi

# 12. gunicorn.conf.py copied
if grep -q 'gunicorn.conf.py' Dockerfile; then
    pass "gunicorn.conf.py referenced in Dockerfile"
else
    fail "gunicorn.conf.py not referenced"
fi

# ── gunicorn.conf.py ──────────────────────────────────────────────────
section "gunicorn.conf.py"

# 13. File exists
if [[ -f gunicorn.conf.py ]]; then
    pass "gunicorn.conf.py exists"
else
    fail "gunicorn.conf.py missing"
fi

# 14. Uses uvicorn workers
if grep -q 'UvicornWorker' gunicorn.conf.py; then
    pass "worker_class = UvicornWorker"
else
    fail "worker_class not UvicornWorker"
fi

# 15. Reads port from env
if grep -q 'XCELSIOR_API_PORT' gunicorn.conf.py; then
    pass "Reads XCELSIOR_API_PORT from env"
else
    fail "Hard-coded port (no XCELSIOR_API_PORT)"
fi

# 16. preload_app for fork safety
if grep -q 'preload_app.*=.*True' gunicorn.conf.py; then
    pass "preload_app = True"
else
    warn "preload_app not True — slower worker startup"
fi

# 17. graceful_timeout for drain
if grep -q 'graceful_timeout' gunicorn.conf.py; then
    pass "graceful_timeout configured"
else
    fail "No graceful_timeout — workers won't drain on shutdown"
fi

# ── nginx/xcelsior.conf ──────────────────────────────────────────────
section "nginx/xcelsior.conf"

# 18. Brace balance
brace_result=$(python3 -c "
f = open('nginx/xcelsior.conf').read()
o, c = f.count('{'), f.count('}')
assert o == c, f'open={o} close={c}'
print(f'{o} pairs balanced')
" 2>&1) && pass "Braces: $brace_result" || fail "Braces: $brace_result"

# 19. Backup server in upstream
if grep -A5 'upstream xcelsior_api' nginx/xcelsior.conf | grep -q 'backup'; then
    pass "Backup server in xcelsior_api upstream"
else
    fail "No backup server in upstream"
fi

# 20. Both ports present
if grep -A5 'upstream xcelsior_api' nginx/xcelsior.conf | grep -q '9500' &&
   grep -A5 'upstream xcelsior_api' nginx/xcelsior.conf | grep -q '9501'; then
    pass "Both ports 9500/9501 in upstream"
else
    fail "Missing port in upstream block"
fi

# 21. proxy_next_upstream configured
if grep -q 'proxy_next_upstream' nginx/xcelsior.conf; then
    pass "proxy_next_upstream configured"
else
    fail "No proxy_next_upstream — no failover during swap"
fi

# 22. proxy_next_upstream includes 502/503/504
if grep 'proxy_next_upstream ' nginx/xcelsior.conf | grep -q 'http_502.*http_503.*http_504'; then
    pass "proxy_next_upstream handles 502/503/504"
else
    fail "proxy_next_upstream missing error codes"
fi

# ── scripts/deploy.sh ────────────────────────────────────────────────
section "scripts/deploy.sh"

# 23. Blue-green state file
if grep -q 'deploy_colour' scripts/deploy.sh; then
    pass "Blue-green state file (.deploy_colour)"
else
    fail "No state file for tracking live colour"
fi

# 24. Standby health check before swap
if grep -q 'standby_ok' scripts/deploy.sh; then
    pass "Standby health check before nginx swap"
else
    fail "No standby health check — could swap to unhealthy service"
fi

# 25. nginx sed swap
if grep -q 'sed.*upstream' scripts/deploy.sh || grep -q "sed.*xcelsior_api" scripts/deploy.sh; then
    pass "nginx upstream sed swap"
else
    fail "No nginx upstream swap logic"
fi

# 26. Graceful stop with timeout
if grep -q 'stop -t 30' scripts/deploy.sh; then
    pass "Graceful stop with 30s drain"
else
    fail "No graceful stop timeout"
fi

# 27. Fallback path
if grep -q 'Falling back to in-place restart' scripts/deploy.sh; then
    pass "Fallback to in-place restart on failure"
else
    fail "No fallback — deploy stuck if standby fails"
fi

# 28. --profile blue on ALL docker compose commands that touch api-blue
profile_cmds=$(grep -n 'docker compose' scripts/deploy.sh | grep -v '#' | grep -v 'profile blue' | grep -iv 'frontend' || true)
if [[ -z "$profile_cmds" ]]; then
    pass "All docker compose commands include --profile blue"
else
    # Some commands legitimately don't need --profile blue (frontend-only)
    # Check if any reference api-blue without profile
    needs_fix=$(echo "$profile_cmds" | grep -i 'api-blue\|standby\|live_service' || true)
    if [[ -z "$needs_fix" ]]; then
        pass "All api-blue-related commands include --profile blue"
    else
        fail "--profile blue missing on: $needs_fix"
    fi
fi

# 29. Build includes --profile blue
if grep -q 'docker compose --profile blue build' scripts/deploy.sh; then
    pass "Build step uses --profile blue"
else
    fail "Build step missing --profile blue — api-blue image won't be built"
fi

# ── .env files ────────────────────────────────────────────────────────
section ".env files"

# 30. XCELSIOR_API_BLUE_PORT in .env
if grep -q 'XCELSIOR_API_BLUE_PORT' .env; then
    pass ".env has XCELSIOR_API_BLUE_PORT"
else
    fail ".env missing XCELSIOR_API_BLUE_PORT"
fi

# 31. XCELSIOR_API_BLUE_PORT in .env.example
if grep -q 'XCELSIOR_API_BLUE_PORT' .env.example; then
    pass ".env.example has XCELSIOR_API_BLUE_PORT"
else
    fail ".env.example missing XCELSIOR_API_BLUE_PORT"
fi

# 32. Ports don't conflict
api_port=$(grep 'XCELSIOR_API_PORT=' .env | head -1 | cut -d= -f2)
blue_port=$(grep 'XCELSIOR_API_BLUE_PORT=' .env | head -1 | cut -d= -f2)
if [[ "$api_port" != "$blue_port" ]]; then
    pass "Ports don't conflict: api=$api_port blue=$blue_port"
else
    fail "Port conflict: both are $api_port"
fi

# ── Summary ──────────────────────────────────────────────────────────
printf "\n\033[1m── Results ──\033[0m\n"
printf "  \033[32m%d passed\033[0m" "$PASS"
[[ $WARN -gt 0 ]] && printf "  \033[33m%d warnings\033[0m" "$WARN"
[[ $FAIL -gt 0 ]] && printf "  \033[31m%d failed\033[0m" "$FAIL"
printf "\n"

exit "$FAIL"
