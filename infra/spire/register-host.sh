#!/usr/bin/env bash
# Register one admitted GPU host in the SPIRE mesh (blueprint §19.2).
#
# Creates the workload registration entry whose SPIFFE ID is exactly what
# `control_plane.identity.spiffe_id_for_host()` computes and
# `parse_worker_spiffe_id()` verifies:
#
#     spiffe://<trust-domain>/worker/host/<host_id>
#
# Idempotent: re-running for a host that is already registered updates the
# entry instead of creating a duplicate.
#
# Usage:
#   infra/spire/register-host.sh <host_id> [--parent <spiffe-id>] [--selector <sel>]
#   infra/spire/register-host.sh --all        # every admitted host in the DB
#
# Environment:
#   XCELSIOR_SPIFFE_TRUST_DOMAIN  trust domain (default xcelsior.ca)
#   SPIRE_SERVER_BIN              spire-server binary (default: spire-server)
#   SPIRE_SERVER_SOCKET           admin socket (default /tmp/spire-server/private/api.sock)
#   XCELSIOR_POSTGRES_DSN         required for --all

set -euo pipefail

TRUST_DOMAIN="${XCELSIOR_SPIFFE_TRUST_DOMAIN:-xcelsior.ca}"
SPIRE_SERVER_BIN="${SPIRE_SERVER_BIN:-spire-server}"
SPIRE_SERVER_SOCKET="${SPIRE_SERVER_SOCKET:-/tmp/spire-server/private/api.sock}"

die() { echo "error: $*" >&2; exit 1; }

# Must match control_plane.identity.spiffe_host_component() exactly —
# a different sanitisation on either side means the API rejects a
# legitimately attested host.
sanitize_host_component() {
    printf '%s' "$1" | sed 's/[^A-Za-z0-9_-]/-/g'
}

spiffe_id_for_host() {
    printf 'spiffe://%s/worker/host/%s' "$TRUST_DOMAIN" "$(sanitize_host_component "$1")"
}

entry_exists() {
    "$SPIRE_SERVER_BIN" entry show \
        -socketPath "$SPIRE_SERVER_SOCKET" \
        -spiffeID "$1" 2>/dev/null | grep -q 'Entry ID'
}

register_one() {
    local host_id="$1"
    local parent_id="$2"
    shift 2
    local selectors=("$@")

    [[ -n "$host_id" ]] || die "empty host_id"
    local spiffe_id
    spiffe_id="$(spiffe_id_for_host "$host_id")"

    local args=(
        -socketPath "$SPIRE_SERVER_SOCKET"
        -spiffeID "$spiffe_id"
        -parentID "$parent_id"
        -x509SVIDTTL 3600
        -dnsName "$host_id"
    )
    for sel in "${selectors[@]}"; do
        args+=(-selector "$sel")
    done

    if entry_exists "$spiffe_id"; then
        echo "updating $spiffe_id"
        local entry_id
        entry_id="$("$SPIRE_SERVER_BIN" entry show \
            -socketPath "$SPIRE_SERVER_SOCKET" \
            -spiffeID "$spiffe_id" | awk '/Entry ID/{print $4; exit}')"
        "$SPIRE_SERVER_BIN" entry update -entryID "$entry_id" "${args[@]}"
    else
        echo "creating $spiffe_id"
        "$SPIRE_SERVER_BIN" entry create "${args[@]}"
    fi
}

parent_id="spiffe://${TRUST_DOMAIN}/spire/agent/x509pop/host"
selectors=()
mode="one"
host_id=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) mode="all"; shift ;;
        --parent) parent_id="$2"; shift 2 ;;
        --selector) selectors+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) host_id="$1"; shift ;;
    esac
done

if [[ ${#selectors[@]} -eq 0 ]]; then
    # Default: the worker agent process running as the xcelsior user.
    selectors=("unix:user:xcelsior")
fi

if [[ "$mode" == "all" ]]; then
    [[ -n "${XCELSIOR_POSTGRES_DSN:-}" ]] || die "--all requires XCELSIOR_POSTGRES_DSN"
    command -v psql >/dev/null || die "--all requires psql"
    # Only admitted hosts get an identity — admission is the control
    # plane's decision and SPIRE must not widen it.
    mapfile -t hosts < <(psql "$XCELSIOR_POSTGRES_DSN" -tAc \
        "SELECT host_id FROM hosts WHERE COALESCE(payload->>'admitted','false') = 'true'")
    [[ ${#hosts[@]} -gt 0 ]] || die "no admitted hosts found"
    for h in "${hosts[@]}"; do
        [[ -n "$h" ]] && register_one "$h" "$parent_id" "${selectors[@]}"
    done
    echo "registered ${#hosts[@]} admitted host(s) in ${TRUST_DOMAIN}"
else
    [[ -n "$host_id" ]] || die "usage: $0 <host_id> | --all"
    register_one "$host_id" "$parent_id" "${selectors[@]}"
fi
