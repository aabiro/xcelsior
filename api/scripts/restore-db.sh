#!/usr/bin/env bash
# Xcelsior PostgreSQL Restore Script
# Restores a database from a backup created by backup-db.sh.
#
# Usage:
#   ./scripts/restore-db.sh                          # Interactive: pick from list
#   ./scripts/restore-db.sh /var/backups/xcelsior/xcelsior_20260326_030000.dump  # Specific file
#   ./scripts/restore-db.sh --latest                  # Restore most recent backup
#
# WARNING: This will DROP and recreate the target database.

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
BACKUP_DIR="${XCELSIOR_BACKUP_DIR:-/var/backups/xcelsior}"
DB_NAME="${XCELSIOR_POSTGRES_DB:-xcelsior}"
DB_USER="${XCELSIOR_POSTGRES_USER:-xcelsior}"
DB_HOST="${XCELSIOR_POSTGRES_HOST:-localhost}"
DB_PORT="${XCELSIOR_POSTGRES_PORT:-${PGPORT:-}}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[restore]${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*" >&2; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }

# ── Read selected project configuration without executing .env ────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${XCELSIOR_PROJECT_DIR:-${SCRIPT_DIR}/..}"
if [[ -z "${XCELSIOR_PROJECT_DIR:-}" && "$(basename "$(dirname "$SCRIPT_DIR")")" == "api" ]]; then
    PROJECT_DIR="${SCRIPT_DIR}/../.."
fi
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    mapfile -d '' -t database_settings < <(python3 - "$ENV_FILE" <<'PY'
from pathlib import Path
import sys

settings = {}
for raw in Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if line.startswith('export '):
        line = line[7:].lstrip()
    key, separator, value = line.partition('=')
    if separator:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        settings[key.strip()] = value
for key in ('DB', 'USER', 'HOST', 'PORT', 'PASSWORD'):
    sys.stdout.write(settings.get('XCELSIOR_POSTGRES_' + key, '') + '\0')
PY
    )
    DB_NAME="${XCELSIOR_POSTGRES_DB:-${database_settings[0]:-$DB_NAME}}"
    DB_USER="${XCELSIOR_POSTGRES_USER:-${database_settings[1]:-$DB_USER}}"
    DB_HOST="${XCELSIOR_POSTGRES_HOST:-${database_settings[2]:-$DB_HOST}}"
    DB_PORT="${XCELSIOR_POSTGRES_PORT:-${database_settings[3]:-$DB_PORT}}"
    export XCELSIOR_POSTGRES_PASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-${database_settings[4]:-}}"
    unset database_settings
fi

if [[ ! "$DB_PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$DB_PORT < 1 || 10#$DB_PORT > 65535 )); then
    err "an explicit valid PostgreSQL port is required; default cluster refused"
    exit 1
fi

# ── Select backup file ───────────────────────────────────────────────
BACKUP_FILE=""

if [[ "${1:-}" == "--latest" ]]; then
    BACKUP_FILE="$(find "$BACKUP_DIR" -name "${DB_NAME}_*.dump" -type f | sort -r | head -1)"
    if [[ -z "$BACKUP_FILE" ]]; then
        err "No backups found in $BACKUP_DIR"
        exit 1
    fi
    log "Selected latest backup: $BACKUP_FILE"

elif [[ -n "${1:-}" && -f "${1:-}" ]]; then
    BACKUP_FILE="$1"
    log "Using specified backup: $BACKUP_FILE"

elif [[ -n "${1:-}" ]]; then
    err "File not found: $1"
    exit 1

else
    # Interactive selection
    mapfile -t BACKUPS < <(find "$BACKUP_DIR" -name "${DB_NAME}_*.dump" -type f | sort -r)
    if [[ ${#BACKUPS[@]} -eq 0 ]]; then
        err "No backups found in $BACKUP_DIR"
        exit 1
    fi

    echo ""
    log "Available backups:"
    echo ""
    for i in "${!BACKUPS[@]}"; do
        SIZE="$(du -h "${BACKUPS[$i]}" | cut -f1)"
        DATE="$(stat -c %y "${BACKUPS[$i]}" | cut -d. -f1)"
        printf "  %2d) %s  (%s, %s)\n" "$((i+1))" "$(basename "${BACKUPS[$i]}")" "$SIZE" "$DATE"
    done
    echo ""
    read -rp "Select backup number [1]: " CHOICE
    CHOICE="${CHOICE:-1}"

    if [[ "$CHOICE" -lt 1 || "$CHOICE" -gt ${#BACKUPS[@]} ]] 2>/dev/null; then
        err "Invalid selection"
        exit 1
    fi
    BACKUP_FILE="${BACKUPS[$((CHOICE-1))]}"
fi

# ── Confirm ──────────────────────────────────────────────────────────
BACKUP_SIZE="$(du -h "$BACKUP_FILE" | cut -f1)"
echo ""
warn "This will DROP database '${DB_NAME}' and restore from:"
echo "  File: $(basename "$BACKUP_FILE")"
echo "  Size: ${BACKUP_SIZE}"
echo "  Host: ${DB_HOST}:${DB_PORT}"
echo ""
read -rp "Type 'yes' to confirm: " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    log "Aborted."
    exit 0
fi

# Validate the complete input before stopping any writer or dropping data.
[[ -s "$BACKUP_FILE" && -f "${BACKUP_FILE}.sha256" ]] || {
    err "a nonempty backup and checksum sidecar are required"; exit 1;
}
expected_checksum="$(awk 'NR == 1 {print $1}' "${BACKUP_FILE}.sha256")"
[[ "$expected_checksum" =~ ^[0-9a-fA-F]{64}$ ]] &&
    [[ "$(sha256sum "$BACKUP_FILE" | awk '{print $1}')" == "$expected_checksum" ]] || {
    err "backup checksum verification failed"; exit 1;
}
pg_restore --list "$BACKUP_FILE" >/dev/null || {
    err "backup catalog verification failed"; exit 1;
}

# Under per-service roles the schema belongs to the migrator: restore every
# object as that role, or Alembic cannot alter it and no runtime grant exists.
# Check the restoring login may act as it before anything is dropped.
OWNER_ROLE="${XCELSIOR_RESTORE_OWNER_ROLE:-xcelsior_migrator}"
owner_exists="$(PGPASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-}" psql \
    -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -Atq -v ON_ERROR_STOP=1 \
    -v role="$OWNER_ROLE" -c "SELECT 1 FROM pg_roles WHERE rolname = :'role'")" || {
    err "cannot inspect database roles; restore refused"; exit 1;
}
restore_role_args=()
if [[ "$owner_exists" == "1" ]]; then
    PGPASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-}" psql \
        -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -Atq -v ON_ERROR_STOP=1 \
        -c "SET ROLE \"${OWNER_ROLE}\"" >/dev/null || {
        err "${DB_USER} cannot act as ${OWNER_ROLE}; restore refused before any change"; exit 1;
    }
    restore_role_args=(--role="$OWNER_ROLE")
    DB_OWNER="$OWNER_ROLE"
else
    DB_OWNER="$DB_USER"
fi

# Every application writer must stop successfully. A failed stop is a refusal.
cd "$PROJECT_DIR"
if command -v docker &>/dev/null && [[ -f docker-compose.yml || -f compose.yaml ]]; then
    configured_services="$(docker compose config --services)" || {
        err "cannot resolve the configured writer services"; exit 1;
    }
    writers=()
    for writer in api api-blue scheduler-worker bg-worker; do
        if [[ $'\n'"$configured_services"$'\n' == *$'\n'"$writer"$'\n'* ]]; then
            writers+=("$writer")
        fi
    done
    [[ ${#writers[@]} -gt 0 ]] || { err "no application writers were identified"; exit 1; }
    docker compose stop "${writers[@]}" || { err "writer shutdown failed; restore refused"; exit 1; }
    running_writers="$(docker compose ps --status running --quiet "${writers[@]}")" || {
        err "cannot verify writer shutdown"; exit 1;
    }
    [[ -z "$running_writers" ]] || { err "application writers remain active"; exit 1; }
elif command -v systemctl &>/dev/null && systemctl is-active --quiet xcelsior-api; then
    sudo systemctl stop xcelsior-api xcelsior-health || { err "writer shutdown failed"; exit 1; }
else
    [[ "${XCELSIOR_RESTORE_WRITERS_QUIESCED:-0}" == "1" ]] || {
        err "writer shutdown cannot be verified; explicitly quiesce remote writers first"; exit 1;
    }
fi

# ── Restore ──────────────────────────────────────────────────────────
log "Dropping existing database '${DB_NAME}'..."
PGPASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-}" dropdb \
    -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
    --if-exists "$DB_NAME"

log "Creating empty database '${DB_NAME}'..."
PGPASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-}" createdb \
    -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" \
    -O "$DB_OWNER" "$DB_NAME"

log "Restoring from backup..."
if PGPASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-}" pg_restore \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --no-owner \
    --no-privileges \
    "${restore_role_args[@]}" \
    --exit-on-error \
    "$BACKUP_FILE"; then
    ok "Database restored successfully"
else
    err "pg_restore failed"
    exit 1
fi
if [[ ${#restore_role_args[@]} -gt 0 ]]; then
    # --no-privileges dropped every runtime grant. Services stay stopped until
    # the role provisioner converges them again.
    warn "Run api/scripts/provision_db_roles.py before starting any service."
fi

# A backup may predate this runtime's required schema. Keep writers stopped;
# the staged schema/runtime handoff must qualify them before traffic resumes.
ok "Database '${DB_NAME}' restored from $(basename "$BACKUP_FILE")."
warn "Application writers remain stopped. Complete the staged migration/runtime handoff before restarting them."
