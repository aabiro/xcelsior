#!/usr/bin/env bash
# Xcelsior PostgreSQL Backup Script
# Runs via cron to create compressed, timestamped database backups.
#
# Features:
#   - Compressed pg_dump (custom format, best compression)
#   - Automatic rotation (keeps last N backups)
#   - Optional notification via Telegram on failure
#   - Exit codes: 0 = success, 1 = failure
#
# Usage:
#   ./scripts/backup-db.sh                   # Run backup with defaults
#   BACKUP_RETAIN_DAYS=30 ./scripts/backup-db.sh  # Keep 30 days
#
# Cron example (daily at 3 AM UTC):
#   0 3 * * * /opt/xcelsior/scripts/backup-db.sh >> /var/log/xcelsior-backup.log 2>&1

set -euo pipefail

# ── Configuration (override via environment) ──────────────────────────
BACKUP_DIR="${XCELSIOR_BACKUP_DIR:-/var/backups/xcelsior}"
BACKUP_RETAIN_DAYS="${XCELSIOR_BACKUP_RETAIN_DAYS:-14}"
DB_NAME="${XCELSIOR_POSTGRES_DB:-xcelsior}"
DB_USER="${XCELSIOR_POSTGRES_USER:-xcelsior}"
DB_HOST="${XCELSIOR_POSTGRES_HOST:-localhost}"
DB_PORT="${XCELSIOR_POSTGRES_PORT:-5432}"

# Telegram alert (optional — reads from .env if sourced)
TG_TOKEN="${XCELSIOR_TG_TOKEN:-}"
TG_CHAT_ID="${XCELSIOR_TG_CHAT_ID:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.dump"
LOCK_FILE="/tmp/xcelsior-backup.lock"
LOG_PREFIX="[xcelsior-backup]"

# ── Helpers ───────────────────────────────────────────────────────────
log()  { echo "${LOG_PREFIX} $(date '+%Y-%m-%d %H:%M:%S') INFO  $*"; }
err()  { echo "${LOG_PREFIX} $(date '+%Y-%m-%d %H:%M:%S') ERROR $*" >&2; }

send_tg_alert() {
    if [[ -n "$TG_TOKEN" && -n "$TG_CHAT_ID" ]]; then
        local msg="⚠️ Xcelsior DB Backup FAILED on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')\n\nError: $1"
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            -d chat_id="$TG_CHAT_ID" \
            -d text="$msg" \
            -d parse_mode="HTML" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT

# ── Prevent concurrent runs ───────────────────────────────────────────
if [[ -f "$LOCK_FILE" ]]; then
    # Check if the PID in the lock file is still running
    if kill -0 "$(cat "$LOCK_FILE" 2>/dev/null)" 2>/dev/null; then
        err "Another backup is already running (PID $(cat "$LOCK_FILE")). Exiting."
        exit 1
    else
        log "Stale lock file found. Removing."
        rm -f "$LOCK_FILE"
    fi
fi
echo $$ > "$LOCK_FILE"

# ── Source .env if available (for credentials/TG tokens) ──────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    # Re-read after sourcing in case .env has overrides
    DB_NAME="${XCELSIOR_POSTGRES_DB:-$DB_NAME}"
    DB_USER="${XCELSIOR_POSTGRES_USER:-$DB_USER}"
    DB_HOST="${XCELSIOR_POSTGRES_HOST:-$DB_HOST}"
    DB_PORT="${XCELSIOR_POSTGRES_PORT:-$DB_PORT}"
    TG_TOKEN="${XCELSIOR_TG_TOKEN:-$TG_TOKEN}"
    TG_CHAT_ID="${XCELSIOR_TG_CHAT_ID:-$TG_CHAT_ID}"
fi

# ── Pre-flight checks ────────────────────────────────────────────────
if ! command -v pg_dump &>/dev/null; then
    err "pg_dump not found. Install postgresql-client."
    send_tg_alert "pg_dump not found"
    exit 1
fi

# ── Create backup directory ──────────────────────────────────────────
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

# ── Run backup ───────────────────────────────────────────────────────
log "Starting backup of database '${DB_NAME}' → ${BACKUP_FILE}"

if PGPASSWORD="${XCELSIOR_POSTGRES_PASSWORD:-}" pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -Fc \
    -Z 9 \
    -f "$BACKUP_FILE"; then

    BACKUP_SIZE="$(du -h "$BACKUP_FILE" | cut -f1)"
    log "Backup completed successfully: ${BACKUP_FILE} (${BACKUP_SIZE})"
else
    err "pg_dump failed with exit code $?"
    send_tg_alert "pg_dump failed with exit code $?"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# ── Verify backup is not empty ───────────────────────────────────────
if [[ ! -s "$BACKUP_FILE" ]]; then
    err "Backup file is empty. Removing."
    send_tg_alert "Backup file was empty"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# ── Rotate old backups ──────────────────────────────────────────────
DELETED=0
while IFS= read -r old_backup; do
    rm -f "$old_backup"
    ((DELETED++))
done < <(find "$BACKUP_DIR" -name "${DB_NAME}_*.dump" -mtime +"$BACKUP_RETAIN_DAYS" -type f)

if [[ $DELETED -gt 0 ]]; then
    log "Rotated ${DELETED} backup(s) older than ${BACKUP_RETAIN_DAYS} days"
fi

# ── Summary ──────────────────────────────────────────────────────────
TOTAL_BACKUPS="$(find "$BACKUP_DIR" -name "${DB_NAME}_*.dump" -type f | wc -l)"
TOTAL_SIZE="$(du -sh "$BACKUP_DIR" | cut -f1)"
log "Backup inventory: ${TOTAL_BACKUPS} backup(s), ${TOTAL_SIZE} total"
log "Done."
