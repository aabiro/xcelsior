#!/usr/bin/env bash
# Restore the newest production backup into a disposable PostgreSQL database,
# migrate it with the current application image, assert control-plane
# invariants, and write machine-readable evidence.

set -euo pipefail
umask 077

BACKUP_DIR="${XCELSIOR_BACKUP_DIR:-/var/backups/xcelsior}"
EVIDENCE_DIR="${XCELSIOR_RESTORE_EVIDENCE_DIR:-${BACKUP_DIR}/restore-evidence}"
APP_DB_ROLE="${XCELSIOR_RESTORE_APP_ROLE:-xcelsior}"
KEEP_RESTORE_DB="${XCELSIOR_RESTORE_KEEP_DATABASE:-0}"
REQUIRE_CHECKSUM="${XCELSIOR_RESTORE_REQUIRE_CHECKSUM:-1}"
RUN_MIGRATIONS="${XCELSIOR_RESTORE_RUN_MIGRATIONS:-1}"
PROJECT_DIR="${XCELSIOR_PROJECT_DIR:-/opt/xcelsior}"
METRICS_DIR="${XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR:-/var/lib/node_exporter/textfile_collector}"
LOG_PREFIX="[xcelsior-restore-drill]"

log() { echo "${LOG_PREFIX} $(date -u '+%Y-%m-%dT%H:%M:%SZ') $*"; }
die() { log "ERROR $*" >&2; exit 1; }

write_textfile_metric() {
    local filename="$1"
    shift
    if ! mkdir -p "$METRICS_DIR" 2>/dev/null; then
        return 0
    fi
    local temporary
    temporary="$(mktemp "${METRICS_DIR}/.${filename}.XXXXXX")" || return 0
    printf '%s\n' "$@" >"$temporary"
    chmod 0644 "$temporary"
    mv -f "$temporary" "${METRICS_DIR}/${filename}"
}

as_postgres() {
    if [[ "$(id -u)" -eq 0 ]]; then
        runuser -u postgres -- "$@"
    elif [[ "$(id -un)" == "postgres" ]]; then
        "$@"
    else
        die "run as root or postgres so the drill cannot depend on the application role"
    fi
}

backup_file="${1:-}"
if [[ -z "$backup_file" ]]; then
    backup_file="$(
        find "$BACKUP_DIR" -maxdepth 1 -type f -name 'xcelsior_*.dump' \
            -printf '%T@ %p\n' |
            sort -nr |
            awk 'NR == 1 {sub(/^[^ ]+ /, ""); print; exit}'
    )"
fi
[[ -n "$backup_file" ]] || die "no xcelsior backup found in $BACKUP_DIR"

backup_dir_real="$(realpath "$BACKUP_DIR")"
backup_real="$(realpath "$backup_file")"
case "$backup_real" in
    "$backup_dir_real"/*.dump) ;;
    *) die "backup must be an explicit .dump file directly under $backup_dir_real" ;;
esac
[[ -s "$backup_real" ]] || die "backup is missing or empty: $backup_real"

checksum_file="${backup_real}.sha256"
if [[ -f "$checksum_file" ]]; then
    (
        cd "$(dirname "$backup_real")"
        sha256sum --check --status "$(basename "$checksum_file")"
    ) || die "backup checksum verification failed"
elif [[ "$REQUIRE_CHECKSUM" == "1" ]]; then
    die "checksum sidecar is required: $checksum_file"
else
    log "WARNING checksum sidecar is absent"
fi
backup_sha256="$(sha256sum "$backup_real" | awk '{print $1}')"

stamp="$(date -u +%Y%m%d_%H%M%S)"
target_db="xcelsior_restore_${stamp}_$$"
[[ "$target_db" =~ ^xcelsior_restore_[0-9]{8}_[0-9]{6}_[0-9]+$ ]] ||
    die "unsafe generated restore database name"
[[ "$target_db" != "xcelsior" ]] || die "refusing to target the production database"

started_epoch="$(date +%s)"
restore_input="$backup_real"
temporary_restore_input=""
if [[ "$(id -u)" -eq 0 ]]; then
    temporary_restore_input="$(mktemp /var/lib/postgresql/.xcelsior-restore-XXXXXX.dump)"
    install -m 0400 -o postgres -g postgres "$backup_real" "$temporary_restore_input"
    restore_input="$temporary_restore_input"
fi

cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$temporary_restore_input" ]]; then
        case "$temporary_restore_input" in
            /var/lib/postgresql/.xcelsior-restore-*.dump)
                rm -f "$temporary_restore_input"
                ;;
        esac
    fi
    if [[ "$KEEP_RESTORE_DB" == "1" ]]; then
        log "keeping isolated database $target_db for inspection"
    else
        as_postgres dropdb --if-exists "$target_db" >/dev/null 2>&1 || true
    fi
    if [[ "$status" -ne 0 ]]; then
        write_textfile_metric "xcelsior_restore_failure.prom" \
            "# TYPE xcelsior_restore_last_failure_timestamp_seconds gauge" \
            "xcelsior_restore_last_failure_timestamp_seconds $(date +%s)"
    fi
    exit "$status"
}
trap cleanup EXIT

log "creating isolated database $target_db"
as_postgres createdb --template=template0 --owner="$APP_DB_ROLE" "$target_db"
log "restoring $(basename "$backup_real")"
as_postgres pg_restore \
    --exit-on-error \
    --no-owner \
    --no-privileges \
    --dbname="$target_db" \
    "$restore_input"

if [[ "$RUN_MIGRATIONS" == "1" ]]; then
    [[ -f "$PROJECT_DIR/docker-compose.yml" ]] ||
        die "docker-compose.yml not found at $PROJECT_DIR"
    log "migrating restored database with the current API image"
    (
        cd "$PROJECT_DIR"
        XCELSIOR_POSTGRES_DB="$target_db" docker compose run --rm --no-deps \
            api python -m alembic upgrade head
    )
fi

log "checking restored control-plane and ledger invariants"
as_postgres psql --dbname="$target_db" --set=ON_ERROR_STOP=1 <<'SQL'
DO $invariants$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM gpu_device_allocations
        WHERE status = 'active'
        GROUP BY host_id, gpu_device_id
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'duplicate active GPU allocation';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM jobs j
        LEFT JOIN job_attempts a ON a.attempt_id = j.active_attempt_id
        WHERE j.active_attempt_id IS NOT NULL
          AND (a.attempt_id IS NULL OR a.job_id <> j.job_id)
    ) THEN
        RAISE EXCEPTION 'job has an invalid active attempt reference';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM job_attempts
        WHERE status IN ('reserved', 'offered', 'claimed', 'starting', 'running', 'stopping')
        GROUP BY job_id
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'job has more than one active attempt';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM wallets
        WHERE balance_micros IS NULL
           OR total_deposited_micros IS NULL
           OR total_spent_micros IS NULL
           OR total_refunded_micros IS NULL
    ) THEN
        RAISE EXCEPTION 'wallet has an unprojected exact-money value';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM wallet_transactions
        WHERE amount_micros IS NULL OR balance_after_micros IS NULL
    ) THEN
        RAISE EXCEPTION 'wallet transaction has an unprojected exact-money value';
    END IF;
END
$invariants$;
SQL

schema_revision="$(
    as_postgres psql --dbname="$target_db" --tuples-only --no-align \
        --command='SELECT version_num FROM alembic_version'
)"
schema_revision="${schema_revision//[[:space:]]/}"
[[ -n "$schema_revision" ]] || die "restored database has no Alembic revision"

read -r users wallets transactions jobs hosts invoices <<<"$(
    as_postgres psql --dbname="$target_db" --tuples-only --no-align --field-separator=' ' \
        --command='SELECT
            (SELECT count(*) FROM users),
            (SELECT count(*) FROM wallets),
            (SELECT count(*) FROM wallet_transactions),
            (SELECT count(*) FROM jobs),
            (SELECT count(*) FROM hosts),
            (SELECT count(*) FROM invoices)'
)"

completed_epoch="$(date +%s)"
duration_seconds=$((completed_epoch - started_epoch))
mkdir -p "$EVIDENCE_DIR"
evidence_file="${EVIDENCE_DIR}/${stamp}.json"
python3 - "$evidence_file" <<PY
import json
from pathlib import Path

evidence = {
    "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
    "backup_file": "$(basename "$backup_real")",
    "backup_sha256": "$backup_sha256",
    "schema_revision": "$schema_revision",
    "duration_seconds": $duration_seconds,
    "invariants": {
        "exclusive_gpu_allocations": True,
        "active_attempt_references": True,
        "one_active_attempt_per_job": True,
        "exact_wallet_projection": True,
    },
    "row_counts": {
        "users": int("$users"),
        "wallets": int("$wallets"),
        "wallet_transactions": int("$transactions"),
        "jobs": int("$jobs"),
        "hosts": int("$hosts"),
        "invoices": int("$invoices"),
    },
}
target = Path("$evidence_file")
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(target)
PY

write_textfile_metric "xcelsior_restore_success.prom" \
    "# TYPE xcelsior_restore_last_success_timestamp_seconds gauge" \
    "xcelsior_restore_last_success_timestamp_seconds $(date +%s)" \
    "# TYPE xcelsior_restore_duration_seconds gauge" \
    "xcelsior_restore_duration_seconds ${duration_seconds}"
log "PASS revision=$schema_revision duration=${duration_seconds}s evidence=$evidence_file"
