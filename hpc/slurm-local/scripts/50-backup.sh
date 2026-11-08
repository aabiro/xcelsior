#!/usr/bin/env bash
# Back up the tower's cluster state: the accounting database, slurmctld's
# state, /etc/slurm and the munge key.
#
#   sudo ./50-backup.sh            → /var/lib/xcl-slurm-test/backups/<UTC time>.tar.gz
#
# The archive holds the munge key and slurmdbd's database password, so it is
# root-only (0600). It lives under /var/lib/xcl-slurm-test, which
# 90-rollback.sh deletes: copy archives off the machine before a rollback.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
xcl_require_root
[[ -d /var/spool/slurmctld ]] || xcl_die "no /var/spool/slurmctld: run this on the tower"

dest="$XCL_STATE/backups"
install -d -m 0700 "$dest"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
umask 077
# A consistent snapshot without stopping slurmdbd.
mariadb-dump --protocol=socket --single-transaction slurm_acct_db >"$work/slurm_acct_db.sql"
tar -C / -czf "$dest/$stamp.tar.gz" \
    --transform "s|^|xcl-slurm-backup-$stamp/|" \
    -C "$work" slurm_acct_db.sql \
    -C / var/spool/slurmctld etc/slurm etc/munge/munge.key
chmod 0600 "$dest/$stamp.tar.gz"
xcl_log "backup written: $dest/$stamp.tar.gz ($(du -h "$dest/$stamp.tar.gz" | cut -f1))"
