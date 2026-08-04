#!/usr/bin/env bash
# READ-ONLY. Identify which commit a file in a deploy *backup* came from.
#
# The companion script, identify_deployed_files.sh, hashes the live tree. That
# answers "what is running now" — but a deploy overwrites the tree, so once you
# have deployed, the question "what was production running before?" is no longer
# answerable from live files. It is still answerable from the backup: deploy.sh
# tars /opt/xcelsior before every sync (`backup_current`, keeps five).
#
# On 2026-08-04 this mattered. Production ran for fifteen hours against a schema
# nineteen migrations behind the code in the repository, and the tree that was
# running is now only in a tarball.
#
# A NO MATCH is a finding, not a failure: it means the tree that was deployed
# corresponds to no commit, i.e. it was rsynced from a dirty working copy, and
# what production was running is attributable to no revision anyone can name.
#
# Usage
#     REMOTE=linuxuser@1.2.3.4 \
#       ./scripts/incident/identify_files_in_backup.sh /opt/xcelsior-backups/xcelsior_20260803_235214.tar.gz
#
#     # list what backups exist
#     REMOTE=… ./scripts/incident/identify_files_in_backup.sh --list
#
# Nothing is written on either host. The tarball is read with `tar -xO` to
# stdout; it is never unpacked.
set -euo pipefail

REMOTE="${REMOTE:-linuxuser@149.28.121.61}"
SSH_OPTS="${SSH_OPTS:--o ControlPath=none}"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
BACKUP_DIR="${BACKUP_DIR:-/opt/xcelsior-backups}"

# The files whose provenance the 2026-08-04 questions turn on: the SSH host-key
# pinning window, the startup-validation swallow path, the committed development
# signing secret and deterministic encryption key, and the operator-scope refusal.
FILES=("routes/terminal.py" "api.py" "security.py" "routes/auth.py")

_ssh() { ssh $SSH_OPTS -i "$SSH_KEY" "$REMOTE" "$@"; }

if [[ "${1:-}" == "--list" ]]; then
    echo "backups on $REMOTE:"
    _ssh "ls -la --time-style=long-iso $BACKUP_DIR/xcelsior_*.tar.gz 2>/dev/null || echo '  (none)'"
    exit 0
fi

TARBALL="${1:-}"
[[ -n "$TARBALL" ]] || { echo "usage: $0 <path-to-tarball-on-remote> | --list" >&2; exit 2; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not inside a git repo" >&2; exit 2; }

echo "repo    : $(git rev-parse --show-toplevel)"
echo "remote  : $REMOTE"
echo "backup  : $TARBALL"
echo

dirty_evidence=0
for f in "${FILES[@]}"; do
    echo "$f"
    # `backup_current` runs `tar -C /opt -czf … xcelsior`, so paths inside the
    # archive are prefixed with the directory name.
    backup_hash="$(_ssh "tar -xOzf '$TARBALL' 'xcelsior/$f' 2>/dev/null | git hash-object --stdin 2>/dev/null || true")"
    if [[ -z "$backup_hash" ]]; then
        echo "  NOT IN BACKUP  (absent from the archive, or the path prefix differs)"
        echo
        continue
    fi
    echo "  blob          : $backup_hash"

    match=""
    while read -r c; do
        blob="$(git rev-parse "$c:$f" 2>/dev/null)" || continue
        if [[ "$blob" == "$backup_hash" ]]; then match="$c"; break; fi
    done < <(git log --all --format='%H' -- "$f")

    if [[ -n "$match" ]]; then
        echo "  MATCHES       : $(git log -1 --format='%h %ad %s' --date=short "$match")"
    else
        echo "  NO MATCH IN HISTORY — the deployed tree was dirty for this file."
        echo "                  What production ran is attributable to no commit."
        dirty_evidence=1
    fi
    echo
done

if (( dirty_evidence )); then
    echo "At least one file matches no commit. That is the answer to the attribution"
    echo "question, in the negative — not a failure of this script."
fi
