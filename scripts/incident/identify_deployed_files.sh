#!/usr/bin/env bash
# Identify EXACTLY which version of each deployed file is running.
#
# Why not `git -C /opt/xcelsior rev-parse HEAD`: deploy.sh rsyncs the deploying
# machine's *working tree*, not a git ref. HEAD there records what the deployer
# had checked out, which says nothing about file contents if the tree was dirty,
# and /opt/xcelsior may have no .git at all. Content is the only ground truth.
#
# Each file's deployed bytes are hashed with git's own blob hash, then matched
# against every version of that path in history. A match names the revision
# exactly. No match means the deployed content corresponds to no commit — the
# tree was dirty when it shipped — which is itself the answer, and the one that
# invalidates any audit assuming the repo can describe production.
#
# Usage, from a checkout of this repo, with SSH access to the VPS:
#     ./scripts/incident/identify_deployed_files.sh
#     REMOTE=linuxuser@149.28.121.61 DEPLOY_DIR=/opt/xcelsior ./scripts/incident/identify_deployed_files.sh
#
# Read-only on both ends.

set -uo pipefail

REMOTE="${REMOTE:-linuxuser@149.28.121.61}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/xcelsior}"

# The four that answer open questions:
#   routes/terminal.py  — the SSH host-key pinning window
#   api.py              — the startup-validation swallow path
#   security.py         — the committed dev JWT secret / deterministic Fernet key
#   routes/auth.py      — the operator-scope refusal (before and after deploy)
FILES=("routes/terminal.py" "api.py" "security.py" "routes/auth.py")

command -v git >/dev/null || { echo "run me from a checkout of the repo" >&2; exit 2; }
git rev-parse --git-dir >/dev/null 2>&1 || { echo "not inside a git repo" >&2; exit 2; }

echo "remote : $REMOTE:$DEPLOY_DIR"
echo "repo   : $(git rev-parse --show-toplevel)"
echo

for f in "${FILES[@]}"; do
    echo "── $f ──────────────────────────────────────────────"

    deployed_hash="$(ssh "$REMOTE" "git hash-object '$DEPLOY_DIR/$f' 2>/dev/null || sha1sum '$DEPLOY_DIR/$f' 2>/dev/null | cut -d' ' -f1")"
    if [[ -z "$deployed_hash" ]]; then
        echo "  MISSING on the server (or unreadable)"
        echo
        continue
    fi
    echo "  deployed blob : $deployed_hash"

    # Every commit that touched this path, newest first; stop at the first match.
    match=""
    while read -r c; do
        [[ -z "$c" ]] && continue
        blob="$(git rev-parse "$c:$f" 2>/dev/null)" || continue
        if [[ "$blob" == "$deployed_hash" ]]; then
            match="$c"
            break
        fi
    done < <(git log --all --format='%H' -- "$f")

    if [[ -n "$match" ]]; then
        echo "  MATCHES       : $(git log -1 --format='%h %ad %s' --date=short "$match")"
        echo "  on branches   : $(git branch -a --contains "$match" 2>/dev/null | tr -d ' *' | paste -sd, - | cut -c1-100)"
    else
        echo "  NO MATCH IN HISTORY"
        echo "  → the deployed content corresponds to no commit. The tree was dirty"
        echo "    when it shipped. Production cannot be described by a revision, and"
        echo "    any question of the form 'was fix X running' must be answered by"
        echo "    reading this file directly, not from git."
    fi
    echo
done

echo "Note: a match tells you the file's content exactly. It does NOT prove the"
echo "whole deployment is at that commit — each file is independent under rsync."
