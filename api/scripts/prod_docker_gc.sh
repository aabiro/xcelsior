#!/usr/bin/env bash
# Weekly Docker trim for prod-hetzner. Installed as /usr/local/sbin/prod-docker-gc
# and run by api/infra/systemd/prod-docker-gc.timer.
#
#   prod-docker-gc [--dry-run]
#
# Removes, and only removes:
#   - images no container (running or stopped) uses, older than 14 days, whose
#     tag is not a deliberate rollback point (XCELSIOR_GC_KEEP_RE);
#   - dangling layers;
#   - build cache older than 7 days.
# Volumes are never touched. Skips the run while a deploy or build is going.
set -Eeuo pipefail

readonly IMAGE_MAX_AGE_HOURS="${XCELSIOR_GC_IMAGE_AGE_HOURS:-336}"
readonly CACHE_MAX_AGE_HOURS="${XCELSIOR_GC_CACHE_AGE_HOURS:-168}"
# Rollback points are kept however old they get.
readonly KEEP_RE="${XCELSIOR_GC_KEEP_RE:-(cutover-|rollout-backup|preserved-live|before-security-hold|-blue)}"
readonly LOCK="${XCELSIOR_DEPLOY_LOCK:-/run/lock/xcelsior-deploy.lock}"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

log() { printf 'prod-docker-gc: %s\n' "$*"; }

if pgrep -f 'docker(-| )(compose|buildx|build|pull)( |$)' >/dev/null; then
  log "a deploy or build is running; skipping"
  exit 0
fi
exec 9>"${LOCK}"
if ! flock -n 9; then
  log "deploy lock ${LOCK} is held; skipping"
  exit 0
fi

avail_before="$(df --output=avail -B1 / | tail -1)"
in_use="$(docker ps -aq | xargs -r docker inspect --format '{{.Image}}' | sort -u)"
# RFC 3339 UTC strings compare correctly as text; no date parsing needed.
cutoff="$(date -u -d "@$(( $(date +%s) - IMAGE_MAX_AGE_HOURS * 3600 ))" +%Y-%m-%dT%H:%M:%S)"

removed=0
kept_rollback=0
while IFS='|' read -r id ref; do
  if grep -qxF "${id}" <<<"${in_use}"; then
    continue
  fi
  if [[ "${ref}" =~ ${KEEP_RE} ]]; then
    kept_rollback=$((kept_rollback + 1))
    continue
  fi
  created="$(docker image inspect --format '{{.Created}}' "${id}")"
  [[ "${created:0:19}" < "${cutoff}" ]] || continue
  if ((DRY_RUN)); then
    log "would remove ${ref} (${id:7:12}, created ${created:0:10})"
  else
    # Untag by name so other tags on the same image survive; dangling by ID.
    if [[ "${ref}" == "<none>:<none>" ]]; then
      docker rmi "${id}" >/dev/null 2>&1 || continue
    else
      docker rmi "${ref}" >/dev/null 2>&1 || continue
    fi
  fi
  removed=$((removed + 1))
done < <(docker image ls --no-trunc --format '{{.ID}}|{{.Repository}}:{{.Tag}}')

if ((DRY_RUN)); then
  log "dry run: ${removed} image tag(s) would go, ${kept_rollback} rollback tag(s) kept"
  docker builder du 2>/dev/null | tail -1 || true
  exit 0
fi

docker image prune -f >/dev/null
docker builder prune -af --filter "until=${CACHE_MAX_AGE_HOURS}h" >/dev/null

avail_after="$(df --output=avail -B1 / | tail -1)"
log "removed ${removed} image tag(s), kept ${kept_rollback} rollback tag(s), freed $(( (avail_after - avail_before) / 1048576 )) MiB on /"
