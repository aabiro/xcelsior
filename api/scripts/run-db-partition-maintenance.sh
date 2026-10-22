#!/usr/bin/env bash
# Native one-shot job. /etc/xcelsior/db-maintenance.env supplies only the
# migration credential; the long-running API service never receives it.
set -euo pipefail
metrics_dir="${XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR:-/var/lib/node_exporter/textfile_collector}"
write_metric() {
    local name="$1" temporary
    shift
    mkdir -p "$metrics_dir" && temporary="$(mktemp "$metrics_dir/.partition-maintenance.XXXXXX")" || return 1
    printf '%s\n' "$@" > "$temporary" && chmod 0644 "$temporary" &&
        mv -f "$temporary" "$metrics_dir/$name"
}
record_outcome() {
    local result=$? success=0
    trap - EXIT
    [[ "$result" == "0" ]] && success=1
    write_metric xcelsior_partition_maintenance_status.prom \
        '# TYPE xcelsior_partition_maintenance_last_run_success gauge' \
        "xcelsior_partition_maintenance_last_run_success $success" ||
        echo 'could not publish partition maintenance outcome' >&2
    if [[ "$success" == "1" ]]; then
        write_metric xcelsior_partition_maintenance_success.prom \
            '# TYPE xcelsior_partition_maintenance_last_success_timestamp_seconds gauge' \
            "xcelsior_partition_maintenance_last_success_timestamp_seconds $(date +%s)" ||
            echo 'could not publish partition maintenance freshness' >&2
    fi
    exit "$result"
}
trap record_outcome EXIT
[[ -n "${XCELSIOR_POSTGRES_DSN_MIGRATOR:-}" ]] || {
    echo 'XCELSIOR_POSTGRES_DSN_MIGRATOR is required' >&2
    exit 1
}
export XCELSIOR_POSTGRES_DSN_MIGRATOR
# Reuse the exact running API image, including after a blue-green handoff.
# Refuse an in-progress handoff with two APIs rather than choosing an image.
mapfile -t api_containers < <(
    docker ps --filter label=com.docker.compose.project=xcelsior \
        --format '{{.ID}} {{.Label "com.docker.compose.service"}}' |
        awk '$2 == "api" || $2 == "api-blue" {print $1}'
)
[[ "${#api_containers[@]}" == "1" ]] || {
    echo 'partition maintenance requires exactly one running API generation' >&2
    exit 1
}
image="$(docker inspect --format '{{.Image}}' "${api_containers[0]}")"
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo 'current API image identity could not be verified' >&2
    exit 1
}
docker run --rm --network host --read-only --no-healthcheck --cap-drop ALL \
    --security-opt no-new-privileges:true --user 10001:10001 \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
    -e XCELSIOR_POSTGRES_DSN_MIGRATOR \
    --entrypoint python "$image" api/scripts/maintain_db_partitions.py
