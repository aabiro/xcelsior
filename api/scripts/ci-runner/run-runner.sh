#!/usr/bin/env bash
# Start one ephemeral, sandboxed Actions runner. Re-run for each job, or loop it.
#
#     ./api/scripts/ci-runner/run-runner.sh            # one job, then exit
#     ./api/scripts/ci-runner/run-runner.sh --loop --workers 2
#
# The registration token is fetched fresh from the API each time and expires in an
# hour. It is not a repository secret and is never written to disk.
#
# READ api/scripts/ci-runner/README.md BEFORE ENABLING THIS. It is safe only in
# combination with a workflow that fork pull requests cannot trigger.
set -euo pipefail

REPO="${XCELSIOR_CI_REPO:-aabiro/xcelsior}"
IMAGE="xcelsior-ci-runner"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=once
workers=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --loop) mode=loop; shift ;;
        --workers)
            [[ "${2:-}" == 1 || "${2:-}" == 2 ]] || {
                echo '--workers must be 1 or 2' >&2; exit 2;
            }
            workers="$2"; shift 2 ;;
        *) echo "Unknown runner option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$workers" != 1 && "$mode" != loop ]]; then
    echo '--workers requires --loop' >&2
    exit 2
fi

command -v gh >/dev/null || { echo "gh CLI is required to mint a registration token" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

# Refuse to run if ANY workflow that targets this runner can be triggered from a
# fork. That is the whole threat model, and checking it here means the check
# cannot be forgotten.
#
# This used to inspect `gates-sandboxed.yml` alone, which was correct while that
# was the only workflow using the runner. It is not any more: `frontend.yml` and
# `mcp.yml` moved here when hosted Actions stayed blocked. A refusal that names
# one file while the runner serves three is a check that looks in the wrong
# place — so the set is *discovered* from the workflows rather than listed.
#
# The same mistake one level up: the workflows read must be those of the
# repository the runner is about to register for. `XCELSIOR_CI_REPO` points this
# runner at another repository (aabiro/aabiro.github.io deploys from it), and
# reading this checkout then validated xcelsior while serving something else. So
# this checkout is still read from HEAD, and any other repository is read from
# its default branch through the API. If that read fails, nothing can be said
# about fork triggers, so the runner does not start.
SANDBOXED_RE='runs-on:\s*\[\s*self-hosted\s*,\s*sandboxed\s*\]'
_checkout_repo="$(git -C "$HERE/../../.." remote get-url origin 2>/dev/null \
    | sed -E 's#\.git$##; s#^.*github\.com[:/]##' || true)"

if [[ "${REPO,,}" == "${_checkout_repo,,}" ]]; then
    _wf_source="HEAD of this checkout"
    list_workflows() { git -C "$HERE/../../.." ls-tree --name-only HEAD .github/workflows/ 2>/dev/null || true; }
    show_workflow() { git -C "$HERE/../../.." show "HEAD:$1" 2>/dev/null || true; }
else
    _wf_source="the default branch of $REPO"
    list_workflows() { gh api "repos/$REPO/contents/.github/workflows" -q '.[] | select(.type == "file") | .path'; }
    show_workflow() { gh api -H "Accept: application/vnd.github.raw+json" "repos/$REPO/contents/$1"; }
fi

if ! _wf_paths="$(list_workflows)"; then
    echo "REFUSING TO START: could not list the workflows in $_wf_source." >&2
    echo "Without them there is no telling whether a fork's PR can reach this runner." >&2
    exit 1
fi

_sandboxed_workflows=()
while IFS= read -r _wf_path; do
    [[ "$_wf_path" == *.yml || "$_wf_path" == *.yaml ]] || continue
    if ! _wf="$(show_workflow "$_wf_path")"; then
        echo "REFUSING TO START: could not read $_wf_path from $_wf_source." >&2
        exit 1
    fi
    grep -qE "$SANDBOXED_RE" <<<"$_wf" || continue
    _sandboxed_workflows+=("$_wf_path")
    # `on:` triggers only — a `paths:` entry mentioning pull_request, or a job
    # named for one, is not a trigger. Anchored at two-space indent, which is
    # where a trigger sits under `on:`.
    if grep -qE '^\s{0,2}pull_request(_target)?\s*:' <<<"$_wf"; then
        echo "REFUSING TO START: $_wf_path in $REPO has a pull_request trigger and runs on" >&2
        echo "this runner. On a public repository that lets anyone's PR run code here." >&2
        exit 1
    fi
done <<<"$_wf_paths"

if [[ ${#_sandboxed_workflows[@]} -eq 0 ]]; then
    echo "WARNING: no workflow in $_wf_source targets [self-hosted, sandboxed]." >&2
    echo "Either they were renamed or this runner is serving nothing." >&2
fi
echo "fork-trigger check: ${#_sandboxed_workflows[@]} workflow(s) in $REPO target this runner, none fork-triggerable"

build() {
    local version cache_dir
    # Check official release metadata before registration. The one-hour API
    # cache bounds requests; a deprecated idle listener refreshes on reconnect.
    version=$(gh api --cache 1h repos/actions/runner/releases/latest \
        -q '.tag_name | ltrimstr("v")') || return 1
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
        echo 'Invalid official runner release version' >&2; return 1;
    }
    cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/xcelsior-ci"
    mkdir -p -m 0700 "$cache_dir"
    (
        flock -x 9
        installed=$(docker image inspect "$IMAGE" \
            --format '{{index .Config.Labels "ca.xcelsior.runner.version"}}' 2>/dev/null || true)
        if [[ "$installed" != "$version" ]]; then
            echo "building $IMAGE with official runner $version"
            docker build -q --build-arg "RUNNER_VERSION=$version" -t "$IMAGE" "$HERE" >/dev/null
        fi
    ) 9>"$cache_dir/image-update.lock"
}

one_job() {
    local token runner_name runner_host runner_id status=0
    build || return 1
    # PID 1 is identical inside every ephemeral container. Generate the name
    # on the host so a second executor cannot replace the first registration.
    runner_host=$(hostname -s)
    runner_name="xcelsior-${runner_host:0:16}-${BASHPID}-${RANDOM}"
    token="$(gh api -X POST "repos/$REPO/actions/runners/registration-token" -q .token)"
    [[ -n "$token" ]] || { echo "could not mint a registration token" >&2; return 1; }
    echo "starting ephemeral runner for $REPO (one job, then unregisters)"
    docker run --rm \
        --name "$runner_name" \
        --label "xcelsior-ci-pool=$$" \
        --init \
        --network bridge \
        --read-only \
        `# tmpfs pages are host RAM, so this one is sized explicitly: an unbounded` \
        `# mount is a way for a job to take the box down, and this host runs the` \
        `# dev pool. uid/gid because a --tmpfs arrives owned by root:root and the` \
        `# container is uid 10001 — the Dockerfile's chown is hidden underneath it.` \
        --tmpfs /tmp:rw,noexec,nosuid,size=1g,uid=10001,gid=10001 \
        `# The workspace and the package caches are volumes, not tmpfs: a checkout` \
        `# plus \`uv sync\` of 118 packages does not belong in RAM on a 15 GB box.` \
        `# Both are anonymous, so --rm destroys them with the container and no job` \
        `# can leave anything for the next one.` \
        --mount type=volume,dst=/home/runner/_work \
        --mount type=volume,dst=/home/runner/.cache \
        `# The runner's own directory, not just its _diag subdirectory: config.sh` \
        `# writes .credentials, .credentials_rsaparams, .runner, .env and .path` \
        `# beside its binaries, and --read-only refuses all of them.` \
        `#` \
        `# An anonymous volume, not a tmpfs: Docker populates it from the image` \
        `# content underneath (ownership included, so no chown or copy is needed),` \
        `# and --rm deletes it with the container. The runner unpacks to 674 MB,` \
        `# which is not worth spending host RAM on.` \
        --mount type=volume,dst=/home/runner/actions-runner \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --pids-limit 512 \
        --memory 4g \
        --cpus 4 \
        -e RUNNER_REPO_URL="https://github.com/$REPO" \
        -e RUNNER_TOKEN="$token" \
        -e RUNNER_NAME="$runner_name" \
        "$IMAGE" || status=$?
    # A runner rejected before its first job never reaches ephemeral removal.
    # Remove only this executor's exact registration, including fatal starts.
    runner_id=$(gh api "repos/$REPO/actions/runners" --paginate \
        -q ".runners[] | select(.name == \"$runner_name\") | .id") || return 1
    if [[ -n "$runner_id" ]]; then
        gh api -X DELETE "repos/$REPO/actions/runners/$runner_id" >/dev/null || return 1
    fi
    return "$status"
}

if [[ "$mode" == loop ]]; then
    worker_pids=()
    stop_workers() {
        trap - TERM INT
        kill -TERM "${worker_pids[@]}" 2>/dev/null || true
        local containers
        containers=$(docker ps -q --filter "label=xcelsior-ci-pool=$$")
        if [[ -n "$containers" ]]; then
            # Only containers belonging to this supervisor; never other runners.
            docker stop --time 30 $containers >/dev/null 2>&1 || true
        fi
        wait "${worker_pids[@]}" 2>/dev/null || true
        exit 0
    }
    trap stop_workers TERM INT
    for ((slot = 1; slot <= workers; slot++)); do
        (while true; do one_job || true; sleep 30; done) &
        worker_pids+=("$!")
    done
    # An unexpected worker exit must reach systemd, which restarts the pool.
    wait -n "${worker_pids[@]}" || true
    stop_workers
else
    one_job
fi
