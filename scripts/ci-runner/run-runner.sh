#!/usr/bin/env bash
# Start one ephemeral, sandboxed Actions runner. Re-run for each job, or loop it.
#
#     ./scripts/ci-runner/run-runner.sh            # one job, then exit
#     ./scripts/ci-runner/run-runner.sh --loop     # keep serving jobs
#
# The registration token is fetched fresh from the API each time and expires in an
# hour. It is not a repository secret and is never written to disk.
#
# READ scripts/ci-runner/README.md BEFORE ENABLING THIS. It is safe only in
# combination with a workflow that fork pull requests cannot trigger.
set -euo pipefail

REPO="${XCELSIOR_CI_REPO:-aabiro/xcelsior}"
IMAGE="xcelsior-ci-runner"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v gh >/dev/null || { echo "gh CLI is required to mint a registration token" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

# Refuse to run if the workflow this serves can be triggered from a fork. That is
# the whole threat model, and checking it here means the check cannot be forgotten.
wf="$(git -C "$HERE/../.." show HEAD:.github/workflows/gates-sandboxed.yml 2>/dev/null || true)"
if [[ -z "$wf" ]]; then
    echo "WARNING: .github/workflows/gates-sandboxed.yml not found on HEAD." >&2
    echo "This runner is only safe alongside a workflow forks cannot trigger." >&2
elif grep -qE '^\s*pull_request(_target)?\s*:' <<<"$wf"; then
    echo "REFUSING TO START: gates-sandboxed.yml has a pull_request trigger." >&2
    echo "On a public repository that lets anyone's PR run code on this runner." >&2
    exit 1
fi

build() {
    echo "building $IMAGE (auditable by design — see the Dockerfile)"
    docker build -q -t "$IMAGE" "$HERE" >/dev/null
}

one_job() {
    local token
    token="$(gh api -X POST "repos/$REPO/actions/runners/registration-token" -q .token)"
    [[ -n "$token" ]] || { echo "could not mint a registration token" >&2; exit 1; }
    echo "starting ephemeral runner for $REPO (one job, then unregisters)"
    docker run --rm \
        --name "xcelsior-ci-runner-$$" \
        --network bridge \
        --read-only \
        --tmpfs /tmp:rw,noexec,nosuid \
        --tmpfs /home/runner/_work:rw,exec \
        --tmpfs /home/runner/actions-runner/_diag:rw \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --pids-limit 512 \
        --memory 6g \
        --cpus 6 \
        -e RUNNER_REPO_URL="https://github.com/$REPO" \
        -e RUNNER_TOKEN="$token" \
        "$IMAGE"
}

build
if [[ "${1:-}" == "--loop" ]]; then
    while true; do one_job || true; sleep 5; done
else
    one_job
fi
