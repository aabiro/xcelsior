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

# Refuse to run if ANY workflow that targets this runner can be triggered from a
# fork. That is the whole threat model, and checking it here means the check
# cannot be forgotten.
#
# This used to inspect `gates-sandboxed.yml` alone, which was correct while that
# was the only workflow using the runner. It is not any more: `frontend.yml` and
# `mcp.yml` moved here when hosted Actions stayed blocked. A refusal that names
# one file while the runner serves three is a check that looks in the wrong
# place — so the set is *discovered* from the workflows rather than listed.
mapfile -t _sandboxed_workflows < <(
    git -C "$HERE/../.." grep -lE 'runs-on:\s*\[\s*self-hosted\s*,\s*sandboxed\s*\]' \
        HEAD -- .github/workflows 2>/dev/null | sed 's|^HEAD:||' || true
)

if [[ ${#_sandboxed_workflows[@]} -eq 0 ]]; then
    echo "WARNING: no workflow on HEAD targets [self-hosted, sandboxed]." >&2
    echo "Either they were renamed or this runner is serving nothing." >&2
fi

for _wf_path in "${_sandboxed_workflows[@]}"; do
    _wf="$(git -C "$HERE/../.." show "HEAD:$_wf_path" 2>/dev/null || true)"
    [[ -n "$_wf" ]] || continue
    # `on:` triggers only — a `paths:` entry mentioning pull_request, or a job
    # named for one, is not a trigger. Anchored at two-space indent, which is
    # where a trigger sits under `on:`.
    if grep -qE '^\s{0,2}pull_request(_target)?\s*:' <<<"$_wf"; then
        echo "REFUSING TO START: $_wf_path has a pull_request trigger and runs on" >&2
        echo "this runner. On a public repository that lets anyone's PR run code here." >&2
        exit 1
    fi
done
echo "fork-trigger check: ${#_sandboxed_workflows[@]} workflow(s) target this runner, none fork-triggerable"

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
