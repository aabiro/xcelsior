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
#
# The same mistake one level up: the workflows read must be those of the
# repository the runner is about to register for. `XCELSIOR_CI_REPO` points this
# runner at another repository (aabiro/aabiro.github.io deploys from it), and
# reading this checkout then validated xcelsior while serving something else. So
# this checkout is still read from HEAD, and any other repository is read from
# its default branch through the API. If that read fails, nothing can be said
# about fork triggers, so the runner does not start.
SANDBOXED_RE='runs-on:\s*\[\s*self-hosted\s*,\s*sandboxed\s*\]'
_checkout_repo="$(git -C "$HERE/../.." remote get-url origin 2>/dev/null \
    | sed -E 's#\.git$##; s#^.*github\.com[:/]##' || true)"

if [[ "${REPO,,}" == "${_checkout_repo,,}" ]]; then
    _wf_source="HEAD of this checkout"
    list_workflows() { git -C "$HERE/../.." ls-tree --name-only HEAD .github/workflows/ 2>/dev/null || true; }
    show_workflow() { git -C "$HERE/../.." show "HEAD:$1" 2>/dev/null || true; }
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
