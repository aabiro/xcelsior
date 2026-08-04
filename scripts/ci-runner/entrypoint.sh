#!/usr/bin/env bash
# One job, then gone.
#
# `--ephemeral` makes the runner unregister after a single job, so nothing a job
# leaves behind — a modified PATH, a poisoned pip cache, a lingering process — can
# reach the next one. GitHub recommends exactly this for public repositories, and a
# public repository is what this is.
set -euo pipefail

: "${RUNNER_TOKEN:?a short-lived registration token is required}"
: "${RUNNER_REPO_URL:?the repository URL is required}"

# Fail loudly if the sandbox has been undermined by the way it was launched. These
# are the two mistakes that would quietly turn this container back into the host.
if [[ -S /var/run/docker.sock ]]; then
    echo "REFUSING TO START: the docker socket is mounted." >&2
    echo "A job could then start a privileged container and reach the host." >&2
    exit 1
fi
for leaked in /home/runner/.ssh/id_rsa /home/runner/.ssh/xcelsior /workspace/.env /home/runner/.env; do
    if [[ -e "$leaked" ]]; then
        echo "REFUSING TO START: $leaked is present inside the sandbox." >&2
        echo "This runner exists so workflow code cannot reach credentials." >&2
        exit 1
    fi
done

./config.sh \
    --url "$RUNNER_REPO_URL" \
    --token "$RUNNER_TOKEN" \
    --name "${RUNNER_NAME:-xcelsior-ephemeral-$$}" \
    --labels "${RUNNER_LABELS:-self-hosted,linux,x64,sandboxed}" \
    --work /home/runner/_work \
    --ephemeral \
    --unattended \
    --replace

exec ./run.sh
