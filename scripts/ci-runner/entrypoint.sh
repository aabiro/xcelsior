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

# `--read-only` makes the container root immutable, but `config.sh` writes its
# registration material into the runner's own directory: `.credentials`,
# `.credentials_rsaparams`, `.runner`, `.env`, `.path`. run-runner.sh therefore
# mounts a writable anonymous volume over it.
#
# Asserted rather than assumed, because the failure it prevents does not look
# like a failure. Without the mount the container still prints "Connected to
# GitHub" before dying on the first unwritable file, and `--loop` turns that into
# a crash loop that mints a registration token every few seconds and never runs a
# job. GitHub reports zero registered runners throughout, which reads exactly
# like a runner nobody ever started.
RUNNER_DIR=/home/runner/actions-runner
if [[ "$(stat -c %d "$RUNNER_DIR")" == "$(stat -c %d /home/runner)" ]]; then
    echo "REFUSING TO START: $RUNNER_DIR is not a separate mount." >&2
    echo "It shares a device with /home/runner, so it is the read-only root and" >&2
    echo "registration will fail partway. Launch via scripts/ci-runner/run-runner.sh," >&2
    echo "which supplies the mount." >&2
    exit 1
fi

# Mounted is not the same as writable. A `--tmpfs` arrives owned by root:root
# unless uid/gid are passed, and this container runs as uid 10001 — which fails
# as eight `cp: Permission denied` lines rather than as one legible error.
for writable in "$RUNNER_DIR" /home/runner/_work; do
    if [[ ! -w "$writable" ]]; then
        echo "REFUSING TO START: $writable is not writable by uid $(id -u)." >&2
        echo "Its tmpfs needs uid=10001,gid=10001 — a mount without them is owned" >&2
        echo "by root and the Dockerfile's chown is hidden underneath it." >&2
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
    --replace \
    `# One job, then this container is gone, so there is nothing for a` \
    `# self-update to improve. Left enabled it unpacks a whole second runner` \
    `# into this tmpfs mid-job and fills it. The image is the upgrade unit —` \
    `# bump RUNNER_VERSION in the Dockerfile.` \
    --disableupdate

exec ./run.sh
