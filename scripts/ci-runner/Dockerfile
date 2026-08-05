# Self-hosted Actions runner: deliberately minimal, deliberately empty.
#
# Written rather than pulled from a third party because the whole point of this
# image is that its contents are auditable. It is the sandbox standing between a
# **public** repository's workflow code and a host that holds the production SSH
# key and a live `.env`.
#
# What is absent is the design:
#   * no docker CLI and no socket — a job cannot reach the host daemon, which also
#     means workflow `services:` do not work here (see README.md);
#   * no mount of `~/.ssh`, `.env`, or the project directory — the job clones the
#     repository itself, like any hosted runner;
#   * an unprivileged user, not in the docker group;
#   * `--ephemeral`: one job per container, then the runner unregisters.
FROM ubuntu:24.04

# Kept current deliberately, because entrypoint.sh passes `--disableupdate`.
# A container that lives for one job has nothing to gain from self-updating and
# a great deal to lose: 2.321.0 downloaded and unpacked 2.336.0 into its own
# tmpfs mid-job, filled it, and died writing its log —
# "No space left on device: _diag/Runner_....log", exit 134.
#
# Bump this when GitHub raises the minimum runner version; rebuilding the image
# is the upgrade path.
ARG RUNNER_VERSION=2.336.0
ENV DEBIAN_FRONTEND=noninteractive

# Only what the static gates need.
# `libicu74` and `libssl3` are the runner's own .NET 6 dependencies. Installed
# here explicitly rather than left to `bin/installdependencies.sh`, which runs
# later in this file — after the apt lists have been removed — and therefore
# cannot install anything. The container built cleanly and then refused to start
# with "Libicu's dependencies is missing for Dotnet Core 6.0", which is the kind
# of defect only launching it finds.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq \
        libicu74 libssl3 \
        python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 -s /bin/bash runner

# `config.sh` writes its registration material — `.credentials`,
# `.credentials_rsaparams`, `.runner`, `.env`, `.path` — into the runner's *own*
# directory, which `--read-only` refuses. The container reported "Connected to
# GitHub" and then died on `.credentials_rsaparams`, relaunched, and repeated: a
# crash loop that minted a fresh registration token every few seconds while never
# running a job, and reported zero registered runners throughout.
#
# So run-runner.sh mounts an anonymous volume over this directory. Docker
# populates such a volume from the image content beneath it, ownership included,
# so the runner simply arrives writable — no copy step, and nothing to keep in
# sync. `docker run --rm` deletes the volume with the container, which is what
# keeps the one-job-then-gone guarantee intact.
#
# A volume rather than a tmpfs because this unpacks to 674 MB: 2.336.0 bundles
# node24 with per-architecture OpenSSL headers. Holding that in RAM would take a
# fifth of this host, and the host runs the dev pool.
RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) rarch=x64 ;; \
         arm64) rarch=arm64 ;; \
         *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
       esac \
    && mkdir -p /home/runner/actions-runner /home/runner/_work \
    && curl -fsSL -o /tmp/runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${rarch}-${RUNNER_VERSION}.tar.gz" \
    && tar xzf /tmp/runner.tar.gz -C /home/runner/actions-runner \
    && rm /tmp/runner.tar.gz \
    && chown -R runner:runner /home/runner

USER runner
WORKDIR /home/runner/actions-runner
COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
ENTRYPOINT ["/home/runner/entrypoint.sh"]
