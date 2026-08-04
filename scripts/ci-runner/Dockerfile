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

ARG RUNNER_VERSION=2.321.0
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
WORKDIR /home/runner/actions-runner

RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) rarch=x64 ;; \
         arm64) rarch=arm64 ;; \
         *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${rarch}-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && chown -R runner:runner /home/runner

USER runner
COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
ENTRYPOINT ["/home/runner/entrypoint.sh"]
