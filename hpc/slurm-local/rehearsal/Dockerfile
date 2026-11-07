# Container rehearsal of the xcl-local cluster. Runs the kit's own install
# steps (users, packages; configure and db at start) on Ubuntu 26.04, so
# the configs and scripts are exercised before they touch real machines.
FROM ubuntu:26.04
ARG ROLE=tower
ENV DEBIAN_FRONTEND=noninteractive XCL_NO_SYSTEMD=1
# The image's built-in "ubuntu" user owns uid 1000; jobs must run as
# aaryn (1000) on both nodes, exactly as on the real machines.
RUN userdel -r ubuntu 2>/dev/null || true \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends python3 iproute2 openssh-server sudo procps ca-certificates kmod \
 && useradd -u 1000 -m -s /bin/bash aaryn
COPY scripts /kit/scripts
COPY conf /kit/conf
RUN if [ "$ROLE" = tower ]; then /kit/scripts/10-install-tower.sh users packages; \
    else /kit/scripts/20-install-asus.sh users packages; fi \
 && cp /var/lib/xcl-slurm-test/manifest /var/lib/xcl-slurm-test/manifest.build
COPY rehearsal/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
