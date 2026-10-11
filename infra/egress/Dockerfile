FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends squid suricata ca-certificates \
    && rm -rf /var/lib/apt/lists/*
CMD ["squid", "-N", "-f", "/etc/xcelsior/egress/squid.conf"]
