FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      nfs-ganesha nfs-ganesha-vfs cryptsetup-bin e2fsprogs util-linux dbus \
      rpcbind netbase \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/log/ganesha /exports/volumes /run/dbus /run/rpcbind \
 && touch /run/rpcbind/rpcbind.xdr /run/rpcbind/portmap.xdr \
 # Ganesha looks in /usr/lib/ganesha; Debian puts FSAL plugins under triplet paths.
 && for d in /usr/lib/*-linux-gnu/ganesha; do \
      [ -d "$d" ] || continue; \
      for f in "$d"/*.so; do \
        ln -sf "$f" "/usr/lib/ganesha/$(basename "$f")"; \
      done; \
    done

COPY ganesha.conf /etc/ganesha/ganesha.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 2049 111 20048

ENTRYPOINT ["/entrypoint.sh"]