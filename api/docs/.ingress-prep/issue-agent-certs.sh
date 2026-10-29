#!/usr/bin/env bash
# Issue the client CA and per-worker certs the agent gateway needs.
#
# NOT RUN. Nothing here has been executed, and it generates no production key
# material until you run it deliberately with OUT set to a real path.
#
# `nginx/agent-xcelsior.conf` sets `ssl_verify_client on` against
# /etc/nginx/certs/agent-client-ca.pem. None of the three certs it needs exist
# on the box, so enabling that vhost today takes over the hostname and refuses
# every agent — worse than the current 410. This is the missing prerequisite.
#
#   OUT=/secure/path ./issue-agent-certs.sh ca
#   OUT=/secure/path ./issue-agent-certs.sh worker aaryn-tuf-rtx2060
#
# The CA key never leaves OUT. Only <host>.crt/<host>.key go to a worker, and
# only agent-client-ca.pem goes to nginx.
set -euo pipefail

OUT="${OUT:-}"
[ -n "$OUT" ] || { echo "set OUT to a directory you control; refusing to write to cwd" >&2; exit 2; }
mkdir -p "$OUT"; chmod 700 "$OUT"

CA_KEY="$OUT/agent-client-ca.key"
CA_CRT="$OUT/agent-client-ca.pem"
DAYS_CA=1825
DAYS_WORKER=365

case "${1:-}" in
  ca)
    [ -f "$CA_KEY" ] && { echo "CA already exists at $CA_KEY — refusing to overwrite" >&2; exit 3; }
    openssl genrsa -out "$CA_KEY" 4096
    chmod 600 "$CA_KEY"
    openssl req -x509 -new -key "$CA_KEY" -sha256 -days "$DAYS_CA" \
      -subj "/O=Xcelsior/CN=Xcelsior Agent Client CA" -out "$CA_CRT"
    echo "CA written: $CA_CRT (public — this is what nginx needs)"
    echo "CA key:     $CA_KEY (never copy this anywhere)"
    ;;
  worker)
    HOST_ID="${2:-}"
    [ -n "$HOST_ID" ] || { echo "usage: $0 worker <host_id>" >&2; exit 2; }
    [ -f "$CA_KEY" ] || { echo "run '$0 ca' first" >&2; exit 3; }
    KEY="$OUT/$HOST_ID.key"; CSR="$OUT/$HOST_ID.csr"; CRT="$OUT/$HOST_ID.crt"
    openssl genrsa -out "$KEY" 2048
    chmod 600 "$KEY"
    # CN carries the host id: agent-xcelsior.conf maps $ssl_client_s_dn to an
    # agent host id, so the CN is the identity, not decoration.
    openssl req -new -key "$KEY" -subj "/O=Xcelsior/CN=$HOST_ID" -out "$CSR"
    openssl x509 -req -in "$CSR" -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
      -days "$DAYS_WORKER" -sha256 -out "$CRT"
    rm -f "$CSR"
    echo "worker cert: $CRT"
    echo "worker key:  $KEY  (copy both to $HOST_ID only)"
    ;;
  *)
    echo "usage: OUT=/secure/path $0 {ca|worker <host_id>}" >&2; exit 2 ;;
esac
