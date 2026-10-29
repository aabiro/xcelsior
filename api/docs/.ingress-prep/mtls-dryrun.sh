#!/usr/bin/env bash
# Local proof that `ssl_verify_client on` does what the vhost claims.
#
# Runs entirely in a throwaway container on this machine. Touches no server,
# no production certificate, no DNS. Generates a self-signed CA in a temp dir
# and destroys it on exit.
#
# Three cases, each of which must fail independently — a config that accepts a
# valid cert proves nothing on its own if it also accepts a missing one.
#
#   1. valid client cert    -> 200
#   2. no client cert       -> refused
#   3. cert from a wrong CA -> refused
#
# Run before the cutover, not as part of it.
set -uo pipefail

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; docker rm -f xcl-mtls-dryrun >/dev/null 2>&1 || true' EXIT
cd "$WORK"

echo "── generating throwaway CAs and certs ──"
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 1 \
  -subj "/CN=dryrun-ca" >/dev/null 2>&1
openssl req -x509 -newkey rsa:2048 -nodes -keyout other-ca.key -out other-ca.pem -days 1 \
  -subj "/CN=other-ca" >/dev/null 2>&1
openssl req -x509 -newkey rsa:2048 -nodes -keyout server.key -out server.pem -days 1 \
  -subj "/CN=localhost" >/dev/null 2>&1

for pair in "client:ca" "impostor:other-ca"; do
  name="${pair%%:*}"; signer="${pair##*:}"
  openssl req -newkey rsa:2048 -nodes -keyout "$name.key" -out "$name.csr" \
    -subj "/CN=$name" >/dev/null 2>&1
  openssl x509 -req -in "$name.csr" -CA "$signer.pem" -CAkey "$signer.key" \
    -CAcreateserial -days 1 -out "$name.crt" >/dev/null 2>&1
done

cat > nginx.conf <<'CONF'
events {}
http {
  server {
    listen 8443 ssl;
    server_name localhost;
    ssl_certificate     /certs/server.pem;
    ssl_certificate_key /certs/server.key;
    # The two directives under test, verbatim from nginx/agent-xcelsior.conf.
    ssl_client_certificate /certs/ca.pem;
    ssl_verify_client on;
    location / { return 200 "verified\n"; }
  }
}
CONF

docker rm -f xcl-mtls-dryrun >/dev/null 2>&1 || true
docker run -d --name xcl-mtls-dryrun -p 18443:8443 \
  -v "$WORK/nginx.conf:/etc/nginx/nginx.conf:ro" \
  -v "$WORK:/certs:ro" nginx:alpine >/dev/null
sleep 3

pass=0; fail=0
check() {  # name expected_regex command...
  local name="$1" expect="$2"; shift 2
  local out; out="$("$@" 2>&1 || true)"
  if grep -qE "$expect" <<<"$out"; then
    echo "  PASS  $name"; pass=$((pass+1))
  else
    echo "  FAIL  $name — got: $(head -c 120 <<<"$out")"; fail=$((fail+1))
  fi
}

echo "── the three cases ──"
check "valid client cert is accepted" "verified" \
  curl -s --cacert server.pem --cert client.crt --key client.key \
       --resolve localhost:18443:127.0.0.1 https://localhost:18443/
check "missing client cert is refused" "400|SSL|handshake|alert" \
  curl -s --cacert server.pem --resolve localhost:18443:127.0.0.1 https://localhost:18443/
check "wrong-CA client cert is refused" "400|SSL|handshake|alert" \
  curl -s --cacert server.pem --cert impostor.crt --key impostor.key \
       --resolve localhost:18443:127.0.0.1 https://localhost:18443/

echo "── $pass passed, $fail failed ──"
[ "$fail" -eq 0 ]
