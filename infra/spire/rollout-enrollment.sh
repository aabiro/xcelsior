#!/usr/bin/env bash
# Production rollout of host enrollment (issue #67) on prod-hetzner. Idempotent;
# run from the repository root as a sudo-capable user:
#
#   bash infra/spire/rollout-enrollment.sh
#
# 1. SPIRE datastore: its own role and database on the local Postgres.
# 2. Long-lived upstream root that signs SPIRE's rotating CA.
# 3. SPIRE server (docker-compose.spire-server.yml) on the mesh address, port 8095.
# 4. The agent gateway also trusts that root. Existing hosts' certificates are
#    unaffected: their CA stays in the bundle.
# 5. A Headscale API key, so the API can mint single-use mesh keys.
# 6. Writes the API's settings to a root-only file and prints what to add to the
#    API's compose override. Deploying the API is the next step, not this one.
#
# Secrets (datastore password, Headscale key) are written only to root-only
# files under /etc/xcelsior/spire/ and never printed.
set -euo pipefail

cd "$(dirname "$0")/../.."
ETC=/etc/xcelsior/spire
VAR=/var/lib/xcelsior-spire
NGINX_BUNDLE=/etc/nginx/certs/agent-client-ca.pem
MESH_ADDRESS=${SPIRE_BIND_ADDRESS:-100.64.0.1}
SPIRE_UID=10002
API_GID=10001 # the API container runs as 10001:10001
IMAGE=ghcr.io/spiffe/spire-server:1.11.2
COMPOSE="docker compose --env-file $ETC/datastore.env -f infra/spire/docker-compose.spire-server.yml"

say() { printf '\n== %s\n' "$*"; }
fail() { echo "refusing: $*" >&2; exit 1; }

say "preflight"
ip -4 addr show | grep -q "inet $MESH_ADDRESS/" || fail "$MESH_ADDRESS is not on this host (mesh down?)"
if ss -ltnH "( sport = :8095 )" | grep -q . && ! docker ps --format '{{.Names}}' | grep -q '^xcelsior-spire-server-spire-server-1$'; then
    fail "port 8095 is taken by something other than the SPIRE server"
fi
command -v psql >/dev/null && command -v openssl >/dev/null && command -v docker >/dev/null || fail "psql, openssl and docker are required"
sudo -n true || fail "needs sudo"
sudo install -d -m 0755 -o root -g root "$ETC"

say "1. datastore"
if sudo test -s "$ETC/datastore.env"; then
    echo "kept $ETC/datastore.env"
else
    pg() { sudo -u postgres psql -v ON_ERROR_STOP=1 -tAc "$1"; }
    [ "$(pg "select 1 from pg_roles where rolname='spire'")" != "1" ] \
        || fail "role 'spire' exists but $ETC/datastore.env does not; set its password and write that file by hand"
    password="$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)"
    pg "create role spire with login password '$password'" >/dev/null
    [ "$(pg "select 1 from pg_database where datname='spire'")" = "1" ] || pg "create database spire owner spire" >/dev/null
    pg "revoke all on database spire from public" >/dev/null
    umask 077
    printf 'SPIRE_DATASTORE_DSN=dbname=spire user=spire password=%s host=127.0.0.1 port=5432 sslmode=disable\nSPIRE_BIND_ADDRESS=%s\n' \
        "$password" "$MESH_ADDRESS" | sudo tee "$ETC/datastore.env" >/dev/null
    unset password
    sudo chmod 0600 "$ETC/datastore.env"
    echo "role and database 'spire' created; DSN in $ETC/datastore.env (root, 0600)"
fi

say "2. upstream root"
if sudo test -s "$ETC/upstream-root.crt"; then
    echo "kept $ETC/upstream-root.crt"
else
    tmp="$(mktemp -d)"
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 3650 \
        -subj "/C=CA/O=Xcelsior/CN=Xcelsior SPIRE Root" \
        -addext "basicConstraints=critical,CA:TRUE,pathlen:1" \
        -addext "keyUsage=critical,keyCertSign,cRLSign" \
        -keyout "$tmp/key" -out "$tmp/crt" 2>/dev/null
    sudo install -m 0400 -o "$SPIRE_UID" -g "$SPIRE_UID" "$tmp/key" "$ETC/upstream-root.key"
    sudo install -m 0444 -o root -g root "$tmp/crt" "$ETC/upstream-root.crt"
    rm -rf "$tmp"
    echo "created; the key is readable only by the SPIRE server's uid $SPIRE_UID"
fi
openssl x509 -in "$ETC/upstream-root.crt" -noout -subject -enddate

say "3. SPIRE server"
sudo install -d -m 0755 -o root -g root "$VAR"
sudo install -d -m 0700 -o "$SPIRE_UID" -g "$SPIRE_UID" "$VAR/server"
sudo install -d -m 0750 -o "$SPIRE_UID" -g "$API_GID" "$VAR/sock"
sudo $COMPOSE up -d
for _ in $(seq 1 60); do
    sudo docker run --rm --user "$SPIRE_UID:$API_GID" --network none -v "$VAR/sock:/run/xcelsior-spire" \
        --entrypoint /opt/spire/bin/spire-server "$IMAGE" healthcheck -socketPath /run/xcelsior-spire/api.sock \
        >/dev/null 2>&1 && break
    sleep 2
done
sudo docker run --rm --user "$SPIRE_UID:$API_GID" --network none -v "$VAR/sock:/run/xcelsior-spire" \
    --entrypoint /opt/spire/bin/spire-server "$IMAGE" healthcheck -socketPath /run/xcelsior-spire/api.sock
ss -ltnH "( sport = :8095 )" | awk '{print "listening on " $4}'
ss -ltnH "( sport = :8095 )" | grep -qv "$MESH_ADDRESS:8095" && fail "8095 is reachable beyond $MESH_ADDRESS"

# The API runs spire-server's CLI against the socket; copy the binary out of the
# same pinned image rather than downloading it separately.
sudo install -d -m 0755 -o root -g root /usr/local/lib/xcelsior
cid="$(sudo docker create "$IMAGE")"
sudo docker cp "$cid:/opt/spire/bin/spire-server" /tmp/spire-server.$$ >/dev/null
sudo docker rm "$cid" >/dev/null
sudo install -m 0755 -o root -g root /tmp/spire-server.$$ /usr/local/lib/xcelsior/spire-server
sudo rm -f /tmp/spire-server.$$
echo "spire-server CLI for the API: /usr/local/lib/xcelsior/spire-server ($(/usr/local/lib/xcelsior/spire-server --version 2>&1 | head -1))"

say "4. agent gateway trusts the root"
fingerprint="$(openssl x509 -in "$ETC/upstream-root.crt" -noout -fingerprint -sha256 | cut -d= -f2)"
# The first base64 line of the PEM is unique to this certificate.
if sudo grep -qxF "$(sed -n 2p "$ETC/upstream-root.crt")" "$NGINX_BUNDLE"; then
    echo "already in $NGINX_BUNDLE ($fingerprint)"
else
    backup="$NGINX_BUNDLE.before-spire-$(date +%Y%m%d%H%M%S)"
    sudo cp -p "$NGINX_BUNDLE" "$backup"
    sudo sh -c "cat '$ETC/upstream-root.crt' >> '$NGINX_BUNDLE'"
    if sudo /usr/sbin/nginx -t 2>/dev/null; then
        sudo systemctl reload nginx
        echo "added and reloaded ($fingerprint); previous bundle: $backup"
    else
        sudo cp -p "$backup" "$NGINX_BUNDLE"
        fail "nginx -t failed with the new bundle; restored $backup"
    fi
fi

say "5. Headscale API key"
if sudo test -s "$ETC/api-enrollment.env" && sudo grep -q '^XCELSIOR_HEADSCALE_API_KEY=.' "$ETC/api-enrollment.env"; then
    echo "kept the key in $ETC/api-enrollment.env"
    key=""
else
    key="$(sudo headscale apikeys create --expiration 365d 2>/dev/null | tail -1)"
    [ -n "$key" ] || fail "headscale apikeys create returned nothing"
fi
user_id="$(sudo headscale users list -o json 2>/dev/null | python3 -c 'import json,sys; print(next(str(u["id"]) for u in json.load(sys.stdin) if u.get("name") == "xcelsior"))')" \
    || fail "headscale has no user named xcelsior"

say "6. API settings"
if [ -n "$key" ]; then
    umask 077
    {
        echo "# Host enrollment (issue #67); merge into /opt/xcelsior/.env"
        echo "XCELSIOR_SPIRE_SERVER_CMD=/usr/local/bin/spire-server"
        echo "XCELSIOR_SPIRE_SERVER_SOCKET=/run/xcelsior-spire/api.sock"
        echo "XCELSIOR_SPIRE_SERVER_ADDRESS=$MESH_ADDRESS"
        echo "XCELSIOR_SPIRE_SERVER_PORT=8095"
        echo "XCELSIOR_HEADSCALE_URL=https://hs.xcelsior.ca"
        echo "XCELSIOR_HEADSCALE_API_KEY=$key"
        echo "XCELSIOR_HEADSCALE_USER_ID=$user_id"
    } | sudo tee "$ETC/api-enrollment.env" >/dev/null
    sudo chmod 0600 "$ETC/api-enrollment.env"
    unset key
fi
echo "settings: $ETC/api-enrollment.env (root, 0600). Merge them into /opt/xcelsior/.env."
cat <<EOF

The API container also needs the socket and the CLI. Add to the api service in
/opt/xcelsior/docker-compose.override.yml, then deploy:

    volumes:
      - $VAR/sock:/run/xcelsior-spire
      - /usr/local/lib/xcelsior/spire-server:/usr/local/bin/spire-server:ro

Check after the deploy (an owner of a host can then enroll it from the wizard):

    docker exec xcelsior-api-1 spire-server healthcheck -socketPath /run/xcelsior-spire/api.sock
EOF
