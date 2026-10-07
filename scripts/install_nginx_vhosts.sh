#!/usr/bin/env bash
# Install rendered nginx vhosts on this host, or leave the previous ones serving.
#
# Run on the target by scripts/deploy.sh (`ssh … bash -s < this`), after it has
# rendered the repo's vhosts for this host's topology and shipped them to
# $SRC_DIR. Every step that can break the public site is checked, and a failure
# restores the files that were serving a moment before:
#
#   1. back up the current vhosts;
#   2. install the new ones and run `nginx -t`;      fail -> restore
#   3. reload, then ask nginx for xcelsior.ca/healthz; not 200 -> restore
#
# Step 3 exists because `nginx -t` passed on 2026-10-05, when a vhost layout
# that needed the stream router was installed without it: the config was valid
# and xcelsior.ca still answered "400 No required SSL certificate" for twelve
# minutes. Only a request through nginx finds that.
#
# Paths are overridable so the test suite can run this script for real against
# a scratch directory with stub binaries.
set -euo pipefail

NGINX_DIR="${NGINX_DIR:-/etc/nginx}"
SRC_DIR="${SRC_DIR:-/tmp/xcelsior-nginx}"
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups}"
HEALTH_HOST="${HEALTH_HOST:-xcelsior.ca}"
VHOSTS="xcelsior headscale headscale-http docs-xcelsior downloads-xcelsior"

backup="$BACKUP_ROOT/nginx-deploy-$(date -u +%Y%m%dT%H%M%S)"
sudo mkdir -p "$backup"
for f in $VHOSTS; do
  if [ -f "$NGINX_DIR/sites-available/$f" ]; then
    sudo cp -p "$NGINX_DIR/sites-available/$f" "$backup/$f"
  fi
done

restore() {
  echo "Restoring the previous vhosts from $backup" >&2
  for f in $VHOSTS; do
    if [ -f "$backup/$f" ]; then
      sudo cp -p "$backup/$f" "$NGINX_DIR/sites-available/$f"
    fi
  done
  sudo nginx -t && sudo systemctl reload nginx
}

for f in $VHOSTS; do
  sudo cp "$SRC_DIR/$f.conf" "$NGINX_DIR/sites-available/$f"
  sudo ln -sf "$NGINX_DIR/sites-available/$f" "$NGINX_DIR/sites-enabled/$f"
done

if ! sudo nginx -t; then
  echo "nginx -t rejected the new vhosts." >&2
  restore
  exit 1
fi

if systemctl is-active --quiet nginx; then
  sudo systemctl reload nginx
else
  sudo systemctl reset-failed nginx 2>/dev/null || true
  sudo systemctl start nginx
fi

code="000"
for _ in 1 2 3 4 5; do
  code=$(curl -sk --max-time 10 --resolve "$HEALTH_HOST:443:127.0.0.1" \
    -o /dev/null -w '%{http_code}' "https://$HEALTH_HOST/healthz" || true)
  [ "$code" = "200" ] && break
  sleep 1
done
if [ "$code" != "200" ]; then
  echo "https://$HEALTH_HOST/healthz answered $code through the new vhosts." >&2
  restore
  exit 1
fi

echo "Installed nginx vhosts; previous ones kept in $backup"
