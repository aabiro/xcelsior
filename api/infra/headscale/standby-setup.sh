#!/usr/bin/env bash
# Bring a Headscale standby to the primary's build and give it a public TLS front.
#
#   sudo api/infra/headscale/standby-setup.sh
#
# Safe to re-run. What it does, in order, stopping at the first failure:
#   1. backs up the standby's binary, config, state and failover tool;
#   2. stops and disables the standby's headscale.service — a standby must not
#      serve; `headscale-failover promote` starts it;
#   3. installs the *primary's* headscale binary (checksum-verified), because a
#      replica is only promotable on the build that wrote it;
#   4. installs the failover tool from this checkout;
#   5. refreshes the replica and prints whether it is promotable;
#   6. issues hs.xcelsior.ca by DNS-01 (Cloudflare) and installs the nginx front.
#
# Still needed afterwards, by hand: forward TCP 443 on the home router to this
# host, so a promoted standby can be reached from outside the LAN.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="/var/backups/headscale-standby-$TS"
LOG="/var/log/headscale-standby-setup-$TS.log"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
echo "== headscale standby setup $TS (log: $LOG)"

# shellcheck disable=SC1091
. /etc/default/headscale-failover
PRIMARY="${XCELSIOR_HEADSCALE_HOST:?set in /etc/default/headscale-failover}"
PRIMARY_USER="${XCELSIOR_HEADSCALE_USER:-root}"
KEY="${XCELSIOR_HEADSCALE_SSH_KEY:-/home/${SUDO_USER:-aaryn}/.ssh/id_ed25519}"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 "$PRIMARY_USER@$PRIMARY")

echo "== 1. backup -> $BACKUP"
mkdir -p "$BACKUP"
cp -a /usr/bin/headscale "$BACKUP/headscale.bin"
cp -a /etc/headscale "$BACKUP/etc-headscale"
cp -a /var/lib/headscale "$BACKUP/var-lib-headscale"
[ -f /usr/local/sbin/headscale-failover ] && cp -a /usr/local/sbin/headscale-failover "$BACKUP/"
chmod 700 "$BACKUP"

echo "== 2. stop the standby (it serves only when promoted)"
systemctl disable --now headscale.service || true
systemctl reset-failed headscale.service || true

echo "== 3. install the primary's headscale build"
want="$("${SSH[@]}" "sha256sum /usr/bin/headscale" | cut -d' ' -f1)"
tmp="$(mktemp /tmp/headscale.XXXXXX)"
scp -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$PRIMARY_USER@$PRIMARY:/usr/bin/headscale" "$tmp"
got="$(sha256sum "$tmp" | cut -d' ' -f1)"
if [ "$want" != "$got" ]; then
  echo "checksum mismatch: primary $want, copied $got" >&2
  rm -f "$tmp"
  exit 1
fi
install -m 0755 -o root -g root "$tmp" /usr/bin/headscale
rm -f "$tmp"
/usr/bin/headscale version

echo "== 4. install the failover tool"
install -m 0755 -o root -g root "$REPO/api/scripts/headscale_failover.py" /usr/local/sbin/headscale-failover
# The current live config predates the ownership fix; promotion rewrites it anyway.
for f in /etc/headscale/config.yaml /etc/headscale/acl.json; do
  [ -f "$f" ] && chown root:headscale "$f" && chmod 0640 "$f"
done

echo "== 5. refresh the replica and check it"
headscale-failover replicate
headscale-failover status --json > "$BACKUP/status.json"
python3 - "$BACKUP/status.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
r, b = s["replica"], s.get("headscale_build", {})
print(f"promotable={r['promotable']} ({r['reason']}), replicated_at={r['replicated_at']}")
print(f"build: standby={b.get('standby','')[:12]} primary={b.get('primary','')[:12]} match={b.get('match')}")
sys.exit(0 if r["promotable"] and b.get("match") else 1)
PY

echo "== 6. TLS front for hs.xcelsior.ca"
apt-get install -y -q certbot python3-certbot-dns-cloudflare > "$BACKUP/apt-certbot.log" 2>&1 \
  || { echo "apt install failed; see $BACKUP/apt-certbot.log" >&2; exit 1; }
CF_INI=/etc/letsencrypt/cloudflare-xcelsior.ini
if [ ! -s "$CF_INI" ]; then
  token="$(grep -E '^CLOUDFLARE_API_TOKEN=' "$REPO/.env" | tail -1 | cut -d= -f2- | sed -E 's/^["'\'']//; s/["'\'']$//')"
  [ -n "$token" ] || { echo "CLOUDFLARE_API_TOKEN not found in $REPO/.env" >&2; exit 1; }
  install -d -m 0700 /etc/letsencrypt
  ( umask 077; printf 'dns_cloudflare_api_token = %s\n' "$token" > "$CF_INI" )
  unset token
fi
certbot certonly --non-interactive --agree-tos -m admin@xcelsior.ca \
  --dns-cloudflare --dns-cloudflare-credentials "$CF_INI" --dns-cloudflare-propagation-seconds 30 \
  --keep-until-expiring --deploy-hook "systemctl reload nginx" -d hs.xcelsior.ca
install -m 0644 "$REPO/api/infra/headscale/standby-nginx.conf" /etc/nginx/sites-available/headscale
ln -sf /etc/nginx/sites-available/headscale /etc/nginx/sites-enabled/headscale
nginx -t
systemctl reload nginx
code="$(curl -sk -o /dev/null -w '%{http_code}' --resolve hs.xcelsior.ca:443:127.0.0.1 https://hs.xcelsior.ca/health || true)"
echo "local front: https://hs.xcelsior.ca/health via 127.0.0.1 -> $code (502 is right: the standby is stopped)"

echo "== done. Next: forward TCP 443 on the router to this host, then check from outside."
