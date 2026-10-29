#!/usr/bin/env bash
# Pre-flight for the 443 SNI-passthrough cutover (runbooks/tls-ingress-cutover.md).
#
#   sudo bash scripts/preflight_tls_ingress.sh
#
# Read-only. Changes nothing, and exits non-zero if the host cannot support the
# restructure — checked here rather than discovered during a reload, because the
# failure mode is losing TLS for every vhost on the box, not just one.
set -uo pipefail

FAIL=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; FAIL=1; }
warn() { printf '  warn  %s\n' "$*"; }

echo "== nginx"
if ! command -v nginx >/dev/null; then
  bad "nginx is not installed"
else
  ok "nginx $(nginx -v 2>&1 | sed 's/.*nginx\///')"

  # The whole design rests on this module. Without it the stream block cannot
  # read SNI, so there is no way to give one vhost passthrough while the rest
  # keep terminating on the same port.
  if nginx -V 2>&1 | grep -q -- '--with-stream_ssl_preread_module'; then
    ok "stream_ssl_preread module present"
  else
    bad "nginx lacks --with-stream_ssl_preread_module — the SNI router cannot work.
        Install nginx-full/nginx-extras (Debian) or nginx-mod-stream (RHEL),
        or the ingress design needs rethinking."
  fi

  # `--with-stream=dynamic` is the common packaging (Ubuntu/Debian), and it
  # means the module is NOT active until something load_module's it. A config
  # using `stream { }` against an unloaded module fails at `nginx -t` with
  # "unknown directive", which reads like a typo rather than a missing module.
  if nginx -V 2>&1 | grep -qE -- '--with-stream(=|[[:space:]]|$)'; then
    if nginx -V 2>&1 | grep -q -- '--with-stream=dynamic'; then
      if nginx -T 2>/dev/null | grep -qE '^\s*load_module.*ngx_stream_module\.so;'; then
        ok "stream module built dynamic AND loaded"
      elif ls /etc/nginx/modules-enabled/ 2>/dev/null | grep -q stream; then
        ok "stream module built dynamic, loaded via modules-enabled/"
      else
        if [ -f /usr/lib/nginx/modules/ngx_stream_module.so ]; then
          bad "stream module is present but not loaded. Add to nginx.conf:
          load_module modules/ngx_stream_module.so;"
        else
          bad "stream module is built dynamic and the .so is not installed.
          On Ubuntu/Debian:  apt-get install libnginx-mod-stream
          (that package drops the .so and an /etc/nginx/modules-enabled entry
          that loads it). Without it the stream block fails as \"unknown
          directive\", which reads like a typo rather than a missing module."
        fi
      fi
    else
      ok "stream module built in statically"
    fi
  else
    bad "nginx was not built with the stream module"
  fi
fi

echo "== ports"
# 8444 is the http backend, 9443 is Envoy. Both must be free; 8443 must stay
# Headscale's, which is precisely why the backend is not on it.
for spec in "8444:nginx http backend" "9443:envoy agent gateway"; do
  port="${spec%%:*}"; what="${spec#*:}"
  if ss -ltn 2>/dev/null | grep -qE "[:.]${port}[[:space:]]"; then
    bad "port ${port} (${what}) is already in use: $(ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" | head -1 | tr -s ' ')"
  else
    ok "port ${port} free (${what})"
  fi
done
if ss -ltn 2>/dev/null | grep -qE '[:.]8443[[:space:]]'; then
  ok "port 8443 in use as expected (Headscale) — the backend deliberately avoids it"
else
  warn "nothing on 8443; Headscale may be down on this host"
fi

echo "== current 443 owners"
if ss -ltn 2>/dev/null | grep -qE '[:.]443[[:space:]]'; then
  ok "something serves 443 today (expected: nginx, about to become the stream router)"
else
  warn "nothing is serving 443 right now"
fi

echo "== config"
if [ -f /etc/nginx/stream-tls-router.conf ]; then
  if grep -qE '^\s*include\s+/etc/nginx/stream-tls-router\.conf;' /etc/nginx/nginx.conf; then
    ok "router installed and included from nginx.conf"
  else
    warn "router file present but not included from nginx.conf (Phase A step 2)"
  fi
  # The include must be a sibling of `http`, not inside it: `stream` is not
  # valid within an http block and nginx will refuse the whole config.
  if awk '/^\s*http\s*\{/{d=1} d&&/stream-tls-router/{print "inside"; exit}' /etc/nginx/nginx.conf | grep -q inside; then
    bad "the include sits INSIDE the http block; stream must be a top-level sibling"
  fi
else
  warn "stream-tls-router.conf not deployed yet"
fi

if command -v nginx >/dev/null; then
  if nginx -t >/dev/null 2>&1; then
    ok "nginx -t passes with the config currently on disk"
  else
    bad "nginx -t FAILS right now — fix that before changing anything:
$(nginx -t 2>&1 | sed 's/^/        /')"
  fi
fi

echo
if [ "$FAIL" -ne 0 ]; then
  echo "NOT READY — resolve the FAIL lines above before Phase A."
  exit 1
fi
echo "Ready for Phase A (runbooks/tls-ingress-cutover.md)."
