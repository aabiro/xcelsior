# TLS ingress cutover — SNI passthrough for the agent gateway

Gives `agent.xcelsior.ca` true TLS passthrough to Envoy (so the worker's client
certificate survives to SPIFFE) while the other six vhosts on 443 keep
terminating in nginx exactly as they do today.

```
        stream (443, ssl_preread)
                 │
   agent.xcelsior.ca ──► 127.0.0.1:9443  Envoy   (Phase B only)
   everything else ────► 127.0.0.1:8444  nginx http
```

**8444, not 8443** — Headscale holds `127.0.0.1:8443` on the control-plane host.

## Before you start

`nginx -V 2>&1 | grep -o with-stream_ssl_preread_module` must print something.
Without that module the router cannot read SNI and nothing below works. Verify
this on the target host; it was never confirmed on the VPS because the box went
dark mid-check.

## Phase A — move the six vhosts behind the router (no behaviour change)

Traffic still terminates in nginx for every name, including
`agent.xcelsior.ca`. This phase only proves the router layer.

1. Copy the seven vhost files and `nginx/stream-tls-router.conf` into place.
2. Include the router from the **top level** of `/etc/nginx/nginx.conf`, a
   sibling of `http` — not from `sites-enabled`, which is included *inside*
   `http`:

   ```nginx
   include /etc/nginx/stream-tls-router.conf;
   ```
3. `sudo nginx -t` — must be clean before anything else.
4. `sudo systemctl reload nginx`. A reload does not drop in-flight
   connections; existing workers and browsers are unaffected.
5. Verify each name still answers and, critically, that **real client IPs**
   survived the extra hop:

   ```bash
   for h in xcelsior.ca docs.xcelsior.ca downloads.xcelsior.ca \
            mcp.xcelsior.ca hs.xcelsior.ca agent.xcelsior.ca; do
     printf '%-24s %s\n' "$h" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$h/")"
   done
   sudo tail -20 /var/log/nginx/access.log   # must NOT be all 127.0.0.1
   ```

   All-127.0.0.1 access logs mean `set_real_ip_from` / `real_ip_header` did not
   take, and six services are now rate-limiting and geolocating against
   localhost. That is a rollback, not a warning.

**Rollback:** remove the include, restore the vhosts' `listen 443 ssl http2;`,
`nginx -t`, reload.

## Phase B — send the agent name to Envoy

Only after SPIRE is up and Envoy is running and healthy on 9443.

1. In `stream-tls-router.conf`, change the one map entry:

   ```nginx
   agent.xcelsior.ca   127.0.0.1:9443;
   ```
2. `sudo nginx -t && sudo systemctl reload nginx`
3. Watch one real GPU host reconnect. It must present its SVID and get 200s on
   `/agent/v2/hosts/heartbeat`.
4. Only once that is proven, set `XCELSIOR_SPIFFE_STRICT=1` and retire
   `nginx/agent-xcelsior.conf`.

**Rollback:** change the map entry back to `127.0.0.1:8444` and reload. The
certificate-DN gateway in `agent-xcelsior.conf` is still there and still works;
that is why it is kept through Phase B.

## The two failures that look like something else

* **PROXY protocol mismatch.** The router sends the PROXY header to *every*
  backend. A backend that is not expecting it reads that line as the first bytes
  of a ClientHello, so every connection fails as malformed TLS while nginx,
  Envoy, SPIRE and the certificates are each individually fine. Envoy needs the
  `proxy_protocol` listener filter (already in
  `infra/envoy/agent-gateway.yaml`); every nginx vhost needs `proxy_protocol` on
  its `listen` line.
* **Two things binding 443.** A leftover `listen 443 ssl` anywhere and nginx
  refuses to start — losing TLS for *all* vhosts, not just the stray one. This
  is why `nginx -t` gates every step.

`tests/test_tls_ingress_router_is_coherent.py` holds both, plus the real-IP
restoration, across all seven vhosts and the Envoy listener.
