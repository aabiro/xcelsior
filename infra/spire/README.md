# SPIRE / SPIFFE worker identity

Live-mesh assets for host-bound worker identity (blueprint §19.2 / §19.3).

## The identity contract

One SPIFFE ID shape is valid for a GPU host, and the control plane
enforces it exactly:

```
spiffe://<trust-domain>/worker/host/<host_id>
```

* `control_plane.identity.spiffe_id_for_host()` computes it.
* `control_plane.identity.parse_worker_spiffe_id()` verifies it — the
  trust domain must match and the path must be exactly this shape.
* `infra/spire/register-host.sh` registers it in SPIRE using the *same*
  host-component sanitisation, so issuer and verifier cannot drift.
* `infra/envoy/agent-gateway.yaml` accepts only URI SANs under
  `spiffe://<trust-domain>/worker/host/` and derives `X-Worker-Host-Id`
  from the verified certificate.

An SVID for any other workload in the mesh — or any ID from another trust
domain — is not a GPU host and is rejected twice: at the Envoy listener
and again in the API.

## Files

| File | Role |
|------|------|
| `docker-compose.spire.yml` | brings up spire-server, spire-agent, and the Envoy agent gateway |
| `server.conf` | SPIRE server (PostgreSQL datastore, x509pop + join_token attestation, 1h SVIDs) |
| `agent.conf` | SPIRE agent for GPU hosts and the gateway host |
| `register-host.sh` | create/update the registration entry for one host or every admitted host |
| `../envoy/agent-gateway.yaml` | SPIFFE-aware mTLS terminator; `/agent/v2/*` only |

## Bring-up

> **Read the four notes below first.** The steps as originally written do not
> work on the current control-plane host, and one of them takes the live fleet
> down. Verified against production 2026-08-29.

```bash
# 1. datastore + secret
#
# The gateway secret is REUSED from the API's own environment, never generated.
# Production already holds one and the live Nginx gateway authenticates with it;
# a fresh value means Envoy presents a secret the API does not hold, every
# identity header is stripped, and the whole fleet is refused mid-cutover.
export SPIRE_DATASTORE_DSN="$(sudo grep -h '^SPIRE_DATASTORE_DSN=' /opt/xcelsior/.env | cut -d= -f2-)"
export XCELSIOR_AGENT_GATEWAY_SECRET="$(sudo grep -h '^XCELSIOR_AGENT_GATEWAY_SECRET=' /opt/xcelsior/.env | cut -d= -f2-)"
test -n "$XCELSIOR_AGENT_GATEWAY_SECRET" || { echo 'refusing: no gateway secret in /opt/xcelsior/.env'; exit 1; }

# 2. mesh — NOT `docker compose up`; see Note 2
sudo -E bash infra/spire/bring-up.sh

# 3. one registration entry per admitted host
XCELSIOR_POSTGRES_DSN=... infra/spire/register-host.sh --all

# 4. flip the API onto gateway identity
#    (same secret as step 1)
XCELSIOR_TRUSTED_AGENT_GATEWAY=1
XCELSIOR_AGENT_GATEWAY_SECRET=<step 1>
XCELSIOR_SPIFFE_TRUST_DOMAIN=xcelsior.ca
XCELSIOR_SPIFFE_STRICT=1
```

### Note 1 — the gateway secret is reused, never generated

Step 1 above now reads both values out of `/opt/xcelsior/.env` and refuses to
continue if the gateway secret is missing, because the earlier version of this
file said `openssl rand -hex 32` and that is a live-fleet outage:
`XCELSIOR_AGENT_GATEWAY_SECRET` is already set in production and the Nginx
gateway authenticates with it. A fresh value means
`gateway_headers_authenticated()` fails on every request, all identity headers
are stripped, and the fleet is refused — during its own cutover, which is the
worst moment to be debugging an authentication change.

`openssl rand -hex 32` is correct **only** on a first-ever install where nothing
holds a secret yet. If you are reading this on an existing deployment, you are
not in that case.

### Note 2 — use `bring-up.sh`; the compose file alone cannot start

`agent.conf` sets `trust_bundle_path = /opt/spire/conf/agent/bootstrap.crt` and
the compose file bind-mounts `./bootstrap.crt`, but that file is the **server's
own trust bundle** — it cannot exist before the server does. A plain
`docker compose up` fails with:

```
could not parse trust bundle: open .../bootstrap.crt: no such file or directory
```

`infra/spire/bring-up.sh` does the only ordering that works — server → export
bundle → everything else — and refuses to start at all on a half-configured
environment rather than leaving a mesh that looks up but cannot issue
identities. It checks, before touching anything:

* `XCELSIOR_AGENT_GATEWAY_SECRET` is set (see Note 1 — a generated one is an
  outage, not a typo)
* `SPIRE_DATASTORE_DSN` is set
* the bind address actually exists on this host, so the server is not published
  where nothing can reach it

and it refuses to write a bundle that contains no certificate, because handing
every agent an unparseable trust bundle fails later and further away.

### Note 3 — the server binds the tailnet, not loopback

**Decided.** The compose file published `127.0.0.1:8081`, so no GPU host
anywhere else could attest — and the agents that need this server are on other
machines by definition.

It now binds `100.64.0.1` (override with `SPIRE_BIND_ADDRESS`), and agents point
at `SPIRE_SERVER_ADDRESS=100.64.0.1` — a **bare address**, because `agent.conf`
carries `server_port = "8081"` separately and a host:port value there produces
`100.64.0.1:8081:8081`.

Publishing it with its own public TLS certificate was the alternative and it is
the wrong shape: a second trust boundary wrapped around the service whose entire
job is being the trust boundary. Xcelsior already runs the private network this
needs — Headscale WireGuard, this host at 100.64.0.1, GPU workers already
joined for other traffic. No new certificate, no new public surface, and it
matches how every other host↔control-plane channel here already works.

Binding a specific address means Docker refuses to start the container when the
tailnet interface is down. That is the correct failure: a SPIRE server reachable
from nowhere serves no one, and failing loudly at start beats agents timing out
later.

### Note 4 — 443 reaches 9443 by SNI passthrough

**Decided, built, and validated — see `runbooks/tls-ingress-cutover.md`.**

The listener was `0.0.0.0:8443`, which collides with Headscale's
`127.0.0.1:8443` on this host under `network_mode: host`. It is now 9443.

`nginx/stream-tls-router.conf` becomes the only thing binding 443: a `stream`
block that reads SNI with `ssl_preread` and terminates nothing. `agent.xcelsior.ca`
gets raw TCP passthrough to Envoy on 9443 so the worker's client certificate
survives; everything else goes to nginx's http block on **8444** (not 8443 —
Headscale) where the existing vhosts terminate exactly as before.

Two things this arrangement gets right and a naive version does not:

* **PROXY protocol on every hop.** After passthrough the backend's peer is
  127.0.0.1, so six services would silently begin logging, rate-limiting and
  geolocating against localhost. Each vhost carries `proxy_protocol` on its
  listen line plus `set_real_ip_from` / `real_ip_header`, and Envoy carries the
  `proxy_protocol` listener filter — without which it reads the PROXY line as a
  ClientHello and every worker connection fails as malformed TLS.
* **The cutover is one map entry.** Phase A moves the vhosts with *no* behaviour
  change (agent traffic still terminates in nginx). Phase B changes one line to
  9443, and changing it back is the rollback. `agent-xcelsior.conf` stays alive
  through both, which is why `XCELSIOR_SPIFFE_STRICT=0` remains correct until a
  real GPU host is proven through Envoy.

Also fixed here: the `xcelsior_api` cluster pointed at `127.0.0.1:9500` alone.
Production runs blue/green and 9501 is the live slot — the same fault that
already cost a session in `nginx/agent-xcelsior.conf`, where every worker
request 502s while every component looks healthy. Both slots are now listed.

### What is verified

Against the real binaries, locally, with no production access:

* `spire-server validate` — server.conf OK
* `envoy --mode validate` — gateway config OK, including the proxy_protocol
  listener filter
* `nginx -t` on the complete restructured arrangement (stream router + all seven
  vhosts on 8444) — **successful**, and with the *same 14 warnings* the original
  configs produce, so nothing was introduced
* `agent.conf` validates once `bootstrap.crt` exists (Note 2)

Held by `tests/test_tls_ingress_router_is_coherent.py` and
`tests/test_agent_gateway_points_at_a_live_backend.py`.

### Still open

* The datastore: run `scripts/provision_spire_datastore.sh`. Dedicated role and
  dedicated database on the shared instance, deliberately outside
  `control_plane/db_roles.py` — SPIRE is not a domain of the app's data, it
  manages its own schema, and it must keep working while the app is broken.
  Database-level isolation is the property you want between the application and
  the thing issuing the application's identities.
* **`nginx -V` must show `--with-stream_ssl_preread_module` on the target
  host.** Never confirmed on the VPS — the box went dark mid-check.
