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

# 2. mesh
docker compose -f infra/spire/docker-compose.spire.yml up -d      # ← SEE NOTE 2

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

### Note 2 — `bootstrap.crt` does not exist yet, and the compose mounts it

`agent.conf` sets `trust_bundle_path = /opt/spire/conf/agent/bootstrap.crt` and
the compose bind-mounts `./bootstrap.crt`, but no such file is in this
directory. `spire-agent validate` fails outright:

```
could not parse trust bundle: open .../bootstrap.crt: no such file or directory
```

It is the server's own trust bundle, so it cannot exist before the server does.
Start the server alone, export the bundle, then start the agent:

```bash
docker compose -f infra/spire/docker-compose.spire.yml up -d spire-server
docker compose -f infra/spire/docker-compose.spire.yml exec -T spire-server \
    /opt/spire/bin/spire-server bundle show > infra/spire/bootstrap.crt
docker compose -f infra/spire/docker-compose.spire.yml up -d
```

### Note 3 — remote GPU hosts cannot reach the server as configured

The compose publishes the server as `127.0.0.1:8081:8081` — loopback on the
control-plane host only. A GPU host elsewhere has nothing to attest against.
Decide deliberately how agents reach `:8081` (tailnet address, or published with
its own TLS) before registering hosts that are not the control plane itself.

### Note 4 — the listener moved to 9443, and 443 does not reach it yet

The Envoy listener was `0.0.0.0:8443`. **Headscale already holds
`127.0.0.1:8443` on this host**, and the listener binds `0.0.0.0` under
`network_mode: host`, so that was a direct collision with the tailnet control
plane. It is now `9443`.

Nothing routes 443 → 9443 yet. That last hop must be **SNI passthrough** for
`agent.xcelsior.ca` — if Nginx terminates TLS first it strips the client
certificate, which is the entire point of SPIFFE mTLS. Nginx currently owns 443
for eight vhosts on this box, so this is a real ingress decision and not a
config tweak.

Also fixed while verifying: the `xcelsior_api` cluster pointed at `127.0.0.1:9500`
alone. Production runs blue/green and **9501 is the live slot** — 9500 has
nothing bound. That is the same fault that already cost a session in
`nginx/agent-xcelsior.conf`: every worker request 502s while Envoy, mTLS, SPIRE
and the API are each individually healthy. Both slots are now listed, and
`tests/test_agent_gateway_points_at_a_live_backend.py` holds it there.

### What is verified

`server.conf` and the Envoy config both validate against the real binaries
(`spire-server validate`, `envoy --mode validate`). `agent.conf` validates once
`bootstrap.crt` exists.

## Migrating from Nginx mTLS

`nginx/agent-xcelsior.conf` is the interim gateway: it validates client
certificates but has a certificate DN, not a SPIFFE ID. Run it with
`XCELSIOR_SPIFFE_STRICT=0` and cut over to Envoy before setting strict
mode. Strict mode is the live-mesh posture and the default in code.

## Fail-closed posture without SPIRE

With no gateway configured at all, production still:

1. requires authentication on `/agent/*` (`XCELSIOR_ENV=production`);
2. maps the caller to a **registered + admitted** `host_id`;
3. returns **503** on host-admission lookup errors (never fail-open);
4. strips untrusted public `X-Worker-*` headers on public/MCP ingress;
5. accepts per-host bearer tokens (§19.2) that are scoped and rotated —
   see `control_plane/agent_tokens.py`.
