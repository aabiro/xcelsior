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

```bash
# 1. datastore + secret
export SPIRE_DATASTORE_DSN='postgresql://spire:...@127.0.0.1:5432/spire'
export XCELSIOR_AGENT_GATEWAY_SECRET="$(openssl rand -hex 32)"

# 2. mesh
docker compose -f infra/spire/docker-compose.spire.yml up -d

# 3. one registration entry per admitted host
XCELSIOR_POSTGRES_DSN=... infra/spire/register-host.sh --all

# 4. flip the API onto gateway identity
#    (same secret as step 1)
XCELSIOR_TRUSTED_AGENT_GATEWAY=1
XCELSIOR_AGENT_GATEWAY_SECRET=<step 1>
XCELSIOR_SPIFFE_TRUST_DOMAIN=xcelsior.ca
XCELSIOR_SPIFFE_STRICT=1
```

`/readyz` refuses to report ready if `XCELSIOR_TRUSTED_AGENT_GATEWAY=1`
without a gateway secret, so a half-configured cutover fails the deploy
gate instead of silently accepting forgeable headers.

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
