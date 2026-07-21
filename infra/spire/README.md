# SPIRE / SPIFFE worker identity (Phase 10 scaffold)

This directory holds **configuration scaffolding** for host-bound worker
identities. A full multi-node SPIRE cluster is an operator deploy residual
and is **not** claimed as live-attested on every control-plane host.

## Intended shape (blueprint §19.2)

- Trust domain: `xcelsior.ca` (override with `XCELSIOR_SPIFFE_TRUST_DOMAIN`)
- Worker SPIFFE ID: `spiffe://xcelsior.ca/worker/host/<host_id>`
- Node attestation establishes host identity; workload SVIDs are short-lived
- Agent gateway (Envoy or Nginx mTLS) validates SVID and sets:
  - `X-Xcelsior-Agent-Gateway: 1`
  - `X-Worker-Host-Id: <host_id>`
  - `X-Worker-Spiffe-Id: <spiffe id>`
- API enables `XCELSIOR_TRUSTED_AGENT_GATEWAY=1` and fails closed without those headers

## Files

| File | Role |
|------|------|
| `server.conf.example` | SPIRE server sketch |
| `agent.conf.example` | SPIRE agent sketch on GPU hosts |
| `README.md` | This document |

## Fail-closed without live SPIRE

Until SPIRE is deployed, production still:

1. Requires authentication on `/agent/*` (`XCELSIOR_ENV=production`)
2. Maps bearer callers to a **registered + admitted** `host_id`
3. Returns **503** on host-admission DB errors (no fail-open)
4. Strips untrusted public `X-Worker-*` headers on MCP/public Nginx

Do not claim SPIRE attestation until agents and the gateway are live.
