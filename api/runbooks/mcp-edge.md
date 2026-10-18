# MCP edge runbook

The connector at `https://mcp.xcelsior.ca/mcp` is the surface every AI assistant
touches. Blue/green rollback already exists; this document says **when to pull
it**, and what to check first for the failures that are specific to this edge —
the ones where nothing in our logs looks wrong.

Adoption plan item X6.32. Companion: [mcp/README.md](../mcp/README.md) for the
deploy mechanics, [docs/mcp-tool-versioning.md](../docs/mcp-tool-versioning.md)
for the change contract.

---

## 0. Sixty-second triage

Run this first, from outside the network if you can:

```bash
python3 scripts/gx0_conformance.py --base https://mcp.xcelsior.ca/mcp
```

It walks TLS → 401 challenge → protected-resource metadata → authorization
server metadata → DCR gates, and names the first thing that broke. If it is
green and users still cannot connect, the problem is provider-side egress or the
authorization server, not the MCP edge — go to §4.

Direct checks, in the order a client makes them:

```bash
curl -sS https://mcp.xcelsior.ca/.well-known/oauth-protected-resource | jq .
curl -sS -D- -o/dev/null -X POST https://mcp.xcelsior.ca/mcp \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | grep -i www-authenticate
curl -sS https://mcp.xcelsior.ca/mcp/health | jq .
curl -sS https://mcp.xcelsior.ca/readyz | jq .
```

---

## 1. The failure that looks like nothing

**Symptom.** Users report "the connector won't connect." Our logs show ordinary
401s. Nothing is erroring.

**Cause.** The 401 is missing `WWW-Authenticate`, or it names a
`resource_metadata` URL that does not resolve. That header is the only
breadcrumb a client has; without it there is nothing to report on either side.

**Check.** The `grep -i www-authenticate` above must print a header containing
`realm=` and `resource_metadata=`. Then fetch that exact URL and confirm 200.

**Fix.** Roll back (§6). This is checked on every deploy in `scripts/deploy.sh`,
so a regression means either the check was bypassed or nginx is not routing
`/.well-known/oauth-protected-resource*` to the MCP upstream.

---

## 2. Tokens suddenly rejected everywhere

**Symptom.** Every authenticated call returns 401 `invalid_token`, immediately
after a deploy or a config change.

**Most likely cause: the resource identifier moved.** Tokens are audience-bound.
If `XCELSIOR_MCP_RESOURCE_AUDIENCE` and the API's `MCP_RESOURCE_AUDIENCE`
disagree — including by a trailing slash or a missing `/mcp` — every existing
token fails its audience check.

```bash
# These three must agree, exactly.
curl -sS https://mcp.xcelsior.ca/.well-known/oauth-protected-resource | jq -r .resource
curl -sS https://mcp.xcelsior.ca/mcp/health | jq -r .resource_audience
grep -E '^XCELSIOR_MCP_RESOURCE_AUDIENCE=' /opt/xcelsior/.env
```

**Second cause: the legacy audience window closed.** Tokens minted before the
migration carry the bare origin and are accepted until
`XCELSIOR_MCP_LEGACY_AUDIENCE_SUNSET` (2026-11-30). After that they are rejected
like any other wrong-audience token — which is correct, and which users
experience as "it stopped working". The fix is for them to reconnect, not for us
to reopen the window; if the volume is large enough to matter, extend the sunset
deliberately and announce it.

**Third cause: JWKS unreachable.** `/readyz` reports `jwks: false`. The MCP
verifies RS256 tokens against the authorization server's published keys.

---

## 3. Operator tools appear on the public connector

**Symptom.** `tools/list` on `mcp.xcelsior.ca` includes `drain_host` or another
platform-global tool.

**Severity: high.** This is a trust-boundary breach, not a cosmetic bug — a
provider snapshots the tool list and shows it to every end user.

**Check.** `curl -sS https://mcp.xcelsior.ca/mcp/health | jq -r .tool_profile`
must print `customer`.

**Fix.** Set `XCELSIOR_MCP_TOOL_PROFILE=customer` and redeploy, or roll back
(§6). Do not wait for a maintenance window.

---

## 4. A specific provider cannot connect, others can

**Symptom.** Claude connects; ChatGPT does not (or vice versa). Our external
conformance job is green.

**Cause.** Provider egress is blocked at the edge — a WAF rule, a rate limit
keyed on a shared IP, or a TLS chain a particular client rejects. A generic
cloud runner proves foreign-network reachability and proves nothing about a
specific provider's egress; this is the gap §4c of the plan is explicit about.

**Check.**

```bash
sudo tail -200 /var/log/nginx/error.log | grep -Ei 'limit|denied|403'
sudo grep -c 'mcp.xcelsior.ca' /var/log/nginx/access.log
```

Look for `limit_req` rejections clustered on one source range. The `/mcp`
location allows `burst=20 nodelay`; a provider fanning out from a shared egress
pool can exceed it in a way a single user never would.

**Fix.** Raise the burst for that location, or exempt the provider's published
egress range. Record the change — the rate limit is a published property.

---

## 5. Rate limits or Redis

`/readyz` reports `redis: false`, or every call returns 429.

The MCP fails **closed** on rate-limit backend loss by design
(`MCP_RATE_LIMIT_FAIL_CLOSED=true`), because a shared limiter that silently
becomes per-process is not a limit at all under multiple replicas. So a Redis
outage presents as a total connector outage, which is intended and is worth
knowing before you go looking for a bug in the MCP.

**Fix.** Restore Redis. Do not set `MCP_RATE_LIMIT_REQUIRE_REDIS=false` to
"unblock users" — that trades a visible outage for an invisible one.

---

## 6. Rollback

Blue/green on `:8770` / `:8771`. Deployment already verifies readiness, an
authenticated `initialize`, `tools/list`, and the 401 challenge against the
standby before switching nginx, so a bad image usually never goes live.

To roll back a switch that did land:

```bash
# On the host, restore the pre-swap nginx configs if they still exist:
for f in /etc/nginx/sites-available/xcelsior /etc/nginx/sites-available/mcp-xcelsior; do
  [ -f "$f.pre-mcp-swap" ] && sudo mv "$f.pre-mcp-swap" "$f"
done
sudo nginx -t && sudo nginx -s reload

# Otherwise flip the upstream by hand: the live colour is the one WITHOUT
# `backup` in the `upstream xcelsior_mcp` block.
sudo -e /etc/nginx/sites-available/mcp-xcelsior
sudo nginx -t && sudo nginx -s reload
```

Then confirm with the §0 triage before standing down.

**Pull the rollback when:** operator tools are visible publicly (§3), the 401
challenge is missing (§1), the resource identifier is wrong (§2), or
authenticated `tools/list` fails for more than one client. **Do not** roll back
for a single tool erroring — that is an upstream API problem and rolling the
MCP back will not fix it.

---

## 7. Published service levels

These are what the security and status pages state, and what an incident is
measured against.

| Objective | Target | Measured by |
|---|---|---|
| Connector availability (`initialize` + `tools/list` succeed) | 99.5% monthly | External conformance job, hourly |
| Discovery availability (metadata + 401 challenge) | 99.9% monthly | Same job |
| Read tool latency, p95 | < 2s | `xcelsior_mcp_tool_duration_seconds` |
| Write/plan tool latency, p95 | < 5s | Same, excluding `watch_instance` |
| Published rate limit | 120 calls/min per principal | `MCP_RATE_LIMIT_PER_MIN` |

An SLO miss — not connector folklore — is the trigger for reviewing a hosting
move (plan §4b). Record misses; do not migrate on a hunch.

---

## 8. Load and soak

Before publishing a rate limit, confirm the edge degrades rather than collapses
at twice it:

```bash
python3 scripts/mcp_soak.py \
  --base https://mcp.xcelsior.ca/mcp \
  --token "$XCELSIOR_MCP_TOKEN" \
  --rate 240 --duration 300
```

Pass condition (gate GX6): at 2× the published limit, excess calls are refused
with 429 and a `Retry-After`, latency for accepted calls stays within the p95
targets above, and no request returns 5xx. A 5xx under load is collapse, not
degradation.
