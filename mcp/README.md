# Xcelsior MCP Server

Hosted Model Context Protocol server for the Xcelsior GPU platform.

## Quick start (local)

```bash
cd mcp
npm ci
XCELSIOR_MCP_API_URL=http://127.0.0.1:8000 npm run dev
```

Liveness: `GET http://localhost:8770/health`
Readiness: `GET http://localhost:8770/readyz`
MCP endpoint: `http://localhost:8770/mcp`

## Connecting

The canonical connector URL is **`https://mcp.xcelsior.ca/mcp`**.

### Default path — OAuth consent (what humans use)

Paste that URL into Claude, ChatGPT, Grok, Copilot Studio, Cursor, or VS Code
and press connect. The client discovers everything it needs on its own:

1. It calls `/mcp` with no credentials and gets `401` plus
   `WWW-Authenticate: Bearer realm="xcelsior", resource_metadata="…"`.
2. It fetches `/.well-known/oauth-protected-resource`, which names the
   authorization server and the exact resource identifier.
3. It identifies itself — by **CIMD** (preferred: `client_id` is an HTTPS URL
   serving the client's own metadata), by **RFC 7591 dynamic registration** at
   `/oauth/register`, or with the pre-provisioned public client
   `xcelsior-connector`.
4. You sign in and approve the requested scopes on Xcelsior's own consent
   screen. Approval is remembered per client, per scope set, per resource, and
   is revocable in **Settings → AI Agents**.
5. It exchanges the code with S256 PKCE at `/oauth/token`
   (`application/x-www-form-urlencoded`) for a ~1h access token bound to
   `https://mcp.xcelsior.ca/mcp`, plus a 30-day refresh token.

No client secret is involved, and nothing has to be pasted by hand.

### Automation path — `client_credentials`

For CI, scripts, and headless agents where no human is present to consent.
Create a machine client in **Dashboard → Settings → Connect AI Agents**, then:

```bash
curl -s -X POST https://xcelsior.ca/oauth/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'grant_type=client_credentials' \
  -d 'resource=https://mcp.xcelsior.ca/mcp' \
  -d "client_id=$MCP_CLIENT_ID" -d "client_secret=$MCP_CLIENT_SECRET"
```

```json
{
  "mcpServers": {
    "xcelsior": {
      "url": "https://mcp.xcelsior.ca/mcp",
      "headers": { "Authorization": "Bearer YOUR_ACCESS_TOKEN" }
    }
  }
}
```

`client_credentials` is fully supported and is not going away — it is simply
not the front door for a person connecting an assistant.

### Resource identifier

`https://mcp.xcelsior.ca/mcp` — the exact URL a user pastes, path included.
Protected-resource metadata, the `resource` parameter on authorization and
token requests, and the `aud` claim all carry that one value. Tokens issued
before the migration carry the bare origin `https://mcp.xcelsior.ca` and remain
valid until **2026-11-30**, after which they are rejected like any other
wrong-audience token.

## Trust surfaces

One codebase, one image, two exposed surfaces (`XCELSIOR_MCP_TOOL_PROFILE`):

| Profile | Where | Tools |
|---|---|---|
| `customer` (default) | `https://mcp.xcelsior.ca/mcp`, submitted to directories | Tenant workflows only |
| `operator` | Separate unlisted host | Adds platform-global host and control-plane tools |

Operator tools are not registered at all under the customer profile, so they
cannot appear in `tools/list` and cannot be called by name. A token without
operator scopes never enumerates them even on the operator host.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `XCELSIOR_MCP_API_URL` | `http://127.0.0.1:8000` | Upstream FastAPI base URL |
| `XCELSIOR_MCP_RESOURCE_AUDIENCE` | `https://mcp.xcelsior.ca/mcp` | Canonical RFC 8707 token audience |
| `XCELSIOR_MCP_LEGACY_RESOURCE_AUDIENCE` | origin of the above | Pre-migration audience, accepted until sunset |
| `XCELSIOR_MCP_LEGACY_AUDIENCE_SUNSET` | `2026-11-30T00:00:00Z` | When the legacy audience stops being accepted |
| `XCELSIOR_MCP_RESOURCE_METADATA_URL` | derived | What the 401 challenge points at |
| `XCELSIOR_MCP_TOOL_PROFILE` | `customer` | `customer` or `operator` |
| `XCELSIOR_MCP_COMPANY_KNOWLEDGE` | `0` | Register ChatGPT company-knowledge `search`/`fetch` |
| `XCELSIOR_DOCS_URL` | `https://docs.xcelsior.ca` | Documentation corpus for company knowledge |
| `MCP_HOST` | `0.0.0.0` | Bind address |
| `MCP_PORT` | `8770` | Listen port |
| `MCP_PATH` | `/mcp` | HTTP path |
| `MCP_RATE_LIMIT_PER_MIN` | `60` | Per-token tool call budget |

## Tools (v2)

- **Discovery:** `list_available_gpus`, `get_spot_prices`, `get_pricing_reference`, `search_marketplace`, `list_tiers`
- **Compute:** `list_instances`, `get_instance`, `get_instance_logs`, `create_instance`, `cancel_instance`, `terminate_instance`
- **Billing:** `get_wallet_balance`, `estimate_job_cost`, `list_invoices`
- **Guardrails:** `should_i_run_this` (estimate + wallet + optional `max_hourly_cad`)
- **Workflows:** `run_training_job`, `schedule_under_budget`
- **Serverless:** `list_serverless_endpoints`, `create_serverless_endpoint`, `run_serverless_job`, `get_serverless_job_status`
- **Monitoring:** `watch_instance` (poll status + telemetry + logs)
- **Diagnostics:** `explain_instance_placement`, `simulate_instance_placement`,
  `get_instance_timeline`, `get_active_lease`, `get_mcp_action_status`
- **Operations (tenant):** `retry_instance`, `reconcile_instance`

Operator profile only: `get_scheduler_health`, `get_host_capacity`,
`list_reconciliation_findings`, `drain_host`, `undrain_host`,
`evict_host_workloads`, `retry_agent_command`.

**Company knowledge (optional, off by default):** `search` and `fetch`, with
OpenAI's exact schemas, over the documentation site, `llms.txt`, the pricing
table, and live marketplace listings. Every result carries an absolute,
human-openable URL and an internal id. Enable with
`XCELSIOR_MCP_COMPANY_KNOWLEDGE=1`; it is deliberately not on for the base
directory submission.

**Resources:** `xcelsior://docs/llms`, `xcelsior://pricing/reference`  
**Prompts:** `cheapest-gpu-now`, `ca-fine-tune`, `serverless-inference`

`create_instance` and `create_serverless_endpoint` always create a
server-bound action plan first. `confirm:true` expresses intent but never
replaces approval; execute with the returned `plan_id` after standing-policy
or human approval. A confirmed call without a plan safely prepares a plan and
returns `approval_required`. Host draining never evicts workloads; eviction has
its own scope and action plan.

## Stdio package (v2)

```bash
npx @xcelsior-gpu/mcp
```

Requires `XCELSIOR_ACCESS_TOKEN`. See `mcp-cli/README.md`.

## Production deploy

Production promotion is blue/green across ports 8770/8771. Deployment runs
readiness, bearer-authenticated protocol initialize, and tools/list against the
standby before switching Nginx; a failure leaves the previous MCP serving.

## Staging smoke

```bash
MCP_CLIENT_ID=... MCP_CLIENT_SECRET=... python3 scripts/mcp_smoke.py
```

## Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector@latest
```

Point at `http://localhost:8770/mcp` with your MCP-audience Bearer token.
