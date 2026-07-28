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

## Agent configuration (Cursor)

```json
{
  "mcpServers": {
    "xcelsior": {
      "url": "https://mcp.xcelsior.ca/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_OAUTH_ACCESS_TOKEN"
      }
    }
  }
}
```

Obtain a token via OAuth `client_credentials` — create a machine client in **Dashboard → Settings → Connect AI Agents**.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `XCELSIOR_MCP_API_URL` | `http://127.0.0.1:8000` | Upstream FastAPI base URL |
| `XCELSIOR_MCP_RESOURCE_AUDIENCE` | `https://mcp.xcelsior.ca` | Required RFC 8707 token audience |
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
  `get_instance_timeline`, `get_active_lease`, `get_scheduler_health`,
  `get_host_capacity`, `list_reconciliation_findings`, `get_mcp_action_status`
- **Operations:** `retry_instance`, `reconcile_instance`, `drain_host`,
  `undrain_host`, `evict_host_workloads`, `retry_agent_command`

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
