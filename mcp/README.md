# Xcelsior MCP Server

Hosted Model Context Protocol server for the Xcelsior GPU platform.

## Quick start (local)

```bash
cd mcp
npm install
XCELSIOR_API_URL=http://127.0.0.1:8000 npm run dev
```

Health: `GET http://localhost:3100/health`  
MCP endpoint: `http://localhost:3100/mcp`

## Agent configuration (Cursor)

```json
{
  "mcpServers": {
    "xcelsior": {
      "url": "https://xcelsior.ca/mcp",
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
| `XCELSIOR_API_URL` | `http://127.0.0.1:8000` | Upstream FastAPI base URL |
| `MCP_HOST` | `0.0.0.0` | Bind address |
| `MCP_PORT` | `3100` | Listen port |
| `MCP_PATH` | `/mcp` | HTTP path |
| `MCP_RATE_LIMIT_PER_MIN` | `60` | Per-token tool call budget |

## Tools (v0.2)

- **Discovery:** `list_available_gpus`, `get_spot_prices`, `get_pricing_reference`, `search_marketplace`, `list_tiers`
- **Compute:** `list_instances`, `get_instance`, `get_instance_logs`, `create_instance`, `cancel_instance`, `terminate_instance`
- **Billing:** `get_wallet_balance`, `estimate_job_cost`, `list_invoices`
- **Guardrails:** `should_i_run_this` (estimate + wallet + optional `max_hourly_cad`)
- **Workflows:** `run_training_job`, `schedule_under_budget`
- **Serverless:** `list_serverless_endpoints`, `create_serverless_endpoint`, `run_serverless_job`, `get_serverless_job_status`
- **Monitoring:** `watch_instance` (poll status + telemetry + logs)

**Resources:** `xcelsior://docs/llms`, `xcelsior://pricing/reference`  
**Prompts:** `cheapest-gpu-now`, `ca-fine-tune`, `serverless-inference`

Destructive tools accept `confirm: false` for preview-only responses.

## Stdio package (v2)

```bash
npx @xcelsior-gpu/mcp
```

Requires `XCELSIOR_ACCESS_TOKEN`. See `mcp-cli/README.md`.

## Production deploy

`docker compose up -d mcp` — listens on `127.0.0.1:3100`. Nginx routes POST/DELETE (and Bearer GET) on `/mcp` to the MCP container; GET without auth serves the marketing page from Next.js.

## Staging smoke

```bash
MCP_CLIENT_ID=... MCP_CLIENT_SECRET=... python3 scripts/mcp_smoke.py
```

## Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector@latest
```

Point at `http://localhost:3100/mcp` with your Bearer token.