# @xcelsior-gpu/mcp

Stdio Model Context Protocol client for the [Xcelsior](https://xcelsior.ca) GPU platform.

## Quick start

```bash
export XCELSIOR_ACCESS_TOKEN="your_oauth_access_token"
npx @xcelsior-gpu/mcp
```

Create a token: **Dashboard → Settings → Connect AI Agents** (or visit [xcelsior.ca/mcp](https://xcelsior.ca/mcp)).

## Or sign in with your browser

No token to copy: `connect` signs you in through your browser and connects to the hosted server at `https://mcp.xcelsior.ca/mcp`, using [mcp-remote](https://www.npmjs.com/package/mcp-remote) for the OAuth sign-in. The sign-in is saved, so it asks only once.

```bash
npx @xcelsior-gpu/mcp connect
```

```json
{
  "mcpServers": {
    "xcelsior": {
      "command": "npx",
      "args": ["-y", "@xcelsior-gpu/mcp", "connect"]
    }
  }
}
```

Set `XCELSIOR_MCP_URL` (or pass a URL after `connect`) to use another server.

## Cursor config

```json
{
  "mcpServers": {
    "xcelsior": {
      "command": "npx",
      "args": ["-y", "@xcelsior-gpu/mcp"],
      "env": {
        "XCELSIOR_ACCESS_TOKEN": "YOUR_OAUTH_ACCESS_TOKEN",
        "XCELSIOR_API_URL": "https://xcelsior.ca"
      }
    }
  }
}
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `XCELSIOR_ACCESS_TOKEN` | — | OAuth Bearer token (required for authenticated calls) |
| `XCELSIOR_API_URL` | `http://127.0.0.1:8000` | Fixed Xcelsior control-plane API base URL (`https://xcelsior.ca` in production) |

## Tool annotations & network boundary

Every tool declares all four MCP boolean annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) and validates inputs with Zod:

- **Fixed control-plane endpoint (`openWorldHint: false`):** Account, instance, volume, serverless, billing, and diagnostic tools communicate exclusively with the configured Xcelsior control-plane endpoint (`XCELSIOR_API_URL`) and read/mutate only the caller's own scoped platform records.
- **Open-world marketplace & public docs (`openWorldHint: true`):** Live third-party GPU marketplace feeds (`list_available_gpus`, `get_spot_prices`, `search_marketplace`, `schedule_under_budget`) and public documentation search (`search`, `fetch`) reflect external supply and published docs that change independently of tenant state.

## Hosted alternative

No local install required — use the hosted server at `https://mcp.xcelsior.ca/mcp` (Streamable HTTP) with OAuth 2.1 Bearer authentication and per-principal rate limiting. See `api/mcp/README.md` in the monorepo.

## Privacy & Policies

- **Privacy Policy:** [https://xcelsior.ca/privacy](https://xcelsior.ca/privacy)
- **Terms of Service:** [https://xcelsior.ca/terms](https://xcelsior.ca/terms)
- **Security & Support:** [https://docs.xcelsior.ca](https://docs.xcelsior.ca)
