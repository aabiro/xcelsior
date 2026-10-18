# @xcelsior-gpu/mcp

Stdio Model Context Protocol client for the [Xcelsior](https://xcelsior.ca) GPU platform.

## Quick start

```bash
export XCELSIOR_ACCESS_TOKEN="your_oauth_access_token"
npx @xcelsior-gpu/mcp
```

Create a token: **Dashboard → Settings → Connect AI Agents** (or visit [xcelsior.ca/mcp](https://xcelsior.ca/mcp)).

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
| `XCELSIOR_ACCESS_TOKEN` | — | OAuth Bearer token (required) |
| `XCELSIOR_API_URL` | `http://127.0.0.1:8000` | API base URL |

## Hosted alternative

No local install required — use the hosted server at `https://xcelsior.ca/mcp` (Streamable HTTP). See `mcp/README.md` in the monorepo.