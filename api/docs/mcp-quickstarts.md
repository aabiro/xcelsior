# Connecting Xcelsior to your assistant

> One connector URL, nine clients. Adoption plan X7.35 — each of these is meant
> to be reproducible from scratch, on a clean machine, by somebody who did not
> write it. If a step here does not work exactly as written, that is a bug in
> the connector, not in your setup; please [tell us](https://xcelsior.ca/support).

**The URL is always the same:**

```
https://mcp.xcelsior.ca/mcp
```

You never need to create a token, copy a secret, or edit a config file to use
the OAuth path. Paste the URL, sign in, approve. Everything below is that, with
the buttons named per client.

---

## Which path do I want?

| You are… | Use |
|---|---|
| A person connecting an assistant | **OAuth** — everything in §1–§7 |
| A CI job, a script, a headless agent | **An agent key** — §8 |

`client_credentials` is fully supported and is not going away. It is simply not
the front door for a human, because a human should not have to hold a secret to
ask what a GPU costs.

---

## 1. Claude (web, desktop, mobile)

1. **Settings → Connectors → Add custom connector.**
2. Paste `https://mcp.xcelsior.ca/mcp`.
3. Press **Connect**. Claude opens Xcelsior's sign-in page.
4. Sign in, review the permissions, press **Approve**.
5. You are back in Claude, connected.

Try: *"What GPUs do you have available right now, and what would 4 hours on the
cheapest one cost?"*

> Custom connectors need a Claude **Pro, Max, Team, or Enterprise** plan.
> On Team and Enterprise an Owner or admin adds it once for the whole
> organization.

## 2. Claude Code

```bash
claude mcp add --transport http xcelsior https://mcp.xcelsior.ca/mcp
```

The first tool call opens your browser to sign in and approve. Claude Code binds
a fresh loopback port each attempt; that is expected and works — port-agnostic
loopback matching is part of the connector contract.

Verify:

```bash
claude mcp list          # xcelsior should be listed and connected
```

## 3. ChatGPT

1. **Settings → Connectors → Add.**
2. Paste `https://mcp.xcelsior.ca/mcp`, choose **OAuth**.
3. Sign in and approve.

Try: *"Using Xcelsior, is there anything under $0.40 an hour with at least 24GB
of VRAM?"*

## 4. Cursor

Cursor supports one-click installs. Use the button on
[xcelsior.ca/mcp](https://xcelsior.ca/mcp), or add it by hand in
`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "xcelsior": { "url": "https://mcp.xcelsior.ca/mcp" }
  }
}
```

No headers block: Cursor performs the OAuth flow when the server asks for it.

## 5. VS Code and GitHub Copilot

`.vscode/mcp.json` in your workspace, or the button on
[xcelsior.ca/mcp](https://xcelsior.ca/mcp):

```json
{
  "servers": {
    "xcelsior": { "type": "http", "url": "https://mcp.xcelsior.ca/mcp" }
  }
}
```

VS Code prompts for authorization on first use. For **Copilot CLI**:

```bash
copilot mcp add xcelsior --transport http --url https://mcp.xcelsior.ca/mcp
```

## 6. Grok (Business / Enterprise)

A team admin adds it once:

1. **Admin → Connectors → Add MCP server.**
2. URL `https://mcp.xcelsior.ca/mcp`.
3. Each team member authorizes with their own Xcelsior account the first time
   they use it — the connection is shared, the account is not.

## 7. Microsoft Copilot Studio

1. In your agent: **Tools → Add a tool → Model Context Protocol.**
2. Server URL `https://mcp.xcelsior.ca/mcp`.
3. Choose **OAuth 2.0**; Copilot Studio discovers our authorization server and
   registers itself dynamically. No client id or secret to paste.

## 8. Automation — CI, scripts, headless agents

Create a machine client in **Dashboard → Settings → Connect AI Agents**, then:

```bash
ACCESS_TOKEN=$(curl -s -X POST https://xcelsior.ca/oauth/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'grant_type=client_credentials' \
  -d 'resource=https://mcp.xcelsior.ca/mcp' \
  -d "client_id=$MCP_CLIENT_ID" -d "client_secret=$MCP_CLIENT_SECRET" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
```

Then point any MCP client at the URL with `Authorization: Bearer $ACCESS_TOKEN`.

`resource` is required and must be exactly `https://mcp.xcelsior.ca/mcp` — the
token is bound to it, and a token minted for anything else is refused at the
edge.

## 9. Local stdio (no OAuth at all)

For a local process that already holds a token:

```bash
XCELSIOR_ACCESS_TOKEN=... npx @xcelsior-gpu/mcp
```

The stdio server always starts, even without a valid token — tool calls return a
clean 401 telling you to refresh, rather than the client reporting the server as
"failed to start".

---

## Verifying a connection

Ask for something read-only and free:

> *"List my Xcelsior instances."*

Then something that proves the guardrails are real:

> *"Launch an RTX 4090 called scratch-test."*

You should get a **plan**, not an instance — a cost estimate, the canonical
spec, and an approval requirement. Nothing is allocated and nothing is billed
until you approve it. If an assistant ever tells you it launched something
without that step, that is a bug we want to hear about immediately.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Couldn't connect" with no detail | The client could not discover our OAuth endpoints. | `curl -i -X POST https://mcp.xcelsior.ca/mcp` — the 401 must carry a `WWW-Authenticate` header naming `resource_metadata`. If it does not, [tell us](https://xcelsior.ca/support); that is our bug. |
| Works once, fails on reconnect | A native client bound a different loopback port. | Should not happen — we match loopback ports RFC 8252-style. Report it. |
| 401 on every call after it worked | Token expired, or bound to the wrong resource. | Reconnect. For automation, check `resource` is exactly `https://mcp.xcelsior.ca/mcp`. |
| "Insufficient scope" | The connection was approved with narrower permissions than the tool needs. | Disconnect and reconnect, approving the wider set. |
| Tools missing that you expected | Host and control-plane tools are not on the public connector, by design. | See [the security page](https://xcelsior.ca/security). |
| Everything returns 429 | You are over 120 calls/min, or our rate-limit backend is down (we fail closed). | Back off using `Retry-After`. If it persists, check [status](https://xcelsior.ca/status). |

## Revoking access

**Dashboard → Settings → AI Agents.** Revoking a connection takes effect on the
next call — there is no window where a revoked assistant keeps working.
