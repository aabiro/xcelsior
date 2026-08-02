# Deep-research prompt — Xcelsior MCP tool surface

Paste everything in the fenced block below into your deep-research tool. It is
written to be **grounded**: it forbids inventing tool names, requires every
candidate to cite a real endpoint, and asks for exclusions with reasons — the
same discipline [mcp-tool-surface-plan.md](./mcp-tool-surface-plan.md) §5 sets
for the enumeration ("I am not going to invent tool names from endpoint paths I
haven't read").

**Before you run it**, generate the inventory it consumes — the prompt is far
weaker without it, because the model will otherwise reason from route names
instead of from what the endpoints actually do:

```bash
./venv/bin/python scripts/generate_endpoint_inventory.py   # GT0 deliverable; see note below
```

That script is itself a GT0 deliverable and does not exist yet. Until it does,
attach these instead: `routes/*.py` for the domains under study, `public/openapi.json`,
`mcp/src/tools/contracts.ts`, `mcp/src/auth/scopes.ts`, and `mcp/tool-surface.json`.

---

```
# Role

You are designing the tool surface for a production MCP (Model Context Protocol)
server that is submitted to the Anthropic and OpenAI connector directories. Your
output will be reviewed by a security reviewer and by the engineers who
implement it. Treat "defensible to a reviewer" as a hard requirement, not a
stylistic preference.

# The product

Xcelsior Compute Inc. operates a distributed GPU marketplace. There are three
distinct user classes, and this matters more than anything else in this brief:

1. **Consumers** — rent GPU capacity. Launch instances, run serverless
   inference, pay for it. This is the only class the current tool surface serves.
2. **Providers** — supply GPU capacity. Register hosts, get scored on
   reputation, onboard for payouts via Stripe Connect and PayPal, get paid.
   **Zero tools today.**
3. **Operators** — Xcelsior staff. Drain hosts, evict workloads, inspect the
   scheduler. Served by a separate unlisted deployment, not the public one.

# Current state (measured, not estimated)

- The FastAPI application exposes ~514 endpoints; ~487 appear in the live
  schema. The MCP server reaches roughly 7% of them.
- 39 tool contracts exist: 30 in the `customer` profile, 37 with `operator`,
  39 with the two opt-in company-knowledge tools (`search`/`fetch`) enabled.
- Target surface is ~91 actions, sized to stay under Gemini Enterprise's
  100-action cap with headroom.
- Zero-coverage domains: storage/volumes, compliance/residency, teams/access,
  payouts/Connect.
- Three scopes are declared but wired to no tool at all: `billing:write`,
  `events:read`, `mcp_actions:approve`.
- The MCP SDK in use (1.29.0) supports **URL Mode Elicitation (SEP-1036)**.

# Constraints that are not negotiable

- **Never invent a tool name from a path you have not read.** Every candidate
  must cite the concrete endpoint(s) it wraps, by method and path, from the
  material provided. If you cannot cite one, the candidate does not exist.
- One tool may wrap several endpoints if it closes one user-visible journey.
  Prefer few high-level tools over many thin REST mirrors — an agent works
  better with ~20 well-named capabilities than 79 raw operations.
- Destructive or spending operations go through a server-bound action plan
  requiring human approval. A `confirm: true` argument is never a substitute
  for approval. Do not design around this.
- No card numbers, tokens, or secrets may ever enter model context. Anything
  requiring a human to personally authorize (adding a card, depositing funds,
  payout onboarding) must use URL Mode Elicitation, not a form.
- Admin/cross-tenant endpoints, webhook receivers, HTML page renderers, and raw
  shell execution are permanently out of scope.
- Total actions must stay ≤100.

# What I want from you

## 1. A candidate tool table

Columns: `tool_name` · `user_class` (consumer/provider/operator) · `domain` ·
`endpoints_wrapped` (method + path, cited) · `journey_it_closes` ·
`read_or_write` · `required_scope` · `needs_approval` (y/n) ·
`elicitation_mode` (none / URL) · `rationale`.

Group by domain and give a per-domain count against this budget:

| Domain | Now | Target |
|---|---:|---:|
| Discovery & pricing | 5 | 8 |
| Instance lifecycle | 13 | 14 |
| Serverless / inference | 5 | 10 |
| Storage & volumes | 0 | 5 |
| Billing & payments | 3 | 13 |
| Payouts / Connect (provider) | 0 | 6 |
| Monitoring & events | 4 | 8 |
| Ops / control plane | 7 | 10 |
| Compliance & residency | 0 | 6 |
| Teams & access | 0 | 5 |
| Meta (plans, status, capabilities) | 1 | 6 |

If you believe a budget line is wrong, say so and justify it — but do not
silently exceed one.

## 2. An exclusion table

Every endpoint you deliberately do **not** expose, with the reason. This table
is as much a deliverable as the inclusions: it is what makes the surface
defensible, and what stops the list quietly growing later.

## 3. Toolset configuration recommendation

Answer these specifically:

- Should the provider surface be a **third deployment profile**, a **scope-gated
  subset of the public one**, or **its own connector entry**? Argue from how
  directory clients snapshot `tools/list`, and from the fact that one human can
  be both a consumer and a provider on the same account.
- At what total tool count does a single flat surface start degrading model
  tool-selection accuracy? Cite evidence, not intuition. Give the threshold at
  which subsetting becomes worth its complexity.
- How do the major clients (Claude, ChatGPT, Gemini Enterprise, Copilot, Grok)
  each handle large tool lists, per-session filtering, and any published caps?
  Note where a server-side `--toolsets` style flag is or is not usable for a
  directory-listed connector that cannot vary flags per session.
- Is there a standard or emerging convention for advertising tool *groups* in
  MCP? If there is none, say so plainly rather than inventing one.

## 4. Naming and description conventions

Recommend a convention across all three user classes, with evidence about what
makes a model pick the right tool. Every description must state purpose, cost,
and refusal conditions.

# Output format

Markdown tables, no prose padding. Where you are uncertain, mark the row
`UNVERIFIED` and say what would settle it. I would rather have thirty cited
candidates than ninety confident guesses.
```
