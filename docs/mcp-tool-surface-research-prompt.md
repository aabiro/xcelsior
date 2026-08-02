# Deep-research prompt — Xcelsior MCP tool surface

Paste the fenced block into your deep-research tool and attach the files in §1.

The prompt is **grounded but open-ended**. It forbids inventing tool names, and
requires every candidate to cite a real endpoint — but it also explicitly invites
the researcher to beat the budget, restructure the domains, and argue that parts
of the existing plan are wrong. The point is to find out whether a fresh pass
does better than the 2026-07-29 audit, not to have it agree with us.

---

## 1. Files to attach

**Attach these five first — they carry most of the signal:**

| File | Why | Size |
|---|---|---|
| `docs/generated/endpoint-inventory.md` | **The single most important attachment.** All 528 operations across 37 modules, with auth guard and summary, in one table. Regenerate with `./venv/bin/python scripts/generate_endpoint_inventory.py`. | 60K |
| `docs/mcp-tool-surface-plan.md` | The 2026-07-29 audit: domain budget, curation rules, payments architecture, gates GT0–GT4, permanent exclusions. | 25K |
| `mcp/src/tools/contracts.ts` | What the 39 tools are today: required scopes, tenant class, annotations. | — |
| `mcp/src/auth/scopes.ts` | The scope enum, including the three scopes wired to no tool. | — |
| `mcp/src/tools/descriptions.ts` | Current descriptions — the thing a model actually reads when choosing. | — |

**Attach if your tool has room:**

| File | Why |
|---|---|
| `docs/mcp-provider-axis-plan.md` | The provider axis, boundary options, P0–P6 sequence. |
| `docs/mcp-enterprise-adoption-plan.md` | Directory requirements, trust-surface architecture (§4a), the security properties that are non-negotiable. |
| `mcp/tool-surface.json` | Machine-readable snapshot of the published surface. |
| `mcp/README.md` | Approval model and action-plan semantics in prose. |
| `public/openapi.json` | The *curated public* spec — 70 paths only. Do **not** mistake it for the full surface; the inventory is the full surface. |

**Do not attach the raw route modules.** `routes/instances.py` alone is 3,790
lines and `routes/` totals ~14k; the inventory exists precisely so a reader gets
all 528 operations without them. Attach a specific module only when the
researcher asks for one by name.

---

```
# Role

You are designing the tool surface for a production MCP (Model Context Protocol)
server that is submitted to the Anthropic and OpenAI connector directories. Your
output will be read by a security reviewer and by the engineers who implement it.
"Defensible to a reviewer" is a hard requirement, not a stylistic preference.

I am giving you an existing plan. **I want you to challenge it.** If the domain
budget is wrong, say so. If a whole domain should not exist, say so. If you can
design a materially better surface than the one budgeted, do that and show me
why it is better. Agreement is not the deliverable — a better answer is.

# The product, and what it is actually for

Xcelsior Compute Inc. operates a distributed GPU marketplace.

**The MCP server is the flagship feature, not a convenience wrapper on a REST
API.** The differentiator is that an AI agent can provision real GPU compute —
launch instances, run training and serverless inference, manage spend — through
a governed interface with scoped tokens, human-approved action plans on
destructive and spending operations, per-tool rate limits, and a full audit
trail. Other providers make you leave the agent to go click in a dashboard. Any
recommendation that dilutes agent-driven GPU provisioning, or that treats it as
a secondary surface, is the wrong answer for this product. Design *toward* that
capability being the best in the market.

The security properties above are the product, not overhead. Do not propose
relaxing them for convenience — propose ways to make them feel effortless.

## Three user classes

1. **Consumers** — rent GPU capacity. Instances, serverless inference, billing.
   The only class the current surface serves.
2. **Providers** — *supply* GPU capacity. Register hosts, earn reputation,
   onboard for payouts via Stripe Connect and PayPal, get paid. **Zero tools.**
3. **Operators** — Xcelsior staff. Drain hosts, evict workloads, inspect the
   scheduler. Served by a separate unlisted deployment.

# Current state (measured, not estimated)

- 528 API operations across 37 modules (see the attached inventory). The MCP
  server reaches roughly 7% of them.
- 39 tool contracts: 30 in the `customer` profile, 37 with `operator`, 39 with
  the two opt-in company-knowledge tools (`search`/`fetch`).
- The prior audit targets ~91 actions, sized under Gemini Enterprise's
  100-action cap with headroom.
- Zero-coverage domains: storage/volumes, compliance/residency, teams/access,
  payouts/Connect.
- Three scopes declared and wired to no tool: `billing:write`, `events:read`,
  `mcp_actions:approve`.
- The MCP SDK in use (1.29.0) supports **URL Mode Elicitation (SEP-1036)**.

# Non-negotiable constraints

- **Never invent a tool name from a path you have not read.** Every candidate
  cites the endpoint(s) it wraps, by method and path, from the inventory. No
  citation, no candidate.
- One tool may wrap several endpoints if it closes one user-visible journey.
  Prefer few high-level capabilities over thin REST mirrors.
- Destructive and spending operations go through a server-bound action plan
  requiring human approval. `confirm: true` is never a substitute for approval.
- No card numbers, tokens, or secrets may enter model context, ever.
- Admin/cross-tenant endpoints, webhook receivers, HTML page renderers, and raw
  shell execution are permanently out of scope.
- Total actions ≤100.

# Deliverables

## 1. Candidate tool table

Columns: `tool_name` · `user_class` · `domain` · `endpoints_wrapped` (method +
path, cited) · `journey_it_closes` · `read_or_write` · `required_scope` ·
`needs_approval` · `elicitation_mode` (none/URL) · `rationale`.

Give a per-domain count against this budget:

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

## 2. Your better alternative — required, not optional

Independently of the table above, propose the surface **you** would design given
the same 528 endpoints, the same constraints, and the ≤100 ceiling. Restructure
the domains if that is better. Merge or split aggressively. Then give me a short,
blunt comparison: what your version does better, what it gives up, and which one
you would actually ship. If your version *is* the budgeted one, say that and why.

Also give me **at least five tool or capability ideas that are not in the budget
at all** — things that fall out of having 528 endpoints and an agent that can
already provision GPUs, that nobody thought to ask for. Rank them by how much
they widen the gap against competitors. Speculative is fine here as long as each
one still cites the endpoints that would make it real.

## 3. Exclusion table

Every endpoint you deliberately do **not** expose, with the reason. This is as
much a deliverable as the inclusions — it is what makes the surface defensible
and what stops the list quietly growing later.

## 4. The boundary design question

The provider (supply-side) surface has no tools. Three shapes are on the table:

- **A. Third deployment profile** — `XCELSIOR_MCP_TOOL_PROFILE=provider` at a
  separate host. Matches the existing architecture; zero risk to the reviewed
  customer listing. But a dual-role human installs twice.
- **B. Scope-gated subset of the public connector** — one connector; provider
  tools registered only when the principal holds provider scopes. Natural for
  dual-role humans. But registration is per-connection, and a directory's frozen
  `tools/list` snapshot may not tolerate a listing that varies by principal.
- **C. Its own directory listing** at its own URL.

**The hard part, and the question I most want answered:** a single human can be
both consumer and provider on the same account — renting an A100 today, listing
their idle 4090 tomorrow. So this cannot be modelled as disjoint tenant types.

Tell me:
1. Can a directory-listed connector vary `tools/list` by principal scope without
   breaking the provider's frozen snapshot? Answer per client — Claude, ChatGPT,
   Gemini Enterprise, Copilot, Grok — and cite documentation.
2. Which shape do you recommend, and what evidence decides it?
3. Is there a standard or emerging MCP convention for advertising tool *groups*
   or toolsets? If there is none, say so plainly rather than inventing one.
4. At what total tool count does a flat surface measurably degrade model
   tool-selection accuracy? Cite evidence, not intuition. That number decides
   whether subsetting is worth its complexity.

## 5. URL Mode Elicitation (SEP-1036)

This is how a user personally authorizes something in their browser — adding a
card, depositing funds, Stripe Connect payout onboarding — with no sensitive
data ever entering model context. Our SDK supports it. We have not shipped it.

Answer specifically:
1. What is the current state of the spec — SEP number, status, which SDK
   versions and which *clients* actually honour URL mode? A server that emits
   `mode: "url"` to a client that ignores it must degrade safely; what does the
   spec say the server should do, and what do real clients do?
2. Has any shipped MCP server done Stripe Connect onboarding or a payment flow
   this way? What did they learn, and what went wrong?
3. What is the correct fallback when a client supports no elicitation at all —
   and how do we make that fallback impossible to confuse with success?
4. Are there security pitfalls specific to URL elicitation — session fixation,
   open redirect, phishing via attacker-influenced URLs, the returned-value
   trust model? What must a server validate before emitting a URL, and after the
   user returns?
5. Is URL elicitation the right mechanism for **payout onboarding** specifically
   — a KYC flow, longer and more sensitive than a card add — or is something
   else more appropriate?

## 6. Naming and description conventions

Recommend a convention spanning all three user classes, with evidence about what
makes a model pick the right tool. Every description must state purpose, cost,
and refusal conditions.

# Output format

Markdown tables, minimal prose. Mark any row you could not verify `UNVERIFIED`
and say what would settle it. I would rather have thirty cited candidates and
five honest unknowns than ninety confident guesses.
```
