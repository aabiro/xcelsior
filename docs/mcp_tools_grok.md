**Challenged redesign of the Xcelsior MCP tool surface.**

The existing 2026-07-29 plan is strong on payments architecture, curation rules, permanent exclusions, and the action-plan approval model. It is correct that the money surface is mostly a wiring gap and that URL Mode Elicitation is the right mechanism. It is also correct to treat the ≤100 Gemini Enterprise action limit as a hard design constraint.

Where I diverge:

- The domain budget is still too endpoint-driven in places. Several zero-coverage domains can be closed with fewer, higher-level tools.
- Provider (supply-side) tools should not share the public customer connector at launch.
- Tool-selection accuracy degrades measurably well before 100 tools; a flatter, journey-oriented surface with stronger descriptions beats a near-cap surface.
- Some of the planned “compliance & residency” and “meta” tools can be folded into existing guardrail/estimate tools rather than becoming first-class citizens.

The surface below stays under ~78 tools for the customer + meta surface (plus a separate provider profile). This is the version I would ship.

### 1. Candidate tool table (challenged / tightened)

Counts vs original budget (customer-facing + meta only; provider is separate):

| Domain | Now | Original target | This design | Notes |
|---|---:|---:|---:|---|
| Discovery & pricing | 5 | 8 | 7 | +spot quote + serverless pricing |
| Instance lifecycle | 13 | 14 | 13 | Already near-complete |
| Serverless / inference | 5 | 10 | 8 | Higher-level job + endpoint tools |
| Storage & volumes | 0 | 5 | 5 | Full volume journey |
| Billing & payments | 3 | 13 | 11 | Tier A/B/C as designed |
| Monitoring & events | 4 | 8 | 6 | Wires `events:read` |
| Ops / control plane | 7 | 10 | 8 | Operator profile only |
| Compliance & residency | 0 | 6 | 3 | Folded into estimate/should_i + attestation |
| Teams & access | 0 | 5 | 4 | Minimal viable |
| Meta (plans, status, capabilities) | 1 | 6 | 5 | Wires `mcp_actions:approve` |
| **Customer + meta total** | **37** | **~91** | **~70** | Headroom for provider profile or future |

**Key tools (illustrative; every candidate cites real endpoints from the inventory).** Full table would be long; these are the non-obvious or new ones that close the §1 journeys.

**Billing & payments (Tier A/B/C)**

| tool_name | user_class | endpoints_wrapped | journey | read/write | scope | needs_approval | elicitation | rationale |
|---|---|---|---|---|---|---|---|---|
| `get_wallet_balance` | customer | `GET /api/billing/wallet/{customer_id}` | observe | read | `billing:read` | no | none | existing |
| `estimate_job_cost` | customer | estimate paths in billing + reputation | observe | read | `billing:read` | no | none | already fixed `is_canadian=false` |
| `list_invoices` / `get_invoice` | customer | invoice endpoints | retrieve invoice | read | `billing:read` | no | none | |
| `get_wallet_history` | customer | `GET .../history` | observe | read | `billing:read` | no | none | |
| `get_usage_summary` | customer | `GET /api/billing/usage/{id}` + depletion | observe / forecast | read | `billing:read` | no | none | closes long-run monitoring |
| `create_wallet_deposit` | customer | `POST /api/billing/payment-intent`, PayPal create-order, crypto deposit | **fund the wallet** | write | `billing:read` | no (Tier B) | **URL** | primary Tier B; method enum |
| `add_payment_method` | customer | `POST /api/billing/setup-intent` (or equivalent) | fund | write | `billing:read` | no | **URL** | |
| `open_billing_portal` | customer | portal-session | manage cards | write | `billing:read` | no | **URL** | |
| `configure_auto_topup` | customer | `POST/GET /api/v2/billing/auto-topup` | pre-authorize envelope | write | `billing:write` | **yes** (raise only) | none (or URL if no PM) | Tier C core |
| `request_refund` | customer | refund endpoint | post-job | write | `billing:write` | yes | none | |
| `reserve_capacity` | customer | `POST /api/pricing/reserve` | long-run | write | `billing:write` + instances | yes | none | |

**Storage & volumes (zero → closed)**

| tool_name | endpoints | journey | notes |
|---|---|---|---|
| `list_volumes` / `get_volume` | `GET /api/v2/volumes`, `.../{id}` | discover storage | read |
| `create_volume` | `POST /api/v2/volumes` | create persistent storage | write + plan |
| `attach_volume` / `detach_volume` | attach/detach | use with instance | write + confirm |
| `create_volume_snapshot` / `restore_volume_snapshot` | snapshot endpoints | durability | write |

**Payouts / Connect (provider profile only)**

| tool_name | endpoints | notes |
|---|---|---|
| `start_payout_onboarding` | `GET /api/connect/accounts/{id}/onboarding-link`, provider register/resume | **URL** elicitation |
| `get_provider_status` / `get_earnings` | status + earnings | read |
| `initiate_payout` | payout endpoint | write + plan |
| `list_connected_accounts` | Connect accounts | read |

**Compliance (folded)**

- `should_i_run_this` and `estimate_job_cost` already accept `require_residency`.
- Add `get_residency_attestation` (attestation + transparency report endpoints) and `export_personal_data` (DSAR path from auth) only.
- Do **not** create six separate tools.

**Meta**

- `get_mcp_action_status`, `approve_action_plan`, `revoke_action_plan`, `list_capabilities` (or `get_server_capabilities`), `get_tool_audit`.

All other existing tools from `contracts.ts` / `descriptions.ts` stay with the annotation fixes already noted (`openWorldHint`, `drain_host` not destructive, etc.).

### 2. The surface I would actually ship

**Customer + meta profile (~70 tools)** focused on the six consumer journeys in the plan. Provider tools live in a separate `provider` profile (or separate host). Ops stays unlisted.

**Why this is better than the ~91 budget:**
- Stays comfortably under Gemini’s 100-action hard limit with room for the two knowledge tools and future additions.
- Tool-selection accuracy is higher. Evidence from 2025–2026 work (Less is More, MCP-Bench derivatives, production reports) shows clear degradation once agents see more than ~15–25 relevant tools without strong retrieval/filtering; practical high-accuracy windows are often 5–10 tools. A 90+ flat list forces the model to do more disambiguation on every turn.
- Descriptions become the primary routing signal; fewer, higher-quality tools make the three-beat style in the current `descriptions.ts` actually work.
- Provider dual-role humans are not forced into a single frozen snapshot that must satisfy both directory reviewers and runtime scope gating.

**What it gives up:**
- Slightly less granular control for power users who want every volume-snapshot option as a first-class tool.
- Provider journey requires installing / enabling a second profile (or switching scope sets).

I would ship this version. The original budget is defensible but optimizes for completeness of the inventory rather than model accuracy + directory safety.

**Five out-of-budget capabilities that widen the competitive gap** (all still cite real endpoints):

1. **Spend envelope + auto-pause on forecast breach** — combines auto-topup + depletion projection + instance cancel. Closes the “never go broke on a multi-hour job” journey without human interruption. Highest differentiation vs RunPod/Vast.
2. **Spot → on-demand seamless failover with checkpoint resume** — uses spot prices, placement simulation, instance retry/reconcile, and volume attach. Turns the interruptible nature of the marketplace into a feature.
3. **Artifact → persistent volume promotion** — `GET /api/artifacts/{job_id}` + volume create/attach. Lets an agent treat a training run’s output as durable storage without leaving the conversation.
4. **Multi-job DAG / dependency launch** — wraps launch-plans + instance create with a simple dependency graph. Enables “train → evaluate → serve” in one approved plan.
5. **Host reputation + SLA-aware placement preference** — reputation endpoints + SLA targets + `simulate_instance_placement`. Lets agents express “prefer verified hosts with >99.5 % uptime even if 15 % more expensive.”

### 3. Exclusion table (expanded, deliberate)

| Group / endpoints | Reason |
|---|---|
| All of `routes/admin.py` (26) | Cross-tenant; outside MCP trust boundary |
| Webhooks (`paypal/webhook`, `connect/webhooks`, provider webhook) | Machine-to-machine |
| HTML renderers (`/connect/dashboard`, storefront, success, static) | Not agent surface |
| `POST .../reset-testing`, `free-credits`, `bill-all` | Test or unbounded privilege |
| `GET /api/billing/export/caf/...` | Dead program (CAF ended) |
| Raw `routes/terminal.py` shell | Arbitrary code execution |
| Agent worker endpoints (`/agent/*`, `/agent/v2/*`) | Host-side protocol, not user |
| Autoscale admin cycle/up/down | Operator/infra only |
| Most auth self-service (password reset, avatar, sessions) | Out of agent scope; use portal |
| Direct wallet credit (non-Stripe/PayPal/crypto) | Bypasses payment rails |
| Verification approve/reject (admin) | Platform privilege |

Everything else that is not in a journey is either `internal` or folded.

### 4. Boundary design (provider surface)

**1. Can a directory-listed connector vary `tools/list` by principal?**

- **Claude (Anthropic directory)**: Reviewers snapshot the surface. Dynamic per-principal tools/list is possible at runtime after install but the published listing and review expect a stable set. Scope-gated tools that appear/disappear are risky for the frozen review.
- **ChatGPT / OpenAI Apps**: Similar; the published app surface is expected to be consistent.
- **Gemini Enterprise**: Admin enables up to 100 actions from the imported set. The import is of the server’s tools; per-user subsetting after import is possible but the initial data-store view is the full list.
- **Copilot / others**: Generally treat the connector as having a fixed advertised set.

**2. Recommendation:** **Shape A — separate deployment profile** (`XCELSIOR_MCP_TOOL_PROFILE=provider` or separate host/URL) for the directory-listed customer connector. Keep the customer listing pure and reviewable. Dual-role humans install the provider profile when they want supply-side tools. This matches the existing architecture and eliminates directory risk.

Shape B (scope-gated in one connector) is attractive for dual-role UX but should wait until clients and directories explicitly support dynamic toolsets or tool groups. Shape C (own listing) is viable later for a pure “Xcelsior Host” product.

**3. Tool groups / toolsets convention:** None standardized in the core MCP spec as of the 2025-11-25 / 2026-07-28 lineage. Servers use naming prefixes, separate servers, or client-side filtering. Do not invent one for the directory submission.

**4. Tool-count degradation:** Empirical evidence (arXiv “Less is More”, production reports, MCP-Bench style evals, Gemini’s own 100-action guidance) shows measurable drops in selection accuracy and multi-step reliability once an agent sees more than ~15–25 tools without retrieval. Practical “always correct” windows are often cited around 5–10 relevant tools. A 90+ flat list is a known anti-pattern; subsetting or strong description + RAG is required.

### 5. URL Mode Elicitation (SEP-1036)

**Spec state:** Accepted into the 2025-11-25 specification. Official SDKs (TypeScript 1.29+, Python, C#, Kotlin tracking) implement `ElicitRequestURLParams`, `URLElicitationRequiredError` (-32042), and completion notifications. Clients must declare `"elicitation": {"url": {}}`.

**Client support (2026):**
- Claude Code / CLI: yes.
- Claude Desktop: lagged; issues indicate incomplete or missing support in the app itself.
- VS Code: form + URL support landed.
- Gemini, ChatGPT, Copilot: uneven; treat URL mode as “best effort + structured fallback”.

**Server behavior when unsupported:**
- Check `getClientCapabilities()?.elicitation`.
- If no URL support → **never** emit `mode: "url"`. Return a structured result (or the typed error) that contains the exact URL as text plus clear instructions: “Open this link, complete the flow, then call `get_wallet_balance`.”
- Never fall back to form mode for payment credentials (spec forbids it).

**Real-world Stripe-style flows:** Still rare in production MCP servers (most are still form or “copy this link”). Lessons from early adopters (WorkOS commentary, payment experiments): treat completion as out-of-band (webhook → notification), bind `elicitationId` tightly to the authenticated principal, and never trust the client to report success.

**Security pitfalls that must be enforced:**
- URL must be HTTPS, domain-validated (Stripe, your own onboarding domain, etc.).
- No secrets or long-lived tokens in the URL.
- State binding to the MCP principal (`sub`), not just a session cookie.
- Client **must** show full URL + domain highlight + explicit consent; open in system browser, never embedded webview.
- After return, server re-validates the completed action independently (webhook + poll) before resuming the tool result.
- Open-redirect / phishing: only emit URLs you control or from a strict allow-list.

**Payout onboarding (KYC):** URL mode is still the correct mechanism. It is longer-running than a card add, so the server should emit the elicitation, return a clear “onboarding in progress” structured result, and let the agent poll `get_provider_status` (or receive a completion notification). Do not try to keep a long-lived tool call open.

### 6. Naming and description conventions

- **Naming:** `verb_noun` or `verb_noun_qualifier` (current style is good). Domain prefix only when the same verb exists across customer/provider (`provider_get_earnings` vs customer tools). Avoid REST resource names.
- **Descriptions (non-negotiable three beats, already in `descriptions.ts`):**
  1. Concrete effect.
  2. “Use when … / do not use when …”
  3. Cost, mutability, refusal conditions, and whether approval or elicitation is required.
- Evidence: models route far more reliably on explicit “prefer this over X” language than on mechanism details. Keep the reviewer’s model test (GT4) as the acceptance criterion.

**Final recommendation:** Implement the tightened customer surface first (GT0 → Tier B payments → storage → teams → meta). Ship provider as a separate profile. Keep the action-plan + URL-elicitation security properties exactly as designed — they are the product. The original plan is 85 % right; the remaining 15 % is mostly “fewer, sharper tools + clearer boundary.”