# Xcelsior MCP — Curated Tool Surface Plan

Companion to [`mcp-enterprise-adoption-plan.md`](./mcp-enterprise-adoption-plan.md). This document
plans phase **X1** in detail. It defines *which* tools exist, *why* each one earns its place, and —
for anything that moves money — exactly how it stays safe.

---

## 1. Goal

> A set of tools that exposes everything needed for **truly complete and safe** use of the platform.

Two words carry the weight.

**Complete** does not mean "one tool per endpoint." It means: an agent holding only an OAuth token
and our published tool descriptions can carry any real user journey from start to finish without
ever dropping to raw HTTP, and without a human having to leave the conversation to go click
something we could have handled.

**Safe** does not mean "read-only." A read-only surface is not safe, it is useless — it pushes the
user to hand the agent an unrestricted API key instead, which is strictly worse. Safe means every
irreversible or costly action is bounded by something the *user* authorized, with an audit trail
that proves it.

### The completeness test

The surface is complete when every journey below closes inside MCP:

| Actor | Journey |
|---|---|
| **Customer** | discover GPU → estimate cost → **fund the wallet** → launch → monitor → fetch logs/artifacts → terminate → retrieve invoice |
| **Customer (long-run)** | pre-authorize a spend envelope **once** → agent runs a multi-hour job unattended without ever going broke or over-spending |
| **Host / provider** | onboard → publish capacity → observe utilization & reputation → **get paid out** |
| **Operator** | scheduler health → placement forensics → drain/undrain/evict → reconcile → retry |
| **Compliance officer** | prove residency → export attestation → answer a DSAR → pull the audit trail |
| **Team admin** | invite/remove members → set roles → review per-member spend |

Each journey gets a scripted end-to-end test in GT3. A journey that requires a raw `curl` is a
failing journey.

---

## 2. Baseline (audited 2026-07-29)

| Fact | Value |
|---|---|
| HTTP endpoints across `routes/*.py` | **514** |
| Registered MCP tools | **37** |
| Billing endpoints | **49** (`routes/billing.py`) |
| Billing tools | **3** — `get_wallet_balance`, `estimate_job_cost`, `list_invoices` |
| Stripe Connect endpoints | **11** (`routes/stripe_connect_v2.py`) |
| Stripe Connect tools | **0** |
| Declared scopes | 17 (`mcp/src/auth/scopes.ts`) |
| **Scopes defined but wired to zero tools** | **`billing:write`, `events:read`, `mcp_actions:approve`** |
| Installed MCP SDK | **`@modelcontextprotocol/sdk` 1.29.0** |
| SDK URL-elicitation support | ✅ `ElicitRequestURLParamsSchema`, `ElicitationCompleteNotificationSchema`, `ElicitationRequiredError` all present |
| Action-plan lifecycle | ✅ create → get → **approve** → revoke → execute (`routes/action_plans.py`) |

Two findings shape everything below.

**Finding A — the money surface is the biggest gap, and it is entirely a wiring gap.** 49 billing
endpoints and 11 Connect endpoints are live and tested; three of them are reachable by an agent.
`billing:write` was reserved in the scope enum and never used. Nothing needs to be *built* on the
API side for most of what follows.

**Finding B — the best-possible payment UX is unblocked today.** The flow you described with
Stripe — tool call opens a browser, you fill in the sensitive part, it redirects back, the job
completes — is a standardized MCP mechanism: **URL Mode Elicitation**, introduced in spec revision
`2025-11-25` as [SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1036).
Our installed SDK already implements it. No dependency bump, no custom protocol.

---

## 3. Curation rules

These are the rules that keep the surface from becoming either anaemic or bloated. Every proposed
tool is checked against all seven.

1. **Tool granularity ≠ endpoint granularity.** Tools are named after *jobs*, endpoints after
   *resources*. 49 billing endpoints collapse to ~13 billing tools because the agent's job is "add
   funds," not "choose between Stripe, PayPal, crypto and Lightning" — that's a `method` enum on
   one tool, plus a discovery tool that reports which rails are enabled.
2. **No tool exists to make the catalog look bigger.** Every tool must appear in at least one
   §1 journey. Tools that fail this are cut, and the cut is recorded with a reason.
3. **Every tool an LLM can misuse gets a `confirm` preview or an action plan.** Never both, never
   neither. Cheap-and-reversible → neither. Destructive-but-bounded → `confirm:false` preview.
   Costly or irreversible → action plan.
4. **Description quality is a shipping requirement.** Each description states what it does, when to
   prefer it over its neighbours, what it costs, and what it will refuse. A reviewer's model reads
   this and nothing else.
5. **Annotations must be true.** Fixes GAP 6 from the adoption plan: `openWorldHint` is currently
   hardcoded `false` for every tool, and `drain_host` is absent from `DESTRUCTIVE`. Both wrong.
6. **≤100 actions, hard.** Gemini Enterprise caps custom MCP data stores at 100 actions. We budget
   to **~92** to leave headroom, and treat the cap as a design constraint rather than a surprise.
7. **Never expose admin, test, or bulk-mutation endpoints.** `reset-testing`, `free-credits` grant,
   `bill-all`, raw webhook receivers, and everything in `routes/admin.py` stay off the surface
   permanently. These are recorded in §7 so the exclusion is deliberate and reviewable.

---

## 4. The payments architecture

This is the centrepiece and the differentiator. The design goal is the one you set: **the absolute
best flow for user convenience that is possible**, without weakening anything.

### 4.1 Three tiers of money movement

| Tier | What it is | Mechanism | Scope | Sensitive data in model context? |
|---|---|---|---|---|
| **A — Observe** | balances, history, forecasts, invoices, pricing, analytics | plain tool call | `billing:read` | never |
| **B — Out-of-band** | user must personally authorize (add a card, deposit, onboard for payouts) | **URL Mode Elicitation** | `billing:read` | **never — by protocol** |
| **C — Envelope** | agent spends autonomously inside limits the user pre-authorized | action plan + `billing:write` | `billing:write` | never |

The tiers are not a spectrum of trust in the model. They are a statement about *who authorized
what, and when*. Tier B is authorized in the moment, by the human, in their browser. Tier C is
authorized in advance, explicitly, with caps.

### 4.2 Tier B — the URL elicitation flow

This is the Stripe experience you described, done to spec.

```
Agent   → create_wallet_deposit(amount_usd: 50, method: "card", idempotency_key: "...")

Server  → POST /api/billing/payment-intent            (existing endpoint, unchanged)
        → elicitation/create {
            mode: "url",
            message: "Complete your $50.00 USD deposit to your Xcelsior wallet.",
            url: "https://checkout.stripe.com/c/pay/cs_live_...",
            elicitationId: "elic_..."
          }

Client  → shows the server name, the reason, and the full URL with the domain highlighted
        → asks for explicit consent  (MUST NOT prefetch, MUST NOT auto-open, MUST use the
                                      system browser rather than an embedded webview)
User    → consents; browser opens; pays on Stripe's page
        → card details never touch the MCP client, the model, or our logs

Stripe  → webhook → wallet credited          (existing handler, unchanged)
Server  → receives {action: "accept"}; awaits deposit confirmation with a bounded timeout
        → optionally emits an ElicitationComplete notification
        → returns structured: { deposited: 50.00, currency: "USD",
                                new_balance: 53.20, transaction_id: "..." }

Agent   → "Topped up. Balance is $53.20 — launching your job now."
```

Why this is the right answer and not merely a convenient one: the spec **forbids** using form-mode
elicitation for API keys, tokens, passwords, or payment credentials. Asking the model to collect a
card number is not a shortcut, it is a spec violation and a PCI problem. URL mode exists precisely
so the sensitive leg happens out-of-band.

**Tier B tools, and the endpoints they already have:**

| Tool | Existing endpoint producing the URL |
|---|---|
| `add_payment_method` | `POST /api/billing/setup-intent` |
| `open_billing_portal` | `POST /api/billing/portal-session` |
| `create_wallet_deposit` | `POST /api/billing/payment-intent` · PayPal · crypto · Lightning (one `method` enum) |
| `start_payout_onboarding` | `GET /api/connect/accounts/{id}/onboarding-link` |

Every one of those endpoints already returns a URL. The work is the elicitation wrapper, the
webhook-confirmation wait, and the fallback below — not new payment plumbing.

### 4.3 The honest fallback — a hard gate, not a nicety

Elicitation is a **client** capability, negotiated at `initialize`. Not every client supports URL
mode; some support form only; some support none. The server must branch on
`getClientCapabilities()?.elicitation` and behave correctly in all three cases.

- **URL mode supported** → the flow above.
- **Not supported / form-only** → return a **structured, actionable result** carrying the URL as
  text: *"Open this link to complete your $50 deposit, then call `get_wallet_balance` to confirm:
  https://…"*. The typed `ElicitationRequiredError` in SDK 1.29 is the primitive for this.
- **Never** → fall back to form mode and ask for card details. Spec-forbidden. This is asserted in
  GT2 by a test that runs the payment tools against a form-only client and fails if any elicitation
  request is emitted with `mode: "form"`.

There is no silent degradation anywhere in this path. A client that can't do the convenient thing
gets told exactly what to do instead.

### 4.4 Tier C — the spend envelope (the highest-value feature here)

The Tier B flow is excellent for a one-off top-up. It is the wrong shape for a six-hour training
run, because it interrupts.

You already have the endpoint: `POST/GET /api/v2/billing/auto-topup`. Wired to MCP it becomes the
thing that lets an agent run unattended:

```
configure_auto_topup(
  threshold_usd: 10,        # when balance drops below this…
  topup_usd:     50,        # …add this much
  monthly_cap_usd: 200,     # …never exceeding this in a calendar month
  expires_at: "2026-08-31"  # …and stop entirely after this date
)
```

Authorized **once**, by the human, and then never again. Inside that envelope the agent transacts
freely; outside it, it cannot transact at all. This is the fal.ai pattern you used, and it is also —
not by coincidence — the same idea as Google AP2's *Intent Mandate*: a signed, bounded, auditable
grant of spending authority.

Setting or raising the envelope is itself a Tier C action: it goes through the existing
`launch-plans` propose → **approve** → execute lifecycle (generalized to `spend-plans`), requires
`billing:write`, and requires a payment method already on file — if there isn't one,
`configure_auto_topup` first returns a Tier B elicitation to `setup-intent`, then resumes.

Additional Tier C tools: `reserve_capacity` (`POST /api/pricing/reserve`), `request_refund`
(`POST /api/billing/refund`), `initiate_payout` (host-side, Connect).

**Non-negotiables for Tier C:**
- Required `idempotency_key` on every tool. The `_meta["xcelsior/idempotency"]` contract field
  already exists in `mcp/src/tools/contracts.ts` — enforce it, don't just declare it.
- Lowering a cap or revoking an envelope needs **no** approval. Only raising does. Safety must
  never be harder than risk.
- Every envelope-funded charge writes an audit row linking back to the approving plan ID. "Which
  human authorized this dollar?" must be answerable by a single query.

### 4.5 On ACP / AP2 / x402 / MPP — deliberately not adopting, deliberately staying compatible

The agentic-payments standards landscape is genuinely unsettled: ACP (OpenAI + Stripe) covers
checkout, AP2 (Google) covers authorization mandates, x402 (Coinbase) covers stablecoin settlement,
MPP (Stripe + Tempo, March 2026) covers streamed micropayments. Mastercard, Visa **and Google** are
all premier members of the x402 Foundation — which tells you nobody is confident who wins.

Adopting one now is a bet with no payoff: none of our target directories require any of them.

What we do instead costs nothing and preserves every option: model the internal flow as
**intent → bounded envelope → charge → signed audit record**, which is the shape all of them
assume. Our action-plan lifecycle plus the existing `mcp/tool-audit` trail is already most of AP2's
audit model. When the market picks a winner, it's an adapter — not a rewrite. Recorded as a
non-goal so it doesn't get quietly adopted later without a decision.

---

### 4.6 Positioning: global-first, residency selectable  *(applied 2026-07-29)*

A distributed GPU marketplace needs supply wherever it is cheapest. Framing the platform as
Canadian narrows the buyer *and* the host pool for no gain, and it was baked into the MCP surface
in five places — including one that was quietly producing wrong numbers.

**Changed:**

| File | Was | Now |
|---|---|---|
| `mcp/src/server.ts:30` | "Canadian data residency and PIPEDA compliance are supported — prefer CA regions when required." | Marketplace framing: hosts compete on price · instance vs. per-token serverless · spot as the default for checkpointable work · discover live rates before launching · residency passed explicitly and **verified**, never assumed |
| `mcp/src/tools/billing.ts:58` | `is_canadian: z.boolean().default(true)` | **parameter removed**; `is_canadian: false` pinned in the request body |
| `mcp/src/lib/guardrails.ts:46` | `require_canada?: boolean` → hardcoded "prefer `ca-east`" note | `require_residency?: string` → generated note naming the requested region and instructing verification |
| `mcp/src/tools/guardrails.ts:37` | `require_canada: boolean` | `require_residency: string` (region code, optional) |
| `mcp/src/prompts/playbooks.ts:53` | prompt `ca-fine-tune`, "Canadian fine-tuning job" | prompt `fine-tune`, optional `require_residency` arg |

**The bug.** `estimate_job_cost` defaulted `is_canadian` to `true`, so every cost estimate applied
Canadian AI Compute Access Fund rebate math. **That program has ended**, which makes the default
unambiguously wrong for everyone: every estimate the MCP served was **understating real cost** by a
rebate no customer can claim. There is no eligible cohort left, so the parameter was removed from
the tool entirely rather than demoted to opt-in, and `is_canadian: false` is pinned in the request
body to neutralise the API's own default.

**Dead program still live API-side — separate cleanup, tracked here.** The fund ending leaves
rebate machinery running in five places:

| Location | What it is |
|---|---|
| `routes/billing.py:1221` | `EstimateRequest.is_canadian: bool = True` — the wrong default itself |
| `routes/billing.py:843` | `GET /api/billing/export/caf/{customer_id}` — a whole export endpoint for the fund |
| `billing.py:3955` | rebate documentation generator |
| `reputation.py:1107` | a second estimate path carrying the same rebate preview |
| `billing.py:4` | module docstring advertising "rebate-ready invoice exports" |

The MCP path is correct regardless of all of it, because the flag is now sent explicitly. But the
dashboard, the CLI, and any direct API consumer still inherit `= True` and are still quoting
rebated prices. That is a live pricing-accuracy issue on those surfaces and wants its own change —
flipping the default, then deciding whether the CAF export endpoint is retired or kept read-only
for customers reconciling historical invoices. `export/caf` is on the §7 never-expose list either
way.

**What replaced the positioning, and why these claims are safe.** No hardcoded percentages, no
comparative claims against named competitors, no numbers that go stale. Every quantitative
statement points at a tool that returns live data:

| Differentiator | Where the number comes from |
|---|---|
| Independent hosts competing on price | `search_marketplace`, `list_available_gpus` |
| Spot materially below on-demand | `get_spot_prices` |
| Per-token serverless on open-weight models, zero idle cost | `/api/v2/serverless/preset-token-pricing` — **real endpoint, currently unexposed to MCP; becomes `list_serverless_model_pricing` in the §5 budget** |
| CAD-denominated pricing | `get_pricing_reference` |
| Sovereignty-vetted hosts (35% premium, `jurisdiction.py:108`) | priced via `estimate_job_cost(sovereignty:true)` — a premium tier for buyers who need it, not the platform's identity |

This is deliberate. The `instructions` string is read by reviewer models during directory
submission; an unverifiable comparative claim there is both a review risk and exactly the kind of
stale hardcoded assertion the honesty rules forbid. Stating facts and pointing at tools lets the
price advantage show up in the data instead of being asserted.

**Residency became stronger, not weaker.** `require_canada: boolean` could only express one
jurisdiction and answered with a hardcoded "prefer `ca-east`". `require_residency: string` accepts
any region, and the returned note instructs the agent to *confirm the selected host reports a
matching jurisdiction before launching* rather than asserting that a region satisfies the
requirement. That is a better compliance posture — and it is what makes GDPR/EU residency a
configuration rather than a rewrite when hosts land there.

**Out of scope here** (flagged, not changed): `chat.py:91`, `ai_assistant.py:3792` and
`ai_assistant.py:4581` still describe the product as "Canada's distributed GPU compute marketplace"
and instruct "Use Canadian English". Those are the web assistant's prompts, not the MCP surface.
Same treatment, separate change.

**Verification:** `npx tsc --noEmit` clean. `grep -rniE 'canad|pipeda|rebate' mcp/src/` returns only
the explanatory comment on the pinned `is_canadian: false` — no Canada-specific positioning or
behaviour remains on the MCP surface.

⚠ `should_i_run_this` and `estimate_job_cost` are **breaking schema changes** to live tools. Correct
to do now, before directory submission creates external dependents — but the stdio package version
must bump and the change must be noted in `mcp/README.md`.

---

## 5. Domain budget

| Domain | Now | Target | Notes |
|---|---:|---:|---|
| Discovery & pricing | 5 | 8 | + spot quote, reserved plans, regions/residency |
| Instance lifecycle | 13 | 14 | mostly complete already |
| Serverless / inference | 5 | 10 | 59 endpoints, thinnest coverage relative to size |
| Storage & volumes | 0 | 5 | 14 endpoints, **zero** tools — a journey blocker |
| **Billing & payments** | **3** | **13** | §4 |
| **Payouts / Connect (host)** | **0** | **6** | §4; host side of the marketplace is invisible today |
| Monitoring & events | 4 | 8 | wires the unused `events:read` scope |
| Ops / control plane | 7 | 10 | |
| Compliance & residency | 0 | 6 | selectable per workload; global-first, see §4.6 |
| Teams & access | 0 | 5 | |
| Meta (plans, status, capabilities) | 1 | 6 | wires the unused `mcp_actions:approve` scope |
| **Total** | **37** | **~91** | under the 100-action Gemini cap with headroom |

Storage, compliance, teams and payouts are all **zero-coverage domains** — no tool reaches them at
all today. Compliance in particular has the inverse of the usual problem: the capability exists in
the API (`routes/jurisdiction.py`, `routes/compliance.py`) and no tool can reach it, so an agent
can neither honour a residency requirement nor prove one was honoured.

### How the per-domain list actually gets produced

I am not going to invent tool names from endpoint paths I haven't read. The enumeration is a
**gated deliverable**, produced by an audit rather than a guess:

1. A script walks the FastAPI route table and emits every endpoint with method, path, auth
   dependency, and docstring → `docs/generated/endpoint-inventory.md`.
2. Each endpoint is tagged against the §1 journeys: `covered` (a tool reaches it), `gap` (a journey
   needs it, no tool), `internal` (never exposed — with a reason), `redundant` (folded into another
   tool).
3. Every `gap` becomes a tool candidate; candidates are merged per rule 1 and checked against rules
   2–7.
4. Output is a reviewed candidate table with the domain budget enforced.

The exclusion reasons are as much a deliverable as the inclusions — that table is what makes the
surface defensible to a security reviewer, and it is what stops the list quietly growing later.

---

## 6. Gates

Consistent with the adoption plan: outcome-gated, no self-attestation, no partial credit.

**GT0 — inventory & annotation truth**
Endpoint inventory generated and committed; every endpoint classified with a reason; zero
unclassified. Annotation audit passes: `openWorldHint` reflects reality per tool (not a hardcoded
`false`), `drain_host` present in `DESTRUCTIVE`, every tool's annotations agree with what it
actually does — verified by a test that cross-checks annotations against observed behaviour, not by
reading the contract file.

**GT1 — journey completeness**
Every §1 journey passes as a scripted end-to-end test against a live staging tenant, using only
MCP tool calls. A journey that needs a raw HTTP call fails the gate. Zero-coverage domains
(storage, compliance, teams, payouts) each close at least one journey.

**GT2 — payment safety**
- Tier B tools emit `mode: "url"` and **never** `mode: "form"` — asserted against a form-only
  client and a no-elicitation client.
- No card number, token, or secret appears in any tool result, log line, or audit row — the
  existing `FORBIDDEN` redaction regex extended and canary-tested with fake PANs.
- Replaying any Tier C call with the same `idempotency_key` produces exactly one charge.
- Raising a cap requires approval; lowering one does not — both asserted.
- An envelope-funded charge is traceable to its approving plan ID in one query.
- Expired and exhausted envelopes both refuse cleanly, with a structured error naming the reason.

**GT3 — scope integrity**
Every tool's declared scope is minimal and enforced: a token holding only `billing:read` is
rejected by every Tier C tool; a token missing `mcp_actions:approve` cannot approve a plan. No
scope remains defined-but-unused, or it is deleted from the enum. Tested with real tokens against
the live server, not with mocks.

**GT4 — surface discipline**
Total actions ≤ 100. Every tool maps to a §1 journey. Every description states purpose, cost, and
refusal conditions. A reviewer-grade model, given only `tools/list` output and no other context,
completes three unseen journeys end to end — the real test of whether the descriptions work.

---

## 7. Permanently excluded

Recorded so the exclusion is a decision, not an oversight. Reviewed at each gate.

| Endpoint / group | Why never exposed |
|---|---|
| `POST /api/billing/wallet/{id}/reset-testing` | test-only; trivially destroys balance state |
| `POST /api/billing/free-credits/{id}` | grants money; admin privilege, not user capability |
| `POST /billing/bill-all` | unbounded bulk mutation; no safe agent framing |
| `POST /api/billing/paypal/webhook`, `POST /api/connect/webhooks` | machine-to-machine receivers; not agent surface |
| `GET /api/billing/export/caf/{id}` | Canadian AI Compute Access Fund export; the program has ended (§4.6) |
| all of `routes/admin.py` (26) | cross-tenant privilege; outside the MCP trust boundary |
| `routes/static.py`, `GET /connect/dashboard|storefront|success` | HTML page renderers |
| raw `routes/terminal.py` shell exec | arbitrary code execution as the agent's caller; the sandboxed instance path is the supported route |

---

## 8. Sequencing

```
GT0  inventory + annotation truth      ← pure audit, no API changes, unblocks everything
  │
  ├─ GT1  journey completeness         ← storage / compliance / teams / payouts
  │
  └─ GT2  payments  ── depends on GT0 only
       │   4.2 Tier B URL elicitation  ← SDK 1.29 already supports it; highest UX payoff
       │   4.4 Tier C spend envelope   ← wires the unused billing:write scope
       │
      GT3  scope integrity
       │
      GT4  surface discipline          ← final ≤100 count and description review
```

GT0 is pure audit and blocks nothing else. **Tier B payments can start immediately after it** — no
dependency bump, no new endpoints, and it delivers the single most visible improvement to the
product's agentic story.

Nothing here loosens the approval model, removes the `client_credentials` front door, or adds a
tool that doesn't close a journey. Those remain non-goals from the adoption plan.
