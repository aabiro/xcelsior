# Xcelsior MCP tool surface — combined plan

**Status:** plan of record for the tool-surface track, superseding the domain
budget in [mcp-tool-surface-plan.md](./mcp-tool-surface-plan.md) §5 where they
conflict. Written 2026-08-02 from three independent research passes
(`mcp_tools_chatgpt.md`, `mcp_tools_gemini.md`, `mcp_tools_grok.md`), each read
in full.

---

## 0. Stop-ship: the authorization layer is broken, and it is worse than reported

The ChatGPT pass called `userHasScope` a release blocker. **I verified it against
the running code, and confirmed a second half it did not find.** Nothing in this
plan may be built before this is fixed, because every proposal in all three
documents adds 25–45 tools on top of this layer.

**Layer 1 — MCP.** [mcp/src/auth/scopes.ts:67](../mcp/src/auth/scopes.ts#L67):

```ts
if (userScopes.includes("api")) return true;
return required.some((s) => userScopes.includes(s));
```

Two independent defects. `api` is a universal bypass, and `.some()` accepts **any
one** of the required scopes. All 41 contracts list `"api"`. Executed:

| Check | Result |
|---|---|
| `userHasScope(["gpu:read"], ["instances:write","gpu:read","marketplace:read","api"])` | **true** — a read-only scope authorizes a spending write |
| `userHasScope(["billing:read"], ["instances:write","billing:read","api"])` | **true** — `run_training_job` with no write scope |
| `userHasScope(["api"], ["hosts:evict"])` | **true** — bypasses an operator scope |

**Layer 2 — the API, which I expected to be the real gate.** It is not.
[routes/_deps.py:1231](../routes/_deps.py#L1231) uses correct all-of semantics,
but carries the *same* `api` bypass, and — decisively — returns immediately
unless `grant_type == "client_credentials"`. An agent-key principal
([oauth_service.py:724](../oauth_service.py#L724)) has **no `grant_type` field at
all**. Executed:

| Principal | `_require_scope(instances:write, billing:write)` |
|---|---|
| agent key with only `gpu:read` | **passed** |
| `client_credentials` holding `api` | **passed** |

So for the Quick Connect agent key — the exact credential our quickstarts tell
users to paste — **declared scopes are decorative at both layers**. This matters
because Quick Connect deliberately issues a *narrowed* set
([oauth_service.py:1275](../oauth_service.py#L1275): no `api`, no
`billing:write`, no operator scopes). The reduction is intended; the enforcement
is absent.

What still holds: operator tools are not *registered* on the customer profile,
so they cannot be enumerated or called there regardless of scope. That structural
gate is doing the work the scope check is failing to do — which is precisely why
the trust boundary must stay structural and never become "just a scope check."

**Fix before anything else:**

1. Replace flat arrays with `{ allOf?, anyOf? }` (the ChatGPT pass's shape is
   correct and I would adopt it verbatim).
2. Delete `api` as an authorization value everywhere. It is a token class, not
   authority. Remove it from all 41 contracts.
3. Make `_require_scope` apply to every machine principal, not only
   `client_credentials`. Key off "has explicit scopes", not grant type.
4. Regression tests at both layers, each verified to fail before the fix.

---

## 1. What each pass got right

| Pass | Its strongest contribution | Where it is wrong |
|---|---|---|
| **ChatGPT** (70K) | The authorization finding. Counting each deployment's budget independently. The `add_payment_method` blocker. Refusing to invent a degradation threshold. Not exposing `approve` to the model. | Slightly over-built at 84 contracts; six compliance tools is more than the journeys need. |
| **Grok** (14K) | Ruthless consolidation — fold compliance into tools that already take the parameter. Cleanest competitive capability list. Honest self-assessment ("85% right"). | Cites some endpoints loosely ("estimate paths in billing"); tables are illustrative, not complete. |
| **Gemini** (37K) | The strategic challenge to data sovereignty, and composite tools that collapse multi-hop sequences into one call. | Internally contradictory: argues 20–30 tools cause "catastrophic collapse", then ships 91. Overstates BFCL. Marks `fund_compute_wallet` destructive, which is wrong. |

### The disagreement worth resolving

| Question | ChatGPT | Grok | Gemini | **Decision** |
|---|---|---|---|---|
| Total contracts | 84 across 4 surfaces | ~70 | 91 flat | **~78 across 4 surfaces**, flagship ≤60 |
| Compliance domain | keep 6 | fold to 3 | **delete entirely** | **keep 2** (see §2) |
| Provider boundary | Shape C | Shape A | Shape A | **A and C are not alternatives** (see §3) |
| Degradation threshold | no defensible number; measure | ~15–25 | 20–30 "collapse" | **measure it** — ChatGPT is right |

On the threshold: two passes gave confident numbers and one refused. The refusal
is the defensible position. The cited evidence (MCP-Atlas at 220 tools, a
121-tool retrieval study, OpenAI's token reduction) establishes that large flat
catalogs hurt — none isolates a breakpoint. We have our own eval harness
(`scripts/mcp_tool_eval.py`, 31 cases). Extending it is cheaper than trusting
someone else's number, and it measures *our* tools against *our* prompts.

---

## 2. Compliance: keep two, not six, and not zero

Gemini's argument is half right. It is **correct** that the Canadian AI Compute
Access Fund has ended — our own plan already records this — and that
`is_canadian` defaulting true was making estimates understate cost. It is
**wrong** that residency should be deleted. Xcelsior Compute Inc. is Canadian,
PIPEDA and Law 25 are live procurement questions for Canadian enterprise buyers,
and "prove where my training data ran" is a claim no GPU marketplace competitor
answers. Deleting it discards the differentiator along with the dead rebate.

Grok's mechanism is the right one: residency is a **constraint on placement**,
not a domain of its own. `should_i_run_this` and `estimate_job_cost` already
accept `require_residency`.

**Decision — residency becomes a parameter, plus exactly two first-class tools:**

- `get_residency_trace` — the auditable artifact proving where a workload ran.
  This is the differentiator; without it the constraint is unverifiable.
- `get_compliance_posture` — one evidence read for procurement (attestation,
  trust tiers, tax, platform controls), replacing six catalog reads.

Everything else — jurisdiction host search, PIA check, retention posture —
becomes a parameter on the placement and quote tools, or waits for a journey
that demands it. **6 → 2.**

---

## 3. The provider boundary: A and C are the same decision at different layers

All three passes framed A (separate deployment profile) and C (separate directory
listing) as competing. They are not. **A is the deployment mechanism; C is the
distribution decision.** You deploy with `XCELSIOR_MCP_TOOL_PROFILE=provider` —
which the §4a architecture already supports and this codebase already implements
— and you list that deployment separately. ChatGPT's own sentence concedes it:
*"Shape A is acceptable as an interim implementation profile, but if the provider
surface is strategically important it should become a real listing."*

**Shape B is rejected**, and the evidence is decisive rather than merely
cautious: OpenAI documents a frozen tool snapshot after approval, with schema
mismatches causing call failures; Microsoft requires an immutable listing for
admin review. A dual-role user who gains provider scopes would not see new tools
in ChatGPT until they uninstalled and reinstalled the connector. That is a
support burden disguised as a feature.

The dual-role objection dissolves on inspection: one identity authorizing two
purpose-specific connectors is normal, and it is how the same person already uses
two applications backed by one account.

**Four surfaces, counted independently** — the old budget's error was totalling
operator tools into a public surface that never serves them:

| Surface | Users | Tools | Posture |
|---|---|---:|---|
| Xcelsior Compute | consumers | ≤60 | public, flagship |
| Xcelsior Provider | providers | ~14 | separate public listing |
| Xcelsior Control Plane | operators | 8 | unlisted, unchanged |
| Xcelsior Knowledge | anyone | 2 | optional read-only listing |

The provider surface is **supply operations**, not six payout functions: host
registration and admission, marketplace publication, spot floor configuration,
reputation and SLA reads, earnings, payout. ChatGPT is right that publishing
capacity — not getting paid — is the supply-side flagship.

---

## 4. Payments: one blocker nobody else caught

`add_payment_method` cannot be built as designed. `POST /api/billing/setup-intent`
returns a Stripe **client secret**, not a browser page. Stripe's guidance is
explicit that client secrets must not be logged or embedded in URLs, so wrapping
this endpoint in URL elicitation produces a non-compliant flow, not a convenient
one.

**Prerequisite:** a first-party hosted payment page (`pay.xcelsior.ca/...`) that
resolves the intent server-side from an authenticated session and renders
Stripe's Payment Element. URL elicitation then points at *that*, never at Stripe
with a secret in the query string. Card-adding tools stay blocked until it
exists. PayPal and crypto rails are unaffected and can ship first.

**URL elicitation, settled facts:** SEP-1036 is Final Standards Track in the
`2025-11-25` revision. Server SDK support (1.29.0) says nothing about client
support, which is uneven — Microsoft Copilot Cowork documents form mode and
explicitly *not yet* URL mode; every other directory client is **UNVERIFIED** in
official docs. So it must be capability-negotiated, never assumed.

Three rules that are not negotiable, and on which all three passes agree:

1. **Never fall back to form mode for payment credentials.** Prohibited by spec
   and by PCI.
2. **`accept` means the user consented to navigate. It does not mean the flow
   succeeded.** Success is derived only from the processor's authoritative state
   via signed webhook. The client and the model are untrusted here.
3. **The fallback must be conspicuously incomplete** — `completed: false`,
   `operation_executed: false`, and prose that opens with "Not completed." A
   fallback that reads like success is worse than an error.

Payout onboarding (KYC) is the correct use of the mechanism, but URL elicitation
is the *transport*, not the onboarding app: elicit to an authenticated Xcelsior
page, which creates the Stripe Account Link. Returning from `return_url` proves
nothing — Stripe says so directly — so completion comes from `account.updated`.
The tool reports `pending_requirements` / `restricted` / `enabled` and never
returns KYC content.

---

## 5. My additions

**5.1 Do not expose `approve` to the model.** I previously called the unused
`mcp_actions:approve` scope a defect. It is not. Wiring approval to the same
agent that prepared the plan collapses the separation the approval mechanism
exists to create. The scope belongs to a human-bound session or first-party UI.
A scope with no tool is the correct state here.

**5.2 Generate the surface from one registry.** The 37-vs-39 drift the ChatGPT
pass spotted is a symptom: contracts, scopes, annotations, descriptions, and
`tool-surface.json` are five files that must agree and are maintained by hand.
`tool-surface.json` is currently a *snapshot* with a drift check. Invert it —
make the policy registry the **source**, and generate contracts, annotations,
advertised scopes, and the manifest from it. Then the drift check becomes
unnecessary because drift becomes unrepresentable.

**5.3 Sequence the expansion behind the fix.** Every pass proposes adding 25–45
tools. Adding them to the authorization layer described in §0 multiplies the
blast radius of a bug that already exists. Order is: fix auth → regenerate from
registry → expand one domain at a time, each behind its own eval run.

**5.4 Idempotency before inference.** `run_serverless_inference` must not be
auto-retried until the backend deduplicates; a retried inference is a second
charge. This is the one place where "safe to retry" is a billing question, not a
correctness question.

**5.5 The competitive capabilities worth building** — merged from all three,
ranked by gap widened per unit of work. Each is a *workflow over primitives*, not
necessarily a named tool, and should earn a name only if an eval shows the name
improves completion:

| Rank | Capability | Why |
|---:|---|---|
| 1 | **Spend runway monitor + auto-pause** (Grok 1, Gemini 2, ChatGPT 8 — all three converged) | Wallet depletion vs. projected job completion, acting before zero. Turns "never wake up broke" into a product property. Wires `billing:write` honestly. |
| 2 | **Checkpoint-aware spot migration** (Grok 2, ChatGPT 2) | Snapshot, stop, relaunch cheaper, verify placement. Converts the marketplace's interruptibility from a liability into the reason to use it. |
| 3 | **Residency-locked launch + trace** (ChatGPT 1) | The compliance differentiator §2 keeps, made executable rather than documentary. |
| 4 | **Provider yield optimizer** (ChatGPT 3) | Admission + reputation + SLA + spot preview → a recommended floor. Gives a solo provider a revenue manager, and makes the provider listing worth installing. |
| 5 | **Failure-to-refund resolution** (ChatGPT 5) | Timeline, logs, lease, events, eligibility → a refund plan. Turns a support ticket into an auditable workflow. |
| 6 | **Reusable environment snapshot** (Gemini 5) | Configure once, launch a sweep of identical nodes. The highest-leverage idea Gemini contributed. |

Deliberately deferred: HPC/Slurm provisioning and cloudburst (Gemini 1, 4) — both
need architecture and cost-policy review well beyond a tool definition.

---

## 6. Sequence

```
S0  Authorization fix (§0)          ← stop-ship; nothing proceeds past this
 │
S1  Single-source registry (§5.2)   ← makes the expansion maintainable
 │
S2  Hosted payment page (§4)        ← unblocks the card rail; PayPal/crypto can precede
 │
 ├─ S3  Storage + artifacts         ← largest zero-coverage domain
 ├─ S4  Billing Tier A/B/C          ← wires billing:write behind real approval
 ├─ S5  Provider surface            ← separate listing, per mcp-provider-axis-plan.md
 └─ S6  Monitoring + teams          ← wires events:read
 │
S7  Selection eval at 39/50/60/78   ← ship the largest surface only if first-tool
                                       accuracy holds and unsafe-write does not regress
```

Gates GT0–GT4 from the surface plan still apply and are unchanged. This plan
changes *what* is built and in *what order*, not the standard it is held to.

---

## 7. Open, and honestly so

- Whether any directory client supports URL elicitation is **UNVERIFIED** for all
  but Microsoft (a documented "not yet"). Settled only by testing each client
  against a real server. Until then, the fallback path is the primary path.
- No public example exists of a production MCP server completing Stripe Connect
  onboarding via SEP-1036. We would be early, which is a reason for the
  conspicuously-incomplete fallback, not a reason to wait.
- The degradation threshold is ours to measure. Nobody has published a number
  that survives scrutiny.
