# Xcelsior MCP — the provider axis

**Status:** plan, not implementation. Written 2026-08-01.
**Depends on:** [mcp-tool-surface-plan.md](./mcp-tool-surface-plan.md) GT0.
**Companion:** [mcp-tool-surface-research-prompt.md](./mcp-tool-surface-research-prompt.md).

> **No tool names are proposed in this document, deliberately.** The surface plan
> sets the rule — "I am not going to invent tool names from endpoint paths I
> haven't read" — and the enumeration is a gated deliverable produced by audit.
> This document defines *the axis, the boundary, and the sequence*. The tools
> themselves arrive from the research and are added under §4 below.

---

## 1. What the axis is

Xcelsior has three user classes. The MCP surface currently models **one and a
half** of them.

| Class | What they do | Tools today |
|---|---|---|
| **Consumer** | rents capacity — instances, serverless, billing | 30 |
| **Provider** | *supplies* capacity — hosts, reputation, payouts | **0** |
| **Operator** | Xcelsior staff — drain, evict, scheduler | 7, unlisted deployment |

The provider is not a minor omission. It is one side of a two-sided
marketplace, and it is invisible to every agent. The API is already there:
`routes/providers.py` carries 15 operations spanning registration, earnings,
payout, Stripe Connect and PayPal onboarding, incorporation status, and
onboarding resume/abandon. `routes/hosts.py` and `routes/host_admission.py` add
the supply-side lifecycle. The surface plan already anticipated this and
budgeted it: **Payouts / Connect (host) — now 0, target 6**, with the note
*"host side of the marketplace is invisible today."*

### Why this is not just "more tools"

The consumer axis and the provider axis differ in ways that change the design:

1. **Money flows the other way.** A consumer *spends*; a provider *receives*.
   Payout onboarding is a KYC flow through Stripe Connect — the single most
   sensitive surface in the product. It cannot be a form-filling tool.
2. **The same human can be both.** A user renting an A100 today may list their
   own idle 4090 tomorrow, on the same account. So the axis cannot be modelled
   as a disjoint tenant type; it is a *capability the same principal may hold*.
3. **Provider actions have blast radius on third parties.** Taking a host
   offline affects the consumer whose job is running on it. That is closer to
   the operator trust class than to the consumer one, and the existing
   approval machinery matters here.

---

## 2. The boundary decision (open — research settles it)

Three viable shapes. **This plan does not pick one**; §3 of the research prompt
asks for the argument, because the answer depends on how directory clients
snapshot `tools/list` and on evidence this document should not fabricate.

| Option | Shape | Argues for | Argues against |
|---|---|---|---|
| **A. Third deployment profile** | `XCELSIOR_MCP_TOOL_PROFILE=provider`, separate host | Matches the existing §4a architecture exactly; zero risk to the reviewed customer listing | A dual-role human needs two connectors; two surfaces to keep in sync |
| **B. Scope-gated subset of the public surface** | one connector; provider tools registered only when the principal holds provider scopes | One connector for a dual-role human — the natural fit for #2 above | Registration is per-connection, and a directory's frozen snapshot may not reflect scope-varying listings; needs verification per client |
| **C. Separate connector entry** | its own directory listing, own URL | Cleanest story for a pure provider; separate review | Doubles submission work, and Track B submission is already the expensive path |

**Constraint that survives all three:** whatever is chosen must preserve the
property the customer profile already has — operator tools are not *registered*
on the public surface, so they cannot be enumerated or called by name. The
provider surface must be equally structural, not a call-time permission check.

**Recommendation to test first:** B, because #2 (dual-role humans) is a real
product fact and A forces those users to install twice. But B is only viable if
scope-varying `tools/list` survives directory snapshotting — that is an
empirical question, and A is the safe fallback the architecture already
supports.

---

## 3. Sequence

Ordered so nothing is built before the thing that tells us what to build.

**P0 — inventory (blocks everything, no code)**
Extend GT0's endpoint inventory to classify every provider-side endpoint in
`routes/providers.py`, `routes/hosts.py`, `routes/host_admission.py`, and the
payout paths in `routes/billing.py` and `routes/connect.py` as `covered` /
`gap` / `internal` / `redundant`, each with a reason. Zero unclassified.

**P1 — journeys**
Write the provider journeys the way §1 of the surface plan writes consumer
journeys — *"list my idle GPU and get paid for it"* end to end. A journey that
needs a raw HTTP call is a gap. These journeys are the acceptance criteria for
every tool that follows, and they are what stops the surface growing tools that
close nothing.

**P2 — boundary decision**
Settle §2 from the research. Record the decision and its evidence here.

**P3 — tool enumeration**
From P0 + P1, produce the candidate table. Budget: **6 actions** for
payouts/Connect per the surface plan, plus whatever host-lifecycle tools P1
proves necessary — argued against the budget, not appended to it.

**P4 — payout onboarding via URL elicitation**
The Stripe Connect KYC flow is exactly the case §4.2 of the surface plan
designs for: the user must personally authorize, in their browser, with no
sensitive data in model context. This must emit `mode: "url"` and never
`mode: "form"`, asserted against a form-only client and a no-elicitation client.
**This is the highest-value single item on the axis** and the one most likely to
be got wrong.

**P5 — approval semantics for supply-side actions**
Decide which provider actions require an action plan. Taking a host offline
while a consumer's job runs on it is not a private act. Reuse the existing
approval machinery; do not invent a second one.

**P6 — gates**
The provider axis closes under the existing gate structure, not a parallel one:
GT1 (journey completeness — payouts is named there as a zero-coverage domain
that must close at least one journey), GT2 (payment safety), GT3 (scope
integrity — provider scopes minimal and enforced with real tokens), GT4
(surface discipline — still ≤100 total).

---

## 4. Where the tools get added

When the research returns, its provider rows land in the surface plan's §5
domain budget and candidate table — **not** in a separate provider tool list.
One budget, one ≤100 ceiling, one exclusion table. A second list is how a
surface quietly doubles.

## 5. Open questions the research must answer

1. Can a directory-listed connector vary `tools/list` by principal scope without
   breaking the provider's frozen snapshot? (Decides §2.)
2. Does any major client expose provider-style "supplier" connectors today, and
   how do they handle the dual-role user?
3. Is Stripe Connect onboarding via URL elicitation something any shipped MCP
   server does already, and what did they learn?
4. What is the real tool-count threshold at which selection accuracy degrades —
   the number that decides whether one surface or two is correct?
