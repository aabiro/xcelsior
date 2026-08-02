<!-- residency-guard: documents-removal — records why the residency model was deleted -->
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

### Fixed 2026-08-02

1. **`{ allOf?, anyOf? }` replaces flat arrays.** `allOf` is cumulative and is
   the default; `anyOf` is reserved for the three tools whose subject may live
   in several domains (`get_mcp_action_status`, `search`, `fetch`).
2. **`api` removed from all 41 contracts.** It survives only as a *legacy broad
   grant* — it is the DCR default ([routes/auth.py:463](../routes/auth.py#L463)),
   so refusing it outright would revoke live credentials. It now satisfies
   tenant scopes and **never** an operator scope, so a tenant automation token
   can no longer reach `hosts:evict` or `mcp_actions:approve`.
3. **`_require_scope` now covers agent keys**, via an explicit machine-credential
   set rather than `grant_type` alone.
4. **The `|| ["api"]` fallback is gone.** Eight call sites resolved an
   unregistered tool to "requires only the broad grant" — fail-open on the one
   path that must fail closed. An unknown tool now denies.

**One correction worth recording.** My first attempt keyed enforcement off "has
explicit scopes", which broke 122 tests. Interactive sessions authenticated by
`authorization_code` carry OIDC identity scopes — `profile`, `email`,
`offline_access` — that say nothing about API authority, and denying every
browser request for lacking API scopes it never held is not enforcement, it is
an outage. The crude `grant_type` check was encoding a real distinction. The gate
is *machine credential*, not *carries scopes*.

**Two tests had pinned the bug in place**, which is how it survived a green
suite. `mcp/tests/unit/contracts.test.ts` had a case named *"does not grant
legacy api unless it is explicitly present"* whose assertion was
`expect(userHasScope(["api"], ["hosts:evict"])).toBe(true)` — the name stated
the rule, the assertion asserted its opposite. `scopes.test.ts` had *"allows api
wildcard"* and *"allows matching scope"*, which locked in the bypass and the
any-one-of semantics respectively. All three are rewritten to state the actual
rules, with the old assertions quoted so the history is not lost.

Regression coverage: `mcp/tests/unit/scope-enforcement.test.ts` (10 cases),
`tests/test_scope_enforcement.py` (13 cases, 3 confirmed to fail against the old
implementation), a cross-language drift test asserting the two operator-scope
sets agree, and a live-token escalation assertion in the real-stack E2E — GT3
asks for real tokens against a real server, because a mock is what passed while
production did not.

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

## 2. Compliance: zero residency tools. Grok and Gemini were right; I was wrong.

I originally wrote that Gemini was "half right" — correct that the Canadian AI
Compute Access Fund had closed, wrong to delete residency, because PIPEDA and
Law 25 are live procurement questions and "prove where my training ran" is a
differentiator. **That was wrong on the facts and wrong on the strategy, and it
is retracted.**

Wrong on the facts: the platform had already decided this. The `_cad` duplication
had been cleaned up behind the pivot, and the sovereignty premium had already
been zeroed in code (`_TIER_SOVEREIGNTY` mapped every tier to `0.0`). I read a
half-finished migration as an argument for keeping the thing being migrated away
from.

Wrong on the strategy, and this is the part that matters. A residency guarantee
is not a feature of a marketplace whose supply is independent hosts in arbitrary
countries. Selling one means either refusing most of your own supply or making a
promise you cannot keep. `privacy.s6_p3` said *"personal information of BC
residents remains in Canada at all times"* — with global hosts that is not a
positioning choice, it is a false statement in a live privacy policy. The
sovereignty tier also charged a **premium** for it, so the platform was billing
extra for a guarantee it structurally could not honour.

### What was removed

| Removed | Why it could not stay |
|---|---|
| `jurisdiction.py`, `routes/jurisdiction.py` | Canada-only province routing, Law 25/PIIDPA/FOIPPA/PHIPA constraints, residency traces, the closed fund |
| `require_residency` on MCP tools | A caller-supplied string that changed placement |
| `sovereignty_premium`, `SOVEREIGNTY_PREMIUM_PCT`, tier `pricing_multiplier` | Charged more for a guarantee the supply model cannot make |
| `is_canadian_compute`, the four fund invoice columns | Recorded a Canada-vs-rest split for a program that ended |
| `x-data-residency: CA`, `x-compliance-version` | Asserted CA residency on **every** response, including work running abroad |
| `XCELSIOR_CANADA_ONLY`, "Canada-Only Routing" setting | The gate itself |
| Tier-driven gVisor isolation | Let a caller influence placement by renaming its own tier |

### The two compliance tools that remain, and why they are the right two

Not a compromise between six and zero — a different question. The surviving
tools answer *"can I trust this platform with my workload?"*, which every buyer
asks, rather than *"where will it run?"*, which this marketplace deliberately
does not answer.

1. **`get_platform_controls`** — scoped tokens, human-approved action plans on
   destructive and spending operations, per-tool rate limits, the audit trail,
   and tax registration. This is what a security review actually asks for, and
   it is true regardless of geography.
2. **`get_workload_audit_trail`** — for a given workload: who authorised it,
   which plan was approved, what ran, on which host, and what it cost. It
   reports **where it did run** as a fact after the event. It never promises
   where it *will* run.

That distinction is the whole design. A trace is evidence; a residency
constraint is a promise. We can produce the first honestly and cannot produce
the second, so the surface offers exactly one of them.

Tax stays and is not compliance theatre: GST/HST is a real obligation of a
Canadian company, computed from the customer's billing address, and it is
unrelated to where a GPU sits. "Built in Canada" in the footer is a fact about
the company, not a claim about anyone's data.


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
