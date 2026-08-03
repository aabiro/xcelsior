# Xcelsior — agent-native implementation plan

**Status:** plan of record for building the tool surface.
**Written:** 2026-08-02. Approved on the strength of one sentence, which is
also this plan's acceptance test.

> An agent SSH'd into the GPU, talking to the platform at the same time,
> driving the controllers that run the environment. **You never have to leave
> the terminal.**

Everything below exists to make that sentence literally true, and to keep it
true. A phase ships only when the sentence is more true than it was before, and
when a test would fail if it stopped being true.

**Companions:** [mcp-tool-surface-synthesis.md](./mcp-tool-surface-synthesis.md)
(what to build and why), [mcp-tool-surface-plan.md](./mcp-tool-surface-plan.md)
(gates GT0–GT4), [mcp-provider-axis-plan.md](./mcp-provider-axis-plan.md),
[mcp-enterprise-adoption-plan.md](./mcp-enterprise-adoption-plan.md).

---

## 0. What has to be true for the claim to hold

The sentence has four load-bearing clauses. Three are the phases; the fourth is
the reason this plan is not just a feature list.

| Clause | What it requires | State today |
|---|---|---|
| *"SSH'd into the GPU"* | launch → running → **connected**, without a dashboard | endpoints exist, unreachable as tools (§0.1) |
| *"talking to the platform at the same time"* | the same session drives instances, spend, and state | the tool surface's whole job |
| *"driving the controllers"* | approval, drain, retry, reconcile — governed, not raw | approval machinery exists; operator split done |
| *"never have to leave"* | **every dead end has a lever inside the terminal** | closer than it looked — spending already works off-session (§0.2); the gap is a *declined* charge (§0.3) |

### 0.1 The access endpoints are authenticated but not authorized

`auto-launch`, `expose`, `stream-ticket` and `ssh/keygen` all sit behind
`_require_auth` — authentication, not authorization. There is no
`_require_scope` on any of them. Exposing them as tools today would mean either
bypassing the scope model or inventing the scopes at the tool layer, and a scope
enforced in one layer only is the exact defect the `api` wildcard was.

**Consequence:** scopes come first, in P0, or the access phase builds on sand.

### 0.2 The money levers: what actually needs a browser, and what does not

The first draft of this section said card payments needed a hosted page before
an agent could touch them. That was too broad, and the distinction matters
because it changes what ships first.

**Charging a card that is already on file is an ordinary API call.** The
platform already does exactly this in `check_low_balance_and_topup`
([billing.py:3602](../billing.py#L3602)):

```python
pi = _stripe_mod.PaymentIntent.create(
    amount=amount_cents, currency="cad",
    customer=w["stripe_customer_id"],
    payment_method=w["stripe_payment_method_id"],
    off_session=True, confirm=True,
)
```

That is a merchant-initiated off-session charge. No browser, no `client_secret`,
no card data anywhere near the caller. *"I'm running low, put $10 on my
account"* is that same call with a manual trigger instead of a threshold — and
*"use my Visa, not the Amex"* is one more argument. **It needs no elicitation at
all.**

So the real split:

| Action | Needs a browser? | Why |
|---|---|---|
| Top up on a saved card | **no** | off-session MIT; the machinery already runs nightly |
| Choose which saved card | **no** | `payment_method_id` is an argument, not a secret |
| Deposit via crypto / Lightning / PayPal | **no** | already `_require_scope` |
| Read balance, history, forecasts | **no** | plain reads |
| **Add a new card** | **yes** | PAN collection. Private by design, and it should stay that way |
| **Recover from an SCA challenge** | **yes** | the issuer demands the cardholder, in person |

Only the last two need a page, and the last one is not a convenience — it is a
correctness requirement, because of the defect below.

### 0.3 SCA failure is unhandled, and it silently drains wallets

An off-session charge can be refused with `authentication_required`: the card's
issuer requires the cardholder to authenticate before that payment completes.
The correct response is to bring them back on-session, once, to satisfy the
challenge.

**Nothing in this codebase handles that.** There is no `authentication_required`
branch, no `CardError` handling, no `requires_action` or `next_action` inspection
anywhere. The failure lands in a generic `except Exception`, increments an error
counter, and writes a log line nobody reads.

The customer-visible consequence: for any card whose issuer demands SCA,
**auto-top-up silently does not work**, and the wallet runs to zero while the
platform believes top-up is configured. The account then hard-stops mid-job.
This is a live defect independent of the agent surface, and it is the single
strongest argument for building the hosted page — not so an agent can take a
card number, but so a *declined* charge has somewhere to send the human.

### 0.4 Staging must be provisioned before P2 and P6 can run

P0 found that a missing or misspelled `XCELSIOR_ENV` disabled authentication,
exposed a signing secret that lives in the source tree, and made stored secrets
recoverable — four security decisions each defaulting to development. Resolution
now fails closed, and the startup gate enforces everywhere except an explicitly
named dev/test context.

**Staging is deliberately not exempt.** It holds real data, so it may not fall
back to the committed JWT secret or the deterministic Fernet key. It is also not
production: `routes/agent.py` grants staging an escape hatch it refuses to the
production VPS, and that distinction is audited, so `is_relaxed_env()` and
`is_production()` are not complements.

The consequence lands on this plan directly. **Gate P2 and Gate P6 both require
journeys against a live staging tenant**, and under the new rules that tenant
refuses to boot without a real asymmetric signing key
(`XCELSIOR_OAUTH_JWT_KEYS_JSON`) and a real `XCELSIOR_SECRETS_KEY`. Provision
both before P2, not at the gate: it is a small task now and a blocked acceptance
test later, and P2 is the phase that makes this plan's headline sentence true.

Scope note: the environment work was necessary and is outside P0 as written. It
ends at the four controls, the startup gate, and the guard on the pattern. It is
not a general secrets-management project.

## 1. How every phase is gated

Same shape each time, because the failures this codebase has actually produced
were failures of *evidence*, not of intent.

**Each phase ships with:**

1. **Behaviour tests, named for the behaviour.** `test_agent_key_cannot_exceed_its_scopes`, not `test_scopes_2`. The name is the specification.
2. **A confirmed-failing check.** Every regression test is run against the code *before* the fix and shown to fail. A test that has never failed is a guess.
3. **A live-credential path.** At least one assertion per phase runs a real token against the real server. GT3's rule: *a mock is what passed while production did not.*
4. **A refusal test.** What the phase makes *impossible* — the scope it denies, the plan it will not execute, the state it will not accept.
5. **An eval delta.** `scripts/mcp_tool_eval.py` re-run at the new tool count. First-tool accuracy must not regress, and unsafe-write selection must not rise.

**Honesty rules, written down because they were broken before:**

- **No budget may be raised to make a test pass.** Ratchets go down or the change does not land. `MAX_LEGACY_FLOAT_CAD_COLUMNS` reached 0 that way.
- **A guard may not be narrowed to make it pass.** The repo-wide vocabulary guard scanned ten filename suffixes and reported clean while four blog posts and a published docs page were full of what it was hunting. It now scans every file and proves it does with planted probe files.
- **A test that asserts a bug is a bug.** `"does not grant legacy api unless it is explicitly present"` asserted the opposite of its own name for months.
- **Green does not mean done.** State what was *not* run, and what a passing suite does not cover.

---

## P0 — Foundations: scopes, one registry, and the drift gates
*Nothing else is safe to build until the surface cannot drift.*

**Backend**

- **Scope the access endpoints.** `instances:connect` (auto-launch, expose, stream-ticket), applied with `_require_scope` at both layers.
- **Deviation, recorded deliberately: `ssh:manage` is superseded by `ssh:read` / `ssh:write`.** This document is the vocabulary of record, so the change is stated here rather than left to differ between the plan and the code.

  This plan named one scope covering "keygen, pubkey registration". Those are different privileges: `/ssh/keygen` mints the *platform's* host-access **private** key server-side, while registration accepts a **public** key the caller already holds. Bundling them would put a private-key-minting endpoint behind a scope an agent is meant to carry — the opposite of P2's own reasoning, *the agent already has a key; it needs the platform to accept it, not to mint one.*

  Splitting them dissolves the name. `/ssh/keygen` becomes admin-only and is never scope-granted at all, so nothing is left for `ssh:manage` to *manage* — the remaining privilege is exactly "read and write my own public keys", which `ssh:read` / `ssh:write` already named at the enforcement sites in `routes/ssh.py`. Adopting `ssh:manage` would have renamed working enforcement to match a bundle that no longer exists.

  **`ssh:manage` is therefore retired, not deferred.** It appears in no scope list, no route, and no tool contract.
- **Scope the billing levers.** `setup-intent` and `portal-session` move from `_get_current_user` to `_require_scope(billing:write)`. The manual top-up path P1 adds is `billing:write` from the start — it charges a real card, so it is never reachable by a read-scoped credential.
- **One policy registry — done, and smaller than this plan assumed.** The five artifacts were never five hand-maintained files. `TOOL_SCOPES` is the source. `TOOL_CONTRACTS` **and the annotations** are computed at import time in `contracts.ts` — from `TOOL_SCOPES` and from the `READ_ONLY` / `DESTRUCTIVE` / `OPEN_WORLD` sets — with no script writing that file, so neither can drift. Descriptions are hand-written prose, which cannot be generated from anything; `tests/unit/descriptions.test.ts` gates them on completeness and content instead, which is the stronger check for prose.

  What was genuinely missing was **assertions that those guarantees hold**, and gates on the two artifacts a script writes:

  | Artifact | How it is protected |
  |---|---|
  | `TOOL_SCOPES` | the source of truth |
  | `TOOL_CONTRACTS` | derived at import; pinned by **reference** equality, so a copy fails |
  | annotations | derived at import; pinned by invariant (nothing is both read-only and destructive) |
  | descriptions | completeness + content: trigger, impact, and read-only / mutating / destructive each self-declaring |
  | `tool-surface.json` | whole-document equality against a fresh generation |
  | public OpenAPI | whole-document equality (pre-existing) |
  | endpoint inventory | whole-document equality against a fresh generation |

  Each gate was verified by planting the drift it exists to catch, not by observing that it passed. The `tool-surface.json` gate needed two attempts: the first two probes were planted in the direction `diffSurface` already treated as breaking, and only a probe in the direction drift actually occurs — the code gains a scope and `npm run surface:update` is forgotten — showed the old checks passing while the published surface under-stated a tool's required scopes.

- **Tool count: 39 total, and the decomposition is load-bearing.** `TOOL_SCOPES` holds 39. `mcp/tool-surface.json` publishes **30** — the customer profile — because it excludes **7 operator tools** (`drain_host`, `undrain_host`, `evict_host_workloads`, `get_host_capacity`, `get_scheduler_health`, `list_reconciliation_findings`, `retry_agent_command`) and **2 company-knowledge tools** (`search`, `fetch`).

  Checking only the published snapshot verifies 30 of 39 and silently skips the operator tools — the ones carrying `hosts:evict`. **Moving the total requires restating the eval baseline in this document in the same commit**, because every later phase's eval delta is measured against it and a stale baseline invalidates every future gate comparison. `tests/test_tool_scope_registry_completeness.py` enforces the number and the decomposition.
- **Host visibility is split, not reclassified.** `GET /hosts` returned the whole fleet to anyone holding `hosts:read` — a scope every provider needs so their worker agent can report its own admission status, which made competitors' capacity readable by anyone who registered a rig. Reclassifying `hosts:read` as operator authority was tried and reverted: providers are not admins, and it broke onboarding. `hosts:read` now answers *your* hosts; platform-wide visibility moved behind `hosts:fleet`, described "(operator)" so a non-admin cannot delegate it.
  - **Known boundary, deliberately not chased here:** ownership resolves through the OAuth client's *individual* creator, matching `_require_host_operator`. Once hosts belong to an organisation rather than a person, that anchor is wrong. It belongs with P6 and the enterprise adoption plan, not P0.
- **GT0 endpoint inventory classification.** Every one of the **516** operations across 36 modules tagged `covered` / `gap` / `internal` / `redundant`, with a reason. Zero unclassified.

  **The number was 528 here and 516 in the generated inventory.** Same class as the 39-vs-30 eval baseline: a count written into prose drifts from the artifact that produces it. `scripts/generate_endpoint_inventory.py` is authoritative; this document quotes it and must be restated when it moves.

  The checked-in inventory was also **stale against the code** — regenerating it changed 14 rows, every one an endpoint scoped during P0 (`setup-intent`, `portal-session`, `/ssh/keygen`, `stream-ticket`, `expose`, `auto-launch`, the privacy writes). Nothing was wrong with the generator; regeneration is simply a step someone must remember, which is precisely the drift P0.3's byte-identical gate exists to make impossible. Until that gate covers this file too, the inventory is only as current as the last person who ran the script.

**Frontend** — none. This phase is invisible on purpose.

**Gate P0**
- Every access and billing endpoint named above refuses a token missing its new scope, asserted with a real token against a live server.
- Regenerating the registry produces byte-identical output; a hand edit to a generated file fails the build.
- `test_no_runtime_ddl`-style inventory test: zero unclassified endpoints.
- **Eval baseline — two numbers, not one.** `scripts/mcp_tool_eval.py` grades what `tools/list` *publishes*, which is the **30-tool customer profile**, not the 39-entry registry. Labelling the baseline "39 tools" conflates the registry total with the graded surface, and a later phase comparing a 39-labelled baseline against a 30-tool run would read a description regression as a count change. Record both: registry total 39, eval surface 30, plus first-tool accuracy and unsafe-write selection as the baseline pair.
- **Blocked, not skipped:** the eval reads tool definitions from a live server by design (*"a reviewer's model sees exactly what `tools/list` publishes"*), so it needs `XCELSIOR_MCP_TOKEN` and `ANTHROPIC_API_KEY`. Without them it reports `BLOCKED(env)` and exits 0 — a gate that cannot run is never green, and this one has not run.

---

## P1 — The money levers: make "never leave" true for spending
*The claim's weakest clause, and mostly cheaper than it first looked.*

**Backend — the part that needs no browser at all**

- **`top_up_wallet(amount, payment_method_id?)`** — an off-session charge
  against a card already on file, reusing the code path auto-top-up already
  runs. Approval-gated because it moves money, idempotency-keyed so a retry
  cannot double-charge, and bounded by the spend envelope below.
- **`list_payment_methods`** — brand, last four, expiry, default. Enough to say
  *"use the Visa"*; nothing that is a secret.
- **`configure_auto_topup`** — threshold, amount, period cap. Raising a cap
  requires approval; lowering one does not.
- **Spend envelope (capability 1)** — depletion projection against projected job
  completion, with a pause action. The thing that makes unattended spend safe
  rather than merely possible.

**Backend — the part that genuinely needs a browser**

- **`pay.xcelsior.ca` hosted page.** Resolves the intent server-side from an
  authenticated first-party session and renders Stripe's **embedded** Payment Element. The
  `client_secret` never leaves the server. Used for **adding a new card** and
  for **SCA recovery** — not for ordinary top-ups.
- **Fix the SCA gap (§0.3) first.** Catch `authentication_required`, persist the
  pending intent, and surface a resume action. This is a live bug on the
  dashboard path today; the agent surface just makes it visible.
- **URL Mode Elicitation (SEP-1036)** for those two flows only. Capability-
  negotiated, never assumed. Client support is documented `not yet` for
  Microsoft and **UNVERIFIED elsewhere**, so the fallback is the primary path
  until proven otherwise: `completed: false`, `operation_executed: false`, and
  prose that opens with *"Not completed."*
- **Success comes only from the processor.** Signed webhook, verified state.
  `accept` means the user consented to navigate, never that the flow succeeded.

**Frontend**

- The hosted page: fast, obviously **embedded** first-party, works on a phone, and states
  plainly what is being authorised and for how much.
- A **"return to your terminal"** completion state. The browser is a detour, not
  a destination.
- An SCA-pending state in the wallet UI, so a challenge that was never completed
  is visible rather than silent.
- Wallet, envelope, and auto-top-up state legible to a human — the same limits
  the agent respects.

**Gate P1**
- A top-up on a saved card completes **with no browser and no elicitation**,
  asserted with a real token against a live server. This is the phase's headline
  behaviour and the first point at which the claim is true for spending.
- Replaying any funding call with the same idempotency key produces exactly one
  charge. Asserted for manual top-up, auto-top-up, and the crypto rails.
- An `authentication_required` decline produces a **resumable pending state**, a
  visible UI state, and a tool result that says the charge did not happen —
  never a generic error. Asserted by forcing the decline with a Stripe test
  card, not by mocking it.
- Three client conditions for the two browser flows: URL-capable, form-only, and
  no-elicitation. All three either complete or say plainly that they did not.
- **No secret in any surface:** card data, `client_secret`, and processor tokens
  appear in no tool result, log, trace, audit row, or error string. Canary-tested
  with fake PANs.
- Raising a spend cap requires approval; lowering one does not. Both asserted.
- An envelope-funded charge is traceable to its approving plan in one query.

## P2 — Access: launch → connected, without a browser
*The first half of the sentence.*

**Backend**

- **`open_instance_access`** — returns a short-lived, first-party connection for a running instance: browser terminal ticket, or the SSH endpoint plus the fingerprint to verify.
- **`register_ssh_key`** — registers the caller's *public* key. The private key never exists server-side and never enters model context. This is what keeps `routes/ssh.py` on the exclusion list while still making the workflow possible: the agent already has a key; it needs the platform to accept it, not to mint one.
- **`watch_instance`** already exists and stays the waiting primitive.

**Frontend**

- Connection details in the instance view that match exactly what the tool returns — same host, same fingerprint, same expiry. A human and an agent looking at the same instance must see the same truth.

**Gate P2**
- A scripted journey — launch, wait, connect, run a command, terminate — completes using **only tool calls**, against a live staging tenant. A journey that needs a raw HTTP call or a dashboard click fails the gate.
- Connection material is short-lived and single-use; a replayed ticket is refused.
- No private key material appears in any tool result. Asserted, not assumed.

---

## P3 — Durable state: artifact → volume promotion
*Capability 3. The substrate under P4 and P5.*

**Backend**

- **`promote_artifact_to_volume`** — a finished run's weights and checkpoints move from artifact storage (90-day `retain_until`, presigned TTL) onto a volume, which has no clock.
- Volume create / attach / detach / snapshot as tools, with detach behind approval since it can disrupt a running workload.

**Frontend**

- On a completed instance: *"this output expires in N days"* with the promote action beside it. The retention clock should be visible to the human too — it is currently invisible, which is how work gets lost.

**Gate P3**
- Promotion is idempotent under retry; a repeated call produces one volume, not two.
- The retention clock is asserted: an artifact past `retain_until` is gone, a promoted volume is not.
- Round-trip: train → promote → mount in a *new* instance → read the weights. Tool calls only.

---

## P4 — The pipeline: one approval for a dependency graph
*Capability 2. The idea that turns the approval gate into leverage.*

**Backend**

- A **dependency primitive**: a plan that carries a graph of stages, approved once, executed in order, with failure semantics that are explicit (halt / continue / retry-stage).
- `train → evaluate → serve` as the reference journey, using P3's promotion between stages.

**Frontend**

- A pipeline view showing the graph, which stage is live, and one approval covering all of it — with the total committed spend stated *before* approval, not after.

**Gate P4**
- One approval, three stages, one audit chain. Asserted end to end.
- A mid-pipeline failure does not silently continue; the declared failure semantics are what happens.
- The approved graph is server-bound: editing any stage after approval invalidates it. Asserted by attempting exactly that.
- Spend is bounded by what was approved. A pipeline cannot exceed its own quote.

---

## P5 — Spot migration and placement preference
*Capabilities 4 and 5. The marketplace's own advantages.*

**Backend**

- **Checkpoint-aware migration:** snapshot → stop → relaunch cheaper → verify placement, using P3 for the checkpoint.
- **Reputation- and SLA-aware placement:** *"prefer verified hosts above 99.5% uptime even at 15% more"* over `simulate_instance_placement`, reputation, and SLA targets.

**Frontend**

- Placement preference as a first-class launch control, showing the price/reliability trade-off it implies before launch.
- Migration history on the instance timeline: what moved, when, why, and what it saved.

**Gate P5**
- A migrated job resumes from its checkpoint, proven by comparing state before and after — not by the absence of an error.
- A placement preference that cannot be satisfied **refuses clearly** rather than silently falling back to the cheapest host. This is the failure mode that would quietly destroy trust.
- Preference is honoured in the audit trail: the chosen host's reputation and SLA at time of placement are recorded.

---

## P6 — The provider surface
*Per [mcp-provider-axis-plan.md](./mcp-provider-axis-plan.md). Capability 6.*

**Backend**

- Separate deployment profile, separate listing. Supply-side journeys: register a host, admission evidence, publish capacity, set a spot floor, read earnings, request payout.
- **Payout onboarding via URL elicitation**, reusing P1's machinery. KYC never enters model context; completion comes from `account.updated`, never from the browser return.
- **Provider yield optimizer** — admission, reputation, SLA and spot preview into a recommended floor.

**Frontend**

- Provider dashboard parity: the same earnings and payout state the tools return.
- Onboarding that survives interruption — a provider who closes the tab mid-KYC can resume.

**Gate P6**
- A provider journey — register → admit → publish → earn → payout — completes through tools plus the browser handoffs, on a live staging tenant.
- A payout is bound to job, amount, currency, destination state, and idempotency key. Replay produces one payout.
- Returning from `return_url` proves nothing: asserted by returning without completing and checking the state is still `pending_requirements`.

---

## P7 — Environment snapshot and sweep
*Capability 7.*

**Backend** — commit a configured instance to a reusable image; launch N identical nodes from it.
**Frontend** — image library with provenance: what it was built from, when, by which run.

**Gate P7** — a sweep of N nodes from one snapshot is byte-identical in environment; a snapshot records its lineage.

---

## 2. Keeping drift off

Drift is what this codebase has actually suffered from — a spec regenerating
from its own output, five files that must agree by hand, a guard narrowed until
it passed. Four mechanisms, each already proven here:

| Mechanism | Proven by | Applied to |
|---|---|---|
| **Generate, never hand-maintain** | the OpenAPI generator read its own output for months; five schemas drifted | P0's registry: contracts, scopes, annotations, descriptions, manifest |
| **Whole-document equality** | comparing only the operation set is how five stale schemas passed | tool surface snapshot, published spec |
| **Ratchets that reach zero** | 418 banned-vocabulary hits → 0; 30 float money columns → 0 | scope coverage, unclassified endpoints, eval regressions |
| **Guards that prove their own reach** | the vocabulary guard plants a probe file of each type it claims to read | any future vocabulary or policy guard |

And one rule with no mechanism, only discipline: **when a check fails, fix the
thing — not the check.** Every serious defect in this codebase's recent history
was reachable because a check had been softened rather than satisfied.

---

## 3. Sequence, and what it depends on

```
P0  scopes + registry + inventory     ← everything; nothing is safe before it
 │
P1  money levers + SCA recovery       ← top-up needs no browser; only adding a
 │   (spend envelope rides along)         card and recovering a decline do
 │
P2  access: launch → connected        ← makes "never leave" true for the terminal
 │
P3  artifact → volume promotion       ← the substrate
 ├──────────────┬───────────────┐
 │              │               │
P4 pipeline   P5 migration +   P7 snapshot + sweep
 │              placement
 │
P6  provider surface (own listing, reuses P1's URL machinery)
```

P4 and P5 both need P3. P6 needs P1. Everything needs P0.

---

## 4. What this plan does not promise

Stated here so no one has to discover it late:

- **URL elicitation client support is unverified.** Microsoft documents *"not yet"*; every other directory client is undocumented. If no client honours it, P1's fallback becomes the permanent path — usable, but one extra step. The plan is built so that outcome is a degradation, not a failure.
- **No public MCP server is known to have completed Stripe Connect onboarding this way.** P6 would be early. That is a reason for the conspicuous fallback, not a reason to wait.
- **The tool-count threshold is unmeasured.** No published evidence isolates one. P0 captures the baseline and every phase re-measures; the number is ours to find.
- **CI is billing-locked.** Every gate in this plan runs locally until that changes. A green push is unverified, and this plan does not pretend otherwise.
