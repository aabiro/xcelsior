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
an agent could touch them. That was too broad twice over, and the distinction
matters because it changes what ships first. The first correction: charging a
saved card needs no browser. The second: *adding* a card is not the agent
surface's job at all — the user does that in the dashboard, and the agent only
ever references what is already on file.

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
| **Add a new card** | **not our problem** | the user adds cards in the dashboard; the agent only ever references what is already on file |
| **Recover from an SCA challenge** | **yes** | the issuer demands the cardholder, in person |

**Adding a card is out of scope for the agent surface entirely.** Not "needs a
browser we will provide" — simply not a thing the MCP server does. Cards are
added where card data belongs: a first-party page the user is already looking
at. The agent reads the resulting list and names one. That removes PAN handling,
the hosted-page build, and an entire class of consent question, and it costs
nothing: a user who has never added a card cannot be topped up by an agent
either way.

So exactly **one** action needs a page, and it is not a convenience — it is a
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
This is a live defect independent of the agent surface, and it is the only
reason a browser appears in P1 at all — not so an agent can take a card number,
but so a *declined* charge has somewhere to send the human.

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

- **Scope the access endpoints.** New scopes: `instances:connect` (auto-launch, expose, stream-ticket), and `ssh:read` / `ssh:write` for key listing and registration. Applied with `_require_scope`, at both layers.

  *This said `ssh:manage`, one scope for both. The implementation split it, and the split is right: read/write separation lets an agent list the keys on an account without being able to add one, and adding a key is the step that grants shell access. `ssh:manage` was never written — it existed only in this sentence — so nothing had to change but the sentence.*
- **Scope the billing levers.** `setup-intent` and `portal-session` move from `_get_current_user` to `_require_scope(billing:write)`. The manual top-up path P1 adds is `billing:write` from the start — it charges a real card, so it is never reachable by a read-scoped credential.
- **One policy registry.** `contracts.ts`, `scopes.ts`, annotations, descriptions and `tool-surface.json` are five hand-maintained files that must agree. Invert it: one registry is the source, the rest are generated. The 37-vs-39 drift ChatGPT spotted becomes unrepresentable rather than merely detected.
- **GT0 endpoint inventory classification. DONE.** All **519** operations tagged
  `covered` / `gap` / `internal` / `redundant`, with a reason. Zero unclassified,
  and `MAX_UNCLASSIFIED` in `tests/test_gt0_classification_ratchet.py` is now `0`
  — a floor rather than a budget, so the next endpoint added must be classified
  in the commit that adds it.

  *The count is 519, not the 528 this line first claimed; the generator reports
  what it finds and 528 was never re-derived after routes moved.* Final tally:
  **287 internal, 153 gap, 57 covered, 22 redundant.**

**Frontend** — none. This phase is invisible on purpose.

**Gate P0**
- Every access and billing endpoint named above refuses a token missing its new scope, asserted with a real token against a live server.
- Regenerating the registry produces byte-identical output; a hand edit to a generated file fails the build.
- `test_no_runtime_ddl`-style inventory test: zero unclassified endpoints.
- Eval baseline **captured** at **45 tools total / 36 published**. *The surface
  has grown three times since, and the figures below describe it as it stood
  when they were measured. Later captures are recorded in `eval-baseline.json`
  and in the §1.5 row of `docs/gate-truth-table.md`; each states its own tool
  count, because a rate that moves after adding tools says nothing unless the
  count moves with it. Do not compare two numbers here without checking both
  counts and the `base` field — a local capture and a deployed one are not the
  same measurement.*

  Measured then:
  `expected_tool_accuracy` **0.8778** (79 of 90 trials, 3 samples x 30 cases),
  abstention 1.0, unsafe-write rate 0. Recorded in `eval-baseline.json`. This is
  the number every later phase is compared against, and it is **below the 0.90
  threshold**, which has not been moved.

  *Two things had to be fixed before the number meant anything. The metric was
  named `first_tool_accuracy` and never measured the first tool — `grade()`
  checks whether the expected tool appears anywhere in one turn's selection — so
  it is now `expected_tool_accuracy`. And a single sample was not a measurement:
  two consecutive runs against an unchanged surface scored 26/30 and 25/30,
  disagreeing on three cases.*

  *One correction came from the eval contradicting the surface it grades.
  `should_i_run_this`'s description tells the model to call it "whenever you are
  about to launch", and `approval-training-repo` marked it down for complying —
  while `approval-launch` and `approval-serverless` already accepted their
  guardrails. Ruled: the guardrail is a legitimate answer. Fixing that, and the
  same inconsistency a new guard found in `approval-budget-launch`, moved
  approval from 6/15 to 9/15 and the overall rate from 0.8556 to 0.8778.*

  *Still failing every sample: `approval-serverless` (called neither the
  guardrail nor the action) and `approval-terminate` (read the instance instead
  of destroying it). Both `cancel` and `terminate` carry a `confirm:false`
  preview, so the caution the model is reaching for already exists inside the
  tool — the descriptions say so and it is not choosing them. `followup` is the
  noisy category now: three of its four cases pass some samples and not others.*
  *Restated for P2, which added both of its tools: `register_ssh_key` (the key
  the platform must accept) and `open_instance_access` (the way in once it
  does). Both are inside the default profile, so both counts move by two.*

  *Restated from "39 tools", and the correction is two separate things. The
  count moved — `TOOL_SCOPES` holds 43 and `tool-surface.json` publishes 34,
  the difference being tools outside the default profile — and
  `tests/test_tool_scope_registry_completeness.py` refuses to let that drift
  silently, which is why this line changes in the same commit as the constant.*

  *The sentence also said "captured", and nothing has been captured.*
  `scripts/mcp_tool_eval.py` grades the surface a **live server** publishes,
  using a model, and needs `XCELSIOR_MCP_TOKEN`, `ANTHROPIC_API_KEY` and
  `XCELSIOR_STAGING_URL` — none of which is configured on this repository. The
  script reports `BLOCKED(env)` and **exits 0** without a credential, so the
  `eval-baseline` job in `live-gates.yml` would install its dependencies, grade
  nothing, and report green. No baseline exists at 39, 43, or any other number.*

---

## P1 — The money levers: make "never leave" true for spending
*The claim's weakest clause, and mostly cheaper than it first looked.*

**Backend — the part that needs no browser at all**

- **`top_up_wallet(amount, card?)`** — an off-session charge against a card
  already on file, reusing the code path auto-top-up already runs.
  Idempotency-keyed so a retry cannot double-charge.

  *This said "approval-gated because it moves money". It is not, and the
  reason is that the consent was already given twice before the tool is
  reachable:*

  1. *The user added the card in the dashboard — a first-party page, which is
     the mandate and the only place card data is ever handled.*
  2. *The user granted `billing:write` deliberately. Quick Connect does not
     carry it, so the token this product tells people to paste **cannot top up
     at all**. Obtaining the capability is an explicit act with a consent
     screen that says what it means.*
  3. *Stripe enforces what remains — Radar, issuer limits, insufficient funds,
     SCA. Those are enforcement we cannot tune better than the processor can,
     and duplicating them means two ceilings that disagree.*

  *A per-transaction approval on top of that re-decides something the user
  decided twice, and makes "never leave the terminal" false for the most common
  action in the phase. The failure it would prevent — an agent funding the
  user's own wallet from the user's own card — moves money **into** the
  account, not out of it.*

  *`mcp_client_policies.per_action_max_micros` is nullable, and NULL already
  means "no ceiling". A deployment that wants one sets it; nothing is built
  here to pre-empt that.*
- **`list_payment_methods`** — brand, last four, expiry, default. Enough to say
  *"use the Visa"*; nothing that is a secret.
- **`configure_auto_topup`** — threshold, amount, period cap. Raising a cap
  requires approval; lowering one does not.
- **Spend envelope (capability 1)** — depletion projection against projected job
  completion, with a pause action. The thing that makes unattended spend safe
  rather than merely possible.

**Backend — the one flow that genuinely needs a browser**

- **Fix the SCA gap (§0.3) first.** Catch `authentication_required`, persist the
  pending intent, and surface a resume action. This is a live bug on the
  dashboard path today; the agent surface just makes it visible.
- **A resume URL, returned as text.** When a saved-card charge is declined with
  `authentication_required`, the tool result says the charge did **not** happen
  and carries a link to the existing dashboard page that completes the
  challenge. No new hosted page: the dashboard already renders the Payment
  Element for a pending intent, and the `client_secret` stays server-side there
  as it does today.
- **No URL Mode Elicitation.** It was specified for two flows; one of them —
  adding a card — is gone, and for the other the difference between elicitation
  and a link is whether the client opens the browser or the user clicks. Client
  support is documented `not yet` for Microsoft and unverified everywhere else,
  so the fallback would have been the real path regardless. Specifying a
  capability that probably will not fire, and gating a phase on three client
  behaviours to support it, buys one click.
- **Success comes only from the processor.** Signed webhook, verified state. A
  user reaching the page means they consented to navigate, never that the flow
  succeeded.

  With elicitation and card-adding both gone, this is now the **only** completion
  signal in the phase — which makes `POST /api/providers/webhook` answering `200`
  to events whose signature it failed to verify a phase-blocking defect rather
  than a tidy-up. Stripe reads `200` as delivered: no retry, and the event never
  appears in `GET /v1/events?delivery_success=false`, which is the documented
  recovery path. It must return `400`.

**Frontend**

- Card management stays where it already is. Adding, removing and defaulting a
  card is a dashboard job; nothing new is built for the agent.
- The SCA resume view: states plainly what is being authorised and for how much,
  works on a phone, and ends with a **"return to your terminal"** state. The
  browser is a detour, not a destination.
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
- **The webhook refuses what it cannot verify.** An event with an unverifiable
  signature returns `400`, so Stripe retries it and it remains visible in
  `delivery_success=false`. Asserted by posting a body signed with the wrong
  secret. This is the gate's second headline, because it is the only completion
  signal the phase has left.
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
- **Payout onboarding via a returned link**, reusing P1's shape rather than its elicitation — which P1 no longer has. KYC never enters model context; completion comes from `account.updated`, never from the browser return. Since the return is already worthless as a signal, elicitation bought nothing here either: the provider clicks a link, and the webhook decides.
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
P1  money levers + SCA recovery       ← top-up needs no browser; only an SCA
 │   (spend envelope rides along)         decline does, and it is a link
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

- **URL elicitation is no longer used, and that was the right call rather than a concession.** Microsoft documented *"not yet"* and every other directory client was undocumented, so the fallback would have been the real path anyway. Since the only remaining browser detour is an SCA decline — and since a returned link reaches the same page the dashboard already serves — the capability bought one click in exchange for capability negotiation and three client conditions in Gate P1. If client support becomes real and common, adding it later is additive and changes no contract.
- **No public MCP server is known to have completed Stripe Connect onboarding this way.** P6 would be early. That is a reason for the conspicuous fallback, not a reason to wait.
- **The tool-count threshold is unmeasured.** No published evidence isolates one. P0 captures the baseline and every phase re-measures; the number is ours to find.
- **CI is billing-locked.** Every gate in this plan runs locally until that changes. A green push is unverified, and this plan does not pretend otherwise.
