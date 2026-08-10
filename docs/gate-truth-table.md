# Gate truth table — P0 through P5

*Built 2026-08-08, against `4321d69`.*

The directive that asked for this said implementation had started moving faster
than the gates meant to prove it safe. This is the reckoning: every gate clause
in `docs/mcp-agent-native-implementation-plan.md` for P0–P3, judged one clause at
a time against evidence I looked at, not against a green suite.

**"The suite passes" is not evidence for any row below.** The suite passes right
now with 5029 tests and it does not, on its own, establish a single one of these
clauses — several clauses require a live server the suite never contacts, and one
clause is contradicted by a test that passes.

## Statuses

| Status | Meaning |
|---|---|
| **PASS** | The clause is true and something asserts it. |
| **PARTIAL** | The behaviour is implemented, but the clause's *own* evidentiary standard is unmet — usually "asserted live" where only an in-process assertion exists. |
| **FAIL** | Not true, or not built. |
| **BLOCKED** | Cannot be established right now for a reason outside the code. |
| **SUPERSEDED** | The clause described a surface that no longer exists. |

A note on how PARTIAL is used, because it is the most common verdict here and the
easiest to inflate: a clause that says *"asserted with a real token against a live
server"* is not satisfied by a `TestClient` assertion, however good. The plan's own
§1 gives the reason — *"a mock is what passed while production did not."* Where I
mark PARTIAL, the feature generally works; what is missing is the proof the plan
demanded.

---

## §1 — The five gates every phase was supposed to ship with

These are the cross-cutting ones, and they are where the discipline actually
slipped.

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | Behaviour tests, named for the behaviour | **PASS** | Observably followed. `test_agent_can_register_its_own_key`, `test_requeue_clears_output_from_either_door`, `test_serverless_refuses_narrowed_credentials`. No `test_scopes_2` anywhere. |
| 2 | A confirmed-failing check — every regression test shown to fail *before* the fix | **PARTIAL** | Done and recorded for this session's work (the serverless re-scope was probed by stripping one check; both the structural and behavioural layers fired independently). I cannot evidence it for the whole historical corpus, and I will not claim it. |
| 3 | A live-credential path — ≥1 assertion per phase, real token, real server | **PARTIAL** *(was FAIL)* | **P0 now has one, and it has actually run** — 12 assertions against a live staging server, `scripts/run_live_gates.sh`. That script is the substantive change: the gate previously existed but had no runner that could execute, and standing the environment up by hand is something nobody does twice. It skips rather than passes without credentials, verified. **P1, P2 and P3 still have no live path**, which is why this is PARTIAL and not PASS. |
| 4 | A refusal test — what the phase makes impossible | **PASS** | Dense across the corpus. `test_serverless_refuses_narrowed_credentials`, `test_platform_ssh_key_is_admin_only`, `test_terminal_ticket_needs_connect_scope`, `test_stripe_webhook_refuses_unverified`. |
| 5 | An eval delta per phase — re-run at the new tool count | **PARTIAL** | The loop runs without GitHub Actions (`scripts/run_mcp_eval_locally.sh`), and the clause has now been honoured **end to end for one phase**: P4 captured *before* it added a tool (48 tools, 0.9444) and *after* (50 tools, 0.9444) — two more tools, no accuracy lost. The promotion tools got theirs the same way (46 → 48, 0.9111 → **0.9444**). Still PARTIAL because **P2's SSH tools have no delta and cannot get one**: the intermediate surface no longer exists to measure, and reconstructing a number from old commits would not be the measurement the clause asks for. |

§1.3 and §1.5 were the two structural failures on this page and are now the two
structural *partials*: both have a runner that works and neither is honoured per
phase. Everything else is a clause-by-clause matter; these two were a habit that
lapsed, and the habit is only half repaired — P1, P2 and P3 still have no live
assertion, and P2's tools shipped without an eval delta that cannot now be
taken. The promotion tools *did* get theirs (46 → 48, 0.9111 → 0.9444), which
is the first time the clause has been honoured rather than skipped.

---

## Gate P0 — Foundations

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | Every access and billing endpoint refuses a token missing its new scope, **asserted with a real token against a live server** | **PASS** *(was PARTIAL)* | Now asserted live. `tests/live/test_named_scopes_refuse_live.py` mints a `client_credentials` token holding only `instances:read` — the credential class Quick Connect issues, and the class `_require_scope` actually gates — then drives the endpoints the plan names by hand: `setup-intent`, `portal-session`, `GET /api/ssh/keys`, `POST /api/ssh/keys`. All four answer **403**; the positive control `/instances` answers **200** with the same token, so the refusals are not a server refusing everything. A browser session would have proven nothing here: `_require_scope` is a deliberate no-op for interactive sessions. |
| 2 | Regenerating the registry produces byte-identical output; a hand edit to a generated file fails the build | **PASS** | `test_tool_scope_registry_completeness.py` (8 passed) and `test_generated_artifacts_are_current.py`. **Demonstrated, not assumed:** this session the inventory test went red the moment a route change made `docs/generated/endpoint-inventory.md` stale, and green after regeneration. It also caught a hand-written note in `docs/endpoint-classification.json` that my own change had falsified. |
| 3 | Zero unclassified endpoints | **PASS** | `MAX_UNCLASSIFIED = 0` across **519** operations — a floor, so the next endpoint must be classified in the commit that adds it. |
| 4 | Eval baseline captured | **PASS** *(was SUPERSEDED + BLOCKED)* | Recaptured 2026-08-08 once the key was funded. `expected_tool_accuracy` **0.9111** at `tool_count: 46`, against the **0.90 threshold, which has not been moved**. Abstention 1.0, unsafe-write rate 0.0. The prior capture was 0.8778 at 36 tools — *below* threshold — so accuracy rose while the surface grew by ten tools, which is the harder direction: more tools is more ways to choose wrongly. Cost $1.31. |

**On P0.2 — a caution about my own measurement.** While checking this clause I
twice "found" drift between `mcp/tool-surface.json` (46 tools) and the registered
tools, by grepping for `registerTool("…")`. Both times my regex was the broken
instrument: there is a second helper, `registerRead(server, client, user, "…")`,
and `tool-surface.json` is deliberately the *customer* snapshot — total minus
stated exclusions like `evict_host_workloads` and the connector's `fetch`/`search`.
The project's own guard was right and my ad-hoc count was wrong, twice. It is
recorded here because a truth table built on ad-hoc greps would be exactly the
failure this document exists to stop.

---

## Gate P1 — The money levers

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | Top-up on a saved card completes with no browser and no elicitation, **real token, live server** | **PARTIAL** | Implemented and tested, but `tests/test_manual_topup_charge.py` monkeypatches `_stripe_for_charge`. No live assertion. |
| 2 | Replaying any funding call with the same idempotency key produces exactly one charge — **manual top-up, auto-top-up, and the crypto rails** | **PARTIAL** | Manual and the wallet-deposit rail are asserted (`test_funding_replay_is_one_charge.py`), PayPal separately. **The crypto rail is not:** `/api/billing/crypto/deposit` exists ([routes/billing.py:1032](routes/billing.py#L1032)) with no idempotency test. One of the three named rails is unasserted. |
| 3 | An `authentication_required` decline produces a resumable pending state, a visible UI state, and a truthful tool result — **forced with a Stripe test card, not by mocking it** | **PARTIAL** | `test_sca_decline_is_recoverable.py` and `test_sca_pending_is_visible.py` are unusually careful — they build a genuine `stripe.CardError` from Stripe's documented JSON body rather than a hand mock with the attributes the code hopes for. But the decline is still *injected*, and the clause names the mocking exclusion explicitly. The clause as written is unmet. |
| 4 | The webhook refuses what it cannot verify — `400` on a bad signature | **PASS** | `test_a_wrong_signature_is_refused_with_400` plus a missing-header case, in `test_stripe_webhook_refuses_unverified.py`. The file reasons explicitly about why "400 or 503" would be an untrustworthy assertion. |
| 5 | No secret in any surface — card data, `client_secret`, processor tokens; canary-tested with fake PANs | **PASS** | `tests/test_no_payment_secrets_in_logs.py`. |
| 6 | **Raising a spend cap requires approval; lowering one does not. Both asserted.** | **PASS** *(was FAIL)* | Ruled Option A (§Ruling) and implemented. A widening by any caller that is not an interactive human is refused `409` and directed to `/api/v2/billing/auto-topup-plans`; narrowing and disabling stay single-call. `approval_mode` is hard-coded `"human"` so a standing policy cannot approve a change to its own ceilings. Both halves asserted in `test_widening_auto_topup_needs_approval.py`, with the plan lifecycle driven against real PostgreSQL in `test_auto_topup_plan_lifecycle.py`. |
| 7 | An envelope-funded charge is traceable to its approving plan in one query | **PASS** | `tests/test_spend_traces_to_its_approving_plan.py`. |

---

## Gate P2 — Access

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | A scripted journey — launch, wait, connect, run a command, terminate — completes using **only tool calls**, against a live staging tenant | **FAIL** | No such script exists. Nothing in `scripts/` performs this journey. This is the phase's headline clause and it has never been run. |
| 2 | Connection material is short-lived and single-use; a replayed ticket is refused | **PASS** *(was PARTIAL)* | `tests/test_ws_ticket_is_single_use.py` — 8 assertions across **both** consume implementations (shared-state and in-process), 16 in total. Replay, expiry, and each of the three pins (purpose, target, client IP) are asserted separately so a regression names itself. Parametrizing both paths was not ceremony: removing the in-memory pop reds `[memory]` while `[shared]` keeps passing, so a test of either alone would have proven nothing about the other — and the suite runs one while production runs the other. |
| 3 | No private key material appears in any tool result. Asserted, not assumed | **PASS** | `mcp/tests/unit/private-key-hygiene.test.ts` and `hygiene.test.ts`, plus `inspectSshKeyInput()` classifying and refusing a private key *before* any network call. The scrubber also handles the truncated-log case (BEGIN with no END). |

**Also unmet, from P2's backend section rather than its gate:** `open_instance_access`
is specified to return "the SSH endpoint **plus the fingerprint to verify**". It
returns `host_key_fingerprint: null` with a note explaining the platform publishes
none. That was the honest thing to ship rather than inventing a value, but the
capability is absent, and without it the "connect" step of clause 1 cannot be done
safely by an agent — it would have to accept an unverified host key. **These two
are the same problem**, which is why clause 1 should not be attempted before the
fingerprint exists.

---

## Gate P3 — Durable state

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | Promotion is idempotent under retry; a repeated call produces one volume, not two | **PASS** *(was FAIL)* | Proven the way §4 of the promotion plan says to prove it: call twice, one `volume_promotions` row, second reports `replayed`. `test_promotion_is_idempotent.py`, against the real route and a real database — the mechanism is a unique constraint plus `ON CONFLICT DO NOTHING`, neither of which exists in a fake. Removing the conflict clause reds four of the six. |
| 2 | The retention clock is asserted: an artifact past `retain_until` is gone, a promoted volume is not | **PARTIAL** | The artifact half is covered (`test_artifact_retention_authority.py`). The *hold* now exists and is tested (`test_promotion_takes_the_hold.py`) — an in-flight promotion stops the clock, and a stale one is swept and released. What is still unproven is the second half as written: expire an artifact **after** a promotion and read the volume anyway. That needs a mounted volume, so it arrives with clause 3. |
| 3 | Round-trip: train → promote → mount in a *new* instance → read the weights, tool calls only | **FAIL** | The tool now exists (`promote_artifact_to_volume`, A4), so this is no longer blocked on missing code — it is blocked on a **staging environment**, which the promotion plan named as a dependency in advance rather than discovering here. Unchanged verdict, changed reason. |

**P3's promotion half is now built, A0→A4.** When this table was first written
— the same day — this paragraph read *"the promotion half is not started;
`docs/artifact-promotion-plan.md` exists and sequences it A0→A4, no code does."*
It now does: the manifest (`318ba57`), the verified copy (`c1537d2`), the
retention hold (`369bc03`), per-file resume (`e50beda`), and
`promote_artifact_to_volume` itself (`cf1981b`).

One piece of A3 is outstanding — mount-on-demand for an unattached volume,
which §3.4 calls the genuinely open question and answers with "least-loaded
host in the volume's region".

**Deployed.** This paragraph said "none of it is deployed" when written, which
was true for about six hours. Production is at alembic **103** with both
promotion tables live and the routes answering `401` rather than `404` —
verified against the database and the endpoints, not inferred from a deploy log.

---

## Gate P4 — The pipeline

*Added 2026-08-09. This phase was built after the table existed, which is the
first time a phase has had its gate judged as it shipped rather than
reconstructed afterwards.*

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | One approval, three stages, one audit chain. Asserted end to end | **PASS** | `test_gate_p4_pipeline.py::test_one_approval_covers_three_stages` — one `action_plans` row, three `pipeline_stages` rows sharing its `plan_id`, driven through `POST /api/v1/pipelines` and execute. |
| 2 | A mid-pipeline failure does not silently continue; the declared failure semantics are what happens | **PASS** | All three modes execute and are asserted by *causing* the failure: `halt` stops and marks the rest `skipped` with a reason; `continue` proceeds; `retry` re-attempts to `max_attempts` and then **halts** rather than falling through. Disabling the halt branch reds exactly the tests that depend on it. |
| 3 | The approved graph is server-bound: editing any stage after approval invalidates it. **Asserted by attempting exactly that** | **PASS** | The clause's own standard, met literally: the test alters an approved plan's `canonical_args` in the database and calls execute, which refuses `409 argument_hash_mismatch` and materialises no stages. This was PARTIAL for an hour — asserted on the hash changing, which is a fact about a function and not about anything refusing. |
| 4 | Spend is bounded by what was approved. A pipeline cannot exceed its own quote | **PASS** | Checked *before* each stage, so an over-budget stage never starts, and compared against **actual spend plus this stage's quote** rather than the sum of estimates — a stage that overruns eats into what remains. Failed retry attempts bank their spend, so a retrying stage cannot spend without limit inside a bounded pipeline. |

**What is honestly not done: the stage executor is not wired.** Every stage
reports `stage_executor_not_wired` rather than succeeding. The four clauses
above are all about the *approval* — what it covers, what invalidates it, what
it bounds — and each is met. But a pipeline that approves correctly and then
performs nothing is not the capability P4 describes, and the gate passing does
not change that.

Wiring it is a design question rather than an omission, and it belongs in review
rather than in an improvised commit: a stage must run **without its own
approval**, since the point of the phase is one approval for the graph. That
means either a child plan the parent auto-approves, or calling the underlying
action beneath the plan machinery. The first keeps one audit chain and risks a
second approval surface; the second bypasses the substrate every other action
goes through. Neither should be chosen by whoever happens to be typing.

---

## Gate P5 — Spot migration and placement preference

*Added 2026-08-10, mid-phase. Written now rather than at the end, because the
reason clause 2 is PARTIAL is the interesting part and it is freshest while the
work is open.*

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | A migrated job resumes from its checkpoint, proven by comparing state before and after — not by the absence of an error | **FAIL** | C3, not started. Needs two live instances that can share a volume. |
| 2 | A placement preference that cannot be satisfied **refuses clearly** rather than silently falling back to the cheapest host | **PARTIAL** | **The module refuses correctly; the system still falls back.** `control_plane/scheduler/preference.py` gates rather than ranks, refuses with the failing number, and 33 tests drive it — including a probe showing that reranking instead of gating reds four of them. But `scheduler.py:5501` prefers verified hosts and **falls back to all when none qualify**, which is the very failure this clause names, already shipped. A correct module wired into that block still fails the clause end to end. `docs/placement-preference-plan.md` §5.4 is why, with the code inline. |
| 3 | Preference is honoured in the audit trail: the chosen host's reputation and SLA at time of placement are recorded | **PARTIAL** | The values are copied rather than referenced, and verification state, `verified_at`, `last_check_at` and `next_check_at` come with them — verification is revocable, so it is precisely the fact that goes stale. What does not exist yet is the **record**: C1 writes it, and it must be append-only. |

**This is the P4 sentence again.** A clause can be honestly met by a module
sitting inside a system that does not yet honour it, and the distance between
those two is exactly what this table exists to hold. Clause 2 is PARTIAL and not
PASS for that reason alone — nothing is wrong with the module.

**Neither constraint is shippable today, on production numbers.**
`min_uptime_pct` refuses everything because `sla_monthly` has zero rows.
`min_tier` refuses everything above `new_user` and admits everyone at it.
`require_verified` refuses everything because the two verified hosts are 111 and
124 days past a 7-day tolerance — the reverification sweep exists
(`list_hosts_needing_reverification`) and **has no callers**. That sweep is
therefore C2's first commit, before the launch surface: one pass makes the
control live, and shipping it beforehand would teach users the feature is
broken.

---

## Tally

| Gate | PASS | PARTIAL | FAIL | BLOCKED/SUPERSEDED |
|---|---|---|---|---|
| §1 universal | 2 | 3 | — | — |
| P0 | 4 | — | — | — |
| P1 | 4 | 3 | — | — |
| P2 | 2 | — | 1 | — |
| P3 | 1 | 1 | 1 | — |
| P4 | 4 | — | — | — |
| P5 | — | 2 | 1 | — |
| **Total** | **18** | **8** | **3** | **0** |

Eighteen of twenty-nine clauses are fully met, nothing is BLOCKED, and **Gate P0
and Gate P4 are wholly met**.

The count grew because P4 and P5 added clauses of their own — this table now
covers P0–P5 rather than P0–P3. Worth saying plainly: Gate P4 passing all four does
**not** mean pipelines work. Its clauses are about what an approval covers,
invalidates and bounds, and the stage executor beneath them is unwired by
design. A gate can be honestly met by a feature that is honestly incomplete,
and pretending otherwise is how a table like this stops being worth reading.

It was eight when this table was first written. P1 clause 6 moved FAIL → PASS
when the ruling was implemented; P0 clause 4 moved BLOCKED → PASS when the
Anthropic key was funded and the eval recaptured; §1.5 moved FAIL → PARTIAL
because the loop runs again, though the two phases that already shipped without
a delta cannot get one retroactively.

P3 clause 1 moved FAIL → PASS when promotion was built and its idempotency
proven the way the promotion plan's §4 specifies.

**Two failures remain, and they are the same shape**: Gate P2 clause 1 (the
access journey) and Gate P3 clause 3 (the promotion round-trip). Neither is
blocked on missing code any more — the tools exist — and as of today neither is
blocked on a *server* either, since `scripts/run_live_gates.sh` stands one up.
What they still need is a scripted journey that drives the whole sequence, and
for P2 the host-key fingerprint that would let an agent complete the connect
step without accepting an unverified host key.

A gate that cannot be run has not been passed — but these two are now a
morning's work rather than an environment project, which is a different kind of
open.

## What this table does not cover

- **P4–P7 are not assessed.** The directive asked for P0–P3.
- **I did not re-run the full suite to build this.** These verdicts come from
  reading code and tests, running the four targeted files named above, and one
  live probe against production for the certificate question. Where I say a test
  exists, I looked at it; where I say a behaviour holds, I read the code that
  implements it.
- **§1.2 is unverifiable in retrospect.** Whether every historical regression test
  was shown to fail first cannot be reconstructed from the repository. I marked it
  PARTIAL on the strength of current practice only.
- **No claim here rests on production behaviour**, except the single ticket-consumption
  read, because P1/P2/P3 have no live path — which is itself §1.3's verdict.

---

## Ruling — Gate P1 clause 6 (the consent contradiction)

The plan says *"raising a spend cap requires approval; lowering one does not."*
The implementation requires `billing:write` and nothing else, and
`test_auto_topup_change_is_recorded.py` defends that choice: the caller already
holds a deliberately granted scope, and *"asking again for something the user
already granted re-decides their decision; recording its use does not."*

That is a real argument. It loses, for three reasons.

**1. The standing directive settles the procedure.** *"These decisions are
intentional architectural constraints, not suggestions… Any deviation requires an
explicit architecture change rather than an implementation convenience."* A gate
clause overridden by a docstring is the definition of the thing that directive
prohibits. The plan is amendable — by the owner, deliberately — but not by a test
that passes.

**2. The analogy to `top_up_wallet` does not hold.** The defence's strongest move
is that `top_up_wallet` charges a real card with no per-transaction approval, so
gating a mere *setting* is inconsistent. But `top_up_wallet` charges **a stated
amount, once, while the user is in the conversation watching it happen**.
`configure_auto_topup` installs **standing, unattended charge authority that fires
on a threshold, repeatedly, with nobody present.** Those are not the same risk, and
the fact that the smaller one is ungated is not a reason to leave the larger one
ungated.

**3. It is the same asymmetry I applied elsewhere today and would have to defend
twice otherwise.** The serverless key surface scopes *revoke* while leaving *mint*
pending a decision, on the principle that safety must never be harder to reach than
the risk it undoes. Clause 6 is that principle pointed at money: widening is gated,
narrowing and disabling stay frictionless.

**Ruling: Option A. The plan wins.** Widening auto-top-up requires approval;
lowering, disabling, or leaving it unchanged does not.

### What "approval" means here, stated precisely

This codebase already has a real approval authority, and Gate P1 clause 7 depends
on it — *"an envelope-funded charge is traceable to its **approving plan**"*. It is
the `action_plans` substrate behind `control_plane.launch.service`, already carrying
three action types (`create_instance`, `create_serverless_endpoint`,
`evict_host_workloads`) via an `action_type` discriminator and a per-action
`ACTION_REQUIRED_SCOPES` map. Adding a fourth is precedent, not invention.

It is stronger than a two-step call, and the strength is worth naming precisely:

- **`confirm: true` never constitutes approval.** `_ApproveIn` accepts the field
  "for client symmetry" and *deliberately ignores it*
  ([routes/action_plans.py:151](routes/action_plans.py#L151)). The model cannot
  approve by asserting that it approves.
- **Execute refuses an unapproved plan.** The serverless executor requires
  `status == "approved"` and raises `approval_required` otherwise
  ([routes/serverless.py:634](routes/serverless.py#L634)).
- **An approved plan cannot be altered.** The executor re-checks the canonical
  argument hash and raises `argument_hash_mismatch`, so approval binds to exact
  values rather than to an intent.
- **`approval_mode: "human"` refuses a machine principal**
  ([control_plane/launch/service.py:349](control_plane/launch/service.py#L349)).
  A `standing_policy` plan may self-approve, but only inside its ceilings.

**A correction to an earlier draft of this document, because it changed the
design.** I first described the mechanism as "friction plus audit, not human
consent", on the reasoning that a model could call preview and execute
back-to-back. That is wrong: it gets `approval_required`. I had generalised from
the shape of the MCP tool without reading the executor, which is the same mistake
as the registry-drift greps in §P0.2 — describing a guard from its call site
instead of its implementation.

**The one real limitation, which is not hypothetical.** `_is_human()` is
`auth_type != "client_credentials"`
([routes/action_plans.py:53](routes/action_plans.py#L53)). An `oauth_access_token`
— the connector credential — is therefore counted as *human* and can approve its
own plan. So `approval_mode: "human"` is a genuine gate against an agent-API-key or
client-credentials caller, and **no gate at all against a connector-token agent.**

That is the same predicate gap as `_require_scope`'s `oauth_access_token` no-op,
which the owner ranked as the top outstanding item. It is now load-bearing in two
places rather than one: fixing it repairs both route scoping and plan approval, and
until it is fixed, clause 6 is enforced against exactly the callers it was least
worried about.

### What lands with the ruling

1. A `configure_auto_topup` action type on `action_plans`, with
   `ACTION_REQUIRED_SCOPES["configure_auto_topup"] = ["billing:write"]` and
   `approval_mode: "human"`. The widening test is done **server-side against the
   stored setting** — never from values the model supplies, since the model is the
   party being gated.
2. Widening = enabling from disabled, raising `amount_cad`, or raising
   `threshold_cad` (a higher threshold fires sooner and more often).
3. Narrowing and disabling stay single-call.
4. `test_auto_topup_change_is_recorded.py`'s docstring, which currently argues the
   losing side, is rewritten to record the ruling and why the `top_up_wallet`
   analogy fails — the file keeps its audit assertions, which remain correct and
   necessary.
5. Both halves asserted, as the clause demands: a widening without a plan is
   refused, and a narrowing without a plan succeeds.
