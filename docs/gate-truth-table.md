# Gate truth table — P0 through P3

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
| 3 | A live-credential path — ≥1 assertion per phase, real token, real server | **FAIL** | P0 has `tests/live/test_scope_refusals_live.py`, but it (a) asserts a *different* property than Gate P0 names — undelegatable operator scopes (`hosts:evict`), not the access/billing scopes — and (b) skips unless `XCELSIOR_LIVE_BASE_URL` + `XCELSIOR_NONADMIN_TOKEN` are set, and its only declared runner is `.github/workflows/live-gates.yml`, which cannot run. **P1, P2 and P3 have no live path at all.** |
| 4 | A refusal test — what the phase makes impossible | **PASS** | Dense across the corpus. `test_serverless_refuses_narrowed_credentials`, `test_platform_ssh_key_is_admin_only`, `test_terminal_ticket_needs_connect_scope`, `test_stripe_webhook_refuses_unverified`. |
| 5 | An eval delta per phase — re-run at the new tool count | **FAIL** | No eval has been run since the baseline was captured at `tool_count: 36`. The surface has grown substantially since. **P2 and P3 both shipped tools with no eval delta at all**, which is the clause simply not being honoured rather than being blocked — though re-running it now *is* blocked (§P0.4). |

Gate §1.3 and §1.5 are the two structural failures on this page. Everything else
is a clause-by-clause matter; these two are a habit that lapsed.

---

## Gate P0 — Foundations

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | Every access and billing endpoint refuses a token missing its new scope, **asserted with a real token against a live server** | **PARTIAL** | *In-process: true.* `instances:connect` on 4 routes, `ssh:read` on 1, `ssh:write` on 2, `billing:write` on 14 — including both endpoints the plan names by hand: `portal-session` ([routes/billing.py:2523](routes/billing.py#L2523)) and `setup-intent` ([routes/billing.py:2546](routes/billing.py#L2546)). *Live: absent.* The one live file asserts a different property (see §1.3). |
| 2 | Regenerating the registry produces byte-identical output; a hand edit to a generated file fails the build | **PASS** | `test_tool_scope_registry_completeness.py` (8 passed) and `test_generated_artifacts_are_current.py`. **Demonstrated, not assumed:** this session the inventory test went red the moment a route change made `docs/generated/endpoint-inventory.md` stale, and green after regeneration. It also caught a hand-written note in `docs/endpoint-classification.json` that my own change had falsified. |
| 3 | Zero unclassified endpoints | **PASS** | `MAX_UNCLASSIFIED = 0` across **519** operations — a floor, so the next endpoint must be classified in the commit that adds it. |
| 4 | Eval baseline captured | **SUPERSEDED + BLOCKED** | Captured at `tool_count: 36`, `expected_tool_accuracy: 0.8778` — **below the 0.90 threshold, which has not been moved.** The surface has since grown well past 36, so the baseline no longer describes the thing it grades. Recapture is blocked on Anthropic credit (~$4.96 of $5 spent). |

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
| 2 | Connection material is short-lived and single-use; a replayed ticket is refused | **PARTIAL** | *The implementation is genuinely correct* — `_consume_ws_ticket` pops the ticket ([routes/_deps.py:1515](routes/_deps.py#L1515)) and additionally pins purpose, target and client IP, with expiry purging. **No test asserts the replay refusal.** The property is true and unguarded, so a regression would be silent. |
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
| 1 | Promotion is idempotent under retry; a repeated call produces one volume, not two | **FAIL** | `promote_artifact_to_volume` **does not exist** — no implementation in any `.ts` or `.py` file. |
| 2 | The retention clock is asserted: an artifact past `retain_until` is gone, a promoted volume is not | **PARTIAL** | The artifact half is covered (`test_artifact_retention_authority.py`, `test_artifacts_janitor.py`). The "promoted volume is not" half cannot exist until promotion does. |
| 3 | Round-trip: train → promote → mount in a *new* instance → read the weights, tool calls only | **FAIL** | Depends on the tool that does not exist. |

**P3 is half-built, and the halves are unequal.** The volume surface shipped — 8
tools in `mcp/src/tools/volumes.ts`, with `detach_volume` behind approval and its
preview naming the attached instance. The *promotion* half — the thing P3 is named
for, and the substrate P4 and P5 are supposed to stand on — is not started.
`docs/artifact-promotion-plan.md` exists and sequences it A0→A4; no code does.

---

## Tally

| Gate | PASS | PARTIAL | FAIL | BLOCKED/SUPERSEDED |
|---|---|---|---|---|
| §1 universal | 2 | 1 | 2 | — |
| P0 | 2 | 1 | — | 1 |
| P1 | 4 | 3 | — | — |
| P2 | 1 | 1 | 1 | — |
| P3 | — | 1 | 2 | — |
| **Total** | **9** | **7** | **5** | **1** |

Nine of twenty-two clauses are fully met. It was eight when this table was
first written; P1 clause 6 moved from FAIL to PASS when the ruling below was
implemented.

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
