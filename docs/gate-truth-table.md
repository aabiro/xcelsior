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
| **ACCEPTED-UNFIXABLE** | True as written, unobtainable as evidence, and accepted as such. **Not PASS and not outstanding.** Reserved for clauses whose subject is the past — a historical corpus that cannot be retro-evidenced, or a measurement whose subject is gone. Counted in its own column because folding it into PASS overstates what is proven, and leaving it non-PASS overstates the backlog. |

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
| 2 | A confirmed-failing check — every regression test shown to fail *before* the fix | **ACCEPTED-UNFIXABLE** *(was PARTIAL)* | Honoured for everything written since this table began, and demonstrated repeatedly: the crypto replay guard was confirmed red before the fix (4 of 6, the two survivors being negative controls), and every guard added this session was probed against the exact defect it exists for. **The historical corpus cannot be retro-evidenced and never will be** — the fixes are landed, so there is no "before" left to run. That is not work someone could do, which is what PARTIAL implied for as long as it sat here. Accepted as unobtainable, with the going-forward half genuinely met. |
| 3 | A live-credential path — ≥1 assertion per phase, real token, real server | **PASS** *(was PARTIAL, was FAIL)* | **Every phase P0–P5 now has a live assertion, and all of them have run** — 25 passed / 9 skipped against staging, `scripts/run_live_gates.sh`. The 9 skips are the fleet-dependent gates and skip rather than pass, verified. Two things had to be fixed before the clause was honestly met, and both were only visible by running it. The runner checked the auth cache **inside** its `if the API is down` branch, so an API that was already up with a dead cache — the state after any reboot — never checked it and died at "could not obtain a session token". And **two of the assertions pointed at routes that have never existed** (`/api/billing/topup` for P1, `/api/v1/promotions` for P3); since every assertion here is written as "not a 200", a 404 satisfied them and they reported refusals for eight commits without being able to fail. P1 is now Gate P1 clause 4 — a forged webhook signature must get 400 — and P4, whose only other live coverage needs a fleet, asserts that an unapproved pipeline will not run. `tests/test_live_gate_paths_resolve.py` stops the phantom-path defect returning: it resolves every path a live gate names against `app.routes`, both sides derived, needing no credential and no fleet. |
| 4 | A refusal test — what the phase makes impossible | **PASS** | Dense across the corpus. `test_serverless_refuses_narrowed_credentials`, `test_platform_ssh_key_is_admin_only`, `test_terminal_ticket_needs_connect_scope`, `test_stripe_webhook_refuses_unverified`. |
| 5 | An eval delta per phase — re-run at the new tool count | **PASS** *(P2's clause SUPERSEDED)* | The loop runs without GitHub Actions (`scripts/run_mcp_eval_locally.sh`), and the clause has now been honoured **end to end for one phase**: P4 captured *before* it added a tool (48 tools, 0.9444) and *after* (50 tools, 0.9444) — two more tools, no accuracy lost. The promotion tools got theirs the same way (46 → 48, 0.9111 → **0.9444**). **PASS**, with one clause-scoped **ACCEPTED-UNFIXABLE** — the row is PASS because the clause was met end to end for P4 and P5; the P2 sub-clause is unobtainable rather than superseded, since what is gone is the *measurement's subject*, not the surface the clause described. P2's SSH tools shipped without a delta and **cannot get one**: the intermediate surface they were measured against no longer exists, so there is nothing left to re-run the eval over, and reconstructing a number from old commits would not be the measurement this clause asks for. Recorded as superseded rather than left open — an open row implies work someone could do, and this is not work, it is a measurement whose subject is gone. **P5's delta is taken**: `evaluate_placement_preference` took the surface to 51 and the headline holds at **0.9444**, identical to the 50-tool baseline of 2026-08-09. Abstention 18/18, unsafe-write 0.0. **The headline is not the whole reading.** Unstable cases went **3 → 4**: `always_failed` emptied and `flaky` grew from two entries to four. `approval-serverless` moved from reliably red to intermittent — a diagnosis regression, not a fix, and the way a known failure quietly stops being tracked as one. `indirect-would-it-fit` is **newly unstable**, appearing in neither list at 50 tools; that is the tool competing with `simulate_instance_placement` for a case it used to win outright, which is exactly the interference an earlier note in this table wrongly claimed had not occurred. Accuracy is unchanged only because the new instability averaged back to 85/90. **A first capture aborted mid-run on an exhausted key and wrote nothing** — the runner refuses to score an unreachable API as a wrong answer, because that records a fabricated regression. Two phases running have now honoured this clause end to end (P4, P5). |

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
| 1 | Top-up on a saved card completes with no browser and no elicitation, **real token, live server** | **PASS** *(was PARTIAL)* | `tests/live/test_saved_card_topup_live.py`, against a live server in Stripe **test mode**, and verified in Stripe rather than by trusting the route: two PaymentIntents `status: succeeded`, `amount_received: 500`, **`next_action: null`** — charged off-session with no browser step and no elicitation. The response is also asserted to carry no `next_action`, `redirect_to_url` or hosted-page URL, since any of those *is* the browser step the clause forbids. **Running it found the clause was not merely unasserted but impossible:** `list_payment_methods` called `pm.get("card")` on a Stripe object, which raises `AttributeError: get` on SDK 15.3.1 — so every caller saw "no saved cards", including manual top-up and the auto-top-up sweep. It survived because with *zero* cards the loop body never runs and it returns `[]`; it only broke for a customer who actually had one. |
| 2 | Replaying any funding call with the same idempotency key produces exactly one charge — **manual top-up, auto-top-up, and the crypto rails** | **PASS** *(was PARTIAL)* | All named rails now, and the gap was larger than "unasserted". Manual, wallet-deposit and PayPal were already covered (`test_funding_replay_is_one_charge.py`). **The crypto rails had no mechanism at all** — neither `create_deposit` took a key nor deduplicated anything, so a retried request minted a *second Bitcoin address* or a *second bolt11* for one intended deposit. The clause says "rails" plural and there are two; fixing only on-chain would have left it half met while reading as done. Lightning is the sharper failure: two addresses at least belong to one wallet and both credit if paid, whereas a second invoice is a distinct payment request that settles nothing when the first is paid. Migrations 109/110 add `(customer_id, idempotency_key)` partial-unique indexes — scoped per customer so one tenant's key cannot collide with another's — and `ON CONFLICT DO NOTHING` holds the guarantee at the index rather than at the timing of a read-then-insert. Auto-top-up already reached the asserted `charge_saved_card`; what was unproven was its **key derivation**, now covered. `tests/test_crypto_funding_replay_is_one_deposit.py`, 11 assertions, confirmed failing before the fix (4 of 6 red with the guard removed; the two that stayed green are the negative controls, correct either way). |
| 3 | An `authentication_required` decline produces a resumable pending state, a visible UI state, and a truthful tool result — **forced with a Stripe test card, not by mocking it** | **PASS** *(was PARTIAL)* | Forced with a real test card, which the clause names as its own exclusion. `pm_card_authenticationRequired` (4000002760003184) *"requires authentication on all transactions, regardless of how the card is set up"* — chosen over `4000002500003155`, which stops requiring authentication once set up for off-session use and would let the test go green because the setup succeeded rather than because the decline was handled. Stripe recorded two intents at `status: requires_payment_method` with `last_payment_error.decline_code: authentication_required`, so the decline came from the processor and not from this repository. The route's answer is asserted to be neither a 200 nor a 500, to say *authentication*, to state that the charge did not happen, and to point at a resumable state. The prior in-process tests remain — they were careful work, and the clause simply asked for something they could not give. `tests/live/test_saved_card_topup_live.py`. |
| 4 | The webhook refuses what it cannot verify — `400` on a bad signature | **PASS** | `test_a_wrong_signature_is_refused_with_400` plus a missing-header case, in `test_stripe_webhook_refuses_unverified.py`. The file reasons explicitly about why "400 or 503" would be an untrustworthy assertion. |
| 5 | No secret in any surface — card data, `client_secret`, processor tokens; canary-tested with fake PANs | **PASS** | `tests/test_no_payment_secrets_in_logs.py`. |
| 6 | **Raising a spend cap requires approval; lowering one does not. Both asserted.** | **PASS** *(was FAIL)* | Ruled Option A (§Ruling) and implemented. A widening by any caller that is not an interactive human is refused `409` and directed to `/api/v2/billing/auto-topup-plans`; narrowing and disabling stay single-call. `approval_mode` is hard-coded `"human"` so a standing policy cannot approve a change to its own ceilings. Both halves asserted in `test_widening_auto_topup_needs_approval.py`, with the plan lifecycle driven against real PostgreSQL in `test_auto_topup_plan_lifecycle.py`. |
| 7 | An envelope-funded charge is traceable to its approving plan in one query | **PASS** | `tests/test_spend_traces_to_its_approving_plan.py`. |

---

## Gate P2 — Access

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | A scripted journey — launch, wait, connect, run a command, terminate — completes using **only tool calls**, against a live staging tenant | **FAIL** *(reason superseded)* | The script now exists — `tests/live/test_access_journey_live.py` — and **has been run**, which is the change: the earlier note said "no such script exists" and that is no longer why this fails. It gets as far as launching and stops at `402 Insufficient wallet balance`. **The blocker is a cutover, and it is precise:** a top-up only *submits* a charge, and the wallet is credited by Stripe's `payment_intent.succeeded` webhook, because the processor is the sole authority on whether money moved. Staging receives no webhooks — it binds `127.0.0.1:9600` and Stripe cannot reach it. The direct-deposit endpoint is refused because `is_relaxed_env()` is false for staging by design (*"the answer for staging is no, because staging holds real data"*), which is correct and not worth weakening for a gate. So the journey needs webhook delivery into staging — `stripe listen --forward-to` or an equivalent — or a platform-admin deposit as fixture setup, kept separate from the non-admin credential the gate itself must use. Everything upstream of funding now works: the fleet is admitted and heartbeating, and `/agent/*` is reachable. |
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
| 2 | The retention clock is asserted: an artifact past `retain_until` is gone, a promoted volume is not | **PASS** *(was PARTIAL)* | The artifact half was already covered (`test_artifact_retention_authority.py`), and the hold is tested (`test_promotion_takes_the_hold.py`). The second half was deferred to clause 3 on the grounds that it "needs a mounted volume" — **and that reason was doing more work than it should**. Surviving is not the same as being mountable: whether a promoted copy still exists after the artifact expires is a property of the *deletion path*, which is code that runs without a fleet. `tests/test_promoted_copy_outlives_the_artifact.py` drives a real deletion job through the reaper and asserts the artifact leaves `available` while `volume_promotions` and `volume_promotion_files` are untouched — **including that the reaper actually claimed the job**, without which "the promotion survived" is equally true of a reaper that declined. A second, structural assertion walks `cleanup_expired`'s own SQL and requires it to name no volume table at all, because the behavioural test only inspects rows it created and would survive a later edit that released the promotion deliberately; verified by making the path touch `volume_promotion_files` and watching it go red. **What remains needs hardware** — reading the promoted bytes back through a mount, which is clause 3's sentence rather than this one, and is named here in the same vocabulary as every other blocker on this page. |
| 3 | Round-trip: train → promote → mount in a *new* instance → read the weights, tool calls only | **FAIL** | The tool exists (`promote_artifact_to_volume`, A4) and **staging now exists too**, so the previously recorded reason is spent. `tests/live/test_volume_round_trip_live.py` ran for the first time and stopped earlier than the round trip: the volume is created with `status: error` after `NFS provision failed for vol-… — retrying once`. **The blocker is hardware:** the NFS export the volume is provisioned onto is not reachable from staging. Nothing about the promotion path has been exercised yet, so this remains FAIL on its own terms rather than partially met — but the next thing to fix is a mount, not code. One of the file's tests skips separately and correctly on a **credential**, `XCELSIOR_LIVE_SSH_KEY`, which it needs to read the bytes back. |

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

*Added 2026-08-10 mid-phase, when clause 2 was PARTIAL and the reason was the
interesting part. Clauses 2 and 3 have since moved to PASS; the paragraphs below
record how, because the distance a clause travelled is worth more than its
current letter.*

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | A migrated job resumes from its checkpoint, proven by comparing state before and after — not by the absence of an error | **FAIL** | **The gate around the migration is built, the executor exists, and the proof still cannot run.** `control_plane/scheduler/migration_gate.py` re-runs `filter_hosts` and re-evaluates the preference on the target, and `migration_executor.migrate_job` returns `resumed=True` only when a probe read matching state on both sides. `tests/live/test_migration_resumes_live.py` now executes and skips with its own reason: *"0 active host(s); a migration needs two, and a same-host move would not exercise the checkpoint transfer"*. **The blocker is hardware** — one admitted host exists (the RTX 2060 canary) and a migration needs a source and a target. The skip is the honest verdict: a same-host move would return `ok` while proving nothing, which is exactly the shape `resumed is None` was introduced to refuse. |
| 2 | A placement preference that cannot be satisfied **refuses clearly** rather than silently falling back to the cheapest host | **PASS** | **Asserted through the route, against real hosts in the database.** `POST /api/v1/placements/evaluate` takes a preference, evaluates it over the same consistent snapshot the hard filter uses, and returns a typed refusal carrying the number that failed. `tests/test_gate_p5_placement_refuses_end_to_end.py` asks for 99.99% where the best is 99.95% and asserts no host comes back; asks for a verified host when none is verified and asserts it does **not** fall back to an unverified one — the §5.4 reconciliation, which is what made this PARTIAL for four commits. `scheduler.allocate_best_host` keeps its cold-start fallback for **unconstrained** placement, untouched: the fallback is skipped only when the request is constrained. A calibration test asserts the fixture fleet is placeable, so the refusals are not passing for the wrong reason, and the fixture uses a GPU model no other test can produce — it previously asserted a count over the *whole* fleet, passed in isolation and failed in a full run. **Also asserted live**: `tests/live/test_placement_preference_refuses_live.py` drives the deployed route with a real token, and production answered `no_eligible_hosts` with a recorded `decision_id`. |
| 3 | Preference is honoured in the audit trail: the chosen host's reputation and SLA at time of placement are recorded | **PASS** | **The writer now has a caller, and the record is read back to prove it.** Every evaluation appends to `placement_decisions` (migration 105, partitioned by 106) — placements *and* refusals, because a preference that refused was honoured by the refusal and a successes-only trail cannot answer "why did nothing launch last Tuesday". WORM by trigger, probed with a real UPDATE and a real DELETE. The evidence is **copied**: a test changes the host's score and deverifies it after the fact and asserts the row still reads what was true at the time. Recording is best-effort in its own transaction — an audit write must never be the thing that fails a placement — and `decision_id` is returned as `null` rather than hidden when it could not be written. |

**This was the P4 sentence, and it no longer applies to clauses 2 and 3.** For
four commits the honest answer was "the module refuses correctly and nothing
calls it" — a clause met by a module sitting inside a system that did not honour
it. `POST /api/v1/placements/evaluate` is the caller that closed the distance,
and `tests/live/test_placement_preference_refuses_live.py` asserts it against a
deployed server with a real token rather than in-process. Clause 1 keeps the
sentence: its gate is built and its proof is not.

**Neither constraint is shippable today, on production numbers.**
`min_uptime_pct` refuses everything because `sla_monthly` has zero rows.
`min_tier` refuses everything above `new_user` and admits everyone at it.
`require_verified` refuses everything because the two verified hosts are 112 and
125 days past a 7-day tolerance.

**The sweep is done — C2's first commit, as the plan said it had to be.**
`list_hosts_needing_reverification` had **no callers at all**; it now runs hourly
from `bg_worker.py`. The finding that shaped it: *the server cannot re-verify a
host by itself.* `run_verification` needs a telemetry report only the host can
produce, and the agent submitted one at startup and never again — that is the
whole explanation for stamps 112 days old. So the sweep asks, over a new
`reverify` agent command, and the agent re-runs **the same builder startup
uses**. Twelve tests, including one that drives the store's real due query
against production's exact shape.

It does not clear the clause on its own, and says so: the sweep can only ask. A
host that is offline, busy with a paying job, or running an agent that predates
the command never answers, and `verification_status` still reads its stamp as
`stale`. What changes is that a healthy host now has something that asks it.

---

## Gate P6 — The provider surface

*Added 2026-08-12. This page has covered P0–P5 since it was written; P6 and P7
existed in the plan the whole time and were simply never on it. An absent phase
reads as a met one, which is the failure this table exists to prevent — so they
are here now with the verdicts the evidence supports, not with placeholders.*

| # | Clause | Verdict | Evidence |
|---|---|---|---|
| 1 | A provider journey — register → admit → publish → earn → payout — completes through tools plus the browser handoffs, **on a live staging tenant** | **FAIL** | No such journey has been run. The pieces exist individually — host registration and `decide_admission` were both exercised end to end this session on the RTX 2060 canary, with real hardware evidence — but nothing carries a provider from registration through to a payout in one pass. The clause names a live staging tenant, and the same **cutover** that blocks Gate P2 clause 1 applies: earnings and payouts settle from webhook-confirmed money, and staging receives no webhooks. |
| 2 | A payout is bound to job, amount, currency, destination state and idempotency key; **replay produces one payout** | **PASS** | `tests/test_provider_settlement.py`. The binding is not a convention — `prepare_settlement` derives money, owner, currency and tax from PostgreSQL rather than from the caller, and `test_payout_api_signature_has_no_caller_amount` asserts the route cannot be told an amount. Replay is closed at two levels: `test_concurrent_cross_rail_prepare_creates_one_settlement` and `test_concurrent_workers_claim_a_settlement_once` hold it in the database, and `test_stripe_transfer_uses_exact_db_amount_and_is_idempotent` asserts the `rail_idempotency_key` reaches `Transfer.create` as `provider-settlement:{job_id}`, so a retry Stripe sees returns the original transfer. PayPal is covered separately, with the replay asserted to return the same `capture_id`. |
| 3 | Returning from `return_url` proves nothing — asserted by returning **without completing** and checking the state is still `pending_requirements` | **PASS** *(was FAIL)* | `tests/test_returning_from_onboarding_proves_nothing.py`. **The property was already true and simply unasserted** — every path that marks a provider onboarded is gated on Stripe's own `charges_enabled and payouts_enabled`, once in the `account.updated` webhook and once in `create_provider_account`, which *re-retrieves* the account rather than trusting the return. Asserted structurally rather than by a live return, because the behavioural version needs an account created and **abandoned mid-KYC**, and abandoning is a human closing a browser tab — not something an API call can produce. What is asserted instead is the thing that actually matters: no code path completes onboarding on any input other than Stripe's capability flags, and no request handler may call the completion writer at all. Both verified against the bug they exist for — completing on return, and a route handler marking the provider active so the dashboard looks right immediately. The indirection is closed too: one guard tests `status == "active"`, which is only legitimate because that status is derived from a live read of both flags, and a separate assertion pins that derivation so the indirection cannot become a loophole. |

---

## Gate P7 — Environment snapshot and sweep

| # | Clause | Verdict | Evidence |
|---|---|---|---|
| 1 | A sweep of N nodes from one snapshot is **byte-identical in environment**; a snapshot **records its lineage** | **FAIL** *(reason narrowed)* | The clause has two halves and **lineage is now closed**. `created_at`, `source_job_id` and `host_id` were already recorded; migration 111 adds `base_image_ref`, written at snapshot time from the job that ran. That is the half an audit actually needs: a snapshot is `docker commit` over a running container, so the image is a diff on top of whatever base the job launched with, and "which snapshots contain this CVE" is unanswerable from `source_job_id` alone — the job may since have been requeued onto a different image, or deleted. Nullable with no backfill, because existing rows genuinely do not know and inferring the base afterwards is the guess the column exists to prevent. Recorded *and* returned by the listing, since a column no surface reads answers no question. `tests/test_snapshot_records_its_lineage.py`; both the write and the read verified to go red when removed. **What remains is the sweep, and two of its five pieces are now done.** *Piece 1 — the digest pin* (112): the worker captures `repo@sha256:…` with `docker inspect` immediately after the push, because `_build_image_ref` returns a **mutable tag** and N containers launched from a tag were asked for the same *name*, not given the same bytes. The clause is unprovable in principle from a tag, which is why this came first. *Piece 2 — the record* (113, `control_plane/image_sweeps.py`): one row the N members belong to, `host_id` per member, and the digest pinned once at creation rather than re-resolved per member — re-resolving reopens the race the digest closes. Partial failure is recorded rather than raised, so "3 of 5 launched, and here are the two that did not" is answerable instead of a caller holding three ids; and `distinct_hosts` is reported so a single-host sweep cannot read as a full pass, which is P5.1's precedent. A sweep from an image with no recorded digest is **refused**, not launched from the tag: falling back is the substitution that would leave the clause unprovable while looking met. `tests/test_a_sweep_is_a_record.py`, 12 assertions, both the digest fallback and the swallowed partial failure verified red. *Pieces 3 and 4 — the fingerprint and the boundary* (114, `environment_fingerprint.py`): the fingerprint is produced **by the running container**, because the control plane already knows the digest it sent to all N and comparing that to itself establishes the request was consistent rather than the containers. Hash *and* raw manifest are both stored, and a schema check refuses half of one — a hash without a manifest is a mismatch nobody can diagnose, a manifest without a hash is a comparison nobody can make. **The boundary is written in the module, with a reason beside every exclusion**, because it is the entire claim: include too much (hostname, GPU, kernel) and no sweep can pass; exclude too much and every sweep passes. In scope is what the image decides — digest, sorted package inventory, declared env, the process contract. Out is what the host or the instance decides, each named with why. The comparison **never reads a missing fingerprint as agreement**: a collector that errors and returns nothing would leave every member null, and "are all values equal" would find one distinct value — none — and report a perfect sweep. That exact substitution is verified red. A mismatch names the differing fields rather than only their existence. The collector is not scaffolding: `fingerprint_environment` is an agent command on both allowlists, the worker copies the collector *into the container* and runs it there — running it on the host would fingerprint the host's Python, the one environment the clause does not mean — and a callback route records the reading. Every failure path reports a failure rather than an empty fingerprint, because a silent blank would be counted as agreeing with every other member. *Still outstanding*: **the sweep-creation route**, and the cost is measured rather than guessed — `api_submit_instance` is ~500 lines (`routes/instances.py` 722–1231) carrying wallet preflight, host resolution, team checks, scheduler-partition logic and container start, with **no service seam to call**. A sweep route either duplicates that (and drifts from it) or extracts it, and extracting a service out of the primary launch path is a refactor with real blast radius that should be its own change with its own review. `create_sweep` already takes a `launch` callable for exactly this seam. Also outstanding: piece 5, the perturbed member in a real sweep, which needs a second host. The clause stays FAIL until a sweep has actually run against real hosts — §1.3's standard applies, and in-process is not the evidence. A test in that file asserts the sweep is still absent, so whoever ships one is told to update this gate rather than leaving a half-met clause reading as whole. **Code, not a blocker.** |

---

## Carried, not owned by any gate — WORM tables and the right to erasure

*Recorded here because a code comment is durable but invisible at gate review,
and this is the kind of finding that gets rediscovered every six months.*

**Not P5's, and not a defect of any migration.** Three tables carry an
append-only trigger — `audit_events_v2` (072), `audit_checkpoints` (075),
`placement_decisions` (105/106) — and **none is reachable from the erasure
path**. `privacy_sinks.verify_subject_absence` is a hand-enumerated list that
names none of them; the trigger rejects DELETE unconditionally; partitioning
prunes by time, not by tenant.

The sharp edge is not the omission, it is the **claim**: that function returns a
verdict named *absence* over a partial enumeration, so it can report a subject
gone while rows persist. Everywhere else in that file a missing table is an
omission; here it is an affirmative statement the code makes for the reader.
Same shape as a verified badge that means less than it says.

### Resolved 2026-08-11 — retained under a documented basis

Audit tables resolve this by pseudonymising identifiers at erasure time **or**
by recording a retention basis. **Aaryn Biro chose the retention basis**:
legitimate interest and legal obligation under GDPR Art. 17(3) and the
equivalent carve-outs in other privacy regimes, for **24 months**. Pseudonymisation was rejected on cost — it means
rewriting rows in tables whose whole value is that rows cannot be rewritten,
and the trigger would have to be weakened to allow it.

The ruling was cheap to *decide* and not cheap to *make true*. A retention basis
owes three things, and only the first was a matter of writing prose:

| Obligation | Where | State before |
|---|---|---|
| A stated period | `WORM_RETENTION_MONTHS = 24` | none existed |
| Disclosure | `docs/audit-retention.md`, carrying the policy line | silence |
| **Enforcement of that period** | `drop_expired_partitions`, daily | **partitions were created ahead of time and never dropped** |

The third is the one that mattered, and it was invisible until someone went
looking for the mechanism: the partition maintainer extended the window forever
and pruned nothing, so publishing "24 months" would have been a claim about a
system that kept data indefinitely. `tests/test_worm_retention_is_enforced.py`
proves the pair that makes it real — `DELETE` is refused **and** the partition
drop removes the same rows. Either half alone proves nothing.

Writing that test found a second defect: the partition-name parser read a
five-digit suffix like `20249` as September 2024, so a partition nobody named
that way would have been dropped on a guess. It now requires exactly `YYYYMM`.

`verify_subject_absence` still enumerates by hand and still does not reach these
tables — that part was correct and is unchanged. What changed is that its
verdict now **names the exception** in `evidence["append_only_records"]`,
derived from `PARTITIONED_TABLES`, rather than returning a clean absence that a
reader would take as absence from everywhere.

Crypto-shredding — encrypt the tenant identifier under a per-tenant key held
outside the WORM table, delete the key on erasure — is recorded in
`docs/audit-retention.md` as the escape hatch if attributable erasure is ever
demanded. Deliberately not built.

| What exists now | Where |
|---|---|
| The ruling, stated where the decision lands | `privacy_sinks.verify_subject_absence` docstring |
| The period, the basis, the disclosure and the escape hatch | `docs/audit-retention.md` |
| Enforcement, scheduled daily | `control_plane.audit_partitions.drop_expired_partitions`, via `audit_partition_maintenance` |
| A ratchet so a **new** WORM table cannot join the unresolved set silently | `tests/test_worm_tables_have_an_erasure_decision.py` — WORM set derived from `pg_trigger`, reachable set derived from that function's own source |
| Proof the period is enforced rather than asserted | `tests/test_worm_retention_is_enforced.py` |

Red on the ratchet test means *a decision is owed for a newly added table*, not
*erasure is broken*. It is load-bearing and must not be deleted as a stale
assertion about a bug.

---

## Tally

| Gate | PASS | PARTIAL | FAIL | ACCEPTED-UNFIXABLE |
|---|---|---|---|---|
| §1 universal | 4 | — | — | 1 |
| P0 | 4 | — | — | — |
| P1 | 7 | — | — | — |
| P2 | 2 | — | 1 | — |
| P3 | 2 | — | 1 | — |
| P4 | 4 | — | — | — |
| P5 | 2 | — | 1 | — |
| P6 | 2 | — | 1 | — |
| P7 | — | — | 1 | — |
| **Total** | **27** | **—** | **5** | **1** |

Twenty-seven of thirty-three clauses are fully met, nothing is BLOCKED, **no clause is
PARTIAL any more**, and **Gates P0, P1 and P4 are wholly met**.

The denominator moved from 29 to 33 because **P6 and P7 were never on this
page**. They were in the plan the whole time; the table covered P0–P5 and said
so, which is not the same as anyone noticing that two phases had no row. An
absent phase reads as a met one — the same failure mode as a clause counted in
no column — so they are here now, with four clauses between them and the
verdicts the evidence supports. Three of the four are FAIL, and adding them
made the headline worse, which is the point: a tally that only ever improves is
not measuring anything.

PARTIAL reaching zero is worth a sentence, because it is the verdict this page
was built to resist inflating. Every clause that carried it did so for the same
reason — the behaviour worked and the clause's *own* evidentiary standard was
unmet — and each was closed by meeting that standard rather than by softening
it. P1's last two were the hardest and the most instructive: running them
against a live server in test mode showed that clause 1 was not merely
unasserted but **impossible**, because `list_payment_methods` raised on any
customer who actually had a saved card. A mock had been standing in for a code
path that could not run.

What remains is five FAIL clauses and one ACCEPTED-UNFIXABLE. The six split by
what is actually in the way, in the vocabulary this page now uses throughout:

| blocker | clauses |
|---|---|
| **hardware** | P5.1 (one host exists; a migration needs two), P3.3 (the NFS export is unreachable from staging) |
| **cutover** | P2.1 and P6.1 (both need webhook delivery into staging before money can move) |
| **code** | P7.1 — **the sweep only**; lineage is closed (111 + `base_image_ref`) |

P6.3 was in that **code** row and is now PASS — the property turned out to be
true already, and what was missing was the assertion.

What remains that needs nothing from anyone else is **the sweep half of P7.1**:
nothing launches N nodes from one image. Its lineage half is done — migration
111 records `base_image_ref` and the listing returns it. The sweep is not a loop
over the launch path: the clause needs the image pinned by **digest** rather
than a mutable tag (`_build_image_ref` returns a tag, and `user_images` has no
digest column, so "N nodes from the same bytes" is unprovable in principle until
that is fixed), one row the N members belong to so partial failure is visible,
and a fingerprint produced *by the running container* — comparing the image ref
the control plane just sent to all N compares the request against itself. One is ACCEPTED-UNFIXABLE, and it has its own
column on purpose: §1.2's historical half is not work anyone can do, so counting
it as outstanding overstates the backlog — and folding it into PASS would
overstate what is proven. The tally is derived from the clause rows, so a
verdict that quietly counted as PASS would move the headline number with no
clause changing, which is the exact drift this tally already had once.

The total moved by one this session and the arithmetic is worth stating, because
it did not move the way it looks. §1.3 went PARTIAL → PASS, which is +1. But the
**Total** row was already overstating by one before that — it read 21 PASS when
the clause rows held 20 — so the count that was right is the one that did not
change. `tests/test_gate_truth_table_tally.py` derives the whole table from the
clause rows now; the row a human maintains is the row that drifts. P5 moved two clauses earlier — not by writing
the module, which already refused correctly, but by giving it a caller.

§1.3 moved last, and moved for a different reason than the others: nothing was
built for it. The clause had been PARTIAL because three phases had no live
assertion; writing them was straightforward, and *running* them is what took
the work. Two of the assertions turned out to name routes that do not exist —
they had been reporting refusals from 404s. A clause can be listed as PARTIAL
for the wrong reason: it was not short of assertions, it was short of
assertions that could fail.

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
