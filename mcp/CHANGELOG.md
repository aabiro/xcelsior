# Xcelsior MCP — surface changelog

Every change to the published tool surface at `https://mcp.xcelsior.ca/mcp`,
newest first. Breaking changes are listed first within a release and carry a
removal date. The contract this changelog serves is
[docs/mcp-tool-versioning.md](../docs/mcp-tool-versioning.md).

The machine-readable form of the current surface is
[`tool-surface.json`](tool-surface.json); a breaking change that is not
reflected here and version-bumped fails the build.

---

## Unreleased

### Breaking

- **Seven tools no longer claim that repeating a call is free.** `run_training_job`,
  `schedule_under_budget`, `create_volume`, `snapshot_volume`, `run_pipeline` and
  `open_instance_access` move to `idempotency: "none"` and
  `idempotentHint: false` (**2.0.0 → 2.1.0**); `schedule_under_budget` also gains
  `openWorldHint: true` (**→ 2.2.0**).

  `idempotency` defaulted to `"keyed"` for anything not read-only, so 25 tools
  advertised *"calling this again has no additional effect"* when only four sent
  an idempotency key. `run_training_job` and `schedule_under_budget` both POST
  `/instance` with no key, so a client that trusted the hint and retried a call
  that appeared to time out would launch **a second instance and be billed for
  both**; `open_instance_access` mints a fresh single-use ticket per call rather
  than returning the previous one.

  **No notice period, and that is deliberate.** §3's notice exists so a client
  is not surprised by a deliberate design change. This is the retraction of a
  promise that was never true, and every day it stands is a day a client can
  lose money by believing it. Tools whose repeat really is harmless are
  unaffected — `terminate_instance`, `register_ssh_key`, which 409s on a
  duplicate fingerprint, and `promote_artifact_to_volume`, whose endpoint
  carries its own idempotency key.

  `schedule_under_budget` reads `/api/v2/gpu/available` and
  `/api/v2/marketplace/spot-prices` — the two live third-party feeds behind
  `list_available_gpus` and `get_spot_prices`, both already flagged — and then
  spends against the answer, so a cached reading must not be assumed to hold.

  **Action:** if you retry these tools on timeout, check state first —
  `list_instances`, `list_volumes`, or the volume's snapshots — rather than
  calling again. Each tool's description now says so.

### Deprecated

- **`attach_volume`'s `instance_id` input** — use `job_id`. Both work today;
  `instance_id` is removed on or after **2026-11-13** (90 days from this entry,
  per §3).

  It was the only tool of seventeen calling the instance identifier
  `instance_id`; every other one — `get_instance`, `terminate_instance`,
  `get_instance_logs`, `create_instance_snapshot` — calls it `job_id`. A model
  holding ids from `list_instances` had no reason to believe they fit here, and
  nothing in the surface said they did. The API field is unchanged; only the
  tool's vocabulary moves.

  Shipped additively rather than as a rename because §3 requires the old shape
  to keep working for the full period, and because a rename would be a breaking
  input change — which the immediate-ship carve-out does **not** cover: that
  carve-out is for retracting a claim that was never true, and `instance_id`
  worked exactly as documented. It was inconsistent, not false.

### Added

- **`list_funding_options`** — what to do when the card is refused.

  P1's promise is that a declined charge is not a dead end, and three rails that
  need **no browser at all** — Bitcoin, Lightning, PayPal — had no tool between
  them. The only answer an agent could give was "go to the dashboard", on the
  axis whose whole premise is not having to.

  One tool, not four capability flags: `is_paypal_enabled` beside
  `is_lightning_enabled` makes a model choose between things that are one answer
  to one question. Each rail is probed independently, and one that errors is
  reported unavailable rather than failing the call — the point is to find the
  rail that works when another just failed.

  **Report `available`, not `enabled`.** A rail can be configured and still be
  down: Lightning currently reads `enabled: true, available: false` because its
  node is waiting on Bitcoin. The description says so, because the difference is
  the whole value of asking.

  Creates no deposit and moves no money.

- **`get_wallet_history`** — where the money went.

  `get_wallet_balance` returned a number and nothing on the surface could explain
  it. "Why is my balance lower than I expected" is the first question after an
  unexpected figure, and the only honest answer an agent could give was to send
  the user to the dashboard. It lists deposits, charges and refunds, newest
  first. `get_spend_envelope` remains the forward-looking one — burn rate and
  runway — and the descriptions point at each other so a model picks the right
  half.

- **`register_provider` and `request_provider_payout`** — P6's journey now runs
  end to end: register → admit → publish → earn → payout, through tools plus the
  browser handoffs.

  `register_provider` enrols the caller and starts Stripe Connect onboarding. It
  became safe to build only when the API stopped letting the caller name
  `provider_id`: until then a retry on a timeout minted a **second** Connect
  account and orphaned the first. The identity now resolves from the credential,
  so a repeat returns the same account. It does **not** return the onboarding
  URL — an AccountLink can set the external bank account — so finish in the
  browser and read what is outstanding with `get_provider_account`.

  `request_provider_payout` settles one completed job. The caller names a job and
  a rail and nothing else: amount, ownership, currency and terminal state are
  derived from PostgreSQL under `FOR UPDATE`, and settlement inserts
  `ON CONFLICT (settlement_key) DO NOTHING` with a per-job idempotency key handed
  to the rail — so replay produces one payout. It is deliberately not
  launch-plan gated: plans bound spend the caller chooses, and here the caller
  chooses no amount at all.

  **Amounts on the stripe rail are integer micros**; the paypal rail returns
  `_cad` fields. The description says so, because reporting one as the other
  overstates by a millionfold.

  Neither is reachable by Quick Connect: `providers:write` enrols a payout
  destination and moves money out of it.

- **`get_reputation_leaderboard` and `get_reputation_history`** — the rest of the
  reputation surface. The leaderboard gives a provider's own score a scale;
  the history is the itemised event log behind a disputed score, which
  `get_reputation_breakdown`'s four totals cannot explain on their own.

  Two sibling routes are deliberately **not** tools. `GET /api/reputation/{id}`
  duplicates what `search_marketplace` already returns on every listing, and
  `POST /api/reputation/verify` is admin-only — granting a verification badge is
  the platform asserting it checked something, and a tool for it would let a
  caller vouch for itself.

- **`claim_reputation_milestones`** — the exit for the journey.

  `get_reputation_journey` reports how many milestones are ready to claim, and
  nothing could claim them: the read-without-an-act shape that left `list_ssh_keys`
  and the serverless cancels missing. It grants only what is already earned —
  progress is recomputed from real account and activity state, so it cannot
  fabricate a milestone — and repeats are skipped rather than granted twice.

  **Not reachable by Quick Connect.** Unlike the four reputation reads, this is
  not a disclosure judgement: `reputation:write` awards points and verification
  badges, which move a provider's tier and therefore the commission they pay.

- **`get_paypal_status`** — the other payout destination. Stripe and PayPal
  onboard separately, so "payouts are blocked" has two possible causes and only
  one was readable.

  The route was returning `merchant_id`, `payer_id` and `tracking_id` — the same
  identifiers its four sibling provider reads redact, under shorter names. The
  redaction boundary had a hole shaped exactly like the fields it was drawn
  around. Nothing rendered them and the dashboard's own response type does not
  declare two of the three, but a tool response lands in model context and audit
  records, so the route now redacts them like its siblings.

- **`get_my_reputation`, `get_reputation_journey`, `get_trust_tiers` and
  `get_reputation_breakdown`** — P6's yield axis.

  Reputation decides a provider's tier, and tier decides the platform commission
  they pay, their ranking in search, and the pricing premium they can charge. So
  it is the half of *"why am I not earning more"* that earnings figures cannot
  answer. `get_reputation_journey` is the actionable one: milestones with live
  progress computed from real account and activity state, so the reply is a list
  of specific next steps rather than general advice.

  `get_trust_tiers` returns thresholds and rewards from the live scoring engine,
  not from documentation — quote what it returns rather than any number you
  remember. Every figure the platform tells providers is derived from
  `reputation.py` through `provider_expectations.py`, and a test fails when the
  messaging and the engine disagree.

  All four are reachable by Quick Connect, decided on the routes: `/me` and
  `/me/journey` resolve the subject from the caller's own credential,
  `/api/trust-tiers` is one public ladder with no personal data, and `breakdown`
  is behind an owner-or-admin check. The plain score stays open by design — a
  buyer comparing hosts needs it — while the history it is derived from does
  not. `reputation:write` is not granted: it claims milestones, awarding points
  and verification badges.

- **`list_providers`, `get_provider_account` and `get_provider_earnings`** — the
  first tools for the provider persona: the person supplying GPUs, not renting
  them.

  P6's clause is "a provider journey — register → admit → publish → earn →
  payout — completes through tools plus the browser handoffs", and thirty-five
  provider endpoints had no tool between them. These two answer the questions a
  provider actually asks: *am I set up to be paid?* and *what have I earned?* —
  the latter separating **earned, paid and pending**, because a figure differing
  from expectation is usually that distinction.

  `list_providers` exists because the other two shipped requiring a `provider_id`
  and named it as the source — while it did not exist. Ids are not guessable, so
  both tools were uncallable by the agent they were written for.

  **All three are reachable with a Quick Connect token.** They were not at first:
  `providers:read` was withheld on the argument that Quick Connect is issued to
  customers and a provider is a different persona. That was reasoned by analogy
  to `ssh:read` and is wrong here. All four routes behind the scope are
  owner-scoped — the listing filters to the caller's own account and returns
  nothing for someone who supplies no hardware, the per-provider routes refuse a
  caller who is not the owner, and every one redacts the Stripe and PayPal
  identifiers. Granting it to a customer discloses nothing; it hands a capability
  only to someone who already has one. `ssh:read` differs precisely there — it
  discloses key material.

  `providers:write` is **not** granted. Register, disconnect and
  resume-onboarding move a payout destination.

  Reads only. Requesting a payout moves money and waits on webhook delivery into
  staging.

- **`get_event_history`** — the recorded event history for one instance or host.

  `events:read` was granted at the consent screen and required by no tool: a
  permission asked of every connecting user that authorised nothing they could
  reach. The check that settled whether to build or drop it — the account-wide
  `/api/events` is `_require_admin`, so it was never an agent capability, while
  the per-entity history is a different question from `get_instance_timeline`.
  §20.3's timeline explains how a job was *placed*; this is the record of what
  happened afterwards, and it covers **hosts**, which no tool could read at all.

  Dropping the scope instead would have forced every existing connector to
  re-consent when the tools were eventually wanted.

- **`revoke_launch_plan`** — withdraw a quoted plan so it can no longer be
  approved or executed.

  Four tools quote a plan — `create_instance`, `create_serverless_endpoint`,
  `run_pipeline`, `create_image_sweep` — and none could withdraw one, so
  "actually, cancel that" had no answer short of a browser. A plan left sitting
  stays approvable later by anyone who can approve it.

  The one write on this surface with **no confirm gate**, deliberately:
  everything else previews because it can spend or destroy, and this only ever
  removes the ability to spend. Requiring a confirmation for the safe direction
  teaches people to click through the ones that matter.

  `approve` remains deliberately absent. The approval is the human in the loop;
  a tool for it would let an agent authorise its own spend.

- **`list_ssh_keys` and `delete_ssh_key`** — see what has shell access, and take
  it away.

  `register_ssh_key` grants shell access — its scope comment says so — and an
  agent could take that step while being unable to show which keys were
  authorised or revoke one. `ssh:read` was declared when the scope was split and
  **no tool ever used it**: the read half was designed and never built.

  **Neither is reachable with a Quick Connect token**, and that is deliberate.
  The connector holds `ssh:write` and not `ssh:read` so it can register its own
  key without enumerating the account's — which would reveal what other machines
  and people can get in. `delete_ssh_key` requires read *as well as* write for
  the same reason: you cannot revoke what you cannot list, and revocation
  disconnects anyone currently using that key. Full-scope credentials are
  unaffected.

- **`get_auto_topup`** — read the current automatic top-up settings without
  changing them.

  There was no read. The only way to learn the settings was to call
  `configure_auto_topup`, which returns the previous values in its response —
  **you had to write in order to read**, on the one surface that authorises
  charges with nobody present. `GET /api/v2/billing/auto-topup` had been
  labelled `covered` in the endpoint classification while nothing called it.

- **`list_pending_verifications`** — the top-ups that stopped because a
  cardholder's bank wants them confirmed. The money has not moved and the wallet
  was not credited.

  `top_up_wallet` can create this state and says so. Nothing could report it
  afterwards, and a balance check cannot: **an unconfirmed charge looks
  identical to one that was never attempted**, so an agent asked "did that go
  through?" would read the old balance and call it settled. The route existed
  and had named the gap itself — *"an agent had no way to tell the user 'one of
  your top-ups needs you'"* — and was never given a tool.

  The **resume** route is deliberately not exposed. It returns a
  `client_secret`, a bearer credential that completes a charge; a tool response
  lands in a model's context and in audit records. The cardholder finishes the
  challenge in a browser from the link they already have.

- **`create_instance_snapshot`, `list_user_images`, `delete_user_image`** — the
  snapshot half of the sweep journey. Capture a working environment from a
  running instance, find it again, and delete it when the storage is no longer
  wanted.

  These exist because `create_image_sweep` shipped **unusable**: it requires an
  `image_id` and nothing on the tool surface could produce or find one. Its own
  input schema referred callers to `list_user_images`, which did not exist.

  `delete_user_image` is annotated destructive and previews before it acts —
  the record cannot be recovered through the API, and anything that referenced
  the image can no longer be launched from it. `create_instance_snapshot`
  refuses to overwrite an existing `name:tag` rather than replacing it, so it is
  safe to repeat.

  **Scope enforcement on the underlying routes was fixed in the same change.**
  `DELETE` and `PATCH /user-images/{image_id}` checked authentication and
  ownership but no scope, so a key issued with `instances:read` could delete or
  rename an image. They now require `instances:write`, and `GET /user-images`
  requires `instances:read`. No published annotation changed, so no version
  bump: this closes a hole rather than altering a promise.

- **`create_image_sweep` and `get_image_sweep`** — P7's sweep, reachable by an
  agent. Launch N instances from one snapshot as a single record whose members
  can be compared for environment drift, and read back which of them agree.

  `create_image_sweep` goes through prepare-approve-execute, not around it.
  Called without `plan_id` it quotes the sweep and returns a plan awaiting
  approval; nothing launches until it is called again with `plan_id` and
  `confirm:true`. **The member count is inside the approved arguments and bound
  by the plan's canonical hash**, so an approval for three members cannot be
  spent on sixty-four — which is what reusing the single-job launch-plan
  endpoint would have allowed, while still producing a record that consent was
  given.

  Not idempotent, and the description says so: each call without `plan_id`
  creates another plan awaiting approval.

- **PostHog MCP analytics.** The pinned `@posthog/mcp` beta captures standard
  MCP lifecycle/tool metadata for HTTP and STDIO, groups stateless calls by the
  authenticated principal, preserves the reviewed tool schemas, strips request
  and response content before send, and flushes on graceful shutdown.
- **Connector OAuth front door.** `WWW-Authenticate` now accompanies every 401,
  naming the protected-resource metadata, so a connector can discover how to
  authenticate. Client identification by CIMD or RFC 7591 dynamic registration,
  a consent screen, and port-agnostic loopback redirects.
- **Canonical resource identifier** is now `https://mcp.xcelsior.ca/mcp` (the
  exact URL a user pastes). Tokens bound to the previous origin
  `https://mcp.xcelsior.ca` remain valid until **2026-11-30**.
- **`search` and `fetch`** (ChatGPT company knowledge) over the documentation
  site, `llms.txt`, pricing, and marketplace listings. Off by default; enable
  with `XCELSIOR_MCP_COMPANY_KNOWLEDGE=1`.

### Changed

- **`configure_auto_topup` can now raise a cap, not only lower one.** Gate P1
  clause 6 requires approval to widen unattended spending, and the API enforces
  it: a widening from a non-human caller is refused **409**, naming
  `/api/v2/billing/auto-topup-plans`. The tool posted straight to
  `/api/v2/billing/auto-topup` and returned that refusal, so an agent could
  narrow auto-top-up and never widen it, while being told about a route it never
  called.

  It is two phases now, the same shape as `create_instance`. A refused widening
  comes back with `preview: true`, a `plan_id` and an `approval_url`; send the
  user there, then call again with `plan_id` and nothing else. **The settings are
  taken from the approved plan, not from the second call**, so an approval cannot
  be spent on different numbers — the tool posts an empty body on purpose.

  Lowering a cap or disabling stays a single call. Putting friction on the safe
  direction is how a control gets routed around.

  Only a 409 becomes a plan. A 500 still surfaces as an error rather than
  quietly turning into something the user is asked to approve.


- **`get_provider_account` now returns what is blocking payouts, not just that
  something is.** A new `payouts` block carries `charges_enabled`,
  `payouts_enabled`, `disabled_reason`, and Stripe's `currently_due` and
  `past_due` — the field names the provider still has to supply, such as
  `external_account` or `individual.id_number`.

  The tool's description already claimed this. It was false: `provider_accounts`
  has no such columns, and the code fetched those exact fields from Stripe, used
  them to derive a one-word `status`, and discarded them. So a provider asking
  why they had not been paid got `restricted` — which collapses "we need your
  bank account" and "we need a photo of your ID" into a word they cannot act on.

  **Read `checked_live` before trusting an empty list.** Stripe is not consulted
  when it is unconfigured or the account has no id, and the call can fail; the
  flag is what separates "nothing outstanding" from "nothing was asked".

  Requirement entries are field *names*, never values — no document, bank number
  or date of birth crosses the surface. The onboarding link is still not a tool:
  an AccountLink can change the payout destination, so it remains a browser
  action behind `providers:write`.

- **Trust-surface split.** The public connector serves the customer profile
  only. `drain_host`, `undrain_host`, `evict_host_workloads`,
  `retry_agent_command`, `get_scheduler_health`, `get_host_capacity`, and
  `list_reconciliation_findings` moved to the unlisted operator profile. They
  were never usable without operator scopes, so no credential loses access it
  could previously exercise.
- **`openWorldHint` corrected to `true`** on `list_available_gpus`,
  `get_spot_prices`, and `search_marketplace` — they read a live third-party
  marketplace. Annotation accuracy is now enforced against the contract at
  registration time.
- **Every tool description rewritten** to state when to use it and what it
  costs or changes. Descriptions are prose, not contract; no version changed.
- **Connector access tokens now live ~1 hour** (previously 15 minutes).
  Refresh tokens are unchanged at 30 days.
