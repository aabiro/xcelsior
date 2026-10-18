# Placement preference that refuses instead of settling

*Plan phase P5, the placement half. "Prefer verified hosts above 99.5% uptime
even at 15% more" — and say so plainly when no such host exists.*

## 0. Directive to the implementer

The prohibitions in `docs/artifact-promotion-plan.md` §0 apply. One more owns
this phase:

**Never satisfy a preference approximately.** If the user asked for 99.5% uptime
and the best available is 99.1%, that is a refusal with a number attached, not a
placement with an asterisk. The plan calls silent fallback *"the failure mode
that would quietly destroy trust"*, and it is right: a preference that sometimes
means nothing is worse than one that does not exist, because the user stops
checking.

## 1. Why this is the same bug twice

P3's mount-on-demand placement already faced this. Region could have been a
preference — "prefer the volume's region, fall back to anywhere" — and that
reading is *more available*, which is why it is tempting. It was made a hard
filter because falling back converts an unroutable promotion into a silently
expensive one, and the user sees neither.

Placement preference is the same shape at a larger scale. "Prefer 99.5% uptime"
that quietly places on a 97% host has not degraded gracefully; it has answered a
different question than the one asked.

The distinction this phase must keep is between a **preference over eligible
hosts** (rank them: cheapest first, or most reliable first) and a **constraint
that gates eligibility** (99.5% or nothing). Both are useful and they are not
interchangeable. The plan's own example contains both: *"above 99.5% uptime"* is
a constraint; *"even at 15% more"* is a bound on what the constraint may cost.

## 2. The shape

```
preference {
  min_uptime_pct     99.5      # constraint: gates eligibility
  min_tier           verified  # constraint
  max_premium_pct    15        # bound: how much more than the cheapest
                               # eligible host the user will pay to get it
}
```

Evaluated in one pass over the hosts a hard filter already admitted:

1. Compute the **baseline** — the cheapest host that satisfies the job's
   requirements, ignoring preference. This is what "15% more" is 15% more *than*,
   and computing it first is what makes the premium meaningful rather than
   relative to whatever happened to be picked.
2. Apply the constraints. If nothing survives, **refuse** with the best
   available figure quoted, so the user can decide whether to relax.
3. Among survivors, take the cheapest. If it exceeds `baseline × (1 + premium)`,
   **refuse** — the user set a price they would pay for reliability and this
   exceeds it.

Every refusal names the number that failed. "No host matched" is not an answer
anyone can act on; "the best available uptime is 99.1%, you asked for 99.5%" is.

## 3. Decisions

### 3.1 Uptime comes from `sla_monthly`, and a host with no history is not 100%

`sla_monthly` carries `total_seconds` and `downtime_seconds`. A host with no row
has no measured uptime — and treating "unmeasured" as "perfect" is how a brand
new host wins a reliability preference over one with a year of evidence.

**A host with no SLA history fails a `min_uptime_pct` constraint.** That is
harsh on new providers and it is the right default: the user asked for evidence
of reliability, and there is none. The provider surface (P6) is where a new host
earns history; it is not this phase's job to grant it by omission.

### 3.2 The premium is measured against the cheapest *eligible* host

Not against the cheapest host overall, and not against the chosen one. Against
the cheapest host that could have run the job at all. Any other baseline makes
"15% more" mean something the user did not say — most sharply if measured
against the chosen host, where the bound becomes vacuous.

### 3.3 What is recorded is what was true at placement time

Gate P5 asks that "the chosen host's reputation and SLA at time of placement are
recorded". Reputation moves. A trail that stores a host id and re-reads its score
later answers "what is this host's reputation now", which is a different question
and a useless one for an incident six weeks on.

So the numbers are **copied into the record**, not referenced.

### 3.4 Refusal is a typed outcome, not an exception

`PlacementRefused` carries which constraint failed and the best available value.
A caller — the launch route, the MCP tool — needs to render the trade-off, and
an exception string cannot be rendered into a choice.

## 4. Gate P5, clause by clause

| Clause | How it is met | How it is proven |
|---|---|---|
| A migrated job resumes from its checkpoint, proven by comparing state before and after | P3 promotion + snapshot/relaunch | **not in this document** — needs two live instances; see §6 |
| A placement preference that cannot be satisfied refuses clearly rather than silently falling back | constraints gate eligibility; refusal carries the failing number | ask for 99.9% where the best is 99.1%, assert the refusal names both |
| Preference is honoured in the audit trail: reputation and SLA at time of placement are recorded | values copied into the record | place, then change the host's score, and assert the record still shows what was true then |

## 5. What would make this the wrong design

- **If most users want "best effort".** Then refusing is hostile and the default
  should be ranking, with constraints opt-in. The plan's wording ("refuses
  clearly") says otherwise, and this follows the plan.
- **If SLA history is too sparse to gate on.** If most hosts have no
  `sla_monthly` row, §3.1 makes every preference refuse, and the feature is
  unusable rather than strict.

### 5.1 Measured against production, 2026-08-10 — and the risk is real

Checked against production (alembic 103) rather than the dev database, which
holds four rows and would have been reassuring for no reason:

| | |
|---|---|
| `sla_monthly` rows | **0** |
| hosts in fleet | 4 |
| `reputation_scores` rows | 14, all `entity_type = host`, all with a tier |
| tiers present | **`new_user`** only |

**`min_uptime_pct` cannot be offered yet.** With no SLA rows at all, every
uptime constraint refuses `insufficient_history` — correct behaviour reaching a
useless outcome. C2 must not expose it until aggregation is populating rows; a
preference that always refuses teaches users the feature is broken.

**`min_tier` cannot be offered either**, and the first version of this section
was wrong to call it "exposable but currently discriminating nothing". Every
host is `new_user`, which is threshold **0** — so the floor admits everyone and
any constraint above it refuses everyone. The control either does nothing or
always refuses, and *the first is worse*, because it looks like it worked. Same
C2 blocker, not a shippable half.

**`require_verified` is exposable today, and it is the one the gate's example
actually names.** `host_verifications` on production: **4 unverified, 2
verified, 1 deverified** — real spread, a populated state machine with its own
checker and recheck schedule. So C2 is **not blocked**; it ships with the
verification constraint and leaves uptime and tier unexposed.

### 5.1a Two vocabulary defects, on two different axes

The check found the tier vocabulary was **invented** —
`unverified`/`basic`/`verified`/`trusted`, which appear nowhere in the system.
Re-pointing it at `TIER_THRESHOLDS` fixed the vocabulary and **landed on the
wrong axis**: `verified` is not a reputation tier here, it is
`host_verifications.state`. So the plan's own example — *"prefer verified hosts
above 99.5% uptime even at 15% more"* — still could not be expressed. That is
what `require_verified` is for, and it is why one fix produced two.

The stored tier is also not trustworthy on its own:
`reputation_scores.tier` carries `server_default 'bronze'` while `final_score`
defaults to `0`, and `score_to_tier(0)` is `new_user`. A row inserted without an
explicit tier sits at **bronze on zero earned score** and would satisfy
`min_tier="bronze"` on no evidence — §3.1's argument in the reputation
dimension, fail-open, written into the schema. `host_tier()` therefore derives
from the score where one exists and falls back to the column only when there is
none. Production does not exhibit it today (all 14 rows are `new_user`, scores
0–50), but it is one INSERT away.

`reputation_scores` is keyed on `entity_id` with `entity_type` discriminating.
Only `host` rows exist today, so the projection **must** filter on it or a
user's reputation can be read as a host's.

### 5.3a Attempted 2026-08-27, and stopped by this section

The launch surface was built to the point of being one hop from enforcement —
migration 115 stores a preference in integer basis points with verified CHECKs,
`POST /instance` accepts one, and `upsert_job` persists it exactly. The next
commit would have been the scheduler gate.

**It was not written, because §5.1 and §5.2 say it must not be.** All four
constraints refuse on real data: no `sla_monthly` rows for `min_uptime_pct`,
every host at `new_user` for `min_tier`, and verified hosts 111 and 124 days past
a 7-day tolerance for `require_verified`. A control that refuses every request is
indistinguishable from a broken one, and wiring the gate would have shipped
exactly that.

Re-measurement was not possible — `xcelsior.ca` has been dark since 2026-08-23,
so the production numbers above are still the evidence. The **local** database
agrees and is worse: `host_verifications` holds 1 deverified and 11 unverified,
**zero verified**, and `sla_monthly` has a single row.

So the storage and transport hops are done and inert. The UI control carries a
`notApplied` banner rather than implying a constraint it cannot honour. What
remains is unchanged and in the order this document already set: **the
verification sweep first**, then the gate, then the banner comes off.

### 5.4 C2 cannot ship the preference beside the scheduler; it must reconcile it

**The silent fallback Gate P5 names is already shipped**, in the live ranking
path at `scheduler.py:5501`:

```python
verified_hosts = [h for h in hosts if h["host_id"] in verified_ids]
if verified_hosts:
    hosts = verified_hosts
# If no verified hosts, fall back to all (for cold-start)
except Exception:
    pass
```

So a correct preference module wired into that block would **still fail the gate
end to end**: the scheduler's existing behaviour when a verification constraint
cannot be met is to drop it. C2 must either let the preference decide
authoritatively for constrained requests, or skip this fallback for them. It
cannot sit beside it.

The bare `except: pass` is fixed now — it made "the verification store is
unreachable" indistinguishable from "every host is verified", since placement
proceeded across all hosts either way and said nothing. It logs. **The fallback
policy itself is left alone deliberately**: this is the unconstrained path every
job takes, and removing cold-start fallback changes where every job lands. That
is a decision, not a tidy-up.

**But logging is only right for the unconstrained path.** A request carrying
`require_verified` must **refuse** when verification cannot be read — on a
constrained request, proceeding is the same silent fallback one layer down, and
C1 would write "verified at placement" onto evidence nobody managed to read.
`choose_host` refuses `verification_unreadable` for exactly that case, and
places normally for unconstrained ones so the fix does not become a blanket
outage whenever the store hiccups.

### 5.5 "Verified" is a stamp, not a current fact — and two subsystems disagree

`VERIFICATION_THRESHOLDS["reverify_interval_sec"]` is 86400 and
`list_hosts_needing_reverification()` implements the due query. When this
section was written its only wrapper, `get_hosts_needing_reverification()`, had
**no callers** — no worker, no scheduler pass, no timer, no route.
`next_check_at` was written and read only inside the function nothing invoked.

**Fixed in §5.2c**: `verification_sweep.run_sweep()` runs hourly from
`bg_worker.py` and asks overdue hosts to prove themselves again. Everything
below still holds, because the sweep can only *ask* — a host that is offline,
busy, or running an agent too old to know the command never answers, and its
stamp goes on ageing exactly as described.

On production, both verified hosts are overdue:

| host | overdue by |
|---|---|
| `test-curl` | **123.9 days** |
| `3736c239-d1e` | **111.2 days** |

That is the third instance of one shape on this page — *measured, but not
recently enough to be evidence* — so it gets the same treatment:
`verification_stale` as its own refusal code, distinct from
`verification_unsatisfiable`, because "no verified hosts exist" is answered by
verifying one and "the checker is not running" is answered by starting the
sweep.

**`require_verified` is unshippable today by its own numbers.** The tolerance
derives to 7 days; prod's two verified hosts are 111 and 124 days overdue. So
the control currently refuses **every** request on real data — the exact
condition that disqualified `min_uptime_pct` in §5.1. Saying "C2 owes both the
gate and a sweep" understated it:

> **The sweep is C2's first commit, before the launch surface.**

One pass makes the control live. Until then it ships dead and teaches users the
same lesson `min_uptime_pct` would have — a control with a 0% satisfaction rate
is indistinguishable from a broken one.

**The gate and the sweep disagreed about NULL, and hosts fell in the gap.** The
gate treats a missing `last_check_at` as stale — refused, correctly — while the
due query required `next_check_at IS NOT NULL`, so a verified host with no
schedule was invisible to the sweep forever: permanently refused by the gate and
permanently unreachable by the only thing that would fix it. A row stranded in
`verifying` had no path at all, since the predicate looked only at `verified`.
Both are fixed — NULL is due, `verifying` is included — so the two defaults now
point the same way: an unstamped host is neither trusted nor ignored.

Prod today: 4 `unverified` rows carry NULL on both timestamps, and the 2
`verified` rows carry stamps. The deadlock is latent rather than active, and it
would have arrived the moment the sweep was wired.

Two subsystems also mean different things by the word. `host_attestation.
validate_attestation` accepts `nras_verify_status in ("verified", "pending")`
for its attested tier — a looser rule on a different axis. C2's evidence must
say *which* verification it copied.

### 5.2 What C1 owes, from the same review

- **The aggregation rule — now settled, because the arithmetic settled it.**
  `MIN_OBSERVATION_SECONDS` quietly answered this: 30 days is 2,592,000s, and a
  February row's maximum is 2,419,200s, so no February row could ever satisfy
  it and no current month could until day 30. The threshold was only reachable
  by **summing across rows**. So the rule is stated rather than implied: the
  projection sums `total_seconds` and `downtime_seconds` over a **trailing
  `OBSERVATION_WINDOW_DAYS`**, and the minimum is a floor on that sum. A
  per-month reading would also make `min_uptime_pct` lurch every 1st, and a
  preference that passes on the 31st and refuses on the 1st is one nobody can
  rely on.
- **The record is append-only.** Copied evidence is worth exactly what it costs
  to rewrite. **Done**: migration 105 `placement_decisions`, WORM by trigger
  (precedent `075`, `072`), with `tests/test_placement_record_is_worm.py`
  probing a real UPDATE and a real DELETE rather than describing the property.
  Refusals are recorded alongside placements — a preference that refused was
  honoured *by* the refusal, and a successes-only table could not answer "why
  did nothing launch last Tuesday". Prices are integer micros, so the premium is
  recomputed from two exact integers rather than stored as a rounded percentage
  nobody can check.
- **What a refusal may disclose — decided.** `best_available` returns a fleet
  aggregate to anyone who probes, and a loop over varying constraints maps the
  distribution. **Accepted for the refusal payload**: every candidate is a host
  the caller could already launch on and whose ask is already public in the
  offer listing, so the aggregate discloses nothing the marketplace does not.
  **Not accepted for the stored record**: `placement_decisions.candidates`
  holds other hosts' prices and states because a refusal is unreadable without
  the field it refused over, which makes the row *internal*. No route reads this
  table; the tenant-facing view has to project down to the chosen host plus
  aggregates, and that projection is part of C2's surface. `reputation_score` in
  tenant-facing evidence stays open with it — `tier` may be the right
  granularity there, and C2 is where it is answered.
- **The projection query is C1's first commit, before anything else.** C0's
  input dict is assembled by nobody yet, so **every field name in it is an
  assumption of exactly the class the tier one was**:
  `verification_state`, `verified_at`, `sla_total_seconds`,
  `reputation_score`. Two survive already —
  `price_cents_per_hour` and `ask_cents_per_hour` are both real columns — but
  the rest are unverified until a query returns them. That query will falsify
  more of C0 than any test in this repository can, which is the generalisation
  of every finding on this page.

### 5.2a The projection, and the four things it falsified

`control_plane/scheduler/host_projection.py`, with
`tests/test_placement_evidence_projection.py` running against a database
migrated to head. The prediction above held: the query falsified more of C0 than
any dict fixture could.

1. **"Two survive already" was itself wrong.** `ask_cents_per_hour` is a real
   `gpu_offers` column, but the host dicts `scheduler.allocate_best_host` ranks
   carry **`cost_per_hour`, in dollars** — which `usable_price` reads as neither
   of its two keys. Against a real candidate list every request would have
   refused `no_priced_hosts`, and a list mixing both shapes would have computed
   the premium **wrong by 100×**. `normalise_price_cents` converts at the
   boundary so `usable_price` sees one key in one unit.
2. **`sla_monthly` inflates uptime for most of every month.** `sla.py` writes
   `total_seconds = days_in_month * 86400` — the *whole* calendar month — while
   capping downtime at `now`. On the 10th of a 31-day month a host that was down
   for a day reads **96.8% instead of 90%**: unelapsed time counted as observed
   uptime, against a gate whose entire job is `min_uptime_pct`. The projection
   clamps each month to elapsed time, and clamps downtime to that.
3. **`entity_type = 'host'` is necessary and not sufficient.** `reputation.py`
   defaults the column to `"host"` at every layer, so production holds 14 rows
   all typed `host`, seven of which are not — six `user-*` ids and one `cust-*`.
   The predicate stays, because a correctly-typed user row must never be read as
   a host's; what actually keeps users out of a placement today is the caller's
   shortlist. **Fixing the writer is an open reputation-module defect.**
4. **A missing key is now an error.** Every absent field fails closed in
   `choose_host`, so one wrong name produces a gate that refuses everything —
   indistinguishable from the gate working. `assert_evidence_shape` raises
   instead, and requires `verification_unavailable` to be a real boolean, since
   it is the one fail-*open* field and a null would disable the unread-evidence
   refusal silently.

Reputation and SLA read failures **propagate**; only the verification read is
caught. Verification has a stated fail-open flag the gate consults; the other
two have none, and swallowing them would yield zeros that read as "no history"
and "no tier" — a universally-refusing gate that looks exactly like a working
one.

### 5.2b Production replayed through it, 2026-08-10

Production runs neither `preference.py` nor `host_projection.py` yet, so its
rows were pulled read-only and replayed through the real modules locally.
Every candidate in the marketplace is a host holding a live offer — three of
them, all at 20 cents/hour:

| preference | answer |
|---|---|
| none | places on the cheapest, +0.0% |
| `require_verified` | **refused** `verification_unsatisfiable` — all three are `unverified` |
| `min_uptime_pct=99.5` | **refused** `insufficient_history` — longest window 0.0 days of 30 required |
| `min_tier=bronze` | **refused** `tier_unsatisfiable` — best available `new_user` |

Four different refusals, four different next steps, none of them a false refusal
caused by a name. That is what this exercise was for.

Two further facts the replay makes plain:

- **The two verified hosts hold no offers.** `require_verified` is unsatisfiable
  today even if the sweep ran, because verified capacity is not sellable
  capacity. They were last checked **112 and 125 days** ago against a one-day
  interval — §5.5, aged by another day.
- **The stored tier and the derived tier agree everywhere**: every host is
  `new_user` at `final_score` 50 against a bronze threshold of 100. The
  `server_default 'bronze'` that motivates deriving has not been hit by a real
  row yet, and is still in the schema.

### 5.2c C2's first commit: the sweep, and why the server cannot just re-verify

`list_hosts_needing_reverification()` has existed and worked for months; its only
wrapper had **no callers**. `next_check_at` was written and never read, so
nothing ever moved a host out of `verified` and production's two verified hosts
drifted 112 and 125 days past a one-day interval. Shipping `require_verified`
over a fact nothing maintains would teach users the feature is broken.

**The server cannot re-verify on its own.** `run_verification(host_id, report)`
needs a fresh telemetry report — GPU model, VRAM, driver, PCIe bandwidth,
temperature, network loss and jitter — that only the host can produce. The agent
submits one **at startup and never again**, which is the entire explanation for
the 112-day stamps: those agents have not restarted. So the sweep *asks*: it
enqueues a `reverify` command on the existing agent channel, and the agent
re-runs the same builder startup uses and POSTs to the same endpoint. One
builder, two callers — two builders would let a scheduled report differ in shape
from a startup report, and the checks would then disagree for reasons nobody
could see.

Three guards, each for a failure the naive version would cause:

- **A host already holding a live request is not asked again.** Asking twice
  runs a 60-second GPU benchmark twice, on hardware someone is paying for.
- **One pass is capped.** The first sweep after this ships sees every overdue
  host at once; a hundred simultaneous benchmarks is a self-inflicted load
  spike. Hosts that miss the cut are still overdue next hour.
- **The agent defers while it has a running container.** A benchmark that steals
  the GPU from a paying job would corrupt that job's timings to refresh a stamp.
  A permanently busy host therefore stays `stale`, which is the honest answer.

**It deliberately does not expire stamps.** Moving an overdue host out of
`verified` would change `scheduler.allocate_best_host`'s preferred set and
therefore where every job lands — the §5.4 reconciliation, which is a decision
and not hygiene. The gate already reads an overdue stamp as `stale`, so a
`require_verified` request is answered correctly either way.

**Rollout ordering.** Agents hard-refuse unknown commands by design, so a fleet
that has not taken the handler refuses these — loudly, once per host per sweep,
self-correcting on upgrade. Deploy, push `upgrade_agent`, then watch `stale`
clear. A flag defaulting off would be a way to ship something that never runs.

### 5.2d Two defects in `105`, found by review, and one open question

**1. WORM without pruning.** `105`'s docstring claimed WORM "like `075` and
`072`" and took the trigger from both and the partitioning from neither.
`072_audit_events_v2.py` is explicit about why that matters — *"partition drops
(retention) are DDL and are unaffected"* — which is the whole mechanism by which
an append-only table stays prunable. `075` needs none of it, being one row per
signed checkpoint; `105` is **per request** and copied the low-volume precedent,
giving a table that grows without bound and whose own trigger forbids the only
statement that could shrink it.

`106` rebuilds it partitioned by month. A partitioned table's key must contain
the partition key, so `decision_id` alone became `(decision_id, decided_at)` —
an ALTER cannot do that. Done before the launch surface writes, because
afterwards it is a migration on data that cannot be DELETEd.

The first draft of `106` *refused* on a non-empty table. That was over-cautious
in a way that would have bricked every database that had ever run the WORM
tests, which deliberately leave rows behind. The move is safe for exactly the
reason WORM works: it blocks UPDATE and DELETE **on rows**, while `INSERT …
SELECT` and `DROP TABLE` are untouched. So the rows are carried across — 63 of
them on the test database, verified.

Partition maintenance is now table-driven (`PARTITIONED_TABLES`) rather than
copied. A maintainer duplicated per table is how one silently stops advancing
while the other looks fine: the DEFAULT partition absorbs the writes and nothing
complains until someone tries to prune.

**2. Two selection policies behind one call.** `allocate_best_host` ranks by
compute efficiency weighted by reputation; `choose_host` took `survivors[0]` in
price order. Routing constrained requests through the second would not only have
skipped the fallback — it would have **changed the ranking** for those requests.
A user who states `require_verified` and nothing else has asked for
verification; answering with the cheapest verified host is answering a question
they did not ask, which is the objection `premium_exceeded` already raises about
placing over a stated bound.

The ranker now lives in `control_plane/scheduler/ranking.py` and neither path
owns it. The constraints decide **who is eligible**; the ranker decides **which
of them**. The premium is still measured cheapest-eligible → chosen, so it still
reads as "what your constraint cost you". A premium bound filters *before* the
ranking, so a survivor inside the cap is never passed over for one outside it,
and the refusal still quotes the cheapest survivor's overage — the least the
user would have to relax by.

Every existing preference test passed unchanged after this, which is itself the
finding: **a fleet where cheapest and most-efficient agree cannot tell the two
policies apart.** `tests/test_placement_ranks_through_one_function.py` uses a
fleet where they disagree, and asserts the fixture still discriminates.

**3. Frequency does not belong on the WORM row.** Recording *every* evaluation is
the right write policy — a preference that refused was honoured by the refusal —
but a caller polling a preference decides the same thing repeatedly, and each row
carries a `candidates` snapshot that scales with the fleet. That is a fleet
snapshot per poll.

A `times_seen` column on `placement_decisions` is **unimplementable**, because
WORM forbids UPDATE. That constraint is correct rather than an obstacle to work
around: "what was decided" is immutable evidence, "how often we were asked" is
operational telemetry with a natural retention policy. Conflating them is what
created the growth problem.

So migration `107` splits them. `placement_decisions` holds one row per **distinct
decision** — fingerprinted over tenant, job, what was asked, what was answered,
and the *sorted candidate states*, because a refusal is only interpretable
against the field it refused over and two refusals sharing a code over different
fleets are different facts. `placement_decision_observations` holds the count, in
a plain table that can be updated and pruned.

Keyed by calendar month, matching the partition boundary. Without that, an
identical decision next March would collapse into a row timestamped today and the
trail would say March never happened. `ON CONFLICT` does the dedupe, so two
concurrent identical evaluations produce one row rather than two. The observation
is written *before* the decision, so `decision_id IS NULL` is the honest record of
"seen, and failed to write down" — with an index for finding exactly those.

**Open — needs a decision, not a claim.** `placement_decisions` is tenant-owned
and sits outside the erasure path: `privacy_sinks.py` names none of it, the
trigger blocks DELETE unconditionally, and partitioning cannot drop one tenant's
rows. This is **not new to `105`** — `audit_events_v2` and `audit_checkpoints`
are in exactly the same position, and a grep for "legal basis", "legitimate
interest" or "right to erasure" finds nothing anywhere in the repository. Audit
tables legitimately resolve this by pseudonymisation or a documented exemption.
Which one applies here is a decision for the owner; recording it is the point.

### 5.3a C3's design, and two findings that change what it can claim

**1. "Re-run attestation on the target" cannot mean what it sounds like.**
`host_attestation.validate_attestation` is a **shape check** — it asserts that an
`attested` tier carries a `tee_evidence_jwt` and a verify status of
`verified`/`pending`. Its own docstring says it *"does not verify JWT crypto"*.
It runs over stored data that does not change, so re-running it at migration
returns the same answer it returned at admission, forever.

And `attested_at` — the timestamp that would make freshness checkable — **has no
readers anywhere in the codebase**. Written into the normalised dict, never
consulted. That is `next_check_at` again, in a third subsystem.

So attestation has the same staleness hole verification had. **It is not fixed
here**, and that is deliberate: the requirement is that migration be *no weaker*
than launch, not that it be stronger. Adding a freshness gate at migration only
would reject a host that a launch would happily accept — asymmetric, and it
would paper over the real defect, which is at admission. Recorded as owed.

**2. There are no checkpoint encryption keys to carry across the move.** §5.3
asks where they live, which presumes they exist. They do not: `StorageConfig`
has no SSE, no KMS, no encryption fields of any kind, and `CHECKPOINT` is simply
an artifact kind alongside logs and datasets. `user_encryption_keys` exists but
is the privacy crypto-shred key for user data, not artifact encryption.

The honest statement is therefore: a migration moves artifacts that are **not
encrypted at rest by this codebase**. Confidentiality rests on bucket access
control, presigned-URL expiry, and the NFS export boundary — none of which move
with the job, so none of which needs custody transfer. **If per-tenant checkpoint
encryption is added later, this is the paragraph that has to be rewritten before
the move path is touched.**

**3. What the migration gate actually does.** It re-runs *the launch gate
itself* — `filter_hosts`, the same Stage-C filter `simulate_placement` uses — on
the target, and then re-evaluates the same preference through
`evaluate_preference`. Not a parallel reimplementation: the ranker lesson
applies here too, and a second admission gate would drift from the first
silently. This is real rather than decorative because `administrative_state`
genuinely changes — a host can be drained, disabled, or go stale between launch
and migration.

The rule it enforces: **"migrated to cheaper" must never become a path onto a
host that would have failed the gate at launch.** A migration that bypasses
admission is a way to reach an unvetted host without ever asking for one.

**What remains blocked.** Clause 1 — *"a migrated job resumes from its
checkpoint, proven by comparing state before and after"* — needs two live
instances that can share a volume, and the available hosts are network-isolated
from each other. The gate above is testable against the database and is; the
resume proof is not, and is left FAIL rather than softened.

**4. Nothing persists a migration outcome, so the timeline clause has no
source.** Found 2026-08-27 while checking whether P5's frontend clause —
*"migration history on the instance timeline: what moved, when, why, and what it
saved"* — was buildable.

`MigrationOutcome` is a frozen dataclass with an `as_dict()`, returned by
`migrate_job` and **discarded by the caller**. There is no table, no column, and
no audit row. Every field the clause asks for is computed and then thrown away:
`source_host_id`/`target_host_id` is *what moved*, `refusal` and the preference
that drove the choice is *why*, and the price delta is *what it saved*.

So the frontend clause is blocked twice over, and the second block outlives the
first. Wiring C3 a caller — the part that needs two live instances sharing a
volume — would still leave the timeline with nothing to read. **A migration
history table is C3's first commit, before the executor gets a caller**, for the
same reason C2's verification sweep comes before C2's launch surface: the record
has to exist at the moment the first migration happens, or the first migration
is the one that goes unrecorded.

Not built here. Storage whose only writer is also unwritten is a schema designed
against a guess about what that writer will produce, and `MigrationOutcome`'s
own shape is the thing most likely to move when C3 acquires a real caller.

What is *not* a reason to defer it: "there is no data yet". An empty migration
timeline is honest and harmless — unlike C2's constraint gate, which would have
refused every request. The reason is the schema, not the emptiness.

### 5.3 What C3 owes

The move re-runs **admission and attestation on the target**, and re-evaluates
the same preference, before anything migrates. "Migrated to cheaper" must never
become a path onto a host that would have failed the gate at launch — a
migration that bypasses admission is a way to reach an unvetted host without
ever asking for one. Where the checkpoint's encryption keys live across the move
has to be stated before that code is written, not discovered during it.

## 6. Sequence

* **C0 — the preference, evaluated and refused.** Pure function over hosts:
  baseline, constraints, premium bound, typed refusal. No route, no storage.
* **C1 — the record.** Reputation and SLA copied at placement time.
* **C2 — the launch surface.** Preference as an input to placement simulation
  and launch, with the trade-off rendered before the user commits.
* **C3 — migration.** Snapshot → stop → relaunch cheaper → verify, using P3 for
  the checkpoint. This is the half needing live instances, and it is last
  deliberately: the placement half is useful alone, and migration without it
  would move jobs onto hosts nobody vetted.
