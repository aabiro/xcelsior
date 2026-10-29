# Paid queue bids and bounded aging

Policy version: `tier-bid-aging-v1`.

The purchased service tier determines the initial queue class and hardware
isolation. Standard is class 1, Premium class 2, and Dedicated class 3. Dedicated
reserves the entire physical host. Queue promotion never changes that scope.

An optional bid is an additional CAD amount per allocated GPU-hour. Stripe owns
the unit price: `queue-bid:all` Prices on each GPU Product currently price one
unit at CAD 0.01/GPU-hour. The accepted quote pins that Price ID, its integer
micro-CAD amount and the customer's integer unit quantity. The bid is included
in the launch hold and running compute rate. Waiting and aging have no charge.

## Exact ordering

Use database time and the current queue-entry timestamp, `jobs.queue_entered_at`.
All candidates in one claim use the same PostgreSQL statement timestamp; audit
records retain microsecond wait precision.
That timestamp survives ordinary updates and hold renewal. A transition back
into queued state starts a new waiting interval; running time earns no credit.
Historical unpriced jobs retain their existing FIFO timestamp fallback.

Let `b` be the original bid in micro-CAD per GPU-hour, `w` the current wait in
seconds, and `g` the price difference between the purchased tier and the next
tier for the same GPU variant and pricing mode. Both tier Prices are pinned at
acceptance. Exclude tax, storage and GPU-exclusivity fees from this difference.

```
steps  = floor(max(w, 0) / 900)
credit = floor(steps * b / 10)
promoted = b > 0 and next_tier_exists and b + credit >= g
effective_class = purchased_class + (1 if promoted else 0)
```

Credit is linear in the original bid, with one rounding to integer micro-CAD;
it is not compounded. A zero bid gets zero credit. Standard can reach Premium
queue class, Premium can reach Dedicated queue class, and Dedicated stays in
class 3. No job can skip two classes.

Within each effective class, customers who **purchased that class** come first.
Their order is bid plus aging credit descending, then FIFO, then job ID for a
stable final tie-break. Promoted jobs follow in a separate FIFO section. Their
order is the time they qualified for promotion, then original queue entry and
job ID. Their bids cannot overtake customers who purchased the higher class.
An in-flight paid claim for overlapping GPU/region capacity also blocks a
promoted claim: another replica cannot use `SKIP LOCKED` to pass the protected
customer. A paid job in capacity backoff does not block otherwise usable capacity.

The qualification time is derived exactly rather than depending on tick timing:

```
promotion_steps = ceil(max(g - b, 0) * 10 / b)  # positive bids only
qualified_at = queue_entered_at + promotion_steps * 900 seconds
```

For the current RTX 4090 base Price, Standard is CAD 0.49 and Premium is 0.784,
so the gap is 0.294/GPU-hour. A 0.20 bid earns 0.02 of credit each 15 minutes:
after 60 minutes its bid plus credit is 0.28; after 75 minutes it is 0.30 and
qualifies for the back of Premium. A 0.30 bid qualifies immediately, still
behind all customers who purchased Premium. The accepted charge stays 0.69 or
0.79/GPU-hour respectively; free credit is never added to it.

## Funding and evidence

An active wallet hold must cover the complete accepted compute rate before a
paid job is claimable. Reservation revalidates the immutable Prices, funded
amount, GPU count and purchased isolation scope. Bid and aging cannot override
host admission, GPU availability, lease fencing or device UUID binding.

`queue_bid_events` records every accepted launch, including zero-bid baselines,
and every successful scheduler claim in the same database transaction as the
corresponding mutation. An audit failure refuses admission or rolls back the
claim. Events retain tenant, job, purchased/effective class, original bid units
and Stripe ID, elapsed wait, aging steps/credit, promotion qualification time,
policy version and full accepted price snapshot. Reclaimed attempts have a
separate claim number. They can be joined to attempts and billing cycles by job
ID to compare waiting, actual allocation and charges before considering a
future cross-tier auction. No cross-tier auction is enabled by this policy.

Named gates: `tests/test_queue_bid_policy.py`,
`tests/test_paid_service_tiers.py`, queue claim/reservation/lease tests, frontend
launch-modal tests, and MCP `paid-queue-bids.test.ts`.
