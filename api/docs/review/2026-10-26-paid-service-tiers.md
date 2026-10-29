# Paid service tiers: price and placement contract

Standard uses its existing Stripe compute Price. Premium adds 60% of Standard;
Dedicated adds 180% and reserves every GPU on one physical host. GPU-only
exclusivity is an optional addition of 40% of Standard on Standard or Premium.
Dedicated already includes GPU exclusivity and never adds that fee again.

The additions apply to compute, per GPU-hour. They do not multiply storage or
tax. A two-GPU Dedicated host is charged for both GPUs, including an idle GPU.
Standard and Premium GPU-only exclusivity allow other customers on different
GPUs of the same host. Dedicated excludes other managed customer workloads from
the physical host. This is an allocation contract, not a regulatory certification.

## Stripe catalog

Through the official Stripe MCP endpoint, 945 new live Prices were created on
the existing GPU Products: 315 Premium, 315 Dedicated, 315 GPU-exclusivity.
The existing 315 Standard Prices remain the base. Old Price IDs were not
deleted or archived and existing subscriptions and payments were not changed.
The bidding extension adds 63 CAD 0.01/GPU-hour bid-unit Prices on the same
GPU Products. Final readback produced a plan with **1,008 matching tier/add-on
Prices and zero to create**, plus the 315 reused Standard Prices.

`billing/stripe_catalog.json` contains the 1,323 exact returned Price objects,
their GPU variants and pricing modes, and a version digest. The sync script
separates network provisioning from database import. Runtime quotes and
placement never contact Stripe. Migration 127 adds immutable integer-micro-CAD
Price snapshots, typed job Price identity and host isolation scope, and attempt
meter/cycle pricing snapshots. The shared catalog belongs to the platform
tenant, `xcelsior-platform`, and is classified in the billing database domain.

For the live RTX 4090 base variant, Standard is CAD 0.49/GPU-hour, Premium
0.784, Dedicated 1.372 and the optional exclusivity addition 0.196. These amounts
were read back from Stripe. A two-GPU Dedicated host is 2.744/hour before tax.
Premium plus GPU-only exclusivity is 0.98/GPU-hour before tax.

## Quote to wallet to scheduler

The modal fetches all tier amounts for the selected GPU, VRAM and pricing mode.
Changing tier, GPU count, add-on or selected volumes updates the displayed total
immediately. Compute, tier, exclusivity, storage and estimated account tax have
separate lines. An unavailable Price blocks launch. PyTorch is first and the
default image; Custom is last, and the top team card is removed.

Both the main and add-on Stripe Price IDs accompany submission. The server
resolves the quote again and refuses a changed ID. Launch-plan approval binds
these identities into the canonical spec and pricing version. Execute returns
a fresh plan after a superseded Price rather than silently substituting it.
The customer must fund the exact compute quote before queue admission.
Billing renews expired queue holds against the accepted price without changing
FIFO timestamps. Insufficient funds record `funding_required` and back off;
the scheduler only reads the resulting financial evidence.

Queue class starts from the purchased tier: Dedicated 3, Premium 2, Standard 1.
Within a purchased class, order is original bid plus free aging credit, then
FIFO. Every completed 15-minute wait interval earns 10% of the original bid in
credit. Bid plus credit can qualify for one higher queue class using the pinned
Stripe tier-price gap. Promoted jobs join a FIFO section behind all customers
who purchased that class; they cannot bid past those customers. Promotion
changes neither the accepted price nor physical isolation. A database-owned
queue-entry timestamp prevents running time from counting toward a later wait.
`queue_bid_events` logs every accepted bid (including zero-bid baselines) and
successful claim transaction, with source Prices, money, aging and ordering
snapshots. See [the exact policy](../queue-bidding-policy.md).

Caller-supplied priority values and old fair-share weights cannot change this
order. Expired or inadequate holds cannot claim paid work. Reservation
revalidates immutable Prices and funding, checks host qualification and GPU
variant, and atomically allocates devices. Migration 128 adds typed bid values,
the queue-entry clock and durable ordering evidence; no financial history is
repriced.

Dedicated requires every physical GPU and an unoccupied, qualified host. A host
row lock serializes shared and Dedicated reservation attempts. Each lease
returns the exact allocated GPU UUIDs. The worker binds Docker to those UUIDs;
a missing v2 allocation cannot fall back to all devices. Purchased exclusivity
also requires readable physical inventory, readable occupancy, idle allocated
GPUs and per-device locks. It refuses occupied hardware without killing other
workloads.

The worker reports physical UUIDs, canonical GPU models and per-device VRAM in
the existing fenced observation protocol. The API records the observation and
projects count/capabilities without granting admission. Inventory sync upgrades
synthetic slots under the same host/device lock order as reservation; it defers
that upgrade while synthetic allocations are active. Paid reservation refuses
synthetic identities. An unreadable driver report disables paid GPU binding.
The worker self-update bundle includes the new isolation module and GPU-aware
security module, as do both installer and manual deployment paths.

Attempt billing freezes both Stripe identities and the exact hourly amount.
Metering and wallet cycles retain that snapshot after catalog retirement or
host-price changes. Paid usage rounds once to integer micro-CAD. The wallet
debit, ledger transaction, Stripe meter outbox entry and billing cycle now commit
together; a cycle failure rolls back the debit.
Invoice lines retain the purchased service tier and both Stripe Price IDs;
GPU-hour quantities count every allocated GPU. The Stripe usage outbox uses
the measured GPU-hours rather than inferring an hour from a log description.

## Qualification and rollout boundaries

New worker distributions include the signed hardware-isolation module in the
installer, API allowlist and Nginx allowlist. Providers must explicitly qualify
hardware with `XCELSIOR_MANAGED_HARDWARE_ONLY=true`. Full-host availability also
requires `XCELSIOR_DEDICATED_HOST_ENABLED=true`. Both default false. The platform
cannot treat an arbitrary mixed-use provider host as isolated; operator
qualification and live hardware acceptance are required.

No production database migration, API deployment or worker deployment was
performed here. The deploy script imports the versioned Stripe catalog after
Alembic and before API activation. Update workers and qualify hosts before
offering live isolation capacity. Existing repair/report-only settings are
unchanged. Historical unpriced billing records keep their captured behavior;
new customer paid launches require transactional scheduling and cannot fall
back to the retired placement path.

## Verification

Tests ran against a disposable PostgreSQL 16 cluster, not the configured shared
or production database. Named gates:

- `tests/test_paid_service_tiers.py`: exact additive prices, stale Price IDs,
  REST quote/submission/hold agreement, paid queue order, insufficient/expired
  funding, whole-host exclusion, GPU-only sharing, immutable snapshots, exact
  wallet debit and transaction rollback.
- `tests/test_hardware_isolation.py`: physical scope, idle checks, qualification,
  per-device locking, release on failure and exact Docker UUID binding.
- `tests/test_launch_plan_*.py`, `tests/test_launch_instance_adapter.py`, queue,
  reservation, lease and worker-v2 tests: approval/idempotency and existing
  transactional lifecycle contracts.
- `tests/test_from_empty_bootstrap.py`, migration ledger/lock, service-role,
  companion schema, no-runtime-DDL, generated OpenAPI/inventory and static
  installer gates. Migrations 127 and 128 were also downgraded and reapplied on
  the disposable cluster before final validation.
- Frontend modal/API tests cover immediate totals, missing prices, image order,
  team banner removal, full-host GPU count and launch payload. TypeScript and
  focused lint checks cover the frontend and MCP changes. `list_tiers` exposes
  the selectable Stripe catalog and `create_instance` accepts paid tier and
  exclusivity Price identities.

Live GPU execution, full-host isolation acceptance and payment settlement remain
unverified. A headless Chrome check of the isolated real modal component, with
real private-API Stripe catalog calls and stubbed auth/locale/router contexts,
verified image order, instant Standard/Premium/GPU-only/bid/whole-host totals and a 390px
viewport without horizontal overflow. Screenshots and browser-check.json are in
artifacts/service-tier-pricing-20261026. The complete local Next dashboard
returned a blank page with a next/dynamic server-render bailout, so a full
dashboard journey is not claimed. The older `tests/test_agent_v2.py` suite also contains existing
sessionless ACK and retired-ingress expectations: ACKs omit the currently
required worker session/claim identity and retired v1 routes return 410. Those
failures were recorded separately; production protocol requirements were not
weakened to make historical tests pass.

Final focused verification recorded 385 passing backend tests, 207 passing
schema/generated/installer tests (one pre-existing historical skip), 18 passing
frontend tests and 127 passing MCP contract tests. New queue boundary,
property, native-tier protection and financial rollback tests are included.

The final concurrency check also proves that a second scheduler cannot use
SKIP LOCKED to pass a locked paying Premium job with a promoted Standard job.
Capacity backoff does not block otherwise usable hardware. The policy, funded
billing and claim suite passed 73 tests after this protection was added.
