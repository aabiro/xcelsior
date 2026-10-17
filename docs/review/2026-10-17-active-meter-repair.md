# Active-meter repair and settlement — 2026-10-17

This increment implements the missing active-meter remedy in B2.9 through the
B3.3 billing authority. B2.9 remains partial: remaining lifecycle remedies,
production false-positive measurement and real worker recovery drills are open.
No production flags were enabled and no push or deployment was performed.

The billing maintenance task can open an attempt meter only with current
job/attempt/host/fence, running intent/status, server start, a valid lease and a
fresh exact running observation. Missing owner or explicit price refuses repair;
spot usage requires its locked rate and GPU count. The job controller retains
its reporting role. The maintenance task keeps its existing API database
identity; scheduler/reconciler grants were not widened.

Open meters store NULL completion and zero duration/cost. They do not debit
wallets or produce invoice lines. Normal completion and orphan repair now share
terminal settlement: the same meter closes at its captured rate and the bound
attempt's terminal timestamp. Missing/reversed/future timestamps or conflicting
binding refuse settlement. Zero prices stay zero; no current-time/current-price
fallback manufactures evidence. Provider payout refuses any open usage meter.

Missing-terminal repair now participates in the caller's transaction, preserving
locked spot/GPU fields and legacy preemptible pricing indicators. It previously
opened and committed a second database connection and discarded pricing fields.
All enforced billing repairs now leave findings/action evidence even when they
were not first encountered in report-only mode. Refused repairs remain visible.
The existing orphan fixture now supplies its actual host binding: an unbound
historical meter is no longer accepted as sufficient settlement evidence.

Evidence was collected with Python 3.12 and fresh local PostgreSQL clusters,
fully migrated through 123. The runner disables dotenv and inherits no service
credentials. It uses Unix sockets, stops/removes each cluster, and retains
logs, JUnit and source-hash manifests under these local paths:

| Run | Result | Evidence directory |
| --- | --- | --- |
| Before implementation | 7 failures, zero skips | `/tmp/xcelsior-active-meter-red-20261017` |
| Expanded billing/invoice/provider gates | 123 passed, zero skips | `/tmp/xcelsior-active-meter-expanded-20261017` |
| Broader control-plane/billing regression | 434 passed, zero skips | `/tmp/xcelsior-active-meter-regression-20261017` |
| Separate API login and runtime configuration | 168 passed, zero skips | `/tmp/xcelsior-active-meter-runtime-20261017` |
| Final combined gate | 436 passed, zero skips | `/tmp/xcelsior-active-meter-final-20261017` |

The final combined manifest records the complete 25-suite invocation and hashes.
The new tests exercise concurrent repair/completion, transaction rollback,
job/attempt/host/lease lock contention, stale/mismatched/missing observations,
duplicate workloads, stopped intent, closed-meter conflicts, owner/rate refusals,
zero-rate preservation, locked multi-GPU spot pricing, invoice exclusion,
terminal evidence refusal, and finding convergence. The runtime proof uses an
actual separate `xcelsior_api` login to open and settle a meter and verifies
scheduler/reconciler INSERT privileges remain absent.

A focused runnable gate is:

```sh
python scripts/run_security_gates.py --output /tmp/xcelsior-active-meter-proof \
  tests/test_active_meter_repair.py \
  tests/test_billing_controller.py \
  tests/test_attempt_scoped_usage_meters.py \
  tests/test_billing_spot_metering.py \
  tests/test_provider_settlement.py \
  tests/test_invoice_reads_match_billed_usage.py \
  tests/test_startup_validation.py \
  tests/test_startup_env_is_wired.py \
  tests/test_db_service_roles.py
```

Ruff and targeted Pyright passed; deployment shell syntax passed. Runtime policy
validation covers all three independent billing flags. Compose maps them to API
and maintenance, and deployment preflight rejects misspellings. The runbook in
`docs/reconciliation.md` describes staged opt-in, refusal evidence and disabling
repair without abandoning already-open meters.

These tests are not production acceptance, a full backend/frontend suite, or a
real GPU-to-invoice journey. Before enabling repair, review live findings and
prove completion/orphan settlement in the target runtime. Wallet debit and
provider payment execution remain their existing separate authorities. An open
meter cannot be safely rolled back by deploying the old completion logic that
mistakes any existing row for already-settled usage.
