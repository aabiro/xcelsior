# Lightning native column contract

Revision 141 removes the remaining Lightning float/epoch twins after archiving
all six original values with per-record SHA-256 checks. It does not create or
credit a payment, change canonical monetary values, or invent payment dates.

The only live stored authorities are integer `amount_msat`, `amount_sats`,
`amount_micros`, `amount_cad_minor`, exact `btc_cad_rate_exact`, and the typed
`*_at_ts` timestamps. API epoch/float fields are derived at the read boundary.
The payment-term trigger protects payer/tenant/currency, all expected amounts,
the exact rate, hash and expected creation/expiry. Status, confirmed payment
and confirmed ledger linkage retain their existing lifecycle writers.

`lightning_retired_projection_archive` is tenant-scoped, migration-owned and
read-only to runtime roles. Archive mismatches, unexpected column types/shapes
and unreviewed dependencies refuse contraction. The short locked transaction
archives, verifies and contracts together. A repeated unit after an interrupted
stamp retains the same evidence. Downgrade retains the canonical contract and
archive; use a reviewed compatible runtime for recovery.

Qualification: 207 tests passed with zero failures, errors or skips using the
isolated PostgreSQL 16 security harness. The selection includes exact archive
retention, mismatch refusal, up/down/up replay, real Lightning creation/replay,
wallet-credit idempotency, reconciliation, immutable money and lifecycle fields,
dedicated-role audit, the migration ledger/lock rules and revision-140 recovery.

Runnable gates: `tests/test_lightning_column_contraction.py`,
`tests/test_lightning_deposits.py`, `tests/test_ln_deposit_reconcile.py`,
`tests/test_shared_state_contract.py`,
`tests/test_crypto_funding_replay_is_one_deposit.py` and
`tests/test_db_service_roles.py`, through `api/scripts/run_security_gates.py`.

This is source qualification. Production remains revision 139 / `30b3daa6`.
Stop every old Lightning/API/billing/background writer before applying 140/141,
retain a fresh verified backup, use the direct migrator, apply reviewed role
privileges for the new archive, and hand off all readers/writers to the image
with floor 141. Verify both historical deposits and the archive against the
pre-change snapshot, check all financial invariants, and perform a full isolated
restore before recording deployment completion. Repairs remain `report_only`.
