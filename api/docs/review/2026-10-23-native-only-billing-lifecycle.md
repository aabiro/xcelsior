# Native billing lifecycle qualification

Stop, start, restart and termination now require a hash-validated native
admission. Billing no longer contains unfenced agent starts, pauses, optimistic
status transitions, direct container termination or pre-settlement volume
detach. Billing anchors use the admitted owner, including after observation
metadata changes. Worker settlement still owns the final running/stopped state.

Admitted pending/stopped jobs without execution authority can terminate in a
job-locked transaction. Live attempts, offered/active leases and unreleased GPU
allocations refuse this path even when their job pointer is missing. The state
change, wallet-hold release and outbox intent commit together. This does not
assert physical removal of preserved worker resources.

`native-only-lifecycle-qualified`: 69 passed, including placement-lock races,
outbox-failure rollback, idempotency and refusal of unadmitted/corrupt work.
The separately frozen schema-133 base passed 3,210 selected tests; this is not
a full-suite result for the later schema-135 candidate. Three previously
documented baseline exclusions remain.

The production-derived disposable rehearsal now reaches 135 and rolls back to
132/replays: all 364 jobs and execution history remain, four exact recorded
specs recover, 360 unavailable specs remain explicitly retained, and all four
retired columns have archived values. Financial rows are outside that private
fixture; synthetic linked-ledger preservation is covered separately by the
column-contraction tests. No production database mutation occurred.

Remaining: remove obsolete periodic unfenced stop redelivery and host
start/pause/reset directives, complete the final combined regression, and
qualify the staged production handoff. Repairs remain `report_only`.
