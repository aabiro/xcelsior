# Explicit historical job disposition

Migration **132** retains historical jobs and their billing, command and
observation links. Recovered execution evidence is recorded as
`retained_recovered`; unavailable execution evidence is recorded as
`retained_unavailable`. Neither is a new customer admission. No job or
financial row is deleted, and no missing launch input or owner is invented.

The migration records a versioned, hashed disposition receipt for each
historical row, including whether identity evidence is available. Runtime
roles can inspect receipts but cannot insert, update or delete them. The
terminal-only exclusion is installed before batches run, protects every
committed disposition, and remains enforced after downgrade. An unqualified
live or attempt-owned row refuses migration. Independently committed batches
survive that refusal and complete on retry after an explicit handoff; the
migration never clears authority itself. The frozen 129 qualification
algorithm also assesses terminal rows written during the native handoff.

Resume returns `historical_job_requires_new_admission` for retained history.
The aggregate audit reports disposition completion separately from complete
spec recovery. `--require-history-disposition` can pass while
`--require-complete-specs` correctly refuses the unrecoverable gaps.

The initial **145-test** integration gate passed, covering bootstrap, ledger,
schema floor, native resume, production-shaped legacy handoff, database roles
and migration lock discipline. After moving the guard ahead of backfill, the
final **36-test** gate passed, including partial batches, refusal, retry,
downgrade, all explicit role denials and the production-derived replay. Both
runs have zero failures, errors or skips. One existing contract-cleanup
placement exclusion remains in the initial gate. Its synthetic live legacy
fixture now explicitly qualifies through 131; 132 must refuse such live rows.
The new terminal-history fixture exercises the complete 132 path.

The minimized 364-job replay recovers four exact specs, retains 360 unavailable
specs and disposes all historical rows without changing original job fields,
host records or execution history. Downgrade/replay preserves receipts and
exclusion. The fixture omits financial rows and wallet-hold references, so this
is not a financial restore. A separate synthetic linked billing cycle keeps
its exact 2,500,000-micro amount through disposition. Minimization also limits
identity recovery proof; missing identity is reported rather than inferred.

Repository Ruff, Black on all ten changed Python files, Pyright and all twelve
worker signatures pass. Exact source hashes, test manifests, minimization and
cleanup receipts are in the [JSON receipt](2026-10-23-historical-disposition.json).
The nine unrelated dirty root files retain their recorded hashes.

Runtime floor is **132**. Deploy the qualified 126-compatible native writers
at `dfacc879` or a separately qualified 126-compatible descendant first,
complete the native service/worker handoff and migrations through 132, then
deploy this runtime. Directly running upgrade-to-head before that handoff is
unsupported. Production still reports schema 125; this candidate has not
been deployed and repairs remain `report_only`.

B16 remains open for strict stored admission specs, remaining JSONB authority
readers and physical columns, durable fairness/preemption, and production
acceptance. The fairness prototype is outside this commit and is not an
acceptance claim.
