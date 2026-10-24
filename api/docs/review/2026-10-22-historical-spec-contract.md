# Recorded execution recovery and first column contract

Migration 129 recovers historical execution specs only from matching saved
job, start-command and fenced-attempt hashes. Command job/attempt IDs, host,
fence and job generation must also agree. The original execution spec and
hash are retained; current payload never becomes an invented approved spec.
Already populated specs remain untouched. Nonterminal rows without specs and
rows with incomplete evidence receive an explicit unavailable assessment.

`job_spec_backfill_evidence` records the assessment, reason, original job and
payload digests and source command/attempt IDs. The evidence is an assessment
at migration time, not a claim that other backups cannot recover more history.
Each bounded batch commits independently. Locked unfinished work refuses the
migration; retry preserves committed batches. A guard refuses any spec update
that changes other job fields, including priority or lifecycle state through
an older trigger. Downgrade to 128 retains recovered specs and evidence.

Runtime roles can read this evidence within their assigned domains and cannot
directly insert, update or delete it. It follows job lifetime through its job
foreign key. The effective-privilege audit also detects grants that would let
runtime roles rewrite this evidence or the retired lease archive, including
grants through `PUBLIC`.

Migration 130 makes `jobs.phase` and `desired_state` non-null and requires a
real queue timestamp for pending jobs. Missing projections use recorded status
and submission time only; unknown facts abort the table unit. The job trigger
now runs only for legacy status, priority or submission-column writes and
changes projections only when those values change. Metadata, billing links,
claims and observation writes no longer reset effective priority, queue age
or native intent. Raw priority changes still reset its priority projection;
this bridge does not implement durable fairness or preemption.

The production-derived rehearsal imports 364 job rows, with exact execution
payloads and fenced evidence for the four hashed jobs. Other payloads retain
identity facts only, and unrelated wallet-hold references are omitted from the
fixture. This is a spec/column rehearsal, not a financial or full-system
restore. It recovers exactly **four specs**, records **360 unavailable
assessments**, preserves the other fixture job fields and execution history,
and passes downgrade/upgrade replay. No missing workload owner, execution
spec or submission time was manufactured.

The aggregate-only audit is runnable as:

```sh
python api/scripts/audit_spec_qualification.py \
  --expected-database xcelsior --expected-port 5434
```

Add `--require-complete-specs` to require every retained job to have qualified
spec evidence. The rehearsal correctly returns exit 2 while the 360 gaps
remain. A matching hash alone does not qualify an unproven legacy snapshot.
The report includes native column constraints and remaining trigger names,
without job IDs, payloads, credentials or DSNs.

The remaining trigger-removal conversions are concrete:

| Writer | Remaining dependency |
| --- | --- |
| `DatabaseOps.upsert_job` | Initial phase, desired state and queue projection for legacy-shaped inserts/updates. |
| Fenced attempt reports | Canonical phase and desired-state projection alongside compatibility status. |
| Lifecycle stop/restart/readmission | Native desired state, phase and queue fields in every transition. |
| Lease expiry and unstarted serverless cancellation | Native terminal/requeue projections. |
| Host upsert and admission decisions | Administrative/availability projections and the admission compatibility payload mirror from migration 082. |

Both projection triggers remain. Final removal needs those writers converted
and a gate exercising them with the triggers absent. The 360 unavailable specs
still require recovery or reviewed historical disposition before a global
non-null spec contract or payload-bridge removal. Remaining JSONB paths,
all-table semantic review and durable fairness/preemption stay open in B16.

The expanded gate passes **2,614 tests** with zero failures/errors/skips and
the same three documented baseline exclusions. The focused gate passes 222
tests and the production-derived drill passes one test. Source hashes,
private-fixture digest, static checks, cleanup and live
read-only boundary evidence are recorded in the
[machine-readable receipt](2026-10-22-historical-spec-contract.json).
Production remains at schema 125 and repairs remain `report_only`; migrations
129/130 have not been applied there. Runtime floor remains 126, preserving the
staged native handoff before retired-lease contraction and subsequent backfills.
