# Admission replay and worker recovery

Admission now records the normalized caller request inside its immutable
execution snapshot. An exact retry returns the recorded job without rerunning
placement, tier, storage or egress defaults. Changed owner, pin, resource or
workload intent refuses replay. Observation inserts and updates cannot invent
an approved execution snapshot from mutable payload.

Worker logs use an attempt-specific atomic spool in the journal's state
directory. Unacknowledged batches survive outages and restart; delivery is at
least once. Adoption resumes forwarding only after the exact lease is
confirmed. A bounded background sweep retries retained terminal backlogs.

Resource setup journals host ownership before privileged operations. Restart
adoption restores exclusive GPU flock ownership. Cleanup waits for a missing
or matching exited container, retains failed operations, transfers shared
mount ownership to the remaining attempt, and releases exclusive resources
last. Workspace identity includes the attempt. A failed post-claim journal
write cannot retry a start against an already consumed lease. Failed terminal
reports retain their original interpretation for recovery and settlement.

The final focused gate passed **232 tests**, with no failures, errors, skips,
or unhandled thread warnings (warnings are errors). It uses the disposable
PostgreSQL runner and isolates Docker/privileged side effects. Repository
Ruff, Black on all 13 changed Python files, Pyright on 247 files with zero
errors/warnings, and all 12 distribution signatures pass. The expanded suite
was running when this receipt was prepared; this receipt does not assert its
result. Source hashes and exact evidence paths are in the
[JSON receipt](2026-10-23-admission-worker-recovery.json).

Two additional edges remain for the next change: retaining terminal intent
when the outer runner raises unexpectedly, and refusing resource adoption
from another host's copied journal. Historical disposition, strict admitted
specs, remaining JSONB/column contraction, durable fairness/preemption and
production acceptance also remain open. The independent history/fairness
prototypes are not part of this commit or its acceptance claim.

No production mutation was performed. The read-only production probe still
reports schema 125; all 364 historical jobs are terminal, with zero active
attempt pointers, leases or allocations. Repairs remain `report_only`, and
all nine unrelated dirty root-worktree files retain their recorded hashes.
