# Durable scheduling candidate

Migration 133 adds versioned operator team policies, persisted queue components,
weighted service credit, a virtual clock and durable preemption plans. It enables
no preemption or force escalation. Runtime roles can read policy revisions but
cannot insert, update or delete them. An operator with the migration connection
publishes a revision through `scheduler.fairness.publish_policy`; the scoped
identity comes from the immutable admission spec, never observation JSON.

Claims calculate administrative class, bounded priority, reservation entitlement,
a five-minute age step capped at 20 and weighted GPU service finish using database
time. The chosen components and revision commit with the claim. Reservation locks
the job, short accounting clock/team state, host and ordered devices; it enforces
team concurrency/GPU limits and spends credit only with successful placement.
The accounting clock serializes a short database transaction across replicas; it
is not a scheduler leader or a lock held during scoring or external work. A
policy revision after claim refuses placement until a fresh queue decision.

Preemption is opt-in per nominated team. A transaction locks the nominee and
host's current job set without waiting, validates native admissions and fresh
host/lease authority, and records a plan, nominated devices and fenced victim
commands together. Only explicitly interruptible or spot jobs are eligible.
Current administrative class, priority and entitlement protect higher-ranked
victims. The plan records `preemption/v1`, its operator revision, deadline and
checkpoint policy `none`; this policy does not promise a workload checkpoint.

The worker durably records terminal interpretation before signaling. Grace sends
TERM without Docker's implicit timed KILL. Force requires an explicit plan policy
and elapsed deadline. Running/uncertain containers retain lease and allocation;
retryable transport loss retains the journal. Deadline reconciliation never frees
an allocation. A nomination becomes ready only after the exact victim attempt is
terminal, its lease is inactive and its device allocations are released. Other
reservations cannot consume held devices. Transfer occurs in a new reservation
with a new fencing token. Scheduler restarts resume persisted plans and bounded
sweeps skip contended plans independently.

Confirmed compute preemption requeues with a new generation. Serverless
preemption closes its worker identity through native settlement rather than
reusing that worker. User cancel/terminate/stop intent takes precedence. Cancelled
or expired nominations release their holds without asserting victim termination.

The corrected scheduling/worker gate passes **169** checks. Policy rights, fresh
bootstrap, ledger and lock discipline pass **60** checks with one explicitly
deselected baseline contract-cleanup case. The final policy-race, scheduling and
fresh generated-inventory gate passes **46** checks. These selections overlap;
their counts are not additive. Exact runner manifests and original failed runs
are retained in the [JSON receipt](2026-10-23-durable-scheduling.json). One earlier
integration failure came from a partition fixture with no native admission. It
now approves its initial workload explicitly; production has no test admission
interceptor. All twelve worker signatures verify.

This is a source candidate. Production remains at 125 and requires the staged
native writer/agent handoff before 131, historical disposition through 132, then
133 and this runtime. Re-provision dedicated role grants after expansion. Keep
all repair policies `report_only`. Migration downgrade deliberately retains
policy, credit, nominations and plan evidence; it does not restore removed
projection triggers or make a pre-133 binary safe automatically.

The broader frozen regression is still running with native-fixture and source
iterator failures under investigation. Lifecycle JSONB authority, redundant
columns and production acceptance remain open. B16 is not marked complete.
