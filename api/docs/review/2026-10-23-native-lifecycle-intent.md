# Native lifecycle intent authority

Migration 134 copies only recorded stop/cancel/terminate/restart intent to the
native `jobs.lifecycle_intent` column. Unknown types or values refuse the whole
table unit. It invents no intent from status. New controllers write the column;
attempt settlement, lease expiry, reconciliation and preemption read it. Final
settlement consumes it atomically. Observation JSON cannot change controller
intent or wake the queue merely by editing an old payload key.

The retained job-reconciliation trigger compares native fields; it only enqueues
work and grants no lifecycle authority. Invalid native intent is rejected by a
validated CHECK. Stop/terminate/cancel/restart delivery remains fenced and atomic
with desired state. The downgrade first copies the latest native value back for
old readers, explicitly removing consumed NULL keys, before dropping the column.
An old restart key cannot return after rollback. All unrelated payload fields,
job records and financial/history links remain intact.

The read-only production probe reports schema 125 and 364 jobs with no lifecycle
intent key; it also confirms all four old placement projection columns are empty.
No production migration was applied. The probe emits aggregates only. Source
gates pass **132** lifecycle/worker/protocol checks and **57** migration/native
writer/reconciliation/bootstrap checks, without failures, errors or skips. A
separate runtime-role/ledger selection passed 106 checks and failed only a stale
README head filename; the final ledger/static selection records that correction.
Selections overlap. Exact manifests, original failures, schema provenance,
cleanup and protected root hashes are in the
[JSON receipt](2026-10-23-native-lifecycle-intent.json).

Apply 134 with **every lifecycle writer quiesced**. Deploy all native-column API,
scheduler and maintenance writers before reopening mutations; old JSONB intent
writers cannot coexist after authority moves. This candidate requires schema
134. Earlier native writer handoffs, history qualification and 133 policy grants
remain prerequisites. Keep repairs `report_only`. Production acceptance and the
remaining redundant placement columns are still open; B16 remains unchecked.
