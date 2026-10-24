# Canonical placement constraints and physical contraction

Native scheduling now enforces the preference stored with admission during
candidate selection, preemption planning and reservation. Verification, measured
uptime, tier and premium constraints cannot be removed by observation metadata.
Reservation revalidates the current candidate fleet and evidence in the same
serializable transaction that binds authority. A revoked verification or changed
premium baseline after claim refuses binding. A failed preference cannot issue
victim stop commands. The selected evidence is stored with the attempt; queue
refusals retain the asked/best-available numbers.

New admissions normalize percent inputs through exact basis-point conversion
before freezing their execution preference. Canonical workload hashes retain
their existing semantics. Observation upserts cannot change an existing
admission. Runtime writers no longer name the four old placement projection
columns. Existing recorded specs are never rehashed or re-admitted.

Migration 135 archives the original four-column projection and any old payload
lifecycle key with a database-checked SHA-256, then drops the columns and removes
that obsolete key. It preserves jobs, native admission, current lifecycle intent
and financial/history links. Unexpected column types, dependencies or archive
disagreement refuse the entire bounded-lock unit; no cascading drop is used.
All runtime roles are denied archive writes. Downgrade restores original
projections and derives a cache for newly admitted jobs from native spec only.
Archive evidence remains; stale JSONB restart intent is never restored.

Serializable failures reported at COMMIT are explicit PostgreSQL aborts and now
receive bounded retries. A real two-connection write-skew test verifies retry
without duplicate writes and verifies isolation does not leak into the pool.
Connection loss during commit remains ambiguous and is never blindly retried.

Validation passes **213** contraction/migration/role/preference/bootstrap/generated
inventory checks (one explicit baseline deselection), **98** scheduling/workflow
checks, **51** queue/transaction/preemption checks and **23** staged migration
entrypoint/harness checks. Counts overlap. Exact manifests, source hashes,
cleanup and protected root hashes are in the
[JSON receipt](2026-10-23-placement-column-contraction.json). Repository Ruff and
Pyright (zero errors/warnings) pass. The read-only production inventory found
364 jobs, with no lifecycle keys or populated placement projections; production
remains at 125.

The runtime minimum remains **134** so these writers can be deployed before
135. Use `run_db_migrations.py --target REVISION` for each qualified stage;
relative revision expressions are refused and migration credentials remain in
the one-shot environment. The normal deploy's automatic upgrade to head is
not the handoff procedure from 125. Follow the earlier 126-native handoff before
131/132, apply 133/grants, quiesce all lifecycle writers for 134, deploy these
134-compatible native preference writers everywhere, then apply 135 and refresh
role grants before acceptance. Repairs remain `report_only`.

This is source qualification. Physical unfenced billing lifecycle fallback
removal, final combined regression and production acceptance remain open. B16
stays unchecked.
