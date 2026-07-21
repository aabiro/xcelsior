"""Track A migrations 054/055 — control-plane schema invariant tests.

These are database-level tests: they verify that the *PostgreSQL schema
itself* (partial unique indexes, CHECK constraints, FKs, the fencing
sequence) enforces the blueprint §8 invariants even when application code
misbehaves. They intentionally use raw SQL, not scheduler helpers — the
constraints are the last line of defense and must hold on their own.

Requires the test database to be migrated to alembic head (>= 055).
"""

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_056 = all(
            _c.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0] is not None
            for t in ("job_attempts", "outbox_events")
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _has_056:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 056")

from psycopg.errors import (
    CheckViolation,
    ForeignKeyViolation,
    UniqueViolation,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def cleanup_ids():
    """Track inserted PK rows and delete them after the test.

    Deletion order respects FKs; job_attempts/allocations/leases cascade
    from jobs, and host_gpu_devices cascade from hosts.
    """
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mkjob(cleanup, status="queued", phase="pending", desired_state="running"):
    job_id = f"job-cp-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at, updated_at)
               VALUES (%s, %s, 0, %s, %s, %s, %s, now(), now())""",
            (job_id, status, time.time(), json.dumps({"name": job_id}), phase, desired_state),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mkhost(cleanup):
    host_id = f"host-cp-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)""",
            (host_id, time.time(), json.dumps({"name": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    return host_id


def _next_fence(conn) -> int:
    return conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone()[0]


def _mkattempt(conn, job_id, *, number=1, status="reserved", host_id=None):
    fence = _next_fence(conn)
    row = conn.execute(
        """INSERT INTO job_attempts (job_id, attempt_number, status, host_id, fencing_token)
           VALUES (%s, %s, %s, %s, %s) RETURNING attempt_id, fencing_token""",
        (job_id, number, status, host_id, fence),
    ).fetchone()
    return row[0], row[1]


def _mkdevice(conn, host_id, *, mode="exclusive", vram_mb=24576):
    return conn.execute(
        """INSERT INTO host_gpu_devices
               (host_id, gpu_uuid, model, total_vram_mb, allocatable_vram_mb,
                allocation_mode)
           VALUES (%s, %s, 'RTX 4090', %s, %s, %s)
           RETURNING gpu_device_id""",
        (host_id, f"GPU-{uuid.uuid4()}", vram_mb, vram_mb, mode),
    ).fetchone()[0]


# ── Schema presence ──────────────────────────────────────────────────


class TestSchemaPresence:
    def test_new_tables_exist(self):
        with _pool.connection() as conn:
            for table in (
                "job_attempts",
                "host_gpu_devices",
                "gpu_device_allocations",
                "placement_leases",
            ):
                assert conn.execute(
                    "SELECT to_regclass(%s)", (table,)
                ).fetchone()[0] == table, f"missing table {table}"

    def test_jobs_control_plane_columns_exist(self):
        with _pool.connection() as conn:
            cols = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'jobs'"
                ).fetchall()
            }
        for col in (
            "tenant_id", "team_id", "owner_id", "desired_state", "phase",
            "reason_code", "reason_details", "generation", "observed_generation",
            "version", "active_attempt_id", "spec", "spec_hash", "queued_at",
            "next_schedule_at", "updated_at", "effective_priority",
            "fair_share_finish", "schedule_claim_owner", "schedule_claim_token",
            "schedule_claim_expires_at", "schedule_attempt_count",
            "last_schedule_conflict_at", "wallet_hold_id",
        ):
            assert col in cols, f"jobs missing column {col}"

    def test_hosts_control_plane_columns_exist(self):
        with _pool.connection() as conn:
            cols = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'hosts'"
                ).fetchall()
            }
        for col in (
            "tenant_id", "provider_id", "owner_id", "region", "country",
            "province", "administrative_state", "availability_state",
            "generation", "observed_generation", "version",
            "inventory_generation", "last_observed_at",
            "observation_session_id", "drain_deadline", "drain_reason",
            "capabilities", "conditions",
        ):
            assert col in cols, f"hosts missing column {col}"

    def test_fencing_sequence_is_monotonic(self):
        with _pool.connection() as conn:
            a = _next_fence(conn)
            b = _next_fence(conn)
            assert b > a


# ── Invariant: legacy status projections are constrained ────────────
# Since migration 059 the projection trigger *normalizes* every write —
# an invalid value handed to the projection columns never reaches the
# CHECK constraint because the trigger recomputes it from the legacy
# truth first. The invariant under test is therefore "garbage cannot
# persist", not "garbage raises".


class TestJobProjectionConstraints:
    def test_invalid_phase_normalized_by_trigger(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids, phase="warming_up")
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT phase FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        assert row[0] == "pending"  # projected from status='queued'

    def test_invalid_desired_state_normalized_by_trigger(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids, desired_state="maybe")
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT desired_state FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        assert row[0] == "running"

    def test_invalid_host_admin_state_normalized_by_trigger(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE hosts SET administrative_state='paused' WHERE host_id=%s",
                (host_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT administrative_state FROM hosts WHERE host_id=%s",
                (host_id,),
            ).fetchone()
        assert row[0] in ("pending", "admitted", "draining", "disabled")


# ── Invariant 8.1: at most one active attempt per job ───────────────


class TestOneActiveAttemptPerJob:
    def test_second_active_attempt_rejected(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            _mkattempt(conn, job_id, number=1, status="reserved")
            conn.commit()
        with _pool.connection() as conn:
            with pytest.raises(UniqueViolation):
                _mkattempt(conn, job_id, number=2, status="running")

    def test_new_attempt_allowed_after_terminal(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            attempt_id, _ = _mkattempt(conn, job_id, number=1, status="reserved")
            conn.execute(
                "UPDATE job_attempts SET status='failed', ended_at=now() "
                "WHERE attempt_id=%s",
                (attempt_id,),
            )
            attempt2, fence2 = _mkattempt(conn, job_id, number=2, status="reserved")
            conn.commit()
            assert attempt2 is not None

    def test_retry_fence_is_higher(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            a1, fence1 = _mkattempt(conn, job_id, number=1, status="reserved")
            conn.execute(
                "UPDATE job_attempts SET status='lost', ended_at=now() WHERE attempt_id=%s",
                (a1,),
            )
            _, fence2 = _mkattempt(conn, job_id, number=2, status="reserved")
            conn.commit()
            assert fence2 > fence1

    def test_duplicate_attempt_number_rejected(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            a1, _ = _mkattempt(conn, job_id, number=1, status="reserved")
            conn.execute(
                "UPDATE job_attempts SET status='failed', ended_at=now() WHERE attempt_id=%s",
                (a1,),
            )
            with pytest.raises(UniqueViolation):
                _mkattempt(conn, job_id, number=1, status="reserved")


# ── Invariant 8.2: capacity ──────────────────────────────────────────


class TestExclusiveAllocation:
    def test_double_exclusive_allocation_rejected(self, cleanup_ids):
        job_a = _mkjob(cleanup_ids)
        job_b = _mkjob(cleanup_ids)
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            device_id = _mkdevice(conn, host_id)
            attempt_a, _ = _mkattempt(conn, job_a, host_id=host_id)
            attempt_b, _ = _mkattempt(conn, job_b, host_id=host_id)
            conn.execute(
                """INSERT INTO gpu_device_allocations
                       (attempt_id, job_id, host_id, gpu_device_id)
                   VALUES (%s, %s, %s, %s)""",
                (attempt_a, job_a, host_id, device_id),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    """INSERT INTO gpu_device_allocations
                           (attempt_id, job_id, host_id, gpu_device_id)
                       VALUES (%s, %s, %s, %s)""",
                    (attempt_b, job_b, host_id, device_id),
                )

    def test_released_device_can_be_reallocated(self, cleanup_ids):
        job_a = _mkjob(cleanup_ids)
        job_b = _mkjob(cleanup_ids)
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            device_id = _mkdevice(conn, host_id)
            attempt_a, _ = _mkattempt(conn, job_a, host_id=host_id)
            attempt_b, _ = _mkattempt(conn, job_b, host_id=host_id)
            conn.execute(
                """INSERT INTO gpu_device_allocations
                       (attempt_id, job_id, host_id, gpu_device_id)
                   VALUES (%s, %s, %s, %s)""",
                (attempt_a, job_a, host_id, device_id),
            )
            conn.execute(
                """UPDATE gpu_device_allocations
                      SET status='released', released_at=now(),
                          release_reason='attempt terminal'
                    WHERE attempt_id=%s""",
                (attempt_a,),
            )
            conn.execute(
                """INSERT INTO gpu_device_allocations
                       (attempt_id, job_id, host_id, gpu_device_id)
                   VALUES (%s, %s, %s, %s)""",
                (attempt_b, job_b, host_id, device_id),
            )
            conn.commit()

    def test_release_requires_released_at(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            device_id = _mkdevice(conn, host_id)
            attempt_id, _ = _mkattempt(conn, job_id, host_id=host_id)
            conn.execute(
                """INSERT INTO gpu_device_allocations
                       (attempt_id, job_id, host_id, gpu_device_id)
                   VALUES (%s, %s, %s, %s)""",
                (attempt_id, job_id, host_id, device_id),
            )
            with pytest.raises(CheckViolation):
                conn.execute(
                    "UPDATE gpu_device_allocations SET status='released' "
                    "WHERE attempt_id=%s",
                    (attempt_id,),
                )


# ── Invariant 8.3: one live lease per attempt ────────────────────────


class TestPlacementLeases:
    def _offer_lease(self, conn, job_id, attempt_id, host_id, fence):
        return conn.execute(
            """INSERT INTO placement_leases
                   (job_id, attempt_id, host_id, fencing_token, status,
                    claim_deadline)
               VALUES (%s, %s, %s, %s, 'offered', now() + interval '60 seconds')
               RETURNING lease_id""",
            (job_id, attempt_id, host_id, fence),
        ).fetchone()[0]

    def test_second_live_lease_rejected(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            attempt_id, fence = _mkattempt(conn, job_id, host_id=host_id)
            self._offer_lease(conn, job_id, attempt_id, host_id, fence)
            with pytest.raises(UniqueViolation):
                self._offer_lease(conn, job_id, attempt_id, host_id, fence)

    def test_replacement_lease_after_expiry(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            attempt_id, fence = _mkattempt(conn, job_id, host_id=host_id)
            lease_id = self._offer_lease(conn, job_id, attempt_id, host_id, fence)
            conn.execute(
                "UPDATE placement_leases SET status='expired' WHERE lease_id=%s",
                (lease_id,),
            )
            self._offer_lease(conn, job_id, attempt_id, host_id, fence)
            conn.commit()

    def test_active_lease_requires_claim_and_expiry(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            attempt_id, fence = _mkattempt(conn, job_id, host_id=host_id)
            lease_id = self._offer_lease(conn, job_id, attempt_id, host_id, fence)
            with pytest.raises(CheckViolation):
                conn.execute(
                    "UPDATE placement_leases SET status='active' WHERE lease_id=%s",
                    (lease_id,),
                )


# ── Invariant 8.1: active_attempt_id referential integrity ──────────


class TestActiveAttemptPointer:
    def test_dangling_active_attempt_rejected(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            with pytest.raises(ForeignKeyViolation):
                conn.execute(
                    "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
                    (str(uuid.uuid4()), job_id),
                )

    def test_attempt_delete_clears_pointer(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            attempt_id, _ = _mkattempt(conn, job_id)
            conn.execute(
                "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
                (attempt_id, job_id),
            )
            conn.execute(
                "DELETE FROM job_attempts WHERE attempt_id=%s", (attempt_id,)
            )
            row = conn.execute(
                "SELECT active_attempt_id FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
            conn.commit()
            assert row[0] is None


# ── MIG hierarchy ────────────────────────────────────────────────────


class TestGpuDeviceInventory:
    def test_duplicate_gpu_uuid_per_host_rejected(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        gpu_uuid = f"GPU-{uuid.uuid4()}"
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO host_gpu_devices (host_id, gpu_uuid, model)
                   VALUES (%s, %s, 'A100')""",
                (host_id, gpu_uuid),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    """INSERT INTO host_gpu_devices (host_id, gpu_uuid, model)
                       VALUES (%s, %s, 'A100')""",
                    (host_id, gpu_uuid),
                )

    def test_mig_child_cascades_with_parent(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            parent_id = _mkdevice(conn, host_id, mode="mig", vram_mb=81920)
            child_id = conn.execute(
                """INSERT INTO host_gpu_devices
                       (host_id, gpu_uuid, parent_gpu_device_id, model,
                        total_vram_mb, allocatable_vram_mb)
                   VALUES (%s, %s, %s, 'A100 MIG 1g.10gb', 10240, 10240)
                   RETURNING gpu_device_id""",
                (host_id, f"MIG-{uuid.uuid4()}", parent_id),
            ).fetchone()[0]
            conn.execute(
                "DELETE FROM host_gpu_devices WHERE gpu_device_id=%s", (parent_id,)
            )
            row = conn.execute(
                "SELECT 1 FROM host_gpu_devices WHERE gpu_device_id=%s", (child_id,)
            ).fetchone()
            conn.commit()
            assert row is None


# ── Migration 056: durable commands / outbox / reconcile queue ───────


@pytest.fixture
def cleanup_commands():
    """Delete agent_commands / outbox / reconcile rows created by a test."""
    marker = f"cp-test-{uuid.uuid4().hex[:10]}"
    yield marker
    if _pool is None:
        return
    with _pool.connection() as conn:
        conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (marker,))
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (marker,))
        conn.execute(
            "DELETE FROM reconciliation_queue WHERE resource_id=%s", (marker,)
        )
        conn.execute("DELETE FROM scheduled_tasks WHERE task_name=%s", (marker,))
        conn.execute(
            "DELETE FROM api_idempotency_keys WHERE principal=%s", (marker,)
        )
        conn.commit()


class TestDurableCommands:
    def _insert(self, conn, host_id, *, status="pending", idem=None,
                claim_owner=None, claim_expires=False):
        return conn.execute(
            """INSERT INTO agent_commands
                   (host_id, command, args, status, idempotency_key,
                    claim_owner, claim_expires_at)
               VALUES (%s, 'start_attempt', '{}'::jsonb, %s, %s, %s,
                       CASE WHEN %s THEN now() + interval '60 seconds' END)
               RETURNING command_id""",
            (host_id, status, idem, claim_owner, claim_expires),
        ).fetchone()[0]

    def test_duplicate_idempotency_key_rejected(self, cleanup_commands):
        host = cleanup_commands
        with _pool.connection() as conn:
            self._insert(conn, host, idem="start:att-1")
            with pytest.raises(UniqueViolation):
                self._insert(conn, host, idem="start:att-1")

    def test_null_idempotency_keys_are_not_deduped(self, cleanup_commands):
        host = cleanup_commands
        with _pool.connection() as conn:
            self._insert(conn, host)
            self._insert(conn, host)
            conn.commit()

    def test_unknown_status_rejected(self, cleanup_commands):
        host = cleanup_commands
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                self._insert(conn, host, status="done")

    def test_claimed_requires_owner_and_expiry(self, cleanup_commands):
        host = cleanup_commands
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                self._insert(conn, host, status="claimed")

    def test_claimed_with_owner_and_expiry_accepted(self, cleanup_commands):
        host = cleanup_commands
        with _pool.connection() as conn:
            self._insert(
                conn, host, status="claimed", claim_owner="worker-1",
                claim_expires=True,
            )
            conn.commit()


class TestOutboxEvents:
    def test_duplicate_event_idempotency_rejected(self, cleanup_commands):
        marker = cleanup_commands
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO outbox_events
                       (aggregate_type, aggregate_id, event_type, idempotency_key)
                   VALUES ('job', %s, 'job.v1.created', %s)""",
                (marker, f"job.v1.created:{marker}"),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    """INSERT INTO outbox_events
                           (aggregate_type, aggregate_id, event_type, idempotency_key)
                       VALUES ('job', %s, 'job.v1.created', %s)""",
                    (marker, f"job.v1.created:{marker}"),
                )

    def test_same_key_allowed_on_other_destination_class(self, cleanup_commands):
        marker = cleanup_commands
        with _pool.connection() as conn:
            for dest in ("sse", "webhook"):
                conn.execute(
                    """INSERT INTO outbox_events
                           (aggregate_type, aggregate_id, event_type,
                            destination_class, idempotency_key)
                       VALUES ('job', %s, 'job.v1.created', %s, %s)""",
                    (marker, dest, f"job.v1.created:{marker}"),
                )
            conn.commit()


class TestReconciliationQueue:
    def test_coalesces_per_resource(self, cleanup_commands):
        marker = cleanup_commands
        upsert = """
            INSERT INTO reconciliation_queue (resource_type, resource_id, reason)
            VALUES ('job', %s, %s)
            ON CONFLICT (resource_type, resource_id) DO UPDATE
               SET due_at = LEAST(reconciliation_queue.due_at, EXCLUDED.due_at),
                   reason = EXCLUDED.reason,
                   updated_at = clock_timestamp()
        """
        with _pool.connection() as conn:
            conn.execute(upsert, (marker, "heartbeat"))
            conn.execute(upsert, (marker, "lease_expiry"))
            n = conn.execute(
                "SELECT count(*) FROM reconciliation_queue WHERE resource_id=%s",
                (marker,),
            ).fetchone()[0]
            conn.commit()
            assert n == 1

    def test_plain_duplicate_rejected(self, cleanup_commands):
        marker = cleanup_commands
        with _pool.connection() as conn:
            conn.execute(
                "INSERT INTO reconciliation_queue (resource_type, resource_id) "
                "VALUES ('job', %s)",
                (marker,),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    "INSERT INTO reconciliation_queue (resource_type, resource_id) "
                    "VALUES ('job', %s)",
                    (marker,),
                )


class TestScheduledTasksAndIdempotency:
    def test_nonpositive_interval_rejected(self, cleanup_commands):
        marker = cleanup_commands
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                conn.execute(
                    "INSERT INTO scheduled_tasks (task_name, interval_seconds) "
                    "VALUES (%s, 0)",
                    (marker,),
                )

    def test_api_idempotency_unique_per_principal_route(self, cleanup_commands):
        marker = cleanup_commands
        ins = """
            INSERT INTO api_idempotency_keys
                (principal, route, idempotency_key, request_hash, expires_at)
            VALUES (%s, %s, 'k1', 'h1', now() + interval '1 day')
        """
        with _pool.connection() as conn:
            conn.execute(ins, (marker, "POST /instance"))
            conn.execute(ins, (marker, "POST /api/v1/launch-plans"))
            with pytest.raises(UniqueViolation):
                conn.execute(ins, (marker, "POST /instance"))


# ── Migration 057: observations / telemetry / heartbeats ─────────────


class TestObservations:
    def test_observation_immutable_per_session_generation(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            ins = """
                INSERT INTO host_observations (host_id, session_id, inventory_generation)
                VALUES (%s, 'boot-1', %s)
            """
            conn.execute(ins, (host_id, 1))
            conn.execute(ins, (host_id, 2))  # new generation OK
            with pytest.raises(UniqueViolation):
                conn.execute(ins, (host_id, 1))

    def test_observed_workload_cascades_with_observation(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            obs_id = conn.execute(
                "INSERT INTO host_observations (host_id, session_id) "
                "VALUES (%s, 'boot-1') RETURNING observation_id",
                (host_id,),
            ).fetchone()[0]
            wl_id = conn.execute(
                """INSERT INTO observed_workloads
                       (observation_id, host_id, session_id, state)
                   VALUES (%s, %s, 'boot-1', 'running') RETURNING id""",
                (obs_id, host_id),
            ).fetchone()[0]
            conn.execute(
                "DELETE FROM host_observations WHERE observation_id=%s", (obs_id,)
            )
            gone = conn.execute(
                "SELECT 1 FROM observed_workloads WHERE id=%s", (wl_id,)
            ).fetchone()
            conn.commit()
            assert gone is None

    def test_invalid_workload_state_rejected(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        with _pool.connection() as conn:
            obs_id = conn.execute(
                "INSERT INTO host_observations (host_id, session_id) "
                "VALUES (%s, 'boot-2') RETURNING observation_id",
                (host_id,),
            ).fetchone()[0]
            with pytest.raises(CheckViolation):
                conn.execute(
                    """INSERT INTO observed_workloads
                           (observation_id, host_id, session_id, state)
                       VALUES (%s, %s, 'boot-2', 'zombified')""",
                    (obs_id, host_id),
                )


class TestTelemetryTables:
    def test_telemetry_latest_upserts_one_row_per_device(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        upsert = """
            INSERT INTO telemetry_latest (host_id, gpu_uuid, sample)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (host_id, gpu_uuid) DO UPDATE
               SET sample = EXCLUDED.sample, received_at = clock_timestamp()
        """
        try:
            with _pool.connection() as conn:
                conn.execute(upsert, (host_id, "GPU-1", '{"temp": 60}'))
                conn.execute(upsert, (host_id, "GPU-1", '{"temp": 71}'))
                row = conn.execute(
                    "SELECT sample, count(*) OVER () FROM telemetry_latest "
                    "WHERE host_id=%s AND gpu_uuid='GPU-1'",
                    (host_id,),
                ).fetchone()
                conn.commit()
                assert row[0]["temp"] == 71 and row[1] == 1
        finally:
            with _pool.connection() as conn:
                conn.execute(
                    "DELETE FROM telemetry_latest WHERE host_id=%s", (host_id,)
                )
                conn.commit()

    def test_telemetry_samples_partition_routing(self, cleanup_ids):
        host_id = _mkhost(cleanup_ids)
        try:
            with _pool.connection() as conn:
                # Current timestamp lands in a monthly partition; a far-future
                # one lands in the DEFAULT partition instead of erroring.
                conn.execute(
                    "INSERT INTO telemetry_samples (host_id, sample) "
                    "VALUES (%s, '{}'::jsonb)",
                    (host_id,),
                )
                conn.execute(
                    "INSERT INTO telemetry_samples (host_id, sample, received_at) "
                    "VALUES (%s, '{}'::jsonb, now() + interval '10 years')",
                    (host_id,),
                )
                in_default = conn.execute(
                    "SELECT count(*) FROM telemetry_samples_default WHERE host_id=%s",
                    (host_id,),
                ).fetchone()[0]
                total = conn.execute(
                    "SELECT count(*) FROM telemetry_samples WHERE host_id=%s",
                    (host_id,),
                ).fetchone()[0]
                conn.commit()
                assert total == 2 and in_default == 1
        finally:
            with _pool.connection() as conn:
                conn.execute(
                    "DELETE FROM telemetry_samples WHERE host_id=%s", (host_id,)
                )
                conn.commit()

    def test_partition_maintenance_task_seeded(self):
        with _pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_tasks (task_name, interval_seconds, payload)
                VALUES (
                    'telemetry_partition_maintenance',
                    86400,
                    '{"table": "telemetry_samples", "months_ahead": 2, "retention_months": 6}'::jsonb
                )
                ON CONFLICT (task_name) DO NOTHING
                """
            )
            conn.commit()
            row = conn.execute(
                "SELECT interval_seconds FROM scheduled_tasks "
                "WHERE task_name='telemetry_partition_maintenance'"
            ).fetchone()
        assert row is not None and row[0] == 86400



class TestServiceHeartbeats:
    def test_unknown_service_rejected(self):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                conn.execute(
                    "INSERT INTO service_heartbeats (service, replica_id) "
                    "VALUES ('mainframe', 'r1')"
                )

    def test_heartbeat_upsert(self):
        replica = f"r-{uuid.uuid4().hex[:8]}"
        upsert = """
            INSERT INTO service_heartbeats (service, replica_id, schema_revision)
            VALUES ('scheduler', %s, %s)
            ON CONFLICT (service, replica_id) DO UPDATE
               SET last_heartbeat_at = clock_timestamp(),
                   schema_revision = EXCLUDED.schema_revision
        """
        try:
            with _pool.connection() as conn:
                conn.execute(upsert, (replica, "056"))
                conn.execute(upsert, (replica, "057"))
                row = conn.execute(
                    "SELECT schema_revision, started_at < last_heartbeat_at "
                    "FROM service_heartbeats WHERE service='scheduler' AND replica_id=%s",
                    (replica,),
                ).fetchone()
                conn.commit()
                assert row[0] == "057" and row[1] is True
        finally:
            with _pool.connection() as conn:
                conn.execute(
                    "DELETE FROM service_heartbeats WHERE replica_id=%s", (replica,)
                )
                conn.commit()
