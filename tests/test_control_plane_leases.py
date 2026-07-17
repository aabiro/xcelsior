"""Lease & fencing engine (§8.3 / §11) — claim gate, CAS renewal, expiry.

Builds real reservations through the Stage B/E pipeline, then drives the
worker-side lease protocol against them.
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
        _migrated = (
            _c.execute("SELECT to_regclass('outbox_events')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 056")

from control_plane.db import run_transaction
from control_plane.leases import (
    FencingViolation,
    LeaseClaimRejected,
    LeaseRenewRejected,
    claim_lease,
    expire_stale_leases,
    release_lease,
    renew_lease,
    require_current_fence,
)
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import reserve_and_bind


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _reserved(fleet, *, lease_claim_ttl_sec=60, lease_renewal_ttl_sec=300):
    """Create job+host, claim, reserve. Returns the Reservation."""
    job_id = f"job-lease-{uuid.uuid4().hex[:10]}"
    host_id = f"host-lease-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now())""",
            (job_id, time.time(), json.dumps({"name": job_id, "gpu_model": job_id})),
        )
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  administrative_state, availability_state)
               VALUES (%s, 'active', %s, '{"admitted": true}', 'admitted', 'ready')""",
            (host_id, time.time()),
        )
        conn.execute(
            """INSERT INTO host_gpu_devices
                   (host_id, gpu_uuid, model, total_vram_mb,
                    allocatable_vram_mb, health)
               VALUES (%s, %s, 'RTX 4090', 24576, 24576, 'healthy')""",
            (host_id, f"GPU-{host_id}"),
        )
        conn.commit()
    fleet["jobs"].append(job_id)
    fleet["hosts"].append(host_id)
    with _pool.connection() as conn:
        # Scoped claim (gpu_model == job_id): immune to queue residue.
        claimed = claim_next_job(
            conn, replica_id="lease-test", scope_gpu_models=[job_id]
        )
        conn.commit()
    assert claimed is not None and claimed.job_id == job_id
    return run_transaction(
        lambda conn: reserve_and_bind(
            conn,
            job_id=job_id,
            claim_token=claimed.claim_token,
            replica_id="lease-test",
            host_id=host_id,
            lease_claim_ttl_sec=lease_claim_ttl_sec,
            lease_renewal_ttl_sec=lease_renewal_ttl_sec,
        )
    )


def _claim_grant(resv, session="wkr-sess-1", **overrides):
    kwargs = dict(
        lease_id=resv.lease_id,
        job_id=resv.job_id,
        attempt_id=resv.attempt_id,
        host_id=resv.host_id,
        fencing_token=resv.fencing_token,
        worker_session_id=session,
    )
    kwargs.update(overrides)
    return run_transaction(lambda conn: claim_lease(conn, **kwargs))


class TestClaimGate:
    def test_exact_tuple_claims_and_attempt_advances(self, fleet):
        resv = _reserved(fleet)
        grant = _claim_grant(resv)
        assert grant.fencing_token == resv.fencing_token
        with _pool.connection() as conn:
            lease = conn.execute(
                "SELECT status, last_worker_session_id FROM placement_leases "
                "WHERE lease_id=%s",
                (resv.lease_id,),
            ).fetchone()
            attempt = conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s",
                (resv.attempt_id,),
            ).fetchone()
        assert lease[0] == "active" and lease[1] == "wkr-sess-1"
        assert attempt[0] == "lease_claimed"

    def test_wrong_fence_rejected(self, fleet):
        resv = _reserved(fleet)
        with pytest.raises(LeaseClaimRejected) as excinfo:
            _claim_grant(resv, fencing_token=resv.fencing_token + 1)
        assert "fencing token" in str(excinfo.value)

    def test_wrong_host_rejected(self, fleet):
        resv = _reserved(fleet)
        with pytest.raises(LeaseClaimRejected) as excinfo:
            _claim_grant(resv, host_id="host-imposter")
        assert "different host" in str(excinfo.value)

    def test_double_claim_rejected(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv)
        with pytest.raises(LeaseClaimRejected) as excinfo:
            _claim_grant(resv, session="wkr-sess-2")
        assert "not claimable" in str(excinfo.value)

    def test_expired_offer_rejected(self, fleet):
        resv = _reserved(fleet, lease_claim_ttl_sec=1)
        time.sleep(1.2)
        with pytest.raises(LeaseClaimRejected) as excinfo:
            _claim_grant(resv)
        assert "offer expired" in str(excinfo.value)


class TestRenewAndRelease:
    def test_renew_extends_expiry(self, fleet):
        resv = _reserved(fleet)
        grant = _claim_grant(resv)
        new_expiry = run_transaction(
            lambda conn: renew_lease(
                conn,
                lease_id=resv.lease_id,
                attempt_id=resv.attempt_id,
                host_id=resv.host_id,
                fencing_token=resv.fencing_token,
                worker_session_id="wkr-sess-1",
            )
        )
        assert new_expiry >= grant.expires_at

    def test_renew_with_stale_fence_rejected(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv)
        with pytest.raises(LeaseRenewRejected):
            run_transaction(
                lambda conn: renew_lease(
                    conn,
                    lease_id=resv.lease_id,
                    attempt_id=resv.attempt_id,
                    host_id=resv.host_id,
                    fencing_token=resv.fencing_token - 1,
                    worker_session_id="wkr-sess-1",
                )
            )

    def test_released_lease_cannot_be_renewed(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv)
        ok = run_transaction(
            lambda conn: release_lease(
                conn,
                lease_id=resv.lease_id,
                attempt_id=resv.attempt_id,
                host_id=resv.host_id,
                fencing_token=resv.fencing_token,
            )
        )
        assert ok is True
        with pytest.raises(LeaseRenewRejected):
            run_transaction(
                lambda conn: renew_lease(
                    conn,
                    lease_id=resv.lease_id,
                    attempt_id=resv.attempt_id,
                    host_id=resv.host_id,
                    fencing_token=resv.fencing_token,
                    worker_session_id="wkr-sess-1",
                )
            )


class TestFenceGate:
    def test_current_authority_passes(self, fleet):
        resv = _reserved(fleet)
        run_transaction(
            lambda conn: require_current_fence(
                conn,
                job_id=resv.job_id,
                attempt_id=resv.attempt_id,
                host_id=resv.host_id,
                fencing_token=resv.fencing_token,
            )
        )

    def test_stale_fence_rejected(self, fleet):
        resv = _reserved(fleet)
        with pytest.raises(FencingViolation):
            run_transaction(
                lambda conn: require_current_fence(
                    conn,
                    job_id=resv.job_id,
                    attempt_id=resv.attempt_id,
                    host_id=resv.host_id,
                    fencing_token=resv.fencing_token - 1,
                )
            )

    def test_fence_rejected_after_attempt_terminal(self, fleet):
        resv = _reserved(fleet)
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE job_attempts SET status='lost', ended_at=now() "
                "WHERE attempt_id=%s",
                (resv.attempt_id,),
            )
            conn.commit()
        with pytest.raises(FencingViolation):
            run_transaction(
                lambda conn: require_current_fence(
                    conn,
                    job_id=resv.job_id,
                    attempt_id=resv.attempt_id,
                    host_id=resv.host_id,
                    fencing_token=resv.fencing_token,
                )
            )


class TestExpirySweep:
    def _state(self, resv):
        with _pool.connection() as conn:
            return {
                "lease": conn.execute(
                    "SELECT status FROM placement_leases WHERE lease_id=%s",
                    (resv.lease_id,),
                ).fetchone()[0],
                "attempt": conn.execute(
                    "SELECT status, failure_code FROM job_attempts "
                    "WHERE attempt_id=%s",
                    (resv.attempt_id,),
                ).fetchone(),
                "allocations": conn.execute(
                    "SELECT count(*) FROM gpu_device_allocations "
                    "WHERE attempt_id=%s AND status='active'",
                    (resv.attempt_id,),
                ).fetchone()[0],
                "command": conn.execute(
                    "SELECT status FROM agent_commands WHERE attempt_id=%s",
                    (resv.attempt_id,),
                ).fetchone()[0],
                "job": conn.execute(
                    "SELECT phase, active_attempt_id, reason_code FROM jobs "
                    "WHERE job_id=%s",
                    (resv.job_id,),
                ).fetchone(),
            }

    def test_unclaimed_offer_expires_and_requeues(self, fleet):
        resv = _reserved(fleet, lease_claim_ttl_sec=1)
        time.sleep(1.2)
        expired = run_transaction(lambda conn: expire_stale_leases(conn, grace_sec=0))
        mine = [e for e in expired if e["lease_id"] == resv.lease_id]
        assert mine and mine[0]["attempt_terminal"] == "failed"
        state = self._state(resv)
        assert state["lease"] == "expired"
        assert state["attempt"][0] == "failed"
        assert state["attempt"][1] == "lease_claim_timeout"
        assert state["allocations"] == 0
        assert state["command"] == "cancelled"
        assert state["job"][0] == "pending" and state["job"][1] is None
        assert state["job"][2] == "lease_claim_timeout"

    def test_renewal_timeout_marks_attempt_lost(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv)
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE placement_leases SET expires_at = clock_timestamp() "
                "- interval '10 minutes' WHERE lease_id=%s",
                (resv.lease_id,),
            )
            conn.commit()
        run_transaction(lambda conn: expire_stale_leases(conn, grace_sec=30))
        state = self._state(resv)
        assert state["lease"] == "expired"
        assert state["attempt"][0] == "lost"
        assert state["job"][0] == "pending"

    def test_requeued_job_gets_higher_fence_on_retry(self, fleet):
        resv = _reserved(fleet, lease_claim_ttl_sec=1)
        time.sleep(1.2)
        run_transaction(lambda conn: expire_stale_leases(conn, grace_sec=0))
        with _pool.connection() as conn:
            claimed = claim_next_job(
                conn, replica_id="lease-test-2", scope_gpu_models=[resv.job_id]
            )
            conn.commit()
        assert claimed is not None and claimed.job_id == resv.job_id
        resv2 = run_transaction(
            lambda conn: reserve_and_bind(
                conn,
                job_id=resv.job_id,
                claim_token=claimed.claim_token,
                replica_id="lease-test-2",
                host_id=resv.host_id,
            )
        )
        assert resv2.attempt_number == 2
        assert resv2.fencing_token > resv.fencing_token
        # Old fence has no authority over the new attempt.
        with pytest.raises(FencingViolation):
            run_transaction(
                lambda conn: require_current_fence(
                    conn,
                    job_id=resv.job_id,
                    attempt_id=resv.attempt_id,
                    host_id=resv.host_id,
                    fencing_token=resv.fencing_token,
                )
            )

    def test_healthy_lease_untouched_by_sweep(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv)
        run_transaction(lambda conn: expire_stale_leases(conn))
        assert self._state(resv)["lease"] == "active"
