"""Explicitly admitted and actually claimed native runtime fixtures."""

from contextlib import contextmanager
import time
import uuid


@contextmanager
def running_instance(*, exposed_ports=(8888, 8443), auto_launch=("jupyter", "vscode")):
    from psycopg.types.json import Jsonb
    from db import _get_pg_pool
    from control_plane.attempts import report_attempt_status
    from control_plane.db import run_transaction
    from control_plane.leases import claim_lease
    from control_plane.scheduler.claim import claim_next_job
    from control_plane.scheduler.reservation import reserve_and_bind
    from tests.native_jobs import admit_fixture_job

    pool = _get_pg_pool()
    job_id = f"abcd{uuid.uuid4().hex[:20]}"
    host_id = f"host-al-{uuid.uuid4().hex[:8]}"
    now = time.time()
    host_payload = {
        "host_id": host_id,
        "status": "active",
        "owner": "alice",
        "registered_at": now,
        "ip": "100.64.1.42",
        "cost_per_hour": 0,
    }
    job_payload = {
        "job_id": job_id,
        "status": "queued",
        "owner": "alice",
        "image": "python:3.11",
        "name": "al-test",
        "submitted_at": now,
        "interactive": True,
        "gpu_model": job_id,
        "num_gpus": 1,
        "auto_launch": list(auto_launch),
        "exposed_ports": list(exposed_ports),
    }
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts(host_id,status,registered_at,payload,admission_state,
            administrative_state,availability_state) VALUES(%s,'active',%s,%s,
            'admitted','admitted','ready')""",
            (host_id, now, Jsonb(host_payload)),
        )
        conn.execute(
            """INSERT INTO host_gpu_devices(host_id,gpu_uuid,model,total_vram_mb,
            allocatable_vram_mb,health) VALUES(%s,%s,'RTX 4090',24576,24576,'healthy')""",
            (host_id, f"GPU-{host_id}"),
        )
        conn.execute(
            """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,phase,
            desired_state,effective_priority,queued_at,owner_id) VALUES(%s,'queued',0,
            %s,%s,'pending','running',0,clock_timestamp(),'alice')""",
            (job_id, now, Jsonb(job_payload)),
        )
        admit_fixture_job(conn, job_id)

    def reserve(conn):
        claim = claim_next_job(conn, replica_id=job_id, scope_gpu_models=[job_id])
        assert claim and claim.job_id == job_id
        return reserve_and_bind(
            conn, job_id=job_id, claim_token=claim.claim_token, replica_id=job_id, host_id=host_id
        )

    r = run_transaction(reserve)
    auth = dict(
        job_id=job_id, host_id=host_id, attempt_id=r.attempt_id, fencing_token=r.fencing_token
    )
    run_transaction(
        lambda conn: claim_lease(
            conn, lease_id=r.lease_id, worker_session_id="auto-launch-test", **auth
        )
    )
    run_transaction(lambda conn: report_attempt_status(conn, status="running", **auth))
    yield {
        "job_id": job_id,
        "host_id": host_id,
        "slug": job_id[:12],
        "authority": {"attempt_id": r.attempt_id, "fencing_token": r.fencing_token},
    }
    with pool.connection() as conn:
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
        conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
