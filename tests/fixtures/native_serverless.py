"""Real native placement for integration tests of serverless billing ticks."""

from contextlib import contextmanager
import time
import uuid

from psycopg.types.json import Jsonb

from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import claim_lease
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import reserve_and_bind
from db import _get_pg_pool
from serverless.billing_authority import capture_quote
from tests.native_jobs import admit_fixture_job


@contextmanager
def running_serverless(repo, endpoint_id, *, age_sec=130):
    endpoint = repo.get_endpoint(endpoint_id)
    assert endpoint is not None
    job_id, host_id = [f"priced-fixture-{uuid.uuid4().hex}" for _ in range(2)]
    worker = repo.create_worker(endpoint_id, scheduler_job_id=job_id, allocated_at=0)
    payload = dict(
        job_id=job_id,
        owner=endpoint["owner_id"],
        image="test/image:1",
        name=job_id,
        gpu_model=job_id,
        job_type="serverless_worker",
        num_gpus=endpoint["gpu_count"],
        serverless_worker_id=worker["worker_id"],
        serverless_endpoint_id=endpoint_id,
        serverless_billing_quote=capture_quote(endpoint),
    )
    pool = _get_pg_pool()
    try:
        with pool.connection() as conn:
            conn.execute(
                """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,phase,
              desired_state,queued_at,effective_priority,owner_id) VALUES(%s,'queued',0,%s,%s,
              'pending','running',clock_timestamp(),0,%s)""",
                (job_id, time.time(), Jsonb(payload), endpoint["owner_id"]),
            )
            admit_fixture_job(conn, job_id)
            conn.execute(
                """INSERT INTO hosts(host_id,status,registered_at,payload,administrative_state,
              availability_state,admission_state) VALUES(%s,'active',%s,%s,'admitted','ready','admitted')""",
                (host_id, time.time(), Jsonb({"admitted": True, "cost_per_hour": 0})),
            )
            for index in range(endpoint["gpu_count"]):
                conn.execute(
                    """INSERT INTO host_gpu_devices(host_id,gpu_uuid,model,total_vram_mb,
                  allocatable_vram_mb,health) VALUES(%s,%s,'RTX 4090',24576,24576,'healthy')""",
                    (host_id, f"GPU-{host_id}-{index}"),
                )

        def reserve(conn):
            claim = claim_next_job(conn, replica_id=job_id, scope_gpu_models=[job_id])
            assert claim and claim.job_id == job_id
            return reserve_and_bind(
                conn,
                job_id=job_id,
                host_id=host_id,
                claim_token=claim.claim_token,
                replica_id=job_id,
            )

        r = run_transaction(reserve)
        auth = dict(
            job_id=job_id, host_id=host_id, attempt_id=r.attempt_id, fencing_token=r.fencing_token
        )
        run_transaction(
            lambda conn: claim_lease(
                conn, lease_id=r.lease_id, worker_session_id="priced-fixture", **auth
            )
        )
        run_transaction(lambda conn: report_attempt_status(conn, status="running", **auth))
        allocated = time.time() - age_sec
        with pool.connection() as conn:
            conn.execute(
                "UPDATE job_attempts SET started_at=to_timestamp(%s) WHERE attempt_id=%s",
                (allocated, r.attempt_id),
            )
            conn.execute(
                "UPDATE usage_meters SET started_at=%s WHERE job_id=%s", (allocated, job_id)
            )
        repo.update_worker(worker["worker_id"], allocated_at=allocated)
        yield repo.get_worker(worker["worker_id"])
    finally:
        with pool.connection() as conn:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM serverless_jobs WHERE worker_id=%s", (worker["worker_id"],))
            conn.execute(
                "DELETE FROM serverless_workers WHERE worker_id=%s", (worker["worker_id"],)
            )
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
