"""Team and tenant quotas apply to admitted work.

Fairness and quota policy key on ``spec.team_id`` → ``spec.tenant_id`` →
owner. Admission never recorded a team or tenant in the spec, so a published
``team:`` (or ``tenant:``) policy matched no job and every quota was inert.
"""

from __future__ import annotations

import time
import uuid

import pytest

import scheduler
from tests._db_helpers import fund_test_owner, price_test_gpu
from tests.native_jobs import insert_schedulable_host

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
    _native = scheduler._active_backend() == "postgres"
except Exception:  # pragma: no cover - skip path
    _pool, _native = None, False

pytestmark = pytest.mark.skipif(not _native, reason="native admission needs PostgreSQL")

from control_plane.db import run_transaction  # noqa: E402
from control_plane.scheduler.claim import claim_next_job  # noqa: E402
from control_plane.scheduler.fairness import TeamPolicy, publish_policy  # noqa: E402
from control_plane.scheduler.reservation import TeamQuotaConflict, reserve_and_bind  # noqa: E402


@pytest.fixture
def team():
    marker = uuid.uuid4().hex[:10]
    team_id, wallet = f"team-{marker}", f"team-wallet-{marker}"
    state = {"team_id": team_id, "wallet": wallet, "jobs": [], "hosts": [], "keys": []}
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO teams (team_id, name, owner_email, created_at, plan, max_members,
                                  billing_customer_id)
               VALUES (%s, %s, %s, %s, 'free', 5, %s)""",
            (team_id, team_id, f"{marker}@example.com", time.time(), wallet),
        )
        conn.commit()
    yield state
    with _pool.connection() as conn:
        for job_id in state["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        for host_id in state["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        conn.execute(
            "DELETE FROM scheduling_team_policies WHERE team_key=ANY(%s)", (state["keys"],)
        )
        conn.execute("DELETE FROM scheduling_team_state WHERE team_key=ANY(%s)", (state["keys"],))
        conn.execute("DELETE FROM teams WHERE team_id=%s", (team_id,))
        conn.commit()


def _submit(state, model, owner):
    job = scheduler.submit_job(
        name=f"quota-{uuid.uuid4().hex[:8]}",
        vram_needed_gb=1,
        gpu_model=price_test_gpu(model),
        owner=fund_test_owner(owner),
    )
    state["jobs"].append(job["job_id"])
    return job["job_id"]


def _spec(job_id):
    with _pool.connection() as conn:
        return conn.execute("SELECT spec FROM jobs WHERE job_id=%s", (job_id,)).fetchone()[0]


def test_admission_records_the_owners_team_and_tenant(team):
    job_id = _submit(team, f"RTX 4090 ident-{uuid.uuid4().hex[:6]}", team["wallet"])
    spec = _spec(job_id)
    assert spec["team_id"] == team["team_id"]
    assert spec["tenant_id"] == team["wallet"]
    assert "team_id" not in spec["admission_request"]  # server identity, not request input


def test_a_personal_wallet_records_its_tenant_and_no_team(team):
    owner = "personal-" + uuid.uuid4().hex[:8]
    spec = _spec(_submit(team, f"RTX 4090 solo-{uuid.uuid4().hex[:6]}", owner))
    assert spec["tenant_id"] == owner and "team_id" not in spec


def test_a_published_team_quota_limits_the_teams_admitted_jobs(team):
    model = f"RTX 4090 quota-{uuid.uuid4().hex[:6]}"
    key = "team:" + team["team_id"]
    team["keys"].append(key)
    with _pool.connection() as conn:
        publish_policy(conn, TeamPolicy(key=key, max_concurrent_jobs=1), created_by="quota-test")
        conn.commit()
    first, second = (_submit(team, model, team["wallet"]) for _ in range(2))
    host_id = f"h-quota-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        insert_schedulable_host(conn, host_id, model=model, gpus=2)
        conn.commit()
    team["hosts"].append(host_id)

    def place(conn):
        claimed = claim_next_job(conn, replica_id="quota-test", scope_gpu_models=[model])
        assert claimed is not None
        return claimed

    claims = [run_transaction(place) for _ in (first, second)]
    assert {c.job_id for c in claims} == {first, second}
    with _pool.connection() as conn:
        teams = {
            r[0]
            for r in conn.execute(
                "SELECT scheduling_team_key FROM jobs WHERE job_id=ANY(%s)", ([first, second],)
            ).fetchall()
        }
    assert teams == {key}

    def reserve(claimed):
        return run_transaction(
            lambda conn: reserve_and_bind(
                conn,
                job_id=claimed.job_id,
                claim_token=claimed.claim_token,
                replica_id="quota-test",
                host_id=host_id,
            )
        )

    reserve(claims[0])
    with pytest.raises(TeamQuotaConflict):
        reserve(claims[1])
