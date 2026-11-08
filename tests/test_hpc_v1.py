"""The read-only v1 HPC endpoints: the reconciler's copy, never the cluster.

/api/v1/hpc/clusters, /api/v1/hpc/jobs and /api/v1/hpc/jobs/{id} read what
workers/slurm_reconciler.py recorded. However often a dashboard or an agent
calls them, the cluster sees nothing (plan §5a). Operators see every job;
anyone else sees their own tenant's, and someone else's job is not-found.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import slurm_adapter as sa
from api import app
from db import pg_transaction
from routes.problem import PROBLEM_MEDIA_TYPE

client = TestClient(app)


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Persistent users and tokens, as the other v1 suites wire them."""
    import api as api_mod
    import oauth_service as oauth_mod
    import routes._deps as deps_mod
    import routes.auth as auth_mod

    monkeypatch.setattr(api_mod, "AUTH_REQUIRED", False)
    monkeypatch.setattr(deps_mod, "AUTH_REQUIRED", False)
    for mod in (api_mod, deps_mod, auth_mod):
        if hasattr(mod, "_USE_PERSISTENT_AUTH"):
            monkeypatch.setattr(mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(oauth_mod, "AUTH_CACHE_BACKEND", "memory")


@pytest.fixture(autouse=True)
def _no_cluster_contact(monkeypatch):
    """Any attempt to reach a cluster from these endpoints fails the test."""

    def _refuse(*_a, **_kw):
        raise AssertionError("a v1 HPC read tried to contact a Slurm cluster")

    monkeypatch.setattr(sa.SlurmRunner, "run", _refuse)
    monkeypatch.setattr(sa, "poll_cluster_jobs", _refuse)
    monkeypatch.setattr(sa, "accounting_window", _refuse)


def _register(admin: bool) -> tuple[dict, str]:
    from db import UserStore

    email = f"hpc-{uuid.uuid4().hex[:10]}@example.com"
    reg = client.post(
        "/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}
    ).json()
    if admin:
        UserStore.update_user(email, {"is_admin": 1})
    return {"Authorization": f"Bearer {reg['access_token']}"}, reg["user"]["customer_id"]


@pytest.fixture
def world():
    """A cluster with three jobs: two for tenant A (one settled), one for B."""
    cid = f"hv-{uuid.uuid4().hex[:8]}"
    admin, _ = _register(admin=True)
    tenant_a, cust_a = _register(admin=False)
    tenant_b, cust_b = _register(admin=False)
    ids = {"a_open": f"{cid}-1", "a_done": f"{cid}-2", "b_open": f"{cid}-3"}
    sa.register_slurm_job(ids["a_open"], "101", cluster_id=cid, tenant_id=cust_a)
    sa.register_slurm_job(ids["a_done"], "102", cluster_id=cid, tenant_id=cust_a)
    sa.register_slurm_job(ids["b_open"], "103", cluster_id=cid, tenant_id=cust_b)
    with pg_transaction() as conn:
        conn.execute(
            "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, poll_interval_seconds, facts) "
            "VALUES (%s, 'generic', 'HPC v1 test', 30, '{\"orphans\": [\"999\"]}'::jsonb)",
            (cid,),
        )
        conn.execute(
            "UPDATE slurm_job_mappings SET observed_state = 'running', slurm_state = 'RUNNING', "
            "last_observed_at = clock_timestamp(), last_changed_at = clock_timestamp() "
            "WHERE cluster_id = %s",
            (cid,),
        )
        conn.execute(
            "UPDATE slurm_job_mappings SET observed_state = 'completed', slurm_state = 'COMPLETED', "
            "exit_code = '0:0', terminal_at = clock_timestamp() WHERE xcelsior_job_id = %s",
            (ids["a_done"],),
        )
    yield {"cid": cid, "admin": admin, "a": tenant_a, "b": tenant_b, "ids": ids}
    with pg_transaction() as conn:
        conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
        conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cid,))


def _jobs(headers, **params):
    r = client.get("/api/v1/hpc/jobs", headers=headers, params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestClusters:
    def test_operators_see_registry_and_health(self, world):
        r = client.get("/api/v1/hpc/clusters", headers=world["admin"])
        assert r.status_code == 200, r.text
        [mine] = [c for c in r.json()["clusters"] if c["cluster_id"] == world["cid"]]
        assert mine["poll_interval_seconds"] == 30
        assert set(mine["health"]) >= {"transport_ok_at", "controller_ok_at", "accounting_ok_at"}
        assert mine["orphan_slurm_job_ids"] == ["999"]

    def test_everyone_else_is_refused_with_a_problem(self, world):
        r = client.get("/api/v1/hpc/clusters", headers=world["a"])
        assert r.status_code == 403
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


class TestJobs:
    def test_operators_see_every_tenant(self, world):
        ids = {
            j["xcelsior_job_id"]
            for j in _jobs(world["admin"], cluster_id=world["cid"], state="all")["jobs"]
        }
        assert ids == set(world["ids"].values())

    def test_a_tenant_sees_only_its_own(self, world):
        a = {
            j["xcelsior_job_id"]
            for j in _jobs(world["a"], cluster_id=world["cid"], state="all")["jobs"]
        }
        b = {
            j["xcelsior_job_id"]
            for j in _jobs(world["b"], cluster_id=world["cid"], state="all")["jobs"]
        }
        assert a == {world["ids"]["a_open"], world["ids"]["a_done"]}
        assert b == {world["ids"]["b_open"]}

    def test_state_filter(self, world):
        open_ = {j["xcelsior_job_id"] for j in _jobs(world["a"], cluster_id=world["cid"])["jobs"]}
        done = _jobs(world["a"], cluster_id=world["cid"], state="settled")["jobs"]
        assert open_ == {world["ids"]["a_open"]}
        assert [j["xcelsior_job_id"] for j in done] == [world["ids"]["a_done"]]
        assert done[0]["settled"] is True and done[0]["exit_code"] == "0:0"
        r = client.get("/api/v1/hpc/jobs", headers=world["a"], params={"state": "everything"})
        assert r.status_code == 422
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)

    def test_cursor_pages_through_without_repeats(self, world):
        first = _jobs(world["admin"], cluster_id=world["cid"], state="all", limit=2)
        assert len(first["jobs"]) == 2 and first["next_cursor"]
        second = _jobs(
            world["admin"],
            cluster_id=world["cid"],
            state="all",
            limit=2,
            cursor=first["next_cursor"],
        )
        assert len(second["jobs"]) == 1 and second["next_cursor"] is None
        seen = [j["xcelsior_job_id"] for j in first["jobs"] + second["jobs"]]
        assert sorted(seen) == sorted(world["ids"].values())

    def test_a_job_unconfirmed_for_three_intervals_is_stale(self, world):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET last_observed_at = clock_timestamp() - interval '10 minutes' "
                "WHERE xcelsior_job_id = %s",
                (world["ids"]["b_open"],),
            )
        by_id = {
            j["xcelsior_job_id"]: j
            for j in _jobs(world["admin"], cluster_id=world["cid"], state="all")["jobs"]
        }
        assert by_id[world["ids"]["b_open"]]["stale"] is True  # 600 s > 3 x 30 s
        assert by_id[world["ids"]["a_open"]]["stale"] is False
        assert by_id[world["ids"]["a_done"]]["stale"] is False  # settled is final


class TestOneJob:
    def test_owner_and_operator_can_read_it(self, world):
        for headers in (world["admin"], world["a"]):
            r = client.get(f"/api/v1/hpc/jobs/{world['ids']['a_open']}", headers=headers)
            assert r.status_code == 200, r.text
            job = r.json()["job"]
            assert job["observed_state"] == "running" and job["slurm_job_id"] == "101"

    def test_someone_elses_job_is_not_found(self, world):
        r = client.get(f"/api/v1/hpc/jobs/{world['ids']['a_open']}", headers=world["b"])
        assert r.status_code == 404
        assert r.json()["code"] == "hpc_job_not_found"
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


class TestAdminStatusReadsTheCache:
    """/api/slurm/status/{id} (the dashboard's refresh) used to poll Slurm on
    every click; it now reads the reconciler's copy."""

    def test_status_is_the_reconcilers_observation(self, world):
        r = client.get(
            "/api/slurm/status/101", headers=world["admin"], params={"cluster_id": world["cid"]}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "reconciler"
        assert body["observed_state"] == "running" and body["slurm_state"] == "RUNNING"
        assert body["xcelsior_job_id"] == world["ids"]["a_open"] and body["settled"] is False

    def test_an_id_on_two_clusters_needs_the_cluster(self, world):
        other = f"{world['cid']}-other"
        sa.register_slurm_job(f"{other}-1", "101", cluster_id=other)
        try:
            r = client.get("/api/slurm/status/101", headers=world["admin"])
            assert r.status_code == 409
            assert "[ambiguous_job_id]" in r.json()["error"]["message"]
        finally:
            with pg_transaction() as conn:
                conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (other,))

    def test_an_untracked_id_is_not_found(self, world):
        r = client.get("/api/slurm/status/987654321", headers=world["admin"])
        assert r.status_code == 404
