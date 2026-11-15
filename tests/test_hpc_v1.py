"""The read-only v1 HPC endpoints: the reconciler's copy, never the cluster.

/api/v1/hpc/clusters, /api/v1/hpc/jobs and /api/v1/hpc/jobs/{id} read what
workers/slurm_reconciler.py recorded. However often a dashboard or an agent
calls them, the cluster sees nothing (plan §5a). Operators see every job;
anyone else sees their own tenant's, and someone else's job is not-found.
"""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

import slurm_adapter as sa
from api import app
from db import pg_connection, pg_transaction
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
    monkeypatch.setattr(sa, "get_slurm_job_output", _refuse)


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
            "exit_code = '0:0', derived_exit_code = '0:0', terminal_at = clock_timestamp() "
            "WHERE xcelsior_job_id = %s",
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


class TestVerdictAndProgress:
    def test_a_settled_job_carries_its_verdict(self, world):
        jobs = {
            j["xcelsior_job_id"]: j
            for j in _jobs(world["admin"], cluster_id=world["cid"], state="all")["jobs"]
        }
        done, running = jobs[world["ids"]["a_done"]], jobs[world["ids"]["a_open"]]
        assert done["verdict"] == "succeeded" and done["derived_exit_code"] == "0:0"
        assert running["verdict"] is None  # open: no verdict yet

    def test_scheduler_progress_is_elapsed_against_the_limit(self, world):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET time_limit_seconds = 600, "
                "started_at = clock_timestamp() - interval '60 seconds' WHERE xcelsior_job_id = %s",
                (world["ids"]["a_open"],),
            )
        job = client.get(f"/api/v1/hpc/jobs/{world['ids']['a_open']}", headers=world["a"]).json()[
            "job"
        ]
        progress = job["scheduler_progress"]
        assert progress["time_limit_seconds"] == 600 and 60 <= progress["elapsed_seconds"] < 90
        assert 0.1 <= progress["fraction"] < 0.15

    def test_a_job_that_left_the_controller_says_it_awaits_accounting(self, world):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET left_controller_at = clock_timestamp() "
                "WHERE xcelsior_job_id = %s",
                (world["ids"]["b_open"],),
            )
        job = client.get(f"/api/v1/hpc/jobs/{world['ids']['b_open']}", headers=world["b"]).json()[
            "job"
        ]
        assert job["awaiting_accounting_since"]

    def test_the_jobs_own_report_appears_once_fetched(self, world):
        url = f"/api/v1/hpc/jobs/{world['ids']['a_open']}"
        assert client.get(url, headers=world["a"]).json()["job"]["reported_progress"] is None
        with pg_transaction() as conn:  # what the reconciler's pass writes
            conn.execute(
                "UPDATE slurm_job_outputs SET content = %s, attempted_at = clock_timestamp(), "
                "fetched_at = clock_timestamp() WHERE mapping_id IN (SELECT mapping_id FROM slurm_job_mappings WHERE xcelsior_job_id = %s) "
                "AND stream = 'progress'",
                (
                    '{"fraction": 0.25, "message": "step 250", "updated_at": %d}' % time.time(),
                    world["ids"]["a_open"],
                ),
            )
        report = client.get(url, headers=world["a"]).json()["job"]["reported_progress"]
        assert report["valid"] and report["fraction"] == 0.25 and report["source"] == "job"


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


class TestJobOutput:
    """/api/v1/hpc/jobs/{id}/output reads the cache the reconciler fills."""

    def test_the_first_read_queues_a_fetch_and_later_reads_see_it(self, world):
        url = f"/api/v1/hpc/jobs/{world['ids']['a_open']}/output"
        r = client.get(url, headers=world["a"])
        assert r.status_code == 200, r.text
        out = r.json()["output"]
        assert out["content"] == "" and out["refresh_pending"] and out["fetched_at"] is None
        assert out["poll_interval_seconds"] == 30

        # What the reconciler's pass writes once it has read the file.
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_outputs SET content = 'step 10 loss 0.4', attempted_at = "
                "clock_timestamp(), fetched_at = clock_timestamp() "
                "WHERE mapping_id IN (SELECT mapping_id FROM slurm_job_mappings WHERE xcelsior_job_id = %s)",
                (world["ids"]["a_open"],),
            )
        out = client.get(url, headers=world["admin"]).json()["output"]
        assert out["content"] == "step 10 loss 0.4" and not out["refresh_pending"]
        assert out["fetched_at"] and out["stream"] == "out" and not out["final"]

    def test_someone_elses_output_is_not_found(self, world):
        r = client.get(f"/api/v1/hpc/jobs/{world['ids']['a_open']}/output", headers=world["b"])
        assert r.status_code == 404 and r.json()["code"] == "hpc_job_not_found"
        with pg_connection() as conn:  # and the refused read filed no request
            assert (
                conn.execute(
                    "SELECT count(*) FROM slurm_job_outputs WHERE mapping_id IN (SELECT mapping_id FROM slurm_job_mappings WHERE xcelsior_job_id = %s)",
                    (world["ids"]["a_open"],),
                ).fetchone()[0]
                == 0
            )

    def test_stream_must_be_out_or_err(self, world):
        r = client.get(
            f"/api/v1/hpc/jobs/{world['ids']['a_open']}/output",
            headers=world["a"],
            params={"stream": "both"},
        )
        assert r.status_code == 422
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)


_KEY = "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"


class TestClusterRegistration:
    """POST /api/v1/hpc/clusters and its connection check: admin only."""

    @pytest.fixture
    def cid(self):
        cid = f"hr-{uuid.uuid4().hex[:8]}"
        yield cid
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cid,))

    @pytest.fixture
    def key_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SLURM_KEY_DIR", str(tmp_path))
        return tmp_path

    def _body(self, cid, **kw):
        body = {
            "profile_key": "generic",
            "cluster_id": cid,
            "ssh_host": "203.0.113.7",
            "remote_user": "xclslurm",
            "known_hosts_line": f"203.0.113.7 ssh-ed25519 {_KEY}",
            "credential_ref": "env:XCELSIOR_SLURM_KEY_LAB",
        }
        body.update(kw)
        return body

    def test_an_admin_registers_it_disabled_and_sees_its_fingerprint(self, world, cid):
        r = client.post("/api/v1/hpc/clusters", headers=world["admin"], json=self._body(cid))
        assert r.status_code == 201, r.text
        assert r.json()["cluster"]["transport"] == "registry"
        listed = client.get("/api/v1/hpc/clusters", headers=world["admin"]).json()["clusters"]
        [mine] = [c for c in listed if c["cluster_id"] == cid]
        assert mine["enabled"] is False  # off until a connection check proves it
        transport = mine["transport"]
        assert transport["kind"] == "registry" and transport["ssh_host"] == "203.0.113.7"
        assert transport["host_key_fingerprint"].startswith("SHA256:")
        assert transport["credential_ref"] == "env:XCELSIOR_SLURM_KEY_LAB"
        assert _KEY not in r.text + str(listed)  # the key line itself is never echoed

    def test_a_key_file_must_be_in_the_key_directory(self, world, cid, key_dir):
        inside = client.post(
            "/api/v1/hpc/clusters",
            headers=world["admin"],
            json=self._body(cid, credential_ref=f"file:{key_dir}/lab.key"),
        )
        assert inside.status_code == 201, inside.text
        for ref in (
            "file:/etc/ssh/ssh_host_ed25519_key",
            f"file:{key_dir}/../escape.key",
            "env:AWS_SECRET_ACCESS_KEY",
            "env:XCELSIOR_STRIPE_SECRET_KEY",
        ):
            r = client.post(
                "/api/v1/hpc/clusters",
                headers=world["admin"],
                json=self._body(cid, credential_ref=ref),
            )
            assert r.status_code == 422 and r.json()["code"] == "credential_ref_not_allowed", ref

    @pytest.mark.parametrize(
        "change,code",
        [
            ({"known_hosts_line": "@cert-authority * ssh-ed25519 AAAA"}, "invalid_host_key"),
            ({"profile_key": "no-such-profile"}, "unknown_profile"),
        ],
    )
    def test_a_bad_registration_is_a_problem(self, world, cid, change, code):
        r = client.post(
            "/api/v1/hpc/clusters", headers=world["admin"], json=self._body(cid, **change)
        )
        assert r.status_code == 422 and r.json()["code"] == code
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)

    def test_only_a_platform_admin(self, world, cid):
        r = client.post("/api/v1/hpc/clusters", headers=world["a"], json=self._body(cid))
        assert r.status_code == 403 and r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
        r = client.post(f"/api/v1/hpc/clusters/{world['cid']}/connection-check", headers=world["a"])
        assert r.status_code == 403

    def test_the_connection_check_reports_what_it_found(self, world, monkeypatch):
        import slurm_reconciler as rc

        # An unknown cluster is found missing before anything is contacted.
        missing = client.post(
            "/api/v1/hpc/clusters/no-such/connection-check", headers=world["admin"]
        )
        assert missing.status_code == 404 and missing.json()["code"] == "hpc_cluster_not_found"
        found = {"ok": False, "error_code": "host_key_mismatch", "error": "refused"}
        monkeypatch.setattr(rc, "check_cluster_connection", lambda cluster_id: found)
        r = client.post(
            f"/api/v1/hpc/clusters/{world['cid']}/connection-check", headers=world["admin"]
        )
        assert r.status_code == 200 and r.json()["check"] == found


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


# ── Submitting and cancelling (blueprint 3.4) ────────────────────────


@pytest.fixture
def hpc(world, monkeypatch):
    """A registered cluster to submit to, and cleanup for the jobs submitted."""
    cid = f"hs-{uuid.uuid4().hex[:8]}"
    with pg_transaction() as conn:
        conn.execute(
            "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, enabled, "
            "marketplace, agent_access, poll_interval_seconds) "
            "VALUES (%s, 'generic', 'submit test', true, true, true, 30)",
            (cid,),
        )
    monkeypatch.setenv("XCELSIOR_SLURM_PLACEMENT", "shadow")
    made: list[str] = []
    ctx = {**world, "cid": cid, "made": made}
    yield ctx
    with pg_transaction() as conn:
        for job_id in made:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (job_id,))
            conn.execute("DELETE FROM slurm_job_mappings WHERE xcelsior_job_id = %s", (job_id,))
            conn.execute("DELETE FROM job_attempts WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cid,))


def _submit(ctx, headers, **body):
    payload = {"cluster_id": ctx["cid"], "command": "nvidia-smi -L", "num_gpus": 1, **body}
    r = client.post("/api/v1/hpc/jobs", headers=headers, json=payload)
    if r.status_code == 202:
        ctx["made"].append(r.json()["job"]["xcelsior_job_id"])
    return r


def _problem(r, status, code):
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert r.json()["code"] == code


class TestSubmit:
    def test_an_admin_queues_a_job_and_can_poll_it_before_slurm_has_it(self, hpc):
        r = _submit(hpc, hpc["admin"])
        assert r.status_code == 202, r.text
        job = r.json()["job"]
        assert job["in_slurm"] is False and job["cluster_id"] == hpc["cid"]
        assert job["job_status"] == "queued" and job["submission_state"] == "not_placed"
        assert r.headers["Location"] == f"/api/v1/hpc/jobs/{job['xcelsior_job_id']}"
        polled = client.get(r.headers["Location"], headers=hpc["admin"])
        assert polled.status_code == 200 and polled.json()["job"]["in_slurm"] is False
        with pg_connection() as conn:
            spec_cluster, tenant = conn.execute(
                "SELECT spec->>'hpc_cluster_id', tenant_id FROM jobs WHERE job_id = %s",
                (job["xcelsior_job_id"],),
            ).fetchone()
        assert spec_cluster == hpc["cid"] and tenant

    def test_a_retry_with_the_same_key_returns_the_same_job(self, hpc):
        key = f"key-{uuid.uuid4().hex}"
        first = _submit(hpc, hpc["admin"], idempotency_key=key)
        again = _submit(hpc, hpc["admin"], idempotency_key=key)
        assert first.status_code == again.status_code == 202
        assert first.json()["job"]["xcelsior_job_id"] == again.json()["job"]["xcelsior_job_id"]
        other = client.post(
            "/api/v1/hpc/jobs",
            headers={**hpc["admin"], "Idempotency-Key": key},
            json={"cluster_id": hpc["cid"], "command": "something else", "num_gpus": 1},
        )
        _problem(other, 409, "idempotency_conflict")

    def test_shadow_mode_takes_new_work_from_admins_only(self, hpc):
        _problem(_submit(hpc, hpc["a"]), 403, "hpc_submission_not_allowed")

    def test_in_a_charging_mode_a_tenant_needs_a_balance(self, hpc, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SLURM_PLACEMENT", "all")
        _problem(_submit(hpc, hpc["a"]), 402, "insufficient_balance")
        me = client.get("/api/auth/me", headers=hpc["a"]).json()
        customer = (me.get("user") or me).get("customer_id")
        with pg_transaction() as conn:  # registration made the wallet, empty
            conn.execute(
                "INSERT INTO wallets (customer_id, balance_micros, status, created_at, updated_at) "
                "VALUES (%s, 5000000, 'active', 1794000000, 1794000000) "
                "ON CONFLICT (customer_id) DO UPDATE SET balance_micros = 5000000",
                (customer,),
            )
        assert _submit(hpc, hpc["a"]).status_code == 202

    def test_placement_off_is_unavailable(self, hpc, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SLURM_PLACEMENT", "off")
        _problem(_submit(hpc, hpc["admin"]), 503, "hpc_submission_unavailable")

    def test_a_disabled_cluster_is_a_conflict(self, hpc):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_clusters SET enabled = false WHERE cluster_id = %s", (hpc["cid"],)
            )
        _problem(_submit(hpc, hpc["admin"]), 409, "hpc_cluster_disabled")

    def test_an_unknown_cluster_is_not_found(self, hpc):
        _problem(
            _submit(hpc, hpc["admin"], cluster_id="no-such-cluster"), 404, "hpc_cluster_not_found"
        )


class TestRefusedClusters:
    def test_an_alliance_cluster_is_refused_registered_or_not_and_whatever_the_mode(
        self, hpc, monkeypatch
    ):
        monkeypatch.setenv("XCELSIOR_SLURM_PLACEMENT", "off")
        # "nibi" is an Alliance profile; nobody registered it here.
        _problem(_submit(hpc, hpc["admin"], cluster_id="nibi"), 403, "research_only_cluster")
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_clusters SET research_only = true, marketplace = false, "
                "agent_access = false WHERE cluster_id = %s",
                (hpc["cid"],),
            )
        _problem(_submit(hpc, hpc["admin"]), 403, "research_only_cluster")

    def test_an_agent_is_refused_where_the_cluster_takes_no_agents(self, hpc, monkeypatch):
        import routes._deps as deps

        monkeypatch.setattr(deps, "_is_interactive_human", lambda user: False)
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_clusters SET agent_access = false WHERE cluster_id = %s",
                (hpc["cid"],),
            )
        _problem(_submit(hpc, hpc["admin"]), 403, "agent_access_denied")
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_clusters SET agent_access = true WHERE cluster_id = %s",
                (hpc["cid"],),
            )
        assert _submit(hpc, hpc["admin"]).status_code == 202

    def test_a_tenant_is_refused_where_the_cluster_is_not_offered(self, hpc, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SLURM_PLACEMENT", "all")
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_clusters SET marketplace = false WHERE cluster_id = %s",
                (hpc["cid"],),
            )
        _problem(_submit(hpc, hpc["a"]), 403, "hpc_cluster_not_offered")


class TestCancel:
    def test_a_queued_job_ends_at_once_and_a_second_cancel_is_harmless(self, hpc):
        job_id = _submit(hpc, hpc["admin"]).json()["job"]["xcelsior_job_id"]
        r = client.delete(f"/api/v1/hpc/jobs/{job_id}", headers=hpc["admin"])
        assert r.status_code == 202 and r.json()["cancel"] == "terminated", r.text
        assert r.json()["job"]["job_status"] == "terminated" and r.json()["job"]["settled"]
        again = client.delete(f"/api/v1/hpc/jobs/{job_id}", headers=hpc["admin"])
        assert again.status_code == 202 and again.json()["cancel"] == "already_finished"

    def test_a_job_in_slurm_is_cancelled_through_the_reconciler(self, hpc):
        job_id = _submit(hpc, hpc["admin"]).json()["job"]["xcelsior_job_id"]
        # What placement and the submitter leave behind for a job in Slurm.
        with pg_transaction() as conn:
            attempt_id = conn.execute(
                "INSERT INTO job_attempts (job_id, attempt_number, status, fencing_token, "
                "job_generation, created_by) VALUES (%s, 1, 'reserved', "
                "nextval('placement_fencing_token_seq'), 1, 'test') RETURNING attempt_id",
                (job_id,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE jobs SET active_attempt_id = %s, status = 'assigned', phase = 'scheduled' "
                "WHERE job_id = %s",
                (attempt_id, job_id),
            )
        sa.register_slurm_job(
            job_id,
            "8123",
            cluster_id=hpc["cid"],
            attempt_id=str(attempt_id),
            submission_name=f"{job_id}-a1",
        )
        r = client.delete(f"/api/v1/hpc/jobs/{job_id}", headers=hpc["admin"])
        assert r.status_code == 202 and r.json()["cancel"] == "requested", r.text
        assert r.json()["job"]["in_slurm"] is True
        with pg_connection() as conn:
            desired, intent = conn.execute(
                "SELECT desired_state, lifecycle_intent FROM jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
            mapping_desired = conn.execute(
                "SELECT desired_state FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
                (job_id,),
            ).fetchone()[0]
        assert (desired, intent, mapping_desired) == ("stopped", "cancel", "cancelled")

    def test_someone_elses_job_is_not_found(self, hpc):
        job_id = _submit(hpc, hpc["admin"]).json()["job"]["xcelsior_job_id"]
        _problem(
            client.delete(f"/api/v1/hpc/jobs/{job_id}", headers=hpc["b"]), 404, "hpc_job_not_found"
        )


# ── Fleet figures, the billing ledger and the submit log ──────────────
# What the operator console adds to the registry: per-cluster job counts, the
# ledger with its daily drift, and the submissions that never became a job.


@pytest.fixture
def ledger(world):
    """The world's settled job with two billed runs (a requeue), two drift
    days (one over tolerance) and two submissions (one rejected)."""
    import json

    from psycopg.types.json import Jsonb

    cid, done = world["cid"], world["ids"]["a_done"]
    jobs = [f"{cid}-s1", f"{cid}-s2"]
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE slurm_clusters SET rate_card = %s, failure_counts = %s WHERE cluster_id = %s",
            (
                Jsonb({"gpu_hour_micros": 360_000, "gpu_model": "RTX 3060", "junk": "x"}),
                Jsonb({"pass:transport_timeout": 2}),
                cid,
            ),
        )
        mapping_id = conn.execute(
            "SELECT mapping_id FROM slurm_job_mappings WHERE xcelsior_job_id = %s", (done,)
        ).fetchone()[0]
        for run, state, elapsed, rule, cost, age in (
            (0, "REQUEUED", 75, "requeued_run", 7500, "2 hours"),
            (1, "COMPLETED", 301, "ran_to_end", 30100, "1 hour"),
        ):
            conn.execute(
                """
                INSERT INTO slurm_billing_segments
                    (mapping_id, run_index, cluster_id, slurm_job_id, xcelsior_job_id,
                     slurm_state, started_at, ended_at, elapsed_sec, gpu_count, billable_sec,
                     billing_rule, gpu_seconds, rate_micros_per_gpu_hour, shadow_cost_micros)
                VALUES (%s, %s, %s, '102', %s, %s,
                        clock_timestamp() - %s::interval - make_interval(secs => %s),
                        clock_timestamp() - %s::interval, %s, 1, %s, %s, %s, 360000, %s)
                """,
                (
                    mapping_id,
                    run,
                    cid,
                    done,
                    state,
                    age,
                    elapsed,
                    age,
                    elapsed,
                    elapsed,
                    rule,
                    elapsed,
                    cost,
                ),
            )
        for back, sacct, ledger_s, pct in ((0, 376, 376, 0), (1, 400, 380, 5)):
            conn.execute(
                """
                INSERT INTO slurm_billing_drift
                    (cluster_id, day, sacct_gpu_seconds, ledger_gpu_seconds, sacct_records,
                     ledger_records, drift_pct, unmatched)
                VALUES (%s, current_date - %s::int, %s, %s, 2, 2, %s, %s)
                """,
                (cid, back, sacct, ledger_s, pct, Jsonb({"only_sacct": ["77@1"]} if pct else {})),
            )
        for job_id, state, code, err in (
            (jobs[0], "submitted", None, None),
            (jobs[1], "rejected", "invalid_partition", "sbatch: error: invalid partition\nmore"),
        ):
            conn.execute(
                "INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, "
                "desired_state, effective_priority, queued_at, spec) "
                "VALUES (%s, 'assigned', 10, %s, %s, 'scheduled', 'running', 10, "
                "clock_timestamp(), %s)",
                (job_id, time.time(), json.dumps({"job_id": job_id}), Jsonb({"name": job_id})),
            )
            attempt_id = conn.execute(
                "INSERT INTO job_attempts (job_id, attempt_number, status, fencing_token, "
                "job_generation, created_by) VALUES (%s, 1, 'reserved', "
                "nextval('placement_fencing_token_seq'), 1, 'test') RETURNING attempt_id",
                (job_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO slurm_submissions (submission_name, cluster_id, xcelsior_job_id, "
                "attempt_id, state, error_code, error, sbatch_calls) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 1)",
                (f"{job_id}-a1", cid, job_id, attempt_id, state, code, err),
            )
    yield world
    with pg_transaction() as conn:
        conn.execute("DELETE FROM slurm_submissions WHERE cluster_id = %s", (cid,))
        conn.execute("DELETE FROM slurm_billing_drift WHERE cluster_id = %s", (cid,))
        conn.execute("DELETE FROM slurm_billing_segments WHERE cluster_id = %s", (cid,))
        for job_id in jobs:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


class TestFleetFigures:
    def test_clusters_count_their_open_jobs(self, world):
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET observed_state = 'pending', "
                "last_observed_at = clock_timestamp() - interval '10 minutes' "
                "WHERE xcelsior_job_id = %s",
                (world["ids"]["b_open"],),
            )
        r = client.get("/api/v1/hpc/clusters", headers=world["admin"])
        [mine] = [c for c in r.json()["clusters"] if c["cluster_id"] == world["cid"]]
        # Two open (the settled one is not counted): one running, one pending and stale.
        assert mine["open_jobs"] == {
            "total": 2,
            "running": 1,
            "pending": 1,
            "stale": 1,
            "awaiting_accounting": 0,
        }
        assert mine["rate_card"] == {} and mine["failure_counts"] == {}
        assert mine["suspend_holds_gres"] is True

    def test_rate_card_and_failures_are_reported(self, ledger):
        r = client.get("/api/v1/hpc/clusters", headers=ledger["admin"])
        [mine] = [c for c in r.json()["clusters"] if c["cluster_id"] == ledger["cid"]]
        # Only the keys the biller reads: an unknown one is not passed through.
        assert mine["rate_card"] == {"gpu_hour_micros": 360_000, "gpu_model": "RTX 3060"}
        assert mine["failure_counts"] == {"pass:transport_timeout": 2}


class TestBillingLedger:
    def _get(self, headers, **params):
        return client.get("/api/v1/hpc/billing", headers=headers, params=params)

    def test_segments_drift_and_totals(self, ledger):
        r = self._get(ledger["admin"], cluster_id=ledger["cid"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tolerance_pct"] == 1.0 and body["next_cursor"] is None
        # Newest run first; each run of a requeued job is its own record.
        assert [(s["run_index"], s["billing_rule"]) for s in body["segments"]] == [
            (1, "ran_to_end"),
            (0, "requeued_run"),
        ]
        first = body["segments"][0]
        assert first["xcelsior_job_id"] == ledger["ids"]["a_done"]
        assert first["billable_seconds"] == 301 and first["shadow_cost_micros"] == 30100
        assert first["charged"] is False and first["meter_id"] is None
        assert body["totals"] == {
            "segments": 2,
            "jobs": 1,
            "gpu_seconds": 376,
            "billable_seconds": 376,
            "shadow_cost_micros": 37600,
            "charged_micros": 0,
            "charged_segments": 0,
            "unpriced_segments": 0,
        }
        today, yesterday = body["drift"]
        assert today["within_tolerance"] is True and today["drift_pct"] == 0
        assert yesterday["within_tolerance"] is False and yesterday["drift_pct"] == 5
        assert yesterday["unmatched"] == {"only_sacct": ["77@1"]}
        assert (yesterday["sacct_gpu_seconds"], yesterday["ledger_gpu_seconds"]) == (400, 380)

    def test_totals_cover_the_window_not_the_page(self, ledger):
        first = self._get(ledger["admin"], cluster_id=ledger["cid"], limit=1).json()
        assert len(first["segments"]) == 1 and first["next_cursor"]
        assert first["totals"]["segments"] == 2
        second = self._get(
            ledger["admin"], cluster_id=ledger["cid"], limit=1, cursor=first["next_cursor"]
        ).json()
        assert [s["run_index"] for s in second["segments"]] == [0]
        assert second["next_cursor"] is None

    def test_another_cluster_sees_none_of_it(self, ledger):
        body = self._get(ledger["admin"], cluster_id="no-such-cluster").json()
        assert body["segments"] == [] and body["drift"] == []
        assert body["totals"]["segments"] == 0 and body["totals"]["shadow_cost_micros"] == 0

    def test_tenants_are_refused_and_bad_input_is_a_problem(self, ledger):
        r = self._get(ledger["a"])
        assert r.status_code == 403
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
        assert self._get(ledger["admin"], cursor="!!").status_code == 422
        assert self._get(ledger["admin"], days=0).status_code == 422


class TestSubmitLog:
    def _get(self, headers, **params):
        return client.get("/api/v1/hpc/submissions", headers=headers, params=params)

    def test_lists_what_never_became_a_job(self, ledger):
        r = self._get(ledger["admin"], cluster_id=ledger["cid"])
        assert r.status_code == 200, r.text
        body = r.json()
        by_state = {s["state"]: s for s in body["submissions"]}
        assert set(by_state) == {"submitted", "rejected"}
        rejected = by_state["rejected"]
        assert rejected["error_code"] == "invalid_partition"
        # sbatch's stderr is cut to its first line.
        assert rejected["error"] == "sbatch: error: invalid partition"
        assert rejected["slurm_job_id"] is None and rejected["sbatch_calls"] == 1
        assert body["counts"]["rejected"] == 1 and body["counts"]["submitted"] == 1
        assert body["counts"]["lost"] == 0

    def test_state_filter(self, ledger):
        only = self._get(ledger["admin"], cluster_id=ledger["cid"], state="rejected").json()
        assert [s["state"] for s in only["submissions"]] == ["rejected"]
        # Nothing here is still waiting on the submitter.
        open_ = self._get(ledger["admin"], cluster_id=ledger["cid"], state="open").json()
        assert open_["submissions"] == []
        # The counts ignore the state filter, so the tabs keep their numbers.
        assert open_["counts"]["rejected"] == 1
        r = self._get(ledger["admin"], state="everything")
        assert r.status_code == 422
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)

    def test_tenants_are_refused(self, ledger):
        r = self._get(ledger["a"])
        assert r.status_code == 403
        assert r.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
