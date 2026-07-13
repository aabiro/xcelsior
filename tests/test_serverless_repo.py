"""Phase 1 — serverless repo + migration smoke tests."""

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from db import _get_pg_pool
from serverless.repo import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_IN_PROGRESS,
    JOB_STATUS_QUEUED,
    EndpointCreate,
    ServerlessRepo,
)


def _pg_tables() -> set[str]:
    pool = _get_pg_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        return {r["tablename"] if isinstance(r, dict) else r[0] for r in rows}


@pytest.fixture(scope="module")
def repo():
    required = {
        "serverless_endpoints",
        "serverless_workers",
        "serverless_jobs",
        "serverless_job_stream_events",
        "serverless_api_keys",
    }
    missing = required - _pg_tables()
    if missing:
        pytest.skip(f"Run alembic upgrade head first; missing tables: {missing}")
    return ServerlessRepo()


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def endpoint(repo: ServerlessRepo, owner_id: str):
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner_id,
            name="test-llama",
            mode="preset",
            model_ref="meta-llama/Llama-3.1-8B-Instruct",
            gpu_tier="rtx4090",
            min_workers=0,
            max_workers=2,
        )
    )
    yield ep
    repo.soft_delete_endpoint(ep["endpoint_id"], owner_id)


class TestMigration037:
    def test_serverless_tables_exist(self):
        tables = _pg_tables()
        for name in (
            "serverless_endpoints",
            "serverless_workers",
            "serverless_jobs",
            "serverless_job_stream_events",
            "serverless_api_keys",
        ):
            assert name in tables

    def test_migration_file_has_downgrade(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "migrations",
            "versions",
            "037_serverless_endpoints_workers_jobs_keys.py",
        )
        with open(path) as f:
            content = f.read()
        assert "def downgrade" in content
        assert "DROP TABLE IF EXISTS serverless_endpoints" in content

    def test_migration_037_chain(self):
        import importlib.util
        import pathlib

        def _load(name: str):
            path = pathlib.Path(__file__).resolve().parent.parent / "migrations" / "versions" / name
            spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        m037 = _load("037_serverless_endpoints_workers_jobs_keys.py")
        assert m037.revision == "037"
        assert m037.down_revision == "036"

    def test_claim_query_uses_skip_locked(self):
        path = os.path.join(os.path.dirname(__file__), "..", "serverless", "repo.py")
        with open(path) as f:
            content = f.read()
        assert "FOR UPDATE SKIP LOCKED" in content


class TestMigration038:
    def test_migration_file_importable_and_chained(self):
        import importlib.util
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "migrations"
            / "versions"
            / "038_serverless_limits_webhooks.py"
        )
        spec = importlib.util.spec_from_file_location("038_serverless_limits_webhooks", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.revision == "038"
        assert mod.down_revision == "037"
        assert hasattr(mod, "upgrade")
        assert hasattr(mod, "downgrade")

    def test_migration_file_has_downgrade(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "migrations",
            "versions",
            "038_serverless_limits_webhooks.py",
        )
        with open(path) as f:
            content = f.read()
        assert "DROP INDEX IF EXISTS idx_serverless_jobs_webhook_retry" in content
        assert "DROP COLUMN IF EXISTS max_queue_size" in content

    def test_max_queue_size_column_exists(self):
        pool = _get_pg_pool()
        with pool.connection() as conn:
            row = conn.execute(
                """
                SELECT column_name, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'serverless_endpoints'
                  AND column_name = 'max_queue_size'
                """
            ).fetchone()
        assert row is not None
        name = row["column_name"] if isinstance(row, dict) else row[0]
        assert name == "max_queue_size"


class TestMigration053:
    @pytest.mark.parametrize(
        ("table_name", "column_name"),
        [
            ("serverless_workers", "billing_exempt"),
            ("serverless_workers", "warm_expires_at"),
            ("serverless_jobs", "billing_exempt"),
        ],
    )
    def test_dashboard_test_billing_columns_exist(self, table_name: str, column_name: str):
        pool = _get_pg_pool()
        with pool.connection() as conn:
            row = conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                  AND column_name = %s
                """,
                (table_name, column_name),
            ).fetchone()
        assert row is not None


class TestServerlessRepoEndpoints:
    def test_create_list_get_patch_delete(self, repo: ServerlessRepo, owner_id: str):
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner_id, name="ep-a", model_ref="model-a")
        )
        listed = repo.list_endpoints(owner_id)
        assert any(r["endpoint_id"] == ep["endpoint_id"] for r in listed)

        got = repo.get_endpoint(ep["endpoint_id"], owner_id=owner_id)
        assert got is not None
        assert got["model_ref"] == "model-a"

        patched = repo.patch_endpoint(
            ep["endpoint_id"], owner_id, {"name": "renamed", "max_workers": 8}
        )
        assert patched is not None
        assert patched["name"] == "renamed"
        assert patched["max_workers"] == 8

        assert repo.soft_delete_endpoint(ep["endpoint_id"], owner_id)
        assert repo.get_endpoint(ep["endpoint_id"], owner_id=owner_id) is None

    def test_owner_isolation_404_semantics(self, repo: ServerlessRepo, endpoint: dict):
        other = f"cust-{uuid.uuid4().hex[:12]}"
        assert repo.get_endpoint(endpoint["endpoint_id"], owner_id=other) is None


class TestServerlessRepoQueue:
    def test_enqueue_claim_complete(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        ep_id = endpoint["endpoint_id"]
        job = repo.enqueue_job(ep_id, owner_id, {"prompt": "hello"})
        assert job["status"] == JOB_STATUS_QUEUED
        assert repo.queue_depth(ep_id) == 1

        worker = repo.create_worker(ep_id, scheduler_job_id=f"job-{uuid.uuid4().hex[:8]}")
        claimed = repo.claim_next_job(ep_id, worker["worker_id"])
        assert claimed is not None
        assert claimed["job_id"] == job["job_id"]
        assert claimed["status"] == JOB_STATUS_IN_PROGRESS
        assert repo.queue_depth(ep_id) == 0

        done = repo.complete_job(
            job["job_id"],
            output={"text": "world"},
            gpu_seconds=12,
            cold_start_seconds=3,
            cost_cad=0.01,
        )
        assert done is not None
        assert done["status"] == JOB_STATUS_COMPLETED
        assert done["gpu_seconds"] == 12
        assert done["cold_start_seconds"] == 3

    def test_billing_exempt_job_persists(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        job = repo.enqueue_job(
            endpoint["endpoint_id"],
            owner_id,
            {"prompt": "free dashboard test"},
            billing_exempt=True,
        )
        assert job["billing_exempt"] is True

    def test_idempotency_returns_same_job(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        ep_id = endpoint["endpoint_id"]
        key = f"idemp-{uuid.uuid4().hex[:8]}"
        j1 = repo.enqueue_job(ep_id, owner_id, {"n": 1}, idempotency_key=key)
        j2 = repo.enqueue_job(ep_id, owner_id, {"n": 2}, idempotency_key=key)
        assert j1["job_id"] == j2["job_id"]
        assert repo.queue_depth(ep_id) == 1

    def test_cancel_queued_job(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        ep_id = endpoint["endpoint_id"]
        job = repo.enqueue_job(ep_id, owner_id, {"x": 1})
        cancelled = repo.cancel_job(job["job_id"], ep_id)
        assert cancelled is not None
        assert cancelled["status"] == JOB_STATUS_CANCELLED

    def test_max_queue_wait_positive(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        ep_id = endpoint["endpoint_id"]
        repo.enqueue_job(ep_id, owner_id, {"wait": True})
        assert repo.max_queue_wait_sec(ep_id) >= 0.0


class TestServerlessRepoStreamsAndKeys:
    def test_stream_events_monotonic_seq(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        ep_id = endpoint["endpoint_id"]
        job = repo.enqueue_job(ep_id, owner_id, {})
        e1 = repo.append_stream_event(job["job_id"], "token", {"t": "a"})
        e2 = repo.append_stream_event(job["job_id"], "token", {"t": "b"})
        assert e2["seq_no"] > e1["seq_no"]
        events = repo.list_stream_events(job["job_id"])
        assert len(events) == 2

    def test_api_key_hash_lookup_and_revoke(self, repo: ServerlessRepo, endpoint: dict, owner_id: str):
        ep_id = endpoint["endpoint_id"]
        created = repo.create_api_key(
            owner_id,
            key_prefix="xcel_test",
            key_hash="hash-" + uuid.uuid4().hex,
            endpoint_id=ep_id,
        )
        by_hash = repo.get_api_key_by_hash(created["key_hash"])
        assert by_hash is not None
        assert by_hash["key_id"] == created["key_id"]

        keys = repo.list_api_keys(owner_id, endpoint_id=ep_id)
        assert len(keys) >= 1

        assert repo.revoke_api_key(created["key_id"], owner_id)
        assert repo.get_api_key_by_hash(created["key_hash"]) is None
