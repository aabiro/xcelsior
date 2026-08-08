"""What a promotion would copy, before anything copies.

A0 of `docs/artifact-promotion-plan.md`. Gate P3 asks for artifact→volume
promotion, and `promote_artifact_to_volume` does not exist — this is the first
piece of it: the manifest, so the shape can be reviewed and so the tool layer
has something truthful to show a user *before* a 40 GB copy starts rather than
after.

Nothing here copies, holds, or writes. That is the point of doing it first.

## The two properties worth testing at this stage

**Tenant scoping is in the query.** `job_id` is caller-supplied and a manifest
is a list of a tenant's file names and sizes. Resolving first and filtering
afterwards is how a read becomes a disclosure, so the assertion is that a
foreign job resolves to *nothing* rather than to something later discarded.

**The digest makes a retry converge.** The idempotency key defaults to a hash of
the resolved set, so the same job with the same artifacts promotes once however
many times it is asked. A job that has since produced new artifacts hashes
differently, which is a *different* promotion rather than a silent no-op against
a stale file list — the failure mode being a user who promotes, trains more, and
promotes again, and gets the first set twice.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = (
            _c.execute("SELECT to_regclass('storage.artifacts')").fetchone()[0] is not None
            and _c.execute("SELECT to_regclass('volume_promotions')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 102")


#: `storage.artifacts` is stricter than the tables elsewhere in this codebase:
#: `artifact_id` is a **uuid**, `retain_until`/`created_at` are **timestamptz**
#: (not the float epoch used by `gpu_allocations` and friends), and
#: `content_type`/`retention_class` are NOT NULL. Every one of those was got
#: wrong by writing the INSERT from the convention next door instead of reading
#: the schema, so the columns are spelled out here rather than assumed.
_INSERT_ARTIFACT = """
    INSERT INTO storage.artifacts
        (artifact_id, tenant_id, job_id, artifact_type, logical_name, state,
         primary_provider, primary_bucket, object_key, content_type,
         retention_class, size_bytes, sha256, retain_until, legal_hold)
    VALUES (%s, %s, %s, 'checkpoint', %s, %s, 'b2', 'bkt', %s,
            'application/octet-stream', 'standard', %s, %s, %s, false)
"""


@pytest.fixture
def job():
    """One job's worth of artifacts, owned by a tenant this test invents."""
    tag = uuid.uuid4().hex[:10]
    tenant = f"tenant-{tag}"
    job_id = f"job-{tag}"
    made = []
    with _pool.connection() as conn:
        # `storage.artifacts.job_id` is a foreign key to `jobs`, so the job has
        # to exist before its artifacts do. Only the NOT NULL columns without a
        # default are supplied; everything else is left to the schema.
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload, owner_id) "
            "VALUES (%s, 'completed', 0, %s, '{}'::jsonb, %s)",
            (job_id, time.time(), tenant),
        )
        for name, size, sha in [
            ("model.safetensors", 4096, "a" * 64),
            ("optimizer.pt", 2048, "b" * 64),
        ]:
            aid = str(uuid.uuid4())
            conn.execute(
                _INSERT_ARTIFACT,
                (aid, tenant, job_id, name, "available", f"k/{aid}", size, sha,
                 datetime.now(timezone.utc) + timedelta(days=1)),
            )
            made.append(aid)
        conn.commit()
    yield {"tenant": tenant, "job_id": job_id, "artifact_ids": made, "tag": tag}
    with _pool.connection() as conn:
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM volume_promotions WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        conn.commit()


def _resolve(job_id: str, tenant: str) -> dict:
    from artifacts import get_artifact_manager

    return get_artifact_manager().resolve_promotion_manifest(job_id, tenant_id=tenant)


def test_the_manifest_lists_the_jobs_artifacts(job):
    m = _resolve(job["job_id"], job["tenant"])
    assert m["ok"] is True
    assert m["file_count"] == 2
    assert m["total_bytes"] == 4096 + 2048
    assert {f["logical_name"] for f in m["files"]} == {"model.safetensors", "optimizer.pt"}


def test_another_tenant_sees_nothing(job):
    """Scoping is in the WHERE clause, not a filter after the fact."""
    m = _resolve(job["job_id"], "some-other-tenant")
    assert m["file_count"] == 0, (
        "a job id belonging to another tenant resolved to files — a manifest is "
        "a list of names and sizes, so this is a disclosure and not merely an "
        "authorization slip"
    )
    assert m["files"] == []


def test_an_artifact_still_uploading_is_not_promotable(job):
    """Its sha256 and size are not yet trustworthy.

    Copying one would write a partial object and verify it against a digest of
    something that was still being written.
    """
    with _pool.connection() as conn:
        conn.execute(
            _INSERT_ARTIFACT,
            (str(uuid.uuid4()), job["tenant"], job["job_id"], "partial.pt",
             "uploading", f"k/up-{job['tag']}", 99, None, None),
        )
        conn.commit()
    m = _resolve(job["job_id"], job["tenant"])
    assert m["file_count"] == 2, "an artifact still uploading was included in the manifest"


def test_the_digest_is_stable_for_the_same_set(job):
    """Two calls, one promotion."""
    assert _resolve(job["job_id"], job["tenant"])["manifest_sha256"] == _resolve(
        job["job_id"], job["tenant"]
    )["manifest_sha256"]


def test_the_digest_changes_when_a_new_artifact_appears(job):
    """A later training run is a different promotion, not a replay of the first.

    If the digest ignored new files, a user who promotes, trains more, and
    promotes again would silently receive the first set twice.
    """
    before = _resolve(job["job_id"], job["tenant"])["manifest_sha256"]
    with _pool.connection() as conn:
        conn.execute(
            _INSERT_ARTIFACT,
            (str(uuid.uuid4()), job["tenant"], job["job_id"], "epoch2.pt",
             "available", f"k/e2-{job['tag']}", 512, "c" * 64, None),
        )
        conn.commit()
    assert _resolve(job["job_id"], job["tenant"])["manifest_sha256"] != before


def test_an_artifact_without_a_digest_is_reported_not_dropped(job):
    """Silently omitting it would make the manifest quietly incomplete.

    An unverifiable copy of someone's weights is worse than a refusal, because
    it looks complete — so the caller is told which files cannot be checked.
    """
    nodigest_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            _INSERT_ARTIFACT,
            (nodigest_id, job["tenant"], job["job_id"], "nodigest.pt",
             "available", f"k/nd-{job['tag']}", 10, None, None),
        )
        conn.commit()
    m = _resolve(job["job_id"], job["tenant"])
    assert nodigest_id in m["unverifiable"]
    assert m["file_count"] == 3, "the unverifiable artifact was dropped rather than flagged"


def test_the_earliest_retention_clock_is_the_one_reported(job):
    """A promotion races the *soonest* expiry in the set, not the average."""
    soon = datetime.now(timezone.utc) + timedelta(minutes=1)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE storage.artifacts SET retain_until = %s WHERE artifact_id = %s",
            (soon, job["artifact_ids"][1]),
        )
        conn.commit()
    m = _resolve(job["job_id"], job["tenant"])
    assert m["earliest_retain_until"] == soon.astimezone(timezone.utc).isoformat()


def test_a_job_with_nothing_promotable_is_an_empty_manifest_not_an_error():
    """"Nothing to promote" and "no such job" must look the same to a caller.

    Distinguishing them would let anyone probe which job ids exist.
    """
    m = _resolve(f"job-does-not-exist-{uuid.uuid4().hex[:8]}", "tenant-nobody")
    assert m["ok"] is True
    assert m["file_count"] == 0
