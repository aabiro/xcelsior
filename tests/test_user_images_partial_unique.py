"""P3/A2 — regression test: soft-deleted user_images must not block
re-creation of (owner, name, tag).

Requires a live Postgres pool (XCELSIOR_DATABASE_URL). Skipped otherwise.
"""

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")


def _img_id() -> str:
    return f"test-img-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def pool():
    try:
        from db import _get_pg_pool
    except Exception:
        pytest.skip("db module not importable")
    try:
        # _get_pg_pool() auto-bootstraps schema (incl. partial unique index)
        # via _ensure_pg_tables() on first connection.
        p = _get_pg_pool()
    except Exception as e:
        pytest.skip(f"no pg pool available: {e}")
    return p


@pytest.fixture
def cleanup_images():
    ids: list[str] = []
    yield ids
    if not ids:
        return
    try:
        from db import _get_pg_pool
        p = _get_pg_pool()
        with p.connection() as conn:
            for iid in ids:
                conn.execute("DELETE FROM user_images WHERE image_id=%s", (iid,))
            conn.commit()
    except Exception:
        pass


def _insert(pool, owner: str, name: str, tag: str, image_id: str):
    now = time.time()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO user_images (
                image_id, owner_id, name, tag, description,
                source_job_id, host_id, image_ref, size_bytes,
                status, created_at, deleted_at
            ) VALUES (%s,%s,%s,%s,'','','','ref',0,'ready',%s,0)
            """,
            (image_id, owner, name, tag, now),
        )
        conn.commit()


def _soft_delete(pool, image_id: str):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE user_images SET deleted_at=%s WHERE image_id=%s",
            (time.time(), image_id),
        )
        conn.commit()


class TestPartialUniqueIndex:
    def test_duplicate_live_rows_blocked(self, pool, cleanup_images):
        import psycopg

        owner = f"test-owner-{uuid.uuid4().hex[:8]}@test"
        name, tag = "a2-dup", "latest"
        i1, i2 = _img_id(), _img_id()
        cleanup_images.extend([i1, i2])

        _insert(pool, owner, name, tag, i1)
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert(pool, owner, name, tag, i2)

    def test_soft_delete_frees_the_slot(self, pool, cleanup_images):
        owner = f"test-owner-{uuid.uuid4().hex[:8]}@test"
        name, tag = "a2-recreate", "v1"
        i1, i2 = _img_id(), _img_id()
        cleanup_images.extend([i1, i2])

        _insert(pool, owner, name, tag, i1)
        _soft_delete(pool, i1)
        # Must succeed: soft-deleted row is excluded from the partial index.
        _insert(pool, owner, name, tag, i2)

        with pool.connection() as conn:
            cur = conn.execute(
                "SELECT image_id, deleted_at FROM user_images "
                "WHERE owner_id=%s AND name=%s AND tag=%s "
                "ORDER BY created_at",
                (owner, name, tag),
            )
            rows = cur.fetchall()
        assert len(rows) == 2
        assert rows[0][1] > 0  # first one soft-deleted
        assert rows[1][1] == 0  # second one live
