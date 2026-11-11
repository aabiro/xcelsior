"""T13b against the real schema: tenant B cannot launch tenant A's snapshot image.

The unit tests inject the lookup. These use the `user_images` table itself, the
real `submit_job`, and the real `/instance` route, so the SQL, the escaping and
the place the check sits are all exercised.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient

import scheduler
import snapshot_image_access as access
from api import app
from db import pg_transaction

REG = "registry.t13b.example.test/xcelsior"
client = TestClient(app)


@pytest.fixture
def snapshots(monkeypatch):
    monkeypatch.setenv("XCELSIOR_REGISTRY_URL", REG)
    tag = uuid.uuid4().hex[:12]
    owner = f"alice-{tag}@example.test"
    ns = f"{REG}/alice-{tag}"
    private, public = f"{ns}/model:latest", f"{ns}/shared:v1"
    ids = [f"img-{tag}-a", f"img-{tag}-b"]
    with pg_transaction() as conn:
        for image_id, name, ref, is_public in (
            (ids[0], "model", private, False),
            (ids[1], "shared", public, True),
        ):
            conn.execute(
                """INSERT INTO user_images
                     (image_id, owner_id, name, tag, image_ref, status, created_at,
                      deleted_at, is_public)
                   VALUES (%s, %s, %s, 'latest', %s, 'ready', %s, 0, %s)""",
                (image_id, owner, name, ref, time.time(), is_public),
            )
    try:
        yield {"owner": owner, "private": private, "public": public, "ns": ns, "tag": tag}
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM user_images WHERE image_id = ANY(%s)", (ids,))


def test_the_real_lookup_separates_owner_stranger_and_public(snapshots):
    assert (
        access.foreign_snapshot_problem(snapshots["private"], ["bob@example.test"])
        == access.REFUSAL
    )
    assert access.foreign_snapshot_problem(snapshots["private"], [snapshots["owner"]]) is None
    assert access.foreign_snapshot_problem(snapshots["public"], ["bob@example.test"]) is None
    other_tag = f"{snapshots['ns']}/model:v2"
    assert access.foreign_snapshot_problem(other_tag, ["bob@example.test"]) == access.REFUSAL


def test_a_wildcard_namespace_does_not_match_someone_elses_rows(snapshots):
    """`alice-%` typed by a stranger must not be read as 'a namespace with no snapshots'
    and must not be matched against Alice's rows as a pattern either."""
    probe = f"{REG}/alice-{snapshots['tag'][:-1]}_/model:latest"
    assert access.foreign_snapshot_problem(probe, ["bob@example.test"]) is None  # no such namespace
    assert (
        access.foreign_snapshot_problem(snapshots["private"], ["bob@example.test"])
        == access.REFUSAL
    )


def test_submit_job_refuses_and_creates_nothing(snapshots):
    name = f"t13b-{snapshots['tag']}"
    with pytest.raises(PermissionError):
        scheduler.submit_job(name, 0, image=snapshots["private"], owner="cus_bob")
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT count(*) FROM jobs WHERE payload::text LIKE %s", (f"%{name}%",)
        ).fetchone()
    assert (row[0] if not isinstance(row, dict) else list(row.values())[0]) == 0


def test_the_launch_route_refuses_a_strangers_reference(snapshots):
    """Refused before any job exists. The route's own image validator answers
    first today (the platform registry is not on the public allow-list); the
    ownership check is what holds if that list ever changes."""
    resp = client.post(
        "/instance",
        json={
            "name": f"t13b-{snapshots['tag']}",
            "vram_needed_gb": 0,
            "tier": "standard",
            "gpu_model": "RTX 2060",
            "image": snapshots["private"],
        },
    )
    assert resp.status_code in (404, 422), resp.text
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT count(*) FROM jobs WHERE payload::text LIKE %s",
            (f"%t13b-{snapshots['tag']}%",),
        ).fetchone()
    assert (row[0] if not isinstance(row, dict) else list(row.values())[0]) == 0


def test_the_owner_can_launch_their_own_snapshot(snapshots):
    """The other half: before this, every launch from a saved template died in
    submit_job as "not from an approved registry", because the platform's own
    registry is not on the public allow-list."""
    job = None
    try:
        job = scheduler.submit_job(
            f"t13b-own-{snapshots['tag']}", 0, image=snapshots["private"], owner=snapshots["owner"]
        )
    except PermissionError:
        pytest.fail("the owner was refused their own snapshot")
    except ValueError as exc:
        # Anything after the image check (pricing, tier) is not this test's subject.
        assert "approved registry" not in str(exc), exc
    finally:
        if job:
            with pg_transaction() as conn:
                conn.execute("DELETE FROM jobs WHERE job_id = %s", (job["job_id"],))


def test_the_allow_list_still_applies_to_everything_else_in_the_registry(snapshots):
    """Only a namespace holding the caller's snapshots is trusted, not the registry."""
    with pytest.raises(ValueError, match="approved registry"):
        scheduler.submit_job(
            "t13b-other", 0, image=f"{REG}/someone-else/tool:v1", owner=snapshots["owner"]
        )
