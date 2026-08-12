"""Gate P7 piece 2: a sweep is a record, not a loop.

A route that calls the launch path N times and returns N job ids is a client
convenience. It leaves "these N came from one snapshot" as an intention held by
whoever made the request — unverifiable afterwards, with partial failure
invisible and nothing for piece 3's fingerprints to be compared across.

What is asserted here is that the record exists and is honest: the digest is
pinned once, a launch failure is *recorded* rather than raised, and an image
that cannot support the claim is refused rather than swept from its mutable tag.

## The refusals are the substance

`image_digest_unknown` is the one that matters. It would be easy to fall back to
`image_ref` when no digest is recorded — the sweep would run, N nodes would
start, and the clause would read as met. It would also be unprovable: N
containers launched from a tag were asked for the same *name*, and a tag can be
re-pushed between the first launch and the last. Refusing keeps "unproven" and
"proven" distinguishable, which is the whole reason migration 112 exists.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as _c:
        _has = _c.execute("SELECT to_regclass('image_sweeps')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no database: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 113")

from control_plane.image_sweeps import (  # noqa: E402
    MAX_SWEEP_SIZE,
    SweepRefused,
    create_sweep,
    read_sweep,
)

DIGEST = "reg.example/o/img@sha256:" + "a" * 64


@pytest.fixture
def image():
    """A ready snapshot with a recorded digest, plus its owner and tenant."""
    image_id = f"img-{uuid.uuid4().hex[:12]}"
    owner = f"owner-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO user_images
                   (image_id, owner_id, name, tag, image_ref, image_digest,
                    status, created_at, deleted_at)
            VALUES (%s, %s, 'sweep-test', 'v1', %s, %s, 'ready', 0, 0)
            """,
            (image_id, owner, "reg.example/o/img:v1", DIGEST),
        )
        conn.commit()
    yield image_id, owner
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "DELETE FROM image_sweep_members WHERE sweep_id IN "
            "(SELECT sweep_id FROM image_sweeps WHERE image_id = %s)",
            (image_id,),
        )
        conn.execute("DELETE FROM image_sweeps WHERE image_id = %s", (image_id,))
        conn.execute("DELETE FROM user_images WHERE image_id = %s", (image_id,))
        conn.commit()


def _launcher(hosts, fail_at=()):
    """A launch callable that reports the host it landed on."""

    def launch(digest: str, index: int) -> dict:
        assert digest == DIGEST, "a member launched against something else"
        if index in fail_at:
            raise RuntimeError(f"no capacity for member {index}")
        return {"job_id": f"job-{uuid.uuid4().hex[:8]}", "host_id": hosts[index % len(hosts)]}

    return launch


# ── The record ────────────────────────────────────────────────────────


def test_the_members_belong_to_one_sweep_with_one_digest(image):
    image_id, owner = image
    with _get_pg_pool().connection() as conn:
        sweep = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=3,
            launch=_launcher(["h-1", "h-2"]),
        )
        conn.commit()

    assert sweep.requested_count == 3
    assert sweep.launched == 3
    assert sweep.image_digest == DIGEST
    assert sweep.state == "running"
    assert {m.index for m in sweep.members} == {0, 1, 2}

    with _get_pg_pool().connection() as conn:
        stored = read_sweep(conn, sweep.sweep_id, tenant_id=owner)
    assert stored is not None, "the sweep is not readable back"
    assert stored.image_digest == DIGEST
    assert stored.launched == 3, "the members were not persisted"


def test_every_member_launches_against_the_same_pinned_digest(image):
    """Pinned once. Re-resolving per member reopens the race it closes."""
    image_id, owner = image
    seen: list[str] = []

    def launch(digest: str, index: int) -> dict:
        seen.append(digest)
        return {"job_id": f"job-{index}", "host_id": "h-1"}

    with _get_pg_pool().connection() as conn:
        create_sweep(
            conn, tenant_id=owner, owner_id=owner, image_id=image_id, count=4, launch=launch
        )
        conn.commit()

    assert len(set(seen)) == 1, f"members launched against different refs: {set(seen)}"
    assert "@sha256:" in seen[0], "a member launched against a mutable tag"


def test_a_partial_failure_is_recorded_rather_than_raised(image):
    """ "3 of 5, and here are the two that did not" is the answer.

    Aborting would discard that *and* strand the instances that did start.
    """
    image_id, owner = image
    with _get_pg_pool().connection() as conn:
        sweep = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=5,
            launch=_launcher(["h-1"], fail_at=(1, 3)),
        )
        conn.commit()

    assert sweep.launched == 3
    assert sweep.state == "running", "a partly-launched sweep is not a failed one"
    failed = {m.index for m in sweep.members if m.state == "failed"}
    assert failed == {1, 3}, f"the failed members are not identifiable: {failed}"
    assert all(m.failure_code for m in sweep.members if m.state == "failed"), (
        "a failed member recorded no reason"
    )


def test_a_sweep_where_nothing_launched_is_failed(image):
    """The distinction the state exists to draw."""
    image_id, owner = image
    with _get_pg_pool().connection() as conn:
        sweep = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=2,
            launch=_launcher(["h-1"], fail_at=(0, 1)),
        )
        conn.commit()
    assert sweep.launched == 0
    assert sweep.state == "failed"


def test_the_distinct_host_count_is_reported(image):
    """A single-host sweep is not refused, but it must not read as a full pass.

    P5.1's skip is the precedent: a same-host migration returns `ok` while
    proving nothing. The count is surfaced so a caller can say what a result
    does and does not establish.
    """
    image_id, owner = image
    with _get_pg_pool().connection() as conn:
        one_host = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=3,
            launch=_launcher(["h-only"]),
        )
        spread = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=3,
            launch=_launcher(["h-1", "h-2", "h-3"]),
        )
        conn.commit()
    assert one_host.distinct_hosts == 1
    assert spread.distinct_hosts == 3
    assert one_host.as_dict()["distinct_hosts"] == 1


# ── The refusals ──────────────────────────────────────────────────────


def test_an_image_with_no_digest_cannot_be_swept():
    """The refusal that keeps "unproven" and "proven" distinguishable."""
    image_id = f"img-{uuid.uuid4().hex[:12]}"
    owner = f"owner-{uuid.uuid4().hex[:8]}"
    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO user_images
                       (image_id, owner_id, name, tag, image_ref, status,
                        created_at, deleted_at)
                VALUES (%s, %s, 'no-digest', 'v1', 'reg/x:v1', 'ready', 0, 0)
                """,
                (image_id, owner),
            )
            conn.commit()
            with pytest.raises(SweepRefused) as refused:
                create_sweep(
                    conn,
                    tenant_id=owner,
                    owner_id=owner,
                    image_id=image_id,
                    count=2,
                    launch=_launcher(["h-1"]),
                )
        assert refused.value.code == "image_digest_unknown"
        assert "mutable tag" in refused.value.detail
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM user_images WHERE image_id = %s", (image_id,))
            conn.commit()


def test_an_unready_image_cannot_be_swept():
    image_id = f"img-{uuid.uuid4().hex[:12]}"
    owner = f"owner-{uuid.uuid4().hex[:8]}"
    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO user_images
                       (image_id, owner_id, name, tag, image_ref, image_digest,
                        status, created_at, deleted_at)
                VALUES (%s, %s, 'pending-img', 'v1', 'reg/x:v1', %s, 'pending', 0, 0)
                """,
                (image_id, owner, DIGEST),
            )
            conn.commit()
            with pytest.raises(SweepRefused) as refused:
                create_sweep(
                    conn,
                    tenant_id=owner,
                    owner_id=owner,
                    image_id=image_id,
                    count=2,
                    launch=_launcher(["h-1"]),
                )
        assert refused.value.code == "image_not_ready"
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM user_images WHERE image_id = %s", (image_id,))
            conn.commit()


def test_another_owners_image_is_not_found_rather_than_forbidden(image):
    """No existence oracle: `forbidden` would confirm the image exists."""
    image_id, _owner = image
    stranger = f"owner-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        with pytest.raises(SweepRefused) as refused:
            create_sweep(
                conn,
                tenant_id=stranger,
                owner_id=stranger,
                image_id=image_id,
                count=1,
                launch=_launcher(["h-1"]),
            )
    assert refused.value.code == "image_not_found"


@pytest.mark.parametrize("count", [0, -1, MAX_SWEEP_SIZE + 1])
def test_an_out_of_range_count_is_refused_before_anything_is_written(image, count):
    """A typed refusal, not a constraint violation surfacing as a 500."""
    image_id, owner = image
    with _get_pg_pool().connection() as conn:
        with pytest.raises(SweepRefused) as refused:
            create_sweep(
                conn,
                tenant_id=owner,
                owner_id=owner,
                image_id=image_id,
                count=count,
                launch=_launcher(["h-1"]),
            )
        assert refused.value.code == "invalid_count"
        remaining = conn.execute(
            "SELECT count(*) FROM image_sweeps WHERE image_id = %s", (image_id,)
        ).fetchone()[0]
    assert remaining == 0, "a refused sweep still wrote a row"


def test_a_sweep_is_not_readable_by_another_tenant(image):
    image_id, owner = image
    with _get_pg_pool().connection() as conn:
        sweep = create_sweep(
            conn,
            tenant_id=owner,
            owner_id=owner,
            image_id=image_id,
            count=1,
            launch=_launcher(["h-1"]),
        )
        conn.commit()
        assert read_sweep(conn, sweep.sweep_id, tenant_id="someone-else") is None
