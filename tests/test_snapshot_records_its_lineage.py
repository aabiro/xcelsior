"""Gate P7, the lineage half: a snapshot records what it was built from.

The clause asks for *"what it was built from, when, by which run"*. Three were
already on `user_images` — `created_at`, `source_job_id`, `host_id`. The base
image was not, and it is the one that answers a question about **contents**
rather than history.

A snapshot is `docker commit` over a running container, so the resulting image
is a diff on top of whatever the job launched with. Knowing which run produced
it tells you nothing about what is underneath the commit. When a CVE lands in a
widely-used base image, "which snapshots contain it" is the question, and
`source_job_id` cannot answer it — the job may since have been requeued onto a
different image, or deleted.

## What is asserted, and what deliberately is not

This file covers lineage only. The other half of Gate P7 — *"a sweep of N nodes
from one snapshot is byte-identical in environment"* — has no implementation at
all: nothing launches N nodes from one image, so there is nothing to compare and
nothing here pretends otherwise. P7 stays FAIL on that basis, and this closes
the half that is real rather than letting a partial answer read as a whole one.
"""

from __future__ import annotations

import os
import re
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as _c:
        _has = (
            _c.execute(
                "SELECT 1 FROM information_schema.columns "
                " WHERE table_name = 'user_images' AND column_name = 'base_image_ref'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no database: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 111")

ROUTES = "routes/instances.py"


# ── The column, and that it is genuinely nullable ─────────────────────


def test_the_column_exists_and_admits_unknown():
    """Existing rows do not know their base, and must be allowed to say so.

    A `NOT NULL` with a default would manufacture an answer for every snapshot
    taken before this shipped — which is the inference the column exists to
    stop anyone making.
    """
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            " WHERE table_name = 'user_images' AND column_name = 'base_image_ref'"
        ).fetchone()
    assert row is not None, "base_image_ref is missing"
    assert row[0] == "YES", "base_image_ref is NOT NULL; old rows cannot say 'unknown'"
    assert row[1] is None, (
        f"base_image_ref has a default ({row[1]!r}); every pre-existing snapshot "
        "would claim a base it never had"
    )


def test_a_row_round_trips_the_base_it_was_built_from():
    """The storage half, against the real table."""
    image_id = f"img-{uuid.uuid4().hex[:12]}"
    owner = f"owner-{uuid.uuid4().hex[:8]}"
    base = "nvidia/cuda:12.4.1-base-ubuntu22.04"
    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO user_images
                       (image_id, owner_id, name, tag, image_ref, base_image_ref,
                        status, created_at, deleted_at)
                VALUES (%s, %s, 'lineage-test', 'v1', %s, %s, 'ready', 0, 0)
                """,
                (image_id, owner, f"reg/{image_id}", base),
            )
            conn.commit()
            stored = conn.execute(
                "SELECT base_image_ref FROM user_images WHERE image_id = %s",
                (image_id,),
            ).fetchone()[0]
        assert stored == base
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM user_images WHERE image_id = %s", (image_id,))
            conn.commit()


# ── The write path, and the read path ─────────────────────────────────


def test_the_snapshot_route_records_the_base_from_the_job():
    """Recorded at snapshot time, from the job, not inferred later.

    The job is the only thing that knows what the container launched with, and
    it can be requeued onto a different image afterwards — so a later lookup
    would answer a different question than the one asked.
    """
    source = __import__("pathlib").Path(ROUTES).read_text(encoding="utf-8")
    insert = re.search(r"INSERT INTO user_images \((.*?)\) VALUES", source, re.S)
    assert insert, "the user_images insert is no longer findable"
    assert "base_image_ref" in insert.group(1), (
        "the snapshot route no longer records base_image_ref; a new snapshot "
        "would have no record of what it was built from"
    )
    assert 'job.get("image")' in source, (
        "the base is no longer read from the job; if it is being inferred from "
        "somewhere else, that source can disagree with what actually ran"
    )


def test_the_listing_returns_the_lineage_it_records():
    """Recorded but unreadable is not recorded.

    Lineage exists to be audited. A column no surface returns answers no
    question — the row would be true and useless.
    """
    source = __import__("pathlib").Path(ROUTES).read_text(encoding="utf-8")
    select = re.search(r'"SELECT image_id, owner_id, name, tag[^;]*?FROM user_images', source, re.S)
    assert select, "the user_images listing query is no longer findable"
    assert "base_image_ref" in select.group(0), (
        "the image listing no longer returns base_image_ref, so the lineage is "
        "stored where nobody can read it"
    )


# ── What this does not claim ──────────────────────────────────────────


def test_the_sweep_half_of_the_clause_is_still_absent():
    """Asserted so the half-met clause cannot quietly read as met.

    Gate P7 has two halves. This file closes lineage. If a sweep ships, this
    test fails and whoever ships it updates the gate — which is the point: the
    reminder lives next to the work rather than in someone's memory.
    """
    import pathlib

    routes = pathlib.Path("routes")
    sweep_routes = [
        f"{path.name}:{num}"
        for path in routes.glob("*.py")
        for num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'@router\.\w+\("[^"]*sweep', line)
    ]
    assert not sweep_routes, (
        f"a sweep endpoint now exists ({sweep_routes}) — Gate P7's other half "
        "may now be assertable. Update the gate rather than deleting this test."
    )
