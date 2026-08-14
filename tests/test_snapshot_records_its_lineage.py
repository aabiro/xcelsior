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


def test_the_sweep_exists_and_pins_a_digest():
    """The ratchet that stood here has fired, and this is what replaces it.

    It asserted the sweep was *still absent* so that whoever shipped one was
    told to update the gate rather than leaving a half-met clause reading as
    whole. It went red the moment the route existed — that is the ratchet
    working — and deleting it would be the failure mode the plan names. So it
    is replaced by the assertion it was holding a place for.

    What is asserted is the property the clause turns on: a sweep launches its
    members against a **digest**, never the mutable tag. Everything else about
    the sweep is covered by `tests/test_a_sweep_is_a_record.py`; this is the
    one fact that connects the sweep back to the snapshot's lineage.
    """
    import pathlib

    routes = pathlib.Path("routes/instances.py").read_text(encoding="utf-8")
    assert '@router.post("/api/v1/image-sweeps"' in routes, (
        "the sweep creation route is gone; if it moved, point this at it"
    )

    start = routes.index("def api_create_image_sweep(")
    body = routes[start : routes.index("\n@router.", start + 1)]
    assert "image=image_digest" in body, (
        "the sweep no longer launches members against the pinned digest. A tag "
        "can be re-pushed between the first member and the last, so 'N nodes "
        "from one snapshot' would be unprovable — which is the whole reason "
        "migration 112 records a digest at all."
    )
    # `image=image_ref`, not the bare word. The route's own docstring explains
    # why the tag is unusable, so a substring check matches the explanation
    # rather than a use — the third time that shape has caught me today, and
    # the reason the assertion names the assignment.
    assert "image=image_ref" not in body, (
        "the sweep launches members from the mutable tag; a tag can be "
        "re-pushed between the first member and the last"
    )


def test_the_sweep_funds_every_member_before_submitting_it():
    """A sweep is N times the spend. The wallet check is not a bulk-path exemption.

    Skipping it would let one call launch 64 instances on a wallet that could
    fund one — the single-instance path's fund gate exists for a reason and a
    bulk path does not get to be the exception.
    """
    import pathlib

    routes = pathlib.Path("routes/instances.py").read_text(encoding="utf-8")
    start = routes.index("def api_create_image_sweep(")
    body = routes[start : routes.index("\n@router.", start + 1)]
    assert "_wallet_preflight(" in body, "the sweep no longer funds members before submitting them"
    assert "link_wallet_hold_to_job" in body, (
        "the sweep no longer links each member's hold to its job, so the money "
        "is held and never attributed"
    )
    assert "release_wallet_hold" in body, (
        "a member whose submit fails no longer releases its hold — the wallet "
        "would keep money reserved against a job that does not exist"
    )


def test_the_listing_mapping_covers_every_column_it_selects():
    """The user-images listing maps rows **positionally**, so a shift is silent.

    Adding `base_image_ref` to the SELECT moved every index after it: the
    mapping kept reading `r[8]` for `size_bytes` and got the image string, and
    `int()` raised. Two unrelated test files caught it, which is luck rather
    than design — nothing tied the column list to the mapping.

    This ties them. It does not force a rewrite to `dict_row`; it just refuses
    to let the two drift, which is the whole failure mode.
    """
    import pathlib

    source = pathlib.Path(ROUTES).read_text(encoding="utf-8")
    select = re.search(
        r'"SELECT (image_id, owner_id, name, tag.*?)"\s*\n?\s*f?"FROM user_images', source, re.S
    )
    assert select, "the user_images listing query is no longer findable"

    columns = [c.strip() for c in re.sub(r'"\s*\n\s*"', "", select.group(1)).split(",")]
    columns = [c for c in columns if c]
    assert len(columns) > 5, f"only parsed {columns} from the SELECT"

    # The mapping immediately after it, by highest positional index used.
    mapping = source[select.end() : select.end() + 2000]
    indices = {int(m) for m in re.findall(r"\br\[(\d+)\]", mapping)}
    assert indices, "no positional row access found after the query"

    assert max(indices) == len(columns) - 1, (
        f"the listing selects {len(columns)} columns but its mapping reads up to "
        f"r[{max(indices)}]. A column was added or removed on one side only, and "
        "every field after it is now reading its neighbour."
    )


# ── The digest: Gate P7's sweep is unprovable without it ──────────────


def test_the_digest_column_admits_unknown():
    """A snapshot whose push succeeded but whose inspect failed knows nothing.

    `NULL` must stay available for that, and for every row predating this. A
    default would hand a sweep a digest to compare that was never observed.
    """
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            " WHERE table_name = 'user_images' AND column_name = 'image_digest'"
        ).fetchone()
    assert row is not None, "image_digest is missing"
    assert row[0] == "YES", "image_digest is NOT NULL; unknown must be expressible"
    assert row[1] is None, f"image_digest has a default ({row[1]!r})"


def test_the_completion_callback_cannot_erase_a_digest_it_does_not_know():
    """An older agent sends no digest. Silence means "I don't know", not "none".

    Overwriting with `''` would delete evidence recorded by a newer agent on a
    retry — and retries are the normal case for this callback.
    """
    import pathlib

    # **Code only.** The first version searched the whole file and matched
    # the *comment* directly above the SQL, so removing the COALESCE left it
    # green. Same match-a-mention defect this suite has caught repeatedly —
    # found by checking the injection landed rather than trusting a pass.
    code = "\n".join(
        line
        for line in pathlib.Path(ROUTES).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "COALESCE(NULLIF(%s, ''), image_digest)" in code, (
        "the completion callback no longer preserves an existing digest when "
        "the agent sends none; a retry from an older agent would erase it"
    )


def test_the_worker_reads_the_digest_at_push_time():
    """From the push that produced it, not from a later registry lookup.

    A later lookup answers "what does this tag point at now", which is a
    different question — and the mutability of the tag is the whole reason the
    digest is needed.
    """
    import pathlib

    agent = pathlib.Path("worker_agent.py").read_text(encoding="utf-8")
    assert "def _read_repo_digest(" in agent, "the worker no longer reads a digest"
    assert '"{{index .RepoDigests 0}}"' in agent, (
        "the digest is no longer read from docker's own record of the push"
    )
    assert '"image_digest": image_digest' in agent, (
        "the worker no longer reports the digest to /user-images/{id}/complete"
    )


def test_a_digest_that_is_not_one_is_not_recorded():
    """`docker inspect` can return `<no value>` when there are no RepoDigests.

    Recording that would give a sweep a string to compare that means nothing,
    and two members agreeing on `<no value>` would read as byte-identical.
    """
    import pathlib

    agent = pathlib.Path("worker_agent.py").read_text(encoding="utf-8")
    assert 'return digest if "@sha256:" in digest else ""' in agent, (
        "the worker no longer checks the digest is shaped like one before reporting it"
    )
