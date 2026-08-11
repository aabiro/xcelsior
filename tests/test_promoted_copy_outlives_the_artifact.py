"""Gate P3 clause 2, second half: an expired artifact does not take the promotion with it.

The clause is *"the retention clock is asserted: an artifact past `retain_until`
is gone, a promoted volume is not"*. The first half was covered. The second was
recorded as waiting on clause 3 because it "needs a mounted volume" — and that
reason was doing more work than it should. **Surviving is not the same as being
mountable.** Whether a promoted copy still exists after the artifact expires is
a property of the *deletion path*, and the deletion path is code that runs here.

What genuinely needs hardware is reading the bytes back through a mount, which
is clause 3's sentence, not this one. So this file proves the part that is code
and says plainly what it does not reach.

## The two ways this could break, and why one test is not enough

**Behaviourally** — the reaper deletes the artifact and cascades into the
promotion records, so the volume's copy is no longer tracked and the user's
weights become an untracked blob. Asserted below against real rows.

**Structurally** — someone later "tidies up" by making the deletion path
release the promotion, and the behavioural test above still passes because it
only checks rows it created. So the deletion path's own SQL is walked and
asserted to name no volume table at all. That check is derived from the
function's source rather than from a list of table names kept beside it.

Neither is redundant: the first catches a cascade in the schema, the second
catches a deliberate edit to the worker.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = all(
            _c.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0] is not None
            for t in ("volume_promotions", "volume_promotion_files", "storage.artifacts")
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("promotion or artifact tables are missing")


@pytest.fixture
def promoted():
    """An expired artifact, and a completed promotion of it onto a volume.

    `retain_until` is in the past and `legal_hold` is false — the artifact is
    unambiguously due for deletion, so nothing below passes because the reaper
    declined to act.
    """
    tenant = str(uuid.uuid4())
    job = f"job-{uuid.uuid4().hex[:12]}"
    artifact = str(uuid.uuid4())
    promotion = str(uuid.uuid4())
    volume = f"vol-{uuid.uuid4().hex[:8]}"

    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO storage.artifacts
                   (artifact_id, tenant_id, artifact_type, logical_name,
                    state, primary_provider, primary_bucket, object_key,
                    content_type, retention_class,
                    size_bytes, legal_hold, retain_until)
            VALUES (%s, %s, 'output', 'weights.bin', 'available',
                    'local', 'test', %s,
                    'application/octet-stream', 'standard',
                    1024, false, clock_timestamp() - interval '30 days')
            """,
            # `job_id` is left null on purpose: it carries a foreign key to
            # `jobs`, and inventing a job row would add a second fixture whose
            # only purpose is to satisfy a constraint this clause is not about.
            # The artifact-to-promotion link that matters here is
            # `volume_promotion_files.artifact_id`.
            (artifact, tenant, f"artifacts/{artifact}/weights.bin"),
        )
        conn.execute(
            """
            INSERT INTO volume_promotions
                   (promotion_id, tenant_id, job_id, volume_id, idempotency_key,
                    manifest_sha256, file_count, total_bytes, state, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, 1, 1024, 'succeeded',
                    clock_timestamp())
            """,
            (promotion, tenant, job, volume, f"promo-{promotion}", "0" * 64),
        )
        conn.execute(
            """
            INSERT INTO volume_promotion_files
                   (promotion_id, tenant_id, artifact_id, logical_name,
                    size_bytes, bytes_written, sha256_verified, state)
            VALUES (%s, %s, %s, 'weights.bin', 1024, 1024, true, 'done')
            """,
            (promotion, tenant, artifact),
        )

    yield {
        "tenant": tenant,
        "job": job,
        "artifact": artifact,
        "promotion": promotion,
        "volume": volume,
    }

    with pg_transaction() as conn:
        conn.execute("DELETE FROM volume_promotion_files WHERE promotion_id = %s", (promotion,))
        conn.execute("DELETE FROM volume_promotions WHERE promotion_id = %s", (promotion,))
        conn.execute("DELETE FROM storage.artifacts WHERE artifact_id = %s", (artifact,))


def _counts(row: dict) -> tuple[int, int, int]:
    with pg_transaction() as conn:
        artifacts = conn.execute(
            "SELECT count(*) FROM storage.artifacts WHERE artifact_id = %s "
            " AND state = 'available'",
            (row["artifact"],),
        ).fetchone()[0]
        promotions = conn.execute(
            "SELECT count(*) FROM volume_promotions WHERE promotion_id = %s",
            (row["promotion"],),
        ).fetchone()[0]
        files = conn.execute(
            "SELECT count(*) FROM volume_promotion_files WHERE promotion_id = %s",
            (row["promotion"],),
        ).fetchone()[0]
    return artifacts, promotions, files


# ── The behaviour ─────────────────────────────────────────────────────


def test_the_fixture_starts_from_the_state_the_clause_describes(promoted):
    """A promoted, expired artifact. Without this the rest asserts nothing."""
    artifacts, promotions, files = _counts(promoted)
    assert (artifacts, promotions, files) == (1, 1, 1)

    with pg_transaction() as conn:
        due = conn.execute(
            "SELECT retain_until < clock_timestamp(), legal_hold "
            "  FROM storage.artifacts WHERE artifact_id = %s",
            (promoted["artifact"],),
        ).fetchone()
    assert due[0] is True, "the artifact is not actually past retain_until"
    assert due[1] is False, "the artifact is under hold, so deletion would decline"


def test_deleting_the_expired_artifact_leaves_the_promotion_records_intact(promoted):
    """The clause: the artifact goes, the promoted copy does not.

    Deletion is driven through the real path — a deletion job the reaper
    claims — rather than by issuing the DELETE this test wants to see, which
    would prove only that the test can write SQL.
    """
    from artifacts import get_artifact_manager

    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO storage.artifact_deletion_jobs
                   (deletion_id, artifact_id, reason, requested_by,
                    state, next_attempt_at)
            VALUES (%s, %s, 'retention_expired', 'test', 'requested',
                    clock_timestamp())
            """,
            (str(uuid.uuid4()), promoted["artifact"]),
        )

    get_artifact_manager().cleanup_expired()

    # **The reaper must actually have run.** Without this, "the promotion
    # survived" is true of a reaper that declined, errored, or found nothing —
    # and the clause would read as proven by a test that watched nothing
    # happen. This is the half that makes the assertion below mean something.
    with pg_transaction() as conn:
        deletion = conn.execute(
            "SELECT state, last_error FROM storage.artifact_deletion_jobs  WHERE artifact_id = %s",
            (promoted["artifact"],),
        ).fetchone()
    assert deletion is not None, "the deletion job vanished"
    assert deletion[0] != "requested", (
        f"the reaper never claimed the deletion job (state={deletion[0]!r}, "
        f"error={deletion[1]!r}); nothing was expired, so the survival of the "
        "promotion below proves nothing"
    )

    artifacts, promotions, files = _counts(promoted)
    assert artifacts == 0, (
        "the artifact is still 'available' after the reaper ran; the first half "
        "of the clause — an artifact past retain_until is gone — did not happen"
    )
    assert promotions == 1, (
        "the promotion record was removed with the artifact; the user's copy on "
        "the volume is now untracked"
    )
    assert files == 1, (
        "the promoted file record was removed with the artifact; nothing now "
        "records that those bytes are on the volume"
    )


# ── The structure ─────────────────────────────────────────────────────


def test_the_deletion_path_names_no_volume_table():
    """Derived from the reaper's own source, not from a list kept beside it.

    The behavioural test above only inspects rows it created, so a future edit
    that made deletion release the promotion could still pass it. This asserts
    the deletion path has no business with volumes at all — which is the real
    reason a promoted copy survives, rather than an accident of what the
    current SQL happens to touch.
    """
    import inspect
    import re

    from artifacts import ArtifactManager

    source = inspect.getsource(ArtifactManager.cleanup_expired)
    tables = set(re.findall(r"(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][\w.]*)", source))
    assert tables, "no SQL tables found in cleanup_expired; the derivation is broken"

    volume_tables = sorted(t for t in tables if "volume" in t.lower())
    assert not volume_tables, (
        f"the artifact deletion path now touches {volume_tables}. Expiring an "
        "artifact must not reach a promoted copy — that copy is the user's, on "
        "their volume, and its retention is not the artifact's clock."
    )


def test_the_reaper_still_touches_the_artifact_tables_it_should():
    """Positive control for the check above.

    A `cleanup_expired` that touched *no* tables would satisfy "names no volume
    table" perfectly. This is what stops the structural assertion passing
    because the derivation stopped finding anything.
    """
    import inspect
    import re

    from artifacts import ArtifactManager

    source = inspect.getsource(ArtifactManager.cleanup_expired)
    tables = {
        t.lower() for t in re.findall(r"(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][\w.]*)", source)
    }
    assert "storage.artifacts" in tables, tables
    assert "storage.artifact_deletion_jobs" in tables, tables
