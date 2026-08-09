"""The two constraints §0 of the promotion plan says must be tested when built.

Quoting it: *"Where a constraint is enforceable in code it gets a test when it is
built — following this repository's rule that a comment asking for something is
not a mechanism. The two that are straightforwardly testable: no presigned URL
appears in a command row (§3.1), and no promotion begins without a hold on every
artifact in its manifest (§3.3)."*

Both are here.

## Why the hold is not optional

§0 lists *"Skip the `legal_hold` and 'handle the delete case if it happens'"*
among the shortcuts that are prohibited because each one looks reasonable at 2am.
The reason it gives: the delete case *"happens precisely on artifacts near
expiry, which are the ones worth promoting, and the symptom is a partial volume
the user believes is complete."*

## Why the URL is not in the command row

§3.1: presigned URLs are read grants on a tenant's weights, and a command row is
queued, logged, and retained. The agent receives `{promotion_id}` and fetches
the manifest over its own authenticated channel.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = (
            _c.execute("SELECT to_regclass('storage.artifacts')").fetchone()[0] is not None
            and _c.execute("SELECT to_regclass('volume_promotions')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 102")


@pytest.fixture
def job():
    """A job with three available artifacts and one still uploading."""
    tag = uuid.uuid4().hex[:10]
    tenant = f"tenant-{tag}"
    job_id = f"job-{tag}"
    available, uploading = [], []

    with pg_transaction() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload, owner_id) "
            "VALUES (%s, 'completed', 0, extract(epoch from now()), '{}'::jsonb, %s)",
            (job_id, tenant),
        )
        for i in range(3):
            aid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO storage.artifacts
                     (artifact_id, tenant_id, job_id, artifact_type, logical_name, state,
                      primary_provider, primary_bucket, object_key, content_type,
                      retention_class, size_bytes, sha256, legal_hold)
                   VALUES (%s, %s, %s, 'checkpoint', %s, 'available', 'local', 'test',
                           %s, 'application/octet-stream', 'standard', 10, %s, false)""",
                (aid, tenant, job_id, f"shard-{i}.pt", f"k/{aid}", f"{i}" * 64),
            )
            available.append(aid)
        aid = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO storage.artifacts
                 (artifact_id, tenant_id, job_id, artifact_type, logical_name, state,
                  primary_provider, primary_bucket, object_key, content_type,
                  retention_class, legal_hold)
               VALUES (%s, %s, %s, 'checkpoint', 'partial.pt', 'uploading', 'local',
                       'test', %s, 'application/octet-stream', 'standard', false)""",
            (aid, tenant, job_id, f"k/{aid}"),
        )
        uploading.append(aid)

    yield {"tenant": tenant, "job_id": job_id, "available": available, "uploading": uploading}

    with pg_transaction() as conn:
        conn.execute("DELETE FROM volume_promotions WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def _holds(tenant: str) -> set[str]:
    with pg_transaction() as conn:
        rows = conn.execute(
            "SELECT artifact_id FROM storage.artifacts "
            "WHERE tenant_id = %s AND legal_hold = true",
            (tenant,),
        ).fetchall()
    return {str(r[0]) for r in rows}


# ── §3.3: no promotion begins without a hold ────────────────────────


def test_the_hold_covers_every_artifact_in_the_manifest(job):
    """The constraint, stated as §0 states it."""
    from artifacts import get_artifact_manager, take_promotion_hold

    manifest = get_artifact_manager().resolve_promotion_manifest(
        job["job_id"], tenant_id=job["tenant"]
    )
    with pg_transaction() as conn:
        held_count = take_promotion_hold(conn, job["job_id"], job["tenant"])

    held = _holds(job["tenant"])
    assert held_count == manifest["file_count"] == 3
    assert held == set(job["available"]), (
        "the hold does not cover exactly the manifest — an artifact the copy "
        "will read can still be deleted mid-copy, and the symptom is a partial "
        "volume the user believes is complete"
    )


def test_an_artifact_outside_the_manifest_is_not_held(job):
    """The hold matches the manifest, not the job.

    An artifact still uploading is not promotable, so holding it would retain
    something the promotion never reads.
    """
    from artifacts import take_promotion_hold

    with pg_transaction() as conn:
        take_promotion_hold(conn, job["job_id"], job["tenant"])
    assert not (_holds(job["tenant"]) & set(job["uploading"]))


def test_the_hold_is_released_on_completion(job):
    from artifacts import release_promotion_hold, take_promotion_hold

    with pg_transaction() as conn:
        take_promotion_hold(conn, job["job_id"], job["tenant"])
    assert _holds(job["tenant"])
    with pg_transaction() as conn:
        release_promotion_hold(conn, job["job_id"], job["tenant"])
    assert _holds(job["tenant"]) == set()


def test_a_stale_promotion_is_swept_and_its_hold_released(job):
    """§3.3: the release belongs in the sweep, "not only on the success path".

    An agent that dies mid-copy leaves a `running` row and artifacts that no
    longer expire. Nothing surfaces that — the user sees a promotion which never
    finishes — so the sweep is the only thing that closes the loop.
    """
    from artifacts import sweep_stale_promotions, take_promotion_hold

    promotion_id = str(uuid.uuid4())
    with pg_transaction() as conn:
        take_promotion_hold(conn, job["job_id"], job["tenant"])
        conn.execute(
            """INSERT INTO volume_promotions
                 (promotion_id, tenant_id, job_id, volume_id, idempotency_key,
                  manifest_sha256, file_count, total_bytes, state, updated_at)
               VALUES (%s, %s, %s, 'vol-x', %s, %s, 3, 30, 'running',
                       clock_timestamp() - interval '2 days')""",
            (promotion_id, job["tenant"], job["job_id"], f"k-{promotion_id}", "a" * 64),
        )
    assert _holds(job["tenant"]), "precondition: the hold was taken"

    with pg_transaction() as conn:
        swept = sweep_stale_promotions(conn)

    assert swept >= 1
    assert _holds(job["tenant"]) == set(), (
        "a stale promotion's hold survived the sweep — the leak §3.3 names, "
        "where a crashed agent retains a tenant's artifacts indefinitely"
    )
    with pg_transaction() as conn:
        state = conn.execute(
            "SELECT state, failure_code FROM volume_promotions WHERE promotion_id = %s",
            (promotion_id,),
        ).fetchone()
    assert state[0] == "abandoned" and state[1] == "stale"


def test_a_live_promotion_is_not_swept(job):
    """The other half. A sweep that abandons running work is worse than none."""
    from artifacts import sweep_stale_promotions

    promotion_id = str(uuid.uuid4())
    with pg_transaction() as conn:
        conn.execute(
            """INSERT INTO volume_promotions
                 (promotion_id, tenant_id, job_id, volume_id, idempotency_key,
                  manifest_sha256, file_count, total_bytes, state)
               VALUES (%s, %s, %s, 'vol-x', %s, %s, 3, 30, 'running')""",
            (promotion_id, job["tenant"], job["job_id"], f"k-{promotion_id}", "b" * 64),
        )
        sweep_stale_promotions(conn)
        state = conn.execute(
            "SELECT state FROM volume_promotions WHERE promotion_id = %s",
            (promotion_id,),
        ).fetchone()
    assert state[0] == "running", "the sweep abandoned a promotion that had just started"


def test_the_create_route_takes_the_hold_before_the_copy_is_enqueued():
    """Order matters, so it is asserted against the source.

    Enqueuing first and holding afterwards leaves a window in which the agent is
    already fetching a manifest whose artifacts the janitor may delete.
    """
    import ast
    import inspect

    import routes.volumes as vol

    # AST, not substring. The first version of this test asserted
    # `"take_promotion_hold" in src`, which the *import line* satisfies — so
    # deleting the actual call left the test green. Probing found that, and it
    # is the same decorative-assertion defect this file exists to prevent
    # elsewhere.
    tree = ast.parse(inspect.cleandoc(inspect.getsource(vol.api_volume_promotion_create)))
    calls = [
        (node.lineno, getattr(node.func, "id", "") or getattr(node.func, "attr", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    names = [name for _, name in calls]
    assert "take_promotion_hold" in names, (
        "the create route never *calls* take_promotion_hold — a promotion would "
        "begin with its artifacts still deletable"
    )
    assert "enqueue_agent_command" in names, "the create route enqueues nothing"
    hold_line = min(ln for ln, name in calls if name == "take_promotion_hold")
    enqueue_line = min(ln for ln, name in calls if name == "enqueue_agent_command")
    assert hold_line < enqueue_line, (
        "the hold is taken after the command is enqueued — the agent can begin "
        "reading artifacts that are not yet held"
    )


# ── §3.1: no presigned URL in a command row ─────────────────────────


def test_the_command_carries_only_the_promotion_id():
    """§3.1, asserted on the argument the route actually enqueues.

    Presigned URLs are read grants on a tenant's weights; a command row is
    queued, logged and retained. A file list would also blow the 16 KB args cap
    on a sharded checkpoint.
    """
    import ast
    import inspect

    import routes.volumes as vol

    src = inspect.getsource(vol.api_volume_promotion_create)
    tree = ast.parse(inspect.cleandoc(src))
    enqueue_args = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "enqueue_agent_command":
            # (host_id, command, args_dict, ...)
            enqueue_args = node.args[2] if len(node.args) > 2 else None
    assert enqueue_args is not None, "the create route does not enqueue a command"
    assert isinstance(enqueue_args, ast.Dict), "the command args are not a literal dict"
    keys = {k.value for k in enqueue_args.keys if isinstance(k, ast.Constant)}
    assert keys == {"promotion_id"}, (
        f"the command row carries {sorted(keys)}; §3.1 allows only promotion_id, "
        "because anything else ends up queued, logged and retained"
    )


def test_the_manifest_endpoint_is_the_only_place_urls_appear():
    """And it is worker-authenticated, which is the point of splitting it."""
    import inspect

    import routes.volumes as vol

    manifest_src = inspect.getsource(vol.api_promotion_manifest_for_agent)
    assert "_require_platform_worker" in manifest_src, (
        "the endpoint that hands out presigned URLs is reachable by something "
        "other than a platform worker"
    )
    preview_src = inspect.getsource(vol.api_volume_promotion_preview)
    assert "generate_download_url" not in preview_src, (
        "the customer-facing preview is generating presigned URLs; only the "
        "worker-authenticated manifest may"
    )
