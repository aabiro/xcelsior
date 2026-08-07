"""Starting a job over starts its output over — through either door.

Two routes requeue an instance, and both call the same authority:

    /instance/{job_id}/requeue          scheduler.requeue_job(user_initiated=True)
    /api/v1/instances/{job_id}/retry    scheduler.requeue_job(user_initiated=True, ...)

They were never competing implementations. What differed was that the log wipe
lived in the *first route* rather than in the shared function, so a retry
through the v1 path — the one `retry_instance` calls, and therefore the one an
agent uses — left the previous attempt's output in place. The next run's logs
began with the last run's failure, which reads as the new attempt failing
identically.

The wipe now lives in `requeue_job`, gated on `user_initiated`:

* both user-facing doors pass it, so they agree;
* automatic failover does not, because a failover that erased the logs
  explaining why it failed over would destroy the evidence for the retry it
  just performed.

Ordering is asserted too. The original call site carried the comment "wipe
previous attempt logs BEFORE requeue — never after, or we delete the lifecycle
lines emit_lifecycle_log just wrote", and that hazard survives the move: inside
`requeue_job` the wipe must still precede the `emit_lifecycle_log` call a few
lines below it.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap

os.environ.setdefault("XCELSIOR_ENV", "test")


def _requeue_source() -> str:
    import scheduler

    return textwrap.dedent(inspect.getsource(scheduler.requeue_job))


def test_both_routes_delegate_to_the_one_authority():
    """Prove the premise. If either stopped delegating, the rest is moot."""
    import routes.control_plane_v1 as v1
    import routes.instances as legacy

    for handler in (legacy.api_requeue_instance, v1.api_v1_instance_retry):
        source = textwrap.dedent(inspect.getsource(handler))
        calls = {
            getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
        }
        assert "requeue_job" in calls, (
            f"{handler.__name__} no longer calls requeue_job, so the two doors "
            "have diverged into separate implementations"
        )


def test_the_wipe_lives_in_the_shared_authority():
    """Not in one route, where it only served that route's callers."""
    assert "_clear_job_output" in _requeue_source(), (
        "requeue_job no longer clears the previous attempt's output, so a retry "
        "through /api/v1/instances/{job_id}/retry leaves stale logs in place"
    )


def test_no_route_clears_the_logs_itself():
    """One implementation, not two that can drift apart.

    A route that keeps its own copy would keep working — and would quietly
    become the only path that behaves correctly the next time the shared one
    changes.
    """
    import routes.instances as legacy

    source = textwrap.dedent(inspect.getsource(legacy.api_requeue_instance))
    assert "DELETE FROM job_logs" not in source, (
        "the requeue route deletes job_logs itself again; the shared authority "
        "already does it, and two copies is how the doors drifted in the first place"
    )


def test_the_wipe_precedes_the_lifecycle_write():
    """The hazard the original comment warned about, asserted rather than trusted.

    `emit_lifecycle_log` writes the queued-transition lines. Clearing logs after
    it would delete exactly those.
    """
    source = _requeue_source()
    wipe = source.index("_clear_job_output(")
    emit = source.index("emit_lifecycle_log(")
    assert wipe < emit, (
        "the log wipe now runs after emit_lifecycle_log, so it deletes the "
        "lifecycle lines that call just wrote — the precise failure the original "
        "call site's comment warned about"
    )


def test_automatic_failover_keeps_its_evidence():
    """The gate is on intent, not on route.

    A failover requeue must not erase the logs that explain the failure it is
    reacting to.
    """
    source = _requeue_source()
    tree = ast.parse(source)
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "user_initiated" not in test_names:
            continue
        called = {
            getattr(c.func, "id", "") or getattr(c.func, "attr", "")
            for stmt in node.body
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
        }
        if "_clear_job_output" in called:
            guarded = True
    assert guarded, (
        "the log wipe is not gated on user_initiated, so automatic failover now "
        "erases the logs describing the failure it is retrying"
    )
