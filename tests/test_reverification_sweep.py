"""The reverification sweep — C2's first commit, and why it comes first.

`list_hosts_needing_reverification()` has existed and worked for months. Its
wrapper had **no callers**, so `next_check_at` was written and never read and a
verified stamp could age indefinitely against a one-day interval. A
`require_verified` control over a fact nothing maintains would teach users the
feature is broken.

These tests drive the pass that asks, and the guards that stop it becoming a
load spike or a queue.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = _c.execute("SELECT to_regclass('agent_commands')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database lacks agent_commands")

import verification_sweep as sweep  # noqa: E402


@pytest.fixture
def hosts():
    tag = uuid.uuid4().hex[:12]
    ids = [f"sweep-{tag}-{n}" for n in range(5)]
    yield ids
    with pg_transaction() as conn:
        conn.execute("DELETE FROM agent_commands WHERE host_id = ANY(%s)", (ids,))


def _queue(conn, host_id, *, command=sweep.REVERIFY_COMMAND, status="pending", ttl=600):
    # `ck_agent_commands_claim_shape` requires an owner and an expiry on a
    # claimed row — a claim that names nobody is not a claim.
    claim = ("worker-1", "clock_timestamp() + interval '5 min'") if status == "claimed" else (None, "NULL")
    conn.execute(
        f"""INSERT INTO agent_commands
              (host_id, command, args, status, created_at, expires_at,
               claim_owner, claim_expires_at)
            VALUES (%s, %s, '{{}}'::jsonb, %s, EXTRACT(EPOCH FROM NOW()),
                    EXTRACT(EPOCH FROM NOW()) + %s, %s, {claim[1]})""",
        (host_id, command, status, ttl, claim[0]),
    )


class _Engine:
    def __init__(self, due):
        self._due = list(due)

    def get_hosts_needing_reverification(self):
        return list(self._due)


@pytest.fixture
def asked(monkeypatch):
    """Capture what the sweep would enqueue, without touching the queue."""
    calls = []

    def _enqueue(host_id, command, args=None, created_by=None, ttl_sec=900):
        calls.append(
            {"host_id": host_id, "command": command, "args": args or {},
             "created_by": created_by, "ttl_sec": ttl_sec}
        )
        return len(calls)

    import routes.agent as agent_routes

    monkeypatch.setattr(agent_routes, "enqueue_agent_command", _enqueue)
    return calls


def _with_due(monkeypatch, due):
    import verification

    monkeypatch.setattr(verification, "get_verification_engine", lambda: _Engine(due))


# ── The command name must exist on both sides ─────────────────────────


def test_the_command_is_in_both_allowlists():
    """The API rejects unknown commands and the agent hard-refuses them.

    `test_worker_agent_allowlist.py` already asserts the two sets are equal;
    this asserts the set they agree on contains **this** command. Equal-and-both-
    wrong turns the whole sweep into a no-op that logs success, which is the
    failure mode this phase keeps finding.
    """
    import worker_agent
    from routes.agent import _AGENT_COMMAND_ALLOWED as api_side

    assert sweep.REVERIFY_COMMAND in api_side, "the API would reject the enqueue"
    assert sweep.REVERIFY_COMMAND in worker_agent._AGENT_COMMAND_ALLOWED, (
        "the agent would hard-refuse the command the sweep sends"
    )


# ── The pass itself ───────────────────────────────────────────────────


def test_every_overdue_host_is_asked_once(monkeypatch, hosts, asked):
    _with_due(monkeypatch, hosts[:3])
    summary = sweep.run_sweep()

    assert summary["due"] == 3
    assert summary["asked"] == 3
    assert [c["host_id"] for c in asked] == hosts[:3]
    assert {c["command"] for c in asked} == {sweep.REVERIFY_COMMAND}
    assert asked[0]["created_by"] == "verification_sweep"
    assert asked[0]["ttl_sec"] == sweep.SWEEP_INTERVAL_SEC, (
        "a request that outlives the interval lets a backlog build for an "
        "offline host"
    )


def test_a_host_already_holding_a_request_is_not_asked_again(monkeypatch, hosts, asked):
    """Asking twice runs a 60-second GPU benchmark twice, on real hardware."""
    with pg_transaction() as conn:
        _queue(conn, hosts[0])

    _with_due(monkeypatch, hosts[:3])
    summary = sweep.run_sweep()

    assert summary["already_pending"] == 1
    assert summary["asked"] == 2
    assert hosts[0] not in {c["host_id"] for c in asked}


def test_an_expired_request_does_not_count_as_pending(monkeypatch, hosts, asked):
    """Otherwise one undelivered command silences the host forever."""
    with pg_transaction() as conn:
        _queue(conn, hosts[0], ttl=-60)

    _with_due(monkeypatch, [hosts[0]])
    assert sweep.run_sweep()["asked"] == 1


def test_a_different_command_does_not_count_as_pending(monkeypatch, hosts, asked):
    with pg_transaction() as conn:
        _queue(conn, hosts[0], command="upgrade_agent")

    _with_due(monkeypatch, [hosts[0]])
    assert sweep.run_sweep()["asked"] == 1


def test_a_claimed_request_does_not_count_as_pending(monkeypatch, hosts, asked):
    """`claimed` means it is on its way to a host that may never report."""
    with pg_transaction() as conn:
        _queue(conn, hosts[0], status="claimed")

    _with_due(monkeypatch, [hosts[0]])
    assert sweep.run_sweep()["asked"] == 1


def test_one_pass_is_capped(monkeypatch, hosts, asked):
    """The first sweep after this ships sees every overdue host at once.

    A hundred hosts running a 60-second GPU benchmark in the same minute is a
    self-inflicted load spike; the ones that miss the cut are still overdue an
    hour later.
    """
    _with_due(monkeypatch, hosts)
    summary = sweep.run_sweep(limit=2)

    assert summary["asked"] == 2
    assert summary["skipped_over_limit"] == 3


def test_one_unreachable_host_does_not_stop_the_rest(monkeypatch, hosts):
    """A queue-full or offline host is not a reason to skip the fleet."""
    calls = []

    def _enqueue(host_id, command, args=None, created_by=None, ttl_sec=900):
        if host_id == hosts[1]:
            raise RuntimeError("agent queue full")
        calls.append(host_id)
        return len(calls)

    import routes.agent as agent_routes

    monkeypatch.setattr(agent_routes, "enqueue_agent_command", _enqueue)
    _with_due(monkeypatch, hosts[:3])
    summary = sweep.run_sweep()

    assert summary["asked"] == 2
    assert summary["failed"] == 1
    assert calls == [hosts[0], hosts[2]]


def test_nothing_due_asks_nobody(monkeypatch, asked):
    _with_due(monkeypatch, [])
    summary = sweep.run_sweep()
    assert summary == {
        "enabled": True, "due": 0, "asked": 0,
        "already_pending": 0, "failed": 0, "skipped_over_limit": 0,
    }
    assert asked == []


def test_the_sweep_can_be_turned_off(monkeypatch, hosts, asked):
    monkeypatch.setenv("XCELSIOR_REVERIFY_SWEEP_ENABLED", "false")
    _with_due(monkeypatch, hosts)
    summary = sweep.run_sweep()
    assert summary["enabled"] is False
    assert summary["asked"] == 0
    assert asked == []


# ── The production condition, end to end ──────────────────────────────


def test_the_real_due_query_finds_the_production_condition(hosts, asked):
    """No monkeypatched engine: the store's own query, on a realistic mix.

    Three hosts, one of each kind a fleet holds — a long-overdue verified stamp,
    a verified stamp not yet due, and an unverified host. Only the first is
    asked. This is the pass that would have caught the drift.
    """
    overdue, current, unverified = hosts[0], hosts[1], hosts[2]
    now = time.time()
    with pg_transaction() as conn:
        for host_id, state, last, nxt in (
            (overdue, "verified", now - 112 * 86400, now - 111 * 86400),
            (current, "verified", now - 3600, now + 82800),
            (unverified, "unverified", None, None),
        ):
            conn.execute(
                """INSERT INTO host_verifications
                     (host_id, verification_id, state, verified_at,
                      last_check_at, next_check_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (host_id, f"v-{host_id}", state, last, last, nxt),
            )

    try:
        summary = sweep.run_sweep()
        assert overdue in {c["host_id"] for c in asked}, (
            "the sweep did not ask the host whose stamp is months past due"
        )
        assert current not in {c["host_id"] for c in asked}
        assert unverified not in {c["host_id"] for c in asked}, (
            "an unverified host has nothing to re-verify; asking it would burn a "
            "benchmark to learn what is already recorded"
        )
        assert summary["asked"] >= 1
    finally:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM host_verifications WHERE host_id = ANY(%s)",
                ([overdue, current, unverified],),
            )


# ── What it deliberately does not do ──────────────────────────────────


def test_the_sweep_does_not_change_any_verification_state(monkeypatch, hosts, asked):
    """Expiring a stamp would change where every job in the marketplace lands.

    `scheduler.allocate_best_host` prefers hosts whose state is `verified`, so
    moving an overdue host out of that state is a placement change, not a
    hygiene change — §5.4's reconciliation, and a C2 decision not made in
    passing. The placement gate already reads an overdue stamp as `stale`.
    """
    host_id = hosts[0]
    now = time.time()
    with pg_transaction() as conn:
        conn.execute(
            """INSERT INTO host_verifications
                 (host_id, verification_id, state, verified_at, last_check_at, next_check_at)
               VALUES (%s, %s, 'verified', %s, %s, %s)""",
            (host_id, f"v-{host_id}", now - 200 * 86400, now - 200 * 86400, now - 199 * 86400),
        )

    _with_due(monkeypatch, [host_id])
    sweep.run_sweep()

    with pg_transaction() as conn:
        state = conn.execute(
            "SELECT state FROM host_verifications WHERE host_id = %s", (host_id,)
        ).fetchone()[0]
        conn.execute("DELETE FROM host_verifications WHERE host_id = %s", (host_id,))

    assert state == "verified", "the sweep changed placement behaviour as a side effect"


# ── The expired rows the drain never gets to ──────────────────────────


def test_expired_commands_are_pruned_without_an_agent_draining():
    """The cleanup must not depend on the thing that is broken.

    `api_agent_commands_drain` deletes expired rows, but only when an agent
    collects commands — so it stops running in exactly the case that produces
    the most garbage: a fleet that cannot reach the API. Production accumulated
    40 expired reverify rows against 2 live ones that way.
    """
    from routes.agent import prune_expired_agent_commands

    tag = uuid.uuid4().hex[:12]
    host = f"prune-{tag}"
    try:
        with pg_transaction() as conn:
            _queue(conn, host, ttl=-60)      # expired
            _queue(conn, host, ttl=-120)     # expired
            _queue(conn, host, ttl=3600)     # live
            before = conn.execute(
                "SELECT count(*) FROM agent_commands WHERE host_id = %s", (host,)
            ).fetchone()[0]
        assert before == 3

        with pg_transaction() as conn:
            prune_expired_agent_commands(conn)
            remaining = conn.execute(
                "SELECT count(*) FROM agent_commands WHERE host_id = %s", (host,)
            ).fetchone()[0]
        assert remaining == 1, "the live command was pruned, or the expired ones were not"
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM agent_commands WHERE host_id = %s", (host,))


def test_an_attempt_bound_command_is_never_pruned():
    """A v2 `start_attempt` carries its own lifecycle and its only delivery.

    Claim/ACK, backoff, dead-letter and cancellation on lease expiry are owned
    by `/agent/v2` and `control_plane.commands`. Deleting one here would destroy
    a placement's sole delivery — the same reason the drain excludes them.
    """
    from routes.agent import prune_expired_agent_commands

    tag = uuid.uuid4().hex[:12]
    host = f"prune-att-{tag}"
    attempt_id = str(uuid.uuid4())
    try:
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO agent_commands
                     (host_id, command, args, status, created_at, expires_at, attempt_id)
                   VALUES (%s, 'start_attempt', '{}'::jsonb, 'pending',
                           EXTRACT(EPOCH FROM NOW()), EXTRACT(EPOCH FROM NOW()) - 60, %s)""",
                (host, attempt_id),
            )
        with pg_transaction() as conn:
            prune_expired_agent_commands(conn)
            remaining = conn.execute(
                "SELECT count(*) FROM agent_commands WHERE host_id = %s", (host,)
            ).fetchone()[0]
        assert remaining == 1, (
            "an expired attempt-bound command was pruned; its lifecycle is not "
            "this sweep's to end"
        )
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM agent_commands WHERE host_id = %s", (host,))


def test_the_sweep_is_registered_with_the_background_worker():
    """A pruner with no caller is the defect this whole branch started from."""
    import inspect

    import bg_worker

    source = inspect.getsource(bg_worker.main)
    assert '"expired_agent_command_sweep"' in source
    assert "prune_expired_agent_commands" in source
