"""Where an unattached volume's promotion runs.

§3.4 of `docs/artifact-promotion-plan.md` calls this "the genuinely open
question": an attached volume has an obvious host — the instance's, which can
already see the mount — and an unattached one has none. The plan weighs three
answers and picks the least-loaded active host **in the volume's region**,
reusing the mount commands the agent already has.

## The two properties that are not interchangeable

**Region is a hard filter.** Volumes are NFS exports. A host outside the region
either cannot reach the export at all or pays a cross-region transfer for every
byte of a multi-gigabyte checkpoint. A "prefer the region, fall back to
anywhere" rule reads as more robust and is worse: it converts an unroutable
promotion into a silently expensive one, and the user sees neither.

**Load is the tie-break, not the filter.** Picking the first row returned would
put every promotion on the same host, which is how a feature that is fine in
testing becomes an I/O problem in production.

## Why "no host" is not an error

`""` means nothing in that region is active. The promotion row is still created:
it is the durable record of what the user asked for, and a sweep can place it
when capacity appears. Refusing would discard a request made for a reason, and
the caller cannot tell "no capacity right now" from "this will never work".
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = _c.execute("SELECT to_regclass('hosts')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")


@pytest.fixture
def hosts():
    """Hosts this test owns, in two regions, with differing load."""
    tag = uuid.uuid4().hex[:8]
    made = {
        "busy_east": f"h-busy-east-{tag}",
        "idle_east": f"h-idle-east-{tag}",
        "idle_west": f"h-idle-west-{tag}",
        "down_east": f"h-down-east-{tag}",
    }
    jobs = []
    with pg_transaction() as conn:
        for key, hid in made.items():
            conn.execute(
                "INSERT INTO hosts (host_id, status, registered_at, payload, region) "
                "VALUES (%s, %s, extract(epoch from now()), '{}'::jsonb, %s)",
                (hid, "offline" if key.startswith("down") else "active",
                 f"reg-west-{tag}" if "west" in key else f"reg-east-{tag}"),
            )
        # Two live jobs on the busy host so ordering by load is decidable.
        for i in range(2):
            jid = f"j-{tag}-{i}"
            conn.execute(
                "INSERT INTO jobs (job_id, status, priority, submitted_at, payload, host_id) "
                "VALUES (%s, 'running', 0, extract(epoch from now()), '{}'::jsonb, %s)",
                (jid, made["busy_east"]),
            )
            jobs.append(jid)

    yield {"ids": made, "east": f"reg-east-{tag}", "west": f"reg-west-{tag}", "tag": tag}

    with pg_transaction() as conn:
        for jid in jobs:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (jid,))
        for hid in made.values():
            conn.execute("DELETE FROM hosts WHERE host_id = %s", (hid,))


def _pick(region: str) -> str:
    from routes.volumes import _pick_promotion_host

    return _pick_promotion_host(region)


def test_the_least_loaded_host_in_the_region_wins(hosts):
    """Load is the tie-break — otherwise every promotion lands on one box."""
    assert _pick(hosts["east"]) == hosts["ids"]["idle_east"], (
        "placement chose the busy host over an idle one in the same region"
    )


def test_a_host_in_another_region_is_never_chosen(hosts):
    """The property that matters most, and the one a 'fallback' would break.

    The west host is idle and would win on load alone. It must lose on region,
    because a volume's NFS export is not reachable from it.
    """
    chosen = _pick(hosts["east"])
    assert chosen != hosts["ids"]["idle_west"], (
        "an out-of-region host was chosen — the export is unreachable from "
        "there, or every byte crosses a region boundary"
    )


def test_an_inactive_host_is_never_chosen(hosts):
    """`status = 'active'` is a filter, not a preference."""
    assert _pick(hosts["east"]) != hosts["ids"]["down_east"]


def test_a_region_with_no_hosts_yields_no_host_rather_than_a_wrong_one(hosts):
    """"Nowhere to run it" must not silently become "run it anywhere"."""
    assert _pick(f"reg-empty-{hosts['tag']}") == "", (
        "a region with no active hosts returned a host anyway — a promotion "
        "would be queued for a machine that cannot reach the volume"
    )


def test_an_empty_region_matches_anything(hosts):
    """A volume with no region recorded is placeable rather than stuck.

    This is the one case where "any host" is right: there is no region to
    violate. It is deliberate, and separated from the region-filtering tests so
    it cannot be mistaken for a fallback.
    """
    assert _pick("") != ""


def test_placement_failure_returns_empty_rather_than_raising(monkeypatch):
    """The create path calls this. It must never be why a promotion 500s."""
    import routes.volumes as vol

    def explode(*a, **k):
        raise RuntimeError("database is on fire")

    monkeypatch.setattr(vol, "control_plane_transaction", explode, raising=False)
    import control_plane.db as cpdb

    monkeypatch.setattr(cpdb, "control_plane_transaction", explode)
    assert vol._pick_promotion_host("anywhere") == ""
