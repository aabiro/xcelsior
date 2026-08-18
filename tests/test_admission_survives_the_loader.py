"""A host the control plane has not admitted must not be placeable.

The property is true today. This pins it because it is held by **two pieces
agreeing**, and each looks correct alone while the pair is one edit from failing
open.

## Why the pair, and not the filter

`filters.host_admitted` reads `host.get("administrative_state") or "admitted"`.
The default is the hazard: a host dict **missing** the field reads as admitted.
Handed a raw `hosts.payload` row — which carries `admitted` and `admission_state`
but not `administrative_state` — the filter passes a host the database records
as `pending`.

Nothing does that today. `snapshot.py` builds the dict the scheduler uses and
sets the field from the column, falling back to the 054 projection
(`"admitted" if host["admitted"] else "pending"`) when the column is empty. So
the loader supplies exactly what the filter's default would otherwise paper
over.

That is the whole risk: a future loader that forgets the field does not fail —
it admits everything, silently, and no test of `host_admitted` alone would
notice because the filter is behaving as written.

## What this asserts

The **composition**: a row whose `administrative_state` is not `admitted`,
carried through the loader's construction, is refused by the filter. Testing
either half alone is what produced two wrong conclusions while writing this —
first that the filter fails open (it does, but nothing feeds it raw payloads),
then that a pending host was placeable (it is not; the loader sets the field).

`h-skep-leg-750fcb` is the real row this was found on: registered 29 days ago,
never observed, `administrative_state = 'pending'`, and `hosts.status = 'active'`.

**Those two disagreeing is correct, not a defect**, and reading it as one is the
mistake this paragraph exists to prevent. Migration 054 split a single `status`
into two questions — `administrative_state` is what an operator *decided*,
`availability_state` is what is *observed* — and 059 added
`control_plane_project_host()` to derive both:

    administrative_state := CASE
        WHEN status = 'disabled'                                  THEN 'disabled'
        WHEN admission_state = 'admitted' AND status = 'draining'  THEN 'draining'
        WHEN admission_state = 'admitted'                          THEN 'admitted'
        ELSE 'pending' END
    availability_state := CASE status
        WHEN 'active' THEN 'ready' WHEN 'dead' THEN 'not_ready' ELSE 'unknown' END

So `status='active'` means the box is reachable and yields
`availability_state='ready'`; `admission_state` was never `admitted`, so the
`ELSE` branch gives `pending`. A machine that is up and not approved. The
columns *should* differ there.

`status` is therefore not dead — it is the write-side input during
expand-contract, still written at registration (`db.py`) and projected forward.
Dropping it is the contract phase, **B16.2**, which
`docs/track-b-implementation-checklist.md` requires to remain last in the chain.
Until then both exist by design.

The `status='active'` selection in `scheduler.py` feeds only heartbeat freshness
and `autoscale_down`, never a placement decision — placement reads
`administrative_state` through the loader above.
"""

from __future__ import annotations

import pytest

from control_plane.scheduler.filters import host_admitted


def _host_dict_as_the_loader_builds_it(
    *, host_id: str, payload: dict, administrative_state: str | None, status: str
) -> dict:
    """Reproduce `control_plane/scheduler/snapshot.py`'s construction.

    Kept as a small copy on purpose. Importing the real builder would mean
    standing up a database connection and a full snapshot for a question about
    three fields, and the copy is checked against the original by
    `test_the_loader_still_sets_the_field_this_relies_on`.
    """
    host = dict(payload)
    host["host_id"] = host_id
    host["status"] = status or host.get("status")
    if not administrative_state:
        administrative_state = "admitted" if host.get("admitted") else "pending"
    host["administrative_state"] = administrative_state
    return host


# ── The property ──────────────────────────────────────────────────────


@pytest.mark.parametrize("state", ["pending", "suspended", "evicted", "quarantined"])
def test_a_host_that_is_not_admitted_is_refused(state: str):
    host = _host_dict_as_the_loader_builds_it(
        host_id="h-probe",
        payload={"admitted": False, "admission_state": state, "region": "ca-east"},
        administrative_state=state,
        status="active",  # the legacy column says active; it is not the authority
    )
    refusal = host_admitted({}, host, None)
    assert refusal is not None, f"a host in {state!r} was accepted for placement"
    assert refusal.code == "host_not_admitted"


def test_an_admitted_host_is_accepted():
    """The other direction, so the guard cannot pass by refusing everything."""
    host = _host_dict_as_the_loader_builds_it(
        host_id="h-probe",
        payload={"admitted": True, "admission_state": "admitted"},
        administrative_state="admitted",
        status="active",
    )
    assert host_admitted({}, host, None) is None


def test_an_empty_column_falls_back_to_the_legacy_flag_and_still_refuses():
    """054's projection. A row predating the column must not become admitted.

    This is the path where the filter's `or "admitted"` default would take over
    if the loader stopped projecting, so it is asserted separately from the
    populated-column case above.
    """
    host = _host_dict_as_the_loader_builds_it(
        host_id="h-legacy",
        payload={"admitted": False},
        administrative_state=None,
        status="active",
    )
    assert host["administrative_state"] == "pending"
    assert host_admitted({}, host, None) is not None


# ── The assumption this rests on ──────────────────────────────────────


def test_the_filter_still_fails_open_without_the_loader():
    """Named, not fixed.

    A raw `hosts.payload` has no `administrative_state`, so the filter admits
    it — including a row the database records as `pending`. That is why the
    property above belongs to the *pair*: the loader is what makes the default
    unreachable. Changing the filter to fail closed would be the obvious fix and
    is not obviously right — every caller would then have to supply the field,
    and one that forgets would break placement entirely rather than loosen it.
    Recorded so the next reader knows the default is load-bearing.
    """
    raw_payload = {"admitted": False, "admission_state": "pending", "host_id": "h-raw"}
    assert "administrative_state" not in raw_payload
    assert host_admitted({}, raw_payload, None) is None, (
        "the filter now refuses a bare payload — if that was deliberate, this "
        "test should be deleted and the loader's projection re-examined"
    )


def test_the_loader_still_sets_the_field_this_relies_on():
    """The copy above must not drift from `snapshot.py`."""
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "control_plane"
        / "scheduler"
        / "snapshot.py"
    ).read_text(encoding="utf-8")
    assert 'host["administrative_state"] = admin_state' in source, (
        "snapshot.py no longer sets administrative_state on the host dict, so "
        "filters.host_admitted falls back to its permissive default and every "
        "host reads as admitted"
    )
    assert '"admitted" if host.get("admitted") else "pending"' in source, (
        "the 054 projection is gone; a row with an empty column no longer "
        "falls back to the legacy admission flag"
    )
