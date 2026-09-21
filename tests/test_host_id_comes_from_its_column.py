"""A host row is identified by its column, not by a copy inside its payload.

`hosts.host_id` is the primary key — what `delete_host`, every `ON CONFLICT`,
and every foreign key resolve by. Registration *also* writes the id into the
`payload` JSON, and both host loaders used to read only the payload. That made
the duplicate load-bearing: a row written by anything that did not copy it in
produced a host dict with no `host_id` at all.

`routes/hosts.py` indexes `h["host_id"]` in nine places, including
`_resolve_host_id`, which every admission route funnels through. So a single
such row did not break lookups *for that host* — it raised `KeyError` and
returned 500 for **every** host lookup in the process, because the failure is
in the scan, not the match.

It survived because the query is written twice: `db.DatabaseOps.load_hosts` for
sqlite and `scheduler._load_hosts_from_conn` for postgres. Fixing the first
alone would have left production untouched.

Found by the first test that ever called the admission routes over HTTP.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from control_plane.db import run_transaction
from scheduler import list_hosts


@pytest.fixture
def host_without_host_id_in_payload():
    """Exactly the shape that broke: the column set, the payload copy absent."""
    host_id = f"no-payload-id-{uuid.uuid4().hex[:10]}"

    def insert(conn):
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  tenant_id, owner_id, admission_state)
               VALUES (%s, 'active', %s, %s::jsonb, %s, %s, 'pending')
               ON CONFLICT (host_id) DO NOTHING""",
            (
                host_id,
                time.time(),
                json.dumps({"gpu_model": "RTX 4090", "total_vram_gb": 24}),
                "tenant-column-test",
                "tenant-column-test",
            ),
        )

    run_transaction(insert, what="test_host_id_column_insert")
    yield host_id

    def drop(conn):
        conn.execute("DELETE FROM hosts WHERE host_id = %s", (host_id,))

    run_transaction(drop, what="test_host_id_column_drop")


def test_the_loader_fills_host_id_from_the_column(host_without_host_id_in_payload) -> None:
    hosts = list_hosts(active_only=False)
    match = [h for h in hosts if h.get("host_id") == host_without_host_id_in_payload]
    assert match, (
        "a host whose payload carries no host_id is invisible to list_hosts; "
        "the primary key is right there in the column"
    )


def test_every_loaded_host_has_an_id(host_without_host_id_in_payload) -> None:
    """The property the callers depend on, stated once.

    Nine call sites index `h["host_id"]` directly. They are correct to, as long
    as this holds — and it is cheaper to hold it here than to make every caller
    defensive.
    """
    missing = [h for h in list_hosts(active_only=False) if not h.get("host_id")]
    assert not missing, (
        f"{len(missing)} loaded host(s) have no host_id; every `h[\"host_id\"]` in "
        "routes/hosts.py raises KeyError on the whole scan, so one bad row is a "
        "500 for every host"
    )


def test_resolving_such_a_host_does_not_fault(host_without_host_id_in_payload) -> None:
    """`_resolve_host_id` is the funnel; a 500 here is a 500 for every host route."""
    from routes.hosts import _resolve_host_id

    resolved, hosts = _resolve_host_id(host_without_host_id_in_payload)
    assert resolved == host_without_host_id_in_payload
    assert hosts


def test_both_loaders_read_the_column() -> None:
    """The query exists twice; the sqlite copy was fixed first and alone.

    Asserted against the source because only one branch runs in any given
    configuration, so a behavioural test can only ever cover the backend it
    happens to be running under — and the one that was broken in production is
    the one tests do not exercise here.
    """
    import inspect

    from db import DatabaseOps
    import scheduler

    for label, fn in (
        ("db.DatabaseOps.load_hosts", DatabaseOps.load_hosts),
        ("scheduler._load_hosts_from_conn", scheduler._load_hosts_from_conn),
    ):
        source = inspect.getsource(fn)
        assert "SELECT host_id, payload" in source, (
            f"{label} still selects only the payload, so it depends on the id "
            "being duplicated inside the JSON"
        )
