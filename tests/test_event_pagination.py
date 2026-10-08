"""Events pagination over the real store, including equal timestamps and new arrivals."""
import base64
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from api import app
from events import get_event_store
import routes.events as routes


@pytest.fixture
def history(monkeypatch):
    monkeypatch.setattr(routes, "_require_admin", lambda request: {"is_admin": True})
    kind = f"pagination-{uuid.uuid4().hex}"
    store = get_event_store()
    base = time.time() - 100
    times = [base, base + 1, base + 1, base + 1, base + 2, base + 3, base + 4]
    ids = [f"{kind}-{i}" for i in range(len(times))]
    with store._conn() as conn:
        for i, (event_id, at) in enumerate(zip(ids, times)):
            conn.execute(
                "INSERT INTO events (event_id,event_type,entity_type,entity_id,timestamp,actor,data,metadata) "
                "VALUES (%s,%s,'test',%s,%s,'test',%s,%s)",
                (event_id, kind, kind, at, Jsonb({"severity": "error"} if i == 2 else {}),
                 Jsonb({"severity": "warning"} if i == 4 else {})),
            )
    yield kind, list(reversed(ids)), store
    with store._conn() as conn:
        conn.execute("DELETE FROM events WHERE entity_id = %s", (kind,))


def page(kind, **params):
    with TestClient(app) as client:
        response = client.get("/api/events", params={"event_type": kind, "limit": 3, **params})
    assert response.status_code == 200, response.text
    return response.json()


def test_pages_visit_every_row_once_newest_first(history):
    kind, expected, _ = history
    seen, cursor = [], None
    for _ in range(4):
        body = page(kind, **({"before": cursor} if cursor else {}))
        assert body["total"] == len(expected)
        seen += [row["event_id"] for row in body["events"]]
        cursor = body["next_cursor"]
        if not cursor:
            break
    assert seen == expected
    assert cursor is None


def test_new_arrivals_do_not_shift_older_pages(history):
    kind, expected, store = history
    first = page(kind)
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO events (event_id,event_type,entity_type,entity_id,timestamp,actor,data,metadata) "
            "VALUES (%s,%s,'test',%s,%s,'test','{}','{}')",
            (kind + "-new", kind, kind, time.time()),
        )
    second = page(kind, before=first["next_cursor"])
    assert [row["event_id"] for row in second["events"]] == expected[3:6]


@pytest.mark.parametrize("severity, index", [("error", 2), ("warning", 4)])
def test_severity_filters_apply_before_paging(history, severity, index):
    kind, _, _ = history
    body = page(kind, severity=severity)
    assert body["total"] == 1
    assert body["events"][0]["event_id"] == f"{kind}-{index}"
    assert body["events"][0]["severity"] == severity
    assert body["next_cursor"] is None


@pytest.mark.parametrize("value", ["not-base64!", "e30", "W10", *[
    base64.urlsafe_b64encode(json.dumps(item).encode()).decode()
    for item in ([float("nan"), "id"], [True, "id"], [1, ""], [1, 2], [1, 2, 3])
]])
def test_invalid_cursors_are_client_errors(history, value):
    kind, _, _ = history
    with TestClient(app) as client:
        response = client.get("/api/events", params={"event_type": kind, "before": value})
    assert response.status_code == 400, response.text


def test_events_still_require_platform_admin(monkeypatch):
    import routes._deps as deps
    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    with TestClient(app) as client:
        response = client.get("/api/events")
    assert response.status_code in {401, 403}
