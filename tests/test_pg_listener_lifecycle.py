"""FastAPI lifespan must not leak PostgreSQL LISTEN threads."""

import threading
import time

from fastapi.testclient import TestClient

from api import app


def _listener_ids() -> set[int | None]:
    return {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "pg-listen"
    }


def test_api_lifespan_stops_its_pg_listener():
    before = _listener_ids()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200

    deadline = time.time() + 3
    while _listener_ids() - before and time.time() < deadline:
        time.sleep(0.05)
    assert _listener_ids() <= before
