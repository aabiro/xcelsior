"""A sweep preview must store a trace id the action_plans table accepts.

ck_action_plans_trace_id allows NULL or 32 hex digits. The sweep route passed
"", so every create_image_sweep preview failed the insert and answered 500.
The live rehearsal of every MCP tool found it on 2026-11-24.
"""

from __future__ import annotations

import inspect
import re

from starlette.requests import Request

from routes import instances

_CHECK = re.compile(r"^[0-9a-f]{32}$")


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw})


def test_no_traceparent_stores_null_not_an_empty_string():
    assert instances._sweep_trace_id(_request({})) is None


def test_a_traceparent_stores_its_trace_id():
    trace = "4bf92f3577b34da6a3ce929d0e0e4736"
    got = instances._sweep_trace_id(_request({"traceparent": f"00-{trace}-00f067aa0ba902b7-01"}))
    assert got == trace and _CHECK.match(got)


def test_a_malformed_traceparent_stores_null():
    assert instances._sweep_trace_id(_request({"traceparent": "00-not-hex-01"})) is None


def test_the_route_no_longer_passes_an_empty_trace_id():
    body = inspect.getsource(instances.api_create_image_sweep)
    assert 'trace_id=""' not in body
    assert "trace_id=_sweep_trace_id(request)" in body
