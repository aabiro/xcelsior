"""A query that failed must not read as a query that found nothing.

Three handlers answered a database failure with a cheerful, empty success:

* `/api/sla/hosts` returned `{"ok": true, "hosts": [], "count": 0}` — which is
  exactly what a healthy engine over an empty fleet returns. It caught
  `Exception as e` and never used `e`, so the failure was discarded with no log
  line anywhere.
* `/api/billing/reservations` returned `ok: true` with an empty list and a
  zeroed summary, telling a customer they hold no reserved-instance
  commitments. "We could not load them" and "you have none" are different
  statements, and the second is the more alarming one to be wrong about.
* `/api/analytics/usage` returned `str(e)` — the raw database exception — to
  the caller, in a 200. Non-admins are scoped to their own data but still reach
  this route, so a psycopg message naming a column and quoting the offending
  value went to them. It is the same reason the `psycopg.DataError` handler in
  `api.py` logs its message rather than sending it.

The response shapes are unchanged, because the dashboard renders them and a
sudden 503 would break a page that currently degrades. What changes is that the
failure is now *visible*: logged, and flagged with `degraded` so a caller can
tell the difference.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth():
    email = f"degraded-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Degraded Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


# ── The raw-exception leak ────────────────────────────────────────────────


def test_analytics_never_returns_the_database_message(auth) -> None:
    """The message quotes the offending value and names the column.

    Asserted structurally as well as on the wire: forcing the failure needs a
    patch inside the handler's own `try`, and a patch that lands outside it
    proves nothing (the first version of this test raised past the handler and
    never exercised the branch).
    """
    import ast
    import inspect
    import textwrap

    from routes.billing import api_usage_analytics

    tree = ast.parse(textwrap.dedent(inspect.getsource(api_usage_analytics)))
    returned_exception = []
    for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
        for node in ast.walk(handler):
            # `str(e)` anywhere in a returned dict is the leak
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "str":
                returned_exception.append(handler.lineno)
    assert not returned_exception, (
        "the handler stringifies the exception into its response again "
        f"(line {returned_exception}); psycopg messages quote the offending "
        "value and name the column"
    )

    r = client.get("/api/analytics/usage", headers=auth)
    assert r.status_code < 500, r.text[:200]


def test_analytics_still_answers_in_the_shape_the_dashboard_expects(auth) -> None:
    r = client.get("/api/analytics/usage", headers=auth)
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    for key in ("ok", "analytics", "summary"):
        assert key in body, f"{key} missing: {body}"


# ── Failures that looked like emptiness ───────────────────────────────────


def test_sla_summary_says_degraded_rather_than_empty(monkeypatch, auth) -> None:
    import routes.sla as sla_mod

    def _explode(*_a, **_k):
        raise RuntimeError("sla engine down")

    monkeypatch.setattr(sla_mod, "get_sla_engine", _explode)

    r = client.get("/api/sla/hosts-summary", headers=auth)
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body.get("degraded") is True, (
        "an SLA engine outage still reports a clean empty fleet; the dashboard "
        f"cannot tell it apart from 'no host has SLA data': {body}"
    )
    assert body["hosts"] == []


def test_sla_summary_is_not_marked_degraded_when_healthy(auth) -> None:
    """Otherwise the flag means nothing."""
    r = client.get("/api/sla/hosts-summary", headers=auth)
    assert r.status_code == 200, r.text[:200]
    assert r.json().get("degraded") is not True


def test_reservations_say_degraded_rather_than_none(monkeypatch, auth) -> None:
    import routes.billing as billing_mod

    class _Engine:
        def list_reservations(self, _customer_id):
            raise RuntimeError("reservations table unavailable")

    monkeypatch.setattr(billing_mod, "get_billing_engine", lambda: _Engine())

    r = client.get(
        "/api/pricing/reservations", params={"customer_id": "cust-probe"}, headers=auth
    )
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body.get("degraded") is True, (
        "a failed lookup still tells the customer they hold no reserved "
        f"instances: {body}"
    )


# ── The discarded exception ───────────────────────────────────────────────


def test_no_handler_in_these_modules_discards_a_caught_exception() -> None:
    """`except Exception as e:` where `e` is never used means no log, no trace.

    That is how the SLA outage became invisible: the name was bound and then
    dropped, so nothing recorded that anything had gone wrong.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for rel in ("routes/sla.py", "routes/billing.py", "routes/admin.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
            if not handler.name:
                continue
            used = any(
                isinstance(n, ast.Name) and n.id == handler.name
                for n in ast.walk(handler)
            )
            # `log.exception(...)` records the traceback without referencing the
            # bound name, so an unused binding beside one is style, not silence.
            # What matters is whether the failure leaves any trace at all.
            logs = any(
                isinstance(n, ast.Call)
                and getattr(getattr(n.func, "value", None), "id", None) == "log"
                for n in ast.walk(handler)
            )
            reraises = any(isinstance(n, ast.Raise) for n in ast.walk(handler))
            if not used and not logs and not reraises:
                offenders.append(
                    f"{rel}:{handler.lineno}: caught as `{handler.name}`, never used, "
                    "nothing logged, nothing re-raised"
                )

    assert not offenders, (
        "these bind the exception and then drop it, so the failure leaves no "
        "trace at all:\n  " + "\n  ".join(offenders)
    )


# ── The one that was giving a legal answer ────────────────────────────────


def test_gst_threshold_refuses_to_guess_when_revenue_cannot_be_read(monkeypatch) -> None:
    """The sharpest case: a failed query produced a confident tax answer.

    `total_rev = 0.0` on exception, then the calculation ran anyway and
    returned `must_register: false` with "Below threshold ($0.00 / $30,000).
    Registration not yet required" — a specific, authoritative-looking
    statement about a statutory obligation, derived from a number that came
    from a failure. The exception was bound and discarded, so nothing was
    logged either.
    """
    import routes.compliance as compliance_mod

    class _Engine:
        def _conn(self):
            raise RuntimeError("usage_meters unavailable")

    monkeypatch.setattr(compliance_mod, "get_billing_engine", lambda: _Engine())

    r = client.get("/api/billing/gst-threshold", headers=_admin_headers())
    assert r.status_code < 500, r.text[:300]
    body = r.json()

    assert body.get("determinable") is False, (
        f"a failed revenue query still produced a registration verdict: {body}"
    )
    assert "must_register" not in body, (
        "the verdict is still present; a caller reading it would act on a "
        f"number that came from an exception: {body}"
    )
    assert body.get("total_revenue_cad") != 0.0, "zero is still being reported as revenue"
    assert "cannot be determined" in body.get("message", "").lower()


def test_gst_threshold_still_answers_when_revenue_can_be_read() -> None:
    """Otherwise the guard above could be satisfied by never answering at all."""
    r = client.get("/api/billing/gst-threshold", headers=_admin_headers())
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body.get("determinable") is True, body
    assert "must_register" in body
    assert isinstance(body.get("total_revenue_cad"), (int, float))
