"""Invoice reads must report real charges without creating billing records."""

import csv
import io
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from billing import BillingEngine
import routes.billing as routes

pytestmark = pytest.mark.needs_db

START = datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp()
END = datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def account(monkeypatch):
    customer = f"invoice-audit-{uuid4().hex}"
    engine = BillingEngine()
    monkeypatch.setattr(routes, "get_billing_engine", lambda: engine)
    # Ownership and credential scopes have their own real-auth endpoint tests.
    # Here a small app isolates invoice rendering and persistence contracts.
    monkeypatch.setattr(routes, "_require_customer_access", lambda *args: None)
    monkeypatch.setattr(routes, "time", SimpleNamespace(time=lambda: END + 86400))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            yield customer, engine, client
        finally:
            with engine._conn() as conn:
                conn.execute("DELETE FROM invoices WHERE customer_id = %s", (customer,))
                conn.execute("DELETE FROM usage_meters WHERE owner = %s", (customer,))
                conn.execute("DELETE FROM billing_cycles WHERE customer_id = %s", (customer,))


def _charge(account, kind="gpu", *, start=START + 100, end=START + 9100, micros=12_500_000):
    customer, engine, _ = account
    resource = uuid4().hex
    with engine._conn() as conn:
        if kind == "gpu":
            conn.execute(
                """INSERT INTO usage_meters
                   (meter_id, job_id, owner, started_at, completed_at, duration_sec,
                    gpu_model, base_rate_per_hour, total_cost_micros)
                   VALUES (%s, %s, %s, %s, %s, %s, 'RTX 4090', 5, %s)""",
                (resource, resource, customer, start, end, end - start, micros),
            )
        else:
            conn.execute(
                """INSERT INTO billing_cycles
                   (cycle_id, job_id, customer_id, resource_type, period_start, period_end,
                    duration_seconds, rate_per_hour, gpu_model, amount_micros, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 1, 'storage', %s, 'charged', %s)""",
                (resource, resource, customer, kind, start, end, end - start, micros, end),
            )


def _read(account, suffix="", **params):
    customer, _, client = account
    return client.get(
        f"/api/billing/invoice/{customer}{suffix}",
        params={"period_start": START, "period_end": END, **params},
    )


@pytest.mark.parametrize("kind", ["preview", "list", "csv", "txt"])
def test_repeated_invoice_reads_do_not_insert_rows(account, kind):
    customer, engine, client = account
    _charge(account)
    for _ in range(2):
        response = (
            client.get(f"/api/billing/invoices/{customer}", params={"limit": 3})
            if kind == "list"
            else _read(account, "/download" if kind in {"csv", "txt"} else "", format=kind)
        )
        assert response.status_code == 200, response.text
    with engine._conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM invoices WHERE customer_id = %s", (customer,)
        ).fetchone()["n"]
    assert count == 0


def test_explicit_invoice_generation_still_persists(account):
    customer, engine, _ = account
    _charge(account)
    invoice = engine.generate_invoice(customer, "", START, END)
    with engine._conn() as conn:
        row = conn.execute(
            "SELECT total_micros FROM invoices WHERE invoice_id = %s", (invoice.invoice_id,)
        ).fetchone()
    assert row["total_micros"] == 14_130_000


def test_csv_uses_canonical_amounts_quantities_units_and_identity(account):
    _charge(account)
    _charge(account, "volume", end=START + 7300, micros=2_000_000)
    preview = _read(account).json()["invoice"]
    response = _read(account, "/download", format="csv")
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0][1] == preview["invoice_id"]
    header = rows[2]
    items = [dict(zip(header, row)) for row in rows[3:5]]
    assert items[0]["Quantity"] == "2.5000"
    assert items[0]["Unit"] == "GPU-hours"
    assert items[0]["Rate (CAD/unit)"] == "5.000000"
    assert items[0]["Amount (CAD)"] == "12.500000"
    assert items[1]["Quantity"] == "2.0000"
    assert items[1]["Unit"] == "volume-hours"
    assert items[1]["Amount (CAD)"] == "2.000000"
    assert {row[0]: row[-1] for row in rows[-3:]} == {
        "Subtotal": "14.50",
        "Tax": "1.89",
        "Total (CAD)": "16.39",
    }


def test_text_receipt_uses_the_same_nonzero_totals_and_id(account):
    _charge(account)
    preview = _read(account).json()["invoice"]
    response = _read(account, "/download", format="txt")
    assert response.status_code == 200
    assert preview["invoice_id"] in response.text
    assert "GPU-hours" in response.text
    assert "$12.50" in response.text
    assert "$1.63" in response.text
    assert "$14.13" in response.text


@pytest.mark.parametrize("kind", ["gpu", "serverless_gpu", "serverless_gpu_cold_start", "volume"])
@pytest.mark.parametrize("at_boundary", [False, True])
def test_charge_is_in_exactly_one_period_even_when_it_spans_months(account, kind, at_boundary):
    end = START if at_boundary else START + 60
    _charge(account, kind, start=START - 3600, end=end)
    previous = _read(account, period_start=START - 31 * 86400, period_end=START).json()["invoice"]
    current = _read(account).json()["invoice"]
    assert previous["subtotal_cad"] == 0
    assert current["subtotal_cad"] == 12.50


def test_monthly_history_uses_calendar_months_and_stable_preview_ids(account):
    customer, _, client = account
    _charge(account)
    responses = [client.get(f"/api/billing/invoices/{customer}?limit=2").json() for _ in range(2)]
    assert responses[0] == responses[1]
    assert responses[0]["count"] == 1
    invoice = responses[0]["invoices"][0]
    assert (invoice["period_start"], invoice["period_end"]) == (START, END)
    assert invoice["invoice_id"] == _read(account).json()["invoice"]["invoice_id"]
    assert invoice["status"] == "draft"
    assert invoice["subtotal_cad"] == 12.50


def test_failed_monthly_query_is_not_reported_as_empty_success(account, monkeypatch):
    customer, engine, client = account

    def unavailable(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(engine, "generate_invoice", unavailable)
    response = client.get(f"/api/billing/invoices/{customer}?limit=1")
    assert response.status_code >= 500


@pytest.mark.parametrize(
    "params",
    [
        {"period_start": END, "period_end": START},
        {"period_end": -1},
        {"period_end": "nan"},
        {"period_start": "inf"},
        {"tax_rate": -1},
        {"tax_rate": "nan"},
        {"tax_rate": 2},
    ],
)
@pytest.mark.parametrize("suffix", ["", "/download"])
def test_invalid_period_or_tax_is_rejected(account, params, suffix):
    assert _read(account, suffix, **params).status_code == 422


def test_unsupported_download_format_is_rejected(account):
    assert _read(account, "/download", format="pdf").status_code == 422


def test_subcent_lines_are_not_rounded_away_before_adding(account):
    # ROUND(sum, 4) per group used to lose both of these 49-micro charges,
    # changing 0.004998 into 0.0049 and rounding the invoice down to zero.
    _charge(account, micros=4900)
    _charge(account, "serverless_gpu", micros=49)
    _charge(account, "volume", micros=49)
    invoice = _read(account, tax_rate=0).json()["invoice"]
    assert sum(item["subtotal_cad"] for item in invoice["line_items"]) == pytest.approx(0.004998)


def test_rounding_totals_uses_currency_half_up(account):
    _charge(account, micros=1_005_000)
    invoice = _read(account, tax_rate=0).json()["invoice"]
    assert invoice["subtotal_cad"] == 1.01
    assert invoice["total_cad"] == 1.01
