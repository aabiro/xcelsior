"""Public workload pages come only from real, shared, reconciled runs.

A page ("vLLM on an RTX 3060: what real runs cost") needs three shared runs
for one quick-start template on one GPU model, and live supply of that GPU.
It carries counts, medians and ranges, never a person, account, job, host or
image. Sharing is the run owner's choice, on and off.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import receipt_pages  # noqa: E402
from receipt_pages import (  # noqa: E402
    MIN_RUNS,
    PAGE_FIELDS,
    build_pages,
    page_slug,
    share_blocker,
)
from tests.test_run_receipts import _client, _register, billed_run  # noqa: E402,F401


def _run(i: int, **over) -> dict:
    run = {
        "template_id": "vllm",
        "gpu_model": "RTX 3060",
        "source": "customer",
        "billed_seconds": 600.0 * (i + 1),
        "total_micros": 20_000 * (i + 1),
        "rate_micros_per_hour": 120_000,
        "pricing_mode": "on_demand",
        "ended_at": 1_763_800_000.0 + i,
        "utilization": 80.0 + i,
    }
    run.update(over)
    return run


def test_three_shared_runs_on_a_gpu_with_supply_make_a_page():
    [page] = build_pages([_run(0), _run(1), _run(2, source="operator")], {"RTX 3060"})
    assert page["slug"] == "vllm-on-rtx-3060" == page_slug("vllm", "RTX 3060")
    assert page["template_label"] == "vLLM" and page["gpu_model"] == "RTX 3060"
    assert (page["runs"], page["customer_runs"], page["operator_runs"]) == (3, 2, 1)
    assert page["median_billed_seconds"] == 1200.0
    assert (page["min_billed_seconds"], page["max_billed_seconds"]) == (600.0, 1800.0)
    assert page["median_total_micros"] == 40_000
    assert (page["min_total_micros"], page["max_total_micros"]) == (20_000, 60_000)
    assert page["median_rate_micros_per_hour"] == 120_000
    assert page["pricing_modes"] == {"on_demand": 3}
    assert page["median_gpu_utilization"] == 81.0 and page["utilization_runs"] == 3
    assert page["latest_month"] == "2025-11"


def test_no_page_below_three_runs_or_without_supply_or_template():
    two = [_run(0), _run(1)]
    assert build_pages(two, {"RTX 3060"}) == []
    three = [_run(0), _run(1), _run(2)]
    assert build_pages(three, set()) == [], "a page never advertises a GPU nobody can rent"
    assert build_pages([_run(i, template_id="retired") for i in range(3)], {"RTX 3060"}) == []
    mixed = [_run(0), _run(1), _run(2, gpu_model="RTX 2060")]
    assert build_pages(mixed, {"RTX 3060", "RTX 2060"}) == []


def test_utilization_shows_only_when_enough_runs_reported_it():
    runs = [_run(0), _run(1, utilization=None), _run(2, utilization=None)]
    [page] = build_pages(runs, {"RTX 3060"})
    assert page["median_gpu_utilization"] is None and page["utilization_runs"] == 1


def test_a_page_carries_only_its_allowlisted_fields():
    [page] = build_pages([_run(i) for i in range(MIN_RUNS)], {"RTX 3060"})
    assert set(page) == set(PAGE_FIELDS)
    for forbidden in ("job_id", "host_id", "owner", "email", "image", "ended_at", "started_at"):
        assert forbidden not in page


_RECEIPT = {
    "final": True,
    "status": "completed",
    "launch_template_id": "pytorch",
    "gpu_model": "RTX 3060",
    "billed_seconds": 120.0,
    "reconciled": True,
}


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({}, None),
        ({"final": False}, "not_finished"),
        ({"status": "failed"}, "ended_abnormally"),
        ({"status": "preempted"}, "ended_abnormally"),
        ({"launch_template_id": None}, "no_template"),
        ({"launch_template_id": "not-a-template"}, "no_template"),
        ({"gpu_model": ""}, "no_gpu"),
        ({"billed_seconds": 59.0}, "too_short"),
        ({"reconciled": False}, "not_reconciled"),
    ],
)
def test_which_runs_can_count(change, blocker):
    assert share_blocker({**_RECEIPT, **change}) == blocker


# ── the owner's switch, against the real ledger ──────────────────────


@pytest.fixture
def template_run(billed_run):
    """The billed run from test_run_receipts, started from the PyTorch template."""
    from psycopg.types.json import Jsonb

    from db import _get_pg_pool

    client, headers, customer_id, native = billed_run
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload = payload || %s WHERE job_id = %s",
            (Jsonb({"launch_template_id": "pytorch", "gpu_model": "RTX 3060"}), native["job_id"]),
        )
    try:
        yield client, headers, customer_id, native
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM receipt_shares WHERE job_id = %s", (native["job_id"],))


@pytest.mark.needs_db
def test_the_owner_shares_and_unshares_a_finished_run(template_run):
    from db import _get_pg_pool

    client, headers, customer_id, native = template_run
    path = f"/api/instances/{native['job_id']}/receipt/share"

    state = client.get(path, headers=headers)
    assert state.status_code == 200, state.text
    body = state.json()
    assert body["shared"] is False and body["eligible"] is True, body
    assert body["template_label"] == "PyTorch"

    on = client.put(path, json={"shared": True}, headers=headers)
    assert on.status_code == 200, on.text
    assert on.json()["shared"] is True
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT owner_id, source, revoked_at FROM receipt_shares WHERE job_id = %s",
            (native["job_id"],),
        ).fetchone()
    assert row == (customer_id, "customer", None)

    def mine_now():
        return [
            r
            for r in receipt_pages._shared_runs(_get_pg_pool())
            if r["template_id"] == "pytorch" and r["gpu_model"] == "RTX 3060"
        ]

    [mine] = mine_now()
    assert mine["source"] == "customer" and mine["billed_seconds"] >= 89

    # A page built from it (MIN_RUNS lowered for one run) names nobody.
    pages = build_pages([mine] * MIN_RUNS, {mine["gpu_model"]})
    text = json.dumps(pages)
    for leaked in (native["job_id"], native["host_id"], customer_id, "python:3.11"):
        assert leaked not in text

    off = client.put(path, json={"shared": False}, headers=headers)
    assert off.status_code == 200 and off.json()["shared"] is False
    assert mine_now() == []


@pytest.mark.needs_db
def test_only_the_owner_can_see_or_change_the_share(template_run):
    client, _, _, native = template_run
    intruder, _ = _register(client, "share-intruder")
    path = f"/api/instances/{native['job_id']}/receipt/share"
    assert client.get(path, headers=intruder).status_code == 404
    assert client.put(path, json={"shared": True}, headers=intruder).status_code == 404


@pytest.mark.needs_db
def test_a_run_without_a_template_cannot_be_shared(billed_run):
    client, headers, _, native = billed_run
    path = f"/api/instances/{native['job_id']}/receipt/share"
    body = client.get(path, headers=headers).json()
    assert body["eligible"] is False and body["blocker"] == "no_template"
    refused = client.put(path, json={"shared": True}, headers=headers)
    assert refused.status_code == 409


def test_the_public_routes_need_no_sign_in_and_404_an_unknown_slug(monkeypatch):
    page = build_pages([_run(i) for i in range(MIN_RUNS)], {"RTX 3060"})
    monkeypatch.setattr(receipt_pages, "public_pages", lambda **_: page)
    client = _client()
    listing = client.get("/api/receipt-pages")
    assert listing.status_code == 200, listing.text
    assert [p["slug"] for p in listing.json()["pages"]] == ["vllm-on-rtx-3060"]
    one = client.get("/api/receipt-pages/vllm-on-rtx-3060")
    assert one.status_code == 200 and one.json()["page"]["runs"] == MIN_RUNS
    assert client.get("/api/receipt-pages/pytorch-on-h100").status_code == 404


def test_the_frontend_page_type_carries_exactly_the_page_fields():
    """A field the API sends but the type drops would never render (parity guard)."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "api/frontend/src/lib/api.ts").read_text()
    body = re.search(r"export interface ReceiptPage \{(.*?)\n\}", source, re.S)
    assert body, "ReceiptPage is missing from api.ts"
    assert set(re.findall(r"^\s+(\w+)\??:", body.group(1), re.M)) == set(PAGE_FIELDS)
