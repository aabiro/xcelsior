"""`/spot-prices` listed a GPU with no name.

Production's `spot_price_history` holds a row whose `gpu_model` is `''`, left
from before `_catalog_gpu_models` filtered blanks. `get_current_spot_prices`
read the latest row per model straight from that table, so both keys of
`/spot-prices` carried it: `prices` had a `""` key and `spot_prices` had a row
with an empty `gpu_model`. The analytics dashboard drew it as a bar with no
label.

The writer only ever receives catalog models, so the leak was the reader. Both
now apply the predicate the catalog uses.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient  # noqa: E402

import spot_pricing as sp  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app)

BLANKS = ("", "   ", "unknown")


@pytest.fixture
def history_with_blank_models():
    if not sp._pg_available():
        pytest.skip("spot_price_history needs Postgres")
    from db import pg_connection

    now = time.time()
    with pg_connection() as conn:
        for model in BLANKS:
            conn.execute(
                """INSERT INTO spot_price_history
                   (gpu_model, clearing_price_cents, supply_count, demand_count, recorded_at)
                   VALUES (%s, 5, 0, 0, %s)""",
                (model, now),
            )
        conn.execute(
            """INSERT INTO spot_price_history
               (gpu_model, clearing_price_cents, supply_count, demand_count, recorded_at)
               VALUES ('RTX 4090', 42, 1, 1, %s)""",
            (now,),
        )
    sp._latest_quotes.clear()
    yield
    with pg_connection() as conn:
        conn.execute(
            "DELETE FROM spot_price_history WHERE recorded_at = %s",
            (now,),
        )
    sp._latest_quotes.clear()


def test_the_lookup_has_no_nameless_gpu(history_with_blank_models):
    prices = client.get("/spot-prices").json()["prices"]
    assert prices.get("RTX 4090") == 0.42, "the real row must still come through"
    assert not set(BLANKS) & set(prices), f"blank models in `prices`: {sorted(set(BLANKS) & set(prices))!r}"


def test_the_rows_have_no_nameless_gpu(history_with_blank_models):
    rows = client.get("/spot-prices").json()["spot_prices"]
    assert any(r["gpu_model"] == "RTX 4090" for r in rows)
    nameless = [r["gpu_model"] for r in rows if r["gpu_model"] in BLANKS]
    assert not nameless, f"`spot_prices` rows with no usable model: {nameless!r}"


@pytest.mark.parametrize("model", BLANKS)
def test_a_nameless_quote_is_not_recorded(model, monkeypatch):
    if not sp._pg_available():
        pytest.skip("spot_price_history needs Postgres")
    from db import pg_connection

    as_of = time.time() + 12_345.678
    quote = sp.SpotQuote(
        gpu_model=model,
        rate_cad=0.05,
        on_demand_cad=0.1,
        savings_pct=50,
        supply=0,
        demand=0,
        spot_cents=5,
        provider_floor_cents=0,
        as_of=as_of,
    )
    try:
        sp.record_spot_history(quote)
        with pg_connection() as conn:
            count = conn.execute(
                "SELECT count(*) FROM spot_price_history WHERE recorded_at = %s", (as_of,)
            ).fetchone()[0]
        assert count == 0, f"a quote for {model!r} was written to history"
    finally:
        with pg_connection() as conn:
            conn.execute("DELETE FROM spot_price_history WHERE recorded_at = %s", (as_of,))
