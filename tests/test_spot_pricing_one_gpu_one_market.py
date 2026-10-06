"""One GPU is one spot market, whatever the driver calls it.

Production's `/spot-prices` listed `NVIDIA GeForce RTX 2060` at $0.12/hr beside
`RTX 2060` at $0.03/hr. The first is what the worker's driver reports; the
second is the catalogue's name. Spot pricing keyed supply, demand, floors and
history on the raw string, so:

* the market for one card was split in two, each side surging on half the
  picture;
* the raw name missed its `gpu_pricing` row, fell back to the host's own cost
  per hour, and quoted four times the catalogue spot rate;
* `lock_spot_rate_for_job` with a raw `host_gpu_model` billed the job at it.

`host_metadata.normalize_gpu_model` already existed and host registration used
it. Spot pricing did not.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import scheduler  # noqa: E402
import spot_pricing as sp  # noqa: E402

RAW = "NVIDIA GeForce RTX 2060"
CANONICAL = "RTX 2060"


def _needs_pg():
    if not sp._pg_available():
        pytest.skip("spot_price_history needs Postgres")


def _history(rows: list[tuple[str, int, float]]):
    from db import pg_connection

    with pg_connection() as conn:
        for model, cents, at in rows:
            conn.execute(
                """INSERT INTO spot_price_history
                   (gpu_model, clearing_price_cents, supply_count, demand_count, recorded_at)
                   VALUES (%s, %s, 0, 0, %s)""",
                (model, cents, at),
            )


def _forget(*ats: float):
    from db import pg_connection

    with pg_connection() as conn:
        for at in ats:
            conn.execute("DELETE FROM spot_price_history WHERE recorded_at = %s", (at,))


@pytest.mark.parametrize("raw_is_newer", [False, True], ids=["catalogue-newer", "raw-newer"])
def test_history_lists_the_card_once_at_its_newest_price(raw_is_newer):
    _needs_pg()
    base = time.time() + 54_321.0
    old, new = base, base + 60
    raw_at, canon_at = (new, old) if raw_is_newer else (old, new)
    _history([(RAW, 12, raw_at), (CANONICAL, 3, canon_at)])
    sp._latest_quotes.clear()
    try:
        prices = sp.get_current_spot_prices()
    finally:
        _forget(old, new)
    assert RAW not in prices, f"the driver's name is still its own market: {sorted(prices)}"
    assert prices[CANONICAL] == (0.12 if raw_is_newer else 0.03), (
        "where two names collapse into one, the newest row is the price"
    )


def test_a_raw_name_quotes_the_catalogue_rate():
    raw = sp.compute_live_spot_quote(RAW, supply=1, demand=0)
    canonical = sp.compute_live_spot_quote(CANONICAL, supply=1, demand=0)
    assert raw.gpu_model == CANONICAL
    assert raw.on_demand_cad == canonical.on_demand_cad
    assert raw.rate_cad == canonical.rate_cad


def test_a_job_on_a_raw_named_host_is_locked_at_the_catalogue_rate():
    canonical = sp.compute_live_spot_quote(CANONICAL).rate_cad
    job: dict = {"job_id": "j-raw", "pricing_mode": "spot"}
    assert sp.lock_spot_rate_for_job(job, host_gpu_model=RAW) == canonical


def test_supply_on_a_raw_named_host_counts_toward_the_card():
    host_id = f"raw-{uuid.uuid4().hex[:8]}"
    before = sp.get_supply_demand(CANONICAL)[0]
    scheduler.register_host(host_id, "127.0.0.1", RAW, 6, 6, cost_per_hour=0.5)
    try:
        hosts = sp._host_supply_by_gpu()
        assert RAW not in hosts, f"supply still keyed by the driver's name: {sorted(hosts)}"
        assert sp.get_supply_demand(CANONICAL)[0] >= before + 1
        assert sp.get_supply_demand(RAW) == sp.get_supply_demand(CANONICAL)
    finally:
        scheduler.remove_host(host_id)


def test_supply_keyed_raw_in_storage_is_still_one_market(monkeypatch):
    """Whatever registration does, a raw row already stored must merge."""
    monkeypatch.setattr(
        scheduler,
        "load_hosts",
        lambda active_only=True: [
            {"host_id": "a", "gpu_model": RAW, "num_gpus": 1, "cost_per_hour": 0.5},
            {"host_id": "b", "gpu_model": CANONICAL, "num_gpus": 2, "cost_per_hour": 0.3},
        ],
    )
    assert sp._host_supply_by_gpu() == {CANONICAL: 3}
    assert sp._host_min_cost_by_gpu() == {CANONICAL: 0.3}
