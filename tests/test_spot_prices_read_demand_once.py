"""`/spot-prices` reads fleet supply and job demand once, not once per GPU model.

Each read loads every host, offer and job. With no cached quotes (a fresh API
process), quoting the catalogue re-read them per model: on the test stack's
database, 55 models took about ten seconds each and the request never answered.
"""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import spot_pricing


def test_uncached_quotes_share_one_supply_and_demand_read(monkeypatch):
    reads = {"hosts": 0, "offers": 0, "jobs": 0}

    def counted(name, value):
        def read():
            reads[name] += 1
            return value

        return read

    models = {f"RTX {n}": 0.5 + n / 100 for n in range(30, 40)}
    monkeypatch.setattr(spot_pricing, "_latest_quotes", {})
    monkeypatch.setattr(spot_pricing, "get_current_spot_prices", lambda: dict(models))
    monkeypatch.setattr(spot_pricing, "_host_supply_by_gpu", counted("hosts", {"RTX 30": 2}))
    monkeypatch.setattr(spot_pricing, "_marketplace_supply_by_gpu", counted("offers", {}))
    monkeypatch.setattr(spot_pricing, "_job_demand_by_gpu", counted("jobs", {"RTX 30": 3}))

    quotes = spot_pricing.get_current_spot_prices_list()

    assert len(quotes) == len(models)
    assert reads == {"hosts": 1, "offers": 1, "jobs": 1}


def test_the_snapshot_answers_canonical_names_like_the_single_read(monkeypatch):
    monkeypatch.setattr(spot_pricing, "_host_supply_by_gpu", lambda: {"RTX 2060": 2})
    monkeypatch.setattr(spot_pricing, "_marketplace_supply_by_gpu", lambda: {"RTX 2060": 1})
    monkeypatch.setattr(spot_pricing, "_job_demand_by_gpu", lambda: {"RTX 2060": 4})

    snapshot = spot_pricing.supply_demand_snapshot()

    assert snapshot("NVIDIA GeForce RTX 2060") == (3, 4)
    assert spot_pricing.get_supply_demand("RTX 2060") == (3, 4)
    assert snapshot("H100") == (0, 0)
