"""The launch picker must receive the same storage rate used by metering."""

from types import SimpleNamespace

import pytest
from starlette.requests import Request


def test_available_volumes_include_canonical_storage_prices(monkeypatch):
    import routes.volumes as routes

    user = {"user_id": "renter", "scopes": ["volumes:read"]}
    volumes = [
        {"volume_id": "available", "name": "Models", "size_gb": 730, "status": "available"},
        {"volume_id": "attached", "size_gb": 50, "status": "attached"},
    ]
    monkeypatch.setattr(routes, "_get_current_user", lambda request: user)
    monkeypatch.setattr(routes, "_volume_owner_ids_readable", lambda caller: {"renter"})
    monkeypatch.setattr(
        routes,
        "get_volume_engine",
        lambda: SimpleNamespace(
            list_volumes_for_owner_ids=lambda owners: volumes,
        ),
    )

    result = routes.api_volumes_available(Request({"type": "http"}))

    assert len(result["volumes"]) == 1
    volume = result["volumes"][0]
    assert volume["volume_id"] == "available"
    assert volume["price_per_gb_month_cad"] == routes.VOLUME_PRICE_PER_GB_MONTH_CAD
    assert volume["monthly_cost_cad"] == pytest.approx(730 * routes.VOLUME_PRICE_PER_GB_MONTH_CAD)
