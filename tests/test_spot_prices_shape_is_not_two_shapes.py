"""`/spot-prices` sends two shapes, and a truthy array hid one of them.

The route returns **both** `prices` (a lookup, `{"A10": 0.42}`) and
`spot_prices` (a **list** of rows carrying supply, demand and the on-demand
comparison). They are not interchangeable.

`frontend/src/lib/api.ts` had them inverted — `spot_prices: Record<string,
number>` with `prices` as an optional fallback — and the launch modal read:

    setSpotPrices(res.spot_prices || res.prices || {})

A non-empty array is **truthy**, so `res.prices` — the correctly-shaped
fallback someone wrote for exactly this case — could never be reached. The
component then did `spotPrices[gpuModel]` against an array, which is `undefined`
for every GPU model, so the per-listing spot price never rendered and every
price fell through to `selectedPricing?.spot_cad ?? cheapestAvailableRate`.

Nothing threw. `||` and `??` chains degrade silently by design, which is what
makes an inverted type this expensive: the wrong branch is indistinguishable
from "no data yet".

## What this pins

That both keys exist and keep their shapes. The frontend now prefers `prices`
and derives the lookup from `spot_prices` only as a fallback, so either key
changing shape breaks a real screen.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _body():
    response = client.get("/spot-prices")
    assert response.status_code == 200, response.text
    return response.json()


def test_both_keys_are_present_and_are_different_shapes():
    body = _body()
    assert isinstance(body["prices"], dict), (
        f"`prices` is a {type(body['prices']).__name__}; the launch modal indexes it by GPU model"
    )
    assert isinstance(body["spot_prices"], list), (
        f"`spot_prices` is a {type(body['spot_prices']).__name__}. If it became a "
        "dict the two keys would be interchangeable and the frontend's fallback "
        "chain would silently pick either."
    )


def test_the_list_rows_carry_what_the_lookup_can_be_derived_from():
    rows = _body()["spot_prices"]
    if not rows:
        return  # an empty market is not a schema failure
    for row in rows[:5]:
        assert "gpu_model" in row and "rate_cad" in row, (
            f"row {sorted(row)} cannot produce the `{{model: rate}}` lookup the "
            "frontend falls back to building"
        )


def test_the_two_keys_agree_where_they_overlap():
    """A lookup that disagreed with the list would make the fallback a lie."""
    body = _body()
    prices, rows = body["prices"], body["spot_prices"]
    for row in rows[:10]:
        model = row.get("gpu_model")
        if model in prices:
            assert abs(float(prices[model]) - float(row["rate_cad"])) < 1e-6, (
                f"{model}: prices says {prices[model]}, spot_prices says "
                f"{row['rate_cad']}. The frontend treats them as the same number."
            )


def test_the_launch_modal_no_longer_prefers_the_array():
    """The `||` chain is the defect; make it un-writable again without failing."""
    modal = (ROOT / "frontend/src/components/instances/launch-instance-modal.tsx").read_text(
        encoding="utf-8"
    )
    assert not re.search(r"res\.spot_prices\s*\|\|\s*res\.prices", modal), (
        "the launch modal prefers `spot_prices` over `prices` again. A non-empty "
        "array is truthy, so `prices` is unreachable and every lookup returns "
        "undefined."
    )
    assert "res.prices ??" in modal, (
        "the modal no longer reads `prices` first; it is the only key already "
        "shaped as the lookup this component indexes"
    )
