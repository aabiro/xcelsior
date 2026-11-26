"""The status page's "Features in preview" list tells the truth, and misses nothing.

`core/labs.py` lists features that are built but not on for everyone, and the
public `GET /api/status` carries each one's live state for xcelsior.ca/status.
Three ways it could lie, each held here:

* **A state that is not the feature's own.** Each entry must follow the switch
  the feature code reads. If it re-read the environment variable it could
  disagree: the Bitcoin switch accepts only `true`, so `XCELSIOR_BTC_ENABLED=1`
  leaves deposits off, and a page that parsed `1` as on would promise a feature
  that refuses.
* **A feature left off the list.** Every flag that `.env.example` ships off is
  either a preview feature or named in `INTERNAL_FLAGS` with a reason, so a new
  off-by-default feature cannot go unlisted by accident.
* **Copy the page cannot show.** Every feature has an English and a French name
  and description, or the page would print a raw translation key.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import labs

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
I18N = ROOT / "api" / "frontend" / "src" / "lib" / "i18n"


def _state(key: str) -> str:
    return next(f["state"] for f in labs.preview_features() if f["key"] == key)


def test_serverless_reads_the_serverless_switch_and_its_allowlist(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_SERVERLESS_ALLOWLIST", raising=False)
    monkeypatch.setenv("XCELSIOR_SERVERLESS_ENABLED", "false")
    assert _state("serverless") == "off"
    monkeypatch.setenv("XCELSIOR_SERVERLESS_ENABLED", "true")
    assert _state("serverless") == "live"
    monkeypatch.setenv("XCELSIOR_SERVERLESS_ALLOWLIST", "cust-a,cust-b")
    assert _state("serverless") == "limited"


def test_serverless_is_off_in_production_when_nothing_turns_it_on(monkeypatch):
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_SERVERLESS_ENABLED", raising=False)
    assert _state("serverless") == "off"


@pytest.mark.parametrize(
    "module_name, attribute, key",
    [("bitcoin", "BTC_ENABLED", "bitcoin"), ("lightning", "LN_ENABLED", "lightning")],
)
def test_deposits_read_the_switch_the_payment_code_reads(monkeypatch, module_name, attribute, key):
    module = __import__(module_name)
    monkeypatch.setattr(module, attribute, False)
    assert _state(key) == "off"
    monkeypatch.setattr(module, attribute, True)
    assert _state(key) == "live"


def test_a_bitcoin_flag_the_payment_code_rejects_reads_as_off(monkeypatch):
    import importlib

    import bitcoin

    monkeypatch.setenv("XCELSIOR_BTC_ENABLED", "1")
    try:
        importlib.reload(bitcoin)
        assert bitcoin.BTC_ENABLED is False
        assert _state("bitcoin") == "off"
    finally:
        monkeypatch.delenv("XCELSIOR_BTC_ENABLED")
        importlib.reload(bitcoin)


def test_wise_payouts_need_the_switch_and_a_configured_account(monkeypatch):
    monkeypatch.setenv("XCELSIOR_WISE_PAYOUTS_ENABLED", "1")
    monkeypatch.delenv("XCELSIOR_WISE_API_TOKEN", raising=False)
    assert _state("wise_payouts") == "off"
    monkeypatch.setenv("XCELSIOR_WISE_API_TOKEN", "test-token")
    monkeypatch.setenv("XCELSIOR_WISE_PROFILE_ID", "12345")
    assert _state("wise_payouts") == "live"
    monkeypatch.setenv("XCELSIOR_WISE_PAYOUTS_ENABLED", "0")
    assert _state("wise_payouts") == "off"


def test_a_switch_that_cannot_be_read_is_reported_off(monkeypatch):
    def broken() -> str:
        raise ImportError("module missing")

    feature = labs.FEATURES[0]
    monkeypatch.setattr(
        labs,
        "FEATURES",
        (labs.PreviewFeature(feature.key, feature.audience, feature.flag, broken),),
    )
    assert labs.preview_features() == [
        {"key": feature.key, "audience": feature.audience, "state": "off"}
    ]


def _off_by_default_flags() -> set[str]:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^(XCELSIOR_[A-Z0-9_]+_ENABLED)=(?:false|0|no|off)\s*$", text, re.M))


def test_every_flag_that_ships_off_is_a_preview_feature_or_named_internal():
    listed = {feature.flag for feature in labs.FEATURES} | set(labs.INTERNAL_FLAGS)
    unlisted = _off_by_default_flags() - listed
    assert not unlisted, (
        f"{sorted(unlisted)} ship off in .env.example but are in neither core/labs.py FEATURES "
        "nor INTERNAL_FLAGS. Add a customer- or provider-facing one to FEATURES (with copy on "
        "the status page), or name it in INTERNAL_FLAGS with the reason no one outside would notice."
    )


def test_every_listed_flag_still_exists():
    stale = [
        flag
        for flag in {feature.flag for feature in labs.FEATURES} | set(labs.INTERNAL_FLAGS)
        if not re.search(rf"^#?\s*{flag}=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M)
    ]
    assert stale == []


@pytest.mark.parametrize("locale", ["en", "fr"])
def test_every_feature_has_a_name_and_description_in_each_language(locale):
    text = (I18N / f"{locale}.ts").read_text(encoding="utf-8")
    keys = set(re.findall(r'^\s*"(status\.feature_[a-z_]+)":', text, re.M))
    needed = {
        f"status.feature_{feature.key}_{part}"
        for feature in labs.FEATURES
        for part in ("name", "desc")
    }
    needed |= {f"status.feature_audience_{feature.audience}" for feature in labs.FEATURES}
    needed |= {f"status.feature_state_{state}" for state in ("live", "limited", "off")}
    assert needed - keys == set()


def test_the_public_status_carries_only_key_audience_and_state():
    from fastapi.testclient import TestClient

    from api import app

    body = TestClient(app).get("/api/status").json()
    features = body["features"]
    assert [f["key"] for f in features] == [feature.key for feature in labs.FEATURES]
    for feature in features:
        assert set(feature) == {"key", "audience", "state"}
        assert feature["state"] in {"live", "limited", "off"}
