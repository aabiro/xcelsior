"""Meta CAPI payload shape. No network."""

import hashlib

import meta_capi


def test_email_and_name_are_normalized_before_hash():
    hashed = meta_capi.hash_email("  Mary@Example.com ")
    assert hashed == hashlib.sha256(b"mary@example.com").hexdigest()
    assert meta_capi.hash_name("Ada") == hashlib.sha256(b"ada").hexdigest()
    assert meta_capi.hash_phone("+1 (416) 555-0100") == hashlib.sha256(b"14165550100").hexdigest()


def test_click_ids_and_ip_are_not_hashed():
    user_data = meta_capi.build_user_data(
        email="ada@xcelsior.ca",
        fbp="fb.1.1710000000000.1234567890",
        fbc="fb.1.1710000000000.IwAR0clickid",
        client_ip="203.0.113.8",
        client_user_agent="Mozilla/5.0",
        external_id="user-123",
        country="ca",
    )
    assert user_data["fbp"] == "fb.1.1710000000000.1234567890"
    assert user_data["fbc"].endswith("IwAR0clickid")
    assert user_data["client_ip_address"] == "203.0.113.8"
    assert user_data["client_user_agent"] == "Mozilla/5.0"
    assert user_data["em"][0] != "ada@xcelsior.ca"
    assert user_data["external_id"][0] != "user-123"
    assert "country" in user_data


def test_event_keeps_the_shared_event_id():
    event = meta_capi.build_event(
        "Purchase",
        "purchase_pi_123",
        custom_data={"value": 25.0, "currency": "CAD"},
        email="ada@xcelsior.ca",
    )
    assert event["event_name"] == "Purchase"
    assert event["event_id"] == "purchase_pi_123"
    assert event["custom_data"]["currency"] == "CAD"
    assert event["action_source"] == "website"


def test_track_event_does_nothing_without_a_token(monkeypatch):
    monkeypatch.delenv("META_CAPI_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("META_PIXEL_ID", raising=False)
    monkeypatch.delenv("META_DATASET_ID", raising=False)
    called = {}

    def _boom(_events):
        called["yes"] = True
        return {}

    monkeypatch.setattr(meta_capi, "deliver", _boom)
    meta_capi.track_event("Purchase", "purchase_1", email="ada@xcelsior.ca")
    assert called == {}


def test_malformed_click_id_is_dropped():
    assert meta_capi._clean_click_id("not-a-cookie", "fbp") is None
    assert meta_capi._clean_click_id("fb.1.1710000000000.1234567890", "fbp")
