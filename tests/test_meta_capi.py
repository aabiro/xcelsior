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


def test_city_state_zip_country_normalization_and_hashing():
    assert meta_capi.hash_city("  St. John's ") == hashlib.sha256(b"stjohns").hexdigest()
    assert meta_capi.hash_state("British Columbia") == hashlib.sha256(b"bc").hexdigest()
    assert meta_capi.hash_state("ON") == hashlib.sha256(b"on").hexdigest()
    assert meta_capi.hash_state("New York") == hashlib.sha256(b"ny").hexdigest()
    assert meta_capi.hash_zip("V6B 1A1") == hashlib.sha256(b"v6b1a1").hexdigest()
    assert meta_capi.hash_zip("10001-5555") == hashlib.sha256(b"10001").hexdigest()
    assert meta_capi.hash_country("Canada") == hashlib.sha256(b"ca").hexdigest()
    assert meta_capi.hash_country("USA") == hashlib.sha256(b"us").hexdigest()


def test_user_and_billing_address_wire_ct_st_zp_country():
    user = {
        "email": "ada@xcelsior.ca",
        "name": "Ada Lovelace",
        "user_id": "u_42",
        "billing_address": {
            "city": "Vancouver",
            "region": "BC",
            "postal": "V6B 1A1",
            "country": "CA",
        },
    }
    identity = meta_capi._identity_from_user(user)
    assert identity["city"] == "Vancouver"
    assert identity["state"] == "BC"
    assert identity["zip_code"] == "V6B 1A1"
    assert identity["country"] == "CA"

    event = meta_capi.build_event(
        "InitiateCheckout",
        "pi_abc",
        email=identity["email"],
        first_name=identity["first_name"],
        last_name=identity["last_name"],
        city=identity["city"],
        state=identity["state"],
        zip_code=identity["zip_code"],
        country=identity["country"],
        billing_address=identity["billing_address"],
    )
    ud = event["user_data"]
    assert ud["ct"] == [hashlib.sha256(b"vancouver").hexdigest()]
    assert ud["st"] == [hashlib.sha256(b"bc").hexdigest()]
    assert ud["zp"] == [hashlib.sha256(b"v6b1a1").hexdigest()]
    assert ud["country"] == [hashlib.sha256(b"ca").hexdigest()]
    assert ud["fn"] == [hashlib.sha256(b"ada").hexdigest()]
    assert ud["ln"] == [hashlib.sha256(b"lovelace").hexdigest()]
