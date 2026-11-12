"""Smoke coverage for routes/hosts.py (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def user_headers():
    email = f"hostcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Host Cov"},
    ).json()
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200
    return {
        "email": email,
        "user_id": reg["user"].get("user_id", email),
        "headers": {"Authorization": f"Bearer {login.json()['access_token']}"},
    }


def test_hosts_register_web(user_headers):
    r = client.post(
        "/api/hosts/register",
        json={
            "hostname": f"cov-{uuid.uuid4().hex[:6]}",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "cost_per_hour": 0.5,
            "country": "CA",
            "province": "ON",
        },
        headers=user_headers["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    host = r.json().get("host") or {}
    assert host.get("host_id")
    assert host.get("gpu_model") == "RTX 4090"
    assert host.get("admitted") is False
    assert host.get("status") == "pending"

    from scheduler import get_marketplace
    from marketplace import get_marketplace_engine

    assert all(
        listing.get("host_id") != host["host_id"] for listing in get_marketplace(active_only=True)
    )
    assert get_marketplace_engine().get_offer_for_host(host["host_id"]) is None


def test_hosts_register_web_requires_auth():
    r = client.post(
        "/api/hosts/register",
        json={
            "hostname": "no-auth",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
    )
    assert r.status_code == 401


def test_hosts_check_admin():
    r = client.post("/hosts/check", headers=_admin_headers())
    assert r.status_code == 200
    assert "results" in r.json()


def test_hosts_list_and_get(user_headers):
    # host_id must match operator identity for non-admin callers
    host_id = user_headers["user_id"]
    r = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "10.0.0.9",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
        headers=user_headers["headers"],
    )
    assert r.status_code == 200

    r2 = client.get("/hosts?active_only=false", headers=user_headers["headers"])
    assert r2.status_code == 200
    assert any(h["host_id"] == host_id for h in r2.json().get("hosts", []))

    r3 = client.get(f"/host/{host_id}", headers=user_headers["headers"])
    assert r3.status_code == 200
    assert r3.json().get("host", {}).get("host_id") == host_id

    # V2 receipt time lives in a column, while the v1 JSON timestamp can stay
    # stale. The installer must be able to observe the actual server heartbeat.
    from db import pg_connection

    observed_at = 1791633601.0
    with pg_connection() as conn:
        conn.execute(
            "UPDATE hosts SET last_observed_at = to_timestamp(%s) WHERE host_id = %s",
            (observed_at, host_id),
        )
        conn.commit()
    observed = client.get(f"/host/{host_id}", headers=user_headers["headers"])
    assert observed.status_code == 200
    assert observed.json()["host"]["last_observed_at"] == observed_at


def test_hosts_compute_scores():
    r = client.get("/compute-scores")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("scores"), dict)


def _machine_token_for_user(user_headers: dict, *, scopes: str = "hosts:read hosts:write") -> str:
    created = client.post(
        "/api/oauth/clients",
        headers=user_headers["headers"],
        json={
            "client_name": "Worker Host Agent",
            "client_type": "confidential",
            "redirect_uris": [],
            "grant_types": ["client_credentials"],
            "scopes": ["hosts:read", "hosts:write"],
        },
    )
    assert created.status_code == 200, created.text
    oauth_client = created.json()["client"]
    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": oauth_client["client_id"],
            "client_secret": oauth_client["client_secret"],
            "scope": scopes,
        },
    )
    assert token_resp.status_code == 200, token_resp.text
    return token_resp.json()["access_token"]


def test_worker_oauth_heartbeat_for_dashboard_host(user_headers):
    """Dashboard-registered hosts use custom host_id; worker OAuth must match owner."""
    reg = client.post(
        "/api/hosts/register",
        json={
            "hostname": f"worker-{uuid.uuid4().hex[:6]}",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "cost_per_hour": 0.35,
            "country": "CA",
            "province": "ON",
        },
        headers=user_headers["headers"],
    )
    assert reg.status_code == 200, reg.text
    host_id = reg.json()["host"]["host_id"]

    machine_token = _machine_token_for_user(user_headers)
    heartbeat = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "192.168.1.50",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert heartbeat.status_code == 200, heartbeat.text
    assert heartbeat.json()["host"]["admitted"] is False
    assert heartbeat.json()["host"]["status"] == "pending"

    from marketplace import get_marketplace_engine

    assert get_marketplace_engine().get_offer_for_host(host_id) is None


def test_worker_oauth_heartbeat_rejects_foreign_host(user_headers):
    owner = user_headers
    other_email = f"other-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": other_email, "password": "StrongPass123!", "name": "Other"},
    )
    other_login = client.post(
        "/api/auth/login", json={"email": other_email, "password": "StrongPass123!"}
    )
    assert other_login.status_code == 200
    other_headers = {"Authorization": f"Bearer {other_login.json()['access_token']}"}

    reg = client.post(
        "/api/hosts/register",
        json={
            "hostname": f"owned-{uuid.uuid4().hex[:6]}",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "cost_per_hour": 0.5,
            "country": "CA",
            "province": "ON",
        },
        headers=owner["headers"],
    )
    assert reg.status_code == 200, reg.text
    host_id = reg.json()["host"]["host_id"]

    machine_token = _machine_token_for_user({"headers": other_headers})
    heartbeat = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "10.0.0.2",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert heartbeat.status_code == 403


def test_worker_oauth_registers_custom_host_id(user_headers):
    """First heartbeat may claim a custom host_id for the OAuth client creator."""
    host_id = f"gpu-2060-{uuid.uuid4().hex[:8]}"
    machine_token = _machine_token_for_user(user_headers)
    heartbeat = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "192.168.1.60",
            "gpu_model": "RTX 2060",
            "total_vram_gb": 6,
            "free_vram_gb": 6,
            "country": "CA",
            "province": "ON",
        },
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert heartbeat.status_code == 200, heartbeat.text

    listed = client.get("/hosts?active_only=false", headers=user_headers["headers"])
    host = next(h for h in listed.json().get("hosts", []) if h["host_id"] == host_id)
    owner = str(host.get("owner") or "")
    assert owner in {user_headers["user_id"], user_headers["email"]} or owner.startswith(
        "client:oauth_"
    )


def test_worker_oauth_status_patch_is_an_authenticated_tombstone(user_headers):
    """An authenticated worker OAuth PATCH is refused and changes nothing.

    Lifecycle observations come only through the fenced attempt protocol
    (`/agent/v2/attempts/status`, covered in `test_agent_v2.py`); the unfenced
    PATCH is retired for every caller, including a correctly scoped worker.
    """
    me = client.get("/api/auth/me", headers=user_headers["headers"])
    assert me.status_code == 200, me.text
    customer_id = me.json()["user"].get("customer_id") or me.json()["user"].get("user_id")
    from billing import get_billing_engine

    get_billing_engine().deposit(customer_id, 100.0, description="oauth-patch test credits")

    resp = client.post(
        "/instance",
        json={"name": "oauth-patch", "vram_needed_gb": 0, "interactive": True, "gpu_model": "RTX 4090"},
        headers=user_headers["headers"],
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["instance"]["job_id"]

    machine_token = _machine_token_for_user(user_headers, scopes="hosts:read hosts:write")
    patch = client.patch(
        f"/instance/{job_id}",
        json={"status": "starting", "host_id": "tower-test"},
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert patch.status_code == 410, patch.text
    assert patch.json()["error"]["code"] == "v2_protocol_required"
    from scheduler import get_job

    job = get_job(job_id)
    assert job["status"] == "queued" and not job.get("host_id")



@pytest.mark.parametrize("location",[
    {"country":"CA","province":""},
    {"country":"CA","province":"ON","region":"ca-east"},
    {"country":"XX","province":"ON"},
    {"country":"US","province":"ON"},
])
def test_host_registration_refuses_missing_or_conflicting_location(user_headers,location):
    response=client.post("/api/hosts/register",headers=user_headers["headers"],json={
        "hostname":"location-validation", "gpu_model":"RTX 4090", "vram_gb":24, **location,
    })
    assert response.status_code == 422, response.text


def test_location_catalog_uses_country_names_and_valid_subdivisions():
    response=client.get("/api/locations?country=CA")
    assert response.status_code == 200
    data=response.json()
    assert len(data["countries"]) >= 249
    assert {item["code"]:item["name"] for item in data["countries"]}["CA"] == "Canada"
    assert {item["code"]:item["name"] for item in data["subdivisions"]}["ON"] == "Ontario"
    assert client.get("/api/locations?country=XX").status_code == 422


def test_address_lookup_requires_an_interactive_account():
    fresh = TestClient(app)
    for path, body in (
        ("/api/locations/search", {"country": "CA", "text": "Test Street"}),
        ("/api/locations/check", {"country": "CA", "province": "ON"}),
    ):
        assert fresh.post(path, json=body).status_code == 401


@pytest.fixture(autouse=True)
def _fresh_photon(monkeypatch):
    """Each test starts with an empty search cache and no back-off pause."""
    import routes.hosts as hosts

    monkeypatch.setattr(hosts, "_PHOTON_CACHE", {})
    monkeypatch.setattr(hosts, "_photon_paused_until", 0.0)


def _photon(monkeypatch, features, calls=None, status=200, by_country=None):
    """Stand in for Photon: record each request, answer with these GeoJSON features.

    ``by_country`` answers a request limited to a country with its own features.
    """
    import httpx

    def provider(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        limit = dict(kwargs["params"]).get("countrycode")
        chosen = (by_country or {}).get(limit, features) if limit else features
        return httpx.Response(status, json={"type": "FeatureCollection", "features": chosen}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", provider)


def _feature(**properties):
    return {"type": "Feature", "properties": {"type": "house", **properties}}


def _search(user_headers, body, **headers):
    return client.post("/api/locations/search", headers={**user_headers["headers"], **headers}, json=body)


def test_address_search_needs_no_key_and_fills_every_field_from_one_request(user_headers, monkeypatch):
    calls = []
    monkeypatch.delenv("XCELSIOR_PHOTON_URL", raising=False)
    _photon(monkeypatch, [_feature(housenumber="300", street="Dufferin Avenue", city="London", state="Ontario", postcode="n6b1z2", countrycode="CA")], calls)
    response = _search(user_headers, {"country": "CA", "text": "300 Dufferin", "lang": "fr"})
    assert response.status_code == 200, response.text
    (suggestion,) = response.json()["suggestions"]
    assert suggestion["address"] == {
        "address_line1": "300 Dufferin Avenue", "address_line2": "", "city": "London",
        "postal_code": "N6B 1Z2", "country": "CA", "province": "ON",
    }
    assert suggestion["description"] == "London, ON, N6B 1Z2, Canada"
    (url, kwargs), = calls  # a chosen country is one request
    assert url == "https://photon.komoot.io/api/"
    params = kwargs["params"]
    assert ("q", "300 Dufferin") in params and ("countrycode", "CA") in params and ("lang", "fr") in params
    assert {value for key, value in params if key == "layer"} == {"house", "street"}
    # Only the query goes out: no cookies, IP or account.
    assert set(kwargs["headers"]) == {"User-Agent"}


def test_address_search_ignores_a_language_photon_does_not_offer(user_headers, monkeypatch):
    calls = []
    _photon(monkeypatch, [], calls)
    _search(user_headers, {"text": "Test Street", "lang": "xx-evil"})
    assert all(key != "lang" for key, _ in calls[0][1]["params"])


def test_address_search_uses_a_self_hosted_photon_when_one_is_set(user_headers, monkeypatch):
    calls = []
    monkeypatch.setenv("XCELSIOR_PHOTON_URL", "http://photon.internal:2322/")
    _photon(monkeypatch, [], calls)
    response = _search(user_headers, {"text": "Test Street"})
    assert response.status_code == 200, response.text
    assert response.json()["suggestions"] == []
    assert calls[0][0] == "http://photon.internal:2322/api/"


def test_a_visitor_sees_their_own_country_first_without_losing_the_world(user_headers, monkeypatch):
    calls = []
    near = [_feature(housenumber="1", street="Grafton Street", city="Dublin", postcode="D02 K198", countrycode="IE")]
    world = [
        _feature(housenumber="1", street="Grafton Street", city="Grafton", state="MA", postcode="01519", countrycode="US"),
        _feature(housenumber="1", street="Grafton Street", city="Dublin", postcode="D02 K198", countrycode="IE"),
    ]
    _photon(monkeypatch, world, calls, by_country={"IE": near})
    response = _search(user_headers, {"text": "1 Grafton Street"}, **{"CF-IPCountry": "IE"})
    lines = [(s["address"]["city"], s["address"]["country"]) for s in response.json()["suggestions"]]
    assert lines == [("Dublin", "IE"), ("Grafton", "US")]  # nearby first, the same door once
    assert [dict(kwargs["params"]).get("countrycode") for _, kwargs in calls] == ["IE", None]
    assert response.json()["suggestions"][0]["address"]["province"] == "D"  # Dublin's county, from the city


def test_a_strong_local_answer_costs_one_request(user_headers, monkeypatch):
    calls = []
    near = [_feature(housenumber=str(n), street="King Street West", city="Toronto", state="ON", postcode="M5V 1J2", countrycode="CA") for n in range(1, 6)]
    _photon(monkeypatch, [], calls, by_country={"CA": near})
    response = _search(user_headers, {"text": "King Street West"}, **{"CF-IPCountry": "CA"})
    assert len(response.json()["suggestions"]) == 5
    assert len(calls) == 1


@pytest.mark.parametrize("header", ["T1", "XX", "CA; x", ""])
def test_a_visitor_country_that_is_not_one_searches_the_world_only(user_headers, monkeypatch, header):
    calls = []
    _photon(monkeypatch, [], calls)
    _search(user_headers, {"text": "10 King Street"}, **({"CF-IPCountry": header} if header else {}))
    assert [dict(kwargs["params"]).get("countrycode") for _, kwargs in calls] == [None]


def test_answers_are_cached_and_a_failing_service_is_left_alone_for_a_minute(user_headers, monkeypatch):
    import httpx

    calls = []
    _photon(monkeypatch, [_feature(housenumber="1", street="Main Street", city="Ottawa", state="ON", countrycode="CA")], calls)
    first = _search(user_headers, {"country": "CA", "text": "1 Main Street"}).json()
    again = _search(user_headers, {"country": "CA", "text": "1 Main Street"}).json()
    assert first == again and len(calls) == 1

    def refused(url, **kwargs):
        calls.append(url)
        raise httpx.ConnectError("refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", refused)
    assert _search(user_headers, {"country": "CA", "text": "2 Main Street"}).status_code == 502
    assert _search(user_headers, {"country": "CA", "text": "3 Main Street"}).status_code == 502
    assert len(calls) == 2  # the second failure never left this server


@pytest.mark.parametrize("state,postcode,province", [
    ("QC", "H3B 5L1", "QC"), ("Québec", "H3B 5L1", "QC"), ("Quebec", None, "QC"),
    (None, "H3B 5L1", "QC"),       # no state named: Quebec's postal prefix H decides
    ("Atlantis", None, ""),        # nothing to go on: the person picks
])
def test_address_search_fills_the_region_only_from_real_evidence(user_headers, monkeypatch, state, postcode, province):
    _photon(monkeypatch, [_feature(housenumber="1000", street="Rue Sainte-Catherine Ouest", city="Montréal", state=state, postcode=postcode, countrycode="CA")])
    response = _search(user_headers, {"text": "1000 Sainte-Catherine"})
    assert response.json()["suggestions"][0]["address"]["province"] == province


def test_regions_come_from_the_name_the_country_uses(user_headers, monkeypatch):
    _photon(monkeypatch, [
        _feature(housenumber="77", street="Unter den Linden", city="Berlin", postcode="10117", countrycode="DE"),
        _feature(housenumber="1", street="9", locality="Marunouchi 1", district="Chiyoda", city="Tokyo", state="東京都", postcode="1000005", countrycode="JP"),
        _feature(housenumber="110", street="Sejong-daero", city="Seoul", postcode="04524", countrycode="KR"),
        _feature(housenumber="42", street="Calle de Alcalá", city="Madrid", state="Comunidad de Madrid", postcode="28014", countrycode="ES"),
        _feature(housenumber="10", street="Downing Street", city="London", state="England", postcode="SW1A 2AA", countrycode="GB"),
    ])
    found = {s["address"]["country"]: s for s in _search(user_headers, {"text": "anything at all"}).json()["suggestions"]}
    assert found["DE"]["address"]["province"] == "BE"          # Berlin is its own Land
    assert found["JP"]["address"] == {
        "address_line1": "Marunouchi 1-9-1", "address_line2": "", "city": "Chiyoda",
        "postal_code": "100-0005", "country": "JP", "province": "13",
    }
    assert found["KR"]["address"]["province"] == "11"
    assert found["ES"]["address"]["province"] == "M"           # the province Spanish post uses
    assert found["GB"]["address"]["province"] == "ENG"
    assert found["GB"]["description"] == "London, SW1A 2AA, United Kingdom"  # the UK post names no region


def test_a_typed_house_number_is_kept_on_a_street_osm_knows_without_it(user_headers, monkeypatch):
    _photon(monkeypatch, [
        {"type": "Feature", "properties": {"type": "street", "name": "Via del Corso", "city": "Roma", "state": "Lazio", "postcode": "00187", "countrycode": "IT"}},
        {"type": "Feature", "properties": {"type": "street", "name": "Corso Vittorio Emanuele II", "city": "Roma", "countrycode": "IT"}},
        {"type": "Feature", "properties": {"type": "street", "name": "Route 300", "city": "Nowhere", "countrycode": "US", "state": "OH"}},
        {"type": "Feature", "properties": {"type": "street", "name": "Elm Street", "city": "Springfield", "countrycode": "US", "state": "IL"}},
    ])
    lines = [s["address"]["address_line1"] for s in _search(user_headers, {"text": "Via del Corso 300"}).json()["suggestions"]]
    # Numbered as typed, in Italian order; a street that already carries the number,
    # or one that wasn't typed at all, is left as it is.
    assert lines == ["Via del Corso 300", "Corso Vittorio Emanuele II 300", "Route 300", "Elm Street"]


def test_address_search_writes_numbers_the_local_way_and_merges_one_door(user_headers, monkeypatch):
    _photon(monkeypatch, [
        _feature(name="Hotel Adlon", housenumber="77", street="Unter den Linden", city="Berlin", postcode="10117", countrycode="DE"),
        _feature(name="Restaurant", housenumber="77", street="Unter den Linden", city="Berlin", postcode="10117", countrycode="DE"),
        _feature(housenumber="110", street="세종대로", city="서울특별시", postcode="04524", countrycode="KR"),
        _feature(housenumber="15", locality="Lhota", countrycode="CZ"),  # a village numbers houses, not streets
        _feature(housenumber="5", city="Toronto", countrycode="CA"),     # no street or village: not an address
        "not a feature",
    ])
    lines = [s["address"]["address_line1"] for s in _search(user_headers, {"text": "Unter den Linden 77"}).json()["suggestions"]]
    assert lines == ["Unter den Linden 77", "세종대로 110", "Lhota 15"]


def test_address_search_with_a_country_drops_results_from_elsewhere(user_headers, monkeypatch):
    _photon(monkeypatch, [
        _feature(housenumber="300", street="London Avenue", city="Lafayette", state="CO", postcode="80026", countrycode="US"),
        _feature(housenumber="300", street="Dufferin Avenue", city="London", state="ON", postcode="N6B 1Z2", countrycode="CA"),
    ])
    response = _search(user_headers, {"country": "CA", "text": "300 London"})
    assert [s["address"]["country"] for s in response.json()["suggestions"]] == ["CA"]


@pytest.mark.parametrize("status,payload", [(503, None), (200, "not json")])
def test_address_search_outage_sends_people_to_manual_entry(user_headers, monkeypatch, status, payload):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: httpx.Response(
        status, content=b"<html>down</html>" if payload else b"{}", request=httpx.Request("GET", url)))
    response = _search(user_headers, {"text": "Test Street"})
    assert response.status_code == 502
    assert "below" in response.json()["error"]["message"]


def test_address_check_is_free_and_returns_postal_format_errors_and_warnings(user_headers, monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("the check must not call a provider"))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("the check must not call a provider"))
    body = {"country": "CA", "province": "AB", "address_line1": "4915 50 St,", "city": "Lloydminster", "postal_code": "s9v0a1"}
    response = client.post("/api/locations/check", headers=user_headers["headers"], json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["address"]["postal_code"] == "S9V 0A1"
    assert result["address"]["address_line1"] == "4915 50 St"
    assert result["errors"] == []
    assert [w["code"] for w in result["warnings"]] == ["postal_code_region"]
    assert "Saskatchewan, not Alberta" in result["warnings"][0]["message"]
    assert result["lines"] == ["4915 50 St", "Lloydminster AB S9V 0A1", "Canada"]

    bad = client.post("/api/locations/check", headers=user_headers["headers"], json={**body, "postal_code": "12345"}).json()
    assert [e["field"] for e in bad["errors"]] == ["postal_code"]
    assert bad["errors"][0]["message"] == "12345 isn't a valid postal code in Canada."


def test_address_check_rejects_an_unknown_country(user_headers):
    response = client.post("/api/locations/check", headers=user_headers["headers"], json={"country": "ZZ"})
    assert response.status_code == 422


# Physical relocation keeps the host identity and operator admission intact.
def _location_host(user_headers):
    response = client.post("/api/hosts/register", json={
        "hostname": f"location-{uuid.uuid4().hex[:8]}", "gpu_model": "RTX 4090",
        "vram_gb": 24, "country": "CA", "province": "ON",
    }, headers=user_headers["headers"])
    assert response.status_code == 200, response.text
    host = response.json()["host"]
    return client.get(f"/host/{host['host_id']}", headers=user_headers["headers"]).json()["host"]


def _save_location(host, user_headers, country="US", province="NY"):
    return client.patch(f"/api/hosts/{host['host_id']}/location", json={
        "country": country, "province": province, "expected_version": host["version"],
    }, headers=user_headers["headers"])


def test_host_owner_can_change_physical_location_repeatedly(user_headers):
    host = _location_host(user_headers)
    first = _save_location(host, user_headers)
    assert first.status_code == 200, first.text
    moved = first.json()["host"]
    assert (moved["country"], moved["province"], moved["region"]) == ("US", "NY", "us-ny")
    assert moved["location_complete"] is True
    assert "New York" in moved["location_label"]
    assert moved["host_id"] == host["host_id"]
    second = _save_location(moved, user_headers, "CA", "BC")
    assert second.status_code == 200, second.text
    assert second.json()["host"]["region"] == "ca-bc"
    assert second.json()["host"]["version"] == moved["version"] + 1
    from db import pg_connection
    with pg_connection() as conn:
        events = conn.execute("SELECT count(*) FROM outbox_events WHERE aggregate_id=%s AND event_type='host.v1.location_changed'", (host["host_id"],)).fetchone()[0]
    assert events == 2


def test_host_location_rejects_missing_subdivision_and_stale_version(user_headers):
    host = _location_host(user_headers)
    assert _save_location(host, user_headers, "US", "").status_code == 422
    assert _save_location(host, user_headers, "ZZ", "").status_code == 422
    assert _save_location(host, user_headers).status_code == 200
    assert _save_location(host, user_headers, "CA", "QC").status_code == 409


def test_location_edit_and_move_readiness_are_owner_only(user_headers):
    host = _location_host(user_headers)
    other = client.post("/api/auth/register", json={
        "email": f"move-other-{uuid.uuid4().hex[:8]}@example.test", "password": "StrongPass123!", "name": "Other Account",
    })
    assert other.status_code == 200
    headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert client.patch(f"/api/hosts/{host['host_id']}/location", json={"country":"US", "province":"NY", "expected_version":host["version"]}, headers=headers).status_code == 404
    token = _machine_token_for_user(user_headers)
    assert client.patch(f"/api/hosts/{host['host_id']}/location", json={"country":"US", "province":"NY", "expected_version":host["version"]}, headers={"Authorization":f"Bearer {token}"}).status_code == 403


def test_stale_heartbeat_cannot_undo_confirmed_location(user_headers):
    from db import DatabaseOps, pg_connection
    host = _location_host(user_headers)
    # This snapshot was read before the owner committed the move.
    stale = dict(host)
    saved = _save_location(host, user_headers)
    assert saved.status_code == 200, saved.text
    with pg_connection() as conn:
        DatabaseOps.upsert_host(conn, stale, backend="postgres")
        conn.commit()
        row = conn.execute("SELECT country,province,region,payload->>'country',payload->>'location_confirmed_at' FROM hosts WHERE host_id=%s", (host["host_id"],)).fetchone()
    assert row[:4] == ("US", "NY", "us-ny", "US")
    assert row[4]
    machine = _machine_token_for_user(user_headers)
    heartbeat = client.put("/host", json={
        "host_id":host["host_id"], "ip":"10.0.0.9", "gpu_model":"RTX 4090",
        "total_vram_gb":24, "free_vram_gb":24, "country":"CA", "province":"ON", "region":"ca-on",
    }, headers={"Authorization":f"Bearer {machine}"})
    assert heartbeat.status_code == 200, heartbeat.text
    assert heartbeat.json()["host"]["region"] == "us-ny"


def test_agent_cannot_forge_location_confirmation(user_headers):
    from db import DatabaseOps, pg_connection
    host = _location_host(user_headers)
    forged = {**host, "location_confirmed_at":123, "location_confirmed_by":"forged"}
    with pg_connection() as conn:
        DatabaseOps.upsert_host(conn, forged, backend="postgres")
        conn.commit()
        payload = conn.execute("SELECT payload FROM hosts WHERE host_id=%s", (host["host_id"],)).fetchone()[0]
    assert "location_confirmed_at" not in payload


def test_admitted_idle_host_location_is_editable_without_a_move_workflow(user_headers):
    from db import pg_connection
    host = _location_host(user_headers)
    with pg_connection() as conn:
        conn.execute("UPDATE hosts SET status='active',admission_state='admitted',administrative_state='admitted',availability_state='ready',last_observed_at=clock_timestamp(),admission_version=7 WHERE host_id=%s", (host["host_id"],))
        before=conn.execute("SELECT last_observed_at,admission_version,administrative_state,availability_state FROM hosts WHERE host_id=%s", (host["host_id"],)).fetchone()
        conn.commit()
    result=_save_location(host, user_headers)
    assert result.status_code == 200,result.text
    moved=result.json()["host"]
    assert moved["status"] == "active"
    assert moved["admission_version"] == 7
    assert moved["region"] == "us-ny"
    with pg_connection() as conn:
        after=conn.execute("SELECT last_observed_at,admission_version,administrative_state,availability_state FROM hosts WHERE host_id=%s", (host["host_id"],)).fetchone()
    assert after == before


def test_move_does_not_interrupt_existing_workloads(user_headers):
    from tests.fixtures.native_runtime import running_instance
    with running_instance(owner=user_headers["user_id"]) as instance:
        host_id=instance["host_id"]
        from db import pg_connection
        with pg_connection() as conn:
            # A located host: recording a first location is not a move.
            conn.execute("UPDATE hosts SET owner_id=%s,tenant_id=%s,country='CA',province='ON',region='ca-on' WHERE host_id=%s", (user_headers["user_id"],user_headers["user_id"],host_id))
            conn.commit()
        host=client.get(f"/host/{host_id}", headers=user_headers["headers"]).json()["host"]
        assert _save_location(host, user_headers).status_code == 409
        from db import pg_connection
        with pg_connection() as conn:
            row=conn.execute("SELECT status,phase FROM jobs WHERE job_id=%s", (instance["job_id"],)).fetchone()
        assert row == ("running", "running")


def test_unlocated_host_records_its_location_while_work_runs(user_headers):
    """A host with no location yet is not moving, so running work cannot block it."""
    from tests.fixtures.native_runtime import running_instance
    with running_instance(owner=user_headers["user_id"]) as instance:
        host_id=instance["host_id"]
        from db import pg_connection
        with pg_connection() as conn:
            conn.execute("UPDATE hosts SET owner_id=%s,tenant_id=%s,country='CA',province='',region='ca-central' WHERE host_id=%s", (user_headers["user_id"],user_headers["user_id"],host_id))
            conn.commit()
        host=client.get(f"/host/{host_id}", headers=user_headers["headers"]).json()["host"]
        assert host["location_complete"] is False
        saved=_save_location(host, user_headers, "CA", "ON")
        assert saved.status_code == 200, saved.text
        located=saved.json()["host"]
        assert (located["country"], located["province"], located["region"]) == ("CA", "ON", "ca-on")
        # Once located, a different site is a move, and the running work blocks it.
        assert _save_location(located, user_headers, "CA", "BC").status_code == 409
        with pg_connection() as conn:
            row=conn.execute("SELECT status,phase FROM jobs WHERE job_id=%s", (instance["job_id"],)).fetchone()
        assert row == ("running", "running")
