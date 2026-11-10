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
    response = fresh.post("/api/locations/search", json={"country":"CA", "text":"Test Street"})
    assert response.status_code == 401


def test_unconfigured_address_search_is_explicit_and_manual_entry_remains_available(user_headers, monkeypatch):
    monkeypatch.delenv("XCELSIOR_LOQATE_API_KEY", raising=False)
    response = client.post("/api/locations/search", headers=user_headers["headers"], json={"country":"CA", "text":"Test Street"})
    assert response.status_code == 503
    assert "manually" in response.json()["error"]["message"]


def test_lookup_uses_post_body_keeps_key_private_and_handles_address_groups(user_headers, monkeypatch):
    import httpx

    calls=[]
    def provider(url, **kwargs):
        calls.append((url,kwargs))
        return httpx.Response(200, json={"Items":[{"Id":"group-1", "Text":"Test Street", "Description":"2 addresses", "Type":"Street"}]}, request=httpx.Request("POST",url))
    monkeypatch.setenv("XCELSIOR_LOQATE_API_KEY", "unit-only-not-a-service-key")
    monkeypatch.setattr(httpx,"post",provider)
    response=client.post("/api/locations/search",headers=user_headers["headers"],json={"country":"CA","text":"Test Street","container":"group-parent"})
    assert response.status_code == 200, response.text
    assert response.json()["suggestions"][0]["is_address"] is False
    url,kwargs=calls[0]
    assert "?" not in url
    assert kwargs["data"]["Key"] == "unit-only-not-a-service-key"
    assert kwargs["data"]["Container"] == "group-parent"
    assert kwargs["data"]["Countries"] == "CA"
    assert kwargs["data"]["IsMiddleware"] == "true"
    assert "unit-only" not in response.text


def test_address_search_without_a_country_does_not_limit_loqate(user_headers, monkeypatch):
    import httpx

    calls = []

    def provider(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"Items": [{"Id": "a1", "Text": "1 Test Street", "Description": "London", "Type": "Address"}]}, request=httpx.Request("POST", url))

    monkeypatch.setenv("XCELSIOR_LOQATE_API_KEY", "unit-only-not-a-service-key")
    monkeypatch.setattr(httpx, "post", provider)
    response = client.post("/api/locations/search", headers=user_headers["headers"], json={"text": "Test Street"})
    assert response.status_code == 200, response.text
    assert "Countries" not in calls[0]["data"]
    # With no country, the server's own IP must not decide the ranking.
    assert calls[0]["data"]["IsMiddleware"] == "true"
    assert "Origin" not in calls[0]["data"]


@pytest.mark.parametrize("body,header,origin", [
    ({"text": "Test Street"}, "gb", "GB"),          # no country: rank near the visitor
    ({"text": "Test Street"}, "T1", None),          # Tor is not a country
    ({"text": "Test Street"}, "XX", None),          # Cloudflare could not tell
    ({"text": "Test Street"}, "CA; x", None),       # malformed
    ({"country": "CA", "text": "Test Street"}, "GB", None),  # a chosen country wins
])
def test_address_search_ranks_near_the_visitor_only_without_a_country(user_headers, monkeypatch, body, header, origin):
    import httpx

    calls = []

    def provider(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"Items": []}, request=httpx.Request("POST", url))

    monkeypatch.setenv("XCELSIOR_LOQATE_API_KEY", "unit-only-not-a-service-key")
    monkeypatch.setattr(httpx, "post", provider)
    headers = {**user_headers["headers"], "CF-IPCountry": header}
    response = client.post("/api/locations/search", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert calls[0]["data"].get("Origin") == origin


def test_address_retrieve_without_a_country_uses_the_returned_country(user_headers, monkeypatch):
    import httpx

    item = {"CountryIso2": "CA", "Line1": "1 Test Street", "City": "London", "PostalCode": "N6A 1A1", "ProvinceCode": "ON"}
    monkeypatch.setenv("XCELSIOR_LOQATE_API_KEY", "unit-only-not-a-service-key")
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(200, json={"Items": [item]}, request=httpx.Request("POST", url)))
    response = client.post("/api/locations/retrieve", headers=user_headers["headers"], json={"address_id": "address-1"})
    assert response.status_code == 200, response.text
    assert response.json()["address"]["country"] == "CA"
    assert response.json()["address"]["province"] == "ON"


@pytest.mark.parametrize("payload,status",[
    ({"Items":[{"Error":"2","Description":"a secret provider error"}]},502),
    ({"Items":[{"CountryIso2":"US","Line1":"foreign address"}]},422),
    ({"Items":[{"CountryIso2":"CA","Line1":[],"ProvinceCode":"ON"}]},502),
])
def test_address_retrieval_refuses_provider_errors_wrong_countries_and_invalid_types(user_headers,monkeypatch,payload,status):
    import httpx

    monkeypatch.setenv("XCELSIOR_LOQATE_API_KEY","unit-only-not-a-service-key")
    monkeypatch.setattr(httpx,"post",lambda url,**kwargs:httpx.Response(200,json=payload,request=httpx.Request("POST",url)))
    response=client.post("/api/locations/retrieve",headers=user_headers["headers"],json={"country":"CA","address_id":"address-1"})
    assert response.status_code == status, response.text
    assert "secret provider" not in response.text


def test_address_retrieval_returns_real_fields_and_never_invents_a_province(user_headers,monkeypatch):
    import httpx

    item={"CountryIso2":"CA","Line1":"1 Test Street","City":"London","PostalCode":"N6A 1A1"}
    monkeypatch.setenv("XCELSIOR_LOQATE_API_KEY","unit-only-not-a-service-key")
    monkeypatch.setattr(httpx,"post",lambda url,**kwargs:httpx.Response(200,json={"Items":[item]},request=httpx.Request("POST",url)))
    response=client.post("/api/locations/retrieve",headers=user_headers["headers"],json={"country":"CA","address_id":"address-1"})
    assert response.status_code == 200,response.text
    assert response.json()["address"]["province"] == ""
    item["ProvinceCode"]="ON"
    response=client.post("/api/locations/retrieve",headers=user_headers["headers"],json={"country":"CA","address_id":"address-1"})
    assert response.json()["address"]["province"] == "ON"


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
