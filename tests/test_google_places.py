"""Session billing, provider isolation, address resolution and failure behavior."""

import httpx
import pytest

from tests.test_hosts_endpoints_coverage import client, user_headers, _fresh_photon  # noqa: F401

PREDICTION = {
    "placePrediction": {
        "placeId": "ChIJ-test",
        "structuredFormat": {
            "mainText": {"text": "300 Dufferin Avenue"},
            "secondaryText": {"text": "London, ON, Canada"},
        },
    }
}


def _details():
    def component(kind, text, short=None):
        return {"types": [kind], "longText": text, "shortText": short or text}

    return {
        "formattedAddress": "300 Dufferin Avenue, London, ON N6B 2P8, Canada",
        "addressComponents": [
            component("street_number", "300"),
            component("route", "Dufferin Avenue"),
            component("locality", "London"),
            component("administrative_area_level_1", "Ontario", "ON"),
            component("postal_code", "N6B 2P8"),
            component("country", "Canada", "CA"),
        ],
    }


def test_google_session_resolves_only_the_selected_prediction(user_headers, monkeypatch):
    calls = []

    def autocomplete(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200, json={"suggestions": [PREDICTION]}, request=httpx.Request("POST", url)
        )

    def details(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json=_details(), request=httpx.Request("GET", url))

    monkeypatch.setenv("XCELSIOR_GOOGLE_PLACES_API_KEY", "private-google-key")
    monkeypatch.setattr(httpx, "post", autocomplete)
    monkeypatch.setattr(httpx, "get", details)
    for text in ("300 Duff", "300 Dufferin"):
        response = client.post(
            "/api/locations/search",
            headers=user_headers["headers"],
            json={
                "text": text,
                "country": "CA",
                "lang": "fr",
                "session_token": "same-session",
            },
        )
        assert response.status_code == 200
        (row,) = response.json()["suggestions"]
        assert row["provider"] == "google" and row["place_id"] == "ChIJ-test"
        assert "address" not in row and "private-google-key" not in response.text
    assert len(calls) == 2  # No speculative details lookups or cross-session caching.
    response = client.post(
        "/api/locations/resolve",
        headers=user_headers["headers"],
        json={
            "place_id": row["place_id"],
            "session_token": "same-session",
            "country": "CA",
            "lang": "fr",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["address"] == {
        "address_line1": "300 Dufferin Avenue",
        "address_line2": "",
        "city": "London",
        "postal_code": "N6B 2P8",
        "country": "CA",
        "province": "ON",
    }
    assert len(calls) == 3
    for _, kwargs in calls[:2]:
        assert kwargs["json"]["sessionToken"] == "same-session"
        assert kwargs["json"]["includedRegionCodes"] == ["ca"]
        assert kwargs["json"]["languageCode"] == "fr"
        assert set(kwargs["headers"]) == {"X-Goog-Api-Key", "X-Goog-FieldMask"}
    assert calls[-1][1]["params"]["sessionToken"] == "same-session"
    assert calls[-1][1]["headers"]["X-Goog-FieldMask"] == "addressComponents,formattedAddress"


@pytest.mark.parametrize("failure", [403, 429, 503, "timeout"])
def test_google_failure_uses_here_and_backs_off(user_headers, monkeypatch, failure):
    attempts = []

    def autocomplete(url, **kwargs):
        attempts.append(url)
        if failure == "timeout":
            raise httpx.ReadTimeout("private-google-key must not be logged")
        return httpx.Response(
            failure,
            json={"error": {"message": "private-google-key"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setenv("XCELSIOR_GOOGLE_PLACES_API_KEY", "private-google-key")
    monkeypatch.setenv("XCELSIOR_HERE_API_KEY", "here-key")
    monkeypatch.setattr(httpx, "post", autocomplete)
    import routes.hosts as hosts

    monkeypatch.setattr(hosts, "_here_suggestions", lambda *a, **k: [{"id": "here-0"}])
    for text in ("300 Duff", "300 Dufferin"):
        response = client.post(
            "/api/locations/search",
            headers=user_headers["headers"],
            json={"text": text, "session_token": "session"},
        )
        assert response.json()["suggestions"] == [{"id": "here-0"}]
    assert len(attempts) == 1


def test_google_empty_results_use_fallback(user_headers, monkeypatch):
    monkeypatch.setenv("XCELSIOR_GOOGLE_PLACES_API_KEY", "key")
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **k: httpx.Response(200, json={}, request=httpx.Request("POST", url)),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **k: httpx.Response(
            200, json={"features": []}, request=httpx.Request("GET", url)
        ),
    )
    response = client.post(
        "/api/locations/search",
        headers=user_headers["headers"],
        json={"text": "New Street", "session_token": "session"},
    )
    assert response.json()["suggestions"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"place_id": "../other", "session_token": "session"},
        {"place_id": "ChIJ-test", "session_token": ""},
        {"place_id": "ChIJ-test", "session_token": "x" * 37},
    ],
)
def test_resolve_rejects_invalid_identifiers_before_provider_call(user_headers, monkeypatch, body):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("provider called"))
    assert (
        client.post(
            "/api/locations/resolve", headers=user_headers["headers"], json=body
        ).status_code
        == 422
    )


def test_resolve_refuses_an_address_in_a_different_country(user_headers, monkeypatch):
    monkeypatch.setenv("XCELSIOR_GOOGLE_PLACES_API_KEY", "key")
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **k: httpx.Response(200, json=_details(), request=httpx.Request("GET", url)),
    )
    response = client.post(
        "/api/locations/resolve",
        headers=user_headers["headers"],
        json={"place_id": "ChIJ-test", "session_token": "session", "country": "US"},
    )
    assert response.status_code == 502
    assert "Canada" not in response.text
