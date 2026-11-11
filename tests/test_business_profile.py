"""A business account's public face: display name, website, tagline and logo.

The image tests build their inputs with Pillow, so nothing here is a real
company's mark.
"""

from __future__ import annotations

import io
import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "100000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "100000")

import pytest
from PIL import Image, ImageDraw

from business_profile import (
    DISPLAY_NAME_MAX,
    LOGO_SIZE,
    MAX_UPLOAD_BYTES,
    TAGLINE_MAX,
    normalize_display_name,
    normalize_tagline,
    normalize_website,
    process_logo,
)

PASSWORD = "StrongPass123!"


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def _decode(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _transparent_mark(colour: tuple[int, int, int]) -> Image.Image:
    """A wide mark with generous transparent margins around it."""
    image = Image.new("RGBA", (600, 400), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((100, 150, 500, 250), fill=(*colour, 255))
    return image


# ── Text fields ─────────────────────────────────────────────────────────


def test_display_name_keeps_visible_text_and_drops_invisible_tricks():
    assert normalize_display_name("  North \t  Labs\n") == "North Labs"
    # A right-to-left override would make the badge read differently from what is stored.
    assert normalize_display_name("Acme‮ Labs\x00") == "Acme Labs"
    # A zero-width non-joiner is spelling in Persian, not a trick.
    assert normalize_display_name("می‌خواهم") == "می‌خواهم"
    assert normalize_display_name("") == ""
    assert normalize_display_name("x" * DISPLAY_NAME_MAX) == "x" * DISPLAY_NAME_MAX
    with pytest.raises(ValueError, match="characters or fewer"):
        normalize_display_name("x" * (DISPLAY_NAME_MAX + 1))


def test_tagline_is_one_line_and_bounded():
    assert normalize_tagline("GPU compute,\n  priced fairly") == "GPU compute, priced fairly"
    with pytest.raises(ValueError, match="characters or fewer"):
        normalize_tagline("y" * (TAGLINE_MAX + 1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("example.com", "https://example.com"),
        ("HTTPS://Example.COM/", "https://example.com"),
        ("www.example.co.uk/about", "https://www.example.co.uk/about"),
        ("http://example.com:8080/a?b=1#section", "http://example.com:8080/a?b=1"),
        ("bücher.de", "https://xn--bcher-kva.de"),
    ],
)
def test_website_is_normalised_to_an_address_a_browser_opens(raw, expected):
    assert normalize_website(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "javascript:alert(1)",
        "ftp://example.com",
        "data:text/html,hi",
        "https://user:secret@example.com",
        "https://localhost",
        "https://10.0.0.1",
        "https://exa mple.com",
        "https://example",
        "https://-bad-.com",
        "https://example.com/" + "a" * 300,
    ],
)
def test_website_refuses_what_is_not_a_public_web_address(raw):
    with pytest.raises(ValueError):
        normalize_website(raw)


# ── Logo processing ─────────────────────────────────────────────────────


def test_dark_transparent_logo_gets_a_light_tile_and_stays_transparent():
    logo = process_logo(_png(_transparent_mark((20, 24, 38))))
    assert logo.tile == "light"
    image = _decode(logo.data)
    assert image.size == (LOGO_SIZE, LOGO_SIZE)
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0))[3] == 0
    # The transparent margin of the upload is not part of the mark: what is
    # drawn is scaled to 72 % of the square and centred.
    left, top, right, bottom = image.getchannel("A").getbbox()
    assert abs((right - left) - int(LOGO_SIZE * 0.72)) <= 2
    assert abs((left + right) / 2 - LOGO_SIZE / 2) <= 1.5
    assert abs((top + bottom) / 2 - LOGO_SIZE / 2) <= 1.5


def test_light_transparent_logo_gets_a_dark_tile():
    logo = process_logo(_png(_transparent_mark((246, 246, 240))))
    assert logo.tile == "dark"


def test_wordmark_on_a_flat_background_is_trimmed_and_padded_not_cropped():
    image = Image.new("RGB", (400, 100), (255, 255, 255))
    ImageDraw.Draw(image).rectangle((40, 30, 360, 70), fill=(10, 10, 10))
    logo = process_logo(_png(image))
    assert logo.tile == "none"
    out = _decode(logo.data).convert("RGB")
    assert out.size == (LOGO_SIZE, LOGO_SIZE)
    # The background was extended: the corners and the band above the
    # wordmark are white, and the wordmark is still whole in the middle.
    for point in ((0, 0), (LOGO_SIZE - 1, LOGO_SIZE - 1), (LOGO_SIZE // 2, 60)):
        assert min(out.getpixel(point)) > 240, point
    assert max(out.getpixel((LOGO_SIZE // 2, LOGO_SIZE // 2))) < 40
    # The white margin of the upload was trimmed, so the wordmark itself is
    # scaled to 72 % of the square, with both ends intact.
    dark = out.convert("L").point(lambda v: 255 if v < 128 else 0).getbbox()
    assert dark is not None
    assert abs((dark[2] - dark[0]) - int(LOGO_SIZE * 0.72)) <= 3
    assert abs((dark[0] + dark[2]) / 2 - LOGO_SIZE / 2) <= 2


def test_mark_on_a_flat_coloured_background_keeps_that_colour_around_it():
    image = Image.new("RGB", (300, 300), (24, 92, 200))
    ImageDraw.Draw(image).ellipse((120, 120, 180, 180), fill=(255, 255, 255))
    out = _decode(process_logo(_png(image)).data).convert("RGB")
    # The small mark is enlarged to the logo's share of the square, and the
    # corners stay the background blue rather than turning white or black.
    corner = out.getpixel((4, 4))
    assert abs(corner[0] - 24) < 6 and abs(corner[1] - 92) < 6 and abs(corner[2] - 200) < 6
    white = out.convert("L").point(lambda v: 255 if v > 200 else 0).getbbox()
    assert white is not None and (white[2] - white[0]) > LOGO_SIZE * 0.6


def test_photo_is_squared_by_cropping_and_metadata_is_dropped():
    image = Image.new("RGB", (300, 150))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 149, 149), fill=(220, 30, 30))
    draw.rectangle((150, 0, 299, 149), fill=(30, 30, 220))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° clockwise to display.
    exif[0x010E] = "secret-location-marker"  # ImageDescription
    raw = io.BytesIO()
    image.save(raw, "JPEG", quality=95, exif=exif.tobytes())

    logo = process_logo(raw.getvalue())
    assert logo.tile == "none"
    assert b"secret-location-marker" not in logo.data
    out = _decode(logo.data)
    assert "exif" not in out.info
    out = out.convert("RGB")
    assert out.size == (LOGO_SIZE, LOGO_SIZE)
    # The orientation was applied before cropping: red is on top, blue below.
    top = out.getpixel((LOGO_SIZE // 4, LOGO_SIZE // 5))
    bottom = out.getpixel((LOGO_SIZE // 4, LOGO_SIZE - LOGO_SIZE // 5))
    assert top[0] > 150 and top[2] < 100, top
    assert bottom[2] > 150 and bottom[0] < 100, bottom


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "empty"),
        (b"GIF89a" + b"\x00" * 64, "PNG, JPEG or WebP"),
        (b"%PDF-1.7" + b"\x00" * 64, "PNG, JPEG or WebP"),
        (b"\x89PNG\r\n\x1a\n" + b"not really a png" * 4, "could not be read"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_UPLOAD_BYTES, "2 MB or smaller"),
    ],
)
def test_logo_refuses_what_is_not_a_usable_image(data, message):
    with pytest.raises(ValueError, match=message):
        process_logo(data)


def test_logo_refuses_a_blank_tiny_or_oversized_image():
    with pytest.raises(ValueError, match="fully transparent"):
        process_logo(_png(Image.new("RGBA", (200, 200), (0, 0, 0, 0))))
    with pytest.raises(ValueError, match="too small"):
        process_logo(_png(Image.new("RGB", (20, 20), (200, 10, 10))))
    # Small on disk, enormous once decoded.
    with pytest.raises(ValueError, match="too large"):
        process_logo(_png(Image.new("1", (7000, 7000))))


# ── API ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


def _account(client) -> dict:
    email = f"business-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    registered = client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD, "name": "Dana Founder"}
    )
    assert registered.status_code in (200, 201), registered.text[:300]
    login = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text[:300]
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def _me(client, headers) -> dict:
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 200, response.text[:300]
    return response.json()["user"]


def _upload(client, headers, data: bytes, name: str = "logo.png"):
    return client.post(
        "/api/auth/me/business-logo",
        headers=headers,
        files={"file": (name, data, "image/png")},
    )


@pytest.mark.needs_db
def test_a_personal_account_has_no_business_profile(client):
    headers = _account(client)
    me = _me(client, headers)
    assert me["business_display_name"] == ""
    assert me["business_logo_url"] is None
    assert me["business_logo_tile"] == "none"

    refused = client.patch("/api/auth/me", json={"business_display_name": "North"}, headers=headers)
    assert refused.status_code == 422
    assert "business account" in refused.text

    refused_logo = _upload(client, headers, _png(_transparent_mark((0, 0, 0))))
    assert refused_logo.status_code == 409


@pytest.mark.needs_db
def test_business_profile_round_trip(client):
    headers = _account(client)
    switched = client.patch(
        "/api/auth/me",
        json={"account_type": "business", "business_name": "North Labs Inc."},
        headers=headers,
    )
    assert switched.status_code == 200, switched.text[:300]

    saved = client.patch(
        "/api/auth/me",
        json={
            "business_display_name": "  North   Labs ",
            "business_website": "northlabs.example.com/",
            "business_tagline": "Compute for\nresearch teams",
        },
        headers=headers,
    )
    assert saved.status_code == 200, saved.text[:300]
    me = _me(client, headers)
    assert me["business_display_name"] == "North Labs"
    assert me["business_website"] == "https://northlabs.example.com"
    assert me["business_tagline"] == "Compute for research teams"

    bad_site = client.patch(
        "/api/auth/me", json={"business_website": "javascript:alert(1)"}, headers=headers
    )
    assert bad_site.status_code == 422
    assert _me(client, headers)["business_website"] == "https://northlabs.example.com"

    try:
        uploaded = _upload(client, headers, _png(_transparent_mark((20, 24, 38))))
        assert uploaded.status_code == 200, uploaded.text[:300]
        body = uploaded.json()
        assert body["business_logo_tile"] == "light"
        assert body["business_logo_url"].startswith("/api/auth/me/business-logo?v=")

        me = _me(client, headers)
        assert me["business_logo_url"] == body["business_logo_url"]
        assert me["business_logo_tile"] == "light"

        served = client.get(body["business_logo_url"], headers=headers)
        assert served.status_code == 200
        assert served.headers["content-type"] in ("image/webp", "image/png")
        assert served.headers["x-content-type-options"] == "nosniff"
        assert served.headers["content-security-policy"] == "default-src 'none'"
        assert _decode(served.content).size == (LOGO_SIZE, LOGO_SIZE)

        rejected = _upload(client, headers, b"%PDF-1.7" + b"\x00" * 64, "logo.pdf")
        assert rejected.status_code == 400
        # A refused upload leaves the stored logo alone.
        assert _me(client, headers)["business_logo_url"] == body["business_logo_url"]

        # Back to a personal account: the logo is hidden, not deleted, and the
        # public fields are kept for when the account becomes a business again.
        personal = client.patch("/api/auth/me", json={"account_type": "individual"}, headers=headers)
        assert personal.status_code == 200, personal.text[:300]
        me = _me(client, headers)
        assert me["business_logo_url"] is None
        assert me["business_display_name"] == "North Labs"
        business = client.patch("/api/auth/me", json={"account_type": "business"}, headers=headers)
        assert business.status_code == 200, business.text[:300]
        assert _me(client, headers)["business_logo_url"] == body["business_logo_url"]

        cleared = client.patch(
            "/api/auth/me",
            json={"business_display_name": "", "business_website": "", "business_tagline": ""},
            headers=headers,
        )
        assert cleared.status_code == 200
        me = _me(client, headers)
        assert (me["business_display_name"], me["business_website"], me["business_tagline"]) == ("", "", "")
    finally:
        removed = client.delete("/api/auth/me/business-logo", headers=headers)
    assert removed.status_code == 200
    assert removed.json()["removed"] is True
    assert client.get("/api/auth/me/business-logo", headers=headers).status_code == 404
    assert _me(client, headers)["business_logo_url"] is None
    again = client.delete("/api/auth/me/business-logo", headers=headers)
    assert again.status_code == 200 and again.json()["removed"] is False


@pytest.mark.needs_db
def test_settings_save_route_echoes_what_it_stored(client):
    """The settings form shows what the API stored, from this route's response."""
    headers = _account(client)
    switched = client.put(
        "/api/auth/me/profile",
        json={"account_type": "business", "business_name": "North Labs Inc."},
        headers=headers,
    )
    assert switched.status_code == 200, switched.text[:300]

    saved = client.put(
        "/api/auth/me/profile",
        json={
            "business_display_name": " North  Labs ",
            "business_website": "NorthLabs.example.com/",
            "business_tagline": "Compute,\tpriced fairly",
        },
        headers=headers,
    )
    assert saved.status_code == 200, saved.text[:300]
    body = saved.json()
    assert body["business_display_name"] == "North Labs"
    assert body["business_website"] == "https://northlabs.example.com"
    assert body["business_tagline"] == "Compute, priced fairly"
    me = _me(client, headers)
    assert me["business_website"] == "https://northlabs.example.com"

    refused = client.put(
        "/api/auth/me/profile", json={"business_website": "ftp://northlabs.example.com"}, headers=headers
    )
    assert refused.status_code == 422
    assert "https://" in refused.json().get("detail", refused.text)


@pytest.mark.needs_db
def test_logo_routes_need_a_signed_in_account():
    from fastapi.testclient import TestClient

    from api import app

    # Its own client: the shared one carries the session cookie of the last login.
    anonymous = TestClient(app)
    assert anonymous.get("/api/auth/me/business-logo").status_code == 401
    assert anonymous.delete("/api/auth/me/business-logo").status_code == 401
    upload = anonymous.post(
        "/api/auth/me/business-logo", files={"file": ("logo.png", b"x", "image/png")}
    )
    assert upload.status_code == 401


def test_connector_tokens_do_not_see_the_business_profile(monkeypatch):
    import routes._deps as deps
    from routes.auth import _filter_auth_identity

    monkeypatch.setattr(deps, "_is_delegated_connector_token", lambda user: True)
    identity = {
        "name": "Dana Founder",
        "business_display_name": "North Labs",
        "business_website": "https://northlabs.example.com",
        "business_tagline": "Compute for research teams",
        "business_logo_url": "/api/auth/me/business-logo?v=1",
        "business_logo_tile": "light",
    }
    filtered = _filter_auth_identity({"scopes": ["profile"]}, identity)
    assert filtered == {"name": "Dana Founder"}
