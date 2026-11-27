"""Every call the Terraform provider makes is in the published spec, and works.

The provider (`terraform-provider-xcelsior`) is built from the public API and
nothing else, so a route it needs that is missing from `fern/openapi.json` is a
resource customers cannot declare. SSH keys and launch plans were both missing
until 2026-11-26, though the routes existed: the provider's instance resource
could not have been written against the published surface at all.

The list here is the provider's, not the API's. A route leaving the spec, or
changing method, fails this test before it breaks someone's `terraform apply`.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
SPEC = json.loads((ROOT / "api" / "fern" / "openapi.json").read_text(encoding="utf-8"))

#: (method, path) for every call in terraform-provider-xcelsior/internal/client.
PROVIDER_CALLS = {
    # xcelsior_instance: launch through a plan, read, terminate.
    ("post", "/api/v1/launch-plans/quote"),
    ("post", "/api/v1/launch-plans"),
    ("get", "/api/v1/launch-plans/{plan_id}"),
    ("post", "/api/v1/launch-plans/{plan_id}/approve"),
    ("post", "/api/v1/launch-plans/{plan_id}/execute"),
    ("post", "/api/v1/launch-plans/{plan_id}/revoke"),
    ("get", "/instance/{job_id}"),
    ("post", "/instances/{job_id}/terminate"),
    ("post", "/instances/{job_id}/cancel"),
    # xcelsior_volume
    ("post", "/api/v2/volumes"),
    ("get", "/api/v2/volumes/{volume_id}"),
    ("delete", "/api/v2/volumes/{volume_id}"),
    # A rename is in place; replacing a volume would destroy its data.
    ("patch", "/api/v2/volumes/{volume_id}"),
    ("post", "/api/v2/volumes/{volume_id}/attach"),
    ("post", "/api/v2/volumes/{volume_id}/detach"),
    # xcelsior_ssh_key
    ("post", "/api/ssh/keys"),
    ("get", "/api/ssh/keys"),
    ("delete", "/api/ssh/keys/{key_id}"),
    # xcelsior_serverless_endpoint
    ("post", "/api/v1/serverless/endpoint-plans"),
    ("post", "/api/v1/serverless/endpoint-plans/{plan_id}/execute"),
    ("get", "/api/v2/serverless/endpoints/{endpoint_id}"),
    ("delete", "/api/v2/serverless/endpoints/{endpoint_id}"),
    # Only `name` is updated in place; worker counts replace (see the design).
    ("patch", "/api/v2/serverless/endpoints/{endpoint_id}"),
    # data sources and the provider's own checks
    ("get", "/api/v2/gpu/available"),
    ("get", "/api/pricing/reference"),
    ("get", "/api/v1/spend-envelope"),
    ("get", "/api/auth/me"),
}


def test_every_provider_call_is_published():
    published = {(m, p) for p, methods in SPEC["paths"].items() for m in methods}
    missing = sorted(PROVIDER_CALLS - published)
    assert not missing, f"the provider calls routes the public spec omits: {missing}"


def test_ssh_key_registration_documents_its_body():
    op = SPEC["paths"]["/api/ssh/keys"]["post"]
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    schema = SPEC["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert set(schema["properties"]) == {"name", "public_key"}


def test_no_published_schema_name_is_private():
    """`_ApproveIn` was about to become a public type; names starting `_` are internal."""
    private = [name for name in SPEC["components"]["schemas"] if name.startswith("_")]
    assert not private, private


# ── The SSH key routes, through the typed body ───────────────────────


@pytest.mark.needs_db
def test_an_ssh_key_round_trips_through_the_typed_body():
    from api import app

    client = TestClient(app)
    email = f"sshkey-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "SSH key"},
    )
    assert reg.status_code == 200, reg.text[:300]
    auth = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    blob = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519" + os.urandom(32)).decode()
    public_key = f"ssh-ed25519 {blob} tf@example"

    created = client.post(
        "/api/ssh/keys", json={"name": "terraform", "public_key": public_key}, headers=auth
    )
    assert created.status_code == 200, created.text[:300]
    key_id = created.json()["id"]
    assert created.json()["fingerprint"].startswith("SHA256:")

    listed = client.get("/api/ssh/keys", headers=auth).json()["keys"]
    assert [k["public_key"] for k in listed if k["id"] == key_id] == [public_key]

    duplicate = client.post("/api/ssh/keys", json={"public_key": public_key}, headers=auth)
    assert duplicate.status_code == 409

    assert client.delete(f"/api/ssh/keys/{key_id}", headers=auth).status_code == 200
    assert client.delete(f"/api/ssh/keys/{key_id}", headers=auth).status_code == 404


@pytest.mark.needs_db
def test_an_empty_ssh_key_is_still_a_400_naming_the_field():
    from api import app

    client = TestClient(app)
    email = f"sshkey-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "SSH key"},
    )
    auth = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    r = client.post("/api/ssh/keys", json={"name": "empty"}, headers=auth)
    assert r.status_code == 400
    assert "public_key" in r.text


# ── A volume renamed in place ────────────────────────────────────────


@pytest.mark.needs_db
def test_a_volume_is_renamed_in_place_through_the_published_patch():
    """`xcelsior_volume` updates `name` without replacing the volume and its data."""
    from api import app

    client = TestClient(app)
    email = f"volrename-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Volume rename"},
    )
    assert reg.status_code == 200, reg.text[:300]
    auth = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    volume_id = None
    try:
        created = client.post(
            "/api/v2/volumes",
            json={"name": f"tf-{uuid.uuid4().hex[:6]}", "size_gb": 1},
            headers=auth,
        )
        assert created.status_code == 200, created.text[:300]
        volume_id = created.json()["volume"]["volume_id"]

        renamed = client.patch(
            f"/api/v2/volumes/{volume_id}", json={"name": "tf-renamed"}, headers=auth
        )
        assert renamed.status_code == 200, renamed.text[:300]

        read = client.get(f"/api/v2/volumes/{volume_id}", headers=auth).json()["volume"]
        assert read["volume_id"] == volume_id
        assert read["name"] == "tf-renamed"
    finally:
        if volume_id:
            from db import _get_pg_pool

            with _get_pg_pool().connection() as conn:
                conn.execute("DELETE FROM volume_attachments WHERE volume_id = %s", (volume_id,))
                conn.execute("DELETE FROM volumes WHERE volume_id = %s", (volume_id,))
                conn.commit()
