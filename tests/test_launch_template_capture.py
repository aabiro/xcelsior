"""Which workload a job launched (Phase 1.6, 2026-11-22).

The launch form's quick-start templates sent only an image string, so "which
workload do first jobs run?" had no answer. The form now sends the template id;
the route records it only when it is a published template and names the image
being launched, and it lands in the job's execution spec.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")


def _client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


@pytest.fixture(scope="module")
def headers():
    client = _client()
    email = f"tmpl-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Template Capture"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def _template(template_id: str) -> dict:
    from security import get_image_templates

    return next(t for t in get_image_templates() if t["id"] == template_id)


@pytest.mark.parametrize(
    "template_id, image_of, detail",
    [
        ("no-such-template", "pytorch", "Unknown launch_template_id"),
        ("pytorch", "tensorflow", "launch_template_id does not match the image"),
    ],
)
def test_a_template_id_is_recorded_only_when_it_is_true(headers, template_id, image_of, detail):
    # A published template's image, so the image allowlist passes and the
    # template check is the one that answers.
    r = _client().post(
        "/instance",
        json={
            "name": "tmpl",
            "gpu_model": "RTX 3060",
            "launch_template_id": template_id,
            "image": _template(image_of)["image"],
        },
        headers=headers,
    )
    assert r.status_code == 422, r.text
    assert detail in r.text


def test_a_malformed_template_id_never_reaches_the_route(headers):
    r = _client().post(
        "/instance",
        json={
            "name": "tmpl",
            "launch_template_id": "Py Torch!",
            "image": _template("pytorch")["image"],
        },
        headers=headers,
    )
    assert r.status_code == 422, r.text
    assert "launch_template_id" in r.text


@pytest.mark.needs_db
def test_the_template_lands_in_the_execution_spec():
    import scheduler
    from db import _get_pg_pool
    from tests._db_helpers import fund_test_owner, price_test_gpu

    template = _template("pytorch")
    job = scheduler.submit_job(
        name=f"tmpl-{uuid.uuid4().hex[:8]}",
        vram_needed_gb=1,
        gpu_model=price_test_gpu(f"RTX 4090 tmpl-{uuid.uuid4().hex[:6]}"),
        owner=fund_test_owner(f"tmpl-owner-{uuid.uuid4().hex[:8]}"),
        image=template["image"],
        launch_template_id="pytorch",
    )
    try:
        with _get_pg_pool().connection() as conn:
            spec = conn.execute(
                "SELECT spec FROM jobs WHERE job_id=%s", (job["job_id"],)
            ).fetchone()[0]
        assert spec["launch_template_id"] == "pytorch"
        assert spec["admission_request"]["launch_template_id"] == "pytorch"
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job["job_id"],))


def test_an_untemplated_job_spec_is_unchanged():
    """Only set when given, so every other job's spec and hash stay as they were."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "workers/scheduler.py").read_text()
    assert 'if launch_template_id:\n        job["launch_template_id"]' in source


def test_the_form_sends_the_template_and_analytics_record_it():
    from pathlib import Path

    modal = (
        Path(__file__).resolve().parents[1]
        / "api/frontend/src/components/instances/launch-instance-modal.tsx"
    ).read_text()
    assert "launch_template_id:" in modal
    event = modal[modal.index('posthog.capture("gpu_instance_launched"') :]
    event = event[: event.index("});")]
    assert "template_id:" in event and "workload_kind:" in event
