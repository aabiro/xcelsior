"""P3.1 — user_images (pod save-as-template) validators + helpers.

Covers the pure-Python surface of routes.instances that doesn't require a
live Postgres pool. Full e2e of POST /instances/{job_id}/snapshot is
exercised by integration tests in staging.
"""

import pytest
from pydantic import ValidationError

from routes.instances import (
    SnapshotIn,
    _UserImageCompleteIn,
    _build_image_ref,
    _owner_slug,
)


# ── SnapshotIn.name ────────────────────────────────────────────────────────


def test_name_auto_lowercases_for_docker_compat():
    s = SnapshotIn(name="MyImage", tag="latest")
    assert s.name == "myimage"  # docker refuses uppercase in repo names


def test_name_strips_surrounding_whitespace():
    s = SnapshotIn(name="  my-img  ", tag="latest")
    assert s.name == "my-img"


def test_name_must_start_with_alnum():
    for bad in (".start", "-leading", "_under"):
        with pytest.raises(ValidationError):
            SnapshotIn(name=bad, tag="latest")


def test_name_rejects_slash_and_colon():
    with pytest.raises(ValidationError):
        SnapshotIn(name="ns/img", tag="latest")
    with pytest.raises(ValidationError):
        SnapshotIn(name="img:tag", tag="latest")


def test_name_length_cap():
    with pytest.raises(ValidationError):
        SnapshotIn(name="a" * 64, tag="latest")


# ── SnapshotIn.tag ─────────────────────────────────────────────────────────


def test_tag_defaults_to_latest_on_empty():
    s = SnapshotIn(name="foo", tag="")
    assert s.tag == "latest"


def test_tag_preserves_case():
    # Tags are case-sensitive per OCI distribution spec.
    s = SnapshotIn(name="foo", tag="RC1")
    assert s.tag == "RC1"


def test_tag_rejects_leading_dash():
    with pytest.raises(ValidationError):
        SnapshotIn(name="foo", tag="-bad")


# ── SnapshotIn.description ─────────────────────────────────────────────────


def test_description_strips_control_chars():
    s = SnapshotIn(name="foo", tag="latest", description="ok\x00\x01text")
    assert "\x00" not in s.description
    assert "text" in s.description


def test_description_newlines_and_tabs_preserved():
    s = SnapshotIn(name="foo", tag="latest", description="line1\nline2\tcol")
    assert s.description == "line1\nline2\tcol"


def test_description_length_capped():
    with pytest.raises(ValidationError):
        SnapshotIn(name="foo", tag="latest", description="x" * 513)


# ── _owner_slug + _build_image_ref ─────────────────────────────────────────


def test_owner_slug_disambiguates_colliding_inputs():
    # Different raw owner IDs that sanitize to the same prefix must
    # still produce distinct slugs (the sha prefix handles this).
    a = _owner_slug("a.b@x")
    b = _owner_slug("a-b-x")
    assert a != b
    assert a.startswith("a-b-x-") and b.startswith("a-b-x-")


def test_owner_slug_lowercases_and_strips_bad_chars():
    assert _owner_slug("User@Example.COM").startswith("user-example-com-")


def test_owner_slug_fallback_for_empty():
    # Empty owner still yields a valid slug (hash prefix keeps it unique).
    s = _owner_slug("")
    assert s.startswith("user-")
    assert len(s) >= len("user-") + 8


def test_build_image_ref_local_only_without_registry(monkeypatch):
    monkeypatch.delenv("XCELSIOR_REGISTRY_URL", raising=False)
    ref = _build_image_ref("alice@example.com", "my-img", "v1")
    assert ref.startswith("xcl-")
    assert ref.endswith(":v1")
    assert "my-img" in ref


def test_build_image_ref_uses_registry_when_configured(monkeypatch):
    monkeypatch.setenv("XCELSIOR_REGISTRY_URL", "registry.xcelsior.ca")
    ref = _build_image_ref("alice@example.com", "my-img", "v1")
    assert ref.startswith("registry.xcelsior.ca/")
    assert ref.endswith("/my-img:v1")


def test_build_image_ref_strips_trailing_slash_in_registry(monkeypatch):
    monkeypatch.setenv("XCELSIOR_REGISTRY_URL", "registry.xcelsior.ca/")
    ref = _build_image_ref("alice@example.com", "my-img", "v1")
    assert "//" not in ref.split(":", 1)[0]  # no double slash in repo path


# ── _UserImageCompleteIn ───────────────────────────────────────────────────


def test_complete_status_must_be_ready_or_failed():
    for good in ("ready", "failed"):
        _UserImageCompleteIn(status=good, size_bytes=0)
    for bad in ("pending", "done", "", "READY"):
        with pytest.raises(ValidationError):
            _UserImageCompleteIn(status=bad, size_bytes=0)


def test_complete_accepts_size_and_error():
    c = _UserImageCompleteIn(status="failed", size_bytes=0, error="boom")
    assert c.error == "boom"
