"""P3/B3 — canonical owner_id helper.

Ensures user_images endpoints have a single source of truth for
resolving the per-user ownership string, avoiding divergence between
positional-default and short-circuit patterns.
"""


def test_canonical_prefers_customer_id_when_set():
    from routes.instances import _canonical_owner_id

    user = {"customer_id": "cus_abc123", "user_id": "u-xyz"}
    assert _canonical_owner_id(user) == "cus_abc123"


def test_canonical_falls_back_when_customer_id_empty_string():
    """customer_id="" is NOT a valid Stripe identity — must fall back."""
    from routes.instances import _canonical_owner_id

    user = {"customer_id": "", "user_id": "u-xyz"}
    assert _canonical_owner_id(user) == "u-xyz"


def test_canonical_falls_back_when_customer_id_missing():
    from routes.instances import _canonical_owner_id

    user = {"user_id": "u-xyz"}
    assert _canonical_owner_id(user) == "u-xyz"


def test_canonical_returns_empty_when_both_missing():
    from routes.instances import _canonical_owner_id

    assert _canonical_owner_id({}) == ""
    assert _canonical_owner_id({"customer_id": "", "user_id": ""}) == ""
    assert _canonical_owner_id({"customer_id": None, "user_id": None}) == ""


def test_canonical_strips_whitespace():
    from routes.instances import _canonical_owner_id

    assert _canonical_owner_id({"customer_id": "  cus_abc  "}) == "cus_abc"
    assert _canonical_owner_id({"customer_id": "", "user_id": "  u-xyz\n"}) == "u-xyz"


def test_user_images_endpoints_use_canonical_helper():
    """Grep guard: all 3 user_images handlers must call the helper."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "routes" / "instances.py").read_text()
    # The three endpoints
    for endpoint in ("api_snapshot_instance", "api_list_user_images", "api_delete_user_image"):
        # Find the function
        idx = src.find(f"def {endpoint}(")
        assert idx >= 0, f"endpoint {endpoint} not found"
        # Body up to the next `def ` at column 0
        body_end = src.find("\n@router", idx)
        if body_end < 0:
            body_end = src.find("\ndef ", idx + 1)
        body = src[idx:body_end if body_end > 0 else len(src)]
        assert "_canonical_owner_id(user)" in body, (
            f"{endpoint} must use _canonical_owner_id(user) helper; "
            f"divergent owner-id patterns are forbidden."
        )
        assert 'user.get("customer_id") or user.get("user_id")' not in body, (
            f"{endpoint} still uses legacy short-circuit pattern"
        )
