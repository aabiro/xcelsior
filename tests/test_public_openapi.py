import json
from pathlib import Path

from scripts.generate_public_openapi import PUBLIC_SURFACE_NOTE, build_public_spec

ROOT = Path(__file__).resolve().parent.parent
FERN_OPENAPI = ROOT / "fern" / "openapi.json"
PUBLIC_OPENAPI = ROOT / "public" / "openapi.json"


def test_checked_in_public_openapi_matches_generator():
    """The published spec must equal a fresh generation *in full*.

    This used to compare only the operation set and tag names, so it answered
    "are the right endpoints published?" and never "does the published document
    still describe them correctly?". Five schemas had silently drifted from the
    app's real models underneath a green check — `ServerlessEndpointCreate`
    alone was missing seven fields that the API accepts. Whole-document
    equality is what actually makes this a drift check.
    """
    checked_in = json.loads(FERN_OPENAPI.read_text(encoding="utf-8"))
    generated = build_public_spec()

    def op_set(spec: dict) -> set[tuple[str, str]]:
        return {
            (path, method)
            for path, methods in spec.get("paths", {}).items()
            for method in methods.keys()
        }

    # Asserted first: a surface change gives a far more readable failure than a
    # whole-document mismatch.
    assert op_set(checked_in) == op_set(generated)
    assert [tag["name"] for tag in checked_in.get("tags", [])] == [
        tag["name"] for tag in generated.get("tags", [])
    ]
    assert checked_in == generated, (
        "fern/openapi.json is stale — run scripts/generate_public_openapi.py"
    )
    assert json.loads(PUBLIC_OPENAPI.read_text(encoding="utf-8")) == generated, (
        "public/openapi.json is stale — run scripts/generate_public_openapi.py"
    )


def test_generator_builds_from_the_live_app_not_its_own_output():
    """Guard the feedback loop that made the generator a fixpoint on a file.

    `api.py` reassigns `app.openapi` to serve the *previously generated* file at
    runtime. A generator that calls `app.openapi()` therefore regenerates from
    its own last output: the surface note accumulated one copy per run (it
    reached ten), component order churned ~900 lines per run, and an allowlisted
    operation whose model changed never updated. Reaching the schema through the
    class is what breaks the loop, so failing here means the loop is back.
    """
    import api

    def _reads_its_own_output():
        raise AssertionError(
            "build_public_spec() called app.openapi(), which api.py overrides to "
            "return the previously generated file — the spec would regenerate "
            "from itself instead of from the live routes"
        )

    original = api.app.openapi
    api.app.openapi = _reads_its_own_output  # type: ignore[method-assign]
    try:
        spec = build_public_spec()
    finally:
        api.app.openapi = original  # type: ignore[method-assign]
    assert spec["paths"], "generator produced no paths"


def test_generated_spec_is_ordered_so_diffs_are_meaningful():
    """Component order must not depend on `PYTHONHASHSEED`.

    The reference walk popped from a set, whose iteration order over strings
    varies per process. Every regeneration reshuffled the components — output
    identical in meaning, ~900 lines different on disk, and a drift check that
    could not tell a real change from noise.
    """
    spec = build_public_spec()
    for section, items in (spec.get("components") or {}).items():
        assert list(items) == sorted(items), f"components.{section} is not sorted"


def test_public_surface_note_appears_exactly_once():
    description = build_public_spec()["info"]["description"]
    assert description.count(PUBLIC_SURFACE_NOTE) == 1


def test_public_openapi_excludes_internal_routes():
    spec = json.loads(FERN_OPENAPI.read_text(encoding="utf-8"))
    paths = spec["paths"]

    assert "/api/auth/me" in paths
    assert "/oauth/token" in paths
    assert "/api/billing/paypal/create-order" in paths
    assert "/api/billing/paypal/capture-order" in paths
    assert set(paths["/api/auth/me"].keys()) == {"get"}

    assert "/host" not in paths
    assert "/hosts" not in paths
    assert "/compute-scores" not in paths
    assert "/host/{host_id}/drain" not in paths
    assert "/billing/bill-all" not in paths
    assert "/agent/versions" not in paths
    assert "/api/auth/device" not in paths
    assert "/api/auth/token" not in paths
    assert "/api/auth/me" in paths and "patch" not in paths["/api/auth/me"]
    assert "/api/auth/me" in paths and "delete" not in paths["/api/auth/me"]
    assert "/api/auth/mfa/methods" not in paths
    assert "/api/notifications" not in paths
    assert "/api/chat" not in paths
    assert "/api/v2/privacy/consents" not in paths
    assert "/api/billing/crypto/enabled" not in paths
    assert "/api/billing/refund" not in paths
    assert "/marketplace/list" not in paths
    assert "/marketplace/{host_id}" not in paths
    assert "/api/v2/marketplace/offers" not in paths
    assert "/api/v2/marketplace/allocate" not in paths
    assert "/api/v2/inference/complete/{request_id}" not in paths
    assert "/api/v2/serverless/endpoints" in paths
    assert "/v1/serverless/{endpoint_id}/run" in paths

    operation_count = sum(len(methods) for methods in paths.values())
    assert 70 <= operation_count <= 85
