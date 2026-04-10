import json
from pathlib import Path

from scripts.generate_public_openapi import build_public_spec


ROOT = Path(__file__).resolve().parent.parent
FERN_OPENAPI = ROOT / "fern" / "openapi.json"


def test_checked_in_public_openapi_matches_generator():
    checked_in = json.loads(FERN_OPENAPI.read_text(encoding="utf-8"))
    generated = build_public_spec()

    def op_set(spec: dict) -> set[tuple[str, str]]:
        return {
            (path, method)
            for path, methods in spec.get("paths", {}).items()
            for method in methods.keys()
        }

    assert op_set(checked_in) == op_set(generated)
    assert [tag["name"] for tag in checked_in.get("tags", [])] == [
        tag["name"] for tag in generated.get("tags", [])
    ]


def test_public_openapi_excludes_internal_routes():
    spec = json.loads(FERN_OPENAPI.read_text(encoding="utf-8"))
    paths = spec["paths"]

    assert "/api/auth/me" in paths
    assert "/oauth/token" in paths
    assert "/api/billing/paypal/create-order" in paths
    assert "/api/billing/paypal/capture-order" in paths

    assert "/host/{host_id}/drain" not in paths
    assert "/billing/bill-all" not in paths
    assert "/agent/versions" not in paths
    assert "/api/auth/device" not in paths
    assert "/api/auth/token" not in paths
