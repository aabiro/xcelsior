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

    operation_count = sum(len(methods) for methods in paths.values())
    assert operation_count <= 60
