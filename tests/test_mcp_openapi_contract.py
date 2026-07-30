"""The committed MCP v1 schema must match the mounted FastAPI v1 routes."""

import json
from pathlib import Path

from fastapi.openapi.utils import get_openapi

from api import app


def test_mcp_openapi_snapshot_matches_api():
    generated = get_openapi(
        title="Xcelsior Control-Plane API v1",
        version="1.0.0",
        routes=app.routes,
    )
    generated["paths"] = {
        path: value
        for path, value in generated["paths"].items()
        if path.startswith("/api/v1/")
    }
    snapshot = json.loads(
        (Path(__file__).parents[1] / "mcp" / "openapi-v1.json").read_text()
    )
    if snapshot != generated:
        # Name the exact drift — "dicts differ" on a 30k-line schema is
        # undiagnosable, and this once flaked from environment-dependent
        # generation rather than a real route change.
        snap_paths, gen_paths = set(snapshot.get("paths", {})), set(generated["paths"])
        snap_schemas = snapshot.get("components", {}).get("schemas", {})
        gen_schemas = generated.get("components", {}).get("schemas", {})
        detail = {
            "paths_only_in_snapshot": sorted(snap_paths - gen_paths),
            "paths_only_in_generated": sorted(gen_paths - snap_paths),
            "paths_changed": sorted(
                p for p in snap_paths & gen_paths
                if snapshot["paths"][p] != generated["paths"][p]
            ),
            "schemas_only_in_snapshot": sorted(set(snap_schemas) - set(gen_schemas)),
            "schemas_only_in_generated": sorted(set(gen_schemas) - set(snap_schemas)),
            "schemas_changed": sorted(
                k for k in set(snap_schemas) & set(gen_schemas)
                if snap_schemas[k] != gen_schemas[k]
            ),
        }
        raise AssertionError(
            "FastAPI v1 changed without regenerating the MCP contract: "
            "refresh mcp/openapi-v1.json and run npm run generate:api. "
            f"Drift: { {k: v for k, v in detail.items() if v} }"
        )
