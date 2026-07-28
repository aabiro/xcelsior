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
    assert snapshot == generated, (
        "FastAPI v1 changed without regenerating the MCP contract: "
        "refresh mcp/openapi-v1.json and run npm run generate:api"
    )
