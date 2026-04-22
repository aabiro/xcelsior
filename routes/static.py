"""Routes: /static/*

Serves a tightly-whitelisted set of static Python files used by the worker
bootstrap + self-update flow. Only files explicitly listed in ``_ALLOWED``
can be served — directory traversal / arbitrary file download is not possible.

Each response includes an ``X-Xcelsior-Agent-SHA256`` header so installers can
verify integrity in a single request (no second round-trip).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter()

# Repository root (…/routes/static.py → parent.parent)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Whitelist: map URL path → real filesystem path under repo root.
# Do NOT add arbitrary files here — each entry is a published contract.
_ALLOWED: dict[str, Path] = {
    "worker_agent.py": _REPO_ROOT / "worker_agent.py",
}


def _read_and_hash(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    return data, sha


# GET + HEAD so `curl -I` / health probes / nginx upstream checks work. HEAD
# returns identical headers with an empty body (Starlette handles this when the
# route is registered with both methods).
@router.api_route("/static/{filename}", methods=["GET", "HEAD"])
def get_static_file(filename: str) -> Response:
    # Defense-in-depth: reject any separator chars before consulting the map.
    # FastAPI already strips path traversal, but a belt-and-braces check costs
    # nothing and makes the whitelist contract explicit.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=404, detail="not found")

    target = _ALLOWED.get(filename)
    if target is None:
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")

    data, sha = _read_and_hash(target)
    # Cache-Control is set by nginx (`add_header ... always`) so we deliberately
    # omit it here to avoid duplicate headers on the wire.
    return Response(
        content=data,
        media_type="text/x-python; charset=utf-8",
        headers={
            "X-Xcelsior-Agent-SHA256": sha,
            "X-Content-Type-Options": "nosniff",
        },
    )
