"""Routes: /static/*

Serves a tightly-whitelisted set of static Python files used by the worker
bootstrap + self-update flow. Only files explicitly listed in ``_ALLOWED``
can be served — directory traversal / arbitrary file download is not possible.

Each response includes integrity headers:
- ``X-Xcelsior-Agent-SHA256`` — sha256 of body
- ``X-Xcelsior-Agent-Signature`` — base64 ed25519 detached signature (B6)
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"

_ALLOWED: dict[str, Path] = {
    "worker_agent.py": _REPO_ROOT / "worker_agent.py",
    "security.py": _REPO_ROOT / "security.py",
    "nvml_telemetry.py": _REPO_ROOT / "nvml_telemetry.py",
    "worker_image_cache.py": _REPO_ROOT / "worker_image_cache.py",
    "worker_nfs.py": _REPO_ROOT / "worker_nfs.py",
    "worker_nvme_cache.py": _REPO_ROOT / "worker_nvme_cache.py",
    "worker_luks_volumes.py": _REPO_ROOT / "worker_luks_volumes.py",
    "worker-requirements.txt": _SCRIPTS / "worker-requirements.txt",
}

_SIGNABLE = frozenset(
    {
        "worker_agent.py",
        "security.py",
        "nvml_telemetry.py",
        "worker_image_cache.py",
        "worker_nfs.py",
        "worker_nvme_cache.py",
        "worker_luks_volumes.py",
    }
)


def _read_and_hash(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    return data, sha


def _signature_b64(filename: str, data: bytes) -> str | None:
    if filename not in _SIGNABLE:
        return None
    sig_path = _SCRIPTS / f"{filename}.sig"
    if not sig_path.is_file():
        # Fall back to shared worker_agent signature file naming convention
        if filename == "worker_agent.py":
            sig_path = _SCRIPTS / "worker_agent.py.sig"
        else:
            sig_path = _SCRIPTS / f"{filename}.sig"
    if not sig_path.is_file():
        return None
    return base64.b64encode(sig_path.read_bytes()).decode("ascii")


@router.api_route("/static/{filename}", methods=["GET", "HEAD"])
def get_static_file(filename: str) -> Response:
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=404, detail="not found")

    target = _ALLOWED.get(filename)
    if target is None:
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")

    data, sha = _read_and_hash(target)
    headers = {
        "X-Xcelsior-Agent-SHA256": sha,
        "X-Content-Type-Options": "nosniff",
    }
    sig = _signature_b64(filename, data)
    if sig:
        headers["X-Xcelsior-Agent-Signature"] = sig
    elif filename in _SIGNABLE:
        raise HTTPException(status_code=503, detail="agent signature unavailable")

    media = "text/plain; charset=utf-8"
    if filename.endswith(".py"):
        media = "text/x-python; charset=utf-8"

    return Response(content=data, media_type=media, headers=headers)