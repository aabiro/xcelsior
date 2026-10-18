"""ASGI mode for custom OpenAI-native or HTTP workers."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("xcelsior.serverless.worker_sdk.asgi")


def serve_asgi(app: Any, *, host: str = "0.0.0.0", port: int | None = None) -> None:
    """
    Run an ASGI app (FastAPI, Starlette, etc.) on the worker HTTP port.
    Used for custom Docker images that expose an OpenAI-compatible server.
    """
    import uvicorn

    listen_port = port or int(os.environ.get("XCELSIOR_HTTP_PORT", "8080"))
    log.info("Starting ASGI worker on %s:%d", host, listen_port)
    uvicorn.run(app, host=host, port=listen_port, log_level="info")