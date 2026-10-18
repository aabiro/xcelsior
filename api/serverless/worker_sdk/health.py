"""Minimal /healthz server for queue-mode workers."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

log = logging.getLogger("xcelsior.serverless.worker_sdk.health")

HealthCheck = Callable[[], dict]


class _HealthHandler(BaseHTTPRequestHandler):
    checker: HealthCheck | None = None

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        log.debug(format, *args)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/healthz", "/health"):
            self.send_response(404)
            self.end_headers()
            return
        body = {"ok": True}
        status = 200
        if self.checker:
            try:
                body = self.checker()
                if not body.get("ok", True):
                    status = 503
            except Exception as e:
                body = {"ok": False, "error": str(e)}
                status = 503
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_health_server(
    port: int = 8080,
    *,
    checker: HealthCheck | None = None,
) -> ThreadingHTTPServer:
    handler = type("BoundHealthHandler", (_HealthHandler,), {"checker": checker})
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="healthz")
    thread.start()
    log.info("Health server listening on :%d/healthz", port)
    return server