"""Threaded HTTP server mimicking vLLM OpenAI endpoints for full proxy E2E tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


class _VLLMHandler(BaseHTTPRequestHandler):
    usage_profiles: dict[str, dict[str, int]] = {
        "chat": {
            "prompt_tokens": 842,
            "completion_tokens": 156,
            "cached_tokens": 320,
            "accepted_tokens": 118,
            "rejected_tokens": 38,
        },
        "embed": {"prompt_tokens": 12000, "completion_tokens": 0, "cached_tokens": 4000},
    }
    requests_log: list[dict[str, Any]] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_POST(self) -> None:
        body = self._read_body()
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            payload = {}
        route = self.path.rstrip("/").split("/v1/")[-1]
        stream = bool(payload.get("stream"))
        _VLLMHandler.requests_log.append(
            {"path": self.path, "route": route, "stream": stream, "body": payload}
        )

        if route == "embeddings":
            usage = self.usage_profiles["embed"]
            resp = {
                "object": "list",
                "data": [{"embedding": [0.1, 0.2], "index": 0}],
                "usage": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "total_tokens": usage["prompt_tokens"],
                    "prompt_tokens_details": {"cached_tokens": usage["cached_tokens"]},
                },
            }
            self._json_response(200, resp)
            return

        if route in ("chat/completions", "completions"):
            usage = self.usage_profiles["chat"]
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunk1 = {
                    "choices": [{"delta": {"content": "Signal"}, "index": 0}],
                }
                self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode())
                final = {
                    "choices": [{"delta": {}, "index": 0}],
                    "usage": {
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "prompt_tokens_details": {"cached_tokens": usage["cached_tokens"]},
                        "completion_tokens_details": {
                            "accepted_tokens": usage.get("accepted_tokens", 0),
                            "rejected_tokens": usage.get("rejected_tokens", 0),
                        },
                    },
                }
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                return
            resp = {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Fleet ready."}}],
                "usage": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "prompt_tokens_details": {"cached_tokens": usage["cached_tokens"]},
                    "completion_tokens_details": {
                        "accepted_tokens": usage.get("accepted_tokens", 0),
                        "rejected_tokens": usage.get("rejected_tokens", 0),
                    },
                },
            }
            self._json_response(200, resp)
            return

        self._json_response(404, {"error": "not found"})

    def _json_response(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_fake_vllm() -> tuple[HTTPServer, int, threading.Thread]:
    """Bind ephemeral port; return (server, port, thread)."""
    _VLLMHandler.requests_log.clear()
    server = HTTPServer(("127.0.0.1", 0), _VLLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, thread


def warm_worker_for_port(port: int) -> dict:
    return {
        "worker_id": "swk-fake-vllm",
        "host_ip": "127.0.0.1",
        "job_payload": {"http_ports": {"8080": port}},
    }