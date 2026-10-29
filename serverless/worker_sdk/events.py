"""Normalized progress events emitted during job execution."""

from __future__ import annotations

from typing import Any


def progress_event(message: str, *, pct: float | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"type": "progress", "message": message}
    if pct is not None:
        body["pct"] = max(0.0, min(100.0, pct))
    return body


def log_event(message: str, *, level: str = "info") -> dict[str, Any]:
    return {"type": "log", "level": level, "message": message}


def output_chunk(data: Any) -> dict[str, Any]:
    return {"type": "output", "data": data}