"""Canonical worker error envelope."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerError:
    status_code: int
    code: str
    message: str
    retryable: bool = False

    def to_dict(self) -> dict:
        return {
            "status_code": self.status_code,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


def error_envelope(err: WorkerError) -> dict:
    return {"error": err.to_dict()}