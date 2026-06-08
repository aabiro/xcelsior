"""HTTP client for worker ↔ control-plane callbacks."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

log = logging.getLogger("xcelsior.serverless.worker_sdk.client")

_RETRYABLE_STATUS = frozenset({502, 503, 504})
_MAX_RETRIES = int(os.environ.get("XCELSIOR_WORKER_HTTP_RETRIES", "6"))


class WorkerClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        worker_id: str | None = None,
        token: str | None = None,
        timeout_sec: float = 30.0,
    ):
        self.base_url = (base_url or os.environ.get("XCELSIOR_API_URL", "http://127.0.0.1:9500")).rstrip(
            "/"
        )
        self.worker_id = worker_id or os.environ.get("XCELSIOR_WORKER_ID", "")
        self.token = token or os.environ.get("XCELSIOR_API_TOKEN", "")
        self.timeout_sec = timeout_sec
        if not self.worker_id:
            raise ValueError("worker_id is required")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def signal_ready(self, *, host_id: str = "") -> dict[str, Any]:
        return self._post(
            f"/api/v2/serverless/workers/{self.worker_id}/ready",
            {"host_id": host_id},
        )

    def heartbeat(self) -> dict[str, Any]:
        return self._post(f"/api/v2/serverless/workers/{self.worker_id}/heartbeat", {})

    def claim_job(self) -> dict[str, Any] | None:
        resp = self._post(f"/api/v2/serverless/workers/{self.worker_id}/jobs/claim", {})
        job = resp.get("job")
        return job if job else None

    def complete_job(
        self,
        job_id: str,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict[str, Any]:
        return self._post(
            f"/api/v2/serverless/workers/{self.worker_id}/jobs/{job_id}/complete",
            {
                "output": output,
                "error": error,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    def append_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            f"/api/v2/serverless/workers/{self.worker_id}/jobs/{job_id}/events",
            {"event_type": event_type, "payload": payload},
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        data = self._get(f"/api/v2/serverless/workers/{self.worker_id}/jobs/{job_id}")
        job = data.get("job")
        return job if job else None

    def is_job_cancelled(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return bool(job and str(job.get("status")) == "CANCELLED")

    def _get(self, path: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.Client(timeout=self.timeout_sec) as client:
                    r = client.get(
                        f"{self.base_url}{path}",
                        headers=self._headers(),
                    )
                    if r.status_code in _RETRYABLE_STATUS:
                        last_exc = RuntimeError(f"HTTP {r.status_code} from {path}")
                    else:
                        r.raise_for_status()
                        data = r.json()
                        if not isinstance(data, dict):
                            raise RuntimeError(f"Unexpected response from {path}")
                        return data
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(min(2**attempt, 30))
        assert last_exc is not None
        raise last_exc

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries — survives blue-green API deploy blips."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.Client(timeout=self.timeout_sec) as client:
                    r = client.post(
                        f"{self.base_url}{path}",
                        json=body,
                        headers=self._headers(),
                    )
                    if r.status_code in _RETRYABLE_STATUS:
                        last_exc = RuntimeError(f"HTTP {r.status_code} from {path}")
                        log.warning(
                            "Worker callback retryable %s (attempt %d/%d)",
                            path,
                            attempt + 1,
                            _MAX_RETRIES,
                        )
                    else:
                        r.raise_for_status()
                        data = r.json()
                        if not isinstance(data, dict):
                            raise RuntimeError(f"Unexpected response from {path}")
                        return data
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
                log.warning(
                    "Worker callback transport error %s (attempt %d/%d): %s",
                    path,
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                )
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(min(2**attempt, 30))
        assert last_exc is not None
        raise last_exc