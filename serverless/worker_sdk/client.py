"""HTTP client for worker ↔ control-plane callbacks."""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import httpx

log = logging.getLogger("xcelsior.serverless.worker_sdk.client")

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
_MAX_RETRIES = int(os.environ.get("XCELSIOR_WORKER_HTTP_RETRIES", "6"))


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
    base = min(2**attempt, 30)
    return random.uniform(0, base)


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
        self._http = httpx.Client(timeout=self.timeout_sec)

    def close(self) -> None:
        self._http.close()

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
        last_response: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = self._http.get(f"{self.base_url}{path}", headers=self._headers())
                if r.status_code in _RETRYABLE_STATUS:
                    last_exc = RuntimeError(f"HTTP {r.status_code} from {path}")
                    last_response = r
                else:
                    r.raise_for_status()
                    data = r.json()
                    if not isinstance(data, dict):
                        raise RuntimeError(f"Unexpected response from {path}")
                    return data
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
                last_response = None
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(_retry_delay(attempt, last_response))
        assert last_exc is not None
        raise last_exc

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries — survives blue-green API deploy blips."""
        last_exc: Exception | None = None
        last_response: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = self._http.post(
                    f"{self.base_url}{path}",
                    json=body,
                    headers=self._headers(),
                )
                if r.status_code in _RETRYABLE_STATUS:
                    last_exc = RuntimeError(f"HTTP {r.status_code} from {path}")
                    last_response = r
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
                last_response = None
                log.warning(
                    "Worker callback transport error %s (attempt %d/%d): %s",
                    path,
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                )
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(_retry_delay(attempt, last_response))
        assert last_exc is not None
        raise last_exc