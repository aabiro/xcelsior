# Xcelsior — Serverless job webhooks (HMAC-signed, SSRF-safe, retried)

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.webhooks")

WEBHOOK_MAX_ATTEMPTS = int(os.environ.get("XCELSIOR_SERVERLESS_WEBHOOK_MAX_ATTEMPTS", "3"))
WEBHOOK_TIMEOUT_SEC = float(os.environ.get("XCELSIOR_SERVERLESS_WEBHOOK_TIMEOUT_SEC", "10"))
WEBHOOK_BACKOFF_BASE_SEC = float(
    os.environ.get("XCELSIOR_SERVERLESS_WEBHOOK_BACKOFF_BASE_SEC", "5")
)
WEBHOOK_MAX_URL_LEN = 2048
WEBHOOK_MAX_BODY_BYTES = int(os.environ.get("XCELSIOR_SERVERLESS_WEBHOOK_MAX_BODY_BYTES", "65536"))

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def _webhook_secret() -> bytes:
    from scheduler import API_TOKEN

    raw = os.environ.get("XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET", API_TOKEN or "")
    return (raw or "xcelsior-serverless-webhook-dev").encode("utf-8")


def _blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def validate_webhook_url(url: str) -> str | None:
    """Return normalized URL or None if unsafe/invalid."""
    u = (url or "").strip()
    if not u or len(u) > WEBHOOK_MAX_URL_LEN:
        return None
    parsed = urlparse(u)
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.hostname or "").strip().lower()
    if not host or host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return None
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, parsed.port or 443):
            if family not in (socket.AF_INET, socket.AF_INET6):
                continue
            ip_str = sockaddr[0]
            addr = ipaddress.ip_address(ip_str.split("%")[0])
            if _blocked_ip(addr):
                return None
    except OSError:
        return None
    return u


def build_webhook_payload(job: dict) -> dict[str, Any]:
    return {
        "id": job.get("job_id"),
        "endpoint_id": job.get("endpoint_id"),
        "status": job.get("status"),
        "output": job.get("output"),
        "error": job.get("error"),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "gpu_seconds": job.get("gpu_seconds"),
        "cold_start_seconds": job.get("cold_start_seconds"),
    }


def sign_payload(body: bytes, timestamp: int) -> str:
    msg = f"{timestamp}.".encode("utf-8") + body
    return hmac.new(_webhook_secret(), msg, hashlib.sha256).hexdigest()


def verify_signature(body: bytes, timestamp: int, signature: str) -> bool:
    expected = sign_payload(body, timestamp)
    return hmac.compare_digest(expected, signature)


def _serialize_webhook_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")


def _fit_webhook_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Shrink output so the serialized webhook body stays within the byte cap."""
    safe = dict(payload)
    output = safe.get("output")
    if output is None:
        return safe

    if len(_serialize_webhook_payload(safe)) <= WEBHOOK_MAX_BODY_BYTES:
        return safe

    try:
        full_raw = json.dumps(output, separators=(",", ":"), default=str)
    except Exception:
        full_raw = str(output)

    best: dict[str, Any] = {"truncated": True, "preview": ""}
    left, right = 0, len(full_raw)
    while left <= right:
        mid = (left + right) // 2
        candidate = {"truncated": True, "preview": full_raw[:mid]}
        safe["output"] = candidate
        size = len(_serialize_webhook_payload(safe))
        if size <= WEBHOOK_MAX_BODY_BYTES:
            best = candidate
            left = mid + 1
        else:
            right = mid - 1

    safe["output"] = best
    return safe


def _post_webhook(url: str, payload: dict[str, Any]) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    safe_payload = _fit_webhook_payload(payload)
    body = _serialize_webhook_payload(safe_payload)
    ts = int(time.time())
    sig = sign_payload(body, ts)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "xcelsior-serverless-webhook/1.0",
            "X-Xcelsior-Timestamp": str(ts),
            "X-Xcelsior-Signature": f"sha256={sig}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_SEC) as resp:
            if 200 <= int(resp.status) < 300:
                return True, ""
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)[:500]


def deliver_job_webhook(repo: ServerlessRepo, job: dict) -> bool:
    """Attempt delivery; update job webhook_* columns. Returns True if delivered."""
    job_id = str(job.get("job_id") or "")
    url = validate_webhook_url(str(job.get("webhook_url") or ""))
    status = str(job.get("status") or "")
    if not job_id or status not in TERMINAL_STATUSES:
        return False
    if not url:
        if job.get("webhook_url"):
            repo.update_job_webhook_status(
                job_id,
                status="skipped",
                last_error="invalid or blocked webhook URL",
            )
        return False

    attempts = int(job.get("webhook_attempts") or 0) + 1
    ok, err = _post_webhook(url, build_webhook_payload(job))
    if ok:
        repo.update_job_webhook_status(job_id, status="delivered", attempts=attempts)
        return True

    if attempts >= WEBHOOK_MAX_ATTEMPTS:
        repo.update_job_webhook_status(
            job_id,
            status="failed",
            attempts=attempts,
            last_error=err,
        )
        log.warning("Webhook delivery failed permanently for job %s: %s", job_id, err)
        return False

    delay = WEBHOOK_BACKOFF_BASE_SEC * (2 ** (attempts - 1))
    repo.update_job_webhook_status(
        job_id,
        status="pending",
        attempts=attempts,
        last_error=err,
        next_retry_at=time.time() + delay,
    )
    log.info(
        "Webhook delivery failed for job %s (attempt %s/%s), retry in %.0fs: %s",
        job_id,
        attempts,
        WEBHOOK_MAX_ATTEMPTS,
        delay,
        err,
    )
    return False


def enqueue_job_webhook(repo: ServerlessRepo, job: dict) -> None:
    """Schedule immediate webhook attempt for a terminal job."""
    if str(job.get("status") or "") not in TERMINAL_STATUSES:
        return
    if not str(job.get("webhook_url") or "").strip():
        return
    deliver_job_webhook(repo, job)


def retry_pending_webhooks(repo: ServerlessRepo) -> int:
    """Background tick — retry due pending webhooks. Returns attempt count."""
    now = time.time()
    jobs = repo.list_jobs_pending_webhook_retry(now=now, limit=50)
    for job in jobs:
        deliver_job_webhook(repo, job)
    return len(jobs)