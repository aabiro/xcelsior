"""Registry health probe + cached state.

Phase E/E7 — Prometheus gauges for container registry reachability.
Phase E/E8 — ``is_registry_healthy()`` guard used by the snapshot
endpoint to decide between enqueuing immediately and deferring the
snapshot into the ``queued_registry_down`` state.

Design notes
------------
- ``ghcr.io/v2/`` returns HTTP 401 without a token. We treat **any**
  HTTP response (including 401/403/404) as reachable, because it proves
  the registry's HTTP stack is up; only network-level exceptions
  (DNS, connect timeout, TLS error, refused) count as "down".
- Probe results are cached at module scope so hot paths (snapshot
  endpoint) can check health in O(1) without doing any I/O. The bg
  worker refreshes the cache every ``PROBE_INTERVAL_SEC``.
- If the env var ``XCELSIOR_REGISTRY_URL`` is unset, ``probe_registry``
  reports ``configured=False`` and the gauges are not exported; this
  is a deployment / operator error, not a transient outage, and
  ``is_registry_healthy()`` returns ``False`` so the snapshot endpoint
  falls through to its existing 503 path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger("xcelsior.registry_health")

# ── Prometheus gauges (no-op fallback if prometheus_client is absent) ──
try:
    from prometheus_client import Gauge as _Gauge  # type: ignore

    _reachable_gauge = _Gauge(
        "xcelsior_registry_reachable",
        "1 if the container registry is reachable, 0 otherwise.",
        ["registry"],
    )
    _last_probe_gauge = _Gauge(
        "xcelsior_registry_last_probe_ts_seconds",
        "Unix timestamp of the last registry health probe.",
        ["registry"],
    )
    _probe_latency_gauge = _Gauge(
        "xcelsior_registry_probe_latency_ms",
        "Duration of the last registry probe in milliseconds.",
        ["registry"],
    )
    _PROM = True
except Exception:  # pragma: no cover - prometheus_client should always be present
    class _NoopGauge:
        def labels(self, **_kw):
            return self
        def set(self, _v):
            return None
    _reachable_gauge = _NoopGauge()  # type: ignore
    _last_probe_gauge = _NoopGauge()  # type: ignore
    _probe_latency_gauge = _NoopGauge()  # type: ignore
    _PROM = False


# ── Cache ───────────────────────────────────────────────────────────────

PROBE_INTERVAL_SEC = int(os.environ.get("XCELSIOR_REGISTRY_PROBE_INTERVAL_SEC", "300"))
PROBE_TIMEOUT_SEC = float(os.environ.get("XCELSIOR_REGISTRY_PROBE_TIMEOUT_SEC", "5"))
# Health is considered stale after 3x the probe interval — long enough
# to survive a missed cycle, short enough that a dead bg_worker doesn't
# leave us with phantom "healthy" state for hours.
_STALE_AFTER_SEC = PROBE_INTERVAL_SEC * 3

_lock = threading.Lock()
_state: dict = {
    "registry": "",
    "configured": False,
    "reachable": False,
    "last_probe_at": 0.0,
    "latency_ms": 0.0,
    "status_code": None,
    "error": None,
}


def _configured_registry() -> str:
    return os.environ.get("XCELSIOR_REGISTRY_URL", "").strip().rstrip("/")


def probe_registry(url: Optional[str] = None) -> dict:
    """Perform a single blocking probe of the registry's ``/v2/`` endpoint.

    Returns a dict snapshot of the current state (also stored in the
    module-level cache + emitted as Prometheus gauges). Safe to call
    from any thread — uses a lock around the cache update.
    """
    registry = (url if url is not None else _configured_registry())
    if not registry:
        with _lock:
            _state.update(
                registry="",
                configured=False,
                reachable=False,
                last_probe_at=time.time(),
                latency_ms=0.0,
                status_code=None,
                error="registry_not_configured",
            )
            return dict(_state)

    # Normalize: if user provided a bare host ("ghcr.io") or with path,
    # always probe ``<scheme>://<host>/v2/`` which is the standardized
    # OCI distribution discovery endpoint.
    probe_url = registry
    if not probe_url.startswith(("http://", "https://")):
        probe_url = "https://" + probe_url
    # Strip any trailing path; we only want the registry root for /v2/.
    try:
        from urllib.parse import urlparse

        parsed = urlparse(probe_url)
        probe_url = f"{parsed.scheme}://{parsed.netloc}/v2/"
    except Exception:
        probe_url = probe_url.rstrip("/") + "/v2/"

    reachable = False
    status_code: Optional[int] = None
    err: Optional[str] = None
    started = time.monotonic()
    try:
        import requests  # type: ignore

        resp = requests.get(probe_url, timeout=PROBE_TIMEOUT_SEC, allow_redirects=True)
        status_code = resp.status_code
        # Any HTTP response proves the registry's HTTP stack is alive.
        # 200/401/403/404 are all fine; we only mark unhealthy on 5xx
        # because that indicates a dead backend behind a live proxy.
        if 200 <= resp.status_code < 500:
            reachable = True
        else:
            err = f"http_{resp.status_code}"
    except Exception as e:
        err = type(e).__name__
    latency_ms = (time.monotonic() - started) * 1000.0
    now = time.time()

    with _lock:
        _state.update(
            registry=registry,
            configured=True,
            reachable=reachable,
            last_probe_at=now,
            latency_ms=latency_ms,
            status_code=status_code,
            error=err,
        )
        snapshot = dict(_state)

    try:
        _reachable_gauge.labels(registry=registry).set(1 if reachable else 0)
        _last_probe_gauge.labels(registry=registry).set(now)
        _probe_latency_gauge.labels(registry=registry).set(latency_ms)
    except Exception:  # pragma: no cover
        pass

    if reachable:
        log.info(
            "registry probe OK registry=%s status=%s latency_ms=%.1f",
            registry, status_code, latency_ms,
        )
    else:
        log.warning(
            "registry probe FAIL registry=%s err=%s status=%s latency_ms=%.1f",
            registry, err, status_code, latency_ms,
        )
    return snapshot


def is_registry_healthy() -> bool:
    """Fast, non-blocking check of the cached health state.

    Returns ``False`` if:
    - The registry is not configured.
    - The most recent probe reported unreachable.
    - The cache is stale (bg probe task stopped running).
    - No probe has ever run (cold start before bg_worker first tick).

    Callers that want up-to-the-moment truth should call
    ``probe_registry()`` directly; that blocks on network I/O.
    """
    with _lock:
        if not _state["configured"] or not _state["reachable"]:
            return False
        age = time.time() - _state["last_probe_at"]
        return age < _STALE_AFTER_SEC


def snapshot() -> dict:
    """Return a copy of the current cached state (for /health payloads)."""
    with _lock:
        return dict(_state)
