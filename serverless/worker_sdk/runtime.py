"""Queue-mode worker runtime — poll, execute handler, report completion."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from serverless.worker_sdk.client import WorkerClient
from serverless.worker_sdk.errors import WorkerError, error_envelope
from serverless.worker_sdk.fitness import FitnessConfig, run_fitness_checks
from serverless.worker_sdk.handler import CancelToken, get_registered_handler, invoke_handler
from serverless.worker_sdk.health import start_health_server

log = logging.getLogger("xcelsior.serverless.worker_sdk.runtime")

_draining = False


def request_drain() -> None:
    global _draining
    _draining = True


def is_draining() -> bool:
    return _draining


def _install_signal_handlers() -> None:
    def _on_signal(signum, _frame):
        log.info("Received signal %s — draining", signum)
        request_drain()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


class _HeartbeatThread(threading.Thread):
    """Background heartbeat so long jobs never starve the control-plane TTL."""

    def __init__(self, client: WorkerClient, interval_sec: float):
        super().__init__(daemon=True, name="worker-heartbeat")
        self._client = client
        self._interval = interval_sec
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._client.heartbeat()
            except Exception as e:
                log.warning("Heartbeat failed: %s", e)

    def stop(self) -> None:
        self._stop.set()


def _execute_job_sync(
    *,
    client: WorkerClient,
    fn,
    job: dict[str, Any],
    cancel: CancelToken,
) -> None:
    job_id = str(job.get("job_id") or "")
    started = time.time()
    correlation_id = str(job.get("correlation_id") or uuid.uuid4())
    log.info(
        "job_start job_id=%s correlation_id=%s",
        job_id,
        correlation_id,
        extra={"job_id": job_id, "correlation_id": correlation_id, "event": "job_start"},
    )
    try:
        if cancel.is_cancelled or client.is_job_cancelled(job_id):
            log.info("Job %s cancelled before execution", job_id)
            return
        output, events, usage = asyncio.run(invoke_handler(fn, job, cancel=cancel))
        if cancel.is_cancelled:
            log.info("Job %s cancelled during execution", job_id)
            return
        for ev in events:
            client.append_event(job_id, str(ev.get("type") or "progress"), ev)
        client.complete_job(
            job_id,
            output=output or {},
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cached_tokens=int(usage.get("cached_tokens") or 0),
            ttft_ms=int(usage.get("ttft_ms") or 0),
        )
        elapsed_ms = int((time.time() - started) * 1000)
        log.info(
            "job_complete job_id=%s elapsed_ms=%d input_tokens=%d output_tokens=%d",
            job_id,
            elapsed_ms,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            extra={"job_id": job_id, "correlation_id": correlation_id, "elapsed_ms": elapsed_ms},
        )
    except WorkerError as e:
        client.append_event(job_id, "error", error_envelope(e))
        client.complete_job(job_id, error=e.message)
    except Exception as e:
        log.exception("Job %s failed", job_id)
        client.complete_job(job_id, error=str(e))


def run_worker(
    *,
    client: WorkerClient | None = None,
    handler=None,
    fitness: FitnessConfig | None = None,
    poll_interval_sec: float = 1.0,
    heartbeat_interval_sec: float = 15.0,
    health_port: int | None = None,
    skip_fitness: bool = False,
    max_concurrency: int | None = None,
) -> None:
    """
    Boot a queue-mode worker:
    1. Fitness checks
    2. /healthz server
    3. Signal ready to control plane
    4. Poll loop: claim up to free slots → execute concurrently → complete
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _install_signal_handlers()

    worker_client = client or WorkerClient()
    fn = handler or get_registered_handler()
    if fn is None:
        log.error("No handler registered — use @handler or pass handler= to run_worker()")
        sys.exit(1)

    if not skip_fitness:
        failures = run_fitness_checks(fitness)
        if failures:
            for reason in failures:
                log.error("Fitness check failed: %s", reason)
            sys.exit(1)

    port = health_port or int(os.environ.get("XCELSIOR_HTTP_PORT", "8080"))
    start_health_server(port, checker=lambda: {"ok": not _draining, "draining": _draining})

    host_id = os.environ.get("XCELSIOR_HOST_ID", "")
    worker_client.signal_ready(host_id=host_id)
    log.info("Worker %s ready", worker_client.worker_id)

    concurrency = max(1, max_concurrency or int(os.environ.get("XCELSIOR_MAX_CONCURRENCY", "1")))

    heartbeat = _HeartbeatThread(worker_client, heartbeat_interval_sec)
    heartbeat.start()

    inflight: dict[str, Future[None]] = {}

    try:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="worker-job") as pool:
            while not _draining:
                done_ids = [jid for jid, fut in inflight.items() if fut.done()]
                for jid in done_ids:
                    inflight.pop(jid, None)

                free_slots = concurrency - len(inflight)
                if free_slots > 0:
                    for _ in range(free_slots):
                        try:
                            job = worker_client.claim_job()
                        except Exception as e:
                            log.warning("Claim failed: %s", e)
                            break
                        if not job:
                            break
                        job_id = str(job.get("job_id") or "")
                        if not job_id:
                            continue
                        cancelled = bool(job.get("cancelled")) or str(job.get("status") or "") == "CANCELLED"
                        cancel = CancelToken(cancelled=cancelled)
                        fut = pool.submit(
                            _execute_job_sync,
                            client=worker_client,
                            fn=fn,
                            job=job,
                            cancel=cancel,
                        )
                        inflight[job_id] = fut
                        log.info("Claimed job %s (%d/%d slots)", job_id, len(inflight), concurrency)

                if not inflight:
                    time.sleep(poll_interval_sec)
                else:
                    time.sleep(min(poll_interval_sec, 0.25))

            if inflight:
                log.info("Draining %d active job(s)...", len(inflight))
                for fut in inflight.values():
                    fut.result(timeout=None)
    finally:
        heartbeat.stop()
        worker_client.close()

    log.info("Drain complete — exiting")