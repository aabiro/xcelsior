"""Queue-mode worker runtime — poll, execute handler, report completion."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Any

from serverless.worker_sdk.client import WorkerClient
from serverless.worker_sdk.errors import WorkerError, error_envelope
from serverless.worker_sdk.fitness import FitnessConfig, run_fitness_checks
from serverless.worker_sdk.handler import get_registered_handler, invoke_handler
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


def run_worker(
    *,
    client: WorkerClient | None = None,
    handler=None,
    fitness: FitnessConfig | None = None,
    poll_interval_sec: float = 1.0,
    heartbeat_interval_sec: float = 15.0,
    health_port: int | None = None,
    skip_fitness: bool = False,
) -> None:
    """
    Boot a queue-mode worker:
    1. Fitness checks
    2. /healthz server
    3. Signal ready to control plane
    4. Poll loop: heartbeat → claim → execute → complete
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

    last_heartbeat = 0.0
    while not _draining:
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval_sec:
            try:
                worker_client.heartbeat()
                last_heartbeat = now
            except Exception as e:
                log.warning("Heartbeat failed: %s", e)

        try:
            job = worker_client.claim_job()
        except Exception as e:
            log.warning("Claim failed: %s", e)
            time.sleep(poll_interval_sec)
            continue

        if not job:
            time.sleep(poll_interval_sec)
            continue

        job_id = str(job.get("job_id") or "")
        log.info("Executing job %s", job_id)
        if worker_client.is_job_cancelled(job_id):
            log.info("Job %s already cancelled — skipping execution", job_id)
            continue
        try:
            output, events = asyncio.run(invoke_handler(fn, job))
            if worker_client.is_job_cancelled(job_id):
                log.info("Job %s cancelled during execution", job_id)
                continue
            for ev in events:
                worker_client.append_event(job_id, str(ev.get("type") or "progress"), ev)
            worker_client.complete_job(job_id, output=output or {})
        except WorkerError as e:
            worker_client.append_event(job_id, "error", error_envelope(e))
            worker_client.complete_job(job_id, error=e.message)
        except Exception as e:
            log.exception("Job %s failed", job_id)
            worker_client.complete_job(job_id, error=str(e))

    log.info("Drain complete — exiting")