# Xcelsior — Serverless job dispatcher (queue → worker assignment)

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from serverless.repo import (
    JOB_STATUS_IN_PROGRESS,
    WORKER_STATE_ERROR,
    WORKER_STATE_TERMINATED,
)

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.dispatcher")


class ServerlessDispatcher:
    def __init__(self, repo: ServerlessRepo | None = None):
        from serverless.repo import ServerlessRepo as _Repo

        self.repo = repo or _Repo()

    def dispatch_for_endpoint(self, endpoint: dict) -> dict | None:
        """
        Assign the oldest queued job to a ready worker with free concurrency.
        Returns claimed job dict or None.
        """
        endpoint_id = str(endpoint["endpoint_id"])
        max_conc = int(endpoint.get("max_concurrency") or 1)
        if self.repo.queue_depth(endpoint_id) == 0:
            return None

        from serverless.prefix_routing import prefix_routing_enabled, select_worker_with_prefix

        worker = None
        next_job = self.repo.peek_next_job(endpoint_id) if hasattr(self.repo, "peek_next_job") else None
        workers = self.repo.list_workers(endpoint_id)
        ready = [
            w
            for w in workers
            if str(w.get("state") or "") in ("ready", "idle")
            and int(w.get("current_concurrency") or 0) < max_conc
        ]
        if prefix_routing_enabled() and next_job and ready:
            worker = select_worker_with_prefix(
                self.repo,
                endpoint_id,
                ready,
                dict(next_job.get("payload") or {}),
            )
        if not worker:
            worker = self.repo.find_ready_worker_with_capacity(endpoint_id, max_conc)
        if not worker:
            return None

        from routes._deps import otel_span

        with otel_span(
            "serverless.job.dispatch",
            {"endpoint_id": endpoint_id, "worker_id": str(worker["worker_id"])},
        ):
            claimed = self.repo.claim_next_job(endpoint_id, str(worker["worker_id"]))
        if not claimed:
            return None

        self.repo.increment_worker_concurrency(str(worker["worker_id"]))
        log.info(
            "Dispatched job %s → worker %s (endpoint %s)",
            claimed["job_id"],
            worker["worker_id"],
            endpoint_id,
        )
        return claimed

    def release_job(self, job_id: str, worker_id: str) -> None:
        """Job finished — decrement worker concurrency."""
        job = self.repo.get_job(job_id)
        if job and str(job.get("worker_id")) == worker_id:
            self.repo.decrement_worker_concurrency(worker_id)

    def handle_worker_lost(self, worker_id: str) -> int:
        """Requeue in-flight jobs when a worker crashes or is reaped."""
        requeued = self.repo.requeue_worker_jobs(worker_id)
        self.repo.update_worker(
            worker_id,
            state=WORKER_STATE_ERROR,
            released_at=time.time(),
        )
        if requeued:
            log.warning("Requeued %d jobs from lost worker %s", requeued, worker_id)
        return requeued

    def terminate_worker(self, worker_id: str) -> int:
        requeued = self.handle_worker_lost(worker_id)
        self.repo.update_worker(worker_id, state=WORKER_STATE_TERMINATED)
        return requeued