# Xcelsior — Serverless TTL reaper (stale jobs, stuck booting, dead scheduler jobs)

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless.service import ServerlessService

log = logging.getLogger("xcelsior.serverless.reaper")

JOB_HEARTBEAT_TTL_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_JOB_HEARTBEAT_TTL_SEC", "120"))
BOOTING_TTL_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_BOOTING_TTL_SEC", "900"))
QUEUE_TTL_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_QUEUE_TTL_SEC", "3600"))
TERMINAL_SCHEDULER_STATUSES = frozenset({"failed", "cancelled", "completed"})


def reap_all(svc: ServerlessService) -> dict:
    """Global reaper tick — stale workers/jobs and stuck booting workers."""
    now = time.time()
    heartbeat_cutoff = now - JOB_HEARTBEAT_TTL_SEC
    booting_cutoff = now - BOOTING_TTL_SEC

    stale_workers = 0
    for w in svc.repo.list_workers_stale_heartbeat(heartbeat_before=heartbeat_cutoff):
        wid = str(w["worker_id"])
        log.warning("Reaping stale heartbeat worker %s (state=%s)", wid, w.get("state"))
        svc.dispatcher.handle_worker_lost(wid)
        svc.deprovision_worker(wid, charge=True)
        stale_workers += 1

    stale_queued = 0
    queue_cutoff = now - QUEUE_TTL_SEC
    for job in svc.repo.list_jobs_stale_queued(queued_before=queue_cutoff):
        job_id = str(job["job_id"])
        log.warning("Failing stale queued job %s", job_id)
        svc.fail_job_without_worker(job_id, error="queue TTL exceeded")
        stale_queued += 1

    timed_out_jobs = 0
    for job in svc.repo.list_jobs_past_request_timeout(now=now):
        job_id = str(job["job_id"])
        worker_id = str(job.get("worker_id") or "")
        reason = "request timeout exceeded"
        log.warning("Failing timed-out job %s", job_id)
        if worker_id:
            svc.worker_complete_job(worker_id, job_id, error=reason)
        else:
            svc.fail_job_without_worker(job_id, error=reason)
        timed_out_jobs += 1

    stuck_booting = 0
    dead_scheduler = 0
    reprovisioned = 0
    handled: set[str] = set()

    for w in svc.repo.list_stuck_booting_workers(booting_before=booting_cutoff):
        wid = str(w["worker_id"])
        if wid in handled:
            continue
        handled.add(wid)
        endpoint_id = str(w["endpoint_id"])
        log.warning("Reaping stuck booting worker %s", wid)
        svc.deprovision_worker(wid, charge=False)
        stuck_booting += 1
        ep = svc.repo.get_endpoint(endpoint_id)
        if ep and svc._endpoint_needs_workers(ep):
            svc.provision_worker(endpoint_id)
            reprovisioned += 1

    for w in svc.repo.list_booting_workers():
        wid = str(w["worker_id"])
        if wid in handled:
            continue
        sched_id = str(w.get("scheduler_job_id") or "")
        if not sched_id:
            continue
        try:
            from scheduler import get_job

            sched_job = get_job(sched_id)
        except Exception:
            sched_job = None
        if not sched_job:
            continue
        if str(sched_job.get("status") or "") not in TERMINAL_SCHEDULER_STATUSES:
            continue
        handled.add(wid)
        endpoint_id = str(w["endpoint_id"])
        log.warning(
            "Reaping booting worker %s — scheduler job %s is %s",
            wid,
            sched_id,
            sched_job.get("status"),
        )
        svc.deprovision_worker(wid, charge=False)
        dead_scheduler += 1
        ep = svc.repo.get_endpoint(endpoint_id)
        if ep and svc._endpoint_needs_workers(ep):
            svc.provision_worker(endpoint_id)
            reprovisioned += 1

    return {
        "stale_workers": stale_workers,
        "stale_queued": stale_queued,
        "timed_out_jobs": timed_out_jobs,
        "stuck_booting": stuck_booting,
        "dead_scheduler": dead_scheduler,
        "reprovisioned": reprovisioned,
    }