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
BOOT_RETRY_LIMIT = int(os.environ.get("XCELSIOR_SERVERLESS_BOOT_RETRY_LIMIT", "3"))
BOOT_RETRY_WINDOW_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_BOOT_RETRY_WINDOW_SEC", "3600"))
TERMINAL_SCHEDULER_STATUSES = frozenset({"failed", "cancelled", "completed"})
ACTIVE_WORKER_STATES = frozenset({"booting", "ready", "idle", "draining"})


def _scheduler_worker_id(job: dict) -> str:
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    return str(
        job.get("serverless_worker_id")
        or job.get("worker_id")
        or payload.get("serverless_worker_id")
        or payload.get("serverless_worker")
        or ""
    )


def _kill_scheduler_job(job: dict) -> bool:
    try:
        from scheduler import kill_job, list_hosts, update_job_status

        host_id = str(job.get("host_id") or "")
        hosts = {str(h.get("host_id") or ""): h for h in list_hosts()}
        host = hosts.get(host_id) if host_id else None
        if host:
            kill_job(job, host)
        else:
            log.warning(
                "Orphan serverless scheduler job %s has no reachable host %s; cancelling record",
                job.get("job_id"),
                host_id or "(none)",
            )
        update_job_status(str(job.get("job_id") or ""), "cancelled")
        return True
    except Exception as exc:
        log.warning("Failed to kill orphan serverless scheduler job %s: %s", job.get("job_id"), exc)
        return False


def _recent_boot_failures(svc: ServerlessService, endpoint_id: str) -> int:
    cutoff = time.time() - max(60, BOOT_RETRY_WINDOW_SEC)
    workers = sorted(
        svc.repo.list_workers(endpoint_id),
        key=lambda w: float(w.get("created_at") or 0),
        reverse=True
    )
    consecutive_failures = 0
    for w in workers:
        if float(w.get("created_at") or 0) < cutoff:
            break
        state = str(w.get("state") or "")
        if state in ("ready", "idle", "draining") or float(w.get("last_heartbeat_at") or 0) > 0:
            break
        if state in ("error", "terminated"):
            consecutive_failures += 1
    return consecutive_failures


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
    orphan_scheduler = 0
    handled: set[str] = set()

    for w in svc.repo.list_stuck_booting_workers(booting_before=booting_cutoff):
        wid = str(w["worker_id"])
        if wid in handled:
            continue
        handled.add(wid)
        endpoint_id = str(w["endpoint_id"])
        log.warning("Reaping stuck booting worker %s", wid)
        svc.repo.update_worker(wid, error_message="stuck booting (TTL exceeded)")
        svc.deprovision_worker(wid, charge=False)
        stuck_booting += 1
        ep = svc.repo.get_endpoint(endpoint_id)
        if ep and svc._endpoint_needs_workers(ep) and _recent_boot_failures(svc, endpoint_id) < BOOT_RETRY_LIMIT:
            svc.provision_worker(endpoint_id)
            reprovisioned += 1
        elif ep and svc._endpoint_needs_workers(ep):
            svc.repo.patch_endpoint(endpoint_id, str(ep["owner_id"]), {"status": "error"})

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
        err_msg = str(sched_job.get("error_message") or f"scheduler job status is {sched_job.get('status')}")
        svc.repo.update_worker(wid, error_message=err_msg[:500])
        svc.deprovision_worker(wid, charge=False)
        dead_scheduler += 1
        ep = svc.repo.get_endpoint(endpoint_id)
        if ep and svc._endpoint_needs_workers(ep) and _recent_boot_failures(svc, endpoint_id) < BOOT_RETRY_LIMIT:
            svc.provision_worker(endpoint_id)
            reprovisioned += 1
        elif ep and svc._endpoint_needs_workers(ep):
            svc.repo.patch_endpoint(endpoint_id, str(ep["owner_id"]), {"status": "error"})

    try:
        from scheduler import list_jobs

        for sched_job in list_jobs():
            if str(sched_job.get("job_type") or "") != "serverless_worker":
                continue
            if str(sched_job.get("status") or "").lower() in TERMINAL_SCHEDULER_STATUSES:
                continue
            worker_id = _scheduler_worker_id(sched_job)
            worker = svc.repo.get_worker(worker_id) if worker_id else None
            endpoint_id = str((worker or {}).get("endpoint_id") or sched_job.get("serverless_endpoint_id") or "")
            endpoint = svc.repo.get_endpoint(endpoint_id) if endpoint_id else None
            if worker and endpoint and str(worker.get("state") or "") in ACTIVE_WORKER_STATES:
                continue
            if _kill_scheduler_job(sched_job):
                orphan_scheduler += 1
            if worker:
                svc.repo.update_worker(
                    str(worker["worker_id"]),
                    state="terminated",
                    released_at=now,
                    error_message="orphan serverless scheduler job cleaned up",
                )
    except Exception as exc:
        log.warning("serverless orphan scheduler cleanup skipped: %s", exc)

    return {
        "stale_workers": stale_workers,
        "stale_queued": stale_queued,
        "timed_out_jobs": timed_out_jobs,
        "stuck_booting": stuck_booting,
        "dead_scheduler": dead_scheduler,
        "orphan_scheduler": orphan_scheduler,
        "reprovisioned": reprovisioned,
    }
