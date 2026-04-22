"""Routes: instances."""

import asyncio
import hmac
import json
import os
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator

from routes._deps import (
    AUTH_REQUIRED,
    _AUTH_COOKIE_NAME,
    _USE_PERSISTENT_AUTH,
    _api_keys,
    _check_ws_connect_rate_limit,
    _consume_ws_ticket,
    _issue_ws_ticket,
    _require_auth,
    _require_scope,
    _sessions,
    _sse_lock,
    _sse_subscribers,
    _user_lock,
    _validate_ws_origin,
    _validate_ws_auth,  # re-exported for backward compat with tests
    broadcast_sse,
    log,
    otel_span,
)
from scheduler import (
    API_TOKEN,
    failover_and_reassign,
    get_job,
    kill_job,
    list_hosts,
    list_jobs,
    list_tiers,
    log,
    process_queue,
    process_queue_binpack,
    requeue_job,
    run_job,
    submit_job,
    update_job_status,
)
from db import UserStore
from collections import defaultdict

# Maximum concurrent active (queued/assigned/starting/running) instances per user
# if the user has no explicit `users.max_concurrent_instances` override.
MAX_CONCURRENT_INSTANCES_PER_USER = int(os.environ.get("MAX_CONCURRENT_INSTANCES", "5"))

_ACTIVE_STATUSES = {"queued", "assigned", "starting", "running"}


def _canonical_owner_id(user: dict) -> str:
    """P3/B3 — single source of truth for per-user ownership strings.

    Prefers the Stripe ``customer_id`` (post-billing identity) but falls
    back to ``user_id`` for accounts that never finished Stripe signup
    (customer_id is "" or missing). Returns "" if neither is available,
    which callers must treat as 401.

    Historically this repo had two patterns — a positional default
    ``user.get("customer_id", user.get("user_id", ""))`` and a
    short-circuit ``user.get("customer_id") or user.get("user_id")`` —
    which diverge when ``customer_id`` is the empty string. The helper
    uses the short-circuit form because customer_id="" is NOT a valid
    Stripe identity and ownership checks must fall back to user_id.
    """
    cid = (user.get("customer_id") or "").strip()
    if cid:
        return cid
    return (user.get("user_id") or "").strip()


# ---------------------------------------------------------------------------
# P3/B4 — per-user snapshot rate limit (sliding window, in-memory).
# Defends against a runaway client or compromised token flooding the
# worker agent with `docker commit` jobs (which are CPU/IO heavy and can
# fill disk). In-memory state is fine for a single-process API; if the
# deployment ever shards across multiple API replicas, move to Redis.
# ---------------------------------------------------------------------------
from collections import deque as _deque  # noqa: E402

_SNAPSHOT_RATE_LIMIT = int(os.environ.get("XCELSIOR_SNAPSHOT_RATE_LIMIT", "5"))
_SNAPSHOT_RATE_WINDOW_SEC = int(
    os.environ.get("XCELSIOR_SNAPSHOT_RATE_WINDOW_SEC", "3600")
)
_SNAPSHOT_RATE_BUCKETS: dict[str, "_deque[float]"] = {}


def _check_snapshot_rate_limit(owner_id: str) -> None:
    """Raise 429 if ``owner_id`` exceeds the configured snapshot quota.

    Default: 5 snapshots per rolling 60 minutes per user. Bypass entirely
    by setting ``XCELSIOR_SNAPSHOT_RATE_LIMIT=0`` (useful for tests and
    internal admin tooling).
    """
    if _SNAPSHOT_RATE_LIMIT <= 0:
        return
    now = time.time()
    bucket = _SNAPSHOT_RATE_BUCKETS.setdefault(owner_id, _deque())
    while bucket and bucket[0] <= now - _SNAPSHOT_RATE_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= _SNAPSHOT_RATE_LIMIT:
        retry_in = int(_SNAPSHOT_RATE_WINDOW_SEC - (now - bucket[0]))
        raise HTTPException(
            429,
            f"Snapshot rate limit exceeded: max {_SNAPSHOT_RATE_LIMIT} per "
            f"{_SNAPSHOT_RATE_WINDOW_SEC // 60} min. Retry in ~{retry_in}s.",
        )
    bucket.append(now)



def _get_user_concurrency_cap(customer_id: str) -> int:
    """Return the active-instance cap for a user.

    Hierarchy (first match wins):
      1. `users.max_concurrent_instances` column if non-NULL — per-user override
         set by admins (e.g. for power users or trial restrictions).
      2. `MAX_CONCURRENT_INSTANCES` env default (5 if unset).

    Future extension: tier-based defaults can slot in between 1 and 2 without
    changing callers. Failures to look up the user fall through to the env
    default — we never block job submission on a metadata query glitch.
    """
    if not customer_id:
        return MAX_CONCURRENT_INSTANCES_PER_USER
    try:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT max_concurrent_instances FROM users WHERE customer_id = %s LIMIT 1",
                (customer_id,),
            )
            row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except Exception as e:
        log.debug("concurrency cap lookup failed for %s: %s", customer_id, e)
    return MAX_CONCURRENT_INSTANCES_PER_USER


def _count_active_instances(customer_id: str) -> int:
    """Count a user's active (non-terminal) instances via a single SQL query.

    Uses `payload->>'owner'` because the jobs table stores owner inside the
    JSONB payload (thin-schema convention — see repo memory). `status = ANY(...)`
    lets PostgreSQL use the status index for the set membership.
    """
    try:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM jobs " "WHERE payload->>'owner' = %s AND status = ANY(%s)",
                (customer_id, list(_ACTIVE_STATUSES)),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as e:
        # If the SQL path breaks, fall back to the Python filter so we never
        # completely lose the gate. (Worst case: O(N) on all jobs once.)
        log.warning("SQL active-instance count failed for %s: %s — falling back", customer_id, e)
        return sum(
            1
            for j in list_jobs()
            if j.get("owner") == customer_id and j.get("status") in _ACTIVE_STATUSES
        )


from routes.agent import _agent_lock, _agent_preempt

router = APIRouter()

# Job log and WebSocket state
_JOB_LOG_MAX = 500
_job_log_buffers: dict[str, list[dict]] = defaultdict(list)
_ws_connections: dict[str, set] = defaultdict(set)


def _check_job_access(user: dict, job_id: str):
    """Verify user owns the job or is admin."""
    if user.get("role") == "admin" or user.get("is_admin"):
        return
    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Instance {job_id} not found")
    job_owner = job.get("owner", "")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    if job_owner and job_owner != customer_id:
        raise HTTPException(403, "Not authorized to access this instance")


class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    host_id: str | None = None
    gpu_model: str | None = None
    nfs_server: str | None = None
    nfs_path: str | None = None
    nfs_mount_point: str | None = None
    image: str | None = None
    interactive: bool = True
    command: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)
    max_bid: float | None = Field(default=None, gt=0)
    volume_ids: list[str] | None = Field(default=None, max_length=16)
    encrypted_workspace: bool = False
    # P2.1 — optional provisioning hooks, all run inside the container with a
    # hard 15 s cap so interactive boot never blocks on them.
    init_script: str | None = Field(default=None, max_length=4096)
    git_repo: str | None = Field(default=None, max_length=512)
    auto_launch: list[str] | None = Field(default=None, max_length=4)
    exposed_ports: list[int] | None = Field(default=None, max_length=8)

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return v
        from security import validate_docker_image

        return validate_docker_image(v)

    @field_validator("init_script")
    @classmethod
    def validate_init_script(cls, v: str | None) -> str | None:
        if not v:
            return v
        # Reject control chars (except \t \n \r) — keeps the script printable
        # so logs don't break tty and prevents smuggling binary blobs.
        bad = {c for c in v if ord(c) < 32 and c not in "\t\n\r"}
        if bad:
            raise ValueError("init_script contains disallowed control characters")
        return v

    @field_validator("git_repo")
    @classmethod
    def validate_git_repo(cls, v: str | None) -> str | None:
        if not v:
            return v
        # Only public https clones — no ssh://, git://, file://, or embedded creds.
        import re

        if not re.fullmatch(
            r"https://[A-Za-z0-9._~-]+(?::\d+)?/[A-Za-z0-9._~\-/]+(?:\.git)?", v
        ):
            raise ValueError("git_repo must be a plain https:// URL (no creds)")
        if "@" in v.split("://", 1)[1]:
            raise ValueError("git_repo must not contain credentials")
        return v

    @field_validator("auto_launch")
    @classmethod
    def validate_auto_launch(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        allowed = {"jupyter", "vscode"}
        out: list[str] = []
        for item in v:
            s = str(item or "").strip().lower()
            if s and s not in allowed:
                raise ValueError(f"auto_launch item must be one of {sorted(allowed)}")
            if s:
                out.append(s)
        # Dedup, preserve order
        seen: set[str] = set()
        dedup: list[str] = []
        for s in out:
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        return dedup

    @field_validator("exposed_ports")
    @classmethod
    def validate_exposed_ports(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        out: list[int] = []
        seen: set[int] = set()
        for p in v:
            try:
                port = int(p)
            except (TypeError, ValueError):
                raise ValueError("exposed_ports must contain integers")
            if port < 1 or port > 65535:
                raise ValueError("exposed_ports entries must be 1..65535")
            # Reserve SSH for the platform; users route via the fixed ssh_port.
            if port == 22:
                raise ValueError("port 22 is reserved — use ssh_port")
            if port in seen:
                continue
            seen.add(port)
            out.append(port)
        return out


class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None
    container_id: str | None = None
    container_name: str | None = None
    ssh_port: int | None = None
    interactive: bool | None = None
    error_message: str | None = None


def _wallet_preflight(customer_id: str):
    """Shared wallet check — raises 402 if wallet is suspended or empty."""
    from billing import get_billing_engine

    be = get_billing_engine()
    wallet = be.get_wallet(customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(402, detail="Wallet suspended — please add funds to resume service")
    if wallet["balance_cad"] <= 0 and wallet.get("grace_until", 0) < time.time():
        raise HTTPException(402, detail="Insufficient wallet balance — please deposit credits")


# ── Image Templates ──


class ImageTemplate(BaseModel):
    id: str
    label: str
    image: str
    default_vram_gb: int
    icon: str
    category: str
    description: str


@router.get("/api/images/templates", tags=["Images"])
def api_image_templates():
    """Return the authoritative list of validated container image templates.

    Single source of truth consumed by frontend, wizard, and API callers.
    """
    from security import get_image_templates

    return {"templates": [ImageTemplate(**t).model_dump() for t in get_image_templates()]}


# ── Helper: _refresh_job ──


def _refresh_job(job_id: str):
    """Re-read a job from the DB to get updated status/host/container fields."""
    for j in list_jobs():
        if j["job_id"] == job_id:
            return j
    return None


@router.post("/instance", tags=["Instances"])
def api_submit_instance(j: JobIn, request: Request):
    """Submit a job to the queue or directly assign to a host.

    If host_id is provided (marketplace launch), the job is assigned directly
    to that host and container start is attempted immediately. Otherwise,
    the job is queued and process_queue runs to find a host.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")

    vram_needed = 0.0
    target_host_id = j.host_id
    target_host = None
    if target_host_id:
        all_hosts = list_hosts(active_only=False)
        hmap_all = {h["host_id"]: h for h in all_hosts}
        target_host = hmap_all.get(target_host_id)
        if not target_host and len(target_host_id) >= 8:
            # Tolerate truncated host IDs (e.g. from display-truncated UI values)
            matches = [h for h in all_hosts if h["host_id"].startswith(target_host_id)]
            if len(matches) == 1:
                target_host = matches[0]
                target_host_id = target_host["host_id"]
                j.host_id = target_host_id
        if not target_host:
            raise HTTPException(status_code=404, detail=f"Host {target_host_id} not found")
        if target_host.get("status") == "draining":
            raise HTTPException(
                status_code=409,
                detail=f"Host {target_host_id} is draining for maintenance and not accepting new instances",
            )
        if target_host.get("status") != "active":
            raise HTTPException(
                status_code=409,
                detail=f"Host {target_host_id} is {target_host.get('status', 'unavailable')} and not accepting new instances",
            )

    # Respect the requested VRAM for scheduler matching and host selection.
    # A zero value remains valid and means "no minimum VRAM preference".
    # Interactive instances get exclusive GPU access (one per host), so force
    # vram_needed=0 to skip fractional VRAM reservation logic which only applies
    # to HPC batch jobs sharing a GPU.
    if j.interactive:
        vram_needed = 0.0
    else:
        vram_needed = max(float(j.vram_needed_gb or 0.0), 0.0)

    # ── Marketplace flow requires a Docker image ──────────────────────
    if target_host_id and not j.image:
        raise HTTPException(
            status_code=422,
            detail="Docker image is required for marketplace launches — select a template or enter a custom image",
        )

    # ── Pre-validate the image exists on the registry ─────────────────
    # Catches invalid tags like `nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04`
    # (the Ubuntu 20.04 CUDA 11.8 cudnn8 variant was never published) BEFORE
    # queueing the job, instead of letting it reach the host and fail on pull.
    # Fail-open on network errors to avoid blocking submissions if Docker Hub
    # or the auth service is having a blip.
    if j.image:
        from security import probe_image_exists

        problem = probe_image_exists(j.image)
        if problem:
            raise HTTPException(status_code=400, detail=problem)

    with otel_span(
        "job.submit", {"job.name": j.name, "job.tier": j.tier or "", "job.num_gpus": j.num_gpus}
    ):
        customer_id = user.get("customer_id", user.get("user_id", ""))
        _wallet_preflight(customer_id)

        # ── Per-user concurrent instance cap ───────────────────
        # Uses single SQL COUNT on payload->>'owner' (see _count_active_instances)
        # instead of an O(N) Python filter over every job in the system.
        user_cap = _get_user_concurrency_cap(customer_id)
        active_count = _count_active_instances(customer_id)
        if active_count >= user_cap:
            raise HTTPException(
                status_code=429,
                detail=f"Concurrent instance limit reached ({user_cap}). "
                "Please stop an existing instance before launching a new one.",
            )

        # ── Validate volume_ids ownership and status ─────────────
        validated_volume_ids = None
        if j.volume_ids:
            from volumes import get_volume_engine

            ve = get_volume_engine()
            validated_volume_ids = []
            seen_vids: set[str] = set()
            # Volumes use user_id as owner_id (not customer_id)
            volume_owner_id = user.get("user_id", user.get("email", ""))
            for vid in j.volume_ids:
                if vid in seen_vids:
                    continue  # deduplicate
                seen_vids.add(vid)
                try:
                    vol = ve.get_volume(vid)
                except Exception:
                    raise HTTPException(status_code=404, detail=f"Volume {vid} not found")
                if not vol:
                    raise HTTPException(status_code=404, detail=f"Volume {vid} not found")
                if vol.get("owner_id") != volume_owner_id:
                    raise HTTPException(status_code=404, detail=f"Volume {vid} not found")
                if vol.get("status") not in ("available", "attached"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Volume {vid} is {vol.get('status', 'unknown')} and cannot be attached",
                    )
                validated_volume_ids.append(vid)

        # Spot path: max_bid present → delegate to spot submission
        if j.max_bid is not None:
            from scheduler import submit_spot_job

            job = submit_spot_job(
                j.name,
                vram_needed,
                j.max_bid,
                j.priority,
                tier=j.tier,
                owner=customer_id,
                image=j.image,
                gpu_model=j.gpu_model,
            )
            event_name = "spot_job_submitted"
        else:
            job = submit_job(
                j.name,
                vram_needed,
                j.priority,
                tier=j.tier,
                num_gpus=j.num_gpus,
                gpu_model=j.gpu_model,
                nfs_server=j.nfs_server,
                nfs_path=j.nfs_path,
                nfs_mount_point=j.nfs_mount_point,
                image=j.image,
                interactive=j.interactive,
                command=j.command,
                ssh_port=j.ssh_port,
                owner=customer_id,
                volume_ids=validated_volume_ids,
                encrypted_workspace=j.encrypted_workspace,
                init_script=j.init_script,
                git_repo=j.git_repo,
                auto_launch=j.auto_launch,
                exposed_ports=j.exposed_ports,
            )
            event_name = "job_submitted"

        # Track job ownership (response-only, already persisted via owner param)
        job["submitted_by"] = user.get("email", "")
        job["customer_id"] = customer_id
        broadcast_sse(event_name, {"job_id": job["job_id"], "name": job["name"]})

        # Lifecycle log — record the queued event so users and admin audit can
        # see the job entered the system even if it never advances further.
        try:
            tier_label = j.tier or "any"
            gpu_label = j.gpu_model or "any"
            push_job_log(
                job["job_id"],
                f"Queued — waiting for GPU matching tier={tier_label}, gpu_model={gpu_label}",
                level="info",
            )
        except Exception:
            pass

        # Direct host assignment: assign + start immediately
        if target_host_id:
            try:
                updated = update_job_status(job["job_id"], "assigned", host_id=target_host_id)
                if updated and updated.get("status") == "assigned":
                    hosts = list_hosts()
                    hmap = {h["host_id"]: h for h in hosts}
                    host = hmap.get(target_host_id)
                    if host:
                        container_id = run_job(updated, host, docker_image=j.image or None)
                        if container_id:
                            job = _refresh_job(job["job_id"]) or job
                            log.info(
                                "Direct launch: job %s running on host %s",
                                job["job_id"],
                                target_host_id,
                            )
                        else:
                            job = _refresh_job(job["job_id"]) or job
                            log.warning(
                                "Direct launch: container start failed for job %s on host %s",
                                job["job_id"],
                                target_host_id,
                            )
                    else:
                        log.warning(
                            "Direct launch: host %s not found, job %s stays queued",
                            target_host_id,
                            job["job_id"],
                        )
                        update_job_status(job["job_id"], "queued")
                        job = _refresh_job(job["job_id"]) or job
            except Exception as e:
                log.error("Direct launch failed for job %s: %s", job["job_id"], e)
                job = _refresh_job(job["job_id"]) or job
        else:
            # Auto-process queue to try to assign immediately
            try:
                process_queue()
                # Refresh job status after queue processing
                job = _refresh_job(job["job_id"]) or job
            except Exception as e:
                log.warning("Queue processing after submit failed: %s", e)

        return {"ok": True, "instance": job}


def _enrich_instance(j: dict, host_map: dict[str, dict]) -> dict:
    """Shared enrichment for every instance response — list and detail.

    Adds docker_image mapping, host GPU info, elapsed/duration, connection
    details, and cost computation.  Single source of truth — no drift.
    """
    # Map internal 'image' → frontend 'docker_image'
    if j.get("image") is not None and not j.get("docker_image"):
        j["docker_image"] = j["image"]

    # Host GPU info — prefer the host_gpu_model stored at assignment time,
    # fall back to live host lookup
    hid = j.get("host_id")
    host = host_map.get(hid) if hid else None
    actual_gpu = j.get("host_gpu_model") or (host.get("gpu_model", "") if host else "")
    if actual_gpu:
        j["gpu_type"] = actual_gpu
        j["host_gpu"] = actual_gpu
        if not j.get("gpu_model"):
            j["gpu_model"] = actual_gpu

    # Elapsed / duration
    started = float(j.get("started_at") or 0)
    completed = float(j.get("completed_at") or 0)
    if started > 0:
        if completed > started:
            j.setdefault("duration_sec", round(completed - started, 2))
            j.setdefault("elapsed_sec", j["duration_sec"])
        elif j.get("status") == "running":
            j.setdefault("elapsed_sec", round(time.time() - started, 2))

    # Connection details (host IP, VRAM) — available once job has run
    if host and j.get("status") in ("running", "starting", "completed", "failed"):
        j.setdefault("host_ip", host.get("ip", ""))
        j.setdefault("host_gpu", host.get("gpu_model", ""))
        j.setdefault("host_vram_gb", host.get("total_vram_gb", 0))
        j.setdefault("gpu_type", host.get("gpu_model", ""))

    # Cost
    elapsed = j.get("elapsed_sec") or j.get("duration_sec") or 0
    if elapsed > 0 and j.get("cost_cad") is None:
        rate = 0.20  # default CAD/hr
        if host and host.get("cost_per_hour"):
            rate = float(host["cost_per_hour"])
        j["cost_cad"] = round((elapsed / 3600) * rate, 4)

    # Encrypted workspace flag
    j.setdefault("encrypted_workspace", bool(j.get("encrypted_workspace")))

    # Attached volumes
    try:
        from volumes import get_volume_engine

        ve = get_volume_engine()
        vols = ve.get_instance_volumes(j.get("job_id", ""))
        j["attached_volumes"] = [
            {
                "volume_id": v["volume_id"],
                "name": v.get("name", ""),
                "size_gb": v.get("size_gb", 0),
                "mount_path": v.get("mount_path", "/workspace"),
                "mode": v.get("mode", "rw"),
                "storage_type": v.get("storage_type", "nfs"),
                "encrypted": v.get("encrypted", False),
            }
            for v in vols
        ]
    except Exception:
        j.setdefault("attached_volumes", [])

    # Storage cost — sum billing_cycles for attached volumes
    volume_ids = j.get("volume_ids", [])
    if volume_ids:
        try:
            from db import _get_pg_pool
            from psycopg.rows import dict_row

            pool = _get_pg_pool()
            with pool.connection() as conn:
                conn.row_factory = dict_row
                row = conn.execute(
                    """SELECT COALESCE(SUM(amount_cad), 0) AS total
                       FROM billing_cycles
                       WHERE job_id = ANY(%s) AND gpu_model = 'storage' AND tier = 'volume'""",
                    (volume_ids,),
                ).fetchone()
            j["storage_cost_cad"] = round(float(row["total"]), 4) if row else 0
        except Exception:
            j.setdefault("storage_cost_cad", 0)
    else:
        j.setdefault("storage_cost_cad", 0)

    return j


@router.get("/instances", tags=["Instances"])
def api_list_instances(status: str | None = None):
    """List jobs. Optional filter by status."""
    jobs = list_jobs(status=status)
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
    for j in jobs:
        _enrich_instance(j, host_map)
    return {"instances": jobs}


@router.get("/instance/{job_id}", tags=["Instances"])
def api_get_instance(job_id: str):
    """Get a specific instance by ID, enriched with connection info."""
    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
    _enrich_instance(j, host_map)
    return {"instance": j}


@router.patch("/instance/{job_id}", tags=["Instances"])
def api_update_instance(job_id: str, update: StatusUpdate):
    """Update a job's status."""
    with otel_span("job.status_update", {"job.id": job_id, "job.status": update.status}):
        try:
            update_job_status(job_id, update.status, host_id=update.host_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Store container info if provided by worker agent
        extras = {}
        if update.container_id:
            extras["container_id"] = update.container_id
        if update.container_name:
            extras["container_name"] = update.container_name
        if update.ssh_port is not None:
            # Gateway-mapped ports only. Container-internal ports (22, etc.)
            # are never valid here — the worker agent must report the mapped
            # host-side port in the 10000-65000 range.
            if not (10000 <= update.ssh_port <= 65000):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"ssh_port {update.ssh_port} outside reserved gateway " "range 10000-65000"
                    ),
                )
            extras["ssh_port"] = update.ssh_port
        if update.interactive is not None:
            extras["interactive"] = update.interactive
        if update.error_message:
            extras["error_message"] = update.error_message[:500]
        if extras:
            from scheduler import _set_job_fields

            _set_job_fields(job_id, **extras)
        # Push error_message as a log line so it appears in the UI log viewer
        if update.error_message and update.status == "failed":
            push_job_log(job_id, f"Instance failed: {update.error_message[:500]}", "error")
        broadcast_sse("job_status", {"job_id": job_id, "status": update.status})
        return {"ok": True, "job_id": job_id, "status": update.status}


class InstanceRenamePayload(BaseModel):
    name: str = Field(min_length=1, max_length=128)


@router.patch("/instance/{job_id}/name", tags=["Instances"])
def api_rename_instance(job_id: str, body: InstanceRenamePayload, request: Request):
    """Rename an instance (owner or admin only)."""
    from routes._deps import _get_current_user, _require_auth

    _require_auth(request)
    user = _get_current_user(request)
    from scheduler import get_job, _set_job_fields

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Instance not found")
    is_admin = user.get("role") == "admin" or bool(user.get("is_admin"))
    owner = job.get("owner", "")
    caller = user.get("customer_id", user.get("user_id", ""))
    if not is_admin and owner != caller:
        raise HTTPException(403, "Not authorized")
    _set_job_fields(job_id, name=body.name.strip())
    broadcast_sse("job_update", {"job_id": job_id, "name": body.name.strip()})
    return {"ok": True, "job_id": job_id, "name": body.name.strip()}


@router.post("/queue/process", tags=["Instances"])
def api_process_queue():
    """Process the job queue — assign jobs to hosts."""
    assigned = process_queue()
    result = [{"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned]
    if result:
        broadcast_sse("queue_processed", {"assigned_count": len(result)})
    return {"assigned": result}


@router.post("/failover", tags=["Instances"])
def api_failover():
    """Run a full failover cycle: check hosts, requeue orphaned jobs, reassign."""
    requeued, assigned = failover_and_reassign()
    return {
        "requeued": [
            {"job_id": j["job_id"], "name": j["name"], "retries": j.get("retries", 0)}
            for j in requeued
        ],
        "assigned": [
            {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned
        ],
    }


@router.post("/instances/{job_id}/cancel", tags=["Instances"])
def api_cancel_instance(job_id: str, request: Request):
    """Cancel a running or queued instance. For interactive instances, stops the container."""
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    jobs = list_jobs()
    job = next((j for j in jobs if j["job_id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    if job.get("status") in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Instance already {job['status']}")

    # If running on a host, kill the container directly and also schedule via agent
    if job.get("host_id") and job.get("status") in ("running", "assigned", "leased"):
        hosts = list_hosts()
        hmap = {h["host_id"]: h for h in hosts}
        host = hmap.get(job["host_id"])
        if host:
            try:
                kill_job(job, host)
            except Exception as e:
                log.warning("Container kill failed for %s: %s", job_id, e)
        with _agent_lock:
            _agent_preempt[job["host_id"]].append(job_id)

    update_job_status(job_id, "cancelled")
    broadcast_sse("job_cancelled", {"job_id": job_id})

    # Detach any volumes still attached to this instance
    try:
        from volumes import get_volume_engine

        get_volume_engine().detach_all_for_instance(job_id)
    except Exception as e:
        log.warning("Volume detach on cancel failed for %s: %s", job_id, e)

    return {"ok": True, "job_id": job_id, "status": "cancelled"}


@router.post("/instance/{job_id}/requeue", tags=["Instances"])
def api_requeue_instance(job_id: str, request: Request):
    """Manually requeue a failed or stuck job."""
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    result = requeue_job(job_id)
    if not result:
        raise HTTPException(
            status_code=400,
            detail=f"Could not requeue job {job_id} (max retries exceeded or not found)",
        )
    # Clear stale logs from previous attempt so the UI starts fresh
    _job_log_buffers.pop(job_id, None)
    try:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM job_logs WHERE job_id = %s", (job_id,))
    except Exception:
        pass  # non-critical — logs will be overwritten anyway
    push_job_log(job_id, "Requeued — waiting for GPU assignment", "info")
    return {"ok": True, "instance": result}


# ── Pause / Resume ───────────────────────────────────────────────────


@router.post("/instances/{job_id}/pause", tags=["Instances"])
def api_pause_instance(job_id: str, request: Request):
    """Pause a running instance. Stops the container but preserves volumes.

    Requires authentication. Only the instance owner or an admin can pause.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    # Owner check (admins can pause any instance)
    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to pause this instance")

    if job.get("status") != "running":
        raise HTTPException(status_code=400, detail=f"Instance is {job.get('status')}, not running")

    from billing import get_billing_engine

    be = get_billing_engine()
    result = be.pause_instance(job_id, reason="user_paused")
    if not result.get("paused"):
        raise HTTPException(status_code=400, detail=result.get("reason", "pause failed"))

    broadcast_sse("instance_paused", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/resume", tags=["Instances"])
def api_resume_instance(job_id: str, request: Request):
    """Resume a paused instance. Restarts the container from preserved state.

    Requires authentication. Only the instance owner or an admin can resume.
    Returns 402 if wallet has insufficient funds.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    # Owner check (admins can resume any instance)
    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to resume this instance")

    status = job.get("status", "")
    if status not in ("paused_low_balance", "user_paused"):
        raise HTTPException(status_code=400, detail=f"Instance is {status}, not paused")

    # Wallet pre-flight — must have funds to resume
    from billing import get_billing_engine

    be = get_billing_engine()
    wallet = be.get_wallet(job_owner or customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(status_code=402, detail="Wallet suspended — please add funds")
    if wallet["balance_cad"] <= 0:
        raise HTTPException(status_code=402, detail="Insufficient wallet balance to resume")

    result = be.resume_instance(job_id)
    if not result.get("resumed"):
        detail = result.get("reason", "resume failed")
        code = 402 if detail == "insufficient_balance" else 400
        raise HTTPException(status_code=code, detail=detail)

    broadcast_sse("instance_resumed", {"job_id": job_id})
    return {"ok": True, "instance": result}


# ── Stop / Start / Restart / Terminate ──────────────────────────────


@router.post("/instances/{job_id}/stop", tags=["Instances"])
def api_stop_instance(job_id: str, request: Request):
    """Gracefully stop a running instance.

    Sends SIGTERM to the container (docker stop -t 10). The container is
    preserved — data, volumes and configuration are intact. Storage billing
    continues at the per-GB rate. The instance can be started again at any time.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to stop this instance")

    if job.get("status") != "running":
        raise HTTPException(
            status_code=400, detail=f"Instance is '{job.get('status')}', must be running to stop"
        )

    from billing import get_billing_engine

    be = get_billing_engine()
    result = be.stop_instance(job_id, reason="user_stopped")
    if not result.get("stopped"):
        raise HTTPException(status_code=500, detail=result.get("reason", "stop failed"))

    broadcast_sse("instance_stopped", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/start", tags=["Instances"])
def api_start_instance(job_id: str, request: Request):
    """Start a stopped instance.

    Restores the exited container via docker start. Container data is
    preserved. Requires a positive wallet balance. Compute billing resumes
    immediately.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to start this instance")

    allowed_statuses = {"stopped", "user_paused", "paused_low_balance"}
    if job.get("status") not in allowed_statuses:
        raise HTTPException(
            status_code=400, detail=f"Instance is '{job.get('status')}', must be stopped to start"
        )

    from billing import get_billing_engine

    be = get_billing_engine()
    wallet = be.get_wallet(job_owner or customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(status_code=402, detail="Wallet suspended — please add funds")
    if wallet["balance_cad"] <= 0:
        raise HTTPException(status_code=402, detail="Insufficient wallet balance to start instance")

    result = be.start_instance(job_id)
    if not result.get("started"):
        detail = result.get("reason", "start failed")
        code = 402 if detail == "insufficient_balance" else 500
        raise HTTPException(status_code=code, detail=detail)

    broadcast_sse("instance_started", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/restart", tags=["Instances"])
def api_restart_instance(job_id: str, request: Request):
    """Restart an instance. Container data is fully preserved.

    Works from both running and stopped states:
    - Running → graceful stop → start (docker stop + docker start)
    - Stopped → start (docker start only)

    Billing is continuous — no gap. Requires a positive wallet balance.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to restart this instance")

    if job.get("status") not in ("running", "stopped"):
        raise HTTPException(
            status_code=400,
            detail=f"Instance is '{job.get('status')}', must be running or stopped to restart",
        )

    from billing import get_billing_engine

    be = get_billing_engine()
    wallet = be.get_wallet(job_owner or customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(status_code=402, detail="Wallet suspended — please add funds")
    if wallet["balance_cad"] <= 0:
        raise HTTPException(
            status_code=402, detail="Insufficient wallet balance to restart instance"
        )

    result = be.restart_instance(job_id)
    if not result.get("restarted"):
        detail = result.get("reason", "restart failed")
        code = 402 if detail == "insufficient_balance" else 500
        raise HTTPException(status_code=code, detail=detail)

    broadcast_sse("instance_restarted", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/admin/instances/{job_id}/reinject-shell", tags=["Instances"])
def api_admin_reinject_shell(job_id: str, request: Request):
    """Admin-only: re-apply MOTD/PS1/sshd setup to the instance's container.

    Enqueues a `reinject_shell` command for the host running this instance.
    The worker agent picks it up within one poll cycle (~5-10 s) and
    re-runs `_inject_ssh_keys` against the container. Idempotent — safe
    to call repeatedly. Returns the enqueued command id.

    Use case: an admin updated the MOTD/PS1 logic or debugged a stuck
    container and needs to refresh shell setup without restarting the
    worker agent.
    """
    user = _require_auth(request)
    if not (user.get("role") == "admin" or user.get("is_admin")):
        raise HTTPException(403, "Admin only")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Instance {job_id} not found")

    host_id = job.get("host_id", "")
    if not host_id:
        raise HTTPException(400, "Instance has no host assigned")

    container_name = job.get("container_name") or f"xcl-{job_id}"
    from routes.agent import enqueue_agent_command

    cmd_id = enqueue_agent_command(
        host_id=host_id,
        command="reinject_shell",
        args={"job_id": job_id, "container_name": container_name},
        created_by=user.get("customer_id", user.get("user_id", "admin")),
    )
    log.info(
        "Admin %s enqueued reinject_shell cmd=%d host=%s job=%s",
        user.get("email", "?"),
        cmd_id,
        host_id,
        job_id,
    )
    return {"ok": True, "command_id": cmd_id, "host_id": host_id, "job_id": job_id}


@router.post("/instances/{job_id}/terminate", tags=["Instances"])
def api_terminate_instance(job_id: str, request: Request):
    """Permanently terminate an instance. This action is irreversible.

    The container and its anonymous volumes are hard-killed and removed.
    Named/NFS volumes are preserved. The instance cannot be restarted
    after termination.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to terminate this instance")

    terminal_statuses = {"terminated", "completed", "failed", "preempted", "cancelled"}
    if job.get("status") in terminal_statuses:
        raise HTTPException(status_code=400, detail=f"Instance already {job.get('status')}")

    from billing import get_billing_engine

    be = get_billing_engine()
    result = be.terminate_instance(job_id)
    if not result.get("terminated"):
        raise HTTPException(status_code=400, detail=result.get("reason", "terminate failed"))

    broadcast_sse("instance_terminated", {"job_id": job_id})
    return {"ok": True, "instance": result}


# ── Helper: push_job_log ──


def push_job_log(
    job_id: str,
    line: str,
    level: str = "info",
    timestamp: float | None = None,
    *,
    persist: bool = True,
):
    """Push a log line into the per-job log buffer + optionally persist to PG + SSE broadcast.

    Three sinks, in this priority:
      1. In-memory ring buffer (`_job_log_buffers`) — fastest path for
         active terminal viewers; trimmed to `_JOB_LOG_MAX` entries per job.
      2. PG `job_logs` table — durable; survives API restart and lets the
         terminal WS replay history for jobs that haven't reached running yet.
         Skipped when *persist=False* (caller already wrote to PG).
      3. SSE broadcast — pushes to any connected dashboard clients live.

    Persisting to PG here (vs. only on worker-uploaded logs via routes/agent.py)
    means lifecycle events like "Queued", "Assigned to host X", "Starting" are
    preserved even if no one was connected at the moment they happened — which
    matters most for short-lived jobs that fail before a user opens the UI.

    The PG write is wrapped in a broad try/except: we never want a logging
    failure to abort the caller (scheduler, worker claim handler, reaper, etc).
    """
    ts = timestamp or time.time()
    entry = {"timestamp": ts, "line": line, "message": line, "level": level}
    buf = _job_log_buffers[job_id]
    buf.append(entry)
    if len(buf) > _JOB_LOG_MAX:
        _job_log_buffers[job_id] = buf[-_JOB_LOG_MAX:]

    # Durable persist — best-effort, never raise. Skipped when caller
    # already wrote to PG (persist=False) to avoid duplicate rows.
    if persist:
        try:
            from db import _get_pg_pool

            with _get_pg_pool().connection() as conn:
                conn.execute(
                    "INSERT INTO job_logs (job_id, ts, level, line) VALUES (%s, %s, %s, %s)",
                    (job_id, ts, level, line),
                )
        except Exception:
            # Intentionally silent — in-memory + SSE still delivered the message.
            pass

    # Also broadcast to general SSE stream
    broadcast_sse("job_log", {"job_id": job_id, **entry})


# ── Helper: _load_pg_logs ──


def _load_pg_logs(job_id: str, limit: int = 200) -> list[dict]:
    """Load recent log lines from PG job_logs table (fallback when in-memory buffer is empty)."""
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    "SELECT ts AS timestamp, level, line, line AS message FROM job_logs "
                    "WHERE job_id = %s ORDER BY ts DESC LIMIT %s",
                    (job_id, limit),
                ).fetchall()
        if not rows:
            return []
        return list(reversed(rows))  # chronological order, already dicts
    except Exception:
        return []


# ── Helper: _job_log_generator ──


async def _job_log_generator(request: Request, job_id: str):
    """Async generator that yields SSE events for a specific job.

    Replays buffered log lines, then live-tails new events
    from the broadcast SSE bus filtered to this job_id.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)

    try:
        # SSE retry hint for automatic browser reconnection
        yield "retry: 3000\n\n"

        # Replay buffered log lines (PG fallback if buffer is empty).
        # Prefer PG if it has MORE rows than the in-memory buffer — covers the
        # cross-process case where scheduler-worker wrote logs that never hit
        # this API process's in-memory buffer.
        in_mem = list(_job_log_buffers.get(job_id, []))
        pg_rows = _load_pg_logs(job_id, limit=500)
        replay = pg_rows if len(pg_rows) >= len(in_mem) else in_mem
        last_ts = 0.0
        for entry in replay:
            data = json.dumps({"job_id": job_id, **entry})
            yield f"event: job_log\ndata: {data}\n\n"
            try:
                last_ts = max(last_ts, float(entry.get("timestamp") or entry.get("ts") or 0))
            except (TypeError, ValueError):
                pass

        yield f"event: connected\ndata: {json.dumps({'job_id': job_id, 'status': 'streaming'})}\n\n"

        # Back off PG polling once a job is terminal and we've observed a few
        # empty polls — terminal jobs won't produce new log rows, so keep the
        # connection alive with just keepalives. Prevents an idle tab from
        # hammering PG forever.
        _TERMINAL_STATES = {"completed", "failed", "cancelled", "terminated", "preempted"}
        empty_polls_after_terminal = 0
        skip_pg_poll = False

        while True:
            if await request.is_disconnected():
                break

            # Poll PG for new log rows since last_ts. This is the ONLY path by
            # which logs written from other processes (scheduler-worker,
            # worker-agent uploads) reach this SSE client — the in-process
            # broadcast bus only sees writes from this same API worker.
            if not skip_pg_poll:
                try:
                    from db import _get_pg_pool
                    from psycopg.rows import dict_row

                    with _get_pg_pool().connection() as conn:
                        with conn.cursor(row_factory=dict_row) as cur:
                            new_rows = cur.execute(
                                "SELECT ts AS timestamp, level, line, line AS message "
                                "FROM job_logs WHERE job_id = %s AND ts > %s "
                                "ORDER BY ts ASC LIMIT 200",
                                (job_id, last_ts),
                            ).fetchall()
                            # Check status once per poll so we can quiesce
                            # terminal jobs and free up DB connections.
                            status_row = cur.execute(
                                "SELECT status FROM jobs WHERE job_id = %s",
                                (job_id,),
                            ).fetchone()
                            cur_status = (status_row or {}).get("status", "") if status_row else ""
                    for row in new_rows:
                        yield f"event: job_log\ndata: {json.dumps({'job_id': job_id, **row})}\n\n"
                        try:
                            last_ts = max(last_ts, float(row.get("timestamp") or 0))
                        except (TypeError, ValueError):
                            pass

                    if cur_status in _TERMINAL_STATES:
                        if not new_rows:
                            empty_polls_after_terminal += 1
                            # After ~30 empty polls (~60s) on a terminal job,
                            # stop polling. Client still gets keepalives and
                            # in-process broadcast events.
                            if empty_polls_after_terminal >= 30:
                                skip_pg_poll = True
                        else:
                            empty_polls_after_terminal = 0
                except Exception:
                    # PG hiccup — keep the stream alive; will retry on next tick.
                    pass

            try:
                msg = await asyncio.wait_for(queue.get(), timeout=2)
                event_type = msg.get("event", "message")
                event_data = msg.get("data", {})
                # Filter: only pass through events for this job
                if event_data.get("job_id") == job_id or event_type in (
                    "job_status",
                    "job_log",
                    "lease_claimed",
                    "lease_released",
                    "job_completed",
                    "job_failed",
                ):
                    if event_data.get("job_id", "") == job_id:
                        yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    finally:
        with _sse_lock:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)


@router.get("/instances/{job_id}/logs/stream", tags=["Instances"])
async def api_instance_log_stream(request: Request, job_id: str):
    """Stream real-time logs for a specific job via Server-Sent Events.

    Connect with `EventSource('/jobs/{job_id}/logs/stream')` in the browser
    or `curl -N` from the CLI. Replays buffered log lines on connect, then
    live-tails new log entries until the client disconnects or the job completes.

    Events emitted:
    - `job_log` — individual log line (data: {job_id, timestamp, line, level})
    - `job_status` — status change (data: {job_id, status})
    - `connected` — initial handshake (data: {job_id, status: "streaming"})
    """
    user = _require_auth(request)
    _require_scope(user, "instances:read")
    # Verify job exists
    jobs = list_jobs()
    if not any(j["job_id"] == job_id for j in jobs):
        raise HTTPException(404, f"Job {job_id} not found")

    return StreamingResponse(
        _job_log_generator(request, job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/instances/{job_id}/logs", tags=["Instances"])
def api_instance_logs(job_id: str, request: Request, limit: int = 100):
    """Get buffered log lines for a job (non-streaming).

    Returns the last `limit` log lines. Tries in-memory buffer first,
    falls back to persistent job_logs table in PG.
    For real-time streaming, use `/jobs/{job_id}/logs/stream` (SSE).
    """
    user = _require_auth(request)
    _require_scope(user, "instances:read")
    _check_job_access(user, job_id)
    limit = min(limit, 10_000)
    buf = _job_log_buffers.get(job_id, [])

    # If in-memory buffer is empty, load from PG
    if not buf:
        buf = _load_pg_logs(job_id, limit=limit)

    return {"ok": True, "job_id": job_id, "logs": buf[-limit:], "total": len(buf)}


@router.get("/instances/{job_id}/logs/download", tags=["Instances"])
def api_instance_logs_download(job_id: str, request: Request):
    """Download all available logs for a job as a plain text file."""
    user = _require_auth(request)
    _check_job_access(user, job_id)
    buf = _job_log_buffers.get(job_id, [])
    if not buf:
        buf = _load_pg_logs(job_id, limit=10_000)
    if not buf:
        raise HTTPException(404, "No logs available for this job")

    lines = []
    for entry in buf:
        ts = entry.get("timestamp", "")
        level = entry.get("level", "")
        msg = entry.get("message", "") or entry.get("line", "")
        lines.append(f"{ts} [{level.upper()}] {msg}")
    text = "\n".join(lines)

    import re

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", job_id)[:128]
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{safe_id}-logs.txt"'},
    )


@router.get("/tiers", tags=["Instances"])
def api_list_tiers():
    """List all priority tiers with their multipliers."""
    return {"tiers": list_tiers()}


@router.post("/api/v2/scheduler/process-binpack", tags=["Jobs"])
def api_process_queue_binpack(canada_only: bool = False, province: str = ""):
    """Process job queue using best-fit-decreasing bin packing."""
    assigned = process_queue_binpack(
        canada_only=canada_only or None,
        province=province or None,
    )
    return {"ok": True, "assigned": assigned, "count": len(assigned)}


@router.post("/api/instances/{job_id}/stream-ticket")
def api_instance_stream_ticket(job_id: str, request: Request) -> dict:
    """Issue a short-lived one-time WebSocket ticket for instance streaming."""
    user = _require_auth(request)
    _check_job_access(user, job_id)
    ticket = _issue_ws_ticket(
        user,
        request=request,
        purpose="instance_stream",
        target=job_id,
    )
    return {
        "ok": True,
        "ticket": ticket["ticket"],
        "expires_in": int(max(0, ticket["expires_at"] - time.time())),
    }


# ── WebSocket Instance Streaming ──────────────────────────────────


@router.websocket("/ws/instances/{job_id}")
async def ws_instance_stream(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time instance updates.

    Sends:
    - ``instance`` — full instance snapshot on connect and on status changes
    - ``job_log`` — individual log lines
    - ``job_status`` — status change notifications
    - ``ping`` — keepalive every 30 s

    Client can send ``{"event": "pong"}`` or ``{"event": "refresh"}``
    to request a fresh instance snapshot.
    """
    if not _validate_ws_origin(
        websocket,
        require_for_cookie_auth=True,
        allow_query_token=False,
    ):
        await websocket.close(code=1008, reason="Invalid origin")
        return

    if not _check_ws_connect_rate_limit(websocket, bucket="instances"):
        await websocket.close(code=4429, reason="Connection rate limit exceeded")
        return

    ticket = websocket.query_params.get("ticket", "").strip()
    if ticket:
        user = _consume_ws_ticket(
            ticket,
            websocket,
            purpose="instance_stream",
            target=job_id,
        )
    else:
        user = _validate_ws_auth(websocket, allow_query_token=False)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Per-instance snapshot loader. Uses a single-row DB lookup (get_job) and
    # a cached host_map rather than scanning list_jobs()/list_hosts() on every
    # status change — those scans on the full tables were blocking the event
    # loop long enough for nginx/Starlette to drop the WS with 1006, producing
    # the 2s reconnect loop and stale UI (status stuck on "assigned" even
    # though the DB had already moved to "running").
    _host_map_cache: dict[str, dict] = {}
    _host_map_ts: float = 0.0

    def _load_snapshot() -> dict | None:
        nonlocal _host_map_cache, _host_map_ts
        from scheduler import get_job

        j = get_job(job_id)
        if not j:
            return None
        # Refresh host map at most every 15s — hosts rarely change and the
        # scan was the expensive part.
        now_ts = time.time()
        if now_ts - _host_map_ts > 15.0:
            _host_map_cache = {h["host_id"]: h for h in list_hosts(active_only=False)}
            _host_map_ts = now_ts
        _enrich_instance(j, _host_map_cache)
        return j

    async def _load_snapshot_async() -> dict | None:
        return await asyncio.to_thread(_load_snapshot)

    async def _safe_send(payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            return False
        except Exception as e:
            log.debug("ws_instance_stream send failed: %s", e)
            return False

    # Initial snapshot
    instance = await _load_snapshot_async()
    if not instance:
        await _safe_send({"event": "error", "data": {"message": "Instance not found"}})
        await websocket.close(code=4004)
        return

    # Ownership check — only job owner or admin may stream
    if not (user.get("role") == "admin" or user.get("is_admin")):
        job_owner = instance.get("owner", "")
        customer_id = user.get("customer_id", user.get("user_id", ""))
        if job_owner and job_owner != customer_id:
            await _safe_send({"event": "error", "data": {"message": "Not authorized"}})
            await websocket.close(code=4003)
            return

    _ws_connections[job_id].add(websocket)
    if not await _safe_send({"event": "instance", "data": instance}):
        _ws_connections[job_id].discard(websocket)
        return

    # Replay buffered logs (PG fallback if buffer is empty)
    replay = list(_job_log_buffers.get(job_id, []))[-50:]
    if not replay:
        replay = await asyncio.to_thread(_load_pg_logs, job_id, 50)
    for entry in replay:
        if not await _safe_send({"event": "job_log", "data": {"job_id": job_id, **entry}}):
            _ws_connections[job_id].discard(websocket)
            return

    # Subscribe to the broadcast SSE bus
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)

    closed = False

    async def _send_loop():
        nonlocal closed
        while not closed:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                if not await _safe_send({"event": "ping", "data": {"ts": time.time()}}):
                    closed = True
                    return
                continue
            event_type = msg.get("event", "message")
            event_data = msg.get("data", {})
            if event_data.get("job_id") != job_id:
                continue
            if not await _safe_send({"event": event_type, "data": event_data}):
                closed = True
                return
            # On status change, send full enriched instance snapshot
            if event_type == "job_status":
                fresh = await _load_snapshot_async()
                if fresh and not await _safe_send({"event": "instance", "data": fresh}):
                    closed = True
                    return

    async def _recv_loop():
        nonlocal closed
        while not closed:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("event") == "refresh":
                    fresh = await _load_snapshot_async()
                    if fresh and not await _safe_send({"event": "instance", "data": fresh}):
                        closed = True
                        return
            except (WebSocketDisconnect, RuntimeError):
                closed = True
                break
            except (json.JSONDecodeError, KeyError):
                pass

    try:
        done, pending = await asyncio.wait(
            [asyncio.ensure_future(_send_loop()), asyncio.ensure_future(_recv_loop())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        closed = True
        _ws_connections[job_id].discard(websocket)
        with _sse_lock:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)
        try:
            await websocket.close()
        except Exception as e:
            log.debug("WS close error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# P3.1 — Pod save-as-template (user_images)
# ─────────────────────────────────────────────────────────────────────────────
#
# A running interactive instance can be turned into a reusable image by
# enqueuing a `snapshot_container` agent command which runs `docker commit`
# on the host. v1 keeps the image local to the host; once
# XCELSIOR_REGISTRY_URL is configured the worker pushes it to a per-user
# namespace. The user_images table is the system-of-record.


# Docker reference spec requires lowercase for the repository portion;
# we auto-lowercase the name in the validator so users aren't surprised
# by a `invalid reference format` failure at commit time on the host.
# Tags remain case-sensitive per the Docker distribution spec.
_IMAGE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_IMAGE_TAG_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,62}$")


def _owner_slug(owner_id: str) -> str:
    """Deterministic, collision-resistant namespace slug for an owner.

    Two different owner_ids can sanitize to the same string
    (e.g. 'a.b@x' and 'a-b-x'), so we append a short sha256 prefix to
    guarantee uniqueness in the local-only image tag path.
    """
    import hashlib
    clean = re.sub(r"[^a-z0-9_-]", "-", (owner_id or "").lower()).strip("-") or "user"
    # Keep registry paths readable but always disambiguated.
    # P3/B5 — 16 hex chars = 64 bits of entropy. An 8-char prefix (32 bits)
    # is vulnerable to birthday collisions at ~65k users sharing a slug;
    # at Xcelsior's expected scale that's borderline. 16 chars pushes the
    # 50% collision point past 2^32 users.
    digest = hashlib.sha256((owner_id or "").encode("utf-8")).hexdigest()[:16]
    return f"{clean[:32]}-{digest}"


def _build_image_ref(owner_id: str, name: str, tag: str) -> str:
    reg = os.environ.get("XCELSIOR_REGISTRY_URL", "").strip().rstrip("/")
    slug = _owner_slug(owner_id)
    if reg:
        return f"{reg}/{slug}/{name}:{tag}"
    # Local-only fallback when no registry configured. Image lives on
    # the source host and is usable there; cross-host reuse requires a
    # registry. Kept intentionally simple for v1.
    return f"xcl-{slug}-{name}:{tag}"


class SnapshotIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=63)
    tag: str = Field(default="latest", max_length=63)
    description: str = Field(default="", max_length=512)

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str) -> str:
        # Docker refuses uppercase in the repository name; enforce early
        # rather than producing a confusing `invalid reference format`
        # error from `docker commit` on the host.
        v = (v or "").strip().lower()
        if not _IMAGE_NAME_RE.match(v):
            raise ValueError("name must match [a-z0-9][a-z0-9._-]* and be <=63 chars")
        return v

    @field_validator("tag")
    @classmethod
    def _v_tag(cls, v: str) -> str:
        v = (v or "latest").strip() or "latest"
        if not _IMAGE_TAG_RE.match(v):
            raise ValueError("tag must match [a-z0-9][a-z0-9._-]*")
        return v

    @field_validator("description")
    @classmethod
    def _v_desc(cls, v: str) -> str:
        v = (v or "").strip()
        # Strip any control chars other than newlines/tabs.
        v = "".join(c for c in v if ord(c) >= 32 or c in "\n\t")
        # P3/B9 — strip Unicode bidirectional override characters. These
        # can be used to mask malicious content in logs / UIs by flipping
        # display order (CVE-2021-42574, "Trojan Source"). We have no
        # legitimate use for LRO/RLO/PDF/LRI/RLI/FSI/PDI in an image
        # description.
        _BIDI_CHARS = {
            "\u202A", "\u202B", "\u202C", "\u202D", "\u202E",  # LRE/RLE/PDF/LRO/RLO
            "\u2066", "\u2067", "\u2068", "\u2069",             # LRI/RLI/FSI/PDI
        }
        if any(c in _BIDI_CHARS for c in v):
            v = "".join(c for c in v if c not in _BIDI_CHARS)
        return v[:512]


def _user_images_pool():
    """Lazy import; keeps pg_pool import out of module-load order."""
    from db import _get_pg_pool  # type: ignore
    return _get_pg_pool()


@router.post("/instances/{job_id}/snapshot", tags=["Instances"])
def api_snapshot_instance(job_id: str, body: SnapshotIn, request: Request):
    """Create a user image from a running container (`docker commit`).

    The real work is performed asynchronously by the worker agent on the
    host the job is currently assigned to. This endpoint validates the
    request, inserts a `user_images` row in `pending`, and enqueues a
    `snapshot_container` directive.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    owner_id = _canonical_owner_id(user)
    if not owner_id:
        raise HTTPException(401, "Authentication required")
    # P3/B4 — throttle before doing any DB / agent work.
    _check_snapshot_rate_limit(owner_id)

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Instance not found")
    is_admin = bool(user.get("is_admin") or user.get("admin"))
    if job.get("owner") != owner_id and not is_admin:
        raise HTTPException(403, "Not your instance")
    status = str(job.get("status") or "")
    # Only running containers can be snapshotted — billing pause / stop
    # flows `docker rm -f` the container, after which `docker commit`
    # has nothing to target.
    if status != "running":
        raise HTTPException(
            400,
            f"Instance must be running to snapshot (got {status!r})",
        )
    host_id = str(job.get("host") or job.get("host_id") or "")
    if not host_id:
        raise HTTPException(409, "Instance has no assigned host")

    image_id = f"img-{uuid.uuid4().hex[:12]}"
    image_ref = _build_image_ref(owner_id, body.name, body.tag)
    container_name = f"xcl-{job_id}"
    now = time.time()

    from routes.agent import enqueue_agent_command

    pool = _user_images_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        # Enforce uniqueness at the app layer so we can return a clean
        # 409 before booking the agent command.
        cur.execute(
            "SELECT image_id FROM user_images "
            "WHERE owner_id=%s AND name=%s AND tag=%s AND deleted_at=0",
            (owner_id, body.name, body.tag),
        )
        existing = cur.fetchone()
        if existing:
            raise HTTPException(
                409,
                f"Image {body.name}:{body.tag} already exists (delete it first to overwrite)",
            )
        cur.execute(
            """
            INSERT INTO user_images (
                image_id, owner_id, name, tag, description,
                source_job_id, host_id, image_ref, size_bytes,
                status, created_at, deleted_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,0,'pending',%s,0)
            """,
            (
                image_id, owner_id, body.name, body.tag, body.description,
                job_id, host_id, image_ref, now,
            ),
        )

    cmd_id = enqueue_agent_command(
        host_id,
        "snapshot_container",
        {
            "image_id": image_id,
            "container_name": container_name,
            "image_ref": image_ref,
            "job_id": job_id,
        },
        created_by=owner_id,
        ttl_sec=3600,
    )

    try:
        broadcast_sse(
            "user_image_pending",
            {
                "image_id": image_id,
                "owner_id": owner_id,
                "name": body.name,
                "tag": body.tag,
                "image_ref": image_ref,
                "source_job_id": job_id,
            },
        )
    except Exception as e:
        log.debug("broadcast_sse(user_image_pending) failed: %s", e)

    log.info(
        "User image snapshot queued image=%s job=%s host=%s owner=%s",
        image_id, job_id, host_id, owner_id,
    )
    return {
        "ok": True,
        "image_id": image_id,
        "image_ref": image_ref,
        "command_id": cmd_id,
        "status": "pending",
    }


@router.get("/user-images", tags=["Instances"])
def api_list_user_images(request: Request):
    """List the authenticated user's saved pod templates."""
    user = _require_auth(request)
    owner_id = _canonical_owner_id(user)
    if not owner_id:
        raise HTTPException(401, "Authentication required")
    pool = _user_images_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT image_id, name, tag, description, source_job_id, host_id,
                   image_ref, size_bytes, status, created_at
            FROM user_images
            WHERE owner_id=%s AND deleted_at=0
            ORDER BY created_at DESC
            LIMIT 500
            """,
            (owner_id,),
        )
        rows = cur.fetchall() or []
    items = [
        {
            "image_id": r[0], "name": r[1], "tag": r[2], "description": r[3],
            "source_job_id": r[4], "host_id": r[5], "image_ref": r[6],
            "size_bytes": int(r[7] or 0), "status": r[8], "created_at": float(r[9]),
        }
        for r in rows
    ]
    return {"images": items}


@router.delete("/user-images/{image_id}", tags=["Instances"])
def api_delete_user_image(image_id: str, request: Request):
    """Soft-delete a user image record. (Underlying docker image not removed.)"""
    user = _require_auth(request)
    owner_id = _canonical_owner_id(user)
    if not owner_id:
        raise HTTPException(401, "Authentication required")
    is_admin = bool(user.get("is_admin") or user.get("admin"))
    pool = _user_images_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT owner_id FROM user_images WHERE image_id=%s AND deleted_at=0",
            (image_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Image not found")
        if row[0] != owner_id and not is_admin:
            raise HTTPException(403, "Not your image")
        cur.execute(
            "UPDATE user_images SET deleted_at=%s WHERE image_id=%s",
            (time.time(), image_id),
        )
    try:
        broadcast_sse("user_image_deleted", {"image_id": image_id, "owner_id": owner_id})
    except Exception as e:
        log.debug("broadcast_sse(user_image_deleted) failed: %s", e)
    return {"ok": True}


class _UserImageCompleteIn(BaseModel):
    status: str
    size_bytes: int = 0
    error: str = ""

    @field_validator("status")
    @classmethod
    def _v_status(cls, v: str) -> str:
        if v not in ("ready", "failed"):
            raise ValueError("status must be 'ready' or 'failed'")
        return v


@router.post("/user-images/{image_id}/complete", tags=["Instances"])
def api_user_image_complete(image_id: str, body: _UserImageCompleteIn, request: Request):
    """Internal: worker agent callback after `docker commit` finishes.

    Authenticated with the same shared agent secret used for
    `/agent/commands/{host_id}`. Flips the pending row into ready/failed.
    """
    from routes.agent import _require_agent_auth

    pool = _user_images_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT owner_id, status, host_id FROM user_images "
            "WHERE image_id=%s AND deleted_at=0",
            (image_id,),
        )
        row = cur.fetchone()
        if not row:
            # Authenticate before revealing existence (timing-equivalent
            # path below handles the success case).
            _require_agent_auth(request)
            raise HTTPException(404, "Image not found")
        owner_id, prev_status, img_host_id = row[0], row[1], row[2]
        # Bind the callback to the host that was originally asked to
        # produce the image. Prevents a compromised host from flipping
        # another host's pending image to ready.
        _require_agent_auth(request, host_id=img_host_id)
        if prev_status not in ("pending",):
            # Idempotent: callback may arrive twice; accept without re-update.
            return {"ok": True, "status": prev_status}
        cur.execute(
            "UPDATE user_images SET status=%s, size_bytes=%s WHERE image_id=%s",
            (body.status, max(0, int(body.size_bytes)), image_id),
        )

    try:
        broadcast_sse(
            "user_image_complete",
            {
                "image_id": image_id,
                "owner_id": owner_id,
                "status": body.status,
                "size_bytes": body.size_bytes,
                "error": (body.error or "")[:500],
            },
        )
    except Exception as e:
        log.debug("broadcast_sse(user_image_complete) failed: %s", e)
    log.info("User image %s completed status=%s size=%d", image_id, body.status, body.size_bytes)
    return {"ok": True}
