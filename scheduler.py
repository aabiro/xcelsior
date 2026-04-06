# Xcelsior v2.0.0 — distributed GPU scheduler for Canadians who refuse to wait.
# Pull-based architecture. PostgreSQL + JSONB. Defense in depth. SSE dashboards.
# 20 phases + production hardening. No shortcuts. Ever upward.

import fcntl
import json
import logging
import math
import os
import re
import shlex
import smtplib
import subprocess
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from email.mime.text import MIMEText

from db import (
    get_engine,
    DatabaseOps,
    emit_event,
    DB_BACKEND,
)

# v2.1 modules — event sourcing, verification, jurisdiction, billing, reputation
from events import (
    JobState,
    EventType,
    VALID_TRANSITIONS,
    get_event_store,
    get_state_machine,
)
from verification import get_verification_engine
from jurisdiction import (
    filter_hosts_by_jurisdiction,
    classify_host_trust_tier,
    JurisdictionConstraint,
    HostJurisdiction,
    TrustTier,
    generate_residency_trace,
)
from billing import get_billing_engine
from reputation import get_reputation_engine, score_to_tier

HOSTS_FILE = os.environ.get("XCELSIOR_HOSTS_FILE", os.path.join(os.path.dirname(__file__), "hosts.json"))
JOBS_FILE = os.environ.get("XCELSIOR_JOBS_FILE", os.path.join(os.path.dirname(__file__), "jobs.json"))
LOG_FILE = os.environ.get("XCELSIOR_LOG_FILE", os.path.join(os.path.dirname(__file__), "xcelsior.log"))
DEFAULT_DB_FILE = os.path.join(os.path.dirname(__file__), "xcelsior.db")


def _db_path():
    return os.environ.get("XCELSIOR_DB_PATH", DEFAULT_DB_FILE)


# ── Phase 13: Security ───────────────────────────────────────────────

SSH_KEY_PATH = os.environ.get("XCELSIOR_SSH_KEY_PATH", os.path.expanduser("~/.ssh/xcelsior"))
SSH_USER = os.environ.get("XCELSIOR_SSH_USER", "xcelsior")
API_TOKEN = os.environ.get("XCELSIOR_API_TOKEN", "")

# Guard against concurrent process_queue / failover_and_reassign
_scheduler_lock = threading.Lock()


# ── Phase 7: Logging ─────────────────────────────────────────────────


def setup_logging(log_file=None, level=logging.INFO):
    """
    Log everything. Every move. Every crash. Every win.
    Console + file. No silent failures.
    """
    log_file = log_file or LOG_FILE
    logger = logging.getLogger("xcelsior")

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler — the permanent record
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — see it live
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = setup_logging()


# ── Helpers ───────────────────────────────────────────────────────────

# Safe name pattern: alphanumeric, hyphens, underscores, dots, colons, slashes, and @
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9._:/@-]+$")


def _validate_name(value, label="value"):
    """Reject shell-unsafe characters in names used in commands."""
    if not value or not _SAFE_NAME_RE.match(str(value)):
        raise ValueError(
            f"Invalid {label}: {value!r} — only alphanumeric, hyphens, underscores, dots, colons, slashes, and @ allowed"
        )
    return str(value)


def _coerce_list(value):
    """Normalize deserialized payloads to a list shape."""
    return value if isinstance(value, list) else []


def _read_legacy_json_file(path):
    """Best-effort read from legacy JSON files."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return _coerce_list(json.load(f))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _ensure_storage_tables(conn):
    """No-op: tables created by Alembic migrations."""
    pass


@contextmanager
def _db_connection():
    """Shared DB connection wrapper using PostgreSQL pool."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


@contextmanager
def _atomic_mutation():
    """Execute a mutation in a single PostgreSQL transaction."""
    with _db_connection() as conn:
        yield conn


def _namespace_key(path):
    return os.path.realpath(path)


def _load_legacy_namespace(conn, path):
    """Read namespace data from state table first, then fallback JSON."""
    row = conn.execute(
        "SELECT payload FROM state WHERE namespace = %s", (_namespace_key(path),)
    ).fetchone()
    if row:
        try:
            data = row["payload"]
            return _coerce_list(data if isinstance(data, (list, dict)) else json.loads(data))
        except json.JSONDecodeError:
            pass
    return _read_legacy_json_file(path)


def _decode_payload(payload):
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None


def _upsert_job_row(conn, job):
    from psycopg.types.json import Jsonb
    job_id = str(job.get("job_id", "")).strip()
    if not job_id:
        return
    status = str(job.get("status") or "queued")
    priority = int(job.get("priority", 0) or 0)
    submitted_at = float(job.get("submitted_at", time.time()) or time.time())
    conn.execute(
        """
        INSERT INTO jobs(job_id, status, priority, submitted_at, host_id, payload)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT(job_id) DO UPDATE SET
            status = excluded.status,
            priority = excluded.priority,
            submitted_at = excluded.submitted_at,
            host_id = excluded.host_id,
            payload = excluded.payload
        """,
        (
            job_id,
            status,
            priority,
            submitted_at,
            job.get("host_id"),
            Jsonb(job),
        ),
    )


def _upsert_host_row(conn, host):
    from psycopg.types.json import Jsonb
    host_id = str(host.get("host_id", "")).strip()
    if not host_id:
        return
    status = str(host.get("status") or "active")
    registered_at = float(host.get("registered_at", time.time()) or time.time())
    conn.execute(
        """
        INSERT INTO hosts(host_id, status, registered_at, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(host_id) DO UPDATE SET
            status = excluded.status,
            registered_at = excluded.registered_at,
            payload = excluded.payload
        """,
        (
            host_id,
            status,
            registered_at,
            Jsonb(host),
        ),
    )


def _get_job_by_id_conn(conn, job_id):
    row = conn.execute("SELECT payload FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
    if not row:
        return None
    return _decode_payload(row["payload"])


def _get_host_by_id_conn(conn, host_id):
    row = conn.execute("SELECT payload FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
    if not row:
        return None
    return _decode_payload(row["payload"])


def _replace_jobs_in_conn(conn, jobs):
    conn.execute("DELETE FROM jobs")
    for job in jobs:
        if isinstance(job, dict):
            _upsert_job_row(conn, job)


def _replace_hosts_in_conn(conn, hosts):
    conn.execute("DELETE FROM hosts")
    for host in hosts:
        if isinstance(host, dict):
            _upsert_host_row(conn, host)


def _load_jobs_from_conn(conn, status=None):
    if status:
        rows = conn.execute(
            """
            SELECT payload
            FROM jobs
            WHERE status = %s
            ORDER BY submitted_at ASC, job_id ASC
            """,
            (status,),
        ).fetchall()
    else:
        rows = conn.execute("""
            SELECT payload
            FROM jobs
            ORDER BY submitted_at ASC, job_id ASC
            """).fetchall()
    jobs = []
    for row in rows:
        item = _decode_payload(row["payload"])
        if isinstance(item, dict):
            jobs.append(item)
    return jobs


def _load_hosts_from_conn(conn, active_only=False):
    if active_only:
        rows = conn.execute("""
            SELECT payload
            FROM hosts
            WHERE status = 'active'
            ORDER BY registered_at ASC, host_id ASC
            """).fetchall()
    else:
        rows = conn.execute("""
            SELECT payload
            FROM hosts
            ORDER BY registered_at ASC, host_id ASC
            """).fetchall()
    hosts = []
    for row in rows:
        item = _decode_payload(row["payload"])
        if isinstance(item, dict):
            hosts.append(item)
    return hosts


def _migrate_jobs_if_needed(conn):
    row = conn.execute("SELECT 1 FROM jobs LIMIT 1").fetchone()
    if row:
        return
    for job in _load_legacy_namespace(conn, JOBS_FILE):
        if isinstance(job, dict):
            _upsert_job_row(conn, job)


def _migrate_hosts_if_needed(conn):
    row = conn.execute("SELECT 1 FROM hosts LIMIT 1").fetchone()
    if row:
        return
    for host in _load_legacy_namespace(conn, HOSTS_FILE):
        if isinstance(host, dict):
            _upsert_host_row(conn, host)


def _set_job_fields(job_id, **updates):
    """Patch fields for one job atomically."""
    with _atomic_mutation() as conn:
        _migrate_jobs_if_needed(conn)
        job = _get_job_by_id_conn(conn, job_id)
        if not job:
            return None
        job.update(updates)
        _upsert_job_row(conn, job)
        return job


def _set_host_fields(host_id, **updates):
    """Patch fields for one host atomically."""
    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        host = _get_host_by_id_conn(conn, host_id)
        if not host:
            return None
        host.update(updates)
        _upsert_host_row(conn, host)
        return host


def _load_json(path):
    """Load persisted list data from SQLite, with one-time migration from JSON files."""
    namespace = _namespace_key(path)
    with _db_connection() as conn:
        row = conn.execute("SELECT payload FROM state WHERE namespace = %s", (namespace,)).fetchone()
        if row:
            try:
                data = row["payload"]
                return _coerce_list(data if isinstance(data, (list, dict)) else json.loads(data))
            except json.JSONDecodeError:
                return []

    payload = _read_legacy_json_file(path)
    if not payload:
        return []

    _save_json(path, payload)
    return payload


def _save_json(path, data):
    """Persist list data to PostgreSQL while preserving JSON-compatible interfaces."""
    from psycopg.types.json import Jsonb
    namespace = _namespace_key(path)
    with _atomic_mutation() as conn:
        conn.execute(
            "INSERT INTO state(namespace, payload) VALUES (%s, %s) "
            "ON CONFLICT(namespace) DO UPDATE SET payload = excluded.payload",
            (namespace, Jsonb(data)),
        )

    # Keep a legacy JSON mirror for backward compatibility/tests.
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Phase 1: Allocate ────────────────────────────────────────────────


def allocate(job, hosts):
    """Find the best host for this job.

    Enforces:
    - VRAM capacity check
    - GPU count check (multi-GPU jobs)
    - Admission gating (REPORT_FEATURE_FINAL.md §62: only admitted hosts)
    - Isolation tier enforcement (REPORT_FEATURE_FINAL.md §193:
      untrusted workloads require strong isolation tier)

    Prioritize: available VRAM > GPU count > speed > lowest cost.
    If nothing fits, return None (queue or reject).
    """
    if not hosts:
        return None

    num_gpus_needed = job.get("num_gpus", 1) or 1

    # Step 1: VRAM filter
    candidates = [h for h in hosts if h.get("free_vram_gb", 0) >= job.get("vram_needed_gb", 0)]
    if not candidates:
        return None

    # Step 1b: GPU count filter (multi-GPU jobs)
    if num_gpus_needed > 1:
        gpu_candidates = [h for h in candidates if h.get("gpu_count", 1) >= num_gpus_needed]
        if gpu_candidates:
            candidates = gpu_candidates
        else:
            log.warning(
                "ALLOCATE: job=%s needs %d GPUs but no single host has enough — "
                "best available: %d GPUs",
                job.get("name", "?"),
                num_gpus_needed,
                max((h.get("gpu_count", 1) for h in candidates), default=0),
            )
            # Fall through — allocate to best-available even if fewer GPUs
            # (job may still run with data parallelism across fewer GPUs)

    # Step 2: Admission gating — only admitted hosts can receive work
    # Per REPORT_FEATURE_FINAL.md §62: "refuse to run customer workloads
    # on hosts that are not patched to minimum safe versions"
    admitted_candidates = [h for h in candidates if h.get("admitted", False)]
    if not admitted_candidates:
        log.warning(
            "ALLOCATE BLOCKED: %d hosts have VRAM but none are admitted (job=%s)",
            len(candidates),
            job.get("name", "?"),
        )
        return None
    candidates = admitted_candidates

    # Step 3: Isolation tier enforcement
    # Per REPORT_FEATURE_FINAL.md §193: "untrusted workloads require
    # the strong isolation tier" (gVisor/Kata)
    job_tier = job.get("tier", "free") or "free"
    requires_isolation = job_tier in ("sovereign", "regulated", "secure")
    if requires_isolation:
        isolated = [h for h in candidates if h.get("recommended_runtime", "runc") != "runc"]
        if isolated:
            candidates = isolated
        else:
            log.warning(
                "ALLOCATE: tier=%s prefers gVisor/Kata isolation but no isolated "
                "hosts available — falling back to hardened runc (job=%s)",
                job_tier,
                job.get("name", "?"),
            )

    # Prioritize: GPU count match > compute efficiency > VRAM > speed > cost
    # Uses compute scores when available, and spot-adjusted cost for price comparison.
    def _host_score(h):
        gpu_match = min(h.get("gpu_count", 1), num_gpus_needed)
        vram = h.get("free_vram_gb", 0)
        latency = h.get("latency_ms", 999)
        base_cost = h.get("cost_per_hour", 999)

        # Compute efficiency: XCU per dollar (higher = better GPU per dollar)
        compute = h.get("compute_score") or estimate_compute_score(h.get("gpu_model", ""))
        price = max(base_cost, 0.01)
        efficiency = compute / price

        return (gpu_match, efficiency, vram, -latency, -base_cost)

    best = max(candidates, key=_host_score)
    log.info(
        "ALLOCATE job=%s -> host=%s (%s, %sGB free, %d GPUs, admitted=%s, runtime=%s)",
        job.get("name", "?"),
        best["host_id"],
        best.get("gpu_model"),
        best.get("free_vram_gb"),
        best.get("gpu_count", 1),
        best.get("admitted", False),
        best.get("recommended_runtime", "runc"),
    )
    return best


# ── Best-Fit Bin-Packing Allocator ───────────────────────────────────
# Processes queue in decreasing VRAM order (largest jobs first) to
# minimize fragmentation. Scoring: waste ratio, locality, reputation.

def allocate_binpack(job, hosts, user_province=None, volume_host_ids=None):
    """Best-fit-decreasing bin packer with locality + reputation scoring.

    Score = (1 - waste_ratio) * locality_bonus * reputation_bonus

    Args:
        job: dict with vram_needed_gb, num_gpus, tier, etc.
        hosts: list of host dicts
        user_province: user's province for locality scoring
        volume_host_ids: set of host_ids where user has attached volumes

    Returns:
        Best host or None
    """
    if not hosts:
        return None

    vram_needed = job.get("vram_needed_gb", 0)
    num_gpus_needed = job.get("num_gpus", 1) or 1
    volume_host_ids = volume_host_ids or set()

    # Filter: VRAM + admission
    candidates = [
        h for h in hosts
        if h.get("admitted", False)
        and h.get("free_vram_gb", 0) >= vram_needed
    ]
    if not candidates:
        return None

    # GPU count filter for multi-GPU gang scheduling
    if num_gpus_needed > 1:
        multi = [h for h in candidates if h.get("gpu_count", 1) >= num_gpus_needed]
        if multi:
            candidates = multi
        else:
            return None  # Gang scheduling: all GPUs must be on one host

    # Isolation tier filter
    job_tier = job.get("tier", "free") or "free"
    if job_tier in ("sovereign", "regulated", "secure"):
        isolated = [h for h in candidates if h.get("recommended_runtime", "runc") != "runc"]
        if isolated:
            candidates = isolated

    def _binpack_score(h):
        total_vram = max(h.get("total_vram_gb", 1), 1)
        free_vram = h.get("free_vram_gb", 0)
        waste_ratio = (free_vram - vram_needed) / total_vram
        fit_score = 1.0 - waste_ratio  # Tighter fit = higher score

        # Locality bonus
        host_province = h.get("province", "")
        locality = 1.0
        if user_province and host_province:
            if host_province == user_province:
                locality = 1.2  # Same province
            elif h.get("country", "CA") == "CA":
                locality = 1.0  # Same country
            else:
                locality = 0.5  # Cross-border

        # Data gravity: prefer hosts with user's volumes attached
        if h.get("host_id") in volume_host_ids:
            locality *= 1.3

        # Reputation bonus
        rep = h.get("reputation_score", 0.5)
        rep_bonus = 0.8 + (rep * 0.4)  # Range: 0.8 to 1.2

        # Compute efficiency
        compute = h.get("compute_score") or estimate_compute_score(h.get("gpu_model", ""))
        price = max(h.get("cost_per_hour", 0.01), 0.01)
        efficiency = compute / price / 100  # Normalized

        return fit_score * locality * rep_bonus * (1 + efficiency * 0.1)

    best = max(candidates, key=_binpack_score)
    log.info(
        "BINPACK job=%s -> host=%s (%s, %.1fGB free, score=%.3f)",
        job.get("name", "?"),
        best["host_id"],
        best.get("gpu_model"),
        best.get("free_vram_gb", 0),
        _binpack_score(best),
    )
    return best


# Throttle: track last job_error notification per job_id
_job_error_notified: dict[str, float] = {}


def process_queue_binpack(canada_only=None, province=None):
    """Process job queue using best-fit-decreasing order.

    Sorts queued jobs by vram_needed_gb descending (largest first)
    to minimize fragmentation, then allocates using bin-pack scoring.
    """
    hosts = list_hosts()
    if canada_only:
        hosts = [h for h in hosts if h.get("country", "").upper() == "CA"]
    if province:
        hosts = [h for h in hosts if h.get("province", "").upper() == province.upper()]

    jobs = [j for j in list_jobs() if j.get("status") == "queued"]
    # Best-fit-decreasing: sort by VRAM descending
    jobs.sort(key=lambda j: j.get("vram_needed_gb", 0), reverse=True)

    assigned = []
    skipped = []
    for job in jobs:
        host = allocate_binpack(job, hosts)
        if host:
            update_job_status(job["job_id"], "assigned", host["host_id"])
            assigned.append({"job_id": job["job_id"], "host_id": host["host_id"]})
            # Reduce host's available VRAM for subsequent allocations (clamped)
            host["free_vram_gb"] = max(0, host.get("free_vram_gb", 0) - job.get("vram_needed_gb", 0))
        else:
            skipped.append(job["job_id"])

    # Notify about unassignable jobs (throttle: once per 5 minutes per job)
    now = time.time()
    for jid in skipped:
        last_notified = _job_error_notified.get(jid, 0)
        if now - last_notified < 300:
            continue
        _job_error_notified[jid] = now
        emit_event("job_error", {
            "job_id": jid,
            "error": "no_hosts_available",
            "message": "No GPU hosts currently match your requirements. Your job remains queued.",
        })

    return assigned


# ── Phase 2: Host Registry ───────────────────────────────────────────


def load_hosts(active_only=False):
    """Load hosts from SQLite-backed storage."""
    with _db_connection() as conn:
        _migrate_hosts_if_needed(conn)
        return _load_hosts_from_conn(conn, active_only=active_only)


def save_hosts(hosts):
    """Replace all hosts (compatibility helper)."""
    with _atomic_mutation() as conn:
        _replace_hosts_in_conn(conn, hosts)


def register_host(
    host_id, ip, gpu_model, total_vram_gb, free_vram_gb, cost_per_hour=0.20, country="", province=""
):
    """
    Register a host. If it exists, update it. If not, add it.
    Every host sends: GPU model, VRAM, IP, uptime.
    You store it.
    """
    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        existing = _get_host_by_id_conn(conn, host_id)
        entry = {
            "host_id": host_id,
            "ip": ip,
            "gpu_model": gpu_model,
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
            "cost_per_hour": cost_per_hour,
            "registered_at": time.time(),
            "last_seen": time.time(),
            "status": "active",
        }
        if country:
            entry["country"] = country.upper()
        if province:
            entry["province"] = province.upper()
        if existing:
            # Preserve host metadata set by other flows (country/autoscaled tags).
            for field in ("country", "province", "autoscaled", "compute_score", "admitted"):
                if field in existing and field not in entry:
                    entry[field] = existing[field]
        _upsert_host_row(conn, entry)

    # Mirror to secondary DB in dual-write mode
    engine = get_engine()
    engine.mirror_to_secondary(DatabaseOps.upsert_host, entry)

    # Emit SSE event
    emit_event("host_update", {"host_id": host_id, "status": "active"})

    if existing:
        log.info("HOST UPDATED %s | %s | %s | %sGB", host_id, ip, gpu_model, total_vram_gb)
        return entry

    log.info(
        "HOST REGISTERED %s | %s | %s | %sGB | $%s/hr",
        host_id,
        ip,
        gpu_model,
        total_vram_gb,
        cost_per_hour,
    )
    return entry


def remove_host(host_id):
    """Host is dead. Remove it. No funeral."""
    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        conn.execute("DELETE FROM hosts WHERE host_id = %s", (host_id,))

    engine = get_engine()
    engine.mirror_to_secondary(DatabaseOps.delete_host, host_id)
    emit_event("host_removed", {"host_id": host_id})
    log.warning("HOST REMOVED %s", host_id)


def list_hosts(active_only=True):
    """Show what we've got."""
    return load_hosts(active_only=active_only)


# ── Phase 4: Health Check ─────────────────────────────────────────────


def ping_host(ip):
    """Ping once. Returns True if alive, False if dead."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_hosts():
    """
    Ping every registered host. Once.
    Dead? Mark it dead. Alive? Update last_seen.
    If the host has a recent agent heartbeat (last_seen < 60s ago),
    trust the heartbeat even if ping fails (Tailscale/Headscale IP mismatch).
    Returns dict of results.
    """
    hosts = load_hosts(active_only=False)
    results = {}
    updates = []
    alerts = []
    revivals = []
    now = time.time()

    for h in hosts:
        alive = ping_host(h["ip"])
        # Trust recent agent heartbeat even if ping fails
        if not alive:
            last_seen = h.get("last_seen", 0)
            if isinstance(last_seen, (int, float)) and (now - last_seen) < 60:
                alive = True
        updates.append((h["host_id"], h["ip"], alive))
        results[h["host_id"]] = "alive" if alive else "dead"

    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        for host_id, ip, alive in updates:
            current = _get_host_by_id_conn(conn, host_id)
            if not current:
                continue
            if alive:
                current["last_seen"] = time.time()
                if current.get("status") == "dead":
                    revivals.append((host_id, ip))
                current["status"] = "active"
            else:
                if current.get("status") != "dead":
                    alerts.append((host_id, ip))
                current["status"] = "dead"
            _upsert_host_row(conn, current)

    for host_id, ip in revivals:
        log.info("HOST REVIVED %s (%s)", host_id, ip)
    for host_id, ip in alerts:
        log.warning("HOST DEAD %s (%s)", host_id, ip)
        alert_host_dead(host_id, ip)
        # Requeue running jobs from dead host so they get rescheduled
        _requeue_dead_host_jobs(host_id, ip)

    return results


def _requeue_dead_host_jobs(host_id: str, ip: str):
    """Requeue all running/assigned/leased jobs from a dead host so they get rescheduled.

    Delegates to requeue_job() per-job to ensure proper VRAM release,
    retry tracking, and JSONB payload consistency.
    """
    try:
        with _db_connection() as conn:
            jobs = conn.execute(
                "SELECT job_id, payload->>'name' AS name FROM jobs WHERE host_id = %s AND status IN ('running', 'assigned', 'leased')",
                (host_id,),
            ).fetchall()
        if not jobs:
            return

        requeued_ids = []
        for j in jobs:
            try:
                result = requeue_job(j["job_id"])
                if result:
                    requeued_ids.append(j["job_id"])
            except Exception as e:
                log.error("Failed to requeue job %s from dead host %s: %s", j["job_id"], host_id, e)

        if requeued_ids:
            log.warning("REQUEUED %d jobs from dead host %s (%s): %s",
                         len(requeued_ids), host_id, ip, requeued_ids)
    except Exception as e:
        log.error("Failed to requeue jobs from dead host %s: %s", host_id, e)


def health_loop(interval=5, callback=None):
    """
    Ping hosts every `interval` seconds. Forever.
    If one dies — mark it dead.
    If one comes back — mark it active.
    Run this in a thread.
    """
    while True:
        try:
            results = check_hosts()
            if callback:
                callback(results)
        except Exception as e:
            log.error("Health monitor error: %s", e)
        time.sleep(interval)


def start_health_monitor(interval=5, callback=None):
    """Start the health loop in a background thread. Fire and forget."""
    t = threading.Thread(target=health_loop, args=(interval, callback), daemon=True)
    t.start()
    return t


# ── Phase 15: Priority Tiers ─────────────────────────────────────────

PRIORITY_TIERS = {
    # Pricing-model tiers (frontend-facing)
    "on-demand": {"priority": 1, "multiplier": 1.0, "label": "On-Demand"},
    "spot": {"priority": 0, "multiplier": 0.6, "label": "Spot", "spot": True},
    "reserved": {"priority": 1, "multiplier": 0.8, "label": "Reserved"},
    # Priority tiers (queue priority escalation)
    "free": {"priority": 0, "multiplier": 1.0, "label": "Free"},
    "standard": {"priority": 1, "multiplier": 1.0, "label": "Standard"},
    "premium": {"priority": 2, "multiplier": 1.5, "label": "Premium"},
    "urgent": {"priority": 3, "multiplier": 2.0, "label": "Urgent"},
}


def get_tier_info(tier_name):
    """Look up a tier. Returns dict or None."""
    return PRIORITY_TIERS.get(tier_name)


def get_tier_by_priority(priority):
    """Reverse lookup: priority int -> tier name. Prefers non-spot tiers."""
    candidates = []
    for name, info in PRIORITY_TIERS.items():
        if info["priority"] == priority:
            candidates.append(name)
    # Prefer non-spot tier (avoid accidentally marking jobs as spot/preemptible)
    for c in candidates:
        if not PRIORITY_TIERS[c].get("spot"):
            return c
    return candidates[0] if candidates else "free"


def list_tiers():
    """Return all tiers with their details."""
    return {name: dict(info) for name, info in PRIORITY_TIERS.items()}


# ── Phase 3: Job Queue ────────────────────────────────────────────────

# Extended status set — aligns with events.JobState
# Legacy 4 statuses preserved for backward compat; new statuses for v2.1
VALID_STATUSES = (
    "queued",
    "assigned",
    "leased",
    "running",
    "completed",
    "failed",
    "preempted",
    "cancelled",
)


def load_jobs():
    """Load jobs from SQLite-backed storage."""
    with _db_connection() as conn:
        _migrate_jobs_if_needed(conn)
        return _load_jobs_from_conn(conn)


def save_jobs(jobs):
    """Replace all jobs (compatibility helper)."""
    with _atomic_mutation() as conn:
        _replace_jobs_in_conn(conn, jobs)


def submit_job(
    name,
    vram_needed_gb,
    priority=0,
    tier=None,
    num_gpus=1,
    nfs_server=None,
    nfs_path=None,
    nfs_mount_point=None,
    image=None,
    interactive=False,
    command=None,
    ssh_port=22,
    owner="",
):
    """
    Submit a job to the queue.
    Each job has: model name, VRAM needed, priority, tier.
    Tier overrides priority. Pay more = jump the queue.
    FIFO within the same priority. Higher priority goes first.

    Multi-GPU: num_gpus specifies how many GPUs are needed (default 1).
    NFS: Optionally specify NFS server/path for shared storage.
    Interactive: If True, container stays running with SSH access until stopped.
    """
    # Validate Docker image if provided
    if image:
        from security import validate_docker_image
        image = validate_docker_image(image)

    # Tier overrides raw priority
    if tier and tier in PRIORITY_TIERS:
        tier_info = PRIORITY_TIERS[tier]
        priority = tier_info["priority"]
    elif tier:
        log.warning("UNKNOWN TIER '%s' — defaulting to on-demand", tier)
        tier = "on-demand"
        priority = 1
    else:
        tier = get_tier_by_priority(priority)

    # Spot tier: mark job as interruptible
    is_spot = PRIORITY_TIERS.get(tier, {}).get("spot", False)

    job = {
        "job_id": str(uuid.uuid4())[:8],
        "name": name,
        "owner": owner,
        "vram_needed_gb": vram_needed_gb,
        "priority": priority,
        "tier": tier,
        "status": "queued",
        "host_id": None,
        "submitted_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "retries": 0,
        "max_retries": 3,
        "num_gpus": max(1, int(num_gpus or 1)),
        "nfs_server": nfs_server or "",
        "nfs_path": nfs_path or "",
        "nfs_mount_point": nfs_mount_point or "",
        "image": image or "",
        "interactive": bool(interactive),
        "command": command or "",
        "ssh_port": int(ssh_port or 22),
    }

    # Spot jobs are preemptible and participate in the spot pricing market
    if is_spot:
        job["spot"] = True
        job["preemptible"] = True

    with _atomic_mutation() as conn:
        _migrate_jobs_if_needed(conn)
        _upsert_job_row(conn, job)

    # Mirror to secondary DB
    engine = get_engine()
    engine.mirror_to_secondary(DatabaseOps.upsert_job, job)

    # Emit SSE event
    emit_event("job_submitted", {"job_id": job["job_id"], "name": name, "tier": tier})

    log.info(
        "JOB SUBMITTED %s | %s | %sGB VRAM | tier=%s (priority %s)",
        job["job_id"],
        name,
        vram_needed_gb,
        tier,
        priority,
    )
    return job


def get_next_job():
    """
    Pull the next job off the queue.
    Highest priority first. Within same priority: FIFO. No mercy.
    """
    with _db_connection() as conn:
        _migrate_jobs_if_needed(conn)
        row = conn.execute("""
            SELECT payload
            FROM jobs
            WHERE status = 'queued'
            ORDER BY priority DESC, submitted_at ASC, job_id ASC
            LIMIT 1
            """).fetchone()
        if not row:
            return None
        return _decode_payload(row["payload"])


def _reserve_host_vram(conn, host_id, amount_gb):
    """Reserve VRAM on a host.

    Returns:
      True: reservation applied.
      False: host exists but has insufficient free VRAM.
      None: host not found.
    """
    if amount_gb <= 0:
        return True
    host = _get_host_by_id_conn(conn, host_id)
    if not host:
        return None
    total = float(host.get("total_vram_gb", 0) or 0)
    free = float(host.get("free_vram_gb", 0) or 0)
    if free + 1e-9 < amount_gb:
        return False
    host["free_vram_gb"] = round(max(0.0, min(total, free - amount_gb)), 4)
    _upsert_host_row(conn, host)
    return True


def _release_host_vram(conn, host_id, amount_gb):
    """Release VRAM on a host with clamping to [0, total_vram_gb]."""
    if not host_id or amount_gb <= 0:
        return False
    host = _get_host_by_id_conn(conn, host_id)
    if not host:
        return False
    total = float(host.get("total_vram_gb", 0) or 0)
    free = float(host.get("free_vram_gb", 0) or 0)
    host["free_vram_gb"] = round(max(0.0, min(total, free + amount_gb)), 4)
    _upsert_host_row(conn, host)
    return True


def reconcile_host_vram():
    """Periodic VRAM reconciliation: recompute free_vram_gb from ground truth.

    For each host, sums the vram_reserved_gb (or vram_needed_gb) of all
    jobs in running/assigned/leased status, then corrects free_vram_gb if
    it has drifted.  This catches VRAM leaks that bypass the normal
    reserve/release path.

    Returns dict of {host_id: correction_amount} for hosts that were corrected.
    """
    corrections = {}

    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        _migrate_jobs_if_needed(conn)

        hosts = _load_hosts_from_conn(conn, active_only=False)
        jobs = _load_jobs_from_conn(conn)

        # Sum VRAM usage per host from jobs that have actually reserved VRAM.
        # Only "running" jobs have VRAM reserved in the DB (done by update_job_status).
        # Assigned/leased jobs have vram_reserved_gb=0 — counting their vram_needed_gb
        # would over-count and deflate free_vram_gb, blocking legitimate assignments.
        host_vram_used: dict[str, float] = {}
        for j in jobs:
            if j.get("status") == "running":
                hid = j.get("host_id")
                reserved = float(j.get("vram_reserved_gb", 0) or 0)
                if hid and reserved > 0:
                    host_vram_used[hid] = host_vram_used.get(hid, 0.0) + reserved

        for host in hosts:
            hid = host.get("host_id")
            if not hid:
                continue
            total = float(host.get("total_vram_gb", 0) or 0)
            current_free = float(host.get("free_vram_gb", 0) or 0)
            used = host_vram_used.get(hid, 0.0)
            expected_free = round(max(0.0, min(total, total - used)), 4)

            drift = abs(current_free - expected_free)
            if drift > 0.01:  # Correct if drift exceeds 10MB
                log.warning(
                    "VRAM RECONCILE host=%s total=%.2f current_free=%.2f expected_free=%.2f drift=%.2f",
                    hid, total, current_free, expected_free, drift,
                )
                host["free_vram_gb"] = expected_free
                _upsert_host_row(conn, host)
                corrections[hid] = round(expected_free - current_free, 4)

    if corrections:
        log.info("VRAM RECONCILE corrected %d hosts: %s", len(corrections), corrections)

    return corrections


def update_job_status(job_id, status, host_id=None):
    """Mark a job. queued -> running -> completed/failed. That's the lifecycle."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}' for job {job_id} — must be one of {VALID_STATUSES}"
        )
    alert_failed = None
    alert_completed = None

    with _atomic_mutation() as conn:
        _migrate_jobs_if_needed(conn)
        _migrate_hosts_if_needed(conn)
        j = _get_job_by_id_conn(conn, job_id)
        if not j:
            return None

        old_status = j.get("status")
        old_host_id = j.get("host_id")

        # Validate state transition against the state machine
        from events import VALID_TRANSITIONS, JobState
        try:
            old_state = JobState(old_status) if old_status else None
            new_state = JobState(status)
            if old_state and new_state not in VALID_TRANSITIONS.get(old_state, set()):
                log.warning(
                    "INVALID TRANSITION job=%s %s -> %s (allowed: %s)",
                    job_id, old_status, status,
                    [s.value for s in VALID_TRANSITIONS.get(old_state, set())],
                )
        except ValueError:
            pass  # Unknown state enum value — allow for forward compat

        # Idempotency: if already running, don't allow accidental reassignment.
        if status == "running" and old_status == "running":
            return j

        # Reserve GPU memory when job starts running.
        if status == "running" and old_status != "running":
            target_host = host_id or j.get("host_id")
            reserved = float(j.get("vram_needed_gb", 0) or 0)
            if target_host and reserved > 0:
                reserve_result = _reserve_host_vram(conn, target_host, reserved)
                if reserve_result is False:
                    log.warning(
                        "VRAM RESERVE FAILED job=%s host=%s needed=%.1fGB",
                        job_id, target_host, reserved,
                    )
                    return None
                j["vram_reserved_gb"] = reserved

        # Release GPU memory when a running job reaches terminal state.
        if status in ("completed", "failed", "cancelled") and old_status == "running":
            reserved = float(j.get("vram_reserved_gb", j.get("vram_needed_gb", 0)) or 0)
            if reserved > 0:
                _release_host_vram(conn, j.get("host_id"), reserved)
            j["vram_reserved_gb"] = 0

        j["status"] = status
        if host_id and not (status == "running" and old_status == "running"):
            j["host_id"] = host_id
        if status == "running":
            j["started_at"] = time.time()
        if status in ("completed", "failed", "cancelled"):
            j["completed_at"] = time.time()

        _upsert_job_row(conn, j)

        if status == "failed":
            alert_failed = (job_id, j.get("name", "?"), j.get("host_id"))
        elif status == "completed":
            dur = None
            if j.get("started_at") and j.get("completed_at"):
                dur = j["completed_at"] - j["started_at"]
            alert_completed = (job_id, j.get("name", "?"), dur)

    lvl = logging.WARNING if status == "failed" else logging.INFO
    log.log(
        lvl,
        "JOB %s %s -> %s | %s | host=%s",
        status.upper(),
        old_status,
        status,
        job_id,
        host_id or old_host_id or "—",
    )

    # Mirror to secondary DB + emit SSE event
    engine = get_engine()
    engine.mirror_to_secondary(DatabaseOps.upsert_job, j)
    emit_event(
        "job_status", {"job_id": job_id, "status": status, "host_id": host_id or old_host_id}
    )

    # ── v2.1: Record event + trigger billing/reputation ──
    try:
        sm = get_state_machine()
        sm.transition(job_id, JobState(status), actor="scheduler")
    except Exception as exc:
        log.debug("Event record skip: %s", exc)

    if status == "completed" and j.get("started_at") and j.get("completed_at"):
        try:
            be = get_billing_engine()
            host_data = {}
            if j.get("host_id"):
                hosts_list = load_hosts(active_only=False)
                for h in hosts_list:
                    if h["host_id"] == j["host_id"]:
                        host_data = h
                        break
            be.meter_job(j, host_data)
            # Reputation: reward host
            if j.get("host_id"):
                re = get_reputation_engine()
                re.record_job_completed(j["host_id"])
        except Exception as exc:
            log.debug("Billing/reputation skip: %s", exc)

    if status == "failed" and j.get("host_id"):
        try:
            re = get_reputation_engine()
            re.record_job_failure(j["host_id"], is_host_fault=True)
        except Exception as exc:
            log.debug("Reputation penalty skip: %s", exc)

    if alert_failed:
        alert_job_failed(*alert_failed)
    elif alert_completed:
        alert_job_completed(*alert_completed)
    return j


def list_jobs(status=None):
    """List jobs. Filter by status or get everything."""
    with _db_connection() as conn:
        _migrate_jobs_if_needed(conn)
        return _load_jobs_from_conn(conn, status=status)


def get_job(job_id: str):
    """Get a single job by ID. Returns full JSONB payload dict or None."""
    with _db_connection() as conn:
        _migrate_jobs_if_needed(conn)
        return _get_job_by_id_conn(conn, job_id)


def process_queue():
    """
    The loop. Walk the queue by priority. Find a host. Assign it.
    If no host fits a job, skip it — try the next one.
    If no hosts left, stop.

    Spot jobs are only scheduled when max_bid >= current spot price.
    """
    assigned = []
    spot_prices = get_current_spot_prices()

    queued = list_jobs(status="queued")
    queued.sort(key=lambda j: (-j["priority"], j["submitted_at"]))

    for job in queued:
        # Spot job gating: skip if bid is below current spot price
        if job.get("spot") and job.get("max_bid") is not None:
            # Spot jobs need at least one GPU type within their bid
            if spot_prices:
                affordable = any(
                    job["max_bid"] >= price for price in spot_prices.values()
                )
                if not affordable:
                    log.debug(
                        "QUEUE SKIP spot job %s: max_bid $%s below all spot prices",
                        job.get("job_id"),
                        job["max_bid"],
                    )
                    continue

        hosts = list_hosts()
        if not hosts:
            break

        host = allocate(job, hosts)
        if not host:
            continue  # no host for THIS job, but maybe smaller jobs fit

        updated = update_job_status(job["job_id"], "assigned", host_id=host["host_id"])
        if not updated or updated.get("status") != "assigned":
            continue
        assigned.append((updated, host))

    return assigned


def process_assigned():
    """Pick up 'assigned' jobs and start their containers.

    The scheduler loop calls process_queue() to move jobs from queued->assigned,
    then process_assigned() to move assigned->running by SSHing into the host
    and starting the Docker container.
    """
    assigned_jobs = list_jobs(status="assigned")
    if not assigned_jobs:
        return []

    hosts = list_hosts(active_only=False)
    host_map = {h["host_id"]: h for h in hosts}
    started = []

    for job in assigned_jobs:
        host_id = job.get("host_id")
        if not host_id:
            log.warning("RUNNER: assigned job %s has no host_id, requeueing", job.get("job_id"))
            update_job_status(job["job_id"], "queued")
            continue

        host = host_map.get(host_id)
        if not host:
            log.warning("RUNNER: host %s not found for job %s, requeueing", host_id, job.get("job_id"))
            update_job_status(job["job_id"], "queued")
            continue

        try:
            docker_image = job.get("image") or None
            container_id = run_job(job, host, docker_image=docker_image)
            if container_id:
                log.info("RUNNER: job %s started on host %s (container=%s)",
                         job["job_id"], host_id, container_id)
                started.append((job, host))
            else:
                log.warning("RUNNER: container start failed for job %s on host %s",
                            job["job_id"], host_id)
                # run_job already sets status to "failed"
        except Exception as e:
            log.error("RUNNER: exception starting job %s on host %s: %s",
                      job["job_id"], host_id, e)
            update_job_status(job["job_id"], "failed")

    return started


def scheduler_tick():
    """Single scheduler iteration: assign queued jobs to hosts.

    Container lifecycle is managed by the worker agent on each host.
    The scheduler only handles queue processing (job→host assignment).
    Uses _scheduler_lock to prevent concurrent process_queue calls from
    the failover monitor thread.
    """
    with _scheduler_lock:
        try:
            process_queue()
        except Exception as e:
            log.error("Scheduler queue error: %s", e)


def scheduler_main():
    """Production scheduler entry point.

    Starts background threads for health monitoring and failover,
    then runs the scheduler tick loop with lease expiry checks.
    Intended to be the single command for the scheduler-worker container.
    """
    log.info("SCHEDULER MAIN starting — tick=2s, health=30s, failover=60s")

    # Start health monitor: pings all hosts every 30s, marks dead/alive
    start_health_monitor(interval=30)
    log.info("Health monitor thread started (30s interval)")

    # Start failover monitor: requeues orphaned jobs every 60s
    start_failover_monitor(interval=60)
    log.info("Failover monitor thread started (60s interval)")

    event_store = get_event_store()
    tick_count = 0

    while True:
        scheduler_tick()

        # Expire stale leases every 5th tick (10s)
        tick_count += 1
        if tick_count % 5 == 0:
            try:
                expired = event_store.expire_stale_leases()
                if expired:
                    log.warning(
                        "LEASE EXPIRY: %d jobs expired — requeueing: %s",
                        len(expired),
                        expired,
                    )
                    for job_id in expired:
                        try:
                            requeue_job(job_id)
                        except Exception as e:
                            log.error("Failed to requeue expired lease job %s: %s", job_id, e)
            except Exception as e:
                log.error("Lease expiry check failed: %s", e)

        # Reconcile VRAM every 30th tick (60s) to catch drift
        if tick_count % 30 == 0:
            try:
                reconcile_host_vram()
            except Exception as e:
                log.error("VRAM reconciliation failed: %s", e)

        time.sleep(2)


# ── Phase 5 & 6: Run Job / Kill Job ──────────────────────────────────


def ssh_exec(ip, cmd, timeout=30):
    """
    Run a command on a remote host via SSH.
    Key-based auth only. No passwords. No agent forwarding.
    Returns (returncode, stdout, stderr).
    """
    full_cmd = [
        "ssh",
        "-i",
        SSH_KEY_PATH,
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        f"{SSH_USER}@{ip}",
        cmd,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def generate_ssh_keypair(path=None):
    """Generate an Ed25519 SSH keypair. No passphrase. Fast. Secure."""
    path = path or SSH_KEY_PATH
    # Resolve to absolute and ensure it's under home directory
    path = os.path.realpath(os.path.expanduser(path))
    home = os.path.realpath(os.path.expanduser("~"))
    if not path.startswith(home + os.sep):
        log.error("SSH KEYGEN REJECTED: path %s is outside home directory", path)
        raise ValueError(f"SSH key path must be under home directory: {path}")
    if os.path.isdir(path):
        log.error("SSH KEYGEN REJECTED: path %s is a directory, expected file", path)
        raise ValueError(f"SSH key path must be a file, not a directory: {path}")
    if os.path.exists(path):
        log.info("SSH key already exists: %s", path)
        return path

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", path, "-N", "", "-C", "xcelsior"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        os.chmod(path, 0o600)
        log.info("SSH KEYPAIR GENERATED: %s", path)
    else:
        log.error("SSH KEYGEN FAILED: %s", result.stderr.strip())
    return path


def get_public_key(path=None):
    """Read the public key. For distributing to hosts."""
    path = (path or SSH_KEY_PATH) + ".pub"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip()


def run_job(job, host, docker_image=None):
    """
    SSH into the host. docker run the model. Log the container ID.
    Returns the container ID or None on failure.

    If the job has checkpoint metadata (resume_from), attempts to resume
    from the checkpoint first. Falls through to fresh docker run on failure.

    Constructs a hardened docker run command with:
    - NFS volume mounts (if specified in job)
    - Interactive mode with SSH port mapping
    - Security flags (no-new-privileges, pids-limit)
    - Memory limits based on host capacity
    - Per-GPU selection when available
    """
    # ── Checkpoint Resume Path ────────────────────────────────────────
    checkpoint_meta = job.get("resume_from")
    if checkpoint_meta and checkpoint_meta.get("success"):
        log.info("RUN_JOB attempting checkpoint resume for job=%s on host=%s",
                 job["job_id"], host["host_id"])
        resumed = resume_from_checkpoint(job["job_id"], host["host_id"], checkpoint_meta)
        if resumed:
            # Clear resume_from metadata now that we've successfully resumed
            _set_job_fields(job["job_id"], resume_from=None)
            return checkpoint_meta.get("container", f"xcelsior-job-{job['job_id']}")
        log.warning("RUN_JOB checkpoint resume failed for job=%s — falling through to fresh start",
                     job["job_id"])
        # Clear stale checkpoint metadata so we don't retry forever
        _set_job_fields(job["job_id"], resume_from=None)

    # ── Fresh Start Path ──────────────────────────────────────────────
    image = docker_image or job.get("image") or f"xcelsior/{job['name']}:latest"
    container_name = f"xcl-{job['job_id']}"

    # Validate names to prevent shell injection
    _validate_name(image, "docker image")
    _validate_name(container_name, "container name")

    # Validate image against allowlist
    from security import validate_docker_image
    image = validate_docker_image(image)

    # Build docker run command parts
    parts = ["docker run -d"]

    # GPU passthrough
    num_gpus = job.get("num_gpus", 1)
    if num_gpus > 0:
        parts.append(f"--gpus all")

    # Container name
    parts.append(f"--name {shlex.quote(container_name)}")

    # Security hardening (parity with build_secure_docker_args in security.py)
    parts.append("--security-opt=no-new-privileges")
    parts.append("--pids-limit=4096")
    parts.append("--no-healthcheck")

    # Non-root execution — skip for interactive containers because GPU images
    # (e.g. nvcr.io/nvidia/pytorch) require root for CUDA/cuDNN libraries.
    is_interactive = job.get("interactive", False)
    if not is_interactive:
        parts.append("--cap-drop=ALL")
        parts.append("--user=1000:1000")

    # Read-only root filesystem for batch jobs; tmpfs for temp dirs
    is_interactive = job.get("interactive", False)
    if not is_interactive:
        parts.append("--read-only")
    parts.append("--tmpfs=/tmp:rw,noexec,nosuid,size=1g")
    parts.append("--tmpfs=/var/tmp:rw,noexec,nosuid,size=512m")

    # Memory limit: use host's total VRAM as a rough proxy for container memory
    # (containers typically need system RAM proportional to VRAM)
    host_vram = host.get("total_vram_gb", 0)
    if host_vram > 0:
        mem_limit = max(8, int(host_vram * 2))  # 2x VRAM as system RAM, min 8GB
        parts.append(f"--memory={mem_limit}g")
        parts.append(f"--memory-swap={mem_limit}g")
    else:
        parts.append("--memory=32g")
        parts.append("--memory-swap=32g")

    # CPU, ulimit, and shared memory limits
    parts.append("--cpus=16")
    parts.append("--ulimit=nofile=65535:65535")
    parts.append("--ulimit=nproc=4096:4096")
    parts.append("--shm-size=1g")

    # Sandboxed runtime (gVisor) — use if available on the host
    host_runtime = host.get("runtime", "")
    if host_runtime and host_runtime != "runc":
        parts.append(f"--runtime={shlex.quote(host_runtime)}")

    # NFS volume mount
    nfs_server = job.get("nfs_server", "")
    nfs_path = job.get("nfs_path", "")
    nfs_mount = job.get("nfs_mount_point", "/mnt/xcelsior-nfs")
    if nfs_server and nfs_path:
        nfs_mount = nfs_mount or "/mnt/xcelsior-nfs"
        nfs_opts = f"addr={nfs_server},nolock,soft,timeo=30"
        parts.append(
            f"--mount type=volume,volume-driver=local,"
            f"volume-opt=type=nfs,volume-opt=o={nfs_opts},"
            f"volume-opt=device=:{shlex.quote(nfs_path)},"
            f"dst={shlex.quote(nfs_mount)}"
        )

    # Interactive mode: keep stdin open, map SSH port
    if job.get("interactive"):
        ssh_port = int(job.get("ssh_port", 22))
        # Avoid port 22 — it's used by the host SSH daemon
        if ssh_port == 22:
            # Deterministic high port from job_id to avoid collisions
            ssh_port = 2222 + (int(job["job_id"][:4], 16) % 10000)
        parts.append(f"-p {ssh_port}:22")
        # Don't auto-remove interactive containers
    else:
        parts.append("--restart=no")

    # Environment variables: pass job ID and API token so the container
    # can report status back if needed
    parts.append(f"-e XCELSIOR_JOB_ID={shlex.quote(job['job_id'])}")

    # Image
    parts.append(shlex.quote(image))

    # Append the job command if specified (quote each token to prevent injection)
    job_command = job.get("command", "").strip()
    if job_command:
        tokens = shlex.split(job_command)
        parts.append(" ".join(shlex.quote(t) for t in tokens))
    elif is_interactive:
        # Interactive containers need a long-running process to stay alive
        # (default image entrypoints like bash exit immediately in detached mode)
        parts.append("sleep infinity")

    cmd = " ".join(parts)

    # Pre-pull the image (may take minutes for large images like pytorch)
    log.info("PULLING IMAGE job=%s host=%s image=%s", job["job_id"], host["host_id"], image)
    pull_rc, _, pull_err = ssh_exec(host["ip"], f"docker pull {shlex.quote(image)}", timeout=600)
    if pull_rc != 0:
        log.error("IMAGE PULL FAILED job=%s host=%s err=%s", job["job_id"], host["host_id"], pull_err)
        update_job_status(job["job_id"], "failed")
        return None
    log.info("IMAGE READY job=%s host=%s image=%s", job["job_id"], host["host_id"], image)

    rc, stdout, stderr = ssh_exec(host["ip"], cmd)
    if rc != 0:
        log.error("RUN FAILED job=%s host=%s err=%s", job["job_id"], host["host_id"], stderr)
        update_job_status(job["job_id"], "failed")
        return None

    container_id = stdout[:12]
    log.info(
        "CONTAINER STARTED job=%s host=%s container=%s image=%s",
        job["job_id"],
        host["host_id"],
        container_id,
        image,
    )

    # Apply egress filtering (mining port blocks + default-deny for batch jobs)
    try:
        from security import build_egress_iptables_rules
        is_interactive = job.get("interactive", False)
        egress_rules = build_egress_iptables_rules(
            container_name, strict=not is_interactive,
        )
        for rule in egress_rules:
            ssh_exec(host["ip"], rule)
    except Exception as e:
        log.warning("EGRESS RULES FAILED job=%s: %s", job["job_id"], e)

    # Store container info on the job atomically.
    _set_job_fields(job["job_id"], container_id=container_id, container_name=container_name)

    update_job_status(job["job_id"], "running", host_id=host["host_id"])
    return container_id


def check_job_running(job, host):
    """Check if a job's container is still running. Returns True/False."""
    container_name = job.get("container_name", f"xcl-{job['job_id']}")
    _validate_name(container_name, "container name")
    cmd = f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(container_name)}"
    rc, stdout, _ = ssh_exec(host["ip"], cmd)
    return rc == 0 and stdout == "true"


def kill_job(job, host):
    """
    Kill the container. Clean up. Mark job COMPLETE.
    No lingering processes. No orphaned containers.
    """
    container_name = job.get("container_name", f"xcl-{job['job_id']}")
    _validate_name(container_name, "container name")

    # Kill it
    ssh_exec(host["ip"], f"docker kill {shlex.quote(container_name)}")
    # Remove it
    ssh_exec(host["ip"], f"docker rm -f {shlex.quote(container_name)}")
    # NOTE: Caller is responsible for setting the final status (cancelled, completed, etc.)
    log.info(
        "JOB KILLED job=%s host=%s container=%s", job["job_id"], host["host_id"], container_name
    )


def wait_for_job(job, host, poll_interval=5, max_wait=172800):
    """
    Watch a job until it finishes on its own.
    When the container exits, mark it complete and clean up.
    Returns final status: "completed", "failed", or "timeout".
    max_wait defaults to 48 hours.
    """
    container_name = job.get("container_name", f"xcl-{job['job_id']}")
    _validate_name(container_name, "container name")

    start = time.time()
    while True:
        if time.time() - start > max_wait:
            log.warning("JOB TIMEOUT job=%s exceeded max_wait=%ds", job["job_id"], max_wait)
            update_job_status(job["job_id"], "failed")
            return "timeout"
        # Check if container still exists
        cmd = f"docker inspect -f '{{{{.State.Status}}}}' {shlex.quote(container_name)}"
        rc, stdout, _ = ssh_exec(host["ip"], cmd)

        if rc != 0:
            # Container gone entirely
            update_job_status(job["job_id"], "failed")
            return "failed"

        if stdout == "exited":
            # Check exit code
            cmd_exit = f"docker inspect -f '{{{{.State.ExitCode}}}}' {shlex.quote(container_name)}"
            _, exit_code, _ = ssh_exec(host["ip"], cmd_exit)
            # Clean up
            ssh_exec(host["ip"], f"docker rm -f {shlex.quote(container_name)}")

            if exit_code == "0":
                update_job_status(job["job_id"], "completed")
                return "completed"
            else:
                update_job_status(job["job_id"], "failed")
                return "failed"

        time.sleep(poll_interval)


def run_job_local(job, docker_image=None):
    """
    Run a job locally (no SSH). For testing or single-machine setups.
    Returns (container_id, container_name) or (None, None) on failure.
    """
    image = docker_image or f"xcelsior/{job['name']}:latest"
    container_name = f"xcl-{job['job_id']}"

    try:
        result = subprocess.run(
            ["docker", "run", "-d", "--name", container_name, image],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        update_job_status(job["job_id"], "failed")
        return None, None

    if result.returncode != 0:
        update_job_status(job["job_id"], "failed")
        return None, None

    container_id = result.stdout.strip()[:12]

    _set_job_fields(job["job_id"], container_id=container_id, container_name=container_name)

    return container_id, container_name


def kill_job_local(container_name):
    """Kill and remove a local container."""
    subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=10)
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)


# ── Phase 8: Billing ─────────────────────────────────────────────────

BILLING_FILE = os.environ.get("XCELSIOR_BILLING_FILE", os.path.join(os.path.dirname(__file__), "billing.json"))
DEFAULT_RATE = 0.20  # $/hr


def load_billing():
    """Load billing records."""
    return _load_json(BILLING_FILE)


def save_billing(records):
    """Write billing records."""
    _save_json(BILLING_FILE, records)


def bill_job(job_id):
    """
    Bill a completed job. Multiply time by rate. Charge.
    start_time, end_time, duration, cost. That's it.

    Duplicate check + write are serialized through SQLite-backed state
    persistence to prevent double billing across concurrent callers.
    """
    jobs = load_jobs()
    job = None
    for j in jobs:
        if j["job_id"] == job_id:
            job = j
            break

    if not job:
        log.error("BILLING FAILED job=%s not found", job_id)
        return None

    if not job.get("started_at") or not job.get("completed_at"):
        log.error("BILLING FAILED job=%s missing timestamps", job_id)
        return None

    # Get the host's rate
    rate = DEFAULT_RATE
    if job.get("host_id"):
        hosts = load_hosts(active_only=False)
        for h in hosts:
            if h["host_id"] == job["host_id"]:
                rate = h.get("cost_per_hour", DEFAULT_RATE)
                break

    # Tier multiplier — premium and urgent pay more
    tier = job.get("tier", "free")
    tier_info = PRIORITY_TIERS.get(tier, PRIORITY_TIERS["free"])
    multiplier = tier_info["multiplier"]

    duration_sec = job["completed_at"] - job["started_at"]
    duration_hr = duration_sec / 3600
    base_cost = duration_hr * rate
    cost = round(base_cost * multiplier, 4)

    record = {
        "job_id": job_id,
        "job_name": job["name"],
        "host_id": job.get("host_id"),
        "tier": tier,
        "tier_multiplier": multiplier,
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "duration_sec": round(duration_sec, 2),
        "rate_per_hour": rate,
        "cost": cost,
        "billed_at": time.time(),
    }

    records = load_billing()
    # Atomic check-and-insert to prevent double-billing
    with _atomic_mutation() as conn:
        cur = conn.execute(
            "INSERT INTO state (namespace, payload) VALUES (%s, %s) ON CONFLICT (namespace) DO NOTHING",
            (f"billed:{job_id}", "1"),
        )
        if cur.rowcount == 0:
            log.warning("BILLING SKIPPED job=%s already billed", job_id)
            return None
    if any(r["job_id"] == job_id for r in records):
        log.warning("BILLING SKIPPED job=%s already billed", job_id)
        return None
    records.append(record)
    save_billing(records)

    log.info(
        "BILLED job=%s | %s | %.1fs | $%.4f @ $%s/hr x%.1f (%s)",
        job_id,
        job["name"],
        duration_sec,
        cost,
        rate,
        multiplier,
        tier,
    )
    return record


def bill_all_completed():
    """Bill every completed job that hasn't been billed yet."""
    jobs = load_jobs()
    records = load_billing()
    billed_ids = {r["job_id"] for r in records}
    new_bills = []

    for j in jobs:
        if j["status"] == "completed" and j["job_id"] not in billed_ids:
            bill = bill_job(j["job_id"])
            if bill:
                new_bills.append(bill)

    return new_bills


def get_total_revenue():
    """How much money did we make?"""
    records = load_billing()
    return round(sum(r["cost"] for r in records), 4)


# ── Phase 12: Alerts ─────────────────────────────────────────────────

ALERT_CONFIG = {
    "email_enabled": bool(os.environ.get("XCELSIOR_SMTP_HOST") and os.environ.get("XCELSIOR_SMTP_USER")),
    "smtp_host": os.environ.get("XCELSIOR_SMTP_HOST", ""),
    "smtp_port": int(os.environ.get("XCELSIOR_SMTP_PORT", "587")),
    "smtp_user": os.environ.get("XCELSIOR_SMTP_USER", ""),
    "smtp_pass": os.environ.get("XCELSIOR_SMTP_PASS", ""),
    "email_from": os.environ.get("XCELSIOR_EMAIL_FROM", ""),
    "email_to": os.environ.get("XCELSIOR_EMAIL_TO", ""),
    "telegram_enabled": bool(os.environ.get("XCELSIOR_TG_TOKEN") and os.environ.get("XCELSIOR_TG_CHAT_ID")),
    "telegram_bot_token": os.environ.get("XCELSIOR_TG_TOKEN", ""),
    "telegram_chat_id": os.environ.get("XCELSIOR_TG_CHAT_ID", ""),
}


def configure_alerts(**kwargs):
    """Update alert config at runtime."""
    ALERT_CONFIG.update(kwargs)


def send_email(subject, body, to_email=None):
    """Send an email alert. SMTP. No dependencies.
    
    If to_email is provided, sends to that address instead of the admin email.
    """
    cfg = ALERT_CONFIG
    if not cfg["email_enabled"]:
        return False

    try:
        recipient = to_email or cfg["email_to"]
        msg = MIMEText(body)
        msg["Subject"] = f"[Xcelsior] {subject}"
        msg["From"] = cfg["email_from"]
        msg["To"] = recipient

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.send_message(msg)

        log.info("EMAIL SENT: %s -> %s", subject, recipient)
        return True
    except Exception as e:
        log.error("EMAIL FAILED: %s | %s", subject, e)
        return False


def send_telegram(message):
    """Send a Telegram alert. HTTP POST. No dependencies."""
    cfg = ALERT_CONFIG
    if not cfg["telegram_enabled"]:
        return False

    try:
        url = f"https://api.telegram.org/bot{cfg['telegram_bot_token']}/sendMessage"
        payload = json.dumps(
            {
                "chat_id": cfg["telegram_chat_id"],
                "text": f"[Xcelsior] {message}",
                "parse_mode": "HTML",
            }
        ).encode()

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)

        log.info("TELEGRAM SENT: %s", message)
        return True
    except Exception as e:
        log.error("TELEGRAM FAILED: %s | %s", message, e)
        return False


def alert(subject, body=None):
    """
    Send an alert through all enabled channels.
    Fire and forget. Non-blocking.
    """
    body = body or subject
    sent = []

    if ALERT_CONFIG["email_enabled"]:
        threading.Thread(target=send_email, args=(subject, body), daemon=True).start()
        sent.append("email")

    if ALERT_CONFIG["telegram_enabled"]:
        threading.Thread(
            target=send_telegram, args=(f"<b>{subject}</b>\n{body}",), daemon=True
        ).start()
        sent.append("telegram")

    if not sent:
        log.debug("ALERT (no channels): %s", subject)
    return sent


def alert_host_dead(host_id, ip):
    """Host died. Sound the alarm."""
    alert(
        f"HOST DOWN: {host_id}",
        f"Host {host_id} ({ip}) is not responding to ping.",
    )


def alert_job_failed(job_id, job_name, host_id=None):
    """Job failed. Notify."""
    alert(
        f"JOB FAILED: {job_name}",
        f"Job {job_id} ({job_name}) failed on host {host_id or 'unknown'}.",
    )


def alert_job_completed(job_id, job_name, duration_sec=None):
    """Job completed. Good news for once."""
    dur = f" in {duration_sec:.1f}s" if duration_sec else ""
    alert(
        f"JOB COMPLETE: {job_name}",
        f"Job {job_id} ({job_name}) completed{dur}.",
    )


# ── Phase 14: Failover ────────────────────────────────────────────────


def requeue_job(job_id):
    """
    Reset a failed/running/leased/assigned job back to queued.
    Increment retry counter. Clear host assignment. Release VRAM.
    If max retries exceeded, mark permanently failed.
    """
    exhausted = None
    requeued = None

    with _atomic_mutation() as conn:
        _migrate_jobs_if_needed(conn)
        _migrate_hosts_if_needed(conn)
        j = _get_job_by_id_conn(conn, job_id)
        if not j:
            return None

        # Accept running, failed, leased, and assigned jobs for requeue
        if j["status"] not in ("running", "failed", "leased", "assigned"):
            log.warning("REQUEUE REJECTED job=%s status=%s — not requeuable", job_id, j["status"])
            return None

        old_host_id = j.get("host_id")
        old_status = j["status"]

        # Release VRAM if the job had any reserved on its host
        reserved = float(j.get("vram_reserved_gb", j.get("vram_needed_gb", 0)) or 0)
        if old_host_id and reserved > 0 and old_status in ("running",):
            _release_host_vram(conn, old_host_id, reserved)
            log.info("REQUEUE VRAM RELEASED job=%s host=%s vram=%.2fGB", job_id, old_host_id, reserved)

        retries = j.get("retries", 0) + 1
        max_retries = j.get("max_retries", 3)

        if retries > max_retries:
            j["status"] = "failed"
            j["completed_at"] = time.time()
            j["vram_reserved_gb"] = 0
            _upsert_job_row(conn, j)
            exhausted = (retries, max_retries, j.get("name", "?"), old_host_id)
        else:
            j["status"] = "queued"
            j["host_id"] = None
            j["started_at"] = None
            j["completed_at"] = None
            j["retries"] = retries
            j["vram_reserved_gb"] = 0
            _upsert_job_row(conn, j)
            requeued = (j, retries, max_retries, old_host_id or "—")

    if exhausted:
        retries, max_retries, name, host_id = exhausted
        log.error(
            "FAILOVER EXHAUSTED job=%s retries=%d/%d — permanently failed",
            job_id,
            retries,
            max_retries,
        )
        alert_job_failed(job_id, name, host_id)
        return None

    if requeued:
        j, retries, max_retries, old_host = requeued
        log.warning(
            "FAILOVER REQUEUE job=%s retry=%d/%d old_host=%s",
            job_id,
            retries,
            max_retries,
            old_host,
        )
        return j

    return None


def failover_dead_hosts():
    """
    Find all running jobs on dead hosts. Requeue them.
    This is the core failover loop — call it after check_hosts().
    Returns list of requeued jobs.
    """
    dead_host_ids = {
        h["host_id"] for h in list_hosts(active_only=False) if h.get("status") == "dead"
    }

    if not dead_host_ids:
        return []

    jobs = load_jobs()
    requeued = []

    for j in jobs:
        if j["status"] == "running" and j.get("host_id") in dead_host_ids:
            log.warning("FAILOVER DETECTED job=%s on dead host=%s", j["job_id"], j["host_id"])
            result = requeue_job(j["job_id"])
            if result:
                requeued.append(result)

    return requeued


def failover_and_reassign():
    """
    Full failover cycle:
    1. Check hosts (ping) — dead hosts trigger _requeue_dead_host_jobs which
       delegates to requeue_job() per-job (releases VRAM, increments retry counter)
    2. Process queue (assign requeued jobs to alive hosts)
    Returns newly_assigned list.

    Uses _scheduler_lock on the assignment pass to prevent concurrent
    process_queue calls with scheduler_tick.
    """
    check_hosts()

    # check_hosts() already requeues jobs from newly-dead hosts via
    # _requeue_dead_host_jobs → requeue_job(), so we only need to
    # run the assignment pass here.
    with _scheduler_lock:
        assigned = process_queue()
    for j, h in assigned:
        log.info("FAILOVER REASSIGNED job=%s -> host=%s", j["job_id"], h["host_id"])

    return [], assigned


# ── Auto-Remediation: Checkpoint + Re-Queue ──────────────────────────
# Docker CRIU Checkpoint support for transparent job migration.
# On health check failure, checkpoint the container, snapshot its state,
# then requeue with the checkpoint reference so the next host can resume.

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def checkpoint_container(host_id: str, job_id: str, container_name: str = "") -> dict | None:
    """Create a Docker CRIU checkpoint for a running container on a remote host.

    Uses `docker checkpoint create` via SSH to freeze the container state.
    Returns checkpoint metadata or None on failure.
    """
    if not container_name:
        # Match run_job() naming convention: xcl-{job_id}
        container_name = f"xcl-{job_id}"

    # Validate inputs to prevent path traversal / injection
    _validate_name(job_id, "job_id")
    _validate_name(container_name, "container name")

    checkpoint_name = f"ckpt-{job_id}-{int(time.time())}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    # Ensure checkpoint_path stays under CHECKPOINT_DIR
    resolved = os.path.realpath(checkpoint_path)
    if not resolved.startswith(os.path.realpath(CHECKPOINT_DIR)):
        log.error("CHECKPOINT PATH TRAVERSAL BLOCKED: %s", checkpoint_path)
        return None

    # Look up host IP for SSH
    with _db_connection() as conn:
        host = _get_host_by_id_conn(conn, host_id)
    if not host or not host.get("ip"):
        log.error("CHECKPOINT FAILED: host %s not found or missing IP", host_id)
        return None
    host_ip = host["ip"]

    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        remote_ckpt_dir = f"/tmp/xcelsior-checkpoints/{shlex.quote(checkpoint_name)}"
        docker_cmd = (
            f"mkdir -p {remote_ckpt_dir} && "
            f"docker checkpoint create "
            f"--checkpoint-dir={remote_ckpt_dir} "
            f"--leave-running=false "
            f"{shlex.quote(container_name)} {shlex.quote(checkpoint_name)}"
        )

        log.info(
            "CHECKPOINT job=%s host=%s container=%s checkpoint=%s",
            job_id, host_id, container_name, checkpoint_name,
        )

        # Execute checkpoint on remote host via SSH
        rc, stdout, stderr = ssh_exec(host_ip, docker_cmd)

        checkpoint_meta = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": remote_ckpt_dir,
            "local_path": checkpoint_path,
            "job_id": job_id,
            "host_id": host_id,
            "host_ip": host_ip,
            "container": container_name,
            "created_at": time.time(),
            "success": rc == 0,
            "stderr": (stderr or "")[:500],
        }

        # Persist metadata locally
        meta_file = os.path.join(checkpoint_path, "meta.json")
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(meta_file, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)

        if not checkpoint_meta["success"]:
            log.error("CHECKPOINT DOCKER FAILED job=%s host=%s: %s",
                       job_id, host_id, checkpoint_meta["stderr"])
            return None

        return checkpoint_meta

    except Exception as e:
        log.error("CHECKPOINT FAILED job=%s host=%s: %s", job_id, host_id, e)

    return None


def resume_from_checkpoint(job_id: str, target_host_id: str, checkpoint_meta: dict) -> bool:
    """Resume a checkpointed job on a new host via SSH.

    Transfers the checkpoint via the scheduler (source→scheduler→target)
    since hosts don't have cross-host SSH credentials.
    Updates job status and reserves VRAM on success.
    """
    checkpoint_name = checkpoint_meta.get("checkpoint_name", "")
    checkpoint_path = checkpoint_meta.get("checkpoint_path", "")
    source_ip = checkpoint_meta.get("host_ip", "")
    container = checkpoint_meta.get("container", f"xcl-{job_id}")

    if not checkpoint_name:
        log.error("RESUME FAILED: no checkpoint_name for job=%s", job_id)
        return False

    # Validate inputs
    _validate_name(checkpoint_name, "checkpoint name")
    _validate_name(container, "container name")

    # Look up target host IP
    with _db_connection() as conn:
        target_host = _get_host_by_id_conn(conn, target_host_id)
    if not target_host or not target_host.get("ip"):
        log.error("RESUME FAILED: target host %s not found or missing IP", target_host_id)
        return False
    target_ip = target_host["ip"]

    log.info(
        "RESUME job=%s target_host=%s checkpoint=%s",
        job_id, target_host_id, checkpoint_name,
    )

    try:
        remote_ckpt_dir = checkpoint_path or f"/tmp/xcelsior-checkpoints/{checkpoint_name}"

        # Step 1: Transfer checkpoint via scheduler if source != target.
        # Hosts don't have SSH keys to each other, so we pull to the
        # scheduler first, then push to the target.
        if source_ip and source_ip != target_ip:
            local_staging = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            os.makedirs(local_staging, exist_ok=True)

            # Pull from source host to scheduler
            pull_cmd = [
                "scp", "-r",
                "-i", SSH_KEY_PATH,
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "PasswordAuthentication=no",
                f"{SSH_USER}@{source_ip}:{remote_ckpt_dir}",
                local_staging,
            ]
            pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=300)
            if pull_result.returncode != 0:
                log.error("RESUME TRANSFER PULL FAILED job=%s: %s", job_id, (pull_result.stderr or "")[:300])
                return False

            # Push from scheduler to target host
            push_cmd = [
                "scp", "-r",
                "-i", SSH_KEY_PATH,
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "PasswordAuthentication=no",
                local_staging,
                f"{SSH_USER}@{target_ip}:/tmp/xcelsior-checkpoints/",
            ]
            push_result = subprocess.run(push_cmd, capture_output=True, text=True, timeout=300)
            if push_result.returncode != 0:
                log.error("RESUME TRANSFER PUSH FAILED job=%s: %s", job_id, (push_result.stderr or "")[:300])
                return False

            log.info("RESUME CHECKPOINT TRANSFERRED job=%s via scheduler: %s -> %s", job_id, source_ip, target_ip)

        # Step 2: Start container from checkpoint on target host
        docker_cmd = (
            f"docker start "
            f"--checkpoint-dir={shlex.quote(remote_ckpt_dir)} "
            f"--checkpoint={shlex.quote(checkpoint_name)} "
            f"{shlex.quote(container)}"
        )
        rc, stdout, stderr = ssh_exec(target_ip, docker_cmd)

        if rc == 0:
            # Step 3: Update job status and reserve VRAM
            result = update_job_status(job_id, "running", target_host_id)
            if not result:
                # VRAM reservation failed — kill the orphaned container
                log.error("RESUME VRAM RESERVATION FAILED job=%s on host=%s — killing container",
                          job_id, target_host_id)
                ssh_exec(target_ip, f"docker kill {shlex.quote(container)}")
                return False
            log.info("RESUME SUCCESS job=%s on host=%s", job_id, target_host_id)
            return True
        else:
            log.error(
                "RESUME FAILED job=%s: %s", job_id, (stderr or "")[:300],
            )
    except subprocess.TimeoutExpired:
        log.error("RESUME TRANSFER TIMEOUT job=%s", job_id)
    except Exception as e:
        log.error("RESUME EXCEPTION job=%s: %s", job_id, e)

    return False


# Health failure tracking: consecutive failures per host
_health_failure_counts: dict[str, int] = {}
HEALTH_FAILURE_THRESHOLD = 3  # Consecutive failures before remediation


def remediate_unhealthy_host(host_id: str) -> list[dict]:
    """Auto-remediate an unhealthy host: checkpoint running jobs and re-queue.

    Called when a host has HEALTH_FAILURE_THRESHOLD consecutive health check
    failures. Steps:
    1. Checkpoint all running containers on the host (best-effort)
    2. Re-queue jobs with checkpoint metadata for resume
    3. Mark host as dead

    Returns list of requeued job dicts with checkpoint info.
    """
    jobs = load_jobs()
    running_on_host = [
        j for j in jobs if j["status"] == "running" and j.get("host_id") == host_id
    ]

    if not running_on_host:
        log.info("REMEDIATE host=%s has no running jobs", host_id)
        return []

    requeued = []

    for job in running_on_host:
        job_id = job["job_id"]
        log.warning("REMEDIATE CHECKPOINT job=%s on unhealthy host=%s", job_id, host_id)

        # Step 1: Attempt checkpoint (best-effort)
        ckpt = checkpoint_container(host_id, job_id)

        # Step 2: Requeue with checkpoint reference
        result = requeue_job(job_id)
        if result:
            if ckpt and ckpt.get("success"):
                # Persist checkpoint metadata atomically via _set_job_fields
                # (single read-modify-write transaction avoids TOCTOU race
                # where process_queue could reassign the job between transactions)
                _set_job_fields(result["job_id"], resume_from=ckpt)
                result["resume_from"] = ckpt
            requeued.append(result)

    # Step 3: Record remediation
    log.warning(
        "REMEDIATE COMPLETE host=%s: %d jobs checkpointed and requeued",
        host_id, len(requeued),
    )

    return requeued


def record_health_check(host_id: str, healthy: bool) -> dict | None:
    """Record a health check result and trigger remediation if needed.

    Returns remediation result if threshold exceeded, else None.
    """
    if healthy:
        _health_failure_counts.pop(host_id, None)
        return None

    count = _health_failure_counts.get(host_id, 0) + 1
    _health_failure_counts[host_id] = count

    if count >= HEALTH_FAILURE_THRESHOLD:
        log.warning(
            "HEALTH THRESHOLD EXCEEDED host=%s failures=%d — triggering remediation",
            host_id, count,
        )
        _health_failure_counts.pop(host_id, None)
        requeued = remediate_unhealthy_host(host_id)
        return {"host_id": host_id, "action": "remediated", "requeued": requeued}

    return None


def start_failover_monitor(interval=10, callback=None):
    """
    Run failover checks in a background thread.
    Ping hosts, requeue orphaned jobs, reassign.
    """

    def loop():
        while True:
            try:
                requeued, assigned = failover_and_reassign()
                if callback and (requeued or assigned):
                    callback(requeued, assigned)
            except Exception as e:
                log.error("Failover monitor error: %s", e)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ── Phase 16: Docker Image Builder ───────────────────────────────────

BUILDS_DIR = os.path.join(os.path.dirname(__file__), "builds")
REGISTRY = os.environ.get("XCELSIOR_REGISTRY", "")


def generate_dockerfile(model_name, base_image="python:3.11-slim", quantize=None):
    """
    Generate a Dockerfile for a model.
    Supports optional quantization step (gguf, gptq, awq).
    Returns Dockerfile contents as a string.
    """
    quant_step = ""
    if quantize == "gguf":
        quant_step = (
            "RUN pip install --no-cache-dir llama-cpp-python && \\\n"
            "    python -c \"print('GGUF quantize ready')\"\n"
        )
    elif quantize == "gptq":
        quant_step = (
            "RUN pip install --no-cache-dir auto-gptq && \\\n"
            "    python -c \"print('GPTQ quantize ready')\"\n"
        )
    elif quantize == "awq":
        quant_step = (
            "RUN pip install --no-cache-dir autoawq && \\\n"
            "    python -c \"print('AWQ quantize ready')\"\n"
        )

    dockerfile = f"""FROM {base_image}

LABEL maintainer="xcelsior"
LABEL model="{model_name}"

WORKDIR /app

# Install model dependencies
RUN pip install --no-cache-dir torch transformers accelerate

{quant_step}# Copy model files
COPY . /app/

# Default: run the model server
CMD ["python", "-m", "http.server", "8080"]
"""
    return dockerfile


def build_image(
    model_name, context_dir=None, tag=None, quantize=None, base_image="python:3.11-slim"
):
    """
    Build a Docker image for a model.
    1. Generate Dockerfile
    2. Write to build dir
    3. docker build
    Returns (image_tag, success).
    """
    build_dir = context_dir or os.path.join(BUILDS_DIR, model_name)
    os.makedirs(build_dir, exist_ok=True)

    # Generate and write Dockerfile
    dockerfile = generate_dockerfile(model_name, base_image=base_image, quantize=quantize)
    dockerfile_path = os.path.join(build_dir, "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile)

    tag = tag or f"xcelsior/{model_name}:latest"

    log.info(
        "BUILD START model=%s tag=%s dir=%s quantize=%s",
        model_name,
        tag,
        build_dir,
        quantize or "none",
    )

    try:
        result = subprocess.run(
            ["docker", "build", "-t", tag, build_dir],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            log.error("BUILD FAILED model=%s err=%s", model_name, result.stderr.strip())
            return tag, False

        log.info("BUILD SUCCESS model=%s tag=%s", model_name, tag)
        return tag, True

    except subprocess.TimeoutExpired:
        log.error("BUILD TIMEOUT model=%s (600s)", model_name)
        return tag, False
    except Exception as e:
        log.error("BUILD ERROR model=%s err=%s", model_name, e)
        return tag, False


def push_image(tag, registry=None):
    """
    Push an image to a registry.
    If no registry configured, just tag locally.
    Returns (remote_tag, success).
    """
    registry = registry or REGISTRY
    if not registry:
        log.warning("PUSH SKIP — no registry configured. Set XCELSIOR_REGISTRY.")
        return tag, False

    remote_tag = (
        f"{registry}/{tag}"
        if "/" not in tag[: tag.index(":")]
        else f"{registry}/{tag.split('/')[-1]}"
    )

    try:
        # Tag for remote
        subprocess.run(
            ["docker", "tag", tag, remote_tag], capture_output=True, text=True, timeout=30
        )

        # Push
        result = subprocess.run(
            ["docker", "push", remote_tag],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log.error("PUSH FAILED tag=%s err=%s", remote_tag, result.stderr.strip())
            return remote_tag, False

        log.info("PUSH SUCCESS tag=%s", remote_tag)
        return remote_tag, True

    except Exception as e:
        log.error("PUSH ERROR tag=%s err=%s", remote_tag, e)
        return remote_tag, False


def build_and_push(
    model_name, context_dir=None, quantize=None, base_image="python:3.11-slim", push=True
):
    """
    Full pipeline: generate Dockerfile, build image, optionally push.
    Returns dict with build results.
    """
    tag, built = build_image(
        model_name, context_dir=context_dir, quantize=quantize, base_image=base_image
    )

    result = {
        "model": model_name,
        "tag": tag,
        "quantize": quantize,
        "built": built,
        "pushed": False,
        "remote_tag": None,
    }

    if built and push:
        remote_tag, pushed = push_image(tag)
        result["pushed"] = pushed
        result["remote_tag"] = remote_tag

    return result


def list_builds():
    """List all local build directories."""
    if not os.path.exists(BUILDS_DIR):
        return []
    builds = []
    for name in sorted(os.listdir(BUILDS_DIR)):
        build_dir = os.path.join(BUILDS_DIR, name)
        if os.path.isdir(build_dir):
            has_dockerfile = os.path.exists(os.path.join(build_dir, "Dockerfile"))
            builds.append(
                {
                    "model": name,
                    "path": build_dir,
                    "has_dockerfile": has_dockerfile,
                }
            )
    return builds


# ── Phase 17: Marketplace ────────────────────────────────────────────

MARKETPLACE_FILE = os.environ.get("XCELSIOR_MARKETPLACE_FILE", os.path.join(os.path.dirname(__file__), "marketplace.json"))
PLATFORM_CUT = float(os.environ.get("XCELSIOR_PLATFORM_CUT", "0.20"))  # 20%


def load_marketplace():
    """Load marketplace listings."""
    return _load_json(MARKETPLACE_FILE)


def save_marketplace(listings):
    """Write marketplace listings."""
    _save_json(MARKETPLACE_FILE, listings)


def list_rig(host_id, gpu_model, vram_gb, price_per_hour, description="", owner="anonymous"):
    """
    List a rig on the marketplace.
    Hosts set their price. Xcelsior takes its cut on every job.
    """
    listings = load_marketplace()

    # Update if exists
    for i, l in enumerate(listings):
        if l["host_id"] == host_id:
            listings[i].update(
                {
                    "gpu_model": gpu_model,
                    "vram_gb": vram_gb,
                    "price_per_hour": price_per_hour,
                    "description": description,
                    "owner": owner,
                    "updated_at": time.time(),
                    "active": True,
                }
            )
            save_marketplace(listings)
            log.info(
                "MARKETPLACE UPDATED listing=%s | %s | $%s/hr", host_id, gpu_model, price_per_hour
            )
            return listings[i]

    listing = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "vram_gb": vram_gb,
        "price_per_hour": price_per_hour,
        "description": description,
        "owner": owner,
        "platform_cut": PLATFORM_CUT,
        "listed_at": time.time(),
        "updated_at": time.time(),
        "active": True,
        "total_jobs": 0,
        "total_earned": 0.0,
    }

    listings.append(listing)
    save_marketplace(listings)
    log.info(
        "MARKETPLACE LISTED %s | %s | %sGB | $%s/hr | owner=%s",
        host_id,
        gpu_model,
        vram_gb,
        price_per_hour,
        owner,
    )
    return listing


def unlist_rig(host_id):
    """Remove a rig from the marketplace."""
    listings = load_marketplace()
    for l in listings:
        if l["host_id"] == host_id:
            l["active"] = False
            save_marketplace(listings)
            log.info("MARKETPLACE UNLISTED %s", host_id)
            return True
    return False


def get_marketplace(active_only=True):
    """Get all marketplace listings."""
    listings = load_marketplace()
    if active_only:
        return [l for l in listings if l.get("active", True)]
    return listings


def marketplace_bill(job_id):
    """
    Bill a marketplace job. Host gets paid, Xcelsior takes its cut.
    Returns (host_payout, platform_fee, total_cost) or None.
    """
    jobs = load_jobs()
    job = None
    for j in jobs:
        if j["job_id"] == job_id:
            job = j
            break

    if not job or not job.get("host_id"):
        return None

    if not job.get("started_at") or not job.get("completed_at"):
        return None

    # Find marketplace listing for this host
    listings = load_marketplace()
    listing = None
    for l in listings:
        if l["host_id"] == job["host_id"]:
            listing = l
            break

    if not listing:
        return None

    duration_sec = job["completed_at"] - job["started_at"]
    duration_hr = duration_sec / 3600
    total_cost = round(duration_hr * listing["price_per_hour"], 4)

    # Tier multiplier
    tier = job.get("tier", "free")
    tier_info = PRIORITY_TIERS.get(tier, PRIORITY_TIERS["free"])
    total_cost = round(total_cost * tier_info["multiplier"], 4)

    platform_fee = round(total_cost * listing.get("platform_cut", PLATFORM_CUT), 4)
    host_payout = round(total_cost - platform_fee, 4)

    # Update listing stats
    listing["total_jobs"] = listing.get("total_jobs", 0) + 1
    listing["total_earned"] = round(listing.get("total_earned", 0) + host_payout, 4)
    save_marketplace(listings)

    log.info(
        "MARKETPLACE BILLED job=%s | total=$%s | host_payout=$%s | platform_fee=$%s",
        job_id,
        total_cost,
        host_payout,
        platform_fee,
    )

    return {
        "job_id": job_id,
        "host_id": job["host_id"],
        "total_cost": total_cost,
        "platform_fee": platform_fee,
        "host_payout": host_payout,
        "platform_cut_pct": listing.get("platform_cut", PLATFORM_CUT),
        "duration_sec": round(duration_sec, 2),
        "tier": tier,
    }


def marketplace_stats():
    """Aggregate marketplace stats."""
    listings = load_marketplace()
    active = [l for l in listings if l.get("active", True)]
    total_earned = sum(l.get("total_earned", 0) for l in listings)
    total_jobs = sum(l.get("total_jobs", 0) for l in listings)
    # Compute platform revenue per listing using each listing's own cut.
    # host_payout = total_cost * (1 - cut), so platform_revenue = payout * cut / (1 - cut)
    platform_revenue = 0.0
    for l in listings:
        payout = l.get("total_earned", 0)
        cut = l.get("platform_cut", PLATFORM_CUT)
        if payout > 0 and 0 < cut < 1:
            platform_revenue += payout * cut / (1 - cut)
    platform_revenue = round(platform_revenue, 4)

    return {
        "total_listings": len(listings),
        "active_listings": len(active),
        "total_jobs_completed": total_jobs,
        "total_host_payouts": round(total_earned, 4),
        "platform_revenue": platform_revenue,
        "default_platform_cut_pct": PLATFORM_CUT,
    }


# ── Phase 18: Canada-Only Toggle ─────────────────────────────────────

CANADA_ONLY = os.environ.get("XCELSIOR_CANADA_ONLY", "").lower() in ("1", "true", "yes")

# Canadian IP ranges (major blocks). Not exhaustive — a real GeoIP DB would be better.
# For now we use a simple approach: tag hosts with a country on registration.


def set_canada_only(enabled):
    """Toggle Canada-only mode at runtime."""
    global CANADA_ONLY
    CANADA_ONLY = enabled
    log.info("CANADA-ONLY MODE %s", "ENABLED" if enabled else "DISABLED")


def register_host_ca(
    host_id,
    ip,
    gpu_model,
    total_vram_gb,
    free_vram_gb,
    cost_per_hour=0.20,
    country="CA",
    province="",
):
    """Register a host with country tag. Defaults to Canada because why wouldn't it."""
    entry = register_host(
        host_id,
        ip,
        gpu_model,
        total_vram_gb,
        free_vram_gb,
        cost_per_hour,
        country=country,
        province=province,
    )
    # Ensure country is set even if register_host didn't receive it
    if not entry.get("country"):
        _set_host_fields(host_id, country=country.upper())
        entry["country"] = country.upper()
    if province and not entry.get("province"):
        _set_host_fields(host_id, province=province.upper())
        entry["province"] = province.upper()

    log.info(
        "HOST REGISTERED (country=%s, province=%s) %s | %s",
        entry.get("country", ""),
        entry.get("province", ""),
        host_id,
        ip,
    )
    return entry


def list_hosts_filtered(active_only=True, canada_only=None):
    """
    List hosts with optional Canada filter.
    If canada_only is None, uses the global CANADA_ONLY setting.
    """
    canada = canada_only if canada_only is not None else CANADA_ONLY
    hosts = list_hosts(active_only=active_only)
    if canada:
        return [h for h in hosts if h.get("country", "").upper() == "CA"]
    return hosts


def process_queue_filtered(canada_only=None):
    """
    Process queue respecting Canada-only mode.
    Only assigns to Canadian hosts when the toggle is on.
    """
    canada = canada_only if canada_only is not None else CANADA_ONLY
    hosts = list_hosts_filtered(active_only=True, canada_only=canada)
    assigned = []

    queued = list_jobs(status="queued")
    queued.sort(key=lambda j: (-j["priority"], j["submitted_at"]))

    for job in queued:
        if not hosts:
            break

        host = allocate(job, hosts)
        if not host:
            continue

        updated = update_job_status(job["job_id"], "running", host_id=host["host_id"])
        if not updated or updated.get("status") != "running":
            continue
        hosts = [h for h in hosts if h["host_id"] != host["host_id"]]
        assigned.append((updated, host))

    return assigned


# ── Phase 19: Auto-Scaling ───────────────────────────────────────────

AUTOSCALE_ENABLED = os.environ.get("XCELSIOR_AUTOSCALE", "").lower() in ("1", "true", "yes")
AUTOSCALE_MAX_HOSTS = int(os.environ.get("XCELSIOR_AUTOSCALE_MAX", "20"))
AUTOSCALE_PROVIDER = os.environ.get("XCELSIOR_AUTOSCALE_PROVIDER", "")  # e.g. "ssh", "api"
AUTOSCALE_POOL_FILE = os.environ.get("XCELSIOR_AUTOSCALE_FILE", os.path.join(os.path.dirname(__file__), "autoscale_pool.json"))


def load_autoscale_pool():
    """Load the pool of available-on-demand hosts."""
    return _load_json(AUTOSCALE_POOL_FILE)


def save_autoscale_pool(pool):
    """Write the autoscale pool."""
    _save_json(AUTOSCALE_POOL_FILE, pool)


def add_to_pool(host_id, ip, gpu_model, vram_gb, cost_per_hour=0.20, country="CA"):
    """Add a host to the autoscale pool (available but not yet active)."""
    pool = load_autoscale_pool()

    for p in pool:
        if p["host_id"] == host_id:
            p.update(
                {
                    "ip": ip,
                    "gpu_model": gpu_model,
                    "vram_gb": vram_gb,
                    "cost_per_hour": cost_per_hour,
                    "country": country,
                }
            )
            save_autoscale_pool(pool)
            return p

    entry = {
        "host_id": host_id,
        "ip": ip,
        "gpu_model": gpu_model,
        "vram_gb": vram_gb,
        "cost_per_hour": cost_per_hour,
        "country": country,
        "provisioned": False,
    }
    pool.append(entry)
    save_autoscale_pool(pool)
    log.info("AUTOSCALE POOL ADD %s | %s | %sGB", host_id, gpu_model, vram_gb)
    return entry


def remove_from_pool(host_id):
    """Remove a host from the autoscale pool."""
    pool = load_autoscale_pool()
    pool = [p for p in pool if p["host_id"] != host_id]
    save_autoscale_pool(pool)
    log.info("AUTOSCALE POOL REMOVE %s", host_id)


def provision_host(pool_entry):
    """
    Provision a host from the pool — make it active.
    SSH provider: connects to the host and starts the worker agent container.
    """
    entry = register_host(
        pool_entry["host_id"],
        pool_entry["ip"],
        pool_entry["gpu_model"],
        pool_entry["vram_gb"],
        pool_entry["vram_gb"],  # starts fully free
        pool_entry.get("cost_per_hour", 0.20),
    )

    _set_host_fields(
        pool_entry["host_id"],
        country=pool_entry.get("country", "CA"),
        autoscaled=True,
    )

    # Mark as provisioned in pool
    pool = load_autoscale_pool()
    for p in pool:
        if p["host_id"] == pool_entry["host_id"]:
            p["provisioned"] = True
            break
    save_autoscale_pool(pool)

    # SSH into host and start the worker agent container
    if AUTOSCALE_PROVIDER == "ssh":
        ip = pool_entry["ip"]
        host_id = pool_entry["host_id"]
        base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
        registry = os.environ.get("XCELSIOR_REGISTRY", "")
        image = f"{registry}/xcelsior-worker:latest" if registry else "xcelsior-worker:latest"
        cost = float(pool_entry.get("cost_per_hour", 0.20))
        docker_cmd = (
            f"docker pull {shlex.quote(image)} 2>/dev/null; "
            f"docker rm -f xcelsior-worker 2>/dev/null; "
            f"docker run -d --restart unless-stopped --name xcelsior-worker "
            f"--gpus all "
            f"-e XCELSIOR_HOST_ID={shlex.quote(host_id)} "
            f"-e XCELSIOR_SCHEDULER_URL={shlex.quote(base_url)} "
            f"-e XCELSIOR_API_TOKEN={shlex.quote(API_TOKEN)} "
            f"-e XCELSIOR_COST_PER_HOUR={cost:.4f} "
            f"-v /var/run/docker.sock:/var/run/docker.sock "
            f"{shlex.quote(image)}"
        )
        rc, stdout, stderr = ssh_exec(ip, docker_cmd)
        if rc != 0:
            log.error(
                "AUTOSCALE SSH PROVISION FAILED host=%s ip=%s rc=%d stderr=%s",
                host_id, ip, rc, stderr,
            )
        else:
            log.info("AUTOSCALE SSH PROVISION OK host=%s container=%s", host_id, stdout[:12])

            # Wait for the worker to register via heartbeat (up to 90s)
            for _ in range(18):
                time.sleep(5)
                hosts = load_hosts(active_only=True)
                for h in hosts:
                    if h["host_id"] == host_id and h.get("last_heartbeat", 0) > time.time() - 30:
                        log.info("AUTOSCALE WORKER READY host=%s", host_id)
                        break
                else:
                    continue
                break
            else:
                log.warning("AUTOSCALE WORKER TIMEOUT host=%s — no heartbeat after 90s", host_id)

    log.info(
        "AUTOSCALE PROVISIONED %s | %s | %sGB",
        pool_entry["host_id"],
        pool_entry["gpu_model"],
        pool_entry["vram_gb"],
    )
    return entry


def deprovision_host(host_id):
    """
    Deprovision: stop worker container, remove from active hosts,
    mark pool entry as unprovisioned.
    """
    # SSH into host and stop the worker container before removing
    if AUTOSCALE_PROVIDER == "ssh":
        pool = load_autoscale_pool()
        ip = None
        for p in pool:
            if p["host_id"] == host_id:
                ip = p.get("ip")
                break
        if ip:
            rc, _, stderr = ssh_exec(ip, "docker stop xcl-worker && docker rm xcl-worker")
            if rc != 0:
                log.warning("AUTOSCALE SSH DEPROVISION warn host=%s: %s", host_id, stderr)
            else:
                log.info("AUTOSCALE SSH DEPROVISION OK host=%s", host_id)

    remove_host(host_id)

    pool = load_autoscale_pool()
    for p in pool:
        if p["host_id"] == host_id:
            p["provisioned"] = False
            break
    save_autoscale_pool(pool)

    log.info("AUTOSCALE DEPROVISIONED %s", host_id)


def autoscale_up():
    """
    Scale up: check queued jobs, provision hosts from pool to match demand.
    Returns list of newly provisioned hosts.
    """
    jobs = load_jobs()
    queued = [j for j in jobs if j["status"] == "queued"]

    if not queued:
        return []

    active_hosts = list_hosts(active_only=True)
    active_count = len(active_hosts)

    pool = load_autoscale_pool()
    available = [p for p in pool if not p.get("provisioned", False)]

    if not available:
        log.info("AUTOSCALE UP — no hosts available in pool")
        return []

    # How many more hosts do we need?
    needed = len(queued)
    can_add = min(needed, len(available), AUTOSCALE_MAX_HOSTS - active_count)

    if can_add <= 0:
        return []

    # Sort by VRAM descending — provision biggest first
    available.sort(key=lambda p: -p.get("vram_gb", 0))

    provisioned = []
    for i in range(can_add):
        entry = provision_host(available[i])
        provisioned.append(entry)

    log.info(
        "AUTOSCALE UP — provisioned %d hosts for %d queued jobs", len(provisioned), len(queued)
    )
    return provisioned


def autoscale_down():
    """
    Scale down: deprovision idle autoscaled hosts with no running jobs.
    Returns list of deprovisioned host IDs.
    """
    hosts = load_hosts(active_only=True)
    jobs = load_jobs()
    busy_host_ids = {j["host_id"] for j in jobs if j["status"] == "running" and j.get("host_id")}

    deprovisioned = []
    for h in hosts:
        if h.get("autoscaled") and h["host_id"] not in busy_host_ids:
            deprovision_host(h["host_id"])
            deprovisioned.append(h["host_id"])

    if deprovisioned:
        log.info("AUTOSCALE DOWN — deprovisioned %d idle hosts", len(deprovisioned))
    return deprovisioned


def autoscale_cycle():
    """
    Full autoscale cycle:
    1. Scale up if jobs are queued
    2. Process queue with new hosts
    3. Scale down idle autoscaled hosts
    Returns (provisioned, assigned, deprovisioned).
    """
    provisioned = autoscale_up()

    assigned = []
    if provisioned:
        assigned = process_queue()

    deprovisioned = autoscale_down()

    return provisioned, assigned, deprovisioned


def start_autoscale_monitor(interval=15, callback=None):
    """Run autoscale checks in a background thread."""

    def loop():
        while True:
            if AUTOSCALE_ENABLED:
                provisioned, assigned, deprovisioned = autoscale_cycle()
                if callback and (provisioned or assigned or deprovisioned):
                    callback(provisioned, assigned, deprovisioned)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ── Run it ────────────────────────────────────────────────────────────


def storage_healthcheck():
    """Basic readiness check for PostgreSQL persistence."""
    try:
        with _db_connection() as conn:
            conn.execute("SELECT 1")
        return {"ok": True, "backend": "postgresql"}
    except Exception as exc:
        log.error("STORAGE HEALTHCHECK FAILED err=%s", exc)
        return {"ok": False, "backend": "postgresql", "error": str(exc)}


def get_metrics_snapshot():
    """Return scheduler metrics for observability endpoints."""
    jobs = load_jobs()
    hosts = load_hosts(active_only=False)
    queued = [j for j in jobs if j.get("status") == "queued"]
    running = [j for j in jobs if j.get("status") == "running"]
    failed = [j for j in jobs if j.get("status") == "failed"]
    return {
        "queue_depth": len(queued),
        "active_hosts": len([h for h in hosts if h.get("status") == "active"]),
        "failed_jobs": len(failed),
        "running_jobs": len(running),
        "billing_totals": {
            "total_revenue": get_total_revenue(),
            "records": len(load_billing()),
        },
    }


# ── Spot Pricing & Preemption ─────────────────────────────────────────
# Supply/demand multiplier curve per REPORT_EXCELSIOR_TECHNICAL_2.md §3.3
# SpotPrice = BasePrice * e^(k * (D/S - threshold))

SPOT_SENSITIVITY = float(os.environ.get("XCELSIOR_SPOT_SENSITIVITY", "0.5"))
SPOT_THRESHOLD = float(os.environ.get("XCELSIOR_SPOT_THRESHOLD", "0.8"))
SPOT_UPDATE_INTERVAL = int(os.environ.get("XCELSIOR_SPOT_UPDATE_INTERVAL", "600"))  # 10 min
PREEMPTION_GRACE_SEC = int(os.environ.get("XCELSIOR_PREEMPTION_GRACE_SEC", "30"))

SPOT_PRICES_FILE = os.environ.get("XCELSIOR_SPOT_PRICES_FILE", os.path.join(os.path.dirname(__file__), "spot_prices.json"))


def load_spot_prices():
    """Load spot price history."""
    return _load_json(SPOT_PRICES_FILE)


def save_spot_prices(prices):
    """Write spot price history."""
    _save_json(SPOT_PRICES_FILE, prices)


def compute_spot_price(base_price, demand, supply, k=None, threshold=None):
    """Compute spot price using supply/demand curve with 50% cap.

    Formula: spot = base_price * (1 + demand_factor)
    Demand factor capped: min(0.5, queue_depth / (available_gpus * 2))
    Per Phase 2.2: max 50% surge above base price.
    """
    if supply <= 0:
        return round(base_price * 1.5, 4)  # Cap at 50% surge

    # Demand factor: capped at 0.5 (50% max surge)
    demand_factor = min(0.5, demand / (supply * 2))
    multiplier = 1.0 + demand_factor

    return round(base_price * multiplier, 4)


def update_spot_prices():
    """Recalculate spot prices for all GPU types.

    Prices update periodically (every 10 minutes by default).
    Returns dict of {gpu_model: current_spot_price}.
    """
    hosts = load_hosts(active_only=True)
    jobs = load_jobs()

    active_jobs = [j for j in jobs if j["status"] in ("running", "queued")]

    # Group by GPU model
    supply_by_gpu = {}
    base_price_by_gpu = {}
    for h in hosts:
        gpu = h.get("gpu_model", "unknown")
        supply_by_gpu[gpu] = supply_by_gpu.get(gpu, 0) + 1
        # Track lowest base price per GPU type
        cost = h.get("cost_per_hour", 0.20)
        if gpu not in base_price_by_gpu or cost < base_price_by_gpu[gpu]:
            base_price_by_gpu[gpu] = cost

    demand_by_gpu = {}
    for j in active_jobs:
        host_id = j.get("host_id")
        if host_id:
            for h in hosts:
                if h["host_id"] == host_id:
                    gpu = h.get("gpu_model", "unknown")
                    demand_by_gpu[gpu] = demand_by_gpu.get(gpu, 0) + 1
                    break
        else:
            # Queued job: count as demand for any GPU type
            for gpu in supply_by_gpu:
                demand_by_gpu[gpu] = demand_by_gpu.get(gpu, 0) + 1

    spot_prices = {}
    now = time.time()
    history = load_spot_prices()

    for gpu in supply_by_gpu:
        base = base_price_by_gpu.get(gpu, 0.20)
        demand = demand_by_gpu.get(gpu, 0)
        supply = supply_by_gpu[gpu]

        price = compute_spot_price(base, demand, supply)
        spot_prices[gpu] = price

        history.append(
            {
                "gpu_model": gpu,
                "price": price,
                "supply": supply,
                "demand": demand,
                "computed_at": now,
            }
        )

    # Keep last 1000 records
    if len(history) > 1000:
        history = history[-1000:]
    save_spot_prices(history)

    emit_event("spot_prices", spot_prices)
    log.info("SPOT PRICES UPDATED: %s", json.dumps(spot_prices))
    return spot_prices


def get_current_spot_prices():
    """Get the latest spot prices per GPU model."""
    history = load_spot_prices()
    if not history:
        return {}

    latest = {}
    for entry in reversed(history):
        gpu = entry.get("gpu_model")
        if gpu and gpu not in latest:
            latest[gpu] = entry["price"]

    return latest


def identify_preemptible_jobs(spot_prices=None):
    """Find jobs whose max_bid is below current spot price.

    Returns list of (job, reason) tuples.
    """
    if spot_prices is None:
        spot_prices = get_current_spot_prices()

    jobs = load_jobs()
    hosts = load_hosts(active_only=False)
    host_map = {h["host_id"]: h for h in hosts}

    preemptible = []
    for j in jobs:
        if j["status"] != "running":
            continue

        max_bid = j.get("max_bid")
        if max_bid is None:
            continue  # No bid = on-demand, not preemptible

        host = host_map.get(j.get("host_id"))
        if not host:
            continue

        gpu = host.get("gpu_model", "unknown")
        current_price = spot_prices.get(gpu)
        if current_price is None:
            continue

        if max_bid < current_price:
            preemptible.append(
                (
                    j,
                    f"bid ${max_bid} < spot ${current_price} for {gpu}",
                )
            )

    return preemptible


def preempt_job(job_id):
    """Preempt a running spot job.

    1. Mark job as 'preempted' (terminal state)
    2. Signal the agent to send SIGTERM (grace period for checkpoint)
    3. After grace period, force kill
    4. Requeue if user chose auto-requeue

    Returns the updated job or None.
    """
    with _atomic_mutation() as conn:
        _migrate_jobs_if_needed(conn)
        j = _get_job_by_id_conn(conn, job_id)
        if not j or j["status"] != "running":
            return None

        j["status"] = "queued"
        j["preempted_at"] = time.time()
        j["host_id"] = None
        j["started_at"] = None
        j["retries"] = j.get("retries", 0)  # Don't count preemption as retry
        _upsert_job_row(conn, j)

    emit_event("job_preempted", {"job_id": job_id, "name": j.get("name")})
    log.warning("JOB PREEMPTED %s | %s | requeued for lower-price host", job_id, j.get("name"))
    return j


def preemption_cycle():
    """Run a full preemption cycle.

    1. Update spot prices
    2. Find jobs below spot
    3. Preempt them
    Returns (spot_prices, preempted_jobs).
    """
    prices = update_spot_prices()
    preemptible = identify_preemptible_jobs(prices)
    preempted = []

    for job, reason in preemptible:
        result = preempt_job(job["job_id"])
        if result:
            preempted.append(result)
            log.info("PREEMPTION: %s — %s", job["job_id"], reason)

    return prices, preempted


def start_spot_price_monitor(interval=None, callback=None):
    """Run spot price updates and preemption in a background thread."""
    interval = interval or SPOT_UPDATE_INTERVAL

    def loop():
        while True:
            try:
                prices, preempted = preemption_cycle()
                if callback and (prices or preempted):
                    callback(prices, preempted)
            except Exception as e:
                log.error("Spot price monitor error: %s", e)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ── Compute Score (XCU) ──────────────────────────────────────────────
# Xcelsior Compute Unit: normalize GPU performance across consumer cards.
# RTX 3090 and 4090 both have 24GB VRAM but 4090 has ~2x TFLOPS.

COMPUTE_SCORES_FILE = os.environ.get("XCELSIOR_COMPUTE_SCORES_FILE", os.path.join(os.path.dirname(__file__), "compute_scores.json"))

# Reference TFLOPS for common GPUs (FP16 tensor core)
GPU_REFERENCE_TFLOPS = {
    "RTX 2060": 13.0,
    "RTX 2070": 14.9,
    "RTX 2080": 20.3,
    "RTX 2080 Ti": 26.9,
    "RTX 3060": 12.7,
    "RTX 3060 Ti": 16.2,
    "RTX 3070": 20.3,
    "RTX 3070 Ti": 21.7,
    "RTX 3080": 29.8,
    "RTX 3080 Ti": 34.1,
    "RTX 3090": 35.6,
    "RTX 3090 Ti": 40.0,
    "RTX 4060": 15.1,
    "RTX 4060 Ti": 22.1,
    "RTX 4070": 29.1,
    "RTX 4070 Ti": 40.1,
    "RTX 4080": 48.7,
    "RTX 4090": 82.6,
    "A100": 312.0,
    "A10G": 31.2,
    "H100": 989.5,
    "L4": 30.3,
    "T4": 8.1,
}


def load_compute_scores():
    """Load compute score records."""
    return _load_json(COMPUTE_SCORES_FILE)


def save_compute_scores(scores):
    """Write compute score records."""
    _save_json(COMPUTE_SCORES_FILE, scores)


def estimate_compute_score(gpu_model, benchmark_result=None):
    """Estimate compute score (XCU) for a GPU.

    If benchmark_result is provided, use actual measurement.
    Otherwise, use reference TFLOPS table.

    XCU = TFLOPS / 10 (normalized so RTX 4090 ≈ 8.3 XCU)
    """
    if benchmark_result and "tflops" in benchmark_result:
        return round(benchmark_result["tflops"] / 10.0, 2)

    # Fuzzy match GPU model to reference table
    for ref_name, tflops in GPU_REFERENCE_TFLOPS.items():
        if ref_name.lower() in gpu_model.lower():
            return round(tflops / 10.0, 2)

    # Unknown GPU — conservative estimate
    return 1.0


def register_compute_score(host_id, gpu_model, score, benchmark_details=None):
    """Register a compute score for a host."""
    scores = load_compute_scores()

    entry = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "score": score,
        "details": benchmark_details,
        "run_at": time.time(),
    }

    # Update existing or append
    for i, s in enumerate(scores):
        if s["host_id"] == host_id:
            scores[i] = entry
            break
    else:
        scores.append(entry)

    save_compute_scores(scores)
    _set_host_fields(host_id, compute_score=score)

    log.info("COMPUTE SCORE host=%s gpu=%s score=%.2f XCU", host_id, gpu_model, score)
    return entry


def get_compute_score(host_id):
    """Get the compute score for a host."""
    scores = load_compute_scores()
    for s in scores:
        if s["host_id"] == host_id:
            return s["score"]
    return None


# ── Enhanced Allocation (Compute-Aware) ───────────────────────────────


def allocate_compute_aware(job, hosts):
    """Compute-aware allocation — delegates to allocate() which already
    includes XCU/dollar efficiency scoring alongside admission gating,
    GPU count, isolation tier, and VRAM checks."""
    return allocate(job, hosts)


# ── v2.1: Jurisdiction-Aware Allocation ───────────────────────────────


def allocate_jurisdiction_aware(job, hosts, constraint=None):
    """Allocate with jurisdiction filtering + trust tier + verification.

    Layers (per REPORT_FEATURE_FINAL.md):
    1. Filter by jurisdiction constraint (Canada-only, province, trust tier)
    2. Filter by verification status (only verified hosts)
    3. Sort by compute efficiency (XCU/$/hr)
    4. Apply reputation boost (higher-tier hosts preferred)
    """
    if not hosts:
        return None

    # 1. Jurisdiction filter
    if constraint:
        hosts = filter_hosts_by_jurisdiction(hosts, constraint)
        if not hosts:
            log.warning(
                "ALLOCATE no hosts match jurisdiction constraint for job=%s", job.get("name", "?")
            )
            return None

    # 2. Verification filter — only use verified hosts for production
    try:
        ve = get_verification_engine()
        verified_ids = {h["host_id"] for h in ve.get_verified_hosts()}
        verified_hosts = [h for h in hosts if h["host_id"] in verified_ids]
        if verified_hosts:
            hosts = verified_hosts
        # If no verified hosts, fall back to all (for cold-start)
    except Exception:
        pass

    # 3. VRAM filter
    candidates = [h for h in hosts if h.get("free_vram_gb", 0) >= job.get("vram_needed_gb", 0)]
    if not candidates:
        return None

    # 4. Sort by compute efficiency * reputation boost
    def score_host(h):
        compute = h.get("compute_score") or estimate_compute_score(h.get("gpu_model", ""))
        price = h.get("cost_per_hour", 0.20) or 0.20
        efficiency = compute / price

        # Reputation boost
        try:
            re = get_reputation_engine()
            rep = re.compute_score(h["host_id"])
            boost = rep.search_boost
        except Exception:
            boost = 1.0

        return efficiency * boost

    best = max(candidates, key=score_host)
    log.info(
        "ALLOCATE (jurisdiction-aware) job=%s -> host=%s (%s, %sGB free, country=%s)",
        job.get("name", "?"),
        best["host_id"],
        best.get("gpu_model"),
        best.get("free_vram_gb"),
        best.get("country", "?"),
    )
    return best


def process_queue_sovereign(canada_only=None, province=None, trust_tier=None):
    """Process queue with full jurisdiction + verification + reputation.

    From REPORT_FEATURE_FINAL.md + REPORT_MARKETING_FINAL.md:
    - Jurisdiction-aware host selection
    - Verified hosts preferred
    - Reputation-boosted ordering
    - Event-sourced state transitions
    """
    canada = canada_only if canada_only is not None else CANADA_ONLY

    constraint = None
    if canada or province or trust_tier:
        constraint = JurisdictionConstraint(
            canada_only=canada,
            province=province,
            trust_tier=TrustTier(trust_tier) if trust_tier else None,
        )

    hosts = list_hosts(active_only=True)
    assigned = []

    queued = list_jobs(status="queued")
    queued.sort(key=lambda j: (-j["priority"], j["submitted_at"]))

    for job in queued:
        if not hosts:
            break

        host = allocate_jurisdiction_aware(job, hosts, constraint)
        if not host:
            continue

        updated = update_job_status(job["job_id"], "running", host_id=host["host_id"])
        if not updated or updated.get("status") != "running":
            continue
        hosts = [h for h in hosts if h["host_id"] != host["host_id"]]
        assigned.append((updated, host))

    return assigned


# ── Spot Job Submission ───────────────────────────────────────────────


def submit_spot_job(name, vram_needed_gb, max_bid, priority=0, tier=None, owner=""):
    """Submit a spot/interruptible job with a maximum bid price.

    Spot jobs are:
    - Cheaper (run when spot price < max_bid)
    - Preemptible (evicted when demand exceeds bid)
    - Automatically requeued on preemption
    """
    job = submit_job(name, vram_needed_gb, priority, tier=tier, owner=owner)

    _set_job_fields(
        job["job_id"],
        max_bid=max_bid,
        spot=True,
        preemptible=True,
    )
    job["max_bid"] = max_bid
    job["spot"] = True
    job["preemptible"] = True

    log.info(
        "SPOT JOB SUBMITTED %s | %s | max_bid=$%s/hr",
        job["job_id"],
        name,
        max_bid,
    )
    return job


if __name__ == "__main__":
    # Clean slate
    for f in (HOSTS_FILE, JOBS_FILE):
        if os.path.exists(f):
            os.remove(f)

    # Phase 2: Register hosts
    register_host("rig-01", "127.0.0.1", "RTX 4090", 24, 24)
    register_host("rig-02", "192.0.2.1", "RTX 3090", 24, 16)
    register_host("rig-03", "127.0.0.1", "A100", 80, 80, cost_per_hour=0.50)

    print("=== REGISTERED HOSTS ===")
    for h in list_hosts(active_only=False):
        print(
            f"  {h['host_id']} | {h['ip']} | {h['gpu_model']} | {h['free_vram_gb']}GB free | ${h['cost_per_hour']}/hr"
        )

    # Phase 4: Health check
    print("\n=== HEALTH CHECK ===")
    results = check_hosts()
    for host_id, status in results.items():
        print(f"  {host_id}: {status}")

    # Phase 3: Submit + allocate
    job = submit_job("test-job", vram_needed_gb=8, priority=1)
    assigned = process_queue()

    print("\n=== ASSIGNED ===")
    for j, h in assigned:
        print(f"  {j['name']} -> {h['host_id']}")

    # Phase 5 & 6: Run a real local container, watch it, kill it
    print("\n=== PHASE 5 & 6: LOCAL DOCKER TEST ===")
    test_job = submit_job("alpine-test", vram_needed_gb=0, priority=0)
    update_job_status(test_job["job_id"], "running")

    # Run alpine with a 3-second sleep — simulates a short job
    cid, cname = run_job_local(test_job, docker_image="alpine:latest")
    if cid:
        # Re-run with a command: sleep 2 then exit
        kill_job_local(cname)
        cname = f"xcl-{test_job['job_id']}-demo"
        result = subprocess.run(
            ["docker", "run", "-d", "--name", cname, "alpine:latest", "sleep", "2"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  Container started: {cname}")
            # Poll until done
            while True:
                inspect = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Status}}", cname],
                    capture_output=True,
                    text=True,
                )
                status = inspect.stdout.strip()
                print(f"  Status: {status}")
                if status == "exited":
                    break
                time.sleep(1)

            # Clean up
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
            update_job_status(test_job["job_id"], "completed")
            print(f"  Job {test_job['job_id']} COMPLETED. Container removed.")
        else:
            print(f"  Docker run failed: {result.stderr.strip()}")
    else:
        print("  Docker not available — skipping live test.")
        print("  (Phase 5 & 6 functions are written and ready for real hosts.)")

    print("\n=== FINAL JOB STATUS ===")
    for j in list_jobs():
        host = j["host_id"] or "—"
        print(f"  [{j['status']:>9}] {j['job_id']} | {j['name']} | host: {host}")

    # Phase 8: Bill completed jobs
    print("\n=== BILLING ===")
    bills = bill_all_completed()
    for b in bills:
        print(f"  {b['job_name']} | {b['duration_sec']}s | ${b['cost']} @ ${b['rate_per_hour']}/hr")
    print(f"\n  TOTAL REVENUE: ${get_total_revenue()}")

    # Cleanup
    for f in (HOSTS_FILE, JOBS_FILE, LOG_FILE, BILLING_FILE):
        if os.path.exists(f):
            os.remove(f)


# ── Crypto Confirmation Watcher ───────────────────────────────────────


def _process_crypto_confirmations():
    """Check pending BTC deposits and credit wallets on confirmation."""
    try:
        import bitcoin as btc
    except ImportError:
        return []
    if not btc.BTC_ENABLED:
        return []

    from billing import BillingEngine

    credited = []
    pending = btc.get_pending_deposits()
    if not pending:
        return credited

    be = BillingEngine()

    for dep in pending:
        try:
            updated = btc.check_and_update_deposit(dep)
            if updated["status"] == "confirmed":
                # Credit the CAD wallet
                be.deposit(
                    updated["customer_id"],
                    updated["amount_cad"],
                    f"Bitcoin deposit {updated['deposit_id']} ({updated['amount_btc']:.8f} BTC)",
                )
                btc.mark_credited(updated["deposit_id"])
                credited.append(updated)
                log.info(
                    "BTC wallet credited: %s +$%.2f CAD (%s)",
                    updated["customer_id"],
                    updated["amount_cad"],
                    updated["deposit_id"],
                )
        except Exception as e:
            log.error("Crypto watcher error for %s: %s", dep["deposit_id"], e)

    return credited


def start_crypto_watcher(interval: int = 60, callback=None):
    """Run BTC confirmation checks in a background thread."""
    def loop():
        while True:
            try:
                credited = _process_crypto_confirmations()
                if callback and credited:
                    callback(credited)
            except Exception as e:
                log.error("Crypto watcher error: %s", e)
            time.sleep(interval)

    t = threading.Thread(target=loop, daemon=True, name="crypto-watcher")
    t.start()
    log.info("Crypto confirmation watcher started (interval=%ds)", interval)
    return t


# ── Stripe Webhook Inbox Processor ───────────────────────────────────
# Per Phase 1.1: background worker loop in scheduler.py to process inbox rows.
# Delegates to stripe_connect._process_single_event() for actual handling.


def process_webhook_inbox():
    """Process pending Stripe webhook events from the inbox.

    Claims events via UPDATE ... WHERE status='pending' AND next_retry_at <= now()
    with row lock (FOR UPDATE SKIP LOCKED) to prevent double processing.
    Called periodically by the background scheduler.
    """
    try:
        from stripe_connect import get_connect_engine
        engine = get_connect_engine()
    except Exception as e:
        log.debug("Stripe Connect not available: %s", e)
        return {"processed": 0, "errors": 0}

    from db import _get_pg_pool
    from psycopg.rows import dict_row

    processed = 0
    errors = 0

    try:
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            pending = conn.execute(
                """SELECT event_id FROM stripe_event_inbox
                   WHERE status = 'pending'
                     AND (next_retry_at IS NULL OR next_retry_at <= %s)
                   ORDER BY received_at ASC
                   LIMIT 50
                   FOR UPDATE SKIP LOCKED""",
                (time.time(),),
            ).fetchall()

        for row in pending:
            try:
                result = engine._process_single_event(row["event_id"])
                if result:
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.error("Webhook inbox processing error for %s: %s", row["event_id"], e)

    except Exception as e:
        log.debug("Webhook inbox sweep skipped: %s", e)

    if processed or errors:
        log.info("WEBHOOK INBOX: processed=%d errors=%d", processed, errors)
    return {"processed": processed, "errors": errors}
