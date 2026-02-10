# Xcelsior v1.0.0 — distributed GPU scheduler for Canadians who refuse to wait.
# 20 phases. No shortcuts. Ever upward.

import fcntl
import json
import logging
import os
import re
import shlex
import smtplib
import subprocess
import threading
import time
import urllib.request
import uuid
from email.mime.text import MIMEText

HOSTS_FILE = os.path.join(os.path.dirname(__file__), "hosts.json")
JOBS_FILE = os.path.join(os.path.dirname(__file__), "jobs.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), "xcelsior.log")

# ── Phase 13: Security ───────────────────────────────────────────────

SSH_KEY_PATH = os.environ.get("XCELSIOR_SSH_KEY_PATH", os.path.expanduser("~/.ssh/xcelsior"))
SSH_USER = os.environ.get("XCELSIOR_SSH_USER", "xcelsior")
API_TOKEN = os.environ.get("XCELSIOR_API_TOKEN", "")


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
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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

# Safe name pattern: alphanumeric, hyphens, underscores, dots, slashes
_SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9._:/@-]+$')


def _validate_name(value, label="value"):
    """Reject shell-unsafe characters in names used in commands."""
    if not value or not _SAFE_NAME_RE.match(str(value)):
        raise ValueError(f"Invalid {label}: {value!r} — only alphanumeric, hyphens, underscores, dots, colons, slashes allowed")
    return str(value)


def _load_json(path):
    """Load a JSON file with file locking. Returns [] if missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _save_json(path, data):
    """Write a JSON file with exclusive file locking.

    Opens in append mode to avoid truncating before the lock is held,
    then seeks to the beginning, truncates, and writes the new data.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with os.fdopen(fd, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
    except Exception:
        # fd is already closed by os.fdopen/context-manager on success;
        # only close manually if os.fdopen itself failed.
        try:
            os.close(fd)
        except OSError:
            pass
        raise


# ── Phase 1: Allocate ────────────────────────────────────────────────

def allocate(job, hosts):
    """
    Find the best host for this job.
    Prioritize: available VRAM > speed > lowest cost.
    If nothing fits, return None (queue or reject).
    """
    if not hosts:
        return None

    candidates = [h for h in hosts if h.get("free_vram_gb", 0) >= job.get("vram_needed_gb", 0)]
    if not candidates:
        return None

    # Prioritize: available VRAM > speed (low latency) > lowest cost
    best = max(candidates, key=lambda h: (
        h.get("free_vram_gb", 0),
        -h.get("latency_ms", 999),
        -h.get("cost_per_hour", 999),
    ))
    log.info("ALLOCATE job=%s -> host=%s (%s, %sGB free)",
             job.get("name", "?"), best["host_id"], best.get("gpu_model"), best.get("free_vram_gb"))
    return best


# ── Phase 2: Host Registry ───────────────────────────────────────────

def load_hosts(active_only=False):
    """Load hosts from JSON. No file? Empty list. Simple."""
    hosts = _load_json(HOSTS_FILE)
    if active_only:
        return [h for h in hosts if h.get("status") == "active"]
    return hosts


def save_hosts(hosts):
    """Write hosts to JSON. File-locked."""
    _save_json(HOSTS_FILE, hosts)


def register_host(host_id, ip, gpu_model, total_vram_gb, free_vram_gb, cost_per_hour=0.20):
    """
    Register a host. If it exists, update it. If not, add it.
    Every host sends: GPU model, VRAM, IP, uptime.
    You store it.
    """
    hosts = load_hosts()

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

    # Update if exists, else append
    for i, h in enumerate(hosts):
        if h["host_id"] == host_id:
            hosts[i] = entry
            save_hosts(hosts)
            log.info("HOST UPDATED %s | %s | %s | %sGB", host_id, ip, gpu_model, total_vram_gb)
            return entry

    hosts.append(entry)
    save_hosts(hosts)
    log.info("HOST REGISTERED %s | %s | %s | %sGB | $%s/hr", host_id, ip, gpu_model, total_vram_gb, cost_per_hour)
    return entry


def remove_host(host_id):
    """Host is dead. Remove it. No funeral."""
    hosts = load_hosts()
    hosts = [h for h in hosts if h["host_id"] != host_id]
    save_hosts(hosts)
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
    Returns dict of results.
    """
    hosts = load_hosts(active_only=False)
    results = {}

    for h in hosts:
        alive = ping_host(h["ip"])
        if alive:
            h["last_seen"] = time.time()
            if h.get("status") == "dead":
                log.info("HOST REVIVED %s (%s)", h["host_id"], h["ip"])
            h["status"] = "active"
            results[h["host_id"]] = "alive"
        else:
            if h.get("status") != "dead":
                log.warning("HOST DEAD %s (%s)", h["host_id"], h["ip"])
                alert_host_dead(h["host_id"], h["ip"])
            h["status"] = "dead"
            results[h["host_id"]] = "dead"

    save_hosts(hosts)
    return results


def health_loop(interval=5, callback=None):
    """
    Ping hosts every `interval` seconds. Forever.
    If one dies — mark it dead.
    If one comes back — mark it active.
    Run this in a thread.
    """
    while True:
        results = check_hosts()
        if callback:
            callback(results)
        time.sleep(interval)


def start_health_monitor(interval=5, callback=None):
    """Start the health loop in a background thread. Fire and forget."""
    t = threading.Thread(target=health_loop, args=(interval, callback), daemon=True)
    t.start()
    return t


# ── Phase 15: Priority Tiers ─────────────────────────────────────────

PRIORITY_TIERS = {
    "free":     {"priority": 0, "multiplier": 1.0,  "label": "Free"},
    "standard": {"priority": 1, "multiplier": 1.0,  "label": "Standard"},
    "premium":  {"priority": 2, "multiplier": 1.5,  "label": "Premium"},
    "urgent":   {"priority": 3, "multiplier": 2.0,  "label": "Urgent"},
}


def get_tier_info(tier_name):
    """Look up a tier. Returns dict or None."""
    return PRIORITY_TIERS.get(tier_name)


def get_tier_by_priority(priority):
    """Reverse lookup: priority int -> tier name."""
    for name, info in PRIORITY_TIERS.items():
        if info["priority"] == priority:
            return name
    return "free"


def list_tiers():
    """Return all tiers with their details."""
    return {name: dict(info) for name, info in PRIORITY_TIERS.items()}


# ── Phase 3: Job Queue ────────────────────────────────────────────────

VALID_STATUSES = ("queued", "running", "completed", "failed")


def load_jobs():
    """Load jobs from JSON. No file? Empty list."""
    return _load_json(JOBS_FILE)


def save_jobs(jobs):
    """Write jobs to JSON."""
    _save_json(JOBS_FILE, jobs)


def submit_job(name, vram_needed_gb, priority=0, tier=None):
    """
    Submit a job to the queue.
    Each job has: model name, VRAM needed, priority, tier.
    Tier overrides priority. Pay more = jump the queue.
    FIFO within the same priority. Higher priority goes first.
    """
    # Tier overrides raw priority
    if tier and tier in PRIORITY_TIERS:
        tier_info = PRIORITY_TIERS[tier]
        priority = tier_info["priority"]
    elif tier:
        log.warning("UNKNOWN TIER '%s' — defaulting to free", tier)
        tier = "free"
        priority = 0
    else:
        tier = get_tier_by_priority(priority)

    jobs = load_jobs()

    job = {
        "job_id": str(uuid.uuid4())[:8],
        "name": name,
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
    }

    jobs.append(job)
    save_jobs(jobs)
    log.info("JOB SUBMITTED %s | %s | %sGB VRAM | tier=%s (priority %s)",
             job["job_id"], name, vram_needed_gb, tier, priority)
    return job


def get_next_job():
    """
    Pull the next job off the queue.
    Highest priority first. Within same priority: FIFO. No mercy.
    """
    jobs = load_jobs()
    queued = [j for j in jobs if j["status"] == "queued"]
    if not queued:
        return None

    # Sort: highest priority first, then earliest submission
    queued.sort(key=lambda j: (-j["priority"], j["submitted_at"]))
    return queued[0]


def update_job_status(job_id, status, host_id=None):
    """Mark a job. queued -> running -> completed/failed. That's the lifecycle."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}' for job {job_id} — must be one of {VALID_STATUSES}")
    jobs = load_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            old_status = j["status"]
            j["status"] = status
            if host_id:
                j["host_id"] = host_id
            if status == "running":
                j["started_at"] = time.time()
            if status in ("completed", "failed"):
                j["completed_at"] = time.time()
            lvl = logging.WARNING if status == "failed" else logging.INFO
            log.log(lvl, "JOB %s %s -> %s | %s | host=%s",
                    status.upper(), old_status, status, job_id, host_id or j.get("host_id", "—"))
            if status == "failed":
                alert_job_failed(job_id, j.get("name", "?"), j.get("host_id"))
            elif status == "completed":
                dur = None
                if j.get("started_at") and j.get("completed_at"):
                    dur = j["completed_at"] - j["started_at"]
                alert_job_completed(job_id, j.get("name", "?"), dur)
            break
    save_jobs(jobs)


def list_jobs(status=None):
    """List jobs. Filter by status or get everything."""
    jobs = load_jobs()
    if status:
        return [j for j in jobs if j["status"] == status]
    return jobs


def process_queue():
    """
    The loop. Walk the queue by priority. Find a host. Assign it.
    If no host fits a job, skip it — try the next one.
    If no hosts left, stop.
    """
    hosts = list_hosts()
    assigned = []

    jobs = load_jobs()
    queued = [j for j in jobs if j["status"] == "queued"]
    queued.sort(key=lambda j: (-j["priority"], j["submitted_at"]))

    for job in queued:
        if not hosts:
            break

        host = allocate(job, hosts)
        if not host:
            continue  # no host for THIS job, but maybe smaller jobs fit

        update_job_status(job["job_id"], "running", host_id=host["host_id"])
        hosts = [h for h in hosts if h["host_id"] != host["host_id"]]
        assigned.append((job, host))

    return assigned


# ── Phase 5 & 6: Run Job / Kill Job ──────────────────────────────────

def ssh_exec(ip, cmd):
    """
    Run a command on a remote host via SSH.
    Key-based auth only. No passwords. No agent forwarding.
    Returns (returncode, stdout, stderr).
    """
    full_cmd = [
        "ssh",
        "-i", SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
        f"{SSH_USER}@{ip}", cmd,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def generate_ssh_keypair(path=None):
    """Generate an Ed25519 SSH keypair. No passphrase. Fast. Secure."""
    path = path or SSH_KEY_PATH
    # Resolve to absolute and ensure it's under home directory
    path = os.path.realpath(os.path.expanduser(path))
    home = os.path.realpath(os.path.expanduser("~"))
    if not path.startswith(home + os.sep) and path != home:
        log.error("SSH KEYGEN REJECTED: path %s is outside home directory", path)
        raise ValueError(f"SSH key path must be under home directory: {path}")
    if os.path.exists(path):
        log.info("SSH key already exists: %s", path)
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", path, "-N", "", "-C", "xcelsior"],
        capture_output=True, text=True,
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
    """
    image = docker_image or f"xcelsior/{job['name']}:latest"
    container_name = f"xcl-{job['job_id']}"

    # Validate names to prevent shell injection
    _validate_name(image, "docker image")
    _validate_name(container_name, "container name")

    cmd = (
        f"docker run -d --gpus all "
        f"--name {shlex.quote(container_name)} "
        f"{shlex.quote(image)}"
    )

    rc, stdout, stderr = ssh_exec(host["ip"], cmd)
    if rc != 0:
        log.error("RUN FAILED job=%s host=%s err=%s", job["job_id"], host["host_id"], stderr)
        update_job_status(job["job_id"], "failed")
        return None

    container_id = stdout[:12]
    log.info("CONTAINER STARTED job=%s host=%s container=%s image=%s",
             job["job_id"], host["host_id"], container_id, image)

    # Store container info on the job
    jobs = load_jobs()
    for j in jobs:
        if j["job_id"] == job["job_id"]:
            j["container_id"] = container_id
            j["container_name"] = container_name
            break
    save_jobs(jobs)

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
    # Mark done
    update_job_status(job["job_id"], "completed")
    log.info("JOB KILLED job=%s host=%s container=%s", job["job_id"], host["host_id"], container_name)


def wait_for_job(job, host, poll_interval=5):
    """
    Watch a job until it finishes on its own.
    When the container exits, mark it complete and clean up.
    Returns final status: "completed" or "failed".
    """
    container_name = job.get("container_name", f"xcl-{job['job_id']}")
    _validate_name(container_name, "container name")

    while True:
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
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        update_job_status(job["job_id"], "failed")
        return None, None

    if result.returncode != 0:
        update_job_status(job["job_id"], "failed")
        return None, None

    container_id = result.stdout.strip()[:12]

    jobs = load_jobs()
    for j in jobs:
        if j["job_id"] == job["job_id"]:
            j["container_id"] = container_id
            j["container_name"] = container_name
            break
    save_jobs(jobs)

    return container_id, container_name


def kill_job_local(container_name):
    """Kill and remove a local container."""
    subprocess.run(["docker", "kill", container_name],
                   capture_output=True, timeout=10)
    subprocess.run(["docker", "rm", "-f", container_name],
                   capture_output=True, timeout=10)


# ── Phase 8: Billing ─────────────────────────────────────────────────

BILLING_FILE = os.path.join(os.path.dirname(__file__), "billing.json")
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

    The duplicate check and write are performed under an exclusive
    file lock so concurrent calls cannot both slip past the check.
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

    # ── Atomic check-and-write under exclusive lock ──────────────
    fd = os.open(BILLING_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with os.fdopen(fd, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            content = f.read()
            records = json.loads(content) if content.strip() else []

            if any(r["job_id"] == job_id for r in records):
                log.warning("BILLING SKIPPED job=%s already billed", job_id)
                return None

            records.append(record)
            f.seek(0)
            f.truncate()
            json.dump(records, f, indent=2)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise

    log.info("BILLED job=%s | %s | %.1fs | $%.4f @ $%s/hr x%.1f (%s)",
             job_id, job["name"], duration_sec, cost, rate, multiplier, tier)
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
    "email_enabled": False,
    "smtp_host": os.environ.get("XCELSIOR_SMTP_HOST", ""),
    "smtp_port": int(os.environ.get("XCELSIOR_SMTP_PORT", "587")),
    "smtp_user": os.environ.get("XCELSIOR_SMTP_USER", ""),
    "smtp_pass": os.environ.get("XCELSIOR_SMTP_PASS", ""),
    "email_from": os.environ.get("XCELSIOR_EMAIL_FROM", ""),
    "email_to": os.environ.get("XCELSIOR_EMAIL_TO", ""),

    "telegram_enabled": False,
    "telegram_bot_token": os.environ.get("XCELSIOR_TG_TOKEN", ""),
    "telegram_chat_id": os.environ.get("XCELSIOR_TG_CHAT_ID", ""),
}


def configure_alerts(**kwargs):
    """Update alert config at runtime."""
    ALERT_CONFIG.update(kwargs)


def send_email(subject, body):
    """Send an email alert. SMTP. No dependencies."""
    cfg = ALERT_CONFIG
    if not cfg["email_enabled"]:
        return False

    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[Xcelsior] {subject}"
        msg["From"] = cfg["email_from"]
        msg["To"] = cfg["email_to"]

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_pass"])
            server.send_message(msg)

        log.info("EMAIL SENT: %s -> %s", subject, cfg["email_to"])
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
        payload = json.dumps({
            "chat_id": cfg["telegram_chat_id"],
            "text": f"[Xcelsior] {message}",
            "parse_mode": "HTML",
        }).encode()

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
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
        threading.Thread(target=send_telegram, args=(f"<b>{subject}</b>\n{body}",), daemon=True).start()
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
    Reset a failed/running job back to queued.
    Increment retry counter. Clear host assignment.
    If max retries exceeded, mark permanently failed.
    """
    jobs = load_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            # Only requeue running or failed jobs
            if j["status"] not in ("running", "failed"):
                log.warning("REQUEUE REJECTED job=%s status=%s — not requeuable",
                            job_id, j["status"])
                return None

            retries = j.get("retries", 0) + 1
            max_retries = j.get("max_retries", 3)

            if retries > max_retries:
                j["status"] = "failed"
                j["completed_at"] = time.time()
                save_jobs(jobs)
                log.error("FAILOVER EXHAUSTED job=%s retries=%d/%d — permanently failed",
                          job_id, retries, max_retries)
                alert_job_failed(job_id, j.get("name", "?"), j.get("host_id"))
                return None

            old_host = j.get("host_id", "—")
            j["status"] = "queued"
            j["host_id"] = None
            j["started_at"] = None
            j["completed_at"] = None
            j["retries"] = retries
            save_jobs(jobs)

            log.warning("FAILOVER REQUEUE job=%s retry=%d/%d old_host=%s",
                        job_id, retries, max_retries, old_host)
            return j
    return None


def failover_dead_hosts():
    """
    Find all running jobs on dead hosts. Requeue them.
    This is the core failover loop — call it after check_hosts().
    Returns list of requeued jobs.
    """
    dead_host_ids = {h["host_id"] for h in list_hosts(active_only=False)
                     if h.get("status") == "dead"}

    if not dead_host_ids:
        return []

    jobs = load_jobs()
    requeued = []

    for j in jobs:
        if j["status"] == "running" and j.get("host_id") in dead_host_ids:
            log.warning("FAILOVER DETECTED job=%s on dead host=%s",
                        j["job_id"], j["host_id"])
            result = requeue_job(j["job_id"])
            if result:
                requeued.append(result)

    return requeued


def failover_and_reassign():
    """
    Full failover cycle:
    1. Check hosts (ping)
    2. Requeue jobs on dead hosts
    3. Process queue (assign requeued jobs to alive hosts)
    Returns (requeued_jobs, newly_assigned).
    """
    check_hosts()
    requeued = failover_dead_hosts()

    assigned = []
    if requeued:
        assigned = process_queue()
        for j, h in assigned:
            log.info("FAILOVER REASSIGNED job=%s -> host=%s", j["job_id"], h["host_id"])

    return requeued, assigned


def start_failover_monitor(interval=10, callback=None):
    """
    Run failover checks in a background thread.
    Ping hosts, requeue orphaned jobs, reassign.
    """
    def loop():
        while True:
            requeued, assigned = failover_and_reassign()
            if callback and (requeued or assigned):
                callback(requeued, assigned)
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


def build_image(model_name, context_dir=None, tag=None, quantize=None, base_image="python:3.11-slim"):
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

    log.info("BUILD START model=%s tag=%s dir=%s quantize=%s",
             model_name, tag, build_dir, quantize or "none")

    try:
        result = subprocess.run(
            ["docker", "build", "-t", tag, build_dir],
            capture_output=True, text=True, timeout=600,
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

    remote_tag = f"{registry}/{tag}" if "/" not in tag[:tag.index(":")] else f"{registry}/{tag.split('/')[-1]}"

    try:
        # Tag for remote
        subprocess.run(["docker", "tag", tag, remote_tag],
                       capture_output=True, text=True, timeout=30)

        # Push
        result = subprocess.run(
            ["docker", "push", remote_tag],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            log.error("PUSH FAILED tag=%s err=%s", remote_tag, result.stderr.strip())
            return remote_tag, False

        log.info("PUSH SUCCESS tag=%s", remote_tag)
        return remote_tag, True

    except Exception as e:
        log.error("PUSH ERROR tag=%s err=%s", remote_tag, e)
        return remote_tag, False


def build_and_push(model_name, context_dir=None, quantize=None, base_image="python:3.11-slim", push=True):
    """
    Full pipeline: generate Dockerfile, build image, optionally push.
    Returns dict with build results.
    """
    tag, built = build_image(model_name, context_dir=context_dir,
                              quantize=quantize, base_image=base_image)

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
            builds.append({
                "model": name,
                "path": build_dir,
                "has_dockerfile": has_dockerfile,
            })
    return builds


# ── Phase 17: Marketplace ────────────────────────────────────────────

MARKETPLACE_FILE = os.path.join(os.path.dirname(__file__), "marketplace.json")
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
            listings[i].update({
                "gpu_model": gpu_model,
                "vram_gb": vram_gb,
                "price_per_hour": price_per_hour,
                "description": description,
                "owner": owner,
                "updated_at": time.time(),
                "active": True,
            })
            save_marketplace(listings)
            log.info("MARKETPLACE UPDATED listing=%s | %s | $%s/hr",
                     host_id, gpu_model, price_per_hour)
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
    log.info("MARKETPLACE LISTED %s | %s | %sGB | $%s/hr | owner=%s",
             host_id, gpu_model, vram_gb, price_per_hour, owner)
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

    log.info("MARKETPLACE BILLED job=%s | total=$%s | host_payout=$%s | platform_fee=$%s",
             job_id, total_cost, host_payout, platform_fee)

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
        "platform_cut_pct": PLATFORM_CUT,
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


def register_host_ca(host_id, ip, gpu_model, total_vram_gb, free_vram_gb,
                      cost_per_hour=0.20, country="CA"):
    """Register a host with country tag. Defaults to Canada because why wouldn't it."""
    entry = register_host(host_id, ip, gpu_model, total_vram_gb, free_vram_gb, cost_per_hour)

    # Add country tag
    hosts = load_hosts(active_only=False)
    for h in hosts:
        if h["host_id"] == host_id:
            h["country"] = country.upper()
            break
    save_hosts(hosts)

    log.info("HOST REGISTERED (country=%s) %s | %s", country.upper(), host_id, ip)
    entry["country"] = country.upper()
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

    jobs = load_jobs()
    queued = [j for j in jobs if j["status"] == "queued"]
    queued.sort(key=lambda j: (-j["priority"], j["submitted_at"]))

    for job in queued:
        if not hosts:
            break

        host = allocate(job, hosts)
        if not host:
            continue

        update_job_status(job["job_id"], "running", host_id=host["host_id"])
        hosts = [h for h in hosts if h["host_id"] != host["host_id"]]
        assigned.append((job, host))

    return assigned


# ── Phase 19: Auto-Scaling ───────────────────────────────────────────

AUTOSCALE_ENABLED = os.environ.get("XCELSIOR_AUTOSCALE", "").lower() in ("1", "true", "yes")
AUTOSCALE_MAX_HOSTS = int(os.environ.get("XCELSIOR_AUTOSCALE_MAX", "20"))
AUTOSCALE_PROVIDER = os.environ.get("XCELSIOR_AUTOSCALE_PROVIDER", "")  # e.g. "ssh", "api"
AUTOSCALE_POOL_FILE = os.path.join(os.path.dirname(__file__), "autoscale_pool.json")


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
            p.update({"ip": ip, "gpu_model": gpu_model, "vram_gb": vram_gb,
                       "cost_per_hour": cost_per_hour, "country": country})
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
    In a real setup, this would spin up a cloud VM or wake a bare-metal node.
    For now: register it as active.
    """
    entry = register_host(
        pool_entry["host_id"],
        pool_entry["ip"],
        pool_entry["gpu_model"],
        pool_entry["vram_gb"],
        pool_entry["vram_gb"],  # starts fully free
        pool_entry.get("cost_per_hour", 0.20),
    )

    # Tag country
    hosts = load_hosts(active_only=False)
    for h in hosts:
        if h["host_id"] == pool_entry["host_id"]:
            h["country"] = pool_entry.get("country", "CA")
            h["autoscaled"] = True
            break
    save_hosts(hosts)

    # Mark as provisioned in pool
    pool = load_autoscale_pool()
    for p in pool:
        if p["host_id"] == pool_entry["host_id"]:
            p["provisioned"] = True
            break
    save_autoscale_pool(pool)

    log.info("AUTOSCALE PROVISIONED %s | %s | %sGB",
             pool_entry["host_id"], pool_entry["gpu_model"], pool_entry["vram_gb"])
    return entry


def deprovision_host(host_id):
    """
    Deprovision: remove from active hosts, mark pool entry as unprovisioned.
    Called when scaling down.
    """
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

    log.info("AUTOSCALE UP — provisioned %d hosts for %d queued jobs",
             len(provisioned), len(queued))
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
        print(f"  {h['host_id']} | {h['ip']} | {h['gpu_model']} | {h['free_vram_gb']}GB free | ${h['cost_per_hour']}/hr")

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
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Container started: {cname}")
            # Poll until done
            while True:
                inspect = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Status}}", cname],
                    capture_output=True, text=True,
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
        host = j['host_id'] or "—"
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
