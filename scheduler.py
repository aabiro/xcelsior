# Xcelsior: distributed GPU scheduler for Canadians who refuse to wait.
# Ever upward. No limits. Pure power.

import json
import logging
import os
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

    best = max(candidates, key=lambda h: (h.get("free_vram_gb", 0), -h.get("latency_ms", 999)))
    log.info("ALLOCATE job=%s -> host=%s (%s, %sGB free)",
             job.get("name", "?"), best["host_id"], best.get("gpu_model"), best.get("free_vram_gb"))
    return best


# ── Phase 2: Host Registry ───────────────────────────────────────────

def load_hosts(active_only=False):
    """Load hosts from JSON. No file? Empty list. Simple."""
    if not os.path.exists(HOSTS_FILE):
        return []
    with open(HOSTS_FILE, "r") as f:
        hosts = json.load(f)
    if active_only:
        return [h for h in hosts if h.get("status") == "active"]
    return hosts


def save_hosts(hosts):
    """Write hosts to JSON. Atomic enough for now."""
    with open(HOSTS_FILE, "w") as f:
        json.dump(hosts, f, indent=2)


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


# ── Phase 3: Job Queue ────────────────────────────────────────────────

VALID_STATUSES = ("queued", "running", "completed", "failed")


def load_jobs():
    """Load jobs from JSON. No file? Empty list."""
    if not os.path.exists(JOBS_FILE):
        return []
    with open(JOBS_FILE, "r") as f:
        return json.load(f)


def save_jobs(jobs):
    """Write jobs to JSON."""
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)


def submit_job(name, vram_needed_gb, priority=0):
    """
    Submit a job to the queue.
    Each job has: model name, VRAM needed, priority.
    FIFO within the same priority. Higher priority goes first.
    """
    jobs = load_jobs()

    job = {
        "job_id": str(uuid.uuid4())[:8],
        "name": name,
        "vram_needed_gb": vram_needed_gb,
        "priority": priority,
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
    log.info("JOB SUBMITTED %s | %s | %sGB VRAM | priority %s", job["job_id"], name, vram_needed_gb, priority)
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

    cmd = (
        f"docker run -d --gpus all "
        f"--name {container_name} "
        f"{image}"
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
    cmd = f"docker inspect -f '{{{{.State.Running}}}}' {container_name}"
    rc, stdout, _ = ssh_exec(host["ip"], cmd)
    return rc == 0 and stdout == "true"


def kill_job(job, host):
    """
    Kill the container. Clean up. Mark job COMPLETE.
    No lingering processes. No orphaned containers.
    """
    container_name = job.get("container_name", f"xcl-{job['job_id']}")

    # Kill it
    ssh_exec(host["ip"], f"docker kill {container_name}")
    # Remove it
    ssh_exec(host["ip"], f"docker rm -f {container_name}")
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

    while True:
        # Check if container still exists
        cmd = f"docker inspect -f '{{{{.State.Status}}}}' {container_name}"
        rc, stdout, _ = ssh_exec(host["ip"], cmd)

        if rc != 0:
            # Container gone entirely
            update_job_status(job["job_id"], "failed")
            return "failed"

        if stdout == "exited":
            # Check exit code
            cmd_exit = f"docker inspect -f '{{{{.State.ExitCode}}}}' {container_name}"
            _, exit_code, _ = ssh_exec(host["ip"], cmd_exit)
            # Clean up
            ssh_exec(host["ip"], f"docker rm -f {container_name}")

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
    if not os.path.exists(BILLING_FILE):
        return []
    with open(BILLING_FILE, "r") as f:
        return json.load(f)


def save_billing(records):
    """Write billing records."""
    with open(BILLING_FILE, "w") as f:
        json.dump(records, f, indent=2)


def bill_job(job_id):
    """
    Bill a completed job. Multiply time by rate. Charge.
    start_time, end_time, duration, cost. That's it.
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

    duration_sec = job["completed_at"] - job["started_at"]
    duration_hr = duration_sec / 3600
    cost = round(duration_hr * rate, 4)

    record = {
        "job_id": job_id,
        "job_name": job["name"],
        "host_id": job.get("host_id"),
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "duration_sec": round(duration_sec, 2),
        "rate_per_hour": rate,
        "cost": cost,
        "billed_at": time.time(),
    }

    records = load_billing()
    records.append(record)
    save_billing(records)

    log.info("BILLED job=%s | %s | %.1fs | $%.4f @ $%s/hr",
             job_id, job["name"], duration_sec, cost, rate)
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
