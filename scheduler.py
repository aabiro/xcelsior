# Excelsior: distributed GPU scheduler for Canadians who refuse to wait.
# Ever upward. No limits. Pure power.

import json
import os
import time
import uuid

HOSTS_FILE = os.path.join(os.path.dirname(__file__), "hosts.json")
JOBS_FILE = os.path.join(os.path.dirname(__file__), "jobs.json")


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
    return best


# ── Phase 2: Host Registry ───────────────────────────────────────────

def load_hosts():
    """Load hosts from JSON. No file? Empty list. Simple."""
    if not os.path.exists(HOSTS_FILE):
        return []
    with open(HOSTS_FILE, "r") as f:
        return json.load(f)


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
            return entry

    hosts.append(entry)
    save_hosts(hosts)
    return entry


def remove_host(host_id):
    """Host is dead. Remove it. No funeral."""
    hosts = load_hosts()
    hosts = [h for h in hosts if h["host_id"] != host_id]
    save_hosts(hosts)


def list_hosts(active_only=True):
    """Show what we've got."""
    hosts = load_hosts()
    if active_only:
        return [h for h in hosts if h.get("status") == "active"]
    return hosts


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
    }

    jobs.append(job)
    save_jobs(jobs)
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
            j["status"] = status
            if host_id:
                j["host_id"] = host_id
            if status == "running":
                j["started_at"] = time.time()
            if status in ("completed", "failed"):
                j["completed_at"] = time.time()
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


# ── Run it ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Clean slate
    for f in (HOSTS_FILE, JOBS_FILE):
        if os.path.exists(f):
            os.remove(f)

    # Phase 2: Register hosts
    register_host("rig-01", "192.168.1.10", "RTX 4090", 24, 24)
    register_host("rig-02", "192.168.1.11", "RTX 3090", 24, 16)
    register_host("rig-03", "192.168.1.12", "A100", 80, 80, cost_per_hour=0.50)

    print("=== REGISTERED HOSTS ===")
    for h in list_hosts():
        print(f"  {h['host_id']} | {h['ip']} | {h['gpu_model']} | {h['free_vram_gb']}GB free | ${h['cost_per_hour']}/hr")

    # Phase 3: Submit jobs
    submit_job("llama3-70b", vram_needed_gb=40, priority=2)
    submit_job("mistral-7b", vram_needed_gb=8, priority=0)
    submit_job("codellama-34b", vram_needed_gb=20, priority=1)
    submit_job("mega-model-200b", vram_needed_gb=160, priority=3)  # nothing can run this

    print("\n=== JOB QUEUE ===")
    for j in list_jobs():
        print(f"  [{j['status']:>9}] {j['job_id']} | {j['name']} | {j['vram_needed_gb']}GB | priority {j['priority']}")

    # Process queue
    assigned = process_queue()

    print("\n=== ASSIGNED ===")
    for job, host in assigned:
        print(f"  {job['name']} -> {host['host_id']} ({host['gpu_model']})")

    print("\n=== FINAL JOB STATUS ===")
    for j in list_jobs():
        host = j['host_id'] or "—"
        print(f"  [{j['status']:>9}] {j['job_id']} | {j['name']} | host: {host}")

    # Cleanup test files
    for f in (HOSTS_FILE, JOBS_FILE):
        if os.path.exists(f):
            os.remove(f)
