# Excelsior: distributed GPU scheduler for Canadians who refuse to wait.
# Ever upward. No limits. Pure power.

import json
import os
import subprocess
import threading
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
            h["status"] = "active"
            results[h["host_id"]] = "alive"
        else:
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


# ── Phase 5 & 6: Run Job / Kill Job ──────────────────────────────────

def ssh_exec(ip, cmd):
    """Run a command on a remote host via SSH. Returns (returncode, stdout, stderr)."""
    full_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        f"root@{ip}", cmd,
    ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def run_job(job, host, docker_image=None):
    """
    SSH into the host. docker run the model. Log the container ID.
    Returns the container ID or None on failure.
    """
    image = docker_image or f"excelsior/{job['name']}:latest"
    container_name = f"xcl-{job['job_id']}"

    cmd = (
        f"docker run -d --gpus all "
        f"--name {container_name} "
        f"{image}"
    )

    rc, stdout, stderr = ssh_exec(host["ip"], cmd)
    if rc != 0:
        update_job_status(job["job_id"], "failed")
        return None

    container_id = stdout[:12]

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
    image = docker_image or f"excelsior/{job['name']}:latest"
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

    # Cleanup
    for f in (HOSTS_FILE, JOBS_FILE):
        if os.path.exists(f):
            os.remove(f)
