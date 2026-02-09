# Excelsior: distributed GPU scheduler for Canadians who refuse to wait.
# Ever upward. No limits. Pure power.

import json
import os
import time

HOSTS_FILE = os.path.join(os.path.dirname(__file__), "hosts.json")


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


# ── Run it ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Register some test hosts
    register_host("rig-01", "192.168.1.10", "RTX 4090", 24, 24)
    register_host("rig-02", "192.168.1.11", "RTX 3090", 24, 16)
    register_host("rig-03", "192.168.1.12", "A100", 80, 80, cost_per_hour=0.50)

    print("=== REGISTERED HOSTS ===")
    for h in list_hosts():
        print(f"  {h['host_id']} | {h['ip']} | {h['gpu_model']} | {h['free_vram_gb']}GB free | ${h['cost_per_hour']}/hr")

    # Test allocation
    job = {"name": "llama3-70b", "vram_needed_gb": 40}
    result = allocate(job, list_hosts())
    print(f"\n=== ALLOCATE '{job['name']}' ===")
    if result:
        print(f"  -> {result['host_id']} ({result['gpu_model']}, {result['free_vram_gb']}GB free)")
    else:
        print("  -> No host available. Queue it.")

    job2 = {"name": "mistral-7b", "vram_needed_gb": 8}
    result2 = allocate(job2, list_hosts())
    print(f"\n=== ALLOCATE '{job2['name']}' ===")
    if result2:
        print(f"  -> {result2['host_id']} ({result2['gpu_model']}, {result2['free_vram_gb']}GB free)")
    else:
        print("  -> No host available. Queue it.")
