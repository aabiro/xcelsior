#!/usr/bin/env python3
"""Live spot preemption validation on local test env + RTX 2060 worker.

Prerequisites:
  bash scripts/deploy.sh --test
  bash scripts/run_worker_test.sh   # in another terminal

Exits 0 when spot runs, on-demand preempts spot, and spot_rate_cad is locked.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time

import requests

DEFAULT_BASE = os.environ.get("XCELSIOR_TEST_BASE", "http://localhost:9501")
DEFAULT_HOST = os.environ.get("XCELSIOR_TEST_HOST_ID", "aaryn-tuf-rtx2060")
DEFAULT_GPU = os.environ.get("XCELSIOR_TEST_GPU_MODEL", "RTX 2060")
LAUNCH_IMAGE = "nvidia/cuda:12.0.0-base-ubuntu22.04"
POLL_SEC = 5
POLL_MAX = 300


def _poll(session: requests.Session, base: str, job_id: str) -> dict:
    deadline = time.time() + POLL_MAX
    last: dict = {}
    while time.time() < deadline:
        r = session.get(f"{base}/instance/{job_id}", timeout=30)
        if r.status_code == 200:
            last = r.json().get("instance") or {}
            if last.get("status") in ("running", "failed", "stopped", "cancelled", "completed", "queued"):
                return last
        time.sleep(POLL_SEC)
    return last


def _wait_status(session: requests.Session, base: str, job_id: str, want: set[str]) -> dict:
    deadline = time.time() + POLL_MAX
    while time.time() < deadline:
        r = session.get(f"{base}/instance/{job_id}", timeout=30)
        if r.status_code == 200:
            inst = r.json().get("instance") or {}
            if inst.get("status") in want:
                return inst
        time.sleep(POLL_SEC)
    raise TimeoutError(f"job {job_id} did not reach {want} within {POLL_MAX}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Spot preemption live test (test env)")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--host-id", default=DEFAULT_HOST)
    parser.add_argument("--gpu-model", default=DEFAULT_GPU)
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    results: dict[str, object] = {"base": base, "host_id": args.host_id}

    health = requests.get(f"{base}/healthz", timeout=10)
    if health.status_code != 200:
        print(json.dumps({"pass": False, "error": "test API not healthy"}, indent=2))
        return 1

    spot_flag = requests.get(f"{base}/api/pricing/spot-enabled", timeout=10)
    results["spot_enabled"] = spot_flag.json().get("enabled") if spot_flag.status_code == 200 else None

    email = f"spot-live-{secrets.token_hex(4)}@xcelsior.ca"
    password = f"TestPass!{secrets.token_hex(6)}"
    s = requests.Session()
    s.headers["Content-Type"] = "application/json"

    reg = s.post(
        f"{base}/api/auth/register",
        json={"email": email, "password": password, "name": "Spot Live Test"},
        timeout=30,
    )
    if reg.status_code not in (200, 409):
        results["register"] = reg.status_code
        results["pass"] = False
        print(json.dumps(results, indent=2))
        return 1

    login = s.post(f"{base}/api/auth/login", json={"email": email, "password": password}, timeout=30)
    if login.status_code != 200:
        results["login"] = login.status_code
        results["pass"] = False
        print(json.dumps(results, indent=2))
        return 1
    login_body = login.json()
    token = login_body["access_token"]
    user = login_body.get("user") or {}
    customer_id = (
        user.get("billing_customer_id")
        or user.get("customer_id")
        or login_body.get("customer_id")
        or email
    )
    s.headers["Authorization"] = f"Bearer {token}"

    dep = s.post(
        f"{base}/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "spot preemption live test"},
        timeout=30,
    )
    results["deposit"] = dep.status_code
    if dep.status_code != 200:
        results["pass"] = False
        print(json.dumps(results, indent=2))
        return 1

    hosts = s.get(f"{base}/hosts", timeout=30)
    host_list = hosts.json().get("hosts") or [] if hosts.status_code == 200 else []
    target = next((h for h in host_list if h.get("host_id") == args.host_id), None)
    results["host_registered"] = target is not None
    results["host_admitted"] = bool(target and target.get("admitted"))
    if not target or not target.get("admitted"):
        results["pass"] = False
        results["hint"] = "Start test worker: bash scripts/run_worker_test.sh"
        print(json.dumps(results, indent=2))
        return 1

    # Do not pass host_id — direct assignment SSH-launches from the API container and
    # fails locally; queue + worker agent is the supported test path.
    spot_launch = s.post(
        f"{base}/instance",
        json={
            "name": f"spot-live-{int(time.time())}",
            "vram_needed_gb": 4.0,
            "pricing_mode": "spot",
            "gpu_model": args.gpu_model,
            "image": LAUNCH_IMAGE,
        },
        timeout=30,
    )
    results["spot_launch_status"] = spot_launch.status_code
    if spot_launch.status_code != 200:
        results["spot_launch_body"] = spot_launch.text[:400]
        results["pass"] = False
        print(json.dumps(results, indent=2))
        return 1

    spot_job_id = spot_launch.json()["instance"]["job_id"]
    results["spot_job_id"] = spot_job_id
    s.post(f"{base}/queue/process", timeout=30)

    for _ in range(12):
        s.post(f"{base}/queue/process", timeout=30)
        spot_inst = _poll(s, base, spot_job_id)
        if spot_inst.get("status") in ("running", "assigned"):
            break
        if spot_inst.get("status") in ("failed", "cancelled"):
            results["spot_failed_status"] = spot_inst.get("status")
            results["pass"] = False
            print(json.dumps(results, indent=2))
            return 1
        time.sleep(POLL_SEC)
    else:
        spot_inst = _poll(s, base, spot_job_id)

    results["spot_running_status"] = spot_inst.get("status")
    results["spot_rate_cad"] = spot_inst.get("spot_rate_cad")
    results["spot_pricing_mode"] = spot_inst.get("pricing_mode")

    if spot_inst.get("status") not in ("running", "assigned"):
        results["pass"] = False
        print(json.dumps(results, indent=2))
        return 1

    if not spot_inst.get("spot_rate_cad"):
        for _ in range(24):
            time.sleep(POLL_SEC)
            spot_inst = _poll(s, base, spot_job_id)
            if spot_inst.get("spot_rate_cad"):
                break
        results["spot_rate_cad"] = spot_inst.get("spot_rate_cad")

    od_launch = s.post(
        f"{base}/instance",
        json={
            "name": f"od-live-{int(time.time())}",
            "vram_needed_gb": 4.0,
            "pricing_mode": "on_demand",
            "gpu_model": args.gpu_model,
            "image": LAUNCH_IMAGE,
        },
        timeout=30,
    )
    results["od_launch_status"] = od_launch.status_code
    if od_launch.status_code != 200:
        results["od_launch_body"] = od_launch.text[:400]
        results["pass"] = False
        print(json.dumps(results, indent=2))
        return 1

    od_job_id = od_launch.json()["instance"]["job_id"]
    results["od_job_id"] = od_job_id
    s.post(f"{base}/queue/process", timeout=30)

    spot_after = _wait_status(s, base, spot_job_id, {"queued", "preempted"})
    results["spot_after_status"] = spot_after.get("status")
    results["spot_preemption_count"] = spot_after.get("preemption_count")

    od_inst = _poll(s, base, od_job_id)
    results["od_status"] = od_inst.get("status")

    preempt_ok = (
        spot_after.get("status") == "queued"
        and int(spot_after.get("preemption_count") or 0) >= 1
        and spot_after.get("pricing_mode") == "spot"
    )
    od_ok = od_inst.get("status") in ("running", "assigned", "starting")
    rate_ok = bool(results.get("spot_rate_cad"))

    results["pass"] = preempt_ok and od_ok and rate_ok

    if not args.skip_cleanup:
        for jid in (spot_job_id, od_job_id):
            s.post(f"{base}/instances/{jid}/cancel", timeout=30)

    print(json.dumps(results, indent=2))
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())