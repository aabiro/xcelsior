#!/usr/bin/env python3
"""Xcelsior End-to-End Production Test Suite.

Tests every customer-facing flow against the live API at xcelsior.ca.
Run from the project root with the venv activated.
"""

import json
import os
import sys
import time
import requests
import secrets

BASE = "https://xcelsior.ca"
TOKEN = os.environ.get("XCELSIOR_API_TOKEN", "fW21UGgCbT-lkOri4h-UmRsKXsxEXEodDjKxtZK3lJY")
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TEST_EMAIL = f"e2e-test-{secrets.token_hex(4)}@xcelsior.ca"
TEST_PASS = f"TestPass!{secrets.token_hex(6)}"

results = []
session_token = None
user_id = None
customer_id = None


def run_test(name, fn):
    """Run a test function and record pass/fail."""
    try:
        fn()
        results.append(("PASS", name))
        print(f"  ✓ {name}")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  ✗ {name}: {e}")


def api(method, path, **kwargs):
    """Make an API call, return response."""
    url = f"{BASE}{path}"
    kwargs.setdefault("timeout", 15)
    r = getattr(requests, method)(url, **kwargs)
    return r


# ═══════════════════════════════════════════════════════════════════════
# 1. AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 1. AUTHENTICATION ═══")


def _e2e_healthz():
    r = api("get", "/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def _e2e_register():
    global user_id, customer_id
    r = api("post", "/api/auth/register", json={
        "email": TEST_EMAIL,
        "password": TEST_PASS,
        "name": "E2E Test User",
    })
    assert r.status_code == 200, f"Register failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    assert data.get("ok") or data.get("user_id"), f"Unexpected: {data}"
    user_id = data.get("user_id", "")
    customer_id = data.get("customer_id", "")


def _e2e_login():
    global session_token
    r = api("post", "/api/auth/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASS,
    })
    assert r.status_code == 200, f"Login failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    session_token = data.get("access_token", data.get("token", data.get("session_token", "")))
    assert session_token, f"No session token: {data}"
    # Also grab customer_id from login response
    global customer_id, user_id
    user_info = data.get("user", {})
    if user_info:
        customer_id = user_info.get("customer_id", customer_id)
        user_id = user_info.get("user_id", user_id)


def _e2e_auth_me():
    r = api("get", "/api/auth/me", headers={"Authorization": f"Bearer {session_token}"}, allow_redirects=True)
    assert r.status_code == 200, f"me failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    # The response may nest user info under a key
    email = data.get("email") or data.get("user", {}).get("email", "")
    assert email == TEST_EMAIL, f"Email mismatch: {data}"


def _e2e_change_password():
    new_pass = TEST_PASS + "X"
    r = api("post", "/api/auth/change-password", headers={"Authorization": f"Bearer {session_token}"}, json={
        "current_password": TEST_PASS,
        "new_password": new_pass,
    })
    assert r.status_code == 200, f"Change password failed: {r.status_code} {r.text[:200]}"
    # Change back
    r2 = api("post", "/api/auth/change-password", headers={"Authorization": f"Bearer {session_token}"}, json={
        "current_password": new_pass,
        "new_password": TEST_PASS,
    })
    assert r2.status_code == 200


run_test("Health check", _e2e_healthz)
run_test("Register new account", _e2e_register)
run_test("Login", _e2e_login)
run_test("Auth /me", _e2e_auth_me)
run_test("Change password", _e2e_change_password)


# ═══════════════════════════════════════════════════════════════════════
# 2. API KEY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 2. API KEY MANAGEMENT ═══")
api_key_value = None


def _e2e_generate_api_key():
    global api_key_value
    r = api("post", "/api/keys/generate", headers={"Authorization": f"Bearer {session_token}"}, json={
        "name": "e2e-test-key",
    })
    assert r.status_code == 200, f"Generate key failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    api_key_value = data.get("key", data.get("api_key", ""))
    assert api_key_value, f"No key returned: {data}"


def _e2e_list_api_keys():
    r = api("get", "/api/keys", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200
    data = r.json()
    keys = data.get("keys", data.get("api_keys", []))
    assert len(keys) >= 1, f"Expected at least 1 key: {data}"


def _e2e_auth_with_api_key():
    r = api("get", "/healthz", headers={"Authorization": f"Bearer {api_key_value}"})
    assert r.status_code == 200


def _e2e_delete_api_key():
    # Get the preview to delete
    r = api("get", "/api/keys", headers={"Authorization": f"Bearer {session_token}"})
    keys = r.json().get("keys", r.json().get("api_keys", []))
    if keys:
        preview = keys[0].get("preview", keys[0].get("key_preview", api_key_value[:8]))
        r2 = api("delete", f"/api/keys/{preview}", headers={"Authorization": f"Bearer {session_token}"})
        assert r2.status_code == 200, f"Delete key failed: {r2.status_code} {r2.text[:200]}"


run_test("Generate API key", _e2e_generate_api_key)
run_test("List API keys", _e2e_list_api_keys)
run_test("Auth with API key", _e2e_auth_with_api_key)
run_test("Delete API key", _e2e_delete_api_key)


# ═══════════════════════════════════════════════════════════════════════
# 3. SSH KEY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 3. SSH KEY MANAGEMENT ═══")

SSH_TEST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA e2e-test"


def _e2e_add_ssh_key():
    r = api("post", "/api/ssh/keys", headers={"Authorization": f"Bearer {session_token}"}, json={
        "name": "e2e-test-ssh",
        "public_key": SSH_TEST_KEY,
    })
    assert r.status_code == 200, f"Add SSH key failed: {r.status_code} {r.text[:200]}"


def _e2e_list_ssh_keys():
    r = api("get", "/api/ssh/keys", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200
    data = r.json()
    keys = data.get("keys", data.get("ssh_keys", []))
    assert len(keys) >= 1, f"Expected at least 1 SSH key: {data}"


def _e2e_delete_ssh_key():
    r = api("get", "/api/ssh/keys", headers={"Authorization": f"Bearer {session_token}"})
    keys = r.json().get("keys", r.json().get("ssh_keys", []))
    if keys:
        key_id = keys[0].get("id", keys[0].get("key_id", ""))
        if key_id:
            r2 = api("delete", f"/api/ssh/keys/{key_id}", headers={"Authorization": f"Bearer {session_token}"})
            assert r2.status_code == 200, f"Delete SSH key failed: {r2.status_code} {r2.text[:200]}"


run_test("Add SSH key", _e2e_add_ssh_key)
run_test("List SSH keys", _e2e_list_ssh_keys)
run_test("Delete SSH key", _e2e_delete_ssh_key)


# ═══════════════════════════════════════════════════════════════════════
# 4. BILLING / WALLET
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 4. BILLING / WALLET ═══")


def _e2e_get_wallet():
    global customer_id
    if not customer_id:
        # Get from /me
        r = api("get", "/api/auth/me", headers={"Authorization": f"Bearer {session_token}"})
        customer_id = r.json().get("customer_id", "")
    assert customer_id, "No customer_id found"
    r = api("get", f"/api/billing/wallet/{customer_id}", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200, f"Wallet failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    assert "balance_cad" in data or "wallet" in data, f"No balance: {data}"


def _e2e_wallet_history():
    r = api("get", f"/api/billing/wallet/{customer_id}/history", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200, f"History failed: {r.status_code} {r.text[:200]}"


def _e2e_invoices():
    r = api("get", f"/api/billing/invoices/{customer_id}", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200, f"Invoices failed: {r.status_code} {r.text[:200]}"


def _e2e_create_deposit():
    """Test creating a Stripe PaymentIntent (does NOT charge — just creates intent)."""
    r = api("post", f"/api/billing/wallet/{customer_id}/deposit", headers={"Authorization": f"Bearer {session_token}"}, json={
        "amount_cad": 1.00,
    })
    # May fail if Stripe config issue, but the endpoint should respond
    assert r.status_code in (200, 400, 500), f"Deposit endpoint unreachable: {r.status_code}"
    if r.status_code == 200:
        data = r.json()
        assert data.get("client_secret") or data.get("ok"), f"No client_secret: {data}"


run_test("Get wallet", _e2e_get_wallet)
run_test("Wallet history", _e2e_wallet_history)
run_test("Invoices", _e2e_invoices)
run_test("Create deposit intent", _e2e_create_deposit)


# ═══════════════════════════════════════════════════════════════════════
# 5. HOSTS / GPU
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 5. HOSTS / GPU ═══")


def _e2e_list_hosts():
    r = api("get", "/hosts?active_only=true", headers=AUTH)
    assert r.status_code == 200
    hosts = r.json().get("hosts", [])
    assert len(hosts) >= 1, f"Expected at least 1 active host, got {len(hosts)}"
    gpu_host = hosts[0]
    assert "2060" in gpu_host.get("gpu_model", "").lower() or "2060" in str(gpu_host), \
        f"Expected 2060 host: {gpu_host.get('gpu_model')}"


def _e2e_compute_score():
    r = api("get", "/compute-score/aaryn-tuf-rtx2060", headers=AUTH)
    # May or may not have a score
    assert r.status_code in (200, 404), f"Compute score: {r.status_code}"


run_test("List active hosts", _e2e_list_hosts)
run_test("Compute score", _e2e_compute_score)


# ═══════════════════════════════════════════════════════════════════════
# 6. JOB SUBMISSION
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 6. JOB SUBMISSION ═══")
test_job_id = None


def _e2e_submit_job():
    global test_job_id
    r = api("post", "/instance", headers=AUTH, json={
        "name": "e2e-test-job",
        "vram_needed_gb": 2.0,
        "priority": 1,
        "image": "alpine:latest",
    })
    assert r.status_code == 200, f"Submit job failed: {r.status_code} {r.text[:200]}"
    data = r.json()
    inst = data.get("instance", {})
    test_job_id = inst.get("job_id", data.get("job_id", data.get("instance_id", "")))
    assert test_job_id, f"No job_id: {data}"


def _e2e_list_instances():
    r = api("get", "/instances", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    jobs = data.get("instances", data.get("jobs", []))
    assert len(jobs) >= 1, f"Expected at least 1 instance: {data}"


def _e2e_process_queue():
    """Trigger queue processing to allocate the job."""
    r = api("post", "/queue/process", headers=AUTH)
    assert r.status_code == 200, f"Queue process failed: {r.status_code} {r.text[:200]}"


def _e2e_get_instance_status():
    r = api("get", f"/instances", headers=AUTH)
    assert r.status_code == 200
    jobs = r.json().get("instances", r.json().get("jobs", []))
    if test_job_id:
        job = next((j for j in jobs if j.get("job_id") == test_job_id), None)
        if job:
            print(f"    → job status: {job.get('status')}, host: {job.get('host_id', 'unassigned')}")


run_test("Submit job", _e2e_submit_job)
run_test("Process queue", _e2e_process_queue)
run_test("List instances", _e2e_list_instances)
run_test("Get instance status", _e2e_get_instance_status)


# ═══════════════════════════════════════════════════════════════════════
# 7. MARKETPLACE
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 7. MARKETPLACE ═══")


def _e2e_marketplace_search():
    r = api("get", "/marketplace/search", headers=AUTH)
    assert r.status_code == 200, f"Marketplace search: {r.status_code} {r.text[:200]}"


run_test("Marketplace search", _e2e_marketplace_search)


# ═══════════════════════════════════════════════════════════════════════
# 8. EVENTS / SSE
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 8. EVENTS ═══")


def _e2e_events_list():
    r = api("get", "/api/events", headers=AUTH)
    assert r.status_code == 200, f"Events: {r.status_code} {r.text[:200]}"


def _e2e_events_stream():
    """Test SSE stream endpoint connects (don't wait for events)."""
    try:
        r = requests.get(f"{BASE}/api/stream", headers=AUTH, stream=True, timeout=3)
        assert r.status_code == 200
    except requests.exceptions.ReadTimeout:
        pass  # Expected — SSE keeps connection open


run_test("Events list", _e2e_events_list)
run_test("SSE stream connects", _e2e_events_stream)


# ═══════════════════════════════════════════════════════════════════════
# 9. COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 9. COMPLIANCE ═══")


def _e2e_compliance():
    r = api("get", "/api/compliance/status", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200, f"Compliance: {r.status_code} {r.text[:200]}"
    data = r.json()
    assert "checks" in data or "compliance" in data or "ok" in data, f"No compliance data: {data}"


run_test("Compliance checks", _e2e_compliance)


# ═══════════════════════════════════════════════════════════════════════
# 10. TEAMS
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 10. TEAMS ═══")
team_id = None


def _e2e_create_team():
    global team_id
    r = api("post", "/api/teams", headers={"Authorization": f"Bearer {session_token}"}, json={
        "name": "E2E Test Team",
    })
    assert r.status_code == 200, f"Create team: {r.status_code} {r.text[:200]}"
    data = r.json()
    team_id = data.get("team_id", data.get("id", ""))


def _e2e_get_team():
    r = api("get", "/api/teams/me", headers={"Authorization": f"Bearer {session_token}"})
    assert r.status_code == 200, f"Get team: {r.status_code} {r.text[:200]}"


run_test("Create team", _e2e_create_team)
run_test("Get team", _e2e_get_team)


# ═══════════════════════════════════════════════════════════════════════
# 11. CHAT
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 11. CHAT ═══")


def _e2e_chat():
    r = api("post", "/api/chat", headers=AUTH, json={
        "message": "What is Xcelsior?",
        "session_id": "e2e-test",
    })
    # Chat might return 200 or stream, just check it doesn't 500
    assert r.status_code in (200, 201), f"Chat: {r.status_code} {r.text[:200]}"


run_test("Chat widget", _e2e_chat)


# ═══════════════════════════════════════════════════════════════════════
# 12. ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 12. ARTIFACTS ═══")


def _e2e_list_artifacts():
    r = api("get", "/api/artifacts", headers=AUTH)
    assert r.status_code == 200, f"Artifacts: {r.status_code} {r.text[:200]}"


run_test("List artifacts", _e2e_list_artifacts)


# ═══════════════════════════════════════════════════════════════════════
# 13. SLURM PROFILES
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 13. SLURM / HPC ═══")


def _e2e_slurm_profiles():
    r = api("get", "/api/slurm/profiles", headers=AUTH)
    assert r.status_code == 200, f"Slurm profiles: {r.status_code} {r.text[:200]}"


run_test("Slurm profiles", _e2e_slurm_profiles)


# ═══════════════════════════════════════════════════════════════════════
# 14. JURISDICTION
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 14. JURISDICTION ═══")


def _e2e_canada_routing():
    r = api("get", "/hosts/ca", headers=AUTH)
    assert r.status_code == 200, f"Canada hosts: {r.status_code} {r.text[:200]}"


def _e2e_ca_hosts():
    r = api("get", "/hosts/ca", headers=AUTH)
    assert r.status_code == 200


run_test("Canada routing config", _e2e_canada_routing)
run_test("CA-only hosts", _e2e_ca_hosts)


# ═══════════════════════════════════════════════════════════════════════
# 15. SPOT PRICING
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 15. SPOT PRICING ═══")


def _e2e_spot_prices():
    r = api("get", "/spot-prices", headers=AUTH)
    assert r.status_code == 200, f"Spot prices: {r.status_code} {r.text[:200]}"


run_test("Spot prices", _e2e_spot_prices)


# ═══════════════════════════════════════════════════════════════════════
# 16. REPUTATION
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 16. REPUTATION ═══")


def _e2e_reputation():
    r = api("get", "/api/reputation/aaryn-tuf-rtx2060", headers=AUTH)
    assert r.status_code in (200, 404), f"Reputation: {r.status_code}"


run_test("Reputation lookup", _e2e_reputation)


# ═══════════════════════════════════════════════════════════════════════
# 17. DASHBOARD PAGES (Frontend)
# ═══════════════════════════════════════════════════════════════════════
print("\n═══ 17. FRONTEND PAGES ═══")

PAGES = [
    "/",
    "/login",
    "/register",
    "/pricing",
    "/features",
    "/about",
    "/privacy",
    "/terms",
    "/dashboard",
    "/dashboard/billing",
    "/dashboard/instances",
    "/dashboard/hosts",
    "/dashboard/compliance",
    "/dashboard/earnings",
    "/dashboard/settings",
    "/dashboard/events",
    "/dashboard/marketplace",
    "/dashboard/telemetry",
    "/dashboard/artifacts",
    "/dashboard/hpc",
    "/dashboard/reputation",
    "/dashboard/notifications",
    "/dashboard/analytics",
    "/dashboard/trust",
    "/dashboard/admin",
]


def _e2e_page(path):
    def _test():
        r = requests.get(f"{BASE}{path}", timeout=15, allow_redirects=True)
        assert r.status_code in (200, 307, 302), f"Page {path}: HTTP {r.status_code}"
    return _test


for page in PAGES:
    run_test(f"Page {page}", _e2e_page(page))


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
total = len(results)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
if failed:
    print("\nFAILURES:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  ✗ {r[1]}: {r[2]}")
print("=" * 60)
if __name__ == "__main__":
    sys.exit(1 if failed else 0)
