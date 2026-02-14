#!/usr/bin/env python3
# Xcelsior CLI v1.0.0
# argparse. No npm. No node. Just Python.

import argparse
import json
import os
import secrets
import sys
import time

from scheduler import (
    register_host, remove_host, list_hosts, check_hosts,
    submit_job, list_jobs, update_job_status, process_queue,
    bill_job, bill_all_completed, get_total_revenue, load_billing,
    start_health_monitor,
    generate_ssh_keypair, get_public_key,
    failover_and_reassign, requeue_job,
    list_tiers, PRIORITY_TIERS,
    build_and_push, list_builds, generate_dockerfile,
    list_rig, unlist_rig, get_marketplace, marketplace_bill, marketplace_stats,
    register_host_ca, list_hosts_filtered, process_queue_filtered,
    set_canada_only,
    add_to_pool, remove_from_pool, load_autoscale_pool,
    autoscale_cycle,
)


def cmd_run(args):
    """Submit a job and optionally process the queue."""
    job = submit_job(args.model, args.vram, args.priority, tier=args.tier,
                     num_gpus=getattr(args, 'gpus', 1),
                     nfs_server=getattr(args, 'nfs_server', None),
                     nfs_path=getattr(args, 'nfs_path', None),
                     image=getattr(args, 'image', None))
    gpus_str = f" | {job.get('num_gpus', 1)} GPUs" if job.get('num_gpus', 1) > 1 else ""
    print(f"Job submitted: {job['job_id']} | {job['name']} | {job['vram_needed_gb']}GB{gpus_str} | tier={job['tier']} (priority {job['priority']})")

    if not args.no_assign:
        assigned = process_queue()
        if assigned:
            for j, h in assigned:
                print(f"  Assigned: {j['name']} -> {h['host_id']} ({h['gpu_model']})")
        else:
            print("  No host available. Job queued.")


def cmd_jobs(args):
    """List jobs."""
    jobs = list_jobs(status=args.status)
    if not jobs:
        print("No jobs.")
        return
    for j in jobs:
        host = j.get("host_id") or "—"
        tier = j.get("tier", "free")
        print(f"  [{j['status']:>9}] {j['job_id']} | {j['name']} | {j['vram_needed_gb']}GB | {tier} | host: {host}")


def cmd_job(args):
    """Get a specific job."""
    for j in list_jobs():
        if j["job_id"] == args.job_id:
            print(json.dumps(j, indent=2))
            return
    print(f"Job {args.job_id} not found.", file=sys.stderr)
    sys.exit(1)


def cmd_cancel(args):
    """Cancel a job."""
    update_job_status(args.job_id, "failed")
    print(f"Job {args.job_id} cancelled.")


def cmd_process(args):
    """Process the queue."""
    assigned = process_queue()
    if not assigned:
        print("Nothing to assign.")
        return
    for j, h in assigned:
        print(f"  {j['name']} ({j['job_id']}) -> {h['host_id']} ({h['gpu_model']})")


def cmd_host_add(args):
    """Register a host."""
    entry = register_host(args.id, args.ip, args.gpu, args.vram, args.free_vram, args.rate,
                          country=getattr(args, 'country', 'CA'),
                          province=getattr(args, 'province', ''))
    prov = entry.get('province', '')
    print(f"Host registered: {entry['host_id']} | {entry['ip']} | {entry['gpu_model']} | {entry.get('country', 'CA')}{(' ' + prov) if prov else ''} | {entry['total_vram_gb']}GB | ${entry['cost_per_hour']}/hr")


def cmd_host_rm(args):
    """Remove a host."""
    remove_host(args.id)
    print(f"Host removed: {args.id}")


def cmd_hosts(args):
    """List hosts."""
    hosts = list_hosts(active_only=not args.all)
    if not hosts:
        print("No hosts.")
        return
    for h in hosts:
        print(f"  [{h['status']:>6}] {h['host_id']} | {h['ip']} | {h['gpu_model']} | {h['free_vram_gb']}GB free | ${h['cost_per_hour']}/hr")


def cmd_ping(args):
    """Health check all hosts."""
    results = check_hosts()
    for host_id, status in results.items():
        print(f"  {host_id}: {status}")


def cmd_bill(args):
    """Bill completed jobs."""
    if args.job_id:
        record = bill_job(args.job_id)
        if record:
            print(f"  Billed: {record['job_name']} | {record['duration_sec']}s | ${record['cost']}")
        else:
            print("Billing failed.", file=sys.stderr)
            sys.exit(1)
    else:
        bills = bill_all_completed()
        for b in bills:
            print(f"  {b['job_name']} | {b['duration_sec']}s | ${b['cost']}")
        if not bills:
            print("Nothing to bill.")


def cmd_revenue(args):
    """Show total revenue."""
    total = get_total_revenue()
    records = load_billing()
    print(f"Total jobs billed: {len(records)}")
    print(f"Total revenue:     ${total}")


def cmd_failover(args):
    """Run failover: check hosts, requeue orphaned jobs, reassign."""
    requeued, assigned = failover_and_reassign()
    if requeued:
        print(f"Requeued {len(requeued)} jobs:")
        for j in requeued:
            print(f"  {j['job_id']} | {j['name']} | retry {j.get('retries', 0)}")
    else:
        print("No jobs to failover.")

    if assigned:
        print(f"\nReassigned {len(assigned)} jobs:")
        for j, h in assigned:
            print(f"  {j['name']} -> {h['host_id']}")


def cmd_requeue(args):
    """Manually requeue a job."""
    result = requeue_job(args.job_id)
    if result:
        print(f"Job {args.job_id} requeued (retry {result.get('retries', 0)})")
    else:
        print(f"Failed to requeue {args.job_id} (max retries exceeded or not found)", file=sys.stderr)
        sys.exit(1)


def cmd_ssh_keygen(args):
    """Generate SSH keypair."""
    path = generate_ssh_keypair()
    pub = get_public_key(path)
    print(f"Key generated: {path}")
    print(f"Public key:    {path}.pub")
    if pub:
        print(f"\nAdd this to your hosts' ~/.ssh/authorized_keys:")
        print(f"  {pub}")


def cmd_ssh_pubkey(args):
    """Show SSH public key."""
    pub = get_public_key()
    if pub:
        print(pub)
    else:
        print("No SSH key found. Run: xcelsior ssh-keygen", file=sys.stderr)
        sys.exit(1)


def cmd_token_gen(args):
    """Generate a secure API token."""
    token = secrets.token_urlsafe(32)
    print(f"Token: {token}")
    print(f"\nAdd to .env:")
    print(f"  XCELSIOR_API_TOKEN={token}")


def cmd_host_add_ca(args):
    """Register a host with country tag."""
    entry = register_host_ca(args.id, args.ip, args.gpu, args.vram, args.free_vram,
                              args.rate, country=args.country,
                              province=getattr(args, 'province', 'ON'))
    prov = entry.get('province', '')
    print(f"Host registered: {entry['host_id']} | {entry['ip']} | {entry['gpu_model']} | {entry.get('country', 'CA')}{(' ' + prov) if prov else ''} | ${entry['cost_per_hour']}/hr")


def cmd_hosts_ca(args):
    """List Canadian hosts only."""
    hosts = list_hosts_filtered(canada_only=True, active_only=not args.all)
    if not hosts:
        print("No Canadian hosts.")
        return
    for h in hosts:
        print(f"  [{h['status']:>6}] {h['host_id']} | {h['ip']} | {h['gpu_model']} | {h.get('country', '?')} | {h['free_vram_gb']}GB free | ${h['cost_per_hour']}/hr")


def cmd_canada(args):
    """Toggle or check Canada-only mode."""
    if args.on:
        set_canada_only(True)
        print("Canada-only mode: ON")
    elif args.off:
        set_canada_only(False)
        print("Canada-only mode: OFF")
    else:
        from scheduler import CANADA_ONLY
        print(f"Canada-only mode: {'ON' if CANADA_ONLY else 'OFF'}")


def cmd_pool(args):
    """List the autoscale pool."""
    pool = load_autoscale_pool()
    if not pool:
        print("Autoscale pool is empty.")
        return
    for p in pool:
        status = "provisioned" if p.get("provisioned") else "available"
        print(f"  [{status:>12}] {p['host_id']} | {p['ip']} | {p['gpu_model']} | {p['vram_gb']}GB | ${p['cost_per_hour']}/hr | {p.get('country', '?')}")


def cmd_pool_add(args):
    """Add a host to the autoscale pool."""
    entry = add_to_pool(args.host_id, args.ip, args.gpu, args.vram,
                         args.rate, args.country)
    print(f"Added to pool: {entry['host_id']} | {entry['gpu_model']} | {entry['vram_gb']}GB")


def cmd_pool_rm(args):
    """Remove a host from the autoscale pool."""
    remove_from_pool(args.host_id)
    print(f"Removed from pool: {args.host_id}")


def cmd_autoscale(args):
    """Run an autoscale cycle."""
    provisioned, assigned, deprovisioned = autoscale_cycle()
    if provisioned:
        print(f"Provisioned {len(provisioned)} hosts:")
        for h in provisioned:
            print(f"  {h['host_id']} | {h['gpu_model']}")
    else:
        print("No hosts provisioned.")

    if assigned:
        print(f"\nAssigned {len(assigned)} jobs:")
        for j, h in assigned:
            print(f"  {j['name']} -> {h['host_id']}")

    if deprovisioned:
        print(f"\nDeprovisioned {len(deprovisioned)} idle hosts:")
        for hid in deprovisioned:
            print(f"  {hid}")


def cmd_market(args):
    """Browse the marketplace."""
    listings = get_marketplace(active_only=not args.all)
    if not listings:
        print("No listings.")
        return
    for l in listings:
        print(f"  {l['host_id']} | {l['gpu_model']} | {l['vram_gb']}GB | ${l['price_per_hour']}/hr | {l['owner']} | jobs: {l.get('total_jobs', 0)} | earned: ${l.get('total_earned', 0)}")


def cmd_market_list(args):
    """List a rig on the marketplace."""
    listing = list_rig(args.host_id, args.gpu, args.vram, args.price,
                        description=args.desc, owner=args.owner)
    print(f"Listed: {listing['host_id']} | {listing['gpu_model']} | ${listing['price_per_hour']}/hr | owner={listing['owner']}")


def cmd_market_unlist(args):
    """Unlist a rig."""
    if unlist_rig(args.host_id):
        print(f"Unlisted: {args.host_id}")
    else:
        print(f"Listing {args.host_id} not found.", file=sys.stderr)
        sys.exit(1)


def cmd_market_stats(args):
    """Marketplace stats."""
    stats = marketplace_stats()
    print(f"  Active listings:   {stats['active_listings']}")
    print(f"  Total listings:    {stats['total_listings']}")
    print(f"  Jobs completed:    {stats['total_jobs_completed']}")
    print(f"  Host payouts:      ${stats['total_host_payouts']}")
    print(f"  Platform revenue:  ${stats.get('platform_revenue', 0)}")
    print(f"  Platform cut:      {stats['default_platform_cut_pct'] * 100:.0f}%")


def cmd_build(args):
    """Build a Docker image for a model."""
    print(f"Building image for {args.model}...")
    if args.dockerfile_only:
        content = generate_dockerfile(args.model, base_image=args.base, quantize=args.quantize)
        print(content)
        return

    result = build_and_push(args.model, context_dir=args.context,
                             quantize=args.quantize, base_image=args.base,
                             push=args.push)
    if result["built"]:
        print(f"  Built: {result['tag']}")
        if result["pushed"]:
            print(f"  Pushed: {result['remote_tag']}")
        elif args.push:
            print("  Push failed. Check registry config.")
    else:
        print("  Build failed.", file=sys.stderr)
        sys.exit(1)


def cmd_builds(args):
    """List local builds."""
    builds = list_builds()
    if not builds:
        print("No builds.")
        return
    for b in builds:
        df = "Dockerfile" if b["has_dockerfile"] else "no Dockerfile"
        print(f"  {b['model']} | {b['path']} | {df}")


def cmd_tiers(args):
    """List available priority tiers."""
    tiers = list_tiers()
    print("Priority tiers:")
    for name, info in tiers.items():
        print(f"  {name:>10} | priority {info['priority']} | {info['multiplier']}x billing | {info['label']}")


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from api import app
    print(f"Starting Xcelsior API on port {args.port}...")
    uvicorn.run(app, host=args.bind, port=args.port)


def cmd_health_start(args):
    """Start continuous host health monitoring."""
    print(f"Starting health monitor (interval: {args.interval}s)...")
    start_health_monitor(interval=args.interval)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nHealth monitor stopped.")


# ── v2.1 CLI Command Handlers ────────────────────────────────────────

def cmd_reputation(args):
    """Get reputation score and tier for a host or user."""
    from reputation import get_reputation_engine
    re = get_reputation_engine()
    score = re.compute_score(args.entity_id)
    d = score.to_dict()
    print(f"Entity: {args.entity_id}")
    print(f"  Tier:   {d.get('tier', 'unknown')}")
    print(f"  Score:  {d.get('total_score', 0)}")
    print(f"  Trust:  {d.get('reliability_score', 0):.2f}")
    for k, v in d.get("components", {}).items():
        print(f"  {k}: {v}")


def cmd_verify(args):
    """Get verification status for a host."""
    from verification import get_verification_engine
    ve = get_verification_engine()
    status = ve.get_host_status(args.host_id)
    if not status:
        print(f"No verification record for host {args.host_id}")
        return
    print(f"Host: {args.host_id}")
    print(f"  State: {status.get('state', 'unknown')}")
    print(f"  Score: {status.get('overall_score', 0)}")
    for check, result in status.get("checks", {}).items():
        print(f"  {check}: {result}")


def cmd_wallet(args):
    """Get billing wallet balance."""
    from billing import get_billing_engine
    be = get_billing_engine()
    wallet = be.get_wallet(args.customer_id)
    print(f"Customer: {args.customer_id}")
    print(f"  Balance: ${wallet.get('balance_cad', 0):.2f} CAD")
    print(f"  Deposits: ${wallet.get('total_deposits_cad', 0):.2f}")
    print(f"  Spent:    ${wallet.get('total_spent_cad', 0):.2f}")


def cmd_deposit(args):
    """Deposit credits into a customer wallet."""
    from billing import get_billing_engine
    be = get_billing_engine()
    result = be.deposit(args.customer_id, args.amount)
    print(f"Deposited ${args.amount:.2f} CAD into {args.customer_id}")
    print(f"  New balance: ${result.get('balance_cad', 0):.2f} CAD")


def cmd_invoice(args):
    """Generate invoice with tax breakdown."""
    from billing import get_billing_engine
    be = get_billing_engine()
    invoice = be.generate_invoice(args.customer_id)
    print(f"Invoice for {args.customer_id}:")
    print(f"  Subtotal: ${invoice.get('subtotal_cad', 0):.2f}")
    print(f"  Tax:      ${invoice.get('tax_cad', 0):.2f}")
    print(f"  Total:    ${invoice.get('total_cad', 0):.2f}")
    for line in invoice.get("lines", []):
        print(f"    {line.get('description', '')}: ${line.get('amount_cad', 0):.2f}")


def cmd_sla(args):
    """Get SLA uptime and violations for a host."""
    from sla import get_sla_engine
    engine = get_sla_engine()
    uptime = engine.get_host_uptime_pct(args.host_id)
    print(f"Host: {args.host_id}")
    print(f"  30-day Uptime: {uptime:.2f}%")
    if args.month:
        rec = engine.get_host_sla(args.host_id, args.month)
        if rec:
            print(f"  Month {args.month}: {rec.uptime_pct:.2f}% uptime, {rec.incidents} incidents")
            if rec.credit_pct > 0:
                print(f"  Credit: {rec.credit_pct}% (${rec.credit_cad:.2f} CAD)")
    violations = engine.get_violations(args.host_id, since=time.time() - 2592000)
    if violations:
        print(f"  Recent violations: {len(violations)}")
        for v in violations[:5]:
            print(f"    {v.get('violation_type')}: {v.get('severity')} "
                  f"({v.get('metric_value'):.1f} vs {v.get('threshold'):.1f})")


def cmd_provider_register(args):
    """Register a GPU provider with Stripe Connect."""
    from stripe_connect import get_stripe_manager
    mgr = get_stripe_manager()
    result = mgr.create_provider_account(
        provider_id=args.provider_id,
        email=args.email,
        provider_type=args.type,
        corporation_name=args.corp_name,
        business_number=args.bn,
        gst_hst_number=args.gst,
        province=args.province,
    )
    print(f"Provider registered: {args.provider_id}")
    print(f"  Status: {result.get('status')}")
    print(f"  Stripe ID: {result.get('stripe_account_id', 'stub')}")
    url = result.get("onboarding_url", "")
    if url:
        print(f"  Onboarding URL: {url}")


def cmd_provider_info(args):
    """Get provider account details."""
    from stripe_connect import get_stripe_manager
    mgr = get_stripe_manager()
    provider = mgr.get_provider(args.provider_id)
    if not provider:
        print(f"Provider {args.provider_id} not found")
        return
    print(f"Provider: {args.provider_id}")
    for k, v in provider.items():
        if k not in ("stripe_account_id",) and v:
            print(f"  {k}: {v}")


def cmd_leaderboard(args):
    """Show reputation leaderboard."""
    from reputation import get_reputation_engine
    re = get_reputation_engine()
    lb = re.get_leaderboard(entity_type=args.type, limit=args.limit)
    print(f"Reputation Leaderboard (top {args.limit} {args.type}s):")
    for i, entry in enumerate(lb, 1):
        print(f"  {i}. {entry.get('entity_id', '?')} — "
              f"{entry.get('tier', '?')} ({entry.get('total_score', 0)} pts)")


def cmd_compliance(args):
    """Show province compliance matrix and tax rates."""
    from billing import PROVINCE_TAX_RATES
    from jurisdiction import PROVINCE_COMPLIANCE
    print("Province Compliance Matrix:")
    print(f"  {'Province':<6} {'Tax Rate':>10} {'Residency':>12} {'PIA Required':>14}")
    print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*14}")
    for prov, rate in sorted(PROVINCE_TAX_RATES.items()):
        comp = PROVINCE_COMPLIANCE.get(prov, {})
        residency = comp.get("residency_required", False)
        pia = comp.get("pia_required", False)
        print(f"  {prov:<6} {rate*100:>9.1f}% {'Yes' if residency else 'No':>12} "
              f"{'Yes' if pia else 'No':>14}")


# ── OAuth2 Device Flow (CLI-to-Web Auth) ─────────────────────────────

TOKEN_FILE = os.path.expanduser("~/.xcelsior/token.json")


def _save_token(token_data):
    """Save OAuth token to disk."""
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    os.chmod(TOKEN_FILE, 0o600)


def _load_token():
    """Load saved OAuth token from disk."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None


def get_api_token():
    """Get API token from env var or saved OAuth token."""
    env_token = os.environ.get("XCELSIOR_API_TOKEN", "")
    if env_token:
        return env_token
    saved = _load_token()
    if saved:
        return saved.get("access_token", "")
    return ""


def cmd_login(args):
    """Authenticate via OAuth2 device flow (RFC 8628).

    Opens browser to verification URL where user enters the displayed code.
    Polls for authorization and saves the bearer token to ~/.xcelsior/token.json.
    """
    import webbrowser

    api_url = getattr(args, "api_url", None) or os.environ.get("XCELSIOR_API_URL", "http://localhost:8000")

    # Step 1: Request device code
    try:
        import requests as req
        resp = req.post(f"{api_url}/api/auth/device", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error: Could not reach Xcelsior API at {api_url}: {e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    device_code = data["device_code"]
    user_code = data["user_code"]
    verification_uri = data["verification_uri"]
    interval = data.get("interval", 5)
    expires_in = data.get("expires_in", 600)

    # Step 2: Show code and open browser
    print()
    print("  ┌─────────────────────────────────────────┐")
    print(f"  │  Your code:  {user_code:^25s}  │")
    print("  └─────────────────────────────────────────┘")
    print()
    print(f"  Open {verification_uri} and enter the code above.")
    print()

    try:
        webbrowser.open(verification_uri)
        print("  Browser opened. Waiting for authorization...")
    except Exception:
        print("  Could not open browser. Please open the URL manually.")

    # Step 3: Poll for token
    deadline = time.time() + expires_in
    while time.time() < deadline:
        time.sleep(interval)
        try:
            poll = req.post(
                f"{api_url}/api/auth/token",
                json={"device_code": device_code},
                timeout=10,
            )
            if poll.status_code == 200:
                token_data = poll.json()
                _save_token(token_data)
                print(f"\n  ✓ Authenticated! Token saved to {TOKEN_FILE}")
                return
            elif poll.status_code == 428:
                # authorization_pending — keep polling
                sys.stdout.write(".")
                sys.stdout.flush()
                continue
            elif poll.status_code == 410:
                print("\n  ✗ Code expired. Run 'xcelsior login' again.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"\n  ✗ Unexpected response: {poll.status_code}", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"\n  ✗ Poll error: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n  ✗ Authorization timed out. Run 'xcelsior login' again.", file=sys.stderr)
    sys.exit(1)


def cmd_logout(args):
    """Remove saved authentication token."""
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
        print("Logged out. Token removed.")
    else:
        print("Not logged in (no saved token).")


def cmd_whoami(args):
    """Show current authentication status."""
    token = get_api_token()
    if not token:
        print("Not authenticated. Run 'xcelsior login' to authenticate.")
        return

    source = "env" if os.environ.get("XCELSIOR_API_TOKEN") else "oauth"
    masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "****"
    print(f"Authenticated ({source}): {masked}")
    if source == "oauth":
        print(f"Token file: {TOKEN_FILE}")


# ── Slurm Commands ───────────────────────────────────────────────────


def cmd_slurm_submit(args):
    """Submit a job to a Slurm cluster."""
    from slurm_adapter import slurm_submit_cli

    job_dict = {
        "job_id": secrets.token_hex(4),
        "name": args.model,
        "vram_needed_gb": args.vram,
        "priority": args.priority,
        "tier": args.tier or "free",
        "image": getattr(args, "image", ""),
    }
    if args.gpus:
        job_dict["num_gpus"] = args.gpus

    result = slurm_submit_cli(job_dict, profile=args.profile, dry_run=args.dry_run)
    print(result)


def cmd_slurm_status(args):
    """Check Slurm job status."""
    from slurm_adapter import slurm_status_cli
    result = slurm_status_cli(
        xcelsior_job_id=args.job_id,
        slurm_job_id=args.slurm_id,
    )
    print(result)


def cmd_slurm_cancel(args):
    """Cancel a Slurm job."""
    from slurm_adapter import cancel_slurm_job
    result = cancel_slurm_job(args.slurm_id)
    if result.get("cancelled"):
        print(f"Slurm job {args.slurm_id} cancelled.")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        prog="xcelsior",
        description="Xcelsior — distributed GPU scheduler",
    )
    sub = parser.add_subparsers(dest="command")

    # xcelsior run
    p_run = sub.add_parser("run", help="Submit a job")
    p_run.add_argument("--model", required=True, help="Model name")
    p_run.add_argument("--vram", type=float, default=0, help="VRAM needed (GB)")
    p_run.add_argument("--priority", type=int, default=0, help="Job priority")
    p_run.add_argument("--tier", choices=list(PRIORITY_TIERS.keys()), default=None,
                       help="Priority tier (overrides --priority)")
    p_run.add_argument("--gpus", type=int, default=1, help="Number of GPUs needed (default 1)")
    p_run.add_argument("--nfs-server", default=None, help="NFS server for shared storage")
    p_run.add_argument("--nfs-path", default=None, help="NFS export path")
    p_run.add_argument("--image", default=None, help="Docker image to run")
    p_run.add_argument("--no-assign", action="store_true", help="Submit only, don't assign")
    p_run.set_defaults(func=cmd_run)

    # xcelsior jobs
    p_jobs = sub.add_parser("jobs", help="List jobs")
    p_jobs.add_argument("--status", help="Filter by status")
    p_jobs.set_defaults(func=cmd_jobs)

    # xcelsior job <id>
    p_job = sub.add_parser("job", help="Get a specific job")
    p_job.add_argument("job_id", help="Job ID")
    p_job.set_defaults(func=cmd_job)

    # xcelsior cancel <id>
    p_cancel = sub.add_parser("cancel", help="Cancel a job")
    p_cancel.add_argument("job_id", help="Job ID")
    p_cancel.set_defaults(func=cmd_cancel)

    # xcelsior process
    p_proc = sub.add_parser("process", help="Process the job queue")
    p_proc.set_defaults(func=cmd_process)

    CA_PROVINCES = ["AB","BC","MB","NB","NL","NS","NT","NU","ON","PE","QC","SK","YT"]

    # xcelsior host add
    p_hadd = sub.add_parser("host-add", help="Register a host")
    p_hadd.add_argument("--id", required=True, help="Host ID")
    p_hadd.add_argument("--ip", required=True, help="Host IP")
    p_hadd.add_argument("--gpu", required=True, help="GPU model")
    p_hadd.add_argument("--vram", type=float, required=True, help="Total VRAM (GB)")
    p_hadd.add_argument("--free-vram", type=float, default=None, help="Free VRAM (GB), defaults to --vram")
    p_hadd.add_argument("--rate", type=float, default=0.20, help="Cost per hour (default $0.20)")
    p_hadd.add_argument("--country", default="CA", help="Country code (default CA)")
    p_hadd.add_argument("--province", default="", choices=CA_PROVINCES,
                        help="Province/territory code (e.g. ON, BC, QC)")
    p_hadd.set_defaults(func=cmd_host_add)

    # xcelsior host-rm
    p_hrm = sub.add_parser("host-rm", help="Remove a host")
    p_hrm.add_argument("id", help="Host ID")
    p_hrm.set_defaults(func=cmd_host_rm)

    # xcelsior hosts
    p_hosts = sub.add_parser("hosts", help="List hosts")
    p_hosts.add_argument("--all", action="store_true", help="Include dead hosts")
    p_hosts.set_defaults(func=cmd_hosts)

    # xcelsior ping
    p_ping = sub.add_parser("ping", help="Health check all hosts")
    p_ping.set_defaults(func=cmd_ping)

    # xcelsior bill
    p_bill = sub.add_parser("bill", help="Bill completed jobs")
    p_bill.add_argument("job_id", nargs="?", help="Bill a specific job (or all if omitted)")
    p_bill.set_defaults(func=cmd_bill)

    # xcelsior revenue
    p_rev = sub.add_parser("revenue", help="Show total revenue")
    p_rev.set_defaults(func=cmd_revenue)

    # xcelsior failover
    p_fo = sub.add_parser("failover", help="Run failover: requeue jobs from dead hosts")
    p_fo.set_defaults(func=cmd_failover)

    # xcelsior requeue
    p_rq = sub.add_parser("requeue", help="Manually requeue a job")
    p_rq.add_argument("job_id", help="Job ID to requeue")
    p_rq.set_defaults(func=cmd_requeue)

    # xcelsior ssh-keygen
    p_sshkg = sub.add_parser("ssh-keygen", help="Generate SSH keypair for host access")
    p_sshkg.set_defaults(func=cmd_ssh_keygen)

    # xcelsior ssh-pubkey
    p_sshpk = sub.add_parser("ssh-pubkey", help="Show SSH public key")
    p_sshpk.set_defaults(func=cmd_ssh_pubkey)

    # xcelsior token-gen
    p_tgen = sub.add_parser("token-gen", help="Generate a secure API token")
    p_tgen.set_defaults(func=cmd_token_gen)

    # xcelsior host-add-ca
    p_haddca = sub.add_parser("host-add-ca", help="Register a host with country tag")
    p_haddca.add_argument("--id", required=True, help="Host ID")
    p_haddca.add_argument("--ip", required=True, help="Host IP")
    p_haddca.add_argument("--gpu", required=True, help="GPU model")
    p_haddca.add_argument("--vram", type=float, required=True, help="Total VRAM (GB)")
    p_haddca.add_argument("--free-vram", type=float, default=None, help="Free VRAM (GB)")
    p_haddca.add_argument("--rate", type=float, default=0.20, help="Cost per hour")
    p_haddca.add_argument("--country", default="CA", help="Country code (default CA)")
    p_haddca.add_argument("--province", default="ON", choices=CA_PROVINCES,
                          help="Province/territory code (default ON)")
    p_haddca.set_defaults(func=cmd_host_add_ca)

    # xcelsior hosts-ca
    p_hostsca = sub.add_parser("hosts-ca", help="List Canadian hosts only")
    p_hostsca.add_argument("--all", action="store_true", help="Include dead hosts")
    p_hostsca.set_defaults(func=cmd_hosts_ca)

    # xcelsior canada
    p_canada = sub.add_parser("canada", help="Toggle or check Canada-only mode")
    p_canada.add_argument("--on", action="store_true", help="Enable Canada-only")
    p_canada.add_argument("--off", action="store_true", help="Disable Canada-only")
    p_canada.set_defaults(func=cmd_canada)

    # xcelsior pool
    p_pool = sub.add_parser("pool", help="List autoscale pool")
    p_pool.set_defaults(func=cmd_pool)

    # xcelsior pool-add
    p_pooladd = sub.add_parser("pool-add", help="Add host to autoscale pool")
    p_pooladd.add_argument("host_id", help="Host ID")
    p_pooladd.add_argument("--ip", required=True, help="Host IP")
    p_pooladd.add_argument("--gpu", required=True, help="GPU model")
    p_pooladd.add_argument("--vram", type=float, required=True, help="VRAM (GB)")
    p_pooladd.add_argument("--rate", type=float, default=0.20, help="Cost per hour")
    p_pooladd.add_argument("--country", default="CA", help="Country code")
    p_pooladd.set_defaults(func=cmd_pool_add)

    # xcelsior pool-rm
    p_poolrm = sub.add_parser("pool-rm", help="Remove host from autoscale pool")
    p_poolrm.add_argument("host_id", help="Host ID")
    p_poolrm.set_defaults(func=cmd_pool_rm)

    # xcelsior autoscale
    p_autoscale = sub.add_parser("autoscale", help="Run an autoscale cycle")
    p_autoscale.set_defaults(func=cmd_autoscale)

    # xcelsior market
    p_market = sub.add_parser("market", help="Browse marketplace listings")
    p_market.add_argument("--all", action="store_true", help="Include inactive listings")
    p_market.set_defaults(func=cmd_market)

    # xcelsior market-list
    p_mlist = sub.add_parser("market-list", help="List a rig on the marketplace")
    p_mlist.add_argument("host_id", help="Host ID")
    p_mlist.add_argument("--gpu", required=True, help="GPU model")
    p_mlist.add_argument("--vram", type=float, required=True, help="VRAM (GB)")
    p_mlist.add_argument("--price", type=float, required=True, help="Price per hour")
    p_mlist.add_argument("--desc", default="", help="Description")
    p_mlist.add_argument("--owner", default="anonymous", help="Owner name")
    p_mlist.set_defaults(func=cmd_market_list)

    # xcelsior market-unlist
    p_munlist = sub.add_parser("market-unlist", help="Unlist a rig from the marketplace")
    p_munlist.add_argument("host_id", help="Host ID to unlist")
    p_munlist.set_defaults(func=cmd_market_unlist)

    # xcelsior market-stats
    p_mstats = sub.add_parser("market-stats", help="Marketplace aggregate stats")
    p_mstats.set_defaults(func=cmd_market_stats)

    # xcelsior build
    p_build = sub.add_parser("build", help="Build a Docker image for a model")
    p_build.add_argument("model", help="Model name")
    p_build.add_argument("--base", default="python:3.11-slim", help="Base Docker image")
    p_build.add_argument("--quantize", choices=["gguf", "gptq", "awq"], default=None,
                         help="Quantization method")
    p_build.add_argument("--context", default=None, help="Build context directory")
    p_build.add_argument("--push", action="store_true", help="Push to registry after build")
    p_build.add_argument("--dockerfile-only", action="store_true", help="Print Dockerfile, don't build")
    p_build.set_defaults(func=cmd_build)

    # xcelsior builds
    p_builds = sub.add_parser("builds", help="List local builds")
    p_builds.set_defaults(func=cmd_builds)

    # xcelsior tiers
    p_tiers = sub.add_parser("tiers", help="List priority tiers and billing multipliers")
    p_tiers.set_defaults(func=cmd_tiers)

    # xcelsior serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    p_serve.add_argument("--bind", default="0.0.0.0", help="Bind address")
    p_serve.set_defaults(func=cmd_serve)

    # xcelsior health-start
    p_health = sub.add_parser("health-start", help="Start host health monitor loop")
    p_health.add_argument("--interval", type=int, default=5, help="Check interval in seconds")
    p_health.set_defaults(func=cmd_health_start)

    # ── v2.1 CLI Commands ─────────────────────────────────────────────

    # xcelsior reputation <entity_id>
    p_rep = sub.add_parser("reputation", help="Get reputation score and tier for a host/user")
    p_rep.add_argument("entity_id", help="Host or user ID")
    p_rep.set_defaults(func=cmd_reputation)

    # xcelsior verify <host_id>
    p_verify = sub.add_parser("verify", help="Get verification status for a host")
    p_verify.add_argument("host_id", help="Host ID to check verification status")
    p_verify.set_defaults(func=cmd_verify)

    # xcelsior wallet <customer_id>
    p_wallet = sub.add_parser("wallet", help="Get billing wallet balance for a customer")
    p_wallet.add_argument("customer_id", help="Customer ID")
    p_wallet.set_defaults(func=cmd_wallet)

    # xcelsior deposit <customer_id> <amount>
    p_deposit = sub.add_parser("deposit", help="Deposit CAD credits into a customer wallet")
    p_deposit.add_argument("customer_id", help="Customer ID")
    p_deposit.add_argument("amount", type=float, help="Amount in CAD")
    p_deposit.set_defaults(func=cmd_deposit)

    # xcelsior invoice <customer_id>
    p_invoice = sub.add_parser("invoice", help="Generate invoice with tax breakdown")
    p_invoice.add_argument("customer_id", help="Customer ID")
    p_invoice.set_defaults(func=cmd_invoice)

    # xcelsior sla <host_id>
    p_sla = sub.add_parser("sla", help="Get SLA uptime and violations for a host")
    p_sla.add_argument("host_id", help="Host ID")
    p_sla.add_argument("--month", default="", help="Month in YYYY-MM format")
    p_sla.set_defaults(func=cmd_sla)

    # xcelsior provider-register
    p_preg = sub.add_parser("provider-register", help="Register a GPU provider (Stripe Connect)")
    p_preg.add_argument("provider_id", help="Unique provider ID")
    p_preg.add_argument("--email", required=True, help="Provider email")
    p_preg.add_argument("--type", default="individual", choices=["individual", "company"],
                        help="Provider type")
    p_preg.add_argument("--corp-name", default="", help="Corporation name (for company type)")
    p_preg.add_argument("--bn", default="", help="CRA Business Number")
    p_preg.add_argument("--gst", default="", help="GST/HST registration number")
    p_preg.add_argument("--province", default="ON", choices=CA_PROVINCES,
                        help="Province/territory code (default ON)")
    p_preg.set_defaults(func=cmd_provider_register)

    # xcelsior provider <provider_id>
    p_prov = sub.add_parser("provider", help="Get provider account details")
    p_prov.add_argument("provider_id", help="Provider ID")
    p_prov.set_defaults(func=cmd_provider_info)

    # xcelsior leaderboard
    p_lb = sub.add_parser("leaderboard", help="Show reputation leaderboard")
    p_lb.add_argument("--type", default="host", help="Entity type (host/user)")
    p_lb.add_argument("--limit", type=int, default=10, help="Number of entries")
    p_lb.set_defaults(func=cmd_leaderboard)

    # xcelsior compliance
    p_comp = sub.add_parser("compliance", help="Show province compliance matrix and tax rates")
    p_comp.set_defaults(func=cmd_compliance)

    # ── v2.2 CLI Commands ─────────────────────────────────────────────

    # xcelsior login
    p_login = sub.add_parser("login", help="Authenticate via browser (OAuth2 device flow)")
    p_login.add_argument("--api-url", dest="api_url", default=None,
                         help="API base URL (default: XCELSIOR_API_URL or localhost:8000)")
    p_login.set_defaults(func=cmd_login)

    # xcelsior logout
    p_logout = sub.add_parser("logout", help="Remove saved authentication token")
    p_logout.set_defaults(func=cmd_logout)

    # xcelsior whoami
    p_whoami = sub.add_parser("whoami", help="Show current authentication status")
    p_whoami.set_defaults(func=cmd_whoami)

    # xcelsior slurm-submit
    p_slurm_sub = sub.add_parser("slurm-submit", help="Submit a job to Slurm cluster")
    p_slurm_sub.add_argument("--model", required=True, help="Model name")
    p_slurm_sub.add_argument("--vram", type=float, default=0, help="VRAM needed (GB)")
    p_slurm_sub.add_argument("--priority", type=int, default=0, help="Priority")
    p_slurm_sub.add_argument("--tier", default=None, help="Priority tier")
    p_slurm_sub.add_argument("--profile", default=None,
                             choices=["nibi", "graham", "narval", "generic"],
                             help="Cluster profile")
    p_slurm_sub.add_argument("--image", default="", help="Docker/Apptainer image")
    p_slurm_sub.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    p_slurm_sub.add_argument("--dry-run", action="store_true", help="Print script without submitting")
    p_slurm_sub.set_defaults(func=cmd_slurm_submit)

    # xcelsior slurm-status
    p_slurm_st = sub.add_parser("slurm-status", help="Check Slurm job status")
    p_slurm_st.add_argument("--job-id", default=None, help="Xcelsior job ID")
    p_slurm_st.add_argument("--slurm-id", default=None, help="Slurm job ID")
    p_slurm_st.set_defaults(func=cmd_slurm_status)

    # xcelsior slurm-cancel
    p_slurm_cancel = sub.add_parser("slurm-cancel", help="Cancel a Slurm job")
    p_slurm_cancel.add_argument("slurm_id", help="Slurm job ID")
    p_slurm_cancel.set_defaults(func=cmd_slurm_cancel)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Default free_vram to total vram for host-add commands
    if args.command in ("host-add", "host-add-ca") and args.free_vram is None:
        args.free_vram = args.vram

    args.func(args)


if __name__ == "__main__":
    main()
