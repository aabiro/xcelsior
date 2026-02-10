#!/usr/bin/env python3
# Xcelsior CLI — Phase 10
# argparse. No npm. No node. Just Python.

import argparse
import json
import secrets
import sys

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
    job = submit_job(args.model, args.vram, args.priority, tier=args.tier)
    print(f"Job submitted: {job['job_id']} | {job['name']} | {job['vram_needed_gb']}GB | tier={job['tier']} (priority {job['priority']})")

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
    entry = register_host(args.id, args.ip, args.gpu, args.vram, args.free_vram, args.rate)
    print(f"Host registered: {entry['host_id']} | {entry['ip']} | {entry['gpu_model']} | {entry['total_vram_gb']}GB | ${entry['cost_per_hour']}/hr")


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
                              args.rate, country=args.country)
    print(f"Host registered: {entry['host_id']} | {entry['ip']} | {entry['gpu_model']} | {entry.get('country', '?')} | ${entry['cost_per_hour']}/hr")


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
    print(f"  Platform cut:      {stats['platform_cut_pct'] * 100:.0f}%")


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

    # xcelsior host add
    p_hadd = sub.add_parser("host-add", help="Register a host")
    p_hadd.add_argument("--id", required=True, help="Host ID")
    p_hadd.add_argument("--ip", required=True, help="Host IP")
    p_hadd.add_argument("--gpu", required=True, help="GPU model")
    p_hadd.add_argument("--vram", type=float, required=True, help="Total VRAM (GB)")
    p_hadd.add_argument("--free-vram", type=float, default=None, help="Free VRAM (GB), defaults to --vram")
    p_hadd.add_argument("--rate", type=float, default=0.20, help="Cost per hour (default $0.20)")
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
