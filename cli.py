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
)


def cmd_run(args):
    """Submit a job and optionally process the queue."""
    job = submit_job(args.model, args.vram, args.priority)
    print(f"Job submitted: {job['job_id']} | {job['name']} | {job['vram_needed_gb']}GB | priority {job['priority']}")

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
        print(f"  [{j['status']:>9}] {j['job_id']} | {j['name']} | {j['vram_needed_gb']}GB | host: {host}")


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

    # xcelsior serve
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    p_serve.add_argument("--bind", default="0.0.0.0", help="Bind address")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Default free_vram to total vram for host-add
    if args.command == "host-add" and args.free_vram is None:
        args.free_vram = args.vram

    args.func(args)


if __name__ == "__main__":
    main()
