#!/usr/bin/env python3
"""Real-stack fixture driver for the MCP §26.4 SDK test.

This is intentionally test-only orchestration: it seeds a schedulable host,
runs the real scheduler transaction once, and returns the fenced authority
tuple a deterministic fake worker must present to the public agent-v2 API.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService
from db import _get_pg_pool


def place(job_id: str, gpu_model: str) -> None:
    marker = uuid.uuid4().hex[:10]
    host_id = f"mcp-e2e-host-{marker}"
    host = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "gpu_count": 2,
        "free_vram_gb": 48.0,
        "total_vram_gb": 48.0,
        "cost_per_hour": 1.0,
        "admitted": True,
        "last_seen": time.time(),
    }
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO hosts (host_id, status, registered_at, payload)
            VALUES (%s, 'active', %s, %s)
            """,
            (host_id, time.time(), json.dumps(host)),
        )
        # The launch job already exists. Pin its requested model to the isolated
        # fixture host so no concurrent CI fleet row can steal this placement.
        conn.execute(
            """
            UPDATE jobs
               SET payload = jsonb_set(payload, '{gpu_model}', to_jsonb(%s::text), true),
                   effective_priority = 9223372036854775806,
                   fair_share_finish = -1,
                   queued_at = clock_timestamp() - interval '1 day',
                   schedule_claim_owner = NULL,
                   schedule_claim_token = NULL,
                   schedule_claim_expires_at = NULL,
                   next_schedule_at = NULL
             WHERE job_id = %s
            """,
            (gpu_model, job_id),
        )
        conn.commit()
    config = SchedulerConfig(
        mode=SchedulerMode.CANARY,
        replica_id=f"mcp-e2e-{marker}",
        canary_gpu_models=frozenset({gpu_model.lower()}),
        canary_host_ids=frozenset({host_id}),
        lease_claim_ttl_sec=600,
        tick_max_placements=1,
    )
    report = SchedulerService(config).tick()
    if not report.placements:
        raise RuntimeError(f"scheduler did not place {job_id}: {report}")
    reservation = report.placements[0]
    with pool.connection() as conn:
        command = conn.execute(
            """
            SELECT args, fencing_token
              FROM agent_commands
             WHERE command_id = %s
            """,
            (reservation.command_id,),
        ).fetchone()
        host_version = conn.execute(
            "SELECT version FROM hosts WHERE host_id=%s", (host_id,)
        ).fetchone()[0]
    args = dict(command[0])
    print(
        json.dumps(
            {
                "host_id": host_id,
                "host_version": int(host_version),
                "job_id": job_id,
                "attempt_id": reservation.attempt_id,
                "lease_id": reservation.lease_id,
                "command_id": reservation.command_id,
                "fencing_token": int(command[1]),
                "worker_session_id": f"mcp-e2e-worker-{marker}",
                "lease_args": args,
            }
        )
    )


def expire(plan_id: str) -> None:
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE action_plans SET expires_at=clock_timestamp()-interval '1 second' "
            "WHERE plan_id=%s",
            (plan_id,),
        )
        conn.commit()
    print(json.dumps({"ok": True, "plan_id": plan_id}))


def host_version(host_id: str) -> None:
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT version FROM hosts WHERE host_id=%s", (host_id,)
        ).fetchone()
    if not row:
        raise RuntimeError(f"host not found: {host_id}")
    print(json.dumps({"host_id": host_id, "version": int(row[0])}))


def trace_chain(job_id: str) -> None:
    """Prove one W3C trace crosses MCP, API, outbox, attempt, and command."""
    with _get_pg_pool().connection() as conn:
        plan = conn.execute(
            """
            SELECT plan_id, trace_id
              FROM action_plans
             WHERE job_id=%s
            """,
            (job_id,),
        ).fetchone()
        if not plan:
            raise RuntimeError(f"action plan not found for job: {job_id}")
        plan_id, plan_trace = str(plan[0]), str(plan[1] or "")
        attempt = conn.execute(
            """
            SELECT trace_id FROM job_attempts
             WHERE job_id=%s ORDER BY attempt_number DESC LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        command = conn.execute(
            """
            SELECT trace_id FROM agent_commands
             WHERE job_id=%s ORDER BY created_at DESC LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        audit = conn.execute(
            """
            SELECT audit_id, trace_id FROM mcp_tool_audit
             WHERE action_plan_id=%s AND tool_name='create_instance'
               AND trace_id=%s
             ORDER BY occurred_at ASC LIMIT 1
            """,
            (plan_id, plan_trace),
        ).fetchone()
        outbox = (
            conn.execute(
                """
                SELECT headers->>'trace_id' FROM outbox_events
                 WHERE aggregate_type='mcp_tool_audit'
                   AND aggregate_id=%s
                 ORDER BY created_at DESC LIMIT 1
                """,
                (str(audit[0]),),
            ).fetchone()
            if audit
            else None
        )
    traces = {
        "plan": plan_trace,
        "attempt": str(attempt[0] or "") if attempt else "",
        "command": str(command[0] or "") if command else "",
        "audit": str(audit[1] or "") if audit else "",
        "outbox": str(outbox[0] or "") if outbox else "",
    }
    print(
        json.dumps(
            {
                "ok": bool(plan_trace)
                and len(plan_trace) == 32
                and all(value == plan_trace for value in traces.values()),
                "plan_id": plan_id,
                "traces": traces,
            }
        )
    )


def spend_counter_check() -> None:
    """Exercise the real Redis Lua reservation, replay, denial, and release."""
    from control_plane.launch import spend_counters

    suffix = uuid.uuid4().hex
    reservation = spend_counters.reserve(
        plan_id=f"plan-{suffix}",
        client_id=f"client-{suffix}",
        tenant_id=f"tenant-{suffix}",
        amount_micros=600,
        hourly_limit_micros=1_000,
        daily_limit_micros=1_000,
    )
    replay = spend_counters.reserve(
        plan_id=f"plan-{suffix}",
        client_id=f"client-{suffix}",
        tenant_id=f"tenant-{suffix}",
        amount_micros=600,
        hourly_limit_micros=1_000,
        daily_limit_micros=1_000,
    )
    denied = False
    try:
        spend_counters.reserve(
            plan_id=f"other-{suffix}",
            client_id=f"client-{suffix}",
            tenant_id=f"tenant-{suffix}",
            amount_micros=600,
            hourly_limit_micros=1_000,
            daily_limit_micros=1_000,
        )
    except spend_counters.SpendLimitExceeded:
        denied = True
    spend_counters.release(reservation)
    print(
        json.dumps(
            {
                "ok": bool(
                    reservation
                    and reservation.backend == "redis"
                    and replay
                    and replay.replay
                    and denied
                )
            }
        )
    )


def cleanup(prefix: str) -> None:
    """Best-effort cleanup scoped exclusively to this test's identifiers."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        job_ids = [
            row[0]
            for row in conn.execute(
                "SELECT job_id FROM jobs WHERE payload->>'name' LIKE %s",
                (f"{prefix}%",),
            ).fetchall()
        ]
        for job_id in job_ids:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id LIKE 'mcp-e2e-host-%'")
        conn.commit()
    print(json.dumps({"ok": True, "jobs": len(job_ids)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    place_parser = sub.add_parser("place")
    place_parser.add_argument("job_id")
    place_parser.add_argument("gpu_model")
    expire_parser = sub.add_parser("expire")
    expire_parser.add_argument("plan_id")
    host_version_parser = sub.add_parser("host-version")
    host_version_parser.add_argument("host_id")
    trace_parser = sub.add_parser("trace-chain")
    trace_parser.add_argument("job_id")
    sub.add_parser("spend-counter-check")
    cleanup_parser = sub.add_parser("cleanup")
    cleanup_parser.add_argument("prefix")
    args = parser.parse_args()
    if args.command == "place":
        place(args.job_id, args.gpu_model)
    elif args.command == "expire":
        expire(args.plan_id)
    elif args.command == "host-version":
        host_version(args.host_id)
    elif args.command == "trace-chain":
        trace_chain(args.job_id)
    elif args.command == "spend-counter-check":
        spend_counter_check()
    else:
        cleanup(args.prefix)


if __name__ == "__main__":
    main()
