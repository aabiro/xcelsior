"""The reverification sweep — ask overdue hosts to prove themselves again.

`VerificationStore.list_hosts_needing_reverification()` implements the due
query. Its only wrapper, `VerificationEngine.get_hosts_needing_reverification()`,
**had no callers** — no worker, no scheduler pass, no timer. `next_check_at` was
written and never read, so nothing ever moved a host out of `verified` and no
host was ever asked to prove itself again. Against a one-day interval, a stamp
could therefore age indefinitely — the drift is measured in months, not hours.

That is why this ships before the preference surface. `require_verified` is a
control over a fact that nothing maintains; shipping the control first would
teach users the feature is broken, and they would be right.

## Why the server cannot simply re-verify

Verification is **agent-push**: `run_verification(host_id, report)` needs a
fresh telemetry report — GPU model, VRAM, driver, PCIe bandwidth, temperature,
network loss and jitter — that only the host can produce. The agent submits one
at startup and **never again**, which is the whole explanation for a months-old
stamp: that agent simply has not restarted.

So the sweep asks. It enqueues a `reverify` command on the host's existing
command channel; the agent re-runs the benchmark and POSTs to `/agent/verify`,
which is the same path startup uses. Nothing about the verification logic
changes — what changes is that something now causes it to happen more than once.

## What this deliberately does not do

It does not expire stamps. Moving a host out of `verified` because a check is
overdue would change `scheduler.allocate_best_host`'s "prefer verified hosts"
set, and therefore where every job in the marketplace lands. That is the
reconciliation §5.4 names as a C2 decision, and it is not made in passing here.
The placement gate already treats an overdue stamp as `stale` on read, so a
`require_verified` request is answered correctly today either way.

## The sweep cannot fire in the failure it detects

**Recovery machinery must not live on the failing side of the boundary it
recovers, and this does.** `run_verification` needs a telemetry report only the
host can produce, so refreshing a stamp requires the host to be reachable — and
a stamp goes stale precisely when it is not. The mechanism for fixing the
condition is unavailable exactly when the condition holds.

That is not hypothetical here. When this shipped, production's stamps were 112
and 125 days old and every worker was locked out by a retired public agent
ingress: the sweep ran correctly, enqueued correctly, deduped correctly, and
produced **zero fresh stamps**, because nothing could drain the commands. Being
C2's first commit did not help, and could not have.

So the honest statement of what this buys: it removes "nothing ever asks" as a
cause. It does not remove "the host never answers", and no amount of sweeping
will, because the answer has to come from the thing that is down.

**Reading the result — the discriminator is correlation, not severity.**
Independent host failures do not synchronise; a shared-dependency failure does.
So *N* hosts reading identically at once is positive evidence of a common cause,
and a common cause sits upstream of the hosts:

* **all stale → look up the stack.** Ingress, DNS, credentials, the gateway.
* **one stale → look at the host.**

That makes it a test rather than a hunch, and it is the inference that was
available and unused: four hosts at 112 and 125 days is *one* fact about the
path to them, not four facts about them. Read as four, it looks like fleet
neglect and invites verifying hosts one at a time — none of which could have
worked, because the answer had to travel back over the link that was down.

Distinct from the scope-mismatch defect this phase kept finding (a mechanism not
covering the case it was cited for). This mechanism covers its case exactly; the
problem is *where it runs relative to what it protects against*. The two
questions predict different searches — "what does this cover?" versus "where
does this run?" — and the second one is the less asked.

## Rollout ordering

Agents hard-refuse unknown commands by design, so a fleet that has not taken the
`reverify` handler yet will refuse these — loudly, once per host per sweep, and
self-correcting the moment it upgrades. Deploy, push `upgrade_agent`, then watch
`stale` clear. The alternative — a flag defaulting off — is a way to ship
something that never runs.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

#: The command the agent dispatches on. Must appear in **both** allowlists —
#: `routes/agent.py::_AGENT_COMMAND_ALLOWED` and
#: `worker_agent.py::_AGENT_COMMAND_ALLOWED` — or the enqueue is rejected at the
#: API boundary and the dispatch is refused at the host.
REVERIFY_COMMAND = "reverify"

#: How often the sweep runs, and how long a queued request stays live. Equal on
#: purpose: at most one live request per host, so a host that is offline for a
#: week accumulates one expired row per sweep rather than a backlog it drains
#: all at once when it returns.
SWEEP_INTERVAL_SEC = int(os.environ.get("XCELSIOR_REVERIFY_SWEEP_INTERVAL_SEC", "3600"))

#: A ceiling on one pass. The first sweep after this ships sees every overdue
#: host at once, and asking a hundred hosts to run a 60-second GPU benchmark in
#: the same minute is a self-inflicted load spike. Overdue hosts that miss the
#: cut are still overdue next hour.
SWEEP_MAX_HOSTS = int(os.environ.get("XCELSIOR_REVERIFY_SWEEP_MAX_HOSTS", "25"))


def sweep_enabled() -> bool:
    return os.environ.get("XCELSIOR_REVERIFY_SWEEP_ENABLED", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def hosts_with_pending_reverify(conn, host_ids: list[str]) -> set[str]:
    """Hosts that already hold a live, undelivered `reverify` request.

    Asking twice is not harmful — the agent would run the benchmark twice — but
    it is a real cost on a real GPU, and a sweep that re-asks every hour while a
    host is offline turns one overdue host into a queue.
    """
    if not host_ids:
        return set()
    rows = conn.execute(
        """SELECT DISTINCT host_id FROM agent_commands
            WHERE host_id = ANY(%s)
              AND command = %s
              AND status = 'pending'
              AND expires_at > EXTRACT(EPOCH FROM NOW())""",
        (host_ids, REVERIFY_COMMAND),
    ).fetchall()
    return {str(r[0]) for r in rows}


def run_sweep(*, now: float | None = None, limit: int | None = None) -> dict:
    """One pass: find the overdue hosts and ask each to prove itself again.

    Returns a summary rather than logging only, so the bg-worker task and the
    tests read the same numbers.
    """
    summary = {
        "enabled": sweep_enabled(),
        "due": 0,
        "asked": 0,
        "already_pending": 0,
        "failed": 0,
        "skipped_over_limit": 0,
    }
    if not summary["enabled"]:
        return summary

    from verification import get_verification_engine

    engine = get_verification_engine()
    due = list(engine.get_hosts_needing_reverification())
    summary["due"] = len(due)
    if not due:
        return summary

    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        pending = hosts_with_pending_reverify(conn, due)

    candidates = [h for h in due if h not in pending]
    summary["already_pending"] = len(due) - len(candidates)

    cap = SWEEP_MAX_HOSTS if limit is None else int(limit)
    if cap >= 0 and len(candidates) > cap:
        summary["skipped_over_limit"] = len(candidates) - cap
        candidates = candidates[:cap]

    from routes.agent import enqueue_agent_command

    for host_id in candidates:
        try:
            enqueue_agent_command(
                host_id,
                REVERIFY_COMMAND,
                {"reason": "scheduled_reverification", "requested_at": now or time.time()},
                created_by="verification_sweep",
                ttl_sec=SWEEP_INTERVAL_SEC,
            )
            summary["asked"] += 1
        except Exception as exc:
            # One unreachable or queue-full host must not stop the rest. The
            # count is returned rather than only logged, because "the sweep ran"
            # and "the sweep did anything" are different facts and the second is
            # the one that matters.
            summary["failed"] += 1
            log.warning("reverification sweep: could not ask host=%s: %s", host_id, exc)

    log.info(
        "reverification sweep: due=%d asked=%d pending=%d failed=%d skipped=%d",
        summary["due"],
        summary["asked"],
        summary["already_pending"],
        summary["failed"],
        summary["skipped_over_limit"],
    )
    return summary
