"""Applying a stated preference to real candidates, and writing down the answer.
 `preference.choose_host` is pure and `host_projection` reads the evidence;
this is the one place they meet a live fleet, so there is one place a preference
is evaluated and one place a decision is recorded.

## Why this hangs off the hard filter rather than the legacy scheduler

`choose_host` documents that it expects hosts *"already past the hard filters —
this decides among hosts that could run the job, not whether any can."*
`launch.service.simulate_placement` already produces exactly that: a consistent
snapshot through `filter_hosts`. Wiring the preference there means the trade-off
is rendered by `preview()` **before the user commits**, which is what C2 asks
for, and it leaves `scheduler.allocate_best_host` — the path every unconstrained
job flows through — untouched.

That is the §5.4 reconciliation, and the shape of it is: **the fallback is
skipped only when the request is constrained.** An unconstrained launch keeps
the cold-start behaviour it has today, including preferring verified hosts and
falling back to all of them. A constrained one is decided here, where a
constraint that cannot be met is a refusal rather than a quieter placement.

## Recording is best-effort, and deliberately so

An audit write must never be the thing that fails a placement. The row is
written in its own transaction after the decision is made: a decision that
happened and was not recorded is a gap in the trail, while a placement refused
because its *audit row* would not insert is an outage caused by bookkeeping.
The gap is logged loudly enough to find.
"""

from __future__ import annotations

import logging
from typing import Sequence

from control_plane.scheduler.host_projection import attach_placement_evidence
from control_plane.scheduler.preference import (
    PlacementChoice,
    PlacementPreference,
    PlacementRefused,
    choose_host,
)

log = logging.getLogger(__name__)


def evaluate_preference(
    conn,
    hosts: Sequence[dict],
    preference: PlacementPreference,
    *,
    now: float | None = None,
) -> tuple[PlacementChoice | PlacementRefused, list[dict]]:
    """Decide among `hosts`, and return the candidates the decision was made over.

    The candidates come back with the decision because a refusal is only
    interpretable against the field it refused over — the record needs both, and
    reconstructing the field afterwards would read a fleet that has since moved.
    """
    if not hosts:
        return (
            PlacementRefused(
                code="no_eligible_hosts",
                detail="no host satisfies the job's requirements",
            ),
            [],
        )
    candidates = attach_placement_evidence(conn, list(hosts), now=now)
    return choose_host(candidates, preference), candidates


def record_decision(
    decision: PlacementChoice | PlacementRefused,
    candidates: Sequence[dict],
    preference: PlacementPreference | None,
    *,
    tenant_id: str,
    job_id: str | None = None,
) -> str | None:
    """Append the decision. Returns its id, or None when it could not be written.

    Best-effort by design — see the module docstring. Every failure path logs;
    none raises.
    """
    if not str(tenant_id or "").strip():
        # Not a silent skip: an unattributed decision is one no tenant-scoped
        # read will ever return, and the caller should see that it asked for a
        # record it will not be able to find.
        log.warning("placement decision not recorded: no tenant to attribute it to")
        return None

    try:
        from control_plane.db import control_plane_transaction
        from control_plane.scheduler.placement_record import record_placement

        with control_plane_transaction() as conn:
            return record_placement(
                conn,
                tenant_id=tenant_id,
                decision=decision,
                candidates=candidates,
                preference=preference,
                job_id=job_id,
            )
    except Exception:
        log.exception(
            "placement decision could not be recorded (tenant=%s job=%s outcome=%s)",
            tenant_id,
            job_id,
            "refused" if isinstance(decision, PlacementRefused) else "placed",
        )
        return None
