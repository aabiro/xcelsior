"""A migration target must pass the gate a launch passes. C3.

The rule, from `docs/placement-preference-plan.md` §5.3: *"migrated to cheaper"
must never become a path onto a host that would have failed the gate at launch.*
A migration that bypasses admission is a way to reach an unvetted host without
ever asking for one — cheaper capacity is exactly the incentive that would make
someone want to.

## It re-runs the launch gate, it does not re-implement it

`filter_hosts` is the Stage-C filter `launch.service.simulate_placement` uses,
and `evaluate_preference` is the same evaluation the constrained placement path
uses. Both are called here rather than reproduced. A second admission gate would
drift from the first silently, which is the defect the shared ranker was
extracted to prevent — two policies behind one name.

That makes this check *real* rather than decorative: `administrative_state` is
recomputed from `admission_state` by a database trigger, and a host can be
drained, disabled, or go stale between the launch and the migration. The answer
genuinely differs from the one launch got.

## This module has no production caller yet, and that is stated on purpose

The migration *executor* — snapshot, stop, relaunch, verify — is not written.
C3 wires it. Until then this is a tested library waiting for its caller, not a
wired path.

**No cause is named here on purpose.** An earlier draft said "blocked on two
live instances that can share a volume", which was wrong: the fleet is dark
because production's agent-ingress cutover is half finished, not for want of
hardware. A docstring is where a cause stops being questioned, so an unverified
one belongs in it least of all — better to say what is true (nothing calls this
yet) and let whoever wires it establish why it had not been.

That is a different thing from the defect this phase kept finding — code that
silently never runs while appearing wired — but the difference only holds if the
module says so. `list_hosts_needing_reverification` was also "correct, waiting"
for months, and nobody knew, and production's verified stamps aged four months
as a result.

## What it deliberately does not check

**Attestation freshness.** `host_attestation.validate_attestation` is a shape
check over stored data that does not change, and `attested_at` has no readers
anywhere — so attestation carries the same staleness hole verification did.
Adding a freshness gate *here only* would reject a target that a launch would
accept, which is the wrong asymmetry: the requirement is that migration be no
weaker than launch, not stronger. The hole belongs to admission and is recorded
as owed in §5.3a rather than patched at the migration boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from control_plane.scheduler.constrained_placement import evaluate_preference
from control_plane.scheduler.preference import (
    PlacementChoice,
    PlacementPreference,
    PlacementRefused,
)

log = logging.getLogger(__name__)


def _job_requirements(job: dict) -> dict:
    """The shape `filter_hosts` reads, taken from the job being migrated.

    A migration inherits the original job's requirements — migrating is not an
    opportunity to quietly relax them onto a smaller card.
    """
    return {
        "gpu_model": job.get("gpu_model"),
        "num_gpus": job.get("num_gpus"),
        "vram_needed_gb": job.get("vram_needed_gb"),
        "region": job.get("region"),
    }


def evaluate_migration_target(
    conn,
    job: dict,
    target_host_id: str,
    *,
    preference: PlacementPreference | None = None,
    now: float | None = None,
) -> PlacementChoice | PlacementRefused:
    """Would this job be allowed to launch on this host right now?

    Returns the same typed outcome the launch path returns, so a caller renders
    one shape of refusal whether the job is starting or moving.

    **`conn` must be a transaction on which nothing has run yet.**
    `take_snapshot` issues `SET TRANSACTION ISOLATION LEVEL REPEATABLE READ`,
    which PostgreSQL rejects once any statement has executed. That is an
    implicit contract the type signature cannot carry, so it is stated here:
    do setup in its own transaction and hand this one a fresh connection.
    """
    from control_plane.scheduler.filters import FilterContext, filter_hosts
    from control_plane.scheduler.snapshot import take_snapshot

    target_host_id = str(target_host_id or "").strip()
    if not target_host_id:
        return PlacementRefused(
            code="no_migration_target",
            detail="a migration needs a named target host",
        )

    snapshot = take_snapshot(conn)
    targets = [h for h in snapshot.hosts if str(h.get("host_id")) == target_host_id]
    if not targets:
        return PlacementRefused(
            code="unknown_migration_target",
            detail=f"host {target_host_id!r} is not in the fleet",
            asked=target_host_id,
        )

    eligible, rejections = filter_hosts(
        _job_requirements(job),
        targets,
        FilterContext(stale_host_ids=snapshot.stale_host_ids),
    )
    if not eligible:
        reasons = _reason_codes(rejections, target_host_id)
        return PlacementRefused(
            code="target_not_admissible",
            detail=(
                f"host {target_host_id!r} would not pass admission for this job "
                f"at launch: {', '.join(reasons) or 'no reason recorded'}. "
                "Migration is not a way onto a host the gate would refuse."
            ),
            asked=target_host_id,
            best_available=reasons[0] if reasons else None,
        )

    if preference is None or preference.is_empty():
        # No stated preference: passing the same hard filter a launch passes is
        # the whole requirement, and `choose_host` over one host would only
        # restate it.
        decision, _candidates = evaluate_preference(
            conn, eligible, PlacementPreference(), now=now
        )
        return decision

    # The preference is re-evaluated on the target because a constraint the
    # original placement satisfied may not hold here — that is the entire reason
    # to re-check rather than trust the earlier decision.
    decision, _candidates = evaluate_preference(conn, eligible, preference, now=now)
    return decision


def _reason_codes(rejections: Any, host_id: str) -> list[str]:
    """Filter reasons for one host, as codes a caller can branch on."""
    entry = None
    if isinstance(rejections, dict):
        entry = rejections.get(host_id)
    if not entry:
        return []
    codes = []
    for reason in entry if isinstance(entry, (list, tuple)) else [entry]:
        code = getattr(reason, "code", None) or (
            reason.get("code") if isinstance(reason, dict) else None
        )
        if code:
            codes.append(str(code))
    return codes


def assert_target_admissible(
    conn,
    job: dict,
    target_host_id: str,
    *,
    preference: PlacementPreference | None = None,
    now: float | None = None,
) -> PlacementChoice:
    """`evaluate_migration_target`, raising instead of returning a refusal.

    For callers on a path where continuing past a refusal is not meaningful —
    a migration executor has nothing sensible to do with "no" except stop.
    The typed refusal is still carried on the exception so the caller can render
    it rather than reformat a string.
    """
    decision = evaluate_migration_target(
        conn, job, target_host_id, preference=preference, now=now
    )
    if isinstance(decision, PlacementRefused):
        raise MigrationRefused(decision)
    return decision


class MigrationRefused(RuntimeError):
    """A migration target that would not have been launchable."""

    def __init__(self, refusal: PlacementRefused):
        super().__init__(refusal.detail)
        self.refusal = refusal

    def as_dict(self) -> dict:
        return self.refusal.as_dict()


def migration_candidates(
    conn,
    job: dict,
    *,
    exclude_host_ids: Sequence[str] = (),
    now: float | None = None,
) -> list[dict]:
    """Every host this job could legally be migrated onto, right now.

    The same filter a launch runs, so a target that appears here is one the gate
    would have admitted at launch — which is the property the whole module
    exists to keep.
    """
    from control_plane.scheduler.filters import FilterContext, filter_hosts

    from control_plane.scheduler.snapshot import take_snapshot

    snapshot = take_snapshot(conn)
    excluded = {str(h) for h in exclude_host_ids if h}
    fleet = [h for h in snapshot.hosts if str(h.get("host_id")) not in excluded]
    eligible, _ = filter_hosts(
        _job_requirements(job),
        fleet,
        FilterContext(stale_host_ids=snapshot.stale_host_ids),
    )
    return eligible
