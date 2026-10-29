"""Move a running job to another host, and prove it resumed.

Gate P5 clause 1: *"a migrated job resumes from its checkpoint, proven by
comparing state before and after — not by the absence of an error."*

This is the orchestration only. Both halves already exist and are reused rather
than reimplemented — `scheduler.checkpoint_container` freezes the source with
CRIU over SSH, `scheduler.resume_from_checkpoint` transfers and restarts on the
target — and `migration_gate.evaluate_migration_target` is the admission check.
A second copy of any of them would drift, which is the defect the shared ranker
was extracted to prevent.

## The order is the design

**Admission first, before anything is frozen.** A job stopped for a migration
that is then refused is a job the user lost for nothing. `assert_target_admissible`
runs against a snapshot taken *before* the checkpoint, so a target that would
fail the launch gate never causes an outage.

**Checkpoint before stop.** `docker checkpoint create` freezes and captures in
one step; there is no window where the container is gone and no checkpoint
exists.

**The source is not torn down until the target is running.** A migration that
destroys the only copy and then fails to resume has converted a running job into
a lost one — strictly worse than not migrating.

## What "resumes" has to mean

Returning `ok` because no exception was raised is what the clause explicitly
rejects. `MigrationOutcome` therefore carries `state_before` and `state_after`,
captured by a caller-supplied probe, and `resumed` is only true when a probe was
supplied *and* its readings match. **No probe means `resumed=None`** — unknown,
never `True`. A migration that cannot be verified is not a migration that
succeeded; it is one nobody checked.

That distinction is the whole clause. Everything else here is plumbing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from control_plane.scheduler.migration_gate import (
    MigrationRefused,
    evaluate_migration_target,
)
from control_plane.scheduler.preference import PlacementPreference, PlacementRefused

log = logging.getLogger(__name__)

#: A callable the caller supplies to read whatever "state" means for this job —
#: a step count, a checkpoint hash, a row count in the workload's own store.
#: Takes the job id, returns anything comparable by `==`.
StateProbe = Callable[[str], Any]


@dataclass(frozen=True)
class MigrationOutcome:
    """What happened, and whether anyone can tell that it worked."""

    ok: bool
    job_id: str
    source_host_id: str | None
    target_host_id: str | None
    #: `True` only when a probe ran on both sides and the readings matched.
    #: `False` when they differed. **`None` when nothing checked** — which is a
    #: different fact from success and must never be rendered as one.
    resumed: bool | None = None
    state_before: Any = None
    state_after: Any = None
    failure_code: str | None = None
    detail: str = ""
    refusal: dict | None = None
    checkpoint: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "job_id": self.job_id,
            "source_host_id": self.source_host_id,
            "target_host_id": self.target_host_id,
            "resumed": self.resumed,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "failure_code": self.failure_code,
            "detail": self.detail,
            "refusal": self.refusal,
        }


def _refused(job_id, source, target, refusal: PlacementRefused) -> MigrationOutcome:
    return MigrationOutcome(
        ok=False,
        job_id=job_id,
        source_host_id=source,
        target_host_id=target,
        failure_code=refusal.code,
        detail=refusal.detail,
        refusal=refusal.as_dict(),
    )


def migrate_job(
    job_id: str,
    target_host_id: str,
    *,
    preference: PlacementPreference | None = None,
    state_probe: StateProbe | None = None,
    now: float | None = None,
) -> MigrationOutcome:
    """Checkpoint `job_id` on its current host and resume it on `target_host_id`.

    Refuses before touching the running container whenever the target would not
    have passed admission at launch.
    """
    from scheduler import checkpoint_container, get_job, resume_from_checkpoint

    job = get_job(job_id)
    if not job:
        return MigrationOutcome(
            ok=False,
            job_id=job_id,
            source_host_id=None,
            target_host_id=target_host_id,
            failure_code="job_not_found",
            detail=f"no job {job_id!r}",
        )

    source_host_id = str(job.get("host_id") or "")
    status = str(job.get("status") or "")
    if status != "running":
        # Nothing to move, and freezing a non-running container would produce a
        # checkpoint of whatever it last was.
        return MigrationOutcome(
            ok=False,
            job_id=job_id,
            source_host_id=source_host_id,
            target_host_id=target_host_id,
            failure_code="job_not_running",
            detail=f"job is {status!r}, not running",
        )
    if source_host_id == str(target_host_id):
        return MigrationOutcome(
            ok=False,
            job_id=job_id,
            source_host_id=source_host_id,
            target_host_id=target_host_id,
            failure_code="already_on_target",
            detail="the job is already on that host",
        )

    # ── 1. Admission, before anything is frozen ──────────────────────
    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        decision = evaluate_migration_target(
            conn, job, target_host_id, preference=preference, now=now
        )
    if isinstance(decision, PlacementRefused):
        log.info(
            "MIGRATE REFUSED job=%s target=%s code=%s", job_id, target_host_id, decision.code
        )
        return _refused(job_id, source_host_id, target_host_id, decision)

    # ── 2. Read the state we will compare against ────────────────────
    state_before = None
    if state_probe is not None:
        try:
            state_before = state_probe(job_id)
        except Exception as exc:
            # A probe that cannot read *before* cannot verify *after*, and a
            # migration nobody can verify is one the clause does not accept.
            return MigrationOutcome(
                ok=False,
                job_id=job_id,
                source_host_id=source_host_id,
                target_host_id=target_host_id,
                failure_code="state_probe_failed",
                detail=f"could not read state before migrating: {exc}",
            )

    # ── 3. Checkpoint the source ─────────────────────────────────────
    checkpoint = checkpoint_container(source_host_id, job_id)
    if not checkpoint:
        return MigrationOutcome(
            ok=False,
            job_id=job_id,
            source_host_id=source_host_id,
            target_host_id=target_host_id,
            state_before=state_before,
            failure_code="checkpoint_failed",
            detail="the source container could not be checkpointed; it is untouched",
        )

    # ── 4. Resume on the target ──────────────────────────────────────
    resumed_ok = resume_from_checkpoint(job_id, target_host_id, checkpoint)
    if not resumed_ok:
        # The source was frozen, not destroyed. Saying so matters: the operator
        # needs to know whether they still have a job.
        return MigrationOutcome(
            ok=False,
            job_id=job_id,
            source_host_id=source_host_id,
            target_host_id=target_host_id,
            state_before=state_before,
            checkpoint=checkpoint,
            failure_code="resume_failed",
            detail=(
                "the checkpoint was taken but the target did not resume it; the "
                "source container was frozen rather than removed"
            ),
        )

    # ── 5. Prove it resumed, rather than assume it ───────────────────
    state_after = None
    resumed: bool | None = None
    if state_probe is not None:
        try:
            state_after = state_probe(job_id)
            resumed = state_after == state_before
        except Exception as exc:
            resumed = False
            log.warning("MIGRATE job=%s state probe failed after resume: %s", job_id, exc)

    log.info(
        "MIGRATED job=%s %s -> %s resumed=%s",
        job_id,
        source_host_id,
        target_host_id,
        resumed,
    )
    return MigrationOutcome(
        ok=True,
        job_id=job_id,
        source_host_id=source_host_id,
        target_host_id=target_host_id,
        resumed=resumed,
        state_before=state_before,
        state_after=state_after,
        checkpoint=checkpoint,
        detail=(
            "resumed and verified"
            if resumed
            else "resumed; **not verified** — no state probe was supplied"
            if resumed is None
            else "resumed but the state did not match"
        ),
    )


def migrate_to_cheapest(
    job_id: str,
    *,
    preference: PlacementPreference | None = None,
    state_probe: StateProbe | None = None,
    exclude_host_ids: tuple[str, ...] = (),
    now: float | None = None,
) -> MigrationOutcome:
    """Pick a legal target for `job_id` and migrate to it.

    "Cheapest" is the *preference's* verdict, not this function's: the same
    `choose_host` a launch uses ranks the candidates, so a migration cannot land
    somewhere a launch would have refused, and a stated preference is honoured
    on the target exactly as it was at launch.
    """
    from control_plane.db import control_plane_transaction
    from control_plane.scheduler.constrained_placement import evaluate_preference
    from control_plane.scheduler.migration_gate import migration_candidates
    from scheduler import get_job

    job = get_job(job_id)
    if not job:
        return MigrationOutcome(
            ok=False, job_id=job_id, source_host_id=None, target_host_id=None,
            failure_code="job_not_found", detail=f"no job {job_id!r}",
        )
    source = str(job.get("host_id") or "")
    excluded = tuple(exclude_host_ids) + ((source,) if source else ())

    with control_plane_transaction() as conn:
        candidates = migration_candidates(conn, job, exclude_host_ids=excluded, now=now)
        if not candidates:
            return MigrationOutcome(
                ok=False, job_id=job_id, source_host_id=source, target_host_id=None,
                failure_code="no_migration_target",
                detail="no other host would pass admission for this job",
            )
        decision, _ = evaluate_preference(
            conn, candidates, preference or PlacementPreference(), now=now
        )

    if isinstance(decision, PlacementRefused):
        return _refused(job_id, source, None, decision)

    return migrate_job(
        job_id,
        str(decision.host.get("host_id")),
        preference=preference,
        state_probe=state_probe,
        now=now,
    )


__all__ = [
    "MigrationOutcome",
    "MigrationRefused",
    "StateProbe",
    "migrate_job",
    "migrate_to_cheapest",
]
