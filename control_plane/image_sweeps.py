"""Create a sweep: N instances from one snapshot, recorded as one thing.

Gate P7 piece 2. The clause is *"a sweep of N nodes from one snapshot is
byte-identical in environment"*, and the reason this is a module rather than a
loop in a route handler is that **the record is the feature**.

Calling the launch path N times and returning N job ids leaves the claim — that
these N came from one snapshot — as an intention held by whoever made the
request. Nothing afterwards can check it, partial failure is invisible, and the
fingerprint comparison in piece 3 has no set to compare across.

## What this refuses, and why refusing is the point

**An image with no recorded digest.** `user_images.image_digest` is nullable
because a snapshot can legitimately not know its own digest — an older agent, or
a push that succeeded while the inspect failed. A sweep from such an image
cannot support a byte-identity claim, so it is refused rather than created
against the mutable tag. Falling back to the tag is the substitution that would
leave the clause unprovable while looking met, which is exactly what migration
112 exists to prevent.

**An image that is not ready.** A sweep from a `pending` snapshot would launch
against bytes that may still be uploading.

## The digest is resolved once

Copied onto the sweep row at creation, and every member launches against that
one string. Resolving per member would reintroduce the race the digest closes:
the tag can move between the first resolution and the last, and then "N nodes
from one snapshot" is false in precisely the way nobody would notice.

## Partial failure is a result, not an exception

A launch that fails for member 3 does not abort the sweep. The member is
recorded `failed` with its reason and the others proceed, because "3 of 5
launched, and here are the two that did not" is the answer an operator needs.
Aborting would destroy that and leave three orphaned instances running.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("xcelsior")

#: Upper bound mirrored from `ck_image_sweeps_count`. Checked here so a caller
#: gets a typed refusal rather than a constraint violation, and checked there so
#: this one cannot be bypassed by another writer.
MAX_SWEEP_SIZE = 64


class SweepRefused(Exception):
    """A sweep that must not be created, with the reason a caller can act on."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class SweepMember:
    index: int
    job_id: str | None = None
    host_id: str | None = None
    state: str = "pending"
    failure_code: str | None = None


@dataclass
class Sweep:
    sweep_id: str
    image_id: str
    image_digest: str
    requested_count: int
    state: str
    members: list[SweepMember] = field(default_factory=list)

    @property
    def launched(self) -> int:
        return sum(1 for m in self.members if m.job_id)

    @property
    def distinct_hosts(self) -> int:
        """How many hosts this sweep actually covers.

        Reported rather than asserted. A sweep landing entirely on one host is
        not wrong, but it establishes much less than one spread across several,
        and a caller cannot tell the difference unless this is surfaced.
        """
        return len({m.host_id for m in self.members if m.host_id})

    def as_dict(self) -> dict:
        return {
            "sweep_id": self.sweep_id,
            "image_id": self.image_id,
            "image_digest": self.image_digest,
            "requested_count": self.requested_count,
            "launched_count": self.launched,
            "distinct_hosts": self.distinct_hosts,
            "state": self.state,
            "members": [
                {
                    "index": m.index,
                    "job_id": m.job_id,
                    "host_id": m.host_id,
                    "state": m.state,
                    "failure_code": m.failure_code,
                }
                for m in self.members
            ],
        }


def _pinned_image(conn, image_id: str, owner_id: str) -> tuple[str, str]:
    """`(image_ref, image_digest)` for an image that may be swept, or refuse."""
    row = conn.execute(
        "SELECT image_ref, image_digest, status, owner_id "
        "  FROM user_images WHERE image_id = %s AND deleted_at = 0",
        (image_id,),
    ).fetchone()
    if not row:
        raise SweepRefused("image_not_found", f"no image {image_id!r}")

    image_ref, digest, status, image_owner = row[0], row[1], row[2], row[3]
    if str(image_owner) != str(owner_id):
        # Not-found rather than forbidden: no existence oracle for another
        # tenant's images.
        raise SweepRefused("image_not_found", f"no image {image_id!r}")
    if status != "ready":
        raise SweepRefused(
            "image_not_ready",
            f"image is {status!r}; a sweep would launch against bytes that may still be uploading",
        )
    if not digest or "@sha256:" not in str(digest):
        raise SweepRefused(
            "image_digest_unknown",
            "this snapshot has no recorded manifest digest, so a sweep from it "
            "cannot establish that the nodes ran the same bytes. Re-snapshot "
            "with an agent that reports one; launching from the mutable tag "
            "would make the guarantee unprovable rather than merely unproven.",
        )
    return str(image_ref), str(digest)


def create_sweep(
    conn,
    *,
    tenant_id: str,
    owner_id: str,
    image_id: str,
    count: int,
    launch: Callable[[str, int], dict[str, Any]],
) -> Sweep:
    """Record a sweep and launch its members. Returns the record, not job ids.

    `launch(image_digest, index)` performs one launch and returns at least
    `{"job_id": ..., "host_id": ...}`; anything it raises is recorded against
    that member and the sweep continues.
    """
    if count < 1 or count > MAX_SWEEP_SIZE:
        raise SweepRefused("invalid_count", f"count must be between 1 and {MAX_SWEEP_SIZE}")

    _, digest = _pinned_image(conn, image_id, owner_id)
    sweep_id = f"swp-{uuid.uuid4().hex[:12]}"

    conn.execute(
        """
        INSERT INTO image_sweeps
               (sweep_id, tenant_id, owner_id, image_id, image_digest,
                requested_count, state)
        VALUES (%s, %s, %s, %s, %s, %s, 'launching')
        """,
        (sweep_id, tenant_id, owner_id, image_id, digest, count),
    )
    for index in range(count):
        conn.execute(
            "INSERT INTO image_sweep_members (sweep_id, member_index, tenant_id) "
            "VALUES (%s, %s, %s)",
            (sweep_id, index, tenant_id),
        )

    members: list[SweepMember] = []
    for index in range(count):
        member = SweepMember(index=index)
        try:
            result = launch(digest, index)
            member.job_id = str(result.get("job_id") or "") or None
            member.host_id = str(result.get("host_id") or "") or None
            if not member.job_id:
                raise RuntimeError("launch returned no job id")
            member.state = "launched"
        except Exception as exc:
            # Recorded, not raised. "3 of 5 launched, and here are the two that
            # did not" is the answer; aborting would discard it and strand the
            # instances that did start.
            member.state = "failed"
            member.failure_code = str(exc)[:200]
            log.warning("sweep %s member %d failed to launch: %s", sweep_id, index, exc)

        conn.execute(
            "UPDATE image_sweep_members SET job_id=%s, host_id=%s, state=%s, "
            "failure_code=%s, updated_at=clock_timestamp() "
            " WHERE sweep_id=%s AND member_index=%s",
            (member.job_id, member.host_id, member.state, member.failure_code, sweep_id, index),
        )
        members.append(member)

    launched = sum(1 for m in members if m.job_id)
    state = "running" if launched else "failed"
    conn.execute(
        "UPDATE image_sweeps SET state=%s, updated_at=clock_timestamp()  WHERE sweep_id=%s",
        (state, sweep_id),
    )

    log.info(
        "SWEEP %s image=%s digest=%s launched=%d/%d hosts=%d",
        sweep_id,
        image_id,
        digest.split("@")[-1][:19],
        launched,
        count,
        len({m.host_id for m in members if m.host_id}),
    )
    return Sweep(
        sweep_id=sweep_id,
        image_id=image_id,
        image_digest=digest,
        requested_count=count,
        state=state,
        members=members,
    )


def read_sweep(conn, sweep_id: str, *, tenant_id: str) -> Sweep | None:
    """A sweep and its members, or `None` when it is not this tenant's."""
    row = conn.execute(
        "SELECT image_id, image_digest, requested_count, state "
        "  FROM image_sweeps WHERE sweep_id = %s AND tenant_id = %s",
        (sweep_id, tenant_id),
    ).fetchone()
    if not row:
        return None
    member_rows = conn.execute(
        "SELECT member_index, job_id, host_id, state, failure_code "
        "  FROM image_sweep_members WHERE sweep_id = %s ORDER BY member_index",
        (sweep_id,),
    ).fetchall()
    return Sweep(
        sweep_id=sweep_id,
        image_id=str(row[0]),
        image_digest=str(row[1]),
        requested_count=int(row[2]),
        state=str(row[3]),
        members=[
            SweepMember(
                index=int(m[0]),
                job_id=m[1],
                host_id=m[2],
                state=str(m[3]),
                failure_code=m[4],
            )
            for m in member_rows
        ],
    )


__all__ = [
    "MAX_SWEEP_SIZE",
    "Sweep",
    "SweepMember",
    "SweepRefused",
    "create_sweep",
    "read_sweep",
]
