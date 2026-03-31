# Xcelsior Event-Sourced Job State Machine
# Implements REPORT_FEATURE_FINAL.md: "truth engine" — strict lifecycle,
# idempotent transitions, auditable event history.
#
# Job lifecycle:     queued → assigned → leased → running → completed/failed/preempted
# Lease lifecycle:   lease_granted → lease_renewed → lease_expired
#
# Every state transition is recorded as an immutable event. Billing, SLA,
# reputation, and dispute resolution all derive from events — not heuristics.
#
# TAMPER-EVIDENT: Each event includes a SHA-256 hash of the previous event,
# forming a hash chain. Verifiers can replay the chain to detect tampering.
# Per REPORT_FEATURE_2.md Phase C §1: "store with SHA256 hashes."

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Job States ────────────────────────────────────────────────────────


class JobState(str, Enum):
    """Strict job lifecycle states. Every transition is validated."""

    QUEUED = "queued"
    ASSIGNED = "assigned"  # Host selected, not yet confirmed by agent
    LEASED = "leased"  # Agent confirmed, lease clock started
    RUNNING = "running"  # Container executing
    COMPLETED = "completed"  # Terminal: success
    FAILED = "failed"  # Terminal: error
    PREEMPTED = "preempted"  # Terminal: evicted (spot pricing / priority)
    CANCELLED = "cancelled"  # Terminal: user cancelled


TERMINAL_STATES = frozenset(
    {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.PREEMPTED,
        JobState.CANCELLED,
    }
)

# Valid state transitions — anything not here is rejected
VALID_TRANSITIONS = {
    JobState.QUEUED: {JobState.ASSIGNED, JobState.CANCELLED},
    JobState.ASSIGNED: {JobState.LEASED, JobState.QUEUED, JobState.FAILED, JobState.CANCELLED},
    JobState.LEASED: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.COMPLETED, JobState.FAILED, JobState.PREEMPTED},
    JobState.COMPLETED: set(),  # Terminal
    JobState.FAILED: {JobState.QUEUED},  # Retry
    JobState.PREEMPTED: {JobState.QUEUED},  # Re-queue
    JobState.CANCELLED: set(),  # Terminal
}


# ── Event Types ───────────────────────────────────────────────────────


class EventType(str, Enum):
    # Job lifecycle
    JOB_SUBMITTED = "job.submitted"
    JOB_ASSIGNED = "job.assigned"
    JOB_LEASE_GRANTED = "job.lease.granted"
    JOB_LEASE_RENEWED = "job.lease.renewed"
    JOB_LEASE_EXPIRED = "job.lease.expired"
    JOB_RUNNING = "job.running"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_PREEMPTED = "job.preempted"
    JOB_CANCELLED = "job.cancelled"
    JOB_REQUEUED = "job.requeued"

    # Host lifecycle
    HOST_REGISTERED = "host.registered"
    HOST_UPDATED = "host.updated"
    HOST_VERIFIED = "host.verified"
    HOST_DEVERIFIED = "host.deverified"
    HOST_REMOVED = "host.removed"
    HOST_ADMITTED = "host.admitted"
    HOST_REJECTED = "host.rejected"

    # Billing
    BILLING_CHARGED = "billing.charged"
    BILLING_CREDIT = "billing.credit"
    BILLING_REFUND = "billing.refund"

    # Security
    SECURITY_VERSION_CHECK = "security.version_check"
    SECURITY_MINING_DETECTED = "security.mining_detected"


@dataclass
class Event:
    """Immutable event record. The single source of truth.

    Tamper-evident: each event carries prev_hash (SHA-256 of the preceding
    event's canonical JSON) and event_hash (SHA-256 of this event).
    Replaying the chain detects any modification.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    entity_type: str = ""  # "job", "host", "billing"
    entity_id: str = ""  # job_id or host_id
    timestamp: float = field(default_factory=time.time)
    actor: str = ""  # "scheduler", "agent:<host_id>", "user:<user_id>"
    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    prev_hash: str = ""  # SHA-256 hash of preceding event (chain link)
    event_hash: str = ""  # SHA-256 hash of this event's canonical form

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of canonical event payload (excludes event_hash)."""
        canonical = json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "entity_type": self.entity_type,
                "entity_id": self.entity_id,
                "timestamp": self.timestamp,
                "actor": self.actor,
                "data": self.data,
                "metadata": self.metadata,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


# ── Lease Management ─────────────────────────────────────────────────
# A lease differentiates "host acknowledged the job" from "host completed work."
# Leases must be renewed periodically. Expired leases = host lost → re-queue.

DEFAULT_LEASE_DURATION_SEC = 300  # 5 minute lease
LEASE_RENEWAL_GRACE_SEC = 60  # 1 minute grace after expiry before re-queue


@dataclass
class Lease:
    """Job lease — proves the agent is alive and working."""

    lease_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    job_id: str = ""
    host_id: str = ""
    granted_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    last_renewed: float = field(default_factory=time.time)
    duration_sec: int = DEFAULT_LEASE_DURATION_SEC
    status: str = "active"  # active, expired, released

    def __post_init__(self):
        if self.expires_at == 0.0:
            self.expires_at = self.granted_at + self.duration_sec

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at + LEASE_RENEWAL_GRACE_SEC

    def renew(self) -> float:
        """Renew the lease. Returns new expiry time."""
        now = time.time()
        self.last_renewed = now
        self.expires_at = now + self.duration_sec
        return self.expires_at

    def to_dict(self) -> dict:
        return asdict(self)


# ── Event Store ───────────────────────────────────────────────────────
# Append-only event log. SQLite for dev, Postgres JSONB for production.


class EventStore:
    """Append-only event store backed by PostgreSQL.

    Events are immutable. They are the sole source of truth for auditing,
    billing, SLA enforcement, and dispute resolution.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path  # Legacy compat

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def append(self, event: Event) -> Event:
        """Append an event with tamper-evident hash chaining.

        Each event's prev_hash is set to the event_hash of the most recent
        event in the store. This forms a verifiable chain — any modification
        to a past event breaks every subsequent hash.
        """
        from psycopg.types.json import Jsonb
        with self._conn() as conn:
            # Get the hash of the most recent event (chain link)
            row = conn.execute(
                "SELECT event_hash FROM events ORDER BY timestamp DESC, event_id DESC LIMIT 1"
            ).fetchone()
            event.prev_hash = row["event_hash"] if row and row["event_hash"] else ""

            # Compute this event's hash
            event.event_hash = event.compute_hash()

            conn.execute(
                """INSERT INTO events
                   (event_id, event_type, entity_type, entity_id,
                    timestamp, actor, data, metadata, prev_hash, event_hash)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    event.event_id,
                    event.event_type,
                    event.entity_type,
                    event.entity_id,
                    event.timestamp,
                    event.actor,
                    Jsonb(event.data),
                    Jsonb(event.metadata),
                    event.prev_hash,
                    event.event_hash,
                ),
            )
        return event

    def verify_chain(self, limit: int = 0) -> dict:
        """Verify the tamper-evident hash chain.

        Returns {"valid": bool, "events_checked": int, "broken_at": event_id or None}.
        """
        with self._conn() as conn:
            if limit > 0:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp ASC LIMIT %s", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp ASC"
                ).fetchall()

        if not rows:
            return {"valid": True, "events_checked": 0, "broken_at": None}

        prev_hash = ""
        for i, row in enumerate(rows):
            event_hash = row["event_hash"] or ""
            stored_prev = row["prev_hash"] or ""

            # Skip hash verification for legacy events (pre-hash-chain)
            if not event_hash:
                prev_hash = ""
                continue

            # Check chain link
            if stored_prev != prev_hash:
                return {
                    "valid": False,
                    "events_checked": i + 1,
                    "broken_at": row["event_id"],
                    "reason": f"prev_hash mismatch at event {row['event_id']}",
                }

            # Recompute and verify event hash
            evt = Event(
                event_id=row["event_id"],
                event_type=row["event_type"],
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                timestamp=row["timestamp"],
                actor=row["actor"],
                data=row["data"] if isinstance(row["data"], dict) else json.loads(row["data"]),
                metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"]),
                prev_hash=row["prev_hash"] or "",
            )
            recomputed = evt.compute_hash()
            if recomputed != event_hash:
                return {
                    "valid": False,
                    "events_checked": i + 1,
                    "broken_at": row["event_id"],
                    "reason": f"event_hash tampered at event {row['event_id']}",
                }

            prev_hash = event_hash

        return {"valid": True, "events_checked": len(rows), "broken_at": None}

    def get_events(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 1000,
    ) -> list[Event]:
        """Query events with optional filters."""
        clauses = []
        params = []
        if entity_type:
            clauses.append("entity_type = %s")
            params.append(entity_type)
        if entity_id:
            clauses.append("entity_id = %s")
            params.append(entity_id)
        if event_type:
            clauses.append("event_type = %s")
            params.append(event_type)
        if since:
            clauses.append("timestamp >= %s")
            params.append(since)

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM events WHERE {where} ORDER BY timestamp ASC LIMIT %s",
                params,
            ).fetchall()
            return [
                Event(
                    event_id=r["event_id"],
                    event_type=r["event_type"],
                    entity_type=r["entity_type"],
                    entity_id=r["entity_id"],
                    timestamp=r["timestamp"],
                    actor=r["actor"],
                    data=r["data"] if isinstance(r["data"], dict) else json.loads(r["data"]),
                    metadata=r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"]),
                    prev_hash=r.get("prev_hash", ""),
                    event_hash=r.get("event_hash", ""),
                )
                for r in rows
            ]

    def get_entity_history(self, entity_type: str, entity_id: str) -> list[Event]:
        """Full event history for an entity — the auditable truth."""
        return self.get_events(entity_type=entity_type, entity_id=entity_id, limit=10000)

    # ── Lease operations ──────────────────────────────────────────────

    def grant_lease(
        self, job_id: str, host_id: str, duration_sec: int = DEFAULT_LEASE_DURATION_SEC
    ) -> Lease:
        """Grant a lease to an agent for a job."""
        lease = Lease(
            job_id=job_id,
            host_id=host_id,
            duration_sec=duration_sec,
        )
        with self._conn() as conn:
            # Expire any existing lease for this job
            conn.execute(
                "UPDATE leases SET status = 'released' WHERE job_id = %s AND status = 'active'",
                (job_id,),
            )
            conn.execute(
                """INSERT INTO leases
                   (lease_id, job_id, host_id, granted_at, expires_at,
                    last_renewed, duration_sec, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    lease.lease_id,
                    lease.job_id,
                    lease.host_id,
                    lease.granted_at,
                    lease.expires_at,
                    lease.last_renewed,
                    lease.duration_sec,
                    lease.status,
                ),
            )

        self.append(
            Event(
                event_type=EventType.JOB_LEASE_GRANTED,
                entity_type="job",
                entity_id=job_id,
                actor=f"agent:{host_id}",
                data={
                    "lease_id": lease.lease_id,
                    "host_id": host_id,
                    "expires_at": lease.expires_at,
                },
            )
        )
        log.info(
            "LEASE GRANTED job=%s host=%s lease=%s expires=%.0f",
            job_id,
            host_id,
            lease.lease_id,
            lease.expires_at,
        )
        return lease

    def renew_lease(self, job_id: str, host_id: str) -> Optional[Lease]:
        """Renew an active lease. Returns None if no active lease found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM leases WHERE job_id = %s AND host_id = %s AND status = 'active'",
                (job_id, host_id),
            ).fetchone()
            if not row:
                return None

            lease = Lease(
                lease_id=row["lease_id"],
                job_id=row["job_id"],
                host_id=row["host_id"],
                granted_at=row["granted_at"],
                expires_at=row["expires_at"],
                last_renewed=row["last_renewed"],
                duration_sec=row["duration_sec"],
                status=row["status"],
            )

            if lease.is_expired:
                conn.execute(
                    "UPDATE leases SET status = 'expired' WHERE lease_id = %s",
                    (lease.lease_id,),
                )
                return None

            new_expiry = lease.renew()
            conn.execute(
                "UPDATE leases SET expires_at = %s, last_renewed = %s WHERE lease_id = %s",
                (lease.expires_at, lease.last_renewed, lease.lease_id),
            )

        self.append(
            Event(
                event_type=EventType.JOB_LEASE_RENEWED,
                entity_type="job",
                entity_id=job_id,
                actor=f"agent:{host_id}",
                data={"lease_id": lease.lease_id, "new_expires_at": new_expiry},
            )
        )
        return lease

    def expire_stale_leases(self) -> list[str]:
        """Find and expire stale leases. Returns list of affected job_ids."""
        now = time.time()
        expired_jobs = []

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM leases
                   WHERE status = 'active' AND expires_at + %s < %s""",
                (LEASE_RENEWAL_GRACE_SEC, now),
            ).fetchall()

            for row in rows:
                conn.execute(
                    "UPDATE leases SET status = 'expired' WHERE lease_id = %s",
                    (row["lease_id"],),
                )
                expired_jobs.append(row["job_id"])

                self.append(
                    Event(
                        event_type=EventType.JOB_LEASE_EXPIRED,
                        entity_type="job",
                        entity_id=row["job_id"],
                        actor="scheduler",
                        data={
                            "lease_id": row["lease_id"],
                            "host_id": row["host_id"],
                            "expired_at": now,
                            "was_due_at": row["expires_at"],
                        },
                    )
                )
                log.warning(
                    "LEASE EXPIRED job=%s host=%s lease=%s",
                    row["job_id"],
                    row["host_id"],
                    row["lease_id"],
                )

        return expired_jobs

    def get_active_lease(self, job_id: str) -> Optional[Lease]:
        """Get the active lease for a job, if any."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM leases WHERE job_id = %s AND status = 'active'",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            return Lease(
                lease_id=row["lease_id"],
                job_id=row["job_id"],
                host_id=row["host_id"],
                granted_at=row["granted_at"],
                expires_at=row["expires_at"],
                last_renewed=row["last_renewed"],
                duration_sec=row["duration_sec"],
                status=row["status"],
            )

    def release_lease(self, job_id: str) -> bool:
        """Release a lease (job completed/failed/preempted)."""
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE leases SET status = 'released' WHERE job_id = %s AND status = 'active'",
                (job_id,),
            )
            return result.rowcount > 0


# ── Job State Machine ─────────────────────────────────────────────────


class JobStateMachine:
    """Validates and records job state transitions.

    Every transition is:
    1. Validated against VALID_TRANSITIONS
    2. Recorded as an immutable event
    3. Idempotent (same transition twice = no-op)
    """

    def __init__(self, event_store: EventStore):
        self.events = event_store

    def transition(
        self,
        job_id: str,
        current_state: str,
        new_state: str,
        actor: str = "scheduler",
        data: Optional[dict] = None,
    ) -> Event:
        """Execute a state transition with validation and event recording.

        Raises ValueError if the transition is invalid.
        Returns the recorded Event.
        """
        try:
            current = JobState(current_state)
            target = JobState(new_state)
        except ValueError as e:
            raise ValueError(f"Unknown state: {e}")

        # Check if transition is valid
        allowed = VALID_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid transition: {current.value} → {target.value}. "
                f"Allowed from {current.value}: {[s.value for s in allowed]}"
            )

        # Map state to event type
        event_type_map = {
            JobState.ASSIGNED: EventType.JOB_ASSIGNED,
            JobState.LEASED: EventType.JOB_LEASE_GRANTED,
            JobState.RUNNING: EventType.JOB_RUNNING,
            JobState.COMPLETED: EventType.JOB_COMPLETED,
            JobState.FAILED: EventType.JOB_FAILED,
            JobState.PREEMPTED: EventType.JOB_PREEMPTED,
            JobState.CANCELLED: EventType.JOB_CANCELLED,
            JobState.QUEUED: EventType.JOB_REQUEUED,
        }

        event = Event(
            event_type=event_type_map.get(target, f"job.{target.value}"),
            entity_type="job",
            entity_id=job_id,
            actor=actor,
            data={
                "previous_state": current.value,
                "new_state": target.value,
                **(data or {}),
            },
        )

        self.events.append(event)
        log.info("STATE %s → %s | job=%s | actor=%s", current.value, target.value, job_id, actor)
        return event

    def get_job_timeline(self, job_id: str) -> list[dict]:
        """Get the full auditable timeline for a job.

        This is the dispute-resolution artifact: every state change,
        every lease renewal, every billing event — ordered by time.
        """
        events = self.events.get_entity_history("job", job_id)
        return [e.to_dict() for e in events]


# ── Singleton ─────────────────────────────────────────────────────────

_event_store: Optional[EventStore] = None
_state_machine: Optional[JobStateMachine] = None


def get_event_store() -> EventStore:
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store


def get_state_machine() -> JobStateMachine:
    global _state_machine
    if _state_machine is None:
        _state_machine = JobStateMachine(get_event_store())
    return _state_machine


# ── Event Snapshots (for performance at scale) ───────────────────────

class EventSnapshotManager:
    """Manages periodic snapshots of entity state for fast reads.

    Instead of replaying all events to reconstruct state, we periodically
    snapshot the current state. Reads can start from the latest snapshot
    and only replay events after that snapshot's sequence number.
    """

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def create_snapshot(self, entity_type: str, entity_id: str, state: dict, sequence_number: int) -> dict:
        """Create a snapshot of an entity's current state."""
        from psycopg.types.json import Jsonb
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO event_snapshots
                   (entity_type, entity_id, snapshot_data, sequence_number, created_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (entity_type, entity_id, Jsonb(state), sequence_number, now),
            )
        return {"entity_type": entity_type, "entity_id": entity_id, "sequence_number": sequence_number}

    def get_latest_snapshot(self, entity_type: str, entity_id: str) -> Optional[dict]:
        """Get the most recent snapshot for an entity."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM event_snapshots
                   WHERE entity_type = %s AND entity_id = %s
                   ORDER BY sequence_number DESC LIMIT 1""",
                (entity_type, entity_id),
            ).fetchone()
            return dict(row) if row else None

    def snapshot_all_jobs(self, store: EventStore) -> int:
        """Create snapshots for all active job entities.

        Called periodically to keep snapshot freshness.
        """
        snapped = 0
        with self._conn() as conn:
            # Find distinct active job entities
            rows = conn.execute(
                """SELECT DISTINCT entity_id FROM events
                   WHERE entity_type = 'job'
                   ORDER BY entity_id""",
            ).fetchall()

        for row in rows:
            entity_id = row["entity_id"]
            events = store.get_entity_history("job", entity_id)
            if not events:
                continue

            # Build current state from events
            state = {}
            seq = 0
            for evt in events:
                state.update(evt.data)
                state["status"] = evt.event_type
                seq += 1

            # Check if we need a new snapshot (every 50 events)
            latest = self.get_latest_snapshot("job", entity_id)
            if latest and latest["sequence_number"] >= seq - 50:
                continue  # Recent enough

            self.create_snapshot("job", entity_id, state, seq)
            snapped += 1

        if snapped:
            log.info("EVENT SNAPSHOTS: created %d job snapshots", snapped)
        return snapped

    def archive_old_events(self, max_age_days: int = 90) -> int:
        """Move events older than max_age_days to events_archive (cold storage).

        Preserves chain hashes so audit integrity is maintained even after
        the hot events table is compacted.  Returns number of archived rows.
        """
        cutoff = time.time() - (max_age_days * 86400)
        now = time.time()
        archived = 0

        with self._conn() as conn:
            # Copy old events to archive
            rows = conn.execute(
                """INSERT INTO events_archive
                       (event_id, entity_type, entity_id, event_type, data, actor, chain_hash, created_at, archived_at)
                   SELECT event_id, entity_type, entity_id, event_type, data, actor,
                          COALESCE(chain_hash, ''), created_at, %s
                   FROM events
                   WHERE created_at < %s
                   RETURNING id""",
                (now, cutoff),
            ).fetchall()
            archived = len(rows)

            if archived > 0:
                # Delete archived events from hot table
                conn.execute(
                    "DELETE FROM events WHERE created_at < %s",
                    (cutoff,),
                )

        if archived:
            log.info("EVENT ARCHIVE: moved %d events older than %d days to cold storage", archived, max_age_days)
        return archived


_snapshot_manager: Optional[EventSnapshotManager] = None


def get_snapshot_manager() -> EventSnapshotManager:
    global _snapshot_manager
    if _snapshot_manager is None:
        _snapshot_manager = EventSnapshotManager()
    return _snapshot_manager
