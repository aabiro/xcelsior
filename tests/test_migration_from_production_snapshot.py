"""Production-shaped migration rehearsal (Track B B1.2).

`test_from_empty_bootstrap.py` proves the DDL applies to an empty database.
It cannot prove the *backfills* work, because a from-empty upgrade has no
rows to backfill: migration 054's batched `SKIP LOCKED` job/host projection,
its hard verification of unmapped legacy statuses, 055's transitional lease
backfill, and 059's drift backfill are all no-ops against empty tables.

This gate builds a 053-shaped database, seeds representative legacy rows —
one job per legacy status, hosts in every admission shape, active and
expired leases, money-bearing billing rows — then upgrades to head and
asserts the properties §26.5 and companion §16.1 require: backfill counts,
uniqueness, nullability, money totals, and state distributions.

A sanitized snapshot of the real production database would be a stronger
input, but it is not available to CI and must not be pulled into a test
environment. Seeding a 053 schema is reproducible, runs anywhere, and
exercises the same code paths.

Blueprint §13.1/§13.8/§26.5, companion §16.1, Track B §B1.2.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import pytest

# Reuse the throwaway-database machinery rather than duplicating it. The
# from-empty gate owns these helpers; this module is a second consumer.
from tests.test_from_empty_bootstrap import (
    EXPECTED_HEAD,
    PROJECT_ROOT,
    _admin_dsn,
    _db_url,
    _drop_database,
    _try_create_database,
)

# The revision the governing documents call the pre-control-plane baseline
# ("The repository currently ends at 053" — companion §14).
BASELINE_REVISION = "053"

# Mirrors migration 054's _PHASE_CASE_SQL / _DESIRED_STATE_CASE_SQL. Kept
# here as an independent restatement on purpose: if someone edits the
# migration's mapping, this test must disagree and fail rather than import
# the same expression and agree with itself.
LEGACY_STATUS_PROJECTION: dict[str, tuple[str, str]] = {
    "queued": ("pending", "running"),
    "preempted": ("pending", "running"),
    "assigned": ("scheduled", "running"),
    "leased": ("scheduled", "running"),
    "starting": ("starting", "running"),
    "restarting": ("starting", "running"),
    "running": ("running", "running"),
    "stopping": ("running", "stopped"),
    "completed": ("succeeded", "running"),
    "failed": ("failed", "running"),
    "cancelled": ("stopped", "stopped"),
    "stopped": ("stopped", "stopped"),
    "paused": ("stopped", "stopped"),
    "terminated": ("stopped", "stopped"),
}

# Money-bearing rows seeded into billing_cycles.amount_cad; the sum must
# survive the upgrade exactly (companion §16.1 "money totals").
#
# These are `double precision` in the current schema. Companion §4.4 rule 6
# requires integer minor units or NUMERIC for money — that defect is real
# and tracked as Track B B9.3a; this rehearsal asserts the upgrade does not
# *change* the values, which is a separate property from storing them in a
# sound type.
SEEDED_AMOUNT_CAD = (12.34, 500.00, 0.01, 9999.99)


def _alembic(
    db_dsn: str,
    *args: str,
    check: bool = True,
    timeout_sec: float = 600,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["XCELSIOR_POSTGRES_DSN"] = db_dsn
    env["XCELSIOR_PG_DSN"] = db_dsn
    env["DATABASE_URL"] = db_dsn
    env["XCELSIOR_DB_BACKEND"] = "postgres"
    r = subprocess.run(
        ["alembic", *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if check:
        assert r.returncode == 0, (
            f"alembic {' '.join(args)} failed:\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
        )
    return r


def _connect(db_dsn: str):
    import psycopg

    return psycopg.connect(db_dsn)


def _seed_053_production_shape(db_dsn: str) -> dict[str, int]:
    """Insert representative legacy rows into a 053-shaped database.

    Returns the seeded counts so post-upgrade assertions can prove nothing
    was lost or duplicated.
    """
    now = time.time()
    counts: dict[str, int] = {}

    with _connect(db_dsn) as conn:
        # ── hosts: every admission shape 059's trigger distinguishes ──
        host_rows = [
            ("host-admitted", "active", {"admitted": True, "gpu_count": 2,
                                         "gpu_model": "RTX 4090", "total_vram_gb": 48}),
            ("host-pending", "pending", {"admitted": False, "gpu_count": 1,
                                         "gpu_model": "RTX 3060", "total_vram_gb": 12}),
            ("host-draining", "draining", {"admitted": True, "gpu_count": 1,
                                           "gpu_model": "A100", "total_vram_gb": 80}),
            ("host-offline", "offline", {"admitted": True, "gpu_count": 1,
                                         "gpu_model": "L40S", "total_vram_gb": 48}),
            # A host whose payload omits `admitted` entirely — the shape
            # that predates the flag. It must still project, not NULL out.
            ("host-legacy", "active", {"gpu_count": 1, "gpu_model": "RTX 2060",
                                       "total_vram_gb": 6}),
        ]
        for host_id, status, payload in host_rows:
            conn.execute(
                "INSERT INTO hosts (host_id, status, registered_at, payload) "
                "VALUES (%s, %s, %s, %s)",
                (host_id, status, now - 3600, json.dumps(payload)),
            )
        counts["hosts"] = len(host_rows)

        # ── jobs: one per legacy status, plus tenancy payload variety ──
        seeded_jobs = 0
        for i, status in enumerate(sorted(LEGACY_STATUS_PROJECTION)):
            payload = {
                "owner": f"user-{i}",
                "tenant_id": f"tenant-{i % 3}",
                "team_id": f"team-{i % 2}",
                "image": "nvidia/cuda:12.4.0-base",
                "gpu_model": "RTX 4090",
            }
            conn.execute(
                "INSERT INTO jobs (job_id, status, priority, submitted_at, "
                "host_id, payload, error_message, pricing_mode) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    f"job-{status}",
                    status,
                    i % 5,
                    now - (600 * i),
                    "host-admitted" if status in ("assigned", "leased", "running") else None,
                    json.dumps(payload),
                    "",
                    "on_demand",
                ),
            )
            seeded_jobs += 1

        # A job whose payload carries no tenancy at all — the oldest shape.
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, "
            "host_id, payload, error_message, pricing_mode) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            ("job-no-tenancy", "queued", 0, now, None, json.dumps({}), "", "on_demand"),
        )
        seeded_jobs += 1
        counts["jobs"] = seeded_jobs

        # ── leases: one active (055 backfills it), one expired (it must not) ──
        conn.execute(
            "INSERT INTO leases (lease_id, job_id, host_id, granted_at, "
            "expires_at, last_renewed, duration_sec, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            ("lease-active", "job-running", "host-admitted",
             now - 60, now + 3600, now - 30, 3600, "active"),
        )
        conn.execute(
            "INSERT INTO leases (lease_id, job_id, host_id, granted_at, "
            "expires_at, last_renewed, duration_sec, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            ("lease-expired", "job-completed", "host-admitted",
             now - 7200, now - 3600, now - 7200, 3600, "expired"),
        )
        counts["leases"] = 2

        # ── money: totals must survive the upgrade exactly ──
        for i, amount in enumerate(SEEDED_AMOUNT_CAD):
            conn.execute(
                "INSERT INTO billing_cycles (cycle_id, customer_id, job_id, "
                "period_start, period_end, duration_seconds, rate_per_hour, "
                "amount_cad, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    f"cycle-{i}",
                    f"user-{i}",
                    "job-running",
                    now - 3600,
                    now,
                    3600.0,
                    round(amount, 2),
                    amount,
                    now,
                ),
            )
        counts["billing_cycles"] = len(SEEDED_AMOUNT_CAD)

        conn.commit()
    return counts


@pytest.fixture
def snapshot_db():
    """A throwaway database upgraded to 053 and seeded with legacy rows."""
    admin = _admin_dsn()
    if admin is None:
        pytest.skip("no Postgres DSN in environment")

    dbname = f"migsnap_{uuid.uuid4().hex[:10]}"
    if not _try_create_database(admin, dbname):
        pytest.skip("CREATE DATABASE unavailable")

    dsn = _db_url(admin, dbname)
    try:
        _alembic(dsn, "upgrade", BASELINE_REVISION)
        yield dsn
    finally:
        _drop_database(admin, dbname)


@pytest.fixture
def seeded_snapshot_db(snapshot_db):
    counts = _seed_053_production_shape(snapshot_db)
    return snapshot_db, counts


# ───────────────────────── the rehearsal ─────────────────────────


def test_production_shaped_upgrade_to_head(seeded_snapshot_db):
    """053 + representative rows → head, with every §16.1 property checked."""
    dsn, seeded = seeded_snapshot_db

    _alembic(dsn, "upgrade", "head")

    with _connect(dsn) as conn:
        # ── head reached ──
        rev = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        assert rev is not None and rev[0] == EXPECTED_HEAD, (
            f"expected head {EXPECTED_HEAD}, database is at {rev}"
        )

        # ── no data loss or duplication ──
        for table, expected in seeded.items():
            actual = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            assert actual is not None and actual[0] == expected, (
                f"{table}: seeded {expected} rows, found {actual[0]} after "
                f"upgrade. A migration lost or duplicated production rows."
            )

        # ── 054 backfill: nullability. The migration raises on unmapped
        #    rows, so reaching head already implies this — assert it
        #    positively so a future migration that reintroduces NULLs is
        #    caught here rather than at NOT NULL time.
        unmapped = conn.execute(
            "SELECT count(*) FROM jobs WHERE phase IS NULL OR desired_state IS NULL"
        ).fetchone()
        assert unmapped is not None and unmapped[0] == 0, (
            f"{unmapped[0]} job rows have no phase/desired_state projection"
        )
        unobserved = conn.execute(
            "SELECT count(*) FROM hosts WHERE last_observed_at IS NULL"
        ).fetchone()
        assert unobserved is not None and unobserved[0] == 0

        # ── 054 backfill: state distribution, per legacy status ──
        for status, (want_phase, want_desired) in LEGACY_STATUS_PROJECTION.items():
            row = conn.execute(
                "SELECT phase, desired_state FROM jobs WHERE job_id = %s",
                (f"job-{status}",),
            ).fetchone()
            assert row is not None, f"job-{status} vanished during upgrade"
            assert row == (want_phase, want_desired), (
                f"legacy status {status!r} projected to {row}, expected "
                f"({want_phase!r}, {want_desired!r}). The migration's status "
                f"mapping changed without updating this rehearsal."
            )

        # ── 054 backfill: tenancy projected out of the JSONB payload ──
        row = conn.execute(
            "SELECT tenant_id, team_id, owner_id FROM jobs WHERE job_id = %s",
            ("job-queued",),
        ).fetchone()
        assert row is not None
        tenant_id, team_id, owner_id = row
        assert tenant_id and tenant_id.startswith("tenant-"), (
            f"tenant_id not projected from payload: {tenant_id!r}"
        )
        assert team_id and team_id.startswith("team-")
        assert owner_id and owner_id.startswith("user-")

        # A payload with no tenancy must still project phase/desired_state
        # (it is a legitimate legacy shape, not a data bug).
        row = conn.execute(
            "SELECT phase, desired_state FROM jobs WHERE job_id = %s",
            ("job-no-tenancy",),
        ).fetchone()
        assert row == ("pending", "running")

        # ── 054 backfill: queue ordering columns are populated ──
        missing_queue_cols = conn.execute(
            "SELECT count(*) FROM jobs WHERE queued_at IS NULL "
            "OR effective_priority IS NULL"
        ).fetchone()
        assert missing_queue_cols is not None and missing_queue_cols[0] == 0, (
            "jobs are missing queued_at/effective_priority — the scheduler's "
            "§10.2 claim ordering would be undefined for migrated rows"
        )

        # ── 055 backfill: the ACTIVE legacy lease became attempt + lease ──
        attempts = conn.execute(
            "SELECT count(*) FROM job_attempts WHERE job_id = %s", ("job-running",)
        ).fetchone()
        assert attempts is not None and attempts[0] == 1, (
            "the active legacy lease on job-running was not backfilled into a "
            "transitional job_attempts row (migration 055)"
        )
        leases = conn.execute(
            "SELECT count(*) FROM placement_leases WHERE job_id = %s", ("job-running",)
        ).fetchone()
        assert leases is not None and leases[0] == 1

        # ...and the EXPIRED one did not. Backfilling a dead lease would
        # hand a terminated job a live attempt.
        stale = conn.execute(
            "SELECT count(*) FROM job_attempts WHERE job_id = %s", ("job-completed",)
        ).fetchone()
        assert stale is not None and stale[0] == 0, (
            "an expired legacy lease was backfilled into an attempt; only "
            "active leases are transitional records (migration 055)"
        )

        # ── uniqueness invariants hold over migrated data (§8.1) ──
        dupes = conn.execute(
            """
            SELECT job_id, count(*) FROM job_attempts
             WHERE status IN ('reserved','command_pending','lease_offered',
                              'lease_claimed','starting','running')
             GROUP BY job_id HAVING count(*) > 1
            """
        ).fetchall()
        assert not dupes, f"migrated data violates uq_job_one_active_attempt: {dupes}"

        # ── 059 trigger: host admission derived from the payload flag ──
        admitted = conn.execute(
            "SELECT administrative_state FROM hosts WHERE host_id = %s",
            ("host-admitted",),
        ).fetchone()
        assert admitted is not None and admitted[0] is not None
        pending = conn.execute(
            "SELECT administrative_state FROM hosts WHERE host_id = %s",
            ("host-pending",),
        ).fetchone()
        assert pending is not None and pending[0] is not None
        assert admitted[0] != pending[0], (
            "an admitted host and a pending host projected to the same "
            "administrative_state; 054's status-only rule was too loose and "
            "059 was supposed to honour the payload `admitted` flag"
        )

        # ── money totals survive exactly ──
        total = conn.execute("SELECT sum(amount_cad) FROM billing_cycles").fetchone()
        assert total is not None
        assert abs(total[0] - sum(SEEDED_AMOUNT_CAD)) < 1e-9, (
            f"billing total changed during upgrade: seeded "
            f"{sum(SEEDED_AMOUNT_CAD)}, found {total[0]}"
        )


def test_production_shaped_upgrade_is_reversible(seeded_snapshot_db):
    """up → down → up over populated tables (§13.8, README rule 6).

    Reversibility on an empty database proves far less than reversibility
    over rows: a downgrade that drops a backfilled column and an upgrade
    that re-derives it must both survive real data.
    """
    dsn, seeded = seeded_snapshot_db

    _alembic(dsn, "upgrade", "head")
    _alembic(dsn, "downgrade", BASELINE_REVISION)

    with _connect(dsn) as conn:
        rev = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        assert rev is not None and rev[0] == BASELINE_REVISION
        # Legacy truth is untouched by the round trip.
        for table, expected in seeded.items():
            actual = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            assert actual is not None and actual[0] == expected, (
                f"{table} lost rows during downgrade to {BASELINE_REVISION}"
            )

    _alembic(dsn, "upgrade", "head")

    with _connect(dsn) as conn:
        rev = conn.execute("SELECT version_num FROM alembic_version").fetchone()
        assert rev is not None and rev[0] == EXPECTED_HEAD
        # The backfill re-derives cleanly the second time.
        unmapped = conn.execute(
            "SELECT count(*) FROM jobs WHERE phase IS NULL OR desired_state IS NULL"
        ).fetchone()
        assert unmapped is not None and unmapped[0] == 0, (
            "the second upgrade left rows unprojected — the backfill is not "
            "idempotent across a down/up cycle"
        )


def test_unknown_legacy_status_aborts_the_migration(snapshot_db):
    """054's hard verification must actually fire (blueprint §13.1).

    Track A claims the backfill aborts on unmapped rows rather than
    silently leaving NULLs. A from-empty upgrade can never demonstrate
    that, so drive it: seed a status the mapping does not know and require
    the upgrade to fail loudly.

    This is the drive that found the original defect — the backfill loop
    re-selected the unmappable row forever (its `ELSE NULL` branch wrote
    NULL back into the column the batch predicate filtered on), so
    `alembic upgrade` hung holding locks and never reached the
    verification step. The short subprocess timeout below keeps a
    regression to that behaviour a fast, legible failure instead of a
    180-second pytest timeout with an unreadable thread dump.
    """
    now = time.time()
    with _connect(snapshot_db) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, "
            "host_id, payload, error_message, pricing_mode) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            ("job-bogus", "teleporting", 0, now, None, json.dumps({}), "", "on_demand"),
        )
        conn.commit()

    try:
        result = _alembic(snapshot_db, "upgrade", "head", check=False, timeout_sec=90)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "migration 054 hung on an unmappable legacy status instead of "
            "aborting. The jobs backfill batch predicate must exclude rows "
            "the status mapping cannot project (`(<phase case>) IS NOT NULL`), "
            "or the loop re-selects them forever while holding its locks."
        )

    assert result.returncode != 0, (
        "migration 054 accepted an unmappable legacy status. The backfill "
        "must abort so the data bug surfaces now, not at NOT NULL time."
    )
    combined = result.stdout + result.stderr
    assert "teleporting" in combined, (
        f"the failure did not name the offending status, so an operator "
        f"cannot act on it:\n{combined[-2000:]}"
    )
