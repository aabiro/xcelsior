"""Archive and remove unused placement projections after native reader handoff.

Run only after all writers stop naming these columns and placement honors the
immutable admission preference. Historical/financial jobs are retained.
"""

from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "135"
down_revision = "134"
branch_labels = None
depends_on = None

COLUMNS = {
    "placement_min_uptime_bps": "INTEGER",
    "placement_min_tier": "TEXT",
    "placement_require_verified": "BOOLEAN NOT NULL DEFAULT FALSE",
    "placement_max_premium_bps": "INTEGER",
}
COLUMN_RECORD = "jsonb_build_object(" + ",".join(f"'{name}',{name}" for name in COLUMNS) + ")"


def _contract(conn):
    conn.execute(text("LOCK TABLE jobs IN ACCESS EXCLUSIVE MODE"))
    count = conn.execute(
        text("""SELECT count(*) FROM information_schema.columns WHERE
        table_schema='public' AND table_name='jobs' AND column_name=ANY(:names)"""),
        {"names": list(COLUMNS)},
    ).scalar_one()
    shape = dict(
        conn.execute(
            text("""SELECT column_name,data_type FROM information_schema.columns
        WHERE table_schema='public' AND table_name='jobs' AND column_name=ANY(:names)"""),
            {"names": list(COLUMNS)},
        ).fetchall()
    )
    if count and shape != {
        "placement_min_uptime_bps": "integer",
        "placement_min_tier": "text",
        "placement_require_verified": "boolean",
        "placement_max_premium_bps": "integer",
    }:
        raise RuntimeError("unexpected placement column types; contraction refused")
    archive_present = conn.execute(
        text("SELECT to_regclass('job_retired_projection_archive')")
    ).scalar_one()
    if count not in (0, 4) or (count == 0 and not archive_present):
        raise RuntimeError("unexpected placement column shape; contraction refused")
    conn.execute(text("""CREATE TABLE IF NOT EXISTS job_retired_projection_archive (
        job_id TEXT PRIMARY KEY,
        tenant_id TEXT,
        record JSONB NOT NULL,
        record_sha256 TEXT NOT NULL CHECK
            (record_sha256=encode(sha256(convert_to(record::text,'UTF8')),'hex')),
        archived_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        source_revision TEXT NOT NULL DEFAULT '135' CHECK(source_revision='135')
    )"""))
    if count == 0:
        return  # A committed unit can precede its Alembic stamp after a crash.
    conn.execute(
        text(f"""INSERT INTO job_retired_projection_archive(job_id,tenant_id,record,record_sha256)
        SELECT job_id,tenant_id,r,encode(sha256(convert_to(r::text,'UTF8')),'hex')
        FROM (SELECT job_id,tenant_id,jsonb_build_object('columns',{COLUMN_RECORD},
            'payload_lifecycle_present',payload?'lifecycle_intent',
            'payload_lifecycle',payload->'lifecycle_intent') AS r FROM jobs) evidence
        ON CONFLICT(job_id) DO NOTHING""")
    )
    mismatch = conn.execute(
        text(f"""SELECT count(*) FROM jobs j LEFT JOIN
        job_retired_projection_archive a USING(job_id)
        WHERE a.job_id IS NULL OR a.record->'columns' IS DISTINCT FROM {COLUMN_RECORD}""")
    ).scalar_one()
    if mismatch:
        raise RuntimeError("placement projection archive mismatch; contraction refused")
    # No CASCADE. Unexpected views/functions must be reviewed, not discarded.
    for name in COLUMNS:
        conn.execute(text(f"ALTER TABLE jobs DROP COLUMN {name}"))
    conn.execute(
        text("UPDATE jobs SET payload=payload-'lifecycle_intent' WHERE payload?'lifecycle_intent'")
    )


def upgrade():
    apply_in_own_transactions(
        [("archive retired job projections", _contract)],
        tables=["jobs", "job_retired_projection_archive", "reconciliation_queue"],
    )


def _restore(conn):
    conn.execute(text("LOCK TABLE jobs IN ACCESS EXCLUSIVE MODE"))
    for name, kind in COLUMNS.items():
        conn.execute(text(f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {name} {kind}"))
    conn.execute(text("""UPDATE jobs j SET
        placement_min_uptime_bps=(a.record->'columns'->>'placement_min_uptime_bps')::int,
        placement_min_tier=a.record->'columns'->>'placement_min_tier',
        placement_require_verified=(a.record->'columns'->>'placement_require_verified')::boolean,
        placement_max_premium_bps=(a.record->'columns'->>'placement_max_premium_bps')::int
        FROM job_retired_projection_archive a WHERE a.job_id=j.job_id"""))
    # Jobs admitted after contraction have no old projection record. Restore
    # their cache only from their explicit native admission, never payload.
    for jid, spec in conn.execute(text("""SELECT j.job_id,j.spec FROM jobs j
        WHERE NOT EXISTS(SELECT 1 FROM job_retired_projection_archive a WHERE a.job_id=j.job_id)
          AND j.spec->>'execution_spec_version'='v1'""")):
        preference = (spec or {}).get("placement_preference") or {}

        def bps(value):
            return (
                None
                if value is None
                else int((Decimal(str(value)) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
            )

        conn.execute(
            text("""UPDATE jobs SET placement_min_uptime_bps=:uptime,
            placement_min_tier=:tier,placement_require_verified=:verified,
            placement_max_premium_bps=:premium WHERE job_id=:jid"""),
            dict(
                jid=jid,
                uptime=bps(preference.get("min_uptime_pct")),
                tier=str(preference.get("min_tier") or "").strip().lower() or None,
                verified=preference.get("require_verified", False),
                premium=bps(preference.get("max_premium_pct")),
            ),
        )
    for name, expression in {
        "ck_jobs_placement_min_uptime_bps_range": "placement_min_uptime_bps IS NULL OR placement_min_uptime_bps BETWEEN 0 AND 10000",
        "ck_jobs_placement_max_premium_bps_range": "placement_max_premium_bps IS NULL OR placement_max_premium_bps BETWEEN 0 AND 100000000",
        "ck_jobs_placement_min_tier_shape": "placement_min_tier IS NULL OR placement_min_tier ~ '^[a-z][a-z_]{0,31}$'",
    }.items():
        conn.execute(text(f"ALTER TABLE jobs DROP CONSTRAINT IF EXISTS {name}"))
        conn.execute(text(f"ALTER TABLE jobs ADD CONSTRAINT {name} CHECK({expression}) NOT VALID"))
        conn.execute(text(f"ALTER TABLE jobs VALIDATE CONSTRAINT {name}"))
    # Audit evidence stays. Do not restore stale JSONB lifecycle keys: revision
    # 134's downgrade explicitly copies the current native intent for old code.


def downgrade():
    apply_in_own_transactions(
        [("restore retired placement projections", _restore)],
        tables=["jobs", "job_retired_projection_archive", "reconciliation_queue"],
    )
