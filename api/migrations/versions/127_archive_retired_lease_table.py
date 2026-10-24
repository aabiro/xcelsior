"""Archive inactive v1 lease rows, then remove their obsolete table/columns.

Apply after the native-only runtime handoff. Active or unknown legacy authority,
unexpected columns and archive mismatches abort the whole transaction. Native
placement leases, job history, billing and event hash chains are untouched.
"""

from alembic import op

revision = "127"
down_revision = "126"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout='5s'")
    op.execute("SET LOCAL statement_timeout='5min'")
    op.execute("LOCK TABLE leases IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM leases WHERE status IS NULL OR status NOT IN ('expired','released')) THEN
            RAISE EXCEPTION 'native cutover required: legacy leases still have authority or unknown status';
        END IF;
        IF (SELECT array_agg(column_name::text ORDER BY column_name)
            FROM information_schema.columns WHERE table_schema='public' AND table_name='leases')
            IS DISTINCT FROM ARRAY['duration_sec','expires_at','granted_at','host_id',
                                  'job_id','last_renewed','lease_id','status']::text[] THEN
            RAISE EXCEPTION 'unexpected legacy lease columns; archive schema must be reviewed';
        END IF;
    END $$""")
    op.execute("""CREATE TABLE legacy_lease_archive (
        lease_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL CHECK (tenant_id<>''),
        tenant_attributed BOOLEAN NOT NULL,
        record JSONB NOT NULL CHECK (record->>'lease_id'=lease_id),
        record_sha256 TEXT NOT NULL CHECK (record_sha256=encode(sha256(convert_to(record::text,'UTF8')),'hex')),
        archived_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        source_revision TEXT NOT NULL DEFAULT '127'
    )""")
    # Unresolved history stays in a system quarantine scope. This is not an
    # invented workload owner; tenant_attributed explicitly records the gap.
    op.execute(
        """INSERT INTO legacy_lease_archive(lease_id,tenant_id,tenant_attributed,record,record_sha256)
        SELECT l.lease_id,COALESCE(NULLIF(j.tenant_id,''),'system:unattributed-legacy-lease-archive'),
               NULLIF(j.tenant_id,'') IS NOT NULL,
               to_jsonb(l),encode(sha256(convert_to(to_jsonb(l)::text,'UTF8')),'hex')
        FROM leases l LEFT JOIN jobs j ON j.job_id=l.job_id"""
    )
    op.execute(
        "CREATE INDEX legacy_lease_archive_tenant ON legacy_lease_archive(tenant_id,lease_id)"
    )
    op.execute("""DO $$ BEGIN
        IF (SELECT count(*) FROM leases)<>(SELECT count(*) FROM legacy_lease_archive)
           OR EXISTS (SELECT 1 FROM leases l LEFT JOIN legacy_lease_archive a USING(lease_id)
                      WHERE a.lease_id IS NULL OR a.record IS DISTINCT FROM to_jsonb(l)) THEN
            RAISE EXCEPTION 'legacy lease archive verification failed';
        END IF;
    END $$""")
    # No CASCADE: an unreviewed dependency must refuse contraction.
    op.execute("DROP TABLE leases")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout='5s'")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM legacy_lease_archive
            WHERE record->>'lease_id' IS DISTINCT FROM lease_id
              OR record_sha256<>encode(sha256(convert_to(record::text,'UTF8')),'hex')) THEN
            RAISE EXCEPTION 'legacy lease archive digest mismatch; restore refused';
        END IF;
    END $$""")
    op.execute("""CREATE TABLE leases (
        lease_id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE, host_id TEXT NOT NULL,
        granted_at DOUBLE PRECISION NOT NULL, expires_at DOUBLE PRECISION NOT NULL,
        last_renewed DOUBLE PRECISION NOT NULL, duration_sec INTEGER DEFAULT 300,
        status TEXT DEFAULT 'active'
    )""")
    op.execute("CREATE INDEX idx_leases_status ON leases(status)")
    op.execute("CREATE INDEX idx_leases_active ON leases(job_id,host_id) WHERE status='active'")
    op.execute(
        "INSERT INTO leases SELECT (jsonb_populate_record(NULL::leases,record)).* FROM legacy_lease_archive"
    )
    op.execute("""DO $$ BEGIN
        IF (SELECT count(*) FROM leases)<>(SELECT count(*) FROM legacy_lease_archive)
           OR EXISTS (SELECT 1 FROM leases l JOIN legacy_lease_archive a USING(lease_id)
                      WHERE a.record IS DISTINCT FROM to_jsonb(l)) THEN
            RAISE EXCEPTION 'legacy lease restore verification failed';
        END IF;
    END $$""")
    op.execute("DROP TABLE legacy_lease_archive")
