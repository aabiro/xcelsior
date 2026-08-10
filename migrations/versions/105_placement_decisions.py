"""The placement record Gate P5 clause 3 asks for. C1 of docs/placement-preference-plan.md.

The clause: *"preference is honoured in the audit trail: the chosen host's
reputation and SLA at time of placement are recorded."*

**Copied, not referenced.** Storing a host id and re-reading its score later
answers "what is this host's reputation *now*" — a different question, and a
useless one when reconstructing an incident weeks afterwards. Verification makes
that concrete: it is revocable, so a host verified at placement can be
deverified the next day and the stored id would then testify to the opposite of
what was true.

**Refusals are recorded too, and that is the more useful half.** The clause is
about honouring a preference; a preference that refused is honoured by the
refusal. Recording only successes would leave the audit trail with no evidence
of the behaviour this whole gate exists to produce, and no way to answer "why
did nothing launch last Tuesday".

**WORM.** "Append-only" enforced by a comment is a convention; enforced by a
trigger it is a property. Copied evidence is worth exactly what it costs to
rewrite. Precedent: `audit_checkpoints` (075), `audit_events_v2` (072).

**Money is integer micros** (rule: no binary floats for money). Prices arrive as
cents per hour and are stored as micros of CAD per hour, so the premium is
recomputed from two exact integers rather than stored as a rounded percentage
that cannot be checked against anything.

**Expand-only** (rule 5): one new table, nothing altered.
"""

from alembic import op

revision = "105"
down_revision = "104"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS placement_decisions (
            decision_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            -- Ledger rule 9 / companion §4.4.10: a tenant-owned table carries
            -- its own tenant_id and leads its index with it.
            tenant_id            TEXT NOT NULL,
            -- Null while a placement is being simulated: the preference is
            -- evaluated before a job exists, and a refusal means no job is ever
            -- created. Recording only decisions that produced a job would drop
            -- every refusal, which is the half this table is for.
            job_id               TEXT,
            host_id              TEXT,
            outcome              TEXT NOT NULL,
            refusal_code         TEXT,
            refusal_detail       TEXT,
            -- What the user asked for, verbatim, so a decision can be re-read
            -- against the preference that produced it rather than against
            -- whatever the defaults became later.
            asked                JSONB NOT NULL DEFAULT '{}'::jsonb,
            -- `preference.placement_evidence()` for the chosen host: reputation,
            -- uptime, verification state and its timestamps.
            evidence             JSONB NOT NULL DEFAULT '{}'::jsonb,
            -- What the choice was made among. A refusal is only interpretable
            -- against the field it refused over.
            candidate_count      INTEGER NOT NULL DEFAULT 0,
            candidates           JSONB NOT NULL DEFAULT '[]'::jsonb,
            baseline_price_micros BIGINT,
            chosen_price_micros   BIGINT,
            decided_at           TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            CONSTRAINT ck_placement_outcome CHECK (outcome IN ('placed', 'refused')),
            -- A placement names a host; a refusal names a code. Without this the
            -- state "placed on nothing" and "refused for no reason" are both
            -- representable, and the trail has a hole where a reader would look.
            CONSTRAINT ck_placement_shape CHECK (
                (outcome = 'placed'  AND host_id IS NOT NULL AND refusal_code IS NULL)
             OR (outcome = 'refused' AND refusal_code IS NOT NULL)
            ),
            -- Prices are never negative, and a zero baseline is what silently
            -- turns every premium into 0%.
            CONSTRAINT ck_placement_prices CHECK (
                (baseline_price_micros IS NULL OR baseline_price_micros > 0)
            AND (chosen_price_micros  IS NULL OR chosen_price_micros  > 0)
            )
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_decisions_tenant "
        "ON placement_decisions (tenant_id, decided_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_decisions_job "
        "ON placement_decisions (job_id) WHERE job_id IS NOT NULL"
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION placement_decisions_immutable() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'placement_decisions is append-only (WORM); % is not permitted', TG_OP
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_placement_decisions_immutable
            BEFORE UPDATE OR DELETE ON placement_decisions
            FOR EACH ROW EXECUTE FUNCTION placement_decisions_immutable()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_placement_decisions_immutable ON placement_decisions"
    )
    op.execute("DROP FUNCTION IF EXISTS placement_decisions_immutable()")
    op.execute("DROP INDEX IF EXISTS idx_placement_decisions_job")
    op.execute("DROP INDEX IF EXISTS idx_placement_decisions_tenant")
    op.execute("DROP TABLE IF EXISTS placement_decisions")
