"""Authoritative host admission and signed compatibility sessions.

Provider-controlled browsers, wizards, and workers may submit useful
compatibility evidence, but they cannot grant scheduler authority.  This
expand migration adds:

* a normalized admission state/version on ``hosts``;
* expiring, proof-of-possession compatibility sessions;
* immutable, classified evidence rows; and
* idempotent operator decisions that cite the evidence they relied on.

Existing admitted hosts are preserved through an explicit grandfathering
evidence/decision pair.  The host projection trigger is tightened so legacy
JSONB writers can no longer promote a host by writing ``payload.admitted``.

Revision ID: 082
Revises: 081
Create Date: 2026-07-30
"""

from alembic import op

revision = "082"
down_revision = "081"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute(
        """
        ALTER TABLE hosts
          ADD COLUMN IF NOT EXISTS admission_state TEXT NOT NULL DEFAULT 'pending',
          ADD COLUMN IF NOT EXISTS admission_version BIGINT NOT NULL DEFAULT 0,
          ADD COLUMN IF NOT EXISTS admitted_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS admission_decision_id UUID
        """
    )
    op.execute(
        """
        ALTER TABLE hosts
          ADD CONSTRAINT ck_hosts_admission_state
          CHECK (admission_state IN
                 ('pending', 'admitted', 'rejected', 'revoked')) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE hosts VALIDATE CONSTRAINT ck_hosts_admission_state"
    )
    op.execute(
        """
        ALTER TABLE hosts
          ADD CONSTRAINT ck_hosts_admission_version_non_negative
          CHECK (admission_version >= 0) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE hosts "
        "VALIDATE CONSTRAINT ck_hosts_admission_version_non_negative"
    )

    op.execute(
        """
        CREATE TABLE host_compatibility_sessions (
            session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
            requested_by TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            helper_public_key_spki BYTEA NOT NULL,
            helper_key_fingerprint TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            challenge_hash TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'created'
                CHECK (state IN ('created', 'consumed', 'expired', 'cancelled')),
            report_digest TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            expires_at TIMESTAMPTZ NOT NULL,
            consumed_at TIMESTAMPTZ,
            UNIQUE (host_id, idempotency_key),
            CHECK (length(helper_key_fingerprint) = 64),
            CHECK (length(token_hash) = 64),
            CHECK (length(challenge_hash) = 64),
            CHECK (report_digest IS NULL OR length(report_digest) = 64)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX host_compatibility_sessions_tenant_host
            ON host_compatibility_sessions (tenant_id, host_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX host_compatibility_sessions_expiry
            ON host_compatibility_sessions (expires_at)
         WHERE state = 'created'
        """
    )

    op.execute(
        """
        CREATE TABLE host_admission_evidence (
            evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
            session_id UUID
                REFERENCES host_compatibility_sessions(session_id)
                ON DELETE SET NULL,
            evidence_type TEXT NOT NULL
                CHECK (evidence_type IN (
                    'compatibility', 'hardware_verification', 'identity',
                    'runtime', 'network', 'storage', 'operator_review',
                    'legacy_grandfathering'
                )),
            source_type TEXT NOT NULL
                CHECK (source_type IN (
                    'provider_helper', 'provider_agent', 'trusted_verifier',
                    'operator', 'migration'
                )),
            trust_level TEXT NOT NULL
                CHECK (trust_level IN ('advisory', 'authoritative')),
            verdict TEXT NOT NULL
                CHECK (verdict IN ('pass', 'fail', 'inconclusive')),
            schema_version TEXT NOT NULL,
            evidence_digest TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            verifier_principal TEXT NOT NULL,
            summary JSONB NOT NULL DEFAULT '{}'::jsonb,
            observed_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            superseded_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (host_id, source_type, idempotency_key),
            CHECK (length(evidence_digest) = 64),
            CHECK (jsonb_typeof(summary) = 'object')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX host_admission_evidence_tenant_latest
            ON host_admission_evidence
               (tenant_id, host_id, evidence_type, observed_at DESC)
         WHERE superseded_at IS NULL
        """
    )
    op.execute(
        """
        CREATE INDEX host_admission_evidence_authoritative_latest
            ON host_admission_evidence
               (host_id, evidence_type, observed_at DESC, created_at DESC)
         WHERE trust_level = 'authoritative' AND superseded_at IS NULL
        """
    )

    op.execute(
        """
        CREATE TABLE host_admission_decisions (
            decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
            decision_version BIGINT NOT NULL CHECK (decision_version > 0),
            action TEXT NOT NULL
                CHECK (action IN ('admit', 'reject', 'revoke', 'grandfather')),
            previous_state TEXT NOT NULL,
            resulting_state TEXT NOT NULL
                CHECK (resulting_state IN
                       ('pending', 'admitted', 'rejected', 'revoked')),
            actor_principal TEXT NOT NULL,
            reason TEXT NOT NULL,
            evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (host_id, decision_version),
            UNIQUE (host_id, idempotency_key),
            CHECK (jsonb_typeof(evidence_ids) = 'array')
        )
        """
    )
    op.execute(
        """
        CREATE INDEX host_admission_decisions_tenant_host
            ON host_admission_decisions
               (tenant_id, host_id, decision_version DESC)
        """
    )

    # Preserve every host that the pre-082 control plane explicitly treated as
    # admitted.  Rows without an affirmative payload flag remain pending even
    # if an old status writer happened to call them active.
    op.execute(
        """
        UPDATE hosts
           SET admission_state = CASE
                   WHEN lower(COALESCE(payload->>'admitted', 'false'))
                        IN ('true', '1', 'yes')
                       THEN 'admitted'
                   ELSE 'pending'
               END,
               admission_version = CASE
                   WHEN lower(COALESCE(payload->>'admitted', 'false'))
                        IN ('true', '1', 'yes')
                       THEN 1
                   ELSE 0
               END,
               admitted_at = CASE
                   WHEN lower(COALESCE(payload->>'admitted', 'false'))
                        IN ('true', '1', 'yes')
                       THEN COALESCE(to_timestamp(NULLIF(registered_at, 0)),
                                     clock_timestamp())
                   ELSE NULL
               END
        """
    )

    op.execute(
        """
        INSERT INTO host_admission_evidence (
            tenant_id, host_id, evidence_type, source_type, trust_level,
            verdict, schema_version, evidence_digest, idempotency_key,
            verifier_principal, summary, observed_at, expires_at
        )
        SELECT COALESCE(NULLIF(tenant_id, ''), NULLIF(owner_id, ''), 'legacy'),
               host_id,
               'legacy_grandfathering',
               'migration',
               'authoritative',
               'pass',
               '1',
               md5('migration-082:legacy-grandfathering:' || host_id)
                   || md5(host_id || ':migration-082'),
               'migration-082',
               'migration:082',
               jsonb_build_object(
                   'basis', 'explicit pre-082 payload.admitted flag',
                   'legacy_status', status
               ),
               clock_timestamp(),
               'infinity'::timestamptz
          FROM hosts
         WHERE admission_state = 'admitted'
        ON CONFLICT (host_id, source_type, idempotency_key) DO NOTHING
        """
    )

    op.execute(
        """
        INSERT INTO host_admission_decisions (
            tenant_id, host_id, decision_version, action, previous_state,
            resulting_state, actor_principal, reason, evidence_ids,
            idempotency_key
        )
        SELECT COALESCE(NULLIF(h.tenant_id, ''), NULLIF(h.owner_id, ''), 'legacy'),
               h.host_id,
               1,
               'grandfather',
               'legacy',
               'admitted',
               'migration:082',
               'Preserved explicit pre-082 admission during expand migration',
               jsonb_build_array(e.evidence_id::text),
               'migration-082'
          FROM hosts h
          JOIN host_admission_evidence e
            ON e.host_id = h.host_id
           AND e.source_type = 'migration'
           AND e.idempotency_key = 'migration-082'
         WHERE h.admission_state = 'admitted'
        ON CONFLICT (host_id, idempotency_key) DO NOTHING
        """
    )

    op.execute(
        """
        UPDATE hosts h
           SET admission_decision_id = d.decision_id
          FROM host_admission_decisions d
         WHERE d.host_id = h.host_id
           AND d.idempotency_key = 'migration-082'
        """
    )

    # Normalized admission state is now the authority.  A legacy JSONB writer
    # may update inventory/heartbeat fields, but payload.admitted is rewritten
    # from the normalized column and can no longer self-promote the host.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION control_plane_project_host()
        RETURNS trigger AS $$
        DECLARE
            base_payload jsonb;
        BEGIN
            base_payload := CASE
                WHEN jsonb_typeof(NEW.payload) = 'object' THEN NEW.payload
                ELSE '{}'::jsonb
            END;

            base_payload := jsonb_set(
                base_payload,
                '{admitted}',
                to_jsonb(NEW.admission_state = 'admitted'),
                true
            );
            base_payload := jsonb_set(
                base_payload,
                '{admission_state}',
                to_jsonb(NEW.admission_state),
                true
            );
            base_payload := jsonb_set(
                base_payload,
                '{admission_version}',
                to_jsonb(NEW.admission_version),
                true
            );
            IF NEW.admission_decision_id IS NOT NULL THEN
                base_payload := jsonb_set(
                    base_payload,
                    '{admission_decision_id}',
                    to_jsonb(NEW.admission_decision_id::text),
                    true
                );
            END IF;
            NEW.payload := base_payload;

            NEW.administrative_state := CASE
                WHEN NEW.status = 'disabled' THEN 'disabled'
                WHEN NEW.admission_state = 'admitted'
                     AND NEW.status = 'draining' THEN 'draining'
                WHEN NEW.admission_state = 'admitted' THEN 'admitted'
                ELSE 'pending'
            END;
            NEW.availability_state := CASE NEW.status
                WHEN 'active' THEN 'ready'
                WHEN 'dead' THEN 'not_ready'
                ELSE 'unknown'
            END;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )

    # Re-run the projection once so the JSONB compatibility view and indexed
    # administrative state exactly match the new authority.
    op.execute("UPDATE hosts SET payload = payload")

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hosts_authoritative_admission
            ON hosts (admission_state, availability_state, last_observed_at)
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute("DROP INDEX IF EXISTS idx_hosts_authoritative_admission")
    op.execute("DROP TABLE IF EXISTS host_admission_decisions")
    op.execute("DROP TABLE IF EXISTS host_admission_evidence")
    op.execute("DROP TABLE IF EXISTS host_compatibility_sessions")

    # Restore the migration-059 projection while the normalized columns still
    # exist, then remove the expand-only authority columns.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION control_plane_project_host()
        RETURNS trigger AS $$
        BEGIN
            NEW.administrative_state := CASE
                WHEN NEW.status = 'disabled' THEN 'disabled'
                WHEN NEW.status = 'draining'
                     OR COALESCE((NEW.payload->>'draining')::boolean, false)
                    THEN 'draining'
                WHEN COALESCE((NEW.payload->>'admitted')::boolean, false)
                    THEN 'admitted'
                ELSE 'pending'
            END;
            NEW.availability_state := CASE NEW.status
                WHEN 'active' THEN 'ready'
                WHEN 'dead' THEN 'not_ready'
                ELSE 'unknown'
            END;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        "ALTER TABLE hosts DROP CONSTRAINT IF EXISTS "
        "ck_hosts_admission_version_non_negative"
    )
    op.execute(
        "ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_admission_state"
    )
    op.execute(
        """
        ALTER TABLE hosts
          DROP COLUMN IF EXISTS admission_decision_id,
          DROP COLUMN IF EXISTS admitted_at,
          DROP COLUMN IF EXISTS admission_version,
          DROP COLUMN IF EXISTS admission_state
        """
    )
