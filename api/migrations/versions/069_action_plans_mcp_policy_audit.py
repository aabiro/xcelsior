"""Action plans, MCP client policy, and MCP tool audit (Track B B2.1).

Blueprint §13.5 (renumbered to the live head per migrations/README.md — the
document's `058` is long spent) and §14: the durable spine of the unified
launch service. Every launch surface — MCP, dashboard, REST — creates the
*same* action plan, quotes against it, gets it approved, and executes it
exactly once.

Three tables:

- ``action_plans`` — the §9.5 state machine
  (``quoted → awaiting_approval → approved → executing →
  succeeded | failed_retryable | failed_terminal``, plus ``revoked`` and
  ``expired``) with the canonical argument hash that binds what was quoted
  to what executes, the price estimate and tolerance that gate a
  re-quote, and the idempotent response so a repeated execute returns the
  original job rather than launching a second.
- ``mcp_client_policies`` — per-client spend and capability limits, the
  thing that lets a plan self-approve inside standing policy (§14.2)
  instead of always bouncing a human.
- ``mcp_tool_audit`` — one redacted record per tool call (§17.10), in the
  audit domain.

Money is integer **micro-CAD**, matching the wallet ledger (migration
068): the plan's estimate becomes a wallet hold, and the two must be the
same unit or the tolerance comparison is a float round-trip.

Expand-only; nothing reads these until the launch service (B2.2+) lands.

Revision ID: 069
Revises: 068
Create Date: 2026-07-22
"""

from typing import Sequence, Union

from alembic import op

revision: str = "069"
down_revision: Union[str, None] = "068"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# §9.5 action-plan lifecycle. `quoted` is the preview; a plan only holds
# funds or a job at `executing`+.
PLAN_STATES = (
    "quoted",
    "awaiting_approval",
    "approved",
    "executing",
    "succeeded",
    "failed_retryable",
    "failed_terminal",
    "revoked",
    "expired",
)

# Who may approve a plan. `standing_policy` = auto-approved inside an
# mcp_client_policies budget; `human` = a dashboard approval; `none` = the
# action is low-risk enough to need no approval step.
APPROVAL_MODES = ("human", "standing_policy", "none")


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")

    states = ", ".join(f"'{s}'" for s in PLAN_STATES)
    modes = ", ".join(f"'{m}'" for m in APPROVAL_MODES)

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS action_plans (
            plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            action_type TEXT NOT NULL,
            -- Principal / tenancy: who owns the plan. A plan cannot be
            -- approved or executed by a different principal or tenant.
            principal_id TEXT NOT NULL,
            client_id TEXT,
            tenant_id TEXT NOT NULL,
            team_id TEXT,
            -- What was quoted. `canonical_args_hash` is the binding: execute
            -- recomputes it and refuses if the caller changed the request
            -- after approval (§8.5).
            canonical_args JSONB NOT NULL,
            canonical_args_hash TEXT NOT NULL,
            spec_hash TEXT,
            -- The quote: estimate in micro-CAD, the pricing snapshot it came
            -- from, and the tolerance beyond which execute must re-quote
            -- rather than silently charge the new price (§15.4).
            quote_id TEXT,
            pricing_version TEXT,
            estimate_micros BIGINT,
            currency CHAR(3) NOT NULL DEFAULT 'CAD',
            price_tolerance_bps INTEGER NOT NULL DEFAULT 500,
            required_scopes TEXT[] NOT NULL DEFAULT '{{}}',
            approval_mode TEXT NOT NULL CHECK (approval_mode IN ({modes})),
            status TEXT NOT NULL DEFAULT 'quoted' CHECK (status IN ({states})),
            -- Lifecycle timestamps.
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            expires_at TIMESTAMPTZ NOT NULL,
            approved_at TIMESTAMPTZ,
            consumed_at TIMESTAMPTZ,
            approved_by TEXT,
            approval_session_id TEXT,
            approval_method TEXT,
            -- Result: the resource this plan produced, and the idempotent
            -- response replayed on a repeated execute.
            resulting_resource_id TEXT,
            idempotent_response JSONB,
            -- Link to the *existing* holds and idempotency authorities
            -- (§13.5: reuse, do not create a second authority). The hold is a
            -- wallet_holds row; `idempotency_key` is the natural key the
            -- execute path writes into api_idempotency_keys so a repeated
            -- execute replays that row rather than launching twice.
            wallet_hold_id UUID
                REFERENCES wallet_holds (hold_id) ON DELETE SET NULL,
            idempotency_key TEXT,
            job_id TEXT,
            -- Why it ended badly, when it did.
            revoked_at TIMESTAMPTZ,
            revoked_reason TEXT,
            failure_code TEXT,
            failure_detail TEXT,
            version BIGINT NOT NULL DEFAULT 1,
            CHECK (estimate_micros IS NULL OR estimate_micros >= 0),
            CHECK (price_tolerance_bps >= 0),
            -- §9.5 state machine, enforced in the row itself so no service
            -- bug can persist an impossible plan. Each clause reads
            -- "status is not S, or the columns S requires are present":
            --   * approved and every later state carry an approval timestamp
            --     (approval_mode='none' still sets approved_at when it
            --     auto-approves — the timestamp records *when*, not *by whom*);
            --   * a consumed plan (executing onward) records consumed_at;
            --   * a succeeded launch produced a job and an idempotent response;
            --   * a failed launch records why;
            --   * a revoked plan records when.
            CONSTRAINT ck_action_plans_state_machine CHECK (
                (status <> 'approved' OR approved_at IS NOT NULL)
                AND (
                    status NOT IN (
                        'executing', 'succeeded',
                        'failed_retryable', 'failed_terminal'
                    )
                    OR (approved_at IS NOT NULL AND consumed_at IS NOT NULL)
                )
                AND (
                    status <> 'succeeded'
                    OR (job_id IS NOT NULL AND idempotent_response IS NOT NULL)
                )
                AND (
                    status NOT IN ('failed_retryable', 'failed_terminal')
                    OR failure_code IS NOT NULL
                )
                AND (status <> 'revoked' OR revoked_at IS NOT NULL)
            )
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_action_plans_principal "
        "ON action_plans (principal_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_action_plans_tenant_status "
        "ON action_plans (tenant_id, status)"
    )
    # Sweep target: plans past expiry that never reached a terminal state.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_action_plans_expiry "
        "ON action_plans (expires_at) "
        "WHERE status IN ('quoted', 'awaiting_approval', 'approved')"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_client_policies (
            policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            client_id TEXT NOT NULL,
            principal_id TEXT,
            tenant_id TEXT NOT NULL,
            allowed_tool_classes TEXT[] NOT NULL DEFAULT '{}',
            -- Spend ceilings, micro-CAD. NULL = no ceiling of that kind.
            per_action_max_micros BIGINT,
            hourly_spend_max_micros BIGINT,
            daily_spend_max_micros BIGINT,
            max_runtime_sec INTEGER,
            max_concurrency INTEGER,
            allowed_gpu_models TEXT[],
            allowed_regions TEXT[],
            allowed_security_modes TEXT[],
            -- Whether a launch inside these limits may self-approve (§14.2).
            auto_approve BOOLEAN NOT NULL DEFAULT false,
            version BIGINT NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (client_id, tenant_id),
            CHECK (per_action_max_micros IS NULL OR per_action_max_micros >= 0),
            CHECK (hourly_spend_max_micros IS NULL OR hourly_spend_max_micros >= 0),
            CHECK (daily_spend_max_micros IS NULL OR daily_spend_max_micros >= 0)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_tool_audit (
            audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            occurred_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            tool_name TEXT NOT NULL,
            tool_version TEXT,
            transport TEXT,
            client_id TEXT,
            principal_id TEXT,
            tenant_id TEXT,
            team_id TEXT,
            scopes_evaluated TEXT[],
            -- Redacted: a *hash* of the canonical arguments, never the
            -- arguments themselves (§17.10 — no secrets, tokens, env,
            -- or raw init scripts in the audit row).
            redacted_args_hash TEXT,
            action_plan_id UUID,
            idempotency_key TEXT,
            api_route TEXT,
            api_status INTEGER,
            problem_type TEXT,
            resource_id TEXT,
            latency_ms INTEGER,
            trace_id TEXT,
            approval_method TEXT,
            outcome TEXT
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_tool_audit_principal "
        "ON mcp_tool_audit (principal_id, occurred_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_mcp_tool_audit_plan "
        "ON mcp_tool_audit (action_plan_id) WHERE action_plan_id IS NOT NULL"
    )

    # OAuth machine-client workspace context (§13.5). A machine principal —
    # a system-managed client or one issued client_credentials tokens — must
    # carry a workspace so a launch it authorizes cannot lose tenant context
    # and land in the wrong (or no) workspace. `workspace_customer_id` is the
    # tenant identity for these rows (added in migration 061); this migration
    # does not add a second tenant column (DA§3, one authority per fact), only
    # the constraint that it be present. Expand-safe: added NOT VALID, then
    # VALIDATE only when no existing machine client violates it — an older
    # workspace-less machine client is left for a data backfill rather than
    # failing the migration.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                 WHERE conname = 'ck_oauth_client_workspace_context'
            ) THEN
                ALTER TABLE oauth_clients
                  ADD CONSTRAINT ck_oauth_client_workspace_context
                  CHECK (
                      (
                          is_system_managed = 0
                          AND NOT (grant_types @> '["client_credentials"]'::jsonb)
                      )
                      OR workspace_customer_id IS NOT NULL
                  ) NOT VALID;
            END IF;
        END $$
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM oauth_clients
                 WHERE workspace_customer_id IS NULL
                   AND (
                       is_system_managed = 1
                       OR grant_types @> '["client_credentials"]'::jsonb
                   )
            ) THEN
                ALTER TABLE oauth_clients
                  VALIDATE CONSTRAINT ck_oauth_client_workspace_context;
            END IF;
        END $$
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE oauth_clients "
        "DROP CONSTRAINT IF EXISTS ck_oauth_client_workspace_context"
    )
    op.execute("DROP TABLE IF EXISTS mcp_tool_audit")
    op.execute("DROP TABLE IF EXISTS mcp_client_policies")
    op.execute("DROP TABLE IF EXISTS action_plans")
