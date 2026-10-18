"""Drop 17 indexes that another index already covers.

A single-column btree index is redundant when a second index has that column as
its *leading* column and carries the same predicate: Postgres will use the
second for every lookup the first could serve. The duplicate still costs a
B-tree write on every INSERT and UPDATE, and the tables here are the write-heavy
ones — `usage_meters`, `wallet_transactions`, `billing_cycles`, `payout_ledger`,
`sessions`.

Five are not leading-column cases but exact duplicates of a UNIQUE constraint's
index, same column and same predicate:

    uq_usage_meters_one_per_attempt  UNIQUE (attempt_id) WHERE attempt_id IS NOT NULL
    idx_usage_meters_attempt                (attempt_id) WHERE attempt_id IS NOT NULL

A unique index is an ordinary index as well, so the second has never served a
query the first could not.

Every pair was checked against `pg_indexes` in a migrated database: same access
method (all btree), same leading column, identical predicate. Nothing here
changes what can be looked up, only how many copies are maintained.

The statements are written out rather than generated from a table. A loop over
`op.execute(f"... {name} ...")` reads as dynamic SQL to
`test_sql_injection_guard`, and the honest answer is not to allowlist the file
but to not interpolate: DDL in a migration is fixed text, so it is written as
fixed text.

`IF EXISTS` / `IF NOT EXISTS` throughout because these were created across many
migrations and some databases predate one or two of them.
"""

from alembic import op

revision = "117"
down_revision = "116"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # benchmarks.host_id — covered by idx_benchmarks_host_run
    op.execute("DROP INDEX IF EXISTS idx_benchmarks_host")
    # billing_cycles.customer_id — covered by idx_billing_cycles_customer_created
    op.execute("DROP INDEX IF EXISTS idx_billing_cycles_customer")
    # casl_consent.user_id — covered by casl_consent_user_id_purpose_key
    op.execute("DROP INDEX IF EXISTS idx_casl_consent_user")
    # crypto_deposits.customer_id — covered by idx_crypto_deposits_customer_created
    op.execute("DROP INDEX IF EXISTS idx_crypto_deposits_customer")
    # inference_endpoints.owner_id — covered by idx_inference_endpoints_owner_created
    op.execute("DROP INDEX IF EXISTS idx_inference_ep_owner")
    # invoices.customer_id — covered by idx_invoices_customer_created
    op.execute("DROP INDEX IF EXISTS idx_invoices_customer")
    # leases.job_id — covered by leases_job_id_key
    op.execute("DROP INDEX IF EXISTS idx_leases_job")
    # ln_deposits.label — covered by ln_deposits_label_key
    op.execute("DROP INDEX IF EXISTS idx_ln_deposits_label")
    # ln_deposits.status — covered by idx_ln_deposits_status_created
    op.execute("DROP INDEX IF EXISTS idx_ln_deposits_status")
    # oauth_refresh_tokens.session_token — covered by oauth_refresh_tokens_session_token_key
    op.execute("DROP INDEX IF EXISTS idx_oauth_refresh_tokens_session")
    # payout_ledger.provider_id — covered by idx_payout_ledger_provider_created
    op.execute("DROP INDEX IF EXISTS idx_payouts_provider")
    # reputation_events.entity_id — covered by idx_reputation_events_entity_created
    op.execute("DROP INDEX IF EXISTS idx_rep_events_entity")
    # sessions.email — covered by idx_sessions_email_type
    op.execute("DROP INDEX IF EXISTS idx_sessions_email")
    # user_ssh_keys.email — covered by idx_user_ssh_keys_email_created
    op.execute("DROP INDEX IF EXISTS idx_ssh_keys_email")
    # usage_meters.attempt_id — covered by uq_usage_meters_one_per_attempt
    op.execute("DROP INDEX IF EXISTS idx_usage_meters_attempt")
    # users.user_id — covered by users_user_id_key
    op.execute("DROP INDEX IF EXISTS idx_users_user_id")
    # wallet_transactions.customer_id — covered by idx_wallet_tx_customer_created
    op.execute("DROP INDEX IF EXISTS idx_wallet_tx_customer")


def downgrade() -> None:
    op.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_host ON benchmarks (host_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_cycles_customer ON billing_cycles (customer_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_casl_consent_user ON casl_consent (user_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_crypto_deposits_customer ON crypto_deposits (customer_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_inference_ep_owner ON inference_endpoints (owner_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices (customer_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_leases_job ON leases (job_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ln_deposits_label ON ln_deposits (label)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ln_deposits_status ON ln_deposits (status)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_session ON oauth_refresh_tokens (session_token)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_payouts_provider ON payout_ledger (provider_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rep_events_entity ON reputation_events (entity_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_email ON sessions (email)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ssh_keys_email ON user_ssh_keys (email)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_meters_attempt ON usage_meters (attempt_id)"
        " WHERE attempt_id IS NOT NULL"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_user_id ON users (user_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wallet_tx_customer ON wallet_transactions (customer_id)"
    )
