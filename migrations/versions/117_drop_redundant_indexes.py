"""Drop 17 indexes that another index already covers.

A single-column btree index is redundant when a second index has that column as
its *leading* column and carries the same predicate: Postgres will use the
second one for every lookup the first could serve. The duplicate still costs a
B-tree write on every INSERT and UPDATE of the table, and the tables here are
the write-heavy ones — `usage_meters`, `wallet_transactions`, `billing_cycles`,
`payout_ledger`, `sessions`, `job`-adjacent history.

Five of the seventeen are not even leading-column cases but exact duplicates of
a UNIQUE constraint's index, on the same column with the same predicate:

    uq_usage_meters_one_per_attempt  UNIQUE (attempt_id) WHERE attempt_id IS NOT NULL
    idx_usage_meters_attempt                (attempt_id) WHERE attempt_id IS NOT NULL

A unique index is an ordinary index as well, so the second one has never served
a query the first could not.

Each pair below was checked against `pg_indexes` in a migrated database: same
access method (all btree), same leading column, identical predicate. Nothing
here changes what can be looked up, only how many copies are maintained while
doing it.

`IF EXISTS` throughout because these were created across many migrations and
some databases predate one or two of them.
"""

from alembic import op

revision = "117"
down_revision = "116"
branch_labels = None
depends_on = None

#: (index to drop, the index that already covers it, its definition)
REDUNDANT = [
    ("idx_benchmarks_host", "idx_benchmarks_host_run", "benchmarks", "host_id"),
    ("idx_billing_cycles_customer", "idx_billing_cycles_customer_created", "billing_cycles", "customer_id"),
    ("idx_casl_consent_user", "casl_consent_user_id_purpose_key", "casl_consent", "user_id"),
    ("idx_crypto_deposits_customer", "idx_crypto_deposits_customer_created", "crypto_deposits", "customer_id"),
    ("idx_inference_ep_owner", "idx_inference_endpoints_owner_created", "inference_endpoints", "owner_id"),
    ("idx_invoices_customer", "idx_invoices_customer_created", "invoices", "customer_id"),
    ("idx_leases_job", "leases_job_id_key", "leases", "job_id"),
    ("idx_ln_deposits_label", "ln_deposits_label_key", "ln_deposits", "label"),
    ("idx_ln_deposits_status", "idx_ln_deposits_status_created", "ln_deposits", "status"),
    ("idx_oauth_refresh_tokens_session", "oauth_refresh_tokens_session_token_key", "oauth_refresh_tokens", "session_token"),
    ("idx_payouts_provider", "idx_payout_ledger_provider_created", "payout_ledger", "provider_id"),
    ("idx_rep_events_entity", "idx_reputation_events_entity_created", "reputation_events", "entity_id"),
    ("idx_sessions_email", "idx_sessions_email_type", "sessions", "email"),
    ("idx_ssh_keys_email", "idx_user_ssh_keys_email_created", "user_ssh_keys", "email"),
    ("idx_usage_meters_attempt", "uq_usage_meters_one_per_attempt", "usage_meters", "attempt_id"),
    ("idx_users_user_id", "users_user_id_key", "users", "user_id"),
    ("idx_wallet_tx_customer", "idx_wallet_tx_customer_created", "wallet_transactions", "customer_id"),
]

#: The one partial index among them; recreated with its predicate on downgrade.
PARTIAL_PREDICATE = {"idx_usage_meters_attempt": "attempt_id IS NOT NULL"}


def upgrade() -> None:
    for name, _covered_by, _table, _column in REDUNDANT:
        op.execute(f"DROP INDEX IF EXISTS {name}")


def downgrade() -> None:
    for name, _covered_by, table, column in REDUNDANT:
        predicate = PARTIAL_PREDICATE.get(name)
        where = f" WHERE {predicate}" if predicate else ""
        op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column}){where}")
