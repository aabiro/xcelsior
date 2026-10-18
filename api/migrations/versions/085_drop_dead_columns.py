"""Drop columns that no code reads or writes.

Selection was evidence-based, not by inspection. Every column of every table in
``public`` (1648 of them) was matched against a word-boundary search over the
whole tree — 8738 Python, SQL, TypeScript and TSX files, excluding vendored
directories and ``migrations/versions`` (a migration mentioning a column is not
a consumer of it). 25 columns matched nowhere.

Ten of those 25 are deliberately **kept**, because "unreferenced by name" is not
the same as "unused":

- ``data_disclosures.disclosure_id``, ``serverless_kv_cache_samples.sample_id``
  and ``host_agent_tokens.rotated_from`` carry key constraints. They are
  unreferenced by name only because the rows are written generically; dropping
  a primary key or a rotation-lineage foreign key would break the table.
- ``wallets.auto_topup_amount_micros`` / ``auto_topup_threshold_micros`` are
  populated on 147 rows. They are the integer half of an unfinished money
  precision migration whose ``_cad`` float counterparts are still what the code
  reads. The right resolution is to finish that cutover — float is the wrong
  representation for money — not to delete the correct columns.
- ``gpu_offers.verified_level`` / ``dlperf_score`` hold data.
- ``provider_accounts.total_paid_out_cad``, ``jobs_hosted`` and
  ``last_payout_at`` are provider financial history. They are empty in
  development, which says nothing about production, and payout records are the
  kind of thing that has to be retained deliberately rather than dropped
  because no code path happens to read them today.

That leaves the 15 columns below: no key constraint, no reader or writer
anywhere in the tree, and no financial or audit meaning.

``downgrade`` restores the columns with their original types and defaults. It
cannot restore data — a dropped column's contents are gone. Anything that
matters must be captured before this runs.
"""

from alembic import op

revision = "085"
down_revision = "084"
branch_labels = None
depends_on = None


DEAD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("action_plans", "failure_detail"),
    ("audit_checkpoints", "object_uri"),
    ("data_disclosures", "entities_affected"),
    ("data_disclosures", "was_mandatory"),
    ("host_gpu_devices", "max_shares"),
    ("host_gpu_devices", "topology_group"),
    ("host_gpu_devices", "condition_details"),
    ("mcp_client_policies", "allowed_tool_classes"),
    ("mcp_client_policies", "max_runtime_sec"),
    ("node_versions", "runc_version"),
    ("node_versions", "nvidia_toolkit_version"),
    ("node_versions", "docker_version"),
    ("node_versions", "last_benchmark_at"),
    ("serverless_batches", "results_json"),
    ("telemetry_snapshots", "fan_speed_pct"),
)


def upgrade() -> None:
    # A dropped column takes an ACCESS EXCLUSIVE lock. It is a catalog-only
    # change so it is fast, but bound the wait rather than letting a deploy
    # queue behind a long-running reader.
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    for table, column in DEAD_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    # Types and defaults reproduced from the live catalog before the drop.
    # Restores structure only; the data is not recoverable from here.
    op.execute("ALTER TABLE action_plans ADD COLUMN IF NOT EXISTS failure_detail TEXT")
    op.execute("ALTER TABLE audit_checkpoints ADD COLUMN IF NOT EXISTS object_uri TEXT")
    op.execute(
        "ALTER TABLE data_disclosures ADD COLUMN IF NOT EXISTS entities_affected INTEGER DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE data_disclosures ADD COLUMN IF NOT EXISTS was_mandatory INTEGER DEFAULT 0"
    )
    op.execute("ALTER TABLE host_gpu_devices ADD COLUMN IF NOT EXISTS max_shares INTEGER DEFAULT 1")
    op.execute("ALTER TABLE host_gpu_devices ADD COLUMN IF NOT EXISTS topology_group TEXT")
    op.execute(
        "ALTER TABLE host_gpu_devices ADD COLUMN IF NOT EXISTS condition_details JSONB "
        "DEFAULT '{}'::jsonb"
    )
    op.execute(
        "ALTER TABLE mcp_client_policies ADD COLUMN IF NOT EXISTS allowed_tool_classes TEXT[] "
        "DEFAULT '{}'::text[]"
    )
    op.execute("ALTER TABLE mcp_client_policies ADD COLUMN IF NOT EXISTS max_runtime_sec INTEGER")
    op.execute("ALTER TABLE node_versions ADD COLUMN IF NOT EXISTS runc_version TEXT")
    op.execute("ALTER TABLE node_versions ADD COLUMN IF NOT EXISTS nvidia_toolkit_version TEXT")
    op.execute("ALTER TABLE node_versions ADD COLUMN IF NOT EXISTS docker_version TEXT")
    op.execute(
        "ALTER TABLE node_versions ADD COLUMN IF NOT EXISTS last_benchmark_at DOUBLE PRECISION"
    )
    op.execute(
        "ALTER TABLE serverless_batches ADD COLUMN IF NOT EXISTS results_json JSONB "
        "DEFAULT '[]'::jsonb"
    )
    op.execute(
        "ALTER TABLE telemetry_snapshots ADD COLUMN IF NOT EXISTS fan_speed_pct DOUBLE PRECISION "
        "DEFAULT 0"
    )
