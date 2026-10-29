"""Alembic environment configuration for Xcelsior migrations."""

import os
import sys
from logging.config import fileConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Auto-load .env from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from alembic import context

config = context.config

# Override sqlalchemy.url from environment if available (match db.resolve_postgres_dsn)
dsn = (
    os.environ.get("XCELSIOR_POSTGRES_DSN")
    or os.environ.get("XCELSIOR_PG_DSN")
    or os.environ.get("DATABASE_URL")
    or config.get_main_option("sqlalchemy.url")
)
if dsn:
    # Ensure we use psycopg3 driver (postgresql+psycopg://), not psycopg2
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    config.set_main_option("sqlalchemy.url", dsn)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline():
    """Run migrations in 'offline' mode — generate SQL without DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# A lock request queues ahead of every later request on the same table, so an
# `ALTER TABLE` that waits behind one long-running read stalls all traffic to
# that table for as long as it waits. Deploys are blue-green — the live API and
# workers keep serving while this runs — so waiting indefinitely is an outage
# and failing fast is not. `migrations/README.md` rule 5 requires this be set.
LOCK_TIMEOUT = os.environ.get("XCELSIOR_MIGRATION_LOCK_TIMEOUT", "5s")


def run_migrations_online():
    """Run migrations in 'online' mode — connect to DB and apply."""
    from sqlalchemy import create_engine, text

    from migrations.lock_safe import checked_timeout

    engine = create_engine(config.get_main_option("sqlalchemy.url"))

    with engine.connect() as connection:
        # `SET` takes no bind parameters; `checked_timeout` is what makes the
        # interpolation of an environment variable safe here.
        connection.execute(text(f"SET lock_timeout = '{checked_timeout(LOCK_TIMEOUT)}'"))
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=None,
            # One transaction per migration, not one for the whole upgrade.
            #
            # With a single enclosing transaction, a 16-migration deploy holds
            # every lock every migration took until the last one commits, and
            # any failure discards the lot — so each retry replays all of them
            # and re-widens the window. That is how migration 095 deadlocked
            # against live traffic on the 2026-08-04 deploy. Per-migration
            # transactions keep each migration atomic, release its locks at its
            # own commit, and make a failed deploy resumable from where it
            # stopped. `migrations/lock_safe.py` additionally *requires* this:
            # it opens a second connection, which would block against locks
            # still held by an earlier migration in a shared transaction.
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
