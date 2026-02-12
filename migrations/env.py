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

# Override sqlalchemy.url from environment if available
dsn = os.environ.get(
    "XCELSIOR_POSTGRES_DSN",
    os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url")),
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


def run_migrations_online():
    """Run migrations in 'online' mode — connect to DB and apply."""
    from sqlalchemy import create_engine

    engine = create_engine(config.get_main_option("sqlalchemy.url"))

    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()

    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
