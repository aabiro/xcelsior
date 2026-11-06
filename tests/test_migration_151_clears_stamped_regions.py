"""Migration 151 clears region labels that defaults stamped and keeps chosen ones."""

import importlib.util
import os
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import NullPool, create_engine, text

MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "api"
    / "migrations"
    / "versions"
    / "151_clear_stamped_regions.py"
)


def _migration():
    spec = importlib.util.spec_from_file_location("migration_151", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine():
    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN", "")
    if not dsn:
        pytest.skip("no PostgreSQL DSN in the environment")
    url = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    engine = create_engine(
        url.replace("postgresql://", "postgresql+psycopg://", 1), poolclass=NullPool
    )
    yield engine
    engine.dispose()


def test_stamped_labels_clear_and_chosen_regions_stay(engine):
    tag = uuid.uuid4().hex[:8]
    volumes = {
        f"vol-m151-east-{tag}": ("ca-east", ""),
        f"vol-m151-central-{tag}": ("CA-Central", ""),
        f"vol-m151-on-{tag}": ("ca-on", ""),
        f"vol-m151-any-{tag}": ("", ""),
    }
    with engine.begin() as conn:
        for volume_id, (region, province) in volumes.items():
            conn.execute(
                text(
                    "INSERT INTO volumes(volume_id, owner_id, name, size_gb, region, province,"
                    " status, created_at) VALUES (:id, :owner, :name, 1, :region, :province,"
                    " 'available', :now)"
                ),
                {
                    "id": volume_id,
                    "owner": f"m151-{tag}",
                    "name": volume_id,
                    "region": region,
                    "province": province,
                    "now": time.time(),
                },
            )
    try:
        unit = _migration()._clear("volumes")
        for _ in range(2):  # idempotent: a resumed migration re-runs every unit
            with engine.begin() as conn:
                unit(conn)
        with engine.connect() as conn:
            stored = dict(
                conn.execute(
                    text("SELECT volume_id, region FROM volumes WHERE owner_id = :owner"),
                    {"owner": f"m151-{tag}"},
                ).fetchall()
            )
        assert stored == {
            f"vol-m151-east-{tag}": "",
            f"vol-m151-central-{tag}": "",
            f"vol-m151-on-{tag}": "ca-on",
            f"vol-m151-any-{tag}": "",
        }
    finally:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM volumes WHERE owner_id = :owner"), {"owner": f"m151-{tag}"}
            )


def test_endpoint_tables_no_longer_default_to_a_region(engine):
    with engine.connect() as conn:
        defaults = dict(
            conn.execute(
                text(
                    "SELECT table_name, column_default FROM information_schema.columns"
                    " WHERE table_schema = 'public' AND column_name = 'region'"
                    " AND table_name IN ('serverless_endpoints', 'inference_endpoints')"
                )
            ).fetchall()
        )
    assert defaults == {
        "serverless_endpoints": "''::text",
        "inference_endpoints": "''::text",
    }
