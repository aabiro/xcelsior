"""Recorded Lightning projections survive a real ordered contract migration."""

from importlib import util
from pathlib import Path

from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


def _migration():
    path = (
        Path(__file__).resolve().parents[1]
        / "api/migrations/versions/141_archive_lightning_compatibility_columns.py"
    )
    spec = util.spec_from_file_location("lightning_contract", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(dsn):
    _alembic(dsn, "upgrade", "140")
    with _connect(dsn) as conn:
        conn.execute(
            """INSERT INTO ln_deposits
            (deposit_id,customer_id,tenant_id,label,bolt11,payment_hash,
             amount_msat,amount_sats,amount_btc,amount_micros,amount_cad_minor,
             btc_cad_rate,btc_cad_rate_exact,status,created_at,expires_at,
             created_at_ts,expires_at_ts)
            VALUES('contract','customer','tenant','contract','bolt','hash',
                1000000,1000,0.000009999999,25000000,2500,123456.12345678,
                123456.12345678,'pending',1700000000,1700000600,
                to_timestamp(1700000000),to_timestamp(1700000600))"""
        )
        canonical = conn.execute(
            """SELECT amount_micros,amount_cad_minor,
            btc_cad_rate_exact,created_at_ts,expires_at_ts,paid_at_ts,credited_at_ts,
            status,wallet_ledger_entry_id FROM ln_deposits WHERE deposit_id='contract'"""
        ).fetchone()
        original = conn.execute(
            "SELECT " + _migration().RECORD + " FROM ln_deposits d WHERE deposit_id='contract'"
        ).fetchone()[0]
    return canonical, original


def test_archive_preserves_exact_retired_values_and_leaves_native_money_unchanged(empty_db):
    canonical, original = _seed(empty_db)
    _alembic(empty_db, "upgrade", "141")
    with _connect(empty_db) as conn:
        assert (
            conn.execute(
                """SELECT amount_micros,amount_cad_minor,
            btc_cad_rate_exact,created_at_ts,expires_at_ts,paid_at_ts,credited_at_ts,
            status,wallet_ledger_entry_id FROM ln_deposits WHERE deposit_id='contract'"""
            ).fetchone()
            == canonical
        )
        assert (
            conn.execute(
                """SELECT record,tenant_id,tenant_attributed,
            record_sha256=encode(sha256(convert_to(record::text,'UTF8')),'hex')
            FROM lightning_retired_projection_archive WHERE deposit_id='contract'"""
            ).fetchone()
            == (original, "tenant", True, True)
        )
        assert (
            conn.execute(
                """SELECT count(*) FROM information_schema.columns
            WHERE table_schema='public' AND table_name='ln_deposits'
            AND column_name=ANY(%s)""",
                (list(_migration().COLUMNS),),
            ).fetchone()
            == (0,)
        )
    _alembic(empty_db, "downgrade", "140")
    _alembic(empty_db, "upgrade", "141")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT record FROM lightning_retired_projection_archive WHERE deposit_id='contract'"
        ).fetchone() == (original,)


def test_conflicting_preexisting_archive_refuses_before_any_column_is_dropped(empty_db):
    _, original = _seed(empty_db)
    _alembic(empty_db, "upgrade", "141")
    _alembic(empty_db, "downgrade", "140")
    migration = _migration()
    with _connect(empty_db) as conn:
        for column in migration.COLUMNS:
            conn.execute(f"ALTER TABLE ln_deposits ADD COLUMN {column} DOUBLE PRECISION")
            conn.execute(
                f"UPDATE ln_deposits SET {column}=%s WHERE deposit_id='contract'",
                (original[column],),
            )
        conn.execute("UPDATE ln_deposits SET btc_cad_rate=1 WHERE deposit_id='contract'")
    result = _alembic(empty_db, "upgrade", "141", check=False)
    assert result.returncode != 0 and "archive mismatch" in result.stderr
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT btc_cad_rate FROM ln_deposits WHERE deposit_id='contract'"
        ).fetchone() == (1.0,)
        assert conn.execute(
            "SELECT record FROM lightning_retired_projection_archive WHERE deposit_id='contract'"
        ).fetchone() == (original,)
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("140",)
