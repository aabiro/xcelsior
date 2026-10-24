"""Connect mapping bootstrap belongs to migration identity, preserving rows."""

import ast
from pathlib import Path

import psycopg

from tests.test_from_empty_bootstrap import empty_db
from tests.test_migration_from_production_snapshot import _alembic

ROOT = Path(__file__).resolve().parents[1]


def test_route_import_has_no_schema_creation_or_connection_side_effect():
    source = (ROOT / "api/routes/stripe_connect_v2.py").read_text()
    tree = ast.parse(source)
    assert "CREATE TABLE" not in source
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_ensure_connect_tables"
        for node in tree.body
    )


def test_connect_migration_preserves_existing_mappings_and_downgrade(empty_db):
    _alembic(empty_db, "upgrade", "138")
    _alembic(empty_db, "upgrade", "139")
    with psycopg.connect(empty_db) as c:
        c.execute(
            "INSERT INTO connect_accounts(display_name,contact_email,stripe_account_id) VALUES('Synthetic mapping','nobody@example.invalid','acct_qualified')"
        )
        c.execute(
            "INSERT INTO connect_products(stripe_product_id,stripe_price_id,name,price_cents,account_id) VALUES('prod_qualified','price_qualified','Synthetic product',123,'acct_qualified')"
        )
        expected = [
            c.execute("SELECT to_jsonb(t) FROM " + name + " t").fetchall()
            for name in ["connect_accounts", "connect_products"]
        ]
    _alembic(empty_db, "downgrade", "138")
    _alembic(empty_db, "upgrade", "139")
    with psycopg.connect(empty_db) as c:
        assert [
            c.execute("SELECT to_jsonb(t) FROM " + name + " t").fetchall()
            for name in ["connect_accounts", "connect_products"]
        ] == expected
