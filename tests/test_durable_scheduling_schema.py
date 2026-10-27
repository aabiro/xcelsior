"""Runtime logins cannot rewrite operator policy or erase policy history."""

import os

import psycopg
import pytest
from psycopg.errors import InsufficientPrivilege

from control_plane import db_roles
from tests.test_db_service_roles import provisioned_roles  # noqa: F401


@pytest.mark.parametrize("service", db_roles.runtime_role_keys())
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE scheduling_team_policies SET weight=weight",
        "DELETE FROM scheduling_team_policies",
        "INSERT INTO scheduling_team_policies(team_key,version,created_by) VALUES('owner:forbidden',1,'runtime')",
    ],
)
def test_policy_revisions_are_operator_owned(provisioned_roles, service, statement):
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        conn.execute(
            psycopg.sql.SQL("SET ROLE {}").format(
                psycopg.sql.Identifier(db_roles.SERVICE_ROLES[service].name)
            )
        )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(statement)


def test_expansion_and_rollback_enable_no_preemption():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "api/migrations/versions/133_durable_scheduling_policy.py"
    spec = importlib.util.spec_from_file_location("durable_scheduling_expansion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        columns = dict(
            conn.execute(
                """SELECT column_name,column_default FROM information_schema.columns
            WHERE table_name='scheduling_team_policies'"""
            ).fetchall()
        )
        assert columns["preemption_enabled"] == "false"
        assert columns["force_after_deadline"] == "false"
        before = conn.execute("SELECT count(*) FROM preemption_plans").fetchone()
        module.downgrade()
        assert conn.execute("SELECT count(*) FROM preemption_plans").fetchone() == before
